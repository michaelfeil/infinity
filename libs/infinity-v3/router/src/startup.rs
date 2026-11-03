use anyhow::Result;
use std::future::Future;

/// Owned startup output whose worker threads must be stopped before a failed
/// peer can return from initialization.
pub(crate) trait Shutdown {
    fn shutdown(self) -> impl Future<Output = ()> + Send;
}

/// Run required frontend and checkpoint initialization together.
///
/// The backend is not polled until startup configuration validation finishes.
/// Both futures are joined even when one fails so blocking tokenizer, processor
/// and native backend initializers drain before startup returns. A successful
/// resource is shut down when its peer fails instead of being dropped while its
/// worker threads are still running. Dropping this future cannot cancel an
/// in-flight `spawn_blocking` task; its owned output is dropped by the blocking
/// worker when it completes.
pub(crate) async fn initialize_concurrently<V, F, B, T, U>(
    validation: V,
    frontend: F,
    backend: B,
) -> Result<(T, U)>
where
    V: Future<Output = Result<()>>,
    F: Future<Output = Result<T>>,
    B: Future<Output = Result<U>>,
    T: Shutdown,
    U: Shutdown,
{
    let backend = async move {
        validation.await?;
        backend.await
    };
    let (frontend, backend) = tokio::join!(frontend, backend);
    match (frontend, backend) {
        (Ok(frontend), Ok(backend)) => Ok((frontend, backend)),
        (Err(error), Ok(backend)) => {
            backend.shutdown().await;
            Err(error)
        }
        (Ok(frontend), Err(error)) => {
            frontend.shutdown().await;
            Err(error)
        }
        (Err(frontend_error), Err(backend_error)) => Err(frontend_error.context(format!(
            "Backend initialization also failed: {backend_error:#}"
        ))),
    }
}

/// Validate every multimodal limit that does not require reading checkpoint
/// assets. Keep these bounds aligned with the active core processor constructors
/// so invalid host or retrieval settings fail before checkpoint weights load.
pub(crate) fn validate_multimodal_config(
    config: &infinity_v3_core::multimodal::MultimodalConfig,
    qwen_max_input_length: Option<usize>,
) -> Result<()> {
    config.validate()?;
    if let Some(max_input_length) = qwen_max_input_length {
        anyhow::ensure!(
            (1..=u32::MAX as usize).contains(&max_input_length),
            "Qwen multimodal max_input_length must be between 1 and u32::MAX"
        );
    }
    Ok(())
}

pub(crate) fn parse_queue_token_limits(
    value: Option<&str>,
) -> Result<Option<infinity_v3_core::queue_budget::QueueTokenLimits>> {
    match value {
        Some(value) => Ok(Some(
            infinity_v3_core::queue_budget::QueueTokenLimits::from_thousands(value)
                .map_err(anyhow::Error::msg)?,
        )),
        None => Ok(None),
    }
}

/// Metadata workers and pooled-text acceleration are unnecessary when every
/// embedding input is prepared by a model-owned processor. Raw-text Qwen
/// variants still use the generic worker path for text and token IDs.
pub(crate) fn generic_workers_can_start_lazily(
    backend_model_type: &infinity_v3_backend::ModelType,
    mistral3_processor: bool,
    qwen_accepts_raw_text: Option<bool>,
) -> bool {
    match backend_model_type {
        infinity_v3_backend::ModelType::Decision => true,
        infinity_v3_backend::ModelType::Embedding(_) => {
            mistral3_processor || qwen_accepts_raw_text == Some(false)
        }
        infinity_v3_backend::ModelType::Classifier => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use infinity_v3_core::multimodal::MultimodalConfig;
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };
    use tokio::sync::oneshot;

    #[derive(Debug)]
    struct Resource(Arc<AtomicBool>);

    impl Shutdown for Resource {
        async fn shutdown(self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    impl Shutdown for () {
        async fn shutdown(self) {}
    }

    fn resource(cleaned: &Arc<AtomicBool>) -> Resource {
        Resource(cleaned.clone())
    }

    async fn assert_validation_blocks_backend(validation: impl Future<Output = Result<()>>) {
        let frontend_cleaned = Arc::new(AtomicBool::new(false));
        let backend_started = Arc::new(AtomicBool::new(false));
        let cleanup = frontend_cleaned.clone();
        let started = backend_started.clone();
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            initialize_concurrently(
                validation,
                async move { Ok::<_, anyhow::Error>(resource(&cleanup)) },
                async move {
                    started.store(true, Ordering::SeqCst);
                    Ok::<_, anyhow::Error>(())
                },
            ),
        )
        .await;
        result
            .expect("startup timed out")
            .expect_err("validation must fail startup");
        assert!(!backend_started.load(Ordering::SeqCst));
        assert!(frontend_cleaned.load(Ordering::SeqCst));
    }

    #[test]
    fn lazy_generic_workers_follow_embedding_routes() {
        use infinity_v3_backend::{ModelType, Pool};
        assert!(generic_workers_can_start_lazily(
            &ModelType::Decision,
            false,
            None
        ));
        assert!(generic_workers_can_start_lazily(
            &ModelType::Embedding(Pool::Mean),
            true,
            None
        ));
        assert!(generic_workers_can_start_lazily(
            &ModelType::Embedding(Pool::Mean),
            false,
            Some(false)
        ));
        assert!(!generic_workers_can_start_lazily(
            &ModelType::Embedding(Pool::Mean),
            false,
            Some(true)
        ));
        assert!(!generic_workers_can_start_lazily(
            &ModelType::Embedding(Pool::Mean),
            false,
            None
        ));
        assert!(!generic_workers_can_start_lazily(
            &ModelType::Classifier,
            true,
            None
        ));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn frontend_and_backend_are_polled_concurrently() {
        let barrier = Arc::new(tokio::sync::Barrier::new(2));
        let frontend_barrier = barrier.clone();
        let backend_barrier = barrier.clone();
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            initialize_concurrently(
                async { Ok(()) },
                async move {
                    frontend_barrier.wait().await;
                    Ok::<_, anyhow::Error>(())
                },
                async move {
                    backend_barrier.wait().await;
                    Ok::<_, anyhow::Error>(())
                },
            ),
        )
        .await;
        assert!(matches!(result, Ok(Ok(((), ())))));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn backend_discovery_failure_after_validation_drains_frontend() {
        let frontend_cleaned = Arc::new(AtomicBool::new(false));
        let validation_finished = Arc::new(AtomicBool::new(false));
        let backend_observed_validation = Arc::new(AtomicBool::new(false));
        let (frontend_ready_backend, wait_frontend_ready_backend) = oneshot::channel();
        let (frontend_ready_release, wait_frontend_ready_release) = oneshot::channel();
        let (backend_failed, wait_backend_failed) = oneshot::channel();
        let (release_frontend, wait_release) = oneshot::channel();
        let cleanup = frontend_cleaned.clone();
        let validation_flag = validation_finished.clone();
        let backend_validation_flag = validation_finished.clone();
        let observed = backend_observed_validation.clone();
        let release = tokio::spawn(async move {
            let _ = wait_frontend_ready_release.await;
            let _ = wait_backend_failed.await;
            let _ = release_frontend.send(());
        });
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            initialize_concurrently(
                async move {
                    validation_flag.store(true, Ordering::SeqCst);
                    Ok(())
                },
                async move {
                    let _ = frontend_ready_backend.send(());
                    let _ = frontend_ready_release.send(());
                    let _ = wait_release.await;
                    Ok::<_, anyhow::Error>(resource(&cleanup))
                },
                async move {
                    let _ = wait_frontend_ready_backend.await;
                    observed.store(
                        backend_validation_flag.load(Ordering::SeqCst),
                        Ordering::SeqCst,
                    );
                    let _ = backend_failed.send(());
                    Err::<Resource, _>(anyhow::anyhow!("device discovery failed"))
                },
            ),
        )
        .await;
        let _ = release.await;
        let error = result
            .expect("startup timed out")
            .expect_err("backend discovery failure must fail startup");
        assert!(format!("{error:#}").contains("device discovery failed"));
        assert!(backend_observed_validation.load(Ordering::SeqCst));
        assert!(frontend_cleaned.load(Ordering::SeqCst));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn backend_failure_drains_frontend_and_cleans_it() {
        let frontend_cleaned = Arc::new(AtomicBool::new(false));
        let (frontend_ready_backend, wait_frontend_ready_backend) = oneshot::channel();
        let (frontend_ready_release, wait_frontend_ready_release) = oneshot::channel();
        let (backend_failed, wait_backend_failed) = oneshot::channel();
        let (release_frontend, wait_release) = oneshot::channel();
        let cleanup = frontend_cleaned.clone();
        let release = tokio::spawn(async move {
            let _ = wait_frontend_ready_release.await;
            let _ = wait_backend_failed.await;
            let _ = release_frontend.send(());
        });
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            initialize_concurrently(
                async { Ok(()) },
                async move {
                    let _ = frontend_ready_backend.send(());
                    let _ = frontend_ready_release.send(());
                    let _ = wait_release.await;
                    Ok::<_, anyhow::Error>(resource(&cleanup))
                },
                async move {
                    let _ = wait_frontend_ready_backend.await;
                    let _ = backend_failed.send(());
                    Err::<Resource, _>(anyhow::anyhow!("backend failed"))
                },
            ),
        )
        .await;
        let _ = release.await;
        assert!(result.is_ok_and(|result| result.is_err()));
        assert!(frontend_cleaned.load(Ordering::SeqCst));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn frontend_failure_drains_backend_and_cleans_it() {
        let backend_cleaned = Arc::new(AtomicBool::new(false));
        let (frontend_failed, wait_frontend_failed) = oneshot::channel();
        let (release_backend, wait_release) = oneshot::channel();
        let backend_cleanup = backend_cleaned.clone();
        let release = tokio::spawn(async move {
            let _ = wait_frontend_failed.await;
            let _ = release_backend.send(());
        });
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            initialize_concurrently(
                async { Ok(()) },
                async move {
                    let _ = frontend_failed.send(());
                    Err::<Resource, _>(anyhow::anyhow!("frontend failed"))
                },
                async move {
                    let _ = wait_release.await;
                    Ok::<_, anyhow::Error>(resource(&backend_cleanup))
                },
            ),
        )
        .await;
        let _ = release.await;
        assert!(result.is_ok_and(|result| result.is_err()));
        assert!(backend_cleaned.load(Ordering::SeqCst));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn invalid_multimodal_limits_prevent_backend_start() {
        let mut config = MultimodalConfig::default();
        assert!(validate_multimodal_config(&config, Some(128)).is_ok());
        config.max_images = 0;
        assert_validation_blocks_backend(
            async move { validate_multimodal_config(&config, Some(128)) },
        )
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn invalid_image_host_pattern_prevents_backend_start() {
        let mut config = MultimodalConfig::default();
        config.allowed_image_hosts = vec!["*.com".into()];
        assert_validation_blocks_backend(async move { validate_multimodal_config(&config, None) })
            .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn invalid_queue_token_limits_prevent_backend_start() {
        assert_validation_blocks_backend(async {
            parse_queue_token_limits(Some("not-a-number")).map(|_| ())
        })
        .await;
    }
}
