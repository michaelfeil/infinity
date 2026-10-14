//! Keeps transport input admission alive while background jobs own request data.
use std::{future::Future, sync::Arc};

#[derive(Clone)]
pub(crate) struct InputMemoryOwner {
    _owner: Arc<dyn Send + Sync>,
}
impl std::fmt::Debug for InputMemoryOwner {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("InputMemoryOwner")
            .finish_non_exhaustive()
    }
}
tokio::task_local! { static INPUT_MEMORY: InputMemoryOwner; }

/// Scope transport admission so queued Core jobs can retain the same owner.
pub async fn with_input_memory<T: Send + Sync + 'static, F: Future>(
    owner: Arc<T>,
    future: F,
) -> F::Output {
    let owner = InputMemoryOwner { _owner: owner };
    INPUT_MEMORY.scope(owner, future).await
}

pub(crate) fn current() -> Option<InputMemoryOwner> {
    INPUT_MEMORY.try_with(Clone::clone).ok()
}

/// Opaque admission ownership retained with a background operation's result.
pub struct InputMemoryGuard {
    _owner: Option<InputMemoryOwner>,
}
impl std::fmt::Debug for InputMemoryGuard {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("InputMemoryGuard")
            .finish_non_exhaustive()
    }
}

/// Keep scoped input admission through detached work and its returned payload.
pub fn spawn_blocking<F, R>(operation: F) -> tokio::task::JoinHandle<(R, InputMemoryGuard)>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    let memory = InputMemoryGuard { _owner: current() };
    tokio::task::spawn_blocking(move || (operation(), memory))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cancelled_preprocessing_keeps_input_admission_until_worker_finishes() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .max_blocking_threads(1)
            .build()
            .unwrap();
        runtime.block_on(async {
            let budget = Arc::new(tokio::sync::Semaphore::new(1));
            let (release, blocked) = std::sync::mpsc::channel();
            let (started, ready) = tokio::sync::oneshot::channel();
            let blocker = tokio::task::spawn_blocking(move || {
                started.send(()).unwrap();
                blocked
                    .recv_timeout(std::time::Duration::from_secs(5))
                    .unwrap();
            });
            ready.await.unwrap();
            let owner = Arc::new(budget.clone().try_acquire_owned().unwrap());
            let (queued, pending) = tokio::sync::oneshot::channel();
            let request = tokio::spawn(with_input_memory(owner, async move {
                let work = spawn_blocking(|| String::from("prepared request"));
                queued.send(()).unwrap();
                work.await
            }));
            pending.await.unwrap();
            request.abort();
            assert!(request.await.unwrap_err().is_cancelled());
            assert_eq!(budget.available_permits(), 0);
            release.send(()).unwrap();
            blocker.await.unwrap();
            tokio::time::timeout(std::time::Duration::from_secs(2), async {
                while budget.available_permits() == 0 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
            let owner = Arc::new(budget.clone().try_acquire_owned().unwrap());
            let (value, memory) = with_input_memory(owner, async {
                spawn_blocking(|| String::from("next request"))
                    .await
                    .unwrap()
            })
            .await;
            assert_eq!(value, "next request");
            assert_eq!(budget.available_permits(), 0);
            drop(value);
            drop(memory);
            assert_eq!(budget.available_permits(), 1);
        });
    }
}
