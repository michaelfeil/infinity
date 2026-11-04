//! Bring-your-own-Python inference backend.
//!
//! Rust owns scheduling, batching geometry and response envelopes. Each backend
//! replica owns one persistent Python worker and an admission-bounded Rust
//! command queue; the native backend worker never blocks on a forward pass. The
//! executor is selected at runtime through an installed entry point or a
//! `module:attribute` factory.

mod batch;
mod envelope;
mod executor;
mod worker;

use batch::{pack_embeddings, pack_predictions, pad_batch, PaddedInputs};
use envelope::{OutputSnapshot, ResponseEnvelope};
use executor::{Executor, ExecutorConfig};
use infinity_v3_backend_core::{
    Backend, BackendError, Batch, Embeddings, Inference, ModelType, Pool, Predictions,
    TokenPredictions,
};
use pyo3::prelude::*;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Semaphore;
use worker::Worker;

/// Runtime configuration supplied by the shared frontend.
#[derive(Clone, Debug, Default)]
pub struct PythonBackendConfig {
    /// JSON object. Reserved keys: `backend`, `queue_depth`; other keys are
    /// passed to the plugin.
    pub config: Option<String>,
}

/// Optional replica capabilities reported by a Python plugin before startup.
/// Protocol-1 plugins without a discovery hook keep these single-device
/// defaults.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PythonCapabilities {
    pub visible_device_count: usize,
    pub supports_replication: bool,
}

impl Default for PythonCapabilities {
    fn default() -> Self {
        Self {
            visible_device_count: 1,
            supports_replication: false,
        }
    }
}

struct ParsedConfig {
    selector: Option<String>,
    queue_size: usize,
    config_json: String,
}

fn parse_config(config: &PythonBackendConfig) -> Result<ParsedConfig, BackendError> {
    let config_json = config.config.clone().unwrap_or_else(|| "{}".into());
    let value: serde_json::Value = serde_json::from_str(&config_json)
        .map_err(|error| BackendError::Start(format!("Invalid --python-config JSON: {error}")))?;
    let object = value
        .as_object()
        .ok_or_else(|| BackendError::Start("--python-config must be a JSON object".into()))?;
    let selector = match object.get("backend") {
        Some(serde_json::Value::String(value)) => Some(value.clone()),
        Some(_) => {
            return Err(BackendError::Start(
                "--python-config backend must be a string".into(),
            ))
        }
        None => None,
    };
    let queue_size = match object.get("queue_depth") {
        Some(value) => value
            .as_u64()
            .filter(|&value| value > 0 && value <= Semaphore::MAX_PERMITS as u64)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| {
                BackendError::Start(
                    "--python-config queue_depth must be a positive integer within the semaphore limit".into(),
                )
            })?,
        None => 32,
    };
    Ok(ParsedConfig {
        selector,
        queue_size,
        config_json,
    })
}

pub struct PythonBackend {
    worker: Worker,
    classifier: bool,
    admission: Arc<Semaphore>,
}

impl PythonBackend {
    pub fn new(
        model_path: &Path,
        dtype: &str,
        model_type: ModelType,
        device_id: usize,
        config: PythonBackendConfig,
    ) -> Result<Self, BackendError> {
        let (pool, classifier, model_type) = match model_type {
            ModelType::Embedding(pool) => (pool, false, "embedding"),
            ModelType::Classifier => (Pool::Cls, true, "classifier"),
            ModelType::Decision => {
                return Err(BackendError::Start(
                    "Python backend does not support decision models".into(),
                ))
            }
        };
        let parsed = parse_config(&config)?;
        let executor = Executor::new(ExecutorConfig {
            selector: parsed.selector,
            model_path: model_path.to_owned(),
            dtype: dtype.to_owned(),
            model_type: model_type.to_owned(),
            pooling: pool.to_string(),
            device_id,
            config_json: parsed.config_json,
            queue_size: parsed.queue_size,
        })?;
        let worker = Worker::start(executor)?;
        Ok(Self {
            worker,
            classifier,
            admission: Arc::new(Semaphore::new(parsed.queue_size)),
        })
    }

    /// Resolve optional replica capabilities using the same selector and
    /// config JSON as backend construction. The hook runs before placement.
    pub fn discover_capabilities(
        config: &PythonBackendConfig,
    ) -> Result<PythonCapabilities, BackendError> {
        let parsed = parse_config(config)?;
        executor::discover_capabilities(parsed.selector, &parsed.config_json)
    }

    fn submit(
        &self,
        route: &'static str,
        inputs: Arc<PaddedInputs>,
        complete: impl FnOnce(OutputSnapshot, &PaddedInputs) -> Result<InferenceResult, BackendError>
            + Send
            + 'static,
    ) -> Result<Inference<InferenceResult>, BackendError> {
        let permit =
            self.admission.clone().try_acquire_owned().map_err(|_| {
                BackendError::Overloaded("Python backend pending queue is full".into())
            })?;
        let (envelope, receiver) = ResponseEnvelope::new(Some(permit));
        self.worker.submit(route, inputs.clone(), envelope)?;
        let owned = inputs;
        Ok(Inference::Deferred(Box::new(move || {
            let result = receiver
                .blocking_recv()
                .map_err(|_| {
                    BackendError::Inference("Python backend executor dropped its response".into())
                })?
                .map_err(|error| {
                    Python::try_attach(|py| {
                        let message = error.to_string();
                        if error.is_instance_of::<pyo3::exceptions::PyMemoryError>(py)
                            || message.contains("queue is full")
                        {
                            BackendError::Overloaded(message)
                        } else if message.contains("executor is closed") {
                            BackendError::Unhealthy
                        } else {
                            BackendError::Inference(format!(
                                "Python backend forward failed: {message}"
                            ))
                        }
                    })
                    .unwrap_or_else(|| {
                        BackendError::Inference(
                            "Python interpreter is unavailable for backend completion".into(),
                        )
                    })
                })?;
            complete(result, &owned)
        })))
    }
}

enum InferenceResult {
    Embeddings(Embeddings),
    Predictions(Predictions),
}

impl Backend for PythonBackend {
    fn submit_embed(&self, batch: Batch) -> Result<Inference<Embeddings>, BackendError> {
        if self.classifier {
            return Err(BackendError::Inference(
                "Python checkpoint is a classifier".into(),
            ));
        }
        let inputs = Arc::new(pad_batch(&batch)?);
        if inputs.pooled_indices.is_empty() && inputs.raw_indices.is_empty() {
            return Err(BackendError::Inference(
                "Python embedding batch has no requested rows".into(),
            ));
        }
        self.submit("embed", inputs, |output, inputs| {
            pack_embeddings(output, inputs).map(InferenceResult::Embeddings)
        })?
        .try_map(|result| match result {
            InferenceResult::Embeddings(value) => Ok(value),
            InferenceResult::Predictions(_) => Err(BackendError::Inference(
                "Python backend returned predictions for an embedding request".into(),
            )),
        })
    }

    fn submit_predict(&self, batch: Batch) -> Result<Inference<Predictions>, BackendError> {
        if !self.classifier {
            return Err(BackendError::Inference(
                "Python checkpoint has no classification head".into(),
            ));
        }
        let inputs = Arc::new(pad_batch(&batch)?);
        self.submit("predict", inputs, |output, inputs| {
            let pool = infinity_v3_backend_core::result_memory::global_pool()?;
            pack_predictions(output, inputs, &pool).map(InferenceResult::Predictions)
        })?
        .try_map(|result| match result {
            InferenceResult::Predictions(value) => Ok(value),
            InferenceResult::Embeddings(_) => Err(BackendError::Inference(
                "Python backend returned embeddings for a prediction request".into(),
            )),
        })
    }

    fn submit_predict_tokens(
        &self,
        _batch: Batch,
    ) -> Result<Inference<TokenPredictions>, BackendError> {
        Err(BackendError::Inference(
            "Python backend does not support token predictions".into(),
        ))
    }

    fn health(&self) -> Result<(), BackendError> {
        self.worker.health()
    }

    fn shutdown(&self) {
        self.worker.close();
    }

    fn embed(&self, batch: Batch) -> Result<Embeddings, BackendError> {
        self.submit_embed(batch)?.complete()
    }

    fn predict(&self, batch: Batch) -> Result<Predictions, BackendError> {
        self.submit_predict(batch)?.complete()
    }

    fn predict_tokens(&self, batch: Batch) -> Result<TokenPredictions, BackendError> {
        self.submit_predict_tokens(batch)?.complete()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use infinity_v3_backend_core::Embedding;
    use std::ffi::CString;
    use std::time::Duration;

    fn install_executor(name: &str, extra: &str, submit: &str) {
        let source = format!(
            r#"PROTOCOL_VERSION = 1
import numpy as np
INIT = 0
{extra}

class Executor:
    def __init__(self):
        global INIT
        INIT += 1
    def submit(self, route, response, payload):
{submit}
    def healthy(self):
        return True
    def close(self):
        pass

def create_executor(**kwargs):
    return Executor()
"#
        );
        Python::attach(|py| {
            let source = CString::new(source).unwrap();
            let filename = CString::new(format!("{name}.py")).unwrap();
            let module_name = CString::new(name).unwrap();
            let module = pyo3::types::PyModule::from_code(
                py,
                source.as_c_str(),
                filename.as_c_str(),
                module_name.as_c_str(),
            )
            .unwrap();
            pyo3::types::PyModule::import(py, "sys")
                .unwrap()
                .getattr("modules")
                .unwrap()
                .set_item(name, module)
                .unwrap();
        });
    }

    fn config_with(json: &str) -> PythonBackendConfig {
        PythonBackendConfig {
            config: Some(json.into()),
        }
    }

    fn config() -> PythonBackendConfig {
        config_with(r#"{"backend":"infinity_v3_python.testing:create_executor","device":"cpu"}"#)
    }

    fn backend_on(name: &str, device_id: usize, queue_depth: usize) -> PythonBackend {
        PythonBackend::new(
            Path::new("."),
            "float32",
            ModelType::Embedding(Pool::Mean),
            device_id,
            config_with(&format!(
                r#"{{"backend":"{name}:create_executor","queue_depth":{queue_depth}}}"#
            )),
        )
        .unwrap()
    }

    fn backend(name: &str, queue_depth: usize) -> PythonBackend {
        backend_on(name, 0, queue_depth)
    }

    fn batch() -> Batch {
        Batch {
            multimodal: vec![],
            input_ids: vec![10, 11, 20, 21, 22],
            token_type_ids: vec![0, 0, 1, 1, 1],
            position_ids: vec![0, 1, 0, 1, 2],
            cumulative_seq_lengths: vec![0, 2, 5],
            max_length: 3,
            pooled_indices: vec![0, 1],
            raw_indices: vec![],
            compact_input_ids: None,
            compact_position_ids: None,
            fold_gather: None,
            scatter_unfold: None,
            tokens: vec![],
            offsets: vec![],
        }
    }

    #[test]
    fn fake_backend_completes_embeddings_through_the_python_executor() {
        let backend = PythonBackend::new(
            Path::new("."),
            "float32",
            ModelType::Embedding(Pool::Mean),
            0,
            config(),
        )
        .unwrap();
        backend.health().unwrap();
        let embeddings = backend.embed(batch()).unwrap();
        assert_eq!(embeddings.len(), 2);
        assert!(matches!(&embeddings[&0], Embedding::Shared(values) if values.len() == 8));
        assert!(matches!(&embeddings[&1], Embedding::Shared(values) if values.len() == 8));
        backend.shutdown();
    }

    #[test]
    fn completed_requests_release_admission_permits() {
        let backend = backend("infinity_v3_python.testing", 1);
        for _ in 0..5 {
            assert_eq!(backend.embed(batch()).unwrap().len(), 2);
        }
        backend.shutdown();
    }

    #[test]
    fn fake_backend_returns_classifier_rows() {
        let backend = PythonBackend::new(
            Path::new("."),
            "float32",
            ModelType::Classifier,
            0,
            config(),
        )
        .unwrap();
        let predictions = backend.predict(batch()).unwrap();
        assert_eq!(predictions.len(), 2);
        assert_eq!(predictions[&0], vec![1.0, 0.0]);
        assert_eq!(predictions[&1], vec![0.0, 1.0]);
        backend.shutdown();
    }

    #[test]
    fn excessive_queue_depth_is_a_startup_error() {
        let result = PythonBackend::new(
            Path::new("."),
            "float32",
            ModelType::Embedding(Pool::Mean),
            0,
            config_with(&format!(
                r#"{{"queue_depth":{}}}"#,
                Semaphore::MAX_PERMITS + 1
            )),
        );
        assert!(matches!(result, Err(BackendError::Start(_))));
    }

    #[test]
    fn submit_errors_complete_and_release_admission() {
        for (name, submit, overload, message) in [
            (
                "_infinity_memory_error_test",
                "response.set_exception(MemoryError('result budget exhausted'))",
                true,
                "result budget exhausted",
            ),
            (
                "_infinity_submit_failure_test",
                "raise RuntimeError('submit exploded')",
                false,
                "submit exploded",
            ),
            (
                "_infinity_plugin_queue_full_test",
                "raise RuntimeError('queue is full')",
                true,
                "queue is full",
            ),
        ] {
            install_executor(name, "", &format!("        {submit}"));
            let backend = backend(name, 1);
            for _ in 0..2 {
                let result = backend.submit_embed(batch()).unwrap().complete();
                let matched = match overload {
                    true => matches!(&result, Err(BackendError::Overloaded(actual))
                        if actual.contains(message)),
                    false => matches!(&result, Err(BackendError::Inference(actual))
                        if actual.contains(message)),
                };
                assert!(matched, "{name} error did not match {message}");
                assert_eq!(backend.admission.available_permits(), 1);
                backend.health().unwrap();
            }
            backend.shutdown();
        }
    }

    #[test]
    fn capability_discovery_defaults_validation_and_config() {
        install_executor(
            "_infinity_capability_test",
            r#"import json

def capabilities(config_json):
    config = json.loads(config_json)
    mode = config.get("mode")
    if mode == "zero":
        return {"visible_device_count": 0, "supports_replication": False}
    if mode == "type":
        return {"visible_device_count": True, "supports_replication": False}
    if mode == "unknown":
        return {"devices": 4}
    return {
        "visible_device_count": int(config.get("devices", 1)),
        "supports_replication": bool(config.get("replicate", False)),
    }"#,
            "        pass",
        );
        let defaults = PythonBackend::discover_capabilities(&config()).unwrap();
        assert_eq!(defaults, PythonCapabilities::default());
        for (mode, expected) in [
            ("zero", "visible_device_count"),
            ("type", "positive integer"),
            ("unknown", "dict with visible_device_count"),
        ] {
            let error = PythonBackend::discover_capabilities(&config_with(&format!(
                r#"{{"backend":"_infinity_capability_test:create_executor","mode":"{mode}"}}"#
            )))
            .unwrap_err();
            assert!(
                matches!(&error, BackendError::Start(message) if message.contains(expected)),
                "unexpected error: {error}"
            );
        }
        let capabilities = PythonBackend::discover_capabilities(&config_with(
            r#"{"backend":"_infinity_capability_test:create_executor","devices":3,"replicate":true}"#,
        ))
        .unwrap();
        assert_eq!(
            capabilities,
            PythonCapabilities {
                visible_device_count: 3,
                supports_replication: true,
            }
        );
    }

    #[test]
    fn blocked_synchronous_plugin_submit_returns_deferred_immediately() {
        install_executor(
            "_infinity_blocked_submit_test",
            r#"import threading
_started = threading.Event()
_release = threading.Event()

def wait_started(timeout):
    return _started.wait(timeout=timeout)

def release():
    _release.set()"#,
            r#"        _started.set()
        if not _release.wait(timeout=2.0):
            raise TimeoutError("release gate timed out")
        rows = int(payload["batch_size"])
        response.set_result(np.full((rows, 8), 3.0, dtype=np.float32))"#,
        );
        let backend = backend("_infinity_blocked_submit_test", 2);
        let deferred = backend.submit_embed(batch()).unwrap();
        let started = Python::attach(|py| {
            let module =
                pyo3::types::PyModule::import(py, "_infinity_blocked_submit_test").unwrap();
            module
                .getattr("wait_started")
                .unwrap()
                .call1((1.0,))
                .unwrap()
                .extract::<bool>()
                .unwrap()
        });
        Python::attach(|py| {
            let module =
                pyo3::types::PyModule::import(py, "_infinity_blocked_submit_test").unwrap();
            module.getattr("release").unwrap().call0().unwrap();
        });
        assert!(
            started,
            "plugin submit did not start while the submission was deferred"
        );
        assert_eq!(deferred.complete().unwrap().len(), 2);
        backend.shutdown();
    }

    #[test]
    fn replica_workers_overlap_python_waits_that_release_the_gil() {
        install_executor(
            "_infinity_overlap_test",
            "import threading\nbarrier = threading.Barrier(2)",
            r#"        barrier.wait(timeout=2)
        rows = int(payload["batch_size"])
        response.set_result(np.full((rows, 8), 4.0, dtype=np.float32))"#,
        );
        let first = backend_on("_infinity_overlap_test", 0, 1);
        let second = backend_on("_infinity_overlap_test", 1, 1);
        let first_deferred = first.submit_embed(batch()).unwrap();
        let second_deferred = second.submit_embed(batch()).unwrap();
        assert_eq!(first_deferred.complete().unwrap().len(), 2);
        assert_eq!(second_deferred.complete().unwrap().len(), 2);
        first.shutdown();
        second.shutdown();
    }

    #[test]
    fn persistent_worker_reuses_one_executor_thread() {
        install_executor(
            "_infinity_worker_reuse_test",
            r#"import threading
SUBMITS = 0
THREADS = set()

def stats():
    return INIT, SUBMITS, len(THREADS)"#,
            r#"        global SUBMITS
        SUBMITS += 1
        THREADS.add(threading.get_ident())
        rows = int(payload["batch_size"])
        response.set_result(np.full((rows, 8), 5.0, dtype=np.float32))"#,
        );
        let backend = backend("_infinity_worker_reuse_test", 2);
        for _ in 0..3 {
            assert_eq!(backend.embed(batch()).unwrap().len(), 2);
        }
        let (initializations, submits, threads) = Python::attach(|py| {
            let module = pyo3::types::PyModule::import(py, "_infinity_worker_reuse_test").unwrap();
            module
                .getattr("stats")
                .unwrap()
                .call0()
                .unwrap()
                .extract::<(usize, usize, usize)>()
                .unwrap()
        });
        assert_eq!((initializations, submits, threads), (1, 3, 1));
        backend.shutdown();
    }

    #[test]
    fn attached_thread_can_start_health_and_shutdown_backend() {
        let (finished, receiver) = std::sync::mpsc::channel();
        let handle = std::thread::spawn(move || {
            let result = Python::attach(|_py| {
                let backend = PythonBackend::new(
                    Path::new("."),
                    "float32",
                    ModelType::Embedding(Pool::Mean),
                    0,
                    config(),
                )?;
                backend.health()?;
                backend.shutdown();
                Ok::<(), BackendError>(())
            });
            let _ = finished.send(result);
        });
        match receiver.recv_timeout(Duration::from_secs(5)) {
            Ok(result) => result.unwrap(),
            Err(error) => panic!("attached-thread backend lifecycle did not finish: {error}"),
        }
        handle.join().unwrap();
    }

    #[test]
    fn shutdown_drains_canceled_and_pending_requests_and_releases_admission() {
        install_executor(
            "_infinity_shutdown_drain_test",
            "import time",
            r#"        time.sleep(0.05)
        rows = int(payload["batch_size"])
        response.set_result(np.full((rows, 8), 6.0, dtype=np.float32))"#,
        );
        let backend = backend("_infinity_shutdown_drain_test", 2);
        let canceled = backend.submit_embed(batch()).unwrap();
        drop(canceled);
        let pending = backend.submit_embed(batch()).unwrap();
        backend.shutdown();
        assert_eq!(pending.complete().unwrap().len(), 2);
        assert_eq!(backend.admission.available_permits(), 2);
    }
}
