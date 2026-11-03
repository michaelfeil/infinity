//! Bring-your-own-Python inference backend.
//!
//! Rust owns scheduling, batching geometry and response envelopes. The native
//! model worker calls the Python executor synchronously, and each submit must
//! complete its response before returning so one forward runs at a time. The
//! executor is selected at runtime through an installed entry point or a
//! `module:attribute` factory.

mod batch;
mod envelope;
mod executor;

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
use tokio::sync::oneshot;

/// Runtime configuration supplied by the shared frontend.
#[derive(Clone, Debug, Default)]
pub struct PythonBackendConfig {
    /// JSON object. Reserved key: `backend`; other keys are passed to the
    /// plugin.
    pub config: Option<String>,
}

fn parse_config_json(config_json: &str) -> Result<serde_json::Value, BackendError> {
    serde_json::from_str(config_json)
        .map_err(|error| BackendError::Start(format!("Invalid --python-config JSON: {error}")))
}

fn config_object(
    value: &serde_json::Value,
) -> Result<&serde_json::Map<String, serde_json::Value>, BackendError> {
    value
        .as_object()
        .ok_or_else(|| BackendError::Start("--python-config must be a JSON object".into()))
}

fn reject_obsolete_queue_depth(
    object: &serde_json::Map<String, serde_json::Value>,
) -> Result<(), BackendError> {
    if object.contains_key("queue_depth") {
        return Err(BackendError::Start(
            "--python-config queue_depth is no longer supported: the Python backend now executes \
             each forward synchronously on the model worker. Remove queue_depth from \
             --python-config; the shared frontend still controls request scheduling."
                .into(),
        ));
    }
    Ok(())
}

pub struct PythonBackend {
    executor: Executor,
    classifier: bool,
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
        let config_json = config.config.unwrap_or_else(|| "{}".into());
        let value = parse_config_json(&config_json)?;
        let object = config_object(&value)?;
        reject_obsolete_queue_depth(object)?;
        let selector = match object.get("backend") {
            Some(serde_json::Value::String(value)) => Some(value.clone()),
            Some(_) => {
                return Err(BackendError::Start(
                    "--python-config backend must be a string".into(),
                ))
            }
            None => None,
        };
        let executor = Executor::new(ExecutorConfig {
            selector,
            model_path: model_path.to_owned(),
            dtype: dtype.to_owned(),
            model_type: model_type.to_owned(),
            pooling: pool.to_string(),
            device_id,
            config_json,
        })?;
        Ok(Self {
            executor,
            classifier,
        })
    }

    fn submit<T>(
        &self,
        route: &str,
        inputs: Arc<PaddedInputs>,
        complete: impl FnOnce(OutputSnapshot, &PaddedInputs) -> Result<T, BackendError>,
    ) -> Result<Inference<T>, BackendError> {
        let output = Python::attach(|py| {
            let (envelope, mut receiver) = ResponseEnvelope::channel(py)
                .map_err(|error| BackendError::Inference(error.to_string()))?;
            let payload = PaddedInputs::to_python(inputs.clone(), py)
                .map_err(|error| BackendError::Inference(error.to_string()))?;
            self.executor.submit(py, route, envelope, payload)?;
            match receiver.try_recv() {
                Ok(Ok(snapshot)) => Ok(snapshot),
                Ok(Err(error)) => Err(forward_error(py, error)),
                Err(oneshot::error::TryRecvError::Empty | oneshot::error::TryRecvError::Closed) => {
                    Err(BackendError::Inference(
                        "Python backend submit returned without completing its response; \
                         call response.set_result or response.set_exception before submit returns"
                            .into(),
                    ))
                }
            }
        })?;
        complete(output, &inputs).map(Inference::Ready)
    }
}

fn forward_error(py: Python<'_>, error: PyErr) -> BackendError {
    if error.is_instance_of::<pyo3::exceptions::PyMemoryError>(py) {
        BackendError::Overloaded(error.to_string())
    } else {
        BackendError::Inference(format!("Python backend forward failed: {error}"))
    }
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
        self.submit("embed", inputs, pack_embeddings)
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
            pack_predictions(output, inputs, &pool)
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
        self.executor.health()
    }

    fn shutdown(&self) {
        self.executor.close();
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

    fn config() -> PythonBackendConfig {
        PythonBackendConfig {
            config: Some(
                r#"{"backend":"infinity_v3_python.testing:create_executor","device":"cpu"}"#.into(),
            ),
        }
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
    fn repeated_synchronous_requests_complete_inline() {
        let backend = PythonBackend::new(
            Path::new("."),
            "float32",
            ModelType::Embedding(Pool::Mean),
            0,
            config(),
        )
        .unwrap();
        for _ in 0..5 {
            let submission = backend.submit_embed(batch()).unwrap();
            assert!(matches!(submission, Inference::Ready(_)));
            let embeddings = submission.complete().unwrap();
            assert_eq!(embeddings.len(), 2);
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
    fn obsolete_queue_depth_is_rejected_with_an_actionable_error() {
        let result = PythonBackend::new(
            Path::new("."),
            "float32",
            ModelType::Embedding(Pool::Mean),
            0,
            PythonBackendConfig {
                config: Some(
                    r#"{"backend":"infinity_v3_python.testing:create_executor","queue_depth":32}"#
                        .into(),
                ),
            },
        );
        match result {
            Err(BackendError::Start(message)) => {
                assert!(message.contains("queue_depth"), "{message}");
                assert!(message.contains("Remove queue_depth"), "{message}");
            }
            Err(error) => panic!("expected a startup error, got {error}"),
            Ok(_) => panic!("queue_depth must be rejected"),
        }
    }

    fn install_module(name: &str, source: &str) {
        Python::attach(|py| {
            let source = std::ffi::CString::new(source).unwrap();
            let filename = std::ffi::CString::new(format!("{name}.py")).unwrap();
            let module_name = std::ffi::CString::new(name).unwrap();
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

    fn expect_error(result: Result<Inference<Embeddings>, BackendError>) -> BackendError {
        match result {
            Ok(_) => panic!("expected a Python backend error"),
            Err(error) => error,
        }
    }

    #[test]
    fn submit_without_completion_fails_with_a_sync_contract_error() {
        install_module(
            "_infinity_pending_test",
            "
PROTOCOL_VERSION = 1
class Executor:
    def submit(self, route, response, payload):
        pass
    def healthy(self):
        return True
    def close(self):
        pass
def create_executor(**kwargs):
    return Executor()
",
        );
        let backend = PythonBackend::new(
            Path::new("."),
            "float32",
            ModelType::Embedding(Pool::Mean),
            0,
            PythonBackendConfig {
                config: Some(r#"{"backend":"_infinity_pending_test:create_executor"}"#.into()),
            },
        )
        .unwrap();
        let started = std::time::Instant::now();
        let error = expect_error(backend.submit_embed(batch()));
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
        assert!(
            matches!(error, BackendError::Inference(ref message)
                if message.contains("without completing") && message.contains("set_result")),
            "{error}"
        );
        backend.shutdown();
    }

    #[test]
    fn submit_callback_errors_are_contained_and_the_backend_recovers() {
        install_module(
            "_infinity_callback_error_test",
            "
import numpy as np
PROTOCOL_VERSION = 1
class Executor:
    def __init__(self):
        self.calls = 0
    def submit(self, route, response, payload):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError('callback exploded')
        if self.calls == 2:
            response.set_exception(ValueError('callback reported failure'))
            return
        response.set_result(np.ones((2, 8), dtype=np.float32))
    def healthy(self):
        return True
    def close(self):
        pass
def create_executor(**kwargs):
    return Executor()
",
        );
        let backend = PythonBackend::new(
            Path::new("."),
            "float32",
            ModelType::Embedding(Pool::Mean),
            0,
            PythonBackendConfig {
                config: Some(
                    r#"{"backend":"_infinity_callback_error_test:create_executor"}"#.into(),
                ),
            },
        )
        .unwrap();
        let error = expect_error(backend.submit_embed(batch()));
        assert!(
            matches!(error, BackendError::Inference(ref message) if message.contains("callback exploded")),
            "{error}"
        );
        let error = expect_error(backend.submit_embed(batch()));
        assert!(
            matches!(error, BackendError::Inference(ref message) if message.contains("callback reported failure")),
            "{error}"
        );
        let embeddings = backend.submit_embed(batch()).unwrap().complete().unwrap();
        assert_eq!(embeddings.len(), 2);
        backend.shutdown();
    }

    #[test]
    fn python_memory_error_completes_as_overload_and_recovers() {
        install_module(
            "_infinity_memory_error_test",
            "
PROTOCOL_VERSION = 1
class Executor:
    def submit(self, route, response, payload):
        self.last_response = response
        response.set_exception(MemoryError('result budget exhausted'))
    def healthy(self):
        return True
    def close(self):
        pass
def create_executor(**kwargs):
    return Executor()
",
        );
        let backend = PythonBackend::new(
            Path::new("."),
            "float32",
            ModelType::Embedding(Pool::Mean),
            0,
            PythonBackendConfig {
                config: Some(r#"{"backend":"_infinity_memory_error_test:create_executor"}"#.into()),
            },
        )
        .unwrap();
        for _ in 0..2 {
            let error = expect_error(backend.submit_embed(batch()));
            assert!(matches!(error, BackendError::Overloaded(ref message)
                if message.contains("result budget exhausted")));
        }
        backend.shutdown();
    }
}
