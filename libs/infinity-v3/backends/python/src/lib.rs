//! Bring-your-own-Python inference backend.
//!
//! Rust owns scheduling, batching geometry and response envelopes. One Python
//! executor thread owns the model; the native backend worker never blocks on a
//! forward pass. The executor is selected at runtime through an installed entry
//! point or a `module:attribute` factory.

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
use tokio::sync::Semaphore;

/// Runtime configuration supplied by the shared frontend.
#[derive(Clone, Debug, Default)]
pub struct PythonBackendConfig {
    /// JSON object. Reserved keys: `backend`, `queue_depth`; other keys are
    /// passed to the plugin.
    pub config: Option<String>,
}

pub struct PythonBackend {
    executor: Executor,
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
        let config_json = config.config.unwrap_or_else(|| "{}".into());
        let value: serde_json::Value = serde_json::from_str(&config_json).map_err(|error| {
            BackendError::Start(format!("Invalid --python-config JSON: {error}"))
        })?;
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
        let executor = Executor::new(ExecutorConfig {
            selector,
            model_path: model_path.to_owned(),
            dtype: dtype.to_owned(),
            model_type: model_type.to_owned(),
            pooling: pool.to_string(),
            device_id,
            config_json,
            queue_size,
        })?;
        Ok(Self {
            executor,
            classifier,
            admission: Arc::new(Semaphore::new(queue_size)),
        })
    }

    fn submit(
        &self,
        route: &str,
        inputs: Arc<PaddedInputs>,
        complete: impl FnOnce(OutputSnapshot, &PaddedInputs) -> Result<InferenceResult, BackendError>
            + Send
            + 'static,
    ) -> Result<Inference<InferenceResult>, BackendError> {
        let permit =
            self.admission.clone().try_acquire_owned().map_err(|_| {
                BackendError::Overloaded("Python backend pending queue is full".into())
            })?;
        let receiver = Python::attach(|py| {
            let (envelope, receiver) = ResponseEnvelope::channel(py, Some(permit))
                .map_err(|error| BackendError::Inference(error.to_string()))?;
            let payload = PaddedInputs::to_python(inputs.clone(), py)
                .map_err(|error| BackendError::Inference(error.to_string()))?;
            self.executor
                .submit(py, route, envelope, payload)
                .map(|_| receiver)
        })?;
        let owned = inputs.clone();
        Ok(Inference::Deferred(Box::new(move || {
            let result = receiver
                .blocking_recv()
                .map_err(|_| {
                    BackendError::Inference("Python backend executor dropped its response".into())
                })?
                .map_err(|error| {
                    Python::try_attach(|py| {
                        if error.is_instance_of::<pyo3::exceptions::PyMemoryError>(py) {
                            BackendError::Overloaded(error.to_string())
                        } else {
                            BackendError::Inference(format!(
                                "Python backend forward failed: {error}"
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
    fn completed_requests_release_admission_permits() {
        let mut config = config();
        config.config = Some(
            r#"{"backend":"infinity_v3_python.testing:create_executor","queue_depth":1}"#.into(),
        );
        let backend = PythonBackend::new(
            Path::new("."),
            "float32",
            ModelType::Embedding(Pool::Mean),
            0,
            config,
        )
        .unwrap();
        for _ in 0..5 {
            let embeddings = backend.embed(batch()).unwrap();
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
    fn excessive_queue_depth_is_a_startup_error() {
        let result = PythonBackend::new(
            Path::new("."),
            "float32",
            ModelType::Embedding(Pool::Mean),
            0,
            PythonBackendConfig {
                config: Some(format!(
                    r#"{{"queue_depth":{}}}"#,
                    Semaphore::MAX_PERMITS + 1
                )),
            },
        );
        assert!(matches!(result, Err(BackendError::Start(_))));
    }
    #[test]
    fn python_memory_error_completes_as_overload_and_releases_admission() {
        Python::attach(|py| {
            let module = pyo3::types::PyModule::from_code(
                py,
                c"
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
                c"_infinity_memory_error_test.py",
                c"_infinity_memory_error_test",
            )
            .unwrap();
            pyo3::types::PyModule::import(py, "sys")
                .unwrap()
                .getattr("modules")
                .unwrap()
                .set_item("_infinity_memory_error_test", module)
                .unwrap();
        });
        let backend = PythonBackend::new(
            Path::new("."),
            "float32",
            ModelType::Embedding(Pool::Mean),
            0,
            PythonBackendConfig {
                config: Some(
                    r#"{"backend":"_infinity_memory_error_test:create_executor","queue_depth":1}"#
                        .into(),
                ),
            },
        )
        .unwrap();
        for _ in 0..2 {
            let result = backend.submit_embed(batch()).unwrap().complete();
            assert!(matches!(result, Err(BackendError::Overloaded(message))
                if message.contains("result budget exhausted")));
        }
        backend.shutdown();
    }
}
