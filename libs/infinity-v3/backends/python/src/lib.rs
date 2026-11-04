//! Bring-your-own-Python inference backend.
//!
//! Rust owns scheduling, batching geometry and response envelopes. By default
//! one Python thread per backend replica executes plugin submissions, so the
//! native backend worker never blocks on a forward pass. A plugin may opt into
//! synchronous single-flight execution; it then runs on the serialized Rust
//! backend worker and must complete the envelope before returning. The executor
//! is selected at runtime through an installed entry point or a
//! `module:attribute` factory.

mod batch;
mod envelope;
mod executor;

use batch::{pack_embeddings, pack_predictions, pad_batch, PaddedInputs};
use envelope::{OutputSnapshot, ResponseEnvelope};
use executor::{Executor, ExecutorConfig, PackedLimits};
use infinity_v3_backend_core::{
    Backend, BackendError, Batch, Embeddings, Inference, ModelType, Pool, Predictions,
    TokenPredictions,
};
use pyo3::prelude::*;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Semaphore;

pub use infinity_v3_backend_core::ServingLimits;

/// Runtime configuration supplied by the shared frontend.
#[derive(Clone, Debug, Default)]
pub struct PythonBackendConfig {
    /// JSON object. The `backend` key selects the plugin; other keys are
    /// passed to the plugin.
    pub config: Option<String>,
    /// Resolved serving limits forwarded to opt-in Python factories.
    pub serving_limits: Option<ServingLimits>,
}

struct ParsedConfig {
    selector: Option<String>,
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
    Ok(ParsedConfig {
        selector,
        config_json,
    })
}

pub struct PythonBackend {
    executor: Executor,
    classifier: bool,
    admission: Arc<Semaphore>,
    packed_limits: Option<PackedLimits>,
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
        let serving_limits = config.serving_limits;
        let parsed = parse_config(&config)?;
        let executor = Executor::new(ExecutorConfig {
            selector: parsed.selector,
            model_path: model_path.to_owned(),
            dtype: dtype.to_owned(),
            model_type: model_type.to_owned(),
            pooling: pool.to_string(),
            device_id,
            config_json: parsed.config_json,
            serving_limits,
        })?;
        let capabilities = match executor.capabilities() {
            Ok(capabilities) => capabilities,
            Err(error) => {
                executor.close();
                return Err(error);
            }
        };
        let packed_limits = capabilities.and_then(|capabilities| capabilities.packed_limits);
        if let (Some(serving), Some(limits)) = (serving_limits, packed_limits) {
            if let Err(error) = validate_capacity(serving, limits) {
                executor.close();
                return Err(error);
            }
        }
        Ok(Self {
            executor,
            classifier,
            admission: Arc::new(Semaphore::new(1)),
            packed_limits,
        })
    }

    /// Discover optional replica capabilities before backend placement.
    pub fn discover_capabilities(
        config: &PythonBackendConfig,
    ) -> Result<(usize, bool), BackendError> {
        let parsed = parse_config(config)?;
        executor::discover_capabilities(parsed.selector, &parsed.config_json)
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
        Ok(Inference::Queued(Box::new(move || {
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

fn effective_max_requests(serving: ServingLimits, limits: PackedLimits) -> usize {
    serving
        .max_batch_requests
        .map_or(limits.max_sequences, |requests| {
            requests.min(limits.max_sequences)
        })
}

fn reachable_total_tokens(serving: ServingLimits, limits: PackedLimits) -> usize {
    let request_bound =
        effective_max_requests(serving, limits).saturating_mul(serving.max_input_length);
    serving.max_batch_tokens.min(request_bound)
}

fn validate_capacity(serving: ServingLimits, limits: PackedLimits) -> Result<(), BackendError> {
    let reachable_total = reachable_total_tokens(serving, limits);
    if reachable_total > limits.max_total_tokens {
        return Err(BackendError::Start(format!(
            "Python backend capacity is smaller than the reachable serving batch: \
             reachable_total_tokens={reachable_total} (max_batch_tokens={}, \
             max_input_length={}, effective_max_requests={}) > plugin \
             packed_limits.max_total_tokens={}. Increase the plugin token capacity or lower \
             --max-batch-tokens/--max-batch-requests.",
            serving.max_batch_tokens,
            serving.max_input_length,
            effective_max_requests(serving, limits),
            limits.max_total_tokens
        )));
    }
    if serving.max_input_length > limits.max_seq_len {
        return Err(BackendError::Start(format!(
            "Python backend capacity is smaller than the resolved serving limits: \
             max_input_length={} > plugin packed_limits.max_seq_len={}. Increase the plugin \
             sequence capacity or lower --max-sequence-length/--max-batch-tokens.",
            serving.max_input_length, limits.max_seq_len
        )));
    }
    Ok(())
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

    fn max_batch_size(&self) -> Option<usize> {
        self.packed_limits.map(|limits| limits.max_sequences)
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
    use serial_test::serial;
    use std::ffi::CString;

    fn install_module(name: &str, source: &str) {
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

    fn config() -> PythonBackendConfig {
        PythonBackendConfig {
            config: Some(
                r#"{"backend":"infinity_v3_python.testing:create_executor","device":"cpu"}"#.into(),
            ),
            serving_limits: None,
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
        let backend = PythonBackend::new(
            Path::new("."),
            "float32",
            ModelType::Embedding(Pool::Mean),
            0,
            config(),
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

    fn python_config(name: &str, serving_limits: Option<ServingLimits>) -> PythonBackendConfig {
        PythonBackendConfig {
            config: Some(format!(r#"{{"backend":"{name}:create_executor"}}"#)),
            serving_limits,
        }
    }

    const CAPABILITY_MODULE: &str = r#"
PROTOCOL_VERSION = 1
RECEIVED = None

class Executor:
    def __init__(self, capabilities):
        self._capabilities = capabilities
    def capabilities(self):
        return self._capabilities
    def submit(self, route, response, payload):
        response.set_exception(ValueError('not used'))
    def healthy(self):
        return True
    def close(self):
        pass

def create_executor(*, serving_limits=None, **kwargs):
    global RECEIVED
    RECEIVED = None if serving_limits is None else dict(serving_limits)
    return Executor(CAPABILITIES)
"#;

    const LIFECYCLE_MODULE: &str = r#"
PROTOCOL_VERSION = 1
CLOSE_COUNT = 0

class Executor:
    def capabilities(self):
        return CAPABILITIES
    def submit(self, route, response, payload):
        response.set_exception(ValueError('not used'))
    def healthy(self):
        return True
    def close(self):
        global CLOSE_COUNT
        CLOSE_COUNT += 1

def create_executor(**kwargs):
    return Executor()
"#;

    const SYNC_MODULE: &str = r#"
PROTOCOL_VERSION = 1
import threading
import numpy as np

CAPABILITY_CALLS = 0
FACTORY_THREAD = None
SUBMIT_THREAD = None
INLINE_COMPLETION = False
SUBMIT_CALLS = 0
CLOSE_COUNT = 0
RECEIVED = None

class Executor:
    SENTINEL = True
    def capabilities(self):
        global CAPABILITY_CALLS
        CAPABILITY_CALLS += 1
        return {
            "protocol_version": 1,
            SYNC_FIELD
            "packed_limits": {
                "max_total_tokens": 10003,
                "max_sequences": 32,
                "max_seq_len": 2301,
            },
        }
    def submit(self, route, response, payload):
        global SUBMIT_THREAD, INLINE_COMPLETION, SUBMIT_CALLS
        SUBMIT_CALLS += 1
        SUBMIT_THREAD = threading.get_ident()
        result = np.zeros((2, 8), dtype=np.float32)
        response.set_result(result)
        try:
            response.set_result(result)
        except RuntimeError as error:
            INLINE_COMPLETION = "already completed" in str(error)
    def healthy(self):
        return True
    def close(self):
        global CLOSE_COUNT
        CLOSE_COUNT += 1

def create_executor(*, serving_limits=None, **kwargs):
    global FACTORY_THREAD, RECEIVED
    FACTORY_THREAD = threading.get_ident()
    RECEIVED = None if serving_limits is None else dict(serving_limits)
    return Executor()
"#;

    fn module_usize(name: &str, attribute: &str) -> usize {
        Python::attach(|py| {
            pyo3::types::PyModule::import(py, name)
                .unwrap()
                .getattr(attribute)
                .unwrap()
                .extract::<usize>()
                .unwrap()
        })
    }

    fn module_u64(name: &str, attribute: &str) -> u64 {
        Python::attach(|py| {
            pyo3::types::PyModule::import(py, name)
                .unwrap()
                .getattr(attribute)
                .unwrap()
                .extract::<u64>()
                .unwrap()
        })
    }

    fn module_bool(name: &str, attribute: &str) -> bool {
        Python::attach(|py| {
            pyo3::types::PyModule::import(py, name)
                .unwrap()
                .getattr(attribute)
                .unwrap()
                .extract::<bool>()
                .unwrap()
        })
    }

    fn with_threaded_executor_sentinel<F: FnOnce()>(body: F) -> usize {
        Python::attach(|py| {
            let module = pyo3::types::PyModule::import(py, "infinity_v3_python.threaded").unwrap();
            let original = module.getattr("ThreadedExecutor").unwrap();
            let source = CString::new(
                "CALLS = 0\nORIGINAL = None\ndef ThreadedExecutor(executor):\n    global CALLS\n    if getattr(executor, 'SENTINEL', False):\n        CALLS += 1\n        raise RuntimeError('threaded sentinel reached')\n    return ORIGINAL(executor)\n",
            )
            .unwrap();
            let sentinel = pyo3::types::PyModule::from_code(
                py,
                source.as_c_str(),
                c"_infinity_threaded_sentinel.py",
                c"_infinity_threaded_sentinel",
            )
            .unwrap();
            sentinel.setattr("ORIGINAL", original.clone()).unwrap();
            module
                .setattr(
                    "ThreadedExecutor",
                    sentinel.getattr("ThreadedExecutor").unwrap(),
                )
                .unwrap();
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(body));
            let calls = sentinel
                .getattr("CALLS")
                .unwrap()
                .extract::<usize>()
                .unwrap();
            module.setattr("ThreadedExecutor", original).unwrap();
            if let Err(payload) = outcome {
                std::panic::resume_unwind(payload);
            }
            calls
        })
    }

    fn received_serving_limits(name: &str) -> std::collections::BTreeMap<String, Option<i64>> {
        Python::attach(|py| {
            pyo3::types::PyModule::import(py, name)
                .unwrap()
                .getattr("RECEIVED")
                .unwrap()
                .extract::<std::collections::BTreeMap<String, Option<i64>>>()
                .unwrap()
        })
    }

    #[test]
    fn explicit_factory_receives_resolved_serving_limits_and_caps_batches() {
        let name = "_infinity_serving_limits_optin";
        install_module(
            name,
            &CAPABILITY_MODULE.replace(
                "CAPABILITIES",
                "{'protocol_version': 1, 'packed_limits': {'max_total_tokens': 10003, 'max_sequences': 64, 'max_seq_len': 2301}}",
            ),
        );
        let backend = PythonBackend::new(
            Path::new("."),
            "float32",
            ModelType::Embedding(Pool::Mean),
            0,
            python_config(
                name,
                Some(ServingLimits {
                    max_input_length: 2301,
                    max_batch_tokens: 10003,
                    max_batch_requests: Some(37),
                }),
            ),
        )
        .unwrap();
        let received = received_serving_limits(name);
        assert_eq!(received.len(), 3);
        assert_eq!(received.get("max_input_length"), Some(&Some(2301)));
        assert_eq!(received.get("max_batch_tokens"), Some(&Some(10003)));
        assert_eq!(received.get("max_batch_requests"), Some(&Some(37)));
        assert_eq!(backend.max_batch_size(), Some(64));
        backend.shutdown();

        let backend = PythonBackend::new(
            Path::new("."),
            "float32",
            ModelType::Embedding(Pool::Mean),
            0,
            python_config(
                name,
                Some(ServingLimits {
                    max_input_length: 2301,
                    max_batch_tokens: 10003,
                    max_batch_requests: None,
                }),
            ),
        )
        .unwrap();
        let received = received_serving_limits(name);
        assert_eq!(received.get("max_batch_requests"), Some(&None));
        backend.shutdown();
    }

    #[test]
    fn legacy_factories_with_kwargs_do_not_receive_serving_limits() {
        let name = "_infinity_legacy_factory";
        install_module(
            name,
            r#"
PROTOCOL_VERSION = 1
RECEIVED = None

class Executor:
    def submit(self, route, response, payload):
        response.set_exception(ValueError('not used'))
    def healthy(self):
        return True
    def close(self):
        pass

def create_executor(**kwargs):
    global RECEIVED
    RECEIVED = sorted(kwargs)
    return Executor()
"#,
        );
        let backend = PythonBackend::new(
            Path::new("."),
            "float32",
            ModelType::Embedding(Pool::Mean),
            0,
            python_config(
                name,
                Some(ServingLimits {
                    max_input_length: 2301,
                    max_batch_tokens: 10003,
                    max_batch_requests: Some(37),
                }),
            ),
        )
        .unwrap();
        let received = Python::attach(|py| {
            pyo3::types::PyModule::import(py, name)
                .unwrap()
                .getattr("RECEIVED")
                .unwrap()
                .extract::<Vec<String>>()
                .unwrap()
        });
        assert_eq!(
            received,
            vec![
                "config_json",
                "device_id",
                "dtype",
                "model_path",
                "model_type",
                "pooling",
            ]
        );
        assert_eq!(backend.max_batch_size(), None);
        backend.shutdown();
    }

    #[test]
    fn capabilities_absent_keep_legacy_behavior_with_serving_limits() {
        let backend = PythonBackend::new(
            Path::new("."),
            "float32",
            ModelType::Embedding(Pool::Mean),
            0,
            PythonBackendConfig {
                config: config().config,
                serving_limits: Some(ServingLimits {
                    max_input_length: 2301,
                    max_batch_tokens: 10003,
                    max_batch_requests: Some(37),
                }),
            },
        )
        .unwrap();
        assert_eq!(backend.max_batch_size(), None);
        backend.shutdown();
    }

    #[test]
    fn full_context_is_accepted_when_the_plugin_declares_it() {
        for (name, context) in [
            ("_infinity_full_context_32768", 32768),
            ("_infinity_full_context_65536", 65536),
        ] {
            install_module(
                name,
                &CAPABILITY_MODULE.replace(
                    "CAPABILITIES",
                    &format!(
                        "{{'protocol_version': 1, 'packed_limits': {{'max_total_tokens': {context}, 'max_sequences': 64, 'max_seq_len': {context}}}}}"
                    ),
                ),
            );
            let backend = PythonBackend::new(
                Path::new("."),
                "float32",
                ModelType::Embedding(Pool::Mean),
                0,
                python_config(
                    name,
                    Some(ServingLimits {
                        max_input_length: context,
                        max_batch_tokens: context,
                        max_batch_requests: None,
                    }),
                ),
            )
            .unwrap();
            assert_eq!(backend.max_batch_size(), Some(64));
            backend.shutdown();
        }
    }

    #[test]
    fn reachable_capacity_validation_matches_router_dispatch_bounds() {
        let narrow = PackedLimits {
            max_total_tokens: 4096,
            max_sequences: 32,
            max_seq_len: 128,
        };
        let bounded = ServingLimits {
            max_input_length: 128,
            max_batch_tokens: 16384,
            max_batch_requests: Some(32),
        };
        assert_eq!(reachable_total_tokens(bounded, narrow), 4096);
        assert!(validate_capacity(bounded, narrow).is_ok());

        let unbounded = ServingLimits {
            max_batch_requests: None,
            ..bounded
        };
        assert_eq!(effective_max_requests(unbounded, narrow), 32);
        assert_eq!(reachable_total_tokens(unbounded, narrow), 4096);
        assert!(validate_capacity(unbounded, narrow).is_ok());

        let odd = PackedLimits {
            max_total_tokens: 4191,
            max_sequences: 33,
            max_seq_len: 127,
        };
        let odd_serving = ServingLimits {
            max_input_length: 127,
            max_batch_tokens: 10003,
            max_batch_requests: None,
        };
        assert_eq!(reachable_total_tokens(odd_serving, odd), 4191);
        assert!(validate_capacity(odd_serving, odd).is_ok());
        assert!(validate_capacity(
            odd_serving,
            PackedLimits {
                max_total_tokens: 4190,
                ..odd
            }
        )
        .is_err());

        let seq_limited = PackedLimits {
            max_total_tokens: 65536,
            max_sequences: 64,
            max_seq_len: 8192,
        };
        assert!(validate_capacity(
            ServingLimits {
                max_input_length: 65536,
                max_batch_tokens: 65536,
                max_batch_requests: None,
            },
            seq_limited,
        )
        .is_err());

        let saturating = ServingLimits {
            max_input_length: usize::MAX,
            max_batch_tokens: usize::MAX,
            max_batch_requests: Some(usize::MAX),
        };
        let huge = PackedLimits {
            max_total_tokens: usize::MAX,
            max_sequences: usize::MAX,
            max_seq_len: usize::MAX,
        };
        assert_eq!(reachable_total_tokens(saturating, huge), usize::MAX);
        assert!(validate_capacity(saturating, huge).is_ok());
    }

    #[test]
    fn narrow_old_capacity_accepts_the_reachable_batch() {
        let name = "_infinity_narrow_old_capacity";
        install_module(
            name,
            &CAPABILITY_MODULE.replace(
                "CAPABILITIES",
                "{'protocol_version': 1, 'packed_limits': {'max_total_tokens': 4096, 'max_sequences': 32, 'max_seq_len': 128}}",
            ),
        );
        let backend = PythonBackend::new(
            Path::new("."),
            "float32",
            ModelType::Embedding(Pool::Mean),
            0,
            python_config(
                name,
                Some(ServingLimits {
                    max_input_length: 128,
                    max_batch_tokens: 16384,
                    max_batch_requests: None,
                }),
            ),
        )
        .unwrap();
        assert_eq!(backend.max_batch_size(), Some(32));
        backend.shutdown();
    }

    #[test]
    fn logical_capacity_is_not_replaced_by_a_physical_bucket() {
        let name = "_infinity_logical_capacity";
        install_module(
            name,
            &CAPABILITY_MODULE.replace(
                "CAPABILITIES",
                "{'protocol_version': 1, 'packed_limits': {'max_total_tokens': 10003, 'max_sequences': 32, 'max_seq_len': 2301, 'max_bucket_tokens': 10016}}",
            ),
        );
        let backend = PythonBackend::new(
            Path::new("."),
            "float32",
            ModelType::Embedding(Pool::Mean),
            0,
            python_config(
                name,
                Some(ServingLimits {
                    max_input_length: 2301,
                    max_batch_tokens: 10003,
                    max_batch_requests: Some(37),
                }),
            ),
        )
        .unwrap();
        assert_eq!(backend.max_batch_size(), Some(32));
        backend.shutdown();
    }

    #[test]
    fn rejected_initialization_closes_the_executor_once() {
        for (name, capabilities) in [
            (
                "_infinity_close_bad_capabilities",
                "{'packed_limits': {'max_total_tokens': True, 'max_sequences': 32, 'max_seq_len': 2301}}",
            ),
            (
                "_infinity_close_underprovisioned",
                "{'protocol_version': 1, 'packed_limits': {'max_total_tokens': 1536, 'max_sequences': 32, 'max_seq_len': 1536}}",
            ),
        ] {
            install_module(name, &LIFECYCLE_MODULE.replace("CAPABILITIES", capabilities));
            let result = PythonBackend::new(
                Path::new("."),
                "float32",
                ModelType::Embedding(Pool::Mean),
                0,
                python_config(
                    name,
                    Some(ServingLimits {
                        max_input_length: 2301,
                        max_batch_tokens: 10003,
                        max_batch_requests: Some(37),
                    }),
                ),
            );
            assert!(matches!(result, Err(BackendError::Start(_))));
            let close_count = Python::attach(|py| {
                pyo3::types::PyModule::import(py, name)
                    .unwrap()
                    .getattr("CLOSE_COUNT")
                    .unwrap()
                    .extract::<usize>()
                    .unwrap()
            });
            assert_eq!(close_count, 1, "{name}");
        }
    }

    #[test]
    fn synchronous_submit_runs_inline_on_caller_thread() {
        let name = "_infinity_sync_inline";
        install_module(
            name,
            &SYNC_MODULE.replace("SYNC_FIELD", "\"synchronous_submit\": True,"),
        );
        let backend = PythonBackend::new(
            Path::new("."),
            "float32",
            ModelType::Embedding(Pool::Mean),
            0,
            python_config(
                name,
                Some(ServingLimits {
                    max_input_length: 2301,
                    max_batch_tokens: 10003,
                    max_batch_requests: Some(37),
                }),
            ),
        )
        .unwrap();
        assert_eq!(backend.max_batch_size(), Some(32));
        let embeddings = backend.embed(batch()).unwrap();
        assert_eq!(embeddings.len(), 2);
        assert_eq!(
            module_u64(name, "SUBMIT_THREAD"),
            module_u64(name, "FACTORY_THREAD"),
            "synchronous submit must run on the caller/Rust worker thread"
        );
        assert!(
            module_bool(name, "INLINE_COMPLETION"),
            "the response envelope must be complete before submit returns"
        );
        assert_eq!(module_usize(name, "SUBMIT_CALLS"), 1);
        assert_eq!(module_usize(name, "CAPABILITY_CALLS"), 1);
        let received = received_serving_limits(name);
        assert_eq!(received.get("max_input_length"), Some(&Some(2301)));
        assert_eq!(received.get("max_batch_tokens"), Some(&Some(10003)));
        assert_eq!(received.get("max_batch_requests"), Some(&Some(37)));
        backend.shutdown();
        backend.shutdown();
        assert_eq!(module_usize(name, "CLOSE_COUNT"), 1);
    }

    #[test]
    #[serial]
    fn synchronous_opt_in_never_constructs_threaded_executor() {
        let name = "_infinity_sync_sentinel";
        install_module(
            name,
            &SYNC_MODULE.replace("SYNC_FIELD", "\"synchronous_submit\": True,"),
        );
        let calls = with_threaded_executor_sentinel(|| {
            let backend = PythonBackend::new(
                Path::new("."),
                "float32",
                ModelType::Embedding(Pool::Mean),
                0,
                python_config(name, None),
            )
            .unwrap();
            backend.embed(batch()).unwrap();
            backend.shutdown();
        });
        assert_eq!(calls, 0, "ThreadedExecutor must not be constructed");
    }

    #[test]
    #[serial]
    fn legacy_plugin_still_constructs_threaded_executor() {
        let name = "_infinity_legacy_sentinel";
        install_module(name, &SYNC_MODULE.replace("SYNC_FIELD", ""));
        let calls = with_threaded_executor_sentinel(|| {
            match PythonBackend::new(
                Path::new("."),
                "float32",
                ModelType::Embedding(Pool::Mean),
                0,
                python_config(name, None),
            ) {
                Err(BackendError::Start(message)) => {
                    assert!(message.contains("threaded sentinel reached"), "{message}");
                }
                Err(error) => panic!("expected a startup error, got {error}"),
                Ok(_) => panic!("the legacy wrapper sentinel must be reached"),
            }
        });
        assert_eq!(calls, 1, "legacy plugins must still use ThreadedExecutor");
    }

    #[test]
    fn legacy_plugin_without_sync_flag_uses_threaded_worker_thread() {
        let name = "_infinity_legacy_thread";
        install_module(name, &SYNC_MODULE.replace("SYNC_FIELD", ""));
        let backend = PythonBackend::new(
            Path::new("."),
            "float32",
            ModelType::Embedding(Pool::Mean),
            0,
            python_config(name, None),
        )
        .unwrap();
        let embeddings = backend.embed(batch()).unwrap();
        assert_eq!(embeddings.len(), 2);
        assert_ne!(
            module_u64(name, "SUBMIT_THREAD"),
            module_u64(name, "FACTORY_THREAD"),
            "legacy submit must run on the ThreadedExecutor worker thread"
        );
        backend.shutdown();
        backend.shutdown();
        assert_eq!(module_usize(name, "CLOSE_COUNT"), 1);
    }

    #[test]
    fn malformed_synchronous_submit_fails_startup_and_closes_once() {
        for (name, capabilities) in [
            ("_infinity_sync_bad_int", "{'synchronous_submit': 1}"),
            (
                "_infinity_sync_bad_string",
                "{'synchronous_submit': 'true'}",
            ),
            ("_infinity_sync_bad_none", "{'synchronous_submit': None}"),
        ] {
            install_module(
                name,
                &LIFECYCLE_MODULE.replace("CAPABILITIES", capabilities),
            );
            let result = PythonBackend::new(
                Path::new("."),
                "float32",
                ModelType::Embedding(Pool::Mean),
                0,
                python_config(name, None),
            );
            match result {
                Err(BackendError::Start(message)) => assert!(
                    message.contains("synchronous_submit must be a boolean"),
                    "{name}: {message}"
                ),
                Err(error) => panic!("{name}: expected a startup error, got {error}"),
                Ok(_) => panic!("{name}: malformed synchronous_submit must be rejected"),
            }
            assert_eq!(module_usize(name, "CLOSE_COUNT"), 1, "{name}");
        }
    }

    #[test]
    fn underprovisioned_plugin_capacity_fails_startup() {
        for (name, capabilities, serving, expected) in [
            (
                "_infinity_under_token",
                "{'protocol_version': 1, 'packed_limits': {'max_total_tokens': 1536, 'max_sequences': 64, 'max_seq_len': 1536}}",
                ServingLimits {
                    max_input_length: 2301,
                    max_batch_tokens: 10003,
                    max_batch_requests: Some(37),
                },
                "reachable_total_tokens=10003",
            ),
            (
                "_infinity_under_seq_32768",
                "{'protocol_version': 1, 'packed_limits': {'max_total_tokens': 32768, 'max_sequences': 64, 'max_seq_len': 8192}}",
                ServingLimits {
                    max_input_length: 32768,
                    max_batch_tokens: 32768,
                    max_batch_requests: None,
                },
                "max_input_length=32768 > plugin packed_limits.max_seq_len=8192",
            ),
            (
                "_infinity_under_seq_65536",
                "{'protocol_version': 1, 'packed_limits': {'max_total_tokens': 65536, 'max_sequences': 64, 'max_seq_len': 8192}}",
                ServingLimits {
                    max_input_length: 65536,
                    max_batch_tokens: 65536,
                    max_batch_requests: None,
                },
                "max_input_length=65536 > plugin packed_limits.max_seq_len=8192",
            ),
        ] {
            install_module(
                name,
                &CAPABILITY_MODULE.replace("CAPABILITIES", capabilities),
            );
            let result = PythonBackend::new(
                Path::new("."),
                "float32",
                ModelType::Embedding(Pool::Mean),
                0,
                python_config(name, Some(serving)),
            );
            match result {
                Err(BackendError::Start(message)) => {
                    assert!(message.contains(expected), "{name}: {message}");
                    assert!(
                        !message.contains("--max-input-length"),
                        "{name}: startup error names a non-existent CLI flag: {message}"
                    );
                    if expected.contains("max_seq_len") {
                        assert!(
                            message.contains("--max-sequence-length"),
                            "{name}: startup error should name the real CLI flag: {message}"
                        );
                    }
                }
                Err(error) => panic!("{name}: expected a startup error, got {error}"),
                Ok(_) => panic!("{name}: underprovisioned capacity must fail startup"),
            }
        }
    }

    #[test]
    fn malformed_capabilities_fail_startup() {
        for (index, (capabilities, expected)) in [
            ("{'packed_limits': True}", "packed_limits must be a mapping"),
            ("{'packed_limits': None}", "packed_limits must be a mapping"),
            (
                "{'protocol_version': 2, 'packed_limits': {'max_total_tokens': 10003, 'max_sequences': 64, 'max_seq_len': 2301}}",
                "protocol 2",
            ),
            (
                "{'packed_limits': {'max_sequences': 64, 'max_seq_len': 2301}}",
                "max_total_tokens is required",
            ),
            (
                "{'packed_limits': {'max_total_tokens': True, 'max_sequences': 64, 'max_seq_len': 2301}}",
                "max_total_tokens must be an integer, not a boolean",
            ),
            (
                "{'packed_limits': {'max_total_tokens': 10003.0, 'max_sequences': 64, 'max_seq_len': 2301}}",
                "max_total_tokens must be an integer",
            ),
            (
                "{'packed_limits': {'max_total_tokens': 10003, 'max_sequences': '64', 'max_seq_len': 2301}}",
                "max_sequences must be an integer",
            ),
            (
                "{'packed_limits': {'max_total_tokens': 10003, 'max_sequences': 64, 'max_seq_len': 0}}",
                "max_seq_len must be a positive integer",
            ),
        ]
        .into_iter()
        .enumerate()
        {
            let name = format!("_infinity_bad_capabilities_{index}");
            install_module(
                &name,
                &CAPABILITY_MODULE.replace("CAPABILITIES", capabilities),
            );
            let result = PythonBackend::new(
                Path::new("."),
                "float32",
                ModelType::Embedding(Pool::Mean),
                0,
                python_config(&name, Some(ServingLimits {
                    max_input_length: 2301,
                    max_batch_tokens: 10003,
                    max_batch_requests: Some(37),
                })),
            );
            match result {
                Err(BackendError::Start(message)) => {
                    assert!(message.contains(expected), "{capabilities}: {message}");
                }
                Err(error) => panic!("{capabilities}: expected a startup error, got {error}"),
                Ok(_) => panic!("{capabilities} must be rejected"),
            }
        }
    }

    #[test]
    fn capabilities_must_be_callable_and_return_a_mapping() {
        for (name, body, expected) in [
            (
                "_infinity_capabilities_not_callable",
                "    capabilities = 1\n",
                "capabilities must be callable",
            ),
            (
                "_infinity_capabilities_not_mapping",
                "    def capabilities(self):\n        return []\n",
                "capabilities must return a mapping",
            ),
        ] {
            install_module(
                name,
                &format!(
                    r#"
PROTOCOL_VERSION = 1
class Executor:
{body}    def submit(self, route, response, payload):
        response.set_exception(ValueError('not used'))
    def healthy(self):
        return True
    def close(self):
        pass
def create_executor(**kwargs):
    return Executor()
"#
                ),
            );
            let result = PythonBackend::new(
                Path::new("."),
                "float32",
                ModelType::Embedding(Pool::Mean),
                0,
                python_config(name, None),
            );
            match result {
                Err(BackendError::Start(message)) => {
                    assert!(message.contains(expected), "{name}: {message}");
                }
                Err(error) => panic!("{name}: expected a startup error, got {error}"),
                Ok(_) => panic!("{name} must be rejected"),
            }
        }
    }

    #[test]
    fn plugin_submit_error_completes_and_releases_admission() {
        install_module(
            "_infinity_submit_error_test",
            r#"
PROTOCOL_VERSION = 1
import json

class Executor:
    def __init__(self, memory):
        self.memory = memory
    def submit(self, route, response, payload):
        self.last_response = response
        if self.memory:
            response.set_exception(MemoryError("result budget exhausted"))
        else:
            raise RuntimeError("queue is full")
    def healthy(self):
        return True
    def close(self):
        pass

def create_executor(**kwargs):
    return Executor(json.loads(kwargs["config_json"])["memory"])
"#,
        );
        for (memory, expected) in [(false, "queue is full"), (true, "result budget exhausted")] {
            let config = serde_json::json!({
                "backend": "_infinity_submit_error_test:create_executor",
                "memory": memory,
            });
            let backend = PythonBackend::new(
                Path::new("."),
                "float32",
                ModelType::Embedding(Pool::Mean),
                0,
                PythonBackendConfig {
                    config: Some(config.to_string()),
                    serving_limits: None,
                },
            )
            .unwrap();
            for _ in 0..2 {
                let result = backend.submit_embed(batch()).unwrap().complete();
                assert!(matches!(result, Err(BackendError::Overloaded(message))
                if message.contains(expected)));
                assert_eq!(backend.admission.available_permits(), 1);
            }
            backend.shutdown();
        }
    }

    #[test]
    fn capability_discovery_uses_defaults_and_plugin_hook() {
        let default = PythonBackend::discover_capabilities(&config()).unwrap();
        assert_eq!(default, (1, false));
        install_module(
            "_infinity_capability_test",
            r#"
PROTOCOL_VERSION = 1
import json

def create_executor(**kwargs):
    pass

def capabilities(config_json):
    config = json.loads(config_json)
    return {
        "visible_device_count": config.get("devices", 1),
        "supports_replication": config.get("replicate", False),
    }
"#,
        );
        let discovered = PythonBackend::discover_capabilities(&PythonBackendConfig {
            config: Some(
                r#"{"backend":"_infinity_capability_test:create_executor","devices":3,"replicate":true}"#
                    .into(),
            ),
            serving_limits: None,
        })
        .unwrap();
        assert_eq!(discovered, (3, true));
        for invalid in ["0", "-1", "true"] {
            assert!(PythonBackend::discover_capabilities(&PythonBackendConfig {
                config: Some(format!(r#"{{"backend":"_infinity_capability_test:create_executor","devices":{invalid}}}"#)),
                serving_limits: None,
            }).is_err());
        }
        assert!(PythonBackend::discover_capabilities(&PythonBackendConfig {
            config: Some(
                r#"{"backend":"_infinity_capability_test:create_executor","replicate":"yes"}"#
                    .into()
            ),
            serving_limits: None,
        })
        .is_err());
    }

    #[test]
    fn startup_negotiates_capabilities_before_dispatch_wrapping() {
        let name = "_infinity_wrapped_capabilities";
        install_module(
            name,
            r#"
PROTOCOL_VERSION = 1
CAPABILITY_CALLS = 0
RECEIVED = None

class Executor:
    def capabilities(self):
        global CAPABILITY_CALLS
        CAPABILITY_CALLS += 1
        return {
            "protocol_version": 1,
            "packed_limits": {
                "max_total_tokens": 8192,
                "max_sequences": 16,
                "max_seq_len": 2048,
            },
        }
    def submit(self, route, response, payload):
        response.set_exception(ValueError('not used'))
    def healthy(self):
        return True
    def close(self):
        pass

def capabilities(config_json):
    return {"visible_device_count": 3, "supports_replication": True}

def create_executor(*, serving_limits=None, **kwargs):
    global RECEIVED
    RECEIVED = dict(serving_limits)
    return Executor()
"#,
        );
        let config = python_config(
            name,
            Some(ServingLimits {
                max_input_length: 2048,
                max_batch_tokens: 8192,
                max_batch_requests: Some(4),
            }),
        );
        assert_eq!(
            PythonBackend::discover_capabilities(&config).unwrap(),
            (3, true)
        );
        let backend = PythonBackend::new(
            Path::new("."),
            "float32",
            ModelType::Embedding(Pool::Mean),
            0,
            config,
        )
        .unwrap();
        let received = received_serving_limits(name);
        assert_eq!(received.get("max_input_length"), Some(&Some(2048)));
        assert_eq!(received.get("max_batch_tokens"), Some(&Some(8192)));
        assert_eq!(received.get("max_batch_requests"), Some(&Some(4)));
        assert_eq!(backend.max_batch_size(), Some(16));
        let calls = Python::attach(|py| {
            pyo3::types::PyModule::import(py, name)
                .unwrap()
                .getattr("CAPABILITY_CALLS")
                .unwrap()
                .extract::<usize>()
                .unwrap()
        });
        assert_eq!(calls, 1, "capabilities must be read once during startup");
        backend.shutdown();
    }
}
