use crate::envelope::ResponseEnvelope;
use crate::ServingLimits;
use infinity_v3_backend_core::BackendError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyInt, PyModule};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

pub(crate) struct ExecutorConfig {
    pub selector: Option<String>,
    pub model_path: PathBuf,
    pub dtype: String,
    pub model_type: String,
    pub pooling: String,
    pub device_id: usize,
    pub config_json: String,
    pub queue_size: usize,
    pub serving_limits: Option<ServingLimits>,
}

pub(crate) struct Executor {
    inner: Py<PyAny>,
    closed: AtomicBool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PackedLimits {
    pub max_total_tokens: usize,
    pub max_sequences: usize,
    pub max_seq_len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BackendCapabilities {
    pub packed_limits: Option<PackedLimits>,
}

impl Executor {
    pub(crate) fn new(config: ExecutorConfig) -> Result<Self, BackendError> {
        Python::initialize();
        Python::attach(|py| {
            add_python_paths(py);
            let factory = load_factory(py, config.selector)?;
            let kwargs = PyDict::new(py);
            kwargs
                .set_item(
                    "model_path",
                    config.model_path.to_string_lossy().into_owned(),
                )
                .map_err(start_error)?;
            kwargs
                .set_item("dtype", config.dtype)
                .map_err(start_error)?;
            kwargs
                .set_item("model_type", config.model_type)
                .map_err(start_error)?;
            kwargs
                .set_item("pooling", config.pooling)
                .map_err(start_error)?;
            kwargs
                .set_item("device_id", config.device_id)
                .map_err(start_error)?;
            kwargs
                .set_item("config_json", config.config_json)
                .map_err(start_error)?;
            kwargs
                .set_item("queue_size", config.queue_size)
                .map_err(start_error)?;
            if let Some(serving_limits) = config.serving_limits {
                if factory_accepts_serving_limits(py, &factory) {
                    let limits = PyDict::new(py);
                    limits
                        .set_item("max_input_length", serving_limits.max_input_length)
                        .map_err(start_error)?;
                    limits
                        .set_item("max_batch_tokens", serving_limits.max_batch_tokens)
                        .map_err(start_error)?;
                    match serving_limits.max_batch_requests {
                        Some(value) => limits.set_item("max_batch_requests", value),
                        None => limits.set_item("max_batch_requests", py.None()),
                    }
                    .map_err(start_error)?;
                    kwargs
                        .set_item("serving_limits", limits)
                        .map_err(start_error)?;
                }
            }
            let inner = factory
                .call((), Some(&kwargs))
                .map_err(|error| {
                    BackendError::Start(format!("Could not start Python backend executor: {error}"))
                })?
                .unbind();
            Ok(Self {
                inner,
                closed: AtomicBool::new(false),
            })
        })
    }

    pub(crate) fn submit(
        &self,
        py: Python<'_>,
        route: &str,
        envelope: Py<ResponseEnvelope>,
        payload: Bound<'_, PyDict>,
    ) -> Result<(), BackendError> {
        if self.closed.load(Ordering::Acquire) {
            return Err(BackendError::Unhealthy);
        }
        self.inner
            .bind(py)
            .call_method1("submit", (route, envelope, payload))
            .map(|_| ())
            .map_err(submit_error)
    }

    /// Optional capability negotiation. `None` means the executor has no
    /// `capabilities` method and keeps the legacy behavior.
    pub(crate) fn capabilities(&self) -> Result<Option<BackendCapabilities>, BackendError> {
        if self.closed.load(Ordering::Acquire) {
            return Err(BackendError::Unhealthy);
        }
        Python::attach(|py| {
            let inner = self.inner.bind(py);
            let capabilities = match inner.getattr("capabilities") {
                Ok(value) => value,
                Err(error) if error.is_instance_of::<pyo3::exceptions::PyAttributeError>(py) => {
                    return Ok(None);
                }
                Err(error) => {
                    return Err(BackendError::Start(format!(
                        "Python backend capabilities could not be read: {error}"
                    )));
                }
            };
            if !capabilities.is_callable() {
                return Err(BackendError::Start(
                    "Python backend capabilities must be callable".into(),
                ));
            }
            let capabilities = capabilities.call0().map_err(|error| {
                BackendError::Start(format!("Python backend capabilities failed: {error}"))
            })?;
            let capabilities = capabilities
                .extract::<Bound<'_, PyDict>>()
                .map_err(|error| {
                    BackendError::Start(format!(
                        "Python backend capabilities must return a mapping: {error}"
                    ))
                })?;
            if let Some(version) = capabilities
                .get_item("protocol_version")
                .map_err(capability_error)?
            {
                if version.is_instance_of::<PyBool>() {
                    return Err(BackendError::Start(
                        "Python backend protocol_version must be an integer".into(),
                    ));
                }
                let version = version.extract::<i64>().map_err(capability_error)?;
                if version != 1 {
                    return Err(BackendError::Start(format!(
                        "Python backend uses protocol {version}; this transport supports \
                         protocol 1"
                    )));
                }
            }
            Ok(Some(BackendCapabilities {
                packed_limits: packed_limits(&capabilities)?,
            }))
        })
    }

    pub(crate) fn health(&self) -> Result<(), BackendError> {
        if self.closed.load(Ordering::Acquire) {
            return Err(BackendError::Unhealthy);
        }
        Python::attach(|py| {
            let healthy = self
                .inner
                .bind(py)
                .call_method0("healthy")
                .and_then(|value| value.extract::<bool>())
                .map_err(|error| {
                    BackendError::Inference(format!("Python backend health check failed: {error}"))
                })?;
            if healthy {
                Ok(())
            } else {
                Err(BackendError::Unhealthy)
            }
        })
    }

    pub(crate) fn close(&self) {
        if self.closed.swap(true, Ordering::AcqRel) {
            return;
        }
        let _ = Python::try_attach(|py| {
            let _ = self.inner.bind(py).call_method0("close");
        });
    }
}

impl Drop for Executor {
    fn drop(&mut self) {
        self.close();
    }
}

fn load_factory<'py>(
    py: Python<'py>,
    selector: Option<String>,
) -> Result<Bound<'py, PyAny>, BackendError> {
    let discovery = PyModule::import(py, "infinity_v3_python.discovery").map_err(|error| {
        BackendError::Start(format!(
            "Could not import infinity_v3_python; install the Python backend package or set INFINITY_PYTHONPATH: {error}"
        ))
    })?;
    discovery
        .getattr("load_factory")
        .and_then(|function| function.call1((selector,)))
        .map_err(|error| BackendError::Start(format!("Could not load Python backend: {error}")))
}

fn factory_accepts_serving_limits(py: Python<'_>, factory: &Bound<'_, PyAny>) -> bool {
    let Ok(discovery) = PyModule::import(py, "infinity_v3_python.discovery") else {
        return false;
    };
    let Ok(helper) = discovery.getattr("factory_accepts_serving_limits") else {
        return false;
    };
    helper
        .call1((factory.clone(),))
        .ok()
        .and_then(|accepted| accepted.extract::<bool>().ok())
        .unwrap_or(false)
}

fn packed_limits(capabilities: &Bound<'_, PyDict>) -> Result<Option<PackedLimits>, BackendError> {
    let Some(limits) = capabilities
        .get_item("packed_limits")
        .map_err(capability_error)?
    else {
        return Ok(None);
    };
    if limits.is_instance_of::<PyBool>() {
        return Err(BackendError::Start(
            "Python backend capabilities.packed_limits must be a mapping".into(),
        ));
    }
    let limits = limits.extract::<Bound<'_, PyDict>>().map_err(|error| {
        BackendError::Start(format!(
            "Python backend capabilities.packed_limits must be a mapping: {error}"
        ))
    })?;
    Ok(Some(PackedLimits {
        max_total_tokens: positive_limit(&limits, "max_total_tokens")?,
        max_sequences: positive_limit(&limits, "max_sequences")?,
        max_seq_len: positive_limit(&limits, "max_seq_len")?,
    }))
}

fn positive_limit(limits: &Bound<'_, PyDict>, name: &str) -> Result<usize, BackendError> {
    let Some(value) = limits.get_item(name).map_err(capability_error)? else {
        return Err(BackendError::Start(format!(
            "Python backend capabilities.packed_limits.{name} is required"
        )));
    };
    if value.is_instance_of::<PyBool>() {
        return Err(BackendError::Start(format!(
            "Python backend capabilities.packed_limits.{name} must be an integer, not a boolean"
        )));
    }
    if !value.is_instance_of::<PyInt>() {
        return Err(BackendError::Start(format!(
            "Python backend capabilities.packed_limits.{name} must be an integer"
        )));
    }
    let value = value.extract::<i64>().map_err(capability_error)?;
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| {
            BackendError::Start(format!(
                "Python backend capabilities.packed_limits.{name} must be a positive integer"
            ))
        })
}

fn capability_error(error: PyErr) -> BackendError {
    BackendError::Start(format!(
        "Python backend capabilities returned an invalid value: {error}"
    ))
}

fn add_python_paths(py: Python<'_>) {
    let mut paths = Vec::new();
    if let Ok(path) = std::env::var("INFINITY_PYTHONPATH") {
        paths.extend(std::env::split_paths(&path));
    }
    paths.push(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../python"));
    if let Ok(executable) = std::env::current_exe() {
        if let Some(directory) = executable.parent() {
            paths.push(directory.join("python"));
        }
    }
    let Ok(sys) = PyModule::import(py, "sys") else {
        return;
    };
    let Ok(path_list) = sys.getattr("path") else {
        return;
    };
    for path in paths.into_iter().rev() {
        if !path.is_dir() {
            continue;
        }
        let _ = path_list.call_method1("insert", (0, path.to_string_lossy().into_owned()));
    }
}

fn start_error(error: PyErr) -> BackendError {
    BackendError::Start(format!(
        "Could not configure Python backend executor: {error}"
    ))
}

fn submit_error(error: PyErr) -> BackendError {
    let message = error.to_string();
    if Python::attach(|py| error.is_instance_of::<pyo3::exceptions::PyMemoryError>(py))
        || message.contains("queue is full")
    {
        BackendError::Overloaded(message)
    } else if message.contains("executor is closed") {
        BackendError::Unhealthy
    } else {
        BackendError::Inference(format!("Python backend submit failed: {message}"))
    }
}
