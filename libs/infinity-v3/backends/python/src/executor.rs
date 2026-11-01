use crate::envelope::ResponseEnvelope;
use infinity_v3_backend_core::BackendError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
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
}

pub(crate) struct Executor {
    inner: Py<PyAny>,
    closed: AtomicBool,
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
