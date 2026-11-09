use crate::envelope::ResponseEnvelope;
use crate::ServingLimits;
use infinity_v3_backend_core::BackendError;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;

pub(crate) struct ExecutorConfig {
    pub selector: Option<String>,
    pub model_path: PathBuf,
    pub dtype: String,
    pub model_type: String,
    pub pooling: String,
    pub device_id: usize,
    pub config_json: String,
    pub serving_limits: ServingLimits,
}

/// Factory completion is separate from inference's numpy response envelope.
#[pyclass]
struct StartupEnvelope {
    sender: Option<mpsc::Sender<PyResult<()>>>,
}

impl StartupEnvelope {
    fn complete(&mut self, result: PyResult<()>) -> PyResult<()> {
        let sender = self
            .sender
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("startup response already completed"))?;
        let _ = sender.send(result);
        Ok(())
    }
}

#[pymethods]
impl StartupEnvelope {
    fn set_result(&mut self, _value: Bound<'_, PyAny>) -> PyResult<()> {
        self.complete(Ok(()))
    }

    fn set_exception(&mut self, error: Bound<'_, PyAny>) -> PyResult<()> {
        self.complete(Err(PyErr::from_value(error)))
    }
}

pub(crate) struct Executor {
    inner: Py<PyAny>,
    closed: AtomicBool,
}

impl Executor {
    pub(crate) fn new(config: ExecutorConfig) -> Result<Self, BackendError> {
        Python::initialize();
        let (inner, receiver) = Python::attach(|py| {
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
            let value = config.serving_limits;
            let limits = PyDict::new(py);
            limits
                .set_item("max_input_length", value.max_input_length)
                .map_err(start_error)?;
            limits
                .set_item("max_batch_tokens", value.max_batch_tokens)
                .map_err(start_error)?;
            limits
                .set_item("max_batch_requests", value.max_batch_requests)
                .map_err(start_error)?;
            kwargs
                .set_item("serving_limits", limits)
                .map_err(start_error)?;
            let (sender, receiver) = mpsc::channel();
            let ready = Py::new(
                py,
                StartupEnvelope {
                    sender: Some(sender),
                },
            )
            .map_err(start_error)?;
            let inner = PyModule::import(py, "infinity_v3_python.threaded")
                .and_then(|module| module.getattr("ThreadedExecutor"))
                .and_then(|executor| executor.call1((factory, kwargs, ready)))
                .map_err(|error| {
                    BackendError::Start(format!(
                        "Could not start Python backend dispatch worker: {error}"
                    ))
                })?
                .unbind();
            Ok::<_, BackendError>((inner, receiver))
        })?;
        // Own the wrapper before waiting so failed factories also shut down
        // their pool. Explicitly release the GIL even if our caller is attached.
        let executor = Self {
            inner,
            closed: AtomicBool::new(false),
        };
        Python::attach(|py| py.detach(move || receiver.recv()))
            .map_err(|_| BackendError::Start("Python factory dropped its startup response".into()))?
            .map_err(|error| {
                BackendError::Start(format!("Could not start Python backend executor: {error}"))
            })?;
        Ok(executor)
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
            if let Err(error) = self.inner.bind(py).call_method0("close") {
                tracing::error!(%error, "Python backend cleanup failed");
            }
        });
    }
}

impl Drop for Executor {
    fn drop(&mut self) {
        self.close();
    }
}

pub(crate) fn discover_capabilities(
    selector: Option<String>,
    config_json: &str,
) -> Result<(usize, bool), BackendError> {
    Python::initialize();
    Python::attach(|py| {
        add_python_paths(py);
        let discovery = PyModule::import(py, "infinity_v3_python.discovery").map_err(|error| {
            BackendError::Start(format!(
                "Could not import infinity_v3_python for capability discovery: {error}"
            ))
        })?;
        let Ok(load_capabilities) = discovery.getattr("load_capabilities") else {
            return Ok((1, false));
        };
        let capabilities = load_capabilities
            .call1((selector, config_json))
            .map_err(|error| {
                BackendError::Start(format!(
                    "Could not discover Python backend capabilities: {error}"
                ))
            })?;
        if capabilities.is_none() {
            return Ok((1, false));
        }
        capabilities.extract::<(usize, bool)>().map_err(|error| {
            BackendError::Start(format!("Python backend capabilities are invalid: {error}"))
        })
    })
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use std::time::Duration;

    fn install(name: &str, source: &str) {
        let name = CString::new(name).unwrap();
        let source = CString::new(source).unwrap();
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::from_code(py, &source, c"factory_test.py", &name).unwrap();
            PyModule::import(py, "sys")
                .unwrap()
                .getattr("modules")
                .unwrap()
                .set_item(name.to_str().unwrap(), module)
                .unwrap();
        });
    }

    fn config(selector: &str, device: usize) -> ExecutorConfig {
        ExecutorConfig {
            selector: Some(selector.into()),
            model_path: PathBuf::from("."),
            dtype: "bfloat16".into(),
            model_type: "embedding".into(),
            pooling: "mean".into(),
            device_id: device,
            config_json: "{}".into(),
            serving_limits: ServingLimits::new(8192, 512, Some(16)).unwrap(),
        }
    }

    #[test]
    #[serial_test::serial]
    fn eight_factories_complete_without_the_waiter_holding_the_gil() {
        install(
            "_infinity_factory_threads",
            r#"
import threading
PROTOCOL_VERSION = 1
barrier = threading.Barrier(8, timeout=5)
state = threading.local()
made = {}
closed = {}
class Plugin:
    def __init__(self):
        self.device = state.device
    def healthy(self):
        return state.device == self.device and made[self.device] == threading.get_ident()
    def close(self):
        assert state.device == self.device
        closed[self.device] = threading.get_ident()
def create_executor(**kwargs):
    assert kwargs['serving_limits'] == dict(max_input_length=512, max_batch_tokens=8192, max_batch_requests=16)
    state.device = kwargs['device_id']
    made[state.device] = threading.get_ident()
    barrier.wait()
    return Plugin()
"#,
        );
        let (done, results) = mpsc::channel();
        let handles = (0..8)
            .map(|device| {
                let done = done.clone();
                std::thread::spawn(move || {
                    let result =
                        Executor::new(config("_infinity_factory_threads:create_executor", device))
                            .and_then(|executor| executor.health());
                    let _ = done.send(result.map_err(|error| error.to_string()));
                })
            })
            .collect::<Vec<_>>();
        for _ in 0..8 {
            results
                .recv_timeout(Duration::from_secs(10))
                .unwrap()
                .unwrap();
        }
        for handle in handles {
            handle.join().unwrap();
        }
        Python::attach(|py| {
            let module = PyModule::import(py, "_infinity_factory_threads").unwrap();
            let made = module.getattr("made").unwrap();
            let closed = module.getattr("closed").unwrap();
            assert_eq!(made.len().unwrap(), 8);
            assert!(made.eq(closed).unwrap());
        });
    }

    #[test]
    #[serial_test::serial]
    fn factory_error_reaches_rust_and_stops_the_startup_thread() {
        install(
            "_infinity_factory_error",
            r#"
import threading
PROTOCOL_VERSION = 1
worker = None
def create_executor(**kwargs):
    global worker
    worker = threading.current_thread()
    raise ValueError('factory initialization exploded')
"#,
        );
        let error = Python::attach(|_| {
            Executor::new(config("_infinity_factory_error:create_executor", 0))
                .err()
                .expect("factory must fail")
        });
        assert!(
            matches!(error, BackendError::Start(message) if message.contains("factory initialization exploded"))
        );
        Python::attach(|py| {
            let module = PyModule::import(py, "_infinity_factory_error").unwrap();
            assert!(!module
                .getattr("worker")
                .unwrap()
                .call_method0("is_alive")
                .unwrap()
                .extract::<bool>()
                .unwrap());
        });
    }
}
