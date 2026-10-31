use numpy::{PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use tokio::sync::{oneshot, OwnedSemaphorePermit};

/// A forward result stays in Python until the Rust completion thread asks for it.
/// The array must be float32 and C-contiguous so the response path can borrow it
/// without silently changing dtype or materializing a strided copy.
pub(crate) type ForwardResult = Result<Py<PyArrayDyn<f32>>, PyErr>;

/// Bridge used by the Python executor to complete a deferred Rust submission.
#[pyclass]
pub(crate) struct ResponseEnvelope {
    sender: Option<oneshot::Sender<ForwardResult>>,
    _lease: Option<OwnedSemaphorePermit>,
}

impl ResponseEnvelope {
    pub(crate) fn channel(
        py: Python<'_>,
        lease: Option<OwnedSemaphorePermit>,
    ) -> PyResult<(Py<Self>, oneshot::Receiver<ForwardResult>)> {
        let (sender, receiver) = oneshot::channel();
        let envelope = Py::new(
            py,
            Self {
                sender: Some(sender),
                _lease: lease,
            },
        )?;
        Ok((envelope, receiver))
    }

    fn complete(&mut self, result: ForwardResult) -> PyResult<()> {
        let sender = self
            .sender
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("response already completed"))?;
        let _ = sender.send(result);
        Ok(())
    }
}

#[pymethods]
impl ResponseEnvelope {
    fn set_result(&mut self, result: Bound<'_, PyArrayDyn<f32>>) -> PyResult<()> {
        if !result.is_c_contiguous() {
            return Err(PyRuntimeError::new_err(
                "Python backend response array must be C-contiguous",
            ));
        }
        if let Ok(array) = result.try_readwrite() {
            let _ = array.make_nonwriteable();
        }
        self.complete(Ok(result.unbind()))
    }

    fn set_exception(&mut self, error: Bound<'_, PyAny>) -> PyResult<()> {
        let error = PyErr::from_value(error);
        self.complete(Err(error))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::PyArray1;
    use pyo3::types::PyDict;
    use std::sync::Arc;
    use tokio::sync::Semaphore;

    fn contiguous_array<'py>(py: Python<'py>) -> Bound<'py, PyArrayDyn<f32>> {
        PyArray1::<f32>::from_slice(py, &[1.0, 2.0])
            .into_any()
            .extract()
            .unwrap()
    }

    #[test]
    fn envelope_is_single_use_and_releases_its_permit() {
        Python::initialize();
        Python::attach(|py| {
            let semaphore = Arc::new(Semaphore::new(1));
            let permit = semaphore.clone().try_acquire_owned().unwrap();
            let (envelope, receiver) = ResponseEnvelope::channel(py, Some(permit)).unwrap();
            envelope
                .borrow_mut(py)
                .set_result(contiguous_array(py))
                .unwrap();
            assert_eq!(
                receiver.blocking_recv().unwrap().unwrap().bind(py).shape(),
                &[2]
            );
            assert!(envelope
                .borrow_mut(py)
                .set_result(contiguous_array(py))
                .is_err());
            drop(envelope);
            assert_eq!(semaphore.available_permits(), 1);
        });
    }

    #[test]
    fn envelope_rejects_non_contiguous_arrays() {
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(
                c"import numpy as np
arr = np.arange(6, dtype=np.float32).reshape(2, 3)[:, ::2]",
                None,
                Some(&locals),
            )
            .unwrap();
            let strided = locals
                .get_item("arr")
                .unwrap()
                .unwrap()
                .extract::<Bound<'_, PyArrayDyn<f32>>>()
                .unwrap();
            let (envelope, _receiver) = ResponseEnvelope::channel(py, None).unwrap();
            assert!(envelope.borrow_mut(py).set_result(strided).is_err());
        });
    }

    #[test]
    fn exception_path_reaches_the_rust_receiver() {
        Python::initialize();
        Python::attach(|py| {
            let (envelope, receiver) = ResponseEnvelope::channel(py, None).unwrap();
            let error = pyo3::exceptions::PyValueError::new_err("boom");
            envelope
                .borrow_mut(py)
                .set_exception(error.value(py).clone().into_any())
                .unwrap();
            let result = receiver.blocking_recv().unwrap();
            assert!(result.is_err());
        });
    }

    #[test]
    fn canceled_receiver_does_not_fail_python_completion() {
        Python::initialize();
        Python::attach(|py| {
            let semaphore = Arc::new(Semaphore::new(1));
            let permit = semaphore.clone().try_acquire_owned().unwrap();
            let (envelope, receiver) = ResponseEnvelope::channel(py, Some(permit)).unwrap();
            drop(receiver);
            envelope
                .borrow_mut(py)
                .set_result(contiguous_array(py))
                .unwrap();
            drop(envelope);
            assert_eq!(semaphore.available_permits(), 1);
        });
    }
}
