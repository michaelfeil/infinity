use infinity_v3_backend_core::result_memory::{self, ResultLease, ResultPool};
use numpy::{PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyMemoryError, PyRuntimeError};
use pyo3::prelude::*;
use std::sync::Arc;
use tokio::sync::{oneshot, OwnedSemaphorePermit};

/// A forward result stays in Rust until the completion thread consumes it.
/// The snapshot owns its float32 values so later mutation through a numpy alias
/// cannot change a response that has already been accepted.
pub(crate) struct OutputSnapshot {
    pub values: Vec<f32>,
    pub rows: usize,
    pub width: usize,
    pub memory: ResultLease,
}

pub(crate) type ForwardResult = Result<OutputSnapshot, PyErr>;

/// Bridge used by the Python executor to complete a deferred Rust submission.
#[pyclass]
pub(crate) struct ResponseEnvelope {
    sender: Option<oneshot::Sender<ForwardResult>>,
    lease: Option<OwnedSemaphorePermit>,
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
                lease,
            },
        )?;
        Ok((envelope, receiver))
    }

    fn complete(&mut self, result: ForwardResult) -> PyResult<()> {
        let sender = self
            .sender
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("response already completed"))?;
        // The admission permit must not outlive completion: the Rust receiver
        // may be waiting for it before it wakes the deferred request.
        self.lease = None;
        let _ = sender.send(result);
        Ok(())
    }
}

/// Copy a Python-owned numpy result into memory owned by Rust. Validation and
/// the result reservation happen before allocation so an invalid or oversized
/// result leaves the envelope available for `set_exception`.
fn snapshot_result(
    result: Bound<'_, PyArrayDyn<f32>>,
    pool: &Arc<ResultPool>,
) -> PyResult<OutputSnapshot> {
    let shape = result.shape();
    let [rows, width] = shape else {
        return Err(PyRuntimeError::new_err(format!(
            "Python backend response array must be rank 2; got rank {}",
            shape.len()
        )));
    };
    let (rows, width) = (*rows, *width);
    if width == 0 {
        return Err(PyRuntimeError::new_err(
            "Python backend response array must have a nonzero width",
        ));
    }
    if !result.is_c_contiguous() {
        return Err(PyRuntimeError::new_err(
            "Python backend response array must be C-contiguous",
        ));
    }
    let elements = rows
        .checked_mul(width)
        .ok_or_else(|| PyRuntimeError::new_err("Python backend response array size overflow"))?;
    let bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            PyRuntimeError::new_err("Python backend response array byte size overflow")
        })?;
    let memory = pool.reserve(bytes).map_err(|error| {
        PyMemoryError::new_err(format!(
            "Python backend result memory could not be reserved: {error}"
        ))
    })?;
    let source = result.try_readonly().map_err(|error| {
        PyRuntimeError::new_err(format!(
            "Python backend response array is not readable: {error}"
        ))
    })?;
    let source = source.as_slice().map_err(|error| {
        PyRuntimeError::new_err(format!(
            "Python backend response array is not contiguous: {error}"
        ))
    })?;
    if source.len() != elements {
        return Err(PyRuntimeError::new_err(
            "Python backend response array length does not match its shape",
        ));
    }
    let mut values = Vec::new();
    values.try_reserve_exact(elements).map_err(|error| {
        PyMemoryError::new_err(format!(
            "Python backend result buffer allocation failed: {error}"
        ))
    })?;
    values.extend_from_slice(source);
    Ok(OutputSnapshot {
        values,
        rows,
        width,
        memory,
    })
}

#[pymethods]
impl ResponseEnvelope {
    fn set_result(&mut self, result: Bound<'_, PyArrayDyn<f32>>) -> PyResult<()> {
        if self.sender.is_none() {
            return Err(PyRuntimeError::new_err("response already completed"));
        }
        let pool = result_memory::global_pool().map_err(|error| {
            PyRuntimeError::new_err(format!(
                "Python backend result memory is unavailable: {error}"
            ))
        })?;
        let snapshot = snapshot_result(result, &pool)?;
        self.complete(Ok(snapshot))
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

    fn contiguous_array<'py>(
        py: Python<'py>,
        values: &[f32],
        rows: usize,
        width: usize,
    ) -> Bound<'py, PyArrayDyn<f32>> {
        PyArray1::<f32>::from_slice(py, values)
            .reshape((rows, width))
            .unwrap()
            .into_any()
            .extract()
            .unwrap()
    }

    #[test]
    fn snapshot_holds_and_releases_an_exhausted_pool_reservation() {
        Python::initialize();
        Python::attach(|py| {
            let pool = ResultPool::new(8).unwrap();
            let held = pool.reserve(8).unwrap();
            let error = snapshot_result(contiguous_array(py, &[1.0, 2.0], 1, 2), &pool)
                .err()
                .unwrap();
            assert!(error.is_instance_of::<PyMemoryError>(py));
            assert_eq!(pool.used(), 8);
            drop(held);
            assert_eq!(pool.used(), 0);
            let snapshot = snapshot_result(contiguous_array(py, &[1.0, 2.0], 1, 2), &pool).unwrap();
            assert_eq!(pool.used(), 8);
            assert_eq!(snapshot.values, vec![1.0, 2.0]);
            assert_eq!((snapshot.rows, snapshot.width), (1, 2));
            drop(snapshot);
            assert_eq!(pool.used(), 0);
        });
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
                .set_result(contiguous_array(py, &[1.0, 2.0, 3.0, 4.0], 2, 2))
                .unwrap();
            assert_eq!(semaphore.available_permits(), 1);
            let snapshot = receiver.blocking_recv().unwrap().unwrap();
            assert_eq!((snapshot.rows, snapshot.width), (2, 2));
            assert_eq!(snapshot.values, vec![1.0, 2.0, 3.0, 4.0]);
            assert!(envelope
                .borrow_mut(py)
                .set_result(contiguous_array(py, &[5.0], 1, 1))
                .is_err());
            drop(envelope);
            assert_eq!(semaphore.available_permits(), 1);
        });
    }

    #[test]
    fn failed_envelope_releases_its_permit_before_delivery() {
        Python::initialize();
        Python::attach(|py| {
            let semaphore = Arc::new(Semaphore::new(1));
            let permit = semaphore.clone().try_acquire_owned().unwrap();
            let (envelope, receiver) = ResponseEnvelope::channel(py, Some(permit)).unwrap();
            let error = pyo3::exceptions::PyValueError::new_err("boom");
            envelope
                .borrow_mut(py)
                .set_exception(error.value(py).clone().into_any())
                .unwrap();
            assert_eq!(semaphore.available_permits(), 1);
            assert!(receiver.blocking_recv().unwrap().is_err());
            drop(envelope);
            assert_eq!(semaphore.available_permits(), 1);
        });
    }

    #[test]
    fn canceled_receiver_still_releases_the_permit() {
        Python::initialize();
        Python::attach(|py| {
            let semaphore = Arc::new(Semaphore::new(1));
            let permit = semaphore.clone().try_acquire_owned().unwrap();
            let (envelope, receiver) = ResponseEnvelope::channel(py, Some(permit)).unwrap();
            drop(receiver);
            envelope
                .borrow_mut(py)
                .set_result(contiguous_array(py, &[1.0, 2.0], 1, 2))
                .unwrap();
            assert_eq!(semaphore.available_permits(), 1);
            drop(envelope);
            assert_eq!(semaphore.available_permits(), 1);
        });
    }

    #[test]
    fn snapshot_ignores_later_mutation_through_the_numpy_base() {
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(
                c"import numpy as np
base = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
view = base.reshape(2, 2)",
                None,
                Some(&locals),
            )
            .unwrap();
            let view = locals
                .get_item("view")
                .unwrap()
                .unwrap()
                .extract::<Bound<'_, PyArrayDyn<f32>>>()
                .unwrap();
            let (envelope, receiver) = ResponseEnvelope::channel(py, None).unwrap();
            envelope.borrow_mut(py).set_result(view).unwrap();
            let snapshot = receiver.blocking_recv().unwrap().unwrap();
            py.run(c"base[:] = 99.0", None, Some(&locals)).unwrap();
            assert_eq!(snapshot.values, vec![1.0, 2.0, 3.0, 4.0]);
        });
    }

    #[test]
    fn invalid_output_keeps_the_envelope_usable_for_an_exception() {
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(
                c"import numpy as np
rank_one = np.zeros(4, dtype=np.float32)
rank_three = np.zeros((1, 2, 2), dtype=np.float32)
zero_width = np.zeros((2, 0), dtype=np.float32)
strided = np.arange(6, dtype=np.float32).reshape(2, 3)[:, ::2]",
                None,
                Some(&locals),
            )
            .unwrap();
            let (envelope, receiver) = ResponseEnvelope::channel(py, None).unwrap();
            for name in ["rank_one", "rank_three", "zero_width", "strided"] {
                let array = locals
                    .get_item(name)
                    .unwrap()
                    .unwrap()
                    .extract::<Bound<'_, PyArrayDyn<f32>>>()
                    .unwrap();
                assert!(
                    envelope.borrow_mut(py).set_result(array).is_err(),
                    "{name} should be rejected"
                );
            }
            let error = pyo3::exceptions::PyValueError::new_err("bad output");
            envelope
                .borrow_mut(py)
                .set_exception(error.value(py).clone().into_any())
                .unwrap();
            assert!(receiver.blocking_recv().unwrap().is_err());
        });
    }
}
