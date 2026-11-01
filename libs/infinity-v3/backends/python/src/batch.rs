use crate::envelope::OutputSnapshot;
use infinity_v3_backend_core::{
    result_memory, BackendError, Batch, Embedding, Embeddings, Predictions, SharedF32,
};
use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;

/// Rectangular input derived from the packed core batch. Accelerator plugins may
/// compile per shape, so one padded width is preferable to ragged rows.
#[derive(Debug, Clone)]
pub(crate) struct PaddedInputs {
    pub batch_size: usize,
    pub seq_len: usize,
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
    pub token_type_ids: Vec<i64>,
    pub position_ids: Vec<i64>,
    pub lengths: Vec<i64>,
    pub pooled_indices: Vec<i64>,
    pub raw_indices: Vec<i64>,
}

/// Keeps the Rust input buffers alive while queued numpy views reference them.
#[pyclass]
struct PaddedInputOwner {
    _inputs: Arc<PaddedInputs>,
}

impl PaddedInputs {
    pub(crate) fn to_python<'py>(
        inputs: Arc<Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let owner = Bound::new(
            py,
            PaddedInputOwner {
                _inputs: inputs.clone(),
            },
        )?
        .into_any();
        let payload = PyDict::new(py);
        payload.set_item("batch_size", inputs.batch_size)?;
        payload.set_item("seq_len", inputs.seq_len)?;
        payload.set_item("input_ids", unsafe {
            PyArray1::borrow_from_array(
                &numpy::ndarray::ArrayView1::from(&inputs.input_ids[..]),
                owner.clone(),
            )
        })?;
        payload.set_item("attention_mask", unsafe {
            PyArray1::borrow_from_array(
                &numpy::ndarray::ArrayView1::from(&inputs.attention_mask[..]),
                owner.clone(),
            )
        })?;
        payload.set_item("token_type_ids", unsafe {
            PyArray1::borrow_from_array(
                &numpy::ndarray::ArrayView1::from(&inputs.token_type_ids[..]),
                owner.clone(),
            )
        })?;
        payload.set_item("position_ids", unsafe {
            PyArray1::borrow_from_array(
                &numpy::ndarray::ArrayView1::from(&inputs.position_ids[..]),
                owner.clone(),
            )
        })?;
        payload.set_item("lengths", unsafe {
            PyArray1::borrow_from_array(
                &numpy::ndarray::ArrayView1::from(&inputs.lengths[..]),
                owner.clone(),
            )
        })?;
        payload.set_item("pooled_indices", unsafe {
            PyArray1::borrow_from_array(
                &numpy::ndarray::ArrayView1::from(&inputs.pooled_indices[..]),
                owner.clone(),
            )
        })?;
        payload.set_item("raw_indices", unsafe {
            PyArray1::borrow_from_array(
                &numpy::ndarray::ArrayView1::from(&inputs.raw_indices[..]),
                owner,
            )
        })?;
        Ok(payload)
    }

    fn raw_row_count(&self) -> Result<usize, BackendError> {
        self.raw_indices.iter().try_fold(0usize, |total, &index| {
            let index = usize::try_from(index)
                .map_err(|_| BackendError::Inference("raw index is negative".into()))?;
            let length = self
                .lengths
                .get(index)
                .copied()
                .ok_or_else(|| BackendError::Inference("raw index is out of range".into()))?;
            let length = usize::try_from(length)
                .map_err(|_| BackendError::Inference("sequence length is negative".into()))?;
            total
                .checked_add(length)
                .ok_or_else(|| BackendError::Inference("raw output row count overflow".into()))
        })
    }
}

pub(crate) fn pad_batch(batch: &Batch) -> Result<PaddedInputs, BackendError> {
    if batch.cumulative_seq_lengths.len() < 2 {
        return Err(BackendError::Inference(
            "Python backend batch has no sequences".into(),
        ));
    }
    if batch.cumulative_seq_lengths.first() != Some(&0) {
        return Err(BackendError::Inference(
            "Python backend batch cumulative lengths must start at zero".into(),
        ));
    }
    if !batch.multimodal.is_empty() && batch.multimodal.iter().any(Option::is_some) {
        return Err(BackendError::Inference(
            "Python backend backend does not support multimodal inputs".into(),
        ));
    }
    if batch.compact_input_ids.is_some()
        || batch.compact_position_ids.is_some()
        || batch.fold_gather.is_some()
        || batch.scatter_unfold.is_some()
    {
        return Err(BackendError::Inference(
            "Python backend backend does not support folded/compact token layouts".into(),
        ));
    }
    let tokens = batch.input_ids.len();
    if batch.token_type_ids.len() != tokens || batch.position_ids.len() != tokens {
        return Err(BackendError::Inference(
            "Python backend batch tensor lengths disagree".into(),
        ));
    }
    let mut previous = 0_u32;
    let mut lengths = Vec::with_capacity(batch.cumulative_seq_lengths.len() - 1);
    for &end in batch.cumulative_seq_lengths.iter().skip(1) {
        if end < previous || end as usize > tokens {
            return Err(BackendError::Inference(
                "Python backend batch cumulative lengths are invalid".into(),
            ));
        }
        lengths.push((end - previous) as usize);
        previous = end;
    }
    let seq_len = lengths.iter().copied().max().unwrap_or(0);
    if seq_len == 0 {
        return Err(BackendError::Inference(
            "Python backend batch contains an empty sequence".into(),
        ));
    }
    let batch_size = lengths.len();
    for index in batch.pooled_indices.iter().chain(&batch.raw_indices) {
        if *index as usize >= batch_size {
            return Err(BackendError::Inference(
                "Python backend output index is out of range".into(),
            ));
        }
    }
    let total = batch_size.checked_mul(seq_len).ok_or_else(|| {
        BackendError::Inference("Python backend padded batch size overflow".into())
    })?;
    let mut input_ids = vec![0_i64; total];
    let mut attention_mask = vec![0_i64; total];
    let mut token_type_ids = vec![0_i64; total];
    let mut position_ids = vec![0_i64; total];
    let mut source_offset = 0usize;
    for (row, &length) in lengths.iter().enumerate() {
        let target_offset = row
            .checked_mul(seq_len)
            .ok_or_else(|| BackendError::Inference("Python backend row offset overflow".into()))?;
        for column in 0..length {
            let source = source_offset + column;
            let target = target_offset + column;
            let input_id = batch.input_ids.get(source).ok_or_else(|| {
                BackendError::Inference("Python backend input token is missing".into())
            })?;
            let token_type = batch.token_type_ids.get(source).ok_or_else(|| {
                BackendError::Inference("Python backend token type is missing".into())
            })?;
            let position = batch.position_ids.get(source).ok_or_else(|| {
                BackendError::Inference("Python backend position is missing".into())
            })?;
            let target_input = input_ids.get_mut(target).ok_or_else(|| {
                BackendError::Inference("Python backend input slot is missing".into())
            })?;
            let target_type = token_type_ids.get_mut(target).ok_or_else(|| {
                BackendError::Inference("Python backend type slot is missing".into())
            })?;
            let target_position = position_ids.get_mut(target).ok_or_else(|| {
                BackendError::Inference("Python backend position slot is missing".into())
            })?;
            let target_mask = attention_mask.get_mut(target).ok_or_else(|| {
                BackendError::Inference("Python backend mask slot is missing".into())
            })?;
            *target_input = i64::from(*input_id);
            *target_type = i64::from(*token_type);
            *target_position = i64::from(*position);
            *target_mask = 1;
        }
        source_offset = source_offset.checked_add(length).ok_or_else(|| {
            BackendError::Inference("Python backend source offset overflow".into())
        })?;
    }
    Ok(PaddedInputs {
        batch_size,
        seq_len,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        lengths: lengths.iter().map(|&length| length as i64).collect(),
        pooled_indices: batch
            .pooled_indices
            .iter()
            .map(|&index| i64::from(index))
            .collect(),
        raw_indices: batch
            .raw_indices
            .iter()
            .map(|&index| i64::from(index))
            .collect(),
    })
}

/// Private, immutable Rust allocation backing shared float rows.
struct FloatOwner(Vec<f32>);

impl AsRef<[u8]> for FloatOwner {
    fn as_ref(&self) -> &[u8] {
        // SAFETY: f32 values are initialized and the private owner never mutates
        // or resizes the allocation after it is handed to Bytes.
        unsafe {
            std::slice::from_raw_parts(
                self.0.as_ptr().cast::<u8>(),
                std::mem::size_of_val(self.0.as_slice()),
            )
        }
    }
}

pub(crate) fn pack_embeddings(
    output: OutputSnapshot,
    inputs: &PaddedInputs,
) -> Result<Embeddings, BackendError> {
    let OutputSnapshot {
        values,
        rows,
        width,
        memory,
    } = output;
    let expected = inputs
        .pooled_indices
        .len()
        .checked_add(inputs.raw_row_count()?)
        .ok_or_else(|| {
            BackendError::Inference("Python backend output row count overflow".into())
        })?;
    if rows != expected {
        return Err(BackendError::Inference(format!(
            "Python backend returned {rows} rows; expected {expected}"
        )));
    }
    let bytes = bytes::Bytes::from_owner(FloatOwner(values));
    let element_size = std::mem::size_of::<f32>();
    let mut result = Embeddings::default();
    result
        .try_reserve(inputs.pooled_indices.len() + inputs.raw_indices.len())
        .map_err(|error| BackendError::Overloaded(error.to_string()))?;
    let mut row = 0usize;
    for &index in &inputs.pooled_indices {
        let offset = row
            .checked_mul(width)
            .and_then(|value| value.checked_mul(element_size))
            .ok_or_else(|| BackendError::Inference("Python backend row offset overflow".into()))?;
        let shared = SharedF32::shared(bytes.clone(), offset, width)
            .map_err(|error| BackendError::Inference(format!("Python backend {error}")))?;
        let index = usize::try_from(index)
            .map_err(|_| BackendError::Inference("pooled index is negative".into()))?;
        result.insert(index, Embedding::Shared(shared));
        row += 1;
    }
    for &index in &inputs.raw_indices {
        let index = usize::try_from(index)
            .map_err(|_| BackendError::Inference("raw index is negative".into()))?;
        let length = *inputs
            .lengths
            .get(index)
            .ok_or_else(|| BackendError::Inference("raw index is out of range".into()))?
            as usize;
        let mut token_rows = Vec::new();
        token_rows
            .try_reserve_exact(length)
            .map_err(|error| BackendError::Overloaded(error.to_string()))?;
        for _ in 0..length {
            let offset = row
                .checked_mul(width)
                .and_then(|value| value.checked_mul(element_size))
                .ok_or_else(|| {
                    BackendError::Inference("Python backend row offset overflow".into())
                })?;
            token_rows.push(
                SharedF32::shared(bytes.clone(), offset, width)
                    .map_err(|error| BackendError::Inference(format!("Python backend {error}")))?,
            );
            row += 1;
        }
        result.insert(index, Embedding::SharedAll(token_rows));
    }
    result.memory = Some(memory);
    Ok(result)
}

pub(crate) fn pack_predictions(
    output: OutputSnapshot,
    inputs: &PaddedInputs,
    pool: &Arc<result_memory::ResultPool>,
) -> Result<Predictions, BackendError> {
    let OutputSnapshot {
        values,
        rows,
        width,
        memory: snapshot_memory,
    } = output;
    if rows != inputs.batch_size {
        return Err(BackendError::Inference(format!(
            "Python backend returned {rows} prediction rows; expected {}",
            inputs.batch_size
        )));
    }
    let bytes = values
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| BackendError::Inference("Python backend result size overflow".into()))?;
    // The snapshot remains live while score rows are copied. Admit that overlap
    // before allocating; transfer only the new reservation to the result.
    let memory = pool.reserve(bytes)?;
    let mut result = Predictions::default();
    result
        .try_reserve(inputs.batch_size)
        .map_err(|error| BackendError::Overloaded(error.to_string()))?;
    for index in 0..inputs.batch_size {
        let start = index.checked_mul(width).ok_or_else(|| {
            BackendError::Inference("Python backend prediction offset overflow".into())
        })?;
        let end = start.checked_add(width).ok_or_else(|| {
            BackendError::Inference("Python backend prediction offset overflow".into())
        })?;
        let row_values = values.get(start..end).ok_or_else(|| {
            BackendError::Inference("Python backend prediction row is missing".into())
        })?;
        let mut scores = Vec::new();
        scores
            .try_reserve_exact(width)
            .map_err(|error| BackendError::Overloaded(error.to_string()))?;
        scores.extend_from_slice(row_values);
        result.insert(index, scores);
    }
    drop(values);
    drop(snapshot_memory);
    result.memory = Some(memory);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::PyArrayMethods;

    fn batch() -> Batch {
        Batch {
            multimodal: vec![],
            input_ids: vec![10, 11, 20, 21, 22],
            token_type_ids: vec![0, 0, 1, 1, 1],
            position_ids: vec![0, 1, 0, 1, 2],
            cumulative_seq_lengths: vec![0, 2, 5],
            max_length: 3,
            pooled_indices: vec![0],
            raw_indices: vec![1],
            compact_input_ids: None,
            compact_position_ids: None,
            fold_gather: None,
            scatter_unfold: None,
            tokens: vec![],
            offsets: vec![],
        }
    }

    #[test]
    fn pads_ragged_batches_and_builds_masks() {
        let padded = pad_batch(&batch()).unwrap();
        assert_eq!((padded.batch_size, padded.seq_len), (2, 3));
        assert_eq!(padded.input_ids, vec![10, 11, 0, 20, 21, 22]);
        assert_eq!(padded.attention_mask, vec![1, 1, 0, 1, 1, 1]);
        assert_eq!(padded.token_type_ids, vec![0, 0, 0, 1, 1, 1]);
        assert_eq!(padded.position_ids, vec![0, 1, 0, 0, 1, 2]);
        assert_eq!(padded.lengths, vec![2, 3]);
        assert_eq!(padded.pooled_indices, vec![0]);
        assert_eq!(padded.raw_indices, vec![1]);
    }

    #[test]
    fn rejects_inconsistent_cumulative_lengths() {
        let mut batch = batch();
        batch.cumulative_seq_lengths = vec![0, 2, 9];
        assert!(pad_batch(&batch).is_err());
    }

    #[test]
    fn input_arrays_borrow_rust_buffers_without_copying() {
        Python::initialize();
        let inputs = Arc::new(pad_batch(&batch()).unwrap());
        let input_ptr = inputs.input_ids.as_ptr();
        Python::attach(|py| {
            let payload = PaddedInputs::to_python(inputs.clone(), py).unwrap();
            let array = payload
                .get_item("input_ids")
                .unwrap()
                .unwrap()
                .extract::<Bound<'_, PyArray1<i64>>>()
                .unwrap();
            assert_eq!(array.readonly().as_slice().unwrap().as_ptr(), input_ptr);
        });
    }

    #[test]
    fn packs_pooled_and_raw_rows_in_request_order() {
        let padded = pad_batch(&batch()).unwrap();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let output = OutputSnapshot {
            memory: result_memory::global_pool()
                .unwrap()
                .reserve(values.len() * 4)
                .unwrap(),
            values,
            rows: 4,
            width: 2,
        };
        let embeddings = pack_embeddings(output, &padded).unwrap();
        let Embedding::Shared(pooled) = &embeddings[&0] else {
            panic!("expected pooled output");
        };
        assert_eq!(pooled.as_ref(), &[1.0, 2.0]);
        let Embedding::SharedAll(raw) = &embeddings[&1] else {
            panic!("expected raw output");
        };
        assert_eq!(raw.len(), 3);
        assert_eq!(raw[0].as_ref(), &[3.0, 4.0]);
        assert_eq!(raw[1].as_ref(), &[5.0, 6.0]);
        assert_eq!(raw[2].as_ref(), &[7.0, 8.0]);
    }
    fn prediction_snapshot(pool: &Arc<result_memory::ResultPool>) -> OutputSnapshot {
        OutputSnapshot {
            values: vec![1.0, 2.0, 3.0, 4.0],
            rows: 2,
            width: 2,
            memory: pool.reserve(16).unwrap(),
        }
    }

    #[test]
    fn predictions_reject_unfunded_copy_and_release_the_snapshot() {
        let pool = result_memory::ResultPool::new(16).unwrap();
        let output = prediction_snapshot(&pool);
        let result = pack_predictions(output, &pad_batch(&batch()).unwrap(), &pool);
        assert!(matches!(result, Err(BackendError::Overloaded(_))));
        assert_eq!(pool.used(), 0);
    }

    #[test]
    fn predictions_transfer_the_reservation_after_copying() {
        let pool = result_memory::ResultPool::new(32).unwrap();
        let output = prediction_snapshot(&pool);
        let result = pack_predictions(output, &pad_batch(&batch()).unwrap(), &pool).unwrap();
        assert_eq!(result[&0], [1.0, 2.0]);
        assert_eq!(result[&1], [3.0, 4.0]);
        assert_eq!(pool.used(), 16);
        drop(result);
        assert_eq!(pool.used(), 0);
    }
}
