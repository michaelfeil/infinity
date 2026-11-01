// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Compact pooled/token output layout shared by both dense executors.

use super::{infer, product};
use infinity_v3_backend_core::{BackendError, Batch, Embedding, Embeddings};

fn range(boundaries: &[u32], index: u32) -> Result<std::ops::Range<u32>, BackendError> {
    let index = index as usize;
    let next = index
        .checked_add(1)
        .ok_or_else(|| infer("Raw output boundary index overflow"))?;
    let start = boundaries
        .get(index)
        .copied()
        .ok_or_else(|| infer("Invalid raw output index"))?;
    let end = boundaries
        .get(next)
        .copied()
        .filter(|end| *end > start)
        .ok_or_else(|| infer("Invalid raw output boundaries"))?;
    Ok(start..end)
}

pub(crate) fn raw_rows(batch: &Batch) -> Result<usize, BackendError> {
    batch.raw_indices.iter().try_fold(0usize, |rows, &index| {
        let span = range(&batch.cumulative_seq_lengths, index)?;
        rows.checked_add((span.end - span.start) as usize)
            .ok_or_else(|| infer("Raw output row count overflow"))
    })
}

pub(crate) fn rows(batch: &Batch) -> Result<usize, BackendError> {
    batch
        .pooled_indices
        .len()
        .checked_add(raw_rows(batch)?)
        .ok_or_else(|| infer("Output row count overflow"))
}

/// Physical output widths: the backbone/pooled staging width may differ from
/// the Dense projected width, while token rows always use the backbone width.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct OutputGeometry {
    pub(crate) pooled_width: usize,
    pub(crate) raw_width: usize,
}

impl OutputGeometry {
    pub(crate) fn new(pooled_width: usize, raw_width: usize) -> Result<Self, BackendError> {
        if pooled_width == 0 || raw_width == 0 {
            return Err(infer("Invalid oxide embedding output width"));
        }
        Ok(Self {
            pooled_width,
            raw_width,
        })
    }

    /// Host/client payload: `P * D` pooled floats followed by `R * H` raw
    /// floats. This deliberately avoids reserving `(P + R) * max(D, H)`.
    pub(crate) fn host_elements(&self, batch: &Batch) -> Result<usize, BackendError> {
        product(batch.pooled_indices.len(), self.pooled_width)?
            .checked_add(product(raw_rows(batch)?, self.raw_width)?)
            .ok_or_else(|| infer("Host output element count overflow"))
    }

    /// Backbone device payload: every selected row keeps the backbone width.
    pub(crate) fn backbone_elements(&self, batch: &Batch) -> Result<usize, BackendError> {
        product(rows(batch)?, self.raw_width)
    }

    /// First raw row inside the backbone allocation, immediately after the
    /// physical pooled prefix.
    pub(crate) fn raw_source_offset(&self, pooled_rows: usize) -> Result<usize, BackendError> {
        product(pooled_rows, self.raw_width)
    }

    /// First raw row inside the host payload, immediately after the projected
    /// pooled prefix.
    pub(crate) fn raw_host_offset(&self, pooled_rows: usize) -> Result<usize, BackendError> {
        product(pooled_rows, self.pooled_width)
    }
}

/// Pooled-only accounting retains the existing payload reservation. Token
/// results additionally own a Vec per row and use the native backend's
/// conservative map-bucket allowance for selected segments.
pub(crate) fn result_bytes(
    batch: &Batch,
    pooled_width: usize,
    raw_width: usize,
) -> Result<usize, BackendError> {
    let raw_rows = raw_rows(batch)?;
    if (!batch.pooled_indices.is_empty() && pooled_width == 0) || (raw_rows > 0 && raw_width == 0) {
        return Err(infer("Invalid oxide embedding output width"));
    }
    let payload = product(batch.pooled_indices.len(), pooled_width)?
        .checked_add(product(raw_rows, raw_width)?)
        .and_then(|elements| product(elements, std::mem::size_of::<f32>()).ok())
        .ok_or_else(|| infer("Token output memory size overflow"))?;
    if batch.raw_indices.is_empty() {
        return Ok(payload);
    }
    let outputs = batch
        .pooled_indices
        .len()
        .checked_add(batch.raw_indices.len())
        .ok_or_else(|| infer("Output count overflow"))?;
    payload
        .checked_add(product(raw_rows, std::mem::size_of::<Vec<f32>>())?)
        .and_then(|bytes| {
            product(outputs, 8 * std::mem::size_of::<(usize, Embedding)>() + 8)
                .ok()
                .and_then(|entries| bytes.checked_add(entries))
        })
        .ok_or_else(|| infer("Token output memory size overflow"))
}

/// The GPU selection tail holds token-row indices rather than segment IDs.
/// Expanding directly into pinned storage needs no temporary index allocation.
pub(crate) fn write_raw_indices(batch: &Batch, out: &mut [u32]) -> Result<(), BackendError> {
    let mut destination = out.iter_mut();
    for &index in &batch.raw_indices {
        for token in range(&batch.cumulative_seq_lengths, index)? {
            *destination
                .next()
                .ok_or_else(|| infer("Raw selection buffer is too short"))? = token;
        }
    }
    if destination.next().is_some() {
        return Err(infer("Raw selection buffer is too long"));
    }
    Ok(())
}

fn copied_row(row: &[f32]) -> Result<Vec<f32>, BackendError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(row.len())
        .map_err(|error| BackendError::Overloaded(error.to_string()))?;
    values.extend_from_slice(row);
    Ok(values)
}

pub(crate) fn pack(
    output: &[f32],
    pooled_width: usize,
    raw_width: usize,
    pooled: &[u32],
    raw: &[u32],
    boundaries: &[u32],
) -> Result<Embeddings, BackendError> {
    if (!pooled.is_empty() && pooled_width == 0) || (!raw.is_empty() && raw_width == 0) {
        return Err(infer("Invalid token output width"));
    }
    let count = pooled
        .len()
        .checked_add(raw.len())
        .ok_or_else(|| infer("Output count overflow"))?;
    let mut results = Embeddings::default();
    results
        .try_reserve(count)
        .map_err(|error| BackendError::Overloaded(error.to_string()))?;
    let mut rest = output;
    for &index in pooled {
        let (row, tail) = rest
            .split_at_checked(pooled_width)
            .ok_or_else(|| infer("Pooled output buffer is too short"))?;
        results.insert(index as usize, Embedding::Pooled(copied_row(row)?));
        rest = tail;
    }
    for &index in raw {
        let span = range(boundaries, index)?;
        let rows = (span.end - span.start) as usize;
        let elements = product(rows, raw_width)?;
        let (values, tail) = rest
            .split_at_checked(elements)
            .ok_or_else(|| infer("Token output buffer is too short"))?;
        let mut tokens = Vec::new();
        tokens
            .try_reserve_exact(rows)
            .map_err(|error| BackendError::Overloaded(error.to_string()))?;
        for row in values.chunks_exact(raw_width) {
            tokens.push(copied_row(row)?);
        }
        // As in the native backend, raw selections follow pooled selections;
        // a repeated sequence replaces the earlier map value.
        results.insert(index as usize, Embedding::All(tokens));
        rest = tail;
    }
    if !rest.is_empty() {
        return Err(infer("Unexpected trailing token output data"));
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn batch() -> Batch {
        Batch {
            input_ids: vec![1; 6],
            token_type_ids: vec![0; 6],
            position_ids: vec![0, 1, 0, 1, 2, 0],
            cumulative_seq_lengths: vec![0, 2, 5, 6],
            max_length: 3,
            pooled_indices: vec![1],
            raw_indices: vec![2, 0, 2],
            multimodal: vec![],
            compact_input_ids: None,
            compact_position_ids: None,
            scatter_unfold: None,
            fold_gather: None,
            tokens: vec![],
            offsets: vec![],
        }
    }

    fn widths_batch(pooled: Vec<u32>, raw: Vec<u32>) -> Batch {
        Batch {
            input_ids: vec![1; 8],
            token_type_ids: vec![0; 8],
            position_ids: vec![0, 1, 0, 1, 2, 0, 1, 2],
            cumulative_seq_lengths: vec![0, 2, 5, 8],
            max_length: 3,
            pooled_indices: pooled,
            raw_indices: raw,
            multimodal: vec![],
            compact_input_ids: None,
            compact_position_ids: None,
            scatter_unfold: None,
            fold_gather: None,
            tokens: vec![],
            offsets: vec![],
        }
    }

    fn mixed_output(
        pooled_rows: usize,
        pooled_width: usize,
        raw_rows: usize,
        raw_width: usize,
    ) -> Vec<f32> {
        let mut values = Vec::new();
        for row in 0..pooled_rows {
            for column in 0..pooled_width {
                values.push((1000 + row * 10 + column) as f32);
            }
        }
        for row in 0..raw_rows {
            for column in 0..raw_width {
                values.push((row * 10 + column) as f32);
            }
        }
        values
    }

    #[test]
    fn mixed_width_geometry_separates_host_and_backbone_payloads() {
        let batch = widths_batch(vec![0, 0, 0], vec![0, 1, 2]);
        for (pooled_width, raw_width) in [(7usize, 3usize), (3, 7)] {
            let geometry = OutputGeometry::new(pooled_width, raw_width).unwrap();
            let host = geometry.host_elements(&batch).unwrap();
            let backbone = geometry.backbone_elements(&batch).unwrap();
            let max_width = rows(&batch).unwrap() * pooled_width.max(raw_width);
            assert_eq!(host, 3 * pooled_width + 8 * raw_width);
            assert_eq!(backbone, (3 + 8) * raw_width);
            assert_ne!(host, max_width, "must not reserve (P + R) * max(D, H)");
            assert_eq!(geometry.raw_source_offset(3).unwrap(), 3 * raw_width);
            assert_eq!(geometry.raw_host_offset(3).unwrap(), 3 * pooled_width);
        }
        assert_eq!(
            OutputGeometry::new(7, 3)
                .unwrap()
                .host_elements(&batch)
                .unwrap(),
            45
        );
        assert_eq!(
            OutputGeometry::new(7, 3)
                .unwrap()
                .backbone_elements(&batch)
                .unwrap(),
            33
        );
        assert_eq!(
            OutputGeometry::new(3, 7)
                .unwrap()
                .host_elements(&batch)
                .unwrap(),
            65
        );
        assert_eq!(
            OutputGeometry::new(3, 7)
                .unwrap()
                .backbone_elements(&batch)
                .unwrap(),
            77
        );
    }

    #[test]
    fn result_bytes_uses_exact_host_geometry_and_pooled_only_reservation() {
        let pooled = widths_batch(vec![0, 1, 2], vec![]);
        assert_eq!(result_bytes(&pooled, 3, 7).unwrap(), 3 * 3 * 4);
        assert_ne!(result_bytes(&pooled, 3, 7).unwrap(), 3 * 7 * 4);
        let mixed = widths_batch(vec![1, 0, 2], vec![0, 2, 2]);
        let bytes = result_bytes(&mixed, 7, 3).unwrap();
        assert!(bytes > (3 * 7 + 8 * 3) * 4);
        assert!(result_bytes(&widths_batch(vec![0], vec![]), 0, 3).is_err());
        assert!(result_bytes(&widths_batch(vec![], vec![0]), 7, 0).is_err());
        assert_eq!(
            result_bytes(&widths_batch(vec![], vec![]), 0, 0).unwrap(),
            0
        );
    }

    #[test]
    fn mixed_pack_preserves_pooled_then_raw_widths_duplicates_and_precedence() {
        let batch = widths_batch(vec![1, 0, 2], vec![0, 2, 2]);
        for (pooled_width, raw_width) in [(7usize, 3usize), (3, 7)] {
            let output = mixed_output(3, pooled_width, 8, raw_width);
            let packed = pack(
                &output,
                pooled_width,
                raw_width,
                &batch.pooled_indices,
                &batch.raw_indices,
                &batch.cumulative_seq_lengths,
            )
            .unwrap();
            assert_eq!(packed.len(), 3);
            let Embedding::Pooled(pooled) = packed.get(&1).unwrap() else {
                panic!("sequence 1 must remain pooled")
            };
            assert_eq!(pooled.len(), pooled_width);
            assert_eq!(pooled[0], 1000.0);
            let Embedding::All(first) = packed.get(&0).unwrap() else {
                panic!("raw must replace pooled sequence 0")
            };
            assert_eq!(first.len(), 2);
            assert!(first.iter().all(|row| row.len() == raw_width));
            assert_eq!(first[0][0], 0.0);
            assert_eq!(first[1][0], 10.0);
            let Embedding::All(last) = packed.get(&2).unwrap() else {
                panic!("raw must replace pooled sequence 2")
            };
            assert_eq!(last.len(), 3);
            assert_eq!(last[0][0], 50.0, "last duplicate selection wins");
            assert_eq!(last[1][0], 60.0);
            assert_eq!(last[2][0], 70.0);
        }
    }

    #[test]
    fn mixed_pack_rejects_trailing_short_and_zero_width_payloads() {
        let batch = widths_batch(vec![1], vec![0]);
        assert!(pack(
            &[],
            7,
            3,
            &batch.pooled_indices,
            &batch.raw_indices,
            &batch.cumulative_seq_lengths
        )
        .is_err());
        assert!(pack(
            &[1., 2., 3., 4., 5., 6., 7.],
            7,
            3,
            &batch.pooled_indices,
            &batch.raw_indices,
            &batch.cumulative_seq_lengths
        )
        .is_err());
        assert!(pack(
            &[0.; 12],
            7,
            3,
            &batch.pooled_indices,
            &batch.raw_indices,
            &batch.cumulative_seq_lengths
        )
        .is_err());
        assert!(pack(
            &[0.; 14],
            7,
            3,
            &batch.pooled_indices,
            &batch.raw_indices,
            &batch.cumulative_seq_lengths
        )
        .is_err());
        assert!(pack(
            &[0.; 10],
            0,
            3,
            &batch.pooled_indices,
            &batch.raw_indices,
            &batch.cumulative_seq_lengths
        )
        .is_err());
        assert!(OutputGeometry::new(usize::MAX, 3)
            .unwrap()
            .host_elements(&widths_batch(vec![0, 1], vec![]))
            .is_err());
    }

    #[test]
    fn ragged_selections_preserve_order_repetition_and_raw_precedence() {
        let mut batch = batch();
        assert_eq!(rows(&batch).unwrap(), 5);
        let mut selected = [0; 4];
        write_raw_indices(&batch, &mut selected).unwrap();
        assert_eq!(selected, [5, 0, 1, 5]);
        batch.pooled_indices = vec![0];
        let values = pack(
            &[9., 10., 20., 21., 30., 31., 40., 41., 50., 51.],
            2,
            2,
            &batch.pooled_indices,
            &batch.raw_indices,
            &batch.cumulative_seq_lengths,
        )
        .unwrap();
        assert_eq!(values.len(), 2);
        let Embedding::All(first) = values.get(&0).unwrap() else {
            panic!("raw must replace pooled")
        };
        assert_eq!(first, &vec![vec![30., 31.], vec![40., 41.]]);
        let Embedding::All(last) = values.get(&2).unwrap() else {
            panic!("raw must remain raw")
        };
        assert_eq!(last, &vec![vec![50., 51.]]);
    }

    #[test]
    fn raw_layout_rejects_bad_boundaries_indices_lengths_and_overflow() {
        let mut batch = batch();
        assert!(write_raw_indices(&batch, &mut [0; 3]).is_err());
        assert!(write_raw_indices(&batch, &mut [0; 5]).is_err());
        batch.raw_indices = vec![3];
        assert!(rows(&batch).is_err());
        batch.raw_indices = vec![0];
        batch.cumulative_seq_lengths = vec![2, 2];
        assert!(rows(&batch).is_err());
        batch.cumulative_seq_lengths = vec![3, 2];
        assert!(rows(&batch).is_err());
        batch.cumulative_seq_lengths = vec![0, 1];
        assert!(result_bytes(&batch, usize::MAX, 2).is_err());
        assert!(pack(&[1.], 2, 2, &[], &[0], &[0, 1]).is_err());
        assert!(pack(&[1., 2., 3.], 2, 2, &[], &[0], &[0, 1]).is_err());
        assert!(pack(&[], 0, 0, &[], &[], &[0]).unwrap().is_empty());
        assert!(pack(&[], 2, 2, &[], &[], &[0]).unwrap().is_empty());
    }
}
