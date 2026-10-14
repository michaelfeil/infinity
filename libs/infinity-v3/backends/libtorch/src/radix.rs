use crate::{filled_vec, inference, reserved_vec};
use text_embeddings_backend_core::{BackendError, Batch};

#[repr(C)]
pub(crate) struct NativeRadix {
    ids: *const i64,
    types: *const i64,
    positions: *const i64,
    scatter: *const i64,
    fold: *const i64,
    compact_tokens: i64,
}

pub(crate) struct OwnedRadix {
    ids: Vec<i64>,
    types: Vec<i64>,
    positions: Vec<i64>,
    scatter: Vec<i64>,
    fold: Vec<i64>,
}

impl OwnedRadix {
    pub(crate) fn from_batch(batch: &Batch) -> Result<Option<Self>, BackendError> {
        let metadata = (
            &batch.compact_input_ids,
            &batch.compact_position_ids,
            &batch.scatter_unfold,
            &batch.fold_gather,
        );
        let (Some(ids), Some(positions), Some(scatter), Some(fold)) = metadata else {
            return if matches!(metadata, (None, None, None, None)) {
                Ok(None)
            } else {
                Err(inference("Incomplete radix metadata"))
            };
        };
        let n = batch.input_ids.len();
        if n == 0
            || batch.token_type_ids.len() != n
            || batch.position_ids.len() != n
            || scatter.len() != n
            || ids.is_empty()
            || positions.len() != ids.len()
            || fold.len() != ids.len()
        {
            return Err(inference("Invalid radix dimensions"));
        }
        let mut used = filled_vec(ids.len(), false)?;
        for &index in scatter {
            let Some(row) = used.get_mut(index as usize) else {
                return Err(inference("Radix scatter index out of bounds"));
            };
            *row = true;
        }
        // The core may append unused alignment rows. Remove them before inference:
        // the native compact allocation contains only rows referenced by real tokens.
        let mut remap = filled_vec(ids.len(), 0i64)?;
        let compact_rows = used.iter().filter(|&&used| used).count();
        let mut result = Self {
            ids: reserved_vec(compact_rows)?,
            types: reserved_vec(compact_rows)?,
            positions: reserved_vec(compact_rows)?,
            scatter: reserved_vec(n)?,
            fold: reserved_vec(compact_rows)?,
        };
        for (old, &is_used) in used.iter().enumerate() {
            if !is_used {
                continue;
            }
            let representative = fold[old] as usize;
            if representative >= n || scatter[representative] as usize != old {
                return Err(inference("Radix fold is not a scatter representative"));
            }
            remap[old] = i64::try_from(result.ids.len()).map_err(inference)?;
            result.ids.push(i64::from(ids[old]));
            result.positions.push(i64::from(positions[old]));
            result
                .types
                .push(i64::from(batch.token_type_ids[representative]));
            result
                .fold
                .push(i64::try_from(representative).map_err(inference)?);
        }
        for (full, &old) in scatter.iter().enumerate() {
            let compact = remap[old as usize];
            let index = compact as usize;
            if result.ids[index] != i64::from(batch.input_ids[full])
                || result.positions[index] != i64::from(batch.position_ids[full])
                || result.types[index] != i64::from(batch.token_type_ids[full])
            {
                return Err(inference(
                    "Radix aliases disagree on token, position or type",
                ));
            }
            result.scatter.push(compact);
        }
        Ok(Some(result))
    }

    pub(crate) fn native(&self) -> NativeRadix {
        NativeRadix {
            ids: self.ids.as_ptr(),
            types: self.types.as_ptr(),
            positions: self.positions.as_ptr(),
            scatter: self.scatter.as_ptr(),
            fold: self.fold.as_ptr(),
            compact_tokens: self.ids.len() as i64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn batch() -> Batch {
        Batch {
            input_ids: vec![1, 2, 1, 3],
            token_type_ids: vec![0; 4],
            position_ids: vec![0, 1, 0, 1],
            cumulative_seq_lengths: vec![0, 2, 4],
            max_length: 2,
            pooled_indices: vec![0, 1],
            raw_indices: vec![],
            multimodal: vec![],
            compact_input_ids: Some(vec![1, 2, 3]),
            compact_position_ids: Some(vec![0, 1, 1]),
            scatter_unfold: Some(vec![0, 1, 0, 2]),
            fold_gather: Some(vec![0, 1, 3]),
            tokens: vec![],
            offsets: vec![],
        }
    }

    #[test]
    fn strips_interior_and_trailing_unused_rows_even_when_original_m_exceeds_n() {
        let mut batch = batch();
        batch.compact_input_ids = Some(vec![999, 1, 999, 2, 3, 999]);
        batch.compact_position_ids = Some(vec![999, 0, 999, 1, 1, 999]);
        batch.scatter_unfold = Some(vec![1, 3, 1, 4]);
        batch.fold_gather = Some(vec![u32::MAX, 0, u32::MAX, 1, 3, u32::MAX]);
        let owned = OwnedRadix::from_batch(&batch).unwrap().unwrap();
        assert_eq!(owned.ids, [1, 2, 3]);
        assert_eq!(owned.positions, [0, 1, 1]);
        assert_eq!(owned.scatter, [0, 1, 0, 2]);
        assert_eq!(owned.fold, [0, 1, 3]);
        assert_eq!(owned.native().compact_tokens, 3);
    }

    #[test]
    fn rejects_partial_metadata_and_invalid_dimensions_or_indices() {
        let mut partial = batch();
        partial.fold_gather = None;
        assert!(OwnedRadix::from_batch(&partial).is_err());
        let mut invalid = batch();
        invalid.scatter_unfold.as_mut().unwrap()[0] = 3;
        assert!(OwnedRadix::from_batch(&invalid).is_err());
        invalid = batch();
        invalid.fold_gather.as_mut().unwrap()[0] = 1;
        assert!(OwnedRadix::from_batch(&invalid).is_err());
        invalid = batch();
        invalid.compact_position_ids.as_mut().unwrap().pop();
        assert!(OwnedRadix::from_batch(&invalid).is_err());
    }

    #[test]
    fn rejects_conflicting_alias_tokens_positions_and_types() {
        for field in 0..3 {
            let mut invalid = batch();
            match field {
                0 => invalid.input_ids[2] = 4,
                1 => invalid.position_ids[2] = 1,
                _ => invalid.token_type_ids[2] = 1,
            }
            assert!(OwnedRadix::from_batch(&invalid).is_err());
        }
    }
}
