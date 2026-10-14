//! Prefix trees partitioned by prepared image context, with original row mappings.
use infinity_v3_backend::{ImagePatches, MultimodalEncoding};
use sha2::{Digest, Sha256};
use std::{collections::HashMap, sync::Arc};

// Tuple order: compact token IDs, compact positions, scatter_unfold, fold_gather.
// scatter[full_row] gives its compact row; fold[compact_row] picks one full
// representative. Attention still sees full sequences. Optional alignment adds
// unused compact rows, which libtorch's OwnedRadix removes before the C ABI.
type Fold = (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>);

pub(crate) fn compute(
    ids: &[u32],
    positions: &[u32],
    cumulative: &[u32],
    media: &[Option<Arc<MultimodalEncoding>>],
    pad: Option<usize>,
) -> Option<Fold> {
    if ids.len() != positions.len()
        || ids.len() > u32::MAX as usize
        || cumulative.first() != Some(&0)
        || cumulative.last().map(|n| *n as usize) != Some(ids.len())
        || cumulative.windows(2).any(|w| w[0] > w[1])
        || pad == Some(0)
        || media
            .iter()
            .flatten()
            .any(|m| m.radix_prefix_len().is_none())
    {
        return None;
    }
    if MultimodalEncoding::allows_radix(media, ids, cumulative) {
        return Some(radix_mlp::compute_fold_and_scatter(
            ids, positions, cumulative, pad,
        ));
    }
    if media.len() + 1 != cumulative.len() {
        return None;
    }
    let mut image_keys: HashMap<*const ImagePatches, [u8; 32]> = HashMap::new();
    let mut buckets: HashMap<[u8; 32], Vec<usize>> = HashMap::new();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for row in 0..media.len() {
        let tokens = &ids[cumulative[row] as usize..cumulative[row + 1] as usize];
        let m = media[row].as_deref();
        if !MultimodalEncoding::same_radix_context(m, m, tokens, tokens) {
            return None;
        }
        let mut hash = Sha256::new();
        if let Some(m) = m.filter(|m| !m.images.is_empty()) {
            let end = m.radix_prefix_len()?;
            hash.update(end.to_le_bytes());
            hash.update(bytemuck::cast_slice(&tokens[..end]));
            for axis in &m.position_ids {
                hash.update(bytemuck::cast_slice(&axis[..end]));
            }
            for (start, image) in &m.images {
                hash.update(start.to_le_bytes());
                let key = image_keys.entry(Arc::as_ptr(image)).or_insert_with(|| {
                    let mut hash = Sha256::new();
                    for value in image
                        .grid_thw
                        .into_iter()
                        .chain([image.patch_dim, image.merge_size])
                    {
                        hash.update(value.to_le_bytes());
                    }
                    hash.update(bytemuck::cast_slice(&image.pixels));
                    hash.finalize().into()
                });
                hash.update(*key);
            }
        }
        let key: [u8; 32] = hash.finalize().into();
        // Verify exact content/prefix equality even when hashes match.
        let matching = buckets.get(&key).and_then(|candidates| {
            candidates.iter().copied().find(|group| {
                let rep = groups[*group][0];
                MultimodalEncoding::same_radix_context(
                    m,
                    media[rep].as_deref(),
                    tokens,
                    &ids[cumulative[rep] as usize..cumulative[rep + 1] as usize],
                )
            })
        });
        let group = matching.unwrap_or_else(|| {
            let group = groups.len();
            groups.push(Vec::new());
            buckets.entry(key).or_default().push(group);
            group
        });
        groups[group].push(row);
    }
    let (mut compact_ids, mut compact_pos, mut scatter, mut fold) =
        (Vec::new(), Vec::new(), vec![0; ids.len()], Vec::new());
    for rows in groups {
        let (mut group_ids, mut group_pos, mut original, mut lengths) =
            (Vec::new(), Vec::new(), Vec::new(), vec![0u32]);
        for row in rows {
            let start = cumulative[row] as usize;
            let end = cumulative[row + 1] as usize;
            group_ids.extend_from_slice(&ids[start..end]);
            group_pos.extend_from_slice(&positions[start..end]);
            original.extend((start..end).map(|i| i as u32));
            lengths.push(group_ids.len() as u32);
        }
        let (local_ids, local_pos, local_scatter, local_fold) =
            radix_mlp::compute_fold_and_scatter(&group_ids, &group_pos, &lengths, None);
        let offset = compact_ids.len() as u32;
        for (i, compact) in local_scatter.into_iter().enumerate() {
            scatter[original[i] as usize] = offset + compact;
        }
        fold.extend(local_fold.into_iter().map(|i| original[i as usize]));
        compact_ids.extend(local_ids);
        compact_pos.extend(local_pos);
    }
    if let Some(pad) = pad {
        let padding = (pad - compact_ids.len() % pad) % pad;
        let size = compact_ids.len().checked_add(padding)?;
        if size > u32::MAX as usize {
            return None;
        }
        compact_ids.resize(size, 0);
        compact_pos.resize(compact_ids.len(), 0);
        fold.resize(compact_ids.len(), 0);
    }
    Some((compact_ids, compact_pos, scatter, fold))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn image(value: f32) -> Arc<MultimodalEncoding> {
        Arc::new(MultimodalEncoding {
            audios: vec![],
            images: vec![(
                1,
                Arc::new(ImagePatches {
                    pixels: vec![value; 6],
                    grid_thw: [1, 1, 2],
                    patch_dim: 3,
                    merge_size: 1,
                }),
            )],
            position_ids: std::array::from_fn(|_| vec![0, 1, 2, 3]),
            memory: None,
        })
    }
    fn assert_reconstruct(ids: &[u32], positions: &[u32], result: &Fold) {
        let (compact, pos, scatter, fold) = result;
        for (i, index) in scatter.iter().enumerate() {
            assert_eq!(compact[*index as usize], ids[i]);
            assert_eq!(pos[*index as usize], positions[i]);
            assert_eq!(ids[fold[*index as usize] as usize], ids[i]);
        }
    }
    #[test]
    fn interleaved_images_and_text_share_only_compatible_prefixes() {
        let a = image(0.);
        let b = image(1.);
        let media = vec![
            Some(a.clone()),
            Some(b.clone()),
            Some(a),
            Some(b),
            None,
            None,
        ];
        let ids: Vec<u32> = (0..6).flat_map(|i| [1, 9, 9, 10 + i]).collect();
        let positions: Vec<u32> = (0..6).flat_map(|_| [0, 1, 2, 3]).collect();
        let cumulative = [0, 4, 8, 12, 16, 20, 24];
        let result = compute(&ids, &positions, &cumulative, &media, Some(8)).unwrap();
        assert_eq!(result.0.len(), 16);
        assert_reconstruct(&ids, &positions, &result);
        assert_eq!(result.2[0], result.2[8]);
        assert_eq!(result.2[4], result.2[12]);
        assert_eq!(result.2[16], result.2[20]);
        assert_ne!(result.2[0], result.2[4]);
        assert_ne!(result.2[0], result.2[16]);
        assert!(MultimodalEncoding::allows_radix_fold(
            &media,
            &ids,
            &cumulative,
            &result.2,
            &result.3
        ));
        let mut wrong = result.2.clone();
        wrong[0] = result.2[4];
        assert!(!MultimodalEncoding::allows_radix_fold(
            &media,
            &ids,
            &cumulative,
            &wrong,
            &result.3
        ));
    }
    #[test]
    fn identical_content_shares_across_allocations_but_positions_do_not() {
        let a = image(0.);
        let same = image(0.);
        let b = image(1.);
        let ids = [1, 9, 9, 2, 1, 9, 9, 3, 1, 9, 9, 4];
        let pos = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3];
        let cu = [0, 4, 8, 12];
        let media = [Some(a.clone()), Some(same), Some(b)];
        let result = compute(&ids, &pos, &cu, &media, None).unwrap();
        assert_eq!(result.2[0], result.2[4]);
        assert_ne!(result.2[0], result.2[8]);
        assert_reconstruct(&ids, &pos, &result);
        let mut changed = MultimodalEncoding {
            audios: vec![],
            images: a.images.clone(),
            position_ids: a.position_ids.clone(),
            memory: None,
        };
        changed.position_ids[1][2] = 7;
        let media = [Some(a), Some(Arc::new(changed)), media[2].clone()];
        let result = compute(&ids, &pos, &cu, &media, None).unwrap();
        assert_ne!(result.2[0], result.2[4]);
        assert!(MultimodalEncoding::allows_radix_fold(
            &media, &ids, &cu, &result.2, &result.3
        ));
    }
    #[test]
    fn every_image_in_the_prefix_must_match() {
        let first = image(0.).images[0].1.clone();
        let a = image(1.).images[0].1.clone();
        let b = image(2.).images[0].1.clone();
        let context = |second: Arc<ImagePatches>| {
            Some(Arc::new(MultimodalEncoding {
                audios: vec![],
                images: vec![(1, first.clone()), (3, second)],
                position_ids: std::array::from_fn(|_| (0..6).collect()),
                memory: None,
            }))
        };
        let media = [context(a.clone()), context(b), context(a)];
        let ids = [1, 9, 9, 9, 9, 2, 1, 9, 9, 9, 9, 3, 1, 9, 9, 9, 9, 4];
        let positions: Vec<u32> = (0..3).flat_map(|_| 0..6).collect();
        let cu = [0, 6, 12, 18];
        let result = compute(&ids, &positions, &cu, &media, None).unwrap();
        assert_eq!(result.2[0], result.2[12]);
        assert_ne!(result.2[0], result.2[6]);
        assert_reconstruct(&ids, &positions, &result);
        assert!(MultimodalEncoding::allows_radix_fold(
            &media, &ids, &cu, &result.2, &result.3
        ));
        let mut wrong = result.2.clone();
        wrong[12] = result.2[6];
        assert!(!MultimodalEncoding::allows_radix_fold(
            &media, &ids, &cu, &wrong, &result.3
        ));
    }
    #[test]
    fn text_fast_path_matches_original_and_malformed_metadata_is_rejected() {
        let ids = [1, 2, 3, 1, 2, 4];
        let pos = [0, 1, 2, 0, 1, 2];
        let cu = [0, 3, 6];
        for pad in [None, Some(8)] {
            assert_eq!(
                compute(&ids, &pos, &cu, &[], pad).unwrap(),
                radix_mlp::compute_fold_and_scatter(&ids, &pos, &cu, pad)
            );
        }
        assert!(compute(&ids, &pos, &[0, 7, 6], &[], None).is_none());
        assert!(compute(&ids, &pos, &cu, &[], Some(0)).is_none());
        let short = [Some(image(0.)), Some(image(1.))];
        assert!(compute(&ids[..4], &pos[..4], &[0, 2, 4], &short, None).is_none());
        assert!(!MultimodalEncoding::allows_radix_fold(
            &short,
            &ids,
            &cu,
            &[99; 6],
            &[0]
        ));
    }
}
