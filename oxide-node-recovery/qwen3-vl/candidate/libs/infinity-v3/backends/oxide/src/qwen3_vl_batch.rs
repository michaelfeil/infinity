// Packed `Batch` -> concrete Qwen3-VL forward plumbing.
//
// This is the missing factory boundary between the shared engine `Batch` /
// `MultimodalEncoding` contract and the concrete `Qwen3VlExecutor`. It keeps the
// original checkpoint config, token ids, cumulative boundaries, prepared pixel
// rows and per-sequence media positions. Nothing is synthesized: a media row
// must already contain the real processor grid/pixels.

use crate::checkpoint::start;
use crate::qwen3_vl_device::{Qwen3VlForward, VisionImage};
use infinity_v3_backend_core::{BackendError, Batch, Pool};

#[derive(Clone, Debug)]
pub struct PackedVlRequest {
    pub forward: Qwen3VlForward,
    pub sequences: usize,
    pub tokens: usize,
    pub has_media: bool,
}

/// Convert one packed engine batch into the concrete VL request.
///
/// `raw_tokens` selects row-major token states (raw / multi-vector wrappers).
/// Otherwise `pool` selects the per-sequence pooled vector.
pub fn pack_batch(
    batch: &Batch,
    pool: Pool,
    raw_tokens: bool,
) -> Result<PackedVlRequest, BackendError> {
    let sequences = batch
        .cumulative_seq_lengths
        .len()
        .checked_sub(1)
        .ok_or_else(|| start("packed VL batch is missing cumulative boundaries"))?;
    let tokens = batch.input_ids.len();
    if sequences == 0
        || batch.cumulative_seq_lengths.first().copied() != Some(0)
        || batch.cumulative_seq_lengths.last().copied() != Some(tokens as u32)
        || batch
            .cumulative_seq_lengths
            .windows(2)
            .any(|pair| pair[0] > pair[1])
    {
        return Err(start("invalid packed cumulative boundaries"));
    }
    if !batch.multimodal.is_empty() && batch.multimodal.len() != sequences {
        return Err(start(
            "multimodal entries must be empty or one per packed sequence",
        ));
    }
    if batch.input_ids.len() != batch.position_ids.len() {
        return Err(start("packed ids and positions disagree"));
    }

    let mut images = Vec::new();
    let mut positions: [Vec<u32>; 3] = std::array::from_fn(|_| Vec::with_capacity(tokens));
    let mut has_positions = false;
    let mut has_media = false;
    for (row, pair) in batch.cumulative_seq_lengths.windows(2).enumerate() {
        let start_token = pair[0] as usize;
        let end_token = pair[1] as usize;
        let length = end_token - start_token;
        let Some(media) = batch.multimodal.get(row).and_then(Option::as_ref) else {
            // A text row in a partially multimodal batch still contributes its
            // own local positions; the three axes coincide for text tokens.
            if batch.multimodal.iter().any(Option::is_some) {
                let local = &batch.position_ids[start_token..end_token];
                for axis in &mut positions {
                    axis.extend_from_slice(local);
                }
                has_positions = true;
            }
            continue;
        };
        if !media.audios.is_empty() {
            return Err(start(
                "Qwen3-VL audio is unsupported; reject audio before any GPU submission",
            ));
        }
        if media.position_ids.iter().any(|axis| axis.len() != length) {
            return Err(start(format!(
                "sequence {row} media positions do not cover its {length} tokens"
            )));
        }
        has_positions = true;
        for (target, axis) in positions.iter_mut().zip(&media.position_ids) {
            target.extend_from_slice(axis);
        }
        for (index, (image_start, image)) in media.images.iter().enumerate() {
            let token_start = start_token
                .checked_add(*image_start)
                .ok_or_else(|| start("image token start overflow"))?;
            if image.merge_size == 0 || image.grid_thw.contains(&0) {
                return Err(start(format!(
                    "sequence {row} image {index} has an empty grid or merge size"
                )));
            }
            let merge_area = image
                .merge_size
                .checked_mul(image.merge_size)
                .filter(|value| *value > 0)
                .ok_or_else(|| start("image merge area overflow"))?;
            let patch_rows = image
                .grid_thw
                .iter()
                .try_fold(1usize, |value, dim| value.checked_mul(*dim))
                .ok_or_else(|| start("image patch row count overflow"))?;
            if !patch_rows.is_multiple_of(merge_area)
                || !image.grid_thw[1].is_multiple_of(image.merge_size)
                || !image.grid_thw[2].is_multiple_of(image.merge_size)
            {
                return Err(start(format!(
                    "sequence {row} image {index} grid is not divisible by merge size"
                )));
            }
            let token_count = patch_rows / merge_area;
            if token_count == 0
                || token_start
                    .checked_add(token_count)
                    .is_none_or(|end| end > end_token)
            {
                return Err(start(format!(
                    "sequence {row} image {index} span exceeds the packed sequence"
                )));
            }
            let expected_pixels = patch_rows
                .checked_mul(image.patch_dim)
                .ok_or_else(|| start("image patch payload size overflow"))?;
            if image.pixels.len() != expected_pixels {
                return Err(start(format!(
                    "sequence {row} image {index} payload has {} values, expected {expected_pixels}",
                    image.pixels.len()
                )));
            }
            images.push(VisionImage {
                name: format!("seq{row}-media{index}"),
                pixels: image.pixels.clone(),
                grid_thw: image.grid_thw,
                merge_size: image.merge_size,
                token_start,
            });
            has_media = true;
        }
    }
    if has_positions && positions.iter().any(|axis| axis.len() != tokens) {
        return Err(start(
            "packed media positions do not cover every token when media is present",
        ));
    }
    images.sort_unstable_by_key(|image| image.token_start);
    for pair in images.windows(2) {
        let end = pair[0]
            .token_start
            .checked_add(pair[0].merged_rows()?)
            .ok_or_else(|| start("packed media span overflow"))?;
        if end > pair[1].token_start {
            return Err(start("packed media token spans overlap"));
        }
    }

    let media_positions = if has_positions { Some(positions) } else { None };
    Ok(PackedVlRequest {
        forward: Qwen3VlForward {
            ids: batch.input_ids.clone(),
            cumulative: batch.cumulative_seq_lengths.clone(),
            images,
            media_positions,
            pool,
            raw_tokens,
        },
        sequences,
        tokens,
        has_media,
    })
}
