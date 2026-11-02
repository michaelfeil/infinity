// Structural vision-tower preparation shared by the full and tower-only paths.
//
// Every function is pure host arithmetic. It reproduces the actual Native
// pre-processing order: merge-group-major patch rows, four-corner learned
// position interpolation, 2-D split-half rotary phases, per-frame attention
// isolation, equal-length frame batching for multi-vector mode, and the
// workspace-driven image batching heuristic from `helpers/vision_batch.h`.

use crate::checkpoint::start;
use infinity_v3_backend_core::BackendError;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImageGrid {
    pub frames: usize,
    pub height: usize,
    pub width: usize,
    pub merge_size: usize,
    pub token_start: usize,
    pub token_count: usize,
}

impl ImageGrid {
    pub fn new(
        frames: usize,
        height: usize,
        width: usize,
        merge_size: usize,
        token_start: usize,
        token_count: usize,
    ) -> Result<Self, BackendError> {
        if frames == 0 || height == 0 || width == 0 || merge_size == 0 {
            return Err(start("image grid dimensions must be positive"));
        }
        if !height.is_multiple_of(merge_size) || !width.is_multiple_of(merge_size) {
            return Err(start("image grid is not divisible by merge size"));
        }
        let merged = merged_count(frames, height, width, merge_size)?;
        if token_count != merged {
            return Err(start(format!(
                "image span declares {token_count} tokens but grid {frames}x{height}x{width} with merge {merge_size} requires {merged}"
            )));
        }
        Ok(Self {
            frames,
            height,
            width,
            merge_size,
            token_start,
            token_count,
        })
    }

    pub fn patch_rows(&self) -> Result<usize, BackendError> {
        self.frames
            .checked_mul(self.height)
            .and_then(|value| value.checked_mul(self.width))
            .ok_or_else(|| start("image patch rows overflow"))
    }
}

pub fn merged_count(
    frames: usize,
    height: usize,
    width: usize,
    merge: usize,
) -> Result<usize, BackendError> {
    if frames == 0 || height == 0 || width == 0 || merge == 0 {
        return Err(start("image grid dimensions must be positive"));
    }
    if !height.is_multiple_of(merge) || !width.is_multiple_of(merge) {
        return Err(start("image grid is not divisible by merge size"));
    }
    let patches = frames
        .checked_mul(height)
        .and_then(|value| value.checked_mul(width))
        .ok_or_else(|| start("image patch count overflow"))?;
    let area = merge
        .checked_mul(merge)
        .ok_or_else(|| start("merge area overflow"))?;
    patches
        .checked_div(area)
        .filter(|count| *count > 0)
        .ok_or_else(|| start("image merge count is zero"))
}

/// Recover the original spatial `(row, column)` from the processor's
/// merge-group-major patch index. `index` is relative to the frame, matching
/// the Native modulo recovery after subtracting the frame offset.
pub fn merge_major_coords(
    index: usize,
    merge: usize,
    height: usize,
    width: usize,
) -> Result<(usize, usize), BackendError> {
    if merge == 0 || height == 0 || width == 0 {
        return Err(start("invalid merge-major geometry"));
    }
    if !height.is_multiple_of(merge) || !width.is_multiple_of(merge) {
        return Err(start("merge-major geometry is not divisible by merge"));
    }
    if index >= height * width {
        return Err(start("merge-major index exceeds frame"));
    }
    let merge_area = merge
        .checked_mul(merge)
        .ok_or_else(|| start("merge area overflow"))?;
    let width_blocks = width / merge;
    let height_blocks = height / merge;
    let column = index % merge + ((index / merge_area) % width_blocks) * merge;
    let row = (index / merge) % merge + ((index / (merge * width)) % height_blocks) * merge;
    Ok((row, column))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PositionCorner {
    pub row: f32,
    pub column: f32,
    pub r0: usize,
    pub r1: usize,
    pub c0: usize,
    pub c1: usize,
    pub dr: f32,
    pub dc: f32,
}

pub fn position_interpolation(
    side: usize,
    height: usize,
    width: usize,
    merge: usize,
) -> Result<Vec<PositionCorner>, BackendError> {
    if side == 0 {
        return Err(start("position table side must be positive"));
    }
    let rows = height
        .checked_mul(width)
        .ok_or_else(|| start("position row count overflow"))?;
    let row_scale = if height == 1 {
        0.0
    } else {
        (side - 1) as f32 / (height - 1) as f32
    };
    let column_scale = if width == 1 {
        0.0
    } else {
        (side - 1) as f32 / (width - 1) as f32
    };
    let mut out = Vec::with_capacity(rows);
    for index in 0..rows {
        let (row, column) = merge_major_coords(index, merge, height, width)?;
        let row_f = row as f32 * row_scale;
        let column_f = column as f32 * column_scale;
        let r0 = row_f.floor() as usize;
        let c0 = column_f.floor() as usize;
        let r1 = (row_f.ceil() as usize).min(side - 1);
        let c1 = (column_f.ceil() as usize).min(side - 1);
        out.push(PositionCorner {
            row: row_f,
            column: column_f,
            r0,
            r1,
            c0,
            c1,
            dr: row_f - r0 as f32,
            dc: column_f - c0 as f32,
        });
    }
    Ok(out)
}

/// `freq[j] = 1 / 10000^(2j/(dim/2))` for `j < dim/4` (Native vision table).
pub fn vision_frequency(dim: usize) -> Result<Vec<f32>, BackendError> {
    if dim == 0 || !dim.is_multiple_of(4) {
        return Err(start("vision head dimension must be divisible by 4"));
    }
    let count = dim / 4;
    let mut out = Vec::with_capacity(count);
    for j in 0..count {
        out.push(1.0f32 / 10_000f32.powf(2.0 * j as f32 / (dim as f32 / 2.0)));
    }
    Ok(out)
}

pub fn vision_angles(dim: usize, row: f32, column: f32) -> Result<Vec<f32>, BackendError> {
    let frequency = vision_frequency(dim)?;
    let mut angles = Vec::with_capacity(frequency.len() * 2);
    for value in &frequency {
        angles.push(row * value);
    }
    for value in &frequency {
        angles.push(column * value);
    }
    Ok(angles)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FramePlan {
    pub cu_seqlens: Vec<i32>,
    pub frame_lengths: Vec<usize>,
    pub image_frame_spans: Vec<(usize, usize)>,
    pub merged_counts: Vec<usize>,
    pub max_frame: usize,
}

pub fn frame_plan(images: &[ImageGrid]) -> Result<FramePlan, BackendError> {
    let mut cu_seqlens = Vec::with_capacity(images.len() + 1);
    cu_seqlens.push(0i32);
    let mut frame_lengths = Vec::new();
    let mut image_frame_spans = Vec::with_capacity(images.len());
    let mut merged_counts = Vec::with_capacity(images.len());
    let mut max_frame = 0usize;
    for image in images {
        let frame_length = image
            .height
            .checked_mul(image.width)
            .ok_or_else(|| start("frame length overflow"))?;
        let first_frame = frame_lengths.len();
        for _ in 0..image.frames {
            let next = (*cu_seqlens.last().unwrap_or(&0) as usize)
                .checked_add(frame_length)
                .ok_or_else(|| start("frame offset overflow"))?;
            if next > i32::MAX as usize {
                return Err(start("vision frame offsets exceed i32"));
            }
            cu_seqlens.push(next as i32);
            frame_lengths.push(frame_length);
        }
        image_frame_spans.push((first_frame, frame_lengths.len()));
        merged_counts.push(merged_count(
            image.frames,
            image.height,
            image.width,
            image.merge_size,
        )?);
        max_frame = max_frame.max(frame_length);
    }
    Ok(FramePlan {
        cu_seqlens,
        frame_lengths,
        image_frame_spans,
        merged_counts,
        max_frame,
    })
}

/// Equal-length frame grouping used only by the multi-vector fallback.
pub fn frame_groups(plan: &FramePlan) -> BTreeMap<usize, Vec<usize>> {
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (frame, length) in plan.frame_lengths.iter().enumerate() {
        groups.entry(*length).or_default().push(frame);
    }
    groups
}

/// Mirror of `vision_batches` with a caller-supplied byte budget. A single
/// oversized image always forms its own batch and image order is preserved.
pub fn vision_batches(
    patch_rows: &[usize],
    hidden: usize,
    intermediate: usize,
    budget_bytes: u64,
) -> Result<Vec<(usize, usize)>, BackendError> {
    if hidden == 0 || intermediate == 0 {
        return Err(start("vision batching geometry must be positive"));
    }
    let per_row = 4u64
        .checked_mul(
            (8u64
                .checked_mul(hidden as u64)
                .ok_or_else(|| start("vision batching hidden overflow"))?)
            .checked_add(
                2u64.checked_mul(intermediate as u64)
                    .ok_or_else(|| start("vision batching width overflow"))?,
            )
            .ok_or_else(|| start("vision batching row estimate overflow"))?,
        )
        .ok_or_else(|| start("vision batching row estimate overflow"))?;
    let limit = usize::try_from((budget_bytes / per_row).max(1)).map_err(start)?;
    let mut batches = Vec::new();
    let mut batch_start = 0usize;
    let mut patches = 0usize;
    for (index, count) in patch_rows.iter().enumerate() {
        if index > batch_start && (patches >= limit || *count > limit.saturating_sub(patches)) {
            batches.push((batch_start, index));
            batch_start = index;
            patches = 0;
        }
        patches = patches
            .checked_add(*count)
            .ok_or_else(|| start("vision batch patch count overflow"))?;
    }
    if batch_start < patch_rows.len() {
        batches.push((batch_start, patch_rows.len()));
    }
    Ok(batches)
}

pub fn image_feature_slices(images: &[ImageGrid]) -> Result<Vec<(usize, usize)>, BackendError> {
    let mut offset = 0usize;
    let mut slices = Vec::with_capacity(images.len());
    for image in images {
        slices.push((offset, image.token_count));
        offset = offset
            .checked_add(image.token_count)
            .ok_or_else(|| start("merged feature offset overflow"))?;
    }
    Ok(slices)
}

/// Deepstack tensors are collected in increasing tower layer order even if the
/// config list is unsorted; `deepstack_merger_list.i` is keyed by list position.
pub fn deepstack_capture_order(
    indexes: &[usize],
    depth: usize,
) -> Result<Vec<(usize, usize)>, BackendError> {
    let mut seen = std::collections::BTreeSet::new();
    let mut order = Vec::with_capacity(indexes.len());
    for (name_index, layer) in indexes.iter().enumerate() {
        if *layer >= depth {
            return Err(start(format!(
                "deepstack layer {layer} is outside vision depth {depth}"
            )));
        }
        if !seen.insert(*layer) {
            return Err(start(format!("duplicate deepstack layer {layer}")));
        }
        order.push((*layer, name_index));
    }
    order.sort_unstable_by_key(|(layer, _)| *layer);
    Ok(order)
}
