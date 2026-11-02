// Host-side media geometry for Qwen3.5 vision-language roots.
//
// The original checkpoints ship two preprocessors:
//   * `preprocessor_config.json`  -> Qwen2VLImageProcessorFast (images)
//   * `video_preprocessor_config.json` -> Qwen3VLVideoProcessor (videos)
//
// Both resize to a multiple of `patch_size * merge_size`, flatten pixels into
// `[temporal_patch * channels * patch * patch]` rows, and order rows by merge
// group before the within-group pixels. The Native tower then consumes
// `grid_thw = [temporal_patches, height_patches, width_patches]`. This module
// reproduces that contract without reading image or video payloads.

use crate::qwen35_vl_config::{MediaPreprocessConfig, VisionConfig};
use infinity_v3_backend_core::BackendError;

fn start(error: impl std::fmt::Display) -> BackendError {
    BackendError::Start(format!("Oxide qwen3.5-vl media: {error}"))
}

fn product(left: usize, right: usize) -> Result<usize, BackendError> {
    left.checked_mul(right)
        .ok_or_else(|| start("dimension overflow"))
}

/// Qwen2VL `smart_resize`, including the factor of `patch * merge`.
///
/// Pixel bounds are total pixel counts (`shortest_edge`/`longest_edge`), not
/// side lengths. The function is the exact round/floor/ceil sequence used by
/// the pinned processor.
pub fn smart_resize(
    height: usize,
    width: usize,
    patch_size: usize,
    merge_size: usize,
    shortest_edge: usize,
    longest_edge: usize,
) -> Result<(usize, usize), BackendError> {
    if height == 0 || width == 0 {
        return Err(start("media height and width must be positive"));
    }
    let factor = product(patch_size, merge_size)?;
    if factor == 0 || height < factor || width < factor {
        return Err(start(format!(
            "media {height}x{width} is smaller than required factor {factor}"
        )));
    }
    if shortest_edge == 0 || longest_edge < shortest_edge {
        return Err(start("invalid media pixel bounds"));
    }
    let round_to_factor = |value: usize| -> Result<usize, BackendError> {
        let scaled = (value as f64 / factor as f64).round();
        let rounded = (scaled as usize)
            .checked_mul(factor)
            .ok_or_else(|| start("resized dimension overflow"))?;
        Ok(rounded.max(factor))
    };
    let mut resized_height = round_to_factor(height)?;
    let mut resized_width = round_to_factor(width)?;
    let total = product(resized_height, resized_width)?;
    if total > longest_edge {
        let beta = ((height as f64 * width as f64) / longest_edge as f64).sqrt();
        if !beta.is_finite() || beta <= 0.0 {
            return Err(start("invalid media resize ratio"));
        }
        resized_height = ((height as f64 / beta / factor as f64).floor() as usize)
            .max(1)
            .checked_mul(factor)
            .ok_or_else(|| start("resized height overflow"))?;
        resized_width = ((width as f64 / beta / factor as f64).floor() as usize)
            .max(1)
            .checked_mul(factor)
            .ok_or_else(|| start("resized width overflow"))?;
    } else if total < shortest_edge {
        let beta = (shortest_edge as f64 / (height as f64 * width as f64)).sqrt();
        if !beta.is_finite() || beta <= 0.0 {
            return Err(start("invalid media resize ratio"));
        }
        resized_height = ((height as f64 * beta / factor as f64).ceil() as usize)
            .max(1)
            .checked_mul(factor)
            .ok_or_else(|| start("resized height overflow"))?;
        resized_width = ((width as f64 * beta / factor as f64).ceil() as usize)
            .max(1)
            .checked_mul(factor)
            .ok_or_else(|| start("resized width overflow"))?;
    }
    Ok((resized_height, resized_width))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MediaGrid {
    pub temporal: usize,
    pub height: usize,
    pub width: usize,
    pub merge: usize,
}

impl MediaGrid {
    pub fn new(
        temporal: usize,
        height: usize,
        width: usize,
        merge: usize,
    ) -> Result<Self, BackendError> {
        if temporal == 0 || height == 0 || width == 0 || merge == 0 {
            return Err(start("media grid dimensions must be positive"));
        }
        if !height.is_multiple_of(merge) || !width.is_multiple_of(merge) {
            return Err(start("media patch grid is not divisible by merge"));
        }
        Ok(Self {
            temporal,
            height,
            width,
            merge,
        })
    }

    pub fn patch_rows(&self) -> Result<usize, BackendError> {
        product(product(self.temporal, self.height)?, self.width)
    }

    pub fn merged_tokens(&self) -> Result<usize, BackendError> {
        let area = product(self.merge, self.merge)?;
        self.patch_rows()?
            .checked_div(area)
            .filter(|count| *count > 0)
            .ok_or_else(|| start("zero merged media tokens"))
    }

    pub fn grid_thw(&self) -> [usize; 3] {
        [self.temporal, self.height, self.width]
    }

    /// Merge-group-major patch index -> spatial `(row, column)` inside a frame.
    /// This is exactly `merge_major_coords` from the Qwen3-VL media foundation.
    pub fn merge_major_coords(&self, index: usize) -> Result<(usize, usize), BackendError> {
        let frame_patches = product(self.height, self.width)?;
        if index >= frame_patches {
            return Err(start("merge-major index exceeds a media frame"));
        }
        let area = product(self.merge, self.merge)?;
        let column = index % self.merge + ((index / area) % (self.width / self.merge)) * self.merge;
        let row = (index / self.merge) % self.merge
            + ((index / (self.merge * self.width)) % (self.height / self.merge)) * self.merge;
        Ok((row, column))
    }

    /// Map one element of a packed patch row back to `(temporal, channel, y, x)`
    /// in a canonical `[frames, channels, height, width]` source tensor.
    ///
    /// `patch_element` is in `0..channels * temporal_patch * patch * patch`,
    /// where the temporal patch is the innermost spatial group. The ordered
    /// frame inside a temporal patch is `0..temporal_patch_size`.
    pub fn source_coordinate(
        &self,
        patch_index: usize,
        patch_element: usize,
        channels: usize,
        temporal_patch: usize,
        patch: usize,
    ) -> Result<(usize, usize, usize, usize), BackendError> {
        let frame_patches = product(self.height, self.width)?;
        let temporal_index = patch_index / frame_patches;
        let spatial_index = patch_index % frame_patches;
        let (row, column) = self.merge_major_coords(spatial_index)?;
        let patch_area = product(patch, patch)?;
        let per_channel = product(temporal_patch, patch_area)?;
        if patch_element >= product(channels, per_channel)? {
            return Err(start("patch element exceeds channels * temporal * patch^2"));
        }
        let channel = patch_element / per_channel;
        let within_channel = patch_element % per_channel;
        let within_temporal = within_channel / patch_area;
        let within_patch = within_channel % patch_area;
        let patch_row = within_patch / patch;
        let patch_column = within_patch % patch;
        let y = product(row, patch)? + patch_row;
        let x = product(column, patch)? + patch_column;
        let frame = product(temporal_index, temporal_patch)? + within_temporal;
        Ok((frame, channel, y, x))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImagePlan {
    pub grid: MediaGrid,
    pub resized_height: usize,
    pub resized_width: usize,
}

/// One still image: `grid_thw = [1, h/patch, w/patch]`.
pub fn plan_image(
    height: usize,
    width: usize,
    media: &MediaPreprocessConfig,
    vision: &VisionConfig,
) -> Result<ImagePlan, BackendError> {
    let (resized_height, resized_width) = smart_resize(
        height,
        width,
        media.patch_size,
        media.merge_size,
        media.image_shortest_edge,
        media.image_longest_edge,
    )?;
    let grid = MediaGrid::new(
        1,
        resized_height / media.patch_size,
        resized_width / media.patch_size,
        media.merge_size,
    )?;
    vision.merged_tokens(grid.grid_thw())?;
    Ok(ImagePlan {
        grid,
        resized_height,
        resized_width,
    })
}

/// One video clip. `temporal_patches = frames / temporal_patch_size`; a video
/// with an odd frame count is rejected instead of silently duplicating a frame
/// the original processor would have grouped differently.
pub fn plan_video(
    height: usize,
    width: usize,
    frames: usize,
    media: &MediaPreprocessConfig,
    vision: &VisionConfig,
) -> Result<ImagePlan, BackendError> {
    if frames == 0 || !frames.is_multiple_of(media.temporal_patch_size) {
        return Err(start(format!(
            "video clip has {frames} frames; temporal patch {} requires an even count",
            media.temporal_patch_size
        )));
    }
    let (resized_height, resized_width) = smart_resize(
        height,
        width,
        media.patch_size,
        media.merge_size,
        media.video_shortest_edge,
        media.video_longest_edge,
    )?;
    let grid = MediaGrid::new(
        frames / media.temporal_patch_size,
        resized_height / media.patch_size,
        resized_width / media.patch_size,
        media.merge_size,
    )?;
    vision.merged_tokens(grid.grid_thw())?;
    Ok(ImagePlan {
        grid,
        resized_height,
        resized_width,
    })
}

/// Mirror of `helpers/vision_batch.h`: include FP32 rotary/normalization
/// temporaries and gated MLP buffers, and never split a single oversized image.
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

/// Pinned text-side markers from the original configs. Media spans are encoded
/// by replacing `image_token_id` / `video_token_id` rows with merged features.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MediaKind {
    Image,
    Video,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MediaSpan {
    pub kind: MediaKind,
    pub token_start: usize,
    pub token_count: usize,
    pub grid: MediaGrid,
}
