// Owned prepared vision execution plan.
//
// Preparation is cold: it computes frame isolation, batch ranges and the fixed
// stage sequence once. `enqueue` walks that immutable plan and performs no heap
// allocation, no device query, no readback and no synchronization. The sink is
// the only entity allowed to submit device work, so a CPU test can verify the
// exact stage order without claiming a GPU pass.

use crate::checkpoint::start;
use crate::qwen3_vl_config::Qwen3VlConfig;
use crate::qwen3_vl_vision::{
    deepstack_capture_order, frame_plan, vision_batches, FramePlan, ImageGrid,
};
use infinity_v3_backend_core::BackendError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BlockStage {
    Norm1,
    Qkv,
    Rotary,
    Attention,
    Projection,
    Residual1,
    Norm2,
    MlpFc1,
    MlpActivation,
    MlpFc2,
    Residual2,
}

pub const BLOCK_STAGES: [BlockStage; 11] = [
    BlockStage::Norm1,
    BlockStage::Qkv,
    BlockStage::Rotary,
    BlockStage::Attention,
    BlockStage::Projection,
    BlockStage::Residual1,
    BlockStage::Norm2,
    BlockStage::MlpFc1,
    BlockStage::MlpActivation,
    BlockStage::MlpFc2,
    BlockStage::Residual2,
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VisionBatch {
    pub image_start: usize,
    pub image_end: usize,
    pub patch_rows: usize,
    pub merged_rows: usize,
    pub cu_seqlens: Vec<i32>,
    pub max_frame: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedVision {
    pub images: Vec<ImageGrid>,
    pub batches: Vec<VisionBatch>,
    pub deepstack: Vec<(usize, usize)>,
    pub depth: usize,
    pub total_patch_rows: usize,
    pub total_merged_rows: usize,
}

pub trait VisionSink {
    type Error;
    fn patch_embed(&mut self, batch: &VisionBatch) -> Result<(), Self::Error>;
    fn position_rotary(&mut self, batch: &VisionBatch) -> Result<(), Self::Error>;
    fn block(
        &mut self,
        batch: &VisionBatch,
        layer: usize,
        stage: BlockStage,
    ) -> Result<(), Self::Error>;
    fn deepstack(
        &mut self,
        batch: &VisionBatch,
        layer: usize,
        name_index: usize,
    ) -> Result<(), Self::Error>;
    fn merger(&mut self, batch: &VisionBatch) -> Result<(), Self::Error>;
}

fn batch_frame_offsets(
    plan: &FramePlan,
    image_start: usize,
    image_end: usize,
) -> Result<(Vec<i32>, usize, usize), BackendError> {
    let mut offsets = Vec::with_capacity(1);
    offsets.push(0i32);
    let mut max_frame = 0usize;
    let mut merged = 0usize;
    for image in image_start..image_end {
        let (first_frame, end_frame) = plan.image_frame_spans[image];
        for frame in first_frame..end_frame {
            let length = plan.frame_lengths[frame];
            let next = (*offsets.last().unwrap_or(&0) as usize)
                .checked_add(length)
                .ok_or_else(|| start("batch frame offset overflow"))?;
            if next > i32::MAX as usize {
                return Err(start("batch frame offsets exceed i32"));
            }
            offsets.push(next as i32);
            max_frame = max_frame.max(length);
        }
        merged = merged
            .checked_add(plan.merged_counts[image])
            .ok_or_else(|| start("batch merged row overflow"))?;
    }
    Ok((offsets, max_frame, merged))
}

impl PreparedVision {
    pub fn prepare(
        config: &Qwen3VlConfig,
        images: &[ImageGrid],
        budget_bytes: u64,
    ) -> Result<Self, BackendError> {
        if images.is_empty() {
            return Err(start("prepared vision requires at least one image"));
        }
        let plan = frame_plan(images)?;
        let mut patch_rows = Vec::with_capacity(images.len());
        for image in images {
            if image.merge_size != config.vision.spatial_merge_size {
                return Err(start(format!(
                    "image merge size {} does not match config merge {}",
                    image.merge_size, config.vision.spatial_merge_size
                )));
            }
            patch_rows.push(image.patch_rows()?);
        }
        let ranges = vision_batches(
            &patch_rows,
            config.vision.hidden_size,
            config.vision.intermediate_size,
            budget_bytes,
        )?;
        let mut batches = Vec::with_capacity(ranges.len());
        let mut total_patch_rows = 0usize;
        let mut total_merged_rows = 0usize;
        for (image_start, image_end) in ranges {
            let patch_rows =
                patch_rows[image_start..image_end]
                    .iter()
                    .try_fold(0usize, |total, count| {
                        total
                            .checked_add(*count)
                            .ok_or_else(|| start("batch patch rows overflow"))
                    })?;
            let (cu_seqlens, max_frame, merged_rows) =
                batch_frame_offsets(&plan, image_start, image_end)?;
            total_patch_rows = total_patch_rows
                .checked_add(patch_rows)
                .ok_or_else(|| start("total patch rows overflow"))?;
            total_merged_rows = total_merged_rows
                .checked_add(merged_rows)
                .ok_or_else(|| start("total merged rows overflow"))?;
            batches.push(VisionBatch {
                image_start,
                image_end,
                patch_rows,
                merged_rows,
                cu_seqlens,
                max_frame,
            });
        }
        let deepstack =
            deepstack_capture_order(&config.vision.deepstack_visual_indexes, config.vision.depth)?;
        Ok(Self {
            images: images.to_vec(),
            batches,
            deepstack,
            depth: config.vision.depth,
            total_patch_rows,
            total_merged_rows,
        })
    }

    pub fn stage_calls(&self) -> usize {
        self.batches.len() * (2 + self.depth * BLOCK_STAGES.len() + self.deepstack.len() + 1)
    }

    /// Walk the immutable plan. All scratch is owned by the sink; this method
    /// performs no allocation, no query, no synchronization and no CPU fallback.
    pub fn enqueue<S: VisionSink>(&self, sink: &mut S) -> Result<(), S::Error> {
        for batch in &self.batches {
            sink.patch_embed(batch)?;
            sink.position_rotary(batch)?;
            for layer in 0..self.depth {
                for stage in BLOCK_STAGES {
                    sink.block(batch, layer, stage)?;
                }
                for &(capture_layer, name_index) in &self.deepstack {
                    if capture_layer == layer {
                        sink.deepstack(batch, layer, name_index)?;
                    }
                }
            }
            sink.merger(batch)?;
        }
        Ok(())
    }
}
