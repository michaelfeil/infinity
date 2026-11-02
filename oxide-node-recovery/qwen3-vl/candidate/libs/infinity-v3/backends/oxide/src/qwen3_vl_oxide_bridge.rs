//! Integration seam between the cold Qwen3-VL plans and existing Oxide
//! primitives. This module intentionally reuses `TensorRef`/`TensorMut`,
//! `math::LinearPlan`, FA4 `AttentionSpec`/`AttentionBatch`, `staging::HostIo`
//! and `upload::WeightUploader`; it does not fork the Dense executor.

use crate::fa4::{AttentionBatch, AttentionSpec, DType, Scale, Window};
use crate::math::LinearPlan;
use crate::model::WeightSource;
use crate::qwen3_vl_config::Qwen3VlConfig;
use crate::qwen3_vl_enqueue::VisionBatch;
use crate::staging::{HostIo, SlotPool};
use crate::tensor::{Bf16, Element, Layout, TensorMut, TensorRef};
use crate::upload::WeightUploader;
use cuda_core::{Context, DeviceBuffer};
use infinity_v3_backend_core::BackendError;
use std::sync::Arc;

fn start(error: impl std::fmt::Display) -> BackendError {
    BackendError::Start(format!("Oxide qwen3-vl bridge: {error}"))
}

/// Real BF16 linear plans for one prepared vision batch. Construction is cold
/// and uses the existing checked GEMM planner; it never queries or launches.
pub(crate) struct VisionLinearPlans {
    pub(crate) patch_embed: LinearPlan<Bf16>,
    pub(crate) qkv: LinearPlan<Bf16>,
    pub(crate) projection: LinearPlan<Bf16>,
    pub(crate) mlp_fc1: LinearPlan<Bf16>,
    pub(crate) mlp_fc2: LinearPlan<Bf16>,
    pub(crate) merger_fc1: LinearPlan<Bf16>,
    pub(crate) merger_fc2: LinearPlan<Bf16>,
}

impl VisionLinearPlans {
    pub(crate) fn prepare(
        config: &Qwen3VlConfig,
        patch_rows: usize,
        merged_rows: usize,
    ) -> Result<Self, BackendError> {
        let rows = patch_rows.max(1);
        let merged_rows = merged_rows.max(1);
        let hidden = config.vision.hidden_size;
        let intermediate = config.vision.intermediate_size;
        let grouped = hidden
            .checked_mul(config.vision.merge_area()?)
            .ok_or_else(|| start("merged vision width overflow"))?;
        let plan = |input: usize, output: usize| {
            LinearPlan::<Bf16>::contiguous(rows, input, output).map_err(start)
        };
        let merged_plan = |input: usize, output: usize| {
            LinearPlan::<Bf16>::contiguous(merged_rows, input, output).map_err(start)
        };
        Ok(Self {
            patch_embed: plan(config.vision.patch_dim()?, hidden)?,
            qkv: plan(
                hidden,
                hidden.checked_mul(3).ok_or_else(|| start("qkv width"))?,
            )?,
            projection: plan(hidden, hidden)?,
            mlp_fc1: plan(hidden, intermediate)?,
            mlp_fc2: plan(intermediate, hidden)?,
            merger_fc1: merged_plan(grouped, grouped)?,
            merger_fc2: merged_plan(grouped, config.vision.out_hidden_size)?,
        })
    }
}

/// Non-causal global vision attention spec. `kv_heads == heads` because the
/// published Qwen3-VL vision tower has no GQA.
pub(crate) fn vision_attention_spec(config: &Qwen3VlConfig) -> AttentionSpec {
    AttentionSpec {
        heads: config.vision.num_heads,
        kv_heads: config.vision.num_heads,
        head_dim: config.vision.head_dim(),
        causal: false,
        window: Window::Global,
        dtype: DType::Bf16,
        scale: Scale::RsqrtHeadDim,
    }
}

/// Frame-isolated packed attention arguments for one prepared batch. The caller
/// supplies device pointers; this function allocates nothing and submits
/// nothing. `cu_seqlens` is the batch-local per-frame offset array.
pub(crate) fn frame_attention_batch(
    config: &Qwen3VlConfig,
    batch: &VisionBatch,
    q: u64,
    k: u64,
    v: u64,
    out: u64,
    q_offsets: u64,
) -> AttentionBatch {
    let heads = config.vision.num_heads;
    let dim = config.vision.head_dim();
    let stride = [
        (heads * dim) as i64,
        dim as i64,
        1,
        (heads * dim) as i64,
        dim as i64,
        1,
        (heads * dim) as i64,
        dim as i64,
        1,
        (heads * dim) as i64,
        dim as i64,
        1,
    ];
    AttentionBatch {
        q,
        k,
        v,
        out,
        q_offsets,
        k_offsets: q_offsets,
        q_tokens: batch.patch_rows,
        k_tokens: batch.patch_rows,
        batch: batch.cu_seqlens.len().saturating_sub(1),
        max_q: batch.max_frame,
        strides: stride,
    }
}

/// Reuse the existing host staging pool rather than allocating a Qwen-specific
/// synchronous scratch path.
pub(crate) fn vision_host_io(
    context: &Arc<Context>,
) -> Result<Arc<SlotPool<HostIo>>, BackendError> {
    HostIo::pool(context)
}

/// Reuse the existing `WeightUploader` (two pinned slots, recorded events, no
/// hot allocation after construction) for the tower tensor list.
pub(crate) struct VisionWeightUploader<'a> {
    inner: &'a mut WeightUploader,
}

impl<'a> VisionWeightUploader<'a> {
    pub(crate) fn new(inner: &'a mut WeightUploader) -> Self {
        Self { inner }
    }

    pub(crate) fn upload(
        &mut self,
        sources: &[&WeightSource],
    ) -> Result<DeviceBuffer<Bf16>, BackendError> {
        self.inner.upload(sources)
    }

    pub(crate) fn finish(&self) -> Result<(), BackendError> {
        self.inner.finish()
    }
}

/// Borrowed checked row views over existing device buffers.
pub(crate) fn checked_rows<'a, E: Element>(
    buffer: &'a DeviceBuffer<E::Storage>,
    rows: usize,
    columns: usize,
) -> Result<TensorRef<'a, E, 2>, crate::tensor::TensorError> {
    TensorRef::new(buffer, Layout::contiguous([rows, columns])?)
}

pub(crate) fn checked_rows_mut<'a, E: Element>(
    buffer: &'a mut DeviceBuffer<E::Storage>,
    rows: usize,
    columns: usize,
) -> Result<TensorMut<'a, E, 2>, crate::tensor::TensorError> {
    TensorMut::new(buffer, Layout::contiguous([rows, columns])?)
}
