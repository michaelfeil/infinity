// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Prepared layer scheduling over three phase-reused BF16 arenas.
//! This leaf submits into the outer engine's already-bound stream; it never
//! owns a completion boundary or creates per-layer temporary allocations.
use crate::cublas::Blas;
use crate::fa4::{AttentionBatch, Fa4};
use crate::gdn_prepared::{GdnBufferView, GdnInputViews, GdnPrepared, GdnState};
use crate::math::{GemmArguments, LinearPlan};
use crate::qwen35_block_attention::PreparedBlockPlan;
use crate::qwen35_block_layout::Qwen35BlockLayout;
use crate::qwen35_layout::Qwen35Layout;
use crate::qwen35_prepared::{NormDispatch, Qwen35Kernels};
use crate::qwen35_weights::{Qwen35DeviceAttention, Qwen35DeviceWeights};
use crate::tensor::Bf16;
use crate::{arg, infer, launch, product, u32_size};
use cuda_core::{DeviceBuffer, Function, Stream};
use half::bf16;
use infinity_v3_backend_core::BackendError;
use std::ops::Range;

mod enqueue;
mod prepare;

pub(crate) struct Slabs<'a> {
    pub residual: &'a DeviceBuffer<Bf16>,
    pub projection: &'a DeviceBuffer<Bf16>,
    pub activation: &'a DeviceBuffer<Bf16>,
}

#[derive(Clone, Copy)]
struct NormOp {
    dispatch: NormDispatch,
    x: u64,
    gamma: u64,
    out: u64,
    rows: u32,
    width: u32,
    heads: u32,
    stride0: u64,
    stride1: u64,
}
impl NormOp {
    #[allow(clippy::too_many_arguments)]
    fn new(
        x: u64,
        gamma: u64,
        out: u64,
        rows: usize,
        width: usize,
        heads: usize,
        stride0: usize,
        stride1: usize,
    ) -> Result<Self, BackendError> {
        crate::qwen35_prepared::validate_norm_alias(x, out, rows, width, heads, stride0, stride1)?;
        Ok(Self {
            dispatch: NormDispatch::select(width, heads, stride0, stride1, &[x, gamma, out])?,
            x,
            gamma,
            out,
            rows: u32_size(rows)?,
            width: u32_size(width)?,
            heads: u32_size(heads)?,
            stride0: stride0 as u64,
            stride1: stride1 as u64,
        })
    }
    unsafe fn enqueue(&self, k: &Qwen35Kernels, s: &Stream, eps: f32) -> Result<(), BackendError> {
        // SAFETY: the outer frame owns all prepared spans through completion.
        unsafe {
            k.norm_bound(
                s,
                self.dispatch,
                self.x,
                self.gamma,
                self.out,
                self.rows,
                self.width,
                self.heads,
                self.stride0,
                self.stride1,
                eps,
            )
        }
    }
}

#[derive(Clone, Copy)]
struct ResidualOp {
    dispatch: NormDispatch,
    x: u64,
    addition: u64,
    gamma: u64,
    out: u64,
    rows: u32,
    width: u32,
}
impl ResidualOp {
    fn new(
        x: u64,
        addition: u64,
        gamma: u64,
        out: u64,
        rows: usize,
        width: usize,
    ) -> Result<Self, BackendError> {
        Ok(Self {
            dispatch: NormDispatch::select(width, 1, width, 0, &[x, addition, gamma, out, x])?,
            x,
            addition,
            gamma,
            out,
            rows: u32_size(rows)?,
            width: u32_size(width)?,
        })
    }
    unsafe fn enqueue(&self, k: &Qwen35Kernels, s: &Stream, eps: f32) -> Result<(), BackendError> {
        // SAFETY: contiguous residual update is in place; normalized is disjoint.
        unsafe {
            k.residual_norm_bound(
                s,
                self.dispatch,
                self.x,
                self.addition,
                self.gamma,
                self.out,
                self.x,
                self.rows,
                self.width,
                eps,
            )
        }
    }
}

enum AttentionStep {
    Full {
        qg: GemmArguments,
        key: GemmArguments,
        value: GemmArguments,
        output: GemmArguments,
        q_norm: NormOp,
        k_norm: NormOp,
    },
    Linear {
        qkv: GemmArguments,
        z: GemmArguments,
        a: GemmArguments,
        b: GemmArguments,
        output: GemmArguments,
    },
}
struct LayerStep {
    attention: AttentionStep,
    post: ResidualOp,
    gate_up: GemmArguments,
    down: GemmArguments,
    next: ResidualOp,
}

/// Fixed launch functions are retained by the model; no symbol query in layers.
pub(crate) struct Qwen35AuxKernels {
    pub join: Function,
    pub silu: Function,
}

pub(crate) struct PreparedWork {
    first: NormOp,
    layers: Vec<LayerStep>,
    attention: AttentionBatch,
    block: Option<PreparedBlockPlan>,
    tokens: u32,
    qheads: u32,
    kvheads: u32,
    intermediate: u32,
    value_heads: u32,
    positions: u64,
    query_gate: u64,
    query_gate_stride: u64,
    eps: f32,
    qkv: Range<usize>,
    z: Range<usize>,
    ab: Range<usize>,
    qkv_width: usize,
    z_width: usize,
    ab_width: usize,
}

fn pointer(base: u64, offset: usize) -> Result<u64, BackendError> {
    base.checked_add(u64::try_from(product(offset, 2)?).map_err(infer)?)
        .ok_or_else(|| infer("Qwen3.5 phase pointer overflow"))
}
fn span(base: u64, len: usize, offset: usize, elements: usize) -> Result<u64, BackendError> {
    if offset.checked_add(elements).is_none_or(|end| end > len) {
        return Err(infer(
            "Qwen3.5 prepared phase span exceeds retained storage",
        ));
    }
    pointer(base, offset)
}
fn linear(
    rows: usize,
    input: usize,
    width: usize,
    x: u64,
    w: u64,
    y: u64,
) -> Result<GemmArguments, BackendError> {
    Ok(LinearPlan::<Bf16>::contiguous(rows, input, width)
        .map_err(infer)?
        .arguments(x, w, y))
}

fn shift(range: Range<usize>, base: usize, name: &str) -> Result<Range<usize>, BackendError> {
    let start = base
        .checked_add(range.start)
        .ok_or_else(|| infer(format!("Qwen3.5 {name} start overflow")))?;
    let end = base
        .checked_add(range.end)
        .ok_or_else(|| infer(format!("Qwen3.5 {name} end overflow")))?;
    Ok(start..end)
}

impl PreparedWork {
    fn gdn_inputs<'a>(&self, slabs: &Slabs<'a>) -> Result<GdnInputViews<'a>, BackendError> {
        Ok(GdnInputViews {
            qkv: GdnBufferView::from_tensor_buffer_range(slabs.projection, self.qkv.clone())?,
            qkv_row_stride: self.qkv_width,
            z: GdnBufferView::from_tensor_buffer_range(slabs.activation, self.z.clone())?,
            z_row_stride: self.z_width,
            ab: GdnBufferView::from_tensor_buffer_range(slabs.activation, self.ab.clone())?,
            ab_row_stride: self.ab_width,
        })
    }

    /// Borrow real prefix or suffix row subranges of the combined T projections.
    fn block_gdn_inputs<'a>(
        &self,
        slabs: &Slabs<'a>,
        layout: &Qwen35BlockLayout,
        suffix: bool,
    ) -> Result<GdnInputViews<'a>, BackendError> {
        let (prefix_qkv, suffix_qkv) = layout.split_rows(layout.channels(), "QKV rows")?;
        let (prefix_z, suffix_z) = layout.split_rows(layout.value_width()?, "Z rows")?;
        let (prefix_ab, suffix_ab) = layout.split_rows(layout.ab_width()?, "AB rows")?;
        let (qkv, z, ab) = if suffix {
            (suffix_qkv, suffix_z, suffix_ab)
        } else {
            (prefix_qkv, prefix_z, prefix_ab)
        };
        Ok(GdnInputViews {
            qkv: GdnBufferView::from_tensor_buffer_range(
                slabs.projection,
                shift(qkv, self.qkv.start, "block QKV rows")?,
            )?,
            qkv_row_stride: self.qkv_width,
            z: GdnBufferView::from_tensor_buffer_range(
                slabs.activation,
                shift(z, self.z.start, "block Z rows")?,
            )?,
            z_row_stride: self.z_width,
            ab: GdnBufferView::from_tensor_buffer_range(
                slabs.activation,
                shift(ab, self.ab.start, "block AB rows")?,
            )?,
            ab_row_stride: self.ab_width,
        })
    }

    fn block_history<'a>(
        &self,
        slabs: &Slabs<'a>,
        layout: &Qwen35BlockLayout,
    ) -> Result<GdnBufferView<'a, bf16>, BackendError> {
        GdnBufferView::from_tensor_buffer_range(
            slabs.projection,
            shift(layout.history_range(), self.qkv.start, "block history rows")?,
        )
    }
}
