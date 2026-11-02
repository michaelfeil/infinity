// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Prepared layer scheduling over three phase-reused BF16 arenas.
//! This leaf submits into the outer engine's already-bound stream; it never
//! owns a completion boundary or creates per-layer temporary allocations.
use crate::cublas::Blas;
use crate::fa4::{AttentionBatch, Fa4};
use crate::gdn_prepared::{
    GdnBlockPair, GdnBufferView, GdnInputViews, GdnOwnedFrame, GdnPrepared, GdnState,
};
use crate::math::{GemmArguments, LinearPlan};
use crate::qwen35_block_attention::PreparedBlockPlan;
use crate::qwen35_block_layout::Qwen35BlockLayout;
use crate::qwen35_config::Qwen35Config;
use crate::qwen35_layout::{AttentionSlabLayout, Qwen35Layout};
use crate::qwen35_moe_config::Qwen35MoeTextGeometry;
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
pub(crate) struct NormOp {
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
    pub(crate) fn new(
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
    pub(crate) unsafe fn enqueue(
        &self,
        k: &Qwen35Kernels,
        s: &Stream,
        eps: f32,
    ) -> Result<(), BackendError> {
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
pub(crate) struct ResidualOp {
    dispatch: NormDispatch,
    x: u64,
    addition: u64,
    gamma: u64,
    out: u64,
    rows: u32,
    width: u32,
}
impl ResidualOp {
    pub(crate) fn new(
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
    pub(crate) unsafe fn enqueue(
        &self,
        k: &Qwen35Kernels,
        s: &Stream,
        eps: f32,
    ) -> Result<(), BackendError> {
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

pub(crate) enum AttentionStep {
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

pub(crate) struct AttentionLayerStep {
    attention: AttentionStep,
    post: ResidualOp,
}

/// Attention geometry shared by the dense donor and the MoE full executor.
/// Only the fields the existing attention/GDN kernels already consumed are
/// retained, so the MoE path does not need a synthetic dense MLP config.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct AttentionGeometry {
    pub(crate) hidden_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) rotary_dim: usize,
    pub(crate) linear_num_key_heads: usize,
    pub(crate) linear_num_value_heads: usize,
    pub(crate) linear_value_head_dim: usize,
    pub(crate) rms_norm_eps: f32,
}

impl AttentionGeometry {
    pub(crate) fn from_config(config: &Qwen35Config) -> Self {
        Self {
            hidden_size: config.hidden_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            rotary_dim: config.rotary_dim,
            linear_num_key_heads: config.linear_num_key_heads,
            linear_num_value_heads: config.linear_num_value_heads,
            linear_value_head_dim: config.linear_value_head_dim,
            rms_norm_eps: config.rms_norm_eps,
        }
    }

    pub(crate) fn from_moe_text(text: &Qwen35MoeTextGeometry) -> Self {
        Self {
            hidden_size: text.hidden_size,
            num_hidden_layers: text.num_hidden_layers,
            num_attention_heads: text.num_attention_heads,
            num_key_value_heads: text.num_key_value_heads,
            head_dim: text.head_dim,
            rotary_dim: text.rotary_dim,
            linear_num_key_heads: text.linear_num_key_heads,
            linear_num_value_heads: text.linear_num_value_heads,
            linear_value_head_dim: text.linear_value_head_dim,
            rms_norm_eps: text.rms_norm_eps,
        }
    }

    pub(crate) fn q_width(&self) -> Result<usize, BackendError> {
        product(self.num_attention_heads, self.head_dim)
    }

    pub(crate) fn kv_width(&self) -> Result<usize, BackendError> {
        product(self.num_key_value_heads, self.head_dim)
    }

    fn linear_widths(&self) -> Result<(usize, usize, usize), BackendError> {
        let key = product(self.linear_num_key_heads, self.head_dim)?;
        let value = product(self.linear_num_value_heads, self.linear_value_head_dim)?;
        let qkv = product(self.linear_num_key_heads, 2)?
            .checked_mul(self.head_dim)
            .and_then(|pair| pair.checked_add(value))
            .ok_or_else(|| infer("linear QKV width overflow"))?;
        Ok((key, value, qkv))
    }
}

/// One layer's uploaded attention weights plus the two per-layer RMSNorm
/// offsets that the shared attention frame binds.
pub(crate) struct AttentionLayerWeights<'a> {
    pub(crate) attention: &'a Qwen35DeviceAttention,
    pub(crate) input_norm: &'a DeviceBuffer<Bf16>,
    pub(crate) post_attention_norm: &'a DeviceBuffer<Bf16>,
}

struct AttentionFrame {
    tokens: u32,
    qheads: u32,
    kvheads: u32,
    value_heads: u32,
    positions: u64,
    query_gate: u64,
    query_gate_stride: u64,
    eps: f32,
    head_dim: u32,
    rotary: u32,
    qkv: Range<usize>,
    z: Range<usize>,
    ab: Range<usize>,
    qkv_width: usize,
    z_width: usize,
    ab_width: usize,
    batch: AttentionBatch,
    block: Option<PreparedBlockPlan>,
    norm: u64,
    output: u64,
    next_gammas: Vec<u64>,
}

/// The attention-only half of a prepared frame. Both the dense donor executor
/// and the Qwen3.5-MoE full executor use this same prepared math, so neither
/// owns a second attention implementation.
pub(crate) struct PreparedAttentionWork {
    first: NormOp,
    layers: Vec<AttentionLayerStep>,
    frame: AttentionFrame,
}

impl PreparedAttentionWork {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        geometry: &AttentionGeometry,
        layers: &[AttentionLayerWeights<'_>],
        embed: &DeviceBuffer<Bf16>,
        final_norm: &DeviceBuffer<Bf16>,
        layout: &impl AttentionSlabLayout,
        slabs: &Slabs<'_>,
        tokens: usize,
        positions: u64,
        cumulative: u64,
        sequences: usize,
        max_len: usize,
    ) -> Result<Self, BackendError> {
        prepare::prepare_attention_work(
            geometry, layers, embed, final_norm, layout, slabs, tokens, positions, cumulative,
            sequences, max_len,
        )
    }

    pub(crate) fn norm_address(&self) -> u64 {
        self.frame.norm
    }

    pub(crate) fn output_address(&self) -> u64 {
        self.frame.output
    }

    pub(crate) fn token_count(&self) -> u32 {
        self.frame.tokens
    }

    pub(crate) fn head_dim(&self) -> u32 {
        self.frame.head_dim
    }

    pub(crate) fn next_gamma(&self, index: usize) -> Result<u64, BackendError> {
        self.frame
            .next_gammas
            .get(index)
            .copied()
            .ok_or_else(|| infer("Qwen3.5 attention layer index is out of range"))
    }

    pub(crate) fn set_block(&mut self, block: PreparedBlockPlan) {
        self.frame.block = Some(block);
    }

    pub(crate) fn batch(&self) -> &AttentionBatch {
        &self.frame.batch
    }

    pub(crate) unsafe fn enqueue_first(
        &self,
        math: &Qwen35Kernels,
        stream: &Stream,
    ) -> Result<(), BackendError> {
        // SAFETY: prepared against the retained residual slab.
        unsafe { self.first.enqueue(math, stream, self.frame.eps) }
    }

    pub(crate) unsafe fn enqueue_post(
        &self,
        index: usize,
        math: &Qwen35Kernels,
        stream: &Stream,
    ) -> Result<(), BackendError> {
        let layer = self
            .layers
            .get(index)
            .ok_or_else(|| infer("Qwen3.5 attention layer index is out of range"))?;
        // SAFETY: prepared against the retained residual/activation slabs.
        unsafe { layer.post.enqueue(math, stream, self.frame.eps) }
    }

    /// Enqueue just the attention step for one layer. FFN work is interleaved
    /// by the caller between this and the prepared next residual.
    ///
    /// # Safety
    /// The caller retains every weight, slab, table, GDN frame and stream owner
    /// through completion, and on error drains or quarantines the stream.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn enqueue_attention(
        &self,
        index: usize,
        layer: &Qwen35DeviceAttention,
        blas: &Blas,
        math: &Qwen35Kernels,
        aux: &Qwen35AuxKernels,
        fa4: &Fa4,
        stream: &Stream,
        slabs: &Slabs<'_>,
        gdn: &mut GdnPrepared,
        prepared: &PreparedGdn<'_, '_>,
        cosine: u64,
        sine: u64,
        table_stride: u32,
    ) -> Result<(), BackendError> {
        enqueue::enqueue_attention_layer(
            self,
            index,
            layer,
            blas,
            math,
            aux,
            fa4,
            stream,
            slabs,
            gdn,
            prepared,
            cosine,
            sine,
            table_stride,
        )
    }

    /// Prepare the one ordinary fresh GDN frame shared by every linear layer.
    ///
    /// # Safety
    /// Same outer Work ownership contract as [`Self::enqueue_attention`].
    pub(crate) unsafe fn prepare_fresh_frame(
        &self,
        gdn: &mut GdnPrepared,
        slabs: &Slabs<'_>,
        cumulative: &[u32],
        uploaded_cumulative: GdnBufferView<'_, u32>,
    ) -> Result<GdnOwnedFrame, BackendError> {
        // SAFETY: the caller retains the GDN prepared owner and metadata input.
        unsafe {
            gdn.prepare_owned_fresh_frame_from_device(
                self.gdn_inputs(slabs)?,
                cumulative,
                uploaded_cumulative,
            )
        }
        .map_err(|failure| failure.error)
    }

    fn gdn_inputs<'a>(&self, slabs: &Slabs<'a>) -> Result<GdnInputViews<'a>, BackendError> {
        Ok(GdnInputViews {
            qkv: GdnBufferView::from_tensor_buffer_range(slabs.projection, self.frame.qkv.clone())?,
            qkv_row_stride: self.frame.qkv_width,
            z: GdnBufferView::from_tensor_buffer_range(slabs.activation, self.frame.z.clone())?,
            z_row_stride: self.frame.z_width,
            ab: GdnBufferView::from_tensor_buffer_range(slabs.activation, self.frame.ab.clone())?,
            ab_row_stride: self.frame.ab_width,
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
                shift(qkv, self.frame.qkv.start, "block QKV rows")?,
            )?,
            qkv_row_stride: self.frame.qkv_width,
            z: GdnBufferView::from_tensor_buffer_range(
                slabs.activation,
                shift(z, self.frame.z.start, "block Z rows")?,
            )?,
            z_row_stride: self.frame.z_width,
            ab: GdnBufferView::from_tensor_buffer_range(
                slabs.activation,
                shift(ab, self.frame.ab.start, "block AB rows")?,
            )?,
            ab_row_stride: self.frame.ab_width,
        })
    }

    fn block_history<'a>(
        &self,
        slabs: &Slabs<'a>,
        layout: &Qwen35BlockLayout,
    ) -> Result<GdnBufferView<'a, bf16>, BackendError> {
        GdnBufferView::from_tensor_buffer_range(
            slabs.projection,
            shift(
                layout.history_range(),
                self.frame.qkv.start,
                "block history rows",
            )?,
        )
    }
}

/// Shared GDN frame choice consumed by the per-layer attention enqueue.
pub(crate) enum PreparedGdn<'borrow, 'frame> {
    Fresh(&'borrow GdnOwnedFrame),
    Block(&'borrow GdnBlockPair<'frame>),
}

struct FfnStep {
    gate_up: GemmArguments,
    down: GemmArguments,
}

pub(crate) struct PreparedWork {
    attention: PreparedAttentionWork,
    ffn: Vec<FfnStep>,
    next: Vec<ResidualOp>,
    intermediate: u32,
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

/// Fixed launch functions are retained by the model; no symbol query in layers.
pub(crate) struct Qwen35AuxKernels {
    pub join: Function,
    pub silu: Function,
}
