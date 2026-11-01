// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Graph-free BF16 dense Qwen2/Llama pooled-embedding executor.
//!
//! The loader in `decoder_model` validates configuration and checkpoint
//! metadata. This module selects concrete kernel/plan geometry once, uploads
//! packed weights and immutable RoPE tables, then expresses one ordered stream
//! of word gather, RMSNorm, biased projections, in-place split-half RoPE, FA4
//! attention, staged SiLU and pooling. No dynamic operator graph is built.
//!
//! The only Llama recipe qualified here is the base-unscaled, causal global,
//! hidden-2048 dense SiLU backbone. It binds the native width-2048 Candle RMS
//! exports, HFMA NeoX RoPE and the rounded DenseMlp SiLU export at load time;
//! Qwen2 keeps its ordinary RMS recipe, non-HFMA RoPE and launch geometry;
//! H1536 retains the six lane values through the same generic reduction.

use super::{
    arg, blocks, download_output, fa4, infer, launch, launch_threads, product, stage_batch, start,
    u32_size, upload_input, HostIoLease, LtPlan, LtProvider, WorkKey,
};
use crate::{
    cublas::Blas,
    decoder_model::{
        DecoderConfig, DecoderFamily, DecoderLayerWeights, DecoderRope, DecoderWeights,
    },
    lt::ShapeError,
    math::LinearPlan,
    rotary::BaseRopeTables,
    staging,
    tensor::Bf16,
    upload::WeightUploader,
};
use cuda_core::{Context, DeviceBuffer, Function, Stream};
use infinity_v3_backend_core::{BackendError, Batch, Pool};
use std::{
    mem::ManuallyDrop,
    path::Path,
    sync::{atomic::Ordering, Arc},
};

/// Explicit construction budget for the largest supported base-RoPE table.
/// The GTE geometry needs 67,109,120 peak payload bytes; this cap leaves room
/// for the immutable tables without silently growing at inference.
const ROPE_TABLE_BYTE_LIMIT: usize = 128 * 1024 * 1024;

/// RMS launch geometry prepared once at load. The width-2048 Llama kernel uses
/// one warp per row and four rows per CTA; Qwen keeps the generic one-row,
/// 256-thread launch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct NormGeometry {
    rows_per_cta: u32,
    threads: u32,
}
const GENERIC_NORM_GEOMETRY: NormGeometry = NormGeometry {
    rows_per_cta: 1,
    threads: 256,
};
const H2048_NORM_GEOMETRY: NormGeometry = NormGeometry {
    rows_per_cta: 4,
    threads: 128,
};

/// Native arithmetic recipe resolved once from loaded metadata. Executor
/// functions and launch geometry are bound from this value; the layer loop
/// never re-evaluates family or width policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DecoderMath {
    Qwen2FullNeox,
    Qwen2H1536FullNeox,
    LlamaH2048Exact,
}
impl DecoderMath {
    pub(crate) fn select(config: &DecoderConfig) -> Result<Self, BackendError> {
        match config.family {
            DecoderFamily::Qwen2 => {
                if config.hidden_size == 896 && config.head_dim == 64 {
                    return Err(start(
                        "Qwen2 H896/d64 requires the unported masked Candle RMS and rotary recipe",
                    ));
                }
                if config.rope != DecoderRope::FullNeoX {
                    return Err(start(
                        "decoder executor requires the native Qwen2 full split-half NeoX RoPE recipe",
                    ));
                }
                Ok(if config.hidden_size == 1536 {
                    Self::Qwen2H1536FullNeox
                } else {
                    Self::Qwen2FullNeox
                })
            }
            DecoderFamily::Llama => {
                if config.hidden_size != 2048 {
                    return Err(start(
                        "decoder executor implements only the BF16 Llama hidden_size 2048 recipe; H4096 is deferred",
                    ));
                }
                if config.rope != DecoderRope::Default {
                    return Err(start(
                        "decoder executor requires the base unscaled Llama default RoPE recipe",
                    ));
                }
                if !config.is_causal {
                    return Err(start(
                        "decoder executor implements only causal global Llama attention",
                    ));
                }
                Ok(Self::LlamaH2048Exact)
            }
        }
    }

    const fn rms_export(self) -> &'static str {
        match self {
            Self::Qwen2FullNeox => "infinity_rms_norm",
            Self::Qwen2H1536FullNeox => "infinity_rms_norm_1536_cached",
            Self::LlamaH2048Exact => "infinity_rms_norm_2048_vector",
        }
    }

    const fn add_rms_export(self) -> &'static str {
        match self {
            Self::Qwen2FullNeox => "infinity_add_rms_norm",
            Self::Qwen2H1536FullNeox => "infinity_add_rms_norm_1536_cached",
            Self::LlamaH2048Exact => "infinity_add_rms_norm_2048_vector",
        }
    }

    const fn rope_export(self) -> &'static str {
        match self {
            Self::Qwen2FullNeox | Self::Qwen2H1536FullNeox => "infinity_rope_neox",
            Self::LlamaH2048Exact => "infinity_rope_neox_hfma",
        }
    }

    /// Native DenseMlp code5 (`RoundedSwiGlu`) uses BF16 pair instructions, so
    /// both decoder families bind the eight-value rounded-intermediate export.
    /// The scalar and paired exports remain available to other callers.
    const fn activation_export(self) -> &'static str {
        match self {
            Self::Qwen2FullNeox | Self::Qwen2H1536FullNeox | Self::LlamaH2048Exact => {
                "infinity_silu_mul_bf16_vector8_native"
            }
        }
    }

    const fn norm_geometry(self) -> NormGeometry {
        match self {
            Self::Qwen2FullNeox | Self::Qwen2H1536FullNeox => GENERIC_NORM_GEOMETRY,
            Self::LlamaH2048Exact => H2048_NORM_GEOMETRY,
        }
    }
}

struct DecoderKernels {
    gather: Function,
    rms_norm: Function,
    add_rms_norm: Function,
    rope_neox: Function,
    silu_mul: Function,
    pool: Function,
    selected_rows: Function,
    bf16_to_f32: Function,
    norm_geometry: NormGeometry,
}
impl DecoderKernels {
    fn load(context: &Arc<Context>, math: DecoderMath) -> Result<Self, BackendError> {
        let path = std::env::var_os(infinity_v3_config::infinity_env!("OXIDE_KERNEL_MODULE"))
            .ok_or_else(|| {
                start("Set INFINITY_OXIDE_KERNEL_MODULE to the artifact compiled by cargo oxide")
            })?;
        let path = std::path::PathBuf::from(path);
        let path = if path.is_dir() {
            let (major, minor) = context.compute_capability().map_err(start)?;
            path.join(format!("sm_{major}{minor}"))
                .join("kernels.cubin")
        } else {
            path
        };
        let data = std::fs::read(&path).map_err(|e| start(format!("{}: {e}", path.display())))?;
        let module = context.load_module_from_image(&data).map_err(start)?;
        Ok(Self {
            gather: module
                .load_function("infinity_embedding_gather")
                .map_err(start)?,
            rms_norm: module.load_function(math.rms_export()).map_err(start)?,
            add_rms_norm: module.load_function(math.add_rms_export()).map_err(start)?,
            rope_neox: module.load_function(math.rope_export()).map_err(start)?,
            silu_mul: module
                .load_function(math.activation_export())
                .map_err(start)?,
            pool: module.load_function("infinity_pool").map_err(start)?,
            selected_rows: module
                .load_function("infinity_selected_rows")
                .map_err(start)?,
            bf16_to_f32: module
                .load_function("infinity_bf16_to_f32")
                .map_err(start)?,
            norm_geometry: math.norm_geometry(),
        })
    }
}

struct DecoderLinear {
    weight: DeviceBuffer<Bf16>,
    bias: Option<DeviceBuffer<Bf16>>,
}
struct DecoderLayer {
    input_norm: DeviceBuffer<Bf16>,
    post_attention_norm: DeviceBuffer<Bf16>,
    qkv: DecoderLinear,
    attention: DecoderLinear,
    gate_up: DecoderLinear,
    down: DecoderLinear,
}
struct DecoderWeightBuffers {
    embed: DeviceBuffer<Bf16>,
    norm: DeviceBuffer<Bf16>,
    layers: Vec<DecoderLayer>,
}
impl DecoderWeightBuffers {
    pub(crate) fn upload(
        uploader: &mut WeightUploader,
        _stream: &Arc<Stream>,
        host: &DecoderWeights,
    ) -> Result<Self, BackendError> {
        let mut layers = Vec::new();
        layers.try_reserve_exact(host.layers.len()).map_err(start)?;
        for layer in &host.layers {
            let qkv_bias = match (&layer.q.bias, &layer.k.bias, &layer.v.bias) {
                (Some(q), Some(k), Some(v)) => Some(uploader.upload(&[q, k, v])?),
                (None, None, None) => None,
                _ => {
                    return Err(start(
                        "decoder Q/K/V projection biases must be all present or all absent",
                    ));
                }
            };
            let gate_up_bias = match (&layer.gate.bias, &layer.up.bias) {
                (Some(gate), Some(up)) => Some(uploader.upload(&[gate, up])?),
                (None, None) => None,
                _ => {
                    return Err(start(
                        "decoder gate/up projection biases must be both present or both absent",
                    ));
                }
            };
            let attention_bias = layer
                .o
                .bias
                .as_ref()
                .map(|bias| uploader.upload(&[bias]))
                .transpose()?;
            let down_bias = layer
                .down
                .bias
                .as_ref()
                .map(|bias| uploader.upload(&[bias]))
                .transpose()?;
            layers.push(DecoderLayer {
                input_norm: uploader.upload(&[&layer.input_norm])?,
                post_attention_norm: uploader.upload(&[&layer.post_attention_norm])?,
                qkv: DecoderLinear {
                    weight: uploader.upload(&[
                        &layer.q.weight,
                        &layer.k.weight,
                        &layer.v.weight,
                    ])?,
                    bias: qkv_bias,
                },
                attention: DecoderLinear {
                    weight: uploader.upload(&[&layer.o.weight])?,
                    bias: attention_bias,
                },
                gate_up: DecoderLinear {
                    weight: uploader.upload(&[&layer.gate.weight, &layer.up.weight])?,
                    bias: gate_up_bias,
                },
                down: DecoderLinear {
                    weight: uploader.upload(&[&layer.down.weight])?,
                    bias: down_bias,
                },
            });
        }
        Ok(Self {
            embed: uploader.upload(&[&host.embed])?,
            norm: uploader.upload(&[&host.norm])?,
            layers,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BiasSupport {
    pub(crate) qkv: bool,
    pub(crate) attention: bool,
    pub(crate) gate_up: bool,
    pub(crate) down: bool,
    pub(crate) requires_plain: bool,
}
impl BiasSupport {
    pub(crate) fn from_layers(host: &DecoderWeights) -> Self {
        let mut support = Self {
            qkv: false,
            attention: false,
            gate_up: false,
            down: false,
            requires_plain: false,
        };
        for layer in &host.layers {
            support.qkv |= layer.q.bias.is_some();
            support.attention |= layer.o.bias.is_some();
            support.gate_up |= layer.gate.bias.is_some();
            support.down |= layer.down.bias.is_some();
            support.requires_plain |= layer.q.bias.is_none()
                || layer.o.bias.is_none()
                || layer.gate.bias.is_none()
                || layer.down.bias.is_none();
        }
        support
    }

    /// A plain cuBLAS handle is created only when at least one loaded
    /// projection has no checkpoint bias. This is a load-time property, not an
    /// M/family heuristic.
    pub(crate) const fn uses_plain(self) -> bool {
        self.requires_plain
    }
}

/// Biases physically present on the last layer, which alone feeds the selected
/// readout. Aggregate `BiasSupport` cannot stand in for this: an earlier layer
/// may carry a bias while the final one does not.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct ReadoutBias {
    attention: bool,
    gate_up: bool,
    down: bool,
}
impl ReadoutBias {
    fn from_last_layer(layer: Option<&DecoderLayerWeights>) -> Self {
        layer.map_or_else(Self::default, |layer| Self {
            attention: layer.o.bias.is_some(),
            gate_up: layer.gate.bias.is_some(),
            down: layer.down.bias.is_some(),
        })
    }
}

/// Full-state and selected-readout row counts are tracked independently so a
/// changed batch refreshes each set exactly once, outside the layer loop.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct PlanRows {
    full: Option<usize>,
    readout: Option<usize>,
}

const PLAIN_SEED_ROWS: usize = 128;

struct DecoderPlainPlans {
    qkv: LinearPlan<Bf16>,
    attention: LinearPlan<Bf16>,
    gate_up: LinearPlan<Bf16>,
    down: LinearPlan<Bf16>,
}

/// Pointwise projections after the final attention gather. Their operand rows
/// are the actual selected count, never the full token capacity or count.
struct DecoderReadoutPlans {
    plain_attention: LinearPlan<Bf16>,
    plain_gate_up: LinearPlan<Bf16>,
    plain_down: LinearPlan<Bf16>,
    bias_attention: Option<LtPlan<Bf16>>,
    bias_gate_up: Option<LtPlan<Bf16>>,
    bias_down: Option<LtPlan<Bf16>>,
}
impl DecoderReadoutPlans {
    fn new(
        provider: &LtProvider,
        c: &DecoderConfig,
        bias: ReadoutBias,
    ) -> Result<Self, BackendError> {
        let gate_up_width = c
            .intermediate_size
            .checked_mul(2)
            .ok_or_else(|| start("decoder gate/up width overflow"))?;
        Ok(Self {
            plain_attention: LinearPlan::contiguous(PLAIN_SEED_ROWS, c.q_width(), c.hidden_size)
                .map_err(infer)?,
            plain_gate_up: LinearPlan::contiguous(PLAIN_SEED_ROWS, c.hidden_size, gate_up_width)
                .map_err(infer)?,
            plain_down: LinearPlan::contiguous(PLAIN_SEED_ROWS, c.intermediate_size, c.hidden_size)
                .map_err(infer)?,
            bias_attention: bias
                .attention
                .then(|| provider.prepare_bias_shape::<Bf16>(c.q_width(), c.hidden_size))
                .transpose()?,
            bias_gate_up: bias
                .gate_up
                .then(|| provider.prepare_bias_shape::<Bf16>(c.hidden_size, gate_up_width))
                .transpose()?,
            bias_down: bias
                .down
                .then(|| provider.prepare_bias_shape::<Bf16>(c.intermediate_size, c.hidden_size))
                .transpose()?,
        })
    }

    fn set_rows(&mut self, provider: &LtProvider, rows: usize) -> Result<(), ShapeError> {
        let rejected = |error| ShapeError::Rejected(infer(error));
        self.plain_attention.set_rows(rows).map_err(rejected)?;
        self.plain_gate_up.set_rows(rows).map_err(rejected)?;
        self.plain_down.set_rows(rows).map_err(rejected)?;
        if let Some(plan) = self.bias_attention.as_mut() {
            provider.set_logical_rows(plan, rows)?;
        }
        if let Some(plan) = self.bias_gate_up.as_mut() {
            provider.set_logical_rows(plan, rows)?;
        }
        if let Some(plan) = self.bias_down.as_mut() {
            provider.set_logical_rows(plan, rows)?;
        }
        Ok(())
    }
}

struct DecoderPlans {
    rows: PlanRows,
    plain: DecoderPlainPlans,
    bias_qkv: Option<LtPlan<Bf16>>,
    bias_attention: Option<LtPlan<Bf16>>,
    bias_gate_up: Option<LtPlan<Bf16>>,
    bias_down: Option<LtPlan<Bf16>>,
    readout: DecoderReadoutPlans,
}
impl DecoderPlans {
    fn new(
        provider: &LtProvider,
        c: &DecoderConfig,
        bias: BiasSupport,
        readout_bias: ReadoutBias,
    ) -> Result<Self, BackendError> {
        let qkv_width = c
            .q_width()
            .checked_add(
                c.kv_width()
                    .checked_mul(2)
                    .ok_or_else(|| start("decoder QKV width overflow"))?,
            )
            .ok_or_else(|| start("decoder QKV width overflow"))?;
        let gate_up_width = c
            .intermediate_size
            .checked_mul(2)
            .ok_or_else(|| start("decoder gate/up width overflow"))?;
        let plain = DecoderPlainPlans {
            qkv: LinearPlan::contiguous(PLAIN_SEED_ROWS, c.hidden_size, qkv_width)
                .map_err(infer)?,
            attention: LinearPlan::contiguous(PLAIN_SEED_ROWS, c.q_width(), c.hidden_size)
                .map_err(infer)?,
            gate_up: LinearPlan::contiguous(PLAIN_SEED_ROWS, c.hidden_size, gate_up_width)
                .map_err(infer)?,
            down: LinearPlan::contiguous(PLAIN_SEED_ROWS, c.intermediate_size, c.hidden_size)
                .map_err(infer)?,
        };
        Ok(Self {
            rows: PlanRows::default(),
            plain,
            bias_qkv: bias
                .qkv
                .then(|| provider.prepare_bias_shape::<Bf16>(c.hidden_size, qkv_width))
                .transpose()?,
            bias_attention: bias
                .attention
                .then(|| provider.prepare_bias_shape::<Bf16>(c.q_width(), c.hidden_size))
                .transpose()?,
            bias_gate_up: bias
                .gate_up
                .then(|| provider.prepare_bias_shape::<Bf16>(c.hidden_size, gate_up_width))
                .transpose()?,
            bias_down: bias
                .down
                .then(|| provider.prepare_bias_shape::<Bf16>(c.intermediate_size, c.hidden_size))
                .transpose()?,
            readout: DecoderReadoutPlans::new(provider, c, readout_bias)?,
        })
    }

    fn set_rows(&mut self, provider: &LtProvider, rows: usize) -> Result<(), ShapeError> {
        crate::math::refresh_plan_rows(&mut self.rows.full, rows, || {
            let rejected = |error| ShapeError::Rejected(infer(error));
            self.plain.qkv.set_rows(rows).map_err(rejected)?;
            self.plain.attention.set_rows(rows).map_err(rejected)?;
            self.plain.gate_up.set_rows(rows).map_err(rejected)?;
            self.plain.down.set_rows(rows).map_err(rejected)?;
            if let Some(plan) = self.bias_qkv.as_mut() {
                provider.set_logical_rows(plan, rows)?;
            }
            if let Some(plan) = self.bias_attention.as_mut() {
                provider.set_logical_rows(plan, rows)?;
            }
            if let Some(plan) = self.bias_gate_up.as_mut() {
                provider.set_logical_rows(plan, rows)?;
            }
            if let Some(plan) = self.bias_down.as_mut() {
                provider.set_logical_rows(plan, rows)?;
            }
            Ok(())
        })
    }

    fn set_readout_rows(&mut self, provider: &LtProvider, rows: usize) -> Result<(), ShapeError> {
        crate::math::refresh_plan_rows(&mut self.rows.readout, rows, || {
            self.readout.set_rows(provider, rows)
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct DecoderGeometry {
    hidden: usize,
    q_width: usize,
    kv_width: usize,
    qkv_width: usize,
    projection_width: usize,
    activation_width: usize,
    silu_groups_per_row: usize,
}
impl DecoderGeometry {
    fn new(c: &DecoderConfig) -> Result<Self, BackendError> {
        let q_width = c.q_width();
        let kv_width = c.kv_width();
        let qkv_width = q_width
            .checked_add(
                kv_width
                    .checked_mul(2)
                    .ok_or_else(|| start("decoder QKV width overflow"))?,
            )
            .ok_or_else(|| start("decoder QKV width overflow"))?;
        let gate_up_width = c
            .intermediate_size
            .checked_mul(2)
            .ok_or_else(|| start("decoder gate/up width overflow"))?;
        let projection_width = qkv_width.max(gate_up_width);
        let activation_width = q_width.max(c.intermediate_size);
        i64::try_from(qkv_width).map_err(start)?;
        i64::try_from(q_width).map_err(start)?;
        i64::try_from(kv_width).map_err(start)?;
        i64::try_from(c.head_dim).map_err(start)?;
        Ok(Self {
            hidden: c.hidden_size,
            q_width,
            kv_width,
            qkv_width,
            projection_width,
            activation_width,
            silu_groups_per_row: c.intermediate_size.div_ceil(8),
        })
    }

    fn attention_strides(&self, head_dim: usize) -> Result<[i64; 12], BackendError> {
        let qkv = i64::try_from(self.qkv_width).map_err(start)?;
        let q = i64::try_from(self.q_width).map_err(start)?;
        let d = i64::try_from(head_dim).map_err(start)?;
        Ok([qkv, d, 1, qkv, d, 1, qkv, d, 1, q, d, 1])
    }
}

#[cfg(test)]
fn validate_recipe(config: &DecoderConfig) -> Result<(), BackendError> {
    DecoderMath::select(config).map(|_| ())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SelectedReadout {
    Cls,
    LastToken,
}
impl SelectedReadout {
    const fn kernel_mode(self) -> u32 {
        match self {
            Self::Cls => 0,
            Self::LastToken => 1,
        }
    }
}

/// Dense Qwen2 CLS/last readout gathers all B sequences in canonical order at
/// the final attention boundary, runs the pointwise projections at M=B, then
/// selects requested outputs. The selected count, including duplicate excess,
/// only controls the final request-order extraction. Mean keeps full states.
pub(crate) fn selected_readout(pool: &Pool, selections: usize) -> Option<SelectedReadout> {
    if selections == 0 {
        return None;
    }
    match pool {
        Pool::Cls => Some(SelectedReadout::Cls),
        Pool::LastToken => Some(SelectedReadout::LastToken),
        Pool::Mean | Pool::Splade => None,
    }
}

pub(crate) struct DecoderModel {
    pub(crate) config: DecoderConfig,
    weights: DecoderWeightBuffers,
    plans: DecoderPlans,
    kernels: DecoderKernels,
    fa4: fa4::Fa4,
    rope_cosine: DeviceBuffer<f32>,
    rope_sine: DeviceBuffer<f32>,
    geometry: DecoderGeometry,
    uses_plain: bool,
}
impl DecoderModel {
    pub(crate) fn new(
        host: DecoderWeights,
        context: &Arc<Context>,
        stream: &Arc<Stream>,
        lt: &LtProvider,
    ) -> Result<Self, BackendError> {
        let config = host.config.clone();
        let math = DecoderMath::select(&config)?;
        let geometry = DecoderGeometry::new(&config)?;
        let attention_spec = fa4::AttentionSpec {
            heads: config.num_attention_heads,
            kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            causal: config.is_causal,
            window: fa4::Window::Global,
            dtype: fa4::DType::Bf16,
            scale: fa4::Scale::RsqrtHeadDim,
        };
        if !fa4::Fa4::supports(&attention_spec) {
            return Err(start(format!(
                "decoder attention H{} KV{} d{} causal={} has no qualified ABI7 FA4 export",
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                config.is_causal
            )));
        }
        let fa4_root = std::env::var_os(infinity_v3_config::infinity_env!("FA4_ROOT"))
            .ok_or_else(|| start("decoder attention requires INFINITY_FA4_ROOT"))?;
        let fa4 = fa4::Fa4::new(context.clone(), Path::new(&fa4_root), attention_spec)?;
        let kernels = DecoderKernels::load(context, math)?;
        let bias_support = BiasSupport::from_layers(&host);
        let readout_bias = ReadoutBias::from_last_layer(host.layers.last());
        let plans = DecoderPlans::new(lt, &config, bias_support, readout_bias)?;
        let mut uploader = WeightUploader::new(stream.clone())?;
        let weights = DecoderWeightBuffers::upload(&mut uploader, stream, &host)?;
        uploader.finish()?;
        let tables = BaseRopeTables::new(
            config.head_dim,
            config.rope_theta,
            config.max_position_embeddings,
            ROPE_TABLE_BYTE_LIMIT,
        )
        .map_err(start)?;
        if tables.head_dim() != config.head_dim
            || tables.position_capacity() != config.max_position_embeddings
            || tables.table_stride() != config.head_dim / 2
            || tables.table_bytes() == 0
        {
            return Err(start("decoder rotary table geometry disagrees with config"));
        }
        let rope_cosine = DeviceBuffer::from_host(stream, tables.cosine()).map_err(start)?;
        let rope_sine = DeviceBuffer::from_host(stream, tables.sine()).map_err(start)?;
        drop(tables);
        Ok(Self {
            config,
            weights,
            plans,
            kernels,
            fa4,
            rope_cosine,
            rope_sine,
            geometry,
            uses_plain: bias_support.uses_plain(),
        })
    }

    pub(crate) fn needs_plain(&self) -> bool {
        self.uses_plain
    }

    pub(crate) fn preflight(
        &mut self,
        provider: &LtProvider,
        batch: &Batch,
        pool: &Pool,
    ) -> Result<(), ShapeError> {
        self.plans.set_rows(provider, batch.input_ids.len())?;
        if selected_readout(pool, batch.pooled_indices.len()).is_some() {
            let sequences = batch
                .cumulative_seq_lengths
                .len()
                .checked_sub(1)
                .filter(|count| *count > 0)
                .ok_or_else(|| {
                    ShapeError::Rejected(infer("Missing decoder sequence boundaries"))
                })?;
            self.plans.set_readout_rows(provider, sequences)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn enqueue(
        &mut self,
        lt: &LtProvider,
        plain_blas: Option<&Blas>,
        stream: &Stream,
        work: &mut DecoderWork,
        batch: &Batch,
        pool: &Pool,
        max_length: usize,
        lease: &mut HostIoLease,
        tokens: usize,
        output_elements: usize,
    ) -> Result<(), BackendError> {
        let c = &self.config;
        let g = self.geometry;
        let sequences = batch
            .cumulative_seq_lengths
            .len()
            .checked_sub(1)
            .ok_or_else(|| infer("Missing decoder sequence boundaries"))?;
        let selections = batch.pooled_indices.len();
        work.check(tokens, sequences, selections, &g)?;
        let readout = selected_readout(pool, selections);
        let h = g.hidden;
        let rows = u32_size(tokens)?;
        let width = u32_size(h)?;
        let eps = c.rms_norm_eps as f32;
        // Each SiLU thread owns an eight-column group. Prepare both actual
        // full/readout grids once before the layer loop, including tail groups.
        let full_silu_grid = blocks(product(tokens, g.silu_groups_per_row)?)?;
        let readout_silu_grid = blocks(product(sequences, g.silu_groups_per_row)?)?;
        let k = &self.kernels;
        let base = work.input.cu_deviceptr();
        let id_bytes = u64::try_from(product(tokens, 4)?).map_err(infer)?;
        let type_bytes = u64::try_from(product(tokens, 12)?).map_err(infer)?;
        let mut ids = base;
        let positions = base
            .checked_add(id_bytes)
            .ok_or_else(|| infer("decoder position address overflow"))?;
        let boundaries = base
            .checked_add(type_bytes)
            .ok_or_else(|| infer("decoder boundary address overflow"))?;
        let selected = boundaries
            .checked_add(
                u64::try_from(product(batch.cumulative_seq_lengths.len(), 4)?).map_err(infer)?,
            )
            .ok_or_else(|| infer("decoder selection address overflow"))?;
        let residual = work.residual.cu_deviceptr();
        let residual_next = work.residual_next.cu_deviceptr();
        let branch = work.branch.cu_deviceptr();
        let normalized = work.normalized.cu_deviceptr();
        let projection = work.projection.cu_deviceptr();
        let activation = work.activation.cu_deviceptr();
        let mut gather_out = residual;
        let mut table = self.weights.embed.cu_deviceptr();
        let mut gather_tokens = rows;
        let mut gather_width = width;
        launch(
            &k.gather,
            stream,
            blocks(product(tokens, h)?)?,
            0,
            &mut [
                arg(&mut ids),
                arg(&mut table),
                arg(&mut gather_out),
                arg(&mut gather_tokens),
                arg(&mut gather_width),
            ],
        )?;

        let mut norm_input = residual;
        let mut norm_weight = self
            .weights
            .layers
            .first()
            .ok_or_else(|| start("decoder checkpoint has no layers"))?
            .input_norm
            .cu_deviceptr();
        let mut norm_output = normalized;
        let mut rms_rows = rows;
        let mut rms_width = width;
        let mut rms_eps = eps;
        launch_threads(
            &k.rms_norm,
            stream,
            (rms_rows.div_ceil(k.norm_geometry.rows_per_cta), 1, 1),
            k.norm_geometry.threads,
            0,
            &mut [
                arg(&mut norm_input),
                arg(&mut norm_weight),
                arg(&mut norm_output),
                arg(&mut rms_rows),
                arg(&mut rms_width),
                arg(&mut rms_eps),
            ],
        )?;

        let strides = g.attention_strides(c.head_dim)?;
        let q_offset = u64::try_from(product(g.q_width, 2)?).map_err(infer)?;
        let k_offset = u64::try_from(product(g.kv_width, 2)?).map_err(infer)?;
        let q_heads = u32_size(c.num_attention_heads)?;
        let kv_heads = u32_size(c.num_key_value_heads)?;
        let head_dim = u32_size(c.head_dim)?;
        let table_stride = u32_size(c.head_dim / 2)?;
        let qkv_stride = u64::try_from(g.qkv_width).map_err(infer)?;
        let mut residual_cur = residual;
        let mut residual_next_ptr = residual_next;
        let mut cos = self.rope_cosine.cu_deviceptr();
        let mut sin = self.rope_sine.cu_deviceptr();
        let mut position_ptr = positions;
        let mut rope_tokens = rows;
        let mut rope_q_heads = q_heads;
        let mut rope_kv_heads = kv_heads;
        let mut rope_head_dim = head_dim;
        let mut rope_rotary_dim = head_dim;
        let mut q_stride = qkv_stride;
        let mut k_stride = qkv_stride;
        let mut rope_table_stride = table_stride;
        let pairs = product(tokens, c.num_attention_heads + c.num_key_value_heads)?
            .checked_mul(c.head_dim / 2)
            .ok_or_else(|| infer("decoder rotary pair count overflow"))?;
        let batch_count = sequences;
        let mut selected_done = false;

        for (index, layer) in self.weights.layers.iter().enumerate() {
            if let Some(bias) = layer.qkv.bias.as_ref() {
                let plan = self
                    .plans
                    .bias_qkv
                    .as_ref()
                    .ok_or_else(|| infer("Missing decoder QKV bias plan"))?;
                unsafe {
                    lt.enqueue(
                        plan,
                        &work.normalized,
                        &layer.qkv.weight,
                        bias,
                        &mut work.projection,
                    )?;
                }
            } else {
                let blas = plain_blas
                    .ok_or_else(|| infer("Missing decoder plain cuBLAS projection handle"))?;
                unsafe {
                    blas.linear(
                        &self.plans.plain.qkv,
                        &work.normalized,
                        &layer.qkv.weight,
                        &mut work.projection,
                    )?;
                }
            }
            let mut q = projection;
            let mut key = projection
                .checked_add(q_offset)
                .ok_or_else(|| infer("decoder Q address overflow"))?;
            let v = key
                .checked_add(k_offset)
                .ok_or_else(|| infer("decoder V address overflow"))?;
            launch(
                &k.rope_neox,
                stream,
                blocks(pairs)?,
                0,
                &mut [
                    arg(&mut q),
                    arg(&mut key),
                    arg(&mut cos),
                    arg(&mut sin),
                    arg(&mut position_ptr),
                    arg(&mut rope_tokens),
                    arg(&mut rope_q_heads),
                    arg(&mut rope_kv_heads),
                    arg(&mut rope_head_dim),
                    arg(&mut rope_rotary_dim),
                    arg(&mut q_stride),
                    arg(&mut k_stride),
                    arg(&mut rope_table_stride),
                ],
            )?;
            let attention = fa4::AttentionBatch {
                q,
                k: key,
                v,
                out: activation,
                q_offsets: boundaries,
                k_offsets: boundaries,
                q_tokens: tokens,
                k_tokens: tokens,
                batch: batch_count,
                max_q: max_length,
                strides,
            };
            // SAFETY: Engine::enqueue_inner bound this context once on this
            // thread; packed Q/K/V, the separate attention output, positions
            // and boundaries remain retained by the engine/workspace through
            // stream completion; the selected spec was qualified at load.
            unsafe { self.fa4.forward_batch_bound(stream, &attention)? };
            if readout.is_some() && index + 1 == self.weights.layers.len() {
                let readout_mode = readout.ok_or_else(|| infer("Missing selected readout mode"))?;
                let gather_widths = g
                    .q_width
                    .checked_add(h)
                    .ok_or_else(|| infer("Decoder selected row width overflow"))?;
                let gather_elements = product(sequences, gather_widths)?;
                // QKV is dead once FA4 returned; the accumulated residual has
                // already been read here, so both destinations are reusable.
                let mut gather_first = activation;
                let mut gather_second = residual_cur;
                let mut gather_bounds = boundaries;
                let mut gather_selected = 0u64;
                let mut gather_first_out = projection;
                let mut gather_second_out = residual_next_ptr;
                let mut gather_count = u32_size(sequences)?;
                let mut gather_first_width = u32_size(g.q_width)?;
                let mut gather_second_width = width;
                let mut gather_mode = readout_mode.kernel_mode();
                launch(
                    &k.selected_rows,
                    stream,
                    blocks(gather_elements)?,
                    0,
                    &mut [
                        arg(&mut gather_first),
                        arg(&mut gather_second),
                        arg(&mut gather_bounds),
                        arg(&mut gather_selected),
                        arg(&mut gather_first_out),
                        arg(&mut gather_second_out),
                        arg(&mut gather_count),
                        arg(&mut gather_first_width),
                        arg(&mut gather_second_width),
                        arg(&mut gather_mode),
                    ],
                )?;

                let selected_rows = u32_size(sequences)?;
                if let Some(bias) = layer.attention.bias.as_ref() {
                    let plan = self
                        .plans
                        .readout
                        .bias_attention
                        .as_ref()
                        .ok_or_else(|| infer("Missing selected attention bias plan"))?;
                    unsafe {
                        lt.enqueue(
                            plan,
                            &work.projection,
                            &layer.attention.weight,
                            bias,
                            &mut work.branch,
                        )?;
                    }
                } else {
                    let blas = plain_blas
                        .ok_or_else(|| infer("Missing decoder plain cuBLAS projection handle"))?;
                    unsafe {
                        blas.linear(
                            &self.plans.readout.plain_attention,
                            &work.projection,
                            &layer.attention.weight,
                            &mut work.branch,
                        )?;
                    }
                }
                let mut add_input = branch;
                let mut add_residual = residual_next_ptr;
                let mut add_weight = layer.post_attention_norm.cu_deviceptr();
                let mut add_output = normalized;
                let mut add_summed = residual_cur;
                let mut add_rows = selected_rows;
                let mut add_width = width;
                let mut add_eps = eps;
                launch_threads(
                    &k.add_rms_norm,
                    stream,
                    (add_rows.div_ceil(k.norm_geometry.rows_per_cta), 1, 1),
                    k.norm_geometry.threads,
                    0,
                    &mut [
                        arg(&mut add_input),
                        arg(&mut add_residual),
                        arg(&mut add_weight),
                        arg(&mut add_output),
                        arg(&mut add_summed),
                        arg(&mut add_rows),
                        arg(&mut add_width),
                        arg(&mut add_eps),
                    ],
                )?;
                let selected_residual = residual_cur;
                let dead_residual = residual_next_ptr;

                if let Some(bias) = layer.gate_up.bias.as_ref() {
                    let plan = self
                        .plans
                        .readout
                        .bias_gate_up
                        .as_ref()
                        .ok_or_else(|| infer("Missing selected gate/up bias plan"))?;
                    unsafe {
                        lt.enqueue(
                            plan,
                            &work.normalized,
                            &layer.gate_up.weight,
                            bias,
                            &mut work.projection,
                        )?;
                    }
                } else {
                    let blas = plain_blas
                        .ok_or_else(|| infer("Missing decoder plain cuBLAS projection handle"))?;
                    unsafe {
                        blas.linear(
                            &self.plans.readout.plain_gate_up,
                            &work.normalized,
                            &layer.gate_up.weight,
                            &mut work.projection,
                        )?;
                    }
                }
                let mut silu_input = projection;
                let mut silu_output = activation;
                let mut silu_rows = selected_rows;
                let mut silu_width = u32_size(c.intermediate_size)?;
                launch(
                    &k.silu_mul,
                    stream,
                    readout_silu_grid,
                    0,
                    &mut [
                        arg(&mut silu_input),
                        arg(&mut silu_output),
                        arg(&mut silu_rows),
                        arg(&mut silu_width),
                    ],
                )?;
                if let Some(bias) = layer.down.bias.as_ref() {
                    let plan = self
                        .plans
                        .readout
                        .bias_down
                        .as_ref()
                        .ok_or_else(|| infer("Missing selected down bias plan"))?;
                    unsafe {
                        lt.enqueue(
                            plan,
                            &work.activation,
                            &layer.down.weight,
                            bias,
                            &mut work.branch,
                        )?;
                    }
                } else {
                    let blas = plain_blas
                        .ok_or_else(|| infer("Missing decoder plain cuBLAS projection handle"))?;
                    unsafe {
                        blas.linear(
                            &self.plans.readout.plain_down,
                            &work.activation,
                            &layer.down.weight,
                            &mut work.branch,
                        )?;
                    }
                }
                let mut add_input = branch;
                let mut add_residual = selected_residual;
                let mut add_weight = self.weights.norm.cu_deviceptr();
                let mut add_output = normalized;
                let mut add_summed = dead_residual;
                let mut add_rows = selected_rows;
                let mut add_width = width;
                let mut add_eps = eps;
                launch_threads(
                    &k.add_rms_norm,
                    stream,
                    (add_rows.div_ceil(k.norm_geometry.rows_per_cta), 1, 1),
                    k.norm_geometry.threads,
                    0,
                    &mut [
                        arg(&mut add_input),
                        arg(&mut add_residual),
                        arg(&mut add_weight),
                        arg(&mut add_output),
                        arg(&mut add_summed),
                        arg(&mut add_rows),
                        arg(&mut add_width),
                        arg(&mut add_eps),
                    ],
                )?;
                let mut cast_input = normalized;
                let mut cast_output = work.pooled.cu_deviceptr();
                let mut cast_selected = selected;
                let mut cast_count = u32_size(selections)?;
                let mut cast_width = width;
                launch(
                    &k.bf16_to_f32,
                    stream,
                    blocks(product(selections, h)?)?,
                    0,
                    &mut [
                        arg(&mut cast_input),
                        arg(&mut cast_output),
                        arg(&mut cast_selected),
                        arg(&mut cast_count),
                        arg(&mut cast_width),
                    ],
                )?;
                selected_done = true;
                continue;
            }
            if let Some(bias) = layer.attention.bias.as_ref() {
                let plan = self
                    .plans
                    .bias_attention
                    .as_ref()
                    .ok_or_else(|| infer("Missing decoder attention bias plan"))?;
                unsafe {
                    lt.enqueue(
                        plan,
                        &work.activation,
                        &layer.attention.weight,
                        bias,
                        &mut work.branch,
                    )?;
                }
            } else {
                let blas = plain_blas
                    .ok_or_else(|| infer("Missing decoder plain cuBLAS projection handle"))?;
                unsafe {
                    blas.linear(
                        &self.plans.plain.attention,
                        &work.activation,
                        &layer.attention.weight,
                        &mut work.branch,
                    )?;
                }
            }
            let mut add_input = branch;
            let mut add_residual = residual_cur;
            let mut add_weight = layer.post_attention_norm.cu_deviceptr();
            let mut add_output = normalized;
            let mut add_summed = residual_next_ptr;
            let mut add_rows = rows;
            let mut add_width = width;
            let mut add_eps = eps;
            launch_threads(
                &k.add_rms_norm,
                stream,
                (add_rows.div_ceil(k.norm_geometry.rows_per_cta), 1, 1),
                k.norm_geometry.threads,
                0,
                &mut [
                    arg(&mut add_input),
                    arg(&mut add_residual),
                    arg(&mut add_weight),
                    arg(&mut add_output),
                    arg(&mut add_summed),
                    arg(&mut add_rows),
                    arg(&mut add_width),
                    arg(&mut add_eps),
                ],
            )?;
            std::mem::swap(&mut residual_cur, &mut residual_next_ptr);

            if let Some(bias) = layer.gate_up.bias.as_ref() {
                let plan = self
                    .plans
                    .bias_gate_up
                    .as_ref()
                    .ok_or_else(|| infer("Missing decoder gate/up bias plan"))?;
                unsafe {
                    lt.enqueue(
                        plan,
                        &work.normalized,
                        &layer.gate_up.weight,
                        bias,
                        &mut work.projection,
                    )?;
                }
            } else {
                let blas = plain_blas
                    .ok_or_else(|| infer("Missing decoder plain cuBLAS projection handle"))?;
                unsafe {
                    blas.linear(
                        &self.plans.plain.gate_up,
                        &work.normalized,
                        &layer.gate_up.weight,
                        &mut work.projection,
                    )?;
                }
            }
            let mut silu_input = projection;
            let mut silu_output = activation;
            let mut silu_rows = rows;
            let mut silu_width = u32_size(c.intermediate_size)?;
            launch(
                &k.silu_mul,
                stream,
                full_silu_grid,
                0,
                &mut [
                    arg(&mut silu_input),
                    arg(&mut silu_output),
                    arg(&mut silu_rows),
                    arg(&mut silu_width),
                ],
            )?;
            if let Some(bias) = layer.down.bias.as_ref() {
                let plan = self
                    .plans
                    .bias_down
                    .as_ref()
                    .ok_or_else(|| infer("Missing decoder down bias plan"))?;
                unsafe {
                    lt.enqueue(
                        plan,
                        &work.activation,
                        &layer.down.weight,
                        bias,
                        &mut work.branch,
                    )?;
                }
            } else {
                let blas = plain_blas
                    .ok_or_else(|| infer("Missing decoder plain cuBLAS projection handle"))?;
                unsafe {
                    blas.linear(
                        &self.plans.plain.down,
                        &work.activation,
                        &layer.down.weight,
                        &mut work.branch,
                    )?;
                }
            }
            if let Some(next) = self.weights.layers.get(index + 1) {
                let mut add_input = branch;
                let mut add_residual = residual_cur;
                let mut add_weight = next.input_norm.cu_deviceptr();
                let mut add_output = normalized;
                let mut add_summed = residual_next_ptr;
                let mut add_rows = rows;
                let mut add_width = width;
                let mut add_eps = eps;
                launch_threads(
                    &k.add_rms_norm,
                    stream,
                    (add_rows.div_ceil(k.norm_geometry.rows_per_cta), 1, 1),
                    k.norm_geometry.threads,
                    0,
                    &mut [
                        arg(&mut add_input),
                        arg(&mut add_residual),
                        arg(&mut add_weight),
                        arg(&mut add_output),
                        arg(&mut add_summed),
                        arg(&mut add_rows),
                        arg(&mut add_width),
                        arg(&mut add_eps),
                    ],
                )?;
                std::mem::swap(&mut residual_cur, &mut residual_next_ptr);
            }
        }

        if !selected_done {
            let mut add_input = branch;
            let mut add_residual = residual_cur;
            let mut add_weight = self.weights.norm.cu_deviceptr();
            let mut add_output = normalized;
            let mut add_summed = residual_next_ptr;
            let mut add_rows = rows;
            let mut add_width = width;
            let mut add_eps = eps;
            launch_threads(
                &k.add_rms_norm,
                stream,
                (add_rows.div_ceil(k.norm_geometry.rows_per_cta), 1, 1),
                k.norm_geometry.threads,
                0,
                &mut [
                    arg(&mut add_input),
                    arg(&mut add_residual),
                    arg(&mut add_weight),
                    arg(&mut add_output),
                    arg(&mut add_summed),
                    arg(&mut add_rows),
                    arg(&mut add_width),
                    arg(&mut add_eps),
                ],
            )?;

            let mut pool_input = normalized;
            let mut pool_bounds = boundaries;
            let mut pool_indices = selected;
            let mut pool_out = work.pooled.cu_deviceptr();
            let mut pool_count = u32_size(selections)?;
            let mut pool_width = width;
            let mut mode = match pool {
                Pool::Cls => 0u32,
                Pool::Mean => 1,
                Pool::LastToken => 2,
                Pool::Splade => return Err(infer("Oxide SPLADE unsupported")),
            };
            launch(
                &k.pool,
                stream,
                blocks(product(selections, h)?)?,
                0,
                &mut [
                    arg(&mut pool_input),
                    arg(&mut pool_bounds),
                    arg(&mut pool_indices),
                    arg(&mut pool_out),
                    arg(&mut pool_count),
                    arg(&mut pool_width),
                    arg(&mut mode),
                ],
            )?;
        }
        work.download(lease.slot_mut()?.output.as_mut_slice(), output_elements)?;
        lease.record()
    }
}

pub(crate) struct DecoderWork {
    stream: Arc<Stream>,
    tokens: usize,
    sequences: usize,
    selections: usize,
    bytes: usize,
    input: ManuallyDrop<DeviceBuffer<u32>>,
    residual: ManuallyDrop<DeviceBuffer<Bf16>>,
    residual_next: ManuallyDrop<DeviceBuffer<Bf16>>,
    branch: ManuallyDrop<DeviceBuffer<Bf16>>,
    normalized: ManuallyDrop<DeviceBuffer<Bf16>>,
    projection: ManuallyDrop<DeviceBuffer<Bf16>>,
    activation: ManuallyDrop<DeviceBuffer<Bf16>>,
    pooled: ManuallyDrop<DeviceBuffer<f32>>,
}
impl DecoderWork {
    pub(crate) fn key(&self) -> WorkKey {
        (self.tokens, self.sequences, self.selections)
    }
    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }
    pub(crate) fn fits(&self, key: WorkKey) -> bool {
        super::capacity_fits(self.key(), key, self.bytes)
    }
    pub(crate) fn allocation_bytes(key: WorkKey, c: &DecoderConfig) -> Result<usize, BackendError> {
        let (tokens, _, selections) = key;
        i32::try_from(tokens).map_err(infer)?;
        let geometry = DecoderGeometry::new(c)?;
        let hidden = product(tokens, geometry.hidden)?;
        let projection = product(tokens, geometry.projection_width)?;
        let activation = product(tokens, geometry.activation_width)?;
        let input = staging::input_capacity(key)?;
        let output = product(selections, geometry.hidden)?;
        product(input, 4)?
            .checked_add(product(output, 4)?)
            .and_then(|n| n.checked_add(product(hidden, 8).ok()?))
            .and_then(|n| n.checked_add(product(projection, 2).ok()?))
            .and_then(|n| n.checked_add(product(activation, 2).ok()?))
            .ok_or_else(|| infer("Decoder workspace arena overflow"))
    }
    pub(crate) fn new(
        stream: Arc<Stream>,
        c: &DecoderConfig,
        key: WorkKey,
    ) -> Result<Self, BackendError> {
        let bytes = Self::allocation_bytes(key, c)?;
        let (tokens, sequences, selections) = key;
        let geometry = DecoderGeometry::new(c)?;
        let input_len = staging::input_capacity(key)?;
        let hidden = product(tokens, geometry.hidden)?;
        let projection = product(tokens, geometry.projection_width)?;
        let activation = product(tokens, geometry.activation_width)?;
        let output = product(selections, geometry.hidden)?;
        let alloc = |len| {
            // SAFETY: private allocation initialized before any vectorized
            // library or kernel read on this ordered stream.
            let buffer = unsafe { DeviceBuffer::<Bf16>::uninitialized_async(&stream, len) }
                .map_err(infer)?;
            unsafe {
                cuda_core::memory::memset_d8_async(
                    buffer.cu_deviceptr(),
                    0,
                    buffer.num_bytes(),
                    stream.cu_stream(),
                )
            }
            .map_err(infer)?;
            Ok::<_, BackendError>(buffer)
        };
        let input = unsafe { DeviceBuffer::<u32>::uninitialized_async(&stream, input_len) }
            .map_err(infer)?;
        let residual = alloc(hidden)?;
        let residual_next = alloc(hidden)?;
        let branch = alloc(hidden)?;
        let normalized = alloc(hidden)?;
        let projection = alloc(projection)?;
        let activation = alloc(activation)?;
        // SAFETY: pooling writes every selected element before download.
        let pooled =
            unsafe { DeviceBuffer::<f32>::uninitialized_async(&stream, output) }.map_err(infer)?;
        metrics::counter!("oxide_arena_allocations_total").increment(1);
        super::LIVE_ARENA_BYTES.fetch_add(bytes, Ordering::Relaxed);
        super::record_live_arena_bytes();
        Ok(Self {
            tokens,
            sequences,
            selections,
            bytes,
            input: ManuallyDrop::new(input),
            residual: ManuallyDrop::new(residual),
            residual_next: ManuallyDrop::new(residual_next),
            branch: ManuallyDrop::new(branch),
            normalized: ManuallyDrop::new(normalized),
            projection: ManuallyDrop::new(projection),
            activation: ManuallyDrop::new(activation),
            pooled: ManuallyDrop::new(pooled),
            stream,
        })
    }
    fn check(
        &self,
        tokens: usize,
        sequences: usize,
        selections: usize,
        g: &DecoderGeometry,
    ) -> Result<(), BackendError> {
        if tokens > self.tokens || sequences > self.sequences || selections > self.selections {
            return Err(infer("Decoder request exceeds retained arena capacity"));
        }
        let input = super::input_elements(
            tokens,
            sequences
                .checked_add(1)
                .ok_or_else(|| infer("Decoder boundary count overflow"))?,
            selections,
        )?;
        if input > staging::input_capacity(self.key())? {
            return Err(infer("Decoder metadata exceeds retained arena"));
        }
        for (len, elements) in [
            (self.residual.len(), product(tokens, g.hidden)?),
            (self.residual_next.len(), product(tokens, g.hidden)?),
            (self.branch.len(), product(tokens, g.hidden)?),
            (self.normalized.len(), product(tokens, g.hidden)?),
            (self.projection.len(), product(tokens, g.projection_width)?),
            (self.activation.len(), product(tokens, g.activation_width)?),
            (self.pooled.len(), product(selections, g.hidden)?),
        ] {
            if elements > len {
                return Err(infer("Decoder slab exceeds retained arena"));
            }
        }
        Ok(())
    }
    pub(crate) fn stage(
        &self,
        b: &Batch,
        hidden: usize,
        lease: &mut HostIoLease,
    ) -> Result<(usize, usize), BackendError> {
        stage_batch(b, hidden, lease)
    }
    pub(crate) fn upload(&self, host: &[u32], elements: usize) -> Result<(), BackendError> {
        upload_input(&self.stream, &self.input, host, elements)
    }
    pub(crate) fn download(&self, host: &mut [f32], elements: usize) -> Result<(), BackendError> {
        download_output(&self.stream, &self.pooled, host, elements)
    }
}
impl Drop for DecoderWork {
    fn drop(&mut self) {
        if self.stream.synchronize().is_ok() {
            // SAFETY: every field is dropped once after completion. Failed
            // drains deliberately retain device allocations.
            unsafe {
                ManuallyDrop::drop(&mut self.input);
                ManuallyDrop::drop(&mut self.residual);
                ManuallyDrop::drop(&mut self.residual_next);
                ManuallyDrop::drop(&mut self.branch);
                ManuallyDrop::drop(&mut self.normalized);
                ManuallyDrop::drop(&mut self.projection);
                ManuallyDrop::drop(&mut self.activation);
                ManuallyDrop::drop(&mut self.pooled);
            }
            super::LIVE_ARENA_BYTES.fetch_sub(self.bytes, Ordering::Relaxed);
            super::record_live_arena_bytes();
        } else {
            metrics::counter!("oxide_arena_retained_bytes_total").increment(self.bytes as u64);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder_model::DecoderConfig;
    use std::{
        fs,
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
    };

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            Self::with_config(
                r#"{
                    "model_type": "qwen2",
                    "architectures": ["Qwen2ForCausalLM"],
                    "hidden_size": 1536,
                    "intermediate_size": 8960,
                    "num_hidden_layers": 28,
                    "num_attention_heads": 12,
                    "num_key_value_heads": 2,
                    "head_dim": 128,
                    "vocab_size": 151646,
                    "max_position_embeddings": 131072,
                    "rms_norm_eps": 1e-6,
                    "rope_theta": 1000000.0,
                    "hidden_act": "silu",
                    "is_causal": false
                }"#,
            )
        }
        fn with_config(config: &str) -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "infinity-decoder-executor-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            fs::write(path.join("config.json"), config).unwrap();
            Self(path)
        }
        fn config(&self) -> DecoderConfig {
            DecoderConfig::load(&self.0).unwrap()
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn decoder_geometry_and_bytes_follow_actual_slabs_not_bert_formula() {
        let fixture = Fixture::new();
        let c = fixture.config();
        let g = DecoderGeometry::new(&c).expect("geometry");
        assert_eq!(g.hidden, 1536);
        assert_eq!(g.q_width, 1536);
        assert_eq!(g.kv_width, 256);
        assert_eq!(g.qkv_width, 2048);
        assert_eq!(g.projection_width, 17920);
        assert_eq!(g.activation_width, 8960);
        assert_eq!(g.silu_groups_per_row, 1120);
        assert_eq!(
            blocks(product(32, g.silu_groups_per_row).unwrap()).unwrap(),
            (140, 1, 1)
        );
        assert_eq!(
            blocks(product(4, g.silu_groups_per_row).unwrap()).unwrap(),
            (18, 1, 1)
        );
        let key = (32usize, 4, 2);
        let bytes = DecoderWork::allocation_bytes(key, &c).expect("bytes");
        let input = staging::input_capacity(key).unwrap();
        let expected = input * 4
            + key.2 * g.hidden * 4
            + key.0 * g.hidden * 8
            + key.0 * g.projection_width * 2
            + key.0 * g.activation_width * 2;
        assert_eq!(bytes, expected);
        let scratch = bytes - input * 4 - key.2 * g.hidden * 4;
        assert_eq!(scratch / key.0, 66048);
        assert!(
            DecoderWork::allocation_bytes((2048, 2, 1), &c).unwrap()
                <= super::super::cache::BYTE_BUDGET
        );
        assert!(
            DecoderWork::allocation_bytes((4096, 2, 1), &c).unwrap()
                > super::super::cache::BYTE_BUDGET
        );
        assert!(DecoderWork::allocation_bytes((usize::MAX, 2, 1), &c).is_err());
    }

    #[test]
    fn rope_table_budget_has_explicit_room_for_gte_capacity() {
        let fixture = Fixture::new();
        let c = fixture.config();
        let required = c
            .max_position_embeddings
            .checked_mul(c.head_dim / 2)
            .and_then(|n| n.checked_mul(2))
            .and_then(|n| n.checked_mul(4))
            .and_then(|n| n.checked_add(c.head_dim / 2 * 4))
            .unwrap();
        assert_eq!(required, 67_109_120);
        assert!(required <= ROPE_TABLE_BYTE_LIMIT);
        let tables = BaseRopeTables::new(
            c.head_dim,
            c.rope_theta,
            c.max_position_embeddings,
            ROPE_TABLE_BYTE_LIMIT,
        )
        .unwrap();
        assert_eq!(tables.head_dim(), 128);
        assert_eq!(tables.position_capacity(), 131072);
        assert_eq!(tables.table_stride(), 64);
        assert_eq!(tables.table_bytes(), 67_108_864);
        assert_eq!(tables.validate_positions(&[0, 131071]), Ok(()));
        assert!(tables.validate_positions(&[131072]).is_err());
    }

    fn tinyllama_config() -> &'static str {
        // Mirrors decoder-loader-fixtures/tinyllama-config.json without a
        // machine-specific or /workspace path.
        r#"{
            "architectures": ["LlamaForCausalLM"],
            "attention_bias": false,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_act": "silu",
            "hidden_size": 2048,
            "initializer_range": 0.02,
            "intermediate_size": 5632,
            "max_position_embeddings": 2048,
            "model_type": "llama",
            "num_attention_heads": 32,
            "num_hidden_layers": 22,
            "num_key_value_heads": 4,
            "pretraining_tp": 1,
            "rms_norm_eps": 1e-05,
            "rope_scaling": null,
            "rope_theta": 10000.0,
            "tie_word_embeddings": false,
            "torch_dtype": "bfloat16",
            "transformers_version": "4.35.0",
            "use_cache": true,
            "vocab_size": 32000
        }"#
    }

    fn fixture_with_overrides(overrides: serde_json::Value) -> Fixture {
        let mut value: serde_json::Value = serde_json::from_str(tinyllama_config()).unwrap();
        for (key, field) in overrides.as_object().unwrap() {
            value[key] = field.clone();
        }
        Fixture::with_config(&serde_json::to_string(&value).unwrap())
    }

    #[test]
    fn qwen2_masked_candle_recipe_is_rejected_before_generic_dispatch() {
        let base = Fixture::new();
        let mut config: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(base.0.join("config.json")).unwrap()).unwrap();
        config["hidden_size"] = 896.into();
        for (head_dim, heads) in [(64, 14), (128, 7)] {
            config["head_dim"] = head_dim.into();
            config["num_attention_heads"] = heads.into();
            config["num_key_value_heads"] = heads.into();
            for causal in [false, true] {
                config["is_causal"] = causal.into();
                let fixture = Fixture::with_config(&config.to_string());
                let loaded = fixture.config();
                if head_dim == 64 {
                    if !causal {
                        assert!(fa4::Fa4::supports(&fa4::AttentionSpec {
                            heads: loaded.num_attention_heads,
                            kv_heads: loaded.num_key_value_heads,
                            head_dim: loaded.head_dim,
                            causal,
                            window: fa4::Window::Global,
                            dtype: fa4::DType::Bf16,
                            scale: fa4::Scale::RsqrtHeadDim,
                        }));
                    }
                    let error = DecoderMath::select(&loaded).unwrap_err().to_string();
                    assert!(error.contains("masked Candle RMS and rotary"));
                } else {
                    assert_eq!(
                        DecoderMath::select(&loaded).unwrap(),
                        DecoderMath::Qwen2FullNeox
                    );
                }
            }
        }
    }

    #[test]
    fn decoder_math_selects_qwen2_and_narrow_llama_h2048_recipes() {
        let fixture = Fixture::new();
        let qwen = fixture.config();
        assert_eq!(qwen.family, DecoderFamily::Qwen2);
        assert!(!qwen.is_causal);
        assert_eq!(qwen.rope, DecoderRope::FullNeoX);
        assert_eq!(
            DecoderMath::select(&qwen).unwrap(),
            DecoderMath::Qwen2H1536FullNeox
        );
        assert_eq!(
            DecoderMath::Qwen2H1536FullNeox.norm_geometry(),
            GENERIC_NORM_GEOMETRY
        );
        assert_eq!(
            DecoderMath::Qwen2H1536FullNeox.rms_export(),
            "infinity_rms_norm_1536_cached"
        );
        assert_eq!(
            DecoderMath::Qwen2H1536FullNeox.add_rms_export(),
            "infinity_add_rms_norm_1536_cached"
        );
        let mut other_qwen = qwen.clone();
        other_qwen.hidden_size = 1024;
        assert_eq!(
            DecoderMath::select(&other_qwen).unwrap(),
            DecoderMath::Qwen2FullNeox
        );
        assert_eq!(
            DecoderMath::Qwen2FullNeox.norm_geometry(),
            GENERIC_NORM_GEOMETRY
        );
        assert_eq!(DecoderMath::Qwen2FullNeox.rms_export(), "infinity_rms_norm");
        assert_eq!(
            DecoderMath::Qwen2FullNeox.rope_export(),
            "infinity_rope_neox"
        );
        let invalid = BaseRopeTables::new(
            qwen.head_dim,
            qwen.rope_theta,
            qwen.max_position_embeddings,
            67_109_119,
        );
        assert!(invalid.is_err());

        let llama = fixture_with_overrides(serde_json::json!({})).config();
        assert_eq!(llama.family, DecoderFamily::Llama);
        assert!(llama.is_causal);
        assert_eq!(llama.rope, DecoderRope::Default);
        assert_eq!(
            DecoderMath::select(&llama).unwrap(),
            DecoderMath::LlamaH2048Exact
        );
        assert_eq!(
            DecoderMath::LlamaH2048Exact.norm_geometry(),
            H2048_NORM_GEOMETRY
        );
        assert_eq!(H2048_NORM_GEOMETRY.rows_per_cta, 4);
        assert_eq!(H2048_NORM_GEOMETRY.threads, 128);
        assert_eq!(
            DecoderMath::LlamaH2048Exact.rms_export(),
            "infinity_rms_norm_2048_vector"
        );
        assert_eq!(
            DecoderMath::LlamaH2048Exact.add_rms_export(),
            "infinity_add_rms_norm_2048_vector"
        );
        assert_eq!(
            DecoderMath::LlamaH2048Exact.rope_export(),
            "infinity_rope_neox_hfma"
        );
        // Native DenseMlp code5 replaces exp/denominator/SiLU with BF16-rounded
        // intermediates; the F32 exp variant would not match that policy.
        assert_eq!(
            DecoderMath::LlamaH2048Exact.activation_export(),
            "infinity_silu_mul_bf16_vector8_native"
        );
        assert_ne!(
            DecoderMath::LlamaH2048Exact.activation_export(),
            "infinity_silu_mul_f32"
        );
        assert_eq!(
            DecoderMath::Qwen2FullNeox.activation_export(),
            "infinity_silu_mul_bf16_vector8_native"
        );

        let h4096 = fixture_with_overrides(serde_json::json!({"hidden_size": 4096}));
        let error = validate_recipe(&h4096.config()).unwrap_err().to_string();
        assert!(error.contains("hidden_size 2048"));
        assert!(error.contains("H4096 is deferred"));

        let non_default_rope = fixture_with_overrides(serde_json::json!({"rope_type": "simple"}));
        let error = validate_recipe(&non_default_rope.config())
            .unwrap_err()
            .to_string();
        assert!(error.contains("base unscaled Llama default RoPE"));

        let noncausal = fixture_with_overrides(serde_json::json!({"is_causal": false}));
        let error = validate_recipe(&noncausal.config())
            .unwrap_err()
            .to_string();
        assert!(error.contains("causal global Llama attention"));

        let simple = Fixture::with_config(
            r#"{
                "model_type": "qwen2",
                "architectures": ["Qwen2ForCausalLM"],
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "vocab_size": 12,
                "max_position_embeddings": 32,
                "rms_norm_eps": 1e-6,
                "rope_type": "simple",
                "hidden_act": "silu",
                "is_causal": false
            }"#,
        );
        let error = validate_recipe(&simple.config()).unwrap_err().to_string();
        assert!(error.contains("full split-half NeoX"));
    }

    #[test]
    fn llama_h2048_geometry_tracks_tinyllama_projection_and_activation_widths() {
        let fixture = fixture_with_overrides(serde_json::json!({}));
        let c = fixture.config();
        assert_eq!((c.hidden_size, c.intermediate_size), (2048, 5632));
        assert_eq!((c.num_attention_heads, c.num_key_value_heads), (32, 4));
        assert_eq!(c.head_dim, 64);
        assert_eq!(c.max_position_embeddings, 2048);
        assert_eq!(c.num_hidden_layers, 22);
        assert_eq!(c.rope_theta, 10_000.0);
        let g = DecoderGeometry::new(&c).expect("geometry");
        assert_eq!(g.hidden, 2048);
        assert_eq!((g.q_width, g.kv_width, g.qkv_width), (2048, 256, 2560));
        assert_eq!(g.projection_width, 11264);
        assert_eq!(g.activation_width, 5632);
        assert_eq!(g.silu_groups_per_row, 704);
        assert_eq!(
            blocks(product(128, g.silu_groups_per_row).unwrap()).unwrap(),
            (352, 1, 1)
        );
        assert_eq!(
            blocks(product(2, g.silu_groups_per_row).unwrap()).unwrap(),
            (6, 1, 1)
        );
        let bytes = DecoderWork::allocation_bytes((2048, 2, 1), &c).expect("bytes");
        assert!(bytes <= super::super::cache::BYTE_BUDGET);
        assert!(
            DecoderWork::allocation_bytes((256, 2, 1), &c).unwrap()
                < DecoderWork::allocation_bytes((2048, 2, 1), &c).unwrap()
        );
    }

    #[test]
    fn plan_rows_retry_failed_readout_refresh_without_invalidating_full_plans() {
        let mut rows = PlanRows {
            full: Some(5),
            readout: Some(2),
        };
        let mut calls = 0;
        let failed = crate::math::refresh_plan_rows(&mut rows.readout, 3, || {
            calls += 1;
            Err("injected readout descriptor check failure")
        });
        assert!(failed.is_err());
        assert_eq!(rows.readout, None);
        assert_eq!(rows.full, Some(5));
        crate::math::refresh_plan_rows(&mut rows.readout, 3, || {
            calls += 1;
            Ok::<_, &str>(())
        })
        .unwrap();
        assert_eq!(calls, 2, "the same failed readout shape must be retried");
        assert_eq!(rows.readout, Some(3));

        // A readout failure must not force the unchanged full-T plans to run
        // their driver checks again.
        crate::math::refresh_plan_rows(&mut rows.full, 5, || {
            calls += 1;
            Ok::<_, &str>(())
        })
        .unwrap();
        assert_eq!(calls, 2);
    }

    #[test]
    fn selected_readout_is_cls_or_last_for_any_nonempty_selection_count() {
        assert_eq!(selected_readout(&Pool::Cls, 1), Some(SelectedReadout::Cls));
        assert_eq!(
            selected_readout(&Pool::LastToken, 2),
            Some(SelectedReadout::LastToken)
        );
        // The bridge always projects canonical B rows; duplicate excess only
        // changes the final request-order extraction, so S>B stays on the fast
        // path and never selects a different row count for the projections.
        assert_eq!(selected_readout(&Pool::Cls, 4), Some(SelectedReadout::Cls));
        assert_eq!(
            selected_readout(&Pool::LastToken, 9),
            Some(SelectedReadout::LastToken)
        );
        assert_eq!(selected_readout(&Pool::Cls, 0), None);
        assert_eq!(selected_readout(&Pool::Mean, 2), None);
        assert_eq!(selected_readout(&Pool::Splade, 2), None);
    }

    #[test]
    fn selected_readout_geometry_keeps_independent_attention_and_hidden_widths() {
        let fixture = Fixture::with_config(
            r#"{
                "model_type": "qwen2",
                "architectures": ["Qwen2ForCausalLM"],
                "hidden_size": 8,
                "intermediate_size": 17,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 6,
                "vocab_size": 12,
                "max_position_embeddings": 32,
                "rms_norm_eps": 1e-6,
                "hidden_act": "silu",
                "is_causal": false
            }"#,
        );
        let c = fixture.config();
        let g = DecoderGeometry::new(&c).expect("geometry");
        assert_eq!(
            (g.hidden, g.q_width, g.kv_width, g.qkv_width),
            (8, 12, 6, 24)
        );
        assert_eq!(g.projection_width, 34);
        assert_eq!(g.activation_width, 17);
        assert_eq!(
            g.silu_groups_per_row, 3,
            "partial groups round each row independently"
        );
        // Canonical B sequence rows fit the token-shaped projection/residual
        // slabs before the final request-order extraction.
        let sequences = 2usize;
        let tokens = 5usize;
        assert_eq!(product(tokens, g.silu_groups_per_row).unwrap(), 15);
        assert_eq!(product(sequences, g.silu_groups_per_row).unwrap(), 6);
        assert!(sequences <= tokens);
        assert!(sequences * g.q_width <= tokens * g.projection_width);
        assert!(sequences * g.hidden <= tokens * g.hidden);
        assert!(sequences * c.intermediate_size <= tokens * g.activation_width);
    }
}
