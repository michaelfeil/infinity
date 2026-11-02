// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Private dense Gemma4 E2B text executor candidate.
//!
//! This is not a public model route. It executes the released dense E2B text
//! geometry (1536 hidden, 35 layers, PLE 256) from real checkpoint views,
//! uploads every text weight once, and performs the Native BF16 norm/PLE/MLP
//! composition. Global d512 attention is supplied by the separate ABI2
//! Head512 provider artifact. Sliding d256 attention is an isolated device
//! leaf added in `device/src/gemma4.rs`; it is a semantic bridge, not a
//! bitwise replacement for Native FlashAttention's online-softmax tree.
//!
//! Text (noncausal) and Rune option_tokens (causal) full-T backbones are both
//! admitted. The crate-private prepared Rune tied option-row/softcap head is
//! implemented on the same canonical embedding and final normalized BF16 Work.
//! Media towers and a public Engine/Backend decision route remain pending.
//! Serving construction rejects a missing qualified FA4 local d256 profile;
//! the isolated FP32-score/softmax leaf is reachable only through the explicit
//! diagnostic constructor, never as an automatic fallback.
#![allow(dead_code)]

use crate::cublas::Blas;
use crate::gemma2_math::{MeanPlan, PreparedGemma2};
use crate::gemma4_option_head::{
    Gemma4OptionHead, Gemma4OptionWork, OptionKernels, PreparedGemma4OptionHead,
};
use crate::gemma4_text_model::{
    Gemma4Config, Gemma4Contract, Gemma4KvPlan, Gemma4KvWeights, Gemma4LayerKind, Gemma4LayerPlan,
    Gemma4Plan, Gemma4Variant, Gemma4Weights,
};
use crate::gemma4_vision_model::Gemma4VisionWork;
use crate::lt::ShapeError;
use crate::math::LinearPlan;
use crate::moe_runtime::{Gemma4MoeLayer, MoeProvider, MoeScratch};
use crate::staging::HostIoLease;
use crate::tensor::{Bf16, F32};
use crate::upload::WeightUploader;
use crate::work_retirement::{CompletedTicket, CompletionTicket};
use crate::{arg, blocks, infer, launch, launch_threads, product, start, u32_size, WorkKey};
use cuda_core::{Context, DeviceBuffer, Function, Stream};
use infinity_head512_aot::{
    Capabilities as Head512Capabilities, DType as Head512DType, DeviceRegion, Head512Batch,
    Head512Spec, Head512Work, Provider as Head512Provider, Window as Head512Window,
};
use infinity_v3_backend_core::{BackendError, Batch, DecisionInput, Pool};
use std::fmt;
use std::mem::ManuallyDrop;
use std::path::{Path, PathBuf};
use std::sync::{atomic::Ordering, Arc};

const E2B_HIDDEN: usize = 1536;
const E2B_INTERMEDIATE: usize = 6144;
const E2B_LAYERS: usize = 35;
const E2B_PLE: usize = 256;
const E2B_HEADS: usize = 8;
const E2B_HEAD_DIM: usize = 256;
const E2B_GLOBAL_HEAD_DIM: usize = 512;
const E2B_KV_HEADS: usize = 1;
const E2B_SHARED_LAYERS: usize = 20;
const E2B_MAX_POSITIONS: usize = 262_144;
const E2B_TABLE_BUDGET_BYTES: usize = 1024 * 1024 * 1024;

#[cfg(test)]
static INJECT_TEXT_FINISH_AFTER_POOLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

const GEMMA4_KERNEL_EXPORTS: [&str; 13] = [
    "infinity_gemma4_norm_bf16_f32scale",
    "infinity_gemma4_residual_norm_bf16_f32scale",
    "infinity_gemma4_unit_norm_bf16",
    "infinity_gemma4_gated_gelu_bf16",
    "infinity_gemma4_ple_gate_bf16",
    "infinity_gemma4_ple_combine_bf16",
    "infinity_gemma4_scale_bf16",
    "infinity_gemma4_rope_tables_bf16",
    "infinity_gemma4_rope_q_bf16",
    "infinity_gemma4_attention_256_bf16",
    "infinity_rope_neox",
    "infinity_embedding_gather_scaled",
    "infinity_gemma4_select_rows_bf16",
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum Gemma4E2bUnsupported {
    MoePending,
    MediaPending,
    RawRowsPending,
    CompactionPending,
    PoolingPending,
    ProviderPending,
    LocalAttentionPending,
    Geometry(String),
}

impl fmt::Display for Gemma4E2bUnsupported {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MoePending => write!(formatter, "Gemma4 E2B MoE routing is pending"),
            Self::MediaPending => write!(formatter, "Gemma4 E2B media towers are pending"),
            Self::RawRowsPending => write!(formatter, "Gemma4 E2B raw token rows are pending"),
            Self::CompactionPending => {
                write!(formatter, "Gemma4 E2B compact/radix batches are pending")
            }
            Self::PoolingPending => {
                write!(formatter, "Gemma4 E2B requested pooling is unsupported")
            }
            Self::ProviderPending => write!(
                formatter,
                "Gemma4 E2B global d512 Head512 provider is not qualified"
            ),
            Self::LocalAttentionPending => write!(
                formatter,
                "Gemma4 E2B local d256 FA4 profile is not qualified; semantic leaf is diagnostic only"
            ),
            Self::Geometry(error) => write!(formatter, "Gemma4 E2B geometry: {error}"),
        }
    }
}

impl std::error::Error for Gemma4E2bUnsupported {}

#[derive(Debug)]
pub(crate) enum Gemma4E2bStartupError {
    Unsupported(Gemma4E2bUnsupported),
    Backend(BackendError),
}

impl fmt::Display for Gemma4E2bStartupError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported(error) => error.fmt(formatter),
            Self::Backend(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for Gemma4E2bStartupError {}
impl From<BackendError> for Gemma4E2bStartupError {
    fn from(error: BackendError) -> Self {
        Self::Backend(error)
    }
}
impl From<Gemma4E2bUnsupported> for Gemma4E2bStartupError {
    fn from(error: Gemma4E2bUnsupported) -> Self {
        Self::Unsupported(error)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Gemma4E2bMath {
    hidden: usize,
    layers: usize,
    ple_width: usize,
    norm_eps: f32,
    embedding_scale: f32,
    ple_model_scale: f32,
    ple_token_scale: f32,
    ple_combine_scale: f32,
    causal: bool,
    local_left: i32,
    local_right: i32,
    local_support: u32,
    local_heads: u32,
    local_kv_heads: usize,
    local_head_dim: usize,
    global_heads: u32,
    global_kv_heads: usize,
    global_head_dim: usize,
    moe: Option<(usize, usize, usize)>,
}

impl Gemma4E2bMath {
    pub(crate) fn admit(weights: &Gemma4Weights) -> Result<Self, Gemma4E2bUnsupported> {
        let config = &weights.config;
        let layers = config.num_hidden_layers;
        if layers == 0 || weights.layers.len() != layers || weights.plan.layers.len() != layers {
            return Err(Gemma4E2bUnsupported::Geometry(
                "Gemma4 layer counts disagree".into(),
            ));
        }
        let causal = weights.plan.causal;
        match config.contract {
            Gemma4Contract::Embedding if causal => {
                return Err(Gemma4E2bUnsupported::Geometry(
                    "noncausal text contract selected a causal Gemma4 plan".into(),
                ))
            }
            Gemma4Contract::RuneOptionTokens if !causal => {
                return Err(Gemma4E2bUnsupported::Geometry(
                    "Rune option_tokens requires a causal Gemma4 plan".into(),
                ))
            }
            _ => {}
        }
        let ple_width = config.hidden_size_per_layer_input;
        match (ple_width, weights.ple.as_ref(), weights.plan.ple) {
            (0, None, None) => {}
            (0, _, _) => {
                return Err(Gemma4E2bUnsupported::Geometry(
                    "Gemma4 PLE weights exist without hidden_size_per_layer_input".into(),
                ))
            }
            (width, Some(ple), Some(plan)) => {
                let total = width.checked_mul(layers).ok_or_else(|| {
                    Gemma4E2bUnsupported::Geometry("Gemma4 PLE width overflow".into())
                })?;
                if plan.width != width
                    || plan.layers != layers
                    || plan.vocab_size != config.vocab_size_per_layer_input
                    || ple.embed_tokens.shape() != [config.vocab_size_per_layer_input, total]
                    || ple.model_projection.shape() != [total, config.hidden_size]
                    || ple.projection_norm.shape() != [width]
                {
                    return Err(Gemma4E2bUnsupported::Geometry(
                        "Gemma4 PLE tensor geometry disagrees".into(),
                    ));
                }
            }
            _ => {
                return Err(Gemma4E2bUnsupported::Geometry(
                    "Gemma4 PLE plan/weights are incomplete".into(),
                ))
            }
        }
        let mut local: Option<(usize, usize, usize, i32, i32)> = None;
        let mut global: Option<(usize, usize, usize)> = None;
        let mut moe: Option<(usize, usize, usize)> = None;
        let mut any_moe = false;
        for (index, (plan, layer)) in weights.plan.layers.iter().zip(&weights.layers).enumerate() {
            if plan.index != index {
                return Err(Gemma4E2bUnsupported::Geometry(
                    "Gemma4 layer plan index disagrees".into(),
                ));
            }
            if plan.moe.is_some() != layer.moe.is_some() {
                return Err(Gemma4E2bUnsupported::Geometry(format!(
                    "Gemma4 layer {index} MoE plan/projection mismatch"
                )));
            }
            match (layer.kv.weights_present(), &plan.kv) {
                (true, Gemma4KvPlan::Producer { .. }) | (false, Gemma4KvPlan::Shared { .. }) => {}
                _ => {
                    return Err(Gemma4E2bUnsupported::Geometry(format!(
                        "Gemma4 layer {index} KV plan/projection mismatch"
                    )))
                }
            }
            let q_heads = plan
                .head_dim
                .checked_div(1)
                .and_then(|_| plan.q_width.checked_div(plan.head_dim))
                .ok_or_else(|| {
                    Gemma4E2bUnsupported::Geometry(format!(
                        "Gemma4 layer {index} Q geometry is empty"
                    ))
                })?;
            if q_heads == 0 || plan.head_dim % 2 != 0 || !plan.head_dim.is_multiple_of(4) {
                return Err(Gemma4E2bUnsupported::Geometry(format!(
                    "Gemma4 layer {index} Q/head geometry is unsupported"
                )));
            }
            if plan.kv_width
                != plan
                    .num_kv_heads
                    .checked_mul(plan.head_dim)
                    .ok_or_else(|| {
                        Gemma4E2bUnsupported::Geometry("Gemma4 KV width overflow".into())
                    })?
            {
                return Err(Gemma4E2bUnsupported::Geometry(format!(
                    "Gemma4 layer {index} KV geometry disagrees"
                )));
            }
            if let Some(ple_layer) = layer.ple.as_ref() {
                if ple_width == 0
                    || plan.ple.is_none()
                    || ple_layer.input_gate.weight.shape()
                        != [ple_width, config.hidden_size].as_slice()
                    || ple_layer.projection.weight.shape()
                        != [config.hidden_size, ple_width].as_slice()
                    || ple_layer.post_norm.shape() != [config.hidden_size]
                {
                    return Err(Gemma4E2bUnsupported::Geometry(format!(
                        "Gemma4 layer {index} PLE geometry disagrees"
                    )));
                }
            } else if ple_width != 0 || plan.ple.is_some() {
                return Err(Gemma4E2bUnsupported::Geometry(format!(
                    "Gemma4 layer {index} PLE projection missing"
                )));
            }
            if let Some(layer_moe) = plan.moe {
                any_moe = true;
                let current = (
                    layer_moe.num_experts,
                    layer_moe.top_k,
                    layer_moe.intermediate_size,
                );
                if let Some(existing) = moe {
                    if existing != current {
                        return Err(Gemma4E2bUnsupported::Geometry(
                            "Gemma4 MoE shape differs across layers".into(),
                        ));
                    }
                } else {
                    moe = Some(current);
                }
            }
            match plan.kind {
                Gemma4LayerKind::Sliding => {
                    let geometry = (
                        q_heads,
                        plan.num_kv_heads,
                        plan.head_dim,
                        plan.window_left,
                        plan.window_right,
                    );
                    if let Some(existing) = local {
                        if existing != geometry {
                            return Err(Gemma4E2bUnsupported::Geometry(
                                "Gemma4 local attention geometry differs across layers".into(),
                            ));
                        }
                    } else {
                        local = Some(geometry);
                    }
                }
                Gemma4LayerKind::Full => {
                    let geometry = (q_heads, plan.num_kv_heads, plan.head_dim);
                    if let Some(existing) = global {
                        if existing != geometry {
                            return Err(Gemma4E2bUnsupported::Geometry(
                                "Gemma4 global attention geometry differs across layers".into(),
                            ));
                        }
                    } else {
                        global = Some(geometry);
                    }
                }
            }
        }
        match weights.plan.variant {
            Gemma4Variant::SparseMoe if !any_moe => {
                return Err(Gemma4E2bUnsupported::Geometry(
                    "Gemma4 sparse variant has no routed layers".into(),
                ))
            }
            Gemma4Variant::Dense if any_moe => {
                return Err(Gemma4E2bUnsupported::Geometry(
                    "Gemma4 dense variant has routed layers".into(),
                ))
            }
            _ => {}
        }
        let (local_heads, local_kv_heads, local_head_dim, local_left, local_right) = local
            .ok_or_else(|| Gemma4E2bUnsupported::Geometry("Gemma4 has no sliding layer".into()))?;
        let (global_heads, global_kv_heads, global_head_dim) = global
            .ok_or_else(|| Gemma4E2bUnsupported::Geometry("Gemma4 has no full layer".into()))?;
        let norm_eps = config.rms_norm_eps as f32;
        if !norm_eps.is_finite() || norm_eps <= 0.0 {
            return Err(Gemma4E2bUnsupported::Geometry(
                "Gemma4 rms_norm_eps is not positive as f32".into(),
            ));
        }
        let embedding_scale = bf16_scalar((config.hidden_size as f64).sqrt() as f32)?;
        let ple_model_scale = if ple_width > 0 {
            bf16_scalar((config.hidden_size as f64).sqrt().recip() as f32)?
        } else {
            0.0
        };
        let ple_token_scale = if ple_width > 0 {
            bf16_scalar((ple_width as f64).sqrt() as f32)?
        } else {
            0.0
        };
        let ple_combine_scale = if ple_width > 0 {
            bf16_scalar(std::f64::consts::FRAC_1_SQRT_2 as f32)?
        } else {
            0.0
        };
        let local_support = u32::try_from(
            (local_left as usize)
                .checked_add(local_right as usize)
                .and_then(|value| value.checked_add(1))
                .ok_or_else(|| {
                    Gemma4E2bUnsupported::Geometry("Gemma4 local support overflow".into())
                })?,
        )
        .map_err(|_| Gemma4E2bUnsupported::Geometry("Gemma4 local support overflow".into()))?;
        Ok(Self {
            hidden: config.hidden_size,
            layers,
            ple_width,
            norm_eps,
            embedding_scale,
            ple_model_scale,
            ple_token_scale,
            ple_combine_scale,
            causal,
            local_left,
            local_right,
            local_support,
            local_heads: u32::try_from(local_heads).map_err(|_| {
                Gemma4E2bUnsupported::Geometry("Gemma4 local head count overflow".into())
            })?,
            local_kv_heads,
            local_head_dim,
            global_heads: u32::try_from(global_heads).map_err(|_| {
                Gemma4E2bUnsupported::Geometry("Gemma4 global head count overflow".into())
            })?,
            global_kv_heads,
            global_head_dim,
            moe,
        })
    }
}

fn reference_threads(width: usize) -> u32 {
    let mut threads = 1usize;
    while threads < width && threads < 1024 {
        threads <<= 1;
    }
    threads as u32
}

fn bf16_scalar(value: f32) -> Result<f32, Gemma4E2bUnsupported> {
    if !value.is_finite() {
        return Err(Gemma4E2bUnsupported::Geometry(
            "BF16 scalar must be finite".into(),
        ));
    }
    let rounded = half::bf16::from_f32(value).to_f32();
    if !rounded.is_finite() {
        return Err(Gemma4E2bUnsupported::Geometry(
            "BF16 scalar is not finite after rounding".into(),
        ));
    }
    Ok(rounded)
}

const ROPE_BLOCK: usize = 256;

/// Cold-checked rotary launch geometry. The published Q/K rope kernels use
/// one scalar thread per rotary pair: tokens * heads * (rotary_dim / 2).
/// `blocks()` rounds that total up to a 256-thread CTA; excess lanes observe an
/// index beyond the pair count and return without touching storage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RopeLaunchPlan {
    total_threads: usize,
}

impl RopeLaunchPlan {
    fn new(
        tokens: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
    ) -> Result<Self, BackendError> {
        if tokens == 0 || q_heads == 0 || head_dim == 0 {
            return Err(infer("Gemma4 rope launch has a zero extent"));
        }
        if rotary_dim == 0 || rotary_dim > head_dim || !rotary_dim.is_multiple_of(2) {
            return Err(infer("Gemma4 rope rotary dimension is invalid"));
        }
        let heads = q_heads
            .checked_add(kv_heads)
            .ok_or_else(|| infer("Gemma4 rope head count overflow"))?;
        let pairs = rotary_dim / 2;
        let total_threads = product(tokens, heads)?
            .checked_mul(pairs)
            .ok_or_else(|| infer("Gemma4 rope pair extent overflow"))?;
        Ok(Self { total_threads })
    }

    fn total_threads(self) -> usize {
        self.total_threads
    }

    fn grid(self) -> Result<(u32, u32, u32), BackendError> {
        blocks(self.total_threads)
    }

    fn grid_threads(self) -> Result<usize, BackendError> {
        let (x, _, _) = self.grid()?;
        usize::try_from(x)
            .ok()
            .and_then(|x| x.checked_mul(ROPE_BLOCK))
            .ok_or_else(|| infer("Gemma4 rope grid thread count overflow"))
    }

    fn padding_threads(self) -> Result<usize, BackendError> {
        self.grid_threads()?
            .checked_sub(self.total_threads)
            .ok_or_else(|| infer("Gemma4 rope grid under-covers rotary pairs"))
    }
}

fn norm_values(
    source: &crate::model::WeightSource,
    width: usize,
) -> Result<Vec<f32>, BackendError> {
    if width == 0 || source.shape() != [width].as_slice() {
        return Err(start("Gemma4 norm source shape disagrees with width"));
    }
    if source.dtype() != safetensors::Dtype::BF16 {
        return Err(start(
            "Gemma4 norm scales must be BF16 in the released checkpoint",
        ));
    }
    let mut bits = Vec::new();
    bits.try_reserve_exact(width).map_err(start)?;
    bits.resize(width, 0u16);
    source.read_bf16(0, &mut bits).map_err(start)?;
    let mut values = Vec::new();
    values.try_reserve_exact(width).map_err(start)?;
    for bits in bits {
        let value = f32::from_bits(u32::from(bits) << 16);
        if !value.is_finite() {
            return Err(start("Gemma4 norm source contains nonfinite values"));
        }
        values.push(value);
    }
    Ok(values)
}

fn upload_norm(
    stream: &Arc<Stream>,
    source: &crate::model::WeightSource,
    width: usize,
) -> Result<DeviceBuffer<f32>, BackendError> {
    DeviceBuffer::from_host(stream, &norm_values(source, width)?).map_err(start)
}

fn upload<const N: usize>(
    uploader: &mut WeightUploader,
    sources: &[&crate::model::WeightSource; N],
    name: &str,
) -> Result<DeviceBuffer<Bf16>, BackendError> {
    for source in sources {
        if source.dtype() != safetensors::Dtype::BF16 {
            return Err(start(format!(
                "Gemma4 E2B prepared path requires BF16 {name}, found {:?}",
                source.dtype()
            )));
        }
    }
    uploader.upload(sources)
}

fn source_f32_values(
    source: &crate::model::WeightSource,
    elements: usize,
) -> Result<Vec<f32>, BackendError> {
    if source.dtype() != safetensors::Dtype::BF16 || source.len() != elements {
        return Err(start(
            "Gemma4 MoE router/per-expert tensors must be BF16 with exact geometry",
        ));
    }
    let mut bits = Vec::new();
    bits.try_reserve_exact(elements).map_err(start)?;
    bits.resize(elements, 0u16);
    source.read_bf16(0, &mut bits).map_err(start)?;
    let mut values = Vec::new();
    values.try_reserve_exact(elements).map_err(start)?;
    for bits in bits {
        let value = f32::from_bits(u32::from(bits) << 16);
        if !value.is_finite() {
            return Err(start("Gemma4 MoE tensor contains nonfinite values"));
        }
        values.push(value);
    }
    Ok(values)
}

fn bf16_source_scalar(source: &crate::model::WeightSource) -> Result<f32, BackendError> {
    if source.dtype() != safetensors::Dtype::BF16 || source.shape() != [1] {
        return Err(start("Gemma4 layer_scalar must be BF16 shape [1]"));
    }
    let mut bits = [0u16; 1];
    source.read_bf16(0, &mut bits).map_err(start)?;
    let value = f32::from_bits(u32::from(bits[0]) << 16);
    if !value.is_finite() {
        return Err(start("Gemma4 layer_scalar is not finite"));
    }
    Ok(value)
}

struct Gemma4Kernels {
    norm: Function,
    residual_norm: Function,
    unit_norm: Function,
    gated_gelu: Function,
    ple_gate: Function,
    ple_combine: Function,
    scale: Function,
    rope_tables: Arc<Function>,
    rope_q: Function,
    attention_256: Function,
    rope_neox: Function,
    gather: Function,
    gather_scaled: Function,
    select_rows: Function,
    pool: Function,
    bf16_to_f32: Function,
    option_softcap: Option<Function>,
    scatter: Function,
    router_scale: Function,
    gemma2: PreparedGemma2,
}

impl Gemma4Kernels {
    fn load(
        context: &Arc<Context>,
        module_path: Option<&Path>,
        require_option_softcap: bool,
    ) -> Result<Self, BackendError> {
        let path = match module_path {
            Some(path) => path.to_path_buf(),
            None => std::env::var_os(infinity_v3_config::infinity_env!("OXIDE_KERNEL_MODULE"))
                .map(PathBuf::from)
                .ok_or_else(|| {
                    start("set INFINITY_OXIDE_KERNEL_MODULE to the cargo-oxide artifact")
                })?,
        };
        let path = if path.is_dir() {
            let (major, minor) = context.compute_capability().map_err(start)?;
            path.join(format!("sm_{major}{minor}"))
                .join("kernels.cubin")
        } else {
            path
        };
        let data =
            std::fs::read(&path).map_err(|error| start(format!("{}: {error}", path.display())))?;
        let module = context.load_module_from_image(&data).map_err(start)?;
        let function = |index: usize| {
            module
                .load_function(GEMMA4_KERNEL_EXPORTS[index])
                .map_err(start)
        };
        let option_softcap = if require_option_softcap {
            Some(
                module
                    .load_function("infinity_gemma4_option_softcap_bf16_to_f32")
                    .map_err(start)?,
            )
        } else {
            None
        };
        Ok(Self {
            norm: function(0)?,
            residual_norm: function(1)?,
            unit_norm: function(2)?,
            gated_gelu: function(3)?,
            ple_gate: function(4)?,
            ple_combine: function(5)?,
            scale: function(6)?,
            rope_tables: Arc::new(function(7)?),
            rope_q: function(8)?,
            attention_256: function(9)?,
            rope_neox: function(10)?,
            gather: module
                .load_function("infinity_embedding_gather")
                .map_err(start)?,
            gather_scaled: function(11)?,
            select_rows: function(12)?,
            pool: module.load_function("infinity_pool").map_err(start)?,
            bf16_to_f32: module
                .load_function("infinity_bf16_to_f32")
                .map_err(start)?,
            option_softcap,
            scatter: module
                .load_function("infinity_gemma4_vision_scatter_rows_bf16")
                .map_err(start)?,
            router_scale: module
                .load_function("infinity_gemma4_router_scale_bf16_to_f32")
                .map_err(start)?,
            gemma2: PreparedGemma2::from_module(&module)
                .map_err(|error| start(format!("Gemma2 mean-pool exports: {error}")))?,
        })
    }
}

struct Gemma4RoPe {
    cosine: DeviceBuffer<f32>,
    sine: DeviceBuffer<f32>,
    stride: u32,
    _inverse: DeviceBuffer<f32>,
}

struct PendingRoPe {
    stream: Arc<Stream>,
    table: Option<Gemma4RoPe>,
}

impl Drop for PendingRoPe {
    fn drop(&mut self) {
        if self.table.is_some() && self.stream.synchronize().is_err() {
            if let Some(table) = self.table.take() {
                std::mem::forget(table);
            }
        }
    }
}

fn inverse_frequencies(
    theta: f64,
    dim: usize,
    rotated_pairs: usize,
) -> Result<Vec<f32>, BackendError> {
    if !theta.is_finite() || theta <= 0.0 {
        return Err(start("Gemma4 rotary theta must be finite and positive"));
    }
    let base = theta as f32;
    if !base.is_finite() || base <= 0.0 {
        return Err(start(
            "Gemma4 rotary theta must be representable as positive f32",
        ));
    }
    if dim == 0 || !dim.is_multiple_of(2) || rotated_pairs == 0 || rotated_pairs > dim / 2 {
        return Err(start("Gemma4 rotary dimensions are invalid"));
    }
    let pairs = dim / 2;
    let mut result = Vec::new();
    result.try_reserve_exact(pairs).map_err(start)?;
    for index in 0..pairs {
        let frequency = if index < rotated_pairs {
            let exponent = (index * 2) as f32 / dim as f32;
            let value = 1.0f32 / base.powf(exponent);
            if !value.is_finite() {
                return Err(start("Gemma4 rotary inverse frequency is not finite"));
            }
            value
        } else {
            0.0
        };
        result.push(frequency);
    }
    Ok(result)
}

fn table_elements(positions: usize, pairs: usize) -> Result<usize, BackendError> {
    let elements = product(positions, pairs)?;
    let bytes = product(elements, core::mem::size_of::<f32>())?;
    let total = bytes
        .checked_mul(2)
        .and_then(|bytes| bytes.checked_add(product(pairs, 4).ok()?))
        .ok_or_else(|| start("Gemma4 rotary table byte size overflow"))?;
    if total > E2B_TABLE_BUDGET_BYTES {
        return Err(start(
            "Gemma4 rotary tables exceed the 1 GiB candidate budget",
        ));
    }
    Ok(elements)
}

impl Gemma4RoPe {
    fn new(
        stream: &Arc<Stream>,
        initializer: Arc<Function>,
        dim: usize,
        rotated_pairs: usize,
        theta: f64,
        positions: usize,
    ) -> Result<Self, BackendError> {
        if positions == 0 || positions > E2B_MAX_POSITIONS {
            return Err(start("Gemma4 rotary positions are invalid"));
        }
        let pairs = dim / 2;
        let elements = table_elements(positions, pairs)?;
        let positions = u32_size(positions)?;
        let pairs = u32_size(pairs)?;
        let inverse =
            DeviceBuffer::from_host(stream, &inverse_frequencies(theta, dim, rotated_pairs)?)
                .map_err(start)?;
        let cosine = DeviceBuffer::<f32>::zeroed(stream, elements).map_err(start)?;
        let sine = DeviceBuffer::<f32>::zeroed(stream, elements).map_err(start)?;
        let mut pending = PendingRoPe {
            stream: stream.clone(),
            table: Some(Self {
                cosine,
                sine,
                stride: pairs,
                _inverse: inverse,
            }),
        };
        let table = pending
            .table
            .as_ref()
            .ok_or_else(|| start("missing Gemma4 rotary owner"))?;
        let mut inv = table._inverse.cu_deviceptr();
        let mut cos = table.cosine.cu_deviceptr();
        let mut sin = table.sine.cu_deviceptr();
        let mut positions = positions;
        let mut pairs = pairs;
        let grid = u32_size(elements.div_ceil(256))?;
        launch_threads(
            &initializer,
            stream,
            (grid, 1, 1),
            256,
            0,
            &mut [
                arg(&mut inv),
                arg(&mut cos),
                arg(&mut sin),
                arg(&mut positions),
                arg(&mut pairs),
            ],
        )?;
        stream.synchronize().map_err(start)?;
        pending
            .table
            .take()
            .ok_or_else(|| start("missing initialized Gemma4 rotary table"))
    }
}

struct LayerBuffers {
    input_norm: DeviceBuffer<f32>,
    post_attention_norm: DeviceBuffer<f32>,
    pre_feedforward_norm: DeviceBuffer<f32>,
    post_feedforward_norm: DeviceBuffer<f32>,
    q_norm: DeviceBuffer<f32>,
    k_norm: Option<DeviceBuffer<f32>>,
    q: DeviceBuffer<Bf16>,
    k: Option<DeviceBuffer<Bf16>>,
    v: Option<DeviceBuffer<Bf16>>,
    o: DeviceBuffer<Bf16>,
    gate_up: DeviceBuffer<Bf16>,
    down: DeviceBuffer<Bf16>,
    layer_scalar: DeviceBuffer<Bf16>,
    layer_scalar_value: f32,
    ple_gate: Option<DeviceBuffer<Bf16>>,
    ple_projection: Option<DeviceBuffer<Bf16>>,
    ple_post_norm: Option<DeviceBuffer<f32>>,
    moe_pre_ff2: Option<DeviceBuffer<f32>>,
    moe_post_ff1: Option<DeviceBuffer<f32>>,
    moe_post_ff2: Option<DeviceBuffer<f32>>,
}

struct WeightBuffers {
    embed: DeviceBuffer<Bf16>,
    final_norm: DeviceBuffer<f32>,
    ple_model: Option<DeviceBuffer<Bf16>>,
    ple_embed: Option<DeviceBuffer<Bf16>>,
    ple_norm: Option<DeviceBuffer<f32>>,
    layers: Vec<LayerBuffers>,
}

struct LayerPlans {
    q: LinearPlan<Bf16>,
    k: Option<LinearPlan<Bf16>>,
    v: Option<LinearPlan<Bf16>>,
    o: LinearPlan<Bf16>,
    gate_up: LinearPlan<Bf16>,
    down: LinearPlan<Bf16>,
    ple_gate: Option<LinearPlan<Bf16>>,
    ple_projection: Option<LinearPlan<Bf16>>,
}

struct Plans {
    rows: usize,
    ple_model: Option<LinearPlan<Bf16>>,
    moe_router: Option<LinearPlan<F32>>,
    layers: Vec<LayerPlans>,
    pooled_rows: Option<usize>,
    pooled_last: Option<LayerPlans>,
}

impl LayerPlans {
    fn new(
        config: &Gemma4Config,
        layer: &Gemma4LayerPlan,
        tokens: usize,
    ) -> Result<Self, BackendError> {
        let hidden = config.hidden_size;
        let ple_width = config.hidden_size_per_layer_input;
        let k = if matches!(layer.kv, Gemma4KvPlan::Shared { .. }) {
            None
        } else {
            Some(LinearPlan::contiguous(tokens, hidden, layer.kv_width).map_err(infer)?)
        };
        let v = k
            .as_ref()
            .map(|_| LinearPlan::contiguous(tokens, hidden, layer.kv_width).map_err(infer))
            .transpose()?;
        Ok(Self {
            q: LinearPlan::contiguous(tokens, hidden, layer.q_width).map_err(infer)?,
            k,
            v,
            o: LinearPlan::contiguous(tokens, layer.q_width, hidden).map_err(infer)?,
            gate_up: LinearPlan::contiguous(
                tokens,
                hidden,
                layer
                    .dense_intermediate_size
                    .checked_mul(2)
                    .ok_or_else(|| infer("Gemma4 gate/up width overflow"))?,
            )
            .map_err(infer)?,
            down: LinearPlan::contiguous(tokens, layer.dense_intermediate_size, hidden)
                .map_err(infer)?,
            ple_gate: if ple_width == 0 {
                None
            } else {
                Some(LinearPlan::contiguous(tokens, hidden, ple_width).map_err(infer)?)
            },
            ple_projection: if ple_width == 0 {
                None
            } else {
                Some(LinearPlan::contiguous(tokens, ple_width, hidden).map_err(infer)?)
            },
        })
    }
}

impl Plans {
    fn new(
        config: &Gemma4Config,
        plan: &Gemma4Plan,
        tokens: usize,
        pooled_rows: Option<usize>,
    ) -> Result<Self, BackendError> {
        let hidden = config.hidden_size;
        let ple_width = config.hidden_size_per_layer_input;
        let mut layers = Vec::new();
        layers.try_reserve_exact(plan.layers.len()).map_err(start)?;
        for layer in &plan.layers {
            layers.push(LayerPlans::new(config, layer, tokens)?);
        }
        let last = plan
            .layers
            .last()
            .ok_or_else(|| infer("Gemma4 plan has no layers"))?;
        let pooled_last = pooled_rows
            .map(|rows| LayerPlans::new(config, last, rows))
            .transpose()?;
        Ok(Self {
            rows: tokens,
            ple_model: match plan.ple {
                Some(ple) if ple_width > 0 => Some(
                    LinearPlan::contiguous(
                        tokens,
                        hidden,
                        ple.layers
                            .checked_mul(ple_width)
                            .ok_or_else(|| infer("Gemma4 PLE width overflow"))?,
                    )
                    .map_err(infer)?,
                ),
                None if ple_width == 0 => None,
                _ => return Err(infer("Gemma4 PLE plan disagrees with config")),
            },
            moe_router: plan
                .layers
                .iter()
                .find_map(|layer| layer.moe)
                .map(|moe| {
                    LinearPlan::<F32>::contiguous(tokens, hidden, moe.num_experts).map_err(infer)
                })
                .transpose()?,
            layers,
            pooled_rows,
            pooled_last,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TextPhase {
    Staged,
    Uploaded,
    Enqueued,
    OutputCopied,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TextUse {
    lease_id: usize,
    phase: TextPhase,
}

/// Pure capacity predicate shared by production stage_text and CPU tests.
#[allow(clippy::too_many_arguments)]
pub(crate) fn text_shape_fits(
    tokens: usize,
    boundaries: usize,
    pooled_rows: usize,
    raw_rows: usize,
    work_tokens: usize,
    work_boundaries: usize,
    work_pooled: usize,
    work_raw: usize,
) -> bool {
    tokens <= work_tokens
        && boundaries <= work_boundaries
        && pooled_rows <= work_pooled
        && raw_rows <= work_raw
}

pub(crate) struct Gemma4E2bWork {
    stream: Arc<Stream>,
    tokens: usize,
    sequences: usize,
    selections: usize,
    raw_rows: usize,
    hidden: usize,
    bytes: usize,
    input: ManuallyDrop<DeviceBuffer<u32>>,
    state: ManuallyDrop<DeviceBuffer<Bf16>>,
    next: ManuallyDrop<DeviceBuffer<Bf16>>,
    normalized: ManuallyDrop<DeviceBuffer<Bf16>>,
    branch: ManuallyDrop<DeviceBuffer<Bf16>>,
    q: ManuallyDrop<DeviceBuffer<Bf16>>,
    k: ManuallyDrop<DeviceBuffer<Bf16>>,
    v: ManuallyDrop<DeviceBuffer<Bf16>>,
    attention: ManuallyDrop<DeviceBuffer<Bf16>>,
    lse: ManuallyDrop<DeviceBuffer<f32>>,
    gate_up: ManuallyDrop<DeviceBuffer<Bf16>>,
    activation: ManuallyDrop<DeviceBuffer<Bf16>>,
    ple: ManuallyDrop<DeviceBuffer<Bf16>>,
    ple_token: ManuallyDrop<DeviceBuffer<Bf16>>,
    output: ManuallyDrop<DeviceBuffer<Bf16>>,
    shared_k: ManuallyDrop<[DeviceBuffer<Bf16>; 2]>,
    shared_v: ManuallyDrop<[DeviceBuffer<Bf16>; 2]>,
    pooled: ManuallyDrop<DeviceBuffer<f32>>,
    raw_output: ManuallyDrop<Option<DeviceBuffer<f32>>>,
    raw_selected: ManuallyDrop<Option<DeviceBuffer<u32>>>,
    text_use: Option<TextUse>,
    moe_scratch: ManuallyDrop<Option<MoeScratch>>,
    router_input: ManuallyDrop<Option<DeviceBuffer<f32>>>,
    last_completion: Option<CompletionTicket>,
    payload_released: bool,
}

impl Gemma4E2bWork {
    pub(crate) fn key(&self) -> WorkKey {
        (self.tokens, self.sequences, self.selections)
    }
    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }
    pub(crate) fn fits(&self, key: WorkKey) -> bool {
        crate::capacity_fits(self.key(), key, self.bytes)
    }

    pub(crate) fn uses_stream(&self, stream: &Stream) -> bool {
        std::ptr::eq(self.stream.as_ref(), stream)
    }

    pub(crate) fn raw_capacity(&self) -> usize {
        self.raw_rows
    }

    pub(crate) fn raw_output(&self) -> Option<&DeviceBuffer<f32>> {
        self.raw_output.as_ref()
    }
    pub(crate) fn allocation_bytes(
        key: WorkKey,
        weights: &Gemma4Weights,
        raw_rows: usize,
    ) -> Result<usize, BackendError> {
        Self::allocation_bytes_moe(key, weights, raw_rows, None)
    }

    pub(crate) fn allocation_bytes_moe(
        key: WorkKey,
        weights: &Gemma4Weights,
        raw_rows: usize,
        moe: Option<&MoeProvider>,
    ) -> Result<usize, BackendError> {
        let (tokens, _, selections) = key;
        let hidden = weights.config.hidden_size;
        let ple_width = weights.config.hidden_size_per_layer_input;
        let ple_total = weights
            .plan
            .ple
            .map(|ple| {
                ple.layers
                    .checked_mul(ple_width)
                    .ok_or_else(|| infer("Gemma4 PLE width overflow"))
            })
            .transpose()?
            .unwrap_or(0);
        let max_q = weights
            .plan
            .layers
            .iter()
            .map(|layer| layer.q_width)
            .max()
            .unwrap_or(0);
        let max_kv = weights
            .plan
            .layers
            .iter()
            .map(|layer| layer.kv_width)
            .max()
            .unwrap_or(0);
        let max_intermediate = weights
            .plan
            .layers
            .iter()
            .map(|layer| layer.dense_intermediate_size)
            .max()
            .unwrap_or(0);
        let mut total = crate::staging::input_capacity(key)?
            .checked_mul(4)
            .ok_or_else(|| infer("Gemma4 input arena overflow"))?;
        let lse_bytes = product(tokens, weights.config.num_attention_heads)?
            .checked_mul(4)
            .ok_or_else(|| infer("Gemma4 LSE bytes overflow"))?;
        total = total
            .checked_add(lse_bytes)
            .ok_or_else(|| infer("Gemma4 workspace arena overflow"))?;
        for (name, elements) in [
            ("state", product(tokens, hidden)?),
            ("next", product(tokens, hidden)?),
            ("normalized", product(tokens, hidden)?),
            ("branch", product(tokens, hidden)?),
            ("q", product(tokens, max_q)?),
            ("k", product(tokens, max_kv)?),
            ("v", product(tokens, max_kv)?),
            ("attention", product(tokens, max_q)?),
            (
                "gate_up",
                product(
                    tokens,
                    max_intermediate
                        .checked_mul(2)
                        .ok_or_else(|| infer("Gemma4 gate/up overflow"))?,
                )?,
            ),
            ("activation", product(tokens, max_intermediate)?),
            ("ple", product(tokens, ple_total)?.max(1)),
            ("ple_token", product(tokens, ple_total)?.max(1)),
            ("output", product(tokens, hidden)?),
            ("shared_k", product(product(tokens, max_kv)?, 2)?),
            ("shared_v", product(product(tokens, max_kv)?, 2)?),
        ] {
            total = total
                .checked_add(
                    elements
                        .checked_mul(2)
                        .ok_or_else(|| infer(format!("Gemma4 {name} bytes overflow")))?,
                )
                .ok_or_else(|| infer("Gemma4 workspace arena overflow"))?;
        }
        let pooled = product(selections, hidden)?
            .checked_mul(4)
            .ok_or_else(|| infer("Gemma4 pooled bytes overflow"))?;
        let raw = product(raw_rows, hidden)?
            .checked_mul(4)
            .ok_or_else(|| infer("Gemma4 raw output bytes overflow"))?;
        let raw_indices = product(raw_rows, 4)?;
        let moe_bytes = match moe {
            Some(provider) => {
                let scratch = MoeScratch::allocation_bytes(provider, tokens)?;
                let router = product(tokens, hidden)?
                    .checked_mul(4)
                    .ok_or_else(|| infer("Gemma4 router input bytes overflow"))?;
                scratch
                    .checked_add(router)
                    .ok_or_else(|| infer("Gemma4 MoE workspace overflow"))?
            }
            None => 0,
        };
        total
            .checked_add(pooled)
            .and_then(|value| value.checked_add(raw))
            .and_then(|value| value.checked_add(raw_indices))
            .and_then(|value| value.checked_add(moe_bytes))
            .ok_or_else(|| infer("Gemma4 workspace arena overflow"))
    }
    pub(crate) fn new(
        stream: Arc<Stream>,
        weights: &Gemma4Weights,
        key: WorkKey,
    ) -> Result<Self, BackendError> {
        Self::new_with_raw(stream, weights, key, 0)
    }
    pub(crate) fn new_with_raw(
        stream: Arc<Stream>,
        weights: &Gemma4Weights,
        key: WorkKey,
        raw_rows: usize,
    ) -> Result<Self, BackendError> {
        Self::new_with_raw_moe(stream, weights, key, raw_rows, None)
    }

    pub(crate) fn new_with_raw_moe(
        stream: Arc<Stream>,
        weights: &Gemma4Weights,
        key: WorkKey,
        raw_rows: usize,
        moe: Option<&MoeProvider>,
    ) -> Result<Self, BackendError> {
        let bytes = Self::allocation_bytes_moe(key, weights, raw_rows, moe)?;
        let (tokens, sequences, selections) = key;
        let hidden = weights.config.hidden_size;
        let ple_width = weights.config.hidden_size_per_layer_input;
        let ple_total = weights
            .plan
            .ple
            .map(|ple| {
                ple.layers
                    .checked_mul(ple_width)
                    .ok_or_else(|| infer("Gemma4 PLE width overflow"))
            })
            .transpose()?
            .unwrap_or(0);
        let max_q = weights
            .plan
            .layers
            .iter()
            .map(|layer| layer.q_width)
            .max()
            .unwrap_or(0);
        let max_kv = weights
            .plan
            .layers
            .iter()
            .map(|layer| layer.kv_width)
            .max()
            .unwrap_or(0);
        let max_intermediate = weights
            .plan
            .layers
            .iter()
            .map(|layer| layer.dense_intermediate_size)
            .max()
            .unwrap_or(0);
        let input_len = crate::staging::input_capacity(key)?;
        let alloc = |len: usize| -> Result<DeviceBuffer<Bf16>, BackendError> {
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
            Ok(buffer)
        };
        let input = unsafe { DeviceBuffer::<u32>::uninitialized_async(&stream, input_len) }
            .map_err(infer)?;
        let state = alloc(product(tokens, hidden)?)?;
        let next = alloc(product(tokens, hidden)?)?;
        let normalized = alloc(product(tokens, hidden)?)?;
        let branch = alloc(product(tokens, hidden)?)?;
        let q = alloc(product(tokens, max_q)?)?;
        let k = alloc(product(tokens, max_kv)?)?;
        let v = alloc(product(tokens, max_kv)?)?;
        let attention = alloc(product(tokens, max_q)?)?;
        let lse = unsafe {
            DeviceBuffer::<f32>::uninitialized_async(
                &stream,
                product(tokens, weights.config.num_attention_heads)?,
            )
        }
        .map_err(infer)?;
        let gate_up = alloc(product(
            tokens,
            max_intermediate
                .checked_mul(2)
                .ok_or_else(|| infer("Gemma4 gate/up overflow"))?,
        )?)?;
        let activation = alloc(product(tokens, max_intermediate)?)?;
        let ple_buf = alloc(product(tokens, ple_total)?.max(1))?;
        let ple_token = alloc(product(tokens, ple_total)?.max(1))?;
        let output = alloc(product(tokens, hidden)?)?;
        let shared_k = [
            alloc(product(tokens, max_kv)?)?,
            alloc(product(tokens, max_kv)?)?,
        ];
        let shared_v = [
            alloc(product(tokens, max_kv)?)?,
            alloc(product(tokens, max_kv)?)?,
        ];
        let pooled = unsafe {
            DeviceBuffer::<f32>::uninitialized_async(&stream, product(selections, hidden)?)
        }
        .map_err(infer)?;
        let raw_output = if raw_rows > 0 {
            Some(
                unsafe {
                    DeviceBuffer::<f32>::uninitialized_async(&stream, product(raw_rows, hidden)?)
                }
                .map_err(infer)?,
            )
        } else {
            None
        };
        let raw_selected = if raw_rows > 0 {
            Some(
                unsafe { DeviceBuffer::<u32>::uninitialized_async(&stream, raw_rows) }
                    .map_err(infer)?,
            )
        } else {
            None
        };
        let moe_scratch = moe
            .map(|provider| MoeScratch::new(provider, stream.clone(), tokens))
            .transpose()?;
        let router_input = moe
            .map(|_| DeviceBuffer::<f32>::zeroed(&stream, product(tokens, hidden)?).map_err(infer))
            .transpose()?;
        crate::LIVE_ARENA_BYTES.fetch_add(bytes, Ordering::Relaxed);
        crate::record_live_arena_bytes();
        Ok(Self {
            stream,
            tokens,
            sequences,
            selections,
            raw_rows,
            hidden,
            bytes,
            input: ManuallyDrop::new(input),
            state: ManuallyDrop::new(state),
            next: ManuallyDrop::new(next),
            normalized: ManuallyDrop::new(normalized),
            branch: ManuallyDrop::new(branch),
            q: ManuallyDrop::new(q),
            k: ManuallyDrop::new(k),
            v: ManuallyDrop::new(v),
            attention: ManuallyDrop::new(attention),
            lse: ManuallyDrop::new(lse),
            gate_up: ManuallyDrop::new(gate_up),
            activation: ManuallyDrop::new(activation),
            ple: ManuallyDrop::new(ple_buf),
            ple_token: ManuallyDrop::new(ple_token),
            output: ManuallyDrop::new(output),
            shared_k: ManuallyDrop::new(shared_k),
            shared_v: ManuallyDrop::new(shared_v),
            pooled: ManuallyDrop::new(pooled),
            raw_output: ManuallyDrop::new(raw_output),
            raw_selected: ManuallyDrop::new(raw_selected),
            text_use: None,
            moe_scratch: ManuallyDrop::new(moe_scratch),
            router_input: ManuallyDrop::new(router_input),
            last_completion: None,
            payload_released: false,
        })
    }
    pub(crate) fn moe_scratch_mut(&mut self) -> Result<&mut MoeScratch, BackendError> {
        self.moe_scratch
            .as_mut()
            .ok_or_else(|| infer("Gemma4 Work has no retained MoE scratch"))
    }

    pub(crate) fn router_input_mut(&mut self) -> Result<&mut DeviceBuffer<f32>, BackendError> {
        self.router_input
            .as_mut()
            .ok_or_else(|| infer("Gemma4 Work has no retained router input"))
    }

    pub(crate) fn mark_last_use(&mut self, ticket: CompletionTicket) {
        self.last_completion = Some(ticket);
    }

    pub(crate) fn completion_ticket(&self) -> Option<CompletionTicket> {
        self.last_completion.clone()
    }

    /// # Safety
    /// The engine context must be current and every request use must have been
    /// marked before its first GPU submission.
    pub(crate) unsafe fn release_if_idle(&mut self) -> Result<bool, BackendError> {
        match self.last_completion.as_ref() {
            None => {
                unsafe { self.drop_payload_once() };
                Ok(true)
            }
            Some(ticket) => match ticket.completed() {
                Some(proof) => {
                    unsafe { self.release_payload_after_completion(&proof)? };
                    Ok(true)
                }
                None => Ok(false),
            },
        }
    }

    /// # Safety
    /// The proof must come from the actual same-stream fence for this Work.
    pub(crate) unsafe fn release_payload_after_completion(
        &mut self,
        proof: &CompletedTicket,
    ) -> Result<(), BackendError> {
        let ticket = self
            .last_completion
            .as_ref()
            .ok_or_else(|| infer("Gemma4 Work has no completion ticket"))?;
        let stream_identity = Arc::as_ptr(&self.stream) as usize;
        if !proof.matches(ticket)
            || proof.stream_identity() != stream_identity
            || ticket.stream_identity() != stream_identity
        {
            return Err(infer(
                "Completion proof does not match this Gemma4 Work request",
            ));
        }
        unsafe { self.drop_payload_once() };
        Ok(())
    }

    /// # Safety
    /// The caller must have observed a successful whole-stream drain.
    pub(crate) unsafe fn release_payload_after_drain(&mut self) {
        unsafe { self.drop_payload_once() };
    }

    unsafe fn drop_payload_once(&mut self) {
        if self.payload_released {
            return;
        }
        self.payload_released = true;
        unsafe {
            ManuallyDrop::drop(&mut self.input);
            ManuallyDrop::drop(&mut self.state);
            ManuallyDrop::drop(&mut self.next);
            ManuallyDrop::drop(&mut self.normalized);
            ManuallyDrop::drop(&mut self.branch);
            ManuallyDrop::drop(&mut self.q);
            ManuallyDrop::drop(&mut self.k);
            ManuallyDrop::drop(&mut self.v);
            ManuallyDrop::drop(&mut self.attention);
            ManuallyDrop::drop(&mut self.lse);
            ManuallyDrop::drop(&mut self.gate_up);
            ManuallyDrop::drop(&mut self.activation);
            ManuallyDrop::drop(&mut self.ple);
            ManuallyDrop::drop(&mut self.ple_token);
            ManuallyDrop::drop(&mut self.output);
            ManuallyDrop::drop(&mut self.shared_k);
            ManuallyDrop::drop(&mut self.shared_v);
            ManuallyDrop::drop(&mut self.pooled);
            ManuallyDrop::drop(&mut self.raw_output);
            ManuallyDrop::drop(&mut self.raw_selected);
            ManuallyDrop::drop(&mut self.moe_scratch);
            ManuallyDrop::drop(&mut self.router_input);
        }
        crate::LIVE_ARENA_BYTES.fetch_sub(self.bytes, Ordering::Relaxed);
        crate::record_live_arena_bytes();
    }

    fn check(
        &self,
        tokens: usize,
        boundaries: usize,
        selections: usize,
    ) -> Result<(), BackendError> {
        if tokens > self.tokens || boundaries > self.sequences || selections > self.selections {
            return Err(infer("Gemma4 E2B request exceeds retained arena"));
        }
        Ok(())
    }

    pub(crate) fn text_phase(&self) -> Option<TextPhase> {
        self.text_use.map(|state| state.phase)
    }

    /// Clear a partially staged text use only after the producing stream was
    /// drained. A failed drain leaves the state intact so Drop can quarantine.
    pub(crate) fn cancel_text_after_failure(&mut self) {
        if self.stream.synchronize().is_ok() {
            self.text_use = None;
        }
    }

    /// Start a text use only when no earlier text use needs its output copied.
    fn begin_text_use(&mut self, lease: &HostIoLease) -> Result<(), BackendError> {
        if let Some(state) = self.text_use {
            if state.phase != TextPhase::OutputCopied {
                return Err(infer(
                    "Gemma4 E2B Work has a previous text use without an output D2H",
                ));
            }
        }
        self.text_use = Some(TextUse {
            lease_id: lease.id(),
            phase: TextPhase::Staged,
        });
        Ok(())
    }

    fn check_text_owner(&self, lease: &HostIoLease, phase: TextPhase) -> Result<(), BackendError> {
        let Some(state) = self.text_use else {
            return Err(infer("Gemma4 E2B Work has no active text use"));
        };
        if state.lease_id != lease.id() || state.phase != phase {
            return Err(infer(
                "Gemma4 E2B text lease/generation does not match the Work state",
            ));
        }
        Ok(())
    }

    /// # Safety
    /// The stream must have completed every enqueue using this Work.
    pub(crate) unsafe fn reset_text_after_drain(&mut self) {
        self.text_use = None;
    }
    pub(crate) fn stage(
        &self,
        batch: &Batch,
        geometry: crate::output::OutputGeometry,
        lease: &mut HostIoLease,
    ) -> Result<(usize, usize), BackendError> {
        if geometry.raw_width != self.hidden || geometry.pooled_width != self.hidden {
            return Err(infer("Gemma4 E2B output geometry must match hidden width"));
        }
        if geometry.backbone_elements(batch)? > self.pooled.len() {
            return Err(infer("Gemma4 E2B output exceeds retained arena"));
        }
        crate::stage_batch(batch, geometry, lease)
    }
    /// Stage a text request into the caller's HostIoLease using the existing
    /// packed layout: 3*T token arrays + B+1 boundaries + P pooled indices +
    /// R raw token rows. The pooled prefix is uploadable into this Work; the raw
    /// tail stays lease-owned so A/B requests cannot overwrite each other's H2D
    /// source. Returns (prefix_elements, raw_tail_offset_words, output_elements).
    pub(crate) fn stage_text(
        &mut self,
        batch: &Batch,
        geometry: crate::output::OutputGeometry,
        lease: &mut HostIoLease,
    ) -> Result<(usize, usize, usize), BackendError> {
        if geometry.raw_width != self.hidden || geometry.pooled_width != self.hidden {
            return Err(infer("Gemma4 E2B output geometry must match hidden width"));
        }
        let tokens = batch.input_ids.len();
        let boundaries = batch.cumulative_seq_lengths.len();
        let pooled_rows = batch.pooled_indices.len();
        let raw_rows = crate::output::raw_rows(batch)?;
        if !text_shape_fits(
            tokens,
            boundaries,
            pooled_rows,
            raw_rows,
            self.tokens,
            self.sequences,
            self.selections,
            self.raw_rows,
        ) {
            return Err(infer(
                "Gemma4 E2B text request exceeds retained Work capacity",
            ));
        }
        if self
            .text_use
            .is_some_and(|state| state.phase != TextPhase::OutputCopied)
        {
            return Err(infer(
                "Gemma4 E2B Work has a previous text use without an output D2H",
            ));
        }
        lease.ensure_stageable()?;
        let (elements, output) = crate::stage_batch(batch, geometry, lease)?;
        let prefix = crate::input_elements(tokens, boundaries, pooled_rows)?;
        let tail = prefix;
        let required = tail
            .checked_add(raw_rows)
            .ok_or_else(|| infer("Gemma4 E2B raw tail offset overflow"))?;
        if elements < required {
            return Err(infer(
                "Gemma4 E2B staged text input is shorter than its raw tail",
            ));
        }
        // Retention begins before the caller's prefix H2D.
        lease.begin_checked()?;
        self.begin_text_use(lease)?;
        Ok((prefix, tail, output))
    }

    /// Upload only the pooled prefix into Work input after checking both host
    /// and destination device extents. Raw rows remain in the lease tail.
    pub(crate) fn upload_text_prefix(
        &mut self,
        lease: &HostIoLease,
        elements: usize,
    ) -> Result<(), BackendError> {
        self.check_text_owner(lease, TextPhase::Staged)?;
        if elements > self.input.len() {
            return Err(infer(
                "Gemma4 E2B text prefix exceeds the Work input destination",
            ));
        }
        crate::upload_input(
            &self.stream,
            &self.input,
            lease.slot()?.input.as_slice(),
            elements,
        )?;
        if let Some(state) = self.text_use.as_mut() {
            state.phase = TextPhase::Uploaded;
        }
        Ok(())
    }

    pub(crate) fn upload(&self, host: &[u32], elements: usize) -> Result<(), BackendError> {
        if elements > host.len() || elements > self.input.len() {
            return Err(infer(
                "Gemma4 E2B upload exceeds the host or device input extent",
            ));
        }
        crate::upload_input(&self.stream, &self.input, host, elements)
    }

    /// Upload the packed text prefix and advance the owned text phase. Raw
    /// selection words remain in the caller's lease tail.
    pub(crate) fn upload_text_and_mark(
        &mut self,
        host: &[u32],
        elements: usize,
    ) -> Result<(), BackendError> {
        self.upload(host, elements)?;
        if let Some(state) = self.text_use.as_mut() {
            state.phase = TextPhase::Uploaded;
        }
        Ok(())
    }
    pub(crate) fn finish(
        &mut self,
        lt: &crate::lt::LtProvider,
        plain_blas: Option<&Blas>,
        pooled_rows: usize,
        lease: &mut HostIoLease,
    ) -> Result<(), BackendError> {
        crate::finish_pooled(
            None,
            lt,
            plain_blas,
            &self.stream,
            &self.pooled,
            pooled_rows,
            0,
            self.hidden,
            lease,
        )
    }
}

impl Drop for Gemma4E2bWork {
    fn drop(&mut self) {
        if self.payload_released {
            return;
        }
        if self.stream.synchronize().is_ok() {
            unsafe { self.drop_payload_once() };
        } else {
            metrics::counter!("oxide_arena_retained_bytes_total").increment(self.bytes as u64);
        }
    }
}

/// Full-width d512 provider boundary. Implementations must have completed cold
/// preparation on the executor's bound context. Warm enqueue performs no
/// context query, allocation, synchronization or graph capture.
pub(crate) trait FullAttentionProvider: Send + Sync {
    fn supports(&self, plan: &Gemma4LayerPlan, causal: bool) -> bool;
    /// # Safety
    /// Pointers must be live BF16 storage in the bound context; offsets and
    /// output/lse regions belong to the caller's outer Work arena.
    #[allow(clippy::too_many_arguments)]
    unsafe fn forward_batch_bound(
        &self,
        stream: &Stream,
        plan: &Gemma4LayerPlan,
        q: u64,
        k: u64,
        v: u64,
        out: u64,
        out_bytes: u64,
        lse: u64,
        lse_bytes: u64,
        boundaries: u64,
        offsets_bytes: usize,
        tokens: usize,
        sequences: usize,
        max_length: usize,
        causal: bool,
    ) -> Result<(), BackendError>;
}

/// Qualified local d256 provider boundary. The serving executor accepts only
/// this route while FA4 has the additive E2B local profile; the isolated
/// FP32-score/softmax leaf is never a silent fallback.
pub(crate) trait LocalAttentionProvider: Send + Sync {
    fn supports(&self, plan: &Gemma4LayerPlan, causal: bool) -> bool;
    /// # Safety
    /// Pointers must be live BF16 storage in the bound context; offsets and
    /// output/lse regions belong to the caller's Work arena.
    #[allow(clippy::too_many_arguments)]
    unsafe fn forward_batch_bound(
        &self,
        stream: &Stream,
        plan: &Gemma4LayerPlan,
        q: u64,
        k: u64,
        v: u64,
        out: u64,
        out_bytes: u64,
        lse: u64,
        lse_bytes: u64,
        boundaries: u64,
        offsets_bytes: usize,
        tokens: usize,
        sequences: usize,
        max_length: usize,
        causal: bool,
    ) -> Result<(), BackendError>;
}

/// Adapter for the Root-qualified ABI2 Head512 artifact.
pub(crate) struct Head512Aot {
    provider: Head512Provider,
}

impl Head512Aot {
    pub(crate) fn load(
        context: &Arc<Context>,
        root: &Path,
        architecture: &str,
        max_optin_shared_memory_bytes: u64,
    ) -> Result<Self, BackendError> {
        context.bind_to_thread().map_err(start)?;
        let mut provider =
            Head512Provider::load_for_device(root, architecture, max_optin_shared_memory_bytes)
                .map_err(|error| start(format!("Head512 AOT load: {error}")))?;
        // SAFETY: the caller owns the bound context and keeps it alive with the
        // provider; this is the single cold preparation call.
        unsafe {
            provider
                .prepare_current_context()
                .map_err(|error| start(format!("Head512 AOT prepare: {error}")))?;
        }
        Ok(Self { provider })
    }

    fn capabilities(&self) -> &Head512Capabilities {
        self.provider.capabilities()
    }
}

// SAFETY: the Engine serializes every use behind its Mutex; these handles
// are owned by the provider and live for the whole model.
unsafe impl Send for Head512Aot {}
unsafe impl Sync for Head512Aot {}

impl FullAttentionProvider for Head512Aot {
    fn supports(&self, plan: &Gemma4LayerPlan, causal: bool) -> bool {
        plan.kind == Gemma4LayerKind::Full
            && plan.head_dim == E2B_GLOBAL_HEAD_DIM
            && plan.q_width
                == plan
                    .q_width
                    .checked_div(plan.head_dim)
                    .and_then(|heads| heads.checked_mul(plan.head_dim))
                    .unwrap_or(0)
            && plan.num_kv_heads > 0
            && plan.kv_width == plan.num_kv_heads * plan.head_dim
            && plan.window_left == -1
            && plan.window_right == -1
            && self.capabilities().supports(Head512Spec {
                dtype: Head512DType::Bf16,
                causal,
                window: Head512Window::Global,
            })
    }

    unsafe fn forward_batch_bound(
        &self,
        stream: &Stream,
        plan: &Gemma4LayerPlan,
        q: u64,
        k: u64,
        v: u64,
        out: u64,
        out_bytes: u64,
        lse: u64,
        lse_bytes: u64,
        boundaries: u64,
        offsets_bytes: usize,
        tokens: usize,
        sequences: usize,
        max_length: usize,
        causal: bool,
    ) -> Result<(), BackendError> {
        if !self.supports(plan, causal) {
            return Err(infer(
                "Head512 AOT provider does not support this E2B layer",
            ));
        }
        let q_width = i64::try_from(plan.q_width).map_err(infer)?;
        let kv_width = i64::try_from(plan.kv_width).map_err(infer)?;
        let head_dim = i64::try_from(plan.head_dim).map_err(infer)?;
        let heads = plan.q_width / plan.head_dim;
        let kv_heads = plan.num_kv_heads;
        let batch = Head512Batch {
            q,
            k,
            v,
            q_offsets: boundaries,
            q_offsets_bytes: offsets_bytes,
            k_offsets: boundaries,
            k_offsets_bytes: offsets_bytes,
            total_q: tokens,
            total_k: tokens,
            batch: sequences,
            max_q: max_length,
            max_k: max_length,
            heads,
            kv_heads,
            scale: 1.0,
            dtype: Head512DType::Bf16,
            causal,
            window: Head512Window::Global,
            strides: [
                q_width, head_dim, 1, kv_width, head_dim, 1, kv_width, head_dim, 1, q_width,
                head_dim, 1,
            ],
        };
        let mut work = Head512Work::new(
            DeviceRegion::new(out, out_bytes).map_err(|error| infer(error.to_string()))?,
            DeviceRegion::new(lse, lse_bytes).map_err(|error| infer(error.to_string()))?,
        );
        self.provider
            .enqueue(&batch, &mut work, stream.cu_stream() as usize as u64)
            .map_err(|error| infer(format!("Head512 AOT enqueue: {error}")))?;
        Ok(())
    }
}

enum LocalAttentionRoute {
    Qualified(Arc<dyn LocalAttentionProvider>),
    DiagnosticLeaf,
}

pub(crate) struct Fa4Local {
    provider: crate::fa4::Fa4,
    causal: bool,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    window: (i32, i32),
}

impl Fa4Local {
    pub(crate) fn load(
        context: Arc<Context>,
        root: &Path,
        math: &Gemma4E2bMath,
    ) -> Result<Self, BackendError> {
        let spec = crate::fa4::AttentionSpec {
            heads: math.local_heads as usize,
            kv_heads: math.local_kv_heads,
            head_dim: math.local_head_dim,
            causal: math.causal,
            window: crate::fa4::Window::Local {
                left: math.local_left,
                right: math.local_right,
            },
            dtype: crate::fa4::DType::Bf16,
            scale: crate::fa4::Scale::Explicit(1.0),
        };
        if !crate::fa4::Fa4::supports(&spec) {
            return Err(start(format!(
                "FA4 bundle lacks the Gemma4 local Q{}/KV{} d{} profile",
                spec.heads, spec.kv_heads, spec.head_dim
            )));
        }
        Ok(Self {
            provider: crate::fa4::Fa4::new(context, root, spec)?,
            causal: math.causal,
            heads: spec.heads,
            kv_heads: spec.kv_heads,
            head_dim: spec.head_dim,
            window: (math.local_left, math.local_right),
        })
    }
}

impl LocalAttentionProvider for Fa4Local {
    fn supports(&self, plan: &Gemma4LayerPlan, causal: bool) -> bool {
        plan.kind == Gemma4LayerKind::Sliding
            && plan.head_dim == self.head_dim
            && plan.num_kv_heads == self.kv_heads
            && plan.q_width == self.heads * self.head_dim
            && plan.kv_width == self.kv_heads * self.head_dim
            && plan.window_left == self.window.0
            && plan.window_right == self.window.1
            && causal == self.causal
    }

    unsafe fn forward_batch_bound(
        &self,
        stream: &Stream,
        plan: &Gemma4LayerPlan,
        q: u64,
        k: u64,
        v: u64,
        out: u64,
        _out_bytes: u64,
        _lse: u64,
        _lse_bytes: u64,
        boundaries: u64,
        _offsets_bytes: usize,
        tokens: usize,
        sequences: usize,
        max_length: usize,
        causal: bool,
    ) -> Result<(), BackendError> {
        if !self.supports(plan, causal) {
            return Err(infer(
                Gemma4E2bUnsupported::LocalAttentionPending.to_string(),
            ));
        }
        let head_dim = i64::try_from(plan.head_dim).map_err(infer)?;
        let q_width = i64::try_from(plan.q_width).map_err(infer)?;
        let kv_width = i64::try_from(plan.kv_width).map_err(infer)?;
        let batch = crate::fa4::AttentionBatch {
            q,
            k,
            v,
            out,
            q_offsets: boundaries,
            k_offsets: boundaries,
            q_tokens: tokens,
            k_tokens: tokens,
            batch: sequences,
            max_q: max_length,
            strides: [
                q_width, head_dim, 1, kv_width, head_dim, 1, kv_width, head_dim, 1, q_width,
                head_dim, 1,
            ],
        };
        // SAFETY: the executor passes live Work pointers and one packed boundary
        // table; the provider was cold-prepared for this bound context and FA4
        // validates geometry before the warm launch.
        unsafe { self.provider.forward_batch_bound(stream, &batch) }
    }
}

/// Image rows projected by the vision tower, injected into the text embedding
/// arena after PLE construction and before the first transformer layer.
pub(crate) struct VisionInjection<'a> {
    pub(crate) work: &'a Gemma4VisionWork,
    pub(crate) rows: &'a [u32],
}

struct Gemma4MoeRuntime {
    provider: MoeProvider,
    layers: Vec<Option<Gemma4MoeLayer>>,
    router_scale: Vec<Option<DeviceBuffer<f32>>>,
    router_projection: Vec<Option<DeviceBuffer<f32>>>,
    per_expert_scale: Vec<Option<DeviceBuffer<f32>>>,
}

impl Gemma4MoeRuntime {
    fn new(
        context: &Arc<Context>,
        stream: &Arc<Stream>,
        weights: &Gemma4Weights,
        math: &Gemma4E2bMath,
        uploader: &mut WeightUploader,
    ) -> Result<Self, BackendError> {
        let (experts, top_k, intermediate) = math
            .moe
            .ok_or_else(|| start("Gemma4 MoE runtime requested without a MoE plan"))?;
        let path = crate::moe_artifact::resolve(context)?;
        // SAFETY: source-controlled deployment sidecar and SHA were validated cold.
        let provider = unsafe {
            MoeProvider::load_gemma4(context, &path, math.hidden, intermediate, experts, top_k)
        }?;
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(weights.layers.len())
            .map_err(start)?;
        let mut router_scale = Vec::new();
        router_scale
            .try_reserve_exact(weights.layers.len())
            .map_err(start)?;
        let mut router_projection = Vec::new();
        router_projection
            .try_reserve_exact(weights.layers.len())
            .map_err(start)?;
        let mut per_expert_scale = Vec::new();
        per_expert_scale
            .try_reserve_exact(weights.layers.len())
            .map_err(start)?;
        for layer in &weights.layers {
            let Some(moe) = layer.moe.as_ref() else {
                layers.push(None);
                router_scale.push(None);
                router_projection.push(None);
                per_expert_scale.push(None);
                continue;
            };
            layers.push(Some(Gemma4MoeLayer::upload(
                uploader,
                &moe.gate_up,
                &moe.down,
                math.hidden,
                intermediate,
                experts,
            )?));
            router_scale.push(Some(
                DeviceBuffer::from_host(
                    stream,
                    &source_f32_values(&moe.router_scale, math.hidden)?,
                )
                .map_err(start)?,
            ));
            router_projection.push(Some(
                DeviceBuffer::from_host(
                    stream,
                    &source_f32_values(
                        &moe.router_projection,
                        experts
                            .checked_mul(math.hidden)
                            .ok_or_else(|| start("Gemma4 router width overflow"))?,
                    )?,
                )
                .map_err(start)?,
            ));
            per_expert_scale.push(Some(
                DeviceBuffer::from_host(
                    stream,
                    &source_f32_values(&moe.per_expert_scale, experts)?,
                )
                .map_err(start)?,
            ));
        }
        Ok(Self {
            provider,
            layers,
            router_scale,
            router_projection,
            per_expert_scale,
        })
    }
}

pub(crate) struct Gemma4E2bExecutor {
    config: Gemma4Config,
    plan: Gemma4Plan,
    math: Gemma4E2bMath,
    kernels: Gemma4Kernels,
    weights: WeightBuffers,
    local_rope: Gemma4RoPe,
    global_rope: Gemma4RoPe,
    plans: Option<Plans>,
    attention: Option<Arc<dyn FullAttentionProvider>>,
    local_attention: LocalAttentionRoute,
    option_head: Option<Gemma4OptionHead>,
    moe: Option<Gemma4MoeRuntime>,
}

impl Gemma4E2bExecutor {
    pub(crate) fn qualify(
        weights: &Gemma4Weights,
        provider: Option<&dyn FullAttentionProvider>,
    ) -> Result<Gemma4E2bMath, Gemma4E2bUnsupported> {
        let math = Gemma4E2bMath::admit(weights)?;
        let provider = provider.ok_or(Gemma4E2bUnsupported::ProviderPending)?;
        for layer in &weights.plan.layers {
            if layer.kind == Gemma4LayerKind::Full && !provider.supports(layer, math.causal) {
                return Err(Gemma4E2bUnsupported::ProviderPending);
            }
        }
        Ok(math)
    }

    pub(crate) fn qualify_serving(
        weights: &Gemma4Weights,
        full: Option<&dyn FullAttentionProvider>,
        local: Option<&dyn LocalAttentionProvider>,
    ) -> Result<Gemma4E2bMath, Gemma4E2bUnsupported> {
        let math = Self::qualify(weights, full)?;
        let local = local.ok_or(Gemma4E2bUnsupported::LocalAttentionPending)?;
        for layer in &weights.plan.layers {
            if layer.kind == Gemma4LayerKind::Sliding && !local.supports(layer, math.causal) {
                return Err(Gemma4E2bUnsupported::LocalAttentionPending);
            }
        }
        Ok(math)
    }

    pub(crate) fn new(
        context: Arc<Context>,
        stream: Arc<Stream>,
        weights: &Gemma4Weights,
        attention: Option<Arc<dyn FullAttentionProvider>>,
        module_path: Option<&Path>,
    ) -> Result<Self, Gemma4E2bStartupError> {
        Self::new_with_local(context, stream, weights, attention, None, module_path)
    }

    pub(crate) fn new_with_local(
        context: Arc<Context>,
        stream: Arc<Stream>,
        weights: &Gemma4Weights,
        attention: Option<Arc<dyn FullAttentionProvider>>,
        local: Option<Arc<dyn LocalAttentionProvider>>,
        module_path: Option<&Path>,
    ) -> Result<Self, Gemma4E2bStartupError> {
        let math = Self::qualify_serving(weights, attention.as_deref(), local.as_deref())?;
        let local = local.ok_or(Gemma4E2bUnsupported::LocalAttentionPending)?;
        Self::new_internal(
            context,
            stream,
            weights,
            attention,
            LocalAttentionRoute::Qualified(local),
            math,
            module_path,
        )
    }

    pub(crate) fn new_with_fa4_local(
        context: Arc<Context>,
        stream: Arc<Stream>,
        weights: &Gemma4Weights,
        attention: Option<Arc<dyn FullAttentionProvider>>,
        fa4_root: &Path,
        module_path: Option<&Path>,
    ) -> Result<Self, Gemma4E2bStartupError> {
        let math = Self::qualify(weights, attention.as_deref())?;
        match Fa4Local::load(context.clone(), fa4_root, &math) {
            Ok(local) => Self::new_with_local(
                context,
                stream,
                weights,
                attention,
                Some(Arc::new(local)),
                module_path,
            ),
            Err(error) => {
                if std::env::var_os(infinity_v3_config::infinity_env!("GEMMA4_SEMANTIC_LOCAL"))
                    .is_none()
                {
                    return Err(error.into());
                }
                // Explicit semantic route for geometries whose exact FA4
                // profile is not in the canonical31 bundle. The device leaf is
                // real CUDA math, but its FP32 score tree is not a numeric
                // replacement for the Native FA4 online-softmax tree.
                Self::new_diagnostic_leaf(context, stream, weights, attention, module_path)
            }
        }
    }

    pub(crate) fn new_diagnostic_leaf(
        context: Arc<Context>,
        stream: Arc<Stream>,
        weights: &Gemma4Weights,
        attention: Option<Arc<dyn FullAttentionProvider>>,
        module_path: Option<&Path>,
    ) -> Result<Self, Gemma4E2bStartupError> {
        let math = Self::qualify(weights, attention.as_deref())?;
        Self::new_internal(
            context,
            stream,
            weights,
            attention,
            LocalAttentionRoute::DiagnosticLeaf,
            math,
            module_path,
        )
    }

    fn new_internal(
        context: Arc<Context>,
        stream: Arc<Stream>,
        weights: &Gemma4Weights,
        attention: Option<Arc<dyn FullAttentionProvider>>,
        local_attention: LocalAttentionRoute,
        math: Gemma4E2bMath,
        module_path: Option<&Path>,
    ) -> Result<Self, Gemma4E2bStartupError> {
        context.bind_to_thread().map_err(start)?;
        let require_option_softcap = weights.config.contract == Gemma4Contract::RuneOptionTokens
            && weights.plan.readout.final_logit_softcap.is_some();
        let kernels = Gemma4Kernels::load(&context, module_path, require_option_softcap)?;
        let option_head = match weights.config.contract {
            Gemma4Contract::RuneOptionTokens => Some(Gemma4OptionHead::from_weights(weights)?),
            Gemma4Contract::Embedding => None,
        };
        let local = weights
            .plan
            .layers
            .iter()
            .find(|layer| layer.kind == Gemma4LayerKind::Sliding)
            .ok_or_else(|| {
                Gemma4E2bUnsupported::Geometry("dense E2B has no sliding layer".into())
            })?;
        let full = weights
            .plan
            .layers
            .iter()
            .find(|layer| layer.kind == Gemma4LayerKind::Full)
            .ok_or_else(|| Gemma4E2bUnsupported::Geometry("dense E2B has no full layer".into()))?;
        let local_rope = Gemma4RoPe::new(
            &stream,
            Arc::clone(&kernels.rope_tables),
            math.local_head_dim,
            math.local_head_dim / 2,
            local.rope.spec.theta,
            weights.config.max_position_embeddings,
        )?;
        let global_rope = Gemma4RoPe::new(
            &stream,
            Arc::clone(&kernels.rope_tables),
            math.global_head_dim,
            full.rope.rotated_pairs,
            full.rope.spec.theta,
            weights.config.max_position_embeddings,
        )?;
        let mut uploader = WeightUploader::new(Arc::clone(&stream))?;
        let uploaded = Self::upload_weights(&mut uploader, &stream, weights, &math)?;
        let moe = if math.moe.is_some() {
            Some(Gemma4MoeRuntime::new(
                &context,
                &stream,
                weights,
                &math,
                &mut uploader,
            )?)
        } else {
            None
        };
        uploader.finish()?;
        Ok(Self {
            config: weights.config.clone(),
            plan: weights.plan.clone(),
            math,
            kernels,
            weights: uploaded,
            local_rope,
            global_rope,
            plans: None,
            attention,
            local_attention,
            option_head,
            moe,
        })
    }

    fn upload_weights(
        uploader: &mut WeightUploader,
        stream: &Arc<Stream>,
        host: &Gemma4Weights,
        math: &Gemma4E2bMath,
    ) -> Result<WeightBuffers, BackendError> {
        // The typed loader rejects every projection bias before this point.
        let embed = upload(uploader, &[&host.embed], "embed_tokens")?;
        let (ple_embed, ple_model, ple_norm) = match host.ple.as_ref() {
            Some(ple) => (
                Some(upload(
                    uploader,
                    &[&ple.embed_tokens],
                    "embed_tokens_per_layer",
                )?),
                Some(upload(
                    uploader,
                    &[&ple.model_projection],
                    "per_layer_model_projection",
                )?),
                Some(upload_norm(stream, &ple.projection_norm, math.ple_width)?),
            ),
            None => (None, None, None),
        };
        let final_norm = upload_norm(stream, &host.final_norm, math.hidden)?;
        let mut layers = Vec::new();
        layers.try_reserve_exact(host.layers.len()).map_err(start)?;
        for (index, layer) in host.layers.iter().enumerate() {
            let plan = host
                .plan
                .layers
                .get(index)
                .ok_or_else(|| start("Gemma4 E2B layer plan missing"))?;
            let (k_norm, k, v) = match &layer.kv {
                Gemma4KvWeights::Producer(kv) => (
                    Some(upload_norm(stream, &kv.k_norm, plan.head_dim)?),
                    Some(upload(uploader, &[&kv.k.weight], "key projection")?),
                    kv.v.as_ref()
                        .map(|linear| upload(uploader, &[&linear.weight], "value projection"))
                        .transpose()?,
                ),
                Gemma4KvWeights::Shared(_) => (None, None, None),
            };
            let ple_layer = layer.ple.as_ref();
            layers.push(LayerBuffers {
                input_norm: upload_norm(stream, &layer.input_norm, math.hidden)?,
                post_attention_norm: upload_norm(stream, &layer.post_attention_norm, math.hidden)?,
                pre_feedforward_norm: upload_norm(
                    stream,
                    &layer.pre_feedforward_norm,
                    math.hidden,
                )?,
                post_feedforward_norm: upload_norm(
                    stream,
                    &layer.post_feedforward_norm,
                    math.hidden,
                )?,
                q_norm: upload_norm(stream, &layer.q_norm, plan.head_dim)?,
                k_norm,
                q: upload(uploader, &[&layer.q.weight], "query projection")?,
                k,
                v,
                o: upload(uploader, &[&layer.o.weight], "attention output projection")?,
                gate_up: upload(uploader, &[&layer.gate.weight, &layer.up.weight], "gate/up")?,
                down: upload(uploader, &[&layer.down.weight], "down")?,
                layer_scalar: upload(uploader, &[&layer.layer_scalar], "layer_scalar")?,
                layer_scalar_value: bf16_source_scalar(&layer.layer_scalar)?,
                ple_gate: ple_layer
                    .map(|ple| upload(uploader, &[&ple.input_gate.weight], "PLE input gate"))
                    .transpose()?,
                ple_projection: ple_layer
                    .map(|ple| upload(uploader, &[&ple.projection.weight], "PLE projection"))
                    .transpose()?,
                ple_post_norm: ple_layer
                    .map(|ple| upload_norm(stream, &ple.post_norm, math.hidden))
                    .transpose()?,
                moe_pre_ff2: layer
                    .moe
                    .as_ref()
                    .map(|moe| upload_norm(stream, &moe.pre_feedforward_norm_2, math.hidden))
                    .transpose()?,
                moe_post_ff1: layer
                    .moe
                    .as_ref()
                    .map(|moe| upload_norm(stream, &moe.post_feedforward_norm_1, math.hidden))
                    .transpose()?,
                moe_post_ff2: layer
                    .moe
                    .as_ref()
                    .map(|moe| upload_norm(stream, &moe.post_feedforward_norm_2, math.hidden))
                    .transpose()?,
            });
        }
        Ok(WeightBuffers {
            embed,
            final_norm,
            ple_model,
            ple_embed,
            ple_norm,
            layers,
        })
    }

    fn prepare_plans(
        &mut self,
        tokens: usize,
        pooled_rows: Option<usize>,
    ) -> Result<(), BackendError> {
        let rebuild = self
            .plans
            .as_ref()
            .is_none_or(|plans| plans.rows != tokens || plans.pooled_rows != pooled_rows);
        if rebuild {
            // Rebuild only host-side GEMM plans; the Work arena is retained.
            self.plans = Some(Plans::new(&self.config, &self.plan, tokens, pooled_rows)?);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn enqueue(
        &mut self,
        plain_blas: Option<&Blas>,
        stream: &Stream,
        work: &mut Gemma4E2bWork,
        batch: &Batch,
        pool: &Pool,
        max_length: usize,
        tokens: usize,
    ) -> Result<(), BackendError> {
        self.enqueue_impl(
            plain_blas,
            stream,
            work,
            batch,
            Some(pool),
            false,
            max_length,
            tokens,
            None,
        )
    }

    /// Text route that admits ordered raw indices; `finish_text` performs the
    /// bounded BF16 gather+widen into the flattened Native output.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn enqueue_text(
        &mut self,
        plain_blas: Option<&Blas>,
        stream: &Stream,
        work: &mut Gemma4E2bWork,
        batch: &Batch,
        pool: &Pool,
        max_length: usize,
        tokens: usize,
        lease: &HostIoLease,
    ) -> Result<(), BackendError> {
        self.enqueue_text_with_vision(
            plain_blas, stream, work, batch, pool, max_length, tokens, lease, None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn enqueue_text_with_vision(
        &mut self,
        plain_blas: Option<&Blas>,
        stream: &Stream,
        work: &mut Gemma4E2bWork,
        batch: &Batch,
        pool: &Pool,
        max_length: usize,
        tokens: usize,
        lease: &HostIoLease,
        vision: Option<VisionInjection<'_>>,
    ) -> Result<(), BackendError> {
        work.check_text_owner(lease, TextPhase::Uploaded)?;
        self.enqueue_impl(
            plain_blas,
            stream,
            work,
            batch,
            Some(pool),
            true,
            max_length,
            tokens,
            vision,
        )?;
        if let Some(state) = work.text_use.as_mut() {
            state.phase = TextPhase::Enqueued;
        }
        Ok(())
    }

    /// Backbone execution with either the ordinary embedding selection tail or
    /// the Rune hidden-only tail. Rune `tei_decide` ignores embedding
    /// pooled/raw selections, so the hidden-only policy keeps the physical
    /// Batch IDs/positions/boundaries and full-T backbone unchanged while
    /// skipping selection admission and the final F32 pooling kernel.
    #[allow(clippy::too_many_arguments)]
    fn enqueue_impl(
        &mut self,
        plain_blas: Option<&Blas>,
        stream: &Stream,
        work: &mut Gemma4E2bWork,
        batch: &Batch,
        pool: Option<&Pool>,
        allow_raw: bool,
        max_length: usize,
        tokens: usize,
        vision: Option<VisionInjection<'_>>,
    ) -> Result<(), BackendError> {
        let blas = plain_blas.ok_or_else(|| infer("Gemma4 E2B requires plain cuBLAS"))?;
        if tokens == 0 || batch.cumulative_seq_lengths.len() < 2 {
            return Err(infer("Gemma4 E2B requires tokens and sequence boundaries"));
        }
        if pool.is_some() && !allow_raw && !batch.raw_indices.is_empty() {
            return Err(infer(Gemma4E2bUnsupported::RawRowsPending.to_string()));
        }
        if batch.multimodal.iter().any(|media| media.is_some()) {
            return Err(infer(Gemma4E2bUnsupported::MediaPending.to_string()));
        }
        if batch.compact_input_ids.is_some()
            || batch.scatter_unfold.is_some()
            || batch.fold_gather.is_some()
        {
            return Err(infer(Gemma4E2bUnsupported::CompactionPending.to_string()));
        }
        if pool.is_some() && batch.pooled_indices.is_empty() {
            return Err(infer(Gemma4E2bUnsupported::PoolingPending.to_string()));
        }
        if max_length == 0 || max_length > self.config.max_position_embeddings {
            return Err(infer(
                "Gemma4 E2B max_length exceeds max_position_embeddings",
            ));
        }
        let mode = match pool {
            Some(Pool::Cls) => 0u32,
            Some(Pool::Mean) => 1u32,
            Some(Pool::LastToken) => 2u32,
            Some(Pool::Splade) => {
                return Err(infer(Gemma4E2bUnsupported::PoolingPending.to_string()))
            }
            None => 0u32,
        };
        // Native's default dense E2B bridge route enables CUDA graphs, so
        // tei_forward uses forward() (full T) for CLS/Last and tei_decide uses
        // full T unless graph capture is disabled or block mode is active.
        // Keep this executor on full T by default. The select_rows export and
        // pooled_last plans are cold infrastructure for a separate explicit
        // Native-eager diagnostic method, never an automatic M=B switch.
        let sequences = batch
            .cumulative_seq_lengths
            .len()
            .checked_sub(1)
            .ok_or_else(|| infer("Gemma4 E2B boundary count underflow"))?;
        // Hidden-only Rune ignores embedding selection metadata; the Work key
        // reserves no selection rows and the final F32 pool is skipped.
        let selections = if pool.is_some() {
            batch.pooled_indices.len()
        } else {
            0
        };
        work.check(tokens, batch.cumulative_seq_lengths.len(), selections)?;
        self.prepare_plans(tokens, None)?;
        let plans = self
            .plans
            .as_ref()
            .ok_or_else(|| infer("missing Gemma4 E2B plans"))?;
        let attention = self
            .attention
            .as_deref()
            .ok_or_else(|| infer(Gemma4E2bUnsupported::ProviderPending.to_string()))?;
        let math = &self.math;
        let weights = &self.weights;
        let kernels = &self.kernels;
        let rows = u32_size(tokens)?;
        let hidden = u32_size(math.hidden)?;
        let ple_width = u32_size(math.ple_width)?;
        let layers_u32 = u32_size(math.layers)?;
        let selections_u32 = u32_size(selections)?;
        let hidden_threads = reference_threads(math.hidden);
        let base = work.input.cu_deviceptr();
        let (position_bytes, boundary_bytes, selection_bytes) =
            crate::input_section_offsets(tokens, batch.cumulative_seq_lengths.len(), selections)?;
        let positions = base
            .checked_add(u64::try_from(position_bytes).map_err(infer)?)
            .ok_or_else(|| infer("Gemma4 position address overflow"))?;
        let mut boundaries = base
            .checked_add(u64::try_from(boundary_bytes).map_err(infer)?)
            .ok_or_else(|| infer("Gemma4 boundary address overflow"))?;
        let selected = base
            .checked_add(u64::try_from(selection_bytes).map_err(infer)?)
            .ok_or_else(|| infer("Gemma4 selection address overflow"))?;
        let state = work.state.cu_deviceptr();
        let next = work.next.cu_deviceptr();
        let normalized = work.normalized.cu_deviceptr();
        let mut current = state;
        let mut spare = next;
        let mut branch = work.branch.cu_deviceptr();
        let q_base = work.q.cu_deviceptr();
        let k_base = work.k.cu_deviceptr();
        let v_base = work.v.cu_deviceptr();
        let attention_out = work.attention.cu_deviceptr();
        let lse = work.lse.cu_deviceptr();
        let lse_bytes = u64::try_from(work.lse.num_bytes()).map_err(infer)?;
        let offsets_bytes = (sequences + 1)
            .checked_mul(4)
            .ok_or_else(|| infer("Gemma4 offset bytes overflow"))?;
        let gate_up = work.gate_up.cu_deviceptr();
        let activation = work.activation.cu_deviceptr();
        let ple = work.ple.cu_deviceptr();
        let ple_token = work.ple_token.cu_deviceptr();
        let attention_out_bytes = u64::try_from(work.attention.num_bytes()).map_err(infer)?;
        let shared_k = [
            work.shared_k[0].cu_deviceptr(),
            work.shared_k[1].cu_deviceptr(),
        ];
        let shared_v = [
            work.shared_v[0].cu_deviceptr(),
            work.shared_v[1].cu_deviceptr(),
        ];

        let mut ids = base;
        let mut table = weights.embed.cu_deviceptr();
        let mut gather_out = state;
        let mut gather_tokens = rows;
        let mut gather_hidden = hidden;
        let mut embedding_scale = math.embedding_scale;
        launch(
            &kernels.gather_scaled,
            stream,
            blocks(product(tokens, math.hidden)?)?,
            0,
            &mut [
                arg(&mut ids),
                arg(&mut table),
                arg(&mut gather_out),
                arg(&mut gather_tokens),
                arg(&mut gather_hidden),
                arg(&mut embedding_scale),
            ],
        )?;

        if math.ple_width > 0 {
            unsafe {
                blas.linear(
                    plans
                        .ple_model
                        .as_ref()
                        .ok_or_else(|| infer("Gemma4 PLE model plan missing"))?,
                    &work.state,
                    weights
                        .ple_model
                        .as_ref()
                        .ok_or_else(|| infer("Gemma4 PLE model weight missing"))?,
                    &mut work.ple,
                )?;
            }
            let ple_elements = product(
                tokens,
                math.layers
                    .checked_mul(math.ple_width)
                    .ok_or_else(|| infer("Gemma4 PLE width overflow"))?,
            )?;
            let mut scale_in = ple;
            let mut scale_out = ple;
            let mut scale_count = u32_size(ple_elements)?;
            let mut ple_model_scale = math.ple_model_scale;
            launch(
                &kernels.scale,
                stream,
                blocks(ple_elements)?,
                0,
                &mut [
                    arg(&mut scale_in),
                    arg(&mut scale_out),
                    arg(&mut scale_count),
                    arg(&mut ple_model_scale),
                ],
            )?;
            let mut norm_in = ple;
            let mut norm_scale = weights
                .ple_norm
                .as_ref()
                .ok_or_else(|| infer("Gemma4 PLE norm missing"))?
                .cu_deviceptr();
            let mut norm_out = ple;
            let mut norm_rows = u32_size(product(tokens, math.layers)?)?;
            let mut norm_width = ple_width;
            let mut norm_heads = 1u32;
            let mut norm_stride = u64::from(ple_width);
            let mut norm_eps = math.norm_eps;
            launch_threads(
                &kernels.norm,
                stream,
                (norm_rows, 1, 1),
                reference_threads(math.ple_width),
                0,
                &mut [
                    arg(&mut norm_in),
                    arg(&mut norm_scale),
                    arg(&mut norm_out),
                    arg(&mut norm_rows),
                    arg(&mut norm_width),
                    arg(&mut norm_heads),
                    arg(&mut norm_stride),
                    arg(&mut norm_eps),
                ],
            )?;
            let mut ple_ids = base;
            let mut ple_table = weights
                .ple_embed
                .as_ref()
                .ok_or_else(|| infer("Gemma4 PLE embedding missing"))?
                .cu_deviceptr();
            let mut ple_out = ple_token;
            let mut ple_tokens = rows;
            let mut ple_total_width = u32_size(
                math.layers
                    .checked_mul(math.ple_width)
                    .ok_or_else(|| infer("Gemma4 PLE width overflow"))?,
            )?;
            let mut ple_token_scale = math.ple_token_scale;
            launch(
                &kernels.gather_scaled,
                stream,
                blocks(product(
                    tokens,
                    math.layers
                        .checked_mul(math.ple_width)
                        .ok_or_else(|| infer("Gemma4 PLE width overflow"))?,
                )?)?,
                0,
                &mut [
                    arg(&mut ple_ids),
                    arg(&mut ple_table),
                    arg(&mut ple_out),
                    arg(&mut ple_tokens),
                    arg(&mut ple_total_width),
                    arg(&mut ple_token_scale),
                ],
            )?;
            let mut combine_projected = ple;
            let mut combine_token = ple_token;
            let mut combine_out = ple;
            let mut combine_count = u32_size(ple_elements)?;
            let mut combine_scale = math.ple_combine_scale;
            launch(
                &kernels.ple_combine,
                stream,
                blocks(ple_elements)?,
                0,
                &mut [
                    arg(&mut combine_projected),
                    arg(&mut combine_token),
                    arg(&mut combine_out),
                    arg(&mut combine_count),
                    arg(&mut combine_scale),
                ],
            )?;
        }
        if let Some(vision) = vision {
            let soft = vision.work.soft_tokens();
            if vision.rows.len() != soft {
                return Err(infer(
                    "Gemma4 vision injection rows differ from projected soft tokens",
                ));
            }
            if vision.rows.iter().any(|row| *row as usize >= tokens) {
                return Err(infer(
                    "Gemma4 vision injection row exceeds the packed text tokens",
                ));
            }
            if vision.work.projected().len() < product(soft, math.hidden)? {
                return Err(infer(
                    "Gemma4 vision projected rows are shorter than the text width",
                ));
            }
            let mut destination = work.state.cu_deviceptr();
            let mut features = vision.work.projected().cu_deviceptr();
            let mut row_ptr = vision.work.scatter_rows_ptr();
            let mut row_count = u32_size(vision.rows.len())?;
            let mut width = hidden;
            launch(
                &kernels.scatter,
                stream,
                blocks(product(vision.rows.len(), math.hidden)?)?,
                0,
                &mut [
                    arg(&mut destination),
                    arg(&mut features),
                    arg(&mut row_ptr),
                    arg(&mut row_count),
                    arg(&mut width),
                ],
            )?;
        }
        let moe_frame = match self.moe.as_ref() {
            Some(moe) => Some(
                moe.provider
                    .prepare(tokens)
                    .map_err(ShapeError::into_backend)?,
            ),
            None => None,
        };
        let local_rope = &self.local_rope;
        let global_rope = &self.global_rope;
        for (index, (layer, buffers)) in self.plan.layers.iter().zip(&weights.layers).enumerate() {
            let layer_plan = layer;
            let rope = if layer_plan.kind == Gemma4LayerKind::Sliding {
                local_rope
            } else {
                global_rope
            };
            let plan = plans
                .layers
                .get(index)
                .ok_or_else(|| infer("Gemma4 layer plan missing"))?;
            let mut norm_input = current;
            let mut norm_weight = buffers.input_norm.cu_deviceptr();
            let mut norm_out = normalized;
            let mut layer_rows = rows;
            let mut layer_width = hidden;
            let mut layer_heads = 1u32;
            let mut layer_stride = u64::from(hidden);
            let mut layer_eps = math.norm_eps;
            launch_threads(
                &kernels.norm,
                stream,
                (layer_rows, 1, 1),
                hidden_threads,
                0,
                &mut [
                    arg(&mut norm_input),
                    arg(&mut norm_weight),
                    arg(&mut norm_out),
                    arg(&mut layer_rows),
                    arg(&mut layer_width),
                    arg(&mut layer_heads),
                    arg(&mut layer_stride),
                    arg(&mut layer_eps),
                ],
            )?;
            unsafe {
                blas.linear(&plan.q, &work.normalized, &buffers.q, &mut work.q)?;
                if let (Some(k_plan), Some(v_plan), Some(k_weight), Some(v_weight)) =
                    (&plan.k, &plan.v, &buffers.k, &buffers.v)
                {
                    blas.linear(k_plan, &work.normalized, k_weight, &mut work.k)?;
                    blas.linear(v_plan, &work.normalized, v_weight, &mut work.v)?;
                }
            }
            if layer_plan.k_equals_v {
                let bytes = product(tokens, layer_plan.kv_width)?
                    .checked_mul(2)
                    .ok_or_else(|| infer("Gemma4 K=V bytes overflow"))?;
                // SAFETY: both buffers are retained by Work and the copy precedes
                // K normalization/rotation on the same ordered stream.
                unsafe {
                    cuda_core::memory::memcpy_dtod_async(v_base, k_base, bytes, stream.cu_stream())
                }
                .map_err(infer)?;
            }
            let q_heads_usize = layer_plan.q_width / layer_plan.head_dim;
            let q_heads = u32_size(q_heads_usize)?;
            let q_width = u32_size(layer_plan.q_width)?;
            let q_head_dim = u32_size(layer_plan.head_dim)?;
            let mut q_ptr = q_base;
            let mut q_norm_scale = buffers.q_norm.cu_deviceptr();
            let mut q_norm_out = q_base;
            let mut q_rows = u32_size(product(tokens, q_heads_usize)?)?;
            let mut q_norm_width = q_head_dim;
            let mut q_norm_heads = q_heads;
            let mut q_stride = u64::from(q_width);
            let mut q_eps = math.norm_eps;
            launch_threads(
                &kernels.norm,
                stream,
                (q_rows, 1, 1),
                reference_threads(layer_plan.head_dim),
                0,
                &mut [
                    arg(&mut q_ptr),
                    arg(&mut q_norm_scale),
                    arg(&mut q_norm_out),
                    arg(&mut q_rows),
                    arg(&mut q_norm_width),
                    arg(&mut q_norm_heads),
                    arg(&mut q_stride),
                    arg(&mut q_eps),
                ],
            )?;
            let kv_heads = u32_size(layer_plan.num_kv_heads)?;
            let kv_width = u32_size(layer_plan.kv_width)?;
            if matches!(layer_plan.kv, Gemma4KvPlan::Producer { .. }) {
                let mut k_ptr = k_base;
                let mut k_norm_scale = buffers
                    .k_norm
                    .as_ref()
                    .ok_or_else(|| infer("Gemma4 K norm missing"))?
                    .cu_deviceptr();
                let mut k_norm_out = k_base;
                let mut k_rows = u32_size(product(tokens, layer_plan.num_kv_heads)?)?;
                let mut k_norm_width = q_head_dim;
                let mut k_norm_heads = kv_heads;
                let mut k_stride = u64::from(kv_width);
                let mut k_eps = math.norm_eps;
                launch_threads(
                    &kernels.norm,
                    stream,
                    (k_rows, 1, 1),
                    reference_threads(layer_plan.head_dim),
                    0,
                    &mut [
                        arg(&mut k_ptr),
                        arg(&mut k_norm_scale),
                        arg(&mut k_norm_out),
                        arg(&mut k_rows),
                        arg(&mut k_norm_width),
                        arg(&mut k_norm_heads),
                        arg(&mut k_stride),
                        arg(&mut k_eps),
                    ],
                )?;
            }
            let mut cos = rope.cosine.cu_deviceptr();
            let mut sin = rope.sine.cu_deviceptr();
            let mut pos = positions;
            let mut r_tokens = rows;
            let mut r_q_heads = q_heads;
            let mut r_kv_heads = kv_heads;
            let mut r_head_dim = q_head_dim;
            let mut r_rotary = q_head_dim;
            let mut r_q_stride = q_stride;
            let mut r_k_stride = u64::from(kv_width);
            let mut r_table = rope.stride;
            if matches!(layer_plan.kv, Gemma4KvPlan::Shared { .. }) {
                let mut q_only = q_base;
                let rope_launch = RopeLaunchPlan::new(
                    tokens,
                    q_heads_usize,
                    0,
                    q_head_dim as usize,
                    r_rotary as usize,
                )?;
                launch(
                    &kernels.rope_q,
                    stream,
                    rope_launch.grid()?,
                    0,
                    &mut [
                        arg(&mut q_only),
                        arg(&mut cos),
                        arg(&mut sin),
                        arg(&mut pos),
                        arg(&mut r_tokens),
                        arg(&mut r_q_heads),
                        arg(&mut r_head_dim),
                        arg(&mut r_rotary),
                        arg(&mut r_q_stride),
                        arg(&mut r_table),
                    ],
                )?;
            } else {
                let mut rope_q = q_base;
                let mut rope_k = k_base;
                let rope_launch = RopeLaunchPlan::new(
                    tokens,
                    q_heads_usize,
                    layer_plan.num_kv_heads,
                    q_head_dim as usize,
                    r_rotary as usize,
                )?;
                launch(
                    &kernels.rope_neox,
                    stream,
                    rope_launch.grid()?,
                    0,
                    &mut [
                        arg(&mut rope_q),
                        arg(&mut rope_k),
                        arg(&mut cos),
                        arg(&mut sin),
                        arg(&mut pos),
                        arg(&mut r_tokens),
                        arg(&mut r_q_heads),
                        arg(&mut r_kv_heads),
                        arg(&mut r_head_dim),
                        arg(&mut r_rotary),
                        arg(&mut r_q_stride),
                        arg(&mut r_k_stride),
                        arg(&mut r_table),
                    ],
                )?;
                let mut v_ptr = v_base;
                let mut v_out = v_base;
                let mut v_rows = u32_size(product(tokens, layer_plan.num_kv_heads)?)?;
                let mut v_width = q_head_dim;
                let mut v_heads = kv_heads;
                let mut v_stride = u64::from(kv_width);
                let mut v_eps = math.norm_eps;
                launch_threads(
                    &kernels.unit_norm,
                    stream,
                    (v_rows, 1, 1),
                    reference_threads(layer_plan.head_dim),
                    0,
                    &mut [
                        arg(&mut v_ptr),
                        arg(&mut v_out),
                        arg(&mut v_rows),
                        arg(&mut v_width),
                        arg(&mut v_heads),
                        arg(&mut v_stride),
                        arg(&mut v_eps),
                    ],
                )?;
            }
            let category = if layer_plan.kind == Gemma4LayerKind::Full {
                0usize
            } else {
                1usize
            };
            if let Gemma4KvPlan::Producer {
                stores_full_length: true,
            } = layer_plan.kv
            {
                let bytes = product(tokens, layer_plan.kv_width)?
                    .checked_mul(2)
                    .ok_or_else(|| infer("Gemma4 shared KV bytes overflow"))?;
                unsafe {
                    cuda_core::memory::memcpy_dtod_async(
                        shared_k[category],
                        k_base,
                        bytes,
                        stream.cu_stream(),
                    )
                    .map_err(infer)?;
                    cuda_core::memory::memcpy_dtod_async(
                        shared_v[category],
                        v_base,
                        bytes,
                        stream.cu_stream(),
                    )
                    .map_err(infer)?;
                }
            }
            let (k_ptr, v_ptr) = if matches!(layer_plan.kv, Gemma4KvPlan::Shared { .. }) {
                (shared_k[category], shared_v[category])
            } else {
                (k_base, v_base)
            };
            match layer_plan.kind {
                Gemma4LayerKind::Full => unsafe {
                    attention.forward_batch_bound(
                        stream,
                        layer_plan,
                        q_base,
                        k_ptr,
                        v_ptr,
                        attention_out,
                        attention_out_bytes,
                        lse,
                        lse_bytes,
                        boundaries,
                        offsets_bytes,
                        tokens,
                        sequences,
                        max_length,
                        math.causal,
                    )?;
                },
                Gemma4LayerKind::Sliding => match &self.local_attention {
                    LocalAttentionRoute::Qualified(provider) => unsafe {
                        provider.forward_batch_bound(
                            stream,
                            layer_plan,
                            q_base,
                            k_ptr,
                            v_ptr,
                            attention_out,
                            attention_out_bytes,
                            lse,
                            lse_bytes,
                            boundaries,
                            offsets_bytes,
                            tokens,
                            sequences,
                            max_length,
                            math.causal,
                        )?;
                    },
                    LocalAttentionRoute::DiagnosticLeaf => {
                        let mut attn_q = q_base;
                        let mut attn_k = k_ptr;
                        let mut attn_v = v_ptr;
                        let mut attn_out = attention_out;
                        let mut attn_batch = u32_size(sequences)?;
                        let mut attn_heads = q_heads;
                        let mut attn_kv_heads = kv_heads;
                        let mut attn_head_dim = q_head_dim;
                        let mut attn_q_stride = q_stride;
                        let mut attn_kv_stride = u64::from(kv_width);
                        let mut attn_max_q = u32_size(max_length)?;
                        let mut attn_support = math.local_support;
                        let mut attn_scale = 1.0f32;
                        let mut attn_causal = u32::from(math.causal);
                        let mut attn_left = math.local_left;
                        let mut attn_right = math.local_right;
                        launch_threads(
                            &kernels.attention_256,
                            stream,
                            (u32_size(max_length)?, q_heads, u32_size(sequences)?),
                            256,
                            math.local_support
                                .checked_mul(4)
                                .ok_or_else(|| infer("Gemma4 local shared bytes overflow"))?,
                            &mut [
                                arg(&mut attn_q),
                                arg(&mut attn_k),
                                arg(&mut attn_v),
                                arg(&mut boundaries),
                                arg(&mut boundaries),
                                arg(&mut attn_out),
                                arg(&mut attn_batch),
                                arg(&mut attn_heads),
                                arg(&mut attn_kv_heads),
                                arg(&mut attn_head_dim),
                                arg(&mut attn_q_stride),
                                arg(&mut attn_kv_stride),
                                arg(&mut attn_max_q),
                                arg(&mut attn_support),
                                arg(&mut attn_scale),
                                arg(&mut attn_causal),
                                arg(&mut attn_left),
                                arg(&mut attn_right),
                            ],
                        )?;
                    }
                },
            }
            unsafe {
                blas.linear(&plan.o, &work.attention, &buffers.o, &mut work.branch)?;
            }
            let mut residual = current;
            let mut rn_scale = buffers.post_attention_norm.cu_deviceptr();
            let mut rn_out = spare;
            let mut rn_rows = rows;
            let mut rn_width = hidden;
            let mut rn_eps = math.norm_eps;
            let mut rn_branch = branch;
            launch_threads(
                &kernels.residual_norm,
                stream,
                (rn_rows, 1, 1),
                hidden_threads,
                0,
                &mut [
                    arg(&mut rn_branch),
                    arg(&mut residual),
                    arg(&mut rn_scale),
                    arg(&mut rn_out),
                    arg(&mut rn_rows),
                    arg(&mut rn_width),
                    arg(&mut rn_eps),
                ],
            )?;
            core::mem::swap(&mut current, &mut spare);
            let mut ff_in = current;
            let mut ff_scale = buffers.pre_feedforward_norm.cu_deviceptr();
            let mut ff_out = normalized;
            let mut ff_rows = rows;
            let mut ff_width = hidden;
            let mut ff_heads = 1u32;
            let mut ff_stride = u64::from(hidden);
            let mut ff_eps = math.norm_eps;
            launch_threads(
                &kernels.norm,
                stream,
                (ff_rows, 1, 1),
                hidden_threads,
                0,
                &mut [
                    arg(&mut ff_in),
                    arg(&mut ff_scale),
                    arg(&mut ff_out),
                    arg(&mut ff_rows),
                    arg(&mut ff_width),
                    arg(&mut ff_heads),
                    arg(&mut ff_stride),
                    arg(&mut ff_eps),
                ],
            )?;
            unsafe {
                blas.linear(
                    &plan.gate_up,
                    &work.normalized,
                    &buffers.gate_up,
                    &mut work.gate_up,
                )?;
            }
            let mut gate_in = gate_up;
            let mut gate_out = activation;
            let mut gate_rows = rows;
            let mut gate_width = u32_size(layer_plan.dense_intermediate_size)?;
            launch(
                &kernels.gated_gelu,
                stream,
                blocks(product(tokens, layer_plan.dense_intermediate_size)?)?,
                0,
                &mut [
                    arg(&mut gate_in),
                    arg(&mut gate_out),
                    arg(&mut gate_rows),
                    arg(&mut gate_width),
                ],
            )?;
            unsafe {
                blas.linear(
                    &plan.down,
                    &work.activation,
                    &buffers.down,
                    &mut work.branch,
                )?;
            }
            if let (
                Some(moe),
                Some(frame),
                Some(router_plan),
                Some(moe_layer),
                Some(router_scale),
                Some(router_projection),
                Some(per_expert_scale),
            ) = (
                self.moe.as_ref(),
                moe_frame.as_ref(),
                plans.moe_router.as_ref(),
                self.moe
                    .as_ref()
                    .and_then(|moe| moe.layers.get(index))
                    .and_then(Option::as_ref),
                self.moe
                    .as_ref()
                    .and_then(|moe| moe.router_scale.get(index))
                    .and_then(Option::as_ref),
                self.moe
                    .as_ref()
                    .and_then(|moe| moe.router_projection.get(index))
                    .and_then(Option::as_ref),
                self.moe
                    .as_ref()
                    .and_then(|moe| moe.per_expert_scale.get(index))
                    .and_then(Option::as_ref),
            ) {
                let pre_ff2 = buffers
                    .moe_pre_ff2
                    .as_ref()
                    .ok_or_else(|| infer("Gemma4 MoE pre-FF2 norm missing"))?
                    .cu_deviceptr();
                let post_ff1 = buffers
                    .moe_post_ff1
                    .as_ref()
                    .ok_or_else(|| infer("Gemma4 MoE post-FF1 norm missing"))?
                    .cu_deviceptr();
                let post_ff2 = buffers
                    .moe_post_ff2
                    .as_ref()
                    .ok_or_else(|| infer("Gemma4 MoE post-FF2 norm missing"))?
                    .cu_deviceptr();
                let normalized_ptr = work.normalized.cu_deviceptr();
                let activation_ptr = work.activation.cu_deviceptr();
                let output_ptr = work.output.cu_deviceptr();
                let mut moe_in = current;
                let mut moe_scale = pre_ff2;
                let mut moe_out = normalized_ptr;
                let mut moe_rows = rows;
                let mut moe_width = hidden;
                let mut moe_heads = 1u32;
                let mut moe_stride = u64::from(hidden);
                let mut moe_eps = math.norm_eps;
                launch_threads(
                    &kernels.norm,
                    stream,
                    (moe_rows, 1, 1),
                    hidden_threads,
                    0,
                    &mut [
                        arg(&mut moe_in),
                        arg(&mut moe_scale),
                        arg(&mut moe_out),
                        arg(&mut moe_rows),
                        arg(&mut moe_width),
                        arg(&mut moe_heads),
                        arg(&mut moe_stride),
                        arg(&mut moe_eps),
                    ],
                )?;
                let mut unit_in = current;
                let mut unit_out = activation_ptr;
                let mut unit_rows = rows;
                let mut unit_width = hidden;
                let mut unit_heads = 1u32;
                let mut unit_stride = u64::from(hidden);
                let mut unit_eps = math.norm_eps;
                launch_threads(
                    &kernels.unit_norm,
                    stream,
                    (unit_rows, 1, 1),
                    hidden_threads,
                    0,
                    &mut [
                        arg(&mut unit_in),
                        arg(&mut unit_out),
                        arg(&mut unit_rows),
                        arg(&mut unit_width),
                        arg(&mut unit_heads),
                        arg(&mut unit_stride),
                        arg(&mut unit_eps),
                    ],
                )?;
                let router_slot: &mut Option<DeviceBuffer<f32>> = &mut work.router_input;
                let router_input = router_slot
                    .as_mut()
                    .ok_or_else(|| infer("Gemma4 Work router input was released"))?;
                let scratch_owner: &mut Option<MoeScratch> = &mut work.moe_scratch;
                let scratch_slot = scratch_owner
                    .as_mut()
                    .ok_or_else(|| infer("Gemma4 Work MoE scratch was released"))?;
                let mut unit_ptr = activation_ptr;
                let mut scale_ptr = router_scale.cu_deviceptr();
                let mut router_ptr = router_input.cu_deviceptr();
                let mut route_count = u32_size(product(tokens, math.hidden)?)?;
                let mut moe_root = (1.0f64 / (math.hidden as f64).sqrt()) as f32;
                launch(
                    &kernels.router_scale,
                    stream,
                    blocks(product(tokens, math.hidden)?)?,
                    0,
                    &mut [
                        arg(&mut unit_ptr),
                        arg(&mut scale_ptr),
                        arg(&mut router_ptr),
                        arg(&mut route_count),
                        arg(&mut moe_root),
                    ],
                )?;
                let args = router_plan.arguments(
                    router_input.cu_deviceptr(),
                    router_projection.cu_deviceptr(),
                    scratch_slot.logits_f32().cu_deviceptr(),
                );
                // SAFETY: every router owner is retained by this request.
                unsafe { blas.enqueue_f32_projection(&args)? };
                // SAFETY: stream order makes the F32 router logits visible to the
                // sidecar, and every buffer remains owned by this request Work.
                unsafe {
                    moe.provider.forward_precomputed(
                        frame,
                        moe_layer,
                        per_expert_scale,
                        stream,
                        &work.normalized,
                        &mut work.output,
                        scratch_slot,
                    )?;
                }
                let mut dense_in = branch;
                let mut dense_scale = post_ff1;
                let mut dense_out = normalized_ptr;
                launch_threads(
                    &kernels.norm,
                    stream,
                    (rows, 1, 1),
                    hidden_threads,
                    0,
                    &mut [
                        arg(&mut dense_in),
                        arg(&mut dense_scale),
                        arg(&mut dense_out),
                        arg(&mut moe_rows),
                        arg(&mut moe_width),
                        arg(&mut moe_heads),
                        arg(&mut moe_stride),
                        arg(&mut moe_eps),
                    ],
                )?;
                let mut routed_in = output_ptr;
                let mut routed_scale = post_ff2;
                let mut routed_out = work.branch.cu_deviceptr();
                launch_threads(
                    &kernels.residual_norm,
                    stream,
                    (rows, 1, 1),
                    hidden_threads,
                    0,
                    &mut [
                        arg(&mut routed_in),
                        arg(&mut dense_out),
                        arg(&mut routed_scale),
                        arg(&mut routed_out),
                        arg(&mut moe_rows),
                        arg(&mut moe_width),
                        arg(&mut moe_eps),
                    ],
                )?;
                branch = work.branch.cu_deviceptr();
            }
            let mut residual = current;
            let mut rn_scale = buffers.post_feedforward_norm.cu_deviceptr();
            let mut rn_out = spare;
            let mut rn_rows = rows;
            let mut rn_width = hidden;
            let mut rn_eps = math.norm_eps;
            let mut rn_branch = branch;
            launch_threads(
                &kernels.residual_norm,
                stream,
                (rn_rows, 1, 1),
                hidden_threads,
                0,
                &mut [
                    arg(&mut rn_branch),
                    arg(&mut residual),
                    arg(&mut rn_scale),
                    arg(&mut rn_out),
                    arg(&mut rn_rows),
                    arg(&mut rn_width),
                    arg(&mut rn_eps),
                ],
            )?;
            core::mem::swap(&mut current, &mut spare);
            // The PLE gate input is the current residual stream, not the
            // initial embedding buffer.
            let current_device = if current == work.state.cu_deviceptr() {
                &work.state
            } else {
                &work.next
            };
            if math.ple_width > 0 {
                unsafe {
                    blas.linear(
                        plan.ple_gate
                            .as_ref()
                            .ok_or_else(|| infer("Gemma4 PLE gate plan missing"))?,
                        current_device,
                        buffers
                            .ple_gate
                            .as_ref()
                            .ok_or_else(|| infer("Gemma4 PLE gate weight missing"))?,
                        &mut work.branch,
                    )?;
                }
                let mut ple_gate_in = branch;
                let mut ple_values = ple;
                let mut ple_gate_out = activation;
                let mut ple_gate_rows = rows;
                let mut ple_gate_width = ple_width;
                let mut ple_layers = layers_u32;
                let mut layer_index = u32_size(index)?;
                launch(
                    &kernels.ple_gate,
                    stream,
                    blocks(product(tokens, math.ple_width)?)?,
                    0,
                    &mut [
                        arg(&mut ple_gate_in),
                        arg(&mut ple_values),
                        arg(&mut ple_gate_out),
                        arg(&mut ple_gate_rows),
                        arg(&mut ple_gate_width),
                        arg(&mut ple_layers),
                        arg(&mut layer_index),
                    ],
                )?;
                unsafe {
                    blas.linear(
                        plan.ple_projection
                            .as_ref()
                            .ok_or_else(|| infer("Gemma4 PLE projection plan missing"))?,
                        &work.activation,
                        buffers
                            .ple_projection
                            .as_ref()
                            .ok_or_else(|| infer("Gemma4 PLE projection weight missing"))?,
                        &mut work.branch,
                    )?;
                }
                let mut residual = current;
                let mut rn_scale = buffers
                    .ple_post_norm
                    .as_ref()
                    .ok_or_else(|| infer("Gemma4 PLE post norm missing"))?
                    .cu_deviceptr();
                let mut rn_out = spare;
                let mut rn_rows = rows;
                let mut rn_width = hidden;
                let mut rn_eps = math.norm_eps;
                let mut rn_branch = branch;
                launch_threads(
                    &kernels.residual_norm,
                    stream,
                    (rn_rows, 1, 1),
                    hidden_threads,
                    0,
                    &mut [
                        arg(&mut rn_branch),
                        arg(&mut residual),
                        arg(&mut rn_scale),
                        arg(&mut rn_out),
                        arg(&mut rn_rows),
                        arg(&mut rn_width),
                        arg(&mut rn_eps),
                    ],
                )?;
                core::mem::swap(&mut current, &mut spare);
            }
            let mut scalar_in = current;
            let mut scalar_out = current;
            let mut scalar_count = u32_size(product(tokens, math.hidden)?)?;
            let mut scalar = buffers.layer_scalar_value;
            launch(
                &kernels.scale,
                stream,
                blocks(product(tokens, math.hidden)?)?,
                0,
                &mut [
                    arg(&mut scalar_in),
                    arg(&mut scalar_out),
                    arg(&mut scalar_count),
                    arg(&mut scalar),
                ],
            )?;
        }
        let mut final_in = current;
        let mut final_scale = weights.final_norm.cu_deviceptr();
        let mut final_out = normalized;
        let mut final_rows = rows;
        let mut final_width = hidden;
        let mut final_heads = 1u32;
        let mut final_stride = u64::from(hidden);
        let mut final_eps = math.norm_eps;
        launch_threads(
            &kernels.norm,
            stream,
            (final_rows, 1, 1),
            hidden_threads,
            0,
            &mut [
                arg(&mut final_in),
                arg(&mut final_scale),
                arg(&mut final_out),
                arg(&mut final_rows),
                arg(&mut final_width),
                arg(&mut final_heads),
                arg(&mut final_stride),
                arg(&mut final_eps),
            ],
        )?;
        if let Some(pool) = pool {
            if selections == 0 {
                return Ok(());
            }
            let mut pool_in = normalized;
            let mut pool_bounds = boundaries;
            let mut pool_selected = selected;
            let mut pool_out = work.pooled.cu_deviceptr();
            if *pool == Pool::Mean {
                // Native bridge.cpp:1478-1489 selects packed BF16 mean for
                // contiguous BF16 h + Mean + pooled_count <= 65535. The
                // published Gemma2/EG2 primitive is reused unchanged; its
                // BF16 partial/tree/reciprocal/mul+0 result is widened to F32.
                let mean = MeanPlan::new(
                    tokens,
                    math.hidden,
                    &batch.cumulative_seq_lengths,
                    &batch.pooled_indices,
                )
                .map_err(|error| infer(format!("Gemma4 E2B mean plan: {error}")))?;
                let geometry = mean
                    .geometry()
                    .map_err(|error| infer(format!("Gemma4 E2B mean grid: {error}")))?;
                let mut pool_count = mean.count();
                let mut pool_width = mean.width();
                launch_threads(
                    &kernels.gemma2.mean,
                    stream,
                    (geometry.grid()[0], geometry.grid()[1], geometry.grid()[2]),
                    geometry.block()[0],
                    0,
                    &mut [
                        arg(&mut pool_in),
                        arg(&mut pool_bounds),
                        arg(&mut pool_selected),
                        arg(&mut pool_out),
                        arg(&mut pool_count),
                        arg(&mut pool_width),
                    ],
                )?;
            } else {
                // CLS/Last remain the generic BF16-bit gather widened to F32.
                let mut pool_count = selections_u32;
                let mut pool_width = hidden;
                let mut pool_mode = mode;
                launch(
                    &kernels.pool,
                    stream,
                    blocks(product(selections, math.hidden)?)?,
                    0,
                    &mut [
                        arg(&mut pool_in),
                        arg(&mut pool_bounds),
                        arg(&mut pool_selected),
                        arg(&mut pool_out),
                        arg(&mut pool_count),
                        arg(&mut pool_width),
                        arg(&mut pool_mode),
                    ],
                )?;
            }
        }
        Ok(())
    }

    /// Complete a text request with pooled selections followed by ordered raw
    /// token spans in the Native flattened layout. Pooled values come from the
    /// existing BF16 gather/PreparedGemma2 mean tree in `work.pooled`; raw rows
    /// use the existing BF16 bit gather + widen export over `work.normalized`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn finish_text(
        &self,
        stream: &Stream,
        work: &mut Gemma4E2bWork,
        batch: &Batch,
        pooled_rows: usize,
        raw_rows: usize,
        lease: &mut HostIoLease,
    ) -> Result<(), BackendError> {
        if !work.uses_stream(stream) {
            return Err(infer("Gemma4 E2B Work belongs to a different stream"));
        }
        if !lease.uses_stream(stream) {
            return Err(infer(
                "Gemma4 E2B HostIo lease belongs to a different stream",
            ));
        }
        if pooled_rows != batch.pooled_indices.len()
            || raw_rows != crate::output::raw_rows(batch)?
            || raw_rows > work.raw_capacity()
        {
            return Err(infer(
                "Gemma4 E2B text output counts disagree with the retained Work",
            ));
        }
        work.check_text_owner(lease, TextPhase::Enqueued)?;
        let pooled_elements = product(pooled_rows, work.hidden)?;
        let raw_elements = product(raw_rows, work.hidden)?;
        let total = pooled_elements
            .checked_add(raw_elements)
            .ok_or_else(|| infer("Gemma4 E2B text output size overflow"))?;
        let prefix_words = crate::input_elements(
            batch.input_ids.len(),
            batch.cumulative_seq_lengths.len(),
            pooled_rows,
        )?;
        let tail_end = prefix_words
            .checked_add(raw_rows)
            .ok_or_else(|| infer("Gemma4 E2B raw tail range overflow"))?;
        if lease.slot()?.output.len() < total {
            return Err(infer("Gemma4 E2B text output exceeds the HostIo lease"));
        }
        let raw_ptr = {
            let input = lease.slot()?.input.as_slice();
            if input.len() < tail_end {
                return Err(infer("Gemma4 E2B raw tail exceeds the leased packed input"));
            }
            input[prefix_words..tail_end].as_ptr()
        };
        // Checked retention starts before the first async output operation.
        lease.begin_checked()?;
        let host = lease.slot_mut()?.output.as_mut_slice();
        crate::download_output_range(stream, &work.pooled, 0, host, 0, pooled_elements)?;
        #[cfg(test)]
        if INJECT_TEXT_FINISH_AFTER_POOLED.swap(false, std::sync::atomic::Ordering::SeqCst) {
            return Err(infer("injected text finish failure after pooled D2H"));
        }
        if raw_rows > 0 {
            let selected = work
                .raw_selected
                .as_ref()
                .ok_or_else(|| infer("Gemma4 E2B raw index device table missing"))?;
            let raw_output = work
                .raw_output
                .as_ref()
                .ok_or_else(|| infer("Gemma4 E2B raw output arena missing"))?;
            // SAFETY: the lease stays begun through the event; the raw tail is
            // immutable after stage_text and is owned by this request lease.
            unsafe {
                cuda_core::memory::memcpy_htod_async(
                    selected.cu_deviceptr(),
                    raw_ptr,
                    product(raw_rows, 4)?,
                    stream.cu_stream(),
                )
            }
            .map_err(infer)?;
            let mut input = work.normalized.cu_deviceptr();
            let mut output = raw_output.cu_deviceptr();
            let mut selected_ptr = selected.cu_deviceptr();
            let mut count = u32_size(raw_rows)?;
            let mut width = u32_size(work.hidden)?;
            launch(
                &self.kernels.bf16_to_f32,
                stream,
                blocks(product(raw_rows, work.hidden)?)?,
                0,
                &mut [
                    arg(&mut input),
                    arg(&mut output),
                    arg(&mut selected_ptr),
                    arg(&mut count),
                    arg(&mut width),
                ],
            )?;
            crate::download_output_range(
                stream,
                raw_output,
                0,
                host,
                pooled_elements,
                raw_elements,
            )?;
        }
        lease.record()?;
        if let Some(state) = work.text_use.as_mut() {
            state.phase = TextPhase::OutputCopied;
        }
        Ok(())
    }

    /// Cold Rune admission before the request lease begins or any H2D/launch
    /// is submitted. The returned prepared head owns only host metadata and
    /// host-only M=1 plans.
    pub(crate) fn moe_provider(&self) -> Option<&MoeProvider> {
        self.moe.as_ref().map(|moe| &moe.provider)
    }

    pub(crate) fn prepare_rune(
        &mut self,
        batch: &Batch,
        requests: &[DecisionInput],
    ) -> Result<PreparedGemma4OptionHead, BackendError> {
        let prepared = {
            let head = self.option_head.as_ref().ok_or_else(|| {
                infer("Gemma4 option head is not admitted for this executor contract")
            })?;
            head.prepare(batch, requests)?
        };
        // Cold host allocation: cache the exact full-T backbone plans before
        // the request lease begins so the warm enqueue does not rebuild them.
        self.prepare_plans(prepared.tokens(), None)?;
        Ok(prepared)
    }

    /// Warm Rune option-head route over the same executor, canonical tied
    /// embedding upload and request Work used by the normal text path.
    ///
    /// `prepared` must come from [`Self::prepare_rune`] before the request
    /// lease begins. This method performs no host allocation and preflights all
    /// option extents, contexts, lease identity and Work state before the first
    /// backbone launch. Caller-side `Gemma4E2bWork::upload` H2D is outside this
    /// method and must be completed before the call.
    /// The caller retains `work`, `option_work`, the executor (canonical
    /// embedding) and `prepared` until its completion event or drain, then
    /// calls `finish_rune` for the one flat D2H and `outputs` for the
    /// Native-shaped results.
    ///
    /// # Safety
    /// `hidden` is `work.normalized` produced by the same ordered stream, and
    /// every borrowed/canonical owner must remain alive and unmodified until
    /// the completion event records and drains. There is no public Engine
    /// route yet; only the crate-private caller can satisfy this contract.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn enqueue_rune(
        &mut self,
        plain_blas: Option<&Blas>,
        stream: &Stream,
        work: &mut Gemma4E2bWork,
        option_work: &mut Gemma4OptionWork,
        batch: &Batch,
        max_length: usize,
        tokens: usize,
        requests: &[DecisionInput],
        ids_lease: &mut HostIoLease,
        prepared: &PreparedGemma4OptionHead,
        vision: Option<VisionInjection<'_>>,
    ) -> Result<(), BackendError> {
        let blas = plain_blas.ok_or_else(|| infer("Gemma4 E2B requires plain cuBLAS"))?;
        if prepared.tokens() != tokens
            || batch.input_ids.len() != tokens
            || !prepared.matches_batch(batch)
            || !prepared.matches_requests(requests)
        {
            return Err(infer(
                "Gemma4 option prepared plan does not match the packed request rows",
            ));
        }
        if !work.uses_stream(stream) {
            return Err(infer("Gemma4 E2B Work belongs to a different stream"));
        }
        if !self
            .plans
            .as_ref()
            .is_some_and(|plans| plans.rows == tokens && plans.pooled_rows.is_none())
        {
            return Err(infer(
                "Gemma4 E2B backbone plans were not prepared cold for this token count",
            ));
        }
        if !option_work.fits(prepared.layout()) {
            return Err(infer(
                "Gemma4 option Work scratch does not match the request plan",
            ));
        }
        if !ids_lease.uses_stream(stream) {
            return Err(infer(
                "Gemma4 option HostIo lease belongs to a different stream",
            ));
        }
        let boundaries = batch.cumulative_seq_lengths.len();
        let (ids_offset, _) = crate::gemma4_option_head::option_id_staging_offset(
            tokens,
            boundaries,
            prepared.total_options(),
        )?;
        {
            let ids_host = &ids_lease.slot()?.input;
            if !prepared.host_ids_match(ids_host, ids_offset)? {
                return Err(infer(
                    "Gemma4 option leased host tail does not match the prepared IDs",
                ));
            }
            let kernels = OptionKernels {
                gather_bf16: &self.kernels.gather,
                widen_bf16: &self.kernels.bf16_to_f32,
                softcap_bf16_to_f32: self.kernels.option_softcap.as_ref(),
            };
            prepared.preflight(
                blas,
                stream,
                &work.normalized,
                &self.weights.embed,
                option_work,
                ids_host,
                ids_offset,
                &kernels,
            )?;
        }
        // The request lease is now retained from before the first backbone H2D
        // through its completion event/drain.
        ids_lease.begin();
        self.enqueue_impl(
            Some(blas),
            stream,
            work,
            batch,
            None,
            false,
            max_length,
            tokens,
            vision,
        )?;
        let kernels = OptionKernels {
            gather_bf16: &self.kernels.gather,
            widen_bf16: &self.kernels.bf16_to_f32,
            softcap_bf16_to_f32: self.kernels.option_softcap.as_ref(),
        };
        // SAFETY: the caller's unsafe contract covers the same-stream hidden
        // source, canonical embedding, both retained Works and the begun lease
        // through the head completion event/drain.
        unsafe {
            prepared.enqueue_from_lease(
                blas,
                stream,
                &work.normalized,
                &self.weights.embed,
                option_work,
                ids_lease,
                ids_offset,
                &kernels,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemma4_text_model::Gemma4Variant;

    fn actual_path() -> &'static str {
        "/workspace/model-performance/michaelfeil/onejev-perf/backbone-comparison-20261023/models/gemma-4-E2B"
    }

    #[test]
    fn native_reference_threads_cover_1536_256_and_512() {
        assert_eq!(reference_threads(1536), 1024);
        assert_eq!(reference_threads(256), 256);
        assert_eq!(reference_threads(512), 512);
        assert_eq!(reference_threads(100), 128);
    }

    #[test]
    fn actual_dense_e2b_loader_and_geometry_admit() {
        let weights =
            Gemma4Weights::load_embedding(Path::new(actual_path())).expect("actual E2B headers");
        let math = Gemma4E2bMath::admit(&weights).expect("dense E2B geometry");
        assert_eq!(math.hidden, 1536);
        assert_eq!(math.layers, 35);
        assert_eq!(math.ple_width, 256);
        assert_eq!(math.local_support, 513);
        assert!(!weights.plan.causal);
        assert_eq!(weights.config.num_kv_shared_layers, 20);
        assert_eq!(weights.plan.kv_producers, [Some(14), Some(13)]);
        assert_eq!(
            weights
                .plan
                .layers
                .iter()
                .filter(|layer| layer.kind == Gemma4LayerKind::Full)
                .count(),
            7
        );
        assert_eq!(
            weights
                .plan
                .layers
                .iter()
                .filter(|layer| layer.kind == Gemma4LayerKind::Sliding)
                .count(),
            28
        );
        for (index, layer) in weights.plan.layers.iter().enumerate() {
            let shared = matches!(layer.kv, Gemma4KvPlan::Shared { .. });
            assert_eq!(shared, index >= 15);
            assert_eq!(
                layer.dense_intermediate_size,
                if shared { 12_288 } else { 6_144 }
            );
        }
        assert_eq!(weights.plan.layers[0].rope.rotated_pairs, 128);
        assert_eq!(weights.plan.layers[4].rope.rotated_pairs, 64);
        for index in 15..35 {
            match &weights.layers[index].kv {
                Gemma4KvWeights::Shared(shadow) => {
                    assert!(shadow.k.is_some());
                    assert!(shadow.v.is_some());
                    assert!(shadow.k_norm.is_some());
                }
                Gemma4KvWeights::Producer(_) => panic!("shared layer {index} produced KV"),
            }
        }
        let mut first_bits = [0u16; 1];
        weights.layers[0]
            .input_norm
            .read_bf16(0, &mut first_bits)
            .expect("read first E2B norm value");
        let values = norm_values(&weights.layers[0].input_norm, 1536).expect("norm values");
        assert_eq!(values[0].to_bits(), u32::from(first_bits[0]) << 16);
    }

    struct FullSupports;

    impl FullAttentionProvider for FullSupports {
        fn supports(&self, plan: &Gemma4LayerPlan, _causal: bool) -> bool {
            plan.kind == Gemma4LayerKind::Full
        }

        unsafe fn forward_batch_bound(
            &self,
            _stream: &Stream,
            _plan: &Gemma4LayerPlan,
            _q: u64,
            _k: u64,
            _v: u64,
            _out: u64,
            _out_bytes: u64,
            _lse: u64,
            _lse_bytes: u64,
            _boundaries: u64,
            _offsets_bytes: usize,
            _tokens: usize,
            _sequences: usize,
            _max_length: usize,
            _causal: bool,
        ) -> Result<(), BackendError> {
            Err(infer("test provider does not execute"))
        }
    }

    fn native_ple_combine(projected: u16, token: u16, scale: f32) -> u16 {
        let sum = half::bf16::from_f32(
            half::bf16::from_bits(projected).to_f32() + half::bf16::from_bits(token).to_f32(),
        )
        .to_f32();
        half::bf16::from_f32(sum * scale).to_bits()
    }

    fn native_rope_pair(inverse: &[f32], position: u32, pair: usize) -> (u16, u16) {
        let phase = position as f32 * inverse[pair];
        (
            half::bf16::from_f32(phase.cos()).to_bits(),
            half::bf16::from_f32(phase.sin()).to_bits(),
        )
    }

    #[test]
    fn ple_combine_matches_native_bf16_add_then_multiply_fixtures() {
        let scale = half::bf16::from_f32(std::f64::consts::FRAC_1_SQRT_2 as f32).to_f32();
        assert_eq!((scale.to_bits() >> 16) as u16, 0x3f35);
        for (projected, token, expected) in [
            (0x3f80, 0x3b80, 0x3f35),
            (0x4000, 0x3f80, 0x4008),
            (0x3f80, 0x3f80, 0x3fb5),
            (0xbf80, 0x3b80, 0xbf34),
            (0x0000, 0x0000, 0x0000),
        ] {
            assert_eq!(
                native_ple_combine(projected, token, scale),
                expected,
                "PLE combine {projected:#06x} + {token:#06x}"
            );
        }
    }

    #[test]
    fn global_rope_padding_is_zero_tail_at_every_position() {
        let local = inverse_frequencies(10_000.0, 256, 128).expect("local inverse frequencies");
        assert_eq!(local.len(), 128);
        assert!(local.iter().all(|value| value.is_finite() && *value > 0.0));
        let global = inverse_frequencies(1_000_000.0, 512, 64).expect("global inverse frequencies");
        assert_eq!(global.len(), 256);
        assert!(global[..64]
            .iter()
            .all(|value| value.is_finite() && *value > 0.0));
        assert!(global[64..].iter().all(|value| *value == 0.0));
        for position in [0, 1, 129, 511] {
            for pair in 0..64 {
                let (cos, sin) = native_rope_pair(&global, position, pair);
                assert!(half::bf16::from_bits(cos).to_f32().is_finite());
                assert!(half::bf16::from_bits(sin).to_f32().is_finite());
            }
            for pair in 64..256 {
                let (cos, sin) = native_rope_pair(&global, position, pair);
                assert_eq!((cos, sin), (0x3f80u16, 0x0000u16));
            }
            let (cos, sin) = native_rope_pair(&local, position, 0);
            if position == 0 {
                assert_eq!((cos, sin), (0x3f80u16, 0x0000u16));
            }
        }
    }

    #[test]
    fn actual_dense_e2b_rune_causal_plan_is_admitted() {
        let weights =
            Gemma4Weights::load_rune(Path::new(actual_path())).expect("actual E2B Rune headers");
        let math = Gemma4E2bExecutor::qualify(&weights, Some(&FullSupports))
            .expect("Rune causal plan qualifies");
        assert!(math.causal);
        assert_eq!((math.local_left, math.local_right), (511, 0));
        assert_eq!(math.local_support, 512);
        assert!(weights.plan.layers[0].window_left == 511);
        assert!(weights.plan.layers[0].window_right == 0);
        assert_eq!(weights.plan.kv_producers, [Some(14), Some(13)]);
    }

    #[test]
    #[ignore = "requires INFINITY_HEAD512_ROOT, OXIDE_KERNEL_MODULE, the local E2B checkpoint, and a matching CUDA device"]
    fn actual_e2b_text_and_rune_causal_execute_without_synthetic_pooling() {
        let checkpoint =
            std::env::var("INFINITY_GEMMA4_E2B_ROOT").unwrap_or_else(|_| actual_path().to_owned());
        let provider_root = PathBuf::from(
            std::env::var("INFINITY_HEAD512_ROOT")
                .expect("set INFINITY_HEAD512_ROOT to the directory containing sm_90"),
        );
        let architecture =
            std::env::var("INFINITY_HEAD512_ARCH").unwrap_or_else(|_| "sm_90".to_owned());
        let max_optin = std::env::var("INFINITY_HEAD512_MAX_OPTIN")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(128 * 1024);
        let context = Context::new(0).expect("CUDA context");
        let stream = context.new_stream().expect("CUDA stream");
        let blas = Blas::new(context.clone(), stream.clone()).expect("plain cuBLAS");
        let attention = Arc::new(
            Head512Aot::load(&context, &provider_root, &architecture, max_optin)
                .expect("qualified Head512 ABI2 provider"),
        );
        let tokens = 3usize;
        let batch = dense_e2b_causal_batch(tokens);
        let key = (
            batch.input_ids.len(),
            batch.cumulative_seq_lengths.len(),
            batch.pooled_indices.len(),
        );
        for (contract, pool) in [
            (Gemma4Contract::Embedding, Pool::Mean),
            (Gemma4Contract::RuneOptionTokens, Pool::LastToken),
        ] {
            let weights = Gemma4Weights::load(Path::new(&checkpoint), contract)
                .expect("actual E2B checkpoint under the requested contract");
            let math = Gemma4E2bExecutor::qualify(&weights, Some(attention.as_ref()))
                .expect("dense E2B contract and provider qualify");
            assert_eq!(math.causal, contract == Gemma4Contract::RuneOptionTokens);
            let mut executor = Gemma4E2bExecutor::new_diagnostic_leaf(
                context.clone(),
                stream.clone(),
                &weights,
                Some(attention.clone()),
                None,
            )
            .expect("dense E2B executor");
            let mut work =
                Gemma4E2bWork::new(stream.clone(), &weights, key).expect("dense E2B Work arena");
            let mut host = vec![0u32; crate::staging::input_capacity(key).expect("input layout")];
            crate::pack_input(
                &mut host,
                tokens,
                &batch.input_ids,
                &batch.position_ids,
                &batch.token_type_ids,
                &batch.cumulative_seq_lengths,
                &batch.pooled_indices,
            )
            .expect("pack E2B input");
            work.upload(&host, host.len()).expect("upload E2B input");
            executor
                .enqueue(
                    Some(&blas),
                    &stream,
                    &mut work,
                    &batch,
                    &pool,
                    tokens,
                    tokens,
                )
                .expect("dense E2B warm enqueue");
            stream.synchronize().expect("dense E2B readout sync");
            let pooled = work.pooled.to_host_vec(&stream).expect("read E2B hidden");
            assert_eq!(pooled.len(), math.hidden);
            assert!(
                pooled.iter().all(|value| value.is_finite()),
                "dense E2B {contract:?} hidden must be finite"
            );
            drop(work);
            drop(executor);
            drop(weights);
        }
    }

    #[test]
    fn rope_launch_plan_covers_pairs_for_t1_t7_q8_kv1_d256_d512() {
        for tokens in [1usize, 7] {
            for dim in [256usize, 512] {
                let plan = RopeLaunchPlan::new(tokens, E2B_HEADS, E2B_KV_HEADS, dim, dim)
                    .expect("realistic E2B rope plan");
                let expected = tokens * (E2B_HEADS + E2B_KV_HEADS) * (dim / 2);
                assert_eq!(plan.total_threads(), expected);
                let covered = plan.grid_threads().expect("rope grid threads");
                assert!(covered >= expected);
                assert_eq!(
                    plan.padding_threads().expect("rope padding"),
                    covered - expected
                );
                assert!(covered - expected < ROPE_BLOCK);
            }
        }
    }

    #[test]
    fn rope_launch_plan_covers_q_only_shared_layers_and_padding() {
        for tokens in [1usize, 7] {
            for dim in [256usize, 512] {
                let plan = RopeLaunchPlan::new(tokens, E2B_HEADS, 0, dim, dim)
                    .expect("shared Q-only rope plan");
                let expected = tokens * E2B_HEADS * (dim / 2);
                assert_eq!(plan.total_threads(), expected);
                let covered = plan.grid_threads().expect("Q-only grid threads");
                assert!(covered >= expected);
                assert_eq!(
                    plan.padding_threads().expect("Q-only padding"),
                    covered - expected
                );
                assert!(covered - expected < ROPE_BLOCK);
            }
        }
    }

    #[test]
    fn e2b_mean_plan_h1536_duplicates_and_ragged_selections() {
        let boundaries = [0, 1, 5, 17, 60];
        let selected = [2, 0, 2, 3, 1, 2];
        let plan = MeanPlan::new(60, E2B_HIDDEN, &boundaries, &selected).expect("H1536 mean plan");
        assert_eq!(plan.count(), 6);
        assert_eq!(plan.width(), E2B_HIDDEN as u32);
        assert_eq!(plan.threads(), 64);
        let geometry = plan.geometry().expect("mean grid");
        assert_eq!(geometry.grid(), [6, 192, 1]);
        assert_eq!(geometry.block(), [64, 1, 1]);
        assert_eq!(plan.hidden_layout().shape(), &[60, E2B_HIDDEN]);
        assert_eq!(plan.output_layout().shape(), &[6, E2B_HIDDEN]);
    }

    #[test]
    fn e2b_mean_plan_h1536_short_spans_keep_min8_lanes() {
        for longest in 1..=4 {
            let boundaries = [0, longest as u32];
            let plan = MeanPlan::new(longest, E2B_HIDDEN, &boundaries, &[0])
                .expect("H1536 short-span mean plan");
            let geometry = plan.geometry().expect("short-span grid");
            assert_eq!(plan.threads(), 8);
            assert_eq!(geometry.block(), [8, 1, 1]);
            assert_eq!(geometry.grid(), [1, 192, 1]);
        }
    }

    #[test]
    fn e2b_mean_plan_h1536_long_span_clamps_to_1024() {
        let plan =
            MeanPlan::new(4096, E2B_HIDDEN, &[0, 4096], &[0]).expect("H1536 long-span mean plan");
        assert_eq!(plan.threads(), 1024);
        assert_eq!(
            plan.geometry().expect("long-span grid").block(),
            [1024, 1, 1]
        );
    }

    #[test]
    fn serving_requires_a_qualified_fa4_local_profile() {
        let weights =
            Gemma4Weights::load_embedding(Path::new(actual_path())).expect("actual E2B headers");
        assert_eq!(
            Gemma4E2bExecutor::qualify_serving(&weights, Some(&FullSupports), None).unwrap_err(),
            Gemma4E2bUnsupported::LocalAttentionPending
        );
    }

    #[test]
    fn plans_arena_and_provider_qualification_are_ready_without_gpu_work() {
        let weights =
            Gemma4Weights::load_embedding(Path::new(actual_path())).expect("actual E2B headers");
        let plans = Plans::new(&weights.config, &weights.plan, 7, None).expect("host plans");
        assert_eq!(plans.layers.len(), 35);
        assert_eq!(plans.rows, 7);
        let bytes = Gemma4E2bWork::allocation_bytes((7, 1, 1), &weights, 0).expect("arena bytes");
        assert!(bytes > 7 * 1536 * 2 * 4);
        let math = Gemma4E2bExecutor::qualify(&weights, Some(&FullSupports))
            .expect("provider qualification");
        assert_eq!(math.ple_token_scale, 16.0);
    }

    #[test]
    fn reports_strong_pending_errors_instead_of_generic_success() {
        let mut weights =
            Gemma4Weights::load_embedding(Path::new(actual_path())).expect("actual E2B headers");
        assert_eq!(
            Gemma4E2bExecutor::qualify(&weights, None).unwrap_err(),
            Gemma4E2bUnsupported::ProviderPending
        );
        weights.plan.variant = Gemma4Variant::SparseMoe;
        assert!(matches!(
            Gemma4E2bMath::admit(&weights),
            Err(Gemma4E2bUnsupported::Geometry(_))
        ));
        let mut weights =
            Gemma4Weights::load_embedding(Path::new(actual_path())).expect("actual E2B headers");
        weights.plan.causal = true;
        assert!(matches!(
            Gemma4E2bMath::admit(&weights),
            Err(Gemma4E2bUnsupported::Geometry(_))
        ));
        let mut weights =
            Gemma4Weights::load_embedding(Path::new(actual_path())).expect("actual E2B headers");
        weights.config.hidden_size = 2816;
        assert!(matches!(
            Gemma4E2bMath::admit(&weights),
            Err(Gemma4E2bUnsupported::Geometry(_))
        ));
    }

    fn case_u32_list(value: &serde_json::Value, field: &str) -> Vec<u32> {
        value[field]
            .as_array()
            .expect("case uint32 list")
            .iter()
            .map(|entry| {
                u32::try_from(entry.as_u64().expect("case uint32")).expect("case uint32 range")
            })
            .collect()
    }

    fn rune_case_batch(case: &serde_json::Value) -> Batch {
        let raw = &case["batch"];
        Batch {
            multimodal: Vec::new(),
            input_ids: case_u32_list(raw, "input_ids"),
            token_type_ids: case_u32_list(raw, "token_type_ids"),
            position_ids: case_u32_list(raw, "position_ids"),
            cumulative_seq_lengths: case_u32_list(raw, "cumulative_seq_lengths"),
            max_length: u32::try_from(raw["max_length"].as_u64().expect("max_length"))
                .expect("max_length range"),
            pooled_indices: case_u32_list(raw, "pooled_indices"),
            raw_indices: case_u32_list(raw, "raw_indices"),
            compact_input_ids: None,
            compact_position_ids: None,
            scatter_unfold: None,
            fold_gather: None,
            tokens: Vec::new(),
            offsets: Vec::new(),
        }
    }

    fn rune_case_requests(case: &serde_json::Value) -> Vec<DecisionInput> {
        case["option"]["token_ids"]
            .as_array()
            .expect("option token id lists")
            .iter()
            .map(|ids| DecisionInput::OptionTokens {
                token_ids: ids
                    .as_array()
                    .expect("option token ids")
                    .iter()
                    .map(|id| {
                        u32::try_from(id.as_u64().expect("option token id"))
                            .expect("option token id range")
                    })
                    .collect(),
            })
            .collect()
    }

    #[test]
    #[ignore = "requires the local E2B checkpoint, canonical FA4 root, Head512 root, oxide kernel module with the additive softcap export, and CUDA"]
    fn actual_rune_whole_json_cases_publish_typed_logits_for_root_comparison() {
        use sha2::{Digest, Sha256};

        let cases_path = std::env::var("INFINITY_RUNE_CASES").unwrap_or_else(|_| {
            "/tmp/infinity-oxide-parity-20261101/gemma4-e2b-native-capi-control-v4/cases.json"
                .to_owned()
        });
        let checkpoint =
            std::env::var("INFINITY_GEMMA4_E2B_ROOT").unwrap_or_else(|_| actual_path().to_owned());
        let head512_root = std::env::var("INFINITY_HEAD512_ROOT")
            .expect("set INFINITY_HEAD512_ROOT to the directory containing sm_90");
        let head512_arch =
            std::env::var("INFINITY_HEAD512_ARCH").unwrap_or_else(|_| "sm_90".to_owned());
        let max_optin = std::env::var("INFINITY_HEAD512_MAX_OPTIN")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(128 * 1024);
        let fa4_root = PathBuf::from(
            std::env::var("INFINITY_FA4_ROOT")
                .expect("set INFINITY_FA4_ROOT to the canonical FA4 AOT root"),
        );
        let cases: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&cases_path).expect("read Rune source cases"))
                .expect("parse Rune source cases");
        assert_eq!(
            cases["schema"], "gemma4-e2b-native-capi-cases/v1",
            "unexpected source case schema"
        );
        let weights =
            Gemma4Weights::load_rune(Path::new(&checkpoint)).expect("actual E2B Rune headers");
        let context = Context::new(0).expect("CUDA context");
        let stream = context.new_stream().expect("CUDA stream");
        let blas = Blas::new(context.clone(), stream.clone()).expect("plain cuBLAS");
        let attention = Arc::new(
            Head512Aot::load(&context, Path::new(&head512_root), &head512_arch, max_optin)
                .expect("qualified Head512 ABI2 provider"),
        );
        let mut executor = Gemma4E2bExecutor::new_with_fa4_local(
            context.clone(),
            stream.clone(),
            &weights,
            Some(attention),
            &fa4_root,
            None,
        )
        .expect("canonical FA4 local Rune executor");
        let gpu = context.device_name().expect("CUDA device name");
        let (major, minor) = context.compute_capability().expect("compute capability");
        let provenance = serde_json::json!({
            "provider": "canonical_fa4_local+head512_abi2",
            "diagnostic_leaf": false,
            "gpu": gpu,
            "compute_capability": format!("sm_{major}{minor}"),
        });
        for case in cases["cases"].as_array().expect("source cases") {
            let batch = rune_case_batch(case);
            let requests = rune_case_requests(case);
            let tokens = batch.input_ids.len();
            let expected_counts = case["option"]["descriptor_count"]
                .as_u64()
                .expect("descriptor_count") as usize;
            let expected_logits =
                case["option"]["logit_count"].as_u64().expect("logit_count") as usize;
            let expected_last = case_u32_list(&case["option"], "last_token_indices");
            let prepared = executor
                .prepare_rune(&batch, &requests)
                .expect("cold Rune option plan");
            assert_eq!(prepared.counts().len(), expected_counts);
            assert_eq!(prepared.total_options(), expected_logits);
            assert_eq!(prepared.last_rows(), expected_last.as_slice());
            let boundaries = batch.cumulative_seq_lengths.len();
            let gpu_key = (tokens, boundaries, 0usize);
            let host_key = (tokens, boundaries, prepared.total_options());
            let mut work =
                Gemma4E2bWork::new(stream.clone(), &weights, gpu_key).expect("E2B Work arena");
            let mut host =
                vec![0u32; crate::staging::input_capacity(gpu_key).expect("Rune input layout")];
            crate::pack_input(
                &mut host,
                tokens,
                &batch.input_ids,
                &batch.position_ids,
                &batch.token_type_ids,
                &batch.cumulative_seq_lengths,
                &[],
            )
            .expect("pack Rune source batch");
            work.upload(&host, host.len())
                .expect("upload Rune source batch");
            let mut option_work =
                Gemma4OptionWork::new(stream.clone(), prepared.layout()).expect("Rune option Work");
            let mut flat_option_ids = Vec::new();
            for request in &requests {
                match request {
                    DecisionInput::OptionTokens { token_ids } => {
                        flat_option_ids.extend_from_slice(token_ids)
                    }
                    DecisionInput::Warmup => flat_option_ids.push(0),
                    _ => panic!("unexpected Rune request"),
                }
            }
            let pool = crate::staging::HostIo::pool(&context).expect("HostIo pool");
            let slot = pool.acquire().expect("Rune lease slot");
            let mut lease = HostIoLease::detached(slot, pool.clone(), stream.clone());
            lease
                .prepare(host_key, prepared.total_options(), false)
                .expect("prepare Rune lease");
            crate::gemma4_option_head::stage_option_ids(&mut lease, host_key, &flat_option_ids)
                .expect("stage Rune option ids");
            // SAFETY: this test owns the executor's canonical embedding, both
            // Works, the prepared head, the retained lease and the ordered
            // stream until the final synchronize below. Canonical FA4 is
            // selected above.
            unsafe {
                executor
                    .enqueue_rune(
                        Some(&blas),
                        &stream,
                        &mut work,
                        &mut option_work,
                        &batch,
                        batch.max_length as usize,
                        tokens,
                        &requests,
                        &mut lease,
                        &prepared,
                        None,
                    )
                    .expect("whole Rune enqueue");
            }
            prepared
                .finish_rune(&stream, &mut option_work, &mut lease)
                .expect("whole Rune D2H event");
            lease.wait().expect("whole Rune lease completion");
            let flat = lease
                .output(prepared.total_options())
                .expect("read flat Rune logits")
                .to_vec();
            let outputs = crate::gemma4_option_head::pack_option_outputs(&flat, prepared.counts())
                .expect("typed Rune outputs");
            assert_eq!(
                outputs
                    .iter()
                    .map(|output| output.logits.len())
                    .sum::<usize>(),
                expected_logits
            );
            assert!(outputs
                .iter()
                .all(|output| output.action_probability == 1.0));
            assert!(
                flat.iter().all(|value| value.is_finite()),
                "whole Rune logits must be finite"
            );
            let mut digest = Sha256::new();
            for value in &flat {
                digest.update(value.to_bits().to_le_bytes());
            }
            let typed_outputs: Vec<serde_json::Value> = outputs
                .iter()
                .enumerate()
                .map(|(descriptor_index, output)| {
                    serde_json::json!({
                        "descriptor_index": descriptor_index,
                        "logits": output.logits,
                        "action_probability": output.action_probability,
                    })
                })
                .collect();
            let option_token_ids: Vec<Vec<u32>> = requests
                .iter()
                .map(|request| match request {
                    DecisionInput::OptionTokens { token_ids } => token_ids.clone(),
                    DecisionInput::Warmup => vec![0],
                    _ => Vec::new(),
                })
                .collect();
            let source_sha = format!(
                "{:x}",
                Sha256::digest(std::fs::read(&cases_path).expect("read source cases"))
            );
            let report = serde_json::json!({
                "schema": "gemma4-rune-whole-candidate/v1",
                "status": "passed",
                "gpu_executed": true,
                "model": "gemma-4-E2B",
                "provider": "canonical_fa4_local+head512_abi2",
                "source_sha256": source_sha,
                "case": case["id"],
                "physical_batch": case["batch"],
                "batch_sha256": case["batch_sha256"],
                "causal": true,
                "final_logit_softcapping": 30.0,
                "tied_word_embeddings": true,
                "option_token_ids": option_token_ids,
                "outputs": typed_outputs,
                "last_rows": prepared.last_rows(),
                "logit_sha256": format!("{:x}", digest.finalize()),
                "provenance": provenance,
            });
            let serialized =
                serde_json::to_string(&report).expect("serialize whole Rune candidate");
            if let Some(root) = std::env::var_os("INFINITY_GEMMA4_RUNE_OUTPUT_DIR") {
                let root = PathBuf::from(root);
                std::fs::create_dir_all(&root).expect("create Rune output directory");
                std::fs::write(
                    root.join(format!("{}.json", case["id"].as_str().expect("case id"))),
                    &serialized,
                )
                .expect("write Rune candidate JSON");
            }
            println!("{serialized}");
        }
    }

    #[test]
    #[ignore = "requires the frozen old97 cubin, the new98 cubin with the additive softcap export, and CUDA; no checkpoint weights"]
    fn actual_old97_missing_softcap_fails_and_new98_loads() {
        let old97 = PathBuf::from(
            std::env::var("INFINITY_GEMMA4_OLD97_KERNEL_MODULE")
                .expect("set INFINITY_GEMMA4_OLD97_KERNEL_MODULE to the frozen old97 cubin"),
        );
        let new98 = PathBuf::from(
            std::env::var("INFINITY_GEMMA4_NEW98_KERNEL_MODULE")
                .expect("set INFINITY_GEMMA4_NEW98_KERNEL_MODULE to the new98 cubin"),
        );
        let context = Context::new(0).expect("CUDA context");
        context.bind_to_thread().expect("bind context");
        // The ordinary embedding/text path must still load the old97 module.
        Gemma4Kernels::load(&context, Some(old97.as_path()), false)
            .expect("old97 ordinary text exports");
        // The capped Rune head requires the additive symbol and must fail cold.
        let missing = Gemma4Kernels::load(&context, Some(old97.as_path()), true);
        let detail = missing
            .as_ref()
            .err()
            .map(ToString::to_string)
            .unwrap_or_default();
        assert!(
            matches!(&missing, Err(BackendError::Start(message)) if message.contains("option_softcap")),
            "old97 missing-export failure must be the typed Rune softcap startup error: {detail}"
        );
        // The new98 leaf must load the required symbol.
        Gemma4Kernels::load(&context, Some(new98.as_path()), true)
            .expect("new98 Rune softcap export");
    }

    fn control_u32_list(value: &serde_json::Value, field: &str) -> Vec<u32> {
        value[field]
            .as_array()
            .expect("control uint32 list")
            .iter()
            .map(|entry| u32::try_from(entry.as_u64().expect("control uint32")).expect("u32 range"))
            .collect()
    }

    fn dense_e2b_causal_batch(tokens: usize) -> Batch {
        Batch {
            multimodal: Vec::new(),
            input_ids: (1..=tokens as u32).collect(),
            token_type_ids: Vec::new(),
            position_ids: (0..tokens as u32).collect(),
            cumulative_seq_lengths: vec![0, tokens as u32],
            max_length: tokens as u32,
            pooled_indices: vec![0],
            raw_indices: Vec::new(),
            compact_input_ids: None,
            compact_position_ids: None,
            scatter_unfold: None,
            fold_gather: None,
            tokens: Vec::new(),
            offsets: vec![(0, tokens)],
        }
    }

    fn control_batch(case: &serde_json::Value) -> Batch {
        let raw = &case["batch"];
        Batch {
            multimodal: Vec::new(),
            input_ids: control_u32_list(raw, "input_ids"),
            token_type_ids: control_u32_list(raw, "token_type_ids"),
            position_ids: control_u32_list(raw, "position_ids"),
            cumulative_seq_lengths: control_u32_list(raw, "cumulative_seq_lengths"),
            max_length: u32::try_from(raw["max_length"].as_u64().expect("max_length"))
                .expect("max_length range"),
            pooled_indices: control_u32_list(raw, "pooled_indices"),
            raw_indices: control_u32_list(raw, "raw_indices"),
            compact_input_ids: None,
            compact_position_ids: None,
            scatter_unfold: None,
            fold_gather: None,
            tokens: Vec::new(),
            offsets: Vec::new(),
        }
    }

    fn control_rune_requests(case: &serde_json::Value) -> Vec<DecisionInput> {
        case["option"]["token_ids"]
            .as_array()
            .expect("option token ids")
            .iter()
            .map(|ids| DecisionInput::OptionTokens {
                token_ids: ids
                    .as_array()
                    .expect("option ids")
                    .iter()
                    .map(|value| {
                        u32::try_from(value.as_u64().expect("option id")).expect("u32 range")
                    })
                    .collect(),
            })
            .collect()
    }

    fn control_cases_path() -> PathBuf {
        std::env::var("INFINITY_GEMMA4_CONTROL_CASES")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(
                    "/tmp/infinity-oxide-parity-20261101/gemma4-e2b-native-capi-control-v4/cases.json",
                )
            })
    }

    #[test]
    fn review_v8_dense_e2b_causal_key_matches_packed_input_bound() {
        let tokens = 3usize;
        let batch = dense_e2b_causal_batch(tokens);
        let key = (
            batch.input_ids.len(),
            batch.cumulative_seq_lengths.len(),
            batch.pooled_indices.len(),
        );
        let capacity = crate::staging::input_capacity(key).expect("input capacity");
        assert_eq!(
            capacity,
            tokens * 3 + batch.cumulative_seq_lengths.len() + batch.pooled_indices.len()
        );
        assert_ne!(
            crate::staging::input_capacity((tokens, 1, 1)).expect("legacy input capacity"),
            capacity,
            "key.1 must be the B+1 boundary count, not the sequence count"
        );
        let mut host = vec![0u32; capacity];
        crate::pack_input(
            &mut host,
            tokens,
            &batch.input_ids,
            &batch.position_ids,
            &batch.token_type_ids,
            &batch.cumulative_seq_lengths,
            &batch.pooled_indices,
        )
        .expect("packed E2B input fits the Work key capacity");
    }

    #[test]
    fn cpu_fullmodel_source_parser_builds_nine_text_and_three_rune_cases() {
        use sha2::Digest;
        let explicitly_set = std::env::var_os("INFINITY_GEMMA4_CONTROL_CASES").is_some();
        let path = control_cases_path();
        let Ok(source) = std::fs::read(&path) else {
            if explicitly_set {
                panic!(
                    "explicit control cases path is unreadable: {}",
                    path.display()
                );
            }
            return;
        };
        let doc: serde_json::Value = serde_json::from_slice(&source).expect("control cases JSON");
        assert_eq!(doc["schema"], "gemma4-e2b-native-capi-cases/v1");
        assert_eq!(doc["source_kind"], "bounded-exact-id-synthetic");
        assert_eq!(doc["tokenizer_used"], false);
        let cases = doc["cases"].as_array().expect("control cases");
        assert_eq!(cases.len(), 3);
        let mut text = Vec::new();
        let mut rune = Vec::new();
        for case in cases {
            let batch = control_batch(case);
            assert_eq!(batch.input_ids.len(), batch.position_ids.len());
            assert_eq!(batch.input_ids.len(), batch.token_type_ids.len());
            assert_eq!(batch.cumulative_seq_lengths.first().copied(), Some(0));
            assert_eq!(
                batch.cumulative_seq_lengths.last().copied(),
                Some(batch.input_ids.len() as u32)
            );
            assert!(batch
                .cumulative_seq_lengths
                .windows(2)
                .all(|pair| pair[1] > pair[0]));
            let raw_rows = crate::output::raw_rows(&batch).expect("raw rows");
            for pool in [Pool::Cls, Pool::Mean, Pool::LastToken] {
                let geometry = crate::output::OutputGeometry::new(E2B_HIDDEN, E2B_HIDDEN)
                    .expect("text geometry");
                let host_elements = geometry
                    .host_elements(&batch)
                    .expect("text output elements");
                assert_eq!(
                    host_elements,
                    (batch.pooled_indices.len() + raw_rows) * E2B_HIDDEN
                );
                text.push(serde_json::json!({
                    "case": case["id"],
                    "pool": pool.to_string(),
                    "input_ids_len": batch.input_ids.len(),
                    "token_type_ids_len": batch.token_type_ids.len(),
                    "position_ids_len": batch.position_ids.len(),
                    "cumulative_seq_lengths": batch.cumulative_seq_lengths,
                    "max_length": batch.max_length,
                    "pooled_indices": batch.pooled_indices,
                    "raw_indices": batch.raw_indices,
                    "raw_rows": raw_rows,
                    "pooled_width": geometry.pooled_width,
                    "raw_width": geometry.raw_width,
                    "host_elements": host_elements,
                }));
            }
            let requests = control_rune_requests(case);
            let boundaries = batch.cumulative_seq_lengths.clone();
            let plan = crate::gemma4_option_head::plan_option_requests(
                262_144,
                E2B_HIDDEN,
                &boundaries,
                batch.input_ids.len(),
                &requests,
            )
            .expect("Rune source plan");
            assert_eq!(
                plan.counts().len(),
                case["option"]["descriptor_count"]
                    .as_u64()
                    .expect("descriptor_count") as usize
            );
            assert_eq!(
                plan.total_options(),
                case["option"]["logit_count"].as_u64().expect("logit_count") as usize
            );
            assert_eq!(
                plan.last_rows(),
                case["option"]["last_token_indices"]
                    .as_array()
                    .expect("last token indices")
                    .iter()
                    .map(|value| {
                        u32::try_from(value.as_u64().expect("last index")).expect("u32 range")
                    })
                    .collect::<Vec<_>>()
                    .as_slice()
            );
            rune.push(serde_json::json!({
                "case": case["id"],
                "option_token_ids": case["option"]["token_ids"],
                "counts": plan.counts(),
                "last_rows": plan.last_rows(),
                "total_options": plan.total_options(),
            }));
        }
        assert_eq!(text.len(), 9);
        assert_eq!(rune.len(), 3);
        let report = serde_json::json!({
            "schema": "oxide-gemma4-fullmodel-source-cpu/v1",
            "status": "passed",
            "source_schema": doc["schema"],
            "source_path": path.display().to_string(),
            "source_sha256": format!("{:x}", sha2::Sha256::digest(&source)),
            "text_cases": text,
            "rune_cases": rune,
            "native_layout": "pooled selections first, then ordered raw token spans, both H1536",
            "typed_map": {"text": 9, "rune": 3},
        });
        if let Ok(output) = std::env::var("INFINITY_GEMMA4_FULLMODEL_CPU_OUTPUT") {
            let output = PathBuf::from(output);
            std::fs::create_dir_all(&output).expect("create fullmodel CPU output");
            std::fs::write(
                output.join("fullmodel-source-cpu.json"),
                serde_json::to_vec_pretty(&report).expect("serialize fullmodel CPU report"),
            )
            .expect("write fullmodel CPU report");
        }
        println!(
            "{}",
            serde_json::to_string(&report).expect("serialize fullmodel CPU report")
        );
    }

    #[test]
    fn review_f0_text_capacity_rejects_smaller_work_before_staging() {
        assert!(text_shape_fits(7, 2, 1, 7, 7, 2, 1, 7));
        assert!(!text_shape_fits(8, 2, 1, 7, 7, 2, 1, 7));
        assert!(!text_shape_fits(7, 2, 1, 8, 7, 2, 1, 7));
        assert!(!text_shape_fits(7, 2, 2, 7, 7, 2, 1, 7));
        assert!(!text_shape_fits(7, 3, 1, 7, 7, 2, 1, 7));
        // Boundary count, not sequence count, is the key.1 capacity.
        assert!(!text_shape_fits(7, 2, 1, 7, 7, 1, 1, 7));
    }

    #[test]
    fn review_v8_stage_text_begins_checked_retention_before_prefix_upload() {
        let source = include_str!("gemma4_e2b_model.rs");
        let stage_start = source
            .find("pub(crate) fn stage_text(")
            .expect("stage_text source");
        let stage_end = source[stage_start..]
            .find("/// Upload only the pooled prefix")
            .map(|offset| stage_start + offset)
            .expect("stage_text end");
        let stage = &source[stage_start..stage_end];
        let retention = stage
            .find("lease.begin_checked()?")
            .expect("checked retention begins in stage_text");
        let text_use = stage
            .find("self.begin_text_use(lease)?")
            .expect("text use begins in stage_text");
        assert!(retention < text_use);
        assert!(!stage.contains("upload_input"));
        let upload = source
            .find("pub(crate) fn upload_text_prefix(")
            .expect("prefix upload source");
        assert!(stage_start + retention < upload);
    }

    #[test]
    fn review_f5_p0_mixed_and_duplicate_raw_counts_use_shared_pack_contract() {
        let boundaries = vec![0u32, 7];
        let make = |pooled: Vec<u32>, raw: Vec<u32>| Batch {
            multimodal: Vec::new(),
            input_ids: vec![1; 7],
            token_type_ids: vec![0; 7],
            position_ids: (0..7).collect(),
            cumulative_seq_lengths: boundaries.clone(),
            max_length: 7,
            pooled_indices: pooled,
            raw_indices: raw,
            compact_input_ids: None,
            compact_position_ids: None,
            scatter_unfold: None,
            fold_gather: None,
            tokens: Vec::new(),
            offsets: Vec::new(),
        };
        let raw_span = |base: f32| {
            (0..7)
                .flat_map(|row| {
                    (0..E2B_HIDDEN).map(move |column| base + (row * E2B_HIDDEN + column) as f32)
                })
                .collect::<Vec<_>>()
        };
        let p0 = make(Vec::new(), vec![0]);
        assert_eq!(crate::output::raw_rows(&p0).unwrap(), 7);
        let p0_values = raw_span(10.0);
        let p0_map = crate::output::pack(
            &p0_values,
            E2B_HIDDEN,
            E2B_HIDDEN,
            &p0.pooled_indices,
            &p0.raw_indices,
            &p0.cumulative_seq_lengths,
        )
        .unwrap();
        match p0_map.get(&0).expect("P0 raw row") {
            infinity_v3_backend_core::Embedding::All(rows) => {
                assert_eq!(rows.len(), 7);
                assert_eq!(rows[0][0], 10.0);
            }
            _ => panic!("P0 must produce raw token rows"),
        }
        let mixed = make(vec![0], vec![0]);
        let mixed_values: Vec<f32> = (0..(1 + 7) * E2B_HIDDEN)
            .map(|index| {
                if index < E2B_HIDDEN {
                    -1.0
                } else {
                    20.0 + index as f32
                }
            })
            .collect();
        let mixed_map = crate::output::pack(
            &mixed_values,
            E2B_HIDDEN,
            E2B_HIDDEN,
            &mixed.pooled_indices,
            &mixed.raw_indices,
            &mixed.cumulative_seq_lengths,
        )
        .unwrap();
        match mixed_map.get(&0).expect("mixed raw override") {
            infinity_v3_backend_core::Embedding::All(rows) => {
                assert_eq!(rows.len(), 7);
                assert_eq!(rows[0][0], 20.0 + E2B_HIDDEN as f32);
            }
            _ => panic!("raw must override pooled for the same index"),
        }
        let duplicate = make(vec![0], vec![0, 0]);
        assert_eq!(crate::output::raw_rows(&duplicate).unwrap(), 14);
        let duplicate_values: Vec<f32> = (0..(1 + 14) * E2B_HIDDEN)
            .map(|index| index as f32)
            .collect();
        let duplicate_map = crate::output::pack(
            &duplicate_values,
            E2B_HIDDEN,
            E2B_HIDDEN,
            &duplicate.pooled_indices,
            &duplicate.raw_indices,
            &duplicate.cumulative_seq_lengths,
        )
        .unwrap();
        match duplicate_map.get(&0).expect("duplicate raw last wins") {
            infinity_v3_backend_core::Embedding::All(rows) => {
                assert_eq!(rows.len(), 7);
                assert_eq!(rows[0][0], ((1 + 7) * E2B_HIDDEN) as f32);
            }
            _ => panic!("duplicate raw must produce last-wins token rows"),
        }
    }

    #[test]
    #[ignore = "requires the local E2B checkpoint, canonical FA4, Head512, oxide kernel module, the 3 source fixtures and CUDA"]
    fn actual_whole_text_nine_cases_with_raw_output_are_ready() {
        let cases_path = control_cases_path();
        let cases: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&cases_path).expect("read control cases"))
                .expect("parse control cases");
        assert_eq!(cases["schema"], "gemma4-e2b-native-capi-cases/v1");
        let checkpoint = std::env::var("INFINITY_GEMMA4_E2B_ROOT").unwrap_or_else(|_| {
            "/workspace/model-performance/michaelfeil/onejev-perf/backbone-comparison-20261023/models/gemma-4-E2B".to_owned()
        });
        let head512_root = PathBuf::from(
            std::env::var("INFINITY_HEAD512_ROOT").expect("set INFINITY_HEAD512_ROOT"),
        );
        let head512_arch =
            std::env::var("INFINITY_HEAD512_ARCH").unwrap_or_else(|_| "sm_90".to_owned());
        let max_optin = std::env::var("INFINITY_HEAD512_MAX_OPTIN")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(128 * 1024);
        let fa4_root =
            PathBuf::from(std::env::var("INFINITY_FA4_ROOT").expect("set INFINITY_FA4_ROOT"));
        let weights =
            Gemma4Weights::load_embedding(Path::new(&checkpoint)).expect("E2B text weights");
        let context = Context::new(0).expect("CUDA context");
        let stream = context.new_stream().expect("CUDA stream");
        let blas = Blas::new(context.clone(), stream.clone()).expect("plain cuBLAS");
        let attention = Arc::new(
            Head512Aot::load(&context, &head512_root, &head512_arch, max_optin)
                .expect("Head512 provider"),
        );
        let mut executor = Gemma4E2bExecutor::new_with_fa4_local(
            context.clone(),
            stream.clone(),
            &weights,
            Some(attention),
            &fa4_root,
            None,
        )
        .expect("canonical FA4 text executor");
        let mut checked = 0usize;
        let text_output_root = PathBuf::from(
            std::env::var_os("INFINITY_GEMMA4_TEXT_OUTPUT_DIR")
                .expect("Root text gate requires INFINITY_GEMMA4_TEXT_OUTPUT_DIR"),
        );
        for case in cases["cases"].as_array().expect("cases") {
            let batch = control_batch(case);
            let tokens = batch.input_ids.len();
            let boundaries = batch.cumulative_seq_lengths.len();
            let raw_rows = crate::output::raw_rows(&batch).expect("raw rows");
            let selections = batch.pooled_indices.len();
            for (pool_name, pool_value) in [
                ("cls", Pool::Cls),
                ("mean", Pool::Mean),
                ("last-token", Pool::LastToken),
            ] {
                let gpu_key = (tokens, boundaries, selections);
                let host_key = (tokens, boundaries, selections + raw_rows);
                let mut work =
                    Gemma4E2bWork::new_with_raw(stream.clone(), &weights, gpu_key, raw_rows)
                        .expect("text Work with raw arena");
                let pool = crate::staging::HostIo::pool(&context).expect("HostIo pool");
                let slot = pool.acquire().expect("text lease slot");
                let mut lease = HostIoLease::detached(slot, pool.clone(), stream.clone());
                let elements = (selections + raw_rows) * E2B_HIDDEN;
                lease
                    .prepare(host_key, elements, false)
                    .expect("prepare text lease");
                let geometry = crate::output::OutputGeometry::new(E2B_HIDDEN, E2B_HIDDEN)
                    .expect("text geometry");
                let (prefix, raw_tail, staged_elements) = work
                    .stage_text(&batch, geometry, &mut lease)
                    .expect("stage text lease");
                assert_eq!(raw_tail, prefix);
                assert_eq!(staged_elements, elements);
                work.upload_text_prefix(&lease, prefix)
                    .expect("upload pooled prefix");
                executor
                    .enqueue_text(
                        Some(&blas),
                        &stream,
                        &mut work,
                        &batch,
                        &pool_value,
                        batch.max_length as usize,
                        tokens,
                        &lease,
                    )
                    .expect("text enqueue");
                executor
                    .finish_text(&stream, &mut work, &batch, selections, raw_rows, &mut lease)
                    .expect("text finish");
                lease.wait().expect("text completion");
                let output = lease.output(elements).expect("text output");
                assert_eq!(output.len(), elements);
                assert!(output.iter().all(|value| value.is_finite()));
                {
                    let root = &text_output_root;
                    use sha2::Digest;
                    let directory = root
                        .join(case["id"].as_str().expect("case id"))
                        .join(pool_name);
                    std::fs::create_dir_all(&directory).expect("create text output directory");
                    let bytes: Vec<u8> = output
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect();
                    std::fs::write(directory.join("output.f32le"), &bytes)
                        .expect("write text physical output");
                    let mut segments = Vec::new();
                    for (row, &sequence) in batch.pooled_indices.iter().enumerate() {
                        segments.push(serde_json::json!({
                            "kind": "pooled",
                            "output_row": row,
                            "sequence": sequence,
                            "offset": row * E2B_HIDDEN,
                            "width": E2B_HIDDEN,
                        }));
                    }
                    let mut raw_row = 0usize;
                    for &sequence in &batch.raw_indices {
                        let start = batch.cumulative_seq_lengths[sequence as usize] as usize;
                        let end = batch.cumulative_seq_lengths[sequence as usize + 1] as usize;
                        for token_row in start..end {
                            segments.push(serde_json::json!({
                                "kind": "raw",
                                "output_row": selections + raw_row,
                                "sequence": sequence,
                                "token_row": token_row,
                                "offset": (selections + raw_row) * E2B_HIDDEN,
                                "width": E2B_HIDDEN,
                            }));
                            raw_row += 1;
                        }
                    }
                    assert_eq!(raw_row, raw_rows);
                    let physical_batch = serde_json::json!({
                        "input_ids": case["batch"]["input_ids"],
                        "token_type_ids": case["batch"]["token_type_ids"],
                        "position_ids": case["batch"]["position_ids"],
                        "cumulative_seq_lengths": case["batch"]["cumulative_seq_lengths"],
                        "max_length": case["batch"]["max_length"],
                        "pooled_indices": case["batch"]["pooled_indices"],
                        "raw_indices": case["batch"]["raw_indices"],
                    });
                    let metadata = serde_json::json!({
                        "schema": "gemma4-text-whole-candidate/v1",
                        "status": "passed",
                        "gpu_executed": true,
                        "case": case["id"],
                        "pool": pool_name,
                        "causal": false,
                        "physical_batch": physical_batch,
                        "batch_sha256": case["batch_sha256"],
                        "width": E2B_HIDDEN,
                        "pooled_rows": selections,
                        "raw_rows": raw_rows,
                        "physical_output_float_count": elements,
                        "ordered_segment_descriptors": segments,
                        "output_sha256": format!("{:x}", sha2::Sha256::digest(&bytes)),
                        "littleendian": true,
                        "no_normalization": true,
                    });
                    std::fs::write(
                        directory.join("metadata.json"),
                        serde_json::to_vec_pretty(&metadata).expect("serialize text metadata"),
                    )
                    .expect("write text metadata");
                }
                checked += 1;
            }
        }
        assert_eq!(checked, 9);
    }

    #[test]
    #[ignore = "requires the local E2B checkpoint, canonical FA4, Head512, oxide kernel module, source fixtures and CUDA"]
    fn actual_text_two_hostio_leases_share_work_with_distinct_raw_tails() {
        let cases: serde_json::Value = serde_json::from_slice(
            &std::fs::read(control_cases_path()).expect("read control cases"),
        )
        .expect("parse control cases");
        let checkpoint = std::env::var("INFINITY_GEMMA4_E2B_ROOT").unwrap_or_else(|_| {
            "/workspace/model-performance/michaelfeil/onejev-perf/backbone-comparison-20261023/models/gemma-4-E2B".to_owned()
        });
        let head512_root = PathBuf::from(
            std::env::var("INFINITY_HEAD512_ROOT").expect("set INFINITY_HEAD512_ROOT"),
        );
        let head512_arch =
            std::env::var("INFINITY_HEAD512_ARCH").unwrap_or_else(|_| "sm_90".to_owned());
        let max_optin = std::env::var("INFINITY_HEAD512_MAX_OPTIN")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(128 * 1024);
        let fa4_root =
            PathBuf::from(std::env::var("INFINITY_FA4_ROOT").expect("set INFINITY_FA4_ROOT"));
        let weights =
            Gemma4Weights::load_embedding(Path::new(&checkpoint)).expect("E2B text weights");
        let context = Context::new(0).expect("CUDA context");
        let stream = context.new_stream().expect("CUDA stream");
        let blas = Blas::new(context.clone(), stream.clone()).expect("plain cuBLAS");
        let attention = Arc::new(
            Head512Aot::load(&context, &head512_root, &head512_arch, max_optin)
                .expect("Head512 provider"),
        );
        let mut executor = Gemma4E2bExecutor::new_with_fa4_local(
            context.clone(),
            stream.clone(),
            &weights,
            Some(attention),
            &fa4_root,
            None,
        )
        .expect("canonical FA4 text executor");
        let case = &cases["cases"][1];
        let base = control_batch(case);
        let mut batch_a = base.clone();
        batch_a.raw_indices = vec![0];
        let mut batch_b = base.clone();
        batch_b.raw_indices = vec![1];
        let tokens = base.input_ids.len();
        let boundaries = base.cumulative_seq_lengths.len();
        let selections = base.pooled_indices.len();
        let raw_a = crate::output::raw_rows(&batch_a).expect("raw A");
        let raw_b = crate::output::raw_rows(&batch_b).expect("raw B");
        assert_ne!(batch_a.raw_indices, batch_b.raw_indices);
        let capacity = (tokens, boundaries, selections);
        let max_raw = raw_a.max(raw_b);
        let run_solo =
            |batch: &Batch, raw_rows: usize, executor: &mut Gemma4E2bExecutor| -> Vec<u32> {
                let mut work =
                    Gemma4E2bWork::new_with_raw(stream.clone(), &weights, capacity, raw_rows)
                        .expect("solo Work");
                let pool = crate::staging::HostIo::pool(&context).expect("HostIo pool");
                let slot = pool.acquire().expect("solo lease slot");
                let mut lease = HostIoLease::detached(slot, pool.clone(), stream.clone());
                let elements = (selections + raw_rows) * E2B_HIDDEN;
                let host_key = (tokens, boundaries, selections + raw_rows);
                lease
                    .prepare(host_key, elements, false)
                    .expect("solo lease");
                let geometry =
                    crate::output::OutputGeometry::new(E2B_HIDDEN, E2B_HIDDEN).expect("geometry");
                let (prefix, _, _) = work
                    .stage_text(batch, geometry, &mut lease)
                    .expect("stage solo");
                work.upload_text_prefix(&lease, prefix)
                    .expect("upload solo prefix");
                executor
                    .enqueue_text(
                        Some(&blas),
                        &stream,
                        &mut work,
                        batch,
                        &Pool::Mean,
                        batch.max_length as usize,
                        tokens,
                        &lease,
                    )
                    .expect("solo enqueue");
                executor
                    .finish_text(&stream, &mut work, batch, selections, raw_rows, &mut lease)
                    .expect("solo finish");
                lease.wait().expect("solo wait");
                lease
                    .output(elements)
                    .expect("solo output")
                    .iter()
                    .map(|value| value.to_bits())
                    .collect()
            };
        let expected_a = run_solo(&batch_a, raw_a, &mut executor);
        let expected_b = run_solo(&batch_b, raw_b, &mut executor);
        let expected_a_again = run_solo(&batch_a, raw_a, &mut executor);
        assert_eq!(
            expected_a, expected_a_again,
            "held A source must be immutable"
        );
        let mut work = Gemma4E2bWork::new_with_raw(stream.clone(), &weights, capacity, max_raw)
            .expect("shared Work");
        let pool = crate::staging::HostIo::pool(&context).expect("HostIo pool");
        let slot_a = pool.acquire().expect("A slot");
        let slot_b = pool.acquire().expect("B slot");
        let mut lease_a = HostIoLease::detached(slot_a, pool.clone(), stream.clone());
        let mut lease_b = HostIoLease::detached(slot_b, pool.clone(), stream.clone());
        let issue = |work: &mut Gemma4E2bWork,
                     lease: &mut HostIoLease,
                     batch: &Batch,
                     raw_rows: usize,
                     executor: &mut Gemma4E2bExecutor| {
            let elements = (selections + raw_rows) * E2B_HIDDEN;
            let host_key = (tokens, boundaries, selections + raw_rows);
            lease
                .prepare(host_key, elements, false)
                .expect("lease prepare");
            let geometry =
                crate::output::OutputGeometry::new(E2B_HIDDEN, E2B_HIDDEN).expect("geometry");
            let (prefix, _, _) = work.stage_text(batch, geometry, lease).expect("stage text");
            work.upload_text_prefix(lease, prefix)
                .expect("upload prefix");
            executor
                .enqueue_text(
                    Some(&blas),
                    &stream,
                    work,
                    batch,
                    &Pool::Mean,
                    batch.max_length as usize,
                    tokens,
                    lease,
                )
                .expect("text enqueue");
            executor
                .finish_text(&stream, work, batch, selections, raw_rows, lease)
                .expect("text finish");
            elements
        };
        let elements_a = issue(&mut work, &mut lease_a, &batch_a, raw_a, &mut executor);
        let elements_b = issue(&mut work, &mut lease_b, &batch_b, raw_b, &mut executor);
        lease_b.wait().expect("B wait");
        lease_a.wait().expect("A wait");
        let actual_a: Vec<u32> = lease_a
            .output(elements_a)
            .expect("A output")
            .iter()
            .map(|value| value.to_bits())
            .collect();
        let actual_b: Vec<u32> = lease_b
            .output(elements_b)
            .expect("B output")
            .iter()
            .map(|value| value.to_bits())
            .collect();
        assert_eq!(actual_a, expected_a, "held A output was overwritten");
        assert_eq!(actual_b, expected_b, "B output differs");
        // Re-recording the same lease is rejected before any new async work.
        assert!(executor
            .finish_text(
                &stream,
                &mut work,
                &batch_a,
                selections,
                raw_a,
                &mut lease_a
            )
            .is_err());
        drop(lease_a);
        drop(lease_b);
        // B cannot stage/upload before A has queued its output D2H.
        let geometry =
            crate::output::OutputGeometry::new(E2B_HIDDEN, E2B_HIDDEN).expect("text geometry");
        let slot_a2 = pool.acquire().expect("A2 slot");
        let slot_b2 = pool.acquire().expect("B2 slot");
        let mut lease_a2 = HostIoLease::detached(slot_a2, pool.clone(), stream.clone());
        let mut lease_b2 = HostIoLease::detached(slot_b2, pool.clone(), stream.clone());
        let elements_a2 = (selections + raw_a) * E2B_HIDDEN;
        let elements_b2 = (selections + raw_b) * E2B_HIDDEN;
        lease_a2
            .prepare((tokens, boundaries, selections + raw_a), elements_a2, false)
            .expect("A2 lease");
        let (prefix_a2, _, _) = work
            .stage_text(&batch_a, geometry, &mut lease_a2)
            .expect("A2 stage");
        work.upload_text_prefix(&lease_a2, prefix_a2)
            .expect("A2 prefix upload");
        executor
            .enqueue_text(
                Some(&blas),
                &stream,
                &mut work,
                &batch_a,
                &Pool::Mean,
                batch_a.max_length as usize,
                tokens,
                &lease_a2,
            )
            .expect("A2 enqueue");
        assert!(
            work.stage_text(&batch_b, geometry, &mut lease_b2).is_err(),
            "B must be rejected before its first pinned write/H2D"
        );
        executor
            .finish_text(
                &stream,
                &mut work,
                &batch_a,
                selections,
                raw_a,
                &mut lease_a2,
            )
            .expect("A2 finish");
        lease_b2
            .prepare((tokens, boundaries, selections + raw_b), elements_b2, false)
            .expect("B2 lease after A finish");
        let (prefix_b2, _, _) = work
            .stage_text(&batch_b, geometry, &mut lease_b2)
            .expect("B2 stage after A output-copied");
        work.upload_text_prefix(&lease_b2, prefix_b2)
            .expect("B2 prefix upload");
        executor
            .enqueue_text(
                Some(&blas),
                &stream,
                &mut work,
                &batch_b,
                &Pool::Mean,
                batch_b.max_length as usize,
                tokens,
                &lease_b2,
            )
            .expect("B2 enqueue");
        executor
            .finish_text(
                &stream,
                &mut work,
                &batch_b,
                selections,
                raw_b,
                &mut lease_b2,
            )
            .expect("B2 finish");
        lease_b2.wait().expect("B2 wait");
        lease_a2.wait().expect("A2 wait");
        let a2: Vec<u32> = lease_a2
            .output(elements_a2)
            .expect("A2 output")
            .iter()
            .map(|value| value.to_bits())
            .collect();
        let b2: Vec<u32> = lease_b2
            .output(elements_b2)
            .expect("B2 output")
            .iter()
            .map(|value| value.to_bits())
            .collect();
        assert_eq!(a2, expected_a);
        assert_eq!(b2, expected_b);
        drop(lease_a2);
        drop(lease_b2);
        unsafe { work.reset_text_after_drain() };
        // Too-small negative with a genuinely fresh pool: a reused slot can
        // retain a rounded larger output buffer and invalidate the witness.
        let small_pool = crate::staging::HostIo::pool(&context).expect("fresh small pool");
        let small_slot = small_pool.acquire().expect("fresh small slot");
        let mut small = HostIoLease::detached(small_slot, small_pool.clone(), stream.clone());
        let small_key = (tokens, boundaries, selections + raw_a);
        let required_output_elements = (selections + raw_a) * E2B_HIDDEN;
        small
            .prepare(small_key, 1, false)
            .expect("small lease prepare");
        assert!(
            small.slot().expect("small slot").output.len() < required_output_elements,
            "fresh small lease must retain a physically smaller output buffer"
        );
        let geometry =
            crate::output::OutputGeometry::new(E2B_HIDDEN, E2B_HIDDEN).expect("geometry");
        assert!(
            work.stage_text(&batch_a, geometry, &mut small).is_err(),
            "too-small physical output must be rejected before upload/enqueue/readout"
        );
        drop(small);
        stream.synchronize().expect("small error drain");
        unsafe { work.reset_text_after_drain() };
        // Injected failure after the pooled D2H: the begun lease must retain
        // pinned storage through Drop, and a fresh lease must recover.
        let injected_slot = pool.acquire().expect("injected slot");
        let mut injected = HostIoLease::detached(injected_slot, pool.clone(), stream.clone());
        let injected_elements = (selections + raw_a) * E2B_HIDDEN;
        injected
            .prepare(
                (tokens, boundaries, selections + raw_a),
                injected_elements,
                false,
            )
            .expect("injected lease prepare");
        let (injected_prefix, _, _) = work
            .stage_text(&batch_a, geometry, &mut injected)
            .expect("injected stage");
        work.upload_text_prefix(&injected, injected_prefix)
            .expect("injected upload");
        executor
            .enqueue_text(
                Some(&blas),
                &stream,
                &mut work,
                &batch_a,
                &Pool::Mean,
                batch_a.max_length as usize,
                tokens,
                &injected,
            )
            .expect("injected enqueue");
        INJECT_TEXT_FINISH_AFTER_POOLED.store(true, std::sync::atomic::Ordering::SeqCst);
        assert!(executor
            .finish_text(
                &stream,
                &mut work,
                &batch_a,
                selections,
                raw_a,
                &mut injected
            )
            .is_err());
        drop(injected);
        stream.synchronize().expect("injected failure drain");
        unsafe { work.reset_text_after_drain() };
        let recovered_after_injection = run_solo(&batch_a, raw_a, &mut executor);
        assert_eq!(
            recovered_after_injection, expected_a,
            "recovery after injected async failure differs"
        );
        // Early lease drop after enqueue, then explicit drain and recovery.
        let early_slot = pool.acquire().expect("early slot");
        let mut early = HostIoLease::detached(early_slot, pool.clone(), stream.clone());
        let _ = issue(&mut work, &mut early, &batch_a, raw_a, &mut executor);
        drop(early);
        stream.synchronize().expect("early drop drain");
        let recovered = run_solo(&batch_a, raw_a, &mut executor);
        assert_eq!(recovered, expected_a, "recovery after early drop differs");
    }
}
