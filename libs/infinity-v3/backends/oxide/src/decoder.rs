// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Graph-free BF16 text decoder executor with dense and routed MLP methods.
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
    arg, blocks, fa4, infer, launch, launch_threads, product, stage_batch, start, u32_size,
    upload_input, HostIoLease, InputLayout, LtPlan, LtProvider, WorkKey,
};
use crate::{
    cublas::Blas,
    decoder_classifier::{
        DecoderClassifierHead, DecoderClassifierHeadHost, DecoderClassifierReadout,
        DecoderClassifierRows, DecoderClassifierScratchBuffers, DecoderClassifierScratchLayout,
        DecoderClassifierTask, MeanRows, RowPlan, RowPlanKind,
    },
    decoder_model::{
        DecoderConfig, DecoderFamily, DecoderLayerWeights, DecoderMlpWeights, DecoderRope,
        DecoderWeights,
    },
    dense::DenseHead,
    lt::ShapeError,
    math::LinearPlan,
    moe_runtime::{MoeFrame, MoeLayer, MoeProvider, MoeScratch},
    rotary::BaseRopeTables,
    staging,
    tensor::Bf16,
    upload::WeightUploader,
    work_retirement::{CompletedTicket, CompletionTicket},
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

/// Mistral H4096 binds the current public 90-export Gemma2 BF16 rotary export.
/// Every standard BF16 dense math recipe binds the shared public Gemma2 BF16
/// mean-pool export for `Pool::Mean` with at most 65535 selections. The old
/// private Mistral kernel names are deliberately not a fallback: a bundle
/// without these public exports is not this ABI and must be qualified
/// separately before any GPU/numerical claim.
pub(crate) const MISTRAL_PUBLIC_ROPE_EXPORT: &str = "infinity_gemma2_rope_tables_bf16";
pub(crate) const PUBLIC_MEAN_EXPORT: &str = "infinity_gemma2_mean_pool_bf16";

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
const H1024_NORM_GEOMETRY: NormGeometry = NormGeometry {
    rows_per_cta: 4,
    threads: 128,
};
/// Ordinary H5120 RMS uses the qualified ATen128 vectorized reduction.
/// Add-RMS remains on its separately qualified generic256 recipe.
const H5120_RMS_GEOMETRY: NormGeometry = NormGeometry {
    rows_per_cta: 1,
    threads: 128,
};
/// Native BF16 width-896 registers the same 4x1 Candle warp geometry as the
/// width-1024 vector kernel, with the actual 896 columns masked inside it.
const H896_NORM_GEOMETRY: NormGeometry = NormGeometry {
    rows_per_cta: 4,
    threads: 128,
};
/// Dense Qwen3 fused per-head Q/K RMSNorm + RoPE launch geometry.
const QK_NORM_ROPE_THREADS: u32 = 128;
const QK_NORM_ROPE_HEAD_ROWS_PER_CTA: usize = 8;

/// Native arithmetic recipe resolved once from loaded metadata. Executor
/// functions and launch geometry are bound from this value; the layer loop
/// never re-evaluates family or width policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DecoderMath {
    Qwen2FullNeox,
    Qwen2H896MaskedCandle,
    Qwen2H1536FullNeox,
    Qwen3H1024Candle,
    Qwen3H2048Ordinary,
    Qwen3H2560Ordinary,
    Qwen3H4096Ordinary,
    Qwen3H5120Ordinary,
    LlamaH2048Exact,
    MistralH4096Ordinary,
    Qwen3MoeH2048Exact,
    Qwen3MoeH4096Ordinary,
}
impl DecoderMath {
    pub(crate) fn select(config: &DecoderConfig) -> Result<Self, BackendError> {
        match config.family {
            DecoderFamily::Qwen2 => {
                if config.rope != DecoderRope::FullNeoX {
                    return Err(start(
                        "decoder executor requires the native Qwen2 full split-half NeoX RoPE recipe",
                    ));
                }
                Ok(if config.hidden_size == 896 && config.head_dim == 64 {
                    // Native exact_decoder_math selects the masked width-896
                    // Candle reduction and model-dtype HFMA rotary for these.
                    Self::Qwen2H896MaskedCandle
                } else if config.hidden_size == 1536 {
                    Self::Qwen2H1536FullNeox
                } else {
                    Self::Qwen2FullNeox
                })
            }
            DecoderFamily::Qwen3 => {
                if config.rope != DecoderRope::FullNeoX {
                    return Err(start(
                        "decoder executor requires the native Qwen3 full split-half NeoX RoPE recipe",
                    ));
                }
                if config.head_dim != 128 {
                    return Err(start(format!(
                        "decoder executor implements only Qwen3 head_dim 128; H{} d{} is deferred",
                        config.hidden_size, config.head_dim
                    )));
                }
                match config.hidden_size {
                    1024 => Ok(Self::Qwen3H1024Candle),
                    2048 => Ok(Self::Qwen3H2048Ordinary),
                    2560 => Ok(Self::Qwen3H2560Ordinary),
                    4096 => Ok(Self::Qwen3H4096Ordinary),
                    5120 => qwen3_h5120_admission(config).map(|()| Self::Qwen3H5120Ordinary),
                    hidden => Err(start(format!(
                        "decoder executor implements only Qwen3 hidden_size 1024, 2048, 2560, 4096 or 5120; H{hidden} is deferred"
                    ))),
                }
            }
            DecoderFamily::Qwen3Moe => {
                let routed = config
                    .routed
                    .as_ref()
                    .ok_or_else(|| start("Missing Qwen3 MoE metadata"))?;
                if config.head_dim != 128
                    || !config.is_causal
                    || config.rope != DecoderRope::FullNeoX
                    || routed.num_experts != 128
                    || routed.num_experts_per_tok != 8
                {
                    return Err(start(
                        "Qwen3 MoE requires BF16 d128/causal/E128/top8 and full NeoX RoPE",
                    ));
                }
                match (
                    config.hidden_size,
                    config.num_attention_heads,
                    config.num_key_value_heads,
                    routed.moe_intermediate_size,
                ) {
                    (2048, 32, 4, 768) => Ok(Self::Qwen3MoeH2048Exact),
                    (4096, 64, 4, 1536) => Ok(Self::Qwen3MoeH4096Ordinary),
                    _ => Err(start(
                        "Qwen3 MoE admits only H2048/Q32/KV4/I768 or H4096/Q64/KV4/I1536 with the Native E128/top8 recipe",
                    )),
                }
            }
            DecoderFamily::Llama => {
                if config.hidden_size != 2048 {
                    return Err(start(
                        "decoder executor implements only the BF16 Llama hidden_size 2048 recipe; H4096 is deferred",
                    ));
                }
                if config.rope != DecoderRope::Default {
                    return Err(start(
                        "decoder executor requires the Llama default RoPE recipe",
                    ));
                }
                // Bidirectional Llama classifiers reuse the same global trunk
                // recipe; only the FA4 mask and mean-pooled head differ.
                Ok(Self::LlamaH2048Exact)
            }
            DecoderFamily::Mistral => {
                if (
                    config.hidden_size,
                    config.head_dim,
                    config.num_attention_heads,
                    config.num_key_value_heads,
                ) != (4096, 128, 32, 8)
                {
                    return Err(start("decoder Mistral requires H4096/d128/heads32/KV8"));
                }
                if config.rope != DecoderRope::Default || !config.is_causal {
                    return Err(start(
                        "decoder Mistral requires base default RoPE and causal attention",
                    ));
                }
                match config.attention_window {
                    None | Some(4096) => Ok(Self::MistralH4096Ordinary),
                    Some(_) => Err(start("decoder Mistral window admits only global or 4096")),
                }
            }
        }
    }

    const fn rms_export(self) -> &'static str {
        match self {
            Self::Qwen2FullNeox => "infinity_rms_norm",
            Self::Qwen2H896MaskedCandle => "infinity_rms_norm_896_exact_masked_vector",
            Self::Qwen2H1536FullNeox => "infinity_rms_norm_1536_vector",
            Self::Qwen3H1024Candle => "infinity_rms_norm_1024_exact_vector",
            Self::Qwen3H2048Ordinary | Self::Qwen3H2560Ordinary => "infinity_rms_norm",
            Self::Qwen3H5120Ordinary => "infinity_rms_norm_5120_aten",
            Self::Qwen3H4096Ordinary | Self::MistralH4096Ordinary => {
                "infinity_rms_norm_4096_vector"
            }
            Self::LlamaH2048Exact | Self::Qwen3MoeH2048Exact => "infinity_rms_norm_2048_vector",
            Self::Qwen3MoeH4096Ordinary => "infinity_rms_norm_4096_vector",
        }
    }

    const fn add_rms_export(self) -> &'static str {
        match self {
            Self::Qwen2FullNeox => "infinity_add_rms_norm",
            Self::Qwen2H896MaskedCandle => "infinity_add_rms_norm_896_exact_masked_vector",
            Self::Qwen2H1536FullNeox => "infinity_add_rms_norm_1536_vector",
            Self::Qwen3H1024Candle => "infinity_add_rms_norm_1024_exact_vector",
            Self::Qwen3H2048Ordinary | Self::Qwen3H2560Ordinary | Self::Qwen3H5120Ordinary => {
                "infinity_add_rms_norm"
            }
            Self::Qwen3H4096Ordinary | Self::MistralH4096Ordinary => {
                "infinity_add_rms_norm_4096_vector"
            }
            Self::LlamaH2048Exact | Self::Qwen3MoeH2048Exact => "infinity_add_rms_norm_2048_vector",
            Self::Qwen3MoeH4096Ordinary => "infinity_add_rms_norm_4096_vector",
        }
    }

    const fn rope_export(self) -> &'static str {
        match self {
            Self::Qwen2FullNeox | Self::Qwen2H1536FullNeox => "infinity_rope_neox",
            Self::Qwen2H896MaskedCandle => "infinity_rope_neox_hfma",
            Self::Qwen3H1024Candle
            | Self::Qwen3H2048Ordinary
            | Self::Qwen3H2560Ordinary
            | Self::Qwen3H4096Ordinary
            | Self::Qwen3H5120Ordinary
            | Self::MistralH4096Ordinary
            | Self::Qwen3MoeH2048Exact
            | Self::Qwen3MoeH4096Ordinary => "infinity_rope_neox",
            Self::LlamaH2048Exact => "infinity_rope_neox_hfma",
        }
    }

    const fn qk_norm_rope_export(self) -> Option<&'static str> {
        match self {
            Self::Qwen3H1024Candle
            | Self::Qwen3H2048Ordinary
            | Self::Qwen3H2560Ordinary
            | Self::Qwen3H4096Ordinary
            | Self::Qwen3H5120Ordinary
            | Self::Qwen3MoeH2048Exact
            | Self::Qwen3MoeH4096Ordinary => Some("infinity_qk_norm_rope_128_bf16"),
            Self::Qwen2FullNeox
            | Self::Qwen2H896MaskedCandle
            | Self::Qwen2H1536FullNeox
            | Self::LlamaH2048Exact
            | Self::MistralH4096Ordinary => None,
        }
    }

    /// Native DenseMlp code5 (`RoundedSwiGlu`) uses BF16 pair instructions, so
    /// both decoder families bind the eight-value rounded-intermediate export.
    /// The scalar and paired exports remain available to other callers.
    const fn activation_export(self) -> &'static str {
        match self {
            Self::Qwen2FullNeox
            | Self::Qwen2H896MaskedCandle
            | Self::Qwen2H1536FullNeox
            | Self::Qwen3H1024Candle
            | Self::Qwen3H2048Ordinary
            | Self::Qwen3H2560Ordinary
            | Self::Qwen3H4096Ordinary
            | Self::Qwen3H5120Ordinary
            | Self::LlamaH2048Exact
            | Self::MistralH4096Ordinary
            | Self::Qwen3MoeH2048Exact
            | Self::Qwen3MoeH4096Ordinary => "infinity_silu_mul_bf16_vector8_native",
        }
    }

    const fn norm_geometry(self) -> NormGeometry {
        match self {
            Self::Qwen2FullNeox
            | Self::Qwen2H1536FullNeox
            | Self::Qwen3H2048Ordinary
            | Self::Qwen3H2560Ordinary
            | Self::Qwen3H4096Ordinary
            | Self::Qwen3H5120Ordinary
            | Self::MistralH4096Ordinary
            | Self::Qwen3MoeH4096Ordinary => GENERIC_NORM_GEOMETRY,
            Self::Qwen3H1024Candle => H1024_NORM_GEOMETRY,
            Self::Qwen2H896MaskedCandle => H896_NORM_GEOMETRY,
            Self::LlamaH2048Exact | Self::Qwen3MoeH2048Exact => H2048_NORM_GEOMETRY,
        }
    }

    const fn rms_geometry(self) -> NormGeometry {
        match self {
            Self::Qwen3H5120Ordinary => H5120_RMS_GEOMETRY,
            _ => self.norm_geometry(),
        }
    }
}

/// H5120 Qwen3 dense uses the two Native d128/GQA topologies. Query
/// width is independently checked as heads*head_dim; Q64 is wider than H5120.
fn qwen3_h5120_admission(config: &DecoderConfig) -> Result<(), BackendError> {
    match (config.num_attention_heads, config.num_key_value_heads) {
        (40, 8) | (64, 8) => Ok(()),
        (heads, kv_heads) => Err(start(format!(
            "decoder Qwen3 H5120 admits only Native Q40/KV8 (q_width 5120) or Q64/KV8 (q_width 8192) at head_dim 128; Q{heads}/KV{kv_heads} is not admitted"
        ))),
    }
}

/// Pooled-mean admission for the shared public BF16 export.
///
/// Every contiguous BF16 decoder `Pool::Mean` request with at most 65535
/// pooled rows uses the same BF16 partial-tree/final-F32-widen route. Larger
/// selection counts keep the existing generic `infinity_pool` F32 reduction;
/// raw-only requests bypass the pooled kernel entirely.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MeanPoolRoute {
    Empty,
    NativeBf16Tree,
    F32Fallback,
}

/// Pure route predicate shared by dense and routed decoder pooling.
pub(crate) fn mean_pool_route(
    pool: &Pool,
    selections: usize,
) -> Result<MeanPoolRoute, BackendError> {
    match pool {
        Pool::Splade => Err(infer("Oxide SPLADE unsupported")),
        _ if selections == 0 => Ok(MeanPoolRoute::Empty),
        Pool::Mean if selections <= usize::from(u16::MAX) => Ok(MeanPoolRoute::NativeBf16Tree),
        Pool::Cls | Pool::Mean | Pool::LastToken => Ok(MeanPoolRoute::F32Fallback),
    }
}

struct DecoderKernels {
    gather: Function,
    rms_norm: Function,
    add_rms_norm: Function,
    rope_neox: Function,
    qk_norm_rope: Option<Function>,
    rope_initializer: Option<Arc<Function>>,
    silu_mul: Function,
    pool: Function,
    mean_kernel: Option<Function>,
    selected_rows: Function,
    bf16_to_f32: Function,
    rms_geometry: NormGeometry,
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
        let qk_norm_rope = math
            .qk_norm_rope_export()
            .map(|export| module.load_function(export).map_err(start))
            .transpose()?;
        let rope_initializer = if math == DecoderMath::MistralH4096Ordinary {
            Some(Arc::new(
                module
                    .load_function(MISTRAL_PUBLIC_ROPE_EXPORT)
                    .map_err(|error| {
                        start(format!(
                            "Mistral requires the current public 90-export Gemma2 ABI bundle export {MISTRAL_PUBLIC_ROPE_EXPORT}: {error}; old private Mistral exports are not accepted"
                        ))
                    })?,
            ))
        } else {
            None
        };
        let routed = matches!(
            math,
            DecoderMath::Qwen3MoeH2048Exact | DecoderMath::Qwen3MoeH4096Ordinary
        );
        // Cold, once-per-model resolution. Standard dense recipes require the
        // public 90-export thinner; routed MoE keeps compatibility with the
        // already-qualified native sidecar bundle when the public export is absent.
        let mean_kernel = if routed {
            module
                .load_function(PUBLIC_MEAN_EXPORT)
                .ok()
                .or_else(|| module.load_function("infinity_mean_pool_bf16_native").ok())
        } else {
            Some(module.load_function(PUBLIC_MEAN_EXPORT).map_err(|error| {
                start(format!(
                    "BF16 dense decoder mean pooling requires the current public 90-export Gemma2 ABI bundle export {PUBLIC_MEAN_EXPORT}: {error}; old private Mistral exports and the unported ATen model-dtype reduction are not accepted"
                ))
            })?)
        };
        Ok(Self {
            gather: module
                .load_function("infinity_embedding_gather")
                .map_err(start)?,
            rms_norm: module.load_function(math.rms_export()).map_err(start)?,
            add_rms_norm: module.load_function(math.add_rms_export()).map_err(start)?,
            rope_neox: module.load_function(math.rope_export()).map_err(start)?,
            qk_norm_rope,
            rope_initializer,
            silu_mul: module
                .load_function(math.activation_export())
                .map_err(start)?,
            pool: module.load_function("infinity_pool").map_err(start)?,
            mean_kernel,
            selected_rows: module
                .load_function("infinity_selected_rows")
                .map_err(start)?,
            bf16_to_f32: module
                .load_function("infinity_bf16_to_f32")
                .map_err(start)?,
            rms_geometry: math.rms_geometry(),
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
    q_norm: Option<DeviceBuffer<Bf16>>,
    k_norm: Option<DeviceBuffer<Bf16>>,
    qkv: DecoderLinear,
    attention: DecoderLinear,
    mlp: DecoderMlp,
}
enum DecoderMlp {
    Dense {
        gate_up: DecoderLinear,
        down: DecoderLinear,
    },
    Routed(MoeLayer),
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
            let attention_bias = layer
                .o
                .bias
                .as_ref()
                .map(|bias| uploader.upload(&[bias]))
                .transpose()?;
            let mlp = match &layer.mlp {
                DecoderMlpWeights::Dense { gate, up, down } => {
                    let bias = match (&gate.bias, &up.bias) {
                        (Some(gate), Some(up)) => Some(uploader.upload(&[gate, up])?),
                        (None, None) => None,
                        _ => return Err(start("decoder gate/up biases must form a complete pair")),
                    };
                    DecoderMlp::Dense {
                        gate_up: DecoderLinear {
                            weight: uploader.upload(&[&gate.weight, &up.weight])?,
                            bias,
                        },
                        down: DecoderLinear {
                            weight: uploader.upload(&[&down.weight])?,
                            bias: down
                                .bias
                                .as_ref()
                                .map(|bias| uploader.upload(&[bias]))
                                .transpose()?,
                        },
                    }
                }
                DecoderMlpWeights::Routed(layer) => DecoderMlp::Routed(MoeLayer::upload(
                    uploader,
                    layer,
                    host.config
                        .routed
                        .as_ref()
                        .ok_or_else(|| start("Missing routed config"))?,
                )?),
            };
            let (q_norm, k_norm) = match (&layer.q_norm, &layer.k_norm) {
                (Some(q), Some(k)) => (Some(uploader.upload(&[q])?), Some(uploader.upload(&[k])?)),
                (None, None) => (None, None),
                _ => {
                    return Err(start("decoder Q/K RMSNorm scales must be a complete pair"));
                }
            };
            layers.push(DecoderLayer {
                input_norm: uploader.upload(&[&layer.input_norm])?,
                post_attention_norm: uploader.upload(&[&layer.post_attention_norm])?,
                q_norm,
                k_norm,
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
                mlp,
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
            match &layer.mlp {
                DecoderMlpWeights::Dense { gate, down, .. } => {
                    support.gate_up |= gate.bias.is_some();
                    support.down |= down.bias.is_some();
                    support.requires_plain |= gate.bias.is_none() || down.bias.is_none();
                }
                DecoderMlpWeights::Routed(_) => support.requires_plain = true,
            }
            support.requires_plain |= layer.q.bias.is_none() || layer.o.bias.is_none();
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
            gate_up: matches!(&layer.mlp, DecoderMlpWeights::Dense { gate, .. } if gate.bias.is_some()),
            down: matches!(&layer.mlp, DecoderMlpWeights::Dense { down, .. } if down.bias.is_some()),
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
pub(crate) struct DecoderGeometry {
    pub(crate) hidden: usize,
    pub(crate) q_width: usize,
    pub(crate) kv_width: usize,
    pub(crate) qkv_width: usize,
    pub(crate) projection_width: usize,
    pub(crate) activation_width: usize,
    pub(crate) silu_groups_per_row: usize,
}
impl DecoderGeometry {
    fn new(c: &DecoderConfig) -> Result<Self, BackendError> {
        Self::from_dimensions(
            c.hidden_size,
            c.intermediate_size,
            c.q_width(),
            c.kv_width(),
            c.head_dim,
        )
    }

    pub(crate) fn from_dimensions(
        hidden: usize,
        intermediate: usize,
        q_width: usize,
        kv_width: usize,
        head_dim: usize,
    ) -> Result<Self, BackendError> {
        if [hidden, intermediate, q_width, kv_width, head_dim].contains(&0) {
            return Err(start("decoder geometry dimensions must be positive"));
        }
        let qkv_width = q_width
            .checked_add(
                kv_width
                    .checked_mul(2)
                    .ok_or_else(|| start("decoder QKV width overflow"))?,
            )
            .ok_or_else(|| start("decoder QKV width overflow"))?;
        let gate_up_width = intermediate
            .checked_mul(2)
            .ok_or_else(|| start("decoder gate/up width overflow"))?;
        let projection_width = qkv_width.max(gate_up_width);
        let activation_width = q_width.max(intermediate);
        i64::try_from(qkv_width).map_err(start)?;
        i64::try_from(q_width).map_err(start)?;
        i64::try_from(kv_width).map_err(start)?;
        i64::try_from(head_dim).map_err(start)?;
        Ok(Self {
            hidden,
            q_width,
            kv_width,
            qkv_width,
            projection_width,
            activation_width,
            silu_groups_per_row: intermediate.div_ceil(8),
        })
    }

    pub(crate) fn attention_strides(&self, head_dim: usize) -> Result<[i64; 12], BackendError> {
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
    pub(crate) const fn kernel_mode(self) -> u32 {
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

fn attention_spec(config: &DecoderConfig) -> Result<fa4::AttentionSpec, BackendError> {
    let window = match config.attention_window {
        None => fa4::Window::Global,
        Some(window) => fa4::Window::Local {
            left: i32::try_from(
                window
                    .checked_sub(1)
                    .ok_or_else(|| start("decoder window underflow"))?,
            )
            .map_err(start)?,
            right: 0,
        },
    };
    Ok(fa4::AttentionSpec {
        heads: config.num_attention_heads,
        kv_heads: config.num_key_value_heads,
        head_dim: config.head_dim,
        causal: config.is_causal,
        window,
        dtype: fa4::DType::Bf16,
        scale: fa4::Scale::RsqrtHeadDim,
    })
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
    selected_readout_enabled: bool,
    head: Option<DecoderClassifierHead>,
    moe: Option<MoeProvider>,
    moe_frame: Option<MoeFrame>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ClassifierMode {
    /// Sequence head: select last-non-PAD rows before the final pointwise seam.
    Selected,
    /// Token head serving sequence readout: full-T forward then head gather.
    FullSequenceGather,
    /// Token head token readout: full-T forward then head projection.
    FullTokens,
    /// Bidirectional sequence head: full-T forward then per-sequence mean.
    FullSequenceMean,
}
fn classifier_mode(plan: Option<&RowPlan>) -> Option<ClassifierMode> {
    plan.map(|plan| match plan.kind() {
        RowPlanKind::Selected => ClassifierMode::Selected,
        RowPlanKind::PackedRows => ClassifierMode::FullSequenceGather,
        RowPlanKind::Tokens => ClassifierMode::FullTokens,
        RowPlanKind::MeanSequence => ClassifierMode::FullSequenceMean,
    })
}

impl DecoderModel {
    #[allow(dead_code)]
    pub(crate) fn new(
        host: DecoderWeights,
        context: &Arc<Context>,
        stream: &Arc<Stream>,
        lt: &LtProvider,
    ) -> Result<Self, BackendError> {
        Self::new_with_classifier(host, None, context, stream, lt)
    }

    pub(crate) fn new_with_classifier(
        host: DecoderWeights,
        head_host: Option<DecoderClassifierHeadHost>,
        context: &Arc<Context>,
        stream: &Arc<Stream>,
        lt: &LtProvider,
    ) -> Result<Self, BackendError> {
        let config = host.config.clone();
        let math = DecoderMath::select(&config)?;
        let geometry = DecoderGeometry::new(&config)?;
        let moe = config
            .routed
            .as_ref()
            .map(|routed| {
                let path = crate::moe_artifact::resolve(context)?;
                // SAFETY: source-controlled deployment sidecar and SHA were validated cold.
                unsafe { MoeProvider::load(context, &path, routed) }
            })
            .transpose()?;
        let attention_spec = attention_spec(&config)?;
        if !fa4::Fa4::supports(&attention_spec) {
            return Err(start(format!(
                "decoder attention H{} KV{} d{} causal={} window={:?} has no qualified ABI7 FA4 export",
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                config.is_causal,
                attention_spec.window
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
        let head = head_host
            .map(|host| DecoderClassifierHead::new(host, context, stream, lt))
            .transpose()?;
        let (rope_cosine, rope_sine) = if math == DecoderMath::MistralH4096Ordinary {
            let initializer = kernels.rope_initializer.as_ref().ok_or_else(|| {
                start("Mistral requires the public Gemma2 BF16 rotary initializer")
            })?;
            crate::native_rope_tables::decoder_128_tables(
                config.rope_theta,
                config.max_position_embeddings,
                ROPE_TABLE_BYTE_LIMIT,
                stream,
                initializer.clone(),
            )?
        } else {
            let tables = match config.rope_scale {
                Some(scale) => BaseRopeTables::new_llama3(
                    config.head_dim,
                    config.rope_theta,
                    scale,
                    config.max_position_embeddings,
                    ROPE_TABLE_BYTE_LIMIT,
                ),
                None => BaseRopeTables::new(
                    config.head_dim,
                    config.rope_theta,
                    config.max_position_embeddings,
                    ROPE_TABLE_BYTE_LIMIT,
                ),
            }
            .map_err(start)?;
            if tables.head_dim() != config.head_dim
                || tables.position_capacity() != config.max_position_embeddings
                || tables.table_stride() != config.head_dim / 2
                || tables.table_bytes() == 0
            {
                return Err(start("decoder rotary table geometry disagrees with config"));
            }
            let cosine = DeviceBuffer::from_host(stream, tables.cosine()).map_err(start)?;
            let sine = DeviceBuffer::from_host(stream, tables.sine()).map_err(start)?;
            (cosine, sine)
        };
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
            selected_readout_enabled: math != DecoderMath::MistralH4096Ordinary,
            head,
            moe,
            moe_frame: None,
        })
    }

    pub(crate) fn workspace_bytes_typed(&self, key: WorkKey) -> Result<usize, ShapeError> {
        let classifier = self
            .classifier_scratch_layout(key)
            .map_err(ShapeError::Rejected)?;
        let base = DecoderWork::allocation_bytes_with_classifier(key, &self.config, classifier)
            .map_err(ShapeError::Rejected)?;
        let extra = self
            .moe
            .as_ref()
            .map(|provider| MoeScratch::allocation_bytes_typed(provider, key.0))
            .transpose()?
            .unwrap_or(0);
        base.checked_add(extra)
            .ok_or_else(|| ShapeError::Rejected(infer("Routed arena byte count overflow")))
    }

    pub(crate) fn workspace_bytes(&self, key: WorkKey) -> Result<usize, BackendError> {
        let classifier = self.classifier_scratch_layout(key)?;
        let base = DecoderWork::allocation_bytes_with_classifier(key, &self.config, classifier)?;
        let extra = self
            .moe
            .as_ref()
            .map(|provider| MoeScratch::allocation_bytes(provider, key.0))
            .transpose()?
            .unwrap_or(0);
        base.checked_add(extra)
            .ok_or_else(|| infer("Routed arena byte count overflow"))
    }

    pub(crate) fn workspace(
        &self,
        stream: Arc<Stream>,
        key: WorkKey,
    ) -> Result<DecoderWork, BackendError> {
        let extra = self
            .moe
            .as_ref()
            .map(|provider| MoeScratch::new(provider, stream.clone(), key.0))
            .transpose()?;
        let classifier = self.classifier_scratch_layout(key)?;
        let mut work = DecoderWork::new_with_classifier(stream, &self.config, key, classifier)?;
        if let Some(extra) = extra {
            let bytes = extra.bytes();
            work.bytes = work
                .bytes
                .checked_add(bytes)
                .ok_or_else(|| infer("Routed arena byte count overflow"))?;
            super::LIVE_ARENA_BYTES.fetch_add(bytes, Ordering::Relaxed);
            super::record_live_arena_bytes();
            work.moe = Some(extra);
        }
        Ok(work)
    }

    pub(crate) fn needs_plain(&self) -> bool {
        self.uses_plain
    }

    pub(crate) fn classifier_task(&self) -> Option<DecoderClassifierTask> {
        self.head.as_ref().map(DecoderClassifierHead::task)
    }

    pub(crate) fn classifier_width(&self) -> Option<usize> {
        self.head.as_ref().map(DecoderClassifierHead::width)
    }

    pub(crate) fn admit_readout(
        &self,
        mode: crate::classifier_head::ReadoutMode,
    ) -> Result<(), BackendError> {
        let Some(head) = self.head.as_ref() else {
            return Ok(());
        };
        let readout = match mode {
            crate::classifier_head::ReadoutMode::Sequence => DecoderClassifierReadout::Sequence,
            crate::classifier_head::ReadoutMode::Tokens => DecoderClassifierReadout::Tokens,
        };
        head.admit(readout)
    }

    pub(crate) fn classifier_scratch_layout(
        &self,
        key: WorkKey,
    ) -> Result<Option<DecoderClassifierScratchLayout>, BackendError> {
        self.head
            .as_ref()
            .map(|head| head.scratch_layout(key))
            .transpose()
    }

    pub(crate) fn stage_classifier_plan(
        &self,
        batch: &Batch,
        mode: crate::classifier_head::ReadoutMode,
        lease: &mut HostIoLease,
    ) -> Result<RowPlan, BackendError> {
        let head = self
            .head
            .as_ref()
            .ok_or_else(|| infer("Decoder classifier head is not configured"))?;
        let readout = match mode {
            crate::classifier_head::ReadoutMode::Sequence => DecoderClassifierReadout::Sequence,
            crate::classifier_head::ReadoutMode::Tokens => DecoderClassifierReadout::Tokens,
        };
        head.admit(readout)?;
        let sequences = batch
            .cumulative_seq_lengths
            .len()
            .checked_sub(1)
            .filter(|count| *count > 0)
            .ok_or_else(|| infer("Missing decoder sequence boundaries"))?;
        let layout = InputLayout::new(
            batch.input_ids.len(),
            batch.cumulative_seq_lengths.len(),
            batch.pooled_indices.len(),
        )?;
        let staging = lease
            .slot_mut()?
            .input
            .as_mut_slice()
            .get_mut(layout.selections_range())
            .ok_or_else(|| infer("Invalid decoder classifier row-id staging layout"))?;
        match (head.task(), readout) {
            (DecoderClassifierTask::Sequence, DecoderClassifierReadout::Sequence) => {
                RowPlan::sequence_into(
                    &batch.input_ids,
                    &batch.cumulative_seq_lengths,
                    head.pad_token_id(),
                    staging,
                )?;
                RowPlan::selected_sequence(sequences, sequences)
            }
            (DecoderClassifierTask::BidirectionalSequence, DecoderClassifierReadout::Sequence) => {
                // Native averages every nonempty logical span; the canonical
                // 0..B indices already staged in `batch.pooled_indices` address
                // those spans without overwriting the selections region.
                RowPlan::mean_sequence(&batch.input_ids, &batch.cumulative_seq_lengths)
            }
            (DecoderClassifierTask::Tokens, DecoderClassifierReadout::Sequence) => {
                RowPlan::sequence_into(
                    &batch.input_ids,
                    &batch.cumulative_seq_lengths,
                    head.pad_token_id(),
                    staging,
                )
            }
            (DecoderClassifierTask::Tokens, DecoderClassifierReadout::Tokens) => {
                RowPlan::tokens(batch.input_ids.len())
            }
            _ => Err(infer(
                "Decoder classifier task does not admit the requested readout",
            )),
        }
    }

    fn classifier_mode(&self, plan: Option<&RowPlan>) -> Option<ClassifierMode> {
        classifier_mode(plan)
    }

    pub(crate) fn prepare_classifier(
        &mut self,
        provider: &LtProvider,
        blas: &Blas,
        stream: &Stream,
        work: &mut DecoderWork,
        batch: &Batch,
        plan: &RowPlan,
    ) -> Result<(), ShapeError> {
        let Some(head) = self.head.as_mut() else {
            return Err(ShapeError::Fatal(infer(
                "Decoder classifier head is not configured",
            )));
        };
        let hidden = self.geometry.hidden;
        let source = DecoderClassifierRows::new(&work.normalized, plan.source_rows(), hidden)
            .map_err(ShapeError::Rejected)?;
        let layout = InputLayout::new(
            batch.input_ids.len(),
            batch.cumulative_seq_lengths.len(),
            batch.pooled_indices.len(),
        )
        .map_err(ShapeError::Rejected)?;
        let row_ids = if plan.requires_gather() {
            Some(
                layout
                    .selections()
                    .address(work.input.cu_deviceptr())
                    .map_err(ShapeError::Fatal)?,
            )
        } else {
            None
        };
        let mean = if plan.requires_mean() {
            let plan_mean = crate::gemma2_math::MeanPlan::new(
                batch.input_ids.len(),
                hidden,
                &batch.cumulative_seq_lengths,
                &batch.pooled_indices,
            )
            .map_err(|error| {
                ShapeError::Rejected(infer(format!("decoder classifier mean plan: {error}")))
            })?;
            let geometry = plan_mean.geometry().map_err(|error| {
                ShapeError::Rejected(infer(format!("decoder classifier mean grid: {error}")))
            })?;
            Some(MeanRows {
                boundaries: layout
                    .boundaries()
                    .address(work.input.cu_deviceptr())
                    .map_err(ShapeError::Fatal)?,
                indices: layout
                    .selections()
                    .address(work.input.cu_deviceptr())
                    .map_err(ShapeError::Fatal)?,
                grid: [geometry.grid()[0], geometry.grid()[1], geometry.grid()[2]],
                threads: geometry.block()[0],
            })
        } else {
            None
        };
        let scratch = work
            .classifier
            .as_deref_mut()
            .ok_or_else(|| ShapeError::Fatal(infer("Decoder classifier scratch is missing")))?;
        // SAFETY: the engine bound this context and every pointer comes from
        // the reserved work arena before the request lease begins.
        unsafe {
            head.prepare(provider, blas, stream, source, plan, row_ids, mean, scratch)?;
        }
        Ok(())
    }

    pub(crate) fn preflight(
        &mut self,
        provider: &LtProvider,
        batch: &Batch,
        pool: &Pool,
    ) -> Result<(), ShapeError> {
        self.plans.set_rows(provider, batch.input_ids.len())?;
        if let Some(moe) = self.moe.as_ref() {
            let tokens = batch.input_ids.len();
            if self.moe_frame.as_ref().map(MoeFrame::tokens) != Some(tokens) {
                self.moe_frame = None;
                self.moe_frame = Some(moe.prepare(tokens)?);
            }
        }
        let classifier_sequence = self
            .head
            .as_ref()
            .is_some_and(|head| head.task() == DecoderClassifierTask::Sequence);
        if classifier_sequence
            || (self.moe.is_none()
                && self.selected_readout_enabled
                && batch.raw_indices.is_empty()
                && selected_readout(pool, batch.pooled_indices.len()).is_some())
        {
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
        tokens: usize,
        classifier_plan: Option<&RowPlan>,
    ) -> Result<(), BackendError> {
        if self.kernels.qk_norm_rope.is_some() {
            self.enqueue_impl::<true>(
                lt,
                plain_blas,
                stream,
                work,
                batch,
                pool,
                max_length,
                tokens,
                classifier_plan,
            )
        } else {
            self.enqueue_impl::<false>(
                lt,
                plain_blas,
                stream,
                work,
                batch,
                pool,
                max_length,
                tokens,
                classifier_plan,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue_impl<const QK_NORM: bool>(
        &mut self,
        lt: &LtProvider,
        plain_blas: Option<&Blas>,
        stream: &Stream,
        work: &mut DecoderWork,
        batch: &Batch,
        pool: &Pool,
        max_length: usize,
        tokens: usize,
        classifier_plan: Option<&RowPlan>,
    ) -> Result<(), BackendError> {
        let c = &self.config;
        let g = self.geometry;
        let sequences = batch
            .cumulative_seq_lengths
            .len()
            .checked_sub(1)
            .ok_or_else(|| infer("Missing decoder sequence boundaries"))?;
        let selections = batch.pooled_indices.len();
        work.check(tokens, sequences, super::output::rows(batch)?, &g)?;
        let classifier = self.classifier_mode(classifier_plan);
        // A decoder classifier owns its row policy: sequence heads select
        // before the final pointwise seam, token heads keep the full T matrix.
        // Ordinary pooled embedding requests keep the canonical-B optimization.
        let readout = if classifier.is_none()
            && self.moe.is_none()
            && self.selected_readout_enabled
            && batch.raw_indices.is_empty()
        {
            selected_readout(pool, selections)
        } else {
            None
        };
        let h = g.hidden;
        let rows = u32_size(tokens)?;
        let width = u32_size(h)?;
        let eps = c.rms_norm_eps as f32;
        // Each SiLU thread owns an eight-column group. Prepare both actual
        // full/readout grids once before the layer loop, including tail groups.
        let full_silu_grid = blocks(product(tokens, g.silu_groups_per_row)?)?;
        let readout_silu_grid = blocks(product(sequences, g.silu_groups_per_row)?)?;
        let k = &self.kernels;
        let layout = InputLayout::new(
            tokens,
            batch.cumulative_seq_lengths.len(),
            batch.pooled_indices.len(),
        )?
        .with_trailer(super::output::raw_rows(batch)?)?;
        let base = work.input.cu_deviceptr();
        let mut ids = layout.ids().address(base)?;
        let positions = layout.positions().address(base)?;
        let boundaries = layout.boundaries().address(base)?;
        let selected = layout.selections().address(base)?;
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
            (rms_rows.div_ceil(k.rms_geometry.rows_per_cta), 1, 1),
            k.rms_geometry.threads,
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
        // Request-constant Q/K norm launch preparation. The false arm binds the
        // plain rotary function as a type-compatible placeholder and never
        // reaches the fused launch.
        let rope_kernel = if QK_NORM {
            k.qk_norm_rope
                .as_ref()
                .ok_or_else(|| infer("Qwen3 recipe is missing the Q/K norm RoPE kernel"))?
        } else {
            &k.rope_neox
        };
        let qk_norm_grid = if QK_NORM {
            let head_rows = product(
                tokens,
                c.num_attention_heads
                    .checked_add(c.num_key_value_heads)
                    .ok_or_else(|| infer("decoder Q/K norm head count overflow"))?,
            )?;
            Some((
                u32_size(head_rows.div_ceil(QK_NORM_ROPE_HEAD_ROWS_PER_CTA))?,
                1,
                1,
            ))
        } else {
            None
        };
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
            if QK_NORM {
                let q_norm = layer
                    .q_norm
                    .as_ref()
                    .ok_or_else(|| infer("Qwen3 layer is missing q_norm"))?;
                let k_norm = layer
                    .k_norm
                    .as_ref()
                    .ok_or_else(|| infer("Qwen3 layer is missing k_norm"))?;
                let grid = qk_norm_grid
                    .ok_or_else(|| infer("Qwen3 recipe is missing the Q/K norm launch grid"))?;
                let mut q_gamma = q_norm.cu_deviceptr();
                let mut k_gamma = k_norm.cu_deviceptr();
                let mut qk_eps = eps;
                launch_threads(
                    rope_kernel,
                    stream,
                    grid,
                    QK_NORM_ROPE_THREADS,
                    0,
                    &mut [
                        arg(&mut q),
                        arg(&mut key),
                        arg(&mut q_gamma),
                        arg(&mut k_gamma),
                        arg(&mut cos),
                        arg(&mut sin),
                        arg(&mut position_ptr),
                        arg(&mut rope_tokens),
                        arg(&mut rope_q_heads),
                        arg(&mut rope_kv_heads),
                        arg(&mut q_stride),
                        arg(&mut k_stride),
                        arg(&mut rope_table_stride),
                        arg(&mut qk_eps),
                    ],
                )?;
            } else {
                launch(
                    rope_kernel,
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
            }
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
            let classifier_selected = classifier == Some(ClassifierMode::Selected);
            if (readout.is_some() || classifier_selected) && index + 1 == self.weights.layers.len()
            {
                if classifier_selected {
                    // Explicit classifier rows: gather both the attention-output
                    // Q-width rows and the accumulated residual H rows at the
                    // host-prepared last-non-PAD token ids before o_proj and the
                    // final pointwise seam. No new shader is introduced.
                    for (table, out, row_width, elements) in [
                        (
                            activation,
                            projection,
                            g.q_width,
                            product(sequences, g.q_width)?,
                        ),
                        (
                            residual_cur,
                            residual_next_ptr,
                            width as usize,
                            product(sequences, h)?,
                        ),
                    ] {
                        let mut gather_ids = selected;
                        let mut gather_table = table;
                        let mut gather_out = out;
                        let mut gather_count = u32_size(sequences)?;
                        let mut gather_width = u32_size(row_width)?;
                        launch(
                            &k.gather,
                            stream,
                            blocks(elements)?,
                            0,
                            &mut [
                                arg(&mut gather_ids),
                                arg(&mut gather_table),
                                arg(&mut gather_out),
                                arg(&mut gather_count),
                                arg(&mut gather_width),
                            ],
                        )?;
                    }
                } else {
                    let readout_mode =
                        readout.ok_or_else(|| infer("Missing selected readout mode"))?;
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
                }

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

                let DecoderMlp::Dense { gate_up, down } = &layer.mlp else {
                    return Err(infer("Routed MoE cannot use selected-row MLP execution"));
                };
                if let Some(bias) = gate_up.bias.as_ref() {
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
                            &gate_up.weight,
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
                            &gate_up.weight,
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
                if let Some(bias) = down.bias.as_ref() {
                    let plan = self
                        .plans
                        .readout
                        .bias_down
                        .as_ref()
                        .ok_or_else(|| infer("Missing selected down bias plan"))?;
                    unsafe {
                        lt.enqueue(plan, &work.activation, &down.weight, bias, &mut work.branch)?;
                    }
                } else {
                    let blas = plain_blas
                        .ok_or_else(|| infer("Missing decoder plain cuBLAS projection handle"))?;
                    unsafe {
                        blas.linear(
                            &self.plans.readout.plain_down,
                            &work.activation,
                            &down.weight,
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
                if !classifier_selected {
                    let mut cast_input = normalized;
                    let mut cast_output = work
                        .pooled
                        .as_ref()
                        .ok_or_else(|| infer("Decoder embedding workspace has no pooled slab"))?
                        .cu_deviceptr();
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
                }
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

            match &layer.mlp {
                DecoderMlp::Dense { gate_up, down } => {
                    if let Some(bias) = gate_up.bias.as_ref() {
                        let plan = self
                            .plans
                            .bias_gate_up
                            .as_ref()
                            .ok_or_else(|| infer("Missing decoder gate/up bias plan"))?;
                        unsafe {
                            lt.enqueue(
                                plan,
                                &work.normalized,
                                &gate_up.weight,
                                bias,
                                &mut work.projection,
                            )?;
                        }
                    } else {
                        let blas = plain_blas.ok_or_else(|| {
                            infer("Missing decoder plain cuBLAS projection handle")
                        })?;
                        unsafe {
                            blas.linear(
                                &self.plans.plain.gate_up,
                                &work.normalized,
                                &gate_up.weight,
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
                    if let Some(bias) = down.bias.as_ref() {
                        let plan = self
                            .plans
                            .bias_down
                            .as_ref()
                            .ok_or_else(|| infer("Missing decoder down bias plan"))?;
                        unsafe {
                            lt.enqueue(
                                plan,
                                &work.activation,
                                &down.weight,
                                bias,
                                &mut work.branch,
                            )?;
                        }
                    } else {
                        let blas = plain_blas.ok_or_else(|| {
                            infer("Missing decoder plain cuBLAS projection handle")
                        })?;
                        unsafe {
                            blas.linear(
                                &self.plans.plain.down,
                                &work.activation,
                                &down.weight,
                                &mut work.branch,
                            )?;
                        }
                    }
                }
                DecoderMlp::Routed(layer) => {
                    let provider = self
                        .moe
                        .as_ref()
                        .ok_or_else(|| infer("Missing routed provider"))?;
                    let frame = self
                        .moe_frame
                        .as_ref()
                        .ok_or_else(|| infer("Missing token-exact routed plan"))?;
                    let scratch = work
                        .moe
                        .as_mut()
                        .ok_or_else(|| infer("Missing routed arena"))?;
                    let blas = plain_blas.ok_or_else(|| infer("Missing router cuBLAS handle"))?;
                    // SAFETY: Engine binds this context and retains model/work until its stream completes.
                    unsafe {
                        provider.forward(
                            frame,
                            layer,
                            blas,
                            &k.bf16_to_f32,
                            stream,
                            &work.normalized,
                            &mut work.branch,
                            scratch,
                        )?;
                    }
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

            if classifier.is_none() {
                let route = mean_pool_route(pool, selections)?;
                if route != MeanPoolRoute::Empty {
                    let mut pool_input = normalized;
                    let mut pool_bounds = boundaries;
                    let mut pool_indices = selected;
                    let mut pool_out = work
                        .pooled
                        .as_ref()
                        .ok_or_else(|| infer("Decoder embedding workspace has no pooled slab"))?
                        .cu_deviceptr();
                    match route {
                        MeanPoolRoute::Empty => {}
                        MeanPoolRoute::NativeBf16Tree => {
                            let mean = k.mean_kernel.as_ref().ok_or_else(|| {
                                start(format!(
                                    "BF16 decoder mean pooling requires the public Gemma2 ABI bundle export {PUBLIC_MEAN_EXPORT} or the qualified routed-native mean export"
                                ))
                            })?;
                            let plan = crate::gemma2_math::MeanPlan::new(
                                tokens,
                                h,
                                &batch.cumulative_seq_lengths,
                                &batch.pooled_indices,
                            )
                            .map_err(|error| start(format!("decoder BF16 mean plan: {error}")))?;
                            let geometry = plan.geometry().map_err(|error| {
                                start(format!("decoder BF16 mean grid: {error}"))
                            })?;
                            let mut mean_count = plan.count();
                            let mut mean_width = plan.width();
                            launch_threads(
                                mean,
                                stream,
                                (geometry.grid()[0], geometry.grid()[1], geometry.grid()[2]),
                                geometry.block()[0],
                                0,
                                &mut [
                                    arg(&mut pool_input),
                                    arg(&mut pool_bounds),
                                    arg(&mut pool_indices),
                                    arg(&mut pool_out),
                                    arg(&mut mean_count),
                                    arg(&mut mean_width),
                                ],
                            )?;
                        }
                        MeanPoolRoute::F32Fallback => {
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
                    }
                }
                if !batch.raw_indices.is_empty() {
                    let pooled = work
                        .pooled
                        .as_ref()
                        .ok_or_else(|| infer("Decoder embedding workspace has no pooled slab"))?;
                    super::enqueue_raw_output(
                        &k.bf16_to_f32,
                        stream,
                        normalized,
                        pooled.cu_deviceptr(),
                        layout.trailer().address(base)?,
                        batch,
                        h,
                    )?;
                }
            }
        }
        if let (Some(plan), Some(classifier_mode)) = (classifier_plan, classifier) {
            let head = self
                .head
                .as_mut()
                .ok_or_else(|| infer("Decoder classifier head is not configured"))?;
            let blas = plain_blas
                .ok_or_else(|| infer("Decoder classifier requires a plain cuBLAS handle"))?;
            let source = DecoderClassifierRows::new(&work.normalized, plan.source_rows(), h)?;
            let row_ids = match classifier_mode {
                ClassifierMode::Selected => None,
                ClassifierMode::FullSequenceGather => Some(layout.selections().address(base)?),
                ClassifierMode::FullTokens | ClassifierMode::FullSequenceMean => None,
            };
            let mean = match classifier_mode {
                ClassifierMode::FullSequenceMean => {
                    let plan_mean = crate::gemma2_math::MeanPlan::new(
                        tokens,
                        h,
                        &batch.cumulative_seq_lengths,
                        &batch.pooled_indices,
                    )
                    .map_err(|error| infer(format!("decoder classifier mean plan: {error}")))?;
                    let geometry = plan_mean
                        .geometry()
                        .map_err(|error| infer(format!("decoder classifier mean grid: {error}")))?;
                    Some(MeanRows {
                        boundaries: layout.boundaries().address(base)?,
                        indices: layout.selections().address(base)?,
                        grid: [geometry.grid()[0], geometry.grid()[1], geometry.grid()[2]],
                        threads: geometry.block()[0],
                    })
                }
                _ => None,
            };
            let scratch = work
                .classifier
                .as_deref_mut()
                .ok_or_else(|| infer("Decoder classifier scratch is missing"))?;
            // SAFETY: every buffer is fixed by the reserved work arena and the
            // caller retains all owners through the ordered stream completion.
            unsafe { head.enqueue(lt, blas, stream, source, row_ids, mean, scratch)? };
        }
        Ok(())
    }
}

/// Classifier requests read F32 logits from head scratch and never run pooled
/// or raw F32 readout, so their physical pooled allocation is zero. Embedding
/// works keep the existing `[selections,H]` F32 owner unchanged.
fn pooled_elements(
    selections: usize,
    hidden: usize,
    has_classifier: bool,
) -> Result<usize, BackendError> {
    if has_classifier {
        Ok(0)
    } else {
        product(selections, hidden)
    }
}

pub(crate) struct DecoderWork {
    pub(crate) stream: Arc<Stream>,
    tokens: usize,
    sequences: usize,
    selections: usize,
    bytes: usize,
    pub(crate) input: ManuallyDrop<DeviceBuffer<u32>>,
    pub(crate) residual: ManuallyDrop<DeviceBuffer<Bf16>>,
    pub(crate) residual_next: ManuallyDrop<DeviceBuffer<Bf16>>,
    pub(crate) branch: ManuallyDrop<DeviceBuffer<Bf16>>,
    pub(crate) normalized: ManuallyDrop<DeviceBuffer<Bf16>>,
    pub(crate) projection: ManuallyDrop<DeviceBuffer<Bf16>>,
    pub(crate) activation: ManuallyDrop<DeviceBuffer<Bf16>>,
    pooled: Option<ManuallyDrop<DeviceBuffer<f32>>>,
    classifier: Option<Box<DecoderClassifierScratchBuffers>>,
    moe: Option<MoeScratch>,
    last_completion: Option<CompletionTicket>,
    released: bool,
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
    #[allow(dead_code)]
    pub(crate) fn allocation_bytes(key: WorkKey, c: &DecoderConfig) -> Result<usize, BackendError> {
        Self::allocation_bytes_with_classifier(key, c, None)
    }

    pub(crate) fn allocation_bytes_with_classifier(
        key: WorkKey,
        c: &DecoderConfig,
        classifier: Option<DecoderClassifierScratchLayout>,
    ) -> Result<usize, BackendError> {
        Self::allocation_bytes_with_geometry(key, &DecoderGeometry::new(c)?, classifier)
    }

    pub(crate) fn allocation_bytes_geometry(
        key: WorkKey,
        geometry: &DecoderGeometry,
    ) -> Result<usize, BackendError> {
        Self::allocation_bytes_with_geometry(key, geometry, None)
    }

    fn allocation_bytes_with_geometry(
        key: WorkKey,
        geometry: &DecoderGeometry,
        classifier: Option<DecoderClassifierScratchLayout>,
    ) -> Result<usize, BackendError> {
        let (tokens, _, selections) = key;
        i32::try_from(tokens).map_err(infer)?;
        let hidden = product(tokens, geometry.hidden)?;
        let projection = product(tokens, geometry.projection_width)?;
        let activation = product(tokens, geometry.activation_width)?;
        let input = staging::input_capacity(key)?;
        let output = pooled_elements(selections, geometry.hidden, classifier.is_some())?;
        let classifier_bytes = classifier
            .map(DecoderClassifierScratchLayout::bytes)
            .transpose()?;
        product(input, 4)?
            .checked_add(product(output, 4)?)
            .and_then(|n| n.checked_add(product(hidden, 8).ok()?))
            .and_then(|n| n.checked_add(product(projection, 2).ok()?))
            .and_then(|n| n.checked_add(product(activation, 2).ok()?))
            .and_then(|n| n.checked_add(classifier_bytes.unwrap_or(0)))
            .ok_or_else(|| infer("Decoder workspace arena overflow"))
    }
    #[allow(dead_code)]
    pub(crate) fn new(
        stream: Arc<Stream>,
        c: &DecoderConfig,
        key: WorkKey,
    ) -> Result<Self, BackendError> {
        Self::new_with_classifier(stream, c, key, None)
    }
    pub(crate) fn new_with_classifier(
        stream: Arc<Stream>,
        c: &DecoderConfig,
        key: WorkKey,
        classifier_layout: Option<DecoderClassifierScratchLayout>,
    ) -> Result<Self, BackendError> {
        Self::new_with_geometry(stream, &DecoderGeometry::new(c)?, key, classifier_layout)
    }

    pub(crate) fn new_geometry(
        stream: Arc<Stream>,
        geometry: &DecoderGeometry,
        key: WorkKey,
    ) -> Result<Self, BackendError> {
        Self::new_with_geometry(stream, geometry, key, None)
    }

    fn new_with_geometry(
        stream: Arc<Stream>,
        geometry: &DecoderGeometry,
        key: WorkKey,
        classifier_layout: Option<DecoderClassifierScratchLayout>,
    ) -> Result<Self, BackendError> {
        let bytes = Self::allocation_bytes_with_geometry(key, geometry, classifier_layout)?;
        let (tokens, sequences, selections) = key;
        let input_len = staging::input_capacity(key)?;
        let hidden = product(tokens, geometry.hidden)?;
        let projection = product(tokens, geometry.projection_width)?;
        let activation = product(tokens, geometry.activation_width)?;
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
        let pooled = if classifier_layout.is_some() {
            None
        } else {
            let output = product(selections, geometry.hidden)?;
            // SAFETY: pooling writes every selected element before download.
            let pooled = unsafe { DeviceBuffer::<f32>::uninitialized_async(&stream, output) }
                .map_err(infer)?;
            Some(ManuallyDrop::new(pooled))
        };
        let classifier = classifier_layout
            .map(|layout| {
                DecoderClassifierScratchBuffers::new(stream.clone(), layout).map(Box::new)
            })
            .transpose()?;
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
            pooled,
            classifier,
            moe: None,
            last_completion: None,
            released: false,
            stream,
        })
    }
    /// Records the tracked completion generation for this Work's last use. The
    /// engine calls this before the first GPU upload of a submission.
    pub(crate) fn mark_last_use(&mut self, ticket: CompletionTicket) {
        self.last_completion = Some(ticket);
    }

    pub(crate) fn completion_ticket(&self) -> Option<CompletionTicket> {
        self.last_completion.clone()
    }

    /// Releases only when the last tracked submission has an actual completion
    /// proof. A ticket-less owner was never marked before an upload and is kept
    /// for the final stream drain rather than dropped without proof.
    pub(crate) unsafe fn release_if_idle(&mut self) -> Result<bool, BackendError> {
        match self.last_completion.as_ref() {
            None => Ok(false),
            Some(ticket) => match ticket.completed() {
                Some(proof) => {
                    // SAFETY: the exact last-use proof is rechecked below.
                    unsafe { self.release_after_completion(&proof)? };
                    Ok(true)
                }
                None => Ok(false),
            },
        }
    }

    /// Releases Decoder-owned buffers after an exact same-stream proof.
    ///
    /// # Safety
    /// The proof MUST come from the actual fence for this Work's tracked
    /// ticket. On any identity mismatch the payload stays owned.
    pub(crate) unsafe fn release_after_completion(
        &mut self,
        proof: &CompletedTicket,
    ) -> Result<(), BackendError> {
        let ticket = self
            .last_completion
            .as_ref()
            .ok_or_else(|| infer("Decoder Work has no completion ticket"))?;
        let stream_identity = Arc::as_ptr(&self.stream) as usize;
        if !proof.matches(ticket)
            || proof.stream_identity() != stream_identity
            || ticket.stream_identity() != stream_identity
        {
            return Err(infer(
                "Completion proof does not match this Decoder Work request",
            ));
        }
        if !self.released {
            // SAFETY: the caller proved the actual same-stream fence for this
            // exact ticket, so no borrowed owner is still in use.
            unsafe { self.release_payload_once() };
        }
        Ok(())
    }

    /// Releases this Work's owned payload after the caller drained the whole
    /// producing stream. A failed drain MUST NOT call this method.
    ///
    /// # Safety
    /// The caller MUST have observed a successful whole-stream drain covering
    /// every use of this Work.
    pub(crate) unsafe fn release_after_drain(&mut self) {
        if !self.released {
            // SAFETY: the caller drained the producing stream for this owner.
            unsafe { self.release_payload_once() };
        }
    }

    unsafe fn release_payload_once(&mut self) {
        if self.released {
            return;
        }
        self.released = true;
        // SAFETY: exactly one owner drops the payload, and this method is only
        // called after a completion proof or an actual stream drain. The MoE
        // scratch shares this arena's retained stream and is released first
        // without a second drain.
        unsafe {
            if let Some(scratch) = self.moe.as_mut() {
                scratch.drop_after_drain();
            }
            ManuallyDrop::drop(&mut self.input);
            ManuallyDrop::drop(&mut self.residual);
            ManuallyDrop::drop(&mut self.residual_next);
            ManuallyDrop::drop(&mut self.branch);
            ManuallyDrop::drop(&mut self.normalized);
            ManuallyDrop::drop(&mut self.projection);
            ManuallyDrop::drop(&mut self.activation);
            if let Some(pooled) = self.pooled.as_mut() {
                ManuallyDrop::drop(pooled);
            }
        }
        if let Some(mut scratch) = self.classifier.take() {
            // SAFETY: this owner is only released after an exact completion
            // proof or a successful whole-stream drain, so the reused outer
            // marks cover the head scratch without another warm synchronize.
            unsafe { scratch.mark_stream_drained() };
            drop(scratch);
        }
        super::LIVE_ARENA_BYTES.fetch_sub(self.bytes, Ordering::Relaxed);
        super::record_live_arena_bytes();
    }
    pub(crate) fn check(
        &self,
        tokens: usize,
        sequences: usize,
        selections: usize,
        g: &DecoderGeometry,
    ) -> Result<(), BackendError> {
        if tokens > self.tokens || sequences > self.sequences || selections > self.selections {
            return Err(infer("Decoder request exceeds retained arena capacity"));
        }
        let input = InputLayout::new(
            tokens,
            sequences
                .checked_add(1)
                .ok_or_else(|| infer("Decoder boundary count overflow"))?,
            selections,
        )?
        .words();
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
        ] {
            if elements > len {
                return Err(infer("Decoder slab exceeds retained arena"));
            }
        }
        if let Some(pooled) = self.pooled.as_ref() {
            if product(selections, g.hidden)? > pooled.len() {
                return Err(infer("Decoder slab exceeds retained arena"));
            }
        }
        Ok(())
    }
    pub(crate) fn stage(
        &self,
        b: &Batch,
        geometry: super::output::OutputGeometry,
        lease: &mut HostIoLease,
    ) -> Result<(usize, usize), BackendError> {
        let pooled = self
            .pooled
            .as_ref()
            .ok_or_else(|| infer("Decoder embedding workspace has no pooled slab"))?;
        if geometry.backbone_elements(b)? > pooled.len() {
            return Err(infer("Decoder backbone output exceeds retained arena"));
        }
        stage_batch(b, geometry, lease)
    }
    pub(crate) fn upload(&self, host: &[u32], elements: usize) -> Result<(), BackendError> {
        upload_input(&self.stream, &self.input, host, elements)
    }
    pub(crate) fn stage_classifier(
        &self,
        b: &Batch,
        geometry: super::output::OutputGeometry,
        lease: &mut HostIoLease,
    ) -> Result<(usize, usize), BackendError> {
        stage_batch(b, geometry, lease)
    }
    pub(crate) fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }
    pub(crate) fn pooled(&self) -> Result<&DeviceBuffer<f32>, BackendError> {
        self.pooled
            .as_deref()
            .ok_or_else(|| infer("Decoder embedding workspace has no pooled slab"))
    }
    pub(crate) fn classifier_scratch(&self) -> Option<&DecoderClassifierScratchBuffers> {
        self.classifier.as_deref()
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn finish(
        &mut self,
        dense: Option<(&mut DenseHead, &mut super::DenseWork)>,
        lt: &LtProvider,
        plain_blas: Option<&Blas>,
        pooled_rows: usize,
        raw_rows: usize,
        backbone_width: usize,
        lease: &mut HostIoLease,
    ) -> Result<(), BackendError> {
        let Self { pooled, stream, .. } = self;
        let pooled = pooled
            .as_ref()
            .ok_or_else(|| infer("Decoder embedding workspace has no pooled slab"))?;
        super::finish_pooled(
            dense,
            lt,
            plain_blas,
            stream,
            pooled,
            pooled_rows,
            raw_rows,
            backbone_width,
            lease,
        )
    }
}
impl Drop for DecoderWork {
    fn drop(&mut self) {
        if self.released {
            return;
        }
        #[cfg(test)]
        crate::note_default_drop_drain();
        if self.stream.synchronize().is_ok() {
            // SAFETY: this destructor just drained the owning stream. Failed
            // drains deliberately retain device allocations.
            unsafe { self.release_payload_once() };
        } else {
            // The Work is the single byte owner and already includes the
            // classifier scratch in self.bytes. Retain the scratch and its
            // stream as part of that charge without a second sync/charge.
            if let Some(scratch) = self.moe.take() {
                std::mem::forget(scratch);
            }
            metrics::counter!("oxide_arena_retained_bytes_total").increment(self.bytes as u64);
            if let Some(scratch) = self.classifier.take() {
                scratch.retain_owned_by_failed_work();
            }
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
    fn mistral_requires_the_current_public_gemma2_exports_without_private_fallback() {
        assert!(crate::gemma2_math::EXPORTS.contains(&MISTRAL_PUBLIC_ROPE_EXPORT));
        assert!(crate::gemma2_math::EXPORTS.contains(&PUBLIC_MEAN_EXPORT));
        assert_eq!(
            MISTRAL_PUBLIC_ROPE_EXPORT,
            "infinity_gemma2_rope_tables_bf16"
        );
        assert_eq!(PUBLIC_MEAN_EXPORT, "infinity_gemma2_mean_pool_bf16");
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

    fn qwen3_fixture(overrides: serde_json::Value) -> Fixture {
        let mut value = serde_json::json!({
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 151936,
            "max_position_embeddings": 40960,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "hidden_act": "silu"
        });
        for (key, field) in overrides.as_object().unwrap() {
            value[key] = field.clone();
        }
        Fixture::with_config(&serde_json::to_string(&value).unwrap())
    }

    #[test]
    fn qwen3_math_binds_native_and_ordinary_recipes_with_fused_qk_norm() {
        let h1024 = qwen3_fixture(serde_json::json!({}));
        let c = h1024.config();
        assert_eq!(c.family, DecoderFamily::Qwen3);
        assert_eq!(
            DecoderMath::select(&c).unwrap(),
            DecoderMath::Qwen3H1024Candle
        );
        assert_eq!(
            DecoderMath::Qwen3H1024Candle.rms_export(),
            "infinity_rms_norm_1024_exact_vector"
        );
        assert_eq!(
            DecoderMath::Qwen3H1024Candle.add_rms_export(),
            "infinity_add_rms_norm_1024_exact_vector"
        );
        assert_eq!(
            DecoderMath::Qwen3H1024Candle.qk_norm_rope_export(),
            Some("infinity_qk_norm_rope_128_bf16")
        );
        assert_eq!(
            DecoderMath::Qwen3H1024Candle.norm_geometry(),
            H1024_NORM_GEOMETRY
        );
        assert_eq!(H1024_NORM_GEOMETRY.rows_per_cta, 4);
        assert_eq!(H1024_NORM_GEOMETRY.threads, 128);
        assert_eq!(QK_NORM_ROPE_THREADS, 128);
        assert_eq!(QK_NORM_ROPE_HEAD_ROWS_PER_CTA, 8);
        let g = DecoderGeometry::new(&c).expect("geometry");
        assert_eq!(
            (g.hidden, g.q_width, g.kv_width, g.qkv_width),
            (1024, 2048, 1024, 4096)
        );
        assert_eq!(g.projection_width, 6144);
        assert_eq!(g.activation_width, 3072);
        assert_eq!(g.silu_groups_per_row, 384);

        let h2048 = qwen3_fixture(serde_json::json!({
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "num_hidden_layers": 28,
            "num_attention_heads": 16
        }));
        let c = h2048.config();
        assert_eq!(
            DecoderMath::select(&c).unwrap(),
            DecoderMath::Qwen3H2048Ordinary
        );
        assert_eq!(
            DecoderMath::Qwen3H2048Ordinary.rms_export(),
            "infinity_rms_norm"
        );
        assert_eq!(
            DecoderMath::Qwen3H2048Ordinary.add_rms_export(),
            "infinity_add_rms_norm"
        );
        assert_eq!(
            DecoderMath::Qwen3H2048Ordinary.qk_norm_rope_export(),
            Some("infinity_qk_norm_rope_128_bf16")
        );
        assert_eq!(
            DecoderMath::Qwen3H2048Ordinary.activation_export(),
            "infinity_silu_mul_bf16_vector8_native"
        );
        assert_eq!(
            DecoderMath::Qwen3H2048Ordinary.norm_geometry(),
            GENERIC_NORM_GEOMETRY
        );
        assert_eq!(GENERIC_NORM_GEOMETRY.rows_per_cta, 1);
        assert_eq!(GENERIC_NORM_GEOMETRY.threads, 256);
        let g = DecoderGeometry::new(&c).expect("geometry");
        assert_eq!(
            (g.hidden, g.q_width, g.kv_width, g.qkv_width),
            (2048, 2048, 1024, 4096)
        );
        assert_eq!(g.projection_width, 12288);
        assert_eq!(g.activation_width, 6144);
        assert_eq!(g.silu_groups_per_row, 768);

        let h2560 = qwen3_fixture(serde_json::json!({
            "hidden_size": 2560,
            "intermediate_size": 9728,
            "num_hidden_layers": 36,
            "num_attention_heads": 32
        }));
        let c = h2560.config();
        assert_eq!(
            DecoderMath::select(&c).unwrap(),
            DecoderMath::Qwen3H2560Ordinary
        );
        assert_eq!(
            DecoderMath::Qwen3H2560Ordinary.rms_export(),
            "infinity_rms_norm"
        );
        assert_eq!(
            DecoderMath::Qwen3H2560Ordinary.add_rms_export(),
            "infinity_add_rms_norm"
        );
        assert_eq!(
            DecoderMath::Qwen3H2560Ordinary.qk_norm_rope_export(),
            Some("infinity_qk_norm_rope_128_bf16")
        );
        assert_eq!(
            DecoderMath::Qwen3H2560Ordinary.activation_export(),
            "infinity_silu_mul_bf16_vector8_native"
        );
        assert_eq!(
            DecoderMath::Qwen3H2560Ordinary.norm_geometry(),
            GENERIC_NORM_GEOMETRY
        );
        let g = DecoderGeometry::new(&c).expect("geometry");
        assert_eq!(
            (g.hidden, g.q_width, g.kv_width, g.qkv_width),
            (2560, 4096, 1024, 6144)
        );
        assert_eq!(g.projection_width, 19456);
        assert_eq!(g.activation_width, 9728);
        assert_eq!(g.silu_groups_per_row, 1216);
        assert!(c.q_width() != c.hidden_size);

        let h4096 = qwen3_fixture(serde_json::json!({
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_hidden_layers": 36,
            "num_attention_heads": 32
        }));
        let c = h4096.config();
        assert_eq!(
            DecoderMath::select(&c).unwrap(),
            DecoderMath::Qwen3H4096Ordinary
        );
        assert_eq!(
            DecoderMath::Qwen3H4096Ordinary.rms_export(),
            "infinity_rms_norm_4096_vector"
        );
        assert_eq!(
            DecoderMath::Qwen3H4096Ordinary.add_rms_export(),
            "infinity_add_rms_norm_4096_vector"
        );
        assert_eq!(
            DecoderMath::Qwen3H4096Ordinary.qk_norm_rope_export(),
            Some("infinity_qk_norm_rope_128_bf16")
        );
        assert_eq!(
            DecoderMath::Qwen3H4096Ordinary.norm_geometry(),
            GENERIC_NORM_GEOMETRY
        );
        let g = DecoderGeometry::new(&c).expect("geometry");
        assert_eq!(
            (g.hidden, g.q_width, g.kv_width, g.qkv_width),
            (4096, 4096, 1024, 6144)
        );
        assert_eq!(g.projection_width, 24576);
        assert_eq!(g.activation_width, 12288);
        for (heads, kv_heads) in [(16, 8), (32, 8)] {
            for causal in [false, true] {
                assert!(fa4::Fa4::supports(&fa4::AttentionSpec {
                    heads,
                    kv_heads,
                    head_dim: 128,
                    causal,
                    window: fa4::Window::Global,
                    dtype: fa4::DType::Bf16,
                    scale: fa4::Scale::RsqrtHeadDim,
                }));
            }
        }
    }

    #[test]
    fn qwen3_h5120_qualified_aten_rms_keeps_add256_and_checked_qkv_grouping() {
        for (heads, intermediate, expected) in [
            (
                40usize,
                17408usize,
                (
                    5120usize, 5120usize, 1024usize, 7168usize, 34816usize, 17408usize, 2176usize,
                ),
            ),
            (64, 25600, (5120, 8192, 1024, 10240, 51200, 25600, 3200)),
        ] {
            let fixture = qwen3_fixture(serde_json::json!({
                "hidden_size": 5120,
                "intermediate_size": intermediate,
                "num_attention_heads": heads,
                "num_key_value_heads": 8,
                "head_dim": 128
            }));
            let c = fixture.config();
            assert_eq!(
                DecoderMath::select(&c).unwrap(),
                DecoderMath::Qwen3H5120Ordinary
            );
            assert_eq!(
                DecoderMath::Qwen3H5120Ordinary.rms_export(),
                "infinity_rms_norm_5120_aten"
            );
            assert_eq!(
                DecoderMath::Qwen3H5120Ordinary.add_rms_export(),
                "infinity_add_rms_norm"
            );
            assert_eq!(
                DecoderMath::Qwen3H5120Ordinary.rope_export(),
                "infinity_rope_neox"
            );
            assert_eq!(
                DecoderMath::Qwen3H5120Ordinary.qk_norm_rope_export(),
                Some("infinity_qk_norm_rope_128_bf16")
            );
            assert_eq!(
                DecoderMath::Qwen3H5120Ordinary.activation_export(),
                "infinity_silu_mul_bf16_vector8_native"
            );
            assert_eq!(
                DecoderMath::Qwen3H5120Ordinary.norm_geometry(),
                GENERIC_NORM_GEOMETRY
            );
            assert_eq!(GENERIC_NORM_GEOMETRY.rows_per_cta, 1);
            assert_eq!(GENERIC_NORM_GEOMETRY.threads, 256);
            let g = DecoderGeometry::new(&c).expect("geometry");
            assert_eq!(
                (
                    g.hidden,
                    g.q_width,
                    g.kv_width,
                    g.qkv_width,
                    g.projection_width,
                    g.activation_width,
                    g.silu_groups_per_row
                ),
                expected
            );
            assert_eq!(
                DecoderMath::Qwen3H5120Ordinary.rms_geometry(),
                H5120_RMS_GEOMETRY
            );
            assert_eq!(DecoderMath::Qwen3H5120Ordinary.rms_geometry().threads, 128);
            assert_eq!(DecoderMath::Qwen3H5120Ordinary.norm_geometry().threads, 256);
            assert_eq!(c.q_width(), heads * 128);
            assert_eq!(c.kv_width(), 1024);
            for causal in [false, true] {
                let mut requested = c.clone();
                requested.is_causal = causal;
                let spec = attention_spec(&requested).expect("attention spec");
                assert_eq!(spec.heads, heads);
                assert_eq!(spec.kv_heads, 8);
                assert_eq!(spec.head_dim, 128);
                assert_eq!(spec.causal, causal);
                assert_eq!(spec.window, fa4::Window::Global);
                assert_eq!(spec.dtype, fa4::DType::Bf16);
                assert_eq!(spec.scale, fa4::Scale::RsqrtHeadDim);
            }
        }
    }

    #[test]
    fn qwen3_h5120_admission_is_limited_to_the_two_native_topologies() {
        for (heads, kv_heads, needle) in [
            (32usize, 8usize, "Q32/KV8"),
            (40, 4, "Q40/KV4"),
            (48, 8, "Q48/KV8"),
        ] {
            let fixture = qwen3_fixture(serde_json::json!({
                "hidden_size": 5120,
                "intermediate_size": 17408,
                "num_attention_heads": heads,
                "num_key_value_heads": kv_heads,
                "head_dim": 128
            }));
            let error = validate_recipe(&fixture.config()).unwrap_err().to_string();
            assert!(error.contains("Qwen3 H5120 admits only"), "{error}");
            assert!(error.contains(needle), "{error}");
        }

        let d256 = qwen3_fixture(serde_json::json!({
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "num_attention_heads": 40,
            "num_key_value_heads": 8,
            "head_dim": 256
        }));
        let error = validate_recipe(&d256.config()).unwrap_err().to_string();
        assert!(error.contains("only Qwen3 head_dim 128"), "{error}");

        let h6144 = qwen3_fixture(serde_json::json!({
            "hidden_size": 6144,
            "intermediate_size": 17408,
            "num_attention_heads": 40,
            "num_key_value_heads": 8,
            "head_dim": 128
        }));
        let error = validate_recipe(&h6144.config()).unwrap_err().to_string();
        assert!(error.contains("4096 or 5120"), "{error}");

        for (hidden, intermediate, heads, expected) in [
            (1024usize, 3072usize, 16usize, DecoderMath::Qwen3H1024Candle),
            (2048, 6144, 16, DecoderMath::Qwen3H2048Ordinary),
            (2560, 9728, 32, DecoderMath::Qwen3H2560Ordinary),
            (4096, 12288, 32, DecoderMath::Qwen3H4096Ordinary),
        ] {
            let fixture = qwen3_fixture(serde_json::json!({
                "hidden_size": hidden,
                "intermediate_size": intermediate,
                "num_attention_heads": heads,
                "num_key_value_heads": 8,
                "head_dim": 128
            }));
            assert_eq!(DecoderMath::select(&fixture.config()).unwrap(), expected);
        }
    }

    #[test]
    fn qwen3_h5120_workspace_bytes_follow_query_width_and_positive_intermediate() {
        let q40 = qwen3_fixture(serde_json::json!({
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "num_attention_heads": 40,
            "num_key_value_heads": 8,
            "head_dim": 128
        }));
        let q64 = qwen3_fixture(serde_json::json!({
            "hidden_size": 5120,
            "intermediate_size": 25600,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "head_dim": 128
        }));
        let key = (5usize, 2, 1);
        for fixture in [&q40, &q64] {
            let c = fixture.config();
            let g = DecoderGeometry::new(&c).expect("geometry");
            let input = staging::input_capacity(key).unwrap();
            let expected = input * 4
                + key.2 * g.hidden * 4
                + key.0 * g.hidden * 8
                + key.0 * g.projection_width * 2
                + key.0 * g.activation_width * 2;
            assert_eq!(
                DecoderWork::allocation_bytes(key, &c).expect("bytes"),
                expected
            );
            let strides = g.attention_strides(c.head_dim).expect("strides");
            assert_eq!(
                &strides[0..3],
                &[i64::try_from(g.qkv_width).unwrap(), 128, 1][..]
            );
            assert_eq!(strides[9], i64::try_from(g.q_width).unwrap());
        }

        let mut narrow40 = q40.config();
        narrow40.intermediate_size = 1024;
        let mut narrow64 = q64.config();
        narrow64.intermediate_size = 1024;
        let bytes40 = DecoderWork::allocation_bytes(key, &narrow40).expect("q40 bytes");
        let bytes64 = DecoderWork::allocation_bytes(key, &narrow64).expect("q64 bytes");
        assert_eq!(
            bytes64 - bytes40,
            key.0 * ((10240 - 7168) + (8192 - 5120)) * 2,
            "projection and activation slabs follow q_width, not hidden_size"
        );

        let mut overflow = q40.config();
        overflow.intermediate_size = usize::MAX;
        assert!(DecoderGeometry::new(&overflow).is_err());
        assert!(DecoderWork::allocation_bytes(key, &overflow).is_err());
        assert!(DecoderWork::allocation_bytes((usize::MAX, 2, 1), &q40.config()).is_err());
    }

    #[test]
    fn qwen3_4b_actual_config_selects_h2560_ordinary_geometry() {
        // Exact metadata from the Qwen3-4B BF16/FP8 comparison checkpoint.
        let fixture = Fixture::with_config(
            r#"{
                "architectures": ["Qwen3ForCausalLM"],
                "attention_bias": false,
                "attention_dropout": 0.0,
                "bos_token_id": 151643,
                "eos_token_id": 151645,
                "head_dim": 128,
                "hidden_act": "silu",
                "hidden_size": 2560,
                "initializer_range": 0.02,
                "intermediate_size": 9728,
                "max_position_embeddings": 40960,
                "max_window_layers": 36,
                "model_type": "qwen3",
                "num_attention_heads": 32,
                "num_hidden_layers": 36,
                "num_key_value_heads": 8,
                "rms_norm_eps": 1e-06,
                "rope_scaling": null,
                "rope_theta": 1000000,
                "sliding_window": null,
                "tie_word_embeddings": true,
                "torch_dtype": "bfloat16",
                "transformers_version": "4.51.2",
                "use_cache": true,
                "use_sliding_window": false,
                "vocab_size": 151665
            }"#,
        );
        let c = fixture.config();
        assert_eq!(c.family, DecoderFamily::Qwen3);
        assert_eq!(
            (c.hidden_size, c.intermediate_size, c.num_hidden_layers),
            (2560, 9728, 36)
        );
        assert_eq!(
            (c.num_attention_heads, c.num_key_value_heads, c.head_dim),
            (32, 8, 128)
        );
        assert_eq!(c.max_position_embeddings, 40960);
        assert_eq!(c.rope_theta, 1_000_000.0);
        assert!(c.is_causal);
        assert_eq!(c.rope, DecoderRope::FullNeoX);
        assert_eq!(
            DecoderMath::select(&c).unwrap(),
            DecoderMath::Qwen3H2560Ordinary
        );
        let g = DecoderGeometry::new(&c).expect("geometry");
        assert_eq!(
            (g.hidden, g.q_width, g.kv_width, g.qkv_width),
            (2560, 4096, 1024, 6144)
        );
        assert_eq!(g.projection_width, 19456);
        assert_eq!(g.activation_width, 9728);
        assert_eq!(g.silu_groups_per_row, 1216);
        assert!(c.q_width() != c.hidden_size);
        for causal in [false, true] {
            assert!(fa4::Fa4::supports(&fa4::AttentionSpec {
                heads: c.num_attention_heads,
                kv_heads: c.num_key_value_heads,
                head_dim: c.head_dim,
                causal,
                window: fa4::Window::Global,
                dtype: fa4::DType::Bf16,
                scale: fa4::Scale::RsqrtHeadDim,
            }));
        }
    }

    #[test]
    fn decoder_auto_prefix_keeps_published_and_falls_back_for_appended() {
        let published = fa4::AttentionSpec {
            heads: 32,
            kv_heads: 8,
            head_dim: 128,
            causal: true,
            window: fa4::Window::Local { left: 64, right: 0 },
            dtype: fa4::DType::Bf16,
            scale: fa4::Scale::RsqrtHeadDim,
        };
        assert!(fa4::Fa4::supports_published(&published));
        let appended = fa4::AttentionSpec {
            heads: 32,
            kv_heads: 4,
            head_dim: 128,
            causal: true,
            window: fa4::Window::Global,
            dtype: fa4::DType::Bf16,
            scale: fa4::Scale::RsqrtHeadDim,
        };
        assert!(fa4::Fa4::supports(&appended));
        assert!(!fa4::Fa4::supports_published(&appended));
    }

    #[test]
    fn qwen3_rejects_unqualified_widths_heads_and_rope_recipes() {
        for (overrides, needle) in [
            (
                serde_json::json!({"head_dim": 64}),
                "only Qwen3 head_dim 128",
            ),
            (
                serde_json::json!({"head_dim": 256}),
                "only Qwen3 head_dim 128",
            ),
            (
                serde_json::json!({"hidden_size": 6144}),
                "hidden_size 1024, 2048, 2560, 4096 or 5120",
            ),
            (
                serde_json::json!({"rope_type": "simple"}),
                "native Qwen3 full split-half NeoX RoPE",
            ),
        ] {
            let fixture = qwen3_fixture(overrides);
            let error = validate_recipe(&fixture.config()).unwrap_err().to_string();
            assert!(error.contains(needle), "{error}");
        }
    }

    #[test]
    fn qwen2_h896_d64_selects_the_masked_candle_recipe() {
        let base = Fixture::new();
        let mut config: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(base.0.join("config.json")).unwrap()).unwrap();
        config["hidden_size"] = 896.into();
        for (head_dim, heads) in [(64, 14), (128, 7)] {
            config["head_dim"] = head_dim.into();
            config["num_attention_heads"] = heads.into();
            config["num_key_value_heads"] = (if heads == 14 { 2 } else { 1 }).into();
            for causal in [false, true] {
                config["is_causal"] = causal.into();
                let fixture = Fixture::with_config(&config.to_string());
                let loaded = fixture.config();
                let math = DecoderMath::select(&loaded).unwrap();
                if head_dim == 64 {
                    assert_eq!(math, DecoderMath::Qwen2H896MaskedCandle);
                    assert_eq!(
                        math.rms_export(),
                        "infinity_rms_norm_896_exact_masked_vector"
                    );
                    assert_eq!(
                        math.add_rms_export(),
                        "infinity_add_rms_norm_896_exact_masked_vector"
                    );
                    assert_eq!(math.rope_export(), "infinity_rope_neox_hfma");
                    assert_eq!(math.qk_norm_rope_export(), None);
                    assert_eq!(math.norm_geometry(), H896_NORM_GEOMETRY);
                } else {
                    assert_eq!(math, DecoderMath::Qwen2FullNeox);
                    assert_eq!(math.rms_export(), "infinity_rms_norm");
                }
            }
        }
    }

    #[test]
    fn documented_dense_gap_configs_select_their_native_recipes() {
        use crate::decoder_classifier::{DecoderClassifierConfig, DecoderClassifierTask};
        let embedding_cases: &[(&str, &str, DecoderMath)] = &[
            (
                "gte-Qwen2-1.5B",
                include_str!("../tests/fixtures/dense-gaps/gte_qwen2_1p5b.json"),
                DecoderMath::Qwen2H1536FullNeox,
            ),
            (
                "gte-Qwen2-7B",
                include_str!("../tests/fixtures/dense-gaps/gte_qwen2_7b.json"),
                DecoderMath::Qwen2FullNeox,
            ),
            (
                "jina-code-0.5b",
                include_str!("../tests/fixtures/dense-gaps/jina_code_0p5b.json"),
                DecoderMath::Qwen2H896MaskedCandle,
            ),
            (
                "jina-code-1.5b",
                include_str!("../tests/fixtures/dense-gaps/jina_code_1p5b.json"),
                DecoderMath::Qwen2H1536FullNeox,
            ),
            (
                "Qwen3-Embedding-0.6B",
                include_str!("../tests/fixtures/dense-gaps/qwen3_embed_0p6b.json"),
                DecoderMath::Qwen3H1024Candle,
            ),
            (
                "Qwen3-Embedding-4B",
                include_str!("../tests/fixtures/dense-gaps/qwen3_embed_4b.json"),
                DecoderMath::Qwen3H2560Ordinary,
            ),
            (
                "Qwen3-Embedding-8B",
                include_str!("../tests/fixtures/dense-gaps/qwen3_embed_8b.json"),
                DecoderMath::Qwen3H4096Ordinary,
            ),
            (
                "harrier-oss-v1-0.6b",
                include_str!("../tests/fixtures/dense-gaps/harrier_0p6b.json"),
                DecoderMath::Qwen3H1024Candle,
            ),
            (
                "voyage-4-nano",
                include_str!("../tests/fixtures/dense-gaps/voyage_4_nano.json"),
                DecoderMath::Qwen3H1024Candle,
            ),
            (
                "pplx-embed-v1-0.6b",
                include_str!("../tests/fixtures/dense-gaps/pplx_0p6b.json"),
                DecoderMath::Qwen3H1024Candle,
            ),
            (
                "pplx-embed-v1-4b",
                include_str!("../tests/fixtures/dense-gaps/pplx_4b.json"),
                DecoderMath::Qwen3H2560Ordinary,
            ),
        ];
        for (name, json, expected) in embedding_cases {
            let value: serde_json::Value = serde_json::from_str(json).unwrap();
            let config = DecoderConfig::from_value(value, false)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            assert_eq!(&DecoderMath::select(&config).unwrap(), expected, "{name}");
        }

        let skywork: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/dense-gaps/skywork_reward_qwen3_0p6b.json"
        ))
        .unwrap();
        let head = DecoderClassifierConfig::from_value(skywork.clone()).unwrap();
        assert_eq!(head.task, DecoderClassifierTask::Sequence);
        assert_eq!(head.num_labels, 1);
        let config = DecoderConfig::from_value(skywork, true).unwrap();
        assert_eq!(
            DecoderMath::select(&config).unwrap(),
            DecoderMath::Qwen3H1024Candle
        );

        let nemotron: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/dense-gaps/nemotron_rerank_1b_v2.json"
        ))
        .unwrap();
        let head = DecoderClassifierConfig::from_value(nemotron.clone()).unwrap();
        assert_eq!(head.task, DecoderClassifierTask::BidirectionalSequence);
        assert_eq!(head.num_labels, 1);
        assert_eq!(head.temperature, 1.0);
        let config = DecoderConfig::from_value(nemotron, true).unwrap();
        assert_eq!(config.family, DecoderFamily::Llama);
        assert!(!config.is_causal);
        assert_eq!(config.head_dim, 64);
        let scale = config.rope_scale.expect("llama3 scaling");
        assert_eq!(
            (scale.factor, scale.original_max_position_embeddings),
            (32.0, 8192.0)
        );
        assert_eq!(
            DecoderMath::select(&config).unwrap(),
            DecoderMath::LlamaH2048Exact
        );
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
            "infinity_rms_norm_1536_vector"
        );
        assert_eq!(
            DecoderMath::Qwen2H1536FullNeox.add_rms_export(),
            "infinity_add_rms_norm_1536_vector"
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
        assert!(error.contains("Llama default RoPE"));

        // Bidirectional Llama reuses the same trunk recipe; only the FA4 mask
        // and the mean-pooled sequence head differ.
        let noncausal = fixture_with_overrides(serde_json::json!({"is_causal": false}));
        assert!(validate_recipe(&noncausal.config()).is_ok());

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

    fn mistral_config() -> DecoderConfig {
        DecoderConfig {
            family: DecoderFamily::Mistral,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 2,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            vocab_size: 32000,
            max_position_embeddings: 32768,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            is_causal: true,
            attention_window: Some(4096),
            rope: DecoderRope::Default,
            rope_scale: None,
            routed: None,
        }
    }

    #[test]
    fn qwen3_moe_h4096_structural_recipe_is_additive_not_blindly_unlocked() {
        use crate::moe_model::RoutedMlpConfig;
        use std::collections::BTreeSet;
        let routed = RoutedMlpConfig {
            hidden_size: 4096,
            intermediate_size: 12288,
            num_hidden_layers: 94,
            moe_intermediate_size: 1536,
            num_experts: 128,
            num_experts_per_tok: 8,
            decoder_sparse_step: 1,
            norm_topk_prob: true,
            mlp_only_layers: BTreeSet::new(),
        };
        let c = DecoderConfig {
            family: DecoderFamily::Qwen3Moe,
            hidden_size: 4096,
            intermediate_size: 12288,
            num_hidden_layers: 94,
            num_attention_heads: 64,
            num_key_value_heads: 4,
            head_dim: 128,
            vocab_size: 151936,
            max_position_embeddings: 40960,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            is_causal: true,
            attention_window: None,
            rope: DecoderRope::FullNeoX,
            rope_scale: None,
            routed: Some(routed),
        };
        let math = DecoderMath::select(&c).unwrap();
        assert_eq!(math, DecoderMath::Qwen3MoeH4096Ordinary);
        assert_eq!(math.rms_export(), "infinity_rms_norm_4096_vector");
        assert_eq!(math.add_rms_export(), "infinity_add_rms_norm_4096_vector");
        assert_eq!(
            math.qk_norm_rope_export(),
            Some("infinity_qk_norm_rope_128_bf16")
        );
        assert_eq!(math.norm_geometry(), GENERIC_NORM_GEOMETRY);
    }

    #[test]
    fn mistral_h4096_math_and_unported_window_selectors() {
        let c = mistral_config();
        assert_eq!(
            DecoderMath::select(&c).unwrap(),
            DecoderMath::MistralH4096Ordinary
        );
        let math = DecoderMath::MistralH4096Ordinary;
        assert_eq!(math.rms_export(), "infinity_rms_norm_4096_vector");
        assert_eq!(math.add_rms_export(), "infinity_add_rms_norm_4096_vector");
        assert_eq!(math.rope_export(), "infinity_rope_neox");
        assert_eq!(
            math.activation_export(),
            "infinity_silu_mul_bf16_vector8_native"
        );
        assert_eq!(math.qk_norm_rope_export(), None);
        assert_eq!(math.norm_geometry(), GENERIC_NORM_GEOMETRY);
        let global = DecoderConfig {
            attention_window: None,
            ..mistral_config()
        };
        assert_eq!(
            DecoderMath::select(&global).unwrap(),
            DecoderMath::MistralH4096Ordinary
        );
        let wrong_hidden = DecoderConfig {
            hidden_size: 3072,
            ..mistral_config()
        };
        let noncausal = DecoderConfig {
            is_causal: false,
            ..mistral_config()
        };
        let window = DecoderConfig {
            attention_window: Some(2048),
            ..mistral_config()
        };
        let wrong_rope = DecoderConfig {
            rope: DecoderRope::FullNeoX,
            ..mistral_config()
        };
        for (config, needle) in [
            (wrong_hidden, "H4096"),
            (noncausal, "causal"),
            (window, "4096"),
            (wrong_rope, "default RoPE"),
        ] {
            let error = validate_recipe(&config).unwrap_err().to_string();
            assert!(error.contains(needle), "{error}");
        }
    }

    #[test]
    fn mistral_local_window_remains_local_and_supported() {
        let spec = attention_spec(&mistral_config()).expect("local spec");
        assert_eq!(
            spec.window,
            fa4::Window::Local {
                left: 4095,
                right: 0
            }
        );
        assert!(fa4::Fa4::supports(&spec));
        let global = DecoderConfig {
            attention_window: None,
            ..mistral_config()
        };
        assert_eq!(
            attention_spec(&global).expect("global spec").window,
            fa4::Window::Global
        );
        let underflow = DecoderConfig {
            attention_window: Some(0),
            ..mistral_config()
        };
        assert!(attention_spec(&underflow).is_err());
    }

    #[test]
    fn mistral_mean_plan_keeps_short_store_lanes_and_canonical_mix_capacity() {
        for longest in 1..=4usize {
            let cu = [0, longest as u32];
            let plan =
                crate::gemma2_math::MeanPlan::new(longest, 4096, &cu, &[0]).expect("mean plan");
            let geometry = plan.geometry().expect("mean grid");
            assert!(geometry.block()[0] >= 8);
            assert_eq!(geometry.grid(), [1, 512, 1]);
        }
        let cu = [0, 3, 7, 12, 15];
        let selected = [2, 0, 2, 1, 3];
        let plan =
            crate::gemma2_math::MeanPlan::new(15, 4096, &cu, &selected).expect("mixed mean plan");
        assert_eq!(plan.count(), 5);
        assert_eq!(plan.threads(), 8);
        assert_eq!(plan.geometry().expect("mixed grid").grid(), [5, 512, 1]);
    }

    #[test]
    fn mean_pool_route_uses_the_shared_bf16_tree_only_for_standard_mean_counts() {
        for pool in [Pool::Cls, Pool::Mean, Pool::LastToken] {
            for selections in [1usize, usize::from(u16::MAX)] {
                let expected = if matches!(&pool, Pool::Mean) {
                    MeanPoolRoute::NativeBf16Tree
                } else {
                    MeanPoolRoute::F32Fallback
                };
                assert_eq!(mean_pool_route(&pool, selections).unwrap(), expected);
            }
            assert_eq!(mean_pool_route(&pool, 0).unwrap(), MeanPoolRoute::Empty);
        }
        assert_eq!(
            mean_pool_route(&Pool::Mean, usize::from(u16::MAX) + 1).unwrap(),
            MeanPoolRoute::F32Fallback
        );
        assert!(mean_pool_route(&Pool::Splade, 1).is_err());
        assert!(mean_pool_route(&Pool::Splade, 0).is_err());
    }

    #[test]
    fn mean_pool_route_keeps_raw_only_requests_out_of_the_pooled_kernel() {
        for pool in [Pool::Cls, Pool::Mean, Pool::LastToken] {
            assert_eq!(mean_pool_route(&pool, 0).unwrap(), MeanPoolRoute::Empty);
            assert_eq!(selected_readout(&pool, 0), None);
        }
    }

    #[test]
    fn shared_bf16_mean_plan_covers_dense_widths_and_short_spans() {
        for width in [1024usize, 2560, 4096, 5120] {
            for rows in 1..=4usize {
                let cu = [0, rows as u32];
                let plan = crate::gemma2_math::MeanPlan::new(rows, width, &cu, &[0])
                    .expect("dense mean plan");
                assert_eq!(plan.count(), 1);
                assert_eq!(plan.width() as usize, width);
                assert_eq!(plan.threads(), crate::gemma2_math::POOL_TILE as u32);
                assert_eq!(
                    plan.geometry().expect("dense mean grid").grid(),
                    [1, width.div_ceil(crate::gemma2_math::POOL_TILE) as u32, 1]
                );
                assert_eq!(plan.output_layout().shape(), &[1, width]);
            }
        }
    }

    #[test]
    fn shared_bf16_mean_plan_keeps_duplicates_and_accepts_more_selections_than_tokens() {
        let cu = [0, 3, 7, 12, 15];
        let selected = [2, 0, 2, 1, 3];
        let plan = crate::gemma2_math::MeanPlan::new(15, 4096, &cu, &selected)
            .expect("duplicate mean plan");
        assert_eq!(plan.count(), selected.len() as u32);
        assert_eq!(
            plan.geometry().expect("duplicate grid").grid()[0],
            selected.len() as u32
        );
        assert_eq!(plan.output_layout().shape(), &[selected.len(), 4096]);

        let cu = [0, 2, 4];
        let selected = [0, 0, 0, 0, 1];
        let plan =
            crate::gemma2_math::MeanPlan::new(4, 1024, &cu, &selected).expect("M>T mean plan");
        assert!(selected.len() > cu.len() - 1);
        assert!(selected.len() > 4, "selections exceed the token count");
        assert_eq!(plan.count(), selected.len() as u32);
        assert_eq!(plan.output_layout().shape(), &[selected.len(), 1024]);
    }

    #[test]
    fn mistral_geometry_and_workspace_capacity_follow_actual_slabs() {
        let c = mistral_config();
        let g = DecoderGeometry::new(&c).expect("geometry");
        assert_eq!(
            (g.hidden, g.q_width, g.kv_width, g.qkv_width),
            (4096, 4096, 1024, 6144)
        );
        assert_eq!(g.projection_width, 28672);
        assert_eq!(g.activation_width, 14336);
        assert_eq!(g.silu_groups_per_row, 1792);
        assert!(DecoderWork::allocation_bytes((128, 2, 1), &c).is_ok());
        assert!(DecoderWork::allocation_bytes((usize::MAX, 2, 1), &c).is_err());
    }
    #[test]
    fn geometry_keeps_query_width_separate_from_hidden_width() {
        let fixture = Fixture::with_config(
            r#"{
                "model_type": "qwen2",
                "architectures": ["Qwen2Model"],
                "hidden_size": 12,
                "intermediate_size": 20,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "vocab_size": 12,
                "max_position_embeddings": 32,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000.0,
                "hidden_act": "silu",
                "is_causal": false
            }"#,
        );
        let c = fixture.config();
        assert_ne!(c.q_width(), c.hidden_size);
        let g = DecoderGeometry::new(&c).expect("geometry");
        assert_eq!(
            (g.hidden, g.q_width, g.kv_width, g.qkv_width),
            (12, 16, 8, 32)
        );
        assert_eq!((g.projection_width, g.activation_width), (40, 20));
        let layout = DecoderClassifierScratchLayout::for_key(c.hidden_size, 3, (32, 4, 2))
            .expect("classifier scratch");
        assert_eq!(layout.gather_elements(), 4 * 12);
        assert_eq!(layout.logits_elements(), 32 * 3);
    }

    #[test]
    fn classifier_plan_kinds_map_to_distinct_trunk_modes() {
        let selected = RowPlan::selected_sequence(3, 3).unwrap();
        let packed = RowPlan::from_row_ids(10, vec![1, 3, 9]).unwrap();
        let tokens = RowPlan::tokens(10).unwrap();
        let mean = RowPlan::mean_sequence(&[1, 2, 3, 4, 5], &[0, 2, 5]).unwrap();
        assert_eq!(
            classifier_mode(Some(&selected)),
            Some(ClassifierMode::Selected)
        );
        assert_eq!(
            classifier_mode(Some(&packed)),
            Some(ClassifierMode::FullSequenceGather)
        );
        assert_eq!(
            classifier_mode(Some(&tokens)),
            Some(ClassifierMode::FullTokens)
        );
        assert_eq!(
            classifier_mode(Some(&mean)),
            Some(ClassifierMode::FullSequenceMean)
        );
        assert_eq!(classifier_mode(None), None);
    }

    #[test]
    fn classifier_scratch_bytes_are_accounted_and_overflow_is_rejected() {
        let fixture = Fixture::new();
        let c = fixture.config();
        let key = (32usize, 4, 2);
        let plain = DecoderWork::allocation_bytes(key, &c).expect("plain bytes");
        let layout = DecoderClassifierScratchLayout::for_key(c.hidden_size, 2, key)
            .expect("classifier layout");
        let with = DecoderWork::allocation_bytes_with_classifier(key, &c, Some(layout))
            .expect("classifier bytes");
        let pooled_bytes = key.2 * c.hidden_size * std::mem::size_of::<f32>();
        assert_eq!(
            with,
            plain - pooled_bytes + layout.bytes().unwrap(),
            "head works replace the unused pooled slab with head scratch"
        );
        assert_eq!(pooled_elements(key.2, c.hidden_size, true).unwrap(), 0);
        assert_eq!(
            pooled_elements(key.2, c.hidden_size, false).unwrap(),
            key.2 * c.hidden_size
        );
        assert!(DecoderWork::allocation_bytes_with_classifier(
            (usize::MAX, 4, 2),
            &c,
            Some(layout)
        )
        .is_err());
        assert!(DecoderWork::allocation_bytes_with_classifier(
            (32, 4, usize::MAX),
            &c,
            Some(layout)
        )
        .is_err());
    }

    #[test]
    fn token_classifier_bytes_exclude_the_unused_token_by_hidden_f32_slab() {
        let fixture = Fixture::with_config(
            r#"{
                "model_type": "qwen3",
                "architectures": ["Qwen3ForCausalLM"],
                "hidden_size": 2560,
                "intermediate_size": 9728,
                "num_hidden_layers": 1,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "vocab_size": 151669,
                "max_position_embeddings": 40960,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000.0,
                "hidden_act": "silu",
                "is_causal": false
            }"#,
        );
        let c = fixture.config();
        let g = DecoderGeometry::new(&c).expect("geometry");
        assert!(
            g.projection_width > g.hidden,
            "D > H layouts must retain their ordinary slab bounds"
        );
        let tokens = 8192usize;
        let labels = 2usize;
        let key = (tokens, 2, tokens);
        let layout = DecoderClassifierScratchLayout::for_key(c.hidden_size, labels, key)
            .expect("classifier layout");
        let plain = DecoderWork::allocation_bytes(key, &c).expect("plain bytes");
        let with = DecoderWork::allocation_bytes_with_classifier(key, &c, Some(layout))
            .expect("classifier bytes");
        let unused_pooled = tokens * c.hidden_size * std::mem::size_of::<f32>();
        assert_eq!(unused_pooled, 83_886_080);
        assert_eq!(
            with,
            plain - unused_pooled + layout.bytes().unwrap(),
            "token classification must not charge T*H*4 pooled storage"
        );
        assert_eq!(pooled_elements(tokens, c.hidden_size, true).unwrap(), 0);
        assert!(
            with < plain,
            "head workspace is smaller than pooled baseline"
        );
    }
}
