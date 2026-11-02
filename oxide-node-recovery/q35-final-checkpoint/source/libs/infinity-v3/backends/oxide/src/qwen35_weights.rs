// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//!
//! Bounded cold ownership of dense Qwen3.5 text weights.
//!
//! This module turns the metadata-only [`Qwen35Weights`] view into the exact
//! immutable device buffers expected by the future text executor. It performs
//! no inference, kernel selection, context queries, or per-request work: every
//! matrix source is streamed once through the shared [`WeightUploader`], and
//! the returned [`Qwen35DeviceWeights`] owns every buffer until the outer model
//! completes. `lm_head` deliberately stays a lazy [`WeightSource`] for the
//! separate head worker; it is never read or uploaded here.
//!
//! Full-attention `qkv` is the canonical packed `[query|gate, K, V]` row
//! layout: the existing uploader concatenates the already per-head packed
//! `q_proj` rows with the key and value rows once. Gated-delta layers keep
//! `qkv` and `z` separate and stack only `a` and `b`. RMSNorm affine weights
//! (final, per-layer input/post-attention, full q/k norm) stay raw BF16 because
//! the executor adds one in its F32 norm math. The gated-delta per-head norm is
//! an ordinary BF16 multiplier.
//!
//! Gated-delta convolution, `A_log`, `dt_bias`, and head norm are small native
//! CPU arrays. They are read into one bounded scratch set and converted with
//! exactly the native casts before the future catalog's cold
//! [`GdnDeviceWeights::upload`] wraps each layer binding in one `Arc`.

use crate::gdn_prepared::{GdnDeviceWeights, GdnGeometry, GdnWeights};
use crate::model::WeightSource;
use crate::qwen35_backbone::Qwen35BackboneGeometry;
use crate::qwen35_config::{Qwen35Config, Qwen35LayerType};
use crate::qwen35_model::{Qwen35AttentionWeights, Qwen35LayerWeights, Qwen35Weights};
use crate::tensor::Bf16;
use crate::upload::WeightUploader;
use cuda_core::{DeviceBuffer, Stream};
use infinity_v3_backend_core::BackendError;
use std::sync::Arc;

const GDN_HEAD_DIM: usize = 128;

fn start(error: impl std::fmt::Display) -> BackendError {
    BackendError::Start(format!("Oxide qwen3.5 weights: {error}"))
}

/// Immutable device weights for the complete dense text backbone.
pub(crate) struct Qwen35DeviceWeights {
    pub(crate) config: Qwen35Config,
    pub(crate) embed: DeviceBuffer<Bf16>,
    pub(crate) final_norm: DeviceBuffer<Bf16>,
    pub(crate) layers: Vec<Qwen35DeviceLayer>,
    pub(crate) lm_head: Option<WeightSource>,
}

/// One uploaded layer. Norms remain raw BF16 offsets; the MLP keeps its
/// concatenated gate/up rows and its separate down projection.
pub(crate) struct Qwen35DeviceLayer {
    pub(crate) input_norm: DeviceBuffer<Bf16>,
    pub(crate) post_attention_norm: DeviceBuffer<Bf16>,
    pub(crate) gate_up: DeviceBuffer<Bf16>,
    pub(crate) down: DeviceBuffer<Bf16>,
    pub(crate) attention: Qwen35DeviceAttention,
}

/// The two attention kinds in their exact upload-time device layout.
pub(crate) enum Qwen35DeviceAttention {
    Full {
        qkv: DeviceBuffer<Bf16>,
        output: DeviceBuffer<Bf16>,
        q_norm: DeviceBuffer<Bf16>,
        k_norm: DeviceBuffer<Bf16>,
    },
    Linear {
        qkv: DeviceBuffer<Bf16>,
        z: DeviceBuffer<Bf16>,
        ab: DeviceBuffer<Bf16>,
        output: DeviceBuffer<Bf16>,
        weights: Arc<GdnDeviceWeights>,
    },
}

impl Qwen35DeviceWeights {
    /// Upload one complete, already-loaded text checkpoint. Configuration and
    /// source shapes are checked before any device allocation. The outer
    /// [`WeightUploader`] is the only matrix uploader and is drained before the
    /// weights become visible to the caller.
    pub(crate) fn upload(host: Qwen35Weights, stream: &Arc<Stream>) -> Result<Self, BackendError> {
        validate_host(&host)?;
        let Qwen35Weights {
            config,
            embeddings,
            norm,
            layers: host_layers,
            lm_head,
            ..
        } = host;
        let mut layers = Vec::new();
        layers.try_reserve_exact(host_layers.len()).map_err(start)?;
        let mut scratch = GdnScratch::new(&config)?;
        let mut uploader = WeightUploader::new(stream.clone())?;
        let embed = uploader.upload(&[&embeddings])?;
        let final_norm = uploader.upload(&[&norm])?;
        for (index, (layer, kind)) in host_layers.iter().zip(&config.layer_types).enumerate() {
            layers.push(upload_layer(
                &mut uploader,
                stream,
                &config,
                layer,
                *kind,
                &mut scratch,
                index,
            )?);
        }
        uploader.finish()?;
        Ok(Self {
            config,
            embed,
            final_norm,
            layers,
            lm_head,
        })
    }
}

fn checked_product(left: usize, right: usize, name: &str) -> Result<usize, BackendError> {
    left.checked_mul(right)
        .ok_or_else(|| start(format!("{name} overflow")))
}

fn checked_add(left: usize, right: usize, name: &str) -> Result<usize, BackendError> {
    left.checked_add(right)
        .ok_or_else(|| start(format!("{name} overflow")))
}

fn full_widths(config: &Qwen35Config) -> Result<(usize, usize), BackendError> {
    let query = checked_product(config.num_attention_heads, config.head_dim, "query width")?;
    let key_value = checked_product(
        config.num_key_value_heads,
        config.head_dim,
        "key/value width",
    )?;
    Ok((query, key_value))
}

fn linear_widths(config: &Qwen35Config) -> Result<(usize, usize), BackendError> {
    let key = checked_product(
        config.linear_num_key_heads,
        GDN_HEAD_DIM,
        "linear key width",
    )?;
    let value = checked_product(
        config.linear_num_value_heads,
        GDN_HEAD_DIM,
        "linear value width",
    )?;
    let qkv = checked_add(
        checked_product(key, 2, "linear QKV width")?,
        value,
        "linear QKV width",
    )?;
    Ok((qkv, value))
}

fn kind_name(kind: Qwen35LayerType) -> &'static str {
    match kind {
        Qwen35LayerType::Full => "full_attention",
        Qwen35LayerType::Linear => "linear_attention",
    }
}

fn check(source: &WeightSource, expected: &[usize], name: &str) -> Result<(), BackendError> {
    if source.shape() != expected {
        return Err(start(format!(
            "{name} has shape {:?}, expected {expected:?}",
            source.shape()
        )));
    }
    Ok(())
}

/// Pure metadata admission. Layer kinds, shapes, and config consistency are
/// rejected here before the uploader allocates any device storage.
fn validate_host(host: &Qwen35Weights) -> Result<(), BackendError> {
    let config = &host.config;
    if config.layer_types.len() != config.num_hidden_layers
        || host.layers.len() != config.num_hidden_layers
    {
        return Err(start(format!(
            "host has {} layers and {} configured layer types for {} configured layers",
            host.layers.len(),
            config.layer_types.len(),
            config.num_hidden_layers
        )));
    }
    let hidden = config.hidden_size;
    let intermediate = config.intermediate_size;
    check(
        &host.embeddings,
        &[config.vocab_size, hidden],
        "embed_tokens.weight",
    )?;
    check(&host.norm, &[hidden], "norm.weight")?;
    if let Some(lm_head) = &host.lm_head {
        check(lm_head, &[config.vocab_size, hidden], "lm_head.weight")?;
    }
    let (query, key_value) = full_widths(config)?;
    let query_gate = checked_product(query, 2, "query-gate width")?;

    for (index, (layer, kind)) in host.layers.iter().zip(&config.layer_types).enumerate() {
        let name = |suffix: &str| format!("layer {index} {suffix}");
        check(
            &layer.input_norm,
            &[hidden],
            &name("input_layernorm.weight"),
        )?;
        check(
            &layer.post_attention_norm,
            &[hidden],
            &name("post_attention_layernorm.weight"),
        )?;
        check(
            &layer.mlp_gate,
            &[intermediate, hidden],
            &name("mlp.gate_proj.weight"),
        )?;
        check(
            &layer.mlp_up,
            &[intermediate, hidden],
            &name("mlp.up_proj.weight"),
        )?;
        check(
            &layer.mlp_down,
            &[hidden, intermediate],
            &name("mlp.down_proj.weight"),
        )?;
        match (*kind, &layer.attention) {
            (
                Qwen35LayerType::Full,
                Qwen35AttentionWeights::Full {
                    query_gate: source,
                    key,
                    value,
                    output,
                    q_norm,
                    k_norm,
                },
            ) => {
                check(source, &[query_gate, hidden], &name("q_proj.weight"))?;
                check(key, &[key_value, hidden], &name("k_proj.weight"))?;
                check(value, &[key_value, hidden], &name("v_proj.weight"))?;
                check(output, &[hidden, query], &name("o_proj.weight"))?;
                check(q_norm, &[config.head_dim], &name("q_norm.weight"))?;
                check(k_norm, &[config.head_dim], &name("k_norm.weight"))?;
            }
            (
                Qwen35LayerType::Linear,
                Qwen35AttentionWeights::Linear {
                    qkv,
                    z,
                    a,
                    b,
                    output,
                    norm,
                    a_log,
                    dt_bias,
                    conv,
                },
            ) => {
                if config.linear_key_head_dim != GDN_HEAD_DIM
                    || config.linear_value_head_dim != GDN_HEAD_DIM
                {
                    return Err(start(
                        "linear attention requires 128-wide GDN key/value heads",
                    ));
                }
                GdnGeometry::new(
                    config.linear_num_key_heads,
                    config.linear_num_value_heads,
                    config.linear_conv_kernel_dim,
                )?;
                let (linear_qkv, linear_value) = linear_widths(config)?;
                let conv_width = checked_product(
                    linear_qkv,
                    config.linear_conv_kernel_dim,
                    "convolution weight size",
                )?;
                check(qkv, &[linear_qkv, hidden], &name("in_proj_qkv.weight"))?;
                check(z, &[linear_value, hidden], &name("in_proj_z.weight"))?;
                check(
                    a,
                    &[config.linear_num_value_heads, hidden],
                    &name("in_proj_a.weight"),
                )?;
                check(
                    b,
                    &[config.linear_num_value_heads, hidden],
                    &name("in_proj_b.weight"),
                )?;
                check(output, &[hidden, linear_value], &name("out_proj.weight"))?;
                check(
                    conv,
                    &[linear_qkv, 1, config.linear_conv_kernel_dim],
                    &name("conv1d.weight"),
                )?;
                if conv.len() != conv_width {
                    return Err(start(format!(
                        "layer {index} conv1d.weight has {} elements, expected {conv_width}",
                        conv.len()
                    )));
                }
                check(a_log, &[config.linear_num_value_heads], &name("A_log"))?;
                check(dt_bias, &[config.linear_num_value_heads], &name("dt_bias"))?;
                check(
                    norm,
                    &[config.linear_value_head_dim],
                    &name("linear norm.weight"),
                )?;
            }
            (configured, actual) => {
                let actual = match actual {
                    Qwen35AttentionWeights::Full { .. } => "full_attention",
                    Qwen35AttentionWeights::Linear { .. } => "linear_attention",
                };
                return Err(start(format!(
                    "layer {index} is configured as {} but the host source is {actual}",
                    kind_name(configured)
                )));
            }
        }
    }
    Ok(())
}

fn upload_layer(
    uploader: &mut WeightUploader,
    stream: &Arc<Stream>,
    config: &Qwen35Config,
    layer: &Qwen35LayerWeights,
    kind: Qwen35LayerType,
    scratch: &mut GdnScratch,
    index: usize,
) -> Result<Qwen35DeviceLayer, BackendError> {
    let input_norm = uploader.upload(&[&layer.input_norm])?;
    let post_attention_norm = uploader.upload(&[&layer.post_attention_norm])?;
    let gate_up = uploader.upload(&[&layer.mlp_gate, &layer.mlp_up])?;
    let down = uploader.upload(&[&layer.mlp_down])?;
    let attention = match (kind, &layer.attention) {
        (
            Qwen35LayerType::Full,
            Qwen35AttentionWeights::Full {
                query_gate,
                key,
                value,
                output,
                q_norm,
                k_norm,
            },
        ) => Qwen35DeviceAttention::Full {
            qkv: uploader.upload(&[query_gate, key, value])?,
            output: uploader.upload(&[output])?,
            q_norm: uploader.upload(&[q_norm])?,
            k_norm: uploader.upload(&[k_norm])?,
        },
        (
            Qwen35LayerType::Linear,
            Qwen35AttentionWeights::Linear {
                qkv,
                z,
                a,
                b,
                output,
                conv,
                a_log,
                dt_bias,
                norm,
            },
        ) => {
            let weights = upload_gdn_weights(stream, config, conv, a_log, dt_bias, norm, scratch)?;
            Qwen35DeviceAttention::Linear {
                qkv: uploader.upload(&[qkv])?,
                z: uploader.upload(&[z])?,
                ab: uploader.upload(&[a, b])?,
                output: uploader.upload(&[output])?,
                weights,
            }
        }
        _ => {
            return Err(start(format!(
                "layer {index} attention kind changed after host validation"
            )));
        }
    };
    Ok(Qwen35DeviceLayer {
        input_norm,
        post_attention_norm,
        gate_up,
        down,
        attention,
    })
}

/// One bounded host scratch set for the small native GDN arrays, reused across
/// every linear layer so no per-layer persistent scratch is retained.
pub(crate) struct GdnScratch {
    bits: Vec<u16>,
    conv: Vec<half::bf16>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    norm: Vec<half::bf16>,
}

impl GdnScratch {
    fn new(config: &Qwen35Config) -> Result<Self, BackendError> {
        let has_linear = config.layer_types.contains(&Qwen35LayerType::Linear);
        let (channels, heads, norm_width, kernel) = if has_linear {
            let (channels, _) = linear_widths(config)?;
            (
                channels,
                config.linear_num_value_heads,
                config.linear_value_head_dim,
                config.linear_conv_kernel_dim,
            )
        } else {
            (0, 0, 0, 0)
        };
        Self::reserve_scratch(channels, heads, norm_width, kernel)
    }

    /// Scratch for a shared-backbone MoE host. The attention geometry is the
    /// same as the dense donor's; only the FFN kind differs.
    pub(crate) fn new_backbone(geometry: &Qwen35BackboneGeometry) -> Result<Self, BackendError> {
        let has_linear = geometry.layer_types.contains(&Qwen35LayerType::Linear);
        let (channels, heads, norm_width, kernel) = if has_linear {
            let (_, _, channels) = geometry.linear_widths()?;
            (
                channels,
                geometry.linear_num_value_heads,
                geometry.linear_value_head_dim,
                geometry.linear_conv_kernel_dim,
            )
        } else {
            (0, 0, 0, 0)
        };
        Self::reserve_scratch(channels, heads, norm_width, kernel)
    }

    fn reserve_scratch(
        channels: usize,
        heads: usize,
        norm_width: usize,
        kernel: usize,
    ) -> Result<Self, BackendError> {
        let conv_elements = checked_product(channels, kernel, "convolution weight size")?;
        let mut scratch = Self {
            bits: Vec::new(),
            conv: Vec::new(),
            a_log: Vec::new(),
            dt_bias: Vec::new(),
            norm: Vec::new(),
        };
        reserve(
            &mut scratch.bits,
            conv_elements.max(heads).max(norm_width),
            "native staging",
        )?;
        reserve(&mut scratch.conv, conv_elements, "convolution weight")?;
        reserve(&mut scratch.a_log, heads, "A_log")?;
        reserve(&mut scratch.dt_bias, heads, "dt_bias")?;
        reserve(&mut scratch.norm, norm_width, "norm weight")?;
        Ok(scratch)
    }
}

fn reserve<T>(values: &mut Vec<T>, capacity: usize, name: &str) -> Result<(), BackendError> {
    values
        .try_reserve_exact(capacity)
        .map_err(|_| start(format!("{name} host allocation failed")))
}

fn read_bits(
    source: &WeightSource,
    expected: usize,
    bits: &mut Vec<u16>,
    name: &str,
) -> Result<(), BackendError> {
    if source.len() != expected {
        return Err(start(format!(
            "{name} has {} elements, expected {expected}",
            source.len()
        )));
    }
    bits.clear();
    bits.try_reserve_exact(expected)
        .map_err(|_| start(format!("{name} staging allocation failed")))?;
    bits.resize(expected, 0u16);
    source.read_bf16(0, bits)
}

fn cast_bf16(bits: &[u16], out: &mut Vec<half::bf16>, name: &str) -> Result<(), BackendError> {
    out.clear();
    out.try_reserve_exact(bits.len())
        .map_err(|_| start(format!("{name} host allocation failed")))?;
    for (index, &raw) in bits.iter().enumerate() {
        let value = half::bf16::from_bits(raw);
        if !value.is_finite() {
            return Err(start(format!("{name} value {index} is not finite")));
        }
        out.push(value);
    }
    Ok(())
}

fn read_gdn_f32(
    source: &WeightSource,
    expected: usize,
    out: &mut Vec<f32>,
    name: &str,
) -> Result<(), BackendError> {
    if source.len() != expected {
        return Err(start(format!(
            "{name} source length differs from {expected}"
        )));
    }
    out.clear();
    out.try_reserve_exact(expected)
        .map_err(|_| start(format!("{name} host allocation failed")))?;
    out.resize(expected, 0.0);
    source.read_f32(0, out)
}

/// Read the four small GDN sources with the native casts and upload one cold
/// device binding. The source arrays stay bounded by convolution width times
/// kernel and never become F32 matrix copies.
fn upload_gdn_weights(
    stream: &Arc<Stream>,
    config: &Qwen35Config,
    conv: &WeightSource,
    a_log: &WeightSource,
    dt_bias: &WeightSource,
    norm: &WeightSource,
    scratch: &mut GdnScratch,
) -> Result<Arc<GdnDeviceWeights>, BackendError> {
    let (channels, _) = linear_widths(config)?;
    upload_gdn_weights_dims(
        stream,
        channels,
        config.linear_conv_kernel_dim,
        config.linear_num_key_heads,
        config.linear_num_value_heads,
        config.linear_value_head_dim,
        conv,
        a_log,
        dt_bias,
        norm,
        scratch,
    )
}

#[allow(clippy::too_many_arguments)]
fn upload_gdn_weights_dims(
    stream: &Arc<Stream>,
    channels: usize,
    kernel: usize,
    key_heads: usize,
    value_heads: usize,
    value_head_dim: usize,
    conv: &WeightSource,
    a_log: &WeightSource,
    dt_bias: &WeightSource,
    norm: &WeightSource,
    scratch: &mut GdnScratch,
) -> Result<Arc<GdnDeviceWeights>, BackendError> {
    let conv_elements = checked_product(channels, kernel, "convolution weight size")?;
    let heads = value_heads;
    read_bits(conv, conv_elements, &mut scratch.bits, "conv1d.weight")?;
    cast_bf16(&scratch.bits, &mut scratch.conv, "conv1d.weight")?;
    read_gdn_f32(a_log, heads, &mut scratch.a_log, "A_log")?;
    read_gdn_f32(dt_bias, heads, &mut scratch.dt_bias, "dt_bias")?;
    read_bits(
        norm,
        value_head_dim,
        &mut scratch.bits,
        "linear norm.weight",
    )?;
    cast_bf16(&scratch.bits, &mut scratch.norm, "linear norm.weight")?;
    let geometry = GdnGeometry::new(key_heads, value_heads, kernel)?;
    let weights = GdnDeviceWeights::upload(
        stream,
        geometry,
        GdnWeights {
            conv: &scratch.conv,
            a_log: &scratch.a_log,
            dt_bias: &scratch.dt_bias,
            norm: &scratch.norm,
        },
    )?;
    Ok(Arc::new(weights))
}

/// Upload one shared-backbone attention layer without requiring a dense MLP.
/// The MoE full executor owns the routed/shared FFN upload separately through
/// `DeviceFfnPlan::upload`, so attention and FFN payloads stay disjoint.
pub(crate) fn upload_backbone_attention(
    uploader: &mut WeightUploader,
    stream: &Arc<Stream>,
    attention: &Qwen35AttentionWeights,
    kind: Qwen35LayerType,
    geometry: &Qwen35BackboneGeometry,
    scratch: &mut GdnScratch,
    index: usize,
) -> Result<Qwen35DeviceAttention, BackendError> {
    match (kind, attention) {
        (
            Qwen35LayerType::Full,
            Qwen35AttentionWeights::Full {
                query_gate,
                key,
                value,
                output,
                q_norm,
                k_norm,
            },
        ) => Ok(Qwen35DeviceAttention::Full {
            qkv: uploader.upload(&[query_gate, key, value])?,
            output: uploader.upload(&[output])?,
            q_norm: uploader.upload(&[q_norm])?,
            k_norm: uploader.upload(&[k_norm])?,
        }),
        (
            Qwen35LayerType::Linear,
            Qwen35AttentionWeights::Linear {
                qkv,
                z,
                a,
                b,
                output,
                conv,
                a_log,
                dt_bias,
                norm,
            },
        ) => {
            let (_, _, channels) = geometry.linear_widths()?;
            let weights = upload_gdn_weights_dims(
                stream,
                channels,
                geometry.linear_conv_kernel_dim,
                geometry.linear_num_key_heads,
                geometry.linear_num_value_heads,
                geometry.linear_value_head_dim,
                conv,
                a_log,
                dt_bias,
                norm,
                scratch,
            )?;
            Ok(Qwen35DeviceAttention::Linear {
                qkv: uploader.upload(&[qkv])?,
                z: uploader.upload(&[z])?,
                ab: uploader.upload(&[a, b])?,
                output: uploader.upload(&[output])?,
                weights,
            })
        }
        _ => Err(start(format!(
            "layer {index} attention kind changed after host validation"
        ))),
    }
}
