// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Staging typed loader for dense Qwen2/Qwen3/Llama and routed Qwen3-MoE
//! pooled embedding backbones.
//!
//! This module validates configuration, sentence-transformers task modules and
//! checkpoint metadata. It deliberately does not select kernels, upload device
//! weights, build rotary tables or claim engine support. Executor integration is
//! a separate step and owns those contracts.

use crate::decoder_classifier::{
    classifier_dense_rejected, reject_declared_dense_modules, DecoderClassifierConfig,
    DecoderClassifierHeadHost,
};
use crate::model::{elements, path_present, read_json, Checkpoint, WeightSource};
use crate::moe_model::{load_routed_layer, RoutedLayerWeights, RoutedMlpConfig};
use infinity_v3_backend_core::BackendError;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::{Component, Path};

fn start(error: impl std::fmt::Display) -> BackendError {
    BackendError::Start(format!("Oxide decoder: {error}"))
}

const DEFAULT_LLAMA_THETA: f64 = 10_000.0;
const DEFAULT_QWEN2_THETA: f64 = 1_000_000.0;
const DEFAULT_QWEN3_THETA: f64 = 1_000_000.0;
/// Qwen3 and Qwen3-MoE per-head Q/K RMSNorm width. Unported variants with a
/// different head dimension are rejected by the executor after metadata
/// validation.
pub(crate) const QWEN3_QK_NORM_WIDTH: usize = 128;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DecoderFamily {
    Qwen2,
    Qwen3,
    Qwen3Moe,
    Llama,
    Mistral,
}
impl DecoderFamily {
    pub(crate) fn from_model_type(model_type: &str) -> Option<Self> {
        match model_type {
            "qwen2" => Some(Self::Qwen2),
            "qwen3" => Some(Self::Qwen3),
            "qwen3_moe" => Some(Self::Qwen3Moe),
            "llama" => Some(Self::Llama),
            "mistral" => Some(Self::Mistral),
            _ => None,
        }
    }

    pub(crate) fn is_decoder_model_type(model_type: &str) -> bool {
        Self::from_model_type(model_type).is_some()
            || matches!(model_type, "ministral3" | "bidirectional_pplx_qwen3")
    }

    /// Known Native decoder aliases/families whose actual arithmetic is not
    /// ported. They must be rejected on the host instead of falling through to
    /// encoder detection. Qwen3.5 remains a separate future Root branch.
    pub(crate) fn is_unsupported_native_decoder_model_type(model_type: &str) -> bool {
        matches!(model_type, "ministral3" | "llama_bidirec" | "mistral3")
    }

    pub(crate) const fn requires_qk_norm(self) -> bool {
        matches!(self, Self::Qwen3 | Self::Qwen3Moe)
    }

    const fn default_theta(self) -> f64 {
        match self {
            Self::Qwen2 => DEFAULT_QWEN2_THETA,
            Self::Qwen3 | Self::Qwen3Moe => DEFAULT_QWEN3_THETA,
            Self::Llama | Self::Mistral => DEFAULT_LLAMA_THETA,
        }
    }

    const fn default_rope(self) -> DecoderRope {
        match self {
            Self::Qwen2 | Self::Qwen3 | Self::Qwen3Moe => DecoderRope::FullNeoX,
            Self::Llama | Self::Mistral => DecoderRope::Default,
        }
    }

    fn accepts_architecture(self, architecture: &str, classifier: bool) -> bool {
        let backbone = match self {
            Self::Qwen2 => matches!(architecture, "Qwen2Model" | "Qwen2ForCausalLM"),
            Self::Qwen3 => matches!(architecture, "Qwen3Model" | "Qwen3ForCausalLM"),
            Self::Qwen3Moe => matches!(architecture, "Qwen3MoeModel" | "Qwen3MoeForCausalLM"),
            Self::Llama => matches!(architecture, "LlamaModel" | "LlamaForCausalLM"),
            Self::Mistral => matches!(architecture, "MistralModel" | "MistralForCausalLM"),
        };
        if backbone {
            return true;
        }
        classifier
            && matches!(
                (self, architecture),
                (
                    Self::Qwen2,
                    "Qwen2ForSequenceClassification" | "Qwen2ForTokenClassification"
                ) | (
                    Self::Qwen3,
                    "Qwen3ForSequenceClassification" | "Qwen3ForTokenClassification"
                ) | (
                    Self::Llama,
                    "LlamaForSequenceClassification" | "LlamaForTokenClassification"
                )
            )
    }
}

/// Loader-level rotary recipe label. The executor maps this to kernel plans;
/// none of these labels imply scaled, partial or interleaved RoPE support.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DecoderRope {
    Simple,
    Default,
    FullNeoX,
}
impl DecoderRope {
    fn from_name(name: &str) -> Option<Self> {
        match name {
            "simple" => Some(Self::Simple),
            "default" => Some(Self::Default),
            "full_neox" | "full-neox" | "fullneox" | "neox" => Some(Self::FullNeoX),
            _ => None,
        }
    }
}

#[derive(Debug, Deserialize)]
struct RawDecoderConfig {
    model_type: String,
    architectures: Option<Vec<String>>,
    hidden_size: Option<usize>,
    intermediate_size: Option<usize>,
    num_hidden_layers: Option<usize>,
    num_attention_heads: Option<usize>,
    num_key_value_heads: Option<usize>,
    head_dim: Option<usize>,
    vocab_size: Option<usize>,
    max_position_embeddings: Option<usize>,
    rms_norm_eps: Option<f64>,
    rope_theta: Option<f64>,
    hidden_act: Option<String>,
    is_causal: Option<bool>,
    use_bidirectional_attention: Option<bool>,
    use_sliding_window: Option<bool>,
    sliding_window: Option<usize>,
    rope_scaling: Option<serde_json::Value>,
    rope_parameters: Option<serde_json::Value>,
    partial_rotary_factor: Option<f64>,
    quantization_config: Option<serde_json::Value>,
    rope_type: Option<String>,
    #[serde(flatten)]
    extra: BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DecoderConfig {
    pub(crate) family: DecoderFamily,
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) vocab_size: usize,
    pub(crate) max_position_embeddings: usize,
    pub(crate) rms_norm_eps: f64,
    pub(crate) rope_theta: f64,
    pub(crate) is_causal: bool,
    pub(crate) attention_window: Option<usize>,
    pub(crate) rope: DecoderRope,
    /// Validated routed-MLP metadata for `qwen3_moe`; always absent for dense
    /// families, whose expert keys remain rejected.
    pub(crate) routed: Option<RoutedMlpConfig>,
}
impl DecoderConfig {
    pub(crate) fn load(path: &Path) -> Result<Self, BackendError> {
        let value: serde_json::Value = read_json(&path.join("config.json"))?;
        let raw = RawDecoderConfig::deserialize(&value).map_err(start)?;
        Self::from_raw(raw, &value, false)
    }

    fn from_value(value: serde_json::Value, classifier: bool) -> Result<Self, BackendError> {
        let raw: RawDecoderConfig = serde_json::from_value(value.clone()).map_err(start)?;
        Self::from_raw(raw, &value, classifier)
    }

    pub(crate) const fn q_width(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    pub(crate) const fn kv_width(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    fn from_raw(
        raw: RawDecoderConfig,
        source: &serde_json::Value,
        classifier: bool,
    ) -> Result<Self, BackendError> {
        if raw.model_type == "ministral3" {
            return Err(start(
                "ministral3 H3072/YaRN/transformers math is not ported; only ordinary Mistral H4096 is supported",
            ));
        }
        let bidirectional_pplx_qwen3 = raw.model_type == "bidirectional_pplx_qwen3";
        let model_type = if bidirectional_pplx_qwen3 {
            "qwen3"
        } else {
            raw.model_type.as_str()
        };
        let family = DecoderFamily::from_model_type(model_type).ok_or_else(|| {
            start(format!(
                "unsupported model_type {}; expected dense qwen2, qwen3, qwen3_moe, llama or mistral",
                raw.model_type
            ))
        })?;
        if let Some(architectures) = raw.architectures.as_deref() {
            for architecture in architectures {
                if !family.accepts_architecture(architecture, classifier) {
                    return Err(start(format!(
                        "unsupported decoder task/architecture {architecture}; only pooled embedding Model/ForCausalLM backbones{}{} are supported",
                        if classifier { " or matching dense classification heads" } else { "" },
                        ""
                    )));
                }
            }
        }
        reject_config_modifiers(&raw, family)?;

        let hidden_size = required_usize(raw.hidden_size, "hidden_size")?;
        let intermediate_size = required_usize(raw.intermediate_size, "intermediate_size")?;
        let num_hidden_layers = required_usize(raw.num_hidden_layers, "num_hidden_layers")?;
        let num_attention_heads = required_usize(raw.num_attention_heads, "num_attention_heads")?;
        let num_key_value_heads = raw.num_key_value_heads.unwrap_or(num_attention_heads);
        let vocab_size = required_usize(raw.vocab_size, "vocab_size")?;
        let max_position_embeddings =
            required_usize(raw.max_position_embeddings, "max_position_embeddings")?;

        for (name, value) in [
            ("hidden_size", hidden_size),
            ("intermediate_size", intermediate_size),
            ("num_hidden_layers", num_hidden_layers),
            ("num_attention_heads", num_attention_heads),
            ("num_key_value_heads", num_key_value_heads),
            ("vocab_size", vocab_size),
            ("max_position_embeddings", max_position_embeddings),
        ] {
            if value == 0 {
                return Err(start(format!("decoder config {name} must be nonzero")));
            }
        }
        if u32::try_from(max_position_embeddings).is_err() {
            return Err(start(
                "decoder max_position_embeddings must fit CUDA position IDs",
            ));
        }
        let head_dim = match raw.head_dim {
            Some(0) => return Err(start("decoder config head_dim must be nonzero")),
            Some(head_dim) => head_dim,
            None => {
                if !hidden_size.is_multiple_of(num_attention_heads) {
                    return Err(start(
                        "decoder hidden_size must divide evenly by num_attention_heads",
                    ));
                }
                hidden_size / num_attention_heads
            }
        };
        if head_dim == 0 || !head_dim.is_multiple_of(2) {
            return Err(start("decoder rotary head_dim must be positive and even"));
        }
        num_attention_heads
            .checked_mul(head_dim)
            .ok_or_else(|| start("decoder query width overflow"))?;
        if !num_attention_heads.is_multiple_of(num_key_value_heads) {
            return Err(start(
                "decoder num_attention_heads must divide evenly by num_key_value_heads",
            ));
        }
        let kv_width = num_key_value_heads
            .checked_mul(head_dim)
            .ok_or_else(|| start("decoder key/value width overflow"))?;

        let rms_norm_eps = raw.rms_norm_eps.unwrap_or(1e-6);
        if !rms_norm_eps.is_finite()
            || rms_norm_eps <= 0.0
            || !(rms_norm_eps as f32).is_finite()
            || rms_norm_eps as f32 <= 0.0
        {
            return Err(start("decoder rms_norm_eps must be finite and positive"));
        }
        let saved_rope_theta = bidirectional_pplx_qwen3
            .then(|| {
                raw.rope_parameters
                    .as_ref()
                    .and_then(|value| value.get("rope_theta"))
                    .and_then(serde_json::Value::as_f64)
            })
            .flatten();
        let rope_theta = raw
            .rope_theta
            .or(saved_rope_theta)
            .unwrap_or(family.default_theta());
        if !rope_theta.is_finite()
            || rope_theta <= 1.0
            || !(rope_theta as f32).is_finite()
            || rope_theta as f32 <= 1.0
        {
            return Err(start(
                "decoder rope_theta must be finite and greater than one",
            ));
        }
        let hidden_act = raw.hidden_act.as_deref().unwrap_or("silu");
        if hidden_act != "silu" {
            return Err(start(format!(
                "unsupported decoder activation {hidden_act}; dense SiLU only"
            )));
        }
        let rope = match raw.rope_type.as_deref() {
            Some(name) => DecoderRope::from_name(name).ok_or_else(|| {
                start(format!(
                    "unsupported decoder rope_type {name}; simple, default or full_neox only"
                ))
            })?,
            None => family.default_rope(),
        };

        let bidirectional = raw.use_bidirectional_attention.unwrap_or(false);
        let is_causal = if bidirectional_pplx_qwen3 {
            // Native normalizes bidirectional_pplx_qwen3 to qwen3 and sets
            // both use_bidirectional_attention and is_causal after loading.
            false
        } else {
            match family {
                DecoderFamily::Qwen2 => raw.is_causal.unwrap_or(true),
                DecoderFamily::Qwen3
                | DecoderFamily::Qwen3Moe
                | DecoderFamily::Llama
                | DecoderFamily::Mistral => raw.is_causal.unwrap_or(!bidirectional),
            }
        };
        let attention_window = match family {
            DecoderFamily::Mistral => raw.sliding_window.filter(|window| *window > 0),
            _ => None,
        };
        if let Some(window) = attention_window {
            i32::try_from(window)
                .map_err(|_| start("decoder Mistral sliding_window exceeds CUDA bounds"))?;
        }

        let routed = match family {
            DecoderFamily::Qwen3Moe => {
                let routed = RoutedMlpConfig::from_json(source)?;
                if routed.hidden_size != hidden_size
                    || routed.intermediate_size != intermediate_size
                    || routed.num_hidden_layers != num_hidden_layers
                {
                    return Err(start(
                        "qwen3_moe routed MLP metadata does not match decoder geometry",
                    ));
                }
                Some(routed)
            }
            _ => None,
        };

        let config = Self {
            family,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            vocab_size,
            max_position_embeddings,
            rms_norm_eps,
            rope_theta,
            is_causal,
            attention_window,
            rope,
            routed,
        };
        config.validate(kv_width)?;
        Ok(config)
    }

    fn validate(&self, kv_width: usize) -> Result<(), BackendError> {
        let h = self.hidden_size;
        let i = self.intermediate_size;
        for (name, dimensions) in [
            ("embed_tokens.weight", [self.vocab_size, h]),
            ("norm.weight", [h, 1]),
            ("q_proj.weight", [self.q_width(), h]),
            ("k_proj.weight", [kv_width, h]),
            ("v_proj.weight", [kv_width, h]),
            ("o_proj.weight", [h, self.q_width()]),
            ("gate_proj.weight", [i, h]),
            ("up_proj.weight", [i, h]),
            ("down_proj.weight", [h, i]),
        ] {
            elements(&dimensions)
                .map_err(|error| start(format!("invalid decoder {name} dimensions: {error}")))?;
        }
        Ok(())
    }
}

fn required_usize(value: Option<usize>, name: &str) -> Result<usize, BackendError> {
    value.ok_or_else(|| start(format!("missing decoder config field {name}")))
}

fn reject_config_modifiers(
    raw: &RawDecoderConfig,
    family: DecoderFamily,
) -> Result<(), BackendError> {
    if raw
        .quantization_config
        .as_ref()
        .is_some_and(|value| !value.is_null() || family != DecoderFamily::Qwen3Moe)
    {
        return Err(start(
            "quantized decoder checkpoints are not supported by this loader",
        ));
    }
    // Native clears rope_scaling for dense Qwen2/Qwen3 even when a checkpoint
    // carries the field. Other families still need unsupported scaling rejected.
    if !matches!(family, DecoderFamily::Qwen2 | DecoderFamily::Qwen3)
        && raw
            .rope_scaling
            .as_ref()
            .is_some_and(|value| !value.is_null())
    {
        return Err(start(
            "scaled decoder RoPE (including Llama3) is not supported yet",
        ));
    }
    if let Some(parameters) = raw
        .rope_parameters
        .as_ref()
        .filter(|value| !value.is_null())
    {
        let parameters = parameters
            .as_object()
            .ok_or_else(|| start("decoder rope_parameters must be an object"))?;
        if let Some(rope_type) = parameters
            .get("rope_type")
            .and_then(serde_json::Value::as_str)
            .filter(|name| !name.is_empty() && !name.eq_ignore_ascii_case("default"))
        {
            return Err(start(format!(
                "decoder rope_parameters scaling {rope_type} (including YaRN) is not supported yet"
            )));
        }
    }
    if raw
        .partial_rotary_factor
        .is_some_and(|factor| factor != 1.0)
    {
        return Err(start(
            "partial rotary decoder configurations are not supported",
        ));
    }
    let active_window = raw.use_sliding_window == Some(true)
        || (raw.use_sliding_window != Some(false)
            && raw
                .sliding_window
                .zip(raw.max_position_embeddings)
                .is_some_and(|(window, maximum)| window < maximum));
    // Native uses every positive Llama window regardless of the optional
    // use_sliding_window flag. This executor only has global Llama attention;
    // reject on the host rather than silently changing its mask.
    let native_llama_window =
        family == DecoderFamily::Llama && raw.sliding_window.is_some_and(|window| window > 0);
    if (active_window || native_llama_window) && family != DecoderFamily::Mistral {
        return Err(start(
            "sliding-window decoder attention is not supported yet",
        ));
    }
    if family == DecoderFamily::Mistral
        && raw.extra.get("pooling").and_then(serde_json::Value::as_str) == Some("avg")
    {
        return Err(start(
            "decoder Mistral config-level average pooling needs unported ATen model-dtype reduction",
        ));
    }
    for key in raw.extra.keys() {
        let key_lower = key.to_ascii_lowercase();
        // `RoutedMlpConfig` validates quantization and shared-expert metadata
        // from the original JSON for qwen3_moe; dense families keep rejecting.
        if family == DecoderFamily::Qwen3Moe
            && matches!(
                key_lower.as_str(),
                "quantization"
                    | "shared_expert_intermediate_size"
                    | "n_shared_experts"
                    | "num_shared_experts"
                    | "shared_expert_gate"
            )
        {
            continue;
        }
        // Routed Qwen3-MoE keys are validated by `RoutedMlpConfig` from the
        // original JSON. Dense families keep rejecting every expert key.
        let routed_key = key_lower.contains("expert")
            || matches!(
                key_lower.as_str(),
                "moe_intermediate_size" | "router_aux_loss_coef"
            );
        let known_routed_key = matches!(
            key_lower.as_str(),
            "num_experts"
                | "num_experts_per_tok"
                | "moe_intermediate_size"
                | "router_aux_loss_coef"
        );
        let unsupported = (routed_key && !(family == DecoderFamily::Qwen3Moe && known_routed_key))
            || key_lower.contains("quantiz")
            || key_lower.contains("vision")
            || key_lower.contains("audio")
            || key_lower.starts_with("image_")
            || key_lower.starts_with("mm_")
            || key_lower.starts_with("multi_modal")
            || key_lower.starts_with("linear_")
            || key_lower.as_str() == "layer_types";
        if unsupported {
            return Err(start(format!(
                "unsupported decoder config modifier {key}; only dense SiLU text backbones are accepted"
            )));
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub(crate) struct DecoderLinear {
    pub(crate) weight: WeightSource,
    pub(crate) bias: Option<WeightSource>,
}

/// Layer MLP storage. Dense checkpoints retain the original gate/up/down
/// projections; routed Qwen3-MoE layers retain only router and expert views and
/// never allocate placeholder dense tensors.
// Retained checkpoint views stay inline; this cold carrier does not add an allocation per layer.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug)]
pub(crate) enum DecoderMlpWeights {
    Dense {
        gate: DecoderLinear,
        up: DecoderLinear,
        down: DecoderLinear,
    },
    Routed(RoutedLayerWeights),
}

impl DecoderMlpWeights {
    #[cfg(test)]
    pub(crate) fn dense(
        &self,
    ) -> Result<(&DecoderLinear, &DecoderLinear, &DecoderLinear), BackendError> {
        match self {
            Self::Dense { gate, up, down } => Ok((gate, up, down)),
            Self::Routed(_) => Err(start(
                "decoder layer MLP is routed; dense gate/up/down projections are unavailable",
            )),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct DecoderLayerWeights {
    pub(crate) input_norm: WeightSource,
    pub(crate) post_attention_norm: WeightSource,
    /// Dense Qwen3 per-head Q/K RMSNorm scales. Present only for Qwen3 and
    /// Qwen3-MoE, always as a complete pair, and absent for Qwen2/Llama.
    pub(crate) q_norm: Option<WeightSource>,
    pub(crate) k_norm: Option<WeightSource>,
    pub(crate) q: DecoderLinear,
    pub(crate) k: DecoderLinear,
    pub(crate) v: DecoderLinear,
    pub(crate) o: DecoderLinear,
    pub(crate) mlp: DecoderMlpWeights,
}

#[derive(Clone, Debug)]
pub(crate) struct DecoderWeights {
    pub(crate) config: DecoderConfig,
    pub(crate) embed: WeightSource,
    pub(crate) norm: WeightSource,
    pub(crate) layers: Vec<DecoderLayerWeights>,
}

fn select_root(checkpoint: &Checkpoint) -> Result<&'static str, BackendError> {
    let suffix = "embed_tokens.weight";
    let anchors = checkpoint
        .names()
        .filter(|name| name.ends_with(suffix))
        .collect::<Vec<_>>();
    let Some(anchor) = anchors.first() else {
        return Err(start(format!("missing weight {suffix}")));
    };
    if anchors.len() != 1 {
        return Err(start(
            "ambiguous checkpoint namespace: multiple embed_tokens.weight roots",
        ));
    }
    let root = if anchor.as_str() == format!("model.{suffix}") {
        "model."
    } else if anchor.as_str() == suffix {
        ""
    } else {
        return Err(start(format!(
            "unsupported decoder checkpoint namespace {anchor}; expected model. or bare root"
        )));
    };
    let opposite = if root.is_empty() { "model." } else { "" };
    let opposite_embed = format!("{opposite}embed_tokens.weight");
    let opposite_norm = format!("{opposite}norm.weight");
    let opposite_layers = format!("{opposite}layers.");
    if let Some(other) = checkpoint.names().find(|name| {
        name.as_str() == opposite_embed
            || name.as_str() == opposite_norm
            || name.starts_with(&opposite_layers)
    }) {
        return Err(start(format!(
            "mixed decoder checkpoint roots: found {other} alongside the {root:?} root"
        )));
    }
    Ok(root)
}

/// Bound the configured layer count by actual header names before allocating
/// any per-layer temporary collection or layer weight vector.
fn validate_checkpoint_capacity(
    config: &DecoderConfig,
    checkpoint: &Checkpoint,
) -> Result<(), BackendError> {
    // Per layer: six attention/norm tensors plus three MLP tensors. A routed
    // layer with fused experts needs exactly a router plus
    // `gate_up_proj`/`down_proj`, and separate per-expert storage only adds
    // names, so this fused lower bound never rejects a valid fused checkpoint.
    // Q/K RMSNorm scales are demanded later by `fetch_qk_norm`, which reports
    // the precise missing tensor instead of masking it behind capacity. Only
    // header names are counted; no tensor payload is read.
    const ATTENTION_NORMS: usize = 6;
    const MLP: usize = 3;
    let per_layer = ATTENTION_NORMS
        .checked_add(MLP)
        .ok_or_else(|| start("decoder checkpoint capacity overflow"))?;
    let mut minimum = 2usize;
    for layer in 0..config.num_hidden_layers {
        if let Some(routed) = &config.routed {
            routed.is_routed_layer(layer)?;
        }
        minimum = minimum
            .checked_add(per_layer)
            .ok_or_else(|| start("decoder checkpoint capacity overflow"))?;
    }
    let available = checkpoint.names().count();
    if available < minimum {
        return Err(start(format!(
            "checkpoint has {available} tensor names, fewer than the {minimum} required for {} decoder layers",
            config.num_hidden_layers
        )));
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn decoder_tensor_shapes(config: &DecoderConfig) -> Vec<(String, Vec<usize>)> {
    let h = config.hidden_size;
    let q = config.q_width();
    let kv = config.kv_width();
    let i = config.intermediate_size;
    let mut out = vec![
        ("embed_tokens.weight".into(), vec![config.vocab_size, h]),
        ("norm.weight".into(), vec![h]),
    ];
    for layer in 0..config.num_hidden_layers {
        let p = format!("layers.{layer}");
        out.extend([
            (format!("{p}.input_layernorm.weight"), vec![h]),
            (format!("{p}.post_attention_layernorm.weight"), vec![h]),
            (format!("{p}.self_attn.q_proj.weight"), vec![q, h]),
            (format!("{p}.self_attn.q_proj.bias"), vec![q]),
            (format!("{p}.self_attn.k_proj.weight"), vec![kv, h]),
            (format!("{p}.self_attn.k_proj.bias"), vec![kv]),
            (format!("{p}.self_attn.v_proj.weight"), vec![kv, h]),
            (format!("{p}.self_attn.v_proj.bias"), vec![kv]),
            (format!("{p}.self_attn.o_proj.weight"), vec![h, q]),
            (format!("{p}.self_attn.o_proj.bias"), vec![h]),
            (format!("{p}.mlp.gate_proj.weight"), vec![i, h]),
            (format!("{p}.mlp.gate_proj.bias"), vec![i]),
            (format!("{p}.mlp.up_proj.weight"), vec![i, h]),
            (format!("{p}.mlp.up_proj.bias"), vec![i]),
            (format!("{p}.mlp.down_proj.weight"), vec![h, i]),
            (format!("{p}.mlp.down_proj.bias"), vec![h]),
        ]);
        if config.family.requires_qk_norm() {
            out.extend([
                (
                    format!("{p}.self_attn.q_norm.weight"),
                    vec![QWEN3_QK_NORM_WIDTH],
                ),
                (
                    format!("{p}.self_attn.k_norm.weight"),
                    vec![QWEN3_QK_NORM_WIDTH],
                ),
            ]);
        }
    }
    out
}

fn fetch_linear(
    checkpoint: &Checkpoint,
    root: &str,
    name: &str,
    weight_shape: &[usize],
    bias_shape: &[usize],
) -> Result<DecoderLinear, BackendError> {
    let weight = checkpoint.source(&format!("{root}{name}.weight"), weight_shape)?;
    let bias = checkpoint.optional_source(&format!("{root}{name}.bias"), bias_shape)?;
    Ok(DecoderLinear { weight, bias })
}

/// Fetch the dense gate/up/down projections with the original bias pairing
/// validation. Routed layers never call this and so never demand dense MLP
/// weights.
fn fetch_dense_mlp(
    checkpoint: &Checkpoint,
    layer: usize,
    prefix: &str,
    hidden: usize,
    intermediate: usize,
) -> Result<DecoderMlpWeights, BackendError> {
    let gate = fetch_linear(
        checkpoint,
        prefix,
        ".mlp.gate_proj",
        &[intermediate, hidden],
        &[intermediate],
    )?;
    let up = fetch_linear(
        checkpoint,
        prefix,
        ".mlp.up_proj",
        &[intermediate, hidden],
        &[intermediate],
    )?;
    if gate.bias.is_some() != up.bias.is_some() {
        return Err(start(format!(
            "decoder layer {layer} gate/up projection biases must be both present or both absent"
        )));
    }
    let down = fetch_linear(
        checkpoint,
        prefix,
        ".mlp.down_proj",
        &[hidden, intermediate],
        &[hidden],
    )?;
    Ok(DecoderMlpWeights::Dense { gate, up, down })
}

fn fetch_qk_norm(
    checkpoint: &Checkpoint,
    layer: usize,
    prefix: &str,
    family: DecoderFamily,
) -> Result<(Option<WeightSource>, Option<WeightSource>), BackendError> {
    let q_name = format!("{prefix}.self_attn.q_norm.weight");
    let k_name = format!("{prefix}.self_attn.k_norm.weight");
    if !family.requires_qk_norm() {
        for name in [&q_name, &k_name] {
            if checkpoint.contains(name) {
                return Err(start(format!(
                    "decoder layer {layer} has {name}, but Q/K RMSNorm scales are only supported for Qwen3 and Qwen3-MoE"
                )));
            }
        }
        return Ok((None, None));
    }
    let shape = [QWEN3_QK_NORM_WIDTH];
    let q = checkpoint.source(&q_name, &shape)?;
    let k = checkpoint.source(&k_name, &shape)?;
    Ok((Some(q), Some(k)))
}

impl DecoderWeights {
    pub(crate) fn load(path: &Path) -> Result<Self, BackendError> {
        let config = DecoderConfig::load(path)?;
        validate_decoder_modules(path)?;
        let checkpoint = Checkpoint::open(path)?;
        Self::from_checkpoint(config, &checkpoint)
    }

    /// Cold classification admission deriving the backbone and the score head
    /// from one authoritative config JSON view and one opened checkpoint. The
    /// ordinary embedding constructor above continues to reject every
    /// classification architecture.
    pub(crate) fn load_classifier(
        path: &Path,
        dense_paths: Option<Vec<String>>,
        value: serde_json::Value,
    ) -> Result<(Self, DecoderClassifierHeadHost), BackendError> {
        let config = DecoderConfig::from_value(value.clone(), true)?;
        let head_config = DecoderClassifierConfig::from_value(value)?;
        validate_decoder_modules(path)?;
        reject_declared_dense_modules(path)?;
        let explicit_dense = dense_paths.as_ref().is_some_and(|paths| !paths.is_empty());
        let discovered = if explicit_dense {
            false
        } else {
            crate::dense::DenseChainHost::load_with_paths(path, config.hidden_size, dense_paths)?
                .is_some()
        };
        if classifier_dense_rejected(explicit_dense, discovered) {
            return Err(start(
                "decoder classifier does not accept Dense / Sentence-Transformers modules",
            ));
        }
        let checkpoint = Checkpoint::open(path)?;
        let weights = Self::from_checkpoint(config, &checkpoint)?;
        let head = DecoderClassifierHeadHost::load_from_checkpoint(&checkpoint, head_config)?;
        Ok((weights, head))
    }

    fn from_checkpoint(
        config: DecoderConfig,
        checkpoint: &Checkpoint,
    ) -> Result<Self, BackendError> {
        validate_checkpoint_capacity(&config, checkpoint)?;
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(config.num_hidden_layers)
            .map_err(start)?;
        let root = select_root(checkpoint)?;
        let h = config.hidden_size;
        let q = config.q_width();
        let kv = config.kv_width();
        let i = config.intermediate_size;

        let embed = checkpoint.source(
            &format!("{root}embed_tokens.weight"),
            &[config.vocab_size, h],
        )?;
        let norm = checkpoint.source(&format!("{root}norm.weight"), &[h])?;
        for layer in 0..config.num_hidden_layers {
            let p = format!("{root}layers.{layer}");
            let q_linear = fetch_linear(checkpoint, &p, ".self_attn.q_proj", &[q, h], &[q])?;
            let k_linear = fetch_linear(checkpoint, &p, ".self_attn.k_proj", &[kv, h], &[kv])?;
            let v_linear = fetch_linear(checkpoint, &p, ".self_attn.v_proj", &[kv, h], &[kv])?;
            if q_linear.bias.is_some() != k_linear.bias.is_some()
                || q_linear.bias.is_some() != v_linear.bias.is_some()
            {
                return Err(start(format!(
                    "decoder layer {layer} Q/K/V projection biases must be all present or all absent"
                )));
            }
            let mlp = match &config.routed {
                Some(routed) => {
                    match load_routed_layer(checkpoint, &format!("{p}."), routed, layer)? {
                        Some(weights) => DecoderMlpWeights::Routed(weights),
                        None => fetch_dense_mlp(checkpoint, layer, &p, h, i)?,
                    }
                }
                None => fetch_dense_mlp(checkpoint, layer, &p, h, i)?,
            };
            let (q_norm, k_norm) = fetch_qk_norm(checkpoint, layer, &p, config.family)?;
            layers.push(DecoderLayerWeights {
                input_norm: checkpoint.source(&format!("{p}.input_layernorm.weight"), &[h])?,
                post_attention_norm: checkpoint
                    .source(&format!("{p}.post_attention_layernorm.weight"), &[h])?,
                q_norm,
                k_norm,
                q: q_linear,
                k: k_linear,
                v: v_linear,
                o: fetch_linear(checkpoint, &p, ".self_attn.o_proj", &[h, q], &[h])?,
                mlp,
            });
        }
        Ok(Self {
            config,
            embed,
            norm,
            layers,
        })
    }
}

pub(crate) fn validate_decoder_modules(path: &Path) -> Result<(), BackendError> {
    let file = path.join("modules.json");
    if !path_present(&file)? {
        return Ok(());
    }
    #[derive(Deserialize)]
    struct Module {
        #[serde(rename = "type")]
        kind: String,
        path: Option<String>,
    }
    let modules: Vec<Module> = read_json(&file)?;
    // Admitted Dense projections are validated and executed by the shared Dense
    // chain loader. Every other module keeps the strict legacy pipeline order.
    let mut strict = Vec::new();
    for module in &modules {
        if crate::dense::is_dense_module_type(&module.kind) {
            continue;
        }
        if !matches!(
            module.kind.as_str(),
            "sentence_transformers.models.Transformer"
                | "sentence_transformers.models.Pooling"
                | "sentence_transformers.models.Normalize"
        ) {
            return Err(start(format!(
                "unsupported sentence-transformers module {}; projections and custom modules must not be ignored",
                module.kind
            )));
        }
        strict.push(module);
    }
    let transformer = strict
        .iter()
        .find(|module| module.kind == "sentence_transformers.models.Transformer")
        .ok_or_else(|| start("missing sentence-transformers Transformer module"))?;
    if transformer
        .path
        .as_deref()
        .is_some_and(|path| !path.is_empty())
    {
        return Err(start(
            "sentence-transformers Transformer module path must be the checkpoint root",
        ));
    }
    let kinds = strict
        .iter()
        .map(|module| module.kind.as_str())
        .collect::<Vec<_>>();
    match kinds.as_slice() {
        [
            "sentence_transformers.models.Transformer",
            "sentence_transformers.models.Pooling",
        ]
        | [
            "sentence_transformers.models.Transformer",
            "sentence_transformers.models.Pooling",
            "sentence_transformers.models.Normalize",
        ] => {}
        _ => {
            return Err(start(
                "unsupported sentence-transformers pipeline order: expected Transformer, Pooling and optional final Normalize",
            ))
        }
    }
    let pooling = strict
        .iter()
        .find(|module| module.kind == "sentence_transformers.models.Pooling")
        .ok_or_else(|| start("missing sentence-transformers Pooling module"))?;
    let relative = pooling.path.as_deref().unwrap_or("");
    let mut components = Path::new(relative).components();
    if !matches!(components.next(), Some(Component::Normal(_)))
        || components.next().is_some()
        || relative.contains('\\')
    {
        return Err(start("invalid sentence-transformers Pooling module path"));
    }
    validate_pooling_config(&path.join(relative).join("config.json"))
}

#[derive(Deserialize)]
struct PoolingConfig {
    #[serde(default)]
    pooling_mode_cls_token: bool,
    #[serde(default)]
    pooling_mode_mean_tokens: bool,
    #[serde(default)]
    pooling_mode_max_tokens: bool,
    #[serde(default)]
    pooling_mode_mean_sqrt_len_tokens: bool,
    #[serde(default)]
    pooling_mode_weightedmean_tokens: bool,
    #[serde(default)]
    pooling_mode_lasttoken: bool,
}

fn validate_pooling_config(path: &Path) -> Result<(), BackendError> {
    let config: PoolingConfig = read_json(path)?;
    let unsupported = [
        config.pooling_mode_cls_token,
        config.pooling_mode_mean_tokens,
        config.pooling_mode_max_tokens,
        config.pooling_mode_mean_sqrt_len_tokens,
        config.pooling_mode_weightedmean_tokens,
    ]
    .into_iter()
    .any(|enabled| enabled);
    if !config.pooling_mode_lasttoken || unsupported {
        return Err(start(
            "decoder pooled embedding task requires last-token pooling only",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{serialize, TensorView};
    use safetensors::Dtype;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "infinity-decoder-weights-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            let fixture = Self(path);
            fixture.raw_config(qwen2_config());
            fixture
        }

        fn raw_config(&self, value: serde_json::Value) {
            fs::write(
                self.0.join("config.json"),
                serde_json::to_vec(&value).unwrap(),
            )
            .unwrap();
        }

        fn config(&self, overrides: serde_json::Value) {
            self.raw_config(with_fields(qwen2_config(), overrides));
        }

        fn config_from_disk(&self) -> DecoderConfig {
            DecoderConfig::load(&self.0).unwrap()
        }

        fn write_entries(&self, entries: &[(String, Vec<usize>)], sharded: bool) {
            let data = entries
                .iter()
                .map(|(_, shape)| {
                    let count = shape.iter().product::<usize>();
                    (0..count)
                        .flat_map(|index| (((index % 11) as f32) / 16.0).to_le_bytes())
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let views = entries
                .iter()
                .zip(&data)
                .map(|((name, shape), bytes)| {
                    (
                        name.clone(),
                        TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
                    )
                })
                .collect::<Vec<_>>();
            if !sharded {
                fs::write(
                    self.0.join("model.safetensors"),
                    serialize(views.iter().map(|(n, v)| (n.clone(), v.clone())), None).unwrap(),
                )
                .unwrap();
                return;
            }
            let per_file = views.len().div_ceil(2).max(1);
            let mut map = BTreeMap::new();
            for (index, chunk) in views.chunks(per_file).enumerate() {
                let file = format!("model-{index}.safetensors");
                fs::write(
                    self.0.join(&file),
                    serialize(chunk.iter().map(|(n, v)| (n.clone(), v.clone())), None).unwrap(),
                )
                .unwrap();
                for (name, _) in chunk {
                    map.insert(name.clone(), file.clone());
                }
            }
            fs::write(
                self.0.join("model.safetensors.index.json"),
                serde_json::to_vec(&serde_json::json!({"weight_map": map})).unwrap(),
            )
            .unwrap();
        }

        fn write_modules(&self, modules: serde_json::Value) {
            fs::write(
                self.0.join("modules.json"),
                serde_json::to_vec(&modules).unwrap(),
            )
            .unwrap();
        }

        fn write_pooling(&self, lasttoken: bool, mean: bool) {
            let dir = self.0.join("1_Pooling");
            fs::create_dir_all(&dir).unwrap();
            fs::write(
                dir.join("config.json"),
                serde_json::to_vec(&serde_json::json!({
                    "pooling_mode_cls_token": false,
                    "pooling_mode_mean_tokens": mean,
                    "pooling_mode_max_tokens": false,
                    "pooling_mode_mean_sqrt_len_tokens": false,
                    "pooling_mode_weightedmean_tokens": false,
                    "pooling_mode_lasttoken": lasttoken
                }))
                .unwrap(),
            )
            .unwrap();
        }

        fn error(&self) -> String {
            match DecoderWeights::load(&self.0) {
                Ok(_) => panic!("invalid decoder fixture was accepted"),
                Err(error) => error.to_string(),
            }
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[derive(Clone, Copy, Default)]
    struct Biases {
        qkv: bool,
        gate_up: bool,
        o: bool,
        down: bool,
    }

    fn qwen2_config() -> serde_json::Value {
        serde_json::json!({
            "model_type": "qwen2",
            "architectures": ["Qwen2ForCausalLM"],
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "vocab_size": 12,
            "max_position_embeddings": 32,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1_000_000.0,
            "hidden_act": "silu",
            "is_causal": false
        })
    }

    fn llama_config() -> serde_json::Value {
        serde_json::json!({
            "model_type": "llama",
            "architectures": ["LlamaModel"],
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "vocab_size": 12,
            "max_position_embeddings": 16,
            "rms_norm_eps": 1e-5
        })
    }

    fn mistral_config() -> serde_json::Value {
        serde_json::json!({
            "model_type": "mistral",
            "architectures": ["MistralForCausalLM"],
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 2,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 32000,
            "max_position_embeddings": 32768,
            "sliding_window": 4096
        })
    }

    fn mistral_small_config() -> serde_json::Value {
        serde_json::json!({
            "model_type": "mistral",
            "architectures": ["MistralForCausalLM"],
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "vocab_size": 12,
            "max_position_embeddings": 8192,
            "sliding_window": 4096
        })
    }

    fn qwen3_config() -> serde_json::Value {
        serde_json::json!({
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 128,
            "vocab_size": 12,
            "max_position_embeddings": 32,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1_000_000.0,
            "hidden_act": "silu"
        })
    }

    fn with_fields(mut value: serde_json::Value, fields: serde_json::Value) -> serde_json::Value {
        for (key, field) in fields.as_object().unwrap() {
            value[key] = field.clone();
        }
        value
    }

    fn qwen3_moe_config() -> serde_json::Value {
        with_fields(
            qwen3_config(),
            serde_json::json!({
                "model_type": "qwen3_moe",
                "architectures": ["Qwen3MoeForCausalLM"],
                "num_hidden_layers": 2,
                "decoder_sparse_step": 2,
                "moe_intermediate_size": 4,
                "num_experts": 2,
                "num_experts_per_tok": 1,
                "mlp_only_layers": [],
                "norm_topk_prob": true,
                "router_aux_loss_coef": 0.001,
                "shared_expert_intermediate_size": null
            }),
        )
    }

    #[test]
    fn loads_qwen3_235b_a22b_text_config_structurally_without_weights() {
        let fixture = Fixture::new();
        fixture.raw_config(serde_json::json!({
            "architectures": ["Qwen3MoeForCausalLM"],
            "attention_bias": false,
            "attention_dropout": 0.0,
            "decoder_sparse_step": 1,
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "max_position_embeddings": 40960,
            "mlp_only_layers": [],
            "model_type": "qwen3_moe",
            "moe_intermediate_size": 1536,
            "norm_topk_prob": true,
            "num_attention_heads": 64,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "num_hidden_layers": 94,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-6,
            "rope_scaling": null,
            "rope_theta": 1000000.0,
            "torch_dtype": "bfloat16",
            "vocab_size": 151936
        }));
        let config = fixture.config_from_disk();
        assert_eq!(config.family, DecoderFamily::Qwen3Moe);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.intermediate_size, 12288);
        assert_eq!(
            (config.num_attention_heads, config.num_key_value_heads),
            (64, 4)
        );
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.num_hidden_layers, 94);
        let routed = config.routed.as_ref().unwrap();
        assert_eq!((routed.num_experts, routed.num_experts_per_tok), (128, 8));
        assert_eq!(routed.moe_intermediate_size, 1536);
        assert_eq!(routed.hidden_size, 4096);
        assert_eq!(
            crate::decoder::DecoderMath::select(&config).unwrap(),
            crate::decoder::DecoderMath::Qwen3MoeH4096Ordinary
        );
    }

    fn entries(config: &DecoderConfig, root: &str, biases: Biases) -> Vec<(String, Vec<usize>)> {
        decoder_tensor_shapes(config)
            .into_iter()
            .filter(|(name, _)| {
                if !name.ends_with(".bias") {
                    return true;
                }
                if name.ends_with(".self_attn.q_proj.bias")
                    || name.ends_with(".self_attn.k_proj.bias")
                    || name.ends_with(".self_attn.v_proj.bias")
                {
                    biases.qkv
                } else if name.ends_with(".mlp.gate_proj.bias")
                    || name.ends_with(".mlp.up_proj.bias")
                {
                    biases.gate_up
                } else if name.ends_with(".self_attn.o_proj.bias") {
                    biases.o
                } else if name.ends_with(".mlp.down_proj.bias") {
                    biases.down
                } else {
                    false
                }
            })
            .map(|(name, shape)| (format!("{root}{name}"), shape))
            .collect()
    }

    fn write_weights(fixture: &Fixture, root: &str, biases: Biases, sharded: bool) {
        let config = fixture.config_from_disk();
        fixture.write_entries(&entries(&config, root, biases), sharded);
    }

    /// Build a real safetensors header for mixed dense/routed Qwen3-MoE layers.
    /// Routed layers drop their dense gate/up/down names and gain either fused
    /// `gate_up_proj`/`down_proj` views or per-expert projections.
    fn moe_entries(
        config: &DecoderConfig,
        routed: &RoutedMlpConfig,
        root: &str,
        fused: bool,
    ) -> Vec<(String, Vec<usize>)> {
        let mut shapes = decoder_tensor_shapes(config);
        let mut routed_tensors = Vec::new();
        for layer in 0..config.num_hidden_layers {
            if !routed.is_routed_layer(layer).unwrap() {
                continue;
            }
            let local = format!("layers.{layer}.mlp.");
            shapes.retain(|(name, _)| !name.starts_with(&local));
            let p = format!("{root}layers.{layer}.mlp");
            routed_tensors.push((
                format!("{p}.gate.weight"),
                vec![routed.num_experts, config.hidden_size],
            ));
            if fused {
                routed_tensors.push((
                    format!("{p}.experts.gate_up_proj"),
                    vec![
                        routed.num_experts,
                        routed.moe_intermediate_size * 2,
                        config.hidden_size,
                    ],
                ));
                routed_tensors.push((
                    format!("{p}.experts.down_proj"),
                    vec![
                        routed.num_experts,
                        config.hidden_size,
                        routed.moe_intermediate_size,
                    ],
                ));
            } else {
                for expert in 0..routed.num_experts {
                    let e = format!("{p}.experts.{expert}");
                    routed_tensors.push((
                        format!("{e}.gate_proj.weight"),
                        vec![routed.moe_intermediate_size, config.hidden_size],
                    ));
                    routed_tensors.push((
                        format!("{e}.up_proj.weight"),
                        vec![routed.moe_intermediate_size, config.hidden_size],
                    ));
                    routed_tensors.push((
                        format!("{e}.down_proj.weight"),
                        vec![config.hidden_size, routed.moe_intermediate_size],
                    ));
                }
            }
        }
        let mut out = shapes
            .into_iter()
            .map(|(name, shape)| (format!("{root}{name}"), shape))
            .collect::<Vec<_>>();
        out.extend(routed_tensors);
        out
    }

    #[test]
    fn loads_qwen2_causal_backbone_with_qkv_biases_and_lm_head_extra() {
        let fixture = Fixture::new();
        fixture.config(serde_json::json!({"is_causal": true}));
        let config = fixture.config_from_disk();
        let mut weights = entries(
            &config,
            "model.",
            Biases {
                qkv: true,
                ..Biases::default()
            },
        );
        weights.push(("lm_head.weight".into(), vec![12, 8]));
        fixture.write_entries(&weights, false);

        let loaded = DecoderWeights::load(&fixture.0).unwrap();
        assert_eq!(loaded.config.family, DecoderFamily::Qwen2);
        assert_eq!(loaded.config.head_dim, 4);
        assert_eq!(loaded.config.q_width(), 8);
        assert_eq!(loaded.config.kv_width(), 4);
        assert!(loaded.config.is_causal);
        assert_eq!(loaded.config.rope, DecoderRope::FullNeoX);
        assert_eq!(loaded.layers.len(), 2);
        assert_eq!(loaded.embed.shape(), &[12, 8]);
        assert_eq!(loaded.norm.shape(), &[8]);
        assert_eq!(loaded.layers[0].input_norm.shape(), &[8]);
        assert_eq!(loaded.layers[0].post_attention_norm.shape(), &[8]);
        assert_eq!(loaded.layers[0].q.weight.shape(), &[8, 8]);
        assert_eq!(loaded.layers[0].k.weight.shape(), &[4, 8]);
        assert_eq!(loaded.layers[0].v.weight.shape(), &[4, 8]);
        assert_eq!(loaded.layers[0].o.weight.shape(), &[8, 8]);
        let (gate, up, down) = loaded.layers[0].mlp.dense().unwrap();
        assert_eq!(gate.weight.shape(), &[16, 8]);
        assert_eq!(up.weight.shape(), &[16, 8]);
        assert_eq!(down.weight.shape(), &[8, 16]);
        assert_eq!(loaded.layers[0].q.bias.as_ref().unwrap().shape(), &[8]);
        assert_eq!(loaded.layers[0].k.bias.as_ref().unwrap().shape(), &[4]);
        assert_eq!(loaded.layers[0].v.bias.as_ref().unwrap().shape(), &[4]);
        assert!(gate.bias.is_none());
        assert!(loaded.layers[0].o.bias.is_none());
        assert!(down.bias.is_none());
    }

    #[test]
    fn loads_bias_free_bare_llama_with_causal_and_silu_defaults() {
        let fixture = Fixture::new();
        fixture.raw_config(llama_config());
        write_weights(&fixture, "", Biases::default(), false);

        let loaded = DecoderWeights::load(&fixture.0).unwrap();
        assert_eq!(loaded.config.family, DecoderFamily::Llama);
        assert_eq!(loaded.config.head_dim, 4);
        assert!(loaded.config.is_causal);
        assert_eq!(loaded.config.rope, DecoderRope::Default);
        assert_eq!(loaded.config.rope_theta, 10_000.0);
        assert_eq!(loaded.layers[0].k.weight.shape(), &[4, 8]);
        assert!(loaded.layers[0].q.bias.is_none());
        assert!(loaded.layers[0].k.bias.is_none());
        assert!(loaded.layers[0].v.bias.is_none());
        assert!(loaded.layers[0].mlp.dense().unwrap().0.bias.is_none());
    }

    #[test]
    fn loads_qwen3_backbone_with_complete_qk_norm_and_independent_widths() {
        let fixture = Fixture::new();
        fixture.raw_config(qwen3_config());
        write_weights(&fixture, "model.", Biases::default(), false);

        let loaded = DecoderWeights::load(&fixture.0).unwrap();
        assert_eq!(loaded.config.family, DecoderFamily::Qwen3);
        assert_eq!(loaded.config.head_dim, 128);
        assert_eq!(loaded.config.q_width(), 256);
        assert_eq!(loaded.config.kv_width(), 128);
        assert!(loaded.config.is_causal);
        assert_eq!(loaded.config.rope, DecoderRope::FullNeoX);
        assert_eq!(loaded.config.rope_theta, 1_000_000.0);
        assert_eq!(loaded.layers[0].q.weight.shape(), &[256, 8]);
        assert_eq!(loaded.layers[0].k.weight.shape(), &[128, 8]);
        assert_eq!(loaded.layers[0].v.weight.shape(), &[128, 8]);
        assert_eq!(loaded.layers[0].o.weight.shape(), &[8, 256]);
        assert_eq!(loaded.layers[0].q_norm.as_ref().unwrap().shape(), &[128]);
        assert_eq!(loaded.layers[0].k_norm.as_ref().unwrap().shape(), &[128]);
        assert!(loaded.layers[0].q.bias.is_none());
        assert!(loaded.layers[0].k.bias.is_none());
        assert!(loaded.layers[0].v.bias.is_none());
        assert!(loaded.layers[0].o.bias.is_none());
        assert!(loaded.layers[0].mlp.dense().unwrap().0.bias.is_none());
    }

    #[test]
    fn loads_qwen3_moe_mixed_dense_and_routed_layers() {
        use crate::moe_model::GateUpSources;
        for fused in [false, true] {
            let fixture = Fixture::new();
            fixture.raw_config(qwen3_moe_config());
            let config = fixture.config_from_disk();
            assert_eq!(config.family, DecoderFamily::Qwen3Moe);
            assert_eq!(config.head_dim, 128);
            assert_eq!(config.q_width(), 256);
            assert_eq!(config.kv_width(), 128);
            assert!(config.is_causal);
            assert_eq!(config.rope, DecoderRope::FullNeoX);
            assert_eq!(config.rope_theta, 1_000_000.0);
            let routed = config.routed.as_ref().unwrap();
            assert_eq!((routed.num_experts, routed.num_experts_per_tok), (2, 1));
            assert!(!routed.is_routed_layer(0).unwrap());
            assert!(routed.is_routed_layer(1).unwrap());

            let weights = moe_entries(&config, routed, "model.", fused);
            fixture.write_entries(&weights, false);
            let loaded = DecoderWeights::load(&fixture.0).unwrap();
            assert_eq!(loaded.layers.len(), 2);
            assert!(matches!(
                &loaded.layers[0].mlp,
                DecoderMlpWeights::Dense { .. }
            ));
            let (gate, up, down) = loaded.layers[0].mlp.dense().unwrap();
            assert_eq!(gate.weight.shape(), &[16, 8]);
            assert_eq!(up.weight.shape(), &[16, 8]);
            assert_eq!(down.weight.shape(), &[8, 16]);
            assert_eq!(loaded.layers[1].q.weight.shape(), &[256, 8]);
            assert_eq!(loaded.layers[1].o.weight.shape(), &[8, 256]);
            assert_eq!(loaded.layers[1].q_norm.as_ref().unwrap().shape(), &[128]);
            assert!(loaded.layers[1].mlp.dense().is_err());
            match &loaded.layers[1].mlp {
                DecoderMlpWeights::Routed(routed_layer) => {
                    assert!(routed_layer.expert(1).is_ok());
                    assert!(routed_layer.expert(2).is_err());
                    let expert = routed_layer.expert(0).unwrap();
                    match expert.gate_up {
                        GateUpSources::Packed(packed) => {
                            assert!(fused);
                            assert_eq!(packed.shape(), &[8, 8]);
                        }
                        GateUpSources::Separate { gate, up } => {
                            assert!(!fused);
                            assert_eq!(gate.shape(), &[4, 8]);
                            assert_eq!(up.shape(), &[4, 8]);
                        }
                    }
                    assert_eq!(expert.down.shape(), &[8, 4]);
                }
                DecoderMlpWeights::Dense { .. } => panic!("layer 1 must be routed"),
            }
        }
    }

    #[test]
    fn qwen3_moe_malformed_fused_wins_over_valid_separate_experts() {
        let fixture = Fixture::new();
        fixture.raw_config(qwen3_moe_config());
        let config = fixture.config_from_disk();
        let routed = config.routed.as_ref().unwrap();
        let mut weights = moe_entries(&config, routed, "model.", false);
        weights.push((
            "model.layers.1.mlp.experts.gate_up_proj".into(),
            vec![routed.num_experts, routed.moe_intermediate_size * 2],
        ));
        fixture.write_entries(&weights, false);
        let error = fixture.error();
        assert!(error.contains("wrong shape"), "{error}");
    }

    #[test]
    fn qwen3_moe_rejects_unsupported_routed_metadata() {
        for (overrides, needle) in [
            (
                serde_json::json!({"num_shared_experts": 1}),
                "not supported",
            ),
            (
                serde_json::json!({"decoder_sparse_step": 0}),
                "decoder_sparse_step",
            ),
            (
                serde_json::json!({"mlp_only_layers": [2]}),
                "mlp_only_layers entry out of range",
            ),
            (serde_json::json!({"quantization": {}}), "not supported"),
            (
                serde_json::json!({"num_experts_per_tok": 3}),
                "num_experts_per_tok must not exceed num_experts",
            ),
        ] {
            let fixture = Fixture::new();
            fixture.raw_config(with_fields(qwen3_moe_config(), overrides));
            let error = DecoderConfig::load(&fixture.0).unwrap_err().to_string();
            assert!(error.contains(needle), "{error}");
        }
    }

    #[test]
    fn normalizes_bidirectional_pplx_qwen3_alias_with_native_precedence() {
        let fixture = Fixture::new();
        let mut value = with_fields(
            qwen3_config(),
            serde_json::json!({
                "model_type": "bidirectional_pplx_qwen3",
                "use_bidirectional_attention": false,
                "is_causal": true,
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": 500000.0
                }
            }),
        );
        value.as_object_mut().unwrap().remove("rope_theta");
        fixture.raw_config(value);
        let config = fixture.config_from_disk();
        assert_eq!(config.family, DecoderFamily::Qwen3);
        assert!(
            !config.is_causal,
            "Native alias forces non-causal attention"
        );
        assert_eq!(
            config.rope_theta, 500000.0,
            "Native falls back to the saved rope_parameters theta"
        );
        assert_eq!(config.rope, DecoderRope::FullNeoX);

        fixture.raw_config(with_fields(
            qwen3_config(),
            serde_json::json!({
                "model_type": "bidirectional_pplx_qwen3",
                "rope_theta": 2000000.0,
                "rope_parameters": {"rope_theta": 500000.0}
            }),
        ));
        let config = fixture.config_from_disk();
        assert_eq!(
            config.rope_theta, 2_000_000.0,
            "an explicit rope_theta wins over the saved fallback"
        );
        assert_eq!(config.family, DecoderFamily::Qwen3);
        assert!(!config.is_causal);
    }

    #[test]
    fn qwen_ignores_rope_scaling_but_rejects_yarn_parameters_and_other_families() {
        for value in [qwen2_config(), qwen3_config()] {
            let fixture = Fixture::new();
            fixture.raw_config(with_fields(
                value,
                serde_json::json!({"rope_scaling": {"type": "llama3", "factor": 8.0}}),
            ));
            let config = fixture.config_from_disk();
            assert_eq!(config.rope, DecoderRope::FullNeoX);
        }

        let fixture = Fixture::new();
        fixture.raw_config(with_fields(
            llama_config(),
            serde_json::json!({"rope_scaling": {"type": "llama3", "factor": 8.0}}),
        ));
        let error = DecoderConfig::load(&fixture.0).unwrap_err().to_string();
        assert!(error.contains("scaled decoder RoPE"), "{error}");

        for value in [qwen2_config(), qwen3_config(), llama_config()] {
            let fixture = Fixture::new();
            fixture.raw_config(with_fields(
                value,
                serde_json::json!({
                    "rope_parameters": {
                        "rope_type": "yarn",
                        "rope_theta": 1000000.0,
                        "factor": 4.0
                    }
                }),
            ));
            let error = DecoderConfig::load(&fixture.0).unwrap_err().to_string();
            assert!(error.contains("rope_parameters scaling yarn"), "{error}");
            assert!(error.contains("YaRN"), "{error}");
        }
    }

    #[test]
    fn rejects_partial_or_malformed_qwen3_qk_norm_weights() {
        let fixture = Fixture::new();
        fixture.raw_config(qwen3_config());
        let config = fixture.config_from_disk();
        let mut missing = entries(&config, "model.", Biases::default());
        missing.retain(|(name, _)| !name.ends_with(".self_attn.k_norm.weight"));
        fixture.write_entries(&missing, false);
        let error = fixture.error();
        assert!(
            error.contains("missing weight model.layers.0.self_attn.k_norm.weight"),
            "{error}"
        );

        let fixture = Fixture::new();
        fixture.raw_config(qwen3_config());
        let config = fixture.config_from_disk();
        let mut malformed = entries(&config, "model.", Biases::default());
        for (name, shape) in &mut malformed {
            if name == "model.layers.0.self_attn.q_norm.weight" {
                *shape = vec![64];
            }
        }
        fixture.write_entries(&malformed, false);
        let error = fixture.error();
        assert!(error.contains("wrong shape"), "{error}");
    }

    #[test]
    fn rejects_qk_norm_weights_on_qwen2_and_llama_checkpoints() {
        for config in [qwen2_config(), llama_config()] {
            let fixture = Fixture::new();
            fixture.raw_config(config);
            let config = fixture.config_from_disk();
            assert!(!config.family.requires_qk_norm());
            let mut weights = entries(&config, "model.", Biases::default());
            weights.push((
                "model.layers.0.self_attn.q_norm.weight".into(),
                vec![QWEN3_QK_NORM_WIDTH],
            ));
            fixture.write_entries(&weights, false);
            let error = fixture.error();
            assert!(error.contains("only supported for Qwen3"), "{error}");
        }
    }

    #[test]
    fn qwen3_geometry_metadata_needs_no_model_weights() {
        for (hidden, intermediate, layers, heads, kv_heads) in [
            (1024usize, 3072, 28, 16, 8),
            (2048, 6144, 28, 16, 8),
            (2560, 9728, 36, 32, 8),
            (4096, 12288, 36, 32, 8),
        ] {
            let fixture = Fixture::new();
            fixture.raw_config(serde_json::json!({
                "model_type": "qwen3",
                "architectures": ["Qwen3ForCausalLM"],
                "hidden_size": hidden,
                "intermediate_size": intermediate,
                "num_hidden_layers": layers,
                "num_attention_heads": heads,
                "num_key_value_heads": kv_heads,
                "head_dim": 128,
                "vocab_size": 151936,
                "max_position_embeddings": 40960,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000.0,
                "hidden_act": "silu"
            }));
            let config = fixture.config_from_disk();
            assert_eq!(config.family, DecoderFamily::Qwen3);
            assert_eq!(config.head_dim, 128);
            assert_eq!(config.q_width(), heads * 128);
            assert_eq!(config.kv_width(), kv_heads * 128);
            assert_eq!(config.num_hidden_layers, layers);
            assert_eq!(config.intermediate_size, intermediate);
            assert_eq!(config.rope_theta, 1_000_000.0);
            assert!(config.is_causal);
        }
    }

    #[test]
    fn rejects_quantized_unported_architectures_and_moe_keys_on_dense_families() {
        for (overrides, needle) in [
            (
                serde_json::json!({"model_type": "qwen2_moe"}),
                "unsupported model_type",
            ),
            (
                serde_json::json!({"quantization_config": {"quant_method": "gptq"}}),
                "quantized decoder",
            ),
            (
                serde_json::json!({"architectures": ["Qwen3ForSequenceClassification"]}),
                "unsupported decoder task/architecture",
            ),
            (
                serde_json::json!({"num_experts": 8}),
                "unsupported decoder config modifier",
            ),
        ] {
            let fixture = Fixture::new();
            fixture.raw_config(with_fields(qwen3_config(), overrides));
            let error = DecoderConfig::load(&fixture.0).unwrap_err().to_string();
            assert!(error.contains(needle), "{error}");
        }
    }

    #[test]
    fn mask_flags_match_native_family_defaults_and_explicit_overrides() {
        let fixture = Fixture::new();
        fixture.raw_config(with_fields(
            llama_config(),
            serde_json::json!({"use_bidirectional_attention": true}),
        ));
        write_weights(&fixture, "", Biases::default(), false);
        assert!(!DecoderWeights::load(&fixture.0).unwrap().config.is_causal);

        for family in [
            DecoderFamily::Qwen2,
            DecoderFamily::Qwen3,
            DecoderFamily::Qwen3Moe,
            DecoderFamily::Llama,
            DecoderFamily::Mistral,
        ] {
            for bidirectional in [None, Some(false), Some(true)] {
                for causal in [None, Some(false), Some(true)] {
                    let mut value = match family {
                        DecoderFamily::Qwen2 => qwen2_config(),
                        DecoderFamily::Qwen3 => qwen3_config(),
                        DecoderFamily::Qwen3Moe => qwen3_moe_config(),
                        DecoderFamily::Llama => llama_config(),
                        DecoderFamily::Mistral => mistral_config(),
                    };
                    let object = value.as_object_mut().unwrap();
                    object.remove("is_causal");
                    if let Some(flag) = bidirectional {
                        object.insert("use_bidirectional_attention".into(), flag.into());
                    }
                    if let Some(flag) = causal {
                        object.insert("is_causal".into(), flag.into());
                    }
                    fixture.raw_config(value);
                    let loaded = DecoderConfig::load(&fixture.0).unwrap();
                    let default = match family {
                        DecoderFamily::Qwen2 => true,
                        DecoderFamily::Qwen3
                        | DecoderFamily::Qwen3Moe
                        | DecoderFamily::Llama
                        | DecoderFamily::Mistral => !bidirectional.unwrap_or(false),
                    };
                    assert_eq!(
                        loaded.is_causal,
                        causal.unwrap_or(default),
                        "family={family:?} bidirectional={bidirectional:?} causal={causal:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn honors_explicit_noncausal_qwen2_and_last_token_modules() {
        let fixture = Fixture::new();
        let config = fixture.config_from_disk();
        assert!(!config.is_causal);
        fixture.write_entries(
            &entries(
                &config,
                "model.",
                Biases {
                    qkv: true,
                    ..Biases::default()
                },
            ),
            false,
        );
        fixture.write_modules(serde_json::json!([
            {"type": "sentence_transformers.models.Transformer", "path": ""},
            {"type": "sentence_transformers.models.Pooling", "path": "1_Pooling"},
            {"type": "sentence_transformers.models.Normalize", "path": "2_Normalize"}
        ]));
        fixture.write_pooling(true, false);

        let loaded = DecoderWeights::load(&fixture.0).unwrap();
        assert_eq!(loaded.config.family, DecoderFamily::Qwen2);
        assert!(!loaded.config.is_causal);
        assert_eq!(loaded.config.rope, DecoderRope::FullNeoX);
    }

    #[test]
    fn loads_sharded_checkpoint_and_enforces_index_ownership() {
        let fixture = Fixture::new();
        write_weights(
            &fixture,
            "model.",
            Biases {
                qkv: true,
                ..Biases::default()
            },
            true,
        );
        let loaded = DecoderWeights::load(&fixture.0).unwrap();
        assert_eq!(loaded.layers.len(), 2);
        assert_eq!(loaded.layers[1].q.bias.as_ref().unwrap().shape(), &[8]);

        let index_path = fixture.0.join("model.safetensors.index.json");
        let mut index: serde_json::Value =
            serde_json::from_slice(&fs::read(&index_path).unwrap()).unwrap();
        let map = index["weight_map"].as_object_mut().unwrap();
        let (name, file) = map.iter().next().unwrap();
        let name = name.clone();
        let other = if file.as_str() == Some("model-0.safetensors") {
            "model-1.safetensors"
        } else {
            "model-0.safetensors"
        };
        map.insert(name, serde_json::json!(other));
        fs::write(&index_path, serde_json::to_vec(&index).unwrap()).unwrap();

        let error = fixture.error();
        assert!(error.contains("not in its indexed shard"), "{error}");
    }

    #[test]
    fn rejects_partial_qkv_and_gate_up_biases_but_preserves_o_and_down() {
        let fixture = Fixture::new();
        let config = fixture.config_from_disk();
        let mut partial_qkv = entries(&config, "model.", Biases::default());
        partial_qkv.push((
            "model.layers.0.self_attn.q_proj.bias".into(),
            vec![config.q_width()],
        ));
        fixture.write_entries(&partial_qkv, false);
        let error = fixture.error();
        assert!(error.contains("Q/K/V projection biases"), "{error}");

        let fixture = Fixture::new();
        let config = fixture.config_from_disk();
        let mut partial_gate = entries(&config, "model.", Biases::default());
        partial_gate.push((
            "model.layers.0.mlp.gate_proj.bias".into(),
            vec![config.intermediate_size],
        ));
        fixture.write_entries(&partial_gate, false);
        let error = fixture.error();
        assert!(error.contains("gate/up projection biases"), "{error}");

        let fixture = Fixture::new();
        let config = fixture.config_from_disk();
        let mut optional = entries(
            &config,
            "model.",
            Biases {
                o: true,
                down: true,
                ..Biases::default()
            },
        );
        optional.retain(|(name, _)| {
            !(name == "model.layers.1.self_attn.o_proj.bias"
                || name == "model.layers.0.mlp.down_proj.bias")
        });
        fixture.write_entries(&optional, false);
        let loaded = DecoderWeights::load(&fixture.0).unwrap();
        assert!(loaded.layers[0].o.bias.is_some());
        assert!(loaded.layers[0].mlp.dense().unwrap().2.bias.is_none());
        assert!(loaded.layers[1].o.bias.is_none());
        assert!(loaded.layers[1].mlp.dense().unwrap().2.bias.is_some());
    }

    #[test]
    fn checkpoint_bias_presence_controls_plain_projection_need() {
        use crate::decoder::BiasSupport;

        let fixture = Fixture::new();
        write_weights(&fixture, "model.", Biases::default(), false);
        let none = BiasSupport::from_layers(&DecoderWeights::load(&fixture.0).unwrap());
        assert_eq!(
            none,
            BiasSupport {
                qkv: false,
                attention: false,
                gate_up: false,
                down: false,
                requires_plain: true
            }
        );
        assert!(none.uses_plain());

        let fixture = Fixture::new();
        write_weights(
            &fixture,
            "model.",
            Biases {
                qkv: true,
                gate_up: true,
                o: true,
                down: true,
            },
            false,
        );
        let all = BiasSupport::from_layers(&DecoderWeights::load(&fixture.0).unwrap());
        assert!(all.qkv && all.attention && all.gate_up && all.down);
        assert!(!all.uses_plain());

        let fixture = Fixture::new();
        let config = fixture.config_from_disk();
        let mut mixed = entries(
            &config,
            "model.",
            Biases {
                qkv: true,
                gate_up: true,
                o: true,
                down: true,
            },
        );
        mixed.retain(|(name, _)| !name.starts_with("model.layers.1.self_attn.o_proj.bias"));
        mixed.retain(|(name, _)| !name.starts_with("model.layers.1.mlp.down_proj.bias"));
        mixed.retain(|(name, _)| !name.starts_with("model.layers.0.self_attn.q_proj.bias"));
        mixed.retain(|(name, _)| !name.starts_with("model.layers.0.self_attn.k_proj.bias"));
        mixed.retain(|(name, _)| !name.starts_with("model.layers.0.self_attn.v_proj.bias"));
        fixture.write_entries(&mixed, false);
        // Per-layer bias groups remain complete, but which layers carry them
        // may vary. The union still needs all four LtBias plans, while layer 0's
        // absent QKV bias still requires a plain cuBLAS projection.
        let mixed = BiasSupport::from_layers(&DecoderWeights::load(&fixture.0).unwrap());
        assert!(mixed.qkv && mixed.attention && mixed.gate_up && mixed.down);
        assert!(mixed.uses_plain());
    }

    #[test]
    fn rejects_positive_native_llama_windows_even_when_flag_is_false() {
        for window in [1, 4096, 32768] {
            let raw = serde_json::json!({
                "model_type": "llama",
                "hidden_size": 2048,
                "intermediate_size": 8192,
                "num_hidden_layers": 1,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "vocab_size": 32000,
                "max_position_embeddings": 32768,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000.0,
                "sliding_window": window,
                "use_sliding_window": false
            });
            let error = DecoderConfig::from_value(raw, false).unwrap_err();
            assert!(error.to_string().contains("sliding-window"));
        }
    }

    #[test]
    fn rejects_unsupported_config_modifiers_and_tasks() {
        for (overrides, needle) in [
            (
                serde_json::json!({"hidden_act": "gelu"}),
                "unsupported decoder activation",
            ),
            (
                serde_json::json!({"quantization_config": {"quant_method": "gptq"}}),
                "quantized decoder",
            ),
            (
                serde_json::json!({"rope_parameters": {"rope_type": "yarn"}}),
                "rope_parameters",
            ),
            (
                serde_json::json!({
                    "model_type": "bidirectional_pplx_qwen3",
                    "architectures": ["Qwen3ForSequenceClassification"]
                }),
                "unsupported decoder task/architecture",
            ),
            (
                serde_json::json!({"partial_rotary_factor": 0.5}),
                "partial rotary",
            ),
            (
                serde_json::json!({"use_sliding_window": true}),
                "sliding-window",
            ),
            (
                serde_json::json!({"num_experts": 4}),
                "unsupported decoder config modifier",
            ),
            (
                serde_json::json!({"layer_types": ["full_attention"]}),
                "unsupported decoder config modifier",
            ),
            (
                serde_json::json!({"vision_config": {}}),
                "unsupported decoder config modifier",
            ),
            (
                serde_json::json!({"architectures": ["Qwen2ForSequenceClassification"]}),
                "unsupported decoder task/architecture",
            ),
            (
                serde_json::json!({"rope_type": "linear"}),
                "unsupported decoder rope_type",
            ),
            (
                serde_json::json!({"model_type": "qwen2_moe"}),
                "unsupported model_type",
            ),
        ] {
            let fixture = Fixture::new();
            fixture.config(overrides);
            let error = DecoderConfig::load(&fixture.0).unwrap_err().to_string();
            assert!(error.contains(needle), "{error}");
        }
    }

    #[test]
    fn rejects_non_last_token_and_custom_sentence_transformer_modules() {
        let fixture = Fixture::new();
        fixture.write_modules(serde_json::json!([
            {"type": "sentence_transformers.models.Transformer"},
            {"type": "sentence_transformers.models.CustomProjection"}
        ]));
        let error = fixture.error();
        assert!(error.contains("must not be ignored"), "{error}");

        let fixture = Fixture::new();
        fixture.write_modules(serde_json::json!([
            {"type": "sentence_transformers.models.Transformer"},
            {"type": "sentence_transformers.models.Normalize"},
            {"type": "sentence_transformers.models.Pooling", "path": "1_Pooling"}
        ]));
        let error = fixture.error();
        assert!(error.contains("pipeline order"), "{error}");

        let fixture = Fixture::new();
        fixture.write_modules(serde_json::json!([
            {"type": "sentence_transformers.models.Transformer"},
            {"type": "sentence_transformers.models.Pooling", "path": "1_Pooling"}
        ]));
        fixture.write_pooling(false, true);
        let error = fixture.error();
        assert!(error.contains("last-token pooling only"), "{error}");

        let fixture = Fixture::new();
        fixture.write_modules(serde_json::json!([
            {"type": "sentence_transformers.models.Transformer", "path": "subdir"},
            {"type": "sentence_transformers.models.Pooling", "path": "1_Pooling"}
        ]));
        fixture.write_pooling(true, false);
        let error = fixture.error();
        assert!(
            error.contains("Transformer module path must be the checkpoint root"),
            "{error}"
        );
    }

    #[test]
    fn admits_dense_entries_for_the_shared_dense_loader() {
        let fixture = Fixture::new();
        let config = fixture.config_from_disk();
        fixture.write_entries(&entries(&config, "model.", Biases::default()), false);
        fixture.write_modules(serde_json::json!([
            {"type": "sentence_transformers.models.Transformer", "path": ""},
            {"type": "sentence_transformers.models.Pooling", "path": "1_Pooling"},
            {"type": "sentence_transformers.models.Dense", "path": "2_Dense"}
        ]));
        fixture.write_pooling(true, false);
        let loaded = DecoderWeights::load(&fixture.0).unwrap();
        assert_eq!(loaded.layers.len(), config.num_hidden_layers);
    }

    #[test]
    fn rejects_malformed_geometry_and_tensor_shapes() {
        for (overrides, needle) in [
            (serde_json::json!({"hidden_size": 0}), "must be nonzero"),
            (
                serde_json::json!({"num_attention_heads": 3}),
                "divide evenly",
            ),
            (serde_json::json!({"head_dim": 3}), "positive and even"),
            (
                serde_json::json!({"num_key_value_heads": 3}),
                "divide evenly by num_key_value_heads",
            ),
            (
                serde_json::json!({
                    "num_attention_heads": usize::MAX,
                    "head_dim": 2,
                    "num_key_value_heads": 1
                }),
                "query width overflow",
            ),
            (serde_json::json!({"rms_norm_eps": 0.0}), "rms_norm_eps"),
            (serde_json::json!({"rope_theta": 1.0}), "rope_theta"),
            (serde_json::json!({"vocab_size": 0}), "must be nonzero"),
            (
                serde_json::json!({"max_position_embeddings": (u32::MAX as u64) + 1}),
                "fit CUDA position IDs",
            ),
        ] {
            let fixture = Fixture::new();
            fixture.config(overrides);
            let error = DecoderConfig::load(&fixture.0).unwrap_err().to_string();
            assert!(error.contains(needle), "{error}");
        }

        let fixture = Fixture::new();
        let config = fixture.config_from_disk();
        let mut malformed = entries(&config, "model.", Biases::default());
        for (name, shape) in &mut malformed {
            if name == "model.layers.0.self_attn.q_proj.weight" {
                *shape = vec![4, 8];
            }
        }
        fixture.write_entries(&malformed, false);
        let error = fixture.error();
        assert!(error.contains("wrong shape"), "{error}");
    }

    #[test]
    fn accepts_explicit_head_dim_distinct_from_hidden_per_head() {
        let fixture = Fixture::new();
        fixture.config(serde_json::json!({"head_dim": 2}));
        let config = fixture.config_from_disk();
        assert_eq!(config.q_width(), 4);
        assert_eq!(config.kv_width(), 2);
        write_weights(&fixture, "model.", Biases::default(), false);
        let loaded = DecoderWeights::load(&fixture.0).unwrap();
        assert_eq!(loaded.config.hidden_size, 8);
        assert_eq!(loaded.config.head_dim, 2);
        assert_eq!(loaded.layers[0].q.weight.shape(), &[4, 8]);
        assert_eq!(loaded.layers[0].k.weight.shape(), &[2, 8]);
        assert_eq!(loaded.layers[0].o.weight.shape(), &[8, 4]);
    }

    #[test]
    fn rejects_layer_count_beyond_validated_header_capacity() {
        let fixture = Fixture::new();
        write_weights(&fixture, "model.", Biases::default(), false);
        fixture.config(serde_json::json!({"num_hidden_layers": 3}));
        let error = fixture.error();
        assert!(error.contains("fewer than the"), "{error}");
    }

    #[test]
    fn rejects_mixed_ambiguous_and_foreign_checkpoint_roots() {
        let fixture = Fixture::new();
        let config = fixture.config_from_disk();
        let mut ambiguous = entries(&config, "model.", Biases::default());
        ambiguous.push(("embed_tokens.weight".into(), vec![12, 8]));
        fixture.write_entries(&ambiguous, false);
        let error = fixture.error();
        assert!(error.contains("ambiguous checkpoint namespace"), "{error}");

        let fixture = Fixture::new();
        let config = fixture.config_from_disk();
        let mut mixed = entries(&config, "model.", Biases::default());
        mixed.push(("layers.0.input_layernorm.weight".into(), vec![8]));
        fixture.write_entries(&mixed, false);
        let error = fixture.error();
        assert!(error.contains("mixed decoder checkpoint roots"), "{error}");

        let fixture = Fixture::new();
        let config = fixture.config_from_disk();
        fixture.write_entries(&entries(&config, "transformer.", Biases::default()), false);
        let error = fixture.error();
        assert!(
            error.contains("unsupported decoder checkpoint namespace"),
            "{error}"
        );
    }

    #[test]
    #[ignore = "requires INFINITY_OXIDE_TEST_DECODER_MODEL to name a genuine decoder checkpoint"]
    fn loads_gte_qwen2_metadata_from_env_fixture() {
        let path = PathBuf::from(
            std::env::var_os("INFINITY_OXIDE_TEST_DECODER_MODEL")
                .expect("INFINITY_OXIDE_TEST_DECODER_MODEL must name a checkpoint directory"),
        );
        let loaded = DecoderWeights::load(&path).unwrap();
        assert_eq!(loaded.config.family, DecoderFamily::Qwen2);
        assert_eq!(loaded.config.hidden_size, 1536);
        assert_eq!(loaded.config.intermediate_size, 8960);
        assert_eq!(loaded.config.head_dim, 128);
        assert_eq!(loaded.config.num_attention_heads, 12);
        assert_eq!(loaded.config.num_key_value_heads, 2);
        assert!(!loaded.config.is_causal);
        assert_eq!(loaded.config.rope, DecoderRope::FullNeoX);
        assert_eq!(loaded.layers.len(), 28);
        assert_eq!(loaded.embed.shape(), &[151646, 1536]);
        assert_eq!(loaded.layers[0].q.weight.shape(), &[1536, 1536]);
        assert_eq!(loaded.layers[0].k.weight.shape(), &[256, 1536]);
        assert_eq!(loaded.layers[0].v.weight.shape(), &[256, 1536]);
        assert_eq!(loaded.layers[0].q.bias.as_ref().unwrap().shape(), &[1536]);
        assert_eq!(loaded.layers[0].k.bias.as_ref().unwrap().shape(), &[256]);
        assert!(loaded.layers[0].mlp.dense().unwrap().0.bias.is_none());
        assert!(loaded.layers[0].o.bias.is_none());
        assert_eq!(
            loaded.layers[27].mlp.dense().unwrap().2.weight.shape(),
            &[1536, 8960]
        );
    }

    #[test]
    fn loads_embedded_tinyllama_metadata_without_global_paths() {
        const CONFIG: &str = r#"{
            "architectures": ["LlamaForCausalLM"],
            "attention_bias": false,
            "hidden_act": "silu",
            "hidden_size": 2048,
            "intermediate_size": 5632,
            "max_position_embeddings": 2048,
            "model_type": "llama",
            "num_attention_heads": 32,
            "num_hidden_layers": 22,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-5,
            "rope_scaling": null,
            "rope_theta": 10000.0,
            "tie_word_embeddings": false,
            "torch_dtype": "bfloat16",
            "vocab_size": 32000
        }"#;
        let fixture = Fixture::new();
        let value: serde_json::Value = serde_json::from_str(CONFIG).unwrap();
        fixture.raw_config(value);
        let config = fixture.config_from_disk();
        assert_eq!(config.family, DecoderFamily::Llama);
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.intermediate_size, 5632);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_key_value_heads, 4);
        assert_eq!(config.kv_width(), 256);
        assert!(config.is_causal);
        assert_eq!(config.rope, DecoderRope::Default);
        assert_eq!(config.rope_theta, 10_000.0);
    }

    #[test]
    fn loads_mistral_defaults_and_saved_window_flag_precedence() {
        let fixture = Fixture::new();
        fixture.raw_config(mistral_config());
        let config = fixture.config_from_disk();
        assert_eq!(config.family, DecoderFamily::Mistral);
        assert!(config.is_causal);
        assert_eq!(config.rope, DecoderRope::Default);
        assert_eq!(config.rope_theta, 10_000.0);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.q_width(), 4096);
        assert_eq!(config.kv_width(), 1024);
        assert_eq!(config.attention_window, Some(4096));

        fixture.raw_config(with_fields(
            mistral_config(),
            serde_json::json!({"use_sliding_window": false}),
        ));
        assert_eq!(fixture.config_from_disk().attention_window, Some(4096));
        fixture.raw_config(with_fields(
            mistral_config(),
            serde_json::json!({"use_sliding_window": true}),
        ));
        assert_eq!(fixture.config_from_disk().attention_window, Some(4096));

        for fields in [
            serde_json::json!({"sliding_window": 0}),
            serde_json::json!({"sliding_window": null, "use_sliding_window": true}),
        ] {
            fixture.raw_config(with_fields(mistral_config(), fields));
            assert_eq!(fixture.config_from_disk().attention_window, None);
        }
        fixture.raw_config(with_fields(
            mistral_config(),
            serde_json::json!({"sliding_window": 2147483648u64}),
        ));
        let error = DecoderConfig::load(&fixture.0).unwrap_err().to_string();
        assert!(error.contains("CUDA bounds"), "{error}");

        fixture.raw_config(with_fields(
            mistral_config(),
            serde_json::json!({"use_bidirectional_attention": true, "is_causal": true}),
        ));
        assert!(fixture.config_from_disk().is_causal);
    }

    #[test]
    fn rejects_mistral_model_dtype_average_pooling() {
        let fixture = Fixture::new();
        fixture.raw_config(with_fields(
            mistral_config(),
            serde_json::json!({"pooling": "avg"}),
        ));
        let error = DecoderConfig::load(&fixture.0).unwrap_err().to_string();
        assert!(
            error.contains("unported ATen model-dtype reduction"),
            "{error}"
        );
    }

    #[test]
    fn rejects_ministral3_instead_of_treating_it_as_ordinary_mistral() {
        let fixture = Fixture::new();
        fixture.raw_config(with_fields(
            mistral_config(),
            serde_json::json!({"model_type": "ministral3"}),
        ));
        let error = DecoderConfig::load(&fixture.0).unwrap_err().to_string();
        assert!(error.contains("ministral3"), "{error}");
        assert!(error.contains("not ported"), "{error}");
        assert_eq!(DecoderFamily::from_model_type("ministral3"), None);
        assert!(DecoderFamily::is_decoder_model_type("ministral3"));
        for alias in ["ministral3", "llama_bidirec", "mistral3"] {
            assert!(
                DecoderFamily::is_unsupported_native_decoder_model_type(alias),
                "{alias} must fail explicitly on the host"
            );
        }
        assert!(!DecoderFamily::is_unsupported_native_decoder_model_type(
            "qwen3_moe"
        ));
        assert!(!DecoderFamily::is_unsupported_native_decoder_model_type(
            "bidirectional_pplx_qwen3"
        ));
        assert!(!DecoderFamily::is_unsupported_native_decoder_model_type(
            "qwen3.5"
        ));
    }

    #[test]
    fn loads_mistral_weights_from_default_model_namespace() {
        let fixture = Fixture::new();
        fixture.raw_config(mistral_small_config());
        write_weights(&fixture, "model.", Biases::default(), false);

        let loaded = DecoderWeights::load(&fixture.0).unwrap();
        assert_eq!(loaded.config.family, DecoderFamily::Mistral);
        assert_eq!(loaded.config.attention_window, Some(4096));
        assert_eq!(loaded.layers.len(), 1);
        assert_eq!(loaded.layers[0].q.weight.shape(), &[8, 8]);
        assert_eq!(loaded.layers[0].k.weight.shape(), &[4, 8]);
        assert_eq!(loaded.layers[0].v.weight.shape(), &[4, 8]);
        assert_eq!(loaded.layers[0].o.weight.shape(), &[8, 8]);
        let (gate, up, down) = loaded.layers[0].mlp.dense().unwrap();
        assert_eq!(gate.weight.shape(), &[16, 8]);
        assert_eq!(up.weight.shape(), &[16, 8]);
        assert_eq!(down.weight.shape(), &[8, 16]);
        assert!(loaded.layers[0].q.bias.is_none());
    }
    #[test]
    fn classifier_admission_uses_one_config_view_and_alias_routing() {
        let mut value = qwen2_config();
        value["architectures"] = serde_json::json!(["Qwen2ForSequenceClassification"]);
        value["id2label"] = serde_json::json!({"0": "no", "1": "yes"});
        value["quantization_config"] = serde_json::Value::Null;
        let config = DecoderConfig::from_value(value.clone(), true).unwrap();
        assert_eq!(config.family, DecoderFamily::Qwen2);
        assert!(
            DecoderConfig::from_value(value.clone(), false).is_err(),
            "ordinary embedding admission must keep rejecting classifier architectures"
        );
        let head = DecoderClassifierConfig::from_value(value).unwrap();
        assert_eq!(head.family, DecoderFamily::Qwen2);
        assert_eq!(head.num_labels, 2);

        let mut quantized = qwen2_config();
        quantized["architectures"] = serde_json::json!(["Qwen2ForSequenceClassification"]);
        quantized["quantization_config"] = serde_json::json!({"quant_method": "gptq"});
        assert!(DecoderConfig::from_value(quantized.clone(), true).is_err());
        assert!(DecoderClassifierConfig::from_value(quantized).is_err());

        let mut alias = qwen3_config();
        alias["model_type"] = serde_json::json!("bidirectional_pplx_qwen3");
        alias["architectures"] = serde_json::json!(["Qwen3ForSequenceClassification"]);
        assert!(DecoderFamily::is_decoder_model_type(
            "bidirectional_pplx_qwen3"
        ));
        let config = DecoderConfig::from_value(alias, true).unwrap();
        assert_eq!(config.family, DecoderFamily::Qwen3);
        assert!(!config.is_causal);
    }

    #[test]
    fn classifier_bundle_loads_backbone_and_root_score_from_one_checkpoint() {
        let fixture = Fixture::new();
        let mut value = qwen2_config();
        value["architectures"] = serde_json::json!(["Qwen2ForSequenceClassification"]);
        value["id2label"] = serde_json::json!({"0": "no", "1": "yes"});
        fixture.raw_config(value.clone());
        let config = DecoderConfig::from_value(value.clone(), true).unwrap();
        let mut weights = entries(&config, "model.", Biases::default());
        weights.push(("score.weight".into(), vec![2, 8]));
        fixture.write_entries(&weights, false);

        assert!(DecoderWeights::load(&fixture.0).is_err());
        let (loaded, head) = DecoderWeights::load_classifier(&fixture.0, None, value).unwrap();
        assert_eq!(loaded.config.family, DecoderFamily::Qwen2);
        assert_eq!(loaded.layers.len(), 2);
        assert_eq!(
            head.config.task,
            crate::decoder_classifier::DecoderClassifierTask::Sequence
        );
        assert_eq!(head.config.num_labels, 2);
        assert_eq!(head.config.hidden_size, 8);
    }
    #[test]
    fn qwen3_h5120_positive_intermediate_and_independent_query_widths() {
        // These are source-admission examples, not checkpoint shape claims.
        for (heads, intermediate) in [(40usize, 17408usize), (64, 25600), (40, 1), (64, 17407)] {
            let fixture = Fixture::new();
            fixture.raw_config(with_fields(
                qwen3_config(),
                serde_json::json!({
                    "hidden_size": 5120,
                    "intermediate_size": intermediate,
                    "num_hidden_layers": 3,
                    "num_attention_heads": heads,
                    "num_key_value_heads": 8,
                    "head_dim": 128
                }),
            ));
            let config = fixture.config_from_disk();
            assert_eq!(config.family, DecoderFamily::Qwen3);
            assert_eq!(
                (config.hidden_size, config.intermediate_size),
                (5120, intermediate)
            );
            assert_eq!(
                (config.num_attention_heads, config.num_key_value_heads),
                (heads, 8)
            );
            assert_eq!(config.head_dim, 128);
            assert_eq!(config.q_width(), heads * 128);
            assert_eq!(config.kv_width(), 1024);
            if heads == 64 {
                assert_ne!(config.q_width(), config.hidden_size);
            }
        }

        let zero = Fixture::new();
        zero.raw_config(with_fields(
            qwen3_config(),
            serde_json::json!({
                "hidden_size": 5120,
                "intermediate_size": 0,
                "num_attention_heads": 40,
                "num_key_value_heads": 8,
                "head_dim": 128
            }),
        ));
        assert!(zero.error().contains("intermediate_size must be nonzero"));

        let overflow = Fixture::new();
        overflow.raw_config(with_fields(
            qwen3_config(),
            serde_json::json!({
                "hidden_size": 5120,
                "intermediate_size": usize::MAX,
                "num_attention_heads": 40,
                "num_key_value_heads": 8,
                "head_dim": 128
            }),
        ));
        assert!(overflow.error().contains("weight dimensions overflow"));

        let beyond_cuda_index = Fixture::new();
        beyond_cuda_index.raw_config(with_fields(
            qwen3_config(),
            serde_json::json!({
                "hidden_size": 5120,
                "intermediate_size": (1usize << 20),
                "num_attention_heads": 40,
                "num_key_value_heads": 8,
                "head_dim": 128
            }),
        ));
        assert!(beyond_cuda_index.error().contains("CUDA indexing range"));
    }
    #[test]
    fn h5120_classifier_config_uses_the_task_aware_constructor_and_hidden_rows() {
        use crate::decoder_classifier::{
            DecoderClassifierConfig, DecoderClassifierScratchLayout, RowPlan,
        };
        for (heads, intermediate) in [(40usize, 17408usize), (64, 25600)] {
            for architecture in [
                "Qwen3ForSequenceClassification",
                "Qwen3ForTokenClassification",
            ] {
                let value = serde_json::json!({
                    "model_type":"qwen3","architectures":[architecture],
                    "hidden_size":5120,"intermediate_size":intermediate,
                    "num_hidden_layers":1,"num_attention_heads":heads,
                    "num_key_value_heads":8,"head_dim":128,
                    "vocab_size":151936,"max_position_embeddings":40960,
                    "rms_norm_eps":1e-6,"rope_theta":1000000.0,
                    "hidden_act":"silu","num_labels":2,"pad_token_id":0
                });
                // Exercise the same classifier-aware config constructor used by
                // load_classifier; embedding construction must reject trained heads.
                assert!(DecoderConfig::from_value(value.clone(), false).is_err());
                let config = DecoderConfig::from_value(value.clone(), true).unwrap();
                assert_eq!(
                    crate::decoder::DecoderMath::select(&config).unwrap(),
                    crate::decoder::DecoderMath::Qwen3H5120Ordinary
                );
                let head = DecoderClassifierConfig::from_value(value).unwrap();
                assert_eq!((head.hidden_size, head.num_labels), (5120, 2));
                assert_eq!(config.q_width(), heads * 128);
                let layout =
                    DecoderClassifierScratchLayout::for_key(head.hidden_size, 2, (5, 2, 2))
                        .unwrap();
                assert_eq!(layout.gather_elements(), 2 * config.hidden_size);
                assert_eq!(layout.logits_elements(), 5 * 2);
                let rows =
                    RowPlan::sequence(&[11, 0, 22, 23, 0], &[0, 2, 5], head.pad_token_id).unwrap();
                assert_eq!(rows.row_ids(), &[0, 3]);
                assert_eq!((rows.rows(), rows.source_rows()), (2, 5));
                assert_eq!(RowPlan::tokens(5).unwrap().rows(), 5);
            }
        }
    }
}
