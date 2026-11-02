// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Mistral3 / Pixtral host preparation and media contracts.
//!
//! This module owns the composite checkpoint contract: the Pixtral tower, the
//! spatial merge projector, and the nested Ministral3 text config. It validates
//! the original cached config and every expected tensor header before any
//! device work. The text decoder itself stays the shared
//! [`crate::decoder::DecoderModel`] implementation; this module never forks a
//! second text executor.
//!
//! Preparation is deliberately separate from execution. [`Mistral3Plan`] is a
//! pure CPU object (config, tensor headers, media descriptor and cold upload
//! plan). [`Mistral3Prepared`] additionally owns uploaded BF16 weights and can
//! enqueue the Pixtral tower; construction of that type requires a real CUDA
//! context. No CPU fallback, graph capture, query, or hot allocation exists in
//! either path.

#![allow(dead_code)] // Prepared host API is bound by the Root GPU route.

use crate::{
    decoder_model::{DecoderConfig, DecoderFamily, DecoderRope, DecoderWeights},
    fa4::{AttentionSpec, DType, Scale, Window},
    model::{Checkpoint, WeightSource},
    start,
};
use infinity_v3_backend_core::{
    media::{ColdUploadPlan, ImageLayoutPolicy, MediaDevice, MediaDtype, MediaFamily, PackedMedia},
    BackendError,
};
use serde::Deserialize;
use std::{collections::BTreeMap, fmt, path::Path, sync::Arc};

/// Index of the placeholder token replaced by Pixtral merge rows.
pub(crate) const DEFAULT_IMAGE_TOKEN_INDEX: usize = 10;
/// Pixtral always uses three input channels.
const VISION_CHANNELS: usize = 3;
/// Bounded media plan so malformed grids fail before arithmetic or allocation.
const MAX_MERGE_SIZE: usize = 8;
const MAX_PATCH_SIZE: usize = 64;

#[derive(Debug, Clone, Deserialize)]
struct RawVisionConfig {
    model_type: Option<String>,
    hidden_size: Option<usize>,
    intermediate_size: Option<usize>,
    num_hidden_layers: Option<usize>,
    num_attention_heads: Option<usize>,
    head_dim: Option<usize>,
    patch_size: Option<usize>,
    image_size: Option<usize>,
    num_channels: Option<usize>,
    hidden_act: Option<String>,
    rope_parameters: Option<serde_json::Value>,
    rope_theta: Option<f64>,
    #[serde(flatten)]
    extra: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
struct RawTextConfig {
    model_type: Option<String>,
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
    rope_parameters: Option<serde_json::Value>,
    llama_4_scaling: Option<serde_json::Value>,
    hidden_act: Option<String>,
    is_causal: Option<bool>,
    use_bidirectional_attention: Option<bool>,
    sliding_window: Option<usize>,
    #[serde(flatten)]
    extra: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
struct RawMistral3Config {
    model_type: String,
    architectures: Option<Vec<String>>,
    text_config: RawTextConfig,
    vision_config: RawVisionConfig,
    image_token_index: Option<usize>,
    spatial_merge_size: Option<usize>,
    vision_feature_layer: Option<serde_json::Value>,
    multimodal_projector_bias: Option<bool>,
    projector_hidden_act: Option<String>,
    rope_theta: Option<f64>,
    #[serde(flatten)]
    extra: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProjectorActivation {
    Gelu,
    GeluTanh,
    Silu,
}

impl ProjectorActivation {
    fn parse(name: &str) -> Result<Self, BackendError> {
        match name {
            "gelu" => Ok(Self::Gelu),
            "gelu_pytorch_tanh" | "gelu_new" => Ok(Self::GeluTanh),
            "silu" | "swish" => Ok(Self::Silu),
            other => Err(start(format!(
                "unsupported Mistral3 projector activation {other}"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct VisionConfig {
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) patch_size: usize,
    pub(crate) image_size: usize,
    pub(crate) num_channels: usize,
    pub(crate) hidden_activation: ProjectorActivation,
    pub(crate) rope_theta: f64,
}

impl VisionConfig {
    pub(crate) fn patch_dim(&self) -> usize {
        VISION_CHANNELS * self.patch_size * self.patch_size
    }

    /// Pixtral isolates each image in attention but uses a global mask inside
    /// that image. The existing native BERT profile is the same 16/16x64
    /// noncausal global geometry; it is reused rather than adding a duplicate
    /// kernel.
    pub(crate) fn attention_spec(&self) -> AttentionSpec {
        AttentionSpec {
            heads: self.num_attention_heads,
            kv_heads: self.num_attention_heads,
            head_dim: self.head_dim,
            causal: false,
            window: Window::Global,
            dtype: DType::Bf16,
            scale: Scale::RsqrtHeadDim,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TextConfig {
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
    pub(crate) rope_type: String,
    pub(crate) yarn: Option<crate::decoder_model::YarnRope>,
    pub(crate) query_scale: Option<crate::decoder_model::Llama4QueryScale>,
    pub(crate) bidirectional_requested: bool,
    pub(crate) hidden_activation_silu: bool,
}

impl TextConfig {
    pub(crate) fn decoder_config(&self) -> DecoderConfig {
        DecoderConfig {
            family: DecoderFamily::Mistral,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
            vocab_size: self.vocab_size,
            max_position_embeddings: self.max_position_embeddings,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            is_causal: self.is_causal,
            attention_window: self.attention_window,
            rope: self.rope,
            yarn: self.yarn.clone(),
            query_scale: self.query_scale.clone(),
            transformers_math: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Mistral3Config {
    pub(crate) architectures: Vec<String>,
    pub(crate) image_token_index: usize,
    pub(crate) spatial_merge_size: usize,
    pub(crate) vision_feature_layer: usize,
    pub(crate) projector_bias: bool,
    pub(crate) projector_activation: ProjectorActivation,
    pub(crate) vision: VisionConfig,
    pub(crate) text: TextConfig,
    pub(crate) classifier: bool,
    pub(crate) classifier_labels: usize,
}

impl Mistral3Config {
    pub(crate) fn load(path: &Path) -> Result<Self, BackendError> {
        let config = path.join("config.json");
        if config.is_file() {
            let bytes = std::fs::read(&config)
                .map_err(|error| start(format!("Mistral3 config read failed: {error}")))?;
            return Self::from_json(&bytes);
        }
        // Some official Mistral releases publish the short-key `params.json`
        // instead of a Transformers `config.json`. Normalize that schema in
        // memory; the checkpoint itself is never rewritten.
        let params = path.join("params.json");
        let bytes = std::fs::read(&params).map_err(|error| {
            start(format!(
                "Mistral3 needs config.json or params.json: {error}"
            ))
        })?;
        let value: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|error| start(format!("Mistral3 params.json is invalid: {error}")))?;
        let normalized = normalize_params(&value)?;
        Self::from_json(&serde_json::to_vec(&normalized).map_err(start)?)
    }

    pub(crate) fn from_json(bytes: &[u8]) -> Result<Self, BackendError> {
        let raw: RawMistral3Config = serde_json::from_slice(bytes)
            .map_err(|error| start(format!("Mistral3 config is invalid: {error}")))?;
        Self::from_raw(raw)
    }

    fn from_raw(raw: RawMistral3Config) -> Result<Self, BackendError> {
        if raw.model_type != "mistral3" {
            return Err(start(format!(
                "expected model_type mistral3, got {}",
                raw.model_type
            )));
        }
        let architectures = raw.architectures.unwrap_or_default();
        for architecture in &architectures {
            if !matches!(
                architecture.as_str(),
                "Mistral3Model"
                    | "Mistral3ForCausalLM"
                    | "Mistral3ForConditionalGeneration"
                    | "Mistral3ForSequenceClassification"
            ) {
                return Err(start(format!(
                    "unsupported Mistral3 architecture {architecture}"
                )));
            }
        }
        let classifier = architectures
            .iter()
            .any(|architecture| architecture == "Mistral3ForSequenceClassification");
        let v = &raw.vision_config;
        if v.model_type.as_deref() != Some("pixtral") {
            return Err(start("Mistral3 requires vision_config.model_type pixtral"));
        }
        let hidden_size = required(v.hidden_size, "vision_config.hidden_size")?;
        let intermediate_size = required(v.intermediate_size, "vision_config.intermediate_size")?;
        let num_hidden_layers = required(v.num_hidden_layers, "vision_config.num_hidden_layers")?;
        let num_attention_heads =
            required(v.num_attention_heads, "vision_config.num_attention_heads")?;
        if hidden_size == 0
            || intermediate_size == 0
            || num_hidden_layers == 0
            || num_attention_heads == 0
            || !hidden_size.is_multiple_of(num_attention_heads)
        {
            return Err(start("invalid Pixtral attention dimensions"));
        }
        let derived_head_dim = hidden_size / num_attention_heads;
        let head_dim = v.head_dim.unwrap_or(derived_head_dim);
        if head_dim == 0 || !head_dim.is_multiple_of(4) {
            return Err(start(
                "Pixtral head_dim must be positive and a multiple of four",
            ));
        }
        if head_dim != derived_head_dim {
            return Err(start(format!(
                "Pixtral head_dim {head_dim} must equal hidden_size / num_attention_heads ({derived_head_dim})"
            )));
        }
        if intermediate_size < hidden_size {
            return Err(start(
                "Pixtral intermediate_size must be at least hidden_size for the prepared arena",
            ));
        }
        let patch_size = required(v.patch_size, "vision_config.patch_size")?;
        if patch_size == 0 || patch_size > MAX_PATCH_SIZE {
            return Err(start("Pixtral patch_size is outside 1..=64"));
        }
        let image_size = v
            .image_size
            .ok_or_else(|| start("missing vision_config.image_size"))?;
        if image_size == 0 || image_size > 4096 {
            return Err(start("Pixtral image_size is outside 1..=4096"));
        }
        let num_channels = v.num_channels.unwrap_or(VISION_CHANNELS);
        if num_channels != VISION_CHANNELS {
            return Err(start("Pixtral requires three input channels"));
        }
        let vision_act = v.hidden_act.as_deref().unwrap_or("silu");
        // SiLU and exact GELU are Native-valid. Tanh GELU is retained as an
        // explicitly typed execution extension; it is not a Native parity claim.
        if !matches!(vision_act, "silu" | "gelu" | "gelu_pytorch_tanh") {
            return Err(start(format!(
                "unsupported Pixtral hidden activation {vision_act}"
            )));
        }
        let vision_activation = ProjectorActivation::parse(vision_act)?;
        if let Some(kind) = v
            .rope_parameters
            .as_ref()
            .and_then(|value| value.get("rope_type"))
            .and_then(serde_json::Value::as_str)
        {
            if kind != "default" {
                return Err(start(format!(
                    "unsupported Pixtral rotary configuration {kind}; default only"
                )));
            }
        }
        let vision_theta = vision_rope_theta(v, raw.rope_theta)?;
        let vision = VisionConfig {
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            head_dim,
            patch_size,
            image_size,
            num_channels,
            hidden_activation: vision_activation,
            rope_theta: vision_theta,
        };

        let t = &raw.text_config;
        if let Some(model_type) = t.model_type.as_deref() {
            if !matches!(model_type, "ministral3" | "mistral") {
                return Err(start(format!(
                    "unsupported Mistral3 text_config.model_type {model_type}"
                )));
            }
        }
        let th = required(t.hidden_size, "text_config.hidden_size")?;
        let ti = required(t.intermediate_size, "text_config.intermediate_size")?;
        let tl = required(t.num_hidden_layers, "text_config.num_hidden_layers")?;
        let thn = required(t.num_attention_heads, "text_config.num_attention_heads")?;
        let tkn = t.num_key_value_heads.unwrap_or(thn);
        let tvd = required(t.vocab_size, "text_config.vocab_size")?;
        let tmp = required(
            t.max_position_embeddings,
            "text_config.max_position_embeddings",
        )?;
        for (name, value) in [
            ("hidden_size", th),
            ("intermediate_size", ti),
            ("num_hidden_layers", tl),
            ("num_attention_heads", thn),
            ("num_key_value_heads", tkn),
            ("vocab_size", tvd),
            ("max_position_embeddings", tmp),
        ] {
            if value == 0 {
                return Err(start(format!(
                    "Mistral3 text_config.{name} must be nonzero"
                )));
            }
        }
        if !thn.is_multiple_of(tkn) {
            return Err(start(
                "Mistral3 text num_attention_heads must divide evenly by num_key_value_heads",
            ));
        }
        let derived_text_head_dim = th / thn;
        let thead = t.head_dim.unwrap_or(derived_text_head_dim);
        if thead == 0 || !thead.is_multiple_of(2) {
            return Err(start("Mistral3 text head_dim must be positive and even"));
        }
        // Native permits an explicit head_dim whose query width differs from
        // hidden_size (for example H3072/H5120 with 32 heads and d128). Do not
        // force the derived hidden/heads value.
        i32::try_from(tmp).map_err(|_| start("Mistral3 max_position_embeddings exceeds u32"))?;
        let (text_theta, rope_type, yarn) = text_rope_parameters(t, raw.rope_theta)?;
        let query_scale = yarn
            .as_ref()
            .map(|yarn| crate::decoder_model::Llama4QueryScale {
                beta: yarn.llama_4_scaling_beta,
                original_max_position_embeddings: yarn.original_max_position_embeddings,
            });
        let rope = DecoderRope::Default;
        let text_act = t.hidden_act.as_deref().unwrap_or("silu");
        if !matches!(text_act, "silu" | "gelu" | "gelu_pytorch_tanh") {
            return Err(start(format!(
                "unsupported Mistral3 text hidden activation {text_act}"
            )));
        }
        let bidirectional = t.use_bidirectional_attention.unwrap_or(false);
        let text = TextConfig {
            hidden_size: th,
            intermediate_size: ti,
            num_hidden_layers: tl,
            num_attention_heads: thn,
            num_key_value_heads: tkn,
            head_dim: thead,
            vocab_size: tvd,
            max_position_embeddings: tmp,
            rms_norm_eps: t.rms_norm_eps.unwrap_or(1e-5),
            rope_theta: text_theta,
            is_causal: t.is_causal.unwrap_or(!bidirectional),
            attention_window: t.sliding_window.filter(|window| *window > 0),
            rope,
            rope_type,
            yarn,
            query_scale,
            bidirectional_requested: bidirectional,
            hidden_activation_silu: text_act == "silu",
        };

        let spatial_merge_size = raw.spatial_merge_size.unwrap_or(2);
        if spatial_merge_size == 0 || spatial_merge_size > MAX_MERGE_SIZE {
            return Err(start("Mistral3 spatial_merge_size is outside 1..=8"));
        }
        let feature_layer = vision_feature_layer(raw.vision_feature_layer.as_ref())?;
        let resolved_feature = if feature_layer < 0 {
            let resolved = i64::try_from(vision.num_hidden_layers)
                .ok()
                .and_then(|depth| depth.checked_add(1))
                .and_then(|depth| depth.checked_add(feature_layer))
                .ok_or_else(|| start("Mistral3 feature layer underflow"))?;
            usize::try_from(resolved).map_err(|_| start("Mistral3 feature layer underflow"))?
        } else {
            usize::try_from(feature_layer).map_err(|_| start("Mistral3 feature layer overflow"))?
        };
        if resolved_feature > vision.num_hidden_layers {
            return Err(start("Mistral3 vision_feature_layer exceeds depth"));
        }
        let projector_bias = raw.multimodal_projector_bias.unwrap_or(true);
        let projector_activation =
            ProjectorActivation::parse(raw.projector_hidden_act.as_deref().unwrap_or("gelu"))?;
        Ok(Self {
            architectures,
            image_token_index: raw.image_token_index.unwrap_or(DEFAULT_IMAGE_TOKEN_INDEX),
            spatial_merge_size,
            vision_feature_layer: resolved_feature,
            projector_bias,
            projector_activation,
            vision,
            text,
            classifier,
            classifier_labels: 0,
        })
    }

    pub(crate) fn text_width(&self) -> usize {
        self.text.hidden_size
    }

    /// Typed reason the shared text executor is not admitted for this config.
    /// `Ok(())` means the exact shared Mistral recipe covers it.
    pub(crate) fn text_readiness(&self) -> Result<(), Mistral3Pending> {
        if !matches!(self.text.rope_type.as_str(), "default" | "yarn") {
            return Err(Mistral3Pending::new(format!(
                "text rope_type {} is not the native default/YaRN recipe",
                self.text.rope_type
            )));
        }
        if !self.text.hidden_activation_silu {
            return Err(Mistral3Pending::new(
                "text hidden_act is not SiLU; only the shared dense SiLU recipe is bound",
            ));
        }
        crate::decoder::DecoderMath::select(&self.text.decoder_config())
            .map(|_| ())
            .map_err(|error| Mistral3Pending::new(error.to_string()))
    }

    /// Typed reason the projector linear path is not admitted.
    pub(crate) fn projector_readiness(&self) -> Result<(), Mistral3Pending> {
        if self.projector_bias {
            return Err(Mistral3Pending::new(
                "projector bias is not implemented by the prepared projector linear path",
            ));
        }
        Ok(())
    }

    /// Typed reason the Pixtral tower cannot bind a compiled FA4 export yet.
    pub(crate) fn vision_readiness(&self) -> Result<Mistral3VisionProfile, Mistral3Pending> {
        let spec = self.vision.attention_spec();
        if !crate::fa4::Fa4::supports(&spec) {
            return Err(Mistral3Pending::new(format!(
                "Pixtral attention H{}/KV{} d{} noncausal-global has no qualified ABI7 export",
                spec.heads, spec.kv_heads, spec.head_dim
            )));
        }
        Ok(Mistral3VisionProfile {
            heads: spec.heads,
            kv_heads: spec.kv_heads,
            head_dim: spec.head_dim,
            causal: spec.causal,
            window: "global",
            pending_provider_compile: true,
        })
    }

    /// Whether a tower snapshot index is the configured projector input. The
    /// executor uses this exact predicate for the post-ln_pre snapshot and each
    /// transformer layer output.
    pub(crate) const fn selects_feature_snapshot(&self, snapshot: usize) -> bool {
        snapshot == self.vision_feature_layer
    }

    pub(crate) fn media_policy(&self) -> Result<ImageLayoutPolicy, BackendError> {
        MediaFamily::Mistral3
            .image_policy()
            .with_patch_size(self.vision.patch_size)
            .map_err(|error| start(format!("invalid Mistral3 image policy: {error}")))
    }
}

fn first(value: &serde_json::Value, keys: &[&str]) -> Option<serde_json::Value> {
    keys.iter().find_map(|key| value.get(*key).cloned())
}

/// Normalize the official Mistral short-key `params.json` into the same nested
/// `RawMistral3Config` view used for `config.json`. Unknown fields are dropped
/// and the checkpoint stays untouched.
fn normalize_params(value: &serde_json::Value) -> Result<serde_json::Value, BackendError> {
    if value.get("text_config").is_some() {
        return Ok(value.clone());
    }
    let dim = first(value, &["dim", "hidden_size"])
        .ok_or_else(|| start("params.json lacks dim/hidden_size"))?;
    let layers = first(value, &["n_layers", "num_hidden_layers"])
        .ok_or_else(|| start("params.json lacks n_layers/num_hidden_layers"))?;
    let heads = first(value, &["n_heads", "num_attention_heads"])
        .ok_or_else(|| start("params.json lacks n_heads/num_attention_heads"))?;
    let head_dim = first(value, &["head_dim"]).unwrap_or_else(|| {
        let dim = dim.as_u64().unwrap_or(0);
        let heads = heads.as_u64().unwrap_or(1).max(1);
        serde_json::json!(dim / heads)
    });
    let text = serde_json::json!({
        "model_type": "ministral3",
        "architectures": value.get("architectures").cloned().unwrap_or(serde_json::json!(["Ministral3ForCausalLM"])),
        "hidden_size": dim,
        "intermediate_size": first(value, &["hidden_dim", "ffn_dim", "intermediate_size"])
            .ok_or_else(|| start("params.json lacks hidden_dim/ffn_dim/intermediate_size"))?,
        "num_hidden_layers": layers,
        "num_attention_heads": heads,
        "num_key_value_heads": first(value, &["n_kv_heads", "num_key_value_heads"])
            .unwrap_or_else(|| heads.clone()),
        "head_dim": head_dim,
        "vocab_size": first(value, &["vocab_size"]).ok_or_else(|| start("params.json lacks vocab_size"))?,
        "max_position_embeddings": first(value, &["max_position_embeddings", "context_length", "max_seq_len"])
            .unwrap_or_else(|| serde_json::json!(32768)),
        "rms_norm_eps": first(value, &["norm_eps", "rms_norm_eps"]).unwrap_or_else(|| serde_json::json!(1e-5)),
        "rope_theta": first(value, &["rope_theta"]).unwrap_or_else(|| serde_json::json!(1000000.0)),
        "rope_parameters": first(value, &["rope_parameters", "yarn"]),
        "llama_4_scaling": first(value, &["llama_4_scaling"]),
        "hidden_act": first(value, &["hidden_act"]).unwrap_or_else(|| serde_json::json!("silu")),
        "is_causal": first(value, &["is_causal"]).unwrap_or(serde_json::json!(true)),
        "use_bidirectional_attention": first(value, &["use_bidirectional_attention"]),
        "sliding_window": first(value, &["sliding_window"]),
    });
    let vision = first(value, &["vision_config", "vision_encoder"]).ok_or_else(|| {
        start("params.json lacks vision_config/vision_encoder; params-only text Mistral is not a Mistral3 multimodal checkpoint")
    })?;
    let mut vision = vision;
    if vision.get("model_type").is_none() {
        vision["model_type"] = serde_json::json!("pixtral");
    }
    if vision.get("image_size").is_none() {
        vision["image_size"] = serde_json::json!(1540);
    }
    if vision.get("num_channels").is_none() {
        vision["num_channels"] = serde_json::json!(3);
    }
    if vision.get("hidden_act").is_none() {
        vision["hidden_act"] = serde_json::json!("silu");
    }
    if vision.get("rope_parameters").is_none() {
        vision["rope_parameters"] =
            serde_json::json!({"rope_theta": 10000.0, "rope_type": "default"});
    }
    Ok(serde_json::json!({
        "model_type": "mistral3",
        "architectures": value.get("architectures").cloned().unwrap_or(serde_json::json!(["Mistral3ForConditionalGeneration"])),
        "image_token_index": first(value, &["image_token_index"]).unwrap_or_else(|| serde_json::json!(10)),
        "spatial_merge_size": first(value, &["spatial_merge_size"]).unwrap_or_else(|| serde_json::json!(2)),
        "vision_feature_layer": first(value, &["vision_feature_layer"]).unwrap_or_else(|| serde_json::json!(-1)),
        "multimodal_projector_bias": first(value, &["multimodal_projector_bias"]).unwrap_or(serde_json::json!(true)),
        "projector_hidden_act": first(value, &["projector_hidden_act"]).unwrap_or_else(|| serde_json::json!("gelu")),
        "rope_theta": first(value, &["rope_theta"]),
        "text_config": text,
        "vision_config": vision,
    }))
}

fn required<T>(value: Option<T>, name: &str) -> Result<T, BackendError> {
    value.ok_or_else(|| start(format!("missing Mistral3 {name}")))
}

fn vision_rope_theta(
    vision: &RawVisionConfig,
    top_level: Option<f64>,
) -> Result<f64, BackendError> {
    let nested = vision
        .rope_parameters
        .as_ref()
        .and_then(|value| value.get("rope_theta"))
        .and_then(serde_json::Value::as_f64);
    let theta = nested
        .or(vision.rope_theta)
        .or(top_level)
        .unwrap_or(10_000.0);
    if !theta.is_finite() || theta <= 1.0 || !(theta as f32).is_finite() {
        return Err(start("Mistral3 vision rope theta must be finite and > 1"));
    }
    Ok(theta)
}

fn text_rope_parameters(
    text: &RawTextConfig,
    top_level: Option<f64>,
) -> Result<(f64, String, Option<crate::decoder_model::YarnRope>), BackendError> {
    let nested = text.rope_parameters.as_ref();
    let theta = nested
        .and_then(|value| value.get("rope_theta"))
        .and_then(serde_json::Value::as_f64)
        .or(text.rope_theta)
        .or(top_level)
        .unwrap_or(10_000.0);
    if !theta.is_finite() || theta <= 1.0 || !(theta as f32).is_finite() {
        return Err(start("Mistral3 text rope theta must be finite and > 1"));
    }
    let kind = nested
        .and_then(|value| value.get("rope_type"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("default")
        .to_ascii_lowercase();
    if kind == "default" {
        return Ok((theta, kind, None));
    }
    if kind != "yarn" {
        return Err(start(format!(
            "unsupported Mistral3 text rotary configuration {kind}"
        )));
    }
    let value = nested.ok_or_else(|| start("YaRN requires rope_parameters"))?;
    let real = |key: &str, fallback: f64| {
        value
            .get(key)
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(fallback)
    };
    // Old checkpoints carry a sibling `llama_4_scaling` object instead of the
    // merged `rope_parameters.llama_4_scaling_beta`; preserve both conventions.
    let llama4 = |key: &str, fallback: f64| {
        text.llama_4_scaling
            .as_ref()
            .filter(|_| value.get("llama_4_scaling_beta").is_none())
            .and_then(|value| value.get(key))
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(fallback)
    };
    let yarn = crate::decoder_model::YarnRope {
        factor: real("factor", 1.0),
        original_max_position_embeddings: real(
            "original_max_position_embeddings",
            llama4("original_max_position_embeddings", 0.0),
        ),
        beta_fast: real("beta_fast", 32.0),
        beta_slow: real("beta_slow", 1.0),
        mscale: real("mscale", 0.0),
        mscale_all_dim: real("mscale_all_dim", 0.0),
        llama_4_scaling_beta: real("llama_4_scaling_beta", llama4("beta", 0.0)),
    };
    yarn.validate()?;
    Ok((theta, kind, Some(yarn)))
}

fn vision_feature_layer(value: Option<&serde_json::Value>) -> Result<i64, BackendError> {
    let Some(value) = value else {
        return Ok(-1);
    };
    if value.is_array() {
        return Err(start(
            "Mistral3 multiple vision feature layers are not supported",
        ));
    }
    value
        .as_i64()
        .ok_or_else(|| start("Mistral3 vision_feature_layer must be an integer"))
}

/// Cold, typed readiness result. Pending work is explicit and never reported as
/// support.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Mistral3Pending {
    reason: String,
}

impl Mistral3Pending {
    fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }

    pub(crate) fn reason(&self) -> &str {
        &self.reason
    }
}

impl fmt::Display for Mistral3Pending {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.reason)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Mistral3VisionProfile {
    pub(crate) heads: usize,
    pub(crate) kv_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) causal: bool,
    pub(crate) window: &'static str,
    /// A profile may exist in `PROFILES` while the Root AOT bundle has not been
    /// compiled for the target SM. That is a provider task, not model support.
    pub(crate) pending_provider_compile: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct TensorPlan {
    pub(crate) name: String,
    pub(crate) source: WeightSource,
    pub(crate) rows: usize,
    pub(crate) columns: usize,
}

impl TensorPlan {
    fn new(name: String, source: WeightSource) -> Result<Self, BackendError> {
        let shape = source.shape();
        let (rows, columns) = match shape {
            [rows] => (*rows, 1),
            [rows, columns] => (*rows, *columns),
            [rows, rest @ ..] if !rest.is_empty() => (
                *rows,
                rest.iter().try_fold(1usize, |size, &dim| {
                    size.checked_mul(dim)
                        .ok_or_else(|| start(format!("Mistral3 tensor {name} shape overflow")))
                })?,
            ),
            _ => return Err(start(format!("Mistral3 tensor {name} has no matrix shape"))),
        };
        if rows == 0 || columns == 0 {
            return Err(start(format!("Mistral3 tensor {name} has a zero extent")));
        }
        Ok(Self {
            name,
            source,
            rows,
            columns,
        })
    }

    pub(crate) fn elements(&self) -> usize {
        self.rows.saturating_mul(self.columns)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mistral3WeightLayout {
    /// HF-converted checkpoint with `model.language_model.*` /
    /// `model.vision_tower.*` / `language_model.score.weight` keys.
    HfConverted,
    /// Raw Mistral-common `tok_embeddings.*` / `layers.*` / `vision_encoder.*`
    /// keys. Native and Oxide require an explicit HF conversion.
    RawMistralCommon,
    Unknown,
}

pub(crate) fn detect_weight_layout<'a>(
    names: impl Iterator<Item = &'a str>,
) -> Mistral3WeightLayout {
    let mut hf = false;
    let mut raw = false;
    for name in names {
        hf |= name.starts_with("model.language_model.")
            || name.starts_with("model.vision_tower.")
            || name.starts_with("model.multi_modal_projector.")
            || name == "language_model.score.weight";
        raw |= name.starts_with("tok_embeddings.")
            || name.starts_with("layers.")
            || name.starts_with("vision_encoder.");
    }
    match (hf, raw) {
        (true, _) => Mistral3WeightLayout::HfConverted,
        (false, true) => Mistral3WeightLayout::RawMistralCommon,
        (false, false) => Mistral3WeightLayout::Unknown,
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Mistral3Weights {
    pub(crate) vision: Vec<TensorPlan>,
    pub(crate) projector: Vec<TensorPlan>,
    pub(crate) text: DecoderWeights,
    pub(crate) score: Option<TensorPlan>,
    pub(crate) retained: Vec<String>,
}

impl Mistral3Weights {
    pub(crate) fn load(path: &Path, config: &Mistral3Config) -> Result<Self, BackendError> {
        let checkpoint = Checkpoint::open(path)?;
        match detect_weight_layout(checkpoint.names().map(String::as_str)) {
            Mistral3WeightLayout::HfConverted => {}
            Mistral3WeightLayout::RawMistralCommon => {
                return Err(start(
                    "Mistral3 params.json checkpoints with raw Mistral-common weights (tok_embeddings/layers/vision_encoder) require HF conversion; Native and Oxide support the HF-converted model.language_model/model.vision_tower layout",
                ));
            }
            Mistral3WeightLayout::Unknown => {
                return Err(start(
                    "Mistral3 checkpoint has no recognized HF-converted weight prefixes; expected model.language_model, model.vision_tower and language_model.score.weight",
                ));
            }
        }
        if let Some(name) = checkpoint
            .names()
            .find(|name| name.starts_with("model.vision_tower.") && name.ends_with(".bias"))
        {
            return Err(start(format!(
                "Mistral3 vision bias {name} is Native-valid but the current prepared linear path is bias-free; provide a bias-free HF conversion or add vision-bias execution before loading payload"
            )));
        }
        let mut vision = Vec::new();
        let mut projector = Vec::new();
        let fetch = |checkpoint: &Checkpoint,
                     out: &mut Vec<TensorPlan>,
                     name: String,
                     shape: &[usize]|
         -> Result<(), BackendError> {
            let source = checkpoint.source(&name, shape)?;
            out.push(TensorPlan::new(name, source)?);
            Ok(())
        };
        let p = "model.";
        fetch(
            &checkpoint,
            &mut vision,
            format!("{p}vision_tower.patch_conv.weight"),
            &[
                config.vision.hidden_size,
                VISION_CHANNELS,
                config.vision.patch_size,
                config.vision.patch_size,
            ],
        )?;
        fetch(
            &checkpoint,
            &mut vision,
            format!("{p}vision_tower.ln_pre.weight"),
            &[config.vision.hidden_size],
        )?;
        let h = config.vision.hidden_size;
        let i = config.vision.intermediate_size;
        for layer in 0..config.vision.num_hidden_layers {
            let base = format!("{p}vision_tower.transformer.layers.{layer}");
            fetch(
                &checkpoint,
                &mut vision,
                format!("{base}.attention_norm.weight"),
                &[h],
            )?;
            fetch(
                &checkpoint,
                &mut vision,
                format!("{base}.ffn_norm.weight"),
                &[h],
            )?;
            for projection in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                fetch(
                    &checkpoint,
                    &mut vision,
                    format!("{base}.attention.{projection}.weight"),
                    &[h, h],
                )?;
            }
            fetch(
                &checkpoint,
                &mut vision,
                format!("{base}.feed_forward.gate_proj.weight"),
                &[i, h],
            )?;
            fetch(
                &checkpoint,
                &mut vision,
                format!("{base}.feed_forward.up_proj.weight"),
                &[i, h],
            )?;
            fetch(
                &checkpoint,
                &mut vision,
                format!("{base}.feed_forward.down_proj.weight"),
                &[h, i],
            )?;
        }
        let merge = config.spatial_merge_size;
        let merged = config
            .vision
            .hidden_size
            .checked_mul(merge)
            .and_then(|value| value.checked_mul(merge))
            .ok_or_else(|| start("Mistral3 projector merge width overflow"))?;
        fetch(
            &checkpoint,
            &mut projector,
            format!("{p}multi_modal_projector.norm.weight"),
            &[h],
        )?;
        fetch(
            &checkpoint,
            &mut projector,
            format!("{p}multi_modal_projector.patch_merger.merging_layer.weight"),
            &[h, merged],
        )?;
        let out = config.text.hidden_size;
        fetch(
            &checkpoint,
            &mut projector,
            format!("{p}multi_modal_projector.linear_1.weight"),
            &[out, h],
        )?;
        fetch(
            &checkpoint,
            &mut projector,
            format!("{p}multi_modal_projector.linear_2.weight"),
            &[out, out],
        )?;
        if config.projector_bias {
            for bias in ["linear_1.bias", "linear_2.bias"] {
                fetch(
                    &checkpoint,
                    &mut projector,
                    format!("{p}multi_modal_projector.{bias}"),
                    &[out],
                )?;
            }
        } else {
            for bias in ["linear_1.bias", "linear_2.bias"] {
                let name = format!("{p}multi_modal_projector.{bias}");
                if checkpoint.contains(&name) {
                    return Err(start(format!(
                        "Mistral3 config disables projector bias but {name} exists"
                    )));
                }
            }
        }
        let text = DecoderWeights::load_with_root(
            path,
            config.text.decoder_config(),
            "model.language_model.",
        )?;
        let score = if config.classifier {
            let name = "language_model.score.weight".to_string();
            let shape = checkpoint
                .shape(&name)
                .ok_or_else(|| start(format!("missing Mistral3 score tensor {name}")))?;
            if shape.len() != 2 || shape[0] == 0 || shape[1] != out {
                return Err(start(format!(
                    "Mistral3 score tensor {name} has shape {shape:?}, expected [labels, {out}]"
                )));
            }
            let source = checkpoint.source(&name, shape)?;
            Some(TensorPlan::new(name, source)?)
        } else {
            None
        };
        let expected: std::collections::BTreeSet<String> = vision
            .iter()
            .chain(&projector)
            .map(|tensor| tensor.name.clone())
            .chain(std::iter::once(score.as_ref().map(|score| score.name.clone())).flatten())
            .collect();
        let mut retained = Vec::new();
        for name in checkpoint.names() {
            if !expected.contains(name) {
                retained.push(name.clone());
            }
        }
        retained.sort();
        Ok(Self {
            vision,
            projector,
            text,
            score,
            retained,
        })
    }

    pub(crate) fn vision_elements(&self) -> usize {
        self.vision.iter().map(TensorPlan::elements).sum()
    }

    pub(crate) fn projector_elements(&self) -> usize {
        self.projector.iter().map(TensorPlan::elements).sum()
    }
}

/// One validated image's pre-merge and marker geometry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Mistral3ImagePlan {
    pub(crate) grid_thw: [usize; 3],
    pub(crate) patch_offset: usize,
    pub(crate) patch_rows: usize,
    pub(crate) token_start: usize,
    pub(crate) token_count: usize,
    pub(crate) sequence_start: usize,
    pub(crate) sequence_length: usize,
    pub(crate) row_positions: Vec<u32>,
    pub(crate) column_positions: Vec<u32>,
    /// Row-major merge block is unpacked channel-major by the projector, so the
    /// plan retains each token's spatial source rows for the merge gather.
    pub(crate) merge_source_rows: Vec<Vec<usize>>,
    pub(crate) marker_indices: Vec<usize>,
}

pub(crate) struct Mistral3MediaPlan {
    pub(crate) media: PackedMedia,
    pub(crate) images: Vec<Mistral3ImagePlan>,
    pub(crate) total_patches: usize,
    pub(crate) total_image_tokens: usize,
    pub(crate) cumulative: Vec<u32>,
}

impl fmt::Debug for Mistral3MediaPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Mistral3MediaPlan")
            .field("family", &self.media.family())
            .field("images", &self.images)
            .field("total_patches", &self.total_patches)
            .field("total_image_tokens", &self.total_image_tokens)
            .field("cumulative", &self.cumulative)
            .finish()
    }
}

impl Mistral3MediaPlan {
    pub(crate) fn build(
        config: &Mistral3Config,
        cumulative: &[u32],
        images: &[(usize, Arc<infinity_v3_backend_core::ImagePatches>)],
    ) -> Result<Self, BackendError> {
        if cumulative.len() < 2 {
            return Err(start("Mistral3 media plan requires packed boundaries"));
        }
        let policy = config.media_policy()?;
        let media = PackedMedia::new_with_policies(
            MediaFamily::Mistral3,
            cumulative,
            images,
            &[],
            Some(policy),
            None,
            None,
        )
        .map_err(|error| start(format!("Mistral3 media layout rejected: {error}")))?;
        let merge = config.spatial_merge_size;
        let merge_area = merge
            .checked_mul(merge)
            .ok_or_else(|| start("Mistral3 merge area overflow"))?;
        let mut planned = Vec::new();
        planned
            .try_reserve_exact(media.images_len())
            .map_err(|_| start("Mistral3 image plan allocation failed"))?;
        let mut patch_offset = 0usize;
        let mut total_image_tokens = 0usize;
        for descriptor in media.image_descriptors() {
            let grid = descriptor.grid_thw;
            let width = grid[2];
            let nw = width / merge;
            let mut row_positions = Vec::new();
            let mut column_positions = Vec::new();
            row_positions
                .try_reserve_exact(descriptor.patch_rows)
                .map_err(|_| start("Mistral3 rotary row plan allocation failed"))?;
            column_positions
                .try_reserve_exact(descriptor.patch_rows)
                .map_err(|_| start("Mistral3 rotary column plan allocation failed"))?;
            for patch in 0..descriptor.patch_rows {
                let row = patch / width;
                let column = patch % width;
                row_positions.push(
                    u32::try_from(row).map_err(|_| start("Mistral3 rotary row exceeds u32"))?,
                );
                column_positions.push(
                    u32::try_from(column)
                        .map_err(|_| start("Mistral3 rotary column exceeds u32"))?,
                );
            }
            let token_count = descriptor.token_count;
            let mut merge_source_rows = Vec::new();
            merge_source_rows
                .try_reserve_exact(token_count)
                .map_err(|_| start("Mistral3 merge plan allocation failed"))?;
            for token_row in 0..token_count {
                let mut rows = vec![0usize; merge_area];
                descriptor
                    .source_rows_for_token(token_row, &mut rows)
                    .map_err(|error| start(format!("Mistral3 merge source rejected: {error}")))?;
                merge_source_rows.push(rows);
            }
            let mut marker_indices = Vec::new();
            marker_indices
                .try_reserve_exact(token_count)
                .map_err(|_| start("Mistral3 marker plan allocation failed"))?;
            for token_row in 0..token_count {
                let row = token_row / nw.max(1);
                let column = token_row % nw.max(1);
                let index = descriptor
                    .token_start
                    .checked_add(
                        row.checked_mul(nw + 1)
                            .and_then(|value| value.checked_add(column))
                            .ok_or_else(|| start("Mistral3 marker index overflow"))?,
                    )
                    .ok_or_else(|| start("Mistral3 marker index overflow"))?;
                marker_indices.push(index);
            }
            total_image_tokens = total_image_tokens
                .checked_add(token_count)
                .ok_or_else(|| start("Mistral3 image token count overflow"))?;
            planned.push(Mistral3ImagePlan {
                grid_thw: grid,
                patch_offset,
                patch_rows: descriptor.patch_rows,
                token_start: descriptor.token_start,
                token_count,
                sequence_start: descriptor.sequence_start,
                sequence_length: descriptor.sequence_length,
                row_positions,
                column_positions,
                merge_source_rows,
                marker_indices,
            });
            patch_offset = patch_offset
                .checked_add(descriptor.patch_rows)
                .ok_or_else(|| start("Mistral3 packed patch count overflow"))?;
        }
        Ok(Self {
            media,
            images: planned,
            total_patches: patch_offset,
            total_image_tokens,
            cumulative: cumulative.to_vec(),
        })
    }

    /// Validate every image marker against the packed token ids. This mirrors
    /// the native wrapper's `image_token_index` placeholder check and sequence
    /// boundary check without modifying ids.
    pub(crate) fn validate_markers(
        &self,
        ids: &[u32],
        image_token_index: usize,
    ) -> Result<(), BackendError> {
        let expected = u32::try_from(image_token_index)
            .map_err(|_| start("Mistral3 image_token_index exceeds u32"))?;
        if ids.len() != self.media.token_count() {
            return Err(start(format!(
                "Mistral3 ids length {} differs from packed token count {}",
                ids.len(),
                self.media.token_count()
            )));
        }
        for image in &self.images {
            let span_end = image
                .sequence_start
                .checked_add(image.sequence_length)
                .ok_or_else(|| start("Mistral3 sequence span overflow"))?;
            for &index in &image.marker_indices {
                let Some(value) = ids.get(index) else {
                    return Err(start(format!(
                        "Mistral3 marker index {index} exceeds {} ids",
                        ids.len()
                    )));
                };
                if *value != expected {
                    return Err(start(format!(
                        "Mistral3 image marker at {index} is {value}, expected {expected}"
                    )));
                }
                if index < image.token_start || index >= span_end {
                    return Err(start("Mistral3 image marker crosses a sequence boundary"));
                }
            }
        }
        for window in self.cumulative.windows(2) {
            if window[0] > window[1] {
                return Err(start("Mistral3 cumulative boundaries are not monotonic"));
            }
        }
        Ok(())
    }
}

/// Pure CPU preparation result. No device allocation, query, context, or enqueue
/// happens while this object is built.
#[derive(Debug)]
pub(crate) struct Mistral3Plan {
    pub(crate) config: Mistral3Config,
    pub(crate) weights: Mistral3Weights,
    pub(crate) inventory: Vec<String>,
    pub(crate) cold: ColdUploadPlan,
    pub(crate) vision_profile: Result<Mistral3VisionProfile, Mistral3Pending>,
    pub(crate) text_profile: Result<(), Mistral3Pending>,
    pub(crate) projector_pending: Option<Mistral3Pending>,
    pub(crate) vision_bytes: usize,
    pub(crate) projector_bytes: usize,
}

impl Mistral3Plan {
    pub(crate) fn load(path: &Path, device_ordinal: usize) -> Result<Self, BackendError> {
        let mut config = Mistral3Config::load(path)?;
        let weights = Mistral3Weights::load(path, &config)?;
        config.classifier_labels = weights.score.as_ref().map_or(0, |score| score.rows);
        let mut inventory: Vec<String> = weights
            .vision
            .iter()
            .chain(&weights.projector)
            .map(|tensor| tensor.name.clone())
            .chain(weights.score.iter().map(|score| score.name.clone()))
            .chain(weights.retained.iter().cloned())
            .collect();
        inventory.sort();
        inventory.dedup();
        let mut cold = ColdUploadPlan::new(1 << 20)
            .map_err(|error| start(format!("Mistral3 cold plan rejected: {error}")))?;
        for tensor in weights.vision.iter().chain(&weights.projector) {
            cold.push(
                &tensor.name,
                tensor.rows,
                tensor.columns,
                MediaDtype::Bf16,
                MediaDevice::Cuda {
                    index: u32::try_from(device_ordinal)
                        .map_err(|_| start("Mistral3 device ordinal exceeds u32"))?,
                },
            )
            .map_err(|error| start(format!("Mistral3 cold plan rejected: {error}")))?;
        }
        let vision_profile = config.vision_readiness();
        let text_profile = config.text_readiness();
        let projector_pending = config.projector_readiness().err();
        Ok(Self {
            vision_bytes: weights
                .vision_elements()
                .checked_mul(2)
                .ok_or_else(|| start("Mistral3 vision byte size overflow"))?,
            projector_bytes: weights
                .projector_elements()
                .checked_mul(2)
                .ok_or_else(|| start("Mistral3 projector byte size overflow"))?,
            config,
            weights,
            inventory,
            cold,
            vision_profile,
            text_profile,
            projector_pending,
        })
    }

    pub(crate) fn media_plan(
        &self,
        cumulative: &[u32],
        images: &[(usize, Arc<infinity_v3_backend_core::ImagePatches>)],
    ) -> Result<Mistral3MediaPlan, BackendError> {
        Mistral3MediaPlan::build(&self.config, cumulative, images)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use infinity_v3_backend_core::ImagePatches;
    use serde_json::json;

    fn config_value(text: serde_json::Value, vision: serde_json::Value) -> serde_json::Value {
        json!({
            "model_type": "mistral3",
            "architectures": ["Mistral3Model"],
            "image_token_index": 10,
            "spatial_merge_size": 2,
            "vision_feature_layer": -1,
            "multimodal_projector_bias": false,
            "projector_hidden_act": "gelu",
            "text_config": text,
            "vision_config": vision,
        })
    }

    fn text_value() -> serde_json::Value {
        json!({
            "model_type": "ministral3",
            "architectures": ["Ministral3Model"],
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 34,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 131072,
            "max_position_embeddings": 32768,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "hidden_act": "silu",
            "is_causal": true,
            "sliding_window": 4096
        })
    }

    fn vision_value() -> serde_json::Value {
        json!({
            "model_type": "pixtral",
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "head_dim": 64,
            "patch_size": 14,
            "image_size": 1540,
            "num_channels": 3,
            "hidden_act": "silu",
            "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"}
        })
    }

    fn parse(value: serde_json::Value) -> Result<Mistral3Config, BackendError> {
        Mistral3Config::from_json(&serde_json::to_vec(&value).unwrap())
    }

    fn text_batch(tokens: usize, vocab: u32) -> infinity_v3_backend_core::Batch {
        infinity_v3_backend_core::Batch {
            multimodal: Vec::new(),
            input_ids: vec![vocab.saturating_sub(1); tokens],
            token_type_ids: vec![0; tokens],
            position_ids: (0..tokens as u32).collect(),
            cumulative_seq_lengths: vec![0, tokens as u32],
            max_length: tokens as u32,
            pooled_indices: Vec::new(),
            raw_indices: Vec::new(),
            compact_input_ids: None,
            compact_position_ids: None,
            scatter_unfold: None,
            fold_gather: None,
            tokens: Vec::new(),
            offsets: Vec::new(),
        }
    }

    fn image_batch(
        config: &Mistral3Config,
        height: usize,
        width: usize,
        tokens: usize,
    ) -> infinity_v3_backend_core::Batch {
        image_batch_with_merge(config, height, width, tokens, config.spatial_merge_size)
    }

    fn image_batch_with_merge(
        config: &Mistral3Config,
        height: usize,
        width: usize,
        tokens: usize,
        merge_size: usize,
    ) -> infinity_v3_backend_core::Batch {
        let patch_dim = config.vision.patch_dim();
        let patches = Arc::new(ImagePatches {
            pixels: vec![0.0; height * width * patch_dim],
            grid_thw: [1, height, width],
            patch_dim,
            merge_size,
        });
        let media = infinity_v3_backend_core::MultimodalEncoding {
            images: vec![(0, patches)],
            audios: Vec::new(),
            position_ids: std::array::from_fn(|_| Vec::new()),
            memory: None,
        };
        infinity_v3_backend_core::Batch {
            multimodal: vec![Some(Arc::new(media))],
            input_ids: vec![0; tokens],
            token_type_ids: vec![0; tokens],
            position_ids: (0..tokens as u32).collect(),
            cumulative_seq_lengths: vec![0, tokens as u32],
            max_length: tokens as u32,
            pooled_indices: Vec::new(),
            raw_indices: Vec::new(),
            compact_input_ids: None,
            compact_position_ids: None,
            scatter_unfold: None,
            fold_gather: None,
            tokens: Vec::new(),
            offsets: Vec::new(),
        }
    }

    #[test]
    fn vision_activation_parses_all_native_admitted_modes() {
        for (name, expected) in [
            ("silu", ProjectorActivation::Silu),
            ("gelu", ProjectorActivation::Gelu),
            ("gelu_pytorch_tanh", ProjectorActivation::GeluTanh),
        ] {
            let mut vision = vision_value();
            vision["hidden_act"] = json!(name);
            assert_eq!(
                parse(config_value(text_value(), vision))
                    .unwrap()
                    .vision
                    .hidden_activation,
                expected
            );
        }
    }

    #[test]
    fn validation_rejects_vocab_and_selector_oob_before_media() {
        let config = parse(config_value(text_value(), vision_value())).unwrap();
        let mut batch = text_batch(2, config.text.vocab_size as u32);
        batch.input_ids[1] = config.text.vocab_size as u32;
        assert!(crate::validate_mistral3(&batch, &config)
            .unwrap_err()
            .to_string()
            .contains("vocab_size"));
        batch.input_ids[1] = 0;
        batch.pooled_indices = vec![1];
        assert!(crate::validate_mistral3(&batch, &config)
            .unwrap_err()
            .to_string()
            .contains("selector exceeds"));
        batch.pooled_indices.clear();
        batch.raw_indices = vec![0, 0];
        assert!(crate::validate_mistral3(&batch, &config).is_ok());
        batch.raw_indices = vec![1];
        assert!(crate::validate_mistral3(&batch, &config)
            .unwrap_err()
            .to_string()
            .contains("selector exceeds"));
    }

    #[test]
    fn validation_rejects_native_invalid_image_grid_and_short_marker_block() {
        let config = parse(config_value(text_value(), vision_value())).unwrap();
        // Native: h/width must be merge-divisible, <= image_size/patch_size, and
        // the complete nh*(nw+1) marker block must fit the sequence.
        let short = image_batch(&config, 4, 4, 5);
        assert!(crate::validate_mistral3(&short, &config)
            .unwrap_err()
            .to_string()
            .contains("marker block"));
        let mut valid = image_batch(&config, 4, 4, 6);
        for index in [0usize, 1, 3, 4] {
            valid.input_ids[index] = config.image_token_index as u32;
        }
        assert_eq!(crate::validate_mistral3(&valid, &config).unwrap(), 6);
        let wrong_marker = image_batch(&config, 4, 4, 6);
        assert!(crate::validate_mistral3(&wrong_marker, &config)
            .unwrap_err()
            .to_string()
            .contains("image marker"));
        let too_wide = image_batch(&config, 112, 2, 2);
        assert!(crate::validate_mistral3(&too_wide, &config)
            .unwrap_err()
            .to_string()
            .contains("image_size"));
        let zero_merge = image_batch_with_merge(&config, 4, 4, 6, 0);
        assert!(crate::validate_mistral3(&zero_merge, &config).is_err());
    }

    #[test]
    fn parses_the_native_composite_geometry_without_stripping_vision() {
        let config = parse(config_value(text_value(), vision_value())).unwrap();
        assert_eq!(config.vision.hidden_size, 1024);
        assert_eq!(config.vision.patch_size, 14);
        assert_eq!(config.vision.num_hidden_layers, 24);
        assert_eq!(config.vision.head_dim, 64);
        assert_eq!(config.text.hidden_size, 4096);
        assert_eq!(config.text.num_hidden_layers, 34);
        assert_eq!(config.spatial_merge_size, 2);
        assert_eq!(config.vision_feature_layer, 24);
        assert!(!config.projector_bias);
        assert_eq!(config.projector_activation, ProjectorActivation::Gelu);
        assert_eq!(config.vision.patch_dim(), 3 * 14 * 14);
        assert!(config.text_readiness().is_ok());
        let profile = config.vision_readiness().unwrap();
        assert_eq!(
            (
                profile.heads,
                profile.kv_heads,
                profile.head_dim,
                profile.causal
            ),
            (16, 16, 64, false)
        );
    }

    #[test]
    fn native_yarn_bidirectional_text_is_admitted() {
        let mut text = text_value();
        text["is_causal"] = json!(false);
        text["sliding_window"] = serde_json::Value::Null;
        text["max_position_embeddings"] = json!(262144);
        text["rope_parameters"] = json!({
            "rope_type": "yarn",
            "rope_theta": 1000000.0,
            "factor": 16.0,
            "original_max_position_embeddings": 16384,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "llama_4_scaling_beta": 0.1
        });
        text["llama_4_scaling"] = json!({"beta": 0.1, "original_max_position_embeddings": 16384});
        let config = parse(config_value(text, vision_value())).unwrap();
        config.text_readiness().unwrap();
        let yarn = config.text.yarn.as_ref().expect("native YaRN retained");
        assert_eq!(yarn.factor, 16.0);
        assert_eq!(yarn.original_max_position_embeddings, 16384.0);
        assert_eq!(yarn.beta_fast, 32.0);
        assert_eq!(yarn.beta_slow, 1.0);
        assert_eq!(yarn.llama_4_scaling_beta, 0.1);
        let query = config
            .text
            .query_scale
            .as_ref()
            .expect("query scale retained");
        assert_eq!(query.beta, 0.1);
        assert_eq!(query.original_max_position_embeddings, 16384.0);
        let decoder = config.text.decoder_config();
        assert!(!decoder.is_causal);
        assert_eq!(decoder.attention_window, None);
        assert!(decoder.yarn.is_some() && decoder.query_scale.is_some());
        // Vision remains a real plan, not a stripped-away tower.
        assert!(config.vision_readiness().is_ok());
    }

    #[test]
    fn rejects_inconsistent_head_dim_and_small_intermediate_arena() {
        let mut vision = vision_value();
        vision["head_dim"] = json!(32);
        let error = parse(config_value(text_value(), vision))
            .unwrap_err()
            .to_string();
        assert!(error.contains("head_dim"), "{error}");

        let mut vision = vision_value();
        vision["intermediate_size"] = json!(512);
        let error = parse(config_value(text_value(), vision))
            .unwrap_err()
            .to_string();
        assert!(error.contains("intermediate_size"), "{error}");

        // Explicit head_dim may differ from hidden_size / heads; Native uses
        // the stored query width rather than deriving it from hidden.
        let mut text = text_value();
        text["head_dim"] = json!(64);
        let config = parse(config_value(text, vision_value())).unwrap();
        assert_eq!(config.text.head_dim, 64);
        assert_eq!(config.text.decoder_config().q_width(), 32 * 64);

        let mut vision = vision_value();
        vision["rope_parameters"] = json!({"rope_type": "yarn", "rope_theta": 10000.0});
        let error = parse(config_value(text_value(), vision))
            .unwrap_err()
            .to_string();
        assert!(error.contains("rotary"), "{error}");
    }

    #[test]
    fn feature_layer_selection_uses_the_matching_snapshot_once() {
        for (feature_layer, depth, expected) in
            [(0usize, 4usize, 0usize), (1, 4, 1), (3, 4, 3), (4, 4, 4)]
        {
            let mut config = parse(config_value(text_value(), vision_value())).unwrap();
            config.vision_feature_layer = feature_layer;
            let mut chosen = None;
            for snapshot in 0..=depth {
                if config.selects_feature_snapshot(snapshot) {
                    assert!(chosen.is_none(), "snapshot selected twice");
                    chosen = Some(snapshot);
                }
            }
            assert_eq!(chosen, Some(expected));
        }
    }

    #[test]
    fn projector_bias_is_rejected_by_readiness() {
        let mut value = config_value(text_value(), vision_value());
        value["multimodal_projector_bias"] = json!(true);
        let config = parse(value).unwrap();
        let pending = config.projector_readiness().unwrap_err();
        assert!(pending.reason().contains("projector bias"), "{pending}");
    }

    #[test]
    fn rejects_multiple_feature_layers_and_wrong_vision_family() {
        let mut value = config_value(text_value(), vision_value());
        value["vision_feature_layer"] = json!([-1, -2]);
        assert!(parse(value).unwrap_err().to_string().contains("multiple"));

        let mut vision = vision_value();
        vision["model_type"] = json!("siglip");
        let error = parse(config_value(text_value(), vision))
            .unwrap_err()
            .to_string();
        assert!(error.contains("pixtral"), "{error}");
    }

    #[test]
    fn image_media_plan_matches_spatial_merge_and_marker_contract() {
        let config = parse(config_value(text_value(), vision_value())).unwrap();
        let grid = [1, 4, 6];
        let patches = Arc::new(ImagePatches {
            pixels: vec![0.0; grid[0] * grid[1] * grid[2] * config.vision.patch_dim()],
            grid_thw: grid,
            patch_dim: config.vision.patch_dim(),
            merge_size: config.spatial_merge_size,
        });
        let cumulative = [0u32, 20];
        let plan = Mistral3MediaPlan::build(&config, &cumulative, &[(2, patches)]).unwrap();
        assert_eq!(plan.total_patches, 24);
        assert_eq!(plan.total_image_tokens, 6);
        assert_eq!(plan.images[0].merge_source_rows[0], vec![0, 1, 6, 7]);
        assert_eq!(plan.images[0].merge_source_rows[1], vec![2, 3, 8, 9]);
        assert_eq!(plan.images[0].marker_indices, vec![2, 3, 4, 6, 7, 8]);
        assert_eq!(plan.images[0].row_positions[0], 0);
        assert_eq!(plan.images[0].column_positions[0], 0);
        assert_eq!(plan.images[0].column_positions[5], 5);

        let mut ids = vec![0u32; 20];
        for &index in &plan.images[0].marker_indices {
            ids[index] = 10;
        }
        plan.validate_markers(&ids, 10).unwrap();
        ids[plan.images[0].marker_indices[0]] = 11;
        assert!(plan
            .validate_markers(&ids, 10)
            .unwrap_err()
            .to_string()
            .contains("expected 10"));
    }

    #[test]
    fn media_plan_rejects_wrong_patch_payload_and_missing_policy_geometry() {
        let config = parse(config_value(text_value(), vision_value())).unwrap();
        let patches = Arc::new(ImagePatches {
            pixels: vec![0.0; 3],
            grid_thw: [1, 4, 6],
            patch_dim: config.vision.patch_dim(),
            merge_size: 2,
        });
        let error = Mistral3MediaPlan::build(&config, &[0, 20], &[(2, patches)])
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("payload length") || error.contains("expected"),
            "{error}"
        );
    }

    fn cached_model_root() -> Option<std::path::PathBuf> {
        let path = std::env::var_os("INFINITY_MISTRAL3_MODEL")?;
        let path = std::path::PathBuf::from(path);
        if path.join("config.json").is_file() {
            Some(path)
        } else {
            None
        }
    }

    #[test]
    fn cold_plan_byte_accounting_matches_vision_and_projector_headers() {
        let config = parse(config_value(text_value(), vision_value())).unwrap();
        let per_layer = 4 * 4 + 4 * 4096 + 2 * 4 * 4096;
        let vision = (1024 * 3 * 14 * 14 + 1024 + 24 * per_layer) * 2;
        let projector = (1024 + 1024 * 4096 + 4096 * 1024 + 4096 * 4096) * 2;
        assert_eq!(config.vision.patch_dim(), 588);
        assert_eq!(config.text_width(), 4096);
        assert!(vision > 0 && projector > 0);
        assert_eq!(config.spatial_merge_size, 2);
    }

    #[test]
    fn cached_processor_fixture_covers_dynamic_duplicate_mixed_readout_and_errors() {
        let Some(root) = cached_model_root() else {
            return;
        };
        let fixture_path = std::env::var_os("INFINITY_MISTRAL3_FIXTURE")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from(
                    "/tmp/oxide-all-model-feature-matrix-v1/mistral3-v2/fixtures/processor-cases.json",
                )
            });
        let Ok(bytes) = std::fs::read(&fixture_path) else {
            return;
        };
        let fixture: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let config = Mistral3Config::load(&root).unwrap();
        let processor = &fixture["processor"];
        assert_eq!(processor["patch_size"], 14);
        assert_eq!(processor["spatial_merge_size"], 2);
        assert_eq!(processor["image_token_index"], 10);
        assert_eq!(processor["image_size"], 1540);
        let assets = &fixture["pixel_assets"];
        let pixel = |name: &str| -> Arc<infinity_v3_backend_core::ImagePatches> {
            let asset = &assets[name];
            let shape = asset["shape"].as_array().unwrap();
            let rows = shape[0].as_u64().unwrap() as usize;
            let columns = shape[1].as_u64().unwrap() as usize;
            let values = asset["values_flat"].as_array().unwrap();
            assert_eq!(values.len(), rows * columns);
            Arc::new(infinity_v3_backend_core::ImagePatches {
                pixels: values
                    .iter()
                    .map(|value| value.as_f64().unwrap() as f32)
                    .collect(),
                grid_thw: [1, 4, 6],
                patch_dim: columns,
                merge_size: 2,
            })
        };
        let small = pixel("grid_1x4x6_d588");
        let large = pixel("grid_1x14x10_d588");
        let image_cases = fixture["image_cases"].as_array().unwrap();
        for case in image_cases {
            let cumulative: Vec<u32> = case["cumulative_token_lengths"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_u64().unwrap() as u32)
                .collect();
            let mut images = Vec::new();
            for image in case["images"].as_array().unwrap() {
                let token_start = image["token_start"].as_u64().unwrap() as usize;
                let grid: Vec<usize> = image["grid_thw"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|value| value.as_u64().unwrap() as usize)
                    .collect();
                let source = if grid[1] == 4 { &small } else { &large };
                let mut patches = (**source).clone();
                patches.grid_thw = [grid[0], grid[1], grid[2]];
                images.push((token_start, Arc::new(patches)));
            }
            let plan = Mistral3MediaPlan::build(&config, &cumulative, &images).unwrap();
            let ids: Vec<u32> = case["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_u64().unwrap() as u32)
                .collect();
            plan.validate_markers(&ids, config.image_token_index)
                .unwrap();
            for (planned, expected) in plan.images.iter().zip(case["images"].as_array().unwrap()) {
                let markers: Vec<usize> = expected["marker_indices"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|value| value.as_u64().unwrap() as usize)
                    .collect();
                assert_eq!(planned.marker_indices, markers);
                assert_eq!(
                    planned.token_count,
                    expected["token_rows"].as_u64().unwrap() as usize
                );
                let merge = expected["merge_source_rows"].as_array().unwrap();
                assert_eq!(planned.merge_source_rows.len(), merge.len());
                for (actual, expected) in planned.merge_source_rows.iter().zip(merge) {
                    let expected: Vec<usize> = expected
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|value| value.as_u64().unwrap() as usize)
                        .collect();
                    assert_eq!(*actual, expected);
                }
            }
        }
        // Readout fixtures describe the pooled/raw split consumed by the shared
        // decoder; they must remain nonempty and disjoint for mixed selection.
        for case in fixture["readout_cases"].as_array().unwrap() {
            let pooled = case["pooled_indices"].as_array().map_or(0, Vec::len);
            let raw = case["raw_indices"].as_array().map_or(0, Vec::len);
            let labels = case["score_logits"].as_array().map_or(0, Vec::len);
            assert!(
                pooled + raw + labels > 0,
                "{} has no readout rows",
                case["name"]
            );
        }
        // Exercise each typed media error with the actual fixture geometry.
        let errors = fixture["error_cases"].as_array().unwrap();
        assert_eq!(errors.len(), 6);
        for case in errors {
            let name = case["name"].as_str().unwrap();
            if name == "marker_id_mismatch" {
                continue;
            }
            let grid = case["grid_thw"].as_array().unwrap();
            let grid = [
                grid[0].as_u64().unwrap() as usize,
                grid[1].as_u64().unwrap() as usize,
                grid[2].as_u64().unwrap() as usize,
            ];
            let mut patches = (*small).clone();
            patches.grid_thw = grid;
            if name == "wrong_patch_dim" {
                patches.patch_dim = case["patch_dim"].as_u64().unwrap() as usize;
                patches.pixels = vec![0.0; grid.iter().product::<usize>() * patches.patch_dim];
            }
            let token_start = case["token_start"].as_u64().unwrap_or(0) as usize;
            let cumulative: Vec<u32> = case
                .get("cumulative_token_lengths")
                .and_then(serde_json::Value::as_array)
                .map(|values| {
                    values
                        .iter()
                        .map(|value| value.as_u64().unwrap() as u32)
                        .collect()
                })
                .unwrap_or_else(|| vec![0, 64]);
            let starts: Vec<usize> = case
                .get("image_token_starts")
                .and_then(serde_json::Value::as_array)
                .map(|values| {
                    values
                        .iter()
                        .map(|value| value.as_u64().unwrap() as usize)
                        .collect()
                })
                .unwrap_or_else(|| vec![token_start]);
            let images: Vec<_> = starts
                .iter()
                .map(|start| (*start, Arc::new(patches.clone())))
                .collect();
            assert!(
                Mistral3MediaPlan::build(&config, &cumulative, &images).is_err(),
                "{name} was accepted"
            );
        }
        // Marker mismatch is a plan validation error, not a layout error.
        if let Some(case) = errors
            .iter()
            .find(|case| case["name"] == "marker_id_mismatch")
        {
            let plan =
                Mistral3MediaPlan::build(&config, &[0, 64], &[(0, Arc::clone(&small))]).unwrap();
            let mut ids = vec![10u32; 64];
            let index = case["bad_marker_index"].as_u64().unwrap() as usize;
            ids[index] = case["bad_marker_value"].as_u64().unwrap() as u32;
            assert!(plan.validate_markers(&ids, 10).is_err());
        }
    }

    #[test]
    fn cached_mistral3_checkpoint_header_inventory_is_complete() {
        let Some(root) = cached_model_root() else {
            return;
        };
        let plan = Mistral3Plan::load(&root, 0).unwrap();
        assert_eq!(plan.config.vision.hidden_size, 1024);
        assert!(plan.weights.vision.len() >= 2 + 24 * 9);
        assert_eq!(plan.weights.projector.len(), 4);
        assert!(plan.cold.total_bytes() > 0);
        assert_eq!(
            plan.cold.tensors().len(),
            plan.weights.vision.len() + plan.weights.projector.len()
        );
        assert!(!plan.inventory.is_empty());
        assert!(plan.weights.text.layers.len() == plan.config.text.num_hidden_layers);
        assert_eq!(plan.weights.projector.len(), 4);
        assert_eq!(plan.inventory.len(), 531);
        assert_eq!(
            plan.cold.total_bytes(),
            (plan.weights.vision_elements() + plan.weights.projector_elements()) * 2
                + plan.cold.padding_bytes()
        );
        assert!(
            crate::mistral3_exec::Mistral3Prepared::pending_reason(&plan).is_none(),
            "the native YaRN/llama_4/bidirectional text recipe is fully admitted"
        );
        assert!(plan.text_profile.is_ok());
        assert!(plan.projector_pending.is_none());
        assert!(plan.vision_profile.is_ok());
        assert!(plan.config.text.yarn.is_some());
        assert!(plan.config.text.query_scale.is_some());
        assert!(!plan.config.text.is_causal);
        // Exact actual-checkpoint composition: 218 vision / 4 projector /
        // 308 language / 1 score = the original 531 tensor headers.
        assert_eq!(plan.weights.vision.len(), 218);
        assert_eq!(plan.weights.projector.len(), 4);
        assert!(
            plan.weights.score.is_some(),
            "classifier score header bound"
        );
        assert_eq!(
            plan.weights.score.as_ref().map(|score| score.name.as_str()),
            Some("language_model.score.weight")
        );
        assert!(plan.config.classifier);
        assert_eq!(plan.weights.score.as_ref().unwrap().rows, 1);
        assert_eq!(plan.config.classifier_labels, 1);
        let admission = crate::admit_mistral3(&root).unwrap();
        assert_eq!(admission.vision_tensors, 218);
        assert_eq!(admission.projector_tensors, 4);
        assert_eq!(admission.text_tensors, 308);
        assert_eq!(admission.score_tensors, 1);
        assert_eq!(
            admission.vision_tensors
                + admission.projector_tensors
                + admission.text_tensors
                + admission.score_tensors,
            531
        );
        assert!(admission.classifier);
        assert_eq!(admission.text_layers, plan.config.text.num_hidden_layers);
        assert_eq!(admission.cold_plan_tensors, plan.cold.tensors().len());
        assert_eq!(admission.cold_plan_bytes, plan.cold.total_bytes());
        assert!(admission
            .vision_profile
            .as_deref()
            .is_some_and(|profile| profile.contains("16heads/16kv/d64")));
        assert!(admission.text_pending.is_none());
        assert!(admission.projector_pending.is_none());
        assert!(admission
            .tensor_inventory
            .iter()
            .any(|name| name == "model.vision_tower.patch_conv.weight"));
    }
    fn official_text(
        hidden: usize,
        intermediate: usize,
        layers: usize,
        theta: f64,
    ) -> serde_json::Value {
        json!({
            "model_type": "ministral3",
            "architectures": ["Ministral3ForCausalLM"],
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "num_hidden_layers": layers,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 131072,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-5,
            "rope_theta": theta,
            "rope_parameters": {
                "rope_type": "yarn",
                "rope_theta": theta,
                "factor": 16.0,
                "original_max_position_embeddings": 16384,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
                "llama_4_scaling_beta": 0.1
            },
            "llama_4_scaling": {"beta": 0.1, "original_max_position_embeddings": 16384},
            "hidden_act": "silu",
            "is_causal": false,
            "use_bidirectional_attention": true,
            "sliding_window": null,
            "tie_word_embeddings": true
        })
    }

    fn official_vision() -> serde_json::Value {
        json!({
            "model_type": "pixtral",
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "head_dim": 64,
            "patch_size": 14,
            "image_size": 1540,
            "num_channels": 3,
            "hidden_act": "silu",
            "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"}
        })
    }

    #[test]
    fn official_3b_8b_14b_wrapper_configs_are_structural() {
        for (label, hidden, intermediate, layers, theta) in [
            ("3b", 3072usize, 9216usize, 26usize, 1_000_000.0f64),
            ("8b", 4096, 14336, 34, 1_000_000.0),
            ("14b", 5120, 16384, 40, 1_000_000_000.0),
        ] {
            let mut value = config_value(
                official_text(hidden, intermediate, layers, theta),
                official_vision(),
            );
            value["architectures"] = json!(["Mistral3ForConditionalGeneration"]);
            value["multimodal_projector_bias"] = json!(false);
            let config = parse(value).unwrap();
            assert_eq!(config.text.hidden_size, hidden, "{label}");
            assert_eq!(config.text.intermediate_size, intermediate, "{label}");
            assert_eq!(config.text.num_hidden_layers, layers, "{label}");
            assert_eq!(config.text.head_dim, 128, "{label}");
            assert_eq!(config.text.num_attention_heads, 32, "{label}");
            assert_eq!(config.text.num_key_value_heads, 8, "{label}");
            assert_eq!(config.text.max_position_embeddings, 262144, "{label}");
            assert!(!config.text.is_causal, "{label}");
            config.text_readiness().unwrap();
            config.vision_readiness().unwrap();
            let decoder = config.text.decoder_config();
            assert!(decoder.transformers_math, "{label}");
            assert!(
                decoder.yarn.is_some() && decoder.query_scale.is_some(),
                "{label}"
            );
            assert_eq!(
                crate::decoder::DecoderMath::select(&decoder).unwrap(),
                crate::decoder::DecoderMath::Mistral3Transformers,
                "{label}"
            );
        }
    }

    #[test]
    fn official_14b_params_json_is_normalized_without_rewriting() {
        let params = json!({
            "dim": 5120,
            "hidden_dim": 16384,
            "n_layers": 40,
            "n_heads": 32,
            "n_kv_heads": 8,
            "head_dim": 128,
            "vocab_size": 131072,
            "max_position_embeddings": 262144,
            "norm_eps": 1e-5,
            "rope_theta": 1000000000.0,
            "rope_parameters": {
                "rope_type": "yarn",
                "rope_theta": 1000000000.0,
                "factor": 16.0,
                "original_max_position_embeddings": 16384,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
                "llama_4_scaling_beta": 0.1
            },
            "llama_4_scaling": {"beta": 0.1, "original_max_position_embeddings": 16384},
            "is_causal": false,
            "vision_encoder": official_vision()
        });
        let normalized = normalize_params(&params).unwrap();
        let config = Mistral3Config::from_json(&serde_json::to_vec(&normalized).unwrap()).unwrap();
        assert_eq!(config.text.hidden_size, 5120);
        assert_eq!(config.text.intermediate_size, 16384);
        assert_eq!(config.text.num_hidden_layers, 40);
        assert_eq!(config.text.rope_theta, 1_000_000_000.0);
        assert!(config.text.yarn.is_some());
        assert!(config.text_readiness().is_ok());
        assert!(config.vision_readiness().is_ok());
    }

    #[test]
    fn params_only_probe_admits_official_short_key_without_config_json() {
        let params = json!({
            "dim": 5120,
            "hidden_dim": 16384,
            "n_layers": 40,
            "n_heads": 32,
            "n_kv_heads": 8,
            "head_dim": 128,
            "vocab_size": 131072,
            "max_position_embeddings": 262144,
            "norm_eps": 1e-5,
            "rope_theta": 1000000000.0,
            "rope_parameters": {
                "rope_type": "yarn",
                "rope_theta": 1000000000.0,
                "factor": 16.0,
                "original_max_position_embeddings": 16384,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
                "llama_4_scaling_beta": 0.1
            },
            "llama_4_scaling": {"beta": 0.1, "original_max_position_embeddings": 16384},
            "is_causal": false,
            "vision_encoder": official_vision()
        });
        let dir = std::env::temp_dir().join(format!(
            "mistral3-params-probe-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("params.json"),
            serde_json::to_vec(&params).unwrap(),
        )
        .unwrap();
        assert!(!dir.join("config.json").exists());
        assert!(crate::params_only_mistral3(&dir).unwrap());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn vision_attention_profile_uses_grouping_ratio_not_absolute_heads() {
        let mut vision = official_vision();
        vision["hidden_size"] = json!(768);
        vision["num_attention_heads"] = json!(12);
        vision["head_dim"] = json!(64);
        let config = parse(config_value(text_value(), vision)).unwrap();
        let profile = config.vision_readiness().unwrap();
        assert_eq!(
            (profile.heads, profile.kv_heads, profile.head_dim),
            (12, 12, 64)
        );
        assert!(crate::fa4::Fa4::supports(&config.vision.attention_spec()));
    }
    #[test]
    fn mistral3_weight_layout_is_explicit_for_hf_and_raw_common_prefixes() {
        assert_eq!(
            detect_weight_layout(
                [
                    "model.language_model.embed_tokens.weight",
                    "model.vision_tower.patch_conv.weight",
                    "language_model.score.weight",
                ]
                .into_iter()
            ),
            Mistral3WeightLayout::HfConverted
        );
        assert_eq!(
            detect_weight_layout(
                [
                    "tok_embeddings.weight",
                    "layers.0.attention.wq.weight",
                    "norm.weight"
                ]
                .into_iter()
            ),
            Mistral3WeightLayout::RawMistralCommon
        );
        assert_eq!(
            detect_weight_layout(["unexpected.weight"].into_iter()),
            Mistral3WeightLayout::Unknown
        );
    }
}
