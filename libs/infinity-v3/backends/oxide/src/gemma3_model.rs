// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//!
//! Typed host-side staging for dense BF16 Gemma3 text checkpoints.
//!
//! Geometry and layer policy follow the pinned legacy Gemma3 path in
//! `gemma_models.cpp` (upstream Candle Gemma3). Upstream Apache-2.0 notices
//! remain in LICENSE and NOTICE. This module only validates configuration and
//! lazy checkpoint metadata. It does not upload weights, build rotary tables,
//! select kernels, or claim GPU engine support.

use crate::decoder_model::{DecoderLayerWeights, DecoderLinear, DecoderMlpWeights};
use crate::model::{elements, read_json, Checkpoint, WeightSource};
use infinity_v3_backend_core::BackendError;
use safetensors::Dtype;
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;

fn start(error: impl std::fmt::Display) -> BackendError {
    BackendError::Start(format!("Oxide gemma3: {error}"))
}

const DEFAULT_THETA: f64 = 10_000.0;
const DEFAULT_RMS_NORM_EPS: f64 = 1e-6;
const DEFAULT_MAX_POSITION_EMBEDDINGS: usize = 8192;
const DEFAULT_SLIDING_WINDOW: usize = 4096;
const DEFAULT_SLIDING_WINDOW_PATTERN: usize = 6;
const DEFAULT_ACTIVATION: &str = "gelu_pytorch_tanh";
const TENSORS_PER_LAYER: usize = 13;
const GLOBAL_TENSORS: usize = 2;
const NATIVE_ROOTS: [&str; 4] = ["model.language_model.", "language_model.", "model.", ""];

#[derive(Debug, Deserialize)]
struct RawGemma3Config {
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
    rope_local_base_freq: Option<f64>,
    sliding_window: Option<usize>,
    #[serde(rename = "_sliding_window_pattern")]
    underscore_sliding_window_pattern: Option<usize>,
    sliding_window_pattern: Option<usize>,
    query_pre_attn_scalar: Option<f64>,
    use_bidirectional_attention: Option<bool>,
    hidden_activation: Option<String>,
    attention_bias: Option<bool>,
    final_logit_softcapping: Option<f64>,
    attn_logit_softcapping: Option<f64>,
    rope_scaling: Option<Value>,
    rope_parameters: Option<Value>,
    partial_rotary_factor: Option<f64>,
    quantization_config: Option<Value>,
    num_kv_shared_layers: Option<i64>,
    enable_moe_block: Option<bool>,
    enable_moe: Option<bool>,
    num_experts: Option<usize>,
    hidden_size_per_layer_input: Option<usize>,
    vision_config: Option<Value>,
    audio_config: Option<Value>,
    text_config: Option<Value>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

/// Validated dense Gemma3 text geometry. Fields are intentionally independent:
/// `head_dim` may differ from `hidden_size / num_attention_heads`, so the query
/// and key/value widths are validated products rather than aliases of `H`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Gemma3Config {
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
    pub(crate) rope_local_base_freq: f64,
    pub(crate) sliding_window: usize,
    pub(crate) sliding_window_pattern: usize,
    pub(crate) query_pre_attn_scalar: f64,
    pub(crate) is_causal: bool,
    pub(crate) global_rope_factor: f32,
}

/// Prepared, engine-neutral attention policy for one layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Gemma3LayerPolicy {
    pub(crate) local: bool,
    pub(crate) window_left: i32,
    pub(crate) window_right: i32,
    pub(crate) theta: f64,
    pub(crate) rope_factor: f32,
    pub(crate) attention_scale: f64,
}

impl Gemma3Config {
    pub(crate) fn load(path: &Path) -> Result<Self, BackendError> {
        let raw: RawGemma3Config = read_json(&path.join("config.json"))?;
        Self::from_raw(raw)
    }

    /// Validated query width. The product was checked during construction.
    pub(crate) fn q_width(&self) -> usize {
        self.num_attention_heads.saturating_mul(self.head_dim)
    }

    /// Validated key/value width. The product was checked during construction.
    pub(crate) fn kv_width(&self) -> usize {
        self.num_key_value_heads.saturating_mul(self.head_dim)
    }

    /// Legacy Gemma3 policy: the native pattern, not `layer_types`, decides
    /// whether a layer is local. Layer `i+1` is global when it is a multiple of
    /// the pattern. Local layers use `rope_local_base_freq`; the linear RoPE
    /// factor applies to global layers only.
    pub(crate) fn layer_policy(&self, index: usize) -> Result<Gemma3LayerPolicy, BackendError> {
        if index >= self.num_hidden_layers {
            return Err(start(format!(
                "gemma3 layer index {index} exceeds configured layer count {}",
                self.num_hidden_layers
            )));
        }
        if self.sliding_window_pattern == 0 {
            return Err(start("gemma3 sliding_window_pattern must be nonzero"));
        }
        if !self.query_pre_attn_scalar.is_finite() || self.query_pre_attn_scalar <= 0.0 {
            return Err(start(
                "gemma3 query_pre_attn_scalar must be finite and positive",
            ));
        }
        let ordinal = index
            .checked_add(1)
            .ok_or_else(|| start("gemma3 layer index overflow"))?;
        let local = ordinal % self.sliding_window_pattern != 0;
        let (window_left, window_right) = if local {
            if self.is_causal {
                let left = i32::try_from(self.sliding_window.saturating_sub(1))
                    .map_err(|_| start("gemma3 sliding window exceeds i32 policy width"))?;
                (left, 0)
            } else {
                let half = i32::try_from(self.sliding_window / 2)
                    .map_err(|_| start("gemma3 sliding window exceeds i32 policy width"))?;
                (half, half)
            }
        } else {
            (-1, -1)
        };
        let theta = if local {
            self.rope_local_base_freq
        } else {
            self.rope_theta
        };
        let rope_factor = if local { 1.0 } else { self.global_rope_factor };
        Ok(Gemma3LayerPolicy {
            local,
            window_left,
            window_right,
            theta,
            rope_factor,
            attention_scale: 1.0 / self.query_pre_attn_scalar.sqrt(),
        })
    }

    fn from_raw(raw: RawGemma3Config) -> Result<Self, BackendError> {
        if !matches!(raw.model_type.as_str(), "gemma3_text" | "gemma3") {
            let reason = match raw.model_type.as_str() {
                "gemma4" | "gemma4_text" => "Gemma4 is outside this scope",
                "embedding_gemma2" => "embedding_gemma2 is outside this scope",
                _ => "only dense Gemma3 text is supported",
            };
            return Err(start(format!(
                "unsupported gemma3 text model_type {}: {reason}",
                raw.model_type
            )));
        }
        if let Some(architectures) = raw.architectures.as_deref() {
            for architecture in architectures {
                if !matches!(
                    architecture.as_str(),
                    "Gemma3TextModel" | "Gemma3ForCausalLM" | "Gemma3Model"
                ) {
                    return Err(start(format!(
                        "unsupported gemma3 text architecture {architecture}; expected Gemma3TextModel, Gemma3ForCausalLM or Gemma3Model"
                    )));
                }
            }
        }
        if non_null(&raw.text_config) {
            return Err(start(
                "nested text_config containers are not supported; pass a flat Gemma3 text config",
            ));
        }
        if non_null(&raw.vision_config) || non_null(&raw.audio_config) {
            return Err(start(
                "multimodal Gemma3 vision/audio containers are not supported; text-only checkpoints only",
            ));
        }
        reject_config_modifiers(&raw)?;

        let hidden_size = required_usize(raw.hidden_size, "hidden_size")?;
        let intermediate_size = required_usize(raw.intermediate_size, "intermediate_size")?;
        let num_hidden_layers = required_usize(raw.num_hidden_layers, "num_hidden_layers")?;
        let num_attention_heads = required_usize(raw.num_attention_heads, "num_attention_heads")?;
        let num_key_value_heads = raw.num_key_value_heads.unwrap_or(num_attention_heads);
        let vocab_size = required_usize(raw.vocab_size, "vocab_size")?;

        for (name, value) in [
            ("hidden_size", hidden_size),
            ("intermediate_size", intermediate_size),
            ("num_hidden_layers", num_hidden_layers),
            ("num_attention_heads", num_attention_heads),
            ("num_key_value_heads", num_key_value_heads),
            ("vocab_size", vocab_size),
        ] {
            if value == 0 {
                return Err(start(format!("gemma3 config {name} must be nonzero")));
            }
        }

        let head_dim = match raw.head_dim {
            Some(0) => return Err(start("gemma3 config head_dim must be nonzero")),
            Some(head_dim) => head_dim,
            None => {
                if hidden_size % num_attention_heads != 0 {
                    return Err(start(
                        "gemma3 hidden_size must divide evenly by num_attention_heads when head_dim is omitted",
                    ));
                }
                hidden_size / num_attention_heads
            }
        };
        if head_dim == 0 || head_dim % 2 != 0 {
            return Err(start("gemma3 rotary head_dim must be positive and even"));
        }
        if num_attention_heads % num_key_value_heads != 0 {
            return Err(start(
                "gemma3 num_attention_heads must divide evenly by num_key_value_heads",
            ));
        }

        let q_width = num_attention_heads
            .checked_mul(head_dim)
            .ok_or_else(|| start("gemma3 query width overflow"))?;
        let kv_width = num_key_value_heads
            .checked_mul(head_dim)
            .ok_or_else(|| start("gemma3 key/value width overflow"))?;
        if q_width == 0 || kv_width == 0 {
            return Err(start("gemma3 query and key/value widths must be nonzero"));
        }
        validate_products(
            hidden_size,
            intermediate_size,
            vocab_size,
            q_width,
            kv_width,
            head_dim,
        )?;

        let max_position_embeddings = raw
            .max_position_embeddings
            .unwrap_or(DEFAULT_MAX_POSITION_EMBEDDINGS);
        if max_position_embeddings == 0 {
            return Err(start(
                "gemma3 config max_position_embeddings must be nonzero",
            ));
        }
        if u32::try_from(max_position_embeddings).is_err() {
            return Err(start(
                "gemma3 max_position_embeddings must fit CUDA position IDs",
            ));
        }

        let rms_norm_eps = positive_scalar(
            raw.rms_norm_eps.unwrap_or(DEFAULT_RMS_NORM_EPS),
            "rms_norm_eps",
        )?;
        let rope_theta = positive_scalar(raw.rope_theta.unwrap_or(DEFAULT_THETA), "rope_theta")?;
        let rope_local_base_freq = positive_scalar(
            raw.rope_local_base_freq.unwrap_or(DEFAULT_THETA),
            "rope_local_base_freq",
        )?;

        let sliding_window = raw.sliding_window.unwrap_or(DEFAULT_SLIDING_WINDOW);
        if sliding_window == 0 {
            return Err(start("gemma3 config sliding_window must be nonzero"));
        }
        if sliding_window > i32::MAX as usize {
            return Err(start("gemma3 sliding_window exceeds i32 policy width"));
        }
        let sliding_window_pattern = raw
            .underscore_sliding_window_pattern
            .or(raw.sliding_window_pattern)
            .unwrap_or(DEFAULT_SLIDING_WINDOW_PATTERN);
        if sliding_window_pattern == 0 {
            return Err(start(
                "gemma3 config sliding_window_pattern must be nonzero",
            ));
        }

        let query_pre_attn_scalar = positive_scalar(
            raw.query_pre_attn_scalar.unwrap_or(head_dim as f64),
            "query_pre_attn_scalar",
        )?;
        let global_rope_factor = parse_global_rope_factor(raw.rope_scaling.as_ref())?;
        let is_causal = !raw.use_bidirectional_attention.unwrap_or(false);

        let activation = raw
            .hidden_activation
            .as_deref()
            .unwrap_or(DEFAULT_ACTIVATION);
        if !matches!(activation, "gelu_pytorch_tanh" | "gelu_new") {
            return Err(start(format!(
                "unsupported gemma3 text activation {activation}; gelu_pytorch_tanh or gelu_new only"
            )));
        }
        if raw.attention_bias == Some(true) {
            return Err(start(
                "gemma3 text attention_bias checkpoints are not supported in this first scope",
            ));
        }

        Ok(Self {
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
            rope_local_base_freq,
            sliding_window,
            sliding_window_pattern,
            query_pre_attn_scalar,
            is_causal,
            global_rope_factor,
        })
    }
}

/// Dense Gemma3 layer sources. `common.post_attention_norm` is the native
/// post-branch `post_attention_layernorm`; the feed-forward pre-norm is kept
/// separately instead of being reinterpreted as a Llama pre-norm.
#[derive(Clone, Debug)]
pub(crate) struct Gemma3LayerWeights {
    pub(crate) common: DecoderLayerWeights,
    pub(crate) pre_feedforward_norm: WeightSource,
    pub(crate) post_feedforward_norm: WeightSource,
}

/// Metadata-only checkpoint layout. Payload bytes are uploaded later by the
/// executor; every source retains its original dtype and shape.
#[derive(Clone, Debug)]
pub(crate) struct Gemma3Weights {
    pub(crate) config: Gemma3Config,
    pub(crate) embed: WeightSource,
    pub(crate) norm: WeightSource,
    pub(crate) layers: Vec<Gemma3LayerWeights>,
}

impl Gemma3Weights {
    pub(crate) fn load(path: &Path) -> Result<Self, BackendError> {
        let config = Gemma3Config::load(path)?;
        crate::decoder_model::validate_decoder_modules(path)?;
        let checkpoint = Checkpoint::open(path)?;
        let root = select_root(&checkpoint)?;
        validate_checkpoint_capacity(&config, &checkpoint, root)?;

        let mut layers = Vec::new();
        layers
            .try_reserve_exact(config.num_hidden_layers)
            .map_err(start)?;
        let h = config.hidden_size;
        let q = config.q_width();
        let kv = config.kv_width();
        let i = config.intermediate_size;
        let d = config.head_dim;

        let embed = checkpoint.source(
            &format!("{root}embed_tokens.weight"),
            &[config.vocab_size, h],
        )?;
        let norm = checkpoint.source(&format!("{root}norm.weight"), &[h])?;

        for layer in 0..config.num_hidden_layers {
            let p = format!("{root}layers.{layer}");
            let input_norm = checkpoint.source(&format!("{p}.input_layernorm.weight"), &[h])?;
            let post_attention_norm =
                checkpoint.source(&format!("{p}.post_attention_layernorm.weight"), &[h])?;
            let pre_feedforward_norm =
                checkpoint.source(&format!("{p}.pre_feedforward_layernorm.weight"), &[h])?;
            let post_feedforward_norm =
                checkpoint.source(&format!("{p}.post_feedforward_layernorm.weight"), &[h])?;
            let q_norm = checkpoint.source(&format!("{p}.self_attn.q_norm.weight"), &[d])?;
            let k_norm = checkpoint.source(&format!("{p}.self_attn.k_norm.weight"), &[d])?;
            let q_linear = fetch_projection(
                &checkpoint,
                layer,
                &format!("{p}.self_attn.q_proj"),
                &[q, h],
            )?;
            let k_linear = fetch_projection(
                &checkpoint,
                layer,
                &format!("{p}.self_attn.k_proj"),
                &[kv, h],
            )?;
            let v_linear = fetch_projection(
                &checkpoint,
                layer,
                &format!("{p}.self_attn.v_proj"),
                &[kv, h],
            )?;
            let o_linear = fetch_projection(
                &checkpoint,
                layer,
                &format!("{p}.self_attn.o_proj"),
                &[h, q],
            )?;
            let gate =
                fetch_projection(&checkpoint, layer, &format!("{p}.mlp.gate_proj"), &[i, h])?;
            let up = fetch_projection(&checkpoint, layer, &format!("{p}.mlp.up_proj"), &[i, h])?;
            let down =
                fetch_projection(&checkpoint, layer, &format!("{p}.mlp.down_proj"), &[h, i])?;
            let common = DecoderLayerWeights {
                input_norm,
                post_attention_norm,
                q_norm: Some(q_norm),
                k_norm: Some(k_norm),
                q: q_linear,
                k: k_linear,
                v: v_linear,
                o: o_linear,
                mlp: DecoderMlpWeights::Dense { gate, up, down },
            };
            layers.push(Gemma3LayerWeights {
                common,
                pre_feedforward_norm,
                post_feedforward_norm,
            });
        }

        require_bf16("embed_tokens", &embed)?;
        require_bf16("norm", &norm)?;
        for layer in &layers {
            let dense = match &layer.common.mlp {
                DecoderMlpWeights::Dense { gate, up, down } => (gate, up, down),
                DecoderMlpWeights::Routed(_) => {
                    return Err(start("gemma3 layer has routed MLP weights"))
                }
            };
            for (name, source) in [
                ("input_layernorm", &layer.common.input_norm),
                (
                    "post_attention_layernorm",
                    &layer.common.post_attention_norm,
                ),
                ("pre_feedforward_layernorm", &layer.pre_feedforward_norm),
                ("post_feedforward_layernorm", &layer.post_feedforward_norm),
                (
                    "q_norm",
                    layer
                        .common
                        .q_norm
                        .as_ref()
                        .ok_or_else(|| start("missing q_norm"))?,
                ),
                (
                    "k_norm",
                    layer
                        .common
                        .k_norm
                        .as_ref()
                        .ok_or_else(|| start("missing k_norm"))?,
                ),
                ("q_proj", &layer.common.q.weight),
                ("k_proj", &layer.common.k.weight),
                ("v_proj", &layer.common.v.weight),
                ("o_proj", &layer.common.o.weight),
                ("gate_proj", &dense.0.weight),
                ("up_proj", &dense.1.weight),
                ("down_proj", &dense.2.weight),
            ] {
                require_bf16(name, source)?;
            }
        }

        Ok(Self {
            config,
            embed,
            norm,
            layers,
        })
    }
}

fn require_bf16(name: &str, source: &WeightSource) -> Result<(), BackendError> {
    if source.dtype() != Dtype::BF16 {
        return Err(start(format!(
            "gemma3 {name} weight must be BF16, found {:?}",
            source.dtype()
        )));
    }
    Ok(())
}

fn required_usize(value: Option<usize>, name: &str) -> Result<usize, BackendError> {
    value.ok_or_else(|| start(format!("missing gemma3 config field {name}")))
}

fn non_null(value: &Option<Value>) -> bool {
    value.as_ref().is_some_and(|value| !value.is_null())
}

fn positive_scalar(value: f64, name: &str) -> Result<f64, BackendError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(start(format!("gemma3 {name} must be finite and positive")));
    }
    Ok(value)
}

fn validate_products(
    hidden_size: usize,
    intermediate_size: usize,
    vocab_size: usize,
    q_width: usize,
    kv_width: usize,
    head_dim: usize,
) -> Result<(), BackendError> {
    for (name, dimensions) in [
        ("embed_tokens", [vocab_size, hidden_size]),
        ("q_proj", [q_width, hidden_size]),
        ("k_proj", [kv_width, hidden_size]),
        ("v_proj", [kv_width, hidden_size]),
        ("o_proj", [hidden_size, q_width]),
        ("gate_proj", [intermediate_size, hidden_size]),
        ("up_proj", [intermediate_size, hidden_size]),
        ("down_proj", [hidden_size, intermediate_size]),
    ] {
        elements(&dimensions)
            .map_err(|error| start(format!("invalid gemma3 {name} dimensions: {error}")))?;
    }
    elements(&[hidden_size])
        .map_err(|error| start(format!("invalid gemma3 norm dimensions: {error}")))?;
    elements(&[head_dim]).map_err(|error| start(format!("invalid gemma3 head_dim: {error}")))?;
    Ok(())
}

fn reject_config_modifiers(raw: &RawGemma3Config) -> Result<(), BackendError> {
    if non_null(&raw.quantization_config) {
        return Err(start(
            "quantized gemma3 checkpoints are not supported by this loader",
        ));
    }
    if non_null(&raw.rope_parameters) {
        return Err(start(
            "gemma3 rope_parameters are not supported; use native rope_scaling default/linear",
        ));
    }
    if raw
        .partial_rotary_factor
        .is_some_and(|factor| factor != 1.0)
    {
        return Err(start(
            "partial rotary gemma3 configurations are not supported",
        ));
    }
    for (name, value) in [
        ("final_logit_softcapping", raw.final_logit_softcapping),
        ("attn_logit_softcapping", raw.attn_logit_softcapping),
    ] {
        if value.is_some_and(|cap| cap != 0.0) {
            return Err(start(format!("gemma3 {name} is outside this first scope")));
        }
    }
    if raw.num_kv_shared_layers.is_some_and(|count| count != 0) {
        return Err(start("shared KV gemma3 layers are not supported"));
    }
    if raw.enable_moe_block == Some(true)
        || raw.enable_moe == Some(true)
        || raw.num_experts.is_some_and(|count| count > 0)
    {
        return Err(start("gemma3 MoE/expert routing is not supported"));
    }
    if raw
        .hidden_size_per_layer_input
        .is_some_and(|width| width != 0)
    {
        return Err(start(
            "gemma3 per-layer embeddings (hidden_size_per_layer_input) are not supported",
        ));
    }
    for (key, value) in &raw.extra {
        if !active_json(value) {
            continue;
        }
        let lower = key.to_ascii_lowercase();
        let unsupported = lower.contains("expert")
            || lower.contains("moe")
            || lower.contains("router")
            || lower.contains("quant")
            || lower.contains("vision")
            || lower.contains("audio")
            || lower.starts_with("image_")
            || lower.starts_with("video_")
            || lower.starts_with("mm_")
            || lower.contains("multi_modal")
            || lower.contains("multimodal")
            || lower.contains("softcap")
            || lower.contains("rope")
            || lower.contains("original_max")
            || lower.contains("per_layer")
            || lower.contains("shared_kv")
            || lower.contains("kv_shared")
            || key == "_fp8_dynamic";
        if unsupported {
            return Err(start(format!(
                "unsupported gemma3 text config modifier {key}; dense Gemma3 text only"
            )));
        }
    }
    Ok(())
}

fn active_json(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(active) => *active,
        Value::Number(number) => number.as_f64() != Some(0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(items) => !items.is_empty(),
    }
}

fn parse_global_rope_factor(value: Option<&Value>) -> Result<f32, BackendError> {
    let Some(value) = value else {
        return Ok(1.0);
    };
    if value.is_null() {
        return Ok(1.0);
    }
    let object = value
        .as_object()
        .ok_or_else(|| start("gemma3 rope_scaling must be a JSON object"))?;
    for key in object.keys() {
        if !matches!(key.as_str(), "rope_type" | "type" | "factor") {
            return Err(start(format!(
                "unsupported gemma3 rope_scaling modifier {key}"
            )));
        }
    }
    if let (Some(left), Some(right)) = (object.get("rope_type"), object.get("type")) {
        if left != right {
            return Err(start(
                "conflicting gemma3 rope_scaling rope_type/type values",
            ));
        }
    }
    let kind = match object.get("rope_type").or_else(|| object.get("type")) {
        None => "default",
        Some(Value::String(kind)) => kind.as_str(),
        Some(_) => return Err(start("gemma3 rope_scaling type must be a string")),
    };
    let factor = match object.get("factor") {
        None | Some(Value::Null) => 1.0,
        Some(Value::Number(number)) => number
            .as_f64()
            .ok_or_else(|| start("gemma3 rope_scaling factor must be a finite number"))?,
        Some(_) => return Err(start("gemma3 rope_scaling factor must be a number")),
    };
    if !factor.is_finite() || factor <= 0.0 {
        return Err(start(
            "gemma3 RoPE scaling factor must be finite and positive",
        ));
    }
    let factor32 = factor as f32;
    if !factor32.is_finite() || factor32 <= 0.0 {
        return Err(start(
            "gemma3 RoPE scaling factor must be representable as a positive f32",
        ));
    }
    match kind {
        "default" => Ok(1.0),
        "linear" => Ok(factor32),
        other => Err(start(format!(
            "unsupported gemma3 RoPE scaling type {other}; default or linear only"
        ))),
    }
}

fn validate_checkpoint_capacity(
    config: &Gemma3Config,
    checkpoint: &Checkpoint,
    root: &str,
) -> Result<(), BackendError> {
    let minimum = config
        .num_hidden_layers
        .checked_mul(TENSORS_PER_LAYER)
        .and_then(|count| count.checked_add(GLOBAL_TENSORS))
        .ok_or_else(|| start("gemma3 checkpoint capacity overflow"))?;
    let available = checkpoint
        .names()
        .filter(|name| name.strip_prefix(root).is_some_and(is_backbone_suffix))
        .count();
    if available < minimum {
        return Err(start(format!(
            "gemma3 checkpoint has {available} tensor names, fewer than the {minimum} required for {} layers",
            config.num_hidden_layers
        )));
    }
    Ok(())
}

fn root_for(name: &str) -> &'static str {
    NATIVE_ROOTS
        .iter()
        .copied()
        .find(|root| name.starts_with(*root))
        .unwrap_or("")
}

fn is_backbone_suffix(suffix: &str) -> bool {
    suffix.starts_with("embed_tokens.")
        || suffix.starts_with("norm.")
        || suffix.starts_with("layers.")
}

fn select_root(checkpoint: &Checkpoint) -> Result<&'static str, BackendError> {
    let mut selected = None;
    let mut anchors = 0usize;
    for name in checkpoint.names() {
        if !name.ends_with("embed_tokens.weight") {
            continue;
        }
        let mut native = false;
        for root in NATIVE_ROOTS {
            if name.strip_prefix(root) == Some("embed_tokens.weight") {
                anchors = anchors
                    .checked_add(1)
                    .ok_or_else(|| start("gemma3 checkpoint anchor overflow"))?;
                selected = Some(root);
                native = true;
            }
        }
        if !native {
            return Err(start(format!(
                "unsupported gemma3 checkpoint namespace {name}; expected a native text root"
            )));
        }
    }
    if anchors == 0 {
        return Err(start("missing weight embed_tokens.weight"));
    }
    if anchors != 1 {
        return Err(start(
            "ambiguous gemma3 checkpoint namespace: multiple embed_tokens.weight roots",
        ));
    }
    let root = selected.ok_or_else(|| start("missing gemma3 checkpoint root"))?;
    for name in checkpoint.names() {
        let actual = root_for(name);
        if actual == root {
            continue;
        }
        let suffix = name.strip_prefix(actual).unwrap_or(name.as_str());
        let foreign_backbone = suffix.ends_with(".norm.weight") || suffix.contains(".layers.");
        if is_backbone_suffix(suffix) || foreign_backbone {
            return Err(start(format!(
                "mixed gemma3 checkpoint roots: found {name} alongside the {root:?} root"
            )));
        }
    }
    Ok(root)
}

fn fetch_projection(
    checkpoint: &Checkpoint,
    layer: usize,
    name: &str,
    weight_shape: &[usize],
) -> Result<DecoderLinear, BackendError> {
    let bias = format!("{name}.bias");
    if checkpoint
        .names()
        .any(|candidate| candidate.as_str() == bias.as_str())
    {
        return Err(start(format!(
            "gemma3 layer {layer} has unsupported projection bias {bias}"
        )));
    }
    Ok(DecoderLinear {
        weight: checkpoint.source(&format!("{name}.weight"), weight_shape)?,
        bias: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{serialize, TensorView};
    use safetensors::Dtype;
    use serde_json::Value;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "infinity-gemma3-weights-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            let fixture = Self(path);
            fixture.raw_config(gemma3_config());
            fixture
        }

        fn raw_config(&self, value: Value) {
            fs::write(
                self.0.join("config.json"),
                serde_json::to_vec(&value).unwrap(),
            )
            .unwrap();
        }

        fn config(&self, overrides: Value) {
            self.raw_config(with_fields(gemma3_config(), overrides));
        }

        fn config_from_disk(&self) -> Gemma3Config {
            Gemma3Config::load(&self.0).unwrap()
        }

        fn write_entries(&self, entries: &[(String, Vec<usize>)]) {
            let data = entries
                .iter()
                .map(|(_, shape)| {
                    let count = shape
                        .iter()
                        .try_fold(1usize, |count, &dim| count.checked_mul(dim))
                        .expect("test fixture shape overflow");
                    vec![0u8; count.checked_mul(2).expect("test fixture byte overflow")]
                })
                .collect::<Vec<_>>();
            let views = entries
                .iter()
                .zip(&data)
                .map(|((name, shape), bytes)| {
                    (
                        name.clone(),
                        TensorView::new(Dtype::BF16, shape.clone(), bytes).unwrap(),
                    )
                })
                .collect::<Vec<_>>();
            fs::write(
                self.0.join("model.safetensors"),
                serialize(
                    views
                        .iter()
                        .map(|(name, view)| (name.clone(), view.clone())),
                    None,
                )
                .unwrap(),
            )
            .unwrap();
        }

        fn write_modules(&self, modules: Value) {
            fs::write(
                self.0.join("modules.json"),
                serde_json::to_vec(&modules).unwrap(),
            )
            .unwrap();
        }

        fn error(&self) -> String {
            match Gemma3Weights::load(&self.0) {
                Ok(_) => panic!("invalid gemma3 fixture was accepted"),
                Err(error) => error.to_string(),
            }
        }

        fn config_error(&self) -> String {
            match Gemma3Config::load(&self.0) {
                Ok(_) => panic!("invalid gemma3 config was accepted"),
                Err(error) => error.to_string(),
            }
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn gemma3_config() -> Value {
        serde_json::json!({
            "model_type": "gemma3_text",
            "architectures": ["Gemma3TextModel"],
            "hidden_size": 10,
            "intermediate_size": 20,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "vocab_size": 12,
            "max_position_embeddings": 32,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10_000.0,
            "rope_local_base_freq": 1_000.0,
            "sliding_window": 4,
            "sliding_window_pattern": 2,
            "query_pre_attn_scalar": 4.0,
            "use_bidirectional_attention": false,
            "hidden_activation": "gelu_pytorch_tanh",
            "attention_bias": false,
            "final_logit_softcapping": null,
            "rope_scaling": null
        })
    }

    fn with_fields(mut value: Value, fields: Value) -> Value {
        {
            let object = value
                .as_object_mut()
                .expect("gemma3 test config must be an object");
            for (key, field) in fields
                .as_object()
                .expect("gemma3 test overrides must be an object")
            {
                object.insert(key.clone(), field.clone());
            }
        }
        value
    }

    fn without_fields(mut value: Value, fields: &[&str]) -> Value {
        {
            let object = value
                .as_object_mut()
                .expect("gemma3 test config must be an object");
            for field in fields {
                object.remove(*field);
            }
        }
        value
    }

    fn gemma_shapes(config: &Gemma3Config, root: &str) -> Vec<(String, Vec<usize>)> {
        let h = config.hidden_size;
        let i = config.intermediate_size;
        let q = config.q_width();
        let kv = config.kv_width();
        let d = config.head_dim;
        let mut out = vec![
            (
                format!("{root}embed_tokens.weight"),
                vec![config.vocab_size, h],
            ),
            (format!("{root}norm.weight"), vec![h]),
        ];
        for layer in 0..config.num_hidden_layers {
            let p = format!("{root}layers.{layer}");
            out.extend([
                (format!("{p}.input_layernorm.weight"), vec![h]),
                (format!("{p}.post_attention_layernorm.weight"), vec![h]),
                (format!("{p}.pre_feedforward_layernorm.weight"), vec![h]),
                (format!("{p}.post_feedforward_layernorm.weight"), vec![h]),
                (format!("{p}.self_attn.q_norm.weight"), vec![d]),
                (format!("{p}.self_attn.k_norm.weight"), vec![d]),
                (format!("{p}.self_attn.q_proj.weight"), vec![q, h]),
                (format!("{p}.self_attn.k_proj.weight"), vec![kv, h]),
                (format!("{p}.self_attn.v_proj.weight"), vec![kv, h]),
                (format!("{p}.self_attn.o_proj.weight"), vec![h, q]),
                (format!("{p}.mlp.gate_proj.weight"), vec![i, h]),
                (format!("{p}.mlp.up_proj.weight"), vec![i, h]),
                (format!("{p}.mlp.down_proj.weight"), vec![h, i]),
            ]);
        }
        out
    }

    #[test]
    fn loads_valid_gemma3_text_weights_with_independent_widths() {
        let fixture = Fixture::new();
        let config = fixture.config_from_disk();
        assert_eq!(config.hidden_size, 10);
        assert_eq!(config.head_dim, 4);
        assert_eq!(config.q_width(), 8);
        assert_eq!(config.kv_width(), 4);
        assert_ne!(config.q_width(), config.hidden_size);
        assert!(config.is_causal);

        let policy0 = config.layer_policy(0).unwrap();
        assert!(policy0.local);
        assert_eq!((policy0.window_left, policy0.window_right), (3, 0));
        assert_eq!(policy0.theta, 1_000.0);
        assert_eq!(policy0.rope_factor, 1.0);
        assert!((policy0.attention_scale - 0.5).abs() < 1e-12);
        let policy1 = config.layer_policy(1).unwrap();
        assert!(!policy1.local);
        assert_eq!((policy1.window_left, policy1.window_right), (-1, -1));
        assert_eq!(policy1.theta, 10_000.0);

        let mut entries = gemma_shapes(&config, "model.language_model.");
        entries.push((
            "lm_head.weight".into(),
            vec![config.vocab_size, config.hidden_size],
        ));
        fixture.write_entries(&entries);

        let loaded = Gemma3Weights::load(&fixture.0).unwrap();
        assert_eq!(loaded.layers.len(), 2);
        assert_eq!(loaded.embed.shape(), &[12, 10]);
        assert_eq!(loaded.norm.shape(), &[10]);
        let layer = &loaded.layers[0];
        assert_eq!(layer.common.input_norm.shape(), &[10]);
        assert_eq!(layer.common.post_attention_norm.shape(), &[10]);
        assert_eq!(layer.pre_feedforward_norm.shape(), &[10]);
        assert_eq!(layer.post_feedforward_norm.shape(), &[10]);
        assert_eq!(layer.common.q_norm.as_ref().unwrap().shape(), &[4]);
        assert_eq!(layer.common.k_norm.as_ref().unwrap().shape(), &[4]);
        assert_eq!(layer.common.q.weight.shape(), &[8, 10]);
        assert_eq!(layer.common.k.weight.shape(), &[4, 10]);
        assert_eq!(layer.common.v.weight.shape(), &[4, 10]);
        assert_eq!(layer.common.o.weight.shape(), &[10, 8]);
        let (gate, up, down) = layer.common.mlp.dense().unwrap();
        assert_eq!(gate.weight.shape(), &[20, 10]);
        assert_eq!(up.weight.shape(), &[20, 10]);
        assert_eq!(down.weight.shape(), &[10, 20]);
        assert!(layer.common.q.bias.is_none());
        assert!(down.bias.is_none());
    }

    #[test]
    fn ignores_hidden_act_like_native_and_bounds_only_real_backbone_capacity() {
        let fixture = Fixture::new();
        fixture.config(serde_json::json!({"hidden_activation": null, "hidden_act": "silu"}));
        let config = fixture.config_from_disk();
        assert_eq!(config.hidden_size, 10);
        let mut entries = vec![
            (
                "model.embed_tokens.weight".to_string(),
                vec![config.vocab_size, config.hidden_size],
            ),
            ("model.norm.weight".to_string(), vec![config.hidden_size]),
        ];
        entries.extend((0..64).map(|i| (format!("unused_{i}.weight"), vec![1])));
        fixture.write_entries(&entries);
        assert!(fixture.error().contains("fewer than"));
    }

    #[test]
    #[ignore = "CPU-only: actual local Gemma3 trained checkpoint metadata; no CUDA"]
    fn actual_trained_gemma3_metadata() {
        let path = std::env::var_os("INFINITY_OXIDE_TEST_GEMMA3_MODEL")
            .expect("Set the actual Gemma3 checkpoint");
        let host = Gemma3Weights::load(Path::new(&path)).unwrap();
        let config = &host.config;
        assert_eq!(
            config.q_width(),
            config.num_attention_heads * config.head_dim
        );
        assert_eq!(
            config.kv_width(),
            config.num_key_value_heads * config.head_dim
        );
        assert_eq!(host.layers.len(), config.num_hidden_layers);
        assert!(config.vocab_size > 0);
        assert_eq!(
            host.layers[0].common.q_norm.as_ref().unwrap().shape(),
            &[config.head_dim]
        );
        assert_eq!(
            host.layers[0].pre_feedforward_norm.shape(),
            &[config.hidden_size]
        );
        for index in 0..config.num_hidden_layers {
            assert_eq!(
                config.layer_policy(index).unwrap().local,
                (index + 1) % config.sliding_window_pattern != 0
            );
        }
    }

    #[test]
    fn parses_harrier_270m_pattern_one_all_global() {
        let fixture = Fixture::new();
        fixture.raw_config(serde_json::json!({
            "_sliding_window_pattern": 1,
            "architectures": ["Gemma3TextModel"],
            "attention_bias": false,
            "head_dim": 256,
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size": 640,
            "intermediate_size": 2048,
            "max_position_embeddings": 32768,
            "model_type": "gemma3_text",
            "num_attention_heads": 4,
            "num_hidden_layers": 18,
            "num_key_value_heads": 1,
            "query_pre_attn_scalar": 256,
            "rms_norm_eps": 1e-6,
            "rope_local_base_freq": 10000.0,
            "rope_scaling": null,
            "rope_theta": 1000000.0,
            "sliding_window": 512,
            "sliding_window_pattern": 1,
            "use_bidirectional_attention": false,
            "vocab_size": 262144
        }));
        let config = fixture.config_from_disk();
        assert_eq!(config.sliding_window_pattern, 1);
        assert_eq!(config.global_rope_factor, 1.0);
        for index in 0..config.num_hidden_layers {
            let policy = config.layer_policy(index).unwrap();
            assert!(!policy.local);
            assert_eq!(policy.theta, 1_000_000.0);
            assert_eq!(policy.rope_factor, 1.0);
            assert!((policy.attention_scale - 1.0 / 16.0).abs() < 1e-15);
        }
    }

    #[test]
    fn parses_kalm_12b_linear_rope_and_local_layers() {
        let fixture = Fixture::new();
        fixture.raw_config(serde_json::json!({
            "_sliding_window_pattern": 6,
            "architectures": ["Gemma3TextModel"],
            "attention_bias": false,
            "head_dim": 256,
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size": 3840,
            "intermediate_size": 15360,
            "max_position_embeddings": 131072,
            "model_type": "gemma3_text",
            "num_attention_heads": 16,
            "num_hidden_layers": 48,
            "num_key_value_heads": 8,
            "query_pre_attn_scalar": 256,
            "rms_norm_eps": 1e-6,
            "rope_local_base_freq": 10000.0,
            "rope_scaling": {"factor": 8.0, "rope_type": "linear"},
            "rope_theta": 1000000.0,
            "sliding_window": 1024,
            "vocab_size": 262208
        }));
        let config = fixture.config_from_disk();
        assert_eq!(config.q_width(), 4096);
        assert_eq!(config.kv_width(), 2048);
        assert_eq!(config.global_rope_factor, 8.0);
        assert_eq!(config.max_position_embeddings, 131_072);
        let local = config.layer_policy(0).unwrap();
        assert!(local.local);
        assert_eq!((local.window_left, local.window_right), (1023, 0));
        assert_eq!(local.theta, 10_000.0);
        assert_eq!(local.rope_factor, 1.0);
        let global = config.layer_policy(5).unwrap();
        assert!(!global.local);
        assert_eq!(global.theta, 1_000_000.0);
        assert_eq!(global.rope_factor, 8.0);
    }

    #[test]
    fn parses_harrier_27b_independent_query_scalar() {
        let fixture = Fixture::new();
        fixture.raw_config(serde_json::json!({
            "_sliding_window_pattern": 6,
            "architectures": ["Gemma3TextModel"],
            "attention_bias": false,
            "head_dim": 128,
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size": 5376,
            "intermediate_size": 21504,
            "max_position_embeddings": 131072,
            "model_type": "gemma3_text",
            "num_attention_heads": 32,
            "num_hidden_layers": 62,
            "num_key_value_heads": 16,
            "query_pre_attn_scalar": 168,
            "rms_norm_eps": 1e-6,
            "rope_local_base_freq": 10000.0,
            "rope_scaling": {"factor": 8.0, "rope_type": "linear"},
            "rope_theta": 1000000.0,
            "sliding_window": 1024,
            "vocab_size": 262208
        }));
        let config = fixture.config_from_disk();
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.q_width(), 4096);
        assert_eq!(config.kv_width(), 2048);
        assert_eq!(config.query_pre_attn_scalar, 168.0);
        let local = config.layer_policy(0).unwrap();
        assert!((local.attention_scale - 1.0 / 168.0f64.sqrt()).abs() < 1e-15);
        assert_eq!((local.window_left, local.window_right), (1023, 0));
        assert!(!config.layer_policy(5).unwrap().local);
    }

    #[test]
    fn accepts_flat_gemma3_alias_and_rejects_nested_or_media_containers() {
        let fixture = Fixture::new();
        let alias = with_fields(
            gemma3_config(),
            serde_json::json!({
                "model_type": "gemma3",
                "architectures": ["Gemma3ForCausalLM"]
            }),
        );
        fixture.raw_config(alias);
        let config = fixture.config_from_disk();
        assert_eq!(config.hidden_size, 10);
        for architecture in ["Gemma3TextModel", "Gemma3ForCausalLM", "Gemma3Model"] {
            fixture.raw_config(with_fields(
                gemma3_config(),
                serde_json::json!({"architectures": [architecture]}),
            ));
            assert!(fixture.config_from_disk().hidden_size > 0);
        }
        fixture.raw_config(with_fields(
            gemma3_config(),
            serde_json::json!({"architectures": ["Gemma3ForSequenceClassification"]}),
        ));
        assert!(fixture
            .config_error()
            .contains("unsupported gemma3 text architecture"));
        fixture.raw_config(with_fields(
            gemma3_config(),
            serde_json::json!({"text_config": {"hidden_size": 10}}),
        ));
        assert!(fixture.config_error().contains("nested text_config"));
        fixture.raw_config(with_fields(
            gemma3_config(),
            serde_json::json!({"vision_config": {"hidden_size": 8}}),
        ));
        assert!(fixture.config_error().contains("multimodal"));
    }

    #[test]
    #[ignore = "CPU-only: actual local harrier-oss-v1-27b config metadata; no CUDA"]
    fn actual_harrier_27b_config_metadata() {
        let path = std::env::var_os("INFINITY_OXIDE_TEST_GEMMA3_CONFIG_27B")
            .expect("Set the actual harrier-oss-v1-27b config directory");
        let config = Gemma3Config::load(Path::new(&path)).unwrap();
        assert_eq!(config.hidden_size, 5376);
        assert_eq!(config.intermediate_size, 21504);
        assert_eq!(config.num_hidden_layers, 62);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_key_value_heads, 16);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.vocab_size, 262_208);
        assert_eq!(config.sliding_window, 1024);
        assert_eq!(config.sliding_window_pattern, 6);
        assert_eq!(config.global_rope_factor, 8.0);
        assert_eq!(config.query_pre_attn_scalar, 168.0);
        assert_eq!(config.q_width(), 4096);
        assert_eq!(config.kv_width(), 2048);
        assert!(config.layer_policy(0).unwrap().local);
        assert!(!config.layer_policy(5).unwrap().local);
        assert!(
            (config.layer_policy(0).unwrap().attention_scale - 1.0 / 168.0f64.sqrt()).abs() < 1e-15
        );
    }

    #[test]
    fn native_pattern_overrides_layer_types_and_prefers_underscore_pattern() {
        let fixture = Fixture::new();
        fixture.raw_config(with_fields(
            gemma3_config(),
            serde_json::json!({
                "layer_types": ["full_attention", "sliding_attention"],
                "sliding_window_pattern": 1,
                "_sliding_window_pattern": 2
            }),
        ));
        let config = fixture.config_from_disk();
        assert_eq!(config.sliding_window_pattern, 2);
        assert!(config.layer_policy(0).unwrap().local);
        assert!(!config.layer_policy(1).unwrap().local);
    }

    #[test]
    fn separates_local_theta_and_global_linear_factor() {
        let fixture = Fixture::new();
        fixture.config(serde_json::json!({
            "rope_theta": 500_000.0,
            "rope_local_base_freq": 1_000.0,
            "rope_scaling": {"rope_type": "linear", "factor": 8.0},
            "query_pre_attn_scalar": 16.0
        }));
        let config = fixture.config_from_disk();
        assert_eq!(config.global_rope_factor, 8.0);
        let local = config.layer_policy(0).unwrap();
        assert_eq!(local.theta, 1_000.0);
        assert_eq!(local.rope_factor, 1.0);
        assert!((local.attention_scale - 0.25).abs() < 1e-12);
        let global = config.layer_policy(1).unwrap();
        assert_eq!(global.theta, 500_000.0);
        assert_eq!(global.rope_factor, 8.0);
    }

    #[test]
    fn bidirectional_config_uses_symmetric_local_window() {
        let fixture = Fixture::new();
        fixture.config(serde_json::json!({
            "use_bidirectional_attention": true,
            "query_pre_attn_scalar": 16.0
        }));
        let config = fixture.config_from_disk();
        assert!(!config.is_causal);
        let local = config.layer_policy(0).unwrap();
        assert_eq!((local.window_left, local.window_right), (2, 2));
        assert!((local.attention_scale - 0.25).abs() < 1e-12);
        let global = config.layer_policy(1).unwrap();
        assert_eq!((global.window_left, global.window_right), (-1, -1));
    }

    #[test]
    fn native_defaults_cover_head_dim_positions_rope_and_pattern() {
        let fixture = Fixture::new();
        let value = without_fields(
            gemma3_config(),
            &[
                "head_dim",
                "max_position_embeddings",
                "sliding_window",
                "sliding_window_pattern",
                "rope_theta",
                "rope_local_base_freq",
                "rms_norm_eps",
                "query_pre_attn_scalar",
                "rope_scaling",
                "use_bidirectional_attention",
            ],
        );
        fixture.raw_config(with_fields(
            value,
            serde_json::json!({
                "hidden_size": 8,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "num_hidden_layers": 6
            }),
        ));
        let config = fixture.config_from_disk();
        assert_eq!(config.head_dim, 4);
        assert_eq!(config.q_width(), 8);
        assert_eq!(config.kv_width(), 4);
        assert_eq!(config.max_position_embeddings, 8192);
        assert_eq!(config.sliding_window, 4096);
        assert_eq!(config.sliding_window_pattern, 6);
        assert_eq!(config.rope_theta, 10_000.0);
        assert_eq!(config.rope_local_base_freq, 10_000.0);
        assert_eq!(config.rms_norm_eps, 1e-6);
        assert_eq!(config.query_pre_attn_scalar, 4.0);
        assert_eq!(config.global_rope_factor, 1.0);
        assert!(config.is_causal);
        assert!(config.layer_policy(0).unwrap().local);
        assert!(!config.layer_policy(5).unwrap().local);
    }

    #[test]
    fn rejects_invalid_geometry_scalars_and_unsupported_features() {
        for (overrides, needle) in [
            (
                serde_json::json!({"hidden_size": 0}),
                "hidden_size must be nonzero",
            ),
            (
                serde_json::json!({"num_hidden_layers": 0}),
                "num_hidden_layers must be nonzero",
            ),
            (
                serde_json::json!({"num_attention_heads": 0}),
                "num_attention_heads must be nonzero",
            ),
            (
                serde_json::json!({"vocab_size": 0}),
                "vocab_size must be nonzero",
            ),
            (
                serde_json::json!({"head_dim": 3}),
                "head_dim must be positive and even",
            ),
            (
                serde_json::json!({"head_dim": usize::MAX - 1}),
                "query width overflow",
            ),
            (
                serde_json::json!({"num_attention_heads": 3, "num_key_value_heads": 2}),
                "divide evenly by num_key_value_heads",
            ),
            (serde_json::json!({"rms_norm_eps": 0.0}), "rms_norm_eps"),
            (serde_json::json!({"rope_theta": -1.0}), "rope_theta"),
            (
                serde_json::json!({"rope_local_base_freq": 0.0}),
                "rope_local_base_freq",
            ),
            (
                serde_json::json!({"query_pre_attn_scalar": 0.0}),
                "query_pre_attn_scalar",
            ),
            (serde_json::json!({"sliding_window": 0}), "sliding_window"),
            (
                serde_json::json!({"sliding_window_pattern": 0}),
                "sliding_window_pattern",
            ),
            (
                serde_json::json!({"max_position_embeddings": (u32::MAX as u64) + 1}),
                "fit CUDA position IDs",
            ),
            (
                serde_json::json!({"model_type": "gemma4"}),
                "unsupported gemma3 text model_type",
            ),
            (
                serde_json::json!({"model_type": "embedding_gemma2"}),
                "unsupported gemma3 text model_type",
            ),
            (
                serde_json::json!({"architectures": ["Gemma3ForConditionalGeneration"]}),
                "unsupported gemma3 text architecture",
            ),
            (
                serde_json::json!({"quantization_config": {"quant_method": "gptq"}}),
                "quantized gemma3",
            ),
            (
                serde_json::json!({"_fp8_dynamic": true}),
                "unsupported gemma3 text config modifier _fp8_dynamic",
            ),
            (
                serde_json::json!({"enable_moe_block": true}),
                "MoE/expert routing",
            ),
            (serde_json::json!({"num_experts": 8}), "MoE/expert routing"),
            (serde_json::json!({"num_kv_shared_layers": 1}), "shared KV"),
            (
                serde_json::json!({"hidden_size_per_layer_input": 16}),
                "per-layer embeddings",
            ),
            (
                serde_json::json!({"hidden_activation": "relu"}),
                "unsupported gemma3 text activation",
            ),
            (
                serde_json::json!({"hidden_activation": "silu"}),
                "unsupported gemma3 text activation",
            ),
            (
                serde_json::json!({"hidden_activation": "gelu"}),
                "unsupported gemma3 text activation",
            ),
            (
                serde_json::json!({"attention_bias": true}),
                "attention_bias",
            ),
            (
                serde_json::json!({"final_logit_softcapping": 30.0}),
                "final_logit_softcapping",
            ),
            (
                serde_json::json!({"rope_scaling": {"rope_type": "yarn"}}),
                "unsupported gemma3 RoPE scaling type",
            ),
            (
                serde_json::json!({"rope_scaling": {"rope_type": "linear", "factor": 0.0}}),
                "RoPE scaling factor",
            ),
            (
                serde_json::json!({"rope_scaling": {"rope_type": "linear", "factor": 8.0, "mrope_section": [1]}}),
                "unsupported gemma3 rope_scaling modifier",
            ),
            (
                serde_json::json!({"rope_parameters": {"rope_type": "linear"}}),
                "rope_parameters",
            ),
            (
                serde_json::json!({"partial_rotary_factor": 0.5}),
                "partial rotary",
            ),
            (serde_json::json!({"text_config": {}}), "nested text_config"),
            (
                serde_json::json!({"vision_config": {}}),
                "multimodal Gemma3",
            ),
        ] {
            let fixture = Fixture::new();
            fixture.config(overrides);
            let error = fixture.config_error();
            assert!(error.contains(needle), "{error}");
        }
    }

    #[test]
    fn rejects_nonfinite_scalars_and_nondivisible_default_head_dim() {
        let mut raw: RawGemma3Config =
            serde_json::from_value(gemma3_config()).expect("valid raw config");
        raw.rope_theta = Some(f64::NAN);
        let error = Gemma3Config::from_raw(raw).unwrap_err().to_string();
        assert!(error.contains("rope_theta"), "{error}");

        let mut raw: RawGemma3Config =
            serde_json::from_value(gemma3_config()).expect("valid raw config");
        raw.rms_norm_eps = Some(f64::INFINITY);
        let error = Gemma3Config::from_raw(raw).unwrap_err().to_string();
        assert!(error.contains("rms_norm_eps"), "{error}");

        let fixture = Fixture::new();
        fixture.raw_config(with_fields(
            without_fields(gemma3_config(), &["head_dim"]),
            serde_json::json!({"num_attention_heads": 3}),
        ));
        let error = fixture.config_error();
        assert!(error.contains("divide evenly"), "{error}");
    }

    #[test]
    fn requires_complete_qk_and_feedforward_norm_metadata() {
        let fixture = Fixture::new();
        let config = fixture.config_from_disk();
        let mut missing_k = gemma_shapes(&config, "model.language_model.");
        missing_k.push((
            "model.language_model.layers.0.unused.weight".into(),
            vec![1],
        ));
        missing_k.retain(|(name, _)| !name.ends_with(".layers.0.self_attn.k_norm.weight"));
        fixture.write_entries(&missing_k);
        let error = fixture.error();
        assert!(
            error.contains("missing weight model.language_model.layers.0.self_attn.k_norm.weight"),
            "{error}"
        );

        let fixture = Fixture::new();
        let config = fixture.config_from_disk();
        let mut wrong_q = gemma_shapes(&config, "model.language_model.");
        for (name, shape) in &mut wrong_q {
            if name.as_str() == "model.language_model.layers.1.self_attn.q_norm.weight" {
                *shape = vec![2];
            }
        }
        fixture.write_entries(&wrong_q);
        let error = fixture.error();
        assert!(error.contains("wrong shape"), "{error}");

        let fixture = Fixture::new();
        let config = fixture.config_from_disk();
        let mut missing_pre = gemma_shapes(&config, "model.language_model.");
        missing_pre.push((
            "model.language_model.layers.0.unused.weight".into(),
            vec![1],
        ));
        missing_pre
            .retain(|(name, _)| !name.ends_with(".layers.1.pre_feedforward_layernorm.weight"));
        fixture.write_entries(&missing_pre);
        let error = fixture.error();
        assert!(
            error.contains("pre_feedforward_layernorm.weight"),
            "{error}"
        );

        let fixture = Fixture::new();
        let config = fixture.config_from_disk();
        let mut wrong_post = gemma_shapes(&config, "model.language_model.");
        for (name, shape) in &mut wrong_post {
            if name.as_str() == "model.language_model.layers.0.post_feedforward_layernorm.weight" {
                *shape = vec![9];
            }
        }
        fixture.write_entries(&wrong_post);
        let error = fixture.error();
        assert!(error.contains("wrong shape"), "{error}");
    }

    #[test]
    fn selects_one_native_root_and_rejects_ambiguous_or_mixed_backbones() {
        let fixture = Fixture::new();
        let config = fixture.config_from_disk();
        fixture.write_entries(&gemma_shapes(&config, "language_model."));
        assert_eq!(
            Gemma3Weights::load(&fixture.0).unwrap().layers.len(),
            config.num_hidden_layers
        );

        let fixture = Fixture::new();
        let config = fixture.config_from_disk();
        let mut ambiguous = gemma_shapes(&config, "model.language_model.");
        ambiguous.push((
            "embed_tokens.weight".into(),
            vec![config.vocab_size, config.hidden_size],
        ));
        fixture.write_entries(&ambiguous);
        let error = fixture.error();
        assert!(
            error.contains("ambiguous gemma3 checkpoint namespace"),
            "{error}"
        );

        let fixture = Fixture::new();
        let config = fixture.config_from_disk();
        let mut mixed = gemma_shapes(&config, "model.language_model.");
        mixed.push((
            "model.layers.0.input_layernorm.weight".into(),
            vec![config.hidden_size],
        ));
        fixture.write_entries(&mixed);
        let error = fixture.error();
        assert!(error.contains("mixed gemma3 checkpoint roots"), "{error}");

        let fixture = Fixture::new();
        let config = fixture.config_from_disk();
        fixture.write_entries(&gemma_shapes(&config, "transformer."));
        let error = fixture.error();
        assert!(
            error.contains("unsupported gemma3 checkpoint namespace"),
            "{error}"
        );
    }

    #[test]
    fn rejects_projection_biases_instead_of_silently_ignoring_them() {
        let fixture = Fixture::new();
        let config = fixture.config_from_disk();
        let mut entries = gemma_shapes(&config, "model.language_model.");
        entries.push((
            "model.language_model.layers.0.self_attn.q_proj.bias".into(),
            vec![config.q_width()],
        ));
        fixture.write_entries(&entries);
        let error = fixture.error();
        assert!(error.contains("unsupported projection bias"), "{error}");

        let fixture = Fixture::new();
        let config = fixture.config_from_disk();
        let mut entries = gemma_shapes(&config, "model.language_model.");
        entries.push((
            "model.language_model.layers.1.mlp.down_proj.bias".into(),
            vec![config.hidden_size],
        ));
        fixture.write_entries(&entries);
        let error = fixture.error();
        assert!(error.contains("unsupported projection bias"), "{error}");
    }

    #[test]
    fn rejects_malicious_layer_counts_before_layer_allocation() {
        let fixture = Fixture::new();
        let config = fixture.config_from_disk();
        fixture.write_entries(&gemma_shapes(&config, "model.language_model."));
        fixture.config(serde_json::json!({"num_hidden_layers": 1_000_000}));
        let error = fixture.error();
        assert!(error.contains("fewer than the"), "{error}");

        let fixture = Fixture::new();
        let config = fixture.config_from_disk();
        fixture.write_entries(&gemma_shapes(&config, "model.language_model."));
        fixture.config(serde_json::json!({"num_hidden_layers": usize::MAX / 2}));
        let error = fixture.error();
        assert!(error.contains("capacity overflow"), "{error}");
    }

    #[test]
    fn delegates_modules_json_validation_to_shared_decoder_policy() {
        let fixture = Fixture::new();
        fixture.write_modules(serde_json::json!([
            {"type": "sentence_transformers.models.Transformer"},
            {"type": "sentence_transformers.models.CustomProjection"}
        ]));
        let error = fixture.error();
        assert!(error.contains("must not be ignored"), "{error}");
    }

    #[test]
    fn admits_trained_dense_head_modules_for_shared_reuse() {
        let fixture = Fixture::new();
        let config = fixture.config_from_disk();
        fixture.write_entries(&gemma_shapes(&config, ""));
        let pooling = fixture.0.join("1_Pooling");
        fs::create_dir_all(&pooling).unwrap();
        fs::write(
            pooling.join("config.json"),
            serde_json::to_vec(&serde_json::json!({
                "pooling_mode_cls_token": false,
                "pooling_mode_mean_tokens": false,
                "pooling_mode_max_tokens": false,
                "pooling_mode_mean_sqrt_len_tokens": false,
                "pooling_mode_weightedmean_tokens": false,
                "pooling_mode_lasttoken": true
            }))
            .unwrap(),
        )
        .unwrap();
        fixture.write_modules(serde_json::json!([
            {"type": "sentence_transformers.models.Transformer", "path": ""},
            {"type": "sentence_transformers.models.Pooling", "path": "1_Pooling"},
            {"type": "sentence_transformers.models.Dense", "path": "2_Dense"},
            {"type": "sentence_transformers.models.Normalize"}
        ]));
        let loaded = Gemma3Weights::load(&fixture.0).unwrap();
        assert_eq!(loaded.layers.len(), 2);
        assert_eq!(loaded.embed.shape(), &[12, 10]);
    }
}
