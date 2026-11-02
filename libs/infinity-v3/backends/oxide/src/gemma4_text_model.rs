// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//!
//! Typed host-side staging and reusable execution plans for Native Gemma4 text.
//!
//! Geometry follows the published `gemma_models.cpp` first variants: dense
//! Gemma4 text with optional PLE and shared KV (E2B/E4B), and sparse MoE text
//! (26B-A4B). The only admitted decision contract is Rune's `option_tokens`
//! causal last-token readout. This module validates configuration and
//! checkpoint metadata, and exposes borrowed tensor spans. It performs no
//! upload, kernel selection, device query, synchronization, or execution.
#![allow(dead_code)]

use crate::model::{elements, path_present, read_json, Checkpoint, WeightSource};
use infinity_v3_backend_core::BackendError;
use safetensors::Dtype;
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;

fn start(error: impl std::fmt::Display) -> BackendError {
    BackendError::Start(format!("Oxide gemma4: {error}"))
}

const DEFAULT_THETA: f64 = 10_000.0;
const DEFAULT_RMS_NORM_EPS: f64 = 1e-6;
const DEFAULT_MAX_POSITION_EMBEDDINGS: usize = 8192;
const DEFAULT_SLIDING_WINDOW: usize = 4096;
const DEFAULT_ACTIVATION: &str = "gelu_pytorch_tanh";
const MAX_LAYER_COUNT: usize = 4096;
const TEXT_ROOT_NAMES: [&str; 4] = ["model.language_model.", "language_model.", "model.", ""];
const NON_TEXT_DECISION_FILES: [&str; 8] = [
    "decision_config.json",
    "joint_head_config.json",
    "vela3v_config.json",
    "vela3_config.json",
    "rl_agent_config.json",
    "readout.safetensors",
    "joint_head.safetensors",
    "decision_adapter.safetensors",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Gemma4Contract {
    RuneOptionTokens,
    Embedding,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Gemma4Variant {
    Dense,
    SparseMoe,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Gemma4LayerKind {
    Sliding,
    Full,
}

impl Gemma4LayerKind {
    fn index(self) -> usize {
        match self {
            Self::Full => 0,
            Self::Sliding => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Gemma4RopeKind {
    Default,
    Proportional,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Gemma4Activation {
    GeluPytorchTanh,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Gemma4NormArithmetic {
    DivSqrt,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Gemma4ValueNorm {
    UnitRms,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Gemma4RopeSpec {
    pub(crate) kind: Gemma4RopeKind,
    pub(crate) theta: f64,
    pub(crate) partial_rotary_factor: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Gemma4KvPlan {
    Producer { stores_full_length: bool },
    Shared { producer: usize },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Gemma4RopePlan {
    pub(crate) spec: Gemma4RopeSpec,
    pub(crate) head_dim: usize,
    pub(crate) frequency_pairs: usize,
    pub(crate) rotated_pairs: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Gemma4MoePlan {
    pub(crate) num_experts: usize,
    pub(crate) top_k: usize,
    pub(crate) intermediate_size: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Gemma4PleLayerPlan {
    pub(crate) width: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Gemma4LayerPlan {
    pub(crate) index: usize,
    pub(crate) kind: Gemma4LayerKind,
    pub(crate) head_dim: usize,
    pub(crate) num_kv_heads: usize,
    pub(crate) q_width: usize,
    pub(crate) kv_width: usize,
    pub(crate) k_equals_v: bool,
    pub(crate) kv: Gemma4KvPlan,
    pub(crate) rope: Gemma4RopePlan,
    pub(crate) window_left: i32,
    pub(crate) window_right: i32,
    pub(crate) attention_scale: f32,
    pub(crate) qk_norm: bool,
    pub(crate) value_norm: Gemma4ValueNorm,
    pub(crate) dense_intermediate_size: usize,
    pub(crate) ple: Option<Gemma4PleLayerPlan>,
    pub(crate) moe: Option<Gemma4MoePlan>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Gemma4PlePlan {
    pub(crate) width: usize,
    pub(crate) layers: usize,
    pub(crate) vocab_size: usize,
    pub(crate) embed_scale: f32,
    pub(crate) model_projection_scale: f32,
    pub(crate) input_scale: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Gemma4ReadoutPlan {
    pub(crate) last_token: bool,
    pub(crate) option_rows: bool,
    pub(crate) final_logit_softcap: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Gemma4Plan {
    pub(crate) variant: Gemma4Variant,
    pub(crate) causal: bool,
    pub(crate) embedding_scale: f32,
    pub(crate) norm_eps: f32,
    pub(crate) norm_arithmetic: Gemma4NormArithmetic,
    pub(crate) activation: Gemma4Activation,
    pub(crate) router_root_scale: Option<f32>,
    pub(crate) layers: Vec<Gemma4LayerPlan>,
    pub(crate) kv_producers: [Option<usize>; 2],
    pub(crate) ple: Option<Gemma4PlePlan>,
    pub(crate) readout: Gemma4ReadoutPlan,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Gemma4Config {
    pub(crate) contract: Gemma4Contract,
    pub(crate) family: String,
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) global_head_dim: usize,
    pub(crate) num_global_key_value_heads: usize,
    pub(crate) vocab_size: usize,
    pub(crate) vocab_size_per_layer_input: usize,
    pub(crate) max_position_embeddings: usize,
    pub(crate) rms_norm_eps: f64,
    pub(crate) sliding_window: usize,
    pub(crate) layer_types: Vec<Gemma4LayerKind>,
    pub(crate) attention_k_eq_v: bool,
    pub(crate) num_kv_shared_layers: usize,
    pub(crate) enable_moe_block: bool,
    pub(crate) num_experts: Option<usize>,
    pub(crate) top_k_experts: Option<usize>,
    pub(crate) moe_intermediate_size: Option<usize>,
    pub(crate) use_double_wide_mlp: bool,
    pub(crate) hidden_size_per_layer_input: usize,
    pub(crate) tie_word_embeddings: bool,
    pub(crate) final_logit_softcapping: Option<f64>,
    pub(crate) activation: Gemma4Activation,
    pub(crate) sliding_rope: Gemma4RopeSpec,
    pub(crate) full_rope: Gemma4RopeSpec,
}

#[derive(Debug, Deserialize)]
struct RawGemma4 {
    model_type: Option<String>,
    architectures: Option<Vec<String>>,
    hidden_size: Option<usize>,
    intermediate_size: Option<usize>,
    num_hidden_layers: Option<usize>,
    num_attention_heads: Option<usize>,
    num_key_value_heads: Option<usize>,
    head_dim: Option<usize>,
    global_head_dim: Option<usize>,
    num_global_key_value_heads: Option<usize>,
    vocab_size: Option<usize>,
    vocab_size_per_layer_input: Option<usize>,
    max_position_embeddings: Option<usize>,
    rms_norm_eps: Option<f64>,
    sliding_window: Option<usize>,
    layer_types: Option<Vec<String>>,
    attention_bias: Option<bool>,
    attention_dropout: Option<f64>,
    attention_k_eq_v: Option<bool>,
    num_kv_shared_layers: Option<i64>,
    enable_moe_block: Option<bool>,
    num_experts: Option<usize>,
    top_k_experts: Option<usize>,
    moe_intermediate_size: Option<usize>,
    expert_intermediate_size: Option<usize>,
    use_double_wide_mlp: Option<bool>,
    hidden_size_per_layer_input: Option<usize>,
    hidden_activation: Option<String>,
    final_logit_softcapping: Option<f64>,
    attn_logit_softcapping: Option<f64>,
    use_bidirectional_attention: Option<Value>,
    tie_word_embeddings: Option<bool>,
    rope_parameters: Option<Value>,
    rope_scaling: Option<Value>,
    partial_rotary_factor: Option<f64>,
    quantization_config: Option<Value>,
    per_layer_config: Option<Value>,
    dtype: Option<String>,
    text_config: Option<Box<RawGemma4>>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

struct ConfigView<'a> {
    root: &'a RawGemma4,
    text: Option<&'a RawGemma4>,
}

impl<'a> ConfigView<'a> {
    fn u(&self, field: impl Fn(&'a RawGemma4) -> Option<usize>) -> Option<usize> {
        self.text.and_then(&field).or_else(|| field(self.root))
    }

    fn f64(&self, field: impl Fn(&'a RawGemma4) -> Option<f64>) -> Option<f64> {
        self.text.and_then(&field).or_else(|| field(self.root))
    }

    /// Native `decision_models.cpp` resolves LM softcapping from the root key
    /// before the nested `text_config` key.
    fn f64_root_first(&self, field: impl Fn(&'a RawGemma4) -> Option<f64>) -> Option<f64> {
        field(self.root).or_else(|| self.text.and_then(&field))
    }

    fn bool(&self, field: impl Fn(&'a RawGemma4) -> Option<bool>) -> Option<bool> {
        self.text.and_then(&field).or_else(|| field(self.root))
    }

    fn i64(&self, field: impl Fn(&'a RawGemma4) -> Option<i64>) -> Option<i64> {
        self.text.and_then(&field).or_else(|| field(self.root))
    }

    fn value(&self, field: impl Fn(&'a RawGemma4) -> Option<&'a Value>) -> Option<&'a Value> {
        self.text.and_then(&field).or_else(|| field(self.root))
    }

    fn string(&self, field: impl Fn(&'a RawGemma4) -> Option<&'a str>) -> Option<&'a str> {
        self.text.and_then(&field).or_else(|| field(self.root))
    }
}

fn required_usize(value: Option<usize>, name: &str) -> Result<usize, BackendError> {
    value.ok_or_else(|| start(format!("missing Gemma4 config field {name}")))
}

fn nonzero(value: Option<usize>, name: &str) -> Result<usize, BackendError> {
    let value = required_usize(value, name)?;
    if value == 0 {
        return Err(start(format!("gemma4 config {name} must be nonzero")));
    }
    Ok(value)
}

fn non_null(value: Option<&Value>) -> bool {
    value.is_some_and(|value| !value.is_null())
}

fn active_object(value: Option<&Value>) -> bool {
    value.is_some_and(|value| value.as_object().is_some_and(|object| !object.is_empty()))
}

fn finite_positive(value: f64, name: &str) -> Result<f64, BackendError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(start(format!("gemma4 {name} must be finite and positive")));
    }
    Ok(value)
}

impl Gemma4Config {
    pub(crate) fn load(path: &Path, contract: Gemma4Contract) -> Result<Self, BackendError> {
        let raw: RawGemma4 = read_json(&path.join("config.json"))?;
        Self::from_raw(raw, contract)
    }

    fn from_raw(root: RawGemma4, contract: Gemma4Contract) -> Result<Self, BackendError> {
        if let Some(text) = root.text_config.as_deref() {
            if text.text_config.is_some() {
                return Err(start("nested text_config.text_config is not supported"));
            }
            if text
                .model_type
                .as_deref()
                .is_some_and(|family| family != "gemma4_text" && family != "gemma4")
            {
                return Err(start(format!(
                    "unsupported nested text model_type {:?}; expected gemma4_text",
                    text.model_type
                )));
            }
        }
        let view = ConfigView {
            root: &root,
            text: root.text_config.as_deref(),
        };
        let family = view
            .string(|raw| raw.model_type.as_deref())
            .ok_or_else(|| start("missing Gemma4 model_type"))?;
        if family != "gemma4" && family != "gemma4_text" {
            return Err(start(format!(
                "unsupported Gemma4 text model_type {family}; gemma4 and gemma4_text only"
            )));
        }
        let architectures = view
            .root
            .architectures
            .as_deref()
            .or_else(|| view.text.and_then(|text| text.architectures.as_deref()));
        if let Some(architectures) = architectures {
            for architecture in architectures {
                if !matches!(
                    architecture.as_str(),
                    "Gemma4ForConditionalGeneration"
                        | "Gemma4ForCausalLM"
                        | "Gemma4TextModel"
                        | "Gemma4Model"
                ) {
                    return Err(start(format!(
                        "unsupported Gemma4 text architecture {architecture}; Rune option-token text backbones only"
                    )));
                }
            }
        }
        reject_config_modifiers(&view)?;

        let hidden_size = nonzero(view.u(|raw| raw.hidden_size), "hidden_size")?;
        let intermediate_size = nonzero(view.u(|raw| raw.intermediate_size), "intermediate_size")?;
        let num_hidden_layers = nonzero(view.u(|raw| raw.num_hidden_layers), "num_hidden_layers")?;
        if num_hidden_layers > MAX_LAYER_COUNT {
            return Err(start(format!(
                "gemma4 num_hidden_layers {num_hidden_layers} exceeds supported bound {MAX_LAYER_COUNT}"
            )));
        }
        let num_attention_heads =
            nonzero(view.u(|raw| raw.num_attention_heads), "num_attention_heads")?;
        let num_key_value_heads = view
            .u(|raw| raw.num_key_value_heads)
            .unwrap_or(num_attention_heads);
        if num_key_value_heads == 0 {
            return Err(start("gemma4 num_key_value_heads must be nonzero"));
        }
        if !num_attention_heads.is_multiple_of(num_key_value_heads) {
            return Err(start(
                "gemma4 num_attention_heads must divide evenly by num_key_value_heads",
            ));
        }
        let head_dim = match view.u(|raw| raw.head_dim) {
            Some(0) => return Err(start("gemma4 config head_dim must be nonzero")),
            Some(head_dim) => head_dim,
            None => {
                if !hidden_size.is_multiple_of(num_attention_heads) {
                    return Err(start(
                        "gemma4 hidden_size must divide evenly by num_attention_heads when head_dim is omitted",
                    ));
                }
                hidden_size / num_attention_heads
            }
        };
        let global_head_dim = view.u(|raw| raw.global_head_dim).unwrap_or(head_dim);
        for (name, value) in [("head_dim", head_dim), ("global_head_dim", global_head_dim)] {
            if value == 0 || !value.is_multiple_of(2) {
                return Err(start(format!(
                    "gemma4 rotary {name} must be positive and even"
                )));
            }
        }
        let num_global_key_value_heads = view
            .u(|raw| raw.num_global_key_value_heads)
            .unwrap_or(num_key_value_heads);
        if num_global_key_value_heads == 0
            || !num_attention_heads.is_multiple_of(num_global_key_value_heads)
        {
            return Err(start(
                "gemma4 num_global_key_value_heads must divide num_attention_heads",
            ));
        }
        let vocab_size = nonzero(view.u(|raw| raw.vocab_size), "vocab_size")?;
        let max_position_embeddings = view
            .u(|raw| raw.max_position_embeddings)
            .unwrap_or(DEFAULT_MAX_POSITION_EMBEDDINGS);
        if max_position_embeddings == 0 {
            return Err(start(
                "gemma4 config max_position_embeddings must be nonzero",
            ));
        }
        if u32::try_from(max_position_embeddings).is_err() {
            return Err(start(
                "gemma4 max_position_embeddings must fit CUDA position IDs",
            ));
        }
        let rms_norm_eps = finite_positive(
            view.f64(|raw| raw.rms_norm_eps)
                .unwrap_or(DEFAULT_RMS_NORM_EPS),
            "rms_norm_eps",
        )?;
        if !(rms_norm_eps as f32).is_finite() || rms_norm_eps as f32 <= 0.0 {
            return Err(start("gemma4 rms_norm_eps must be positive as f32"));
        }
        let sliding_window = view
            .u(|raw| raw.sliding_window)
            .unwrap_or(DEFAULT_SLIDING_WINDOW);
        if sliding_window == 0 || sliding_window > i32::MAX as usize {
            return Err(start(
                "gemma4 sliding_window must be nonzero and fit the causal i32 window",
            ));
        }
        let layer_types = view
            .text
            .and_then(|text| text.layer_types.clone())
            .or_else(|| view.root.layer_types.clone());
        let layer_types = match layer_types {
            Some(types) => {
                if types.len() != num_hidden_layers {
                    return Err(start(format!(
                        "gemma4 layer_types has {} entries for {num_hidden_layers} layers",
                        types.len()
                    )));
                }
                types
                    .into_iter()
                    .map(|kind| match kind.as_str() {
                        "sliding_attention" => Ok(Gemma4LayerKind::Sliding),
                        "full_attention" => Ok(Gemma4LayerKind::Full),
                        other => Err(start(format!(
                            "unsupported Gemma4 attention type {other}; sliding_attention or full_attention only"
                        ))),
                    })
                    .collect::<Result<Vec<_>, _>>()?
            }
            None => vec![Gemma4LayerKind::Full; num_hidden_layers],
        };
        let attention_k_eq_v = view.bool(|raw| raw.attention_k_eq_v).unwrap_or(false);
        if !attention_k_eq_v && view.u(|raw| raw.num_global_key_value_heads).is_some() {
            return Err(start(
                "gemma4 num_global_key_value_heads requires attention_k_eq_v",
            ));
        }
        let shared_layers = view.i64(|raw| raw.num_kv_shared_layers).unwrap_or(0);
        if shared_layers < 0 {
            return Err(start("gemma4 num_kv_shared_layers must be nonnegative"));
        }
        let num_kv_shared_layers = usize::try_from(shared_layers)
            .map_err(|_| start("gemma4 num_kv_shared_layers exceeds usize"))?;
        if num_kv_shared_layers > num_hidden_layers {
            return Err(start(
                "gemma4 num_kv_shared_layers exceeds num_hidden_layers",
            ));
        }
        let first_shared = num_hidden_layers
            .checked_sub(num_kv_shared_layers)
            .ok_or_else(|| start("gemma4 shared KV layer count overflow"))?;
        if num_kv_shared_layers > 0 && first_shared == 0 {
            return Err(start(
                "gemma4 shared KV layers need at least one producer layer",
            ));
        }
        let enable_moe_block = view.bool(|raw| raw.enable_moe_block).unwrap_or(false);
        let num_experts = view.u(|raw| raw.num_experts);
        let top_k_experts = view.u(|raw| raw.top_k_experts);
        let moe_intermediate_size = view.u(|raw| raw.moe_intermediate_size);
        if let Some(expert_intermediate_size) = view.u(|raw| raw.expert_intermediate_size) {
            if Some(expert_intermediate_size) != moe_intermediate_size {
                return Err(start(
                    "gemma4 expert_intermediate_size conflicts with moe_intermediate_size",
                ));
            }
        }
        if enable_moe_block {
            let count = num_experts
                .filter(|count| *count > 0)
                .ok_or_else(|| start("gemma4 enable_moe_block requires nonzero num_experts"))?;
            let top = top_k_experts
                .filter(|top| *top > 0)
                .ok_or_else(|| start("gemma4 enable_moe_block requires nonzero top_k_experts"))?;
            if top > count {
                return Err(start("gemma4 top_k_experts must not exceed num_experts"));
            }
            if moe_intermediate_size.is_none_or(|width| width == 0) {
                return Err(start(
                    "gemma4 enable_moe_block requires nonzero moe_intermediate_size",
                ));
            }
        } else if num_experts.is_some_and(|count| count > 0)
            || top_k_experts.is_some_and(|top| top > 0)
            || moe_intermediate_size.is_some_and(|width| width > 0)
        {
            return Err(start(
                "gemma4 expert routing fields require enable_moe_block=true",
            ));
        }
        let use_double_wide_mlp = view.bool(|raw| raw.use_double_wide_mlp).unwrap_or(false);
        let hidden_size_per_layer_input =
            view.u(|raw| raw.hidden_size_per_layer_input).unwrap_or(0);
        let vocab_size_per_layer_input = view
            .u(|raw| raw.vocab_size_per_layer_input)
            .unwrap_or(vocab_size);
        let tie_word_embeddings = root.tie_word_embeddings.unwrap_or(false)
            || root
                .text_config
                .as_deref()
                .and_then(|text| text.tie_word_embeddings)
                .unwrap_or(false);
        let final_logit_softcapping = match view.f64_root_first(|raw| raw.final_logit_softcapping) {
            Some(cap) => {
                if !cap.is_finite() || cap <= 0.0 {
                    return Err(start(
                        "gemma4 final_logit_softcapping must be finite and positive",
                    ));
                }
                Some(cap)
            }
            None => None,
        };
        let activation = match view
            .string(|raw| raw.hidden_activation.as_deref())
            .unwrap_or(DEFAULT_ACTIVATION)
        {
            "gelu_pytorch_tanh" | "gelu_new" => Gemma4Activation::GeluPytorchTanh,
            other => {
                return Err(start(format!(
                    "unsupported Gemma4 text activation {other}; gelu_pytorch_tanh or gelu_new only"
                )))
            }
        };
        let rope_parameters = view.value(|raw| raw.rope_parameters.as_ref());
        let sliding_rope = parse_rope(rope_parameters, "sliding_attention", 1.0)?;
        let full_rope = parse_rope(rope_parameters, "full_attention", 1.0)?;

        let config = Self {
            contract,
            family: family.to_owned(),
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            global_head_dim,
            num_global_key_value_heads,
            vocab_size,
            vocab_size_per_layer_input,
            max_position_embeddings,
            rms_norm_eps,
            sliding_window,
            layer_types,
            attention_k_eq_v,
            num_kv_shared_layers,
            enable_moe_block,
            num_experts,
            top_k_experts,
            moe_intermediate_size,
            use_double_wide_mlp,
            hidden_size_per_layer_input,
            tie_word_embeddings,
            final_logit_softcapping,
            activation,
            sliding_rope,
            full_rope,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), BackendError> {
        for (name, value) in [
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("head_dim", self.head_dim),
            ("global_head_dim", self.global_head_dim),
            ("vocab_size", self.vocab_size),
            ("max_position_embeddings", self.max_position_embeddings),
            ("sliding_window", self.sliding_window),
        ] {
            if value == 0 {
                return Err(start(format!("gemma4 config {name} must be nonzero")));
            }
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err(start(
                "gemma4 num_attention_heads must divide evenly by num_key_value_heads",
            ));
        }
        if !self.head_dim.is_multiple_of(2) || !self.global_head_dim.is_multiple_of(2) {
            return Err(start("gemma4 rotary head dimensions must be even"));
        }
        for (name, dimensions) in [
            ("embedding", vec![self.vocab_size, self.hidden_size]),
            (
                "query",
                vec![
                    self.num_attention_heads,
                    self.global_head_dim,
                    self.hidden_size,
                ],
            ),
            ("mlp", vec![self.intermediate_size, self.hidden_size]),
        ] {
            elements(&dimensions)
                .map_err(|error| start(format!("gemma4 {name} dimension overflow: {error}")))?;
        }
        if self.hidden_size_per_layer_input > 0 {
            if self.vocab_size_per_layer_input == 0 {
                return Err(start(
                    "gemma4 vocab_size_per_layer_input must be nonzero when PLE is enabled",
                ));
            }
            let width = self
                .num_hidden_layers
                .checked_mul(self.hidden_size_per_layer_input)
                .ok_or_else(|| start("gemma4 PLE width overflow"))?;
            for (name, dimensions) in [
                (
                    "PLE embedding",
                    vec![self.vocab_size_per_layer_input, width],
                ),
                ("PLE projection", vec![width, self.hidden_size]),
            ] {
                wide_elements(&dimensions)
                    .map_err(|error| start(format!("gemma4 {name} dimension overflow: {error}")))?;
            }
        }
        if self.enable_moe_block {
            if let (Some(count), Some(width)) = (self.num_experts, self.moe_intermediate_size) {
                let gate_up = width
                    .checked_mul(2)
                    .ok_or_else(|| start("gemma4 MoE gate/up width overflow"))?;
                wide_elements(&[count, gate_up, self.hidden_size]).map_err(|error| {
                    start(format!("gemma4 MoE expert dimension overflow: {error}"))
                })?;
                wide_elements(&[count, self.hidden_size, width]).map_err(|error| {
                    start(format!(
                        "gemma4 MoE expert down dimension overflow: {error}"
                    ))
                })?;
            }
        }
        Ok(())
    }
}

fn wide_elements(shape: &[usize]) -> Result<usize, BackendError> {
    let count = shape
        .iter()
        .try_fold(1usize, |count, dim| count.checked_mul(*dim))
        .ok_or_else(|| start("Gemma4 weight dimensions overflow"))?;
    if count > isize::MAX as usize {
        return Err(start("Gemma4 weight exceeds host indexing range"));
    }
    Ok(count)
}

fn parse_rope(
    parameters: Option<&Value>,
    layer_type: &str,
    default_partial: f64,
) -> Result<Gemma4RopeSpec, BackendError> {
    let Some(parameters) = parameters else {
        return Ok(Gemma4RopeSpec {
            kind: Gemma4RopeKind::Default,
            theta: DEFAULT_THETA,
            partial_rotary_factor: default_partial,
        });
    };
    if parameters.is_null() {
        return Ok(Gemma4RopeSpec {
            kind: Gemma4RopeKind::Default,
            theta: DEFAULT_THETA,
            partial_rotary_factor: default_partial,
        });
    }
    let object = parameters
        .as_object()
        .ok_or_else(|| start("gemma4 rope_parameters must be a JSON object"))?;
    for key in object.keys() {
        if !matches!(key.as_str(), "sliding_attention" | "full_attention") {
            return Err(start(format!(
                "unsupported gemma4 rope_parameters key {key}"
            )));
        }
    }
    let Some(specific) = object.get(layer_type) else {
        return Ok(Gemma4RopeSpec {
            kind: Gemma4RopeKind::Default,
            theta: DEFAULT_THETA,
            partial_rotary_factor: default_partial,
        });
    };
    if specific.is_null() {
        return Ok(Gemma4RopeSpec {
            kind: Gemma4RopeKind::Default,
            theta: DEFAULT_THETA,
            partial_rotary_factor: default_partial,
        });
    }
    let specific = specific.as_object().ok_or_else(|| {
        start(format!(
            "gemma4 rope_parameters.{layer_type} must be an object"
        ))
    })?;
    for key in specific.keys() {
        if !matches!(
            key.as_str(),
            "rope_type" | "type" | "rope_theta" | "partial_rotary_factor" | "factor"
        ) {
            return Err(start(format!(
                "unsupported gemma4 rope_parameters.{layer_type}.{key}"
            )));
        }
    }
    let kind_name = match specific.get("rope_type").or_else(|| specific.get("type")) {
        None => None,
        Some(Value::String(name)) => Some(name.as_str()),
        Some(_) => {
            return Err(start(format!(
                "gemma4 rope_parameters.{layer_type} rope_type must be a string"
            )))
        }
    };
    if let (Some(left), Some(right)) = (specific.get("rope_type"), specific.get("type")) {
        if left != right {
            return Err(start(format!(
                "conflicting gemma4 rope_parameters.{layer_type} rope_type/type values"
            )));
        }
    }
    let theta = match specific.get("rope_theta") {
        None | Some(Value::Null) => DEFAULT_THETA,
        Some(Value::Number(number)) => number.as_f64().ok_or_else(|| {
            start(format!(
                "gemma4 rope_parameters.{layer_type}.rope_theta must be finite"
            ))
        })?,
        Some(_) => {
            return Err(start(format!(
                "gemma4 rope_parameters.{layer_type}.rope_theta must be a number"
            )))
        }
    };
    if !theta.is_finite() || theta <= 0.0 || !(theta as f32).is_finite() {
        return Err(start(format!(
            "gemma4 rope_parameters.{layer_type}.rope_theta must be positive as f32"
        )));
    }
    let partial = match specific.get("partial_rotary_factor") {
        None | Some(Value::Null) => default_partial,
        Some(Value::Number(number)) => number.as_f64().ok_or_else(|| {
            start(format!(
                "gemma4 rope_parameters.{layer_type}.partial_rotary_factor must be finite"
            ))
        })?,
        Some(_) => {
            return Err(start(format!(
                "gemma4 rope_parameters.{layer_type}.partial_rotary_factor must be a number"
            )))
        }
    };
    if !partial.is_finite() || partial <= 0.0 || partial > 1.0 {
        return Err(start(format!(
            "gemma4 rope_parameters.{layer_type}.partial_rotary_factor must be in (0, 1]"
        )));
    }
    let factor = match specific.get("factor") {
        None | Some(Value::Null) => 1.0,
        Some(Value::Number(number)) => number.as_f64().ok_or_else(|| {
            start(format!(
                "gemma4 rope_parameters.{layer_type}.factor must be finite"
            ))
        })?,
        Some(_) => {
            return Err(start(format!(
                "gemma4 rope_parameters.{layer_type}.factor must be a number"
            )))
        }
    };
    if !factor.is_finite() || factor != 1.0 {
        return Err(start(format!(
            "gemma4 rope_parameters.{layer_type}.factor is unsupported; only 1.0"
        )));
    }
    let kind = match kind_name {
        None => {
            if partial < 1.0 {
                Gemma4RopeKind::Proportional
            } else {
                Gemma4RopeKind::Default
            }
        }
        Some("default") => {
            if partial != 1.0 {
                return Err(start(format!(
                    "gemma4 rope_parameters.{layer_type} default RoPE requires partial_rotary_factor 1"
                )));
            }
            Gemma4RopeKind::Default
        }
        Some("proportional") => Gemma4RopeKind::Proportional,
        Some(other) => {
            return Err(start(format!(
                "unsupported gemma4 RoPE type {other}; default or proportional only"
            )))
        }
    };
    Ok(Gemma4RopeSpec {
        kind,
        theta,
        partial_rotary_factor: partial,
    })
}

fn reject_config_modifiers(view: &ConfigView<'_>) -> Result<(), BackendError> {
    for raw in std::iter::once(view.root).chain(view.text) {
        if non_null(raw.quantization_config.as_ref()) {
            return Err(start("quantized Gemma4 text checkpoints are unsupported"));
        }
        if non_null(raw.rope_scaling.as_ref()) {
            return Err(start(
                "gemma4 rope_scaling is unsupported; use rope_parameters",
            ));
        }
        if active_object(raw.per_layer_config.as_ref()) {
            return Err(start(
                "gemma4 per_layer_config overrides are outside the first scope",
            ));
        }
        if raw
            .partial_rotary_factor
            .is_some_and(|factor| factor != 1.0)
        {
            return Err(start(
                "gemma4 top-level partial_rotary_factor is unsupported; use rope_parameters",
            ));
        }
        if raw.attention_bias == Some(true) {
            return Err(start(
                "gemma4 attention_bias checkpoints are outside the first scope",
            ));
        }
        if raw.attention_dropout.is_some_and(|dropout| dropout != 0.0) {
            return Err(start("gemma4 attention_dropout must be zero"));
        }
        if raw.attn_logit_softcapping.is_some_and(|cap| cap != 0.0) {
            return Err(start(
                "gemma4 attn_logit_softcapping is outside the text contract",
            ));
        }
        if let Some(value) = raw.use_bidirectional_attention.as_ref() {
            match value {
                Value::Null => {}
                Value::String(value) if value == "vision" => {}
                Value::String(value) if value == "all" => {
                    return Err(start(
                        "gemma4 use_bidirectional_attention=all conflicts with the Rune option-token contract",
                    ))
                }
                _ => {
                    return Err(start(
                        "unsupported gemma4 use_bidirectional_attention value",
                    ))
                }
            }
        }
        for (key, value) in &raw.extra {
            let lower = key.to_ascii_lowercase();
            if (lower.contains("quant") || lower.contains("mrope")) && active_json(value) {
                return Err(start(format!(
                    "unsupported Gemma4 text config modifier {key}"
                )));
            }
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

impl Gemma4Plan {
    pub(crate) fn from_config(config: &Gemma4Config) -> Result<Self, BackendError> {
        let variant = if config.enable_moe_block {
            Gemma4Variant::SparseMoe
        } else {
            Gemma4Variant::Dense
        };
        let first_shared = config
            .num_hidden_layers
            .checked_sub(config.num_kv_shared_layers)
            .ok_or_else(|| start("gemma4 shared KV layer count overflow"))?;
        let mut producers = [None; 2];
        if config.num_kv_shared_layers > 0 {
            for (index, kind) in config.layer_types.iter().copied().enumerate() {
                if index < first_shared {
                    producers[kind.index()] = Some(index);
                }
            }
        }
        let moe = config
            .enable_moe_block
            .then(|| {
                Ok::<_, BackendError>(Gemma4MoePlan {
                    num_experts: config
                        .num_experts
                        .ok_or_else(|| start("gemma4 MoE num_experts is missing"))?,
                    top_k: config
                        .top_k_experts
                        .ok_or_else(|| start("gemma4 MoE top_k_experts is missing"))?,
                    intermediate_size: config
                        .moe_intermediate_size
                        .ok_or_else(|| start("gemma4 MoE moe_intermediate_size is missing"))?,
                })
            })
            .transpose()?;
        let ple = (config.hidden_size_per_layer_input > 0).then_some(Gemma4PlePlan {
            width: config.hidden_size_per_layer_input,
            layers: config.num_hidden_layers,
            vocab_size: config.vocab_size_per_layer_input,
            embed_scale: (config.hidden_size_per_layer_input as f64).sqrt() as f32,
            model_projection_scale: (config.hidden_size as f64).sqrt().recip() as f32,
            input_scale: std::f64::consts::FRAC_1_SQRT_2 as f32,
        });
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(config.num_hidden_layers)
            .map_err(start)?;
        for index in 0..config.num_hidden_layers {
            let kind = config.layer_types[index];
            let k_equals_v = kind == Gemma4LayerKind::Full && config.attention_k_eq_v;
            let head_dim = match kind {
                Gemma4LayerKind::Sliding => config.head_dim,
                Gemma4LayerKind::Full => config.global_head_dim,
            };
            let num_kv_heads = if k_equals_v {
                config.num_global_key_value_heads
            } else {
                config.num_key_value_heads
            };
            if num_kv_heads == 0 || !config.num_attention_heads.is_multiple_of(num_kv_heads) {
                return Err(start(format!(
                    "invalid Gemma4 KV head geometry at layer {index}"
                )));
            }
            let q_width = config
                .num_attention_heads
                .checked_mul(head_dim)
                .ok_or_else(|| start(format!("Gemma4 query width overflow at layer {index}")))?;
            let kv_width = num_kv_heads
                .checked_mul(head_dim)
                .ok_or_else(|| start(format!("Gemma4 KV width overflow at layer {index}")))?;
            let shared = config.num_kv_shared_layers > 0 && index >= first_shared;
            let kv = if shared {
                let producer = producers[kind.index()].ok_or_else(|| {
                    start(format!(
                        "Gemma4 shared KV layer {index} has no producer for {:?}",
                        kind
                    ))
                })?;
                Gemma4KvPlan::Shared { producer }
            } else {
                Gemma4KvPlan::Producer {
                    stores_full_length: config.num_kv_shared_layers > 0
                        && producers[kind.index()] == Some(index),
                }
            };
            let dense_intermediate_size = if config.use_double_wide_mlp && shared {
                config
                    .intermediate_size
                    .checked_mul(2)
                    .ok_or_else(|| start("Gemma4 double-wide MLP width overflow"))?
            } else {
                config.intermediate_size
            };
            let rope_spec = match kind {
                Gemma4LayerKind::Sliding => config.sliding_rope,
                Gemma4LayerKind::Full => config.full_rope,
            };
            let frequency_pairs = head_dim / 2;
            let rotated = (rope_spec.partial_rotary_factor * frequency_pairs as f64).floor();
            if !rotated.is_finite() || rotated < 0.0 || rotated > frequency_pairs as f64 {
                return Err(start(format!(
                    "invalid Gemma4 partial rotary factor at layer {index}"
                )));
            }
            let rotated_pairs = rotated as usize;
            if rope_spec.kind == Gemma4RopeKind::Default && rotated_pairs != frequency_pairs {
                return Err(start(format!(
                    "Gemma4 default RoPE must rotate every pair at layer {index}"
                )));
            }
            let (window_left, window_right) = match (kind, config.contract) {
                (Gemma4LayerKind::Sliding, Gemma4Contract::RuneOptionTokens) => (
                    i32::try_from(config.sliding_window - 1)
                        .map_err(|_| start("Gemma4 sliding window exceeds i32"))?,
                    0,
                ),
                (Gemma4LayerKind::Sliding, Gemma4Contract::Embedding) => {
                    let half = i32::try_from(config.sliding_window / 2)
                        .map_err(|_| start("Gemma4 sliding window exceeds i32"))?;
                    (half, half)
                }
                (Gemma4LayerKind::Full, _) => (-1, -1),
            };
            elements(&[q_width, config.hidden_size])
                .map_err(|error| start(format!("Gemma4 query shape overflow: {error}")))?;
            elements(&[kv_width, config.hidden_size])
                .map_err(|error| start(format!("Gemma4 KV shape overflow: {error}")))?;
            elements(&[dense_intermediate_size, config.hidden_size])
                .map_err(|error| start(format!("Gemma4 MLP shape overflow: {error}")))?;
            layers.push(Gemma4LayerPlan {
                index,
                kind,
                head_dim,
                num_kv_heads,
                q_width,
                kv_width,
                k_equals_v,
                kv,
                rope: Gemma4RopePlan {
                    spec: rope_spec,
                    head_dim,
                    frequency_pairs,
                    rotated_pairs,
                },
                window_left,
                window_right,
                attention_scale: 1.0,
                qk_norm: true,
                value_norm: Gemma4ValueNorm::UnitRms,
                dense_intermediate_size,
                ple: ple.map(|ple| Gemma4PleLayerPlan { width: ple.width }),
                moe,
            });
        }
        if let Some(moe) = moe {
            elements(&[
                moe.num_experts,
                moe.intermediate_size
                    .checked_mul(2)
                    .ok_or_else(|| start("Gemma4 expert gate/up width overflow"))?,
                config.hidden_size,
            ])
            .map_err(|error| start(format!("Gemma4 expert gate/up shape overflow: {error}")))?;
            elements(&[moe.num_experts, config.hidden_size, moe.intermediate_size])
                .map_err(|error| start(format!("Gemma4 expert down shape overflow: {error}")))?;
        }
        Ok(Self {
            variant,
            causal: config.contract == Gemma4Contract::RuneOptionTokens,
            embedding_scale: (config.hidden_size as f64).sqrt() as f32,
            norm_eps: config.rms_norm_eps as f32,
            norm_arithmetic: Gemma4NormArithmetic::DivSqrt,
            activation: config.activation,
            router_root_scale: moe.map(|_| (config.hidden_size as f64).sqrt().recip() as f32),
            layers,
            kv_producers: producers,
            ple,
            readout: Gemma4ReadoutPlan {
                last_token: true,
                option_rows: config.contract == Gemma4Contract::RuneOptionTokens,
                final_logit_softcap: config.final_logit_softcapping,
            },
        })
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Gemma4LinearWeights {
    pub(crate) weight: WeightSource,
}

#[derive(Clone, Debug)]
pub(crate) struct Gemma4KvProducerWeights {
    pub(crate) k: Gemma4LinearWeights,
    pub(crate) v: Option<Gemma4LinearWeights>,
    pub(crate) k_norm: WeightSource,
}

#[derive(Clone, Debug)]
pub(crate) struct Gemma4KvIgnoredWeights {
    pub(crate) k: Option<WeightSource>,
    pub(crate) v: Option<WeightSource>,
    pub(crate) k_norm: Option<WeightSource>,
}

#[derive(Clone, Debug)]
pub(crate) enum Gemma4KvWeights {
    Producer(Gemma4KvProducerWeights),
    Shared(Gemma4KvIgnoredWeights),
}

impl Gemma4KvWeights {
    pub(crate) fn weights_present(&self) -> bool {
        matches!(self, Self::Producer(_))
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Gemma4PleLayerWeights {
    pub(crate) input_gate: Gemma4LinearWeights,
    pub(crate) projection: Gemma4LinearWeights,
    pub(crate) post_norm: WeightSource,
}

#[derive(Clone, Debug)]
pub(crate) struct Gemma4MoeWeights {
    pub(crate) router_scale: WeightSource,
    pub(crate) router_projection: WeightSource,
    pub(crate) per_expert_scale: WeightSource,
    pub(crate) gate_up: WeightSource,
    pub(crate) down: WeightSource,
    pub(crate) pre_feedforward_norm_2: WeightSource,
    pub(crate) post_feedforward_norm_1: WeightSource,
    pub(crate) post_feedforward_norm_2: WeightSource,
}

#[derive(Clone, Debug)]
pub(crate) struct Gemma4LayerWeights {
    pub(crate) input_norm: WeightSource,
    pub(crate) post_attention_norm: WeightSource,
    pub(crate) pre_feedforward_norm: WeightSource,
    pub(crate) post_feedforward_norm: WeightSource,
    pub(crate) q: Gemma4LinearWeights,
    pub(crate) q_norm: WeightSource,
    pub(crate) kv: Gemma4KvWeights,
    pub(crate) o: Gemma4LinearWeights,
    pub(crate) gate: Gemma4LinearWeights,
    pub(crate) up: Gemma4LinearWeights,
    pub(crate) down: Gemma4LinearWeights,
    pub(crate) layer_scalar: WeightSource,
    pub(crate) ple: Option<Gemma4PleLayerWeights>,
    pub(crate) moe: Option<Gemma4MoeWeights>,
}

#[derive(Clone, Debug)]
pub(crate) struct Gemma4PleWeights {
    pub(crate) embed_tokens: WeightSource,
    pub(crate) model_projection: WeightSource,
    pub(crate) projection_norm: WeightSource,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Gemma4LmHeadPlan {
    TiedEmbedding { ignored_explicit: bool },
    Explicit,
}

#[derive(Clone, Debug)]
pub(crate) enum Gemma4LmHeadWeights {
    TiedEmbedding { ignored_explicit: bool },
    Explicit(Box<WeightSource>),
}

#[derive(Clone, Debug)]
pub(crate) struct Gemma4Weights {
    pub(crate) config: Gemma4Config,
    pub(crate) plan: Gemma4Plan,
    pub(crate) dtype: Dtype,
    pub(crate) embed: WeightSource,
    pub(crate) final_norm: WeightSource,
    pub(crate) ple: Option<Gemma4PleWeights>,
    pub(crate) layers: Vec<Gemma4LayerWeights>,
    pub(crate) lm_head: Gemma4LmHeadWeights,
    pub(crate) lm_head_plan: Gemma4LmHeadPlan,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Gemma4RowSpan<'a> {
    source: &'a WeightSource,
    row: usize,
    width: usize,
    offset_elements: usize,
}

impl<'a> Gemma4RowSpan<'a> {
    pub(crate) fn row(self) -> usize {
        self.row
    }

    pub(crate) fn width(self) -> usize {
        self.width
    }

    pub(crate) fn offset_elements(self) -> usize {
        self.offset_elements
    }

    pub(crate) fn source_shape(self) -> &'a [usize] {
        self.source.shape()
    }

    pub(crate) fn read_bf16(self, out: &mut [u16]) -> Result<(), BackendError> {
        if out.len() != self.width {
            return Err(start("Gemma4 LM row output width mismatch"));
        }
        self.source.read_bf16(self.offset_elements, out)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Gemma4RuntimeLayout {
    pub(crate) tokens: usize,
    pub(crate) batch: usize,
    pub(crate) max_sequence: usize,
    pub(crate) hidden_elements: usize,
    pub(crate) max_q_elements: usize,
    pub(crate) max_kv_elements: usize,
    pub(crate) max_dense_intermediate_elements: usize,
    pub(crate) moe_router_elements: usize,
    pub(crate) moe_expert_elements: usize,
    pub(crate) ple_elements: usize,
    pub(crate) option_logit_elements: usize,
    pub(crate) estimated_bytes: usize,
}

fn checked_product(left: usize, right: usize, what: &str) -> Result<usize, BackendError> {
    left.checked_mul(right)
        .ok_or_else(|| start(format!("Gemma4 {what} overflow")))
}

fn checked_sum(left: usize, right: usize, what: &str) -> Result<usize, BackendError> {
    left.checked_add(right)
        .ok_or_else(|| start(format!("Gemma4 {what} overflow")))
}

impl Gemma4Weights {
    pub(crate) fn load_rune(path: &Path) -> Result<Self, BackendError> {
        Self::load(path, Gemma4Contract::RuneOptionTokens)
    }

    pub(crate) fn load_embedding(path: &Path) -> Result<Self, BackendError> {
        Self::load(path, Gemma4Contract::Embedding)
    }

    pub(crate) fn load(path: &Path, contract: Gemma4Contract) -> Result<Self, BackendError> {
        let config = Gemma4Config::load(path, contract)?;
        validate_decision_files(path)?;
        crate::decoder_model::validate_decoder_modules(path)?;
        let checkpoint = Checkpoint::open(path)?;
        let root = select_root(&checkpoint)?;
        validate_capacity(&config, &checkpoint, root)?;
        let plan = Gemma4Plan::from_config(&config)?;

        let hidden = config.hidden_size;
        let embed = checkpoint.source(
            &format!("{root}embed_tokens.weight"),
            &[config.vocab_size, hidden],
        )?;
        let final_norm = checkpoint.source(&format!("{root}norm.weight"), &[hidden])?;
        let ple = if config.hidden_size_per_layer_input > 0 {
            let width = config
                .num_hidden_layers
                .checked_mul(config.hidden_size_per_layer_input)
                .ok_or_else(|| start("gemma4 PLE width overflow"))?;
            Some(Gemma4PleWeights {
                embed_tokens: checkpoint.source_wide(
                    &format!("{root}embed_tokens_per_layer.weight"),
                    &[config.vocab_size_per_layer_input, width],
                )?,
                model_projection: checkpoint.source(
                    &format!("{root}per_layer_model_projection.weight"),
                    &[width, hidden],
                )?,
                projection_norm: checkpoint.source(
                    &format!("{root}per_layer_projection_norm.weight"),
                    &[config.hidden_size_per_layer_input],
                )?,
            })
        } else {
            for suffix in [
                "embed_tokens_per_layer.weight",
                "per_layer_model_projection.weight",
                "per_layer_projection_norm.weight",
            ] {
                if checkpoint.contains(&format!("{root}{suffix}")) {
                    return Err(start(format!(
                        "gemma4 PLE tensor {suffix} is present but hidden_size_per_layer_input is zero"
                    )));
                }
            }
            None
        };
        let lm_head = load_lm_head(&checkpoint, &config)?;

        let mut layers = Vec::new();
        layers
            .try_reserve_exact(config.num_hidden_layers)
            .map_err(start)?;
        for layer in &plan.layers {
            let p = format!("{root}layers.{}.", layer.index);
            let q = load_linear(
                &checkpoint,
                layer.index,
                &format!("{p}self_attn.q_proj"),
                &[layer.q_width, hidden],
            )?;
            let q_norm =
                checkpoint.source(&format!("{p}self_attn.q_norm.weight"), &[layer.head_dim])?;
            let o = load_linear(
                &checkpoint,
                layer.index,
                &format!("{p}self_attn.o_proj"),
                &[hidden, layer.q_width],
            )?;
            let kv = match layer.kv {
                Gemma4KvPlan::Producer { .. } => {
                    let k = load_linear(
                        &checkpoint,
                        layer.index,
                        &format!("{p}self_attn.k_proj"),
                        &[layer.kv_width, hidden],
                    )?;
                    let k_norm = checkpoint
                        .source(&format!("{p}self_attn.k_norm.weight"), &[layer.head_dim])?;
                    let v = if layer.k_equals_v {
                        if checkpoint.contains(&format!("{p}self_attn.v_proj.weight")) {
                            return Err(start(format!(
                                "gemma4 layer {} ties K/V but declares v_proj",
                                layer.index
                            )));
                        }
                        None
                    } else {
                        Some(load_linear(
                            &checkpoint,
                            layer.index,
                            &format!("{p}self_attn.v_proj"),
                            &[layer.kv_width, hidden],
                        )?)
                    };
                    Gemma4KvWeights::Producer(Gemma4KvProducerWeights { k, v, k_norm })
                }
                Gemma4KvPlan::Shared { .. } => Gemma4KvWeights::Shared(Gemma4KvIgnoredWeights {
                    k: optional_tensor(
                        &checkpoint,
                        &format!("{p}self_attn.k_proj.weight"),
                        &[layer.kv_width, hidden],
                    )?,
                    v: optional_tensor(
                        &checkpoint,
                        &format!("{p}self_attn.v_proj.weight"),
                        &[layer.kv_width, hidden],
                    )?,
                    k_norm: optional_tensor(
                        &checkpoint,
                        &format!("{p}self_attn.k_norm.weight"),
                        &[layer.head_dim],
                    )?,
                }),
            };
            if checkpoint.contains(&format!("{p}self_attn.v_norm.weight")) {
                return Err(start(format!(
                    "gemma4 layer {} declares an unsupported scaled V norm",
                    layer.index
                )));
            }
            let gate = load_linear(
                &checkpoint,
                layer.index,
                &format!("{p}mlp.gate_proj"),
                &[layer.dense_intermediate_size, hidden],
            )?;
            let up = load_linear(
                &checkpoint,
                layer.index,
                &format!("{p}mlp.up_proj"),
                &[layer.dense_intermediate_size, hidden],
            )?;
            let down = load_linear(
                &checkpoint,
                layer.index,
                &format!("{p}mlp.down_proj"),
                &[hidden, layer.dense_intermediate_size],
            )?;
            let ple_layer = match layer.ple {
                Some(ple) => Some(Gemma4PleLayerWeights {
                    input_gate: load_linear(
                        &checkpoint,
                        layer.index,
                        &format!("{p}per_layer_input_gate"),
                        &[ple.width, hidden],
                    )?,
                    projection: load_linear(
                        &checkpoint,
                        layer.index,
                        &format!("{p}per_layer_projection"),
                        &[hidden, ple.width],
                    )?,
                    post_norm: checkpoint
                        .source(&format!("{p}post_per_layer_input_norm.weight"), &[hidden])?,
                }),
                None => {
                    for suffix in [
                        "per_layer_input_gate.weight",
                        "per_layer_projection.weight",
                        "post_per_layer_input_norm.weight",
                    ] {
                        if checkpoint.contains(&format!("{p}{suffix}")) {
                            return Err(start(format!(
                                "gemma4 layer {} has PLE tensor {suffix} without hidden_size_per_layer_input",
                                layer.index
                            )));
                        }
                    }
                    None
                }
            };
            let moe = match layer.moe {
                Some(moe) => {
                    let gate_up_width = moe
                        .intermediate_size
                        .checked_mul(2)
                        .ok_or_else(|| start("Gemma4 expert gate/up width overflow"))?;
                    Some(Gemma4MoeWeights {
                        router_scale: checkpoint.source(&format!("{p}router.scale"), &[hidden])?,
                        router_projection: checkpoint.source(
                            &format!("{p}router.proj.weight"),
                            &[moe.num_experts, hidden],
                        )?,
                        per_expert_scale: checkpoint
                            .source(&format!("{p}router.per_expert_scale"), &[moe.num_experts])?,
                        gate_up: checkpoint.source(
                            &format!("{p}experts.gate_up_proj"),
                            &[moe.num_experts, gate_up_width, hidden],
                        )?,
                        down: checkpoint.source(
                            &format!("{p}experts.down_proj"),
                            &[moe.num_experts, hidden, moe.intermediate_size],
                        )?,
                        pre_feedforward_norm_2: checkpoint
                            .source(&format!("{p}pre_feedforward_layernorm_2.weight"), &[hidden])?,
                        post_feedforward_norm_1: checkpoint.source(
                            &format!("{p}post_feedforward_layernorm_1.weight"),
                            &[hidden],
                        )?,
                        post_feedforward_norm_2: checkpoint.source(
                            &format!("{p}post_feedforward_layernorm_2.weight"),
                            &[hidden],
                        )?,
                    })
                }
                None => {
                    for suffix in [
                        "experts.gate_up_proj",
                        "experts.down_proj",
                        "router.scale",
                        "router.proj.weight",
                        "router.per_expert_scale",
                        "pre_feedforward_layernorm_2.weight",
                        "post_feedforward_layernorm_1.weight",
                        "post_feedforward_layernorm_2.weight",
                    ] {
                        if checkpoint.contains(&format!("{p}{suffix}")) {
                            return Err(start(format!(
                                "gemma4 layer {} has MoE tensor {suffix} without enable_moe_block",
                                layer.index
                            )));
                        }
                    }
                    None
                }
            };
            layers.push(Gemma4LayerWeights {
                input_norm: checkpoint.source(&format!("{p}input_layernorm.weight"), &[hidden])?,
                post_attention_norm: checkpoint
                    .source(&format!("{p}post_attention_layernorm.weight"), &[hidden])?,
                pre_feedforward_norm: checkpoint
                    .source(&format!("{p}pre_feedforward_layernorm.weight"), &[hidden])?,
                post_feedforward_norm: checkpoint
                    .source(&format!("{p}post_feedforward_layernorm.weight"), &[hidden])?,
                q,
                q_norm,
                kv,
                o,
                gate,
                up,
                down,
                layer_scalar: checkpoint.source(&format!("{p}layer_scalar"), &[1])?,
                ple: ple_layer,
                moe,
            });
        }
        let dtype = embed.dtype();
        if dtype != Dtype::BF16 {
            return Err(start(format!(
                "Rune option-token Gemma4 text requires BF16 weights, found {dtype:?}"
            )));
        }
        for source in committed_sources(&embed, &final_norm, ple.as_ref(), &layers, &lm_head) {
            if source.dtype() != dtype {
                return Err(start(format!(
                    "mixed Gemma4 weight dtypes: expected {dtype:?}, found {:?}",
                    source.dtype()
                )));
            }
        }
        let lm_head_plan = match &lm_head {
            Gemma4LmHeadWeights::TiedEmbedding { ignored_explicit } => {
                Gemma4LmHeadPlan::TiedEmbedding {
                    ignored_explicit: *ignored_explicit,
                }
            }
            Gemma4LmHeadWeights::Explicit(_) => Gemma4LmHeadPlan::Explicit,
        };
        Ok(Self {
            config,
            plan,
            dtype,
            embed,
            final_norm,
            ple,
            layers,
            lm_head,
            lm_head_plan,
        })
    }

    pub(crate) fn lm_head_row(&self, token_id: usize) -> Result<Gemma4RowSpan<'_>, BackendError> {
        if token_id >= self.config.vocab_size {
            return Err(start(format!(
                "Gemma4 option token {token_id} exceeds vocabulary {}",
                self.config.vocab_size
            )));
        }
        let source = match &self.lm_head {
            Gemma4LmHeadWeights::TiedEmbedding { .. } => &self.embed,
            Gemma4LmHeadWeights::Explicit(source) => source.as_ref(),
        };
        let offset_elements = token_id
            .checked_mul(self.config.hidden_size)
            .ok_or_else(|| start("Gemma4 LM row offset overflow"))?;
        let end = offset_elements
            .checked_add(self.config.hidden_size)
            .ok_or_else(|| start("Gemma4 LM row end overflow"))?;
        if end > source.len() {
            return Err(start("Gemma4 LM row exceeds its weight source"));
        }
        Ok(Gemma4RowSpan {
            source,
            row: token_id,
            width: self.config.hidden_size,
            offset_elements,
        })
    }

    pub(crate) fn runtime_layout(
        &self,
        tokens: usize,
        batch: usize,
        max_sequence: usize,
        option_count: usize,
    ) -> Result<Gemma4RuntimeLayout, BackendError> {
        if tokens == 0 || batch == 0 || batch > tokens {
            return Err(start(
                "Gemma4 runtime requires nonempty packed sequences and a positive batch",
            ));
        }
        if max_sequence == 0 || max_sequence > self.config.max_position_embeddings {
            return Err(start(
                "Gemma4 runtime sequence exceeds max_position_embeddings",
            ));
        }
        let capacity = checked_product(batch, max_sequence, "packed sequence capacity")?;
        if tokens > capacity {
            return Err(start(
                "Gemma4 runtime token count exceeds packed sequence capacity",
            ));
        }
        if option_count > self.config.vocab_size {
            return Err(start("Gemma4 option count exceeds vocabulary"));
        }
        let hidden_elements = checked_product(tokens, self.config.hidden_size, "hidden rows")?;
        let mut max_q_elements = 0;
        let mut max_kv_elements = 0;
        let mut max_dense = 0;
        let mut moe_router = 0;
        let mut moe_expert = 0;
        for layer in &self.plan.layers {
            let q = checked_product(tokens, layer.q_width, "query rows")?;
            let kv = checked_product(
                checked_product(tokens, layer.kv_width, "KV rows")?,
                2,
                "packed KV rows",
            )?;
            let dense = checked_product(tokens, layer.dense_intermediate_size, "dense rows")?;
            max_q_elements = max_q_elements.max(q);
            max_kv_elements = max_kv_elements.max(kv);
            max_dense = max_dense.max(dense);
            if let Some(moe) = layer.moe {
                moe_router =
                    moe_router.max(checked_product(tokens, moe.num_experts, "MoE router rows")?);
                moe_expert = moe_expert.max(checked_product(
                    checked_product(tokens, moe.top_k, "MoE selected rows")?,
                    moe.intermediate_size,
                    "MoE expert rows",
                )?);
            }
        }
        let ple_elements = match self.plan.ple {
            Some(ple) => checked_product(
                checked_product(tokens, ple.layers, "PLE layers")?,
                ple.width,
                "PLE rows",
            )?,
            None => 0,
        };
        let option_logit_elements = checked_product(option_count, 1, "option logits")?;
        let element_bytes = 2usize;
        let mut estimated_bytes = checked_product(hidden_elements, element_bytes, "hidden bytes")?;
        for value in [
            max_q_elements,
            max_kv_elements,
            max_dense,
            moe_expert,
            ple_elements,
        ] {
            estimated_bytes = checked_sum(
                estimated_bytes,
                checked_product(value, element_bytes, "activation bytes")?,
                "activation byte total",
            )?;
        }
        estimated_bytes = checked_sum(
            estimated_bytes,
            checked_product(moe_router, 4, "router bytes")?,
            "router byte total",
        )?;
        estimated_bytes = checked_sum(
            estimated_bytes,
            checked_product(option_logit_elements, 4, "logit bytes")?,
            "logit byte total",
        )?;
        Ok(Gemma4RuntimeLayout {
            tokens,
            batch,
            max_sequence,
            hidden_elements,
            max_q_elements,
            max_kv_elements,
            max_dense_intermediate_elements: max_dense,
            moe_router_elements: moe_router,
            moe_expert_elements: moe_expert,
            ple_elements,
            option_logit_elements,
            estimated_bytes,
        })
    }
}

fn committed_sources<'a>(
    embed: &'a WeightSource,
    final_norm: &'a WeightSource,
    ple: Option<&'a Gemma4PleWeights>,
    layers: &'a [Gemma4LayerWeights],
    lm_head: &'a Gemma4LmHeadWeights,
) -> Vec<&'a WeightSource> {
    let mut sources = vec![embed, final_norm];
    if let Some(ple) = ple {
        sources.extend([
            &ple.embed_tokens,
            &ple.model_projection,
            &ple.projection_norm,
        ]);
    }
    if let Gemma4LmHeadWeights::Explicit(source) = lm_head {
        sources.push(source.as_ref());
    }
    for layer in layers {
        sources.extend([
            &layer.input_norm,
            &layer.post_attention_norm,
            &layer.pre_feedforward_norm,
            &layer.post_feedforward_norm,
            &layer.q.weight,
            &layer.q_norm,
            &layer.o.weight,
            &layer.gate.weight,
            &layer.up.weight,
            &layer.down.weight,
            &layer.layer_scalar,
        ]);
        match &layer.kv {
            Gemma4KvWeights::Producer(kv) => {
                sources.push(&kv.k.weight);
                if let Some(v) = &kv.v {
                    sources.push(&v.weight);
                }
                sources.push(&kv.k_norm);
            }
            Gemma4KvWeights::Shared(kv) => {
                sources.extend(kv.k.iter());
                sources.extend(kv.v.iter());
                sources.extend(kv.k_norm.iter());
            }
        }
        if let Some(ple) = &layer.ple {
            sources.extend([
                &ple.input_gate.weight,
                &ple.projection.weight,
                &ple.post_norm,
            ]);
        }
        if let Some(moe) = &layer.moe {
            sources.extend([
                &moe.router_scale,
                &moe.router_projection,
                &moe.per_expert_scale,
                &moe.gate_up,
                &moe.down,
                &moe.pre_feedforward_norm_2,
                &moe.post_feedforward_norm_1,
                &moe.post_feedforward_norm_2,
            ]);
        }
    }
    sources
}

fn load_linear(
    checkpoint: &Checkpoint,
    layer: usize,
    name: &str,
    weight_shape: &[usize],
) -> Result<Gemma4LinearWeights, BackendError> {
    let bias = format!("{name}.bias");
    if checkpoint.contains(&bias) {
        return Err(start(format!(
            "gemma4 layer {layer} has unsupported projection bias {bias}"
        )));
    }
    Ok(Gemma4LinearWeights {
        weight: checkpoint.source(&format!("{name}.weight"), weight_shape)?,
    })
}

fn optional_tensor(
    checkpoint: &Checkpoint,
    name: &str,
    shape: &[usize],
) -> Result<Option<WeightSource>, BackendError> {
    checkpoint.optional_source(name, shape)
}

fn load_lm_head(
    checkpoint: &Checkpoint,
    config: &Gemma4Config,
) -> Result<Gemma4LmHeadWeights, BackendError> {
    let explicit =
        checkpoint.optional_source("lm_head.weight", &[config.vocab_size, config.hidden_size])?;
    if config.tie_word_embeddings {
        return Ok(Gemma4LmHeadWeights::TiedEmbedding {
            ignored_explicit: explicit.is_some(),
        });
    }
    let explicit = explicit.ok_or_else(|| {
        start("gemma4 untied Rune option-token checkpoint requires lm_head.weight")
    })?;
    Ok(Gemma4LmHeadWeights::Explicit(Box::new(explicit)))
}

fn validate_decision_files(path: &Path) -> Result<(), BackendError> {
    for file in NON_TEXT_DECISION_FILES {
        if path_present(&path.join(file))? {
            return Err(start(format!(
                "Gemma4 Rune option-token contract does not use decision file {file}"
            )));
        }
    }
    Ok(())
}

fn validate_capacity(
    config: &Gemma4Config,
    checkpoint: &Checkpoint,
    root: &str,
) -> Result<(), BackendError> {
    let minimum = config
        .num_hidden_layers
        .checked_mul(5)
        .and_then(|count| count.checked_add(3))
        .ok_or_else(|| start("gemma4 checkpoint capacity overflow"))?;
    let available = checkpoint
        .names()
        .filter(|name| name.strip_prefix(root).is_some_and(is_text_backbone_suffix))
        .count();
    if available < minimum {
        return Err(start(format!(
            "gemma4 checkpoint has {available} text tensor names, fewer than the {minimum} required for {} layers",
            config.num_hidden_layers
        )));
    }
    Ok(())
}

fn root_for(name: &str) -> &'static str {
    TEXT_ROOT_NAMES
        .iter()
        .copied()
        .find(|root| name.starts_with(*root))
        .unwrap_or("")
}

fn is_text_backbone_suffix(suffix: &str) -> bool {
    suffix == "embed_tokens.weight"
        || suffix == "embed_tokens_per_layer.weight"
        || suffix == "norm.weight"
        || suffix == "per_layer_model_projection.weight"
        || suffix == "per_layer_projection_norm.weight"
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
        for root in TEXT_ROOT_NAMES {
            if name.strip_prefix(root) == Some("embed_tokens.weight") {
                anchors = anchors
                    .checked_add(1)
                    .ok_or_else(|| start("gemma4 checkpoint anchor overflow"))?;
                selected = Some(root);
                native = true;
            }
        }
        if !native {
            return Err(start(format!(
                "unsupported gemma4 checkpoint namespace {name}; expected a native text root"
            )));
        }
    }
    if anchors == 0 {
        return Err(start("missing weight embed_tokens.weight"));
    }
    if anchors != 1 {
        return Err(start(
            "ambiguous gemma4 checkpoint namespace: multiple embed_tokens.weight roots",
        ));
    }
    let root = selected.ok_or_else(|| start("missing gemma4 checkpoint root"))?;
    for name in checkpoint.names() {
        let actual = root_for(name);
        if actual == root {
            continue;
        }
        let suffix = name.strip_prefix(actual).unwrap_or(name.as_str());
        if is_text_backbone_suffix(suffix) {
            return Err(start(format!(
                "mixed gemma4 checkpoint roots: found {name} alongside the {root:?} root"
            )));
        }
    }
    Ok(root)
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{serialize, TensorView};
    use serde_json::Value;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "infinity-gemma4-weights-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            let fixture = Self(path);
            fixture.raw_config(gemma4_dense_config());
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
            self.raw_config(with_fields(gemma4_dense_config(), overrides));
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

        fn write_entries_with_indexed_tensor(
            &self,
            entries: &[(String, Vec<usize>)],
            indexed_suffix: &str,
        ) {
            let data = entries
                .iter()
                .map(|(name, shape)| {
                    let count = shape
                        .iter()
                        .try_fold(1usize, |count, &dim| count.checked_mul(dim))
                        .expect("test fixture shape overflow");
                    let mut bytes =
                        vec![0u8; count.checked_mul(2).expect("test fixture byte overflow")];
                    if name.ends_with(indexed_suffix) {
                        for (index, chunk) in bytes.chunks_exact_mut(2).enumerate() {
                            let value = u16::try_from(index).expect("test fixture index");
                            chunk.copy_from_slice(&value.to_le_bytes());
                        }
                    }
                    bytes
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

        fn error(&self) -> String {
            match Gemma4Weights::load_rune(&self.0) {
                Ok(_) => panic!("invalid gemma4 fixture was accepted"),
                Err(error) => error.to_string(),
            }
        }

        fn config_error(&self) -> String {
            match Gemma4Config::load(&self.0, Gemma4Contract::RuneOptionTokens) {
                Ok(_) => panic!("invalid gemma4 config was accepted"),
                Err(error) => error.to_string(),
            }
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn gemma4_dense_config() -> Value {
        serde_json::json!({
            "model_type": "gemma4",
            "architectures": ["Gemma4ForConditionalGeneration"],
            "tie_word_embeddings": true,
            "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": 8,
                "intermediate_size": 12,
                "num_hidden_layers": 4,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "global_head_dim": 8,
                "num_global_key_value_heads": null,
                "vocab_size": 12,
                "vocab_size_per_layer_input": 12,
                "max_position_embeddings": 32,
                "rms_norm_eps": 1e-6,
                "sliding_window": 4,
                "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
                "attention_bias": false,
                "attention_dropout": 0.0,
                "attention_k_eq_v": false,
                "num_kv_shared_layers": 1,
                "enable_moe_block": false,
                "use_double_wide_mlp": true,
                "hidden_size_per_layer_input": 2,
                "hidden_activation": "gelu_pytorch_tanh",
                "final_logit_softcapping": 30.0,
                "use_bidirectional_attention": null,
                "tie_word_embeddings": true,
                "rope_parameters": {
                    "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                    "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1000000.0}
                }
            }
        })
    }

    fn gemma4_moe_config() -> Value {
        serde_json::json!({
            "model_type": "gemma4",
            "architectures": ["Gemma4ForConditionalGeneration"],
            "tie_word_embeddings": true,
            "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": 8,
                "intermediate_size": 12,
                "num_hidden_layers": 3,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "global_head_dim": 8,
                "num_global_key_value_heads": 1,
                "vocab_size": 12,
                "max_position_embeddings": 32,
                "rms_norm_eps": 1e-6,
                "sliding_window": 4,
                "layer_types": ["sliding_attention", "full_attention", "full_attention"],
                "attention_bias": false,
                "attention_dropout": 0.0,
                "attention_k_eq_v": true,
                "num_kv_shared_layers": 0,
                "enable_moe_block": true,
                "num_experts": 4,
                "top_k_experts": 2,
                "moe_intermediate_size": 3,
                "use_double_wide_mlp": false,
                "hidden_size_per_layer_input": 0,
                "hidden_activation": "gelu_pytorch_tanh",
                "final_logit_softcapping": 30.0,
                "use_bidirectional_attention": "vision",
                "tie_word_embeddings": true,
                "rope_parameters": {
                    "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                    "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1000000.0}
                }
            }
        })
    }

    fn with_fields(mut value: Value, fields: Value) -> Value {
        {
            let object = value
                .as_object_mut()
                .expect("gemma4 test config must be an object");
            for (key, field) in fields
                .as_object()
                .expect("gemma4 test overrides must be an object")
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
                .expect("gemma4 test config must be an object");
            for field in fields {
                object.remove(*field);
            }
        }
        value
    }

    fn text_config_mut(value: &mut Value) -> &mut serde_json::Map<String, Value> {
        value["text_config"]
            .as_object_mut()
            .expect("gemma4 fixture text_config")
    }

    fn with_text_fields(mut value: Value, fields: Value) -> Value {
        for (key, field) in fields
            .as_object()
            .expect("gemma4 text overrides must be an object")
        {
            text_config_mut(&mut value).insert(key.clone(), field.clone());
        }
        value
    }

    fn without_text_fields(mut value: Value, fields: &[&str]) -> Value {
        for field in fields {
            text_config_mut(&mut value).remove(*field);
        }
        value
    }

    fn config_from(value: Value) -> Gemma4Config {
        Gemma4Config::from_raw(
            serde_json::from_value(value).expect("raw gemma4 config"),
            Gemma4Contract::RuneOptionTokens,
        )
        .expect("valid gemma4 config")
    }

    fn shapes(config: &Gemma4Config, root: &str) -> Vec<(String, Vec<usize>)> {
        let h = config.hidden_size;
        let mut out = vec![
            (
                format!("{root}embed_tokens.weight"),
                vec![config.vocab_size, h],
            ),
            (format!("{root}norm.weight"), vec![h]),
        ];
        if config.hidden_size_per_layer_input > 0 {
            let width = config.num_hidden_layers * config.hidden_size_per_layer_input;
            out.extend([
                (
                    format!("{root}embed_tokens_per_layer.weight"),
                    vec![config.vocab_size_per_layer_input, width],
                ),
                (
                    format!("{root}per_layer_model_projection.weight"),
                    vec![width, h],
                ),
                (
                    format!("{root}per_layer_projection_norm.weight"),
                    vec![config.hidden_size_per_layer_input],
                ),
            ]);
        }
        let plan = Gemma4Plan::from_config(config).unwrap();
        for layer in &plan.layers {
            let p = format!("{root}layers.{}", layer.index);
            out.extend([
                (format!("{p}.input_layernorm.weight"), vec![h]),
                (format!("{p}.post_attention_layernorm.weight"), vec![h]),
                (format!("{p}.pre_feedforward_layernorm.weight"), vec![h]),
                (format!("{p}.post_feedforward_layernorm.weight"), vec![h]),
                (format!("{p}.layer_scalar"), vec![1]),
                (
                    format!("{p}.self_attn.q_proj.weight"),
                    vec![layer.q_width, h],
                ),
                (format!("{p}.self_attn.q_norm.weight"), vec![layer.head_dim]),
                (
                    format!("{p}.self_attn.o_proj.weight"),
                    vec![h, layer.q_width],
                ),
                (
                    format!("{p}.mlp.gate_proj.weight"),
                    vec![layer.dense_intermediate_size, h],
                ),
                (
                    format!("{p}.mlp.up_proj.weight"),
                    vec![layer.dense_intermediate_size, h],
                ),
                (
                    format!("{p}.mlp.down_proj.weight"),
                    vec![h, layer.dense_intermediate_size],
                ),
            ]);
            match layer.kv {
                Gemma4KvPlan::Producer { .. } => {
                    out.push((
                        format!("{p}.self_attn.k_proj.weight"),
                        vec![layer.kv_width, h],
                    ));
                    out.push((format!("{p}.self_attn.k_norm.weight"), vec![layer.head_dim]));
                    if !layer.k_equals_v {
                        out.push((
                            format!("{p}.self_attn.v_proj.weight"),
                            vec![layer.kv_width, h],
                        ));
                    }
                }
                Gemma4KvPlan::Shared { .. } => {}
            }
            if let Some(ple) = layer.ple {
                out.extend([
                    (
                        format!("{p}.per_layer_input_gate.weight"),
                        vec![ple.width, h],
                    ),
                    (
                        format!("{p}.per_layer_projection.weight"),
                        vec![h, ple.width],
                    ),
                    (format!("{p}.post_per_layer_input_norm.weight"), vec![h]),
                ]);
            }
            if let Some(moe) = layer.moe {
                out.extend([
                    (format!("{p}.router.scale"), vec![h]),
                    (format!("{p}.router.proj.weight"), vec![moe.num_experts, h]),
                    (
                        format!("{p}.router.per_expert_scale"),
                        vec![moe.num_experts],
                    ),
                    (
                        format!("{p}.experts.gate_up_proj"),
                        vec![moe.num_experts, moe.intermediate_size * 2, h],
                    ),
                    (
                        format!("{p}.experts.down_proj"),
                        vec![moe.num_experts, h, moe.intermediate_size],
                    ),
                    (format!("{p}.pre_feedforward_layernorm_2.weight"), vec![h]),
                    (format!("{p}.post_feedforward_layernorm_1.weight"), vec![h]),
                    (format!("{p}.post_feedforward_layernorm_2.weight"), vec![h]),
                ]);
            }
        }
        out
    }

    #[test]
    fn loads_dense_ple_with_shared_kv_and_double_wide_mlp() {
        let fixture = Fixture::new();
        let config = config_from(gemma4_dense_config());
        let plan = Gemma4Plan::from_config(&config).unwrap();
        assert_eq!(plan.variant, Gemma4Variant::Dense);
        assert!(plan.causal);
        assert_eq!(plan.kv_producers, [Some(1), Some(2)]);
        assert_eq!(plan.layers[0].kind, Gemma4LayerKind::Sliding);
        assert_eq!(plan.layers[0].head_dim, 4);
        assert_eq!(plan.layers[0].q_width, 8);
        assert_eq!(plan.layers[0].kv_width, 4);
        assert_eq!(plan.layers[0].rope.rotated_pairs, 2);
        assert_eq!(plan.layers[1].kind, Gemma4LayerKind::Full);
        assert_eq!(plan.layers[1].head_dim, 8);
        assert_eq!(plan.layers[1].q_width, 16);
        assert_eq!(plan.layers[1].kv_width, 8);
        assert_eq!(plan.layers[1].rope.rotated_pairs, 1);
        assert_eq!(plan.layers[1].rope.frequency_pairs, 4);
        assert_eq!(plan.layers[3].kv, Gemma4KvPlan::Shared { producer: 1 });
        assert_eq!(plan.layers[3].dense_intermediate_size, 24);
        assert_eq!(plan.layers[2].dense_intermediate_size, 12);
        assert_eq!(plan.embedding_scale, (8.0f64).sqrt() as f32);
        assert_eq!(plan.norm_eps, 1e-6f32);
        assert_eq!(plan.norm_arithmetic, Gemma4NormArithmetic::DivSqrt);
        assert_eq!(plan.activation, Gemma4Activation::GeluPytorchTanh);
        assert!(plan.layers[0].qk_norm);
        assert_eq!(plan.layers[0].value_norm, Gemma4ValueNorm::UnitRms);
        assert_eq!(plan.layers[0].attention_scale, 1.0);
        assert_eq!(plan.ple.unwrap().embed_scale, (2.0f64).sqrt() as f32);
        assert_eq!(
            plan.ple.unwrap().model_projection_scale,
            (8.0f64).sqrt().recip() as f32
        );
        assert_eq!(plan.readout.final_logit_softcap, Some(30.0));

        fixture.write_entries(&shapes(&config, "model.language_model."));
        let weights = Gemma4Weights::load_rune(&fixture.0).unwrap();
        assert_eq!(weights.dtype, Dtype::BF16);
        assert_eq!(weights.layers.len(), 4);
        assert_eq!(weights.embed.shape(), &[12, 8]);
        assert_eq!(weights.ple.as_ref().unwrap().embed_tokens.shape(), &[12, 8]);
        assert_eq!(
            weights.plan.layers[0].kv,
            Gemma4KvPlan::Producer {
                stores_full_length: false
            }
        );
        assert_eq!(
            weights.plan.layers[2].kv,
            Gemma4KvPlan::Producer {
                stores_full_length: true
            }
        );
        assert_eq!(
            weights.plan.layers[3].kv,
            Gemma4KvPlan::Shared { producer: 1 }
        );
        assert!(matches!(
            weights.layers[3].kv,
            Gemma4KvWeights::Shared(Gemma4KvIgnoredWeights {
                k: None,
                v: None,
                k_norm: None
            })
        ));
        assert_eq!(
            weights.lm_head_plan,
            Gemma4LmHeadPlan::TiedEmbedding {
                ignored_explicit: false
            }
        );
        let row = weights.lm_head_row(5).unwrap();
        assert_eq!(row.row(), 5);
        assert_eq!(row.width(), 8);
        assert_eq!(row.offset_elements(), 40);
        assert_eq!(row.source_shape(), &[12, 8]);
    }

    #[test]
    fn tied_kv_without_global_head_count_uses_native_fallback() {
        let value = with_text_fields(
            gemma4_dense_config(),
            serde_json::json!({
                "attention_k_eq_v": true,
                "num_global_key_value_heads": null
            }),
        );
        let config = config_from(value);
        let plan = Gemma4Plan::from_config(&config).unwrap();
        assert!(plan.layers[1].k_equals_v);
        assert_eq!(plan.layers[1].num_kv_heads, config.num_key_value_heads);
        assert_eq!(
            plan.layers[1].kv_width,
            config.num_key_value_heads * config.global_head_dim
        );
    }

    #[test]
    fn loads_sparse_moe_with_tied_kv_and_explicit_lm_head_rows() {
        let fixture = Fixture::new();
        fixture.raw_config(gemma4_moe_config());
        let config = config_from(gemma4_moe_config());
        let plan = Gemma4Plan::from_config(&config).unwrap();
        assert_eq!(plan.variant, Gemma4Variant::SparseMoe);
        assert_eq!(plan.layers[1].kind, Gemma4LayerKind::Full);
        assert!(!plan.layers[0].k_equals_v);
        assert!(plan.layers[1].k_equals_v);
        assert_eq!(plan.layers[1].kv_width, 8);
        assert_eq!(plan.layers[1].moe.unwrap().num_experts, 4);
        assert_eq!(plan.router_root_scale, Some((8.0f64).sqrt().recip() as f32));
        let entries = shapes(&config, "model.language_model.");
        fixture.write_entries(&entries);
        let weights = Gemma4Weights::load_rune(&fixture.0).unwrap();
        assert_eq!(weights.layers.len(), 3);
        assert!(matches!(
            weights.layers[1].kv,
            Gemma4KvWeights::Producer(Gemma4KvProducerWeights { v: None, .. })
        ));
        assert!(weights.layers[0].moe.is_some());
        assert_eq!(weights.lm_head_row(11).unwrap().offset_elements(), 88);
        let layout = weights.runtime_layout(6, 2, 6, 4).unwrap();
        assert_eq!(layout.tokens, 6);
        assert_eq!(layout.moe_router_elements, 24);
        assert!(layout.estimated_bytes > 0);
        assert!(
            crate::gemma4_e2b_model::Gemma4E2bMath::admit(&weights).is_ok(),
            "Native sparse MoE text geometry must reach the executor admission"
        );
    }

    #[test]
    fn lm_head_rows_read_borrowed_embedding_spans() {
        let fixture = Fixture::new();
        let config = config_from(gemma4_dense_config());
        fixture.write_entries_with_indexed_tensor(
            &shapes(&config, "model.language_model."),
            "embed_tokens.weight",
        );
        let weights = Gemma4Weights::load_rune(&fixture.0).unwrap();
        let span = weights.lm_head_row(5).unwrap();
        assert_eq!(span.offset_elements(), 40);
        let mut row = vec![0u16; config.hidden_size];
        span.read_bf16(&mut row).unwrap();
        assert_eq!(row, (40u16..48).collect::<Vec<_>>());
        assert!(span.read_bf16(&mut [0u16; 1]).is_err());
    }

    #[test]
    fn borrowed_shadows_are_validated_but_never_promoted() {
        let fixture = Fixture::new();
        let config = config_from(gemma4_dense_config());
        let mut entries = shapes(&config, "model.language_model.");
        let p = "model.language_model.layers.3.self_attn";
        entries.extend([
            (format!("{p}.k_proj.weight"), vec![8, 8]),
            (format!("{p}.v_proj.weight"), vec![8, 8]),
            (format!("{p}.k_norm.weight"), vec![8]),
        ]);
        fixture.write_entries(&entries);
        let weights = Gemma4Weights::load_rune(&fixture.0).unwrap();
        match &weights.layers[3].kv {
            Gemma4KvWeights::Shared(shared) => {
                assert!(shared.k.is_some());
                assert!(shared.v.is_some());
                assert!(shared.k_norm.is_some());
            }
            _ => panic!("shared layer produced active KV weights"),
        }
        assert_eq!(
            weights.plan.layers[3].kv,
            Gemma4KvPlan::Shared { producer: 1 }
        );
    }

    #[test]
    fn explicit_lm_head_is_used_only_when_untied() {
        let fixture = Fixture::new();
        let mut config_value = gemma4_dense_config();
        text_config_mut(&mut config_value).insert("tie_word_embeddings".into(), false.into());
        config_value
            .as_object_mut()
            .unwrap()
            .insert("tie_word_embeddings".into(), false.into());
        fixture.raw_config(config_value.clone());
        let config = config_from(config_value);
        let mut entries = shapes(&config, "model.language_model.");
        entries.push(("lm_head.weight".into(), vec![12, 8]));
        fixture.write_entries(&entries);
        let weights = Gemma4Weights::load_rune(&fixture.0).unwrap();
        assert_eq!(weights.lm_head_plan, Gemma4LmHeadPlan::Explicit);
        assert_eq!(weights.lm_head_row(3).unwrap().offset_elements(), 24);
    }

    #[test]
    fn tied_lm_head_can_record_an_ignored_explicit_tensor() {
        let fixture = Fixture::new();
        let config = config_from(gemma4_dense_config());
        let mut entries = shapes(&config, "model.language_model.");
        entries.push(("lm_head.weight".into(), vec![12, 8]));
        fixture.write_entries(&entries);
        let weights = Gemma4Weights::load_rune(&fixture.0).unwrap();
        assert_eq!(
            weights.lm_head_plan,
            Gemma4LmHeadPlan::TiedEmbedding {
                ignored_explicit: true
            }
        );
    }

    #[test]
    fn rejects_missing_and_wrong_text_tensors() {
        let fixture = Fixture::new();
        let config = config_from(gemma4_dense_config());
        let mut entries = shapes(&config, "model.language_model.");
        entries.retain(|(name, _)| !name.ends_with(".layers.1.self_attn.k_norm.weight"));
        fixture.write_entries(&entries);
        assert!(fixture
            .error()
            .contains("missing weight model.language_model.layers.1.self_attn.k_norm.weight"));

        let fixture = Fixture::new();
        let mut entries = shapes(&config, "model.language_model.");
        for (name, shape) in &mut entries {
            if name.ends_with(".layers.2.mlp.up_proj.weight") {
                *shape = vec![13, 8];
            }
        }
        fixture.write_entries(&entries);
        assert!(fixture.error().contains("wrong shape"));

        let fixture = Fixture::new();
        let mut entries = shapes(&config, "model.language_model.");
        entries.retain(|(name, _)| !name.ends_with("per_layer_projection_norm.weight"));
        fixture.write_entries(&entries);
        assert!(fixture.error().contains("per_layer_projection_norm.weight"));
    }

    #[test]
    fn rejects_moe_geometry_and_presence_mismatches() {
        let config = config_from(gemma4_moe_config());

        let fixture = Fixture::new();
        fixture.raw_config(gemma4_moe_config());
        let mut entries = shapes(&config, "model.language_model.");
        entries.retain(|(name, _)| !name.ends_with("experts.gate_up_proj"));
        fixture.write_entries(&entries);
        assert!(fixture.error().contains("experts.gate_up_proj"));

        let fixture = Fixture::new();
        fixture.raw_config(gemma4_moe_config());
        let mut entries = shapes(&config, "model.language_model.");
        for (name, shape) in &mut entries {
            if name.ends_with("experts.gate_up_proj") {
                *shape = vec![4, 5, 8];
            }
        }
        fixture.write_entries(&entries);
        assert!(fixture.error().contains("wrong shape"));

        let dense = config_from(gemma4_dense_config());
        let fixture = Fixture::new();
        fixture.raw_config(gemma4_dense_config());
        let mut entries = shapes(&dense, "model.language_model.");
        entries.push((
            "model.language_model.layers.0.router.proj.weight".into(),
            vec![4, 8],
        ));
        fixture.write_entries(&entries);
        assert!(fixture
            .error()
            .contains("MoE tensor router.proj.weight without enable_moe_block"));
    }

    #[test]
    fn rejects_shared_kv_without_a_producer_and_tied_v_mismatch() {
        let fixture = Fixture::new();
        let value = with_text_fields(
            gemma4_dense_config(),
            serde_json::json!({"num_kv_shared_layers": 4}),
        );
        fixture.raw_config(value);
        assert!(fixture
            .config_error()
            .contains("shared KV layers need at least one producer"));

        let value = with_text_fields(
            gemma4_dense_config(),
            serde_json::json!({"layer_types": ["sliding_attention", "sliding_attention", "full_attention", "full_attention"]}),
        );
        let config = config_from(value);
        let plan = Gemma4Plan::from_config(&config).unwrap();
        assert_eq!(plan.kv_producers, [Some(2), Some(1)]);

        let value = with_text_fields(
            gemma4_dense_config(),
            serde_json::json!({
                "layer_types": ["full_attention", "full_attention", "full_attention", "full_attention"],
                "num_kv_shared_layers": 0,
                "use_double_wide_mlp": false,
                "attention_k_eq_v": true,
                "num_global_key_value_heads": 1
            }),
        );
        let config = config_from(value.clone());
        let mut entries = shapes(&config, "model.language_model.");
        entries.push((
            "model.language_model.layers.0.self_attn.v_proj.weight".into(),
            vec![8, 8],
        ));
        let fixture = Fixture::new();
        fixture.raw_config(value);
        fixture.write_entries(&entries);
        assert!(fixture.error().contains("ties K/V but declares v_proj"));
    }

    #[test]
    fn rejects_configuration_contradictions_and_unsupported_features() {
        for (overrides, needle) in [
            (
                serde_json::json!({"hidden_size": 0}),
                "hidden_size must be nonzero",
            ),
            (
                serde_json::json!({"num_hidden_layers": 0}),
                "num_hidden_layers must be nonzero",
            ),
            (serde_json::json!({"head_dim": 3}), "positive and even"),
            (
                serde_json::json!({"global_head_dim": 5}),
                "positive and even",
            ),
            (
                serde_json::json!({"num_attention_heads": 3, "num_key_value_heads": 2}),
                "divide evenly",
            ),
            (
                serde_json::json!({"layer_types": ["full_attention"]}),
                "layer_types has 1 entries",
            ),
            (
                serde_json::json!({"layer_types": ["full_attention", "local", "full_attention", "full_attention"]}),
                "unsupported Gemma4 attention type",
            ),
            (
                serde_json::json!({"attention_bias": true}),
                "attention_bias",
            ),
            (
                serde_json::json!({"attention_dropout": 0.1}),
                "attention_dropout",
            ),
            (
                serde_json::json!({"num_global_key_value_heads": 1}),
                "requires attention_k_eq_v",
            ),
            (
                serde_json::json!({"enable_moe_block": true}),
                "requires nonzero num_experts",
            ),
            (
                serde_json::json!({"num_experts": 4}),
                "expert routing fields require",
            ),
            (
                serde_json::json!({"use_bidirectional_attention": "all"}),
                "use_bidirectional_attention=all",
            ),
            (
                serde_json::json!({"use_bidirectional_attention": false}),
                "use_bidirectional_attention",
            ),
            (
                serde_json::json!({"hidden_activation": "silu"}),
                "unsupported Gemma4 text activation",
            ),
            (
                serde_json::json!({"quantization_config": {"quant_method": "gptq"}}),
                "quantized",
            ),
            (
                serde_json::json!({"rope_scaling": {"type": "linear"}}),
                "rope_scaling",
            ),
            (
                serde_json::json!({"per_layer_config": {"0": {"head_dim": 4}}}),
                "per_layer_config",
            ),
            (
                serde_json::json!({"partial_rotary_factor": 0.5}),
                "top-level partial_rotary_factor",
            ),
            (
                serde_json::json!({"final_logit_softcapping": 0.0}),
                "final_logit_softcapping",
            ),
            (
                serde_json::json!({"attn_logit_softcapping": 30.0}),
                "attn_logit_softcapping",
            ),
            (
                serde_json::json!({"vocab_size_per_layer_input": 0}),
                "vocab_size_per_layer_input must be nonzero when PLE",
            ),
            (serde_json::json!({"sliding_window": 0}), "sliding_window"),
            (
                serde_json::json!({"max_position_embeddings": (u32::MAX as u64) + 1}),
                "fit CUDA position IDs",
            ),
            (
                serde_json::json!({"architectures": ["Gemma4ForSequenceClassification"]}),
                "unsupported Gemma4 text architecture",
            ),
            (
                serde_json::json!({"model_type": "gemma3"}),
                "unsupported nested text model_type",
            ),
        ] {
            let fixture = Fixture::new();
            let mut value = with_text_fields(gemma4_dense_config(), overrides.clone());
            if overrides.get("architectures").is_some() || overrides.get("model_type").is_some() {
                value = with_fields(value, overrides.clone());
            }
            fixture.raw_config(value);
            let error = fixture.config_error();
            assert!(error.contains(needle), "{overrides}: {error}");
        }
    }

    #[test]
    fn rejects_rope_variants_and_overflowed_geometry() {
        for (rope, needle) in [
            (
                serde_json::json!({"full_attention": {"rope_type": "yarn", "rope_theta": 10000.0}}),
                "unsupported gemma4 RoPE type",
            ),
            (
                serde_json::json!({"full_attention": {"rope_type": "linear", "factor": 8.0, "rope_theta": 10000.0}}),
                "factor is unsupported",
            ),
            (
                serde_json::json!({"full_attention": {"rope_type": "default", "partial_rotary_factor": 0.25, "rope_theta": 10000.0}}),
                "default RoPE requires partial_rotary_factor 1",
            ),
            (
                serde_json::json!({"full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.0, "rope_theta": 10000.0}}),
                "partial_rotary_factor",
            ),
            (
                serde_json::json!({"full_attention": {"rope_type": "proportional", "rope_theta": 0.0}}),
                "rope_theta",
            ),
            (
                serde_json::json!({"full_attention": {"rope_type": "proportional", "rope_theta": 10000.0, "mrope_section": [1]}}),
                "unsupported gemma4 rope_parameters",
            ),
        ] {
            let fixture = Fixture::new();
            fixture.raw_config(with_text_fields(
                gemma4_dense_config(),
                serde_json::json!({"rope_parameters": rope}),
            ));
            let error = fixture.config_error();
            assert!(error.contains(needle), "{error}");
        }

        let fixture = Fixture::new();
        fixture.raw_config(with_text_fields(
            gemma4_dense_config(),
            serde_json::json!({
                "hidden_size_per_layer_input": usize::MAX,
                "num_hidden_layers": 2,
                "layer_types": ["full_attention", "full_attention"]
            }),
        ));
        let error = fixture.config_error();
        assert!(error.contains("overflow"), "{error}");

        let fixture = Fixture::new();
        fixture.raw_config(with_text_fields(
            gemma4_dense_config(),
            serde_json::json!({"num_hidden_layers": MAX_LAYER_COUNT + 1}),
        ));
        let error = fixture.config_error();
        assert!(error.contains("exceeds supported bound"), "{error}");

        let fixture = Fixture::new();
        fixture.raw_config(with_text_fields(
            gemma4_moe_config(),
            serde_json::json!({"num_experts": usize::MAX}),
        ));
        let error = fixture.config_error();
        assert!(error.contains("overflow"), "{error}");

        let fixture = Fixture::new();
        fixture.raw_config(gemma4_dense_config());
        let config = config_from(gemma4_dense_config());
        fixture.write_entries(&shapes(&config, "model.language_model."));
        let weights = Gemma4Weights::load_rune(&fixture.0).unwrap();
        assert!(weights
            .runtime_layout(usize::MAX, 1, 1, 1)
            .unwrap_err()
            .to_string()
            .contains("packed sequence capacity"));
        assert!(checked_product(usize::MAX, 2, "test bytes")
            .unwrap_err()
            .to_string()
            .contains("overflow"));
        assert!(weights
            .runtime_layout(1, 2, 1, 1)
            .unwrap_err()
            .to_string()
            .contains("positive batch"));
        assert!(weights
            .runtime_layout(1, 1, 33, 1)
            .unwrap_err()
            .to_string()
            .contains("max_position_embeddings"));
        assert!(weights
            .runtime_layout(10, 2, 4, 1)
            .unwrap_err()
            .to_string()
            .contains("packed sequence capacity"));
    }

    #[test]
    fn rejects_namespaces_and_non_rune_decision_files() {
        let fixture = Fixture::new();
        let config = config_from(gemma4_dense_config());
        fixture.write_entries(&shapes(&config, "language_model."));
        assert_eq!(
            Gemma4Weights::load_rune(&fixture.0).unwrap().layers.len(),
            4
        );

        let fixture = Fixture::new();
        let config = config_from(gemma4_dense_config());
        let mut ambiguous = shapes(&config, "model.language_model.");
        ambiguous.push((
            "embed_tokens.weight".into(),
            vec![config.vocab_size, config.hidden_size],
        ));
        fixture.write_entries(&ambiguous);
        assert!(fixture
            .error()
            .contains("ambiguous gemma4 checkpoint namespace"));

        let fixture = Fixture::new();
        let config = config_from(gemma4_dense_config());
        let mut mixed = shapes(&config, "model.language_model.");
        mixed.push((
            "model.layers.0.input_layernorm.weight".into(),
            vec![config.hidden_size],
        ));
        fixture.write_entries(&mixed);
        assert!(fixture.error().contains("mixed gemma4 checkpoint roots"));

        let fixture = Fixture::new();
        let config = config_from(gemma4_dense_config());
        fixture.write_entries(&shapes(&config, "transformer."));
        assert!(fixture
            .error()
            .contains("unsupported gemma4 checkpoint namespace"));

        let fixture = Fixture::new();
        let config = config_from(gemma4_dense_config());
        fixture.write_entries(&shapes(&config, "model.language_model."));
        fs::write(fixture.0.join("decision_config.json"), b"{}").unwrap();
        assert!(fixture.error().contains("does not use decision file"));

        let fixture = Fixture::new();
        let config = config_from(gemma4_dense_config());
        fixture.write_entries(&shapes(&config, "model.language_model."));
        let mut value = gemma4_dense_config();
        text_config_mut(&mut value).insert("tie_word_embeddings".into(), false.into());
        value
            .as_object_mut()
            .unwrap()
            .insert("tie_word_embeddings".into(), false.into());
        fixture.raw_config(value);
        assert!(fixture.error().contains("requires lm_head.weight"));
    }

    #[test]
    fn rejects_projection_bias_and_invalid_lm_head_rows() {
        let fixture = Fixture::new();
        let config = config_from(gemma4_dense_config());
        let mut entries = shapes(&config, "model.language_model.");
        entries.push((
            "model.language_model.layers.0.mlp.down_proj.bias".into(),
            vec![config.hidden_size],
        ));
        fixture.write_entries(&entries);
        assert!(fixture.error().contains("unsupported projection bias"));

        let fixture = Fixture::new();
        let config = config_from(gemma4_dense_config());
        fixture.write_entries(&shapes(&config, "model.language_model."));
        let weights = Gemma4Weights::load_rune(&fixture.0).unwrap();
        assert!(weights.lm_head_row(12).is_err());
    }

    #[test]
    fn uses_native_defaults_for_missing_optional_fields() {
        let value = without_text_fields(
            gemma4_dense_config(),
            &[
                "head_dim",
                "global_head_dim",
                "max_position_embeddings",
                "rms_norm_eps",
                "sliding_window",
                "layer_types",
                "num_kv_shared_layers",
                "hidden_size_per_layer_input",
                "hidden_activation",
                "final_logit_softcapping",
                "rope_parameters",
            ],
        );
        let value = with_text_fields(
            value,
            serde_json::json!({
                "hidden_size": 8,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "num_hidden_layers": 2,
                "intermediate_size": 12,
                "use_double_wide_mlp": false
            }),
        );
        let config = config_from(value);
        assert_eq!(config.head_dim, 4);
        assert_eq!(config.global_head_dim, 4);
        assert_eq!(config.max_position_embeddings, 8192);
        assert_eq!(config.sliding_window, 4096);
        assert_eq!(config.hidden_size_per_layer_input, 0);
        assert_eq!(config.sliding_rope.theta, 10000.0);
        assert_eq!(config.full_rope.theta, 10000.0);
        assert_eq!(config.full_rope.partial_rotary_factor, 1.0);
        assert_eq!(config.layer_types, vec![Gemma4LayerKind::Full; 2]);
    }

    #[test]
    fn actual_rune_26b_and_e4b_configs_admit_native_geometry() {
        let rune = std::env::var_os("INFINITY_GEMMA4_RUNE_CONFIG")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from(
                    "/tmp/oxide-docs-parity-20261102/model-metadata/michaelfeil--rune-26b-a4b",
                )
            });
        if rune.join("config.json").is_file() {
            let config = Gemma4Config::load(&rune, Gemma4Contract::RuneOptionTokens)
                .expect("Rune 26B config");
            assert_eq!(config.hidden_size, 2816);
            assert_eq!(config.num_hidden_layers, 30);
            assert_eq!(config.num_attention_heads, 16);
            assert_eq!(config.num_key_value_heads, 8);
            assert_eq!(config.num_global_key_value_heads, 2);
            assert_eq!(config.head_dim, 256);
            assert_eq!(config.global_head_dim, 512);
            assert_eq!(config.hidden_size_per_layer_input, 0);
            assert!(config.attention_k_eq_v);
            assert!(config.enable_moe_block);
            assert_eq!(config.num_experts, Some(128));
            assert_eq!(config.top_k_experts, Some(8));
            assert_eq!(config.moe_intermediate_size, Some(704));
            let plan = Gemma4Plan::from_config(&config).expect("Rune 26B plan");
            assert_eq!(plan.variant, Gemma4Variant::SparseMoe);
            assert!(plan.layers.iter().all(|layer| layer.moe.is_some()));
            assert_eq!(plan.kv_producers, [None, None]);
        }
        let e4b = std::env::var_os("INFINITY_GEMMA4_E4B_CONFIG")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from(
                    "/workspace/model-performance/michaelfeil/onejev-perf/backbone-comparison-20261023/models/gemma-4-E4B",
                )
            });
        if e4b.join("config.json").is_file() {
            let config = Gemma4Config::load(&e4b, Gemma4Contract::Embedding).expect("E4B config");
            assert_eq!(config.hidden_size, 2560);
            assert_eq!(config.num_hidden_layers, 42);
            assert_eq!(config.num_attention_heads, 8);
            assert_eq!(config.num_key_value_heads, 2);
            assert_eq!(config.hidden_size_per_layer_input, 256);
            assert_eq!(config.num_kv_shared_layers, 18);
            let plan = Gemma4Plan::from_config(&config).expect("E4B plan");
            assert_eq!(plan.variant, Gemma4Variant::Dense);
            assert!(plan.kv_producers.iter().any(Option::is_some));
        }
    }

    #[test]
    #[ignore = "CPU-only: actual local Gemma4 BF16 checkpoint headers; set INFINITY_OXIDE_TEST_GEMMA4_*"]
    fn actual_gemma4_checkpoint_headers() {
        for (variable, hidden, layers, experts, ple, shared, producers, double_wide) in [
            (
                "INFINITY_OXIDE_TEST_GEMMA4_E2B",
                1536,
                35,
                None,
                256,
                20,
                [Some(14), Some(13)],
                true,
            ),
            (
                "INFINITY_OXIDE_TEST_GEMMA4_E4B",
                2560,
                42,
                None,
                256,
                18,
                [Some(23), Some(22)],
                false,
            ),
            (
                "INFINITY_OXIDE_TEST_GEMMA4_26B",
                2816,
                30,
                Some(128),
                0,
                0,
                [None, None],
                false,
            ),
        ] {
            let Some(path) = std::env::var_os(variable) else {
                continue;
            };
            let weights = Gemma4Weights::load_rune(Path::new(&path)).unwrap();
            assert_eq!(weights.config.hidden_size, hidden);
            assert_eq!(weights.config.num_hidden_layers, layers);
            assert_eq!(weights.config.num_experts, experts);
            assert_eq!(weights.config.hidden_size_per_layer_input, ple);
            assert_eq!(weights.config.num_kv_shared_layers, shared);
            assert_eq!(weights.plan.kv_producers, producers);
            assert_eq!(weights.layers.len(), layers);
            assert_eq!(
                weights.plan.variant,
                if experts.is_some() {
                    Gemma4Variant::SparseMoe
                } else {
                    Gemma4Variant::Dense
                }
            );
            assert!(weights
                .plan
                .layers
                .iter()
                .any(|layer| { layer.kind == Gemma4LayerKind::Full && layer.head_dim == 512 }));
            assert!(weights
                .plan
                .layers
                .iter()
                .any(|layer| { layer.kind == Gemma4LayerKind::Sliding && layer.head_dim == 256 }));
            if double_wide {
                assert!(weights.plan.layers.iter().any(|layer| {
                    layer.dense_intermediate_size == 2 * weights.config.intermediate_size
                }));
            }
            let row = weights.lm_head_row(262143).unwrap();
            assert_eq!(row.width(), hidden);
            let _ = weights.runtime_layout(128, 1, 128, 26).unwrap();
        }
    }
}
