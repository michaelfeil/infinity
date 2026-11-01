// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Staging typed loader for dense Qwen2/Llama pooled embedding backbones.
//!
//! This module validates configuration, sentence-transformers task modules and
//! checkpoint metadata. It deliberately does not select kernels, upload device
//! weights, build rotary tables or claim engine support. Executor integration is
//! a separate step and owns those contracts.

use crate::model::{elements, path_present, read_json, Checkpoint, WeightSource};
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
/// Dense Qwen3 per-head Q/K RMSNorm width. Unported variants with a different
/// head dimension are rejected by the executor after metadata validation.
pub(crate) const QWEN3_QK_NORM_WIDTH: usize = 128;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DecoderFamily {
    Qwen2,
    Qwen3,
    Llama,
}
impl DecoderFamily {
    pub(crate) fn from_model_type(model_type: &str) -> Option<Self> {
        match model_type {
            "qwen2" => Some(Self::Qwen2),
            "qwen3" => Some(Self::Qwen3),
            "llama" => Some(Self::Llama),
            _ => None,
        }
    }

    pub(crate) const fn requires_qk_norm(self) -> bool {
        matches!(self, Self::Qwen3)
    }

    const fn default_theta(self) -> f64 {
        match self {
            Self::Qwen2 => DEFAULT_QWEN2_THETA,
            Self::Qwen3 => DEFAULT_QWEN3_THETA,
            Self::Llama => DEFAULT_LLAMA_THETA,
        }
    }

    const fn default_rope(self) -> DecoderRope {
        match self {
            Self::Qwen2 | Self::Qwen3 => DecoderRope::FullNeoX,
            Self::Llama => DecoderRope::Default,
        }
    }

    fn accepts_architecture(self, architecture: &str) -> bool {
        match self {
            Self::Qwen2 => matches!(architecture, "Qwen2Model" | "Qwen2ForCausalLM"),
            Self::Qwen3 => matches!(architecture, "Qwen3Model" | "Qwen3ForCausalLM"),
            Self::Llama => matches!(architecture, "LlamaModel" | "LlamaForCausalLM"),
        }
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
    pub(crate) rope: DecoderRope,
}
impl DecoderConfig {
    pub(crate) fn load(path: &Path) -> Result<Self, BackendError> {
        let raw: RawDecoderConfig = read_json(&path.join("config.json"))?;
        Self::from_raw(raw)
    }

    pub(crate) const fn q_width(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    pub(crate) const fn kv_width(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    fn from_raw(raw: RawDecoderConfig) -> Result<Self, BackendError> {
        let family = DecoderFamily::from_model_type(&raw.model_type).ok_or_else(|| {
            start(format!(
                "unsupported model_type {}; expected dense qwen2, qwen3 or llama",
                raw.model_type
            ))
        })?;
        if let Some(architectures) = raw.architectures.as_deref() {
            for architecture in architectures {
                if !family.accepts_architecture(architecture) {
                    return Err(start(format!(
                        "unsupported decoder task/architecture {architecture}; only pooled embedding Model/ForCausalLM backbones are supported"
                    )));
                }
            }
        }
        reject_config_modifiers(&raw)?;

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
        let rope_theta = raw.rope_theta.unwrap_or(family.default_theta());
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
        let is_causal = match family {
            DecoderFamily::Qwen2 => raw.is_causal.unwrap_or(true),
            DecoderFamily::Qwen3 => raw.is_causal.unwrap_or(!bidirectional),
            DecoderFamily::Llama => raw.is_causal.unwrap_or(!bidirectional),
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
            rope,
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

fn reject_config_modifiers(raw: &RawDecoderConfig) -> Result<(), BackendError> {
    if raw.quantization_config.is_some() {
        return Err(start(
            "quantized decoder checkpoints are not supported by this loader",
        ));
    }
    if raw
        .rope_scaling
        .as_ref()
        .is_some_and(|value| !value.is_null())
    {
        return Err(start(
            "scaled decoder RoPE (including Llama3) is not supported yet",
        ));
    }
    if raw
        .rope_parameters
        .as_ref()
        .is_some_and(|value| !value.is_null())
    {
        return Err(start("decoder rope_parameters are not supported yet"));
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
    if active_window {
        return Err(start(
            "sliding-window decoder attention is not supported yet",
        ));
    }
    for key in raw.extra.keys() {
        let key_lower = key.to_ascii_lowercase();
        let unsupported = key_lower.contains("expert")
            || key_lower.contains("quantiz")
            || key_lower.contains("vision")
            || key_lower.contains("audio")
            || key_lower.starts_with("image_")
            || key_lower.starts_with("mm_")
            || key_lower.starts_with("multi_modal")
            || key_lower.starts_with("linear_")
            || matches!(
                key_lower.as_str(),
                "layer_types" | "moe_intermediate_size" | "router_aux_loss_coef"
            );
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

#[derive(Clone, Debug)]
pub(crate) struct DecoderLayerWeights {
    pub(crate) input_norm: WeightSource,
    pub(crate) post_attention_norm: WeightSource,
    /// Dense Qwen3 per-head Q/K RMSNorm scales. Present only for Qwen3, always
    /// as a complete pair, and absent for Qwen2/Llama checkpoints.
    pub(crate) q_norm: Option<WeightSource>,
    pub(crate) k_norm: Option<WeightSource>,
    pub(crate) q: DecoderLinear,
    pub(crate) k: DecoderLinear,
    pub(crate) v: DecoderLinear,
    pub(crate) o: DecoderLinear,
    pub(crate) gate: DecoderLinear,
    pub(crate) up: DecoderLinear,
    pub(crate) down: DecoderLinear,
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
    let minimum = config
        .num_hidden_layers
        .checked_mul(9)
        .and_then(|count| count.checked_add(2))
        .ok_or_else(|| start("decoder checkpoint capacity overflow"))?;
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
fn decoder_tensor_shapes(config: &DecoderConfig) -> Vec<(String, Vec<usize>)> {
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
        if config.family == DecoderFamily::Qwen3 {
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
                    "decoder layer {layer} has {name}, but Q/K RMSNorm scales are only supported for dense Qwen3"
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
        validate_checkpoint_capacity(&config, &checkpoint)?;
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(config.num_hidden_layers)
            .map_err(start)?;
        let root = select_root(&checkpoint)?;
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
            let q_linear = fetch_linear(&checkpoint, &p, ".self_attn.q_proj", &[q, h], &[q])?;
            let k_linear = fetch_linear(&checkpoint, &p, ".self_attn.k_proj", &[kv, h], &[kv])?;
            let v_linear = fetch_linear(&checkpoint, &p, ".self_attn.v_proj", &[kv, h], &[kv])?;
            if q_linear.bias.is_some() != k_linear.bias.is_some()
                || q_linear.bias.is_some() != v_linear.bias.is_some()
            {
                return Err(start(format!(
                    "decoder layer {layer} Q/K/V projection biases must be all present or all absent"
                )));
            }
            let gate = fetch_linear(&checkpoint, &p, ".mlp.gate_proj", &[i, h], &[i])?;
            let up = fetch_linear(&checkpoint, &p, ".mlp.up_proj", &[i, h], &[i])?;
            if gate.bias.is_some() != up.bias.is_some() {
                return Err(start(format!(
                    "decoder layer {layer} gate/up projection biases must be both present or both absent"
                )));
            }
            let (q_norm, k_norm) = fetch_qk_norm(&checkpoint, layer, &p, config.family)?;
            layers.push(DecoderLayerWeights {
                input_norm: checkpoint.source(&format!("{p}.input_layernorm.weight"), &[h])?,
                post_attention_norm: checkpoint
                    .source(&format!("{p}.post_attention_layernorm.weight"), &[h])?,
                q_norm,
                k_norm,
                q: q_linear,
                k: k_linear,
                v: v_linear,
                o: fetch_linear(&checkpoint, &p, ".self_attn.o_proj", &[h, q], &[h])?,
                gate,
                up,
                down: fetch_linear(&checkpoint, &p, ".mlp.down_proj", &[h, i], &[h])?,
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

fn validate_decoder_modules(path: &Path) -> Result<(), BackendError> {
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
        assert_eq!(loaded.layers[0].gate.weight.shape(), &[16, 8]);
        assert_eq!(loaded.layers[0].up.weight.shape(), &[16, 8]);
        assert_eq!(loaded.layers[0].down.weight.shape(), &[8, 16]);
        assert_eq!(loaded.layers[0].q.bias.as_ref().unwrap().shape(), &[8]);
        assert_eq!(loaded.layers[0].k.bias.as_ref().unwrap().shape(), &[4]);
        assert_eq!(loaded.layers[0].v.bias.as_ref().unwrap().shape(), &[4]);
        assert!(loaded.layers[0].gate.bias.is_none());
        assert!(loaded.layers[0].o.bias.is_none());
        assert!(loaded.layers[0].down.bias.is_none());
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
        assert!(loaded.layers[0].gate.bias.is_none());
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
        assert!(loaded.layers[0].gate.bias.is_none());
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
            assert!(error.contains("only supported for dense Qwen3"), "{error}");
        }
    }

    #[test]
    fn qwen3_06_and_8b_geometry_metadata_needs_no_model_weights() {
        for (hidden, intermediate, layers, heads, kv_heads) in
            [(1024usize, 3072, 28, 16, 8), (4096, 12288, 36, 32, 8)]
        {
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
    fn rejects_qwen3_moe_quantized_and_unported_architectures() {
        for (overrides, needle) in [
            (
                serde_json::json!({"model_type": "qwen3_moe"}),
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
            DecoderFamily::Llama,
        ] {
            for bidirectional in [None, Some(false), Some(true)] {
                for causal in [None, Some(false), Some(true)] {
                    let mut value = match family {
                        DecoderFamily::Qwen2 => qwen2_config(),
                        DecoderFamily::Qwen3 => qwen3_config(),
                        DecoderFamily::Llama => llama_config(),
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
                        DecoderFamily::Qwen3 => !bidirectional.unwrap_or(false),
                        DecoderFamily::Llama => !bidirectional.unwrap_or(false),
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
        assert!(loaded.layers[0].down.bias.is_none());
        assert!(loaded.layers[1].o.bias.is_none());
        assert!(loaded.layers[1].down.bias.is_some());
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
                serde_json::json!({"rope_scaling": {"type": "llama3", "factor": 8.0}}),
                "scaled decoder RoPE",
            ),
            (
                serde_json::json!({"rope_parameters": {"rope_type": "yarn"}}),
                "rope_parameters",
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
        assert!(loaded.layers[0].gate.bias.is_none());
        assert!(loaded.layers[0].o.bias.is_none());
        assert_eq!(loaded.layers[27].down.weight.shape(), &[1536, 8960]);
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
}
