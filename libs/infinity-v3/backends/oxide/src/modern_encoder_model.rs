// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! ModernBERT checkpoint adapter.
//!
//! ModernBERT keeps a biasless prenorm encoder with packed QKV, a packed
//! up/gate projection and RoPE. This module validates the model metadata and
//! creates checked metadata views into the safetensors shards; it does not read
//! or materialize tensor payloads. The eventual engine owns uploads and math.

use crate::model::{
    read_json, BertConfig, Checkpoint, EncoderFamily, EncoderRecipe, FeedForwardRecipe,
    NormArithmetic, PositionRecipe, RotaryRecipe, WeightSource,
};
use crate::start;
use infinity_v3_backend_core::BackendError;
use serde::Deserialize;
use std::collections::BTreeSet;
use std::path::Path;

const MODERN_HIDDEN: usize = 768;
const MODERN_HEADS: usize = 12;
const MAX_LAYERS: usize = 256;
const MIN_POSITIONS: usize = 1;
const MAX_POSITIONS: usize = 8192;
const DEFAULT_NORM_EPS: f64 = 1e-12;
const DEFAULT_GLOBAL_EVERY: usize = 3;
const DEFAULT_GLOBAL_THETA: f64 = 160_000.0;
const DEFAULT_LOCAL_THETA: f64 = 10_000.0;
const REQUIRED_LOCAL_ATTENTION: usize = 128;
const WORD_SUFFIX: &str = "embeddings.tok_embeddings.weight";
const MLM_HEAD_DENSE: &str = "head.dense.weight";
const MLM_HEAD_NORM: &str = "head.norm.weight";
const MLM_DECODER_BIAS: &str = "decoder.bias";

#[derive(Debug, Deserialize)]
struct RawModernConfig {
    model_type: String,
    architectures: Option<Vec<String>>,
    hidden_size: Option<usize>,
    intermediate_size: Option<usize>,
    num_attention_heads: Option<usize>,
    num_hidden_layers: Option<usize>,
    max_position_embeddings: Option<usize>,
    vocab_size: Option<usize>,
    type_vocab_size: Option<usize>,
    norm_eps: Option<f64>,
    hidden_activation: Option<String>,
    attention_bias: Option<bool>,
    mlp_bias: Option<bool>,
    norm_bias: Option<bool>,
    global_attn_every_n_layers: Option<usize>,
    local_attention: Option<usize>,
    global_rope_theta: Option<f64>,
    local_rope_theta: Option<f64>,
    position_embedding_type: Option<String>,
    rope_scaling: Option<serde_json::Value>,
    quantization_config: Option<serde_json::Value>,
    is_decoder: Option<bool>,
    add_cross_attention: Option<bool>,
    torch_dtype: Option<String>,
    dtype: Option<String>,
}

fn required_usize(value: Option<usize>, name: &str) -> Result<usize, BackendError> {
    value.ok_or_else(|| start(format!("missing modernbert config field {name}")))
}

fn check_activation(value: &str, field: &str) -> Result<(), BackendError> {
    match value {
        // ModernBERT's default and the explicit PyTorch spelling both select
        // the tanh GELU approximation used by the native engine.
        "gelu" | "gelu_pytorch_tanh" | "gelu_new" | "gelu_fast" | "gelu_approx" => Ok(()),
        other => Err(start(format!(
            "unsupported modernbert {field} {other}; expected gelu or gelu_pytorch_tanh"
        ))),
    }
}

fn check_theta(value: f64, name: &str) -> Result<(), BackendError> {
    let narrow = value as f32;
    if !value.is_finite() || value <= 1.0 || !narrow.is_finite() || narrow <= 1.0 {
        return Err(start(format!(
            "modernbert {name} rope theta must be finite and greater than one"
        )));
    }
    Ok(())
}

fn require_u32_shape(shape: &[usize]) -> Result<(), BackendError> {
    let mut count = 1usize;
    for dimension in shape {
        count = count
            .checked_mul(*dimension)
            .ok_or_else(|| start("modernbert weight dimensions overflow"))?;
        if count > u32::MAX as usize {
            return Err(start(
                "modernbert weight shape exceeds the u32 element range",
            ));
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub(crate) struct ModernConfig {
    pub geometry: BertConfig,
    pub global_theta: f64,
    pub local_theta: f64,
    pub global_every: usize,
    pub local_radius: usize,
}

impl ModernConfig {
    pub(crate) fn load(path: &Path) -> Result<Self, BackendError> {
        let raw: RawModernConfig = read_json(&path.join("config.json"))?;
        Self::from_raw(raw)
    }

    fn from_raw(raw: RawModernConfig) -> Result<Self, BackendError> {
        if raw.model_type != "modernbert" {
            return Err(start(format!(
                "unsupported modernbert model_type {}; expected modernbert",
                raw.model_type
            )));
        }
        if let Some(architectures) = raw.architectures.as_ref() {
            for architecture in architectures {
                if !matches!(
                    architecture.as_str(),
                    "ModernBertModel" | "ModernBertForMaskedLM"
                ) {
                    return Err(start(format!(
                        "unsupported modernbert architecture {architecture}; expected ModernBertModel or ModernBertForMaskedLM"
                    )));
                }
            }
        }

        let hidden_size = required_usize(raw.hidden_size, "hidden_size")?;
        let intermediate_size = required_usize(raw.intermediate_size, "intermediate_size")?;
        let num_attention_heads = required_usize(raw.num_attention_heads, "num_attention_heads")?;
        let num_hidden_layers = required_usize(raw.num_hidden_layers, "num_hidden_layers")?;
        let max_position_embeddings =
            required_usize(raw.max_position_embeddings, "max_position_embeddings")?;
        let vocab_size = required_usize(raw.vocab_size, "vocab_size")?;

        if hidden_size != MODERN_HIDDEN {
            return Err(start(format!(
                "modernbert hidden_size must be {MODERN_HIDDEN}, got {hidden_size}"
            )));
        }
        if num_attention_heads != MODERN_HEADS {
            return Err(start(format!(
                "modernbert num_attention_heads must be {MODERN_HEADS}, got {num_attention_heads}"
            )));
        }
        if num_hidden_layers == 0 || num_hidden_layers > MAX_LAYERS {
            return Err(start(format!(
                "modernbert num_hidden_layers must be between 1 and {MAX_LAYERS}, got {num_hidden_layers}"
            )));
        }
        if intermediate_size == 0 || intermediate_size % 8 != 0 {
            return Err(start(format!(
                "modernbert intermediate_size must be a positive multiple of 8, got {intermediate_size}"
            )));
        }
        if vocab_size == 0 {
            return Err(start("modernbert vocab_size must be positive"));
        }
        if !(MIN_POSITIONS..=MAX_POSITIONS).contains(&max_position_embeddings) {
            return Err(start(format!(
                "modernbert max_position_embeddings must be between {MIN_POSITIONS} and {MAX_POSITIONS}, got {max_position_embeddings}"
            )));
        }

        let norm_eps = raw.norm_eps.unwrap_or(DEFAULT_NORM_EPS);
        if !norm_eps.is_finite()
            || norm_eps <= 0.0
            || !(norm_eps as f32).is_finite()
            || norm_eps as f32 <= 0.0
        {
            return Err(start(
                "modernbert norm_eps must be finite and positive in f32 and f64",
            ));
        }

        if let Some(value) = raw.hidden_activation.as_deref() {
            check_activation(value, "hidden_activation")?;
        }
        for (name, value) in [
            ("attention_bias", raw.attention_bias),
            ("mlp_bias", raw.mlp_bias),
            ("norm_bias", raw.norm_bias),
        ] {
            if value.unwrap_or(false) {
                return Err(start(format!(
                    "modernbert {name} must be false; backbone linear layers and norms are biasless"
                )));
            }
        }

        if let Some(type_vocab_size) = raw.type_vocab_size {
            if type_vocab_size != 0 {
                return Err(start(format!(
                    "modernbert type_vocab_size must be zero or absent, got {type_vocab_size}"
                )));
            }
        }

        let global_every = raw
            .global_attn_every_n_layers
            .unwrap_or(DEFAULT_GLOBAL_EVERY);
        if global_every == 0 {
            return Err(start(
                "modernbert global attention interval must be positive",
            ));
        }

        let local_attention = raw.local_attention.unwrap_or(REQUIRED_LOCAL_ATTENTION);
        if local_attention != REQUIRED_LOCAL_ATTENTION {
            return Err(start(format!(
                "modernbert local_attention must be exactly {REQUIRED_LOCAL_ATTENTION}, got {local_attention}"
            )));
        }
        let local_radius = local_attention / 2;

        let global_theta = raw.global_rope_theta.unwrap_or(DEFAULT_GLOBAL_THETA);
        let local_theta = raw.local_rope_theta.unwrap_or(DEFAULT_LOCAL_THETA);
        check_theta(global_theta, "global")?;
        check_theta(local_theta, "local")?;

        match raw.position_embedding_type.as_deref() {
            // Historical HF checkpoints may still say "absolute"; the native
            // ModernBERT recipe always uses the rotary positions below.
            None | Some("absolute") | Some("rope") => {}
            Some(other) => {
                return Err(start(format!(
                    "unsupported modernbert position_embedding_type {other}; native ModernBERT always uses RoPE"
                )))
            }
        }
        if raw
            .rope_scaling
            .as_ref()
            .is_some_and(|scaling| !scaling.is_null())
        {
            return Err(start(
                "modernbert rope_scaling is not implemented by the native Rotary recipe",
            ));
        }

        if raw
            .quantization_config
            .as_ref()
            .is_some_and(|value| !value.is_null())
        {
            return Err(start(
                "Modern quantized checkpoint metadata is not supported",
            ));
        }
        if raw.is_decoder.unwrap_or(false) {
            return Err(start("modernbert decoder mode is not supported"));
        }
        if raw.add_cross_attention.unwrap_or(false) {
            return Err(start("modernbert cross-attention is not supported"));
        }
        for (name, value) in [
            ("torch_dtype", raw.torch_dtype.as_deref()),
            ("dtype", raw.dtype.as_deref()),
        ] {
            if let Some(value) = value {
                let lowered = value.to_ascii_lowercase();
                if lowered.contains("float8") || lowered.contains("fp8") {
                    return Err(start(format!(
                        "FP8 modernbert {name} formats are not supported"
                    )));
                }
            }
        }

        let qkv_rows = hidden_size
            .checked_mul(3)
            .ok_or_else(|| start("modernbert QKV row count overflow"))?;
        let wi_rows = intermediate_size
            .checked_mul(2)
            .ok_or_else(|| start("modernbert Wi row count overflow"))?;
        for shape in [
            [vocab_size, hidden_size],
            [qkv_rows, hidden_size],
            [hidden_size, hidden_size],
            [wi_rows, hidden_size],
            [hidden_size, intermediate_size],
        ] {
            require_u32_shape(&shape)?;
        }

        let geometry = build_geometry(
            hidden_size,
            intermediate_size,
            num_attention_heads,
            num_hidden_layers,
            max_position_embeddings,
            vocab_size,
            norm_eps,
            global_every,
            local_attention,
            global_theta,
            local_theta,
            raw.architectures.as_deref(),
            raw.type_vocab_size,
        )?;
        verify_geometry(&geometry, norm_eps, global_theta)?;

        Ok(Self {
            geometry,
            global_theta,
            local_theta,
            global_every,
            local_radius,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn build_geometry(
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    max_position_embeddings: usize,
    vocab_size: usize,
    norm_eps: f64,
    global_every: usize,
    local_attention: usize,
    global_theta: f64,
    local_theta: f64,
    architectures: Option<&[String]>,
    configured_type_vocab_size: Option<usize>,
) -> Result<BertConfig, BackendError> {
    let _ = (global_every, local_attention, local_theta, architectures);
    Ok(BertConfig::modern_geometry(
        hidden_size,
        intermediate_size,
        num_attention_heads,
        num_hidden_layers,
        max_position_embeddings,
        vocab_size,
        norm_eps,
        global_theta,
        configured_type_vocab_size,
    ))
}

fn verify_geometry(
    geometry: &BertConfig,
    norm_eps: f64,
    global_theta: f64,
) -> Result<(), BackendError> {
    let expected = EncoderRecipe {
        embedding: NormArithmetic::RoundedAddBeforeStats,
        residual: NormArithmetic::RoundedAddBeforeStats,
        positions: PositionRecipe::Rotary(RotaryRecipe {
            theta: global_theta,
            ntk_factor: None,
        }),
        feed_forward: FeedForwardRecipe::GatedGeluUpFirst,
    };
    if geometry.family != EncoderFamily::Modern
        || geometry.type_vocab_size != 1
        || geometry.layer_norm_eps.to_bits() != norm_eps.to_bits()
        || geometry.recipe != expected
    {
        return Err(start(
            "modernbert geometry config did not produce the native ModernBERT recipe",
        ));
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub(crate) struct ModernLayerWeights {
    pub qkv: WeightSource,
    pub attention: WeightSource,
    pub up_gate: [WeightSource; 2],
    pub output: WeightSource,
    pub attention_gamma: Option<WeightSource>,
    pub mlp_gamma: WeightSource,
}

#[derive(Clone, Debug)]
pub(crate) struct ModernWeights {
    pub config: ModernConfig,
    pub word: WeightSource,
    pub embedding_gamma: WeightSource,
    pub final_gamma: WeightSource,
    pub layers: Vec<ModernLayerWeights>,
}

fn source(
    checkpoint: &Checkpoint,
    root: &str,
    relative: &str,
    shape: &[usize],
    used: &mut BTreeSet<String>,
) -> Result<WeightSource, BackendError> {
    let name = format!("{root}{relative}");
    let weight = checkpoint.source(&name, shape)?;
    used.insert(name);
    Ok(weight)
}

fn select_root(names: &BTreeSet<String>) -> Result<&'static str, BackendError> {
    let anchors = names
        .iter()
        .filter(|name| name.ends_with(WORD_SUFFIX))
        .collect::<Vec<_>>();
    let Some(anchor) = anchors.first() else {
        return Err(start(format!("missing weight {WORD_SUFFIX}")));
    };
    if anchors.len() != 1 {
        return Err(start(
            "ambiguous modernbert checkpoint namespace: multiple embeddings.tok_embeddings.weight roots",
        ));
    }
    match anchor.as_str() {
        "model.embeddings.tok_embeddings.weight" => Ok("model."),
        "embeddings.tok_embeddings.weight" => Ok(""),
        other => Err(start(format!(
            "unsupported modernbert checkpoint namespace root: {other}"
        ))),
    }
}

fn is_mlm_header(name: &str) -> bool {
    matches!(
        name,
        MLM_HEAD_DENSE
            | MLM_HEAD_NORM
            | MLM_DECODER_BIAS
            | "decoder.weight"
            | "head.dense.bias"
            | "head.norm.bias"
    )
}

fn has_backbone_body(body: &str) -> bool {
    body.starts_with("embeddings.")
        || body.starts_with("layers.")
        || body.starts_with("final_norm.")
}

fn validate_namespace(root: &str, names: &BTreeSet<String>) -> Result<(), BackendError> {
    for name in names {
        if is_mlm_header(name) {
            continue;
        }
        if root == "model." {
            if !name.starts_with("model.") && has_backbone_body(name) {
                return Err(start(format!(
                    "mixed modernbert checkpoint roots: {name} does not use the model. root"
                )));
            }
        } else if let Some(body) = name.strip_prefix("model.") {
            if has_backbone_body(body) {
                return Err(start(format!(
                    "mixed modernbert checkpoint roots: {name} unexpectedly uses the model. root"
                )));
            }
        }
    }
    Ok(())
}

fn validate_mlm_headers(
    checkpoint: &Checkpoint,
    config: &ModernConfig,
) -> Result<(), BackendError> {
    let hidden = config.geometry.hidden_size;
    let vocab = config.geometry.vocab_size;
    for (name, shape) in [
        (MLM_HEAD_DENSE, vec![hidden, hidden]),
        (MLM_HEAD_NORM, vec![hidden]),
        (MLM_DECODER_BIAS, vec![vocab]),
        ("decoder.weight", vec![vocab, hidden]),
        ("head.dense.bias", vec![hidden]),
        ("head.norm.bias", vec![hidden]),
    ] {
        let _ = checkpoint.optional_source(name, &shape)?;
    }
    Ok(())
}

fn reject_unexpected(checkpoint: &Checkpoint, used: &BTreeSet<String>) -> Result<(), BackendError> {
    for name in checkpoint.names() {
        if used.contains(name) || is_mlm_header(name) {
            continue;
        }
        if name.ends_with(".bias") {
            return Err(start(format!(
                "unexpected modernbert bias tensor {name}; backbone linear layers and norms are biasless"
            )));
        }
        return Err(start(format!(
            "unsupported modernbert checkpoint tensor {name}; only optional MLM head tensors may be ignored"
        )));
    }
    Ok(())
}

pub(crate) fn canonical_up_gate(
    weight: &WeightSource,
    hidden: usize,
    intermediate: usize,
) -> Result<[WeightSource; 2], BackendError> {
    let rows = intermediate
        .checked_mul(2)
        .ok_or_else(|| start("Modern Wi row overflow"))?;
    if weight.shape() != [rows, hidden] {
        return Err(start("Modern Wi canonicalization shape mismatch"));
    }
    let offset = intermediate
        .checked_mul(hidden)
        .ok_or_else(|| start("Modern Wi view offset overflow"))?;
    Ok([
        weight.slice(offset, vec![intermediate, hidden])?,
        weight.slice(0, vec![intermediate, hidden])?,
    ])
}

impl ModernWeights {
    pub(crate) fn load(path: &Path) -> Result<Self, BackendError> {
        let config = ModernConfig::load(path)?;
        crate::model::validate_modules(path)?;

        let checkpoint = Checkpoint::open(path)?;
        let names = checkpoint.names().cloned().collect::<BTreeSet<_>>();
        let root = select_root(&names)?;
        validate_namespace(root, &names)?;
        validate_mlm_headers(&checkpoint, &config)?;

        let hidden = config.geometry.hidden_size;
        let intermediate = config.geometry.intermediate_size;
        let vocab = config.geometry.vocab_size;
        let layer_count = config.geometry.num_hidden_layers;
        let qkv_rows = hidden
            .checked_mul(3)
            .ok_or_else(|| start("modernbert QKV row count overflow"))?;
        let wi_rows = intermediate
            .checked_mul(2)
            .ok_or_else(|| start("modernbert Wi row count overflow"))?;
        let mut used = BTreeSet::new();

        let word = source(
            &checkpoint,
            root,
            "embeddings.tok_embeddings.weight",
            &[vocab, hidden],
            &mut used,
        )?;
        let embedding_gamma = source(
            &checkpoint,
            root,
            "embeddings.norm.weight",
            &[hidden],
            &mut used,
        )?;
        let final_gamma = source(&checkpoint, root, "final_norm.weight", &[hidden], &mut used)?;

        let mut layers = Vec::new();
        layers.try_reserve_exact(layer_count).map_err(start)?;
        for layer_index in 0..layer_count {
            let prefix = format!("layers.{layer_index}");
            let qkv = source(
                &checkpoint,
                root,
                &format!("{prefix}.attn.Wqkv.weight"),
                &[qkv_rows, hidden],
                &mut used,
            )?;
            let attention = source(
                &checkpoint,
                root,
                &format!("{prefix}.attn.Wo.weight"),
                &[hidden, hidden],
                &mut used,
            )?;
            let wi = source(
                &checkpoint,
                root,
                &format!("{prefix}.mlp.Wi.weight"),
                &[wi_rows, hidden],
                &mut used,
            )?;
            // The checkpoint stores gate rows first. The prepared Oxide gate consumes
            // up first, so expose two checked row views of the same source.
            let up_gate = canonical_up_gate(&wi, hidden, intermediate)?;
            let output = source(
                &checkpoint,
                root,
                &format!("{prefix}.mlp.Wo.weight"),
                &[hidden, intermediate],
                &mut used,
            )?;
            let mlp_gamma = source(
                &checkpoint,
                root,
                &format!("{prefix}.mlp_norm.weight"),
                &[hidden],
                &mut used,
            )?;
            let attention_gamma = if layer_index == 0 {
                let name = format!("{root}{prefix}.attn_norm.weight");
                if checkpoint.contains(&name) {
                    return Err(start(format!(
                        "unexpected first-layer modernbert attention norm {name}; layer zero has no attention norm"
                    )));
                }
                None
            } else {
                Some(source(
                    &checkpoint,
                    root,
                    &format!("{prefix}.attn_norm.weight"),
                    &[hidden],
                    &mut used,
                )?)
            };
            layers.push(ModernLayerWeights {
                qkv,
                attention,
                up_gate,
                output,
                attention_gamma,
                mlp_gamma,
            });
        }

        let expected_count = layer_count
            .checked_mul(6)
            .and_then(|count| count.checked_add(2))
            .ok_or_else(|| start("modernbert backbone tensor count overflow"))?;
        if used.len() != expected_count {
            return Err(start(format!(
                "modernbert checkpoint tensor count mismatch: expected {expected_count} backbone tensors, loaded {}",
                used.len()
            )));
        }
        reject_unexpected(&checkpoint, &used)?;

        Ok(Self {
            config,
            word,
            embedding_gamma,
            final_gamma,
            layers,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{serialize, TensorView};
    use safetensors::Dtype;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    const HIDDEN: usize = 768;
    const INTERMEDIATE: usize = 8;
    const VOCAB: usize = 2;
    const POSITIONS: usize = 16;

    struct Fixture(PathBuf);

    impl Fixture {
        fn new(layers: usize) -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "infinity-modern-loader-{}-{}-{}",
                module_path!().replace("::", "-"),
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).unwrap();
            let fixture = Self(path);
            fixture.config(layers, serde_json::json!({}));
            fixture
        }

        fn raw_config(&self, value: serde_json::Value) {
            std::fs::write(
                self.0.join("config.json"),
                serde_json::to_vec(&value).unwrap(),
            )
            .unwrap();
        }

        fn config(&self, layers: usize, overrides: serde_json::Value) {
            let mut value = serde_json::json!({
                "model_type": "modernbert",
                "architectures": ["ModernBertForMaskedLM"],
                "hidden_size": HIDDEN,
                "intermediate_size": INTERMEDIATE,
                "num_attention_heads": 12,
                "num_hidden_layers": layers,
                "max_position_embeddings": POSITIONS,
                "vocab_size": VOCAB,
                "norm_eps": 1e-5,
                "layer_norm_eps": 1e-5,
                "hidden_activation": "gelu_pytorch_tanh",
                "attention_bias": false,
                "mlp_bias": false,
                "norm_bias": false,
                "global_attn_every_n_layers": 3,
                "local_attention": 128,
                "global_rope_theta": 160000.0,
                "local_rope_theta": 10000.0,
                "position_embedding_type": "absolute"
            });
            for (key, val) in overrides.as_object().unwrap() {
                value[key] = val.clone();
            }
            self.raw_config(value);
        }

        fn entries(layers: usize, root: &str) -> Vec<(String, Vec<usize>, String)> {
            let mut entries = vec![
                (
                    format!("{root}embeddings.tok_embeddings.weight"),
                    vec![VOCAB, HIDDEN],
                    "word".into(),
                ),
                (
                    format!("{root}embeddings.norm.weight"),
                    vec![HIDDEN],
                    "embedding_gamma".into(),
                ),
                (
                    format!("{root}final_norm.weight"),
                    vec![HIDDEN],
                    "final_gamma".into(),
                ),
            ];
            for layer in 0..layers {
                let prefix = format!("{root}layers.{layer}");
                for (suffix, shape, canonical) in [
                    (
                        "attn.Wqkv.weight",
                        vec![3 * HIDDEN, HIDDEN],
                        format!("layer{layer}.qkv"),
                    ),
                    (
                        "attn.Wo.weight",
                        vec![HIDDEN, HIDDEN],
                        format!("layer{layer}.attention"),
                    ),
                    (
                        "mlp.Wi.weight",
                        vec![2 * INTERMEDIATE, HIDDEN],
                        format!("layer{layer}.wi"),
                    ),
                    (
                        "mlp.Wo.weight",
                        vec![HIDDEN, INTERMEDIATE],
                        format!("layer{layer}.output"),
                    ),
                    (
                        "mlp_norm.weight",
                        vec![HIDDEN],
                        format!("layer{layer}.mlp_gamma"),
                    ),
                ] {
                    entries.push((format!("{prefix}.{suffix}"), shape, canonical));
                }
                if layer > 0 {
                    entries.push((
                        format!("{prefix}.attn_norm.weight"),
                        vec![HIDDEN],
                        format!("layer{layer}.attention_gamma"),
                    ));
                }
            }
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            entries
        }

        fn weights(
            &self,
            layers: usize,
            root: &str,
            dtype: Dtype,
            mlm: bool,
            omit: &[&str],
            extra: &[(&str, Vec<usize>)],
        ) {
            let mut entries = Self::entries(layers, root);
            entries.retain(|(_, _, canonical)| !omit.contains(&canonical.as_str()));
            if mlm {
                entries.push((
                    MLM_HEAD_DENSE.into(),
                    vec![HIDDEN, HIDDEN],
                    "head_dense".into(),
                ));
                entries.push((MLM_HEAD_NORM.into(), vec![HIDDEN], "head_norm".into()));
                entries.push((MLM_DECODER_BIAS.into(), vec![VOCAB], "decoder_bias".into()));
            }
            for (name, shape) in extra {
                entries.push(((*name).into(), shape.clone(), (*name).into()));
            }
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            self.write_entries(entries, dtype);
        }

        fn write_entries(&self, entries: Vec<(String, Vec<usize>, String)>, dtype: Dtype) {
            let data = entries
                .iter()
                .map(|(_, shape, canonical)| {
                    let count = shape.iter().product::<usize>();
                    let mut bytes = Vec::new();
                    for index in 0..count {
                        let value = test_value(canonical, index);
                        match dtype {
                            Dtype::F32 => bytes.extend_from_slice(&value.to_le_bytes()),
                            Dtype::F16 => bytes.extend_from_slice(
                                &half::f16::from_f32(value).to_bits().to_le_bytes(),
                            ),
                            Dtype::BF16 => bytes.extend_from_slice(
                                &half::bf16::from_f32(value).to_bits().to_le_bytes(),
                            ),
                            other => panic!("unsupported test dtype {other:?}"),
                        }
                    }
                    bytes
                })
                .collect::<Vec<_>>();
            let views = entries
                .iter()
                .zip(&data)
                .map(|((name, shape, _), bytes)| {
                    (
                        name.clone(),
                        TensorView::new(dtype, shape.clone(), bytes).unwrap(),
                    )
                })
                .collect::<Vec<_>>();
            std::fs::write(
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
            match ModernWeights::load(&self.0) {
                Ok(_) => panic!("invalid modernbert fixture was accepted"),
                Err(error) => error.to_string(),
            }
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn test_value(name: &str, index: usize) -> f32 {
        if name.ends_with("gamma") || name == "head_norm" {
            1.0
        } else if name == "decoder_bias" {
            (index as f32 + 1.0) / 4096.0
        } else {
            ((index % 1009) as f32 + 0.5) / 1024.0
        }
    }

    fn bf16(value: f32) -> u16 {
        half::bf16::from_f32(value).to_bits()
    }

    fn expected(dtype: Dtype, value: f32) -> u16 {
        match dtype {
            Dtype::F16 => bf16(half::f16::from_f32(value).to_f32()),
            _ => bf16(value),
        }
    }

    fn read(source: &WeightSource, offset: usize) -> u16 {
        let mut out = [0u16; 1];
        source.read_bf16(offset, &mut out).unwrap();
        out[0]
    }

    #[test]
    fn loads_both_roots_mlm_headers_and_all_float_dtypes() {
        for dtype in [Dtype::F32, Dtype::BF16, Dtype::F16] {
            for root in ["model.", ""] {
                let fixture = Fixture::new(1);
                fixture.weights(1, root, dtype, true, &[], &[]);
                let weights = ModernWeights::load(&fixture.0).unwrap();
                let config = &weights.config;
                assert_eq!(config.geometry.family, EncoderFamily::Modern);
                assert_eq!(config.geometry.hidden_size, HIDDEN);
                assert_eq!(config.geometry.intermediate_size, INTERMEDIATE);
                assert_eq!(config.geometry.num_attention_heads, 12);
                assert_eq!(config.geometry.num_hidden_layers, 1);
                assert_eq!(config.geometry.max_position_embeddings, POSITIONS);
                assert_eq!(config.geometry.vocab_size, VOCAB);
                assert_eq!(config.geometry.type_vocab_size, 1);
                assert_eq!(config.global_theta, 160_000.0);
                assert_eq!(config.local_theta, 10_000.0);
                assert_eq!(config.global_every, 3);
                assert_eq!(config.local_radius, 64);
                assert_eq!(
                    config.geometry.recipe.embedding,
                    NormArithmetic::RoundedAddBeforeStats
                );
                assert_eq!(
                    config.geometry.recipe.residual,
                    NormArithmetic::RoundedAddBeforeStats
                );
                assert_eq!(
                    config.geometry.recipe.feed_forward,
                    FeedForwardRecipe::GatedGeluUpFirst
                );
                assert!(matches!(
                    config.geometry.recipe.positions,
                    PositionRecipe::Rotary(RotaryRecipe {
                        theta,
                        ntk_factor: None
                    }) if theta == 160_000.0
                ));

                assert_eq!(weights.word.shape(), &[VOCAB, HIDDEN]);
                assert_eq!(weights.word.dtype(), dtype);
                assert_eq!(weights.embedding_gamma.shape(), &[HIDDEN]);
                assert_eq!(weights.final_gamma.shape(), &[HIDDEN]);
                assert_eq!(
                    read(&weights.word, 0),
                    expected(dtype, test_value("word", 0))
                );
                assert_eq!(
                    read(&weights.embedding_gamma, HIDDEN - 1),
                    expected(dtype, 1.0)
                );
                assert_eq!(weights.layers.len(), 1);
                let layer = &weights.layers[0];
                assert!(layer.attention_gamma.is_none());
                assert_eq!(layer.qkv.shape(), &[3 * HIDDEN, HIDDEN]);
                assert_eq!(layer.attention.shape(), &[HIDDEN, HIDDEN]);
                assert_eq!(layer.output.shape(), &[HIDDEN, INTERMEDIATE]);
                assert_eq!(layer.mlp_gamma.shape(), &[HIDDEN]);
                assert_eq!(layer.up_gate[0].shape(), &[INTERMEDIATE, HIDDEN]);
                assert_eq!(layer.up_gate[1].shape(), &[INTERMEDIATE, HIDDEN]);

                let row = INTERMEDIATE * HIDDEN;
                assert_eq!(
                    read(&layer.up_gate[0], 0),
                    expected(dtype, test_value("layer0.wi", row))
                );
                assert_eq!(
                    read(&layer.up_gate[0], row - 1),
                    expected(dtype, test_value("layer0.wi", 2 * row - 1))
                );
                assert_eq!(
                    read(&layer.up_gate[1], 0),
                    expected(dtype, test_value("layer0.wi", 0))
                );
                assert_eq!(
                    read(&layer.up_gate[1], row - 1),
                    expected(dtype, test_value("layer0.wi", row - 1))
                );
            }
        }
    }

    #[test]
    fn loads_two_layers_and_enforces_first_skip_contract() {
        let fixture = Fixture::new(2);
        fixture.weights(2, "model.", Dtype::F32, false, &[], &[]);
        let weights = ModernWeights::load(&fixture.0).unwrap();
        assert_eq!(weights.layers.len(), 2);
        assert!(weights.layers[0].attention_gamma.is_none());
        let attention_gamma = weights.layers[1].attention_gamma.as_ref().unwrap();
        assert_eq!(attention_gamma.shape(), &[HIDDEN]);
        assert_eq!(read(attention_gamma, 0), bf16(1.0));

        let fixture = Fixture::new(1);
        fixture.weights(
            1,
            "model.",
            Dtype::F32,
            false,
            &[],
            &[("model.layers.0.attn_norm.weight", vec![HIDDEN])],
        );
        assert!(fixture
            .error()
            .contains("first-layer modernbert attention norm"));

        let fixture = Fixture::new(2);
        fixture.weights(
            2,
            "model.",
            Dtype::F32,
            false,
            &["layer1.attention_gamma"],
            &[],
        );
        assert!(fixture.error().contains("missing weight"));
    }

    #[test]
    fn requires_backbone_norms_and_rejects_missing_layers() {
        let fixture = Fixture::new(1);
        fixture.weights(1, "model.", Dtype::F32, false, &["final_gamma"], &[]);
        assert!(fixture.error().contains("missing weight"));

        let fixture = Fixture::new(1);
        fixture.weights(1, "model.", Dtype::F32, false, &["embedding_gamma"], &[]);
        assert!(fixture.error().contains("missing weight"));

        let fixture = Fixture::new(2);
        fixture.weights(2, "model.", Dtype::F32, false, &["layer1.qkv"], &[]);
        assert!(fixture.error().contains("missing weight"));
    }

    #[test]
    fn rejects_wrong_shapes_extra_biases_and_unknown_tensors() {
        let fixture = Fixture::new(1);
        let mut entries = Fixture::entries(1, "model.");
        for entry in &mut entries {
            if entry.2 == "layer0.attention" {
                entry.1 = vec![HIDDEN, HIDDEN - 1];
            }
        }
        fixture.write_entries(entries, Dtype::F32);
        assert!(fixture.error().contains("wrong shape"));

        let fixture = Fixture::new(1);
        fixture.weights(
            1,
            "model.",
            Dtype::F32,
            false,
            &[],
            &[("model.layers.0.attn.Wo.bias", vec![HIDDEN])],
        );
        assert!(fixture.error().contains("bias tensor"));

        let fixture = Fixture::new(1);
        fixture.weights(
            1,
            "model.",
            Dtype::F32,
            false,
            &[],
            &[("classifier.weight", vec![VOCAB, HIDDEN])],
        );
        assert!(fixture.error().contains("classifier.weight"));

        let fixture = Fixture::new(1);
        fixture.weights(
            1,
            "model.",
            Dtype::F32,
            false,
            &[],
            &[("decoder.weight", vec![VOCAB, HIDDEN])],
        );
        assert!(
            ModernWeights::load(&fixture.0).is_ok(),
            "materialized tied decoder weights are irrelevant to backbone embeddings"
        );
    }

    #[test]
    fn validates_optional_mlm_header_metadata_when_present() {
        let fixture = Fixture::new(1);
        fixture.weights(
            1,
            "model.",
            Dtype::F32,
            false,
            &[],
            &[(MLM_HEAD_DENSE, vec![HIDDEN, HIDDEN - 1])],
        );
        assert!(fixture.error().contains("wrong shape"));

        let fixture = Fixture::new(1);
        fixture.weights(
            1,
            "model.",
            Dtype::F32,
            false,
            &[],
            &[(MLM_DECODER_BIAS, vec![VOCAB + 1])],
        );
        assert!(fixture.error().contains("wrong shape"));
    }

    #[test]
    fn rejects_ambiguous_and_mixed_checkpoint_roots() {
        let fixture = Fixture::new(1);
        fixture.weights(
            1,
            "model.",
            Dtype::F32,
            false,
            &[],
            &[("embeddings.tok_embeddings.weight", vec![VOCAB, HIDDEN])],
        );
        assert!(fixture
            .error()
            .contains("ambiguous modernbert checkpoint namespace"));

        let fixture = Fixture::new(1);
        fixture.weights(
            1,
            "model.",
            Dtype::F32,
            false,
            &[],
            &[("embeddings.norm.weight", vec![HIDDEN])],
        );
        assert!(fixture
            .error()
            .contains("mixed modernbert checkpoint roots"));

        let fixture = Fixture::new(1);
        fixture.weights(
            1,
            "",
            Dtype::F32,
            false,
            &[],
            &[("model.layers.0.attn.Wo.weight", vec![HIDDEN, HIDDEN])],
        );
        assert!(fixture
            .error()
            .contains("mixed modernbert checkpoint roots"));
    }

    #[test]
    fn rejects_unsupported_config_fields_before_reading_weights() {
        for (overrides, expected) in [
            (serde_json::json!({"model_type": "bert"}), "model_type"),
            (
                serde_json::json!({"architectures": ["BertModel"]}),
                "architecture",
            ),
            (serde_json::json!({"hidden_size": 384}), "hidden_size"),
            (
                serde_json::json!({"num_attention_heads": 6}),
                "num_attention_heads",
            ),
            (
                serde_json::json!({"num_hidden_layers": 0}),
                "num_hidden_layers",
            ),
            (
                serde_json::json!({"num_hidden_layers": 257}),
                "num_hidden_layers",
            ),
            (
                serde_json::json!({"intermediate_size": 0}),
                "intermediate_size",
            ),
            (
                serde_json::json!({"intermediate_size": 7}),
                "intermediate_size",
            ),
            (serde_json::json!({"vocab_size": 0}), "vocab_size"),
            (
                serde_json::json!({"max_position_embeddings": 0}),
                "max_position_embeddings",
            ),
            (
                serde_json::json!({"max_position_embeddings": 8193}),
                "max_position_embeddings",
            ),
            (serde_json::json!({"norm_eps": 0.0}), "norm_eps"),
            (serde_json::json!({"norm_eps": -1.0}), "norm_eps"),
            (
                serde_json::json!({"hidden_activation": "gelu_erf"}),
                "hidden_activation",
            ),
            (
                serde_json::json!({"attention_bias": true}),
                "attention_bias",
            ),
            (serde_json::json!({"mlp_bias": true}), "mlp_bias"),
            (serde_json::json!({"norm_bias": true}), "norm_bias"),
            (
                serde_json::json!({"global_attn_every_n_layers": 0}),
                "global attention",
            ),
            (
                serde_json::json!({"local_attention": 64}),
                "local_attention",
            ),
            (
                serde_json::json!({"global_rope_theta": 1.0}),
                "global rope theta",
            ),
            (
                serde_json::json!({"local_rope_theta": 0.0}),
                "local rope theta",
            ),
            (serde_json::json!({"is_decoder": true}), "decoder mode"),
            (
                serde_json::json!({"add_cross_attention": true}),
                "cross-attention",
            ),
            (serde_json::json!({"type_vocab_size": 2}), "type_vocab_size"),
            (
                serde_json::json!({"rope_scaling": {"type": "linear", "factor": 2.0}}),
                "rope_scaling",
            ),
            (
                serde_json::json!({"position_embedding_type": "relative_key"}),
                "position_embedding_type",
            ),
            (serde_json::json!({"torch_dtype": "float8_e4m3fn"}), "FP8"),
        ] {
            let fixture = Fixture::new(1);
            fixture.config(1, overrides);
            let error = fixture.error();
            assert!(error.contains(expected), "{error}");
        }
    }

    #[test]
    fn native_modern_norm_eps_ignores_legacy_fields_and_uses_native_default() {
        let fixture = Fixture::new(1);
        fixture.config(
            1,
            serde_json::json!({"norm_eps":null,"layer_norm_eps":1e-5}),
        );
        assert_eq!(
            ModernConfig::load(&fixture.0)
                .unwrap()
                .geometry
                .layer_norm_eps,
            1e-12
        );
        fixture.config(
            1,
            serde_json::json!({"norm_eps":1e-5,"layer_norm_eps":1e-9,"hidden_act":"gelu_erf"}),
        );
        assert_eq!(
            ModernConfig::load(&fixture.0)
                .unwrap()
                .geometry
                .layer_norm_eps,
            1e-5
        );
        fixture.raw_config(serde_json::json!({
            "model_type":"modernbert","dim":768,"hidden_dim":1152,"n_heads":12,
            "n_layers":22,"max_seq_length":8192,"n_vocab":50368
        }));
        assert!(
            ModernConfig::load(&fixture.0).is_err(),
            "foreign-family aliases must not silently change Modern geometry"
        );
    }

    #[test]
    fn parses_reference_modernbert_cache_profile_metadata() {
        let fixture = Fixture::new(1);
        fixture.raw_config(serde_json::json!({
            "model_type": "modernbert",
            "architectures": ["ModernBertForMaskedLM"],
            "hidden_size": 768,
            "intermediate_size": 1152,
            "num_attention_heads": 12,
            "num_hidden_layers": 22,
            "max_position_embeddings": 8192,
            "vocab_size": 50368,
            "norm_eps": 1e-5,
            "layer_norm_eps": 1e-5,
            "hidden_activation": "gelu",
            "attention_bias": false,
            "mlp_bias": false,
            "norm_bias": false,
            "global_attn_every_n_layers": 3,
            "local_attention": 128,
            "global_rope_theta": 160000.0,
            "local_rope_theta": 10000.0,
            "position_embedding_type": "absolute",
            "decoder_bias": true,
            "type_vocab_size": 0
        }));
        let config = ModernConfig::load(&fixture.0).unwrap();
        assert_eq!(config.geometry.num_hidden_layers, 22);
        assert_eq!(config.geometry.intermediate_size, 1152);
        assert_eq!(config.geometry.max_position_embeddings, 8192);
        assert_eq!(config.geometry.vocab_size, 50368);
        assert_eq!(config.geometry.type_vocab_size, 1);
        assert_eq!(config.global_every, 3);
        assert_eq!(config.local_radius, 64);
        assert_eq!(
            config.geometry.recipe.feed_forward,
            FeedForwardRecipe::GatedGeluUpFirst
        );
    }

    #[test]
    fn rejects_configs_whose_tensor_shape_products_do_not_fit_u32() {
        let fixture = Fixture::new(1);
        let huge_vocab = (u32::MAX as usize) / HIDDEN + 1;
        fixture.config(1, serde_json::json!({"vocab_size": huge_vocab}));
        let error = match ModernConfig::load(&fixture.0) {
            Ok(_) => panic!("oversized modernbert shape was accepted"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("u32"), "{error}");
    }
}
