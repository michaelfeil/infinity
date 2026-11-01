// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Dense Nomic BERT (v1.5) checkpoint adapter.
//!
//! The actual checkpoint uses a bare namespace, packed QKV, and `fc12`/`fc11`
//! gated feed-forward matrices. This loader creates checked metadata views only;
//! payload bytes stay in the shard until the shared GTE-style engine reads them.

use crate::model::{
    read_json, select_root, validate_modules, BertConfig, Checkpoint, EncoderFamily, WeightSource,
};
use crate::start;
use infinity_v3_backend_core::BackendError;
use serde::Deserialize;
use std::collections::BTreeSet;
use std::path::Path;

const HIDDEN: usize = 768;
const HEADS: usize = 12;
const MAX_LAYERS: usize = 256;
const MAX_ROPE_POSITIONS: usize = 8192;
const DEFAULT_EPS: f64 = 1e-12;
const DEFAULT_THETA: f64 = 1000.0;
const DEFAULT_TYPES: usize = 2;

#[derive(Debug, Deserialize)]
struct RawRopeParameters {
    #[serde(rename = "rope_type")]
    kind: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawNomicConfig {
    model_type: String,
    architectures: Option<Vec<String>>,
    n_embd: Option<usize>,
    n_head: Option<usize>,
    n_layer: Option<usize>,
    n_inner: Option<usize>,
    vocab_size: Option<usize>,
    type_vocab_size: Option<usize>,
    layer_norm_epsilon: Option<f64>,
    max_position_embeddings: Option<usize>,
    n_positions: Option<usize>,
    activation_function: Option<String>,
    rotary_emb_base: Option<f64>,
    rotary_emb_fraction: Option<f64>,
    rotary_emb_interleaved: Option<bool>,
    rotary_emb_scale_base: Option<serde_json::Value>,
    rotary_scaling_factor: Option<serde_json::Value>,
    rope_scaling: Option<serde_json::Value>,
    rope_parameters: Option<RawRopeParameters>,
    use_rms_norm: Option<bool>,
    prenorm: Option<bool>,
    causal: Option<bool>,
    is_decoder: Option<bool>,
    add_cross_attention: Option<bool>,
    parallel_block: Option<bool>,
    parallel_block_tied_norm: Option<bool>,
    use_moe: Option<bool>,
    moe: Option<serde_json::Value>,
    num_experts: Option<usize>,
    moe_every_n_layers: Option<usize>,
    quantization_config: Option<serde_json::Value>,
    qkv_proj_bias: Option<bool>,
    mlp_fc1_bias: Option<bool>,
    mlp_fc2_bias: Option<bool>,
}

fn checked_shape(shape: &[usize]) -> Result<(), BackendError> {
    let count = shape
        .iter()
        .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
        .ok_or_else(|| start("nomic weight dimensions overflow"))?;
    if count > u32::MAX as usize {
        return Err(start("nomic weight shape exceeds the u32 element range"));
    }
    Ok(())
}

fn nonnull(value: Option<&serde_json::Value>) -> bool {
    value.is_some_and(|value| !value.is_null())
}

impl RawNomicConfig {
    fn load(path: &Path) -> Result<BertConfig, BackendError> {
        let raw: Self = read_json(&path.join("config.json"))?;
        Self::geometry(raw)
    }

    fn geometry(raw: Self) -> Result<BertConfig, BackendError> {
        if !matches!(raw.model_type.as_str(), "nomic_bert" | "nomic-bert") {
            return Err(start(format!(
                "unsupported nomic model_type {}; expected nomic_bert or nomic-bert",
                raw.model_type
            )));
        }
        if raw.architectures.as_deref().is_some_and(|architectures| {
            architectures
                .iter()
                .any(|architecture| architecture != "NomicBertModel")
        }) {
            return Err(start(
                "unsupported nomic architecture; expected NomicBertModel",
            ));
        }
        if raw.activation_function.as_deref() != Some("swiglu") {
            return Err(start("nomic activation_function must be swiglu"));
        }
        if raw.use_rms_norm.unwrap_or(false) {
            return Err(start("nomic RMS normalization is not supported"));
        }
        if raw.prenorm.unwrap_or(false) {
            return Err(start("nomic post-norm requires prenorm=false"));
        }
        if raw.causal.unwrap_or(false) {
            return Err(start("nomic causal attention is not supported"));
        }
        if raw.is_decoder.unwrap_or(false) {
            return Err(start("nomic decoder mode is not supported"));
        }
        if raw.add_cross_attention.unwrap_or(false) {
            return Err(start("nomic cross-attention is not supported"));
        }
        if raw.parallel_block.unwrap_or(false) || raw.parallel_block_tied_norm.unwrap_or(false) {
            return Err(start("nomic parallel block modifiers are not supported"));
        }
        if raw.use_moe.unwrap_or(false)
            || nonnull(raw.moe.as_ref())
            || raw.num_experts.is_some_and(|count| count > 0)
            || raw.moe_every_n_layers.is_some_and(|count| count > 0)
        {
            return Err(start(
                "nomic mixture-of-experts checkpoints are not supported",
            ));
        }
        if nonnull(raw.quantization_config.as_ref()) {
            return Err(start("nomic quantized checkpoints are not supported"));
        }
        if raw.qkv_proj_bias.unwrap_or(false)
            || raw.mlp_fc1_bias.unwrap_or(false)
            || raw.mlp_fc2_bias.unwrap_or(false)
        {
            return Err(start("nomic projection biases are not supported"));
        }
        if raw
            .rotary_emb_fraction
            .is_some_and(|fraction| fraction != 1.0)
        {
            return Err(start("nomic rotary_emb_fraction must be 1.0"));
        }
        if raw.rotary_emb_interleaved.unwrap_or(false) {
            return Err(start(
                "nomic interleaved rotary embeddings are not supported",
            ));
        }
        if nonnull(raw.rotary_emb_scale_base.as_ref())
            || nonnull(raw.rotary_scaling_factor.as_ref())
            || nonnull(raw.rope_scaling.as_ref())
        {
            return Err(start("nomic dynamic rotary scaling is not supported"));
        }
        if raw.rope_parameters.as_ref().is_some_and(|parameters| {
            parameters
                .kind
                .as_deref()
                .is_some_and(|kind| kind != "default")
        }) {
            return Err(start(
                "nomic rope_parameters must use the default rotary type",
            ));
        }

        // Native Nomic options use these names and defaults. Legacy HF
        // aliases are deliberately not allowed to change the arithmetic.
        let hidden = raw.n_embd.unwrap_or(HIDDEN);
        let heads = raw.n_head.unwrap_or(HEADS);
        let layers = raw.n_layer.unwrap_or(12);
        let intermediate = raw.n_inner.unwrap_or(3072);
        let vocab = raw
            .vocab_size
            .ok_or_else(|| start("missing nomic config field vocab_size"))?;
        let type_vocab = raw.type_vocab_size.unwrap_or(DEFAULT_TYPES);
        let rope_positions = raw.n_positions.unwrap_or(MAX_ROPE_POSITIONS);
        // This optional HF tokenizer bound does not replace native n_positions.
        let frontend = raw.max_position_embeddings.unwrap_or(rope_positions);
        let eps = raw.layer_norm_epsilon.unwrap_or(DEFAULT_EPS);
        let theta = raw.rotary_emb_base.unwrap_or(DEFAULT_THETA);

        if hidden != HIDDEN || heads != HEADS {
            return Err(start(
                "nomic hidden size must be 768 with 12 attention heads",
            ));
        }
        if intermediate == 0 || !intermediate.is_multiple_of(8) {
            return Err(start(
                "nomic intermediate size must be a positive multiple of 8",
            ));
        }
        if layers == 0 || layers > MAX_LAYERS {
            return Err(start(format!(
                "nomic num_hidden_layers must be between 1 and {MAX_LAYERS}"
            )));
        }
        if vocab == 0 || type_vocab == 0 {
            return Err(start("nomic vocabulary dimensions must be positive"));
        }
        if frontend == 0
            || rope_positions == 0
            || rope_positions > MAX_ROPE_POSITIONS
            || frontend > rope_positions
            || u32::try_from(frontend).is_err()
            || u32::try_from(rope_positions).is_err()
        {
            return Err(start(
                "invalid nomic frontend positions or native RoPE capacity",
            ));
        }
        let eps_f32 = eps as f32;
        if !eps.is_finite() || eps <= 0.0 || !eps_f32.is_finite() || eps_f32 <= 0.0 {
            return Err(start(
                "nomic layer_norm_epsilon must be finite and positive in f32",
            ));
        }
        let theta_f32 = theta as f32;
        if !theta.is_finite() || theta <= 1.0 || !theta_f32.is_finite() || theta_f32 <= 1.0 {
            return Err(start(
                "nomic rotary_emb_base must be finite and greater than one in f32",
            ));
        }

        let qkv_rows = hidden
            .checked_mul(3)
            .ok_or_else(|| start("nomic QKV row count overflow"))?;
        for shape in [
            [vocab, hidden],
            [type_vocab, hidden],
            [qkv_rows, hidden],
            [hidden, hidden],
            [intermediate, hidden],
            [hidden, intermediate],
        ] {
            checked_shape(&shape)?;
        }
        layers
            .checked_mul(9)
            .and_then(|count| count.checked_add(4))
            .ok_or_else(|| start("nomic tensor count overflow"))?;

        Ok(BertConfig::nomic_geometry(
            hidden,
            intermediate,
            heads,
            layers,
            rope_positions,
            vocab,
            type_vocab,
            eps,
            theta,
            rope_positions,
        ))
    }
}

#[derive(Clone, Debug)]
pub(crate) struct MatrixSources {
    pub(crate) parts: Vec<WeightSource>,
}

#[derive(Clone, Debug)]
pub(crate) struct NomicLayerWeights {
    pub qkv: MatrixSources,
    pub attention: MatrixSources,
    pub intermediate: MatrixSources,
    pub output: MatrixSources,
    pub attention_gamma: WeightSource,
    pub attention_beta: WeightSource,
    pub output_gamma: WeightSource,
    pub output_beta: WeightSource,
}

#[derive(Clone, Debug)]
pub(crate) struct NomicWeights {
    pub config: BertConfig,
    pub word: WeightSource,
    pub token_type: WeightSource,
    pub gamma: WeightSource,
    pub beta: WeightSource,
    pub layers: Vec<NomicLayerWeights>,
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

fn validate_namespace(root: &str, names: &BTreeSet<String>) -> Result<(), BackendError> {
    for name in names {
        let foreign_root = if root.is_empty() {
            name.starts_with("bert.") || name.starts_with("model.")
        } else {
            !name.starts_with(root)
        };
        if foreign_root {
            return Err(start(format!(
                "incoherent nomic checkpoint namespace: {name} does not use the {root:?} root"
            )));
        }
    }
    Ok(())
}

fn reject_unexpected(checkpoint: &Checkpoint, used: &BTreeSet<String>) -> Result<(), BackendError> {
    for name in checkpoint.names() {
        if used.contains(name) {
            continue;
        }
        if name.ends_with(".bias") {
            return Err(start(format!(
                "unexpected nomic bias tensor {name}; all matrices are biasless"
            )));
        }
        return Err(start(format!("unsupported nomic checkpoint tensor {name}")));
    }
    Ok(())
}

impl NomicWeights {
    pub(crate) fn load(path: &Path) -> Result<Self, BackendError> {
        let config = RawNomicConfig::load(path)?;
        validate_modules(path)?;
        let checkpoint = Checkpoint::open(path)?;
        let names = checkpoint.names().cloned().collect::<BTreeSet<_>>();
        let root = select_root(EncoderFamily::Nomic, &names)?;
        validate_namespace(root, &names)?;

        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
        let vocab = config.vocab_size;
        let types = config.type_vocab_size;
        let qkv_rows = hidden
            .checked_mul(3)
            .ok_or_else(|| start("nomic QKV row count overflow"))?;
        let mut used = BTreeSet::new();

        let word = source(
            &checkpoint,
            root,
            "embeddings.word_embeddings.weight",
            &[vocab, hidden],
            &mut used,
        )?;
        let token_type = source(
            &checkpoint,
            root,
            "embeddings.token_type_embeddings.weight",
            &[types, hidden],
            &mut used,
        )?;
        let gamma = source(&checkpoint, root, "emb_ln.weight", &[hidden], &mut used)?;
        let beta = source(&checkpoint, root, "emb_ln.bias", &[hidden], &mut used)?;

        let mut layers = Vec::new();
        layers
            .try_reserve_exact(config.num_hidden_layers)
            .map_err(start)?;
        for index in 0..config.num_hidden_layers {
            let prefix = format!("encoder.layers.{index}");
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
                &format!("{prefix}.attn.out_proj.weight"),
                &[hidden, hidden],
                &mut used,
            )?;
            let gate = source(
                &checkpoint,
                root,
                &format!("{prefix}.mlp.fc12.weight"),
                &[intermediate, hidden],
                &mut used,
            )?;
            let up = source(
                &checkpoint,
                root,
                &format!("{prefix}.mlp.fc11.weight"),
                &[intermediate, hidden],
                &mut used,
            )?;
            let output = source(
                &checkpoint,
                root,
                &format!("{prefix}.mlp.fc2.weight"),
                &[hidden, intermediate],
                &mut used,
            )?;
            let attention_gamma = source(
                &checkpoint,
                root,
                &format!("{prefix}.norm1.weight"),
                &[hidden],
                &mut used,
            )?;
            let attention_beta = source(
                &checkpoint,
                root,
                &format!("{prefix}.norm1.bias"),
                &[hidden],
                &mut used,
            )?;
            let output_gamma = source(
                &checkpoint,
                root,
                &format!("{prefix}.norm2.weight"),
                &[hidden],
                &mut used,
            )?;
            let output_beta = source(
                &checkpoint,
                root,
                &format!("{prefix}.norm2.bias"),
                &[hidden],
                &mut used,
            )?;
            layers.push(NomicLayerWeights {
                qkv: MatrixSources { parts: vec![qkv] },
                attention: MatrixSources {
                    parts: vec![attention],
                },
                intermediate: MatrixSources {
                    parts: vec![gate, up],
                },
                output: MatrixSources {
                    parts: vec![output],
                },
                attention_gamma,
                attention_beta,
                output_gamma,
                output_beta,
            });
        }

        let expected = config
            .num_hidden_layers
            .checked_mul(9)
            .and_then(|count| count.checked_add(4))
            .ok_or_else(|| start("nomic tensor count overflow"))?;
        if used.len() != expected {
            return Err(start(format!(
                "nomic checkpoint tensor count mismatch: expected {expected}, loaded {}",
                used.len()
            )));
        }
        reject_unexpected(&checkpoint, &used)?;

        Ok(Self {
            config,
            word,
            token_type,
            gamma,
            beta,
            layers,
        })
    }
}

#[cfg(test)]
#[path = "../tests/support/nomic_loader_tests.rs"]
mod tests;
