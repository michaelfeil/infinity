// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Checked metadata adapter for Jina embeddings v2 ALiBi BERT checkpoints.
//!
//! The source config keeps `model_type = "bert"`; the native runtime selects
//! `Family::Jina` from the `jina-bert-implementation` name marker. This loader
//! mirrors that split, validates the exact Jina tensor set, and retains only
//! shard-backed [`WeightSource`] views. It never reads weight payloads.
//!

use crate::model::{
    read_json, select_root, validate_modules, BertConfig, Checkpoint, EncoderFamily, WeightSource,
};
use crate::start;
use infinity_v3_backend_core::BackendError;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

const JINA_HIDDEN: usize = 512;
const JINA_HEADS: usize = 8;
const JINA_HEAD_DIM: usize = 64;
const JINA_MAX_LAYERS: usize = 256;
const JINA_MAX_POSITIONS: usize = 8192;
const JINA_DEFAULT_EPS: f64 = 1e-12;
const JINA_DEFAULT_TYPES: usize = 2;
const JINA_IMPLEMENTATION: &str = "jina-bert-implementation";
const JINA_CODE_MARKER: &str = "jina-bert-v2-qk-post-norm";
const POOLER_WEIGHT: &str = "pooler.dense.weight";
const POOLER_BIAS: &str = "pooler.dense.bias";

#[derive(Debug, Deserialize)]
struct RawAutoMap {
    #[serde(rename = "AutoConfig")]
    auto_config: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct RawJinaMarker {
    pub(crate) model_type: Option<String>,
    #[serde(rename = "_name_or_path")]
    name_or_path: Option<String>,
    auto_map: Option<RawAutoMap>,
}

#[derive(Debug, Deserialize)]
struct RawJinaConfig {
    model_type: String,
    architectures: Option<Vec<String>>,
    #[serde(rename = "_name_or_path")]
    name_or_path: Option<String>,
    auto_map: Option<RawAutoMap>,
    hidden_size: Option<usize>,
    intermediate_size: Option<usize>,
    num_attention_heads: Option<usize>,
    num_hidden_layers: Option<usize>,
    max_position_embeddings: Option<usize>,
    vocab_size: Option<usize>,
    type_vocab_size: Option<usize>,
    layer_norm_eps: Option<f64>,
    feed_forward_type: Option<String>,
    hidden_act: Option<String>,
    position_embedding_type: Option<String>,
    #[serde(flatten)]
    extra: BTreeMap<String, serde_json::Value>,
}

fn required_usize(value: Option<usize>, name: &str) -> Result<usize, BackendError> {
    value.ok_or_else(|| start(format!("missing Jina config field {name}")))
}

fn enabled(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Null => false,
        serde_json::Value::Bool(enabled) => *enabled,
        serde_json::Value::Number(number) => number.as_f64().is_some_and(|value| value != 0.0),
        serde_json::Value::String(text) => !text.is_empty(),
        serde_json::Value::Array(items) => !items.is_empty(),
        serde_json::Value::Object(map) => !map.is_empty(),
    }
}

fn reject_extra(raw: &RawJinaConfig, names: &[&str], message: &str) -> Result<(), BackendError> {
    if names
        .iter()
        .any(|name| raw.extra.get(*name).is_some_and(enabled))
    {
        return Err(start(message));
    }
    Ok(())
}

fn marker_in(name_or_path: Option<&str>, auto_map: Option<&RawAutoMap>, marker: &str) -> bool {
    name_or_path.is_some_and(|name| name.contains(marker))
        || auto_map
            .and_then(|map| map.auto_config.as_deref())
            .is_some_and(|name| name.contains(marker))
}

fn has_marker(raw: &RawJinaConfig, marker: &str) -> bool {
    marker_in(raw.name_or_path.as_deref(), raw.auto_map.as_ref(), marker)
}

/// Classify a cold config header as a Jina checkpoint without requiring the
/// full config schema. Jina checkpoints keep the ordinary `bert` model type,
/// so this must run before the generic BERT loader accepts or rejects the file.
pub(crate) fn classify_marker(raw: &RawJinaMarker) -> Result<bool, BackendError> {
    if marker_in(
        raw.name_or_path.as_deref(),
        raw.auto_map.as_ref(),
        JINA_CODE_MARKER,
    ) {
        return Err(start(
            "Jina QK post-norm requires a separately prepared encoder policy",
        ));
    }
    Ok(raw.model_type.as_deref() == Some("bert")
        && marker_in(
            raw.name_or_path.as_deref(),
            raw.auto_map.as_ref(),
            JINA_IMPLEMENTATION,
        )
        && !marker_in(
            raw.name_or_path.as_deref(),
            raw.auto_map.as_ref(),
            JINA_CODE_MARKER,
        ))
}

/// Native selection order checks the JinaCode name marker before the ordinary
/// Jina implementation marker. Keep this query non-mutating so the legacy
/// `classify_marker` rejection remains correct for callers that assume the
/// QK-post-norm executor is still absent.
pub(crate) fn is_jina_code_marker(raw: &RawJinaMarker) -> bool {
    raw.model_type.as_deref() == Some("bert")
        && marker_in(
            raw.name_or_path.as_deref(),
            raw.auto_map.as_ref(),
            JINA_CODE_MARKER,
        )
}

#[cfg(test)]
pub(crate) fn detects(path: &Path) -> Result<bool, BackendError> {
    let raw: RawJinaMarker = read_json(&path.join("config.json"))?;
    classify_marker(&raw)
}

fn checked_shape(shape: &[usize]) -> Result<(), BackendError> {
    let count = shape
        .iter()
        .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
        .ok_or_else(|| start("Jina weight dimensions overflow"))?;
    if count > i32::MAX as usize {
        return Err(start("Jina weight shape exceeds the CUDA indexing range"));
    }
    Ok(())
}

/// Bounded, config-derived Jina geometry. The native source keeps the runtime
/// model type as `bert`; the parent geometry factory owns family selection and
/// the prepared ALiBi position policy.
#[derive(Clone, Debug)]
pub(crate) struct JinaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
    pub type_vocab_size: usize,
    pub layer_norm_eps: f64,
}

impl JinaConfig {
    fn load(path: &Path) -> Result<Self, BackendError> {
        let raw: RawJinaConfig = read_json(&path.join("config.json"))?;
        Self::from_raw(raw)
    }

    fn from_raw(raw: RawJinaConfig) -> Result<Self, BackendError> {
        if raw.model_type != "bert" {
            return Err(start(format!(
                "Jina source model_type must stay bert, got {}",
                raw.model_type
            )));
        }
        if raw.architectures.as_deref().is_some_and(|architectures| {
            architectures.iter().any(|architecture| {
                !matches!(
                    architecture.as_str(),
                    "JinaBertModel" | "JinaBertForMaskedLM" | "JinaBertForPreTraining"
                )
            })
        }) {
            return Err(start(
                "unsupported Jina architecture; expected a base or masked-language encoder",
            ));
        }
        if has_marker(&raw, JINA_CODE_MARKER) {
            return Err(start(
                "Jina QK post-norm checkpoints are not supported by the ALiBi loader",
            ));
        }
        if !has_marker(&raw, JINA_IMPLEMENTATION) {
            return Err(start(
                "Jina name marker is missing; ordinary BERT checkpoints are not accepted",
            ));
        }
        for (names, message) in [
            (
                &["is_decoder", "add_cross_attention"][..],
                "Jina decoder and cross-attention modes are not supported",
            ),
            (
                &["parallel_block", "parallel_block_tied_norm"][..],
                "Jina parallel block modifiers are not supported",
            ),
            (
                &["use_moe", "moe", "num_experts", "moe_every_n_layers"][..],
                "Jina mixture-of-experts checkpoints are not supported",
            ),
            (
                &["quantization_config"][..],
                "Jina quantized checkpoints are not supported",
            ),
            (
                &[
                    "rope_scaling",
                    "rope_parameters",
                    "rotary_emb_base",
                    "rotary_emb_fraction",
                    "rotary_emb_interleaved",
                ][..],
                "Jina rotary positions are not supported",
            ),
            (
                &["qk_post_norm", "qk_postnorm", "use_qk_norm"][..],
                "Jina QK post-normalization is not supported",
            ),
        ] {
            reject_extra(&raw, names, message)?;
        }
        if raw.position_embedding_type.as_deref() != Some("alibi") {
            return Err(start("Jina position_embedding_type must be alibi"));
        }
        if raw.feed_forward_type.as_deref() != Some("geglu") {
            return Err(start("Jina feed_forward_type must be geglu"));
        }
        if raw.hidden_act.as_deref() != Some("gelu") {
            return Err(start("Jina hidden_act must be gelu"));
        }

        let hidden = required_usize(raw.hidden_size, "hidden_size")?;
        let intermediate = required_usize(raw.intermediate_size, "intermediate_size")?;
        let heads = required_usize(raw.num_attention_heads, "num_attention_heads")?;
        let layers = required_usize(raw.num_hidden_layers, "num_hidden_layers")?;
        let positions = required_usize(raw.max_position_embeddings, "max_position_embeddings")?;
        let vocab = required_usize(raw.vocab_size, "vocab_size")?;
        let types = raw.type_vocab_size.unwrap_or(JINA_DEFAULT_TYPES);
        let eps = raw.layer_norm_eps.unwrap_or(JINA_DEFAULT_EPS);

        if hidden != JINA_HIDDEN || heads != JINA_HEADS || hidden / heads != JINA_HEAD_DIM {
            return Err(start(
                "Jina geometry must use hidden_size 512 with 8 heads and head_dim 64",
            ));
        }
        if intermediate == 0 || !intermediate.is_multiple_of(8) {
            return Err(start(
                "Jina intermediate_size must be a positive multiple of 8",
            ));
        }
        if layers == 0 || layers > JINA_MAX_LAYERS {
            return Err(start(format!(
                "Jina num_hidden_layers must be between 1 and {JINA_MAX_LAYERS}"
            )));
        }
        if vocab == 0 || types == 0 {
            return Err(start("Jina vocabulary dimensions must be positive"));
        }
        if positions == 0 || positions > JINA_MAX_POSITIONS {
            return Err(start(format!(
                "Jina max_position_embeddings must be between 1 and {JINA_MAX_POSITIONS}"
            )));
        }
        let eps_f32 = eps as f32;
        if !eps.is_finite() || eps <= 0.0 || !eps_f32.is_finite() || eps_f32 <= 0.0 {
            return Err(start(
                "Jina layer_norm_eps must be finite and positive in f32",
            ));
        }

        let gated_rows = intermediate
            .checked_mul(2)
            .ok_or_else(|| start("Jina GEGLU row count overflow"))?;
        for shape in [
            [vocab, hidden],
            [types, hidden],
            [hidden, hidden],
            [gated_rows, hidden],
            [hidden, intermediate],
        ] {
            checked_shape(&shape)?;
        }
        layers
            .checked_mul(15)
            .and_then(|count| count.checked_add(4))
            .ok_or_else(|| start("Jina tensor count overflow"))?;

        Ok(Self {
            hidden_size: hidden,
            intermediate_size: intermediate,
            num_attention_heads: heads,
            num_hidden_layers: layers,
            max_position_embeddings: positions,
            vocab_size: vocab,
            type_vocab_size: types,
            layer_norm_eps: eps,
        })
    }

    /// Checked arena geometry with the native ALiBi arithmetic recipe.
    pub(crate) fn geometry(&self) -> BertConfig {
        BertConfig::jina_geometry(
            self.hidden_size,
            self.intermediate_size,
            self.num_attention_heads,
            self.num_hidden_layers,
            self.max_position_embeddings,
            self.vocab_size,
            self.type_vocab_size,
            self.layer_norm_eps,
        )
    }
}

#[derive(Clone, Debug)]
pub(crate) struct JinaProjectionWeights {
    pub weight_parts: Vec<WeightSource>,
    pub bias_parts: Option<Vec<WeightSource>>,
}

#[derive(Clone, Debug)]
pub(crate) struct JinaLayerWeights {
    pub qkv: JinaProjectionWeights,
    pub attention: JinaProjectionWeights,
    pub intermediate: JinaProjectionWeights,
    pub output: JinaProjectionWeights,
    pub attention_gamma: WeightSource,
    pub attention_beta: WeightSource,
    pub output_gamma: WeightSource,
    pub output_beta: WeightSource,
}

#[derive(Clone, Debug)]
pub(crate) struct JinaWeights {
    pub geometry: BertConfig,
    pub word: WeightSource,
    pub token_type: WeightSource,
    pub gamma: WeightSource,
    pub beta: WeightSource,
    pub layers: Vec<JinaLayerWeights>,
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

fn row_slice(
    source: &WeightSource,
    first_row: usize,
    rows: usize,
    columns: usize,
) -> Result<WeightSource, BackendError> {
    let offset = first_row
        .checked_mul(columns)
        .ok_or_else(|| start("Jina row slice offset overflow"))?;
    source.slice(offset, vec![rows, columns])
}

fn validate_namespace(root: &str, names: &BTreeSet<String>) -> Result<(), BackendError> {
    const FOREIGN_ROOTS: [&str; 7] = [
        "bert.",
        "model.",
        "roberta.",
        "xlm-roberta.",
        "camembert.",
        "distilbert.",
        "new.",
    ];
    for name in names {
        if name.as_str() == POOLER_WEIGHT || name.as_str() == POOLER_BIAS {
            continue;
        }
        let coherent = if root.is_empty() {
            !FOREIGN_ROOTS.iter().any(|prefix| name.starts_with(*prefix))
        } else {
            name.starts_with(root)
        };
        if !coherent {
            return Err(start(format!(
                "incoherent Jina checkpoint namespace: {name} does not use the {root:?} root"
            )));
        }
    }
    Ok(())
}

fn load_pooler(
    checkpoint: &Checkpoint,
    root: &str,
    hidden: usize,
    used: &mut BTreeSet<String>,
) -> Result<bool, BackendError> {
    let mut candidates = vec![(POOLER_WEIGHT.to_owned(), POOLER_BIAS.to_owned())];
    if !root.is_empty() {
        candidates.push((
            format!("{root}{POOLER_WEIGHT}"),
            format!("{root}{POOLER_BIAS}"),
        ));
    }
    let mut complete = Vec::new();
    for (weight_name, bias_name) in candidates {
        let weight = checkpoint.optional_source(&weight_name, &[hidden, hidden])?;
        let bias = checkpoint.optional_source(&bias_name, &[hidden])?;
        match (weight, bias) {
            (None, None) => {}
            (Some(_), Some(_)) => complete.push((weight_name, bias_name)),
            _ => {
                return Err(start(format!(
                    "incomplete Jina pooler pair at {weight_name} / {bias_name}"
                )))
            }
        }
    }
    let Some((weight_name, bias_name)) = complete.pop() else {
        return Ok(false);
    };
    if !complete.is_empty() {
        return Err(start("ambiguous Jina pooler namespace"));
    }
    used.insert(weight_name);
    used.insert(bias_name);
    Ok(true)
}

fn reject_unexpected(checkpoint: &Checkpoint, used: &BTreeSet<String>) -> Result<(), BackendError> {
    for name in checkpoint.names() {
        if used.contains(name) {
            continue;
        }
        if name.ends_with(".mlp.gated_layers.bias") {
            return Err(start(format!(
                "unexpected Jina gated projection bias {name}; gated_layers is biasless"
            )));
        }
        if name.ends_with(".bias") {
            return Err(start(format!("unexpected Jina bias tensor {name}")));
        }
        return Err(start(format!("unsupported Jina checkpoint tensor {name}")));
    }
    Ok(())
}

impl JinaWeights {
    pub(crate) fn load(path: &Path) -> Result<Self, BackendError> {
        let config = JinaConfig::load(path)?;
        let geometry = config.geometry();
        validate_modules(path)?;
        let checkpoint = Checkpoint::open(path)?;
        let names = checkpoint.names().cloned().collect::<BTreeSet<_>>();
        let root = select_root(EncoderFamily::Bert, &names)?;
        validate_namespace(root, &names)?;

        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
        let vocab = config.vocab_size;
        let types = config.type_vocab_size;
        let gated_rows = intermediate
            .checked_mul(2)
            .ok_or_else(|| start("Jina GEGLU row count overflow"))?;
        let mut used = BTreeSet::new();

        let mut tensor =
            |relative: &str, shape: &[usize]| source(&checkpoint, root, relative, shape, &mut used);
        let word = tensor("embeddings.word_embeddings.weight", &[vocab, hidden])?;
        let token_type = tensor("embeddings.token_type_embeddings.weight", &[types, hidden])?;
        let gamma = tensor("embeddings.LayerNorm.weight", &[hidden])?;
        let beta = tensor("embeddings.LayerNorm.bias", &[hidden])?;

        let mut layers = Vec::new();
        layers
            .try_reserve_exact(config.num_hidden_layers)
            .map_err(start)?;
        for index in 0..config.num_hidden_layers {
            let prefix = format!("encoder.layer.{index}");
            let name = |relative: &str| format!("{prefix}.{relative}");
            let query_weight = tensor(&name("attention.self.query.weight"), &[hidden, hidden])?;
            let query_bias = tensor(&name("attention.self.query.bias"), &[hidden])?;
            let key_weight = tensor(&name("attention.self.key.weight"), &[hidden, hidden])?;
            let key_bias = tensor(&name("attention.self.key.bias"), &[hidden])?;
            let value_weight = tensor(&name("attention.self.value.weight"), &[hidden, hidden])?;
            let value_bias = tensor(&name("attention.self.value.bias"), &[hidden])?;
            let attention_weight =
                tensor(&name("attention.output.dense.weight"), &[hidden, hidden])?;
            let attention_bias = tensor(&name("attention.output.dense.bias"), &[hidden])?;
            let gated = tensor(&name("mlp.gated_layers.weight"), &[gated_rows, hidden])?;
            // Jina uses gate-first GEGLU: rows [0, I) are activated and rows
            // [I, 2I) are the up projection. The qualified shared GTE kernel is
            // up-first, so expose cold row views in [up, gate] order.
            let up = row_slice(&gated, intermediate, intermediate, hidden)?;
            let gate = row_slice(&gated, 0, intermediate, hidden)?;
            let output_weight = tensor(&name("mlp.wo.weight"), &[hidden, intermediate])?;
            let output_bias = tensor(&name("mlp.wo.bias"), &[hidden])?;
            let attention_gamma = tensor(&name("attention.output.LayerNorm.weight"), &[hidden])?;
            let attention_beta = tensor(&name("attention.output.LayerNorm.bias"), &[hidden])?;
            let output_gamma = tensor(&name("mlp.layernorm.weight"), &[hidden])?;
            let output_beta = tensor(&name("mlp.layernorm.bias"), &[hidden])?;
            layers.push(JinaLayerWeights {
                qkv: JinaProjectionWeights {
                    weight_parts: vec![query_weight, key_weight, value_weight],
                    bias_parts: Some(vec![query_bias, key_bias, value_bias]),
                },
                attention: JinaProjectionWeights {
                    weight_parts: vec![attention_weight],
                    bias_parts: Some(vec![attention_bias]),
                },
                intermediate: JinaProjectionWeights {
                    weight_parts: vec![up, gate],
                    bias_parts: None,
                },
                output: JinaProjectionWeights {
                    weight_parts: vec![output_weight],
                    bias_parts: Some(vec![output_bias]),
                },
                attention_gamma,
                attention_beta,
                output_gamma,
                output_beta,
            });
        }

        let has_pooler = load_pooler(&checkpoint, root, hidden, &mut used)?;
        let expected = config
            .num_hidden_layers
            .checked_mul(15)
            .and_then(|count| count.checked_add(4))
            .and_then(|count| count.checked_add(if has_pooler { 2 } else { 0 }))
            .ok_or_else(|| start("Jina tensor count overflow"))?;
        if used.len() != expected {
            return Err(start(format!(
                "Jina checkpoint tensor count mismatch: expected {expected}, loaded {}",
                used.len()
            )));
        }
        reject_unexpected(&checkpoint, &used)?;

        Ok(Self {
            geometry,
            word,
            token_type,
            gamma,
            beta,
            layers,
        })
    }
}

#[cfg(test)]
#[path = "../tests/support/jina_loader_tests.rs"]
mod jina_loader_tests;
