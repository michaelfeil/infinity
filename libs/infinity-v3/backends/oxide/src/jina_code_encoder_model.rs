// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! JinaCode (`jina-bert-v2-qk-post-norm`) cold checkpoint adapter.
//!
//! This module validates the source config, selects the coherent bare/`bert.`
//! namespace, checks the exact F16 tensor inventory, and retains only
//! shard-backed [`WeightSource`] views. It never reads, copies, or materializes
//! tensor payloads. The existing `Checkpoint`/`WeightSource` machinery performs
//! bounded reads and the shared uploader performs finite F16-to-BF16 conversion.
//!
//! The published JinaCode arithmetic is intentionally recorded here rather than
//! executed:
//! - Q, K, and V projections stay in query/key/value order.
//! - `attention.self.layer_norm_q` and `layer_norm_k` are full-width H768
//!   affine LayerNorms applied after the head split is flattened again.
//! - `attention.output.LayerNorm` is followed by the extra post-attention
//!   `layer_norm_1` residual normalization, and `layer_norm_2` follows the MLP.
//! - `mlp.up_gated_layer.weight` has packed `[up | gate]` rows. The activation is
//!   `up * GELU(gate)`, not `GELU(up) * gate`; the latter reverses the published
//!   JinaCode layout.
//! - Positions are ALiBi. There is no absolute position table and no rotary
//!   state. The prepared executor, Q/K strided kernel, and twelve-head ALiBi
//!   implementation are deliberately outside this cold descriptor.

use crate::model::{
    read_json, select_root, start, validate_modules, BertConfig, Checkpoint, EncoderFamily,
    EncoderRecipe, FeedForwardRecipe, LinearWeights, NormArithmetic, PositionRecipe, WeightSource,
};
use infinity_v3_backend_core::BackendError;
use safetensors::Dtype;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

const JINA_CODE_MARKER: &str = "jina-bert-v2-qk-post-norm";
const HIDDEN_SIZE: usize = 768;
const INTERMEDIATE_SIZE: usize = 3072;
const NUM_HEADS: usize = 12;
const NUM_LAYERS: usize = 12;
const VOCAB_SIZE: usize = 61056;
const TYPE_VOCAB_SIZE: usize = 2;
const MAX_POSITIONS: usize = 8192;
const LAYER_NORM_EPS: f64 = 1e-12;
const TENSORS_PER_LAYER: usize = 21;
const EMBEDDING_TENSORS: usize = 4;
const POOLER_TENSORS: usize = 2;

/// Packed QKV consumers must preserve this output-feature order.
pub(crate) const PACKED_QKV_ORDER: [&str; 3] = ["query", "key", "value"];
/// The packed up/gated projection stores the unactivated up half first.
pub(crate) const PACKED_MLP_ORDER: [&str; 2] = ["up", "gate"];

const ALLOWED_EXTRA_CONFIG: [&str; 12] = [
    "attention_probs_dropout_prob",
    "attn_implementation",
    "classifier_dropout",
    "emb_pooler",
    "gradient_checkpointing",
    "hidden_dropout_prob",
    "initializer_range",
    "model_max_length",
    "pad_token_id",
    "torch_dtype",
    "transformers_version",
    "use_cache",
];

const UNSUPPORTED_EXTRA_CONFIG: [(&str, &str); 16] = [
    ("is_decoder", "JinaCode decoder mode is not supported"),
    (
        "add_cross_attention",
        "JinaCode cross-attention is not supported",
    ),
    (
        "parallel_block",
        "JinaCode parallel block modifiers are not supported",
    ),
    (
        "parallel_block_tied_norm",
        "JinaCode parallel block modifiers are not supported",
    ),
    (
        "use_moe",
        "JinaCode mixture-of-experts checkpoints are not supported",
    ),
    (
        "moe",
        "JinaCode mixture-of-experts checkpoints are not supported",
    ),
    (
        "num_experts",
        "JinaCode mixture-of-experts checkpoints are not supported",
    ),
    (
        "moe_every_n_layers",
        "JinaCode mixture-of-experts checkpoints are not supported",
    ),
    (
        "quantization_config",
        "JinaCode quantized checkpoints are not supported",
    ),
    (
        "rope_scaling",
        "JinaCode rotary positions are not supported",
    ),
    (
        "rope_parameters",
        "JinaCode rotary positions are not supported",
    ),
    (
        "rotary_emb_base",
        "JinaCode rotary positions are not supported",
    ),
    (
        "rotary_emb_fraction",
        "JinaCode rotary positions are not supported",
    ),
    (
        "rotary_emb_interleaved",
        "JinaCode rotary positions are not supported",
    ),
    (
        "qk_post_norm",
        "JinaCode QK post-norm is selected by the native name marker, not a config override",
    ),
    (
        "use_qk_norm",
        "JinaCode QK post-norm is selected by the native name marker, not a config override",
    ),
];

#[derive(Debug, Deserialize)]
struct RawAutoMap {
    #[serde(rename = "AutoConfig")]
    auto_config: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawJinaCodeConfig {
    model_type: String,
    architectures: Option<Vec<String>>,
    #[serde(rename = "_name_or_path")]
    name_or_path: Option<String>,
    auto_map: Option<RawAutoMap>,
    feed_forward_type: Option<String>,
    hidden_act: Option<String>,
    hidden_size: Option<usize>,
    intermediate_size: Option<usize>,
    layer_norm_eps: Option<f64>,
    max_position_embeddings: Option<usize>,
    num_attention_heads: Option<usize>,
    num_hidden_layers: Option<usize>,
    position_embedding_type: Option<String>,
    type_vocab_size: Option<usize>,
    vocab_size: Option<usize>,
    #[serde(flatten)]
    extra: BTreeMap<String, serde_json::Value>,
}

fn required_usize(value: Option<usize>, name: &str) -> Result<usize, BackendError> {
    value.ok_or_else(|| start(format!("missing JinaCode config field {name}")))
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

fn marker_in(name_or_path: Option<&str>, auto_map: Option<&RawAutoMap>) -> bool {
    name_or_path.is_some_and(|name| name.contains(JINA_CODE_MARKER))
        || auto_map
            .and_then(|map| map.auto_config.as_deref())
            .is_some_and(|name| name.contains(JINA_CODE_MARKER))
}

fn reject_extra_config(extra: &BTreeMap<String, serde_json::Value>) -> Result<(), BackendError> {
    for (name, message) in UNSUPPORTED_EXTRA_CONFIG {
        if extra.get(name).is_some_and(enabled) {
            return Err(start(message));
        }
    }
    for name in extra.keys() {
        let supported = ALLOWED_EXTRA_CONFIG.contains(&name.as_str())
            || UNSUPPORTED_EXTRA_CONFIG
                .iter()
                .any(|(known, _)| *known == name);
        if !supported {
            return Err(start(format!("unknown JinaCode config metadata {name}")));
        }
    }
    Ok(())
}

impl RawJinaCodeConfig {
    fn load(path: &Path) -> Result<BertConfig, BackendError> {
        let raw: Self = read_json(&path.join("config.json"))?;
        Self::descriptor(raw)
    }

    fn descriptor(raw: Self) -> Result<BertConfig, BackendError> {
        if raw.model_type != "bert" {
            return Err(start(format!(
                "JinaCode source model_type must stay bert, got {}",
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
                "unsupported JinaCode architecture; expected a base or masked-language encoder",
            ));
        }
        if !marker_in(raw.name_or_path.as_deref(), raw.auto_map.as_ref()) {
            return Err(start(format!(
                "JinaCode name marker {JINA_CODE_MARKER} is missing from _name_or_path and auto_map.AutoConfig"
            )));
        }
        if raw.position_embedding_type.as_deref() != Some("alibi") {
            return Err(start("JinaCode position_embedding_type must be alibi"));
        }
        if raw.feed_forward_type.as_deref() != Some("geglu") {
            return Err(start("JinaCode feed_forward_type must be geglu"));
        }
        if raw.hidden_act.as_deref() != Some("gelu") {
            return Err(start("JinaCode hidden_act must be gelu"));
        }
        reject_extra_config(&raw.extra)?;

        let hidden = required_usize(raw.hidden_size, "hidden_size")?;
        let intermediate = required_usize(raw.intermediate_size, "intermediate_size")?;
        let heads = required_usize(raw.num_attention_heads, "num_attention_heads")?;
        let layers = required_usize(raw.num_hidden_layers, "num_hidden_layers")?;
        let positions = required_usize(raw.max_position_embeddings, "max_position_embeddings")?;
        let vocab = required_usize(raw.vocab_size, "vocab_size")?;
        let types = raw.type_vocab_size.unwrap_or(TYPE_VOCAB_SIZE);
        let eps = raw.layer_norm_eps.unwrap_or(LAYER_NORM_EPS);

        if hidden != HIDDEN_SIZE
            || intermediate != INTERMEDIATE_SIZE
            || heads != NUM_HEADS
            || layers != NUM_LAYERS
            || positions != MAX_POSITIONS
            || vocab != VOCAB_SIZE
            || types != TYPE_VOCAB_SIZE
            || eps != LAYER_NORM_EPS
        {
            return Err(start(
                "JinaCode requires H768/I3072, 12 heads, 12 layers, vocab 61056, 2 token types, 8192 ALiBi positions, and layer_norm_eps 1e-12",
            ));
        }

        let eps_f32 = eps as f32;
        if !eps.is_finite() || eps <= 0.0 || !eps_f32.is_finite() || eps_f32 <= 0.0 {
            return Err(start(
                "JinaCode layer_norm_eps must be finite and positive in f32",
            ));
        }
        intermediate
            .checked_mul(2)
            .ok_or_else(|| start("JinaCode packed up/gate row count overflow"))?;

        Ok(BertConfig::jina_code_geometry(
            hidden,
            intermediate,
            heads,
            layers,
            positions,
            vocab,
            types,
            eps,
        ))
    }
}

/// Cold, checked JinaCode descriptor. It deliberately does not include a
/// prepared execution plan; that must be constructed separately once the
/// JinaCode Q/K strided kernel and twelve-head ALiBi executor are integrated.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct JinaCodeConfig {
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) vocab_size: usize,
    pub(crate) type_vocab_size: usize,
    pub(crate) max_position_embeddings: usize,
    pub(crate) layer_norm_eps: f64,
    pub(crate) recipe: EncoderRecipe,
    pub(crate) packed_qkv_order: [&'static str; 3],
    pub(crate) packed_mlp_order: [&'static str; 2],
}

impl JinaCodeConfig {
    pub(crate) fn geometry(&self) -> BertConfig {
        BertConfig::jina_code_geometry(
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

pub(crate) struct LayerNormWeights {
    pub(crate) gamma: WeightSource,
    pub(crate) beta: WeightSource,
}

pub(crate) struct JinaCodeLayerWeights {
    pub(crate) query: LinearWeights,
    pub(crate) key: LinearWeights,
    pub(crate) value: LinearWeights,
    pub(crate) query_norm: LayerNormWeights,
    pub(crate) key_norm: LayerNormWeights,
    pub(crate) attention: LinearWeights,
    pub(crate) attention_norm: LayerNormWeights,
    pub(crate) layer_norm_1: LayerNormWeights,
    pub(crate) layer_norm_2: LayerNormWeights,
    pub(crate) mlp_up_gated: WeightSource,
    pub(crate) mlp_down: LinearWeights,
}

pub(crate) struct JinaCodeWeights {
    pub(crate) config: JinaCodeConfig,
    pub(crate) word: WeightSource,
    pub(crate) token_type: WeightSource,
    pub(crate) embedding_norm: LayerNormWeights,
    pub(crate) layers: Vec<JinaCodeLayerWeights>,
    // Validated optional checkpoint projection; native embedding tasks ignore it.
    pub(crate) _pooler: Option<LinearWeights>,
}

fn descriptor(config: &BertConfig) -> Result<JinaCodeConfig, BackendError> {
    let valid = config.family == EncoderFamily::JinaCode
        && config.hidden_size == HIDDEN_SIZE
        && config.intermediate_size == INTERMEDIATE_SIZE
        && config.num_attention_heads == NUM_HEADS
        && config.num_hidden_layers == NUM_LAYERS
        && config.vocab_size == VOCAB_SIZE
        && config.type_vocab_size == TYPE_VOCAB_SIZE
        && config.max_position_embeddings == MAX_POSITIONS
        && config.layer_norm_eps == LAYER_NORM_EPS
        && config.recipe.embedding == NormArithmetic::ExactWordTypeBiased
        && config.recipe.residual == NormArithmetic::ExactBiased
        && config.recipe.positions == PositionRecipe::AlibiModelBias
        && config.recipe.feed_forward == FeedForwardRecipe::GatedGeluUpFirst;
    if !valid {
        return Err(start(
            "JinaCode loader requires the validated H768/12-head/12-layer ALiBi GEGLU config",
        ));
    }
    Ok(JinaCodeConfig {
        hidden_size: config.hidden_size,
        intermediate_size: config.intermediate_size,
        num_attention_heads: config.num_attention_heads,
        num_hidden_layers: config.num_hidden_layers,
        vocab_size: config.vocab_size,
        type_vocab_size: config.type_vocab_size,
        max_position_embeddings: config.max_position_embeddings,
        layer_norm_eps: config.layer_norm_eps,
        recipe: config.recipe,
        packed_qkv_order: PACKED_QKV_ORDER,
        packed_mlp_order: PACKED_MLP_ORDER,
    })
}

fn validate_root(root: &str, names: &BTreeSet<String>) -> Result<(), BackendError> {
    const FOREIGN_ROOTS: [&str; 7] = [
        "bert.",
        "model.",
        "roberta.",
        "xlm-roberta.",
        "camembert.",
        "distilbert.",
        "new.",
    ];
    match root {
        "" => {
            if let Some(name) = names
                .iter()
                .find(|name| FOREIGN_ROOTS.iter().any(|prefix| name.starts_with(*prefix)))
            {
                return Err(start(format!(
                    "incoherent JinaCode checkpoint namespace: {name} uses a foreign root"
                )));
            }
        }
        "bert." => {
            if let Some(name) = names.iter().find(|name| !name.starts_with(root)) {
                return Err(start(format!(
                    "incoherent JinaCode checkpoint namespace: {name} lacks the bert. root"
                )));
            }
        }
        _ => {
            return Err(start(format!(
                "unsupported JinaCode checkpoint namespace root {root}"
            )))
        }
    }
    Ok(())
}

fn f16_source(
    checkpoint: &Checkpoint,
    root: &str,
    relative: &str,
    shape: &[usize],
    used: &mut BTreeSet<String>,
) -> Result<WeightSource, BackendError> {
    let name = format!("{root}{relative}");
    let weight = checkpoint.source(&name, shape)?;
    if weight.dtype() != Dtype::F16 {
        return Err(start(format!(
            "unsupported JinaCode dtype {:?} for {name}; expected F16",
            weight.dtype()
        )));
    }
    used.insert(name);
    Ok(weight)
}

fn dense(
    checkpoint: &Checkpoint,
    root: &str,
    prefix: &str,
    weight_shape: &[usize],
    bias_shape: &[usize],
    used: &mut BTreeSet<String>,
) -> Result<LinearWeights, BackendError> {
    Ok(LinearWeights {
        weight: f16_source(
            checkpoint,
            root,
            &format!("{prefix}.weight"),
            weight_shape,
            used,
        )?,
        bias: f16_source(
            checkpoint,
            root,
            &format!("{prefix}.bias"),
            bias_shape,
            used,
        )?,
    })
}

fn layer_norm(
    checkpoint: &Checkpoint,
    root: &str,
    prefix: &str,
    hidden: usize,
    used: &mut BTreeSet<String>,
) -> Result<LayerNormWeights, BackendError> {
    Ok(LayerNormWeights {
        gamma: f16_source(
            checkpoint,
            root,
            &format!("{prefix}.weight"),
            &[hidden],
            used,
        )?,
        beta: f16_source(checkpoint, root, &format!("{prefix}.bias"), &[hidden], used)?,
    })
}

fn unexpected(name: &str) -> BackendError {
    if name.contains("classifier.") {
        start(format!("unsupported JinaCode classification tensor {name}"))
    } else if name.contains("position_embeddings") {
        start(format!(
            "unsupported JinaCode absolute position tensor {name}; ALiBi has no table"
        ))
    } else if name.contains("rotary") {
        start(format!("unsupported JinaCode rotary tensor {name}"))
    } else if name.ends_with("mlp.up_gated_layer.bias") {
        start(format!(
            "unsupported JinaCode up-projection bias {name}; the packed up/gate GEMM is bias-free"
        ))
    } else if name.ends_with(".bias") {
        start(format!("unexpected JinaCode bias tensor {name}"))
    } else {
        start(format!("unknown JinaCode checkpoint tensor {name}"))
    }
}

fn reject_unexpected(checkpoint: &Checkpoint, used: &BTreeSet<String>) -> Result<(), BackendError> {
    for name in checkpoint.names() {
        if !used.contains(name) {
            return Err(unexpected(name));
        }
    }
    Ok(())
}

pub(crate) fn load(path: &Path) -> Result<JinaCodeWeights, BackendError> {
    let config = RawJinaCodeConfig::load(path)?;
    load_with_config(path, config)
}

pub(crate) fn load_with_config(
    path: &Path,
    config: BertConfig,
) -> Result<JinaCodeWeights, BackendError> {
    let config = descriptor(&config)?;
    validate_modules(path)?;
    let checkpoint = Checkpoint::open(path)?;
    let names = checkpoint.names().cloned().collect::<BTreeSet<_>>();
    let root = select_root(EncoderFamily::JinaCode, &names)?;
    validate_root(root, &names)?;

    let hidden = config.hidden_size;
    let intermediate = config.intermediate_size;
    let up_gated_rows = intermediate
        .checked_mul(2)
        .ok_or_else(|| start("JinaCode packed up/gate row count overflow"))?;
    let mut used = BTreeSet::new();

    let word = f16_source(
        &checkpoint,
        root,
        "embeddings.word_embeddings.weight",
        &[config.vocab_size, hidden],
        &mut used,
    )?;
    let token_type = f16_source(
        &checkpoint,
        root,
        "embeddings.token_type_embeddings.weight",
        &[config.type_vocab_size, hidden],
        &mut used,
    )?;
    let embedding_norm = layer_norm(&checkpoint, root, "embeddings.LayerNorm", hidden, &mut used)?;

    let mut layers = Vec::new();
    layers
        .try_reserve_exact(config.num_hidden_layers)
        .map_err(start)?;
    for layer_index in 0..config.num_hidden_layers {
        let prefix = format!("encoder.layer.{layer_index}");
        let query = dense(
            &checkpoint,
            root,
            &format!("{prefix}.attention.self.query"),
            &[hidden, hidden],
            &[hidden],
            &mut used,
        )?;
        let key = dense(
            &checkpoint,
            root,
            &format!("{prefix}.attention.self.key"),
            &[hidden, hidden],
            &[hidden],
            &mut used,
        )?;
        let value = dense(
            &checkpoint,
            root,
            &format!("{prefix}.attention.self.value"),
            &[hidden, hidden],
            &[hidden],
            &mut used,
        )?;
        let query_norm = layer_norm(
            &checkpoint,
            root,
            &format!("{prefix}.attention.self.layer_norm_q"),
            hidden,
            &mut used,
        )?;
        let key_norm = layer_norm(
            &checkpoint,
            root,
            &format!("{prefix}.attention.self.layer_norm_k"),
            hidden,
            &mut used,
        )?;
        let attention = dense(
            &checkpoint,
            root,
            &format!("{prefix}.attention.output.dense"),
            &[hidden, hidden],
            &[hidden],
            &mut used,
        )?;
        let attention_norm = layer_norm(
            &checkpoint,
            root,
            &format!("{prefix}.attention.output.LayerNorm"),
            hidden,
            &mut used,
        )?;
        let layer_norm_1 = layer_norm(
            &checkpoint,
            root,
            &format!("{prefix}.layer_norm_1"),
            hidden,
            &mut used,
        )?;
        let layer_norm_2 = layer_norm(
            &checkpoint,
            root,
            &format!("{prefix}.layer_norm_2"),
            hidden,
            &mut used,
        )?;
        let mlp_up_gated = f16_source(
            &checkpoint,
            root,
            &format!("{prefix}.mlp.up_gated_layer.weight"),
            &[up_gated_rows, hidden],
            &mut used,
        )?;
        let mlp_down = dense(
            &checkpoint,
            root,
            &format!("{prefix}.mlp.down_layer"),
            &[hidden, intermediate],
            &[hidden],
            &mut used,
        )?;
        layers.push(JinaCodeLayerWeights {
            query,
            key,
            value,
            query_norm,
            key_norm,
            attention,
            attention_norm,
            layer_norm_1,
            layer_norm_2,
            mlp_up_gated,
            mlp_down,
        });
    }

    let pooler_weight = format!("{root}pooler.dense.weight");
    let pooler_bias = format!("{root}pooler.dense.bias");
    let pooler = match (
        checkpoint.contains(&pooler_weight),
        checkpoint.contains(&pooler_bias),
    ) {
        (false, false) => None,
        (true, true) => Some(dense(
            &checkpoint,
            root,
            "pooler.dense",
            &[hidden, hidden],
            &[hidden],
            &mut used,
        )?),
        _ => {
            return Err(start(
                "incomplete JinaCode pooler pair; weight and bias must both be present",
            ))
        }
    };

    reject_unexpected(&checkpoint, &used)?;
    let expected_backbone = config
        .num_hidden_layers
        .checked_mul(TENSORS_PER_LAYER)
        .and_then(|count| count.checked_add(EMBEDDING_TENSORS))
        .ok_or_else(|| start("JinaCode tensor inventory overflow"))?;
    let expected = expected_backbone
        .checked_add(if pooler.is_some() { POOLER_TENSORS } else { 0 })
        .ok_or_else(|| start("JinaCode tensor inventory overflow"))?;
    if used.len() != expected {
        return Err(start(format!(
            "JinaCode tensor inventory mismatch: expected {expected}, used {}",
            used.len()
        )));
    }

    Ok(JinaCodeWeights {
        config,
        word,
        token_type,
        embedding_norm,
        layers,
        _pooler: pooler,
    })
}
