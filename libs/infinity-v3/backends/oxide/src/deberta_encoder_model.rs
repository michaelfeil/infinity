// Copyright Infinity Core Contributors
// Licensed under the repository LICENSE.

use std::path::Path;

use serde_json::Value;

use crate::model::{
    read_json, start, BertConfig, Checkpoint, LayerWeights, LinearWeights, WeightSource,
};
use infinity_v3_backend_core::BackendError;

const HIDDEN_SIZE: usize = 768;
const NUM_HEADS: usize = 12;
const HEAD_SIZE: usize = 64;
const INTERMEDIATE_SIZE: usize = 3072;
const MAX_LAYERS: usize = 128;
const MAX_POSITIONS: usize = 8192;
const MAX_SPAN: usize = 4096;
const MAX_CLASSIFIER_WIDTH: usize = 32_768;

#[derive(Clone, Debug)]
pub(crate) struct DebertaConfig {
    pub(crate) geometry: BertConfig,
    pub(crate) span: usize,
    pub(crate) max_relative: usize,
    pub(crate) c2p: bool,
    pub(crate) p2c: bool,
    pub(crate) share_att_key: bool,
    pub(crate) relative_layer_norm: bool,
    pub(crate) logarithmic: bool,
}

pub(crate) struct RelativeLayerWeights {
    pub(crate) key: Option<LinearWeights>,
    pub(crate) query: Option<LinearWeights>,
}

pub(crate) struct DebertaWeights {
    pub(crate) config: DebertaConfig,
    pub(crate) word: WeightSource,
    pub(crate) gamma: WeightSource,
    pub(crate) beta: WeightSource,
    pub(crate) relative: WeightSource,
    pub(crate) relative_gamma: Option<WeightSource>,
    pub(crate) relative_beta: Option<WeightSource>,
    pub(crate) layers: Vec<LayerWeights>,
    pub(crate) relative_layers: Vec<RelativeLayerWeights>,
    pub(crate) head: Option<DebertaHead>,
}

pub(crate) struct DebertaHead {
    pub(crate) pooler: Option<LinearWeights>,
    pub(crate) classifier: LinearWeights,
    pub(crate) width: usize,
    pub(crate) tokenwise: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum HeadKind {
    Base,
    Sequence,
    Token,
}

struct ParsedConfig {
    config: DebertaConfig,
    head: HeadKind,
    vocab: usize,
    layers: usize,
    intermediate: usize,
    hidden: usize,
}

pub(crate) fn load_config(path: &Path) -> Result<DebertaConfig, BackendError> {
    Ok(load_config_full(path)?.config)
}

fn load_config_full(path: &Path) -> Result<ParsedConfig, BackendError> {
    let value = read_json(&path.join("config.json"))?;
    parse_config_value(&value)
}

pub(crate) fn load(path: &Path) -> Result<DebertaWeights, BackendError> {
    let parsed = load_config_full(path)?;
    crate::model::validate_modules(path)?;
    let checkpoint = Checkpoint::open(path)?;
    let ParsedConfig {
        config,
        head: head_kind,
        vocab,
        layers: layer_count,
        intermediate,
        hidden,
    } = parsed;

    let prefixed = checkpoint.contains("deberta.embeddings.word_embeddings.weight");
    let prefix = if prefixed { "deberta." } else { "" };

    for name in checkpoint.names() {
        if !is_allowed_tensor(
            name,
            prefix,
            layer_count,
            config.relative_layer_norm,
            head_kind,
            config.share_att_key,
        ) {
            return Err(start(format!(
                "DeBERTa checkpoint contains unsupported or mixed-namespace tensor: {name}"
            )));
        }
    }

    let word = checkpoint.source(
        &format!("{prefix}embeddings.word_embeddings.weight"),
        &[vocab, hidden],
    )?;
    let gamma = checkpoint.source(&format!("{prefix}embeddings.LayerNorm.weight"), &[hidden])?;
    let beta = checkpoint.source(&format!("{prefix}embeddings.LayerNorm.bias"), &[hidden])?;

    let relative_rows = config
        .span
        .checked_mul(2)
        .ok_or_else(|| start("DeBERTa relative embedding row count overflows"))?;
    let relative = checkpoint.source(
        &format!("{prefix}encoder.rel_embeddings.weight"),
        &[relative_rows, hidden],
    )?;
    let (relative_gamma, relative_beta) = if config.relative_layer_norm {
        let gamma = checkpoint.source(&format!("{prefix}encoder.LayerNorm.weight"), &[hidden])?;
        let beta = checkpoint.source(&format!("{prefix}encoder.LayerNorm.bias"), &[hidden])?;
        (Some(gamma), Some(beta))
    } else {
        (None, None)
    };

    let mut layers = Vec::new();
    layers
        .try_reserve_exact(layer_count)
        .map_err(|_| start("DeBERTa layer table allocation failed"))?;
    let mut relative_layers = Vec::new();
    relative_layers
        .try_reserve_exact(layer_count)
        .map_err(|_| start("DeBERTa relative layer table allocation failed"))?;
    for index in 0..layer_count {
        let block = format!("encoder.layer.{index}.");
        let self_attention = format!("{block}attention.self.");
        let query = load_linear(
            &checkpoint,
            prefix,
            &format!("{self_attention}query_proj"),
            hidden,
            hidden,
        )?;
        let key = load_linear(
            &checkpoint,
            prefix,
            &format!("{self_attention}key_proj"),
            hidden,
            hidden,
        )?;
        let value = load_linear(
            &checkpoint,
            prefix,
            &format!("{self_attention}value_proj"),
            hidden,
            hidden,
        )?;
        let attention = load_linear(
            &checkpoint,
            prefix,
            &format!("{block}attention.output.dense"),
            hidden,
            hidden,
        )?;
        let (attention_gamma, attention_beta) = load_norm(
            &checkpoint,
            prefix,
            &format!("{block}attention.output.LayerNorm"),
            hidden,
        )?;
        let intermediate_dense = load_linear(
            &checkpoint,
            prefix,
            &format!("{block}intermediate.dense"),
            intermediate,
            hidden,
        )?;
        let output = load_linear(
            &checkpoint,
            prefix,
            &format!("{block}output.dense"),
            hidden,
            intermediate,
        )?;
        let (output_gamma, output_beta) = load_norm(
            &checkpoint,
            prefix,
            &format!("{block}output.LayerNorm"),
            hidden,
        )?;
        let relative_key = if config.c2p {
            if config.share_att_key {
                Some(LinearWeights {
                    weight: key.weight.clone(),
                    bias: key.bias.clone(),
                })
            } else {
                Some(load_linear(
                    &checkpoint,
                    prefix,
                    &format!("{self_attention}pos_key_proj"),
                    hidden,
                    hidden,
                )?)
            }
        } else {
            None
        };
        let relative_query = if config.p2c {
            if config.share_att_key {
                Some(LinearWeights {
                    weight: query.weight.clone(),
                    bias: query.bias.clone(),
                })
            } else {
                Some(load_linear(
                    &checkpoint,
                    prefix,
                    &format!("{self_attention}pos_query_proj"),
                    hidden,
                    hidden,
                )?)
            }
        } else {
            None
        };
        relative_layers.push(RelativeLayerWeights {
            key: relative_key,
            query: relative_query,
        });
        layers.push(LayerWeights {
            query,
            key,
            value,
            attention,
            intermediate: intermediate_dense,
            output,
            attention_gamma,
            attention_beta,
            output_gamma,
            output_beta,
        });
    }

    let head = match head_kind {
        HeadKind::Base => {
            if checkpoint.contains("classifier.weight")
                || checkpoint.contains("classifier.bias")
                || checkpoint.contains("pooler.dense.weight")
                || checkpoint.contains("pooler.dense.bias")
            {
                return Err(start(
                    "DeBERTa base model checkpoint must not contain pooler or classifier tensors",
                ));
            }
            None
        }
        HeadKind::Sequence => {
            require_pair(&checkpoint, "pooler.dense.weight", "pooler.dense.bias")?;
            require_pair(&checkpoint, "classifier.weight", "classifier.bias")?;
            let pooler = Some(load_linear(
                &checkpoint,
                "",
                "pooler.dense",
                hidden,
                hidden,
            )?);
            let width = classifier_width(&checkpoint)?;
            let classifier = load_linear(&checkpoint, "", "classifier", width, hidden)?;
            Some(DebertaHead {
                pooler,
                classifier,
                width,
                tokenwise: false,
            })
        }
        HeadKind::Token => {
            require_pair(&checkpoint, "classifier.weight", "classifier.bias")?;
            let width = classifier_width(&checkpoint)?;
            let classifier = load_linear(&checkpoint, "", "classifier", width, hidden)?;
            let pooler = if checkpoint.contains("pooler.dense.weight")
                || checkpoint.contains("pooler.dense.bias")
            {
                require_pair(&checkpoint, "pooler.dense.weight", "pooler.dense.bias")?;
                Some(load_linear(
                    &checkpoint,
                    "",
                    "pooler.dense",
                    hidden,
                    hidden,
                )?)
            } else {
                None
            };
            Some(DebertaHead {
                pooler,
                classifier,
                width,
                tokenwise: true,
            })
        }
    };

    Ok(DebertaWeights {
        config,
        word,
        gamma,
        beta,
        relative,
        relative_gamma,
        relative_beta,
        layers,
        relative_layers,
        head,
    })
}

fn load_linear(
    checkpoint: &Checkpoint,
    prefix: &str,
    stem: &str,
    out_features: usize,
    in_features: usize,
) -> Result<LinearWeights, BackendError> {
    let weight_name = format!("{prefix}{stem}.weight");
    let bias_name = format!("{prefix}{stem}.bias");
    let weight = checkpoint.source(&weight_name, &[out_features, in_features])?;
    let bias = checkpoint.source(&bias_name, &[out_features])?;
    Ok(LinearWeights { weight, bias })
}

fn load_norm(
    checkpoint: &Checkpoint,
    prefix: &str,
    stem: &str,
    hidden: usize,
) -> Result<(WeightSource, WeightSource), BackendError> {
    let gamma = checkpoint.source(&format!("{prefix}{stem}.weight"), &[hidden])?;
    let beta = checkpoint.source(&format!("{prefix}{stem}.bias"), &[hidden])?;
    Ok((gamma, beta))
}

fn require_pair(checkpoint: &Checkpoint, weight: &str, bias: &str) -> Result<(), BackendError> {
    let has_weight = checkpoint.contains(weight);
    let has_bias = checkpoint.contains(bias);
    if has_weight && has_bias {
        Ok(())
    } else if has_weight || has_bias {
        Err(start(format!(
            "DeBERTa head tensor pair is incomplete: {weight} / {bias}"
        )))
    } else {
        Err(start(format!(
            "DeBERTa head tensor pair is missing: {weight} / {bias}"
        )))
    }
}

fn classifier_width(checkpoint: &Checkpoint) -> Result<usize, BackendError> {
    let shape = checkpoint
        .shape("classifier.weight")
        .ok_or_else(|| start("DeBERTa classifier.weight metadata is missing"))?;
    let width = shape
        .first()
        .copied()
        .ok_or_else(|| start("DeBERTa classifier.weight has no first dimension"))?;
    if width == 0 || width > MAX_CLASSIFIER_WIDTH {
        return Err(start(format!(
            "DeBERTa classifier width {width} is outside 1..={MAX_CLASSIFIER_WIDTH}"
        )));
    }
    Ok(width)
}

fn is_allowed_tensor(
    name: &str,
    prefix: &str,
    layers: usize,
    relative_layer_norm: bool,
    head: HeadKind,
    share_att_key: bool,
) -> bool {
    if head_tensor_allowed(name, head) {
        return true;
    }
    let Some(rest) = name.strip_prefix(prefix) else {
        return false;
    };
    match rest {
        "embeddings.word_embeddings.weight"
        | "embeddings.LayerNorm.weight"
        | "embeddings.LayerNorm.bias"
        | "encoder.rel_embeddings.weight" => true,
        "encoder.LayerNorm.weight" | "encoder.LayerNorm.bias" => relative_layer_norm,
        _ => layer_tensor_allowed(rest, layers, share_att_key),
    }
}

fn head_tensor_allowed(name: &str, head: HeadKind) -> bool {
    match head {
        HeadKind::Base => false,
        HeadKind::Sequence => matches!(
            name,
            "pooler.dense.weight" | "pooler.dense.bias" | "classifier.weight" | "classifier.bias"
        ),
        HeadKind::Token => matches!(
            name,
            "pooler.dense.weight" | "pooler.dense.bias" | "classifier.weight" | "classifier.bias"
        ),
    }
}

fn layer_tensor_allowed(name: &str, layers: usize, share_att_key: bool) -> bool {
    let Some(after_encoder) = name.strip_prefix("encoder.layer.") else {
        return false;
    };
    let Some((index, suffix)) = after_encoder.split_once('.') else {
        return false;
    };
    let Ok(index) = index.parse::<usize>() else {
        return false;
    };
    if index >= layers {
        return false;
    }
    if !share_att_key
        && matches!(
            suffix,
            "attention.self.pos_key_proj.weight"
                | "attention.self.pos_key_proj.bias"
                | "attention.self.pos_query_proj.weight"
                | "attention.self.pos_query_proj.bias"
        )
    {
        return true;
    }
    matches!(
        suffix,
        "attention.self.query_proj.weight"
            | "attention.self.query_proj.bias"
            | "attention.self.key_proj.weight"
            | "attention.self.key_proj.bias"
            | "attention.self.value_proj.weight"
            | "attention.self.value_proj.bias"
            | "attention.output.dense.weight"
            | "attention.output.dense.bias"
            | "attention.output.LayerNorm.weight"
            | "attention.output.LayerNorm.bias"
            | "intermediate.dense.weight"
            | "intermediate.dense.bias"
            | "output.dense.weight"
            | "output.dense.bias"
            | "output.LayerNorm.weight"
            | "output.LayerNorm.bias"
    )
}

fn parse_config_value(value: &Value) -> Result<ParsedConfig, BackendError> {
    let model_type = json_str(value, "model_type", None)?;
    if model_type != "deberta-v2" {
        return Err(start(format!(
            "Unsupported DeBERTa model_type '{model_type}'; expected deberta-v2"
        )));
    }

    let (architectures, head) = parse_architectures(value)?;
    let vocab = json_positive_usize(value, "vocab_size", None)?;
    let layers = json_positive_usize(value, "num_hidden_layers", Some(MAX_LAYERS))?;
    let max_positions = json_positive_usize(value, "max_position_embeddings", Some(MAX_POSITIONS))?;
    let hidden = json_positive_usize(value, "hidden_size", None)?;
    let heads = json_positive_usize(value, "num_attention_heads", None)?;
    let intermediate = json_positive_usize(value, "intermediate_size", None)?;

    require_exact("hidden_size", hidden, HIDDEN_SIZE)?;
    require_exact("num_attention_heads", heads, NUM_HEADS)?;
    require_exact("intermediate_size", intermediate, INTERMEDIATE_SIZE)?;
    if hidden % heads != 0 {
        return Err(start(
            "DeBERTa hidden_size is not divisible by num_attention_heads",
        ));
    }
    require_exact("derived attention_head_size", hidden / heads, HEAD_SIZE)?;
    if matches!(
        value.get("attention_head_size"),
        Some(entry) if !entry.is_null()
    ) {
        let supplied = json_positive_usize(value, "attention_head_size", None)?;
        require_exact("attention_head_size", supplied, HEAD_SIZE)?;
    }

    let eps = config_eps(value)?;
    if !eps.is_finite() || eps <= 0.0 {
        return Err(start(format!(
            "DeBERTa layer_norm_eps must be finite and positive, got {eps}"
        )));
    }

    let relative_attention = json_bool(value, "relative_attention", false)?;
    if !relative_attention {
        return Err(start("DeBERTa relative_attention=false is not admitted"));
    }
    let share_att_key = json_bool(value, "share_att_key", false)?;
    let position_biased_input = json_bool(value, "position_biased_input", true)?;
    if position_biased_input {
        return Err(start(
            "DeBERTa position_biased_input=true is not admitted; absolute positions are unsupported",
        ));
    }
    let type_vocab_size = json_i64(value, "type_vocab_size", Some(0))?;
    if type_vocab_size != 0 {
        return Err(start(
            "DeBERTa type_vocab_size must be 0; token type embeddings are unsupported",
        ));
    }

    let hidden_act = json_str(value, "hidden_act", Some("gelu"))?;
    if hidden_act != "gelu" {
        return Err(start(format!(
            "Unsupported DeBERTa hidden_act '{hidden_act}'; expected gelu"
        )));
    }
    let pooler_hidden_act = json_str(value, "pooler_hidden_act", Some("gelu"))?;
    if pooler_hidden_act != "gelu" {
        return Err(start(format!(
            "Unsupported DeBERTa pooler_hidden_act '{pooler_hidden_act}'; expected gelu"
        )));
    }

    let norm_rel_ebd = json_str(value, "norm_rel_ebd", Some("none"))?;
    let relative_layer_norm = match norm_rel_ebd {
        "" | "none" => false,
        "layer_norm" | "none|layer_norm" | "layer_norm|none" => true,
        _ => {
            return Err(start(format!(
                "Unsupported DeBERTa norm_rel_ebd '{norm_rel_ebd}'"
            )));
        }
    };

    let conv_kernel_size = json_i64(value, "conv_kernel_size", Some(0))?;
    if conv_kernel_size != 0 {
        return Err(start("DeBERTa convolution geometry is not admitted"));
    }
    let conv_groups = json_i64(value, "conv_groups", Some(1))?;
    if conv_groups != 1 {
        return Err(start("DeBERTa conv_groups must be 1 in the admitted scope"));
    }
    for key in ["embed_proj", "embed_proj_bias"] {
        if json_enabled(value, key)? {
            return Err(start(format!(
                "DeBERTa embedding projection modifier '{key}' is not admitted"
            )));
        }
    }

    for key in [
        "is_decoder",
        "add_cross_attention",
        "use_cache",
        "scale_attention_softmax_in_fp32",
        "scale_attn_by_inverse_layer_idx",
        "reorder_and_upcast_attn",
        "apply_residual_connection_post_layernorm",
        "use_relative_position_bias",
    ] {
        if json_bool(value, key, false)? {
            return Err(start(format!(
                "DeBERTa numerical/decoder modifier '{key}' is not admitted"
            )));
        }
    }
    let cross_layers = json_i64(value, "num_cross_attention_layers", Some(0))?;
    if cross_layers > 0 {
        return Err(start("DeBERTa cross-attention layers are not admitted"));
    }
    for key in [
        "quantization_config",
        "quant_config",
        "quantize",
        "load_in_4bit",
        "load_in_8bit",
        "use_bitsandbytes",
        "bnb_4bit_quant_type",
    ] {
        if json_enabled(value, key)? {
            return Err(start(format!(
                "DeBERTa quantization modifier '{key}' is not admitted"
            )));
        }
    }

    let max_relative_raw = json_i64(value, "max_relative_positions", Some(-1))?;
    let max_relative_i64 = if max_relative_raw < 1 {
        i64::try_from(max_positions)
            .map_err(|_| start("DeBERTa max_position_embeddings is too large"))?
    } else {
        max_relative_raw
    };
    let max_relative = usize::try_from(max_relative_i64)
        .map_err(|_| start("DeBERTa max_relative_positions must be non-negative"))?;
    let buckets = json_i64(value, "position_buckets", Some(-1))?;
    let span = if buckets > 0 {
        usize::try_from(buckets).map_err(|_| start("DeBERTa position_buckets is too large"))?
    } else {
        max_relative
    };
    if span == 0 || span > MAX_SPAN {
        return Err(start(format!(
            "DeBERTa relative span {span} is outside 1..={MAX_SPAN}"
        )));
    }
    if buckets > 0 {
        if span < 4 {
            return Err(start("DeBERTa logarithmic buckets require span >= 4"));
        }
        let midpoint = span as f64 / 2.0;
        if (max_relative as f64) <= midpoint + 1.0 {
            return Err(start(
                "DeBERTa max_relative_positions must exceed span/2 + 1 for logarithmic buckets",
            ));
        }
    }

    let (c2p, p2c) = parse_pos_att_type(value)?;
    if !c2p && !p2c {
        return Err(start("DeBERTa requires at least one of c2p or p2c"));
    }

    let geometry = BertConfig::deberta_geometry(vocab, layers, max_positions, eps, architectures)?;

    Ok(ParsedConfig {
        config: DebertaConfig {
            geometry,
            span,
            max_relative,
            c2p,
            p2c,
            share_att_key,
            relative_layer_norm,
            logarithmic: buckets > 0,
        },
        head,
        vocab,
        layers,
        intermediate,
        hidden,
    })
}

fn parse_architectures(value: &Value) -> Result<(Vec<String>, HeadKind), BackendError> {
    let entries = value
        .get("architectures")
        .and_then(Value::as_array)
        .ok_or_else(|| start("DeBERTa config must contain an architectures array"))?;
    if entries.len() != 1 {
        return Err(start(
            "DeBERTa config must contain exactly one native architecture",
        ));
    }
    let architecture = entries
        .first()
        .and_then(Value::as_str)
        .ok_or_else(|| start("DeBERTa architecture entry must be a string"))?;
    let head = match architecture {
        "DebertaV2Model" => HeadKind::Base,
        "DebertaV2ForSequenceClassification" => HeadKind::Sequence,
        "DebertaV2ForTokenClassification" => HeadKind::Token,
        _ => {
            return Err(start(format!(
                "Unsupported DeBERTa architecture '{architecture}'"
            )));
        }
    };
    Ok((vec![architecture.to_owned()], head))
}

fn parse_pos_att_type(value: &Value) -> Result<(bool, bool), BackendError> {
    let entry = value
        .get("pos_att_type")
        .ok_or_else(|| start("DeBERTa config is missing pos_att_type"))?;
    let mut tokens = Vec::new();
    match entry {
        Value::String(flags) => {
            for token in flags.split('|') {
                let token = token.trim();
                if token.is_empty() {
                    return Err(start("DeBERTa pos_att_type contains an empty token"));
                }
                tokens.push(token);
            }
        }
        Value::Array(entries) => {
            tokens
                .try_reserve_exact(entries.len())
                .map_err(|_| start("DeBERTa pos_att_type allocation failed"))?;
            for entry in entries {
                let token = entry
                    .as_str()
                    .ok_or_else(|| start("DeBERTa pos_att_type entries must be strings"))?
                    .trim();
                if token.is_empty() {
                    return Err(start("DeBERTa pos_att_type contains an empty token"));
                }
                tokens.push(token);
            }
        }
        _ => {
            return Err(start(
                "DeBERTa pos_att_type must be a '|' separated string or an array",
            ));
        }
    }
    if tokens.is_empty() {
        return Err(start("DeBERTa requires at least one of c2p or p2c"));
    }
    let mut c2p = false;
    let mut p2c = false;
    for token in tokens {
        match token {
            "c2p" => {
                if c2p {
                    return Err(start("Duplicate DeBERTa pos_att_type token c2p"));
                }
                c2p = true;
            }
            "p2c" => {
                if p2c {
                    return Err(start("Duplicate DeBERTa pos_att_type token p2c"));
                }
                p2c = true;
            }
            _ => {
                return Err(start(format!(
                    "Unsupported DeBERTa pos_att_type token '{token}'"
                )));
            }
        }
    }
    Ok((c2p, p2c))
}

fn require_exact(name: &str, actual: usize, expected: usize) -> Result<(), BackendError> {
    if actual == expected {
        Ok(())
    } else {
        Err(start(format!(
            "DeBERTa {name} {actual} is outside the admitted scope; expected {expected}"
        )))
    }
}

fn config_eps(value: &Value) -> Result<f64, BackendError> {
    let primary = value.get("layer_norm_eps").filter(|entry| !entry.is_null());
    let alias = value.get("eps").filter(|entry| !entry.is_null());
    match (primary, alias) {
        (None, None) => Ok(1e-7),
        (Some(entry), None) => json_f64_value(entry, "layer_norm_eps"),
        (None, Some(entry)) => {
            if json_f64_value(entry, "eps")?.to_bits() != 1e-7f64.to_bits() {
                return Err(start(
                    "DeBERTa reads layer_norm_eps; eps alone may only repeat the native default",
                ));
            }
            Ok(1e-7)
        }
        (Some(primary), Some(alias)) => {
            let primary = json_f64_value(primary, "layer_norm_eps")?;
            let alias = json_f64_value(alias, "eps")?;
            if primary.to_bits() != alias.to_bits() {
                return Err(start(
                    "DeBERTa config contains conflicting layer_norm_eps and eps values",
                ));
            }
            Ok(primary)
        }
    }
}

fn json_i64(value: &Value, key: &str, default: Option<i64>) -> Result<i64, BackendError> {
    match value.get(key) {
        None | Some(Value::Null) => {
            default.ok_or_else(|| start(format!("DeBERTa config is missing integer field '{key}'")))
        }
        Some(Value::Number(number)) => {
            if let Some(integer) = number.as_i64() {
                Ok(integer)
            } else if let Some(integer) = number.as_u64() {
                i64::try_from(integer)
                    .map_err(|_| start(format!("DeBERTa field '{key}' is too large")))
            } else if let Some(float) = number.as_f64() {
                if float.is_finite()
                    && float.fract() == 0.0
                    && float >= i64::MIN as f64
                    && float <= i64::MAX as f64
                {
                    Ok(float as i64)
                } else {
                    Err(start(format!("DeBERTa field '{key}' must be an integer")))
                }
            } else {
                Err(start(format!("DeBERTa field '{key}' must be an integer")))
            }
        }
        Some(_) => Err(start(format!("DeBERTa field '{key}' must be an integer"))),
    }
}

fn json_positive_usize(
    value: &Value,
    key: &str,
    maximum: Option<usize>,
) -> Result<usize, BackendError> {
    let raw = json_i64(value, key, None)?;
    let parsed = usize::try_from(raw)
        .map_err(|_| start(format!("DeBERTa field '{key}' must be positive")))?;
    if parsed == 0 {
        return Err(start(format!("DeBERTa field '{key}' must be positive")));
    }
    if let Some(maximum) = maximum {
        if parsed > maximum {
            return Err(start(format!(
                "DeBERTa field '{key}' {parsed} exceeds {maximum}"
            )));
        }
    }
    Ok(parsed)
}

fn json_f64_value(value: &Value, key: &str) -> Result<f64, BackendError> {
    value
        .as_f64()
        .ok_or_else(|| start(format!("DeBERTa field '{key}' must be a number")))
}

fn json_bool(value: &Value, key: &str, default: bool) -> Result<bool, BackendError> {
    match value.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Bool(flag)) => Ok(*flag),
        Some(Value::Number(number)) => match number.as_i64() {
            Some(0) => Ok(false),
            Some(1) => Ok(true),
            _ => Err(start(format!(
                "DeBERTa field '{key}' must be a boolean or 0/1"
            ))),
        },
        Some(_) => Err(start(format!("DeBERTa field '{key}' must be a boolean"))),
    }
}

fn json_str<'a>(
    value: &'a Value,
    key: &str,
    default: Option<&'a str>,
) -> Result<&'a str, BackendError> {
    match value.get(key) {
        None | Some(Value::Null) => {
            default.ok_or_else(|| start(format!("DeBERTa config is missing string field '{key}'")))
        }
        Some(Value::String(text)) => Ok(text.as_str()),
        Some(_) => Err(start(format!("DeBERTa field '{key}' must be a string"))),
    }
}

fn json_enabled(value: &Value, key: &str) -> Result<bool, BackendError> {
    let Some(entry) = value.get(key) else {
        return Ok(false);
    };
    match entry {
        Value::Null => Ok(false),
        Value::Bool(flag) => Ok(*flag),
        Value::Number(number) => {
            if let Some(integer) = number.as_i64() {
                Ok(integer != 0)
            } else if let Some(float) = number.as_f64() {
                Ok(float != 0.0)
            } else {
                Ok(true)
            }
        }
        Value::String(text) => {
            Ok(!text.is_empty() && text != "none" && text != "None" && text != "false")
        }
        Value::Array(entries) => Ok(!entries.is_empty()),
        Value::Object(entries) => Ok(!entries.is_empty()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn base_config() -> Value {
        json!({
            "model_type": "deberta-v2",
            "architectures": ["DebertaV2Model"],
            "vocab_size": 1000,
            "hidden_size": 768,
            "num_hidden_layers": 2,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "relative_attention": true,
            "share_att_key": true,
            "position_biased_input": false,
            "type_vocab_size": 0,
            "pos_att_type": "c2p|p2c"
        })
    }

    fn apply(config: &mut Value, patch: Value) {
        let patch = patch.as_object().unwrap();
        let object = config.as_object_mut().unwrap();
        for (key, value) in patch {
            object.insert(key.clone(), value.clone());
        }
    }

    fn with(mut config: Value, patch: Value) -> Value {
        apply(&mut config, patch);
        config
    }

    fn without(mut config: Value, key: &str) -> Value {
        config.as_object_mut().unwrap().remove(key);
        config
    }

    fn parse_ok(config: &Value) -> ParsedConfig {
        parse_config_value(config).unwrap()
    }

    fn assert_rejected(config: &Value) {
        assert!(
            parse_config_value(config).is_err(),
            "config should have been rejected: {config}"
        );
    }

    struct HeaderFixture(std::path::PathBuf);
    impl Drop for HeaderFixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn header_fixture(
        edit: impl FnOnce(&mut std::collections::BTreeMap<String, Vec<usize>>),
    ) -> HeaderFixture {
        header_fixture_with(|_| {}, edit)
    }

    fn header_fixture_with(
        edit_config: impl FnOnce(&mut Value),
        edit: impl FnOnce(&mut std::collections::BTreeMap<String, Vec<usize>>),
    ) -> HeaderFixture {
        use std::io::Write;
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "oxide-deberta-header-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        let mut config = with(
            base_config(),
            json!({"architectures":["DebertaV2ForSequenceClassification"],"norm_rel_ebd":"layer_norm","position_buckets":256}),
        );
        edit_config(&mut config);
        std::fs::write(
            path.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        let mut shapes = std::collections::BTreeMap::new();
        shapes.insert(
            "deberta.embeddings.word_embeddings.weight".into(),
            vec![1000, 768],
        );
        for stem in ["deberta.embeddings.LayerNorm", "deberta.encoder.LayerNorm"] {
            shapes.insert(format!("{stem}.weight"), vec![768]);
            shapes.insert(format!("{stem}.bias"), vec![768]);
        }
        shapes.insert(
            "deberta.encoder.rel_embeddings.weight".into(),
            vec![512, 768],
        );
        for layer in 0..2 {
            for (stem, output, input) in [
                ("attention.self.query_proj", 768, 768),
                ("attention.self.key_proj", 768, 768),
                ("attention.self.value_proj", 768, 768),
                ("attention.output.dense", 768, 768),
                ("intermediate.dense", 3072, 768),
                ("output.dense", 768, 3072),
            ] {
                shapes.insert(
                    format!("deberta.encoder.layer.{layer}.{stem}.weight"),
                    vec![output, input],
                );
                shapes.insert(
                    format!("deberta.encoder.layer.{layer}.{stem}.bias"),
                    vec![output],
                );
            }
            for stem in ["attention.output.LayerNorm", "output.LayerNorm"] {
                shapes.insert(
                    format!("deberta.encoder.layer.{layer}.{stem}.weight"),
                    vec![768],
                );
                shapes.insert(
                    format!("deberta.encoder.layer.{layer}.{stem}.bias"),
                    vec![768],
                );
            }
        }
        for (stem, output, input) in [("pooler.dense", 768, 768), ("classifier", 1, 768)] {
            shapes.insert(format!("{stem}.weight"), vec![output, input]);
            shapes.insert(format!("{stem}.bias"), vec![output]);
        }
        edit(&mut shapes);
        let mut header = serde_json::Map::new();
        let mut offset = 0usize;
        for (name, shape) in shapes {
            let end = offset + shape.iter().product::<usize>() * 2;
            header.insert(
                name,
                json!({"dtype":"BF16","shape":shape,"data_offsets":[offset,end]}),
            );
            offset = end;
        }
        let bytes = serde_json::to_vec(&header).unwrap();
        let mut file = std::fs::File::create(path.join("model.safetensors")).unwrap();
        file.write_all(&(bytes.len() as u64).to_le_bytes()).unwrap();
        file.write_all(&bytes).unwrap();
        file.set_len(8 + bytes.len() as u64 + offset as u64)
            .unwrap();
        HeaderFixture(path)
    }

    fn add_dedicated(
        shapes: &mut std::collections::BTreeMap<String, Vec<usize>>,
        layer: usize,
        stem: &str,
    ) {
        shapes.insert(
            format!("deberta.encoder.layer.{layer}.attention.self.{stem}.weight"),
            vec![768, 768],
        );
        shapes.insert(
            format!("deberta.encoder.layer.{layer}.attention.self.{stem}.bias"),
            vec![768],
        );
    }

    fn mark_weight(fixture: &HeaderFixture, name: &str, marker: u16) {
        use std::io::{Read, Seek, SeekFrom, Write};
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(fixture.0.join("model.safetensors"))
            .unwrap();
        let mut bytes = [0; 8];
        file.read_exact(&mut bytes).unwrap();
        let length = u64::from_le_bytes(bytes);
        let mut header = vec![0; length as usize];
        file.read_exact(&mut header).unwrap();
        let document: Value = serde_json::from_slice(&header).unwrap();
        let offset = document[name]["data_offsets"][0].as_u64().unwrap();
        file.seek(SeekFrom::Start(8 + length + offset)).unwrap();
        file.write_all(&marker.to_le_bytes()).unwrap();
    }

    fn first_weight(source: &WeightSource) -> u16 {
        let mut element = [0];
        source.read_bf16(0, &mut element).unwrap();
        element[0]
    }

    #[test]
    fn loads_native_shared_projection_names_and_unprefixed_heads_lazily() {
        let fixture = header_fixture(|_| {});
        let host = load(&fixture.0).unwrap();
        assert_eq!(host.layers.len(), 2);
        assert_eq!(host.relative.shape(), &[512, 768]);
        assert!(host.relative_gamma.is_some());
        assert!(host.relative_beta.is_some());
        let head = host.head.unwrap();
        assert_eq!(head.width, 1);
        assert!(head.pooler.is_some());
    }

    #[test]
    fn share_att_key_defaults_and_nonboolean_values_are_policed() {
        let parsed = parse_ok(&with(base_config(), json!({"share_att_key": false})));
        assert!(!parsed.config.share_att_key);
        let parsed = parse_ok(&without(base_config(), "share_att_key"));
        assert!(!parsed.config.share_att_key);
        let parsed = parse_ok(&with(base_config(), json!({"share_att_key": true})));
        assert!(parsed.config.share_att_key);
        for bad in [json!("false"), json!(2), json!({"enabled": false})] {
            assert_rejected(&with(base_config(), json!({"share_att_key": bad})));
        }
    }

    #[test]
    fn shared_relative_projections_clone_content_sources_lazily() {
        let fixture = header_fixture(|_| {});
        for layer in 0..2 {
            for (stem, marker) in [("key_proj", 0x3f80), ("query_proj", 0x4000)] {
                mark_weight(
                    &fixture,
                    &format!("deberta.encoder.layer.{layer}.attention.self.{stem}.weight"),
                    marker,
                );
            }
        }
        let host = load(&fixture.0).unwrap();
        assert!(host.config.share_att_key);
        assert_eq!(host.relative_layers.len(), host.layers.len());
        for (relative, layer) in host.relative_layers.iter().zip(&host.layers) {
            let key = relative.key.as_ref().unwrap();
            let query = relative.query.as_ref().unwrap();
            assert_eq!(key.weight.shape(), layer.key.weight.shape());
            assert_eq!(key.bias.shape(), layer.key.bias.shape());
            assert_eq!(query.weight.shape(), layer.query.weight.shape());
            assert_eq!(query.bias.shape(), layer.query.bias.shape());
            assert_eq!(first_weight(&key.weight), 0x3f80);
            assert_eq!(first_weight(&query.weight), 0x4000);
        }
    }

    #[test]
    fn dedicated_relative_projections_load_both_terms() {
        let fixture = header_fixture_with(
            |config| apply(config, json!({"share_att_key": false})),
            |shapes| {
                for layer in 0..2 {
                    add_dedicated(shapes, layer, "pos_key_proj");
                    add_dedicated(shapes, layer, "pos_query_proj");
                }
            },
        );
        for layer in 0..2 {
            for (stem, marker) in [
                ("key_proj", 0x3f80),
                ("query_proj", 0x4000),
                ("pos_key_proj", 0x4040),
                ("pos_query_proj", 0x4080),
            ] {
                mark_weight(
                    &fixture,
                    &format!("deberta.encoder.layer.{layer}.attention.self.{stem}.weight"),
                    marker,
                );
            }
        }
        let host = load(&fixture.0).unwrap();
        assert!(!host.config.share_att_key);
        assert_eq!(host.relative_layers.len(), host.layers.len());
        for relative in &host.relative_layers {
            let key = relative.key.as_ref().unwrap();
            let query = relative.query.as_ref().unwrap();
            assert_eq!(key.weight.shape(), &[768, 768]);
            assert_eq!(key.bias.shape(), &[768]);
            assert_eq!(query.weight.shape(), &[768, 768]);
            assert_eq!(query.bias.shape(), &[768]);
            assert_eq!(first_weight(&key.weight), 0x4040);
            assert_eq!(first_weight(&query.weight), 0x4080);
        }
    }

    #[test]
    fn dedicated_relative_projections_load_enabled_single_term_only() {
        for (pos_att_type, stem, expect_key, expect_query) in [
            ("c2p", "pos_key_proj", true, false),
            ("p2c", "pos_query_proj", false, true),
        ] {
            let fixture = header_fixture_with(
                |config| {
                    apply(
                        config,
                        json!({"share_att_key": false, "pos_att_type": pos_att_type}),
                    );
                },
                |shapes| {
                    for layer in 0..2 {
                        add_dedicated(shapes, layer, stem);
                    }
                },
            );
            let host = load(&fixture.0).unwrap();
            for relative in &host.relative_layers {
                assert_eq!(relative.key.is_some(), expect_key);
                assert_eq!(relative.query.is_some(), expect_query);
            }
        }
    }

    #[test]
    fn dedicated_disabled_terms_are_admitted_but_not_read() {
        let fixture = header_fixture_with(
            |config| {
                apply(
                    config,
                    json!({"share_att_key": false, "pos_att_type": "c2p"}),
                );
            },
            |shapes| {
                for layer in 0..2 {
                    add_dedicated(shapes, layer, "pos_key_proj");
                }
                shapes.insert(
                    "deberta.encoder.layer.0.attention.self.pos_query_proj.weight".into(),
                    vec![1, 1],
                );
                shapes.insert(
                    "deberta.encoder.layer.0.attention.self.pos_query_proj.bias".into(),
                    vec![1],
                );
            },
        );
        let host = load(&fixture.0).unwrap();
        for relative in &host.relative_layers {
            assert!(relative.key.is_some());
            assert!(relative.query.is_none());
        }
    }

    #[test]
    fn dedicated_relative_projection_pairs_are_required() {
        for removed in [
            "pos_key_proj.weight",
            "pos_key_proj.bias",
            "pos_query_proj.weight",
            "pos_query_proj.bias",
        ] {
            let fixture = header_fixture_with(
                |config| apply(config, json!({"share_att_key": false})),
                |shapes| {
                    for layer in 0..2 {
                        add_dedicated(shapes, layer, "pos_key_proj");
                        add_dedicated(shapes, layer, "pos_query_proj");
                    }
                    shapes.remove(&format!("deberta.encoder.layer.0.attention.self.{removed}"));
                },
            );
            assert!(load(&fixture.0).is_err(), "missing {removed}");
        }
    }

    #[test]
    fn public_backend_rejects_missing_dedicated_weights_and_shapes_before_cuda() {
        for invalid in [
            "pos_key_proj.weight",
            "pos_key_proj.bias",
            "wrong_query_shape",
        ] {
            let fixture = header_fixture_with(
                |config| apply(config, json!({"share_att_key": false})),
                |shapes| {
                    for layer in 0..2 {
                        add_dedicated(shapes, layer, "pos_key_proj");
                        add_dedicated(shapes, layer, "pos_query_proj");
                    }
                    if invalid == "wrong_query_shape" {
                        shapes.insert(
                            "deberta.encoder.layer.0.attention.self.pos_query_proj.weight".into(),
                            vec![768, 767],
                        );
                    } else {
                        shapes.remove(&format!("deberta.encoder.layer.0.attention.self.{invalid}"));
                    }
                },
            );
            let error = crate::OxideBackend::new(
                &fixture.0,
                "bfloat16",
                infinity_v3_backend_core::ModelType::Classifier,
                0,
            )
            .err()
            .expect("malformed dedicated checkpoint rejected before driver")
            .to_string();
            let expected = if invalid == "wrong_query_shape" {
                "pos_query_proj.weight"
            } else {
                invalid
            };
            assert!(
                error.contains(expected),
                "specific metadata rejection expected, got {error}"
            );
        }
    }

    #[test]
    fn dedicated_relative_projection_shapes_are_exact() {
        for (name, shape) in [
            ("pos_key_proj.weight", vec![768, 767]),
            ("pos_key_proj.bias", vec![767]),
            ("pos_query_proj.weight", vec![767, 768]),
            ("pos_query_proj.bias", vec![767]),
        ] {
            let fixture = header_fixture_with(
                |config| apply(config, json!({"share_att_key": false})),
                |shapes| {
                    for layer in 0..2 {
                        add_dedicated(shapes, layer, "pos_key_proj");
                        add_dedicated(shapes, layer, "pos_query_proj");
                    }
                    shapes.insert(
                        format!("deberta.encoder.layer.0.attention.self.{name}"),
                        shape,
                    );
                },
            );
            assert!(load(&fixture.0).is_err(), "wrong shape for {name}");
        }
    }

    #[test]
    fn dedicated_relative_names_do_not_mix_namespaces_or_modes() {
        let shared_with_dedicated = header_fixture(|shapes| {
            add_dedicated(shapes, 0, "pos_key_proj");
        });
        assert!(load(&shared_with_dedicated.0).is_err());

        let unprefixed = header_fixture_with(
            |config| apply(config, json!({"share_att_key": false})),
            |shapes| {
                for layer in 0..2 {
                    add_dedicated(shapes, layer, "pos_key_proj");
                    add_dedicated(shapes, layer, "pos_query_proj");
                }
                shapes.remove("deberta.encoder.layer.0.attention.self.pos_key_proj.weight");
                shapes.insert(
                    "encoder.layer.0.attention.self.pos_key_proj.weight".into(),
                    vec![768, 768],
                );
            },
        );
        assert!(load(&unprefixed.0).is_err());
    }

    #[test]
    fn malformed_checkpoint_headers_fail_before_any_weight_payload_read() {
        let missing = header_fixture(|shapes| {
            shapes.remove("deberta.encoder.layer.1.attention.self.value_proj.bias");
        });
        assert!(load(&missing.0).is_err());
        let wrong = header_fixture(|shapes| {
            shapes.insert(
                "deberta.encoder.rel_embeddings.weight".into(),
                vec![511, 768],
            );
        });
        assert!(load(&wrong.0).is_err());
        let mixed = header_fixture(|shapes| {
            shapes.insert("embeddings.word_embeddings.weight".into(), vec![1000, 768]);
        });
        assert!(load(&mixed.0).is_err());
        let prefixed_head = header_fixture(|shapes| {
            shapes.remove("classifier.bias");
            shapes.insert("deberta.classifier.bias".into(), vec![1]);
        });
        assert!(load(&prefixed_head.0).is_err());
    }

    #[test]
    #[ignore = "CPU-only actual checkpoint admission; requires INFINITY_DEBERTA_MODEL"]
    fn actual_trained_checkpoint_admits_without_cuda_or_payload_reads() {
        let path = std::env::var_os("INFINITY_DEBERTA_MODEL").expect("model env");
        let host = load(Path::new(&path)).expect("actual trained metadata admission");
        assert_eq!(
            host.config.geometry.family,
            crate::model::EncoderFamily::Deberta
        );
        assert_eq!(host.layers.len(), host.config.geometry.num_hidden_layers);
        assert_eq!(host.relative_layers.len(), host.layers.len());
        assert_eq!(host.word.shape(), &[host.config.geometry.vocab_size, 768]);
        assert_eq!(host.relative.shape(), &[2 * host.config.span, 768]);
        let head = host.head.expect("trained classification head");
        assert_eq!(head.width, 1);
        assert!(!head.tokenwise);
        assert!(head.pooler.is_some());
    }

    #[test]
    fn defaults_select_shared_relative_positions() {
        let parsed = parse_ok(&base_config());
        assert_eq!(parsed.config.span, 512);
        assert_eq!(parsed.config.max_relative, 512);
        assert!(parsed.config.c2p);
        assert!(parsed.config.p2c);
        assert!(parsed.config.share_att_key);
        assert!(!parsed.config.relative_layer_norm);
        assert_eq!(parsed.head, HeadKind::Base);
        assert_eq!(parsed.vocab, 1000);
        assert_eq!(parsed.layers, 2);
    }

    #[test]
    fn position_bucket_defaults_and_boundaries() {
        let parsed = parse_ok(&with(base_config(), json!({"max_relative_positions": 64})));
        assert_eq!(parsed.config.max_relative, 64);
        assert_eq!(parsed.config.span, 64);

        let parsed = parse_ok(&with(
            base_config(),
            json!({"max_relative_positions": 0, "position_buckets": 0}),
        ));
        assert_eq!(parsed.config.max_relative, 512);
        assert_eq!(parsed.config.span, 512);

        let parsed = parse_ok(&with(
            base_config(),
            json!({"max_relative_positions": 512, "position_buckets": 64}),
        ));
        assert_eq!(parsed.config.max_relative, 512);
        assert_eq!(parsed.config.span, 64);

        let parsed = parse_ok(&with(base_config(), json!({"position_buckets": 64})));
        assert_eq!(parsed.config.max_relative, 512);
        assert_eq!(parsed.config.span, 64);

        parse_ok(&with(
            base_config(),
            json!({"max_relative_positions": 4, "position_buckets": 4}),
        ));
        assert_rejected(&with(
            base_config(),
            json!({"max_relative_positions": 3, "position_buckets": 4}),
        ));
        parse_ok(&with(
            base_config(),
            json!({"max_relative_positions": 4096, "position_buckets": 4096}),
        ));
        assert_rejected(&with(
            base_config(),
            json!({"max_relative_positions": 4096, "position_buckets": 4097}),
        ));
        assert_rejected(&with(
            base_config(),
            json!({"max_relative_positions": 4097}),
        ));
        assert_rejected(&with(
            base_config(),
            json!({"max_relative_positions": 512, "position_buckets": 3}),
        ));
    }

    #[test]
    fn relative_normalization_modes_are_exact() {
        for mode in ["none", "layer_norm", "none|layer_norm", "layer_norm|none"] {
            let parsed = parse_ok(&with(base_config(), json!({"norm_rel_ebd": mode})));
            assert_eq!(
                parsed.config.relative_layer_norm,
                mode.contains("layer_norm"),
                "mode {mode}"
            );
        }
        let parsed = parse_ok(&with(base_config(), json!({"norm_rel_ebd": ""})));
        assert!(!parsed.config.relative_layer_norm);
        for mode in [
            "none|none",
            "layer_norm|layer_norm",
            "NONE",
            "layer_norm|",
            " layer_norm",
        ] {
            assert_rejected(&with(base_config(), json!({"norm_rel_ebd": mode})));
        }
    }

    #[test]
    fn pos_att_type_accepts_string_and_array_without_duplicates() {
        let parsed = parse_ok(&with(base_config(), json!({"pos_att_type": "c2p"})));
        assert!(parsed.config.c2p);
        assert!(!parsed.config.p2c);

        let parsed = parse_ok(&with(base_config(), json!({"pos_att_type": ["p2c"]})));
        assert!(!parsed.config.c2p);
        assert!(parsed.config.p2c);

        let parsed = parse_ok(&with(
            base_config(),
            json!({"pos_att_type": [" c2p ", "p2c"]}),
        ));
        assert!(parsed.config.c2p);
        assert!(parsed.config.p2c);

        for bad in [
            json!(""),
            json!("|"),
            json!("c2p||p2c"),
            json!("c2p|"),
            json!([]),
            json!(["c2p", ""]),
            json!("c2p|c2p"),
            json!("p2c|p2c"),
            json!("c2p|unknown"),
            json!(7),
            json!(["c2p", 7]),
            json!("C2P"),
        ] {
            assert_rejected(&with(base_config(), json!({"pos_att_type": bad})));
        }
    }

    #[test]
    fn unsupported_arithmetic_and_decoder_modifiers_are_rejected() {
        for patch in [
            json!({"relative_attention": false}),
            json!({"position_biased_input": true}),
            json!({"type_vocab_size": 1}),
            json!({"hidden_act": "gelu_new"}),
            json!({"hidden_act": "relu"}),
            json!({"pooler_hidden_act": "relu"}),
            json!({"conv_kernel_size": 3}),
            json!({"conv_groups": 2}),
            json!({"embed_proj": true}),
            json!({"is_decoder": true}),
            json!({"add_cross_attention": true}),
            json!({"use_cache": true}),
            json!({"num_cross_attention_layers": 1}),
            json!({"quantization_config": {"bits": 4}}),
            json!({"load_in_4bit": true}),
            json!({"scale_attention_softmax_in_fp32": true}),
            json!({"scale_attn_by_inverse_layer_idx": true}),
            json!({"reorder_and_upcast_attn": true}),
            json!({"apply_residual_connection_post_layernorm": true}),
            json!({"use_relative_position_bias": true}),
        ] {
            assert_rejected(&with(base_config(), patch));
        }
        assert_rejected(&without(base_config(), "relative_attention"));
        assert_rejected(&without(base_config(), "position_biased_input"));
    }

    #[test]
    fn malformed_geometry_is_rejected_before_weight_loading() {
        for patch in [
            json!({"eps": 1e-6}),
            json!({"model_type": "deberta"}),
            json!({"hidden_size": 512}),
            json!({"num_attention_heads": 8}),
            json!({"intermediate_size": 2048}),
            json!({"attention_head_size": 32}),
            json!({"attention_head_size": 128}),
            json!({"num_hidden_layers": 0}),
            json!({"num_hidden_layers": 129}),
            json!({"max_position_embeddings": 0}),
            json!({"max_position_embeddings": 8193}),
            json!({"vocab_size": 0}),
            json!({"layer_norm_eps": 0.0}),
            json!({"layer_norm_eps": -1e-7}),
            json!({"layer_norm_eps": "1e-7"}),
            json!({"eps": 1e-7, "layer_norm_eps": 1e-6}),
        ] {
            assert_rejected(&with(base_config(), patch));
        }
        let parsed = parse_ok(&with(base_config(), json!({"attention_head_size": 64})));
        assert_eq!(parsed.config.span, 512);
        let parsed = parse_ok(&with(
            base_config(),
            json!({"layer_norm_eps": 1e-7, "eps": 1e-7}),
        ));
        assert_eq!(parsed.config.span, 512);
    }

    #[test]
    fn harmless_metadata_does_not_change_relative_geometry() {
        let parsed = parse_ok(&with(
            base_config(),
            json!({
                "hidden_dropout_prob": 0.1,
                "attention_probs_dropout_prob": 0.1,
                "initializer_range": 0.02,
                "pad_token_id": 0,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "torch_dtype": "bfloat16",
                "transformers_version": "ignored",
                "_name_or_path": "ignored",
                "id2label": {"0": "a"},
                "label2id": {"a": 0},
                "sentence_transformers": true,
                "legacy": true,
                "position_buckets": 64,
                "max_relative_positions": 512
            }),
        ));
        assert_eq!(parsed.config.span, 64);
        assert_eq!(parsed.config.max_relative, 512);
    }

    #[test]
    fn architectures_select_head_and_are_unambiguous() {
        let cases = [
            ("DebertaV2Model", HeadKind::Base),
            ("DebertaV2ForSequenceClassification", HeadKind::Sequence),
            ("DebertaV2ForTokenClassification", HeadKind::Token),
        ];
        for (architecture, kind) in cases {
            let parsed = parse_ok(&with(
                base_config(),
                json!({"architectures": [architecture]}),
            ));
            assert_eq!(parsed.head, kind);
        }
        for bad in [
            json!({"architectures": []}),
            json!({"architectures": ["DebertaV2Model", "DebertaV2ForSequenceClassification"]}),
            json!({"architectures": ["DebertaV2ForMaskedLM"]}),
            json!({"architectures": "DebertaV2Model"}),
            json!({"architectures": [7]}),
        ] {
            assert_rejected(&with(base_config(), bad));
        }
        assert_rejected(&without(base_config(), "architectures"));
    }

    #[test]
    fn checkpoint_namespace_and_name_families_are_exact() {
        let sequence = HeadKind::Sequence;
        let base = HeadKind::Base;
        assert!(is_allowed_tensor(
            "embeddings.word_embeddings.weight",
            "",
            2,
            false,
            sequence,
            true
        ));
        assert!(is_allowed_tensor(
            "deberta.embeddings.word_embeddings.weight",
            "deberta.",
            2,
            false,
            sequence,
            true
        ));
        assert!(!is_allowed_tensor(
            "embeddings.word_embeddings.weight",
            "deberta.",
            2,
            false,
            sequence,
            true
        ));
        assert!(!is_allowed_tensor(
            "deberta.embeddings.word_embeddings.weight",
            "",
            2,
            false,
            sequence,
            true
        ));
        assert!(is_allowed_tensor(
            "deberta.encoder.rel_embeddings.weight",
            "deberta.",
            2,
            false,
            sequence,
            true
        ));
        assert!(!is_allowed_tensor(
            "deberta.encoder.LayerNorm.weight",
            "deberta.",
            2,
            false,
            sequence,
            true
        ));
        assert!(is_allowed_tensor(
            "deberta.encoder.LayerNorm.weight",
            "deberta.",
            2,
            true,
            sequence,
            true
        ));

        for name in [
            "deberta.embeddings.position_embeddings.weight",
            "deberta.embeddings.token_type_embeddings.weight",
            "deberta.embeddings.embed_proj.weight",
            "deberta.embeddings.embed_proj.bias",
            "deberta.encoder.conv.conv.weight",
            "deberta.encoder.conv.LayerNorm.weight",
            "deberta.encoder.layer.0.attention.self.pos_key_proj.weight",
            "deberta.encoder.layer.0.attention.self.pos_query_proj.weight",
            "deberta.encoder.layer.0.attention.self.query_proj.weight.extra",
            "deberta.encoder.layer.2.attention.self.query_proj.weight",
            "deberta.lm_head.weight",
        ] {
            assert!(
                !is_allowed_tensor(name, "deberta.", 2, true, sequence, true),
                "name should be rejected: {name}"
            );
        }

        for name in [
            "deberta.encoder.layer.0.attention.self.query_proj.weight",
            "deberta.encoder.layer.0.attention.self.key_proj.weight",
            "deberta.encoder.layer.0.attention.self.value_proj.weight",
            "deberta.encoder.layer.1.attention.output.dense.bias",
            "deberta.encoder.layer.1.attention.output.LayerNorm.weight",
            "deberta.encoder.layer.1.intermediate.dense.weight",
            "deberta.encoder.layer.1.output.dense.weight",
            "deberta.encoder.layer.1.output.LayerNorm.bias",
        ] {
            assert!(
                is_allowed_tensor(name, "deberta.", 2, false, sequence, true),
                "name should be allowed: {name}"
            );
        }

        assert!(is_allowed_tensor(
            "classifier.weight",
            "deberta.",
            2,
            false,
            sequence,
            true
        ));
        assert!(is_allowed_tensor(
            "pooler.dense.bias",
            "deberta.",
            2,
            false,
            sequence,
            true
        ));
        assert!(!is_allowed_tensor(
            "classifier.weight",
            "deberta.",
            2,
            false,
            base,
            true
        ));
        assert!(!is_allowed_tensor(
            "pooler.dense.weight",
            "deberta.",
            2,
            false,
            base,
            true
        ));
        assert!(!is_allowed_tensor(
            "deberta.classifier.weight",
            "deberta.",
            2,
            false,
            sequence,
            true
        ));
        assert!(is_allowed_tensor(
            "deberta.encoder.layer.0.attention.self.pos_key_proj.weight",
            "deberta.",
            2,
            false,
            sequence,
            false
        ));
        assert!(is_allowed_tensor(
            "deberta.encoder.layer.1.attention.self.pos_query_proj.bias",
            "deberta.",
            2,
            false,
            sequence,
            false
        ));
        assert!(!is_allowed_tensor(
            "deberta.encoder.layer.0.attention.self.pos_key_proj.weight",
            "deberta.",
            2,
            false,
            sequence,
            true
        ));
        assert!(!is_allowed_tensor(
            "encoder.layer.0.attention.self.pos_key_proj.weight",
            "deberta.",
            2,
            false,
            sequence,
            false
        ));
    }
}
