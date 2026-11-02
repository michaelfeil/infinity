// Copyright Infinity Core Contributors
// Licensed under the repository LICENSE.

use std::path::Path;

use serde_json::Value;

use crate::model::{
    read_json, start, BertConfig, Checkpoint, LayerWeights, LinearWeights, WeightSource,
};
use infinity_v3_backend_core::BackendError;

const HEAD_SIZE: usize = 64;
const MAX_LAYERS: usize = 128;
const MAX_POSITIONS: usize = 8192;
const MAX_SPAN: usize = 4096;
const MAX_CLASSIFIER_WIDTH: usize = 32_768;
const MAX_EMBEDDING_SIZE: usize = 32_768;
const MAX_TYPE_VOCAB: usize = 32_768;
const MAX_CONV_KERNEL: usize = 15;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DebertaConvActivation {
    Gelu,
    Tanh,
    Relu,
    Silu,
    GeluTanh,
}

impl DebertaConvActivation {
    pub(crate) fn code(self) -> u32 {
        match self {
            Self::Gelu => 0,
            Self::Tanh => 1,
            Self::Relu => 2,
            Self::Silu => 3,
            Self::GeluTanh => 4,
        }
    }

    fn parse(value: &str) -> Result<Self, BackendError> {
        match value {
            "gelu" => Ok(Self::Gelu),
            "tanh" => Ok(Self::Tanh),
            "relu" => Ok(Self::Relu),
            "silu" => Ok(Self::Silu),
            "gelu_new" | "gelu_pytorch_tanh" => Ok(Self::GeluTanh),
            _ => Err(start(format!(
                "Unsupported DeBERTa conv_act '{value}'; expected gelu, gelu_new, gelu_pytorch_tanh, tanh, relu or silu"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DebertaConvConfig {
    pub(crate) kernel: usize,
    pub(crate) groups: usize,
    pub(crate) activation: DebertaConvActivation,
}

impl DebertaConvConfig {
    pub(crate) fn input_channels_per_group(self, hidden: usize) -> Result<usize, BackendError> {
        hidden
            .checked_div(self.groups)
            .filter(|width| *width > 0 && width.checked_mul(self.groups) == Some(hidden))
            .ok_or_else(|| start("DeBERTa conv_groups must divide hidden_size"))
    }

    pub(crate) fn weight_shape(self, hidden: usize) -> Result<[usize; 3], BackendError> {
        Ok([hidden, self.input_channels_per_group(hidden)?, self.kernel])
    }
}

#[derive(Clone, Debug)]
pub(crate) struct DebertaConfig {
    pub(crate) geometry: BertConfig,
    pub(crate) span: usize,
    pub(crate) max_relative: usize,
    pub(crate) c2p: bool,
    pub(crate) p2c: bool,
    pub(crate) share_att_key: bool,
    pub(crate) relative_layer_norm: bool,
    pub(crate) relative_attention: bool,
    pub(crate) logarithmic: bool,
    pub(crate) embedding_size: usize,
    pub(crate) position_biased_input: bool,
    pub(crate) conv: Option<DebertaConvConfig>,
}

pub(crate) struct DebertaConvWeights {
    pub(crate) weight: WeightSource,
    pub(crate) bias: WeightSource,
    pub(crate) gamma: WeightSource,
    pub(crate) beta: WeightSource,
}

pub(crate) struct RelativeLayerWeights {
    pub(crate) key: Option<LinearWeights>,
    pub(crate) query: Option<LinearWeights>,
}

pub(crate) struct DebertaWeights {
    pub(crate) config: DebertaConfig,
    pub(crate) word: WeightSource,
    pub(crate) position: Option<WeightSource>,
    pub(crate) token_type: Option<WeightSource>,
    pub(crate) embed_proj: Option<WeightSource>,
    pub(crate) gamma: WeightSource,
    pub(crate) beta: WeightSource,
    pub(crate) relative: Option<WeightSource>,
    pub(crate) relative_gamma: Option<WeightSource>,
    pub(crate) relative_beta: Option<WeightSource>,
    pub(crate) conv: Option<DebertaConvWeights>,
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
        if !is_allowed_tensor(name, prefix, &config, layer_count, head_kind) {
            return Err(start(format!(
                "DeBERTa checkpoint contains unsupported or mixed-namespace tensor: {name}"
            )));
        }
    }

    let embedding_size = config.embedding_size;
    let word = checkpoint.source(
        &format!("{prefix}embeddings.word_embeddings.weight"),
        &[vocab, embedding_size],
    )?;
    let position = if config.position_biased_input {
        Some(checkpoint.source(
            &format!("{prefix}embeddings.position_embeddings.weight"),
            &[config.geometry.max_position_embeddings, embedding_size],
        )?)
    } else {
        None
    };
    let token_type = if config.geometry.type_vocab_size > 0 {
        Some(checkpoint.source(
            &format!("{prefix}embeddings.token_type_embeddings.weight"),
            &[config.geometry.type_vocab_size, embedding_size],
        )?)
    } else {
        None
    };
    let embed_proj = if embedding_size != hidden {
        Some(checkpoint.source(
            &format!("{prefix}embeddings.embed_proj.weight"),
            &[hidden, embedding_size],
        )?)
    } else {
        None
    };
    let gamma = checkpoint.source(&format!("{prefix}embeddings.LayerNorm.weight"), &[hidden])?;
    let beta = checkpoint.source(&format!("{prefix}embeddings.LayerNorm.bias"), &[hidden])?;

    let relative = if config.relative_attention {
        let relative_rows = config
            .span
            .checked_mul(2)
            .ok_or_else(|| start("DeBERTa relative embedding row count overflows"))?;
        Some(checkpoint.source(
            &format!("{prefix}encoder.rel_embeddings.weight"),
            &[relative_rows, hidden],
        )?)
    } else {
        None
    };
    let (relative_gamma, relative_beta) = if config.relative_layer_norm {
        let gamma = checkpoint.source(&format!("{prefix}encoder.LayerNorm.weight"), &[hidden])?;
        let beta = checkpoint.source(&format!("{prefix}encoder.LayerNorm.bias"), &[hidden])?;
        (Some(gamma), Some(beta))
    } else {
        (None, None)
    };
    let conv = match config.conv {
        Some(conv) => Some(DebertaConvWeights {
            weight: checkpoint.source(
                &format!("{prefix}encoder.conv.conv.weight"),
                &conv.weight_shape(hidden)?,
            )?,
            bias: checkpoint.source(&format!("{prefix}encoder.conv.conv.bias"), &[hidden])?,
            gamma: checkpoint
                .source(&format!("{prefix}encoder.conv.LayerNorm.weight"), &[hidden])?,
            beta: checkpoint.source(&format!("{prefix}encoder.conv.LayerNorm.bias"), &[hidden])?,
        }),
        None => None,
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
        position,
        token_type,
        embed_proj,
        gamma,
        beta,
        relative,
        relative_gamma,
        relative_beta,
        conv,
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
    config: &DebertaConfig,
    layers: usize,
    head: HeadKind,
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
        | "embeddings.LayerNorm.bias" => true,
        "embeddings.position_embeddings.weight" => config.position_biased_input,
        "embeddings.token_type_embeddings.weight" => config.geometry.type_vocab_size > 0,
        "embeddings.embed_proj.weight" => config.embedding_size != config.geometry.hidden_size,
        "encoder.rel_embeddings.weight" => config.relative_attention,
        "encoder.LayerNorm.weight" | "encoder.LayerNorm.bias" => config.relative_layer_norm,
        "encoder.conv.conv.weight"
        | "encoder.conv.conv.bias"
        | "encoder.conv.LayerNorm.weight"
        | "encoder.conv.LayerNorm.bias" => config.conv.is_some(),
        _ => layer_tensor_allowed(
            rest,
            layers,
            config.relative_attention,
            config.share_att_key,
        ),
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

fn layer_tensor_allowed(
    name: &str,
    layers: usize,
    relative_attention: bool,
    share_att_key: bool,
) -> bool {
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
    if relative_attention
        && !share_att_key
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

    if !matches!(
        (hidden, heads, intermediate),
        (384, 6, 1536) | (768, 12, 3072) | (1024, 16, 4096)
    ) {
        return Err(start(
            "DeBERTa supports only H384/Q6/I1536, H768/Q12/I3072, or H1024/Q16/I4096",
        ));
    }
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

    let embedding_size = match value.get("embedding_size") {
        None | Some(Value::Null) => hidden,
        Some(_) => json_positive_usize(value, "embedding_size", Some(MAX_EMBEDDING_SIZE))?,
    };
    let relative_attention = json_bool(value, "relative_attention", false)?;
    let share_att_key = if relative_attention {
        json_bool(value, "share_att_key", false)?
    } else {
        false
    };
    let position_biased_input = json_bool(value, "position_biased_input", true)?;
    let type_vocab = json_i64(value, "type_vocab_size", Some(0))?;
    let type_vocab_size = usize::try_from(type_vocab)
        .ok()
        .filter(|size| *size <= MAX_TYPE_VOCAB)
        .ok_or_else(|| {
            start(format!(
                "DeBERTa type_vocab_size {type_vocab} is outside 0..={MAX_TYPE_VOCAB}"
            ))
        })?;

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
    if value
        .get("pooler_hidden_size")
        .filter(|entry| !entry.is_null())
        .is_some()
    {
        let pooler_hidden_size = json_positive_usize(value, "pooler_hidden_size", None)?;
        require_exact("pooler_hidden_size", pooler_hidden_size, hidden)?;
    }

    let norm_rel_ebd = json_str(value, "norm_rel_ebd", Some("none"))?;
    let mut relative_layer_norm = false;
    for token in norm_rel_ebd.split('|') {
        match token.trim().to_ascii_lowercase().as_str() {
            "" | "none" => {}
            "layer_norm" => relative_layer_norm = true,
            other => {
                return Err(start(format!(
                    "Unsupported DeBERTa norm_rel_ebd token '{other}'"
                )));
            }
        }
    }

    let conv_kernel_size = json_i64(value, "conv_kernel_size", Some(0))?;
    let conv_kernel_size = usize::try_from(conv_kernel_size).map_err(|_| {
        start(format!(
            "DeBERTa conv_kernel_size {conv_kernel_size} must be non-negative"
        ))
    })?;
    if conv_kernel_size != 0 && hidden != 768 {
        return Err(start("DeBERTa convolution currently requires H768"));
    }
    let conv = if conv_kernel_size == 0 {
        None
    } else {
        if conv_kernel_size > MAX_CONV_KERNEL || conv_kernel_size % 2 == 0 {
            return Err(start(format!(
                "DeBERTa conv_kernel_size {conv_kernel_size} must be odd and at most {MAX_CONV_KERNEL}"
            )));
        }
        let conv_groups = json_i64(value, "conv_groups", Some(1))?;
        let conv_groups = usize::try_from(conv_groups).map_err(|_| {
            start(format!(
                "DeBERTa conv_groups {conv_groups} must be positive"
            ))
        })?;
        if conv_groups == 0 || hidden % conv_groups != 0 {
            return Err(start(format!(
                "DeBERTa conv_groups {conv_groups} must divide hidden_size {hidden}"
            )));
        }
        if !json_bool(value, "conv_bias", true)? {
            return Err(start("DeBERTa conv_bias=false is not admitted"));
        }
        let activation = DebertaConvActivation::parse(json_str(value, "conv_act", Some("tanh"))?)?;
        Some(DebertaConvConfig {
            kernel: conv_kernel_size,
            groups: conv_groups,
            activation,
        })
    };
    if json_enabled(value, "embed_proj_bias")? {
        return Err(start(
            "DeBERTa embed_proj_bias=true is not admitted; the native projection is bias-free",
        ));
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

    let (span, max_relative, logarithmic, c2p, p2c) = if relative_attention {
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
            return Err(start(
                "DeBERTa relative attention requires at least one of c2p or p2c",
            ));
        }
        (span, max_relative, buckets > 0, c2p, p2c)
    } else {
        (0, 0, false, false, false)
    };

    let mut geometry =
        BertConfig::deberta_geometry(vocab, layers, max_positions, eps, architectures)?;
    geometry.hidden_size = hidden;
    geometry.num_attention_heads = heads;
    geometry.intermediate_size = intermediate;
    geometry.type_vocab_size = type_vocab_size;
    geometry.configured_type_vocab_size = Some(type_vocab_size);
    geometry.position_biased_input = position_biased_input;

    Ok(ParsedConfig {
        config: DebertaConfig {
            geometry,
            span,
            max_relative,
            c2p,
            p2c,
            share_att_key,
            relative_layer_norm,
            relative_attention,
            logarithmic,
            embedding_size,
            position_biased_input,
            conv,
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
    let mut tokens: Vec<String> = Vec::new();
    match entry {
        Value::String(flags) => {
            for token in flags.to_ascii_lowercase().split('|') {
                let token = token.trim();
                if token.is_empty() {
                    return Err(start("DeBERTa pos_att_type contains an empty token"));
                }
                tokens.push(token.to_owned());
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
                tokens.push(token.to_owned());
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
        match token.as_str() {
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

    const MEDICAL_NER_CONFIG: &str = include_str!("../tests/fixtures/medical-ner-config.json");
    const MEDICAL_NER_CONFIG_SHA256: &str =
        "95ac7e777dd3d118d09dce4f66beffc1a17e428edcf3c359849a26ade02f3936";

    fn actual_medical_ner_config() -> Value {
        serde_json::from_str(MEDICAL_NER_CONFIG).expect("parse pinned Medical-NER config")
    }

    #[test]
    fn pinned_mixedbread_rerankers_admit_their_actual_three_geometries() {
        for (bytes, hidden, heads, intermediate) in [
            (
                include_str!("../tests/fixtures/mixedbread-rerank-xsmall-config.json"),
                384,
                6,
                1536,
            ),
            (
                include_str!("../tests/fixtures/mixedbread-rerank-base-config.json"),
                768,
                12,
                3072,
            ),
            (
                include_str!("../tests/fixtures/mixedbread-rerank-large-config.json"),
                1024,
                16,
                4096,
            ),
        ] {
            let value: Value = serde_json::from_str(bytes).unwrap();
            let parsed = parse_ok(&value);
            assert_eq!(
                (
                    parsed.hidden,
                    parsed.config.geometry.num_attention_heads,
                    parsed.intermediate
                ),
                (hidden, heads, intermediate)
            );
            assert!(parsed.config.relative_attention);
            assert!(parsed.config.relative_layer_norm);
            assert!(parsed.config.conv.is_none());
            assert_eq!(parsed.config.embedding_size, hidden);
            let mut wrong = value.clone();
            wrong["num_attention_heads"] = json!(heads + 1);
            assert_rejected(&wrong);
        }
    }

    #[test]
    fn wide_relative_checkpoint_headers_keep_projection_and_head_input_widths() {
        for (hidden, heads, intermediate) in [(384, 6, 1536), (1024, 16, 4096)] {
            let fixture = header_fixture_with(
                |c| {
                    apply(
                        c,
                        json!({"hidden_size":hidden,"num_attention_heads":heads,"intermediate_size":intermediate}),
                    )
                },
                |shapes| {
                    for shape in shapes.values_mut() {
                        for dim in shape.iter_mut() {
                            if *dim == 768 {
                                *dim = hidden;
                            } else if *dim == 3072 {
                                *dim = intermediate;
                            }
                        }
                    }
                },
            );
            let host = load(&fixture.0).unwrap();
            assert_eq!(host.config.geometry.hidden_size, hidden);
            for projection in [
                &host.layers[0].query,
                &host.layers[0].key,
                &host.layers[0].value,
            ] {
                assert_eq!(projection.weight.shape(), &[hidden, hidden]);
            }
            assert_eq!(host.relative.as_ref().unwrap().shape(), &[512, hidden]);
            assert_eq!(
                host.head.as_ref().unwrap().classifier.weight.shape(),
                &[1, hidden]
            );
        }
    }

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

    fn write_header_fixture(
        config_bytes: &[u8],
        shapes: std::collections::BTreeMap<String, Vec<usize>>,
    ) -> HeaderFixture {
        use std::io::Write;
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "oxide-deberta-header-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        std::fs::write(path.join("config.json"), config_bytes).unwrap();
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

    fn header_fixture_with(
        edit_config: impl FnOnce(&mut Value),
        edit: impl FnOnce(&mut std::collections::BTreeMap<String, Vec<usize>>),
    ) -> HeaderFixture {
        let mut config = with(
            base_config(),
            json!({"architectures":["DebertaV2ForSequenceClassification"],"norm_rel_ebd":"layer_norm","position_buckets":256}),
        );
        edit_config(&mut config);
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
        write_header_fixture(&serde_json::to_vec(&config).unwrap(), shapes)
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
        assert_eq!(host.relative.as_ref().unwrap().shape(), &[512, 768]);
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
    fn full_embedding_projection_type_position_and_convolution_weights_load() {
        let fixture = header_fixture_with(
            |config| {
                apply(
                    config,
                    json!({
                        "position_biased_input": true,
                        "type_vocab_size": 2,
                        "embedding_size": 96,
                        "conv_kernel_size": 3,
                        "conv_groups": 2,
                        "conv_act": "gelu"
                    }),
                )
            },
            |shapes| {
                shapes.insert(
                    "deberta.embeddings.word_embeddings.weight".into(),
                    vec![1000, 96],
                );
                shapes.insert(
                    "deberta.embeddings.position_embeddings.weight".into(),
                    vec![512, 96],
                );
                shapes.insert(
                    "deberta.embeddings.token_type_embeddings.weight".into(),
                    vec![2, 96],
                );
                shapes.insert("deberta.embeddings.embed_proj.weight".into(), vec![768, 96]);
                shapes.insert("deberta.encoder.conv.conv.weight".into(), vec![768, 384, 3]);
                shapes.insert("deberta.encoder.conv.conv.bias".into(), vec![768]);
                shapes.insert("deberta.encoder.conv.LayerNorm.weight".into(), vec![768]);
                shapes.insert("deberta.encoder.conv.LayerNorm.bias".into(), vec![768]);
            },
        );
        let host = load(&fixture.0).unwrap();
        assert!(host.config.relative_attention);
        assert!(host.config.position_biased_input);
        assert_eq!(host.config.embedding_size, 96);
        assert_eq!(host.config.geometry.type_vocab_size, 2);
        assert_eq!(host.word.shape(), &[1000, 96]);
        assert_eq!(host.position.as_ref().unwrap().shape(), &[512, 96]);
        assert_eq!(host.token_type.as_ref().unwrap().shape(), &[2, 96]);
        assert_eq!(host.embed_proj.as_ref().unwrap().shape(), &[768, 96]);
        assert_eq!(host.conv.as_ref().unwrap().weight.shape(), &[768, 384, 3]);
        assert_eq!(host.conv.as_ref().unwrap().bias.shape(), &[768]);
        assert!(host.relative.is_some());
        let head = host.head.unwrap();
        assert!(head.pooler.is_some());
        assert_eq!(head.width, 1);
    }

    #[test]
    fn malformed_embedding_and_convolution_checkpoint_shapes_are_rejected() {
        for (name, shape) in [
            ("deberta.embeddings.word_embeddings.weight", vec![1000, 95]),
            (
                "deberta.embeddings.position_embeddings.weight",
                vec![511, 96],
            ),
            (
                "deberta.embeddings.token_type_embeddings.weight",
                vec![1, 96],
            ),
            ("deberta.embeddings.embed_proj.weight", vec![768, 95]),
            ("deberta.encoder.conv.conv.weight", vec![768, 384, 4]),
            ("deberta.encoder.conv.conv.bias", vec![767]),
            ("deberta.encoder.conv.LayerNorm.weight", vec![767]),
        ] {
            let fixture = full_embedding_fixture(|shapes| {
                shapes.insert(name.into(), shape.clone());
            });
            assert!(load(&fixture.0).is_err(), "wrong shape for {name}");
        }
        for name in [
            "deberta.embeddings.position_embeddings.weight",
            "deberta.embeddings.token_type_embeddings.weight",
            "deberta.embeddings.embed_proj.weight",
            "deberta.encoder.conv.conv.weight",
            "deberta.encoder.conv.conv.bias",
            "deberta.encoder.conv.LayerNorm.bias",
        ] {
            let fixture = full_embedding_fixture(|shapes| {
                shapes.remove(name);
            });
            assert!(load(&fixture.0).is_err(), "missing {name}");
        }
        let fixture = full_embedding_fixture(|shapes| {
            shapes.insert("deberta.embeddings.embed_proj.bias".into(), vec![768]);
        });
        assert!(load(&fixture.0).is_err());
    }

    #[test]
    fn non_relative_absolute_position_and_type_checkpoint_loads_without_relative_table() {
        let fixture = header_fixture_with(
            |config| {
                apply(
                    config,
                    json!({
                        "relative_attention": false,
                        "pos_att_type": [],
                        "position_biased_input": true,
                        "type_vocab_size": 2
                    }),
                )
            },
            |shapes| {
                shapes.remove("deberta.encoder.rel_embeddings.weight");
                shapes.insert(
                    "deberta.embeddings.position_embeddings.weight".into(),
                    vec![512, 768],
                );
                shapes.insert(
                    "deberta.embeddings.token_type_embeddings.weight".into(),
                    vec![2, 768],
                );
            },
        );
        let host = load(&fixture.0).unwrap();
        assert!(!host.config.relative_attention);
        assert!(host.config.position_biased_input);
        assert_eq!(host.config.geometry.type_vocab_size, 2);
        assert!(host.position.is_some());
        assert!(host.token_type.is_some());
        assert!(host.relative.is_none());
        assert!(host
            .relative_layers
            .iter()
            .all(|layer| layer.key.is_none() && layer.query.is_none()));
    }

    fn full_embedding_fixture(
        edit: impl FnOnce(&mut std::collections::BTreeMap<String, Vec<usize>>),
    ) -> HeaderFixture {
        header_fixture_with(
            |config| {
                apply(
                    config,
                    json!({
                        "position_biased_input": true,
                        "type_vocab_size": 2,
                        "embedding_size": 96,
                        "conv_kernel_size": 3,
                        "conv_groups": 2,
                        "conv_act": "gelu"
                    }),
                )
            },
            |shapes| {
                shapes.insert(
                    "deberta.embeddings.word_embeddings.weight".into(),
                    vec![1000, 96],
                );
                shapes.insert(
                    "deberta.embeddings.position_embeddings.weight".into(),
                    vec![512, 96],
                );
                shapes.insert(
                    "deberta.embeddings.token_type_embeddings.weight".into(),
                    vec![2, 96],
                );
                shapes.insert("deberta.embeddings.embed_proj.weight".into(), vec![768, 96]);
                shapes.insert("deberta.encoder.conv.conv.weight".into(), vec![768, 384, 3]);
                shapes.insert("deberta.encoder.conv.conv.bias".into(), vec![768]);
                shapes.insert("deberta.encoder.conv.LayerNorm.weight".into(), vec![768]);
                shapes.insert("deberta.encoder.conv.LayerNorm.bias".into(), vec![768]);
                edit(shapes);
            },
        )
    }

    fn medical_ner_header_fixture(
        edit: impl FnOnce(&mut std::collections::BTreeMap<String, Vec<usize>>),
    ) -> HeaderFixture {
        let mut shapes = std::collections::BTreeMap::new();
        shapes.insert(
            "deberta.embeddings.word_embeddings.weight".into(),
            vec![128_100, 768],
        );
        for stem in ["deberta.embeddings.LayerNorm", "deberta.encoder.LayerNorm"] {
            shapes.insert(format!("{stem}.weight"), vec![768]);
            shapes.insert(format!("{stem}.bias"), vec![768]);
        }
        shapes.insert(
            "deberta.encoder.rel_embeddings.weight".into(),
            vec![512, 768],
        );
        for layer in 0..12 {
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
        shapes.insert("classifier.weight".into(), vec![83, 768]);
        shapes.insert("classifier.bias".into(), vec![83]);
        edit(&mut shapes);
        write_header_fixture(MEDICAL_NER_CONFIG.as_bytes(), shapes)
    }

    #[test]
    fn actual_medical_ner_saved_config_matches_pinned_provenance() {
        use sha2::{Digest, Sha256};
        let digest = Sha256::digest(MEDICAL_NER_CONFIG.as_bytes());
        let actual = digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(actual, MEDICAL_NER_CONFIG_SHA256);
    }

    #[test]
    fn actual_medical_ner_saved_config_is_admitted_without_weights_or_cuda() {
        let parsed = parse_ok(&actual_medical_ner_config());
        assert_eq!(parsed.head, HeadKind::Token);
        assert_eq!(parsed.hidden, 768);
        assert_eq!(parsed.vocab, 128_100);
        assert_eq!(parsed.layers, 12);
        assert_eq!(parsed.intermediate, 3072);
        let config = parsed.config;
        assert_eq!(config.geometry.hidden_size, 768);
        assert_eq!(config.geometry.num_hidden_layers, 12);
        assert_eq!(config.geometry.num_attention_heads, 12);
        assert_eq!(config.geometry.intermediate_size, 3072);
        assert_eq!(config.geometry.vocab_size, 128_100);
        assert_eq!(config.geometry.max_position_embeddings, 512);
        assert_eq!(config.geometry.type_vocab_size, 0);
        assert!(!config.geometry.position_biased_input);
        assert_eq!(config.embedding_size, 768);
        assert!(config.relative_attention);
        assert!(config.share_att_key);
        assert!(config.c2p);
        assert!(config.p2c);
        assert_eq!(config.span, 256);
        assert_eq!(config.max_relative, 512);
        assert!(config.logarithmic);
        assert!(config.relative_layer_norm);
        assert!(config.conv.is_none());
    }

    #[test]
    fn actual_medical_ner_checkpoint_headers_admit_without_payload_reads() {
        let fixture = medical_ner_header_fixture(|_| {});
        let host = load(&fixture.0).expect("Medical-NER metadata admission");
        assert_eq!(
            host.config.geometry.family,
            crate::model::EncoderFamily::Deberta
        );
        assert_eq!(host.layers.len(), 12);
        assert_eq!(host.relative_layers.len(), 12);
        assert_eq!(host.word.shape(), &[128_100, 768]);
        assert!(host.position.is_none());
        assert!(host.token_type.is_none());
        assert!(host.embed_proj.is_none());
        assert!(host.conv.is_none());
        assert_eq!(host.relative.as_ref().unwrap().shape(), &[512, 768]);
        let head = host.head.expect("token classifier head");
        assert!(head.tokenwise);
        assert!(head.pooler.is_none());
        assert_eq!(head.width, 83);
        assert_eq!(head.classifier.weight.shape(), &[83, 768]);
        assert_eq!(head.classifier.bias.shape(), &[83]);
    }

    #[test]
    fn actual_medical_ner_wrong_widths_and_shapes_are_rejected_before_weights() {
        for patch in [
            json!({"hidden_size": 384}),
            json!({"hidden_size": 1536}),
            json!({"num_attention_heads": 6}),
            json!({"num_attention_heads": 24}),
            json!({"intermediate_size": 1536}),
            json!({"intermediate_size": 768}),
            json!({"attention_head_size": 32}),
            json!({"attention_head_size": 128}),
            json!({"vocab_size": 0}),
            json!({"num_hidden_layers": 0}),
            json!({"norm_rel_ebd": "unsupported"}),
        ] {
            assert_rejected(&with(actual_medical_ner_config(), patch));
        }
        for (name, shape) in [
            (
                "deberta.embeddings.word_embeddings.weight",
                vec![128_100, 767],
            ),
            ("deberta.encoder.rel_embeddings.weight", vec![511, 768]),
            ("deberta.embeddings.LayerNorm.weight", vec![767]),
            (
                "deberta.encoder.layer.0.attention.self.query_proj.weight",
                vec![768, 767],
            ),
            (
                "deberta.encoder.layer.11.output.dense.weight",
                vec![767, 3072],
            ),
            ("classifier.weight", vec![83, 767]),
        ] {
            let fixture = medical_ner_header_fixture(|shapes| {
                shapes.insert(name.into(), shape.clone());
            });
            assert!(load(&fixture.0).is_err(), "wrong shape for {name}");
        }
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
        assert_eq!(
            host.relative.as_ref().unwrap().shape(),
            &[2 * host.config.span, 768]
        );
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
        for mode in ["unknown", "layer_norm|unknown", "NONE|bogus"] {
            assert_rejected(&with(base_config(), json!({"norm_rel_ebd": mode})));
        }
        let parsed = parse_ok(&with(base_config(), json!({"norm_rel_ebd": "NONE"})));
        assert!(!parsed.config.relative_layer_norm);
        for mode in [" Layer_Norm ", "none|Layer_Norm"] {
            let parsed = parse_ok(&with(base_config(), json!({"norm_rel_ebd": mode})));
            assert!(parsed.config.relative_layer_norm, "mode {mode}");
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
            json!("C2P|unknown"),
        ] {
            assert_rejected(&with(base_config(), json!({"pos_att_type": bad})));
        }
        let parsed = parse_ok(&with(base_config(), json!({"pos_att_type": "C2P|P2C"})));
        assert!(parsed.config.c2p);
        assert!(parsed.config.p2c);
    }

    #[test]
    fn unsupported_arithmetic_and_decoder_modifiers_are_rejected() {
        for patch in [
            json!({"hidden_act": "gelu_new"}),
            json!({"hidden_act": "relu"}),
            json!({"pooler_hidden_act": "relu"}),
            json!({"embed_proj_bias": true}),
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
        let parsed = parse_ok(&without(base_config(), "relative_attention"));
        assert!(!parsed.config.relative_attention);
        assert!(!parsed.config.c2p);
        assert!(!parsed.config.p2c);
        let parsed = parse_ok(&without(base_config(), "position_biased_input"));
        assert!(parsed.config.position_biased_input);
    }

    #[test]
    fn embedding_projection_position_type_and_convolution_variants_are_admitted() {
        let parsed = parse_ok(&with(
            base_config(),
            json!({
                "position_biased_input": true,
                "type_vocab_size": 2,
                "embedding_size": 96,
                "conv_kernel_size": 3,
                "conv_groups": 2,
                "conv_act": "gelu"
            }),
        ));
        assert!(parsed.config.relative_attention);
        assert_eq!(parsed.config.geometry.type_vocab_size, 2);
        assert_eq!(parsed.config.embedding_size, 96);
        assert!(parsed.config.position_biased_input);
        assert_eq!(
            parsed.config.conv,
            Some(DebertaConvConfig {
                kernel: 3,
                groups: 2,
                activation: DebertaConvActivation::Gelu,
            })
        );
        assert_eq!(
            parsed.config.conv.unwrap().weight_shape(768).unwrap(),
            [768, 384, 3]
        );

        let parsed = parse_ok(&with(
            base_config(),
            json!({"relative_attention": false, "pos_att_type": []}),
        ));
        assert!(!parsed.config.relative_attention);
        assert_eq!(parsed.config.span, 0);
        assert!(!parsed.config.c2p);
        assert!(!parsed.config.p2c);

        let parsed = parse_ok(&with(base_config(), json!({"conv_kernel_size": 3})));
        assert_eq!(
            parsed.config.conv.unwrap().activation,
            DebertaConvActivation::Tanh
        );
        assert_eq!(parsed.config.embedding_size, 768);
        assert!(!parsed.config.position_biased_input);
        for name in ["gelu_new", "gelu_pytorch_tanh"] {
            let parsed = parse_ok(&with(
                base_config(),
                json!({"conv_kernel_size": 3, "conv_act": name}),
            ));
            assert_eq!(
                parsed.config.conv.unwrap().activation,
                DebertaConvActivation::GeluTanh
            );
        }
    }

    #[test]
    fn malformed_embedding_convolution_and_pooler_geometry_is_rejected() {
        for patch in [
            json!({"embedding_size": 0}),
            json!({"embedding_size": 32769}),
            json!({"conv_kernel_size": -2}),
            json!({"conv_kernel_size": 2}),
            json!({"conv_kernel_size": 17}),
            json!({"conv_kernel_size": 3, "conv_groups": 0}),
            json!({"conv_kernel_size": 3, "conv_groups": 5}),
            json!({"conv_kernel_size": 3, "conv_act": "mish"}),
            json!({"conv_kernel_size": 3, "conv_act": "linear"}),
            json!({"conv_kernel_size": 3, "conv_bias": false}),
            json!({"embed_proj_bias": true}),
            json!({"pooler_hidden_size": 384}),
            json!({"type_vocab_size": -1}),
            json!({"type_vocab_size": 32769}),
        ] {
            assert_rejected(&with(base_config(), patch));
        }
        parse_ok(&with(
            base_config(),
            json!({"conv_kernel_size": 5, "conv_groups": 8, "conv_act": "silu"}),
        ));
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
        let shared = parse_ok(&base_config()).config;
        let dedicated = parse_ok(&with(base_config(), json!({"share_att_key": false}))).config;
        let normalized =
            parse_ok(&with(base_config(), json!({"norm_rel_ebd": "layer_norm"}))).config;
        let full = parse_ok(&with(
            base_config(),
            json!({
                "position_biased_input": true,
                "type_vocab_size": 2,
                "embedding_size": 96,
                "conv_kernel_size": 3,
                "conv_groups": 2,
                "conv_act": "gelu"
            }),
        ))
        .config;
        assert!(is_allowed_tensor(
            "embeddings.word_embeddings.weight",
            "",
            &shared,
            2,
            sequence
        ));
        assert!(is_allowed_tensor(
            "deberta.embeddings.word_embeddings.weight",
            "deberta.",
            &shared,
            2,
            sequence
        ));
        assert!(!is_allowed_tensor(
            "embeddings.word_embeddings.weight",
            "deberta.",
            &shared,
            2,
            sequence
        ));
        assert!(!is_allowed_tensor(
            "deberta.embeddings.word_embeddings.weight",
            "",
            &shared,
            2,
            sequence
        ));
        assert!(is_allowed_tensor(
            "deberta.encoder.rel_embeddings.weight",
            "deberta.",
            &shared,
            2,
            sequence
        ));
        assert!(!is_allowed_tensor(
            "deberta.encoder.LayerNorm.weight",
            "deberta.",
            &shared,
            2,
            sequence
        ));
        assert!(is_allowed_tensor(
            "deberta.encoder.LayerNorm.weight",
            "deberta.",
            &normalized,
            2,
            sequence
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
                !is_allowed_tensor(name, "deberta.", &shared, 2, sequence),
                "name should be rejected: {name}"
            );
        }

        for name in [
            "deberta.embeddings.position_embeddings.weight",
            "deberta.embeddings.token_type_embeddings.weight",
            "deberta.embeddings.embed_proj.weight",
            "deberta.encoder.conv.conv.weight",
            "deberta.encoder.conv.conv.bias",
            "deberta.encoder.conv.LayerNorm.weight",
            "deberta.encoder.conv.LayerNorm.bias",
        ] {
            assert!(
                is_allowed_tensor(name, "deberta.", &full, 2, sequence),
                "name should be allowed for the full recipe: {name}"
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
                is_allowed_tensor(name, "deberta.", &shared, 2, sequence),
                "name should be allowed: {name}"
            );
        }

        assert!(is_allowed_tensor(
            "classifier.weight",
            "deberta.",
            &shared,
            2,
            sequence
        ));
        assert!(is_allowed_tensor(
            "pooler.dense.bias",
            "deberta.",
            &shared,
            2,
            sequence
        ));
        assert!(!is_allowed_tensor(
            "classifier.weight",
            "deberta.",
            &shared,
            2,
            base
        ));
        assert!(!is_allowed_tensor(
            "pooler.dense.weight",
            "deberta.",
            &shared,
            2,
            base
        ));
        assert!(!is_allowed_tensor(
            "deberta.classifier.weight",
            "deberta.",
            &shared,
            2,
            sequence
        ));
        assert!(is_allowed_tensor(
            "deberta.encoder.layer.0.attention.self.pos_key_proj.weight",
            "deberta.",
            &dedicated,
            2,
            sequence
        ));
        assert!(is_allowed_tensor(
            "deberta.encoder.layer.1.attention.self.pos_query_proj.bias",
            "deberta.",
            &dedicated,
            2,
            sequence
        ));
        assert!(!is_allowed_tensor(
            "deberta.encoder.layer.0.attention.self.pos_key_proj.weight",
            "deberta.",
            &shared,
            2,
            sequence
        ));
        assert!(!is_allowed_tensor(
            "encoder.layer.0.attention.self.pos_key_proj.weight",
            "deberta.",
            &dedicated,
            2,
            sequence
        ));
    }
}
