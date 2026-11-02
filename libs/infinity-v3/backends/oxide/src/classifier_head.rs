// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Bounded trained BERT-family classifier admission and readout.
//!
//! BF16 BERT, RoBERTa, XLM-RoBERTa and CamemBERT sequence/token classifier
//! checkpoints are admitted from their configured backbone geometry and their
//! actual trained head tensors. The ordinary embedding loader continues to
//! reject classifier architectures.

use crate::{
    arg, blocks,
    cublas::Blas,
    head_linear::{HeadLinearPlan, PreparedHeadLinear},
    infer, launch,
    lt::{LtProvider, ShapeError},
    model::{
        BertConfig, Checkpoint, EncoderFamily, FeedForwardRecipe, LinearWeights, NormArithmetic,
        PositionRecipe, WeightSource,
    },
    product,
    tensor::Bf16,
    u32_size,
    upload::WeightUploader,
    WorkKey,
};
use cuda_core::{Context, DeviceBuffer, Function, Stream};
use infinity_v3_backend_core::{BackendError, Batch, Predictions, TokenPredictions};
use std::{
    mem::{size_of, ManuallyDrop},
    path::{Path, PathBuf},
    sync::Arc,
};

const MAX_HEAD_WIDTH: usize = 32_768;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ClassifierTask {
    Sequence,
    Tokens,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReadoutMode {
    Sequence,
    Tokens,
}

/// Native sequence readout uses CLS by default, except ModernBERT whose
/// configured `classifier_pooling` may select mean or last-token reduction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ClassifierPooling {
    First,
    Mean,
    Last,
}

/// The activation stage after the first trained head projection. The BERT
/// dense/out_proj head selects tanh, DeBERTa's context pooler selects exact
/// GELU, and ModernBERT selects its configured tanh-GELU/ReLU/SiLU recipe.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum HeadActivation {
    Tanh,
    GeluErf,
    GeluTanh,
    Relu,
    Silu,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ClassifierReadout {
    pub(crate) task: ClassifierTask,
    pub(crate) width: usize,
}

impl ClassifierReadout {
    pub(crate) fn width(self) -> usize {
        self.width
    }
}

#[derive(Clone, Debug)]
pub(crate) struct HeadLinear {
    pub(crate) weight: WeightSource,
    pub(crate) bias: Option<WeightSource>,
}

impl HeadLinear {
    fn from_biases(weight: WeightSource, bias: WeightSource) -> Self {
        Self {
            weight,
            bias: Some(bias),
        }
    }

    fn input(&self) -> Option<usize> {
        self.weight.shape().get(1).copied()
    }
}

#[derive(Clone, Debug)]
pub(crate) struct HeadNorm {
    pub(crate) gamma: WeightSource,
    pub(crate) beta: Option<WeightSource>,
    pub(crate) eps: f32,
}

#[derive(Clone, Debug)]
pub(crate) struct HeadStage {
    pub(crate) linear: HeadLinear,
    pub(crate) activation: HeadActivation,
    pub(crate) norm: Option<HeadNorm>,
}

#[derive(Clone, Debug)]
pub(crate) struct ClassifierHeadHost {
    pub(crate) readout: ClassifierReadout,
    pub(crate) classifier: HeadLinear,
    pub(crate) hidden: usize,
    pub(crate) first: Option<HeadStage>,
    first_sequence: bool,
    first_tokens: bool,
    requires_first_for_sequence: bool,
    pooling: ClassifierPooling,
}

fn task_for_architecture(family: EncoderFamily, architecture: &str) -> Option<ClassifierTask> {
    match (family, architecture) {
        (EncoderFamily::Bert, "BertForSequenceClassification" | "BertForTokenClassification") => {
            Some(if architecture == "BertForSequenceClassification" {
                ClassifierTask::Sequence
            } else {
                ClassifierTask::Tokens
            })
        }
        (
            EncoderFamily::Roberta,
            "RobertaForSequenceClassification" | "RobertaForTokenClassification",
        ) => Some(if architecture == "RobertaForSequenceClassification" {
            ClassifierTask::Sequence
        } else {
            ClassifierTask::Tokens
        }),
        (
            EncoderFamily::XlmRoberta,
            "XLMRobertaForSequenceClassification" | "XLMRobertaForTokenClassification",
        ) => Some(if architecture == "XLMRobertaForSequenceClassification" {
            ClassifierTask::Sequence
        } else {
            ClassifierTask::Tokens
        }),
        (
            EncoderFamily::Camembert,
            "CamembertForSequenceClassification" | "CamembertForTokenClassification",
        ) => Some(if architecture == "CamembertForSequenceClassification" {
            ClassifierTask::Sequence
        } else {
            ClassifierTask::Tokens
        }),
        (EncoderFamily::Gte, "NewForSequenceClassification") => Some(ClassifierTask::Sequence),
        (EncoderFamily::Gte, "NewForTokenClassification") => Some(ClassifierTask::Tokens),
        (EncoderFamily::Deberta, "DebertaV2ForSequenceClassification") => {
            Some(ClassifierTask::Sequence)
        }
        (EncoderFamily::Deberta, "DebertaV2ForTokenClassification") => Some(ClassifierTask::Tokens),
        _ => None,
    }
}

fn task_for(config: &BertConfig) -> Option<ClassifierTask> {
    let [architecture] = config.architectures.as_slice() else {
        return None;
    };
    task_for_architecture(config.family, architecture)
}

/// Cold task policy, after the classifier constructor validates its geometry.
pub(crate) fn is_classifier_config(config: &BertConfig) -> bool {
    task_for(config).is_some()
}

/// Native DeBERTa binds prediction mode to the trained architecture, and GTE
/// has no token-classification head at all. Ordinary BERT retains its existing
/// dual readout behavior.
pub(crate) fn admit_readout(config: &BertConfig, mode: ReadoutMode) -> Result<(), BackendError> {
    if config.family == EncoderFamily::Gte && mode == ReadoutMode::Tokens {
        return Err(infer("GTE does not support token classification"));
    }
    if config.family != EncoderFamily::Deberta {
        return Ok(());
    }
    let task =
        task_for(config).ok_or_else(|| infer("DeBERTa classifier architecture is missing"))?;
    if matches!(
        (task, mode),
        (ClassifierTask::Sequence, ReadoutMode::Sequence)
            | (ClassifierTask::Tokens, ReadoutMode::Tokens)
    ) {
        Ok(())
    } else {
        Err(infer(
            "DeBERTa prediction mode must match the trained classifier architecture",
        ))
    }
}

fn architecture_name(config: &BertConfig) -> &str {
    config
        .architectures
        .first()
        .map_or("classifier", String::as_str)
}

fn admit_geometry(config: &BertConfig, task: ClassifierTask) -> Result<(), BackendError> {
    if config.family == EncoderFamily::Gte && task == ClassifierTask::Tokens {
        return Err(crate::start("GTE does not support token classification"));
    }
    let (family_ok, recipe_ok) = match config.family {
        EncoderFamily::Bert
        | EncoderFamily::Roberta
        | EncoderFamily::XlmRoberta
        | EncoderFamily::Camembert => (
            true,
            matches!(
                config.recipe.embedding,
                NormArithmetic::ExactBiased | NormArithmetic::RoundedAddBeforeStats
            ) && matches!(
                config.recipe.residual,
                NormArithmetic::ExactBiased | NormArithmetic::RoundedAddBeforeStats
            ) && config.recipe.positions == PositionRecipe::Absolute
                && config.recipe.feed_forward == FeedForwardRecipe::DenseGelu,
        ),
        EncoderFamily::Gte => (
            true,
            config.recipe.embedding == NormArithmetic::AtenEmbedding
                && config.recipe.residual == NormArithmetic::ExactBiased
                && matches!(config.recipe.positions, PositionRecipe::Rotary(_))
                && config.recipe.feed_forward == FeedForwardRecipe::GatedGeluUpFirst,
        ),
        _ => (false, false),
    };
    if !family_ok {
        return Err(crate::start(format!(
            "unsupported {} classifier family",
            architecture_name(config)
        )));
    }
    if !recipe_ok || config.hidden_size > MAX_HEAD_WIDTH {
        return Err(crate::start(format!(
            "unsupported {} classifier geometry or arithmetic recipe",
            architecture_name(config)
        )));
    }
    Ok(())
}

fn classifier_width(checkpoint: &Checkpoint, weight: &str) -> Result<usize, BackendError> {
    let shape = checkpoint
        .shape(weight)
        .ok_or_else(|| crate::start(format!("missing classifier tensor {weight}")))?;
    let [width, _] = shape else {
        return Err(crate::start(format!(
            "classifier tensor {weight} must be rank two"
        )));
    };
    if *width == 0 || *width > MAX_HEAD_WIDTH {
        return Err(crate::start(format!(
            "classifier width {width} is outside 1..={MAX_HEAD_WIDTH}"
        )));
    }
    Ok(*width)
}

fn exact_pair(
    checkpoint: &Checkpoint,
    weight: &str,
    bias: &str,
    hidden: usize,
) -> Result<Option<LinearWeights>, BackendError> {
    if checkpoint.shape(weight) != Some([hidden, hidden].as_slice())
        || checkpoint.shape(bias) != Some([hidden].as_slice())
    {
        return Ok(None);
    }
    Ok(Some(LinearWeights {
        weight: checkpoint.source(weight, &[hidden, hidden])?,
        bias: checkpoint.source(bias, &[hidden])?,
    }))
}

impl ClassifierHeadHost {
    pub(crate) fn from_deberta(head: crate::deberta_encoder_model::DebertaHead) -> Self {
        let hidden = head.classifier.weight.shape().get(1).copied().unwrap_or(0);
        let tokenwise = head.tokenwise;
        Self {
            readout: ClassifierReadout {
                task: if tokenwise {
                    ClassifierTask::Tokens
                } else {
                    ClassifierTask::Sequence
                },
                width: head.width,
            },
            classifier: HeadLinear::from_biases(head.classifier.weight, head.classifier.bias),
            hidden,
            first: head.pooler.map(|pooler| HeadStage {
                linear: HeadLinear::from_biases(pooler.weight, pooler.bias),
                activation: HeadActivation::GeluErf,
                norm: None,
            }),
            first_sequence: !tokenwise,
            first_tokens: false,
            requires_first_for_sequence: !tokenwise,
            pooling: ClassifierPooling::First,
        }
    }

    fn load_bert_head(
        checkpoint: &Checkpoint,
        config: &BertConfig,
        task: ClassifierTask,
    ) -> Result<Self, BackendError> {
        let hidden = config.hidden_size;
        let mlp_names = [
            "classifier.dense.weight",
            "classifier.dense.bias",
            "classifier.out_proj.weight",
            "classifier.out_proj.bias",
        ];
        let present = mlp_names
            .iter()
            .filter(|name| checkpoint.contains(name))
            .count();
        if family_is_gte(config.family) {
            if present != 0 {
                return Err(crate::start(
                    "GTE classifier does not use classifier.dense/out_proj",
                ));
            }
            return Self::load_gte_head(checkpoint, hidden, task);
        }
        if present != 0 {
            if present != mlp_names.len() {
                return Err(crate::start(
                    "classifier dense and out_proj tensors must be complete or absent",
                ));
            }
            let width = classifier_width(checkpoint, "classifier.out_proj.weight")?;
            let first = HeadStage {
                linear: HeadLinear::from_biases(
                    checkpoint.source("classifier.dense.weight", &[hidden, hidden])?,
                    checkpoint.source("classifier.dense.bias", &[hidden])?,
                ),
                activation: HeadActivation::Tanh,
                norm: None,
            };
            let classifier = HeadLinear::from_biases(
                checkpoint.source("classifier.out_proj.weight", &[width, hidden])?,
                checkpoint.source("classifier.out_proj.bias", &[width])?,
            );
            return Ok(Self {
                readout: ClassifierReadout { task, width },
                classifier,
                hidden,
                first: Some(first),
                first_sequence: true,
                first_tokens: true,
                requires_first_for_sequence: false,
                pooling: ClassifierPooling::First,
            });
        }
        let width = classifier_width(checkpoint, "classifier.weight")?;
        let classifier = HeadLinear::from_biases(
            checkpoint.source("classifier.weight", &[width, hidden])?,
            checkpoint.source("classifier.bias", &[width])?,
        );
        // Native's optional classifier pooler namespace is independent of the
        // backbone prefix and is only used by the sequence readout.
        let pooler_weight = "bert.pooler.dense.weight";
        let pooler_bias = "bert.pooler.dense.bias";
        let pooler = match (
            checkpoint.contains(pooler_weight),
            checkpoint.contains(pooler_bias),
        ) {
            (true, true) => Some(HeadStage {
                linear: HeadLinear::from_biases(
                    checkpoint.source(pooler_weight, &[hidden, hidden])?,
                    checkpoint.source(pooler_bias, &[hidden])?,
                ),
                activation: HeadActivation::Tanh,
                norm: None,
            }),
            (false, false) => None,
            _ => {
                return Err(crate::start(
                    "classifier pooler must provide both dense weight and bias",
                ))
            }
        };
        Ok(Self {
            readout: ClassifierReadout { task, width },
            classifier,
            hidden,
            first: pooler,
            first_sequence: true,
            first_tokens: false,
            requires_first_for_sequence: false,
            pooling: ClassifierPooling::First,
        })
    }

    fn load_gte_head(
        checkpoint: &Checkpoint,
        hidden: usize,
        task: ClassifierTask,
    ) -> Result<Self, BackendError> {
        if task == ClassifierTask::Tokens {
            return Err(crate::start("GTE does not support token classification"));
        }
        let width = classifier_width(checkpoint, "classifier.weight")?;
        let classifier_bias = checkpoint
            .contains("classifier.bias")
            .then(|| checkpoint.source("classifier.bias", &[width]))
            .transpose()?;
        let classifier = HeadLinear {
            weight: checkpoint.source("classifier.weight", &[width, hidden])?,
            bias: classifier_bias,
        };
        // Native GTE tries a complete new.pooler, then a complete root pooler.
        // An incomplete preferred namespace falls back rather than failing.
        let pooler = exact_pair(
            checkpoint,
            "new.pooler.dense.weight",
            "new.pooler.dense.bias",
            hidden,
        )?
        .or(exact_pair(
            checkpoint,
            "pooler.dense.weight",
            "pooler.dense.bias",
            hidden,
        )?);
        Ok(Self {
            readout: ClassifierReadout { task, width },
            classifier,
            hidden,
            first: pooler.map(|pooler| HeadStage {
                linear: HeadLinear::from_biases(pooler.weight, pooler.bias),
                activation: HeadActivation::Tanh,
                norm: None,
            }),
            first_sequence: true,
            first_tokens: false,
            requires_first_for_sequence: false,
            pooling: ClassifierPooling::First,
        })
    }

    pub(crate) fn load(checkpoint: &Checkpoint, config: &BertConfig) -> Result<Self, BackendError> {
        let task =
            task_for(config).ok_or_else(|| crate::start("unsupported classifier architecture"))?;
        admit_geometry(config, task)?;
        Self::load_bert_head(checkpoint, config, task)
    }

    /// ModernBERT's sequence head is a configured pooling reduction followed by
    /// a biasless `head.dense`, activation, biasless `head.norm`, and the root
    /// classifier projection. Token readout uses only that root projection,
    /// matching the Native encoder executor.
    pub(crate) fn load_modern(
        checkpoint: &Checkpoint,
        config: &crate::modern_encoder_model::ModernConfig,
    ) -> Result<Self, BackendError> {
        let policy = config
            .classifier
            .as_ref()
            .ok_or_else(|| crate::start("ModernBERT classifier policy is missing"))?;
        let hidden = config.geometry.hidden_size;
        let width = classifier_width(checkpoint, "classifier.weight")?;
        let classifier_bias = checkpoint
            .contains("classifier.bias")
            .then(|| checkpoint.source("classifier.bias", &[width]))
            .transpose()?;
        if let Some(expected) = policy.classifier_bias {
            if expected != classifier_bias.is_some() {
                return Err(crate::start(format!(
                    "ModernBERT classifier_bias={expected} disagrees with the checkpoint bias tensor"
                )));
            }
        }
        for forbidden in ["head.dense.bias", "head.norm.bias"] {
            if checkpoint.contains(forbidden) {
                return Err(crate::start(format!(
                    "ModernBERT classifier head must not contain {forbidden}"
                )));
            }
        }
        let dense = checkpoint
            .contains("head.dense.weight")
            .then(|| checkpoint.source("head.dense.weight", &[hidden, hidden]))
            .transpose()?;
        let norm = checkpoint
            .contains("head.norm.weight")
            .then(|| checkpoint.source("head.norm.weight", &[hidden]))
            .transpose()?;
        let first = match (dense, norm) {
            (Some(weight), Some(gamma)) => Some(HeadStage {
                linear: HeadLinear { weight, bias: None },
                activation: policy.activation,
                norm: Some(HeadNorm {
                    gamma,
                    beta: None,
                    eps: config.geometry.layer_norm_eps as f32,
                }),
            }),
            (None, None) => None,
            _ => {
                return Err(crate::start(
                    "ModernBERT head.dense and head.norm tensors must be complete or absent",
                ))
            }
        };
        if policy.task == ClassifierTask::Sequence && first.is_none() {
            return Err(crate::start(
                "ModernBERT sequence classifier requires head.dense and head.norm",
            ));
        }
        Ok(Self {
            readout: ClassifierReadout {
                task: policy.task,
                width,
            },
            classifier: HeadLinear {
                weight: checkpoint.source("classifier.weight", &[width, hidden])?,
                bias: classifier_bias,
            },
            hidden,
            first_sequence: first.is_some(),
            first_tokens: false,
            first,
            requires_first_for_sequence: true,
            pooling: policy.pooling,
        })
    }
}

fn family_is_gte(family: EncoderFamily) -> bool {
    family == EncoderFamily::Gte
}

/// Validate task and module metadata before creating any CUDA state.
pub(crate) fn validate_admission(
    path: &Path,
    dense_paths: Option<Vec<String>>,
) -> Result<(), BackendError> {
    let raw: serde_json::Value = crate::model::read_json(&path.join("config.json"))?;
    if raw.get("model_type").and_then(serde_json::Value::as_str) == Some("deberta-v2") {
        let config = crate::deberta_encoder_model::load_config(path)?;
        if task_for(&config.geometry).is_none() {
            return Err(crate::start("DeBERTa classifier architecture required"));
        }
        if dense_paths.as_ref().is_some_and(|paths| !paths.is_empty())
            || crate::dense::DenseChainHost::load_with_paths(
                path,
                config.geometry.hidden_size,
                dense_paths,
            )?
            .is_some()
        {
            return Err(crate::start(
                "DeBERTa classifier does not accept Dense modules",
            ));
        }
        return Ok(());
    }
    if raw.get("model_type").and_then(serde_json::Value::as_str) == Some("modernbert") {
        let config = crate::modern_encoder_model::ModernConfig::load_classifier(path)?;
        if dense_paths.as_ref().is_some_and(|paths| !paths.is_empty()) {
            return Err(crate::start(
                "ModernBERT classifier does not accept an explicit Dense / Sentence-Transformers override",
            ));
        }
        if dense_paths.is_none() {
            let _ = crate::dense::DenseChainHost::load_with_paths(
                path,
                config.geometry.hidden_size,
                None,
            )?;
        }
        let checkpoint = Checkpoint::open(path)?;
        let _ = ClassifierHeadHost::load_modern(&checkpoint, &config)?;
        return Ok(());
    }
    let config = BertConfig::load_classifier(path)?;
    let task =
        task_for(&config).ok_or_else(|| crate::start("unsupported classifier architecture"))?;
    admit_geometry(&config, task)?;
    if dense_paths.as_ref().is_some_and(|paths| !paths.is_empty()) {
        return Err(crate::start(
            "Classifier admission does not accept an explicit Dense / Sentence-Transformers override",
        ));
    }
    if dense_paths.is_none() {
        // Native classifiers ignore Dense modules discovered from modules.json,
        // but malformed declared embedding metadata still fails closed.
        let _ = crate::dense::DenseChainHost::load_with_paths(path, config.hidden_size, None)?;
    }
    // Validate the actual trained-head tensor source and every referenced
    // shape before any CUDA context or weight upload exists.
    let checkpoint = Checkpoint::open(path)?;
    let _ = ClassifierHeadHost::load(&checkpoint, &config)?;
    Ok(())
}

struct ClassifierKernels {
    bias_seed: Function,
    gather_first: Function,
    tanh: Function,
    gelu_erf: Option<Function>,
    gelu_tanh: Option<Function>,
    relu: Option<Function>,
    silu: Option<Function>,
    layer_norm: Option<Function>,
    f32_to_bf16: Option<Function>,
    mean_pool: Option<Function>,
    pool: Option<Function>,
    cast: Function,
}

impl ClassifierKernels {
    #[cfg(test)]
    fn load(context: &Arc<Context>) -> Result<Self, BackendError> {
        Self::load_with_erf(context, false)
    }
    fn load_with_erf(context: &Arc<Context>, erf: bool) -> Result<Self, BackendError> {
        let path = std::env::var_os(infinity_v3_config::infinity_env!("OXIDE_KERNEL_MODULE"))
            .ok_or_else(|| {
                crate::start(
                    "Set INFINITY_OXIDE_KERNEL_MODULE to the artifact compiled by cargo oxide",
                )
            })?;
        let path = PathBuf::from(path);
        let path = if path.is_dir() {
            let (major, minor) = context.compute_capability().map_err(crate::start)?;
            path.join(format!("sm_{major}{minor}"))
                .join("kernels.cubin")
        } else {
            path
        };
        let data = std::fs::read(&path)
            .map_err(|error| crate::start(format!("{}: {error}", path.display())))?;
        let module = context
            .load_module_from_image(&data)
            .map_err(crate::start)?;
        let optional = |name: &str| module.load_function(name).ok();
        Ok(Self {
            bias_seed: module
                .load_function("infinity_classifier_bias_bf16")
                .map_err(crate::start)?,
            gather_first: module
                .load_function("infinity_classifier_gather_first_bf16")
                .map_err(crate::start)?,
            tanh: module
                .load_function("infinity_dense_tanh_bf16_inplace")
                .map_err(crate::start)?,
            gelu_erf: if erf {
                Some(
                    module
                        .load_function("infinity_gelu_exact")
                        .map_err(crate::start)?,
                )
            } else {
                optional("infinity_gelu_exact")
            },
            gelu_tanh: optional("infinity_encoder_activation_gelu_tanh_bf16"),
            relu: optional("infinity_encoder_activation_relu_bf16"),
            silu: optional("infinity_encoder_activation_silu_bf16"),
            layer_norm: optional("infinity_plain_layer_norm_bf16_aten"),
            f32_to_bf16: optional("infinity_f32_to_bf16"),
            mean_pool: optional("infinity_mean_pool_bf16_native"),
            pool: optional("infinity_pool"),
            cast: module
                .load_function("infinity_bf16_to_f32")
                .map_err(crate::start)?,
        })
    }
}

struct DeviceHeadLinear {
    weight: DeviceBuffer<Bf16>,
    bias: DeviceBuffer<Bf16>,
}

struct DeviceHeadStage {
    device: DeviceHeadLinear,
    norm: Option<DeviceHeadNorm>,
    activation: HeadActivation,
    plan: HeadLinearPlan,
}

struct DeviceHeadNorm {
    gamma: DeviceBuffer<Bf16>,
    beta: Option<DeviceBuffer<Bf16>>,
    eps: f32,
}

struct DeviceClassifier {
    device: DeviceHeadLinear,
    plan: HeadLinearPlan,
}

fn upload_head_linear(
    uploader: &mut WeightUploader,
    stream: &Arc<Stream>,
    host: &HeadLinear,
) -> Result<DeviceHeadLinear, BackendError> {
    let weight = uploader.upload(&[&host.weight])?;
    let bias = match host.bias.as_ref() {
        Some(bias) => uploader.upload(&[bias])?,
        None => {
            let width = host.weight.shape().first().copied().unwrap_or(0);
            DeviceBuffer::<Bf16>::zeroed(stream, width).map_err(crate::start)?
        }
    };
    Ok(DeviceHeadLinear { weight, bias })
}

/// Scratch capacities reserved from the shared request shape key. Sequence
/// requests need two hidden-width BF16 row buffers; token requests reuse the
/// packed backbone output directly and only need one logit-width buffer. Mean
/// and last-token ModernBERT pooling stage their reduction through one F32 row
/// buffer before the BF16 head.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ClassifierScratchLayout {
    first: usize,
    second: usize,
    pooled: usize,
    logits: usize,
}

impl ClassifierScratchLayout {
    #[cfg(test)]
    pub(crate) fn for_key(hidden: usize, width: usize, key: WorkKey) -> Result<Self, BackendError> {
        Self::for_key_with_token_mlp(hidden, width, key, false)
    }

    #[cfg(test)]
    fn for_key_with_token_mlp(
        hidden: usize,
        width: usize,
        key: WorkKey,
        token_mlp: bool,
    ) -> Result<Self, BackendError> {
        Self::for_key_with_pooling(hidden, width, key, token_mlp, false)
    }

    fn for_key_with_pooling(
        hidden: usize,
        width: usize,
        key: WorkKey,
        token_mlp: bool,
        pooling: bool,
    ) -> Result<Self, BackendError> {
        // Both readout modes share one model head. Reserve the sequence
        // gather/pooler rows and the larger token-logit row extent together so
        // either method can reuse the same bounded request workspace. A trained
        // dense/out_proj token head additionally materializes every token row
        // between its dense/activation and out_proj stages.
        let sequence_rows = key.1;
        let token_rows = key.0;
        let sequence_hidden = product(sequence_rows, hidden)?;
        let token_logits = product(token_rows, width)?;
        let sequence_logits = product(sequence_rows, width)?;
        let token_hidden = if token_mlp {
            product(token_rows, hidden)?
        } else {
            0
        };
        let first = sequence_hidden.max(sequence_logits).max(token_hidden);
        let second = sequence_hidden.max(token_logits);
        let pooled = if pooling { sequence_hidden } else { 0 };
        let logits = sequence_logits.max(token_logits);
        for elements in [first, second, pooled, logits] {
            u32_size(elements)?;
        }
        Ok(Self {
            first,
            second,
            pooled,
            logits,
        })
    }

    pub(crate) fn bytes(self) -> Result<usize, BackendError> {
        product(self.first, size_of::<Bf16>())?
            .checked_add(product(self.second, size_of::<Bf16>())?)
            .and_then(|bytes| bytes.checked_add(product(self.pooled, size_of::<f32>()).ok()?))
            .and_then(|bytes| bytes.checked_add(product(self.logits, size_of::<f32>()).ok()?))
            .ok_or_else(|| infer("classifier workspace byte size overflow"))
    }
}

pub(crate) struct ClassifierScratchBuffers {
    stream: Arc<Stream>,
    first: Option<DeviceBuffer<Bf16>>,
    second: Option<DeviceBuffer<Bf16>>,
    pooled: ManuallyDrop<Option<DeviceBuffer<f32>>>,
    logits: ManuallyDrop<DeviceBuffer<f32>>,
    stream_drained: bool,
}

fn initialized_bf16(
    stream: &Arc<Stream>,
    elements: usize,
) -> Result<DeviceBuffer<Bf16>, BackendError> {
    if elements == 0 {
        return Err(infer("classifier scratch buffer must be nonzero"));
    }
    // SAFETY: private allocation initialized before any vectorized read.
    let buffer =
        unsafe { DeviceBuffer::<Bf16>::uninitialized_async(stream, elements) }.map_err(infer)?;
    if let Err(error) = unsafe {
        cuda_core::memory::memset_d8_async(
            buffer.cu_deviceptr(),
            0,
            buffer.num_bytes(),
            stream.cu_stream(),
        )
    } {
        if stream.synchronize().is_err() {
            crate::record_uncharged_retained_bytes(buffer.num_bytes());
            std::mem::forget(buffer);
        }
        return Err(infer(error));
    }
    Ok(buffer)
}

fn drop_after_uncertain(stream: &Stream, owners: Vec<DeviceBuffer<Bf16>>) {
    if stream.synchronize().is_err() {
        let bytes = owners.iter().fold(0usize, |bytes, owner| {
            bytes.saturating_add(owner.num_bytes())
        });
        crate::record_uncharged_retained_bytes(bytes);
        for owner in owners {
            std::mem::forget(owner);
        }
    }
}

impl ClassifierScratchBuffers {
    pub(crate) fn new(
        stream: Arc<Stream>,
        layout: ClassifierScratchLayout,
    ) -> Result<Self, BackendError> {
        let first = (layout.first > 0)
            .then(|| initialized_bf16(&stream, layout.first))
            .transpose()?;
        let second = match (layout.second > 0)
            .then(|| initialized_bf16(&stream, layout.second))
            .transpose()
        {
            Ok(second) => second,
            Err(error) => {
                drop_after_uncertain(&stream, first.into_iter().collect());
                return Err(error);
            }
        };
        // SAFETY: the pooling kernel writes every reduced element before use.
        let pooled = match (layout.pooled > 0)
            .then(|| unsafe { DeviceBuffer::<f32>::uninitialized_async(&stream, layout.pooled) })
            .transpose()
        {
            Ok(pooled) => pooled,
            Err(error) => {
                let mut owners = Vec::new();
                owners.extend(first);
                owners.extend(second);
                drop_after_uncertain(&stream, owners);
                return Err(infer(error));
            }
        };
        // SAFETY: the cast kernel writes every logit element before readback.
        let logits =
            match unsafe { DeviceBuffer::<f32>::uninitialized_async(&stream, layout.logits) } {
                Ok(buffer) => buffer,
                Err(error) => {
                    let mut owners = Vec::new();
                    owners.extend(first);
                    owners.extend(second);
                    drop_after_uncertain(&stream, owners);
                    return Err(infer(error));
                }
            };
        Ok(Self {
            stream,
            first,
            second,
            pooled: ManuallyDrop::new(pooled),
            logits: ManuallyDrop::new(logits),
            stream_drained: false,
        })
    }

    /// # Safety
    /// The owning stream must have completed all work using these buffers.
    pub(crate) unsafe fn mark_stream_drained(&mut self) {
        self.stream_drained = true;
    }

    pub(crate) fn source(&self) -> Option<&DeviceBuffer<Bf16>> {
        self.first.as_ref()
    }

    fn pair_mut(&mut self) -> Result<(&DeviceBuffer<Bf16>, &mut DeviceBuffer<Bf16>), BackendError> {
        let first = self
            .first
            .as_ref()
            .ok_or_else(|| infer("missing classifier row-gather scratch"))?;
        let second = self
            .second
            .as_mut()
            .ok_or_else(|| infer("missing classifier hidden scratch"))?;
        Ok((first, second))
    }

    fn swapped_mut(
        &mut self,
    ) -> Result<(&DeviceBuffer<Bf16>, &mut DeviceBuffer<Bf16>), BackendError> {
        let second = self
            .second
            .as_ref()
            .ok_or_else(|| infer("missing classifier hidden scratch"))?;
        let first = self
            .first
            .as_mut()
            .ok_or_else(|| infer("missing classifier row-gather scratch"))?;
        Ok((second, first))
    }

    fn pooled(&self) -> Option<&DeviceBuffer<f32>> {
        (*self.pooled).as_ref()
    }

    pub(crate) fn logits(&self) -> &DeviceBuffer<f32> {
        &self.logits
    }
}

impl Drop for ClassifierScratchBuffers {
    fn drop(&mut self) {
        if !self.stream_drained {
            #[cfg(test)]
            crate::note_default_drop_drain();
        }
        if self.stream_drained || self.stream.synchronize().is_ok() {
            self.first.take();
            self.second.take();
            // SAFETY: this destructor just drained the owning stream.
            unsafe {
                ManuallyDrop::drop(&mut self.pooled);
                ManuallyDrop::drop(&mut self.logits);
            }
        } else {
            let bytes = self
                .first
                .as_ref()
                .map_or(0, DeviceBuffer::num_bytes)
                .saturating_add(self.second.as_ref().map_or(0, DeviceBuffer::num_bytes))
                .saturating_add((*self.pooled).as_ref().map_or(0, DeviceBuffer::num_bytes))
                .saturating_add(self.logits.num_bytes());
            crate::record_uncharged_retained_bytes(bytes);
            if let Some(first) = self.first.take() {
                std::mem::forget(first);
            }
            if let Some(second) = self.second.take() {
                std::mem::forget(second);
            }
        }
    }
}

pub(crate) struct ClassifierHead {
    readout: ClassifierReadout,
    hidden: usize,
    first: Option<DeviceHeadStage>,
    first_sequence: bool,
    first_tokens: bool,
    requires_first_for_sequence: bool,
    pooling: ClassifierPooling,
    classifier: DeviceClassifier,
    kernels: ClassifierKernels,
    prepared: Option<ClassifierPrepared>,
}

struct ClassifierPrepared {
    mode: ReadoutMode,
    rows: usize,
    source: u64,
    stream: usize,
    first: Option<PreparedHeadLinear>,
    classifier: PreparedHeadLinear,
}

fn validate_bounds(bounds: &[u32], tokens: usize, rows: usize) -> Result<(), ShapeError> {
    if bounds.len()
        != rows
            .checked_add(1)
            .ok_or_else(|| ShapeError::Rejected(infer("classifier boundary count overflow")))?
    {
        return Err(ShapeError::Rejected(infer(
            "classifier boundary table does not match sequence count",
        )));
    }
    if bounds.first() != Some(&0) || bounds.last().copied().map(|end| end as usize) != Some(tokens)
    {
        return Err(ShapeError::Rejected(infer(
            "classifier boundary table must start at zero and end at the token count",
        )));
    }
    for pair in bounds.windows(2) {
        let (Some(&start), Some(&end)) = (pair.first(), pair.get(1)) else {
            return Err(ShapeError::Rejected(infer(
                "classifier boundary table is malformed",
            )));
        };
        if end <= start {
            return Err(ShapeError::Rejected(infer(
                "classifier sequences must have nonzero length",
            )));
        }
    }
    Ok(())
}

fn validate_source_span(available: usize, tokens: usize, hidden: usize) -> Result<(), ShapeError> {
    if available < product(tokens, hidden).map_err(ShapeError::Rejected)? {
        return Err(ShapeError::Rejected(infer(
            "classifier source rows are shorter than the packed token count",
        )));
    }
    Ok(())
}

fn require_canonical(batch: &Batch, mode: ReadoutMode) -> Result<(), ShapeError> {
    let sequences = batch
        .cumulative_seq_lengths
        .len()
        .checked_sub(1)
        .filter(|n| *n > 0)
        .ok_or_else(|| ShapeError::Rejected(infer("classifier batch is empty")))?;
    let canonical = |indices: &[u32]| {
        indices.len() == sequences
            && indices
                .iter()
                .enumerate()
                .all(|(index, &value)| usize::try_from(value) == Ok(index))
    };
    match mode {
        ReadoutMode::Sequence => {
            if !canonical(&batch.pooled_indices) || !batch.raw_indices.is_empty() {
                return Err(ShapeError::Rejected(infer(
                    "classifier sequence readout requires canonical pooled indices 0..B",
                )));
            }
        }
        ReadoutMode::Tokens => {
            if !canonical(&batch.raw_indices) || !batch.pooled_indices.is_empty() {
                return Err(ShapeError::Rejected(infer(
                    "classifier token readout requires canonical raw indices 0..B",
                )));
            }
        }
    }
    Ok(())
}

impl ClassifierHead {
    pub(crate) fn new(
        host: ClassifierHeadHost,
        context: &Arc<Context>,
        stream: &Arc<Stream>,
        provider: &LtProvider,
    ) -> Result<Self, BackendError> {
        if host.first_tokens && host.first.is_none() {
            return Err(crate::start("trained token head requires its dense stage"));
        }
        if host.requires_first_for_sequence
            && host.first.is_none()
            && host.readout.task == ClassifierTask::Sequence
        {
            return Err(crate::start(
                "sequence classifier requires its trained head stage",
            ));
        }
        let erf = host
            .first
            .as_ref()
            .is_some_and(|stage| stage.activation == HeadActivation::GeluErf);
        let kernels = ClassifierKernels::load_with_erf(context, erf)?;
        let hidden = host.hidden;
        if hidden == 0 {
            return Err(crate::start("classifier hidden width must be nonzero"));
        }
        if let Some(stage) = host.first.as_ref() {
            if stage.linear.input() != Some(hidden)
                || stage.linear.weight.shape().first().copied() != Some(hidden)
            {
                return Err(crate::start(format!(
                    "classifier first projection must be [{hidden}, {hidden}]"
                )));
            }
            if let Some(norm) = stage.norm.as_ref() {
                if norm.gamma.shape() != [hidden]
                    || norm
                        .beta
                        .as_ref()
                        .is_some_and(|beta| beta.shape() != [hidden])
                {
                    return Err(crate::start(format!(
                        "classifier head norm must use hidden width {hidden}"
                    )));
                }
            }
            match stage.activation {
                HeadActivation::Tanh => {}
                HeadActivation::GeluErf if kernels.gelu_erf.is_none() => {
                    return Err(crate::start("classifier GELU-ERF kernel is unavailable"))
                }
                HeadActivation::GeluTanh if kernels.gelu_tanh.is_none() => {
                    return Err(crate::start(
                        "classifier tanh-GELU activation kernel is unavailable",
                    ))
                }
                HeadActivation::Relu if kernels.relu.is_none() => {
                    return Err(crate::start(
                        "classifier ReLU activation kernel is unavailable",
                    ))
                }
                HeadActivation::Silu if kernels.silu.is_none() => {
                    return Err(crate::start(
                        "classifier SiLU activation kernel is unavailable",
                    ))
                }
                _ => {}
            }
            if stage.norm.is_some() && kernels.layer_norm.is_none() {
                return Err(crate::start("classifier head norm kernel is unavailable"));
            }
        }
        if host.pooling != ClassifierPooling::First {
            if kernels.f32_to_bf16.is_none() {
                return Err(crate::start(
                    "classifier pooling requires the F32-to-BF16 kernel",
                ));
            }
            let kernel = match host.pooling {
                ClassifierPooling::Mean => kernels.mean_pool.as_ref(),
                ClassifierPooling::Last => kernels.pool.as_ref(),
                ClassifierPooling::First => None,
            };
            if kernel.is_none() {
                return Err(crate::start(
                    "classifier pooling reduction kernel is unavailable",
                ));
            }
        }
        if host.classifier.input() != Some(hidden) {
            return Err(crate::start(format!(
                "classifier projection input width must be {hidden}"
            )));
        }
        let width = host.readout.width;
        if host.classifier.weight.shape().first().copied() != Some(width) {
            return Err(crate::start(format!(
                "classifier projection output width must be {width}"
            )));
        }
        if host
            .classifier
            .bias
            .as_ref()
            .is_some_and(|bias| bias.shape() != [width])
        {
            return Err(crate::start(format!(
                "classifier bias must use output width {width}"
            )));
        }
        let mut first_plan = host
            .first
            .as_ref()
            .map(|stage| HeadLinearPlan::new(provider, hidden, hidden, stage.linear.bias.is_some()))
            .transpose()
            .map_err(ShapeError::into_backend)?;
        let classifier_plan =
            HeadLinearPlan::new(provider, hidden, width, host.classifier.bias.is_some())
                .map_err(ShapeError::into_backend)?;
        let mut uploader = WeightUploader::new(stream.clone())?;
        let first = host
            .first
            .as_ref()
            .map(|stage| {
                let device = upload_head_linear(&mut uploader, stream, &stage.linear)?;
                let norm = stage
                    .norm
                    .as_ref()
                    .map(|norm| {
                        Ok::<_, BackendError>(DeviceHeadNorm {
                            gamma: uploader.upload(&[&norm.gamma])?,
                            beta: match norm.beta.as_ref() {
                                Some(beta) => Some(uploader.upload(&[beta])?),
                                None => None,
                            },
                            eps: norm.eps,
                        })
                    })
                    .transpose()?;
                Ok::<_, BackendError>(DeviceHeadStage {
                    device,
                    norm,
                    activation: stage.activation,
                    plan: first_plan
                        .take()
                        .ok_or_else(|| infer("missing classifier first-stage plan"))?,
                })
            })
            .transpose()?;
        let classifier = DeviceClassifier {
            device: upload_head_linear(&mut uploader, stream, &host.classifier)?,
            plan: classifier_plan,
        };
        uploader.finish()?;
        Ok(Self {
            readout: host.readout,
            hidden,
            first,
            first_sequence: host.first_sequence,
            first_tokens: host.first_tokens,
            requires_first_for_sequence: host.requires_first_for_sequence,
            pooling: host.pooling,
            classifier,
            kernels,
            prepared: None,
        })
    }

    pub(crate) fn readout(&self) -> ClassifierReadout {
        self.readout
    }

    pub(crate) fn scratch_layout(
        &self,
        key: WorkKey,
    ) -> Result<ClassifierScratchLayout, BackendError> {
        ClassifierScratchLayout::for_key_with_pooling(
            self.hidden,
            self.readout.width,
            key,
            self.first_tokens,
            self.pooling != ClassifierPooling::First,
        )
    }

    pub(crate) fn rows(&self, batch: &Batch, mode: ReadoutMode) -> Result<usize, BackendError> {
        let sequences = batch
            .cumulative_seq_lengths
            .len()
            .checked_sub(1)
            .filter(|n| *n > 0)
            .ok_or_else(|| infer("classifier batch must contain at least one sequence"))?;
        match mode {
            ReadoutMode::Sequence => Ok(sequences),
            ReadoutMode::Tokens => Ok(batch.input_ids.len()),
        }
    }

    pub(crate) fn preflight(
        &self,
        provider: &LtProvider,
        batch: &Batch,
        mode: ReadoutMode,
    ) -> Result<usize, ShapeError> {
        let rows = self.rows(batch, mode).map_err(ShapeError::Rejected)?;
        require_canonical(batch, mode)?;
        validate_bounds(
            &batch.cumulative_seq_lengths,
            batch.input_ids.len(),
            batch.cumulative_seq_lengths.len().saturating_sub(1),
        )?;
        if mode == ReadoutMode::Sequence && self.requires_first_for_sequence && self.first.is_none()
        {
            return Err(ShapeError::Rejected(infer(
                "sequence classifier requires its trained head stage",
            )));
        }
        match mode {
            ReadoutMode::Sequence => {
                if self.first_sequence {
                    if let Some(stage) = self.first.as_ref() {
                        stage.plan.preflight_rows(rows)?;
                    }
                }
            }
            ReadoutMode::Tokens => {
                if self.first_tokens {
                    let stage = self.first.as_ref().ok_or_else(|| {
                        ShapeError::Rejected(infer("trained token head requires its dense stage"))
                    })?;
                    stage.plan.preflight_rows(rows)?;
                }
            }
        }
        self.classifier.plan.preflight_rows(rows)?;
        let _ = provider;
        Ok(rows)
    }

    /// Prepare every actual head pointer and shape before the request lease
    /// starts. A cold shape may run the native heuristic / `AlgoCheck`; cache
    /// hits do not. This performs no allocation.
    ///
    /// # Safety
    /// The engine context must already be bound on this thread.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn prepare(
        &mut self,
        provider: &LtProvider,
        blas: &Blas,
        batch: &Batch,
        mode: ReadoutMode,
        stream: &Stream,
        source: &DeviceBuffer<Bf16>,
        scratch: &mut ClassifierScratchBuffers,
    ) -> Result<(), ShapeError> {
        self.prepared = None;
        let rows = self.preflight(provider, batch, mode)?;
        let hidden = self.hidden;
        let tokens = batch.input_ids.len();
        validate_source_span(source.len(), tokens, hidden)?;
        if !Arc::ptr_eq(source.context(), provider.context())
            || !Arc::ptr_eq(stream.context(), provider.context())
            || !Arc::ptr_eq(blas.context(), provider.context())
        {
            return Err(ShapeError::Fatal(infer(
                "classifier source, stream and handles must share their owning CUDA context",
            )));
        }
        if self.pooling != ClassifierPooling::First
            && mode == ReadoutMode::Sequence
            && scratch.pooled().is_none()
        {
            return Err(ShapeError::Fatal(infer(
                "missing classifier F32 pooling scratch",
            )));
        }
        let source_pointer = source.cu_deviceptr();
        let prepared_stream = stream.cu_stream() as usize;
        match mode {
            ReadoutMode::Sequence => {
                if scratch.source().is_none() {
                    return Err(ShapeError::Fatal(infer(
                        "missing classifier row-gather scratch",
                    )));
                }
                if self.first_sequence && self.first.is_some() {
                    let first_prepared = {
                        let (gather, hidden_buffer) =
                            scratch.pair_mut().map_err(ShapeError::Fatal)?;
                        let stage = self
                            .first
                            .as_mut()
                            .ok_or_else(|| ShapeError::Fatal(infer("missing classifier head")))?;
                        unsafe {
                            stage.plan.prepare(
                                provider,
                                blas,
                                rows,
                                stream,
                                gather,
                                &stage.device.weight,
                                &stage.device.bias,
                                hidden_buffer,
                            )?
                        }
                    };
                    let classifier_prepared = {
                        let (hidden_buffer, gather) =
                            scratch.swapped_mut().map_err(ShapeError::Fatal)?;
                        unsafe {
                            self.classifier.plan.prepare(
                                provider,
                                blas,
                                rows,
                                stream,
                                hidden_buffer,
                                &self.classifier.device.weight,
                                &self.classifier.device.bias,
                                gather,
                            )?
                        }
                    };
                    self.prepared = Some(ClassifierPrepared {
                        mode,
                        rows,
                        source: source_pointer,
                        stream: prepared_stream,
                        first: Some(first_prepared),
                        classifier: classifier_prepared,
                    });
                } else {
                    let classifier_prepared = {
                        let (gather, second) = scratch.pair_mut().map_err(ShapeError::Fatal)?;
                        unsafe {
                            self.classifier.plan.prepare(
                                provider,
                                blas,
                                rows,
                                stream,
                                gather,
                                &self.classifier.device.weight,
                                &self.classifier.device.bias,
                                second,
                            )?
                        }
                    };
                    self.prepared = Some(ClassifierPrepared {
                        mode,
                        rows,
                        source: source_pointer,
                        stream: prepared_stream,
                        first: None,
                        classifier: classifier_prepared,
                    });
                }
            }
            ReadoutMode::Tokens => {
                if self.first_tokens {
                    let first_prepared = {
                        let hidden_buffer = scratch.first.as_mut().ok_or_else(|| {
                            ShapeError::Fatal(infer("missing classifier token hidden scratch"))
                        })?;
                        let stage = self
                            .first
                            .as_mut()
                            .ok_or_else(|| ShapeError::Fatal(infer("missing classifier head")))?;
                        unsafe {
                            stage.plan.prepare(
                                provider,
                                blas,
                                rows,
                                stream,
                                source,
                                &stage.device.weight,
                                &stage.device.bias,
                                hidden_buffer,
                            )?
                        }
                    };
                    let classifier_prepared = {
                        let hidden_buffer = scratch.first.as_ref().ok_or_else(|| {
                            ShapeError::Fatal(infer("missing classifier token hidden scratch"))
                        })?;
                        let output = scratch.second.as_mut().ok_or_else(|| {
                            ShapeError::Fatal(infer("missing classifier logit scratch"))
                        })?;
                        unsafe {
                            self.classifier.plan.prepare(
                                provider,
                                blas,
                                rows,
                                stream,
                                hidden_buffer,
                                &self.classifier.device.weight,
                                &self.classifier.device.bias,
                                output,
                            )?
                        }
                    };
                    self.prepared = Some(ClassifierPrepared {
                        mode,
                        rows,
                        source: source_pointer,
                        stream: prepared_stream,
                        first: Some(first_prepared),
                        classifier: classifier_prepared,
                    });
                } else {
                    let output = scratch.second.as_mut().ok_or_else(|| {
                        ShapeError::Fatal(infer("missing classifier logit scratch"))
                    })?;
                    let classifier_prepared = unsafe {
                        self.classifier.plan.prepare(
                            provider,
                            blas,
                            rows,
                            stream,
                            source,
                            &self.classifier.device.weight,
                            &self.classifier.device.bias,
                            output,
                        )?
                    };
                    self.prepared = Some(ClassifierPrepared {
                        mode,
                        rows,
                        source: source_pointer,
                        stream: prepared_stream,
                        first: None,
                        classifier: classifier_prepared,
                    });
                }
            }
        }
        Ok(())
    }

    /// Launch pooling/head/classifier/cast on the shared ordered stream.
    /// `bounds` is the already-uploaded device cumulative table, not a host
    /// slice pointer. `selected` is the uploaded canonical sequence selection.
    /// CPU span validation happened during preparation.
    ///
    /// # Safety
    /// The caller retains the source, uploaded bounds/selections, scratch,
    /// weights and module through stream completion and passes the same buffers
    /// used during preparation.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn enqueue(
        &mut self,
        provider: &LtProvider,
        blas: &Blas,
        stream: &Stream,
        source: &DeviceBuffer<Bf16>,
        bounds: u64,
        selected: u64,
        scratch: &mut ClassifierScratchBuffers,
    ) -> Result<(), BackendError> {
        let prepared = self
            .prepared
            .take()
            .ok_or_else(|| infer("classifier readout was not prepared before launch"))?;
        let ClassifierPrepared {
            mode,
            rows,
            source: prepared_source,
            stream: prepared_stream,
            first: first_prepared,
            classifier,
        } = prepared;
        if source.cu_deviceptr() != prepared_source
            || stream.cu_stream() as usize != prepared_stream
        {
            return Err(infer(
                "classifier source or stream differs from prepared metadata",
            ));
        }
        let hidden = self.hidden;
        let width = self.readout.width;
        match mode {
            ReadoutMode::Sequence => {
                pool_classifier_rows(
                    &self.kernels,
                    self.pooling,
                    stream,
                    source,
                    bounds,
                    selected,
                    scratch,
                    rows,
                    hidden,
                )?;
                if let Some(first_prepared) = first_prepared {
                    {
                        let (gather, hidden_buffer) = scratch.pair_mut()?;
                        let stage = self
                            .first
                            .as_mut()
                            .ok_or_else(|| infer("missing classifier head"))?;
                        // SAFETY: the caller's retained-buffer contract covers
                        // the preselected head invocation.
                        unsafe {
                            stage.plan.launch(
                                &first_prepared,
                                provider,
                                blas,
                                &self.kernels.bias_seed,
                                stream,
                                gather,
                                &stage.device.weight,
                                &stage.device.bias,
                                hidden_buffer,
                            )?;
                        }
                        apply_classifier_activation(
                            &self.kernels,
                            stage.activation,
                            stream,
                            hidden_buffer,
                            rows,
                            hidden,
                        )?;
                        if let Some(norm) = stage.norm.as_ref() {
                            apply_classifier_norm(
                                &self.kernels,
                                stream,
                                norm,
                                hidden_buffer,
                                rows,
                                hidden,
                                norm.eps,
                            )?;
                        }
                    }
                    {
                        let (hidden_buffer, gather) = scratch.swapped_mut()?;
                        // SAFETY: the caller's retained-buffer contract covers
                        // the preselected classifier invocation.
                        unsafe {
                            self.classifier.plan.launch(
                                &classifier,
                                provider,
                                blas,
                                &self.kernels.bias_seed,
                                stream,
                                hidden_buffer,
                                &self.classifier.device.weight,
                                &self.classifier.device.bias,
                                gather,
                            )?;
                        }
                    }
                    cast_to_f32(
                        &self.kernels,
                        stream,
                        scratch.source(),
                        scratch.logits(),
                        rows,
                        width,
                    )
                } else {
                    let (gather, second) = scratch.pair_mut()?;
                    // SAFETY: the caller's retained-buffer contract covers the
                    // preselected classifier invocation.
                    unsafe {
                        self.classifier.plan.launch(
                            &classifier,
                            provider,
                            blas,
                            &self.kernels.bias_seed,
                            stream,
                            gather,
                            &self.classifier.device.weight,
                            &self.classifier.device.bias,
                            second,
                        )?;
                    }
                    cast_to_f32(
                        &self.kernels,
                        stream,
                        scratch.second.as_ref(),
                        scratch.logits(),
                        rows,
                        width,
                    )
                }
            }
            ReadoutMode::Tokens => {
                if let Some(first_prepared) = first_prepared {
                    {
                        let token_hidden = scratch
                            .first
                            .as_mut()
                            .ok_or_else(|| infer("missing classifier token hidden scratch"))?;
                        let stage = self
                            .first
                            .as_mut()
                            .ok_or_else(|| infer("missing classifier head"))?;
                        // SAFETY: the caller's retained-buffer contract covers
                        // the preselected token head invocation.
                        unsafe {
                            stage.plan.launch(
                                &first_prepared,
                                provider,
                                blas,
                                &self.kernels.bias_seed,
                                stream,
                                source,
                                &stage.device.weight,
                                &stage.device.bias,
                                token_hidden,
                            )?;
                        }
                        apply_classifier_activation(
                            &self.kernels,
                            stage.activation,
                            stream,
                            token_hidden,
                            rows,
                            hidden,
                        )?;
                        if let Some(norm) = stage.norm.as_ref() {
                            apply_classifier_norm(
                                &self.kernels,
                                stream,
                                norm,
                                token_hidden,
                                rows,
                                hidden,
                                norm.eps,
                            )?;
                        }
                    }
                    {
                        let token_hidden = scratch
                            .first
                            .as_ref()
                            .ok_or_else(|| infer("missing classifier token hidden scratch"))?;
                        let output = scratch
                            .second
                            .as_mut()
                            .ok_or_else(|| infer("missing classifier logit scratch"))?;
                        // SAFETY: the caller's retained-buffer contract covers
                        // the preselected token projection.
                        unsafe {
                            self.classifier.plan.launch(
                                &classifier,
                                provider,
                                blas,
                                &self.kernels.bias_seed,
                                stream,
                                token_hidden,
                                &self.classifier.device.weight,
                                &self.classifier.device.bias,
                                output,
                            )?;
                        }
                    }
                } else {
                    let output = scratch
                        .second
                        .as_mut()
                        .ok_or_else(|| infer("missing classifier logit scratch"))?;
                    // SAFETY: the caller's retained-buffer contract covers the
                    // preselected token projection.
                    unsafe {
                        self.classifier.plan.launch(
                            &classifier,
                            provider,
                            blas,
                            &self.kernels.bias_seed,
                            stream,
                            source,
                            &self.classifier.device.weight,
                            &self.classifier.device.bias,
                            output,
                        )?;
                    }
                }
                cast_to_f32(
                    &self.kernels,
                    stream,
                    scratch.second.as_ref(),
                    scratch.logits(),
                    rows,
                    width,
                )
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn pool_classifier_rows(
    kernels: &ClassifierKernels,
    pooling: ClassifierPooling,
    stream: &Stream,
    source: &DeviceBuffer<Bf16>,
    bounds: u64,
    selected: u64,
    scratch: &mut ClassifierScratchBuffers,
    rows: usize,
    hidden: usize,
) -> Result<(), BackendError> {
    let output = scratch
        .first
        .as_ref()
        .ok_or_else(|| infer("missing classifier row-gather scratch"))?;
    match pooling {
        ClassifierPooling::First => {
            gather_first(kernels, stream, source, bounds, output, rows, hidden)
        }
        ClassifierPooling::Mean | ClassifierPooling::Last => {
            if selected == 0 {
                return Err(infer(
                    "classifier pooling requires the uploaded sequence selections",
                ));
            }
            let pooled = (*scratch.pooled)
                .as_ref()
                .ok_or_else(|| infer("missing classifier F32 pooling scratch"))?;
            let mut source_pointer = source.cu_deviceptr();
            let mut bounds_pointer = bounds;
            let mut selected_pointer = selected;
            let mut pooled_pointer = pooled.cu_deviceptr();
            let mut count = u32_size(rows)?;
            let mut width = u32_size(hidden)?;
            if pooling == ClassifierPooling::Mean {
                let kernel = kernels
                    .mean_pool
                    .as_ref()
                    .ok_or_else(|| infer("classifier mean-pool kernel is unavailable"))?;
                launch(
                    kernel,
                    stream,
                    (count, u32_size(hidden.div_ceil(8))?, 1),
                    0,
                    &mut [
                        arg(&mut source_pointer),
                        arg(&mut bounds_pointer),
                        arg(&mut selected_pointer),
                        arg(&mut pooled_pointer),
                        arg(&mut count),
                        arg(&mut width),
                    ],
                )?;
            } else {
                let kernel = kernels
                    .pool
                    .as_ref()
                    .ok_or_else(|| infer("classifier last-token pool kernel is unavailable"))?;
                let mut mode = 2u32;
                launch(
                    kernel,
                    stream,
                    blocks(product(rows, hidden)?)?,
                    0,
                    &mut [
                        arg(&mut source_pointer),
                        arg(&mut bounds_pointer),
                        arg(&mut selected_pointer),
                        arg(&mut pooled_pointer),
                        arg(&mut count),
                        arg(&mut width),
                        arg(&mut mode),
                    ],
                )?;
            }
            cast_f32_to_bf16(kernels, stream, pooled, output, product(rows, hidden)?)
        }
    }
}

fn apply_classifier_activation(
    kernels: &ClassifierKernels,
    activation: HeadActivation,
    stream: &Stream,
    values: &mut DeviceBuffer<Bf16>,
    rows: usize,
    width: usize,
) -> Result<(), BackendError> {
    match activation {
        HeadActivation::Tanh => apply_tanh(kernels, stream, values, rows, width),
        HeadActivation::GeluErf => {
            let kernel = kernels
                .gelu_erf
                .as_ref()
                .ok_or_else(|| infer("classifier GELU-ERF kernel is unavailable"))?;
            let mut input = values.cu_deviceptr();
            let mut count = u32_size(product(rows, width)?)?;
            launch(
                kernel,
                stream,
                blocks(count as usize)?,
                0,
                &mut [arg(&mut input), arg(&mut count)],
            )
        }
        HeadActivation::GeluTanh | HeadActivation::Relu | HeadActivation::Silu => {
            let kernel = match activation {
                HeadActivation::GeluTanh => kernels.gelu_tanh.as_ref(),
                HeadActivation::Relu => kernels.relu.as_ref(),
                HeadActivation::Silu => kernels.silu.as_ref(),
                _ => None,
            }
            .ok_or_else(|| infer("classifier pointwise activation kernel is unavailable"))?;
            let mut input = values.cu_deviceptr();
            let mut output = input;
            let mut bias = 0u64;
            let mut rows = u32_size(rows)?;
            let mut width = u32_size(width)?;
            let mut input_stride = width;
            let mut output_stride = width;
            launch(
                kernel,
                stream,
                blocks(product(rows as usize, width as usize)?)?,
                0,
                &mut [
                    arg(&mut input),
                    arg(&mut bias),
                    arg(&mut output),
                    arg(&mut rows),
                    arg(&mut width),
                    arg(&mut input_stride),
                    arg(&mut output_stride),
                ],
            )
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_classifier_norm(
    kernels: &ClassifierKernels,
    stream: &Stream,
    norm: &DeviceHeadNorm,
    values: &mut DeviceBuffer<Bf16>,
    rows: usize,
    width: usize,
    eps: f32,
) -> Result<(), BackendError> {
    let kernel = kernels
        .layer_norm
        .as_ref()
        .ok_or_else(|| infer("classifier head norm kernel is unavailable"))?;
    let mut input = values.cu_deviceptr();
    let mut gamma = norm.gamma.cu_deviceptr();
    let mut beta = norm.beta.as_ref().map_or(0, DeviceBuffer::cu_deviceptr);
    let mut output = input;
    let mut rows = u32_size(rows)?;
    let mut width = u32_size(width)?;
    let mut eps = eps;
    crate::launch_threads(
        kernel,
        stream,
        (rows, 1, 1),
        128,
        0,
        &mut [
            arg(&mut input),
            arg(&mut gamma),
            arg(&mut beta),
            arg(&mut output),
            arg(&mut rows),
            arg(&mut width),
            arg(&mut eps),
        ],
    )
}

fn cast_f32_to_bf16(
    kernels: &ClassifierKernels,
    stream: &Stream,
    input: &DeviceBuffer<f32>,
    output: &DeviceBuffer<Bf16>,
    elements: usize,
) -> Result<(), BackendError> {
    let kernel = kernels
        .f32_to_bf16
        .as_ref()
        .ok_or_else(|| infer("classifier F32-to-BF16 kernel is unavailable"))?;
    let mut input = input.cu_deviceptr();
    let mut output = output.cu_deviceptr();
    let mut elements = u32_size(elements)?;
    launch(
        kernel,
        stream,
        blocks(elements as usize)?,
        0,
        &mut [arg(&mut input), arg(&mut output), arg(&mut elements)],
    )
}

#[allow(clippy::too_many_arguments)]
fn gather_first(
    kernels: &ClassifierKernels,
    stream: &Stream,
    source: &DeviceBuffer<Bf16>,
    bounds: u64,
    output: &DeviceBuffer<Bf16>,
    rows: usize,
    width: usize,
) -> Result<(), BackendError> {
    if bounds == 0 {
        return Err(infer(
            "classifier gather requires the uploaded device boundary table",
        ));
    }
    let mut source_pointer = source.cu_deviceptr();
    let mut bounds_pointer = bounds;
    let mut out_pointer = output.cu_deviceptr();
    let mut rows_u32 = u32_size(rows)?;
    let mut width_u32 = u32_size(width)?;
    launch(
        &kernels.gather_first,
        stream,
        blocks(product(rows, width)?)?,
        0,
        &mut [
            arg(&mut source_pointer),
            arg(&mut bounds_pointer),
            arg(&mut out_pointer),
            arg(&mut rows_u32),
            arg(&mut width_u32),
        ],
    )
}

fn apply_tanh(
    kernels: &ClassifierKernels,
    stream: &Stream,
    values: &mut DeviceBuffer<Bf16>,
    rows: usize,
    width: usize,
) -> Result<(), BackendError> {
    let mut values = values.cu_deviceptr();
    let mut count = u32_size(product(rows, width)?)?;
    launch(
        &kernels.tanh,
        stream,
        blocks(count as usize)?,
        0,
        &mut [arg(&mut values), arg(&mut count)],
    )
}

fn cast_to_f32(
    kernels: &ClassifierKernels,
    stream: &Stream,
    input: Option<&DeviceBuffer<Bf16>>,
    output: &DeviceBuffer<f32>,
    rows: usize,
    width: usize,
) -> Result<(), BackendError> {
    let input = input.ok_or_else(|| infer("missing classifier cast input"))?;
    let elements = product(rows, width)?;
    let mut input = input.cu_deviceptr();
    let mut output = output.cu_deviceptr();
    let mut selected = 0u64;
    let mut rows = u32_size(rows)?;
    let mut width = u32_size(width)?;
    launch(
        &kernels.cast,
        stream,
        blocks(elements)?,
        0,
        &mut [
            arg(&mut input),
            arg(&mut output),
            arg(&mut selected),
            arg(&mut rows),
            arg(&mut width),
        ],
    )
}

pub(crate) fn canonicalize_indices(
    indices: &mut Vec<u32>,
    count: usize,
) -> Result<(), BackendError> {
    if count > u32::MAX as usize {
        return Err(BackendError::Inference(
            "Classifier canonical row count exceeds u32".into(),
        ));
    }
    indices.clear();
    indices
        .try_reserve(count)
        .map_err(|error| BackendError::Overloaded(error.to_string()))?;
    for index in 0..count {
        indices.push(u32::try_from(index).map_err(|_| {
            BackendError::Inference("Classifier canonical row index exceeds u32".into())
        })?);
    }
    Ok(())
}

#[cfg(test)]
fn canonical_indices(count: usize) -> Result<Vec<u32>, BackendError> {
    let mut indices = Vec::new();
    canonicalize_indices(&mut indices, count)?;
    Ok(indices)
}

fn copied(values: &[f32]) -> Result<Vec<f32>, BackendError> {
    let mut row = Vec::new();
    row.try_reserve_exact(values.len())
        .map_err(|error| BackendError::Overloaded(error.to_string()))?;
    row.extend_from_slice(values);
    Ok(row)
}

pub(crate) fn pack_predictions(
    output: &[f32],
    rows: usize,
    width: usize,
) -> Result<Predictions, BackendError> {
    let expected = product(rows, width).map_err(|_| {
        BackendError::Inference("Classifier prediction element count overflow".into())
    })?;
    if output.len() != expected {
        return Err(infer("Classifier prediction payload length mismatch"));
    }
    let mut predictions = Predictions::default();
    predictions
        .try_reserve(rows)
        .map_err(|error| BackendError::Overloaded(error.to_string()))?;
    for (index, row) in output.chunks_exact(width).enumerate() {
        predictions.insert(index, copied(row)?);
    }
    Ok(predictions)
}

pub(crate) fn pack_token_predictions(
    output: &[f32],
    boundaries: &[u32],
    width: usize,
) -> Result<TokenPredictions, BackendError> {
    if boundaries.len() < 2 || boundaries.first() != Some(&0) {
        return Err(infer("Classifier token boundaries are malformed"));
    }
    let tokens = *boundaries
        .last()
        .ok_or_else(|| infer("Classifier token boundaries are empty"))? as usize;
    let expected = product(tokens, width)
        .map_err(|_| BackendError::Inference("Classifier token element count overflow".into()))?;
    if output.len() != expected {
        return Err(infer("Classifier token payload length mismatch"));
    }
    let sequences = boundaries.len() - 1;
    let mut predictions = TokenPredictions::default();
    predictions
        .try_reserve(sequences)
        .map_err(|error| BackendError::Overloaded(error.to_string()))?;
    let mut offset = 0usize;
    for (index, pair) in boundaries.windows(2).enumerate() {
        let (Some(&start), Some(&end)) = (pair.first(), pair.get(1)) else {
            return Err(infer("Classifier token boundaries are malformed"));
        };
        let rows = end
            .checked_sub(start)
            .filter(|n| *n > 0)
            .ok_or_else(|| infer("Classifier token sequence has an invalid boundary"))?
            as usize;
        let elements = product(rows, width)?;
        let source = output
            .get(offset..offset.saturating_add(elements))
            .ok_or_else(|| infer("Classifier token payload is too short"))?;
        let mut tokens = Vec::new();
        tokens
            .try_reserve_exact(rows)
            .map_err(|error| BackendError::Overloaded(error.to_string()))?;
        for row in source.chunks_exact(width) {
            tokens.push(copied(row)?);
        }
        predictions.insert(index, tokens);
        offset += elements;
    }
    if offset != output.len() {
        return Err(infer("Classifier token payload has trailing data"));
    }
    Ok(predictions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{model::read_json, Work};
    use safetensors::tensor::{serialize, TensorView};
    use safetensors::Dtype;
    use std::{
        collections::BTreeMap,
        fs,
        path::{Path, PathBuf},
        sync::atomic::{AtomicU64, Ordering},
    };

    struct Fixture(PathBuf);

    impl Fixture {
        fn new(config: serde_json::Value) -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "infinity-classifier-head-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            fs::write(
                path.join("config.json"),
                serde_json::to_vec(&config).unwrap(),
            )
            .unwrap();
            Self(path)
        }

        fn weights(&self, entries: &[(String, Vec<usize>)]) {
            let values = entries
                .iter()
                .map(|(name, shape)| {
                    let count = shape.iter().product::<usize>();
                    let bytes = (0..count)
                        .flat_map(|index| ((index % 17) as f32 * 0.125).to_le_bytes())
                        .collect::<Vec<_>>();
                    (name.clone(), shape.clone(), bytes)
                })
                .collect::<Vec<_>>();
            let mut tensors = BTreeMap::new();
            for (name, shape, bytes) in &values {
                tensors.insert(
                    name.clone(),
                    TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
                );
            }
            fs::write(
                self.0.join("model.safetensors"),
                serialize(tensors.iter().map(|(n, v)| (n.clone(), v.clone())), None).unwrap(),
            )
            .unwrap();
        }

        fn config(&self) -> BertConfig {
            BertConfig::load_classifier(&self.0).unwrap()
        }

        fn load(&self) -> Result<ClassifierHeadHost, BackendError> {
            let config = self.config();
            let checkpoint = Checkpoint::open(&self.0)?;
            ClassifierHeadHost::load(&checkpoint, &config)
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn config(
        task: &str,
        hidden: usize,
        intermediate: usize,
        layers: usize,
        vocab: usize,
    ) -> serde_json::Value {
        serde_json::json!({
            "model_type": "bert",
            "architectures": [task],
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "num_attention_heads": 12,
            "num_hidden_layers": layers,
            "max_position_embeddings": 512,
            "vocab_size": vocab,
            "type_vocab_size": 2,
            "layer_norm_eps": 1e-12,
            "hidden_act": "gelu"
        })
    }

    fn sequence_entries(with_pooler: bool) -> Vec<(String, Vec<usize>)> {
        let mut entries = vec![
            ("classifier.weight".into(), vec![1, 384]),
            ("classifier.bias".into(), vec![1]),
        ];
        if with_pooler {
            entries.push(("bert.pooler.dense.weight".into(), vec![384, 384]));
            entries.push(("bert.pooler.dense.bias".into(), vec![384]));
        }
        entries
    }

    fn token_entries(with_pooler: bool) -> Vec<(String, Vec<usize>)> {
        let mut entries = vec![
            ("classifier.weight".into(), vec![9, 768]),
            ("classifier.bias".into(), vec![9]),
        ];
        if with_pooler {
            entries.push(("bert.pooler.dense.weight".into(), vec![768, 768]));
            entries.push(("bert.pooler.dense.bias".into(), vec![768]));
        }
        entries
    }

    #[allow(clippy::too_many_arguments)]
    fn family_config(
        model_type: &str,
        architecture: &str,
        hidden: usize,
        intermediate: usize,
        heads: usize,
        layers: usize,
        vocab: usize,
        positions: usize,
        type_vocab: usize,
        eps: f64,
    ) -> serde_json::Value {
        serde_json::json!({
            "model_type": model_type,
            "architectures": [architecture],
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "num_attention_heads": heads,
            "num_hidden_layers": layers,
            "max_position_embeddings": positions,
            "vocab_size": vocab,
            "type_vocab_size": type_vocab,
            "layer_norm_eps": eps,
            "hidden_act": "gelu",
            "position_embedding_type": "absolute"
        })
    }

    fn mlp_entries(hidden: usize, width: usize) -> Vec<(String, Vec<usize>)> {
        vec![
            ("classifier.dense.weight".into(), vec![hidden, hidden]),
            ("classifier.dense.bias".into(), vec![hidden]),
            ("classifier.out_proj.weight".into(), vec![width, hidden]),
            ("classifier.out_proj.bias".into(), vec![width]),
        ]
    }

    fn plain_entries(hidden: usize, width: usize) -> Vec<(String, Vec<usize>)> {
        vec![
            ("classifier.weight".into(), vec![width, hidden]),
            ("classifier.bias".into(), vec![width]),
        ]
    }

    #[test]
    fn deberta_prediction_mode_matches_its_trained_architecture() {
        for (architecture, mode, wrong) in [
            (
                "DebertaV2ForSequenceClassification",
                ReadoutMode::Sequence,
                ReadoutMode::Tokens,
            ),
            (
                "DebertaV2ForTokenClassification",
                ReadoutMode::Tokens,
                ReadoutMode::Sequence,
            ),
        ] {
            let c = BertConfig::deberta_geometry(1000, 2, 512, 1e-7, vec![architecture.into()])
                .unwrap();
            assert!(admit_readout(&c, mode).is_ok());
            assert!(admit_readout(&c, wrong).is_err());
        }
    }

    #[test]
    fn wide_sequence_heads_reserve_the_classifier_destination() {
        let layout = ClassifierScratchLayout::for_key(768, 1024, (129, 4, 4)).unwrap();
        assert_eq!(layout.first, 4 * 1024);
        assert_eq!(layout.second, 129 * 1024);
        assert_eq!(layout.logits, 129 * 1024);
    }

    #[test]
    fn roberta_token_mlp_reserves_every_packed_hidden_row() {
        let layout =
            ClassifierScratchLayout::for_key_with_token_mlp(768, 9, (64, 16, 64), true).unwrap();
        assert_eq!(layout.first, 64 * 768);
        assert_eq!(layout.second, 16 * 768);
        assert_eq!(layout.logits, 64 * 9);
        let plain =
            ClassifierScratchLayout::for_key_with_token_mlp(768, 9, (64, 16, 64), false).unwrap();
        assert_eq!(plain.first, 16 * 768);
        assert_eq!(plain.second, 16 * 768);
    }

    #[test]
    fn admits_bert_sequence_pooler_and_token_plain_heads() {
        let sequence = Fixture::new(config("BertForSequenceClassification", 384, 1536, 6, 30522));
        sequence.weights(&sequence_entries(true));
        let host = sequence.load().unwrap();
        assert_eq!(host.readout.task, ClassifierTask::Sequence);
        assert_eq!(host.readout.width, 1);
        assert_eq!(host.hidden, 384);
        assert!(host.first.is_some());
        assert!(!host.first_tokens);

        let tokens = Fixture::new(config("BertForTokenClassification", 768, 3072, 12, 28996));
        tokens.weights(&token_entries(false));
        let host = tokens.load().unwrap();
        assert_eq!(host.readout.task, ClassifierTask::Tokens);
        assert_eq!(host.readout.width, 9);
        assert_eq!(host.hidden, 768);
        assert!(host.first.is_none());
        assert!(!host.first_tokens);

        let tokens_with_pooler =
            Fixture::new(config("BertForTokenClassification", 768, 3072, 12, 28996));
        tokens_with_pooler.weights(&token_entries(true));
        assert!(tokens_with_pooler.load().unwrap().first.is_some());
    }

    #[test]
    fn admits_roberta_family_dense_tanh_and_plain_heads_at_configured_widths() {
        for (model_type, architecture, vocab) in [
            ("roberta", "RobertaForSequenceClassification", 50265),
            ("xlm-roberta", "XLMRobertaForSequenceClassification", 250002),
            ("camembert", "CamembertForSequenceClassification", 32005),
        ] {
            let fixture = Fixture::new(family_config(
                model_type,
                architecture,
                1024,
                4096,
                16,
                24,
                vocab,
                514,
                1,
                1e-5,
            ));
            fixture.weights(&mlp_entries(1024, 3));
            let host = fixture.load().unwrap();
            assert_eq!(host.readout.task, ClassifierTask::Sequence);
            assert_eq!(host.readout.width, 3);
            assert_eq!(host.hidden, 1024);
            assert!(host.first.is_some());
            assert!(host.first_tokens);
            assert!(validate_admission(&fixture.0, None).is_ok());
        }
        for (model_type, architecture, hidden, intermediate, heads, layers, vocab) in [
            (
                "roberta",
                "RobertaForTokenClassification",
                768,
                3072,
                12,
                12,
                50265,
            ),
            (
                "xlm-roberta",
                "XLMRobertaForTokenClassification",
                1024,
                4096,
                16,
                24,
                250002,
            ),
            (
                "camembert",
                "CamembertForTokenClassification",
                768,
                3072,
                12,
                12,
                32005,
            ),
        ] {
            let fixture = Fixture::new(family_config(
                model_type,
                architecture,
                hidden,
                intermediate,
                heads,
                layers,
                vocab,
                514,
                1,
                1e-5,
            ));
            fixture.weights(&plain_entries(hidden, 9));
            let host = fixture.load().unwrap();
            assert_eq!(host.readout.task, ClassifierTask::Tokens);
            assert_eq!(host.readout.width, 9);
            assert_eq!(host.hidden, hidden);
            assert!(host.first.is_none());
            assert!(!host.first_tokens);
        }
        let token_mlp = Fixture::new(family_config(
            "roberta",
            "RobertaForTokenClassification",
            768,
            3072,
            12,
            12,
            50265,
            514,
            1,
            1e-5,
        ));
        token_mlp.weights(&mlp_entries(768, 7));
        let host = token_mlp.load().unwrap();
        assert_eq!(host.readout.task, ClassifierTask::Tokens);
        assert_eq!(host.readout.width, 7);
        assert!(host.first_tokens);
    }

    fn gte_classifier_config(architecture: &str) -> serde_json::Value {
        serde_json::json!({
            "model_type": "new",
            "architectures": [architecture],
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_attention_heads": 12,
            "num_hidden_layers": 1,
            "max_position_embeddings": 8192,
            "vocab_size": 250048,
            "type_vocab_size": 0,
            "layer_norm_eps": 1e-12,
            "hidden_act": "gelu",
            "position_embedding_type": "rope",
            "rope_theta": 20000.0,
            "rope_scaling": {"factor": 8.0, "type": "ntk"},
            "layer_norm_type": "layer_norm",
            "pack_qkv": true
        })
    }

    fn modern_classifier_config(
        architecture: &str,
        hidden: usize,
        heads: usize,
        intermediate: usize,
    ) -> serde_json::Value {
        serde_json::json!({
            "model_type": "modernbert",
            "architectures": [architecture],
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "num_attention_heads": heads,
            "num_hidden_layers": 1,
            "max_position_embeddings": 8192,
            "vocab_size": 50368,
            "norm_eps": 1e-5,
            "hidden_activation": "gelu",
            "classifier_activation": "gelu",
            "classifier_pooling": "mean",
            "classifier_bias": false,
            "attention_bias": false,
            "mlp_bias": false,
            "norm_bias": false,
            "global_attn_every_n_layers": 3,
            "local_attention": 128,
            "global_rope_theta": 160000.0,
            "local_rope_theta": 10000.0,
            "position_embedding_type": "absolute"
        })
    }

    fn modern_sequence_entries(hidden: usize, width: usize) -> Vec<(String, Vec<usize>)> {
        vec![
            ("classifier.weight".into(), vec![width, hidden]),
            ("head.dense.weight".into(), vec![hidden, hidden]),
            ("head.norm.weight".into(), vec![hidden]),
        ]
    }

    #[test]
    fn admits_gte_multilingual_sequence_pooler_precedence_and_rejects_tokens() {
        let fixture = Fixture::new(gte_classifier_config("NewForSequenceClassification"));
        fixture.weights(&[
            ("classifier.weight".into(), vec![1, 768]),
            ("classifier.bias".into(), vec![1]),
            ("new.pooler.dense.weight".into(), vec![768, 768]),
            ("new.pooler.dense.bias".into(), vec![768]),
            ("pooler.dense.weight".into(), vec![768, 768]),
            ("pooler.dense.bias".into(), vec![768]),
        ]);
        let host = fixture.load().unwrap();
        assert_eq!(host.readout.task, ClassifierTask::Sequence);
        assert_eq!(host.readout.width, 1);
        assert_eq!(host.hidden, 768);
        assert!(host.first.is_some());
        assert!(host.first_sequence);
        assert!(!host.first_tokens);
        assert_eq!(host.pooling, ClassifierPooling::First);

        let fallback = Fixture::new(gte_classifier_config("NewForSequenceClassification"));
        fallback.weights(&[
            ("classifier.weight".into(), vec![1, 768]),
            ("classifier.bias".into(), vec![1]),
            ("new.pooler.dense.weight".into(), vec![768, 768]),
            ("pooler.dense.weight".into(), vec![768, 768]),
            ("pooler.dense.bias".into(), vec![768]),
        ]);
        assert!(fallback.load().unwrap().first.is_some());

        let incomplete = Fixture::new(gte_classifier_config("NewForSequenceClassification"));
        incomplete.weights(&[
            ("classifier.weight".into(), vec![1, 768]),
            ("classifier.bias".into(), vec![1]),
            ("new.pooler.dense.weight".into(), vec![768, 768]),
        ]);
        assert!(incomplete.load().unwrap().first.is_none());

        let token = Fixture::new(gte_classifier_config("NewForTokenClassification"));
        token.weights(&[
            ("classifier.weight".into(), vec![1, 768]),
            ("classifier.bias".into(), vec![1]),
        ]);
        let error = token.load().unwrap_err().to_string();
        assert!(
            error.contains("GTE does not support token classification"),
            "{error}"
        );
    }

    #[test]
    fn admits_modern_sequence_mean_gelu_and_token_wide_heads_from_tensors() {
        let sequence = Fixture::new(modern_classifier_config(
            "ModernBertForSequenceClassification",
            768,
            12,
            1152,
        ));
        let config =
            crate::modern_encoder_model::ModernConfig::load_classifier(&sequence.0).unwrap();
        assert_eq!(config.geometry.hidden_size, 768);
        assert_eq!(config.geometry.num_attention_heads, 12);
        assert_eq!(config.geometry.intermediate_size, 1152);
        let policy = config.classifier.unwrap();
        assert_eq!(policy.task, ClassifierTask::Sequence);
        assert_eq!(policy.pooling, ClassifierPooling::Mean);
        assert_eq!(policy.activation, HeadActivation::GeluTanh);
        assert_eq!(policy.classifier_bias, Some(false));
        sequence.weights(&modern_sequence_entries(768, 1));
        let checkpoint = Checkpoint::open(&sequence.0).unwrap();
        let host = ClassifierHeadHost::load_modern(&checkpoint, &config).unwrap();
        assert_eq!(host.readout.task, ClassifierTask::Sequence);
        assert_eq!(host.readout.width, 1);
        assert_eq!(host.hidden, 768);
        assert!(host.first.is_some());
        assert!(host.requires_first_for_sequence);
        assert_eq!(host.pooling, ClassifierPooling::Mean);
        assert!(host.classifier.bias.is_none());

        let token = Fixture::new(modern_classifier_config(
            "ModernBertForTokenClassification",
            1024,
            16,
            2624,
        ));
        let token_config =
            crate::modern_encoder_model::ModernConfig::load_classifier(&token.0).unwrap();
        assert_eq!(token_config.geometry.hidden_size, 1024);
        assert_eq!(token_config.geometry.num_attention_heads, 16);
        assert_eq!(token_config.geometry.intermediate_size, 2624);
        token.weights(&modern_sequence_entries(1024, 3));
        let checkpoint = Checkpoint::open(&token.0).unwrap();
        let host = ClassifierHeadHost::load_modern(&checkpoint, &token_config).unwrap();
        assert_eq!(host.readout.task, ClassifierTask::Tokens);
        assert_eq!(host.readout.width, 3);
        assert_eq!(host.hidden, 1024);
        assert!(host.first.is_some());

        let token_plain = Fixture::new(modern_classifier_config(
            "ModernBertForTokenClassification",
            1024,
            16,
            2624,
        ));
        token_plain.weights(&[("classifier.weight".into(), vec![3, 1024])]);
        let token_config =
            crate::modern_encoder_model::ModernConfig::load_classifier(&token_plain.0).unwrap();
        let checkpoint = Checkpoint::open(&token_plain.0).unwrap();
        let host = ClassifierHeadHost::load_modern(&checkpoint, &token_config).unwrap();
        assert!(host.first.is_none());
        assert_eq!(host.readout.task, ClassifierTask::Tokens);
    }

    #[test]
    fn rejects_incomplete_modern_heads_bias_mismatch_and_sequence_without_head() {
        let missing_norm = Fixture::new(modern_classifier_config(
            "ModernBertForSequenceClassification",
            768,
            12,
            1152,
        ));
        let config =
            crate::modern_encoder_model::ModernConfig::load_classifier(&missing_norm.0).unwrap();
        missing_norm.weights(&[
            ("classifier.weight".into(), vec![1, 768]),
            ("head.dense.weight".into(), vec![768, 768]),
        ]);
        let checkpoint = Checkpoint::open(&missing_norm.0).unwrap();
        assert!(ClassifierHeadHost::load_modern(&checkpoint, &config).is_err());

        let biased = Fixture::new(modern_classifier_config(
            "ModernBertForSequenceClassification",
            768,
            12,
            1152,
        ));
        let config = crate::modern_encoder_model::ModernConfig::load_classifier(&biased.0).unwrap();
        biased.weights(&[
            ("classifier.weight".into(), vec![1, 768]),
            ("classifier.bias".into(), vec![1]),
            ("head.dense.weight".into(), vec![768, 768]),
            ("head.norm.weight".into(), vec![768]),
        ]);
        let checkpoint = Checkpoint::open(&biased.0).unwrap();
        let error = ClassifierHeadHost::load_modern(&checkpoint, &config)
            .unwrap_err()
            .to_string();
        assert!(error.contains("classifier_bias"), "{error}");

        let head_bias = Fixture::new(modern_classifier_config(
            "ModernBertForSequenceClassification",
            768,
            12,
            1152,
        ));
        let config =
            crate::modern_encoder_model::ModernConfig::load_classifier(&head_bias.0).unwrap();
        head_bias.weights(&[
            ("classifier.weight".into(), vec![1, 768]),
            ("head.dense.weight".into(), vec![768, 768]),
            ("head.dense.bias".into(), vec![768]),
            ("head.norm.weight".into(), vec![768]),
        ]);
        let checkpoint = Checkpoint::open(&head_bias.0).unwrap();
        let error = ClassifierHeadHost::load_modern(&checkpoint, &config)
            .unwrap_err()
            .to_string();
        assert!(error.contains("head.dense.bias"), "{error}");
    }

    #[test]
    fn modern_pooling_scratch_is_bounded_and_only_reserved_for_pooling_heads() {
        let key = (64usize, 4usize, 4usize);
        let plain =
            ClassifierScratchLayout::for_key_with_pooling(1024, 3, key, false, false).unwrap();
        let pooled =
            ClassifierScratchLayout::for_key_with_pooling(1024, 3, key, false, true).unwrap();
        assert_eq!(pooled.pooled, 4 * 1024);
        assert_eq!(
            pooled.bytes().unwrap() - plain.bytes().unwrap(),
            4 * 1024 * 4
        );
        assert!(ClassifierScratchLayout::for_key_with_pooling(
            8,
            3,
            (usize::MAX, 2, 1),
            false,
            true
        )
        .is_err());
    }

    #[test]
    fn rejects_mismatched_architectures_geometry_and_incomplete_heads() {
        let wrong_architecture = Fixture::new(family_config(
            "roberta",
            "XLMRobertaForTokenClassification",
            768,
            3072,
            12,
            12,
            50265,
            514,
            1,
            1e-5,
        ));
        assert!(BertConfig::load_classifier(&wrong_architecture.0).is_err());

        let wrong_recipe = Fixture::new(serde_json::json!({
            "model_type":"bert",
            "architectures":["BertForSequenceClassification"],
            "hidden_size":384,"intermediate_size":1536,
            "num_attention_heads":12,"num_hidden_layers":6,
            "max_position_embeddings":512,"vocab_size":30522,
            "type_vocab_size":2,"layer_norm_eps":1e-12,"hidden_act":"relu"
        }));
        assert!(BertConfig::load_classifier(&wrong_recipe.0).is_err());

        let optional_pooler =
            Fixture::new(config("BertForSequenceClassification", 384, 1536, 6, 30522));
        optional_pooler.weights(&sequence_entries(false));
        assert!(optional_pooler.load().unwrap().first.is_none());

        let incomplete_pooler =
            Fixture::new(config("BertForSequenceClassification", 384, 1536, 6, 30522));
        incomplete_pooler.weights(&[
            ("classifier.weight".into(), vec![1, 384]),
            ("classifier.bias".into(), vec![1]),
            ("bert.pooler.dense.weight".into(), vec![384, 384]),
        ]);
        assert!(incomplete_pooler.load().is_err());

        let partial_mlp =
            Fixture::new(config("BertForSequenceClassification", 384, 1536, 6, 30522));
        partial_mlp.weights(&[
            ("classifier.dense.weight".into(), vec![384, 384]),
            ("classifier.dense.bias".into(), vec![384]),
        ]);
        assert!(partial_mlp.load().is_err());

        let wrong_out = Fixture::new(family_config(
            "roberta",
            "RobertaForSequenceClassification",
            768,
            3072,
            12,
            12,
            50265,
            514,
            1,
            1e-5,
        ));
        wrong_out.weights(&[
            ("classifier.dense.weight".into(), vec![768, 768]),
            ("classifier.dense.bias".into(), vec![768]),
            ("classifier.out_proj.weight".into(), vec![3, 384]),
            ("classifier.out_proj.bias".into(), vec![3]),
        ]);
        assert!(wrong_out.load().is_err());
    }

    #[test]
    fn ordinary_config_loader_still_rejects_classifier_architectures() {
        let fixture = Fixture::new(config("BertForSequenceClassification", 384, 1536, 6, 30522));
        assert!(BertConfig::load(&fixture.0).is_err());
        assert!(BertConfig::load_classifier(&fixture.0).is_ok());
        let raw: serde_json::Value = read_json(&fixture.0.join("config.json")).unwrap();
        assert_eq!(raw["architectures"][0], "BertForSequenceClassification");
    }

    #[test]
    fn facade_empty_dense_selection_ignores_discovery_but_explicit_override_rejects() {
        let fixture = Fixture::new(config("BertForSequenceClassification", 384, 1536, 6, 30522));
        fixture.weights(&sequence_entries(false));
        std::fs::write(fixture.0.join("modules.json"), br#"[{"idx":0,"name":"0","path":"","type":"sentence_transformers.models.Transformer"},{"idx":1,"name":"1","path":"missing_dense","type":"sentence_transformers.models.Dense"}]"#).unwrap();
        // This is the actual value supplied by oxide_effective_dense_paths for
        // a classifier's discovered modules. It reaches head admission without
        // trying to load the irrelevant embedding Dense module.
        assert!(validate_admission(&fixture.0, Some(Vec::new())).is_ok());
        assert!(validate_admission(&fixture.0, Some(vec!["missing_dense".into()])).is_err());
        // Direct API discovery is a separate contract and still validates its
        // declared embedding metadata; it is not the normal facade path.
        assert!(validate_admission(&fixture.0, None).is_err());
    }

    #[test]
    fn actual_admission_rejects_zero_classifier_width_before_cuda() {
        let fixture = Fixture::new(config("BertForSequenceClassification", 384, 1536, 6, 30522));
        fixture.weights(&[
            ("classifier.weight".into(), vec![0, 384]),
            ("classifier.bias".into(), vec![0]),
        ]);
        let error = validate_admission(&fixture.0, Some(Vec::new()))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("width") || error.contains("shape"),
            "{error}"
        );
    }

    #[test]
    fn actual_admission_rejects_integer_head_storage_before_cuda() {
        let fixture = Fixture::new(config("BertForSequenceClassification", 384, 1536, 6, 30522));
        let weight_bytes = vec![0u8; 384 * 8];
        let bias_bytes = vec![0u8; 4];
        let tensors = BTreeMap::from([
            (
                "classifier.weight",
                TensorView::new(Dtype::I64, vec![1, 384], &weight_bytes).unwrap(),
            ),
            (
                "classifier.bias",
                TensorView::new(Dtype::F32, vec![1], &bias_bytes).unwrap(),
            ),
        ]);
        fs::write(
            fixture.0.join("model.safetensors"),
            serialize(tensors.iter().map(|(n, v)| (*n, v.clone())), None).unwrap(),
        )
        .unwrap();
        let error = validate_admission(&fixture.0, Some(Vec::new()))
            .unwrap_err()
            .to_string();
        assert!(error.contains("dtype") || error.contains("I64"), "{error}");
    }

    #[test]
    fn dispatch_matches_native_bias_pointer_shape_gate_including_single_rows() {
        assert!(!crate::biased_linear::selects_native_lt(1, 384, 1));
        assert!(!crate::biased_linear::selects_native_lt(5, 384, 1));
        assert!(crate::biased_linear::selects_native_lt(1, 384, 384));
        assert!(crate::biased_linear::selects_native_lt(2, 384, 384));
        assert!(crate::biased_linear::selects_native_lt(1, 768, 9));
        assert!(crate::biased_linear::selects_native_lt(2, 768, 9));
    }

    #[test]
    fn readout_packing_preserves_sequence_and_token_order() {
        let sequence = pack_predictions(&[1., 2., 3.], 3, 1).unwrap();
        assert_eq!(sequence.get(&0), Some(&vec![1.]));
        assert_eq!(sequence.get(&1), Some(&vec![2.]));
        assert_eq!(sequence.get(&2), Some(&vec![3.]));
        assert!(pack_predictions(&[1., 2.], 3, 1).is_err());

        let tokens = pack_token_predictions(&[0., 1., 2., 3., 4., 5.], &[0, 2, 3], 2).unwrap();
        assert_eq!(tokens.get(&0).unwrap(), &vec![vec![0., 1.], vec![2., 3.]]);
        assert_eq!(tokens.get(&1).unwrap(), &vec![vec![4., 5.]]);
        assert!(pack_token_predictions(&[0.; 4], &[0, 0, 2], 2).is_err());
        assert!(pack_token_predictions(&[0.; 3], &[0, 2], 2).is_err());
    }

    #[test]
    fn canonical_indices_override_supplied_selections_and_reject_overflow() {
        let indices = canonical_indices(4).unwrap();
        assert_eq!(indices, vec![0, 1, 2, 3]);
        assert_eq!(
            canonical_indices(0).unwrap(),
            Vec::<u32>::new(),
            "empty canonical batches are rejected later by admission"
        );
        assert!(canonical_indices(u32::MAX as usize + 1).is_err());
    }

    #[test]
    fn canonicalization_reuses_capacity_and_preserves_overflow_rejection() {
        let mut values = Vec::with_capacity(64);
        values.extend([42, 42]);
        let pointer = values.as_ptr();
        canonicalize_indices(&mut values, 32).unwrap();
        assert_eq!(values.as_ptr(), pointer);
        assert_eq!(values, (0..32).collect::<Vec<u32>>());
        canonicalize_indices(&mut values, 2).unwrap();
        assert_eq!(values.as_ptr(), pointer);
        assert_eq!(values, [0, 1]);
        assert!(canonicalize_indices(&mut values, u32::MAX as usize + 1).is_err());
        assert_eq!(values, [0, 1], "invalid admission must preserve the buffer");
    }

    #[test]
    fn native_pooler_namespace_ignores_unprefixed_pooler_tensors() {
        let fixture = Fixture::new(config("BertForSequenceClassification", 384, 1536, 6, 30522));
        let mut entries = sequence_entries(false);
        entries.push(("pooler.dense.weight".into(), vec![384, 384]));
        entries.push(("pooler.dense.bias".into(), vec![384]));
        fixture.weights(&entries);
        assert!(fixture.load().unwrap().first.is_none());
        entries.extend(sequence_entries(true).into_iter().skip(2));
        fixture.weights(&entries);
        assert!(fixture.load().unwrap().first.is_some());
    }

    #[test]
    fn classifier_admission_rejects_decoder_before_cuda_and_discovers_dense() {
        let decoder = Fixture::new(serde_json::json!({"model_type":"llama", "hidden_size":1024}));
        assert!(validate_admission(&decoder.0, None).is_err());
        let missing_head =
            Fixture::new(config("BertForSequenceClassification", 384, 1536, 6, 30522));
        assert!(
            validate_admission(&missing_head.0, None).is_err(),
            "cold admission must require the trained head tensors, not only config"
        );
        let fixture = Fixture::new(config("BertForSequenceClassification", 384, 1536, 6, 30522));
        fixture.weights(&sequence_entries(true));
        assert!(validate_admission(&fixture.0, None).is_ok());
        // Transformer.path is deliberately absent; it cannot hide a Dense head.
        fs::write(
            fixture.0.join("modules.json"),
            serde_json::to_vec(&serde_json::json!([
                {"type":"sentence_transformers.models.Transformer"},
                {"type":"sentence_transformers.models.Dense", "path":"2_Dense"}
            ]))
            .unwrap(),
        )
        .unwrap();
        let dense = fixture.0.join("2_Dense");
        fs::create_dir(&dense).unwrap();
        fs::write(
            dense.join("config.json"),
            serde_json::to_vec(&serde_json::json!({
                "in_features":384,"out_features":2,"bias":false,
                "activation_function":"torch.nn.modules.linear.Identity"
            }))
            .unwrap(),
        )
        .unwrap();
        let bytes = vec![0u8; 2 * 384 * 4];
        let view = TensorView::new(Dtype::F32, vec![2, 384], &bytes).unwrap();
        fs::write(
            dense.join("model.safetensors"),
            serialize([("linear.weight", view)], None).unwrap(),
        )
        .unwrap();
        assert!(
            validate_admission(&fixture.0, None).is_ok(),
            "discovered embedding Dense modules are ignored by classifiers, matching Native"
        );
        let error = validate_admission(&fixture.0, Some(vec!["2_Dense".into()]))
            .unwrap_err()
            .to_string();
        assert!(error.contains("explicit Dense"), "{error}");
        assert!(
            validate_admission(&fixture.0, Some(Vec::new())).is_ok(),
            "explicit [] suppresses Dense execution"
        );
        fs::remove_file(dense.join("model.safetensors")).unwrap();
        assert!(
            validate_admission(&fixture.0, None).is_err(),
            "malformed declared Dense metadata must never silently become backbone"
        );
    }

    #[test]
    fn scratch_layout_covers_both_readout_modes_and_checked_products() {
        let shared = ClassifierScratchLayout::for_key(384, 1, (64, 16, 16)).unwrap();
        assert_eq!(shared.first, 16 * 384);
        assert_eq!(shared.second, 16 * 384);
        assert_eq!(shared.logits, 64, "token readout uses every packed row");
        assert!(shared.bytes().unwrap() > 0);

        let tokens = ClassifierScratchLayout::for_key(768, 9, (64, 16, 64)).unwrap();
        assert_eq!(tokens.first, 16 * 768);
        assert_eq!(
            tokens.second,
            16 * 768,
            "sequence hidden scratch dominates the token logits"
        );
        assert_eq!(tokens.logits, 64 * 9);
        let token_dominant = ClassifierScratchLayout::for_key(8, 9, (512, 2, 16)).unwrap();
        assert_eq!(token_dominant.first, 2 * 9);
        assert_eq!(token_dominant.second, 512 * 9);
        assert_eq!(token_dominant.logits, 512 * 9);

        assert!(ClassifierScratchLayout::for_key(384, 1, (2, usize::MAX, 1)).is_err());
        assert!(ClassifierScratchLayout::for_key(768, 9, (usize::MAX, 2, 1)).is_err());
    }

    #[test]
    fn both_readout_modes_accept_only_their_canonical_selection() {
        let mut batch = Batch {
            input_ids: vec![1; 3],
            position_ids: vec![0, 0, 1],
            token_type_ids: vec![0; 3],
            cumulative_seq_lengths: vec![0, 1, 3],
            max_length: 2,
            pooled_indices: vec![0, 1],
            raw_indices: vec![],
            multimodal: vec![],
            compact_input_ids: None,
            compact_position_ids: None,
            scatter_unfold: None,
            fold_gather: None,
            tokens: vec![],
            offsets: vec![],
        };
        assert!(require_canonical(&batch, ReadoutMode::Sequence).is_ok());
        assert!(require_canonical(&batch, ReadoutMode::Tokens).is_err());
        batch.pooled_indices.clear();
        batch.raw_indices = vec![0, 1];
        assert!(require_canonical(&batch, ReadoutMode::Tokens).is_ok());
        assert!(require_canonical(&batch, ReadoutMode::Sequence).is_err());
        batch.raw_indices = vec![1, 0];
        assert!(require_canonical(&batch, ReadoutMode::Tokens).is_err());
    }

    #[test]
    fn classifier_result_memory_and_workspace_use_width_rows() {
        let config: BertConfig = serde_json::from_value(serde_json::json!({
            "model_type":"bert","hidden_size":4,"intermediate_size":8,
            "num_attention_heads":2,"num_hidden_layers":1,
            "max_position_embeddings":8,"vocab_size":4,"type_vocab_size":2
        }))
        .unwrap();
        let key = (32usize, 4usize, 2usize);
        let base = Work::<Bf16>::allocation_bytes(key, &config).unwrap();
        let layout = ClassifierScratchLayout::for_key(4, 1, key).unwrap();
        let with =
            Work::<Bf16>::allocation_bytes_with_classifier(key, &config, Some(layout)).unwrap();
        assert_eq!(with - base, layout.bytes().unwrap());

        let sequence = Batch {
            input_ids: vec![1; 3],
            position_ids: vec![0, 0, 1],
            token_type_ids: vec![0; 3],
            cumulative_seq_lengths: vec![0, 1, 3],
            max_length: 2,
            pooled_indices: vec![0, 1],
            raw_indices: vec![],
            multimodal: vec![],
            compact_input_ids: None,
            compact_position_ids: None,
            scatter_unfold: None,
            fold_gather: None,
            tokens: vec![],
            offsets: vec![],
        };
        assert_eq!(
            crate::output::result_bytes(&sequence, 1, 1).unwrap(),
            2 * size_of::<f32>()
        );
        let tokens = Batch {
            pooled_indices: vec![],
            raw_indices: vec![0, 1],
            ..sequence
        };
        assert_eq!(
            crate::output::OutputGeometry::new(9, 9)
                .unwrap()
                .host_elements(&tokens)
                .unwrap(),
            3 * 9
        );
        assert!(crate::output::result_bytes(&tokens, 9, 9).unwrap() > 3 * 9 * 4);
    }

    #[test]
    #[ignore = "Requires CUDA device 0 and INFINITY_OXIDE_KERNEL_MODULE with union53 classifier exports"]
    fn gpu_prepared_boundary_rejects_mismatched_output_cross_stream_and_seed_context() {
        use crate::{biased_linear::BiasedLinear, cublas::Blas, lt::LtProvider};
        use cuda_core::{Context, DeviceBuffer};
        assert!(
            Context::device_count().unwrap() > 0,
            "GPU control must execute"
        );
        let context = Context::new(0).unwrap();
        let stream = context.new_stream().unwrap();
        let other_stream = context.new_stream().unwrap();
        let provider = LtProvider::new(context.clone(), stream.clone()).unwrap();
        let blas = Blas::new(context.clone(), stream.clone()).unwrap();
        let kernels = ClassifierKernels::load(&context).unwrap();
        let x = DeviceBuffer::<Bf16>::zeroed(&stream, 8).unwrap();
        let weight = DeviceBuffer::<Bf16>::zeroed(&stream, 4).unwrap();
        let bias = DeviceBuffer::<Bf16>::zeroed(&stream, 1).unwrap();
        let mut first = DeviceBuffer::<Bf16>::zeroed(&stream, 2).unwrap();
        let mut second = DeviceBuffer::<Bf16>::zeroed(&stream, 2).unwrap();
        let mut plan = BiasedLinear::new(&provider, 4, 1).unwrap();
        context.bind_to_thread().unwrap();
        let prepared =
            unsafe { plan.prepare(&provider, &blas, 2, &stream, &x, &weight, &bias, &mut first) }
                .unwrap();
        assert!(unsafe {
            plan.launch(
                &prepared,
                &provider,
                &blas,
                &kernels.bias_seed,
                &stream,
                &x,
                &weight,
                &bias,
                &mut second,
            )
        }
        .is_err());
        assert!(unsafe {
            plan.launch(
                &prepared,
                &provider,
                &blas,
                &kernels.bias_seed,
                &other_stream,
                &x,
                &weight,
                &bias,
                &mut first,
            )
        }
        .is_err());
        if let Ok(other_context) = Context::new(0) {
            if let Ok(other_kernels) = ClassifierKernels::load(&other_context) {
                assert!(unsafe {
                    plan.launch(
                        &prepared,
                        &provider,
                        &blas,
                        &other_kernels.bias_seed,
                        &stream,
                        &x,
                        &weight,
                        &bias,
                        &mut first,
                    )
                }
                .is_err());
            }
        }
        unsafe {
            plan.launch(
                &prepared,
                &provider,
                &blas,
                &kernels.bias_seed,
                &stream,
                &x,
                &weight,
                &bias,
                &mut first,
            )
        }
        .unwrap();
        stream.synchronize().unwrap();
    }

    #[test]
    fn host_boundary_spans_are_checked_without_claiming_device_validation() {
        assert!(validate_bounds(&[0, 2, 5], 5, 2).is_ok());
        assert!(validate_bounds(&[0, 2, 5], 6, 2).is_err());
        assert!(validate_bounds(&[0, 2, 2], 2, 2).is_err());
        assert!(validate_bounds(&[0, 2], 2, 2).is_err());
        assert!(validate_bounds(&[1, 2, 5], 5, 2).is_err());
        assert!(validate_bounds(&[0, 2, 5], 5, 1).is_err());
    }

    fn ragged_gather_fixture() -> ([u16; 18], [u32; 4], [u16; 9]) {
        (
            [
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            ],
            [0, 2, 5, 6],
            [1, 2, 3, 7, 8, 9, 16, 17, 18],
        )
    }

    #[test]
    fn gather_source_admission_covers_ragged_control_extent_and_overflow() {
        let (source, bounds, _) = ragged_gather_fixture();
        validate_bounds(&bounds, 6, 3).unwrap();
        validate_source_span(source.len(), 6, 3).unwrap();
        assert!(
            validate_source_span(12, 6, 3).is_err(),
            "old malformed fixture must reject before launch"
        );
        assert!(validate_source_span(17, 6, 3).is_err());
        assert!(validate_source_span(usize::MAX, usize::MAX, 2).is_err());
    }

    #[test]
    #[ignore = "Root GPU: actual MiniLM classifier and kernels53 required"]
    fn gpu_sequence_source_span_and_identity_reject_before_gather_then_recover() {
        assert!(
            Context::device_count().unwrap() > 0,
            "GPU control must execute"
        );
        let model = PathBuf::from(
            std::env::var_os("INFINITY_OXIDE_CLASSIFIER_MINILM_MODEL")
                .expect("set actual MiniLM classifier"),
        );
        let config = BertConfig::load_classifier(&model).unwrap();
        let checkpoint = Checkpoint::open(&model).unwrap();
        let context = Context::new(0).unwrap();
        let stream = context.new_stream().unwrap();
        let provider = LtProvider::new_native_default(context.clone(), stream.clone()).unwrap();
        let blas = Blas::new(context.clone(), stream.clone()).unwrap();
        let mut head = ClassifierHead::new(
            ClassifierHeadHost::load(&checkpoint, &config).unwrap(),
            &context,
            &stream,
            &provider,
        )
        .unwrap();
        let batch = Batch {
            input_ids: vec![0; 6],
            position_ids: vec![0, 1, 0, 1, 2, 3],
            token_type_ids: vec![0; 6],
            cumulative_seq_lengths: vec![0, 2, 6],
            max_length: 4,
            pooled_indices: vec![0, 1],
            raw_indices: vec![],
            multimodal: vec![],
            compact_input_ids: None,
            compact_position_ids: None,
            scatter_unfold: None,
            fold_gather: None,
            tokens: vec![],
            offsets: vec![],
        };
        let mut scratch =
            ClassifierScratchBuffers::new(stream.clone(), head.scratch_layout((6, 2, 2)).unwrap())
                .unwrap();
        let source = DeviceBuffer::<Bf16>::zeroed(&stream, 6 * 384).unwrap();
        let swapped = DeviceBuffer::<Bf16>::zeroed(&stream, 6 * 384).unwrap();
        let short = DeviceBuffer::<Bf16>::zeroed(&stream, 6 * 384 - 1).unwrap();
        let bounds = DeviceBuffer::from_host(&stream, &batch.cumulative_seq_lengths).unwrap();
        stream.synchronize().unwrap();
        context.bind_to_thread().unwrap();
        assert!(matches!(
            unsafe {
                head.prepare(
                    &provider,
                    &blas,
                    &batch,
                    ReadoutMode::Sequence,
                    &stream,
                    &short,
                    &mut scratch,
                )
            },
            Err(ShapeError::Rejected(_))
        ));
        unsafe {
            head.prepare(
                &provider,
                &blas,
                &batch,
                ReadoutMode::Sequence,
                &stream,
                &source,
                &mut scratch,
            )
        }
        .unwrap();
        assert!(unsafe {
            head.enqueue(
                &provider,
                &blas,
                &stream,
                &swapped,
                bounds.cu_deviceptr(),
                0,
                &mut scratch,
            )
        }
        .is_err());
        unsafe {
            head.prepare(
                &provider,
                &blas,
                &batch,
                ReadoutMode::Sequence,
                &stream,
                &source,
                &mut scratch,
            )
        }
        .unwrap();
        unsafe {
            head.enqueue(
                &provider,
                &blas,
                &stream,
                &source,
                bounds.cu_deviceptr(),
                0,
                &mut scratch,
            )
        }
        .unwrap();
        // Sequence mode writes B*C, while this shared arena can hold T*C.
        // Read only the initialized logical prefix, just like HostIoLease.
        let mut logical_output = [0f32; 2];
        unsafe {
            cuda_core::memory::memcpy_dtoh_async(
                logical_output.as_mut_ptr(),
                scratch.logits().cu_deviceptr(),
                std::mem::size_of_val(&logical_output),
                stream.cu_stream(),
            )
        }
        .unwrap();
        stream.synchronize().unwrap();
        assert!(logical_output.iter().all(|x| x.is_finite()));
    }

    #[test]
    #[ignore = "Requires CUDA device 0 and INFINITY_OXIDE_KERNEL_MODULE with union53 classifier exports"]
    fn gpu_gather_uses_uploaded_device_bounds_and_exact_bf16_bits() {
        use cuda_core::{memory, Context, DeviceBuffer};
        assert!(
            Context::device_count().unwrap() > 0,
            "GPU control must execute"
        );
        let context = Context::new(0).unwrap();
        let stream = context.new_stream().unwrap();
        let kernels = ClassifierKernels::load(&context).unwrap();
        let (source_bits, bounds, expected) = ragged_gather_fixture();
        let source = DeviceBuffer::<Bf16>::zeroed(&stream, source_bits.len()).unwrap();
        let device_bounds = DeviceBuffer::<u32>::zeroed(&stream, bounds.len()).unwrap();
        let output = DeviceBuffer::<Bf16>::zeroed(&stream, 9).unwrap();
        unsafe {
            memory::memcpy_htod_async(
                source.cu_deviceptr(),
                source_bits.as_ptr(),
                source_bits.len() * 2,
                stream.cu_stream(),
            )
            .unwrap();
            memory::memcpy_htod_async(
                device_bounds.cu_deviceptr(),
                bounds.as_ptr(),
                bounds.len() * 4,
                stream.cu_stream(),
            )
            .unwrap();
        }
        context.bind_to_thread().unwrap();
        gather_first(
            &kernels,
            &stream,
            &source,
            device_bounds.cu_deviceptr(),
            &output,
            3,
            3,
        )
        .unwrap();
        stream.synchronize().unwrap();
        let mut actual = [0u16; 9];
        unsafe {
            memory::memcpy_dtoh_async(
                actual.as_mut_ptr(),
                output.cu_deviceptr(),
                actual.len() * 2,
                stream.cu_stream(),
            )
        }
        .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(actual, expected);
    }

    fn _assert_common_imports(_: &Path) {}
}

#[cfg(test)]
#[path = "classifier_head/gold_control.rs"]
mod gold_control;
