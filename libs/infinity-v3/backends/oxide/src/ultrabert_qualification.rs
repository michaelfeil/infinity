// Copyright Infinity Core Contributors
// Licensed under the repository LICENSE.

//! UltraBERT checkpoint validation and execution planning.
//!
//! The ordinary factory uses these CPU checks before creating the GPU executor:
//! config geometry, tensor names/shapes/dtypes, trained heads, per-layer RoPE
//! bases and symmetric attention windows. Matching FA4 profiles are required.
//! The separate qualification report records validation coverage; it does not
//! establish whole-model numerical or performance qualification.
//!
//! Loading a plan performs no CUDA work and does not rewrite checkpoint files.
//! Bare (`embed_tokens.weight`, `layers.*`, `norm.weight`) and consistent
//! prefixed namespaces are supported.

use crate::fa4::{matching_profile_name, AttentionSpec, DType as Fa4DType, Scale, Window};
use crate::model::{read_json, Checkpoint, WeightSource};
use crate::ultrabert_executor::UltraBertGpuRequirements;
use infinity_v3_backend_core::BackendError;
use safetensors::Dtype as StoredDtype;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Component, Path};

pub const QUALIFICATION_MARKER: &str = "ultrabert-qualification-only-v1";
const DEFAULT_MAX_POSITIONS: usize = 32_768;
const DEFAULT_NUM_LABELS: usize = 2;
const DEFAULT_PAD_TOKEN_ID: u32 = 0;
const MAX_LAYERS: usize = 128;
const MAX_POSITIONS_LIMIT: usize = 1 << 20;
const MAX_HIDDEN: usize = 16_384;
const MAX_INTERMEDIATE: usize = 65_536;

/// Task selected by the checkpoint architecture. An override may not turn an
/// embedding checkpoint into a classifier or drop a trained classifier head.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UltraBertTask {
    Embedding,
    SequenceClassification,
    TokenClassification,
    Decision,
}

impl UltraBertTask {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Embedding => "embedding",
            Self::SequenceClassification => "sequence_classification",
            Self::TokenClassification => "token_classification",
            Self::Decision => "decision",
        }
    }
    pub(crate) const fn requires_classifier(self) -> bool {
        matches!(
            self,
            Self::SequenceClassification | Self::TokenClassification
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UltraBertPooling {
    Cls,
    Mean,
    LastToken,
    Token,
}

/// Typed rejection or execution block. The `code` is stable for CPU checks and
/// the `Display` text is suitable for a startup error.
#[derive(Clone, Debug, PartialEq)]
pub enum UltraBertError {
    UnsupportedModelType {
        model_type: String,
    },
    MissingArchitectures,
    AmbiguousArchitectures {
        count: usize,
    },
    UnsupportedArchitecture {
        architecture: String,
    },
    TaskOverrideMismatch {
        inferred: UltraBertTask,
        requested: UltraBertTask,
    },
    MissingConfigField {
        field: &'static str,
    },
    ZeroConfigField {
        field: &'static str,
    },
    ConfigFieldTooLarge {
        field: &'static str,
        value: usize,
        max: usize,
    },
    HiddenHeadsMismatch {
        hidden: usize,
        heads: usize,
    },
    HeadDimMismatch {
        head_dim: usize,
        expected: usize,
    },
    UnsupportedKvHeads {
        heads: usize,
        kv_heads: usize,
    },
    LayerListLength {
        field: &'static str,
        layers: usize,
        got: usize,
    },
    UnsupportedActivation {
        activation: String,
    },
    BiasNotAdmitted {
        field: &'static str,
    },
    CausalNotAdmitted,
    InvalidEps {
        eps: f64,
    },
    InvalidRopeBase {
        layer: usize,
    },
    InvalidAttentionWindow {
        layer: usize,
    },
    MixedAttentionWindows,
    QuantizedCheckpoint,
    UnsupportedConfigModifier {
        key: String,
    },
    UnsupportedNamespace {
        name: String,
    },
    AmbiguousRoot {
        count: usize,
    },
    MissingTensor {
        name: String,
    },
    UnexpectedTensor {
        name: String,
    },
    ShapeMismatch {
        name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    UnsupportedStoredDtype {
        name: String,
        dtype: String,
    },
    UnsupportedRuntimeDtype {
        dtype: String,
    },
    ClassifierMismatch {
        expected_labels: usize,
        actual_labels: usize,
    },
    MissingClassifier,
    UnexpectedClassifier,
    UnsupportedModule {
        module: String,
    },
    MissingPooling,
    UnsupportedPooling {
        mode: String,
    },
    Checkpoint {
        message: String,
    },
    BatchEmpty,
    BatchOffsets {
        message: String,
    },
    SequenceTooLong {
        length: usize,
        max: usize,
    },
    TokenOutOfRange {
        token: u32,
        vocab: usize,
    },
    MissingFa4Profile {
        scope: String,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        window: String,
    },
    ExecutionSetup {
        message: String,
    },
    ExecutionModeMismatch {
        checkpoint: UltraBertTask,
        requested: &'static str,
    },
}

impl UltraBertError {
    pub const fn code(&self) -> &'static str {
        match self {
            Self::UnsupportedModelType { .. } => "unsupported_model_type",
            Self::MissingArchitectures => "missing_architectures",
            Self::AmbiguousArchitectures { .. } => "ambiguous_architectures",
            Self::UnsupportedArchitecture { .. } => "unsupported_architecture",
            Self::TaskOverrideMismatch { .. } => "task_override_mismatch",
            Self::MissingConfigField { .. } => "missing_config_field",
            Self::ZeroConfigField { .. } => "zero_config_field",
            Self::ConfigFieldTooLarge { .. } => "config_field_too_large",
            Self::HiddenHeadsMismatch { .. } => "hidden_heads_mismatch",
            Self::HeadDimMismatch { .. } => "head_dim_mismatch",
            Self::UnsupportedKvHeads { .. } => "unsupported_kv_heads",
            Self::LayerListLength { .. } => "layer_list_length",
            Self::UnsupportedActivation { .. } => "unsupported_activation",
            Self::BiasNotAdmitted { .. } => "bias_not_admitted",
            Self::CausalNotAdmitted => "causal_not_admitted",
            Self::InvalidEps { .. } => "invalid_eps",
            Self::InvalidRopeBase { .. } => "invalid_rope_base",
            Self::InvalidAttentionWindow { .. } => "invalid_attention_window",
            Self::MixedAttentionWindows => "mixed_attention_windows",
            Self::QuantizedCheckpoint => "quantized_checkpoint",
            Self::UnsupportedConfigModifier { .. } => "unsupported_config_modifier",
            Self::UnsupportedNamespace { .. } => "unsupported_namespace",
            Self::AmbiguousRoot { .. } => "ambiguous_root",
            Self::MissingTensor { .. } => "missing_tensor",
            Self::UnexpectedTensor { .. } => "unexpected_tensor",
            Self::ShapeMismatch { .. } => "shape_mismatch",
            Self::UnsupportedStoredDtype { .. } => "unsupported_stored_dtype",
            Self::UnsupportedRuntimeDtype { .. } => "unsupported_runtime_dtype",
            Self::ClassifierMismatch { .. } => "classifier_mismatch",
            Self::MissingClassifier => "missing_classifier",
            Self::UnexpectedClassifier => "unexpected_classifier",
            Self::UnsupportedModule { .. } => "unsupported_module",
            Self::MissingPooling => "missing_pooling",
            Self::UnsupportedPooling { .. } => "unsupported_pooling",
            Self::Checkpoint { .. } => "checkpoint",
            Self::BatchEmpty => "batch_empty",
            Self::BatchOffsets { .. } => "batch_offsets",
            Self::SequenceTooLong { .. } => "sequence_too_long",
            Self::TokenOutOfRange { .. } => "token_out_of_range",
            Self::MissingFa4Profile { .. } => "missing_fa4_profile",
            Self::ExecutionSetup { .. } => "execution_setup",
            Self::ExecutionModeMismatch { .. } => "execution_mode_mismatch",
        }
    }
}

impl fmt::Display for UltraBertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedModelType { model_type } => {
                write!(f, "unsupported model_type {model_type}; expected ultrabert")
            }
            Self::MissingArchitectures => write!(f, "config.architectures is missing or empty"),
            Self::AmbiguousArchitectures { count } => {
                write!(f, "config.architectures has {count} entries; exactly one UltraBERT task is required")
            }
            Self::UnsupportedArchitecture { architecture } => {
                write!(f, "unsupported UltraBERT architecture {architecture}")
            }
            Self::TaskOverrideMismatch { inferred, requested } => write!(
                f,
                "task override {requested:?} conflicts with checkpoint architecture {inferred:?}"
            ),
            Self::MissingConfigField { field } => write!(f, "missing config field {field}"),
            Self::ZeroConfigField { field } => write!(f, "config field {field} must be nonzero"),
            Self::ConfigFieldTooLarge { field, value, max } => {
                write!(f, "config field {field}={value} exceeds the qualification bound {max}")
            }
            Self::HiddenHeadsMismatch { hidden, heads } => write!(
                f,
                "hidden_size {hidden} must equal num_attention_heads {heads} * head_dim"
            ),
            Self::HeadDimMismatch { head_dim, expected } => {
                write!(f, "head_dim {head_dim} does not match hidden_size / heads {expected}")
            }
            Self::UnsupportedKvHeads { heads, kv_heads } => write!(
                f,
                "qualification scope requires num_key_value_heads == num_attention_heads; got {kv_heads} vs {heads}"
            ),
            Self::LayerListLength { field, layers, got } => {
                write!(f, "{field} has {got} entries for {layers} layers")
            }
            Self::UnsupportedActivation { activation } => {
                write!(f, "unsupported UltraBERT activation {activation}; dense SiLU only")
            }
            Self::BiasNotAdmitted { field } => write!(f, "{field} must be false"),
            Self::CausalNotAdmitted => write!(f, "UltraBERT qualification requires bidirectional attention"),
            Self::InvalidEps { eps } => write!(f, "rms_norm_eps {eps} must be finite and positive"),
            Self::InvalidRopeBase { layer } => {
                write!(f, "layer {layer} rope_theta must be finite and greater than one")
            }
            Self::InvalidAttentionWindow { layer } => {
                write!(f, "layer {layer} attention window must be null or positive")
            }
            Self::MixedAttentionWindows => write!(
                f,
                "mixed local attention windows are not admitted; one symmetric local window is required"
            ),
            Self::QuantizedCheckpoint => write!(f, "quantized UltraBERT checkpoints are not admitted"),
            Self::UnsupportedConfigModifier { key } => {
                write!(f, "unsupported UltraBERT config modifier {key}")
            }
            Self::UnsupportedNamespace { name } => {
                write!(f, "unsupported UltraBERT checkpoint namespace at {name}")
            }
            Self::AmbiguousRoot { count } => {
                write!(f, "ambiguous UltraBERT checkpoint root: {count} embed_tokens.weight tensors")
            }
            Self::MissingTensor { name } => write!(f, "missing UltraBERT tensor {name}"),
            Self::UnexpectedTensor { name } => {
                write!(f, "unexpected UltraBERT tensor {name}; refusing to drop trained weights")
            }
            Self::ShapeMismatch { name, expected, actual } => {
                write!(f, "tensor {name} shape {actual:?} does not match expected {expected:?}")
            }
            Self::UnsupportedStoredDtype { name, dtype } => {
                write!(f, "tensor {name} has unsupported stored dtype {dtype}")
            }
            Self::UnsupportedRuntimeDtype { dtype } => write!(
                f,
                "unsupported UltraBERT runtime dtype {dtype}; bfloat16 only"
            ),
            Self::ClassifierMismatch { expected_labels, actual_labels } => write!(
                f,
                "classifier has {actual_labels} rows but num_labels/id2label requires {expected_labels}"
            ),
            Self::MissingClassifier => write!(f, "classification checkpoint is missing classifier.weight/bias"),
            Self::UnexpectedClassifier => write!(
                f,
                "embedding checkpoint contains classifier weights; select the classification task instead of dropping the head"
            ),
            Self::UnsupportedModule { module } => {
                write!(f, "unsupported sentence-transformers module {module}; refusing to ignore it")
            }
            Self::MissingPooling => write!(f, "embedding checkpoint has no Pooling module/config"),
            Self::UnsupportedPooling { mode } => write!(f, "unsupported UltraBERT pooling mode {mode}"),
            Self::Checkpoint { message } => write!(f, "checkpoint header error: {message}"),
            Self::BatchEmpty => write!(f, "batch has no tokens"),
            Self::BatchOffsets { message } => write!(f, "invalid cumulative offsets: {message}"),
            Self::SequenceTooLong { length, max } => {
                write!(f, "sequence length {length} exceeds max_position_embeddings {max}")
            }
            Self::TokenOutOfRange { token, vocab } => {
                write!(f, "token id {token} is outside vocabulary {vocab}")
            }
            Self::MissingFa4Profile { scope, heads, kv_heads, head_dim, window } => write!(
                f,
                "missing FA4 profile for {scope} attention: heads={heads} kv_heads={kv_heads} d={head_dim} causal=false window={window}"
            ),
            Self::ExecutionSetup { message } => {
                write!(f, "UltraBERT execution setup failed: {message}")
            }
            Self::ExecutionModeMismatch { checkpoint, requested } => write!(
                f,
                "execution mode {requested} does not match checkpoint task {checkpoint:?}"
            ),
        }
    }
}

impl std::error::Error for UltraBertError {}

impl From<UltraBertError> for BackendError {
    fn from(value: UltraBertError) -> Self {
        BackendError::Start(format!("UltraBERT qualification: {value}"))
    }
}

impl From<BackendError> for UltraBertError {
    fn from(value: BackendError) -> Self {
        Self::Checkpoint {
            message: value.to_string(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct RawUltraBertConfig {
    #[serde(default)]
    architectures: Vec<String>,
    hidden_size: Option<usize>,
    intermediate_size: Option<usize>,
    num_hidden_layers: Option<usize>,
    num_attention_heads: Option<usize>,
    num_key_value_heads: Option<usize>,
    head_dim: Option<usize>,
    vocab_size: Option<usize>,
    max_position_embeddings: Option<usize>,
    rms_norm_eps: Option<f64>,
    hidden_act: Option<String>,
    attention_bias: Option<bool>,
    mlp_bias: Option<bool>,
    is_causal: Option<bool>,
    attention_windows: Option<Vec<Option<usize>>>,
    sliding_window: Option<usize>,
    layer_types: Option<Vec<String>>,
    rope_theta: Option<serde_json::Value>,
    rope_parameters: Option<BTreeMap<String, serde_json::Value>>,
    quantization_config: Option<serde_json::Value>,
    num_labels: Option<usize>,
    id2label: Option<BTreeMap<String, String>>,
    pad_token_id: Option<u32>,
    decision: Option<serde_json::Value>,
    #[serde(flatten)]
    extra: BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum UltraBertModelKind {
    Text,
    Vision,
    Decision,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub(crate) struct UltraBertVisionConfig {
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) image_size: usize,
    pub(crate) patch_size: usize,
    pub(crate) pixel_shuffle_factor: usize,
    pub(crate) layer_norm_eps: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub(crate) enum UltraBertDecisionKind {
    Text,
    Vision,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub(crate) struct UltraBertDecisionConfig {
    pub(crate) projection: usize,
    pub(crate) kind: UltraBertDecisionKind,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct UltraBertConfig {
    pub(crate) kind: UltraBertModelKind,
    pub(crate) task: UltraBertTask,
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) vocab_size: usize,
    pub(crate) max_position_embeddings: usize,
    pub(crate) rms_norm_eps: f64,
    pub(crate) hidden_act: String,
    pub(crate) attention_windows: Vec<Option<usize>>,
    pub(crate) rope_theta: Vec<f64>,
    pub(crate) num_labels: Option<usize>,
    pub(crate) pad_token_id: u32,
    pub(crate) vision: Option<UltraBertVisionConfig>,
    pub(crate) decision: Option<UltraBertDecisionConfig>,
}

fn required_usize(value: Option<usize>, field: &'static str) -> Result<usize, UltraBertError> {
    value.ok_or(UltraBertError::MissingConfigField { field })
}

fn required_nonzero(value: Option<usize>, field: &'static str) -> Result<usize, UltraBertError> {
    let value = required_usize(value, field)?;
    if value == 0 {
        return Err(UltraBertError::ZeroConfigField { field });
    }
    if value > MAX_HIDDEN.max(MAX_INTERMEDIATE) && field != "num_hidden_layers" {
        return Err(UltraBertError::ConfigFieldTooLarge {
            field,
            value,
            max: MAX_HIDDEN.max(MAX_INTERMEDIATE),
        });
    }
    Ok(value)
}

fn infer_task(architectures: &[String]) -> Result<UltraBertTask, UltraBertError> {
    match architectures {
        [] => Err(UltraBertError::MissingArchitectures),
        [architecture] => match architecture.as_str() {
            "UltrabertModel" => Ok(UltraBertTask::Embedding),
            "UltrabertForSequenceClassification" => Ok(UltraBertTask::SequenceClassification),
            "UltrabertForTokenClassification" => Ok(UltraBertTask::TokenClassification),
            other => Err(UltraBertError::UnsupportedArchitecture {
                architecture: other.to_string(),
            }),
        },
        _ => Err(UltraBertError::AmbiguousArchitectures {
            count: architectures.len(),
        }),
    }
}

fn json_usize(value: &serde_json::Value) -> Option<usize> {
    value.as_u64().and_then(|value| usize::try_from(value).ok())
}

fn json_f64(value: &serde_json::Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_str().and_then(|text| text.parse().ok()))
}

fn layer_rope_base(
    rope_parameters: Option<&BTreeMap<String, serde_json::Value>>,
    layer_type: Option<&str>,
    default: f64,
) -> f64 {
    let Some(parameters) = rope_parameters else {
        return default;
    };
    parameters
        .get(layer_type.unwrap_or("full_attention"))
        .or_else(|| parameters.get("full_attention"))
        .and_then(|value| value.get("rope_theta"))
        .and_then(json_f64)
        .unwrap_or(default)
}

fn parse_rope_bases(
    raw: &RawUltraBertConfig,
    layers: usize,
    default: f64,
) -> Result<Vec<f64>, UltraBertError> {
    if let Some(value) = &raw.rope_theta {
        if let Some(array) = value.as_array() {
            let bases = array.iter().filter_map(json_f64).collect::<Vec<_>>();
            if bases.len() != array.len() || bases.len() != layers {
                return Err(UltraBertError::LayerListLength {
                    field: "rope_theta",
                    layers,
                    got: bases.len(),
                });
            }
            return Ok(bases);
        }
        if let Some(base) = json_f64(value) {
            return Ok(vec![base; layers]);
        }
    }
    if raw.rope_parameters.is_some() || raw.layer_types.is_some() {
        let types = raw.layer_types.as_deref();
        if let Some(types) = types {
            if types.len() != layers {
                return Err(UltraBertError::LayerListLength {
                    field: "layer_types",
                    layers,
                    got: types.len(),
                });
            }
        }
        return Ok((0..layers)
            .map(|layer| {
                layer_rope_base(
                    raw.rope_parameters.as_ref(),
                    types.map(|t| t[layer].as_str()),
                    default,
                )
            })
            .collect());
    }
    Err(UltraBertError::MissingConfigField {
        field: "rope_theta",
    })
}

fn parse_windows(
    raw: &RawUltraBertConfig,
    layers: usize,
) -> Result<Vec<Option<usize>>, UltraBertError> {
    if let Some(windows) = &raw.attention_windows {
        if windows.len() != layers {
            return Err(UltraBertError::LayerListLength {
                field: "attention_windows",
                layers,
                got: windows.len(),
            });
        }
        return Ok(windows.clone());
    }
    let Some(types) = &raw.layer_types else {
        return Err(UltraBertError::MissingConfigField {
            field: "attention_windows",
        });
    };
    if types.len() != layers {
        return Err(UltraBertError::LayerListLength {
            field: "layer_types",
            layers,
            got: types.len(),
        });
    }
    let mut windows = Vec::with_capacity(layers);
    for (layer, kind) in types.iter().enumerate() {
        match kind.as_str() {
            "full_attention" => windows.push(None),
            "sliding_attention" => windows.push(Some(raw.sliding_window.ok_or(
                UltraBertError::MissingConfigField {
                    field: "sliding_window",
                },
            )?)),
            other => {
                return Err(UltraBertError::UnsupportedConfigModifier {
                    key: format!("layer_types.{layer}={other}"),
                })
            }
        }
    }
    Ok(windows)
}

fn parse_vision(value: &serde_json::Value) -> Result<UltraBertVisionConfig, UltraBertError> {
    let hidden_size = value.get("hidden_size").and_then(json_usize).ok_or(
        UltraBertError::MissingConfigField {
            field: "vision_config.hidden_size",
        },
    )?;
    let intermediate_size = value.get("intermediate_size").and_then(json_usize).ok_or(
        UltraBertError::MissingConfigField {
            field: "vision_config.intermediate_size",
        },
    )?;
    let num_hidden_layers = value.get("num_hidden_layers").and_then(json_usize).ok_or(
        UltraBertError::MissingConfigField {
            field: "vision_config.num_hidden_layers",
        },
    )?;
    let num_attention_heads = value
        .get("num_attention_heads")
        .and_then(json_usize)
        .ok_or(UltraBertError::MissingConfigField {
            field: "vision_config.num_attention_heads",
        })?;
    let image_size =
        value
            .get("image_size")
            .and_then(json_usize)
            .ok_or(UltraBertError::MissingConfigField {
                field: "vision_config.image_size",
            })?;
    let patch_size =
        value
            .get("patch_size")
            .and_then(json_usize)
            .ok_or(UltraBertError::MissingConfigField {
                field: "vision_config.patch_size",
            })?;
    let layer_norm_eps = value
        .get("layer_norm_eps")
        .and_then(json_f64)
        .unwrap_or(1e-6);
    if hidden_size != 768
        || intermediate_size != 3072
        || num_hidden_layers != 12
        || num_attention_heads != 12
        || image_size != 512
        || patch_size != 16
    {
        return Err(UltraBertError::UnsupportedConfigModifier {
            key: "vision_config geometry".into(),
        });
    }
    Ok(UltraBertVisionConfig {
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        image_size,
        patch_size,
        pixel_shuffle_factor: 4,
        layer_norm_eps,
    })
}

fn decision_projection(value: &serde_json::Value) -> usize {
    value
        .get("decision")
        .and_then(|decision| decision.get("proj_dim"))
        .and_then(json_usize)
        .unwrap_or(256)
}

impl UltraBertConfig {
    #[allow(dead_code)]
    pub(crate) fn load(path: &Path) -> Result<Self, UltraBertError> {
        Self::load_with_task(path, None)
    }

    #[cfg(test)]
    fn from_raw(
        raw: RawUltraBertConfig,
        task_override: Option<UltraBertTask>,
    ) -> Result<Self, UltraBertError> {
        let value = serde_json::to_value(raw).map_err(|error| UltraBertError::Checkpoint {
            message: error.to_string(),
        })?;
        Self::load_value(value, Path::new(""), false, task_override)
    }

    pub(crate) fn load_with_task(
        path: &Path,
        task_override: Option<UltraBertTask>,
    ) -> Result<Self, UltraBertError> {
        let value: serde_json::Value =
            read_json(&path.join("config.json")).map_err(UltraBertError::from)?;
        let decision_present = path.join("vela3_config.json").is_file()
            || path.join("vela3v_config.json").is_file()
            || value.get("decision").is_some_and(|value| !value.is_null());
        Self::load_value(value, path, decision_present, task_override)
    }

    fn load_value(
        value: serde_json::Value,
        root: &Path,
        decision_present: bool,
        task_override: Option<UltraBertTask>,
    ) -> Result<Self, UltraBertError> {
        let model_type = value
            .get("model_type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        let vision = if model_type == "ultravbert" {
            let vision = value
                .get("vision_config")
                .ok_or(UltraBertError::MissingConfigField {
                    field: "vision_config",
                })
                .and_then(parse_vision)?;
            if value
                .get("connector_norm")
                .and_then(serde_json::Value::as_bool)
                != Some(true)
            {
                return Err(UltraBertError::UnsupportedConfigModifier {
                    key: "connector_norm".into(),
                });
            }
            if value.get("pixel_shuffle_factor").and_then(json_usize) != Some(4) {
                return Err(UltraBertError::UnsupportedConfigModifier {
                    key: "pixel_shuffle_factor".into(),
                });
            }
            Some(vision)
        } else if model_type == "ultrabert" {
            None
        } else {
            return Err(UltraBertError::UnsupportedModelType {
                model_type: model_type.to_string(),
            });
        };
        if model_type == "ultrabert"
            && value
                .get("vision_config")
                .is_some_and(|value| !value.is_null())
        {
            return Err(UltraBertError::UnsupportedConfigModifier {
                key: "vision_config".into(),
            });
        }
        let source = if model_type == "ultravbert" {
            value
                .get("text_config")
                .cloned()
                .ok_or(UltraBertError::MissingConfigField {
                    field: "text_config",
                })?
        } else {
            value.clone()
        };
        let mut flattened = source;
        let object = flattened
            .as_object_mut()
            .ok_or(UltraBertError::MissingConfigField { field: "config" })?;
        object.insert(
            "model_type".into(),
            serde_json::Value::String("ultrabert".into()),
        );
        for key in [
            "rope_theta",
            "rope_parameters",
            "layer_types",
            "sliding_window",
            "attention_windows",
            "architectures",
        ] {
            if !object.contains_key(key) {
                if let Some(value) = value.get(key) {
                    object.insert(key.into(), value.clone());
                }
            }
        }
        let mut raw: RawUltraBertConfig =
            serde_json::from_value(flattened).map_err(|error| UltraBertError::Checkpoint {
                message: error.to_string(),
            })?;
        if model_type == "ultravbert" {
            raw.architectures = vec!["UltrabertModel".into()];
        }
        if raw
            .quantization_config
            .as_ref()
            .is_some_and(|value| !value.is_null())
        {
            return Err(UltraBertError::QuantizedCheckpoint);
        }
        for (key, value) in &raw.extra {
            let lower = key.to_ascii_lowercase();
            let unsupported = matches!(
                lower.as_str(),
                "rope_scaling"
                    | "partial_rotary_factor"
                    | "moe_intermediate_size"
                    | "vision_config"
                    | "text_config"
            ) || lower.contains("expert")
                || lower.contains("quantiz")
                || lower.starts_with("linear_")
                || lower.starts_with("image_")
                || lower.starts_with("audio_")
                || lower.starts_with("video_");
            if unsupported && !value.is_null() {
                return Err(UltraBertError::UnsupportedConfigModifier { key: key.clone() });
            }
        }
        let inferred = if raw.architectures.is_empty() && decision_present {
            UltraBertTask::Decision
        } else {
            infer_task(&raw.architectures)?
        };
        let task = match task_override {
            Some(UltraBertTask::Decision) if decision_present => UltraBertTask::Decision,
            Some(requested) if requested != inferred => {
                return Err(UltraBertError::TaskOverrideMismatch {
                    inferred,
                    requested,
                })
            }
            Some(requested) => requested,
            None => inferred,
        };
        if task == UltraBertTask::Decision && !decision_present {
            return Err(UltraBertError::MissingConfigField {
                field: "vela3_config.json",
            });
        }
        let hidden_size = required_nonzero(raw.hidden_size, "hidden_size")?;
        let intermediate_size = required_nonzero(raw.intermediate_size, "intermediate_size")?;
        let num_hidden_layers = required_usize(raw.num_hidden_layers, "num_hidden_layers")?;
        if num_hidden_layers == 0 {
            return Err(UltraBertError::ZeroConfigField {
                field: "num_hidden_layers",
            });
        }
        if num_hidden_layers > MAX_LAYERS {
            return Err(UltraBertError::ConfigFieldTooLarge {
                field: "num_hidden_layers",
                value: num_hidden_layers,
                max: MAX_LAYERS,
            });
        }
        let num_attention_heads = required_nonzero(raw.num_attention_heads, "num_attention_heads")?;
        let num_key_value_heads = raw.num_key_value_heads.unwrap_or(num_attention_heads);
        if num_key_value_heads == 0 {
            return Err(UltraBertError::ZeroConfigField {
                field: "num_key_value_heads",
            });
        }
        if num_key_value_heads > num_attention_heads
            || !num_attention_heads.is_multiple_of(num_key_value_heads)
        {
            return Err(UltraBertError::UnsupportedKvHeads {
                heads: num_attention_heads,
                kv_heads: num_key_value_heads,
            });
        }
        let vocab_size = required_nonzero(raw.vocab_size, "vocab_size")?;
        let max_position_embeddings = raw.max_position_embeddings.unwrap_or(DEFAULT_MAX_POSITIONS);
        if max_position_embeddings == 0 {
            return Err(UltraBertError::ZeroConfigField {
                field: "max_position_embeddings",
            });
        }
        if max_position_embeddings > MAX_POSITIONS_LIMIT {
            return Err(UltraBertError::ConfigFieldTooLarge {
                field: "max_position_embeddings",
                value: max_position_embeddings,
                max: MAX_POSITIONS_LIMIT,
            });
        }
        let head_dim = raw.head_dim.unwrap_or_else(|| {
            if hidden_size.is_multiple_of(num_attention_heads) {
                hidden_size / num_attention_heads
            } else {
                0
            }
        });
        if head_dim == 0 || !head_dim.is_multiple_of(2) {
            return Err(UltraBertError::HeadDimMismatch {
                head_dim,
                expected: hidden_size / num_attention_heads.max(1),
            });
        }
        let expected_head_dim = hidden_size / num_attention_heads.max(1);
        if head_dim != expected_head_dim {
            return Err(UltraBertError::HeadDimMismatch {
                head_dim,
                expected: expected_head_dim,
            });
        }
        if num_attention_heads
            .checked_mul(head_dim)
            .is_none_or(|width| width != hidden_size)
        {
            return Err(UltraBertError::HiddenHeadsMismatch {
                hidden: hidden_size,
                heads: num_attention_heads,
            });
        }
        let eps = raw.rms_norm_eps.unwrap_or(1e-6);
        if !eps.is_finite() || eps <= 0.0 || !(eps as f32).is_finite() || eps as f32 <= 0.0 {
            return Err(UltraBertError::InvalidEps { eps });
        }
        let hidden_act = raw.hidden_act.clone().unwrap_or_else(|| "silu".to_string());
        if hidden_act != "silu" {
            return Err(UltraBertError::UnsupportedActivation {
                activation: hidden_act,
            });
        }
        if raw.attention_bias.unwrap_or(false) {
            return Err(UltraBertError::BiasNotAdmitted {
                field: "attention_bias",
            });
        }
        if raw.mlp_bias.unwrap_or(false) {
            return Err(UltraBertError::BiasNotAdmitted { field: "mlp_bias" });
        }
        if raw.is_causal.unwrap_or(false) {
            return Err(UltraBertError::CausalNotAdmitted);
        }
        let attention_windows = parse_windows(&raw, num_hidden_layers)?;
        for (layer, window) in attention_windows.iter().enumerate() {
            if window.is_some_and(|window| window == 0 || window > i32::MAX as usize) {
                return Err(UltraBertError::InvalidAttentionWindow { layer });
            }
        }
        let rope_theta = parse_rope_bases(&raw, num_hidden_layers, 10_000.0)?;
        for (layer, base) in rope_theta.iter().enumerate() {
            if !base.is_finite()
                || *base <= 1.0
                || !(*base as f32).is_finite()
                || *base as f32 <= 1.0
            {
                return Err(UltraBertError::InvalidRopeBase { layer });
            }
        }
        let num_labels = if task.requires_classifier() {
            let labels = raw
                .num_labels
                .or_else(|| raw.id2label.as_ref().map(BTreeMap::len))
                .unwrap_or(DEFAULT_NUM_LABELS);
            if labels == 0 {
                return Err(UltraBertError::ZeroConfigField {
                    field: "num_labels",
                });
            }
            Some(labels)
        } else {
            None
        };
        let decision = if task == UltraBertTask::Decision {
            let kind = if vision.is_some() {
                UltraBertDecisionKind::Vision
            } else {
                UltraBertDecisionKind::Text
            };
            let projection = if kind == UltraBertDecisionKind::Vision {
                0
            } else {
                let vela = root
                    .join("vela3_config.json")
                    .is_file()
                    .then(|| root.join("vela3_config.json"))
                    .or_else(|| {
                        root.join("vela3v_config.json")
                            .is_file()
                            .then(|| root.join("vela3v_config.json"))
                    });
                let projection = if let Some(decision) = raw.decision.as_ref() {
                    decision_projection(&serde_json::json!({"decision": decision}))
                } else if let Some(path) = vela {
                    let value: serde_json::Value =
                        read_json(&path).map_err(UltraBertError::from)?;
                    decision_projection(&value)
                } else {
                    decision_projection(&serde_json::Value::Null)
                };
                if projection == 0 {
                    return Err(UltraBertError::ZeroConfigField {
                        field: "decision.proj_dim",
                    });
                }
                projection
            };
            Some(UltraBertDecisionConfig { projection, kind })
        } else {
            None
        };
        let kind = if vision.is_some() {
            UltraBertModelKind::Vision
        } else if decision.is_some() {
            UltraBertModelKind::Decision
        } else {
            UltraBertModelKind::Text
        };
        Ok(Self {
            kind,
            task,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            vocab_size,
            max_position_embeddings,
            rms_norm_eps: eps,
            hidden_act,
            attention_windows,
            rope_theta,
            num_labels,
            pad_token_id: raw.pad_token_id.unwrap_or(DEFAULT_PAD_TOKEN_ID),
            vision,
            decision,
        })
    }

    fn q_width(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    fn kv_width(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    pub(crate) fn rope_bases(&self) -> (Vec<f64>, Vec<usize>) {
        let mut bases = Vec::new();
        let mut indices = Vec::with_capacity(self.rope_theta.len());
        for base in &self.rope_theta {
            let index = bases
                .iter()
                .position(|candidate: &f64| candidate == base)
                .unwrap_or_else(|| {
                    bases.push(*base);
                    bases.len() - 1
                });
            indices.push(index);
        }
        (bases, indices)
    }
}

#[derive(Debug, Deserialize)]
struct RawModule {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    path: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawPoolingConfig {
    #[serde(default)]
    pooling_mode: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawSentenceBertConfig {
    #[serde(default)]
    prompts: Option<serde_json::Value>,
}

#[derive(Clone, Debug, Serialize)]
pub struct UltraBertLayerPlan {
    pub index: usize,
    pub window: Option<usize>,
    pub rope_base: f64,
    pub rope_index: usize,
    pub attention_scope: String,
    pub fa4_profile: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct UltraBertFa4Requirement {
    pub scope: String,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub causal: bool,
    pub window: String,
    pub matched_profile: Option<String>,
    pub present: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct UltraBertGeometryReport {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub hidden_act: String,
    pub local_attention_layers: usize,
    pub global_attention_layers: usize,
    pub pad_token_id: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct UltraBertReport {
    pub status: &'static str,
    pub qualification_marker: &'static str,
    pub source_path: String,
    pub task: UltraBertTask,
    pub runtime_dtype: String,
    pub geometry: UltraBertGeometryReport,
    pub pooling: UltraBertPooling,
    pub weight_prefix: String,
    pub tensor_count: usize,
    pub stored_dtypes: Vec<String>,
    pub layers: Vec<UltraBertLayerPlan>,
    pub fa4_requirements: Vec<UltraBertFa4Requirement>,
    pub missing_fa4_profiles: Vec<String>,
    pub blockers: Vec<String>,
    pub production_factory_admitted: bool,
    pub executor_implemented: bool,
    pub execution_ready: bool,
    pub gpu_execution_ready: bool,
    pub root_native_comparison: bool,
    pub query_prompt: Option<String>,
    pub document_prompt: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct UltraBertForwardPlan {
    pub tokens: usize,
    pub batch: usize,
    pub max_length: usize,
    pub positions: Vec<u32>,
    pub local_layers: usize,
    pub global_layers: usize,
    pub fa4_requirements: Vec<UltraBertFa4Requirement>,
    pub execution_ready: bool,
    pub blockers: Vec<String>,
}

#[derive(Clone, Debug)]
pub(crate) struct UltraBertLayerWeights {
    pub(crate) input_norm: WeightSource,
    pub(crate) post_attention_norm: WeightSource,
    pub(crate) q: WeightSource,
    pub(crate) k: WeightSource,
    pub(crate) v: WeightSource,
    pub(crate) o: WeightSource,
    pub(crate) gate: WeightSource,
    pub(crate) up: WeightSource,
    pub(crate) down: WeightSource,
}

#[derive(Clone, Debug)]
pub(crate) struct UltraBertLinearSource {
    pub(crate) weight: WeightSource,
    pub(crate) bias: Option<WeightSource>,
}

#[derive(Clone, Debug)]
pub(crate) struct UltraBertVisionLayerWeights {
    pub(crate) norm1: UltraBertLinearSource,
    pub(crate) q: UltraBertLinearSource,
    pub(crate) k: UltraBertLinearSource,
    pub(crate) v: UltraBertLinearSource,
    pub(crate) o: UltraBertLinearSource,
    pub(crate) norm2: UltraBertLinearSource,
    pub(crate) fc1: UltraBertLinearSource,
    pub(crate) fc2: UltraBertLinearSource,
}

#[derive(Clone, Debug)]
pub(crate) struct UltraBertVisionWeights {
    pub(crate) patch: UltraBertLinearSource,
    pub(crate) position: WeightSource,
    pub(crate) layers: Vec<UltraBertVisionLayerWeights>,
    pub(crate) post_norm: UltraBertLinearSource,
    pub(crate) connector: UltraBertLinearSource,
    pub(crate) connector_norm: WeightSource,
}

#[derive(Clone, Debug)]
pub(crate) struct UltraBertTextDecisionWeights {
    pub(crate) norm: UltraBertLinearSource,
    pub(crate) option: UltraBertLinearSource,
    pub(crate) question: UltraBertLinearSource,
    pub(crate) state: UltraBertLinearSource,
    pub(crate) first: UltraBertLinearSource,
    pub(crate) last: UltraBertLinearSource,
    pub(crate) log_tau: WeightSource,
}

#[derive(Clone, Debug)]
pub(crate) struct UltraBertVisionDecisionWeights {
    pub(crate) first: UltraBertLinearSource,
    pub(crate) norm: UltraBertLinearSource,
    pub(crate) last: UltraBertLinearSource,
}

#[derive(Clone, Debug)]
#[allow(clippy::large_enum_variant)]
pub(crate) enum UltraBertDecisionWeights {
    Text(UltraBertTextDecisionWeights),
    Vision(UltraBertVisionDecisionWeights),
}

#[derive(Clone, Debug)]
pub(crate) struct UltraBertWeights {
    pub(crate) embed: WeightSource,
    pub(crate) norm: WeightSource,
    pub(crate) layers: Vec<UltraBertLayerWeights>,
    pub(crate) classifier_weight: Option<WeightSource>,
    pub(crate) classifier_bias: Option<WeightSource>,
    pub(crate) vision: Option<UltraBertVisionWeights>,
    pub(crate) decision: Option<UltraBertDecisionWeights>,
}

/// Checkpoint validator shared by the serving factory and qualification tools.
/// Its CPU report does not establish whole-model numerical qualification.
#[derive(Debug)]
pub struct UltraBertQualification {
    #[allow(dead_code)]
    config: UltraBertConfig,
    #[allow(dead_code)]
    weights: UltraBertWeights,
    report: UltraBertReport,
}

impl UltraBertQualification {
    pub fn open(
        path: &Path,
        runtime_dtype: &str,
        task_override: Option<UltraBertTask>,
    ) -> Result<Self, UltraBertError> {
        if runtime_dtype != "bfloat16" {
            return Err(UltraBertError::UnsupportedRuntimeDtype {
                dtype: runtime_dtype.to_string(),
            });
        }
        let task = UltraBertConfig::load_with_task(path, task_override)?;
        let (pooling, query_prompt, document_prompt) = load_task_modules(path, task.task)?;
        let checkpoint = if task.vision.is_some() {
            let extras: &[&str] = if task.decision.is_some() {
                &["backbone_h2.safetensors", "decision_adapter.safetensors"]
            } else {
                &["backbone_h2.safetensors"]
            };
            Checkpoint::open_with_extra(path, extras).map_err(UltraBertError::from)?
        } else {
            Checkpoint::open(path).map_err(UltraBertError::from)?
        };
        let prefix = select_root(&checkpoint)?;
        let weights = load_weights(&checkpoint, &task, prefix)?;
        let (_, rope_indices) = task.rope_bases();
        let mut layers = Vec::with_capacity(task.num_hidden_layers);
        let mut requirements: Vec<UltraBertFa4Requirement> = Vec::new();
        for (layer, ((window, rope_base), rope_index)) in task
            .attention_windows
            .iter()
            .copied()
            .zip(task.rope_theta.iter().copied())
            .zip(rope_indices.iter().copied())
            .enumerate()
        {
            let spec_window = window.map_or(Window::Global, |window| Window::Local {
                left: i32::try_from(window).unwrap_or(i32::MAX),
                right: i32::try_from(window).unwrap_or(i32::MAX),
            });
            let spec = AttentionSpec {
                heads: task.num_attention_heads,
                kv_heads: task.num_key_value_heads,
                head_dim: task.head_dim,
                causal: false,
                window: spec_window,
                dtype: Fa4DType::Bf16,
                scale: Scale::RsqrtHeadDim,
            };
            let matched = matching_profile_name(&spec);
            let scope = if window.is_some() { "local" } else { "global" };
            let window_name = window.map_or_else(
                || "global".to_string(),
                |window| format!("local{window}/{window}"),
            );
            let requirement = UltraBertFa4Requirement {
                scope: scope.to_string(),
                heads: task.num_attention_heads,
                kv_heads: task.num_key_value_heads,
                head_dim: task.head_dim,
                causal: false,
                window: window_name.clone(),
                matched_profile: matched.map(str::to_string),
                present: matched.is_some(),
            };
            if !requirements.iter().any(|existing| {
                existing.scope == requirement.scope
                    && existing.window == requirement.window
                    && existing.matched_profile == requirement.matched_profile
            }) {
                requirements.push(requirement.clone());
            }
            layers.push(UltraBertLayerPlan {
                index: layer,
                window,
                rope_base,
                rope_index,
                attention_scope: scope.to_string(),
                fa4_profile: matched.map(str::to_string),
            });
        }
        let missing_fa4_profiles = requirements
            .iter()
            .filter(|requirement| !requirement.present)
            .map(|requirement| {
                format!(
                    "{} {} (heads={} kv_heads={} d={} window={})",
                    requirement.scope,
                    requirement
                        .matched_profile
                        .as_deref()
                        .unwrap_or(requirement.window.as_str()),
                    requirement.heads,
                    requirement.kv_heads,
                    requirement.head_dim,
                    requirement.window
                )
            })
            .collect::<Vec<_>>();
        let local_attention_layers = task
            .attention_windows
            .iter()
            .filter(|window| window.is_some())
            .count();
        let global_attention_layers = task.num_hidden_layers - local_attention_layers;
        let mut stored_dtypes = BTreeSet::new();
        for source in weights.all_sources() {
            stored_dtypes.insert(format_stored_dtype(source.dtype()));
        }
        let mut tensor_count =
            2 + task.num_hidden_layers * 9 + usize::from(task.task.requires_classifier()) * 2;
        if let Some(vision) = &task.vision {
            tensor_count += 3 + vision.num_hidden_layers * 16 + 4;
        }
        if let Some(decision) = &task.decision {
            tensor_count += match decision.kind {
                UltraBertDecisionKind::Text => 13,
                UltraBertDecisionKind::Vision => 6,
            };
        }
        let mut blockers = Vec::new();
        for requirement in &requirements {
            if !requirement.present {
                blockers.push(format!(
                    "missing FA4 profile: {} heads={} kv_heads={} d={} causal=false window={}",
                    requirement.scope,
                    requirement.heads,
                    requirement.kv_heads,
                    requirement.head_dim,
                    requirement.window
                ));
            }
        }
        blockers.push("Root Native comparison has not been executed".into());
        blockers.push(
            "GPU execution and Root Native comparison were not performed in this CPU-only pass"
                .into(),
        );
        let report = UltraBertReport {
            status: "qualification_only",
            qualification_marker: QUALIFICATION_MARKER,
            source_path: path.display().to_string(),
            task: task.task,
            runtime_dtype: runtime_dtype.to_string(),
            geometry: UltraBertGeometryReport {
                hidden_size: task.hidden_size,
                intermediate_size: task.intermediate_size,
                num_hidden_layers: task.num_hidden_layers,
                num_attention_heads: task.num_attention_heads,
                num_key_value_heads: task.num_key_value_heads,
                head_dim: task.head_dim,
                vocab_size: task.vocab_size,
                max_position_embeddings: task.max_position_embeddings,
                rms_norm_eps: task.rms_norm_eps,
                hidden_act: task.hidden_act.clone(),
                local_attention_layers,
                global_attention_layers,
                pad_token_id: task.pad_token_id,
            },
            pooling,
            weight_prefix: prefix.to_string(),
            tensor_count,
            stored_dtypes: stored_dtypes.into_iter().collect(),
            layers,
            fa4_requirements: requirements,
            missing_fa4_profiles,
            blockers,
            production_factory_admitted: false,
            executor_implemented: true,
            execution_ready: false,
            gpu_execution_ready: false,
            root_native_comparison: false,
            query_prompt,
            document_prompt,
        };
        Ok(Self {
            config: task,
            weights,
            report,
        })
    }

    pub fn report(&self) -> &UltraBertReport {
        &self.report
    }

    #[allow(dead_code)]
    pub(crate) fn config(&self) -> &UltraBertConfig {
        &self.config
    }

    #[allow(dead_code)]
    pub(crate) fn weights(&self) -> &UltraBertWeights {
        &self.weights
    }

    /// Validate a packed batch without adding padding or a graph. Offsets are
    /// exact sequence lengths and must cover `input_ids` exactly.
    pub fn plan_forward(
        &self,
        input_ids: &[u32],
        cumulative_offsets: &[u32],
    ) -> Result<UltraBertForwardPlan, UltraBertError> {
        if input_ids.is_empty() {
            return Err(UltraBertError::BatchEmpty);
        }
        if cumulative_offsets.len() < 2 || cumulative_offsets[0] != 0 {
            return Err(UltraBertError::BatchOffsets {
                message: "offsets must start at zero and contain at least one sequence".into(),
            });
        }
        if cumulative_offsets.windows(2).any(|pair| pair[1] <= pair[0]) {
            return Err(UltraBertError::BatchOffsets {
                message: "each sequence must be nonempty and offsets strictly increasing".into(),
            });
        }
        let end = *cumulative_offsets.last().unwrap_or(&0);
        if end as usize != input_ids.len() {
            return Err(UltraBertError::BatchOffsets {
                message: format!(
                    "final offset {end} does not equal token count {}",
                    input_ids.len()
                ),
            });
        }
        let mut max_length = 0usize;
        for pair in cumulative_offsets.windows(2) {
            max_length = max_length.max((pair[1] - pair[0]) as usize);
        }
        if max_length > self.config.max_position_embeddings {
            return Err(UltraBertError::SequenceTooLong {
                length: max_length,
                max: self.config.max_position_embeddings,
            });
        }
        for &token in input_ids {
            if token as usize >= self.config.vocab_size {
                return Err(UltraBertError::TokenOutOfRange {
                    token,
                    vocab: self.config.vocab_size,
                });
            }
        }
        let positions = (0..input_ids.len())
            .map(|position| u32::try_from(position).unwrap_or(u32::MAX))
            .collect();
        Ok(UltraBertForwardPlan {
            tokens: input_ids.len(),
            batch: cumulative_offsets.len() - 1,
            max_length,
            positions,
            local_layers: self.report.geometry.local_attention_layers,
            global_layers: self.report.geometry.global_attention_layers,
            fa4_requirements: self.report.fa4_requirements.clone(),
            execution_ready: false,
            blockers: self.report.blockers.clone(),
        })
    }

    /// Typed GPU preparation requirements. This is CPU-only and returns the
    /// exact FA4 profiles, RoPE bases, kernel exports and classifier geometry
    /// that `prepare_gpu` will consume.
    pub fn gpu_requirements(&self) -> UltraBertGpuRequirements {
        crate::ultrabert_executor::requirements(&self.config, &self.report.fa4_requirements)
    }

    /// CPU-only typed readiness check. At the pinned source this returns
    /// `MissingFa4Profile` for the 128/128 local window before any CUDA object
    /// is created.
    pub fn require_gpu_ready(&self) -> Result<(), UltraBertError> {
        crate::ultrabert_executor::validate_requirements(&self.gpu_requirements())
    }

    /// Build the real GPU executor. This creates a CUDA context-bound provider,
    /// uploads weights, allocates immutable RoPE tables, loads both FA4
    /// instances and prepares the classifier plan. It does not execute a model.
    pub fn prepare_gpu(
        &self,
        context: std::sync::Arc<cuda_core::Context>,
        stream: std::sync::Arc<cuda_core::Stream>,
        fa4_root: &std::path::Path,
    ) -> Result<crate::ultrabert_executor::UltraBertExecutor, UltraBertError> {
        let lt = crate::lt::LtProvider::new(context.clone(), stream.clone()).map_err(|error| {
            UltraBertError::ExecutionSetup {
                message: error.to_string(),
            }
        })?;
        let blas = crate::cublas::Blas::new(context.clone(), stream.clone()).map_err(|error| {
            UltraBertError::ExecutionSetup {
                message: error.to_string(),
            }
        })?;
        self.prepare_gpu_with(context, stream, fa4_root, lt, blas)
    }

    /// Build the real GPU executor with provider handles created by the caller
    /// so the production Engine can share the same ordered stream and context.
    pub(crate) fn prepare_gpu_with(
        &self,
        context: std::sync::Arc<cuda_core::Context>,
        stream: std::sync::Arc<cuda_core::Stream>,
        fa4_root: &std::path::Path,
        lt: crate::lt::LtProvider,
        blas: crate::cublas::Blas,
    ) -> Result<crate::ultrabert_executor::UltraBertExecutor, UltraBertError> {
        let requirements = self.gpu_requirements();
        crate::ultrabert_executor::validate_requirements(&requirements)?;
        crate::ultrabert_executor::UltraBertExecutor::new(
            &self.config,
            &self.weights,
            &requirements,
            context,
            stream,
            lt,
            blas,
            fa4_root,
        )
    }
}

fn format_stored_dtype(dtype: StoredDtype) -> String {
    match dtype {
        StoredDtype::F32 => "F32".into(),
        StoredDtype::F16 => "F16".into(),
        StoredDtype::BF16 => "BF16".into(),
        other => format!("{other:?}"),
    }
}

fn select_root(checkpoint: &Checkpoint) -> Result<&'static str, UltraBertError> {
    let suffix = "embed_tokens.weight";
    let anchors = checkpoint
        .names()
        .filter(|name| name.ends_with(suffix))
        .collect::<Vec<_>>();
    match anchors.as_slice() {
        [] => Err(UltraBertError::MissingTensor {
            name: suffix.into(),
        }),
        [anchor] => {
            let name = anchor.as_str();
            if name == suffix {
                Ok("")
            } else if name == format!("model.{suffix}") {
                Ok("model.")
            } else if name == format!("ultrabert.{suffix}") {
                Ok("ultrabert.")
            } else if name == format!("encoder.{suffix}") {
                Ok("encoder.")
            } else if name == format!("text_model.{suffix}") {
                Ok("text_model.")
            } else if name == format!("ultrabert.text_model.{suffix}") {
                Ok("ultrabert.text_model.")
            } else {
                Err(UltraBertError::UnsupportedNamespace {
                    name: name.to_string(),
                })
            }
        }
        _ => Err(UltraBertError::AmbiguousRoot {
            count: anchors.len(),
        }),
    }
}

fn expected_shapes(config: &UltraBertConfig, prefix: &str) -> BTreeMap<String, Vec<usize>> {
    let h = config.hidden_size;
    let q = config.q_width();
    let kv = config.kv_width();
    let i = config.intermediate_size;
    let mut expected = BTreeMap::new();
    expected.insert(
        format!("{prefix}embed_tokens.weight"),
        vec![config.vocab_size, h],
    );
    expected.insert(format!("{prefix}norm.weight"), vec![h]);
    for layer in 0..config.num_hidden_layers {
        let p = format!("{prefix}layers.{layer}");
        expected.insert(format!("{p}.input_layernorm.weight"), vec![h]);
        expected.insert(format!("{p}.post_attention_layernorm.weight"), vec![h]);
        expected.insert(format!("{p}.self_attn.q_proj.weight"), vec![q, h]);
        expected.insert(format!("{p}.self_attn.k_proj.weight"), vec![kv, h]);
        expected.insert(format!("{p}.self_attn.v_proj.weight"), vec![kv, h]);
        expected.insert(format!("{p}.self_attn.o_proj.weight"), vec![h, q]);
        expected.insert(format!("{p}.mlp.gate_proj.weight"), vec![i, h]);
        expected.insert(format!("{p}.mlp.up_proj.weight"), vec![i, h]);
        expected.insert(format!("{p}.mlp.down_proj.weight"), vec![h, i]);
    }
    if config.task.requires_classifier() {
        let labels = config.num_labels.unwrap_or(DEFAULT_NUM_LABELS);
        expected.insert("classifier.weight".into(), vec![labels, h]);
        expected.insert("classifier.bias".into(), vec![labels]);
    }
    if let Some(vision) = &config.vision {
        let width = vision.hidden_size;
        let inner = vision.intermediate_size;
        let patches = vision.image_size / vision.patch_size;
        let patch_width = patches * patches;
        expected.insert(
            "vision_model.embeddings.patch_embedding.weight".into(),
            vec![width, 3, vision.patch_size, vision.patch_size],
        );
        expected.insert(
            "vision_model.embeddings.patch_embedding.bias".into(),
            vec![width],
        );
        expected.insert(
            "vision_model.embeddings.position_embedding.weight".into(),
            vec![patch_width, width],
        );
        for layer in 0..vision.num_hidden_layers {
            let p = format!("vision_model.encoder.layers.{layer}");
            for norm in ["layer_norm1", "layer_norm2"] {
                expected.insert(format!("{p}.{norm}.weight"), vec![width]);
                expected.insert(format!("{p}.{norm}.bias"), vec![width]);
            }
            for name in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                expected.insert(format!("{p}.self_attn.{name}.weight"), vec![width, width]);
                expected.insert(format!("{p}.self_attn.{name}.bias"), vec![width]);
            }
            expected.insert(format!("{p}.mlp.fc1.weight"), vec![inner, width]);
            expected.insert(format!("{p}.mlp.fc1.bias"), vec![inner]);
            expected.insert(format!("{p}.mlp.fc2.weight"), vec![width, inner]);
            expected.insert(format!("{p}.mlp.fc2.bias"), vec![width]);
        }
        expected.insert("vision_model.post_layernorm.weight".into(), vec![width]);
        expected.insert("vision_model.post_layernorm.bias".into(), vec![width]);
        expected.insert(
            "connector.modality_projection.weight".into(),
            vec![
                width,
                width * vision.pixel_shuffle_factor * vision.pixel_shuffle_factor,
            ],
        );
        expected.insert("connector.norm.weight".into(), vec![width]);
    }
    if let Some(decision) = &config.decision {
        match decision.kind {
            UltraBertDecisionKind::Text => {
                let projection = decision.projection;
                for (name, shape) in [
                    ("norm", vec![h]),
                    ("w_o", vec![projection, h]),
                    ("w_q", vec![projection, h]),
                    ("w_p", vec![projection, h]),
                    ("cls_mlp.0", vec![2 * h, h]),
                    ("cls_mlp.3", vec![1, 2 * h]),
                ] {
                    expected.insert(format!("{name}.weight"), shape.clone());
                    expected.insert(format!("{name}.bias"), vec![shape[0]]);
                }
                expected.insert("log_tau".into(), vec![1]);
            }
            UltraBertDecisionKind::Vision => {
                for (name, shape) in [
                    ("head.net.0", vec![h, h]),
                    ("head.net.2", vec![h]),
                    ("head.net.3", vec![1, h]),
                ] {
                    expected.insert(format!("{name}.weight"), shape.clone());
                    expected.insert(format!("{name}.bias"), vec![shape[0]]);
                }
            }
        }
    }
    expected
}

fn load_weights(
    checkpoint: &Checkpoint,
    config: &UltraBertConfig,
    prefix: &str,
) -> Result<UltraBertWeights, UltraBertError> {
    let expected = expected_shapes(config, prefix);
    for name in checkpoint.names() {
        if !expected.contains_key(name) {
            return Err(UltraBertError::UnexpectedTensor { name: name.clone() });
        }
    }
    for (name, shape) in &expected {
        let Some(actual) = checkpoint.shape(name) else {
            return Err(UltraBertError::MissingTensor { name: name.clone() });
        };
        if actual != shape.as_slice() {
            return Err(UltraBertError::ShapeMismatch {
                name: name.clone(),
                expected: shape.clone(),
                actual: actual.to_vec(),
            });
        }
    }
    let fetch = |name: &str| -> Result<WeightSource, UltraBertError> {
        let shape = expected
            .get(name)
            .ok_or_else(|| UltraBertError::MissingTensor {
                name: name.to_string(),
            })?;
        let source = checkpoint
            .source(name, shape)
            .map_err(UltraBertError::from)?;
        match source.dtype() {
            StoredDtype::F32 | StoredDtype::F16 | StoredDtype::BF16 => Ok(source),
            other => Err(UltraBertError::UnsupportedStoredDtype {
                name: name.to_string(),
                dtype: format_stored_dtype(other),
            }),
        }
    };
    let linear = |name: &str| -> Result<UltraBertLinearSource, UltraBertError> {
        Ok(UltraBertLinearSource {
            weight: fetch(&format!("{name}.weight"))?,
            bias: Some(fetch(&format!("{name}.bias"))?),
        })
    };
    let h = config.hidden_size;
    let q = config.q_width();
    let kv = config.kv_width();
    let i = config.intermediate_size;
    let embed = fetch(&format!("{prefix}embed_tokens.weight"))?;
    let norm = fetch(&format!("{prefix}norm.weight"))?;
    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for layer in 0..config.num_hidden_layers {
        let p = format!("{prefix}layers.{layer}");
        layers.push(UltraBertLayerWeights {
            input_norm: fetch(&format!("{p}.input_layernorm.weight"))?,
            post_attention_norm: fetch(&format!("{p}.post_attention_layernorm.weight"))?,
            q: fetch(&format!("{p}.self_attn.q_proj.weight"))?,
            k: fetch(&format!("{p}.self_attn.k_proj.weight"))?,
            v: fetch(&format!("{p}.self_attn.v_proj.weight"))?,
            o: fetch(&format!("{p}.self_attn.o_proj.weight"))?,
            gate: fetch(&format!("{p}.mlp.gate_proj.weight"))?,
            up: fetch(&format!("{p}.mlp.up_proj.weight"))?,
            down: fetch(&format!("{p}.mlp.down_proj.weight"))?,
        });
        let _ = (h, q, kv, i);
    }
    let (classifier_weight, classifier_bias) = if config.task.requires_classifier() {
        (
            Some(fetch("classifier.weight")?),
            Some(fetch("classifier.bias")?),
        )
    } else {
        if checkpoint.contains("classifier.weight")
            || checkpoint.contains("classifier.bias")
            || checkpoint.contains(&format!("{prefix}classifier.weight"))
            || checkpoint.contains(&format!("{prefix}classifier.bias"))
        {
            return Err(UltraBertError::UnexpectedClassifier);
        }
        (None, None)
    };
    let vision = if let Some(vision) = &config.vision {
        let mut vision_layers = Vec::with_capacity(vision.num_hidden_layers);
        for layer in 0..vision.num_hidden_layers {
            let p = format!("vision_model.encoder.layers.{layer}");
            vision_layers.push(UltraBertVisionLayerWeights {
                norm1: UltraBertLinearSource {
                    weight: fetch(&format!("{p}.layer_norm1.weight"))?,
                    bias: Some(fetch(&format!("{p}.layer_norm1.bias"))?),
                },
                q: linear(&format!("{p}.self_attn.q_proj"))?,
                k: linear(&format!("{p}.self_attn.k_proj"))?,
                v: linear(&format!("{p}.self_attn.v_proj"))?,
                o: linear(&format!("{p}.self_attn.out_proj"))?,
                norm2: UltraBertLinearSource {
                    weight: fetch(&format!("{p}.layer_norm2.weight"))?,
                    bias: Some(fetch(&format!("{p}.layer_norm2.bias"))?),
                },
                fc1: linear(&format!("{p}.mlp.fc1"))?,
                fc2: linear(&format!("{p}.mlp.fc2"))?,
            });
        }
        Some(UltraBertVisionWeights {
            patch: UltraBertLinearSource {
                weight: fetch("vision_model.embeddings.patch_embedding.weight")?,
                bias: Some(fetch("vision_model.embeddings.patch_embedding.bias")?),
            },
            position: fetch("vision_model.embeddings.position_embedding.weight")?,
            layers: vision_layers,
            post_norm: UltraBertLinearSource {
                weight: fetch("vision_model.post_layernorm.weight")?,
                bias: Some(fetch("vision_model.post_layernorm.bias")?),
            },
            connector: UltraBertLinearSource {
                weight: fetch("connector.modality_projection.weight")?,
                bias: None,
            },
            connector_norm: fetch("connector.norm.weight")?,
        })
    } else {
        None
    };
    let decision = match config.decision.as_ref().map(|decision| decision.kind) {
        Some(kind) => {
            let require_f32 =
                |name: &str, source: WeightSource| -> Result<WeightSource, UltraBertError> {
                    if source.dtype() != StoredDtype::F32 {
                        return Err(UltraBertError::UnsupportedStoredDtype {
                            name: name.to_string(),
                            dtype: format_stored_dtype(source.dtype()),
                        });
                    }
                    Ok(source)
                };
            let decision_linear = |name: &str| -> Result<UltraBertLinearSource, UltraBertError> {
                Ok(UltraBertLinearSource {
                    weight: require_f32(
                        &format!("{name}.weight"),
                        fetch(&format!("{name}.weight"))?,
                    )?,
                    bias: Some(require_f32(
                        &format!("{name}.bias"),
                        fetch(&format!("{name}.bias"))?,
                    )?),
                })
            };
            Some(match kind {
                UltraBertDecisionKind::Text => {
                    UltraBertDecisionWeights::Text(UltraBertTextDecisionWeights {
                        norm: decision_linear("norm")?,
                        option: decision_linear("w_o")?,
                        question: decision_linear("w_q")?,
                        state: decision_linear("w_p")?,
                        first: decision_linear("cls_mlp.0")?,
                        last: decision_linear("cls_mlp.3")?,
                        log_tau: require_f32("log_tau", fetch("log_tau")?)?,
                    })
                }
                UltraBertDecisionKind::Vision => {
                    UltraBertDecisionWeights::Vision(UltraBertVisionDecisionWeights {
                        first: decision_linear("head.net.0")?,
                        norm: decision_linear("head.net.2")?,
                        last: decision_linear("head.net.3")?,
                    })
                }
            })
        }
        None => None,
    };
    Ok(UltraBertWeights {
        embed,
        norm,
        layers,
        classifier_weight,
        classifier_bias,
        vision,
        decision,
    })
}

impl UltraBertWeights {
    #[allow(dead_code)]
    fn all_sources(&self) -> Vec<&WeightSource> {
        let mut sources = vec![&self.embed, &self.norm];
        for layer in &self.layers {
            sources.extend([
                &layer.input_norm,
                &layer.post_attention_norm,
                &layer.q,
                &layer.k,
                &layer.v,
                &layer.o,
                &layer.gate,
                &layer.up,
                &layer.down,
            ]);
        }
        if let Some(weight) = &self.classifier_weight {
            sources.push(weight);
        }
        if let Some(bias) = &self.classifier_bias {
            sources.push(bias);
        }
        fn push_linear<'a>(sources: &mut Vec<&'a WeightSource>, linear: &'a UltraBertLinearSource) {
            sources.push(&linear.weight);
            if let Some(bias) = &linear.bias {
                sources.push(bias);
            }
        }
        if let Some(vision) = &self.vision {
            push_linear(&mut sources, &vision.patch);
            sources.push(&vision.position);
            for layer in &vision.layers {
                push_linear(&mut sources, &layer.norm1);
                push_linear(&mut sources, &layer.q);
                push_linear(&mut sources, &layer.k);
                push_linear(&mut sources, &layer.v);
                push_linear(&mut sources, &layer.o);
                push_linear(&mut sources, &layer.norm2);
                push_linear(&mut sources, &layer.fc1);
                push_linear(&mut sources, &layer.fc2);
            }
            push_linear(&mut sources, &vision.post_norm);
            push_linear(&mut sources, &vision.connector);
            sources.push(&vision.connector_norm);
        }
        if let Some(decision) = &self.decision {
            match decision {
                UltraBertDecisionWeights::Text(decision) => {
                    for linear in [
                        &decision.norm,
                        &decision.option,
                        &decision.question,
                        &decision.state,
                        &decision.first,
                        &decision.last,
                    ] {
                        sources.push(&linear.weight);
                        if let Some(bias) = &linear.bias {
                            sources.push(bias);
                        }
                    }
                    sources.push(&decision.log_tau);
                }
                UltraBertDecisionWeights::Vision(decision) => {
                    for linear in [&decision.first, &decision.norm, &decision.last] {
                        sources.push(&linear.weight);
                        if let Some(bias) = &linear.bias {
                            sources.push(bias);
                        }
                    }
                }
            }
        }
        sources
    }
}

fn load_task_modules(
    path: &Path,
    task: UltraBertTask,
) -> Result<(UltraBertPooling, Option<String>, Option<String>), UltraBertError> {
    let modules_path = path.join("modules.json");
    let mut pooling_path = None;
    if modules_path.is_file() {
        let modules: Vec<RawModule> = read_json(&modules_path).map_err(UltraBertError::from)?;
        for module in &modules {
            let kind = module.kind.as_str();
            if kind.ends_with("Transformer") {
                if module
                    .path
                    .as_deref()
                    .is_some_and(|module_path| !module_path.is_empty())
                {
                    return Err(UltraBertError::UnsupportedModule {
                        module: format!("{kind} at non-root path"),
                    });
                }
            } else if kind.ends_with("Pooling") {
                if pooling_path.is_some() {
                    return Err(UltraBertError::UnsupportedModule {
                        module: format!("duplicate {kind}"),
                    });
                }
                let relative = module.path.as_deref().unwrap_or("");
                let mut components = Path::new(relative).components();
                if !matches!(components.next(), Some(Component::Normal(_)))
                    || components.next().is_some()
                    || relative.contains('\\')
                {
                    return Err(UltraBertError::UnsupportedModule {
                        module: format!("{kind} path {relative:?}"),
                    });
                }
                pooling_path = Some(path.join(relative).join("config.json"));
            } else {
                // Normalize, Dense, custom heads and other modules are not
                // silently ignored: they are trained/onward pipeline behavior.
                return Err(UltraBertError::UnsupportedModule {
                    module: kind.to_string(),
                });
            }
        }
    }
    let parsed_pooling = if let Some(pooling_path) = pooling_path.or_else(|| {
        let default = path.join("1_Pooling").join("config.json");
        default.is_file().then_some(default)
    }) {
        let config: RawPoolingConfig = read_json(&pooling_path).map_err(UltraBertError::from)?;
        Some(match config.pooling_mode.as_deref().unwrap_or("") {
            "cls" => UltraBertPooling::Cls,
            "mean" => UltraBertPooling::Mean,
            "lasttoken" | "last_token" => UltraBertPooling::LastToken,
            "" => return Err(UltraBertError::MissingPooling),
            other => {
                return Err(UltraBertError::UnsupportedPooling {
                    mode: other.to_string(),
                })
            }
        })
    } else {
        None
    };
    let pooling = match task {
        UltraBertTask::Embedding => parsed_pooling.ok_or(UltraBertError::MissingPooling)?,
        // The trained sequence head reads the first token, and the token head is
        // tokenwise. A Pooling module must not change those readouts.
        UltraBertTask::SequenceClassification => UltraBertPooling::Cls,
        UltraBertTask::TokenClassification => UltraBertPooling::Token,
        UltraBertTask::Decision => UltraBertPooling::Cls,
    };
    let sentence_config = path.join("config_sentence_transformers.json");
    let (query_prompt, document_prompt) = if sentence_config.is_file() {
        let config: RawSentenceBertConfig =
            read_json(&sentence_config).map_err(UltraBertError::from)?;
        let prompts = config.prompts.unwrap_or(serde_json::Value::Null);
        (
            prompts
                .get("query")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string),
            prompts
                .get("document")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string),
        )
    } else {
        (None, None)
    };
    Ok((pooling, query_prompt, document_prompt))
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{serialize, TensorView};
    use std::fs;
    use std::path::Path;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn base_config() -> serde_json::Value {
        serde_json::json!({
            "architectures": ["UltrabertModel"],
            "model_type": "ultrabert",
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "vocab_size": 32,
            "max_position_embeddings": 64,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu",
            "attention_bias": false,
            "mlp_bias": false,
            "attention_windows": [2, null],
            "rope_theta": [10000.0, 2560000.0]
        })
    }

    fn parse(value: serde_json::Value) -> Result<UltraBertConfig, UltraBertError> {
        UltraBertConfig::from_raw(serde_json::from_value(value).unwrap(), None)
    }

    fn with(
        mut value: serde_json::Value,
        key: &str,
        replacement: serde_json::Value,
    ) -> serde_json::Value {
        value[key] = replacement;
        value
    }

    #[test]
    fn parses_the_published_ultrabert_geometry_shape() {
        let mut value = base_config();
        value["hidden_size"] = serde_json::json!(768);
        value["num_attention_heads"] = serde_json::json!(6);
        value["num_key_value_heads"] = serde_json::json!(6);
        value["head_dim"] = serde_json::json!(128);
        value["num_hidden_layers"] = serde_json::json!(3);
        value["attention_windows"] = serde_json::json!([128, 128, null]);
        value["rope_theta"] = serde_json::json!([10000.0, 10000.0, 2560000.0]);
        let config = parse(value).unwrap();
        assert_eq!(config.task, UltraBertTask::Embedding);
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.num_attention_heads, 6);
        let (bases, indices) = config.rope_bases();
        assert_eq!(bases, vec![10000.0, 2560000.0]);
        assert_eq!(indices, vec![0, 0, 1]);
    }

    #[test]
    fn rejects_causal_bias_and_wrong_activations() {
        assert_eq!(
            parse(with(base_config(), "is_causal", serde_json::json!(true)))
                .unwrap_err()
                .code(),
            "causal_not_admitted"
        );
        assert_eq!(
            parse(with(
                base_config(),
                "attention_bias",
                serde_json::json!(true)
            ))
            .unwrap_err()
            .code(),
            "bias_not_admitted"
        );
        assert_eq!(
            parse(with(base_config(), "hidden_act", serde_json::json!("gelu")))
                .unwrap_err()
                .code(),
            "unsupported_activation"
        );
    }

    #[test]
    fn rejects_mismatched_layer_lists_and_geometry() {
        assert_eq!(
            parse(with(
                base_config(),
                "attention_windows",
                serde_json::json!([2])
            ))
            .unwrap_err()
            .code(),
            "layer_list_length"
        );
        assert_eq!(
            parse(with(
                base_config(),
                "rope_theta",
                serde_json::json!([10000.0])
            ))
            .unwrap_err()
            .code(),
            "layer_list_length"
        );
        let mut value = base_config();
        value["head_dim"] = serde_json::json!(8);
        assert_eq!(parse(value).unwrap_err().code(), "head_dim_mismatch");
        let mut value = base_config();
        value["num_key_value_heads"] = serde_json::json!(1);
        assert_eq!(parse(value).unwrap().num_key_value_heads, 1);
        let mut value = base_config();
        value["hidden_size"] = serde_json::json!(12);
        value["num_attention_heads"] = serde_json::json!(3);
        value["num_key_value_heads"] = serde_json::json!(2);
        value["head_dim"] = serde_json::json!(4);
        assert_eq!(parse(value).unwrap_err().code(), "unsupported_kv_heads");
    }

    fn batch(input_ids: Vec<u32>, cumulative_seq_lengths: Vec<u32>) -> crate::Batch {
        let sequences = cumulative_seq_lengths.len() - 1;
        let length = input_ids.len();
        crate::Batch {
            multimodal: vec![None; sequences],
            input_ids,
            token_type_ids: vec![0; length],
            position_ids: (0..length as u32).collect(),
            cumulative_seq_lengths,
            max_length: 2,
            pooled_indices: vec![],
            raw_indices: vec![],
            compact_input_ids: None,
            compact_position_ids: None,
            scatter_unfold: None,
            fold_gather: None,
            tokens: vec![],
            offsets: vec![],
        }
    }

    #[test]
    fn keeps_per_layer_symmetric_attention_windows() {
        let mut value = base_config();
        value["attention_windows"] = serde_json::json!([2, 3]);
        let config = parse(value).unwrap();
        assert_eq!(config.attention_windows, vec![Some(2), Some(3)]);
    }

    #[test]
    fn batch_validation_accepts_raw_only_and_rejects_bad_offsets() {
        let config = parse(base_config()).unwrap();
        let mut raw_only = batch(vec![0, 1, 2, 3], vec![0, 2, 4]);
        raw_only.raw_indices = vec![0, 1];
        assert!(crate::ultrabert_executor::validate_batch(&config, &raw_only).is_ok());
        assert!(crate::ultrabert_executor::validate_selection(
            &raw_only,
            crate::ultrabert_executor::UltraBertExecutionMode::Embedding(
                crate::ultrabert_executor::UltraBertEmbeddingPool::Mean
            )
        )
        .is_ok());

        let mut bad = batch(vec![0, 1], vec![0, 2]);
        bad.token_type_ids.clear();
        assert_eq!(
            crate::ultrabert_executor::validate_batch(&config, &bad)
                .unwrap_err()
                .code(),
            "batch_offsets"
        );

        let mut out_of_range = batch(vec![31, 32], vec![0, 2]);
        assert_eq!(
            crate::ultrabert_executor::validate_batch(&config, &out_of_range)
                .unwrap_err()
                .code(),
            "token_out_of_range"
        );
        out_of_range.input_ids = vec![0, 1];
        assert!(crate::ultrabert_executor::validate_batch(&config, &out_of_range).is_ok());
    }

    #[test]
    fn sequence_classifier_requires_canonical_pooled_indices() {
        let mut value = base_config();
        value["architectures"] = serde_json::json!(["UltrabertForSequenceClassification"]);
        value["num_labels"] = serde_json::json!(3);
        let _config = parse(value).unwrap();
        let mut batch = batch(vec![0, 1, 2, 3], vec![0, 2, 4]);
        batch.pooled_indices = vec![1, 0];
        assert_eq!(
            crate::ultrabert_executor::validate_selection(
                &batch,
                crate::ultrabert_executor::UltraBertExecutionMode::SequenceClassification
            )
            .unwrap_err()
            .code(),
            "execution_setup"
        );
        batch.pooled_indices = vec![0, 1];
        assert!(crate::ultrabert_executor::validate_selection(
            &batch,
            crate::ultrabert_executor::UltraBertExecutionMode::SequenceClassification
        )
        .is_ok());
    }

    #[test]
    fn rejects_quantized_and_vision_modifiers() {
        let mut value = base_config();
        value["quantization_config"] = serde_json::json!({"quant_method": "fp8"});
        assert_eq!(parse(value).unwrap_err().code(), "quantized_checkpoint");
        let mut value = base_config();
        value["vision_config"] = serde_json::json!({"depth": 1});
        assert_eq!(
            parse(value).unwrap_err().code(),
            "unsupported_config_modifier"
        );
    }

    #[test]
    fn task_override_cannot_drop_or_add_a_head() {
        let mut value = base_config();
        value["architectures"] = serde_json::json!(["UltrabertForSequenceClassification"]);
        value["num_labels"] = serde_json::json!(3);
        let raw: RawUltraBertConfig = serde_json::from_value(value).unwrap();
        assert_eq!(
            UltraBertConfig::from_raw(raw.clone(), Some(UltraBertTask::Embedding))
                .unwrap_err()
                .code(),
            "task_override_mismatch"
        );
        let config = UltraBertConfig::from_raw(raw, None).unwrap();
        assert_eq!(config.task, UltraBertTask::SequenceClassification);
        assert_eq!(config.num_labels, Some(3));
    }

    struct Fixture(std::path::PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "infinity-ultrabert-qualification-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
        fn write_config(&self, value: serde_json::Value) {
            fs::write(
                self.0.join("config.json"),
                serde_json::to_vec(&value).unwrap(),
            )
            .unwrap();
        }
        fn write_pooling(&self) {
            fs::create_dir_all(self.0.join("1_Pooling")).unwrap();
            fs::write(
                self.0.join("1_Pooling").join("config.json"),
                br#"{"embedding_dimension":8,"pooling_mode":"cls","include_prompt":true}"#,
            )
            .unwrap();
        }
        fn write_modules(&self) {
            fs::write(
                self.0.join("modules.json"),
                br#"[{"idx":0,"name":"0","path":"","type":"sentence_transformers.base.modules.transformer.Transformer"},{"idx":1,"name":"1","path":"1_Pooling","type":"sentence_transformers.sentence_transformer.modules.pooling.Pooling"}]"#,
            )
            .unwrap();
        }
        fn write_tensors(&self, config: &serde_json::Value, omit: Option<&str>, extra_bias: bool) {
            let h = config["hidden_size"].as_u64().unwrap() as usize;
            let i = config["intermediate_size"].as_u64().unwrap() as usize;
            let v = config["vocab_size"].as_u64().unwrap() as usize;
            let layers = config["num_hidden_layers"].as_u64().unwrap() as usize;
            let heads = config["num_attention_heads"].as_u64().unwrap() as usize;
            let kv_heads = config
                .get("num_key_value_heads")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(heads as u64) as usize;
            let head_dim = config
                .get("head_dim")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or((h / heads) as u64) as usize;
            let q_width = heads * head_dim;
            let kv_width = kv_heads * head_dim;
            let mut tensors: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
            let mut push = |name: String, shape: Vec<usize>| {
                if omit == Some(name.as_str()) {
                    return;
                }
                let count: usize = shape.iter().product();
                let bytes = vec![0u8; count * 4];
                tensors.push((name, shape, bytes));
            };
            push("embed_tokens.weight".into(), vec![v, h]);
            push("norm.weight".into(), vec![h]);
            for layer in 0..layers {
                let p = format!("layers.{layer}");
                push(format!("{p}.input_layernorm.weight"), vec![h]);
                push(format!("{p}.post_attention_layernorm.weight"), vec![h]);
                push(format!("{p}.self_attn.q_proj.weight"), vec![q_width, h]);
                push(format!("{p}.self_attn.k_proj.weight"), vec![kv_width, h]);
                push(format!("{p}.self_attn.v_proj.weight"), vec![kv_width, h]);
                push(format!("{p}.self_attn.o_proj.weight"), vec![h, q_width]);
                push(format!("{p}.mlp.gate_proj.weight"), vec![i, h]);
                push(format!("{p}.mlp.up_proj.weight"), vec![i, h]);
                push(format!("{p}.mlp.down_proj.weight"), vec![h, i]);
            }
            let architectures = config["architectures"]
                .as_array()
                .cloned()
                .unwrap_or_default();
            if architectures.iter().any(|architecture| {
                architecture
                    .as_str()
                    .is_some_and(|name| name.ends_with("Classification"))
            }) {
                let labels = config["num_labels"].as_u64().unwrap_or(2) as usize;
                push("classifier.weight".into(), vec![labels, h]);
                push("classifier.bias".into(), vec![labels]);
            }
            if extra_bias {
                push("layers.0.self_attn.q_proj.bias".into(), vec![h]);
            }
            let views = tensors
                .iter()
                .map(|(name, shape, bytes)| {
                    (
                        name.as_str(),
                        TensorView::new(StoredDtype::F32, shape.clone(), bytes).unwrap(),
                    )
                })
                .collect::<Vec<_>>();
            let bytes = serialize(views, None).unwrap();
            fs::write(self.0.join("model.safetensors"), bytes).unwrap();
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn fixture() -> (Fixture, serde_json::Value) {
        let fixture = Fixture::new();
        let value = base_config();
        fixture.write_config(value.clone());
        fixture.write_pooling();
        fixture.write_modules();
        fixture.write_tensors(&value, None, false);
        (fixture, value)
    }

    #[test]
    fn opens_a_cpu_header_fixture_without_execution_readiness() {
        let (fixture, _) = fixture();
        let qualification = UltraBertQualification::open(&fixture.0, "bfloat16", None).unwrap();
        let report = qualification.report();
        assert_eq!(report.status, "qualification_only");
        assert!(!report.production_factory_admitted);
        assert!(!report.execution_ready);
        assert!(!report.root_native_comparison);
        assert_eq!(report.tensor_count, 2 + 2 * 9);
        assert_eq!(report.stored_dtypes, vec!["F32"]);
        assert_eq!(report.geometry.local_attention_layers, 1);
        assert_eq!(report.geometry.global_attention_layers, 1);
        assert!(report
            .missing_fa4_profiles
            .iter()
            .any(|entry| entry.contains("local")));
        assert_eq!(report.pooling, UltraBertPooling::Cls);
        let plan = qualification.plan_forward(&[0, 1, 2], &[0, 3]).unwrap();
        assert_eq!(plan.tokens, 3);
        assert_eq!(plan.batch, 1);
        assert_eq!(plan.max_length, 3);
        assert!(!plan.execution_ready);
        let error = qualification.require_gpu_ready().unwrap_err();
        assert_eq!(error.code(), "missing_fa4_profile");
        assert!(!qualification.gpu_requirements().missing_profiles.is_empty());
        assert_eq!(
            qualification
                .plan_forward(&[0, 1, 2], &[0, 3])
                .unwrap()
                .tokens,
            3
        );
    }

    #[test]
    fn rejects_missing_and_extra_tensors_with_typed_errors() {
        let fixture = Fixture::new();
        let value = base_config();
        fixture.write_config(value.clone());
        fixture.write_pooling();
        fixture.write_modules();
        fixture.write_tensors(&value, Some("layers.1.mlp.down_proj.weight"), false);
        let error = UltraBertQualification::open(&fixture.0, "bfloat16", None).unwrap_err();
        assert_eq!(error.code(), "missing_tensor");

        let fixture = Fixture::new();
        fixture.write_config(value.clone());
        fixture.write_pooling();
        fixture.write_modules();
        fixture.write_tensors(&value, None, true);
        let error = UltraBertQualification::open(&fixture.0, "bfloat16", None).unwrap_err();
        assert_eq!(error.code(), "unexpected_tensor");
    }

    #[test]
    fn rejects_unknown_pipeline_modules_instead_of_dropping_them() {
        let (fixture, _) = fixture();
        fs::write(
            fixture.0.join("modules.json"),
            br#"[{"idx":0,"name":"0","path":"","type":"sentence_transformers.base.modules.transformer.Transformer"},{"idx":1,"name":"1","path":"2_Dense","type":"sentence_transformers.models.Dense"}]"#,
        )
        .unwrap();
        let error = UltraBertQualification::open(&fixture.0, "bfloat16", None).unwrap_err();
        assert_eq!(error.code(), "unsupported_module");
    }

    #[test]
    fn rejects_non_bfloat16_runtime_and_bad_batches() {
        let (fixture, _) = fixture();
        assert_eq!(
            UltraBertQualification::open(&fixture.0, "float16", None)
                .unwrap_err()
                .code(),
            "unsupported_runtime_dtype"
        );
        let qualification = UltraBertQualification::open(&fixture.0, "bfloat16", None).unwrap();
        assert_eq!(
            qualification.plan_forward(&[], &[0]).unwrap_err().code(),
            "batch_empty"
        );
        assert_eq!(
            qualification
                .plan_forward(&[1], &[0, 2])
                .unwrap_err()
                .code(),
            "batch_offsets"
        );
        assert_eq!(
            qualification
                .plan_forward(&[999], &[0, 1])
                .unwrap_err()
                .code(),
            "token_out_of_range"
        );
    }

    #[test]
    fn accepts_a_trained_classifier_head_without_dropping_it() {
        let fixture = Fixture::new();
        let mut value = base_config();
        value["architectures"] = serde_json::json!(["UltrabertForSequenceClassification"]);
        value["num_labels"] = serde_json::json!(3);
        fixture.write_config(value.clone());
        fixture.write_pooling();
        fixture.write_modules();
        fixture.write_tensors(&value, None, false);
        let qualification = UltraBertQualification::open(&fixture.0, "bfloat16", None).unwrap();
        assert_eq!(
            qualification.report().task,
            UltraBertTask::SequenceClassification
        );
        assert_eq!(qualification.report().tensor_count, 2 + 2 * 9 + 2);
        assert_eq!(
            qualification.plan_forward(&[0, 1], &[0, 2]).unwrap().tokens,
            2
        );
        let requirements = qualification.gpu_requirements();
        assert_eq!(requirements.task, UltraBertTask::SequenceClassification);
        assert_eq!(requirements.classifier_labels, Some(3));
        assert_eq!(requirements.kernel_exports.len(), 8);
    }

    #[test]
    fn token_classifier_reports_tokenwise_pooling() {
        let fixture = Fixture::new();
        let mut value = base_config();
        value["architectures"] = serde_json::json!(["UltrabertForTokenClassification"]);
        value["num_labels"] = serde_json::json!(4);
        fixture.write_config(value.clone());
        fixture.write_pooling();
        fixture.write_modules();
        fixture.write_tensors(&value, None, false);
        let qualification = UltraBertQualification::open(&fixture.0, "bfloat16", None).unwrap();
        assert_eq!(
            qualification.report().task,
            UltraBertTask::TokenClassification
        );
        assert_eq!(qualification.report().pooling, UltraBertPooling::Token);
    }

    #[test]
    fn production_encoder_loader_rejects_ultrabert_config() {
        let fixture = Fixture::new();
        fixture.write_config(base_config());
        // The production encoder loader must reject the UltraBERT config. The
        // exact message may be the unsupported model_type or a schema mismatch
        // (UltraBERT stores rope_theta per layer), but it must not admit it.
        assert!(crate::model::BertConfig::load(&fixture.0).is_err());
    }

    #[test]
    fn production_factory_selects_ultrabert() {
        for model_type in ["ultrabert", "ultravbert"] {
            let head: crate::jina_encoder_model::RawJinaMarker =
                serde_json::from_value(serde_json::json!({"model_type": model_type})).unwrap();
            assert_eq!(
                crate::selects_architecture(&head).unwrap(),
                crate::CheckpointArchitecture::UltraBert
            );
        }
    }

    #[test]
    fn admits_bidirectional_gqa_with_per_layer_windows() {
        let mut value = base_config();
        value["num_attention_heads"] = serde_json::json!(4);
        value["num_key_value_heads"] = serde_json::json!(2);
        value["head_dim"] = serde_json::json!(2);
        value["attention_windows"] = serde_json::json!([2, 3]);
        value["rope_theta"] = serde_json::json!([10000.0, 2560000.0]);
        let config = parse(value).unwrap();
        assert_eq!(config.num_attention_heads, 4);
        assert_eq!(config.num_key_value_heads, 2);
        assert_eq!(config.q_width(), 8);
        assert_eq!(config.kv_width(), 4);
        assert_eq!(config.attention_windows, vec![Some(2), Some(3)]);
        assert_eq!(config.rope_bases().0, vec![10000.0, 2560000.0]);
    }

    #[test]
    fn loads_bidirectional_gqa_checkpoint_with_packed_kv_dimensions() {
        let fixture = Fixture::new();
        let mut value = base_config();
        value["num_attention_heads"] = serde_json::json!(4);
        value["num_key_value_heads"] = serde_json::json!(2);
        value["head_dim"] = serde_json::json!(2);
        fixture.write_config(value.clone());
        fixture.write_pooling();
        fixture.write_modules();
        fixture.write_tensors(&value, None, false);
        let qualification = UltraBertQualification::open(&fixture.0, "bfloat16", None).unwrap();
        assert_eq!(qualification.config().q_width(), 8);
        assert_eq!(qualification.config().kv_width(), 4);
        assert_eq!(qualification.weights().layers.len(), 2);
        assert_eq!(qualification.weights().layers[0].k.shape(), &[4, 8]);
        assert_eq!(qualification.weights().layers[0].v.shape(), &[4, 8]);
        assert_eq!(qualification.weights().layers[0].o.shape(), &[8, 8]);
    }

    #[test]
    fn derives_ultravbert_text_schedule_and_vision_geometry() {
        let value = serde_json::json!({
            "model_type": "ultravbert",
            "architectures": ["UltraVBertModel"],
            "pixel_shuffle_factor": 4,
            "connector_norm": true,
            "vision_config": {
                "hidden_size": 768,
                "intermediate_size": 3072,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "image_size": 512,
                "patch_size": 16,
                "layer_norm_eps": 1e-6
            },
            "text_config": {
                "architectures": ["UltrabertForMaskedLM"],
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "vocab_size": 32,
                "max_position_embeddings": 64,
                "rms_norm_eps": 1e-6,
                "hidden_act": "silu",
                "layer_types": ["sliding_attention", "full_attention"],
                "sliding_window": 2,
                "rope_parameters": {
                    "sliding_attention": {"rope_theta": 10000.0},
                    "full_attention": {"rope_theta": 2560000.0}
                }
            },
            "decision": {"proj_dim": 256}
        });
        let config = UltraBertConfig::load_value(
            value.clone(),
            Path::new(""),
            true,
            Some(UltraBertTask::Decision),
        )
        .unwrap();
        assert_eq!(config.kind, UltraBertModelKind::Vision);
        assert_eq!(config.task, UltraBertTask::Decision);
        assert_eq!(config.attention_windows, vec![Some(2), None]);
        assert_eq!(config.rope_theta, vec![10000.0, 2560000.0]);
        assert_eq!(
            config.decision,
            Some(UltraBertDecisionConfig {
                projection: 0,
                kind: UltraBertDecisionKind::Vision
            })
        );
        let vision = config.vision.unwrap();
        assert_eq!(vision.num_attention_heads, 12);
        assert_eq!(vision.pixel_shuffle_factor, 4);
    }

    #[test]
    fn official_vela3_metadata_configs_if_configured() {
        let Ok(root) = std::env::var("ULTRABERT_METADATA_ROOT") else {
            return;
        };
        let root = Path::new(&root);
        let fast = root.join("vllm-sr--vela3-fast").join("config.json");
        if fast.is_file() {
            let value: serde_json::Value = read_json(&fast).unwrap();
            let config = UltraBertConfig::load_value(
                value,
                fast.parent().unwrap_or(root),
                true,
                Some(UltraBertTask::Decision),
            )
            .unwrap();
            assert_eq!(config.kind, UltraBertModelKind::Decision);
            assert_eq!(config.hidden_size, 1024);
            assert_eq!(config.num_attention_heads, 8);
            assert_eq!(config.num_key_value_heads, 8);
            assert_eq!(config.attention_windows.len(), 24);
            assert_eq!(config.rope_bases().0, vec![10000.0, 2_560_000.0]);
            assert_eq!(
                config.decision,
                Some(UltraBertDecisionConfig {
                    projection: 256,
                    kind: UltraBertDecisionKind::Text
                })
            );
        }
        let vision = root.join("vllm-sr--vela3-fast-vision").join("config.json");
        if vision.is_file() {
            let value: serde_json::Value = read_json(&vision).unwrap();
            let config = UltraBertConfig::load_value(
                value,
                vision.parent().unwrap_or(root),
                true,
                Some(UltraBertTask::Decision),
            )
            .unwrap();
            assert_eq!(config.kind, UltraBertModelKind::Vision);
            assert_eq!(
                config.vision.map(|vision| vision.num_hidden_layers),
                Some(12)
            );
            assert_eq!(
                config
                    .attention_windows
                    .iter()
                    .filter(|w| w.is_none())
                    .count(),
                8
            );
        }
    }

    #[test]
    fn builds_text_decision_markers_from_laya_requests() {
        let mut value = base_config();
        value["decision"] = serde_json::json!({"proj_dim": 256});
        let config =
            UltraBertConfig::load_value(value, Path::new(""), true, Some(UltraBertTask::Decision))
                .unwrap();
        let batch = batch(vec![0; 5], vec![0, 5]);
        let request = infinity_v3_backend_core::DecisionInput::Laya {
            question_type: 0,
            markers: vec![1, 0, 4, 4, 2],
        };
        let plan =
            crate::ultrabert_executor::UltraBertDecisionPlan::new(&config, &batch, &[request])
                .unwrap();
        assert_eq!(plan.counts, vec![2]);
        assert_eq!(plan.max_options, 2);
    }

    #[test]
    fn builds_vision_decision_spans_from_clef_requests() {
        let value = serde_json::json!({
            "model_type": "ultravbert",
            "architectures": ["UltraVBertModel"],
            "pixel_shuffle_factor": 4,
            "connector_norm": true,
            "vision_config": {
                "hidden_size": 768,
                "intermediate_size": 3072,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "image_size": 512,
                "patch_size": 16,
                "layer_norm_eps": 1e-6
            },
            "text_config": {
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "vocab_size": 32,
                "max_position_embeddings": 64,
                "rms_norm_eps": 1e-6,
                "hidden_act": "silu",
                "layer_types": ["sliding_attention", "full_attention"],
                "sliding_window": 2,
                "rope_parameters": {
                    "sliding_attention": {"rope_theta": 10000.0},
                    "full_attention": {"rope_theta": 2560000.0}
                }
            },
            "decision": {}
        });
        let config =
            UltraBertConfig::load_value(value, Path::new(""), true, Some(UltraBertTask::Decision))
                .unwrap();
        let batch = batch(vec![0; 5], vec![0, 5]);
        let request = infinity_v3_backend_core::DecisionInput::Clef {
            fields: vec![infinity_v3_backend_core::ClefField {
                kind: 0,
                question: (0, 1),
                options: vec![(0, 2), (2, 5)],
            }],
        };
        let plan =
            crate::ultrabert_executor::UltraBertDecisionPlan::new(&config, &batch, &[request])
                .unwrap();
        assert_eq!(plan.counts, vec![2]);
        assert_eq!(plan.max_options, 2);
    }

    fn text_decision_config() -> UltraBertConfig {
        let mut value = base_config();
        value["decision"] = serde_json::json!({"proj_dim": 256});
        UltraBertConfig::load_value(value, Path::new(""), true, Some(UltraBertTask::Decision))
            .unwrap()
    }

    #[test]
    fn rejects_malformed_text_decision_markers_before_dispatch() {
        let config = text_decision_config();
        let batch = batch(vec![0; 5], vec![0, 5]);
        for markers in [
            vec![5, 0, 4, 1],
            vec![1, 0, 6, 1],
            vec![1, 0, 4, 5],
            vec![1, 0],
        ] {
            let request = infinity_v3_backend_core::DecisionInput::Laya {
                question_type: 0,
                markers,
            };
            assert!(crate::ultrabert_executor::UltraBertDecisionPlan::new(
                &config,
                &batch,
                &[request],
            )
            .is_err());
        }
        let request = infinity_v3_backend_core::DecisionInput::Warmup;
        let plan =
            crate::ultrabert_executor::UltraBertDecisionPlan::new(&config, &batch, &[request])
                .unwrap();
        assert_eq!(plan.logit_elements().unwrap(), 1);
    }

    #[test]
    fn rejects_malformed_vision_decision_spans_and_protocols() {
        let value = serde_json::json!({
            "model_type": "ultravbert",
            "architectures": ["UltraVBertModel"],
            "pixel_shuffle_factor": 4,
            "connector_norm": true,
            "vision_config": {
                "hidden_size": 768,
                "intermediate_size": 3072,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "image_size": 512,
                "patch_size": 16,
                "layer_norm_eps": 1e-6
            },
            "text_config": {
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "vocab_size": 32,
                "max_position_embeddings": 64,
                "rms_norm_eps": 1e-6,
                "hidden_act": "silu",
                "layer_types": ["sliding_attention", "full_attention"],
                "sliding_window": 2,
                "rope_parameters": {
                    "sliding_attention": {"rope_theta": 10000.0},
                    "full_attention": {"rope_theta": 2560000.0}
                }
            },
            "decision": {}
        });
        let config =
            UltraBertConfig::load_value(value, Path::new(""), true, Some(UltraBertTask::Decision))
                .unwrap();
        let batch = batch(vec![0; 5], vec![0, 5]);
        for options in [vec![(0usize, 0usize)], vec![(3, 6)], vec![(4, 2)]] {
            let request = infinity_v3_backend_core::DecisionInput::Clef {
                fields: vec![infinity_v3_backend_core::ClefField {
                    kind: 0,
                    question: (0, 1),
                    options,
                }],
            };
            assert!(crate::ultrabert_executor::UltraBertDecisionPlan::new(
                &config,
                &batch,
                &[request],
            )
            .is_err());
        }
        let laya = infinity_v3_backend_core::DecisionInput::Laya {
            question_type: 0,
            markers: vec![1, 0, 4, 1],
        };
        assert!(
            crate::ultrabert_executor::UltraBertDecisionPlan::new(&config, &batch, &[laya],)
                .is_err()
        );
    }

    #[test]
    fn real_published_checkpoint_header_if_configured() {
        let Ok(path) = std::env::var("ULTRABERT_MODEL") else {
            return;
        };
        let qualification =
            UltraBertQualification::open(Path::new(&path), "bfloat16", None).unwrap();
        let report = qualification.report();
        assert_eq!(report.tensor_count, 218);
        assert_eq!(report.geometry.num_hidden_layers, 24);
        assert_eq!(report.geometry.hidden_size, 768);
        assert_eq!(report.geometry.num_attention_heads, 6);
        assert_eq!(report.geometry.head_dim, 128);
        assert_eq!(report.geometry.local_attention_layers, 16);
        assert_eq!(report.geometry.global_attention_layers, 8);
        assert_eq!(report.weight_prefix, "");
        assert!(report.fa4_requirements.iter().any(|requirement| {
            requirement.scope == "global"
                && requirement.matched_profile.as_deref() == Some("mha128")
        }));
        assert!(report.missing_fa4_profiles.is_empty());
        assert!(report.fa4_requirements.iter().any(|requirement| {
            requirement.scope == "local"
                && requirement.present
                && requirement.matched_profile.as_deref() == Some("local128")
        }));
        assert!(!report.execution_ready);
        let plan = qualification
            .plan_forward(&[0, 32, 33, 34, 35], &[0, 5])
            .unwrap();
        assert_eq!(plan.tokens, 5);
        assert_eq!(plan.local_layers, 16);
        assert_eq!(plan.global_layers, 8);
        assert!(qualification.require_gpu_ready().is_ok());
        let requirements = qualification.gpu_requirements();
        assert_eq!(requirements.task, UltraBertTask::Embedding);
        assert_eq!(requirements.rope_bases, vec![10000.0, 2_560_000.0]);
        assert_eq!(requirements.local_attention_layers, 16);
        assert_eq!(requirements.global_attention_layers, 8);
        assert_eq!(requirements.classifier_labels, None);
        assert_eq!(requirements.kernel_exports.len(), 8);
        assert!(requirements.fa4_requirements.iter().any(|requirement| {
            requirement.present && requirement.matched_profile.as_deref() == Some("mha128")
        }));
        assert!(requirements.fa4_requirements.iter().any(|requirement| {
            requirement.scope == "local"
                && requirement.present
                && requirement.matched_profile.as_deref() == Some("local128")
        }));
        assert!(requirements.missing_profiles.is_empty());
    }
}
