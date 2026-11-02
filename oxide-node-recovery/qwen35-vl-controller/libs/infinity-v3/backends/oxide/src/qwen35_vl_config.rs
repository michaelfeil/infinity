// Checked cold configuration for Qwen3.5 / Qwen3.5-MoE vision-language roots.
//
// This module deliberately does not touch the qualified dense text foundation
// (`qwen35_config`). It parses the *root* conditional-generation checkpoint,
// which is a different contract: the vision tower is described by the root
// `vision_config` object and the text geometry is nested under `text_config`.
//
// Ground truth:
// * `backends/libtorch/cpp/models/qwen35_models.cpp` accepts the four root
//   `model_type` values qwen3_5 / qwen3_5_text / qwen3_5_moe / qwen3_5_moe_text
//   and composes the Qwen3VL vision tower with `only_vision=true`.
// * `backends/libtorch/cpp/models/qwen3_vl_models.cpp` reads
//   `vision_config.{hidden_size,num_heads,spatial_merge_size,depth,
//   intermediate_size,patch_size,temporal_patch_size,in_channels,
//   num_position_embeddings,out_hidden_size,hidden_act,
//   deepstack_visual_indexes}` and always names the tower `model.visual.*`.
// * Published Qwen3.5 configs use `rope_parameters` (not `rope_scaling`),
//   `partial_rotary_factor=0.25` (rotary width 64 of head_dim 256) and
//   `mrope_section=[11,11,10]` = rotary/2.
// * The original preprocessors are `Qwen2VLImageProcessorFast` and
//   `Qwen3VLVideoProcessor` with patch 16, temporal patch 2 and merge 2.

use crate::model::read_json;
use infinity_v3_backend_core::BackendError;
use serde_json::Value;
use std::path::Path;

fn start(error: impl std::fmt::Display) -> BackendError {
    BackendError::Start(format!("Oxide qwen3.5-vl: {error}"))
}

pub const DEFAULT_RMS_NORM_EPS: f64 = 1e-6;
pub const DEFAULT_ROPE_THETA: f64 = 10_000_000.0;
pub const DEFAULT_MAX_POSITION_EMBEDDINGS: usize = 32_768;
pub const DEFAULT_VISION_ACT: &str = "gelu_pytorch_tanh";
pub const DEFAULT_TEXT_DTYPE: &str = "bfloat16";

/// Root and nested `model_type` aliases accepted by the Native constructor.
pub const ROOT_MODEL_TYPES: [&str; 4] =
    ["qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"];
pub const VISION_MODEL_TYPES: [&str; 4] = [
    "qwen3_5",
    "qwen3_5_moe",
    "qwen3_5_vision",
    "qwen3_5_moe_vision",
];
pub const TEXT_MODEL_TYPES: [&str; 4] =
    ["qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"];

/// Conditional-generation architectures that own a vision tower. Causal-LM and
/// text-model architectures are accepted by the Native text loader but have no
/// `vision_config`; they are rejected here instead of silently dropping media.
pub const VL_ARCHITECTURES: [&str; 2] = [
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
];
pub const TEXT_ONLY_ARCHITECTURES: [&str; 6] = [
    "Qwen3_5ForCausalLM",
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5TextModel",
    "Qwen3_5MoeTextModel",
    "Qwen3_5Model",
    "Qwen3_5MoeModel",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Dtype {
    F16,
    Bf16,
    F32,
}

impl Dtype {
    pub fn parse(name: &str) -> Result<Self, BackendError> {
        match name {
            "float16" | "f16" | "half" => Ok(Self::F16),
            "bfloat16" | "bf16" => Ok(Self::Bf16),
            "float32" | "f32" => Ok(Self::F32),
            other => Err(start(format!("unsupported model dtype {other}"))),
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::F16 => "float16",
            Self::Bf16 => "bfloat16",
            Self::F32 => "float32",
        }
    }

    pub const fn width(self) -> usize {
        match self {
            Self::F16 | Self::Bf16 => 2,
            Self::F32 => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TextLayerKind {
    Full,
    Linear,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub spatial_merge_size: usize,
    pub depth: usize,
    pub intermediate_size: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub in_channels: usize,
    pub num_position_embeddings: usize,
    pub out_hidden_size: usize,
    pub hidden_act: String,
    pub deepstack_visual_indexes: Vec<usize>,
    pub model_type: String,
}

impl VisionConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads.max(1)
    }

    pub fn position_side(&self) -> usize {
        (self.num_position_embeddings as f64).sqrt().round() as usize
    }

    pub fn patch_dim(&self) -> Result<usize, BackendError> {
        self.in_channels
            .checked_mul(self.temporal_patch_size)
            .and_then(|value| value.checked_mul(self.patch_size))
            .and_then(|value| value.checked_mul(self.patch_size))
            .ok_or_else(|| start("vision patch dimension overflow"))
    }

    /// Per-temporal-slice width handled by the Native fused projection split.
    pub fn temporal_patch_dim(&self) -> Result<usize, BackendError> {
        self.in_channels
            .checked_mul(self.patch_size)
            .and_then(|value| value.checked_mul(self.patch_size))
            .ok_or_else(|| start("vision temporal patch dimension overflow"))
    }

    pub fn merge_area(&self) -> Result<usize, BackendError> {
        self.spatial_merge_size
            .checked_mul(self.spatial_merge_size)
            .ok_or_else(|| start("vision merge area overflow"))
    }

    pub fn merged_tokens(&self, grid_thw: [usize; 3]) -> Result<usize, BackendError> {
        let patches = grid_thw.iter().try_fold(1usize, |count, dim| {
            count
                .checked_mul(*dim)
                .ok_or_else(|| start("image patch count overflow"))
        })?;
        let area = self.merge_area()?;
        if grid_thw.contains(&0)
            || area == 0
            || !grid_thw[1].is_multiple_of(self.spatial_merge_size)
            || !grid_thw[2].is_multiple_of(self.spatial_merge_size)
            || !patches.is_multiple_of(area)
        {
            return Err(start(format!(
                "invalid Qwen3.5 vision grid {grid_thw:?} for merge {}",
                self.spatial_merge_size
            )));
        }
        Ok(patches / area)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub hidden_act: String,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub mrope_interleaved: bool,
    pub mrope_section: [usize; 3],
    pub layer_types: Vec<TextLayerKind>,
    pub attn_output_gate: bool,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub dtype: Dtype,
    pub model_type: String,
}

impl TextConfig {
    pub fn q_width(&self) -> Result<usize, BackendError> {
        self.num_attention_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| start("query width overflow"))
    }

    pub fn kv_width(&self) -> Result<usize, BackendError> {
        self.num_key_value_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| start("key/value width overflow"))
    }

    pub fn is_moe(&self) -> bool {
        self.num_experts > 0 || self.model_type.contains("moe")
    }

    pub fn mrope_section_sum(&self) -> Result<usize, BackendError> {
        self.mrope_section.iter().try_fold(0usize, |sum, value| {
            sum.checked_add(*value)
                .ok_or_else(|| start("mRoPE section sum overflow"))
        })
    }
}

/// The two original media preprocessors, reduced to the fields that affect
/// token geometry. Mean/std are retained verbatim but never rewritten.
#[derive(Clone, Debug, PartialEq)]
pub struct MediaPreprocessConfig {
    pub image_shortest_edge: usize,
    pub image_longest_edge: usize,
    pub video_shortest_edge: usize,
    pub video_longest_edge: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub merge_size: usize,
    pub image_mean: Vec<f64>,
    pub image_std: Vec<f64>,
    pub image_processor_type: String,
    pub video_processor_type: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Qwen35VlConfig {
    pub raw: Value,
    pub architectures: Vec<String>,
    pub model_type: String,
    pub tie_word_embeddings: bool,
    pub text: TextConfig,
    pub vision: VisionConfig,
    pub media: MediaPreprocessConfig,
    pub image_token_id: Option<i64>,
    pub video_token_id: Option<i64>,
    pub vision_start_token_id: Option<i64>,
    pub vision_end_token_id: Option<i64>,
}

fn object<'a>(
    value: &'a Value,
    name: &str,
) -> Result<&'a serde_json::Map<String, Value>, BackendError> {
    value
        .get(name)
        .and_then(Value::as_object)
        .ok_or_else(|| start(format!("missing object {name}")))
}

fn req_usize(object: &serde_json::Map<String, Value>, name: &str) -> Result<usize, BackendError> {
    object
        .get(name)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| start(format!("missing or invalid positive integer {name}")))
}

fn opt_usize(
    object: &serde_json::Map<String, Value>,
    name: &str,
    fallback: usize,
) -> Result<usize, BackendError> {
    match object.get(name) {
        None | Some(Value::Null) => Ok(fallback),
        Some(value) => value
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| start(format!("invalid integer {name}"))),
    }
}

fn opt_f64(
    object: &serde_json::Map<String, Value>,
    name: &str,
    fallback: f64,
) -> Result<f64, BackendError> {
    match object.get(name) {
        None | Some(Value::Null) => Ok(fallback),
        Some(value) => value
            .as_f64()
            .ok_or_else(|| start(format!("invalid real {name}"))),
    }
}

fn opt_bool(
    object: &serde_json::Map<String, Value>,
    name: &str,
    fallback: bool,
) -> Result<bool, BackendError> {
    match object.get(name) {
        None | Some(Value::Null) => Ok(fallback),
        Some(value) => value
            .as_bool()
            .ok_or_else(|| start(format!("invalid boolean {name}"))),
    }
}

fn opt_string(
    object: &serde_json::Map<String, Value>,
    name: &str,
    fallback: &str,
) -> Result<String, BackendError> {
    match object.get(name) {
        None | Some(Value::Null) => Ok(fallback.to_owned()),
        Some(value) => value
            .as_str()
            .map(str::to_owned)
            .ok_or_else(|| start(format!("invalid string {name}"))),
    }
}

fn parse_dtype(
    root: &Value,
    text: Option<&serde_json::Map<String, Value>>,
) -> Result<Dtype, BackendError> {
    for candidate in [text, root.as_object()].into_iter().flatten() {
        if let Some(value) = candidate
            .get("dtype")
            .or_else(|| candidate.get("torch_dtype"))
        {
            if !value.is_null() {
                return Dtype::parse(
                    value
                        .as_str()
                        .ok_or_else(|| start("dtype must be a string"))?,
                );
            }
        }
    }
    Dtype::parse(DEFAULT_TEXT_DTYPE)
}

fn parse_vision(value: &Value, text_hidden: usize) -> Result<VisionConfig, BackendError> {
    let object = object(value, "vision_config")?;
    let indexes = match object.get("deepstack_visual_indexes") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(values)) => values
            .iter()
            .map(|entry| {
                entry
                    .as_u64()
                    .and_then(|value| usize::try_from(value).ok())
                    .ok_or_else(|| start("invalid deepstack_visual_indexes entry"))
            })
            .collect::<Result<Vec<_>, _>>()?,
        Some(_) => return Err(start("deepstack_visual_indexes must be an array")),
    };
    let config = VisionConfig {
        hidden_size: req_usize(object, "hidden_size")?,
        num_heads: req_usize(object, "num_heads")?,
        spatial_merge_size: opt_usize(object, "spatial_merge_size", 2)?,
        depth: req_usize(object, "depth")?,
        intermediate_size: req_usize(object, "intermediate_size")?,
        patch_size: req_usize(object, "patch_size")?,
        temporal_patch_size: req_usize(object, "temporal_patch_size")?,
        in_channels: opt_usize(object, "in_channels", 3)?,
        num_position_embeddings: req_usize(object, "num_position_embeddings")?,
        out_hidden_size: opt_usize(object, "out_hidden_size", text_hidden)?,
        hidden_act: opt_string(object, "hidden_act", DEFAULT_VISION_ACT)?,
        deepstack_visual_indexes: indexes,
        model_type: opt_string(object, "model_type", "qwen3_5")?,
    };
    if !VISION_MODEL_TYPES.contains(&config.model_type.as_str()) {
        return Err(start(format!(
            "unsupported vision_config.model_type {}; expected one of {VISION_MODEL_TYPES:?}",
            config.model_type
        )));
    }
    Ok(config)
}

fn parse_rope(
    object: &serde_json::Map<String, Value>,
    head_dim: usize,
) -> Result<(usize, f64, bool, [usize; 3]), BackendError> {
    let rope = object
        .get("rope_parameters")
        .or_else(|| object.get("rope_scaling"))
        .and_then(Value::as_object);
    let factor = match rope.and_then(|rope| rope.get("partial_rotary_factor")) {
        Some(value) => value
            .as_f64()
            .ok_or_else(|| start("invalid partial_rotary_factor"))?,
        None => opt_f64(object, "partial_rotary_factor", 1.0)?,
    };
    if !factor.is_finite() || factor <= 0.0 || factor > 1.0 {
        return Err(start("partial_rotary_factor must be within (0, 1]"));
    }
    let scaled = head_dim as f64 * factor;
    let rounded = scaled.round();
    if !rounded.is_finite()
        || (scaled - rounded).abs() > 1e-9
        || rounded <= 0.0
        || rounded > head_dim as f64
    {
        return Err(start(
            "partial_rotary_factor must produce an integral rotary width",
        ));
    }
    let rotary = rounded as usize;
    if rotary == 0 || !rotary.is_multiple_of(2) {
        return Err(start("rotary width must be positive and even"));
    }
    let theta = match rope.and_then(|rope| rope.get("rope_theta")) {
        Some(value) => value.as_f64().ok_or_else(|| start("invalid rope_theta"))?,
        None => opt_f64(object, "rope_theta", DEFAULT_ROPE_THETA)?,
    };
    let interleaved = match rope.and_then(|rope| rope.get("mrope_interleaved")) {
        Some(value) => value
            .as_bool()
            .ok_or_else(|| start("invalid mrope_interleaved"))?,
        None => opt_bool(object, "mrope_interleaved", false)?,
    };
    let section = match rope.and_then(|rope| rope.get("mrope_section")) {
        Some(Value::Array(values)) if values.len() == 3 => {
            let parse = |entry: &Value| {
                entry
                    .as_u64()
                    .and_then(|value| usize::try_from(value).ok())
                    .ok_or_else(|| start("invalid mrope_section entry"))
            };
            [parse(&values[0])?, parse(&values[1])?, parse(&values[2])?]
        }
        Some(_) => return Err(start("mrope_section must have exactly three entries")),
        None => [0, 0, 0],
    };
    Ok((rotary, theta, interleaved, section))
}

fn parse_layer_types(
    object: &serde_json::Map<String, Value>,
    layers: usize,
) -> Result<Vec<TextLayerKind>, BackendError> {
    let values = match object.get("layer_types") {
        None | Some(Value::Null) => return Ok(vec![TextLayerKind::Full; layers]),
        Some(Value::Array(values)) => values,
        Some(_) => return Err(start("layer_types must be an array")),
    };
    if values.len() != layers {
        return Err(start(format!(
            "layer_types has {} entries but num_hidden_layers is {layers}",
            values.len()
        )));
    }
    values
        .iter()
        .map(|value| match value.as_str() {
            Some("full_attention") => Ok(TextLayerKind::Full),
            Some("linear_attention") => Ok(TextLayerKind::Linear),
            other => Err(start(format!("unsupported text layer type {other:?}"))),
        })
        .collect()
}

fn parse_text(value: &Value) -> Result<TextConfig, BackendError> {
    let object = object(value, "text_config")?;
    let hidden_size = req_usize(object, "hidden_size")?;
    let num_attention_heads = req_usize(object, "num_attention_heads")?;
    let head_dim = opt_usize(object, "head_dim", hidden_size / num_attention_heads.max(1))?;
    let (rotary_dim, rope_theta, mrope_interleaved, mrope_section) = parse_rope(object, head_dim)?;
    let num_hidden_layers = req_usize(object, "num_hidden_layers")?;
    let model_type = opt_string(object, "model_type", "qwen3_5_text")?;
    if !TEXT_MODEL_TYPES.contains(&model_type.as_str()) {
        return Err(start(format!(
            "unsupported text_config.model_type {model_type}; expected one of {TEXT_MODEL_TYPES:?}"
        )));
    }
    let num_experts = opt_usize(object, "num_experts", 0)?;
    let intermediate = if object.contains_key("intermediate_size") {
        req_usize(object, "intermediate_size")?
    } else if num_experts > 0 {
        0
    } else {
        return Err(start("missing text_config.intermediate_size"));
    };
    Ok(TextConfig {
        hidden_size,
        intermediate_size: intermediate,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads: opt_usize(object, "num_key_value_heads", num_attention_heads)?,
        head_dim,
        rotary_dim,
        vocab_size: req_usize(object, "vocab_size")?,
        rms_norm_eps: opt_f64(object, "rms_norm_eps", DEFAULT_RMS_NORM_EPS)?,
        hidden_act: opt_string(object, "hidden_act", "silu")?,
        max_position_embeddings: opt_usize(
            object,
            "max_position_embeddings",
            DEFAULT_MAX_POSITION_EMBEDDINGS,
        )?,
        rope_theta,
        mrope_interleaved,
        mrope_section,
        layer_types: parse_layer_types(object, num_hidden_layers)?,
        attn_output_gate: opt_bool(object, "attn_output_gate", false)?,
        num_experts,
        num_experts_per_tok: opt_usize(object, "num_experts_per_tok", 0)?,
        moe_intermediate_size: opt_usize(object, "moe_intermediate_size", 0)?,
        shared_expert_intermediate_size: opt_usize(object, "shared_expert_intermediate_size", 0)?,
        dtype: parse_dtype(value, Some(object))?,
        model_type,
    })
}

fn parse_media(root: &Path) -> Result<MediaPreprocessConfig, BackendError> {
    // Published snapshots ship separate image/video preprocessor JSON files;
    // the Elephant/Mammoth full checkpoints ship one combined processor_config
    // with nested image_processor/video_processor objects. Both are original
    // model files and neither is rewritten.
    let combined: Option<Value> = if root.join("processor_config.json").is_file() {
        Some(
            read_json(&root.join("processor_config.json"))
                .map_err(|error| start(format!("processor_config.json: {error}")))?,
        )
    } else {
        None
    };
    let image_value = if root.join("preprocessor_config.json").is_file() {
        read_json(&root.join("preprocessor_config.json"))
            .map_err(|error| start(format!("preprocessor_config.json: {error}")))?
    } else {
        combined
            .as_ref()
            .and_then(|value| value.get("image_processor"))
            .cloned()
            .ok_or_else(|| {
                start("missing preprocessor_config.json or processor_config.image_processor")
            })?
    };
    let video_value = if root.join("video_preprocessor_config.json").is_file() {
        read_json(&root.join("video_preprocessor_config.json"))
            .map_err(|error| start(format!("video_preprocessor_config.json: {error}")))?
    } else {
        combined
            .as_ref()
            .and_then(|value| value.get("video_processor"))
            .cloned()
            .ok_or_else(|| {
                start("missing video_preprocessor_config.json or processor_config.video_processor")
            })?
    };
    let image_object = image_value
        .as_object()
        .ok_or_else(|| start("image processor config is not an object"))?;
    let video_object = video_value
        .as_object()
        .ok_or_else(|| start("video processor config is not an object"))?;
    let size = |object: &serde_json::Map<String, Value>| -> Result<(usize, usize), BackendError> {
        let size = object
            .get("size")
            .and_then(Value::as_object)
            .ok_or_else(|| start("missing media size object"))?;
        Ok((
            req_usize(size, "shortest_edge")?,
            req_usize(size, "longest_edge")?,
        ))
    };
    let (image_shortest_edge, image_longest_edge) = size(image_object)?;
    let (video_shortest_edge, video_longest_edge) = size(video_object)?;
    let patch_size = req_usize(image_object, "patch_size")?;
    let temporal_patch_size = req_usize(image_object, "temporal_patch_size")?;
    let merge_size = req_usize(image_object, "merge_size")?;
    let floats =
        |object: &serde_json::Map<String, Value>, name: &str| -> Result<Vec<f64>, BackendError> {
            match object.get(name) {
                Some(Value::Array(values)) => values
                    .iter()
                    .map(|value| {
                        value
                            .as_f64()
                            .ok_or_else(|| start(format!("invalid {name} entry")))
                    })
                    .collect(),
                _ => Err(start(format!("missing {name} array"))),
            }
        };
    Ok(MediaPreprocessConfig {
        image_shortest_edge,
        image_longest_edge,
        video_shortest_edge,
        video_longest_edge,
        patch_size,
        temporal_patch_size,
        merge_size,
        image_mean: floats(image_object, "image_mean")?,
        image_std: floats(image_object, "image_std")?,
        image_processor_type: opt_string(image_object, "image_processor_type", "")?,
        video_processor_type: opt_string(video_object, "video_processor_type", "")?,
    })
}

fn opt_id(object: &serde_json::Map<String, Value>, name: &str) -> Option<i64> {
    object.get(name).and_then(Value::as_i64)
}

impl Qwen35VlConfig {
    pub fn from_path(root: &Path) -> Result<Self, BackendError> {
        let raw: Value = read_json(&root.join("config.json"))?;
        let mut config = Self::from_value(raw)?;
        config.media = parse_media(root)?;
        config.validate_media()?;
        Ok(config)
    }

    pub fn from_json(bytes: &[u8]) -> Result<Self, BackendError> {
        let raw: Value = serde_json::from_slice(bytes).map_err(start)?;
        Self::from_value(raw)
    }

    /// Load config-only metadata from a checkpoint that intentionally has no
    /// media preprocessors (for example a cached text/vision weight sidecar).
    /// Tower tensor presence and full shapes still validate; no media policy is
    /// invented and no preprocessor file is read.
    pub fn from_config_path(root: &Path) -> Result<Self, BackendError> {
        let raw: Value = read_json(&root.join("config.json"))?;
        Self::from_value(raw)
    }

    pub fn from_value(raw: Value) -> Result<Self, BackendError> {
        let object = raw
            .as_object()
            .ok_or_else(|| start("config.json is not an object"))?;
        let architectures = match object.get("architectures") {
            Some(Value::Array(values)) => values
                .iter()
                .map(|value| {
                    value
                        .as_str()
                        .map(str::to_owned)
                        .ok_or_else(|| start("invalid architectures entry"))
                })
                .collect::<Result<Vec<_>, _>>()?,
            Some(_) => return Err(start("architectures must be an array")),
            None => Vec::new(),
        };
        let text = parse_text(&raw)?;
        let vision = parse_vision(&raw, text.hidden_size)?;
        let model_type = opt_string(object, "model_type", "")?;
        let config = Self {
            architectures,
            model_type,
            tie_word_embeddings: opt_bool(object, "tie_word_embeddings", false)?,
            text,
            vision,
            media: MediaPreprocessConfig {
                image_shortest_edge: 0,
                image_longest_edge: 0,
                video_shortest_edge: 0,
                video_longest_edge: 0,
                patch_size: 0,
                temporal_patch_size: 0,
                merge_size: 0,
                image_mean: Vec::new(),
                image_std: Vec::new(),
                image_processor_type: String::new(),
                video_processor_type: String::new(),
            },
            image_token_id: opt_id(object, "image_token_id"),
            video_token_id: opt_id(object, "video_token_id"),
            vision_start_token_id: opt_id(object, "vision_start_token_id"),
            vision_end_token_id: opt_id(object, "vision_end_token_id"),
            raw,
        };
        config.validate()?;
        Ok(config)
    }

    /// True when the root declares a conditional-generation architecture and a
    /// vision tower. Text-only roots are rejected instead of admitted as VL.
    pub fn validate(&self) -> Result<(), BackendError> {
        if !ROOT_MODEL_TYPES.contains(&self.model_type.as_str()) {
            return Err(start(format!(
                "expected a Qwen3.5 root model_type {ROOT_MODEL_TYPES:?}, got {}",
                self.model_type
            )));
        }
        if self.model_type.contains("moe") != self.text.model_type.contains("moe") {
            return Err(start(
                "root and text_config disagree about dense versus MoE text",
            ));
        }
        if !self.architectures.is_empty() {
            let mut seen_vl = false;
            for architecture in &self.architectures {
                if VL_ARCHITECTURES.contains(&architecture.as_str()) {
                    seen_vl = true;
                } else if TEXT_ONLY_ARCHITECTURES.contains(&architecture.as_str()) {
                    return Err(start(format!(
                        "architecture {architecture} is text-only and has no vision tower"
                    )));
                } else {
                    return Err(start(format!(
                        "unsupported Qwen3.5 architecture {architecture}"
                    )));
                }
            }
            if !seen_vl {
                return Err(start(
                    "architectures declare no conditional-generation root",
                ));
            }
        }
        let vision = &self.vision;
        if vision.hidden_size == 0
            || vision.num_heads == 0
            || vision.depth == 0
            || vision.spatial_merge_size == 0
            || vision.patch_size == 0
            || vision.temporal_patch_size != 2
            || vision.in_channels != 3
        {
            return Err(start("invalid Qwen3.5 vision geometry"));
        }
        if !vision.hidden_size.is_multiple_of(vision.num_heads)
            || !vision.head_dim().is_multiple_of(4)
        {
            return Err(start(
                "vision hidden_size must divide by num_heads and head_dim must be divisible by 4",
            ));
        }
        let side = vision.position_side();
        if vision.num_position_embeddings == 0
            || side.saturating_mul(side) != vision.num_position_embeddings
        {
            return Err(start(
                "vision num_position_embeddings must be a perfect square (pos_embed table)",
            ));
        }
        if vision.out_hidden_size != self.text.hidden_size {
            return Err(start(
                "vision out_hidden_size must equal text_config.hidden_size",
            ));
        }
        let mut seen = std::collections::BTreeSet::new();
        for index in &vision.deepstack_visual_indexes {
            if *index >= vision.depth {
                return Err(start(format!(
                    "deepstack vision index {index} is outside depth {}",
                    vision.depth
                )));
            }
            if !seen.insert(*index) {
                return Err(start(format!("duplicate deepstack vision index {index}")));
            }
        }
        let text = &self.text;
        if text.hidden_size == 0
            || text.num_attention_heads == 0
            || text.num_key_value_heads == 0
            || text.head_dim == 0
            || text.num_hidden_layers == 0
            || text.vocab_size == 0
        {
            return Err(start("invalid Qwen3.5 text geometry"));
        }
        if text.num_key_value_heads > text.num_attention_heads
            || !text
                .num_attention_heads
                .is_multiple_of(text.num_key_value_heads)
        {
            return Err(start("text num_key_value_heads is invalid"));
        }
        if !(text.rms_norm_eps.is_finite() && text.rms_norm_eps > 0.0) {
            return Err(start("text rms_norm_eps must be positive and finite"));
        }
        if !(text.rope_theta.is_finite() && text.rope_theta > 0.0) {
            return Err(start("text rope_theta must be positive and finite"));
        }
        if !text.mrope_interleaved {
            return Err(start(
                "Qwen3.5 vision requires interleaved text mRoPE (mrope_interleaved=true)",
            ));
        }
        if text.mrope_section_sum()? != text.rotary_dim / 2 {
            return Err(start(format!(
                "mRoPE sections {:?} do not sum to rotary/2 ({})",
                text.mrope_section,
                text.rotary_dim / 2
            )));
        }
        if !text.attn_output_gate {
            return Err(start(
                "Qwen3.5 conditional generation requires attn_output_gate=true",
            ));
        }
        if text.is_moe() {
            if text.num_experts == 0
                || text.num_experts_per_tok == 0
                || text.num_experts_per_tok > text.num_experts
                || text.moe_intermediate_size == 0
                || text.shared_expert_intermediate_size == 0
            {
                return Err(start("invalid Qwen3.5 MoE text geometry"));
            }
        } else if text.intermediate_size == 0 {
            return Err(start("dense Qwen3.5 text requires intermediate_size"));
        }
        if self.vision.deepstack_visual_indexes.is_empty() {
            // Published Qwen3.5 roots carry an empty list. Accept it, but never
            // synthesize deepstack tensors for an empty list.
        }
        Ok(())
    }

    pub fn validate_media(&self) -> Result<(), BackendError> {
        let media = &self.media;
        if media.patch_size != self.vision.patch_size
            || media.temporal_patch_size != self.vision.temporal_patch_size
            || media.merge_size != self.vision.spatial_merge_size
        {
            return Err(start(
                "image/video preprocessors disagree with vision patch geometry",
            ));
        }
        if media.image_shortest_edge == 0
            || media.image_longest_edge < media.image_shortest_edge
            || media.video_shortest_edge == 0
            || media.video_longest_edge < media.video_shortest_edge
        {
            return Err(start("invalid preprocessor pixel bounds"));
        }
        if media.image_mean.len() != self.vision.in_channels
            || media.image_std.len() != self.vision.in_channels
        {
            return Err(start(
                "preprocessor mean/std must cover every image channel",
            ));
        }
        Ok(())
    }

    /// Number of additive vision tensors for the published checkpoint layout.
    pub fn expected_vision_tensors(&self) -> usize {
        9 + 12 * self.vision.depth
    }
}
