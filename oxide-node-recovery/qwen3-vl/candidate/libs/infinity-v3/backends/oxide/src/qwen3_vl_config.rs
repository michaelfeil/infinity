// Checked cold configuration for Qwen3-VL.
//
// The field names, defaults and validation follow the actual Native constructor
// in `backends/libtorch/cpp/models/qwen3_vl_models.cpp` and the dense Qwen3
// decoder in `decoder_models.cpp`. The original JSON value is retained verbatim;
// no caller-visible config key is rewritten or synthesized.

use crate::checkpoint::{read_json, start};
use infinity_v3_backend_core::BackendError;
use serde_json::{Map, Value};
use std::path::Path;

pub const DEFAULT_RMS_NORM_EPS: f64 = 1e-6;
pub const DEFAULT_ROPE_THETA: f64 = 1_000_000.0;
pub const DEFAULT_VISION_ACT: &str = "gelu_pytorch_tanh";
pub const DEFAULT_TEXT_ACT: &str = "silu";
/// ColQwen3 checkpoints project token states to this width before MaxSim.
pub const COLQWEN3_TOKEN_WIDTH: usize = 320;
const DEFAULT_MAX_POSITION_EMBEDDINGS: usize = 32_768;

/// Native aliases that share the concrete Qwen3-VL adapter. These are the only
/// aliases admitted here; the generic Qwen3/Mistral families are owned by other
/// agents and must not be routed through this module.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QwenVlAlias {
    Qwen3Vl,
    ColQwen3,
    NemotronColEmbed,
}

impl QwenVlAlias {
    pub fn from_model_type(model_type: &str) -> Option<Self> {
        match model_type {
            "qwen3_vl" => Some(Self::Qwen3Vl),
            "colqwen3" => Some(Self::ColQwen3),
            "qwen3_vl_nemotron_embed" => Some(Self::NemotronColEmbed),
            _ => None,
        }
    }

    pub const fn model_type(self) -> &'static str {
        match self {
            Self::Qwen3Vl => "qwen3_vl",
            Self::ColQwen3 => "colqwen3",
            Self::NemotronColEmbed => "qwen3_vl_nemotron_embed",
        }
    }
}

/// Trained readout/wrapper actually selected by the Native LibTorch loader.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReadoutKind {
    Embedding,
    ColQwen3,
    NemotronColEmbed,
    /// Generic trained relevance head over pooled VL rows. This is not a
    /// Native Qwen3-VL family; it is admitted only when the checkpoint carries
    /// a real `score.weight` and the caller explicitly asks for it.
    Rerank,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReadoutConfig {
    pub kind: ReadoutKind,
    pub multi_vector: bool,
    pub skip_final_norm: bool,
    pub transformers_math: bool,
    pub token_projection_width: Option<usize>,
    pub normalize: bool,
    /// L2 denominator floor. ColQwen3 uses `finfo(dtype).eps`.
    pub norm_eps: f64,
}

impl ReadoutConfig {
    pub fn embedding() -> Self {
        Self {
            kind: ReadoutKind::Embedding,
            multi_vector: false,
            skip_final_norm: false,
            transformers_math: false,
            token_projection_width: None,
            normalize: false,
            norm_eps: 1e-12,
        }
    }

    /// Explicit opt-in trained relevance head. No Native Qwen3-VL model_type
    /// selects this wrapper; it is provided for a caller that verified the
    /// checkpoint's `score.weight`.
    pub fn rerank() -> Self {
        Self {
            kind: ReadoutKind::Rerank,
            multi_vector: false,
            skip_final_norm: false,
            transformers_math: false,
            token_projection_width: None,
            normalize: false,
            norm_eps: 1e-12,
        }
    }

    fn from_model(alias: QwenVlAlias, architectures: &[String]) -> Self {
        match alias {
            QwenVlAlias::ColQwen3 => Self {
                kind: ReadoutKind::ColQwen3,
                multi_vector: true,
                skip_final_norm: false,
                transformers_math: true,
                token_projection_width: Some(COLQWEN3_TOKEN_WIDTH),
                normalize: true,
                norm_eps: 1e-12,
            },
            QwenVlAlias::NemotronColEmbed => Self {
                kind: ReadoutKind::NemotronColEmbed,
                multi_vector: true,
                skip_final_norm: true,
                transformers_math: true,
                token_projection_width: None,
                normalize: true,
                norm_eps: 1e-12,
            },
            QwenVlAlias::Qwen3Vl if architectures.iter().any(|value| value == "HF_ColBERT") => {
                Self {
                    kind: ReadoutKind::Embedding,
                    multi_vector: true,
                    skip_final_norm: false,
                    transformers_math: false,
                    token_projection_width: None,
                    normalize: true,
                    norm_eps: 1e-12,
                }
            }
            QwenVlAlias::Qwen3Vl => Self::embedding(),
        }
    }
}

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
    pub dtype: Dtype,
}

impl VisionConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    /// Native derives `side = round(sqrt(pos_embed.size(0)))` and requires a
    /// square table. Validation enforces the same invariant.
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
                "invalid Qwen3-VL image grid {grid_thw:?} for merge {}",
                self.spatial_merge_size
            )));
        }
        Ok(patches / area)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub hidden_act: String,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub mrope_interleaved: bool,
    pub mrope_section: [usize; 3],
    pub dtype: Dtype,
    pub model_type: String,
}

impl TextConfig {
    pub fn q_width(&self) -> Result<usize, BackendError> {
        checked_product(self.num_attention_heads, self.head_dim, "query width")
    }

    pub fn kv_width(&self) -> Result<usize, BackendError> {
        checked_product(self.num_key_value_heads, self.head_dim, "key/value width")
    }

    pub fn text_dim(&self) -> usize {
        self.head_dim
    }

    pub fn mrope_section_sum(&self) -> Result<usize, BackendError> {
        self.mrope_section.iter().try_fold(0usize, |sum, value| {
            sum.checked_add(*value)
                .ok_or_else(|| start("mRoPE section sum overflow"))
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Qwen3VlConfig {
    pub raw: Value,
    pub architectures: Vec<String>,
    pub model_type: String,
    pub alias: QwenVlAlias,
    pub readout: ReadoutConfig,
    pub tie_word_embeddings: bool,
    pub text: TextConfig,
    pub vision: VisionConfig,
    pub image_token_id: Option<i64>,
    pub video_token_id: Option<i64>,
    pub vision_start_token_id: Option<i64>,
    pub vision_end_token_id: Option<i64>,
}

fn object<'a>(value: &'a Value, name: &str) -> Result<&'a Map<String, Value>, BackendError> {
    value
        .get(name)
        .and_then(Value::as_object)
        .ok_or_else(|| start(format!("missing object {name}")))
}

fn req_usize(object: &Map<String, Value>, name: &str) -> Result<usize, BackendError> {
    object
        .get(name)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| start(format!("missing or invalid positive integer {name}")))
}

fn opt_usize(
    object: &Map<String, Value>,
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

fn opt_f64(object: &Map<String, Value>, name: &str, fallback: f64) -> Result<f64, BackendError> {
    match object.get(name) {
        None | Some(Value::Null) => Ok(fallback),
        Some(value) => value
            .as_f64()
            .ok_or_else(|| start(format!("invalid real {name}"))),
    }
}

fn opt_bool(object: &Map<String, Value>, name: &str, fallback: bool) -> Result<bool, BackendError> {
    match object.get(name) {
        None | Some(Value::Null) => Ok(fallback),
        Some(value) => value
            .as_bool()
            .ok_or_else(|| start(format!("invalid boolean {name}"))),
    }
}

fn opt_string(
    object: &Map<String, Value>,
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

fn checked_product(left: usize, right: usize, name: &str) -> Result<usize, BackendError> {
    left.checked_mul(right)
        .ok_or_else(|| start(format!("{name} overflow")))
}

fn parse_dtype(
    object: &Map<String, Value>,
    fallback: Option<Dtype>,
) -> Result<Dtype, BackendError> {
    match object.get("dtype").or_else(|| object.get("torch_dtype")) {
        Some(value) => Dtype::parse(
            value
                .as_str()
                .ok_or_else(|| start("dtype must be a string"))?,
        ),
        None => fallback.ok_or_else(|| start("missing dtype")),
    }
}

fn parse_vision(value: &Value, effective: Dtype) -> Result<VisionConfig, BackendError> {
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
    Ok(VisionConfig {
        hidden_size: req_usize(object, "hidden_size")?,
        num_heads: req_usize(object, "num_heads")?,
        spatial_merge_size: opt_usize(object, "spatial_merge_size", 2)?,
        depth: req_usize(object, "depth")?,
        intermediate_size: req_usize(object, "intermediate_size")?,
        patch_size: req_usize(object, "patch_size")?,
        temporal_patch_size: req_usize(object, "temporal_patch_size")?,
        in_channels: opt_usize(object, "in_channels", 3)?,
        num_position_embeddings: req_usize(object, "num_position_embeddings")?,
        out_hidden_size: req_usize(object, "out_hidden_size")?,
        hidden_act: opt_string(object, "hidden_act", DEFAULT_VISION_ACT)?,
        deepstack_visual_indexes: indexes,
        dtype: effective,
    })
}

fn parse_text(value: &Value, effective: Dtype) -> Result<TextConfig, BackendError> {
    let object = object(value, "text_config")?;
    let hidden_size = req_usize(object, "hidden_size")?;
    let num_attention_heads = req_usize(object, "num_attention_heads")?;
    let head_dim = opt_usize(object, "head_dim", hidden_size / num_attention_heads.max(1))?;
    // Native normalizes Nemotron ColEmbed's `rope_parameters` into
    // `text_config.rope_scaling` before constructing the Qwen3-VL adapter.
    // Both spellings are accepted here and the original JSON is untouched.
    let rope = object
        .get("rope_scaling")
        .or_else(|| object.get("rope_parameters"))
        .and_then(Value::as_object);
    let mrope_interleaved = match rope.and_then(|rope| rope.get("mrope_interleaved")) {
        Some(value) => value
            .as_bool()
            .ok_or_else(|| start("invalid mrope_interleaved"))?,
        None => false,
    };
    let mrope_section = match rope.and_then(|rope| rope.get("mrope_section")) {
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
    Ok(TextConfig {
        hidden_size,
        num_attention_heads,
        num_key_value_heads: opt_usize(object, "num_key_value_heads", num_attention_heads)?,
        head_dim,
        intermediate_size: req_usize(object, "intermediate_size")?,
        num_hidden_layers: req_usize(object, "num_hidden_layers")?,
        vocab_size: req_usize(object, "vocab_size")?,
        rms_norm_eps: opt_f64(object, "rms_norm_eps", DEFAULT_RMS_NORM_EPS)?,
        hidden_act: opt_string(object, "hidden_act", DEFAULT_TEXT_ACT)?,
        max_position_embeddings: opt_usize(
            object,
            "max_position_embeddings",
            DEFAULT_MAX_POSITION_EMBEDDINGS,
        )?,
        rope_theta: if object.get("rope_theta").is_some() {
            opt_f64(object, "rope_theta", DEFAULT_ROPE_THETA)?
        } else {
            rope.and_then(|rope| rope.get("rope_theta"))
                .and_then(Value::as_f64)
                .unwrap_or(DEFAULT_ROPE_THETA)
        },
        mrope_interleaved,
        mrope_section,
        dtype: effective,
        model_type: opt_string(object, "model_type", "qwen3_vl_text")?,
    })
}

fn opt_id(object: &Map<String, Value>, name: &str) -> Option<i64> {
    object.get(name).and_then(Value::as_i64)
}

/// Detect a Qwen3-VL-family checkpoint without building the full config.
/// Requires a real `vision_config`; text-only configs are not routed here.
pub fn detect_alias(root: &Path) -> Result<Option<QwenVlAlias>, BackendError> {
    let raw: Value = read_json(&root.join("config.json"))?;
    let Some(object) = raw.as_object() else {
        return Ok(None);
    };
    if !object.get("vision_config").is_some_and(Value::is_object) {
        return Ok(None);
    }
    Ok(object
        .get("model_type")
        .and_then(Value::as_str)
        .and_then(QwenVlAlias::from_model_type))
}

impl Qwen3VlConfig {
    pub fn from_path(root: &Path) -> Result<Self, BackendError> {
        let raw: Value = read_json(&root.join("config.json"))?;
        Self::from_value(raw)
    }

    pub fn from_json(bytes: &[u8]) -> Result<Self, BackendError> {
        let raw: Value = serde_json::from_slice(bytes).map_err(start)?;
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
        let model_type = opt_string(object, "model_type", "")?;
        let alias = QwenVlAlias::from_model_type(&model_type).ok_or_else(|| {
            start(format!(
                "expected model_type qwen3_vl/colqwen3/qwen3_vl_nemotron_embed, got {model_type}"
            ))
        })?;
        // The runtime/checkpoint storage dtype is the top-level dtype. A
        // ColQwen3 config can declare `text_config.dtype=float32` while its
        // safetensors are BF16 and the model runs BF16; Native converts weights
        // to the selected runtime dtype, so sub-config dtype is not the adapter
        // storage contract. Keep the original JSON in `raw`.
        let top_dtype = match parse_dtype(object, None) {
            Ok(dtype) => dtype,
            Err(top_error) => {
                let text_dtype = raw
                    .get("text_config")
                    .and_then(Value::as_object)
                    .ok_or_else(|| start(top_error.to_string()))
                    .and_then(|text| parse_dtype(text, None))?;
                text_dtype
            }
        };
        let mut readout = ReadoutConfig::from_model(alias, &architectures);
        if readout.multi_vector {
            readout.norm_eps = match top_dtype {
                Dtype::Bf16 => 1.0 / 128.0,
                Dtype::F16 => 1.0 / 1024.0,
                Dtype::F32 => f32::EPSILON as f64,
            };
        }
        let config = Self {
            readout,
            architectures,
            model_type,
            alias,
            tie_word_embeddings: opt_bool(object, "tie_word_embeddings", false)?,
            text: parse_text(&raw, top_dtype)?,
            vision: parse_vision(&raw, top_dtype)?,
            image_token_id: opt_id(object, "image_token_id"),
            video_token_id: opt_id(object, "video_token_id"),
            vision_start_token_id: opt_id(object, "vision_start_token_id"),
            vision_end_token_id: opt_id(object, "vision_end_token_id"),
            raw,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn plain_text_dim(&self) -> usize {
        self.text.text_dim()
    }

    pub fn is_tower_only_compatible(&self) -> bool {
        QwenVlAlias::from_model_type(&self.model_type).is_some()
    }

    pub fn validate(&self) -> Result<(), BackendError> {
        QwenVlAlias::from_model_type(&self.model_type).ok_or_else(|| {
            start(format!(
                "expected model_type qwen3_vl/colqwen3/qwen3_vl_nemotron_embed, got {}",
                self.model_type
            ))
        })?;
        let vision = &self.vision;
        if vision.hidden_size == 0
            || vision.num_heads == 0
            || vision.depth == 0
            || vision.spatial_merge_size == 0
            || vision.patch_size == 0
            || vision.temporal_patch_size != 2
            || vision.in_channels != 3
        {
            return Err(start("invalid Qwen3-VL vision geometry"));
        }
        if !vision.hidden_size.is_multiple_of(vision.num_heads)
            || !vision.head_dim().is_multiple_of(4)
        {
            return Err(start(
                "vision hidden_size must divide by num_heads and head_dim must be divisible by 4",
            ));
        }
        if vision.num_position_embeddings == 0 || vision.out_hidden_size != self.text.hidden_size {
            return Err(start(
                "vision num_position_embeddings/out_hidden_size is inconsistent with text",
            ));
        }
        let side = (vision.num_position_embeddings as f64).sqrt().round() as usize;
        if side == 0 || side.saturating_mul(side) != vision.num_position_embeddings {
            return Err(start(
                "vision num_position_embeddings must be a perfect square (pos_embed table)",
            ));
        }
        for index in &vision.deepstack_visual_indexes {
            if *index >= vision.depth {
                return Err(start(format!(
                    "deepstack vision index {index} is outside depth {}",
                    vision.depth
                )));
            }
        }
        let mut seen = std::collections::BTreeSet::new();
        for index in &vision.deepstack_visual_indexes {
            if !seen.insert(*index) {
                return Err(start(format!("duplicate deepstack vision index {index}")));
            }
        }
        let text = &self.text;
        if text.hidden_size == 0
            || text.num_attention_heads == 0
            || text.num_key_value_heads == 0
            || text.head_dim == 0
            || text.intermediate_size == 0
            || text.num_hidden_layers == 0
            || text.vocab_size == 0
        {
            return Err(start("invalid Qwen3-VL text geometry"));
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
                "Qwen3-VL requires interleaved text mRoPE (mrope_interleaved=true)",
            ));
        }
        if text.mrope_section_sum()? != text.text_dim() / 2 {
            return Err(start(format!(
                "mRoPE sections {:?} do not sum to head_dim/2 ({})",
                text.mrope_section,
                text.text_dim() / 2
            )));
        }
        if self.readout.multi_vector
            && self.readout.kind == ReadoutKind::Embedding
            && self.architectures.iter().all(|value| value != "HF_ColBERT")
        {
            return Err(start(
                "multi-vector Qwen3-VL requires HF_ColBERT, colqwen3 or qwen3_vl_nemotron_embed",
            ));
        }
        if self
            .readout
            .token_projection_width
            .is_some_and(|width| width == 0)
        {
            return Err(start("multi-vector token projection width is zero"));
        }
        if self.vision.dtype != self.text.dtype {
            return Err(start(
                "vision and text checkpoint dtypes must match for this adapter",
            ));
        }
        if self.vision.dtype == Dtype::F32 {
            return Err(start(
                "F32 Qwen3-VL weights are outside the BF16/F16 adapter contract",
            ));
        }
        Ok(())
    }

    /// Native defaults to the non-exact BF16 path for published Qwen3-VL
    /// checkpoints. The exact FP16 temporal split additionally requires a
    /// `qwen3_5*` model type; this adapter owns `qwen3_vl`, so it always reports
    /// the default path unless a caller explicitly selects multi-vector mode.
    pub fn vision_profile(&self, multi_vector: bool) -> crate::qwen3_vl_contract::VisionProfile {
        if multi_vector || self.readout.multi_vector {
            crate::qwen3_vl_contract::VisionProfile::MultiVector
        } else {
            crate::qwen3_vl_contract::VisionProfile::Default16
        }
    }
}
