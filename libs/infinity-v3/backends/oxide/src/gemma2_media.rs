// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Typed cold geometry for the Native EmbeddingGemma2 vision and audio towers.
//!
//! This module owns configuration parsing, the complete tower tensor inventory,
//! and host-side weight views. It never creates a CUDA context, uploads a weight,
//! or launches a kernel. Device execution lives in the sibling vision/audio
//! executors, which consume these checked shapes.
//!
//! The arithmetic source is `backends/libtorch/cpp/models/gemma_models.cpp`
//! (`vision()` and `audio()`) plus the Gemma4 Candle reference. Norm scales are
//! read as BF16 and widened to F32 with no +1 offset. Audio clipped linears keep
//! every BF16 input/output bound from the checkpoint instead of dropping them.
#![allow(dead_code)]

use crate::model::{Checkpoint, WeightSource};
use infinity_v3_backend_core::media::{
    AudioDescriptor, AudioLayoutPolicy, ImageDescriptor, ImageLayoutPolicy, MediaError,
    MediaFamily, PackedMedia,
};
use infinity_v3_backend_core::{BackendError, Batch};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;

fn start(error: impl std::fmt::Display) -> BackendError {
    BackendError::Start(format!("Oxide embedding_gemma2 media: {error}"))
}

fn media_error(error: MediaError) -> BackendError {
    BackendError::Start(format!("Oxide embedding_gemma2 media: {error}"))
}

pub(crate) const VISION_PATCH_SIZE: usize = 16;
pub(crate) const VISION_MERGE: usize = 3;
pub(crate) const AUDIO_FEATURE_SIZE: usize = 128;
pub(crate) const AUDIO_SUBSAMPLE_STRIDES: [usize; 2] = [2, 2];
const AUDIO_CONV_KERNEL: usize = 3;
const VISUAL_TEXT_WIDTH: usize = 512;

/// Warm tower runtime and its owned prepared-row handoff. Re-exported here so
/// the shared text executor has one `gemma2_media` import surface.
#[allow(unused_imports)] // Shared text executor import surface.
pub(crate) use crate::gemma2_media_exec::{MediaRuntime, PreparedMedia};

/// Placeholder token ids that bound every media span in a packed sequence.
///
/// These are media-owned so the tower executor cannot depend on the text
/// module's private config types. The shared text executor passes the same
/// seven ids it already validated.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MediaTokens {
    pub image: u32,
    pub video: u32,
    pub audio: u32,
    pub begin_image: u32,
    pub end_image: u32,
    pub begin_audio: u32,
    pub end_audio: u32,
}

impl MediaTokens {
    pub(crate) fn all(self) -> [u32; 7] {
        [
            self.image,
            self.video,
            self.audio,
            self.begin_image,
            self.end_image,
            self.begin_audio,
            self.end_audio,
        ]
    }

    pub(crate) fn is_placeholder(self, id: u32) -> bool {
        id == self.image || id == self.audio
    }
}

impl From<[u32; 7]> for MediaTokens {
    fn from(
        [image, video, audio, begin_image, end_image, begin_audio, end_audio]: [u32; 7],
    ) -> Self {
        Self {
            image,
            video,
            audio,
            begin_image,
            end_image,
            begin_audio,
            end_audio,
        }
    }
}

/// Tower metadata needed by the media executor, independent of the text model.
///
/// It is built either from the released `config.json` here or by the shared
/// text config, which owns the same fields.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct MediaConfig {
    pub text_hidden: usize,
    pub vision_layers: usize,
    pub audio_layers: usize,
    pub vision_config: Option<serde_json::Value>,
    pub audio_config: Option<serde_json::Value>,
    pub tokens: Option<MediaTokens>,
}

impl MediaConfig {
    pub(crate) fn from_config_value(
        raw: &serde_json::Value,
        text_hidden: usize,
    ) -> Result<Self, BackendError> {
        let vision_layers = tower_layers(raw.get("vision_config"), "vision_config")?;
        let audio_layers = tower_layers(raw.get("audio_config"), "audio_config")?;
        let keys = [
            "image_token_id",
            "video_token_id",
            "audio_token_id",
            "boi_token_id",
            "eoi_token_id",
            "boa_token_id",
            "eoa_token_index",
        ];
        let present = keys
            .iter()
            .filter(|name| {
                raw.get(**name)
                    .and_then(serde_json::Value::as_u64)
                    .is_some()
            })
            .count();
        let tokens = if present == 0 {
            None
        } else if present == keys.len() {
            let read = |name: &str| -> Result<u32, BackendError> {
                raw.get(name)
                    .and_then(serde_json::Value::as_u64)
                    .and_then(|value| u32::try_from(value).ok())
                    .ok_or_else(|| start(format!("missing or invalid {name}")))
            };
            let tokens = MediaTokens::from([
                read(keys[0])?,
                read(keys[1])?,
                read(keys[2])?,
                read(keys[3])?,
                read(keys[4])?,
                read(keys[5])?,
                read(keys[6])?,
            ]);
            let mut sorted = tokens.all().to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            if sorted.len() != tokens.all().len() {
                return Err(start("media placeholder token ids must be distinct"));
            }
            Some(tokens)
        } else {
            return Err(start(
                "media placeholder token ids must be present together or absent",
            ));
        };
        if (vision_layers > 0 || audio_layers > 0) && tokens.is_none() {
            return Err(start(
                "media towers require the complete media placeholder token id set",
            ));
        }
        Ok(Self {
            text_hidden,
            vision_layers,
            audio_layers,
            vision_config: raw.get("vision_config").cloned(),
            audio_config: raw.get("audio_config").cloned(),
            tokens,
        })
    }
}

fn tower_layers(value: Option<&serde_json::Value>, name: &str) -> Result<usize, BackendError> {
    let Some(value) = value else {
        return Ok(0);
    };
    if value.is_null() {
        return Ok(0);
    }
    if !value.is_object() {
        return Err(start(format!("{name} must be a JSON object")));
    }
    let count = value["num_hidden_layers"]
        .as_u64()
        .ok_or_else(|| start(format!("missing {name}.num_hidden_layers")))?;
    usize::try_from(count).map_err(|_| start(format!("{name} layer count overflow")))
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub patch_size: usize,
    pub pooling: usize,
    pub max_positions: usize,
    pub position_table_rows: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
    pub activation: VisionActivation,
    pub standardize: bool,
    pub use_clipped_linears: bool,
    pub output_width: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum VisionActivation {
    GeluTanh,
    Gelu,
    Silu,
}

impl VisionConfig {
    pub(crate) fn parse(
        value: &serde_json::Value,
        text_width: usize,
    ) -> Result<Self, BackendError> {
        let required = |name: &str| -> Result<usize, BackendError> {
            value[name]
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .filter(|value| *value > 0)
                .ok_or_else(|| start(format!("invalid vision_config {name}")))
        };
        let hidden_size = required("hidden_size")?;
        let layers = required("num_hidden_layers")?;
        let heads = required("num_attention_heads")?;
        let head_dim = value["head_dim"]
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(hidden_size / heads);
        let kv_heads = value["num_key_value_heads"]
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(heads);
        let intermediate_size = required("intermediate_size")?;
        let patch_size = value["patch_size"]
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(16);
        let pooling = value["pooling_kernel_size"]
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(3);
        let max_positions = value["max_position_embeddings"]
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(131072);
        let position_table_rows = value["position_embedding_size"]
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(max_positions);
        let eps = value["rms_norm_eps"].as_f64().unwrap_or(1e-6);
        let theta = value["rope_parameters"]["rope_theta"]
            .as_f64()
            .or_else(|| value["rope_theta"].as_f64())
            .unwrap_or(100.0);
        let activation = match value["hidden_activation"]
            .as_str()
            .unwrap_or("gelu_pytorch_tanh")
        {
            "gelu_pytorch_tanh" => VisionActivation::GeluTanh,
            "gelu" => VisionActivation::Gelu,
            "silu" => VisionActivation::Silu,
            other => {
                return Err(start(format!(
                    "unsupported vision_config hidden_activation {other}"
                )))
            }
        };
        let standardize = value["standardize"].as_bool().unwrap_or(false);
        let use_clipped_linears = value["use_clipped_linears"].as_bool().unwrap_or(false);
        if hidden_size == 0
            || layers == 0
            || heads == 0
            || kv_heads == 0
            || !heads.is_multiple_of(kv_heads)
            || head_dim == 0
            || head_dim > 256
            || !head_dim.is_multiple_of(4)
            || head_dim.checked_mul(heads) != Some(hidden_size)
            || head_dim.checked_mul(kv_heads).is_none()
            || intermediate_size == 0
            || patch_size != VISION_PATCH_SIZE
            || pooling != VISION_MERGE
            || max_positions == 0
            || position_table_rows == 0
            || !eps.is_finite()
            || eps <= 0.0
            || !theta.is_finite()
            || theta <= 0.0
            || use_clipped_linears
        {
            return Err(start(
                "unsupported Gemma4 vision geometry or clipped-linear contract",
            ));
        }
        if !(head_dim / 2).is_multiple_of(4) {
            return Err(start("vision axial rotary width is not divisible by four"));
        }
        Ok(Self {
            hidden_size,
            intermediate_size,
            layers,
            heads,
            kv_heads,
            head_dim,
            patch_size,
            pooling,
            max_positions,
            position_table_rows,
            rms_norm_eps: eps as f32,
            rope_theta: theta,
            activation,
            standardize,
            use_clipped_linears,
            output_width: text_width,
        })
    }

    pub(crate) fn patch_dim(&self) -> usize {
        self.patch_size * self.patch_size * 3
    }

    pub(crate) fn token_rows(&self, grid: [usize; 3]) -> Result<usize, BackendError> {
        if grid[0] != 1 {
            return Err(start("Gemma4 images must be single-frame"));
        }
        let (h, w) = (grid[1], grid[2]);
        if h == 0 || w == 0 || h % self.pooling != 0 || w % self.pooling != 0 {
            return Err(start(
                "Gemma4 image grid is not divisible by the pooling kernel",
            ));
        }
        (h / self.pooling)
            .checked_mul(w / self.pooling)
            .ok_or_else(|| start("vision token row overflow"))
    }

    pub(crate) fn rope_inverse(&self) -> Vec<f32> {
        let pairs = self.head_dim / 4;
        (0..pairs)
            .map(|index| {
                (1.0f64
                    / self
                        .rope_theta
                        .powf(4.0 * index as f64 / self.head_dim as f64)) as f32
            })
            .collect()
    }

    pub(crate) fn image_policy(&self) -> Result<ImageLayoutPolicy, BackendError> {
        Ok(MediaFamily::EmbeddingGemma2.image_policy())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct AudioConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub chunk: usize,
    pub context_left: usize,
    pub context_right: usize,
    pub past: usize,
    pub relative_span: usize,
    pub residual_weight: f32,
    pub rms_norm_eps: f32,
    pub logit_cap: f32,
    pub invalid_logits: f32,
    pub gradient_clipping: f32,
    pub conv_kernel: usize,
    pub ss_channels: [usize; 2],
    pub ss_strides: [usize; 2],
    pub output_proj_dims: usize,
    pub use_clipped_linears: bool,
    pub activation: AudioActivation,
    pub output_width: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AudioActivation {
    Silu,
    GeluTanh,
    Gelu,
}

impl AudioConfig {
    pub(crate) fn parse(
        value: &serde_json::Value,
        text_width: usize,
    ) -> Result<Self, BackendError> {
        let required = |name: &str| -> Result<usize, BackendError> {
            value[name]
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .filter(|value| *value > 0)
                .ok_or_else(|| start(format!("invalid audio_config {name}")))
        };
        let hidden_size = required("hidden_size")?;
        let layers = required("num_hidden_layers")?;
        let heads = required("num_attention_heads")?;
        let chunk = required("attention_chunk_size")?;
        let context_left = required("attention_context_left")?;
        let context_right = value["attention_context_right"]
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(0);
        let past = context_left
            .checked_sub(1)
            .ok_or_else(|| start("audio context_left must be positive"))?;
        let relative_span = chunk
            .checked_add(past)
            .and_then(|value| value.checked_add(context_right))
            .and_then(|value| value.checked_div(2))
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| start("audio relative span overflow"))?;
        let eps = value["rms_norm_eps"].as_f64().unwrap_or(1e-6);
        let residual = value["residual_weight"].as_f64().unwrap_or(0.5);
        let cap = value["attention_logit_cap"].as_f64().unwrap_or(50.0);
        let invalid = value["attention_invalid_logits_value"]
            .as_f64()
            .unwrap_or(-1e9);
        let clipping = value["gradient_clipping"].as_f64().unwrap_or(1e10);
        let conv_kernel = value["conv_kernel_size"]
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(5);
        let channels = value["subsampling_conv_channels"]
            .as_array()
            .ok_or_else(|| start("missing audio subsampling_conv_channels"))?;
        if channels.len() != 2 {
            return Err(start(
                "audio subsampling_conv_channels must have two entries",
            ));
        }
        let channel = |index: usize| -> Result<usize, BackendError> {
            channels[index]
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .filter(|value| *value > 0)
                .ok_or_else(|| start("invalid audio subsampling channel"))
        };
        let ss_channels = [channel(0)?, channel(1)?];
        let ss_strides = parse_strides(value)?;
        let output_proj_dims = required("output_proj_dims")?;
        let use_clipped_linears = value["use_clipped_linears"].as_bool().unwrap_or(false);
        let activation = match value["hidden_act"].as_str().unwrap_or("silu") {
            "silu" | "swish" => AudioActivation::Silu,
            "gelu_pytorch_tanh" => AudioActivation::GeluTanh,
            "gelu" => AudioActivation::Gelu,
            other => {
                return Err(start(format!(
                    "unsupported audio_config hidden_act {other}"
                )))
            }
        };
        if hidden_size % heads != 0
            || layers == 0
            || chunk == 0
            || chunk > 64
            || context_left == 0
            || relative_span == 0
            || relative_span > 16
            || hidden_size / heads > 256
            || conv_kernel == 0
            || output_proj_dims == 0
            || !eps.is_finite()
            || eps <= 0.0
            || !residual.is_finite()
            || !cap.is_finite()
            || cap <= 0.0
            || !invalid.is_finite()
            || !clipping.is_finite()
            || clipping <= 0.0
            || ss_strides != AUDIO_SUBSAMPLE_STRIDES
        {
            return Err(start("unsupported Gemma4 audio geometry"));
        }
        Ok(Self {
            hidden_size,
            intermediate_size: 0,
            layers,
            heads,
            head_dim: hidden_size / heads,
            chunk,
            context_left,
            context_right,
            past,
            relative_span,
            residual_weight: residual as f32,
            rms_norm_eps: eps as f32,
            logit_cap: cap as f32,
            invalid_logits: invalid as f32,
            gradient_clipping: clipping.min(f32::MAX as f64) as f32,
            conv_kernel,
            ss_channels,
            ss_strides,
            output_proj_dims,
            use_clipped_linears,
            activation,
            output_width: text_width,
        })
    }

    pub(crate) fn audio_policy(&self) -> Result<AudioLayoutPolicy, BackendError> {
        AudioLayoutPolicy::for_embedding_gemma2(AUDIO_FEATURE_SIZE, self.ss_strides)
            .map_err(media_error)
    }

    /// Native `gemma_audio_conv2d` output width after both stride-2 blocks.
    pub(crate) fn subsampled_width(&self) -> Result<usize, BackendError> {
        let mut width = AUDIO_FEATURE_SIZE;
        for stride in self.ss_strides {
            width = width
                .checked_add(2)
                .and_then(|value| value.checked_sub(AUDIO_CONV_KERNEL))
                .map(|value| value / stride + 1)
                .ok_or_else(|| start("audio convolution width overflow"))?;
        }
        Ok(width)
    }

    pub(crate) fn projected_width(&self) -> Result<usize, BackendError> {
        self.ss_channels[1]
            .checked_mul(self.subsampled_width()?)
            .ok_or_else(|| start("audio projected width overflow"))
    }
}

fn parse_strides(value: &serde_json::Value) -> Result<[usize; 2], BackendError> {
    let Some(value) = value.get("sscp_conv_stride_size") else {
        return Ok(AUDIO_SUBSAMPLE_STRIDES);
    };
    let stride = |index: usize| -> Option<usize> {
        let entry = value.get(index)?;
        let scalar = if entry.is_array() {
            entry.get(0)?
        } else {
            entry
        };
        scalar
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .filter(|&value| value > 0)
    };
    match (stride(0), stride(1)) {
        (Some(first), Some(second)) => Ok([first, second]),
        _ => Err(start("invalid audio sscp_conv_stride_size")),
    }
}

#[derive(Clone, Debug)]
pub(crate) struct MediaLinear {
    pub weight: WeightSource,
    pub bias: Option<WeightSource>,
    pub input_clip: Option<[f32; 2]>,
    pub output_clip: Option<[f32; 2]>,
    pub shape: [usize; 2],
}

impl MediaLinear {
    pub(crate) fn elements(&self) -> Result<usize, BackendError> {
        self.shape[0]
            .checked_mul(self.shape[1])
            .ok_or_else(|| start("linear element overflow"))
    }
}

#[derive(Clone, Debug)]
pub(crate) struct VisionLayerHost {
    pub input_norm: WeightSource,
    pub post_attention_norm: WeightSource,
    pub pre_feedforward_norm: WeightSource,
    pub post_feedforward_norm: WeightSource,
    pub q_norm: WeightSource,
    pub k_norm: WeightSource,
    pub q: MediaLinear,
    pub k: MediaLinear,
    pub v: MediaLinear,
    pub o: MediaLinear,
    pub gate: MediaLinear,
    pub up: MediaLinear,
    pub down: MediaLinear,
}

#[derive(Clone, Debug)]
pub(crate) struct VisionHost {
    pub patch_projection: MediaLinear,
    pub position_table: WeightSource,
    pub embed_projection: MediaLinear,
    pub layers: Vec<VisionLayerHost>,
}

#[derive(Clone, Debug)]
pub(crate) struct AudioMlpHost {
    pub pre_norm: WeightSource,
    pub first: MediaLinear,
    pub second: MediaLinear,
    pub post_norm: WeightSource,
}

#[derive(Clone, Debug)]
pub(crate) struct AudioLayerHost {
    pub feed_forward1: AudioMlpHost,
    pub feed_forward2: AudioMlpHost,
    pub norm_pre_attn: WeightSource,
    pub norm_post_attn: WeightSource,
    pub norm_out: WeightSource,
    pub q: MediaLinear,
    pub k: MediaLinear,
    pub v: MediaLinear,
    pub relative_k: MediaLinear,
    pub post: MediaLinear,
    pub per_dim_scale: WeightSource,
    pub lconv_pre_norm: WeightSource,
    pub linear_start: MediaLinear,
    pub depthwise_conv1d: WeightSource,
    pub conv_norm: WeightSource,
    pub linear_end: MediaLinear,
}

#[derive(Clone, Debug)]
pub(crate) struct AudioHost {
    pub subsample0_conv: WeightSource,
    pub subsample0_norm: WeightSource,
    pub subsample1_conv: WeightSource,
    pub subsample1_norm: WeightSource,
    pub input_projection: MediaLinear,
    pub output_projection: MediaLinear,
    pub embed_projection: MediaLinear,
    pub layers: Vec<AudioLayerHost>,
}

#[derive(Clone, Debug)]
pub(crate) struct MediaHost {
    pub vision: Option<VisionHost>,
    pub audio: Option<AudioHost>,
    pub vision_config: Option<VisionConfig>,
    pub audio_config: Option<AudioConfig>,
    pub tokens: Option<MediaTokens>,
    pub tree: BTreeMap<String, (Vec<usize>, safetensors::Dtype)>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TowerRoot {
    Bare,
    Model,
}

impl TowerRoot {
    const fn prefix(self) -> &'static str {
        match self {
            Self::Bare => "",
            Self::Model => "model.",
        }
    }
}

fn select_tower_root(checkpoint: &Checkpoint) -> Result<TowerRoot, BackendError> {
    let bare = checkpoint.contains("vision_tower.patch_embedder.input_proj.weight")
        || checkpoint.contains("audio_tower.subsample_conv_projection.layer0.conv.weight");
    let model = checkpoint.contains("model.vision_tower.patch_embedder.input_proj.weight")
        || checkpoint.contains("model.audio_tower.subsample_conv_projection.layer0.conv.weight");
    match (bare, model) {
        (true, false) => Ok(TowerRoot::Bare),
        (false, true) => Ok(TowerRoot::Model),
        (true, true) => Err(start(
            "checkpoint contains both bare and model-prefixed media towers",
        )),
        (false, false) => Err(start("checkpoint contains no media tower weights")),
    }
}

fn node<'a>(value: &'a serde_json::Value, path: &str) -> &'a serde_json::Value {
    let mut current = value;
    for part in path.split('.') {
        current = &current[part];
    }
    current
}

/// Build every checkpoint tensor name the towers must use for this config.
/// The returned set is compared with the actual header, so no scalar bound or
/// unused projection can be silently dropped.
fn expected_tensors(
    root: TowerRoot,
    vision: Option<&VisionConfig>,
    audio: Option<&AudioConfig>,
) -> BTreeMap<String, Vec<usize>> {
    let mut out = BTreeMap::new();
    let p = root.prefix();
    let insert = |out: &mut BTreeMap<String, Vec<usize>>, name: String, shape: Vec<usize>| {
        out.insert(name, shape);
    };
    if let Some(v) = vision {
        let h = v.hidden_size;
        let i = v.intermediate_size;
        let d = v.head_dim;
        insert(
            &mut out,
            format!("{p}vision_tower.patch_embedder.input_proj.weight"),
            vec![h, v.patch_dim()],
        );
        insert(
            &mut out,
            format!("{p}vision_tower.patch_embedder.position_embedding_table"),
            vec![2, v.position_table_rows, h],
        );
        insert(
            &mut out,
            format!("{p}embed_vision.embedding_projection.weight"),
            vec![v.output_width, h],
        );
        if v.standardize {
            insert(&mut out, format!("{p}vision_tower.std_bias"), vec![h]);
            insert(&mut out, format!("{p}vision_tower.std_scale"), vec![h]);
        }
        for layer in 0..v.layers {
            let b = format!("{p}vision_tower.encoder.layers.{layer}");
            insert(&mut out, format!("{b}.input_layernorm.weight"), vec![h]);
            insert(
                &mut out,
                format!("{b}.post_attention_layernorm.weight"),
                vec![h],
            );
            insert(
                &mut out,
                format!("{b}.pre_feedforward_layernorm.weight"),
                vec![h],
            );
            insert(
                &mut out,
                format!("{b}.post_feedforward_layernorm.weight"),
                vec![h],
            );
            insert(&mut out, format!("{b}.self_attn.q_norm.weight"), vec![d]);
            insert(&mut out, format!("{b}.self_attn.k_norm.weight"), vec![d]);
            insert(
                &mut out,
                format!("{b}.self_attn.q_proj.linear.weight"),
                vec![h, h],
            );
            insert(
                &mut out,
                format!("{b}.self_attn.k_proj.linear.weight"),
                vec![v.head_dim * v.kv_heads, h],
            );
            insert(
                &mut out,
                format!("{b}.self_attn.v_proj.linear.weight"),
                vec![v.head_dim * v.kv_heads, h],
            );
            insert(
                &mut out,
                format!("{b}.self_attn.o_proj.linear.weight"),
                vec![h, h],
            );
            insert(
                &mut out,
                format!("{b}.mlp.gate_proj.linear.weight"),
                vec![i, h],
            );
            insert(
                &mut out,
                format!("{b}.mlp.up_proj.linear.weight"),
                vec![i, h],
            );
            insert(
                &mut out,
                format!("{b}.mlp.down_proj.linear.weight"),
                vec![h, i],
            );
        }
    }
    if let Some(a) = audio {
        let h = a.hidden_size;
        let i = a.intermediate_size;
        let k = a.conv_kernel;
        insert(
            &mut out,
            format!("{p}audio_tower.subsample_conv_projection.layer0.conv.weight"),
            vec![a.ss_channels[0], 1, AUDIO_CONV_KERNEL, AUDIO_CONV_KERNEL],
        );
        insert(
            &mut out,
            format!("{p}audio_tower.subsample_conv_projection.layer0.norm.weight"),
            vec![a.ss_channels[0]],
        );
        insert(
            &mut out,
            format!("{p}audio_tower.subsample_conv_projection.layer1.conv.weight"),
            vec![
                a.ss_channels[1],
                a.ss_channels[0],
                AUDIO_CONV_KERNEL,
                AUDIO_CONV_KERNEL,
            ],
        );
        insert(
            &mut out,
            format!("{p}audio_tower.subsample_conv_projection.layer1.norm.weight"),
            vec![a.ss_channels[1]],
        );
        let projected = a.projected_width().unwrap_or(0);
        insert(
            &mut out,
            format!("{p}audio_tower.subsample_conv_projection.input_proj_linear.weight"),
            vec![h, projected],
        );
        insert(
            &mut out,
            format!("{p}audio_tower.output_proj.weight"),
            vec![a.output_proj_dims, h],
        );
        insert(
            &mut out,
            format!("{p}audio_tower.output_proj.bias"),
            vec![a.output_proj_dims],
        );
        insert(
            &mut out,
            format!("{p}embed_audio.embedding_projection.weight"),
            vec![a.output_width, a.output_proj_dims],
        );
        for layer in 0..a.layers {
            let b = format!("{p}audio_tower.layers.{layer}");
            for (name, shape) in [
                ("norm_pre_attn.weight", vec![h]),
                ("norm_post_attn.weight", vec![h]),
                ("norm_out.weight", vec![h]),
                ("self_attn.q_proj.linear.weight", vec![h, h]),
                ("self_attn.k_proj.linear.weight", vec![h, h]),
                ("self_attn.v_proj.linear.weight", vec![h, h]),
                ("self_attn.relative_k_proj.weight", vec![h, h]),
                ("self_attn.post.linear.weight", vec![h, h]),
                ("self_attn.per_dim_scale", vec![a.head_dim]),
                ("lconv1d.pre_layer_norm.weight", vec![h]),
                ("lconv1d.linear_start.linear.weight", vec![2 * h, h]),
                ("lconv1d.depthwise_conv1d.weight", vec![h, 1, k]),
                ("lconv1d.conv_norm.weight", vec![h]),
                ("lconv1d.linear_end.linear.weight", vec![h, h]),
            ] {
                insert(&mut out, format!("{b}.{name}"), shape);
            }
            for mlp in ["feed_forward1", "feed_forward2"] {
                insert(
                    &mut out,
                    format!("{b}.{mlp}.pre_layer_norm.weight"),
                    vec![h],
                );
                insert(
                    &mut out,
                    format!("{b}.{mlp}.post_layer_norm.weight"),
                    vec![h],
                );
                insert(
                    &mut out,
                    format!("{b}.{mlp}.ffw_layer_1.linear.weight"),
                    vec![i, h],
                );
                insert(
                    &mut out,
                    format!("{b}.{mlp}.ffw_layer_2.linear.weight"),
                    vec![h, i],
                );
            }
            if a.use_clipped_linears {
                for linear in [
                    "self_attn.q_proj",
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                    "self_attn.post",
                    "lconv1d.linear_start",
                    "lconv1d.linear_end",
                    "feed_forward1.ffw_layer_1",
                    "feed_forward1.ffw_layer_2",
                    "feed_forward2.ffw_layer_1",
                    "feed_forward2.ffw_layer_2",
                ] {
                    for bound in ["input_min", "input_max", "output_min", "output_max"] {
                        insert(&mut out, format!("{b}.{linear}.{bound}"), vec![]);
                    }
                }
            }
        }
    }
    out
}

fn scalar_bf16(source: &WeightSource, name: &str) -> Result<f32, BackendError> {
    if source.dtype() != safetensors::Dtype::BF16 || !source.shape().is_empty() {
        return Err(start(format!(
            "{name} must be a scalar BF16 checkpoint tensor"
        )));
    }
    let mut bits = [0u16; 1];
    source
        .read_bf16(0, &mut bits)
        .map_err(|error| start(format!("{name} read failed: {error}")))?;
    Ok(f32::from_bits(u32::from(bits[0]) << 16))
}

fn media_linear(
    checkpoint: &Checkpoint,
    name: &str,
    shape: [usize; 2],
    bias: Option<usize>,
    clipped: bool,
) -> Result<MediaLinear, BackendError> {
    let direct = format!("{name}.weight");
    let linear = format!("{name}.linear.weight");
    let weight_name = if checkpoint.contains(&direct) {
        direct
    } else {
        linear
    };
    let weight = checkpoint.source(&weight_name, &shape)?;
    if weight.dtype() != safetensors::Dtype::BF16 {
        return Err(start(format!(
            "{weight_name} must be BF16 in the released checkpoint"
        )));
    }
    let bias = match bias {
        Some(width) => Some(checkpoint.source(&format!("{name}.bias"), &[width])?),
        None => {
            if checkpoint.contains(&format!("{name}.bias")) {
                return Err(start(format!(
                    "{name}.bias is present but this Native projection is biasless"
                )));
            }
            None
        }
    };
    let (input_clip, output_clip) = if clipped {
        let min = scalar_bf16(
            &checkpoint.source(&format!("{name}.input_min"), &[])?,
            &format!("{name}.input_min"),
        )?;
        let max = scalar_bf16(
            &checkpoint.source(&format!("{name}.input_max"), &[])?,
            &format!("{name}.input_max"),
        )?;
        let out_min = scalar_bf16(
            &checkpoint.source(&format!("{name}.output_min"), &[])?,
            &format!("{name}.output_min"),
        )?;
        let out_max = scalar_bf16(
            &checkpoint.source(&format!("{name}.output_max"), &[])?,
            &format!("{name}.output_max"),
        )?;
        if !(min <= max && out_min <= out_max) {
            return Err(start(format!("{name} has inverted BF16 clip bounds")));
        }
        (Some([min, max]), Some([out_min, out_max]))
    } else {
        (None, None)
    };
    Ok(MediaLinear {
        weight,
        bias,
        input_clip,
        output_clip,
        shape,
    })
}

fn norm_source(
    checkpoint: &Checkpoint,
    name: &str,
    width: usize,
) -> Result<WeightSource, BackendError> {
    let source = checkpoint.source(name, &[width])?;
    if source.dtype() != safetensors::Dtype::BF16 {
        return Err(start(format!("{name} must be BF16")));
    }
    Ok(source)
}

impl MediaHost {
    pub(crate) fn load(path: &Path, config: &MediaConfig) -> Result<Option<Self>, BackendError> {
        let media = config;
        if media.vision_layers == 0 && media.audio_layers == 0 {
            return Ok(None);
        }
        let checkpoint = Checkpoint::open(path)?;
        let root = select_tower_root(&checkpoint)?;
        let text_width = config.text_hidden;
        let vision = media
            .vision_config
            .as_ref()
            .filter(|_| media.vision_layers > 0)
            .map(|value| VisionConfig::parse(value, text_width))
            .transpose()?;
        let mut audio = media
            .audio_config
            .as_ref()
            .filter(|_| media.audio_layers > 0)
            .map(|value| AudioConfig::parse(value, text_width))
            .transpose()?;
        if let Some(v) = &vision {
            if v.layers != media.vision_layers {
                return Err(start("vision_config layer count disagrees with the header"));
            }
        }
        if let Some(a) = &audio {
            if a.layers != media.audio_layers {
                return Err(start("audio_config layer count disagrees with the header"));
            }
            let intermediate = checkpoint
                .shape(&format!(
                    "{}audio_tower.layers.0.feed_forward1.ffw_layer_1.linear.weight",
                    root.prefix()
                ))
                .and_then(|shape| shape.first().copied())
                .filter(|value| *value > 0)
                .ok_or_else(|| start("missing audio feed-forward projection shape"))?;
            audio = Some(AudioConfig {
                intermediate_size: intermediate,
                ..a.clone()
            });
        }
        let expected = expected_tensors(root, vision.as_ref(), audio.as_ref());
        let actual: BTreeSet<&String> = checkpoint
            .names()
            .filter(|name| {
                let bare = name.starts_with("vision_tower.")
                    || name.starts_with("audio_tower.")
                    || name.starts_with("embed_vision.")
                    || name.starts_with("embed_audio.");
                let model = name.starts_with("model.vision_tower.")
                    || name.starts_with("model.audio_tower.")
                    || name.starts_with("model.embed_vision.")
                    || name.starts_with("model.embed_audio.");
                bare || model
            })
            .collect();
        let expected_names: BTreeSet<&String> = expected.keys().collect();
        if actual != expected_names {
            let missing: Vec<_> = expected_names
                .difference(&actual)
                .take(8)
                .cloned()
                .collect();
            let extra: Vec<_> = actual
                .difference(&expected_names)
                .take(8)
                .cloned()
                .collect();
            return Err(start(format!(
                "media tensor inventory mismatch: missing {missing:?} extra {extra:?}"
            )));
        }
        let mut tree = BTreeMap::new();
        for (name, shape) in &expected {
            let source = checkpoint.source(name, shape)?;
            if source.dtype() != safetensors::Dtype::BF16 {
                return Err(start(format!("{name} must be BF16")));
            }
            tree.insert(name.clone(), (shape.clone(), source.dtype()));
        }
        let vision_cfg = vision.clone();
        let audio_cfg = audio.clone();
        let vision = vision
            .as_ref()
            .map(|v| load_vision(&checkpoint, root, v))
            .transpose()?;
        let audio = audio
            .as_ref()
            .map(|a| load_audio(&checkpoint, root, a))
            .transpose()?;
        Ok(Some(Self {
            vision,
            audio,
            vision_config: vision_cfg,
            audio_config: audio_cfg,
            tokens: media.tokens,
            tree,
        }))
    }

    pub(crate) fn tensor_bytes(&self) -> Result<usize, BackendError> {
        self.tree.values().try_fold(0usize, |total, (shape, _)| {
            let elements = shape.iter().try_fold(1usize, |size, &dim| {
                size.checked_mul(dim)
                    .ok_or_else(|| start("media tensor element overflow"))
            })?;
            total
                .checked_add(
                    elements
                        .checked_mul(2)
                        .ok_or_else(|| start("media byte overflow"))?,
                )
                .ok_or_else(|| start("media tensor bytes overflow"))
        })
    }
}

fn load_vision(
    checkpoint: &Checkpoint,
    root: TowerRoot,
    config: &VisionConfig,
) -> Result<VisionHost, BackendError> {
    let p = root.prefix();
    let h = config.hidden_size;
    let d = config.head_dim;
    let i = config.intermediate_size;
    let mut layers = Vec::new();
    layers
        .try_reserve_exact(config.layers)
        .map_err(|_| start("vision layer allocation failed"))?;
    for layer in 0..config.layers {
        let b = format!("{p}vision_tower.encoder.layers.{layer}");
        layers.push(VisionLayerHost {
            input_norm: norm_source(checkpoint, &format!("{b}.input_layernorm.weight"), h)?,
            post_attention_norm: norm_source(
                checkpoint,
                &format!("{b}.post_attention_layernorm.weight"),
                h,
            )?,
            pre_feedforward_norm: norm_source(
                checkpoint,
                &format!("{b}.pre_feedforward_layernorm.weight"),
                h,
            )?,
            post_feedforward_norm: norm_source(
                checkpoint,
                &format!("{b}.post_feedforward_layernorm.weight"),
                h,
            )?,
            q_norm: norm_source(checkpoint, &format!("{b}.self_attn.q_norm.weight"), d)?,
            k_norm: norm_source(checkpoint, &format!("{b}.self_attn.k_norm.weight"), d)?,
            q: media_linear(
                checkpoint,
                &format!("{b}.self_attn.q_proj"),
                [h, h],
                None,
                false,
            )?,
            k: media_linear(
                checkpoint,
                &format!("{b}.self_attn.k_proj"),
                [d * config.kv_heads, h],
                None,
                false,
            )?,
            v: media_linear(
                checkpoint,
                &format!("{b}.self_attn.v_proj"),
                [d * config.kv_heads, h],
                None,
                false,
            )?,
            o: media_linear(
                checkpoint,
                &format!("{b}.self_attn.o_proj"),
                [h, h],
                None,
                false,
            )?,
            gate: media_linear(
                checkpoint,
                &format!("{b}.mlp.gate_proj"),
                [i, h],
                None,
                false,
            )?,
            up: media_linear(checkpoint, &format!("{b}.mlp.up_proj"), [i, h], None, false)?,
            down: media_linear(
                checkpoint,
                &format!("{b}.mlp.down_proj"),
                [h, i],
                None,
                false,
            )?,
        });
    }
    Ok(VisionHost {
        patch_projection: media_linear(
            checkpoint,
            &format!("{p}vision_tower.patch_embedder.input_proj"),
            [h, config.patch_dim()],
            None,
            false,
        )?,
        position_table: checkpoint.source(
            &format!("{p}vision_tower.patch_embedder.position_embedding_table"),
            &[2, config.position_table_rows, h],
        )?,
        embed_projection: media_linear(
            checkpoint,
            &format!("{p}embed_vision.embedding_projection"),
            [config.output_width, h],
            None,
            false,
        )?,
        layers,
    })
}

fn load_audio(
    checkpoint: &Checkpoint,
    root: TowerRoot,
    config: &AudioConfig,
) -> Result<AudioHost, BackendError> {
    let p = root.prefix();
    let h = config.hidden_size;
    let i = config.intermediate_size;
    let clipped = config.use_clipped_linears;
    let mlp = |b: &str| -> Result<AudioMlpHost, BackendError> {
        Ok(AudioMlpHost {
            pre_norm: norm_source(checkpoint, &format!("{b}.pre_layer_norm.weight"), h)?,
            first: media_linear(
                checkpoint,
                &format!("{b}.ffw_layer_1"),
                [i, h],
                None,
                clipped,
            )?,
            second: media_linear(
                checkpoint,
                &format!("{b}.ffw_layer_2"),
                [h, i],
                None,
                clipped,
            )?,
            post_norm: norm_source(checkpoint, &format!("{b}.post_layer_norm.weight"), h)?,
        })
    };
    let mut layers = Vec::new();
    layers
        .try_reserve_exact(config.layers)
        .map_err(|_| start("audio layer allocation failed"))?;
    for layer in 0..config.layers {
        let b = format!("{p}audio_tower.layers.{layer}");
        layers.push(AudioLayerHost {
            feed_forward1: mlp(&format!("{b}.feed_forward1"))?,
            feed_forward2: mlp(&format!("{b}.feed_forward2"))?,
            norm_pre_attn: norm_source(checkpoint, &format!("{b}.norm_pre_attn.weight"), h)?,
            norm_post_attn: norm_source(checkpoint, &format!("{b}.norm_post_attn.weight"), h)?,
            norm_out: norm_source(checkpoint, &format!("{b}.norm_out.weight"), h)?,
            q: media_linear(
                checkpoint,
                &format!("{b}.self_attn.q_proj"),
                [h, h],
                None,
                clipped,
            )?,
            k: media_linear(
                checkpoint,
                &format!("{b}.self_attn.k_proj"),
                [h, h],
                None,
                clipped,
            )?,
            v: media_linear(
                checkpoint,
                &format!("{b}.self_attn.v_proj"),
                [h, h],
                None,
                clipped,
            )?,
            relative_k: media_linear(
                checkpoint,
                &format!("{b}.self_attn.relative_k_proj"),
                [h, h],
                None,
                false,
            )?,
            post: media_linear(
                checkpoint,
                &format!("{b}.self_attn.post"),
                [h, h],
                None,
                clipped,
            )?,
            per_dim_scale: checkpoint
                .source(&format!("{b}.self_attn.per_dim_scale"), &[config.head_dim])?,
            lconv_pre_norm: norm_source(
                checkpoint,
                &format!("{b}.lconv1d.pre_layer_norm.weight"),
                h,
            )?,
            linear_start: media_linear(
                checkpoint,
                &format!("{b}.lconv1d.linear_start"),
                [2 * h, h],
                None,
                clipped,
            )?,
            depthwise_conv1d: checkpoint.source(
                &format!("{b}.lconv1d.depthwise_conv1d.weight"),
                &[h, 1, config.conv_kernel],
            )?,
            conv_norm: norm_source(checkpoint, &format!("{b}.lconv1d.conv_norm.weight"), h)?,
            linear_end: media_linear(
                checkpoint,
                &format!("{b}.lconv1d.linear_end"),
                [h, h],
                None,
                clipped,
            )?,
        });
    }
    Ok(AudioHost {
        subsample0_conv: checkpoint.source(
            &format!("{p}audio_tower.subsample_conv_projection.layer0.conv.weight"),
            &[
                config.ss_channels[0],
                1,
                AUDIO_CONV_KERNEL,
                AUDIO_CONV_KERNEL,
            ],
        )?,
        subsample0_norm: norm_source(
            checkpoint,
            &format!("{p}audio_tower.subsample_conv_projection.layer0.norm.weight"),
            config.ss_channels[0],
        )?,
        subsample1_conv: checkpoint.source(
            &format!("{p}audio_tower.subsample_conv_projection.layer1.conv.weight"),
            &[
                config.ss_channels[1],
                config.ss_channels[0],
                AUDIO_CONV_KERNEL,
                AUDIO_CONV_KERNEL,
            ],
        )?,
        subsample1_norm: norm_source(
            checkpoint,
            &format!("{p}audio_tower.subsample_conv_projection.layer1.norm.weight"),
            config.ss_channels[1],
        )?,
        input_projection: media_linear(
            checkpoint,
            &format!("{p}audio_tower.subsample_conv_projection.input_proj_linear"),
            [h, config.projected_width()?],
            None,
            false,
        )?,
        output_projection: media_linear(
            checkpoint,
            &format!("{p}audio_tower.output_proj"),
            [config.output_proj_dims, h],
            Some(config.output_proj_dims),
            false,
        )?,
        embed_projection: media_linear(
            checkpoint,
            &format!("{p}embed_audio.embedding_projection"),
            [config.output_width, config.output_proj_dims],
            None,
            false,
        )?,
        layers,
    })
}

/// One validated media span with absolute packed-token coordinates.
#[derive(Clone, Debug)]
pub(crate) struct MediaSpan {
    pub token_start: usize,
    pub token_count: usize,
    pub sequence_start: usize,
    pub sequence_length: usize,
    pub kind: MediaSpanKind,
}

#[derive(Clone, Debug)]
pub(crate) enum MediaSpanKind {
    Image {
        descriptor: ImageDescriptor,
        pixels: Arc<Vec<f32>>,
    },
    Audio {
        descriptor: AudioDescriptor,
        features: Arc<Vec<f32>>,
        mask: Arc<Vec<u8>>,
    },
}

#[derive(Debug)]
pub(crate) struct MediaRequest {
    pub spans: Vec<MediaSpan>,
    pub row_indices: Vec<u32>,
    pub rows: usize,
    /// Retains the processor's admission permits for the whole request.
    pub encodings: Vec<Arc<infinity_v3_backend_core::MultimodalEncoding>>,
}

impl MediaRequest {
    pub(crate) fn plan(
        batch: &Batch,
        media: &MediaConfig,
        text_hidden: usize,
        tokens: Option<MediaTokens>,
    ) -> Result<Option<Self>, BackendError> {
        if batch.multimodal.is_empty() || batch.multimodal.iter().all(Option::is_none) {
            return Ok(None);
        }
        if batch.multimodal.len() + 1 != batch.cumulative_seq_lengths.len() {
            return Err(start("multimodal entries must match sequence boundaries"));
        }
        let tokens = tokens.ok_or_else(|| start("media tokens are absent from the config"))?;
        let vision = media
            .vision_config
            .as_ref()
            .filter(|_| media.vision_layers > 0)
            .map(|value| VisionConfig::parse(value, text_hidden))
            .transpose()?;
        let audio = media
            .audio_config
            .as_ref()
            .filter(|_| media.audio_layers > 0)
            .map(|value| AudioConfig::parse(value, text_hidden))
            .transpose()?;
        let mut images = Vec::new();
        let mut audios = Vec::new();
        let mut encodings = Vec::new();
        for (index, entry) in batch.multimodal.iter().enumerate() {
            let Some(encoding) = entry else {
                continue;
            };
            encodings.push(Arc::clone(encoding));
            let base = batch.cumulative_seq_lengths[index] as usize;
            for (span_start, patches) in &encoding.images {
                if vision.is_none() {
                    return Err(start("image input requires a vision tower"));
                }
                images.push((
                    base.checked_add(*span_start)
                        .ok_or_else(|| start("image span overflow"))?,
                    Arc::clone(patches),
                ));
            }
            for (span_start, features) in &encoding.audios {
                if audio.is_none() {
                    return Err(start("audio input requires an audio tower"));
                }
                audios.push((
                    base.checked_add(*span_start)
                        .ok_or_else(|| start("audio span overflow"))?,
                    Arc::clone(features),
                ));
            }
        }
        if images.is_empty() && audios.is_empty() {
            return Ok(None);
        }
        // Use the shared media foundation for every grid/span/mask check.
        let audio_policy = audio.as_ref().map(AudioConfig::audio_policy).transpose()?;
        let packed = PackedMedia::new(
            MediaFamily::EmbeddingGemma2,
            &batch.cumulative_seq_lengths,
            &images,
            &audios,
            audio_policy,
            None,
        )
        .map_err(media_error)?;
        if packed.token_count() != batch.input_ids.len() {
            return Err(start("packed media token count disagrees with input ids"));
        }
        let mut spans = Vec::new();
        let mut row_indices = Vec::new();
        for descriptor in packed.image_descriptors() {
            let (payload_start, patches) = images
                .iter()
                .find(|(payload_start, _)| *payload_start == descriptor.token_start)
                .ok_or_else(|| start("missing image payload for descriptor"))?;
            if *payload_start != descriptor.token_start {
                return Err(start("image descriptor token start drifted"));
            }
            let expected = vision
                .as_ref()
                .ok_or_else(|| start("vision config disappeared"))?
                .token_rows(descriptor.grid_thw)?;
            if expected != descriptor.token_rows {
                return Err(start(
                    "vision token rows disagree with the media descriptor",
                ));
            }
            for row in descriptor.token_start..descriptor.token_start + descriptor.token_count {
                row_indices.push(u32::try_from(row).map_err(|_| start("media row exceeds u32"))?);
            }
            spans.push(MediaSpan {
                token_start: descriptor.token_start,
                token_count: descriptor.token_count,
                sequence_start: descriptor.sequence_start,
                sequence_length: descriptor.sequence_length,
                kind: MediaSpanKind::Image {
                    descriptor: descriptor.clone(),
                    pixels: Arc::new(patches.pixels.clone()),
                },
            });
        }
        for descriptor in packed.audio_descriptors() {
            let (payload_start, features) = audios
                .iter()
                .find(|(payload_start, _)| *payload_start == descriptor.token_start)
                .ok_or_else(|| start("missing audio payload for descriptor"))?;
            if *payload_start != descriptor.token_start {
                return Err(start("audio descriptor token start drifted"));
            }
            for row in descriptor.token_start..descriptor.token_start + descriptor.token_count {
                row_indices.push(u32::try_from(row).map_err(|_| start("media row exceeds u32"))?);
            }
            spans.push(MediaSpan {
                token_start: descriptor.token_start,
                token_count: descriptor.token_count,
                sequence_start: descriptor.sequence_start,
                sequence_length: descriptor.sequence_length,
                kind: MediaSpanKind::Audio {
                    descriptor: descriptor.clone(),
                    features: Arc::new(features.values.clone()),
                    mask: Arc::new(features.mask.clone()),
                },
            });
        }
        spans.sort_by_key(|span| span.token_start);
        row_indices.sort_unstable();
        let duplicate = row_indices.windows(2).any(|pair| pair[0] == pair[1]);
        if duplicate {
            return Err(start("media spans share a destination text row"));
        }
        let rows = row_indices.len();
        if rows == 0
            || row_indices.last().copied().map(|row| row as usize) >= Some(batch.input_ids.len())
        {
            return Err(start("media rows are empty or outside the packed input"));
        }
        // Media token ids must be exactly the configured placeholders.
        for &row in &row_indices {
            let id = batch.input_ids[row as usize];
            if ![tokens.image, tokens.audio].contains(&id) {
                return Err(start(format!(
                    "media destination row {row} is not a media placeholder token"
                )));
            }
        }
        Ok(Some(Self {
            spans,
            row_indices,
            rows,
            encodings,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use infinity_v3_backend_core::{AudioFeatures, ImagePatches, MultimodalEncoding};
    use std::path::PathBuf;

    const ACTUAL_MODEL: &str =
        "/workspace/model-performance/michaelfeil/benchmark-overview/models/google--embeddinggemma-2";

    fn fixture_config() -> MediaConfig {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/embedding_gemma2/config.json");
        let raw: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).expect("released config fixture"))
                .expect("valid released config");
        MediaConfig::from_config_value(&raw, 512).expect("parse released media config")
    }

    fn actual_config() -> Option<MediaConfig> {
        let path = PathBuf::from(ACTUAL_MODEL);
        if !path.join("config.json").is_file() {
            return None;
        }
        let raw: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path.join("config.json")).ok()?).ok()?;
        Some(MediaConfig::from_config_value(&raw, 512).expect("parse actual media config"))
    }

    #[test]
    fn actual_media_inventory_covers_every_tower_tensor() {
        let Some(config) = actual_config() else {
            return;
        };
        let host = MediaHost::load(Path::new(ACTUAL_MODEL), &config)
            .expect("load actual media header")
            .expect("media host");
        assert_eq!(host.tree.len(), 963);
        assert!(host.tensor_bytes().unwrap() > 830_000_000);
        assert_eq!(host.vision.as_ref().unwrap().layers.len(), 16);
        assert_eq!(host.audio.as_ref().unwrap().layers.len(), 12);
    }

    #[test]
    fn actual_vision_and_audio_geometry_is_native() {
        let config = fixture_config();
        let vision =
            VisionConfig::parse(config.vision_config.as_ref().unwrap(), config.text_hidden)
                .unwrap();
        assert_eq!(vision.hidden_size, 768);
        assert_eq!(vision.heads, 12);
        assert_eq!(vision.kv_heads, 12);
        assert_eq!(vision.head_dim, 64);
        assert_eq!(vision.intermediate_size, 3072);
        assert_eq!(vision.token_rows([1, 48, 48]).unwrap(), 256);
        assert_eq!(vision.rope_inverse().len(), 16);
        let audio =
            AudioConfig::parse(config.audio_config.as_ref().unwrap(), config.text_hidden).unwrap();
        assert_eq!(audio.hidden_size, 1024);
        assert_eq!(audio.head_dim, 128);
        assert_eq!(audio.past, 12);
        assert_eq!(audio.relative_span, 13);
        assert_eq!(audio.ss_channels, [128, 32]);
        assert_eq!(audio.conv_kernel, 5);
        assert!(audio.use_clipped_linears);
        assert_eq!(audio.audio_policy().unwrap().feature_size(), 128);
    }

    fn fixture_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/embedding_gemma2_media")
    }

    fn read_f32(path: &Path) -> Vec<f32> {
        let bytes = std::fs::read(path).expect("fixture bytes");
        assert_eq!(bytes.len() % 4, 0);
        bytes
            .chunks_exact(4)
            .map(|value| f32::from_le_bytes(value.try_into().unwrap()))
            .collect()
    }

    fn image_fixture() -> (MediaTokens, Arc<ImagePatches>) {
        let dir = fixture_dir();
        let meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("cats_coco.json")).unwrap()).unwrap();
        let pixels = read_f32(&dir.join("cats_coco.patches.f32"));
        assert_eq!(
            pixels.len(),
            meta["patch_values"].as_u64().unwrap() as usize
        );
        let mut hash = sha2::Sha256::new();
        use sha2::Digest;
        for value in &pixels {
            hash.update(value.to_le_bytes());
        }
        assert_eq!(
            format!("{:x}", hash.finalize()),
            meta["patches_sha256"].as_str().unwrap()
        );
        let patches = Arc::new(ImagePatches {
            pixels,
            grid_thw: [
                meta["grid_thw"][0].as_u64().unwrap() as usize,
                meta["grid_thw"][1].as_u64().unwrap() as usize,
                meta["grid_thw"][2].as_u64().unwrap() as usize,
            ],
            patch_dim: meta["patch_dim"].as_u64().unwrap() as usize,
            merge_size: meta["merge_size"].as_u64().unwrap() as usize,
        });
        let config = fixture_config();
        (config.tokens.expect("media tokens"), patches)
    }

    fn audio_fixture() -> Arc<AudioFeatures> {
        let dir = fixture_dir();
        let meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("cat_meow.json")).unwrap()).unwrap();
        let values = read_f32(&dir.join("cat_meow.features.f32"));
        let mask = std::fs::read(dir.join("cat_meow.mask.u8")).unwrap();
        use sha2::Digest;
        let mut hash = sha2::Sha256::new();
        for value in &values {
            hash.update(value.to_le_bytes());
        }
        assert_eq!(
            format!("{:x}", hash.finalize()),
            meta["values_sha256"].as_str().unwrap()
        );
        let mut hash = sha2::Sha256::new();
        hash.update(&mask);
        assert_eq!(
            format!("{:x}", hash.finalize()),
            meta["mask_sha256"].as_str().unwrap()
        );
        assert_eq!(mask.len(), meta["frames"].as_u64().unwrap() as usize);
        Arc::new(AudioFeatures {
            values,
            mask,
            feature_size: 128,
            subsample_strides: [2, 2],
        })
    }

    fn batch_with(
        tokens: MediaTokens,
        images: Vec<(usize, Arc<ImagePatches>)>,
        audios: Vec<(usize, Arc<AudioFeatures>)>,
    ) -> Batch {
        let mut input_ids = vec![2u32];
        let mut image_starts = Vec::new();
        for (_, image) in &images {
            input_ids.push(tokens.begin_image);
            image_starts.push(input_ids.len());
            let count: usize = image.token_count();
            input_ids.extend(std::iter::repeat_n(tokens.image, count));
            input_ids.push(tokens.end_image);
        }
        let mut audio_starts = Vec::new();
        for (_, audio) in &audios {
            input_ids.push(tokens.begin_audio);
            audio_starts.push(input_ids.len());
            let count: usize = audio.token_count();
            input_ids.extend(std::iter::repeat_n(tokens.audio, count));
            input_ids.push(tokens.end_audio);
        }
        let images = image_starts
            .into_iter()
            .zip(images)
            .map(|(start, (_, image))| (start, image))
            .collect();
        let audios = audio_starts
            .into_iter()
            .zip(audios)
            .map(|(start, (_, audio))| (start, audio))
            .collect();
        let length = input_ids.len() as u32;
        let position_ids: Vec<u32> = (0..length).collect();
        let encoding = Arc::new(MultimodalEncoding {
            images,
            audios,
            position_ids: std::array::from_fn(|_| position_ids.clone()),
            memory: None,
        });
        Batch {
            multimodal: vec![Some(encoding)],
            input_ids,
            token_type_ids: vec![0; length as usize],
            position_ids: position_ids.clone(),
            cumulative_seq_lengths: vec![0, length],
            max_length: length,
            pooled_indices: vec![0],
            raw_indices: Vec::new(),
            compact_input_ids: None,
            compact_position_ids: None,
            scatter_unfold: None,
            fold_gather: None,
            tokens: Vec::new(),
            offsets: Vec::new(),
        }
    }

    #[test]
    fn actual_cached_image_and_audio_plan_matches_pinned_fixtures() {
        let config = fixture_config();
        let (tokens, image) = image_fixture();
        let audio = audio_fixture();
        let expected_audio_rows = audio.token_count();
        let expected_image_rows = image.token_count();
        let batch = batch_with(tokens, vec![(0, image)], vec![(0, audio)]);
        let request = MediaRequest::plan(&batch, &config, config.text_hidden, Some(tokens))
            .expect("media plan")
            .expect("media request");
        assert_eq!(request.spans.len(), 2);
        assert_eq!(request.rows, expected_image_rows + expected_audio_rows);
        assert_eq!(request.row_indices.len(), request.rows);
        let (image_start, image_span) = request
            .spans
            .iter()
            .find_map(|span| match &span.kind {
                MediaSpanKind::Image { descriptor, .. } => {
                    Some((descriptor.token_start, descriptor.token_count))
                }
                _ => None,
            })
            .expect("image span");
        assert_eq!(image_start, 2);
        assert_eq!(image_span, 266);
        let (audio_start, audio_span) = request
            .spans
            .iter()
            .find_map(|span| match &span.kind {
                MediaSpanKind::Audio { descriptor, .. } => {
                    Some((descriptor.token_start, descriptor.token_count))
                }
                _ => None,
            })
            .expect("audio span");
        assert_eq!(audio_start, 2 + 266 + 2);
        assert_eq!(audio_span, 103);
        let expected: Vec<u32> = (2u32..2 + 266).chain(270u32..270 + 103).collect();
        assert_eq!(request.row_indices, expected);
        assert_eq!(request.encodings.len(), 1);
    }

    #[test]
    fn media_plan_rejects_wrong_payload_and_non_placeholder_rows() {
        let config = fixture_config();
        let (tokens, image) = image_fixture();
        let mut broken = (*image).clone();
        broken.pixels.pop();
        let batch = batch_with(tokens, vec![(0, Arc::new(broken))], Vec::new());
        assert!(MediaRequest::plan(&batch, &config, config.text_hidden, Some(tokens)).is_err());

        let audio = audio_fixture();
        let mut broken = AudioFeatures {
            values: audio.values.clone(),
            mask: audio.mask.clone(),
            feature_size: audio.feature_size,
            subsample_strides: audio.subsample_strides,
        };
        broken.values.truncate(127);
        let batch = batch_with(tokens, Vec::new(), vec![(0, Arc::new(broken))]);
        assert!(MediaRequest::plan(&batch, &config, config.text_hidden, Some(tokens)).is_err());

        let batch = batch_with(tokens, vec![(0, image)], Vec::new());
        let mut wrong = batch.clone();
        for id in &mut wrong.input_ids[2..2 + 266] {
            *id = 7;
        }
        assert!(MediaRequest::plan(&wrong, &config, config.text_hidden, Some(tokens)).is_err());
    }

    #[test]
    fn media_request_retains_and_releases_the_processor_owner() {
        let config = fixture_config();
        let (tokens, image) = image_fixture();
        let batch = batch_with(tokens, vec![(0, image)], Vec::new());
        let owner = Arc::clone(batch.multimodal[0].as_ref().unwrap());
        let baseline = Arc::strong_count(&owner);
        let request = MediaRequest::plan(&batch, &config, config.text_hidden, Some(tokens))
            .unwrap()
            .unwrap();
        assert_eq!(Arc::strong_count(&owner), baseline + 1);
        drop(request);
        assert_eq!(Arc::strong_count(&owner), baseline);
    }

    #[test]
    fn interleaved_media_spans_are_token_sorted_before_preparation() {
        let config = fixture_config();
        let (tokens, image) = image_fixture();
        let audio = audio_fixture();
        let mut input_ids = vec![2u32];
        let mut audios = Vec::new();
        let mut images = Vec::new();
        for _ in 0..2 {
            input_ids.push(tokens.begin_audio);
            audios.push((input_ids.len(), Arc::clone(&audio)));
            input_ids.extend(std::iter::repeat_n(tokens.audio, audio.token_count()));
            input_ids.push(tokens.end_audio);
            if audios.len() == 1 {
                input_ids.push(tokens.begin_image);
                images.push((input_ids.len(), Arc::clone(&image)));
                input_ids.extend(std::iter::repeat_n(tokens.image, image.token_count()));
                input_ids.push(tokens.end_image);
            }
        }
        let length = input_ids.len() as u32;
        let position_ids: Vec<u32> = (0..length).collect();
        let encoding = Arc::new(MultimodalEncoding {
            images,
            audios,
            position_ids: std::array::from_fn(|_| position_ids.clone()),
            memory: None,
        });
        let batch = Batch {
            multimodal: vec![Some(encoding)],
            input_ids,
            token_type_ids: vec![0; length as usize],
            position_ids,
            cumulative_seq_lengths: vec![0, length],
            max_length: length,
            pooled_indices: vec![0],
            raw_indices: Vec::new(),
            compact_input_ids: None,
            compact_position_ids: None,
            scatter_unfold: None,
            fold_gather: None,
            tokens: Vec::new(),
            offsets: Vec::new(),
        };
        let request = MediaRequest::plan(&batch, &config, config.text_hidden, Some(tokens))
            .unwrap()
            .unwrap();
        assert_eq!(request.spans.len(), 3);
        assert!(request
            .spans
            .windows(2)
            .all(|pair| pair[0].token_start < pair[1].token_start));
        assert!(matches!(request.spans[0].kind, MediaSpanKind::Audio { .. }));
        assert!(matches!(request.spans[1].kind, MediaSpanKind::Image { .. }));
        assert!(matches!(request.spans[2].kind, MediaSpanKind::Audio { .. }));
        assert_eq!(
            request.row_indices.len(),
            audio.token_count() * 2 + image.token_count()
        );
    }
}
