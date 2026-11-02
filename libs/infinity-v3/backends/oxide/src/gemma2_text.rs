// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//!
//! Typed cold staging for the Native Gemma2 text checkpoint.
//!
//! Geometry follows the Native Gemma path in
//! `backends/libtorch/cpp/models/gemma_models.cpp`: nested `text_config`,
//! per-layer `head_dim`/KV overrides, doubled bidirectional sliding windows,
//! shared-KV producer mapping, the `[tokens, layers, ple_width]` PLE layout,
//! and the final `embedding_projection` before any pooling. This module only
//! validates configuration and lazily references checkpoint metadata through
//! the shared `Checkpoint`/`WeightSource` views. Text loading is independent of
//! the optional media tower metadata; tower weights are never bound here.
#![cfg_attr(not(test), allow(dead_code))]

use crate::decoder_model::DecoderLinear;
use crate::model::{elements, read_json, Checkpoint, WeightSource};
use infinity_v3_backend_core::BackendError;
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;

fn start(error: impl std::fmt::Display) -> BackendError {
    BackendError::Start(format!("Oxide gemma2: {error}"))
}

const NATIVE_ROOTS: [&str; 4] = ["model.language_model.", "language_model.", "model.", ""];
const MEDIA_PREFIXES: [&str; 8] = [
    "vision_tower.",
    "audio_tower.",
    "embed_vision.",
    "embed_audio.",
    "model.vision_tower.",
    "model.audio_tower.",
    "model.embed_vision.",
    "model.embed_audio.",
];
const DEFAULT_RMS_NORM_EPS: f64 = 1e-6;
const DEFAULT_ROPE_THETA: f64 = 10_000.0;
const DEFAULT_MAX_POSITION_EMBEDDINGS: usize = 8192;
const DEFAULT_SLIDING_WINDOW: usize = 4096;
const IMAGE_SOFT_TOKENS: [usize; 5] = [70, 140, 280, 560, 1120];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Gemma2Route {
    Text,
    Image,
    Video,
    Audio,
}

impl Gemma2Route {
    const fn label(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Image => "image",
            Self::Video => "video",
            Self::Audio => "audio",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Gemma2AttentionKind {
    Sliding,
    Full,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Gemma2Activation {
    GeluTanh,
    Gelu,
    Silu,
    Relu,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Gemma2PleLayout {
    TokenLayersWidth,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Gemma2MediaTokens {
    pub(crate) image: u32,
    pub(crate) video: u32,
    pub(crate) audio: u32,
    pub(crate) begin_image: u32,
    pub(crate) end_image: u32,
    pub(crate) begin_audio: u32,
    pub(crate) end_audio: u32,
}

impl Gemma2MediaTokens {
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
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Gemma2Processor {
    pub(crate) image_max_soft_tokens: usize,
    pub(crate) video_max_frames: usize,
    pub(crate) video_fps: f64,
    pub(crate) audio_sampling_rate: usize,
    pub(crate) audio_feature_size: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Gemma2Media {
    pub(crate) tokens: Option<Gemma2MediaTokens>,
    pub(crate) vision_layers: usize,
    pub(crate) audio_layers: usize,
    pub(crate) processor: Option<Gemma2Processor>,
    /// Original tower metadata retained for typed media metadata parsing.
    pub(crate) vision_config: Option<Value>,
    pub(crate) audio_config: Option<Value>,
}

impl Gemma2Media {
    pub(crate) fn to_media_config(&self, text_hidden: usize) -> crate::gemma2_media::MediaConfig {
        crate::gemma2_media::MediaConfig {
            text_hidden,
            vision_layers: self.vision_layers,
            audio_layers: self.audio_layers,
            vision_config: self.vision_config.clone(),
            audio_config: self.audio_config.clone(),
            tokens: self
                .tokens
                .map(|tokens| crate::gemma2_media::MediaTokens::from(tokens.all())),
        }
    }

    pub(crate) fn check_route(&self, route: Gemma2Route) -> Result<(), BackendError> {
        match route {
            Gemma2Route::Text => Ok(()),
            Gemma2Route::Image if self.vision_layers > 0 => Ok(()),
            Gemma2Route::Image => Err(start(
                "image inputs require a configured Gemma4 vision tower",
            )),
            Gemma2Route::Audio if self.audio_layers > 0 => Ok(()),
            Gemma2Route::Audio => Err(start(
                "audio inputs require a configured Gemma4 audio tower",
            )),
            Gemma2Route::Video => Err(start(
                "video inputs are prepared as image frames; the dedicated video route is not admitted",
            )),
        }
    }

    pub(crate) fn check_text_token_ids(&self, ids: &[u32]) -> Result<(), BackendError> {
        let Some(tokens) = self.tokens else {
            return Ok(());
        };
        let media = tokens.all();
        if let Some(id) = ids.iter().find(|id| media.contains(id)) {
            return Err(start(format!(
                "media token id {id} cannot be executed by the Gemma2 text route"
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Gemma2LayerConfig {
    pub(crate) index: usize,
    pub(crate) kind: Gemma2AttentionKind,
    pub(crate) head_dim: usize,
    pub(crate) kv_heads: usize,
    pub(crate) q_width: usize,
    pub(crate) kv_width: usize,
    pub(crate) window_left: i32,
    pub(crate) window_right: i32,
    pub(crate) rope_theta: f64,
    pub(crate) shared: bool,
    pub(crate) store_kv: bool,
    pub(crate) producer: Option<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Gemma2TextConfig {
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) vocab_size: usize,
    pub(crate) max_position_embeddings: usize,
    pub(crate) rms_norm_eps: f64,
    pub(crate) ple_width: usize,
    pub(crate) embedding_dim: usize,
    pub(crate) sliding_window: usize,
    pub(crate) rope_sliding_theta: f64,
    pub(crate) rope_full_theta: f64,
    pub(crate) activation: Gemma2Activation,
    pub(crate) layers: Vec<Gemma2LayerConfig>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Gemma2Config {
    pub(crate) text: Gemma2TextConfig,
    pub(crate) media: Gemma2Media,
}

impl Gemma2Config {
    pub(crate) fn load(path: &Path) -> Result<Self, BackendError> {
        let raw: RawGemma2Config = read_json(&path.join("config.json"))?;
        let media = parse_media_config(&raw)?;
        let text = parse_text_config(raw.text_config.as_ref())?;
        let processor = parse_processor(path)?;
        Ok(Self {
            text,
            media: Gemma2Media { processor, ..media },
        })
    }

    pub(crate) fn output_width(&self) -> usize {
        self.text.embedding_dim
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Gemma2KvOwner {
    Produced,
    Shared { producer: usize },
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Gemma2LayerPlan {
    pub(crate) index: usize,
    pub(crate) kind: Gemma2AttentionKind,
    pub(crate) head_dim: usize,
    pub(crate) q_heads: usize,
    pub(crate) kv_heads: usize,
    pub(crate) q_width: usize,
    pub(crate) kv_width: usize,
    pub(crate) window_left: i32,
    pub(crate) window_right: i32,
    pub(crate) rope_theta: f64,
    pub(crate) rope_pairs: usize,
    pub(crate) attention_scale: f64,
    pub(crate) kv_owner: Gemma2KvOwner,
    pub(crate) has_ple: bool,
    pub(crate) has_layer_scalar: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Gemma2ExecutionPlan {
    pub(crate) hidden_size: usize,
    pub(crate) ple_width: usize,
    pub(crate) ple_projection_rows: usize,
    pub(crate) ple_layout: Gemma2PleLayout,
    pub(crate) embedding_dim: usize,
    pub(crate) embedding_scale: f64,
    pub(crate) ple_scale: f64,
    pub(crate) rms_norm_eps: f64,
    pub(crate) activation: Gemma2Activation,
    pub(crate) final_projection_before_pooling: bool,
    pub(crate) layers: Vec<Gemma2LayerPlan>,
}

impl Gemma2ExecutionPlan {
    fn prepare(config: &Gemma2TextConfig) -> Result<Self, BackendError> {
        let ple_projection_rows = config
            .num_hidden_layers
            .checked_mul(config.ple_width)
            .ok_or_else(|| start("PLE projection row count overflow"))?;
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(config.layers.len())
            .map_err(start)?;
        for layer in &config.layers {
            let rope_pairs = layer.head_dim / 2;
            let kv_owner = match layer.producer {
                Some(producer) => Gemma2KvOwner::Shared { producer },
                None => Gemma2KvOwner::Produced,
            };
            layers.push(Gemma2LayerPlan {
                index: layer.index,
                kind: layer.kind,
                head_dim: layer.head_dim,
                q_heads: config.num_attention_heads,
                kv_heads: layer.kv_heads,
                q_width: layer.q_width,
                kv_width: layer.kv_width,
                window_left: layer.window_left,
                window_right: layer.window_right,
                rope_theta: layer.rope_theta,
                rope_pairs,
                attention_scale: 1.0,
                kv_owner,
                has_ple: config.ple_width > 0,
                has_layer_scalar: true,
            });
        }
        Ok(Self {
            hidden_size: config.hidden_size,
            ple_width: config.ple_width,
            ple_projection_rows,
            ple_layout: Gemma2PleLayout::TokenLayersWidth,
            embedding_dim: config.embedding_dim,
            embedding_scale: (config.hidden_size as f64).sqrt(),
            ple_scale: (config.hidden_size as f64).sqrt().recip(),
            rms_norm_eps: config.rms_norm_eps,
            activation: config.activation,
            final_projection_before_pooling: true,
            layers,
        })
    }

    pub(crate) fn check_route(&self, route: Gemma2Route) -> Result<(), BackendError> {
        match route {
            Gemma2Route::Text => Ok(()),
            other => Err(start(format!(
                "{} inputs are outside the oxide Gemma2 text plan",
                other.label()
            ))),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Gemma2PleWeights {
    pub(crate) model_projection: DecoderLinear,
    pub(crate) projection_norm: WeightSource,
}

#[derive(Clone, Debug)]
pub(crate) struct Gemma2LayerWeights {
    pub(crate) input_norm: WeightSource,
    pub(crate) post_attention_norm: WeightSource,
    pub(crate) pre_feedforward_norm: WeightSource,
    pub(crate) post_feedforward_norm: WeightSource,
    pub(crate) q_norm: WeightSource,
    pub(crate) k_norm: Option<WeightSource>,
    pub(crate) q: DecoderLinear,
    pub(crate) k: Option<DecoderLinear>,
    pub(crate) v: Option<DecoderLinear>,
    pub(crate) o: DecoderLinear,
    pub(crate) gate: DecoderLinear,
    pub(crate) up: DecoderLinear,
    pub(crate) down: DecoderLinear,
    pub(crate) layer_scalar: WeightSource,
    pub(crate) ple_input_gate: DecoderLinear,
    pub(crate) ple_projection: DecoderLinear,
    pub(crate) ple_post_norm: WeightSource,
}

#[derive(Clone, Debug)]
pub(crate) struct Gemma2Weights {
    pub(crate) embed: WeightSource,
    pub(crate) embedding_projection: DecoderLinear,
    pub(crate) final_norm: WeightSource,
    pub(crate) ple: Gemma2PleWeights,
    pub(crate) layers: Vec<Gemma2LayerWeights>,
}

#[derive(Clone, Debug)]
pub(crate) struct Gemma2Text {
    pub(crate) config: Gemma2Config,
    #[allow(dead_code)]
    pub(crate) plan: Gemma2ExecutionPlan,
    pub(crate) weights: Gemma2Weights,
}

impl Gemma2Text {
    pub(crate) fn load(path: &Path) -> Result<Self, BackendError> {
        let config = Gemma2Config::load(path)?;
        let checkpoint = Checkpoint::open(path)?;
        let root = select_root(&checkpoint)?;
        let text = &config.text;
        let hidden = text.hidden_size;
        let ple_width = text.ple_width;
        let ple_rows = text
            .num_hidden_layers
            .checked_mul(ple_width)
            .ok_or_else(|| start("PLE projection row count overflow"))?;

        let embed = checkpoint.source(
            &format!("{root}embed_tokens.weight"),
            &[text.vocab_size, hidden],
        )?;
        let embedding_projection = fetch_linear(
            &checkpoint,
            &format!("{root}embedding_projection"),
            &[text.embedding_dim, hidden],
            true,
            "final embedding projection",
        )?;
        let final_norm = checkpoint.source(&format!("{root}norm.weight"), &[hidden])?;
        let ple = Gemma2PleWeights {
            model_projection: fetch_linear(
                &checkpoint,
                &format!("{root}ple.per_layer_model_projection"),
                &[ple_rows, hidden],
                true,
                "PLE model projection",
            )?,
            projection_norm: checkpoint.source(
                &format!("{root}ple.per_layer_projection_norm.weight"),
                &[ple_width],
            )?,
        };

        let mut layers = Vec::new();
        layers.try_reserve_exact(text.layers.len()).map_err(start)?;
        for layer in &text.layers {
            let p = format!("{root}layers.{}", layer.index);
            let input_norm =
                checkpoint.source(&format!("{p}.input_layernorm.weight"), &[hidden])?;
            let post_attention_norm =
                checkpoint.source(&format!("{p}.post_attention_layernorm.weight"), &[hidden])?;
            let pre_feedforward_norm =
                checkpoint.source(&format!("{p}.pre_feedforward_layernorm.weight"), &[hidden])?;
            let post_feedforward_norm =
                checkpoint.source(&format!("{p}.post_feedforward_layernorm.weight"), &[hidden])?;
            let q_norm =
                checkpoint.source(&format!("{p}.self_attn.q_norm.weight"), &[layer.head_dim])?;
            let q = fetch_linear(
                &checkpoint,
                &format!("{p}.self_attn.q_proj"),
                &[layer.q_width, hidden],
                true,
                "query projection",
            )?;
            let (k_norm, k, v) = if layer.shared {
                (None, None, None)
            } else {
                (
                    Some(
                        checkpoint
                            .source(&format!("{p}.self_attn.k_norm.weight"), &[layer.head_dim])?,
                    ),
                    Some(fetch_linear(
                        &checkpoint,
                        &format!("{p}.self_attn.k_proj"),
                        &[layer.kv_width, hidden],
                        true,
                        "key projection",
                    )?),
                    Some(fetch_linear(
                        &checkpoint,
                        &format!("{p}.self_attn.v_proj"),
                        &[layer.kv_width, hidden],
                        true,
                        "value projection",
                    )?),
                )
            };
            let o = fetch_linear(
                &checkpoint,
                &format!("{p}.self_attn.o_proj"),
                &[hidden, layer.q_width],
                true,
                "attention output projection",
            )?;
            let gate = fetch_linear(
                &checkpoint,
                &format!("{p}.mlp.gate_proj"),
                &[text.intermediate_size, hidden],
                false,
                "MLP gate projection",
            )?;
            let up = fetch_linear(
                &checkpoint,
                &format!("{p}.mlp.up_proj"),
                &[text.intermediate_size, hidden],
                false,
                "MLP up projection",
            )?;
            let down = fetch_linear(
                &checkpoint,
                &format!("{p}.mlp.down_proj"),
                &[hidden, text.intermediate_size],
                true,
                "MLP down projection",
            )?;
            let layer_scalar = checkpoint.source(&format!("{p}.layer_scalar"), &[1])?;
            let ple_input_gate = fetch_linear(
                &checkpoint,
                &format!("{p}.ple_block.per_layer_input_gate"),
                &[ple_width, hidden],
                true,
                "PLE input gate",
            )?;
            let ple_projection = fetch_linear(
                &checkpoint,
                &format!("{p}.ple_block.per_layer_projection"),
                &[hidden, ple_width],
                true,
                "PLE output projection",
            )?;
            let ple_post_norm = checkpoint.source(
                &format!("{p}.ple_block.post_per_layer_input_norm.weight"),
                &[hidden],
            )?;
            layers.push(Gemma2LayerWeights {
                input_norm,
                post_attention_norm,
                pre_feedforward_norm,
                post_feedforward_norm,
                q_norm,
                k_norm,
                q,
                k,
                v,
                o,
                gate,
                up,
                down,
                layer_scalar,
                ple_input_gate,
                ple_projection,
                ple_post_norm,
            });
        }

        let plan = Gemma2ExecutionPlan::prepare(text)?;
        Ok(Self {
            config,
            plan,
            weights: Gemma2Weights {
                embed,
                embedding_projection,
                final_norm,
                ple,
                layers,
            },
        })
    }
}

#[derive(Debug, Deserialize)]
struct RawGemma2Config {
    model_type: String,
    architectures: Option<Vec<String>>,
    text_config: Option<Value>,
    vision_config: Option<Value>,
    audio_config: Option<Value>,
    image_token_id: Option<u32>,
    video_token_id: Option<u32>,
    audio_token_id: Option<u32>,
    boi_token_id: Option<u32>,
    eoi_token_id: Option<u32>,
    boa_token_id: Option<u32>,
    eoa_token_index: Option<u32>,
    quantization_config: Option<Value>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

#[derive(Debug, Deserialize)]
struct RawTextConfig {
    model_type: Option<String>,
    hidden_size: Option<usize>,
    intermediate_size: Option<usize>,
    num_hidden_layers: Option<usize>,
    num_attention_heads: Option<usize>,
    num_key_value_heads: Option<usize>,
    head_dim: Option<usize>,
    vocab_size: Option<usize>,
    max_position_embeddings: Option<usize>,
    rms_norm_eps: Option<f64>,
    hidden_size_per_layer_input: Option<usize>,
    embedding_dim: Option<usize>,
    sliding_window: Option<usize>,
    layer_types: Option<Vec<String>>,
    per_layer_config: Option<BTreeMap<String, RawLayerOverride>>,
    rope_parameters: Option<Value>,
    hidden_activation: Option<String>,
    attention_bias: Option<bool>,
    attention_dropout: Option<f64>,
    global_head_dim: Option<usize>,
    num_global_key_value_heads: Option<usize>,
    attention_k_eq_v: Option<bool>,
    num_kv_shared_layers: Option<i64>,
    partial_rotary_factor: Option<f64>,
    quantization_config: Option<Value>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

#[derive(Debug, Deserialize)]
struct RawLayerOverride {
    head_dim: Option<usize>,
    num_key_value_heads: Option<usize>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

#[derive(Clone, Copy, Debug, Default)]
struct LayerOverride {
    head_dim: Option<usize>,
    kv_heads: Option<usize>,
}

fn parse_text_config(value: Option<&Value>) -> Result<Gemma2TextConfig, BackendError> {
    let value = value.ok_or_else(|| start("missing nested text_config"))?;
    let raw: RawTextConfig =
        serde_json::from_value(value.clone()).map_err(|error| start(error.to_string()))?;
    if raw
        .model_type
        .as_deref()
        .is_some_and(|kind| kind != "embedding_gemma2_text")
    {
        return Err(start(format!(
            "unsupported gemma2 text model_type {}",
            raw.model_type.as_deref().unwrap_or_default()
        )));
    }
    reject_text_modifiers(&raw)?;

    let hidden_size = required_usize(raw.hidden_size, "hidden_size")?;
    let intermediate_size = required_usize(raw.intermediate_size, "intermediate_size")?;
    let num_hidden_layers = required_usize(raw.num_hidden_layers, "num_hidden_layers")?;
    let num_attention_heads = required_usize(raw.num_attention_heads, "num_attention_heads")?;
    let vocab_size = required_usize(raw.vocab_size, "vocab_size")?;
    let ple_width = required_usize(
        raw.hidden_size_per_layer_input,
        "hidden_size_per_layer_input",
    )?;
    let embedding_dim = required_usize(raw.embedding_dim, "embedding_dim")?;
    for (name, value) in [
        ("hidden_size", hidden_size),
        ("intermediate_size", intermediate_size),
        ("num_hidden_layers", num_hidden_layers),
        ("num_attention_heads", num_attention_heads),
        ("vocab_size", vocab_size),
        ("hidden_size_per_layer_input", ple_width),
        ("embedding_dim", embedding_dim),
    ] {
        if value == 0 {
            return Err(start(format!("config {name} must be nonzero")));
        }
    }
    let ple_rows = num_hidden_layers
        .checked_mul(ple_width)
        .ok_or_else(|| start("PLE projection row count overflow"))?;
    elements(&[ple_rows, hidden_size])
        .map_err(|error| start(format!("invalid PLE model projection: {error}")))?;
    elements(&[hidden_size, ple_width])
        .map_err(|error| start(format!("invalid PLE layer projection: {error}")))?;

    let default_head_dim = match raw.head_dim {
        Some(0) => return Err(start("config head_dim must be nonzero")),
        Some(head_dim) => head_dim,
        None => {
            if hidden_size % num_attention_heads != 0 {
                return Err(start(
                    "hidden_size must divide evenly by num_attention_heads when head_dim is omitted",
                ));
            }
            hidden_size / num_attention_heads
        }
    };
    if default_head_dim == 0 || default_head_dim % 2 != 0 {
        return Err(start("rotary head_dim must be positive and even"));
    }
    let default_kv_heads = raw.num_key_value_heads.unwrap_or(num_attention_heads);
    if default_kv_heads == 0 || num_attention_heads % default_kv_heads != 0 {
        return Err(start(
            "num_attention_heads must divide evenly by num_key_value_heads",
        ));
    }
    let default_q_width = num_attention_heads
        .checked_mul(default_head_dim)
        .ok_or_else(|| start("query width overflow"))?;
    let default_kv_width = default_kv_heads
        .checked_mul(default_head_dim)
        .ok_or_else(|| start("key/value width overflow"))?;
    validate_products(
        hidden_size,
        intermediate_size,
        vocab_size,
        default_q_width,
        default_kv_width,
        default_head_dim,
    )?;

    let max_position_embeddings = raw
        .max_position_embeddings
        .unwrap_or(DEFAULT_MAX_POSITION_EMBEDDINGS);
    if max_position_embeddings == 0 || u32::try_from(max_position_embeddings).is_err() {
        return Err(start(
            "max_position_embeddings must be positive and fit CUDA position IDs",
        ));
    }
    let rms_norm_eps = positive_scalar(
        raw.rms_norm_eps.unwrap_or(DEFAULT_RMS_NORM_EPS),
        "rms_norm_eps",
    )?;
    let sliding_window = raw.sliding_window.unwrap_or(DEFAULT_SLIDING_WINDOW);
    if sliding_window == 0 || sliding_window > i32::MAX as usize {
        return Err(start(
            "sliding_window must be positive and fit i32 policy width",
        ));
    }
    let window = sliding_window
        .checked_mul(2)
        .ok_or_else(|| start("bidirectional sliding window overflow"))?;
    let window_radius =
        i32::try_from(window / 2).map_err(|_| start("sliding window exceeds i32 policy width"))?;
    let (rope_sliding_theta, rope_full_theta) = parse_rope(raw.rope_parameters.as_ref())?;
    let activation = parse_activation(raw.hidden_activation.as_deref())?;
    let overrides = parse_layer_overrides(raw.per_layer_config.as_ref(), num_hidden_layers)?;
    let kinds = parse_layer_types(raw.layer_types.as_deref(), num_hidden_layers)?;
    let shared_layers = match raw.num_kv_shared_layers.unwrap_or(0) {
        count if count < 0 => return Err(start("num_kv_shared_layers must be nonnegative")),
        count => usize::try_from(count).map_err(|_| start("num_kv_shared_layers overflow"))?,
    };
    if shared_layers > num_hidden_layers {
        return Err(start("num_kv_shared_layers exceeds num_hidden_layers"));
    }
    let sharing = shared_kv_layout(&kinds, shared_layers)?;

    let mut layers = Vec::new();
    layers.try_reserve_exact(num_hidden_layers).map_err(start)?;
    for (index, kind) in kinds.iter().copied().enumerate() {
        let override_config = overrides.get(&index).copied().unwrap_or_default();
        let head_dim = override_config.head_dim.unwrap_or(default_head_dim);
        let kv_heads = override_config.kv_heads.unwrap_or(default_kv_heads);
        if head_dim == 0 || head_dim % 2 != 0 {
            return Err(start(format!(
                "layer {index} head_dim must be positive and even"
            )));
        }
        if kv_heads == 0 || num_attention_heads % kv_heads != 0 {
            return Err(start(format!(
                "layer {index} num_key_value_heads must be nonzero and divide attention heads"
            )));
        }
        let q_width = num_attention_heads
            .checked_mul(head_dim)
            .ok_or_else(|| start(format!("layer {index} query width overflow")))?;
        let kv_width = kv_heads
            .checked_mul(head_dim)
            .ok_or_else(|| start(format!("layer {index} key/value width overflow")))?;
        let (window_left, window_right) = match kind {
            Gemma2AttentionKind::Sliding => (window_radius, window_radius),
            Gemma2AttentionKind::Full => (-1, -1),
        };
        let rope_theta = match kind {
            Gemma2AttentionKind::Sliding => rope_sliding_theta,
            Gemma2AttentionKind::Full => rope_full_theta,
        };
        let owner = sharing
            .get(index)
            .copied()
            .ok_or_else(|| start("shared KV layout is incomplete"))?;
        layers.push(Gemma2LayerConfig {
            index,
            kind,
            head_dim,
            kv_heads,
            q_width,
            kv_width,
            window_left,
            window_right,
            rope_theta,
            shared: owner.shared,
            store_kv: owner.store,
            producer: owner.producer,
        });
    }

    Ok(Gemma2TextConfig {
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        vocab_size,
        max_position_embeddings,
        rms_norm_eps,
        ple_width,
        embedding_dim,
        sliding_window,
        rope_sliding_theta,
        rope_full_theta,
        activation,
        layers,
    })
}

fn parse_media_config(raw: &RawGemma2Config) -> Result<Gemma2Media, BackendError> {
    if raw.model_type != "embedding_gemma2" {
        let reason = match raw.model_type.as_str() {
            "gemma4" | "gemma4_text" => "Gemma4 is a separate family foundation",
            "gemma3" | "gemma3_text" => "Gemma3 is a separate family foundation",
            _ => "only gemma2 containers are supported",
        };
        return Err(start(format!(
            "unsupported model_type {}: {reason}",
            raw.model_type
        )));
    }
    if let Some(architectures) = raw.architectures.as_deref() {
        if architectures.is_empty()
            || architectures
                .iter()
                .any(|architecture| architecture != "EmbeddingGemma2Model")
        {
            return Err(start(format!(
                "unsupported gemma2 architectures {architectures:?}; expected EmbeddingGemma2Model"
            )));
        }
    }
    if non_null(&raw.quantization_config) {
        return Err(start("quantized gemma2 checkpoints are not supported"));
    }
    reject_top_level_modifiers(raw)?;

    let vision_layers = tower_layers(raw.vision_config.as_ref(), "vision_config")?;
    let audio_layers = tower_layers(raw.audio_config.as_ref(), "audio_config")?;
    if vision_layers > 0 {
        validate_vision_config(raw.vision_config.as_ref(), vision_layers)?;
    }
    if audio_layers > 0 {
        validate_audio_config(raw.audio_config.as_ref())?;
    }

    let tokens = [
        raw.image_token_id,
        raw.video_token_id,
        raw.audio_token_id,
        raw.boi_token_id,
        raw.eoi_token_id,
        raw.boa_token_id,
        raw.eoa_token_index,
    ];
    let present = tokens.iter().filter(|token| token.is_some()).count();
    let tokens = if present == 0 {
        None
    } else if present == tokens.len() {
        let [image, video, audio, begin_image, end_image, begin_audio, end_audio] = tokens;
        let tokens = Gemma2MediaTokens {
            image: image.ok_or_else(|| start("missing image_token_id"))?,
            video: video.ok_or_else(|| start("missing video_token_id"))?,
            audio: audio.ok_or_else(|| start("missing audio_token_id"))?,
            begin_image: begin_image.ok_or_else(|| start("missing boi_token_id"))?,
            end_image: end_image.ok_or_else(|| start("missing eoi_token_id"))?,
            begin_audio: begin_audio.ok_or_else(|| start("missing boa_token_id"))?,
            end_audio: end_audio.ok_or_else(|| start("missing eoa_token_index"))?,
        };
        let mut sorted = tokens.all().to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.len() != tokens.all().len() {
            return Err(start("gemma2 media token ids must be distinct"));
        }
        Some(tokens)
    } else {
        return Err(start(
            "gemma2 media token ids must be present together or absent",
        ));
    };
    if (vision_layers > 0 || audio_layers > 0) && tokens.is_none() {
        return Err(start(
            "media towers require the complete gemma2 media token id set",
        ));
    }

    Ok(Gemma2Media {
        tokens,
        vision_layers,
        audio_layers,
        processor: None,
        vision_config: raw.vision_config.clone(),
        audio_config: raw.audio_config.clone(),
    })
}

fn parse_processor(path: &Path) -> Result<Option<Gemma2Processor>, BackendError> {
    let file = path.join("processor_config.json");
    let metadata = match std::fs::metadata(&file) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(start(error)),
    };
    if !metadata.is_file() {
        return Err(start("processor_config.json must be a regular file"));
    }
    let value: Value = read_json(&file)?;
    let image = &value["image_processor"];
    let video = &value["video_processor"];
    let audio = &value["feature_extractor"];
    let image_max_soft_tokens = validate_image_processor(image, "image_processor")?;
    let video_max_soft_tokens = validate_image_processor(video, "video_processor")?;
    if !matches!(video_max_soft_tokens, 70 | 140 | 280 | 560 | 1120) {
        return Err(start("unsupported video_processor max_soft_tokens"));
    }
    if !matches!(
        &video["overflow_strategy"],
        Value::String(strategy) if strategy == "uniform"
    ) || !json_bool(video, "add_timestamps", false)
        || !json_bool(video, "do_sample_frames", true)
    {
        return Err(start("unsupported video_processor sampling contract"));
    }
    let fps = video["fps"]
        .as_f64()
        .filter(|fps| fps.is_finite() && *fps > 0.0)
        .ok_or_else(|| start("invalid video_processor fps"))?;
    let max_frames = video["max_frames"]
        .as_u64()
        .filter(|frames| (1..=32).contains(frames))
        .ok_or_else(|| start("invalid video_processor max_frames"))?;
    let max_frames = usize::try_from(max_frames).map_err(|_| start("max_frames overflow"))?;

    let sampling_rate = json_usize(audio, "sampling_rate")?;
    let feature_size = json_usize(audio, "feature_size")?;
    if sampling_rate != 16_000
        || feature_size != 128
        || json_usize(audio, "frame_length")? != 320
        || json_usize(audio, "hop_length")? != 160
        || json_usize(audio, "fft_length")? != 512
        || json_f64(audio, "preemphasis")? != 0.0
        || json_f64(audio, "dither")? != 0.0
        || json_f64(audio, "input_scale_factor")? != 1.0
        || json_f64(audio, "min_frequency")? != 0.0
        || json_f64(audio, "max_frequency")? != 8_000.0
        || json_f64(audio, "mel_floor")? != 0.001
        || !audio["per_bin_mean"].is_null()
        || !audio["per_bin_stddev"].is_null()
        || !json_bool(audio, "fft_overdrive", false)
    {
        return Err(start(
            "unsupported Gemma4 audio feature extractor configuration",
        ));
    }
    Ok(Some(Gemma2Processor {
        image_max_soft_tokens,
        video_max_frames: max_frames,
        video_fps: fps,
        audio_sampling_rate: sampling_rate,
        audio_feature_size: feature_size,
    }))
}

fn validate_image_processor(value: &Value, name: &str) -> Result<usize, BackendError> {
    let max_soft_tokens = value["max_soft_tokens"]
        .as_u64()
        .and_then(|tokens| usize::try_from(tokens).ok())
        .ok_or_else(|| start(format!("invalid {name} max_soft_tokens")))?;
    if value["patch_size"].as_u64() != Some(16)
        || value["pooling_kernel_size"].as_u64() != Some(3)
        || !IMAGE_SOFT_TOKENS.contains(&max_soft_tokens)
        || value["resample"].as_u64() != Some(3)
        || !json_bool(value, "do_resize", true)
        || !json_bool(value, "do_rescale", true)
        || !json_bool(value, "do_normalize", false)
        || !json_bool(value, "do_convert_rgb", true)
        || value["rescale_factor"].as_f64() != Some(1.0 / 255.0)
    {
        return Err(start(format!("unsupported {name} image contract")));
    }
    Ok(max_soft_tokens)
}

fn validate_vision_config(value: Option<&Value>, layers: usize) -> Result<(), BackendError> {
    let value = value.ok_or_else(|| start("missing vision_config"))?;
    let hidden = value["hidden_size"]
        .as_u64()
        .and_then(|hidden| usize::try_from(hidden).ok())
        .filter(|hidden| *hidden > 0)
        .ok_or_else(|| start("invalid vision_config hidden_size"))?;
    let heads = value["num_attention_heads"]
        .as_u64()
        .and_then(|heads| usize::try_from(heads).ok())
        .filter(|heads| *heads > 0)
        .ok_or_else(|| start("invalid vision_config num_attention_heads"))?;
    let head_dim = value["head_dim"]
        .as_u64()
        .and_then(|dim| usize::try_from(dim).ok())
        .unwrap_or(hidden / heads);
    if head_dim == 0 || head_dim % 4 != 0 || layers == 0 {
        return Err(start("invalid vision_config rotary head_dim"));
    }
    let theta = value["rope_parameters"]["rope_theta"]
        .as_f64()
        .unwrap_or(100.0);
    if !theta.is_finite() || theta <= 0.0 {
        return Err(start("invalid vision_config rope theta"));
    }
    Ok(())
}

fn validate_audio_config(value: Option<&Value>) -> Result<(), BackendError> {
    let value = value.ok_or_else(|| start("missing audio_config"))?;
    value["hidden_size"]
        .as_u64()
        .and_then(|hidden| usize::try_from(hidden).ok())
        .filter(|hidden| *hidden > 0)
        .ok_or_else(|| start("invalid audio_config hidden_size"))?;
    let chunk = value["attention_chunk_size"]
        .as_u64()
        .ok_or_else(|| start("invalid audio_config attention_chunk_size"))?;
    let left = value["attention_context_left"]
        .as_u64()
        .ok_or_else(|| start("invalid audio_config attention_context_left"))?;
    let right = value["attention_context_right"]
        .as_u64()
        .ok_or_else(|| start("invalid audio_config attention_context_right"))?;
    if chunk == 0 || left == 0 || left > i64::MAX as u64 {
        return Err(start("invalid audio_config relative attention geometry"));
    }
    let past = left - 1;
    if chunk > i64::MAX as u64 - past || right > i64::MAX as u64 - chunk - past {
        return Err(start("audio_config relative attention geometry overflow"));
    }
    Ok(())
}

fn tower_layers(value: Option<&Value>, name: &str) -> Result<usize, BackendError> {
    let Some(value) = value else {
        return Ok(0);
    };
    if value.is_null() {
        return Ok(0);
    }
    if !value.is_object() {
        return Err(start(format!("{name} must be a JSON object")));
    }
    let count = value["num_hidden_layers"].as_u64().unwrap_or(0);
    usize::try_from(count).map_err(|_| start(format!("{name} layer count overflow")))
}

#[derive(Clone, Copy, Debug, Default)]
struct SharedKvLayout {
    shared: bool,
    store: bool,
    producer: Option<usize>,
}

fn shared_kv_layout(
    kinds: &[Gemma2AttentionKind],
    shared_layers: usize,
) -> Result<Vec<SharedKvLayout>, BackendError> {
    let first_shared = kinds
        .len()
        .checked_sub(shared_layers)
        .ok_or_else(|| start("num_kv_shared_layers exceeds num_hidden_layers"))?;
    let mut layout = vec![SharedKvLayout::default(); kinds.len()];
    if shared_layers == 0 {
        return Ok(layout);
    }
    for (index, kind) in kinds.iter().copied().enumerate().take(first_shared) {
        let later_same_kind = kinds
            .iter()
            .skip(index + 1)
            .take(first_shared.saturating_sub(index + 1))
            .any(|candidate| *candidate == kind);
        if let Some(entry) = layout.get_mut(index) {
            entry.store = !later_same_kind;
        }
    }
    let mut producers = [None, None];
    for (index, kind) in kinds.iter().copied().enumerate().take(first_shared) {
        let slot = match kind {
            Gemma2AttentionKind::Sliding => 0,
            Gemma2AttentionKind::Full => 1,
        };
        if layout.get(index).is_some_and(|entry| entry.store) {
            producers[slot] = Some(index);
        }
    }
    for (index, kind) in kinds.iter().copied().enumerate().skip(first_shared) {
        let slot = match kind {
            Gemma2AttentionKind::Sliding => 0,
            Gemma2AttentionKind::Full => 1,
        };
        let producer = producers[slot].ok_or_else(|| {
            start(format!(
                "shared KV layer {index} has no preceding {kind:?} producer"
            ))
        })?;
        if let Some(entry) = layout.get_mut(index) {
            entry.shared = true;
            entry.producer = Some(producer);
        }
    }
    Ok(layout)
}

fn parse_layer_types(
    value: Option<&[String]>,
    layers: usize,
) -> Result<Vec<Gemma2AttentionKind>, BackendError> {
    let Some(kinds) = value else {
        return Ok(vec![Gemma2AttentionKind::Full; layers]);
    };
    if kinds.len() != layers {
        return Err(start(format!(
            "layer_types has {} entries for {layers} layers",
            kinds.len()
        )));
    }
    kinds
        .iter()
        .map(|kind| match kind.as_str() {
            "sliding_attention" => Ok(Gemma2AttentionKind::Sliding),
            "full_attention" => Ok(Gemma2AttentionKind::Full),
            other => Err(start(format!("unsupported attention type {other}"))),
        })
        .collect()
}

fn parse_layer_overrides(
    value: Option<&BTreeMap<String, RawLayerOverride>>,
    layers: usize,
) -> Result<BTreeMap<usize, LayerOverride>, BackendError> {
    let mut overrides = BTreeMap::new();
    let Some(value) = value else {
        return Ok(overrides);
    };
    for (spelling, raw) in value {
        let Some(index) = parse_layer_index(spelling) else {
            continue;
        };
        if index >= layers {
            continue;
        }
        if raw.extra.values().any(active_json) {
            return Err(start(format!(
                "unsupported per_layer_config field for layer {index}"
            )));
        }
        if raw.head_dim == Some(0) || raw.num_key_value_heads == Some(0) {
            return Err(start(format!(
                "per_layer_config layer {index} width must be nonzero"
            )));
        }
        let entry = overrides.entry(index).or_default();
        merge_override(entry, raw, index)?;
    }
    Ok(overrides)
}

fn parse_layer_index(value: &str) -> Option<usize> {
    let digits = value.strip_prefix('+').unwrap_or(value);
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

fn merge_override(
    entry: &mut LayerOverride,
    raw: &RawLayerOverride,
    index: usize,
) -> Result<(), BackendError> {
    merge_override_value(&mut entry.head_dim, raw.head_dim, index, "head_dim")?;
    merge_override_value(
        &mut entry.kv_heads,
        raw.num_key_value_heads,
        index,
        "num_key_value_heads",
    )
}

fn merge_override_value(
    target: &mut Option<usize>,
    value: Option<usize>,
    index: usize,
    name: &str,
) -> Result<(), BackendError> {
    let Some(value) = value else {
        return Ok(());
    };
    if target.is_some_and(|existing| existing != value) {
        return Err(start(format!(
            "conflicting per_layer_config aliases for layer {index} {name}"
        )));
    }
    *target = Some(value);
    Ok(())
}

fn parse_rope(value: Option<&Value>) -> Result<(f64, f64), BackendError> {
    let Some(value) = value else {
        return Ok((DEFAULT_ROPE_THETA, DEFAULT_ROPE_THETA));
    };
    if value.is_null() {
        return Ok((DEFAULT_ROPE_THETA, DEFAULT_ROPE_THETA));
    }
    let object = value
        .as_object()
        .ok_or_else(|| start("rope_parameters must be a JSON object"))?;
    let mut sliding = DEFAULT_ROPE_THETA;
    let mut full = DEFAULT_ROPE_THETA;
    for (kind, value) in object {
        let theta = match kind.as_str() {
            "sliding_attention" => &mut sliding,
            "full_attention" => &mut full,
            other => {
                return Err(start(format!(
                    "unsupported rope_parameters attention kind {other}"
                )))
            }
        };
        let params = value
            .as_object()
            .ok_or_else(|| start("rope_parameters kind must be a JSON object"))?;
        for (name, parameter) in params {
            match name.as_str() {
                "rope_theta" => {
                    *theta = parameter.as_f64().ok_or_else(|| {
                        start("rope_parameters rope_theta must be a finite number")
                    })?;
                }
                "rope_type" | "type" => {
                    if parameter.as_str() != Some("default") {
                        return Err(start("only default gemma2 RoPE is supported"));
                    }
                }
                "partial_rotary_factor" => {
                    if parameter.as_f64() != Some(1.0) {
                        return Err(start(
                            "partial rotary gemma2 configurations are not supported",
                        ));
                    }
                }
                other => {
                    return Err(start(format!(
                        "unsupported rope_parameters modifier {other}"
                    )))
                }
            }
        }
        if !theta.is_finite() || *theta <= 0.0 {
            return Err(start("rope theta must be finite and positive"));
        }
    }
    Ok((sliding, full))
}

fn parse_activation(value: Option<&str>) -> Result<Gemma2Activation, BackendError> {
    match value.unwrap_or("gelu_pytorch_tanh") {
        "gelu_pytorch_tanh" | "gelu_new" => Ok(Gemma2Activation::GeluTanh),
        "gelu" => Ok(Gemma2Activation::Gelu),
        "silu" | "swish" => Ok(Gemma2Activation::Silu),
        "relu" => Ok(Gemma2Activation::Relu),
        other => Err(start(format!("unsupported activation {other}"))),
    }
}

fn reject_top_level_modifiers(raw: &RawGemma2Config) -> Result<(), BackendError> {
    for (key, value) in &raw.extra {
        if !active_json(value) {
            continue;
        }
        let lower = key.to_ascii_lowercase();
        if lower.contains("quant")
            || lower.contains("fp8")
            || lower.contains("expert")
            || lower.contains("moe")
            || lower.contains("router")
            || lower.contains("softcap")
        {
            return Err(start(format!("unsupported gemma2 config modifier {key}")));
        }
    }
    Ok(())
}

fn reject_text_modifiers(raw: &RawTextConfig) -> Result<(), BackendError> {
    if raw.attention_bias == Some(true) {
        return Err(start("attention_bias checkpoints are not supported"));
    }
    if raw.attention_dropout.is_some_and(|dropout| dropout != 0.0) {
        return Err(start("attention dropout must be zero for inference"));
    }
    if raw.global_head_dim.is_some_and(|dim| dim != 0) {
        return Err(start(
            "global_head_dim is ignored by Native gemma2; use per_layer_config head_dim",
        ));
    }
    if raw
        .num_global_key_value_heads
        .is_some_and(|heads| heads != 0)
    {
        return Err(start(
            "num_global_key_value_heads is ignored by Native gemma2",
        ));
    }
    if raw.attention_k_eq_v == Some(true) {
        return Err(start("attention_k_eq_v is not supported for gemma2"));
    }
    if raw
        .partial_rotary_factor
        .is_some_and(|factor| factor != 1.0)
    {
        return Err(start("partial rotary gemma2 is not supported"));
    }
    if non_null(&raw.quantization_config) {
        return Err(start("quantized gemma2 checkpoints are not supported"));
    }
    if raw
        .extra
        .get("tie_word_embeddings")
        .is_some_and(|value| value.as_bool() == Some(true))
    {
        return Err(start(
            "tied output embeddings are not supported; Native uses the separate embedding_projection weight",
        ));
    }
    for (key, value) in &raw.extra {
        if !active_json(value) {
            continue;
        }
        let lower = key.to_ascii_lowercase();
        let unsupported = lower.contains("quant")
            || lower.contains("expert")
            || lower.contains("moe")
            || lower.contains("router")
            || lower.contains("softcap")
            || lower.contains("rope_scaling")
            || lower.contains("sliding_window_pattern")
            || lower.contains("multimodal")
            || lower.contains("multi_modal")
            || lower.contains("vision")
            || lower.contains("audio")
            || lower.contains("image_")
            || lower.contains("video_")
            || lower.contains("mm_");
        if unsupported {
            return Err(start(format!(
                "unsupported gemma2 text config modifier {key}"
            )));
        }
    }
    Ok(())
}

fn validate_products(
    hidden_size: usize,
    intermediate_size: usize,
    vocab_size: usize,
    q_width: usize,
    kv_width: usize,
    head_dim: usize,
) -> Result<(), BackendError> {
    for (name, dimensions) in [
        ("embed_tokens", [vocab_size, hidden_size]),
        ("q_proj", [q_width, hidden_size]),
        ("k_proj", [kv_width, hidden_size]),
        ("v_proj", [kv_width, hidden_size]),
        ("o_proj", [hidden_size, q_width]),
        ("gate_proj", [intermediate_size, hidden_size]),
        ("up_proj", [intermediate_size, hidden_size]),
        ("down_proj", [hidden_size, intermediate_size]),
    ] {
        elements(&dimensions)
            .map_err(|error| start(format!("invalid {name} dimensions: {error}")))?;
    }
    elements(&[hidden_size]).map_err(|error| start(format!("invalid norm dimensions: {error}")))?;
    elements(&[head_dim]).map_err(|error| start(format!("invalid head_dim: {error}")))?;
    Ok(())
}

fn required_usize(value: Option<usize>, name: &str) -> Result<usize, BackendError> {
    value.ok_or_else(|| start(format!("missing config field {name}")))
}

fn positive_scalar(value: f64, name: &str) -> Result<f64, BackendError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(start(format!("{name} must be finite and positive")));
    }
    Ok(value)
}

fn non_null(value: &Option<Value>) -> bool {
    value.as_ref().is_some_and(|value| !value.is_null())
}

fn active_json(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(active) => *active,
        Value::Number(number) => number.as_f64() != Some(0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(items) => !items.is_empty(),
    }
}

fn json_bool(value: &Value, name: &str, expected: bool) -> bool {
    value[name].as_bool() == Some(expected)
}

fn json_usize(value: &Value, name: &str) -> Result<usize, BackendError> {
    value
        .get(name)
        .and_then(Value::as_u64)
        .and_then(|number| usize::try_from(number).ok())
        .ok_or_else(|| start(format!("{name} must be a nonnegative integer")))
}

fn json_f64(value: &Value, name: &str) -> Result<f64, BackendError> {
    value
        .get(name)
        .and_then(Value::as_f64)
        .filter(|number| number.is_finite())
        .ok_or_else(|| start(format!("{name} must be finite")))
}

fn select_root(checkpoint: &Checkpoint) -> Result<&'static str, BackendError> {
    let mut selected = None;
    let mut anchors = 0usize;
    for name in checkpoint.names() {
        for root in NATIVE_ROOTS {
            if name.strip_prefix(root) == Some("embed_tokens.weight") {
                anchors = anchors
                    .checked_add(1)
                    .ok_or_else(|| start("checkpoint anchor overflow"))?;
                selected = Some(root);
            }
        }
        if name.ends_with("embed_tokens.weight")
            && !NATIVE_ROOTS
                .iter()
                .any(|root| name.strip_prefix(root) == Some("embed_tokens.weight"))
        {
            return Err(start(format!(
                "unsupported gemma2 checkpoint namespace {name}"
            )));
        }
    }
    if anchors == 0 {
        return Err(start("missing weight embed_tokens.weight"));
    }
    if anchors != 1 {
        return Err(start(
            "ambiguous gemma2 checkpoint: multiple embed_tokens.weight roots",
        ));
    }
    let root = selected.ok_or_else(|| start("missing gemma2 text root"))?;
    for name in checkpoint.names() {
        if media_tensor(name) || root_for(name) == root {
            continue;
        }
        let suffix = name.strip_prefix(root_for(name)).unwrap_or(name.as_str());
        if suffix.starts_with("layers.")
            || suffix.ends_with(".norm.weight")
            || suffix == "embed_tokens.weight"
        {
            return Err(start(format!(
                "mixed gemma2 text roots: found {name} alongside the {root:?} root"
            )));
        }
    }
    Ok(root)
}

fn root_for(name: &str) -> &'static str {
    NATIVE_ROOTS
        .iter()
        .copied()
        .find(|root| name.starts_with(*root))
        .unwrap_or("")
}

fn media_tensor(name: &str) -> bool {
    MEDIA_PREFIXES.iter().any(|prefix| name.starts_with(prefix))
}

fn fetch_linear(
    checkpoint: &Checkpoint,
    name: &str,
    weight_shape: &[usize],
    allow_bias: bool,
    role: &str,
) -> Result<DecoderLinear, BackendError> {
    let weight = checkpoint.source(&format!("{name}.weight"), weight_shape)?;
    let bias_name = format!("{name}.bias");
    if !allow_bias && checkpoint.contains(&bias_name) {
        return Err(start(format!(
            "Native {role} ignores {bias_name}; refusing to load it"
        )));
    }
    let rows = weight_shape
        .first()
        .copied()
        .ok_or_else(|| start(format!("empty weight shape for {name}")))?;
    let bias = checkpoint.optional_source(&bias_name, &[rows])?;
    Ok(DecoderLinear { weight, bias })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct TempConfig {
        root: PathBuf,
    }

    impl TempConfig {
        fn new(config: &Value) -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let root = std::env::temp_dir().join(format!(
                "infinity-gemma2-text-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&root).expect("create config fixture directory");
            fs::write(
                root.join("config.json"),
                serde_json::to_vec(config).expect("serialize config fixture"),
            )
            .expect("write config fixture");
            Self { root }
        }

        fn load(&self) -> Result<Gemma2Config, BackendError> {
            Gemma2Config::load(&self.root)
        }
    }

    impl Drop for TempConfig {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    /// Exact saved official `google/embeddinggemma-2` config fixture.
    fn fixture_config() -> Value {
        serde_json::from_str(include_str!(
            "../tests/fixtures/embedding_gemma2/config.json"
        ))
        .expect("parse official config fixture")
    }

    fn config_error(config: &Value) -> String {
        let fixture = TempConfig::new(config);
        match fixture.load() {
            Ok(_) => panic!("invalid gemma2 config fixture was accepted"),
            Err(error) => error.to_string(),
        }
    }

    #[test]
    fn parses_native_text_config_and_plans_embedding_geometry() {
        let fixture = TempConfig::new(&fixture_config());
        let config = fixture.load().expect("parse Gemma2 fixture config");
        assert_eq!(config.output_width(), 768);
        let media = config.media.to_media_config(config.text.hidden_size);
        assert_eq!(media.text_hidden, 512);
        assert_eq!(media.vision_layers, 16);
        assert_eq!(media.audio_layers, 12);
        assert!(media.vision_config.is_some());
        assert!(media.audio_config.is_some());
        assert_eq!(
            media.tokens.map(|tokens| tokens.all()),
            Some([258_880, 258_884, 258_881, 255_999, 258_882, 256_000, 258_883])
        );

        let text = &config.text;
        assert_eq!(text.hidden_size, 512);
        assert_eq!(text.intermediate_size, 2048);
        assert_eq!(text.num_hidden_layers, 24);
        assert_eq!(text.num_attention_heads, 4);
        assert_eq!(text.vocab_size, 262_144);
        assert_eq!(text.max_position_embeddings, 262_144);
        assert_eq!(text.ple_width, 512);
        assert_eq!(text.embedding_dim, 768);
        assert_eq!(text.sliding_window, 512);
        assert_eq!(text.rope_sliding_theta, 10_000.0);
        assert_eq!(text.rope_full_theta, 1_000_000.0);
        assert_eq!(text.activation, Gemma2Activation::GeluTanh);

        let plan = Gemma2ExecutionPlan::prepare(text).expect("prepare text execution plan");
        assert_eq!(plan.ple_projection_rows, 24 * 512);
        assert_eq!(plan.ple_layout, Gemma2PleLayout::TokenLayersWidth);
        assert_eq!(plan.embedding_scale, 512f64.sqrt());
        assert_eq!(plan.ple_scale, 512f64.sqrt().recip());
        assert_eq!(plan.rms_norm_eps, 1e-6);
        assert!(plan.final_projection_before_pooling);
        assert_eq!(plan.layers.len(), 24);

        let layer0 = &plan.layers[0];
        assert_eq!(layer0.kind, Gemma2AttentionKind::Sliding);
        assert_eq!(layer0.head_dim, 256);
        assert_eq!(layer0.q_heads, 4);
        assert_eq!(layer0.kv_heads, 2);
        assert_eq!(layer0.q_width, 1024);
        assert_eq!(layer0.kv_width, 512);
        assert_eq!((layer0.window_left, layer0.window_right), (512, 512));
        assert_eq!(layer0.rope_theta, 10_000.0);
        assert_eq!(layer0.rope_pairs, 128);
        assert_eq!(layer0.attention_scale, 1.0);
        assert_eq!(layer0.kv_owner, Gemma2KvOwner::Produced);
        assert!(layer0.has_ple && layer0.has_layer_scalar);

        for index in [5usize, 11, 17, 23] {
            let layer = &plan.layers[index];
            assert_eq!(layer.kind, Gemma2AttentionKind::Full);
            assert_eq!(layer.head_dim, 512);
            assert_eq!(layer.kv_heads, 1);
            assert_eq!(layer.q_width, 2048);
            assert_eq!(layer.kv_width, 512);
            assert_eq!((layer.window_left, layer.window_right), (-1, -1));
            assert_eq!(layer.rope_theta, 1_000_000.0);
            assert_eq!(layer.rope_pairs, 256);
        }
        assert!(plan
            .layers
            .iter()
            .all(|layer| matches!(layer.kv_owner, Gemma2KvOwner::Produced)));
        assert!(plan.check_route(Gemma2Route::Text).is_ok());
        assert!(plan.check_route(Gemma2Route::Image).is_err());

        let media = &config.media;
        assert_eq!(media.vision_layers, 16);
        assert_eq!(media.audio_layers, 12);
        assert_eq!(
            media.tokens,
            Some(Gemma2MediaTokens {
                image: 258_880,
                video: 258_884,
                audio: 258_881,
                begin_image: 255_999,
                end_image: 258_882,
                begin_audio: 256_000,
                end_audio: 258_883,
            })
        );
        assert!(media.check_route(Gemma2Route::Text).is_ok());
        assert!(media.check_route(Gemma2Route::Image).is_ok());
        assert!(media.check_route(Gemma2Route::Audio).is_ok());
        assert!(media.check_route(Gemma2Route::Video).is_err());
        assert!(media.check_text_token_ids(&[258_880]).is_err());
        assert!(media.check_text_token_ids(&[5, 6, 7]).is_ok());
    }

    #[test]
    fn shared_kv_layout_maps_layers_to_last_same_kind_producer() {
        let mut config = fixture_config();
        config["text_config"]["num_kv_shared_layers"] = json!(2);
        let fixture = TempConfig::new(&config);
        let config = fixture.load().expect("parse shared KV config");
        let layers = &config.text.layers;

        assert!(layers[17].store_kv);
        assert!(layers[21].store_kv);
        assert!(!layers[17].shared && !layers[21].shared);
        for index in [22usize, 23] {
            assert!(layers[index].shared);
            assert!(!layers[index].store_kv);
        }
        assert_eq!(layers[22].producer, Some(21));
        assert_eq!(layers[23].producer, Some(17));
    }

    #[test]
    fn rejects_unsupported_text_modifiers_and_overflow() {
        let mut config = fixture_config();
        config["text_config"]["attention_bias"] = json!(true);
        let error = config_error(&config);
        assert!(error.contains("attention_bias"), "{error}");

        let mut config = fixture_config();
        config["text_config"]["tie_word_embeddings"] = json!(true);
        let error = config_error(&config);
        assert!(error.contains("tied output embeddings"), "{error}");

        let mut config = fixture_config();
        config["text_config"]["layer_types"][0] = json!("global_attention");
        let error = config_error(&config);
        assert!(error.contains("unsupported attention type"), "{error}");

        let mut config = fixture_config();
        config["model_type"] = json!("gemma4");
        let error = config_error(&config);
        assert!(error.contains("unsupported model_type gemma4"), "{error}");

        let mut config = fixture_config();
        config["text_config"]["num_hidden_layers"] = json!(u64::MAX);
        config["text_config"]["hidden_size_per_layer_input"] = json!(2);
        let error = config_error(&config);
        assert!(
            error.contains("PLE projection row count overflow"),
            "{error}"
        );

        let mut config = fixture_config();
        config["text_config"]["num_attention_heads"] = json!(u64::MAX);
        config["text_config"]["head_dim"] = json!(2);
        config["text_config"]["num_key_value_heads"] = json!(u64::MAX);
        let error = config_error(&config);
        assert!(error.contains("query width overflow"), "{error}");
    }
}
