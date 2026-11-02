// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Concrete Gemma4 E2B vision tower and multimodal text injection.
//!
//! The published Native `gemma_models.cpp` vision path is the structural
//! reference: FP32 pixel scaling, a bias-free patch projection, 2-D position
//! embeddings, per-head Q/K RMS norm plus unit V norm, two-axis rotary
//! positions, packed bidirectional attention, spatial pooling with a
//! `sqrt(hidden)` scale, a final unit RMS norm, and the
//! `embed_vision.embedding_projection` GEMM.
//!
//! The released E2B checkpoint also carries trained clipped-linears bounds.
//! Native currently rejects `vision_config.use_clipped_linears=true`, so this
//! leaf implements the trained contract (input and output clamps around every
//! clipped projection) and records the Native rejection as the explicitly
//! unsupported cold geometry instead of silently dropping trained bounds.
//!
//! This module owns real weight spans, padded uploads, reusable workspaces,
//! GEMM plans, FA4 attention dispatch, OwnedIO staging, and row injection. It
//! does not create a CUDA context, compile device code, or claim GPU
//! qualification by itself; the Root gate executes the compiled release path.
#![allow(dead_code)]

use crate::cublas::Blas;
use crate::fa4::{AttentionBatch, AttentionSpec, DType, Fa4, Scale, Window};
use crate::math::LinearPlan;
use crate::model::{Checkpoint, WeightSource};
use crate::staging::HostIoLease;
use crate::tensor::Bf16;
use crate::{arg, blocks, infer, launch, launch_threads, product, start, u32_size};
use cuda_core::{Context, DeviceBuffer, Function, Stream};
use infinity_v3_backend_core::BackendError;
use serde_json::Value;
use std::collections::BTreeSet;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{atomic::Ordering, Arc};

const VISION_PATCH_SIZE: usize = 16;
const VISION_POOL: usize = 3;
const VISION_HEAD_DIM: usize = 64;
const VISION_PADDED_HEADS: usize = 16;
const VISION_ROPE_PAIRS: usize = VISION_HEAD_DIM / 4;
const VISION_MAX_LAYERS: usize = 64;
const VISION_MAX_HEADS: usize = 16;
const VISION_WORKSPACE_LIMIT: usize = 2 * 1024 * 1024 * 1024;
const VISION_ROPE_POSITION_LIMIT: usize = 1 << 20;

const VISION_KERNEL_EXPORTS: [&str; 11] = [
    "infinity_gemma4_norm_bf16_f32scale",
    "infinity_gemma4_residual_norm_bf16_f32scale",
    "infinity_gemma4_unit_norm_bf16",
    "infinity_gemma4_gated_gelu_bf16",
    "infinity_gemma4_vision_clip_rows_bf16",
    "infinity_gemma4_vision_position_add_bf16",
    "infinity_gemma4_vision_rope_tables_bf16",
    "infinity_gemma4_vision_rope2d_bf16",
    "infinity_gemma4_vision_pool_bf16",
    "infinity_gemma4_vision_pool_standardize_rows_bf16",
    "infinity_bf16_to_f32",
];

const VISION_SCATTER_EXPORT: &str = "infinity_gemma4_vision_scatter_rows_bf16";

/// Cold, typed rejection. No variant implies that device work was submitted.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum Gemma4VisionUnsupported {
    Geometry(String),
    Dtype(String),
    Clipped(String),
    Activation(String),
    Workspace(String),
    Feature(&'static str),
}

impl fmt::Display for Gemma4VisionUnsupported {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Geometry(error) => write!(formatter, "Gemma4 vision geometry: {error}"),
            Self::Dtype(error) => write!(formatter, "Gemma4 vision dtype: {error}"),
            Self::Clipped(error) => write!(formatter, "Gemma4 vision clipped linears: {error}"),
            Self::Activation(error) => write!(formatter, "Gemma4 vision activation: {error}"),
            Self::Workspace(error) => write!(formatter, "Gemma4 vision workspace: {error}"),
            Self::Feature(name) => write!(formatter, "Gemma4 vision feature is pending: {name}"),
        }
    }
}

impl std::error::Error for Gemma4VisionUnsupported {}

impl From<Gemma4VisionUnsupported> for BackendError {
    fn from(error: Gemma4VisionUnsupported) -> Self {
        start(error)
    }
}

fn unsupported_geometry(error: impl Into<String>) -> Gemma4VisionUnsupported {
    Gemma4VisionUnsupported::Geometry(error.into())
}

fn required_usize(value: Option<&Value>, name: &str) -> Result<usize, Gemma4VisionUnsupported> {
    value
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .filter(|value| *value > 0)
        .ok_or_else(|| unsupported_geometry(format!("{name} must be a positive integer")))
}

fn required_f64(value: Option<&Value>, name: &str) -> Result<f64, Gemma4VisionUnsupported> {
    let value = value
        .and_then(Value::as_f64)
        .ok_or_else(|| unsupported_geometry(format!("{name} must be a number")))?;
    if !value.is_finite() {
        return Err(unsupported_geometry(format!("{name} must be finite")));
    }
    Ok(value)
}

fn required_bool(value: Option<&Value>, name: &str) -> Result<bool, Gemma4VisionUnsupported> {
    value
        .and_then(Value::as_bool)
        .ok_or_else(|| unsupported_geometry(format!("{name} must be a boolean")))
}

/// Exact E2B vision geometry retained from the released config. The carefully
/// bounded generic form allows other width/head/layer counts only when every
/// stage remains inside the checked 16-head padded FA4 `bert` profile.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Gemma4VisionConfig {
    pub(crate) hidden: usize,
    pub(crate) intermediate: usize,
    pub(crate) layers: usize,
    pub(crate) heads: usize,
    pub(crate) kv_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) patch: usize,
    pub(crate) pool: usize,
    pub(crate) eps: f32,
    pub(crate) theta: f64,
    pub(crate) position_size: usize,
    pub(crate) text_hidden: usize,
    pub(crate) use_clipped_linears: bool,
    pub(crate) standardize: bool,
}

impl Gemma4VisionConfig {
    pub(crate) fn load(root: &Path) -> Result<Self, BackendError> {
        let raw = std::fs::read(root.join("config.json")).map_err(start)?;
        let value: Value = serde_json::from_slice(&raw).map_err(start)?;
        Self::from_model_json(&value).map_err(Into::into)
    }

    pub(crate) fn from_model_json(value: &Value) -> Result<Self, Gemma4VisionUnsupported> {
        let vision = &value["vision_config"];
        if !vision.is_object() {
            return Err(unsupported_geometry("vision_config is missing"));
        }
        let text = &value["text_config"];
        let hidden = required_usize(vision.get("hidden_size"), "hidden_size")?;
        let heads = required_usize(vision.get("num_attention_heads"), "num_attention_heads")?;
        let kv_heads = vision
            .get("num_key_value_heads")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(heads);
        let head_dim = vision
            .get("head_dim")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or_else(|| hidden / heads.max(1));
        let patch = required_usize(vision.get("patch_size"), "patch_size")?;
        let pool = required_usize(vision.get("pooling_kernel_size"), "pooling_kernel_size")?;
        let layers = required_usize(vision.get("num_hidden_layers"), "num_hidden_layers")?;
        let intermediate = required_usize(vision.get("intermediate_size"), "intermediate_size")?;
        let position_size = required_usize(
            vision.get("position_embedding_size"),
            "position_embedding_size",
        )?;
        let text_hidden = required_usize(text.get("hidden_size"), "text_config.hidden_size")?;
        let eps = required_f64(vision.get("rms_norm_eps"), "rms_norm_eps")?;
        let rope = &vision["rope_parameters"];
        let rope_type = rope
            .get("rope_type")
            .and_then(Value::as_str)
            .ok_or_else(|| unsupported_geometry("rope_parameters.rope_type is missing"))?;
        if rope_type != "default" {
            return Err(unsupported_geometry(format!(
                "rope type {rope_type} is not the published 2-D default"
            )));
        }
        let theta = required_f64(rope.get("rope_theta"), "rope_parameters.rope_theta")?;
        let standardize = required_bool(vision.get("standardize"), "standardize")?;
        let use_clipped_linears =
            required_bool(vision.get("use_clipped_linears"), "use_clipped_linears")?;
        let activation = vision
            .get("hidden_activation")
            .and_then(Value::as_str)
            .ok_or_else(|| unsupported_geometry("hidden_activation is missing"))?;
        if activation != "gelu_pytorch_tanh" {
            return Err(Gemma4VisionUnsupported::Activation(format!(
                "expected gelu_pytorch_tanh, found {activation}"
            )));
        }
        if patch != VISION_PATCH_SIZE {
            return Err(unsupported_geometry(format!(
                "patch size {patch} is not the supported {VISION_PATCH_SIZE}"
            )));
        }
        if pool != VISION_POOL {
            return Err(unsupported_geometry(format!(
                "pooling kernel {pool} is not the supported {VISION_POOL}"
            )));
        }
        if head_dim == 0 || !head_dim.is_multiple_of(4) {
            return Err(unsupported_geometry(format!(
                "head_dim {head_dim} must be positive and divisible by four"
            )));
        }
        if kv_heads != heads {
            return Err(unsupported_geometry(
                "the qualified vision path requires multi-head (non-GQA) attention",
            ));
        }
        if heads == 0 || heads > VISION_MAX_HEADS {
            return Err(unsupported_geometry(format!(
                "head count {heads} is outside 1..={VISION_MAX_HEADS}"
            )));
        }
        if hidden != heads * head_dim {
            return Err(unsupported_geometry(format!(
                "hidden {hidden} != heads {heads} * head_dim {head_dim}"
            )));
        }
        if layers == 0 || layers > VISION_MAX_LAYERS {
            return Err(unsupported_geometry(format!(
                "layer count {layers} is outside 1..={VISION_MAX_LAYERS}"
            )));
        }
        if !(eps.is_finite() && eps > 0.0) {
            return Err(unsupported_geometry(
                "rms_norm_eps must be finite and positive",
            ));
        }
        if !(theta.is_finite() && theta > 0.0) {
            return Err(unsupported_geometry(
                "rope_theta must be finite and positive",
            ));
        }
        if vision.get("attention_bias").and_then(Value::as_bool) == Some(true) {
            return Err(Gemma4VisionUnsupported::Feature("vision projection biases"));
        }
        if vision
            .get("attention_dropout")
            .and_then(Value::as_f64)
            .unwrap_or(0.0)
            != 0.0
        {
            return Err(Gemma4VisionUnsupported::Feature("vision attention dropout"));
        }
        if position_size == 0 {
            return Err(unsupported_geometry("position_embedding_size is zero"));
        }
        Ok(Self {
            hidden,
            intermediate,
            layers,
            heads,
            kv_heads,
            head_dim,
            patch,
            pool,
            eps: eps as f32,
            theta,
            position_size,
            text_hidden,
            use_clipped_linears,
            standardize,
        })
    }

    pub(crate) fn patch_dim(&self) -> usize {
        self.patch * self.patch * 3
    }

    pub(crate) fn padded_width(&self) -> usize {
        VISION_PADDED_HEADS * self.head_dim
    }

    pub(crate) fn soft_tokens(
        &self,
        grid_thw: [usize; 3],
    ) -> Result<usize, Gemma4VisionUnsupported> {
        let [frames, height, width] = grid_thw;
        if frames != 1 {
            return Err(unsupported_geometry("vision images must have one frame"));
        }
        if height == 0 || width == 0 {
            return Err(unsupported_geometry(
                "vision grid dimensions must be positive",
            ));
        }
        if height % self.pool != 0 || width % self.pool != 0 {
            return Err(unsupported_geometry(
                "vision grid must be divisible by the pooling kernel",
            ));
        }
        (height / self.pool)
            .checked_mul(width / self.pool)
            .ok_or_else(|| unsupported_geometry("soft-token count overflow"))
    }

    pub(crate) fn inverse_frequencies(&self) -> Vec<f32> {
        (0..self.head_dim / 4)
            .map(|index| {
                let exponent = 4.0 * index as f64 / self.head_dim as f64;
                1.0 / (self.theta.powf(exponent) as f32)
            })
            .collect()
    }

    pub(crate) fn inventory_len(&self) -> usize {
        let mut per_layer: usize = 13 + if self.use_clipped_linears { 28 } else { 0 };
        per_layer = per_layer
            .checked_mul(self.layers)
            .expect("checked vision inventory");
        3 + per_layer + usize::from(self.standardize) * 2
    }
}

/// One clippable projection with trained bounds retained as BF16 values.
pub(crate) struct VisionLinearSource {
    pub(crate) weight: WeightSource,
    pub(crate) input_min: f32,
    pub(crate) input_max: f32,
    pub(crate) output_min: f32,
    pub(crate) output_max: f32,
}

pub(crate) struct VisionLayerSource {
    pub(crate) input_norm: WeightSource,
    pub(crate) post_attention_norm: WeightSource,
    pub(crate) pre_feedforward_norm: WeightSource,
    pub(crate) post_feedforward_norm: WeightSource,
    pub(crate) q_norm: WeightSource,
    pub(crate) k_norm: WeightSource,
    pub(crate) q: VisionLinearSource,
    pub(crate) k: VisionLinearSource,
    pub(crate) v: VisionLinearSource,
    pub(crate) o: VisionLinearSource,
    pub(crate) gate: VisionLinearSource,
    pub(crate) up: VisionLinearSource,
    pub(crate) down: VisionLinearSource,
}

pub(crate) struct Gemma4VisionWeights {
    pub(crate) config: Gemma4VisionConfig,
    pub(crate) patch: WeightSource,
    pub(crate) positions: WeightSource,
    pub(crate) projection: WeightSource,
    pub(crate) std_bias: Option<WeightSource>,
    pub(crate) std_scale: Option<WeightSource>,
    pub(crate) layers: Vec<VisionLayerSource>,
    pub(crate) checkpoint_tensors: usize,
}

fn expect_bf16(source: &WeightSource, name: &str) -> Result<(), Gemma4VisionUnsupported> {
    if source.dtype() != safetensors::Dtype::BF16 {
        return Err(Gemma4VisionUnsupported::Dtype(format!(
            "{name} must be BF16, found {:?}",
            source.dtype()
        )));
    }
    Ok(())
}

fn read_scalar(checkpoint: &Checkpoint, name: &str) -> Result<f32, BackendError> {
    let source = checkpoint.source(name, &[])?;
    expect_bf16(&source, name)?;
    let mut bits = [0u16; 1];
    source.read_bf16(0, &mut bits)?;
    let value = f32::from_bits(u32::from(bits[0]) << 16);
    if value.is_nan() {
        return Err(start(format!("Gemma4 vision bound {name} is NaN")));
    }
    Ok(value)
}

fn linear_source(
    checkpoint: &Checkpoint,
    base: &str,
    input: usize,
    output: usize,
    clipped: bool,
) -> Result<VisionLinearSource, BackendError> {
    let linear = format!("{base}.linear.weight");
    let plain = format!("{base}.weight");
    let name = if checkpoint.contains(&linear) {
        linear
    } else {
        plain
    };
    let weight = checkpoint.source(&name, &[output, input])?;
    expect_bf16(&weight, &name)?;
    let (input_min, input_max, output_min, output_max) = if clipped {
        let input_min = read_scalar(checkpoint, &format!("{base}.input_min"))?;
        let input_max = read_scalar(checkpoint, &format!("{base}.input_max"))?;
        let output_min = read_scalar(checkpoint, &format!("{base}.output_min"))?;
        let output_max = read_scalar(checkpoint, &format!("{base}.output_max"))?;
        if !(input_min.is_finite()
            && input_max.is_finite()
            && output_min.is_finite()
            && output_max.is_finite())
        {
            return Err(start(format!(
                "Gemma4 vision trained clipped bounds must be finite for {base}"
            )));
        }
        if input_min > input_max || output_min > output_max {
            return Err(start(format!(
                "Gemma4 vision clipped bounds are inverted for {base}"
            )));
        }
        (input_min, input_max, output_min, output_max)
    } else {
        (
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::INFINITY,
        )
    };
    Ok(VisionLinearSource {
        weight,
        input_min,
        input_max,
        output_min,
        output_max,
    })
}

impl Gemma4VisionWeights {
    pub(crate) fn load(root: &Path) -> Result<Self, BackendError> {
        let config = Gemma4VisionConfig::load(root)?;
        let checkpoint = Checkpoint::open(root)?;
        let checkpoint_tensors = checkpoint.names().count();
        let hidden = config.hidden;
        let layer_root = |index: usize| format!("model.vision_tower.encoder.layers.{index}");
        let patch = checkpoint.source(
            "model.vision_tower.patch_embedder.input_proj.weight",
            &[hidden, config.patch_dim()],
        )?;
        expect_bf16(&patch, "vision patch input_proj")?;
        let positions = checkpoint.source(
            "model.vision_tower.patch_embedder.position_embedding_table",
            &[2, config.position_size, hidden],
        )?;
        expect_bf16(&positions, "vision position_embedding_table")?;
        let projection = checkpoint.source(
            "model.embed_vision.embedding_projection.weight",
            &[config.text_hidden, hidden],
        )?;
        expect_bf16(&projection, "vision embedding_projection")?;
        let (std_bias, std_scale) = if config.standardize {
            let bias = checkpoint.source("model.vision_tower.std_bias", &[hidden])?;
            let scale = checkpoint.source("model.vision_tower.std_scale", &[hidden])?;
            expect_bf16(&bias, "vision std_bias")?;
            expect_bf16(&scale, "vision std_scale")?;
            (Some(bias), Some(scale))
        } else {
            (None, None)
        };
        let mut expected = BTreeSet::new();
        expected.insert("model.vision_tower.patch_embedder.input_proj.weight".to_owned());
        expected.insert("model.vision_tower.patch_embedder.position_embedding_table".to_owned());
        expected.insert("model.embed_vision.embedding_projection.weight".to_owned());
        if config.standardize {
            expected.insert("model.vision_tower.std_bias".to_owned());
            expected.insert("model.vision_tower.std_scale".to_owned());
        }
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(config.layers)
            .map_err(|_| start("Gemma4 vision layer inventory overflow"))?;
        for index in 0..config.layers {
            let base = layer_root(index);
            let norm = |suffix: &str,
                        expected: &mut BTreeSet<String>|
             -> Result<WeightSource, BackendError> {
                let name = format!("{base}.{suffix}.weight");
                expected.insert(name.clone());
                let source = checkpoint.source(&name, &[hidden])?;
                expect_bf16(&source, &name)?;
                Ok(source)
            };
            let qk_norm = |suffix: &str,
                           expected: &mut BTreeSet<String>|
             -> Result<WeightSource, BackendError> {
                let name = format!("{base}.{suffix}.weight");
                expected.insert(name.clone());
                let source = checkpoint.source(&name, &[config.head_dim])?;
                expect_bf16(&source, &name)?;
                Ok(source)
            };
            let linear =
                |suffix: &str, input: usize, output: usize, expected: &mut BTreeSet<String>| {
                    let linear_base = format!("{base}.{suffix}");
                    if config.use_clipped_linears {
                        for bound in ["input_min", "input_max", "output_min", "output_max"] {
                            expected.insert(format!("{linear_base}.{bound}"));
                        }
                    }
                    expected.insert(
                        if checkpoint.contains(&format!("{linear_base}.linear.weight")) {
                            format!("{linear_base}.linear.weight")
                        } else {
                            format!("{linear_base}.weight")
                        },
                    );
                    linear_source(
                        &checkpoint,
                        &linear_base,
                        input,
                        output,
                        config.use_clipped_linears,
                    )
                };
            layers.push(VisionLayerSource {
                input_norm: norm("input_layernorm", &mut expected)?,
                post_attention_norm: norm("post_attention_layernorm", &mut expected)?,
                pre_feedforward_norm: norm("pre_feedforward_layernorm", &mut expected)?,
                post_feedforward_norm: norm("post_feedforward_layernorm", &mut expected)?,
                q_norm: qk_norm("self_attn.q_norm", &mut expected)?,
                k_norm: qk_norm("self_attn.k_norm", &mut expected)?,
                q: linear("self_attn.q_proj", hidden, hidden, &mut expected)?,
                k: linear("self_attn.k_proj", hidden, hidden, &mut expected)?,
                v: linear("self_attn.v_proj", hidden, hidden, &mut expected)?,
                o: linear("self_attn.o_proj", hidden, hidden, &mut expected)?,
                gate: linear("mlp.gate_proj", hidden, config.intermediate, &mut expected)?,
                up: linear("mlp.up_proj", hidden, config.intermediate, &mut expected)?,
                down: linear("mlp.down_proj", config.intermediate, hidden, &mut expected)?,
            });
        }
        let actual: BTreeSet<String> = checkpoint
            .names()
            .filter(|name| {
                name.starts_with("model.vision_tower.") || name.starts_with("model.embed_vision.")
            })
            .cloned()
            .collect();
        if actual != expected {
            let missing: Vec<_> = expected.difference(&actual).take(4).cloned().collect();
            let extra: Vec<_> = actual.difference(&expected).take(4).cloned().collect();
            return Err(start(format!(
                "Gemma4 vision tensor inventory differs ({} expected, {} actual; missing {:?}; extra {:?})",
                expected.len(),
                actual.len(),
                missing,
                extra
            )));
        }
        if actual.len() != config.inventory_len() {
            return Err(start(
                "Gemma4 vision inventory length disagrees with geometry",
            ));
        }
        Ok(Self {
            config,
            patch,
            positions,
            projection,
            std_bias,
            std_scale,
            layers,
            checkpoint_tensors,
        })
    }

    /// Read one BF16 tensor into host bit words for checked padding/upload.
    pub(crate) fn read_bits(source: &WeightSource) -> Result<Vec<u16>, BackendError> {
        let count = source
            .shape()
            .iter()
            .try_fold(1usize, |total, dim| total.checked_mul(*dim))
            .ok_or_else(|| start("Gemma4 vision tensor element overflow"))?;
        let mut bits = Vec::new();
        bits.try_reserve_exact(count)
            .map_err(|_| start("Gemma4 vision tensor allocation overflow"))?;
        bits.resize(count, 0u16);
        source.read_bf16(0, &mut bits)?;
        Ok(bits)
    }
}

/// Checked cold geometry for one packed image request. All images share the
/// production unpadded row-major patch contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct VisionBatchGeometry {
    pub(crate) patches: usize,
    pub(crate) soft_tokens: usize,
    pub(crate) images: usize,
    pub(crate) max_patches: usize,
    pub(crate) max_width: usize,
    pub(crate) max_height: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Gemma4VisionImage<'a> {
    pub(crate) pixels: &'a [f32],
    pub(crate) grid_thw: [usize; 3],
}

pub(crate) fn admit_images(
    config: &Gemma4VisionConfig,
    images: &[Gemma4VisionImage<'_>],
) -> Result<VisionBatchGeometry, Gemma4VisionUnsupported> {
    if images.is_empty() {
        return Err(unsupported_geometry("at least one image is required"));
    }
    if images.len() > 32 {
        return Err(unsupported_geometry("at most 32 images per request"));
    }
    let patch_dim = config.patch_dim();
    let mut patches = 0usize;
    let mut soft = 0usize;
    let mut max_patches = 0usize;
    let mut max_width = 0usize;
    let mut max_height = 0usize;
    for image in images {
        let [frames, height, width] = image.grid_thw;
        if frames != 1 {
            return Err(unsupported_geometry("vision images must have one frame"));
        }
        if height == 0 || width == 0 || height > VISION_ROPE_POSITION_LIMIT {
            return Err(unsupported_geometry(
                "vision grid must be positive and bounded",
            ));
        }
        if height > config.position_size || width > config.position_size {
            return Err(unsupported_geometry(
                "vision grid exceeds the trained position-embedding table",
            ));
        }
        let count = height
            .checked_mul(width)
            .ok_or_else(|| unsupported_geometry("patch count overflow"))?;
        let expected = count
            .checked_mul(patch_dim)
            .ok_or_else(|| unsupported_geometry("pixel element overflow"))?;
        if image.pixels.len() != expected {
            return Err(unsupported_geometry(format!(
                "image pixels {} != {height}x{width}x{patch_dim}",
                image.pixels.len()
            )));
        }
        if !image.pixels.iter().all(|value| value.is_finite()) {
            return Err(unsupported_geometry("image pixels must be finite"));
        }
        patches = patches
            .checked_add(count)
            .ok_or_else(|| unsupported_geometry("total patch count overflow"))?;
        if patches > VISION_POSITION_LIMIT {
            return Err(Gemma4VisionUnsupported::Workspace(
                "packed patch count exceeds the qualified workspace limit".into(),
            ));
        }
        soft = soft
            .checked_add(config.soft_tokens(image.grid_thw)?)
            .ok_or_else(|| unsupported_geometry("total soft-token count overflow"))?;
        max_patches = max_patches.max(count);
        max_width = max_width.max(width);
        max_height = max_height.max(height);
    }
    let bytes = workspace_bytes(config, patches, soft, images.len())?;
    if bytes > VISION_WORKSPACE_LIMIT {
        return Err(Gemma4VisionUnsupported::Workspace(format!(
            "estimated {bytes} bytes exceeds {VISION_WORKSPACE_LIMIT}"
        )));
    }
    Ok(VisionBatchGeometry {
        patches,
        soft_tokens: soft,
        images: images.len(),
        max_patches,
        max_width,
        max_height,
    })
}

const VISION_POSITION_LIMIT: usize = 65_536;

fn workspace_bytes(
    config: &Gemma4VisionConfig,
    patches: usize,
    soft: usize,
    images: usize,
) -> Result<usize, Gemma4VisionUnsupported> {
    let padded = config.padded_width();
    let mut total = 0usize;
    let mut add =
        |elements: usize, width: usize, bytes: usize| -> Result<(), Gemma4VisionUnsupported> {
            let amount = elements
                .checked_mul(width)
                .and_then(|n| n.checked_mul(bytes))
                .ok_or_else(|| Gemma4VisionUnsupported::Workspace("workspace overflow".into()))?;
            total = total
                .checked_add(amount)
                .ok_or_else(|| Gemma4VisionUnsupported::Workspace("workspace overflow".into()))?;
            Ok(())
        };
    // Every allocation performed by `Gemma4VisionWork::new`, in the same units.
    add(patches, 1, 4)?; // columns
    add(patches, 1, 4)?; // rows
    add(
        images
            .checked_add(1)
            .ok_or_else(|| Gemma4VisionUnsupported::Workspace("metadata overflow".into()))?,
        1,
        4,
    )?; // cumulative
    add(config.head_dim / 4, 1, 4)?; // inverse frequencies
    add(patches, config.head_dim / 2, 2)?; // cosine
    add(patches, config.head_dim / 2, 2)?; // sine
    add(patches, config.hidden, 2)?; // state
    add(patches, config.hidden, 2)?; // normalized
    add(patches, config.hidden, 2)?; // clipped
    add(patches, padded, 2)?; // q
    add(patches, padded, 2)?; // k
    add(patches, padded, 2)?; // v
    add(patches, padded, 2)?; // attention
    add(patches, padded, 2)?; // rope_out
    add(patches, config.hidden, 2)?; // branch
    add(patches, config.intermediate, 4)?; // gate_up (two BF16 halves)
    add(patches, config.intermediate, 2)?; // activation
    add(soft, config.hidden, 2)?; // pooled
    add(soft, config.text_hidden, 2)?; // projected BF16
    add(soft, config.text_hidden, 4)?; // F32 widening
    add(patches, config.patch_dim(), 2)?; // staged pixels
    add(soft, 1, 4)?; // retained scatter row ids
    Ok(total)
}

/// Deterministic host patch/position mapping. Patches are row-major (y, x) and
/// every patch stores `patch * patch * 3` channel-major bytes.
type PatchPositions = (Vec<u32>, Vec<u32>, Vec<u32>);

pub(crate) fn patch_positions(
    images: &[Gemma4VisionImage<'_>],
) -> Result<PatchPositions, Gemma4VisionUnsupported> {
    let mut columns = Vec::new();
    let mut rows = Vec::new();
    let mut cumulative = vec![0u32];
    for image in images {
        let [frames, height, width] = image.grid_thw;
        if frames != 1 {
            return Err(unsupported_geometry("vision images must have one frame"));
        }
        let count = height
            .checked_mul(width)
            .ok_or_else(|| unsupported_geometry("patch count overflow"))?;
        let end = cumulative
            .last()
            .copied()
            .unwrap_or(0)
            .checked_add(
                u32::try_from(count)
                    .map_err(|_| unsupported_geometry("packed patch count exceeds int32"))?,
            )
            .ok_or_else(|| unsupported_geometry("packed patch count overflow"))?;
        cumulative.push(end);
        for index in 0..count {
            columns.push(
                u32::try_from(index % width)
                    .map_err(|_| unsupported_geometry("patch column exceeds int32"))?,
            );
            rows.push(
                u32::try_from(index / width)
                    .map_err(|_| unsupported_geometry("patch row exceeds int32"))?,
            );
        }
    }
    Ok((columns, rows, cumulative))
}

/// BF16 rounding helper shared by the CPU reference and fixture checks.
pub(crate) fn bf16_round(value: f32) -> f32 {
    half::bf16::from_f32(value).to_f32()
}

pub(crate) fn bf16_bits(value: f32) -> u16 {
    half::bf16::from_f32(value).to_bits()
}

pub(crate) fn bf16_value(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn clamp_bf16(value: f32, min: f32, max: f32) -> f32 {
    // Bounds and inputs are already BF16-exact; clamping preserves a BF16 value.
    value.max(min).min(max)
}

fn rms_norm_bf16_row(input: &[f32], scale: Option<&[f32]>, eps: f32) -> Vec<f32> {
    let width = input.len();
    let sum = input.iter().map(|value| value * value).sum::<f32>();
    let mean = sum / width as f32;
    let denominator = (mean + eps).sqrt();
    let mut output = vec![0.0f32; width];
    for column in 0..width {
        let normalized = input[column] / denominator;
        let scaled = match scale {
            Some(scale) => normalized * scale[column],
            None => normalized,
        };
        output[column] = bf16_round(scaled);
    }
    output
}

const GELU_ALPHA: f32 = (std::f64::consts::FRAC_2_SQRT_PI * std::f64::consts::FRAC_1_SQRT_2) as f32;

fn gelu_tanh_bf16(value: f32) -> f32 {
    let square = bf16_round(value * value);
    let cube = bf16_round(square * value);
    let product = bf16_round(bf16_round(0.044715f32) * cube);
    let alpha = bf16_round(value + product);
    let angle = bf16_round(bf16_round(GELU_ALPHA) * alpha);
    let tangent = bf16_round(angle.tanh());
    let half = bf16_round(0.5f32 * value);
    let plus = bf16_round(1.0f32 + tangent);
    bf16_round(half * plus)
}

fn apply_rope2d_bf16(
    input: &[f32],
    cos: &[f32],
    sin: &[f32],
    heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let tokens = input.len() / (heads * head_dim);
    let quarter = head_dim / 4;
    let mut output = vec![0.0f32; input.len()];
    for token in 0..tokens {
        for head in 0..heads {
            for axis in 0..2 {
                let base = (token * heads + head) * head_dim + axis * head_dim / 2;
                for part in 0..quarter {
                    let a = input[base + part];
                    let b = input[base + quarter + part];
                    let c = cos[token * head_dim / 2 + axis * quarter + part];
                    let s = sin[token * head_dim / 2 + axis * quarter + part];
                    let first = bf16_round(a * c) - bf16_round(b * s);
                    let second = bf16_round(b * c) + bf16_round(a * s);
                    output[base + part] = bf16_round(first);
                    output[base + quarter + part] = bf16_round(second);
                }
            }
        }
    }
    output
}

/// Independent host reference used by CPU admission. It deliberately keeps
/// every published BF16 activation boundary while accumulating GEMMs in F32.
#[cfg(test)]
pub(crate) struct HostVisionWeights {
    patch: Vec<f32>,
    positions: Vec<f32>,
    projection: Vec<f32>,
    layers: Vec<HostVisionLayer>,
}

#[cfg(test)]
struct HostVisionLayer {
    input_norm: Vec<f32>,
    post_attention_norm: Vec<f32>,
    pre_feedforward_norm: Vec<f32>,
    post_feedforward_norm: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    q: HostLinear,
    k: HostLinear,
    v: HostLinear,
    o: HostLinear,
    gate: HostLinear,
    up: HostLinear,
    down: HostLinear,
}

#[cfg(test)]
struct HostLinear {
    weight: Vec<f32>,
    input_min: f32,
    input_max: f32,
    output_min: f32,
    output_max: f32,
}

#[cfg(test)]
impl HostVisionWeights {
    fn read_f32(source: &WeightSource) -> Result<Vec<f32>, BackendError> {
        Ok(Gemma4VisionWeights::read_bits(source)?
            .into_iter()
            .map(bf16_value)
            .collect())
    }

    fn read_linear(source: &VisionLinearSource) -> Result<HostLinear, BackendError> {
        Ok(HostLinear {
            weight: Self::read_f32(&source.weight)?,
            input_min: source.input_min,
            input_max: source.input_max,
            output_min: source.output_min,
            output_max: source.output_max,
        })
    }

    fn pad_columns(
        linear: &mut HostLinear,
        input: usize,
        output: usize,
        padded: usize,
    ) -> Result<(), BackendError> {
        if linear.weight.len() != input * output || padded < input {
            return Err(start("Gemma4 vision host O padding differs"));
        }
        let mut weight = vec![0.0f32; padded * output];
        for row in 0..output {
            weight[row * padded..row * padded + input]
                .copy_from_slice(&linear.weight[row * input..(row + 1) * input]);
        }
        linear.weight = weight;
        Ok(())
    }

    pub(crate) fn read(weights: &Gemma4VisionWeights) -> Result<Self, BackendError> {
        let mut layers = Vec::new();
        for layer in &weights.layers {
            let mut o = Self::read_linear(&layer.o)?;
            Self::pad_columns(
                &mut o,
                weights.config.hidden,
                weights.config.hidden,
                weights.config.padded_width(),
            )?;
            layers.push(HostVisionLayer {
                input_norm: Self::read_f32(&layer.input_norm)?,
                post_attention_norm: Self::read_f32(&layer.post_attention_norm)?,
                pre_feedforward_norm: Self::read_f32(&layer.pre_feedforward_norm)?,
                post_feedforward_norm: Self::read_f32(&layer.post_feedforward_norm)?,
                q_norm: Self::read_f32(&layer.q_norm)?,
                k_norm: Self::read_f32(&layer.k_norm)?,
                q: Self::read_linear(&layer.q)?,
                k: Self::read_linear(&layer.k)?,
                v: Self::read_linear(&layer.v)?,
                o,
                gate: Self::read_linear(&layer.gate)?,
                up: Self::read_linear(&layer.up)?,
                down: Self::read_linear(&layer.down)?,
            });
        }
        Ok(Self {
            patch: Self::read_f32(&weights.patch)?,
            positions: Self::read_f32(&weights.positions)?,
            projection: Self::read_f32(&weights.projection)?,
            layers,
        })
    }
}

#[cfg(test)]
fn host_linear_width(
    linear: &HostLinear,
    input: &[f32],
    input_width: usize,
    rows: usize,
) -> Vec<f32> {
    let out_width = linear.weight.len() / input_width;
    let mut output = vec![0.0f32; rows * out_width];
    for row in 0..rows {
        for out in 0..out_width {
            let mut sum = 0.0f32;
            for column in 0..input_width {
                let value = clamp_bf16(
                    input[row * input_width + column],
                    linear.input_min,
                    linear.input_max,
                );
                sum += value * linear.weight[out * input_width + column];
            }
            let rounded = bf16_round(sum);
            output[row * out_width + out] =
                clamp_bf16(rounded, linear.output_min, linear.output_max);
        }
    }
    output
}

#[cfg(test)]
pub(crate) fn reference_forward(
    config: &Gemma4VisionConfig,
    weights: &HostVisionWeights,
    image: Gemma4VisionImage<'_>,
) -> Result<Vec<f32>, Gemma4VisionUnsupported> {
    let geometry = admit_images(config, &[image])?;
    let [_, height, width] = image.grid_thw;
    let hidden = config.hidden;
    let padded = config.padded_width();
    let head_dim = config.head_dim;
    let heads = config.heads;
    let tokens = geometry.patches;
    let mut state = vec![0.0f32; tokens * hidden];
    let (columns, rows, _) = patch_positions(&[image])?;
    let patch = &image.pixels;
    for token in 0..tokens {
        for column in 0..hidden {
            let mut sum = 0.0f32;
            let x = columns[token] as usize;
            let y = rows[token] as usize;
            for k in 0..config.patch_dim() {
                let value = bf16_round(2.0 * (patch[token * config.patch_dim() + k] - 0.5));
                sum += value * weights.patch[column * config.patch_dim() + k];
            }
            let mut value = bf16_round(sum);
            let x_index = x;
            let y_index = y;
            let x_emb = weights.positions[x_index * hidden + column];
            let y_emb =
                weights.positions[config.position_size * hidden + y_index * hidden + column];
            let position = bf16_round(bf16_round(x_emb) + bf16_round(y_emb));
            value = bf16_round(value + position);
            state[token * hidden + column] = value;
        }
    }
    let inverse = config.inverse_frequencies();
    let mut cos = vec![0.0f32; tokens * head_dim / 2];
    let mut sin = vec![0.0f32; tokens * head_dim / 2];
    for token in 0..tokens {
        for part in 0..VISION_ROPE_PAIRS {
            let x_phase = columns[token] as f32 * inverse[part];
            let y_phase = rows[token] as f32 * inverse[part];
            cos[token * head_dim / 2 + part] = bf16_round(x_phase.cos());
            sin[token * head_dim / 2 + part] = bf16_round(x_phase.sin());
            cos[token * head_dim / 2 + VISION_ROPE_PAIRS + part] = bf16_round(y_phase.cos());
            sin[token * head_dim / 2 + VISION_ROPE_PAIRS + part] = bf16_round(y_phase.sin());
        }
    }
    for layer in &weights.layers {
        let normalized = {
            let mut out = Vec::with_capacity(tokens * hidden);
            for token in 0..tokens {
                out.extend(rms_norm_bf16_row(
                    &state[token * hidden..(token + 1) * hidden],
                    Some(&layer.input_norm),
                    config.eps,
                ));
            }
            out
        };
        let mut q = vec![0.0f32; tokens * padded];
        let mut k = vec![0.0f32; tokens * padded];
        let mut v = vec![0.0f32; tokens * padded];
        let qw = host_linear_width(&layer.q, &normalized, hidden, tokens);
        let kw = host_linear_width(&layer.k, &normalized, hidden, tokens);
        let vw = host_linear_width(&layer.v, &normalized, hidden, tokens);
        for token in 0..tokens {
            q[token * padded..token * padded + hidden]
                .copy_from_slice(&qw[token * hidden..(token + 1) * hidden]);
            k[token * padded..token * padded + hidden]
                .copy_from_slice(&kw[token * hidden..(token + 1) * hidden]);
            v[token * padded..token * padded + hidden]
                .copy_from_slice(&vw[token * hidden..(token + 1) * hidden]);
        }
        for token in 0..tokens {
            for head in 0..VISION_PADDED_HEADS {
                let base = token * padded + head * head_dim;
                if head < heads {
                    let qrow = rms_norm_bf16_row(
                        &q[base..base + head_dim],
                        Some(&layer.q_norm),
                        config.eps,
                    );
                    let krow = rms_norm_bf16_row(
                        &k[base..base + head_dim],
                        Some(&layer.k_norm),
                        config.eps,
                    );
                    let vrow = rms_norm_bf16_row(&v[base..base + head_dim], None, config.eps);
                    q[base..base + head_dim].copy_from_slice(&qrow);
                    k[base..base + head_dim].copy_from_slice(&krow);
                    v[base..base + head_dim].copy_from_slice(&vrow);
                }
            }
        }
        let qrope = apply_rope2d_bf16(&q, &cos, &sin, VISION_PADDED_HEADS, head_dim);
        let krope = apply_rope2d_bf16(&k, &cos, &sin, VISION_PADDED_HEADS, head_dim);
        let mut attention = vec![0.0f32; tokens * padded];
        let mut scores = vec![0.0f32; tokens];
        for token in 0..tokens {
            for head in 0..VISION_PADDED_HEADS {
                let qbase = token * padded + head * head_dim;
                let mut max = f32::NEG_INFINITY;
                for (key, score) in scores.iter_mut().enumerate() {
                    let kbase = key * padded + head * head_dim;
                    let mut dot = 0.0f32;
                    for dim in 0..head_dim {
                        dot += qrope[qbase + dim] * krope[kbase + dim];
                    }
                    *score = dot;
                    max = max.max(dot);
                }
                let mut sum = 0.0f32;
                for score in &mut scores {
                    *score = (*score - max).exp();
                    sum += *score;
                }
                for (key, score) in scores.iter().enumerate() {
                    let weight = bf16_round(*score / sum);
                    let kbase = key * padded + head * head_dim;
                    for dim in 0..head_dim {
                        attention[qbase + dim] += weight * v[kbase + dim];
                    }
                }
                for dim in 0..head_dim {
                    attention[qbase + dim] = clamp_bf16(
                        bf16_round(attention[qbase + dim]),
                        layer.o.input_min,
                        layer.o.input_max,
                    );
                }
            }
        }
        let o = host_linear_width(&layer.o, &attention, padded, tokens);
        let mut branch = vec![0.0f32; tokens * hidden];
        for token in 0..tokens {
            let normed = rms_norm_bf16_row(
                &o[token * hidden..(token + 1) * hidden],
                Some(&layer.post_attention_norm),
                config.eps,
            );
            for column in 0..hidden {
                branch[token * hidden + column] =
                    bf16_round(state[token * hidden + column] + normed[column]);
            }
        }
        state = branch;
        let normalized = {
            let mut out = Vec::with_capacity(tokens * hidden);
            for token in 0..tokens {
                out.extend(rms_norm_bf16_row(
                    &state[token * hidden..(token + 1) * hidden],
                    Some(&layer.pre_feedforward_norm),
                    config.eps,
                ));
            }
            out
        };
        let gate = host_linear_width(&layer.gate, &normalized, hidden, tokens);
        let up = host_linear_width(&layer.up, &normalized, hidden, tokens);
        let mut active = vec![0.0f32; tokens * config.intermediate];
        for index in 0..active.len() {
            active[index] = bf16_round(gelu_tanh_bf16(gate[index]) * up[index]);
        }
        let down = host_linear_width(&layer.down, &active, config.intermediate, tokens);
        let mut next = vec![0.0f32; tokens * hidden];
        for token in 0..tokens {
            let normed = rms_norm_bf16_row(
                &down[token * hidden..(token + 1) * hidden],
                Some(&layer.post_feedforward_norm),
                config.eps,
            );
            for column in 0..hidden {
                next[token * hidden + column] =
                    bf16_round(state[token * hidden + column] + normed[column]);
            }
        }
        state = next;
    }
    // Spatial pooling plus final unit norm and projection.
    let pool = config.pool;
    let soft_height = height / pool;
    let soft_width = width / pool;
    let mut pooled = vec![0.0f32; geometry.soft_tokens * hidden];
    for token in 0..tokens {
        let bin = (columns[token] as usize / pool) + (rows[token] as usize / pool) * soft_width;
        for column in 0..hidden {
            pooled[bin * hidden + column] += state[token * hidden + column] / (pool * pool) as f32;
        }
    }
    let root = (hidden as f32).sqrt();
    for value in &mut pooled {
        *value = bf16_round(bf16_round(*value) * root);
    }
    let mut normed = Vec::with_capacity(pooled.len());
    for row in 0..geometry.soft_tokens {
        normed.extend(rms_norm_bf16_row(
            &pooled[row * hidden..(row + 1) * hidden],
            None,
            config.eps,
        ));
    }
    let projection = host_linear_width(
        &HostLinear {
            weight: weights.projection.clone(),
            input_min: f32::NEG_INFINITY,
            input_max: f32::INFINITY,
            output_min: f32::NEG_INFINITY,
            output_max: f32::INFINITY,
        },
        &normed,
        hidden,
        geometry.soft_tokens,
    );
    let _ = soft_height;
    Ok(projection)
}

/// Injection mapping for the production placeholder contract. Features are in
/// image order and each image contributes exactly its soft-token rows.
pub(crate) fn injection_rows(
    input_ids: &[u32],
    image_token_id: u32,
    features: &[Vec<u16>],
    width: usize,
) -> Result<Vec<usize>, Gemma4VisionUnsupported> {
    if features.is_empty() {
        return Err(unsupported_geometry("no image features to inject"));
    }
    let expected = features
        .iter()
        .try_fold(0usize, |total, rows| {
            total.checked_add(rows.len() / width.max(1))
        })
        .ok_or_else(|| unsupported_geometry("feature row count overflow"))?;
    if features
        .iter()
        .any(|rows| width == 0 || rows.len() % width != 0)
    {
        return Err(unsupported_geometry(
            "image feature rows must be whole text hidden-width rows",
        ));
    }
    let rows: Vec<usize> = input_ids
        .iter()
        .enumerate()
        .filter_map(|(index, id)| (*id == image_token_id).then_some(index))
        .collect();
    if rows.len() != expected {
        return Err(unsupported_geometry(format!(
            "{} image placeholders != {expected} soft-token rows",
            rows.len()
        )));
    }
    Ok(rows)
}

/// Copy BF16 image feature rows into the gathered text embedding buffer.
pub(crate) fn inject_features_host(
    embeddings: &mut [u16],
    input_ids: &[u32],
    image_token_id: u32,
    features: &[Vec<u16>],
    width: usize,
) -> Result<(), Gemma4VisionUnsupported> {
    let rows = injection_rows(input_ids, image_token_id, features, width)?;
    let mut cursor = 0usize;
    for feature in features {
        for feature_row in feature.chunks_exact(width) {
            let row = rows[cursor];
            let destination = row
                .checked_mul(width)
                .ok_or_else(|| unsupported_geometry("embedding row offset overflow"))?;
            let target = embeddings
                .get_mut(destination..destination + width)
                .ok_or_else(|| unsupported_geometry("embedding row out of range"))?;
            target.copy_from_slice(feature_row);
            cursor += 1;
        }
    }
    Ok(())
}

/// Loaded device functions for the concrete vision path.
pub(crate) struct Gemma4VisionKernels {
    norm: Function,
    residual_norm: Function,
    unit_norm: Function,
    gated_gelu: Function,
    clip_rows: Function,
    position_add: Function,
    rope_tables: Function,
    rope2d: Function,
    pool: Function,
    pool_standardize: Function,
    scatter: Function,
    bf16_to_f32: Function,
}

impl Gemma4VisionKernels {
    pub(crate) fn load(
        context: &Arc<Context>,
        module_path: Option<&Path>,
    ) -> Result<Self, BackendError> {
        let path = match module_path {
            Some(path) => path.to_path_buf(),
            None => std::env::var_os(infinity_v3_config::infinity_env!("OXIDE_KERNEL_MODULE"))
                .map(PathBuf::from)
                .ok_or_else(|| start("set INFINITY_OXIDE_KERNEL_MODULE to the vision artifact"))?,
        };
        let path = if path.is_dir() {
            let (major, minor) = context.compute_capability().map_err(start)?;
            path.join(format!("sm_{major}{minor}"))
                .join("kernels.cubin")
        } else {
            path
        };
        let data = std::fs::read(&path).map_err(start)?;
        let module = context.load_module_from_image(&data).map_err(start)?;
        let function = |index: usize| {
            module
                .load_function(VISION_KERNEL_EXPORTS[index])
                .map_err(start)
        };
        Ok(Self {
            norm: function(0)?,
            residual_norm: function(1)?,
            unit_norm: function(2)?,
            gated_gelu: function(3)?,
            clip_rows: function(4)?,
            position_add: function(5)?,
            rope_tables: function(6)?,
            rope2d: function(7)?,
            pool: function(8)?,
            pool_standardize: function(9)?,
            scatter: module.load_function(VISION_SCATTER_EXPORT).map_err(start)?,
            bf16_to_f32: function(10)?,
        })
    }
}

fn reference_threads(width: usize) -> u32 {
    let mut threads = 1usize;
    while threads < width && threads < 1024 {
        threads <<= 1;
    }
    threads as u32
}

/// One retained vision arena. All allocations stay owned until the stream has
/// completed every enqueue that references them.
pub(crate) struct Gemma4VisionWork {
    stream: Arc<Stream>,
    geometry: VisionBatchGeometry,
    bytes: usize,
    columns: DeviceBuffer<u32>,
    rows: DeviceBuffer<u32>,
    cumulative: DeviceBuffer<u32>,
    image_grids: Vec<[usize; 3]>,
    inverse: DeviceBuffer<f32>,
    cosine: DeviceBuffer<Bf16>,
    sine: DeviceBuffer<Bf16>,
    state: DeviceBuffer<Bf16>,
    normalized: DeviceBuffer<Bf16>,
    clipped: DeviceBuffer<Bf16>,
    q: DeviceBuffer<Bf16>,
    k: DeviceBuffer<Bf16>,
    v: DeviceBuffer<Bf16>,
    attention: DeviceBuffer<Bf16>,
    branch: DeviceBuffer<Bf16>,
    gate_up: DeviceBuffer<Bf16>,
    activation: DeviceBuffer<Bf16>,
    rope_out: DeviceBuffer<Bf16>,
    pooled: DeviceBuffer<Bf16>,
    projected: DeviceBuffer<Bf16>,
    f32_out: DeviceBuffer<f32>,
    pixels: DeviceBuffer<Bf16>,
    scatter_rows: DeviceBuffer<u32>,
}

impl Gemma4VisionWork {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        stream: Arc<Stream>,
        config: &Gemma4VisionConfig,
        images: &[Gemma4VisionImage<'_>],
    ) -> Result<Self, BackendError> {
        let geometry = admit_images(config, images)?;
        let bytes = workspace_bytes(config, geometry.patches, geometry.soft_tokens, images.len())?;
        let (columns, rows, cumulative) = patch_positions(images)?;
        let alloc = |elements: usize| -> Result<DeviceBuffer<Bf16>, BackendError> {
            DeviceBuffer::zeroed(&stream, elements).map_err(infer)
        };
        let patches = geometry.patches;
        let padded = config.padded_width();
        let state = alloc(product(patches, config.hidden)?)?;
        let normalized = alloc(product(patches, config.hidden)?)?;
        let clipped = alloc(product(patches, config.hidden)?)?;
        let q = alloc(product(patches, padded)?)?;
        let k = alloc(product(patches, padded)?)?;
        let v = alloc(product(patches, padded)?)?;
        let attention = alloc(product(patches, padded)?)?;
        let branch = alloc(product(patches, config.hidden)?)?;
        let gate_up = alloc(product(
            patches,
            config
                .intermediate
                .checked_mul(2)
                .ok_or_else(|| infer("Gemma4 vision gate/up overflow"))?,
        )?)?;
        let activation = alloc(product(patches, config.intermediate)?)?;
        let rope_out = alloc(product(patches, padded)?)?;
        let pooled = alloc(product(geometry.soft_tokens, config.hidden)?)?;
        let projected = alloc(product(geometry.soft_tokens, config.text_hidden)?)?;
        let f32_out =
            DeviceBuffer::zeroed(&stream, product(geometry.soft_tokens, config.text_hidden)?)
                .map_err(infer)?;
        let pixels = alloc(product(patches, config.patch_dim())?)?;
        let scatter_rows = DeviceBuffer::zeroed(&stream, geometry.soft_tokens).map_err(infer)?;
        let image_grids: Vec<[usize; 3]> = images.iter().map(|image| image.grid_thw).collect();
        let inverse =
            DeviceBuffer::from_host(&stream, &config.inverse_frequencies()).map_err(infer)?;
        let cosine = alloc(product(patches, config.head_dim / 2)?)?;
        let sine = alloc(product(patches, config.head_dim / 2)?)?;
        crate::LIVE_ARENA_BYTES.fetch_add(bytes, Ordering::Relaxed);
        crate::record_live_arena_bytes();
        Ok(Self {
            stream: stream.clone(),
            geometry,
            bytes,
            columns: DeviceBuffer::from_host(&stream, &columns).map_err(infer)?,
            rows: DeviceBuffer::from_host(&stream, &rows).map_err(infer)?,
            cumulative: DeviceBuffer::from_host(&stream, &cumulative).map_err(infer)?,
            image_grids,
            inverse,
            cosine,
            sine,
            state,
            normalized,
            clipped,
            q,
            k,
            v,
            attention,
            branch,
            gate_up,
            activation,
            rope_out,
            pooled,
            projected,
            f32_out,
            pixels,
            scatter_rows,
        })
    }

    /// Expose the projected text-hidden rows produced by the last completed
    /// `encode`. The owner stays with the request until text completion.
    pub(crate) fn projected(&self) -> &DeviceBuffer<Bf16> {
        &self.projected
    }

    pub(crate) fn soft_tokens(&self) -> usize {
        self.geometry.soft_tokens
    }

    /// Copy the ordered text injection row ids into retained device scratch.
    pub(crate) fn set_scatter_rows(&mut self, rows: &[u32]) -> Result<(), BackendError> {
        if rows.len() != self.geometry.soft_tokens {
            return Err(infer("Gemma4 vision scatter row count differs"));
        }
        if rows.len() > self.scatter_rows.len() {
            return Err(infer("Gemma4 vision scatter arena is too short"));
        }
        self.scatter_rows
            .copy_from_host(&self.stream, rows)
            .map_err(infer)
    }

    pub(crate) fn scatter_rows_ptr(&self) -> u64 {
        self.scatter_rows.cu_deviceptr()
    }

    pub(crate) fn geometry(&self) -> VisionBatchGeometry {
        self.geometry
    }

    pub(crate) fn bytes(&self) -> usize {
        self.columns.num_bytes()
            + self.rows.num_bytes()
            + self.cumulative.num_bytes()
            + self.inverse.num_bytes()
            + self.cosine.num_bytes()
            + self.sine.num_bytes()
            + self.state.num_bytes()
            + self.normalized.num_bytes()
            + self.clipped.num_bytes()
            + self.q.num_bytes()
            + self.k.num_bytes()
            + self.v.num_bytes()
            + self.attention.num_bytes()
            + self.branch.num_bytes()
            + self.gate_up.num_bytes()
            + self.activation.num_bytes()
            + self.rope_out.num_bytes()
            + self.pooled.num_bytes()
            + self.projected.num_bytes()
            + self.f32_out.num_bytes()
            + self.pixels.num_bytes()
            + self.scatter_rows.num_bytes()
    }

    pub(crate) fn forward_bytes(&self) -> usize {
        self.projected.num_bytes()
    }

    /// Stage packed BF16-normalized pixels from the caller's OwnedIO arena. The
    /// arena key only bounds capacity; the work owns the device destination.
    pub(crate) fn stage_pixels(
        &mut self,
        config: &Gemma4VisionConfig,
        images: &[Gemma4VisionImage<'_>],
        lease: &mut HostIoLease,
    ) -> Result<usize, BackendError> {
        let geometry = admit_images(config, images)?;
        if geometry != self.geometry {
            return Err(infer(
                "Gemma4 vision lease geometry differs from the retained work",
            ));
        }
        let elements = product(geometry.patches, config.patch_dim())?;
        let words = images
            .iter()
            .try_fold(0usize, |total, image| {
                total.checked_add(image.pixels.len().div_ceil(2))
            })
            .ok_or_else(|| infer("Gemma4 vision staging word overflow"))?;
        let key = (words.div_ceil(3), 0, 0);
        lease.ensure_stageable()?;
        lease.prepare(
            key,
            product(geometry.soft_tokens, config.text_hidden)?,
            true,
        )?;
        let (host_ptr, host_len) = {
            let host = lease.slot_mut()?.input.as_mut_slice();
            (host.as_mut_ptr(), host.len())
        };
        if host_len < words {
            return Err(infer("Gemma4 vision staging arena is too short"));
        }
        // SAFETY: the pointer names the lease-owned pinned arena, `host_len`
        // was read from that same allocation, and no lease method is called
        // while the temporary slice is alive.
        let host = unsafe { std::slice::from_raw_parts_mut(host_ptr, host_len) };
        let mut cursor = 0usize;
        for image in images {
            for chunk in image.pixels.chunks_exact(2) {
                let first = bf16_bits(2.0 * (chunk[0] - 0.5));
                let second = bf16_bits(2.0 * (chunk[1] - 0.5));
                host[cursor] = u32::from(first) | (u32::from(second) << 16);
                cursor += 1;
            }
            if image.pixels.len() % 2 == 1 {
                let last = image.pixels[image.pixels.len() - 1];
                host[cursor] = u32::from(bf16_bits(2.0 * (last - 0.5)));
                cursor += 1;
            }
        }
        if cursor != words {
            return Err(infer("Gemma4 vision staging word count differs"));
        }
        lease.begin_checked()?;
        // SAFETY: the lease owns the pinned source and the work owns the
        // destination; both remain live until the caller waits for the stream.
        unsafe {
            cuda_core::memory::memcpy_htod_async(
                self.pixels.cu_deviceptr(),
                host_ptr.cast_const(),
                elements * 2,
                self.stream.cu_stream(),
            )
        }
        .map_err(infer)?;
        Ok(elements)
    }
}

impl Drop for Gemma4VisionWork {
    fn drop(&mut self) {
        // Owned arenas are released only after the supported stream drains.
        // On an unhealthy context the process owns final teardown; the explicit
        // Root gate runs memcheck/initcheck and a process-isolated negative.
        if self.stream.synchronize().is_ok() {
            crate::LIVE_ARENA_BYTES.fetch_sub(self.bytes, Ordering::Relaxed);
            crate::record_live_arena_bytes();
        } else {
            metrics::counter!("oxide_arena_retained_bytes_total").increment(self.bytes as u64);
        }
    }
}

struct VisionLinearBuffers {
    weight: DeviceBuffer<Bf16>,
    input_min: f32,
    input_max: f32,
    output_min: f32,
    output_max: f32,
}

struct VisionLayerBuffers {
    input_norm: DeviceBuffer<f32>,
    post_attention_norm: DeviceBuffer<f32>,
    pre_feedforward_norm: DeviceBuffer<f32>,
    post_feedforward_norm: DeviceBuffer<f32>,
    q_norm: DeviceBuffer<f32>,
    k_norm: DeviceBuffer<f32>,
    q: VisionLinearBuffers,
    k: VisionLinearBuffers,
    v: VisionLinearBuffers,
    o: VisionLinearBuffers,
    gate: VisionLinearBuffers,
    up: VisionLinearBuffers,
    down: VisionLinearBuffers,
}

struct VisionWeightBuffers {
    patch: DeviceBuffer<Bf16>,
    positions: DeviceBuffer<Bf16>,
    projection: DeviceBuffer<Bf16>,
    std_bias: Option<DeviceBuffer<f32>>,
    std_scale: Option<DeviceBuffer<f32>>,
    layers: Vec<VisionLayerBuffers>,
}

fn upload_norm_source(
    stream: &Arc<Stream>,
    source: &WeightSource,
    width: usize,
) -> Result<DeviceBuffer<f32>, BackendError> {
    expect_bf16(source, "vision norm")?;
    let bits = Gemma4VisionWeights::read_bits(source)?;
    if bits.len() != width {
        return Err(start("Gemma4 vision norm width differs"));
    }
    let values: Vec<f32> = bits.into_iter().map(bf16_value).collect();
    DeviceBuffer::from_host(stream, &values).map_err(infer)
}

fn upload_plain(
    stream: &Arc<Stream>,
    source: &WeightSource,
) -> Result<DeviceBuffer<Bf16>, BackendError> {
    expect_bf16(source, "vision tensor")?;
    let bits = Gemma4VisionWeights::read_bits(source)?;
    let values: Vec<Bf16> = bits.into_iter().map(Bf16::from_bits).collect();
    DeviceBuffer::from_host(stream, &values).map_err(infer)
}

fn pad_rows_bits(
    bits: &[u16],
    input: usize,
    output: usize,
    padded: usize,
) -> Result<Vec<u16>, BackendError> {
    if padded < output {
        return Err(start("Gemma4 vision projection padding is too small"));
    }
    if bits.len()
        != input
            .checked_mul(output)
            .ok_or_else(|| start("vision projection overflow"))?
    {
        return Err(start("Gemma4 vision projection shape differs"));
    }
    let mut padded_bits = vec![
        0u16;
        input
            .checked_mul(padded)
            .ok_or_else(|| start("vision projection overflow"))?
    ];
    padded_bits[..bits.len()].copy_from_slice(bits);
    Ok(padded_bits)
}

fn pad_columns_bits(
    bits: &[u16],
    input: usize,
    output: usize,
    padded: usize,
) -> Result<Vec<u16>, BackendError> {
    if padded < input {
        return Err(start("Gemma4 vision projection padding is too small"));
    }
    if bits.len()
        != input
            .checked_mul(output)
            .ok_or_else(|| start("vision projection overflow"))?
    {
        return Err(start("Gemma4 vision projection shape differs"));
    }
    let mut padded_bits = vec![
        0u16;
        padded
            .checked_mul(output)
            .ok_or_else(|| start("vision projection overflow"))?
    ];
    for row in 0..output {
        padded_bits[row * padded..row * padded + input]
            .copy_from_slice(&bits[row * input..(row + 1) * input]);
    }
    Ok(padded_bits)
}

/// Pad output rows with exact BF16 zeros, preserving trained row order.
fn pad_weight_rows(
    source: &WeightSource,
    input: usize,
    output: usize,
    padded: usize,
) -> Result<Vec<Bf16>, BackendError> {
    expect_bf16(source, "vision projection")?;
    let bits = Gemma4VisionWeights::read_bits(source)?;
    Ok(pad_rows_bits(&bits, input, output, padded)?
        .into_iter()
        .map(Bf16::from_bits)
        .collect())
}

/// Pad input columns with exact BF16 zeros, preserving trained row order.
fn pad_weight_columns(
    source: &WeightSource,
    input: usize,
    output: usize,
    padded: usize,
) -> Result<Vec<Bf16>, BackendError> {
    expect_bf16(source, "vision projection")?;
    let bits = Gemma4VisionWeights::read_bits(source)?;
    Ok(pad_columns_bits(&bits, input, output, padded)?
        .into_iter()
        .map(Bf16::from_bits)
        .collect())
}

fn upload_linear(
    stream: &Arc<Stream>,
    source: &VisionLinearSource,
    input: usize,
    output: usize,
    padded: usize,
    pad_columns: bool,
) -> Result<VisionLinearBuffers, BackendError> {
    let host = if pad_columns {
        pad_weight_columns(&source.weight, input, output, padded)?
    } else {
        pad_weight_rows(&source.weight, input, output, padded)?
    };
    Ok(VisionLinearBuffers {
        weight: DeviceBuffer::from_host(stream, &host).map_err(infer)?,
        input_min: source.input_min,
        input_max: source.input_max,
        output_min: source.output_min,
        output_max: source.output_max,
    })
}

fn upload_weights(
    stream: &Arc<Stream>,
    weights: &Gemma4VisionWeights,
) -> Result<VisionWeightBuffers, BackendError> {
    let hidden = weights.config.hidden;
    let intermediate = weights.config.intermediate;
    let padded = weights.config.padded_width();
    let mut layers = Vec::new();
    for layer in &weights.layers {
        layers.push(VisionLayerBuffers {
            input_norm: upload_norm_source(stream, &layer.input_norm, hidden)?,
            post_attention_norm: upload_norm_source(stream, &layer.post_attention_norm, hidden)?,
            pre_feedforward_norm: upload_norm_source(stream, &layer.pre_feedforward_norm, hidden)?,
            post_feedforward_norm: upload_norm_source(
                stream,
                &layer.post_feedforward_norm,
                hidden,
            )?,
            q_norm: upload_norm_source(stream, &layer.q_norm, weights.config.head_dim)?,
            k_norm: upload_norm_source(stream, &layer.k_norm, weights.config.head_dim)?,
            q: upload_linear(stream, &layer.q, hidden, hidden, padded, false)?,
            k: upload_linear(stream, &layer.k, hidden, hidden, padded, false)?,
            v: upload_linear(stream, &layer.v, hidden, hidden, padded, false)?,
            o: upload_linear(stream, &layer.o, hidden, hidden, padded, true)?,
            gate: upload_linear(
                stream,
                &layer.gate,
                hidden,
                intermediate,
                intermediate,
                false,
            )?,
            up: upload_linear(stream, &layer.up, hidden, intermediate, intermediate, false)?,
            down: upload_linear(stream, &layer.down, intermediate, hidden, hidden, false)?,
        });
    }
    Ok(VisionWeightBuffers {
        patch: upload_plain(stream, &weights.patch)?,
        positions: upload_plain(stream, &weights.positions)?,
        projection: upload_plain(stream, &weights.projection)?,
        std_bias: weights
            .std_bias
            .as_ref()
            .map(|source| upload_norm_source(stream, source, hidden))
            .transpose()?,
        std_scale: weights
            .std_scale
            .as_ref()
            .map(|source| upload_norm_source(stream, source, hidden))
            .transpose()?,
        layers,
    })
}

struct VisionPlans {
    patch: LinearPlan<Bf16>,
    qkv: LinearPlan<Bf16>,
    o: LinearPlan<Bf16>,
    gate: LinearPlan<Bf16>,
    up: LinearPlan<Bf16>,
    down: LinearPlan<Bf16>,
    projection: LinearPlan<Bf16>,
}

impl VisionPlans {
    fn new(
        config: &Gemma4VisionConfig,
        geometry: VisionBatchGeometry,
    ) -> Result<Self, BackendError> {
        let hidden = config.hidden;
        let intermediate = config.intermediate;
        let padded = config.padded_width();
        let patches = geometry.patches;
        let soft = geometry.soft_tokens;
        let packed = intermediate
            .checked_mul(2)
            .ok_or_else(|| infer("Gemma4 vision gate/up overflow"))?;
        let gate = LinearPlan::new(
            crate::tensor::Layout::contiguous([patches, hidden]).map_err(infer)?,
            crate::tensor::Layout::contiguous([intermediate, hidden]).map_err(infer)?,
            crate::tensor::Layout::strided([patches, intermediate], [packed, 1]).map_err(infer)?,
        )
        .map_err(infer)?;
        Ok(Self {
            patch: LinearPlan::contiguous(patches, config.patch_dim(), hidden).map_err(infer)?,
            qkv: LinearPlan::contiguous(patches, hidden, padded).map_err(infer)?,
            o: LinearPlan::contiguous(patches, padded, hidden).map_err(infer)?,
            gate,
            up: LinearPlan::new(
                crate::tensor::Layout::contiguous([patches, hidden]).map_err(infer)?,
                crate::tensor::Layout::contiguous([intermediate, hidden]).map_err(infer)?,
                crate::tensor::Layout::strided([patches, intermediate], [packed, 1])
                    .map_err(infer)?,
            )
            .map_err(infer)?,
            down: LinearPlan::contiguous(patches, intermediate, hidden).map_err(infer)?,
            projection: LinearPlan::contiguous(soft, hidden, config.text_hidden).map_err(infer)?,
        })
    }
}

/// Concrete vision executor over released checkpoint weights. Construction is
/// cold and allocates only owned device weights plus kernel/FA4 handles.
pub(crate) struct Gemma4VisionExecutor {
    config: Gemma4VisionConfig,
    context: Arc<Context>,
    stream: Arc<Stream>,
    kernels: Gemma4VisionKernels,
    weights: VisionWeightBuffers,
    attention: Fa4,
}

impl Gemma4VisionExecutor {
    pub(crate) fn new(
        context: Arc<Context>,
        stream: Arc<Stream>,
        weights: &Gemma4VisionWeights,
        module_path: Option<&Path>,
        fa4_root: &Path,
    ) -> Result<Self, BackendError> {
        context.bind_to_thread().map_err(start)?;
        let kernels = Gemma4VisionKernels::load(&context, module_path)?;
        let spec = AttentionSpec {
            heads: VISION_PADDED_HEADS,
            kv_heads: VISION_PADDED_HEADS,
            head_dim: weights.config.head_dim,
            causal: false,
            window: Window::Global,
            dtype: DType::Bf16,
            scale: Scale::Explicit(1.0),
        };
        if !Fa4::supports(&spec) {
            return Err(start(
                "Gemma4 vision requires the canonical noncausal 16/16x64 FA4 profile",
            ));
        }
        let attention = Fa4::new(context.clone(), fa4_root, spec)?;
        let uploaded = upload_weights(&stream, weights)?;
        stream.synchronize().map_err(start)?;
        Ok(Self {
            config: weights.config.clone(),
            context,
            stream,
            kernels,
            weights: uploaded,
            attention,
        })
    }

    pub(crate) fn config(&self) -> &Gemma4VisionConfig {
        &self.config
    }

    pub(crate) fn context(&self) -> &Arc<Context> {
        &self.context
    }

    pub(crate) fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }

    /// Stage, enqueue, widen, and drain one full vision request. The caller
    /// retains `work` and the lease until this returns; the lease owns the
    /// returned F32 rows.
    pub(crate) fn encode(
        &self,
        blas: &Blas,
        work: &mut Gemma4VisionWork,
        images: &[Gemma4VisionImage<'_>],
        lease: &mut HostIoLease,
    ) -> Result<(), BackendError> {
        work.stage_pixels(&self.config, images, lease)?;
        self.enqueue(blas, work)?;
        self.finish(work, lease)?;
        lease.wait()
    }

    /// Validate ordered injection rows before any device upload.
    pub(crate) fn validate_scatter_rows(rows: &[u32], tokens: usize) -> Result<(), BackendError> {
        if let Some(row) = rows.iter().find(|row| **row as usize >= tokens) {
            return Err(infer(format!(
                "Gemma4 vision scatter row {row} exceeds {tokens} text tokens"
            )));
        }
        Ok(())
    }

    /// Scatter ordered image feature rows into a retained BF16 embedding arena.
    pub(crate) fn scatter_features(
        &self,
        work: &mut Gemma4VisionWork,
        embeddings: &mut DeviceBuffer<Bf16>,
        rows: &[u32],
        tokens: usize,
    ) -> Result<(), BackendError> {
        let expected = work.geometry.soft_tokens;
        if rows.len() != expected {
            return Err(infer("Gemma4 vision scatter row count differs"));
        }
        if rows.len() > work.scatter_rows.len() {
            return Err(infer("Gemma4 vision scatter arena is too short"));
        }
        Self::validate_scatter_rows(rows, tokens)?;
        let width = self.config.text_hidden;
        if embeddings.len() < product(tokens, width)? {
            return Err(infer("Gemma4 vision embedding arena is too short"));
        }
        work.scatter_rows
            .copy_from_host(&self.stream, rows)
            .map_err(infer)?;
        let mut destination = embeddings.cu_deviceptr();
        let mut features = work.projected.cu_deviceptr();
        let mut row_ptr = work.scatter_rows.cu_deviceptr();
        let mut row_count = u32_size(rows.len())?;
        let mut width = u32_size(width)?;
        launch(
            &self.kernels.scatter,
            &self.stream,
            blocks(product(rows.len(), self.config.text_hidden)?)?,
            0,
            &mut [
                arg(&mut destination),
                arg(&mut features),
                arg(&mut row_ptr),
                arg(&mut row_count),
                arg(&mut width),
            ],
        )?;
        self.stream.synchronize().map_err(infer)
    }

    #[allow(clippy::too_many_arguments)]
    fn norm(
        &self,
        kernel: &Function,
        input: u64,
        scale: u64,
        out: u64,
        rows: usize,
        width: usize,
        heads: usize,
        stride: usize,
        eps: f32,
    ) -> Result<(), BackendError> {
        let mut input = input;
        let mut scale = scale;
        let mut out = out;
        let mut rows = u32_size(rows)?;
        let mut width = u32_size(width)?;
        let mut heads = u32_size(heads)?;
        let mut stride = u64::try_from(stride).map_err(infer)?;
        let mut eps = eps;
        launch_threads(
            kernel,
            &self.stream,
            (rows, 1, 1),
            reference_threads(width as usize),
            0,
            &mut [
                arg(&mut input),
                arg(&mut scale),
                arg(&mut out),
                arg(&mut rows),
                arg(&mut width),
                arg(&mut heads),
                arg(&mut stride),
                arg(&mut eps),
            ],
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn unit_norm(
        &self,
        input: u64,
        out: u64,
        rows: usize,
        width: usize,
        heads: usize,
        stride: usize,
        eps: f32,
    ) -> Result<(), BackendError> {
        let mut input = input;
        let mut out = out;
        let mut rows = u32_size(rows)?;
        let mut width = u32_size(width)?;
        let mut heads = u32_size(heads)?;
        let mut stride = u64::try_from(stride).map_err(infer)?;
        let mut eps = eps;
        launch_threads(
            &self.kernels.unit_norm,
            &self.stream,
            (rows, 1, 1),
            reference_threads(width as usize),
            0,
            &mut [
                arg(&mut input),
                arg(&mut out),
                arg(&mut rows),
                arg(&mut width),
                arg(&mut heads),
                arg(&mut stride),
                arg(&mut eps),
            ],
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn clip(
        &self,
        input: u64,
        out: u64,
        rows: usize,
        width: usize,
        stride: usize,
        min: f32,
        max: f32,
    ) -> Result<(), BackendError> {
        let mut input = input;
        let mut out = out;
        let mut rows = u32_size(rows)?;
        let mut width = u32_size(width)?;
        let mut stride = u32_size(stride)?;
        let mut min = min;
        let mut max = max;
        launch(
            &self.kernels.clip_rows,
            &self.stream,
            blocks(product(rows as usize, width as usize)?)?,
            0,
            &mut [
                arg(&mut input),
                arg(&mut out),
                arg(&mut rows),
                arg(&mut width),
                arg(&mut stride),
                arg(&mut min),
                arg(&mut max),
            ],
        )
    }

    pub(crate) fn enqueue(
        &self,
        blas: &Blas,
        work: &mut Gemma4VisionWork,
    ) -> Result<(), BackendError> {
        if !Arc::ptr_eq(self.stream.context(), &self.context) {
            return Err(infer("Gemma4 vision stream context differs"));
        }
        self.context.bind_to_thread().map_err(infer)?;
        let config = &self.config;
        let geometry = work.geometry;
        let hidden = config.hidden;
        let intermediate = config.intermediate;
        let padded = config.padded_width();
        let patches = geometry.patches;
        let soft = geometry.soft_tokens;
        let layers = config.layers;
        let plans = VisionPlans::new(config, geometry)?;

        // 1. Patch projection and trained 2-D position embedding add.
        unsafe {
            blas.linear(
                &plans.patch,
                &work.pixels,
                &self.weights.patch,
                &mut work.state,
            )?;
        }
        let mut hidden_ptr = work.state.cu_deviceptr();
        let mut table = self.weights.positions.cu_deviceptr();
        let mut columns = work.columns.cu_deviceptr();
        let mut rows = work.rows.cu_deviceptr();
        let mut position_out = work.state.cu_deviceptr();
        let mut position_tokens = u32_size(patches)?;
        let mut position_hidden = u32_size(hidden)?;
        let mut position_size = u32_size(config.position_size)?;
        launch(
            &self.kernels.position_add,
            &self.stream,
            blocks(product(patches, hidden)?)?,
            0,
            &mut [
                arg(&mut hidden_ptr),
                arg(&mut table),
                arg(&mut columns),
                arg(&mut rows),
                arg(&mut position_out),
                arg(&mut position_tokens),
                arg(&mut position_hidden),
                arg(&mut position_size),
            ],
        )?;

        // 2. Compute per-token two-axis BF16 rotary tables once per request.
        let mut columns = work.columns.cu_deviceptr();
        let mut rows = work.rows.cu_deviceptr();
        let mut inverse = work.inverse.cu_deviceptr();
        let mut cosine = work.cosine.cu_deviceptr();
        let mut sine = work.sine.cu_deviceptr();
        let mut rope_tokens = u32_size(patches)?;
        let mut rope_pairs = u32_size(config.head_dim / 4)?;
        launch(
            &self.kernels.rope_tables,
            &self.stream,
            blocks(product(patches, config.head_dim / 4)?)?,
            0,
            &mut [
                arg(&mut columns),
                arg(&mut rows),
                arg(&mut inverse),
                arg(&mut cosine),
                arg(&mut sine),
                arg(&mut rope_tokens),
                arg(&mut rope_pairs),
            ],
        )?;

        for (index, layer) in self.weights.layers.iter().enumerate() {
            if index >= layers {
                return Err(infer("Gemma4 vision layer inventory overflow"));
            }
            // Attention input norm.
            self.norm(
                &self.kernels.norm,
                work.state.cu_deviceptr(),
                layer.input_norm.cu_deviceptr(),
                work.normalized.cu_deviceptr(),
                patches,
                hidden,
                1,
                hidden,
                config.eps,
            )?;
            // Clipped Q/K/V projections into padded width.
            for (linear, destination) in [
                (&layer.q, &mut work.q),
                (&layer.k, &mut work.k),
                (&layer.v, &mut work.v),
            ] {
                self.clip(
                    work.normalized.cu_deviceptr(),
                    work.clipped.cu_deviceptr(),
                    patches,
                    hidden,
                    hidden,
                    linear.input_min,
                    linear.input_max,
                )?;
                unsafe {
                    blas.linear(&plans.qkv, &work.clipped, &linear.weight, destination)?;
                }
                self.clip(
                    destination.cu_deviceptr(),
                    destination.cu_deviceptr(),
                    patches,
                    hidden,
                    padded,
                    linear.output_min,
                    linear.output_max,
                )?;
            }
            // Q/K norm with trained scales; V is a unit RMS norm.
            self.norm(
                &self.kernels.norm,
                work.q.cu_deviceptr(),
                layer.q_norm.cu_deviceptr(),
                work.q.cu_deviceptr(),
                patches * VISION_PADDED_HEADS,
                config.head_dim,
                VISION_PADDED_HEADS,
                padded,
                config.eps,
            )?;
            self.norm(
                &self.kernels.norm,
                work.k.cu_deviceptr(),
                layer.k_norm.cu_deviceptr(),
                work.k.cu_deviceptr(),
                patches * VISION_PADDED_HEADS,
                config.head_dim,
                VISION_PADDED_HEADS,
                padded,
                config.eps,
            )?;
            self.unit_norm(
                work.v.cu_deviceptr(),
                work.v.cu_deviceptr(),
                patches * VISION_PADDED_HEADS,
                config.head_dim,
                VISION_PADDED_HEADS,
                padded,
                config.eps,
            )?;
            // Two-axis rotary on Q and K only, into a separate output arena.
            let rope_total = product(patches, padded)?;
            for source in [&mut work.q, &mut work.k] {
                let mut input = source.cu_deviceptr();
                let mut out = work.rope_out.cu_deviceptr();
                let mut cos = work.cosine.cu_deviceptr();
                let mut sin = work.sine.cu_deviceptr();
                let mut elements = u32_size(rope_total)?;
                let mut heads = u32_size(VISION_PADDED_HEADS)?;
                let mut dim = u32_size(config.head_dim)?;
                let mut token_stride = u64::try_from(padded).map_err(infer)?;
                let mut head_stride = u64::try_from(config.head_dim).map_err(infer)?;
                launch(
                    &self.kernels.rope2d,
                    &self.stream,
                    blocks(rope_total)?,
                    0,
                    &mut [
                        arg(&mut input),
                        arg(&mut cos),
                        arg(&mut sin),
                        arg(&mut out),
                        arg(&mut elements),
                        arg(&mut heads),
                        arg(&mut dim),
                        arg(&mut token_stride),
                        arg(&mut head_stride),
                    ],
                )?;
                std::mem::swap(source, &mut work.rope_out);
            }
            // Packed bidirectional attention, one sequence per image.
            let attention = AttentionBatch {
                q: work.q.cu_deviceptr(),
                k: work.k.cu_deviceptr(),
                v: work.v.cu_deviceptr(),
                out: work.attention.cu_deviceptr(),
                q_offsets: work.cumulative.cu_deviceptr(),
                k_offsets: work.cumulative.cu_deviceptr(),
                q_tokens: patches,
                k_tokens: patches,
                batch: geometry.images,
                max_q: geometry.max_patches,
                strides: [
                    padded as i64,
                    config.head_dim as i64,
                    1,
                    padded as i64,
                    config.head_dim as i64,
                    1,
                    padded as i64,
                    config.head_dim as i64,
                    1,
                    padded as i64,
                    config.head_dim as i64,
                    1,
                ],
            };
            self.context.bind_to_thread().map_err(infer)?;
            // SAFETY: Work retains Q/K/V/output and boundaries; all pointers
            // belong to this context and are non-aliasing.
            unsafe {
                self.attention
                    .forward_batch_bound(&self.stream, &attention)?
            };
            // Output projection with trained clipping and residual norm.
            self.clip(
                work.attention.cu_deviceptr(),
                work.attention.cu_deviceptr(),
                patches,
                hidden,
                padded,
                layer.o.input_min,
                layer.o.input_max,
            )?;
            unsafe {
                blas.linear(&plans.o, &work.attention, &layer.o.weight, &mut work.branch)?;
            }
            self.clip(
                work.branch.cu_deviceptr(),
                work.branch.cu_deviceptr(),
                patches,
                hidden,
                hidden,
                layer.o.output_min,
                layer.o.output_max,
            )?;
            let mut branch = work.branch.cu_deviceptr();
            let mut residual = work.state.cu_deviceptr();
            let mut scale = layer.post_attention_norm.cu_deviceptr();
            let mut out = work.state.cu_deviceptr();
            let mut rows = u32_size(patches)?;
            let mut width = u32_size(hidden)?;
            let mut eps = config.eps;
            launch_threads(
                &self.kernels.residual_norm,
                &self.stream,
                (rows, 1, 1),
                reference_threads(hidden),
                0,
                &mut [
                    arg(&mut branch),
                    arg(&mut residual),
                    arg(&mut scale),
                    arg(&mut out),
                    arg(&mut rows),
                    arg(&mut width),
                    arg(&mut eps),
                ],
            )?;
            // Feed-forward norm.
            self.norm(
                &self.kernels.norm,
                work.state.cu_deviceptr(),
                layer.pre_feedforward_norm.cu_deviceptr(),
                work.normalized.cu_deviceptr(),
                patches,
                hidden,
                1,
                hidden,
                config.eps,
            )?;
            // Gated MLP with trained clipped projections.
            self.clip(
                work.normalized.cu_deviceptr(),
                work.clipped.cu_deviceptr(),
                patches,
                hidden,
                hidden,
                layer.gate.input_min,
                layer.gate.input_max,
            )?;
            unsafe {
                blas.linear_subrange(
                    &plans.gate,
                    &work.clipped,
                    0,
                    &layer.gate.weight,
                    0,
                    &mut work.gate_up,
                    0,
                )?;
            }
            self.clip(
                work.gate_up.cu_deviceptr(),
                work.gate_up.cu_deviceptr(),
                patches,
                intermediate,
                intermediate * 2,
                layer.gate.output_min,
                layer.gate.output_max,
            )?;
            self.clip(
                work.normalized.cu_deviceptr(),
                work.clipped.cu_deviceptr(),
                patches,
                hidden,
                hidden,
                layer.up.input_min,
                layer.up.input_max,
            )?;
            unsafe {
                blas.linear_subrange(
                    &plans.up,
                    &work.clipped,
                    0,
                    &layer.up.weight,
                    0,
                    &mut work.gate_up,
                    intermediate,
                )?;
            }
            self.clip(
                work.gate_up
                    .cu_deviceptr()
                    .checked_add((intermediate * 2) as u64)
                    .ok_or_else(|| infer("Gemma4 vision gate/up offset overflow"))?,
                work.gate_up
                    .cu_deviceptr()
                    .checked_add((intermediate * 2) as u64)
                    .ok_or_else(|| infer("Gemma4 vision gate/up offset overflow"))?,
                patches,
                intermediate,
                intermediate * 2,
                layer.up.output_min,
                layer.up.output_max,
            )?;
            let mut gate_up = work.gate_up.cu_deviceptr();
            let mut active = work.activation.cu_deviceptr();
            let mut active_rows = u32_size(patches)?;
            let mut active_width = u32_size(intermediate)?;
            launch(
                &self.kernels.gated_gelu,
                &self.stream,
                blocks(product(patches, intermediate)?)?,
                0,
                &mut [
                    arg(&mut gate_up),
                    arg(&mut active),
                    arg(&mut active_rows),
                    arg(&mut active_width),
                ],
            )?;
            self.clip(
                work.activation.cu_deviceptr(),
                work.activation.cu_deviceptr(),
                patches,
                intermediate,
                intermediate,
                layer.down.input_min,
                layer.down.input_max,
            )?;
            unsafe {
                blas.linear(
                    &plans.down,
                    &work.activation,
                    &layer.down.weight,
                    &mut work.branch,
                )?;
            }
            self.clip(
                work.branch.cu_deviceptr(),
                work.branch.cu_deviceptr(),
                patches,
                hidden,
                hidden,
                layer.down.output_min,
                layer.down.output_max,
            )?;
            let mut branch = work.branch.cu_deviceptr();
            let mut residual = work.state.cu_deviceptr();
            let mut scale = layer.post_feedforward_norm.cu_deviceptr();
            let mut out = work.state.cu_deviceptr();
            let mut rows = u32_size(patches)?;
            let mut width = u32_size(hidden)?;
            let mut eps = config.eps;
            launch_threads(
                &self.kernels.residual_norm,
                &self.stream,
                (rows, 1, 1),
                reference_threads(hidden),
                0,
                &mut [
                    arg(&mut branch),
                    arg(&mut residual),
                    arg(&mut scale),
                    arg(&mut out),
                    arg(&mut rows),
                    arg(&mut width),
                    arg(&mut eps),
                ],
            )?;
        }

        // 3. Spatial pool per image, then final unit norm and projection.
        let mut patch_offset = 0usize;
        let mut soft_offset = 0usize;
        for image in &work.image_grids {
            let [_, height, width] = *image;
            let image_patches = height * width;
            let image_soft = config.soft_tokens(*image)?;
            let mut input = work
                .state
                .cu_deviceptr()
                .checked_add(u64::try_from(patch_offset * hidden * 2).map_err(infer)?)
                .ok_or_else(|| infer("Gemma4 vision pool input overflow"))?;
            let mut out = work
                .pooled
                .cu_deviceptr()
                .checked_add(u64::try_from(soft_offset * hidden * 2).map_err(infer)?)
                .ok_or_else(|| infer("Gemma4 vision pool output overflow"))?;
            let mut height_arg = u32_size(height)?;
            let mut width_arg = u32_size(width)?;
            let mut hidden_arg = u32_size(hidden)?;
            let mut kernel_arg = u32_size(config.pool)?;
            let mut inverse = 1.0f32 / (config.pool * config.pool) as f32;
            let mut scale = (hidden as f32).sqrt();
            if config.standardize {
                let mut bias = self
                    .weights
                    .std_bias
                    .as_ref()
                    .ok_or_else(|| infer("Gemma4 vision std_bias missing"))?
                    .cu_deviceptr();
                let mut std_scale = self
                    .weights
                    .std_scale
                    .as_ref()
                    .ok_or_else(|| infer("Gemma4 vision std_scale missing"))?
                    .cu_deviceptr();
                launch(
                    &self.kernels.pool_standardize,
                    &self.stream,
                    blocks(product(image_soft, hidden)?)?,
                    0,
                    &mut [
                        arg(&mut input),
                        arg(&mut bias),
                        arg(&mut std_scale),
                        arg(&mut out),
                        arg(&mut height_arg),
                        arg(&mut width_arg),
                        arg(&mut hidden_arg),
                        arg(&mut kernel_arg),
                        arg(&mut inverse),
                        arg(&mut scale),
                    ],
                )?;
            } else {
                launch(
                    &self.kernels.pool,
                    &self.stream,
                    blocks(product(image_soft, hidden)?)?,
                    0,
                    &mut [
                        arg(&mut input),
                        arg(&mut out),
                        arg(&mut height_arg),
                        arg(&mut width_arg),
                        arg(&mut hidden_arg),
                        arg(&mut kernel_arg),
                        arg(&mut inverse),
                        arg(&mut scale),
                    ],
                )?;
            }
            patch_offset += image_patches;
            soft_offset += image_soft;
        }
        self.unit_norm(
            work.pooled.cu_deviceptr(),
            work.normalized.cu_deviceptr(),
            soft,
            hidden,
            1,
            hidden,
            config.eps,
        )?;
        unsafe {
            blas.linear(
                &plans.projection,
                &work.normalized,
                &self.weights.projection,
                &mut work.projected,
            )?;
        }
        Ok(())
    }

    /// Widen the projected BF16 rows into the lease's F32 output arena and
    /// record one completion event. The caller must retain the work until the
    /// lease is waited/drained.
    pub(crate) fn finish(
        &self,
        work: &Gemma4VisionWork,
        lease: &mut HostIoLease,
    ) -> Result<(), BackendError> {
        let elements = product(work.geometry.soft_tokens, self.config.text_hidden)?;
        let mut input = work.projected.cu_deviceptr();
        let mut out = work.f32_out.cu_deviceptr();
        let mut selected = 0u64;
        let mut count = u32_size(work.geometry.soft_tokens)?;
        let mut width = u32_size(self.config.text_hidden)?;
        launch(
            &self.kernels.bf16_to_f32,
            &self.stream,
            blocks(elements)?,
            0,
            &mut [
                arg(&mut input),
                arg(&mut out),
                arg(&mut selected),
                arg(&mut count),
                arg(&mut width),
            ],
        )?;
        {
            let output = lease.slot_mut()?.output.as_mut_slice();
            let slice = output
                .get_mut(..elements)
                .ok_or_else(|| infer("Gemma4 vision output arena is too short"))?;
            // SAFETY: the lease owns a pinned output arena of at least
            // `elements` F32 values; the work retains the source until the
            // recorded event completes.
            unsafe {
                cuda_core::memory::memcpy_dtoh_async(
                    slice.as_mut_ptr(),
                    work.f32_out.cu_deviceptr(),
                    elements * 4,
                    self.stream.cu_stream(),
                )
            }
            .map_err(infer)?;
        }
        lease.record()
    }
}

/// Model-owned host I/O for vision: one bounded pinned pool and the exact
/// completion stream. Leases are returned to the pool after `wait`.
pub(crate) struct Gemma4VisionOwnedIo {
    pool: Arc<crate::staging::SlotPool<crate::staging::HostIo>>,
    stream: Arc<Stream>,
}

impl Gemma4VisionOwnedIo {
    pub(crate) fn new(context: &Arc<Context>, stream: &Arc<Stream>) -> Result<Self, BackendError> {
        Ok(Self {
            pool: crate::staging::HostIo::pool(context)?,
            stream: stream.clone(),
        })
    }

    pub(crate) fn lease(&self) -> Result<HostIoLease, BackendError> {
        let slot = self.pool.acquire()?;
        Ok(HostIoLease::owned(
            slot,
            self.pool.clone(),
            self.stream.clone(),
        ))
    }
}

/// Cold engine holding the uploaded vision tower, FA4 provider, plain GEMM
/// provider, and model-owned I/O. One instance owns no request arena; callers
/// retain a [`Gemma4VisionWork`] for reuse across images of the same geometry.
pub(crate) struct Gemma4VisionEngine {
    executor: Gemma4VisionExecutor,
    owned: Gemma4VisionOwnedIo,
    blas: Blas,
}

impl Gemma4VisionEngine {
    pub(crate) fn new(
        context: Arc<Context>,
        stream: Arc<Stream>,
        weights: &Gemma4VisionWeights,
        module_path: Option<&Path>,
        fa4_root: &Path,
    ) -> Result<Self, BackendError> {
        let executor = Gemma4VisionExecutor::new(
            context.clone(),
            stream.clone(),
            weights,
            module_path,
            fa4_root,
        )?;
        let blas = Blas::new(context.clone(), stream.clone())?;
        let owned = Gemma4VisionOwnedIo::new(&context, &stream)?;
        Ok(Self {
            executor,
            owned,
            blas,
        })
    }

    pub(crate) fn executor(&self) -> &Gemma4VisionExecutor {
        &self.executor
    }

    /// Encode one packed image request and return one F32 feature matrix per
    /// image in input order. The work arena is released after the lease drains.
    pub(crate) fn encode_images(
        &self,
        images: &[Gemma4VisionImage<'_>],
    ) -> Result<Vec<Vec<f32>>, BackendError> {
        let config = self.executor.config();
        let mut work = Gemma4VisionWork::new(self.executor.stream().clone(), config, images)?;
        let mut lease = self.owned.lease()?;
        self.executor
            .encode(&self.blas, &mut work, images, &mut lease)?;
        let elements = product(work.geometry().soft_tokens, config.text_hidden)?;
        let output = lease.output(elements)?.to_vec();
        let mut result = Vec::with_capacity(images.len());
        let mut offset = 0usize;
        for image in images {
            let rows = config.soft_tokens(image.grid_thw)?;
            let count = product(rows, config.text_hidden)?;
            result.push(output[offset..offset + count].to_vec());
            offset += count;
        }
        Ok(result)
    }

    /// Encode into a retained Work whose projected BF16 rows can be injected
    /// into the text embedding arena on the same ordered stream.
    pub(crate) fn encode_request(
        &self,
        blas: &Blas,
        images: &[Gemma4VisionImage<'_>],
    ) -> Result<Gemma4VisionWork, BackendError> {
        let config = self.executor.config();
        let mut work = Gemma4VisionWork::new(self.executor.stream().clone(), config, images)?;
        let mut lease = self.owned.lease()?;
        self.executor.encode(blas, &mut work, images, &mut lease)?;
        Ok(work)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn released_e2b_geometry_is_admitted_and_exact() {
        let raw = std::fs::read(
            "/workspace/model-performance/michaelfeil/onejev-perf/backbone-comparison-20261023/models/gemma-4-E2B/config.json",
        )
        .unwrap();
        let value: Value = serde_json::from_slice(&raw).unwrap();
        let config = Gemma4VisionConfig::from_model_json(&value).unwrap();
        assert_eq!(config.hidden, 768);
        assert_eq!(config.intermediate, 3072);
        assert_eq!(config.layers, 16);
        assert_eq!(config.heads, 12);
        assert_eq!(config.kv_heads, 12);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.padded_width(), 1024);
        assert_eq!(config.text_hidden, 1536);
        assert_eq!(config.inventory_len(), 659);
        assert!(config.use_clipped_linears);
        assert_eq!(config.inverse_frequencies().len(), 16);
    }

    #[test]
    fn rune_26b_vision_geometry_admits_d72_standardize() {
        let path = std::env::var_os("INFINITY_GEMMA4_RUNE_CONFIG")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from(
                    "/tmp/oxide-docs-parity-20261102/model-metadata/michaelfeil--rune-26b-a4b",
                )
            });
        let raw = match std::fs::read(path.join("config.json")) {
            Ok(raw) => raw,
            Err(_) => return,
        };
        let value: Value = serde_json::from_slice(&raw).unwrap();
        let config = Gemma4VisionConfig::from_model_json(&value).unwrap();
        assert_eq!(config.hidden, 1152);
        assert_eq!(config.head_dim, 72);
        assert_eq!(config.heads, 16);
        assert_eq!(config.layers, 27);
        assert_eq!(config.intermediate, 4304);
        assert!(config.standardize);
        assert!(!config.use_clipped_linears);
        assert_eq!(config.text_hidden, 2816);
        assert_eq!(config.inverse_frequencies().len(), 18);
        assert_eq!(config.inventory_len(), 3 + 27 * 13 + 2);
    }

    #[test]
    fn cold_geometry_rejections_are_typed() {
        let raw = std::fs::read(
            "/workspace/model-performance/michaelfeil/onejev-perf/backbone-comparison-20261023/models/gemma-4-E2B/config.json",
        )
        .unwrap();
        let value: Value = serde_json::from_slice(&raw).unwrap();
        for (path, replacement) in [
            (vec!["vision_config", "patch_size"], Value::from(14)),
            (vec!["vision_config", "pooling_kernel_size"], Value::from(2)),
            (vec!["vision_config", "head_dim"], Value::from(32)),
            (vec!["vision_config", "num_key_value_heads"], Value::from(4)),
            (
                vec!["vision_config", "hidden_activation"],
                Value::from("silu"),
            ),
        ] {
            let mut altered = value.clone();
            let mut cursor = &mut altered;
            for key in &path[..path.len() - 1] {
                cursor = &mut cursor[*key];
            }
            cursor[path[path.len() - 1]] = replacement;
            assert!(
                Gemma4VisionConfig::from_model_json(&altered).is_err(),
                "{path:?}"
            );
        }
    }

    #[test]
    fn packed_patch_positions_are_row_major_and_resume() {
        let a = vec![0.0f32; 3 * 4 * 768];
        let b = vec![0.0f32; 6 * 2 * 768];
        let images = [
            Gemma4VisionImage {
                pixels: &a,
                grid_thw: [1, 3, 4],
            },
            Gemma4VisionImage {
                pixels: &b,
                grid_thw: [1, 6, 2],
            },
        ];
        let (columns, rows, cumulative) = patch_positions(&images).unwrap();
        assert_eq!(cumulative, [0, 12, 24]);
        assert_eq!(&columns[..4], &[0, 1, 2, 3]);
        assert_eq!(&rows[..4], &[0, 0, 0, 0]);
        assert_eq!(columns[11], 3);
        assert_eq!(rows[11], 2);
        assert_eq!(&columns[12..16], &[0, 1, 0, 1]);
        assert_eq!(&rows[12..16], &[0, 0, 1, 1]);
    }

    #[test]
    fn bf16_rounding_matches_half() {
        assert_eq!(bf16_round(1.0), 1.0);
        assert_eq!(bf16_value(bf16_bits(-3.5)), -3.5);
        assert_eq!(bf16_round(1.003_906_3), 1.0);
        assert_eq!(bf16_round(1.0078125), 1.0 + 1.0 / 128.0);
    }

    #[test]
    fn injection_maps_placeholders_in_order() {
        let width = 2;
        let features = vec![vec![1u16, 2, 3, 4], vec![5u16, 6, 7, 8, 9, 10]];
        let ids = [7u32, 9, 9, 8, 9, 9, 9];
        let mut embeddings = vec![0u16; ids.len() * width];
        inject_features_host(&mut embeddings, &ids, 9, &features, width).unwrap();
        assert_eq!(embeddings, vec![0, 0, 1, 2, 3, 4, 0, 0, 5, 6, 7, 8, 9, 10]);
        let mut too_few = vec![0u16; 2 * width];
        assert!(inject_features_host(&mut too_few, &[9, 9, 9], 9, &features, width).is_err());
    }

    #[test]
    fn released_checkpoint_inventory_is_complete() {
        let root = std::path::Path::new(
            "/workspace/model-performance/michaelfeil/onejev-perf/backbone-comparison-20261023/models/gemma-4-E2B",
        );
        if !root.join("model.safetensors").exists() {
            return;
        }
        let weights = Gemma4VisionWeights::load(root).unwrap();
        assert_eq!(weights.checkpoint_tensors, 2011);
        assert_eq!(weights.layers.len(), 16);
        assert_eq!(weights.config.inventory_len(), 659);
        assert_eq!(weights.patch.shape(), &[768, 768]);
        assert_eq!(weights.positions.shape(), &[2, 10240, 768]);
        assert_eq!(weights.projection.shape(), &[1536, 768]);
        assert_eq!(weights.layers[0].q.weight.shape(), &[768, 768]);
        assert_eq!(weights.layers[0].gate.weight.shape(), &[3072, 768]);
        assert_eq!(weights.layers[0].down.weight.shape(), &[768, 3072]);
        assert!(weights.layers[0].q.input_min < 0.0);
        assert!(weights.layers[0].q.input_max > 0.0);
    }

    #[test]
    fn reference_forward_is_deterministic_and_finite_on_a_small_grid() {
        let root = std::path::Path::new(
            "/workspace/model-performance/michaelfeil/onejev-perf/backbone-comparison-20261023/models/gemma-4-E2B",
        );
        if !root.join("model.safetensors").exists() {
            return;
        }
        // Component-scale reference uses only the released geometry, not the
        // full checkpoint payload, so the CPU gate stays offline and bounded.
        let value: Value =
            serde_json::from_slice(&std::fs::read(root.join("config.json")).unwrap()).unwrap();
        let mut config = Gemma4VisionConfig::from_model_json(&value).unwrap();
        config.layers = 1;
        config.intermediate = 64;
        let hidden = config.hidden;
        let patch_dim = config.patch_dim();
        let weights = HostVisionWeights {
            patch: (0..hidden * patch_dim)
                .map(|i| ((i % 17) as f32 - 8.0) / 64.0)
                .collect(),
            positions: vec![0.0; 2 * config.position_size * hidden],
            projection: (0..config.text_hidden * hidden)
                .map(|i| ((i % 13) as f32 - 6.0) / 64.0)
                .collect(),
            layers: vec![HostVisionLayer {
                input_norm: vec![1.0; hidden],
                post_attention_norm: vec![1.0; hidden],
                pre_feedforward_norm: vec![1.0; hidden],
                post_feedforward_norm: vec![1.0; hidden],
                q_norm: vec![1.0; config.head_dim],
                k_norm: vec![1.0; config.head_dim],
                q: HostLinear {
                    weight: (0..hidden * hidden)
                        .map(|i| ((i % 7) as f32 - 3.0) / 128.0)
                        .collect(),
                    input_min: -4.0,
                    input_max: 4.0,
                    output_min: -8.0,
                    output_max: 8.0,
                },
                k: HostLinear {
                    weight: (0..hidden * hidden)
                        .map(|i| ((i % 5) as f32 - 2.0) / 128.0)
                        .collect(),
                    input_min: -4.0,
                    input_max: 4.0,
                    output_min: -8.0,
                    output_max: 8.0,
                },
                v: HostLinear {
                    weight: (0..hidden * hidden)
                        .map(|i| ((i % 11) as f32 - 5.0) / 128.0)
                        .collect(),
                    input_min: -4.0,
                    input_max: 4.0,
                    output_min: -8.0,
                    output_max: 8.0,
                },
                o: HostLinear {
                    weight: (0..hidden * config.padded_width())
                        .map(|i| ((i % 3) as f32 - 1.0) / 128.0)
                        .collect(),
                    input_min: -8.0,
                    input_max: 8.0,
                    output_min: -8.0,
                    output_max: 8.0,
                },
                gate: HostLinear {
                    weight: (0..64 * hidden)
                        .map(|i| ((i % 9) as f32 - 4.0) / 128.0)
                        .collect(),
                    input_min: -4.0,
                    input_max: 4.0,
                    output_min: -8.0,
                    output_max: 8.0,
                },
                up: HostLinear {
                    weight: (0..64 * hidden)
                        .map(|i| ((i % 6) as f32 - 3.0) / 128.0)
                        .collect(),
                    input_min: -4.0,
                    input_max: 4.0,
                    output_min: -8.0,
                    output_max: 8.0,
                },
                down: HostLinear {
                    weight: (0..hidden * 64)
                        .map(|i| ((i % 5) as f32 - 2.0) / 128.0)
                        .collect(),
                    input_min: -8.0,
                    input_max: 8.0,
                    output_min: -8.0,
                    output_max: 8.0,
                },
            }],
        };
        let pixels: Vec<f32> = (0..3 * 3 * patch_dim)
            .map(|i| (i % 256) as f32 / 255.0)
            .collect();
        let image = Gemma4VisionImage {
            pixels: &pixels,
            grid_thw: [1, 3, 3],
        };
        let first = reference_forward(&config, &weights, image).unwrap();
        let second = reference_forward(&config, &weights, image).unwrap();
        assert_eq!(first, second);
        assert!(first.iter().all(|value| value.is_finite()));
        assert_eq!(first.len(), config.text_hidden);
    }
    #[test]
    fn production_preprocessor_fixture_matches_pinned_contract() {
        use sha2::{Digest, Sha256};
        let meta: Value =
            serde_json::from_str(include_str!("../tests/fixtures/gemma4_vision/fixture.json"))
                .unwrap();
        assert_eq!(meta["schema"], "gemma4-vision-real-image-fixture/v1");
        assert_eq!(meta["source_image"], "cats_coco_sample.jpg");
        assert_eq!(meta["patch_dim"], 768);
        assert_eq!(meta["merge_size"], 3);
        assert_eq!(meta["patch_count"], 2394);
        assert_eq!(meta["soft_tokens"], 266);
        let bytes: &[u8] = include_bytes!("../tests/fixtures/gemma4_vision/pixels.f32le");
        assert_eq!(bytes.len(), 2394 * 768 * 4);
        assert_eq!(
            format!("{:x}", Sha256::digest(bytes)),
            meta["pixels_f32le_sha256"].as_str().unwrap()
        );
        let pixels: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert!(pixels
            .iter()
            .all(|value| value.is_finite() && (0.0..=1.0).contains(value)));
        let root = std::path::Path::new(
            "/workspace/model-performance/michaelfeil/onejev-perf/backbone-comparison-20261023/models/gemma-4-E2B",
        );
        if !root.join("config.json").exists() {
            return;
        }
        let config = Gemma4VisionConfig::load(root).unwrap();
        let image = Gemma4VisionImage {
            pixels: &pixels,
            grid_thw: [1, 42, 57],
        };
        let geometry = admit_images(&config, &[image]).unwrap();
        assert_eq!(geometry.patches, 2394);
        assert_eq!(geometry.soft_tokens, 266);
        let (_, _, cumulative) = patch_positions(&[image]).unwrap();
        assert_eq!(cumulative, [0, 2394]);
    }

    /// Actual release binary + H100 gate. The Root controller supplies the
    /// pinned module and canonical FA4 packet; the author never runs this.
    #[test]
    #[ignore = "Root GPU gate; requires an actual H100 and pinned module/FA4 artifacts"]
    fn root_actual_checkpoint_vision_matches_cpu_reference_and_repeats() {
        use sha2::{Digest, Sha256};
        if std::env::var_os("INFINITY_GEMMA4_VISION_ROOT").is_none() {
            return;
        }
        let checkpoint = std::env::var("INFINITY_GEMMA4_E2B_ROOT").unwrap_or_else(|_| {
            "/workspace/model-performance/michaelfeil/onejev-perf/backbone-comparison-20261023/models/gemma-4-E2B".to_owned()
        });
        let module = PathBuf::from(
            std::env::var_os("INFINITY_OXIDE_KERNEL_MODULE")
                .expect("Root vision gate requires INFINITY_OXIDE_KERNEL_MODULE"),
        );
        let fa4 = PathBuf::from(
            std::env::var_os("INFINITY_FA4_ROOT")
                .expect("Root vision gate requires INFINITY_FA4_ROOT"),
        );
        let output_dir = PathBuf::from(
            std::env::var_os("INFINITY_GEMMA4_VISION_OUTPUT")
                .expect("Root vision gate requires INFINITY_GEMMA4_VISION_OUTPUT"),
        );
        std::fs::create_dir_all(&output_dir).unwrap();
        let weights = Gemma4VisionWeights::load(Path::new(&checkpoint)).unwrap();
        let config = weights.config.clone();
        let host = HostVisionWeights::read(&weights).unwrap();
        let bytes: &[u8] = include_bytes!("../tests/fixtures/gemma4_vision/pixels.f32le");
        let full_pixels: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        let patch = config.patch_dim();
        let width = 57usize;
        let mut crop = Vec::with_capacity(9 * patch);
        for y in 0..3usize {
            for x in 0..3usize {
                let index = y * width + x;
                crop.extend_from_slice(&full_pixels[index * patch..(index + 1) * patch]);
            }
        }
        let small = Gemma4VisionImage {
            pixels: &crop,
            grid_thw: [1, 3, 3],
        };
        let expected = reference_forward(&config, &host, small).unwrap();
        let context = Context::new(0).expect("CUDA context");
        let stream = context.new_stream().expect("CUDA stream");
        let engine = Gemma4VisionEngine::new(
            context.clone(),
            stream.clone(),
            &weights,
            Some(&module),
            &fa4,
        )
        .expect("vision engine");
        let actual = engine.encode_images(&[small]).unwrap();
        assert_eq!(actual.len(), 1);
        let actual = &actual[0];
        assert_eq!(actual.len(), config.text_hidden);
        let again = engine.encode_images(&[small]).unwrap();
        assert_eq!(actual, &again[0], "vision repeat must be bit-identical");
        let mut a2 = 0.0f64;
        let mut b2 = 0.0f64;
        let mut dot = 0.0f64;
        let mut max_abs = 0.0f64;
        for (a, b) in expected.iter().zip(actual) {
            assert!(a.is_finite() && b.is_finite());
            let (a, b) = (*a as f64, *b as f64);
            a2 += a * a;
            b2 += b * b;
            dot += a * b;
            max_abs = max_abs.max((a - b).abs());
        }
        let cosine = dot / (a2.sqrt() * b2.sqrt());
        let relative = ((b2 + a2 - 2.0 * dot) / a2).sqrt();
        assert!(cosine >= 0.999, "cosine {cosine}");
        assert!(max_abs <= 0.25, "max_abs {max_abs}");
        assert!(relative <= 0.02, "relative_l2 {relative}");
        // Full production fixture on GPU only: finite, ordered, repeatable.
        let full = Gemma4VisionImage {
            pixels: &full_pixels,
            grid_thw: [1, 42, 57],
        };
        let full_out = engine.encode_images(&[full]).unwrap();
        assert_eq!(full_out.len(), 1);
        assert_eq!(full_out[0].len(), 266 * config.text_hidden);
        assert!(full_out[0].iter().all(|value| value.is_finite()));
        let full_again = engine.encode_images(&[full]).unwrap();
        assert_eq!(full_out[0], full_again[0]);
        let actual_bytes: Vec<u8> = actual
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let full_bytes: Vec<u8> = full_out[0]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        std::fs::write(output_dir.join("small.f32le"), &actual_bytes).unwrap();
        std::fs::write(output_dir.join("full.f32le"), &full_bytes).unwrap();
        let metadata = serde_json::json!({
            "schema": "gemma4-vision-root-candidate/v1",
            "status": "passed",
            "gpu_executed": true,
            "case": "cats-3x3-and-full",
            "checkpoint_tensors": weights.checkpoint_tensors,
            "vision_tensors": config.inventory_len(),
            "grid_thw": [1, 42, 57],
            "soft_tokens": 266,
            "small_grid_thw": [1, 3, 3],
            "small_output_sha256": format!("{:x}", Sha256::digest(&actual_bytes)),
            "full_output_sha256": format!("{:x}", Sha256::digest(&full_bytes)),
            "cpu_reference": "gemma4_vision_model::reference_forward",
            "cosine": cosine,
            "relative_l2": relative,
            "max_abs": max_abs,
            "repeat_bit_identity": true,
            "kernel_module": module,
            "fa4_root": fa4,
            "fixture_pixels_sha256": format!("{:x}", Sha256::digest(bytes)),
        });
        std::fs::write(
            output_dir.join("metadata.json"),
            serde_json::to_vec_pretty(&metadata).unwrap(),
        )
        .unwrap();
    }
    #[test]
    fn padded_projection_layouts_keep_trained_values_and_zero_tails() {
        let source = [1u16, 2, 3, 4];
        let rows = pad_rows_bits(&source, 2, 2, 4).unwrap();
        assert_eq!(rows, [1, 2, 3, 4, 0, 0, 0, 0]);
        let columns = pad_columns_bits(&source, 2, 2, 3).unwrap();
        assert_eq!(columns, [1, 2, 0, 3, 4, 0]);
        assert!(pad_rows_bits(&source, 2, 2, 1).is_err());
        assert!(pad_columns_bits(&source, 2, 2, 1).is_err());
    }

    #[test]
    fn image_admission_rejects_unsupported_patch_contracts() {
        let raw = std::fs::read(
            "/workspace/model-performance/michaelfeil/onejev-perf/backbone-comparison-20261023/models/gemma-4-E2B/config.json",
        )
        .unwrap();
        let config =
            Gemma4VisionConfig::from_model_json(&serde_json::from_slice(&raw).unwrap()).unwrap();
        let patch = config.patch_dim();
        let pixels = vec![0.0f32; 3 * 3 * patch];
        assert!(admit_images(&config, &[]).is_err());
        assert!(admit_images(
            &config,
            &[Gemma4VisionImage {
                pixels: &pixels[..pixels.len() - 1],
                grid_thw: [1, 3, 3]
            }]
        )
        .is_err());
        assert!(admit_images(
            &config,
            &[Gemma4VisionImage {
                pixels: &pixels,
                grid_thw: [2, 3, 3]
            }]
        )
        .is_err());
        assert!(admit_images(
            &config,
            &[Gemma4VisionImage {
                pixels: &pixels,
                grid_thw: [1, 4, 3]
            }]
        )
        .is_err());
        let nan = vec![f32::NAN; pixels.len()];
        assert!(admit_images(
            &config,
            &[Gemma4VisionImage {
                pixels: &nan,
                grid_thw: [1, 3, 3]
            }]
        )
        .is_err());
    }
    /// Heavy but CPU-only check of the real-checkpoint reference used by the
    /// Root gate. Ignored in ordinary admission; Root may run it before GPU.
    #[test]
    #[ignore = "heavy real-checkpoint CPU reference; no GPU required"]
    fn root_actual_checkpoint_cpu_reference_smoke() {
        let checkpoint = std::env::var("INFINITY_GEMMA4_E2B_ROOT").unwrap_or_else(|_| {
            "/workspace/model-performance/michaelfeil/onejev-perf/backbone-comparison-20261023/models/gemma-4-E2B".to_owned()
        });
        let weights = Gemma4VisionWeights::load(Path::new(&checkpoint)).unwrap();
        let config = weights.config.clone();
        let host = HostVisionWeights::read(&weights).unwrap();
        let bytes: &[u8] = include_bytes!("../tests/fixtures/gemma4_vision/pixels.f32le");
        let full: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        let patch = config.patch_dim();
        let width = 57usize;
        let mut crop = Vec::with_capacity(9 * patch);
        for y in 0..3usize {
            for x in 0..3usize {
                let index = y * width + x;
                crop.extend_from_slice(&full[index * patch..(index + 1) * patch]);
            }
        }
        let image = Gemma4VisionImage {
            pixels: &crop,
            grid_thw: [1, 3, 3],
        };
        let output = reference_forward(&config, &host, image).unwrap();
        assert_eq!(output.len(), config.text_hidden);
        assert!(output.iter().all(|value| value.is_finite()));
        assert!(output.iter().any(|value| *value != 0.0));
    }
    #[test]
    fn workspace_guard_counts_the_full_arena() {
        let raw = std::fs::read(
            "/workspace/model-performance/michaelfeil/onejev-perf/backbone-comparison-20261023/models/gemma-4-E2B/config.json",
        )
        .unwrap();
        let config =
            Gemma4VisionConfig::from_model_json(&serde_json::from_slice(&raw).unwrap()).unwrap();
        // A request the old estimator admitted at 2,133,109,584 bytes while
        // Work::new actually allocated more than the 2 GiB limit.
        assert!(workspace_bytes(&config, 61_254, 6_806, 1).unwrap() > VISION_WORKSPACE_LIMIT);
        assert!(workspace_bytes(&config, 2_394, 266, 1).unwrap() < VISION_WORKSPACE_LIMIT);
    }

    #[test]
    fn scatter_rows_are_bounds_checked() {
        assert!(Gemma4VisionExecutor::validate_scatter_rows(&[0, 3, 7], 8).is_ok());
        assert!(Gemma4VisionExecutor::validate_scatter_rows(&[0, 8], 8).is_err());
    }
}
