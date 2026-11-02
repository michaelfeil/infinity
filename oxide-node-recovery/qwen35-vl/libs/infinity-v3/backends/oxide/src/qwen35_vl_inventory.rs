// Cold inventory for the Qwen3.5 vision tower.
//
// Published conditional-generation checkpoints name every tower tensor under
// `model.visual.*`. `qwen3_vl_models.cpp` also accepts a canonical `visual.*`
// spelling but rejects an ambiguous mix. This module derives the exact expected
// names and shapes from config alone, so the original checkpoint can be
// validated without reading a single payload byte.

use crate::model::Checkpoint;
use crate::qwen35_vl_config::Qwen35VlConfig;
use infinity_v3_backend_core::BackendError;
use std::collections::BTreeSet;
use std::path::Path;

fn start(error: impl std::fmt::Display) -> BackendError {
    BackendError::Start(format!("Oxide qwen3.5-vl inventory: {error}"))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpectedTensor {
    pub name: String,
    pub shape: Vec<usize>,
}

fn tensor(name: impl Into<String>, shape: Vec<usize>) -> ExpectedTensor {
    ExpectedTensor {
        name: name.into(),
        shape,
    }
}

/// Squared vision attention width used by the fused QKV projection.
pub fn qkv_width(config: &Qwen35VlConfig) -> Result<usize, BackendError> {
    config
        .vision
        .hidden_size
        .checked_mul(3)
        .ok_or_else(|| start("vision qkv width overflow"))
}

/// All 153 (depth 12), 297 (depth 24) or 333 (depth 27) published tower
/// tensors, in deterministic layer order.
pub fn expected_vision_tensors(
    config: &Qwen35VlConfig,
) -> Result<Vec<ExpectedTensor>, BackendError> {
    let vision = &config.vision;
    let hidden = vision.hidden_size;
    let intermediate = vision.intermediate_size;
    let patch_dim = vision.patch_dim()?;
    let grouped = hidden
        .checked_mul(vision.merge_area()?)
        .ok_or_else(|| start("vision grouped width overflow"))?;
    let out_hidden = vision.out_hidden_size;
    let qkv = qkv_width(config)?;
    let mut tensors = Vec::with_capacity(9 + 12 * vision.depth);
    tensors.push(tensor(
        "model.visual.patch_embed.proj.weight",
        vec![
            hidden,
            vision.in_channels,
            vision.temporal_patch_size,
            vision.patch_size,
            vision.patch_size,
        ],
    ));
    debug_assert_eq!(
        patch_dim,
        vision.in_channels * vision.temporal_patch_size * vision.patch_size * vision.patch_size
    );
    tensors.push(tensor("model.visual.patch_embed.proj.bias", vec![hidden]));
    tensors.push(tensor(
        "model.visual.pos_embed.weight",
        vec![vision.num_position_embeddings, hidden],
    ));
    for layer in 0..vision.depth {
        let prefix = format!("model.visual.blocks.{layer}");
        tensors.push(tensor(format!("{prefix}.norm1.weight"), vec![hidden]));
        tensors.push(tensor(format!("{prefix}.norm1.bias"), vec![hidden]));
        tensors.push(tensor(
            format!("{prefix}.attn.qkv.weight"),
            vec![qkv, hidden],
        ));
        tensors.push(tensor(format!("{prefix}.attn.qkv.bias"), vec![qkv]));
        tensors.push(tensor(
            format!("{prefix}.attn.proj.weight"),
            vec![hidden, hidden],
        ));
        tensors.push(tensor(format!("{prefix}.attn.proj.bias"), vec![hidden]));
        tensors.push(tensor(format!("{prefix}.norm2.weight"), vec![hidden]));
        tensors.push(tensor(format!("{prefix}.norm2.bias"), vec![hidden]));
        tensors.push(tensor(
            format!("{prefix}.mlp.linear_fc1.weight"),
            vec![intermediate, hidden],
        ));
        tensors.push(tensor(
            format!("{prefix}.mlp.linear_fc1.bias"),
            vec![intermediate],
        ));
        tensors.push(tensor(
            format!("{prefix}.mlp.linear_fc2.weight"),
            vec![hidden, intermediate],
        ));
        tensors.push(tensor(
            format!("{prefix}.mlp.linear_fc2.bias"),
            vec![hidden],
        ));
    }
    tensors.push(tensor("model.visual.merger.norm.weight", vec![hidden]));
    tensors.push(tensor("model.visual.merger.norm.bias", vec![hidden]));
    tensors.push(tensor(
        "model.visual.merger.linear_fc1.weight",
        vec![grouped, grouped],
    ));
    tensors.push(tensor("model.visual.merger.linear_fc1.bias", vec![grouped]));
    tensors.push(tensor(
        "model.visual.merger.linear_fc2.weight",
        vec![out_hidden, grouped],
    ));
    tensors.push(tensor(
        "model.visual.merger.linear_fc2.bias",
        vec![out_hidden],
    ));
    for (name_index, _) in vision.deepstack_visual_indexes.iter().enumerate() {
        let prefix = format!("model.visual.deepstack_merger_list.{name_index}");
        tensors.push(tensor(format!("{prefix}.norm.weight"), vec![grouped]));
        tensors.push(tensor(format!("{prefix}.norm.bias"), vec![grouped]));
        tensors.push(tensor(
            format!("{prefix}.linear_fc1.weight"),
            vec![grouped, grouped],
        ));
        tensors.push(tensor(format!("{prefix}.linear_fc1.bias"), vec![grouped]));
        tensors.push(tensor(
            format!("{prefix}.linear_fc2.weight"),
            vec![out_hidden, grouped],
        ));
        tensors.push(tensor(
            format!("{prefix}.linear_fc2.bias"),
            vec![out_hidden],
        ));
    }
    Ok(tensors)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InventoryReport {
    pub vision_tensors: usize,
    pub text_tensors: usize,
    pub total_tensors: usize,
    pub deepstack_tensors: usize,
}

/// Validate tower metadata against an already-open checkpoint. Payloads remain
/// untouched, and text tensors are only counted so disjoint ownership is clear.
pub fn validate_vision_inventory(
    config: &Qwen35VlConfig,
    checkpoint: &Checkpoint,
) -> Result<InventoryReport, BackendError> {
    let expected = expected_vision_tensors(config)?;
    let mut seen = BTreeSet::new();
    for tensor in &expected {
        if !seen.insert(tensor.name.clone()) {
            return Err(start(format!(
                "derived duplicate vision tensor {}",
                tensor.name
            )));
        }
        let shape = checkpoint
            .shape(&tensor.name)
            .ok_or_else(|| start(format!("missing vision tensor {}", tensor.name)))?;
        if shape != tensor.shape.as_slice() {
            return Err(start(format!(
                "vision tensor {} has shape {shape:?}, expected {:?}",
                tensor.name, tensor.shape
            )));
        }
    }
    for name in checkpoint.names() {
        if name.starts_with("visual.") {
            return Err(start(format!(
                "ambiguous non-canonical visual namespace at {name}"
            )));
        }
        if name.starts_with("model.visual.") && !seen.contains(name) {
            return Err(start(format!("unexpected extra vision tensor {name}")));
        }
    }
    let mut text_tensors = 0usize;
    let mut total_tensors = 0usize;
    for name in checkpoint.names() {
        total_tensors += 1;
        if !name.starts_with("model.visual.") {
            text_tensors += 1;
        }
    }
    Ok(InventoryReport {
        vision_tensors: expected.len(),
        text_tensors,
        total_tensors,
        deepstack_tensors: 6 * config.vision.deepstack_visual_indexes.len(),
    })
}

pub fn validate_vision_inventory_path(
    config: &Qwen35VlConfig,
    path: &Path,
) -> Result<InventoryReport, BackendError> {
    let checkpoint = Checkpoint::open(path)?;
    validate_vision_inventory(config, &checkpoint)
}
