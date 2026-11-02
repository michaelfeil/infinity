// Expected tensor inventory and checkpoint binding for Qwen3-VL.
//
// The expected names and shapes are derived from the actual Native accessors:
// `model.visual.*` for the tower and `model.language_model.*` for the dense Qwen3
// text decoder. Optional biases are admitted only where the Native decoder
// accepts them. A tensor that is present but unused is reported, never silently
// dropped.

use crate::checkpoint::{start, CheckpointDtype, CheckpointMetadata, TensorMeta};
use crate::qwen3_vl_config::{Qwen3VlConfig, TextConfig};
use infinity_v3_backend_core::BackendError;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

pub const VISION_PREFIX: &str = "model.visual.";
pub const TEXT_PREFIX: &str = "model.language_model.";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum TensorKind {
    Vision,
    Deepstack,
    Text,
    OptionalTextBias,
    Readout,
}

impl TensorKind {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Vision => "vision",
            Self::Deepstack => "deepstack",
            Self::Text => "text",
            Self::OptionalTextBias => "optional_text_bias",
            Self::Readout => "readout",
        }
    }

    pub const fn optional(self) -> bool {
        matches!(self, Self::OptionalTextBias | Self::Readout)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpectedTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub kind: TensorKind,
    pub layer: Option<usize>,
    pub deepstack_index: Option<usize>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InventoryScope {
    TowerOnly,
    Full,
}

#[derive(Clone, Debug)]
pub struct BoundTensor {
    pub expected: ExpectedTensor,
    pub actual: TensorMeta,
}

#[derive(Clone, Debug)]
pub struct InventoryReport {
    pub scope: InventoryScope,
    pub bound: Vec<BoundTensor>,
    pub vision_count: usize,
    pub deepstack_count: usize,
    pub text_count: usize,
    pub optional_text_bias_count: usize,
    pub readout_count: usize,
    pub total_elements: u64,
    pub total_bytes: u64,
    pub vision_layers: Vec<usize>,
    pub text_layers: Vec<usize>,
    pub deepstack_layers: Vec<usize>,
    pub shards: BTreeMap<String, usize>,
}

impl InventoryReport {
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.bound.iter().map(|bound| bound.actual.name.as_str())
    }
}

fn push(
    out: &mut Vec<ExpectedTensor>,
    name: String,
    shape: Vec<usize>,
    kind: TensorKind,
    layer: Option<usize>,
    deepstack_index: Option<usize>,
) {
    out.push(ExpectedTensor {
        name,
        shape,
        kind,
        layer,
        deepstack_index,
    });
}

fn push_norm(
    out: &mut Vec<ExpectedTensor>,
    prefix: &str,
    width: usize,
    kind: TensorKind,
    layer: Option<usize>,
    deepstack_index: Option<usize>,
) {
    push(
        out,
        format!("{prefix}.weight"),
        vec![width],
        kind,
        layer,
        deepstack_index,
    );
    push(
        out,
        format!("{prefix}.bias"),
        vec![width],
        kind,
        layer,
        deepstack_index,
    );
}

fn push_linear(
    out: &mut Vec<ExpectedTensor>,
    prefix: &str,
    rows: usize,
    columns: usize,
    kind: TensorKind,
    layer: Option<usize>,
    deepstack_index: Option<usize>,
) {
    push(
        out,
        format!("{prefix}.weight"),
        vec![rows, columns],
        kind,
        layer,
        deepstack_index,
    );
    push(
        out,
        format!("{prefix}.bias"),
        vec![rows],
        kind,
        layer,
        deepstack_index,
    );
}

fn vision_expected(
    config: &Qwen3VlConfig,
    out: &mut Vec<ExpectedTensor>,
) -> Result<(), BackendError> {
    let vision = &config.vision;
    let hidden = vision.hidden_size;
    let merge_area = vision.merge_area()?;
    let grouped = hidden
        .checked_mul(merge_area)
        .ok_or_else(|| start("merged vision width overflow"))?;
    let patch_dim = vision.patch_dim()?;
    push(
        out,
        format!("{VISION_PREFIX}patch_embed.proj.weight"),
        vec![
            hidden,
            vision.in_channels,
            vision.temporal_patch_size,
            vision.patch_size,
            vision.patch_size,
        ],
        TensorKind::Vision,
        None,
        None,
    );
    push(
        out,
        format!("{VISION_PREFIX}patch_embed.proj.bias"),
        vec![hidden],
        TensorKind::Vision,
        None,
        None,
    );
    push(
        out,
        format!("{VISION_PREFIX}pos_embed.weight"),
        vec![vision.num_position_embeddings, hidden],
        TensorKind::Vision,
        None,
        None,
    );
    if patch_dim
        != vision.in_channels * vision.temporal_patch_size * vision.patch_size * vision.patch_size
    {
        return Err(start("patch dimension mismatch"));
    }
    for layer in 0..vision.depth {
        let prefix = format!("{VISION_PREFIX}blocks.{layer}");
        push_norm(
            out,
            &format!("{prefix}.norm1"),
            hidden,
            TensorKind::Vision,
            Some(layer),
            None,
        );
        push_linear(
            out,
            &format!("{prefix}.attn.qkv"),
            hidden * 3,
            hidden,
            TensorKind::Vision,
            Some(layer),
            None,
        );
        push_linear(
            out,
            &format!("{prefix}.attn.proj"),
            hidden,
            hidden,
            TensorKind::Vision,
            Some(layer),
            None,
        );
        push_norm(
            out,
            &format!("{prefix}.norm2"),
            hidden,
            TensorKind::Vision,
            Some(layer),
            None,
        );
        push_linear(
            out,
            &format!("{prefix}.mlp.linear_fc1"),
            vision.intermediate_size,
            hidden,
            TensorKind::Vision,
            Some(layer),
            None,
        );
        push_linear(
            out,
            &format!("{prefix}.mlp.linear_fc2"),
            hidden,
            vision.intermediate_size,
            TensorKind::Vision,
            Some(layer),
            None,
        );
    }
    push_norm(
        out,
        &format!("{VISION_PREFIX}merger.norm"),
        hidden,
        TensorKind::Vision,
        None,
        None,
    );
    push_linear(
        out,
        &format!("{VISION_PREFIX}merger.linear_fc1"),
        grouped,
        grouped,
        TensorKind::Vision,
        None,
        None,
    );
    push_linear(
        out,
        &format!("{VISION_PREFIX}merger.linear_fc2"),
        vision.out_hidden_size,
        grouped,
        TensorKind::Vision,
        None,
        None,
    );
    for (index, layer) in vision.deepstack_visual_indexes.iter().enumerate() {
        let prefix = format!("{VISION_PREFIX}deepstack_merger_list.{index}");
        push_norm(
            out,
            &format!("{prefix}.norm"),
            grouped,
            TensorKind::Deepstack,
            Some(*layer),
            Some(index),
        );
        push_linear(
            out,
            &format!("{prefix}.linear_fc1"),
            grouped,
            grouped,
            TensorKind::Deepstack,
            Some(*layer),
            Some(index),
        );
        push_linear(
            out,
            &format!("{prefix}.linear_fc2"),
            vision.out_hidden_size,
            grouped,
            TensorKind::Deepstack,
            Some(*layer),
            Some(index),
        );
    }
    Ok(())
}

fn text_expected(text: &TextConfig, out: &mut Vec<ExpectedTensor>) -> Result<(), BackendError> {
    let hidden = text.hidden_size;
    let q_width = text.q_width()?;
    let kv_width = text.kv_width()?;
    let intermediate = text.intermediate_size;
    push(
        out,
        format!("{TEXT_PREFIX}embed_tokens.weight"),
        vec![text.vocab_size, hidden],
        TensorKind::Text,
        None,
        None,
    );
    push(
        out,
        format!("{TEXT_PREFIX}norm.weight"),
        vec![hidden],
        TensorKind::Text,
        None,
        None,
    );
    // Embedding checkpoints can carry a tied LM head that Native ignores.
    push(
        out,
        "lm_head.weight".to_owned(),
        vec![text.vocab_size, hidden],
        TensorKind::OptionalTextBias,
        None,
        None,
    );
    push(
        out,
        "lm_head.bias".to_owned(),
        vec![text.vocab_size],
        TensorKind::OptionalTextBias,
        None,
        None,
    );
    for layer in 0..text.num_hidden_layers {
        let prefix = format!("{TEXT_PREFIX}layers.{layer}");
        push(
            out,
            format!("{prefix}.input_layernorm.weight"),
            vec![hidden],
            TensorKind::Text,
            Some(layer),
            None,
        );
        push(
            out,
            format!("{prefix}.post_attention_layernorm.weight"),
            vec![hidden],
            TensorKind::Text,
            Some(layer),
            None,
        );
        for (name, rows, columns) in [
            ("q_proj", q_width, hidden),
            ("k_proj", kv_width, hidden),
            ("v_proj", kv_width, hidden),
            ("o_proj", hidden, q_width),
        ] {
            push(
                out,
                format!("{prefix}.self_attn.{name}.weight"),
                vec![rows, columns],
                TensorKind::Text,
                Some(layer),
                None,
            );
            push(
                out,
                format!("{prefix}.self_attn.{name}.bias"),
                vec![rows],
                TensorKind::OptionalTextBias,
                Some(layer),
                None,
            );
        }
        for name in ["q_norm", "k_norm"] {
            push(
                out,
                format!("{prefix}.self_attn.{name}.weight"),
                vec![text.head_dim],
                TensorKind::Text,
                Some(layer),
                None,
            );
        }
        for (name, rows, columns) in [
            ("gate_proj", intermediate, hidden),
            ("up_proj", intermediate, hidden),
            ("down_proj", hidden, intermediate),
        ] {
            push(
                out,
                format!("{prefix}.mlp.{name}.weight"),
                vec![rows, columns],
                TensorKind::Text,
                Some(layer),
                None,
            );
            push(
                out,
                format!("{prefix}.mlp.{name}.bias"),
                vec![rows],
                TensorKind::OptionalTextBias,
                Some(layer),
                None,
            );
        }
    }
    Ok(())
}

fn readout_expected(config: &Qwen3VlConfig, out: &mut Vec<ExpectedTensor>) {
    use crate::qwen3_vl_config::ReadoutKind;
    if !config.readout.multi_vector {
        return;
    }
    match config.readout.kind {
        ReadoutKind::ColQwen3 => {
            let width = config
                .readout
                .token_projection_width
                .unwrap_or(crate::qwen3_vl_config::COLQWEN3_TOKEN_WIDTH);
            for prefix in ["embedding_proj_layer", "linear"] {
                push(
                    out,
                    format!("{prefix}.weight"),
                    vec![width, config.text.hidden_size],
                    TensorKind::Readout,
                    None,
                    None,
                );
                push(
                    out,
                    format!("{prefix}.bias"),
                    vec![width],
                    TensorKind::Readout,
                    None,
                    None,
                );
            }
        }
        ReadoutKind::Embedding => {
            // HF_ColBERT's projection width comes from the checkpoint, so the
            // inventory binds it without inventing a shape.
            push(
                out,
                "linear.weight".to_owned(),
                Vec::new(),
                TensorKind::Readout,
                None,
                None,
            );
            push(
                out,
                "linear.bias".to_owned(),
                Vec::new(),
                TensorKind::Readout,
                None,
                None,
            );
        }
        ReadoutKind::Rerank => {
            push(
                out,
                "score.weight".to_owned(),
                Vec::new(),
                TensorKind::Readout,
                None,
                None,
            );
            push(
                out,
                "score.bias".to_owned(),
                Vec::new(),
                TensorKind::Readout,
                None,
                None,
            );
        }
        ReadoutKind::NemotronColEmbed => {}
    }
}

pub fn expected_tensors(
    config: &Qwen3VlConfig,
    scope: InventoryScope,
) -> Result<Vec<ExpectedTensor>, BackendError> {
    let mut expected = Vec::new();
    vision_expected(config, &mut expected)?;
    if scope == InventoryScope::Full {
        text_expected(&config.text, &mut expected)?;
    }
    readout_expected(config, &mut expected);
    Ok(expected)
}

fn canonical_name(name: &str) -> Option<String> {
    // ColQwen3 stores the same VLM under `vlm.`; Native strips the namespace.
    let name = name.strip_prefix("vlm.").unwrap_or(name);
    if name.starts_with(VISION_PREFIX) {
        return Some(name.to_owned());
    }
    if let Some(rest) = name.strip_prefix("visual.") {
        return Some(format!("{VISION_PREFIX}{rest}"));
    }
    if name.starts_with(TEXT_PREFIX) {
        return Some(name.to_owned());
    }
    if let Some(rest) = name.strip_prefix("language_model.") {
        return Some(format!("{TEXT_PREFIX}{rest}"));
    }
    // Trained readout tensors extracted by Native's bridge before the adapter.
    if matches!(
        name,
        "embedding_proj_layer.weight"
            | "embedding_proj_layer.bias"
            | "linear.weight"
            | "linear.bias"
            | "score.weight"
            | "score.bias"
            | "lm_head.weight"
            | "lm_head.bias"
    ) {
        return Some(name.to_owned());
    }
    None
}

/// Native requires QKV biases to be all present or all absent, and gate/up
/// biases to be pairwise present. `o_proj` and `down_proj` biases are
/// independent. Inventory must reject a partial group before any load.
pub fn validate_optional_bias_groups(
    text: &TextConfig,
    present: &BTreeSet<String>,
) -> Result<(), BackendError> {
    for layer in 0..text.num_hidden_layers {
        let prefix = format!("{TEXT_PREFIX}layers.{layer}");
        let grouped: [(&[&str], usize); 2] = [
            (&["q_proj", "k_proj", "v_proj"], 3usize),
            (&["gate_proj", "up_proj"], 2usize),
        ];
        for (names, full) in grouped {
            let count = names
                .iter()
                .filter(|name| {
                    present.contains(&format!("{prefix}.self_attn.{name}.bias"))
                        || present.contains(&format!("{prefix}.mlp.{name}.bias"))
                })
                .count();
            if count != 0 && count != full {
                return Err(start(format!(
                    "layer {layer} has a partial bias group {names:?}: {count} of {full}"
                )));
            }
        }
    }
    Ok(())
}

pub fn validate_checkpoint(
    config: &Qwen3VlConfig,
    metadata: &CheckpointMetadata,
    scope: InventoryScope,
) -> Result<InventoryReport, BackendError> {
    let expected = expected_tensors(config, scope)?;
    let mut canonical: BTreeMap<String, (&TensorMeta, String)> = BTreeMap::new();
    let mut unused = Vec::new();
    for (name, meta) in &metadata.tensors {
        match canonical_name(name) {
            Some(key) => {
                if let Some((previous, original)) = canonical.get(&key) {
                    if original != name {
                        return Err(start(format!(
                            "ambiguous Qwen3-VL weight {key}: both {original} and {name} are present"
                        )));
                    }
                    let _ = previous;
                }
                canonical.insert(key, (meta, name.clone()));
            }
            None => unused.push(name.clone()),
        }
    }
    if !unused.is_empty() {
        return Err(start(format!(
            "checkpoint contains tensors outside the Qwen3-VL adapter: {unused:?}"
        )));
    }
    let mut bound = Vec::with_capacity(expected.len());
    for expected_tensor in expected {
        let Some((actual, original)) = canonical.remove(&expected_tensor.name) else {
            if expected_tensor.kind.optional() {
                continue;
            }
            return Err(start(format!(
                "missing Qwen3-VL tensor {}",
                expected_tensor.name
            )));
        };
        if actual.dtype != CheckpointDtype::Bf16 && actual.dtype != CheckpointDtype::F16 {
            return Err(start(format!(
                "tensor {} has unsupported dtype {}",
                expected_tensor.name,
                actual.dtype.name()
            )));
        }
        let declared = match config.vision.dtype {
            crate::qwen3_vl_config::Dtype::Bf16 => CheckpointDtype::Bf16,
            crate::qwen3_vl_config::Dtype::F16 => CheckpointDtype::F16,
            crate::qwen3_vl_config::Dtype::F32 => CheckpointDtype::F32,
        };
        if actual.dtype != declared {
            return Err(start(format!(
                "tensor {} dtype {} does not match config dtype {}",
                expected_tensor.name,
                actual.dtype.name(),
                config.vision.dtype.name()
            )));
        }
        if !expected_tensor.shape.is_empty() && actual.shape != expected_tensor.shape {
            return Err(start(format!(
                "tensor {} has shape {:?}, expected {:?} (source {original})",
                expected_tensor.name, actual.shape, expected_tensor.shape
            )));
        }
        bound.push(BoundTensor {
            expected: expected_tensor,
            actual: actual.clone(),
        });
    }
    if scope == InventoryScope::TowerOnly {
        for key in canonical.keys() {
            if !key.starts_with(TEXT_PREFIX) {
                return Err(start(format!("unexpected Qwen3-VL tensor {key}")));
            }
        }
    } else if let Some((name, _)) = canonical.into_iter().next() {
        return Err(start(format!("unexpected Qwen3-VL tensor {name}")));
    }
    let present_biases: BTreeSet<String> = bound
        .iter()
        .filter(|tensor| tensor.expected.kind == TensorKind::OptionalTextBias)
        .map(|tensor| tensor.expected.name.clone())
        .collect();
    validate_optional_bias_groups(&config.text, &present_biases)?;
    let mut vision_count = 0;
    let mut deepstack_count = 0;
    let mut text_count = 0;
    let mut optional_text_bias_count = 0;
    let mut readout_count = 0usize;
    let mut total_elements = 0u64;
    let mut total_bytes = 0u64;
    let mut vision_layers = BTreeSet::new();
    let mut text_layers = BTreeSet::new();
    let mut deepstack_layers = BTreeSet::new();
    let mut shards = BTreeMap::new();
    for tensor in &bound {
        match tensor.expected.kind {
            TensorKind::Vision => vision_count += 1,
            TensorKind::Deepstack => deepstack_count += 1,
            TensorKind::Text => text_count += 1,
            TensorKind::OptionalTextBias => optional_text_bias_count += 1,
            TensorKind::Readout => readout_count += 1,
        }
        if let Some(layer) = tensor.expected.layer {
            match tensor.expected.kind {
                TensorKind::Vision => {
                    vision_layers.insert(layer);
                }
                TensorKind::Deepstack => {
                    deepstack_layers.insert(layer);
                }
                TensorKind::Text | TensorKind::OptionalTextBias | TensorKind::Readout => {
                    text_layers.insert(layer);
                }
            }
        }
        total_elements = total_elements
            .checked_add(u64::try_from(tensor.actual.elements()?).map_err(start)?)
            .ok_or_else(|| start("inventory element count overflow"))?;
        total_bytes = total_bytes
            .checked_add(tensor.actual.bytes()?)
            .ok_or_else(|| start("inventory byte count overflow"))?;
        *shards.entry(tensor.actual.shard.clone()).or_insert(0) += 1;
    }
    Ok(InventoryReport {
        scope,
        bound,
        vision_count,
        deepstack_count,
        text_count,
        optional_text_bias_count,
        readout_count,
        total_elements,
        total_bytes,
        vision_layers: vision_layers.into_iter().collect(),
        text_layers: text_layers.into_iter().collect(),
        deepstack_layers: deepstack_layers.into_iter().collect(),
        shards,
    })
}

/// Open the actual checkpoint config/headers and validate them. No payload is
/// read and no device is created.
pub fn validate_checkpoint_path(
    root: &Path,
    scope: InventoryScope,
) -> Result<(Qwen3VlConfig, InventoryReport), BackendError> {
    let config = Qwen3VlConfig::from_path(root)?;
    let metadata = CheckpointMetadata::open(root)?;
    let report = validate_checkpoint(&config, &metadata, scope)?;
    Ok((config, report))
}

pub fn deepstack_mapping(config: &Qwen3VlConfig) -> Vec<(usize, usize)> {
    config
        .vision
        .deepstack_visual_indexes
        .iter()
        .enumerate()
        .map(|(name_index, layer)| (*layer, name_index))
        .collect()
}
