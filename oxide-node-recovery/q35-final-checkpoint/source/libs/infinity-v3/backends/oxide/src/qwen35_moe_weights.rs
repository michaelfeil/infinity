// Cold, selective Qwen3.5-MoE weight manifest.
//
// Native (`qwen35_models.cpp`) binds one router and one shared expert per
// decoder layer. Routed experts are preferred in their packed checkpoint form:
//
//   mlp.gate.weight                    [E, H]
//   mlp.experts.gate_up_proj           [E, 2I, H]   (gate rows then up rows)
//   mlp.experts.down_proj              [E, H, I]
//
// Older/alternate checkpoints may expose separate experts:
//
//   mlp.experts.{e}.gate_proj.weight   [I, H]
//   mlp.experts.{e}.up_proj.weight     [I, H]
//   mlp.experts.{e}.down_proj.weight   [H, I]
//
// Native prefers the packed tensors when present. This manifest preserves that
// precedence, validates every dimension, and keeps only lazy range views. It
// never concatenates, transposes, or allocates expert payloads on the host. The
// separate fallback keeps one metadata view per tensor (no bytes).
//
// The shared expert is always separate in the checkpoint:
//   mlp.shared_expert.gate_proj.weight [S, H]
//   mlp.shared_expert.up_proj.weight   [S, H]
//   mlp.shared_expert.down_proj.weight [H, S]
//   mlp.shared_expert_gate.weight      [1, H]
//
// Attention/norm/embedding tensors stay owned by the existing dense Qwen3.5
// binding. Each layer records its exact `attention_prefix` so the dense
// executor can share its GDN/full-attention sources instead of this module
// duplicating the full model.

use crate::model::{start, Checkpoint, WeightSource};
use crate::qwen35_moe_config::Qwen35MoeConfig;
use crate::qwen35_moe_contract::reject_pytorch_checkpoint;
use crate::qwen35_moe_schedule::{MoeLayerShapes, Qwen35MoeSchedule};
use infinity_v3_backend_core::BackendError;
use safetensors::Dtype;
use std::path::Path;

const NATIVE_ROOTS: [&str; 4] = ["model.language_model.", "language_model.", "model.", ""];

#[derive(Clone, Debug)]
#[allow(clippy::large_enum_variant)]
pub(crate) enum GateUpSources {
    Packed(WeightSource),
    Separate {
        gate: WeightSource,
        up: WeightSource,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct ExpertSources {
    pub(crate) gate_up: GateUpSources,
    pub(crate) down: WeightSource,
}

#[derive(Clone, Debug)]
struct SeparateExpert {
    gate: WeightSource,
    up: WeightSource,
    down: WeightSource,
}

#[derive(Clone, Debug)]
#[allow(clippy::large_enum_variant)]
enum Storage {
    Packed {
        gate_up: WeightSource,
        down: WeightSource,
    },
    Separate(Vec<SeparateExpert>),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ExpertStorageKind {
    Packed,
    Separate,
}

#[derive(Clone, Debug)]
pub(crate) struct RoutedLayerWeights {
    pub(crate) router: WeightSource,
    storage: Storage,
    pub(crate) hidden: usize,
    pub(crate) intermediate: usize,
    pub(crate) experts: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct SharedExpertSources {
    pub(crate) gate: WeightSource,
    pub(crate) up: WeightSource,
    pub(crate) down: WeightSource,
    pub(crate) gate_logit: WeightSource,
}

#[derive(Clone, Debug)]
pub(crate) struct Qwen35MoeLayerWeights {
    // `input_norm`/`post_attention_norm` are Native offset RMSNorm weights.
    pub(crate) index: usize,
    pub(crate) attention_prefix: String,
    pub(crate) input_norm: WeightSource,
    pub(crate) post_attention_norm: WeightSource,
    pub(crate) routed: RoutedLayerWeights,
    pub(crate) shared: SharedExpertSources,
    pub(crate) shapes: MoeLayerShapes,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SeparateExpertBinding {
    pub(crate) expert: usize,
    pub(crate) gate: String,
    pub(crate) up: String,
    pub(crate) down: String,
}

/// Declarative checkpoint-name mapping for one layer. Packed checkpoints
/// produce exactly three routed bindings; separate checkpoints produce the
/// three shared bindings plus three names per expert. No payload or duplicate
/// BF16 view is created.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct LayerBindingPlan {
    pub(crate) index: usize,
    pub(crate) attention_prefix: String,
    pub(crate) input_norm: String,
    pub(crate) post_attention_norm: String,
    pub(crate) router: String,
    pub(crate) packed_gate_up: Option<String>,
    pub(crate) packed_down: Option<String>,
    pub(crate) separate_experts: Vec<SeparateExpertBinding>,
    pub(crate) shared_gate: String,
    pub(crate) shared_up: String,
    pub(crate) shared_down: String,
    pub(crate) shared_gate_logit: String,
    pub(crate) dtype: Dtype,
}

#[derive(Clone, Debug)]
pub(crate) struct Qwen35MoeWeights {
    pub(crate) config: Qwen35MoeConfig,
    pub(crate) root: String,
    pub(crate) embeddings: WeightSource,
    pub(crate) norm: WeightSource,
    pub(crate) lm_head: Option<WeightSource>,
    pub(crate) layers: Vec<Qwen35MoeLayerWeights>,
}

impl Qwen35MoeWeights {
    pub(crate) fn load(path: &Path) -> Result<Self, BackendError> {
        // The donor `Checkpoint` also admits PyTorch archives. This foundation
        // deliberately admits safetensors only, so standalone and donor
        // behavior agree (review F6).
        reject_pytorch_checkpoint(path)?;
        let config = Qwen35MoeConfig::load(path)?;
        let checkpoint = Checkpoint::open(path)?;
        Self::bind(&checkpoint, config)
    }

    /// Bind against an already-open checkpoint. Useful for tests and for
    /// future selective startup code; the checkpoint remains the owner of
    /// shard file handles.
    pub(crate) fn bind(
        checkpoint: &Checkpoint,
        config: Qwen35MoeConfig,
    ) -> Result<Self, BackendError> {
        if config.dtype != crate::qwen35_moe_config::Qwen35MoeDtype::Bf16 {
            return Err(start(
                "Qwen3.5-MoE cold manifest admits BF16 checkpoints only",
            ));
        }
        let root = select_root(checkpoint)?;
        let hidden = config.text.hidden_size;
        let schedule = Qwen35MoeSchedule::new(&config)?;

        let embeddings = check_source(
            checkpoint.source(
                &format!("{root}embed_tokens.weight"),
                &[config.text.vocab_size, hidden],
            )?,
            "embed_tokens.weight",
        )?;
        let norm = check_source(
            checkpoint.source(&format!("{root}norm.weight"), &[hidden])?,
            "norm.weight",
        )?;
        let lm_head = if config.tie_word_embeddings {
            checkpoint.optional_source("lm_head.weight", &[config.text.vocab_size, hidden])?
        } else {
            Some(checkpoint.source("lm_head.weight", &[config.text.vocab_size, hidden])?)
        }
        .map(|source| check_source(source, "lm_head.weight"))
        .transpose()?;

        let mut layers = Vec::new();
        layers
            .try_reserve_exact(schedule.layers().len())
            .map_err(start)?;
        for plan in schedule.layers() {
            let prefix = format!("{root}layers.{}", plan.index);
            let input_norm = check_source(
                checkpoint.source(&format!("{prefix}.input_layernorm.weight"), &[hidden])?,
                "input_layernorm.weight",
            )?;
            let post_attention_norm = check_source(
                checkpoint.source(
                    &format!("{prefix}.post_attention_layernorm.weight"),
                    &[hidden],
                )?,
                "post_attention_layernorm.weight",
            )?;
            let routed = bind_routed(checkpoint, &prefix, &config, plan.shapes)?;
            let shared = bind_shared(checkpoint, &prefix, &config, plan.shapes)?;
            layers.push(Qwen35MoeLayerWeights {
                index: plan.index,
                attention_prefix: format!("{root}layers.{}.", plan.index),
                input_norm,
                post_attention_norm,
                routed,
                shared,
                shapes: plan.shapes,
            });
        }
        let weights = Self {
            config,
            root: root.to_owned(),
            embeddings,
            norm,
            lm_head,
            layers,
        };
        weights.validate_manifest()?;
        Ok(weights)
    }

    fn validate_manifest(&self) -> Result<(), BackendError> {
        let schedule = Qwen35MoeSchedule::new(&self.config)?;
        if self.layers.len() != schedule.layers().len() {
            return Err(start("Qwen3.5-MoE manifest layer count mismatch"));
        }
        for (layer, plan) in self.layers.iter().zip(schedule.layers()) {
            if layer.index != plan.index || layer.shapes != plan.shapes {
                return Err(start(format!(
                    "Qwen3.5-MoE layer {} disagrees with its schedule",
                    plan.index
                )));
            }
            layer.routed.validate()?;
        }
        Ok(())
    }

    pub(crate) fn schedule(&self) -> Result<Qwen35MoeSchedule, BackendError> {
        Qwen35MoeSchedule::new(&self.config)
    }

    pub(crate) fn layer_binding_plan(
        &self,
        index: usize,
    ) -> Result<LayerBindingPlan, BackendError> {
        self.layers
            .get(index)
            .map(Qwen35MoeLayerWeights::binding_plan)
            .ok_or_else(|| start("binding plan layer index out of range"))
    }

    pub(crate) fn routed_storage_kind(
        &self,
        index: usize,
    ) -> Result<ExpertStorageKind, BackendError> {
        self.layers
            .get(index)
            .map(|layer| layer.routed.storage_kind())
            .ok_or_else(|| start("expert layer index out of range"))
    }
}

impl Qwen35MoeLayerWeights {
    pub(crate) fn binding_plan(&self) -> LayerBindingPlan {
        let stem = format!("{}mlp", self.attention_prefix);
        let (packed_gate_up, packed_down, separate_experts) = match &self.routed.storage {
            Storage::Packed { .. } => (
                Some(format!("{stem}.experts.gate_up_proj")),
                Some(format!("{stem}.experts.down_proj")),
                Vec::new(),
            ),
            Storage::Separate(experts) => {
                let mut bindings = Vec::new();
                bindings
                    .try_reserve_exact(experts.len())
                    .expect("binding plan length was already validated");
                for expert in 0..experts.len() {
                    let expert_stem = format!("{stem}.experts.{expert}");
                    bindings.push(SeparateExpertBinding {
                        expert,
                        gate: format!("{expert_stem}.gate_proj.weight"),
                        up: format!("{expert_stem}.up_proj.weight"),
                        down: format!("{expert_stem}.down_proj.weight"),
                    });
                }
                (None, None, bindings)
            }
        };
        LayerBindingPlan {
            index: self.index,
            attention_prefix: self.attention_prefix.clone(),
            input_norm: format!("{}input_layernorm.weight", self.attention_prefix),
            post_attention_norm: format!(
                "{}post_attention_layernorm.weight",
                self.attention_prefix
            ),
            router: format!("{stem}.gate.weight"),
            packed_gate_up,
            packed_down,
            separate_experts,
            shared_gate: format!("{stem}.shared_expert.gate_proj.weight"),
            shared_up: format!("{stem}.shared_expert.up_proj.weight"),
            shared_down: format!("{stem}.shared_expert.down_proj.weight"),
            shared_gate_logit: format!("{stem}.shared_expert_gate.weight"),
            dtype: Dtype::BF16,
        }
    }
}

impl RoutedLayerWeights {
    pub(crate) fn storage_kind(&self) -> ExpertStorageKind {
        match self.storage {
            Storage::Packed { .. } => ExpertStorageKind::Packed,
            Storage::Separate(_) => ExpertStorageKind::Separate,
        }
    }

    pub(crate) fn validate(&self) -> Result<(), BackendError> {
        let two_i = checked_product(self.intermediate, 2, "gate/up width")?;
        match &self.storage {
            Storage::Packed { gate_up, down } => {
                if gate_up.shape() != [self.experts, two_i, self.hidden] {
                    return Err(start("packed gate/up expert shape mismatch"));
                }
                if down.shape() != [self.experts, self.hidden, self.intermediate] {
                    return Err(start("packed down expert shape mismatch"));
                }
            }
            Storage::Separate(experts) => {
                if experts.len() != self.experts {
                    return Err(start("separate expert table length mismatch"));
                }
                for (index, expert) in experts.iter().enumerate() {
                    if expert.gate.shape() != [self.intermediate, self.hidden]
                        || expert.up.shape() != [self.intermediate, self.hidden]
                        || expert.down.shape() != [self.hidden, self.intermediate]
                    {
                        return Err(start(format!("separate expert {index} shape mismatch")));
                    }
                }
            }
        }
        Ok(())
    }

    pub(crate) fn expert(&self, index: usize) -> Result<ExpertSources, BackendError> {
        if index >= self.experts {
            return Err(start("expert index out of range"));
        }
        match &self.storage {
            Storage::Packed { gate_up, down } => {
                let two_i = checked_product(self.intermediate, 2, "gate/up dimension overflow")?;
                let gate_up_elements =
                    checked_product(two_i, self.hidden, "gate/up size overflow")?;
                let down_elements =
                    checked_product(self.hidden, self.intermediate, "down size overflow")?;
                let gate_up_offset =
                    checked_product(index, gate_up_elements, "gate/up offset overflow")?;
                let down_offset = checked_product(index, down_elements, "down offset overflow")?;
                Ok(ExpertSources {
                    gate_up: GateUpSources::Packed(
                        gate_up.slice(gate_up_offset, vec![two_i, self.hidden])?,
                    ),
                    down: down.slice(down_offset, vec![self.hidden, self.intermediate])?,
                })
            }
            Storage::Separate(experts) => {
                let source = experts
                    .get(index)
                    .ok_or_else(|| start("expert index out of range"))?;
                Ok(ExpertSources {
                    gate_up: GateUpSources::Separate {
                        gate: source.gate.slice(0, source.gate.shape().to_vec())?,
                        up: source.up.slice(0, source.up.shape().to_vec())?,
                    },
                    down: source.down.slice(0, source.down.shape().to_vec())?,
                })
            }
        }
    }

    /// Element offset of one packed expert without creating a view. This is the
    /// exact row-major offset Native's `[E, 2I, H]` checkpoint layout implies.
    pub(crate) fn packed_expert_offsets(
        &self,
        index: usize,
    ) -> Result<Option<(usize, usize)>, BackendError> {
        if index >= self.experts {
            return Err(start("expert index out of range"));
        }
        match &self.storage {
            Storage::Packed { .. } => {
                let gate_up = checked_product(
                    checked_product(self.intermediate, 2, "gate/up width")?,
                    self.hidden,
                    "gate/up size",
                )?;
                let down = checked_product(self.hidden, self.intermediate, "down size")?;
                Ok(Some((
                    checked_product(index, gate_up, "gate/up offset")?,
                    checked_product(index, down, "down offset")?,
                )))
            }
            Storage::Separate(_) => Ok(None),
        }
    }
}

fn check_source(source: WeightSource, name: &str) -> Result<WeightSource, BackendError> {
    if source.dtype() != Dtype::BF16 {
        return Err(start(format!(
            "Qwen3.5-MoE cold manifest requires BF16 for {name}, found {:?}",
            source.dtype()
        )));
    }
    Ok(source)
}

fn bind_routed(
    checkpoint: &Checkpoint,
    prefix: &str,
    config: &Qwen35MoeConfig,
    shapes: MoeLayerShapes,
) -> Result<RoutedLayerWeights, BackendError> {
    let router = check_source(
        checkpoint.source(&format!("{prefix}.mlp.gate.weight"), &shapes.router)?,
        "mlp.gate.weight",
    )?;
    let fused_name = format!("{prefix}.mlp.experts.gate_up_proj");
    let storage = if checkpoint.contains(&fused_name) {
        let gate_up = check_source(
            checkpoint.source(&fused_name, &shapes.routed_gate_up)?,
            "mlp.experts.gate_up_proj",
        )?;
        let down = check_source(
            checkpoint.source(
                &format!("{prefix}.mlp.experts.down_proj"),
                &shapes.routed_down,
            )?,
            "mlp.experts.down_proj",
        )?;
        Storage::Packed { gate_up, down }
    } else {
        let mut experts = Vec::new();
        experts
            .try_reserve_exact(config.moe.num_experts)
            .map_err(|_| start("expert view allocation failed"))?;
        let gate_shape = [config.moe.moe_intermediate_size, config.text.hidden_size];
        let down_shape = [config.text.hidden_size, config.moe.moe_intermediate_size];
        for expert in 0..config.moe.num_experts {
            let stem = format!("{prefix}.mlp.experts.{expert}");
            let gate = check_source(
                checkpoint.source(&format!("{stem}.gate_proj.weight"), &gate_shape)?,
                "experts.{expert}.gate_proj.weight",
            )?;
            let up = check_source(
                checkpoint.source(&format!("{stem}.up_proj.weight"), &gate_shape)?,
                "experts.{expert}.up_proj.weight",
            )?;
            let down = check_source(
                checkpoint.source(&format!("{stem}.down_proj.weight"), &down_shape)?,
                "experts.{expert}.down_proj.weight",
            )?;
            experts.push(SeparateExpert { gate, up, down });
        }
        Storage::Separate(experts)
    };
    Ok(RoutedLayerWeights {
        router,
        storage,
        hidden: config.text.hidden_size,
        intermediate: config.moe.moe_intermediate_size,
        experts: config.moe.num_experts,
    })
}

fn bind_shared(
    checkpoint: &Checkpoint,
    prefix: &str,
    config: &Qwen35MoeConfig,
    shapes: MoeLayerShapes,
) -> Result<SharedExpertSources, BackendError> {
    let stem = format!("{prefix}.mlp.shared_expert");
    let width = config.moe.shared_expert_intermediate_size;
    let hidden = config.text.hidden_size;
    let gate = check_source(
        checkpoint.source(&format!("{stem}.gate_proj.weight"), &[width, hidden])?,
        "shared_expert.gate_proj.weight",
    )?;
    let up = check_source(
        checkpoint.source(&format!("{stem}.up_proj.weight"), &[width, hidden])?,
        "shared_expert.up_proj.weight",
    )?;
    let down = check_source(
        checkpoint.source(&format!("{stem}.down_proj.weight"), &[hidden, width])?,
        "shared_expert.down_proj.weight",
    )?;
    let gate_logit = check_source(
        checkpoint.source(
            &format!("{prefix}.mlp.shared_expert_gate.weight"),
            &shapes.shared_gate,
        )?,
        "shared_expert_gate.weight",
    )?;
    Ok(SharedExpertSources {
        gate,
        up,
        down,
        gate_logit,
    })
}

fn checked_product(left: usize, right: usize, name: &str) -> Result<usize, BackendError> {
    left.checked_mul(right)
        .ok_or_else(|| start(format!("{name} overflow")))
}

fn is_backbone_suffix(suffix: &str) -> bool {
    suffix.starts_with("embed_tokens.")
        || suffix.starts_with("norm.")
        || suffix.starts_with("layers.")
}

/// Select exactly one complete Native text root. This mirrors the dense
/// Qwen3.5 selector so MoE and GDN share the same namespace rules. Vision
/// (`model.visual.*`) and MTP (`mtp.*`) tensors never count as text backbone
/// anchors, and `mtp.layers.*` is never mistaken for a decoder layer.
fn select_root(checkpoint: &Checkpoint) -> Result<&'static str, BackendError> {
    let mut selected = None;
    let mut anchors = 0usize;
    for name in checkpoint.names() {
        for root in NATIVE_ROOTS {
            if name.strip_prefix(root) == Some("embed_tokens.weight") {
                anchors = anchors
                    .checked_add(1)
                    .ok_or_else(|| start("Qwen3.5-MoE checkpoint anchor overflow"))?;
                selected = Some(root);
            }
        }
    }
    if anchors == 0 {
        return Err(start("missing weight embed_tokens.weight"));
    }
    if anchors != 1 {
        return Err(start(
            "ambiguous Qwen3.5-MoE checkpoint namespace: multiple embed_tokens.weight roots",
        ));
    }
    let root = selected.ok_or_else(|| start("missing Qwen3.5-MoE checkpoint root"))?;
    for name in checkpoint.names() {
        for other in NATIVE_ROOTS {
            if other == root {
                continue;
            }
            if name.strip_prefix(other).is_some_and(is_backbone_suffix) {
                return Err(start(format!(
                    "mixed Qwen3.5-MoE checkpoint roots: found {name} alongside the {root:?} root"
                )));
            }
        }
    }
    for suffix in ["norm.weight", "layers.0.input_layernorm.weight"] {
        let name = format!("{root}{suffix}");
        if !checkpoint.contains(&name) {
            return Err(start(format!(
                "partial Qwen3.5-MoE text root {root:?}: missing weight {name}"
            )));
        }
    }
    Ok(root)
}
