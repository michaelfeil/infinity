// Checked cold configuration for the Qwen3.5-MoE text backbone.
//
// Schema and defaults follow the current Native constructor in
// `backends/libtorch/cpp/models/qwen35_models.cpp`:
//
// * `num_experts > 0` selects the routed FFN path for every decoder layer;
// * `num_experts_per_tok` defaults to 8 and must not exceed `num_experts`;
// * `norm_topk_prob` defaults to true (the Native `cfg.boolean(..., true)`);
// * the shared expert is always present, with a sigmoid `shared_expert_gate`;
// * `layer_types` drives the hybrid full-attention / gated-delta schedule;
// * MTP and vision tensors are not part of the text decoder layer list.
//
// The parser is metadata-only. It never opens shards, allocates expert payloads,
// queries a device, or claims an executor. The only accepted runtime dtype is
// BF16, mirroring the Native accelerated MoE path.

use crate::model::{elements, start};
use crate::qwen35_config::Qwen35LayerType;
use infinity_v3_backend_core::BackendError;
use serde::Deserialize;
use serde_json::Value;
use std::path::Path;

const DEFAULT_RMS_NORM_EPS: f64 = 1e-6;
// Native `qwen35_models.cpp:1187` default: 10_000. The cached 35B config
// supplies 10_000_000 explicitly, so this only affects configs that omit it.
const DEFAULT_ROPE_THETA: f64 = 10_000.0;
const DEFAULT_MAX_POSITION_EMBEDDINGS: usize = 32_768;
const DEFAULT_EXPERTS_PER_TOK: usize = 8;
const DELTA_HEAD_DIM: usize = 128;
const DEFAULT_LINEAR_CONV_KERNEL_DIM: usize = 4;
const SUPPORTED_ARCHITECTURES: [&str; 4] = [
    "Qwen3_5MoeForConditionalGeneration",
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5MoeModel",
    "Qwen3_5MoeTextModel",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Qwen35MoeDtype {
    Bf16,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Qwen35MoeTextGeometry {
    pub(crate) hidden_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) rotary_dim: usize,
    pub(crate) vocab_size: usize,
    pub(crate) max_position_embeddings: usize,
    pub(crate) linear_num_key_heads: usize,
    pub(crate) linear_num_value_heads: usize,
    pub(crate) linear_key_head_dim: usize,
    pub(crate) linear_value_head_dim: usize,
    pub(crate) linear_conv_kernel_dim: usize,
    pub(crate) rms_norm_eps: f32,
    pub(crate) rope_theta: f32,
    pub(crate) mrope_interleaved: bool,
    pub(crate) mrope_section: Option<[usize; 3]>,
    pub(crate) layer_types: Vec<Qwen35LayerType>,
    pub(crate) attn_output_gate: bool,
}

impl Qwen35MoeTextGeometry {
    pub(crate) fn q_width(&self) -> Result<usize, BackendError> {
        checked_product(self.num_attention_heads, self.head_dim, "query width")
    }

    pub(crate) fn kv_width(&self) -> Result<usize, BackendError> {
        checked_product(self.num_key_value_heads, self.head_dim, "key/value width")
    }

    pub(crate) fn full_attention_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|kind| **kind == Qwen35LayerType::Full)
            .count()
    }

    pub(crate) fn linear_attention_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|kind| **kind == Qwen35LayerType::Linear)
            .count()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Qwen35MoeSpec {
    pub(crate) num_experts: usize,
    pub(crate) num_experts_per_tok: usize,
    pub(crate) moe_intermediate_size: usize,
    pub(crate) shared_expert_intermediate_size: usize,
    pub(crate) norm_topk_prob: bool,
    pub(crate) router_aux_loss_coef: Option<f32>,
}

impl Qwen35MoeSpec {
    pub(crate) fn gate_up_width(&self) -> Result<usize, BackendError> {
        checked_product(self.moe_intermediate_size, 2, "routed gate/up width")
    }

    pub(crate) fn shared_gate_up_width(&self) -> Result<usize, BackendError> {
        checked_product(
            self.shared_expert_intermediate_size,
            2,
            "shared gate/up width",
        )
    }

    /// Metadata-only shape products. This is deliberately not an allocation.
    pub(crate) fn checked_tensor_elements(
        &self,
        hidden: usize,
    ) -> Result<MoeTensorElements, BackendError> {
        let two_i = self.gate_up_width()?;
        let two_s = self.shared_gate_up_width()?;
        Ok(MoeTensorElements {
            router: elements(&[self.num_experts, hidden])?,
            routed_gate_up: elements(&[self.num_experts, two_i, hidden])?,
            routed_down: elements(&[self.num_experts, hidden, self.moe_intermediate_size])?,
            shared_gate_up: elements(&[two_s, hidden])?,
            shared_down: elements(&[hidden, self.shared_expert_intermediate_size])?,
            shared_gate: elements(&[1, hidden])?,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MoeTensorElements {
    pub(crate) router: usize,
    pub(crate) routed_gate_up: usize,
    pub(crate) routed_down: usize,
    pub(crate) shared_gate_up: usize,
    pub(crate) shared_down: usize,
    pub(crate) shared_gate: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Qwen35MoeConfig {
    pub(crate) text: Qwen35MoeTextGeometry,
    pub(crate) moe: Qwen35MoeSpec,
    pub(crate) dtype: Qwen35MoeDtype,
    pub(crate) root_model_type: String,
    pub(crate) text_model_type: String,
    pub(crate) architectures: Vec<String>,
    pub(crate) tie_word_embeddings: bool,
    pub(crate) vision_depth: usize,
    pub(crate) mtp_num_hidden_layers: usize,
    pub(crate) full_attention_noncausal: bool,
}

/// Explicitly describes why this foundation is not a model route. Callers must
/// not present Qwen3.5-MoE as implemented based only on this module.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Qwen35MoeAdmission {
    pub(crate) config_admitted: bool,
    pub(crate) executor_implemented: bool,
    pub(crate) whole_checkpoint_implemented: bool,
    pub(crate) blockers: Vec<&'static str>,
}

macro_rules! required {
    ($text:expr, $raw:expr, $field:ident) => {
        resolve_field($text, $raw, |config| config.$field, stringify!($field))?
            .ok_or_else(|| start(format!("missing config field {}", stringify!($field))))?
    };
}

macro_rules! optional {
    ($text:expr, $raw:expr, $field:ident, $default:expr) => {
        resolve_field($text, $raw, |config| config.$field, stringify!($field))?.unwrap_or($default)
    };
}

impl Qwen35MoeConfig {
    pub fn load(path: &Path) -> Result<Self, BackendError> {
        let raw: RawQwen35MoeConfig = crate::model::read_json(&path.join("config.json"))?;
        Self::from_raw(raw)
    }

    pub fn from_json(value: &Value) -> Result<Self, BackendError> {
        let raw: RawQwen35MoeConfig = serde_json::from_value(value.clone()).map_err(start)?;
        Self::from_raw(raw)
    }

    pub(crate) fn from_raw(raw: RawQwen35MoeConfig) -> Result<Self, BackendError> {
        let text = raw.text_config.as_deref();
        let root_type = raw
            .model_type
            .as_deref()
            .ok_or_else(|| start("missing model_type"))?;
        let (root_model_type, text_model_type) = match root_type {
            "qwen3_5_moe" => {
                let nested =
                    text.ok_or_else(|| start("qwen3_5_moe root requires a text_config object"))?;
                if nested.text_config.is_some() {
                    return Err(start("nested text_config containers are unsupported"));
                }
                let nested_type = nested
                    .model_type
                    .as_deref()
                    .ok_or_else(|| start("qwen3_5_moe text_config requires model_type"))?;
                if nested_type != "qwen3_5_moe_text" {
                    return Err(start(format!(
                        "unsupported text_config model_type {nested_type}; expected qwen3_5_moe_text"
                    )));
                }
                (root_type, nested_type)
            }
            "qwen3_5_moe_text" => {
                if text.is_some() {
                    return Err(start(
                        "qwen3_5_moe_text root must not also define text_config",
                    ));
                }
                (root_type, root_type)
            }
            other if other.contains("moe") => {
                return Err(start(format!(
                    "unsupported MoE model_type {other}; expected qwen3_5_moe or qwen3_5_moe_text"
                )))
            }
            other => {
                return Err(start(format!(
                    "model_type {other} is not a Qwen3.5-MoE text model"
                )))
            }
        };

        for config in std::iter::once(&raw).chain(text) {
            if active(&config.quantization_config) {
                return Err(start(
                    "quantized Qwen3.5-MoE checkpoints are unsupported; no FP4/FP8 pretence",
                ));
            }
            if config.attention_bias == Some(true) {
                return Err(start(
                    "attention_bias Qwen3.5-MoE checkpoints are unsupported",
                ));
            }
            if config
                .rope_scaling
                .as_ref()
                .is_some_and(|value| !value.is_null())
            {
                return Err(start("scaled RoPE selectors are unsupported"));
            }
            if config
                .hidden_act
                .as_deref()
                .is_some_and(|act| act != "silu")
            {
                return Err(start("only SiLU Qwen3.5-MoE activations are supported"));
            }
            if config.shared_expert_gate == Some(false) {
                return Err(start(
                    "Qwen3.5-MoE native always applies the shared-expert sigmoid gate",
                ));
            }
            for (name, value) in [
                ("n_shared_experts", config.n_shared_experts),
                ("num_shared_experts", config.num_shared_experts),
            ] {
                if value.is_some_and(|count| count != 1) {
                    return Err(start(format!(
                        "{name} must be 1 for the native Qwen3.5-MoE shared expert"
                    )));
                }
            }
            if let Some(architectures) = &config.architectures {
                for architecture in architectures {
                    if !SUPPORTED_ARCHITECTURES.contains(&architecture.as_str()) {
                        return Err(start(format!(
                            "unsupported architecture {architecture}; Qwen3.5-MoE text tasks only"
                        )));
                    }
                }
            }
        }

        let hidden_size = required!(text, &raw, hidden_size);
        let num_hidden_layers = required!(text, &raw, num_hidden_layers);
        let num_attention_heads = required!(text, &raw, num_attention_heads);
        let num_key_value_heads = optional!(text, &raw, num_key_value_heads, num_attention_heads);
        let vocab_size = required!(text, &raw, vocab_size);
        let max_position_embeddings = optional!(
            text,
            &raw,
            max_position_embeddings,
            DEFAULT_MAX_POSITION_EMBEDDINGS
        );
        for (name, value) in [
            ("hidden_size", hidden_size),
            ("num_hidden_layers", num_hidden_layers),
            ("num_attention_heads", num_attention_heads),
            ("num_key_value_heads", num_key_value_heads),
            ("vocab_size", vocab_size),
            ("max_position_embeddings", max_position_embeddings),
        ] {
            if value == 0 {
                return Err(start(format!("config {name} must be nonzero")));
            }
        }
        let head_dim = match resolve_field(text, &raw, |config| config.head_dim, "head_dim")? {
            Some(0) => return Err(start("config head_dim must be nonzero")),
            Some(head_dim) => head_dim,
            None if hidden_size.is_multiple_of(num_attention_heads) => {
                hidden_size / num_attention_heads
            }
            None => return Err(start(
                "hidden_size must divide evenly by num_attention_heads when head_dim is omitted",
            )),
        };

        let raw_layer_types = resolve_field(
            text,
            &raw,
            |config| config.layer_types.clone(),
            "layer_types",
        )?
        .ok_or_else(|| start("missing config field layer_types"))?;
        if raw_layer_types.len() != num_hidden_layers {
            return Err(start(format!(
                "layer_types has {} entries for {num_hidden_layers} layers",
                raw_layer_types.len()
            )));
        }
        let mut layer_types = Vec::new();
        layer_types
            .try_reserve_exact(raw_layer_types.len())
            .map_err(start)?;
        for (index, kind) in raw_layer_types.iter().enumerate() {
            layer_types.push(match kind.as_str() {
                "full_attention" => Qwen35LayerType::Full,
                "linear_attention" => Qwen35LayerType::Linear,
                other => {
                    return Err(start(format!(
                        "unsupported layer_types[{index}] {other}; expected full_attention or linear_attention"
                    )))
                }
            });
        }
        let full_attention_interval = optional!(text, &raw, full_attention_interval, 0usize);
        validate_full_attention_interval(full_attention_interval, &layer_types)?;

        let attn_output_gate = optional!(text, &raw, attn_output_gate, true);
        if layer_types.contains(&Qwen35LayerType::Full) && !attn_output_gate {
            return Err(start(
                "full attention requires the packed query/output gate",
            ));
        }

        let rope_nested = text.and_then(|config| config.rope_parameters.as_ref());
        let rope_root = raw.rope_parameters.as_ref();
        let rope_type = merge_values(
            [
                rope_nested.and_then(|rope| rope.rope_type.clone()),
                rope_root.and_then(|rope| rope.rope_type.clone()),
                text.and_then(|config| config.rope_type.clone()),
                raw.rope_type.clone(),
            ],
            "rope_type",
        )?;
        if rope_type.as_deref().is_some_and(|kind| kind != "default") {
            return Err(start("only default rope_type is supported"));
        }
        let rope_theta = merge_values(
            [
                rope_nested.and_then(|rope| rope.rope_theta),
                rope_root.and_then(|rope| rope.rope_theta),
                text.and_then(|config| config.rope_theta),
                raw.rope_theta,
            ],
            "rope_theta",
        )?
        .unwrap_or(DEFAULT_ROPE_THETA);
        let rope_theta = positive_f32(rope_theta, "rope_theta")?;
        let rms_norm_eps = positive_f32(
            resolve_field(text, &raw, |config| config.rms_norm_eps, "rms_norm_eps")?
                .unwrap_or(DEFAULT_RMS_NORM_EPS),
            "rms_norm_eps",
        )?;
        let partial_rotary_factor = merge_values(
            [
                rope_nested.and_then(|rope| rope.partial_rotary_factor),
                rope_root.and_then(|rope| rope.partial_rotary_factor),
                text.and_then(|config| config.partial_rotary_factor),
                raw.partial_rotary_factor,
            ],
            "partial_rotary_factor",
        )?
        .unwrap_or(1.0);
        let rotary_dim = rotary_width(head_dim, partial_rotary_factor)?;
        let mrope_interleaved = merge_values(
            [
                rope_nested.and_then(|rope| rope.mrope_interleaved),
                rope_root.and_then(|rope| rope.mrope_interleaved),
            ],
            "mrope_interleaved",
        )?
        .unwrap_or(false);
        let raw_sections = merge_values(
            [
                rope_nested.and_then(|rope| rope.mrope_section.clone()),
                rope_root.and_then(|rope| rope.mrope_section.clone()),
            ],
            "mrope_section",
        )?;
        let mrope_section = match raw_sections {
            Some(sections) => Some(validate_mrope_sections(&sections, rotary_dim)?),
            None if mrope_interleaved => {
                return Err(start("interleaved mRoPE requires mrope_section"))
            }
            None => None,
        };

        let linear_num_key_heads = optional!(text, &raw, linear_num_key_heads, 0);
        let linear_num_value_heads = optional!(text, &raw, linear_num_value_heads, 0);
        let linear_key_head_dim = optional!(text, &raw, linear_key_head_dim, DELTA_HEAD_DIM);
        let linear_value_head_dim = optional!(text, &raw, linear_value_head_dim, DELTA_HEAD_DIM);
        let linear_conv_kernel_dim = optional!(
            text,
            &raw,
            linear_conv_kernel_dim,
            DEFAULT_LINEAR_CONV_KERNEL_DIM
        );

        let text_geometry = Qwen35MoeTextGeometry {
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rotary_dim,
            vocab_size,
            max_position_embeddings,
            linear_num_key_heads,
            linear_num_value_heads,
            linear_key_head_dim,
            linear_value_head_dim,
            linear_conv_kernel_dim,
            rms_norm_eps,
            rope_theta,
            mrope_interleaved,
            mrope_section,
            layer_types,
            attn_output_gate,
        };
        validate_text_geometry(&text_geometry)?;

        let num_experts = positive_usize(
            resolve_field(text, &raw, |config| config.num_experts, "num_experts")?,
            "num_experts",
        )?;
        let num_experts_per_tok = resolve_field(
            text,
            &raw,
            |config| config.num_experts_per_tok,
            "num_experts_per_tok",
        )?
        .unwrap_or(DEFAULT_EXPERTS_PER_TOK);
        if num_experts_per_tok == 0 || num_experts_per_tok > num_experts {
            return Err(start("num_experts_per_tok must be in 1..=num_experts"));
        }
        let moe_intermediate_size = positive_usize(
            resolve_field(
                text,
                &raw,
                |config| config.moe_intermediate_size,
                "moe_intermediate_size",
            )?,
            "moe_intermediate_size",
        )?;
        let shared_expert_intermediate_size = resolve_field(
            text,
            &raw,
            |config| config.shared_expert_intermediate_size,
            "shared_expert_intermediate_size",
        )?
        .unwrap_or(moe_intermediate_size);
        if shared_expert_intermediate_size == 0 {
            return Err(start("shared_expert_intermediate_size must be nonzero"));
        }
        let norm_topk_prob =
            resolve_field(text, &raw, |config| config.norm_topk_prob, "norm_topk_prob")?
                .unwrap_or(true);
        let router_aux_loss_coef = resolve_field(
            text,
            &raw,
            |config| config.router_aux_loss_coef,
            "router_aux_loss_coef",
        )?
        .map(|value| {
            let narrowed = value as f32;
            if !value.is_finite() || value < 0.0 || !narrowed.is_finite() {
                return Err(start("router_aux_loss_coef must be finite and nonnegative"));
            }
            Ok(narrowed)
        })
        .transpose()?;

        let mlp_only = resolve_field(
            text,
            &raw,
            |config| config.mlp_only_layers.clone(),
            "mlp_only_layers",
        )?
        .unwrap_or_default();
        if !mlp_only.is_empty() {
            return Err(start(
                "Qwen3.5-MoE Native has one global FFN kind; mlp_only_layers overrides are unsupported",
            ));
        }
        let decoder_sparse_step = optional!(text, &raw, decoder_sparse_step, 1usize);
        if decoder_sparse_step != 1 {
            return Err(start("decoder_sparse_step must be 1 for Qwen3.5-MoE"));
        }

        let dtype = match resolve_field(text, &raw, |config| config.dtype.clone(), "dtype")? {
            None => Qwen35MoeDtype::Bf16,
            Some(kind) if kind == "bfloat16" || kind == "bf16" => Qwen35MoeDtype::Bf16,
            Some(kind) => {
                return Err(start(format!(
                    "unsupported Qwen3.5-MoE dtype {kind}; only bfloat16 is admitted"
                )))
            }
        };
        let tie_word_embeddings = optional!(text, &raw, tie_word_embeddings, false);
        let mtp_num_hidden_layers = optional!(text, &raw, mtp_num_hidden_layers, 0usize);
        if mtp_num_hidden_layers > 1 {
            return Err(start(
                "at most one MTP layer is defined by the current Qwen3.5-MoE checkpoint schema",
            ));
        }
        let vision_depth = raw
            .vision_config
            .as_ref()
            .and_then(|vision| vision.depth)
            .unwrap_or(0);
        let bidirectional = optional!(text, &raw, use_bidirectional_attention, false);
        let attention_mode = resolve_field(
            text,
            &raw,
            |config| config.decision_attention_mode.clone(),
            "decision_attention_mode",
        )?
        .unwrap_or_else(|| "causal".to_owned());
        let full_attention_noncausal = match attention_mode.as_str() {
            "causal" => {
                if bidirectional {
                    return Err(start(
                        "use_bidirectional_attention conflicts with causal decision_attention_mode",
                    ));
                }
                false
            }
            "noncausal_full_attention" => true,
            other => {
                return Err(start(format!(
                    "unsupported ordinary full-attention mode {other}"
                )))
            }
        };

        let config = Self {
            moe: Qwen35MoeSpec {
                num_experts,
                num_experts_per_tok,
                moe_intermediate_size,
                shared_expert_intermediate_size,
                norm_topk_prob,
                router_aux_loss_coef,
            },
            text: text_geometry,
            dtype,
            root_model_type: root_model_type.to_owned(),
            text_model_type: text_model_type.to_owned(),
            architectures: merge_values(
                [
                    text.and_then(|config| config.architectures.clone()),
                    raw.architectures.clone(),
                ],
                "architectures",
            )?
            .unwrap_or_default(),
            tie_word_embeddings,
            vision_depth,
            mtp_num_hidden_layers,
            full_attention_noncausal,
        };
        config.validate_shapes()?;
        Ok(config)
    }

    fn validate_shapes(&self) -> Result<(), BackendError> {
        let tensor = self.moe.checked_tensor_elements(self.text.hidden_size)?;
        for (name, count) in [
            ("router", tensor.router),
            ("routed gate/up", tensor.routed_gate_up),
            ("routed down", tensor.routed_down),
            ("shared gate/up", tensor.shared_gate_up),
            ("shared down", tensor.shared_down),
            ("shared gate", tensor.shared_gate),
        ] {
            if count == 0 {
                return Err(start(format!("{name} shape is empty")));
            }
        }
        elements(&[self.text.vocab_size, self.text.hidden_size])?;
        let q = self.text.q_width()?;
        let kv = self.text.kv_width()?;
        q.checked_mul(2)
            .ok_or_else(|| start("query-gate width overflow"))?;
        kv.checked_add(q)
            .ok_or_else(|| start("full attention width overflow"))?;
        if self.text.linear_attention_layers() > 0 {
            let key = checked_product(
                self.text.linear_num_key_heads,
                self.text.linear_key_head_dim,
                "linear key width",
            )?;
            let value = checked_product(
                self.text.linear_num_value_heads,
                self.text.linear_value_head_dim,
                "linear value width",
            )?;
            key.checked_mul(2)
                .and_then(|width| width.checked_add(value))
                .ok_or_else(|| start("linear QKV width overflow"))?;
        }
        Ok(())
    }

    /// Pure contract selection. `cuda_bf16` is supplied by the caller; this
    /// cold foundation never queries a device. The result names the Native
    /// path whose semantics the provider would use.
    pub(crate) fn native_contract(
        &self,
        cuda_bf16: bool,
    ) -> crate::qwen35_moe_contract::Qwen35MoeNativeContract {
        crate::qwen35_moe_contract::select_native_contract(cuda_bf16, self)
    }

    pub(crate) fn admission(&self) -> Qwen35MoeAdmission {
        let mut blockers = vec![
            "Oxide Qwen3.5-MoE text executor is not implemented",
            "full-attention and gated-delta device operators for MoE are not admitted",
            "warm quality/performance parity has not run",
        ];
        if self.vision_depth > 0 {
            blockers.push("vision tower and image tokens are not implemented");
        }
        if self.mtp_num_hidden_layers > 0 {
            blockers.push("MTP head is present in the checkpoint but not implemented");
        }
        Qwen35MoeAdmission {
            config_admitted: true,
            executor_implemented: false,
            whole_checkpoint_implemented: false,
            blockers,
        }
    }
}

fn validate_full_attention_interval(
    interval: usize,
    layer_types: &[Qwen35LayerType],
) -> Result<(), BackendError> {
    if interval == 0 {
        return Ok(());
    }
    let full = layer_types
        .iter()
        .filter(|kind| **kind == Qwen35LayerType::Full)
        .count();
    let expected = layer_types.len() / interval;
    if expected == 0 || full != expected {
        return Err(start(format!(
            "full_attention_interval {interval} implies {expected} full-attention layers, found {full}"
        )));
    }
    for (index, kind) in layer_types.iter().enumerate() {
        let scheduled = (index + 1).is_multiple_of(interval);
        if scheduled != (*kind == Qwen35LayerType::Full) {
            return Err(start(format!(
                "layer_types[{index}] disagrees with full_attention_interval {interval}"
            )));
        }
    }
    Ok(())
}

fn validate_text_geometry(geometry: &Qwen35MoeTextGeometry) -> Result<(), BackendError> {
    if !geometry
        .num_attention_heads
        .is_multiple_of(geometry.num_key_value_heads)
    {
        return Err(start(
            "num_attention_heads must divide evenly by num_key_value_heads",
        ));
    }
    if geometry.rotary_dim > geometry.head_dim || !geometry.rotary_dim.is_multiple_of(2) {
        return Err(start(
            "rotary_dim must be even, positive and no wider than head_dim",
        ));
    }
    if !geometry.rms_norm_eps.is_finite() || geometry.rms_norm_eps <= 0.0 {
        return Err(start("rms_norm_eps must be finite and positive"));
    }
    if !geometry.rope_theta.is_finite() || geometry.rope_theta <= 0.0 {
        return Err(start("rope_theta must be finite and positive"));
    }
    if u32::try_from(geometry.max_position_embeddings).is_err() {
        return Err(start("max_position_embeddings must fit CUDA position IDs"));
    }
    geometry.q_width()?;
    geometry.kv_width()?;
    geometry
        .q_width()?
        .checked_mul(2)
        .ok_or_else(|| start("query-gate width overflow"))?;
    if geometry.linear_attention_layers() > 0 {
        if geometry.linear_key_head_dim != DELTA_HEAD_DIM
            || geometry.linear_value_head_dim != DELTA_HEAD_DIM
        {
            return Err(start(
                "native Qwen3.5 GDN requires 128-wide key/value heads",
            ));
        }
        for (name, value) in [
            ("linear_num_key_heads", geometry.linear_num_key_heads),
            ("linear_num_value_heads", geometry.linear_num_value_heads),
            ("linear_key_head_dim", geometry.linear_key_head_dim),
            ("linear_value_head_dim", geometry.linear_value_head_dim),
            ("linear_conv_kernel_dim", geometry.linear_conv_kernel_dim),
        ] {
            if value == 0 {
                return Err(start(format!("config {name} must be nonzero")));
            }
        }
        if !geometry
            .linear_num_value_heads
            .is_multiple_of(geometry.linear_num_key_heads)
        {
            return Err(start(
                "linear_num_value_heads must divide evenly by linear_num_key_heads",
            ));
        }
    }
    Ok(())
}

fn positive_usize(value: Option<usize>, name: &str) -> Result<usize, BackendError> {
    let value = value.ok_or_else(|| start(format!("missing config field {name}")))?;
    if value == 0 || value > i32::MAX as usize {
        return Err(start(format!("{name} must be a positive i32")));
    }
    Ok(value)
}

pub(crate) fn checked_product(
    left: usize,
    right: usize,
    name: &str,
) -> Result<usize, BackendError> {
    left.checked_mul(right)
        .ok_or_else(|| start(format!("{name} overflow")))
}

fn positive_f32(value: f64, name: &str) -> Result<f32, BackendError> {
    let narrowed = value as f32;
    if !value.is_finite() || value <= 0.0 || !narrowed.is_finite() || narrowed <= 0.0 {
        return Err(start(format!(
            "{name} must be finite, positive and representable as f32"
        )));
    }
    Ok(narrowed)
}

fn rotary_width(head_dim: usize, factor: f64) -> Result<usize, BackendError> {
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
    if !rotary.is_multiple_of(2) {
        return Err(start("rotary_dim must be positive and even"));
    }
    Ok(rotary)
}

fn validate_mrope_sections(
    sections: &[usize],
    rotary_dim: usize,
) -> Result<[usize; 3], BackendError> {
    if sections.len() != 3 || sections.contains(&0) {
        return Err(start(
            "mrope_section must contain three positive text sections",
        ));
    }
    let total = sections
        .iter()
        .try_fold(0usize, |total, section| total.checked_add(*section))
        .ok_or_else(|| start("mrope_section width overflow"))?;
    if total != rotary_dim / 2 {
        return Err(start(format!(
            "mrope_section sums to {total}, expected {}",
            rotary_dim / 2
        )));
    }
    Ok([sections[0], sections[1], sections[2]])
}

fn active(value: &Option<Value>) -> bool {
    value.as_ref().is_some_and(|value| !value.is_null())
}

fn resolve_field<T: PartialEq>(
    nested: Option<&RawQwen35MoeConfig>,
    root: &RawQwen35MoeConfig,
    field: fn(&RawQwen35MoeConfig) -> Option<T>,
    name: &str,
) -> Result<Option<T>, BackendError> {
    match (nested.and_then(field), field(root)) {
        (Some(nested), Some(root)) if nested != root => Err(start(format!(
            "conflicting {name} values across text_config and root"
        ))),
        (Some(nested), _) => Ok(Some(nested)),
        (None, root) => Ok(root),
    }
}

fn merge_values<T, I>(values: I, name: &str) -> Result<Option<T>, BackendError>
where
    T: PartialEq,
    I: IntoIterator<Item = Option<T>>,
{
    let mut selected = None;
    for value in values.into_iter().flatten() {
        if let Some(previous) = selected.as_ref() {
            if previous != &value {
                return Err(start(format!("conflicting {name} selectors")));
            }
        } else {
            selected = Some(value);
        }
    }
    Ok(selected)
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
struct RawRopeParameters {
    rope_type: Option<String>,
    rope_theta: Option<f64>,
    partial_rotary_factor: Option<f64>,
    mrope_interleaved: Option<bool>,
    mrope_section: Option<Vec<usize>>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
struct RawVisionConfig {
    depth: Option<usize>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub(crate) struct RawQwen35MoeConfig {
    model_type: Option<String>,
    architectures: Option<Vec<String>>,
    dtype: Option<String>,
    hidden_size: Option<usize>,
    num_hidden_layers: Option<usize>,
    num_attention_heads: Option<usize>,
    num_key_value_heads: Option<usize>,
    head_dim: Option<usize>,
    vocab_size: Option<usize>,
    max_position_embeddings: Option<usize>,
    rms_norm_eps: Option<f64>,
    rope_theta: Option<f64>,
    rope_type: Option<String>,
    partial_rotary_factor: Option<f64>,
    rope_parameters: Option<RawRopeParameters>,
    rope_scaling: Option<Value>,
    layer_types: Option<Vec<String>>,
    full_attention_interval: Option<usize>,
    attn_output_gate: Option<bool>,
    attention_bias: Option<bool>,
    hidden_act: Option<String>,
    linear_num_key_heads: Option<usize>,
    linear_num_value_heads: Option<usize>,
    linear_key_head_dim: Option<usize>,
    linear_value_head_dim: Option<usize>,
    linear_conv_kernel_dim: Option<usize>,
    num_experts: Option<usize>,
    num_experts_per_tok: Option<usize>,
    moe_intermediate_size: Option<usize>,
    shared_expert_intermediate_size: Option<usize>,
    shared_expert_gate: Option<bool>,
    n_shared_experts: Option<usize>,
    num_shared_experts: Option<usize>,
    norm_topk_prob: Option<bool>,
    router_aux_loss_coef: Option<f64>,
    mlp_only_layers: Option<Vec<usize>>,
    decoder_sparse_step: Option<usize>,
    tie_word_embeddings: Option<bool>,
    mtp_num_hidden_layers: Option<usize>,
    use_bidirectional_attention: Option<bool>,
    decision_attention_mode: Option<String>,
    quantization_config: Option<Value>,
    vision_config: Option<RawVisionConfig>,
    text_config: Option<Box<RawQwen35MoeConfig>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tiny() -> Value {
        json!({
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "dtype": "bfloat16",
                "hidden_size": 16,
                "num_hidden_layers": 4,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "vocab_size": 32,
                "max_position_embeddings": 64,
                "rms_norm_eps": 1e-6,
                "layer_types": [
                    "linear_attention", "linear_attention",
                    "linear_attention", "full_attention"
                ],
                "full_attention_interval": 4,
                "attn_output_gate": true,
                "hidden_act": "silu",
                "linear_num_key_heads": 1,
                "linear_num_value_heads": 2,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 6,
                "shared_expert_intermediate_size": 6,
                "norm_topk_prob": true,
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": 10_000_000.0,
                    "partial_rotary_factor": 1.0
                }
            }
        })
    }

    fn with(key: &str, value: Value) -> Value {
        let mut config = tiny();
        config["text_config"][key] = value;
        config
    }

    #[test]
    fn tiny_checked_config_records_exact_router_and_schedule() {
        let config = Qwen35MoeConfig::from_json(&tiny()).unwrap();
        assert_eq!(config.text.num_hidden_layers, 4);
        assert_eq!(config.text.full_attention_layers(), 1);
        assert_eq!(config.text.linear_attention_layers(), 3);
        assert_eq!(config.moe.num_experts, 4);
        assert_eq!(config.moe.num_experts_per_tok, 2);
        assert!(config.moe.norm_topk_prob);
        assert_eq!(config.moe.shared_expert_intermediate_size, 6);
        assert_eq!(config.dtype, Qwen35MoeDtype::Bf16);
        assert!(!config.admission().executor_implemented);
    }

    #[test]
    fn cached_actual_metadata_is_cold_and_complete() {
        let path = Path::new(
            "/workspace/model-performance/michaelfeil/onejev-perf/models/qwen35-35b-a3b/config.json",
        );
        if !path.is_file() {
            if std::env::var_os("QWEN35_MOE_REQUIRE_CACHED").is_some() {
                panic!("cached 35B config required but absent");
            }
            eprintln!("cached 35B config absent; skipping metadata test");
            return;
        }
        let value: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let config = Qwen35MoeConfig::from_json(&value).unwrap();
        assert_eq!(config.text.hidden_size, 2048);
        assert_eq!(config.text.num_hidden_layers, 40);
        assert_eq!(config.text.num_attention_heads, 16);
        assert_eq!(config.text.num_key_value_heads, 2);
        assert_eq!(config.text.head_dim, 256);
        assert_eq!(config.text.rotary_dim, 64);
        assert_eq!(config.text.mrope_section, Some([11, 11, 10]));
        assert_eq!(config.text.full_attention_layers(), 10);
        assert_eq!(config.text.linear_attention_layers(), 30);
        assert_eq!(config.moe.num_experts, 256);
        assert_eq!(config.moe.num_experts_per_tok, 8);
        assert_eq!(config.moe.moe_intermediate_size, 512);
        assert_eq!(config.moe.shared_expert_intermediate_size, 512);
        assert!(config.moe.norm_topk_prob);
        assert_eq!(config.vision_depth, 27);
        assert_eq!(config.mtp_num_hidden_layers, 1);
        let elements = config
            .moe
            .checked_tensor_elements(config.text.hidden_size)
            .unwrap();
        assert_eq!(elements.routed_gate_up, 256 * 1024 * 2048);
        assert_eq!(elements.routed_down, 256 * 2048 * 512);
        assert!(!config.admission().whole_checkpoint_implemented);
    }

    #[test]
    fn omitted_rope_theta_uses_native_ten_thousand_default() {
        let mut value = tiny();
        value["text_config"]["rope_parameters"]
            .as_object_mut()
            .unwrap()
            .remove("rope_theta");
        let config = Qwen35MoeConfig::from_json(&value).unwrap();
        assert_eq!(config.text.rope_theta, 10_000.0);

        let mut native = tiny();
        native["text_config"]["num_experts"] = json!(256);
        native["text_config"]["num_experts_per_tok"] = json!(8);
        native["text_config"]["moe_intermediate_size"] = json!(8);
        native["text_config"]["shared_expert_intermediate_size"] = json!(8);
        let native = Qwen35MoeConfig::from_json(&native).unwrap();
        assert_eq!(
            native.native_contract(true),
            crate::qwen35_moe_contract::Qwen35MoeNativeContract::RoutedMoeCuda
        );
        assert_eq!(
            native.native_contract(false),
            crate::qwen35_moe_contract::Qwen35MoeNativeContract::DecoderMoeForwardFallback
        );
    }

    #[test]
    fn rejects_dense_types_bad_schedules_and_future_dtypes() {
        for value in [
            with("layer_types", json!(["full_attention"])),
            with(
                "layer_types",
                json!([
                    "sliding_attention",
                    "linear_attention",
                    "linear_attention",
                    "full_attention"
                ]),
            ),
            with("full_attention_interval", json!(2)),
            with("mlp_only_layers", json!([1])),
            with("decoder_sparse_step", json!(2)),
            with("dtype", json!("float8_e4m3fn")),
            with("dtype", json!("float32")),
            with("num_experts_per_tok", json!(5)),
            with("num_experts", json!(0)),
            with("moe_intermediate_size", json!(0)),
            with("shared_expert_gate", json!(false)),
            with("quantization_config", json!({"quant_method": "fp8"})),
        ] {
            assert!(Qwen35MoeConfig::from_json(&value).is_err(), "{value}");
        }
        let mut dense = tiny();
        dense["model_type"] = json!("qwen3_5");
        assert!(Qwen35MoeConfig::from_json(&dense).is_err());
    }

    #[test]
    fn rejects_conflicting_nested_and_root_geometry() {
        let mut config = tiny();
        config["hidden_size"] = json!(32);
        assert!(Qwen35MoeConfig::from_json(&config).is_err());
        let mut config = tiny();
        config["text_config"]["num_experts"] = json!(8);
        config["num_experts"] = json!(4);
        assert!(Qwen35MoeConfig::from_json(&config).is_err());
    }
    #[test]
    fn actual_metadata_fixture_with_native_bidirectional_selector_is_noncausal() {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/qwen35-moe-bidirectional-config.json");
        let value: Value = serde_json::from_slice(&std::fs::read(&fixture).unwrap()).unwrap();
        let config = Qwen35MoeConfig::from_json(&value).unwrap();
        assert!(config.full_attention_noncausal);
        assert_eq!(config.text.hidden_size, 2048);
        assert_eq!(config.text.num_hidden_layers, 40);
        assert_eq!(config.text.num_attention_heads, 16);
        assert_eq!(config.text.num_key_value_heads, 2);
        assert_eq!(config.text.head_dim, 256);
        assert_eq!(config.moe.num_experts, 256);
        assert_eq!(config.moe.num_experts_per_tok, 8);

        // The original checkpoint config is untouched and remains causal.
        let actual = Path::new("/workspace/model-performance/michaelfeil/onejev-perf/models/qwen35-35b-a3b/config.json");
        if actual.is_file() {
            let original = Qwen35MoeConfig::load(actual.parent().unwrap()).unwrap();
            assert!(!original.full_attention_noncausal);
        }
    }
}
