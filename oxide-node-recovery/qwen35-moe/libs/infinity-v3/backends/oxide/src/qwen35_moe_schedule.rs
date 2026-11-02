// Cold layer schedule for the hybrid Qwen3.5-MoE text backbone.
//
// Native selects the attention implementation from `layer_types[i]` and the
// feed-forward implementation from one global `num_experts` value. Every
// decoder layer therefore uses the same routed + shared FFN geometry while its
// attention block alternates between full attention and gated delta. This
// module makes that schedule explicit and checked; it does not duplicate the
// dense Qwen3.5 executor.

use crate::qwen35_config::Qwen35LayerType;
use crate::qwen35_moe_config::{checked_product, Qwen35MoeConfig};
use infinity_v3_backend_core::BackendError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MoeLayerShapes {
    pub(crate) router: [usize; 2],
    pub(crate) routed_gate_up: [usize; 3],
    pub(crate) routed_down: [usize; 3],
    pub(crate) shared_gate_up: [usize; 2],
    pub(crate) shared_down: [usize; 2],
    pub(crate) shared_gate: [usize; 2],
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Qwen35MoeLayerPlan {
    pub(crate) index: usize,
    pub(crate) attention: Qwen35LayerType,
    pub(crate) shapes: MoeLayerShapes,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Qwen35MoeSchedule {
    layers: Vec<Qwen35MoeLayerPlan>,
    full_attention_layers: usize,
    linear_attention_layers: usize,
}

impl Qwen35MoeSchedule {
    pub(crate) fn new(config: &Qwen35MoeConfig) -> Result<Self, BackendError> {
        let text = &config.text;
        let moe = &config.moe;
        if text.layer_types.len() != text.num_hidden_layers {
            return Err(infinity_v3_backend_core::BackendError::Start(format!(
                "Qwen3.5-MoE schedule has {} layer types for {} layers",
                text.layer_types.len(),
                text.num_hidden_layers
            )));
        }
        let hidden = text.hidden_size;
        let two_i = moe.gate_up_width()?;
        let two_s = moe.shared_gate_up_width()?;
        let shapes = MoeLayerShapes {
            router: [moe.num_experts, hidden],
            routed_gate_up: [moe.num_experts, two_i, hidden],
            routed_down: [moe.num_experts, hidden, moe.moe_intermediate_size],
            shared_gate_up: [two_s, hidden],
            shared_down: [hidden, moe.shared_expert_intermediate_size],
            shared_gate: [1, hidden],
        };
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(text.num_hidden_layers)
            .map_err(|_| {
                infinity_v3_backend_core::BackendError::Start(
                    "Qwen3.5-MoE layer allocation failed".into(),
                )
            })?;
        let mut full_attention_layers = 0usize;
        let mut linear_attention_layers = 0usize;
        for (index, attention) in text.layer_types.iter().copied().enumerate() {
            match attention {
                Qwen35LayerType::Full => full_attention_layers += 1,
                Qwen35LayerType::Linear => linear_attention_layers += 1,
            }
            layers.push(Qwen35MoeLayerPlan {
                index,
                attention,
                shapes,
            });
        }
        if full_attention_layers
            .checked_add(linear_attention_layers)
            .ok_or_else(|| {
                infinity_v3_backend_core::BackendError::Start(
                    "Qwen3.5-MoE layer count overflow".into(),
                )
            })?
            != text.num_hidden_layers
        {
            return Err(infinity_v3_backend_core::BackendError::Start(
                "Qwen3.5-MoE layer count mismatch".into(),
            ));
        }
        Ok(Self {
            layers,
            full_attention_layers,
            linear_attention_layers,
        })
    }

    pub(crate) fn layers(&self) -> &[Qwen35MoeLayerPlan] {
        &self.layers
    }

    pub(crate) fn layer(&self, index: usize) -> Result<&Qwen35MoeLayerPlan, BackendError> {
        self.layers.get(index).ok_or_else(|| {
            infinity_v3_backend_core::BackendError::Start(format!(
                "Qwen3.5-MoE layer index {index} is outside {} scheduled layers",
                self.layers.len()
            ))
        })
    }

    pub(crate) const fn full_attention_layers(&self) -> usize {
        self.full_attention_layers
    }

    pub(crate) const fn linear_attention_layers(&self) -> usize {
        self.linear_attention_layers
    }

    /// Byte size of one layer's routed+shared FFN payload at the configured
    /// dtype. Used by cold admission/telemetry only; nothing is allocated.
    pub(crate) fn routed_payload_bytes(
        &self,
        config: &Qwen35MoeConfig,
    ) -> Result<usize, BackendError> {
        let elements = config
            .moe
            .checked_tensor_elements(config.text.hidden_size)?;
        let routed = elements
            .routed_gate_up
            .checked_add(elements.routed_down)
            .ok_or_else(|| {
                infinity_v3_backend_core::BackendError::Start(
                    "Qwen3.5-MoE routed payload overflow".into(),
                )
            })?;
        let shared = elements
            .shared_gate_up
            .checked_add(elements.shared_down)
            .and_then(|value| value.checked_add(elements.shared_gate))
            .ok_or_else(|| {
                infinity_v3_backend_core::BackendError::Start(
                    "Qwen3.5-MoE shared payload overflow".into(),
                )
            })?;
        checked_product(
            routed.checked_add(shared).ok_or_else(|| {
                infinity_v3_backend_core::BackendError::Start(
                    "Qwen3.5-MoE FFN payload overflow".into(),
                )
            })?,
            2,
            "Qwen3.5-MoE FFN BF16 payload bytes",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen35_moe_config::Qwen35MoeConfig;
    use serde_json::json;

    fn config() -> Qwen35MoeConfig {
        Qwen35MoeConfig::from_json(&json!({
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 16,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "vocab_size": 32,
            "max_position_embeddings": 64,
            "layer_types": [
                "linear_attention", "linear_attention",
                "linear_attention", "full_attention"
            ],
            "full_attention_interval": 4,
            "attn_output_gate": true,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 6,
            "shared_expert_intermediate_size": 6
        }))
        .unwrap()
    }

    #[test]
    fn hybrid_schedule_is_exact_and_checked() {
        let config = config();
        let schedule = Qwen35MoeSchedule::new(&config).unwrap();
        assert_eq!(schedule.full_attention_layers(), 1);
        assert_eq!(schedule.linear_attention_layers(), 3);
        assert_eq!(
            schedule.layer(0).unwrap().attention,
            Qwen35LayerType::Linear
        );
        assert_eq!(schedule.layer(3).unwrap().attention, Qwen35LayerType::Full);
        assert_eq!(schedule.layer(3).unwrap().shapes.router, [4, 16]);
        assert_eq!(
            schedule.layer(3).unwrap().shapes.routed_gate_up,
            [4, 12, 16]
        );
        assert_eq!(schedule.layer(3).unwrap().shapes.routed_down, [4, 16, 6]);
        assert_eq!(schedule.layer(3).unwrap().shapes.shared_gate_up, [12, 16]);
        assert_eq!(schedule.layer(3).unwrap().shapes.shared_down, [16, 6]);
        assert_eq!(schedule.layer(3).unwrap().shapes.shared_gate, [1, 16]);
        assert!(schedule.layer(4).is_err());
        assert_eq!(
            schedule.routed_payload_bytes(&config).unwrap(),
            2 * (4 * 12 * 16 + 4 * 16 * 6 + 12 * 16 + 16 * 6 + 16)
        );
    }
}
