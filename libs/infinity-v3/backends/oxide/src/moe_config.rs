// SPDX-License-Identifier: LicenseRef-Infinity-Restricted-MIT-1.0

use crate::model::{elements, start};
use infinity_v3_backend_core::BackendError;
use serde_json::Value;
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RoutedMlpConfig {
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) moe_intermediate_size: usize,
    pub(crate) num_experts: usize,
    pub(crate) num_experts_per_tok: usize,
    pub(crate) decoder_sparse_step: usize,
    pub(crate) norm_topk_prob: bool,
    pub(crate) mlp_only_layers: BTreeSet<usize>,
}

impl RoutedMlpConfig {
    pub(crate) fn from_json(value: &Value) -> Result<Self, BackendError> {
        let object = value
            .as_object()
            .ok_or_else(|| start("qwen3_moe config must be a JSON object"))?;
        if object.get("model_type").and_then(Value::as_str) != Some("qwen3_moe") {
            return Err(start("model_type must be qwen3_moe"));
        }
        for field in [
            "quantization_config",
            "quantization",
            "shared_expert_intermediate_size",
            "n_shared_experts",
            "num_shared_experts",
            "shared_expert_gate",
        ] {
            if let Some(value) = object.get(field) {
                if !value.is_null() {
                    return Err(start(format!("{field} is not supported")));
                }
            }
        }

        let hidden_size = positive_usize(object.get("hidden_size"), "hidden_size")?;
        let intermediate_size =
            positive_usize(object.get("intermediate_size"), "intermediate_size")?;
        let num_hidden_layers =
            positive_usize(object.get("num_hidden_layers"), "num_hidden_layers")?;
        let moe_intermediate_size =
            positive_usize(object.get("moe_intermediate_size"), "moe_intermediate_size")?;
        let num_experts = positive_usize(object.get("num_experts"), "num_experts")?;
        let num_experts_per_tok =
            positive_usize(object.get("num_experts_per_tok"), "num_experts_per_tok")?;
        if num_experts_per_tok > num_experts {
            return Err(start("num_experts_per_tok must not exceed num_experts"));
        }

        let decoder_sparse_step = match object.get("decoder_sparse_step") {
            Some(value) => positive_usize(Some(value), "decoder_sparse_step")?,
            None => 1,
        };

        match object.get("hidden_act") {
            None => {}
            Some(Value::String(activation)) if activation.as_str() == "silu" => {}
            Some(_) => return Err(start("hidden_act must be silu")),
        }

        let norm_topk_prob = match object.get("norm_topk_prob") {
            None => true,
            Some(Value::Bool(enabled)) => *enabled,
            Some(_) => return Err(start("norm_topk_prob must be a boolean")),
        };

        let mut mlp_only_layers = BTreeSet::new();
        match object.get("mlp_only_layers") {
            None => {}
            Some(Value::Array(indices)) => {
                for item in indices {
                    let raw = item.as_u64().ok_or_else(|| {
                        start("mlp_only_layers entries must be nonnegative integers")
                    })?;
                    if raw > usize::MAX as u64 {
                        return Err(start("mlp_only_layers entry does not fit usize"));
                    }
                    let index = raw as usize;
                    if index >= num_hidden_layers {
                        return Err(start("mlp_only_layers entry out of range"));
                    }
                    mlp_only_layers.insert(index);
                }
            }
            Some(_) => return Err(start("mlp_only_layers must be an array")),
        }

        let config = Self {
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            moe_intermediate_size,
            num_experts,
            num_experts_per_tok,
            decoder_sparse_step,
            norm_topk_prob,
            mlp_only_layers,
        };
        config.validate_shapes()?;
        Ok(config)
    }

    pub(crate) fn is_routed_layer(&self, index: usize) -> Result<bool, BackendError> {
        let next = index
            .checked_add(1)
            .ok_or_else(|| start("layer index overflows usize"))?;
        if index >= self.num_hidden_layers {
            return Err(start(format!(
                "layer index {index} is outside num_hidden_layers {}",
                self.num_hidden_layers
            )));
        }
        if self.decoder_sparse_step == 0 {
            return Err(start("decoder_sparse_step must be positive"));
        }
        Ok(!self.mlp_only_layers.contains(&index) && next % self.decoder_sparse_step == 0)
    }

    fn validate_shapes(&self) -> Result<(), BackendError> {
        let expert_width = self
            .moe_intermediate_size
            .checked_mul(2)
            .ok_or_else(|| start("moe_intermediate_size * 2 overflows usize"))?;
        let dense_width = self
            .intermediate_size
            .checked_mul(2)
            .ok_or_else(|| start("intermediate_size * 2 overflows usize"))?;
        elements(&[self.num_experts, self.hidden_size])?;
        elements(&[self.num_experts, expert_width, self.hidden_size])?;
        elements(&[
            self.num_experts,
            self.hidden_size,
            self.moe_intermediate_size,
        ])?;
        elements(&[dense_width, self.hidden_size])?;
        Ok(())
    }
}

fn positive_usize(value: Option<&Value>, field: &str) -> Result<usize, BackendError> {
    let value = value.ok_or_else(|| start(format!("missing {field}")))?;
    let raw = value
        .as_u64()
        .ok_or_else(|| start(format!("{field} must be a positive integer")))?;
    if raw == 0 || raw > i32::MAX as u64 {
        return Err(start(format!("{field} must be a positive i32")));
    }
    Ok(raw as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn base() -> Value {
        json!({"model_type":"qwen3_moe","hidden_size":4,"intermediate_size":4,
            "num_hidden_layers":4,"moe_intermediate_size":2,
            "num_experts":4,"num_experts_per_tok":2})
    }
    fn parse(overrides: &[(&str, Value)]) -> Result<RoutedMlpConfig, BackendError> {
        let mut value = base();
        if let Value::Object(object) = &mut value {
            for (key, item) in overrides {
                object.insert((*key).to_owned(), item.clone());
            }
        }
        RoutedMlpConfig::from_json(&value)
    }

    #[test]
    fn large_expert_tensor_metadata_needs_no_payload_allocation() {
        let config = RoutedMlpConfig::from_json(&json!({
            "model_type":"qwen3_moe","hidden_size":2048,"intermediate_size":6144,
            "num_hidden_layers":48,"moe_intermediate_size":768,
            "num_experts":128,"num_experts_per_tok":8
        }))
        .unwrap();
        assert_eq!(config.num_experts, 128);
        assert!(config.is_routed_layer(47).unwrap());
        assert!(config.is_routed_layer(48).is_err());
    }

    #[test]
    fn routing_defaults_sparse_step_and_dense_override() -> Result<(), BackendError> {
        let config = parse(&[
            ("decoder_sparse_step", json!(2)),
            ("mlp_only_layers", json!([1, 1])),
        ])?;
        assert!(config.norm_topk_prob);
        assert_eq!(config.mlp_only_layers.len(), 1);
        assert!(!config.is_routed_layer(1)?);
        assert!(!config.is_routed_layer(2)?);
        assert!(config.is_routed_layer(3)?);
        assert!(config.is_routed_layer(4).is_err());
        let default = parse(&[])?;
        assert_eq!(default.decoder_sparse_step, 1);
        assert!(default.mlp_only_layers.is_empty());
        assert!(default.is_routed_layer(0)?);
        Ok(())
    }

    #[test]
    fn rejects_bad_dimensions_types_and_topk() {
        for (key, value) in [
            ("hidden_size", json!(0)),
            ("hidden_size", json!(-1)),
            ("hidden_size", json!("4")),
            ("moe_intermediate_size", json!(1.5)),
            ("num_experts_per_tok", json!(5)),
            ("num_hidden_layers", json!(null)),
        ] {
            assert!(parse(&[(key, value)]).is_err());
        }
    }

    #[test]
    fn rejects_bad_layer_indices_and_model_type() {
        for value in [json!([4]), json!([-1]), json!([0, "1"]), json!({})] {
            assert!(parse(&[("mlp_only_layers", value)]).is_err());
        }
        assert!(parse(&[("model_type", json!("qwen2_moe"))]).is_err());
        assert!(RoutedMlpConfig::from_json(&json!([])).is_err());
    }

    #[test]
    fn rejects_huge_products_and_bad_sparse_step() {
        for (key, value) in [
            ("hidden_size", json!(i32::MAX)),
            ("moe_intermediate_size", json!(i32::MAX)),
            ("num_hidden_layers", json!(i64::from(i32::MAX) + 1)),
            ("decoder_sparse_step", json!(0)),
            ("decoder_sparse_step", json!(-1)),
            ("decoder_sparse_step", json!(true)),
        ] {
            assert!(parse(&[(key, value)]).is_err());
        }
    }

    #[test]
    fn rejects_quantization_and_shared_experts() {
        for field in [
            "quantization_config",
            "quantization",
            "shared_expert_intermediate_size",
            "n_shared_experts",
            "num_shared_experts",
            "shared_expert_gate",
        ] {
            assert!(parse(&[(field, json!({}))]).is_err());
        }
        assert!(parse(&[("quantization_config", json!(null))]).is_ok());
    }
    #[test]
    fn activation_and_norm_types() -> Result<(), BackendError> {
        for value in [json!("gelu"), json!(null), json!(1)] {
            assert!(parse(&[("hidden_act", value)]).is_err());
        }
        for value in [json!("true"), json!(1), json!(null)] {
            assert!(parse(&[("norm_topk_prob", value)]).is_err());
        }
        let config = parse(&[
            ("hidden_act", json!("silu")),
            ("norm_topk_prob", json!(false)),
        ])?;
        assert!(!config.norm_topk_prob);
        Ok(())
    }
}
