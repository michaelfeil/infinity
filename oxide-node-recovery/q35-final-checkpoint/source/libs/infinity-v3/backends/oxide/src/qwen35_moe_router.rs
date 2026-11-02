// Scalar CPU semantic reference for the Qwen3.5-MoE router.
//
// This is NOT an exact Native numerical oracle. It models the semantic
// contract of `qwen35_moe_route_256_8_f32`:
//
//   probabilities = softmax(logits, -1)                 // FP32
//   ids           = top-8 by probability, ties by lower expert id
//   weights       = selected FP32 probabilities
//   if renormalize { weights /= sum(weights) }          // FP32
//
// CUDA performs the max and probability sums with tree reductions; this scalar
// implementation uses sequential FP32 sums. Consequently:
//
// * top-k ids are exact only for clearly separated probabilities;
// * FP32 weights are tolerance-based, never bit-exact;
// * no BF16 routing cast exists on the accelerated path.
//
// The `decoder_moe_forward` fallback casts routing to the input dtype after
// renormalisation. `fallback_bf16_weights` exposes that fallback-only view for
// contrast and must not be used to describe `routed_moe_cuda`.
//
// Expert/shared non-linearities and combine order live in
// `qwen35_moe_contract`; they model CUDA rounding points, not exact bits.

use crate::qwen35_moe_config::Qwen35MoeSpec;
use crate::qwen35_moe_contract::route_topk_f32;
use infinity_v3_backend_core::BackendError;

fn start(error: impl std::fmt::Display) -> BackendError {
    BackendError::Start(format!("Oxide qwen3.5-moe router: {error}"))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RouterPlan {
    experts: usize,
    top_k: usize,
    renormalize: bool,
    hidden: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Route {
    ids: Vec<u32>,
    weights: Vec<f32>,
}

impl RouterPlan {
    pub(crate) fn new(spec: &Qwen35MoeSpec, hidden: usize) -> Result<Self, BackendError> {
        if spec.num_experts == 0 || spec.num_experts > u32::MAX as usize {
            return Err(start("num_experts must fit a positive u32"));
        }
        if spec.num_experts_per_tok == 0 || spec.num_experts_per_tok > spec.num_experts {
            return Err(start("top-k must be in 1..=num_experts"));
        }
        if hidden == 0 {
            return Err(start("hidden size must be nonzero"));
        }
        Ok(Self {
            experts: spec.num_experts,
            top_k: spec.num_experts_per_tok,
            renormalize: spec.norm_topk_prob,
            hidden,
        })
    }

    pub(crate) const fn experts(&self) -> usize {
        self.experts
    }

    pub(crate) const fn top_k(&self) -> usize {
        self.top_k
    }

    pub(crate) const fn renormalize(&self) -> bool {
        self.renormalize
    }

    pub(crate) const fn hidden(&self) -> usize {
        self.hidden
    }

    pub(crate) fn route(&self, logits: &[f32]) -> Result<Route, BackendError> {
        if logits.len() != self.experts {
            return Err(start(format!(
                "router logits have {} values; expected {}",
                logits.len(),
                self.experts
            )));
        }
        let (ids, weights) = route_topk_f32(logits, self.top_k, self.renormalize)?;
        Ok(Route { ids, weights })
    }

    pub(crate) fn route_rows(&self, logits: &[f32]) -> Result<Vec<Route>, BackendError> {
        if !logits.len().is_multiple_of(self.experts) {
            return Err(start("router matrix width does not divide rows"));
        }
        let rows = logits.len() / self.experts;
        let mut routes = Vec::new();
        routes
            .try_reserve_exact(rows)
            .map_err(|_| start("router row allocation failed"))?;
        for row in logits.chunks_exact(self.experts) {
            routes.push(self.route(row)?);
        }
        Ok(routes)
    }
}

impl Route {
    pub(crate) fn ids(&self) -> &[u32] {
        &self.ids
    }

    pub(crate) fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Fallback-only view: `decoder_moe_forward` casts routing to the input
    /// dtype before dispatch. The accelerated Qwen3.5 path keeps FP32.
    pub(crate) fn fallback_bf16_weights(&self) -> Vec<u16> {
        self.weights
            .iter()
            .map(|weight| half::bf16::from_f32(*weight).to_bits())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen35_moe_config::Qwen35MoeConfig;
    use serde_json::json;

    fn spec(experts: usize, top: usize, renorm: bool) -> Qwen35MoeSpec {
        Qwen35MoeConfig::from_json(&json!({
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "head_dim": 8,
            "vocab_size": 8,
            "layer_types": ["full_attention"],
            "attn_output_gate": true,
            "num_experts": experts,
            "num_experts_per_tok": top,
            "norm_topk_prob": renorm,
            "moe_intermediate_size": 8,
            "shared_expert_intermediate_size": 8
        }))
        .unwrap()
        .moe
    }

    #[test]
    fn separated_scores_keep_ids_and_f32_weights_are_tolerance_only() {
        let plan = RouterPlan::new(&spec(8, 2, true), 8).unwrap();
        let route = plan
            .route(&[9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0])
            .unwrap();
        assert_eq!(route.ids(), &[0, 1]);
        assert!((route.weights().iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(route.weights()[0] > route.weights()[1]);
        assert_eq!(route.weights().len(), 2);
    }

    #[test]
    fn exact_ties_use_lower_expert_ids() {
        let plan = RouterPlan::new(&spec(8, 2, true), 8).unwrap();
        let route = plan.route(&[1.0; 8]).unwrap();
        assert_eq!(route.ids(), &[0, 1]);
    }

    #[test]
    fn fallback_cast_is_separate_from_accelerated_f32_weights() {
        let plan = RouterPlan::new(&spec(8, 1, true), 8).unwrap();
        let route = plan
            .route(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
            .unwrap();
        assert_eq!(route.ids(), &[7]);
        assert_eq!(route.weights().len(), 1);
        assert_eq!(route.fallback_bf16_weights().len(), 1);
        assert_eq!(
            route.fallback_bf16_weights()[0],
            half::bf16::from_f32(route.weights()[0]).to_bits()
        );
    }

    #[test]
    fn rejects_nonfinite_bad_shapes_and_bad_top_k() {
        let plan = RouterPlan::new(&spec(8, 2, true), 8).unwrap();
        assert!(plan.route(&[f32::NAN; 8]).is_err());
        assert!(plan.route(&[f32::INFINITY; 8]).is_err());
        assert!(plan.route(&[0.0; 7]).is_err());
        assert!(plan.route_rows(&[0.0; 17]).is_err());
        let base = spec(8, 2, true);
        let zero = Qwen35MoeSpec {
            num_experts_per_tok: 0,
            ..base.clone()
        };
        assert!(RouterPlan::new(&zero, 8).is_err());
        let too_many = Qwen35MoeSpec {
            num_experts_per_tok: 9,
            ..base
        };
        assert!(RouterPlan::new(&too_many, 8).is_err());
        assert!(RouterPlan::new(&spec(8, 2, true), 0).is_err());
    }

    #[test]
    fn route_rows_preserves_row_order() {
        let plan = RouterPlan::new(&spec(8, 1, true), 8).unwrap();
        let mut rows = [0.0f32; 16];
        rows[2] = 1.0;
        rows[8] = 1.0;
        let routes = plan.route_rows(&rows).unwrap();
        assert_eq!(routes.len(), 2);
        assert_eq!(routes[0].ids(), &[2]);
        assert_eq!(routes[1].ids(), &[0]);
    }
}
