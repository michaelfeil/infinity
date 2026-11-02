// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Completed-result split and Native Clef action probability.

use crate::{clef_decision::infer, clef_decision_plan::ClefRequestPlan};
use infinity_v3_backend_core::{BackendError, DecisionOutput};

/// Write Native action probability `1.0` for every request and split the
/// flattened option logits back into request outputs. Called only after the
/// engine/lease proves completion; performs no GPU work.
pub(crate) fn finish_outputs(
    plan: &ClefRequestPlan,
    host: &mut [f32],
) -> Result<Vec<DecisionOutput>, BackendError> {
    if host.len() < plan.options + plan.requests {
        return Err(infer(
            "Clef result buffer is shorter than logits plus actions",
        ));
    }
    for index in 0..plan.requests {
        host[plan.options + index] = 1.0;
    }
    let mut outputs = Vec::new();
    outputs
        .try_reserve_exact(plan.requests)
        .map_err(|error| infer(format!("Clef result allocation failed: {error}")))?;
    let mut offset = 0usize;
    for &count in &plan.counts {
        let end = offset
            .checked_add(count)
            .ok_or_else(|| infer("Clef result split overflow"))?;
        if end > plan.options {
            return Err(infer("Clef result split exceeds option logits"));
        }
        outputs.push(DecisionOutput {
            logits: host[offset..end].to_vec(),
            action_probability: 1.0,
            memory: None,
        });
        offset = end;
    }
    Ok(outputs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{clef_decision_plan::plan_requests, decision_head::PackedDecisionBatch};
    use infinity_v3_backend_core::{ClefField, DecisionInput};

    #[test]
    fn finish_splits_field_order_and_sets_action_probability_one() {
        let batch = PackedDecisionBatch::new(&[0, 3, 5]).unwrap();
        let ids = [3, 4, 5, 6, 7];
        let requests = [
            DecisionInput::Clef {
                fields: vec![ClefField {
                    kind: 0,
                    question: (0, 1),
                    options: vec![(1, 2), (2, 3)],
                }],
            },
            DecisionInput::Warmup,
        ];
        let plan = plan_requests(&batch, &ids, &requests).unwrap();
        let mut host = [0.0f32; 5];
        host[0] = 1.0;
        host[1] = 2.0;
        host[2] = 3.0;
        let outputs = finish_outputs(&plan, &mut host).unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].logits, vec![1.0, 2.0]);
        assert_eq!(outputs[1].logits, vec![3.0]);
        assert_eq!(outputs[0].action_probability, 1.0);
        assert_eq!(outputs[1].action_probability, 1.0);
        assert_eq!(&host[3..], &[1.0, 1.0]);
    }
}
