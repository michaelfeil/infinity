// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! CPU metadata/span/Warmup tests for the Clef request plan.

use super::*;
use infinity_v3_backend_core::ClefField;

fn request() -> DecisionInput {
    DecisionInput::Clef {
        fields: vec![
            ClefField {
                kind: 0,
                question: (0, 1),
                options: vec![(1, 2), (2, 3)],
            },
            ClefField {
                kind: 2,
                question: (1, 2),
                options: vec![(0, 2)],
            },
        ],
    }
}

#[test]
fn plans_field_option_order_and_warmup_synthetic_field() {
    let batch = PackedDecisionBatch::new(&[0, 3, 5]).unwrap();
    let ids = [3, 4, 5, 6, 7];
    let plan = plan_requests(&batch, &ids, &[request(), DecisionInput::Warmup]).unwrap();
    assert_eq!(plan.tokens, 5);
    assert_eq!(plan.requests, 2);
    assert_eq!(plan.fields, 3);
    assert_eq!(plan.options, 4);
    assert_eq!(plan.option_tokens, 1 + 1 + 2 + 1);
    assert_eq!(plan.counts, vec![3, 1]);
    assert_eq!(plan.output_elements().unwrap(), 6);
    let l = plan.layout;
    let q = &plan.metadata[l.question_boundaries..l.question_boundaries + 6];
    assert_eq!(q, &[0, 1, 1, 2, 3, 4]);
    let o = &plan.metadata[l.option_boundaries..l.option_boundaries + 8];
    assert_eq!(o, &[1, 2, 2, 3, 0, 2, 3, 4]);
    assert_eq!(
        &plan.metadata[l.field_request..l.field_request + 3],
        &[0, 0, 1]
    );
    assert_eq!(&plan.metadata[l.field_kind..l.field_kind + 3], &[0, 2, 0]);
    assert_eq!(
        &plan.metadata[l.request_option_offsets..l.request_option_offsets + 3],
        &[0, 3, 4]
    );
    assert_eq!(
        &plan.metadata[l.lexical_boundaries..l.lexical_boundaries + 5],
        &[0, 1, 2, 4, 5]
    );
    assert_eq!(
        &plan.metadata[l.option_token_ids..l.option_token_ids + 5],
        &[4, 5, 3, 4, 6]
    );
}

#[test]
fn rejects_invalid_spans_kinds_and_counts() {
    let batch = PackedDecisionBatch::new(&[0, 3, 5]).unwrap();
    let ids = [3, 4, 5, 6, 7];
    for request in [
        DecisionInput::Clef { fields: vec![] },
        DecisionInput::Clef {
            fields: vec![ClefField {
                kind: 3,
                question: (0, 1),
                options: vec![(1, 2)],
            }],
        },
        DecisionInput::Clef {
            fields: vec![ClefField {
                kind: 0,
                question: (0, 4),
                options: vec![(1, 2)],
            }],
        },
        DecisionInput::Clef {
            fields: vec![ClefField {
                kind: 0,
                question: (0, 1),
                options: vec![(1, 4)],
            }],
        },
        DecisionInput::Clef {
            fields: vec![ClefField {
                kind: 0,
                question: (0, 1),
                options: vec![],
            }],
        },
        DecisionInput::OptionTokens { token_ids: vec![1] },
    ] {
        assert!(plan_requests(&batch, &ids, &[request, DecisionInput::Warmup]).is_err());
    }
    assert!(plan_requests(&batch, &ids[..4], &[request()]).is_err());
    assert!(plan_requests(&batch, &ids, &[request()]).is_err());
}
