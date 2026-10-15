//! Independent full-sequence Transformers references versus cached block execution.
use infinity_backend_libtorch::LibtorchBackend;
use serde::Deserialize;
use std::path::Path;
use text_embeddings_backend_core::{Backend, Batch, DecisionInput, ModelType};

#[derive(Deserialize)]
struct Case {
    input_ids: Vec<u32>,
    prefix_length: usize,
    token_ids: Vec<u32>,
    logits: Vec<f32>,
    #[serde(default)]
    cached_logits: Option<Vec<f32>>,
}
#[derive(Deserialize)]
struct Reference {
    model: String,
    cases: Vec<Case>,
}
fn batch(cases: &[&Case]) -> Batch {
    let mut batch = Batch {
        input_ids: vec![],
        token_type_ids: vec![],
        position_ids: vec![],
        cumulative_seq_lengths: vec![0],
        max_length: 0,
        pooled_indices: vec![],
        raw_indices: vec![],
        multimodal: vec![],
        compact_input_ids: None,
        compact_position_ids: None,
        scatter_unfold: None,
        fold_gather: None,
        tokens: vec![],
        offsets: vec![],
    };
    for case in cases {
        batch.input_ids.extend(&case.input_ids);
        batch.token_type_ids.extend(vec![0; case.input_ids.len()]);
        batch.position_ids.extend(0..case.input_ids.len() as u32);
        batch
            .cumulative_seq_lengths
            .push(batch.input_ids.len() as u32);
        batch.max_length = batch.max_length.max(case.input_ids.len() as u32);
    }
    batch
}
fn metadata(cases: &[&Case]) -> Vec<DecisionInput> {
    cases
        .iter()
        .map(|c| DecisionInput::BlockOptionTokens {
            token_ids: c.token_ids.clone(),
            prefix_length: c.prefix_length,
        })
        .collect()
}
#[test]
#[ignore = "Generate scripts/onejev-block-reference.py and set ONEJEV_BLOCK_REFERENCE"]
fn block_decisions_match_full_transformers() -> Result<(), Box<dyn std::error::Error>> {
    let reference: Reference =
        serde_json::from_slice(&std::fs::read(std::env::var("ONEJEV_BLOCK_REFERENCE")?)?)?;
    let dtype = std::env::var("ONEJEV_DTYPE").unwrap_or("float32".into());
    let device = std::env::var("ONEJEV_DEVICE").unwrap_or("cpu".into());
    let backend = LibtorchBackend::new(
        Path::new(&reference.model),
        &dtype,
        ModelType::Decision,
        &device,
    )?;
    assert!(!backend.supports_radix_mlp());
    let compare = |actual: &[f32], expected: &[f32]| {
        assert_eq!(actual.len(), expected.len());
        // BF16 full-sequence and split-prefix GEMMs/recurrent reductions have
        // different rounding. Bound raw logits and probabilities independently.
        let tolerance = if dtype == "float32" { 0.001 } else { 0.1 };
        for (a, b) in actual.iter().zip(expected) {
            assert!(
                a.is_finite() && (a - b).abs() <= tolerance,
                "{dtype}: actual={a}, reference={b}, delta={}",
                (a - b).abs()
            );
        }
        let softmax = |values: &[f32]| {
            let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let values: Vec<_> = values.iter().map(|v| (v - maximum).exp()).collect();
            let sum: f32 = values.iter().sum();
            values.into_iter().map(|v| v / sum).collect::<Vec<_>>()
        };
        for (a, b) in softmax(actual).iter().zip(softmax(expected)) {
            assert!(
                (a - b).abs() < if dtype == "float32" { 0.001 } else { 0.02 },
                "Probability drift: {a} vs {b}"
            );
        }
    };
    let mut cases: Vec<_> = reference.cases.iter().collect();
    assert!(cases.len() >= 4);
    for _ in 0..2 {
        let results = backend.decide(batch(&cases), metadata(&cases))?;
        for (result, case) in results.iter().zip(&cases) {
            eprintln!(
                "full comparison actual={:?} reference={:?}",
                result.logits, case.logits
            );
        }
        for (result, case) in results.iter().zip(&cases) {
            compare(&result.logits, &case.logits);
            if let Some(cached) = &case.cached_logits {
                compare(&result.logits, cached);
            }
            let single = backend.decide(batch(&[case]), metadata(&[case]))?;
            assert_eq!(
                result.logits, single[0].logits,
                "Whole-prefix sharing changed native outputs"
            );
            eprintln!(
                "tokens={} prefix={} logits={:?}",
                case.input_ids.len(),
                case.prefix_length,
                result.logits
            );
        }
        cases.reverse();
    }
    let first = cases[0];
    for prefix_length in [0, first.input_ids.len() + 1] {
        assert!(backend
            .decide(
                batch(&[first]),
                vec![DecisionInput::BlockOptionTokens {
                    token_ids: first.token_ids.clone(),
                    prefix_length
                }]
            )
            .is_err());
    }
    assert!(backend
        .decide(
            batch(&[first]),
            vec![DecisionInput::OptionTokens {
                token_ids: first.token_ids.clone()
            }]
        )
        .is_err());
    assert!(backend
        .decide(batch(&[first]), vec![DecisionInput::Warmup])
        .is_ok());
    Ok(())
}
