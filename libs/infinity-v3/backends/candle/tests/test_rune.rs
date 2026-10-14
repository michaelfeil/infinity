#![cfg(feature = "flash-attn")]

use anyhow::{Context, Result};
use infinity_v3_backend_candle::CandleBackend;
use infinity_v3_backend_core::{Backend, Batch, DecisionInput, ModelType};
use serde_json::Value;
use std::path::PathBuf;

fn batch(rows: &[Vec<u32>]) -> Batch {
    let mut batch = Batch {
        multimodal: vec![],
        input_ids: vec![],
        token_type_ids: vec![],
        position_ids: vec![],
        cumulative_seq_lengths: vec![0],
        max_length: 0,
        pooled_indices: vec![],
        raw_indices: vec![],
        compact_input_ids: None,
        compact_position_ids: None,
        fold_gather: None,
        scatter_unfold: None,
        tokens: vec![],
        offsets: vec![],
    };
    for row in rows {
        batch.input_ids.extend(row);
        batch.token_type_ids.extend(vec![0; row.len()]);
        batch.position_ids.extend(0..row.len() as u32);
        batch
            .cumulative_seq_lengths
            .push(batch.input_ids.len() as u32);
        batch.max_length = batch.max_length.max(row.len() as u32);
    }
    batch
}

#[test]
#[ignore = "requires Rune checkpoint, token fixture and CUDA"]
fn rune_radix_preserves_selected_logits() -> Result<()> {
    let root = PathBuf::from(std::env::var("RUNE_CHECKPOINT_DIR")?);
    let fixture: Value = serde_json::from_slice(&std::fs::read(std::env::var("RUNE_FIXTURE")?)?)?;
    let rows: Vec<Vec<u32>> = fixture["sequences"]
        .as_array()
        .context("sequences")?
        .iter()
        .map(|s| serde_json::from_value(s["ids"].clone()))
        .collect::<Result<_, _>>()?;
    let inputs: Vec<DecisionInput> = fixture["sequences"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| {
            Ok(DecisionInput::OptionTokens {
                token_ids: serde_json::from_value(s["option_token_ids"].clone())?,
            })
        })
        .collect::<Result<_>>()?;
    let backend = CandleBackend::new(&root, "bfloat16".into(), ModelType::Decision, None, 0)?;
    assert!(backend.supports_radix_mlp());
    let plain = backend.decide(batch(&rows), inputs.clone())?;
    let prefix = (0..rows.iter().map(Vec::len).min().unwrap())
        .take_while(|&i| rows.iter().all(|r| r[i] == rows[0][i]))
        .count();
    assert!(prefix > 0);
    let mut folded = batch(&rows);
    let mut gather = Vec::new();
    let mut scatter = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        let start = folded.cumulative_seq_lengths[i];
        for position in 0..row.len() {
            if i > 0 && position < prefix {
                scatter.push(position as u32);
            } else {
                scatter.push(gather.len() as u32);
                gather.push(start + position as u32);
            }
        }
    }
    folded.compact_input_ids = Some(
        gather
            .iter()
            .map(|&i| folded.input_ids[i as usize])
            .collect(),
    );
    folded.compact_position_ids = Some(
        gather
            .iter()
            .map(|&i| folded.position_ids[i as usize])
            .collect(),
    );
    folded.fold_gather = Some(gather);
    folded.scatter_unfold = Some(scatter);
    let compact = backend.decide(folded, inputs.clone())?;
    for (i, (folded, plain)) in compact.iter().zip(&plain).enumerate() {
        assert_eq!(folded.logits, plain.logits, "Radix row {i}");
    }
    Ok(())
}
