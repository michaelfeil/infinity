#![cfg(feature = "flash-attn")]

use anyhow::{Context, Result};
use infinity_v3_backend_candle::CandleBackend;
use infinity_v3_backend_core::{Backend, Batch, ModelType, Pool};
use std::path::PathBuf;
use tokenizers::Tokenizer;

#[test]
#[ignore = "requires a local dense Gemma4 E2B classifier checkpoint and CUDA"]
fn gemma4_classifier_matches_reference_with_variable_length_attention() -> Result<()> {
    let root =
        PathBuf::from(std::env::var("GEMMA4_MODEL_ROOT").context("GEMMA4_MODEL_ROOT is not set")?);
    let tokenizer = Tokenizer::from_file(root.join("tokenizer.json"))
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let backend = CandleBackend::new(&root, "bfloat16".into(), ModelType::Classifier, None, 0)?;
    let mut batch = Batch {
        multimodal: vec![],
        input_ids: vec![],
        token_type_ids: vec![],
        position_ids: vec![],
        cumulative_seq_lengths: vec![0],
        max_length: 0,
        pooled_indices: vec![0, 1],
        raw_indices: vec![],
        compact_input_ids: None,
        compact_position_ids: None,
        scatter_unfold: None,
        fold_gather: None,
        tokens: vec![],
        offsets: vec![],
    };
    for input in [
        "Is Paris in France? Answer yes or no.",
        "Is the moon made of cheese? Answer yes or no.",
    ] {
        let encoding = tokenizer
            .encode(input, true)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let length = encoding.len() as u32;
        batch.input_ids.extend_from_slice(encoding.get_ids());
        batch
            .token_type_ids
            .extend_from_slice(encoding.get_type_ids());
        batch.position_ids.extend(0..length);
        batch
            .cumulative_seq_lengths
            .push(batch.input_ids.len() as u32);
        batch.max_length = batch.max_length.max(length);
    }
    let embedding_batch = batch.clone();
    let predictions = backend.predict(batch)?;
    let expected = [[-20.75f32, -20.375f32], [-13.4375f32, -9.8125f32]];
    for (index, expected) in expected.iter().enumerate() {
        let actual = &predictions[&index];
        for (actual, expected) in actual.iter().zip(expected) {
            // BF16 FlashAttention differs by up to 0.5 from the padded reference.
            assert!(
                (actual - expected).abs() <= 0.5,
                "prediction {index}: expected {expected}, got {actual}",
            );
        }
    }
    drop(backend);
    let embedding = CandleBackend::new(
        &root,
        "bfloat16".into(),
        ModelType::Embedding(Pool::LastToken),
        None,
        0,
    )?;
    assert_eq!(embedding.embed(embedding_batch)?.len(), 2);
    Ok(())
}
