#![cfg(not(feature = "cuda"))]
use anyhow::Result;
use infinity_v3_backend_candle::CandleBackend;
use infinity_v3_backend_core::{Backend, Batch, Embedding, ModelType, Pool};
use std::{path::PathBuf, time::Instant};

#[test]
#[ignore = "requires CPU_RAGGED_MODEL_DIR checkpoint; writes a parity/benchmark artifact"]
fn cpu_ragged_checkpoint() -> Result<()> {
    let root = PathBuf::from(std::env::var("CPU_RAGGED_MODEL_DIR")?);
    let pool = match std::env::var("CPU_RAGGED_POOL").as_deref() {
        Ok("cls") => Pool::Cls,
        Ok("last") => Pool::LastToken,
        _ => Pool::Mean,
    };
    let dtype = std::env::var("CPU_RAGGED_DTYPE").unwrap_or_else(|_| "float32".into());
    let tolerance = if dtype == "float16" { 0.02 } else { 1e-4 };
    let backend = CandleBackend::new(&root, dtype, ModelType::Embedding(pool.clone()), None, 0)?;
    let lengths: Vec<usize> =
        serde_json::from_str(&std::env::var("CPU_RAGGED_LENGTHS").unwrap_or("[7,31,3,17]".into()))?;
    let cumulative: Vec<u32> = std::iter::once(0)
        .chain(lengths.iter().scan(0, |s, &n| {
            *s += n as u32;
            Some(*s)
        }))
        .collect();
    let count = *cumulative.last().unwrap() as usize;
    let position_offset = std::env::var("CPU_RAGGED_POSITION_OFFSET")
        .unwrap_or("0".into())
        .parse::<u32>()?;
    let batch = Batch {
        multimodal: vec![],
        input_ids: (0..count).map(|i| 10 + (i % 37) as u32).collect(),
        token_type_ids: vec![0; count],
        position_ids: lengths
            .iter()
            .flat_map(|&n| position_offset..position_offset + n as u32)
            .collect(),
        cumulative_seq_lengths: cumulative,
        max_length: *lengths.iter().max().unwrap() as u32,
        pooled_indices: (0..lengths.len() as u32).collect(),
        raw_indices: vec![],
        compact_input_ids: None,
        compact_position_ids: None,
        scatter_unfold: None,
        fold_gather: None,
        tokens: vec![],
        offsets: vec![],
    };
    let output = backend.embed(batch.clone())?;
    let pooled: Vec<_> = (0..lengths.len())
        .map(|i| match &output[&i] {
            Embedding::Pooled(x) => x.clone(),
            _ => panic!("expected pooled"),
        })
        .collect();
    let mut raw_batch = batch.clone();
    raw_batch.pooled_indices = vec![];
    raw_batch.raw_indices = (0..lengths.len() as u32).collect();
    let output = backend.embed(raw_batch.clone())?;
    let raw: Vec<_> = (0..lengths.len())
        .map(|i| match &output[&i] {
            Embedding::All(x) => x.clone(),
            _ => panic!("expected raw"),
        })
        .collect();
    {
        for (i, tokens) in raw.iter().enumerate() {
            assert_eq!(tokens.len(), lengths[i], "raw sequence order changed");
            for dim in 0..pooled[i].len() {
                let mean = match pool {
                    Pool::Cls => tokens[0][dim],
                    Pool::LastToken => tokens.last().unwrap()[dim],
                    _ => tokens.iter().map(|token| token[dim]).sum::<f32>() / tokens.len() as f32,
                };
                assert!(
                    (mean - pooled[i][dim]).abs() < tolerance * (1.0 + mean.abs()),
                    "raw/pooled mismatch at sequence {i}, dimension {dim}"
                );
            }
        }
        raw_batch.raw_indices.reverse();
        let reordered = backend.embed(raw_batch)?;
        for (i, expected) in raw.iter().enumerate() {
            match &reordered[&i] {
                Embedding::All(actual) => {
                    assert_eq!(actual, expected, "raw selection changed sequence {i}")
                }
                _ => panic!("expected raw"),
            }
        }
    }
    if lengths.len() > 1 {
        let mut mixed = batch.clone();
        mixed.pooled_indices = vec![0];
        mixed.raw_indices = vec![(lengths.len() - 1) as u32];
        let output = backend.embed(mixed)?;
        match &output[&0] {
            Embedding::Pooled(actual) => assert_eq!(actual, &pooled[0]),
            _ => panic!("expected pooled"),
        }
        match &output[&(lengths.len() - 1)] {
            Embedding::All(actual) => assert_eq!(actual, raw.last().unwrap()),
            _ => panic!("expected raw"),
        }
    }
    for _ in 0..2 {
        backend.embed(batch.clone())?;
    }
    let start = Instant::now();
    for _ in 0..5 {
        backend.embed(batch.clone())?;
    }
    let report = serde_json::json!({"pooled":pooled,"raw":raw,"latency_ms":start.elapsed().as_secs_f64()*200.});
    std::fs::write(
        std::env::var("CPU_RAGGED_RESULT")?,
        serde_json::to_vec(&report)?,
    )?;
    Ok(())
}

#[test]
#[ignore = "requires CPU_RAGGED_REVIEW_FIXTURES with classifier and unsupported Llama checkpoints"]
fn cpu_ragged_dispatch_regressions() -> Result<()> {
    let root = PathBuf::from(std::env::var("CPU_RAGGED_REVIEW_FIXTURES")?);
    for fixture in ["llama-attention-bias", "llama-mlp-bias", "llama-head-dim"] {
        let error = match CandleBackend::new(
            &root.join(fixture),
            "float32".into(),
            ModelType::Embedding(Pool::Mean),
            None,
            0,
        ) {
            Ok(_) => panic!("unsupported Llama settings were silently accepted: {fixture}"),
            Err(error) => error,
        };
        assert!(
            error
                .to_string()
                .contains("Packed Llama requires bias-free projections"),
            "unexpected error for {fixture}: {error}"
        );
    }
    Ok(())
}
