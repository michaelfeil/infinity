// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Direct backend probe; HTTP parity/performance remains the release gate.
use infinity_backend_oxide::OxideBackend;
use infinity_v3_backend_core::{Backend, Batch, Embedding, ModelType, Pool};
use std::{path::Path, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    let Some(path) = args.get(1) else {
        return Err("usage: probe MODEL_PATH [TOKENS] [ITERATIONS]".into());
    };
    let len = args
        .get(2)
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(128);
    let iterations = args
        .get(3)
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(100);
    let backend = OxideBackend::new(
        Path::new(path),
        "bfloat16",
        ModelType::Embedding(Pool::Mean),
        args.get(4).map(|v| v.parse()).transpose()?.unwrap_or(1),
    )?;
    let ids = (0..len).map(|i| 20 + (i * 13 % 20000) as u32).collect();
    let batch = Batch {
        input_ids: ids,
        position_ids: (0..len).map(|i| i as u32).collect(),
        token_type_ids: vec![0; len],
        cumulative_seq_lengths: vec![0, len.try_into()?],
        max_length: len.try_into()?,
        pooled_indices: vec![0],
        raw_indices: vec![],
        multimodal: vec![],
        tokens: vec![],
        offsets: vec![],
        compact_input_ids: None,
        compact_position_ids: None,
        scatter_unfold: None,
        fold_gather: None,
    };
    let mut last = backend.embed(batch.clone())?;
    for _ in 0..20 {
        last = backend.embed(batch.clone())?;
    }
    let mut samples = Vec::new();
    for _ in 0..iterations {
        let start = Instant::now();
        last = backend.embed(batch.clone())?;
        samples.push(start.elapsed().as_secs_f64() * 1000.);
    }
    let vector = match last.get(&0) {
        Some(Embedding::Pooled(v)) => v,
        _ => return Err("Missing pooled output".into()),
    };
    println!(
        "{}",
        serde_json::json!({"tokens":len,"samples_ms":samples,"embedding":vector})
    );
    Ok(())
}
