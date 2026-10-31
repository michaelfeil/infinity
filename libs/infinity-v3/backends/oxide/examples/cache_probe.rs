// SPDX-License-Identifier: LicenseRef-Infinity-Restricted-MIT-1.0
// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Finite native workspace-cache and CUDA-graph replay validator.
use infinity_backend_oxide::OxideBackend;
use infinity_v3_backend_core::{Backend, Batch, Embedding, ModelType, Pool};
use std::{error::Error, path::Path};

const TOLERANCE: f32 = 1e-6;
type Rows = Vec<Vec<f32>>;

fn make_batch(lengths: &[usize], variant: usize) -> Result<Batch, Box<dyn Error>> {
    let (mut input_ids, mut position_ids, mut token_type_ids) =
        (Vec::new(), Vec::new(), Vec::new());
    let mut cumulative_seq_lengths = vec![0u32];
    let (mut total, mut max_length) = (0u32, 0usize);
    for &length in lengths {
        max_length = max_length.max(length);
        let end = u32::try_from(length)?
            .checked_add(total)
            .ok_or("token count overflow")?;
        cumulative_seq_lengths.push(end);
        total = end;
        for position in 0..length {
            let token = position
                .checked_mul(13)
                .and_then(|n| n.checked_add(variant.checked_mul(777)?))
                .map(|n| n % 20000)
                .and_then(|n| u32::try_from(n).ok())
                .and_then(|n| n.checked_add(20))
                .ok_or("token id overflow")?;
            input_ids.push(token);
            position_ids.push(u32::try_from(position)?);
            token_type_ids.push(0);
        }
    }
    let sequences = u32::try_from(lengths.len())?;
    Ok(Batch {
        input_ids,
        position_ids,
        token_type_ids,
        cumulative_seq_lengths,
        max_length: u32::try_from(max_length)?,
        pooled_indices: (0..sequences).collect(),
        raw_indices: vec![],
        multimodal: vec![],
        tokens: vec![],
        offsets: vec![],
        compact_input_ids: None,
        compact_position_ids: None,
        scatter_unfold: None,
        fold_gather: None,
    })
}

fn run(backend: &OxideBackend, batch: &Batch) -> Result<Rows, Box<dyn Error>> {
    let embeddings = backend.embed(batch.clone())?;
    let mut rows = Vec::new();
    for &index in &batch.pooled_indices {
        match embeddings.get(&usize::try_from(index)?) {
            Some(Embedding::Pooled(row)) => rows.push(row.clone()),
            _ => return Err("missing pooled embedding".into()),
        }
    }
    Ok(rows)
}

fn max_abs_diff(left: &[Vec<f32>], right: &[Vec<f32>]) -> Result<f32, Box<dyn Error>> {
    if left.len() != right.len() {
        return Err("embedding count mismatch".into());
    }
    left.iter().zip(right).try_fold(0.0f32, |worst, (a, b)| {
        if a.len() != b.len() {
            return Err("embedding width mismatch".into());
        }
        a.iter().zip(b).try_fold(worst, |worst, (x, y)| {
            if !x.is_finite() || !y.is_finite() {
                return Err("non-finite embedding".into());
            }
            Ok(worst.max((x - y).abs()))
        })
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    let path = args
        .get(1)
        .ok_or("usage: cache_probe MODEL_PATH [DEVICE_ID]")?;
    let device_id = match args.get(2) {
        Some(value) => value.parse::<usize>()?,
        None => 0,
    };
    // OxideBackend::new inherits INFINITY_OXIDE_ATTENTION and graph settings.
    let backend = OxideBackend::new(
        Path::new(path),
        "bfloat16",
        ModelType::Embedding(Pool::Mean),
        device_id,
    )?;
    let mut cases: Vec<(String, Vec<usize>)> = Vec::new();
    for length in [
        1usize, 7, 8, 16, 31, 32, 33, 63, 64, 65, 96, 127, 128, 129, 192, 224, 255, 256,
    ] {
        cases.push((format!("single_{length}"), vec![length]));
    }
    for (name, lengths) in [
        ("batch8_16", vec![16usize; 8]),
        ("batch8_64", vec![64; 8]),
        ("batch8_128", vec![128; 8]),
        ("ragged_8", vec![7, 31, 64, 127, 128, 255, 256, 256]),
    ] {
        cases.push((name.into(), lengths));
    }
    let mut baselines: Vec<Rows> = Vec::new();
    let mut replay_checks = 0usize;
    for (name, lengths) in &cases {
        let base = make_batch(lengths, 0)?;
        let changed_batch = make_batch(lengths, 1)?;
        let baseline = run(&backend, &base)?;
        max_abs_diff(&baseline, &baseline)?;
        for _ in 0..3 {
            let changed = run(&backend, &changed_batch)?;
            if max_abs_diff(&baseline, &changed)? <= TOLERANCE {
                return Err(format!("{name} changed output matched baseline").into());
            }
            let restored = run(&backend, &base)?;
            if max_abs_diff(&baseline, &restored)? > TOLERANCE {
                return Err(format!("{name} graph replay mismatch").into());
            }
            replay_checks = replay_checks.checked_add(2).ok_or("check count overflow")?;
        }
        baselines.push(baseline);
    }
    for ((name, lengths), expected) in cases.iter().zip(baselines.iter()).rev() {
        let actual = run(&backend, &make_batch(lengths, 0)?)?;
        if max_abs_diff(expected, &actual)? > TOLERANCE {
            return Err(format!("{name} evicted-cache mismatch").into());
        }
        replay_checks = replay_checks.checked_add(1).ok_or("check count overflow")?;
    }
    backend.health()?;
    let shapes = cases.iter().map(|(name, _)| name).collect::<Vec<_>>();
    println!(
        "{}",
        serde_json::json!({"status":"ok","shapes":shapes,"replay_checks":replay_checks})
    );
    Ok(())
}
