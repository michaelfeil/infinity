// SPDX-License-Identifier: LicenseRef-Infinity-Restricted-MIT-1.0
// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Root GPU probe for real trained Dense safetensors versus `.bin` cold loads.
//!
//! The two model directories must describe the same trained model: one with
//! safetensors Dense modules and one with stored PyTorch `.bin` Dense modules.
//! This is a spot probe, not a parity, speed, or FP16-serving qualification.
use infinity_backend_oxide::OxideBackend;
use infinity_v3_backend_core::{Backend, Batch, Embedding, ModelType, Pool};
use std::{error::Error, path::Path};

const TOLERANCE: f32 = 1e-6;

fn make_batch(length: usize) -> Result<Batch, Box<dyn Error>> {
    let input_ids = (0..length)
        .map(|position| {
            position
                .checked_mul(13)
                .map(|n| n % 97)
                .and_then(|n| u32::try_from(n).ok())
                .and_then(|n| n.checked_add(1))
                .ok_or("token id overflow")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let position_ids = (0..length)
        .map(u32::try_from)
        .collect::<Result<Vec<_>, _>>()?;
    let rows = length;
    let length = u32::try_from(length)?;
    Ok(Batch {
        input_ids,
        position_ids,
        token_type_ids: vec![0; rows],
        cumulative_seq_lengths: vec![0, length],
        max_length: length,
        pooled_indices: vec![0],
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

fn run(backend: &OxideBackend, batch: &Batch) -> Result<Vec<f32>, Box<dyn Error>> {
    let embeddings = backend.embed(batch.clone())?;
    match embeddings.get(&0) {
        Some(Embedding::Pooled(row)) => Ok(row.clone()),
        _ => Err("missing pooled embedding".into()),
    }
}

fn max_abs_diff(left: &[f32], right: &[f32]) -> Result<f32, Box<dyn Error>> {
    if left.len() != right.len() {
        return Err("embedding width mismatch".into());
    }
    left.iter().zip(right).try_fold(0.0f32, |worst, (x, y)| {
        if !x.is_finite() || !y.is_finite() {
            return Err("non-finite embedding".into());
        }
        Ok(worst.max((x - y).abs()))
    })
}

fn bits_hash(values: &[f32]) -> u64 {
    values.iter().fold(0xcbf29ce484222325u64, |hash, value| {
        let bits = u64::from(value.to_bits());
        (hash ^ bits).wrapping_mul(0x100000001b3)
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    let safe_path = args
        .get(1)
        .ok_or("usage: dense_safe_bin_probe SAFETENSORS_MODEL BIN_MODEL [DEVICE_ID] [TOKENS]")?;
    let bin_path = args
        .get(2)
        .ok_or("usage: dense_safe_bin_probe SAFETENSORS_MODEL BIN_MODEL [DEVICE_ID] [TOKENS]")?;
    let device_id = args
        .get(3)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(0);
    let tokens = args
        .get(4)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(128);
    let safe = OxideBackend::new(
        Path::new(safe_path),
        "bfloat16",
        ModelType::Embedding(Pool::Mean),
        device_id,
    )?;
    let bin = OxideBackend::new(
        Path::new(bin_path),
        "bfloat16",
        ModelType::Embedding(Pool::Mean),
        device_id,
    )?;
    let batch = make_batch(tokens)?;
    let safe_first = run(&safe, &batch)?;
    let safe_repeat = run(&safe, &batch)?;
    let bin_first = run(&bin, &batch)?;
    let safe_restored = run(&safe, &batch)?;
    let safe_bin = max_abs_diff(&safe_first, &bin_first)?;
    let safe_repeat = max_abs_diff(&safe_first, &safe_repeat)?;
    let safe_restored = max_abs_diff(&safe_first, &safe_restored)?;
    if safe_bin > TOLERANCE || safe_repeat > TOLERANCE || safe_restored > TOLERANCE {
        return Err(format!(
            "probe tolerance exceeded: safe_bin={safe_bin} safe_repeat={safe_repeat} safe_restored={safe_restored}"
        )
        .into());
    }
    safe.health()?;
    bin.health()?;
    println!(
        "{}",
        serde_json::json!({
            "status": "ok",
            "probe": "dense_safe_bin",
            "tokens": tokens,
            "device": device_id,
            "tolerance": TOLERANCE,
            "safe_bin_max_abs": safe_bin,
            "safe_repeat_max_abs": safe_repeat,
            "safe_restored_max_abs": safe_restored,
            "safe_bits_hash": bits_hash(&safe_first),
            "bin_bits_hash": bits_hash(&bin_first)
        })
    );
    Ok(())
}
