// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Root-only paired-reference vector producer for the real Mistral3/Pixtral
//! engine. It refuses to run without `INFINITY_MISTRAL3_RUN_GPU=1`.
//!
//! The token ids, image markers and pixel rows match the frozen b66 Native
//! runner exactly, so a Root controller can pair Native gold against this
//! fresh optimized Oxide candidate byte-for-byte or with an explicit tolerance.

use infinity_v3_backend_core::{Batch, ImagePatches, Pool};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

const SEED: u64 = 0x4d49_5354;

type Images = Vec<(usize, Arc<ImagePatches>)>;
type Case = (&'static str, Batch, Images, bool);

fn env(name: &str) -> Result<String, Box<dyn std::error::Error>> {
    std::env::var(name).map_err(|_| format!("set {name}").into())
}

fn text_ids(tokens: usize) -> Vec<u32> {
    (0..tokens)
        .map(|i| (1 + (SEED + i as u64 * 37) % 1000) as u32)
        .collect()
}

fn batch(ids: Vec<u32>, pooled: bool) -> Batch {
    let tokens = ids.len();
    Batch {
        multimodal: vec![None],
        input_ids: ids,
        token_type_ids: vec![0; tokens],
        position_ids: (0..tokens as u32).collect(),
        cumulative_seq_lengths: vec![0, tokens as u32],
        max_length: tokens as u32,
        pooled_indices: if pooled { vec![0] } else { Vec::new() },
        raw_indices: Vec::new(),
        compact_input_ids: None,
        compact_position_ids: None,
        scatter_unfold: None,
        fold_gather: None,
        tokens: Vec::new(),
        offsets: vec![(0, tokens)],
    }
}

fn image_case(
    height: usize,
    width: usize,
    sequence_tokens: usize,
    token_start: usize,
    patch_size: usize,
    merge: usize,
) -> (Batch, Images) {
    let mut ids = text_ids(sequence_tokens);
    let merge_width = width / merge;
    let merge_tokens = (height / merge) * merge_width;
    for i in 0..merge_tokens {
        let row = i / merge_width;
        let column = i % merge_width;
        ids[token_start + row * (merge_width + 1) + column] = 10;
    }
    let patch_dim = 3 * patch_size * patch_size;
    let elements = height * width * patch_dim;
    let mut pixels = Vec::with_capacity(elements);
    for i in 0..elements {
        let sample = (i as u32).wrapping_mul(2_654_435_761) ^ 0x9e37_79b9;
        pixels.push((sample & 0xffff) as f32 / 65535.0);
    }
    let patches = Arc::new(ImagePatches {
        pixels,
        grid_thw: [1, height, width],
        patch_dim,
        merge_size: merge,
    });
    (batch(ids, true), vec![(token_start, patches)])
}

fn write_f32(path: &Path, values: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path, bytes)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var_os("INFINITY_MISTRAL3_RUN_GPU").as_deref() != Some(std::ffi::OsStr::new("1")) {
        eprintln!("set INFINITY_MISTRAL3_RUN_GPU=1 (Root GPU only)");
        return Ok(());
    }
    let model = env("INFINITY_MISTRAL3_MODEL")?;
    let fa4 = env("INFINITY_FA4_ROOT")?;
    let out = PathBuf::from(env("INFINITY_MISTRAL3_OUT_DIR")?);
    let ordinal: usize = std::env::var("INFINITY_MISTRAL3_DEVICE")
        .unwrap_or_else(|_| "0".to_owned())
        .parse()?;
    let warmup: usize = std::env::var("INFINITY_MISTRAL3_WARMUP")
        .unwrap_or_else(|_| "3".to_owned())
        .parse()?;
    let iterations: usize = std::env::var("INFINITY_MISTRAL3_ITERATIONS")
        .unwrap_or_else(|_| "20".to_owned())
        .parse()?;
    fs::create_dir_all(&out)?;

    let open = Instant::now();
    let mut engine = infinity_backend_oxide::Mistral3Qualification::open(
        Path::new(&model),
        ordinal,
        Path::new(&fa4),
        256,
        4,
        2,
    )?;
    let create_ns = open.elapsed().as_nanos();

    let mut manifest = BTreeMap::new();
    let cases: Vec<Case> = vec![
        ("text-mean-16", batch(text_ids(16), true), Vec::new(), false),
        (
            "text-mean-128",
            batch(text_ids(128), true),
            Vec::new(),
            false,
        ),
        ("text-score-32", batch(text_ids(32), true), Vec::new(), true),
        {
            let (batch, images) = image_case(4, 6, 12, 2, 14, 2);
            ("image-mean-1x4x6", batch, images, false)
        },
        {
            let (batch, images) = image_case(8, 8, 24, 2, 14, 2);
            ("image-score-1x8x8", batch, images, true)
        },
    ];

    let mut first_run = 0u128;
    for (name, batch, images, score) in &cases {
        let start = Instant::now();
        let run = engine.run_batch(batch, &Pool::Mean, images)?;
        let elapsed = start.elapsed().as_nanos();
        if *name == "text-mean-16" {
            first_run = elapsed;
        }
        let values: &[f32] = if *score { &run.logits } else { &run.pooled };
        write_f32(&out.join(format!("{name}.pooled.f32")), values)?;
        write_f32(&out.join(format!("{name}.raw.f32")), &run.raw)?;
        write_f32(&out.join(format!("{name}.logits.f32")), &run.logits)?;
        manifest.insert(
            name.to_string(),
            serde_json::json!({
                "elements": values.len(),
                "pooled_elements": run.pooled.len(),
                "raw_elements": run.raw.len(),
                "logit_elements": run.logits.len(),
                "score_mode": score,
                "elapsed_ns": elapsed,
                "checksum": values.iter().map(|v| *v as f64).sum::<f64>(),
            }),
        );
    }

    // Retained-workspace repeat/perf loop on the actual image-mean case.
    let (perf_batch, perf_images) = image_case(4, 6, 12, 2, 14, 2);
    for _ in 0..warmup {
        engine.run_batch(&perf_batch, &Pool::Mean, &perf_images)?;
    }
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        let _ = engine.run_batch(&perf_batch, &Pool::Mean, &perf_images)?;
        samples.push(start.elapsed().as_nanos() as u64);
    }
    samples.sort_unstable();
    let p50 = samples.get(samples.len() / 2).copied().unwrap_or(0);
    let p99 = samples
        .get((samples.len() * 99 / 100).min(samples.len().saturating_sub(1)))
        .copied()
        .unwrap_or(0);

    let candidate = serde_json::json!({
        "schema": "infinity-oxide-mistral3-candidate/v1",
        "checkpoint": model,
        "device": ordinal,
        "tensor_headers": 531,
        "create_ns": create_ns,
        "first_run_ns": first_run,
        "cases": manifest,
        "perf": {
            "warmup": warmup,
            "iterations": iterations,
            "min_ns": samples.first().copied().unwrap_or(0),
            "max_ns": samples.last().copied().unwrap_or(0),
            "p50_ns": p50,
            "p99_ns": p99,
        },
    });
    fs::write(
        out.join("candidate.json"),
        serde_json::to_vec_pretty(&candidate)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&candidate)?);
    drop(engine);
    Ok(())
}
