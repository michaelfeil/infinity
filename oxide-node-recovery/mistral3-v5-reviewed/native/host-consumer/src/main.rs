// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Unified host consumer for the normal Mistral3 `OxideBackend::new` factory.
//!
//! This binary exists for the one granted opt3/no-LTO independent consumer run.
//! It exercises the actual factory route (not the qualification-only handle):
//! text embedding, image embedding, classifier score and backend lifetime.
//! It is a binary only; it is not part of the optimized all-test-target build.

use infinity_backend_oxide::OxideBackend;
use infinity_v3_backend_core::{
    Backend, Batch, Embedding, ImagePatches, ModelType, MultimodalEncoding, Pool,
};
use serde_json::json;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

const SEED: u64 = 0x4d49_5354;

fn env(name: &str) -> Result<String, Box<dyn std::error::Error>> {
    std::env::var(name).map_err(|_| format!("set {name}").into())
}

fn text_ids(tokens: usize) -> Vec<u32> {
    (0..tokens)
        .map(|i| (1 + (SEED + i as u64 * 37) % 1000) as u32)
        .collect()
}

fn batch(ids: Vec<u32>, pooled: bool, media: Option<Arc<MultimodalEncoding>>) -> Batch {
    let tokens = ids.len();
    Batch {
        multimodal: if media.is_some() { vec![media] } else { vec![] },
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

fn batch_positions(ids: Vec<u32>, positions: Vec<u32>, media: Option<Arc<MultimodalEncoding>>) -> Batch {
    let tokens = ids.len();
    assert_eq!(positions.len(), tokens);
    Batch {
        multimodal: if media.is_some() { vec![media] } else { vec![] },
        input_ids: ids,
        token_type_ids: vec![0; tokens],
        position_ids: positions,
        cumulative_seq_lengths: vec![0, tokens as u32],
        max_length: tokens as u32,
        pooled_indices: vec![0],
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
    position_offset: usize,
) -> (Batch, Arc<ImagePatches>) {
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
    let media = Arc::new(MultimodalEncoding {
        images: vec![(token_start, patches.clone())],
        audios: vec![],
        position_ids: std::array::from_fn(|_| {
            (position_offset..position_offset + sequence_tokens)
                .map(|value| value as u32)
                .collect()
        }),
        memory: None,
    });
    (
        batch_positions(
            ids,
            (position_offset..position_offset + sequence_tokens)
                .map(|value| value as u32)
                .collect(),
            Some(media),
        ),
        patches,
    )
}

fn write_f32(path: &Path, values: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path, bytes)?;
    Ok(())
}

fn pooled(embedding: &Embedding) -> &[f32] {
    match embedding {
        Embedding::Pooled(values) => values,
        Embedding::All(rows) => rows.first().map(Vec::as_slice).unwrap_or(&[]),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var_os("INFINITY_MISTRAL3_RUN_GPU").as_deref() != Some(std::ffi::OsStr::new("1")) {
        eprintln!("set INFINITY_MISTRAL3_RUN_GPU=1 (Root GPU only)");
        return Ok(());
    }
    let model = PathBuf::from(env("INFINITY_MISTRAL3_MODEL")?);
    let out = PathBuf::from(env("INFINITY_OXIDE_MISTRAL3_OUT_DIR")?);
    let ordinal: usize = std::env::var("INFINITY_MISTRAL3_DEVICE")
        .unwrap_or_else(|_| "0".to_owned())
        .parse()?;
    let iterations: usize = std::env::var("INFINITY_MISTRAL3_ITERATIONS")
        .unwrap_or_else(|_| "20".to_owned())
        .parse()?;
    let warmup: usize = std::env::var("INFINITY_MISTRAL3_WARMUP")
        .unwrap_or_else(|_| "3".to_owned())
        .parse()?;
    if !std::env::var("CUDA_VISIBLE_DEVICES").is_ok_and(|value| !value.is_empty()) {
        return Err("CUDA_VISIBLE_DEVICES must be set by Root".into());
    }
    env("FA4_ROOT")?;
    env("INFINITY_OXIDE_KERNEL_MODULE")?;
    fs::create_dir_all(&out)?;

    let raw_config: serde_json::Value = serde_json::from_slice(
        &fs::read(model.join("config.json")).or_else(|_| fs::read(model.join("params.json")))?,
    )?;
    let classifier = raw_config
        .get("architectures")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|values| {
            values.iter().any(|value| {
                value
                    .as_str()
                    .is_some_and(|name| name.ends_with("ForSequenceClassification"))
            })
        });
    let model_type = if classifier {
        ModelType::Classifier
    } else {
        ModelType::Embedding(Pool::Mean)
    };

    let open = Instant::now();
    let backend = OxideBackend::new(&model, "bfloat16", model_type.clone(), ordinal)?;
    let create_ns = open.elapsed().as_nanos();
    backend.health()?;
    let mut manifest = BTreeMap::new();

    let text_cases = [("text-mean-16", 16usize), ("text-mean-128", 128)];
    for (name, tokens) in text_cases {
        let start = Instant::now();
        let result = backend.embed(batch(text_ids(tokens), true, None))?;
        let elapsed = start.elapsed().as_nanos();
        let values = result
            .get(&0)
            .map(pooled)
            .ok_or("missing text embedding")?;
        write_f32(&out.join(format!("{name}.pooled.f32")), values)?;
        manifest.insert(
            name.to_string(),
            json!({"elements": values.len(), "elapsed_ns": elapsed}),
        );
    }

    for (name, positions) in [
        ("text-yarn-16384", vec![16383, 16384, 16385, 16386]),
        ("text-yarn-32768", vec![32767, 32768, 32769, 32770]),
        ("text-yarn-262140", vec![262140, 262141, 262142, 262143]),
    ] {
        let start = Instant::now();
        let result = backend.embed(batch_positions(
            text_ids(positions.len()),
            positions,
            None,
        ))?;
        let elapsed = start.elapsed().as_nanos();
        let values = result.get(&0).map(pooled).ok_or("missing YaRN text embedding")?;
        write_f32(&out.join(format!("{name}.pooled.f32")), values)?;
        manifest.insert(
            name.to_string(),
            json!({"elements": values.len(), "elapsed_ns": elapsed, "positions": "explicit"}),
        );
    }

    if classifier {
        let start = Instant::now();
        let result = backend.predict(batch(text_ids(32), true, None))?;
        let elapsed = start.elapsed().as_nanos();
        let values = result.get(&0).ok_or("missing text logits")?;
        write_f32(&out.join("text-score-32.logits.f32"), values)?;
        manifest.insert(
            "text-score-32".to_string(),
            json!({"elements": values.len(), "elapsed_ns": elapsed}),
        );
    }

    let (image_batch, _) = image_case(4, 6, 12, 2, 14, 2, 0);
    let start = Instant::now();
    let result = backend.embed(image_batch)?;
    let elapsed = start.elapsed().as_nanos();
    let values = result
        .get(&0)
        .map(pooled)
        .ok_or("missing image embedding")?;
    write_f32(&out.join("image-mean-1x4x6.pooled.f32"), values)?;
    manifest.insert(
        "image-mean-1x4x6".to_string(),
        json!({"elements": values.len(), "elapsed_ns": elapsed}),
    );

    // Explicit raw token output through the normal factory.
    let mut raw_batch = batch(text_ids(16), false, None);
    raw_batch.raw_indices = vec![0];
    let start = Instant::now();
    let raw_result = backend.embed(raw_batch)?;
    let elapsed = start.elapsed().as_nanos();
    let raw_rows = match raw_result.get(&0) {
        Some(Embedding::All(rows)) => rows,
        Some(Embedding::Pooled(_)) | None => return Err("missing raw token output".into()),
    };
    let flat: Vec<f32> = raw_rows.iter().flatten().copied().collect();
    write_f32(&out.join("text-raw-16.raw.f32"), &flat)?;
    manifest.insert(
        "text-raw-16".to_string(),
        json!({"rows": raw_rows.len(), "elements": flat.len(), "elapsed_ns": elapsed}),
    );

    // Image + selected classifier on the actual patch path.
    let (image_yarn_batch, _) = image_case(4, 6, 12, 2, 14, 2, 16384);
    let start = Instant::now();
    let result = backend.embed(image_yarn_batch)?;
    let elapsed = start.elapsed().as_nanos();
    let values = result.get(&0).map(pooled).ok_or("missing shifted image embedding")?;
    write_f32(&out.join("image-yarn-16384.pooled.f32"), values)?;
    manifest.insert(
        "image-yarn-16384".to_string(),
        json!({"elements": values.len(), "elapsed_ns": elapsed, "positions": "explicit"}),
    );

    if classifier {
        let (image_batch, _) = image_case(8, 8, 24, 2, 14, 2, 0);
        let start = Instant::now();
        let result = backend.predict(image_batch)?;
        let elapsed = start.elapsed().as_nanos();
        let values = result.get(&0).ok_or("missing image logits")?;
        write_f32(&out.join("image-score-1x8x8.logits.f32"), values)?;
        manifest.insert(
            "image-score-1x8x8".to_string(),
            json!({"elements": values.len(), "elapsed_ns": elapsed}),
        );
    }

    // Deferred lease + health, then a fresh factory open/drop to exercise the
    // real lifetime path without reusing the qualification handle.
    let deferred = backend.submit_embed(batch(text_ids(16), true, None))?.complete()?;
    if deferred.get(&0).is_none() {
        return Err("deferred embedding missing".into());
    }
    backend.health()?;
    let lifetime_start = Instant::now();
    {
        let reopened = OxideBackend::new(&model, "bfloat16", model_type, ordinal)?;
        reopened.health()?;
        let _ = reopened.embed(batch(text_ids(8), true, None))?;
    }
    let lifetime_ns = lifetime_start.elapsed().as_nanos();

    let mut perf_samples = Vec::with_capacity(iterations);
    for _ in 0..warmup {
        let _ = backend.embed(batch(text_ids(16), true, None))?;
    }
    for _ in 0..iterations {
        let start = Instant::now();
        let _ = backend.embed(batch(text_ids(16), true, None))?;
        perf_samples.push(start.elapsed().as_nanos() as u64);
    }
    perf_samples.sort_unstable();
    let p50 = perf_samples.get(perf_samples.len() / 2).copied().unwrap_or(0);
    let p99 = perf_samples
        .get((perf_samples.len() * 99 / 100).min(perf_samples.len().saturating_sub(1)))
        .copied()
        .unwrap_or(0);

    let candidate = json!({
        "schema": "oxide-mistral3-host-consumer/v1",
        "factory": "OxideBackend::new",
        "checkpoint": model,
        "classifier": classifier,
        "device": ordinal,
        "create_ns": create_ns,
        "lifetime_reopen_ns": lifetime_ns,
        "cases": manifest,
        "perf": {"warmup": warmup, "iterations": iterations, "p50_ns": p50, "p99_ns": p99},
    });
    fs::write(out.join("candidate.json"), serde_json::to_vec_pretty(&candidate)?)?;
    println!("{}", serde_json::to_string_pretty(&candidate)?);
    drop(backend);
    Ok(())
}
