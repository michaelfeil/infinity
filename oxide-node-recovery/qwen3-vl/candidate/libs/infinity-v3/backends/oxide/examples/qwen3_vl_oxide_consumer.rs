// Optional Qwen3-VL Oxide consumer for Root GPU gates.
//
// Reads the same real processor fixtures as the Native b66 consumer, runs the
// concrete Oxide tower + dense decoder, and writes f32 outputs plus SHA-256 so a
// fresh coupled Native-vs-Oxide comparison can be made without reshaping data.
//
// This binary needs a CUDA device at runtime. The author builds it on CPU only.

use infinity_backend_oxide::qwen3_vl_fixtures::{fixture_batch, pool_from_name, read_fixture_file};
use infinity_backend_oxide::qwen3_vl_service::Qwen3VlModel;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn usage() -> ! {
    eprintln!(
        "usage: qwen3_vl_oxide_consumer CHECKPOINT FIXTURES OUTPUT_DIR \
         [--fixture NAME] [--raw] [--pool cls|mean|last_token] [--iterations N] [--device N]"
    );
    std::process::exit(2)
}

fn pool_label(pool: &infinity_v3_backend_core::Pool) -> &'static str {
    use infinity_v3_backend_core::Pool;
    match pool {
        Pool::Cls => "cls",
        Pool::Mean => "mean",
        Pool::LastToken => "last",
        Pool::Splade => "splade",
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let checkpoint = PathBuf::from(args.next().unwrap_or_else(|| usage()));
    let fixtures = PathBuf::from(args.next().unwrap_or_else(|| usage()));
    let output = PathBuf::from(args.next().unwrap_or_else(|| usage()));
    let mut selected: Option<String> = None;
    let mut raw = false;
    let mut iterations = 1usize;
    let mut device = 0usize;
    let mut pool_override: Option<String> = None;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--fixture" => selected = Some(args.next().unwrap_or_else(|| usage())),
            "--raw" => raw = true,
            "--pool" => pool_override = Some(args.next().unwrap_or_else(|| usage())),
            "--iterations" => {
                iterations = args
                    .next()
                    .unwrap_or_else(|| usage())
                    .parse()
                    .map_err(|error| format!("invalid iterations: {error}"))?
            }
            "--device" => {
                device = args
                    .next()
                    .unwrap_or_else(|| usage())
                    .parse()
                    .map_err(|error| format!("invalid device: {error}"))?
            }
            other => return Err(format!("unsupported argument {other}").into()),
        }
    }
    let manifest = read_fixture_file(&fixtures)?;
    let fixture_root = fixtures.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(&output)?;
    let context = std::sync::Arc::new(cuda_core::Context::new(device)?);
    let stream = context.new_stream()?;
    let mut model = Qwen3VlModel::open(&checkpoint, &context, &stream)?;
    let mut records = Vec::new();
    for fixture in &manifest.fixtures {
        if selected.as_deref().is_some_and(|name| name != fixture.name) {
            continue;
        }
        let use_raw = raw || model.supports_multi_vector();
        let batch = fixture_batch(fixture, fixture_root, use_raw)?;
        let pool = pool_from_name(pool_override.as_deref().or(fixture.pool.as_deref()))?;
        let mut timings = Vec::new();
        let mut last = None;
        for _ in 0..iterations.max(1) {
            let started = Instant::now();
            let result = model.forward_batch(&batch, pool.clone())?;
            timings.push(started.elapsed().as_secs_f64() * 1e3);
            last = Some(result);
        }
        let result = last.ok_or("no iteration ran")?;
        let values = if let Some(tokens) = result.tokens.as_ref() {
            tokens
        } else {
            result.pooled.as_ref().ok_or("fixture produced no output")?
        };
        let kind = if use_raw {
            "tokens".to_owned()
        } else {
            format!("pool{}", pool_label(&pool))
        };
        let path = output.join(format!("{}.{kind}.f32", fixture.name));
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(&path, &bytes)?;
        let sha = format!("{:x}", Sha256::digest(&bytes));
        records.push(json!({
            "fixture": fixture.name,
            "kind": kind,
            "width": result.width,
            "sequences": result.sequences,
            "values": values.len(),
            "cumulative": result.cumulative,
            "sha256": sha,
            "path": path,
            "iterations_ms": timings,
            "finite": values.iter().all(|value| value.is_finite()),
        }));
    }
    let report = json!({
        "schema": "infinity-qwen3-vl-oxide-consumer/v1",
        "checkpoint": checkpoint,
        "fixtures": fixtures,
        "device": device,
        "records": records,
    });
    std::fs::write(
        output.join("oxide-consumer.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(())
}
