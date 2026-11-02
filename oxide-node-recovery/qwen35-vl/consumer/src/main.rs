use infinity_backend_oxide::{
    plan_image, plan_video, prepare_text_fusion, vision_forward, HostImage, MediaGrid, MediaKind,
    MediaSpan, OxideBackend, Qwen35VlConfig, Qwen35VlEngine, Qwen35VlOptions, Qwen35VlRequest,
    Qwen35VlTower, VisionHostWeights,
};
use infinity_v3_backend_core::{
    Backend, BackendError, Batch, DecisionInput, Embedding, Embeddings, ImagePatches, Inference,
    ModelType, MultimodalEncoding, Pool,
};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

fn usage() -> ! {
    eprintln!(
        "usage: qwen35-vl-consumer <inventory|plan|plan-full|tower-cpu|tower-device|whole-factory|whole-oxide|owning-lifecycle|lifetime|perf> \
         --model PATH --media PATH [--out PATH] [--device N] [--iterations N]"
    );
    std::process::exit(2)
}

fn value<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|index| args.get(index + 1))
        .map(String::as_str)
}

fn required<'a>(args: &'a [String], name: &str) -> &'a str {
    value(args, name).unwrap_or_else(|| {
        eprintln!("missing {name}");
        usage()
    })
}

fn read_json(path: &Path) -> Result<Value, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|error| format!("{}: {error}", path.display()))
}

fn fixture_images(manifest: &Value, base: &Path) -> Result<Vec<HostImage>, String> {
    let fixtures = manifest
        .get("fixtures")
        .and_then(Value::as_array)
        .ok_or("manifest has no fixtures array")?;
    let mut images = Vec::new();
    for fixture in fixtures {
        if fixture.get("kind").and_then(Value::as_str) != Some("image") {
            continue;
        }
        let name = fixture
            .get("name")
            .and_then(Value::as_str)
            .ok_or("fixture name")?
            .to_owned();
        let grid = fixture
            .get("grid")
            .and_then(Value::as_array)
            .ok_or("fixture grid")?;
        if grid.len() != 3 {
            return Err(format!("fixture {name} grid must have three entries"));
        }
        let dimension = |index: usize| -> Result<usize, String> {
            grid[index]
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| format!("fixture {name} invalid grid entry"))
        };
        let merge = fixture
            .get("merge")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or("fixture merge")?;
        let media = MediaGrid::new(dimension(0)?, dimension(1)?, dimension(2)?, merge)
            .map_err(|error| error.to_string())?;
        let pixels_path = fixture
            .get("pixels")
            .and_then(Value::as_str)
            .ok_or("fixture pixels path")?;
        let pixels_path = if Path::new(pixels_path).is_absolute() {
            PathBuf::from(pixels_path)
        } else {
            base.join(pixels_path)
        };
        let bytes = std::fs::read(&pixels_path)
            .map_err(|error| format!("{}: {error}", pixels_path.display()))?;
        if !bytes.len().is_multiple_of(4) {
            return Err(format!("{} is not raw f32", pixels_path.display()));
        }
        let pixels = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four bytes")))
            .collect::<Vec<_>>();
        let token_start = fixture
            .get("token_start")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(0);
        images.push(HostImage {
            name,
            pixels,
            grid: media,
            token_start,
        });
    }
    Ok(images)
}

fn plan_report(config: &Qwen35VlConfig, manifest: &Value) -> Result<Value, String> {
    let fixtures = manifest
        .get("fixtures")
        .and_then(Value::as_array)
        .ok_or("manifest has no fixtures array")?;
    let mut report = Vec::new();
    for fixture in fixtures {
        let name = fixture
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("fixture");
        let kind = fixture
            .get("kind")
            .and_then(Value::as_str)
            .unwrap_or("image");
        let height = fixture
            .get("height")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or("fixture height")?;
        let width = fixture
            .get("width")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or("fixture width")?;
        let frames = fixture
            .get("frames")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(1);
        let plan = if kind == "video" {
            plan_video(height, width, frames, &config.media, &config.vision)
        } else {
            plan_image(height, width, &config.media, &config.vision)
        }
        .map_err(|error| error.to_string())?;
        let token_start = fixture
            .get("token_start")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(0);
        let span = MediaSpan {
            kind: if kind == "video" {
                MediaKind::Video
            } else {
                MediaKind::Image
            },
            token_start,
            token_count: plan
                .grid
                .merged_tokens()
                .map_err(|error| error.to_string())?,
            grid: plan.grid,
        };
        let tokens = token_start + span.token_count + 2;
        let fusion =
            prepare_text_fusion(config, tokens, &[span]).map_err(|error| error.to_string())?;
        report.push(json!({
            "name": name,
            "kind": kind,
            "resized": [plan.resized_height, plan.resized_width],
            "grid_thw": plan.grid.grid_thw(),
            "merged_tokens": span.token_count,
            "positions": fusion.positions,
            "mrope_layers": fusion.mrope_layers(config),
            "gdn_layers": fusion.gdn_layers(config),
        }));
    }
    Ok(Value::Array(report))
}

struct EmbeddingRow {
    index: usize,
    kind: &'static str,
    values: Vec<f32>,
    nested: Value,
}

fn embedding_rows(embeddings: Embeddings) -> Result<Vec<EmbeddingRow>, String> {
    let mut rows = Vec::new();
    for (index, embedding) in embeddings {
        match embedding {
            Embedding::Pooled(values) => rows.push(EmbeddingRow {
                index,
                kind: "pooled",
                nested: json!(&values),
                values,
            }),
            Embedding::All(values) => rows.push(EmbeddingRow {
                index,
                kind: "all",
                values: values.iter().flatten().copied().collect(),
                nested: json!(&values),
            }),
        }
    }
    rows.sort_by_key(|row| row.index);
    if rows.is_empty() {
        return Err("owning lifecycle produced no embedding rows".to_owned());
    }
    Ok(rows)
}

fn complete_rows(inference: Inference<Embeddings>) -> Result<Vec<EmbeddingRow>, String> {
    embedding_rows(inference.complete().map_err(|error| error.to_string())?)
}

fn row_json(rows: &[EmbeddingRow]) -> Value {
    Value::Array(
        rows.iter()
            .map(|row| json!({"index": row.index, "kind": row.kind, "values": &row.nested}))
            .collect(),
    )
}

fn compare_rows(label: &str, baseline: &[EmbeddingRow], other: &[EmbeddingRow]) -> Value {
    let coverage = baseline.len() == other.len()
        && baseline
            .iter()
            .zip(other)
            .all(|(a, b)| a.index == b.index && a.kind == b.kind);
    let finite_count = other
        .iter()
        .flat_map(|row| row.values.iter())
        .filter(|value| value.is_finite())
        .count();
    let total_count = other.iter().map(|row| row.values.len()).sum::<usize>();
    let mut max_abs = 0.0f32;
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    if coverage {
        for (a, b) in baseline.iter().zip(other) {
            for (x, y) in a.values.iter().zip(&b.values) {
                let delta = (x - y).abs();
                if delta > max_abs {
                    max_abs = delta;
                }
                dot += x * y;
                norm_a += x * x;
                norm_b += y * y;
            }
        }
    }
    let cosine = if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a.sqrt() * norm_b.sqrt())
    } else {
        0.0
    };
    let nonempty_finite = total_count > 0 && finite_count == total_count;
    let passed = coverage
        && nonempty_finite
        && total_count == baseline.iter().map(|row| row.values.len()).sum::<usize>()
        && max_abs <= 1e-5
        && cosine >= 0.9999;
    json!({
        "label": label,
        "coverage": coverage,
        "rows": other.len(),
        "elements": total_count,
        "finite_elements": finite_count,
        "nonempty_finite": nonempty_finite,
        "max_abs": max_abs,
        "cosine": cosine,
        "passed": passed,
    })
}

fn changed_shape_batch(mut batch: Batch) -> Result<Batch, String> {
    let last = *batch
        .cumulative_seq_lengths
        .last()
        .ok_or("changed shape requires cumulative lengths")?;
    let next = last.checked_add(1).ok_or("changed shape length overflow")?;
    batch.input_ids.push(0);
    batch.token_type_ids.push(0);
    let position = batch
        .position_ids
        .last()
        .copied()
        .unwrap_or(last)
        .checked_add(1)
        .ok_or("changed shape position overflow")?;
    batch.position_ids.push(position);
    let last_index = batch.cumulative_seq_lengths.len() - 1;
    batch.cumulative_seq_lengths[last_index] = next;
    batch.max_length = next;
    Ok(batch)
}

fn invalid_media_batch(mut batch: Batch) -> Result<Batch, String> {
    let source = batch
        .multimodal
        .iter()
        .find_map(|slot| slot.as_ref())
        .ok_or("invalid media requires an existing image encoding")?;
    let mut images = Vec::new();
    for (start, image) in &source.images {
        let mut invalid = (**image).clone();
        invalid.pixels.pop();
        images.push((*start, Arc::new(invalid)));
    }
    if images.is_empty() {
        return Err("invalid media requires an existing image patch".to_owned());
    }
    let encoding = MultimodalEncoding {
        images,
        audios: source
            .audios
            .iter()
            .map(|(start, audio)| (*start, Arc::clone(audio)))
            .collect(),
        position_ids: source.position_ids.clone(),
        memory: source.memory.clone(),
    };
    let sequences = batch.cumulative_seq_lengths.len().saturating_sub(1);
    let encoding = Arc::new(encoding);
    batch.multimodal = (0..sequences)
        .map(|index| (index == 0).then(|| Arc::clone(&encoding)))
        .collect();
    Ok(batch)
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        usage();
    }
    let command = args[1].as_str();
    let model = PathBuf::from(required(&args, "--model"));
    let config = match Qwen35VlConfig::from_path(&model) {
        Ok(config) => config,
        Err(error) if error.to_string().contains("preprocessor_config.json") => {
            Qwen35VlConfig::from_config_path(&model).map_err(|error| error.to_string())?
        }
        Err(error) => return Err(error.to_string()),
    };
    let media_path = value(&args, "--media").map(PathBuf::from);
    let manifest = match &media_path {
        Some(path) => read_json(path)?,
        None => json!({"fixtures": []}),
    };
    let base = media_path
        .as_deref()
        .and_then(Path::parent)
        .unwrap_or_else(|| Path::new("."));
    let out = value(&args, "--out");
    match command {
        "inventory" => {
            let report = infinity_backend_oxide::validate_vision_inventory_path(&config, &model)
                .map_err(|error| error.to_string())?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "model": model,
                    "model_type": config.model_type,
                    "text_model_type": config.text.model_type,
                    "vision_depth": config.vision.depth,
                    "rotary_dim": config.text.rotary_dim,
                    "mrope_section": config.text.mrope_section,
                    "deepstack": config.vision.deepstack_visual_indexes,
                    "vision_tensors": report.vision_tensors,
                    "text_tensors": report.text_tensors,
                    "total_tensors": report.total_tensors,
                }))
                .map_err(|error| error.to_string())?
            );
        }
        "plan" => {
            let report = plan_report(&config, &manifest)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&report).map_err(|error| error.to_string())?
            );
        }
        "plan-full" => {
            let plan = Qwen35VlEngine::run_plan(&model).map_err(|error| error.to_string())?;
            let modes: Vec<Value> = plan
                .modes
                .iter()
                .map(|mode| {
                    json!({
                        "component": mode.component,
                        "heads": mode.heads,
                        "kv_heads": mode.kv_heads,
                        "head_dim": mode.head_dim,
                        "causal": mode.causal,
                        "expected_mode": mode.expected_mode,
                        "registry_mode": mode.registry_mode,
                        "present": mode.present(),
                        "note": mode.note,
                    })
                })
                .collect();
            let model_type = if plan.text_hidden == 4096 || plan.text_hidden == 5120 {
                infinity_v3_backend_core::ModelType::Decision
            } else {
                infinity_v3_backend_core::ModelType::Embedding(Pool::Mean)
            };
            let proof =
                infinity_backend_oxide::normal_factory_proof(&model, "bfloat16", &model_type)
                    .map(|proof| {
                        json!({
                            "is_qwen35_checkpoint": proof.is_qwen35_checkpoint,
                            "serving_route": proof.serving_route,
                            "dtype_accepted": proof.dtype_accepted,
                            "run_plan_ready": proof.run_plan_ready,
                            "gaps": proof.gaps,
                        })
                    })
                    .map_err(|error| error.to_string())?;
            let text = serde_json::to_string_pretty(&json!({
                "path": plan.path,
                "family": plan.family,
                "dtype": plan.dtype,
                "vision_tensors": plan.vision_tensors,
                "text_hidden": plan.text_hidden,
                "text_layers": plan.text_layers,
                "text_causal": plan.text_causal,
                "modes": modes,
                "text_admission": plan.text_admission,
                "moe_owner_required": plan.moe_owner_required,
                "fp16_supported": plan.fp16_supported,
                "projection_score_route": plan.projection_score_route,
                "gaps": plan.gaps,
                "normal_factory_ready": plan.normal_factory_ready,
                "factory_proof": proof,
            }))
            .map_err(|error| error.to_string())?;
            write_or_print(out, &text)?;
        }
        "tower-cpu" => {
            let images = fixture_images(&manifest, base)?;
            let weights = VisionHostWeights::from_checkpoint(&model, &config)
                .map_err(|error| error.to_string())?;
            let mut rows = Vec::new();
            for image in &images {
                let output = vision_forward(&config, &weights, std::slice::from_ref(image))
                    .map_err(|error| error.to_string())?;
                rows.push(json!({
                    "name": image.name,
                    "merged_rows": output.merged_rows,
                    "out_hidden": output.out_hidden,
                    "merged": output.merged,
                    "deepstack": output.deepstack,
                }));
            }
            let text = serde_json::to_string_pretty(&json!({"arm": "oxide-cpu", "rows": rows}))
                .map_err(|error| error.to_string())?;
            write_or_print(out, &text)?;
        }
        "tower-device" => {
            let device = value(&args, "--device")
                .unwrap_or("0")
                .parse::<u32>()
                .map_err(|error| format!("device: {error}"))?;
            let tower = Qwen35VlTower::open(&model, device).map_err(|error| error.to_string())?;
            let images = fixture_images(&manifest, base)?;
            let outputs = tower
                .image_features(&images)
                .map_err(|error| error.to_string())?;
            let rows: Vec<Value> = images
                .iter()
                .zip(outputs)
                .map(|(image, output)| {
                    json!({
                        "name": image.name,
                        "merged_rows": output.merged_rows,
                        "out_hidden": output.out_hidden,
                        "merged": output.merged,
                        "deepstack": output.deepstack,
                    })
                })
                .collect();
            let text = serde_json::to_string_pretty(&json!({"arm": "oxide-device", "rows": rows}))
                .map_err(|error| error.to_string())?;
            write_or_print(out, &text)?;
        }
        "whole-factory" => {
            let device = value(&args, "--device")
                .unwrap_or("0")
                .parse::<usize>()
                .map_err(|error| format!("device: {error}"))?;
            let pool = match value(&args, "--pool").unwrap_or("mean") {
                "cls" => Pool::Cls,
                "mean" => Pool::Mean,
                "last" | "last_token" => Pool::LastToken,
                other => return Err(format!("unsupported pool {other}")),
            };
            let (batch, model_type) = normal_factory_batch(&manifest, base, pool)?;
            let backend = OxideBackend::new(&model, "bfloat16", model_type, device)
                .map_err(|error| error.to_string())?;
            match backend {
                _ if matches!(
                    manifest.get("text").and_then(|t| t.get("decision")),
                    Some(Value::Bool(true))
                ) =>
                {
                    let options = manifest["text"]["option_tokens"]
                        .as_array()
                        .ok_or("decision Batch requires text.option_tokens")?
                        .iter()
                        .map(|value| {
                            value
                                .as_u64()
                                .and_then(|v| u32::try_from(v).ok())
                                .ok_or("invalid option token")
                        })
                        .collect::<Result<Vec<u32>, _>>()?;
                    let prefix_length = manifest["text"]["prefix_length"]
                        .as_u64()
                        .and_then(|v| usize::try_from(v).ok())
                        .unwrap_or(0);
                    let decisions = vec![DecisionInput::BlockOptionTokens {
                        token_ids: options,
                        prefix_length,
                    }];
                    let outputs = backend
                        .decide(batch, decisions)
                        .map_err(|error| error.to_string())?;
                    let rows: Vec<Value> = outputs
                        .into_iter()
                        .map(|output| json!({"kind":"decision","logits":output.logits,"action_probability":output.action_probability}))
                        .collect();
                    let text = serde_json::to_string_pretty(
                        &json!({"arm":"oxide-whole-factory","rows":rows}),
                    )
                    .map_err(|error| error.to_string())?;
                    write_or_print(out, &text)?;
                }
                _ => {
                    let embeddings = backend.embed(batch).map_err(|error| error.to_string())?;
                    let mut rows = Vec::new();
                    for (index, embedding) in embeddings {
                        match embedding {
                            Embedding::Pooled(values) => {
                                rows.push(json!({"index":index,"kind":"pooled","values":values}))
                            }
                            Embedding::All(values) => {
                                rows.push(json!({"index":index,"kind":"all","values":values}))
                            }
                        }
                    }
                    let text = serde_json::to_string_pretty(
                        &json!({"arm":"oxide-whole-factory","rows":rows}),
                    )
                    .map_err(|error| error.to_string())?;
                    write_or_print(out, &text)?;
                }
            }
        }
        "whole-oxide" => {
            let device = value(&args, "--device")
                .unwrap_or("0")
                .parse::<usize>()
                .map_err(|error| format!("device: {error}"))?;
            let pool = match value(&args, "--pool").unwrap_or("mean") {
                "cls" => Pool::Cls,
                "mean" => Pool::Mean,
                "last" | "last_token" => Pool::LastToken,
                other => return Err(format!("unsupported pool {other}")),
            };
            let text = dto::text_options(&args)?;
            let engine =
                Qwen35VlEngine::new(&model, pool, device, text, Qwen35VlOptions::default())
                    .map_err(|error| error.to_string())?;
            let batch: Batch = dto::batch(&manifest)?;
            let requests = whole_requests(&manifest, base)?;
            let embeddings = engine
                .embed(batch, &requests)
                .map_err(|error| error.to_string())?;
            let mut rows = Vec::new();
            for (index, embedding) in embeddings {
                match embedding {
                    Embedding::Pooled(values) => rows.push(json!({
                        "index": index, "kind": "pooled", "values": values
                    })),
                    Embedding::All(values) => rows.push(json!({
                        "index": index, "kind": "all", "values": values
                    })),
                }
            }
            let text = serde_json::to_string_pretty(&json!({"arm": "oxide-whole", "rows": rows}))
                .map_err(|error| error.to_string())?;
            write_or_print(out, &text)?;
        }
        "owning-lifecycle" => {
            let device = value(&args, "--device")
                .unwrap_or("0")
                .parse::<usize>()
                .map_err(|error| format!("device: {error}"))?;
            let pool = match value(&args, "--pool").unwrap_or("mean") {
                "cls" => Pool::Cls,
                "mean" => Pool::Mean,
                "last" | "last_token" => Pool::LastToken,
                other => return Err(format!("unsupported pool {other}")),
            };
            let (batch, model_type) = normal_factory_batch(&manifest, base, pool)?;
            if !matches!(&model_type, ModelType::Embedding(_)) {
                return Err(
                    "owning-lifecycle requires the 0.8B embedding decision=false manifest"
                        .to_owned(),
                );
            }
            let backend = OxideBackend::new(&model, "bfloat16", model_type, device)
                .map_err(|error| error.to_string())?;
            let baseline = complete_rows(
                backend
                    .submit_embed(batch.clone())
                    .map_err(|error| error.to_string())?,
            )?;
            let a = backend
                .submit_embed(batch.clone())
                .map_err(|error| error.to_string())?;
            let b = backend
                .submit_embed(batch.clone())
                .map_err(|error| error.to_string())?;
            drop(a);
            let b_rows = complete_rows(b)?;
            let changed = changed_shape_batch(batch.clone())?;
            let changed_rows = complete_rows(
                backend
                    .submit_embed(changed)
                    .map_err(|error| error.to_string())?,
            )?;
            let invalid = invalid_media_batch(batch.clone())?;
            let invalid_error = match backend.submit_embed(invalid) {
                Ok(_) => {
                    return Err(
                        "invalid media was accepted by owning lifecycle preflight".to_owned()
                    )
                }
                Err(BackendError::Inference(message)) if !message.is_empty() => message,
                Err(other) => {
                    return Err(format!(
                        "invalid media returned non-preflight error: {other}"
                    ))
                }
            };
            let recovery_rows = complete_rows(
                backend
                    .submit_embed(batch)
                    .map_err(|error| error.to_string())?,
            )?;
            let basis = compare_rows("b_vs_baseline", &baseline, &b_rows);
            let recovery = compare_rows("recovery_vs_baseline", &baseline, &recovery_rows);
            let changed_finite = changed_rows
                .iter()
                .flat_map(|row| row.values.iter())
                .all(|value| value.is_finite());
            let passed = basis
                .get("passed")
                .and_then(Value::as_bool)
                .unwrap_or(false)
                && recovery
                    .get("passed")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                && changed_finite
                && !changed_rows.is_empty();
            let report = json!({
                "arm": "oxide-owning-lifecycle",
                "model_type": "Embedding",
                "baseline": row_json(&baseline),
                "held_a": {"dropped_uncalled": true},
                "completed_b": row_json(&b_rows),
                "comparison_b_vs_baseline": basis,
                "changed_shape": {"rows": row_json(&changed_rows), "nonempty_finite": changed_finite},
                "invalid_media": {"rejected": true, "error": invalid_error},
                "recovery": row_json(&recovery_rows),
                "comparison_recovery_vs_baseline": recovery,
                "blocking_scope": "normal blocking Backend::submit_embed; no held-third or fake Overloaded claim",
                "status": if passed { "passed" } else { "failed" },
            });
            let text = serde_json::to_string_pretty(&report).map_err(|error| error.to_string())?;
            write_or_print(out, &text)?;
            if !passed {
                return Err("owning lifecycle comparison failed".to_owned());
            }
        }
        "lifetime" => {
            let device = value(&args, "--device")
                .unwrap_or("0")
                .parse::<u32>()
                .map_err(|error| format!("device: {error}"))?;
            let iterations = value(&args, "--iterations")
                .unwrap_or("4")
                .parse::<usize>()
                .map_err(|error| format!("iterations: {error}"))?;
            let images = fixture_images(&manifest, base)?;
            let mut samples = Vec::new();
            for iteration in 0..iterations {
                let tower =
                    Qwen35VlTower::open(&model, device).map_err(|error| error.to_string())?;
                let start = Instant::now();
                let outputs = tower
                    .image_features(&images)
                    .map_err(|error| error.to_string())?;
                samples.push(json!({
                    "iteration": iteration,
                    "milliseconds": start.elapsed().as_secs_f64() * 1000.0,
                    "rows": outputs.iter().map(|output| output.merged_rows).collect::<Vec<_>>(),
                }));
            }
            let text =
                serde_json::to_string_pretty(&json!({"arm": "oxide-lifetime", "samples": samples}))
                    .map_err(|error| error.to_string())?;
            write_or_print(out, &text)?;
        }
        "perf" => {
            let device = value(&args, "--device")
                .unwrap_or("0")
                .parse::<u32>()
                .map_err(|error| format!("device: {error}"))?;
            let iterations = value(&args, "--iterations")
                .unwrap_or("10")
                .parse::<usize>()
                .map_err(|error| format!("iterations: {error}"))?;
            let images = fixture_images(&manifest, base)?;
            let tower = Qwen35VlTower::open(&model, device).map_err(|error| error.to_string())?;
            let mut samples = Vec::new();
            for _ in 0..iterations {
                let start = Instant::now();
                let outputs = tower
                    .image_features(&images)
                    .map_err(|error| error.to_string())?;
                let rows: usize = outputs.iter().map(|output| output.merged_rows).sum();
                samples.push(json!({
                    "milliseconds": start.elapsed().as_secs_f64() * 1000.0,
                    "merged_rows": rows,
                }));
            }
            let text =
                serde_json::to_string_pretty(&json!({"arm": "oxide-perf", "samples": samples}))
                    .map_err(|error| error.to_string())?;
            write_or_print(out, &text)?;
        }
        "compare" => {
            return Err("compare is performed by root-v1/qwen35_vl_root_controller.py".to_owned());
        }
        other => return Err(format!("unknown command {other}")),
    }
    Ok(())
}

fn write_or_print(out: Option<&str>, text: &str) -> Result<(), String> {
    match out {
        Some(path) => {
            std::fs::write(path, format!("{text}\n")).map_err(|error| format!("{path}: {error}"))
        }
        None => {
            println!("{text}");
            Ok(())
        }
    }
}

fn whole_requests(manifest: &Value, base: &Path) -> Result<Vec<Qwen35VlRequest>, String> {
    let fixtures = manifest
        .get("fixtures")
        .and_then(Value::as_array)
        .ok_or("manifest has no fixtures array")?;
    let mut requests = Vec::new();
    for fixture in fixtures {
        let name = fixture
            .get("name")
            .and_then(Value::as_str)
            .ok_or("fixture name")?
            .to_owned();
        let kind = match fixture
            .get("kind")
            .and_then(Value::as_str)
            .unwrap_or("image")
        {
            "image" => MediaKind::Image,
            "video" => MediaKind::Video,
            other => return Err(format!("unsupported media kind {other}")),
        };
        let grid = fixture
            .get("grid")
            .and_then(Value::as_array)
            .ok_or("fixture grid")?;
        if grid.len() != 3 {
            return Err(format!("fixture {name} grid must have three entries"));
        }
        let dimension = |index: usize| -> Result<usize, String> {
            grid[index]
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| format!("fixture {name} invalid grid entry"))
        };
        let merge = fixture
            .get("merge")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or("fixture merge")?;
        let media = MediaGrid::new(dimension(0)?, dimension(1)?, dimension(2)?, merge)
            .map_err(|error| error.to_string())?;
        let pixels_path = fixture
            .get("pixels")
            .and_then(Value::as_str)
            .ok_or("fixture pixels path")?;
        let pixels_path = if Path::new(pixels_path).is_absolute() {
            PathBuf::from(pixels_path)
        } else {
            base.join(pixels_path)
        };
        let bytes = std::fs::read(&pixels_path)
            .map_err(|error| format!("{}: {error}", pixels_path.display()))?;
        let pixels = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four bytes")))
            .collect();
        let token_start = fixture
            .get("token_start")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(0);
        requests.push(Qwen35VlRequest {
            name,
            kind,
            pixels,
            grid: media,
            token_start,
        });
    }
    Ok(requests)
}

mod dto {
    use super::*;

    pub fn text_options(
        args: &[String],
    ) -> Result<infinity_backend_oxide::Qwen35TextOptions, String> {
        let path = |name: &str, env: &str| -> Result<PathBuf, String> {
            value(args, name)
                .map(PathBuf::from)
                .or_else(|| std::env::var_os(env).map(PathBuf::from))
                .ok_or_else(|| format!("missing {name} or {env}"))
        };
        Ok(infinity_backend_oxide::Qwen35TextOptions {
            causal: !args.iter().any(|arg| arg == "--noncausal"),
            fa4_root: path("--fa4-root", "FA4_ROOT")?,
            gdn_raw_kernels: path("--gdn-raw-kernels", "OXIDE_GDN_RAW_KERNELS")?,
            gdn_provider: path("--gdn-provider", "OXIDE_GDN_PROVIDER")?,
            gdn_aux_kernels: path("--gdn-aux-kernels", "OXIDE_GDN_AUX_KERNELS")?,
        })
    }

    pub fn batch(manifest: &Value) -> Result<Batch, String> {
        let text = manifest.get("text").ok_or("manifest has no text object")?;
        let list_u32 = |name: &str| -> Result<Vec<u32>, String> {
            text.get(name)
                .and_then(Value::as_array)
                .map(|values| {
                    values
                        .iter()
                        .map(|value| {
                            value
                                .as_u64()
                                .and_then(|value| u32::try_from(value).ok())
                                .ok_or_else(|| format!("invalid text.{name}"))
                        })
                        .collect()
                })
                .transpose()
                .map(|value| value.unwrap_or_default())
        };
        let input_ids = list_u32("input_ids")?;
        if input_ids.is_empty() {
            return Err("text.input_ids is empty".to_owned());
        }
        let cumulative = list_u32("cumulative_seq_lengths")?;
        if cumulative.is_empty() {
            return Err("text.cumulative_seq_lengths is empty".to_owned());
        }
        let position_ids = {
            let positions = list_u32("position_ids")?;
            if positions.is_empty() {
                (0..input_ids.len() as u32).collect()
            } else {
                positions
            }
        };
        let token_type_ids = {
            let types = list_u32("token_type_ids")?;
            if types.is_empty() {
                vec![0; input_ids.len()]
            } else {
                types
            }
        };
        let max_length = *cumulative.last().ok_or("empty cumulative")?;
        Ok(Batch {
            multimodal: Vec::new(),
            input_ids,
            token_type_ids,
            position_ids,
            cumulative_seq_lengths: cumulative,
            max_length,
            pooled_indices: list_u32("pooled_indices")?,
            raw_indices: list_u32("raw_indices")?,
            compact_input_ids: None,
            compact_position_ids: None,
            scatter_unfold: None,
            fold_gather: None,
            tokens: Vec::new(),
            offsets: Vec::new(),
        })
    }
}

fn normal_factory_batch(
    manifest: &Value,
    base: &Path,
    pool: Pool,
) -> Result<(Batch, ModelType), String> {
    let mut batch = dto::batch(manifest)?;
    let mut images = Vec::new();
    for fixture in manifest
        .get("fixtures")
        .and_then(Value::as_array)
        .ok_or("manifest has no fixtures array")?
    {
        if fixture
            .get("kind")
            .and_then(Value::as_str)
            .unwrap_or("image")
            != "image"
        {
            return Err("normal factory smoke currently uses image fixtures".to_owned());
        }
        let grid = fixture
            .get("grid")
            .and_then(Value::as_array)
            .ok_or("fixture grid")?;
        if grid.len() != 3 {
            return Err("fixture grid must have three entries".to_owned());
        }
        let dimension = |index: usize| -> Result<usize, String> {
            grid[index]
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| "invalid fixture grid".to_owned())
        };
        let patch_dim = fixture
            .get("patch_dim")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or("fixture patch_dim")?;
        let merge = fixture
            .get("merge")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or("fixture merge")?;
        let pixels_path = fixture
            .get("pixels")
            .and_then(Value::as_str)
            .ok_or("fixture pixels path")?;
        let pixels_path = if Path::new(pixels_path).is_absolute() {
            PathBuf::from(pixels_path)
        } else {
            base.join(pixels_path)
        };
        let bytes = std::fs::read(&pixels_path)
            .map_err(|error| format!("{}: {error}", pixels_path.display()))?;
        let pixels = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four bytes")))
            .collect::<Vec<_>>();
        let token_start = fixture
            .get("token_start")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(0);
        images.push((
            token_start,
            Arc::new(ImagePatches {
                pixels,
                grid_thw: [dimension(0)?, dimension(1)?, dimension(2)?],
                patch_dim,
                merge_size: merge,
            }),
        ));
    }
    let encoding = MultimodalEncoding {
        images,
        audios: Vec::new(),
        position_ids: std::array::from_fn(|_| Vec::new()),
        memory: None,
    };
    let sequences = batch.cumulative_seq_lengths.len().saturating_sub(1);
    batch.multimodal = (0..sequences)
        .map(|index| {
            (index == 0).then(|| {
                Arc::new(MultimodalEncoding {
                    images: encoding.images.clone(),
                    audios: Vec::new(),
                    position_ids: std::array::from_fn(|_| Vec::new()),
                    memory: None,
                })
            })
        })
        .collect();
    let decision = manifest
        .get("text")
        .and_then(|text| text.get("decision"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let model_type = if decision {
        ModelType::Decision
    } else {
        ModelType::Embedding(pool)
    };
    Ok((batch, model_type))
}
