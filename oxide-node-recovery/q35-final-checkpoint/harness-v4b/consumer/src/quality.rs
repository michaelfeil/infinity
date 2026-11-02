use crate::batch::{canonical_expected, physical_layout};
use crate::cli::{ensure_model_is_directory, Args};
use crate::fixtures::{backend_batch, load_cases};
use crate::metadata;
use crate::outputs::canonicalize;
use crate::report::{output_json, variant_matches, variant_name, Report};
use infinity_v3_backend::{Backend, BackendKind, DType, ModelType, Submission, TorchDevice};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs;

fn hidden_width(model: &std::path::Path) -> Result<usize, String> {
    let data = fs::read(model.join("config.json")).map_err(|e| format!("config.json: {e}"))?;
    let value: Value =
        serde_json::from_slice(&data).map_err(|e| format!("config.json parse: {e}"))?;
    let text = value.get("text_config").unwrap_or(&value);
    text.get("hidden_size")
        .or_else(|| value.get("hidden_size"))
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .filter(|value| *value > 0)
        .ok_or_else(|| "config.json has no positive hidden_size".into())
}

fn submission_name(submission: &Submission<infinity_v3_backend::Embeddings>) -> &'static str {
    match submission {
        Submission::Ready(..) => "Ready",
        Submission::Deferred(..) => "Deferred",
    }
}

fn batch_fingerprint(case: &crate::fixtures::LoadedCase) -> String {
    let data = serde_json::to_vec(&case.batch).unwrap_or_default();
    format!("{:x}", Sha256::digest(data))
}

pub async fn run(args: &Args, report: &mut Report) -> Result<(), String> {
    let model = args
        .model
        .as_ref()
        .ok_or_else(|| "--model is required".to_string())?;
    ensure_model_is_directory(model)?;
    let pool = args
        .pool
        .clone()
        .ok_or_else(|| "--pool is required".to_string())?;
    let hidden = hidden_width(model)?;
    let cases = load_cases(&args.cases, args.limit)?;

    let metadata_before = metadata::collect(
        model,
        report.product_source,
        report.product_freeze_sha256,
        report.canonical31_provider,
    )?;
    report.metadata = Some(
        json!({"before": metadata_before.clone(), "after": null, "before_after_equal": false, "payload_hashed": false}),
    );
    report.phase(
        "metadata-before",
        "ok",
        json!({"config": metadata_before["config"], "provider": metadata_before["provider"]}),
    );
    report.phase(
        "constructor",
        "start",
        json!({
            "route": "infinity_v3_backend::Backend::new_selected",
            "kind": "oxide",
            "torch_device": "cuda",
            "dtype": "bfloat16",
            "pool": pool.to_string(),
            "engine_count": 1,
            "no_device_count_query": true,
        }),
    );
    let backend = Backend::new_selected(
        model.clone(),
        None,
        DType::Bfloat16,
        ModelType::Embedding(pool.clone()),
        None,
        0,
        false,
        BackendKind::Oxide,
        TorchDevice::Cuda,
    )
    .await
    .map_err(|error| report.record_backend_error("quality/constructor", &error))?;
    report.engine_count = 1;
    report.normal_factory_route = true;
    if let Err(error) = backend.health().await {
        return Err(report.record_backend_error("quality/health-before", &error));
    }
    report.phase("constructor", "ok", json!({"engine_count": 1}));
    report.phase("health", "ok", json!({"before": true}));

    let mut results = Vec::new();
    let mut failures = Vec::new();
    report.metadata = Some(
        json!({"before": metadata_before.clone(), "after": null, "before_after_equal": false, "payload_hashed": false}),
    );
    report.quality = Some(
        json!({"route": "normal-service-factory", "pool": pool.to_string(), "shape": {"D": hidden, "H": hidden}, "cases": [], "failure_count": 0, "numeric_empty_outputs_are_semantic_only": true}),
    );
    report.checkpoint();
    for item in cases {
        let name = item.case.name.clone();
        let fingerprint = batch_fingerprint(&item);
        if item.case.valid {
            let layout = physical_layout(&item.batch, hidden, hidden)
                .map_err(|e| format!("{name}: physical layout: {e}"))?;
            let (expected, canonical) = canonical_expected(&layout)
                .map_err(|e| format!("{name}: canonical expectation: {e}"))?;
            let batch = backend_batch(&item)?;
            let submission = backend.submit_embed(batch).await.map_err(|error| {
                report.record_backend_error(&format!("quality/{name}/submit"), &error)
            })?;
            let submission_type = submission_name(&submission).to_string();
            if layout.physical_float_count == 0 && !matches!(submission, Submission::Ready(..)) {
                return Err(format!(
                    "{name}: no-selection request must return Ready with no queued output"
                ));
            }
            let started = std::time::Instant::now();
            let (embeddings, duration) = submission.complete().await.map_err(|error| {
                report.record_backend_error(&format!("quality/{name}/complete"), &error)
            })?;
            let canonical_outputs =
                canonicalize(&embeddings, &expected, &canonical, hidden, hidden)
                    .map_err(|e| format!("{name}: canonicalize: {e}"))?;
            let output = output_json(canonical_outputs);
            results.push(json!({
                "name": name,
                "valid": true,
                "status": "ok",
                "batch_file": item.path.display().to_string(),
                "batch_sha256": item.file_sha256,
                "batch_fingerprint_sha256": fingerprint,
                "submission_type": submission_type,
                "backend_duration_ns": duration.as_nanos(),
                "wall_duration_ns": started.elapsed().as_nanos(),
                "shape": {"D": hidden, "H": hidden},
                "requested": {
                    "input_ids": item.batch.input_ids,
                    "position_ids": item.batch.position_ids,
                    "token_type_ids": item.batch.token_type_ids,
                    "cumulative_seq_lengths": item.batch.cumulative_seq_lengths,
                    "pooled_indices": item.batch.pooled_indices,
                    "raw_indices": item.batch.raw_indices,
                    "max_length": item.batch.max_length,
                },
                "output": output,
            }));
        } else {
            let mut batch = backend_batch(&item)?;
            if item.case.special.as_deref() == Some("empty") {
                batch.multimodal.clear();
            }
            let expected = item
                .case
                .expected_error_variant
                .as_deref()
                .ok_or_else(|| format!("{name}: invalid case lacks expected variant"))?;
            match backend.submit_embed(batch).await {
                Ok(_) => {
                    let message = format!("{name}: invalid case unexpectedly admitted");
                    failures.push(message.clone());
                    results.push(json!({"name": name, "valid": false, "status": "unexpected-ok"}));
                }
                Err(error) => {
                    let actual = variant_name(&error);
                    let matched = variant_matches(&error, expected);
                    if !matched {
                        failures.push(report.record_backend_error(
                            &format!("quality/{name}/invalid-variant"),
                            &error,
                        ));
                    }
                    results.push(json!({
                        "name": name,
                        "valid": false,
                        "status": if matched {"ok"} else {"variant-mismatch"},
                        "batch_file": item.path.display().to_string(),
                        "batch_sha256": item.file_sha256,
                        "batch_fingerprint_sha256": fingerprint,
                        "expected_error_variant": expected,
                        "actual_error_variant": actual,
                        "error": error.to_string(),
                        "requested": {
                            "input_ids": item.batch.input_ids,
                            "position_ids": item.batch.position_ids,
                            "cumulative_seq_lengths": item.batch.cumulative_seq_lengths,
                            "pooled_indices": item.batch.pooled_indices,
                            "raw_indices": item.batch.raw_indices,
                            "max_length": item.batch.max_length,
                            "special": item.case.special,
                        },
                    }));
                }
            }
        }
        // Persist completed vectors before another request can fail.
        if let Some(quality) = report.quality.as_mut() {
            quality["cases"] = json!(results);
            quality["failure_count"] = json!(failures.len());
        }
        report.checkpoint();
    }
    let metadata_after = metadata::collect(
        model,
        report.product_source,
        report.product_freeze_sha256,
        report.canonical31_provider,
    )?;
    let metadata_equal = metadata_before == metadata_after;
    report.metadata = Some(json!({
        "before": metadata_before,
        "after": metadata_after,
        "before_after_equal": metadata_equal,
        "payload_hashed": false,
    }));
    report.checkpoint();
    if !metadata_equal {
        return Err("quality metadata before/after identity changed".into());
    }
    report.phase(
        "metadata-after",
        "ok",
        json!({"equal": true, "payload_hashed": false}),
    );
    let healthy_after = backend.health().await;
    match healthy_after {
        Ok(()) => report.phase("health", "ok", json!({"after": true})),
        Err(error) => {
            failures.push(report.record_backend_error("quality/health-after", &error));
            report.phase(
                "health",
                "error",
                json!({"after": false, "error": error.to_string()}),
            );
        }
    }
    report.quality = Some(json!({
        "route": "normal-service-factory",
        "normal_factory": "infinity_v3_backend::Backend::new_selected(BackendKind::Oxide, TorchDevice::Cuda)",
        "pool": pool.to_string(),
        "shape": {"D": hidden, "H": hidden},
        "cases": results,
        "failure_count": failures.len(),
        "numeric_empty_outputs_are_semantic_only": true,
    }));
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures.join("; "))
    }
}
