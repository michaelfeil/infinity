use crate::batch::{canonical_expected, physical_layout};
use crate::cli::{ensure_model_is_directory, Args};
use crate::fixtures::{backend_batch, load_cases, LoadedCase};
use crate::outputs::canonicalize;
use crate::report::{output_json, variant_matches, variant_name, Report};
use infinity_backend_oxide::Qwen35MoeBackend;
use infinity_v3_backend::{BackendKind, DType, Embeddings, ModelType, Submission, TorchDevice};
use infinity_v3_backend_core::{Backend as CoreBackend, BackendError, Inference};
use serde_json::{json, Value};
use std::path::Path;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

#[derive(Debug, Clone, PartialEq, Eq)]
struct EntryIdentity {
    index: usize,
    kind: String,
    shape: Vec<usize>,
    words_u32: Vec<u32>,
}

struct Captured {
    json: Value,
    entries: Vec<EntryIdentity>,
}

struct BlockingSubmit<T> {
    started: mpsc::Receiver<()>,
    result: mpsc::Receiver<Result<T, BackendError>>,
    handle: JoinHandle<()>,
}

fn spawn_blocking_submit<T, F>(job: F) -> BlockingSubmit<T>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, BackendError> + Send + 'static,
{
    let (started_sender, started) = mpsc::channel();
    let (result_sender, result) = mpsc::channel();
    let handle = std::thread::Builder::new()
        .name("q35-core-blocking-submit".into())
        .spawn(move || {
            let _ = started_sender.send(());
            let _ = result_sender.send(job());
        })
        .expect("spawn core blocking submit worker");
    BlockingSubmit {
        started,
        result,
        handle,
    }
}

fn receive_bounded<T>(
    receiver: &mpsc::Receiver<Result<T, BackendError>>,
    timeout: Duration,
) -> Result<Result<T, BackendError>, String> {
    receiver.recv_timeout(timeout).map_err(|error| match error {
        RecvTimeoutError::Timeout => {
            format!("blocked submit did not wake within {timeout:?}")
        }
        RecvTimeoutError::Disconnected => {
            "blocked submit worker exited without a result".to_string()
        }
    })
}

fn hidden_width(model: &Path) -> Result<usize, String> {
    let data = std::fs::read(model.join("config.json")).map_err(|e| format!("config.json: {e}"))?;
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

fn find<'a>(cases: &'a [LoadedCase], name: &str) -> Result<&'a LoadedCase, String> {
    cases
        .iter()
        .find(|case| case.case.name == name)
        .ok_or_else(|| format!("fixture {name} is absent"))
}

fn capture(embeddings: &Embeddings, item: &LoadedCase, hidden: usize) -> Result<Captured, String> {
    let layout = physical_layout(&item.batch, hidden, hidden)
        .map_err(|e| format!("{}: physical layout: {e}", item.case.name))?;
    let (expected, canonical) = canonical_expected(&layout)
        .map_err(|e| format!("{}: canonical expectation: {e}", item.case.name))?;
    let canonical = canonicalize(embeddings, &expected, &canonical, hidden, hidden)
        .map_err(|e| format!("{}: canonicalize: {e}", item.case.name))?;
    let entries = canonical
        .outputs
        .iter()
        .map(|entry| EntryIdentity {
            index: entry.index,
            kind: entry.kind.clone(),
            shape: entry.shape.clone(),
            words_u32: entry.values.iter().map(|value| value.to_bits()).collect(),
        })
        .collect();
    Ok(Captured {
        json: output_json(canonical),
        entries,
    })
}

fn submission_name(submission: &Submission<Embeddings>) -> &'static str {
    match submission {
        Submission::Ready(..) => "Ready",
        Submission::Deferred(..) => "Deferred",
    }
}

fn submission_deferred(submission: &Submission<Embeddings>) -> bool {
    matches!(submission, Submission::Deferred(..))
}

fn core_deferred<T>(inference: &Inference<T>) -> bool {
    matches!(inference, Inference::Deferred(..))
}

fn compare(label: &str, actual: &Captured, baseline: &Captured) -> Result<(), String> {
    if actual.entries.len() != baseline.entries.len() {
        return Err(format!(
            "{label}: canonical entry count {} differs from baseline {}",
            actual.entries.len(),
            baseline.entries.len()
        ));
    }
    for (actual, baseline) in actual.entries.iter().zip(&baseline.entries) {
        if actual.index != baseline.index
            || actual.kind != baseline.kind
            || actual.shape != baseline.shape
        {
            return Err(format!(
                "{label}: canonical identity differs index/kind/shape {:?}/{:?}/{:?} vs {:?}/{:?}/{:?}",
                actual.index, actual.kind, actual.shape,
                baseline.index, baseline.kind, baseline.shape
            ));
        }
        if actual.words_u32 != baseline.words_u32 {
            return Err(format!(
                "{label}: canonical F32 words differ at index {} kind {}",
                actual.index, actual.kind
            ));
        }
    }
    Ok(())
}

pub async fn run_service(args: &Args, report: &mut Report) -> Result<(), String> {
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
    let a_case = find(&cases, "t7_a")?;
    let b_case = find(&cases, "t7_b")?;
    let metadata_before = crate::metadata::collect(
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
        json!({"route": "normal-service-factory", "pool": pool.to_string(), "engine_count": 1}),
    );
    let backend = infinity_v3_backend::Backend::new_selected(
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
    .map_err(|error| report.record_backend_error("service/constructor", &error))?;
    report.engine_count = 1;
    report.normal_factory_route = true;
    if let Err(error) = backend.health().await {
        return Err(report.record_backend_error("service/health-before", &error));
    }
    report.phase("constructor", "ok", json!({"engine_count": 1}));

    let base_a = capture(
        &backend
            .submit_embed(backend_batch(a_case)?)
            .await
            .map_err(|error| report.record_backend_error("baseline A submit", &error))?
            .complete()
            .await
            .map_err(|error| report.record_backend_error("baseline A complete", &error))?
            .0,
        a_case,
        hidden,
    )?;
    let base_b = capture(
        &backend
            .submit_embed(backend_batch(b_case)?)
            .await
            .map_err(|error| report.record_backend_error("baseline B submit", &error))?
            .complete()
            .await
            .map_err(|error| report.record_backend_error("baseline B complete", &error))?
            .0,
        b_case,
        hidden,
    )?;
    if base_a.entries == base_b.entries {
        return Err("service baseline A and B are not distinct".into());
    }
    report.phase(
        "baselines",
        "ok",
        json!({"A": base_a.json, "B": base_b.json, "distinct": true}),
    );

    let a = backend
        .submit_embed(backend_batch(a_case)?)
        .await
        .map_err(|error| report.record_backend_error("service burst A submit", &error))?;
    let b = backend
        .submit_embed(backend_batch(b_case)?)
        .await
        .map_err(|error| report.record_backend_error("service burst B submit", &error))?;
    let c = backend
        .submit_embed(backend_batch(a_case)?)
        .await
        .map_err(|error| report.record_backend_error("service burst C=A submit", &error))?;
    let a_type = submission_name(&a);
    let b_type = submission_name(&b);
    let c_type = submission_name(&c);
    let service_deferred = submission_deferred(&a)
        && submission_deferred(&b)
        && submission_deferred(&c);
    if !service_deferred {
        return Err(format!(
            "service submission contract requires three Deferred submissions, got {a_type}/{b_type}/{c_type}"
        ));
    }
    let c_result = c
        .complete()
        .await
        .map_err(|error| report.record_backend_error("service burst C complete", &error))?
        .0;
    let b_result = b
        .complete()
        .await
        .map_err(|error| report.record_backend_error("service burst B complete", &error))?
        .0;
    let a_result = a
        .complete()
        .await
        .map_err(|error| report.record_backend_error("service burst A complete", &error))?
        .0;
    let held_c = capture(&c_result, a_case, hidden)?;
    let held_b = capture(&b_result, b_case, hidden)?;
    let held_a = capture(&a_result, a_case, hidden)?;
    compare("service burst C=A", &held_c, &base_a)?;
    compare("service burst B", &held_b, &base_b)?;
    compare("service burst A", &held_a, &base_a)?;
    report.phase(
        "warmheld",
        "ok",
        json!({
            "submission_type_A": a_type,
            "submission_type_B": b_type,
            "submission_type_C": c_type,
            "burst": ["A", "B", "C=A"],
            "public_submission": true,
            "service_submission_deferred": service_deferred,
            "completion_thread_independent": true,
            "not_claiming_unconsumed_gpu_frame_from_service_handle": true,
            "completed_order": ["C", "B", "A"],
            "A": held_a.json,
            "B": held_b.json,
            "C": held_c.json,
        }),
    );
    drop(c_result);
    drop(b_result);
    drop(a_result);
    backend
        .health()
        .await
        .map_err(|error| report.record_backend_error("service health held", &error))?;
    let recovered = capture(
        &backend
            .submit_embed(backend_batch(a_case)?)
            .await
            .map_err(|error| report.record_backend_error("service recovery submit", &error))?
            .complete()
            .await
            .map_err(|error| report.record_backend_error("service recovery complete", &error))?
            .0,
        a_case,
        hidden,
    )?;
    compare("service recovery A", &recovered, &base_a)?;
    report.phase(
        "recover",
        "ok",
        json!({"drops": 2, "health": true, "recovered_A": recovered.json}),
    );
    let a = backend
        .submit_embed(backend_batch(a_case)?)
        .await
        .map_err(|error| report.record_backend_error("service drop-A submit", &error))?;
    let b = backend
        .submit_embed(backend_batch(b_case)?)
        .await
        .map_err(|error| report.record_backend_error("service drop-B submit", &error))?;
    if !submission_deferred(&a) || !submission_deferred(&b) {
        return Err("service drop-A/B handles were not both Deferred".into());
    }
    drop(a);
    let b_result = b
        .complete()
        .await
        .map_err(|error| report.record_backend_error("service drop-A complete B", &error))?
        .0;
    let b_captured = capture(&b_result, b_case, hidden)?;
    compare("service drop-A B", &b_captured, &base_b)?;
    let retry_a = capture(
        &backend
            .submit_embed(backend_batch(a_case)?)
            .await
            .map_err(|error| report.record_backend_error("service drop-A retry A submit", &error))?
            .complete()
            .await
            .map_err(|error| {
                report.record_backend_error("service drop-A retry A complete", &error)
            })?
            .0,
        a_case,
        hidden,
    )?;
    compare("service drop-A retry A", &retry_a, &base_a)?;
    report.phase(
        "drop",
        "ok",
        json!({"dropped_A_while_B_handle_held": true, "B": b_captured.json, "retry_A": retry_a.json}),
    );
    let metadata_after = crate::metadata::collect(
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
        return Err("service metadata before/after identity changed".into());
    }
    report.phase(
        "metadata-after",
        "ok",
        json!({"equal": true, "payload_hashed": false}),
    );
    report.life = Some(json!({
        "kind": "service-life",
        "factory": "normal-service-factory",
        "baseline": {"A": base_a.json, "B": base_b.json},
        "all_values_are_completed_f32_words": true,
    }));
    report.submission_type = Some(a_type.to_string());
    Ok(())
}

fn core_capture(
    backend: &Qwen35MoeBackend,
    item: &LoadedCase,
    hidden: usize,
    report: &mut Report,
) -> Result<Captured, String> {
    let inference = backend
        .submit_embed(backend_batch(item)?)
        .map_err(|error| {
            report.record_backend_error(&format!("core/{}/submit", item.case.name), &error)
        })?;
    let embeddings = inference.complete().map_err(|error| {
        report.record_backend_error(&format!("core/{}/complete", item.case.name), &error)
    })?;
    capture(&embeddings, item, hidden)
}

fn core_infer(
    backend: &Qwen35MoeBackend,
    item: &LoadedCase,
    report: &mut Report,
) -> Result<Inference<Embeddings>, String> {
    backend.submit_embed(backend_batch(item)?).map_err(|error| {
        report.record_backend_error(&format!("core/{}/submit", item.case.name), &error)
    })
}

fn finish_inference(
    backend: &Qwen35MoeBackend,
    inference: Inference<Embeddings>,
    item: &LoadedCase,
    hidden: usize,
    report: &mut Report,
) -> Result<Captured, String> {
    let embeddings = inference.complete().map_err(|error| {
        report.record_backend_error(&format!("core/{}/complete", item.case.name), &error)
    })?;
    let _ = backend;
    capture(&embeddings, item, hidden)
}

pub async fn run_core(args: &Args, report: &mut Report) -> Result<(), String> {
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
    let a_case = find(&cases, "t7_a")?;
    let b_case = find(&cases, "t7_b")?;
    let t9_case = find(&cases, "t9")?;
    let metadata_before = crate::metadata::collect(
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
            "route": "exact-factory-selected-public-Qwen35MoeBackend::new",
            "adapter_lifecycle": true,
            "normal_service_factory_quality": false,
            "engine_count": 1,
        }),
    );
    let backend = Arc::new(
        Qwen35MoeBackend::new(model, "bfloat16", ModelType::Embedding(pool.clone()), 0)
            .map_err(|error| report.record_backend_error("core/constructor", &error))?,
    );
    report.engine_count = 1;
    report.normal_factory_route = false;
    if let Err(error) = backend.health() {
        return Err(report.record_backend_error("core/health-before", &error));
    }
    report.phase("constructor", "ok", json!({"engine_count": 1}));

    let base_a = core_capture(&backend, a_case, hidden, report)?;
    let base_b = core_capture(&backend, b_case, hidden, report)?;
    if base_a.entries == base_b.entries {
        return Err("core baseline A and B are not distinct".into());
    }
    report.phase(
        "baselines",
        "ok",
        json!({"A": base_a.json, "B": base_b.json, "distinct": true}),
    );

    let a = core_infer(&backend, a_case, report)?;
    let b = core_infer(&backend, b_case, report)?;
    if !core_deferred(&a) || !core_deferred(&b) {
        return Err("core A/B were not both genuinely unconsumed Inference::Deferred".into());
    }
    let c_batch = backend_batch(a_case)?;
    let c_backend = Arc::clone(&backend);
    let pending = spawn_blocking_submit(move || c_backend.submit_embed(c_batch));
    pending
        .started
        .recv_timeout(Duration::from_secs(2))
        .map_err(|_| "core third-held submit worker did not start".to_string())?;
    match pending.result.recv_timeout(Duration::from_millis(50)) {
        Err(RecvTimeoutError::Timeout) => {}
        Ok(Ok(_)) => return Err("core third submit returned while both A and B were held".into()),
        Ok(Err(error)) => {
            return Err(report.record_backend_error("core/third-held-submit-early", &error));
        }
        Err(RecvTimeoutError::Disconnected) => {
            return Err("core third submit worker exited without a result".into());
        }
    }
    let b_result = finish_inference(&backend, b, b_case, hidden, report)?;
    let c_inference = receive_bounded(&pending.result, Duration::from_secs(5))?
        .map_err(|error| report.record_backend_error("core/third-held-submit", &error))?;
    pending
        .handle
        .join()
        .map_err(|_| "core third submit worker panicked".to_string())?;
    if !core_deferred(&c_inference) {
        return Err("core third C=A was not Deferred after B completed".into());
    }
    let c_result = finish_inference(&backend, c_inference, a_case, hidden, report)?;
    let a_result = finish_inference(&backend, a, a_case, hidden, report)?;
    compare("core third C=A", &c_result, &base_a)?;
    compare("core reverse A", &a_result, &base_a)?;
    report.phase(
        "warmheld",
        "ok",
        json!({
            "A": "Deferred",
            "B": "Deferred",
            "third": "blocked-then-Deferred",
            "third_blocked_while_both_held_ms": 50,
            "completed_order": ["B", "C", "A"],
            "A_result": a_result.json,
            "B_result": b_result.json,
            "C_result": c_result.json,
            "unconsumed_deferred_asserted": true,
        }),
    );

    let a = core_infer(&backend, a_case, report)?;
    let b = core_infer(&backend, b_case, report)?;
    if !core_deferred(&a) || !core_deferred(&b) {
        return Err("core drop-A/B were not both Deferred".into());
    }
    drop(a);
    let c = core_infer(&backend, a_case, report)?;
    if !core_deferred(&c) {
        return Err("core C=A submitted while B held was not Deferred".into());
    }
    let c_result = finish_inference(&backend, c, a_case, hidden, report)?;
    let b_result = finish_inference(&backend, b, b_case, hidden, report)?;
    compare("core C=A while B held", &c_result, &base_a)?;
    compare("core B after C/A release", &b_result, &base_b)?;
    report.phase(
        "drop",
        "ok",
        json!({
            "dropped": "A",
            "C_submitted_before_B_completed": true,
            "C": c_result.json,
            "B": b_result.json,
        }),
    );

    let held = core_infer(&backend, a_case, report)?;
    let growth = core_infer(&backend, t9_case, report)?;
    if !core_deferred(&held) || !core_deferred(&growth) {
        return Err("core held T7/T9 growth were not both Deferred".into());
    }
    let t9_result = finish_inference(&backend, growth, t9_case, hidden, report)?;
    let held_result = finish_inference(&backend, held, a_case, hidden, report)?;
    compare("growth finish A", &held_result, &base_a)?;
    let t9_retry = core_capture(&backend, t9_case, hidden, report)?;
    compare("new shape T9 retry", &t9_retry, &t9_result)?;
    report.phase(
        "growth",
        "ok",
        json!({
            "from": "T7",
            "to": "T9",
            "held": "A",
            "common_slots": 2,
            "independent_work_keys": true,
            "growth_submission": "Deferred",
            "completed_order": ["T9", "A"],
            "A": held_result.json,
            "T9": t9_result.json,
            "T9_retry": t9_retry.json,
        }),
    );

    let mut invalid_results = Vec::new();
    let mut invalid_failures = Vec::new();
    for item in cases.iter().filter(|item| !item.case.valid) {
        let expected = item
            .case
            .expected_error_variant
            .as_deref()
            .ok_or_else(|| format!("{} lacks expected variant", item.case.name))?;
        match backend.submit_embed(backend_batch(item)?) {
            Err(error) if variant_matches(&error, expected) => {
                invalid_results.push(json!({
                    "name": item.case.name,
                    "expected": expected,
                    "actual": variant_name(&error),
                    "status": "ok",
                    "special": item.case.special,
                }));
            }
            Err(error) => {
                invalid_failures.push(report.record_backend_error(
                    &format!("core/{}/invalid-variant", item.case.name),
                    &error,
                ));
                invalid_results.push(json!({
                    "name": item.case.name,
                    "expected": expected,
                    "actual": variant_name(&error),
                    "status": "variant-mismatch",
                }));
            }
            Ok(_) => {
                invalid_failures.push(format!("{}: unexpectedly admitted", item.case.name));
                invalid_results.push(json!({
                    "name": item.case.name,
                    "expected": expected,
                    "actual": "admitted",
                    "status": "unexpected-ok",
                }));
            }
        }
    }
    report.phase(
        "invalid-preenqueue",
        if invalid_failures.is_empty() {
            "ok"
        } else {
            "error"
        },
        json!({"cases": invalid_results, "failures": invalid_failures}),
    );
    if !invalid_failures.is_empty() {
        return Err(invalid_failures.join("; "));
    }
    backend
        .health()
        .map_err(|error| report.record_backend_error("core health recover", &error))?;

    let a = core_infer(&backend, a_case, report)?;
    let b = core_infer(&backend, b_case, report)?;
    if !core_deferred(&a) || !core_deferred(&b) {
        return Err("final modeldrop A/B were not both Deferred".into());
    }
    drop(backend);
    drop(a);
    let b_result = finish_inference_detached(b, b_case, hidden, report)?;
    compare("final modeldrop B", &b_result, &base_b)?;
    report.phase(
        "modeldrop",
        "ok",
        json!({
            "adapter_dropped_with_A_B_deferred": true,
            "A_dropped_unconsumed": true,
            "B_completed_after_adapter_drop": true,
            "B": b_result.json,
            "no_reload_after_final_drop": true,
        }),
    );
    let metadata_after = crate::metadata::collect(
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
        return Err("core metadata before/after identity changed".into());
    }
    report.phase(
        "metadata-after",
        "ok",
        json!({"equal": true, "payload_hashed": false}),
    );

    report.life = Some(json!({
        "kind": "core-life",
        "adapter_lifecycle": true,
        "normal_service_factory_quality": false,
        "baseline": {"A": base_a.json, "B": base_b.json},
        "invalid_cases": invalid_results,
        "all_values_are_completed_f32_words": true,
    }));
    Ok(())
}

fn finish_inference_detached(
    inference: Inference<Embeddings>,
    item: &LoadedCase,
    hidden: usize,
    report: &mut Report,
) -> Result<Captured, String> {
    let embeddings = inference.complete().map_err(|error| {
        report.record_backend_error(
            &format!("core/{}/detached-complete", item.case.name),
            &error,
        )
    })?;
    capture(&embeddings, item, hidden)
}

#[cfg(test)]
mod tests {
    use super::spawn_blocking_submit;
    use infinity_v3_backend_core::BackendError;
    use std::sync::mpsc;
    use std::time::Duration;

    #[test]
    fn blocking_submit_worker_stays_blocked_then_wakes() {
        let (gate_sender, gate_receiver) = mpsc::channel::<()>();
        let worker = spawn_blocking_submit(move || {
            gate_receiver
                .recv()
                .map_err(|_| BackendError::Unhealthy)?;
            Ok(7u32)
        });
        worker
            .started
            .recv_timeout(Duration::from_secs(2))
            .unwrap();
        assert!(worker
            .result
            .recv_timeout(Duration::from_millis(50))
            .is_err());
        gate_sender.send(()).unwrap();
        let value = worker
            .result
            .recv_timeout(Duration::from_secs(5))
            .unwrap()
            .unwrap();
        assert_eq!(value, 7);
        worker.handle.join().unwrap();
    }
}
