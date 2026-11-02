use infinity_v3_backend_core::result_memory::ResultPool;
use infinity_v3_backend_core::BackendError;
use infinity_v3_backend_core::{Embedding, Embeddings};
use q35moe_unified_runtime::batch::{
    canonical_expected, load_batch, physical_layout, to_backend_batch,
};
use q35moe_unified_runtime::cli::{parse, Mode};
use q35moe_unified_runtime::fixtures::{
    backend_batch, load_cases, GENERATED_SHA256_PENDING,
};
use q35moe_unified_runtime::outputs::canonicalize;
use q35moe_unified_runtime::report::{output_json, Report};
use std::path::PathBuf;
use std::process::Command;

fn manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures/manifest.json")
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../fixtures")
        .join(name)
}

#[test]
fn strict_cli_rejects_duplicate_unknown_and_missing_pool() {
    assert!(parse(&[
        "bin".into(),
        "quality".into(),
        "--pool".into(),
        "cls".into(),
        "--pool".into(),
        "mean".into(),
        "--model".into(),
        "x".into(),
        "--cases".into(),
        "c".into()
    ])
    .is_err());
    assert!(parse(&[
        "bin".into(),
        "metadata".into(),
        "--bogus".into(),
        "x".into()
    ])
    .is_err());
    assert!(parse(&[
        "bin".into(),
        "quality".into(),
        "--model".into(),
        "x".into(),
        "--cases".into(),
        "c".into()
    ])
    .is_err());
    assert!(parse(&[
        "bin".into(),
        "validate-only".into(),
        "--limit".into(),
        "0".into()
    ])
    .is_err());
}

#[test]
fn validate_only_mode_and_flag_need_no_model() {
    let a = parse(&[
        "bin".into(),
        "validate-only".into(),
        "--cases".into(),
        "c".into(),
    ])
    .unwrap();
    assert_eq!(a.mode, Mode::ValidateOnly);
    assert!(a.validate_only);
    assert!(a.model.is_none());
    let b = parse(&[
        "bin".into(),
        "metadata".into(),
        "--model".into(),
        "m".into(),
        "--cases".into(),
        "c".into(),
        "--validate-only".into(),
    ])
    .unwrap();
    assert_eq!(b.mode, Mode::ValidateOnly);
}

#[test]
fn canonical_fixtures_load_and_invalid_specials_are_typed() {
    let cases = load_cases(&manifest(), None).unwrap();
    assert_eq!(cases.len(), 25);
    assert_eq!(cases[0].case.name, "t7_a");
    let valid: Vec<_> = cases.iter().filter(|case| case.case.valid).collect();
    let invalid: Vec<_> = cases.iter().filter(|case| !case.case.valid).collect();
    assert_eq!(valid.len(), 15);
    assert_eq!(invalid.len(), 10);
    for item in &invalid {
        assert!(item.case.expected_error_variant.is_some());
    }
    // Invalid requests must reach the actual backend rather than fail in the harness.
    for item in &cases {
        backend_batch(item)
            .unwrap_or_else(|e| panic!("fixture {} cannot reach backend: {e}", item.case.name));
    }
    let media = cases
        .iter()
        .find(|case| case.case.special.as_deref() == Some("media"))
        .unwrap();
    assert!(backend_batch(media).unwrap().multimodal[0].is_some());
    let compact = cases
        .iter()
        .find(|case| case.case.special.as_deref() == Some("compact"))
        .unwrap();
    assert!(backend_batch(compact).unwrap().compact_input_ids.is_some());

    for name in ["duplicate_t7", "duplicate_raw_t7"] {
        let item = cases.iter().find(|case| case.case.name == name).unwrap();
        assert!(item.case.valid);
        assert!(item.case.expected_error_variant.is_none());
        assert!(item.case.special.is_none());
    }
    for name in [
        "duplicate_pooled_t7",
        "bounded_duplicate_ragged",
        "rawoverride_t7",
    ] {
        let item = cases.iter().find(|case| case.case.name == name).unwrap();
        assert!(item.case.valid);
    }
    let no_selection = cases
        .iter()
        .find(|case| case.case.name == "t7_no_selection")
        .unwrap();
    assert!(no_selection.case.valid);
    assert!(no_selection.batch.pooled_indices.is_empty());
    assert!(no_selection.batch.raw_indices.is_empty());
    for name in [
        "invalid_token_nosel",
        "invalid_media_nosel",
        "invalid_compact_nosel",
    ] {
        let item = cases.iter().find(|case| case.case.name == name).unwrap();
        assert!(!item.case.valid);
        assert!(item.batch.pooled_indices.is_empty());
        assert!(item.batch.raw_indices.is_empty());
        assert!(item.case.expected_error_variant.is_some());
    }
    let empty = cases
        .iter()
        .find(|case| case.case.special.as_deref() == Some("empty"))
        .unwrap();
    assert!(to_backend_batch(&empty.batch).input_ids.is_empty());
}

#[test]
fn t9_fixture_is_exact_synthetic_nine_token_batch() {
    let cases = load_cases(&manifest(), None).unwrap();
    let item = cases.iter().find(|case| case.case.name == "t9").unwrap();
    assert!(item.case.valid);
    assert_eq!(
        item.batch.input_ids,
        (0..9u32).map(|index| 20 + 13 * index).collect::<Vec<_>>()
    );
    assert_eq!(
        item.batch.position_ids,
        (0..9u32).collect::<Vec<_>>()
    );
    assert_eq!(item.batch.token_type_ids, vec![0; 9]);
    assert_eq!(item.batch.cumulative_seq_lengths, vec![0, 9]);
    assert_eq!(item.batch.max_length, 9);
    assert_eq!(item.batch.pooled_indices, vec![0]);
    assert!(item.batch.raw_indices.is_empty());
    assert_eq!(item.file_sha256.len(), 64);
    assert!(item.case.sha256 == GENERATED_SHA256_PENDING || item.case.sha256.len() == 64);
}

#[test]
fn duplicate_and_raw_override_fixtures_are_valid_without_capacity_guard() {
    let cases = load_cases(&manifest(), None).unwrap();
    let hidden = 8usize;
    for name in ["duplicate_t7", "duplicate_raw_t7"] {
        let item = cases.iter().find(|case| case.case.name == name).unwrap();
        assert!(item.case.valid);
        assert!(item.case.expected_error_variant.is_none());
        assert!(item.case.special.is_none());
        let layout = physical_layout(&item.batch, hidden, hidden).unwrap();
        let (_expected, canonical) = canonical_expected(&layout).unwrap();
        assert!(canonical.duplicate_selection_indices.contains(&0));
        assert!(canonical.raw_overrides_pooled.contains(&0));
    }
}

#[test]
fn canonical_serialization_contains_f32_values_and_bits_for_bounded_duplicates() {
    let (_, batch) = load_batch(&fixture("bounded_duplicate_ragged.json")).unwrap();
    let hidden = 8usize;
    let layout = physical_layout(&batch, hidden, hidden).unwrap();
    let (expected, canonical) = canonical_expected(&layout).unwrap();
    let mut result = Embeddings::default();
    result.insert(0, Embedding::All(vec![vec![0.25; 8]]));
    let canonical = canonicalize(&result, &expected, &canonical, hidden, hidden).unwrap();
    let json = output_json(canonical);
    let outputs = json["outputs"].as_array().unwrap();
    assert_eq!(outputs.len(), 1);
    for output in outputs {
        let values = output["values"].as_array().unwrap();
        let bits = output["bits_u32"].as_array().unwrap();
        assert_eq!(values.len(), bits.len());
        for (value, bits) in values.iter().zip(bits.iter()) {
            let value = value.as_f64().unwrap() as f32;
            assert_eq!(value.to_bits() as u64, bits.as_u64().unwrap());
        }
    }
    assert!(json["facts"]["raw_overrides_pooled"]
        .as_array()
        .unwrap()
        .is_empty());
    assert_eq!(
        json["facts"]["duplicate_selection_indices"],
        serde_json::json!([0])
    );
}

#[test]
fn canonical_raw_override_identity_is_retained() {
    let (_, batch) = load_batch(&fixture("rawoverride_t7.json")).unwrap();
    let hidden = 8usize;
    let layout = physical_layout(&batch, hidden, hidden).unwrap();
    let (expected, canonical) = canonical_expected(&layout).unwrap();
    let mut result = Embeddings::default();
    result.insert(
        0,
        Embedding::All((0..7).map(|row| vec![row as f32 + 0.5; 8]).collect()),
    );
    let canonical = canonicalize(&result, &expected, &canonical, hidden, hidden).unwrap();
    assert_eq!(canonical.outputs.len(), 1);
    assert_eq!(canonical.outputs[0].kind, "raw");
    assert_eq!(canonical.outputs[0].shape, vec![7, hidden]);
    assert_eq!(canonical.facts.raw_overrides_pooled, vec![0]);
}

#[test]
fn empty_selection_is_semantic_only_not_numeric_vacuous() {
    let (_, mut batch) = load_batch(&fixture("t7_a.json")).unwrap();
    batch.pooled_indices.clear();
    batch.raw_indices.clear();
    let hidden = 8usize;
    let layout = physical_layout(&batch, hidden, hidden).unwrap();
    let (expected, canonical) = canonical_expected(&layout).unwrap();
    let result = Embeddings::default();
    let canonical = canonicalize(&result, &expected, &canonical, hidden, hidden).unwrap();
    assert_eq!(canonical.facts.entry_count, 0);
    assert_eq!(canonical.facts.float_count, 0);
}

#[test]
fn actual_result_pool_exhausts_types_and_recovers_after_drop() {
    let pool = ResultPool::new(64).unwrap();
    let lease = pool.reserve(64).unwrap();
    let clone = lease.clone();
    assert!(matches!(
        pool.reserve(1),
        Err(infinity_v3_backend_core::BackendError::Overloaded(_))
    ));
    drop(lease);
    assert_eq!(pool.used(), 64);
    drop(clone);
    assert_eq!(pool.used(), 0);
    let recovered = pool.reserve(64).unwrap();
    drop(recovered);
    assert_eq!(pool.used(), 0);
}

#[test]
fn dev_binary_validate_only_parses_actual_cli_without_gpu_or_engine() {
    let report =
        std::env::temp_dir().join(format!("q35moe-validate-only-{}.json", std::process::id()));
    let output = Command::new(env!("CARGO_BIN_EXE_q35moe-unified-runtime"))
        .arg("validate-only")
        .arg("--cases")
        .arg(manifest())
        .arg("--report")
        .arg(&report)
        .env("CUDA_VISIBLE_DEVICES", "")
        .env("CARGO_NET_OFFLINE", "true")
        .output()
        .unwrap();
    assert!(output.status.success(), "{:?}", output);
    let data: serde_json::Value = serde_json::from_slice(&std::fs::read(&report).unwrap()).unwrap();
    assert_eq!(data["mode"], "validate-only");
    assert_eq!(data["engine_count"], 0);
    assert_eq!(data["normal_factory_route"], false);
    assert_eq!(data["phases"][0]["detail"]["gpu_path_taken"], false);
    assert_eq!(data["phases"][0]["detail"]["model_loaded"], false);
    let _ = std::fs::remove_file(report);
}

#[test]
fn dev_binary_metadata_reads_headers_only_without_gpu() {
    let report = std::env::temp_dir().join(format!("q35moe-metadata-{}.json", std::process::id()));
    let model = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures/metadata-model");
    let output = Command::new(env!("CARGO_BIN_EXE_q35moe-unified-runtime"))
        .arg("metadata")
        .arg("--model")
        .arg(&model)
        .arg("--cases")
        .arg(manifest())
        .arg("--report")
        .arg(&report)
        .env("CUDA_VISIBLE_DEVICES", "")
        .env("CARGO_NET_OFFLINE", "true")
        .output()
        .unwrap();
    assert!(output.status.success(), "{:?}", output);
    let data: serde_json::Value = serde_json::from_slice(&std::fs::read(&report).unwrap()).unwrap();
    assert_eq!(data["mode"], "metadata");
    assert_eq!(data["engine_count"], 0);
    assert_eq!(data["metadata"]["engine_loaded"], false);
    assert_eq!(data["metadata"]["context_created"], false);
    assert_eq!(data["metadata"]["tensor_payload_read_bytes"], 0);
    assert_eq!(
        data["metadata"]["safetensors_headers"][0]["payload_read_bytes"],
        0
    );
    let _ = std::fs::remove_file(report);
}

#[test]
fn typed_unhealthy_envelope_is_not_runtime_fatal_qualification() {
    let mut report = Report::new("cpu-envelope");
    let message = report.record_backend_error("cpu/envelope", &BackendError::Unhealthy);
    assert!(message.contains("Unhealthy"));
    assert_eq!(report.typed_errors.len(), 1);
    assert_eq!(report.typed_errors[0]["variant"], "Unhealthy");
    assert_eq!(
        report.typed_errors[0]["runtime_failed_fence_qualification"],
        false
    );
    let json = serde_json::to_value(&report).unwrap();
    assert_eq!(json["typed_errors"][0]["stage"], "cpu/envelope");
}

#[test]
fn metadata_before_after_surface_has_six_exact_envs_and_order_identity() {
    let model = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures/metadata-model");
    let before = q35moe_unified_runtime::metadata::collect(
        &model,
        q35moe_unified_runtime::report::PRODUCT_SOURCE,
        q35moe_unified_runtime::report::PRODUCT_FREEZE_SHA256,
        q35moe_unified_runtime::report::CANONICAL31_PROVIDER,
    )
    .unwrap();
    let after = q35moe_unified_runtime::metadata::collect(
        &model,
        q35moe_unified_runtime::report::PRODUCT_SOURCE,
        q35moe_unified_runtime::report::PRODUCT_FREEZE_SHA256,
        q35moe_unified_runtime::report::CANONICAL31_PROVIDER,
    )
    .unwrap();
    assert_eq!(before, after);
    assert_eq!(before["config"]["q_heads"].as_u64(), None);
    assert!(before["config"]["layer_order_sha256"].is_string());
    let envs = before["provider"]["envs"].as_object().unwrap();
    assert_eq!(envs.len(), 6);
    for name in q35moe_unified_runtime::metadata::PROVIDER_ENVS {
        assert!(envs.contains_key(name));
    }
    assert_eq!(before["tensor_payload_read_bytes"], 0);
}
