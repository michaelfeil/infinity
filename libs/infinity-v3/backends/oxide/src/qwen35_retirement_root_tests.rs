// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Root-only witness that same-shape cache pressure retires a held Qwen3.5
//! owner without draining the stream, then releases it by exact completion
//! proof. Finite lifetime/accounting only; no quality, admission or perf claim.

use super::*;
use std::path::{Path, PathBuf};

const MIB: usize = 1024 * 1024;
const PREFIX: usize = 3;

fn source(name: &str) -> PathBuf {
    let path = PathBuf::from(
        std::env::var_os(name).unwrap_or_else(|| panic!("missing explicit source {name}")),
    );
    assert!(!path.as_os_str().is_empty(), "empty source {name}");
    assert!(path.exists(), "missing source {name}: {}", path.display());
    path
}

fn options() -> Qwen35TextOptions {
    Qwen35TextOptions {
        causal: false,
        fa4_root: source("QWEN35_CONTROL_FA4_ROOT"),
        gdn_raw_kernels: source("QWEN35_CONTROL_GDN_RAW_KERNELS"),
        gdn_provider: source("QWEN35_CONTROL_GDN_PROVIDER"),
        gdn_aux_kernels: source("QWEN35_CONTROL_GDN_AUX_KERNELS"),
    }
}

fn u32_array(object: &serde_json::Map<String, serde_json::Value>, key: &str) -> Vec<u32> {
    object
        .get(key)
        .and_then(serde_json::Value::as_array)
        .unwrap_or_else(|| panic!("fixture case needs {key}"))
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .unwrap_or_else(|| panic!("fixture {key} must be exact u32"))
        })
        .collect()
}

fn fixture_rows(path: &Path) -> (Vec<u32>, Vec<u32>) {
    let root: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    assert_eq!(
        root.get("schema").and_then(serde_json::Value::as_str),
        Some("infinity-onejev-physical-cases/v1")
    );
    let first = root
        .get("cases")
        .and_then(serde_json::Value::as_array)
        .and_then(|cases| cases.first())
        .and_then(serde_json::Value::as_object)
        .expect("fixture needs at least one case object");
    let ids = u32_array(first, "input_ids");
    assert!(ids.len() >= PREFIX);
    let rows = first
        .get("token_ids")
        .and_then(serde_json::Value::as_array)
        .map(|_| u32_array(first, "token_ids"))
        .filter(|rows| !rows.is_empty())
        .unwrap_or_else(|| ids.iter().copied().take(PREFIX).collect());
    (ids, rows)
}

fn synthetic(prefix_ids: &[u32], tokens: usize) -> Batch {
    assert!(!prefix_ids.is_empty());
    let input_ids = (0..tokens)
        .map(|index| prefix_ids[index % prefix_ids.len()])
        .collect::<Vec<_>>();
    Batch {
        input_ids,
        position_ids: (0..tokens as u32).collect(),
        token_type_ids: vec![0; tokens],
        cumulative_seq_lengths: vec![0, tokens as u32],
        max_length: tokens as u32,
        pooled_indices: Vec::new(),
        raw_indices: Vec::new(),
        multimodal: Vec::new(),
        compact_input_ids: None,
        compact_position_ids: None,
        scatter_unfold: None,
        fold_gather: None,
        tokens: Vec::new(),
        offsets: Vec::new(),
    }
}

fn request(rows: &[u32]) -> DecisionInput {
    DecisionInput::BlockOptionTokens {
        token_ids: rows.to_vec(),
        prefix_length: PREFIX,
    }
}

fn bits(outputs: Vec<DecisionOutput>) -> Vec<Vec<u32>> {
    outputs
        .into_iter()
        .map(|output| {
            assert!(!output.logits.is_empty());
            assert!(output.logits.iter().all(|value| value.is_finite()));
            output.logits.into_iter().map(f32::to_bits).collect()
        })
        .collect()
}

fn submit_bits(backend: &OxideBackend, batch: Batch, request: DecisionInput) -> Vec<Vec<u32>> {
    bits(
        backend
            .submit_qwen35_block_decision_candidate(batch, vec![request])
            .unwrap()
            .complete()
            .unwrap(),
    )
}

#[test]
#[ignore = "Requires CUDA Qwen3.5 kernels/FA4, QWEN35_BLOCK_MODEL, QWEN35_CASES and QWEN35_CONTROL_* sources"]
fn same_shape_submit_under_cache_pressure_retires_held_owner_without_drain() {
    let model = source("QWEN35_BLOCK_MODEL");
    let cases = source("QWEN35_CASES");
    let (fixture, rows) = fixture_rows(&cases);
    let prefix_ids = &fixture[..PREFIX];
    let backend =
        OxideBackend::new_qwen35_dynamic_block_decision_candidate(&model, 0, options()).unwrap();
    let a = synthetic(prefix_ids, 7);
    let b = synthetic(prefix_ids, 128);
    let req_a = request(&rows);
    let req_b = request(&rows);
    let base_a = submit_bits(&backend, a.clone(), req_a.clone());

    // A failed pre-upload request can leave a constructed but never-submitted
    // arena. Its constructor fence is sufficient; it must not consume a
    // permanent retirement slot or invoke the default stream drain.
    {
        let mut engine = backend.engine.lock().unwrap();
        let active = engine
            .workspace
            .as_ref()
            .expect("baseline A workspace")
            .key();
        let cold_key = (active.0 * 2, active.1, active.2);
        let stream = engine.stream.clone();
        let live_before = crate::LIVE_ARENA_BYTES.load(Ordering::Relaxed);
        let cold = engine.model.workspace(stream, cold_key).unwrap();
        let Workspace::Qwen35(work) = &cold else {
            panic!("cold Qwen3.5 workspace expected")
        };
        assert!(work.completion_ticket().is_none());
        let drains = qwen35_work_default_drop_drains();
        engine.retire_workspace(cold).unwrap();
        assert_eq!(engine.retired.len(), 0);
        assert_eq!(engine.retired.bytes(), 0);
        assert_eq!(qwen35_work_default_drop_drains(), drains);
        assert_eq!(crate::LIVE_ARENA_BYTES.load(Ordering::Relaxed), live_before);
    }

    // Baseline A leaves A active. Hold A then B, completing only B: B evicts the
    // prior active A into the cache while A's deferred host lease is still held.
    let held_a = backend
        .submit_qwen35_block_decision_candidate(a.clone(), vec![req_a.clone()])
        .unwrap();
    assert!(matches!(&held_a, Inference::Deferred(_)));
    let held_b = backend
        .submit_qwen35_block_decision_candidate(b.clone(), vec![req_b.clone()])
        .unwrap();
    assert!(matches!(&held_b, Inference::Deferred(_)));
    let base_b = bits(held_b.complete().unwrap());
    assert!(matches!(&held_a, Inference::Deferred(_)));

    // Move the existing cache entries under the engine mutex into a forced small
    // cache, keeping A with zero reservation and without dropping its owner.
    let a_bytes = {
        let mut engine = backend.engine.lock().unwrap();
        let entries = engine.cached.drain();
        let active_bytes = engine
            .workspace
            .as_ref()
            .expect("active B workspace")
            .bytes();
        let mut small = crate::cache::WorkspaceCache::with_limits(7, active_bytes + MIB);
        let mut cached_a = None;
        for (key, workspace) in entries {
            if cached_a.is_none() && key.0 == 32 {
                cached_a = Some((key, workspace));
            } else {
                engine.retire_workspace(workspace).unwrap();
            }
        }
        let (key, workspace) = cached_a.expect("prior A must remain cached");
        let bytes = workspace.bytes();
        assert!(bytes < small.byte_budget());
        assert!(small.put(key, workspace, bytes).is_empty());
        engine.cached = small;
        bytes
    };

    let drop_before = qwen35_work_default_drop_drains();
    let lease_fallback_before = crate::staging::tracked_lease_fallback_drains();
    let proof_before = qwen35_retired_proof_releases();
    let drain_before = qwen35_retired_drain_releases();

    // Same-shape B needs the active bytes plus the retired A bytes, which no
    // longer fit the forced budget. Admission must be typed and non-fatal.
    let error = match backend.submit_qwen35_block_decision_candidate(b.clone(), vec![req_b.clone()])
    {
        Ok(_) => panic!("pressure request was admitted under the forced budget"),
        Err(error) => error,
    };
    assert!(matches!(error, BackendError::Overloaded(_)), "{error:?}");
    {
        let engine = backend.engine.lock().unwrap();
        assert_eq!(engine.retired.len(), 1);
        assert_eq!(engine.retired.bytes(), a_bytes);
        let active_bytes = engine
            .workspace
            .as_ref()
            .expect("active B workspace")
            .bytes();
        assert!(active_bytes + engine.retired.bytes() > engine.cached.byte_budget());
        assert_eq!(qwen35_retired_proof_releases(), proof_before);
        assert_eq!(qwen35_retired_drain_releases(), drain_before);
    }
    assert_eq!(qwen35_work_default_drop_drains(), drop_before);
    assert_eq!(
        crate::staging::tracked_lease_fallback_drains(),
        lease_fallback_before
    );

    // Completing A publishes its exact ticket. A later same-shape B collects
    // that proof, releases A's payload without a stream drain, and succeeds.
    assert_eq!(bits(held_a.complete().unwrap()), base_a);
    assert_eq!(submit_bits(&backend, b.clone(), req_b.clone()), base_b);
    assert_eq!(qwen35_retired_proof_releases(), proof_before + 1);
    assert_eq!(qwen35_retired_drain_releases(), drain_before);
    assert_eq!(qwen35_work_default_drop_drains(), drop_before);
    assert_eq!(
        crate::staging::tracked_lease_fallback_drains(),
        lease_fallback_before
    );
    {
        let engine = backend.engine.lock().unwrap();
        assert_eq!(engine.retired.len(), 0);
        assert_eq!(engine.retired.bytes(), 0);
    }

    assert_eq!(submit_bits(&backend, b, req_b), base_b);
    backend.health().unwrap();
}

#[test]
#[ignore = "Requires CUDA Qwen3.5 kernels/FA4, QWEN35_BLOCK_MODEL, QWEN35_CASES and QWEN35_CONTROL_* sources"]
fn dynamic_prefix_held_results_survive_same_shape_and_growth() {
    eprintln!("phase:dynamic-held constructor");
    let model = source("QWEN35_BLOCK_MODEL");
    let (fixture, rows) = fixture_rows(&source("QWEN35_CASES"));
    let backend =
        OxideBackend::new_qwen35_dynamic_block_decision_candidate(&model, 0, options()).unwrap();
    let clean = || {
        (
            qwen35_work_default_drop_drains(),
            crate::staging::tracked_lease_fallback_drains(),
            qwen35_retired_drain_releases(),
        )
    };
    let baseline = clean();

    eprintln!("phase:dynamic-held baselines");
    let a = synthetic(&fixture, 7);
    let c = synthetic(&fixture, 7);
    let b = synthetic(&fixture, 129);
    let req_a = request(&rows);
    let req_c = DecisionInput::BlockOptionTokens {
        token_ids: rows.clone(),
        prefix_length: 4,
    };
    let mut lm_b = rows.clone();
    lm_b.reverse();
    let req_b = DecisionInput::BlockOptionTokens {
        token_ids: lm_b,
        prefix_length: 5,
    };
    let base_a = submit_bits(&backend, a.clone(), req_a.clone());
    let base_c = submit_bits(&backend, c.clone(), req_c.clone());
    let base_b = submit_bits(&backend, b.clone(), req_b.clone());
    assert_eq!(backend.io.active_leases(), 0);
    assert_eq!(clean(), baseline);

    eprintln!("phase:dynamic-held held-shape");
    let held_a = backend
        .submit_qwen35_block_decision_candidate(a.clone(), vec![req_a.clone()])
        .unwrap();
    let held_b = backend
        .submit_qwen35_block_decision_candidate(b.clone(), vec![req_b.clone()])
        .unwrap();
    assert!(matches!(&held_a, Inference::Deferred(_)));
    assert!(matches!(&held_b, Inference::Deferred(_)));
    assert_eq!(backend.io.active_leases(), 2);
    assert_eq!(clean(), baseline);
    assert_eq!(bits(held_b.complete().unwrap()), base_b);
    assert_eq!(bits(held_a.complete().unwrap()), base_a);
    assert_eq!(backend.io.active_leases(), 0);
    assert_eq!(clean(), baseline);

    eprintln!("phase:dynamic-held held-sameT");
    let held_c = backend
        .submit_qwen35_block_decision_candidate(c.clone(), vec![req_c.clone()])
        .unwrap();
    let arena_c = {
        let engine = backend.engine.lock().unwrap();
        let work = engine.workspace.as_ref().expect("C arena");
        (work.key(), work.bytes())
    };
    let held_a = backend
        .submit_qwen35_block_decision_candidate(a.clone(), vec![req_a.clone()])
        .unwrap();
    let arena_a = {
        let engine = backend.engine.lock().unwrap();
        let work = engine.workspace.as_ref().expect("A arena");
        (work.key(), work.bytes())
    };
    assert!(matches!(&held_c, Inference::Deferred(_)));
    assert!(matches!(&held_a, Inference::Deferred(_)));
    assert_eq!(arena_a, arena_c, "same T must retain the same arena");
    assert_eq!(backend.io.active_leases(), 2);
    assert_eq!(clean(), baseline);
    assert_eq!(bits(held_a.complete().unwrap()), base_a);
    assert_eq!(bits(held_c.complete().unwrap()), base_c);
    assert_eq!(backend.io.active_leases(), 0);
    assert_eq!(clean(), baseline);

    eprintln!("phase:dynamic-held drop");
    let dropped = backend
        .submit_qwen35_block_decision_candidate(b, vec![req_b])
        .unwrap();
    assert!(matches!(&dropped, Inference::Deferred(_)));
    assert_eq!(backend.io.active_leases(), 1);
    drop(dropped);
    assert_eq!(backend.io.active_leases(), 0);
    assert_eq!(clean(), baseline);

    eprintln!("phase:dynamic-held reject");
    let invalid = |prefix_length| match backend.submit_qwen35_block_decision_candidate(
        a.clone(),
        vec![DecisionInput::BlockOptionTokens {
            token_ids: rows.clone(),
            prefix_length,
        }],
    ) {
        Ok(_) => panic!("invalid prefix {prefix_length} was admitted"),
        Err(error) => error,
    };
    assert!(matches!(invalid(0), BackendError::Inference(_)));
    assert!(matches!(invalid(8), BackendError::Inference(_)));
    assert_eq!(backend.io.active_leases(), 0);
    assert_eq!(clean(), baseline);

    eprintln!("phase:dynamic-held recovery");
    assert_eq!(submit_bits(&backend, a, request(&rows)), base_a);
    assert_eq!(clean(), baseline);

    eprintln!("phase:dynamic-held healthy");
    backend.health().unwrap();
    assert_eq!(clean(), baseline);
}
