// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Root-only witnesses for submit-path owner retirement.
//!
//! These tests use actual cached qualified checkpoints. A separate backend
//! instance first produces the exact paired baseline bits, then a fresh engine
//! exercises held submissions, growth and cache pressure. Every held result is
//! compared bit-for-bit with that baseline. No quality, admission or
//! performance claim is made. Every test is ignored unless Root explicitly
//! supplies the model variables and a CUDA device.

use super::*;
use std::path::PathBuf;

type BitRows = Vec<(usize, Vec<u32>)>;

fn source(name: &str) -> PathBuf {
    let path = PathBuf::from(
        std::env::var_os(name).unwrap_or_else(|| panic!("missing explicit source {name}")),
    );
    assert!(!path.as_os_str().is_empty(), "empty source {name}");
    assert!(path.exists(), "missing source {name}: {}", path.display());
    path
}

fn load(name: &str, model_type: ModelType) -> OxideBackend {
    OxideBackend::new(&source(name), "bfloat16", model_type, 0)
        .unwrap_or_else(|error| panic!("failed to load {name}: {error}"))
}

fn synthetic(prefix: &[u32], tokens: usize, sequences: usize) -> Batch {
    assert!(!prefix.is_empty());
    assert!(tokens >= sequences && sequences > 0);
    let base = tokens / sequences;
    let extra = tokens % sequences;
    let mut lengths = Vec::with_capacity(sequences);
    let mut input_ids = Vec::with_capacity(tokens);
    let mut position_ids = Vec::with_capacity(tokens);
    for sequence in 0..sequences {
        let length = base + usize::from(sequence < extra);
        for position in 0..length {
            input_ids.push(prefix[input_ids.len() % prefix.len()]);
            position_ids.push(position as u32);
        }
        lengths.push(length);
    }
    let mut cumulative_seq_lengths = vec![0u32];
    let mut total = 0u32;
    for length in lengths {
        total = total
            .checked_add(u32::try_from(length).unwrap())
            .expect("synthetic token count overflow");
        cumulative_seq_lengths.push(total);
    }
    let pooled_indices = (0..u32::try_from(sequences).unwrap()).collect();
    let max_length = cumulative_seq_lengths
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .max()
        .unwrap();
    Batch {
        input_ids,
        position_ids,
        token_type_ids: vec![0; tokens],
        cumulative_seq_lengths,
        max_length,
        pooled_indices,
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

fn embedding_bits(result: Embeddings, width: usize) -> BitRows {
    let mut rows = result
        .into_iter()
        .map(|(index, embedding)| {
            let Embedding::Pooled(row) = embedding else {
                panic!("expected pooled embedding");
            };
            assert_eq!(row.len(), width);
            (index, row.into_iter().map(f32::to_bits).collect())
        })
        .collect::<Vec<_>>();
    rows.sort_by_key(|(index, _)| *index);
    rows
}

fn prediction_bits(result: Predictions) -> BitRows {
    let mut rows = result
        .into_iter()
        .map(|(index, row)| {
            assert!(!row.is_empty());
            (index, row.into_iter().map(f32::to_bits).collect())
        })
        .collect::<Vec<_>>();
    rows.sort_by_key(|(index, _)| *index);
    rows
}

fn clean() -> (usize, usize, usize) {
    (
        default_drop_drains(),
        retired_proof_releases(),
        retired_drain_releases(),
    )
}

fn assert_deferred<T>(value: &Inference<T>) {
    assert!(
        matches!(value, Inference::Deferred(_)),
        "held request must complete through its own lease"
    );
}

fn expect_overloaded<T>(result: Result<Inference<T>, BackendError>) {
    match result {
        Ok(_) => panic!("pressure request was admitted under the forced budget"),
        Err(error) => assert!(matches!(error, BackendError::Overloaded(_)), "{error:?}"),
    }
}

fn assert_no_submit_drain(baseline: (usize, usize, usize)) {
    assert_eq!(default_drop_drains(), baseline.0);
    assert_eq!(retired_drain_releases(), baseline.2);
}

/// Keep the cached owner for `keep` and rebuild a cache whose budget is exactly
/// `active + kept - 1`. The next same-shape request must evict that owner, fail
/// typed admission while it is unproven, then release it by exact proof.
fn force_cached_pressure(backend: &OxideBackend, keep: WorkKey) -> usize {
    let mut engine = backend.engine.lock().unwrap();
    let active_bytes = engine.workspace.as_ref().expect("active workspace").bytes();
    let entries = engine.cached.drain();
    let mut kept = None;
    for (key, workspace) in entries {
        if kept.is_none() && key == keep {
            kept = Some((key, workspace));
        } else {
            engine.retire_workspace(workspace).unwrap();
        }
    }
    let (key, workspace) = kept.expect("prior owner must remain cached");
    let bytes = workspace.bytes();
    let budget = active_bytes
        .checked_add(bytes)
        .and_then(|total| total.checked_sub(1))
        .expect("cache pressure budget overflow");
    let mut small = cache::WorkspaceCache::with_limits(7, budget);
    assert!(small.put(key, workspace, bytes).is_empty());
    engine.cached = small;
    bytes
}

fn baselines(name: &str, model_type: ModelType, a: &Batch, b: &Batch) -> (usize, BitRows, BitRows) {
    let gold = load(name, model_type);
    let width = gold.output_width;
    let base_a = embedding_bits(gold.embed(a.clone()).unwrap(), width);
    let base_b = embedding_bits(gold.embed(b.clone()).unwrap(), width);
    (width, base_a, base_b)
}

#[test]
#[ignore = "Root CUDA only: set RETIREMENT_BERT_MODEL and INFINITY_FA4_ROOT"]
fn bert_held_owner_survives_cache_pressure_and_releases_by_proof() {
    let name = "RETIREMENT_BERT_MODEL";
    let model_type = ModelType::Embedding(Pool::Mean);
    let a = synthetic(&[1, 2, 3], 7, 1);
    let b = synthetic(&[1, 2, 3], 128, 1);
    let (width, base_a, base_b) = baselines(name, model_type.clone(), &a, &b);
    let backend = load(name, model_type);
    let baseline = clean();

    let held_a = backend.submit_embed(a.clone()).unwrap();
    let held_b = backend.submit_embed(b.clone()).unwrap();
    assert_deferred(&held_a);
    assert_deferred(&held_b);
    assert_eq!(clean(), baseline);
    assert_eq!(embedding_bits(held_b.complete().unwrap(), width), base_b);
    assert_eq!(clean(), baseline);

    let a_bytes = force_cached_pressure(&backend, work_key(&a).unwrap());
    assert!(a_bytes > 0);
    expect_overloaded(backend.submit_embed(b.clone()));
    {
        let engine = backend.engine.lock().unwrap();
        assert_eq!(engine.retired.len(), 1);
        assert_eq!(engine.retired.bytes(), a_bytes);
    }
    assert_eq!(clean(), baseline);

    assert_eq!(embedding_bits(held_a.complete().unwrap(), width), base_a);
    let proof_before = retired_proof_releases();
    assert_eq!(
        embedding_bits(
            backend.submit_embed(b.clone()).unwrap().complete().unwrap(),
            width
        ),
        base_b
    );
    assert!(retired_proof_releases() > proof_before);
    {
        let engine = backend.engine.lock().unwrap();
        assert_eq!(engine.retired.len(), 0);
        assert_eq!(engine.retired.bytes(), 0);
    }
    assert_no_submit_drain(baseline);

    // A never-marked constructor owner still carries queued cold-GPU init. It
    // must be retained as unproven instead of invoking a submit-path drain.
    {
        let mut engine = backend.engine.lock().unwrap();
        let active = engine.workspace.as_ref().expect("active B").key();
        let cold_key = (active.0.saturating_mul(2), active.1, active.2);
        let cold = engine
            .model
            .workspace(engine.stream.clone(), cold_key)
            .unwrap();
        assert!(cold.completion_ticket().is_none());
        let drains = default_drop_drains();
        engine.retire_workspace(cold).unwrap();
        assert_eq!(engine.retired.len(), 1);
        assert_eq!(default_drop_drains(), drains);
    }
    backend.health().unwrap();
}

#[test]
#[ignore = "Root CUDA only: set RETIREMENT_QWEN3_MODEL and INFINITY_FA4_ROOT"]
fn qwen3_decoder_growth_and_pressure_release_held_owner_without_drain() {
    let name = "RETIREMENT_QWEN3_MODEL";
    let model_type = ModelType::Embedding(Pool::Mean);
    let a = synthetic(&[1, 2, 3], 7, 1);
    let b = synthetic(&[1, 2, 3], 129, 1);
    let (width, base_a, base_b) = baselines(name, model_type.clone(), &a, &b);
    let backend = load(name, model_type);
    let baseline = clean();

    let held_a = backend.submit_embed(a.clone()).unwrap();
    assert_deferred(&held_a);
    let a_bytes = {
        let engine = backend.engine.lock().unwrap();
        engine.workspace.as_ref().expect("A workspace").bytes()
    };
    let held_b = backend.submit_embed(b.clone()).unwrap();
    assert_deferred(&held_b);
    {
        let engine = backend.engine.lock().unwrap();
        let b_bytes = engine.workspace.as_ref().expect("B workspace").bytes();
        assert!(b_bytes > a_bytes, "Qwen3 workspace must grow");
    }
    assert_eq!(clean(), baseline);
    assert_eq!(embedding_bits(held_b.complete().unwrap(), width), base_b);

    let cached_a_bytes = force_cached_pressure(&backend, work_key(&a).unwrap());
    assert!(cached_a_bytes > 0);
    expect_overloaded(backend.submit_embed(b.clone()));
    assert_eq!(clean(), baseline);
    assert_eq!(embedding_bits(held_a.complete().unwrap(), width), base_a);
    let proof_before = retired_proof_releases();
    assert_eq!(
        embedding_bits(
            backend.submit_embed(b.clone()).unwrap().complete().unwrap(),
            width
        ),
        base_b
    );
    assert!(retired_proof_releases() > proof_before);
    assert_no_submit_drain(baseline);
    backend.health().unwrap();
}

#[test]
#[ignore = "Root CUDA only: set RETIREMENT_DENSE_MODEL and INFINITY_FA4_ROOT"]
fn dense_growth_retires_held_owner_and_releases_by_proof() {
    let name = "RETIREMENT_DENSE_MODEL";
    let model_type = ModelType::Embedding(Pool::Mean);
    let a = synthetic(&[1, 2, 3], 8, 1);
    let b = synthetic(&[1, 2, 3], 16, 9);
    let (width, base_a, base_b) = baselines(name, model_type.clone(), &a, &b);
    let backend = load(name, model_type);
    {
        let engine = backend.engine.lock().unwrap();
        assert!(
            (*engine.dense).is_some(),
            "model must declare a Dense module"
        );
    }
    let baseline = clean();

    let held_a = backend.submit_embed(a.clone()).unwrap();
    assert_deferred(&held_a);
    let held_b = backend.submit_embed(b.clone()).unwrap();
    assert_deferred(&held_b);
    assert_eq!(clean(), baseline);
    {
        let engine = backend.engine.lock().unwrap();
        assert_eq!(engine.retired.len(), 1, "old Dense must be retired");
        assert!(engine.retired.bytes() > 0);
    }
    assert_eq!(embedding_bits(held_b.complete().unwrap(), width), base_b);
    assert_eq!(embedding_bits(held_a.complete().unwrap(), width), base_a);
    let proof_before = retired_proof_releases();
    assert_eq!(
        embedding_bits(
            backend.submit_embed(b.clone()).unwrap().complete().unwrap(),
            width
        ),
        base_b
    );
    assert!(retired_proof_releases() > proof_before);
    assert_no_submit_drain(baseline);
    backend.health().unwrap();
}

#[test]
#[ignore = "Root CUDA only: set RETIREMENT_CLASSIFIER_MODEL and INFINITY_FA4_ROOT"]
fn classifier_held_owner_retires_nested_scratch_without_drain() {
    let name = "RETIREMENT_CLASSIFIER_MODEL";
    let a = synthetic(&[1, 2, 3], 7, 1);
    let b = synthetic(&[1, 2, 3], 128, 1);
    let (base_a, base_b) = {
        let gold = load(name, ModelType::Classifier);
        (
            prediction_bits(gold.predict(a.clone()).unwrap()),
            prediction_bits(gold.predict(b.clone()).unwrap()),
        )
    };
    let backend = load(name, ModelType::Classifier);
    let baseline = clean();

    let held_a = backend.submit_predict(a.clone()).unwrap();
    let held_b = backend.submit_predict(b.clone()).unwrap();
    assert_deferred(&held_a);
    assert_deferred(&held_b);
    assert_eq!(clean(), baseline);
    assert_eq!(prediction_bits(held_b.complete().unwrap()), base_b);
    assert_eq!(clean(), baseline);

    force_cached_pressure(&backend, work_key(&a).unwrap());
    expect_overloaded(backend.submit_predict(b.clone()));
    assert_eq!(clean(), baseline);
    assert_eq!(prediction_bits(held_a.complete().unwrap()), base_a);
    let proof_before = retired_proof_releases();
    assert_eq!(
        prediction_bits(
            backend
                .submit_predict(b.clone())
                .unwrap()
                .complete()
                .unwrap()
        ),
        base_b
    );
    assert!(retired_proof_releases() > proof_before);
    assert_no_submit_drain(baseline);
    backend.health().unwrap();
}

#[test]
#[ignore = "Root CUDA only: set RETIREMENT_DEBERTA_MODEL and INFINITY_FA4_ROOT"]
fn deberta_held_owner_retires_deberta_scratch_without_drain() {
    let name = "RETIREMENT_DEBERTA_MODEL";
    let model_type = ModelType::Embedding(Pool::Mean);
    let a = synthetic(&[1, 2, 3], 7, 1);
    let b = synthetic(&[1, 2, 3], 128, 1);
    let (width, base_a, base_b) = baselines(name, model_type.clone(), &a, &b);
    let backend = load(name, model_type);
    let baseline = clean();

    let held_a = backend.submit_embed(a.clone()).unwrap();
    let held_b = backend.submit_embed(b.clone()).unwrap();
    assert_deferred(&held_a);
    assert_deferred(&held_b);
    assert_eq!(clean(), baseline);
    assert_eq!(embedding_bits(held_b.complete().unwrap(), width), base_b);
    assert_eq!(clean(), baseline);

    force_cached_pressure(&backend, work_key(&a).unwrap());
    expect_overloaded(backend.submit_embed(b.clone()));
    assert_eq!(clean(), baseline);
    assert_eq!(embedding_bits(held_a.complete().unwrap(), width), base_a);
    let proof_before = retired_proof_releases();
    assert_eq!(
        embedding_bits(
            backend.submit_embed(b.clone()).unwrap().complete().unwrap(),
            width
        ),
        base_b
    );
    assert!(retired_proof_releases() > proof_before);
    assert_no_submit_drain(baseline);
    backend.health().unwrap();
}
