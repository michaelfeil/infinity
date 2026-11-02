// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Unified production-factory consumer gate for the three Gemma3 text checkpoints.
//!
//! Ignored by default: this is the Root GPU0 qualification recipe, not a CPU
//! stand-in. It uses only the public `OxideBackend` factory and `Backend`
//! contract, proving cold factory construction, shared Work ownership,
//! `Inference::Deferred` completion, and raw/pooled parity. Native LibTorch
//! graphs stay at their unchanged defaults in the controller; Oxide remains
//! graph-free.

use std::collections::BTreeMap;

use infinity_backend_oxide::OxideBackend;
use infinity_v3_backend_core::{
    Backend, BackendError, Batch, Embedding, Embeddings, Inference, ModelType, Pool,
};

fn model_path() -> std::path::PathBuf {
    std::env::var_os("INFINITY_OXIDE_TEST_GEMMA3_MODEL")
        .map(std::path::PathBuf::from)
        .expect("set INFINITY_OXIDE_TEST_GEMMA3_MODEL to a Gemma3 text checkpoint")
}

fn model_config(path: &std::path::Path) -> (usize, u32, u32) {
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path.join("config.json")).unwrap()).unwrap();
    assert_eq!(config["model_type"], "gemma3_text");
    let hidden = config["hidden_size"].as_u64().unwrap() as usize;
    assert!(
        matches!(hidden, 640 | 3840 | 5376),
        "unexpected Gemma3 hidden width {hidden}"
    );
    (
        hidden,
        config["vocab_size"].as_u64().unwrap() as u32,
        config["max_position_embeddings"].as_u64().unwrap() as u32,
    )
}

fn factory(path: &std::path::Path) -> OxideBackend {
    OxideBackend::new(path, "bfloat16", ModelType::Embedding(Pool::LastToken), 0)
        .expect("cold Gemma3 production factory")
}

fn fixture(lengths: &[u32], pooled: &[u32], raw: &[u32], vocab: u32, max_positions: u32) -> Batch {
    let mut cumulative_seq_lengths = vec![0];
    let mut input_ids = Vec::new();
    let mut position_ids = Vec::new();
    for &length in lengths {
        assert!(length > 0 && length <= max_positions);
        for position in 0..length {
            input_ids.push(20 + (input_ids.len() as u32 * 13) % (vocab - 32));
            position_ids.push(position);
        }
        cumulative_seq_lengths.push(input_ids.len() as u32);
    }
    Batch {
        multimodal: Vec::new(),
        token_type_ids: vec![0; input_ids.len()],
        input_ids,
        position_ids,
        cumulative_seq_lengths,
        max_length: lengths.iter().copied().max().unwrap_or(0),
        pooled_indices: pooled.to_vec(),
        raw_indices: raw.to_vec(),
        compact_input_ids: None,
        compact_position_ids: None,
        scatter_unfold: None,
        fold_gather: None,
        tokens: Vec::new(),
        offsets: Vec::new(),
    }
}

fn rows(result: &Embeddings, batch: &Batch, hidden: usize) -> BTreeMap<usize, Vec<Vec<u32>>> {
    let mut out = BTreeMap::new();
    for (&index, value) in result.iter() {
        let rows: Vec<&[f32]> = match value {
            Embedding::Pooled(row) => vec![row],
            Embedding::All(tokens) => tokens.iter().map(Vec::as_slice).collect(),
        };
        let bits = rows
            .into_iter()
            .map(|row| {
                assert_eq!(row.len(), hidden);
                assert!(row.iter().all(|value| value.is_finite()));
                row.iter().map(|value| value.to_bits()).collect()
            })
            .collect();
        assert!(out.insert(index, bits).is_none());
    }
    let expected = batch
        .pooled_indices
        .iter()
        .chain(&batch.raw_indices)
        .map(|&index| index as usize)
        .collect::<std::collections::BTreeSet<_>>();
    let actual = out
        .keys()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(actual, expected);
    out
}

#[test]
#[ignore = "Root GPU0: actual Gemma3 cold factory, Work/Deferred lifecycle and raw/pooled parity"]
fn gemma3_factory_work_deferred_and_raw_pooled_parity() {
    let path = model_path();
    let (hidden, vocab, max_positions) = model_config(&path);
    let backend = factory(&path);
    backend.health().unwrap();

    // Selected-row and full-token paths use different GEMM row counts. Check
    // repeatability within each route; Native quality needs per-route gold.
    for lengths in [vec![7u32, 3], vec![1, 17, 5], vec![33], vec![2, 71, 1]] {
        let selected = (0..lengths.len() as u32).collect::<Vec<_>>();
        let pooled = fixture(&lengths, &selected, &[], vocab, max_positions);
        let raw = fixture(&lengths, &[], &selected, vocab, max_positions);
        let pooled_first = rows(&backend.embed(pooled.clone()).unwrap(), &pooled, hidden);
        let pooled_second = rows(&backend.embed(pooled.clone()).unwrap(), &pooled, hidden);
        assert_eq!(pooled_first, pooled_second);
        let raw_first = rows(&backend.embed(raw.clone()).unwrap(), &raw, hidden);
        let raw_second = rows(&backend.embed(raw.clone()).unwrap(), &raw, hidden);
        assert_eq!(raw_first, raw_second);
    }

    // Duplicate/public selection order must not change canonical rows.
    let base = fixture(&[7, 3, 11], &[1, 0, 2], &[], vocab, max_positions);
    let duplicate = fixture(&[7, 3, 11], &[2, 0, 2, 1, 0], &[], vocab, max_positions);
    let base_rows = rows(&backend.embed(base.clone()).unwrap(), &base, hidden);
    let duplicate_rows = rows(
        &backend.embed(duplicate.clone()).unwrap(),
        &duplicate,
        hidden,
    );
    for (index, expected) in &base_rows {
        assert_eq!(duplicate_rows.get(index).unwrap(), expected);
    }

    // Deferred completion keeps its own arena/stream owner and completes out of order.
    let a = fixture(&[7, 3], &[1, 0, 1], &[], vocab, max_positions);
    let b = fixture(&[1, 17, 5], &[2, 0, 1], &[1, 1], vocab, max_positions);
    let expected_a = rows(&backend.embed(a.clone()).unwrap(), &a, hidden);
    let expected_b = rows(&backend.embed(b.clone()).unwrap(), &b, hidden);
    let held_a = backend.submit_embed(a.clone()).unwrap();
    let held_b = backend.submit_embed(b.clone()).unwrap();
    assert!(matches!(&held_a, Inference::Deferred(_)));
    assert!(matches!(&held_b, Inference::Deferred(_)));
    assert_eq!(rows(&held_b.complete().unwrap(), &b, hidden), expected_b);
    assert_eq!(rows(&held_a.complete().unwrap(), &a, hidden), expected_a);

    // Dropping a pending request must leave the engine healthy and the Work reusable.
    let held = backend.submit_embed(b.clone()).unwrap();
    drop(held);
    backend.health().unwrap();
    assert_eq!(
        rows(&backend.embed(b.clone()).unwrap(), &b, hidden),
        expected_b
    );
    let held = backend.submit_embed(a.clone()).unwrap();
    drop(backend);
    assert_eq!(rows(&held.complete().unwrap(), &a, hidden), expected_a);
}

#[test]
#[ignore = "Root GPU0: actual Gemma3 admission rejects without poisoning the factory"]
fn gemma3_factory_rejections_leave_health_green() {
    let path = model_path();
    let (hidden, vocab, max_positions) = model_config(&path);
    let backend = factory(&path);
    let good = fixture(&[7, 3], &[0, 1], &[], vocab, max_positions);
    let expected = rows(&backend.embed(good.clone()).unwrap(), &good, hidden);

    let mut invalid = good.clone();
    invalid.input_ids[0] = vocab;
    assert!(matches!(
        backend.submit_embed(invalid),
        Err(BackendError::Inference(_))
    ));
    backend.health().unwrap();
    let mut invalid = good.clone();
    invalid.position_ids[0] = max_positions;
    assert!(matches!(
        backend.submit_embed(invalid),
        Err(BackendError::Inference(_))
    ));
    backend.health().unwrap();
    let mut ignored_types = good;
    ignored_types.token_type_ids.fill(u32::MAX);
    let actual = rows(
        &backend.embed(ignored_types).unwrap(),
        &fixture(&[7, 3], &[0, 1], &[], vocab, max_positions),
        hidden,
    );
    assert_eq!(actual, expected);
    backend.health().unwrap();
}
