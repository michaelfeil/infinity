// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Root GPU lifecycle gate for the admitted H384 MiniLM FP16 encoder.
//!
//! This test is intentionally ignored on CPU-only workers. It exercises the
//! actual backend, all three pools, raw and mixed readouts, deferred out-of-order
//! completion, token types, positions, invalid-input recovery and health.
use infinity_backend_oxide::OxideBackend;
use infinity_v3_backend_core::{
    Backend, BackendError, Batch, Embedding, Embeddings, Inference, ModelType, Pool,
};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const HIDDEN: usize = 384;

fn model_path() -> PathBuf {
    std::env::var_os("INFINITY_OXIDE_TEST_MINILM_F16_MODEL")
        .map(PathBuf::from)
        .expect("set INFINITY_OXIDE_TEST_MINILM_F16_MODEL to the H384 MiniLM checkpoint")
}

fn validate_profile(path: &Path) {
    let bytes = std::fs::read(path.join("config.json")).expect("read config.json");
    let config: serde_json::Value = serde_json::from_slice(&bytes).expect("parse config.json");
    for (name, expected) in [
        ("hidden_size", 384u64),
        ("intermediate_size", 1536),
        ("num_attention_heads", 12),
        ("num_hidden_layers", 6),
    ] {
        assert_eq!(
            config.get(name).and_then(serde_json::Value::as_u64),
            Some(expected),
            "config.json {name}"
        );
    }
    assert_eq!(
        config.get("model_type").and_then(serde_json::Value::as_str),
        Some("bert")
    );
}

fn make_batch(lengths: &[usize], pooled: &[u32], raw: &[u32]) -> Batch {
    let total: usize = lengths.iter().sum();
    let mut input_ids = Vec::with_capacity(total);
    let mut position_ids = Vec::with_capacity(total);
    let mut token_type_ids = Vec::with_capacity(total);
    let mut cumulative_seq_lengths = Vec::with_capacity(lengths.len() + 1);
    cumulative_seq_lengths.push(0u32);
    for (sequence, &length) in lengths.iter().enumerate() {
        for position in 0..length {
            let index = input_ids.len();
            input_ids.push(20 + (index * 37 % 20_000) as u32);
            position_ids.push(position as u32);
            token_type_ids.push(((sequence + position) % 2) as u32);
        }
        cumulative_seq_lengths.push(input_ids.len() as u32);
    }
    Batch {
        multimodal: Vec::new(),
        input_ids,
        token_type_ids,
        position_ids,
        cumulative_seq_lengths,
        max_length: lengths.iter().copied().max().unwrap_or(0) as u32,
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

/// Normalized observable readout: explicit pooled/raw shape, sequence keys and
/// exact F32 bits. `Embedding::All` entries are retained rather than filtered.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Readout {
    Pooled(Vec<u32>),
    Raw(Vec<Vec<u32>>),
}

fn observe(embeddings: Embeddings) -> BTreeMap<usize, Readout> {
    let mut observed = BTreeMap::new();
    for (index, embedding) in embeddings {
        let readout = match embedding {
            Embedding::Pooled(values) => {
                assert_eq!(values.len(), HIDDEN, "sequence {index} pooled width");
                assert!(
                    values.iter().all(|value| value.is_finite()),
                    "sequence {index} pooled values are not finite"
                );
                Readout::Pooled(values.iter().map(|value| value.to_bits()).collect())
            }
            Embedding::All(rows) => {
                assert!(!rows.is_empty(), "sequence {index} raw rows are empty");
                Readout::Raw(
                    rows.into_iter()
                        .map(|row| {
                            assert_eq!(row.len(), HIDDEN, "sequence {index} raw width");
                            assert!(
                                row.iter().all(|value| value.is_finite()),
                                "sequence {index} raw values are not finite"
                            );
                            row.iter().map(|value| value.to_bits()).collect()
                        })
                        .collect(),
                )
            }
        };
        assert!(
            observed.insert(index, readout).is_none(),
            "duplicate sequence {index}"
        );
    }
    observed
}

fn assert_keys(readout: &BTreeMap<usize, Readout>, mut keys: Vec<usize>) {
    keys.sort_unstable();
    assert_eq!(
        readout.keys().copied().collect::<Vec<_>>(),
        keys,
        "unexpected readout sequence keys"
    );
}

fn pooled_values(readout: &BTreeMap<usize, Readout>, key: usize) -> Vec<u32> {
    match readout.get(&key) {
        Some(Readout::Pooled(values)) => values.clone(),
        other => panic!("sequence {key} expected pooled values, observed {other:?}"),
    }
}

fn raw_rows(readout: &BTreeMap<usize, Readout>, key: usize) -> Vec<Vec<u32>> {
    match readout.get(&key) {
        Some(Readout::Raw(rows)) => rows.clone(),
        other => panic!("sequence {key} expected raw rows, observed {other:?}"),
    }
}

#[test]
#[ignore = "Requires CUDA, INFINITY_OXIDE_TEST_MINILM_F16_MODEL, INFINITY_OXIDE_KERNEL_MODULE, INFINITY_FA4_ROOT and an F16 mha32 profile"]
fn fp16_minilm_all_pools_raw_mixed_deferred_types_positions_recovery() {
    let path = model_path();
    validate_profile(&path);
    for pool in [Pool::Mean, Pool::Cls, Pool::LastToken] {
        let label = format!("{pool}");
        let backend = OxideBackend::new(&path, "float16", ModelType::Embedding(pool), 0)
            .unwrap_or_else(|error| panic!("{label}: F16 backend construction failed: {error:?}"));
        backend.health().expect("healthy F16 backend");

        let a = make_batch(&[7, 3], &[1, 0], &[]);
        let b = make_batch(&[1, 17, 5], &[2, 0], &[]);
        let raw_a = make_batch(&[11, 2], &[0], &[1, 0]);
        let mut mixed_b = b.clone();
        mixed_b.raw_indices = vec![1];

        let baseline_a = observe(backend.embed(a.clone()).expect("sync A"));
        assert_keys(&baseline_a, vec![0, 1]);
        let baseline_b = observe(backend.embed(b.clone()).expect("sync B"));
        assert_keys(&baseline_b, vec![0, 2]);
        assert_eq!(
            observe(backend.embed(a.clone()).expect("sync A repeat")),
            baseline_a,
            "{label}: synchronous results are not bit exact"
        );

        let raw_first = observe(backend.embed(raw_a.clone()).expect("raw readout"));
        assert_keys(&raw_first, vec![0, 1]);
        assert_eq!(raw_rows(&raw_first, 0).len(), 11);
        assert_eq!(raw_rows(&raw_first, 1).len(), 2);
        assert_eq!(
            observe(backend.embed(raw_a.clone()).expect("raw readout repeat")),
            raw_first,
            "{label}: raw readout is not bit exact across repeats"
        );

        let mixed_first = observe(backend.embed(mixed_b.clone()).expect("mixed readout"));
        assert_keys(&mixed_first, vec![0, 1, 2]);
        for key in [2usize, 0usize] {
            assert_eq!(
                pooled_values(&mixed_first, key),
                pooled_values(&baseline_b, key),
                "{label}: mixed pooled sequence {key}"
            );
        }
        assert_eq!(raw_rows(&mixed_first, 1).len(), 17);
        assert_eq!(
            observe(
                backend
                    .embed(mixed_b.clone())
                    .expect("mixed readout repeat")
            ),
            mixed_first,
            "{label}: mixed readout is not bit exact across repeats"
        );

        // Two-lease A/B/A schedule. A completes and returns its lease before the
        // second A is admitted while B still owns the other lease. Submission
        // order is A, B, A; completion order is first A, second A, B.
        let held_a = backend.submit_embed(a.clone()).expect("deferred A submit");
        let held_b = backend.submit_embed(b.clone()).expect("deferred B submit");
        assert!(matches!(&held_a, Inference::Deferred(_)));
        assert!(matches!(&held_b, Inference::Deferred(_)));
        assert_eq!(
            observe(held_a.complete().expect("deferred first A completion")),
            baseline_a,
            "{label}: deferred first A"
        );
        let held_a2 = backend
            .submit_embed(a.clone())
            .expect("deferred second A submit while B is held");
        assert!(matches!(&held_a2, Inference::Deferred(_)));
        assert_eq!(
            observe(held_a2.complete().expect("deferred second A completion")),
            baseline_a,
            "{label}: deferred second A"
        );
        assert_eq!(
            observe(held_b.complete().expect("deferred B completion")),
            baseline_b,
            "{label}: deferred B"
        );
        backend.health().unwrap_or_else(|error| {
            panic!("{label}: unhealthy after deferred schedule: {error:?}")
        });
        // Grow the actual model arena for the first time while A still owns
        // the other output lease, then verify out-of-order completion.
        let c = make_batch(&[129], &[0], &[]);
        let held_a = backend
            .submit_embed(a.clone())
            .expect("held A before arena growth");
        let held_c = backend
            .submit_embed(c.clone())
            .expect("first C grows arena");
        let grown_c = observe(held_c.complete().expect("grown C completes first"));
        assert_eq!(
            observe(held_a.complete().expect("held A after arena growth")),
            baseline_a
        );
        assert_eq!(
            grown_c,
            observe(backend.embed(c).expect("grown C reference repeat"))
        );
        assert_eq!(
            observe(backend.embed(b.clone()).expect("B after arena growth")),
            baseline_b
        );

        for (name, mutate) in [
            (
                "token type",
                (|batch: &mut Batch| batch.token_type_ids[0] = u32::MAX) as fn(&mut Batch),
            ),
            ("position", |batch: &mut Batch| {
                batch.position_ids[0] = u32::MAX
            }),
            ("boundary", |batch: &mut Batch| {
                let last = batch.cumulative_seq_lengths.last_mut().unwrap();
                *last += 1;
            }),
            ("pooled index", |batch: &mut Batch| {
                batch.pooled_indices[0] = u32::MAX
            }),
        ] {
            let mut invalid = a.clone();
            mutate(&mut invalid);
            let error = match backend.submit_embed(invalid) {
                Ok(_) => panic!("{label}: invalid {name} was admitted"),
                Err(error) => error,
            };
            assert!(
                matches!(error, BackendError::Inference(_)),
                "{label}: {name} expected typed Inference, got {error:?}"
            );
            backend
                .health()
                .unwrap_or_else(|error| panic!("{label}: unhealthy after {name}: {error:?}"));
            assert_eq!(
                observe(backend.embed(a.clone()).expect("post-rejection A")),
                baseline_a,
                "{label}: recovery after invalid {name}"
            );
        }
    }
}
