// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Gemma3 request ownership and admission controls over the shared decoder arena.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use infinity_backend_oxide::OxideBackend;
use infinity_v3_backend_core::{
    Backend, BackendError, Batch, Embedding, Embeddings, Inference, ModelType, Pool,
};

const HIDDEN: usize = 640;
const VOCAB: u32 = 262_144;
const MAX_POSITION: u32 = 32_768;

type OutputBits = BTreeMap<usize, Vec<Vec<u32>>>;

fn fixture(lengths: &[u32], pooled: &[u32], raw: &[u32]) -> Batch {
    let mut cumulative_seq_lengths = vec![0];
    let mut input_ids = Vec::new();
    let mut position_ids = Vec::new();
    for &length in lengths {
        assert!(length > 0 && length <= MAX_POSITION);
        for position in 0..length {
            input_ids.push(20 + (input_ids.len() as u32 * 13) % 20_000);
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
        max_length: lengths.iter().copied().max().unwrap(),
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

fn bits(result: &Embeddings, batch: &Batch) -> OutputBits {
    let raw: BTreeSet<_> = batch.raw_indices.iter().map(|&i| i as usize).collect();
    let selected: BTreeSet<_> = batch
        .pooled_indices
        .iter()
        .chain(&batch.raw_indices)
        .map(|&i| i as usize)
        .collect();
    assert_eq!(result.len(), selected.len());
    let mut output = BTreeMap::new();
    for (&index, embedding) in result.iter() {
        assert!(selected.contains(&index));
        let rows: Vec<&[f32]> = match embedding {
            Embedding::Pooled(row) => {
                assert!(!raw.contains(&index));
                vec![row]
            }
            Embedding::All(rows) => {
                assert!(raw.contains(&index));
                let length =
                    batch.cumulative_seq_lengths[index + 1] - batch.cumulative_seq_lengths[index];
                assert_eq!(rows.len(), length as usize);
                rows.iter().map(Vec::as_slice).collect()
            }
        };
        let rows = rows
            .into_iter()
            .map(|row| {
                assert_eq!(row.len(), HIDDEN);
                assert!(row.iter().all(|value| value.is_finite()));
                row.iter().map(|value| value.to_bits()).collect()
            })
            .collect();
        assert!(output.insert(index, rows).is_none());
    }
    assert_eq!(output.keys().copied().collect::<BTreeSet<_>>(), selected);
    output
}

fn completed(call: Inference<Embeddings>, batch: &Batch) -> OutputBits {
    bits(&call.complete().unwrap(), batch)
}

fn deferred(backend: &OxideBackend, batch: &Batch) -> Inference<Embeddings> {
    let call = backend.submit_embed(batch.clone()).unwrap();
    assert!(matches!(&call, Inference::Deferred(_)));
    call
}

fn reference(backend: &OxideBackend, batch: &Batch) -> OutputBits {
    let first = bits(&backend.embed(batch.clone()).unwrap(), batch);
    assert_eq!(bits(&backend.embed(batch.clone()).unwrap(), batch), first);
    first
}

fn rejected(backend: &OxideBackend, batch: Batch) {
    let error = match backend.submit_embed(batch) {
        Ok(_) => panic!("invalid Gemma3 request was admitted"),
        Err(error) => error,
    };
    assert!(matches!(error, BackendError::Inference(_)), "{error:?}");
    backend.health().unwrap();
}

fn run_pool(path: &std::path::Path, pool: Pool) {
    let backend = OxideBackend::new(path, "bfloat16", ModelType::Embedding(pool), 0).unwrap();
    // Different T/B/readout shapes exercise both arena reuse and replacement.
    // Separate baselines intentionally preserve the real GEMM M for each mode:
    // pooled selected-row execution may differ from full-token raw execution.
    let a = fixture(&[7, 3], &[1, 0, 1], &[]);
    let b = fixture(&[1, 17, 5], &[2, 0, 1], &[1, 1]);
    let c = fixture(&[33], &[], &[0, 0]);
    let d = fixture(&[2, 71, 1], &[2, 0], &[]);
    let base_a = reference(&backend, &a);
    let base_b = reference(&backend, &b);
    let base_c = reference(&backend, &c);
    let base_d = reference(&backend, &d);

    // Gemma3 does not consume type IDs; length remains part of admission.
    for batch in [&a, &b, &c] {
        let expected = reference(&backend, batch);
        let mut changed = batch.clone();
        for (index, kind) in changed.token_type_ids.iter_mut().enumerate() {
            *kind = if index % 2 == 0 { u32::MAX } else { 1 };
        }
        assert_eq!(reference(&backend, &changed), expected);
    }

    // Same physical token rows and full-token route: duplicates and pooled/raw
    // overlap must only affect public selection, never the raw token values.
    let all_raw = fixture(&[1, 17, 5], &[], &[2, 1, 0]);
    let all_raw_bits = reference(&backend, &all_raw);
    assert_eq!(base_b.get(&1), all_raw_bits.get(&1));

    let held_a = deferred(&backend, &a);
    let held_b = deferred(&backend, &b);
    assert_eq!(completed(held_b, &b), base_b);
    assert_eq!(completed(held_a, &a), base_a);
    let held_a = deferred(&backend, &a);
    drop(deferred(&backend, &c));
    assert_eq!(reference(&backend, &d), base_d);
    assert_eq!(completed(held_a, &a), base_a);
    assert_eq!(reference(&backend, &c), base_c);

    // Every early rejection occurs while another real result owns its arena.
    let held_b = deferred(&backend, &b);
    let mut invalid = a.clone();
    invalid.input_ids[0] = VOCAB;
    rejected(&backend, invalid);
    let mut invalid = a.clone();
    invalid.position_ids[0] = MAX_POSITION;
    rejected(&backend, invalid);
    let mut invalid = a.clone();
    invalid.token_type_ids.pop();
    rejected(&backend, invalid);
    let mut invalid = a.clone();
    invalid.cumulative_seq_lengths[1] = 0;
    rejected(&backend, invalid);
    let mut invalid = a.clone();
    invalid.raw_indices = vec![2];
    rejected(&backend, invalid);
    assert_eq!(completed(held_b, &b), base_b);
    assert_eq!(reference(&backend, &a), base_a);

    let empty = fixture(&[7, 3], &[], &[]);
    assert!(backend.embed(empty).unwrap().is_empty());
    backend.health().unwrap();

    // Pending operations retain their stream, context and allocation owner even
    // when the model facade and another pending operation are dropped first.
    let held_a = deferred(&backend, &a);
    let held_c = deferred(&backend, &c);
    drop(backend);
    drop(held_a);
    assert_eq!(completed(held_c, &c), base_c);
}

#[test]
#[ignore = "Requires CUDA, Gemma3 kernels/FA4, and INFINITY_OXIDE_TEST_GEMMA3_MODEL"]
fn gemma3_shared_arena_shape_changes_cancellation_and_admission_recovery() {
    let path = PathBuf::from(std::env::var_os("INFINITY_OXIDE_TEST_GEMMA3_MODEL").unwrap());
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path.join("config.json")).unwrap()).unwrap();
    assert_eq!(config["model_type"], "gemma3_text");
    assert_eq!(config["hidden_size"], HIDDEN);
    assert_eq!(config["vocab_size"], VOCAB);
    assert_eq!(config["max_position_embeddings"], MAX_POSITION);
    for pool in [Pool::Cls, Pool::Mean, Pool::LastToken] {
        run_pool(&path, pool);
    }
}

#[test]
fn duplicate_mixed_fixture_checks_public_shapes_and_raw_override() {
    let batch = fixture(&[1, 2], &[1, 0, 1], &[1, 1]);
    let mut result = Embeddings::default();
    result.insert(0, Embedding::Pooled(vec![1.0; HIDDEN]));
    result.insert(1, Embedding::All(vec![vec![-0.0; HIDDEN]; 2]));
    let readout = bits(&result, &batch);
    assert_eq!(readout.len(), 2);
    assert_eq!(readout[&0].len(), 1);
    assert_eq!(readout[&1].len(), 2);
    assert!(readout[&1][0]
        .iter()
        .all(|&word| word == (-0.0f32).to_bits()));
}
