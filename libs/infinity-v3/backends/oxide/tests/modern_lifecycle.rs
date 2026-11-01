// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use infinity_backend_oxide::OxideBackend;
use infinity_v3_backend_core::{
    Backend, BackendError, Batch, Embedding, Embeddings, Inference, ModelType, Pool,
};

type PooledBits = BTreeMap<u32, Vec<u32>>;
type NamedBaseline<'a> = (&'a Batch, &'a [u32], &'a PooledBits, &'a str);

const HIDDEN_SIZE: usize = 768;
const INTERMEDIATE_SIZE: u64 = 1152;
const MAX_LENGTH: u32 = 8192;
const INVALID_TOKEN_ID: u32 = 50368;
const TOKEN_ID_LIMIT: u32 = 20000;
const SELECT_A: &[u32] = &[1, 0];
const SELECT_B: &[u32] = &[2, 0];
const SELECT_C: &[u32] = &[0];
const SELECT_D: &[u32] = &[1, 2, 0];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TypeIds {
    Zero,
    Ones,
    Mixed,
}

fn model_path() -> PathBuf {
    std::env::var_os("INFINITY_OXIDE_TEST_MODERN_ENCODER_MODEL")
        .map(PathBuf::from)
        .expect(
            "Set INFINITY_OXIDE_TEST_MODERN_ENCODER_MODEL to the ModernBERT base model directory",
        )
}

fn validate_model_profile(path: &Path) {
    let config_path = path.join("config.json");
    let bytes = std::fs::read(&config_path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", config_path.display()));
    let config: serde_json::Value = serde_json::from_slice(&bytes)
        .unwrap_or_else(|error| panic!("failed to parse {}: {error}", config_path.display()));
    let model_type = config
        .get("model_type")
        .and_then(|value| value.as_str())
        .unwrap_or_else(|| panic!("{} has no string model_type", config_path.display()));
    assert!(
        model_type == "modernbert",
        "{} has unsupported model_type {model_type:?}",
        config_path.display()
    );
    assert_eq!(
        config.get("hidden_size").and_then(|value| value.as_u64()),
        Some(HIDDEN_SIZE as u64),
        "{} has unexpected hidden_size",
        config_path.display()
    );
    assert_eq!(
        config
            .get("intermediate_size")
            .and_then(|value| value.as_u64()),
        Some(INTERMEDIATE_SIZE),
        "{} has unexpected intermediate_size",
        config_path.display()
    );
}

fn type_value(kind: TypeIds, index: usize) -> u32 {
    match kind {
        TypeIds::Zero => 0,
        TypeIds::Ones => 1,
        TypeIds::Mixed => match index % 3 {
            0 => u32::MAX,
            1 => 1,
            _ => 0,
        },
    }
}

fn make_batch(seq_lens: &[usize], pooled_indices: &[u32], types: TypeIds) -> Batch {
    assert!(
        !seq_lens.is_empty(),
        "a fixture batch needs at least one sequence"
    );
    let total: usize = seq_lens.iter().copied().sum();
    let mut input_ids = Vec::with_capacity(total);
    let mut token_type_ids = Vec::with_capacity(total);
    let mut position_ids = Vec::with_capacity(total);
    let mut cumulative_seq_lengths = Vec::with_capacity(seq_lens.len() + 1);
    cumulative_seq_lengths.push(0);
    let mut flat = 0usize;
    for &len in seq_lens {
        assert!(len > 0, "fixture sequences must be non-empty");
        assert!(
            len <= MAX_LENGTH as usize,
            "fixture sequence length {len} exceeds the local model profile"
        );
        for position in 0..len {
            input_ids.push(((flat % TOKEN_ID_LIMIT as usize) + 1) as u32);
            token_type_ids.push(type_value(types, flat));
            position_ids.push(position as u32);
            flat += 1;
        }
        cumulative_seq_lengths.push(flat as u32);
    }
    Batch {
        multimodal: Vec::new(),
        input_ids,
        token_type_ids,
        position_ids,
        cumulative_seq_lengths,
        max_length: *seq_lens.iter().max().unwrap() as u32,
        pooled_indices: pooled_indices.to_vec(),
        raw_indices: Vec::new(),
        compact_input_ids: None,
        compact_position_ids: None,
        scatter_unfold: None,
        fold_gather: None,
        tokens: Vec::new(),
        offsets: Vec::new(),
    }
}

fn batch_a() -> Batch {
    make_batch(&[7, 3], SELECT_A, TypeIds::Zero)
}

fn batch_b() -> Batch {
    make_batch(&[1, 17, 5], SELECT_B, TypeIds::Zero)
}

fn batch_c() -> Batch {
    make_batch(&[33], SELECT_C, TypeIds::Zero)
}

fn batch_d() -> Batch {
    make_batch(&[2, 71, 1], SELECT_D, TypeIds::Zero)
}

fn with_type_ids(batch: &Batch, types: TypeIds) -> Batch {
    let mut changed = batch.clone();
    for (index, value) in changed.token_type_ids.iter_mut().enumerate() {
        *value = type_value(types, index);
    }
    changed
}

fn oversized_batch() -> Batch {
    let total = (1usize << 20) + 1;
    let mut input_ids = Vec::with_capacity(total);
    let mut token_type_ids = Vec::with_capacity(total);
    let mut position_ids = Vec::with_capacity(total);
    input_ids.resize(total, 1);
    token_type_ids.resize(total, 0);
    position_ids.extend((0..total).map(|index| (index % MAX_LENGTH as usize) as u32));
    let mut cumulative_seq_lengths = Vec::with_capacity(65);
    cumulative_seq_lengths.push(0);
    let mut remaining = total;
    while remaining > 0 {
        let chunk = remaining.min(MAX_LENGTH as usize);
        let last = *cumulative_seq_lengths.last().unwrap() as usize;
        cumulative_seq_lengths.push((last + chunk) as u32);
        remaining -= chunk;
    }
    Batch {
        multimodal: Vec::new(),
        input_ids,
        token_type_ids,
        position_ids,
        cumulative_seq_lengths,
        max_length: MAX_LENGTH,
        pooled_indices: vec![0],
        raw_indices: Vec::new(),
        compact_input_ids: None,
        compact_position_ids: None,
        scatter_unfold: None,
        fold_gather: None,
        tokens: Vec::new(),
        offsets: Vec::new(),
    }
}

fn normalize(embeddings: &Embeddings, selected: &[u32], context: &str) -> BTreeMap<u32, Vec<u32>> {
    let expected: BTreeSet<u32> = selected.iter().copied().collect();
    assert_eq!(
        embeddings.len(),
        expected.len(),
        "{context}: expected exactly {} unique output keys",
        expected.len()
    );
    let mut normalized = BTreeMap::new();
    for (key, embedding) in embeddings.iter() {
        let key = u32::try_from(*key).expect("sequence key fits u32");
        assert!(
            expected.contains(&key),
            "{context}: unexpected output key {key}"
        );
        let values = match embedding {
            Embedding::Pooled(values) => values,
            Embedding::All(_) => panic!("{context}: key {key} returned non-pooled embeddings"),
        };
        assert_eq!(
            values.len(),
            HIDDEN_SIZE,
            "{context}: key {key} has width {} instead of {HIDDEN_SIZE}",
            values.len()
        );
        let mut bits = Vec::with_capacity(values.len());
        for (index, value) in values.iter().enumerate() {
            assert!(
                value.is_finite(),
                "{context}: key {key} value {index} is not finite"
            );
            bits.push(value.to_bits());
        }
        assert!(
            normalized.insert(key, bits).is_none(),
            "{context}: duplicate output key {key}"
        );
    }
    assert_eq!(
        normalized.keys().copied().collect::<BTreeSet<_>>(),
        expected,
        "{context}: output keys do not match the unique selected keys"
    );
    normalized
}

fn sync_baseline(
    backend: &OxideBackend,
    batch: &Batch,
    selected: &[u32],
    context: &str,
) -> BTreeMap<u32, Vec<u32>> {
    let first_embeddings = backend
        .embed(batch.clone())
        .unwrap_or_else(|error| panic!("{context}: first synchronous baseline failed: {error:?}"));
    let first = normalize(
        &first_embeddings,
        selected,
        &format!("{context} baseline first"),
    );
    drop(first_embeddings);
    let second_embeddings = backend
        .embed(batch.clone())
        .unwrap_or_else(|error| panic!("{context}: second synchronous baseline failed: {error:?}"));
    let second = normalize(
        &second_embeddings,
        selected,
        &format!("{context} baseline second"),
    );
    assert_eq!(
        first, second,
        "{context}: synchronous baselines are not bit exact"
    );
    first
}

fn complete_normalized(
    inference: Inference<Embeddings>,
    selected: &[u32],
    context: &str,
) -> BTreeMap<u32, Vec<u32>> {
    let embeddings = inference
        .complete()
        .unwrap_or_else(|error| panic!("{context}: deferred completion failed: {error:?}"));
    normalize(&embeddings, selected, context)
}

fn assert_deferred<T>(inference: &Inference<T>, context: &str) {
    assert!(
        matches!(inference, Inference::Deferred(_)),
        "{context}: submission was not deferred"
    );
}

fn assert_rejected(
    backend: &OxideBackend,
    invalid: Batch,
    selected: &[u32],
    baseline: &BTreeMap<u32, Vec<u32>>,
    context: &str,
) {
    let error = match backend.submit_embed(invalid) {
        Ok(_) => panic!("{context}: invalid batch was admitted"),
        Err(error) => error,
    };
    assert!(
        matches!(&error, BackendError::Inference(_)),
        "{context}: expected typed BackendError::Inference, got {error:?}"
    );
    assert!(
        !matches!(&error, BackendError::Unhealthy),
        "{context}: rejection poisoned backend health"
    );
    backend.health().unwrap_or_else(|error| {
        panic!("{context}: backend is unhealthy after rejection: {error:?}")
    });
    let embeddings = backend
        .embed(batch_a())
        .unwrap_or_else(|error| panic!("{context}: post-rejection A embed failed: {error:?}"));
    let actual = normalize(
        &embeddings,
        selected,
        &format!("{context} post-rejection A"),
    );
    assert_eq!(
        &actual, baseline,
        "{context}: post-rejection A does not match its baseline"
    );
}

fn rows_bucket(rows: usize) -> usize {
    rows.max(32).next_power_of_two()
}

type RawBits = BTreeMap<u32, Vec<Vec<u32>>>;

fn readout_bits(embeddings: &Embeddings, batch: &Batch) -> (PooledBits, RawBits) {
    let pooled_keys: BTreeSet<_> = batch.pooled_indices.iter().copied().collect();
    let raw_keys: BTreeSet<_> = batch.raw_indices.iter().copied().collect();
    assert!(pooled_keys.is_disjoint(&raw_keys));
    assert_eq!(embeddings.len(), pooled_keys.len() + raw_keys.len());
    let mut pooled = BTreeMap::new();
    let mut raw = BTreeMap::new();
    for (&key, embedding) in embeddings.iter() {
        let key = u32::try_from(key).unwrap();
        let rows: Vec<&[f32]> = match embedding {
            Embedding::Pooled(row) => {
                assert!(pooled_keys.contains(&key));
                vec![row]
            }
            Embedding::All(rows) => {
                assert!(raw_keys.contains(&key));
                let start = batch.cumulative_seq_lengths[key as usize];
                let end = batch.cumulative_seq_lengths[key as usize + 1];
                assert_eq!(rows.len(), (end - start) as usize);
                rows.iter().map(Vec::as_slice).collect()
            }
        };
        let bits: Vec<Vec<u32>> = rows
            .into_iter()
            .map(|row| {
                assert_eq!(row.len(), HIDDEN_SIZE);
                assert!(row.iter().all(|value| value.is_finite()));
                row.iter().map(|value| value.to_bits()).collect()
            })
            .collect();
        if pooled_keys.contains(&key) {
            assert!(pooled
                .insert(key, bits.into_iter().next().unwrap())
                .is_none());
        } else {
            assert!(raw.insert(key, bits).is_none());
        }
    }
    assert_eq!(pooled.keys().copied().collect::<BTreeSet<_>>(), pooled_keys);
    assert_eq!(raw.keys().copied().collect::<BTreeSet<_>>(), raw_keys);
    (pooled, raw)
}

fn selected_readout(batch: &Batch, pooled: &[u32], raw: &[u32]) -> Batch {
    let mut selected = batch.clone();
    selected.pooled_indices = pooled.to_vec();
    selected.raw_indices = raw.to_vec();
    selected
}

// Same physical token batches isolate packing, selection and arena lifetime
// from attention/GEMM differences that arise when physical batching changes.
fn verify_raw_and_mixed_readout(
    backend: &OxideBackend,
    a: &Batch,
    b: &Batch,
    pooled_a: &PooledBits,
    pooled_b: &PooledBits,
) {
    let sync = |batch: &Batch| readout_bits(&backend.embed(batch.clone()).unwrap(), batch);
    let raw_a = selected_readout(a, &[], &[1, 0]);
    let raw_b = selected_readout(b, &[], &[2, 1, 0]);
    let raw_a_bits = sync(&raw_a);
    let raw_b_bits = sync(&raw_b);
    assert!(raw_a_bits.0.is_empty());
    assert!(raw_b_bits.0.is_empty());
    assert_eq!(sync(&selected_readout(a, &[], &[1, 1, 0])), raw_a_bits);
    assert_eq!(sync(&selected_readout(a, &[], &[0, 1, 1])), raw_a_bits);

    let mixed_a = selected_readout(a, &[0], &[1, 1]);
    let mixed_b = selected_readout(b, &[2, 0], &[1]);
    let mixed_a_bits = sync(&mixed_a);
    let mixed_b_bits = sync(&mixed_b);
    assert_eq!(mixed_a_bits.0.get(&0), pooled_a.get(&0));
    assert_eq!(mixed_a_bits.1.get(&1), raw_a_bits.1.get(&1));
    assert_eq!(&mixed_b_bits.0, pooled_b);
    assert_eq!(mixed_b_bits.1.get(&1), raw_b_bits.1.get(&1));
    for types in [TypeIds::Ones, TypeIds::Mixed] {
        assert_eq!(sync(&with_type_ids(&raw_a, types)), raw_a_bits);
        assert_eq!(sync(&with_type_ids(&mixed_a, types)), mixed_a_bits);
    }

    let held_raw = backend.submit_embed(raw_a.clone()).unwrap();
    let held_mixed = backend.submit_embed(mixed_b.clone()).unwrap();
    assert_deferred(&held_raw, "raw A held across mixed B");
    assert_deferred(&held_mixed, "mixed B held across raw A");
    assert_eq!(
        readout_bits(&held_mixed.complete().unwrap(), &mixed_b),
        mixed_b_bits
    );
    assert_eq!(
        readout_bits(&held_raw.complete().unwrap(), &raw_a),
        raw_a_bits
    );
    assert_eq!(
        normalize(&backend.embed(a.clone()).unwrap(), SELECT_A, "raw recovery"),
        *pooled_a
    );
}

fn run_pool(path: &Path, pool: Pool) {
    validate_model_profile(path);
    let label = format!("pool {pool}");
    let backend = OxideBackend::new(path, "bfloat16", ModelType::Embedding(pool), 0)
        .unwrap_or_else(|error| panic!("{label}: failed to create GPU 0 backend: {error:?}"));
    backend
        .health()
        .unwrap_or_else(|error| panic!("{label}: backend health check failed: {error:?}"));

    let a = batch_a();
    let b = batch_b();
    let c = batch_c();
    let d = batch_d();
    assert_eq!(a.input_ids.len(), 10);
    assert_eq!(b.input_ids.len(), 23);
    assert_eq!(c.input_ids.len(), 33);
    assert_eq!(d.input_ids.len(), 74);
    assert_eq!(
        rows_bucket(a.input_ids.len()),
        rows_bucket(b.input_ids.len())
    );
    assert_ne!(
        rows_bucket(a.input_ids.len()),
        rows_bucket(d.input_ids.len())
    );

    let base_a = sync_baseline(&backend, &a, SELECT_A, &format!("{label} A"));
    let base_b = sync_baseline(&backend, &b, SELECT_B, &format!("{label} B"));
    let base_c = sync_baseline(&backend, &c, SELECT_C, &format!("{label} C"));
    let base_d = sync_baseline(&backend, &d, SELECT_D, &format!("{label} D"));
    verify_raw_and_mixed_readout(&backend, &a, &b, &base_a, &base_b);

    let mut duplicate_a = a.clone();
    duplicate_a.pooled_indices = vec![1, 1, 0];
    let held_duplicate = backend
        .submit_embed(duplicate_a)
        .unwrap_or_else(|error| panic!("{label}: duplicate-index submit failed: {error:?}"));
    assert_deferred(&held_duplicate, &format!("{label} duplicate indices"));
    let duplicate = complete_normalized(
        held_duplicate,
        &[1, 1, 0],
        &format!("{label} duplicate indices"),
    );
    assert_eq!(
        duplicate, base_a,
        "{label}: duplicate/reordered A indices changed pooled values"
    );
    assert!(
        !base_b.contains_key(&1),
        "{label}: B leaked an output for unselected sequence 1"
    );

    for types in [TypeIds::Ones, TypeIds::Mixed] {
        for (batch, selected, baseline, name) in
            [(&a, SELECT_A, &base_a, "A"), (&b, SELECT_B, &base_b, "B")]
        {
            let embeddings = backend
                .embed(with_type_ids(batch, types))
                .unwrap_or_else(|error| {
                    panic!("{label}: {name} token_type_ids {types:?} failed: {error:?}")
                });
            let actual = normalize(
                &embeddings,
                selected,
                &format!("{label} {name} token_type_ids {types:?}"),
            );
            assert_eq!(
                &actual, baseline,
                "{label}: {name} token_type_ids {types:?} changed embedding bits"
            );
        }
    }

    let held_a = backend
        .submit_embed(a.clone())
        .unwrap_or_else(|error| panic!("{label}: A submit failed: {error:?}"));
    let held_b = backend
        .submit_embed(b.clone())
        .unwrap_or_else(|error| panic!("{label}: B submit failed: {error:?}"));
    assert_deferred(&held_a, &format!("{label} A first inflight"));
    assert_deferred(&held_b, &format!("{label} B second inflight"));
    let deferred_b = complete_normalized(held_b, SELECT_B, &format!("{label} deferred B first"));
    // Completing B must return its individual slot even while A is held.
    let held_c = backend.submit_embed(c.clone()).unwrap();
    assert_deferred(&held_c, &format!("{label} replacement C while A is held"));
    let replacement_c = complete_normalized(held_c, SELECT_C, &format!("{label} replacement C"));
    assert_eq!(replacement_c, base_c);
    let deferred_a = complete_normalized(held_a, SELECT_A, &format!("{label} deferred A second"));
    assert_eq!(deferred_b, base_b, "{label}: deferred B mismatch");
    assert_eq!(deferred_a, base_a, "{label}: deferred A mismatch");

    let held_a = backend
        .submit_embed(a.clone())
        .unwrap_or_else(|error| panic!("{label}: changed-bucket A submit failed: {error:?}"));
    let held_d = backend
        .submit_embed(d.clone())
        .unwrap_or_else(|error| panic!("{label}: changed-bucket D submit failed: {error:?}"));
    assert_deferred(&held_a, &format!("{label} changed-bucket A"));
    assert_deferred(&held_d, &format!("{label} changed-bucket D"));
    let deferred_d = complete_normalized(held_d, SELECT_D, &format!("{label} deferred D first"));
    let deferred_a = complete_normalized(held_a, SELECT_A, &format!("{label} deferred A second"));
    assert_eq!(
        deferred_d, base_d,
        "{label}: changed-bucket deferred D mismatch"
    );
    assert_eq!(
        deferred_a, base_a,
        "{label}: changed-bucket deferred A mismatch"
    );

    drop(
        backend
            .submit_embed(d.clone())
            .unwrap_or_else(|error| panic!("{label}: pending D submit failed: {error:?}")),
    );
    let retried_d = complete_normalized(
        backend
            .submit_embed(d.clone())
            .unwrap_or_else(|error| panic!("{label}: retried D submit failed: {error:?}")),
        SELECT_D,
        &format!("{label} retried D"),
    );
    let retried_a = complete_normalized(
        backend
            .submit_embed(a.clone())
            .unwrap_or_else(|error| panic!("{label}: retried A submit failed: {error:?}")),
        SELECT_A,
        &format!("{label} retried A"),
    );
    assert_eq!(retried_d, base_d, "{label}: retried D mismatch");
    assert_eq!(retried_a, base_a, "{label}: retried A mismatch");

    for cycle in 0..3 {
        let held_a = backend
            .submit_embed(a.clone())
            .unwrap_or_else(|error| panic!("{label}: cycle {cycle} A submit failed: {error:?}"));
        let held_b = backend
            .submit_embed(b.clone())
            .unwrap_or_else(|error| panic!("{label}: cycle {cycle} B submit failed: {error:?}"));
        assert_deferred(&held_a, &format!("{label} cycle {cycle} A"));
        assert_deferred(&held_b, &format!("{label} cycle {cycle} B"));
        let churn_b = complete_normalized(
            held_b,
            SELECT_B,
            &format!("{label} cycle {cycle} B complete"),
        );
        let churn_a = complete_normalized(
            held_a,
            SELECT_A,
            &format!("{label} cycle {cycle} A complete"),
        );
        assert_eq!(churn_b, base_b, "{label}: cycle {cycle} B mismatch");
        assert_eq!(churn_a, base_a, "{label}: cycle {cycle} A mismatch");
    }

    let sequential: [NamedBaseline<'_>; 5] = [
        (&a, SELECT_A, &base_a, "A"),
        (&b, SELECT_B, &base_b, "B"),
        (&c, SELECT_C, &base_c, "C"),
        (&d, SELECT_D, &base_d, "D"),
        (&a, SELECT_A, &base_a, "A"),
    ];
    for (batch, selected, baseline, name) in sequential {
        let actual = complete_normalized(
            backend.submit_embed(batch.clone()).unwrap_or_else(|error| {
                panic!("{label}: sequential {name} submit failed: {error:?}")
            }),
            selected,
            &format!("{label} sequential {name}"),
        );
        assert_eq!(&actual, baseline, "{label}: sequential {name} mismatch");
    }

    let mut invalids: Vec<(&str, Batch)> = Vec::new();
    let mut invalid = a.clone();
    invalid.input_ids[0] = INVALID_TOKEN_ID;
    invalids.push(("bad token id 50368", invalid));
    let mut invalid = a.clone();
    invalid.input_ids[0] = u32::MAX;
    invalids.push(("bad token id u32::MAX", invalid));
    let mut invalid = a.clone();
    invalid.position_ids[0] = MAX_LENGTH;
    invalids.push(("bad position 8192", invalid));
    let mut invalid = a.clone();
    invalid.position_ids[0] = u32::MAX;
    invalids.push(("bad position u32::MAX", invalid));
    let mut invalid = a.clone();
    invalid.position_ids.pop();
    invalids.push(("position length mismatch", invalid));
    let mut invalid = a.clone();
    invalid.token_type_ids.pop();
    invalids.push(("token type length mismatch", invalid));
    let mut invalid = a.clone();
    invalid.cumulative_seq_lengths[0] = 1;
    invalids.push(("malformed cumulative", invalid));
    let mut invalid = a.clone();
    let last = invalid.cumulative_seq_lengths.len() - 1;
    invalid.cumulative_seq_lengths.swap(last, last - 1);
    invalids.push(("nonincreasing cumulative", invalid));
    let mut invalid = a.clone();
    *invalid.cumulative_seq_lengths.last_mut().unwrap() += 1;
    invalids.push(("cumulative last bound mismatch", invalid));
    let mut invalid = a.clone();
    invalid.pooled_indices[0] = u32::MAX;
    invalids.push(("invalid pooled index u32::MAX", invalid));

    for (name, invalid) in invalids {
        assert_rejected(
            &backend,
            invalid,
            SELECT_A,
            &base_a,
            &format!("{label} {name}"),
        );
    }

    let oversized = oversized_batch();
    let total = oversized.input_ids.len();
    assert_eq!(total, (1usize << 20) + 1);
    assert!(total.checked_mul(2304).unwrap() <= u32::MAX as usize);
    assert!(
        total.next_power_of_two().checked_mul(2304).unwrap() > u32::MAX as usize,
        "{label}: oversized fixture does not exceed the device bucket bound"
    );
    let error = match backend.submit_embed(oversized) {
        Ok(_) => panic!("{label}: oversized gated bucket was admitted"),
        Err(error) => error,
    };
    assert!(
        matches!(&error, BackendError::Inference(_)),
        "{label}: expected CPU admission rejection, got {error:?}"
    );
    assert!(
        !matches!(&error, BackendError::Unhealthy),
        "{label}: oversized rejection poisoned backend health"
    );
    backend
        .health()
        .unwrap_or_else(|error| panic!("{label}: unhealthy after oversized rejection: {error:?}"));
    let after_oversized = backend
        .embed(a.clone())
        .unwrap_or_else(|error| panic!("{label}: post-oversized A embed failed: {error:?}"));
    let after_oversized = normalize(
        &after_oversized,
        SELECT_A,
        &format!("{label} post-oversized A"),
    );
    assert_eq!(
        after_oversized, base_a,
        "{label}: post-oversized A mismatch"
    );

    let held_a = backend
        .submit_embed(a.clone())
        .unwrap_or_else(|error| panic!("{label}: pre-drop A submit failed: {error:?}"));
    let held_b = backend
        .submit_embed(b.clone())
        .unwrap_or_else(|error| panic!("{label}: pre-drop B submit failed: {error:?}"));
    assert_deferred(&held_a, &format!("{label} pre-drop A"));
    assert_deferred(&held_b, &format!("{label} pre-drop B"));
    drop(backend);
    drop(held_a);
    let survived_b =
        complete_normalized(held_b, SELECT_B, &format!("{label} B after backend drop"));
    assert_eq!(
        survived_b, base_b,
        "{label}: B did not survive backend drop while pending"
    );
}

#[test]
#[ignore = "Requires NVIDIA GPU 0, oxide kernels, and INFINITY_OXIDE_TEST_MODERN_ENCODER_MODEL"]
fn modern_encoder_deferred_shape_changes_and_rejected_request_recovery() {
    let path = model_path();
    validate_model_profile(&path);
    for pool in [Pool::Cls, Pool::Mean, Pool::LastToken] {
        run_pool(&path, pool);
    }
}

#[test]
fn fixture_constructor_preserves_shapes_positions_types_and_selection() {
    let a = batch_a();
    let b = batch_b();
    let c = batch_c();
    let d = batch_d();

    assert_eq!(a.cumulative_seq_lengths, vec![0, 7, 10]);
    assert_eq!(&a.position_ids[..7], &[0, 1, 2, 3, 4, 5, 6]);
    assert_eq!(&a.position_ids[7..], &[0, 1, 2]);
    assert!(a
        .input_ids
        .iter()
        .all(|&token| (1..=TOKEN_ID_LIMIT).contains(&token)));
    assert!(a.token_type_ids.iter().all(|&token_type| token_type == 0));
    assert_eq!(a.pooled_indices.as_slice(), SELECT_A);
    assert!(a.raw_indices.is_empty());
    assert_eq!(a.max_length, 7);
    assert_eq!(
        rows_bucket(a.input_ids.len()),
        rows_bucket(b.input_ids.len())
    );

    assert_eq!(b.cumulative_seq_lengths, vec![0, 1, 18, 23]);
    assert_eq!(&b.position_ids[..1], &[0]);
    assert_eq!(
        b.position_ids[1..18].to_vec(),
        (0..17).collect::<Vec<u32>>()
    );
    assert_eq!(&b.position_ids[18..], &[0, 1, 2, 3, 4]);
    let b_keys: BTreeSet<u32> = b.pooled_indices.iter().copied().collect();
    assert_eq!(b_keys, BTreeSet::from([0, 2]));
    assert!(!b_keys.contains(&1));

    assert_eq!(c.input_ids.len(), 33);
    assert_eq!(d.input_ids.len(), 74);
    assert_eq!(d.cumulative_seq_lengths, vec![0, 2, 73, 74]);
    assert_eq!(
        d.pooled_indices.iter().copied().collect::<BTreeSet<_>>(),
        BTreeSet::from([0, 1, 2])
    );

    let mut duplicate = a.clone();
    duplicate.pooled_indices = vec![1, 1, 0];
    assert_eq!(
        duplicate
            .pooled_indices
            .iter()
            .copied()
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([0, 1])
    );
}

#[test]
fn oversized_fixture_is_cpu_constructible_but_exceeds_device_bucket() {
    let batch = oversized_batch();
    let total = batch.input_ids.len();
    let seq_lens: Vec<usize> = batch
        .cumulative_seq_lengths
        .windows(2)
        .map(|window| (window[1] - window[0]) as usize)
        .collect();

    assert_eq!(total, (1usize << 20) + 1);
    assert!(seq_lens.len() > 1);
    assert!(seq_lens[..seq_lens.len() - 1]
        .iter()
        .all(|&len| len == MAX_LENGTH as usize));
    assert_eq!(*seq_lens.last().unwrap(), 1);
    assert_eq!(batch.max_length, MAX_LENGTH);
    assert_eq!(batch.pooled_indices, vec![0]);
    assert!(batch.input_ids.iter().all(|&token| token == 1));
    assert!(batch
        .token_type_ids
        .iter()
        .all(|&token_type| token_type == 0));
    assert!(total.checked_mul(2304).unwrap() <= u32::MAX as usize);
    assert!(total.next_power_of_two().checked_mul(2304).unwrap() > u32::MAX as usize);
}
