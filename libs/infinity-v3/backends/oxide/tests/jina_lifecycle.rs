// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use infinity_backend_oxide::OxideBackend;
use infinity_v3_backend_core::{
    Backend, BackendError, Batch, Embedding, Embeddings, Inference, ModelType, Pool,
};

type PooledBits = BTreeMap<u32, Vec<u32>>;
type RawBits = BTreeMap<u32, Vec<Vec<u32>>>;

const HIDDEN_SIZE: usize = 512;
const INTERMEDIATE_SIZE: usize = 2048;
const NUM_LAYERS: usize = 4;
const ATTENTION_HEADS: usize = 8;
const MAX_LENGTH: u32 = 8192;
const INVALID_TOKEN_ID: u32 = 30528;
const TOKEN_ID_LIMIT: u32 = 30527;
const GATED_SCRATCH_ELEMENTS_PER_TOKEN: usize = 4096;
const OVERSIZED_TOTAL: usize = (1usize << 20) - 1;
const SELECT_A: &[u32] = &[1, 0];
const SELECT_B: &[u32] = &[2, 0];
const SELECT_C: &[u32] = &[0];
const SELECT_D: &[u32] = &[1, 2, 0];

const JINA_CONFIG: &str = r#"{
  "architectures": ["JinaBertModel"],
  "_name_or_path": "jinaai/jina-bert-implementation",
  "feed_forward_type": "geglu",
  "hidden_act": "gelu",
  "layer_norm_eps": 1e-12,
  "hidden_size": 512,
  "intermediate_size": 2048,
  "max_position_embeddings": 8192,
  "model_type": "bert",
  "num_attention_heads": 8,
  "num_hidden_layers": 4,
  "position_embedding_type": "alibi",
  "type_vocab_size": 2,
  "vocab_size": 30528
}"#;

#[derive(Clone, Copy, Debug)]
enum TypeIds {
    Zero,
    Ones,
    Mixed,
}

fn model_path() -> PathBuf {
    std::env::var_os("INFINITY_OXIDE_TEST_JINA_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            panic!(
                "set INFINITY_OXIDE_TEST_JINA_MODEL to a local Jina small model directory; \
                 this GPU lifecycle test does not skip silently when the model is absent"
            )
        })
}

fn validate(path: &Path) {
    let config_path = path.join("config.json");
    let config: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&config_path)
            .unwrap_or_else(|error| panic!("read {}: {error}", config_path.display())),
    )
    .unwrap();
    validate_config(&config);
}

fn validate_config(config: &serde_json::Value) {
    let dim = |field: &str| {
        config
            .get(field)
            .and_then(serde_json::Value::as_u64)
            .unwrap_or_else(|| panic!("missing Jina config field {field}"))
    };
    assert_eq!(config["model_type"].as_str(), Some("bert"));
    assert_eq!(config["position_embedding_type"].as_str(), Some("alibi"));
    assert_eq!(config["feed_forward_type"].as_str(), Some("geglu"));
    assert_eq!(config["hidden_act"].as_str(), Some("gelu"));
    assert_eq!(config["layer_norm_eps"].as_f64(), Some(1e-12));
    let name = config["_name_or_path"].as_str().unwrap_or("");
    let auto = config["auto_map"]["AutoConfig"].as_str().unwrap_or("");
    assert!(name.contains("jina-bert-implementation") || auto.contains("jina-bert-implementation"));
    assert!(
        !name.contains("jina-bert-v2-qk-post-norm") && !auto.contains("jina-bert-v2-qk-post-norm")
    );
    assert_eq!(
        (
            dim("hidden_size"),
            dim("intermediate_size"),
            dim("num_hidden_layers"),
            dim("num_attention_heads"),
            dim("max_position_embeddings"),
            dim("type_vocab_size")
        ),
        (
            HIDDEN_SIZE as u64,
            INTERMEDIATE_SIZE as u64,
            NUM_LAYERS as u64,
            ATTENTION_HEADS as u64,
            MAX_LENGTH as u64,
            2
        )
    );
    assert_eq!(dim("vocab_size"), INVALID_TOKEN_ID as u64);
}

fn type_value(kind: TypeIds, index: usize) -> u32 {
    match kind {
        TypeIds::Zero => 0,
        TypeIds::Ones => 1,
        TypeIds::Mixed => (index % 2) as u32,
    }
}

macro_rules! batch {
    ($ids:expr, $types:expr, $pos:expr, $cum:expr, $max:expr, $pooled:expr) => {
        Batch {
            multimodal: Vec::new(),
            input_ids: $ids,
            token_type_ids: $types,
            position_ids: $pos,
            cumulative_seq_lengths: $cum,
            max_length: $max,
            pooled_indices: $pooled,
            raw_indices: Vec::new(),
            compact_input_ids: None,
            compact_position_ids: None,
            scatter_unfold: None,
            fold_gather: None,
            tokens: Vec::new(),
            offsets: Vec::new(),
        }
    };
}

fn make_batch(seq_lens: &[usize], pooled_indices: &[u32], types: TypeIds) -> Batch {
    let total: usize = seq_lens.iter().sum();
    let (mut input_ids, mut token_type_ids, mut position_ids) = (
        Vec::with_capacity(total),
        Vec::with_capacity(total),
        Vec::with_capacity(total),
    );
    let (mut cumulative, mut flat, mut max_length) = (vec![0], 0usize, 0u32);
    for &len in seq_lens {
        assert!(len > 0 && len <= MAX_LENGTH as usize);
        max_length = max_length.max(len as u32);
        for position in 0..len {
            input_ids.push(((flat % TOKEN_ID_LIMIT as usize) + 1) as u32);
            token_type_ids.push(type_value(types, flat));
            position_ids.push(position as u32);
            flat += 1;
        }
        cumulative.push(flat as u32);
    }
    batch!(
        input_ids,
        token_type_ids,
        position_ids,
        cumulative,
        max_length,
        pooled_indices.to_vec()
    )
}

fn fixtures() -> (Batch, Batch, Batch, Batch) {
    (
        make_batch(&[7, 3], SELECT_A, TypeIds::Zero),
        make_batch(&[1, 17, 5], SELECT_B, TypeIds::Zero),
        make_batch(&[33], SELECT_C, TypeIds::Zero),
        make_batch(&[2, 71, 1], SELECT_D, TypeIds::Zero),
    )
}

fn with_type_ids(batch: &Batch, types: TypeIds) -> Batch {
    let mut changed = batch.clone();
    for (index, value) in changed.token_type_ids.iter_mut().enumerate() {
        *value = type_value(types, index);
    }
    changed
}

fn oversized_batch() -> Batch {
    let mut cumulative = vec![0u32];
    let mut remaining = OVERSIZED_TOTAL;
    while remaining > 0 {
        let chunk = remaining.min(MAX_LENGTH as usize);
        let last = *cumulative.last().unwrap() as usize;
        cumulative.push((last + chunk) as u32);
        remaining -= chunk;
    }
    let mut input_ids = Vec::with_capacity(OVERSIZED_TOTAL);
    let mut token_type_ids = Vec::with_capacity(OVERSIZED_TOTAL);
    let mut position_ids = Vec::with_capacity(OVERSIZED_TOTAL);
    for index in 0..OVERSIZED_TOTAL {
        input_ids.push(((index % TOKEN_ID_LIMIT as usize) + 1) as u32);
        token_type_ids.push(0);
        position_ids.push((index % MAX_LENGTH as usize) as u32);
    }
    batch!(
        input_ids,
        token_type_ids,
        position_ids,
        cumulative,
        MAX_LENGTH,
        vec![0]
    )
}

fn norm(embeddings: &Embeddings, selected: &[u32]) -> PooledBits {
    let expected: BTreeSet<u32> = selected.iter().copied().collect();
    assert_eq!(embeddings.len(), expected.len());
    let mut out = BTreeMap::new();
    for (&key, embedding) in embeddings.iter() {
        let key = u32::try_from(key).unwrap();
        assert!(expected.contains(&key));
        let Embedding::Pooled(values) = embedding else {
            panic!("key {key} returned raw embeddings")
        };
        assert_eq!(values.len(), HIDDEN_SIZE);
        assert!(values.iter().all(|value| value.is_finite()));
        let bits = values.iter().map(|value| value.to_bits()).collect();
        assert!(out.insert(key, bits).is_none());
    }
    assert_eq!(out.keys().copied().collect::<BTreeSet<_>>(), expected);
    out
}

fn baseline(backend: &OxideBackend, batch: &Batch, selected: &[u32]) -> PooledBits {
    let first = norm(&backend.embed(batch.clone()).unwrap(), selected);
    let second = norm(&backend.embed(batch.clone()).unwrap(), selected);
    assert_eq!(first, second, "baselines are not bit exact");
    first
}

fn finish(inference: Inference<Embeddings>, selected: &[u32]) -> PooledBits {
    norm(&inference.complete().unwrap(), selected)
}

fn deferred(backend: &OxideBackend, batch: &Batch) -> Inference<Embeddings> {
    let inference = backend.submit_embed(batch.clone()).unwrap();
    assert!(matches!(&inference, Inference::Deferred(_)));
    inference
}

fn seq(backend: &OxideBackend, batch: &Batch, selected: &[u32]) -> PooledBits {
    finish(backend.submit_embed(batch.clone()).unwrap(), selected)
}

fn reject(backend: &OxideBackend, invalid: Batch, baseline: &PooledBits, context: &str) {
    let Err(error) = backend.submit_embed(invalid) else {
        panic!("{context}: invalid batch admitted")
    };
    assert!(matches!(&error, BackendError::Inference(_)));
    assert!(!matches!(&error, BackendError::Unhealthy));
    backend.health().unwrap();
    let (a, _, _, _) = fixtures();
    assert_eq!(&norm(&backend.embed(a).unwrap(), SELECT_A), baseline);
}

fn rows_bucket(rows: usize) -> usize {
    rows.max(32).next_power_of_two()
}

fn readout_bits(embeddings: &Embeddings, batch: &Batch) -> (PooledBits, RawBits) {
    let pkeys: BTreeSet<u32> = batch.pooled_indices.iter().copied().collect();
    let rkeys: BTreeSet<u32> = batch.raw_indices.iter().copied().collect();
    assert!(pkeys.is_disjoint(&rkeys));
    assert_eq!(embeddings.len(), pkeys.len() + rkeys.len());
    let (mut pooled, mut raw) = (BTreeMap::new(), BTreeMap::new());
    for (&key, embedding) in embeddings.iter() {
        let key = u32::try_from(key).unwrap();
        let rows: Vec<&[f32]> = match embedding {
            Embedding::Pooled(row) => {
                assert!(pkeys.contains(&key));
                vec![row]
            }
            Embedding::All(rows) => {
                assert!(rkeys.contains(&key));
                let (start, end) = (
                    batch.cumulative_seq_lengths[key as usize],
                    batch.cumulative_seq_lengths[key as usize + 1],
                );
                assert_eq!(rows.len(), (end - start) as usize);
                rows.iter().map(Vec::as_slice).collect()
            }
        };
        let bits: Vec<Vec<u32>> = rows
            .iter()
            .map(|row| {
                assert_eq!(row.len(), HIDDEN_SIZE);
                assert!(row.iter().all(|value| value.is_finite()));
                row.iter().map(|value| value.to_bits()).collect()
            })
            .collect();
        if pkeys.contains(&key) {
            assert!(pooled.insert(key, bits[0].clone()).is_none());
        } else {
            assert!(raw.insert(key, bits).is_none());
        }
    }
    assert_eq!(
        (
            pooled.keys().copied().collect::<BTreeSet<_>>(),
            raw.keys().copied().collect::<BTreeSet<_>>()
        ),
        (pkeys, rkeys)
    );
    (pooled, raw)
}

fn selected(batch: &Batch, pooled: &[u32], raw: &[u32]) -> Batch {
    let mut selected = batch.clone();
    selected.pooled_indices = pooled.to_vec();
    selected.raw_indices = raw.to_vec();
    selected
}

fn mutate(batch: &Batch, change: impl FnOnce(&mut Batch)) -> Batch {
    let mut changed = batch.clone();
    change(&mut changed);
    changed
}

fn verify_raw_and_mixed(
    backend: &OxideBackend,
    a: &Batch,
    b: &Batch,
    pa: &PooledBits,
    pb: &PooledBits,
) {
    let sync = |batch: &Batch| readout_bits(&backend.embed(batch.clone()).unwrap(), batch);
    let raw_a = selected(a, &[], &[1, 0]);
    let raw_b = selected(b, &[], &[2, 1, 0]);
    let raw_a_bits = sync(&raw_a);
    let raw_b_bits = sync(&raw_b);
    assert!(raw_a_bits.0.is_empty() && raw_b_bits.0.is_empty());
    let keys = |bits: &RawBits| bits.keys().copied().collect::<BTreeSet<_>>();
    assert_eq!(keys(&raw_a_bits.1), BTreeSet::from([0, 1]));
    assert_eq!(keys(&raw_b_bits.1), BTreeSet::from([0, 1, 2]));
    assert_eq!(sync(&selected(a, &[], &[1, 1, 0])), raw_a_bits);
    assert_eq!(sync(&selected(a, &[], &[0, 1, 1])), raw_a_bits);
    assert_eq!(sync(&selected(b, &[], &[2, 1, 0, 2])), raw_b_bits);

    let mixed_a = selected(a, &[0], &[1, 1]);
    let mixed_b = selected(b, &[2, 0], &[1]);
    let mixed_a_bits = sync(&mixed_a);
    let mixed_b_bits = sync(&mixed_b);
    assert_eq!(mixed_a_bits.0.get(&0), pa.get(&0));
    assert_eq!(mixed_a_bits.1.get(&1), raw_a_bits.1.get(&1));
    assert_eq!(&mixed_b_bits.0, pb);
    assert_eq!(mixed_b_bits.1.get(&1), raw_b_bits.1.get(&1));
    assert_eq!(sync(&selected(a, &[0, 0], &[1, 1])), mixed_a_bits);
    assert_eq!(sync(&selected(b, &[0, 2], &[1])), mixed_b_bits);

    let held_raw = deferred(backend, &raw_a);
    let held_mixed = deferred(backend, &mixed_b);
    assert_eq!(
        readout_bits(&held_mixed.complete().unwrap(), &mixed_b),
        mixed_b_bits
    );
    assert_eq!(
        readout_bits(&held_raw.complete().unwrap(), &raw_a),
        raw_a_bits
    );

    let (typed_a, typed_b) = (
        sync(&with_type_ids(a, TypeIds::Ones)),
        sync(&with_type_ids(b, TypeIds::Ones)),
    );
    let (typed_raw_a, typed_raw_b) = (
        sync(&with_type_ids(&raw_a, TypeIds::Ones)),
        sync(&with_type_ids(&raw_b, TypeIds::Ones)),
    );
    let (typed_mixed_a, typed_mixed_b) = (
        sync(&with_type_ids(&mixed_a, TypeIds::Ones)),
        sync(&with_type_ids(&mixed_b, TypeIds::Ones)),
    );
    assert_ne!(&typed_a.0, pa);
    assert_ne!(&typed_raw_a, &raw_a_bits);
    assert_eq!(typed_mixed_a.0.get(&0), typed_a.0.get(&0));
    assert_eq!(typed_mixed_a.1.get(&1), typed_raw_a.1.get(&1));
    assert_eq!(&typed_mixed_b.0, &typed_b.0);
    assert_eq!(typed_mixed_b.1.get(&1), typed_raw_b.1.get(&1));
}

fn verify_positions_are_ignored(
    backend: &OxideBackend,
    a: &Batch,
    b: &Batch,
    base_a: &PooledBits,
    base_b: &PooledBits,
) {
    let all_max = mutate(a, |batch| {
        for position in &mut batch.position_ids {
            *position = MAX_LENGTH - 1;
        }
    });
    let reversed = mutate(a, |batch| batch.position_ids.reverse());
    let reinterpreted_b = mutate(b, |batch| {
        for (index, position) in batch.position_ids.iter_mut().enumerate() {
            *position = ((index * 37 + 8191) % MAX_LENGTH as usize) as u32;
        }
    });
    assert_ne!(all_max.position_ids, a.position_ids);
    assert_ne!(reversed.position_ids, a.position_ids);
    assert_ne!(reinterpreted_b.position_ids, b.position_ids);
    assert!(all_max
        .position_ids
        .iter()
        .all(|&position| position < MAX_LENGTH));
    assert!(reversed
        .position_ids
        .iter()
        .all(|&position| position < MAX_LENGTH));
    assert!(reinterpreted_b
        .position_ids
        .iter()
        .all(|&position| position < MAX_LENGTH));
    assert_eq!(&norm(&backend.embed(all_max).unwrap(), SELECT_A), base_a);
    assert_eq!(&norm(&backend.embed(reversed).unwrap(), SELECT_A), base_a);
    assert_eq!(
        &norm(&backend.embed(reinterpreted_b).unwrap(), SELECT_B),
        base_b
    );
}

fn verify_short_unequal_mixed(backend: &OxideBackend) {
    let e = make_batch(&[3, 1, 5, 2], &[2, 0], TypeIds::Mixed);
    assert_eq!(e.cumulative_seq_lengths, vec![0, 3, 4, 9, 11]);
    assert_eq!(e.max_length, 5);
    assert!(e.token_type_ids.contains(&0) && e.token_type_ids.contains(&1));
    let pooled = baseline(backend, &e, &[2, 0]);
    let raw = selected(&e, &[], &[3, 1, 3]);
    let raw_bits = readout_bits(&backend.embed(raw.clone()).unwrap(), &raw);
    assert_eq!(
        raw_bits.1.keys().copied().collect::<BTreeSet<_>>(),
        BTreeSet::from([1, 3])
    );
    let mixed = selected(&e, &[2, 2], &[0, 3]);
    let mixed_bits = readout_bits(&backend.embed(mixed.clone()).unwrap(), &mixed);
    assert_eq!(mixed_bits.0.get(&2), pooled.get(&2));
    assert_eq!(mixed_bits.1.get(&3), raw_bits.1.get(&3));
}

fn verify_full_length_cls(backend: &OxideBackend, a: &Batch, base_a: &PooledBits) {
    let long = make_batch(&[MAX_LENGTH as usize], &[0], TypeIds::Zero);
    assert_eq!(long.cumulative_seq_lengths, vec![0, MAX_LENGTH]);
    assert_eq!(long.input_ids.len(), MAX_LENGTH as usize);
    assert_eq!(long.max_length, MAX_LENGTH);
    let bits = norm(&backend.embed(long).unwrap(), &[0]);
    assert_eq!(bits.len(), 1);
    let vector = bits.get(&0).unwrap();
    assert_eq!(vector.len(), HIDDEN_SIZE);
    assert!(vector
        .iter()
        .all(|value| f32::from_bits(*value).is_finite()));
    assert_eq!(&norm(&backend.embed(a.clone()).unwrap(), SELECT_A), base_a);
}

fn invalid_batches(a: &Batch) -> Vec<(&'static str, Batch)> {
    vec![
        (
            "bad token 30528",
            mutate(a, |b| b.input_ids[0] = INVALID_TOKEN_ID),
        ),
        ("bad token MAX", mutate(a, |b| b.input_ids[0] = u32::MAX)),
        ("bad type 2", mutate(a, |b| b.token_type_ids[0] = 2)),
        (
            "bad type MAX",
            mutate(a, |b| b.token_type_ids[0] = u32::MAX),
        ),
        (
            "bad pos 8192",
            mutate(a, |b| b.position_ids[0] = MAX_LENGTH),
        ),
        ("bad pos MAX", mutate(a, |b| b.position_ids[0] = u32::MAX)),
        (
            "input len",
            mutate(a, |b| {
                b.input_ids.pop();
            }),
        ),
        (
            "pos len",
            mutate(a, |b| {
                b.position_ids.pop();
            }),
        ),
        (
            "type len",
            mutate(a, |b| {
                b.token_type_ids.pop();
            }),
        ),
        (
            "cum len",
            mutate(a, |b| {
                b.cumulative_seq_lengths.pop();
            }),
        ),
        ("cum zero", mutate(a, |b| b.cumulative_seq_lengths[0] = 1)),
        (
            "cum order",
            mutate(a, |b| {
                let n = b.cumulative_seq_lengths.len();
                b.cumulative_seq_lengths.swap(n - 1, n - 2);
            }),
        ),
        (
            "cum bound",
            mutate(a, |b| *b.cumulative_seq_lengths.last_mut().unwrap() += 1),
        ),
        ("bad pooled", mutate(a, |b| b.pooled_indices[0] = u32::MAX)),
        ("bad raw", mutate(a, |b| b.raw_indices.push(u32::MAX))),
    ]
}

// Exercise ABA while retaining at most the backend's two output leases.
fn deferred_aba_with_two_slots<L, O>(
    mut submit_a: impl FnMut() -> L,
    mut submit_b: impl FnMut() -> L,
    mut complete_a: impl FnMut(L) -> O,
    mut complete_b: impl FnMut(L) -> O,
) -> (O, O, O) {
    let first_a = submit_a();
    let held_b = submit_b();
    // Completion returns A's host slot before the next submission. B stays
    // retained across A's second enqueue, so this still checks real ABA reuse.
    let first_a = complete_a(first_a);
    let second_a = submit_a();
    let held_b = complete_b(held_b);
    let second_a = complete_a(second_a);
    (first_a, held_b, second_a)
}

#[test]
fn deferred_aba_respects_two_slot_backpressure_without_losing_overlap() {
    use std::{cell::RefCell, rc::Rc};
    #[derive(Default)]
    struct State {
        held: BTreeSet<char>,
        submitted_a: usize,
        max_live: usize,
        b_overlapped_a_again: bool,
    }
    struct Lease {
        label: char,
        state: Rc<RefCell<State>>,
    }
    impl Drop for Lease {
        fn drop(&mut self) {
            assert!(self.state.borrow_mut().held.remove(&self.label));
        }
    }
    let state = Rc::new(RefCell::new(State::default()));
    let submit = |label| {
        let mut s = state.borrow_mut();
        assert!(
            s.held.len() < 2,
            "third retained lease would block the actual backend"
        );
        if label == 'A' {
            s.submitted_a += 1;
            if s.submitted_a == 2 {
                s.b_overlapped_a_again = s.held.contains(&'B');
            }
        }
        assert!(s.held.insert(label));
        s.max_live = s.max_live.max(s.held.len());
        drop(s);
        Lease {
            label,
            state: state.clone(),
        }
    };
    let finish = |lease: Lease| {
        let label = lease.label;
        drop(lease);
        label
    };
    assert_eq!(
        deferred_aba_with_two_slots(|| submit('A'), || submit('B'), finish, finish),
        ('A', 'B', 'A')
    );
    let s = state.borrow();
    assert!(s.held.is_empty());
    assert_eq!(s.max_live, 2);
    assert!(s.b_overlapped_a_again);
}

fn run_pool(path: &Path, pool: Pool) {
    eprintln!("Jina lifecycle stage: init");
    validate(path);
    let label = format!("pool {pool}");
    let is_cls = matches!(pool, Pool::Cls);
    let backend = OxideBackend::new(path, "bfloat16", ModelType::Embedding(pool), 0).unwrap();
    backend.health().unwrap();
    let (a, b, c, d) = fixtures();
    assert_eq!(
        (
            a.input_ids.len(),
            b.input_ids.len(),
            c.input_ids.len(),
            d.input_ids.len()
        ),
        (10, 23, 33, 74)
    );
    assert_eq!(
        rows_bucket(a.input_ids.len()),
        rows_bucket(b.input_ids.len())
    );
    assert_ne!(
        rows_bucket(a.input_ids.len()),
        rows_bucket(d.input_ids.len())
    );

    let (base_a, base_b) = (
        baseline(&backend, &a, SELECT_A),
        baseline(&backend, &b, SELECT_B),
    );
    let (base_c, base_d) = (
        baseline(&backend, &c, SELECT_C),
        baseline(&backend, &d, SELECT_D),
    );
    eprintln!("Jina lifecycle stage: raw-mixed");
    verify_raw_and_mixed(&backend, &a, &b, &base_a, &base_b);
    verify_positions_are_ignored(&backend, &a, &b, &base_a, &base_b);
    verify_short_unequal_mixed(&backend);
    if is_cls {
        verify_full_length_cls(&backend, &a, &base_a);
    }

    for types in [TypeIds::Ones, TypeIds::Mixed] {
        for (batch, selected, reference) in [(&a, SELECT_A, &base_a), (&b, SELECT_B, &base_b)] {
            let first = norm(
                &backend.embed(with_type_ids(batch, types)).unwrap(),
                selected,
            );
            let second = norm(
                &backend.embed(with_type_ids(batch, types)).unwrap(),
                selected,
            );
            assert_eq!(first, second, "{label}: unstable {types:?}");
            assert_ne!(&first, reference, "{label}: unchanged {types:?}");
        }
    }
    assert_eq!(norm(&backend.embed(a.clone()).unwrap(), SELECT_A), base_a);

    eprintln!("Jina lifecycle stage: bounded-aba");
    let (aba_a, aba_b, aba_a_again) = deferred_aba_with_two_slots(
        || deferred(&backend, &a),
        || deferred(&backend, &b),
        |value| finish(value, SELECT_A),
        |value| finish(value, SELECT_B),
    );
    assert_eq!(
        (&aba_a, &aba_b, &aba_a_again),
        (&base_a, &base_b, &base_a),
        "{label}: deferred ABA"
    );

    let (held_a, held_b) = (deferred(&backend, &a), deferred(&backend, &b));
    let out_of_order_b = finish(held_b, SELECT_B);
    let held_c = deferred(&backend, &c);
    let while_a_held_c = finish(held_c, SELECT_C);
    let out_of_order_a = finish(held_a, SELECT_A);
    assert_eq!(
        (&out_of_order_b, &while_a_held_c, &out_of_order_a),
        (&base_b, &base_c, &base_a)
    );

    let (held_a, held_d) = (deferred(&backend, &a), deferred(&backend, &d));
    let changed_d = finish(held_d, SELECT_D);
    let changed_a = finish(held_a, SELECT_A);
    assert_eq!((&changed_d, &changed_a), (&base_d, &base_a));

    drop(deferred(&backend, &d));
    assert_eq!(
        (&seq(&backend, &d, SELECT_D), &seq(&backend, &a, SELECT_A)),
        (&base_d, &base_a)
    );

    for cycle in 0..3 {
        let (held_a, held_d) = (deferred(&backend, &a), deferred(&backend, &d));
        let (churn_d, churn_a) = (finish(held_d, SELECT_D), finish(held_a, SELECT_A));
        assert_eq!(
            (&churn_a, &churn_d),
            (&base_a, &base_d),
            "{label}: shape churn cycle {cycle}"
        );
    }

    eprintln!("Jina lifecycle stage: invalid-and-recovery");
    for (name, invalid) in invalid_batches(&a) {
        reject(&backend, invalid, &base_a, &format!("{label}: {name}"));
    }

    eprintln!("Jina lifecycle stage: oversized-recovery");
    let oversized = oversized_batch();
    let total = oversized.input_ids.len();
    assert_eq!(total, OVERSIZED_TOTAL);
    assert!(total * GATED_SCRATCH_ELEMENTS_PER_TOKEN <= u32::MAX as usize);
    assert!(total.next_power_of_two() * GATED_SCRATCH_ELEMENTS_PER_TOKEN > u32::MAX as usize);
    let Err(error) = backend.submit_embed(oversized) else {
        panic!("{label}: gated bucket admitted")
    };
    assert!(matches!(&error, BackendError::Inference(_)));
    assert!(!matches!(&error, BackendError::Unhealthy));
    backend.health().unwrap();
    assert_eq!(&norm(&backend.embed(a.clone()).unwrap(), SELECT_A), &base_a);
    assert_eq!(&norm(&backend.embed(d.clone()).unwrap(), SELECT_D), &base_d);
    let (recover_a, recover_d) = (deferred(&backend, &a), deferred(&backend, &d));
    assert_eq!(finish(recover_a, SELECT_A), base_a);
    assert_eq!(finish(recover_d, SELECT_D), base_d);

    eprintln!("Jina lifecycle stage: drop-owner");
    let (held_a, held_b) = (deferred(&backend, &a), deferred(&backend, &b));
    drop(backend);
    drop(held_a);
    assert_eq!(finish(held_b, SELECT_B), base_b);
}

#[test]
#[ignore = "Requires NVIDIA GPU 0, oxide kernels, and INFINITY_OXIDE_TEST_JINA_MODEL"]
fn jina_deferred_raw_mixed_aba_types_positions_and_recovery() {
    let path = model_path();
    for pool in [Pool::Cls, Pool::Mean, Pool::LastToken] {
        run_pool(&path, pool);
    }
}

#[test]
fn fixture_constructor_preserves_shapes_positions_types_and_selection() {
    let (a, b, c, d) = fixtures();
    assert_eq!(a.cumulative_seq_lengths, vec![0, 7, 10]);
    assert_eq!(&a.position_ids[..7], &[0, 1, 2, 3, 4, 5, 6]);
    assert_eq!(&a.position_ids[7..], &[0, 1, 2]);
    assert!(a.position_ids.iter().all(|&position| position < MAX_LENGTH));
    assert!(a
        .input_ids
        .iter()
        .all(|&token| (1..INVALID_TOKEN_ID).contains(&token)));
    assert!(
        a.token_type_ids.iter().all(|&token_type| token_type == 0)
            && a.raw_indices.is_empty()
            && a.max_length == 7
    );
    assert_eq!(a.pooled_indices.as_slice(), SELECT_A);
    assert_eq!(
        rows_bucket(a.input_ids.len()),
        rows_bucket(b.input_ids.len())
    );
    assert_ne!(
        rows_bucket(a.input_ids.len()),
        rows_bucket(d.input_ids.len())
    );

    assert_eq!(b.cumulative_seq_lengths, vec![0, 1, 18, 23]);
    assert_eq!(&b.position_ids[..1], &[0]);
    assert_eq!(
        b.position_ids[1..18].to_vec(),
        (0..17).collect::<Vec<u32>>()
    );
    assert_eq!(&b.position_ids[18..], &[0, 1, 2, 3, 4]);
    assert_eq!(
        b.pooled_indices.iter().copied().collect::<BTreeSet<_>>(),
        BTreeSet::from([0, 2])
    );

    assert_eq!((c.input_ids.len(), d.input_ids.len()), (33, 74));
    assert_eq!(
        (
            d.cumulative_seq_lengths.as_slice(),
            d.pooled_indices.iter().copied().collect::<BTreeSet<_>>()
        ),
        (&[0u32, 2, 73, 74][..], BTreeSet::from([0, 1, 2]))
    );

    let ones = with_type_ids(&a, TypeIds::Ones);
    let mixed = with_type_ids(&a, TypeIds::Mixed);
    assert!(ones
        .token_type_ids
        .iter()
        .all(|&token_type| token_type == 1));
    assert!(
        mixed
            .token_type_ids
            .iter()
            .all(|&token_type| token_type <= 1)
            && mixed.token_type_ids.contains(&1)
    );

    let at_upper_position_bound = mutate(&a, |batch| {
        for position in &mut batch.position_ids {
            *position = MAX_LENGTH - 1;
        }
    });
    assert!(at_upper_position_bound
        .position_ids
        .iter()
        .all(|&position| position == MAX_LENGTH - 1));

    let invalids = invalid_batches(&a);
    assert!(invalids.iter().any(|(name, _)| *name == "bad token 30528"));
    assert!(invalids.iter().any(|(name, _)| *name == "bad type 2"));
    assert!(invalids.iter().any(|(name, _)| *name == "bad pos 8192"));
    assert!(invalids.iter().any(|(name, _)| *name == "input len"));
    assert!(invalids.iter().any(|(name, _)| *name == "pos len"));
    assert!(invalids.iter().any(|(name, _)| *name == "type len"));
    assert!(invalids
        .iter()
        .any(|(_, batch)| batch.token_type_ids.contains(&u32::MAX)));
}

#[test]
fn oversized_fixture_is_cpu_constructible_but_exceeds_planner_bucket() {
    let batch = oversized_batch();
    let total = batch.input_ids.len();
    let seq_lens: Vec<usize> = batch
        .cumulative_seq_lengths
        .windows(2)
        .map(|window| (window[1] - window[0]) as usize)
        .collect();
    assert_eq!(total, OVERSIZED_TOTAL);
    assert_eq!(seq_lens.len(), 128);
    assert!(seq_lens[..seq_lens.len() - 1]
        .iter()
        .all(|&len| len == MAX_LENGTH as usize));
    assert_eq!(*seq_lens.last().unwrap(), MAX_LENGTH as usize - 1);
    assert_eq!(
        (batch.max_length, batch.pooled_indices.as_slice()),
        (MAX_LENGTH, &[0u32][..])
    );
    assert!(batch
        .input_ids
        .iter()
        .all(|&token| (1..INVALID_TOKEN_ID).contains(&token)));
    assert!(batch
        .token_type_ids
        .iter()
        .all(|&token_type| token_type == 0));
    assert!(batch
        .position_ids
        .iter()
        .all(|&position| position < MAX_LENGTH));
    assert!(total * GATED_SCRATCH_ELEMENTS_PER_TOKEN <= u32::MAX as usize);
    assert!((total + 1) * GATED_SCRATCH_ELEMENTS_PER_TOKEN > u32::MAX as usize);
    assert!(total.next_power_of_two() * GATED_SCRATCH_ELEMENTS_PER_TOKEN > u32::MAX as usize);
}

#[test]
fn actual_config_fixture_uses_native_jina_fields_without_cuda() {
    let config = serde_json::from_str(JINA_CONFIG).unwrap();
    validate_config(&config);
    assert_eq!(config["hidden_size"].as_u64(), Some(HIDDEN_SIZE as u64));
    assert_eq!(
        config["intermediate_size"].as_u64(),
        Some(INTERMEDIATE_SIZE as u64)
    );
    assert_eq!(
        config["num_hidden_layers"].as_u64(),
        Some(NUM_LAYERS as u64)
    );
    assert_eq!(
        config["max_position_embeddings"].as_u64(),
        Some(MAX_LENGTH as u64)
    );
    assert_eq!(config["vocab_size"].as_u64(), Some(INVALID_TOKEN_ID as u64));
}
