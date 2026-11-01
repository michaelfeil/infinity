// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
use infinity_backend_oxide::OxideBackend;
use infinity_v3_backend_core::{Backend, Batch, Embedding, Embeddings, ModelType, Pool};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Clone, Copy)]
enum RawPool {
    Mean,
    Cls,
    LastToken,
}
fn raw_batch(lengths: &[usize], pooled: Vec<u32>, raw: Vec<u32>) -> Batch {
    let mut input_ids = Vec::new();
    let mut position_ids = Vec::new();
    let mut cumulative_seq_lengths = vec![0u32];
    for &length in lengths {
        for position in 0..length {
            let token = input_ids.len() as u32;
            input_ids.push(20u32 + (token * 13 % 20000));
            position_ids.push(position as u32);
        }
        cumulative_seq_lengths.push(input_ids.len() as u32);
    }
    let tokens = input_ids.len();
    Batch {
        multimodal: vec![],
        input_ids,
        token_type_ids: vec![0; tokens],
        position_ids,
        cumulative_seq_lengths,
        max_length: lengths.iter().copied().max().unwrap_or(0) as u32,
        pooled_indices: pooled,
        raw_indices: raw,
        compact_input_ids: None,
        compact_position_ids: None,
        scatter_unfold: None,
        fold_gather: None,
        tokens: vec![],
        offsets: vec![],
    }
}
fn into_map(embeddings: Embeddings) -> BTreeMap<usize, Embedding> {
    let mut map = BTreeMap::new();
    for (index, embedding) in embeddings {
        assert!(map.insert(index, embedding).is_none());
    }
    map
}
fn all_map(embeddings: Embeddings) -> BTreeMap<usize, Vec<Vec<f32>>> {
    let mut map = BTreeMap::new();
    for (index, embedding) in embeddings {
        let Embedding::All(rows) = embedding else {
            panic!("sequence {index} must return raw token rows");
        };
        assert!(map.insert(index, rows).is_none());
    }
    map
}
fn assert_bits(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
}
fn assert_rows(actual: &[Vec<f32>], expected: &[Vec<f32>]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_bits(actual, expected);
    }
}
fn assert_all_map(
    actual: &BTreeMap<usize, Vec<Vec<f32>>>,
    expected: &BTreeMap<usize, Vec<Vec<f32>>>,
) {
    assert_eq!(actual.len(), expected.len());
    for (index, expected_rows) in expected {
        let actual_rows = actual
            .get(index)
            .unwrap_or_else(|| panic!("missing raw sequence {index}"));
        assert_rows(actual_rows, expected_rows);
    }
}
fn assert_width(
    outputs: &BTreeMap<usize, Vec<Vec<f32>>>,
    expected_lengths: &[(usize, usize)],
) -> usize {
    assert_eq!(outputs.len(), expected_lengths.len());
    let width = outputs
        .values()
        .flat_map(|sequence| sequence.iter())
        .map(Vec::len)
        .next()
        .expect("raw output must contain at least one row");
    assert!(width > 0);
    for &(index, tokens) in expected_lengths {
        let sequence = outputs
            .get(&index)
            .unwrap_or_else(|| panic!("missing raw sequence {index}"));
        assert_eq!(sequence.len(), tokens, "token count for sequence {index}");
        for row in sequence {
            assert_eq!(row.len(), width, "width for sequence {index}");
            assert!(
                row.iter().all(|value| value.is_finite()),
                "finite values for sequence {index}"
            );
        }
    }
    width
}
fn assert_mean(actual: &[f32], rows: &[Vec<f32>]) {
    assert!(!rows.is_empty());
    let mut mean = vec![0.0f64; actual.len()];
    for row in rows {
        assert_eq!(row.len(), mean.len());
        for (sum, value) in mean.iter_mut().zip(row) {
            *sum += f64::from(*value);
        }
    }
    for (actual, sum) in actual.iter().zip(mean) {
        assert!((f64::from(*actual) - sum / rows.len() as f64).abs() <= 1e-6);
    }
}

#[test]
#[ignore = "requires CUDA device 0, INFINITY_OXIDE_TEST_RAW_MODEL, and oxide kernels"]
fn raw_token_outputs_and_deferred_leases() {
    let model = std::env::var_os("INFINITY_OXIDE_TEST_RAW_MODEL")
        .expect("set INFINITY_OXIDE_TEST_RAW_MODEL to a checked model directory");
    let kind = match std::env::var("INFINITY_OXIDE_TEST_RAW_POOL").as_deref() {
        Ok("mean") | Err(_) => RawPool::Mean,
        Ok("cls") => RawPool::Cls,
        Ok("last-token") => RawPool::LastToken,
        Ok(value) => {
            panic!("INFINITY_OXIDE_TEST_RAW_POOL must be mean, cls or last-token, got {value}")
        }
    };
    let pool = match kind {
        RawPool::Mean => Pool::Mean,
        RawPool::Cls => Pool::Cls,
        RawPool::LastToken => Pool::LastToken,
    };
    let backend = OxideBackend::new(Path::new(&model), "bfloat16", ModelType::Embedding(pool), 0)
        .expect("construct oxide backend on device 0");
    backend.health().expect("healthy backend before requests");

    let baseline = all_map(
        backend
            .embed(raw_batch(&[3, 5, 2], vec![], vec![0, 1, 2]))
            .expect("baseline raw request"),
    );
    assert_width(&baseline, &[(0, 3), (1, 5), (2, 2)]);

    let mixed = into_map(
        backend
            .embed(raw_batch(&[3, 5, 2], vec![1], vec![2, 0, 2]))
            .expect("mixed pooled/raw request"),
    );
    assert_eq!(mixed.len(), 3);
    for index in [0usize, 2] {
        match mixed.get(&index) {
            Some(Embedding::All(rows)) => assert_rows(rows, &baseline[&index]),
            _ => panic!("raw sequence {index} must remain the raw variant"),
        }
    }
    let pooled_one = match mixed.get(&1) {
        Some(Embedding::Pooled(values)) => values,
        _ => panic!("sequence 1 must remain the pooled variant"),
    };
    let raw_one = &baseline[&1];
    match kind {
        RawPool::Cls => assert_bits(pooled_one, &raw_one[0]),
        RawPool::LastToken => assert_bits(pooled_one, raw_one.last().expect("nonempty sequence")),
        RawPool::Mean => assert_mean(pooled_one, raw_one),
    }

    let overlap = into_map(
        backend
            .embed(raw_batch(&[3, 5, 2], vec![0], vec![0]))
            .expect("overlapping pooled/raw request"),
    );
    assert_eq!(overlap.len(), 1);
    match overlap.get(&0) {
        Some(Embedding::All(rows)) => assert_rows(rows, &baseline[&0]),
        _ => panic!("raw selection must replace the overlapping pooled selection"),
    }

    let a = raw_batch(&[3, 5, 2], vec![], vec![0, 1, 2]);
    let b = raw_batch(&[7, 1], vec![], vec![0, 1]);
    let expected_a = all_map(backend.embed(a.clone()).expect("expected raw A"));
    let expected_b = all_map(backend.embed(b.clone()).expect("expected raw B"));
    assert_width(&expected_b, &[(0, 7), (1, 1)]);
    let held_a = backend.submit_embed(a.clone()).expect("submit raw A");
    let held_b = backend.submit_embed(b.clone()).expect("submit raw B");
    let actual_b = all_map(held_b.complete().expect("complete raw B first"));
    let actual_a = all_map(held_a.complete().expect("complete raw A second"));
    assert_all_map(&actual_a, &expected_a);
    assert_all_map(&actual_b, &expected_b);

    drop(
        backend
            .submit_embed(a.clone())
            .expect("pending dropped inference"),
    );
    backend
        .health()
        .expect("healthy after dropping one pending lease");
    let changed = raw_batch(&[4, 6, 2, 5], vec![], vec![0, 2, 3]);
    let expected_changed = all_map(
        backend
            .embed(changed.clone())
            .expect("expected changed shape"),
    );
    assert_width(&expected_changed, &[(0, 4), (2, 2), (3, 5)]);
    let actual_changed = all_map(backend.embed(changed).expect("changed-shape request"));
    assert_all_map(&actual_changed, &expected_changed);
    backend
        .health()
        .expect("healthy after changed-shape recovery");

    let invalid = raw_batch(&[3, 5, 2], vec![], vec![3]);
    assert!(
        backend.embed(invalid).is_err(),
        "invalid raw segment must be rejected"
    );
    backend.health().expect("healthy after invalid raw index");
    let retry = all_map(
        backend
            .embed(raw_batch(&[3, 5, 2], vec![], vec![0, 2]))
            .expect("valid raw retry"),
    );
    assert_eq!(retry.len(), 2);
    for index in [0usize, 2] {
        assert_rows(&retry[&index], &baseline[&index]);
    }
    backend.health().expect("healthy after valid retry");
}
