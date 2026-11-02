// Copyright Infinity Core Contributors
// Licensed under the repository LICENSE.

//! Root-only unified factory consumer for the plain UltraBERT route.
//!
//! It exercises the actual `OxideBackend::new` factory, Deferred completion,
//! B1 plus ragged growing/shrinking batches and repeated lifecycle reuse. The
//! JSON report can be compared with the Native LibTorch path on the Root host.

use infinity_backend_oxide::OxideBackend;
use infinity_v3_backend_core::{Backend, BackendError, Batch, Embedding, ModelType, Pool};
use serde::Serialize;
use std::path::Path;

#[derive(Serialize)]
struct Case {
    name: String,
    tokens: Vec<u32>,
    boundaries: Vec<u32>,
    pooled: Vec<Vec<f32>>,
    raw: Vec<Vec<Vec<f32>>>,
    repeated_bits_equal: bool,
}

#[derive(Serialize)]
struct Report {
    model: String,
    cases: Vec<Case>,
    lifecycle: Vec<String>,
    status: String,
}

fn batch(tokens: Vec<u32>, boundaries: Vec<u32>, raw_all: bool) -> Batch {
    let sequences = boundaries.len() - 1;
    let length = tokens.len();
    Batch {
        multimodal: vec![None; sequences],
        input_ids: tokens,
        token_type_ids: vec![0; length],
        position_ids: (0..length as u32).collect(),
        cumulative_seq_lengths: boundaries,
        max_length: length as u32,
        pooled_indices: (0..sequences as u32).collect(),
        raw_indices: if raw_all {
            (0..sequences as u32).collect()
        } else {
            Vec::new()
        },
        compact_input_ids: None,
        compact_position_ids: None,
        scatter_unfold: None,
        fold_gather: None,
        tokens: Vec::new(),
        offsets: Vec::new(),
    }
}

fn unpack(embeddings: infinity_v3_backend_core::Embeddings) -> (Vec<Vec<f32>>, Vec<Vec<Vec<f32>>>) {
    let mut pooled = Vec::new();
    let mut raw = Vec::new();
    let mut entries = embeddings.into_iter().collect::<Vec<_>>();
    entries.sort_by_key(|(index, _)| *index);
    for (_, value) in entries {
        match value {
            Embedding::Pooled(row) => pooled.push(row),
            Embedding::All(rows) => raw.push(rows),
        }
    }
    (pooled, raw)
}

fn bits(pooled: &[Vec<f32>], raw: &[Vec<Vec<f32>>]) -> Vec<u32> {
    pooled
        .iter()
        .chain(raw.iter().flatten())
        .flatten()
        .map(|value| value.to_bits())
        .collect()
}

fn run(
    backend: &OxideBackend,
    name: &str,
    tokens: Vec<u32>,
    boundaries: Vec<u32>,
) -> Result<Case, BackendError> {
    let request = batch(tokens.clone(), boundaries.clone(), true);
    let first = backend.embed(request.clone())?;
    let second = backend.embed(request)?;
    let (pooled, raw) = unpack(first);
    let (pooled_second, raw_second) = unpack(second);
    Ok(Case {
        name: name.into(),
        tokens,
        boundaries,
        repeated_bits_equal: bits(&pooled, &raw) == bits(&pooled_second, &raw_second),
        pooled,
        raw,
    })
}

#[test]
#[ignore = "Root CUDA only: set ULTRABERT_MODEL, INFINITY_FA4_ROOT, INFINITY_OXIDE_KERNEL_MODULE"]
fn factory_consumer_b1_ragged_growing_shrinking_and_lifecycle() {
    let Ok(model) = std::env::var("ULTRABERT_MODEL") else {
        return;
    };
    let backend = OxideBackend::new(
        Path::new(&model),
        "bfloat16",
        ModelType::Embedding(Pool::Cls),
        0,
    )
    .expect("factory construction");
    let tokens = |length: usize| {
        (0..length)
            .map(|index| (index % 31) as u32)
            .collect::<Vec<_>>()
    };
    let mut cases = vec![
        run(&backend, "b1", tokens(5), vec![0, 5]).expect("B1"),
        run(&backend, "grow", tokens(6), vec![0, 1, 3, 6]).expect("growing ragged"),
        run(&backend, "shrink", tokens(6), vec![0, 4, 5, 6]).expect("shrinking ragged"),
        run(&backend, "wide", tokens(8), vec![0, 3, 7, 8]).expect("wide ragged"),
    ];
    let mut lifecycle = Vec::new();
    for attempt in 0..4 {
        cases.push(
            run(
                &backend,
                &format!("life{attempt}"),
                tokens(4),
                vec![0, 2, 4],
            )
            .expect("lifecycle"),
        );
        lifecycle.push(format!(
            "attempt {attempt} completed and retained backend health"
        ));
        backend.health().expect("health after request");
    }
    let report = Report {
        model,
        cases,
        lifecycle,
        status: "executed".into(),
    };
    let bytes = serde_json::to_vec_pretty(&report).expect("report serialization");
    if let Ok(path) = std::env::var("ULTRABERT_CONSUMER_REPORT") {
        std::fs::write(path, bytes).expect("report write");
    } else {
        println!("{}", String::from_utf8_lossy(&bytes));
    }
}
