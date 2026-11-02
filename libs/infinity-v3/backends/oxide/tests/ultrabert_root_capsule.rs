// Copyright Infinity Core Contributors
// Licensed under the repository LICENSE.

//! Root-only whole-model UltraBERT comparison capsule.
//!
//! This test is ignored by default. It requires the pinned checkpoint, a
//! matching FA4 provider, the oxide kernel bundle and CUDA. It compares
//! CLS/mean/last pooled vectors and sampled raw rows with an independently
//! generated reference JSON, then exercises typed lifecycle/capacity paths.
//!
//! Reference generation (Root, Transformers available):
//!   python3 libs/infinity-v3/scripts/ultrabert-reference.py \
//!     --model /path/to/ultrabert-embed-150m --head embedding \
//!     --device cpu --output /tmp/ultrabert-reference.json
//!
//! Run:
//!   CUDA_VISIBLE_DEVICES=0 taskset -c 120-127 \
//!     ULTRABERT_MODEL=/path/to/ultrabert-embed-150m \
//!     ULTRABERT_REFERENCE=/tmp/ultrabert-reference.json \
//!     INFINITY_FA4_ROOT=/path/to/infinityfa4 \
//!     INFINITY_OXIDE_KERNEL_MODULE=/path/to/oxide-kernels \
//!     ULTRABERT_CAPSULE_REPORT=/tmp/ultrabert-capsule.json \
//!     cargo +1.93.1 test -p infinity-backend-oxide \
//!       --test ultrabert_root_capsule -- --ignored --nocapture
//!
//! No production factory route is used.

use cuda_core::Context;
use infinity_backend_oxide::ultrabert_executor::{
    UltraBertEmbeddingPool, UltraBertExecutionMode, UltraBertOutput,
};
use infinity_backend_oxide::ultrabert_qualification::UltraBertQualification;
use infinity_v3_backend_core::Batch;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Deserialize)]
struct ReferenceCase {
    name: String,
    input_ids: Vec<u32>,
    pooled: Vec<f32>,
    mean: Vec<f32>,
    last: Vec<f32>,
    raw_indices: Vec<usize>,
    raw: Vec<Vec<f32>>,
}

#[derive(Debug, Serialize)]
struct CaseReport {
    name: String,
    tokens: usize,
    cls_cosine: f64,
    cls_relative_l2: f64,
    mean_cosine: f64,
    mean_relative_l2: f64,
    last_cosine: f64,
    last_relative_l2: f64,
    raw_cosines: Vec<f64>,
    raw_relative_l2: Vec<f64>,
}

#[derive(Debug, Serialize)]
struct CapsuleReport {
    checkpoint: String,
    reference: String,
    readiness: String,
    cases: Vec<CaseReport>,
    lifecycle: Vec<String>,
    capacity: Vec<String>,
    status: String,
}

fn batch(input_ids: Vec<u32>, pooled: Vec<u32>, raw: Vec<u32>, maximum: u32) -> Batch {
    let sequences = 1usize;
    let length = input_ids.len();
    Batch {
        multimodal: vec![None; sequences],
        token_type_ids: vec![0; length],
        position_ids: (0..length as u32).collect(),
        cumulative_seq_lengths: vec![0, length as u32],
        max_length: maximum,
        input_ids,
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

fn cosine(actual: &[f32], expected: &[f32]) -> f64 {
    let dot: f64 = actual
        .iter()
        .zip(expected)
        .map(|(&a, &b)| a as f64 * b as f64)
        .sum();
    let actual_norm: f64 = actual.iter().map(|&value| (value as f64).powi(2)).sum();
    let expected_norm: f64 = expected.iter().map(|&value| (value as f64).powi(2)).sum();
    dot / (actual_norm * expected_norm).sqrt()
}

fn relative_l2(actual: &[f32], expected: &[f32]) -> f64 {
    let error: f64 = actual
        .iter()
        .zip(expected)
        .map(|(&a, &b)| ((a - b) as f64).powi(2))
        .sum();
    let expected_norm: f64 = expected.iter().map(|&value| (value as f64).powi(2)).sum();
    (error / expected_norm).sqrt()
}

fn assert_match(actual: &[f32], expected: &[f32], label: &str) -> (f64, f64) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    let cosine = cosine(actual, expected);
    let relative = relative_l2(actual, expected);
    assert!(
        cosine > 0.999 && relative < 0.04,
        "{label}: cosine={cosine}, relative_l2={relative}"
    );
    (cosine, relative)
}

fn pooled(
    executor: &mut infinity_backend_oxide::ultrabert_executor::UltraBertExecutor,
    input_ids: Vec<u32>,
    pool: UltraBertEmbeddingPool,
    maximum: u32,
) -> Vec<f32> {
    let batch = batch(input_ids, vec![0], vec![], maximum);
    match executor
        .forward(&batch, UltraBertExecutionMode::Embedding(pool))
        .expect("pooled forward")
    {
        UltraBertOutput::Embeddings { pooled, raw } => {
            assert!(raw.is_empty());
            assert_eq!(pooled.len(), 1);
            pooled.into_iter().next().unwrap()
        }
        other => panic!("unexpected output {other:?}"),
    }
}

fn raw_rows(
    executor: &mut infinity_backend_oxide::ultrabert_executor::UltraBertExecutor,
    input_ids: Vec<u32>,
    maximum: u32,
) -> Vec<Vec<f32>> {
    let batch = batch(input_ids, vec![], vec![0], maximum);
    match executor
        .forward(
            &batch,
            UltraBertExecutionMode::Embedding(UltraBertEmbeddingPool::Cls),
        )
        .expect("raw forward")
    {
        UltraBertOutput::Embeddings { pooled, raw } => {
            assert!(pooled.is_empty());
            assert_eq!(raw.len(), 1);
            raw.into_iter().next().unwrap()
        }
        other => panic!("unexpected output {other:?}"),
    }
}

#[test]
#[ignore = "requires CUDA, pinned UltraBERT checkpoint, FA4 provider and oxide kernel bundle"]
fn root_whole_model_capsule() {
    let model = std::env::var("ULTRABERT_MODEL").expect("ULTRABERT_MODEL");
    let reference = std::env::var("ULTRABERT_REFERENCE").expect("ULTRABERT_REFERENCE");
    let fa4_root = std::env::var("INFINITY_FA4_ROOT").expect("INFINITY_FA4_ROOT");
    assert!(
        std::env::var_os("INFINITY_OXIDE_KERNEL_MODULE").is_some(),
        "INFINITY_OXIDE_KERNEL_MODULE"
    );
    let device_id = std::env::var("ULTRABERT_DEVICE_ID")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let cases: Vec<ReferenceCase> =
        serde_json::from_slice(&std::fs::read(&reference).expect("reference json"))
            .expect("reference schema");
    assert!(cases.len() >= 5, "reference needs at least five cases");

    let qualification = UltraBertQualification::open(Path::new(&model), "bfloat16", None)
        .expect("qualification header");
    qualification
        .require_gpu_ready()
        .expect("FA4 local128 + mha128 readiness");
    let context = Context::new(device_id).expect("CUDA context");
    let stream = context.new_stream().expect("CUDA stream");
    let mut executor = qualification
        .prepare_gpu(context, stream, Path::new(&fa4_root))
        .expect("prepare executor");

    let mut report = CapsuleReport {
        checkpoint: model.clone(),
        reference: reference.clone(),
        readiness: "local128+mha128".into(),
        cases: Vec::new(),
        lifecycle: vec![],
        capacity: vec![],
        status: "running".into(),
    };

    for case in &cases {
        let maximum = case.input_ids.len() as u32;
        let cls = pooled(
            &mut executor,
            case.input_ids.clone(),
            UltraBertEmbeddingPool::Cls,
            maximum,
        );
        let (cls_cosine, cls_relative_l2) = assert_match(&cls, &case.pooled, "cls");
        let mean = pooled(
            &mut executor,
            case.input_ids.clone(),
            UltraBertEmbeddingPool::Mean,
            maximum,
        );
        let (mean_cosine, mean_relative_l2) = assert_match(&mean, &case.mean, "mean");
        let last = pooled(
            &mut executor,
            case.input_ids.clone(),
            UltraBertEmbeddingPool::LastToken,
            maximum,
        );
        let (last_cosine, last_relative_l2) = assert_match(&last, &case.last, "last");
        let raw = raw_rows(&mut executor, case.input_ids.clone(), maximum);
        assert_eq!(raw.len(), case.input_ids.len());
        let mut raw_cosines = Vec::new();
        let mut raw_relative_l2 = Vec::new();
        for (&token, expected) in case.raw_indices.iter().zip(&case.raw) {
            let actual = raw.get(token).expect("sampled raw row");
            let (cosine, relative) = assert_match(actual, expected, "raw");
            raw_cosines.push(cosine);
            raw_relative_l2.push(relative);
        }
        report.cases.push(CaseReport {
            name: case.name.clone(),
            tokens: case.input_ids.len(),
            cls_cosine,
            cls_relative_l2,
            mean_cosine,
            mean_relative_l2,
            last_cosine,
            last_relative_l2,
            raw_cosines,
            raw_relative_l2,
        });
    }

    // Typed lifecycle: repeated forward, invalid offsets, out-of-range token,
    // and a valid repeat after the errors.
    let first = &cases[0];
    let repeat_a = pooled(
        &mut executor,
        first.input_ids.clone(),
        UltraBertEmbeddingPool::Cls,
        first.input_ids.len() as u32,
    );
    let repeat_b = pooled(
        &mut executor,
        first.input_ids.clone(),
        UltraBertEmbeddingPool::Cls,
        first.input_ids.len() as u32,
    );
    assert_eq!(repeat_a, repeat_b);
    report
        .lifecycle
        .push("repeated CLS forward is bit-identical".into());

    let mixed_batch = batch(
        first.input_ids.clone(),
        vec![0],
        vec![0],
        first.input_ids.len() as u32,
    );
    match executor
        .forward(
            &mixed_batch,
            UltraBertExecutionMode::Embedding(UltraBertEmbeddingPool::Cls),
        )
        .expect("mixed forward")
    {
        UltraBertOutput::Embeddings { pooled, raw } => {
            assert_eq!(pooled.len(), 1);
            assert_eq!(raw.len(), 1);
            assert_match(&pooled[0], &first.pooled, "mixed cls");
            let token = *first.raw_indices.first().expect("sampled raw index");
            assert_match(&raw[0][token], &first.raw[0], "mixed raw");
        }
        other => panic!("unexpected output {other:?}"),
    }
    report
        .lifecycle
        .push("mixed pooled+raw output preserves both readouts".into());

    let invalid = Batch {
        cumulative_seq_lengths: vec![0, first.input_ids.len() as u32 + 1],
        ..batch(first.input_ids.clone(), vec![0], vec![], 1)
    };
    let error = executor
        .forward(
            &invalid,
            UltraBertExecutionMode::Embedding(UltraBertEmbeddingPool::Cls),
        )
        .unwrap_err();
    assert_eq!(error.code(), "batch_offsets");
    report
        .lifecycle
        .push("invalid offsets rejected with batch_offsets".into());

    let mut out_of_range = first.input_ids.clone();
    out_of_range[0] = qualification.report().geometry.vocab_size as u32;
    let error = executor
        .forward(
            &batch(out_of_range, vec![0], vec![], first.input_ids.len() as u32),
            UltraBertExecutionMode::Embedding(UltraBertEmbeddingPool::Cls),
        )
        .unwrap_err();
    assert_eq!(error.code(), "token_out_of_range");
    report
        .lifecycle
        .push("out-of-range token rejected with token_out_of_range".into());

    let after_errors = pooled(
        &mut executor,
        first.input_ids.clone(),
        UltraBertEmbeddingPool::Cls,
        first.input_ids.len() as u32,
    );
    assert_match(&after_errors, &first.pooled, "post-error cls");
    report
        .lifecycle
        .push("valid forward recovered after typed errors".into());

    // Capacity: reject a sequence over max_position_embeddings before GPU work.
    let maximum = qualification.report().geometry.max_position_embeddings;
    let too_long = vec![0u32; maximum.saturating_add(1)];
    let error = executor
        .forward(
            &batch(too_long, vec![0], vec![], maximum as u32),
            UltraBertExecutionMode::Embedding(UltraBertEmbeddingPool::Cls),
        )
        .unwrap_err();
    assert_eq!(error.code(), "sequence_too_long");
    report
        .capacity
        .push("oversized sequence rejected with sequence_too_long".into());
    report
        .capacity
        .push("workspace capacity reused across pooled/raw/error cases".into());

    report.status = "passed".into();
    if let Ok(path) = std::env::var("ULTRABERT_CAPSULE_REPORT") {
        std::fs::write(
            path,
            serde_json::to_vec_pretty(&report).expect("report json"),
        )
        .expect("write report");
    }
    eprintln!(
        "UltraBERT capsule PASS cases={} lifecycle={} capacity={}",
        report.cases.len(),
        report.lifecycle.len(),
        report.capacity.len()
    );
}
