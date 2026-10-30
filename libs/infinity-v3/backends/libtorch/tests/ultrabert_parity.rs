//! Optional comparisons with independently generated public Hugging Face fixtures.
use infinity_backend_libtorch::LibtorchBackend;
fn empty_batch(pooled: Vec<u32>, raw: Vec<u32>) -> text_embeddings_backend_core::Batch {
    text_embeddings_backend_core::Batch {
        input_ids: vec![],
        token_type_ids: vec![],
        position_ids: vec![],
        cumulative_seq_lengths: vec![0],
        max_length: 0,
        pooled_indices: pooled,
        raw_indices: raw,
        multimodal: vec![],
        compact_input_ids: None,
        compact_position_ids: None,
        scatter_unfold: None,
        fold_gather: None,
        tokens: vec![],
        offsets: vec![],
    }
}
fn device() -> String {
    std::env::var("ULTRABERT_DEVICE").unwrap_or_else(|_| "cpu".into())
}

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
use serde::Deserialize;
use std::path::PathBuf;
use text_embeddings_backend_core::{Backend, Embedding, ModelType, Pool};

#[derive(Deserialize)]
struct Reference {
    name: String,
    input_ids: Vec<u32>,
    pooled: Vec<f32>,
    mean: Vec<f32>,
    last: Vec<f32>,
    raw_indices: Vec<usize>,
    raw: Vec<Vec<f32>>,
}

fn input(
    cases: &[&Reference],
    pooled: Vec<u32>,
    raw: Vec<u32>,
) -> text_embeddings_backend_core::Batch {
    let mut batch = empty_batch(pooled, raw);
    for case in cases {
        batch.input_ids.extend(&case.input_ids);
        batch.position_ids.extend(0..case.input_ids.len() as u32);
        batch
            .token_type_ids
            .extend((0..case.input_ids.len()).map(|i| u32::from(i >= case.input_ids.len() / 2)));
        batch
            .cumulative_seq_lengths
            .push(batch.input_ids.len() as u32);
        batch.max_length = batch.max_length.max(case.input_ids.len() as u32);
    }
    batch
}

fn compare(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len());
    let dot: f64 = actual
        .iter()
        .zip(expected)
        .map(|(&a, &b)| a as f64 * b as f64)
        .sum();
    let norm = |x: &[f32]| x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>();
    let cosine = dot / (norm(actual) * norm(expected)).sqrt();
    let error = actual
        .iter()
        .zip(expected)
        .map(|(&a, &b)| ((a - b) as f64).powi(2))
        .sum::<f64>();
    let relative = (error / norm(expected)).sqrt();
    assert!(
        cosine > 0.999 && relative < 0.04,
        "{label}: cosine={cosine}, relative_l2={relative}"
    );
}

// Generate the reference with the checkpoint's HF AutoModel (SDPA, float32),
// retaining CLS, mean, last, and sampled raw vectors for each input_ids array.
#[test]
#[ignore = "requires ULTRABERT_MODEL and independent HF ULTRABERT_REFERENCE JSON"]
fn ultrabert_matches_hf() -> Result<()> {
    let model = PathBuf::from(std::env::var("ULTRABERT_MODEL")?);
    let reference = std::fs::read(std::env::var("ULTRABERT_REFERENCE")?)?;
    let cases: Vec<Reference> = serde_json::from_slice(&reference)?;
    assert!(cases.len() >= 5);
    let dtype = std::env::var("ULTRABERT_DTYPE").unwrap_or("float32".into());
    let backend = LibtorchBackend::new(&model, &dtype, ModelType::Embedding(Pool::Cls), &device())?;
    for case in &cases {
        let outputs = backend.embed(input(&[case], vec![0], vec![]))?;
        let Embedding::Pooled(actual) = &outputs[&0] else {
            panic!("expected pooled output")
        };
        compare(actual, &case.pooled, &case.name);
        eprintln!("{}: {} tokens matched HF", case.name, case.input_ids.len());
    }
    // Reordered raw rows mixed with a pooled row exercise ragged isolation and selection.
    let mixed = backend.embed(input(
        &[&cases[0], &cases[1], &cases[4]],
        vec![1],
        vec![2, 0],
    ))?;
    let Embedding::Pooled(actual) = &mixed[&1] else {
        panic!("expected pooled output")
    };
    compare(actual, &cases[1].pooled, "mixed CLS");
    for (index, case) in [(0, &cases[0]), (2, &cases[4])] {
        let Embedding::All(actual) = &mixed[&index] else {
            panic!("expected raw output")
        };
        assert_eq!(actual.len(), case.input_ids.len());
        for (&token, expected) in case.raw_indices.iter().zip(&case.raw) {
            compare(&actual[token], expected, "mixed raw");
        }
    }
    for (pool, expected) in [
        (Pool::Mean, &cases[4].mean),
        (Pool::LastToken, &cases[4].last),
    ] {
        let backend = LibtorchBackend::new(&model, &dtype, ModelType::Embedding(pool), &device())?;
        let outputs = backend.embed(input(&[&cases[4]], vec![0], vec![]))?;
        let Embedding::Pooled(actual) = &outputs[&0] else {
            panic!("expected pooled output")
        };
        compare(actual, expected, "alternate pooling");
    }
    Ok(())
}

mod classification {
    use super::{device, empty_batch};
    use infinity_backend_libtorch::LibtorchBackend;

    use super::Result;
    use serde::Deserialize;
    use std::path::Path;
    use text_embeddings_backend_core::{Backend, Batch, ModelType};

    #[derive(Deserialize)]
    struct Case {
        input_ids: Vec<u32>,
        logits: Vec<f32>,
        token_logits: Option<Vec<Vec<f32>>>,
    }

    #[derive(Deserialize)]
    struct Reference {
        model: String,
        token: bool,
        labels: usize,
        cases: Vec<Case>,
    }

    fn input(cases: &[&Case]) -> Batch {
        let mut batch = empty_batch((0..cases.len() as u32).collect(), vec![]);
        for case in cases {
            batch.input_ids.extend(&case.input_ids);
            batch.position_ids.extend(0..case.input_ids.len() as u32);
            batch.token_type_ids.extend(
                (0..case.input_ids.len()).map(|i| u32::from(i >= case.input_ids.len() / 2)),
            );
            batch
                .cumulative_seq_lengths
                .push(batch.input_ids.len() as u32);
            batch.max_length = batch.max_length.max(case.input_ids.len() as u32);
        }
        batch
    }

    fn compare(actual: &[f32], expected: &[f32], dtype: &str) {
        assert_eq!(actual.len(), expected.len());
        let (absolute, relative) = match dtype {
            "bfloat16" => (0.02, 0.04),
            "float16" => (0.003, 0.015),
            _ => (0.0002, 0.0002),
        };
        for (&actual, &expected) in actual.iter().zip(expected) {
            assert!(actual.is_finite());
            assert!(
                (actual - expected).abs() <= absolute + relative * expected.abs(),
                "{dtype}: actual={actual}, expected={expected}"
            );
        }
    }

    // References use HF's published sequence/token classes in eval mode, float32 SDPA.
    // Include the trained 150M/380M rerankers and deterministic two-/three-label
    // fixtures; synthetic heads validate layouts, not classification quality.
    #[test]
    #[ignore = "requires ULTRABERT_CLASSIFICATION_REFERENCE JSON and referenced checkpoints"]
    fn ultrabert_classifiers_match_hf() -> Result<()> {
        let refs: Vec<Reference> = serde_json::from_slice(&std::fs::read(std::env::var(
            "ULTRABERT_CLASSIFICATION_REFERENCE",
        )?)?)?;
        assert!(refs.iter().any(|r| r.token));
        assert!(refs.iter().any(|r| r.labels == 1));
        assert!(refs.iter().any(|r| r.labels > 1));
        let dtype = std::env::var("ULTRABERT_DTYPE").unwrap_or("float32".into());
        for reference in refs {
            let backend = LibtorchBackend::new(
                Path::new(&reference.model),
                &dtype,
                ModelType::Classifier,
                &device(),
            )?;
            assert!(!backend.supports_radix_mlp());
            let cases: Vec<_> = reference.cases.iter().collect();
            let batched = backend.predict(input(&cases))?;
            assert_eq!(batched.len(), cases.len());
            for (i, case) in cases.iter().enumerate() {
                assert_eq!(batched[&i].len(), reference.labels);
                compare(&batched[&i], &case.logits, &dtype);
                let single = backend.predict(input(&[case]))?;
                compare(&single[&0], &case.logits, &dtype);
            }
            if reference.token {
                let predictions = backend.predict_tokens(input(&cases))?;
                for (i, case) in cases.iter().enumerate() {
                    assert_eq!(predictions[&i].len(), case.input_ids.len());
                    for (actual, expected) in predictions[&i]
                        .iter()
                        .zip(case.token_logits.as_ref().unwrap())
                    {
                        compare(actual, expected, &dtype);
                    }
                }
            } else {
                assert!(backend.predict_tokens(input(&[cases[0]])).is_err());
            }
            eprintln!(
                "{}: {} ragged/single cases, {} labels, token={}",
                reference.model,
                cases.len(),
                reference.labels,
                reference.token
            );
        }
        Ok(())
    }
}

#[test]
#[ignore = "Requires CUDA, ULTRABERT_MODEL/REFERENCE and INFINITY_TORCH_CUDA_GRAPHS=1"]
fn ultrabert_cuda_graphs_capture() -> Result<()> {
    assert_eq!(std::env::var("INFINITY_TORCH_CUDA_GRAPHS")?.as_str(), "1");
    let model = PathBuf::from(std::env::var("ULTRABERT_MODEL")?);
    let cases: Vec<Reference> =
        serde_json::from_slice(&std::fs::read(std::env::var("ULTRABERT_REFERENCE")?)?)?;
    let backend = LibtorchBackend::new(
        &model,
        "bfloat16",
        ModelType::Embedding(Pool::Cls),
        "cuda:0",
    )?;
    for _ in 0..8 {
        let outputs = backend.embed(input(&[&cases[0]], vec![0], vec![]))?;
        let Embedding::Pooled(actual) = &outputs[&0] else {
            panic!("expected pooled embedding")
        };
        compare(actual, &cases[0].pooled, "CUDA graph CLS");
    }
    assert!(
        backend.cuda_graph_count() > 0,
        "UltraBERT text execution did not capture a CUDA graph"
    );
    Ok(())
}
