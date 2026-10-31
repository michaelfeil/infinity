use infinity_v3_backend_core::{Backend, Batch, Embedding, ModelType, Pool};
use infinity_v3_backend_python::{PythonBackend, PythonBackendConfig};
use serial_test::serial;
use std::path::Path;

fn config() -> PythonBackendConfig {
    PythonBackendConfig {
        config: std::env::var("INFINITY_PYTHON_CONFIG").ok(),
    }
}

fn batch() -> Batch {
    Batch {
        multimodal: vec![],
        input_ids: vec![101, 102, 103, 201, 202],
        token_type_ids: vec![0, 0, 0, 0, 0],
        position_ids: vec![0, 1, 2, 0, 1],
        cumulative_seq_lengths: vec![0, 3, 5],
        max_length: 3,
        pooled_indices: vec![0, 1],
        raw_indices: vec![],
        compact_input_ids: None,
        compact_position_ids: None,
        fold_gather: None,
        scatter_unfold: None,
        tokens: vec![],
        offsets: vec![],
    }
}

#[test]
#[serial]
fn configured_backend_returns_embeddings() {
    let Ok(model) = std::env::var("INFINITY_PYTHON_SMOKE_MODEL") else {
        return;
    };
    let backend = PythonBackend::new(
        Path::new(&model),
        "bfloat16",
        ModelType::Embedding(Pool::Mean),
        0,
        config(),
    )
    .unwrap();
    backend.health().unwrap();
    let embeddings = backend.embed(batch()).unwrap();
    assert_eq!(embeddings.len(), 2);
    for index in 0..2 {
        let Embedding::Shared(values) = &embeddings[&index] else {
            panic!("expected pooled embedding for row {index}");
        };
        assert!(!values.is_empty());
        assert!(values.iter().all(|value| value.is_finite()));
    }
    backend.shutdown();
}

#[test]
#[serial]
fn configured_backend_returns_classifier_rows() {
    let Ok(model) = std::env::var("INFINITY_PYTHON_CLASSIFIER_MODEL") else {
        return;
    };
    let backend = PythonBackend::new(
        Path::new(&model),
        "bfloat16",
        ModelType::Classifier,
        0,
        config(),
    )
    .unwrap();
    let predictions = backend.predict(batch()).unwrap();
    assert_eq!(predictions.len(), 2);
    assert!(predictions.values().all(|row| row.len() >= 2));
    backend.shutdown();
}
