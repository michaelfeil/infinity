//! Opt-in actual checkpoint oracle. Weights are pinned externally, never downloaded by tests.
use infinity_backend_libtorch::LibtorchBackend;
use text_embeddings_backend_core::{Backend, Batch, DecisionInput, ModelType};
#[test]
#[ignore = "Requires INFINITY_LAYA_MODEL pointing to pinned trained Laya weights"]
fn trained_laya_cpu_matches_published_upstream_fixture() {
    let path = std::env::var("INFINITY_LAYA_MODEL").expect("INFINITY_LAYA_MODEL is required");
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../router/tests/fixtures/laya-systemone.json"
    ))
    .unwrap();
    let model = LibtorchBackend::new(
        std::path::Path::new(&path),
        "float32",
        ModelType::Decision,
        "cpu",
    )
    .unwrap();
    let mut batch = Batch {
        input_ids: vec![],
        token_type_ids: vec![],
        position_ids: vec![],
        cumulative_seq_lengths: vec![0],
        max_length: 0,
        pooled_indices: vec![],
        raw_indices: vec![],
        multimodal: vec![],
        compact_input_ids: None,
        compact_position_ids: None,
        scatter_unfold: None,
        fold_gather: None,
        tokens: vec![],
        offsets: vec![],
    };
    let sequences = fixture["sequences"].as_array().unwrap();
    let mut requests = vec![];
    for (question_type, seq) in sequences.iter().enumerate() {
        let ids: Vec<u32> = serde_json::from_value(seq["ids"].clone()).unwrap();
        batch.max_length = batch.max_length.max(ids.len() as u32);
        batch.position_ids.extend(0..ids.len() as u32);
        batch.token_type_ids.extend(vec![0; ids.len()]);
        batch.input_ids.extend(ids);
        batch
            .cumulative_seq_lengths
            .push(batch.input_ids.len() as u32);
        requests.push(DecisionInput::Laya {
            question_type,
            markers: serde_json::from_value(seq["markers"].clone()).unwrap(),
        });
    }
    let actual = model.decide(batch, requests).unwrap();
    for (output, seq) in actual.iter().zip(sequences) {
        let expected: Vec<f32> = serde_json::from_value(seq["logits"].clone()).unwrap();
        assert_eq!(output.logits.len(), expected.len());
        for (actual, expected) in output.logits.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 0.025,
                "{}: {actual} != {expected}",
                seq["id"]
            );
        }
        assert!(
            (output.action_probability - seq["action_probability"].as_f64().unwrap() as f32).abs()
                < 0.01
        );
    }
}
