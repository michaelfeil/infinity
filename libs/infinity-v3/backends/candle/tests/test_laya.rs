//! Set LAYA_CHECKPOINT_DIR to run this test against a local copy of the
//! published convaiinnovations/laya-typed-decisions checkpoint.
use anyhow::Result;
use candle::{DType, Device};
use infinity_v3_backend_candle::LayaModel;
use tokenizers::Tokenizer;

#[test]
fn laya_checkpoint_scores_all_markers() -> Result<()> {
    let Ok(path) = std::env::var("LAYA_CHECKPOINT_DIR") else {
        return Ok(());
    };
    let path = std::path::Path::new(&path);
    let (model, config) = LayaModel::from_model_dir(path, DType::F32, &Device::Cpu, false)?;
    let tokenizer = Tokenizer::from_file(path.join("tokenizer/tokenizer.json"))
        .map_err(|err| anyhow::anyhow!(err.to_string()))?;
    let cls = tokenizer.token_to_id("[CLS]").unwrap();
    let sep = tokenizer.token_to_id("[SEP]").unwrap();
    let mask = tokenizer.token_to_id("[MASK]").unwrap();
    let mut ids = vec![cls];
    ids.extend(
        tokenizer
            .encode("choice question: Which team?", false)
            .map_err(|err| anyhow::anyhow!(err.to_string()))?
            .get_ids(),
    );
    ids.push(sep);
    let first = ids.len();
    ids.push(mask);
    ids.extend(
        tokenizer
            .encode(" billing", false)
            .map_err(|err| anyhow::anyhow!(err.to_string()))?
            .get_ids(),
    );
    let second = ids.len();
    ids.push(mask);
    ids.extend(
        tokenizer
            .encode(" support", false)
            .map_err(|err| anyhow::anyhow!(err.to_string()))?
            .get_ids(),
    );
    ids.push(sep);
    ids.extend(
        tokenizer
            .encode("I was billed twice", false)
            .map_err(|err| anyhow::anyhow!(err.to_string()))?
            .get_ids(),
    );
    ids.push(sep);
    let result = model.forward(&ids, &[first, second], 0)?;
    assert_eq!(result.logits.len(), 2);
    assert!(result.logits.iter().all(|x| x.is_finite()));
    assert!((0.0..=1.0).contains(&result.action_probability));
    // Same token IDs and marker positions as the upstream Laya/PyTorch fixture.
    let reference = [2.1987615_f32, 0.16098782];
    for (actual, expected) in result.logits.iter().zip(reference) {
        assert!((actual - expected).abs() < 0.02, "{actual} != {expected}");
    }
    assert!(config.temperature_for(0, 2).is_finite());
    Ok(())
}

#[test]
fn laya_batch_preserves_question_types_and_marker_positions() -> Result<()> {
    use infinity_v3_backend_candle::CandleBackend;
    use infinity_v3_backend_core::{Backend, Batch, DecisionInput, ModelType};
    let Ok(path) = std::env::var("LAYA_CHECKPOINT_DIR") else {
        return Ok(());
    };
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../router/tests/fixtures/laya-systemone.json"
    ))?;
    let model = CandleBackend::new(
        std::path::Path::new(&path),
        "float32".into(),
        ModelType::Decision,
        None,
        0,
    )?;
    let sequences = fixture["sequences"].as_array().unwrap();
    let mut batch = Batch {
        multimodal: vec![],
        input_ids: vec![],
        token_type_ids: vec![],
        position_ids: vec![],
        cumulative_seq_lengths: vec![0],
        max_length: 0,
        pooled_indices: vec![],
        raw_indices: vec![],
        compact_input_ids: None,
        compact_position_ids: None,
        scatter_unfold: None,
        fold_gather: None,
        tokens: vec![],
        offsets: vec![],
    };
    let mut inputs = vec![];
    for (question_type, seq) in sequences.iter().enumerate() {
        let ids: Vec<u32> = serde_json::from_value(seq["ids"].clone())?;
        batch.max_length = batch.max_length.max(ids.len() as u32);
        batch.token_type_ids.extend(vec![0; ids.len()]);
        batch.position_ids.extend(0..ids.len() as u32);
        batch.input_ids.extend(ids);
        batch
            .cumulative_seq_lengths
            .push(batch.input_ids.len() as u32);
        inputs.push(DecisionInput::Laya {
            question_type,
            markers: serde_json::from_value(seq["markers"].clone())?,
        });
    }
    let outputs = model.decide(batch.clone(), inputs.clone())?;
    // Packed sequence boundaries must isolate each question in mixed-length batches.
    for (i, output) in outputs.iter().enumerate() {
        let start = batch.cumulative_seq_lengths[i] as usize;
        let end = batch.cumulative_seq_lengths[i + 1] as usize;
        let mut single = batch.clone();
        single.input_ids = batch.input_ids[start..end].to_vec();
        single.position_ids = batch.position_ids[start..end].to_vec();
        single.token_type_ids = batch.token_type_ids[start..end].to_vec();
        single.cumulative_seq_lengths = vec![0, (end - start) as u32];
        single.max_length = (end - start) as u32;
        let alone = model.decide(single, vec![inputs[i].clone()])?;
        for (batched, single) in output.logits.iter().zip(&alone[0].logits) {
            assert!(
                (batched - single).abs() < 0.001,
                "batch: {batched}, single: {single}"
            );
        }
    }
    assert_eq!(outputs.len(), sequences.len());
    for (output, seq) in outputs.iter().zip(sequences) {
        let expected: Vec<f32> = serde_json::from_value(seq["logits"].clone())?;
        assert_eq!(output.logits.len(), expected.len());
        for (actual, expected) in output.logits.iter().zip(expected) {
            assert!((actual - expected).abs() < 0.025, "{actual} != {expected}");
        }
        assert!(
            (output.action_probability - seq["action_probability"].as_f64().unwrap() as f32).abs()
                < 0.01
        );
    }
    Ok(())
}

#[cfg(all(feature = "experimental-fp8", not(feature = "flash-attn")))]
#[test]
#[ignore = "requires CUDA and LAYA_CHECKPOINT_DIR"]
fn laya_fp8_without_flash_attention_fails_at_startup() -> Result<()> {
    use infinity_v3_backend_candle::CandleBackend;
    use infinity_v3_backend_core::ModelType;
    let path = std::env::var("LAYA_CHECKPOINT_DIR")?;
    let error = CandleBackend::new_with_fp8(
        std::path::Path::new(&path),
        "bfloat16".into(),
        ModelType::Decision,
        None,
        0,
        true,
    )
    .err()
    .expect("Laya FP8 must reject a build without flash-attn at startup");
    assert!(error.to_string().contains("flash-attn"), "{error}");
    Ok(())
}
