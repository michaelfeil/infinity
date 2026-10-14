#![cfg(feature = "experimental-deberta")]
use candle::{DType, Device, Tensor};
use infinity_v3_backend_candle::CandleBackend;
use infinity_v3_backend_core::{Backend, Batch, Embedding, ModelType, Pool};

#[test]
#[ignore = "requires H100, FA4 DeBERTa AOT bundle, and generated Transformers fixtures"]
fn packed_deberta_matches_transformers() -> anyhow::Result<()> {
    let root = std::path::PathBuf::from(std::env::var("INFINITY_DEBERTA_FIXTURES")?);
    let mut tested = 0;
    for entry in std::fs::read_dir(root)? {
        let path = entry?.path();
        if !path.is_dir() {
            continue;
        }
        tested += 1;
        let data: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path.join("inputs.json"))?)?;
        let lengths: Vec<usize> = serde_json::from_value(data["lengths"].clone())?;
        let input_ids: Vec<u32> = serde_json::from_value(data["input_ids"].clone())?;
        let token_type_ids: Vec<u32> = serde_json::from_value(data["token_type_ids"].clone())?;
        let mut offsets = vec![0u32];
        for &n in &lengths {
            offsets.push(offsets.last().unwrap() + n as u32);
        }
        for dtype in ["float16", "bfloat16"] {
            let expected =
                candle::safetensors::load(path.join(format!("{dtype}.safetensors")), &Device::Cpu)?;
            let expected = expected["hidden"].to_vec2::<f32>()?;
            let backend = CandleBackend::new(
                &path,
                dtype.into(),
                ModelType::Embedding(Pool::Mean),
                None,
                0,
            )?;
            let batch = Batch {
                multimodal: vec![],
                input_ids: input_ids.clone(),
                token_type_ids: token_type_ids.clone(),
                position_ids: lengths.iter().flat_map(|&n| 0..n as u32).collect(),
                cumulative_seq_lengths: offsets.clone(),
                max_length: *lengths.iter().max().unwrap() as u32,
                pooled_indices: vec![],
                raw_indices: (0..lengths.len() as u32).collect(),
                compact_input_ids: None,
                compact_position_ids: None,
                scatter_unfold: None,
                fold_gather: None,
                tokens: vec![],
                offsets: vec![],
            };
            let actual = backend.embed(batch.clone())?;
            let mut max = 0f32;
            let mut sq = 0f64;
            let mut count = 0usize;
            for i in 0..lengths.len() {
                let Embedding::All(rows) = &actual[&i] else {
                    panic!("expected token outputs")
                };
                assert_eq!(rows.len(), lengths[i]);
                for (a, b) in rows
                    .iter()
                    .zip(&expected[offsets[i] as usize..offsets[i + 1] as usize])
                {
                    assert_eq!(a.len(), b.len());
                    for (&a, &b) in a.iter().zip(b) {
                        assert!(a.is_finite());
                        let e = (a - b).abs();
                        max = max.max(e);
                        sq += (e as f64).powi(2);
                        count += 1;
                    }
                }
            }
            let rms = (sq / count as f64).sqrt();
            eprintln!("{} {dtype}: max_abs={max} rms={rms}", path.display());
            assert!(max < if dtype == "float16" { 0.012 } else { 0.09 });
            assert!(rms < if dtype == "float16" { 0.002 } else { 0.015 });
            let config: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(path.join("config.json"))?)?;
            let arch = config["architectures"][0].as_str().unwrap_or("");
            if arch.ends_with("Classification") {
                let backend =
                    CandleBackend::new(&path, dtype.into(), ModelType::Classifier, None, 0)?;
                let refs = candle::safetensors::load(
                    path.join(format!("{dtype}.safetensors")),
                    &Device::Cpu,
                )?;
                let expected = refs["logits"].to_vec2::<f32>()?;
                let actual: Vec<Vec<f32>> = if arch.ends_with("TokenClassification") {
                    let predictions = backend.predict_tokens(batch.clone())?;
                    (0..lengths.len())
                        .flat_map(|i| predictions[&i].clone())
                        .collect()
                } else {
                    let mut b = batch.clone();
                    b.raw_indices.clear();
                    b.pooled_indices = (0..lengths.len() as u32).collect();
                    let predictions = backend.predict(b)?;
                    (0..lengths.len())
                        .map(|i| predictions[&i].clone())
                        .collect()
                };
                assert_eq!(actual.len(), expected.len());
                let e = actual
                    .iter()
                    .flatten()
                    .zip(expected.iter().flatten())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                eprintln!("{} {dtype} classifier max_abs={e}", path.display());
                assert!(e < if dtype == "float16" { 0.003 } else { 0.02 });
            }
            // A reordered raw subset with simultaneous pooling must preserve request identity.
            let mut mixed = batch;
            mixed.raw_indices = vec![3, 0];
            mixed.pooled_indices = vec![1, 4];
            let mixed = backend.embed(mixed)?;
            for i in [3usize, 0] {
                let (Embedding::All(a), Embedding::All(b)) = (&mixed[&i], &actual[&i]) else {
                    panic!()
                };
                assert_eq!(a, b);
            }
            for i in [1usize, 4] {
                let Embedding::Pooled(p) = &mixed[&i] else {
                    panic!()
                };
                assert!(p.iter().all(|v| v.is_finite()));
                // Compare with the same dtype's per-sequence mean reference.
                let range = &expected[offsets[i] as usize..offsets[i + 1] as usize];
                let tensor = Tensor::from_vec(
                    range.iter().flatten().copied().collect::<Vec<_>>(),
                    (range.len(), range[0].len()),
                    &Device::Cpu,
                )?
                .to_dtype(if dtype == "float16" {
                    DType::F16
                } else {
                    DType::BF16
                })?;
                let mean = tensor.to_dtype(DType::F32)?.mean(0)?.to_vec1::<f32>()?;
                assert!(p.iter().zip(mean).all(|(a, b)| (a - b).abs() < 0.04));
            }
        }
    }
    assert!(tested >= 8, "expected all eight architecture fixtures");
    Ok(())
}
