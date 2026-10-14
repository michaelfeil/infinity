#![cfg(feature = "flash-attn")]
use anyhow::{Context, Result};
use candle::{DType, Device, IndexOp};
use infinity_v3_backend_candle::CandleBackend;
use infinity_v3_backend_core::{
    AudioFeatures, Backend, Batch, ImagePatches, ModelType, MultimodalEncoding, Pool,
};
use serde::Deserialize;
use std::{path::PathBuf, sync::Arc};

#[derive(Deserialize)]
struct Case {
    name: String,
    sequences: Vec<Vec<u32>>,
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let dot = a
        .iter()
        .zip(b)
        .map(|(a, b)| f64::from(*a) * f64::from(*b))
        .sum::<f64>();
    let norm = |a: &[f32]| a.iter().map(|a| f64::from(*a).powi(2)).sum::<f64>().sqrt();
    dot / (norm(a) * norm(b))
}

#[test]
#[ignore = "requires released EmbeddingGemma2 checkpoint, independent Transformers fixtures, and CUDA"]
fn released_checkpoint_matches_transformers_for_every_modality() -> Result<()> {
    let root = PathBuf::from(
        std::env::var("EMBEDDINGGEMMA2_MODEL_ROOT").context("EMBEDDINGGEMMA2_MODEL_ROOT")?,
    );
    let fixture = PathBuf::from(
        std::env::var("EMBEDDINGGEMMA2_FIXTURE_DIR").context("EMBEDDINGGEMMA2_FIXTURE_DIR")?,
    );
    let cases: Vec<Case> = serde_json::from_slice(&std::fs::read(fixture.join("manifest.json"))?)?;
    let backend = CandleBackend::new(
        &root,
        "bfloat16".into(),
        ModelType::Embedding(Pool::Mean),
        None,
        0,
    )?;
    let mut failures = Vec::new();
    for case in cases {
        let inputs = candle::safetensors::load(
            fixture.join(format!("{}.inputs.safetensors", case.name)),
            &Device::Cpu,
        )?;
        let expected = candle::safetensors::load(
            fixture.join(format!("{}.outputs.safetensors", case.name)),
            &Device::Cpu,
        )?;
        let mut batch = Batch {
            input_ids: vec![],
            token_type_ids: vec![],
            position_ids: vec![],
            cumulative_seq_lengths: vec![0],
            max_length: 0,
            pooled_indices: (0..case.sequences.len() as u32).collect(),
            raw_indices: vec![],
            compact_input_ids: None,
            compact_position_ids: None,
            scatter_unfold: None,
            fold_gather: None,
            tokens: vec![],
            offsets: vec![],
            multimodal: vec![],
        };
        let mut image_idx = 0;
        let mut video_idx = 0;
        let mut audio_idx = 0;
        for sequence in &case.sequences {
            let mut images = Vec::new();
            let mut audios = Vec::new();
            let mut cursor = 0;
            while cursor < sequence.len() {
                let id = sequence[cursor];
                if matches!(id, 258880 | 258884) {
                    let (pixels_name, positions_name, index) = if id == 258880 {
                        let index = image_idx;
                        image_idx += 1;
                        ("pixel_values", "image_position_ids", index)
                    } else {
                        let index = video_idx;
                        video_idx += 1;
                        ("pixel_values_videos", "video_position_ids", index)
                    };
                    let positions = inputs[positions_name]
                        .i(index)?
                        .to_dtype(DType::I64)?
                        .to_vec2::<i64>()?;
                    let w = positions
                        .iter()
                        .filter(|p| p[0] >= 0)
                        .map(|p| p[0])
                        .max()
                        .unwrap() as usize
                        + 1;
                    let h = positions
                        .iter()
                        .filter(|p| p[1] >= 0)
                        .map(|p| p[1])
                        .max()
                        .unwrap() as usize
                        + 1;
                    let pixels = inputs[pixels_name]
                        .i(index)?
                        .narrow(0, 0, h * w)?
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    let image = Arc::new(ImagePatches {
                        pixels,
                        grid_thw: [1, h, w],
                        patch_dim: 768,
                        merge_size: 3,
                    });
                    assert_eq!(
                        sequence[cursor..].iter().take_while(|&&v| v == id).count(),
                        image.token_count()
                    );
                    cursor += image.token_count();
                    images.push((cursor - image.token_count(), image));
                } else if id == 258881 {
                    let features = inputs["input_features"]
                        .i(audio_idx)?
                        .to_dtype(DType::F32)?;
                    let mask = inputs["input_features_mask"]
                        .i(audio_idx)?
                        .to_dtype(DType::U8)?
                        .to_vec1::<u8>()?;
                    audio_idx += 1;
                    let audio = Arc::new(AudioFeatures {
                        values: features.flatten_all()?.to_vec1::<f32>()?,
                        mask,
                        feature_size: 128,
                    });
                    assert_eq!(
                        sequence[cursor..].iter().take_while(|&&v| v == id).count(),
                        audio.token_count()
                    );
                    cursor += audio.token_count();
                    audios.push((cursor - audio.token_count(), audio));
                } else {
                    cursor += 1;
                }
            }
            batch.multimodal.push(Some(Arc::new(MultimodalEncoding {
                images,
                audios,
                position_ids: std::array::from_fn(|_| (0..sequence.len() as u32).collect()),
                memory: None,
            })));
            batch.input_ids.extend(sequence);
            batch.token_type_ids.extend(vec![0; sequence.len()]);
            batch.position_ids.extend(0..sequence.len() as u32);
            batch
                .cumulative_seq_lengths
                .push(batch.input_ids.len() as u32);
            batch.max_length = batch.max_length.max(sequence.len() as u32);
        }
        let actual = backend.embed(batch.clone())?;
        for row in 0..case.sequences.len() {
            let expected = expected["pooled"].i(row)?.to_vec1::<f32>()?;
            let actual = match &actual[&row] {
                infinity_v3_backend_core::Embedding::Pooled(values) => values,
                _ => anyhow::bail!("expected pooled embeddings"),
            };
            let text = matches!(case.name.as_str(), "text" | "text_batch" | "text_window");
            let similarity = cosine(actual, &expected);
            println!("{} row {row}: pooled cosine {similarity:.8}", case.name);
            assert_eq!(actual.len(), 768);
            if similarity <= if text { 0.999 } else { 0.995 } {
                failures.push(format!("{} pooled {row}: {similarity}", case.name));
            }
        }
        batch.pooled_indices.clear();
        batch.raw_indices = (0..case.sequences.len() as u32).collect();
        let actual = backend.embed(batch)?;
        let mut offset = 0;
        for (row, sequence) in case.sequences.iter().enumerate() {
            let expected = expected["raw"]
                .narrow(0, offset, sequence.len())?
                .to_vec2::<f32>()?;
            offset += sequence.len();
            let actual = match &actual[&row] {
                infinity_v3_backend_core::Embedding::All(tokens) => tokens,
                _ => anyhow::bail!("expected raw token embeddings"),
            };
            assert_eq!(actual.len(), expected.len());
            assert!(actual
                .iter()
                .all(|token| token.len() == 768 && token.iter().all(|v| v.is_finite())));
            let a: Vec<_> = actual.iter().flatten().copied().collect();
            let b: Vec<_> = expected.iter().flatten().copied().collect();
            let similarity = cosine(&a, &b);
            let difference = a
                .iter()
                .zip(&b)
                .map(|(a, b)| f64::from(a - b).powi(2))
                .sum::<f64>();
            let reference = b.iter().map(|b| f64::from(*b).powi(2)).sum::<f64>();
            let relative_error = (difference / reference).sqrt();
            // BF16 attention changes low-norm media token directions even within
            // Transformers: eager versus SDPA reaches 0.31 minimum token cosine
            // and 0.13 relative L2 on the text/image fixture. Compare the full output
            // energy rather than thresholding each low-norm token direction.
            let text = matches!(case.name.as_str(), "text" | "text_batch" | "text_window");
            println!(
                "{} row {row}: raw cosine {similarity:.8}, relative L2 {relative_error:.6}",
                case.name
            );
            // The 30-second audio control with FP32 convolutions/BF16 outputs
            // reaches 0.974 raw cosine and 0.227 relative L2 within Transformers
            // itself; pooled cosine stays above 0.999. Shorter media stay tighter.
            let long_audio = case.name == "audio_limit";
            let minimum_cosine = if text {
                0.999
            } else if long_audio {
                0.97
            } else {
                0.99
            };
            let maximum_error = if text {
                0.04
            } else if long_audio {
                0.24
            } else {
                0.15
            };
            if similarity <= minimum_cosine || relative_error >= maximum_error {
                failures.push(format!(
                    "{} raw {row}: cosine {similarity}, relative L2 {relative_error}",
                    case.name
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
    Ok(())
}
