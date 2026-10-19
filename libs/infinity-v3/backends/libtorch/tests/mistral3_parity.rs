//! Independent Transformers 5.12.1 fixtures generated from the public Ministral 3 model.
//! Generate with scripts/mistral3-reference.py --tiny --output /tmp/mistral3-reference.
use infinity_backend_libtorch::LibtorchBackend;
use safetensors::SafeTensors;
use std::{path::PathBuf, sync::Arc};
use text_embeddings_backend_core::{
    Backend, Batch, Embedding, ImagePatches, ModelType, MultimodalEncoding, Pool,
};
fn f32_values(t: safetensors::tensor::TensorView<'_>) -> Vec<f32> {
    t.data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn i64_values(t: safetensors::tensor::TensorView<'_>) -> Vec<i64> {
    t.data()
        .chunks_exact(8)
        .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
#[test]
#[ignore = "Generate public Transformers fixtures and set MISTRAL3_REFERENCE"]
fn mistral3_packed_hidden_states_match_transformers() {
    let root = PathBuf::from(std::env::var("MISTRAL3_REFERENCE").expect("set MISTRAL3_REFERENCE"));
    let model = std::env::var("MISTRAL3_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| root.join("model"));
    let device = std::env::var("MISTRAL3_TEST_DEVICE").unwrap_or_else(|_| "cpu".into());
    let dtype = std::env::var("MISTRAL3_TEST_DTYPE").unwrap_or_else(|_| "float32".into());
    let pool = match std::env::var("MISTRAL3_TEST_POOLING").as_deref() {
        Ok("mean") => Pool::Mean,
        Ok("last-token") | Err(_) => Pool::LastToken,
        Ok(value) => panic!("Unsupported test pooling: {value}"),
    };
    let backend =
        LibtorchBackend::new(&model, &dtype, ModelType::Embedding(pool.clone()), &device).unwrap();
    let cases: Vec<serde_json::Value> =
        serde_json::from_slice(&std::fs::read(root.join("cases.json")).unwrap()).unwrap();
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
    let mut expected = Vec::new();
    let mut expected_pooled = Vec::new();
    for (row, case) in cases.iter().enumerate() {
        let bytes =
            std::fs::read(root.join(format!("{}.safetensors", case["name"].as_str().unwrap())))
                .unwrap();
        let tensors = SafeTensors::deserialize(&bytes).unwrap();
        let ids = i64_values(tensors.tensor("ids").unwrap())
            .into_iter()
            .map(|v| v as u32)
            .collect::<Vec<_>>();
        let mut images = vec![];
        let mut cursor = 0usize;
        if case["images"].as_u64().unwrap() > 0 {
            let sizes = i64_values(tensors.tensor("sizes").unwrap());
            for (i, size) in sizes.chunks_exact(2).enumerate() {
                let h = size[0] as usize / 14;
                let w = size[1] as usize / 14;
                let start = cursor + ids[cursor..].iter().position(|v| *v == 10).unwrap();
                images.push((
                    start,
                    Arc::new(ImagePatches {
                        pixels: f32_values(tensors.tensor(&format!("pixels_{i}")).unwrap()),
                        grid_thw: [1, h, w],
                        patch_dim: 3 * 14 * 14,
                        merge_size: 2,
                    }),
                ));
                cursor = start + (h / 2) * (w / 2 + 1);
            }
        }
        let positions = (0..ids.len() as u32).collect::<Vec<_>>();
        batch.multimodal.push(Some(Arc::new(MultimodalEncoding {
            images,
            audios: vec![],
            position_ids: std::array::from_fn(|_| positions.clone()),
            memory: None,
        })));
        batch.input_ids.extend(&ids);
        batch.position_ids.extend(positions);
        batch.token_type_ids.extend(vec![0; ids.len()]);
        batch
            .cumulative_seq_lengths
            .push(batch.input_ids.len() as u32);
        batch.max_length = batch.max_length.max(ids.len() as u32);
        batch.raw_indices.push(row as u32);
        let t = tensors.tensor("hidden").unwrap();
        let shape = t.shape().to_vec();
        let values = f32_values(t);
        expected_pooled.push(
            tensors
                .tensor("pooled")
                .map(f32_values)
                .unwrap_or_else(|_| {
                    assert!(
                        matches!(pool, Pool::LastToken),
                        "Mean pooling requires a pooled reference"
                    );
                    values[values.len() - shape[1]..].to_vec()
                }),
        );
        expected.push((shape, values));
    }
    let mut invalid = batch.clone();
    let media = invalid.multimodal[1].as_ref().unwrap();
    let (_, image) = &media.images[0];
    // Feature rows still fit, but the interleaved row-break/end tokens do not.
    invalid.multimodal[1] = Some(Arc::new(MultimodalEncoding {
        images: vec![(expected[1].0[0] - image.token_count(), image.clone())],
        audios: vec![],
        position_ids: media.position_ids.clone(),
        memory: None,
    }));
    assert!(
        backend.embed(invalid).is_err(),
        "Image rows must not cross a packed sequence boundary"
    );
    let mut pooled_batch = batch.clone();
    pooled_batch.pooled_indices = pooled_batch.raw_indices.iter().rev().copied().collect();
    pooled_batch.raw_indices.clear();
    let pooled_output = backend.embed(pooled_batch).unwrap();
    let output = backend.embed(batch).unwrap();
    for (row, wanted) in expected_pooled.iter().enumerate() {
        let Embedding::Pooled(got) = &pooled_output[&row] else {
            panic!("expected pooled embedding")
        };
        assert_close(
            got,
            wanted,
            &dtype,
            &format!("{} pooled", cases[row]["name"]),
        );
    }
    for (row, (shape, wanted)) in expected.iter().enumerate() {
        let Embedding::All(got) = &output[&row] else {
            panic!("expected unpooled hidden states")
        };
        assert_eq!(got.len(), shape[0]);
        assert_eq!(got[0].len(), shape[1]);
        let actual = got.iter().flatten().copied().collect::<Vec<_>>();
        assert_close(&actual, wanted, &dtype, &cases[row]["name"].to_string());
    }
}

fn assert_close(actual: &[f32], wanted: &[f32], dtype: &str, name: &str) {
    assert_eq!(actual.len(), wanted.len());
    let max = actual
        .iter()
        .zip(wanted)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let dot: f64 = actual
        .iter()
        .zip(wanted)
        .map(|(a, b)| *a as f64 * *b as f64)
        .sum();
    let norm = |a: &[f32]| a.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
    let cosine = dot / (norm(actual) * norm(wanted));
    println!("{name}: max={max}, cosine={cosine}");
    let relative = actual
        .iter()
        .zip(wanted)
        .map(|(a, b)| ((*a as f64) - (*b as f64)).powi(2))
        .sum::<f64>()
        .sqrt()
        / norm(wanted);
    // BF16 eager vs SDPA itself differs on this trained model. FP32 is the
    // strict implementation check; BF16 additionally bounds relative drift.
    assert!(if dtype == "float32" {
        max < 0.002
    } else {
        relative < 0.05
    });
    assert!(cosine > if dtype == "float32" { 0.99999 } else { 0.999 });
}
