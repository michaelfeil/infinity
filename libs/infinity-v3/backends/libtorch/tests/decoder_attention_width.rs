//! Pruned decoders may project attention wider than their residual stream.
use infinity_backend_libtorch::LibtorchBackend;
use safetensors::{serialize_to_file, tensor::TensorView, Dtype};
use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
use text_embeddings_backend_core::{Backend, Batch, Embedding, ModelType, Pool};

struct Fixture(PathBuf);
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn fixture(family: &str) -> Fixture {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "infinity-decoder-width-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&path).unwrap();
    // Same 3:1 query/KV head ratio and 3:2 attention/residual width as Nemotron 3 1B.
    let mut config = serde_json::json!({"model_type":family,"hidden_size":16,"vocab_size":32,
        "num_hidden_layers":2,"intermediate_size":24,"num_attention_heads":3,
        "num_key_value_heads":1,"head_dim":8,"max_position_embeddings":32,
        "hidden_act":"silu","rms_norm_eps":1e-5,"rope_theta":10000.,
        "attention_bias":false,"use_bidirectional_attention":true,"is_causal":false});
    if family == "qwen3_moe" {
        config["num_experts"] = 2.into();
        config["num_experts_per_tok"] = 1.into();
        config["moe_intermediate_size"] = 24.into();
        config["decoder_sparse_step"] = 1.into();
    }
    std::fs::write(
        path.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let mut tensors = BTreeMap::new();
    let mut add = |name: String, shape: Vec<usize>, norm: bool| {
        let salt = name.bytes().map(usize::from).sum::<usize>();
        let bytes = (0..shape.iter().product::<usize>())
            .flat_map(|i| {
                (((i + salt) as f32 * 0.13).sin() * 0.08 + if norm { 1. } else { 0. }).to_le_bytes()
            })
            .collect::<Vec<_>>();
        tensors.insert(name, (shape, bytes));
    };
    add("model.embed_tokens.weight".into(), vec![32, 16], false);
    add("model.norm.weight".into(), vec![16], true);
    for i in 0..2 {
        let p = format!("model.layers.{i}");
        for name in ["input_layernorm", "post_attention_layernorm"] {
            add(format!("{p}.{name}.weight"), vec![16], true);
        }
        for (name, width) in [("q_proj", 24), ("k_proj", 8), ("v_proj", 8)] {
            add(
                format!("{p}.self_attn.{name}.weight"),
                vec![width, 16],
                false,
            );
            if family == "qwen2" {
                add(format!("{p}.self_attn.{name}.bias"), vec![width], false);
            }
        }
        add(format!("{p}.self_attn.o_proj.weight"), vec![16, 24], false);
        if family.starts_with("qwen3") {
            for name in ["q_norm", "k_norm"] {
                add(format!("{p}.self_attn.{name}.weight"), vec![8], true);
            }
        }
        let mlps = if family == "qwen3_moe" {
            add(format!("{p}.mlp.gate.weight"), vec![2, 16], false);
            vec![format!("{p}.mlp.experts.0"), format!("{p}.mlp.experts.1")]
        } else {
            vec![format!("{p}.mlp")]
        };
        for mlp in mlps {
            for name in ["gate_proj", "up_proj"] {
                add(format!("{mlp}.{name}.weight"), vec![24, 16], false);
            }
            add(format!("{mlp}.down_proj.weight"), vec![16, 24], false);
        }
    }
    let views = tensors.iter().map(|(name, (shape, bytes))| {
        (
            name.as_str(),
            TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
        )
    });
    serialize_to_file(views, None, &path.join("model.safetensors")).unwrap();
    Fixture(path)
}

fn batch(rows: &[&[u32]]) -> Batch {
    let mut b = Batch {
        input_ids: vec![],
        token_type_ids: vec![],
        position_ids: vec![],
        cumulative_seq_lengths: vec![0],
        max_length: 0,
        pooled_indices: (0..rows.len() as u32).collect(),
        raw_indices: vec![],
        multimodal: vec![],
        compact_input_ids: None,
        compact_position_ids: None,
        scatter_unfold: None,
        fold_gather: None,
        tokens: vec![],
        offsets: vec![],
    };
    for row in rows {
        b.input_ids.extend_from_slice(row);
        b.token_type_ids.extend(vec![0; row.len()]);
        b.position_ids.extend(0..row.len() as u32);
        b.cumulative_seq_lengths.push(b.input_ids.len() as u32);
        b.max_length = b.max_length.max(row.len() as u32);
    }
    b
}

#[test]
fn explicit_decoder_head_width_executes_ragged_batches() {
    for family in [
        "llama",
        "llama_bidirec",
        "mistral",
        "ministral3",
        "qwen2",
        "qwen3",
        "qwen3_moe",
    ] {
        let f = fixture(family);
        let model =
            LibtorchBackend::new(&f.0, "float32", ModelType::Embedding(Pool::Mean), "cpu").unwrap();
        assert!(
            !model.supports_radix_mlp(),
            "{family} must remain bidirectional"
        );
        let rows: [&[u32]; 2] = [&[1, 2, 3], &[4, 5, 6, 7, 8]];
        let together = model.embed(batch(&rows)).unwrap();
        for (i, row) in rows.iter().enumerate() {
            let alone = model.embed(batch(&[row])).unwrap();
            let Embedding::Pooled(a) = &together[&i] else {
                panic!("pooled")
            };
            let Embedding::Pooled(b) = &alone[&0] else {
                panic!("pooled")
            };
            assert_eq!(a.len(), 16);
            for (a, b) in a.iter().zip(b) {
                assert!(
                    a.is_finite() && (a - b).abs() < 1e-5,
                    "{family}: {a} != {b}"
                );
            }
        }
    }
}

#[test]
fn independent_width_still_requires_an_explicit_positive_decoder_head_dim() {
    let f = fixture("ministral3");
    let path = f.0.join("config.json");
    let original: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let mut invalid = vec![];
    for dim in [serde_json::Value::Null, 0.into(), (-8).into()] {
        let mut c = original.clone();
        if dim.is_null() {
            c.as_object_mut().unwrap().remove("head_dim");
        } else {
            c["head_dim"] = dim;
        }
        invalid.push(c);
    }
    let mut encoder = original.clone();
    encoder["model_type"] = "bert".into();
    invalid.push(encoder);
    for config in invalid {
        std::fs::write(&path, serde_json::to_vec(&config).unwrap()).unwrap();
        match LibtorchBackend::new(&f.0, "float32", ModelType::Embedding(Pool::Mean), "cpu") {
            Err(e) => assert!(e.to_string().contains("Invalid model dimensions"), "{e}"),
            Ok(_) => panic!("invalid geometry accepted: {config}"),
        }
    }
    // Passing the scalar check must still validate the saved Q/K/V projection sizes.
    let mut wrong_projection = original;
    wrong_projection["head_dim"] = 16.into();
    std::fs::write(&path, serde_json::to_vec(&wrong_projection).unwrap()).unwrap();
    match LibtorchBackend::new(&f.0, "float32", ModelType::Embedding(Pool::Mean), "cpu") {
        Err(e) => assert!(
            e.to_string().contains("Invalid decoder tensor dimensions"),
            "{e}"
        ),
        Ok(_) => panic!("incompatible projection weights accepted"),
    }
}

#[derive(serde::Deserialize)]
struct ReferenceCase {
    name: String,
    input_ids: Vec<u32>,
    mean: Vec<f32>,
    raw_indices: Vec<usize>,
    raw: Vec<Vec<f32>>,
}

fn compare(actual: &[f32], expected: &[f32], name: &str, dtype: &str) -> bool {
    assert_eq!(actual.len(), expected.len());
    assert!(actual.iter().all(|v| v.is_finite()));
    let norm = |v: &[f32]| v.iter().map(|&x| (x as f64).powi(2)).sum::<f64>();
    let dot = actual
        .iter()
        .zip(expected)
        .map(|(&a, &b)| a as f64 * b as f64)
        .sum::<f64>();
    let cosine = dot / (norm(actual) * norm(expected)).sqrt();
    let error = actual
        .iter()
        .zip(expected)
        .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
        .sum::<f64>();
    let relative = (error / norm(expected)).sqrt();
    let (min_cosine, max_relative) = if dtype == "float32" {
        (0.99999, 0.001)
    } else {
        (0.999, 0.04)
    };
    eprintln!("{name}: cosine={cosine:.9}, relative_l2={relative:.9}");
    cosine > min_cosine && relative < max_relative
}

#[test]
#[ignore = "Generate scripts/nemotron3-reference.py and set NEMOTRON3_REFERENCE; requires trained weights"]
fn nemotron3_packed_embeddings_match_transformers() {
    #[derive(serde::Deserialize)]
    struct Reference {
        model: String,
        dtype: String,
        cases: Vec<ReferenceCase>,
    }
    let reference = std::env::var("NEMOTRON3_REFERENCE").expect("set NEMOTRON3_REFERENCE");
    let r: Reference = serde_json::from_slice(&std::fs::read(reference).unwrap()).unwrap();
    let device = std::env::var("NEMOTRON3_DEVICE").unwrap_or_else(|_| "cuda:0".into());
    let model = LibtorchBackend::new(
        std::path::Path::new(&r.model),
        &r.dtype,
        ModelType::Embedding(Pool::Mean),
        &device,
    )
    .unwrap();
    assert!(
        !model.supports_radix_mlp(),
        "Nemotron must preserve bidirectional attention"
    );
    assert!(r.cases.len() >= 8);
    let rows = r
        .cases
        .iter()
        .map(|c| c.input_ids.as_slice())
        .collect::<Vec<_>>();
    let pooled = model.embed(batch(&rows)).unwrap();
    for (i, c) in r.cases.iter().enumerate() {
        let Embedding::Pooled(actual) = &pooled[&i] else {
            panic!("pooled")
        };
        assert_eq!(actual.len(), 2048);
        assert!(compare(actual, &c.mean, &c.name, &r.dtype));
    }
    // Validate the embedding contract in BF16. Some individual token states in
    // this trained checkpoint are cancellation-sensitive even between HF BF16
    // and HF FP32; use FP32 for strict internal-state comparisons instead.
    if r.dtype == "bfloat16" {
        return;
    }
    // FP32 raw states catch attention, position and pooling errors separately.
    let mut raw_batch = batch(&rows);
    raw_batch.pooled_indices.clear();
    raw_batch.raw_indices = (0..rows.len() as u32).collect();
    let raw = model.embed(raw_batch).unwrap();
    let mut raw_matched = true;
    for (i, c) in r.cases.iter().enumerate() {
        let Embedding::All(actual) = &raw[&i] else {
            panic!("raw")
        };
        assert_eq!(actual.len(), c.input_ids.len());
        for (&token, expected) in c.raw_indices.iter().zip(&c.raw) {
            raw_matched &= compare(
                &actual[token],
                expected,
                &format!("{} token {token}", c.name),
                &r.dtype,
            );
        }
    }
    assert!(raw_matched, "raw token states did not match Transformers");
}
