//! Synthetic CUDA checkpoints exercise both runtimes without network downloads.
#![cfg(feature = "benchmark-cuda")]
use infinity_backend_libtorch::LibtorchBackend;
use safetensors::{
    serialize_to_file,
    tensor::{Dtype, TensorView},
};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};
use text_embeddings_backend_candle::CandleBackend;
use text_embeddings_backend_core::{Backend, Batch, Embedding, ModelType, Pool};

struct Fixture(PathBuf);
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn fixture(kind: &str, bidirectional: bool, fused_experts: bool) -> Fixture {
    let dir = std::env::temp_dir().join(format!(
        "tei-decoder-parity-{}-{kind}-{bidirectional}-{fused_experts}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let moe = kind == "qwen3_moe";
    let mut config = serde_json::json!({"model_type":kind,"architectures":["Qwen3Model"],
        "vocab_size":32,"hidden_size":32,"intermediate_size":48,"num_hidden_layers":2,
        "num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,"hidden_act":"silu",
        "max_position_embeddings":32,"initializer_range":0.02,"rms_norm_eps":0.000001,
        "rope_theta":10000.0,"attention_bias":false,"use_sliding_window":false,
        "eos_token_id":2,"use_bidirectional_attention":bidirectional,"is_causal":!bidirectional});
    if kind.starts_with("qwen3") {
        config["num_labels"] = 16.into();
        config["use_linear_output_projection"] = true.into();
        config["linear_output_size"] = 24.into();
    }
    if moe {
        config["num_experts"] = 3.into();
        config["num_experts_per_tok"] = 2.into();
        config["moe_intermediate_size"] = 24.into();
        config["decoder_sparse_step"] = 2.into();
    }
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let mut tensors: BTreeMap<String, (Vec<usize>, Vec<u8>)> = BTreeMap::new();
    let mut add = |name: String, shape: Vec<usize>, norm: bool| {
        let seed = name.bytes().map(usize::from).sum::<usize>();
        let bytes = (0..shape.iter().product::<usize>())
            .flat_map(|i| {
                let value = ((i + seed) as f32 * 0.13).sin() * 0.08 + if norm { 1. } else { 0. };
                value.to_le_bytes()
            })
            .collect();
        tensors.insert(name, (shape, bytes));
    };
    add("model.embed_tokens.weight".into(), vec![32, 32], false);
    add("model.norm.weight".into(), vec![32], true);
    if kind.starts_with("qwen3") {
        add("linear.weight".into(), vec![16, 32], false);
        add(
            "model.linear_output_projection.weight".into(),
            vec![24, 32],
            false,
        );
        add(
            "model.linear_output_projection.bias".into(),
            vec![24],
            false,
        );
    }
    add("score.weight".into(), vec![2, 32], false);
    if fused_experts {
        add("score.bias".into(), vec![2], false);
    }
    for i in 0..2 {
        let p = format!("model.layers.{i}.");
        add(format!("{p}input_layernorm.weight"), vec![32], true);
        add(
            format!("{p}post_attention_layernorm.weight"),
            vec![32],
            true,
        );
        for (name, width) in [("q_proj", 32), ("k_proj", 16), ("v_proj", 16)] {
            add(
                format!("{p}self_attn.{name}.weight"),
                vec![width, 32],
                false,
            );
            if kind == "qwen2" {
                add(format!("{p}self_attn.{name}.bias"), vec![width], false);
            }
        }
        add(format!("{p}self_attn.o_proj.weight"), vec![32, 32], false);
        if kind.starts_with("qwen3") {
            for name in ["q_norm", "k_norm"] {
                add(format!("{p}self_attn.{name}.weight"), vec![8], true);
            }
        }
        if moe && i == 1 {
            add(format!("{p}mlp.gate.weight"), vec![3, 32], false);
            if fused_experts {
                add(
                    format!("{p}mlp.experts.gate_up_proj"),
                    vec![3, 48, 32],
                    false,
                );
                add(format!("{p}mlp.experts.down_proj"), vec![3, 32, 24], false);
            } else {
                for e in 0..3 {
                    for name in ["gate_proj", "up_proj"] {
                        add(
                            format!("{p}mlp.experts.{e}.{name}.weight"),
                            vec![24, 32],
                            false,
                        );
                    }
                    add(
                        format!("{p}mlp.experts.{e}.down_proj.weight"),
                        vec![32, 24],
                        false,
                    );
                }
            }
        } else {
            for name in ["gate_proj", "up_proj"] {
                add(format!("{p}mlp.{name}.weight"), vec![48, 32], false);
            }
            add(format!("{p}mlp.down_proj.weight"), vec![32, 48], false);
        }
    }
    let views = tensors.iter().map(|(name, (shape, bytes))| {
        (
            name.as_str(),
            TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
        )
    });
    serialize_to_file(views, None, &dir.join("model.safetensors")).unwrap();
    Fixture(dir)
}
fn batch() -> Batch {
    Batch {
        input_ids: vec![3, 4, 5, 6, 7, 8, 9, 10, 11],
        token_type_ids: vec![0; 9],
        position_ids: vec![0, 1, 2, 0, 1, 2, 3, 4, 5],
        cumulative_seq_lengths: vec![0, 3, 9],
        max_length: 6,
        pooled_indices: vec![0],
        raw_indices: vec![1],
        multimodal: vec![],
        compact_input_ids: None,
        compact_position_ids: None,
        scatter_unfold: None,
        fold_gather: None,
        tokens: vec![],
        offsets: vec![],
    }
}
fn compare(path: &Path, pool: Pool) {
    compare_dtype(path, pool, "float16");
}
fn compare_dtype(path: &Path, pool: Pool, dtype: &str) {
    let torch =
        LibtorchBackend::new(path, dtype, ModelType::Embedding(pool.clone()), "cuda:0").unwrap();
    let candle =
        CandleBackend::new(path, dtype.into(), ModelType::Embedding(pool), None, 1).unwrap();
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path.join("config.json")).unwrap()).unwrap();
    assert_eq!(
        torch.supports_radix_mlp(),
        !config["use_bidirectional_attention"].as_bool().unwrap(),
        "Router capability must match the tested causal/bidirectional decoder"
    );
    let actual = torch.embed(batch()).unwrap();
    let expected = candle.embed(batch()).unwrap();
    assert_eq!(actual.len(), expected.len());
    for (row, a) in actual {
        let b = &expected[&row];
        let (a, b) = match (a, b) {
            (Embedding::Pooled(a), Embedding::Pooled(b)) => (a, b.clone()),
            (Embedding::All(a), Embedding::All(b)) => (
                a.into_iter().flatten().collect(),
                b.iter().flatten().copied().collect(),
            ),
            _ => panic!("output type differs"),
        };
        assert_eq!(a.len(), b.len());
        let error = a
            .iter()
            .zip(&b)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let dot = a
            .iter()
            .zip(&b)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum::<f64>();
        let norm = |v: &[f32]| v.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt();
        let cosine = dot / (norm(&a) * norm(&b));
        assert!(
            error < 0.03 && cosine > 0.999,
            "row={row} max_abs={error} cosine={cosine}"
        );
    }
}
#[test]
#[ignore = "requires two CUDA GPUs and LibTorch 2.14.1"]
fn candle_decoder_parity() {
    for kind in ["llama", "mistral", "qwen2", "qwen3", "qwen3_moe"] {
        for bidirectional in [false, true] {
            for fused in if kind == "qwen3_moe" {
                vec![false, true]
            } else {
                vec![false]
            } {
                let checkpoint = fixture(kind, bidirectional, fused);
                for pool in [Pool::LastToken, Pool::Mean] {
                    eprintln!("parity kind={kind} bidirectional={bidirectional} fused={fused} pool={pool:?}");
                    compare(&checkpoint.0, pool.clone());
                    if kind == "qwen3_moe" {
                        compare_dtype(&checkpoint.0, pool, "bfloat16");
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires two CUDA GPUs and LibTorch 2.14.1"]
fn candle_decoder_rope_and_windows_parity() {
    for (kind, variant) in [
        ("llama", "llama3"),
        ("llama", "ntk"),
        ("mistral", "window"),
        ("ministral3", "yarn"),
    ] {
        for bidirectional in [false, true] {
            let checkpoint = fixture(kind, bidirectional, false);
            let path = checkpoint.0.join("config.json");
            let mut cfg: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
            match variant {
                "llama3" => {
                    cfg["rope_scaling"] = serde_json::json!({"rope_type":"llama3", "factor":4., "low_freq_factor":1., "high_freq_factor":4., "original_max_position_embeddings":16})
                }
                "ntk" => cfg["rope_scaling"] = serde_json::json!({"type":"dynamic", "factor":2.}),
                "window" => cfg["sliding_window"] = 3.into(),
                "yarn" => {
                    cfg["rope_parameters"] = serde_json::json!({"rope_type":"yarn", "rope_theta":10000., "factor":4., "beta_fast":32., "beta_slow":1., "original_max_position_embeddings":4, "mscale":1., "mscale_all_dim":0., "llama_4_scaling_beta":0.1});
                    // Official Ministral3 omits the legacy root theta. The
                    // four-position original context also exercises beta scaling.
                    cfg.as_object_mut().unwrap().remove("rope_theta");
                }
                _ => unreachable!(),
            }
            std::fs::write(&path, serde_json::to_vec(&cfg).unwrap()).unwrap();
            eprintln!(
                "RoPE/window parity kind={kind} variant={variant} bidirectional={bidirectional}"
            );
            compare(&checkpoint.0, Pool::LastToken);
        }
    }
}

#[test]
#[ignore = "requires two CUDA GPUs and LibTorch 2.14.1"]
fn candle_decoder_classifier_parity() {
    for (kind, architecture) in [
        ("llama", "LlamaForSequenceClassification"),
        ("qwen2", "Qwen2ForSequenceClassification"),
        ("qwen3", "Qwen3ForSequenceClassification"),
        ("qwen3_moe", "Qwen3MoeForSequenceClassification"),
    ] {
        for bias in [false, true] {
            let checkpoint = fixture(kind, false, bias);
            let path = checkpoint.0.join("config.json");
            let mut cfg: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
            cfg["architectures"] = serde_json::json!([architecture]);
            cfg["num_labels"] = 2.into();
            cfg["id2label"] = serde_json::json!({"0":"no","1":"yes"});
            cfg["pad_token_id"] = 0.into();
            cfg["use_linear_output_projection"] = false.into();
            std::fs::write(&path, serde_json::to_vec(&cfg).unwrap()).unwrap();
            let torch =
                LibtorchBackend::new(&checkpoint.0, "float16", ModelType::Classifier, "cuda:0")
                    .unwrap();
            let candle = CandleBackend::new(
                &checkpoint.0,
                "float16".into(),
                ModelType::Classifier,
                None,
                1,
            )
            .unwrap();
            let mut input = batch();
            input.input_ids = vec![1, 2, 0, 0, 3, 0, 4, 0, 0];
            input.position_ids = vec![0, 1, 2, 3, 0, 1, 2, 0, 1];
            input.cumulative_seq_lengths = vec![0, 4, 7, 9];
            input.max_length = 4;
            input.pooled_indices = vec![0, 1, 2];
            input.raw_indices.clear();
            let actual = torch.predict(input.clone()).unwrap();
            let expected = candle.predict(input).unwrap();
            for row in 0..3 {
                let error = actual[&row]
                    .iter()
                    .zip(&expected[&row])
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(
                    error < 0.005,
                    "classifier kind={kind} bias={bias} row={row} max_abs={error}"
                );
            }
            eprintln!("classifier parity kind={kind} bias={bias} passed");
        }
    }
}
