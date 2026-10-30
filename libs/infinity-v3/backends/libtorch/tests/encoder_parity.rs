#![recursion_limit = "256"]
use infinity_backend_libtorch::LibtorchBackend;
use safetensors::tensor::{serialize, Dtype, TensorView};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
use text_embeddings_backend_core::{Backend, Batch, Embedding, ModelType, Pool};

const H: usize = 32;
const I: usize = 48;
struct Fixture(PathBuf);
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn fixture(family: &str) -> Fixture {
    let variant = family;
    let family = if family.starts_with("nomic-") {
        "nomic_bert"
    } else {
        family
    };
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "tei-encoder-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&path).unwrap();
    let mut cfg = serde_json::json!({
        "model_type":family,"vocab_size":64,"hidden_size":H,"dim":H,"n_embd":H,
        "num_attention_heads":4,"n_heads":4,"n_head":4,"intermediate_size":I,"hidden_dim":I,"n_inner":I,
        "num_hidden_layers":2,"n_layers":2,"n_layer":2,"hidden_act":"gelu","activation":"gelu",
        "activation_function":"gelu","hidden_activation":"gelu","classifier_activation":"gelu",
        "max_position_embeddings":32,"n_positions":32,"type_vocab_size":2,"pad_token_id":0,
        "layer_norm_eps":1e-5,"layer_norm_epsilon":1e-5,"norm_eps":1e-5,
        "hidden_dropout_prob":0.,"attention_probs_dropout_prob":0.,"initializer_range":0.02,
        "initializer_cutoff_factor":2.,"norm_bias":false,"attention_bias":false,"attention_dropout":0.,
        "eos_token_id":2,"bos_token_id":1,"cls_token_id":1,"sep_token_id":2,
        "global_rope_theta":160000.,"local_rope_theta":10000.,"global_attn_every_n_layers":2,"local_attention":4,
        "prenorm":false,"rotary_emb_fraction":1.,"qkv_proj_bias":true,"rotary_emb_base":1000.,
        "rotary_emb_interleaved":false,"mlp_fc1_bias":true,"mlp_fc2_bias":true,
        "layer_norm_type":"layer_norm","position_embedding_type":"rope","rope_theta":10000.,
        "num_labels":3,"id2label":{"0":"A","1":"B","2":"C"}
    });
    if variant == "nomic-gated" {
        cfg["activation_function"] = "swiglu".into();
    }
    if variant == "nomic-moe" {
        cfg["moe_every_n_layers"] = 2.into();
        cfg["num_experts"] = 4.into();
        cfg["moe_top_k"] = 2.into();
    }
    if variant == "nomic-scaled" {
        cfg["rotary_scaling_factor"] = 2.into();
        cfg["max_trained_positions"] = 4.into();
    }
    if family == "jina" || family == "jina-code" {
        cfg["model_type"] = "bert".into();
        cfg["position_embedding_type"] = "alibi".into();
        cfg["_name_or_path"] = if family == "jina" {
            "jinaai/jina-bert-implementation"
        } else {
            "jinaai/jina-bert-v2-qk-post-norm"
        }
        .into();
    }
    if family == "nomic_bert" {
        // Published Nomic configurations use GPT-style dimension names.
        // Exercise the complete loader without duplicate generic aliases.
        for key in [
            "hidden_size",
            "dim",
            "num_attention_heads",
            "n_heads",
            "num_hidden_layers",
            "n_layers",
            "intermediate_size",
            "hidden_dim",
            "max_position_embeddings",
        ] {
            cfg.as_object_mut().unwrap().remove(key);
        }
    }
    fs::write(path.join("config.json"), serde_json::to_vec(&cfg).unwrap()).unwrap();
    let mut data: Vec<(String, Vec<usize>, Vec<u8>)> = vec![];
    let mut add = |name: String, shape: Vec<usize>| {
        let is_norm =
            (name.contains("norm") || name.contains("LayerNorm") || name.contains("emb_ln"))
                && name.ends_with("weight");
        let seed = data.len() as f32 * 0.17;
        let bytes = (0..shape.iter().product())
            .flat_map(|i| {
                let value = if is_norm {
                    1. + (i as f32 * 0.37 + seed).sin() * 0.05
                } else {
                    (i as f32 * 0.19 + seed).sin() * 0.08
                };
                value.to_le_bytes()
            })
            .collect();
        data.push((name, shape, bytes));
    };
    macro_rules! norm {
        ($name:expr) => {
            add(format!("{}.weight", $name), vec![H]);
            add(format!("{}.bias", $name), vec![H]);
        };
    }
    macro_rules! linear {
        ($name:expr,$out:expr,$input:expr,$bias:expr) => {
            add(format!("{}.weight", $name), vec![$out, $input]);
            if $bias {
                add(format!("{}.bias", $name), vec![$out]);
            }
        };
    }
    add(
        if family == "modernbert" {
            "embeddings.tok_embeddings.weight"
        } else {
            "embeddings.word_embeddings.weight"
        }
        .into(),
        vec![64, H],
    );
    if family == "modernbert" {
        add("embeddings.norm.weight".into(), vec![H]);
    } else {
        norm!(if family == "nomic_bert" {
            "emb_ln"
        } else {
            "embeddings.LayerNorm"
        });
    }
    if family != "modernbert" && family != "distilbert" {
        add("embeddings.token_type_embeddings.weight".into(), vec![2, H]);
    }
    if family == "distilbert" || family == "mpnet" {
        add("embeddings.position_embeddings.weight".into(), vec![32, H]);
    }
    if family == "mpnet" {
        add("encoder.relative_attention_bias.weight".into(), vec![32, 4]);
    }
    for i in 0..2 {
        let p = format!(
            "{}{i}.",
            match family {
                "modernbert" => "layers.",
                "nomic_bert" => "encoder.layers.",
                "distilbert" => "transformer.layer.",
                _ => "encoder.layer.",
            }
        );
        match family {
            "mpnet" => {
                for q in ["q", "k", "v", "o"] {
                    linear!(format!("{p}attention.attn.{q}"), H, H, true);
                }
                norm!(format!("{p}attention.LayerNorm"));
                norm!(format!("{p}output.LayerNorm"));
                linear!(format!("{p}intermediate.dense"), I, H, true);
                linear!(format!("{p}output.dense"), H, I, true);
            }
            "distilbert" => {
                for q in ["q_lin", "k_lin", "v_lin", "out_lin"] {
                    linear!(format!("{p}attention.{q}"), H, H, true);
                }
                linear!(format!("{p}ffn.lin1"), I, H, true);
                linear!(format!("{p}ffn.lin2"), H, I, true);
                norm!(format!("{p}sa_layer_norm"));
                norm!(format!("{p}output_layer_norm"));
            }
            "modernbert" => {
                linear!(format!("{p}attn.Wqkv"), 3 * H, H, false);
                linear!(format!("{p}attn.Wo"), H, H, false);
                linear!(format!("{p}mlp.Wi"), 2 * I, H, false);
                linear!(format!("{p}mlp.Wo"), H, I, false);
                if i > 0 {
                    add(format!("{p}attn_norm.weight"), vec![H]);
                }
                add(format!("{p}mlp_norm.weight"), vec![H]);
            }
            "nomic_bert" => {
                linear!(format!("{p}attn.Wqkv"), 3 * H, H, true);
                linear!(format!("{p}attn.out_proj"), H, H, true);
                if variant == "nomic-moe" && i == 1 {
                    linear!(format!("{p}mlp.router.layer"), 4, H, false);
                    add(format!("{p}mlp.experts.mlp.w1"), vec![4 * I, H]);
                    add(format!("{p}mlp.experts.mlp.w2"), vec![4 * I, H]);
                    add(format!("{p}mlp.experts.bias"), vec![H]);
                } else {
                    if variant == "nomic-gated" {
                        linear!(format!("{p}mlp.fc11"), I, H, true);
                        linear!(format!("{p}mlp.fc12"), I, H, true);
                    } else {
                        linear!(format!("{p}mlp.fc1"), I, H, true);
                    }
                    linear!(format!("{p}mlp.fc2"), H, I, true);
                }
                norm!(format!("{p}norm1"));
                norm!(format!("{p}norm2"));
            }
            "new" => {
                linear!(format!("{p}attention.qkv_proj"), 3 * H, H, true);
                linear!(format!("{p}attention.o_proj"), H, H, true);
                linear!(format!("{p}mlp.up_gate_proj"), 2 * I, H, false);
                linear!(format!("{p}mlp.down_proj"), H, I, true);
                norm!(format!("{p}attn_ln"));
                norm!(format!("{p}mlp_ln"));
            }
            _ => {
                for q in ["query", "key", "value"] {
                    linear!(format!("{p}attention.self.{q}"), H, H, true);
                }
                linear!(format!("{p}attention.output.dense"), H, H, true);
                norm!(format!("{p}attention.output.LayerNorm"));
                if family == "jina-code" {
                    norm!(format!("{p}attention.self.layer_norm_q"));
                    norm!(format!("{p}attention.self.layer_norm_k"));
                    norm!(format!("{p}layer_norm_1"));
                    norm!(format!("{p}layer_norm_2"));
                    linear!(format!("{p}mlp.up_gated_layer"), 2 * I, H, false);
                    linear!(format!("{p}mlp.down_layer"), H, I, true);
                } else {
                    linear!(format!("{p}mlp.gated_layers"), 2 * I, H, false);
                    linear!(format!("{p}mlp.wo"), H, I, true);
                    norm!(format!("{p}mlp.layernorm"));
                }
            }
        }
    }
    if family == "modernbert" {
        add("final_norm.weight".into(), vec![H]);
        linear!("head.dense", H, H, false);
        add("head.norm.weight".into(), vec![H]);
    }
    if ["modernbert", "new", "jina"].contains(&family) {
        linear!("classifier", 3, H, true);
    }
    if family == "distilbert" {
        linear!("vocab_transform", H, H, true);
        norm!("vocab_layer_norm");
        linear!("vocab_projector", 64, H, true);
    }
    let views = data
        .iter()
        .map(|(n, s, d)| {
            (
                n.clone(),
                TensorView::new(Dtype::F32, s.clone(), d).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    fs::write(
        path.join("model.safetensors"),
        serialize(views, None).unwrap(),
    )
    .unwrap();
    Fixture(path)
}
fn batch() -> Batch {
    Batch {
        input_ids: vec![1, 4, 7, 2, 5, 8, 3, 6],
        token_type_ids: vec![0, 0, 0, 1, 1, 1, 1, 1],
        position_ids: vec![0, 1, 2, 0, 1, 2, 3, 4],
        cumulative_seq_lengths: vec![0, 3, 8],
        max_length: 5,
        pooled_indices: vec![1],
        raw_indices: vec![0],
        multimodal: vec![],
        compact_input_ids: None,
        compact_position_ids: None,
        scatter_unfold: None,
        fold_gather: None,
        tokens: vec![],
        offsets: vec![],
    }
}
fn values(e: &Embedding) -> Vec<f32> {
    match e {
        Embedding::Pooled(v) => v.clone(),
        Embedding::All(v) => v.iter().flatten().copied().collect(),
    }
}

#[test]
fn encoders_cpu_preserve_packed_sequence_boundaries() {
    for family in [
        "distilbert",
        "modernbert",
        "nomic_bert",
        "nomic-gated",
        "nomic-moe",
        "new",
        "jina",
        "jina-code",
        "mpnet",
    ] {
        let fixture = fixture(family);
        let model = LibtorchBackend::new(
            &fixture.0,
            "float32",
            ModelType::Embedding(Pool::Mean),
            "cpu",
        )
        .unwrap();
        let packed = model.embed(batch()).unwrap();
        let mut single = batch();
        single.input_ids = single.input_ids[3..].to_vec();
        single.token_type_ids = vec![1; 5];
        single.position_ids = vec![0, 1, 2, 3, 4];
        single.cumulative_seq_lengths = vec![0, 5];
        single.pooled_indices = vec![0];
        single.raw_indices.clear();
        let separate = model.embed(single).unwrap();
        for (a, b) in values(&packed[&1]).iter().zip(values(&separate[&0])) {
            assert!(
                (a - b).abs() < 1e-5,
                "{family}: sequence contamination {a} vs {b}"
            );
        }
    }
}

#[test]
fn splade_selected_spans_preserve_mixed_raw_order() {
    let fixture = fixture("distilbert");
    let cuda_device =
        std::env::var("INFINITY_TEST_CUDA_DEVICE").unwrap_or_else(|_| "cuda:0".into());
    let mut devices = vec![("cpu", "float32", 0.00001)];
    if std::env::var_os("INFINITY_TEST_CUDA").is_some() {
        devices.push((cuda_device.as_str(), "float16", 0.012));
    }
    for (device, dtype, tolerance) in devices {
        let backend = LibtorchBackend::new(
            &fixture.0,
            dtype,
            ModelType::Embedding(Pool::Splade),
            device,
        )
        .unwrap();
        let mut input = batch();
        input.input_ids.extend([9, 10]);
        input.token_type_ids.extend([0, 0]);
        input.position_ids.extend([0, 1]);
        input.cumulative_seq_lengths.push(10);
        input.pooled_indices = vec![2, 1];
        let mixed = backend.embed(input.clone()).unwrap();
        let mut pooled = input.clone();
        pooled.pooled_indices = vec![0, 1, 2];
        pooled.raw_indices.clear();
        let pooled = backend.embed(pooled).unwrap();
        input.pooled_indices.clear();
        let raw = backend.embed(input).unwrap();
        for row in [2, 1] {
            assert_eq!(values(&mixed[&row]).len(), 64);
            for (a, b) in values(&mixed[&row]).iter().zip(values(&pooled[&row])) {
                assert!(
                    (a - b).abs() <= tolerance,
                    "selected SPLADE span {row}: {a} != {b}"
                );
            }
        }
        assert_eq!(
            values(&mixed[&0]),
            values(&raw[&0]),
            "raw hidden rows changed during packed SPLADE"
        );
    }
}

#[test]
fn mpnet_cuda_varlen_matches_cpu() {
    // MPNet has no packed Candle implementation, so compare the native GPU
    // memory-efficient varlen operator to its independent CPU SDPA path.
    if std::env::var_os("INFINITY_TEST_CUDA").is_none() {
        return;
    }
    let fixture = fixture("mpnet");
    let cpu = LibtorchBackend::new(
        &fixture.0,
        "float32",
        ModelType::Embedding(Pool::Mean),
        "cpu",
    )
    .unwrap();
    let gpu = LibtorchBackend::new(
        &fixture.0,
        "float16",
        ModelType::Embedding(Pool::Mean),
        "cuda:2",
    )
    .unwrap();
    let expected = cpu.embed(batch()).unwrap();
    let mut got = gpu.embed(batch()).unwrap();
    if std::env::var(infinity_v3_config::infinity_env!("TORCH_CUDA_GRAPHS"))
        .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
    {
        // The default admission policy captures after three identical requests.
        for _ in 0..2 {
            got = gpu.embed(batch()).unwrap();
        }
        assert!(
            gpu.cuda_graph_count() > 0,
            "MPNet CUDA graph was not captured"
        );
    }
    for row in [0, 1] {
        for (a, b) in values(&got[&row]).iter().zip(values(&expected[&row])) {
            assert!((a - b).abs() < 0.008, "MPNet varlen {a} != {b}");
        }
    }
}

#[test]
#[ignore = "Generate encoder_reference_fixtures.py and set INFINITY_ENCODER_REFERENCE"]
fn mpnet_and_deberta_match_independent_transformers() {
    let root = PathBuf::from(std::env::var("INFINITY_ENCODER_REFERENCE").unwrap());
    let mut count = 0;
    for entry in fs::read_dir(root).unwrap() {
        let path = entry.unwrap().path();
        if !path.is_dir() {
            continue;
        }
        count += 1;
        let reference = fs::read(path.join("reference.safetensors")).unwrap();
        let tensors = safetensors::SafeTensors::deserialize(&reference).unwrap();
        let tensor = tensors.tensor("hidden").unwrap();
        let expected = tensor
            .data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        for (dtype, device, tolerance) in [("float32", "cpu", 2e-5), ("float16", "cuda:2", 0.006)] {
            let model =
                LibtorchBackend::new(&path, dtype, ModelType::Embedding(Pool::Mean), device)
                    .unwrap();
            let mut b = input_json_batch(&path);
            b.token_type_ids = vec![
                if path.file_name().unwrap() == "mpnet" {
                    0
                } else {
                    1
                };
                b.input_ids.len()
            ];
            b.pooled_indices.clear();
            b.raw_indices = vec![0, 1];
            let result = model.embed(b).unwrap();
            let actual = [values(&result[&0]), values(&result[&1])].concat();
            assert_eq!(actual.len(), expected.len());
            let max = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            println!("{} {dtype} Transformers max abs {max}", path.display());
            assert!(
                max < tolerance,
                "{} {dtype} reference error {max}",
                path.display()
            );
            if let Ok(logits) = tensors.tensor("logits") {
                let reference = logits
                    .data()
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect::<Vec<_>>();
                let classifier =
                    LibtorchBackend::new(&path, dtype, ModelType::Classifier, device).unwrap();
                let input = input_json_batch(&path);
                let actual = if path
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .ends_with("token")
                {
                    let prediction = classifier.predict_tokens(input).unwrap();
                    [
                        prediction[&0].iter().flatten().copied().collect::<Vec<_>>(),
                        prediction[&1].iter().flatten().copied().collect::<Vec<_>>(),
                    ]
                    .concat()
                } else {
                    let prediction = classifier.predict(input).unwrap();
                    [prediction[&0].clone(), prediction[&1].clone()].concat()
                };
                for (a, b) in actual.iter().zip(reference) {
                    assert!((a - b).abs() < tolerance, "DeBERTa classifier {a} != {b}");
                }
            }
        }
    }
    assert_eq!(
        count, 7,
        "Expected MPNet, four DeBERTa encoders, two classifiers"
    );
}

fn input_json_batch(path: &std::path::Path) -> Batch {
    let data: serde_json::Value =
        serde_json::from_slice(&fs::read(path.join("inputs.json")).unwrap()).unwrap();
    let lengths: Vec<u32> = serde_json::from_value(data["lengths"].clone()).unwrap();
    let mut input = batch();
    input.input_ids = serde_json::from_value(data["ids"].clone()).unwrap();
    input.token_type_ids = vec![1; input.input_ids.len()];
    input.position_ids = lengths.iter().flat_map(|&n| 0..n).collect();
    input.cumulative_seq_lengths = vec![0];
    for &n in &lengths {
        input
            .cumulative_seq_lengths
            .push(input.cumulative_seq_lengths.last().unwrap() + n);
    }
    input.max_length = *lengths.iter().max().unwrap();
    input.pooled_indices = (0..lengths.len() as u32).collect();
    input.raw_indices.clear();
    input
}
