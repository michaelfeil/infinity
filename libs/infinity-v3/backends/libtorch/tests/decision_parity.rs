//! Synthetic typed decision checkpoints; no network or Python runtime.
use infinity_backend_libtorch::LibtorchBackend;
use safetensors::{
    serialize_to_file,
    tensor::{Dtype, TensorView},
};
use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
use text_embeddings_backend_core::{Backend, Batch, ClefField, DecisionInput, ModelType};
const H: usize = 16;
struct Fixture(PathBuf);
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
#[derive(Default)]
struct Weights(BTreeMap<String, (Vec<usize>, Vec<u8>)>);
impl Weights {
    fn add(&mut self, name: &str, shape: &[usize], norm: bool) {
        let seed = name.bytes().map(usize::from).sum::<usize>() as f32;
        let data = (0..shape.iter().product())
            .flat_map(|i| {
                let value = if norm {
                    1.
                } else {
                    (i as f32 * 0.17 + seed * 0.01).sin() * 0.04
                };
                value.to_le_bytes()
            })
            .collect();
        self.0.insert(name.into(), (shape.into(), data));
    }
    fn linear(&mut self, name: &str, input: usize, output: usize, bias: bool) {
        self.add(&format!("{name}.weight"), &[output, input], false);
        if bias {
            self.add(&format!("{name}.bias"), &[output], false);
        }
    }
    fn norm(&mut self, name: &str, width: usize) {
        self.add(&format!("{name}.weight"), &[width], true);
        self.add(&format!("{name}.bias"), &[width], false);
    }
    fn attention(&mut self, name: &str) {
        self.add(&format!("{name}.in_proj_weight"), &[3 * H, H], false);
        self.add(&format!("{name}.in_proj_bias"), &[3 * H], false);
        self.linear(&format!("{name}.out_proj"), H, H, true);
    }
    fn save(&self, path: PathBuf) {
        let views = self
            .0
            .iter()
            .map(|(n, (s, d))| {
                (
                    n.clone(),
                    TensorView::new(Dtype::F32, s.clone(), d).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        serialize_to_file(views, None, &path).unwrap();
    }
}
fn config_file(path: &std::path::Path, name: &str, value: serde_json::Value) {
    std::fs::write(path.join(name), serde_json::to_vec(&value).unwrap()).unwrap();
}
fn fixture_with_heads(kind: &str, noncausal: bool, query_heads: usize) -> Fixture {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "tei-decision-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&path).unwrap();
    let mut w = Weights::default();
    if kind == "laya" {
        std::fs::create_dir(path.join("encoder")).unwrap();
        config_file(
            &path,
            "encoder/config.json",
            serde_json::json!({"model_type":"modernbert","hidden_size":H,"vocab_size":32,"num_attention_heads":2,"num_hidden_layers":2,"intermediate_size":24,"max_position_embeddings":32,"hidden_activation":"gelu","classifier_activation":"gelu","norm_eps":1e-5,"norm_bias":false,"attention_bias":false,"attention_dropout":0.,"hidden_dropout_prob":0.,"initializer_range":0.02,"initializer_cutoff_factor":2.,"pad_token_id":0,"eos_token_id":2,"bos_token_id":1,"cls_token_id":1,"sep_token_id":2,"global_rope_theta":160000.,"local_rope_theta":10000.,"global_attn_every_n_layers":2,"local_attention":4}),
        );
        config_file(
            &path,
            "rl_agent_config.json",
            serde_json::json!({"encoder":"synthetic","head_layers":1,"max_len":32,"head_max_len":32}),
        );
        w.add("encoder.embeddings.tok_embeddings.weight", &[32, H], false);
        w.add("encoder.embeddings.norm.weight", &[H], true);
        w.add("encoder.final_norm.weight", &[H], true);
        for i in 0..2 {
            let p = format!("encoder.layers.{i}");
            w.linear(&format!("{p}.attn.Wqkv"), H, 3 * H, false);
            w.linear(&format!("{p}.attn.Wo"), H, H, false);
            w.linear(&format!("{p}.mlp.Wi"), H, 48, false);
            w.linear(&format!("{p}.mlp.Wo"), 24, H, false);
            w.add(&format!("{p}.mlp_norm.weight"), &[H], true);
            if i > 0 {
                w.add(&format!("{p}.attn_norm.weight"), &[H], true);
            }
        }
        w.add("type_emb.weight", &[3, H], false);
        w.norm("scorer.0", H);
        w.linear("scorer.1", H, H, true);
        w.linear("scorer.3", H, 1, true);
        w.linear("act_head.0", H + 4, 256, true);
        w.linear("act_head.2", 256, 2, true);
        w.norm("head.layers.0.norm1", H);
        w.norm("head.layers.0.norm2", H);
        w.attention("head.layers.0.self_attn");
        w.linear("head.layers.0.linear1", H, 4 * H, true);
        w.linear("head.layers.0.linear2", 4 * H, H, true);
    } else {
        config_file(
            &path,
            "config.json",
            serde_json::json!({"model_type":"qwen3_5","hidden_size":H,"vocab_size":32,"num_attention_heads":query_heads,"num_key_value_heads":1,"head_dim":8,"num_hidden_layers":1,"intermediate_size":24,"max_position_embeddings":32,"rms_norm_eps":1e-6,"hidden_act":"silu","layer_types":["full_attention"],"linear_num_key_heads":1,"linear_num_value_heads":1,"linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4,"rope_parameters":{"rope_type":"default","rope_theta":10000.,"partial_rotary_factor":0.5}}),
        );
        w.add("model.embed_tokens.weight", &[32, H], false);
        w.add("model.norm.weight", &[H], true);
        w.add("lm_head.weight", &[32, H], false);
        let p = "model.layers.0";
        w.add(&format!("{p}.input_layernorm.weight"), &[H], true);
        w.add(&format!("{p}.post_attention_layernorm.weight"), &[H], true);
        for (name, out) in [
            ("q_proj", 2 * query_heads * 8),
            ("k_proj", 8),
            ("v_proj", 8),
            ("o_proj", H),
        ] {
            w.linear(
                &format!("{p}.self_attn.{name}"),
                if name == "o_proj" { query_heads * 8 } else { H },
                out,
                false,
            );
        }
        w.add(&format!("{p}.self_attn.q_norm.weight"), &[8], true);
        w.add(&format!("{p}.self_attn.k_norm.weight"), &[8], true);
        w.linear(&format!("{p}.mlp.gate_proj"), H, 24, false);
        w.linear(&format!("{p}.mlp.up_proj"), H, 24, false);
        w.linear(&format!("{p}.mlp.down_proj"), 24, H, false);
        if kind == "pplx" {
            config_file(
                &path,
                "decision_config.json",
                serde_json::json!({"pooling":"last","attention_mode":if noncausal {"noncausal_full_attention"}else{"causal"}}),
            );
            let mut head = Weights::default();
            head.add("weight", &[255, H], false);
            head.save(path.join("readout.safetensors"));
        }
        if kind == "clef" {
            config_file(
                &path,
                "joint_head_config.json",
                serde_json::json!({"hidden_size":H,"width":H,"routing_layers":1,"layers":1,"heads":2,"feedforward":32}),
            );
            let mut head = Weights::default();
            for name in [
                "hidden_norm",
                "option_summary_norm",
                "field_norm",
                "option_norm",
            ] {
                head.norm(name, H);
            }
            for name in [
                "memory_projection",
                "question_projection",
                "option_question_projection",
                "global_projection",
                "option_context_projection",
                "option_lexical_projection",
            ] {
                head.linear(name, H, H, false);
            }
            head.add("type_embedding.weight", &[3, H], false);
            head.linear("residual_scorer.0", 4 * H, H, true);
            head.linear("residual_scorer.3", H, 1, true);
            for name in ["prior_logit_scale", "joint_logit_scale", "residual_gate"] {
                head.add(name, &[], false);
            }
            let p = "evidence_layers.0";
            for name in ["query_norm", "memory_norm", "feedforward_norm"] {
                head.norm(&format!("{p}.{name}"), H);
            }
            head.attention(&format!("{p}.attention"));
            head.linear(&format!("{p}.feedforward.0"), H, 32, true);
            head.linear(&format!("{p}.feedforward.3"), 32, H, true);
            let p = "layers.0";
            for name in ["norm1", "norm2", "norm3"] {
                head.norm(&format!("{p}.{name}"), H);
            }
            head.attention(&format!("{p}.self_attn"));
            head.attention(&format!("{p}.multihead_attn"));
            head.linear(&format!("{p}.linear1"), H, 32, true);
            head.linear(&format!("{p}.linear2"), 32, H, true);
            head.save(path.join("joint_head.safetensors"));
        }
    }
    w.save(path.join("model.safetensors"));
    Fixture(path)
}
fn fixture(kind: &str, noncausal: bool) -> Fixture {
    fixture_with_heads(kind, noncausal, 2)
}
fn batch() -> Batch {
    Batch {
        multimodal: vec![],
        input_ids: vec![3, 4, 5, 6, 7],
        token_type_ids: vec![0; 5],
        position_ids: vec![0, 1, 2, 0, 1],
        cumulative_seq_lengths: vec![0, 3, 5],
        max_length: 3,
        pooled_indices: vec![],
        raw_indices: vec![],
        compact_input_ids: None,
        compact_position_ids: None,
        scatter_unfold: None,
        fold_gather: None,
        tokens: vec![],
        offsets: vec![],
    }
}
fn requests(kind: &str) -> Vec<DecisionInput> {
    let first = match kind {
        "laya" => DecisionInput::Laya {
            question_type: 2,
            markers: vec![0, 2],
        },
        "clef" => DecisionInput::Clef {
            fields: vec![
                ClefField {
                    kind: 0,
                    question: (0, 1),
                    options: vec![(1, 2), (2, 3)],
                },
                ClefField {
                    kind: 2,
                    question: (1, 2),
                    options: vec![(0, 2)],
                },
            ],
        },
        "pplx" => DecisionInput::OptionTokens {
            token_ids: vec![0, 1],
        },
        _ => DecisionInput::OptionTokens {
            token_ids: vec![7, 3],
        },
    };
    vec![first, DecisionInput::Warmup]
}
#[test]
fn decision_startup_ragged_outputs_and_recovery_cpu() {
    for kind in ["laya", "option_tokens", "pplx", "clef"] {
        let f = fixture(kind, false);
        let model = LibtorchBackend::new(&f.0, "float32", ModelType::Decision, "cpu").unwrap();
        let inputs = requests(kind);
        let outputs = model.decide(batch(), inputs.clone()).unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].logits.len(), if kind == "clef" { 3 } else { 2 });
        assert_eq!(outputs[1].logits.len(), 1);
        assert!(outputs
            .iter()
            .all(|o| o.logits.iter().all(|x| x.is_finite())
                && o.action_probability.is_finite()
                && (0.0..=1.0).contains(&o.action_probability)));
        assert!(model.decide(batch(), vec![DecisionInput::Warmup]).is_err());
        let wrong = if kind == "laya" {
            DecisionInput::Laya {
                question_type: 0,
                markers: vec![3],
            }
        } else if kind == "clef" {
            DecisionInput::Clef {
                fields: vec![ClefField {
                    kind: 0,
                    question: (0, 4),
                    options: vec![(0, 1)],
                }],
            }
        } else {
            DecisionInput::OptionTokens {
                token_ids: vec![999],
            }
        };
        assert!(model
            .decide(batch(), vec![wrong, DecisionInput::Warmup])
            .is_err());
        let recovered = model.decide(batch(), inputs).unwrap();
        for (a, b) in outputs.iter().zip(recovered) {
            assert_eq!(a.logits, b.logits);
            assert_eq!(a.action_probability, b.action_probability);
        }
    }
}

#[test]
fn qwen35_explicit_head_dimension_preserves_independent_attention_width() {
    let f = fixture_with_heads("pplx", true, 3);
    let model = LibtorchBackend::new(&f.0, "bfloat16", ModelType::Decision, "cpu").unwrap();
    let actual = model.decide(batch(), requests("pplx")).unwrap();
    assert_eq!(actual[0].logits.len(), 2);
    assert!(actual
        .iter()
        .all(|o| o.logits.iter().all(|x| x.is_finite())));
    let config_path = f.0.join("config.json");
    let original: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&config_path).unwrap()).unwrap();
    let mut invalid = original.clone();
    invalid["head_dim"] = 0.into();
    std::fs::write(&config_path, serde_json::to_vec(&invalid).unwrap()).unwrap();
    let reject_dimensions =
        |model_type, dtype| match LibtorchBackend::new(&f.0, dtype, model_type, "cpu") {
            Err(error) => assert!(
                error.to_string().contains("Invalid model dimensions"),
                "Wrong validation failure: {error}"
            ),
            Ok(_) => panic!("Invalid model dimensions accepted"),
        };
    reject_dimensions(ModelType::Decision, "bfloat16");
    invalid = original;
    invalid["model_type"] = "bert".into();
    std::fs::write(&config_path, serde_json::to_vec(&invalid).unwrap()).unwrap();
    reject_dimensions(
        ModelType::Embedding(text_embeddings_backend_core::Pool::Cls),
        "float32",
    );
}
