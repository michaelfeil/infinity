// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.

use super::*;
use half::{bf16, f16};
use safetensors::Dtype;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

const H: usize = 512;
const I: usize = 8;
const V: usize = 4;
const T: usize = 2;

#[derive(Clone)]
struct TensorSpec {
    name: String,
    shape: Vec<usize>,
    probes: Vec<(usize, f32)>,
}

impl TensorSpec {
    fn new(root: &str, relative: &str, shape: &[usize]) -> Self {
        Self {
            name: format!("{root}{relative}"),
            shape: shape.to_vec(),
            probes: Vec::new(),
        }
    }
}

fn layer(root: &str, relative: &str, shape: &[usize]) -> TensorSpec {
    TensorSpec::new(root, &format!("encoder.layer.0.{relative}"), shape)
}

fn default_specs(root: &str) -> Vec<TensorSpec> {
    vec![
        TensorSpec::new(root, "embeddings.word_embeddings.weight", &[V, H]),
        TensorSpec::new(root, "embeddings.token_type_embeddings.weight", &[T, H]),
        TensorSpec::new(root, "embeddings.LayerNorm.weight", &[H]),
        TensorSpec::new(root, "embeddings.LayerNorm.bias", &[H]),
        layer(root, "attention.self.query.weight", &[H, H]),
        layer(root, "attention.self.query.bias", &[H]),
        layer(root, "attention.self.key.weight", &[H, H]),
        layer(root, "attention.self.key.bias", &[H]),
        layer(root, "attention.self.value.weight", &[H, H]),
        layer(root, "attention.self.value.bias", &[H]),
        layer(root, "attention.output.dense.weight", &[H, H]),
        layer(root, "attention.output.dense.bias", &[H]),
        layer(root, "attention.output.LayerNorm.weight", &[H]),
        layer(root, "attention.output.LayerNorm.bias", &[H]),
        layer(root, "mlp.gated_layers.weight", &[I * 2, H]),
        layer(root, "mlp.wo.weight", &[H, I]),
        layer(root, "mlp.wo.bias", &[H]),
        layer(root, "mlp.layernorm.weight", &[H]),
        layer(root, "mlp.layernorm.bias", &[H]),
    ]
}

fn default_config() -> Value {
    json!({
        "_name_or_path": "jinaai/jina-bert-implementation",
        "architectures": ["JinaBertForMaskedLM"],
        "attention_probs_dropout_prob": 0.0,
        "classifier_dropout": null,
        "emb_pooler": "mean",
        "feed_forward_type": "geglu",
        "gradient_checkpointing": false,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": H,
        "initializer_range": 0.02,
        "intermediate_size": I,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 8192,
        "model_max_length": 8192,
        "model_type": "bert",
        "num_attention_heads": 8,
        "num_hidden_layers": 1,
        "pad_token_id": 0,
        "position_embedding_type": "alibi",
        "torch_dtype": "float32",
        "transformers_version": "4.26.0",
        "type_vocab_size": T,
        "use_cache": true,
        "vocab_size": V
    })
}

struct Fixture {
    path: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "infinity-jina-loader-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        let fixture = Self { path };
        fixture.config(json!({}));
        fixture
    }

    fn config(&self, overrides: Value) {
        let mut config = default_config();
        for (key, value) in overrides.as_object().expect("config overrides") {
            if value.is_null() {
                config.as_object_mut().unwrap().remove(key.as_str());
            } else {
                config[key.as_str()] = value.clone();
            }
        }
        fs::write(
            self.path.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
    }

    fn write(&self, specs: &[TensorSpec]) {
        let mut entries = BTreeMap::new();
        let mut placements = Vec::new();
        let mut cursor = 0usize;
        for spec in specs {
            let count = spec
                .shape
                .iter()
                .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
                .unwrap();
            let end = cursor.checked_add(count.checked_mul(2).unwrap()).unwrap();
            entries.insert(
                spec.name.clone(),
                json!({
                    "dtype": "F16",
                    "shape": &spec.shape,
                    "data_offsets": [cursor, end],
                }),
            );
            placements.push((spec, cursor));
            cursor = end;
        }
        let mut header = serde_json::to_vec(&entries).unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let data_start = 8usize.checked_add(header.len()).unwrap();
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(self.path.join("model.safetensors"))
            .unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.set_len((data_start + cursor) as u64).unwrap();
        for (spec, offset_bytes) in placements {
            for (index, value) in &spec.probes {
                let byte = data_start
                    .checked_add(offset_bytes)
                    .and_then(|offset| offset.checked_add(index.checked_mul(2)?))
                    .unwrap();
                file.seek(SeekFrom::Start(byte as u64)).unwrap();
                file.write_all(&f16::from_f32(*value).to_bits().to_le_bytes())
                    .unwrap();
            }
        }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn load(fixture: &Fixture) -> JinaWeights {
    JinaWeights::load(&fixture.path).unwrap()
}

fn rejected(fixture: &Fixture) -> bool {
    JinaWeights::load(&fixture.path).is_err()
}

fn without(mut specs: Vec<TensorSpec>, suffix: &str) -> Vec<TensorSpec> {
    specs.retain(|spec| !spec.name.ends_with(suffix));
    specs
}

fn probe_bits(source: &crate::model::WeightSource) -> u16 {
    let mut bits = [0u16; 1];
    source.read_bf16(0, &mut bits).unwrap();
    bits[0]
}

fn expected_bits(value: f32) -> u16 {
    bf16::from_f32(f16::from_f32(value).to_f32()).to_bits()
}

#[test]
fn loads_bare_and_bert_prefixed_parts() {
    for root in ["", "bert."] {
        let fixture = Fixture::new();
        fixture.write(&default_specs(root));
        let weights = load(&fixture);
        assert_eq!(weights.geometry.hidden_size, H);
        assert_eq!(weights.geometry.intermediate_size, I);
        assert_eq!(weights.layers.len(), 1);
        assert_eq!(weights.word.shape(), &[V, H]);
        assert_eq!(weights.token_type.shape(), &[T, H]);
        assert_eq!(weights.gamma.shape(), &[H]);
        assert_eq!(weights.beta.shape(), &[H]);

        let layer = &weights.layers[0];
        assert_eq!(layer.qkv.weight_parts.len(), 3);
        assert_eq!(layer.qkv.bias_parts.as_ref().unwrap().len(), 3);
        for part in &layer.qkv.weight_parts {
            assert_eq!(part.shape(), &[H, H]);
        }
        for part in layer.qkv.bias_parts.as_ref().unwrap() {
            assert_eq!(part.shape(), &[H]);
        }
        assert_eq!(layer.attention.weight_parts.len(), 1);
        assert_eq!(layer.attention.bias_parts.as_ref().unwrap().len(), 1);
        assert_eq!(layer.intermediate.weight_parts.len(), 2);
        assert!(layer.intermediate.bias_parts.is_none());
        assert_eq!(layer.intermediate.weight_parts[0].shape(), &[I, H]);
        assert_eq!(layer.intermediate.weight_parts[1].shape(), &[I, H]);
        assert_eq!(layer.output.weight_parts[0].shape(), &[H, I]);
        assert_eq!(layer.output.bias_parts.as_ref().unwrap()[0].shape(), &[H]);
        for norm in [
            &layer.attention_gamma,
            &layer.attention_beta,
            &layer.output_gamma,
            &layer.output_beta,
        ] {
            assert_eq!(norm.shape(), &[H]);
        }
    }
}

#[test]
fn canonicalizes_gate_first_geglu_to_up_then_gate() {
    let fixture = Fixture::new();
    let mut specs = default_specs("");
    for spec in &mut specs {
        if spec.name.ends_with("mlp.gated_layers.weight") {
            spec.probes = vec![(0, 3.25), (I * H, -2.75)];
        }
    }
    fixture.write(&specs);
    let weights = load(&fixture);
    let parts = &weights.layers[0].intermediate.weight_parts;
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0].shape(), &[I, H]);
    assert_eq!(parts[1].shape(), &[I, H]);
    assert_eq!(probe_bits(&parts[0]), expected_bits(-2.75));
    assert_eq!(probe_bits(&parts[1]), expected_bits(3.25));
}

#[test]
fn accepts_name_or_path_and_auto_map_markers_with_f16_headers() {
    let fixture = Fixture::new();
    fixture.write(&default_specs(""));
    let weights = load(&fixture);
    assert_eq!(weights.geometry.hidden_size, H);
    assert_eq!(weights.geometry.num_attention_heads, 8);
    assert_eq!(weights.geometry.num_hidden_layers, 1);
    assert_eq!(weights.geometry.max_position_embeddings, 8192);
    assert_eq!(weights.geometry.vocab_size, V);
    assert_eq!(weights.geometry.type_vocab_size, T);
    assert_eq!(weights.geometry.hidden_size, H);
    assert_eq!(weights.geometry.intermediate_size, I);
    assert_eq!(weights.geometry.num_attention_heads, 8);
    assert_eq!(weights.geometry.num_hidden_layers, 1);
    assert_eq!(weights.geometry.max_position_embeddings, 8192);
    assert_eq!(weights.geometry.vocab_size, V);
    assert_eq!(weights.geometry.type_vocab_size, T);
    assert_eq!(weights.word.dtype(), Dtype::F16);

    fixture.config(json!({
        "_name_or_path": null,
        "auto_map": {
            "AutoConfig": "jinaai/jina-bert-implementation--configuration_bert.JinaBertConfig"
        }
    }));
    let weights = load(&fixture);
    assert_eq!(weights.geometry.vocab_size, V);
}

#[test]
fn rejects_non_jina_and_unsupported_configs() {
    let fixture = Fixture::new();
    fixture.write(&default_specs(""));
    let cases = [
        (
            "ordinary bert",
            json!({"_name_or_path": "bert-base-uncased"}),
        ),
        (
            "jina code",
            json!({"_name_or_path": "jinaai/jina-bert-v2-qk-post-norm"}),
        ),
        (
            "ordinary architecture",
            json!({"architectures": ["BertModel"]}),
        ),
        ("hidden", json!({"hidden_size": 256})),
        ("heads", json!({"num_attention_heads": 4})),
        ("intermediate", json!({"intermediate_size": 7})),
        ("eps", json!({"layer_norm_eps": 0.0})),
        (
            "rotary",
            json!({"position_embedding_type": "rope", "rope_scaling": {"type": "linear"}}),
        ),
        (
            "quantized",
            json!({"quantization_config": {"load_in_4bit": true}}),
        ),
        ("moe", json!({"num_experts": 2})),
        ("decoder", json!({"is_decoder": true})),
        ("cross attention", json!({"add_cross_attention": true})),
        ("parallel", json!({"parallel_block": true})),
        ("qk post norm", json!({"qk_post_norm": true})),
        ("feed forward", json!({"feed_forward_type": "mlp"})),
        ("activation", json!({"hidden_act": "relu"})),
    ];
    for (name, overrides) in cases {
        fixture.config(overrides);
        assert!(rejected(&fixture), "accepted {name} config");
    }
}

#[test]
fn binds_type_vocab_rows_and_rejects_learned_positions() {
    let fixture = Fixture::new();
    fixture.config(json!({"type_vocab_size": null}));
    fixture.write(&default_specs(""));
    let weights = load(&fixture);
    assert_eq!(weights.geometry.type_vocab_size, T);
    assert_eq!(weights.token_type.shape(), &[T, H]);

    fixture.config(json!({"type_vocab_size": T + 1}));
    assert!(rejected(&fixture), "accepted mismatched type vocab");

    fixture.config(json!({}));
    let mut specs = default_specs("");
    specs.push(TensorSpec::new(
        "",
        "embeddings.position_embeddings.weight",
        &[8192, H],
    ));
    fixture.write(&specs);
    assert!(rejected(&fixture), "accepted learned position table");
}

#[test]
fn accepts_exact_pooler_pair_and_rejects_half_or_wrong_shapes() {
    let fixture = Fixture::new();
    fixture.write(&default_specs(""));
    assert!(JinaWeights::load(&fixture.path).is_ok());

    for root in ["", "bert."] {
        let mut specs = default_specs(root);
        specs.push(TensorSpec::new(root, "pooler.dense.weight", &[H, H]));
        specs.push(TensorSpec::new(root, "pooler.dense.bias", &[H]));
        fixture.write(&specs);
        assert!(JinaWeights::load(&fixture.path).is_ok());
    }

    let mut specs = default_specs("");
    specs.push(TensorSpec::new("", "pooler.dense.weight", &[H, H + 1]));
    specs.push(TensorSpec::new("", "pooler.dense.bias", &[H]));
    fixture.write(&specs);
    assert!(rejected(&fixture), "accepted wrong pooler shape");

    let mut specs = default_specs("");
    specs.push(TensorSpec::new("", "pooler.dense.weight", &[H, H]));
    fixture.write(&specs);
    assert!(rejected(&fixture), "accepted half pooler pair");
}

#[test]
fn rejects_missing_malformed_and_stray_tensors() {
    let fixture = Fixture::new();

    fixture.write(&without(
        default_specs(""),
        "embeddings.token_type_embeddings.weight",
    ));
    assert!(rejected(&fixture), "accepted missing type embeddings");

    fixture.write(&without(
        default_specs(""),
        "attention.output.LayerNorm.weight",
    ));
    assert!(rejected(&fixture), "accepted missing attention norm");

    let mut specs = default_specs("");
    specs.push(layer("", "mlp.gated_layers.bias", &[I * 2]));
    fixture.write(&specs);
    assert!(rejected(&fixture), "accepted gated projection bias");

    let mut specs = default_specs("");
    for spec in &mut specs {
        if spec.name.ends_with("attention.self.query.weight") {
            spec.shape = vec![H, H + 1];
        }
    }
    fixture.write(&specs);
    assert!(rejected(&fixture), "accepted malformed QKV weight");
}

#[test]
fn rejects_ambiguous_and_mixed_namespaces() {
    let fixture = Fixture::new();

    let mut specs = default_specs("");
    specs.push(TensorSpec::new(
        "bert.",
        "embeddings.word_embeddings.weight",
        &[V, H],
    ));
    fixture.write(&specs);
    assert!(rejected(&fixture), "accepted two word embedding roots");

    let mut specs = default_specs("");
    specs.push(TensorSpec::new(
        "bert.",
        "encoder.layer.0.mlp.wo.weight",
        &[H, I],
    ));
    fixture.write(&specs);
    assert!(rejected(&fixture), "accepted bare and bert-prefixed mix");

    let mut specs = default_specs("bert.");
    specs.push(TensorSpec::new(
        "",
        "encoder.layer.0.mlp.wo.weight",
        &[H, I],
    ));
    fixture.write(&specs);
    assert!(rejected(&fixture), "accepted bert-prefixed and bare mix");
}

#[test]
fn production_marker_routing_fails_closed_for_qk_post_norm_and_heads() {
    let fixture = Fixture::new();
    fixture.write(&default_specs(""));
    assert!(detects(&fixture.path).unwrap());
    fixture.config(json!({"auto_map":null,"_name_or_path":"ordinary-bert"}));
    assert!(!detects(&fixture.path).unwrap());
    fixture.config(json!({"_name_or_path":"jina-bert-v2-qk-post-norm"}));
    assert!(detects(&fixture.path).is_err());
    for architecture in [
        "JinaBertForSequenceClassification",
        "JinaBertForTokenClassification",
        "JinaBertUnknown",
    ] {
        fixture.config(json!({"architectures":[architecture]}));
        assert!(detects(&fixture.path).unwrap());
        assert!(JinaWeights::load(&fixture.path).is_err(), "{architecture}");
    }
}

#[test]
#[ignore = "CPU-only real checkpoint header check; set INFINITY_OXIDE_TEST_JINA_MODEL"]
fn actual_jina_checkpoint_metadata_cpu() {
    let path = PathBuf::from(
        std::env::var_os("INFINITY_OXIDE_TEST_JINA_MODEL").expect("actual Jina model path"),
    );
    let host = JinaWeights::load(&path).unwrap();
    let c = &host.geometry;
    assert_eq!(
        (
            c.hidden_size,
            c.num_attention_heads,
            c.num_hidden_layers,
            c.intermediate_size
        ),
        (512, 8, 4, 2048)
    );
    assert_eq!(
        (c.vocab_size, c.type_vocab_size, c.max_position_embeddings),
        (30528, 2, 8192)
    );
    assert_eq!(c.family, EncoderFamily::Jina);
    assert_eq!(
        c.recipe.positions,
        crate::model::PositionRecipe::AlibiModelBias
    );
    let checkpoint = Checkpoint::open(&path).unwrap();
    assert_eq!(checkpoint.names().count(), 66);
    for name in checkpoint.names() {
        let shape = checkpoint.shape(name).unwrap();
        assert_eq!(
            checkpoint.source(name, shape).unwrap().dtype(),
            Dtype::F16,
            "{name}"
        );
    }
    assert_eq!(host.layers.len(), 4);
    for layer in &host.layers {
        assert_eq!(layer.qkv.weight_parts.len(), 3);
        assert_eq!(layer.qkv.bias_parts.as_ref().unwrap().len(), 3);
        assert!(layer.intermediate.bias_parts.is_none());
        assert_eq!(layer.intermediate.weight_parts.len(), 2);
    }
    println!("actual Jina-small header:66F16 tensors,H512,L4,I2048,8heads,type_vocab2,positions8192; ignored config torch_dtype=float32 in favor of stored tensor dtype");
}
