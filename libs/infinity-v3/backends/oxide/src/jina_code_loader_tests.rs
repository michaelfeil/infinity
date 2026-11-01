use crate::jina_code_encoder_model::{load, PACKED_MLP_ORDER, PACKED_QKV_ORDER};
use crate::model::{EncoderFamily, FeedForwardRecipe, NormArithmetic, PositionRecipe};
use safetensors::tensor::{serialize_to_file, TensorView};
use safetensors::Dtype;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const HIDDEN: usize = 768;
const INTERMEDIATE: usize = 3072;
const VOCAB: usize = 61056;
const LAYERS: usize = 12;
const MAX_TENSOR_BYTES: usize = VOCAB * HIDDEN * 2;

type Specs = BTreeMap<String, (Vec<usize>, Dtype)>;

struct Fixture(PathBuf);

impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "jina-code-loader-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }

    fn config_only(config: &Value) -> Self {
        let fixture = Self::new();
        fixture.write_config(config);
        fixture
    }

    fn checkpoint(config: &Value, specs: &Specs, overrides: &BTreeMap<String, Vec<u8>>) -> Self {
        let fixture = Self::new();
        fixture.write_config(config);
        write_safetensors(&fixture.0.join("model.safetensors"), specs, overrides);
        fixture
    }

    fn write_config(&self, config: &Value) {
        fs::write(
            self.0.join("config.json"),
            serde_json::to_vec(config).unwrap(),
        )
        .unwrap();
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn base_config() -> Value {
    json!({
        "_name_or_path": "jinaai/jina-bert-v2-qk-post-norm",
        "architectures": ["JinaBertForMaskedLM"],
        "attention_probs_dropout_prob": 0.0,
        "attn_implementation": "torch",
        "auto_map": {
            "AutoConfig": "jinaai/jina-bert-v2-qk-post-norm--configuration_bert.JinaBertConfig",
            "AutoModel": "jinaai/jina-bert-v2-qk-post-norm--modeling_bert.JinaBertModel",
            "AutoModelForMaskedLM": "jinaai/jina-bert-v2-qk-post-norm--modeling_bert.JinaBertForMaskedLM"
        },
        "classifier_dropout": null,
        "emb_pooler": "mean",
        "feed_forward_type": "geglu",
        "gradient_checkpointing": false,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.0,
        "hidden_size": HIDDEN,
        "initializer_range": 0.02,
        "intermediate_size": INTERMEDIATE,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 8192,
        "model_max_length": 8192,
        "model_type": "bert",
        "num_attention_heads": 12,
        "num_hidden_layers": LAYERS,
        "pad_token_id": 0,
        "position_embedding_type": "alibi",
        "torch_dtype": "float16",
        "transformers_version": "4.35.2",
        "type_vocab_size": 2,
        "use_cache": true,
        "vocab_size": VOCAB
    })
}

fn add(specs: &mut Specs, name: String, shape: &[usize]) {
    specs.insert(name, (shape.to_vec(), Dtype::F16));
}

fn base_specs(root: &str, pooler: bool) -> Specs {
    let mut specs = Specs::new();
    add(
        &mut specs,
        format!("{root}embeddings.word_embeddings.weight"),
        &[VOCAB, HIDDEN],
    );
    add(
        &mut specs,
        format!("{root}embeddings.token_type_embeddings.weight"),
        &[2, HIDDEN],
    );
    add(
        &mut specs,
        format!("{root}embeddings.LayerNorm.weight"),
        &[HIDDEN],
    );
    add(
        &mut specs,
        format!("{root}embeddings.LayerNorm.bias"),
        &[HIDDEN],
    );
    for layer in 0..LAYERS {
        let p = format!("{root}encoder.layer.{layer}");
        for projection in ["query", "key", "value"] {
            add(
                &mut specs,
                format!("{p}.attention.self.{projection}.weight"),
                &[HIDDEN, HIDDEN],
            );
            add(
                &mut specs,
                format!("{p}.attention.self.{projection}.bias"),
                &[HIDDEN],
            );
        }
        for norm in ["layer_norm_q", "layer_norm_k"] {
            add(
                &mut specs,
                format!("{p}.attention.self.{norm}.weight"),
                &[HIDDEN],
            );
            add(
                &mut specs,
                format!("{p}.attention.self.{norm}.bias"),
                &[HIDDEN],
            );
        }
        add(
            &mut specs,
            format!("{p}.attention.output.dense.weight"),
            &[HIDDEN, HIDDEN],
        );
        add(
            &mut specs,
            format!("{p}.attention.output.dense.bias"),
            &[HIDDEN],
        );
        add(
            &mut specs,
            format!("{p}.attention.output.LayerNorm.weight"),
            &[HIDDEN],
        );
        add(
            &mut specs,
            format!("{p}.attention.output.LayerNorm.bias"),
            &[HIDDEN],
        );
        add(&mut specs, format!("{p}.layer_norm_1.weight"), &[HIDDEN]);
        add(&mut specs, format!("{p}.layer_norm_1.bias"), &[HIDDEN]);
        add(&mut specs, format!("{p}.layer_norm_2.weight"), &[HIDDEN]);
        add(&mut specs, format!("{p}.layer_norm_2.bias"), &[HIDDEN]);
        add(
            &mut specs,
            format!("{p}.mlp.up_gated_layer.weight"),
            &[2 * INTERMEDIATE, HIDDEN],
        );
        add(
            &mut specs,
            format!("{p}.mlp.down_layer.weight"),
            &[HIDDEN, INTERMEDIATE],
        );
        add(&mut specs, format!("{p}.mlp.down_layer.bias"), &[HIDDEN]);
    }
    if pooler {
        add(
            &mut specs,
            format!("{root}pooler.dense.weight"),
            &[HIDDEN, HIDDEN],
        );
        add(&mut specs, format!("{root}pooler.dense.bias"), &[HIDDEN]);
    }
    specs
}

fn dtype_width(dtype: Dtype) -> usize {
    match dtype {
        Dtype::F32 => 4,
        Dtype::F16 | Dtype::BF16 => 2,
        other => panic!("unsupported fixture dtype {other:?}"),
    }
}

fn tensor_bytes(shape: &[usize], dtype: Dtype) -> usize {
    shape.iter().product::<usize>() * dtype_width(dtype)
}

fn write_safetensors(path: &Path, specs: &Specs, overrides: &BTreeMap<String, Vec<u8>>) {
    let zeros = vec![0u8; MAX_TENSOR_BYTES];
    let mut views = Vec::with_capacity(specs.len());
    for (name, (shape, dtype)) in specs {
        let length = tensor_bytes(shape, *dtype);
        let data = overrides
            .get(name)
            .map(Vec::as_slice)
            .unwrap_or_else(|| &zeros[..length]);
        assert_eq!(data.len(), length, "fixture override length for {name}");
        views.push((
            name.clone(),
            TensorView::new(*dtype, shape.clone(), data).unwrap(),
        ));
    }
    serialize_to_file(views, None, path).unwrap();
}

fn load_error(path: &Path) -> String {
    match load(path) {
        Ok(_) => panic!("expected the fixture to be rejected"),
        Err(error) => error.to_string(),
    }
}

fn error_for(specs: Specs, config: Value) -> String {
    let fixture = Fixture::checkpoint(&config, &specs, &BTreeMap::new());
    load_error(fixture.path())
}

#[test]
fn loads_published_258_tensor_inventory_with_bare_root() {
    let specs = base_specs("", true);
    assert_eq!(specs.len(), 258);
    let fixture = Fixture::checkpoint(&base_config(), &specs, &BTreeMap::new());
    let weights = load(fixture.path()).unwrap();

    assert_eq!(weights.layers.len(), 12);
    assert_eq!(weights.config.hidden_size, 768);
    assert_eq!(weights.config.intermediate_size, 3072);
    assert_eq!(weights.config.num_attention_heads, 12);
    assert_eq!(weights.config.num_hidden_layers, 12);
    assert_eq!(weights.config.vocab_size, 61056);
    assert_eq!(weights.config.type_vocab_size, 2);
    assert_eq!(weights.config.max_position_embeddings, 8192);
    assert_eq!(weights.config.layer_norm_eps, 1e-12);
    assert_eq!(
        weights.config.recipe.embedding,
        NormArithmetic::ExactWordTypeBiased
    );
    assert_eq!(weights.config.recipe.residual, NormArithmetic::ExactBiased);
    assert_eq!(
        weights.config.recipe.positions,
        PositionRecipe::AlibiModelBias
    );
    assert_eq!(
        weights.config.recipe.feed_forward,
        FeedForwardRecipe::GatedGeluUpFirst
    );
    assert_eq!(weights.config.packed_qkv_order, PACKED_QKV_ORDER);
    assert_eq!(weights.config.packed_mlp_order, PACKED_MLP_ORDER);
    assert_eq!(weights.config.geometry().family, EncoderFamily::JinaCode);
    assert_eq!(weights.word.shape(), &[61056, 768]);
    assert_eq!(weights.word.dtype(), Dtype::F16);
    assert_eq!(weights.token_type.shape(), &[2, 768]);
    assert_eq!(weights.embedding_norm.gamma.shape(), &[768]);
    assert_eq!(weights.layers[0].query.weight.shape(), &[768, 768]);
    assert_eq!(weights.layers[0].key.bias.shape(), &[768]);
    assert_eq!(weights.layers[0].value.weight.shape(), &[768, 768]);
    assert_eq!(weights.layers[0].query_norm.gamma.shape(), &[768]);
    assert_eq!(weights.layers[0].key_norm.beta.shape(), &[768]);
    assert_eq!(weights.layers[0].attention.weight.shape(), &[768, 768]);
    assert_eq!(weights.layers[0].attention_norm.beta.shape(), &[768]);
    assert_eq!(weights.layers[0].layer_norm_1.gamma.shape(), &[768]);
    assert_eq!(weights.layers[0].layer_norm_2.beta.shape(), &[768]);
    assert_eq!(weights.layers[0].mlp_up_gated.shape(), &[6144, 768]);
    assert_eq!(weights.layers[0].mlp_down.weight.shape(), &[768, 3072]);
    assert!(weights._pooler.is_some());

    let mut converted = [1u16; 4];
    weights
        .embedding_norm
        .gamma
        .read_bf16(0, &mut converted)
        .unwrap();
    assert_eq!(converted, [0; 4]);
}

#[test]
fn loads_bert_root_without_pooler() {
    let specs = base_specs("bert.", false);
    assert_eq!(specs.len(), 256);
    let fixture = Fixture::checkpoint(&base_config(), &specs, &BTreeMap::new());
    let weights = load(fixture.path()).unwrap();
    assert_eq!(weights.layers.len(), 12);
    assert!(weights._pooler.is_none());
}

#[test]
fn accepts_auto_map_marker_without_name_or_path() {
    let mut config = base_config();
    config.as_object_mut().unwrap().remove("_name_or_path");
    let specs = base_specs("", false);
    let fixture = Fixture::checkpoint(&config, &specs, &BTreeMap::new());
    let weights = load(fixture.path()).unwrap();
    assert_eq!(weights.layers.len(), 12);
}

#[test]
fn rejects_mixed_bare_and_bert_namespace() {
    let mut specs = base_specs("", true);
    let entry = specs
        .remove("encoder.layer.0.attention.self.query.weight")
        .unwrap();
    specs.insert(
        "bert.encoder.layer.0.attention.self.query.weight".into(),
        entry,
    );
    let message = error_for(specs, base_config());
    assert!(message.contains("foreign root"), "{message}");
}

#[test]
fn rejects_wrong_dtype() {
    let mut specs = base_specs("", true);
    specs.insert(
        "encoder.layer.0.attention.self.layer_norm_q.weight".into(),
        (vec![HIDDEN], Dtype::F32),
    );
    let message = error_for(specs, base_config());
    assert!(message.contains("expected F16"), "{message}");
}

#[test]
fn rejects_non_full_width_qk_norm_shape() {
    let mut specs = base_specs("", true);
    specs.insert(
        "encoder.layer.0.attention.self.layer_norm_q.weight".into(),
        (vec![12, 64], Dtype::F16),
    );
    let message = error_for(specs, base_config());
    assert!(message.contains("wrong shape"), "{message}");
}

#[test]
fn rejects_absolute_position_table() {
    let mut specs = base_specs("", true);
    add(
        &mut specs,
        "embeddings.position_embeddings.weight".into(),
        &[8192, HIDDEN],
    );
    let message = error_for(specs, base_config());
    assert!(message.contains("absolute position"), "{message}");
}

#[test]
fn rejects_rotary_tensor() {
    let mut specs = base_specs("", true);
    add(
        &mut specs,
        "encoder.layer.0.attention.self.rotary_emb.inv_freq".into(),
        &[64],
    );
    let message = error_for(specs, base_config());
    assert!(message.contains("rotary tensor"), "{message}");
}

#[test]
fn rejects_packed_up_projection_bias() {
    let mut specs = base_specs("", true);
    add(
        &mut specs,
        "encoder.layer.0.mlp.up_gated_layer.bias".into(),
        &[2 * INTERMEDIATE],
    );
    let message = error_for(specs, base_config());
    assert!(message.contains("up-projection bias"), "{message}");
}

#[test]
fn rejects_classifier_tensors() {
    let mut specs = base_specs("", true);
    add(&mut specs, "classifier.weight".into(), &[2, HIDDEN]);
    let message = error_for(specs, base_config());
    assert!(message.contains("classification tensor"), "{message}");
}

#[test]
fn rejects_partial_pooler_pair() {
    let mut specs = base_specs("", true);
    specs.remove("pooler.dense.bias");
    let message = error_for(specs, base_config());
    assert!(
        message.contains("incomplete JinaCode pooler pair"),
        "{message}"
    );
}

#[test]
fn rejects_unknown_tensor_metadata() {
    let mut specs = base_specs("", true);
    add(&mut specs, "surprise.weight".into(), &[1]);
    let message = error_for(specs, base_config());
    assert!(
        message.contains("unknown JinaCode checkpoint tensor"),
        "{message}"
    );
}

#[test]
fn rejects_unknown_config_metadata() {
    let mut config = base_config();
    config["surprise"] = json!(true);
    let fixture = Fixture::config_only(&config);
    let message = load_error(fixture.path());
    assert!(
        message.contains("unknown JinaCode config metadata"),
        "{message}"
    );
}

#[test]
fn rejects_missing_native_marker() {
    let mut config = base_config();
    config.as_object_mut().unwrap().remove("_name_or_path");
    config.as_object_mut().unwrap().remove("auto_map");
    let fixture = Fixture::config_only(&config);
    let message = load_error(fixture.path());
    assert!(message.contains("name marker"), "{message}");
}

#[test]
fn rejects_rotary_config_metadata() {
    let mut config = base_config();
    config["rope_scaling"] = json!({"type": "linear", "factor": 2.0});
    let fixture = Fixture::config_only(&config);
    let message = load_error(fixture.path());
    assert!(message.contains("rotary positions"), "{message}");
}

#[test]
fn rejects_non_exact_config_geometry() {
    let mut config = base_config();
    config["layer_norm_eps"] = json!(1e-13);
    let fixture = Fixture::config_only(&config);
    let message = load_error(fixture.path());
    assert!(message.contains("layer_norm_eps 1e-12"), "{message}");
}

#[test]
fn rejects_nonfinite_f16_during_bf16_conversion() {
    let specs = base_specs("", true);
    let mut nan = vec![0u8; HIDDEN * 2];
    nan[..2].copy_from_slice(&0x7e00u16.to_le_bytes());
    let mut overrides = BTreeMap::new();
    overrides.insert("embeddings.LayerNorm.weight".into(), nan);
    let fixture = Fixture::checkpoint(&base_config(), &specs, &overrides);
    let weights = load(fixture.path()).unwrap();
    let mut out = [0u16; 4];
    let error = weights
        .embedding_norm
        .gamma
        .read_bf16(0, &mut out)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("invalid or nonfinite checkpoint weights"),
        "{error}"
    );
}
