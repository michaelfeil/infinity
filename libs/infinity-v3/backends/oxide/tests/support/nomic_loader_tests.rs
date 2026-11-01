use super::*;
use crate::model::{Checkpoint, EncoderFamily, PositionRecipe, RotaryRecipe};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

const H: usize = HIDDEN;
const I: usize = 8;
const V: usize = 4;
const T: usize = 2;

struct Fixture {
    path: PathBuf,
    layers: usize,
}

impl Fixture {
    fn new(layers: usize) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "infinity-nomic-loader-{}-{}-{}",
            module_path!().replace("::", "-"),
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        let fixture = Self { path, layers };
        fixture.config(json!({}));
        fixture
    }

    fn base(&self) -> Value {
        json!({
            "model_type": "nomic_bert",
            "architectures": ["NomicBertModel"],
            "n_embd": H, "n_head": 12, "n_layer": self.layers, "n_inner": I,
            "vocab_size": V, "type_vocab_size": T,
            "layer_norm_epsilon": 1e-5,
            "max_position_embeddings": 2048,
            "n_positions": 8192,
            "activation_function": "swiglu",
            "rotary_emb_base": 1000.0
        })
    }

    fn raw_config(&self, value: Value) {
        std::fs::write(
            self.path.join("config.json"),
            serde_json::to_vec(&value).unwrap(),
        )
        .unwrap();
    }

    fn config(&self, overrides: Value) {
        let mut value = self.base();
        for (key, val) in overrides.as_object().unwrap() {
            value[key.as_str()] = val.clone();
        }
        self.raw_config(value);
    }

    fn entries(&self, root: &str) -> Vec<(String, Vec<usize>)> {
        let mut entries = vec![
            (format!("{root}emb_ln.weight"), vec![H]),
            (format!("{root}emb_ln.bias"), vec![H]),
            (
                format!("{root}embeddings.word_embeddings.weight"),
                vec![V, H],
            ),
            (
                format!("{root}embeddings.token_type_embeddings.weight"),
                vec![T, H],
            ),
        ];
        for layer in 0..self.layers {
            let prefix = format!("{root}encoder.layers.{layer}");
            for (suffix, shape) in [
                ("attn.Wqkv.weight", vec![3 * H, H]),
                ("attn.out_proj.weight", vec![H, H]),
                ("mlp.fc12.weight", vec![I, H]),
                ("mlp.fc11.weight", vec![I, H]),
                ("mlp.fc2.weight", vec![H, I]),
                ("norm1.weight", vec![H]),
                ("norm1.bias", vec![H]),
                ("norm2.weight", vec![H]),
                ("norm2.bias", vec![H]),
            ] {
                entries.push((format!("{prefix}.{suffix}"), shape));
            }
        }
        assert_eq!(entries.len(), 4 + 9 * self.layers);
        entries
    }

    fn write(
        &self,
        file_name: &str,
        entries: &[(String, Vec<usize>)],
        probes: &[(&str, usize, f32)],
    ) {
        let mut sorted = entries.to_vec();
        sorted.sort_by(|left, right| left.0.cmp(&right.0));
        let mut header = serde_json::Map::new();
        let mut locations = BTreeMap::new();
        let mut payload_len = 0u64;
        for (name, shape) in &sorted {
            let count = shape
                .iter()
                .try_fold(1usize, |a, b| a.checked_mul(*b))
                .unwrap();
            let bytes = u64::try_from(count.checked_mul(4).unwrap()).unwrap();
            let begin = payload_len;
            payload_len = payload_len.checked_add(bytes).unwrap();
            header.insert(
                name.clone(),
                json!({"dtype":"F32","shape":shape,"data_offsets":[begin,payload_len]}),
            );
            locations.insert(name.clone(), begin);
        }
        let header_bytes = serde_json::to_vec(&Value::Object(header)).unwrap();
        let data_start = 8 + u64::try_from(header_bytes.len()).unwrap();
        let mut file = File::create(self.path.join(file_name)).unwrap();
        file.set_len(data_start + payload_len).unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();
        file.write_all(&u64::try_from(header_bytes.len()).unwrap().to_le_bytes())
            .unwrap();
        file.write_all(&header_bytes).unwrap();
        for &(name, index, value) in probes {
            let begin = *locations.get(name).unwrap();
            let offset = data_start + begin + u64::try_from(index.checked_mul(4).unwrap()).unwrap();
            file.seek(SeekFrom::Start(offset)).unwrap();
            file.write_all(&value.to_le_bytes()).unwrap();
        }
        file.flush().unwrap();
    }

    fn weights(&self, root: &str) {
        self.write(
            "model.safetensors",
            &self.entries(root),
            &[
                (&format!("{root}encoder.layers.0.mlp.fc12.weight"), 0, 3.25),
                (&format!("{root}encoder.layers.0.mlp.fc11.weight"), 0, -2.75),
            ],
        );
    }

    fn config_error(&self) -> String {
        match RawNomicConfig::load(&self.path) {
            Ok(_) => panic!("invalid nomic config was accepted"),
            Err(error) => error.to_string(),
        }
    }

    fn error(&self) -> String {
        match NomicWeights::load(&self.path) {
            Ok(_) => panic!("invalid nomic fixture was accepted"),
            Err(error) => error.to_string(),
        }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn read(source: &WeightSource) -> u16 {
    let mut bits = [0u16; 1];
    source.read_bf16(0, &mut bits).unwrap();
    bits[0]
}

#[test]
fn loads_sparse_native_bare_root_with_gate_before_up() {
    {
        let root = "";
        let fixture = Fixture::new(1);
        fixture.weights(root);
        let weights = NomicWeights::load(&fixture.path).unwrap();
        assert_eq!(weights.config.family, EncoderFamily::Nomic);
        assert_eq!(weights.config.num_hidden_layers, 1);
        assert_eq!(weights.layers.len(), 1);
        let layer = &weights.layers[0];
        assert_eq!(layer.qkv.parts.len(), 1);
        assert_eq!(layer.attention.parts.len(), 1);
        assert_eq!(layer.output.parts.len(), 1);
        assert_eq!(layer.intermediate.parts.len(), 2);
        assert_eq!(read(&layer.intermediate.parts[0]), 0x4050);
        assert_eq!(read(&layer.intermediate.parts[1]), 0xc030);
    }
}

#[test]
fn native_defaults_and_values_ignore_legacy_aliases() {
    let fixture = Fixture::new(1);
    fixture.raw_config(json!({
        "model_type": "nomic-bert",
        "n_embd": 768, "n_head": 12, "n_layer": 1, "n_inner": 8,
        "vocab_size": V, "type_vocab_size": T,
        "max_position_embeddings": 2048, "activation_function": "swiglu",
        "hidden_size": 384, "num_attention_heads": 6,
        "num_hidden_layers": 9, "intermediate_size": 16,
        "layer_norm_eps": 0.5, "rotary_emb_base": null,
        "rope_parameters": {"rope_type": "default", "rope_theta": 9999.0}
    }));
    let config = RawNomicConfig::load(&fixture.path).unwrap();
    assert_eq!(config.hidden_size, H);
    assert_eq!(config.num_attention_heads, 12);
    assert_eq!(config.num_hidden_layers, 1);
    assert_eq!(config.intermediate_size, I);
    assert_eq!(config.layer_norm_eps.to_bits(), DEFAULT_EPS.to_bits());
    assert!(matches!(
        config.recipe.positions,
        PositionRecipe::Rotary(RotaryRecipe { theta, ntk_factor: None })
            if theta == DEFAULT_THETA
    ));

    fixture.raw_config(json!({
        "model_type": "nomic_bert", "vocab_size": V,
        "max_position_embeddings": 2048, "activation_function": "swiglu"
    }));
    let config = RawNomicConfig::load(&fixture.path).unwrap();
    assert_eq!(config.hidden_size, 768);
    assert_eq!(config.num_attention_heads, 12);
    assert_eq!(config.num_hidden_layers, 12);
    assert_eq!(config.intermediate_size, 3072);
    fixture.raw_config(json!({
        "model_type": "nomic_bert", "vocab_size": V,
        "n_positions": 4096, "activation_function": "swiglu"
    }));
    let config = RawNomicConfig::load(&fixture.path).unwrap();
    assert_eq!(config.max_position_embeddings, 4096);
    assert_eq!(config.rotary_table_positions, Some(4096));
}

#[test]
fn backend_uses_native_positions_instead_of_frontend_tokenizer_cap() {
    let fixture = Fixture::new(1);
    fixture.config(json!({"n_positions": 8192, "max_position_embeddings": 2048}));
    assert_eq!(
        RawNomicConfig::load(&fixture.path)
            .unwrap()
            .max_position_embeddings,
        8192
    );
    fixture.config(json!({"n_positions": 4096, "max_position_embeddings": 2048}));
    assert_eq!(
        RawNomicConfig::load(&fixture.path)
            .unwrap()
            .max_position_embeddings,
        4096
    );
}

#[test]
fn rejects_invalid_geometry_and_frontend_table_relation() {
    for (overrides, expected) in [
        (json!({"n_embd": 384}), "hidden size"),
        (json!({"n_head": 6}), "hidden size"),
        (json!({"n_inner": 7}), "intermediate"),
        (json!({"n_layer": 0}), "num_hidden_layers"),
        (json!({"vocab_size": 0}), "vocabulary"),
        (json!({"type_vocab_size": 0}), "vocabulary"),
        (json!({"n_positions": 8193}), "frontend positions"),
        (
            json!({"n_positions": 1024, "max_position_embeddings": 2048}),
            "frontend positions",
        ),
        (json!({"max_position_embeddings": 0}), "frontend positions"),
        (json!({"layer_norm_epsilon": 0.0}), "layer_norm_epsilon"),
        (json!({"rotary_emb_base": 1.0}), "rotary_emb_base"),
    ] {
        let fixture = Fixture::new(1);
        fixture.config(overrides);
        let error = fixture.config_error();
        assert!(error.contains(expected), "{error}");
    }
}

#[test]
fn requires_activation_and_rejects_moe_and_rotary_modifiers() {
    for (overrides, expected) in [
        (json!({"use_moe": true}), "mixture"),
        (json!({"moe_every_n_layers": 1}), "mixture"),
        (json!({"num_experts": 2}), "mixture"),
        (json!({"quantization_config": {"bits": 4}}), "quantized"),
        (json!({"rotary_emb_fraction": 0.5}), "rotary_emb_fraction"),
        (json!({"rotary_emb_interleaved": true}), "interleaved"),
        (json!({"rotary_emb_scale_base": 2.0}), "dynamic rotary"),
        (
            json!({"rotary_scaling_factor": {"type": "linear"}}),
            "dynamic rotary",
        ),
        (
            json!({"rope_scaling": {"type": "linear"}}),
            "dynamic rotary",
        ),
        (
            json!({"rope_parameters": {"rope_type": "dynamic"}}),
            "rope_parameters",
        ),
        (json!({"use_rms_norm": true}), "RMS"),
        (json!({"parallel_block": true}), "parallel"),
    ] {
        let fixture = Fixture::new(1);
        fixture.config(overrides);
        let error = fixture.config_error();
        assert!(error.contains(expected), "{error}");
    }
    let fixture = Fixture::new(1);
    let mut value = fixture.base();
    value.as_object_mut().unwrap().remove("activation_function");
    fixture.raw_config(value);
    assert!(fixture.config_error().contains("activation_function"));
}

#[test]
fn rejects_missing_tables_norms_wrong_qkv_and_projection_biases() {
    let fixture = Fixture::new(1);
    let mut entries = fixture.entries("");
    entries.retain(|(name, _)| !name.ends_with("token_type_embeddings.weight"));
    fixture.write("model.safetensors", &entries, &[]);
    assert!(fixture.error().contains("missing weight"));

    let fixture = Fixture::new(1);
    let mut entries = fixture.entries("");
    entries.retain(|(name, _)| !name.ends_with("norm1.bias"));
    fixture.write("model.safetensors", &entries, &[]);
    assert!(fixture.error().contains("missing weight"));

    let fixture = Fixture::new(1);
    let mut entries = fixture.entries("");
    for (name, shape) in &mut entries {
        if name.ends_with("attn.Wqkv.weight") {
            *shape = vec![3 * H, H - 1];
        }
    }
    fixture.write("model.safetensors", &entries, &[]);
    assert!(fixture.error().contains("wrong shape"));

    let fixture = Fixture::new(1);
    let mut entries = fixture.entries("");
    entries.push(("encoder.layers.0.attn.out_proj.bias".into(), vec![H]));
    fixture.write("model.safetensors", &entries, &[]);
    assert!(fixture.error().contains("bias tensor"));
}

#[test]
fn rejects_ambiguous_and_mixed_checkpoint_namespaces() {
    let fixture = Fixture::new(1);
    let mut entries = fixture.entries("");
    entries.push(("bert.embeddings.word_embeddings.weight".into(), vec![V, H]));
    fixture.write("model.safetensors", &entries, &[]);
    assert!(fixture.error().contains("ambiguous checkpoint namespace"));

    let fixture = Fixture::new(1);
    let mut entries = fixture.entries("");
    entries.push((
        "model.encoder.layers.0.attn.out_proj.weight".into(),
        vec![H, H],
    ));
    fixture.write("model.safetensors", &entries, &[]);
    assert!(fixture
        .error()
        .contains("incoherent nomic checkpoint namespace"));

    let fixture = Fixture::new(1);
    let mut entries = fixture.entries("bert.");
    entries.push(("encoder.layers.0.attn.out_proj.weight".into(), vec![H, H]));
    fixture.write("model.safetensors", &entries, &[]);
    assert!(fixture.error().contains("unsupported checkpoint namespace"));
}

#[test]
fn checkpoint_guards_missing_sources_and_shard_ownership() {
    let fixture = Fixture::new(1);
    let entries = fixture.entries("");
    fixture.write("shard-a.safetensors", &entries, &[]);
    fixture.write(
        "shard-b.safetensors",
        &[("dummy.weight".into(), vec![1])],
        &[],
    );
    let mut weight_map: BTreeMap<String, String> = BTreeMap::new();
    for (name, _) in &entries {
        weight_map.insert(name.clone(), "shard-a.safetensors".into());
    }
    let word = "embeddings.word_embeddings.weight".to_string();
    weight_map.insert(word.clone(), "shard-b.safetensors".into());
    weight_map.insert("dummy.weight".into(), "shard-b.safetensors".into());
    std::fs::write(
        fixture.path.join("model.safetensors.index.json"),
        serde_json::to_vec(&json!({"weight_map": weight_map})).unwrap(),
    )
    .unwrap();

    let checkpoint = Checkpoint::open(&fixture.path).unwrap();
    let error = checkpoint.source(&word, &[V, H]).unwrap_err().to_string();
    assert!(error.contains("not in its indexed shard"), "{error}");
    assert!(checkpoint
        .source("missing.weight", &[1])
        .unwrap_err()
        .to_string()
        .contains("missing weight"));
}
