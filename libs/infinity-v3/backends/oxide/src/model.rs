// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.

use infinity_v3_backend_core::BackendError;
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Component, Path};

fn start(error: impl std::fmt::Display) -> BackendError {
    BackendError::Start(format!("Plain CUDA BERT: {error}"))
}
fn epsilon() -> f64 {
    1e-12
}
fn type_vocab() -> usize {
    2
}
fn gelu() -> String {
    "gelu".into()
}
fn absolute() -> String {
    "absolute".into()
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct BertConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
    #[serde(default = "type_vocab")]
    pub type_vocab_size: usize,
    #[serde(default = "epsilon")]
    pub layer_norm_eps: f64,
    model_type: String,
    #[serde(default = "gelu")]
    hidden_act: String,
    #[serde(default = "absolute")]
    position_embedding_type: String,
    #[serde(default)]
    is_decoder: bool,
    #[serde(default)]
    add_cross_attention: bool,
}
impl BertConfig {
    fn validate(&self) -> Result<(), BackendError> {
        if self.model_type != "bert"
            || self.is_decoder
            || self.add_cross_attention
            || self.position_embedding_type != "absolute"
            || self.hidden_act != "gelu"
        {
            return Err(start(
                "only encoder BERT with absolute positions and exact GELU is supported",
            ));
        }
        if self.hidden_size == 0
            || self.intermediate_size == 0
            || self.num_attention_heads == 0
            || self.num_hidden_layers == 0
            || self.vocab_size == 0
            || self.type_vocab_size == 0
            || self.max_position_embeddings == 0
            || self.max_position_embeddings > 4096
            || self.num_hidden_layers > 256
            || self.num_attention_heads > 65535
            || !self.hidden_size.is_multiple_of(self.num_attention_heads)
            || self.hidden_size / self.num_attention_heads > 256
            || !self.layer_norm_eps.is_finite()
            || self.layer_norm_eps <= 0.0
            || !(self.layer_norm_eps as f32).is_finite()
            || self.layer_norm_eps as f32 <= 0.0
        {
            return Err(start(
                "invalid or unsupported BERT dimensions / layer_norm_eps",
            ));
        }
        for dimensions in [
            [self.vocab_size, self.hidden_size],
            [self.max_position_embeddings, self.hidden_size],
            [self.type_vocab_size, self.hidden_size],
            [self.hidden_size, self.hidden_size],
            [self.intermediate_size, self.hidden_size],
        ] {
            elements(&dimensions)?;
        }
        Ok(())
    }
}

pub(crate) struct LinearWeights {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}
pub(crate) struct LayerWeights {
    pub query: LinearWeights,
    pub key: LinearWeights,
    pub value: LinearWeights,
    pub attention: LinearWeights,
    pub intermediate: LinearWeights,
    pub output: LinearWeights,
    pub attention_gamma: Vec<f32>,
    pub attention_beta: Vec<f32>,
    pub output_gamma: Vec<f32>,
    pub output_beta: Vec<f32>,
}
pub(crate) struct BertWeights {
    pub config: BertConfig,
    pub word: Vec<f32>,
    pub position: Vec<f32>,
    pub token_type: Vec<f32>,
    pub gamma: Vec<f32>,
    pub beta: Vec<f32>,
    pub layers: Vec<LayerWeights>,
}

fn elements(shape: &[usize]) -> Result<usize, BackendError> {
    let n = shape
        .iter()
        .try_fold(1usize, |n, d| n.checked_mul(*d))
        .ok_or_else(|| start("weight dimensions overflow"))?;
    if n > i32::MAX as usize {
        return Err(start("weight exceeds CUDA indexing range"));
    }
    n.checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| start("weight byte size overflow"))?;
    Ok(n)
}
fn validate_modules(path: &Path) -> Result<(), BackendError> {
    let file = path.join("modules.json");
    if !file.exists() {
        return Ok(());
    }
    #[derive(Deserialize)]
    struct Module {
        #[serde(rename = "type")]
        kind: String,
    }
    let modules: Vec<Module> =
        serde_json::from_slice(&std::fs::read(file).map_err(start)?).map_err(start)?;
    for module in &modules {
        if !matches!(
            module.kind.as_str(),
            "sentence_transformers.models.Transformer"
                | "sentence_transformers.models.Pooling"
                | "sentence_transformers.models.Normalize"
        ) {
            return Err(start(format!(
                "unsupported sentence-transformers module {}; projections must not be ignored",
                module.kind
            )));
        }
    }
    // Normalize is recognized here; the HTTP request owns output normalization.
    let kinds = modules
        .iter()
        .map(|module| module.kind.as_str())
        .collect::<Vec<_>>();
    match kinds.as_slice() {
        ["sentence_transformers.models.Transformer", "sentence_transformers.models.Pooling"] => Ok(()),
        ["sentence_transformers.models.Transformer", "sentence_transformers.models.Pooling", "sentence_transformers.models.Normalize"] => Ok(()),
        _ => Err(start("unsupported sentence-transformers pipeline order: expected Transformer, Pooling and optional final Normalize")),
    }
}
type CheckpointFiles = (Vec<String>, Option<BTreeMap<String, String>>);

fn checkpoint_files(path: &Path) -> Result<CheckpointFiles, BackendError> {
    if path.join("model.safetensors").is_file() {
        return Ok((vec!["model.safetensors".into()], None));
    }
    #[derive(Deserialize)]
    struct Index {
        weight_map: BTreeMap<String, String>,
    }
    let index: Index = serde_json::from_slice(
        &std::fs::read(path.join("model.safetensors.index.json")).map_err(start)?,
    )
    .map_err(start)?;
    if index.weight_map.is_empty() {
        return Err(start("empty safetensors shard index"));
    }
    let mut files = BTreeSet::new();
    for file in index.weight_map.values() {
        let mut components = Path::new(file).components();
        if !matches!(components.next(), Some(Component::Normal(_)))
            || components.next().is_some()
            || !file.ends_with(".safetensors")
            || file.contains('\\')
        {
            return Err(start("invalid safetensors shard path"));
        }
        files.insert(file.clone());
    }
    Ok((files.into_iter().collect(), Some(index.weight_map)))
}

/// Hub snapshots use ../../blobs/<digest> symlinks within one cached repo.
/// All other checkpoints remain confined to their canonical model directory.
fn permitted_weight(root: &Path, resolved: &Path) -> bool {
    if resolved.starts_with(root) {
        return true;
    }
    let Some(digest) = resolved.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    if !matches!(digest.len(), 40 | 64) || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return false;
    }
    let Some(snapshots) = root.parent() else {
        return false;
    };
    if snapshots.file_name().and_then(|name| name.to_str()) != Some("snapshots") {
        return false;
    }
    let Some(repo) = snapshots.parent() else {
        return false;
    };
    if !repo
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with("models--") && name.len() > "models--".len())
    {
        return false;
    }
    // Compare the canonical target against this exact flat directory. A blobs
    // directory or blob symlink pointing elsewhere is not a permitted escape.
    resolved.parent() == Some(repo.join("blobs").as_path())
}
fn required(config: &BertConfig) -> HashMap<String, Vec<usize>> {
    let h = config.hidden_size;
    let mut out = HashMap::new();
    for (name, shape) in [
        (
            "embeddings.word_embeddings.weight",
            vec![config.vocab_size, h],
        ),
        (
            "embeddings.position_embeddings.weight",
            vec![config.max_position_embeddings, h],
        ),
        (
            "embeddings.token_type_embeddings.weight",
            vec![config.type_vocab_size, h],
        ),
        ("embeddings.LayerNorm.weight", vec![h]),
        ("embeddings.LayerNorm.bias", vec![h]),
    ] {
        out.insert(name.into(), shape);
    }
    for i in 0..config.num_hidden_layers {
        for (name, rows, columns) in [
            ("attention.self.query", h, h),
            ("attention.self.key", h, h),
            ("attention.self.value", h, h),
            ("attention.output.dense", h, h),
            ("intermediate.dense", config.intermediate_size, h),
            ("output.dense", h, config.intermediate_size),
        ] {
            out.insert(
                format!("encoder.layer.{i}.{name}.weight"),
                vec![rows, columns],
            );
            out.insert(format!("encoder.layer.{i}.{name}.bias"), vec![rows]);
        }
        for name in ["attention.output.LayerNorm", "output.LayerNorm"] {
            for suffix in ["weight", "bias"] {
                out.insert(format!("encoder.layer.{i}.{name}.{suffix}"), vec![h]);
            }
        }
    }
    out
}
fn decode(tensor: &safetensors::tensor::TensorView<'_>) -> Result<Vec<f32>, BackendError> {
    let count = elements(tensor.shape())?;
    let mut values = Vec::new();
    values.try_reserve_exact(count).map_err(start)?;
    match tensor.dtype() {
        Dtype::F32 => {
            for bytes in tensor.data().chunks_exact(4) {
                values.push(f32::from_le_bytes(bytes.try_into().map_err(start)?));
            }
        }
        Dtype::F16 | Dtype::BF16 => {
            for bytes in tensor.data().chunks_exact(2) {
                let bits = u16::from_le_bytes(bytes.try_into().map_err(start)?);
                values.push(if tensor.dtype() == Dtype::F16 {
                    half::f16::from_bits(bits).to_f32()
                } else {
                    half::bf16::from_bits(bits).to_f32()
                });
            }
        }
        other => return Err(start(format!("unsupported weight dtype {other:?}"))),
    }
    if values.len() != count || values.iter().any(|x| !x.is_finite()) {
        return Err(start("invalid or nonfinite checkpoint weights"));
    }
    Ok(values)
}
fn take(weights: &mut HashMap<String, Vec<f32>>, name: &str) -> Result<Vec<f32>, BackendError> {
    weights
        .remove(name)
        .ok_or_else(|| start(format!("missing weight {name}")))
}
fn linear(
    weights: &mut HashMap<String, Vec<f32>>,
    prefix: &str,
) -> Result<LinearWeights, BackendError> {
    Ok(LinearWeights {
        weight: take(weights, &format!("{prefix}.weight"))?,
        bias: take(weights, &format!("{prefix}.bias"))?,
    })
}
impl BertWeights {
    pub(crate) fn load(path: &Path) -> Result<Self, BackendError> {
        let config: BertConfig =
            serde_json::from_slice(&std::fs::read(path.join("config.json")).map_err(start)?)
                .map_err(start)?;
        config.validate()?;
        validate_modules(path)?;
        let expected = required(&config);
        let (files, index) = checkpoint_files(path)?;
        let root = path.canonicalize().map_err(start)?;
        let mut weights = HashMap::new();
        for file in files {
            let resolved = path.join(&file).canonicalize().map_err(start)?;
            if !permitted_weight(&root, &resolved) {
                return Err(start("checkpoint shard escapes model directory"));
            }
            let bytes = std::fs::read(resolved).map_err(start)?;
            let tensors = SafeTensors::deserialize(&bytes).map_err(start)?;
            for (name, tensor) in tensors.tensors() {
                let key = name.strip_prefix("bert.").unwrap_or(&name);
                let Some(shape) = expected.get(key) else {
                    continue;
                };
                if index
                    .as_ref()
                    .is_some_and(|index| index.get(&name) != Some(&file))
                {
                    return Err(start(format!("weight {name} is not in its indexed shard")));
                }
                if tensor.shape() != shape {
                    return Err(start(format!(
                        "wrong shape for {name}: expected {shape:?}, got {:?}",
                        tensor.shape()
                    )));
                }
                if weights.insert(key.to_owned(), decode(&tensor)?).is_some() {
                    return Err(start(format!("duplicate weight {key}")));
                }
            }
        }
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(config.num_hidden_layers)
            .map_err(start)?;
        for i in 0..config.num_hidden_layers {
            let p = format!("encoder.layer.{i}");
            layers.push(LayerWeights {
                query: linear(&mut weights, &format!("{p}.attention.self.query"))?,
                key: linear(&mut weights, &format!("{p}.attention.self.key"))?,
                value: linear(&mut weights, &format!("{p}.attention.self.value"))?,
                attention: linear(&mut weights, &format!("{p}.attention.output.dense"))?,
                intermediate: linear(&mut weights, &format!("{p}.intermediate.dense"))?,
                output: linear(&mut weights, &format!("{p}.output.dense"))?,
                attention_gamma: take(
                    &mut weights,
                    &format!("{p}.attention.output.LayerNorm.weight"),
                )?,
                attention_beta: take(
                    &mut weights,
                    &format!("{p}.attention.output.LayerNorm.bias"),
                )?,
                output_gamma: take(&mut weights, &format!("{p}.output.LayerNorm.weight"))?,
                output_beta: take(&mut weights, &format!("{p}.output.LayerNorm.bias"))?,
            });
        }
        Ok(Self {
            word: take(&mut weights, "embeddings.word_embeddings.weight")?,
            position: take(&mut weights, "embeddings.position_embeddings.weight")?,
            token_type: take(&mut weights, "embeddings.token_type_embeddings.weight")?,
            gamma: take(&mut weights, "embeddings.LayerNorm.weight")?,
            beta: take(&mut weights, "embeddings.LayerNorm.bias")?,
            config,
            layers,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{serialize, TensorView};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "infinity-cuda-weights-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).unwrap();
            let fixture = Self(path);
            fixture.config(serde_json::json!({}));
            fixture
        }
        fn config(&self, overrides: serde_json::Value) {
            let mut value = serde_json::json!({"model_type":"bert", "hidden_size":2,
                "intermediate_size":4, "num_attention_heads":1, "num_hidden_layers":1,
                "max_position_embeddings":8, "vocab_size":4});
            for (key, val) in overrides.as_object().unwrap() {
                value[key] = val.clone();
            }
            std::fs::write(
                self.0.join("config.json"),
                serde_json::to_vec(&value).unwrap(),
            )
            .unwrap();
        }
        fn weights(&self, dtype: Dtype, prefixed: bool, sharded: bool) {
            let config: BertConfig =
                serde_json::from_slice(&std::fs::read(self.0.join("config.json")).unwrap())
                    .unwrap();
            let mut entries = required(&config).into_iter().collect::<Vec<_>>();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            let data = entries
                .iter()
                .map(|(name, shape)| {
                    let value = if name.contains("LayerNorm.weight") {
                        1.0f32
                    } else {
                        0.25f32
                    };
                    let scalar = match dtype {
                        Dtype::F32 => value.to_le_bytes().to_vec(),
                        Dtype::F16 => half::f16::from_f32(value).to_bits().to_le_bytes().to_vec(),
                        Dtype::BF16 => half::bf16::from_f32(value).to_bits().to_le_bytes().to_vec(),
                        _ => unreachable!(),
                    };
                    scalar.repeat(shape.iter().product())
                })
                .collect::<Vec<_>>();
            let views = entries
                .iter()
                .zip(&data)
                .map(|((name, shape), bytes)| {
                    (
                        format!("{}{name}", if prefixed { "bert." } else { "" }),
                        TensorView::new(dtype, shape.clone(), bytes).unwrap(),
                    )
                })
                .collect::<Vec<_>>();
            if !sharded {
                std::fs::write(
                    self.0.join("model.safetensors"),
                    serialize(views.iter().map(|(n, v)| (n.clone(), v.clone())), None).unwrap(),
                )
                .unwrap();
            } else {
                let mut map = BTreeMap::new();
                for (i, chunk) in views.chunks(views.len().div_ceil(2)).enumerate() {
                    let file = format!("model-{i}.safetensors");
                    std::fs::write(
                        self.0.join(&file),
                        serialize(chunk.iter().map(|(n, v)| (n.clone(), v.clone())), None).unwrap(),
                    )
                    .unwrap();
                    for (name, _) in chunk {
                        map.insert(name.clone(), file.clone());
                    }
                }
                std::fs::write(
                    self.0.join("model.safetensors.index.json"),
                    serde_json::to_vec(&serde_json::json!({"weight_map":map})).unwrap(),
                )
                .unwrap();
            }
        }
        fn error(&self) -> String {
            match BertWeights::load(&self.0) {
                Ok(_) => panic!("invalid fixture was accepted"),
                Err(e) => e.to_string(),
            }
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn loads_float_precisions_and_bert_prefix() {
        for dtype in [Dtype::F32, Dtype::F16, Dtype::BF16] {
            let fixture = Fixture::new();
            fixture.weights(dtype, true, false);
            let weights = BertWeights::load(&fixture.0).unwrap();
            assert_eq!(weights.layers.len(), 1);
            assert_eq!(weights.word, vec![0.25; 8]);
            assert_eq!(weights.gamma, vec![1.0; 2]);
            assert_eq!(weights.config.type_vocab_size, 2);
            assert_eq!(weights.config.layer_norm_eps, 1e-12);
        }
    }
    #[test]
    fn loads_standard_sharded_bare_checkpoint() {
        let fixture = Fixture::new();
        fixture.weights(Dtype::F32, false, true);
        let weights = BertWeights::load(&fixture.0).unwrap();
        assert_eq!(weights.layers[0].intermediate.weight, vec![0.25; 8]);
        assert_eq!(weights.layers[0].output.bias, vec![0.25; 2]);
    }
    #[test]
    fn rejects_incompatible_model_before_checkpoint_or_gpu() {
        for overrides in [
            serde_json::json!({"model_type":"roberta"}),
            serde_json::json!({"is_decoder":true}),
            serde_json::json!({"add_cross_attention":true}),
            serde_json::json!({"hidden_act":"gelu_new"}),
            serde_json::json!({"position_embedding_type":"relative_key"}),
            serde_json::json!({"num_attention_heads":0}),
            serde_json::json!({"hidden_size":3,"num_attention_heads":2}),
            serde_json::json!({"max_position_embeddings":8192}),
        ] {
            let fixture = Fixture::new();
            fixture.config(overrides);
            let error = fixture.error();
            assert!(
                error.contains("only encoder BERT") || error.contains("invalid or unsupported"),
                "{error}"
            );
        }
    }
    #[test]
    fn refuses_trained_projection_in_sentence_transformers_pipeline() {
        let fixture = Fixture::new();
        std::fs::write(fixture.0.join("modules.json"),br#"[{"type":"sentence_transformers.models.Transformer"},{"type":"sentence_transformers.models.Dense"}]"#).unwrap();
        assert!(fixture.error().contains("projections must not be ignored"));
    }
    #[test]
    fn accepts_only_supported_sentence_transformers_pipeline_order() {
        let fixture = Fixture::new();
        fixture.weights(Dtype::F32, false, false);
        for kinds in [
            vec!["Transformer", "Pooling"],
            vec!["Transformer", "Pooling", "Normalize"],
        ] {
            let modules = kinds
                .into_iter()
                .map(|kind| {
                    serde_json::json!({
                        "type": format!("sentence_transformers.models.{kind}")
                    })
                })
                .collect::<Vec<_>>();
            std::fs::write(
                fixture.0.join("modules.json"),
                serde_json::to_vec(&modules).unwrap(),
            )
            .unwrap();
            assert!(BertWeights::load(&fixture.0).is_ok());
        }
        for kinds in [
            vec!["Transformer", "Normalize", "Pooling"],
            vec!["Transformer", "Pooling", "Normalize", "Normalize"],
            vec!["Transformer"],
        ] {
            let modules = kinds
                .into_iter()
                .map(|kind| {
                    serde_json::json!({
                        "type": format!("sentence_transformers.models.{kind}")
                    })
                })
                .collect::<Vec<_>>();
            std::fs::write(
                fixture.0.join("modules.json"),
                serde_json::to_vec(&modules).unwrap(),
            )
            .unwrap();
            assert!(fixture.error().contains("pipeline order"));
        }
    }
    #[test]
    fn rejects_shard_traversal_and_absolute_paths() {
        for file in [
            "../outside.safetensors",
            "/tmp/outside.safetensors",
            "sub/model.safetensors",
            "..\\outside.safetensors",
        ] {
            let fixture = Fixture::new();
            std::fs::write(
                fixture.0.join("model.safetensors.index.json"),
                serde_json::to_vec(
                    &serde_json::json!({"weight_map":{"embeddings.word_embeddings.weight":file}}),
                )
                .unwrap(),
            )
            .unwrap();
            assert!(fixture.error().contains("invalid safetensors shard path"));
        }
    }
    #[cfg(unix)]
    #[test]
    fn loads_hub_snapshot_weight_symlinks_for_single_and_sharded_models() {
        for sharded in [false, true] {
            let fixture = Fixture::new();
            fixture.weights(Dtype::F32, true, sharded);
            let repo = fixture.0.join("models--example--tiny-bert");
            let snapshot = repo.join("snapshots").join("a".repeat(40));
            let blobs = repo.join("blobs");
            std::fs::create_dir_all(&snapshot).unwrap();
            std::fs::create_dir_all(&blobs).unwrap();
            std::fs::copy(fixture.0.join("config.json"), snapshot.join("config.json")).unwrap();
            if sharded {
                std::fs::copy(
                    fixture.0.join("model.safetensors.index.json"),
                    snapshot.join("model.safetensors.index.json"),
                )
                .unwrap();
            }
            let weights = std::fs::read_dir(&fixture.0)
                .unwrap()
                .map(|entry| entry.unwrap().path())
                .filter(|path| path.extension().is_some_and(|ext| ext == "safetensors"))
                .collect::<Vec<_>>();
            for (index, file) in weights.iter().enumerate() {
                let digest = if index == 0 {
                    format!("{index:040x}")
                } else {
                    format!("{index:064x}")
                };
                std::fs::rename(file, blobs.join(&digest)).unwrap();
                std::os::unix::fs::symlink(
                    Path::new("../../blobs").join(&digest),
                    snapshot.join(file.file_name().unwrap()),
                )
                .unwrap();
            }
            let model = BertWeights::load(&snapshot).unwrap();
            assert_eq!(model.word, vec![0.25; 8]);
            assert_eq!(model.layers.len(), 1);
        }
    }
    #[cfg(unix)]
    #[test]
    fn rejects_unrelated_weight_symlinks_even_from_a_hub_snapshot() {
        for hub in [false, true] {
            let fixture = Fixture::new();
            let external = Fixture::new();
            external.weights(Dtype::F32, false, false);
            let external_blob = external.0.join("b".repeat(64));
            std::fs::rename(external.0.join("model.safetensors"), &external_blob).unwrap();
            let root = if hub {
                let path = fixture
                    .0
                    .join("models--example--tiny-bert/snapshots")
                    .join("a".repeat(40));
                std::fs::create_dir_all(&path).unwrap();
                std::fs::copy(fixture.0.join("config.json"), path.join("config.json")).unwrap();
                path
            } else {
                fixture.0.clone()
            };
            std::os::unix::fs::symlink(&external_blob, root.join("model.safetensors")).unwrap();
            match BertWeights::load(&root) {
                Ok(_) => panic!("unrelated external weight symlink accepted"),
                Err(error) => assert!(error.to_string().contains("escapes model directory")),
            }
        }
    }
    #[test]
    fn rejects_wrong_shape_and_nonfinite_weights() {
        for (shape, value, message) in [
            (vec![1], 0.0f32, "wrong shape"),
            (vec![4, 2], f32::NAN, "nonfinite"),
        ] {
            let fixture = Fixture::new();
            let data = value.to_le_bytes().repeat(shape.iter().product());
            let tensor = TensorView::new(Dtype::F32, shape, &data).unwrap();
            std::fs::write(
                fixture.0.join("model.safetensors"),
                serialize([("embeddings.word_embeddings.weight", tensor)], None).unwrap(),
            )
            .unwrap();
            assert!(fixture.error().contains(message));
        }
    }
}
