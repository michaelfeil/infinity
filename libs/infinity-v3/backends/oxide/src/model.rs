// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.

use infinity_v3_backend_core::BackendError;
use safetensors::{
    tensor::{Metadata, TensorInfo},
    Dtype,
};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Component, Path};
use std::sync::{Arc, Mutex};

const SAFETENSORS_PREFIX_BYTES: u64 = 8;
const MAX_HEADER_BYTES: u64 = 100_000_000;
const READ_SCRATCH_BYTES: usize = 64 * 1024;
const NONFINITE: &str = "invalid or nonfinite checkpoint weights";
const MAX_SIDECAR_BYTES: u64 = 16 * 1024 * 1024;

fn start(error: impl std::fmt::Display) -> BackendError {
    BackendError::Start(format!("Plain CUDA BERT: {error}"))
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, BackendError> {
    let file = File::open(path).map_err(start)?;
    let metadata = file.metadata().map_err(start)?;
    if !metadata.is_file() || metadata.len() > MAX_SIDECAR_BYTES {
        return Err(start(
            "checkpoint JSON must be a regular file of at most 16 MiB",
        ));
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(usize::try_from(metadata.len()).map_err(start)?)
        .map_err(start)?;
    file.take(MAX_SIDECAR_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(start)?;
    if bytes.len() as u64 > MAX_SIDECAR_BYTES {
        return Err(start("checkpoint JSON exceeds 16 MiB"));
    }
    serde_json::from_slice(&bytes).map_err(start)
}

fn path_present(path: &Path) -> Result<bool, BackendError> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(start(error)),
    }
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
                "only encoder BERT with absolute positions and tanh GELU is supported",
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

#[derive(Clone, Debug)]
pub(crate) struct WeightSource {
    file: Arc<Mutex<File>>,
    file_len: u64,
    data_start: u64,
    data_offset: u64,
    dtype: Dtype,
    shape: Vec<usize>,
    count: usize,
    width: usize,
}
impl WeightSource {
    fn from_header(
        file: Arc<Mutex<File>>,
        file_len: u64,
        data_start: u64,
        info: &TensorHeader,
        name: &str,
    ) -> Result<Self, BackendError> {
        let count = elements(&info.shape)?;
        let width = match info.dtype {
            Dtype::F32 => 4,
            Dtype::F16 | Dtype::BF16 => 2,
            other => {
                return Err(start(format!(
                    "unsupported weight dtype {other:?} for {name}"
                )))
            }
        };
        let byte_len = count
            .checked_mul(width)
            .ok_or_else(|| start(format!("weight {name} byte size overflow")))?;
        let (begin, end) = info.data_offsets;
        let declared = end
            .checked_sub(begin)
            .ok_or_else(|| start(format!("invalid data range for {name}")))?;
        if declared != byte_len {
            return Err(start(format!("invalid byte length for {name}")));
        }
        let data_offset = u64::try_from(begin)
            .map_err(|_| start(format!("weight offset overflow for {name}")))?;
        let byte_len = u64::try_from(byte_len)
            .map_err(|_| start(format!("weight byte size overflow for {name}")))?;
        let absolute = data_start
            .checked_add(data_offset)
            .ok_or_else(|| start(format!("weight file offset overflow for {name}")))?;
        let absolute_end = absolute
            .checked_add(byte_len)
            .ok_or_else(|| start(format!("weight file offset overflow for {name}")))?;
        if absolute_end > file_len {
            return Err(start(format!("weight {name} exceeds checkpoint file")));
        }
        Ok(Self {
            file,
            file_len,
            data_start,
            data_offset,
            dtype: info.dtype,
            shape: info.shape.clone(),
            count,
            width,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.count
    }

    #[cfg(test)]
    pub(crate) fn dtype(&self) -> Dtype {
        self.dtype
    }

    pub(crate) fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub(crate) fn read_bf16(
        &self,
        offset_elements: usize,
        out: &mut [u16],
    ) -> Result<(), BackendError> {
        let end_element = offset_elements
            .checked_add(out.len())
            .ok_or_else(|| start("weight read range overflow"))?;
        if end_element > self.count {
            return Err(start("weight read range exceeds tensor"));
        }
        if out.is_empty() {
            return Ok(());
        }
        let offset_bytes = offset_elements
            .checked_mul(self.width)
            .ok_or_else(|| start("weight byte offset overflow"))?;
        let offset_bytes =
            u64::try_from(offset_bytes).map_err(|_| start("weight byte offset overflow"))?;
        let file_offset = self
            .data_start
            .checked_add(self.data_offset)
            .and_then(|base| base.checked_add(offset_bytes))
            .ok_or_else(|| start("weight file offset overflow"))?;
        if self.width == 0 {
            return Err(start("invalid weight element width"));
        }
        let per_read = READ_SCRATCH_BYTES / self.width;
        if per_read == 0 {
            return Err(start("invalid weight element width"));
        }
        let mut file = self
            .file
            .lock()
            .map_err(|_| start("checkpoint shard lock is poisoned"))?;
        let current_len = file
            .metadata()
            .map_err(|error| start(format!("checkpoint shard metadata failed: {error}")))?
            .len();
        if current_len != self.file_len {
            return Err(start("checkpoint shard changed length after validation"));
        }
        file.seek(SeekFrom::Start(file_offset))
            .map_err(|error| start(format!("checkpoint shard seek failed: {error}")))?;
        let mut scratch = [0u8; READ_SCRATCH_BYTES];
        let mut done = 0usize;
        while done < out.len() {
            let take = (out.len() - done).min(per_read);
            let byte_count = take
                .checked_mul(self.width)
                .ok_or_else(|| start("weight chunk size overflow"))?;
            let bytes = scratch
                .get_mut(..byte_count)
                .ok_or_else(|| start("weight scratch buffer is too small"))?;
            file.read_exact(bytes)
                .map_err(|error| start(format!("checkpoint shard read failed: {error}")))?;
            let end = done
                .checked_add(take)
                .ok_or_else(|| start("weight output range overflow"))?;
            let target = out
                .get_mut(done..end)
                .ok_or_else(|| start("weight output range error"))?;
            convert_source(self.dtype, bytes, target)?;
            done = end;
        }
        Ok(())
    }
}

fn to_bf16(value: f32) -> Result<u16, BackendError> {
    if !value.is_finite() {
        return Err(start(NONFINITE));
    }
    let bits = half::bf16::from_f32(value).to_bits();
    if !half::bf16::from_bits(bits).is_finite() {
        return Err(start(NONFINITE));
    }
    Ok(bits)
}

fn convert_source(dtype: Dtype, bytes: &[u8], out: &mut [u16]) -> Result<(), BackendError> {
    let width = match dtype {
        Dtype::F32 => 4,
        Dtype::F16 | Dtype::BF16 => 2,
        other => return Err(start(format!("unsupported weight dtype {other:?}"))),
    };
    let expected = out
        .len()
        .checked_mul(width)
        .ok_or_else(|| start("weight conversion size overflow"))?;
    if bytes.len() != expected {
        return Err(start("weight source data length mismatch"));
    }
    match dtype {
        Dtype::BF16 => {
            for (chunk, slot) in bytes.chunks_exact(2).zip(out.iter_mut()) {
                let raw: [u8; 2] = chunk
                    .try_into()
                    .map_err(|_| start("invalid BF16 weight data"))?;
                let bits = u16::from_le_bytes(raw);
                if !half::bf16::from_bits(bits).is_finite() {
                    return Err(start(NONFINITE));
                }
                *slot = bits;
            }
        }
        Dtype::F16 => {
            for (chunk, slot) in bytes.chunks_exact(2).zip(out.iter_mut()) {
                let raw: [u8; 2] = chunk
                    .try_into()
                    .map_err(|_| start("invalid F16 weight data"))?;
                let value = half::f16::from_bits(u16::from_le_bytes(raw)).to_f32();
                *slot = to_bf16(value)?;
            }
        }
        Dtype::F32 => {
            for (chunk, slot) in bytes.chunks_exact(4).zip(out.iter_mut()) {
                let raw: [u8; 4] = chunk
                    .try_into()
                    .map_err(|_| start("invalid F32 weight data"))?;
                *slot = to_bf16(f32::from_le_bytes(raw))?;
            }
        }
        other => return Err(start(format!("unsupported weight dtype {other:?}"))),
    }
    Ok(())
}

pub(crate) struct LinearWeights {
    pub weight: WeightSource,
    pub bias: WeightSource,
}
pub(crate) struct LayerWeights {
    pub query: LinearWeights,
    pub key: LinearWeights,
    pub value: LinearWeights,
    pub attention: LinearWeights,
    pub intermediate: LinearWeights,
    pub output: LinearWeights,
    pub attention_gamma: WeightSource,
    pub attention_beta: WeightSource,
    pub output_gamma: WeightSource,
    pub output_beta: WeightSource,
}
pub(crate) struct BertWeights {
    pub config: BertConfig,
    pub word: WeightSource,
    pub position: WeightSource,
    pub token_type: WeightSource,
    pub gamma: WeightSource,
    pub beta: WeightSource,
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
    if !path_present(&file)? {
        return Ok(());
    }
    #[derive(Deserialize)]
    struct Module {
        #[serde(rename = "type")]
        kind: String,
    }
    let modules: Vec<Module> = read_json(&file)?;
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
    if path_present(&path.join("model.safetensors"))? {
        return Ok((vec!["model.safetensors".into()], None));
    }
    #[derive(Deserialize)]
    struct Index {
        weight_map: BTreeMap<String, String>,
    }
    let index: Index = read_json(&path.join("model.safetensors.index.json"))?;
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

type TensorHeader = TensorInfo;
struct ShardHeader {
    file: Arc<Mutex<File>>,
    file_len: u64,
    data_start: u64,
    tensors: Vec<(String, TensorHeader)>,
}

fn open_shard(path: &Path) -> Result<ShardHeader, BackendError> {
    let mut file = File::open(path).map_err(start)?;
    let file_info = file.metadata().map_err(start)?;
    if !file_info.is_file() {
        return Err(start("Safetensors input must be a regular file"));
    }
    let file_len = file_info.len();
    let mut length_bytes = [0u8; 8];
    file.read_exact(&mut length_bytes)
        .map_err(|error| start(format!("invalid safetensors length header: {error}")))?;
    let header_len = u64::from_le_bytes(length_bytes);
    if header_len == 0 || header_len > MAX_HEADER_BYTES {
        return Err(start("invalid or oversized safetensors metadata header"));
    }
    let data_start = SAFETENSORS_PREFIX_BYTES
        .checked_add(header_len)
        .ok_or_else(|| start("safetensors data offset overflow"))?;
    if data_start > file_len {
        return Err(start("safetensors metadata header exceeds file length"));
    }
    let header_len_usize = usize::try_from(header_len)
        .map_err(|_| start("safetensors metadata header is too large"))?;
    let mut header = Vec::new();
    header
        .try_reserve_exact(header_len_usize)
        .map_err(|_| start("safetensors metadata header allocation failed"))?;
    header.resize(header_len_usize, 0);
    file.read_exact(&mut header)
        .map_err(|error| start(format!("invalid safetensors metadata header: {error}")))?;
    // Reuse the safetensors parser's shape/bit-width/range validation rather
    // than maintaining a second tensor container parser.
    let metadata: Metadata = serde_json::from_slice(&header).map_err(start)?;
    if data_start.checked_add(u64::try_from(metadata.data_len()).map_err(start)?) != Some(file_len)
    {
        return Err(start("safetensors file length mismatch"));
    }
    let tensors = metadata
        .offset_keys()
        .into_iter()
        .map(|name| {
            let info = metadata
                .info(&name)
                .cloned()
                .ok_or_else(|| start("Missing tensor metadata"))?;
            Ok((name, info))
        })
        .collect::<Result<Vec<_>, BackendError>>()?;
    Ok(ShardHeader {
        file: Arc::new(Mutex::new(file)),
        file_len,
        data_start,
        tensors,
    })
}

fn take(
    weights: &mut HashMap<String, WeightSource>,
    name: &str,
) -> Result<WeightSource, BackendError> {
    weights
        .remove(name)
        .ok_or_else(|| start(format!("missing weight {name}")))
}
fn linear(
    weights: &mut HashMap<String, WeightSource>,
    prefix: &str,
) -> Result<LinearWeights, BackendError> {
    Ok(LinearWeights {
        weight: take(weights, &format!("{prefix}.weight"))?,
        bias: take(weights, &format!("{prefix}.bias"))?,
    })
}
impl BertWeights {
    pub(crate) fn load(path: &Path) -> Result<Self, BackendError> {
        let config: BertConfig = read_json(&path.join("config.json"))?;
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
            let shard = open_shard(&resolved)?;
            for (name, info) in &shard.tensors {
                let key = name.strip_prefix("bert.").unwrap_or(name);
                let Some(shape) = expected.get(key) else {
                    continue;
                };
                if index
                    .as_ref()
                    .is_some_and(|index| index.get(name) != Some(&file))
                {
                    return Err(start(format!("weight {name} is not in its indexed shard")));
                }
                if info.shape != *shape {
                    return Err(start(format!(
                        "wrong shape for {name}: expected {shape:?}, got {:?}",
                        info.shape
                    )));
                }
                let source = WeightSource::from_header(
                    Arc::clone(&shard.file),
                    shard.file_len,
                    shard.data_start,
                    info,
                    name,
                )?;
                if weights.insert(key.to_owned(), source).is_some() {
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
    use std::fs::OpenOptions;
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
            self.weight_values(dtype, prefixed, sharded, |name, _| {
                if name.contains("LayerNorm.weight") {
                    1.0
                } else {
                    0.25
                }
            });
        }
        fn weight_values(
            &self,
            dtype: Dtype,
            prefixed: bool,
            sharded: bool,
            value: impl Fn(&str, usize) -> f32,
        ) {
            let config: BertConfig =
                serde_json::from_slice(&std::fs::read(self.0.join("config.json")).unwrap())
                    .unwrap();
            let mut entries = required(&config).into_iter().collect::<Vec<_>>();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            let data = entries
                .iter()
                .map(|(name, shape)| {
                    let count = shape.iter().product::<usize>();
                    let mut bytes = Vec::new();
                    for index in 0..count {
                        let scalar = value(name, index);
                        match dtype {
                            Dtype::F32 => bytes.extend_from_slice(&scalar.to_le_bytes()),
                            Dtype::F16 => bytes.extend_from_slice(
                                &half::f16::from_f32(scalar).to_bits().to_le_bytes(),
                            ),
                            Dtype::BF16 => bytes.extend_from_slice(
                                &half::bf16::from_f32(scalar).to_bits().to_le_bytes(),
                            ),
                            _ => unreachable!(),
                        }
                    }
                    bytes
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

    fn bf16(value: f32) -> u16 {
        half::bf16::from_f32(value).to_bits()
    }
    fn read_all(source: &WeightSource) -> Vec<u16> {
        let mut out = vec![0u16; source.len()];
        source.read_bf16(0, &mut out).unwrap();
        out
    }
    fn expected(values: &[f32]) -> Vec<u16> {
        values.iter().map(|value| bf16(*value)).collect()
    }

    #[test]
    fn loads_float_precisions_and_bert_prefix() {
        for dtype in [Dtype::F32, Dtype::F16, Dtype::BF16] {
            let fixture = Fixture::new();
            fixture.weights(dtype, true, false);
            let weights = BertWeights::load(&fixture.0).unwrap();
            assert_eq!(weights.layers.len(), 1);
            assert_eq!(read_all(&weights.word), expected(&[0.25; 8]));
            assert_eq!(read_all(&weights.gamma), expected(&[1.0; 2]));
            assert_eq!(weights.word.len(), 8);
            assert_eq!(weights.word.shape(), &[4, 2]);
            assert_eq!(weights.word.dtype(), dtype);
            assert_eq!(weights.config.type_vocab_size, 2);
            assert_eq!(weights.config.layer_norm_eps, 1e-12);
        }
    }
    #[test]
    fn loads_standard_sharded_bare_checkpoint() {
        let fixture = Fixture::new();
        fixture.weights(Dtype::F32, false, true);
        let weights = BertWeights::load(&fixture.0).unwrap();
        assert_eq!(
            read_all(&weights.layers[0].intermediate.weight),
            expected(&[0.25; 8])
        );
        assert_eq!(
            read_all(&weights.layers[0].output.bias),
            expected(&[0.25; 2])
        );
    }
    #[test]
    fn reads_bounded_element_ranges() {
        let fixture = Fixture::new();
        fixture.weight_values(Dtype::F32, false, false, |_, index| index as f32 + 0.5);
        let weights = BertWeights::load(&fixture.0).unwrap();
        let word = &weights.word;
        let mut out = [0u16; 4];
        word.read_bf16(3, &mut out).unwrap();
        assert_eq!(out, [3usize, 4, 5, 6].map(|index| bf16(index as f32 + 0.5)));
        let mut tail = [0u16; 1];
        let last = word.len() - 1;
        word.read_bf16(last, &mut tail).unwrap();
        assert_eq!(tail[0], bf16(last as f32 + 0.5));
        word.read_bf16(word.len(), &mut []).unwrap();
        assert!(word.read_bf16(word.len(), &mut [0u16; 1]).is_err());
        assert!(word.read_bf16(usize::MAX, &mut [0u16; 1]).is_err());
    }
    #[test]
    fn converts_source_floats_to_bf16_bits() {
        for (dtype, value) in [(Dtype::F32, 1.0f32 / 3.0), (Dtype::F16, 0.1f32)] {
            let fixture = Fixture::new();
            fixture.weight_values(dtype, false, false, |_, _| value);
            let weights = BertWeights::load(&fixture.0).unwrap();
            let source = if dtype == Dtype::F16 {
                half::f16::from_f32(value).to_f32()
            } else {
                value
            };
            let expected = bf16(source);
            assert!(read_all(&weights.word).iter().all(|bits| *bits == expected));
        }
    }

    #[test]
    fn preserves_bf16_and_rounds_ties_to_even() {
        let bytes = [0x3f81_8000u32, 0x3f80_8000, 0xbf81_8000]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        let mut out = [0u16; 3];
        convert_source(Dtype::F32, &bytes, &mut out).unwrap();
        assert_eq!(out, [0x3f82, 0x3f80, 0xbf82]);
        let mut unchanged = [0u16; 1];
        convert_source(Dtype::BF16, &[0x81, 0x3f], &mut unchanged).unwrap();
        assert_eq!(unchanged, [0x3f81]);
    }

    #[test]
    fn rejects_oversized_json_before_reading_payload() {
        let fixture = Fixture::new();
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(fixture.0.join("config.json"))
            .unwrap();
        file.set_len(MAX_SIDECAR_BYTES + 1).unwrap();
        assert!(fixture.error().contains("16 MiB"));
    }

    #[cfg(unix)]
    #[test]
    fn broken_pipeline_sidecar_cannot_bypass_module_validation() {
        let fixture = Fixture::new();
        std::os::unix::fs::symlink("missing.json", fixture.0.join("modules.json")).unwrap();
        assert!(BertWeights::load(&fixture.0).is_err());
    }
    #[test]
    fn detects_checkpoint_length_changes_on_read() {
        for extend in [false, true] {
            let fixture = Fixture::new();
            fixture.weights(Dtype::F32, false, false);
            let weights = BertWeights::load(&fixture.0).unwrap();
            let file = OpenOptions::new()
                .write(true)
                .open(fixture.0.join("model.safetensors"))
                .unwrap();
            let length = file.metadata().unwrap().len();
            file.set_len(if extend { length + 8 } else { length - 1 })
                .unwrap();
            let mut out = vec![0u16; weights.word.len()];
            let error = weights.word.read_bf16(0, &mut out).unwrap_err().to_string();
            assert!(error.contains("changed length"), "{error}");
        }
    }
    #[test]
    fn rejects_checkpoint_with_missing_or_extra_payload_bytes_at_load() {
        for extend in [false, true] {
            let fixture = Fixture::new();
            fixture.weights(Dtype::F32, false, false);
            let file = OpenOptions::new()
                .write(true)
                .open(fixture.0.join("model.safetensors"))
                .unwrap();
            let length = file.metadata().unwrap().len();
            file.set_len(if extend { length + 1 } else { length - 1 })
                .unwrap();
            drop(file);
            let error = fixture.error();
            assert!(error.contains("length mismatch"), "{error}");
        }
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
            assert_eq!(read_all(&model.word), expected(&[0.25; 8]));
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
    fn rejects_wrong_shape_weights() {
        let fixture = Fixture::new();
        let shape = vec![1];
        let data = 0.0f32.to_le_bytes().to_vec();
        let tensor = TensorView::new(Dtype::F32, shape, &data).unwrap();
        std::fs::write(
            fixture.0.join("model.safetensors"),
            serialize([("embeddings.word_embeddings.weight", tensor)], None).unwrap(),
        )
        .unwrap();
        assert!(fixture.error().contains("wrong shape"));
    }
    #[test]
    fn rejects_nonfinite_checkpoint_values_on_read() {
        for (dtype, value) in [
            (Dtype::F32, f32::NAN),
            (Dtype::F32, f32::INFINITY),
            (Dtype::F32, f32::MAX),
            (Dtype::F16, f32::NAN),
            (Dtype::F16, f32::INFINITY),
            (Dtype::BF16, f32::NAN),
            (Dtype::BF16, f32::INFINITY),
        ] {
            let fixture = Fixture::new();
            fixture.weight_values(dtype, false, false, |_, _| value);
            let weights = BertWeights::load(&fixture.0).unwrap();
            let mut out = vec![0u16; weights.word.len()];
            let error = weights.word.read_bf16(0, &mut out).unwrap_err().to_string();
            assert!(error.contains("nonfinite"), "{dtype:?} {value}: {error}");
        }
    }

    #[test]
    #[ignore = "requires CUDA 13 GPU runtime"]
    fn gpu_streams_multiple_chunks_into_final_packed_weights() {
        use crate::upload::WeightUploader;
        use cuda_core::Context;
        // More than two staging slots, plus an odd tail, for every source type.
        let count = 600_003usize;
        let fixture = Fixture::new();
        let values = (0..count)
            .map(|i| ((i % 997) as f32 - 498.0) / 37.0)
            .collect::<Vec<_>>();
        let floats = values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        let halfs = values
            .iter()
            .flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        let bfloats = values
            .iter()
            .flat_map(|v| half::bf16::from_f32(*v).to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        let views = [
            (
                "q",
                TensorView::new(Dtype::F32, vec![count], &floats).unwrap(),
            ),
            (
                "k",
                TensorView::new(Dtype::F16, vec![count], &halfs).unwrap(),
            ),
            (
                "v",
                TensorView::new(Dtype::BF16, vec![count], &bfloats).unwrap(),
            ),
        ];
        std::fs::write(
            fixture.0.join("model.safetensors"),
            serialize(views, None).unwrap(),
        )
        .unwrap();
        let shard = open_shard(&fixture.0.join("model.safetensors")).unwrap();
        let sources = ["q", "k", "v"].map(|name| {
            let info = &shard.tensors.iter().find(|(key, _)| key == name).unwrap().1;
            WeightSource::from_header(
                shard.file.clone(),
                shard.file_len,
                shard.data_start,
                info,
                name,
            )
            .unwrap()
        });
        let context = Context::new(0).unwrap();
        let stream = context.new_stream().unwrap();
        let mut uploader = WeightUploader::new(stream.clone()).unwrap();
        let packed = uploader
            .upload(&sources.iter().collect::<Vec<_>>())
            .unwrap();
        uploader.finish().unwrap();
        let actual = packed
            .to_host_vec(&stream)
            .unwrap()
            .into_iter()
            .map(crate::tensor::Bf16::to_bits)
            .collect::<Vec<_>>();
        let expected = values
            .iter()
            .map(|v| half::bf16::from_f32(*v).to_bits())
            .chain(
                values
                    .iter()
                    .map(|v| half::bf16::from_f32(half::f16::from_f32(*v).to_f32()).to_bits()),
            )
            .chain(values.iter().map(|v| half::bf16::from_f32(*v).to_bits()))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
        // A later read fails after previous chunks have already enqueued. The
        // uploader must drain without publishing incomplete weights or reusing
        // a pinned slot still read by CUDA.
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .open(fixture.0.join("model.safetensors"))
            .unwrap();
        file.seek(SeekFrom::Start(
            shard.data_start + sources[0].data_offset + ((count - 1) * 4) as u64,
        ))
        .unwrap();
        file.write_all(&f32::NAN.to_le_bytes()).unwrap();
        assert!(uploader.upload(&[&sources[0]]).is_err());
        uploader.finish().unwrap();
    }
}
