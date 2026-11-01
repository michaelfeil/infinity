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
    BackendError::Start(format!("Oxide: {error}"))
}

pub(crate) fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, BackendError> {
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

pub(crate) fn path_present(path: &Path) -> Result<bool, BackendError> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(start(error)),
    }
}
const BERT_MAX_POSITION_EMBEDDINGS: usize = 4096;
const DISTIL_LAYER_NORM_EPS: f64 = 1e-12;
const DISTIL_TOKEN_TYPE_ROWS: usize = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum EncoderFamily {
    Bert,
    Roberta,
    XlmRoberta,
    Camembert,
    DistilBert,
}
impl EncoderFamily {
    fn from_model_type(model_type: &str) -> Option<Self> {
        match model_type {
            "bert" => Some(Self::Bert),
            "roberta" => Some(Self::Roberta),
            "xlm-roberta" => Some(Self::XlmRoberta),
            "camembert" => Some(Self::Camembert),
            "distilbert" => Some(Self::DistilBert),
            _ => None,
        }
    }

    fn roots(self) -> &'static [&'static str] {
        match self {
            Self::Bert => &["bert.", ""],
            Self::Roberta | Self::XlmRoberta | Self::Camembert => {
                &["roberta.", "xlm-roberta.", "camembert.", ""]
            }
            Self::DistilBert => &["distilbert.", ""],
        }
    }

    fn accepts_architecture(self, architecture: &str) -> bool {
        match self {
            Self::Bert => matches!(
                architecture,
                "BertModel" | "BertForMaskedLM" | "BertForPreTraining"
            ),
            Self::Roberta => matches!(
                architecture,
                "RobertaModel" | "RobertaForMaskedLM" | "RobertaForPreTraining"
            ),
            Self::XlmRoberta => matches!(
                architecture,
                "XLMRobertaModel" | "XLMRobertaForMaskedLM" | "XLMRobertaForPreTraining"
            ),
            Self::Camembert => matches!(architecture, "CamembertModel" | "CamembertForMaskedLM"),
            Self::DistilBert => matches!(architecture, "DistilBertModel" | "DistilBertForMaskedLM"),
        }
    }
}

/// Arithmetic selected at load time for the eventual engine kernels.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NormArithmetic {
    /// FP32 residual statistics with the biased BERT-family LayerNorm primitive.
    ExactBiased,
    /// Round embedding/residual additions to BF16 before normalization statistics.
    RoundedAddBeforeStats,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct EncoderRecipe {
    pub(crate) embedding: NormArithmetic,
    pub(crate) residual: NormArithmetic,
}
impl EncoderRecipe {
    fn for_family(family: EncoderFamily, hidden_size: usize) -> Self {
        if family != EncoderFamily::DistilBert && matches!(hidden_size, 384 | 768) {
            Self {
                embedding: NormArithmetic::ExactBiased,
                residual: NormArithmetic::ExactBiased,
            }
        } else {
            Self {
                embedding: NormArithmetic::RoundedAddBeforeStats,
                residual: NormArithmetic::RoundedAddBeforeStats,
            }
        }
    }
}

#[derive(Debug, Deserialize)]
struct RawEncoderConfig {
    model_type: String,
    architectures: Option<Vec<String>>,
    hidden_size: Option<usize>,
    intermediate_size: Option<usize>,
    num_attention_heads: Option<usize>,
    num_hidden_layers: Option<usize>,
    max_position_embeddings: Option<usize>,
    vocab_size: Option<usize>,
    type_vocab_size: Option<usize>,
    layer_norm_eps: Option<f64>,
    hidden_act: Option<String>,
    activation: Option<String>,
    position_embedding_type: Option<String>,
    is_decoder: Option<bool>,
    add_cross_attention: Option<bool>,
    dim: Option<usize>,
    hidden_dim: Option<usize>,
    n_heads: Option<usize>,
    n_layers: Option<usize>,
    // DistilBERT's sinusoidal flag controls initialization, not lookup of a
    // checkpoint position table. The stored table remains required below.
}

impl<'de> Deserialize<'de> for BertConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = RawEncoderConfig::deserialize(deserializer)?;
        Self::from_raw(raw).map_err(serde::de::Error::custom)
    }
}

fn required_usize(value: Option<usize>, name: &str) -> Result<usize, BackendError> {
    value.ok_or_else(|| start(format!("missing encoder config field {name}")))
}
fn aliased_usize(
    primary: Option<usize>,
    alias: Option<usize>,
    primary_name: &str,
    alias_name: &str,
) -> Result<usize, BackendError> {
    match (primary, alias) {
        (Some(primary), Some(alias)) if primary != alias => Err(start(format!(
            "ambiguous encoder config {primary_name}={primary} and {alias_name}={alias}"
        ))),
        (Some(value), _) | (None, Some(value)) => Ok(value),
        (None, None) => Err(start(format!(
            "missing encoder config field {primary_name}"
        ))),
    }
}
fn aliased_string(
    primary: Option<String>,
    alias: Option<String>,
    primary_name: &str,
    alias_name: &str,
) -> Result<String, BackendError> {
    match (primary, alias) {
        (Some(primary), Some(alias)) if primary != alias => Err(start(format!(
            "ambiguous encoder config {primary_name}={primary} and {alias_name}={alias}"
        ))),
        (Some(value), _) | (None, Some(value)) => Ok(value),
        (None, None) => Ok("gelu".into()),
    }
}

#[derive(Clone, Debug)]
pub(crate) struct BertConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
    pub type_vocab_size: usize,
    configured_type_vocab_size: Option<usize>,
    pub layer_norm_eps: f64,
    pub(crate) family: EncoderFamily,
    // The encoder executor consumes this once it is integrated into lib.rs.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) recipe: EncoderRecipe,
    hidden_act: String,
    position_embedding_type: String,
    is_decoder: bool,
    add_cross_attention: bool,
}
impl BertConfig {
    fn load(path: &Path) -> Result<Self, BackendError> {
        let raw: RawEncoderConfig = read_json(&path.join("config.json"))?;
        Self::from_raw(raw)
    }

    fn from_raw(raw: RawEncoderConfig) -> Result<Self, BackendError> {
        let family = EncoderFamily::from_model_type(&raw.model_type).ok_or_else(|| {
            start(format!(
                "unsupported encoder model_type {}; expected bert, roberta, xlm-roberta, camembert or distilbert",
                raw.model_type
            ))
        })?;
        if let Some(architectures) = raw.architectures.as_deref() {
            if let Some(architecture) = architectures
                .iter()
                .find(|architecture| !family.accepts_architecture(architecture))
            {
                return Err(start(format!(
                    "unsupported encoder architecture {architecture} for model_type {}",
                    raw.model_type
                )));
            }
        }
        let distil = family == EncoderFamily::DistilBert;
        if distil
            && raw
                .layer_norm_eps
                .is_some_and(|epsilon| epsilon != DISTIL_LAYER_NORM_EPS)
        {
            return Err(start("DistilBERT layer_norm_eps must be 1e-12"));
        }
        let hidden_size = if distil {
            aliased_usize(raw.dim, raw.hidden_size, "dim", "hidden_size")?
        } else {
            required_usize(raw.hidden_size, "hidden_size")?
        };
        let intermediate_size = if distil {
            aliased_usize(
                raw.hidden_dim,
                raw.intermediate_size,
                "hidden_dim",
                "intermediate_size",
            )?
        } else {
            required_usize(raw.intermediate_size, "intermediate_size")?
        };
        let num_attention_heads = if distil {
            aliased_usize(
                raw.n_heads,
                raw.num_attention_heads,
                "n_heads",
                "num_attention_heads",
            )?
        } else {
            required_usize(raw.num_attention_heads, "num_attention_heads")?
        };
        let num_hidden_layers = if distil {
            aliased_usize(
                raw.n_layers,
                raw.num_hidden_layers,
                "n_layers",
                "num_hidden_layers",
            )?
        } else {
            required_usize(raw.num_hidden_layers, "num_hidden_layers")?
        };
        let hidden_act = if distil {
            aliased_string(raw.activation, raw.hidden_act, "activation", "hidden_act")?
        } else {
            raw.hidden_act.unwrap_or_else(|| "gelu".into())
        };
        let config = Self {
            hidden_size,
            intermediate_size,
            num_attention_heads,
            num_hidden_layers,
            max_position_embeddings: required_usize(
                raw.max_position_embeddings,
                "max_position_embeddings",
            )?,
            vocab_size: required_usize(raw.vocab_size, "vocab_size")?,
            type_vocab_size: raw.type_vocab_size.unwrap_or(DISTIL_TOKEN_TYPE_ROWS),
            configured_type_vocab_size: raw.type_vocab_size,
            layer_norm_eps: if distil {
                DISTIL_LAYER_NORM_EPS
            } else {
                raw.layer_norm_eps.unwrap_or(1e-12)
            },
            family,
            recipe: EncoderRecipe::for_family(family, hidden_size),
            hidden_act,
            position_embedding_type: raw
                .position_embedding_type
                .unwrap_or_else(|| "absolute".into()),
            is_decoder: raw.is_decoder.unwrap_or(false),
            add_cross_attention: raw.add_cross_attention.unwrap_or(false),
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), BackendError> {
        if self.is_decoder
            || self.add_cross_attention
            || self.position_embedding_type != "absolute"
            || self.hidden_act != "gelu"
        {
            return Err(start(
                "only encoder BERT-family models with absolute positions and tanh GELU are supported",
            ));
        }
        if self.hidden_size == 0
            || self.intermediate_size == 0
            || self.num_attention_heads == 0
            || self.num_hidden_layers == 0
            || self.vocab_size == 0
            || self.type_vocab_size == 0
            || self.max_position_embeddings == 0
            || u32::try_from(self.max_position_embeddings).is_err()
            || (self.family == EncoderFamily::Bert
                && self.max_position_embeddings > BERT_MAX_POSITION_EMBEDDINGS)
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
        // CUDA position IDs are u32; checked tensor dimensions bound the actual
        // lookup allocation. Attention-provider sequence limits are separate.
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
    file: Option<Arc<Mutex<File>>>,
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
            file: Some(file),
            file_len,
            data_start,
            data_offset,
            dtype: info.dtype,
            shape: info.shape.clone(),
            count,
            width,
        })
    }

    fn zeroed(shape: Vec<usize>) -> Result<Self, BackendError> {
        let count = elements(&shape)?;
        Ok(Self {
            file: None,
            file_len: 0,
            data_start: 0,
            data_offset: 0,
            dtype: Dtype::BF16,
            shape,
            count,
            width: 2,
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
        let Some(file) = self.file.as_ref() else {
            out.fill(0);
            return Ok(());
        };
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
        let mut file = file
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

pub(crate) fn elements(shape: &[usize]) -> Result<usize, BackendError> {
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

#[derive(Clone, Copy)]
struct CheckpointLayout {
    family: EncoderFamily,
    root: &'static str,
    layer_prefix: &'static str,
}
impl CheckpointLayout {
    fn actual(&self, canonical: &str) -> Result<String, BackendError> {
        let body = if self.family == EncoderFamily::DistilBert {
            self.distil_name(canonical)?
        } else {
            canonical.to_owned()
        };
        Ok(format!("{}{body}", self.root))
    }

    fn distil_name(&self, canonical: &str) -> Result<String, BackendError> {
        let Some(rest) = canonical.strip_prefix("encoder.layer.") else {
            return Ok(canonical.to_owned());
        };
        let (layer, tail) = rest
            .split_once('.')
            .ok_or_else(|| start(format!("invalid DistilBERT layer tensor {canonical}")))?;
        let mapped = [
            ("attention.self.query.", "attention.q_lin."),
            ("attention.self.key.", "attention.k_lin."),
            ("attention.self.value.", "attention.v_lin."),
            ("attention.output.dense.", "attention.out_lin."),
            ("intermediate.dense.", "ffn.lin1."),
            ("output.dense.", "ffn.lin2."),
            ("attention.output.LayerNorm.", "sa_layer_norm."),
            ("output.LayerNorm.", "output_layer_norm."),
        ]
        .iter()
        .find_map(|(from, to)| {
            tail.strip_prefix(from)
                .map(|suffix| format!("{to}{suffix}"))
        })
        .ok_or_else(|| start(format!("unmapped DistilBERT layer tensor {canonical}")))?;
        Ok(format!("{}.{layer}.{mapped}", self.layer_prefix))
    }
}

fn select_root(
    family: EncoderFamily,
    names: &BTreeSet<String>,
) -> Result<&'static str, BackendError> {
    let suffix = "embeddings.word_embeddings.weight";
    let anchors = names
        .iter()
        .filter(|name| name.ends_with(suffix))
        .collect::<Vec<_>>();
    let Some(anchor) = anchors.first() else {
        return Err(start(format!("missing weight {suffix}")));
    };
    if anchors.len() != 1 {
        return Err(start(
            "ambiguous checkpoint namespace: multiple embeddings.word_embeddings.weight roots",
        ));
    }
    family
        .roots()
        .iter()
        .copied()
        .find(|root| format!("{root}{suffix}") == anchor.as_str())
        .ok_or_else(|| {
            start(format!(
                "unsupported checkpoint namespace for model_type {:?}: {anchor}",
                family
            ))
        })
}

fn select_layer_prefix(
    family: EncoderFamily,
    root: &str,
    names: &BTreeSet<String>,
) -> Result<&'static str, BackendError> {
    if family != EncoderFamily::DistilBert {
        return Ok("encoder.layer");
    }
    let transformer = format!("{root}transformer.layer.");
    let encoder = format!("{root}encoder.layer.");
    let has_transformer = names.iter().any(|name| name.starts_with(&transformer));
    let has_encoder = names.iter().any(|name| name.starts_with(&encoder));
    match (has_transformer, has_encoder) {
        (true, false) => Ok("transformer.layer"),
        (false, true) => Ok("encoder.layer"),
        (true, true) => Err(start(
            "ambiguous DistilBERT layer namespace: transformer.layer and encoder.layer both present",
        )),
        (false, false) => Err(start(
            "missing DistilBERT transformer.layer or encoder.layer weights",
        )),
    }
}

fn reject_stray_token_types(
    layout: &CheckpointLayout,
    names: &BTreeSet<String>,
) -> Result<(), BackendError> {
    let suffix = "embeddings.token_type_embeddings.weight";
    let expected = layout.actual(suffix)?;
    if let Some(stray) = names
        .iter()
        .find(|name| name.ends_with(suffix) && name.as_str() != expected)
    {
        return Err(start(format!(
            "incoherent token-type embedding namespace {stray}"
        )));
    }
    Ok(())
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

/// One validated safetensors checkpoint laid out by the loader.
///
/// `open` preserves the shard path, index ownership, header and duplicate
/// guards used by the encoder loader. `source` then exposes only the requested
/// tensor metadata; payload bytes are read later by `WeightSource` in bounded
/// chunks. The helper itself never allocates a whole tensor or model.
pub(crate) struct Checkpoint {
    files: Vec<String>,
    index: Option<BTreeMap<String, String>>,
    shards: Vec<ShardHeader>,
    tensors: BTreeMap<String, (usize, TensorHeader)>,
}
impl Checkpoint {
    pub(crate) fn open(path: &Path) -> Result<Self, BackendError> {
        let (files, index) = checkpoint_files(path)?;
        let root = path.canonicalize().map_err(start)?;
        let mut shards = Vec::new();
        for file in &files {
            let resolved = path.join(file).canonicalize().map_err(start)?;
            if !permitted_weight(&root, &resolved) {
                return Err(start("checkpoint shard escapes model directory"));
            }
            shards.push(open_shard(&resolved)?);
        }
        let mut tensors = BTreeMap::new();
        for (shard_index, shard) in shards.iter().enumerate() {
            for (name, info) in &shard.tensors {
                if tensors
                    .insert(name.clone(), (shard_index, info.clone()))
                    .is_some()
                {
                    return Err(start(format!("duplicate checkpoint tensor {name}")));
                }
            }
        }
        Ok(Self {
            files,
            index,
            shards,
            tensors,
        })
    }

    pub(crate) fn names(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }

    pub(crate) fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub(crate) fn shape(&self, name: &str) -> Option<&[usize]> {
        self.tensors
            .get(name)
            .map(|(_, info)| info.shape.as_slice())
    }

    pub(crate) fn source(&self, name: &str, shape: &[usize]) -> Result<WeightSource, BackendError> {
        self.optional_source(name, shape)?
            .ok_or_else(|| start(format!("missing weight {name}")))
    }

    pub(crate) fn optional_source(
        &self,
        name: &str,
        shape: &[usize],
    ) -> Result<Option<WeightSource>, BackendError> {
        let Some((shard_index, info)) = self.tensors.get(name) else {
            return Ok(None);
        };
        let Some(file) = self.files.get(*shard_index) else {
            return Err(start(format!("invalid shard ownership for {name}")));
        };
        if self
            .index
            .as_ref()
            .is_some_and(|index| index.get(name) != Some(file))
        {
            return Err(start(format!("weight {name} is not in its indexed shard")));
        }
        if info.shape.as_slice() != shape {
            return Err(start(format!(
                "wrong shape for {name}: expected {shape:?}, got {:?}",
                info.shape
            )));
        }
        let Some(shard) = self.shards.get(*shard_index) else {
            return Err(start(format!("invalid shard ownership for {name}")));
        };
        let source = WeightSource::from_header(
            Arc::clone(&shard.file),
            shard.file_len,
            shard.data_start,
            info,
            name,
        )?;
        Ok(Some(source))
    }
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
        let mut config = BertConfig::load(path)?;
        validate_modules(path)?;
        let checkpoint = Checkpoint::open(path)?;
        let names = checkpoint.names().cloned().collect::<BTreeSet<_>>();
        let root_prefix = select_root(config.family, &names)?;
        let layer_prefix = select_layer_prefix(config.family, root_prefix, &names)?;
        let layout = CheckpointLayout {
            family: config.family,
            root: root_prefix,
            layer_prefix,
        };
        reject_stray_token_types(&layout, &names)?;

        if config.family == EncoderFamily::DistilBert {
            let name = layout.actual("embeddings.token_type_embeddings.weight")?;
            let shape = checkpoint.shape(&name);
            config.type_vocab_size = match shape {
                Some([rows, width]) if *rows > 0 && *width == config.hidden_size => {
                    if config
                        .configured_type_vocab_size
                        .is_some_and(|configured| configured != *rows)
                    {
                        return Err(start(format!(
                            "wrong shape for {name}: checkpoint rows {rows} disagree with configured type_vocab_size {}",
                            config.type_vocab_size
                        )));
                    }
                    *rows
                }
                Some(shape) => {
                    return Err(start(format!(
                        "wrong shape for {name}: expected nonempty [rows, {}], got {shape:?}",
                        config.hidden_size
                    )))
                }
                // No segment embeddings means no contribution for either
                // ordinary tokenizer pair ID, regardless of a config hint.
                None => DISTIL_TOKEN_TYPE_ROWS,
            };
            config.validate()?;
        }

        let mut expected = HashMap::new();
        for (canonical, shape) in required(&config) {
            let actual = layout.actual(&canonical)?;
            if expected.insert(actual, (canonical, shape)).is_some() {
                return Err(start("duplicate expected checkpoint tensor"));
            }
        }
        let mut weights = HashMap::new();
        let mut missing = Vec::new();
        for (name, (canonical, shape)) in expected {
            if !checkpoint.contains(&name) {
                if config.family == EncoderFamily::DistilBert
                    && canonical == "embeddings.token_type_embeddings.weight"
                {
                    continue;
                }
                missing.push(canonical);
                continue;
            }
            let source = checkpoint.source(&name, &shape)?;
            if weights.insert(canonical, source).is_some() {
                return Err(start("duplicate weight in checkpoint mapping"));
            }
        }
        if let Some(canonical) = missing.into_iter().next() {
            return Err(start(format!("missing weight {canonical}")));
        }
        let token_type = match weights.remove("embeddings.token_type_embeddings.weight") {
            Some(source) => source,
            None if config.family == EncoderFamily::DistilBert => {
                WeightSource::zeroed(vec![config.type_vocab_size, config.hidden_size])?
            }
            None => {
                return Err(start(
                    "missing weight embeddings.token_type_embeddings.weight",
                ))
            }
        };
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
            token_type,
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
            self.raw_config(value);
        }
        fn raw_config(&self, value: serde_json::Value) {
            std::fs::write(
                self.0.join("config.json"),
                serde_json::to_vec(&value).unwrap(),
            )
            .unwrap();
        }
        fn config_from_disk(&self) -> BertConfig {
            BertConfig::load(&self.0).unwrap()
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
            let config = self.config_from_disk();
            let layout = CheckpointLayout {
                family: EncoderFamily::Bert,
                root: if prefixed { "bert." } else { "" },
                layer_prefix: "encoder.layer",
            };
            self.write_layout(&config, layout, dtype, sharded, value);
        }
        fn write_layout(
            &self,
            config: &BertConfig,
            layout: CheckpointLayout,
            dtype: Dtype,
            sharded: bool,
            value: impl Fn(&str, usize) -> f32,
        ) {
            self.write_layout_omitting(config, layout, dtype, sharded, &[], value);
        }
        fn write_layout_omitting(
            &self,
            config: &BertConfig,
            layout: CheckpointLayout,
            dtype: Dtype,
            sharded: bool,
            omitted: &[&str],
            value: impl Fn(&str, usize) -> f32,
        ) {
            let mut entries = required(config)
                .into_iter()
                .filter(|(canonical, _)| !omitted.contains(&canonical.as_str()))
                .map(|(canonical, shape)| (layout.actual(&canonical).unwrap(), shape, canonical))
                .collect::<Vec<_>>();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            self.write_entries(dtype, sharded, &entries, value);
        }
        fn write_entries(
            &self,
            dtype: Dtype,
            sharded: bool,
            entries: &[(String, Vec<usize>, String)],
            value: impl Fn(&str, usize) -> f32,
        ) {
            let data = entries
                .iter()
                .map(|(_, shape, canonical)| {
                    let count = shape.iter().product::<usize>();
                    let mut bytes = Vec::new();
                    for index in 0..count {
                        let scalar = value(canonical, index);
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
                .map(|((name, shape, _), bytes)| {
                    (
                        name.clone(),
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
            serde_json::json!({"is_decoder":true}),
            serde_json::json!({"add_cross_attention":true}),
            serde_json::json!({"hidden_act":"gelu_new"}),
            serde_json::json!({"position_embedding_type":"relative_key"}),
            serde_json::json!({"num_attention_heads":0}),
            serde_json::json!({"hidden_size":3,"num_attention_heads":2}),
            serde_json::json!({"max_position_embeddings":8192}),
            serde_json::json!({"model_type":"custom-encoder"}),
            serde_json::json!({"architectures":["CustomEncoder"]}),
        ] {
            let fixture = Fixture::new();
            fixture.config(overrides);
            let error = fixture.error();
            assert!(
                error.contains("only encoder BERT")
                    || error.contains("invalid or unsupported")
                    || error.contains("unsupported encoder"),
                "{error}"
            );
        }
    }
    fn roberta_config(model_type: &str, architecture: &str) -> serde_json::Value {
        serde_json::json!({
            "model_type": model_type,
            "architectures": [architecture],
            "hidden_size": 2,
            "intermediate_size": 4,
            "num_attention_heads": 1,
            "num_hidden_layers": 1,
            "max_position_embeddings": 8,
            "vocab_size": 4,
            "type_vocab_size": 1,
            "layer_norm_eps": 1e-5,
            "hidden_act": "gelu",
            "position_embedding_type": "absolute"
        })
    }
    fn distil_config() -> serde_json::Value {
        serde_json::json!({
            "model_type": "distilbert",
            "architectures": ["DistilBertModel"],
            "dim": 2,
            "hidden_dim": 4,
            "n_heads": 1,
            "n_layers": 1,
            "max_position_embeddings": 8,
            "vocab_size": 4,
            "activation": "gelu",
            "sinusoidal_pos_embds": false,
            "pad_token_id": 0
        })
    }
    fn fixture_values(name: &str, index: usize) -> f32 {
        if name.ends_with("LayerNorm.weight") || name.ends_with("LayerNorm.bias") {
            1.0
        } else {
            (index + 1) as f32 / 16.0
        }
    }
    fn with_fields(mut value: serde_json::Value, fields: serde_json::Value) -> serde_json::Value {
        for (key, field) in fields.as_object().unwrap() {
            value[key] = field.clone();
        }
        value
    }

    #[test]
    fn loads_roberta_family_namespaces_and_canonical_layers() {
        for (model_type, architecture, root, family) in [
            ("roberta", "RobertaModel", "", EncoderFamily::Roberta),
            (
                "roberta",
                "RobertaForMaskedLM",
                "roberta.",
                EncoderFamily::Roberta,
            ),
            (
                "xlm-roberta",
                "XLMRobertaModel",
                "xlm-roberta.",
                EncoderFamily::XlmRoberta,
            ),
            (
                "camembert",
                "CamembertForMaskedLM",
                "camembert.",
                EncoderFamily::Camembert,
            ),
        ] {
            let fixture = Fixture::new();
            fixture.raw_config(roberta_config(model_type, architecture));
            let config = fixture.config_from_disk();
            assert_eq!(config.family, family);
            assert_eq!(
                config.recipe.embedding,
                NormArithmetic::RoundedAddBeforeStats
            );
            let layout = CheckpointLayout {
                family,
                root,
                layer_prefix: "encoder.layer",
            };
            fixture.write_layout(&config, layout, Dtype::F32, false, fixture_values);
            let weights = BertWeights::load(&fixture.0).unwrap();
            assert_eq!(weights.config.family, family);
            assert_eq!(weights.config.type_vocab_size, 1);
            assert_eq!(weights.config.layer_norm_eps, 1e-5);
            assert_eq!(
                read_all(&weights.word),
                expected(&(1..=8).map(|value| value as f32 / 16.0).collect::<Vec<_>>())
            );
            assert_eq!(
                read_all(&weights.layers[0].query.weight),
                expected(&[0.0625, 0.125, 0.1875, 0.25])
            );
            assert_eq!(
                read_all(&weights.layers[0].output_gamma),
                expected(&[1.0; 2])
            );
        }
    }

    #[test]
    fn loads_extended_encoder_position_tables_without_checkpoint_specific_caps() {
        for (value, layout) in [
            (
                roberta_config("xlm-roberta", "XLMRobertaModel"),
                CheckpointLayout {
                    family: EncoderFamily::XlmRoberta,
                    root: "xlm-roberta.",
                    layer_prefix: "encoder.layer",
                },
            ),
            (
                distil_config(),
                CheckpointLayout {
                    family: EncoderFamily::DistilBert,
                    root: "distilbert.",
                    layer_prefix: "transformer.layer",
                },
            ),
        ] {
            for positions in [8194usize, 8195, 16386] {
                let fixture = Fixture::new();
                fixture.raw_config(with_fields(
                    value.clone(),
                    serde_json::json!({"max_position_embeddings": positions}),
                ));
                let config = fixture.config_from_disk();
                fixture.write_layout(&config, layout, Dtype::F32, false, |_, index| {
                    (index % 7) as f32
                });
                let weights = BertWeights::load(&fixture.0).unwrap();
                assert_eq!(weights.position.shape(), &[positions, 2]);
                let mut tail = [0u16; 4];
                let end = positions * 2;
                weights.position.read_bf16(end - 4, &mut tail).unwrap();
                assert_eq!(
                    tail.to_vec(),
                    (end - 4..end)
                        .map(|index| bf16((index % 7) as f32))
                        .collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn rejects_encoder_position_tables_outside_cuda_lookup_dimensions() {
        for value in [roberta_config("roberta", "RobertaModel"), distil_config()] {
            let fixture = Fixture::new();
            fixture.raw_config(with_fields(
                value.clone(),
                serde_json::json!({"max_position_embeddings": i32::MAX as usize / 2 + 1}),
            ));
            let error = BertConfig::load(&fixture.0).unwrap_err().to_string();
            assert!(error.contains("CUDA indexing range"), "{error}");
            fixture.raw_config(with_fields(
                value,
                serde_json::json!({"max_position_embeddings": u32::MAX as u64 + 1}),
            ));
            let error = BertConfig::load(&fixture.0).unwrap_err().to_string();
            assert!(error.contains("invalid or unsupported"), "{error}");
        }
    }

    #[test]
    fn defaults_roberta_family_configs_to_reference_norm_and_type_vocabulary() {
        for (model_type, architecture) in [
            ("roberta", "RobertaModel"),
            ("xlm-roberta", "XLMRobertaModel"),
            ("camembert", "CamembertModel"),
        ] {
            let fixture = Fixture::new();
            let mut value = roberta_config(model_type, architecture);
            value.as_object_mut().unwrap().remove("type_vocab_size");
            value.as_object_mut().unwrap().remove("layer_norm_eps");
            fixture.raw_config(value);
            let config = fixture.config_from_disk();
            let layout = CheckpointLayout {
                family: config.family,
                root: "roberta.",
                layer_prefix: "encoder.layer",
            };
            fixture.write_layout(&config, layout, Dtype::F32, false, fixture_values);
            let weights = BertWeights::load(&fixture.0).unwrap();
            assert_eq!(weights.config.type_vocab_size, 2);
            assert_eq!(weights.config.layer_norm_eps, 1e-12);
            assert_eq!(weights.token_type.shape(), &[2, 2]);
        }
    }

    #[test]
    fn loads_distilbert_namespaces_aliases_and_zero_token_types() {
        let omitted = ["embeddings.token_type_embeddings.weight"];
        for root in ["", "distilbert."] {
            for layer_prefix in ["transformer.layer", "encoder.layer"] {
                let fixture = Fixture::new();
                fixture.raw_config(distil_config());
                let config = fixture.config_from_disk();
                assert_eq!(config.family, EncoderFamily::DistilBert);
                assert_eq!(config.layer_norm_eps, 1e-12);
                assert_eq!(config.type_vocab_size, 2);
                assert_eq!(
                    config.recipe,
                    EncoderRecipe {
                        embedding: NormArithmetic::RoundedAddBeforeStats,
                        residual: NormArithmetic::RoundedAddBeforeStats,
                    }
                );
                let layout = CheckpointLayout {
                    family: EncoderFamily::DistilBert,
                    root,
                    layer_prefix,
                };
                fixture.write_layout_omitting(
                    &config,
                    layout,
                    Dtype::F32,
                    false,
                    &omitted,
                    fixture_values,
                );
                let weights = BertWeights::load(&fixture.0).unwrap();
                assert_eq!(weights.token_type.shape(), &[2, 2]);
                assert!(read_all(&weights.token_type).iter().all(|bits| *bits == 0));
                assert_eq!(
                    read_all(&weights.word),
                    expected(&(1..=8).map(|value| value as f32 / 16.0).collect::<Vec<_>>())
                );
                assert_eq!(
                    read_all(&weights.layers[0].query.weight),
                    expected(&[0.0625, 0.125, 0.1875, 0.25])
                );
                assert_eq!(
                    read_all(&weights.layers[0].attention_gamma),
                    expected(&[1.0; 2])
                );
                assert_eq!(weights.layers[0].intermediate.weight.shape(), &[4, 2]);
            }
        }
    }

    #[test]
    fn validates_and_preserves_optional_distilbert_token_type_tables() {
        let layout = CheckpointLayout {
            family: EncoderFamily::DistilBert,
            root: "distilbert.",
            layer_prefix: "transformer.layer",
        };
        for rows in [1usize, 2, 3] {
            for explicit in [false, true] {
                let fixture = Fixture::new();
                let value = if explicit {
                    with_fields(
                        distil_config(),
                        serde_json::json!({"type_vocab_size": rows}),
                    )
                } else {
                    distil_config()
                };
                fixture.raw_config(value);
                let mut config = fixture.config_from_disk();
                config.type_vocab_size = rows;
                fixture.write_layout(&config, layout, Dtype::F32, rows == 3, fixture_values);
                let weights = BertWeights::load(&fixture.0).unwrap();
                assert_eq!(weights.config.type_vocab_size, rows);
                assert_eq!(weights.token_type.shape(), &[rows, 2]);
                assert_eq!(
                    read_all(&weights.token_type),
                    expected(
                        &(1..=rows * 2)
                            .map(|value| value as f32 / 16.0)
                            .collect::<Vec<_>>()
                    )
                );
            }
        }
    }

    #[test]
    fn rejects_invalid_or_config_mismatched_distilbert_token_type_tables() {
        let layout = CheckpointLayout {
            family: EncoderFamily::DistilBert,
            root: "distilbert.",
            layer_prefix: "transformer.layer",
        };
        for shape in [vec![0, 2], vec![3, 3], vec![3], vec![3, 2]] {
            let fixture = Fixture::new();
            fixture.raw_config(with_fields(
                distil_config(),
                serde_json::json!({"type_vocab_size": 2}),
            ));
            let config = fixture.config_from_disk();
            let entries = required(&config)
                .into_iter()
                .map(|(canonical, expected)| {
                    let actual_shape = if canonical == "embeddings.token_type_embeddings.weight" {
                        shape.clone()
                    } else {
                        expected
                    };
                    (layout.actual(&canonical).unwrap(), actual_shape, canonical)
                })
                .collect::<Vec<_>>();
            fixture.write_entries(Dtype::F32, false, &entries, fixture_values);
            let error = fixture.error();
            assert!(error.contains("wrong shape"), "{error}");
            if shape == [3, 2] {
                assert!(error.contains("configured type_vocab_size"), "{error}");
            }
        }
    }

    #[test]
    fn missing_distilbert_token_type_table_keeps_both_tokenizer_pair_ids_zero() {
        for configured_rows in [1, 3] {
            let fixture = Fixture::new();
            fixture.raw_config(with_fields(
                distil_config(),
                serde_json::json!({"type_vocab_size": configured_rows}),
            ));
            let config = fixture.config_from_disk();
            fixture.write_layout_omitting(
                &config,
                CheckpointLayout {
                    family: EncoderFamily::DistilBert,
                    root: "",
                    layer_prefix: "transformer.layer",
                },
                Dtype::F32,
                false,
                &["embeddings.token_type_embeddings.weight"],
                fixture_values,
            );
            let weights = BertWeights::load(&fixture.0).unwrap();
            assert_eq!(weights.config.type_vocab_size, 2);
            assert_eq!(weights.token_type.shape(), &[2, 2]);
            assert_eq!(read_all(&weights.token_type), vec![0; 4]);
        }
    }

    #[test]
    fn preserves_explicit_distilbert_sinusoidal_position_table() {
        let fixture = Fixture::new();
        fixture.raw_config(with_fields(
            distil_config(),
            serde_json::json!({"sinusoidal_pos_embds": true}),
        ));
        let config = fixture.config_from_disk();
        let layout = CheckpointLayout {
            family: EncoderFamily::DistilBert,
            root: "distilbert.",
            layer_prefix: "transformer.layer",
        };
        fixture.write_layout(&config, layout, Dtype::F32, false, fixture_values);
        let weights = BertWeights::load(&fixture.0).unwrap();
        assert_eq!(
            read_all(&weights.position),
            expected(
                &(1..=16)
                    .map(|value| value as f32 / 16.0)
                    .collect::<Vec<_>>()
            )
        );
        fixture.write_layout_omitting(
            &config,
            layout,
            Dtype::F32,
            false,
            &["embeddings.position_embeddings.weight"],
            fixture_values,
        );
        let error = fixture.error();
        assert!(
            error.contains("missing weight embeddings.position_embeddings.weight"),
            "{error}"
        );
    }

    #[test]
    fn rejects_incoherent_distilbert_token_type_namespace() {
        let fixture = Fixture::new();
        fixture.raw_config(distil_config());
        let config = fixture.config_from_disk();
        let layout = CheckpointLayout {
            family: EncoderFamily::DistilBert,
            root: "distilbert.",
            layer_prefix: "transformer.layer",
        };
        let mut entries = required(&config)
            .into_iter()
            .filter(|(canonical, _)| canonical != "embeddings.token_type_embeddings.weight")
            .map(|(canonical, shape)| (layout.actual(&canonical).unwrap(), shape, canonical))
            .collect::<Vec<_>>();
        entries.push((
            "embeddings.token_type_embeddings.weight".into(),
            vec![2, 2],
            "embeddings.token_type_embeddings.weight".into(),
        ));
        fixture.write_entries(Dtype::F32, false, &entries, fixture_values);
        let error = fixture.error();
        assert!(error.contains("incoherent token-type"), "{error}");
    }

    #[test]
    fn rejects_mixed_distilbert_layer_namespaces() {
        let fixture = Fixture::new();
        fixture.raw_config(distil_config());
        let config = fixture.config_from_disk();
        let transformer = CheckpointLayout {
            family: EncoderFamily::DistilBert,
            root: "distilbert.",
            layer_prefix: "transformer.layer",
        };
        let encoder = CheckpointLayout {
            family: EncoderFamily::DistilBert,
            root: "distilbert.",
            layer_prefix: "encoder.layer",
        };
        let canonical = "encoder.layer.0.attention.self.query.weight";
        let from = transformer.actual(canonical).unwrap();
        let to = encoder.actual(canonical).unwrap();
        let mut entries = required(&config)
            .into_iter()
            .map(|(canonical, shape)| (transformer.actual(&canonical).unwrap(), shape, canonical))
            .collect::<Vec<_>>();
        for (actual, _, _) in &mut entries {
            if *actual == from {
                *actual = to.clone();
            }
        }
        fixture.write_entries(Dtype::F32, false, &entries, fixture_values);
        let error = fixture.error();
        assert!(
            error.contains("ambiguous DistilBERT layer namespace"),
            "{error}"
        );
    }

    #[test]
    fn rejects_ambiguous_encoder_roots() {
        let fixture = Fixture::new();
        fixture.raw_config(roberta_config("roberta", "RobertaModel"));
        let config = fixture.config_from_disk();
        let layout = CheckpointLayout {
            family: EncoderFamily::Roberta,
            root: "roberta.",
            layer_prefix: "encoder.layer",
        };
        let mut entries = required(&config)
            .into_iter()
            .map(|(canonical, shape)| (layout.actual(&canonical).unwrap(), shape, canonical))
            .collect::<Vec<_>>();
        entries.push((
            "camembert.embeddings.word_embeddings.weight".into(),
            vec![4, 2],
            "embeddings.word_embeddings.weight".into(),
        ));
        fixture.write_entries(Dtype::F32, false, &entries, fixture_values);
        let error = fixture.error();
        assert!(error.contains("ambiguous checkpoint namespace"), "{error}");
    }

    #[test]
    fn rejects_unqualified_distilbert_config_forms() {
        for (fields, message) in [
            (
                serde_json::json!({"type_vocab_size": 0}),
                "invalid or unsupported",
            ),
            (
                serde_json::json!({"activation": "relu"}),
                "only encoder BERT",
            ),
            (
                serde_json::json!({"hidden_size": 4}),
                "ambiguous encoder config dim",
            ),
        ] {
            let fixture = Fixture::new();
            fixture.raw_config(with_fields(distil_config(), fields));
            let error = BertConfig::load(&fixture.0).unwrap_err().to_string();
            assert!(error.contains(message), "{error}");
        }
    }

    #[test]
    fn load_time_recipe_distinguishes_exact_biased_and_rounded_arithmetic() {
        for (value, expected) in [
            (
                serde_json::json!({"model_type":"bert", "hidden_size":768,
                    "intermediate_size":3072, "num_attention_heads":12,
                    "num_hidden_layers":1, "max_position_embeddings":512, "vocab_size":4}),
                EncoderRecipe {
                    embedding: NormArithmetic::ExactBiased,
                    residual: NormArithmetic::ExactBiased,
                },
            ),
            (
                serde_json::json!({"model_type":"bert", "hidden_size":1024,
                    "intermediate_size":4096, "num_attention_heads":16,
                    "num_hidden_layers":1, "max_position_embeddings":512, "vocab_size":4}),
                EncoderRecipe {
                    embedding: NormArithmetic::RoundedAddBeforeStats,
                    residual: NormArithmetic::RoundedAddBeforeStats,
                },
            ),
            (
                serde_json::json!({"model_type":"roberta", "hidden_size":768,
                    "intermediate_size":3072, "num_attention_heads":12,
                    "num_hidden_layers":1, "max_position_embeddings":512, "vocab_size":4}),
                EncoderRecipe {
                    embedding: NormArithmetic::ExactBiased,
                    residual: NormArithmetic::ExactBiased,
                },
            ),
            (
                serde_json::json!({"model_type":"roberta", "hidden_size":1024,
                    "intermediate_size":4096, "num_attention_heads":16,
                    "num_hidden_layers":1, "max_position_embeddings":512, "vocab_size":4}),
                EncoderRecipe {
                    embedding: NormArithmetic::RoundedAddBeforeStats,
                    residual: NormArithmetic::RoundedAddBeforeStats,
                },
            ),
            (
                serde_json::json!({"model_type":"distilbert", "dim":768,
                    "hidden_dim":3072, "n_heads":12, "n_layers":1,
                    "max_position_embeddings":512, "vocab_size":4,
                    "activation":"gelu", "sinusoidal_pos_embds":false}),
                EncoderRecipe {
                    embedding: NormArithmetic::RoundedAddBeforeStats,
                    residual: NormArithmetic::RoundedAddBeforeStats,
                },
            ),
        ] {
            let fixture = Fixture::new();
            fixture.raw_config(value);
            assert_eq!(fixture.config_from_disk().recipe, expected);
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
