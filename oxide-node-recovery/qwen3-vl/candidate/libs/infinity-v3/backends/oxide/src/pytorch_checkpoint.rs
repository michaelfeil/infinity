// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Bounded PyTorch `pytorch_model.bin` metadata and range-backed weight sources.
//!
//! This adapter uses the vendored bounded reader to parse the pickle metadata only. It
//! never materializes a whole checkpoint or tensor: `PyTorchWeightSource::read_bf16` asks
//! for at most one 64 KiB logical range at a time, and the reader caps every storage window
//! it performs independently.
//!
//! # CRC publication policy
//!
//! Range reads are unverified. Before the first byte of a retained tensor is handed to the
//! bounded uploader, this adapter verifies the whole backing storage entry once for that
//! retained tensor handle (clones share the same state). `Verified` and `NoChecksum` may be
//! published. Legacy entries have no CRC in the format and are reported internally as
//! `NoChecksum`; TAR entries are likewise accepted as `NoChecksum` because the reader holds
//! the archive in memory at open and has no per-entry CRC. Neither claims verification.
//! Compressed ZIP entries are rejected for backend publication with a typed load error: the
//! frozen reader's deflated range access re-inflates from the entry start for every bounded
//! window, which would make cold load quadratic. Real `torch.save` files use stored entries.
//! Any other unsupported verification is likewise a startup error, not a silent fallback.
//!
//! TAR is accepted for compatibility with the existing reader, but the reader holds the whole
//! TAR archive in memory at open. Ranges then slice that in-memory archive, so archive size is
//! not bounded even though each requested range is. TAR has no per-entry CRC here and is
//! reported as `NoChecksum`; it is never claimed verified.

#[cfg(test)]
use crate::model::to_bf16;
use crate::model::{
    convert_16, elements, permitted_cache_blob, permitted_weight, start, Bf16Target, F16Target,
    Source16, Target16,
};
use infinity_v3_backend_core::BackendError;
use pytorch_reader::{
    ChecksumLimitation, ChecksumStatus, DType as TorchDType, PytorchReader, Tensor,
};
use safetensors::Dtype;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::{Component, Path};
use std::sync::{Arc, OnceLock};

const READ_SCRATCH_BYTES: usize = 64 * 1024;
const PYTORCH_SINGLE: &str = "pytorch_model.bin";
const PYTORCH_INDEX: &str = "pytorch_model.bin.index.json";
const SAFETENSORS_SINGLE: &str = "model.safetensors";
const SAFETENSORS_INDEX: &str = "model.safetensors.index.json";

type CheckpointFiles = (Vec<String>, Option<BTreeMap<String, String>>);

/// Cold-load checkpoint encoding. Only loaders branch on it; no inference path sees it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DataFormat {
    Safetensors,
    PyTorch,
}

/// Prefer the native/Hugging Face safetensors layout, then fall back to PyTorch `.bin`.
pub(crate) fn detect_format(path: &Path) -> Result<DataFormat, BackendError> {
    if crate::model::path_present(&path.join(SAFETENSORS_SINGLE))?
        || crate::model::path_present(&path.join(SAFETENSORS_INDEX))?
    {
        return Ok(DataFormat::Safetensors);
    }
    if crate::model::path_present(&path.join(PYTORCH_SINGLE))?
        || crate::model::path_present(&path.join(PYTORCH_INDEX))?
    {
        return Ok(DataFormat::PyTorch);
    }
    Err(start(format!(
        "checkpoint not found: expected {SAFETENSORS_SINGLE}, {SAFETENSORS_INDEX}, {PYTORCH_SINGLE}, or {PYTORCH_INDEX}"
    )))
}

/// A checksum outcome that may be published for a loaded tensor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PublishedChecksum {
    /// The whole backing entry was hashed and its declared CRC matched.
    Verified,
    /// The format carries no CRC, or the entry declares CRC 0 after structural validation.
    NoChecksum,
}

/// A validated PyTorch checkpoint: metadata only, plus lazy range readers per tensor.
#[derive(Debug)]
pub(crate) struct PyTorchCheckpoint {
    files: Vec<String>,
    index: Option<BTreeMap<String, String>>,
    tensors: BTreeMap<String, (usize, Tensor)>,
}

impl PyTorchCheckpoint {
    pub(crate) fn open(path: &Path) -> Result<Self, BackendError> {
        let root = path.canonicalize().map_err(start)?;
        Self::open_shards(&root, |resolved| permitted_weight(&root, resolved))
    }

    /// Open a nested module checkpoint while keeping the enclosing model root as
    /// the trusted Hub snapshot boundary.
    ///
    /// The module must canonicalize inside the model directory. Its shards may
    /// either stay inside the module or use the model snapshot's exact
    /// same-repository cache blob exception; symlinks to unrelated model files,
    /// other repositories, or arbitrary ancestors are rejected. Dense module
    /// admission also fails closed when both a single file and its index exist;
    /// backbone single-file preference is unchanged.
    pub(crate) fn open_module(model_root: &Path, module_root: &Path) -> Result<Self, BackendError> {
        let model_root = model_root.canonicalize().map_err(start)?;
        let module_root = module_root.canonicalize().map_err(start)?;
        if !module_root.starts_with(&model_root) {
            return Err(start("Dense module path escapes the model directory"));
        }
        if crate::model::path_present(&module_root.join(PYTORCH_SINGLE))?
            && crate::model::path_present(&module_root.join(PYTORCH_INDEX))?
        {
            return Err(start(
                "ambiguous PyTorch checkpoint: both pytorch_model.bin and pytorch_model.bin.index.json are present",
            ));
        }
        Self::open_shards(&module_root, |resolved| {
            resolved.starts_with(&module_root) || permitted_cache_blob(&model_root, resolved)
        })
    }

    fn open_shards(root: &Path, permit: impl Fn(&Path) -> bool) -> Result<Self, BackendError> {
        let (files, index) = checkpoint_files(root)?;
        let mut tensors = BTreeMap::new();
        let mut readers = Vec::with_capacity(files.len());
        for file in &files {
            let resolved = root.join(file).canonicalize().map_err(start)?;
            if !permit(&resolved) {
                return Err(start("checkpoint shard escapes model directory"));
            }
            let reader = open_reader(&resolved, file)?;
            for (name, tensor) in reader.tensors() {
                if tensors
                    .insert(name.clone(), (readers.len(), tensor.clone()))
                    .is_some()
                {
                    return Err(start(format!("duplicate checkpoint tensor {name}")));
                }
            }
            readers.push(reader);
        }
        if let Some(index) = &index {
            for (name, file) in index {
                let Some((shard, _)) = tensors.get(name) else {
                    return Err(start(format!(
                        "indexed PyTorch tensor {name} is missing from {file}"
                    )));
                };
                if files.get(*shard) != Some(file) {
                    return Err(start(format!("weight {name} is not in its indexed shard")));
                }
            }
        }
        Ok(Self {
            files,
            index,
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
        self.tensors.get(name).map(|(_, tensor)| tensor.shape())
    }

    #[cfg(test)]
    pub(crate) fn source(
        &self,
        name: &str,
        shape: &[usize],
    ) -> Result<PyTorchWeightSource, BackendError> {
        self.optional_source(name, shape)?
            .ok_or_else(|| start(format!("missing weight {name}")))
    }

    pub(crate) fn optional_source(
        &self,
        name: &str,
        shape: &[usize],
    ) -> Result<Option<PyTorchWeightSource>, BackendError> {
        let Some((shard, tensor)) = self.tensors.get(name) else {
            return Ok(None);
        };
        let Some(file) = self.files.get(*shard) else {
            return Err(start(format!("invalid shard ownership for {name}")));
        };
        if self
            .index
            .as_ref()
            .is_some_and(|index| index.get(name) != Some(file))
        {
            return Err(start(format!("weight {name} is not in its indexed shard")));
        }
        if tensor.shape() != shape {
            return Err(start(format!(
                "wrong shape for {name}: expected {shape:?}, got {:?}",
                tensor.shape()
            )));
        }
        Ok(Some(PyTorchWeightSource::new(name, tensor.clone())?))
    }
}

fn checkpoint_files(path: &Path) -> Result<CheckpointFiles, BackendError> {
    if crate::model::path_present(&path.join(PYTORCH_SINGLE))? {
        return Ok((vec![PYTORCH_SINGLE.into()], None));
    }
    #[derive(Deserialize)]
    struct Index {
        weight_map: BTreeMap<String, String>,
    }
    let index: Index = crate::model::read_json(&path.join(PYTORCH_INDEX))?;
    if index.weight_map.is_empty() {
        return Err(start("empty PyTorch shard index"));
    }
    let mut files = std::collections::BTreeSet::new();
    for file in index.weight_map.values() {
        let mut components = Path::new(file).components();
        if !matches!(components.next(), Some(Component::Normal(_)))
            || components.next().is_some()
            || !file.ends_with(".bin")
            || file.contains('\\')
        {
            return Err(start("invalid PyTorch shard path"));
        }
        files.insert(file.clone());
    }
    Ok((files.into_iter().collect(), Some(index.weight_map)))
}

/// Open one shard, selecting a conventional top-level state-dict wrapper if present.
fn open_reader(path: &Path, file: &str) -> Result<PytorchReader, BackendError> {
    let reader = PytorchReader::new(path)
        .map_err(|error| start(format!("PyTorch shard {file}: {error}")))?;
    if reader.is_empty() {
        for key in ["state_dict", "model_state_dict", "model"] {
            if let Ok(wrapped) = PytorchReader::with_top_level_key(path, key) {
                if !wrapped.is_empty() {
                    return Ok(wrapped);
                }
            }
        }
        return Ok(reader);
    }
    let keys = reader
        .tensors()
        .keys()
        .map(String::as_str)
        .collect::<Vec<_>>();
    let wrapper = if keys.iter().all(|name| name.starts_with("state_dict.")) {
        Some("state_dict")
    } else if keys
        .iter()
        .all(|name| name.starts_with("model_state_dict."))
    {
        Some("model_state_dict")
    } else {
        None
    };
    if let Some(key) = wrapper {
        if let Ok(wrapped) = PytorchReader::with_top_level_key(path, key) {
            if !wrapped.is_empty() {
                return Ok(wrapped);
            }
        }
    }
    Ok(reader)
}

/// A retained tensor handle whose whole backing entry is verified once on first use.
#[derive(Clone, Debug)]
pub(crate) struct PyTorchWeightSource {
    tensor: Tensor,
    dtype: TorchDType,
    count: usize,
    width: usize,
    shape: Vec<usize>,
    base_element: usize,
    verification: Arc<OnceLock<Result<PublishedChecksum, String>>>,
}

impl PyTorchWeightSource {
    fn new(name: &str, tensor: Tensor) -> Result<Self, BackendError> {
        let dtype = tensor.dtype();
        let width = match dtype {
            TorchDType::F32 => 4,
            TorchDType::F16 | TorchDType::BF16 => 2,
            other => {
                return Err(start(format!(
                    "unsupported PyTorch weight dtype {other:?} for {name}"
                )))
            }
        };
        let count = tensor.num_elements();
        count
            .checked_mul(width)
            .ok_or_else(|| start(format!("weight {name} byte size overflow")))?;
        let shape = tensor.shape().to_vec();
        Ok(Self {
            tensor,
            dtype,
            count,
            width,
            shape,
            base_element: 0,
            verification: Arc::new(OnceLock::new()),
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.count
    }

    pub(crate) fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub(crate) fn width(&self) -> usize {
        self.width
    }

    /// View a contiguous logical slice without copying or materializing payload.
    pub(crate) fn slice(
        &self,
        offset_elements: usize,
        shape: Vec<usize>,
    ) -> Result<Self, BackendError> {
        let count = elements(&shape)?;
        let end = offset_elements
            .checked_add(count)
            .ok_or_else(|| start("weight slice range overflow"))?;
        if end > self.count {
            return Err(start("weight slice exceeds source tensor"));
        }
        let base_element = self
            .base_element
            .checked_add(offset_elements)
            .ok_or_else(|| start("weight slice base overflow"))?;
        Ok(Self {
            tensor: self.tensor.clone(),
            dtype: self.dtype,
            count,
            width: self.width,
            shape,
            base_element,
            verification: Arc::clone(&self.verification),
        })
    }

    pub(crate) fn safetensors_dtype(&self) -> Dtype {
        match self.dtype {
            TorchDType::F32 => Dtype::F32,
            TorchDType::F16 => Dtype::F16,
            TorchDType::BF16 => Dtype::BF16,
            _ => Dtype::BF16,
        }
    }

    /// Verify the whole backing entry once for this retained tensor handle.
    pub(crate) fn verify(&self) -> Result<PublishedChecksum, BackendError> {
        match self
            .verification
            .get_or_init(|| publication_checksum(&self.tensor))
        {
            Ok(status) => Ok(*status),
            Err(message) => Err(start(message.clone())),
        }
    }

    pub(crate) fn read_bf16(
        &self,
        offset_elements: usize,
        out: &mut [u16],
    ) -> Result<(), BackendError> {
        self.read_target::<Bf16Target, true>(offset_elements, out)
    }

    pub(crate) fn read_f16(
        &self,
        offset_elements: usize,
        out: &mut [u16],
    ) -> Result<(), BackendError> {
        self.read_target::<F16Target, true>(offset_elements, out)
    }

    pub(crate) fn read_target<T: Target16, const NATIVE: bool>(
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
        // Publication policy: no byte is returned to the uploader before the whole backing
        // entry has been structurally validated and, when declared, CRC-checked.
        self.verify()?;
        let per_read = READ_SCRATCH_BYTES / self.width;
        if per_read == 0 {
            return Err(start("invalid weight element width"));
        }
        let source = source16(self.dtype)?;
        let mut done = 0usize;
        while done < out.len() {
            let take = (out.len() - done).min(per_read);
            let range_start = self
                .base_element
                .checked_add(offset_elements)
                .and_then(|base| base.checked_add(done))
                .ok_or_else(|| start("weight read range overflow"))?;
            let bytes = self
                .tensor
                .read_range(range_start, take)
                .map_err(|error| start(format!("PyTorch weight range read failed: {error}")))?;
            let end = done
                .checked_add(take)
                .ok_or_else(|| start("weight output range overflow"))?;
            let target = out
                .get_mut(done..end)
                .ok_or_else(|| start("weight output range error"))?;
            convert_16::<T, NATIVE>(source, &bytes, target)?;
            done = end;
        }
        Ok(())
    }
}

fn source16(dtype: TorchDType) -> Result<Source16, BackendError> {
    match dtype {
        TorchDType::F32 => Ok(Source16::F32),
        TorchDType::F16 => Ok(Source16::F16),
        TorchDType::BF16 => Ok(Source16::Bf16),
        other => Err(start(format!("unsupported weight dtype {other:?}"))),
    }
}

fn publication_checksum(tensor: &Tensor) -> Result<PublishedChecksum, String> {
    match tensor.verify_storage() {
        Ok(ChecksumStatus::Verified) => Ok(PublishedChecksum::Verified),
        Ok(ChecksumStatus::NoChecksum) => Ok(PublishedChecksum::NoChecksum),
        // Neither container carries a per-entry CRC. Accept them as an explicit internal
        // NoChecksum outcome; never call them verified.
        Ok(ChecksumStatus::Unsupported(ChecksumLimitation::LegacyFormat))
        | Ok(ChecksumStatus::Unsupported(ChecksumLimitation::TarFormat)) => {
            Ok(PublishedChecksum::NoChecksum)
        }
        Ok(ChecksumStatus::Unsupported(ChecksumLimitation::CompressedZipEntry)) => Err(
            "compressed ZIP entries are unsupported for bounded backend publication; \
             use a stored ZIP checkpoint (the standard torch.save layout)"
                .to_string(),
        ),
        Ok(ChecksumStatus::Unsupported(_)) | Ok(_) => {
            Err("unsupported PyTorch storage verification".to_string())
        }
        Err(error) => Err(format!("PyTorch storage verification failed: {error}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder_model::{decoder_tensor_shapes, DecoderConfig, DecoderWeights};
    use crate::model::{required, BertConfig, BertWeights, Checkpoint};
    use crate::modern_encoder_model::{ModernConfig, ModernWeights};
    use safetensors::tensor::{serialize, TensorView};
    use std::collections::{BTreeMap, BTreeSet};
    use std::fs::File;
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    #[derive(Clone)]
    struct TensorSpec {
        name: String,
        dtype: TorchDType,
        shape: Vec<usize>,
        values: Vec<f32>,
    }

    #[derive(Clone)]
    struct StorageSpec {
        dtype: TorchDType,
        bytes: Vec<u8>,
        elements: usize,
    }

    #[derive(Clone)]
    struct EntrySpec {
        name: String,
        storage: usize,
        offset: usize,
        shape: Vec<usize>,
        stride: Vec<usize>,
    }

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "infinity-oxide-pytorch-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn width(dtype: TorchDType) -> usize {
        match dtype {
            TorchDType::F32 => 4,
            TorchDType::F16 | TorchDType::BF16 => 2,
            TorchDType::I64 => 8,
            other => panic!("test fixture has no width for {other:?}"),
        }
    }

    fn encode(dtype: TorchDType, values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * width(dtype));
        for &value in values {
            match dtype {
                TorchDType::F32 => bytes.extend_from_slice(&value.to_le_bytes()),
                TorchDType::F16 => {
                    bytes.extend_from_slice(&half::f16::from_f32(value).to_bits().to_le_bytes())
                }
                TorchDType::BF16 => {
                    bytes.extend_from_slice(&half::bf16::from_f32(value).to_bits().to_le_bytes())
                }
                TorchDType::I64 => bytes.extend_from_slice(&(value as i64).to_le_bytes()),
                other => panic!("test fixture has no encoding for {other:?}"),
            }
        }
        bytes
    }

    fn row_major_stride(shape: &[usize]) -> Vec<usize> {
        let mut stride = vec![0usize; shape.len()];
        let mut product = 1usize;
        for index in (0..shape.len()).rev() {
            stride[index] = product;
            product *= shape[index];
        }
        stride
    }

    fn element_count(shape: &[usize]) -> usize {
        shape.iter().product()
    }

    fn value(name: &str, index: usize) -> f32 {
        ((index * 17 + name.len() * 13) % 251) as f32 / 17.0 - 7.0
    }

    fn storage_index(shape: &[usize], stride: &[usize], offset: usize, mut linear: usize) -> usize {
        let mut index = offset;
        for (&dim, &step) in shape.iter().zip(stride).rev() {
            index += (linear % dim) * step;
            linear /= dim;
        }
        index
    }

    fn write_bert_config(dir: &Path) {
        std::fs::write(
            dir.join("config.json"),
            serde_json::to_vec(&serde_json::json!({
                "model_type": "bert",
                "hidden_size": 2,
                "intermediate_size": 4,
                "num_attention_heads": 1,
                "num_hidden_layers": 1,
                "max_position_embeddings": 8,
                "vocab_size": 4,
                "type_vocab_size": 2
            }))
            .unwrap(),
        )
        .unwrap();
    }

    fn write_gte_config(dir: &Path) {
        std::fs::write(
            dir.join("config.json"),
            serde_json::to_vec(&serde_json::json!({
                "model_type": "new",
                "architectures": ["NewModel"],
                "hidden_size": 768,
                "intermediate_size": 3072,
                "num_attention_heads": 12,
                "num_hidden_layers": 1,
                "max_position_embeddings": 16,
                "vocab_size": 2,
                "layer_norm_type": "layer_norm",
                "pack_qkv": true,
                "position_embedding_type": "rope",
                "hidden_act": "gelu"
            }))
            .unwrap(),
        )
        .unwrap();
    }

    fn write_modern_config(dir: &Path) {
        std::fs::write(
            dir.join("config.json"),
            serde_json::to_vec(&serde_json::json!({
                "model_type": "modernbert",
                "architectures": ["ModernBertForMaskedLM"],
                "hidden_size": 768,
                "intermediate_size": 8,
                "num_attention_heads": 12,
                "num_hidden_layers": 1,
                "max_position_embeddings": 16,
                "vocab_size": 2,
                "norm_eps": 1e-5,
                "layer_norm_eps": 1e-5,
                "hidden_activation": "gelu_pytorch_tanh",
                "attention_bias": false,
                "mlp_bias": false,
                "norm_bias": false,
                "global_attn_every_n_layers": 3,
                "local_attention": 128,
                "global_rope_theta": 160000.0,
                "local_rope_theta": 10000.0,
                "position_embedding_type": "absolute"
            }))
            .unwrap(),
        )
        .unwrap();
    }

    #[derive(Clone)]
    struct RawSpec {
        name: String,
        dtype: TorchDType,
        shape: Vec<usize>,
        bytes: Vec<u8>,
    }

    fn raw_zero(name: &str, dtype: TorchDType, shape: Vec<usize>) -> RawSpec {
        RawSpec {
            name: name.into(),
            dtype,
            bytes: vec![0u8; element_count(&shape) * width(dtype)],
            shape,
        }
    }

    fn f32_rows(
        name: &str,
        rows: usize,
        cols: usize,
        value: impl Fn(usize, usize) -> f32,
    ) -> RawSpec {
        let mut bytes = Vec::with_capacity(rows * cols * 4);
        for row in 0..rows {
            for column in 0..cols {
                bytes.extend_from_slice(&value(row, column).to_le_bytes());
            }
        }
        RawSpec {
            name: name.into(),
            dtype: TorchDType::F32,
            shape: vec![rows, cols],
            bytes,
        }
    }

    fn raw_f32_row(name: &str, values: impl IntoIterator<Item = f32>) -> RawSpec {
        let values = values.into_iter().collect::<Vec<_>>();
        RawSpec {
            name: name.into(),
            dtype: TorchDType::F32,
            shape: vec![values.len()],
            bytes: values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
        }
    }

    fn raw_contiguous_parts(specs: &[RawSpec]) -> (Vec<StorageSpec>, Vec<EntrySpec>) {
        let storages = specs
            .iter()
            .map(|spec| StorageSpec {
                dtype: spec.dtype,
                bytes: spec.bytes.clone(),
                elements: element_count(&spec.shape),
            })
            .collect::<Vec<_>>();
        let entries = specs
            .iter()
            .enumerate()
            .map(|(index, spec)| EntrySpec {
                name: spec.name.clone(),
                storage: index,
                offset: 0,
                shape: spec.shape.clone(),
                stride: row_major_stride(&spec.shape),
            })
            .collect::<Vec<_>>();
        (storages, entries)
    }

    fn write_bin_sharded_raw(dir: &Path, files: [&str; 2], specs: &[RawSpec], split: usize) {
        let (first, second) = specs.split_at(split);
        let mut weight_map = BTreeMap::new();
        for (file, chunk) in files.iter().zip([first, second]) {
            let (storages, entries) = raw_contiguous_parts(chunk);
            write_bin(
                dir,
                file,
                &storages,
                &entries,
                zip::CompressionMethod::Stored,
            );
            for spec in chunk {
                weight_map.insert(spec.name.clone(), (*file).to_string());
            }
        }
        std::fs::write(
            dir.join("pytorch_model.bin.index.json"),
            serde_json::to_vec(&serde_json::json!({"weight_map": weight_map})).unwrap(),
        )
        .unwrap();
    }

    fn assert_slice(source: &crate::model::WeightSource, offset: usize, expected: &[f32]) {
        let mut actual = vec![0u16; expected.len()];
        source.read_bf16(offset, &mut actual).unwrap();
        let expected = expected
            .iter()
            .map(|value| half::bf16::from_f32(*value).to_bits())
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    fn bert_specs(dir: &Path, dtype: TorchDType) -> Vec<TensorSpec> {
        let config = BertConfig::load(dir).unwrap();
        let mut required = required(&config).into_iter().collect::<Vec<_>>();
        required.sort_by(|left, right| left.0.cmp(&right.0));
        required
            .into_iter()
            .map(|(name, shape)| {
                let values = (0..element_count(&shape))
                    .map(|index| value(&name, index))
                    .collect();
                TensorSpec {
                    name,
                    dtype,
                    shape,
                    values,
                }
            })
            .collect()
    }

    fn write_safetensors(dir: &Path, specs: &[TensorSpec]) {
        let payloads = specs
            .iter()
            .map(|spec| (spec.name.clone(), encode(spec.dtype, &spec.values)))
            .collect::<Vec<_>>();
        let views = specs
            .iter()
            .zip(&payloads)
            .map(|(spec, (name, bytes))| {
                let dtype = match spec.dtype {
                    TorchDType::F32 => Dtype::F32,
                    TorchDType::F16 => Dtype::F16,
                    TorchDType::BF16 => Dtype::BF16,
                    other => panic!("no safetensors dtype for {other:?}"),
                };
                (
                    name.clone(),
                    TensorView::new(dtype, spec.shape.clone(), bytes).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        std::fs::write(
            dir.join("model.safetensors"),
            serialize(views, None).unwrap(),
        )
        .unwrap();
    }

    fn put_unicode(out: &mut Vec<u8>, value: &str) {
        out.push(0x58);
        out.extend_from_slice(&u32::try_from(value.len()).unwrap().to_le_bytes());
        out.extend_from_slice(value.as_bytes());
    }

    fn put_int(out: &mut Vec<u8>, value: usize) {
        if let Ok(small) = u8::try_from(value) {
            out.push(0x4b);
            out.push(small);
        } else {
            out.push(0x4a);
            out.extend_from_slice(&i32::try_from(value).unwrap().to_le_bytes());
        }
    }

    fn put_global(out: &mut Vec<u8>, module: &str, name: &str) {
        out.push(b'c');
        out.extend_from_slice(module.as_bytes());
        out.push(b'\n');
        out.extend_from_slice(name.as_bytes());
        out.push(b'\n');
    }

    fn put_tuple(out: &mut Vec<u8>, items: impl FnOnce(&mut Vec<u8>)) {
        out.push(0x28);
        items(out);
        out.push(0x74);
    }

    fn storage_class(dtype: TorchDType) -> &'static str {
        match dtype {
            TorchDType::F32 => "FloatStorage",
            TorchDType::F16 => "HalfStorage",
            TorchDType::BF16 => "BFloat16Storage",
            TorchDType::I64 => "LongStorage",
            other => panic!("test fixture has no storage class for {other:?}"),
        }
    }

    fn data_pickle(entries: &[EntrySpec], storages: &[StorageSpec]) -> Vec<u8> {
        let mut out = vec![0x80, 0x02, b'}'];
        for entry in entries {
            let storage = &storages[entry.storage];
            put_unicode(&mut out, &entry.name);
            put_global(&mut out, "torch._utils", "_rebuild_tensor_v2");
            out.push(0x28);
            put_tuple(&mut out, |out| {
                put_unicode(out, "storage");
                put_global(out, "torch", storage_class(storage.dtype));
                put_unicode(out, &entry.storage.to_string());
                put_unicode(out, "cpu");
                put_int(out, storage.elements);
            });
            out.push(b'Q');
            put_int(&mut out, entry.offset);
            put_tuple(&mut out, |out| {
                for &dim in &entry.shape {
                    put_int(out, dim);
                }
            });
            put_tuple(&mut out, |out| {
                for &step in &entry.stride {
                    put_int(out, step);
                }
            });
            out.push(0x89);
            out.push(b'N');
            out.push(0x74);
            out.push(0x52);
            out.push(0x73);
        }
        out.push(b'.');
        out
    }

    fn write_bin(
        dir: &Path,
        file: &str,
        storages: &[StorageSpec],
        entries: &[EntrySpec],
        compression: zip::CompressionMethod,
    ) {
        let output = File::create(dir.join(file)).unwrap();
        let mut writer = zip::ZipWriter::new(output);
        let options = zip::write::SimpleFileOptions::default().compression_method(compression);
        writer.start_file("data.pkl", options).unwrap();
        writer.write_all(&data_pickle(entries, storages)).unwrap();
        for (index, storage) in storages.iter().enumerate() {
            assert_eq!(storage.bytes.len(), storage.elements * width(storage.dtype));
            writer.start_file(format!("data/{index}"), options).unwrap();
            writer.write_all(&storage.bytes).unwrap();
        }
        for (name, value) in [
            ("version", "3"),
            ("byteorder", "little"),
            (".format_version", "1"),
        ] {
            writer.start_file(name, options).unwrap();
            writer.write_all(value.as_bytes()).unwrap();
        }
        writer.finish().unwrap();
    }

    fn contiguous_parts(specs: &[TensorSpec]) -> (Vec<StorageSpec>, Vec<EntrySpec>) {
        let storages = specs
            .iter()
            .map(|spec| StorageSpec {
                dtype: spec.dtype,
                bytes: encode(spec.dtype, &spec.values),
                elements: spec.values.len(),
            })
            .collect::<Vec<_>>();
        let entries = specs
            .iter()
            .enumerate()
            .map(|(index, spec)| EntrySpec {
                name: spec.name.clone(),
                storage: index,
                offset: 0,
                shape: spec.shape.clone(),
                stride: row_major_stride(&spec.shape),
            })
            .collect::<Vec<_>>();
        (storages, entries)
    }

    fn write_bin_single(dir: &Path, specs: &[TensorSpec], compression: zip::CompressionMethod) {
        let (storages, entries) = contiguous_parts(specs);
        write_bin(dir, "pytorch_model.bin", &storages, &entries, compression);
    }

    fn read_all(checkpoint: &Checkpoint, name: &str, shape: &[usize]) -> Vec<u16> {
        let source = checkpoint.source(name, shape).unwrap();
        let mut out = vec![0u16; source.len()];
        source.read_bf16(0, &mut out).unwrap();
        out
    }

    #[test]
    fn loads_synthetic_bert_bin_with_safetensors_equivalent_bits() {
        for dtype in [TorchDType::F32, TorchDType::F16, TorchDType::BF16] {
            let safe = Fixture::new();
            write_bert_config(&safe.0);
            let specs = bert_specs(&safe.0, dtype);
            write_safetensors(&safe.0, &specs);
            let bin = Fixture::new();
            write_bert_config(&bin.0);
            write_bin_single(&bin.0, &specs, zip::CompressionMethod::Stored);

            let safe_checkpoint = Checkpoint::open(&safe.0).unwrap();
            let bin_checkpoint = Checkpoint::open(&bin.0).unwrap();
            assert_eq!(bin_checkpoint.names().count(), specs.len());
            for spec in &specs {
                let expected = read_all(&safe_checkpoint, &spec.name, &spec.shape);
                let actual = read_all(&bin_checkpoint, &spec.name, &spec.shape);
                assert_eq!(actual, expected, "{} {dtype:?}", spec.name);
            }

            let weights = BertWeights::load(&bin.0).unwrap();
            assert_eq!(weights.layers.len(), 1);
            assert_eq!(weights.word.shape(), &[4, 2]);
            assert_eq!(
                weights.word.dtype(),
                bin_checkpoint
                    .source("embeddings.word_embeddings.weight", &[4, 2])
                    .unwrap()
                    .dtype()
            );
        }
    }

    #[test]
    fn prefers_safetensors_when_both_formats_are_present() {
        let dir = Fixture::new();
        write_bert_config(&dir.0);
        let specs = bert_specs(&dir.0, TorchDType::F32);
        write_safetensors(&dir.0, &specs);
        let mut altered = specs.clone();
        for spec in &mut altered {
            spec.values.fill(9.0);
        }
        write_bin_single(&dir.0, &altered, zip::CompressionMethod::Stored);

        let checkpoint = Checkpoint::open(&dir.0).unwrap();
        let word = specs
            .iter()
            .find(|spec| spec.name == "embeddings.word_embeddings.weight")
            .unwrap();
        let actual = read_all(
            &checkpoint,
            "embeddings.word_embeddings.weight",
            &word.shape,
        );
        assert_eq!(actual[0], to_bf16(word.values[0]).unwrap());
        assert_ne!(actual[0], to_bf16(9.0).unwrap());
    }

    #[test]
    fn backbone_single_bin_keeps_preference_over_stale_index() {
        let dir = Fixture::new();
        let specs = vec![TensorSpec {
            name: "weight".into(),
            dtype: TorchDType::F32,
            shape: vec![2],
            values: vec![1.0, 2.0],
        }];
        write_bin_single(&dir.0, &specs, zip::CompressionMethod::Stored);
        std::fs::write(
            dir.0.join("pytorch_model.bin.index.json"),
            serde_json::to_vec(&serde_json::json!({
                "weight_map": {"weight": "missing.bin"}
            }))
            .unwrap(),
        )
        .unwrap();

        let checkpoint = Checkpoint::open(&dir.0).unwrap();
        assert_eq!(
            read_all(&checkpoint, "weight", &[2]),
            [to_bf16(1.0).unwrap(), to_bf16(2.0).unwrap()]
        );
    }

    #[test]
    fn loads_indexed_bin_shards_and_rejects_invalid_indices() {
        let dir = Fixture::new();
        write_bert_config(&dir.0);
        let specs = bert_specs(&dir.0, TorchDType::F32);
        let split = specs.len() / 2;
        let files = [
            "pytorch_model-00001-of-00002.bin",
            "pytorch_model-00002-of-00002.bin",
        ];
        let mut weight_map = BTreeMap::new();
        let (first, second) = specs.split_at(split);
        for (file, chunk) in files.iter().zip([first, second]) {
            let (storages, entries) = contiguous_parts(chunk);
            write_bin(
                &dir.0,
                file,
                &storages,
                &entries,
                zip::CompressionMethod::Stored,
            );
            for spec in chunk {
                weight_map.insert(spec.name.clone(), (*file).to_string());
            }
        }
        std::fs::write(
            dir.0.join("pytorch_model.bin.index.json"),
            serde_json::to_vec(&serde_json::json!({"weight_map": weight_map})).unwrap(),
        )
        .unwrap();
        let checkpoint = Checkpoint::open(&dir.0).unwrap();
        for spec in &specs {
            let expected = spec
                .values
                .iter()
                .map(|value| to_bf16(*value).unwrap())
                .collect::<Vec<_>>();
            assert_eq!(
                read_all(&checkpoint, &spec.name, &spec.shape),
                expected,
                "{}",
                spec.name
            );
        }

        let bad = Fixture::new();
        std::fs::write(
            bad.0.join("pytorch_model.bin.index.json"),
            serde_json::to_vec(&serde_json::json!({
                "weight_map": {"x": "../outside.bin"}
            }))
            .unwrap(),
        )
        .unwrap();
        assert!(PyTorchCheckpoint::open(&bad.0)
            .unwrap_err()
            .to_string()
            .contains("invalid PyTorch shard path"));

        let missing = Fixture::new();
        let (storages, entries) = contiguous_parts(&specs[..1]);
        write_bin(
            &missing.0,
            files[0],
            &storages,
            &entries,
            zip::CompressionMethod::Stored,
        );
        std::fs::write(
            missing.0.join("pytorch_model.bin.index.json"),
            serde_json::to_vec(&serde_json::json!({
                "weight_map": {"ghost.weight": files[0]}
            }))
            .unwrap(),
        )
        .unwrap();
        assert!(PyTorchCheckpoint::open(&missing.0)
            .unwrap_err()
            .to_string()
            .contains("missing from"));

        let wrong = Fixture::new();
        let mut wrong_map = BTreeMap::new();
        for (file, chunk) in files.iter().zip([first, second]) {
            let (storages, entries) = contiguous_parts(chunk);
            write_bin(
                &wrong.0,
                file,
                &storages,
                &entries,
                zip::CompressionMethod::Stored,
            );
            for spec in chunk {
                wrong_map.insert(spec.name.clone(), (*file).to_string());
            }
        }
        wrong_map.insert(specs[0].name.clone(), files[1].to_string());
        std::fs::write(
            wrong.0.join("pytorch_model.bin.index.json"),
            serde_json::to_vec(&serde_json::json!({"weight_map": wrong_map})).unwrap(),
        )
        .unwrap();
        assert!(PyTorchCheckpoint::open(&wrong.0)
            .unwrap_err()
            .to_string()
            .contains("not in its indexed shard"));
    }

    #[test]
    fn skips_irrelevant_buffers_and_rejects_unsupported_requested_dtype() {
        let dir = Fixture::new();
        write_bert_config(&dir.0);
        let mut specs = bert_specs(&dir.0, TorchDType::F32);
        specs.push(TensorSpec {
            name: "position_ids".into(),
            dtype: TorchDType::I64,
            shape: vec![1, 8],
            values: (0..8).map(|index| index as f32).collect(),
        });
        write_bin_single(&dir.0, &specs, zip::CompressionMethod::Stored);
        let weights = BertWeights::load(&dir.0).unwrap();
        assert_eq!(weights.layers.len(), 1);
        let checkpoint = Checkpoint::open(&dir.0).unwrap();
        assert!(checkpoint.contains("position_ids"));
        let error = checkpoint
            .source("position_ids", &[1, 8])
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("unsupported PyTorch weight dtype"),
            "{error}"
        );
    }

    #[test]
    fn reads_noncontiguous_shared_storage_and_matches_contiguous_safetensors() {
        let storage_values = (0..64)
            .map(|index| index as f32 * 0.25 - 3.0)
            .collect::<Vec<_>>();
        let entries = vec![
            EntrySpec {
                name: "permuted".into(),
                storage: 0,
                offset: 5,
                shape: vec![2, 3, 4],
                stride: vec![12, 1, 4],
            },
            EntrySpec {
                name: "expanded".into(),
                storage: 0,
                offset: 1,
                shape: vec![2, 3],
                stride: vec![0, 1],
            },
        ];
        let storages = vec![StorageSpec {
            dtype: TorchDType::F32,
            bytes: encode(TorchDType::F32, &storage_values),
            elements: storage_values.len(),
        }];
        let bin = Fixture::new();
        write_bin(
            &bin.0,
            "pytorch_model.bin",
            &storages,
            &entries,
            zip::CompressionMethod::Stored,
        );
        let specs = entries
            .iter()
            .map(|entry| TensorSpec {
                name: entry.name.clone(),
                dtype: TorchDType::F32,
                shape: entry.shape.clone(),
                values: (0..element_count(&entry.shape))
                    .map(|linear| {
                        storage_values
                            [storage_index(&entry.shape, &entry.stride, entry.offset, linear)]
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();
        let safe = Fixture::new();
        write_safetensors(&safe.0, &specs);

        let bin_checkpoint = Checkpoint::open(&bin.0).unwrap();
        let safe_checkpoint = Checkpoint::open(&safe.0).unwrap();
        for spec in &specs {
            assert_eq!(
                read_all(&bin_checkpoint, &spec.name, &spec.shape),
                read_all(&safe_checkpoint, &spec.name, &spec.shape),
                "{}",
                spec.name
            );
        }
        let source = bin_checkpoint.source("permuted", &[2, 3, 4]).unwrap();
        let mut slice = [0u16; 5];
        source.read_bf16(3, &mut slice).unwrap();
        let expected = read_all(&safe_checkpoint, "permuted", &[2, 3, 4])[3..8].to_vec();
        assert_eq!(slice.to_vec(), expected);
    }

    #[test]
    fn reads_f16_from_sharded_native_views_offsets_and_rejects_nonfinite() {
        let dir = Fixture::new();
        let f16_raw = [0x0000u16, 0x8000, 0x0001, 0x03ff, 0x3c00, 0x7bff];
        let f32_values = [1.0f32 / 3.0, f32::from_bits(0x3f80_1000), -0.0];
        let bf16_raw = [0x3f80u16, 0xbf80, 0x3f00, 0x0000, 0x8000];
        let files = [
            "pytorch_model-00001-of-00002.bin",
            "pytorch_model-00002-of-00002.bin",
        ];
        let first_storages = vec![
            StorageSpec {
                dtype: TorchDType::F16,
                bytes: f16_raw.map(u16::to_ne_bytes).concat(),
                elements: f16_raw.len(),
            },
            StorageSpec {
                dtype: TorchDType::F32,
                bytes: f32_values.map(f32::to_ne_bytes).concat(),
                elements: f32_values.len(),
            },
        ];
        let first_entries = vec![
            EntrySpec {
                name: "f16".into(),
                storage: 0,
                offset: 1,
                shape: vec![5],
                stride: vec![1],
            },
            EntrySpec {
                name: "f16_view".into(),
                storage: 0,
                offset: 1,
                shape: vec![2, 2],
                stride: vec![3, 1],
            },
            EntrySpec {
                name: "f32".into(),
                storage: 1,
                offset: 0,
                shape: vec![3],
                stride: vec![1],
            },
        ];
        write_bin(
            &dir.0,
            files[0],
            &first_storages,
            &first_entries,
            zip::CompressionMethod::Stored,
        );

        let second_storages = vec![
            StorageSpec {
                dtype: TorchDType::BF16,
                bytes: bf16_raw.map(u16::to_ne_bytes).concat(),
                elements: bf16_raw.len(),
            },
            StorageSpec {
                dtype: TorchDType::F16,
                bytes: 0x7c00u16.to_ne_bytes().to_vec(),
                elements: 1,
            },
            StorageSpec {
                dtype: TorchDType::F16,
                bytes: 0x7e00u16.to_ne_bytes().to_vec(),
                elements: 1,
            },
            StorageSpec {
                dtype: TorchDType::F32,
                bytes: f32::MAX.to_ne_bytes().to_vec(),
                elements: 1,
            },
            StorageSpec {
                dtype: TorchDType::BF16,
                bytes: 0x7f7fu16.to_ne_bytes().to_vec(),
                elements: 1,
            },
        ];
        let second_entries = [
            ("bf16", 0usize, vec![5]),
            ("f16_inf", 1, vec![1]),
            ("f16_nan", 2, vec![1]),
            ("f32_overflow", 3, vec![1]),
            ("bf16_overflow", 4, vec![1]),
        ]
        .map(|(name, storage, shape)| EntrySpec {
            name: name.into(),
            storage,
            offset: 0,
            shape,
            stride: vec![1],
        });
        write_bin(
            &dir.0,
            files[1],
            &second_storages,
            &second_entries,
            zip::CompressionMethod::Stored,
        );
        let weight_map = first_entries
            .iter()
            .map(|entry| (entry.name.clone(), files[0]))
            .chain(
                second_entries
                    .iter()
                    .map(|entry| (entry.name.clone(), files[1])),
            )
            .collect::<BTreeMap<_, _>>();
        std::fs::write(
            dir.0.join("pytorch_model.bin.index.json"),
            serde_json::to_vec(&serde_json::json!({"weight_map": weight_map})).unwrap(),
        )
        .unwrap();

        let checkpoint = Checkpoint::open(&dir.0).unwrap();
        let source = checkpoint.source("f16", &[5]).unwrap();
        let mut out = [0u16; 5];
        source.read_f16(0, &mut out).unwrap();
        assert_eq!(out, f16_raw[1..]);
        let view = checkpoint.source("f16_view", &[2, 2]).unwrap();
        let mut out = [0u16; 4];
        view.read_f16(0, &mut out).unwrap();
        assert_eq!(out, [f16_raw[1], f16_raw[2], f16_raw[4], f16_raw[5]]);

        let source = checkpoint.source("f32", &[3]).unwrap();
        let mut out = [0u16; 3];
        source.read_f16(0, &mut out).unwrap();
        assert_eq!(out[0], half::f16::from_f32(f32_values[0]).to_bits());
        assert_eq!(out[1], 0x3c00, "ties round to even");
        assert_eq!(out[2], 0x8000, "signed zero is preserved");

        let source = checkpoint.source("bf16", &[5]).unwrap();
        let mut out = [0u16; 5];
        source.read_f16(0, &mut out).unwrap();
        let expected = bf16_raw
            .map(|bits| half::f16::from_f32(half::bf16::from_bits(bits).to_f32()).to_bits());
        assert_eq!(out, expected);

        for name in ["f16_inf", "f16_nan", "f32_overflow", "bf16_overflow"] {
            let source = checkpoint.source(name, &[1]).unwrap();
            let error = source.read_f16(0, &mut [0u16; 1]).unwrap_err().to_string();
            assert!(error.contains("nonfinite"), "{name}: {error}");
        }

        let source = checkpoint.source("f16", &[5]).unwrap();
        let mut before = [0u16; 5];
        source.read_f16(0, &mut before).unwrap();
        std::fs::rename(dir.0.join(files[0]), dir.0.join("renamed.bin")).unwrap();
        let mut after = [0u16; 5];
        source.read_f16(0, &mut after).unwrap();
        assert_eq!(before, after);
    }

    #[test]
    fn f16_reads_nested_nonzero_bin_slices() {
        let dir = Fixture::new();
        let raw = [
            0x0400u16, 0x0401, 0x0402, 0x0403, 0x0404, 0x0405, 0x0406, 0x0407, 0x0408, 0x0409,
            0x040a, 0x040b,
        ];
        let storages = vec![StorageSpec {
            dtype: TorchDType::F16,
            bytes: raw.map(u16::to_ne_bytes).concat(),
            elements: raw.len(),
        }];
        let entries = vec![EntrySpec {
            name: "f16_nested".into(),
            storage: 0,
            offset: 1,
            shape: vec![10],
            stride: vec![1],
        }];
        write_bin(
            &dir.0,
            "pytorch_model.bin",
            &storages,
            &entries,
            zip::CompressionMethod::Stored,
        );
        let checkpoint = Checkpoint::open(&dir.0).unwrap();
        let source = checkpoint.source("f16_nested", &[10]).unwrap();
        let first = source.slice(2, vec![6]).unwrap();
        assert_eq!(first.shape(), &[6]);
        let nested = first.slice(3, vec![2]).unwrap();
        assert_eq!(nested.shape(), &[2]);
        let mut out = [0u16; 2];
        nested.read_f16(0, &mut out).unwrap();
        assert_eq!(out, [raw[6], raw[7]]);
        let mut tail = [0u16; 1];
        nested.read_f16(1, &mut tail).unwrap();
        assert_eq!(tail[0], raw[7]);
        assert!(nested.read_f16(2, &mut [0u16; 1]).is_err());
        assert!(first.slice(5, vec![2]).is_err());
    }

    #[test]
    fn verifies_whole_backing_storage_before_publishing_ranges() {
        let elements = 17_500usize;
        let values = (0..elements)
            .map(|index| (index % 251) as f32)
            .collect::<Vec<_>>();
        let specs = vec![TensorSpec {
            name: "blob.weight".into(),
            dtype: TorchDType::F32,
            shape: vec![elements],
            values,
        }];
        let valid = Fixture::new();
        write_bin_single(&valid.0, &specs, zip::CompressionMethod::Stored);
        let checkpoint = PyTorchCheckpoint::open(&valid.0).unwrap();
        let source = checkpoint.source("blob.weight", &[elements]).unwrap();
        assert_eq!(source.verify().unwrap(), PublishedChecksum::Verified);
        let expected = specs[0]
            .values
            .iter()
            .map(|value| to_bf16(*value).unwrap())
            .collect::<Vec<_>>();
        let mut read = vec![0u16; elements];
        source.read_bf16(0, &mut read).unwrap();
        assert_eq!(read, expected);

        // A compressed entry is rejected before a range read. The frozen reader's deflated
        // access re-inflates from the entry start per window, so backend publication must not
        // accept it until a genuinely sequential bounded reader exists.
        let deflated = Fixture::new();
        write_bin_single(&deflated.0, &specs, zip::CompressionMethod::Deflated);
        let checkpoint = PyTorchCheckpoint::open(&deflated.0).unwrap();
        let source = checkpoint.source("blob.weight", &[elements]).unwrap();
        let error = source.verify().unwrap_err().to_string();
        assert!(
            error.contains("compressed ZIP entries are unsupported"),
            "{error}"
        );
        let mut deflated_read = vec![0u16; elements];
        let error = source
            .read_bf16(0, &mut deflated_read)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("compressed ZIP entries are unsupported"),
            "{error}"
        );

        let data_start = {
            let mut archive =
                zip::ZipArchive::new(File::open(valid.0.join("pytorch_model.bin")).unwrap())
                    .unwrap();
            let start = archive
                .by_name("data/0")
                .unwrap()
                .data_start()
                .expect("stored entry data start");
            start as usize
        };
        let mut bytes = std::fs::read(valid.0.join("pytorch_model.bin")).unwrap();
        bytes[data_start + 64 * 1024 + 3] ^= 0xff;
        std::fs::write(valid.0.join("pytorch_model.bin"), bytes).unwrap();
        let checkpoint = PyTorchCheckpoint::open(&valid.0).unwrap();
        let source = checkpoint.source("blob.weight", &[elements]).unwrap();
        let error = source.read_bf16(0, &mut read).unwrap_err().to_string();
        assert!(error.contains("verification failed"), "{error}");
    }

    #[test]
    fn tar_storage_is_accepted_as_explicit_no_checksum() {
        let fixture = Fixture::new();
        std::fs::copy(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/tar_float32.tar"),
            fixture.0.join("pytorch_model.bin"),
        )
        .unwrap();
        let checkpoint = PyTorchCheckpoint::open(&fixture.0).unwrap();
        let source = checkpoint.source("tensor", &[4]).unwrap();
        assert_eq!(source.verify().unwrap(), PublishedChecksum::NoChecksum);
        let mut out = [0u16; 4];
        source.read_bf16(0, &mut out).unwrap();
        assert_eq!(out[0], to_bf16(1.0).unwrap());
        assert_eq!(out[1], to_bf16(2.5).unwrap());
        assert_eq!(out[2], to_bf16(-3.7).unwrap());
        assert_eq!(out[3], to_bf16(0.0).unwrap());
    }

    #[test]
    fn legacy_storage_is_explicit_no_checksum_and_not_falsely_verified() {
        let fixture = Fixture::new();
        std::fs::copy(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/legacy_with_offsets.pt"),
            fixture.0.join("pytorch_model.bin"),
        )
        .unwrap();
        let checkpoint = PyTorchCheckpoint::open(&fixture.0).unwrap();
        let source = checkpoint.source("tensor1", &[10]).unwrap();
        assert_eq!(source.verify().unwrap(), PublishedChecksum::NoChecksum);
        let mut out = [0u16; 10];
        source.read_bf16(0, &mut out).unwrap();
        assert_eq!(out[0], to_bf16(1.0).unwrap());
        assert_eq!(out[9], to_bf16(1.09).unwrap());
    }

    #[test]
    fn retained_source_reads_after_path_rename() {
        let dir = Fixture::new();
        let specs = vec![TensorSpec {
            name: "weight".into(),
            dtype: TorchDType::F32,
            shape: vec![4],
            values: vec![1.0, 2.0, 3.0, 4.0],
        }];
        write_bin_single(&dir.0, &specs, zip::CompressionMethod::Stored);
        let checkpoint = Checkpoint::open(&dir.0).unwrap();
        let source = checkpoint.source("weight", &[4]).unwrap();
        let mut before = [0u16; 4];
        source.read_bf16(0, &mut before).unwrap();
        std::fs::rename(dir.0.join("pytorch_model.bin"), dir.0.join("renamed.bin")).unwrap();
        let mut after = [0u16; 4];
        source.read_bf16(0, &mut after).unwrap();
        assert_eq!(before, after);
    }

    fn link_or_copy(source: &Path, target: &Path) {
        if std::fs::hard_link(source, target).is_err() {
            std::fs::copy(source, target).unwrap();
        }
    }

    /// Copy only the native config and the `.bin` artifact into a fresh directory.
    /// The production loader must not be allowed to fall back to the safetensors file when
    /// the equivalence test asks whether the `.bin` is actually admitted.
    fn prepare_bin_only(source: &Path) -> Fixture {
        let destination = Fixture::new();
        std::fs::copy(
            source.join("config.json"),
            destination.0.join("config.json"),
        )
        .unwrap();
        let single = source.join("pytorch_model.bin");
        let index = source.join("pytorch_model.bin.index.json");
        if single.is_file() {
            link_or_copy(&single, &destination.0.join("pytorch_model.bin"));
        } else if index.is_file() {
            let value: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&index).unwrap()).unwrap();
            let mut shards = BTreeSet::new();
            for file in value["weight_map"]
                .as_object()
                .expect("weight_map")
                .values()
            {
                shards.insert(file.as_str().expect("shard name").to_string());
            }
            link_or_copy(&index, &destination.0.join("pytorch_model.bin.index.json"));
            for shard in shards {
                link_or_copy(&source.join(&shard), &destination.0.join(&shard));
            }
        } else {
            panic!("prepared directory has no pytorch_model.bin or index");
        }
        destination
    }

    fn config_model_type(dir: &Path) -> String {
        let value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
        value["model_type"]
            .as_str()
            .expect("config model_type")
            .to_string()
    }

    fn assert_production_loader_admits(dir: &Path) {
        match config_model_type(dir).as_str() {
            "bert" | "roberta" | "xlm-roberta" | "camembert" | "distilbert" | "new" | "gte" => {
                let config = BertConfig::load(dir).expect("encoder config");
                let weights = BertWeights::load(dir).expect("encoder .bin loader");
                assert_eq!(weights.layers.len(), config.num_hidden_layers);
                assert_eq!(weights.config.hidden_size, config.hidden_size);
            }
            "modernbert" => {
                let config = ModernConfig::load(dir).expect("modern config");
                let weights = ModernWeights::load(dir).expect("modern .bin loader");
                assert_eq!(weights.layers.len(), config.geometry.num_hidden_layers);
                assert_eq!(
                    weights.config.geometry.hidden_size,
                    config.geometry.hidden_size
                );
            }
            "qwen2" | "qwen3" | "llama" => {
                let config = DecoderConfig::load(dir).expect("decoder config");
                let weights = DecoderWeights::load(dir).expect("decoder .bin loader");
                assert_eq!(weights.layers.len(), config.num_hidden_layers);
                assert_eq!(weights.config.hidden_size, config.hidden_size);
            }
            other => panic!("unsupported review model_type {other}"),
        }
    }

    #[test]
    #[ignore = "requires INFINITY_OXIDE_EQUIV_DIR with model.safetensors, pytorch_model.bin and config.json"]
    fn external_trained_bin_matches_safetensors() {
        let dir = PathBuf::from(
            std::env::var_os("INFINITY_OXIDE_EQUIV_DIR")
                .expect("set INFINITY_OXIDE_EQUIV_DIR to the prepared model directory"),
        );
        let safe = Checkpoint::open(&dir).expect("safetensors checkpoint");
        let pytorch = PyTorchCheckpoint::open(&dir).expect("pytorch_model.bin checkpoint");
        let total_names = safe.names().count();
        let mut compared = 0usize;
        let mut skipped_non_float = 0usize;
        for name in safe.names() {
            let Some(shape) = safe.shape(name) else {
                panic!("missing safetensors shape for {name}");
            };
            let expected_source = match safe.source(name, shape) {
                Ok(source) => source,
                Err(error) if error.to_string().contains("unsupported weight dtype") => {
                    skipped_non_float += 1;
                    continue;
                }
                Err(error) => panic!("safetensors source {name}: {error}"),
            };
            let Some(py_shape) = pytorch.shape(name) else {
                panic!("pytorch_model.bin is missing {name}");
            };
            assert_eq!(py_shape, shape, "shape mismatch for {name}");
            let actual_source = pytorch.source(name, shape).expect("pytorch source");
            let mut expected = vec![0u16; expected_source.len()];
            expected_source.read_bf16(0, &mut expected).unwrap();
            let mut actual = vec![0u16; actual_source.len()];
            actual_source.read_bf16(0, &mut actual).unwrap();
            assert_eq!(actual, expected, "bit mismatch for {name}");
            compared += 1;
        }
        assert_eq!(
            compared + skipped_non_float,
            total_names,
            "every safetensors name must be compared or explicitly skipped as non-float"
        );
        assert!(compared > 0, "no floating model weights compared");
        assert_production_loader_admits(&prepare_bin_only(&dir).0);
        eprintln!(
            "compared {compared} floating weight sources bit-for-bit; \
             skipped {skipped_non_float} non-float buffers; production loader admitted the .bin"
        );
    }

    #[test]
    fn loads_sharded_gte_bin_and_reads_exact_packed_qkv_slices() {
        let h = 768usize;
        let i = 3072usize;
        let qkv_rows = 3 * h;
        let mut specs = vec![
            f32_rows(
                "encoder.layer.0.attention.qkv_proj.weight",
                qkv_rows,
                h,
                |row, column| (row * 1000 + column) as f32,
            ),
            raw_f32_row(
                "encoder.layer.0.attention.qkv_proj.bias",
                (0..qkv_rows).map(|row| 1_000_000.0 + row as f32),
            ),
            raw_zero(
                "embeddings.word_embeddings.weight",
                TorchDType::BF16,
                vec![2, h],
            ),
            raw_zero("embeddings.LayerNorm.weight", TorchDType::BF16, vec![h]),
            raw_zero("embeddings.LayerNorm.bias", TorchDType::BF16, vec![h]),
            raw_zero(
                "encoder.layer.0.attention.o_proj.weight",
                TorchDType::BF16,
                vec![h, h],
            ),
            raw_zero(
                "encoder.layer.0.attention.o_proj.bias",
                TorchDType::BF16,
                vec![h],
            ),
            raw_zero(
                "encoder.layer.0.mlp.up_gate_proj.weight",
                TorchDType::BF16,
                vec![2 * i, h],
            ),
            raw_zero(
                "encoder.layer.0.mlp.down_proj.weight",
                TorchDType::BF16,
                vec![h, i],
            ),
            raw_zero(
                "encoder.layer.0.mlp.down_proj.bias",
                TorchDType::BF16,
                vec![h],
            ),
            raw_zero("encoder.layer.0.attn_ln.weight", TorchDType::BF16, vec![h]),
            raw_zero("encoder.layer.0.attn_ln.bias", TorchDType::BF16, vec![h]),
            raw_zero("encoder.layer.0.mlp_ln.weight", TorchDType::BF16, vec![h]),
            raw_zero("encoder.layer.0.mlp_ln.bias", TorchDType::BF16, vec![h]),
        ];
        specs.sort_by(|left, right| left.name.cmp(&right.name));
        let split = specs
            .iter()
            .position(|spec| spec.name == "encoder.layer.0.mlp.down_proj.weight")
            .unwrap();
        let dir = Fixture::new();
        write_gte_config(&dir.0);
        write_bin_sharded_raw(
            &dir.0,
            ["model-00001-of-00002.bin", "model-00002-of-00002.bin"],
            &specs,
            split,
        );
        let weights = BertWeights::load(&dir.0).unwrap();
        assert_eq!(weights.layers.len(), 1);
        assert_eq!(weights.word.shape(), &[2, h]);
        let layer = &weights.layers[0];
        assert_slice(&layer.query.weight, 0, &[0.0, 1.0, 2.0, 3.0]);
        assert_slice(&layer.query.weight, 767 * h, &[767_000.0, 767_001.0]);
        assert_slice(&layer.key.weight, 0, &[768_000.0, 768_001.0, 768_002.0]);
        assert_slice(
            &layer.value.weight,
            0,
            &[1_536_000.0, 1_536_001.0, 1_536_002.0],
        );
        assert_slice(&layer.query.bias, 0, &[1_000_000.0, 1_000_001.0]);
        assert_slice(&layer.key.bias, 0, &[1_000_768.0, 1_000_769.0]);
        assert_slice(&layer.value.bias, 0, &[1_001_536.0, 1_001_537.0]);
        // A second slice keeps the first slice's nonzero base offset.
        let nested = layer.value.weight.slice(2 * h, vec![1, h]).unwrap();
        assert_slice(&nested, 0, &[1_538_000.0, 1_538_001.0]);
        assert_eq!(nested.shape(), &[1, h]);
    }

    #[test]
    fn loads_sharded_modern_bin_and_reads_exact_up_gate_slices() {
        let h = 768usize;
        let i = 8usize;
        let mut specs = vec![
            f32_rows("model.layers.0.mlp.Wi.weight", 2 * i, h, |row, column| {
                (row * 1000 + column) as f32
            }),
            raw_zero(
                "model.embeddings.tok_embeddings.weight",
                TorchDType::BF16,
                vec![2, h],
            ),
            raw_zero("model.embeddings.norm.weight", TorchDType::BF16, vec![h]),
            raw_zero("model.final_norm.weight", TorchDType::BF16, vec![h]),
            raw_zero(
                "model.layers.0.attn.Wqkv.weight",
                TorchDType::BF16,
                vec![3 * h, h],
            ),
            raw_zero(
                "model.layers.0.attn.Wo.weight",
                TorchDType::BF16,
                vec![h, h],
            ),
            raw_zero("model.layers.0.mlp.Wo.weight", TorchDType::BF16, vec![h, i]),
            raw_zero("model.layers.0.mlp_norm.weight", TorchDType::BF16, vec![h]),
        ];
        specs.sort_by(|left, right| left.name.cmp(&right.name));
        let split = specs
            .iter()
            .position(|spec| spec.name == "model.final_norm.weight")
            .unwrap();
        let dir = Fixture::new();
        write_modern_config(&dir.0);
        write_bin_sharded_raw(
            &dir.0,
            ["model-00001-of-00002.bin", "model-00002-of-00002.bin"],
            &specs,
            split,
        );
        let weights = ModernWeights::load(&dir.0).unwrap();
        assert_eq!(weights.layers.len(), 1);
        assert_eq!(weights.word.shape(), &[2, h]);
        let [up, gate] = &weights.layers[0].up_gate;
        assert_eq!(up.shape(), &[i, h]);
        assert_eq!(gate.shape(), &[i, h]);
        assert_slice(up, 0, &[8_000.0, 8_001.0, 8_002.0]);
        assert_slice(gate, 0, &[0.0, 1.0, 2.0]);
        assert_slice(up, 7 * h, &[15_000.0, 15_001.0]);
        assert_slice(gate, 7 * h, &[7_000.0, 7_001.0]);
        // Nested slice starts at a nonzero offset inside the already-offset up view.
        let nested = up.slice(2 * h, vec![1, h]).unwrap();
        assert_slice(&nested, 0, &[10_000.0, 10_001.0]);
        assert_eq!(nested.shape(), &[1, h]);
    }

    #[test]
    fn loads_current_decoder_families_from_bin_metadata() {
        for (model_type, hidden, intermediate, heads, kv, head_dim, vocab, causal) in [
            (
                "qwen2", 8usize, 16usize, 2usize, 1usize, 4usize, 5usize, false,
            ),
            ("qwen3", 256, 16, 2, 1, 128, 5, false),
            ("llama", 8, 16, 2, 1, 4, 5, true),
        ] {
            let dir = Fixture::new();
            std::fs::write(
                dir.0.join("config.json"),
                serde_json::to_vec(&serde_json::json!({
                    "model_type": model_type,
                    "hidden_size": hidden,
                    "intermediate_size": intermediate,
                    "num_hidden_layers": 1,
                    "num_attention_heads": heads,
                    "num_key_value_heads": kv,
                    "head_dim": head_dim,
                    "vocab_size": vocab,
                    "max_position_embeddings": 16,
                    "hidden_act": "silu",
                    "is_causal": causal
                }))
                .unwrap(),
            )
            .unwrap();
            let config = DecoderConfig::load(&dir.0).unwrap();
            let specs = decoder_tensor_shapes(&config)
                .into_iter()
                .map(|(name, shape)| {
                    let values = (0..element_count(&shape))
                        .map(|index| value(&name, index))
                        .collect();
                    TensorSpec {
                        name,
                        dtype: TorchDType::F32,
                        shape,
                        values,
                    }
                })
                .collect::<Vec<_>>();
            write_bin_single(&dir.0, &specs, zip::CompressionMethod::Stored);
            let weights = DecoderWeights::load(&dir.0).unwrap();
            assert_eq!(weights.layers.len(), 1, "{model_type}");
            assert_eq!(weights.embed.len(), vocab * hidden, "{model_type}");
            assert_eq!(
                weights.layers.first().unwrap().q.weight.shape(),
                &[config.q_width(), hidden],
                "{model_type}"
            );
        }
    }

    #[test]
    fn loads_tiny_decoder_metadata_from_bin() {
        let dir = Fixture::new();
        std::fs::write(
            dir.0.join("config.json"),
            serde_json::to_vec(&serde_json::json!({
                "model_type": "qwen2",
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "vocab_size": 5,
                "max_position_embeddings": 16,
                "hidden_act": "silu",
                "is_causal": false
            }))
            .unwrap(),
        )
        .unwrap();
        let config = DecoderConfig::load(&dir.0).unwrap();
        let specs = decoder_tensor_shapes(&config)
            .into_iter()
            .map(|(name, shape)| {
                let values = (0..element_count(&shape))
                    .map(|index| value(&name, index))
                    .collect();
                TensorSpec {
                    name,
                    dtype: TorchDType::F32,
                    shape,
                    values,
                }
            })
            .collect::<Vec<_>>();
        write_bin_single(&dir.0, &specs, zip::CompressionMethod::Stored);
        let weights = DecoderWeights::load(&dir.0).unwrap();
        assert_eq!(weights.layers.len(), 1);
        let mut out = vec![0u16; weights.embed.len()];
        weights.embed.read_bf16(0, &mut out).unwrap();
        assert_eq!(out.len(), 5 * 8);
        assert_eq!(weights.layers.first().unwrap().q.weight.shape(), &[8, 8]);
    }
}
