// Cold safetensors metadata reader.
//
// This module only reads the 8-byte header length and the JSON header of each
// shard. It never reads tensor payloads, creates a device context, or imports a
// framework. It is the metadata floor used by the inventory, weight-binding and
// Native CAPI control-packet modules.

use infinity_v3_backend_core::BackendError;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Component, Path};
use std::sync::{Arc, Mutex};

const SAFETENSORS_PREFIX_BYTES: u64 = 8;
const MAX_HEADER_BYTES: u64 = 100_000_000;
const MAX_SIDECAR_BYTES: u64 = 16 * 1024 * 1024;

pub(crate) fn start(error: impl std::fmt::Display) -> BackendError {
    BackendError::Start(format!("Oxide qwen3-vl checkpoint: {error}"))
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

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum CheckpointDtype {
    F32,
    F16,
    Bf16,
}

impl CheckpointDtype {
    pub fn from_name(name: &str) -> Result<Self, BackendError> {
        match name {
            "F32" | "f32" => Ok(Self::F32),
            "F16" | "f16" => Ok(Self::F16),
            "BF16" | "bf16" => Ok(Self::Bf16),
            other => Err(start(format!("unsupported checkpoint dtype {other}"))),
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Bf16 => "BF16",
        }
    }

    pub const fn width(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorMeta {
    pub name: String,
    pub dtype: CheckpointDtype,
    pub shape: Vec<usize>,
    pub shard: String,
    pub data_offsets: (u64, u64),
}

impl TensorMeta {
    pub fn elements(&self) -> Result<usize, BackendError> {
        self.shape.iter().try_fold(1usize, |count, dim| {
            count
                .checked_mul(*dim)
                .ok_or_else(|| start(format!("{} dimension overflow", self.name)))
        })
    }

    pub fn bytes(&self) -> Result<u64, BackendError> {
        let elements = u64::try_from(self.elements()?).map_err(start)?;
        elements
            .checked_mul(self.dtype.width() as u64)
            .ok_or_else(|| start(format!("{} byte size overflow", self.name)))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShardMeta {
    pub name: String,
    pub file_bytes: u64,
    pub header_bytes: u64,
    pub data_start: u64,
    pub data_length: u64,
    pub tensor_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckpointMetadata {
    pub root: std::path::PathBuf,
    pub shards: BTreeMap<String, ShardMeta>,
    pub tensors: BTreeMap<String, TensorMeta>,
}

#[derive(Deserialize)]
struct RawIndex {
    weight_map: BTreeMap<String, String>,
}

#[derive(Deserialize)]
struct RawTensor {
    dtype: String,
    shape: Vec<i64>,
    data_offsets: [u64; 2],
}

fn validate_shard_name(file: &str) -> Result<(), BackendError> {
    let mut components = Path::new(file).components();
    if !matches!(components.next(), Some(Component::Normal(_)))
        || components.next().is_some()
        || !file.ends_with(".safetensors")
        || file.contains('\\')
    {
        return Err(start(format!("invalid safetensors shard path {file}")));
    }
    Ok(())
}

fn read_header(path: &Path) -> Result<(u64, BTreeMap<String, RawTensor>), BackendError> {
    let mut file = File::open(path).map_err(start)?;
    let file_bytes = file.metadata().map_err(start)?.len();
    let mut prefix = [0u8; SAFETENSORS_PREFIX_BYTES as usize];
    file.read_exact(&mut prefix).map_err(start)?;
    let header_bytes = u64::from_le_bytes(prefix);
    if header_bytes == 0 || header_bytes > MAX_HEADER_BYTES {
        return Err(start(format!(
            "invalid safetensors header length {header_bytes} in {}",
            path.display()
        )));
    }
    let header_end = SAFETENSORS_PREFIX_BYTES
        .checked_add(header_bytes)
        .ok_or_else(|| start("safetensors header offset overflow"))?;
    if header_end > file_bytes {
        return Err(start(format!(
            "safetensors header exceeds {}",
            path.display()
        )));
    }
    file.seek(SeekFrom::Start(SAFETENSORS_PREFIX_BYTES))
        .map_err(start)?;
    let mut bytes = vec![0u8; usize::try_from(header_bytes).map_err(start)?];
    file.read_exact(&mut bytes).map_err(start)?;
    let value: serde_json::Value = serde_json::from_slice(&bytes).map_err(start)?;
    let object = value
        .as_object()
        .ok_or_else(|| start("safetensors header is not an object"))?;
    let mut tensors = BTreeMap::new();
    for (name, raw) in object {
        if name == "__metadata__" {
            continue;
        }
        let tensor: RawTensor = serde_json::from_value(raw.clone()).map_err(start)?;
        if tensor.shape.iter().any(|dim| *dim < 0) {
            return Err(start(format!("negative tensor dimension for {name}")));
        }
        let shape = tensor
            .shape
            .into_iter()
            .map(|dim| usize::try_from(dim).map_err(start))
            .collect::<Result<Vec<_>, _>>()?;
        tensors.insert(
            name.clone(),
            RawTensor {
                dtype: tensor.dtype,
                shape: shape
                    .iter()
                    .map(|dim| i64::try_from(*dim).map_err(start))
                    .collect::<Result<Vec<_>, _>>()?,
                data_offsets: tensor.data_offsets,
            },
        );
    }
    Ok((header_bytes, tensors))
}

impl CheckpointMetadata {
    pub fn open(root: &Path) -> Result<Self, BackendError> {
        let single = root.join("model.safetensors");
        let (files, weight_map): (Vec<String>, Option<BTreeMap<String, String>>) =
            if path_present(&single)? {
                (vec!["model.safetensors".to_owned()], None)
            } else {
                let index: RawIndex = read_json(&root.join("model.safetensors.index.json"))?;
                if index.weight_map.is_empty() {
                    return Err(start("empty safetensors shard index"));
                }
                let mut unique = BTreeSet::new();
                for file in index.weight_map.values() {
                    validate_shard_name(file)?;
                    unique.insert(file.clone());
                }
                (unique.into_iter().collect(), Some(index.weight_map))
            };

        let mut shards = BTreeMap::new();
        let mut tensors = BTreeMap::new();
        for file in &files {
            let path = root.join(file);
            let file_bytes = std::fs::metadata(&path).map_err(start)?.len();
            let (header_bytes, raw) = read_header(&path)?;
            let data_start = SAFETENSORS_PREFIX_BYTES
                .checked_add(header_bytes)
                .ok_or_else(|| start("safetensors data start overflow"))?;
            let data_length = file_bytes
                .checked_sub(data_start)
                .ok_or_else(|| start("safetensors data length underflow"))?;
            let mut max_end = 0u64;
            for (name, raw_tensor) in raw {
                if let Some(expected) = weight_map.as_ref() {
                    if expected.get(&name) != Some(file) {
                        return Err(start(format!(
                            "tensor {name} is not mapped to {file} by the index"
                        )));
                    }
                }
                let (begin, end) = (raw_tensor.data_offsets[0], raw_tensor.data_offsets[1]);
                if begin > end || end > data_length {
                    return Err(start(format!("tensor {name} range exceeds {file}")));
                }
                let dtype = CheckpointDtype::from_name(&raw_tensor.dtype)?;
                let shape: Vec<usize> = raw_tensor
                    .shape
                    .into_iter()
                    .map(|dim| usize::try_from(dim).map_err(start))
                    .collect::<Result<Vec<_>, _>>()?;
                let meta = TensorMeta {
                    name: name.clone(),
                    dtype,
                    shape,
                    shard: file.clone(),
                    data_offsets: (begin, end),
                };
                let declared = meta.bytes()?;
                if end - begin != declared {
                    return Err(start(format!(
                        "tensor {name} declares {} bytes but its shape/dtype require {declared}",
                        end - begin
                    )));
                }
                if tensors.insert(name.clone(), meta).is_some() {
                    return Err(start(format!("duplicate tensor {name} across shards")));
                }
                max_end = max_end.max(end);
            }
            if max_end != data_length {
                return Err(start(format!(
                    "shard {file} payload ends at {max_end}, file declares {data_length}"
                )));
            }
            shards.insert(
                file.clone(),
                ShardMeta {
                    name: file.clone(),
                    file_bytes,
                    header_bytes,
                    data_start,
                    data_length,
                    tensor_count: 0,
                },
            );
        }
        for tensor in tensors.values() {
            if let Some(shard) = shards.get_mut(&tensor.shard) {
                shard.tensor_count += 1;
            }
        }
        if let Some(expected) = weight_map {
            if expected.len() != tensors.len() {
                return Err(start(
                    "index and safetensors headers disagree on tensor count",
                ));
            }
        }
        Ok(Self {
            root: root.to_path_buf(),
            shards,
            tensors,
        })
    }

    pub fn total_elements(&self) -> Result<u64, BackendError> {
        self.tensors.values().try_fold(0u64, |total, tensor| {
            total
                .checked_add(u64::try_from(tensor.elements()?).map_err(start)?)
                .ok_or_else(|| start("checkpoint element count overflow"))
        })
    }

    pub fn total_bytes(&self) -> Result<u64, BackendError> {
        self.tensors.values().try_fold(0u64, |total, tensor| {
            total
                .checked_add(tensor.bytes()?)
                .ok_or_else(|| start("checkpoint byte count overflow"))
        })
    }

    pub fn shard_file(&self, shard: &str) -> std::path::PathBuf {
        self.root.join(shard)
    }
}

/// Borrowable file handle for the future uploader. The CPU-only tests never
/// open these handles; they only validate the bounded ranges.
#[derive(Clone)]
pub struct ShardFile {
    pub shard: String,
    pub data_start: u64,
    pub data_length: u64,
    file: Arc<Mutex<File>>,
}

impl ShardFile {
    pub fn open(root: &Path, shard: &str) -> Result<Self, BackendError> {
        validate_shard_name(shard)?;
        let file = File::open(root.join(shard)).map_err(start)?;
        let file_bytes = file.metadata().map_err(start)?.len();
        let mut prefix = [0u8; SAFETENSORS_PREFIX_BYTES as usize];
        let mut reader = file.try_clone().map_err(start)?;
        reader.read_exact(&mut prefix).map_err(start)?;
        let header_bytes = u64::from_le_bytes(prefix);
        let data_start = SAFETENSORS_PREFIX_BYTES
            .checked_add(header_bytes)
            .ok_or_else(|| start("safetensors data start overflow"))?;
        let data_length = file_bytes
            .checked_sub(data_start)
            .ok_or_else(|| start("safetensors data length underflow"))?;
        Ok(Self {
            shard: shard.to_owned(),
            data_start,
            data_length,
            file: Arc::new(Mutex::new(file)),
        })
    }

    pub fn clone_handle(&self) -> Arc<Mutex<File>> {
        Arc::clone(&self.file)
    }
}
