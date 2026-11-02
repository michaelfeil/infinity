use crate::batch::{self, BatchJson};
use infinity_v3_backend_core::MultimodalEncoding;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const GENERATED_SHA256_PENDING: &str = "GENERATE-WITH-control/generate_fixtures.py";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Case {
    pub name: String,
    pub file: String,
    pub sha256: String,
    pub valid: bool,
    pub expected_error_variant: Option<String>,
    pub special: Option<String>,
    pub token_count: usize,
    pub sequence_count: usize,
    pub pooled_count: usize,
    pub raw_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub schema: String,
    pub batch_fields: Vec<String>,
    pub ids_rule: String,
    pub cases: Vec<Case>,
}

#[derive(Debug, Clone)]
pub struct LoadedCase {
    pub case: Case,
    pub path: PathBuf,
    pub batch: BatchJson,
    pub file_sha256: String,
}

pub fn load_manifest(path: &Path) -> Result<Manifest, String> {
    let data = fs::read(path).map_err(|e| format!("read manifest {}: {e}", path.display()))?;
    let manifest: Manifest =
        serde_json::from_slice(&data).map_err(|e| format!("parse manifest: {e}"))?;
    if manifest.schema != "oxide-q35moe-unified-fixtures/v1" {
        return Err(format!("unsupported fixture schema {}", manifest.schema));
    }
    if manifest.batch_fields
        != [
            "input_ids",
            "position_ids",
            "token_type_ids",
            "cumulative_seq_lengths",
            "pooled_indices",
            "raw_indices",
            "max_length",
        ]
    {
        return Err("manifest batch_fields are not the canonical seven".into());
    }
    Ok(manifest)
}

pub fn load_cases(manifest_path: &Path, limit: Option<usize>) -> Result<Vec<LoadedCase>, String> {
    let manifest = load_manifest(manifest_path)?;
    let root = manifest_path
        .parent()
        .ok_or_else(|| "manifest has no parent directory".to_string())?;
    let cases = manifest.cases.into_iter().take(limit.unwrap_or(usize::MAX));
    let mut loaded = Vec::new();
    for case in cases {
        let path = root.join(&case.file);
        let data = fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
        if data.len() as u64 > batch::MAX_BATCH_JSON {
            return Err(format!("{} exceeds 32 MiB", path.display()));
        }
        let digest = format!("{:x}", Sha256::digest(&data));
        let pending_generated = case.name == "t9" && case.sha256 == GENERATED_SHA256_PENDING;
        if !pending_generated && digest != case.sha256 {
            return Err(format!("{} sha256 mismatch", path.display()));
        }
        let parsed: BatchJson = serde_json::from_slice(&data)
            .map_err(|e| format!("parse exact seven-field batch {}: {e}", path.display()))?;
        if case.valid {
            batch::validate(&parsed).map_err(|e| format!("{}: {e}", case.name))?;
        }
        loaded.push(LoadedCase {
            case,
            path,
            batch: parsed,
            file_sha256: digest,
        });
    }
    Ok(loaded)
}

pub fn backend_batch(item: &LoadedCase) -> Result<infinity_v3_backend_core::Batch, String> {
    let mut value = batch::to_backend_batch(&item.batch);
    match item.case.special.as_deref() {
        None => {}
        Some("media") => {
            value.multimodal = vec![Some(Arc::new(MultimodalEncoding {
                images: Vec::new(),
                audios: Vec::new(),
                position_ids: [Vec::new(), Vec::new(), Vec::new()],
                memory: None,
            }))]
        }
        Some("compact") => value.compact_input_ids = Some(value.input_ids.clone()),
        Some("empty") => {}
        Some(other) => return Err(format!("unknown special fixture kind {other}")),
    }
    Ok(value)
}
