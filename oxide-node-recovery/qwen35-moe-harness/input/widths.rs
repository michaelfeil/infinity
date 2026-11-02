use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Component, Path, PathBuf};

use crate::batch::{BatchError, MAX_BATCH_JSON};

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DenseWidth {
    pub order: usize,
    pub module_index: usize,
    pub name: String,
    pub relative_path: String,
    pub in_features: usize,
    pub out_features: usize,
    pub config_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelWidths {
    pub model: String,
    pub hidden_width: usize,
    pub pooled_width: usize,
    pub dense_layers: usize,
    pub dense: Vec<DenseWidth>,
    pub config_sha256: String,
    pub modules_sha256: String,
    pub pooling_config_sha256: String,
    pub width_source: String,
}

fn width_error(message: impl Into<String>) -> BatchError {
    BatchError(message.into())
}

fn read_json(path: &Path) -> Result<Value, BatchError> {
    let data = fs::read(path)
        .map_err(|error| width_error(format!("cannot read metadata {path:?}: {error}")))?;
    serde_json::from_slice(&data)
        .map_err(|error| width_error(format!("invalid metadata JSON {path:?}: {error}")))
}

fn sha256_file(path: &Path) -> Result<String, BatchError> {
    let data = fs::read(path)
        .map_err(|error| width_error(format!("cannot hash metadata {path:?}: {error}")))?;
    if data.len() as u64 > MAX_BATCH_JSON {
        return Err(width_error(format!(
            "metadata file is unreasonably large: {path:?}"
        )));
    }
    Ok(format!("{:x}", Sha256::digest(&data)))
}

fn positive(value: Option<&Value>, name: &str) -> Result<usize, BatchError> {
    let raw = value
        .and_then(Value::as_u64)
        .ok_or_else(|| width_error(format!("{name} must be a positive integer")))?;
    if raw == 0 {
        return Err(width_error(format!("{name} must be positive")));
    }
    usize::try_from(raw).map_err(|_| width_error(format!("{name} does not fit usize")))
}

fn relative_dir(root: &Path, value: &str) -> Result<PathBuf, BatchError> {
    if value.is_empty() {
        return Ok(root.to_path_buf());
    }
    if value.contains('\\') {
        return Err(width_error(format!(
            "module paths must use POSIX separators: {value:?}"
        )));
    }
    let relative = Path::new(value);
    if relative.is_absolute()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(width_error(format!(
            "module path must stay relative and canonical: {value:?}"
        )));
    }
    let path = root.join(relative);
    let resolved = path
        .canonicalize()
        .map_err(|error| width_error(format!("missing module directory {value:?}: {error}")))?;
    let root_resolved = root
        .canonicalize()
        .map_err(|error| width_error(format!("missing model root: {error}")))?;
    if resolved != root_resolved && !resolved.starts_with(&root_resolved) {
        return Err(width_error(format!(
            "module path escapes the model root: {value:?}"
        )));
    }
    if !resolved.is_dir() {
        return Err(width_error(format!(
            "module path is not a directory: {value:?}"
        )));
    }
    Ok(resolved)
}

pub fn read_model_widths(
    model: &Path,
    expected_hidden: Option<usize>,
    expected_pooled: Option<usize>,
) -> Result<ModelWidths, BatchError> {
    let model = model
        .canonicalize()
        .map_err(|error| width_error(format!("missing model directory {model:?}: {error}")))?;
    let config_path = model.join("config.json");
    let config = read_json(&config_path)?;
    let hidden = config.get("hidden_size").or_else(|| config.get("dim"));
    let hidden_width = positive(hidden, "model hidden width")?;
    if let Some(expected) = expected_hidden {
        if expected != hidden_width {
            return Err(width_error(format!(
                "expected hidden width {expected} does not match configured {hidden_width}"
            )));
        }
    }

    let modules_path = model.join("modules.json");
    let modules = read_json(&modules_path)?;
    let entries = modules
        .as_array()
        .ok_or_else(|| width_error("modules.json must be an ordered array"))?;
    if entries.is_empty() {
        return Err(width_error("modules.json must not be empty"));
    }

    let mut seen_transformer = false;
    let mut seen_pooling = false;
    let mut pooling_config_sha = String::new();
    let mut dense = Vec::new();
    let mut current = hidden_width;

    for (position, entry) in entries.iter().enumerate() {
        let object = entry
            .as_object()
            .ok_or_else(|| width_error(format!("modules[{position}] must be an object")))?;
        let module_type = object
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| width_error(format!("modules[{position}] needs a string type")))?;
        let module_path = object
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| width_error(format!("modules[{position}] needs a string path")))?;
        let index = object
            .get("idx")
            .and_then(Value::as_u64)
            .unwrap_or(position as u64);
        if index != position as u64 {
            return Err(width_error(format!(
                "modules[{position}] idx must equal its ordered position"
            )));
        }
        if module_type.ends_with(".Transformer") {
            if position != 0 || seen_transformer || seen_pooling || !dense.is_empty() {
                return Err(width_error(
                    "exactly one leading Transformer module is required",
                ));
            }
            if !module_path.is_empty() {
                return Err(width_error("the Transformer module path must be empty"));
            }
            seen_transformer = true;
        } else if module_type.ends_with(".Pooling") {
            if !seen_transformer || seen_pooling || !dense.is_empty() {
                return Err(width_error(
                    "exactly one Pooling module must follow the Transformer",
                ));
            }
            let path = relative_dir(&model, module_path)?;
            let pooling_path = path.join("config.json");
            let pooling = read_json(&pooling_path)?;
            let dimension = positive(
                pooling.get("word_embedding_dimension"),
                "pooling word_embedding_dimension",
            )?;
            if dimension != hidden_width {
                return Err(width_error(format!(
                    "pooling dimension {dimension} does not match hidden width {hidden_width}"
                )));
            }
            pooling_config_sha = sha256_file(&pooling_path)?;
            seen_pooling = true;
        } else if module_type.ends_with(".Dense") {
            if !seen_pooling {
                return Err(width_error("Dense modules must follow the Pooling module"));
            }
            let path = relative_dir(&model, module_path)?;
            let dense_path = path.join("config.json");
            let dense_config = read_json(&dense_path)?;
            let in_features = positive(dense_config.get("in_features"), "Dense in_features")?;
            let out_features = positive(dense_config.get("out_features"), "Dense out_features")?;
            if in_features != current {
                return Err(width_error(format!(
                    "Dense in_features {in_features} does not continue width {current}"
                )));
            }
            dense.push(DenseWidth {
                order: dense.len(),
                module_index: position,
                name: object
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or(module_path)
                    .to_string(),
                relative_path: module_path.to_string(),
                in_features,
                out_features,
                config_sha256: sha256_file(&dense_path)?,
            });
            current = out_features;
        } else {
            return Err(width_error(format!(
                "unsupported SentenceTransformers module type {module_type:?}"
            )));
        }
    }

    if !seen_transformer || !seen_pooling {
        return Err(width_error(
            "the ordered module array needs Transformer then Pooling",
        ));
    }
    if dense.is_empty() {
        return Err(width_error(
            "the ordered module array needs at least one Dense module",
        ));
    }
    let pooled_width = current;
    if let Some(expected) = expected_pooled {
        if expected != pooled_width {
            return Err(width_error(format!(
                "expected pooled width {expected} does not match configured {pooled_width}"
            )));
        }
    }

    Ok(ModelWidths {
        model: model.to_string_lossy().into_owned(),
        hidden_width,
        pooled_width,
        dense_layers: dense.len(),
        dense,
        config_sha256: sha256_file(&config_path)?,
        modules_sha256: sha256_file(&modules_path)?,
        pooling_config_sha256: pooling_config_sha,
        width_source: "config.json + modules.json + ordered Dense config.json".into(),
    })
}
