use crate::cli::{ensure_model_is_directory, Args};
use crate::report::Report;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

pub const PROVIDER_ENVS: [&str; 6] = [
    "QWEN35_GDN_KERNELS",
    "QWEN35_GDN_PROVIDER",
    "QWEN35_GDN_AUX",
    "INFINITY_MOE_AOT_ROOT",
    "INFINITY_FA4_ROOT",
    "INFINITY_OXIDE_KERNEL_MODULE",
];

fn sha256(data: &[u8]) -> String {
    format!("{:x}", Sha256::digest(data))
}

fn read_limited(path: &Path, limit: u64) -> Result<Vec<u8>, String> {
    let metadata = fs::metadata(path).map_err(|e| format!("metadata {}: {e}", path.display()))?;
    if metadata.len() > limit {
        return Err(format!("{} exceeds metadata limit", path.display()));
    }
    fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))
}

fn stat_identity(path: &Path) -> Result<Value, String> {
    let metadata = fs::metadata(path).map_err(|e| format!("stat {}: {e}", path.display()))?;
    Ok(json!({
        "len": metadata.len(),
        "modified_unix_ns": metadata.mtime() as i128 * 1_000_000_000 + metadata.mtime_nsec() as i128,
        "inode": metadata.ino(),
        "device": metadata.dev(),
    }))
}

fn config_facts(config: &Value) -> Value {
    let text = config.get("text_config").unwrap_or(config);
    let get = |name: &str| text.get(name).cloned().unwrap_or(Value::Null);
    let layer_types = get("layer_types");
    let layer_order_sha256 = match &layer_types {
        Value::Null => Value::Null,
        value => Value::String(sha256(&serde_json::to_vec(value).unwrap_or_default())),
    };
    json!({
        "model_type": config.get("model_type").cloned().unwrap_or(Value::Null),
        "text_model_type": text.get("model_type").cloned().unwrap_or(Value::Null),
        "architectures": config.get("architectures").cloned().unwrap_or(Value::Null),
        "class": config.get("architectures").cloned().unwrap_or(Value::Null),
        "hidden_size": get("hidden_size"),
        "num_hidden_layers": get("num_hidden_layers"),
        "vocab_size": get("vocab_size"),
        "max_position_embeddings": get("max_position_embeddings"),
        "num_experts": get("num_experts"),
        "num_experts_per_tok": get("num_experts_per_tok"),
        "moe_intermediate_size": get("moe_intermediate_size"),
        "shared_expert_intermediate_size": get("shared_expert_intermediate_size"),
        "q_heads": get("num_attention_heads"),
        "kv_heads": get("num_key_value_heads"),
        "head_dim": get("head_dim"),
        "layer_types": layer_types,
        "layer_order_sha256": layer_order_sha256,
        "dtype": if get("dtype").is_null() { get("torch_dtype") } else { get("dtype") },
    })
}

fn header_facts(path: &Path) -> Result<Value, String> {
    let stat = stat_identity(path)?;
    let mut file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let file_bytes = file.metadata().map_err(|e| e.to_string())?.len();
    if file_bytes < 8 {
        return Err(format!(
            "{} is shorter than a safetensors length",
            path.display()
        ));
    }
    let mut length = [0u8; 8];
    file.read_exact(&mut length).map_err(|e| e.to_string())?;
    let header_len = u64::from_le_bytes(length);
    if header_len == 0 || header_len > 128 << 20 || header_len > file_bytes.saturating_sub(8) {
        return Err(format!(
            "{} has invalid safetensors header length",
            path.display()
        ));
    }
    let mut header = vec![0u8; header_len as usize];
    file.read_exact(&mut header).map_err(|e| e.to_string())?;
    file.seek(SeekFrom::End(0)).map_err(|e| e.to_string())?;
    let parsed: Value = serde_json::from_slice(&header)
        .map_err(|e| format!("{} safetensors header JSON: {e}", path.display()))?;
    let object = parsed
        .as_object()
        .ok_or_else(|| format!("{} safetensors header is not an object", path.display()))?;
    let mut dtypes = BTreeMap::<String, usize>::new();
    let mut tensor_count = 0usize;
    for (name, tensor) in object {
        if name == "__metadata__" {
            continue;
        }
        tensor_count += 1;
        if let Some(dtype) = tensor.get("dtype").and_then(Value::as_str) {
            *dtypes.entry(dtype.to_string()).or_default() += 1;
        }
    }
    let metadata = object.get("__metadata__").cloned().unwrap_or(Value::Null);
    let mut payload = 0u64;
    for (name, tensor) in object {
        if name == "__metadata__" {
            continue;
        }
        let offsets = tensor.get("data_offsets").and_then(Value::as_array);
        if let Some(offsets) = offsets {
            if let (Some(start), Some(end)) = (
                offsets.first().and_then(Value::as_u64),
                offsets.get(1).and_then(Value::as_u64),
            ) {
                if end < start {
                    return Err(format!(
                        "{} tensor {name} has reversed offsets",
                        path.display()
                    ));
                }
                payload = payload.max(end);
            }
        }
    }
    if payload > file_bytes.saturating_sub(8 + header_len) {
        return Err(format!(
            "{} tensor payload exceeds file size",
            path.display()
        ));
    }
    Ok(json!({
        "file": path.display().to_string(),
        "file_bytes": file_bytes,
        "stat": stat,
        "header_bytes": header_len,
        "header_sha256": sha256(&header),
        "tensor_count": tensor_count,
        "dtypes": dtypes,
        "metadata": metadata,
        "declared_payload_bytes": payload,
        "payload_read_bytes": 0,
    }))
}

fn index_facts(path: &Path) -> Result<Value, String> {
    let stat = stat_identity(path)?;
    let data = read_limited(path, 32 << 20)?;
    let parsed: Value =
        serde_json::from_slice(&data).map_err(|e| format!("{} index JSON: {e}", path.display()))?;
    let weight_map = parsed
        .get("weight_map")
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{} has no weight_map", path.display()))?;
    let shards: std::collections::BTreeSet<String> = weight_map
        .values()
        .filter_map(Value::as_str)
        .map(str::to_string)
        .collect();
    Ok(json!({
        "file": path.display().to_string(),
        "stat": stat,
        "bytes": data.len(),
        "sha256": sha256(&data),
        "tensor_count": weight_map.len(),
        "shards": shards,
    }))
}

pub fn collect(
    model: &Path,
    source: &str,
    freeze: &str,
    provider_root: &str,
) -> Result<Value, String> {
    ensure_model_is_directory(model)?;
    let config_path = model.join("config.json");
    let config_data = read_limited(&config_path, 32 << 20)?;
    let config: Value =
        serde_json::from_slice(&config_data).map_err(|e| format!("config.json parse: {e}"))?;
    let mut headers = Vec::new();
    let mut indices = Vec::new();
    let mut module_hashes = BTreeMap::new();
    let mut files = fs::read_dir(model)
        .map_err(|e| format!("read model dir {}: {e}", model.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .collect::<Vec<PathBuf>>();
    files.sort();
    for path in &files {
        let name = path
            .file_name()
            .and_then(|v| v.to_str())
            .unwrap_or_default();
        if name.ends_with(".safetensors") {
            headers.push(header_facts(path)?);
        } else if name.ends_with(".safetensors.index.json") {
            indices.push(index_facts(path)?);
        } else if matches!(
            name,
            "modules.json" | "config_sentence_transformers.json" | "tokenizer_config.json"
        ) {
            let data = read_limited(path, 32 << 20)?;
            module_hashes.insert(name.to_string(), sha256(&data));
        }
    }
    if headers.is_empty() {
        return Err("metadata: no safetensors files found".into());
    }
    if indices.is_empty() {
        return Err("metadata: no safetensors index found".into());
    }
    let envs: BTreeMap<&str, Option<String>> = PROVIDER_ENVS
        .iter()
        .map(|name| (*name, std::env::var(name).ok()))
        .collect();
    let provider = json!({
        "canonical31_provider": provider_root,
        "source": source,
        "source_freeze_sha256": freeze,
        "envs": envs,
    });
    Ok(json!({
        "model": model.display().to_string(),
        "config_sha256": sha256(&config_data),
        "config": config_facts(&config),
        "safetensors_headers": headers,
        "indices": indices,
        "module_identity_sha256": module_hashes,
        "provider": provider,
        "collector_created_engine": false,
        "collector_created_context": false,
        "tensor_payload_read_bytes": 0,
    }))
}

pub fn run(args: &Args, report: &mut Report) -> Result<(), String> {
    let model = args
        .model
        .as_ref()
        .ok_or_else(|| "--model is required for metadata".to_string())?;
    report.engine_count = 0;
    report.normal_factory_route = false;
    let mut value = collect(
        model,
        report.product_source,
        report.product_freeze_sha256,
        report.canonical31_provider,
    )?;
    // This mode performs no constructor; runtime snapshots make no global engine claim.
    value["engine_loaded"] = json!(false);
    value["context_created"] = json!(false);
    report.metadata = Some(value);
    report.phase(
        "metadata",
        "ok",
        json!({"config_only": true, "headers_only": true, "engine_count": 0}),
    );
    Ok(())
}
