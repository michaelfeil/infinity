// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Shared model-metadata fallback for the normal service downloader/router.
//!
//! Transformers checkpoints publish `config.json`; some official Mistral
//! releases publish only the short-key `params.json`. This module normalizes
//! that schema in memory without rewriting the checkpoint. Raw
//! Mistral-common weight files still require HF conversion and are rejected by
//! the Oxide weight-layout check, not silently treated as HF keys.

use serde_json::{json, Value};
use std::path::Path;

fn first(value: &Value, keys: &[&str]) -> Option<Value> {
    keys.iter().find_map(|key| value.get(*key).cloned())
}

/// Read `config.json`, or normalize `params.json` when the former is absent.
pub fn read_model_config(root: &Path) -> Result<Value, String> {
    let config = root.join("config.json");
    match std::fs::read(&config) {
        Ok(bytes) => {
            return serde_json::from_slice(&bytes)
                .map_err(|error| format!("advertised config.json is malformed: {error}"));
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(format!(
                "config.json exists but cannot be read (no params fallback): {error}"
            ));
        }
    }
    // Only an actual NotFound may select params.json. Permission, directory,
    // I/O and malformed-config cases are errors, never silent fallbacks.
    let params = root.join("params.json");
    let bytes = std::fs::read(&params)
        .map_err(|error| format!("config.json missing and params.json unavailable: {error}"))?;
    let value: Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("params.json is malformed: {error}"))?;
    normalize_params(&value)
}

/// Normalize the official Mistral short-key `params.json` into the nested
/// Transformers metadata view used by the service and Mistral3 loader.
pub fn normalize_params(value: &Value) -> Result<Value, String> {
    if value.get("text_config").is_some() {
        return Ok(value.clone());
    }
    let dim = first(value, &["dim", "hidden_size"]).ok_or("params.json lacks dim/hidden_size")?;
    let layers = first(value, &["n_layers", "num_hidden_layers"])
        .ok_or("params.json lacks n_layers/num_hidden_layers")?;
    let heads = first(value, &["n_heads", "num_attention_heads"])
        .ok_or("params.json lacks n_heads/num_attention_heads")?;
    let head_dim = first(value, &["head_dim"]).unwrap_or_else(|| {
        let dim = dim.as_u64().unwrap_or(0);
        let heads = heads.as_u64().unwrap_or(1).max(1);
        json!(dim / heads)
    });
    let max_positions = first(
        value,
        &["max_position_embeddings", "context_length", "max_seq_len"],
    )
    .unwrap_or_else(|| json!(32768));
    let text = json!({
        "model_type": "ministral3",
        "architectures": value.get("architectures").cloned().unwrap_or(json!(["Ministral3ForCausalLM"])),
        "hidden_size": dim,
        "intermediate_size": first(value, &["hidden_dim", "ffn_dim", "intermediate_size"])
            .ok_or("params.json lacks hidden_dim/ffn_dim/intermediate_size")?,
        "num_hidden_layers": layers,
        "num_attention_heads": heads,
        "num_key_value_heads": first(value, &["n_kv_heads", "num_key_value_heads"]).unwrap_or_else(|| heads.clone()),
        "head_dim": head_dim,
        "vocab_size": first(value, &["vocab_size"]).ok_or("params.json lacks vocab_size")?,
        "max_position_embeddings": max_positions,
        "rms_norm_eps": first(value, &["norm_eps", "rms_norm_eps"]).unwrap_or_else(|| json!(1e-5)),
        "rope_theta": first(value, &["rope_theta"]).unwrap_or_else(|| json!(1000000.0)),
        "rope_parameters": first(value, &["rope_parameters", "yarn"]),
        "llama_4_scaling": first(value, &["llama_4_scaling"]),
        "hidden_act": first(value, &["hidden_act"]).unwrap_or_else(|| json!("silu")),
        "is_causal": first(value, &["is_causal"]).unwrap_or_else(|| json!(true)),
        "use_bidirectional_attention": first(value, &["use_bidirectional_attention"]),
        "sliding_window": first(value, &["sliding_window"]),
        "dtype": first(value, &["dtype", "torch_dtype"]).unwrap_or_else(|| json!("bfloat16")),
    });
    let mut vision = first(value, &["vision_config", "vision_encoder"]).ok_or(
        "params.json lacks vision_config/vision_encoder; params-only text Mistral is not a Mistral3 multimodal checkpoint",
    )?;
    if vision.get("model_type").is_none() {
        vision["model_type"] = json!("pixtral");
    }
    if vision.get("image_size").is_none() {
        vision["image_size"] = json!(1540);
    }
    if vision.get("num_channels").is_none() {
        vision["num_channels"] = json!(3);
    }
    if vision.get("hidden_act").is_none() {
        vision["hidden_act"] = json!("silu");
    }
    if vision.get("rope_parameters").is_none() {
        vision["rope_parameters"] = json!({"rope_theta": 10000.0, "rope_type": "default"});
    }
    Ok(json!({
        "model_type": "mistral3",
        "architectures": value.get("architectures").cloned().unwrap_or(json!(["Mistral3ForConditionalGeneration"])),
        "dtype": first(value, &["dtype", "torch_dtype"]).unwrap_or_else(|| json!("bfloat16")),
        "torch_dtype": first(value, &["dtype", "torch_dtype"]),
        "max_position_embeddings": max_positions,
        "vocab_size": first(value, &["vocab_size"]),
        "image_token_index": first(value, &["image_token_index"]).unwrap_or_else(|| json!(10)),
        "spatial_merge_size": first(value, &["spatial_merge_size"]).unwrap_or_else(|| json!(2)),
        "vision_feature_layer": first(value, &["vision_feature_layer"]).unwrap_or_else(|| json!(-1)),
        "multimodal_projector_bias": first(value, &["multimodal_projector_bias"]).unwrap_or_else(|| json!(true)),
        "projector_hidden_act": first(value, &["projector_hidden_act"]).unwrap_or_else(|| json!("gelu")),
        "rope_theta": first(value, &["rope_theta"]),
        "id2label": first(value, &["id2label"]),
        "label2id": first(value, &["label2id"]),
        "num_labels": first(value, &["num_labels"]),
        "text_config": text,
        "vision_config": vision,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn params_only_14b_metadata_is_reachable() {
        let root = std::env::temp_dir().join(format!(
            "infinity-model-metadata-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("params.json"),
            serde_json::to_vec(&json!({
                "dim": 5120,
                "hidden_dim": 16384,
                "n_layers": 40,
                "n_heads": 32,
                "n_kv_heads": 8,
                "head_dim": 128,
                "vocab_size": 131072,
                "max_position_embeddings": 262144,
                "norm_eps": 1e-5,
                "rope_theta": 1000000000.0,
                "rope_parameters": {"rope_type":"yarn","factor":16.0,"original_max_position_embeddings":16384},
                "vision_encoder": {"hidden_size":1024,"intermediate_size":4096,"num_hidden_layers":24,"num_attention_heads":16,"head_dim":64,"patch_size":14}
            }))
            .unwrap(),
        )
        .unwrap();
        let config = read_model_config(&root).unwrap();
        assert_eq!(config["model_type"], "mistral3");
        assert_eq!(config["text_config"]["hidden_size"], 5120);
        assert_eq!(config["text_config"]["num_hidden_layers"], 40);
        assert_eq!(config["text_config"]["max_position_embeddings"], 262144);
        assert_eq!(config["vision_config"]["hidden_size"], 1024);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn malformed_or_unreadable_config_never_falls_back_to_params() {
        let root = std::env::temp_dir().join(format!(
            "infinity-model-metadata-bad-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("config.json"), b"{not-json").unwrap();
        std::fs::write(root.join("params.json"), b"{\"dim\":5120}").unwrap();
        assert!(read_model_config(&root).is_err());
        std::fs::remove_dir_all(&root).unwrap();

        let root = std::env::temp_dir().join(format!(
            "infinity-model-metadata-dir-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(root.join("config.json")).unwrap();
        std::fs::write(root.join("params.json"), b"{\"dim\":5120}").unwrap();
        assert!(read_model_config(&root).is_err());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn params_without_vision_is_rejected_instead_of_inventing_pixtral() {
        let root = std::env::temp_dir().join(format!(
            "infinity-model-metadata-novision-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("params.json"),
            br#"{"dim":5120,"hidden_dim":16384,"n_layers":40,"n_heads":32,"n_kv_heads":8,"head_dim":128,"vocab_size":131072}"#,
        )
        .unwrap();
        assert!(read_model_config(&root).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn config_json_wins_when_both_files_exist() {
        let root: PathBuf = std::env::temp_dir().join(format!(
            "infinity-model-metadata-both-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("config.json"),
            b"{\"model_type\":\"mistral3\",\"text_config\":{\"hidden_size\":4096}}",
        )
        .unwrap();
        std::fs::write(root.join("params.json"), b"{\"dim\":5120}").unwrap();
        let config = read_model_config(&root).unwrap();
        assert_eq!(config["text_config"]["hidden_size"], 4096);
        std::fs::remove_dir_all(root).unwrap();
    }
}
