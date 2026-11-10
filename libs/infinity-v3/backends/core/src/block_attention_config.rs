//! Block-attention metadata shared by the router and backends.
use serde_json::Value;
use std::path::Path;

/// Resolve model-owned block attention, rejecting incompatible readout metadata.
pub fn block_attention(path: &Path) -> Result<bool, String> {
    // Some decision models keep their backbone under encoder/ instead.
    let bytes = match std::fs::read(path.join("config.json")) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error.to_string()),
    };
    let config: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    resolve_config(&config)
}

fn resolve_config(config: &Value) -> Result<bool, String> {
    let current = config.get("modeling_overrides");
    let legacy = config.get("baseten_modeling_overrides");
    if let (Some(current), Some(legacy)) = (current, legacy) {
        if current != legacy {
            return Err("Conflicting decision modeling overrides".into());
        }
    }
    resolve(current.or(legacy))
}

fn resolve(overrides: Option<&Value>) -> Result<bool, String> {
    let Some(value) = overrides else {
        return Ok(false);
    };
    if !value.is_object()
        || value
            .get("readout")
            .is_some_and(|v| v.as_str() != Some("slot_lm"))
    {
        return Err("Decision metadata requires a slot_lm readout".into());
    }
    let attention = value
        .get("attention")
        .and_then(Value::as_object)
        .ok_or("Decision metadata requires an attention map")?;
    // Gemma checkpoints declare their causal non-full layers as sliding_attention;
    // Qwen checkpoints declare linear_attention. Other layer keys fail closed.
    for key in attention.keys() {
        if !matches!(
            key.as_str(),
            "full_attention" | "linear_attention" | "sliding_attention"
        ) {
            return Err(format!(
                "Unsupported attention layer key in model overrides: {key}"
            ));
        }
    }
    let mut non_full = false;
    for key in ["linear_attention", "sliding_attention"] {
        if let Some(mode) = attention.get(key) {
            non_full = true;
            if mode.as_str() != Some("causal") {
                return Err(format!("Decision metadata requires causal {key}"));
            }
        }
    }
    if !non_full {
        return Err("Decision metadata requires causal non-full attention".into());
    }
    let block = match attention.get("full_attention").and_then(Value::as_str) {
        Some("causal") => false,
        Some("block_bidirectional") => true,
        _ => return Err("Unsupported full-attention mode in model overrides".into()),
    };
    Ok(block)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn nested_encoder_layout_does_not_require_a_root_config() {
        let path = std::env::temp_dir().join(format!(
            "block-config-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(path.join("encoder")).unwrap();
        std::fs::write(
            path.join("encoder/config.json"),
            br#"{"model_type":"modernbert"}"#,
        )
        .unwrap();
        assert_eq!(block_attention(&path), Ok(false));
        std::fs::write(path.join("config.json"), b"invalid JSON").unwrap();
        assert!(block_attention(&path).is_err());
        std::fs::remove_file(path.join("config.json")).unwrap();
        std::fs::create_dir(path.join("config.json")).unwrap();
        assert!(block_attention(&path).is_err());
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn metadata_modes_fail_closed() {
        let overrides = json!({"attention":{"full_attention":"block_bidirectional","linear_attention":"causal"},"readout":"slot_lm"});
        assert_eq!(resolve(None), Ok(false));
        assert_eq!(resolve(Some(&overrides)), Ok(true));
        let causal = json!({"attention":{"full_attention":"causal","linear_attention":"causal"},"readout":"slot_lm"});
        assert_eq!(resolve(Some(&causal)), Ok(false));
        let gemma = json!({"attention":{"full_attention":"block_bidirectional","sliding_attention":"causal"},"readout":"slot_lm"});
        assert_eq!(resolve(Some(&gemma)), Ok(true));
        let gemma_causal = json!({"attention":{"full_attention":"causal","sliding_attention":"causal"},"readout":"slot_lm"});
        assert_eq!(resolve(Some(&gemma_causal)), Ok(false));
        for invalid in [
            json!(null),
            json!({}),
            json!({"attention":{"full_attention":"bidirectional","linear_attention":"causal"}}),
            json!({"attention":{"full_attention":"block_bidirectional","linear_attention":"noncausal"}}),
            json!({"attention":{"full_attention":"block_bidirectional","sliding_attention":"noncausal"}}),
            json!({"attention":{"full_attention":"block_bidirectional","sliding_attention":"causal","mamba":"causal"}}),
            json!({"attention":{"full_attention":"block_bidirectional"},"readout":"slot_lm"}),
            json!({"attention":{"full_attention":"block_bidirectional","linear_attention":"causal"},"readout":"pointer"}),
        ] {
            assert!(resolve(Some(&invalid)).is_err());
        }
    }

    #[test]
    fn current_and_legacy_metadata_select_the_same_attention() {
        let overrides = json!({"attention":{"full_attention":"block_bidirectional","linear_attention":"causal"},"readout":"slot_lm"});
        assert_eq!(resolve_config(&json!({})), Ok(false));
        assert_eq!(
            resolve_config(&json!({"modeling_overrides": overrides})),
            Ok(true)
        );
        assert_eq!(
            resolve_config(&json!({"baseten_modeling_overrides": overrides})),
            Ok(true)
        );
        assert_eq!(
            resolve_config(&json!({
                "modeling_overrides": overrides,
                "baseten_modeling_overrides": overrides
            })),
            Ok(true)
        );
        assert!(resolve_config(&json!({"modeling_overrides": null})).is_err());
        assert!(resolve_config(&json!({
            "modeling_overrides": overrides,
            "baseten_modeling_overrides": {
                "attention": {"full_attention": "causal", "linear_attention": "causal"}
            }
        }))
        .is_err());
    }
}
