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
    resolve(config.get("baseten_modeling_overrides"))
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
    let attention = &value["attention"];
    if attention["linear_attention"].as_str() != Some("causal") {
        return Err("Decision metadata requires causal linear attention".into());
    }
    let block = match attention["full_attention"].as_str() {
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
        for invalid in [
            json!(null),
            json!({}),
            json!({"attention":{"full_attention":"bidirectional","linear_attention":"causal"}}),
            json!({"attention":{"full_attention":"block_bidirectional","linear_attention":"noncausal"}}),
            json!({"attention":{"full_attention":"block_bidirectional","linear_attention":"causal"},"readout":"pointer"}),
        ] {
            assert!(resolve(Some(&invalid)).is_err());
        }
    }
}
