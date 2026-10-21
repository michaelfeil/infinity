//! Recognize Perplexity exports by their saved architecture and module contract.
use serde_json::Value;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    Dense,
    ContextV1,
    LateV1,
    ContextV2,
    LateV2,
}

impl Profile {
    pub fn detect(config: &Value, modules: &Value) -> Option<Self> {
        match config["model_type"].as_str()? {
            "bidirectional_pplx_qwen3" => {
                if config["auto_map"]["AutoModel"].as_str()
                    == Some("modeling.PPLXQwen3ContextualModel")
                {
                    Some(Self::ContextV1)
                } else if modules.as_array().is_some_and(|modules| {
                    modules.iter().any(|m| m["type"] == "pylate.models.Dense.Dense")
                }) {
                    Some(Self::LateV1)
                } else {
                    Some(Self::Dense)
                }
            }
            "pplx_contextual_qwen3_5" => Some(Self::ContextV2),
            "qwen3_5" if modules.as_array().is_some_and(|modules| {
                modules.iter().any(|m| m["type"] == "sentence_transformers.multi_vector_encoder.modules.MultiVectorMask")
            }) => Some(Self::LateV2),
            _ => None,
        }
    }

    pub fn load(path: &Path) -> Result<Option<Self>, String> {
        let config: Value = serde_json::from_slice(
            &std::fs::read(path.join("config.json")).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        let modules = match std::fs::read(path.join("modules.json")) {
            Ok(bytes) => serde_json::from_slice(&bytes).map_err(|e| e.to_string())?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Value::Null,
            Err(e) => return Err(e.to_string()),
        };
        Ok(Self::detect(&config, &modules))
    }

    pub fn late(self) -> bool {
        matches!(self, Self::LateV1 | Self::LateV2)
    }
    pub fn contextual(self) -> bool {
        matches!(self, Self::ContextV1 | Self::ContextV2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn distinguishes_backbones_from_their_embedding_heads() {
        assert_eq!(
            Profile::detect(&json!({"model_type":"qwen3_5"}), &Value::Null),
            None
        );
        assert_eq!(
            Profile::detect(
                &json!({"model_type":"bidirectional_pplx_qwen3"}),
                &Value::Null
            ),
            Some(Profile::Dense)
        );
        assert_eq!(
            Profile::detect(
                &json!({"model_type":"bidirectional_pplx_qwen3","auto_map":{"AutoModel":"modeling.PPLXQwen3ContextualModel"}}),
                &Value::Null
            ),
            Some(Profile::ContextV1)
        );
        assert_eq!(
            Profile::detect(
                &json!({"model_type":"bidirectional_pplx_qwen3"}),
                &json!([{"type":"pylate.models.Dense.Dense"}])
            ),
            Some(Profile::LateV1)
        );
        assert_eq!(
            Profile::detect(
                &json!({"model_type":"pplx_contextual_qwen3_5"}),
                &Value::Null
            ),
            Some(Profile::ContextV2)
        );
        assert_eq!(
            Profile::detect(
                &json!({"model_type":"qwen3_5"}),
                &json!([{"type":"sentence_transformers.multi_vector_encoder.modules.MultiVectorMask"}])
            ),
            Some(Profile::LateV2)
        );
    }
}
