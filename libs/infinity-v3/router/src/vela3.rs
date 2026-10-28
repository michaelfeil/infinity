//! Vela package dependencies and the boundary of native core-decision support.
use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::{Map, Value};
use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
};

// Keep Vela package layout handling out of the shared artifact downloader.
pub(crate) async fn download_artifacts(
    api: &hf_hub::api::tokio::ApiRepo,
    pool_config: bool,
) -> Result<PathBuf> {
    let config_path = api.get("config.json").await?;
    let config: Value = serde_json::from_slice(&std::fs::read(&config_path)?)?;
    let metadata: &[&str] = if config["model_type"] == "ultravbert" {
        &[
            "vela3v_config.json",
            "processor_config.json",
            "router_questions.json",
        ]
    } else {
        &["vela3_config.json", "router_questions.json"]
    };
    for file in metadata {
        api.get(file).await?;
    }
    if config["auto_map"]["AutoModel"] == "modeling_vela3.Vela3MiniModel" {
        // Wrapper packages contain no tokenizer; startup resolves their pinned D3 base.
        return Ok(config_path
            .parent()
            .context("Hub config path has no parent directory")?
            .to_path_buf());
    }
    infinity_v3_core::download::download_artifacts(api, pool_config)
        .await
        .map_err(Into::into)
}

#[derive(Deserialize)]
pub(crate) struct Base {
    pub repo_id: String,
    pub revision: String,
}

pub(crate) struct Policy {
    base: Option<Base>,
    // Keys are instruction texts for encoders, question IDs for D3 packages.
    // None blocks the key unconditionally; Some blocks only its trained wordings.
    routes: HashMap<String, Option<HashSet<String>>>,
}

impl Policy {
    pub fn load(root: &Path) -> Result<Self> {
        let vision = root.join("vela3v_config.json").exists();
        let config: Value = serde_json::from_slice(&std::fs::read(root.join(if vision {
            "vela3v_config.json"
        } else {
            "vela3_config.json"
        }))?)?;
        let base = config
            .get("base")
            .cloned()
            .map(serde_json::from_value::<Base>)
            .transpose()?;
        if let Some(base) = &base {
            anyhow::ensure!(
                config["format"] == "vela3/3",
                "Unsupported Vela3 package format"
            );
            anyhow::ensure!(
                base.repo_id.starts_with("vllm-sr/d3-")
                    && base.revision.len() == 40
                    && base.revision.bytes().all(|b| b.is_ascii_hexdigit()),
                "Vela3 requires an immutable D3 base revision"
            );
        } else {
            anyhow::ensure!(
                config["format"] == "vela3-encoder/1" || config["format"] == "vela3v-encoder/1",
                "Unsupported Vela3 encoder package format"
            );
        }
        let questions: Value =
            serde_json::from_slice(&std::fs::read(root.join("router_questions.json")).context(
                "Vela3 needs router_questions.json to distinguish trained router requests",
            )?)?;
        let by_wording = base.is_none();
        let routes = if by_wording {
            questions["questions"]
                .as_object()
                .into_iter()
                .flat_map(|m| m.keys())
                .map(|text| (normalize(text), None))
                .collect()
        } else {
            config["router_question_ids"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(|id| {
                    let wordings = (config["router_gate"] == "id_and_wording").then(|| {
                        let item = &questions["trained_questions"][id];
                        let variants: Vec<&Value> = if item.get("question").is_some() {
                            vec![item]
                        } else {
                            item.as_object()
                                .into_iter()
                                .flat_map(|m| m.values())
                                .collect()
                        };
                        variants
                            .into_iter()
                            .filter_map(|v| v["question"]["instructions"].as_str())
                            .map(normalize)
                            .collect()
                    });
                    (id.to_owned(), wordings)
                })
                .collect()
        };
        Ok(Self { base, routes })
    }

    pub fn base(&self) -> Option<&Base> {
        self.base.as_ref()
    }

    pub fn validate(&self, questions: &Map<String, Value>) -> Result<(), String> {
        for (id, q) in questions {
            let wording = q.get("instructions").and_then(Value::as_str).map(normalize);
            let key = if self.base.is_none() {
                wording.as_deref()
            } else {
                Some(id.as_str())
            };
            let router = key
                .and_then(|key| self.routes.get(key))
                .is_some_and(|texts| {
                    texts
                        .as_ref()
                        .is_none_or(|texts| wording.as_ref().is_some_and(|s| texts.contains(s)))
                });
            if router || q.get("decision_adapter") == Some(&Value::Bool(true)) {
                return Err("Vela3 task-adapted router questions and decision_adapter are not yet supported; use a general decision question".into());
            }
        }
        Ok(())
    }
}

fn normalize(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    #[ignore = "requires downloaded Vela3 checkpoints in VELA3_HUB_CACHE"]
    async fn checkpoint_packages_resolve_native_metadata() -> Result<()> {
        let api = hf_hub::api::tokio::ApiBuilder::from_env()
            .with_cache_dir(std::env::var("VELA3_HUB_CACHE")?.into())
            .with_progress(false)
            .build()?;
        for name in ["fast", "fast-vision", "lite", "mini", "flash", "turbo"] {
            let repo = api.model(format!("vllm-sr/vela3-{name}"));
            let root = download_artifacts(&repo, false).await?;
            let policy = Policy::load(&root)?;
            if let Some(base) = policy.base() {
                assert_eq!(base.repo_id, format!("vllm-sr/d3-{name}"));
            } else {
                assert!(root.join("tokenizer.json").is_file());
            }
        }
        Ok(())
    }

    #[test]
    fn task_adapted_questions_cannot_silently_use_the_base() {
        let mut policy = Policy {
            base: Some(Base {
                repo_id: "vllm-sr/d3-mini".into(),
                revision: "0".repeat(40),
            }),
            routes: HashMap::from([(
                "domain".into(),
                Some(HashSet::from(["Which domain?".into()])),
            )]),
        };
        let request = |id: &str, text: &str| {
            serde_json::from_value::<Map<String,Value>>(serde_json::json!({id:{"type":"choice","instructions":text,"criteria":{"a":null,"b":null}}})).unwrap()
        };
        assert!(policy
            .validate(&request("domain", "Which   domain? "))
            .is_err());
        assert!(policy
            .validate(&request("domain", "A general decision"))
            .is_ok());
        assert!(policy.validate(&request("q", "Which domain?")).is_ok());
        policy.base = None;
        policy.routes = HashMap::from([("Which domain?".into(), None)]);
        assert!(policy.validate(&request("q", "Which domain?")).is_err());
    }
}
