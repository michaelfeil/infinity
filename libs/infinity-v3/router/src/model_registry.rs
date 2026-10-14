//! Resident models selected by their public served names.
use crate::{ModelRuntime, ModelSettings};
use anyhow::{ensure, Result};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

#[derive(Clone)]
pub struct ModelRegistry {
    models: Arc<Vec<Arc<ModelRuntime>>>,
    names: Arc<HashMap<String, usize>>,
}

pub(crate) fn validate_configs(configs: &[ModelSettings]) -> Result<()> {
    ensure!(!configs.is_empty(), "At least one model is required");
    let mut names = HashSet::new();
    for config in configs {
        ensure!(
            !config.model_id.trim().is_empty(),
            "Model IDs must not be empty"
        );
        let (id, protocol) = crate::model_aliases::resolve(
            config.model_id.clone(),
            config.decision_protocol.clone(),
            config.pooling.as_ref(),
        )?;
        ensure!(
            protocol.is_none() || config.pooling.is_none(),
            "Model {}: decision-protocol conflicts with pooling",
            config.model_id
        );
        let name = config.served_model_name.as_deref().unwrap_or(&id);
        ensure!(
            !name.trim().is_empty(),
            "Served model names must not be empty"
        );
        ensure!(
            names.insert(name.to_owned()),
            "Duplicate served model name: {name}"
        );
        ensure!(
            config.max_batch_tokens > 0
                && config.max_concurrent_requests > 0
                && config.max_client_batch_size > 0
                && config.max_decision_questions > 0,
            "Model {name}: request and token limits must be positive"
        );
        ensure!(
            config.max_batch_requests != Some(0),
            "Model {name}: max-batch-requests must be positive"
        );
        ensure!(
            config.default_prompt.is_none() || config.default_prompt_name.is_none(),
            "Model {name}: default-prompt conflicts with default-prompt-name"
        );
        ensure!(
            config.device_id.is_none() || config.backend_device_ids.is_none(),
            "Model {name}: device-id conflicts with backend-device-ids"
        );
        use infinity_v3_backend::{BackendKind, TorchDevice};
        ensure!(
            match config.backend_kind {
                BackendKind::Candle => cfg!(feature = "candle"),
                BackendKind::Libtorch => cfg!(feature = "libtorch"),
            },
            "Model {name}: requested backend was not compiled"
        );
        let torch = config.backend_kind.resolve_device(config.torch_device)?;
        if configs.len() > 1 && config.backend_kind.supports_replication(torch) {
            ensure!(config.device_id.is_some() || config.backend_device_ids.is_some(), "Model {name}: multi-model CUDA deployments require explicit device-id or backend-device-ids");
        }
        crate::replicas::resolve_devices(
            config.backend_device_ids.as_deref(),
            config.device_id,
            config.backend_kind,
            torch,
        )?;
        if config.backend_kind == BackendKind::Libtorch && torch == TorchDevice::Cpu {
            ensure!(
                config.backend_device_ids.is_none(),
                "Model {name}: CPU does not support backend-device-ids"
            );
        }
    }
    Ok(())
}

impl ModelRegistry {
    pub fn new(models: Vec<Arc<ModelRuntime>>) -> Result<Self> {
        ensure!(
            !models.is_empty(),
            "At least one initialized model is required"
        );
        let mut names = HashMap::new();
        for (index, model) in models.iter().enumerate() {
            ensure!(
                names.insert(model.info.model_id.clone(), index).is_none(),
                "Duplicate served model name: {}",
                model.info.model_id
            );
        }
        Ok(Self {
            models: Arc::new(models),
            names: Arc::new(names),
        })
    }
    pub fn models(&self) -> &[Arc<ModelRuntime>] {
        &self.models
    }
    pub fn resolve(&self, name: Option<&str>) -> Result<Arc<ModelRuntime>, String> {
        // Preserve existing advisory names when exactly one model is deployed.
        let index = if self.models.len() == 1 {
            0
        } else {
            let name = name.ok_or_else(|| {
                "A model name is required when multiple models are deployed".to_string()
            })?;
            *self.names.get(name).ok_or_else(|| {
                format!(
                    "Unknown model {name:?}; available models: {}",
                    self.models
                        .iter()
                        .map(|m| m.info.model_id.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })?
        };
        Ok(self.models[index].clone())
    }
    pub async fn healthy(&self) -> bool {
        futures::future::join_all(self.models.iter().map(|m| m.infer.health()))
            .await
            .into_iter()
            .all(|healthy| healthy)
    }
    pub async fn shutdown(&self) {
        for model in self.models.iter() {
            model.shutdown().await;
        }
    }
}
