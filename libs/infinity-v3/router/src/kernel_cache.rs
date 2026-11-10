use anyhow::{Context, Result};
use infinity_v3_backend::{BackendKind, TorchDevice};
use infinity_v3_router::ModelSettings;
use std::path::PathBuf;

// Called once from synchronous main before CUDA or Tokio can read these variables.
pub(super) fn configure(models: &[ModelSettings]) -> Result<Option<PathBuf>> {
    if !models.iter().any(|model| {
        model.backend_kind == BackendKind::Libtorch && model.torch_device != TorchDevice::Cpu
    }) {
        return Ok(None);
    }
    let Some(prefix) = [
        infinity_v3_config::infinity_env!("KERNEL_CACHE_PREFIX"),
        "INFINITY_KERNEL_CACHE_PREFIX",
        "KERNEL_CACHE_PREFIX",
    ]
    .into_iter()
    .find_map(std::env::var_os)
    .filter(|value| !value.is_empty()) else {
        return Ok(None);
    };
    let variables = ["CUDA_CACHE_PATH", "PYTORCH_KERNEL_CACHE_PATH"]
        .into_iter()
        .filter(|name| std::env::var_os(name).is_none())
        .collect::<Vec<_>>();
    if variables.is_empty() {
        return Ok(None);
    }
    let directory = std::path::absolute(prefix).context("Could not resolve kernel cache prefix")?;
    for variable in variables {
        std::env::set_var(variable, &directory);
    }
    Ok(Some(directory))
}
