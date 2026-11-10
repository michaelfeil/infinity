use anyhow::{ensure, Context, Result};
use infinity_v3_backend::{BackendKind, TorchDevice};
use infinity_v3_router::ModelSettings;
use std::{collections::BTreeSet, path::Path, path::PathBuf, process::Command};

// Called from synchronous main before CUDA or Tokio can read these variables.
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
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .context("Could not run nvidia-smi for kernel cache setup")?;
    ensure!(
        output.status.success(),
        "nvidia-smi failed ({}): {}",
        output.status,
        String::from_utf8_lossy(&output.stderr).trim()
    );
    let prefix = std::path::absolute(prefix).context("Could not resolve kernel cache prefix")?;
    let directory = cache_directory(&prefix, std::str::from_utf8(&output.stdout)?)?;
    std::fs::create_dir_all(&directory).with_context(|| {
        format!(
            "Could not create kernel cache directory {}",
            directory.display()
        )
    })?;
    for variable in variables {
        std::env::set_var(variable, &directory);
    }
    Ok(Some(directory))
}

fn cache_directory(prefix: &Path, metadata: &str) -> Result<PathBuf> {
    let mut driver = None;
    let mut architectures = BTreeSet::new();
    for row in metadata.lines().filter(|row| !row.trim().is_empty()) {
        let (version, capability) = row
            .split_once(',')
            .context("Invalid nvidia-smi GPU metadata")?;
        let version = version.trim();
        ensure!(
            version.split('.').all(decimal),
            "Invalid NVIDIA driver version"
        );
        ensure!(
            driver.unwrap_or(version) == version,
            "Multiple NVIDIA driver versions"
        );
        driver = Some(version);
        let (major, minor) = capability
            .trim()
            .split_once('.')
            .context("Invalid GPU compute capability")?;
        ensure!(
            decimal(major) && decimal(minor),
            "Invalid GPU compute capability"
        );
        let (major, minor) = (major.parse::<u32>()?, minor.parse::<u32>()?);
        ensure!(major > 0 && minor < 10, "Invalid GPU compute capability");
        architectures.insert((major, minor));
    }
    let sm = architectures
        .into_iter()
        .map(|(major, minor)| format!("sm{major}{minor}"))
        .collect::<Vec<_>>()
        .join("-");
    Ok(prefix
        .join(driver.context("nvidia-smi reported no GPUs")?)
        .join(sm))
}

fn decimal(value: &str) -> bool {
    !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn namespace_tracks_driver_and_sorted_gpu_architectures() {
        for (metadata, suffix) in [
            ("580.126.09, 10.0\n", "580.126.09/sm100"),
            ("580.1, 10.0\n580.1, 9.0\n580.1, 9.0\n", "580.1/sm90-sm100"),
        ] {
            assert_eq!(
                cache_directory(Path::new("/cache"), metadata).unwrap(),
                Path::new("/cache").join(suffix)
            );
        }
    }

    #[test]
    fn malformed_metadata_cannot_create_a_cache_namespace() {
        for metadata in [
            "",
            "../580, 9.0",
            "580..1, 9.0",
            "580.1, N/A",
            "580.1, 0.0",
            "580.1, 9.10",
            "580.1, 9.0\n581.1, 9.0",
        ] {
            assert!(
                cache_directory(Path::new("/cache"), metadata).is_err(),
                "{metadata}"
            );
        }
    }
}
