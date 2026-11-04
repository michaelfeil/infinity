use anyhow::{bail, ensure, Result};
use infinity_v3_backend::PythonCapabilities;
use std::collections::HashSet;
use std::future::Future;

/// Poll every replica's startup concurrently, preserving device order. Drain all
/// startups before returning an error so native initialization cannot outlive
/// startup failure and successful replicas are dropped together.
pub(crate) async fn initialize_all<T, F>(
    initializers: impl IntoIterator<Item = F>,
) -> Result<Vec<T>>
where
    F: Future<Output = Result<T>>,
{
    futures::future::join_all(initializers)
        .await
        .into_iter()
        .collect()
}

pub(crate) fn resolve_devices(
    value: Option<&str>,
    device: Option<usize>,
    kind: infinity_v3_backend::BackendKind,
    torch: infinity_v3_backend::TorchDevice,
    python: Option<PythonCapabilities>,
) -> Result<Vec<usize>> {
    use infinity_v3_backend::{BackendKind, TorchDevice};
    if kind == BackendKind::Libtorch && torch == TorchDevice::Cpu {
        ensure!(device.is_none_or(|id| id == 0), "CPU uses device index 0");
    }
    if kind == BackendKind::PythonBackend {
        let capabilities = python.unwrap_or_default();
        let count = capabilities.visible_device_count;
        ensure!(count > 0, "Python backend reports no visible devices");
        if !capabilities.supports_replication {
            ensure!(
                value.is_none(),
                "backend-device-ids requires a replicated backend"
            );
            let device = device.unwrap_or(0);
            ensure!(
                device < count,
                "Device {device} is not visible (visible device count: {count})"
            );
            return Ok(vec![device]);
        }
        return resolve_devices_with(value, device, true, || Ok(count));
    }
    resolve_devices_with(
        value,
        device,
        kind.supports_replication(torch, python),
        || Ok(kind.visible_device_count(torch, python)?),
    )
}

fn resolve_devices_with(
    value: Option<&str>,
    device: Option<usize>,
    replication: bool,
    visible_count: impl FnOnce() -> Result<usize>,
) -> Result<Vec<usize>> {
    if !replication {
        ensure!(
            value.is_none(),
            "backend-device-ids requires a replicated backend"
        );
        return Ok(vec![device.unwrap_or(0)]);
    }
    let count = visible_count()?;
    match (value, device) {
        (Some(value), _) => parse_devices(value, count),
        (None, Some(device)) => parse_devices(&device.to_string(), count),
        (None, None) => parse_devices("auto", count),
    }
}

fn parse_devices(value: &str, count: usize) -> Result<Vec<usize>> {
    let devices = if value.trim() == "auto" {
        (0..count).collect::<Vec<_>>()
    } else {
        value
            .split(',')
            .map(|s| s.trim().parse::<usize>())
            .collect::<Result<Vec<_>, _>>()?
    };
    ensure!(!devices.is_empty(), "No visible devices selected");
    let mut seen = HashSet::new();
    for &device in &devices {
        ensure!(
            device < count,
            "Device {device} is not visible (visible device count: {count})"
        );
        if !seen.insert(device) {
            bail!("Duplicate device {device}");
        }
    }
    Ok(devices)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn visible_devices_and_explicit_override() {
        assert_eq!(
            resolve_devices_with(None, None, true, || Ok(8)).unwrap(),
            (0..8).collect::<Vec<_>>()
        );
        assert_eq!(
            resolve_devices_with(None, Some(2), true, || Ok(8)).unwrap(),
            vec![2]
        );
        assert_eq!(parse_devices("2,0", 8).unwrap(), vec![2, 0]);
        assert!(parse_devices("0,0", 8).is_err());
        assert!(parse_devices("8", 8).is_err());
        assert_eq!(
            resolve_devices_with(None, None, false, || panic!("CPU must not query CUDA")).unwrap(),
            vec![0]
        );
        use infinity_v3_backend::{BackendKind, TorchDevice};
        assert!(!BackendKind::Libtorch.supports_replication(TorchDevice::Cpu, None));
        assert!(BackendKind::Libtorch.supports_replication(TorchDevice::Cuda, None));
        assert_eq!(
            resolve_devices(None, None, BackendKind::Libtorch, TorchDevice::Cpu, None).unwrap(),
            vec![0]
        );
        assert!(resolve_devices(
            Some("auto"),
            None,
            BackendKind::Libtorch,
            TorchDevice::Cpu,
            None
        )
        .is_err());
        assert!(
            resolve_devices(None, Some(1), BackendKind::Libtorch, TorchDevice::Cpu, None).is_err()
        );
    }

    #[test]
    fn python_capabilities_control_replica_placement() {
        use infinity_v3_backend::{BackendKind, TorchDevice};
        let kind = BackendKind::PythonBackend;
        let torch = TorchDevice::Auto;
        let single = PythonCapabilities::default();
        assert_eq!(
            resolve_devices(None, None, kind, torch, Some(single)).unwrap(),
            vec![0]
        );
        assert!(resolve_devices(Some("auto"), None, kind, torch, Some(single)).is_err());
        assert!(resolve_devices(None, Some(1), kind, torch, Some(single)).is_err());

        let manual = PythonCapabilities {
            visible_device_count: 4,
            supports_replication: false,
        };
        assert_eq!(
            resolve_devices(None, Some(2), kind, torch, Some(manual)).unwrap(),
            vec![2]
        );
        assert!(resolve_devices(Some("0,1"), None, kind, torch, Some(manual)).is_err());

        let multi = PythonCapabilities {
            visible_device_count: 3,
            supports_replication: true,
        };
        assert_eq!(
            resolve_devices(None, None, kind, torch, Some(multi)).unwrap(),
            vec![0, 1, 2]
        );
        assert_eq!(
            resolve_devices(Some("2,0"), None, kind, torch, Some(multi)).unwrap(),
            vec![2, 0]
        );
        assert_eq!(
            resolve_devices(None, Some(1), kind, torch, Some(multi)).unwrap(),
            vec![1]
        );
        assert!(resolve_devices(Some("3"), None, kind, torch, Some(multi)).is_err());
        assert!(resolve_devices(Some("0,0"), None, kind, torch, Some(multi)).is_err());
    }
}
