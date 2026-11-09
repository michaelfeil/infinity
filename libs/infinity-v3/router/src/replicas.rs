use anyhow::{bail, ensure, Result};
use infinity_v3_backend::DeviceCapabilities;
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
    capabilities: DeviceCapabilities,
) -> Result<Vec<usize>> {
    let count = capabilities.visible_device_count;
    ensure!(count > 0, "Backend reports no visible devices");
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
    fn generic_capabilities_select_devices() {
        let single = DeviceCapabilities::default();
        assert_eq!(resolve_devices(None, None, single).unwrap(), vec![0]);
        assert_eq!(resolve_devices(None, Some(0), single).unwrap(), vec![0]);
        assert!(resolve_devices(Some("auto"), None, single).is_err());
        assert!(resolve_devices(None, Some(1), single).is_err());

        let manual = DeviceCapabilities {
            visible_device_count: 4,
            supports_replication: false,
        };
        assert_eq!(resolve_devices(None, Some(2), manual).unwrap(), vec![2]);
        assert!(resolve_devices(Some("0,1"), None, manual).is_err());

        let replicated = DeviceCapabilities {
            visible_device_count: 3,
            supports_replication: true,
        };
        assert_eq!(
            resolve_devices(None, None, replicated).unwrap(),
            vec![0, 1, 2]
        );
        assert_eq!(
            resolve_devices(Some("2,0"), None, replicated).unwrap(),
            vec![2, 0]
        );
        assert_eq!(resolve_devices(None, Some(1), replicated).unwrap(), vec![1]);
        assert!(resolve_devices(Some("3"), None, replicated).is_err());
        assert!(resolve_devices(Some("0,0"), None, replicated).is_err());
    }
}
