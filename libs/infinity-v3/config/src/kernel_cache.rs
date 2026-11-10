//! Opt-in, process-wide NVIDIA kernel-cache directory derivation.
use std::collections::BTreeSet;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

const PREFIX_LITERAL: &str = "INFINITY_KERNEL_CACHE_PREFIX";
const PREFIX_UNPREFIXED: &str = "KERNEL_CACHE_PREFIX";

/// A derived cache directory and the environment variables callers must still set.
#[derive(Debug)]
pub struct KernelCache {
    pub directory: PathBuf,
    pub variables: Vec<(&'static str, PathBuf)>,
}

/// Derive the kernel-cache directory without mutating the process environment.
pub fn prepare() -> Result<Option<KernelCache>, String> {
    prepare_with(
        |name| std::env::var_os(name),
        std::env::current_dir,
        probe_nvidia_smi,
        |directory| std::fs::create_dir_all(directory),
    )
}

fn probe_nvidia_smi() -> Result<String, String> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .map_err(|error| {
            format!("Could not run nvidia-smi to derive kernel cache path: {error}")
        })?;
    if !output.status.success() {
        return Err(format!(
            "nvidia-smi failed with status {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    String::from_utf8(output.stdout)
        .map_err(|error| format!("nvidia-smi output was not valid UTF-8: {error}"))
}

fn prepare_with<L, C, P, M>(
    lookup: L,
    current_dir: C,
    probe: P,
    create_dir: M,
) -> Result<Option<KernelCache>, String>
where
    L: Fn(&str) -> Option<OsString>,
    C: FnOnce() -> std::io::Result<PathBuf>,
    P: FnOnce() -> Result<String, String>,
    M: FnOnce(&Path) -> std::io::Result<()>,
{
    let Some(prefix) = prefix_with(infinity_env!("KERNEL_CACHE_PREFIX"), &lookup) else {
        return Ok(None);
    };
    if prefix.is_empty() {
        return Ok(None);
    }
    let cuda = lookup("CUDA_CACHE_PATH");
    let pytorch = lookup("PYTORCH_KERNEL_CACHE_PATH");
    if cuda.is_some() && pytorch.is_some() {
        return Ok(None);
    }

    let mut directory = PathBuf::from(prefix);
    if !directory.is_absolute() {
        directory = current_dir()
            .map_err(|error| format!("Could not resolve relative kernel cache prefix: {error}"))?
            .join(directory);
    }
    let (driver, sm) = parse_probe(&probe()?)?;
    directory.push(driver);
    directory.push(sm);
    create_dir(&directory).map_err(|error| {
        format!(
            "Could not create kernel cache directory {}: {error}",
            directory.display()
        )
    })?;

    let mut variables = Vec::with_capacity(2);
    if cuda.is_none() {
        variables.push(("CUDA_CACHE_PATH", directory.clone()));
    }
    if pytorch.is_none() {
        variables.push(("PYTORCH_KERNEL_CACHE_PATH", directory.clone()));
    }
    Ok(Some(KernelCache {
        directory,
        variables,
    }))
}

fn prefix_with<L>(compiled: &str, lookup: &L) -> Option<OsString>
where
    L: Fn(&str) -> Option<OsString>,
{
    [
        Some(compiled),
        (compiled != PREFIX_LITERAL).then_some(PREFIX_LITERAL),
        Some(PREFIX_UNPREFIXED),
    ]
    .into_iter()
    .flatten()
    .find_map(lookup)
}

fn parse_probe(output: &str) -> Result<(String, String), String> {
    let mut driver: Option<String> = None;
    let mut capabilities = BTreeSet::new();
    for (index, line) in output.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let row = index + 1;
        let (row_driver, capability) = line.split_once(',').ok_or_else(|| {
            format!("nvidia-smi row {row} is not driver,compute_cap CSV: {line:?}")
        })?;
        let row_driver = row_driver.trim();
        validate_driver(row_driver).map_err(|error| format!("nvidia-smi row {row}: {error}"))?;
        let capability = parse_capability(capability.trim())
            .map_err(|error| format!("nvidia-smi row {row}: {error}"))?;
        if let Some(expected) = &driver {
            if expected != row_driver {
                return Err(format!(
                    "nvidia-smi reported multiple driver versions: {expected} and {row_driver}"
                ));
            }
        } else {
            driver = Some(row_driver.to_owned());
        }
        capabilities.insert(capability);
    }
    let driver = driver.ok_or_else(|| "nvidia-smi reported no GPUs".to_owned())?;
    let sm = capabilities
        .into_iter()
        .map(|(major, minor)| format!("sm{major}{minor}"))
        .collect::<Vec<_>>()
        .join("-");
    Ok((driver, sm))
}

fn validate_driver(driver: &str) -> Result<(), String> {
    if driver.is_empty()
        || driver
            .split('.')
            .any(|part| part.is_empty() || !part.bytes().all(|byte| byte.is_ascii_digit()))
    {
        return Err(format!("invalid driver version {driver:?}"));
    }
    Ok(())
}

fn parse_capability(value: &str) -> Result<(u32, u32), String> {
    let (major, minor) = value
        .split_once('.')
        .ok_or_else(|| format!("invalid compute capability {value:?}"))?;
    let parse = |part: &str| {
        if part.is_empty() || !part.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(format!("invalid compute capability {value:?}"));
        }
        part.parse::<u32>()
            .map_err(|_| format!("invalid compute capability {value:?}"))
    };
    let capability = (parse(major)?, parse(minor)?);
    if capability.0 == 0 || capability.1 > 9 {
        return Err(format!("invalid compute capability {value:?}"));
    }
    Ok(capability)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PREFIX: &str = infinity_env!("KERNEL_CACHE_PREFIX");

    fn probe_failure() -> Result<String, String> {
        Err("probe must not run".into())
    }

    fn unused_cwd() -> std::io::Result<PathBuf> {
        Err(std::io::Error::other("unused"))
    }

    fn run<L, P>(lookup: L, probe: P) -> Result<Option<KernelCache>, String>
    where
        L: Fn(&str) -> Option<OsString>,
        P: FnOnce() -> Result<String, String>,
    {
        prepare_with(lookup, unused_cwd, probe, |_| Ok(()))
    }

    fn cache_lookup() -> impl Fn(&str) -> Option<OsString> {
        |name| (name == PREFIX).then(|| OsString::from("/cache"))
    }

    #[test]
    fn disabled_or_fully_overridden_prefix_does_not_probe() {
        assert!(run(|_| None, probe_failure).unwrap().is_none());
        assert!(
            run(|name| (name == PREFIX).then(OsString::new), probe_failure)
                .unwrap()
                .is_none()
        );
        let both = |name: &str| match name {
            "CUDA_CACHE_PATH" | "PYTORCH_KERNEL_CACHE_PATH" => Some(OsString::new()),
            _ if name == PREFIX => Some("/cache".into()),
            _ => None,
        };
        assert!(run(both, probe_failure).unwrap().is_none());
    }

    #[test]
    fn derives_path_and_returns_only_missing_variables() {
        let expected = PathBuf::from("/cache/535.183.01/sm90");
        let partial = |name: &str| match name {
            "CUDA_CACHE_PATH" => Some(OsString::new()),
            _ if name == PREFIX => Some("/cache".into()),
            _ => None,
        };
        let cache = prepare_with(
            partial,
            unused_cwd,
            || Ok("535.183.01, 9.0\n".into()),
            |directory| {
                assert_eq!(directory, expected);
                Ok(())
            },
        )
        .unwrap()
        .unwrap();
        assert_eq!(cache.directory, expected);
        assert_eq!(
            cache.variables,
            vec![("PYTORCH_KERNEL_CACHE_PATH", expected.clone())]
        );

        let cache = run(cache_lookup(), || Ok("535.183.01, 9.0\n".into()))
            .unwrap()
            .unwrap();
        assert_eq!(
            cache.variables,
            vec![
                ("CUDA_CACHE_PATH", expected.clone()),
                ("PYTORCH_KERNEL_CACHE_PATH", expected),
            ]
        );
    }

    #[test]
    fn prefix_precedence_supports_custom_base_and_literal_fallbacks() {
        let custom = "BEI_KERNEL_CACHE_PREFIX";
        let only = |target: &'static str, value: &'static str| {
            move |name: &str| (name == target).then(|| OsString::from(value))
        };
        for (target, value) in [
            (custom, "/custom"),
            (PREFIX_LITERAL, "/literal"),
            (PREFIX_UNPREFIXED, "/unprefixed"),
        ] {
            assert_eq!(
                prefix_with(custom, &only(target, value)),
                Some(OsString::from(value))
            );
        }
    }

    #[test]
    fn mixed_architectures_are_sorted_and_deduplicated() {
        let cache = run(cache_lookup(), || {
            Ok("535.1, 10.0\n535.1, 9.0\n535.1, 9.0\n535.1, 8.6\n".into())
        })
        .unwrap()
        .unwrap();
        assert_eq!(
            cache.directory,
            PathBuf::from("/cache/535.1/sm86-sm90-sm100")
        );
    }

    #[test]
    fn malformed_or_failed_probes_are_reported() {
        for output in ["535..1, 9.0\n", "535.1, 9\n", "535.1, 9.x\n"] {
            let error = run(cache_lookup(), || Ok(output.into())).unwrap_err();
            assert!(error.contains("nvidia-smi row 1"), "{error}");
        }
        let error = run(cache_lookup(), || Err("command failed".into())).unwrap_err();
        assert!(error.contains("command failed"), "{error}");
        let error = run(cache_lookup(), || Ok(String::new())).unwrap_err();
        assert!(error.contains("no GPUs"), "{error}");
    }
}
