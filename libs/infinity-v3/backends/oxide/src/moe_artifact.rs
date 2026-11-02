// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
// SPDX-License-Identifier: LicenseRef-Infinity-Restricted-MIT-1.0
//! Cold, architecture-specific sidecar resolution under the existing kernel bundle.

use crate::model::start;
use cuda_core::Context;
use infinity_v3_backend_core::BackendError;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

#[derive(Deserialize)]
struct Manifest {
    abi_version: u32,
    targets: Vec<Target>,
}
#[derive(Deserialize)]
struct Target {
    arch: i32,
    library: String,
    library_sha256: String,
}

fn verify(root: &Path, arch: i32) -> Result<PathBuf, BackendError> {
    let manifest: Manifest = crate::model::read_json(&root.join("BUILD_MANIFEST.json"))?;
    if manifest.abi_version != 1 {
        return Err(start("Unsupported MoE sidecar manifest ABI"));
    }
    let mut matches = manifest.targets.iter().filter(|target| target.arch == arch);
    let target = matches
        .next()
        .ok_or_else(|| start("MoE sidecar has no target for this GPU"))?;
    if matches.next().is_some() || !matches!(arch, 90 | 100 | 120) {
        return Err(start("Invalid or duplicate MoE sidecar architecture"));
    }
    // Fixed local names prevent absolute paths, parent traversal and alternate ABI files.
    if target.library != format!("liboxide_moe_sm{arch}.so")
        || target.library_sha256.len() != 64
        || !target
            .library_sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(start("Invalid MoE sidecar library name or SHA256"));
    }
    let path = root.join(&target.library);
    let bytes = std::fs::read(&path).map_err(start)?;
    let actual = format!("{:x}", Sha256::digest(&bytes));
    if actual != target.library_sha256 {
        return Err(start("MoE sidecar library SHA256 mismatch"));
    }
    Ok(path)
}

fn bundle_root(module: &Path, arch: i32) -> Result<PathBuf, BackendError> {
    if module.is_dir() {
        return Ok(module.to_path_buf());
    }
    let parent = module
        .parent()
        .ok_or_else(|| start("Kernel module has no sidecar parent"))?;
    if parent.file_name() == Some(std::ffi::OsStr::new(&format!("sm_{arch}"))) {
        return parent
            .parent()
            .map(Path::to_path_buf)
            .ok_or_else(|| start("Kernel architecture directory has no bundle parent"));
    }
    Ok(parent.to_path_buf())
}

pub(crate) fn resolve(context: &Arc<Context>) -> Result<PathBuf, BackendError> {
    let module = std::env::var_os(infinity_v3_config::infinity_env!("OXIDE_KERNEL_MODULE"))
        .ok_or_else(|| start("MoE requires the existing INFINITY_OXIDE_KERNEL_MODULE bundle"))?;
    let module = PathBuf::from(module);
    let (major, minor) = context.compute_capability().map_err(start)?;
    let arch = major
        .checked_mul(10)
        .and_then(|major| major.checked_add(minor))
        .ok_or_else(|| start("MoE architecture overflow"))?;
    let root = bundle_root(&module, arch)?;
    verify(&root.join("moe"), arch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    struct Directory(PathBuf);
    impl Drop for Directory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    #[test]
    fn architecture_cubin_and_flat_cubin_find_the_same_bundle_sidecar() {
        let bundle = Path::new("/tmp/no-created-moe-bundle-path");
        assert_eq!(
            bundle_root(&bundle.join("sm_90/kernels.cubin"), 90).unwrap(),
            bundle
        );
        assert_eq!(
            bundle_root(&bundle.join("kernels.cubin"), 90).unwrap(),
            bundle
        );
        assert_eq!(
            bundle_root(&bundle.join("sm_120/kernels.cubin"), 120).unwrap(),
            bundle
        );
    }

    #[test]
    fn sidecar_rejects_changed_library_duplicate_architecture_and_path_escape() {
        let root = Directory(std::env::temp_dir().join(format!(
            "oxide-moe-sidecar-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        )));
        std::fs::create_dir(&root.0).unwrap();
        let library = root.0.join("liboxide_moe_sm90.so");
        std::fs::write(&library, b"CPU-only artifact integrity fixture").unwrap();
        let target = serde_json::json!({"arch":90,"library":"liboxide_moe_sm90.so","library_sha256":format!("{:x}", Sha256::digest(std::fs::read(&library).unwrap()))});
        let save = |targets: serde_json::Value| {
            std::fs::write(
                root.0.join("BUILD_MANIFEST.json"),
                serde_json::json!({"abi_version":1,"targets":targets}).to_string(),
            )
            .unwrap()
        };
        save(serde_json::json!([target]));
        assert_eq!(verify(&root.0, 90).unwrap(), library);
        std::fs::write(&library, b"changed artifact").unwrap();
        assert!(verify(&root.0, 90).is_err());
        save(serde_json::json!([target, target]));
        assert!(verify(&root.0, 90).is_err());
        let mut escaped = target.clone();
        escaped["library"] = "../liboxide_moe_sm90.so".into();
        save(serde_json::json!([escaped]));
        assert!(verify(&root.0, 90).is_err());
        assert!(verify(&root.0, 100).is_err());
    }
}
