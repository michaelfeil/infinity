"""Copy notices from packages reported by Cargo's JSON build output."""
# Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tomllib


def is_notice(name):
    upper = name.upper()
    return any(marker in upper for marker in (
        "LICENSE", "LICENCE", "LICENSING", "NOTICE", "COPYING", "AUTHORS", "COPYRIGHT",
    ))


messages, output = map(Path, sys.argv[1:])
packages = {
    record["package_id"]: Path(record["manifest_path"])
    for line in messages.read_text().splitlines()
    if (record := json.loads(line)).get("reason") == "compiler-artifact"
}
if not packages:
    raise RuntimeError("Cargo reported no packages; refusing to omit dependency notices")
output.mkdir(parents=True, exist_ok=True)
index = []
for package_id, manifest in sorted(packages.items()):
    root = manifest.parent
    package = tomllib.loads(manifest.read_text()).get("package", {})
    destination = output / (root.name + "-" + hashlib.sha256(package_id.encode()).hexdigest()[:12])
    candidates = {}
    for directory, subdirectories, files in os.walk(root):
        subdirectories[:] = [
            name for name in subdirectories
            if name not in {".git", "target", "node_modules", "__pycache__"}
            and (Path(directory) / name).resolve() != output.resolve()
        ]
        for name in files:
            source = Path(directory) / name
            relative = source.relative_to(root)
            if is_notice(name) or any(
                part.upper() == "LICENSES" for part in relative.parts[:-1]
            ):
                candidates[relative] = source
    # Git/workspace packages often keep their shared license above the crate.
    for parent in (root, *root.parents):
        workspace_manifest = parent / "Cargo.toml"
        if not workspace_manifest.is_file():
            continue
        workspace = tomllib.loads(workspace_manifest.read_text()).get("workspace")
        if workspace is None:
            continue
        for source in parent.iterdir():
            if source.is_file() and is_notice(source.name):
                candidates[Path("workspace") / source.name] = source
            elif source.is_dir() and source.name.upper() == "LICENSES":
                for notice in source.rglob("*"):
                    if notice.is_file():
                        candidates[Path("workspace") / notice.relative_to(parent)] = notice
        license_file = package.get("license-file", package.get("license_file"))
        if isinstance(license_file, dict) and license_file.get("workspace"):
            workspace_package = workspace.get("package", {})
            license_file = workspace_package.get("license-file", workspace_package.get("license_file"))
            if isinstance(license_file, str):
                source = parent / license_file
                candidates[Path("workspace") / source.name] = source
        break
    license_file = package.get("license-file", package.get("license_file"))
    if isinstance(license_file, str):
        source = root / license_file
        candidates[Path("declared") / source.name] = source
    # Some published crates put license/authorship information only in their
    # manifest and README rather than shipping a separate license document.
    if not candidates:
        for source in root.iterdir():
            if source.is_file() and source.name.upper().startswith("README"):
                candidates[source.relative_to(root)] = source
    candidates[Path("Cargo.toml")] = manifest
    for relative, source in sorted(candidates.items()):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    index.append({
        "package": package_id,
        "directory": destination.name,
        "notices": [str(path) for path in sorted(candidates)],
    })
(output / "index.json").write_text(json.dumps(index, indent=2) + "\n")
print(f"Collected Cargo notices for {len(index)} build packages")
