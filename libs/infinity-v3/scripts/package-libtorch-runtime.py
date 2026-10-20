"""Collect the native server, SDK libraries, and notices for Docker's runtime stage."""
# Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
import fnmatch
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Keep runtime-loaded modules that ELF DT_NEEDED cannot discover. In particular,
# cuDNN's engine libraries and NVRTC builtins are serving dependencies.
LAZY_LIBRARIES = (
    "libcudnn*.so*", "libcaffe2_nvrtc.so", "libtorch_cuda_linalg.so",
    "libnvrtc-builtins.so.*",
)
# Supplied by the runtime image's OS packages or the NVIDIA container runtime.
EXTERNAL_LIBRARIES = {
    "ld-linux-x86-64.so.2", "libc.so.6", "libm.so.6", "libdl.so.2",
    "libpthread.so.0", "librt.so.1", "libutil.so.1", "libgcc_s.so.1",
    "libstdc++.so.6", "libssl.so.3", "libcrypto.so.3", "libz.so.1",
    "libcuda.so.1", "libnvidia-ml.so.1",
}


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def needed(path):
    elf = subprocess.check_output(["readelf", "--dynamic", str(path)], text=True)
    return re.findall(r"\(NEEDED\).*?\[(.*?)\]", elf)


def runtime_libraries(roots, site_packages):
    candidates = {}
    for path in sorted([*(site_packages / "torch/lib").glob("*.so*"),
                        *(site_packages / "nvidia").rglob("*.so*")]):
        if "stubs" not in path.parts and path.name not in {
            "libcuda.so", "libcuda.so.1", "libtorch_python.so",
        }:
            candidates.setdefault(path.name, []).append(path)
    selected = {}

    def select(name):
        if name not in selected:
            paths = candidates[name]
            if len(paths) > 1 and len({digest(path) for path in paths}) != 1:
                raise RuntimeError(f"Conflicting runtime libraries: {name}")
            selected[name] = paths[0]
        return selected[name]

    pending = list(roots)
    provided = {path.name for path in roots}
    for pattern in LAZY_LIBRARIES:
        matches = fnmatch.filter(candidates, pattern)
        if not matches:
            raise RuntimeError(f"Missing runtime-loaded SDK library: {pattern}")
        pending.extend(select(name) for name in matches)
    visited = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        for name in needed(path):
            if name in provided:
                continue
            if name in candidates:
                pending.append(select(name))
            elif name not in EXTERNAL_LIBRARIES:
                raise RuntimeError(f"Unresolved runtime dependency {name} required by {path}")
    return [selected[name] for name in sorted(selected)]


def package_fa4(root, output, expected):
    if not root.is_dir():
        if os.environ.get("INFINITY_FA4_ROOT"):
            raise RuntimeError("InfinityFA4 was requested but its installed bundles are missing")
        return []
    for arch in expected:
        if not (root / f"sm_{arch.rstrip('a')}" / "manifest.json").is_file():
            raise RuntimeError(f"Missing InfinityFA4 bundle for {arch}")
    libraries = []
    shared = {}
    for bundle in sorted(root.iterdir()):
        if not bundle.is_dir() or not re.fullmatch(r"sm_[0-9]+", bundle.name):
            raise RuntimeError(f"Unexpected InfinityFA4 bundle: {bundle}")
        manifest = json.loads((bundle / "manifest.json").read_text())
        if manifest.get("name") != "InfinityFA4" or manifest.get("native_abi_version") != 1:
            raise RuntimeError(f"Unsupported InfinityFA4 bundle ABI: {bundle}")
        if manifest["architecture"] not in {bundle.name, bundle.name + "a"}:
            raise RuntimeError(f"InfinityFA4 architecture mismatch: {bundle}")
        if set(manifest["libraries"]) != {"libinfinityfa4.so", "libcute_dsl_runtime.so", "libtvm_ffi.so"}:
            raise RuntimeError(f"Incomplete InfinityFA4 runtime manifest: {bundle}")
        destination = output / bundle.name
        destination.mkdir(parents=True)
        shutil.copy2(bundle / "manifest.json", destination)
        for name, expected_hash in manifest["libraries"].items():
            library = bundle / name
            if digest(library) != expected_hash:
                raise RuntimeError(f"InfinityFA4 library does not match its manifest: {library}")
            libraries.append(library)
            key = (name, expected_hash)
            # Relative symlinks retain each bundle's $ORIGIN lookup and manifest
            # hashes while keeping identical CuTe/TVM runtimes only once.
            if name != "libinfinityfa4.so" and key in shared:
                (destination / name).symlink_to(os.path.relpath(shared[key], destination))
            else:
                shutil.copy2(library, destination / name)
                shared[key] = destination / name
    return libraries


def package(native, site_packages, output):
    bridge = native / "lib/libtei_torch.so"
    elf = subprocess.check_output(["cuobjdump", "--list-elf", str(bridge)], text=True)
    actual = set(re.findall(r"\.sm_(\d+[a-z]?)\.", elf))
    expected = {
        re.sub(r"-(real|virtual)$", "", arch)
        for arch in os.environ["LIBTORCH_CUDA_ARCHITECTURES"].split(";")
        if not arch.endswith("-virtual")
    }
    if not expected <= actual:
        raise RuntimeError(f"Native bridge lacks CUDA targets: {expected - actual}")
    for directory in ["bin", "lib", "licenses"]:
        (output / directory).mkdir(parents=True, exist_ok=True)
    server = Path("target/release/infinity_v3")
    fa4 = package_fa4(native / "lib/infinityfa4", output / "lib/infinityfa4", expected)
    libraries = runtime_libraries([server, bridge, *fa4], site_packages)
    shutil.copy2(server, output / "bin/infinity_v3")
    for source in [bridge, *libraries]:
        shutil.copy2(source, output / "lib" / source.name)
    shutil.copytree(native / "share/licenses/tei_torch", output / "licenses/native")
    for distribution in sorted(site_packages.glob("*.dist-info")):
        notices = distribution / "licenses"
        if notices.is_dir():
            shutil.copytree(notices, output / "licenses/dependencies" / distribution.name, dirs_exist_ok=True)
        for notice in distribution.iterdir():
            if notice.is_file() and notice.name.upper().startswith(("LICENSE", "NOTICE", "COPYING", "AUTHORS")):
                destination = output / "licenses/dependencies" / distribution.name
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copy2(notice, destination / notice.name)
    (output / "cuda-architectures.txt").write_text("\n".join(sorted(actual)) + "\n")
    (output / "sdk-requirements.txt").write_text(
        "".join(sorted(f"{dist.metadata['Name']}=={dist.version}\n"
                       for dist in importlib.metadata.distributions(path=[str(site_packages)]))))


def main(messages, site_packages, output):
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Runtime output must be empty: {output}")
    native = [
        Path(message["out_dir"])
        for line in messages.read_text().splitlines()
        if (message := json.loads(line)).get("reason") == "build-script-executed"
        and "infinity-backend-libtorch" in message["package_id"]
    ]
    if len(native) != 1:
        raise RuntimeError(f"Expected one native bridge build, got {native}")
    package(native[0], site_packages, output)


if __name__ == "__main__":
    main(*map(Path, sys.argv[1:]))
