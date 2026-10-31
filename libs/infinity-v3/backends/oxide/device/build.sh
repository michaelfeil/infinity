#!/usr/bin/env bash
# Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
# Requires the pinned cargo-oxide compiler and a CUDA toolkit providing ptxas.
# Usage: CARGO_OXIDE=/path/to/cargo-oxide ./build.sh /path/to/kernel-bundle
set -euo pipefail
device_root="$(cd -- "$(dirname -- "$0")" && pwd)"
output_dir="${1:?Usage: build.sh OUTPUT_DIRECTORY}"
python3 - "$device_root" "$output_dir" "${CARGO_OXIDE:-cargo-oxide}" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

root, output = [Path(p).resolve() for p in sys.argv[1:3]]
compiler = sys.argv[3]
toolkit = os.environ.get("CUDA_TOOLKIT_PATH", os.environ.get("CUDA_HOME"))
ptxas = os.environ.get("PTXAS", str(Path(toolkit) / "bin/ptxas") if toolkit else "ptxas")
cuobjdump = os.environ.get("CUOBJDUMP", str(Path(toolkit) / "bin/cuobjdump") if toolkit else "cuobjdump")
environment = dict(os.environ, RUSTUP_TOOLCHAIN="1.99.1", RUSTC_BOOTSTRAP="1")
symbols = sorted(["infinity_attention", "infinity_embedding_norm", "infinity_gelu_bias",
                  "infinity_layer_norm", "infinity_pool", "infinity_qkv"])

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def command(args, log):
    start = time.perf_counter()
    with log.open("w") as stream:
        subprocess.run(args, cwd=root, env=environment, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)
    return time.perf_counter() - start

manifest = {
    "abi_version": 1,
    "compiler": "cuda-oxide",
    "upstream_revision": "999094fe568fc3080f80d49e9c44e3680ad1c968",
    "toolchain": "1.99.1",
    "compute_dtype": "bfloat16",
    "output_dtype": "float32",
    "sources": {name: digest(root / name) for name in
                ["Cargo.toml", "Cargo.lock", "rust-toolchain.toml", "src/main.rs"]},
    "kernels": symbols,
    "qualification": "Every architecture is independently compiled from Rust and assembled to a native cubin; hardware runtime validation is separate.",
    "architectures": [],
}
output.mkdir(parents=True, exist_ok=True)
for arch, hardware in [("sm_90", "H100"), ("sm_100", "B200"),
                       ("sm_120", "RTX PRO 6000 Blackwell")]:
    directory = output / arch
    directory.mkdir(exist_ok=True)
    build = [compiler, "build", "--arch", arch]
    build_seconds = command(build, directory / "build.log")
    ptx, cubin = directory / "kernels.ptx", directory / "kernels.cubin"
    shutil.copy2(root / "infinity_oxide_kernels.ptx", ptx)
    text = ptx.read_text()
    targets = re.findall(r"^\.target\s+(\w+)", text, re.MULTILINE)
    entries = sorted(re.findall(r"\.visible\s+\.entry\s+(\w+)\s*\(", text))
    if targets != [arch] or entries != symbols:
        raise RuntimeError(f"Incorrect compiler target or kernel exports: {arch}: {targets}, {entries}")
    assemble = [ptxas, "-arch=" + arch, "-v", str(ptx), "-o", str(cubin)]
    assemble_seconds = command(assemble, directory / "ptxas.log")
    binary_symbols = subprocess.check_output([cuobjdump, "--dump-elf-symbols", str(cubin)],
                                             text=True, env=environment)
    (directory / "symbols.log").write_text(binary_symbols)
    native_entries = sorted(re.findall(r"STO_ENTRY\s+(\w+)", binary_symbols))
    if native_entries != symbols:
        raise RuntimeError(f"Incorrect native cubin exports for {arch}: {native_entries}")
    manifest["architectures"].append({
        "architecture": arch, "hardware": hardware,
        "ptx_version": re.search(r"^\.version\s+([\d.]+)", text, re.MULTILINE)[1],
        "ptx": str(ptx.relative_to(output)), "cubin": str(cubin.relative_to(output)),
        "ptx_sha256": digest(ptx), "cubin_sha256": digest(cubin),
        "build_seconds": build_seconds, "assemble_seconds": assemble_seconds,
        "build_command": build, "assemble_command": assemble,
        "symbols_verified": native_entries,
    })
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"{arch} ({hardware}): all six Rust kernels compiled and assembled", flush=True)
PY
