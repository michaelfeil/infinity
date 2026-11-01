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
symbols = sorted(["infinity_attention", "infinity_embedding_norm", "infinity_embedding_norm_384",
                  "infinity_embedding_norm_768_exact", "infinity_embedding_norm_768_exact_vector",
                  "infinity_embedding_norm_1024_rounded_native",
                  "infinity_embedding_norm_rounded",
                  "infinity_dense_tanh_bf16_inplace", "infinity_f32_to_bf16",
                  "infinity_embedding_norm_word_768_aten", "infinity_gated_gelu_up_first",
                  "infinity_embedding_word_norm_768_candle", "infinity_embedding_word_type_norm_768_exact", "infinity_rounded_residual_norm_768_candle",
                  "infinity_embedding_word_type_norm_512_exact", "infinity_layer_norm_512_rounded_native",
                  "infinity_base_rope_tables_64_bf16",
                  "infinity_gelu_bias", "infinity_layer_norm", "infinity_layer_norm_384",
                  "infinity_layer_norm_768_exact", "infinity_layer_norm_768_exact_vector",
                  "infinity_layer_norm_1024_rounded_native",
                  "infinity_layer_norm_rounded",
                  "infinity_mean_pool", "infinity_pool", "infinity_qkv",
                  "infinity_qk_norm_rope_128_bf16",
                  "infinity_rms_norm", "infinity_rms_norm_rounded",
                  "infinity_rms_norm_2048_exact", "infinity_add_rms_norm_2048_exact",
                  "infinity_rms_norm_2048_vector", "infinity_add_rms_norm_2048_vector",
                  "infinity_rms_norm_1024_exact_vector", "infinity_add_rms_norm_1024_exact_vector",
                  "infinity_rms_norm_1536_cached", "infinity_add_rms_norm_1536_cached",
                  "infinity_rms_norm_1536_vector", "infinity_add_rms_norm_1536_vector",
                  "infinity_rms_norm_4096_vector", "infinity_add_rms_norm_4096_vector",
                  "infinity_silu_mul_bf16_pairs",
                  "infinity_silu_mul_bf16_pairs_native",
                  "infinity_silu_mul_bf16_vector8_native",
                  "infinity_add_rms_norm", "infinity_add_rms_norm_rounded",
                  "infinity_rope_neox", "infinity_rope_neox_hfma",
                  "infinity_rope_interleaved", "infinity_rope_interleaved_hfma",
                  "infinity_silu_mul", "infinity_silu_mul_f32",
                  "infinity_embedding_gather", "infinity_embedding_gather_scaled",
                  "infinity_selected_rows", "infinity_bf16_to_f32"])
# Additive FP16 H384 MiniLM elementwise exports; the original list above is unchanged.
symbols = sorted(symbols + [
    "infinity_embedding_norm_384_f16", "infinity_layer_norm_384_f16",
    "infinity_plain_norm_384_f16", "infinity_mean_pool_f16",
    "infinity_pool_f16", "infinity_qkv_f16", "infinity_f16_to_f32",
])

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
    "compute_dtypes": ["bfloat16", "float16"],
    "compute_dtype_default_compatibility": "Legacy default retained for existing BF16-only consumers; compute_dtypes is the authoritative mixed-precision kernel set.",
    "output_dtype": "float32",
    "sources": {name: digest(root / name) for name in
                ["Cargo.toml", "Cargo.lock", "rust-toolchain.toml", "src/main.rs", "src/norm384.rs", "src/norm512.rs", "src/norm768.rs", "src/norm768_vector.rs", "src/nomic_embedding.rs", "src/norm1024.rs", "src/norm1024_exact_vector.rs", "src/norm2048.rs", "src/norm2048_vector.rs",
                 "src/decoder.rs", "src/dense.rs", "src/encoder_prenorm.rs", "src/gated.rs", "src/aten_embedding.rs", "src/rms1536.rs", "src/rms1536_vector.rs", "src/rms4096_vector.rs", "src/silu_pairs.rs", "src/silu_vector8.rs", "src/readout.rs", "src/qk_norm_rope_128.rs"]},
    "kernels": symbols,
    "qualification": "Every architecture is independently compiled from Rust and assembled to a native cubin; hardware runtime validation is separate.",
    "architectures": [],
}
# Hash the additive F16 module without rewriting the phase1 source list above.
manifest["sources"]["src/encoder_f16.rs"] = digest(root / "src/encoder_f16.rs")
manifest["sources"] = dict(sorted(manifest["sources"].items()))
output.mkdir(parents=True, exist_ok=True)
# Carry the newly adapted primitive's attribution with standalone bundles.
shutil.copy2(root / "LICENSE-PyTorch-BSD3", output / "LICENSE-PyTorch-BSD3")
shutil.copy2(root / "LICENSE-Candle-BSD3", output / "LICENSE-Candle-BSD3")
shutil.copy2(root.parent / "NOTICE", output / "NOTICE")
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
    print(f"{arch} ({hardware}): all {len(symbols)} Rust kernels compiled and assembled", flush=True)
PY
