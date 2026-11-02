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
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl

root, output = [Path(p).resolve() for p in sys.argv[1:3]]

def resolve_from_root(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()

def path_or_command(value):
    if os.sep in value or (os.altsep and os.altsep in value) or value.startswith("~"):
        return str(resolve_from_root(value))
    return value

compiler = path_or_command(sys.argv[3])
toolkit_value = os.environ.get("CUDA_TOOLKIT_PATH", os.environ.get("CUDA_HOME"))
toolkit = str(resolve_from_root(toolkit_value)) if toolkit_value else None
ptxas_value = os.environ.get("PTXAS")
ptxas = (path_or_command(ptxas_value) if ptxas_value
         else (str(Path(toolkit) / "bin/ptxas") if toolkit else "ptxas"))
cuobjdump_value = os.environ.get("CUOBJDUMP")
cuobjdump = (path_or_command(cuobjdump_value) if cuobjdump_value
             else (str(Path(toolkit) / "bin/cuobjdump") if toolkit else "cuobjdump"))
environment = dict(os.environ, RUSTUP_TOOLCHAIN="1.99.1", RUSTC_BOOTSTRAP="1")
# The compiler now runs from a snapshot; keep any caller-supplied CUDA roots
# absolute so downstream lookups do not resolve relative to the snapshot.
for name in ("CUDA_TOOLKIT_PATH", "CUDA_HOME"):
    if name in os.environ:
        environment[name] = str(resolve_from_root(os.environ[name]))
architectures = [("sm_90", "H100"), ("sm_100", "B200"),
                 ("sm_120", "RTX PRO 6000 Blackwell")]
jobs_text = os.environ.get("CARGO_BUILD_JOBS", "3")
try:
    total_jobs = int(jobs_text)
except ValueError:
    raise SystemExit(f"CARGO_BUILD_JOBS must be a positive integer, got {jobs_text!r}")
if total_jobs < 1:
    raise SystemExit(f"CARGO_BUILD_JOBS must be a positive integer, got {jobs_text!r}")
# cargo-oxide reads configuration only from cwd/.cargo/cuda-oxide.toml. A
# private snapshot would hide a device-local file (including relative backend
# paths and extra rustflags), so that case falls back to the original root.
source_lock = os.open(root, os.O_RDONLY)
fcntl.flock(source_lock, fcntl.LOCK_EX)
local_config_mode = (root / ".cargo" / "cuda-oxide.toml").is_file()
if local_config_mode:
    worker_count = 1
    job_counts = [total_jobs] * len(architectures)
    print("Detected device/.cargo/cuda-oxide.toml: compiling architectures "
          "sequentially from the original source root to preserve local "
          "cargo-oxide configuration.", flush=True)
else:
    worker_count = min(total_jobs, len(architectures))
    base_jobs, extra_jobs = divmod(total_jobs, worker_count)
    job_counts = [base_jobs + (index < extra_jobs)
                  for index in range(len(architectures))]
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
                  "infinity_jina_code_qk_norm_768",
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
symbols = sorted(symbols + ["infinity_deberta_unpack_scaled", "infinity_deberta_relative_transpose",
                           "infinity_deberta_divide", "infinity_deberta_signed_buckets",
                           "infinity_deberta_embedding_word_norm_768", "infinity_deberta_layer_norm_768"])
symbols = sorted(symbols + [
    "infinity_embedding_norm_384_f16", "infinity_layer_norm_384_f16",
    "infinity_plain_norm_384_f16", "infinity_mean_pool_f16",
    "infinity_pool_f16", "infinity_qkv_f16", "infinity_f16_to_f32",
])

symbols = sorted(symbols + ["infinity_classifier_bias_bf16", "infinity_classifier_gather_first_bf16", "infinity_classifier_embedding_norm_384_exact", "infinity_classifier_layer_norm_384_exact"])

symbols = sorted(symbols + ["infinity_embedding_norm_mpnet_768_aten", "infinity_layer_norm_mpnet_768_aten", "infinity_gelu_exact"])
symbols = sorted(symbols + ["infinity_gemma2_norm_bf16_f32scale",
                            "infinity_gemma2_residual_norm_bf16_f32scale",
                            "infinity_gemma2_unit_norm_bf16",
                            "infinity_gemma2_gated_gelu_bf16",
                            "infinity_gemma2_ple_gate_bf16",
                            "infinity_gemma2_scale_bf16",
                            "infinity_gemma2_rope_tables_bf16",
                            "infinity_gemma2_mean_pool_bf16"])

# Additive optional encoder-activation exports live in their own leaf file so
# composing another additive leaf cannot overwrite this source. The original
# 86-name list above remains unchanged.
encoder_activation_exports = [
    line.strip()
    for line in (root / "encoder_activation_exports.txt").read_text().splitlines()
    if line.strip()
]
symbols = sorted(symbols + encoder_activation_exports)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def command(args, log, env, cwd=root):
    start = time.perf_counter()
    with log.open("w") as stream:
        subprocess.run(args, cwd=cwd, env=env, stdout=stream,
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
                 "src/decoder.rs", "src/dense.rs", "src/encoder_prenorm.rs", "src/gated.rs", "src/aten_embedding.rs", "src/mpnet_norm.rs", "src/gelu_exact.rs", "src/rms1536.rs", "src/rms1536_vector.rs", "src/rms4096_vector.rs", "src/silu_pairs.rs", "src/silu_vector8.rs", "src/readout.rs", "src/qk_norm_rope_128.rs"]},
    "kernels": symbols,
    "qualification": "Every architecture is independently compiled from Rust and assembled to a native cubin; hardware runtime validation is separate.",
    "architectures": [],
}
# Hash the additive F16 module without rewriting the phase1 source list above.
manifest["sources"]["src/encoder_f16.rs"] = digest(root / "src/encoder_f16.rs")
manifest["sources"]["src/jina_code_qk.rs"] = digest(root / "src/jina_code_qk.rs")
manifest["sources"]["src/classifier.rs"] = digest(root / "src/classifier.rs")
manifest["sources"]["src/classifier_norm384.rs"] = digest(root / "src/classifier_norm384.rs")
manifest["sources"]["src/gemma2.rs"] = digest(root / "src/gemma2.rs")
manifest["sources"]["src/encoder_activation.rs"] = digest(root / "src/encoder_activation.rs")
manifest["sources"]["encoder_activation_exports.txt"] = digest(root / "encoder_activation_exports.txt")
for name in ["src/deberta_math.rs", "src/deberta_norm.rs", "src/gelu_exact.rs"]:
    manifest["sources"][name] = digest(root / name)
manifest["sources"] = dict(sorted(manifest["sources"].items()))
output.mkdir(parents=True, exist_ok=True)
work_root = (output.parent / (output.name + ".work")).resolve()
target_base_value = os.environ.get("CARGO_TARGET_DIR")
target_base = resolve_from_root(target_base_value) if target_base_value else None

def build_arch(index, arch, hardware, source_root):
    directory = output / arch
    directory.mkdir(parents=True, exist_ok=True)
    arch_work = work_root / arch
    ptx_dir = arch_work / "ptx"
    target_dir = (target_base / arch) if target_base else (arch_work / "target")
    # Never accept a stale PTX left by an earlier failed invocation.
    shutil.rmtree(ptx_dir, ignore_errors=True)
    ptx_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    job_env = dict(environment, CUDA_OXIDE_PTX_DIR=str(ptx_dir),
                   CARGO_TARGET_DIR=str(target_dir),
                   CARGO_BUILD_JOBS=str(job_counts[index]))
    build = [compiler, "build", "--arch", arch]
    build_seconds = command(build, directory / "build.log", job_env, source_root)
    produced = ptx_dir / "infinity_oxide_kernels.ptx"
    if not produced.is_file():
        raise RuntimeError(f"Compiler did not emit {produced.name} for {arch} in {ptx_dir}")
    ptx, cubin = directory / "kernels.ptx", directory / "kernels.cubin"
    shutil.copy2(produced, ptx)
    text = ptx.read_text()
    targets = re.findall(r"^\.target\s+(\w+)", text, re.MULTILINE)
    entries = sorted(re.findall(r"\.visible\s+\.entry\s+(\w+)\s*\(", text))
    if targets != [arch] or entries != symbols:
        raise RuntimeError(f"Incorrect compiler target or kernel exports: {arch}: {targets}, {entries}")
    assemble = [ptxas, "-arch=" + arch, "-v", str(ptx), "-o", str(cubin)]
    assemble_seconds = command(assemble, directory / "ptxas.log", environment)
    binary_symbols = subprocess.check_output([cuobjdump, "--dump-elf-symbols", str(cubin)],
                                             text=True, env=environment)
    (directory / "symbols.log").write_text(binary_symbols)
    native_entries = sorted(re.findall(r"STO_ENTRY\s+(\w+)", binary_symbols))
    if native_entries != symbols:
        raise RuntimeError(f"Incorrect native cubin exports for {arch}: {native_entries}")
    elf_text = subprocess.check_output([cuobjdump, "--dump-elf", str(cubin)],
                                       text=True, env=environment)
    native_arch = re.search(r"\bsm=(\d+)\b", elf_text)
    if native_arch is None or f"sm_{native_arch[1]}" != arch:
        raise RuntimeError(f"Incorrect native cubin architecture for {arch}: "
                           f"{native_arch[1] if native_arch else 'missing'}")
    return {
        "architecture": arch, "hardware": hardware,
        "ptx_version": re.search(r"^\.version\s+([\d.]+)", text, re.MULTILINE)[1],
        "ptx": str(ptx.relative_to(output)), "cubin": str(cubin.relative_to(output)),
        "ptx_sha256": digest(ptx), "cubin_sha256": digest(cubin),
        "build_seconds": build_seconds, "assemble_seconds": assemble_seconds,
        "build_command": build, "assemble_command": assemble,
        "symbols_verified": native_entries,
    }

def snapshot_arch(invocation_root, arch):
    # cargo-oxide rewrites <example>/src/main.rs unconditionally. Give every
    # compiler process a private source copy; it must never touch the real root.
    destination = invocation_root / arch / "libs/infinity-v3/backends/oxide/device"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(root / "src", destination / "src")
    for name in ["Cargo.toml", "Cargo.lock", "rust-toolchain.toml",
                 "encoder_activation_exports.txt", "LICENSE-Candle-BSD3",
                 "LICENSE-PyTorch-BSD3"]:
        shutil.copy2(root / name, destination / name)
    notice = root.parent / "NOTICE"
    if notice.is_file():
        shutil.copy2(notice, destination.parent / "NOTICE")
    license_file = root.parent.parent.parent / "LICENSE"
    if license_file.is_file():
        shutil.copy2(license_file, destination.parent.parent.parent / "LICENSE")
    return destination

def write_manifest_atomic(path, text):
    descriptor, temporary = tempfile.mkstemp(prefix=".manifest.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "w") as stream:
            stream.write(text)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise

work_root.mkdir(parents=True, exist_ok=True)
work_lock = os.open(work_root, os.O_RDONLY)
try:
    fcntl.flock(work_lock, fcntl.LOCK_EX)
    # Attribution belongs to this output; write it only after the work-dir lock
    # so a different source root cannot race the same output directory.
    shutil.copy2(root / "LICENSE-PyTorch-BSD3", output / "LICENSE-PyTorch-BSD3")
    shutil.copy2(root / "LICENSE-Candle-BSD3", output / "LICENSE-Candle-BSD3")
    shutil.copy2(root.parent / "NOTICE", output / "NOTICE")
    manifest_path = output / "manifest.json"
    manifest_path.unlink(missing_ok=True)
    if local_config_mode:
        invocation_root = None
        source_roots = {arch: root for arch, _hardware in architectures}
    else:
        workspace_parent = root / ".oxide-artifacts" / "build-workspaces"
        workspace_parent.mkdir(parents=True, exist_ok=True)
        invocation_root = Path(tempfile.mkdtemp(prefix="build-", dir=workspace_parent))
        source_roots = {arch: snapshot_arch(invocation_root, arch)
                        for arch, _hardware in architectures}
    try:
        results = {}
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(build_arch, index, arch, hardware, source_roots[arch]):
                    (arch, hardware)
                for index, (arch, hardware) in enumerate(architectures)
            }
            for future in as_completed(futures):
                arch, _hardware = futures[future]
                results[arch] = future.result()
        manifest["architectures"] = [results[arch] for arch, _hardware in architectures]
        write_manifest_atomic(manifest_path, json.dumps(manifest, indent=2))
        for arch, hardware in architectures:
            print(f"{arch} ({hardware}): all {len(symbols)} Rust kernels compiled and assembled", flush=True)
    finally:
        if invocation_root is not None:
            shutil.rmtree(invocation_root, ignore_errors=True)
finally:
    fcntl.flock(work_lock, fcntl.LOCK_UN)
    os.close(work_lock)
    os.close(source_lock)
PY
