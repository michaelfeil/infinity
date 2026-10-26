# SPDX-License-Identifier: MIT
# Adapted from candle-flash-attn-v4; see provenance.json and LICENSE-MIT.
"""Build an InfinityFA4 FP16/BF16 native bundle; Python is needed only at build time.

Use the tested FA4 revision e9cf2c1651d2303191eb40a739a3c135fda00999 and
nvidia-cutlass-dsl 4.7.1 in an isolated environment. Pass the CuTe runtime
library directory (cu12/lib for CUDA12 serving) and the tvm_ffi package root.
Use --compile-only --export-only without a GPU, then --link-only inside the deployment
build container to avoid requiring a newer libstdc++ than the serving image.
The LibTorch build consumes a parent directory of sm_NN bundles through
INFINITY_FA4_ROOT and installs them beside libtei_torch.so.
"""
import argparse
import contextlib
import os
import hashlib
import importlib.metadata
import json
import pathlib
import shutil
import subprocess
from portable_host import export_portable, host_target, validate_host_object


p = argparse.ArgumentParser(description=__doc__)
p.add_argument("output", type=pathlib.Path)
p.add_argument("--runtime-dir", type=pathlib.Path)
p.add_argument("--ffi-root", type=pathlib.Path)
p.add_argument("--cxx", default="g++")
phase = p.add_mutually_exclusive_group()
phase.add_argument("--export-only", action="store_true", help="Export kernels before linking in the deployment toolchain")
phase.add_argument("--link-only", action="store_true", help="Link existing exports; no Python GPU packages required")
p.add_argument("--deberta", action="store_true", help="Also export packed DeBERTa relative attention")
p.add_argument("--compile-only", action="store_true", help="Export using fake tensors without a build GPU; runtime validation is still required")
p.add_argument("--arch", choices=["sm_80", "sm_86", "sm_89", "sm_90a", "sm_100a", "sm_120"], default="sm_90a", help="Target architecture; one native bundle per architecture")
a = p.parse_args()
if a.compile_only and a.link_only:
    p.error("--compile-only applies to export, not --link-only")
capability = int(a.arch.removeprefix("sm_").removesuffix("a"))
a.output.mkdir(parents=True, exist_ok=True)
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

if not a.link_only:
    # The manifest must describe the generated code, including when the caller
    # inherited architecture overrides from an earlier build.
    os.environ["FLASH_ATTENTION_ARCH"] = a.arch
    os.environ["CUTE_DSL_ARCH"] = a.arch
    if a.compile_only:
        os.environ["FLASH_ATTENTION_NUM_SMS"] = str({80: 108, 86: 72, 89: 58, 90: 132, 100: 148, 120: 188}[capability])
    else:
        os.environ.pop("FLASH_ATTENTION_NUM_SMS", None)
    from fa4_source import stage_fa4_sources
    staged_source, softmax_backport = stage_fa4_sources()
    import torch
    import flash_attn.cute.interface as interface
    if not a.compile_only and torch.cuda.get_device_capability() != divmod(capability, 10):
        raise RuntimeError(f"Exporting {a.arch} requires a matching GPU or --compile-only")
    if importlib.metadata.version("nvidia-cutlass-dsl") != "4.7.1":
        raise RuntimeError("The private export API is validated with cutlass-dsl 4.7.1 only")
    a.output.mkdir(parents=True, exist_ok=True)
    objects, kernels = [], []
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode() if a.compile_only else contextlib.nullcontext():
        cu = torch.tensor([0, 127, 256], device="cuda", dtype=torch.int32)
        for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16)]:
            profiles = []
            for line in pathlib.Path(__file__).with_name("profiles.inc").read_text().splitlines():
                if line.startswith("FA4_PROFILE("):
                    name, *fields = line.removeprefix("FA4_PROFILE(").removesuffix(")").split(",")
                    h, hk, d, causal, local = map(int, fields)
                    window = (64, 0) if causal and local else (64, 64) if local else (None, None)
                    profiles.append((name.strip(), h, hk, d, bool(causal), window))
            for name, h, hk, d, causal, window in profiles:
                q = torch.zeros((256, h, d), device="cuda", dtype=dtype)
                k = torch.zeros((256, hk, d), device="cuda", dtype=dtype)
                previous = set(interface._flash_attn_fwd.compile_cache.cache)
                if d == 256 and capability in (80, 86, 89, 120):
                    # 64x64 fits the 99 KiB shared-memory budget of SM86/89/120.
                    interface._flash_attn_fwd(q, k, k, cu_seqlens_q=cu, cu_seqlens_k=cu,
                                             max_seqlen_q=129, max_seqlen_k=129,
                                             tile_mn=(64, 64), causal=causal,
                                             window_size_left=window[0], window_size_right=window[1],
                                             disable_scheduler_metadata=True)
                else:
                    interface.flash_attn_varlen_func(q, k, k, cu_seqlens_q=cu, cu_seqlens_k=cu,
                                                    max_seqlen_q=129, max_seqlen_k=129,
                                                    causal=causal, window_size=window,
                                                    disable_scheduler_metadata=True)
                fresh = set(interface._flash_attn_fwd.compile_cache.cache) - previous
                if len(fresh) != 1:
                    raise RuntimeError(f"Unexpected compilation count for {name}: {len(fresh)}")
                compiled = interface._flash_attn_fwd.compile_cache.cache[fresh.pop()]
                symbol = f"fa4_{name}_{dtype_name}"
                obj = a.output / f"{symbol}.o"
                export_portable(compiled, obj, symbol)
                objects.append(str(obj))
                kernels.append(dict(symbol=symbol, heads=h, kv_heads=hk, d=d,
                                    causal=causal, window=window, dtype=dtype_name))
        # Export the upstream varlen lookup aid separately: existing callers
        # retain their original kernels and never need to supply extra tensors.
        if capability == 90:
            previous_threshold = interface.BIN_BATCH_SEARCH_THRESH
            interface.BIN_BATCH_SEARCH_THRESH = 0
            try:
                for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16)]:
                    for name, h, hk, d, causal, window in profiles:
                        if d != 128:
                            continue
                        q = torch.zeros((256, h, d), device="cuda", dtype=dtype)
                        k = torch.zeros((256, hk, d), device="cuda", dtype=dtype)
                        previous = set(interface._flash_attn_fwd.compile_cache.cache)
                        # Keep the key tiles and softmax arithmetic; only the
                        # short-query work layout changes for global attention.
                        tile = (192, 128) if not causal and window == (None, None) else None
                        interface._flash_attn_fwd(q, k, k, cu_seqlens_q=cu, cu_seqlens_k=cu,
                            max_seqlen_q=129, max_seqlen_k=129, causal=causal,
                            window_size_left=window[0], window_size_right=window[1],
                            tile_mn=tile, intra_wg_overlap=False if tile else None,
                            disable_scheduler_metadata=True)
                        fresh = set(interface._flash_attn_fwd.compile_cache.cache) - previous
                        if len(fresh) != 1:
                            raise RuntimeError(f"Unexpected batch metadata compilation count for {name}")
                        compiled = interface._flash_attn_fwd.compile_cache.cache[fresh.pop()]
                        symbol = f"fa4_batch_{name}_{dtype_name}"
                        obj = a.output / f"{symbol}.o"
                        export_portable(compiled, obj, symbol)
                        objects.append(str(obj))
                        kernels.append(dict(symbol=symbol, heads=h, kv_heads=hk, d=d,
                                            causal=causal, window=window, dtype=dtype_name,
                                            batch_metadata=True))
            finally:
                interface.BIN_BATCH_SEARCH_THRESH = previous_threshold
        if a.deberta:
            # SM90 varlen scheduling derives its grid from runtime tensor sizes and
            # cumulative lengths. The 129 below is an export example, not a bound
            # baked into FlashAttentionForwardSm90.__call__. See the long-sequence
            # Native regression before changing the pinned upstream implementation.
            from deberta_score import deberta_score
            for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16)]:
                q = torch.zeros((256, 12, 64), device="cuda", dtype=dtype)
                rel = torch.zeros((12, 256, 512), device="cuda", dtype=dtype)
                lut = torch.zeros(257, device="cuda", dtype=torch.int32)
                previous = set(interface._flash_attn_fwd.compile_cache.cache)
                interface.flash_attn_varlen_func(q, q, q, cu_seqlens_q=cu, cu_seqlens_k=cu,
                    max_seqlen_q=129, max_seqlen_k=129, softmax_scale=1.,
                    score_mod=deberta_score, aux_tensors=[rel, rel, lut],
                    disable_scheduler_metadata=True)
                fresh = set(interface._flash_attn_fwd.compile_cache.cache) - previous
                if len(fresh) != 1:
                    raise RuntimeError("Unexpected DeBERTa compilation count")
                compiled = interface._flash_attn_fwd.compile_cache.cache[fresh.pop()]
                symbol = f"fa4_deberta_{dtype_name}"
                obj = a.output / f"{symbol}.o"
                export_portable(compiled, obj, symbol)
                objects.append(str(obj))
                kernels.append(dict(symbol=symbol, heads=12, kv_heads=12, d=64,
                                    causal=False, window=[None, None], dtype=dtype_name))
    manifest = {
        "abi_version": 7,
        "host_target": host_target(),
        "softmax_backport": softmax_backport,
        "architecture": a.arch,
        "executed_during_export": not a.compile_only,
        "target_sm_count": int(os.environ["FLASH_ATTENTION_NUM_SMS"]) if a.compile_only else torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count,
        "deberta_score_sha256": digest(pathlib.Path(__file__).with_name("deberta_score.py")) if a.deberta else None,
        "kernels": kernels,
        "fa4_python_sources": {str(f.relative_to(pathlib.Path(interface.__file__).parent)): digest(f)
                               for f in sorted(pathlib.Path(interface.__file__).parent.rglob("*.py"))},
        "cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
        "tvm_ffi": importlib.metadata.version("apache-tvm-ffi"),
        "objects": {pathlib.Path(obj).name: digest(pathlib.Path(obj)) for obj in objects},
    }
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
else:
    manifest = json.loads((a.output / "manifest.json").read_text())
    if manifest.get("abi_version") != 7:
        raise RuntimeError("Re-export the native bundle for ABI version 7")
    if manifest.get("host_target") != host_target():
        raise RuntimeError("Re-export FA4 with the portable host target before linking")
    objects = [str(a.output / (k["symbol"] + ".o")) for k in manifest["kernels"]]
    for obj in objects:
        path = pathlib.Path(obj)
        if digest(path) != manifest["objects"][path.name]:
            raise RuntimeError(f"Export changed since manifest was created: {path}")
for obj in objects:
    validate_host_object(pathlib.Path(obj))
if a.export_only:
    print(f"Exported kernels to {a.output}; link with a deployment-compatible C++ compiler", flush=True)
    raise SystemExit(0)
if a.runtime_dir is None or a.ffi_root is None:
    p.error("Linking requires --runtime-dir and --ffi-root")
if manifest["architecture"] not in ["sm_80", "sm_86", "sm_89", "sm_90a", "sm_100a", "sm_120"]:
    raise RuntimeError("Unsupported bundle architecture")
capability = int(manifest["architecture"].removeprefix("sm_").removesuffix("a"))
bridge = pathlib.Path(__file__).with_name("bridge.cpp")
subprocess.run([a.cxx, "-std=c++17", "-O2", "-fPIC", "-shared", str(bridge),
                *objects, "-DINFINITY_FA4_PORTABLE_HOST=1", f"-DFA4_COMPUTE_CAPABILITY={capability}", f"-I{a.ffi_root / 'include'}",
                f"-L{a.runtime_dir}", "-lcute_dsl_runtime",
                f"-L{a.ffi_root / 'lib'}", "-ltvm_ffi",
                "-Wl,-rpath,$ORIGIN", "-Wl,-z,defs",
                *(["-DINFINITY_FA4_BATCH_HINTS"] if any(k.get("batch_metadata") for k in manifest["kernels"]) else []),
                *(["-DFA4_DEBERTA"] if any(k["symbol"].startswith("fa4_deberta_") for k in manifest["kernels"]) else []),
                "-o", str(a.output / "libinfinityfa4.so")], check=True)
validate_host_object(a.output / "libinfinityfa4.so")
for source in [a.runtime_dir / "libcute_dsl_runtime.so",
               a.ffi_root / "lib/libtvm_ffi.so"]:
    shutil.copy2(source, a.output / source.name)
# Retain redistribution notices for both native runtime dependencies, including
# split export/link builds where packages are provided through explicit roots.
for root in {a.ffi_root.parent, a.runtime_dir.resolve().parents[2]}:
    for distribution in root.glob("*.dist-info"):
        if not distribution.name.startswith(("nvidia_cutlass_dsl", "apache_tvm_ffi")):
            continue
        destination = a.output / "licenses" / distribution.name
        for notice in distribution.iterdir():
            if notice.name.upper().startswith(("LICENSE", "NOTICE", "COPYING", "AUTHORS")):
                destination.mkdir(parents=True, exist_ok=True)
                if notice.is_dir():
                    shutil.copytree(notice, destination / notice.name, dirs_exist_ok=True)
                else:
                    shutil.copy2(notice, destination / notice.name)
manifest.update({
    "name": "InfinityFA4",
    "native_abi_version": 1,
    "bridge_source_sha256": digest(bridge),
    "cxx": subprocess.check_output([a.cxx, "--version"], text=True).splitlines()[0],
    "libraries": {f.name: digest(f) for f in sorted(a.output.glob("*.so"))},
})
(a.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(f"Linked {len(manifest['kernels'])} kernels and native bridge in {a.output}", flush=True)
