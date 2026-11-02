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
import os
import hashlib
import importlib.metadata
import json
import pathlib
import shutil
import subprocess
from portable_host import export_portable, host_target, validate_host_object
import build_contract


p = argparse.ArgumentParser(description=__doc__)
p.add_argument("output", type=pathlib.Path)
p.add_argument("--runtime-dir", type=pathlib.Path)
p.add_argument("--ffi-root", type=pathlib.Path)
p.add_argument("--cxx", default="g++")
phase = p.add_mutually_exclusive_group()
phase.add_argument("--export-only", action="store_true", help="Export kernels before linking in the deployment toolchain")
phase.add_argument("--link-only", action="store_true", help="Link existing exports; no Python GPU packages required")
p.add_argument("--deberta", action="store_true", help="Also export packed DeBERTa relative attention")
p.add_argument("--deberta-exact-mul", action="store_true", help="Include the BF16 fixed-divisor DeBERTa score hook")
p.add_argument("--alibi", action="store_true", help="Also export packed Jina ALiBi model-bias attention")
p.add_argument("--jinacode", action="store_true", help="Also export native JinaCode F32-FMA ALiBi attention")
p.add_argument("--alibi12", action="store_true", help="Also export JinaBase 12-head model-dtype ALiBi attention")
p.add_argument("--only-symbol", nargs="+", metavar="SYMBOL", help="Development-only: export only these canonical symbols (requires --compile-only --export-only)")
p.add_argument("--compile-only", action="store_true", help="Export using fake tensors without a build GPU; runtime validation is still required")
p.add_argument("--arch", choices=["sm_80", "sm_86", "sm_89", "sm_90a", "sm_100a", "sm_120"], default="sm_90a", help="Target architecture; one native bundle per architecture")
a = p.parse_args()
if a.compile_only and a.link_only:
    p.error("--compile-only applies to export, not --link-only")
if a.deberta_exact_mul and not a.deberta and not a.link_only:
    p.error("--deberta-exact-mul requires --deberta")
capability = int(a.arch.removeprefix("sm_").removesuffix("a"))
dev_only = bool(a.only_symbol)
if dev_only:
    if not (a.compile_only and a.export_only):
        p.error("--only-symbol requires --compile-only and --export-only")
    only_symbols = list(dict.fromkeys(a.only_symbol))
    if len(only_symbols) != len(a.only_symbol):
        p.error("--only-symbol contains duplicate symbols")
    profiles_inc = pathlib.Path(__file__).with_name("profiles.inc")
    profile_rows = build_contract.validate_profiles_inc(profiles_inc)
    dev_flags = {
        "include_deberta": a.deberta,
        "include_deberta_exact_mul": a.deberta_exact_mul,
        "include_alibi": a.alibi,
        "include_jinacode": a.jinacode,
        "include_alibi12": a.alibi12,
    }
    canonical_symbols = build_contract.expected_kernel_symbols(
        [row[0] for row in profile_rows], **dev_flags)
    unknown = [symbol for symbol in only_symbols if symbol not in canonical_symbols]
    if unknown:
        p.error("unknown --only-symbol for this canonical build: " + ", ".join(unknown))
    if (a.output / "manifest.json").exists() or (a.output / "libinfinityfa4.so").exists():
        raise RuntimeError("refusing development-only export into an existing canonical bundle")
    (a.output / "development-manifest.json").unlink(missing_ok=True)
a.output.mkdir(parents=True, exist_ok=True)
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def registry_metadata():
    rows = []
    for line in pathlib.Path(__file__).with_name("profiles.inc").read_text().splitlines():
        if not line.startswith("FA4_PROFILE("):
            continue
        name, *fields = line.removeprefix("FA4_PROFILE(").removesuffix(")").split(",")
        h, hk, d, causal, local = map(int, fields)
        rows.append((name.strip(), h, hk, d, bool(causal), bool(local)))
    namespaces = {19: "legacy19", 20: "published20", 21: "private23", 22: "private23",
                  23: "private23", 24: "registry24", 25: "canonical26", 26: "canonical26",
                  27: "canonical27", 28: "canonical28", 29: "canonical29", 30: "canonical30",
                  31: "canonical31", 38: "canonical38"}
    namespace = namespaces.get(len(rows))
    if namespace is None:
        raise RuntimeError(f"unsupported canonical profile count {len(rows)}")
    sequence = "|".join(
        f"{index}:{name}:{h}:{hk}:{d}:{int(causal)}:{int(local)}"
        for index, (name, h, hk, d, causal, local) in enumerate(rows))
    return {
        "namespace": namespace,
        "mode_count": len(rows),
        "mode_mask": f"{(1 << len(rows)) - 1:#x}",
        "profile_sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
    }

def compile_isolated(label, compile_fn):
    # A noncausal d256 local profile can reuse an earlier generic kernel cache
    # entry. Clear before every export so a cache hit never masquerades as a
    # fresh compile, and require exactly one entry for this profile.
    cache = interface._flash_attn_fwd.compile_cache.cache
    cache.clear()
    if len(cache) != 0:
        raise RuntimeError(f"{label}: compile cache did not clear")
    compile_fn()
    if len(cache) != 1:
        raise RuntimeError(f"Unexpected compilation count for {label}: {len(cache)}")
    return next(iter(cache.values()))

def clear_compile_cache(label):
    cache = interface._flash_attn_fwd.compile_cache.cache
    cache.clear()
    if len(cache) != 0:
        raise RuntimeError(f"{label}: compile cache did not clear after export")

if not a.link_only:
    # Canonical31 is an AOT source build. Reject a visible GPU, a persistent
    # JIT cache, and external ptxas before importing torch/flash_attn.
    export_env = build_contract.validate_compile_env(os.environ, a.compile_only)
    if a.runtime_dir is None or a.ffi_root is None:
        p.error("canonical31 export requires --runtime-dir and --ffi-root")
    profiles_inc = pathlib.Path(__file__).with_name("profiles.inc")
    profile_rows = build_contract.validate_profiles_inc(profiles_inc)
    os.environ["FLASH_ATTENTION_ARCH"] = a.arch
    os.environ["CUTE_DSL_ARCH"] = a.arch
    os.environ["FLASH_ATTENTION_NUM_SMS"] = str({80: 108, 86: 72, 89: 58, 90: 132, 100: 148, 120: 188}[capability])
    from fa4_source import stage_fa4_sources
    staged_source, softmax_backport = stage_fa4_sources()
    gemma_local_patch = None
    if capability == 100:
        from gemma_sm100_local import patch_interface
        gemma_local_patch = patch_interface(pathlib.Path(staged_source.name) / 'flash_attn/cute/interface.py')
    lock_path = pathlib.Path(__file__).with_name("fa4_source_lock.json")
    source_lock = build_contract.load_source_lock(lock_path)
    if softmax_backport != source_lock["softmax_policy"]:
        raise RuntimeError("staged FA4 softmax policy differs from the source lock")
    source_identity_before = build_contract.verify_source_identity(
        pathlib.Path(staged_source.name) / "flash_attn/cute", capability, source_lock)
    import torch
    def require_no_cuda(label):
        if torch.cuda.is_initialized():
            raise RuntimeError(f"CUDA context initialized unexpectedly: {label}")
    require_no_cuda("after torch import")
    import flash_attn.cute.interface as interface
    require_no_cuda("after flash_attn import")
    source_identity_after = build_contract.verify_source_identity(
        pathlib.Path(interface.__file__).parent, capability, source_lock)
    if importlib.metadata.version("nvidia-cutlass-dsl") != "4.7.1":
        raise RuntimeError("The private export API is validated with cutlass-dsl 4.7.1 only")
    a.output.mkdir(parents=True, exist_ok=True)
    from compile_only import install_export_only_compile
    compile_only_state = install_export_only_compile(interface.cute)
    objects, kernels = [], []
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode():
        cu = torch.tensor([0, 127, 256], device="cuda", dtype=torch.int32)
        for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16)]:
            profiles = []
            for mode, row in enumerate(profile_rows):
                name, h, hk, d, causal, _local = row
                window = build_contract.canonical_window(mode, name, causal, _local)
                profiles.append((mode, name, h, hk, d, causal, window))
            for mode, name, h, hk, d, causal, window in profiles:
                symbol = f"fa4_{name}_{dtype_name}"
                if dev_only and symbol not in only_symbols:
                    continue
                q = torch.zeros((256, h, d), device="cuda", dtype=dtype)
                k = torch.zeros((256, hk, d), device="cuda", dtype=dtype)
                def compile_profile():
                    # SM100 d256 local must use the generic one-CTA path; the
                    # dedicated two-CTA hd256 kernel rejects local windows. This
                    # covers Gemma3, EmbeddingGemma2 and Gemma4 local d256 profiles.
                    if d == 256 and window[0] is not None and capability == 100:
                        interface._flash_attn_fwd(q, k, k, cu_seqlens_q=cu, cu_seqlens_k=cu,
                                                 max_seqlen_q=129, max_seqlen_k=129, tile_mn=(128, 64),
                                                 num_splits=1, pack_gqa=False, causal=causal,
                                                 window_size_left=window[0], window_size_right=window[1],
                                                 disable_scheduler_metadata=True)
                    elif d == 256 and capability in (80, 86, 89, 120):
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
                compiled = compile_isolated(name, compile_profile)
                obj = a.output / f"{symbol}.o"
                export_portable(compiled, obj, symbol)
                require_no_cuda(f"after export {symbol}")
                clear_compile_cache(name)
                objects.append(str(obj))
                kernels.append(build_contract.profile_kernel(
                    mode, name, h, hk, d, causal, window, dtype_name))
        if a.deberta:
            # SM90 varlen scheduling derives its grid from runtime tensor sizes and
            # cumulative lengths. The 129 below is an export example, not a bound
            # baked into FlashAttentionForwardSm90.__call__. See the long-sequence
            # Native regression before changing the pinned upstream implementation.
            from deberta_score import deberta_score
            for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16)]:
                symbol = f"fa4_deberta_{dtype_name}"
                if dev_only and symbol not in only_symbols:
                    continue
                q = torch.zeros((256, 12, 64), device="cuda", dtype=dtype)
                rel = torch.zeros((12, 256, 512), device="cuda", dtype=dtype)
                lut = torch.zeros(257, device="cuda", dtype=torch.int32)
                def compile_deberta():
                    interface.flash_attn_varlen_func(q, q, q, cu_seqlens_q=cu, cu_seqlens_k=cu,
                        max_seqlen_q=129, max_seqlen_k=129, softmax_scale=1.,
                        score_mod=deberta_score, aux_tensors=[rel, rel, lut],
                        disable_scheduler_metadata=True)
                compiled = compile_isolated(f"deberta_{dtype_name}", compile_deberta)
                obj = a.output / f"{symbol}.o"
                export_portable(compiled, obj, symbol)
                require_no_cuda(f"after export {symbol}")
                clear_compile_cache(f"deberta_{dtype_name}")
                objects.append(str(obj))
                kernels.append(dict(symbol=symbol, heads=12, kv_heads=12, d=64,
                                    causal=False, window=[None, None], dtype=dtype_name))
        if a.deberta_exact_mul:
            from deberta_mul_score import deberta_exact_mul_both, deberta_exact_mul_single
            for name, callback, divisor, reciprocal in [
                    ("both", deberta_exact_mul_both, 13.875, 0x3d939a86),
                    ("single", deberta_exact_mul_single, 11.3125, 0x3db509e7)]:
                symbol = f"fa4_deberta_exact_mul_{name}_bf16"
                if dev_only and symbol not in only_symbols:
                    continue
                q = torch.zeros((256, 12, 64), device="cuda", dtype=torch.bfloat16)
                rel = torch.zeros((12, 256, 512), device="cuda", dtype=torch.bfloat16)
                lut = torch.zeros(257, device="cuda", dtype=torch.int32)
                def compile_exact_mul():
                    interface.flash_attn_varlen_func(q, q, q, cu_seqlens_q=cu, cu_seqlens_k=cu,
                        max_seqlen_q=129, max_seqlen_k=129, softmax_scale=1.,
                        score_mod=callback, aux_tensors=[rel, rel, lut],
                        disable_scheduler_metadata=True)
                compiled = compile_isolated(f"deberta_exact_mul_{name}", compile_exact_mul)
                obj = a.output / f"{symbol}.o"
                export_portable(compiled, obj, symbol)
                require_no_cuda(f"after export {symbol}")
                clear_compile_cache(f"deberta_exact_mul_{name}")
                objects.append(str(obj))
                kernels.append(dict(symbol=symbol, heads=12, kv_heads=12, d=64,
                                    causal=False, window=[None, None], dtype="bf16",
                                    relative_score_input="raw_bf16_matmul",
                                    relative_divisor_f32=divisor,
                                    relative_reciprocal_f32_bits=reciprocal,
                                    relative_score_transform="mul.rn.f32+cvt.rn.bf16x2.f32+bit_widen_f32"))
        if a.alibi:
            # Same fake-tensor compile-only path as the ordinary exports.  The
            # 129 below is an export example, not a runtime sequence cap: the
            # packed offsets determine the actual varlen grid.
            from alibi_score import alibi_model_bias_bf16, alibi_model_bias_fp16
            for dtype_name, dtype, score_mod in [
                    ("fp16", torch.float16, alibi_model_bias_fp16),
                    ("bf16", torch.bfloat16, alibi_model_bias_bf16)]:
                symbol = f"fa4_alibi_model_bias_{dtype_name}"
                if dev_only and symbol not in only_symbols:
                    continue
                q = torch.zeros((256, 8, 64), device="cuda", dtype=dtype)
                slopes = torch.zeros(8, device="cuda", dtype=torch.float32)
                def compile_alibi():
                    interface.flash_attn_varlen_func(q, q, q, cu_seqlens_q=cu, cu_seqlens_k=cu,
                        max_seqlen_q=129, max_seqlen_k=129, softmax_scale=1.,
                        score_mod=score_mod, aux_tensors=[slopes],
                        disable_scheduler_metadata=True)
                compiled = compile_isolated(f"alibi_{dtype_name}", compile_alibi)
                obj = a.output / f"{symbol}.o"
                export_portable(compiled, obj, symbol)
                require_no_cuda(f"after export {symbol}")
                clear_compile_cache(f"alibi_{dtype_name}")
                objects.append(str(obj))
                kernels.append(dict(symbol=symbol, heads=8, kv_heads=8, d=64,
                                    causal=False, window=[None, None], dtype=dtype_name,
                                    kind="alibi_model_bias", score_source="alibi_score.py",
                                    score_abi=1, qk_scale=0.125, external_softmax_scale=1.0,
                                    bias_rounding="model_dtype"))
        if a.jinacode:
            # The model's native Flash recipe retains F32 bias/FMA precision.
            # Packed Q/K/V row strides match the shared encoder workspace.
            from alibi12_native_score import alibi12_native_flash_bf16, alibi12_native_flash_fp16
            for dtype_name, dtype, score_mod in [
                    ("fp16", torch.float16, alibi12_native_flash_fp16),
                    ("bf16", torch.bfloat16, alibi12_native_flash_bf16)]:
                symbol = f"fa4_jinacode_flash_alibi_{dtype_name}"
                if dev_only and symbol not in only_symbols:
                    continue
                packed = torch.zeros((256, 2304), device="cuda", dtype=dtype)
                q = packed[:, :768].view(256, 12, 64)
                k = packed[:, 768:1536].view(256, 12, 64)
                v = packed[:, 1536:].view(256, 12, 64)
                slopes = torch.zeros(12, device="cuda", dtype=torch.float32)
                def compile_jinacode():
                    interface.flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
                        max_seqlen_q=129, max_seqlen_k=129, softmax_scale=1.,
                        score_mod=score_mod, aux_tensors=[slopes],
                        disable_scheduler_metadata=True)
                compiled = compile_isolated(f"jinacode_{dtype_name}", compile_jinacode)
                obj = a.output / f"{symbol}.o"
                export_portable(compiled, obj, symbol)
                require_no_cuda(f"after export {symbol}")
                clear_compile_cache(f"jinacode_{dtype_name}")
                objects.append(str(obj))
                kernels.append(dict(symbol=symbol, heads=12, kv_heads=12, d=64,
                                    causal=False, window=[None, None], dtype=dtype_name,
                                    kind="alibi12_native_flash", score_source="alibi12_native_score.py",
                                    score_abi=1, qk_scale=0.125, external_softmax_scale=1.0,
                                    bias_rounding="float32_fma_unscaled_then_pow2_scale",
                                    alibi_heads=12, alibi_slope_count=12))
        if a.alibi12:
            # JinaBase uses the ordinary model-dtype bias contract with twelve
            # heads. This is additive and does not reuse the JinaCode FMA hook.
            from alibi12_model_bias_score import (
                alibi12_model_bias_bf16, alibi12_model_bias_fp16)
            for dtype_name, dtype, score_mod in [
                    ("fp16", torch.float16, alibi12_model_bias_fp16),
                    ("bf16", torch.bfloat16, alibi12_model_bias_bf16)]:
                symbol = f"fa4_alibi12_model_bias_{dtype_name}"
                if dev_only and symbol not in only_symbols:
                    continue
                q = torch.zeros((256, 12, 64), device="cuda", dtype=dtype)
                slopes = torch.zeros(12, device="cuda", dtype=torch.float32)
                def compile_alibi12():
                    interface.flash_attn_varlen_func(q, q, q, cu_seqlens_q=cu, cu_seqlens_k=cu,
                        max_seqlen_q=129, max_seqlen_k=129, softmax_scale=1.,
                        score_mod=score_mod, aux_tensors=[slopes],
                        disable_scheduler_metadata=True)
                compiled = compile_isolated(f"alibi12_{dtype_name}", compile_alibi12)
                obj = a.output / f"{symbol}.o"
                export_portable(compiled, obj, symbol)
                require_no_cuda(f"after export {symbol}")
                clear_compile_cache(f"alibi12_{dtype_name}")
                objects.append(str(obj))
                kernels.append(dict(symbol=symbol, heads=12, kv_heads=12, d=64,
                                    causal=False, window=[None, None], dtype=dtype_name,
                                    kind="alibi12_model_bias",
                                    score_source="alibi12_model_bias_score.py",
                                    score_abi=1, qk_scale=0.125,
                                    external_softmax_scale=1.0,
                                    bias_rounding="model_dtype",
                                    alibi_heads=12, alibi_slope_count=12))
    require_no_cuda("after all exports")
    if dev_only:
        profile_names = [row[0] for row in profile_rows]
        canonical = build_contract.expected_kernel_symbols(profile_names, **dev_flags)
        selected = [symbol for symbol in canonical if symbol in only_symbols]
        by_symbol = {kernel["symbol"]: kernel for kernel in kernels}
        missing = [symbol for symbol in selected if symbol not in by_symbol]
        extra = [symbol for symbol in by_symbol if symbol not in selected]
        if missing or extra:
            raise RuntimeError(
                f"development export inventory differs: missing={missing} extra={extra}")
        if len(compile_only_state["calls"]) != len(selected):
            raise RuntimeError(
                f"development compile callback count {len(compile_only_state['calls'])} "
                f"!= selected object count {len(selected)}")
        for symbol in selected:
            validate_host_object(a.output / f"{symbol}.o")
        modes = {f"fa4_{row[0]}_{dtype}": mode
                 for mode, row in enumerate(profile_rows) for dtype in build_contract.DTYPES}
        descriptors = []
        for symbol in selected:
            descriptor = dict(by_symbol[symbol])
            if symbol in modes:
                descriptor["canonical_mode"] = modes[symbol]
            descriptor["object"] = f"{symbol}.o"
            descriptor["object_sha256"] = digest(a.output / f"{symbol}.o")
            descriptors.append(descriptor)
        development = {
            "schema": "infinity-fa4-development-export/v1",
            "kind": "development-only",
            "development_only": True,
            "production_admissible": False,
            "architecture": a.arch,
            "target_sm_count": int(os.environ["FLASH_ATTENTION_NUM_SMS"]),
            "host_target": host_target(),
            "export_env": export_env,
            "softmax_backport": softmax_backport,
            "gemma_sm100_local_source_patch": gemma_local_patch,
            "fa4_source_lock_sha256": digest(lock_path),
            "fa4_source_identity": {
                "before_import": source_identity_before,
                "after_import": source_identity_after,
            },
            "fa4_python_sources": {
                str(path.relative_to(pathlib.Path(interface.__file__).parent)): digest(path)
                for path in sorted(pathlib.Path(interface.__file__).parent.rglob("*.py"))},
            "export_runtime_pins": build_contract.runtime_pins(a.ffi_root, a.runtime_dir),
            "cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
            "tvm_ffi": importlib.metadata.version("apache-tvm-ffi"),
            "selected_symbols": selected,
            "descriptors": descriptors,
            "object_hashes": {f"{symbol}.o": digest(a.output / f"{symbol}.o")
                              for symbol in selected},
            "provenance": {
                "kind": "development-only-selected-symbol-export",
                "canonical_profile_count": len(profile_rows),
                "profile_sequence_sha256": build_contract.profile_sequence_sha(profile_rows),
                "selected_only": True,
            },
            "compile_cache_isolation": {
                "policy": "clear-cache-before-and-after-every-export",
                "exports": len(selected),
                "compile_callbacks": len(compile_only_state["calls"]),
                "cache_entries_after_each_compile": 1,
            },
        }
        (a.output / "development-manifest.json").write_text(
            json.dumps(development, indent=2))
        print(f"Exported {len(selected)} development-only symbols to {a.output}",
              flush=True)
        raise SystemExit(0)
    profile_names = [row[0] for row in profile_rows]
    kernel_flags = {
        "include_deberta": a.deberta,
        "include_deberta_exact_mul": a.deberta_exact_mul,
        "include_alibi": a.alibi,
        "include_jinacode": a.jinacode,
        "include_alibi12": a.alibi12,
    }
    kernels = build_contract.canonical_metadata_order(
        kernels, profile_names, **kernel_flags)
    expected_symbols = build_contract.validate_kernel_order(
        kernels, profile_names, **kernel_flags)
    build_contract.validate_window_contract(kernels, profile_names)
    link_object_order = [
        f"{symbol}.o" for symbol in build_contract.expected_link_symbols(
            profile_names, **kernel_flags)]
    exported_object_order = [pathlib.Path(obj).name for obj in objects]
    if exported_object_order != link_object_order:
        raise RuntimeError(
            "canonical38 export/link object order changed: "
            f"expected={link_object_order} actual={exported_object_order}")
    export_runtime_pins = build_contract.runtime_pins(a.ffi_root, a.runtime_dir)
    manifest = {
        "abi_version": 7,
        "registry": registry_metadata(),
        "gemma_sm100_local_source_patch": gemma_local_patch,
        "host_target": host_target(),
        "softmax_backport": softmax_backport,
        "architecture": a.arch,
        "executed_during_export": False,
        "target_sm_count": int(os.environ["FLASH_ATTENTION_NUM_SMS"]),
        "export_env": export_env,
        "persistent_jit_cache_disabled": True,
        "fa4_source_lock_sha256": digest(lock_path),
        "fa4_source_identity": {
            "before_import": source_identity_before,
            "after_import": source_identity_after,
        },
        "export_runtime_pins": export_runtime_pins,
        "canonical31_inventory": {
            "exact_order": True,
            "symbols": expected_symbols,
        },
        "link_object_order": link_object_order,
        "deberta_score_sha256": digest(pathlib.Path(__file__).with_name("deberta_score.py")) if a.deberta else None,
        "kernels": kernels,
        "fa4_python_sources": {str(f.relative_to(pathlib.Path(interface.__file__).parent)): digest(f)
                               for f in sorted(pathlib.Path(interface.__file__).parent.rglob("*.py"))},
        "cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
        "tvm_ffi": importlib.metadata.version("apache-tvm-ffi"),
        "objects": {pathlib.Path(obj).name: digest(pathlib.Path(obj)) for obj in objects},
    }
    if a.alibi:
        manifest["alibi_score_sha256"] = digest(pathlib.Path(__file__).with_name("alibi_score.py"))
    if a.jinacode:
        manifest["alibi12_native_score_sha256"] = digest(pathlib.Path(__file__).with_name("alibi12_native_score.py"))
    if a.alibi12:
        manifest["alibi12_model_bias_score_sha256"] = digest(pathlib.Path(__file__).with_name("alibi12_model_bias_score.py"))
    if a.deberta_exact_mul:
        manifest["deberta_mul_score_sha256"] = digest(pathlib.Path(__file__).with_name("deberta_mul_score.py"))
    if a.compile_only:
        if len(compile_only_state["calls"]) != len(kernels):
            raise RuntimeError(
                f"compile callback count {len(compile_only_state['calls'])} != kernel count {len(kernels)}")
        manifest["compile_only_engine_control"] = compile_only_state
        manifest["compile_cache_isolation"] = {
            "policy": "clear-cache-before-and-after-every-export",
            "exports": len(kernels),
            "compile_callbacks": len(compile_only_state["calls"]),
            "cache_entries_after_each_compile": 1,
        }
    build_contract.validate_manifest_inventory(
        manifest, profile_rows, require_all_helpers=False)
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
else:
    if (a.output / "development-manifest.json").exists():
        raise RuntimeError("refusing --link-only on a development-only export")
    manifest = json.loads((a.output / "manifest.json").read_text())
    if manifest["architecture"] != a.arch:
        raise RuntimeError("Link target differs from the exported architecture")
    for package, field in [("nvidia-cutlass-dsl", "cutlass_dsl"), ("apache-tvm-ffi", "tvm_ffi")]:
        if importlib.metadata.version(package) != manifest[field]:
            raise RuntimeError("Link runtime/compiler version differs from export: " + package)
    if a.ffi_root is None or a.runtime_dir is None:
        raise RuntimeError("Linking requires --runtime-dir and --ffi-root")
    if manifest.get("abi_version") != 7:
        raise RuntimeError("Re-export the native bundle for ABI version 7")
    if manifest.get("host_target") != host_target():
        raise RuntimeError("Re-export FA4 with the portable host target before linking")
    profiles_inc = pathlib.Path(__file__).with_name("profiles.inc")
    profile_rows = build_contract.validate_profiles_inc(profiles_inc)
    link_object_order = build_contract.validate_manifest_inventory(
        manifest, profile_rows)["link_object_order"]
    objects = [str(a.output / name) for name in link_object_order]
    for obj in objects:
        path = pathlib.Path(obj)
        if digest(path) != manifest["objects"][path.name]:
            raise RuntimeError(f"Export changed since manifest was created: {path}")
for obj in objects:
    validate_host_object(pathlib.Path(obj))
if not dev_only and (a.output / "development-manifest.json").exists():
    (a.output / "development-manifest.json").unlink()
if a.export_only:
    print(f"Exported kernels to {a.output}; link with a deployment-compatible C++ compiler", flush=True)
    raise SystemExit(0)
if a.runtime_dir is None or a.ffi_root is None:
    p.error("Linking requires --runtime-dir and --ffi-root")
build_contract.verify_runtime_pins(manifest, a.ffi_root, a.runtime_dir)
if manifest["architecture"] not in ["sm_80", "sm_86", "sm_89", "sm_90a", "sm_100a", "sm_120"]:
    raise RuntimeError("Unsupported bundle architecture")
capability = int(manifest["architecture"].removeprefix("sm_").removesuffix("a"))
bridge = pathlib.Path(__file__).with_name("bridge.cpp")
if any(k["symbol"].startswith(("fa4_alibi_model_bias_", "fa4_jinacode_flash_alibi_", "fa4_alibi12_model_bias_")) for k in manifest["kernels"]):
    from alibi_score import native_link_defines
    link_defines = native_link_defines(manifest["kernels"])
else:
    # Keep the pre-ALiBi link path independent of the additive module.
    link_defines = ["-DFA4_DEBERTA"] if any(k["symbol"].startswith("fa4_deberta_") for k in manifest["kernels"]) else []
if any(k["symbol"].startswith("fa4_deberta_exact_mul_") for k in manifest["kernels"]):
    link_defines.append("-DFA4_DEBERTA_EXACT_MUL")
registry = manifest.get("registry") or {}
registry_defines = []
if "mode_count" in registry:
    registry_defines.append(f"-DFA4_REGISTRY_MODE_COUNT={int(registry['mode_count'])}")
if "namespace" in registry:
    registry_defines.append(f'-DFA4_REGISTRY_NAMESPACE="{registry["namespace"]}"')
if not a.link_only:
    require_no_cuda("before link")
subprocess.run([a.cxx, "-std=c++17", "-O2", "-fPIC", "-shared", str(bridge),
                *objects, "-DINFINITY_FA4_PORTABLE_HOST=1", f"-DFA4_COMPUTE_CAPABILITY={capability}", f"-I{a.ffi_root / 'include'}",
                *registry_defines,
                f"-L{a.runtime_dir}", "-lcute_dsl_runtime",
                f"-L{a.ffi_root / 'lib'}", "-ltvm_ffi",
                "-Wl,-rpath,$ORIGIN", "-Wl,-z,defs",
                *link_defines,
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
    "runtime_pins_verified_on_link": True,
})
(a.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(f"Linked {len(manifest['kernels'])} kernels and native bridge in {a.output}", flush=True)
