#!/opt/torch-sdk/bin/python3.12
"""Matched physical-batch native CAPI quality runner.

The runner preserves one caller-provided physical batch exactly: no
tokenization, truncation, padding, reordering, splitting or rebatching.  It
validates the strict batch contract on the CPU, builds pooled-first then raw
segments, calls the staged ``tei_forward`` ABI directly, and records canonical
Rust ``Embeddings`` map semantics plus physical duplicate bit identity.

Torch and the staged native_base/verify_bindings modules are imported lazily by
runtime and validation paths only.  ``--validate-batch-only`` is pure CPU and
never imports Torch.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time

import model_checkpoint as mc
import native_batch as nb

ROOT = Path(__file__).resolve().parent
DEFAULT_LIBRARY = Path("/tmp/native214-fa4-mode20/install/lib/libtei_torch.so")
DEFAULT_HELPER = ROOT / "bench_helper.so"
CAPI_CANDIDATES = [
    Path("/tmp/infinity-plain-cuda/libs/infinity-v3/backends/libtorch/cpp"),
    Path("/tmp/oxide-infinity/libs/infinity-v3/backends/libtorch/cpp"),
    Path("/tmp/oxide-infinity/backends/libtorch/cpp"),
]
POOL_CODES = {"cls": 0, "mean": 1, "last-token": 2}
SCHEMA = "infinity-native-raw-probe/v1"
GRAPH_KEYS = ("hits", "misses", "captures", "attempts", "evictions",
              "live_bytes", "admitted_bytes")
QUIET_NAN = struct.pack("<I", 0x7FC00000)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def maybe_sha(path):
    try:
        return sha256_file(path) if Path(path).is_file() else None
    except OSError:
        return None


def resolve_capi(explicit):
    candidates = ([Path(explicit)] if explicit else []) + CAPI_CANDIDATES + [ROOT]
    for candidate in candidates:
        if (candidate / "bridge.h").is_file():
            return candidate.resolve()
    raise ValueError("missing CAPI bridge.h; pass --capi-cpp")


def source_hashes(capi):
    names = ["native_raw_probe.py", "native_batch.py", "native_base.py",
             "model_checkpoint.py", "verify_bindings.py"]
    result = {name: maybe_sha(ROOT / name) for name in names}
    result["bridge.h"] = maybe_sha(Path(capi) / "bridge.h")
    for name in ("bridge.cpp", "native_capi_bench.cpp", "build_helpers.py"):
        path = Path(capi) / name
        if path.is_file():
            result[name] = maybe_sha(path)
    return result


def load_model_metadata(model, pool, with_checkpoint):
    if pool not in POOL_CODES:
        raise ValueError(f"unsupported pool: {pool}")
    raw_config = mc.read_json(Path(model) / "config.json")
    case = mc.case_for_config(raw_config)
    base_pool = mc.CASES[case][7]
    config, profile = mc.normalize_config(raw_config, base_pool)
    profile = dict(profile)
    profile["base_pool"] = base_pool
    profile["requested_pool"] = pool
    result = {"model": str(Path(model).resolve()), "case": case,
              "base_pool": base_pool, "raw_config": raw_config,
              "config": config, "profile": profile,
              "config_sha256": maybe_sha(Path(model) / "config.json")}
    if with_checkpoint:
        checkpoint = mc.Checkpoint(model)
        stats = checkpoint.validate_shapes(config, profile)
        result["checkpoint"] = checkpoint
        result["weight_stats"] = stats
        result["checkpoint_metadata_sha256"] = hashlib.sha256(
            json.dumps(stats, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return result


def require_host_vendor_env():
    if os.environ.get("INFINITY_BENCH_REQUIRE_HOST_VENDOR") != "1":
        raise ValueError("native controls require INFINITY_BENCH_REQUIRE_HOST_VENDOR=1")


def node_paths(args):
    library = Path(args.library).resolve()
    helper = Path(args.helper).resolve()
    return library, helper


def native_sdk_record(args, library):
    return {"requested": str(args.library), "path": str(library),
            "default": str(DEFAULT_LIBRARY),
            "sha256": maybe_sha(library)}


def write_json(path, report):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def common_report(args, meta, batch_sha, batch, layout, capi, library, helper):
    return {
        "schema": SCHEMA,
        "backend": "native-capi-tei_forward",
        "native_sdk": native_sdk_record(args, library),
        "source_sha256": source_hashes(capi),
        "helper": {"path": str(helper), "sha256": maybe_sha(helper)},
        "helper_sha256": maybe_sha(helper),
        "profile": meta["profile"],
        "model": meta["model"],
        "pool": args.pool,
        "pool_code": POOL_CODES[args.pool],
        "device": args.device,
        "dtype": args.dtype,
        "width": int(meta["config"]["hidden_size"]),
        "physical_batch": batch,
        "batch_sha256": batch_sha,
        "batch_facts": nb.batch_facts(batch) if batch is not None else None,
        "layout": layout,
        "no_normalization": True,
        "runtime_graph_proof": False,
    }


def graph_counts(library, handle):
    counters = (C.c_uint64 * len(GRAPH_KEYS))()
    status = library.tei_graph_stats(handle, counters, len(GRAPH_KEYS))
    if status != 0:
        detail = library.tei_error()
        text = detail.decode() if detail else ""
        raise RuntimeError(f"tei_graph_stats failed (status {status}: {text})")
    return (dict(zip(GRAPH_KEYS, (int(value) for value in counters))),
            int(library.tei_graph_count(handle)))


def c_i64(values):
    return None if not values else (C.c_int64 * len(values))(*values)


def c_i32(values):
    return None if not values else (C.c_int32 * len(values))(*values)


def build_call_arrays(batch):
    return {
        "ids": c_i64(batch["input_ids"]),
        "types": c_i64(batch["token_type_ids"]),
        "positions": c_i64(batch["position_ids"]),
        "cumulative": c_i32(batch["cumulative_seq_lengths"]),
        "pooled": c_i64(batch["pooled_indices"]),
        "raw": c_i64(batch["raw_indices"]),
    }


def run_physical_calls(library, handle, batch, layout, pool_code, repeat):
    nb.require_little_endian()
    arrays = build_call_arrays(batch)
    capacity = layout["physical_float_count"]
    storage = nb.make_guarded_buffer(capacity)
    backing = (C.c_char * len(storage)).from_buffer(storage)
    output = C.cast(backing, C.POINTER(C.c_float))
    batch_count = len(batch["cumulative_seq_lengths"]) - 1
    records = []
    first_flat = None
    first_outputs = None
    final_flat = None
    final_outputs = None
    final_facts = None
    for call in range(repeat):
        if capacity:
            storage[:capacity * 4] = QUIET_NAN * capacity
        storage[capacity * 4:] = nb.guard_bytes()
        stats_before, graph_before = graph_counts(library, handle)
        started = time.perf_counter()
        status = library.tei_forward(
            handle, arrays["ids"], arrays["types"], arrays["positions"],
            arrays["cumulative"], batch_count, batch["max_length"], pool_code,
            arrays["pooled"], len(batch["pooled_indices"]),
            arrays["raw"], len(batch["raw_indices"]),
            output, capacity, None, None, 0, None, 0)
        if status != 0:
            detail = library.tei_error()
            kind = library.tei_last_error_kind()
            text = detail.decode() if detail else ""
            raise RuntimeError(f"tei_forward failed (status {status}, kind {kind}: {text})")
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        stats_after, graph_after = graph_counts(library, handle)
        nb.check_guards(storage, capacity)
        flat = bytes(storage[:capacity * 4])
        outputs, facts = nb.canonical_outputs(batch, layout, flat)
        if first_flat is None:
            first_flat, first_outputs = flat, outputs
        final_flat, final_outputs, final_facts = flat, outputs, facts
        records.append({
            "call": call,
            "elapsed_ms": elapsed_ms,
            "graph_count_before": graph_before,
            "graph_count_after": graph_after,
            "graph_stats_before": stats_before,
            "graph_stats_after": stats_after,
            "observed_replay": stats_after["hits"] > stats_before["hits"],
            "observed_capture": (stats_after["captures"] > stats_before["captures"]
                                 or graph_after > graph_before),
            "physical_sha256": hashlib.sha256(flat).hexdigest(),
            "bit_identical_to_first": flat == first_flat,
            "canonical_identical_to_first": outputs == first_outputs,
        })
    return records, final_outputs, final_facts, final_flat, repeat - 1


def validate_batch_only(args):
    batch_sha, batch = nb.load_batch(args.batch)
    meta = load_model_metadata(args.model, args.pool, with_checkpoint=False)
    layout = nb.physical_layout(batch, meta["config"]["hidden_size"])
    capi = resolve_capi(args.capi_cpp)
    library, helper = node_paths(args)
    if "torch" in sys.modules:
        raise RuntimeError("--validate-batch-only imported Torch")
    report = common_report(args, meta, batch_sha, batch, layout, capi, library, helper)
    report.update({
        "status": "batch-validation-only",
        "mode": "pure-cpu-no-torch",
        "torch_imported": False,
        "cuda_initialized": False,
        "runtime_graph_proof": False,
        "checkpoint": None,
        "vendor_binding_proof": None,
        "note": "batch contract/layout only; no SDK load, Torch import, engine or CUDA call",
    })
    write_json(args.output, report)
    print(json.dumps({
        "status": report["status"], "case": meta["case"],
        "batch_sha256": batch_sha, "pool": args.pool,
        "physical_float_count": layout["physical_float_count"],
        "output": str(args.output),
    }), flush=True)


def validate_only(args):
    batch = None
    batch_sha = None
    layout = None
    if args.batch:
        batch_sha, batch = nb.load_batch(args.batch)
    meta = load_model_metadata(args.model, args.pool, with_checkpoint=True)
    if batch is not None:
        layout = nb.physical_layout(batch, meta["config"]["hidden_size"])
    library, helper = node_paths(args)
    for path, label in [(library, "native library"), (helper, "helper")]:
        if not path.is_file():
            raise SystemExit(f"missing {label}: {path}")
    require_host_vendor_env()
    capi = resolve_capi(args.capi_cpp)
    abi = nb.check_forward_abi((capi / "bridge.h").read_text())
    import torch
    before = bool(torch.cuda.is_initialized())
    import native_base
    import verify_bindings
    if not hasattr(torch.ops.oxide_native_capi_bench, "bench"):
        torch.ops.load_library(str(helper))
    helper_loaded = hasattr(torch.ops.oxide_native_capi_bench, "bench")
    binding = verify_bindings.verify(library)
    if (before or binding["gpu_initialized"]
            or not binding["all_blas_relocations_host_vendor"]
            or binding["probe_dependency_cublaslt_version"] != 130200):
        raise RuntimeError("host-vendor relocation proof failed before CUDA initialization")
    after = bool(torch.cuda.is_initialized())
    if after:
        raise RuntimeError("CPU validation unexpectedly initialized CUDA")
    options = native_base.build_options(meta["raw_config"], args.pool)
    report = common_report(args, meta, batch_sha, batch, layout, capi, library, helper)
    report.update({
        "status": "cpu-validation-only",
        "mode": "metadata-sdk-vendor-bindings",
        "native_config": meta["config"],
        "raw_options": options,
        "raw_options_sha256": hashlib.sha256(
            json.dumps(options, sort_keys=True, allow_nan=False).encode()).hexdigest(),
        "checkpoint": meta["weight_stats"],
        "checkpoint_metadata_sha256": meta["checkpoint_metadata_sha256"],
        "abi": abi,
        "graphs_env_requested": bool(options["_cuda_graphs"]),
        "helper_loaded_for_dependency_proof": helper_loaded,
        "vendor_binding_proof": binding,
        "cuda_initialized_before": before,
        "cuda_initialized_after": after,
        "cuda_initialized": False,
        "engine_created": False,
        "runtime_graph_proof": False,
        "runtime_graph_proof_reason": "validate-only never creates or runs an engine",
        "no_runtime_graph_proof": True,
    })
    write_json(args.output, report)
    print(json.dumps({
        "status": report["status"], "case": meta["case"], "pool": args.pool,
        "cuda_initialized": False, "engine_created": False,
        "vendor_version": binding["probe_dependency_cublaslt_version"],
        "output": str(args.output),
    }), flush=True)


def run_runtime(args):
    if args.repeat < 1:
        raise ValueError("--repeat must be at least 1")
    if args.dtype != "bfloat16":
        raise ValueError("these native controls require BF16 serving")
    batch_sha, batch = nb.load_batch(args.batch)
    meta = load_model_metadata(args.model, args.pool, with_checkpoint=True)
    layout = nb.physical_layout(batch, meta["config"]["hidden_size"])
    library_path, helper = node_paths(args)
    for path, label in [(library_path, "native library"), (helper, "helper")]:
        if not path.is_file():
            raise SystemExit(f"missing {label}: {path}")
    require_host_vendor_env()
    capi = resolve_capi(args.capi_cpp)
    abi = nb.check_forward_abi((capi / "bridge.h").read_text())
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    import native_base
    import verify_bindings
    if not hasattr(torch.ops.oxide_native_capi_bench, "bench"):
        torch.ops.load_library(str(helper))
    if not hasattr(torch.ops.oxide_native_capi_bench, "bench"):
        raise RuntimeError("oxide_native_capi_bench::bench did not register")
    binding = verify_bindings.verify(library_path)
    if (binding["gpu_initialized"]
            or not binding["all_blas_relocations_host_vendor"]
            or binding["probe_dependency_cublaslt_version"] != 130200):
        raise RuntimeError("host-vendor relocation proof failed before GPU initialization")
    options = native_base.build_options(meta["raw_config"], args.pool)
    graphs_enabled = bool(options["_cuda_graphs"])
    if graphs_enabled and not args.device.startswith("cuda"):
        raise ValueError("CUDA graphs require a CUDA device")
    library = native_base.bind_library(library_path)
    if len(library.tei_forward.argtypes) != 19:
        raise RuntimeError("staged bind_library tei_forward ABI is not 19 arguments")
    handle = None
    try:
        handle = native_base.create_engine(library, meta["config"], options,
                                           args.device, 2, True)
        native_base.upload_weights(library, handle, meta["checkpoint"])
        native_base.check_status(library, library.tei_ready(handle), "tei_ready")
        width = int(meta["config"]["hidden_size"])
        output_width = int(library.tei_output_width(handle))
        pooled_width = int(library.tei_pooled_width(handle))
        if output_width != width or pooled_width != width:
            raise RuntimeError(f"unexpected native widths: expected={width} "
                               f"output={output_width} pooled={pooled_width}")
        records, outputs, canonical, flat, result_call_index = run_physical_calls(
            library, handle, batch, layout, POOL_CODES[args.pool], args.repeat)
        report = common_report(args, meta, batch_sha, batch, layout, capi,
                               library_path, helper)
        report.update({
            "status": "ok",
            "native_config": meta["config"],
            "raw_options": options,
            "raw_options_sha256": hashlib.sha256(
                json.dumps(options, sort_keys=True, allow_nan=False).encode()).hexdigest(),
            "checkpoint": meta["weight_stats"],
            "checkpoint_metadata_sha256": meta["checkpoint_metadata_sha256"],
            "abi": abi,
            "repeat": args.repeat,
            "result_call_index": result_call_index,
            "outputs": outputs,
            "canonical": canonical,
            "finite": True,
            "first_native_flat_sha256": records[0]["physical_sha256"],
            "native_flat_sha256": hashlib.sha256(flat).hexdigest(),
            "vendor_binding_proof": binding,
            "graphs": {
                "requested_env": {name: os.environ.get(name) for name in (
                    "INFINITY_TORCH_CUDA_GRAPHS", "INFINITY_TORCH_CUDA_GRAPH_REUSE",
                    "INFINITY_TORCH_CUDA_GRAPH_MAX_TOKENS",
                    "INFINITY_TORCH_CUDA_GRAPH_ENTRIES",
                    "INFINITY_TORCH_CUDA_GRAPH_MAX_BYTES",
                    "INFINITY_TORCH_CUDA_GRAPH_MAX_CAPTURES")},
                "options_requested": graphs_enabled,
                "observed_replay": any(call["observed_replay"] for call in records),
                "observed_capture": any(call["observed_capture"] for call in records),
                "runtime_graph_proof": False,
                "note": "env/options only request graphs; replay requires an observed hits increase",
            },
            "calls": records,
            "repeat_bit_identity": all(call["bit_identical_to_first"] for call in records),
        })
        if not report["repeat_bit_identity"]:
            report["status"] = "repeat-bit-difference"
        write_json(args.output, report)
        print(json.dumps({
            "status": report["status"], "case": meta["case"], "pool": args.pool,
            "batch_sha256": batch_sha, "repeat": args.repeat,
            "physical_float_count": layout["physical_float_count"],
            "entry_count": canonical["entry_count"],
            "result_call_index": result_call_index,
            "repeat_bit_identity": report["repeat_bit_identity"],
            "observed_replay": report["graphs"]["observed_replay"],
            "observed_capture": report["graphs"]["observed_capture"],
            "output": str(args.output),
        }), flush=True)
    finally:
        if handle is not None:
            library.tei_destroy(handle)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch", default=None)
    parser.add_argument("--pool", required=True, choices=sorted(POOL_CODES))
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--library",
                        default=os.environ.get("INFINITY_NATIVE_SDK", str(DEFAULT_LIBRARY)))
    parser.add_argument("--helper", default=str(DEFAULT_HELPER))
    parser.add_argument("--capi-cpp", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--validate-batch-only", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.validate_only and args.validate_batch_only:
        raise SystemExit("choose --validate-only or --validate-batch-only")
    if args.validate_batch_only:
        if not args.batch:
            raise SystemExit("--validate-batch-only requires --batch")
        validate_batch_only(args)
        return
    if args.validate_only:
        validate_only(args)
        return
    if not args.batch:
        raise SystemExit("--batch is required unless --validate-only is used")
    run_runtime(args)


if __name__ == "__main__":
    main()
