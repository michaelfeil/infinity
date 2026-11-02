#!/opt/torch-sdk/bin/python3.12
"""Direct isolated profile20-LibTorch CAPI control for the oxide native probe.

The runner creates the native engine exactly as the Rust LibTorch backend does
for the five validated cached encoder/Qwen2 controls plus the two dense BF16
Qwen3 embedding controls, uploads one typed safetensors view at a time
(including indexed shards), lets the C++ adapter time synchronous tei_forward
calls, and writes the same samples_ms/embedding/tokens shape used by the oxide
probe. Only the external qwen3-native-control staged files are used.
"""
import argparse
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

import torch
from model_checkpoint import Checkpoint, normalize_config, read_json

ROOT = Path(__file__).resolve().parent
CAPI_CANDIDATES = [
    Path("/tmp/infinity-plain-cuda/libs/infinity-v3/backends/libtorch/cpp"),
    Path("/tmp/oxide-infinity/libs/infinity-v3/backends/libtorch/cpp"),
    Path("/tmp/oxide-infinity/backends/libtorch/cpp"),
]
DEFAULT_LIBRARY = Path("/tmp/native214-fa4-mode20/install/lib/libtei_torch.so")
DEFAULT_MODEL = Path("/tmp/oxide-minilm")
POOL_CODES = {"cls": 0, "mean": 1, "last-token": 2}
POOL_OPTIONS = {"cls": "cls", "mean": "mean", "last-token": "last_token"}
SUPPORTED_CONFIGS = {
    "qwen3-0.6b": {"directory": "Qwen--Qwen3-Embedding-0.6B",
                   "hidden": 1024, "heads": 16, "layers": 28, "intermediate": 3072,
                   "vocab": 151669, "positions": 32768, "qwidth": 2048, "kv_width": 1024},
    "qwen3-8b": {"directory": "Qwen--Qwen3-Embedding-8B",
                 "hidden": 4096, "heads": 32, "layers": 36, "intermediate": 12288,
                 "vocab": 151665, "positions": 40960, "qwidth": 4096, "kv_width": 1024},
}


def supported_config(model, config, profile):
    """Bind an admitted Qwen3 recipe to the actual staged model directory."""
    case = profile["case"]
    spec = SUPPORTED_CONFIGS.get(case)
    if spec is None:
        return None
    actual = (config["hidden_size"], config["num_attention_heads"],
              config["num_hidden_layers"], config["intermediate_size"],
              config["vocab_size"], config["max_position_embeddings"],
              profile["qwidth"], profile["kv_width"])
    expected = (spec["hidden"], spec["heads"], spec["layers"], spec["intermediate"],
                spec["vocab"], spec["positions"], spec["qwidth"], spec["kv_width"])
    if actual != expected or Path(model).name != spec["directory"]:
        raise ValueError(f"{case} requires actual model config/directory "
                         f"{spec['directory']}: {actual} != {expected}")
    return {"case": case, "model_directory": spec["directory"], "dtype": "bfloat16",
            "pooling": profile["pool"], "qwidth": profile["qwidth"],
            "kv_width": profile["kv_width"], "head_dim": profile["head_dim"],
            "q_norm": True, "k_norm": True, "source_dtype": profile["source_dtype"]}


def resolve_capi(explicit):
    candidates = ([Path(explicit)] if explicit else []) + CAPI_CANDIDATES
    for candidate in candidates:
        if (candidate / "bridge.h").is_file():
            return candidate.resolve()
    raise ValueError("missing CAPI bridge.h; pass --capi-cpp")


class TeiConfig(C.Structure):
    _fields_ = [(name, C.c_int64) for name in
                ["hidden", "heads", "layers", "intermediate", "vocab", "positions", "types"]]
    _fields_ += [("epsilon", C.c_double), ("activation", C.c_int32)]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def env_flag(name):
    return os.environ.get(name, "").lower() in ("1", "true")


def env_int(name, default, minimum):
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw, 10)
    except ValueError as exc:
        raise ValueError(f"invalid {name}: {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def env_byte_size(name):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "0":
        return 0
    value = raw.strip()
    if value.endswith("GiB"):
        amount, unit = value[:-3].strip(), 1 << 30
    elif value.endswith("GB"):
        amount, unit = value[:-2].strip(), 1_000_000_000
    elif value.endswith("MiB"):
        amount, unit = value[:-3].strip(), 1 << 20
    elif value.endswith("MB"):
        amount, unit = value[:-2].strip(), 1_000_000
    else:
        raise ValueError(f"invalid {name}: {raw!r} (expected GB or GiB)")
    try:
        return int(float(amount) * unit)
    except ValueError as exc:
        raise ValueError(f"invalid {name}: {raw!r}") from exc


def flatten_json(value, key="", output=None):
    if output is None:
        output = []
    if isinstance(value, dict):
        for name, item in value.items():
            flatten_json(item, f"{key}.{name}" if key else name, output)
    elif isinstance(value, list):
        output.append((key, json.dumps(value, separators=(",", ":"))))
        for index, item in enumerate(value):
            flatten_json(item, f"{key}.{index}", output)
    elif value is None:
        pass
    elif isinstance(value, str):
        output.append((key, value))
    else:
        output.append((key, json.dumps(value, separators=(",", ":"))))
    return output


def build_options(config, pool):
    options = json.loads(json.dumps(config))
    options["_classifier"] = False
    options["_multi_vector"] = False
    options["_dense_count"] = 0
    options["_pooling"] = "embedding"
    options["_embedding_pool"] = POOL_OPTIONS[pool]
    options["_fp8_dynamic"] = False
    requested_scope = os.environ.get("INFINITY_TORCH_FP8_SCOPE")
    if requested_scope is None:
        options["_fp8_scope"] = "mlp"
    elif requested_scope == "mlp":
        options["_fp8_scope"] = "mlp"
    elif requested_scope == "all":
        raise ValueError("INFINITY_TORCH_FP8_SCOPE=all requires dynamic FP8; this baseline is fp8=false")
    else:
        raise ValueError("INFINITY_TORCH_FP8_SCOPE must be mlp or all")
    options["_prefix_cache_max_bytes"] = env_byte_size("INFINITY_TORCH_PREFIX_CACHE_SIZE")
    options["_prefix_cache_max_tokens"] = env_int("INFINITY_TORCH_PREFIX_CACHE_MAX_TOKENS", 8192, 1)
    options["_prefix_cache_entries"] = env_int("INFINITY_TORCH_PREFIX_CACHE_ENTRIES", 64, 1)
    options["_cuda_graphs"] = env_flag("INFINITY_TORCH_CUDA_GRAPHS")
    options["_media_cuda_graphs"] = env_flag("INFINITY_TORCH_MEDIA_CUDA_GRAPHS")
    options["_cuda_graph_max_tokens"] = env_int("INFINITY_TORCH_CUDA_GRAPH_MAX_TOKENS", 4096, 1)
    options["_retained_workspace_max_bytes"] = env_int(
        "INFINITY_TORCH_RETAINED_WORKSPACE_MAX_BYTES", 512 << 20, 1)
    options["_input_upload_max_bytes"] = env_int(
        "INFINITY_TORCH_INPUT_UPLOAD_MAX_BYTES", 512 << 20, 1)
    options["_output_staging_max_bytes"] = env_int(
        "INFINITY_TORCH_OUTPUT_STAGING_MAX_BYTES", 512 << 20, 1)
    options["_cuda_graph_entries"] = env_int("INFINITY_TORCH_CUDA_GRAPH_ENTRIES", 4, 0)
    options["_attention_bias_max_bytes"] = env_int(
        "INFINITY_TORCH_ATTENTION_BIAS_MAX_BYTES", 512 << 20, 1)
    options["_cuda_graph_reuse"] = env_int("INFINITY_TORCH_CUDA_GRAPH_REUSE", 3, 1)
    options["_cuda_graph_max_bytes"] = env_int(
        "INFINITY_TORCH_CUDA_GRAPH_MAX_BYTES", 512 << 20, 1)
    options["_cuda_graph_max_captures"] = env_int(
        "INFINITY_TORCH_CUDA_GRAPH_MAX_CAPTURES", 8, 0)
    options["_cudnn_varlen"] = env_flag("INFINITY_TORCH_CUDNN_VARLEN")
    return options


def bind_library(path):
    library = C.CDLL(str(path), mode=C.RTLD_GLOBAL)
    library.tei_error.restype = C.c_char_p
    library.tei_last_error_kind.restype = C.c_int32
    library.tei_create.argtypes = [C.POINTER(TeiConfig), C.c_char_p, C.c_int32]
    library.tei_create.restype = C.c_void_p
    library.tei_device_preflight.argtypes = [C.c_char_p]
    library.tei_device_preflight.restype = C.c_int32
    library.tei_option.argtypes = [C.c_void_p, C.c_char_p, C.c_char_p]
    library.tei_option.restype = C.c_int32
    library.tei_weight.argtypes = [C.c_void_p, C.c_char_p, C.c_void_p,
                                   C.POINTER(C.c_int64), C.c_size_t, C.c_int32]
    library.tei_weight.restype = C.c_int32
    library.tei_ready.argtypes = [C.c_void_p]
    library.tei_ready.restype = C.c_int32
    library.tei_output_width.argtypes = [C.c_void_p]
    library.tei_output_width.restype = C.c_int64
    library.tei_pooled_width.argtypes = [C.c_void_p]
    library.tei_pooled_width.restype = C.c_int64
    library.tei_graph_count.argtypes = [C.c_void_p]
    library.tei_graph_count.restype = C.c_int64
    library.tei_graph_stats.argtypes = [C.c_void_p, C.POINTER(C.c_uint64), C.c_size_t]
    library.tei_graph_stats.restype = C.c_int32
    library.tei_forward.argtypes = [
        C.c_void_p, C.POINTER(C.c_int64), C.POINTER(C.c_int64),
        C.POINTER(C.c_int64), C.POINTER(C.c_int32), C.c_int64, C.c_int64,
        C.c_int32, C.POINTER(C.c_int64), C.c_size_t, C.POINTER(C.c_int64),
        C.c_size_t, C.POINTER(C.c_float), C.c_size_t, C.c_void_p, C.c_void_p,
        C.c_size_t, C.c_void_p, C.c_size_t,
    ]
    library.tei_forward.restype = C.c_int32
    library.tei_destroy.argtypes = [C.c_void_p]
    library.tei_destroy.restype = None
    return library


def library_error(library):
    text = library.tei_error()
    detail = text.decode() if text else ""
    kind = library.tei_last_error_kind()
    return f"status kind {kind}" + (f": {detail}" if detail else "")


def check_status(library, status, operation):
    if status != 0:
        raise RuntimeError(f"{operation} failed ({library_error(library)})")


def create_engine(library, config, options, device, dtype_code, preflight):
    activation = {"gelu": 0, "gelu_new": 1, "gelu_pytorch_tanh": 1,
                  "relu": 2, "silu": 3}.get(config["hidden_act"], 0)
    # Exact Rust NativeConfig defaults; effective decoder/Distil epsilon
    # remains controlled by the untouched raw options sent below.
    epsilon = config["layer_norm_eps"]
    types = config["type_vocab_size"]
    native = TeiConfig(
        config["hidden_size"],
        config["num_attention_heads"],
        config["num_hidden_layers"],
        config["intermediate_size"],
        config["vocab_size"],
        config["max_position_embeddings"],
        types,
        epsilon,
        activation,
    )
    if device.startswith("cuda") and preflight:
        check_status(library, library.tei_device_preflight(device.encode()), "tei_device_preflight")
    handle = library.tei_create(C.byref(native), device.encode(), dtype_code)
    if not handle:
        raise RuntimeError(f"tei_create failed ({library_error(library)})")
    try:
        for key, value in sorted(flatten_json(options)):
            check_status(library, library.tei_option(handle, key.encode(), value.encode()),
                         f"tei_option {key}")
    except Exception:
        library.tei_destroy(handle)
        raise
    return handle


def upload_weights(library, handle, checkpoint):
    codes = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}
    def upload_one(name, value):
        shape = (C.c_int64 * value.dim())(*value.shape)
        check_status(library, library.tei_weight(
            handle, name.encode(), C.c_void_p(value.data_ptr()),
            shape, value.dim(), codes[value.dtype]), f"tei_weight {name}")
    checkpoint.upload(torch, upload_one)

def policy_metadata():
    result = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_initialized": torch.cuda.is_initialized(),
    }
    for name, getter in {
        "preferred_blas_library": lambda: torch.backends.cuda.preferred_blas_library(),
        "allow_bf16_reduced_precision_reduction":
            lambda: torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "allow_fp16_reduced_precision_reduction":
            lambda: torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "allow_tf32": lambda: torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": lambda: torch.backends.cudnn.allow_tf32,
        "deterministic_algorithms": lambda: torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": lambda: torch.get_float32_matmul_precision(),
    }.items():
        try:
            value = getter()
            if isinstance(value, (str, int, float, bool)) or value is None:
                result[name] = value
            elif hasattr(value, "name"):
                result[name] = value.name
            else:
                result[name] = str(value)
        except Exception as exc:  # metadata is diagnostic only
            result[name] = f"unavailable: {exc}"
    return result


class DlInfo(C.Structure):
    _fields_ = [("name", C.c_char_p), ("base", C.c_void_p),
                ("symbol", C.c_char_p), ("address", C.c_void_p)]


def symbol_path(function):
    try:
        libdl = C.CDLL("libdl.so.2")
        libdl.dladdr.argtypes = [C.c_void_p, C.POINTER(DlInfo)]
        libdl.dladdr.restype = C.c_int
        info = DlInfo()
        pointer = C.cast(function, C.c_void_p)
        if libdl.dladdr(pointer, C.byref(info)) and info.name:
            return info.name.decode()
    except Exception:
        pass
    return None


def mapped_libraries():
    found = {}
    try:
        lines = Path("/proc/self/maps").read_text().splitlines()
    except OSError:
        return found
    for line in lines:
        parts = line.split()
        if len(parts) < 6:
            continue
        path = parts[-1]
        if path.endswith(" (deleted)"):
            path = path[:-10]
        name = os.path.basename(path)
        if name.startswith(("libcublas", "libcublasLt", "libcudart", "libcudnn")):
            found.setdefault(name, set()).add(path)
    return {name: sorted(paths) for name, paths in sorted(found.items())}


def probe_cublas(mapped):
    torch_lib = Path(torch.__file__).resolve().parent
    search = []
    for entry in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        if entry:
            search.append(Path(entry))
    search.extend([
        Path("/usr/local/cuda/lib64"),
        Path("/usr/local/cuda-13.1/lib64"),
        torch_lib.parent / "nvidia/cu13/lib",
        torch_lib.parent / "nvidia/cublas/lib",
        Path("/tmp/oxide-host-cublas-130200"),
    ])
    candidates = []
    for basename in ("libcublasLt.so.13", "libcublas.so.13"):
        candidates.extend(mapped.get(basename, []))
    if not candidates:
        for directory in search:
            candidate = directory / "libcublasLt.so.13"
            if candidate.is_file():
                candidates.append(str(candidate))
                break
    results = []
    seen = set()
    for path in candidates:
        path = str(Path(path).resolve())
        if path in seen:
            continue
        seen.add(path)
        item = {"path": path}
        try:
            library = C.CDLL(path, mode=C.RTLD_LOCAL)
            for symbol in ("cublasLtGetVersion", "cublasLtGetCudartVersion"):
                try:
                    function = getattr(library, symbol)
                    function.argtypes = []
                    function.restype = C.c_size_t
                    item[symbol] = int(function())
                    item[f"{symbol}_path"] = symbol_path(function)
                except Exception as exc:
                    item[f"{symbol}_error"] = str(exc)
        except Exception as exc:
            item["load_error"] = str(exc)
        results.append(item)
    return results


def vendor_metadata():
    mapped = mapped_libraries()
    return {
        "policy": policy_metadata(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "environment": {
            "LD_PRELOAD": os.environ.get("LD_PRELOAD"),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
        },
        "mapped_cuda_libraries": mapped,
        "cublas_probe": probe_cublas(mapped),
        "notes": [
            "Mapped or dladdr-resolved paths alone do not prove which vendor "
            "LibTorch/native BLAS calls bind to, and say nothing about GPU performance.",
            "Use LD_PRELOAD with the host 13.2.0.0 cublas pair to control the vendor; "
            "default native 13.1.0.1 remains the fallback.",
        ],
    }


def percentile(sorted_values, fraction):
    if not sorted_values:
        return None
    return sorted_values[min(len(sorted_values) - 1,
                             max(0, int(fraction * (len(sorted_values) - 1))))]


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--library", default=str(DEFAULT_LIBRARY))
    parser.add_argument("--helper", default=str(ROOT / "bench_helper.so"))
    parser.add_argument("--capi-cpp", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--pool", default="mean", choices=sorted(POOL_CODES))
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-preflight", action="store_true")
    parser.add_argument("--validate-only", action="store_true", help="CPU-only metadata and vendor-binding proof; never creates or runs an engine")
    args = parser.parse_args()

    if args.tokens <= 0 or args.iterations <= 0 or args.warmup < 0:
        raise SystemExit("tokens and iterations must be positive; warmup must be non-negative")
    model = Path(args.model).resolve()
    library_path = Path(args.library).resolve()
    helper_path = Path(args.helper).resolve()
    capi = resolve_capi(args.capi_cpp)
    for path, label in [(model / "config.json", "config"),
                        (library_path, "native library"), (helper_path, "helper")]:
        if not path.is_file():
            raise SystemExit(f"missing {label}: {path}")

    raw_config = read_json(model / "config.json")
    config, profile = normalize_config(raw_config, args.pool)
    supported = supported_config(model, config, profile)
    if args.dtype != "bfloat16":
        raise ValueError("these cached native controls require BF16 serving")
    if args.tokens > config["max_position_embeddings"] or args.tokens > (1 << 20):
        raise ValueError("synthetic explicit position range exceeds checkpoint/helper capacity")
    checkpoint = Checkpoint(model)
    weight_stats = checkpoint.validate_shapes(config, profile)
    checkpoint_metadata_sha256 = hashlib.sha256(json.dumps(weight_stats, sort_keys=True).encode()).hexdigest()
    options = build_options(raw_config, args.pool)
    graphs_enabled = bool(options["_cuda_graphs"])
    if os.environ.get("INFINITY_BENCH_REQUIRE_HOST_VENDOR") != "1":
        raise ValueError("strong native controls require INFINITY_BENCH_REQUIRE_HOST_VENDOR=1")
    if not hasattr(torch.ops.oxide_native_capi_bench, "bench"):
        torch.ops.load_library(str(helper_path))
    if not hasattr(torch.ops.oxide_native_capi_bench, "bench"):
        raise RuntimeError("oxide_native_capi_bench::bench did not register")

    binding_proof = None
    if os.environ.get("INFINITY_BENCH_REQUIRE_HOST_VENDOR") == "1":
        from verify_bindings import verify
        binding_proof = verify(library_path)
        if (binding_proof["gpu_initialized"]
                or not binding_proof["all_blas_relocations_host_vendor"]
                or binding_proof["probe_dependency_cublaslt_version"] != 130200):
            raise RuntimeError("Host-vendor relocation proof failed before GPU initialization")

    if args.validate_only:
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU validation unexpectedly initialized CUDA")
        report = {"status": "cpu-validation-only", "profile": profile,
                  "native_config": config, "raw_options_preserved": True,
                  "supported_config": supported,
                  "checkpoint": weight_stats, "checkpoint_metadata_sha256": checkpoint_metadata_sha256,
                  "vendor_binding_proof": binding_proof, "helper_sha256": sha256_file(helper_path),
                  "library_sha256": sha256_file(library_path), "cuda_initialized": False,
                  "runtime_graphs_validated": False}
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"status": report["status"], "profile": profile["case"],
                          "supported_config": supported,
                          "floating_tensors": weight_stats["floating_tensors"],
                          "shards": len(weight_stats["files"]), "cuda_initialized": False,
                          "vendor_version": binding_proof["probe_dependency_cublaslt_version"]}), flush=True)
        return
    if not graphs_enabled:
        raise ValueError("strong native controls require INFINITY_TORCH_CUDA_GRAPHS=1")
    if graphs_enabled and not args.device.startswith("cuda"):
        raise ValueError("INFINITY_TORCH_CUDA_GRAPHS=1 requires a CUDA device; refusing to "
                         "mislable a CPU run as graph-enabled")
    if args.device.startswith("cuda") and args.dtype not in ("float16", "bfloat16"):
        raise ValueError("the native Rust backend requires float16 or bfloat16 on CUDA")
    dtype_codes = {"auto": 0, "float32": 0, "float16": 1, "bfloat16": 2}
    dtype_code = dtype_codes[args.dtype]
    pool_code = POOL_CODES[args.pool]
    expected_width = int(config["hidden_size"])

    library = bind_library(library_path)
    handle = None
    try:
        handle = create_engine(library, config, options, args.device, dtype_code,
                               not args.no_preflight)
        upload_weights(library, handle, checkpoint)
        check_status(library, library.tei_ready(handle), "tei_ready")
        output_width = library.tei_output_width(handle)
        pooled_width = library.tei_pooled_width(handle)
        if output_width != expected_width or pooled_width != expected_width:
            raise RuntimeError(
                f"unexpected native widths: expected={expected_width} "
                f"output={output_width} pooled={pooled_width}")

        samples_us, embedding, graph_tensor = torch.ops.oxide_native_capi_bench.bench(
            int(handle), args.tokens, args.iterations, args.warmup, pool_code)
        samples_ms = (samples_us.double().cpu() / 1000.0).tolist()
        embedding_values = embedding.float().cpu().reshape(-1).tolist()
        if (len(embedding_values) != expected_width
                or not all(math.isfinite(v) for v in embedding_values)):
            raise RuntimeError("benchmark adapter returned an invalid embedding")
        graph_values = [int(v) for v in graph_tensor.cpu().tolist()]
        if len(graph_values) != 10:
            raise RuntimeError("benchmark adapter returned malformed graph stats")
        keys = ["graph_count", "hits", "misses", "captures", "attempts", "evictions",
                "live_bytes", "admitted_bytes", "prefix_hits", "prefix_captures"]
        graph_stats = dict(zip(keys, graph_values))

        if graphs_enabled:
            if (graph_stats["graph_count"] <= 0 or graph_stats["captures"] <= 0
                    or graph_stats["hits"] < args.iterations or graph_stats["attempts"] < graph_stats["captures"]):
                graph_validation = {
                    "ok": False,
                    "reason": "graphs requested but capture/hit counters are incomplete",
                }
            else:
                graph_validation = {"ok": True}
        else:
            if (graph_stats["graph_count"] != 0 or graph_stats["captures"] != 0
                    or graph_stats["hits"] != 0):
                graph_validation = {
                    "ok": False,
                    "reason": "graphs disabled but graph counters are nonzero",
                }
            else:
                graph_validation = {"ok": True}

        ordered = sorted(samples_ms)
        summary = {
            "samples": len(samples_ms),
            "mean_ms": statistics.fmean(samples_ms),
            "p50_ms": statistics.median(samples_ms),
            "p99_ms": percentile(ordered, 0.99),
            "min_ms": ordered[0],
            "max_ms": ordered[-1],
        }
        report = {
            "scope": f"Direct isolated profile20 LibTorch CAPI {config['model_type']} baseline, "
                     f"one sequence, {args.pool} pooling, no normalization, "
                     "synchronous tei_forward",
            "model_type": config["model_type"],
            "profile": profile,
            "supported_config": supported,
            "native_config": config,
            "pool": args.pool,
            "device": args.device,
            "dtype": args.dtype,
            "tokens": args.tokens,
            "iterations": args.iterations,
            "warmup": args.warmup,
            "graphs_enabled": graphs_enabled,
            "graph_stats": graph_stats,
            "graphs_validation": graph_validation,
            "samples_ms": samples_ms,
            "embedding": embedding_values,
            "timing": summary,
            "weights": weight_stats,
            "provenance": {
                "library": str(library_path),
                "library_sha256": sha256_file(library_path),
                "checkpoint": str(model),
                "checkpoint_metadata_sha256": checkpoint_metadata_sha256,
                "config_sha256": sha256_file(model / "config.json"),
                "capi_include": str(capi),
                "source_sha256": {
                    "bridge.h": sha256_file(capi / "bridge.h"),
                    "bridge.cpp": sha256_file(capi / "bridge.cpp"),
                    "native_capi_bench.cpp": sha256_file(ROOT / "native_capi_bench.cpp"),
                    "build_helpers.py": sha256_file(ROOT / "build_helpers.py"),
                    "run_native_bench.py": sha256_file(Path(__file__)),
                    "model_checkpoint.py": sha256_file(ROOT / "model_checkpoint.py"),
                },
                "helper": str(helper_path),
                "helper_sha256": sha256_file(helper_path),
            },
            "vendor": vendor_metadata(),
        }
        report["status"] = "ok" if graph_validation["ok"] else "graphs-validation-failed"
        report["vendor"]["binding_proof"] = binding_proof
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")

        print(json.dumps({
            "status": report["status"],
            "model_type": config["model_type"],
            "pool": args.pool,
            "device": args.device,
            "dtype": args.dtype,
            "tokens": args.tokens,
            "graphs_enabled": graphs_enabled,
            "timing": summary,
            "graph_stats": graph_stats,
            "graphs_validation": graph_validation,
            "library_sha256": report["provenance"]["library_sha256"],
            "checkpoint_metadata_sha256": report["provenance"]["checkpoint_metadata_sha256"],
            "output": str(output),
        }), flush=True)
        if not graph_validation["ok"]:
            raise RuntimeError(graph_validation["reason"])
    finally:
        if handle is not None:
            library.tei_destroy(handle)


if __name__ == "__main__":
    main()
