#!/opt/torch-sdk/bin/python3.12
"""Root-container light CPU stage for Qwen3.5 MoE (v4).

Actual LIGHT path preserved from v3 with one documented comparator fact:
config_identity carries text_model_type derived from text_config.model_type.
No engine, weights, tei_ready or tei_forward. Torch is imported only inside the
delegated container SDK/GOT proof. Header/index metadata only; no payload read.
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import io
import json
import math
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOCAL_INPUT = ROOT / "input"
SOURCE_INPUT = LOCAL_INPUT if LOCAL_INPUT.is_dir() else (
    ROOT.parent / "input" if (ROOT.parent / "input").is_dir()
    else Path("/tmp/oxide-q35moe-native-runtime-control-v1/input"))
if SOURCE_INPUT.is_dir():
    sys.path.insert(0, str(SOURCE_INPUT))

import model_checkpoint as mc  # noqa: E402
import native_batch as nb  # noqa: E402

DEFAULT_MODEL = Path("/bench/michaelfeil/onejev-perf/models/qwen35-35b-a3b")
DEFAULT_LIBRARY = Path("/tmp/native214-fa4-mode20/install/lib/libtei_torch.so")
DEFAULT_VENDOR = Path("/tmp/oxide-host-cublas-130200")
EXPECTED_SDK_SHA256 = "b66d70b61b39e1bf0d565aca33bac7c5e21849677396ce5216f5baf73027d37e"
EXPECTED_TORCH_PREFIX = "2.14.1+cu130"
EXPECTED_TORCH_GIT_PREFIX = "5c488"
EXPECTED_CUBLASLT_VERSION = 130200
EXPECTED_TEI_FORWARD_ARGS = 19
EXPECTED_TENSORS = 1811
EXPECTED_CONFIG_SHA256 = "5e4d7f74fec2f360eb9cfbfcd6ec0c4c76e684d3a11caaed259d9fd9bfbc7944"
EXPECTED_DTYPE_COUNTS = {"BF16": 1751, "F32": 60}
EXPECTED_SHARDS = 14
EXPECTED_MODEL_BYTES = 71903878016
Q35 = {"hidden_size": 2048, "num_hidden_layers": 40, "num_attention_heads": 16,
       "num_key_value_heads": 2, "head_dim": 256, "num_experts": 256,
       "num_experts_per_tok": 8, "moe_intermediate_size": 512}
ALLOWED_MODEL_TYPES = {"qwen3_5_moe", "qwen3_5_moe_text"}
ALLOWED_LAYER_TYPES = {"full_attention", "linear_attention"}
ALLOWED_HEADER_DTYPES = {"BF16", "F32"}
SCHEMA = "oxide-q35moe-native-light/v1"
PIN_SOURCE_FILES = ("native_light.py", "native_gold.py", "strict_compare.py",
                    "root_control.py", "remote_guard.py", "container_sdk_got.py",
                    "bridge.h", "native_base.py", "native_raw_probe.py",
                    "native_batch.py", "model_checkpoint.py", "verify_bindings.py",
                    "bridge.cpp", "qwen35_models.cpp")


class LightError(ValueError):
    pass


class TeiConfig(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int64) for name in
                ["hidden", "heads", "layers", "intermediate", "vocab", "positions", "types"]]
    _fields_ += [("epsilon", ctypes.c_double), ("activation", ctypes.c_int32)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def text_config(raw: dict) -> dict:
    text = raw.get("text_config", raw)
    if not isinstance(text, dict):
        raise LightError("config.text_config must be an object")
    return text


def _int(text: dict, key: str) -> int:
    value = text.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LightError(f"config field {key} must be a positive integer, got {value!r}")
    return value


def _optional_int(text: dict, key: str, default: int) -> int:
    value = text.get(key, default)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LightError(f"config field {key} must be a nonnegative integer, got {value!r}")
    return value


def load_raw_config(model: Path) -> tuple[dict, str]:
    path = Path(model) / "config.json"
    if not path.is_file():
        raise LightError(f"missing original config.json: {path}")
    raw = mc.read_json(path)
    if not isinstance(raw, dict):
        raise LightError("config.json must be a JSON object")
    return raw, sha256_file(path)


def validate_q35_config(raw: dict) -> dict:
    text = text_config(raw)
    model_type = raw.get("model_type") or text.get("model_type")
    if model_type not in ALLOWED_MODEL_TYPES:
        raise LightError(f"unsupported Qwen3.5 MoE model_type: {model_type!r}")
    architecture = raw.get("architectures", text.get("architectures"))
    if (not isinstance(architecture, list) or not architecture
            or not all(isinstance(item, str) and item for item in architecture)):
        raise LightError("config architectures must be a nonempty string list")
    if "qwen3" not in " ".join(architecture).lower():
        raise LightError(f"architecture list is not Qwen3.5: {architecture!r}")
    layers = _int(text, "num_hidden_layers")
    checks = {"num_hidden_layers": layers, "hidden_size": _int(text, "hidden_size"),
              "num_attention_heads": _int(text, "num_attention_heads"),
              "num_key_value_heads": _int(text, "num_key_value_heads"),
              "head_dim": _int(text, "head_dim"), "num_experts": _int(text, "num_experts"),
              "num_experts_per_tok": _int(text, "num_experts_per_tok"),
              "moe_intermediate_size": _int(text, "moe_intermediate_size")}
    for key, value in Q35.items():
        if checks[key] != value:
            raise LightError(f"{key} is not the exact Q35MoE value {value}")
    layer_types = text.get("layer_types")
    if not isinstance(layer_types, list) or len(layer_types) != layers:
        raise LightError("layer_types must describe all 40 Qwen3.5 layers")
    if not all(item in ALLOWED_LAYER_TYPES for item in layer_types):
        raise LightError(f"unsupported layer order: {layer_types!r}")
    dtype = text.get("dtype")
    if dtype is None:
        dtype = text.get("torch_dtype")
    if dtype not in ("bfloat16", "bf16"):
        raise LightError(f"Q35MoE source dtype must be BF16, got {dtype!r}")
    epsilon = text.get("rms_norm_eps", text.get("layer_norm_eps", 1e-6))
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon):
        raise LightError(f"invalid norm epsilon: {epsilon!r}")
    activation = text.get("hidden_act", text.get("hidden_activation", "silu"))
    if activation != "silu":
        raise LightError(f"Q35MoE requires silu, got {activation!r}")
    return {"raw_text": text, "model_type": model_type,
            "text_model_type": text.get("model_type"),
            "architecture_class": list(architecture), "hidden_size": Q35["hidden_size"],
            "layer_count": layers, "layer_types": list(layer_types),
            "num_attention_heads": Q35["num_attention_heads"],
            "num_key_value_heads": Q35["num_key_value_heads"], "head_dim": Q35["head_dim"],
            "num_experts": Q35["num_experts"], "num_experts_per_tok": Q35["num_experts_per_tok"],
            "moe_intermediate_size": Q35["moe_intermediate_size"],
            "dense_intermediate_size": _optional_int(text, "intermediate_size", 0),
            "vocab_size": _int(text, "vocab_size"),
            "max_position_embeddings": _int(text, "max_position_embeddings"),
            "norm_epsilon": float(epsilon), "hidden_activation": activation,
            "source_dtype": "bfloat16"}


def header_identity(checkpoint, expected_tensors: int = EXPECTED_TENSORS) -> dict:
    infos = checkpoint.tensors
    if len(infos) != expected_tensors:
        raise LightError(f"expected {expected_tensors} headers, got {len(infos)}")
    dtypes, layers, prefixes = {}, set(), {}
    for name, info in infos.items():
        if info.dtype not in ALLOWED_HEADER_DTYPES:
            raise LightError(f"unexpected header dtype {info.dtype} for {name}")
        dtypes[info.dtype] = dtypes.get(info.dtype, 0) + 1
        for match in re.finditer(r"(?:^|\.)layers\.(\d+)\.", name):
            layers.add(int(match.group(1)))
        for candidate in ("model.language_model.", "language_model.", "model.", ""):
            if name == candidate + "embed_tokens.weight":
                prefixes[candidate] = prefixes.get(candidate, 0) + 1
    if layers != set(range(Q35["num_hidden_layers"])):
        raise LightError(f"header layer order is incomplete: {sorted(layers)}")
    if len(prefixes) != 1:
        raise LightError(f"expected one embedding prefix, got {sorted(prefixes)}")
    prefix = next(iter(prefixes))
    for layer in range(Q35["num_hidden_layers"]):
        for suffix in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            if f"{prefix}layers.{layer}.{suffix}" not in infos:
                raise LightError(f"missing header {prefix}layers.{layer}.{suffix}")
    records = [[name, infos[name].dtype, list(infos[name].shape),
                infos[name].start, infos[name].end] for name in sorted(infos)]
    files = [{"name": file["path"].name, "bytes": file["size"],
              "inode": file["inode"], "mtime_ns": file["mtime_ns"],
              "header_sha256": file["header_sha256"]} for file in checkpoint.files]
    total_bytes = sum(file["bytes"] for file in files)
    if dict(sorted(dtypes.items())) != EXPECTED_DTYPE_COUNTS:
        raise LightError(f"header dtype counts {dtypes} != {EXPECTED_DTYPE_COUNTS}")
    if len(files) != EXPECTED_SHARDS:
        raise LightError(f"checkpoint has {len(files)} shards, expected {EXPECTED_SHARDS}")
    if total_bytes != EXPECTED_MODEL_BYTES:
        raise LightError(f"checkpoint bytes {total_bytes} != {EXPECTED_MODEL_BYTES}")
    return {"tensor_count": len(infos), "dtype_counts": dict(sorted(dtypes.items())),
            "weight_prefix": prefix, "layer_indices": sorted(layers),
            "header_digest_sha256": sha256_text(
                json.dumps(records, separators=(",", ":"), sort_keys=True)),
            "files": files, "total_bytes": total_bytes, "payload_read": False}


def identity_snapshot(model: Path) -> dict:
    raw, config_sha = load_raw_config(model)
    if config_sha != EXPECTED_CONFIG_SHA256:
        raise LightError(f"config.json sha256 {config_sha} != {EXPECTED_CONFIG_SHA256}")
    config = validate_q35_config(raw)
    checkpoint = mc.Checkpoint(model)
    header = header_identity(checkpoint)
    identity = {key: config[key] for key in (
        "architecture_class", "model_type", "text_model_type", "hidden_size",
        "layer_count", "layer_types", "num_attention_heads", "num_key_value_heads",
        "head_dim", "num_experts", "num_experts_per_tok", "moe_intermediate_size",
        "dense_intermediate_size", "vocab_size", "max_position_embeddings",
        "source_dtype")}
    identity["config_sha256"] = config_sha
    identity["layer_order_sha256"] = sha256_text(
        json.dumps(config["layer_types"], separators=(",", ":")))
    combined = sha256_text(json.dumps({"config": identity, "header": header},
                                      sort_keys=True, separators=(",", ":")))
    return {"config_identity": identity, "header_identity": header,
            "identity_sha256": combined, "raw_config": raw, "config": config,
            "checkpoint": checkpoint}


def check_bridge_header(path: Path, expected_args: int = EXPECTED_TEI_FORWARD_ARGS) -> dict:
    if not path.is_file():
        raise LightError(f"missing bridge.h: {path}")
    text = path.read_text()
    fields = re.search(
        r"struct\s+TeiConfig\s*\{\s*"
        r"int64_t\s+hidden\s*,\s*heads\s*,\s*layers\s*,\s*intermediate\s*,\s*vocab\s*,\s*positions\s*,\s*types\s*;\s*"
        r"double\s+epsilon\s*;\s*int32_t\s+activation\s*;\s*\}\s*;", text)
    if not fields:
        raise LightError("bridge.h TeiConfig field order/layout changed")
    abi = nb.check_forward_abi(text)
    if abi["arg_count"] != expected_args:
        raise LightError(f"tei_forward args {abi['arg_count']} != {expected_args}")
    return {"path": str(path), "sha256": sha256_file(path),
            "tei_forward_arg_count": abi["arg_count"], "tei_forward_abi": abi,
            "tei_config_size_bytes": ctypes.sizeof(TeiConfig),
            "tei_config_declaration": "int64 x7; double epsilon; int32 activation"}


def run_sdk_proof(args) -> dict:
    import container_sdk_got as csg

    out = Path(args.output).with_name(Path(args.output).stem + ".sdk-proof.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    argv = ["container_sdk_got.py", "--sdk", str(Path(args.sdk)),
            "--expected-sdk-sha", EXPECTED_SDK_SHA256,
            "--vendor", str(Path(args.vendor)),
            "--expected-cublaslt-version", str(EXPECTED_CUBLASLT_VERSION),
            "--bridge-header", str(Path(args.bridge_header)),
            "--expected-tei-forward-args", str(EXPECTED_TEI_FORWARD_ARGS),
            "--expected-torch-prefix", EXPECTED_TORCH_PREFIX,
            "--expected-torch-git-prefix", EXPECTED_TORCH_GIT_PREFIX,
            "--allow-cuda-visible-devices", "", "--out", str(out)]
    old, captured = sys.argv, io.StringIO()
    sys.argv = argv
    try:
        with contextlib.redirect_stdout(captured):
            code = csg.main()
    finally:
        sys.argv = old
    if not out.is_file():
        raise LightError("container SDK proof did not write its report")
    report = json.loads(out.read_text())
    if code != 0 or not report.get("ok"):
        raise LightError(f"container SDK/GOT proof failed: {report.get('errors')}")
    checks = {
        "sdk_sha256": report.get("sdk_sha256") == EXPECTED_SDK_SHA256,
        "sdk_loaded": report.get("sdk_loaded") is True,
        "tei_forward_symbol": report.get("tei_forward_symbol") is True,
        "tei_forward_args": report.get("tei_forward_args") == EXPECTED_TEI_FORWARD_ARGS,
        "got_targets": report.get("cublas_got_target_count", 0) > 0,
        "got_vendor_130200": bool(report.get("cublas_got_targets")) and all(
            item.get("vendor_130200") for item in report["cublas_got_targets"]),
        "cublaslt_version": report.get("vendor", {}).get("cublaslt_version")
        == EXPECTED_CUBLASLT_VERSION,
        "torch_prefix": str(report.get("torch_version", "")).startswith(EXPECTED_TORCH_PREFIX),
        "torch_git": str(report.get("torch_git_version", "")).startswith(EXPECTED_TORCH_GIT_PREFIX),
        "cuda_not_initialized": (report.get("torch_cuda_initialized_before") is False
                                 and report.get("torch_cuda_initialized_after") is False),
        "cuda_visible_devices": report.get("cuda_visible_devices") == "",
        "ld_bind_now": report.get("ld_bind_now") == "1",
        "no_engine": report.get("engine_created") is False
        and report.get("engine_or_forward_called") is False,
    }
    failed = sorted(name for name, ok in checks.items() if not ok)
    if failed:
        raise LightError(f"container SDK proof summary checks failed: {failed}")
    return {"report": report, "checks": checks, "path": str(out)}


def pins_snapshot(args) -> dict:
    identity = identity_snapshot(args.model)
    bridge = check_bridge_header(args.bridge_header)
    sources = {}
    for name in PIN_SOURCE_FILES:
        path = ROOT / name
        if not path.is_file():
            path = SOURCE_INPUT / name
        if not path.is_file():
            raise LightError(f"required source is missing: {name}")
        sources[name] = sha256_file(path)
    provider = {"vendor": str(Path(args.vendor)), "cublaslt_version": None,
                "ld_preload": os.environ.get("LD_PRELOAD"),
                "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "ld_bind_now": os.environ.get("LD_BIND_NOW")}
    try:
        vendor_lib = ctypes.CDLL(str(Path(args.vendor) / "libcublasLt.so.13"),
                                 mode=os.RTLD_LOCAL)
        vendor_lib.cublasLtGetVersion.restype = ctypes.c_size_t
        provider["cublaslt_version"] = int(vendor_lib.cublasLtGetVersion())
    except Exception as error:
        provider["cublaslt_version_error"] = str(error)
    payload = {"schema": "oxide-q35moe-native-pins/v4", "sources": sources,
               "binaries": {"python": sys.executable, "python_version": sys.version,
                            "library": str(Path(args.sdk).resolve()),
                            "library_sha256": sha256_file(args.sdk) if Path(args.sdk).is_file() else None,
                            "bridge": bridge},
               "providers": provider,
               "module": {"native_light_sha256": sources.get("native_light.py")},
               "model_config": identity["config_identity"],
               "model_header": identity["header_identity"],
               "identity_sha256": identity["identity_sha256"],
               "payload_hashed": False}
    payload["pins_digest"] = sha256_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", action="store_true")
    parser.add_argument("--pins-only", action="store_true")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--sdk", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--vendor", type=Path, default=DEFAULT_VENDOR)
    parser.add_argument("--bridge-header", type=Path, default=SOURCE_INPUT / "bridge.h")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.pins_only:
        report = pins_snapshot(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"status": "pins-ok", "pins_digest": report["pins_digest"],
                          "output": str(args.output)}))
        return 0
    if not args.container:
        raise SystemExit("native_light.py is Root-container-only; pass --container")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["LD_BIND_NOW"] = "1"
    errors = []
    before = proof = after = header = None
    try:
        header = check_bridge_header(args.bridge_header)
        before = identity_snapshot(args.model)
        proof = run_sdk_proof(args)
        after = identity_snapshot(args.model)
        if before["identity_sha256"] != after["identity_sha256"]:
            errors.append("model/config/header identity changed during the light proof")
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    report = {"schema": SCHEMA, "stage": "native-light",
              "status": "ok" if not errors else "failed", "ok": not errors,
              "container_only": True,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "ld_bind_now": os.environ.get("LD_BIND_NOW"),
              "engine_created": False, "engine_or_forward_called": False,
              "cuda_initialized": False, "payload_read": False,
              "torch_imported_by_delegated_proof": proof is not None,
              "bridge": header, "model": str(Path(args.model).resolve()),
              "library": str(Path(args.sdk).resolve()),
              "config_identity": before["config_identity"] if before else None,
              "header_identity": before["header_identity"] if before else None,
              "identity_sha256_before": before["identity_sha256"] if before else None,
              "identity_sha256_after": after["identity_sha256"] if after else None,
              "identity_stable": bool(before and after
                                      and before["identity_sha256"] == after["identity_sha256"]),
              "sdk_proof_path": proof["path"] if proof else None,
              "sdk_proof_checks": proof["checks"] if proof else None,
              "sdk_proof_ok": bool(proof), "errors": errors}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "sdk_proof_ok": report["sdk_proof_ok"],
                      "tensor_count": (report["header_identity"] or {}).get("tensor_count"),
                      "output": str(args.output)}))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
