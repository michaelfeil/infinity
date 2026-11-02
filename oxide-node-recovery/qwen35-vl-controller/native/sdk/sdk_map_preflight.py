#!/usr/bin/env python3
"""CPU-only inside-SDK map/policy preflight. No engine, no GPU.

Order matters: typed container model path and expected config/index hashes are
checked first, before any Torch identity call or model load. The host path is
recorded only as provenance and is never used as a container filesystem path.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
import struct
import subprocess
from pathlib import Path

B66_SHA = "b66d70b61b39e1bf0d565aca33bac7c5e21849677396ce5216f5baf73027d37e"
TORCH_VERSION = "2.14.1+cu130"
TORCH_GIT = "5c4886908584029761b579af026dcfb627c84070"
GOT_VERSION = 130200
UNION124_SHA = "56863d52"
CONTAINER_MODEL_PREFIX = "/bench/michaelfeil"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def header(path):
    with Path(path).open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(length))


def validate_model(args):
    """Validate the typed container path and actual config/index hashes."""
    failures = []
    container = Path(args.model_container)
    if not str(container).startswith(CONTAINER_MODEL_PREFIX + "/"):
        failures.append(
            f"model-container must be a {CONTAINER_MODEL_PREFIX}/... path, got {container}")
    if str(container) == args.host_model_path:
        failures.append("host model path was passed as a container path")
    config_path = container / "config.json"
    index_path = container / "model.safetensors.index.json"
    config_sha = None
    index_sha = None
    tensor_count = None
    vision = None
    config = None
    if not config_path.is_file():
        failures.append(f"container model config missing: {config_path}")
    else:
        config_sha = sha(config_path)
        config = json.loads(config_path.read_text())
        if config_sha != args.expected_config_sha256:
            failures.append(f"container config SHA {config_sha} != expected")
    if args.model_format == "index":
        if not index_path.is_file():
            failures.append(f"container model index missing: {index_path}")
        else:
            index_sha = sha(index_path)
            if index_sha != args.expected_index_sha256:
                failures.append(f"container index SHA {index_sha} != expected")
            index = json.loads(index_path.read_text())
            shards = sorted(set(index["weight_map"].values()))
            headers = {}
            for shard in shards:
                shard_path = container / shard
                if not shard_path.is_file():
                    failures.append(f"container shard missing: {shard}")
                    continue
                headers.update({k: v for k, v in header(shard_path).items()
                                if k != "__metadata__"})
            tensor_count = len(index["weight_map"])
            vision = sum(1 for name in headers if name.startswith("model.visual."))
    else:
        weights = container / "model.safetensors"
        if not weights.is_file():
            failures.append(f"container single weights missing: {weights}")
    model = {"name": args.model_name,
             "host_path": args.host_model_path,
             "container_path": str(container),
             "container_prefix": CONTAINER_MODEL_PREFIX,
             "format": args.model_format,
             "config_sha256": config_sha,
             "index_sha256": index_sha,
             "tensor_count": tensor_count,
             "vision_tensor_count": vision,
             "model_type": config.get("model_type") if config else None,
             "text_model_type": (config.get("text_config", {}).get("model_type")
                                 if config else None)}
    return {"model": model, "config": config, "failures": failures}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--torch-python", required=True)
    parser.add_argument("--vendor", required=True)
    parser.add_argument("--union124", required=True)
    parser.add_argument("--model-container", required=True)
    parser.add_argument("--host-model-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-format", choices=("index", "single"), required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-index-sha256")
    parser.add_argument("--media", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    # 1. Typed container path and model identity hashes before Torch.
    model_result = validate_model(args)
    model = model_result["model"]
    config = model_result["config"]
    failures = list(model_result["failures"])
    if failures:
        report = {"schema": "oxide-qwen35vl-sdk-map-preflight/v1",
                  "status": "failed", "stage": "model-map-validation",
                  "model": model, "failures": failures,
                  "gpu_initialized": False, "engine_created": False}
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        (out / "sdk-map-preflight.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"status": "failed", "failures": failures}))
        raise SystemExit(1)

    # 2. Pinned library/Torch/vendor identities.
    b66 = sha(args.library)
    if b66 != B66_SHA:
        failures.append("b66 SHA changed")
    torch = subprocess.run(
        [args.torch_python, "-c",
         "import torch;print(torch.__version__);print(torch.version.git_version)"],
        capture_output=True, text=True,
        env={k: v for k, v in os.environ.items()
             if not k.startswith(("CUDA_", "INFINITY_", "LD_"))}
        | {"CUDA_VISIBLE_DEVICES": ""})
    torch_lines = torch.stdout.strip().splitlines()
    if torch.returncode != 0 or torch_lines[:2] != [TORCH_VERSION, TORCH_GIT]:
        failures.append(f"torch identity changed: {torch.stdout!r} {torch.stderr!r}")
    env = dict(os.environ,
               LD_LIBRARY_PATH=args.vendor + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
    ldd = subprocess.run(["ldd", "-r", args.library], capture_output=True, text=True, env=env)
    cublas_lines = [line for line in ldd.stdout.splitlines() if "libcublas" in line]
    if ldd.returncode != 0 or "undefined symbol" in ldd.stdout:
        failures.append("ldd -r b66 failed")
    if not cublas_lines or any(args.vendor not in line for line in cublas_lines):
        failures.append(f"b66 cublas relocations are not vendor-bound: {cublas_lines}")
    lt = C.CDLL(str(Path(args.vendor) / "libcublasLt.so.13"))
    lt.cublasLtGetVersion.restype = C.c_size_t
    version = int(lt.cublasLtGetVersion())
    if version != GOT_VERSION:
        failures.append(f"cublasLt version {version} != {GOT_VERSION}")
    union = json.loads(Path(args.union124, "manifest.json").read_text())
    union_manifest_sha = sha(Path(args.union124, "manifest.json"))
    if not union_manifest_sha.startswith(UNION124_SHA):
        failures.append(f"union124 manifest {union_manifest_sha} lacks {UNION124_SHA}")
    archs = {entry["architecture"]: entry["symbols_verified"]
             for entry in union["architectures"]}
    if set(archs) != {"sm_90", "sm_100", "sm_120"} or any(len(v) != 124 for v in archs.values()):
        failures.append(f"union124 architecture/symbol proof incomplete: {list(archs)}")

    # 3. Explicit remote fixture paths and shape/token guards.
    media = json.loads(Path(args.media).read_text())
    fixture_reports = []
    for fixture in media.get("fixtures", []):
        pixels = Path(fixture["pixels"])
        rows, patch_dim = int(fixture["rows"]), int(fixture["patch_dim"])
        if not pixels.is_file():
            failures.append(f"fixture missing: {pixels}")
            fixture_reports.append({"name": fixture["name"], "path": str(pixels),
                                    "ok": False})
            continue
        size = pixels.stat().st_size
        expected = rows * patch_dim * 4
        grid = fixture["grid"]; merge = fixture["merge"]
        tokens = grid[0] * grid[1] * grid[2] // (merge * merge)
        ok = size == expected and tokens == fixture["token_count"]
        if not ok:
            failures.append(f"fixture {fixture['name']} shape/token guard failed")
        fixture_reports.append({"name": fixture["name"], "path": str(pixels),
                                "bytes": size, "rows": rows, "patch_dim": patch_dim,
                                "grid": grid, "merge": merge, "tokens": tokens,
                                "ok": ok})

    report = {
        "schema": "oxide-qwen35vl-sdk-map-preflight/v1",
        "status": "passed" if not failures else "failed",
        "stage": "inside-sdk-preflight",
        "model": model,
        "b66": {"sha256": b66, "path": args.library},
        "torch": torch_lines,
        "cublasLt": version,
        "vendor": args.vendor,
        "union124": {"manifest_sha256": union_manifest_sha,
                     "architectures": {k: len(v) for k, v in archs.items()}},
        "fixtures": fixture_reports,
        "model_checked_before_torch": True,
        "gpu_initialized": False,
        "engine_created": False,
        "failures": failures,
    }
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "sdk-map-preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "failures": failures}))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
