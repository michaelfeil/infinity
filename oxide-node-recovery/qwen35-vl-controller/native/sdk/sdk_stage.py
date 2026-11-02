#!/usr/bin/env python3
"""CPU-only SDK/Torch/GOT/model/fixture stage validation. No engine, no GPU."""
import argparse, ctypes as C, hashlib, json, os, struct, subprocess, sys
from pathlib import Path

B66_SHA = "b66d70b61b39e1bf0d565aca33bac7c5e21849677396ce5216f5baf73027d37e"
TORCH_VERSION = "2.14.1+cu130"
TORCH_GIT = "5c4886908584029761b579af026dcfb627c84070"
GOT_VERSION = 130200
UNION124_SHA = "56863d52"

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def header(path):
    with Path(path).open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--library", required=True)
    p.add_argument("--torch-python", required=True)
    p.add_argument("--vendor", required=True)
    p.add_argument("--union124", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--media", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    failures = []
    b66 = sha(a.library)
    if b66 != B66_SHA: failures.append("b66 SHA changed")
    torch = subprocess.run([a.torch_python, "-c",
        "import torch;print(torch.__version__);print(torch.version.git_version)"],
        capture_output=True, text=True,
        env={k: v for k, v in os.environ.items() if not k.startswith(("CUDA_", "INFINITY_", "LD_"))}
        | {"CUDA_VISIBLE_DEVICES": ""})
    torch_lines = torch.stdout.strip().splitlines()
    if torch.returncode != 0 or torch_lines[:2] != [TORCH_VERSION, TORCH_GIT]:
        failures.append(f"torch identity changed: {torch.stdout!r} {torch.stderr!r}")
    env = dict(os.environ, LD_LIBRARY_PATH=a.vendor + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
    ldd = subprocess.run(["ldd", "-r", a.library], capture_output=True, text=True, env=env)
    cublas_lines = [line for line in ldd.stdout.splitlines() if "libcublas" in line]
    if ldd.returncode != 0 or "undefined symbol" in ldd.stdout:
        failures.append("ldd -r b66 failed")
    if not cublas_lines or any(a.vendor not in line for line in cublas_lines):
        failures.append(f"b66 cublas relocations are not vendor-bound: {cublas_lines}")
    lt = C.CDLL(str(Path(a.vendor) / "libcublasLt.so.13"))
    lt.cublasLtGetVersion.restype = C.c_size_t
    version = int(lt.cublasLtGetVersion())
    if version != GOT_VERSION: failures.append(f"cublasLt version {version} != {GOT_VERSION}")
    union = json.loads(Path(a.union124, "manifest.json").read_text())
    union_manifest_sha = sha(Path(a.union124, "manifest.json"))
    if not union_manifest_sha.startswith(UNION124_SHA):
        failures.append(f"union124 manifest {union_manifest_sha} lacks {UNION124_SHA}")
    archs = {entry["architecture"]: entry["symbols_verified"] for entry in union["architectures"]}
    if set(archs) != {"sm_90", "sm_100", "sm_120"} or any(len(v) != 124 for v in archs.values()):
        failures.append(f"union124 architecture/symbol proof incomplete: {list(archs)}")
    model = Path(a.model)
    config = json.loads((model / "config.json").read_text())
    index_path = model / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        shards = sorted(set(index["weight_map"].values()))
        headers = {}
        for shard in shards:
            headers.update({k: v for k, v in header(model / shard).items() if k != "__metadata__"})
        tensor_count = len(index["weight_map"])
    else:
        headers = {k: v for k, v in header(model / "model.safetensors").items() if k != "__metadata__"}
        tensor_count = len(headers)
    vision = sum(1 for name in headers if name.startswith("model.visual."))
    media = json.loads(Path(a.media).read_text())
    fixture_reports = []
    for fixture in media.get("fixtures", []):
        pixels = Path(fixture["pixels"])
        rows, patch_dim = int(fixture["rows"]), int(fixture["patch_dim"])
        size = pixels.stat().st_size
        expected = rows * patch_dim * 4
        grid = fixture["grid"]; merge = fixture["merge"]
        tokens = grid[0] * grid[1] * grid[2] // (merge * merge)
        ok = size == expected and tokens == fixture["token_count"]
        if not ok: failures.append(f"fixture {fixture['name']} shape/token guard failed")
        fixture_reports.append({"name": fixture["name"], "bytes": size, "rows": rows,
                                "patch_dim": patch_dim, "grid": grid, "merge": merge,
                                "tokens": tokens, "ok": ok})
    report = {"schema": "oxide-qwen35vl-sdk-cpu-stage/v1",
              "status": "passed" if not failures else "failed",
              "b66": {"sha256": b66, "path": a.library}, "torch": torch_lines,
              "cublasLt": version, "vendor": a.vendor,
              "union124": {"manifest_sha256": union_manifest_sha, "architectures": {k: len(v) for k, v in archs.items()}},
              "model": str(model), "model_type": config.get("model_type"),
              "text_model_type": config.get("text_config", {}).get("model_type"),
              "tensor_count": tensor_count, "vision_tensor_count": vision,
              "config_sha256": sha(model / "config.json"),
              "index_sha256": sha(index_path) if index_path.is_file() else None,
              "fixtures": fixture_reports, "gpu_initialized": False,
              "engine_created": False, "failures": failures}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "sdk-stage.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "failures": failures}))
    return 0 if not failures else 1

if __name__ == "__main__":
    raise SystemExit(main())
