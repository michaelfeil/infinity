#!/usr/bin/env python3
"""Root Native SDK CPU stage for the actual 531-tensor Mistral3 checkpoint.

This controller does not import Torch, does not load the SDK itself, and does
not create a CUDA context. It runs the prebuilt b66 C++ wrapper on the CPU,
which loads all 4 shards / 531 tensors and executes the reference forwards
before the GPU stage. The wrapper links the existing frozen SDK; the Native
implementation is not modified or recompiled by this packet.

Offline preflight:

    CUDA_VISIBLE_DEVICES='' python3 root_native_cpu_stage.py --preflight-only \
      --checkpoint /workspace/.../nvidia--Nemotron-3.5-Rerank-8B-BF16

Root CPU stage:

    CUDA_VISIBLE_DEVICES='' python3 root_native_cpu_stage.py \
      --checkpoint /workspace/.../nvidia--Nemotron-3.5-Rerank-8B-BF16 \
      --runner /path/mistral3_b66_runner \
      --out /tmp/mistral3-native-cpu-stage-fresh \
      --seconds 1200
"""
from __future__ import annotations
import argparse, hashlib, json, os, struct, subprocess, sys
from pathlib import Path

EXPECTED_CONFIG = "e1f442a21c464cf54259e7ca2ab09f6777f50a61fbf017efed8e6044894b4303"
EXPECTED_INDEX = "e393acf4755990784568ec2bc8de5d67f94f36b86d056c370b090084e80ea51e"
EXPECTED_TENSORS = 531
EXPECTED_CATEGORIES = {"vision": 218, "projector": 4, "language": 308, "score": 1}
DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U64": 8, "U32": 4, "U16": 2, "U8": 1, "BOOL": 1}

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def header(path: Path) -> dict:
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"short safetensors prefix: {path}")
        (n,) = struct.unpack("<Q", raw)
        if not 0 < n < 100_000_000:
            raise ValueError(f"bad safetensors header length: {path}")
        return json.loads(f.read(n))

def inventory(checkpoint: Path) -> dict:
    index = json.loads((checkpoint / "model.safetensors.index.json").read_bytes())
    names = {}
    for shard in sorted(set(index["weight_map"].values())):
        if Path(shard).name != shard or not shard.endswith(".safetensors"):
            raise ValueError(f"unsafe shard {shard}")
        for name, info in header(checkpoint / shard).items():
            if name == "__metadata__":
                continue
            if name in names:
                raise ValueError(f"duplicate tensor {name}")
            expected = DTYPE_BYTES[info["dtype"]]
            for dim in info["shape"]:
                expected *= int(dim)
            if info["data_offsets"][1] - info["data_offsets"][0] != expected:
                raise ValueError(f"bad offsets for {name}")
            names[name] = shard
    cats = {
        "vision": sum(n.startswith("model.vision_tower.") for n in names),
        "projector": sum(n.startswith("model.multi_modal_projector.") for n in names),
        "language": sum(n.startswith("model.language_model.") for n in names),
        "score": sum(n == "language_model.score.weight" for n in names),
    }
    return {"tensor_count": len(names), "shards": len(set(index["weight_map"].values())), "categories": cats, "index": index}

def preflight(checkpoint: Path) -> dict:
    errors = []
    config = (checkpoint / "config.json").read_bytes()
    if hashlib.sha256(config).hexdigest() != EXPECTED_CONFIG:
        errors.append("config sha256 mismatch")
    if sha(checkpoint / "model.safetensors.index.json") != EXPECTED_INDEX:
        errors.append("index sha256 mismatch")
    inv = inventory(checkpoint)
    if inv["tensor_count"] != EXPECTED_TENSORS:
        errors.append(f"tensor count {inv['tensor_count']} != {EXPECTED_TENSORS}")
    if inv["categories"] != EXPECTED_CATEGORIES:
        errors.append(f"category mismatch {inv['categories']}")
    return {"ok": not errors, "errors": errors, "checkpoint": str(checkpoint), **{k: v for k, v in inv.items() if k != "index"}}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--runner")
    ap.add_argument("--sdk")
    ap.add_argument("--sdk-sha256", default="b66d70b61b39e1bf0d565aca33bac7c5e21849677396ce5216f5baf73027d37e")
    ap.add_argument("--out")
    ap.add_argument("--seconds", type=int, default=1200)
    ap.add_argument("--preflight-only", action="store_true")
    a = ap.parse_args()
    checkpoint = Path(a.checkpoint)
    checks = preflight(checkpoint)
    if a.preflight_only:
        print(json.dumps(checks, indent=2))
        return 0 if checks["ok"] else 2
    if not checks["ok"]:
        print(json.dumps(checks, indent=2), file=sys.stderr)
        return 2
    if not a.runner or not a.out:
        print("--runner and --out are required for the CPU stage", file=sys.stderr)
        return 2
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        print("refusing: the Native CPU stage requires CUDA_VISIBLE_DEVICES=''", file=sys.stderr)
        return 2
    if a.sdk:
        sdk_sha = sha(Path(a.sdk))
        if sdk_sha != a.sdk_sha256:
            print(
                f"refusing: SDK sha256 {sdk_sha} != frozen {a.sdk_sha256}",
                file=sys.stderr,
            )
            return 2
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        print(f"refusing non-fresh output {out}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    guard = Path(__file__).resolve().parent / "remote_guard.py"
    vectors = out / "native-cpu-vectors"
    log = out / "native-cpu.log"
    command = [
        "/usr/bin/python3", str(guard),
        "--state", str(out / "cpu-guard.json"),
        "--seconds", str(a.seconds),
        "--", "taskset", "-c", "136-143",
        a.runner, "--checkpoint", str(checkpoint), "--device", "cpu",
        "--cpu-stage", "--iterations", "1", "--warmup", "0",
        "--emit-vectors", str(vectors),
    ]
    with log.open("w") as handle:
        proc = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    if proc.returncode != 0:
        print(f"Native CPU stage failed ({proc.returncode}); see {log}", file=sys.stderr)
        return proc.returncode
    text = log.read_text()
    start = text.find("{")
    if start < 0:
        print(f"Native CPU stage produced no JSON; see {log}", file=sys.stderr)
        return 1
    native = json.loads(text[start:])
    errors = []
    if native.get("tensors") != EXPECTED_TENSORS:
        errors.append(f"runner tensors {native.get('tensors')} != {EXPECTED_TENSORS}")
    if native.get("shards") != 4:
        errors.append(f"runner shards {native.get('shards')} != 4")
    if native.get("cpu_stage") is not True:
        errors.append("runner did not report cpu_stage=true")
    for name in [
        "text-mean-16", "text-mean-128", "text-score-32",
        "image-mean-1x4x6", "image-score-1x8x8", "image-mean-1x4x6-deferred",
        "text-yarn-16384", "text-yarn-32768", "text-yarn-262140",
        "image-yarn-16384",
    ]:
        if not (vectors / f"{name}.f32").is_file():
            errors.append(f"missing vector {name}.f32")
    report = {
        "schema": "oxide-mistral3-native-cpu-stage/v1",
        "checkpoint": str(checkpoint),
        "preflight": checks,
        "native": native,
        "vectors": {p.name: sha(p) for p in sorted(vectors.glob("*.f32"))},
        "errors": errors,
        "ok": not errors,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_imported": False,
        "cuda_context_created": False,
        "network_used": False,
    }
    (out / "native-cpu-stage.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1

if __name__ == "__main__":
    sys.exit(main())
