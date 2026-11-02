#!/usr/bin/env python3
"""Root-only paired Mistral3 qualification controller.

Native b66 SDK gold and the fresh optimized Oxide candidate are run in separate
processes on the same original checkpoint, then paired by deterministic case.
This script never imports Torch, never creates a CUDA context itself, never
rewrites the checkpoint/config, and never flattens or drops YaRN /
llama_4_scaling fields. Root must provide the real GPU/Torch environment and the
compiled Native runner and Oxide candidate binary.

CPU-only preflight (safe offline, no GPU, no binaries executed):

    python3 root_paired_controller.py --preflight-only \
        --checkpoint /workspace/.../nvidia--Nemotron-3.5-Rerank-8B-BF16

Root GPU0 run (must be explicitly unlocked):

    CUDA_VISIBLE_DEVICES=0 INFINITY_OXIDE_ROOT_GPU0_LOCKED=1 \
      python3 root_paired_controller.py \
        --checkpoint /workspace/.../nvidia--Nemotron-3.5-Rerank-8B-BF16 \
        --native-runner /path/mistral3_b66_runner \
        --candidate-bin /path/mistral3_vectors \
        --fa4-root /path/fa4 \
        --out /tmp/mistral3-paired-$(date +%s) \
        --iterations 20 --warmup 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
from pathlib import Path

EXPECTED_CONFIG_SHA256 = "e1f442a21c464cf54259e7ca2ab09f6777f50a61fbf017efed8e6044894b4303"
EXPECTED_INDEX_SHA256 = "e393acf4755990784568ec2bc8de5d67f94f36b86d056c370b090084e80ea51e"
EXPECTED_TENSORS = 531
EXPECTED_CATEGORIES = {"vision": 218, "projector": 4, "language": 308, "score": 1}
SCORE_CASES = {"text-score-32", "image-score-1x8x8"}
MEAN_CASES = {
    "text-mean-16",
    "text-mean-128",
    "image-mean-1x4x6",
    "image-mean-1x4x6-deferred",
    "image-score-1x8x8",
    "text-yarn-16384",
    "text-yarn-32768",
    "text-yarn-262140",
    "image-yarn-16384",
}
DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U64": 8, "U32": 4, "U16": 2, "U8": 1, "BOOL": 1}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_header(path: Path) -> dict:
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError(f"short safetensors prefix: {path}")
        (length,) = struct.unpack("<Q", raw)
        if not 0 < length < 100_000_000:
            raise ValueError(f"invalid safetensors header length {length}: {path}")
        header = json.loads(handle.read(length))
    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype = info["dtype"]
        shape = [int(v) for v in info["shape"]]
        offsets = [int(v) for v in info["data_offsets"]]
        expected = DTYPE_BYTES[dtype]
        for dim in shape:
            expected *= dim
        if offsets[1] - offsets[0] != expected:
            raise ValueError(f"offset/size mismatch for {name}")
        tensors[name] = {"dtype": dtype, "shape": shape, "data_offsets": offsets}
    return tensors


def inventory(checkpoint: Path) -> tuple[dict, dict, dict]:
    config_bytes = (checkpoint / "config.json").read_bytes()
    config = json.loads(config_bytes)
    index_path = checkpoint / "model.safetensors.index.json"
    index = json.loads(index_path.read_bytes())
    weight_map = index["weight_map"]
    names = {}
    for shard in sorted(set(weight_map.values())):
        if Path(shard).name != shard or not shard.endswith(".safetensors"):
            raise ValueError(f"unsafe shard name {shard}")
        for name, info in read_header(checkpoint / shard).items():
            if name in names:
                raise ValueError(f"duplicate tensor {name}")
            names[name] = {**info, "shard": shard}
    if set(names) != set(weight_map):
        raise ValueError("index/header tensor names disagree")
    categories = [
        name for name in names if name.startswith("model.vision_tower.")
    ], [
        name for name in names if name.startswith("model.multi_modal_projector.")
    ], [
        name for name in names if name.startswith("model.language_model.")
    ], [name for name in names if name == "language_model.score.weight"]
    counts = {
        "vision": len(categories[0]),
        "projector": len(categories[1]),
        "language": len(categories[2]),
        "score": len(categories[3]),
    }
    return config, names, {"index": index, "counts": counts, "config_sha256": hashlib.sha256(config_bytes).hexdigest(), "index_sha256": sha256_file(index_path)}


def preflight(checkpoint: Path) -> dict:
    config, names, meta = inventory(checkpoint)
    errors = []
    if meta["config_sha256"] != EXPECTED_CONFIG_SHA256:
        errors.append("config sha256 differs from the frozen original")
    if meta["index_sha256"] != EXPECTED_INDEX_SHA256:
        errors.append("index sha256 differs from the frozen original")
    if len(names) != EXPECTED_TENSORS:
        errors.append(f"tensor count {len(names)} != {EXPECTED_TENSORS}")
    for key, expected in EXPECTED_CATEGORIES.items():
        if meta["counts"][key] != expected:
            errors.append(f"{key} count {meta['counts'][key]} != {expected}")
    text = config.get("text_config", {})
    rope = text.get("rope_parameters", {})
    if rope.get("rope_type") != "yarn":
        errors.append("text_config.rope_parameters.rope_type is not nested YaRN")
    for key in ("factor", "beta_fast", "beta_slow", "mscale", "mscale_all_dim", "llama_4_scaling_beta", "original_max_position_embeddings"):
        if key not in rope:
            errors.append(f"text_config.rope_parameters.{key} is missing (no flattening allowed)")
    llama4 = text.get("llama_4_scaling", {})
    if llama4.get("beta") != 0.1 or llama4.get("original_max_position_embeddings") != 16384:
        errors.append("text_config.llama_4_scaling was flattened or changed")
    if text.get("is_causal") is not False:
        errors.append("text_config.is_causal is not the bidirectional recipe")
    if config.get("multimodal_projector_bias") is not False:
        errors.append("multimodal_projector_bias is not the actual checkpoint value")
    return {
        "ok": not errors,
        "errors": errors,
        "checkpoint": str(checkpoint),
        "tensor_count": len(names),
        "shard_count": len(set(meta["index"]["weight_map"].values())),
        "categories": meta["counts"],
        "config_sha256": meta["config_sha256"],
        "index_sha256": meta["index_sha256"],
        "config_preserved_nested": {
            "rope_parameters": rope,
            "llama_4_scaling": llama4,
        },
        "payload_read": False,
        "torch_imported": False,
        "cuda_context_created": False,
    }


def read_f32(path: Path) -> list[float]:
    data = path.read_bytes()
    if len(data) % 4:
        raise ValueError(f"{path} is not a whole f32 vector")
    return [v[0] for v in struct.iter_unpack("<f", data)]


def compare(gold: list[float], candidate: list[float]) -> dict:
    if len(gold) != len(candidate):
        return {"ok": False, "reason": f"element count {len(candidate)} != {len(gold)}"}
    max_abs = 0.0
    dot = 0.0
    gold_norm = 0.0
    candidate_norm = 0.0
    diff_norm = 0.0
    exact = 0
    for a, b in zip(gold, candidate):
        delta = abs(a - b)
        max_abs = max(max_abs, delta)
        if delta == 0.0:
            exact += 1
        dot += a * b
        gold_norm += a * a
        candidate_norm += b * b
        diff_norm += delta * delta
    cosine = dot / math.sqrt(gold_norm * candidate_norm) if gold_norm and candidate_norm else 1.0
    relative_l2 = math.sqrt(diff_norm / gold_norm) if gold_norm else 0.0
    return {
        "ok": True,
        "elements": len(gold),
        "exact": exact,
        "max_abs": max_abs,
        "cosine": cosine,
        "relative_l2": relative_l2,
    }


def run(command: list[str], env: dict, log: Path) -> dict:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        process = subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if process.returncode != 0:
        raise RuntimeError(f"{command[0]} failed with {process.returncode}; see {log}")
    text = log.read_text()
    start = text.find("{")
    if start < 0:
        raise RuntimeError(f"{command[0]} produced no JSON; see {log}")
    return json.loads(text[start:])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--native-runner", default="")
    parser.add_argument("--candidate-bin", default="")
    parser.add_argument("--fa4-root", default="")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=5e-2)
    parser.add_argument("--min-cosine", type=float, default=0.9999)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    checks = preflight(checkpoint)
    if args.preflight_only:
        print(json.dumps(checks, indent=2))
        return 0 if checks["ok"] else 2
    if not checks["ok"]:
        print(json.dumps(checks, indent=2), file=sys.stderr)
        return 2
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        print("refusing: CUDA_VISIBLE_DEVICES must be set by Root", file=sys.stderr)
        return 2
    if os.environ.get("INFINITY_OXIDE_ROOT_GPU0_LOCKED") != "1":
        print("refusing: INFINITY_OXIDE_ROOT_GPU0_LOCKED=1 is required", file=sys.stderr)
        return 2
    for flag, value in (("--native-runner", args.native_runner), ("--candidate-bin", args.candidate_bin), ("--fa4-root", args.fa4_root), ("--out", args.out)):
        if not value:
            print(f"refusing: {flag} is required for a Root run", file=sys.stderr)
            return 2
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        print(f"refusing: output directory is not fresh: {out}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    config_before = sha256_file(checkpoint / "config.json")
    index_before = sha256_file(checkpoint / "model.safetensors.index.json")

    native_dir = out / "native"
    candidate_dir = out / "candidate"
    env = {**os.environ, "INFINITY_MISTRAL3_MODEL": str(checkpoint), "INFINITY_MISTRAL3_DEVICE": "0", "INFINITY_FA4_ROOT": args.fa4_root, "INFINITY_MISTRAL3_OUT_DIR": str(candidate_dir), "INFINITY_MISTRAL3_RUN_GPU": "1", "INFINITY_MISTRAL3_ITERATIONS": str(args.iterations), "INFINITY_MISTRAL3_WARMUP": str(args.warmup), "INFINITY_MISTRAL3_GOLD_DIR": str(native_dir)}

    native = run([args.native_runner, "--checkpoint", str(checkpoint), "--device", "cuda:0", "--iterations", str(args.iterations), "--warmup", str(args.warmup), "--emit-vectors", str(native_dir)], env, out / "native.log")
    candidate = run([args.candidate_bin], env, out / "candidate.log")
    # A second fresh candidate process proves create/load/run/destroy repeatability.
    candidate_repeat_dir = out / "candidate-repeat"
    repeat_env = {**env, "INFINITY_MISTRAL3_OUT_DIR": str(candidate_repeat_dir)}
    candidate_repeat = run([args.candidate_bin], repeat_env, out / "candidate-repeat.log")

    pairs = {}
    failures = []
    for case in sorted(MEAN_CASES):
        source = "logits" if case in SCORE_CASES else "pooled"
        # The deferred Native case consumes the same image-mean request; pair it
        # with the candidate's synchronous image-mean vector.
        candidate_case = case.removesuffix("-deferred")
        gold_path = native_dir / f"{case}.f32"
        candidate_path = candidate_dir / f"{candidate_case}.{source}.f32"
        if not gold_path.is_file() or not candidate_path.is_file():
            failures.append(f"{case}: missing paired vector")
            continue
        result = compare(read_f32(gold_path), read_f32(candidate_path))
        result["source"] = source
        result["pass"] = (
            result.get("ok", False)
            and result["max_abs"] <= args.tolerance
            and result["cosine"] >= args.min_cosine
        )
        pairs[case] = result
        if not result["pass"]:
            failures.append(f"{case}: max_abs/cosine outside tolerance")

    for case in sorted(MEAN_CASES - {"image-mean-1x4x6-deferred"}):
        first = candidate_dir / f"{case}.pooled.f32"
        repeat = candidate_repeat_dir / f"{case}.pooled.f32"
        if first.is_file() and repeat.is_file():
            result = compare(read_f32(first), read_f32(repeat))
            pairs[f"{case}-repeat"] = {**result, "pass": result.get("exact") == result.get("elements")}
            if result.get("exact") != result.get("elements"):
                failures.append(f"{case}: repeat is not deterministic")

    counters = native.get("perf_counters", {})
    lifecycle = {
        "native": {
            key: counters.get(key)
            for key in ("create_ns", "option_ns", "load_ns", "ready_ns", "forward_sync_ns", "forward_submit_ns", "output_wait_ns", "output_destroy_ns", "destroy_ns", "forward_calls", "deferred_calls")
        },
        "candidate": {
            "create_ns": candidate.get("create_ns"),
            "first_run_ns": candidate.get("first_run_ns"),
            "repeatable": not any("repeat" in failure for failure in failures),
        },
        "native_tensors": native.get("tensors"),
        "native_shards": native.get("shards"),
    }
    if lifecycle["native_tensors"] != EXPECTED_TENSORS or lifecycle["native_shards"] != 4:
        failures.append("Native runner did not load the original 531 tensors from 4 shards")
    if not lifecycle["candidate"]["create_ns"] or not lifecycle["candidate"]["first_run_ns"]:
        failures.append("candidate lifecycle timings are missing")

    native_sync = native.get("perf", {}).get("synchronous_image_mean", {})
    native_deferred = native.get("perf", {}).get("deferred_image_mean", {})
    candidate_perf = candidate.get("perf", {})
    def delta(a, b):
        if not a or not b:
            return None
        return (b - a) / a * 100.0
    perf = {
        "native_sync": native_sync,
        "native_deferred": native_deferred,
        "candidate": candidate_perf,
        "p50_delta_percent": delta(native_sync.get("p50_ns"), candidate_perf.get("p50_ns")),
        "p99_delta_percent": delta(native_sync.get("p99_ns"), candidate_perf.get("p99_ns")),
        "target_5_percent_faster_p50": bool(native_sync.get("p50_ns") and candidate_perf.get("p50_ns") and candidate_perf["p50_ns"] <= native_sync["p50_ns"] * 0.95),
        "no_cuda_graphs_oxide": True,
        "claim": "Profile-window timings only; not an all-shape or all-hardware performance claim.",
    }

    config_after = sha256_file(checkpoint / "config.json")
    index_after = sha256_file(checkpoint / "model.safetensors.index.json")
    preserved = config_after == config_before == EXPECTED_CONFIG_SHA256 and index_after == index_before == EXPECTED_INDEX_SHA256
    if not preserved:
        failures.append("checkpoint config/index changed during the run")
    report = {
        "schema": "oxide-mistral3-paired-root-controller/v1",
        "checkpoint": str(checkpoint),
        "preflight": checks,
        "pairs": pairs,
        "lifecycle": lifecycle,
        "perf": perf,
        "config_index_preserved": preserved,
        "no_flatten_yarn_llama4": True,
        "no_checkpoint_rewrites": preserved,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "failures": failures,
        "status": "pass" if not failures else "fail",
        "gold_source": "Native b66 SDK runner",
        "candidate_source": "fresh optimized Oxide Mistral3 engine binary",
    }
    for name, value in (
        ("preflight.json", checks),
        ("gold.json", native),
        ("candidate.json", candidate),
        ("lifecycle.json", lifecycle),
        ("perf.json", perf),
        ("report.json", report),
    ):
        (out / name).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
