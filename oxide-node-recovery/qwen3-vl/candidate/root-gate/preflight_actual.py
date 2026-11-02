#!/usr/bin/env python3
"""Fresh coupled Qwen3-VL CPU preflight.

Runs BEFORE any GPU arm with CUDA_VISIBLE_DEVICES='' in the existing
infinity-torch214-compare-sdk container. It validates the actual checkpoint,
packets, fixtures, Native b66 identity/exports/ABI and the host consumer
binaries. It never creates a CUDA context or imports Torch CUDA.

On a host without the SDK use --skip-sdk to validate the checkpoint, packet,
fixtures and local consumer identities only.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import struct
import subprocess
from pathlib import Path

MARKER = "QWEN3_VL_E582_CPU_PREFLIGHT_COMPLETE"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def headers(root: Path):
    index = json.loads((root / "model.safetensors.index.json").read_text())
    tensors = {}
    shards = {}
    for name in sorted(set(index["weight_map"].values())):
        path = root / name
        with path.open("rb") as handle:
            length = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(length))
        rows = {key: value for key, value in header.items() if key != "__metadata__"}
        shards[name] = {
            "file_bytes": path.stat().st_size,
            "header_bytes": length,
            "tensor_count": len(rows),
        }
        for tensor, info in rows.items():
            if tensor in tensors:
                raise SystemExit(f"duplicate tensor {tensor}")
            tensors[tensor] = {
                "dtype": info["dtype"],
                "shape": [int(value) for value in info["shape"]],
                "data_offsets": [int(value) for value in info["data_offsets"]],
                "shard": name,
            }
    return index, shards, tensors


def abi(b66: Path):
    lib = ctypes.CDLL(str(b66))

    class TeiConfig(ctypes.Structure):
        _fields_ = [
            ("hidden", ctypes.c_int64),
            ("heads", ctypes.c_int64),
            ("layers", ctypes.c_int64),
            ("intermediate", ctypes.c_int64),
            ("vocab", ctypes.c_int64),
            ("positions", ctypes.c_int64),
            ("types", ctypes.c_int64),
            ("epsilon", ctypes.c_double),
            ("activation", ctypes.c_int32),
        ]

    class TeiImage(ctypes.Structure):
        _fields_ = [
            ("pixels", ctypes.c_void_p),
            ("rows", ctypes.c_int64),
            ("patch_dim", ctypes.c_int64),
            ("grid", ctypes.c_int64 * 3),
            ("merge_size", ctypes.c_int64),
            ("token_start", ctypes.c_int64),
            ("token_count", ctypes.c_int64),
            ("sequence_start", ctypes.c_int64),
            ("feature_key", ctypes.c_ubyte * 32),
            ("has_feature_key", ctypes.c_ubyte),
        ]

    class TeiWeight(ctypes.Structure):
        _fields_ = [
            ("name", ctypes.c_char_p),
            ("shape", ctypes.POINTER(ctypes.c_int64)),
            ("rank", ctypes.c_size_t),
            ("dtype", ctypes.c_int32),
            ("offset", ctypes.c_uint64),
            ("length", ctypes.c_uint64),
        ]

    sizes = {
        "TeiConfig": ctypes.sizeof(TeiConfig),
        "TeiImage": ctypes.sizeof(TeiImage),
        "TeiWeight": ctypes.sizeof(TeiWeight),
    }
    if sizes != {"TeiConfig": 72, "TeiImage": 120, "TeiWeight": 48}:
        raise SystemExit(f"ABI sizes mismatch: {sizes}")
    lib.tei_device_preflight.argtypes = [ctypes.c_char_p]
    lib.tei_device_preflight.restype = ctypes.c_int32
    if lib.tei_device_preflight(b"cpu") != 0:
        raise SystemExit("tei_device_preflight(cpu) failed")
    lib.tei_create.argtypes = [ctypes.POINTER(TeiConfig), ctypes.c_char_p, ctypes.c_int32]
    lib.tei_create.restype = ctypes.c_void_p
    lib.tei_option.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
    lib.tei_option.restype = ctypes.c_int32
    lib.tei_destroy.argtypes = [ctypes.c_void_p]
    lib.tei_destroy.restype = None
    config = TeiConfig(4096, 32, 36, 12288, 151936, 262144, 0, 1e-6, 3)
    handle = lib.tei_create(ctypes.byref(config), b"cpu", 2)
    if not handle:
        raise SystemExit("tei_create(cpu, BF16) failed")
    try:
        for key, value in [
            (b"model_type", b"qwen3_vl"),
            (b"_multi_vector", b"false"),
            (b"_cuda_graphs", b"false"),
            (b"_media_cuda_graphs", b"false"),
        ]:
            if lib.tei_option(handle, key, value) != 0:
                raise SystemExit(f"tei_option {key} failed")
    finally:
        lib.tei_destroy(handle)
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--b66", type=Path)
    parser.add_argument("--torch-python", default="/opt/torch-sdk/bin/python3.12")
    parser.add_argument("--vendor", type=Path)
    parser.add_argument("--checkpoint-hashes", type=Path, required=True)
    parser.add_argument("--native-consumer", type=Path, required=True)
    parser.add_argument("--oxide-consumer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-sdk", action="store_true")
    args = parser.parse_args()

    build = json.loads(args.build.read_text())
    packet = json.loads(args.packet.read_text())
    hashes = json.loads(args.checkpoint_hashes.read_text())
    failures = []

    def check(name: str, condition: bool, detail) -> None:
        if not condition:
            failures.append({"check": name, "detail": detail})

    check(
        "cuda_hidden",
        os.environ.get("CUDA_VISIBLE_DEVICES", "") in ("", "-1"),
        os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    for name, expected in hashes.items():
        path = args.checkpoint / name
        check(
            f"hash:{name}",
            path.is_file()
            and path.stat().st_size == expected["bytes"]
            and sha256(path) == expected["sha256"],
            f"{path} hash/size mismatch",
        )
    index, shards, tensors = headers(args.checkpoint)
    total = sum(value["data_offsets"][1] - value["data_offsets"][0] for value in tensors.values())
    check("tensor_count", len(tensors) == 749, len(tensors))
    check("total_bytes", total == 16289587680, total)
    check("packet_tensor_count", packet["inventory"]["tensor_count"] == 749, packet["inventory"]["tensor_count"])
    packet_weights = {weight["original_name"]: weight for weight in packet["weights"]}
    check("packet_names", set(packet_weights) == set(tensors), "packet/checkpoint name mismatch")
    for name, meta in tensors.items():
        weight = packet_weights.get(name)
        if weight is None:
            continue
        expected = (
            weight["shard"],
            weight["dtype"],
            weight["shape"],
            weight["source_offset"],
            weight["source_length"],
        )
        actual = (
            meta["shard"],
            meta["dtype"],
            meta["shape"],
            meta["data_offsets"][0],
            meta["data_offsets"][1] - meta["data_offsets"][0],
        )
        if expected != actual:
            failures.append({"check": f"weight:{name}", "detail": "packet weight drift"})
            break
    original = json.loads((args.checkpoint / "config.json").read_text())
    check("original_config", packet["original_config"] == original, "packet config differs")
    check("packet_base", packet["base_commit"] == build["base_commit"], packet["base_commit"])
    check("profile", packet.get("runtime_profile") == "native_production", packet.get("runtime_profile"))
    check("graphs", packet.get("runtime_options", {}).get("_cuda_graphs") == "true", "native graphs disabled")
    check(
        "media_graphs",
        packet.get("runtime_options", {}).get("_media_cuda_graphs") == "false",
        "media graphs not at recorded default",
    )
    for key in ("native", "oxide"):
        binary = args.__dict__[f"{key}_consumer"]
        expected = build[f"{key}_binaries"]["release_consumer"]
        check(f"{key}_binary_path", str(binary) == expected["path"], str(binary))
        check(f"{key}_binary_sha", binary.is_file() and sha256(binary) == expected["sha256"], str(binary))
    fixtures = json.loads(args.fixtures.read_text())
    names = set()
    for fixture in fixtures["fixtures"]:
        names.add(fixture["name"])
        cumulative = fixture["cumulative"]
        if cumulative[0] != 0 or cumulative[-1] != len(fixture["ids"]) or any(
            left > right for left, right in zip(cumulative, cumulative[1:])
        ):
            failures.append({"check": f"fixture:{fixture['name']}", "detail": "bad cumulative"})
        positions = fixture.get("media_positions")
        if positions is not None and any(len(axis) != len(fixture["ids"]) for axis in positions):
            failures.append({"check": f"fixture:{fixture['name']}", "detail": "bad media positions"})
        for image in fixture.get("images", []):
            path = Path(image["pixels"])
            if not path.is_file() or path.stat().st_size != image["rows"] * image["patch_dim"] * 4:
                failures.append({"check": f"fixture:{fixture['name']}:{path}", "detail": "pixel file missing/wrong size"})
            if image["token_count"] * image["merge_size"] * image["merge_size"] != image["rows"]:
                failures.append({"check": f"fixture:{fixture['name']}:{path}", "detail": "grid/token mismatch"})
    check("fixture_count", len(names) == 8, len(names))

    b66_sha = None
    sizes = None
    torch = None
    cublaslt = None
    if not args.skip_sdk:
        b66 = args.b66
        b66_sha = sha256(b66)
        check("b66_sha256", b66_sha == build["b66"]["sha256"], b66_sha)
        exports = set(subprocess.run(["nm", "-D", "--defined-only", str(b66)], text=True, capture_output=True).stdout.split())
        required = [
            "tei_create",
            "tei_device_preflight",
            "tei_option",
            "tei_load_safetensors",
            "tei_ready",
            "tei_forward",
            "tei_forward_submit",
            "tei_output_wait",
            "tei_output_destroy",
            "tei_graph_count",
            "tei_graph_stats",
            "tei_prefix_cache_supported",
            "tei_prefix_cache_stats",
            "tei_prefix_cache_clear",
            "tei_output_width",
            "tei_pooled_width",
            "tei_classification_width",
            "tei_destroy",
            "tei_error",
            "tei_last_error_kind",
        ]
        for name in required:
            check(f"export:{name}", name in exports, "missing")
        dynamic = subprocess.run(["readelf", "-d", str(b66)], text=True, capture_output=True).stdout
        check("cuda_link", "libtorch_cuda" in dynamic, "no libtorch_cuda dependency")
        sizes = abi(b66)
        probe = "import torch,json;print(json.dumps({'version':torch.__version__,'git':torch.version.git_version,'cuda':torch.version.cuda,'file':torch.__file__}))"
        torch = json.loads(subprocess.check_output([args.torch_python, "-c", probe], text=True))
        expected_torch = build["b66"]["torch"]
        check("torch_version", torch["version"] == expected_torch["version"], torch)
        check("torch_git", torch["git"] == expected_torch["git_version"], torch["git"])
        check(
            "torch_path",
            str(Path(torch["file"]).resolve().parent) == expected_torch["root"],
            torch["file"],
        )
        check("torch_include", Path(expected_torch["include"]).is_dir(), expected_torch["include"])
        check("torch_lib", Path(expected_torch["lib"]).is_dir(), expected_torch["lib"])
        lt = ctypes.CDLL(str(args.vendor / "libcublasLt.so.13"))
        lt.cublasLtGetVersion.restype = ctypes.c_size_t
        cublaslt = lt.cublasLtGetVersion()
        check("cublaslt", cublaslt == 130200, cublaslt)

    result = {
        "schema": "oxide-qwen3-vl-e582-cpu-preflight/v1",
        "marker": MARKER,
        "container": build["container"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "sdk_checked": not args.skip_sdk,
        "b66": str(args.b66) if args.b66 else None,
        "b66_sha256": b66_sha,
        "torch": torch,
        "cublaslt_version": cublaslt,
        "abi_sizes": sizes,
        "checkpoint": str(args.checkpoint),
        "tensor_count": len(tensors),
        "total_bytes": total,
        "fixtures": len(names),
        "native_consumer_sha256": sha256(args.native_consumer),
        "oxide_consumer_sha256": sha256(args.oxide_consumer),
        "failures": failures,
        "status": "passed" if not failures else "failed",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"marker": MARKER, "status": result["status"], "failures": len(failures)}))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
