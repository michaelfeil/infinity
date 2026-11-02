#!/usr/bin/env python3
"""Whole-model Native b66 CPU probe for Qwen3-VL.

Reads the original /bench checkpoint through the packet's exact shard offsets,
loads all weights through tei_load_safetensors on CPU, and runs one real
processor fixture with CUDA_VISIBLE_DEVICES=''. It never creates a CUDA context.

Root-only execution. The author did not run this in the SDK container.
"""
from __future__ import annotations
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path

class TeiConfig(C.Structure):
    _fields_ = [("hidden", C.c_int64), ("heads", C.c_int64), ("layers", C.c_int64),
                ("intermediate", C.c_int64), ("vocab", C.c_int64), ("positions", C.c_int64),
                ("types", C.c_int64), ("epsilon", C.c_double), ("activation", C.c_int32)]

class TeiImage(C.Structure):
    _fields_ = [("pixels", C.POINTER(C.c_float)), ("rows", C.c_int64), ("patch_dim", C.c_int64),
                ("grid", C.c_int64 * 3), ("merge_size", C.c_int64), ("token_start", C.c_int64),
                ("token_count", C.c_int64), ("sequence_start", C.c_int64),
                ("feature_key", C.c_ubyte * 32), ("has_feature_key", C.c_ubyte)]

class TeiWeight(C.Structure):
    _fields_ = [("name", C.c_char_p), ("shape", C.POINTER(C.c_int64)), ("rank", C.c_size_t),
                ("dtype", C.c_int32), ("offset", C.c_uint64), ("length", C.c_uint64)]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def error_text(lib) -> str:
    lib.tei_last_error_kind.restype = C.c_int32
    lib.tei_error.restype = C.c_char_p
    kind = lib.tei_last_error_kind()
    text = lib.tei_error()
    return f"kind={kind} {text.decode(errors='replace') if text else 'no error text'}"


def load_packet(lib, packet_path: Path):
    packet = json.loads(packet_path.read_text())
    cfg = packet["tei_config"]
    config = TeiConfig(cfg["hidden"], cfg["heads"], cfg["layers"], cfg["intermediate"],
                       cfg["vocab"], cfg["positions"], cfg["types"], cfg["epsilon"],
                       cfg["activation"])
    lib.tei_create.argtypes = [C.POINTER(TeiConfig), C.c_char_p, C.c_int32]
    lib.tei_create.restype = C.c_void_p
    handle = lib.tei_create(C.byref(config), b"cpu", 2)
    if not handle:
        raise RuntimeError(f"tei_create(cpu): {error_text(lib)}")
    lib.tei_option.argtypes = [C.c_void_p, C.c_char_p, C.c_char_p]
    lib.tei_option.restype = C.c_int32
    options = dict(packet.get("config_options") or {})
    options.update(packet.get("runtime_options") or {})
    options["_cuda_graphs"] = "false"
    options["_media_cuda_graphs"] = "false"
    for key, value in options.items():
        if lib.tei_option(handle, key.encode(), str(value).encode()) != 0:
            raise RuntimeError(f"tei_option {key}: {error_text(lib)}")
    lib.tei_load_safetensors.argtypes = [C.c_void_p, C.c_int32, C.c_uint64, C.c_uint64,
                                         C.POINTER(TeiWeight), C.c_size_t, C.c_size_t,
                                         C.c_size_t, C.c_int32]
    lib.tei_load_safetensors.restype = C.c_int32
    root = Path(packet["checkpoint"])
    weights_by_shard = {}
    for weight in packet["weights"]:
        weights_by_shard.setdefault(weight["shard"], []).append(weight)
    for shard in packet["inventory"]["shards"]:
        entries = weights_by_shard.get(shard["shard"], [])
        names = [C.c_char_p(w["original_name"].encode()) for w in entries]
        shapes = [(C.c_int64 * len(w["shape"]))(*w["shape"]) for w in entries]
        weights = []
        for index, w in enumerate(entries):
            weights.append(TeiWeight(names[index], shapes[index], len(w["shape"]),
                                     2 if w["dtype"] == "BF16" else 1,
                                     w["source_offset"], w["source_length"]))
        array = (TeiWeight * len(weights))(*weights)
        path = root / shard["shard"]
        fd = os.open(path, os.O_RDONLY)
        try:
            status = lib.tei_load_safetensors(handle, fd, shard["data_start"],
                                              shard["data_length"], array, len(weights),
                                              4, 64 << 20, 0)
        finally:
            os.close(fd)
        if status != 0:
            raise RuntimeError(f"tei_load_safetensors {shard['shard']}: {error_text(lib)}")
    lib.tei_ready.argtypes = [C.c_void_p]
    lib.tei_ready.restype = C.c_int32
    if lib.tei_ready(handle) != 0:
        raise RuntimeError(f"tei_ready: {error_text(lib)}")
    for name in ("tei_output_width", "tei_pooled_width", "tei_classification_width"):
        fn = getattr(lib, name)
        fn.argtypes = [C.c_void_p]
        fn.restype = C.c_int64
    return packet, handle


def fixture_inputs(fixture, fixture_root: Path):
    ids = [int(v) for v in fixture["ids"]]
    types = [int(v) for v in fixture.get("types") or [0] * len(ids)]
    positions = [int(v) for v in fixture["positions"]]
    cumulative = [int(v) for v in fixture["cumulative"]]
    media_positions = fixture.get("media_positions")
    media = None
    if media_positions is not None:
        media = []
        for axis in media_positions:
            media.extend(int(v) for v in axis)
    images = []
    owned = []
    for image in fixture.get("images") or []:
        path = Path(image["pixels"])
        if not path.is_absolute():
            path = fixture_root / path
        values = path.read_bytes()
        if len(values) % 4:
            raise RuntimeError(f"{path} is not F32")
        pixels = (C.c_float * (len(values) // 4)).from_buffer_copy(values)
        owned.append(pixels)
        images.append((image, pixels))
    return ids, types, positions, cumulative, media, images, owned


def run_fixture(lib, handle, fixture, fixture_root: Path, packet):
    ids, types, positions, cumulative, media, images, owned = fixture_inputs(fixture, fixture_root)
    sequences = len(cumulative) - 1
    width = int(lib.tei_output_width(handle))
    pooled_width = int(lib.tei_pooled_width(handle))
    raw_rows = sum(cumulative[i + 1] - cumulative[i] for i in range(sequences))
    capacity = sequences * pooled_width + raw_rows * width
    output = (C.c_float * capacity)()
    tei_images = []
    for image, pixels in images:
        entry = TeiImage()
        entry.pixels = pixels
        entry.rows = image["rows"]
        entry.patch_dim = image["patch_dim"]
        entry.grid = (C.c_int64 * 3)(*image["grid_thw"])
        entry.merge_size = image["merge_size"]
        entry.token_start = image["token_start"]
        entry.token_count = image["token_count"]
        entry.sequence_start = 0
        tei_images.append(entry)
    image_array = (TeiImage * len(tei_images))(*tei_images) if tei_images else None
    media_array = (C.c_int64 * len(media))(*media) if media is not None else None
    ids_array = (C.c_int64 * len(ids))(*ids)
    types_array = (C.c_int64 * len(types))(*types)
    positions_array = (C.c_int64 * len(positions))(*positions)
    cumulative_array = (C.c_int32 * len(cumulative))(*cumulative)
    pooled = (C.c_int64 * sequences)(*range(sequences))
    raw = (C.c_int64 * sequences)(*range(sequences))
    pool_code = {"cls": 0, "mean": 1, "last_token": 2}.get(fixture.get("pool"), 2)
    lib.tei_forward.argtypes = [C.c_void_p, C.POINTER(C.c_int64), C.POINTER(C.c_int64),
                                C.POINTER(C.c_int64), C.POINTER(C.c_int32), C.c_int64, C.c_int64,
                                C.c_int32, C.POINTER(C.c_int64), C.c_size_t, C.POINTER(C.c_int64),
                                C.c_size_t, C.POINTER(C.c_float), C.c_size_t, C.POINTER(C.c_int64),
                                C.POINTER(TeiImage), C.c_size_t, C.c_void_p, C.c_size_t]
    lib.tei_forward.restype = C.c_int32
    status = lib.tei_forward(handle, ids_array, types_array, positions_array, cumulative_array,
                             sequences, max(cumulative[i + 1] - cumulative[i] for i in range(sequences)),
                             pool_code, pooled, sequences, raw, sequences, output, capacity,
                             media_array, image_array, len(tei_images), None, 0)
    if status != 0:
        raise RuntimeError(f"tei_forward {fixture['name']}: {error_text(lib)}")
    values = list(output)
    return {
        "fixture": fixture["name"], "tokens": len(ids), "sequences": sequences,
        "output_width": width, "pooled_width": pooled_width, "capacity": capacity,
        "finite": all(v == v and abs(v) != float("inf") for v in values),
        "sha256": hashlib.sha256(bytes(output)).hexdigest(),
        "first": values[:8],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--library-sha", default=None)
    parser.add_argument("--packet", required=True)
    parser.add_argument("--fixtures", required=True)
    parser.add_argument("--fixture-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fixture", action="append")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1"):
        raise SystemExit("CUDA_VISIBLE_DEVICES must be empty for the CPU stage")
    library_sha = hashlib.sha256(Path(args.library).read_bytes()).hexdigest()
    if args.library_sha is not None and library_sha != args.library_sha:
        raise SystemExit(f"b66 sha mismatch: {library_sha} != {args.library_sha}")
    lib = C.CDLL(args.library)
    packet, handle = load_packet(lib, Path(args.packet))
    try:
        data = json.loads(Path(args.fixtures).read_text())
        fixtures = data["fixtures"]
        if args.fixture:
            fixtures = [f for f in fixtures if f["name"] in args.fixture]
        records = []
        for fixture in fixtures:
            records.append(run_fixture(lib, handle, fixture, Path(args.fixture_root), packet))
        report = {"schema": "oxide-qwen3-vl-native-cpu-whole/v1",
                  "library": args.library, "library_sha256": library_sha, "packet": args.packet,
                  "checkpoint": packet["checkpoint"], "records": records,
                  "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                  "status": "passed" if all(r["finite"] for r in records) else "failed"}
    finally:
        lib.tei_destroy.argtypes = [C.c_void_p]
        lib.tei_destroy.restype = None
        lib.tei_destroy(handle)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "fixtures": len(records)}))


if __name__ == "__main__":
    main()
