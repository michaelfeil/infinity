#!/usr/bin/env python3
"""Complete Native b66 whole-model gold producer for Qwen3.5-VL.

Runs inside the pinned SDK container with the actual b66 library. It loads every
original checkpoint tensor through the Native C loader (no Torch, no network,
no surrogate weights) and runs tei_forward with the original image spans and
external 3-axis mRoPE positions. Output is the Native pooled/raw hidden states.
"""
from __future__ import annotations
import argparse, ctypes as C, json, math, os, struct, sys
from pathlib import Path

BF16 = 2
F16 = 1
F32 = 0
DTYPE = {"BF16": BF16, "F16": F16, "F32": F32}


class TeiConfig(C.Structure):
    _fields_ = [
        ("hidden", C.c_int64), ("heads", C.c_int64), ("layers", C.c_int64),
        ("intermediate", C.c_int64), ("vocab", C.c_int64),
        ("positions", C.c_int64), ("types", C.c_int64),
        ("epsilon", C.c_double), ("activation", C.c_int32),
    ]


class TeiImage(C.Structure):
    _fields_ = [
        ("pixels", C.POINTER(C.c_float)),
        ("rows", C.c_int64), ("patch_dim", C.c_int64),
        ("grid", C.c_int64 * 3), ("merge_size", C.c_int64),
        ("token_start", C.c_int64), ("token_count", C.c_int64),
        ("sequence_start", C.c_int64),
        ("feature_key", C.c_uint8 * 32), ("has_feature_key", C.c_uint8),
    ]


class TeiAudio(C.Structure):
    _fields_ = [
        ("values", C.POINTER(C.c_float)), ("mask", C.POINTER(C.c_uint8)),
        ("frames", C.c_int64), ("feature_size", C.c_int64),
        ("token_start", C.c_int64), ("token_count", C.c_int64),
    ]


class TeiDecisionField(C.Structure):
    _fields_ = [("kind", C.c_int64), ("question_start", C.c_int64),
                ("question_end", C.c_int64), ("options", C.POINTER(C.c_int64)),
                ("option_count", C.c_size_t)]


class TeiDecision(C.Structure):
    _fields_ = [("kind", C.c_int32), ("question_type", C.c_int64),
                ("markers", C.POINTER(C.c_int64)), ("marker_count", C.c_size_t),
                ("token_ids", C.POINTER(C.c_int64)), ("token_count", C.c_size_t),
                ("fields", C.POINTER(TeiDecisionField)), ("field_count", C.c_size_t)]


class TeiWeight(C.Structure):
    _fields_ = [
        ("name", C.c_char_p), ("shape", C.POINTER(C.c_int64)),
        ("rank", C.c_size_t), ("dtype", C.c_int32),
        ("offset", C.c_uint64), ("length", C.c_uint64),
    ]


def flatten(value, prefix, out):
    if isinstance(value, dict):
        for key, item in value.items():
            flatten(item, f"{prefix}.{key}" if prefix else key, out)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            flatten(item, f"{prefix}.{index}", out)
    elif isinstance(value, bool):
        out[prefix] = "true" if value else "false"
    elif value is not None:
        out[prefix] = str(value)


def read_header(path: Path):
    with path.open("rb") as handle:
        raw = handle.read(8)
        (length,) = struct.unpack("<Q", raw)
        header = json.loads(handle.read(length))
    return length, header


def load_shard(lib, handle, path: Path):
    length, header = read_header(path)
    data_start = 8 + length
    file_len = path.stat().st_size
    names, shapes, descs = [], [], []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype = DTYPE.get(info["dtype"])
        if dtype is None:
            raise SystemExit(f"unsupported Native dtype {info['dtype']} for {name}")
        shape = (C.c_int64 * len(info["shape"]))(*[int(v) for v in info["shape"]])
        begin, end = info["data_offsets"]
        raw_name = name.encode()
        names.append(raw_name); shapes.append(shape)
        descs.append(TeiWeight(name=raw_name, shape=shape, rank=len(info["shape"]),
                               dtype=dtype, offset=int(begin), length=int(end - begin)))
    array = (TeiWeight * len(descs))(*descs)
    with path.open("rb") as handle_file:
        fd = handle_file.fileno()
        status = lib.tei_load_safetensors(
            handle, fd, C.c_uint64(data_start), C.c_uint64(file_len - data_start),
            array, len(descs), 4, C.c_size_t(64 << 20), 0,
        )
    if status != 0:
        raise SystemExit(f"tei_load_safetensors {path.name}: {lib.tei_error().decode()}")


def media_positions(tokens: int, spans):
    ordered = sorted(spans, key=lambda item: item["token_start"])
    axes = [[], [], []]
    cursor = 0
    nxt = 0
    for span in ordered:
        t, h, w = span["grid"]
        merge = span["merge"]
        assert h % merge == 0 and w % merge == 0
        length = t * h * w // (merge * merge)
        if length != span["token_count"] or span["token_start"] < cursor:
            raise SystemExit("invalid media span")
        end = span["token_start"] + length
        if end > tokens:
            raise SystemExit("media span exceeds tokens")
        for _ in range(cursor, span["token_start"]):
            for axis in axes:
                axis.append(nxt)
            nxt += 1
        height, width = h // merge, w // merge
        for temporal in range(t):
            for row in range(height):
                for column in range(width):
                    axes[0].append(nxt + temporal)
                    axes[1].append(nxt + row)
                    axes[2].append(nxt + column)
        nxt += max(height, width, t)
        cursor = end
    for _ in range(cursor, tokens):
        for axis in axes:
            axis.append(nxt)
        nxt += 1
    if any(len(axis) != tokens for axis in axes):
        raise SystemExit("position length mismatch")
    return axes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--media", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pool", default="mean")
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--cuda-graphs", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--decision", action="store_true")
    args = parser.parse_args()
    model = Path(args.model)
    config = json.loads((model / "config.json").read_text())
    text = config["text_config"]
    if args.plan_only:
        media = json.loads(Path(args.media).read_text())
        batch = media["text"]
        tokens = len(batch["input_ids"])
        spans = []
        for fixture in media.get("fixtures", []):
            grid = fixture["grid"]; merge = fixture["merge"]
            rows = grid[0] * grid[1] * grid[2]
            spans.append({"grid": grid, "merge": merge,
                          "token_start": int(fixture.get("token_start", 0)),
                          "token_count": rows // (merge * merge)})
        axes = media_positions(tokens, spans)
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        (out / "native-whole-plan.json").write_text(json.dumps({
            "model": str(model), "tokens": tokens, "spans": spans,
            "positions": axes,
        }, indent=2) + "\n")
        print(json.dumps({"status": "plan-only", "tokens": tokens, "out": str(out)}))
        return
    if not args.allow_gpu or os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise SystemExit("Native whole-model requires --allow-gpu and CUDA_VISIBLE_DEVICES=0")
    # Native accepts the published qwen3_5/qwen3_5_moe root; no separate VL type.
    lib = C.CDLL(args.library)
    lib.tei_create.restype = C.c_void_p
    lib.tei_create.argtypes = [C.POINTER(TeiConfig), C.c_char_p, C.c_int32]
    lib.tei_option.restype = C.c_int32
    lib.tei_option.argtypes = [C.c_void_p, C.c_char_p, C.c_char_p]
    lib.tei_load_safetensors.restype = C.c_int32
    lib.tei_load_safetensors.argtypes = [C.c_void_p, C.c_int, C.c_uint64, C.c_uint64,
                                         C.POINTER(TeiWeight), C.c_size_t, C.c_size_t,
                                         C.c_size_t, C.c_int32]
    lib.tei_ready.restype = C.c_int32; lib.tei_ready.argtypes = [C.c_void_p]
    lib.tei_error.restype = C.c_char_p
    lib.tei_destroy.argtypes = [C.c_void_p]
    lib.tei_decision_counts.restype = C.c_int32
    lib.tei_decision_counts.argtypes = [C.c_void_p, C.POINTER(TeiDecision), C.c_size_t,
                                        C.POINTER(C.c_int64)]
    lib.tei_decide.restype = C.c_int32
    lib.tei_decide.argtypes = [C.c_void_p, C.POINTER(C.c_int64), C.POINTER(C.c_int64),
                               C.POINTER(C.c_int64), C.POINTER(C.c_int32), C.c_int64,
                               C.c_int64, C.POINTER(TeiDecision), C.POINTER(C.c_float),
                               C.c_size_t, C.POINTER(C.c_float), C.POINTER(C.c_int64),
                               C.POINTER(TeiImage), C.c_size_t, C.POINTER(TeiAudio),
                               C.c_size_t]
    lib.tei_forward.restype = C.c_int32
    lib.tei_forward.argtypes = [C.c_void_p, C.POINTER(C.c_int64), C.POINTER(C.c_int64),
                                C.POINTER(C.c_int64), C.POINTER(C.c_int32), C.c_int64,
                                C.c_int64, C.c_int32, C.POINTER(C.c_int64), C.c_size_t,
                                C.POINTER(C.c_int64), C.c_size_t, C.POINTER(C.c_float),
                                C.c_size_t, C.POINTER(C.c_int64), C.POINTER(TeiImage),
                                C.c_size_t, C.POINTER(TeiAudio), C.c_size_t]
    cfg = TeiConfig(
        hidden=int(text["hidden_size"]), heads=int(text["num_attention_heads"]),
        layers=int(text["num_hidden_layers"]),
        intermediate=int(text.get("intermediate_size", 0) or 0),
        vocab=int(text["vocab_size"]), positions=int(text["max_position_embeddings"]),
        types=0, epsilon=float(text["rms_norm_eps"]), activation=0,
    )
    handle = lib.tei_create(C.byref(cfg), b"cuda", BF16)
    if not handle:
        raise SystemExit(f"tei_create: {lib.tei_error().decode()}")
    try:
        options = {}
        flatten(config, "", options)
        options["_cuda_graphs"] = "true" if args.cuda_graphs else "false"
        options["_pooling"] = args.pool
        for key, value in sorted(options.items()):
            status = lib.tei_option(handle, key.encode(), value.encode())
            if status != 0:
                raise SystemExit(f"tei_option {key}: {lib.tei_error().decode()}")
        index = json.loads((model / "model.safetensors.index.json").read_text())
        for shard in sorted(set(index["weight_map"].values())):
            load_shard(lib, handle, model / shard)
        status = lib.tei_ready(handle)
        if status != 0:
            raise SystemExit(f"tei_ready: {lib.tei_error().decode()}")
        media = json.loads(Path(args.media).read_text())
        fixture_base = Path(args.media).parent
        batch = media["text"]
        ids = batch["input_ids"]
        tokens = len(ids)
        cumulative = batch["cumulative_seq_lengths"]
        positions = batch.get("position_ids") or list(range(tokens))
        types = batch.get("token_type_ids") or [0] * tokens
        pooled = batch.get("pooled_indices") or [0]
        raw = batch.get("raw_indices") or []
        spans = []
        images, keep = [], []
        for fixture in media.get("fixtures", []):
            grid = fixture["grid"]; merge = fixture["merge"]
            rows = grid[0] * grid[1] * grid[2]
            patch_dim = fixture["patch_dim"]
            pixel_path = Path(fixture["pixels"])
            if not pixel_path.is_absolute():
                pixel_path = fixture_base / pixel_path
            values = struct.unpack(f"<{rows*patch_dim}f", pixel_path.read_bytes())
            pixels = (C.c_float * len(values))(*values)
            token_count = rows // (merge * merge)
            image = TeiImage(pixels=pixels, rows=rows, patch_dim=patch_dim,
                             grid=(C.c_int64 * 3)(*grid), merge_size=merge,
                             token_start=int(fixture.get("token_start", 0)),
                             token_count=token_count,
                             sequence_start=int(fixture.get("sequence_start", 0)))
            images.append(image); keep.append(pixels)
            spans.append({"grid": grid, "merge": merge,
                          "token_start": image.token_start, "token_count": token_count})
        axes = media_positions(tokens, spans)
        flat_positions = [value for axis in axes for value in axis]
        media_positions_array = (C.c_int64 * len(flat_positions))(*flat_positions)
        id_array = (C.c_int64 * tokens)(*ids)
        type_array = (C.c_int64 * tokens)(*types)
        position_array = (C.c_int64 * tokens)(*positions)
        cumulative_array = (C.c_int32 * len(cumulative))(*cumulative)
        pooled_array = (C.c_int64 * len(pooled))(*pooled) if pooled else None
        raw_array = (C.c_int64 * len(raw))(*raw) if raw else None
        image_array = (TeiImage * len(images))(*images) if images else None
        hidden = int(text["hidden_size"])
        if args.decision:
            option_tokens = (media["text"].get("option_tokens") or [])
            prefix_length = int(media["text"].get("prefix_length") or 0)
            option_array = (C.c_int64 * len(option_tokens))(*option_tokens) if option_tokens else None
            decision = TeiDecision(kind=4, question_type=prefix_length, markers=None,
                                   marker_count=0, token_ids=option_array,
                                   token_count=len(option_tokens), fields=None,
                                   field_count=0)
            counts = (C.c_int64 * 1)(0)
            status = lib.tei_decision_counts(handle, C.byref(decision), 1, counts)
            if status != 0 or counts[0] <= 0:
                raise SystemExit(f"tei_decision_counts: {lib.tei_error().decode()}")
            capacity = int(counts[0])
            logits = (C.c_float * capacity)()
            actions = (C.c_float * 1)()
            status = lib.tei_decide(
                handle, id_array, type_array, position_array, cumulative_array,
                len(cumulative) - 1, max(cumulative), C.byref(decision), logits,
                capacity, actions, media_positions_array, image_array, len(images),
                None, 0,
            )
            if status != 0:
                raise SystemExit(f"tei_decide: {lib.tei_error().decode()}")
            rows = [{"kind": "decision", "logits": list(logits),
                     "action_probability": float(actions[0])}]
            out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
            (out / "native-whole.json").write_text(json.dumps({
                "arm": "native-b66-whole", "decision": True,
                "tokens": tokens, "rows": rows,
            }, indent=2) + "\n")
            print(json.dumps({"status": "ok", "decision": True, "rows": len(rows)}))
            return
        raw_rows = sum((cumulative[i + 1] - cumulative[i]) for i in raw) if raw else 0
        capacity = len(pooled) * hidden + raw_rows * hidden
        output = (C.c_float * capacity)()
        pool_code = {"cls": 0, "mean": 1, "last": 2, "last_token": 2}[args.pool]
        status = lib.tei_forward(
            handle, id_array, type_array, position_array, cumulative_array,
            len(cumulative) - 1, max(cumulative), pool_code,
            pooled_array, len(pooled), raw_array, len(raw),
            output, capacity, media_positions_array, image_array, len(images),
            None, 0,
        )
        if status != 0:
            raise SystemExit(f"tei_forward: {lib.tei_error().decode()}")
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        values = list(output)
        rows = []
        offset = 0
        for index_value in pooled:
            rows.append({"index": index_value, "kind": "pooled",
                         "values": values[offset:offset + hidden]})
            offset += hidden
        for index_value in raw:
            start = cumulative[index_value]
            length = cumulative[index_value + 1] - start
            rows.append({"index": index_value, "kind": "all",
                         "values": values[offset:offset + length * hidden]})
            offset += length * hidden
        (out / "native-whole.json").write_text(json.dumps({
            "arm": "native-b66-whole", "pool": args.pool,
            "tokens": tokens, "rows": rows,
        }, indent=2) + "\n")
        print(json.dumps({"status": "ok", "rows": len(rows), "tokens": tokens}))
    finally:
        lib.tei_destroy(handle)


if __name__ == "__main__":
    main()
