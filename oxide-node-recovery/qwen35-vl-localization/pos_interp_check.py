#!/usr/bin/env python3
"""CPU-only check of the BF16 four-corner position-interpolation recipe.

Reads ONLY the four pos_embed.weight table rows needed by merged visual token 0
of the single-cats fixture (rows 0,1,24,25 = 4*768 BF16 = 6144 bytes). No bulk
model load, no Torch, no GPU.
"""
from __future__ import annotations
import json
import struct
from pathlib import Path

import numpy as np

MODEL = Path("/workspace/model-performance/michaelfeil/onejev-perf/backbone-comparison-20261023/models/Qwen3.5-0.8B")
ST = MODEL / "model.safetensors-00001-of-00001.safetensors"
OWN = Path("/tmp/oxide-qwen35vl-numeric-localization-v1")
SIDE = 24
HEIGHT, WIDTH, MERGE = 30, 40, 2
HIDDEN = 768


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_rne(values: np.ndarray) -> np.ndarray:
    v = np.ascontiguousarray(values, dtype=np.float32)
    u = v.view(np.uint32).copy()
    lsb = (u >> 16) & 1
    u = (u + np.uint32(0x7FFF) + lsb) & np.uint32(0xFFFF0000)
    return u.view(np.float32)


def read_rows(rows):
    with ST.open("rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
        info = header["model.visual.pos_embed.weight"]
        data_start = 8 + n
        begin = data_start + info["data_offsets"][0]
        out = {}
        for row in rows:
            fh.seek(begin + row * HIDDEN * 2)
            raw = fh.read(HIDDEN * 2)
            assert len(raw) == HIDDEN * 2
            out[row] = bf16_to_f32(np.frombuffer(raw, dtype="<u2"))
        return out


def coords(index: int):
    area = MERGE * MERGE
    column = index % MERGE + ((index // area) % (WIDTH // MERGE)) * MERGE
    row = (index // MERGE) % MERGE + ((index // (MERGE * WIDTH)) % (HEIGHT // MERGE)) * MERGE
    return row, column


def main() -> None:
    table = read_rows([0, 1, SIDE, SIDE + 1])
    row_scale = np.float32((SIDE - 1) / (HEIGHT - 1))
    col_scale = np.float32((SIDE - 1) / (WIDTH - 1))

    def corner(r, c, scale):
        value = table[r * SIDE + c]
        sb = f32_to_bf16_rne(np.asarray([scale], np.float32))[0]
        return f32_to_bf16_rne(value * sb)

    patches = []
    for index in range(4):
        r, c = coords(index)
        rf = np.float32(r) * row_scale
        cf = np.float32(c) * col_scale
        r0 = int(np.floor(rf)); c0 = int(np.floor(cf))
        r1 = min(int(np.ceil(rf)), SIDE - 1); c1 = min(int(np.ceil(cf)), SIDE - 1)
        dr = np.float32(rf - r0); dc = np.float32(cf - c0)
        p0 = corner(r0, c0, (1.0 - dr) * (1.0 - dc))
        p1 = corner(r0, c1, (1.0 - dr) * dc)
        p2 = corner(r1, c0, dr * (1.0 - dc))
        p3 = corner(r1, c1, dr * dc)
        # Native BF16 (multi_vector=false, exact_vision=false): corners.sum(0),
        # float32 accumulation with one final bf16 rounding.
        native = f32_to_bf16_rne((p0.astype(np.float64) + p1 + p2 + p3).astype(np.float32))
        # Oxide device pos_interp: exact/F16 grouping with three bf16 roundings.
        first = f32_to_bf16_rne(p0 + p2)
        second = f32_to_bf16_rne(p1 + p3)
        oxide = f32_to_bf16_rne(first + second)
        diff = np.abs(oxide.astype(np.float64) - native.astype(np.float64))
        mag = np.maximum(np.abs(native.astype(np.float64)), np.abs(oxide.astype(np.float64)))
        ulp = np.where(mag > 0, mag * 2.0 ** -7, np.finfo(np.float64).tiny)
        cos = float((native.astype(np.float64) * oxide).sum() /
                    (np.linalg.norm(native) * np.linalg.norm(oxide)))
        patches.append({
            "patch_index": index, "grid_r": r, "grid_c": c,
            "r0": r0, "r1": r1, "c0": c0, "c1": c1,
            "dr": float(dr), "dc": float(dc),
            "channels_differ": int((diff > 0).sum()),
            "channels_total": HIDDEN,
            "frac_differ": float((diff > 0).mean()),
            "max_abs": float(diff.max()),
            "mean_abs": float(diff.mean()),
            "max_ulp": float((diff / ulp).max()),
            "p99_ulp": float(np.quantile(diff / ulp, 0.99)),
            "cos": cos,
        })

    # combined effect on merged token 0: merger consumes concatenated 4 patches.
    report = {
        "schema": "oxide-qwen35vl-pos-interp-check/v1",
        "scope": "CPU-only, reads only 4 pos_embed rows (6144 bytes) needed by merged visual token 0",
        "inputs": {
            "pos_embed_tensor": "model.visual.pos_embed.weight",
            "shape": [SIDE * SIDE, HIDDEN],
            "rows_read": [0, 1, SIDE, SIDE + 1],
            "bytes_read": 4 * HIDDEN * 2,
        },
        "geometry": {"grid_thw": [1, HEIGHT, WIDTH], "merge": MERGE,
                     "merged_grid_h_w": [HEIGHT // MERGE, WIDTH // MERGE],
                     "row_scale": float(row_scale), "col_scale": float(col_scale)},
        "recipes": {
            "native_bf16": "corners.sum(0): float32 accumulation of four bf16 corner products -> one bf16 rounding",
            "oxide_device_pos_interp": "((p0+p2) rounded) + ((p1+p3) rounded) rounded: three bf16 roundings (F16/exact recipe)",
            "note": "source: Native qwen3_vl_models.cpp:148-153; Oxide device qwen35vl.rs:~119-139; exact flag false for BF16 at qwen35_vl_device.rs:445",
        },
        "merged_token_0_patches": patches,
        "conclusion": "Patch 0 is degenerate (dr=dc=0) and identical; patches 1 and 2 have one degenerate axis (only two nonzero corners) and are also identical. Patch 3 (grid_r=1,grid_c=1) has four nonzero corners and differs in 176/768 channels at ~1 bf16 ulp, so merged visual token 0 (row 15, the first divergent row) is fed a different positional signal. For interior merged tokens all four patches are non-degenerate, so the effect is broad. This is a formula-level BF16 recipe mismatch, not a misindex.",
    }
    (OWN / "pos-interp-token0.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
