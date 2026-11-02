#!/usr/bin/env python3
"""CPU-only numeric localization for the paired Qwen3.5-0.8B single-cats run.

Reads ONLY the immutable paired outputs and the actual model config. Uses
Python/NumPy (no Torch, no GPU, no network). Writes compact aggregates.
"""
from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

PAIRED = Path("/tmp/oxide-qwen35vl-sdk-map-policy-output/paired-0.8B-single-v4")
CASE = PAIRED / "Qwen3.5-0.8B-single-cats"
OWN = Path("/tmp/oxide-qwen35vl-numeric-localization-v1")
MODEL = Path("/workspace/model-performance/michaelfeil/onejev-perf/backbone-comparison-20261023/models/Qwen3.5-0.8B")

NATIVE = CASE / "native/native-whole.json"
OXIDE = CASE / "oxide.json"
ROOTRAW = CASE / "root-raw-diagnostic.json"
MEDIA = PAIRED / "cases/single-cats.media.json"
REPORT = PAIRED / "report.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pin(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def bf16_rne(values: np.ndarray) -> np.ndarray:
    """Round float32 array to bfloat16 (round-to-nearest-even), kept in float32."""
    v = np.ascontiguousarray(values, dtype=np.float32)
    u = v.view(np.uint32).copy()
    lsb = (u >> 16) & 1
    u = (u + np.uint32(0x7FFF) + lsb) & np.uint32(0xFFFF0000)
    return u.view(np.float32)


def main() -> None:
    native = json.loads(NATIVE.read_text())
    oxide = json.loads(OXIDE.read_text())
    rootraw = json.loads(ROOTRAW.read_text())
    media = json.loads(MEDIA.read_text())
    config = json.loads((MODEL / "config.json").read_text())

    nrows = {}
    for row in native["rows"]:
        nrows[(int(row.get("index", 0)), row.get("kind"))] = row
    orows = {}
    for row in oxide["rows"]:
        orows[(int(row.get("index", 0)), row.get("kind"))] = row

    pooled = np.asarray(nrows[(0, "pooled")]["values"], dtype=np.float32)
    nall = np.asarray(nrows[(0, "all")]["values"], dtype=np.float32)
    oall = np.asarray(orows[(0, "all")]["values"], dtype=np.float32)
    tokens = int(native["tokens"])
    hidden = int(config["text_config"]["hidden_size"])
    n = nall.reshape(tokens, hidden)
    o = oall.reshape(tokens, hidden)

    nbits = n.view(np.uint32)
    obits = o.view(np.uint32)
    eq = nbits == obits
    diff = o.astype(np.float64) - n.astype(np.float64)
    adiff = np.abs(diff)

    fixture = media["fixtures"][0]
    grid = [int(x) for x in fixture["grid"]]
    merge = int(fixture["merge"])
    token_start = int(fixture["token_start"])
    token_count = int(fixture["token_count"])
    vision_h, vision_w = grid[1] // merge, grid[2] // merge
    vision_end = token_start + token_count - 1
    assert vision_h * vision_w == token_count
    assert len(media["text"]["input_ids"]) == tokens

    first_row = int(np.argmax(~eq.all(axis=1))) if (~eq.all(axis=1)).any() else None
    rows_any = np.where(~eq.all(axis=1))[0]
    prefix_exact_rows = 0
    while prefix_exact_rows < tokens and eq[prefix_exact_rows].all():
        prefix_exact_rows += 1

    # per-row compact stats
    row_rows = []
    for i in range(tokens):
        a = n[i].astype(np.float64)
        b = o[i].astype(np.float64)
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        den = na * nb
        cos = float(np.dot(a, b) / den) if den > 0 else float("nan")
        row_rows.append({
            "row": i,
            "region": "prefix" if i < token_start else ("visual" if i <= vision_end else "suffix"),
            "grid_r": (i - token_start) // vision_w if token_start <= i <= vision_end else "",
            "grid_c": (i - token_start) % vision_w if token_start <= i <= vision_end else "",
            "mismatch": int((~eq[i]).sum()),
            "mean_abs": float(adiff[i].mean()),
            "max_abs": float(adiff[i].max()),
            "argmax_col": int(adiff[i].argmax()),
            "cos": cos,
            "norm_native": na,
            "norm_oxide": nb,
        })
    with (OWN / "per-row.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row_rows[0]))
        w.writeheader()
        w.writerows(row_rows)

    # per-column compact stats over the visual span
    vis = slice(token_start, vision_end + 1)
    col_mean = adiff[vis].mean(0)
    col_max = adiff[vis].max(0)
    col_rms = np.sqrt((diff[vis] ** 2).mean(0))
    col_mismatch = (~eq[vis]).sum(0)
    col_native_absmax = np.abs(n[vis]).max(0)
    col_native_absmean = np.abs(n[vis]).mean(0)
    col_argmax_rows = [int(np.argmax(adiff[vis][:, c])) + token_start for c in range(hidden)]
    order = np.argsort(col_mean)[::-1]
    with (OWN / "per-column.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["col", "mismatch_rows", "mean_abs", "max_abs", "rms", "native_absmax", "native_absmean", "max_row"])
        for c in order:
            w.writerow([int(c), int(col_mismatch[c]), float(col_mean[c]), float(col_max[c]),
                        float(col_rms[c]), float(col_native_absmax[c]), float(col_native_absmean[c]),
                        int(col_argmax_rows[c])])

    # top element mismatches
    flat = np.argsort(adiff.ravel())[::-1][:64]
    top = []
    for f in flat:
        i, j = divmod(int(f), hidden)
        top.append({"row": i, "col": j, "native": float(n[i, j]), "oxide": float(o[i, j]),
                    "diff": float(diff[i, j]), "abs_diff": float(adiff[i, j])})
    with (OWN / "top-mismatch.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(top[0]))
        w.writeheader()
        w.writerows(top)

    # permutation / shift / layout diagnostics
    nv = n[vis].astype(np.float64)
    ov = o[vis].astype(np.float64)

    def cos_raw(a, b):
        return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))

    shifts = {}
    for k in range(-3, 4):
        idx = np.arange(token_count)
        j = idx + k
        m = (j >= 0) & (j < token_count)
        shifts[str(k)] = cos_raw(ov[idx[m]], nv[j[m]])

    nn = ov / np.linalg.norm(ov, axis=1, keepdims=True)
    nm = nv / np.linalg.norm(nv, axis=1, keepdims=True)
    cc = nn @ nm.T
    best = cc.argmax(1)
    identity = np.arange(token_count)
    nn_identity = int((best == identity).sum())
    non_identity = [{"vis_row": int(i), "best_vis_row": int(best[i]),
                     "best_cos": float(cc[i, best[i]]), "identity_cos": float(cc[i, i])}
                    for i in np.where(best != identity)[0]]

    # hidden-dimension permutation check
    nc = nv - nv.mean(0)
    oc = ov - ov.mean(0)
    ncn = nc / np.linalg.norm(nc, axis=0, keepdims=True)
    ocn = oc / np.linalg.norm(oc, axis=0, keepdims=True)
    colcc = ocn.T @ ncn
    col_best = colcc.argmax(1)
    col_identity = int((col_best == np.arange(hidden)).sum())

    nvv = nv.reshape(vision_h, vision_w, hidden)
    ovv = ov.reshape(vision_h, vision_w, hidden)
    layout = {
        "identity": cos_raw(ovv.reshape(-1, hidden), nvv.reshape(-1, hidden)),
        "transpose": cos_raw(ovv.transpose(1, 0, 2).reshape(-1, hidden), nvv.reshape(-1, hidden)),
        "flip_rows": cos_raw(ovv[::-1].reshape(-1, hidden), nvv.reshape(-1, hidden)),
        "flip_cols": cos_raw(ovv[:, ::-1].reshape(-1, hidden), nvv.reshape(-1, hidden)),
        "reverse_flat": cos_raw(ov[::-1], nv),
        "reverse_hidden_cols": cos_raw(ov[:, ::-1], nv),
        "roll_r1": cos_raw(np.roll(ovv, 1, axis=0).reshape(-1, hidden), nvv.reshape(-1, hidden)),
        "roll_c1": cos_raw(np.roll(ovv, 1, axis=1).reshape(-1, hidden), nvv.reshape(-1, hidden)),
    }

    # per-grid-row / per-grid-col aggregates
    grid_rows = []
    for r in range(vision_h):
        seg = slice(r * vision_w, (r + 1) * vision_w)
        a = nv[seg]
        b = ov[seg]
        grid_rows.append({"grid_r": r, "token_first": token_start + r * vision_w,
                          "token_last": token_start + r * vision_w + vision_w - 1,
                          "mean_abs": float(adiff[vis][seg].mean()),
                          "max_abs": float(adiff[vis][seg].max()),
                          "cos": cos_raw(a, b)})
    grid_cols = []
    for c in range(vision_w):
        a = nvv[:, c, :]
        b = ovv[:, c, :]
        grid_cols.append({"grid_c": c,
                          "mean_abs": float(adiff[vis].reshape(vision_h, vision_w, hidden)[:, c, :].mean()),
                          "max_abs": float(adiff[vis].reshape(vision_h, vision_w, hidden)[:, c, :].max()),
                          "cos": cos_raw(a, b)})

    # zero / blank checks
    zero_rows_native = np.where(np.all(n == 0, axis=1))[0].tolist()
    zero_rows_oxide = np.where(np.all(o == 0, axis=1))[0].tolist()
    zero_cols_native = int(np.all(n == 0, axis=0).sum())
    zero_cols_oxide = int(np.all(o == 0, axis=0).sum())
    z_oxide_zero_native_nonzero = int(((o == 0) & (n != 0)).sum())
    z_native_zero_oxide_nonzero = int(((n == 0) & (o != 0)).sum())
    z_both = int(((n == 0) & (o == 0)).sum())

    # pooled semantics
    pooled_vs_mean = {
        "max_abs": float(np.abs(pooled - n.astype(np.float64).mean(0)).max()),
        "mean_abs": float(np.abs(pooled - n.astype(np.float64).mean(0)).mean()),
        "cos": cos_raw(pooled.astype(np.float64), n.astype(np.float64).mean(0)),
        "pooled_vs_all_row0_cos": cos_raw(pooled.astype(np.float64), n[0].astype(np.float64)),
    }

    # coverage keys (separate from numeric)
    coverage = {
        "native_keys": sorted([f"{k[0]}:{k[1]}" for k in nrows]),
        "oxide_keys": sorted([f"{k[0]}:{k[1]}" for k in orows]),
        "keys_equal": sorted([f"{k[0]}:{k[1]}" for k in nrows]) == sorted([f"{k[0]}:{k[1]}" for k in orows]),
        "native_only": sorted([f"{k[0]}:{k[1]}" for k in set(nrows) - set(orows)]),
        "oxide_only": sorted([f"{k[0]}:{k[1]}" for k in set(orows) - set(nrows)]),
        "canonical_raw_override_drops": [f"{k[0]}:{k[1]}" for k in nrows
                                         if k[1] == "pooled" and (k[0], "all") in nrows],
    }

    # dtype / ulp summary
    native_bf16_exact = float((bf16_rne(n) == n).mean())
    oxide_bf16_exact = float((bf16_rne(o) == o).mean())
    mag = np.maximum(np.abs(n.astype(np.float64)), np.abs(o.astype(np.float64)))
    ulp = np.where(mag > 0, mag * 2.0 ** -7, np.finfo(np.float64).tiny)
    ulp_units = adiff[vis] / ulp[vis]

    # pos-interp recipe scope over the actual grid (geometry only, no weights)
    row_scale = np.float32((24 - 1) / (grid[1] - 1))
    col_scale = np.float32((24 - 1) / (grid[2] - 1))
    both_nonzero = 0
    either_nonzero = 0
    degenerate = 0
    affected = [False] * (grid[1] * grid[2])
    for index in range(grid[1] * grid[2]):
        area = merge * merge
        column = index % merge + ((index // area) % (grid[2] // merge)) * merge
        row = (index // merge) % merge + ((index // (merge * grid[2])) % (grid[1] // merge)) * merge
        rf = np.float32(row) * row_scale
        cf = np.float32(column) * col_scale
        dr = rf - np.floor(rf)
        dc = cf - np.floor(cf)
        both = bool(dr > 0 and dc > 0)
        both_nonzero += both
        either_nonzero += bool(dr > 0 or dc > 0)
        degenerate += bool(dr == 0 and dc == 0)
        affected[index] = both
    merged_with_affected = 0
    for br in range(grid[1] // merge):
        for bc in range(grid[2] // merge):
            hit = False
            for i in range(merge):
                for j in range(merge):
                    index = (br * merge + i) * grid[2] + (bc * merge + j)
                    if affected[index]:
                        hit = True
            merged_with_affected += int(hit)
    pos_check_path = OWN / "pos-interp-token0.json"
    pos_check = json.loads(pos_check_path.read_text()) if pos_check_path.exists() else None
    pos_interp_evidence = {
        "geometry_scope": {
            "patch_total": grid[1] * grid[2],
            "patches_with_both_dr_dc_nonzero": both_nonzero,
            "patches_with_either_nonzero": either_nonzero,
            "degenerate_corner_patches": degenerate,
            "merged_tokens_with_at_least_one_affected_patch": merged_with_affected,
            "merged_tokens_total": vision_h * vision_w,
        },
        "token0_check": pos_check,
    }

    summary = {
        "schema": "oxide-qwen35vl-numeric-localization/v1",
        "scope": "CPU-only localization of the immutable paired Native/Oxide single-cats output; no GPU/Build/Torch.",
        "inputs": {name: pin(path) for name, path in {
            "native_whole": NATIVE, "oxide": OXIDE, "root_raw_diagnostic": ROOTRAW,
            "media": MEDIA, "controller_report": REPORT,
            "config_json": MODEL / "config.json",
            "index_json": MODEL / "model.safetensors.index.json",
        }.items()},
        "geometry": {
            "tokens": tokens, "hidden": hidden, "elements": int(n.size),
            "input_ids_len": len(media["text"]["input_ids"]),
            "vision_start_token_input_id": 151652,
            "image_token_input_id": 151655,
            "vision_end_token_input_id": 151653,
            "divergent_from_row": first_row,
            "prefix_exact_rows": prefix_exact_rows,
            "exact_prefix_rows_0_to": prefix_exact_rows - 1,
            "vision_start": token_start, "vision_end": vision_end,
            "vision_token_count": token_count,
            "grid_thw": grid, "merge": merge,
            "merged_grid_h_w": [vision_h, vision_w],
            "visual_mapping": "vis_index = (row-token_start); grid_r=vis//20; grid_c=vis%20 (row-major merged grid)",
            "suffix_first": vision_end + 1, "suffix_len": tokens - vision_end - 1,
        },
        "root_raw_echo": rootraw,
        "coverage": coverage,
        "numeric": {
            "flat_reshape_ok": bool(n.size == tokens * hidden == o.size),
            "element_mismatches": int((~eq).sum()),
            "element_mismatches_visual": int((~eq[vis]).sum()),
            "visual_elements": int(eq[vis].size),
            "exact_equal_frac_prefix": float(eq[:token_start].mean()),
            "exact_equal_frac_visual": float(eq[vis].mean()),
            "exact_equal_frac_suffix": float(eq[vision_end + 1:].mean()),
            "native_bf16_exact_frac": native_bf16_exact,
            "oxide_bf16_exact_frac": oxide_bf16_exact,
            "abs_diff_quantiles_visual": {str(q): float(np.quantile(adiff[vis], q))
                                          for q in (0.5, 0.9, 0.99, 0.999, 1.0)},
            "abs_diff_quantiles_suffix": {str(q): float(np.quantile(adiff[vision_end + 1:], q))
                                          for q in (0.5, 0.9, 0.99, 1.0)},
            "ulp_units_visual": {str(q): float(np.quantile(ulp_units, q))
                                 for q in (0.5, 0.9, 0.99, 0.999, 1.0)},
        },
        "ordering_diagnostics": {
            "row_shift_cos": shifts,
            "row_shift_argmax": max(shifts, key=shifts.get),
            "row_nn_identity_count": nn_identity,
            "row_nn_total": token_count,
            "row_nn_non_identity": non_identity,
            "hidden_col_nn_identity_count": col_identity,
            "hidden_col_total": hidden,
            "layout_cos": layout,
            "layout_best": max(layout, key=layout.get),
        },
        "grid_aggregates": {"rows": grid_rows, "cols": grid_cols},
        "blank_checks": {
            "zero_rows_native": zero_rows_native,
            "zero_rows_oxide": zero_rows_oxide,
            "zero_cols_native": zero_cols_native,
            "zero_cols_oxide": zero_cols_oxide,
            "oxide_zero_native_nonzero": z_oxide_zero_native_nonzero,
            "native_zero_oxide_nonzero": z_native_zero_oxide_nonzero,
            "both_zero": z_both,
        },
        "pooled_semantics": pooled_vs_mean,
        "pos_interp_evidence": pos_interp_evidence,
        "per_grid_row_csv": str(OWN / "per-row.csv"),
        "per_column_csv": str(OWN / "per-column.csv"),
        "top_mismatch_csv": str(OWN / "top-mismatch.csv"),
        "claims": [
            "Rows 0..14 are bit-exact; row 15 is the first divergent row and equals vision_start (input_ids[15]=151655).",
            "Row 15 is merged visual token 0 (grid_r=0,grid_c=0); its mRoPE axes are (15,15,15), identical to a text token at that index, so the only new input causing row-15 divergence is the merged visual embedding 0.",
            "mRoPE span construction matches across sources: Native media_positions (native_whole_model_policy.py:201-234) and Oxide media_position_ids (qwen35_vl_text.rs:75-164) both map image token (t,r,c) to axes (next+t, next+r, next+c), advance next by max(merged_h,merged_w,t), and give row 15 axes (15,15,15); the CPU plan confirms axes (15..29),(15..34) over the visual span and suffix resumes at 35.",
            "Reshape/flatten is correct: 349*1024=357376, row 0 bit-exact, hidden-column NN identity 1024/1024.",
            "No row shift or permutation: shift 0 dominates (cos ~0.9886 vs <=0.33 for +-1); NN identity 295/300; the 5 non-identity rows are the catastrophically divergent rows, not a permutation.",
            "No layout swap: transpose/flip/reverse all collapse to cos 0.2-0.3.",
            "No blank rows/columns in either output; the sparse exact zeros are not whole-row/column coverage holes.",
            "The coverage failure is separate: native rows are {(0,pooled),(0,all)} and oxide rows are {(0,all)}; canonical raw override drops (0,pooled).",
            "The numeric seed is inside the vision tower / image-embedding path, not the text decoder: the text-only prefix is bit-exact and the fusion is a replace/memcpy.",
            "Differences are bf16-ulp-scale and spread across the whole visual span; catastrophic tokens are high-magnitude outlier-activation tokens, consistent with numeric amplification.",
            "Source-level formula divergence found: device pos_interp kernel always uses the F16/exact four-corner grouping (two extra bf16 rounds) while BF16 Native uses corners.sum(0) (single rounding); exact flag is false for BF16 and the device tower admits BF16 only.",
            "Geometry scope of the pos_interp mismatch: 1064/1200 patches have both dr>0 and dc>0 (the regime where the two recipes differ), only 4 corner patches are degenerate, and 280/300 merged visual tokens contain at least one affected patch.",
            "Token-0 check with actual checkpoint rows 0,1,24,25: patch 3 differs in 176/768 channels at up to 14 bf16 ulp; patches 0-2 are identical because a degenerate axis leaves only two nonzero corners. Merged visual token 0 (row 15) is therefore fed a different positional signal.",
        ],
        "next_bounded_probe": {
            "kind": "ONE bounded Root host later; only after the CPU-ready fix below is applied. No build/GPU now.",
            "goal": "Confirm the pos_interp BF16 recipe is the dominant seed and bound any residual reduction-order difference.",
            "cpu_ready_fix": "Oxide device tower admits BF16 only (qwen35_vl_device.rs:425-428 rejects non-BF16), and Native exact_vision requires F16 (qwen3_vl_models.cpp:343-345), so device pos_interp must use the BF16 single-rounded four-corner sum unconditionally. Replace the three-round exact grouping in device/src/qwen35vl.rs pos_interp with one round_bf16(c00+c10+c01+c11). The Rust host reference qwen35_vl_vision.rs:331-345 already implements this BF16 branch.",
            "steps": [
                "Apply the CPU-ready device pos_interp change (suggested-pos-interp-fix.md) and add a unit test asserting the device recipe equals the host BF16 branch on the actual four table rows.",
                "Run exactly ONE paired host on single-cats with the same immutable inputs, then re-run analyze_localize.py on the new paired directory.",
                "If visual-span max_abs still exceeds tolerance, run the existing tower-only capture (tower_probe vs tower-device) once to isolate residual GEMM-plan/LayerNorm reduction-order differences; do not repeat the full paired GPU run.",
            ],
            "acceptance": "A single paired host rerun after the fix should collapse the visual-span max_abs from 29.75 toward the bf16-ulp floor; if it does not, the residual is the GEMM-plan/LayerNorm reduction order, not mapping.",
        },
    }

    (OWN / "DIAGNOSIS.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": "ok", "first_row": first_row, "prefix_exact": prefix_exact_rows,
                      "mismatches": int((~eq).sum()), "visual": int((~eq[vis]).sum()),
                      "nn_identity": nn_identity, "col_identity": col_identity,
                      "layout_best": max(layout, key=layout.get),
                      "shift_argmax": max(shifts, key=shifts.get)}, indent=2))


if __name__ == "__main__":
    main()
