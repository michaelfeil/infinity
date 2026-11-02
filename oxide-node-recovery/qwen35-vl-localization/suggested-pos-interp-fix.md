# Suggested CPU-ready fix: BF16 four-corner position interpolation in the Oxide device tower

Status: **suggestion only, not applied** (source trees are read-only for this worker).
Owner: Root / source owner. No build was performed here.

## Why

The paired run is BF16 on both sides:

- Native whole-model helper hardcodes `tei_create(..., BF16)`
  (`/tmp/oxide-qwen35vl-sdk-map-policy-v1/native/native_whole_model_policy.py:159`,
  `BF16 = 2` at `:19`).
- Native Qwen3.5 on CUDA rejects non-BF16
  (`.../qwen35_models.cpp:1186`), and `exact_vision` requires `at::kHalf`
  (`.../qwen3_vl_models.cpp:342-350`). With BF16, `exact_vision` is **false**.
- Therefore Native takes the `corners.sum(0)` branch
  (`.../qwen3_vl_models.cpp:148-153`): four BF16 corner products are accumulated in
  float32 with **one** final BF16 rounding.
- The Oxide device tower admits BF16 only and rejects F16
  (`libs/infinity-v3/backends/oxide/src/qwen35_vl_device.rs:425-428`), so its
  `exact` flag at `:445` is always false.
- But the device kernel `pos_interp` unconditionally applies the **exact/F16**
  grouping with two extra BF16 rounds
  (`libs/infinity-v3/backends/oxide/device/src/qwen35vl.rs:137-139` = Native
  `exact_vision` branch `(c00+c10)+(c01+c11)`).
- The Oxide host reference already distinguishes the two recipes
  (`libs/infinity-v3/backends/oxide/src/qwen35_vl_vision.rs:331-345`), so the
  device kernel is internally inconsistent with its own BF16 reference.

This is the only formula-level (not merely reduction-order) divergence found in
the vision tower. It affects 1064/1200 patches (both `dr>0` and `dc>0`) and
280/300 merged visual tokens; only the 4 corner patches are degenerate.

## Exact change

`libs/infinity-v3/backends/oxide/device/src/qwen35vl.rs`, inside
`pub unsafe fn pos_interp` (currently lines 137-139):

```diff
-    let first = rounded(corner(r0, c0, (1.0 - dr) * (1.0 - dc)) + corner(r1, c0, dr * (1.0 - dc)));
-    let second = rounded(corner(r0, c1, (1.0 - dr) * dc) + corner(r1, c1, dr * dc));
-    let positional = rounded(first + second);
+    // Device vision is BF16-only; Native BF16 uses corners.sum(0): one f32
+    // accumulation of the four bf16 corner products and a single bf16 cast.
+    let positional = rounded(
+        corner(r0, c0, (1.0 - dr) * (1.0 - dc))
+            + corner(r0, c1, (1.0 - dr) * dc)
+            + corner(r1, c0, dr * (1.0 - dc))
+            + corner(r1, c1, dr * dc),
+    );
```

This exactly matches the host reference `else` branch at
`qwen35_vl_vision.rs:337-345` (`round_bf16(v0 + v1 + v2 + v3)`).

If F16 device support is ever added, gate the grouping on `config.text.dtype`
instead of hardcoding; until then the device path is BF16-only.

## CPU verification already done (this worker)

`pos_interp_check.py` reads only the four `pos_embed.weight` rows needed by merged
visual token 0 (rows 0, 1, 24, 25; 6144 bytes) and applies both recipes:

- patch 0 `(0,0)`, patch 1 `(0,1)`, patch 2 `(1,0)` are identical (degenerate
  axis leaves only two nonzero corners);
- patch 3 `(1,1)` differs in **176/768** channels, max `0.0078125`, up to
  **14.2 bf16 ulp**, p99 1.05 ulp (`pos-interp-token0.json`).

So merged token 0 (text row 15, the first divergent row) is fed a different
positional signal even before any GEMM/LN difference.

## One bounded next Root probe

1. Apply this change + a unit test that the device recipe equals the host BF16
   branch on real table rows (CPU-only).
2. Run exactly **one** paired host on `single-cats` with the same immutable
   inputs, then re-run `analyze_localize.py` on the new paired directory.
3. If the visual-span `max_abs` does not fall to the bf16-ulp floor, run the
   existing tower-only capture once (Native `tower_probe.cpp` pinned to
   `bfloat16`, Oxide `tower-device`) to isolate residual GEMM-plan / LayerNorm
   reduction-order differences. Do not repeat the full paired GPU run.

Acceptance: row 15..314 max|Δ| should collapse from 29.75 toward the bf16-ulp
floor and row-15 should no longer be the first catastrophic row.
