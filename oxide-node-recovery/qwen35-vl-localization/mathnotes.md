# Qwen3.5-0.8B single-cats: CPU numeric localization notes

Immutable inputs (never modified):
`/tmp/oxide-qwen35vl-sdk-map-policy-output/paired-0.8B-single-v4/`
- `Qwen3.5-0.8B-single-cats/native/native-whole.json` (native-b66-whole, BF16)
- `Qwen3.5-0.8B-single-cats/oxide.json` (oxide-whole-factory, BF16, device tower)
- `Qwen3.5-0.8B-single-cats/root-raw-diagnostic.json`
- `cases/single-cats.media.json`, `report.json`

Actual checkpoint config:
`/workspace/model-performance/michaelfeil/onejev-perf/backbone-comparison-20261023/models/Qwen3.5-0.8B/config.json`
(text hidden 1024, 24 layers, 18 linear-attention + 6 full-attention, mrope_section
[11,11,10], partial_rotary_factor 0.25, vision hidden 768, depth 12, out_hidden
1024, patch 16, temporal 2, merge 2, deepstack []).

All arithmetic below is NumPy/CPU on the dumped JSON. No Torch, GPU, network,
Docker, or compile.

## 1. Geometry and flattening

- `native.tokens = 349`, `hidden = 1024`, elements `349*1024 = 357376`.
- Native rows: `(index 0, kind pooled, len 1024)` and `(index 0, kind all, len
  357376)`.
- Oxide rows: `(index 0, kind all, nested 349 x 1024)`.
- `native all` reshaped to `[349,1024]` and `oxide all` reshaped to `[349,1024]`
  have identical size; row 0 is bit-exact. Flatten order is correct.
- Fixture: `grid=[1,30,40]`, `merge=2`, `token_start=15`, `token_count=300`,
  so the visual span is rows **15..314** inclusive and the merged grid is
  **15 x 20** (row-major merged-token order).
- Token ids: `input_ids[14]=151652` (vision start), `input_ids[15..314]=151655`
  (300 image tokens), `input_ids[315]=151653` (vision end). Rows 315..348 are the
  34-token suffix.

## 2. Coverage failure is separate from numeric failure

The controller failed before numeric work because it compares row key sets:

```
native keys = {(0,"pooled"), (0,"all")}
oxide  keys = {(0,"all")}
```

`qwen35_vl_map_policy_controller_v4.py:543-546` raises
`"whole rows are empty or coverage differs"` when the sets differ. The canonical
raw-overrides-pooled rule drops `(0,"pooled")`, leaving `{(0,"all")}` on both
sides. That is a **coverage/selector fix**, not a numeric fix.

`pooled` is the mean over the 349 token rows (cos 0.999996 vs `mean(native all)`,
max abs 0.0575) and is unrelated to `all[0]` (cos 0.0568), so it must not be
compared against a token row.

## 3. Exact prefix and first divergence

Bit-level comparison (f32 bit patterns, both outputs are 100% bf16-exact):

- rows **0..14**: every one of `15*1024 = 15360` elements is bit-identical.
- row **15**: 963/1024 elements differ; this is the **first divergent row**.
- Visual span 15..314: 294037/307200 elements differ (95.7%).
- Suffix 315..348: every row differs, but small (row mean |Δ| <= 0.0695).

Row 15 is merged visual token 0, `grid_r=0, grid_c=0`. Its external mRoPE axes
are `(15,15,15)` (next=15, +row 0, +column 0), i.e. identical to a text token at
that index. Since rows 0..14 are bit-exact, the only new input at row 15 is the
merged visual embedding 0. Therefore the numeric seed is in the **vision tower /
image-embedding path**, not a general text-decoder difference and not mRoPE
phase code (text positions and row-15 position are handled by the same exact
path).

`exact prefix = rows 0..14`, `first divergence = row 15 = vision_start`.
`vision_end = row 314`; suffix begins at row 315 and differs only by propagation.

mRoPE construction matches across the two sources: Native `media_positions`
(`native/native_whole_model_policy.py:201-234`) and Oxide `media_position_ids`
(`qwen35_vl_text.rs:75-164`) both map image token `(t,r,c)` to axes
`(next+t, next+r, next+c)` and advance `next` by
`max(merged_h, merged_w, t)`. For single-cats `next=15`, merged grid 15x20, so
the visual axes are row `15..29`, column `15..34`, and the suffix resumes at
`next=35`; row 15 is `(15,15,15)`. The CPU plan
(`cpu-0.8B-single-v4/.../native-whole-plan.json`) confirms these axes exactly.
So mRoPE position construction is not the source of the divergence.

## 4. Row / grid localization

Per-grid-row aggregates over the 15x20 merged grid (see `per-row.csv`):

- mean |Δ| rises with grid row: r0 0.086, r3 0.305, r5 0.338, r7 0.457,
  r12 0.467, r13 0.401, r14 0.483; corr(grid_r, row mean|Δ|) = 0.186.
- This is sequence-position accumulation, not a spatial layout error.

Catastrophic tokens (row: grid (r,c), max|Δ|, cos):

| row | vis | grid | max abs | cos |
|----|----|------|---------|-----|
| 165 | 150 | (7,10) | 15.77 | 0.5190 |
| 272 | 257 | (12,17) | 11.78 | 0.7282 |
| 309 | 294 | (14,14) | **29.75** | 0.4079 |
| 300 | 285 | (14,5) | 11.31 | 0.8161 |
| 79  | 64  | (3,4)  | 11.31 | 0.9159 |
| 278 | 263 | (13,3) | 10.38 | 0.8474 |
| 132 | 117 | (5,17) | 10.00 | 0.8508 |
| 172 | 157 | (7,17) | 9.09  | 0.8520 |
| 283 | 268 | (13,8) | 8.88  | 0.8770 |
| 176 | 161 | (8,1)  | 8.03  | 0.8878 |

The global max is row **309, col 415**: native `29.75`, oxide `0.0`, cos 0.408.
These tokens are high-magnitude outlier-activation tokens; the local recipe
difference is amplified there (see §6/§7).

## 5. Misindex diagnostics (all negative)

- **Row shift**: visual-span aggregate cosine is
  0.9886 at shift 0 vs 0.3258 (+1), 0.3258 (-1), <=0.30 for |shift|>=2.
  Full-sequence shift 0 also dominates (0.9899 vs ~0.35).
- **Row nearest-neighbor**: oxide visual row i best matches native row i for
  **295/300** rows. The 5 non-identity rows are the catastrophic rows
  (vis 117, 150, 263, 285, 294); their best matches are low-cosine (0.71-0.93)
  and do not form a permutation.
- **Hidden-column nearest-neighbor**: **1024/1024** oxide columns best match the
  same native column; identity correlation min 0.956, median 0.986.
- **Layout hypotheses** on the 15x20 grid: identity 0.9886; transpose 0.2132;
  flip rows 0.2910; flip cols 0.2454; reverse flat 0.2073; reverse hidden cols
  0.0083; roll r1 0.3214; roll c1 0.3248.
- **Intra-2x2 merged-block reorder**: all 24 permutations score 0.229-0.316,
  far below identity.

Conclusion: no shift, no row permutation, no transpose/reversal, no hidden-dim
permutation. The visual token order and hidden layout are canonical.

## 6. Column / outlier analysis

- Differences are bf16-ulp scale: outputs are 100% bf16-exact. Visual-span
  |Δ| quantiles: p50 0.125, p90 0.523, p99 3.01, p99.9 6.89, max 29.75.
  In units of one bf16 ulp of the local magnitude: p50 6.1, p90 47.0,
  p99 184, max 256.
- Per-column mean |Δ| correlates with native column magnitude
  (corr 0.691), i.e. the error is roughly relative, not a fixed channel offset.
- Repeated per-row argmax-diff columns: 150 (20 rows), 415 (15), 3 (8), 829 (7),
  152 (6), 283 (5), 897/624/196/16 (4). Top channels by mean |Δ| / native absmax:
  415 (mean 0.642, absmax 35.5), 16 (0.541, 23.4), 150 (0.521, 32.0),
  829 (0.485, 23.9), 624 (0.483, 16.4), 111 (0.458, 19.3), 3 (0.436, 29.1),
  283 (0.428, 20.5), 196 (0.426, 16.8), 39 (0.417, 15.7).
- These are the high-magnitude / outlier-activation channels. The pattern
  (relative error + rare catastrophic tokens) is numeric amplification of a
  small perturbation, not a channel permutation or a masked value.

## 7. Blank / zero checks

- Zero rows: native **0**, oxide **0**.
- Zero columns: native **0**, oxide **0**.
- Sparse exact zeros: oxide zero/native nonzero 250 elements, native zero/oxide
  nonzero 271, both zero 27 (whole tensor). These are bf16 zeros inside
  activations, not blank rows or coverage holes.
- No row/column is empty; the controller's "empty" wording is the coverage-key
  mismatch in §2, not a data hole.

## 8. Source-level numeric seed: BF16 position-interpolation recipe

Source comparison (see `source-worker-vision-ops.md` for full citations) found
exactly one formula-level divergence in the vision tower:

- Native BF16 (`exact_vision=false` because `exact_vision` requires F16;
  `multi_vector=false`) computes four-corner positional embeddings with
  `corners.sum(0)`: one float32 accumulation, **single** bf16 rounding
  (`qwen3_vl_models.cpp:143-153`).
- Oxide device `pos_interp` unconditionally uses the F16/exact grouping
  `round(round(c00+c10) + round(c01+c11))`: **three** bf16 roundings
  (`device/src/qwen35vl.rs:137-139`).
- The Oxide device tower admits BF16 only and rejects F16
  (`qwen35_vl_device.rs:425-428`), so the `exact` flag at `:445` is always false;
  the kernel hardcodes the wrong recipe. The host reference already implements
  the BF16 single-round branch (`qwen35_vl_vision.rs:331-345`), so device and
  host disagree internally.

Geometry scope of the mismatch: 1196/1200 patches are non-degenerate; 1064/1200
have **both** `dr>0` and `dc>0` (the regime where the recipes can differ); only
4 corner patches are exact. 280/300 merged visual tokens contain at least one
affected patch.

Direct token-0 check with the actual checkpoint (only 4 `pos_embed.weight` rows
read, 6144 bytes; `pos-interp-token0.json`):

- patch 0 `(0,0)`: dr=dc=0 -> identical;
- patch 1 `(0,1)`: dr=0 (only two nonzero corners) -> identical;
- patch 2 `(1,0)`: dc=0 (only two nonzero corners) -> identical;
- patch 3 `(1,1)`: four nonzero corners -> **176/768 channels differ**, max
  0.0078125, max 14.2 bf16 ulp, p99 1.05 ulp.

So merged visual token 0 (row 15) already gets a different positional signal,
and interior merged tokens (all four patches non-degenerate) get a broader one.
This is sufficient to seed the observed ~1-ulp drift and the outlier
amplification.

Residual (not formula-level, still possible): the vision GEMM plan (generic
`prepare_bias_shape` vs Native `at::linear` at M=1200) and the hand-rolled
LayerNorm reduction order vs torch Welford. See
`source-worker-vision-ops.md` §8.

## 9. Why text is exact and image is not

- The prefix rows 0..14 are bit-exact under the same text decoder (18 linear-
  attention + 6 full-attention layers) and the same text mRoPE kernel, so the
  text path is bit-identical.
- Image fusion is a replace/copy of bf16 embedding rows on both sides
  (`qwen35_vl_engine.rs:151-184`; Native `copy_`), not an add/scale/norm.
- Therefore the divergence originates in the merged vision features (recipe
  mismatch above), and the suffix differs only because it attends to the
  already-diverged image tokens.

## 10. Deliverables and next probe

Files in this directory:
- `DIAGNOSIS.json` - machine-readable findings, pins, claims, next probe.
- `per-row.csv` - 349 rows: region, grid coords, mismatch count, mean/max |Δ|,
  argmax column, cosine, norms.
- `per-column.csv` - 1024 hidden columns over the visual span.
- `top-mismatch.csv` - top 64 elements with native/oxide values.
- `pos-interp-token0.json`, `pos_interp_check.py` - token-0 recipe check.
- `source-worker-vision-ops.md` - read-only source operator comparison.
- `suggested-pos-interp-fix.md` - exact CPU-ready device kernel change.
- `pins.json` - input and source pins.

Next probe (one bounded Root host, only after the CPU-ready fix):
apply the device `pos_interp` BF16 single-round change, run exactly one paired
host on `single-cats`, and re-run `analyze_localize.py`. If residual max|Δ| stays
above tolerance, do one tower-only capture (Native `tower_probe` pinned to
`bfloat16` vs Oxide `tower-device`) to isolate GEMM-plan/LayerNorm order. Do not
repeat the full paired GPU run.
