# Qwen3.5-0.8B VL: vision-tower numeric localization (Oxide vs frozen Native)

Read-only source comparison. No GPU/build/torch was run; the only arithmetic quoted is a
pure-Python bf16 simulation of the position-interpolation recipe (see §2/§9).

Roots:
- OX = `/tmp/oxide-qwen35vl-sdk-map-policy-v1/libs/infinity-v3/backends/oxide`
- NT = `/tmp/oxide-all-model-feature-matrix-v1/qwen35-vl-lifecycle-fix-v1/libs/infinity-v3/backends/libtorch/cpp`

Pairing assumption (important, and verified from source): the compared run is **BF16** on both
sides. Native hard-requires BF16 for Qwen3.5 on CUDA (`NT/models/qwen35_models.cpp:1186`), and
`exact_vision` is `device.is_cuda() && dtype == at::kHalf && model_type.startswith("qwen3_5") &&
kernel==[H,3,2,16,16]` (`NT/models/qwen3_vl_models.cpp:342-350`). Therefore **every F16/"exact"
Native branch below is dead in this pairing**, and the Native tower being compared is:
flat patch GEMM (`:117-121`), `corners.sum(0)` positions (`:148-153`), `at::layer_norm` (`:41-47`),
`at::gelu(tanh)` (`:383-398`), FA4 attention (`:250-264`).
Oxide device is BF16-only and rejects F16 outright (`OX/src/qwen35_vl_device.rs:425-429`), so its
own `exact` flag (`:445`) is always false and the two-temporal-slice path is dead there too.

---

## 1. Patch / temporal projection — verdict: **MATCH** (in the BF16 pairing)

| | Native (BF16) | Oxide device (BF16) |
|---|---|---|
| op | one flattened `at::linear(pixels[rows,1536], W[768,1536], bias)` `NT:117-121` | one LtPlan GEMM `OX/src/qwen35_vl_device.rs:779-791` |
| f32 accumulation, single bf16 round + fused bias epilogue | `at::linear` bf16 → cuBLASLt bias epilogue | `Epilogue::Bias`, `beta=0` `OX/src/lt.rs:748-755,1201-1290` |
| pixels cast | `.to(device,dtype)` (RNE) `NT:118` | `infinity_f32_to_bf16` = `cvt_bf16x2_f32` (RNE) `OX/device/src/dense.rs:17-28` |
| temporal slices | folded into K=1536 | folded into K=1536 (`patch_flat`) |

The two-GEMM temporal split exists on both sides but is unreachable here:
`NT:100-106` (guarded by `exact_vision`) and `OX/src/qwen35_vl_vision.rs:544-581` (host reference,
`exact` at `:490-491`), plus the device de-interleaved `patch_t0/patch_t1` (`OX/src/qwen35_vl_device.rs:446-459`,
`OX/device/src/qwen35vl.rs:206-231`) which can never be `Some` because `new()` rejects F16.
**The device module doc comment `OX/src/qwen35_vl_device.rs:9-14` ("two GEMMs run, their BF16 sums are
added") describes dead code and is misleading.**

t=1 handling: neither side duplicates or zeros a second temporal frame. With
`temporal_patch_size=2` the packed row already carries `C*2*P*P` elements (the preprocessor puts the
still frame in both temporal slots), so the K=1536 GEMM legitimately sees the frame twice. Oxide's
`temporal_slice_bf16` (`OX/device/src/qwen35vl.rs:206-231`) indexes `(c*T+t)*P*P` which is the same
layout Native's `.view({-1,3,2,16,16})` produces, but it is unused. No padding, no zero-fill.

Residual risk: `at::linear(x,W,b)` vs cuBLASLt `Epilogue::Bias` can differ if torch's addmm does not
select the bias epilogue (then it would be `round(round(gemm)+bias)` — two roundings). Provider-level,
see §8 hypothesis 2.

## 2. Learned position interpolation — verdict: **DIFFERS** (the one real recipe divergence)

Everything except the corner-summation grouping matches:

- coordinates: `NT:129-132` `c = i%merge + ((i/merge²)%(W/merge))*merge`,
  `r = (i/merge)%merge + ((i/(merge*W))%(H/merge))*merge` ≡
  `OX/src/qwen35_vl_media.rs:137-147` and the in-kernel copy `OX/device/src/qwen35vl.rs:110-113`.
  Row↔row, column↔column; no transposition.
- scales: `(side-1)/(H-1)`, `(side-1)/(W-1)`, forced 0 for a 1-wide axis — identical
  (`NT:133-134` vs `OX/device/src/qwen35vl.rs:114-123`).
- **scale is rounded to bf16 before the multiply** on both sides: `scale.to(dtype)` `NT:140` vs
  `rounded(scale)` then `rounded(table*scale)` `OX/device/src/qwen35vl.rs:132-136`. Match.
- `states = states + positional` in model dtype (`NT:154` vs `OX/device/src/qwen35vl.rs:141-143`). Match.

**Divergence — corner summation order / rounding depth.**
Native BF16 (`multi_vector=false`, `exact_vision=false`) reduces the four bf16 corners with
`corners.sum(0)` (`NT:143-153`), i.e. one torch reduction: f32 accumulator, **single** final bf16
cast. Oxide's device kernel unconditionally uses the *exact/F16* grouping with **two extra bf16
rounds**:

```
OX/device/src/qwen35vl.rs:137  first  = rounded(corner(r0,c0,..) + corner(r1,c0,..))   // rb(c00+c10)
OX/device/src/qwen35vl.rs:138  second = rounded(corner(r0,c1,..) + corner(r1,c1,..))   // rb(c01+c11)
OX/device/src/qwen35vl.rs:139  positional = rounded(first + second)                   // rb(first+second)
```

which is precisely `NT:151-153`, the **exact_vision branch** `(c0+c2)+(c1+c3)`, not the BF16 branch.
The host reference is aware of the dtype distinction (`OX/src/qwen35_vl_vision.rs:331-345`: `exact`
grouping vs single `round_bf16(v0+v1+v2+v3)`), but the device kernel hard-codes the exact grouping
regardless of dtype — an internal inconsistency inside Oxide and the only place where the *formula*
(not just reduction order) differs.

Quantified with a pure-Python RNE-bf16 simulation of the real geometry (H=30,W=40,side=24,
table ~ N(0,0.03)): **28.4 %** of the 1200×768 positional entries differ; ~1.2 % differ by >4 ulp of
the result and the worst cases reach ~1.9e3 ulp (near-cancellation between the two column pairs).
Even if Native's reduce used bf16 rather than f32 accumulation, 30 % still differ, so the grouping
alone is sufficient.

Note: a small absolute perturbation (~1 bf16 ulp of the corner magnitude, i.e. ~1e-4) survives the
later `rb(state + positional)` with probability ≈ `δ/ulp(state)`, so this seeds a few thousand
1-ulp flips in the very first tower activation — exactly the observed drift scale, and it is present
for **every** token from the first op onward.

## 3. Vision rotary (split-half 2-D) — verdict: **MATCH**

- frequency `f_j = 1/10000^(2j/(dim/2))`, `j < dim/4`: `NT:352-358` ≡
  `OX/src/qwen35_vl_vision.rs:352-361` ≡ `OX/device/src/qwen35vl.rs:178-180`.
- layout: Native `angles = cat([r*freq, c*freq])` (`NT:156-158`) means phase `i < half/2` is
  the **row** axis, `i >= half/2` the **column** axis; pair is `(i, i+half)` (`NT:174-177`).
  Oxide: `phase < quarter ? row : column`, pair `(phase, phase+half)`
  (`OX/device/src/qwen35vl.rs:178-189`, host `OX/src/qwen35_vl_vision.rs:385-396`). Identical.
- rounding depth: Native casts the operands to f32, computes `a*cos-b*sin` in f32 from **f32**
  `cos/sin` (not bf16-rounded) and rounds **once** when `.to(dtype)` at `NT:194-198`.
  Oxide also multiplies f32 and rounds once (`OX/device/src/qwen35vl.rs:193-200`). Match.
  (Contrast: the *text* rope deliberately rounds each product, `OX/device/src/qwen35_attention.rs:178-184`
  — that is the correct match for the text-side Native recipe, not for vision.)
- only residual: `10_000f32.powf` (device libdevice, ≤2 ulp) vs host `std::pow` float (`NT:354`).
  Angle ≤ 39·f, so ≤ ~1e-6 relative on cos/sin → ~1e-5 chance of a bf16 flip per element. Negligible
  but not zero.

## 4. Vision attention — verdict: **MATCH** (same provider, same math)

- per-frame, non-causal, no cross-frame: Native builds `offsets` = `[0, H*W, 2*H*W, ...]` per frame
  (`NT:162-167`) and calls with `max_q=max_k=max_frame`, `causal=false` (`NT:250-264`). Oxide builds
  `[0, frame_len, ...]` (`OX/src/qwen35_vl_device.rs:818-831`) with `batch = grid.temporal`,
  `max_q = H*W`, `causal:false`, `Window::Global` (`:474-482`, `:904-920`).
- **Both land in `infinity_fa4`**: Native `packed_cuda_attention` → `infinity_fa4(...)` first
  (`NT/helpers/attention.h:142-160`), Oxide `Fa4::forward_batch_bound`. Same library, same
  (12 heads, 12 kv, head_dim 64, bf16, non-causal) → same central mode (asserted in
  `OX/src/qwen35_vl_tests.rs:500`).
- scale: `1/sqrt(64) = 0.125` exactly on both sides (`NT:235,257` vs `Scale::RsqrtHeadDim`,
  `OX/src/fa4.rs:646-651`) — no rounding difference.
- no window/padding/mask on either side.
- Only structural difference: Native hands FA4 three *separate contiguous* bf16 tensors, Oxide hands
  strided views inside one packed `[rows,3*hidden]` buffer (`OX/src/qwen35_vl_device.rs:832-847`).
  Strides do not change the per-row K-loop, so this is expected to be bit-identical; it is the one
  thing here I could not verify without running.

## 5. Merger — verdict: **MATCH** (grouping/normalization order), activation/GEMM caveats

- grouping: Native `x.view({-1, hidden*merge*merge})` (`NT:55`) = 4 consecutive merge-major patch
  rows concatenated in full; Oxide normalizes each patch row then feeds the contiguous
  `[merged, 3072]` slab straight to the 3072-wide GEMM (`OX/src/qwen35_vl_device.rs:965-983`;
  host identity `grouped_view` `OX/src/qwen35_vl_vision.rs:456-471`). Same permutation.
- normalization point: Native final merger is called with `post=false` (`NT:279`), so it is
  `ln(x, merger.norm)` **per patch row, width 768, before grouping** (`NT:56`);
  Oxide: `logical rows = rows, width = hidden` before the GEMM (`:967-974`). Match.
  (Deepstack mergers use `post=true` — grouped width 3072 — but `deepstack_visual_indexes: []`.)
- activation: GELU-tanh on both sides (`NT:366-372` for non-exact → `at::gelu(tanh)`;
  `OX/src/qwen35_vl_device.rs:984` → `gelu_tanh_bf16`). Formula identical, constant differs
  (§8 hypothesis 4).
- final linear 3072→1024 with bias: `NT:58` (`at::linear` fused bias) vs `OX:985-993`
  (`prepare_bias_shape(grouped, out_hidden)` + `Epilogue::Bias`). Same recipe class.
- latent bug (not triggered by `gelu_pytorch_tanh`, `OX/src/qwen35_vl_config.rs:30`): the host
  reference's `"gelu"` branch is an **identity** (`OX/src/qwen35_vl_vision.rs:654-658`) where Native
  uses exact erf GELU (`NT:392`). If any root ever sets `hidden_act: "gelu"`, Oxide is outright
  wrong (missing erf), not just differently rounded.

## 6. Explicit BF16 rounding points / 0.0-vs-large

Rounding points that agree:
- every GEMM output: one bf16 round, f32 accumulation (`OX/src/lt.rs` bias epilogue; Native cublas).
- residual adds: `rb(bf16(a)+bf16(b))` `OX/src/qwen35_vl_vision.rs:281-290`,
  `OX/device/src/qwen35vl.rs:31-41` ≡ bf16 `states + projected` (`NT:267,270`).
- LN output: `(x-mean)*inv*gamma+beta` in f32, one bf16 round (`OX/device/src/main.rs:260-273`,
  `OX/src/qwen35_vl_vision.rs:271-276`) ≡ torch LN.
- GELU/SiLU output: one bf16 round (`OX/device/src/qwen35vl.rs:60-84`).

Rounding points that **differ**:
1. position corners: 2 extra bf16 rounds (§2).
2. LayerNorm statistics: Oxide is a hand-rolled 256-lane two-pass sum with warp-shuffle tree
   (`OX/device/src/main.rs:75-108,219-273`); Native is `at::layer_norm` (Welford/vectorized)
   `NT:41-47`. Both f32, different summation order/tree → ~1e-7 relative; flips a bf16 output
   for ~1e-5 of elements per norm (≈ hundreds per layer over 1200×768).
3. GEMM provider/plan: Oxide vision plans come from the **generic** `prepare_bias_shape`
   (`OX/src/qwen35_vl_device.rs:465-472,500-508`; `OX/src/lt.rs:748-755`), i.e. heuristic chosen at
   M=128 with `MAX_WORKSPACE_BYTES = 0` and reused for M=1200. Native `at::linear` chooses its
   cuBLASLt algorithm for the actual M with a 32 MB workspace. The codebase has an explicit
   "native-default" plan for exactly this purpose (`OX/src/lt.rs:951-971`,
   `OX/src/biased_linear.rs:225`) — the vision tower does not use it. Different algo ⇒ different
   f32 split/accumulation order ⇒ scattered 1-ulp bf16 flips in every projection.
4. GELU-tanh constant: `0.797_884_6` (`OX/device/src/qwen35vl.rs:67`) vs
   `0.7978845608028654` (`NT/helpers/dense_mlp.cpp:35`); tanh via `libm::tanhf` vs torch's
   `compat::tanh`. ~5e-8 relative → rare boundary flips.
5. rotary frequency computed on device (`powf`) vs host (`std::pow`) — §3.

Exact-0.0-vs-large: I found **no code path that zeroes an output**. `zero_patch_bias` (`OX/src/qwen35_vl_device.rs:197,453`)
is filled with bf16 zeros but only feeds the dead F16 two-GEMM path; `ws.branch`/`ws.attention`
are `memset` at allocation (`:105-121`) and always overwritten. The only "0.0 vs large" mechanism is
relative: corner cancellation in §2 can return a value differing by ~10³ ulp from Native's near-zero
value (absolute error still ~1 bf16 ulp of the corner). Large *absolute* outliers are more likely a
consequence, not a cause: a 1-ulp flip in q/k before softmax changes an attention probability by
~0.4 %, and where two logits are nearly tied the attention mixture changes macroscopically → the
"few catastrophic tokens".

## 7. Image-embedding fusion — verdict: **MATCH**

- Native: `at::embedding(...)`, then `states.narrow(0, token_start, token_count).copy_(features)`
  (`NT/models/qwen35_models.cpp:864-886`, `features = batch[i].to(dtype)`); the Qwen3-VL generic path
  is `h.index_copy_(0, indices, visual)` (`NT/models/decoder_models.cpp:1253-1255`). **Replace, no
  add, no scale, no layernorm.**
- Oxide: `memcpy_dtod_async` of bf16 `image_merged` over the gathered embedding rows
  (`OX/src/qwen35_vl_engine.rs:151-184`); host reference `fuse_visual_rows` is a
  `copy_from_slice` (`OX/src/qwen35_vl_text.rs:288-325`). Same replace semantics, same bf16 payload
  (Native's `.to(dtype)` is a no-op at bf16).
- Alternative seed ruled out: the per-token mRoPE phases are *not* the explanation. Native rounds
  `cos/sin` to bf16 (`NT/models/qwen35_models.cpp:441-448`) and Oxide's text rope kernel does the
  same (`OX/device/src/qwen35_attention.rs:170-184, 276-277`); since tokens 0..14 are bit-exact under
  the very same kernel, the phase code path is identical and only the *features* differ at token 15.

---

## 8. Ranked numeric-seed operators

| # | Operator | Why it produces ~1-ulp drift | Outlier amplification | Evidence |
|---|---|---|---|---|
| 1 | **Position-interpolation corner grouping** (device uses Native's F16/exact grouping with 2 extra bf16 rounds; Native BF16 uses one f32-accumulated `sum(0)`) | deterministic, present at the first tower op, 28 % of positional entries already differ, ~1.2 % by >4 ulp | near-cancellation ⇒ 10³-ulp relative errors in isolated (token, hidden) entries | `OX/device/src/qwen35vl.rs:137-139` vs `NT/models/qwen3_vl_models.cpp:143-153`; dead `exact` at `OX/src/qwen35_vl_device.rs:425-429,445`; host inconsistency `OX/src/qwen35_vl_vision.rs:331-345` |
| 2 | **GEMM plan/heuristic mismatch** (generic LtPlan vs torch `at::linear`) | different cuBLASLt algo for M=1200 ⇒ different f32 accumulation order in patch/qkv/proj/mlp/merger; affects every element of every projection | 1-ulp flips in q/k feed softmax; small prob changes amplified by 12 layers | `OX/src/qwen35_vl_device.rs:465-472,500-508`; `OX/src/lt.rs:748-755,951-971`; `NT/models/qwen3_vl_models.cpp:117-121,266-270` |
| 3 | **LayerNorm statistics reduction** (device warp-tree two-pass vs torch Welford) | different f32 sum tree ⇒ ~1e-7 rel; rare bf16 flips in every normed row | a flipped normed element shifts a whole attention row slightly | `OX/device/src/main.rs:75-108,219-273` vs `NT/models/qwen3_vl_models.cpp:41-47` |
| 4 | **GELU-tanh constant/tanh impl** | `0.7978846` vs `0.7978845608…`; `libm::tanhf` vs torch tanh; ×12 layers ×2 MLPs | only boundary flips, 1 ulp | `OX/device/src/qwen35vl.rs:60-71` vs `NT/helpers/dense_mlp.cpp:35` |
| 5 | Vision rotary frequency `powf` vs `std::pow` | ≤2 ulp f32 on f, angle ≤39·f ⇒ ~1e-6 rel on cos/sin | negligible | `OX/device/src/qwen35vl.rs:180` vs `NT/models/qwen3_vl_models.cpp:352-358` |

Explicitly **not** the seed: patch/temporal projection (§1, same single 1536-wide GEMM), rotary
layout/rounding (§3), attention (§4, same FA4 call), merger grouping/normalization order (§5),
fusion (§7).

Nothing in the vision tower is *outright wrong* for this checkpoint, with two caveats that are wrong
but inactive: the host `hidden_act == "gelu"` identity (`OX/src/qwen35_vl_vision.rs:654-658`) and the
described-but-dead two-GEMM temporal path (`OX/src/qwen35_vl_device.rs:9-14,446-459`). The
position-interpolation grouping is a **recipe-selection bug** (F16 recipe applied to a BF16 run),
not a typo.

## 9. Single best CPU-verifiable hypothesis and the exact quantity to compare

**Hypothesis.** The merged-feature divergence is seeded by the learned-position interpolation: Oxide's
device kernel (`OX/device/src/qwen35vl.rs:137-139`) rounds `(c00+c10)` and `(c01+c11)` to BF16 before
the final add, while the Native scanner under test reduces the four corners once in f32
(`NT/models/qwen3_vl_models.cpp:143-153`, BF16 branch). Everything else in the tower is either
identical or provider-level.

**Exact quantity to compare (CPU-only, no GPU, no torch, no checkpoint execution).** For
`grid_thw=[1,30,40]`, `merge=2`, `side=sqrt(num_position_embeddings)`, take the checkpoint's real
`model.visual.pos_embed.weight` `[side²,768]` bf16 table and produce the `[1200,768]` **positional
matrix** (per-frame merge-major order) twice:

- A (Native BF16): `corner = bf16(t_bf16 * bf16(scale))`;
  `pos = bf16( ((c00+c01)+c10)+c11 )` with the four-term sum accumulated in f32, one final cast.
- B (Oxide device): `pos = bf16( bf16(c00+c10) + bf16(c01+c11) )`.

Report, per token and hidden index: number of differing bf16 bit patterns, max |Δ| in bf16 ulps, and
the full `[1200,768]` A vs B byte diff. Expected from the recipe alone (synthetic-table estimate):
~28 % of entries differ, ~1 % by more than 4 ulp, worst ~10³ ulp.

Then confirm the amplification ladder on the CPU reference path, which reproduces the same recipe:
run `vision_forward` (`OX/src/qwen35_vl_vision.rs:473`) with `exact=false` (its `:337-345` branch =
recipe A) against a build with `exact=true` (its `:331-336` branch = recipe B) and diff
(a) states right after `states + positional` (`:595-598`), (b) `blocks.0.norm1` input, (c) the final
`merger` rows. If (a) already shows differing bit patterns beyond recipe B's own choice, hypothesis 2
(GEMM plan) must be added; if (a) matches the Native dump but the merged rows do not, the Ladder
moves to hypothesis 2/3.

Pre-flight check that should be done first (cheap, rules out a whole class of noise): confirm the
Native probe (`NT/../native/tower_probe.cpp:69-74`) was invoked with dtype `bfloat16`, not
`float16`. If it was `float16`, Native ran the entire `exact_vision` recipe in FP16 storage
(`NT/models/qwen3_vl_models.cpp:342-350`) while Oxide can only run BF16 — that mismatch alone would
produce exactly this 1-10-bf16-ulp drift with outliers and would supersede all of §8.
