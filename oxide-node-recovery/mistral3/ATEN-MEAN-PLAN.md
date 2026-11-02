# Mistral3 Native mean: ATen 5c488 reduction investigation (V6 WIP)

Status: **NOT bit-exact qualified**. The V5 leaf comment claiming an exact
ATen order is withdrawn. No Torch/GPU run or device compile was performed.

## Native operation

Native Mistral3 (`mistral3_models.cpp:401-408`) sets
`model_dtype_mean_pool=true` and computes:

```cpp
(value.sum(0, true) / value.size(0)).to(at::kFloat)
```

for a hidden-state tensor shaped `[sequence_tokens, hidden]`. This is an
explicit BF16 `sum` reduction over dim 0, followed by BF16 division, then a
widening cast. It is not `torch.mean`.

## Pinned primary source (V7)

`/tmp/oxide-mistral3-aten-5c488-mean-source-v1/` now contains the actual pinned
`Reduce.cuh`, `ReduceSumProdKernel.cu`, `Reduce.cu`, `BinaryDivTrueKernel.cu`,
`BinaryMulKernel.cu`, `BFloat16.h`, `LICENSE` and provenance. The unsanctioned
derivation is `/tmp/oxide-mistral3-aten-mean-analysis-v1/{REPORT.md,
reference_vectors.py,mistral3_model_mean_aten.rs}`. It confirms sequential row
order first diverges at T=5, uses four interleaved F32 accumulators, a block-y
warp tree at T>=64, and device-dependent CTA/global splitting at T>=1021.

The true-divide leaf was also resolved: `BinaryDivTrueKernel.cu` takes the
CPU-scalar `a * reciprocal(b)` path, with the reciprocal computed in double and
cast to the opmath F32 type. The V7 candidate now uses that multiply recipe, not
`sum / T`. This does not qualify the sum order.

## Pinned PyTorch source available locally

`/tmp/oxide-fa4-build-env/lib/python3.12/site-packages/torch/version.py`
reports `git_version = 5c4886908584029761b579af026dcfb627c84070` and version
`2.14.1+cu130`. Its installed headers include:

- `torch/include/ATen/native/cuda/Reduce.cuh`
- `torch/include/ATen/native/SharedReduceOps.h`

No complete `ReduceSumProdKernel.cu`/`ReduceOps.cpp` source is present in the
local packet, so the sum launch is inspected through the installed 5c488
headers and PyTorch source paths only.

## Why the V5 sequential leaf is not provably identical

For a contiguous `[S, H]` BF16 value reduced over dim 0 (output `[H]`), the
TensorIterator reduction strides are non-fastest along the reduction axis.
`setReduceConfig` therefore maps output features to block.x and reduction rows
to block.y (`Reduce.cuh` around `setReduceConfig`). The exact path depends on
runtime `S`, `H`, warp size, `multiProcessorCount`, and device limits:

- `dim0 = num_outputs` (H) and `dim1 = inputs_per_output` (S).
- `block_width`/`block_height` are derived from `last_pow2` and the 512-thread
  cap; output vectorization may reduce `dim0` first.
- `warp_split_threshold = min(block_height * 16, 256)`.
- If `values_per_thread() < warp_split_threshold`, each output lane sums all S
  rows sequentially, and a per-feature sequential F32 reduction is the same
  order.
- If `values_per_thread() >= warp_split_threshold`, `input_mult[BLOCK_Y]`
  splits the reduction across warp rows and the result is combined through
  shared-memory `block_y_reduce` (and possibly CTA/global reduction). A single
  thread's sequential order is then different.

Thus the V5 leaf is a candidate only for short sequences where ATen does not
split the reduction across warps. It cannot be claimed bit-exact for the full
serving range. The accumulation type also needs confirmation: ATen reduction
uses `arg_t = opmath_type<scalar_t>` (expected F32 for BF16), while the final
sum is cast back to BF16 before the separate BF16 divide.

## Required Root-controlled plan

1. On the actual pinned 5c488 CUDA build, capture `sum(dim=0)` and the full
   Native vector for fixed BF16 inputs at `H=4096` and sequence lengths
   spanning the warp-split threshold (1..8192, including `S=1,4,8,15,16,17,32,
   64,128,256,1024,4096,8192`) plus image/ragged cases.
2. Capture the effective reduction config (debug harness or a patched local
   PyTorch build) for those shapes: block width/height, `input_mult`,
   `step_input`, output vector size, CTA count, and vectorization flags.
3. Implement either an ATen-order two-stage reduction matching that config or
   define an explicit qualified tolerance. Do not claim bit-exact until the
   vectors match bit-for-bit over the control set.
4. Keep the mean export blocked from central AOT compilation until this
   review/Root-control closure. The Rust loader may keep the export optional;
   serving must fail closed if the qualified export is absent.

Central device owner: `/root/parity_catalog` (union 130 held; likely 131 for
this and the gated GELU export). See `DEVICE-REQUEST.md`.
