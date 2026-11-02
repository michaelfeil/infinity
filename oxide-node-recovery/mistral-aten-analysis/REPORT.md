# ATen 5c488 BF16 `sum(dim=0)` order for Mistral3 mean pooling

Analysis only. No product source was edited, and no GPU/Torch/CUDA/AOT/network
run was performed. Outputs in this directory:

- `REPORT.md` - derivation, order, config thresholds, fixture plan.
- `mistral3_model_mean_aten.rs` - proposed source-only device function.
- `reference_vectors.py` - exact CPU reference and assertions.
- `PROVENANCE.md` - pinned provenance and license terms.

## 1. Pinned primary source and the actual operation

`mistral3_models.cpp:405-409`:

```cpp
return (value.sum(0, true) / value.size(0)).to(at::kFloat);
```

`value` is contiguous BF16 `[T,H]`, `H` in {3072,4096,5120}. The reduction is
`sum(dim=0, keepdim=true)`, then BF16 true division by `T`, then an exact widening
to F32.

Pinned files and blob IDs are in `/tmp/oxide-mistral3-aten-5c488-mean-source-v1/PROVENANCE.json`
(commit `5c4886908584029761b579af026dcfb627c84070`). All line references below
are to those pinned copies.

Dispatch chain:

1. `ReduceSumProdKernel.cu:205-225` selects `OpFunctor<at::BFloat16, float>` for
   a BF16 iterator. With the default `out_t = scalar_t`, the instantiation is
   `sum_functor<scalar_t=BFloat16, acc_t=float, out_t=BFloat16>`.
2. `ReduceSumProdKernel.cu:13-29` therefore calls
   `gpu_reduce_kernel<BFloat16, BFloat16, vt0=4, input_vec_size=8>` with
   `sum_combine = a + b` in `float`.
3. `Reduce.cuh:1187-1210` infers `arg_t = float`; `can_accumulate_in_output` is
   false for BF16/BF16, but that only affects partial-output accumulation, which
   is not used here. Every partial sum is F32.
4. `Reduce.cuh:775-782` calls `ops.project`, i.e. one `float -> at::BFloat16`
   cast, only after the complete F32 sum.

So the sum is **F32 accumulation with one final BF16 round-to-nearest-even
projection**. There is no per-add BF16 rounding.

## 2. Why the shape takes the "vectorize along output" path

For contiguous `[T,H]` reduced over dim 0, TensorIterator puts the reduced dim
first. Input element strides are `[H, 1]` (`bytes: [2H, 2]`); reduce dims = 1.

`Reduce.cuh:1053-1082`:

- `strides(input)[0] = 2H > strides(input)[1] = 2`, so
  `reduction_on_fastest_striding_dimension = false`.
- hence `dim0 = num_outputs = H`, `dim1 = inputs_per_output = T`,
  `fastest_moving_stride = 2 = sizeof(BFloat16)`.

`Reduce.cuh:1104-1116`: case 2 applies, so `output_vec_size` is computed and
`dim0 /= output_vec_size`. `vectorize_input` stays false; the `input_vec_size=8`
specialization in `sum_functor` is dead for this layout. Do not model an 8-wide
input vector reduction here.

`Reduce.cuh:1013-1036` (`get_output_vec_size`) starts at 4 and halves while any
of these is not divisible by the candidate: the input address in BF16 elements,
`H`, and the reduced-dim stride in BF16 elements (`H`). For the admitted
`H = 3072/4096/5120` all are divisible by 4 (and by 8 for the row starts used by
`narrow(0,start,length)`), so:

```text
output_vec_size = 4, dim0 = H/4, fastest_moving_stride == sizeof(BF16)
```

Each lane loads four adjacent BF16 features (`aligned_vector<BF16,4>` = 8 bytes),
not eight.

## 3. Exact launch configuration from `setReduceConfig`

`Reduce.cuh:1038-1185`, with `arg_t = float`, `scalar_t = BF16`, `vt0 = 4`:

`ReduceConfig::element_size_bytes = sizeof(float) = 4` (staging/shared account
in F32 even though the tensor is BF16).

`set_block_dimension<BFloat16>` (`Reduce.cuh:98-107`):

```text
max_num_threads = mnt_wrapper<BFloat16>::MAX_NUM_THREADS / output_vec_size
                = 512 / 4 = 128
dim0_pow2 = min(last_pow2(H/4), 128) = 128
dim1_pow2 = min(last_pow2(T), 128)
block_width  = min(dim0_pow2, warp_size=32) = 32
block_height = min(dim1_pow2, 128/32) = min(last_pow2(T), 4)
block_width  = min(128, 128/block_height)
num_threads  = block_width * block_height = 128
```

Therefore:

| T | block | step_input | block_y reduction | grid.x | regime |
|---|---|---|---|---|---|
| 1 | (128,1) | 1 | no | ceil(H/512) | one thread per feature |
| 2,3 | (64,2) | 1 | no | ceil(H/512) | one thread per feature |
| 4..63 | (32,4) | 1 | no | ceil(H/512) | one thread per feature |
| 64..1020 | (32,4) | 4 | yes | ceil(H/128) | 4 warps split rows |
| >=1021 | (32,4) | 4C | yes | (ceil(H/128), C) | global staged reduction |

Why the switches:

- `Reduce.cuh:1124-1132`: because reduction is not on the fastest dimension,
  `output_mult[0] = split_output(block_width)`, so `step_output = block_width`;
  `input_mult[0]` stays 0. There is no block-x reduction in any admitted shape.
- `Reduce.cuh:1139-1155`: `warp_split_threshold = min(block_height*16, 256)`.
  For `T >= 4`, `block_height = 4`, threshold 64, and
  `values_per_thread = ceil(T/1) = T`. Thus `T >= 64` sets
  `input_mult[1] = 1`, `step_input = 4`; otherwise
  `output_mult[1] = block_width` and warps own different outputs.
- `Reduce.cuh:1162-1183`: global reduction is additionally allowed when
  `input_mult[1] != 0`, `values_per_thread = ceil(T/4) >= 256`, and
  `grid.x <= target_grid_size`. `ceil(T/4) >= 256` starts at **T = 1021**, not
  1024 (1021/4 rounds up to 256; 1020/4 = 255).

For the emitted H values, `grid.x` is 24/32/40 in the split regime, while
`target_grid_size = num_mp * (maxThreadsPerSM/128)` is in the hundreds/thousands
on current data-centre GPUs, so the `grid.x <= target_grid_size` test passes
there. A very small device could fail it and remain in the no-global regime.
Global split then uses:

```text
ctas1 = ceil(target_grid_size / grid.x)
ctas2 = ceil(values_per_thread / 16)
ctas3 = ceil(values_per_thread / 256)
C = ctas_per_output = clamp(ctas1, ctas3, ctas2)
input_mult[2] = 4, step_input = 4C
```

`C` is device dependent through `num_mp` and `maxThreadsPerSM`; it must be
captured, not guessed. Illustrative A100 (108 SM, 2048 threads/SM, target 1728)
values at `T=4096`: `C = 64/54/44` for `H = 3072/4096/5120`. On an H100 SXM
(132 SM, target 2112) the same entries are `64/64/53`.

At `T=65535/65536` nothing changes structurally: the same global regime applies,
`C` merely grows, and `T*H <= 335,544,320 < 2^31` for all three H, so the
TensorIterator remains 32-bit and the recursive sub-iterator path in
`Reduce.cuh:1235-1243` is not entered. The arithmetic order is still the formula
in section 5 with the larger C.

## 4. Element order, strides, padding, and vector alignment

For feature `j` and row `r`, the BF16 element is at `input[r*H + j]`:

- feature stride = 1 element = 2 bytes;
- reduced-dim row stride = `H` elements = `2H` bytes;
- no padding; `H` is even and divisible by 4 for all admitted widths, so every
  row start is 8-byte aligned and a 4xBF16 load is legal;
- `narrow(0,start,length)` only changes the storage offset by `start*H` BF16
  elements, a multiple of 4, so `get_output_vec_size` remains 4;
- output features are handled in adjacent groups of four. In the split regimes
  a lane's four features are the same for all four `threadIdx.y` warps.

If a future call supplies a non-contiguous view or a misaligned storage offset,
`output_vec_size` can drop to 2 or 1, which changes `dim0`, `step_output`, the
lane mapping and therefore the reduction order. That case is outside this
analysis.

## 5. Exact reduction order

Let `add(a,b)` mean IEEE-754 binary32 round-to-nearest-even addition. Define the
per-thread primitive for a starting row and row stride:

```text
R(start, stride):
    a0 = a1 = a2 = a3 = +0.0f
    row = start
    while row + 3*stride < T:
        a0 = add(a0, x[row])
        a1 = add(a1, x[row +     stride])
        a2 = add(a2, x[row + 2 * stride])
        a3 = add(a3, x[row + 3 * stride])
        row += 4*stride
    for i = 0..3 while row < T:
        a_i = add(a_i, x[row])
        row += stride
    return add(add(add(a0, a1), a2), a3)     // left fold, not a balanced tree
```

This is `thread_reduce_impl` (`Reduce.cuh:560-631`) with `vt0 = 4` and
`output_vec_size = 4`. The tail rows go into accumulators `0,1,2,3` in that
order. The accumulator combine is a **left fold**.

Regimes:

```text
T = 1..63:        S = R(0, 1)
T = 64..1020:     p_y = R(y, 4), y = 0..3
                  S = add(add(p_0, p_2), add(p_1, p_3))
T >= 1021, C > 1: p_{b,y} = R(y + 4b, 4C), b = 0..C-1, y = 0..3
                  q_b = add(add(p_{b,0}, p_{b,2}), add(p_{b,1}, p_{b,3}))
                  u_y = left fold over b = y, y+4, y+8, ... of q_b
                  S = add(add(u_0, u_2), add(u_1, u_3))
```

The middle formula is the `block_y_reduce` halving tree
(`Reduce.cuh:672-689`): offset 2 first pairs `y` with `y+2`, then offset 1
pairs `0` with `1`. The global formula additionally stages each `q_b`
(`Reduce.cuh:784-816`), elects the last CTA with `atomicAdd`
(`Reduce.cuh:691-703`), folds CTA partials by `threadIdx.y` and `blockDim.y`
(`Reduce.cuh:862-874`), then applies `block_y_reduce` again
(`Reduce.cuh:875-878`).

Consequences:

- `T <= 4`: `R(0,1)` is exactly left-to-right row order, so ATen and the
  candidate sequential sum agree.
- `T = 5`: ATen computes `(((x0+x4)+x1)+x2)+x3`, while the candidate's
  `mistral3_model_mean.rs:37-45` loop computes
  `((((x0+x1)+x2)+x3)+x4)`. The first difference is the tail row x4 being folded
  into accumulator 0 before the accumulator combine.
- For larger T the ATen order is a four-way interleave by row index modulo 4,
  followed by a left fold; for `T >= 64` it is a further four-way interleave by
  `threadIdx.y` followed by a pairwise block-y tree; for `T >= 1021` it is a
  further staging fold over CTAs by `cta modulo 4` followed by another pairwise
  block-y tree.

## 6. Precise rounding

- Sum combine: `arg_t = float`; `a + b` is binary32 RNE.
- Sum output: one `at::BFloat16(float)` / `__float2bfloat16` conversion, which
  is round-to-nearest-even with ties to even on the supported CUDA targets
  (`c10/util/BFloat16.h` at the same SHA is the exact needed primary file; it is
  not in the fetched set). The proposed Rust uses the repository's existing
  `bf16_from_f32`, which is `cvt_bf16x2_f32`, also RNE.
- Division: the candidate's `f32_from_bf16(sum) / T` in F32 then one BF16 RNE
  round is retained in the proposed source. The exact ATen true-divide path is
  **not** in the fetched set. The needed primary file is
  `aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu` at
  `5c4886908584029761b579af026dcfb627c84070`. If its CPU-scalar branch is the
  reciprocal-multiply optimization, also pin
  `aten/src/ATen/native/cuda/BinaryMulKernel.cu`; in that case the candidate's
  divide is not bit-exact for non-power-of-two T and must become
  `sum_bf16 * (1/T in opmath)` with the corresponding mul rounding.
- The `size(0)` operand is an int64 CPU scalar, so the CPU-scalar path of true
  division (not a general tensor/tensor divide) is the one to audit.

## 7. CPU reference vectors that break sequential order

`reference_vectors.py` implements exactly the formulas above (stdlib only,
binary32 RNE additions, BF16 RNE conversions). It asserts:

```text
T=5, eps = BF16 0x3280 = 2^-26
x = [1.0, eps, -1.0, eps, eps]
ATen sum  0x3280   ATen mean  0x314d
sequential sum 0x3300  sequential mean 0x31cd

T=64 sparse
x = [1.0, -1.0, eps, eps, 0, ... 60 zeros]
ATen sum  0x0000   ATen mean  0x0000
sequential sum 0x3300  sequential mean 0x3000

T=1021 sparse, C=16 (the global path)
x = [1.0, -1.0, eps, eps, 0, ... 1017 zeros]
ATen sum  0x0000   ATen mean  0x0000
sequential sum 0x3300  sequential mean 0x2e00
```

The T=64 and T=1021 vectors are sparse, so the whole tensor is specified by the
four nonzero positions. They are not random coincidences: the block-y tree pairs
`y=0` with `y=2` and `y=1` with `y=3`, exactly cancelling the large values before
the eps values are combined; sequential order leaves two eps values behind.
`T=1021, C=16` is conditional only on the host-captured `C`; the arithmetic
formula is exact for that C.

Run:

```sh
taskset -c 152-159 python3 reference_vectors.py
```

## 8. Next runtime fixture plan (GPU control separate)

1. On the real target GPU, capture and record:
   `num_mp`, `maxThreadsPerMultiProcessor`, `block_width`, `block_height`,
   `num_threads`, `input_mult[3]`, `output_mult[2]`, `step_input`,
   `step_output`, `ctas_per_output`, `grid`, `output_vec_size`,
   `vectorize_input`, `values_per_thread`. `Reduce.cu:19-54` already prints all
   of these through `operator<<(ReduceConfig)`.
2. Sweep `T = 1,2,3,4,5,6,7,8,63,64,1020,1021,2048,3072,4096,65535,65536` and
   `H = 3072,4096,5120` (or the serving subset), with the real
   `states.narrow(0,start,length)` view offset.
3. Differential-check exact BF16 bits of `sum(0,true)` and exact F32 bits of the
   final pooled mean against:
   - `reference_vectors.py` for the structured vectors;
   - the proposed Rust order for all random and sparse cases;
   - the candidate sequential order, to record every divergence.
4. Add the division discriminator: use a non-power-of-two T (e.g. 5, 1021,
   3072) and compare true-divide vs reciprocal-multiply recipes. Pin
   `BinaryDivTrueKernel.cu` before accepting the current division line.
5. Confirm `T=65535/65536` enters no new arithmetic regime and no 32-bit
   sub-iterator split; only `C` changes.
6. Do not qualify a fast path from a single output element: compare full `[H]`
   rows, and include sparse cancellation vectors because random dense sums often
   round to the same final BF16 value even when the F32 sums differ.
