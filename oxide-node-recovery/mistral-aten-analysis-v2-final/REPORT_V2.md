# V2: exact ATen 5c488 BF16 sum and CPU-scalar true-divide mean

Analysis only. V1 in `/tmp/oxide-mistral3-aten-mean-analysis-v1/` is preserved
unchanged. No product source was edited. No GPU, Torch execution, Docker,
CUDA build, AOT build, or network access was used by the author.

Files in this V2 output:

- `REPORT_V2.md` - this report.
- `mistral3_model_mean_aten_v2.rs` - proposed one-thread-per-feature kernel.
- `reference_vectors_v2.py` - CPU tests for reduction order and rounding.
- `NativeTorchCUDAcontrol.py` - Root-run GPU0 control generator.
- `PROVENANCE_V2.md` - pinned supplement provenance and license terms.

## 1. Supplement additions

New exact primary files at commit
`5c4886908584029761b579af026dcfb627c84070`
(from `PROVENANCE-supplement.json`):

| File | Blob |
|---|---|
| `aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu` | `c7345633edd15ba66d59a84d9991fb77bf7c7944` |
| `aten/src/ATen/native/cuda/BinaryMulKernel.cu` | `26c4b15a7c6f73136e47ee8a1c921eca9f35a317` |
| `c10/util/BFloat16.h` | `6d3510cd5be83614320044338c10ab470eae13d4` |
| `LICENSE` | `c23172f7aff0254e4f0f163fb2e6e355cbaec5f4` |

`BFloat16.h` is a 44-byte forwarding header:

```cpp
#include <torch/headeronly/util/BFloat16.h>
```

The actual conversion body therefore still needs
`torch/headeronly/util/BFloat16.h` at the same commit. The reduction and division
arguments below do not depend on guessing that body: CUDA's BF16 conversion is
round-to-nearest-even, and the repository helper `bf16_from_f32` uses the CUDA
BF16 convert instruction. The CPU tests pin the RNE tie behavior explicitly.

## 2. Exact division: reciprocal multiply, not divide

`BinaryDivTrueKernel.cu:29-60` has the decisive branch:

```cpp
if (iter.is_cpu_scalar(2)) {
  using opmath_t = at::opmath_type<scalar_t>;
  using high_prec_t = ... double ...;
  auto inv_b = static_cast<opmath_t>(
      high_prec_t(1.0) / iter.scalar_value<high_prec_t>(2));
  iter.remove_operand(2);
  gpu_kernel(
      iter,
      BUnaryFunctor<scalar_t, scalar_t, scalar_t, MulFunctor<opmath_t>>(
          MulFunctor<opmath_t>(), inv_b));
}
```

`value.size(0)` is an int64 **CPU scalar**, so `iter.is_cpu_scalar(2)` is true.
For `scalar_t = at::BFloat16`, `opmath_t = float` and `high_prec_t = double`.
Therefore the exact mean is:

```text
inv   = f32( f64(1.0) / f64(T) )
v     = BF16( f32( f32_from_bf16(sum_bf16) * inv ) )
mean  = f32_from_bf16(v)
```

There is no FP32 division in ATen. The candidate's
`f32_from_bf16(sum) / (T as f32)` is algebraically different.

The exact `BUnaryFunctor` body is in
`aten/src/ATen/native/cuda/BinaryInternal.h` and the `gpu_kernel` conversion
rules are in `aten/src/ATen/native/cuda/Loops.cuh`; neither is in the fetched
supplement. Their template arguments (`MulFunctor<opmath_t>`, `inv_b` in
`opmath_t`) are what make the multiplication F32 and the return cast BF16. Root's
GPU control will confirm the final bits directly.

### Observable consequence in the admitted domain

Exhaustive CPU check over every positive finite BF16 `sum` (negative finite is
sign-symmetric; zero/inf/NaN behave identically) for every `T = 1..4096`:

```text
BF16( f32(sum * f32(1/T in f64->f32)) )
    == BF16( f32(sum / f32(T)) )
```

for every tested pair. The full sweep took 137 s with two workers. So the
candidate divide is observationally equivalent at the final BF16 mean for this
domain, even though the actual recipe is reciprocal multiply. V2 still
implements the exact reciprocal recipe.

## 3. Proposed kernel: one thread per feature

The key simplification requested is valid: the per-feature ATen sum is a fixed
sequence of F32 additions. Parallel mapping does not change that sequence.
Therefore one CUDA thread can load one feature down the T rows and replay:

1. `reduce_feature`: ATen `thread_reduce_impl`, four F32 accumulators,
   row stride 1 (T<64), 4 (local warp split), or 4C (global split), then a left
   fold `((a0+a1)+a2)+a3`.
2. T in 64..1020: four warp partials `p_y = R(y,4)`, then
   `((p0+p2)+(p1+p3))`.
3. T >= 1021 with C>1: for each CTA `b`, form
   `q_b = ((p_{b,0}+p_{b,2})+(p_{b,1}+p_{b,3}))`, fold `q_b` into
   `u_{b mod 4}` in increasing `b`, then `((u0+u2)+(u1+u3))`.
4. Project the final F32 sum to BF16 once, then apply the exact reciprocal mean.

No global atomics, semaphores, staging buffers, or cross-thread communication
are needed. The kernel only reads `input`, `cu_seqlens`, `selected` and writes
`out`. This is slower than ATen's parallel CTA tree but is suitable for
postprocessing and gives exact order replay.

Launch in the proposed source:

```text
block = 256
grid  = (ceil(width/256), sequences)
thread -> (sequence = blockIdx.y, feature = blockIdx.x*blockDim.x + threadIdx.x)
```

For ragged pooling, each `(sequence, feature)` thread reads
`start = cu_seqlens[selected[sequence]]`,
`end = cu_seqlens[selected[sequence]+1]`, `T = end-start`, and computes its own
`C`. Different sequences in the same kernel can have different T, C, and
reduction regime. Output is F32 `[sequences, width]`.

## 4. Host config formula for actual `ctas_per_output`

The Rust device helper and the Python control both implement the exact
`Reduce.cuh:1162-1183` formula. With host-captured `num_sm` and
`max_threads_per_sm`:

```text
if T < 1021 or max_threads_per_sm < 128:  C = 1
blocks_per_sm = floor(max_threads_per_sm / 128)
target        = num_sm * blocks_per_sm
grid          = ceil(H / 128)                 # T >= 64, H divisible by 4
if grid > target:  C = 1
vpt   = ceil(T / 4)
c1    = ceil(target / grid)
c2    = ceil(vpt / 16)
c3    = ceil(vpt / 256)
C     = clamp(c1, c3, c2) = min(max(c1, c3), c2)
if C <= 1: C = 1
```

Illustrative actual values (the control records the real values from the GPU):

| T | H=3072 A100 / H100 | H=4096 A100 / H100 | H=5120 A100 / H100 |
|---|---|---|---|
| 1021 | 16 / 16 | 16 / 16 | 16 / 16 |
| 1024 | 16 / 16 | 16 / 16 | 16 / 16 |
| 2048 | 32 / 32 | 32 / 32 | 32 / 32 |
| 4096 | 64 / 64 | 54 / 64 | 44 / 53 |

A100 target: 108 SM * (2048/128) = 1728. H100 SXM target:
132 SM * (2048/128) = 2112. `C` is device-dependent and must not be hard-coded.

## 5. CPU tests

`reference_vectors_v2.py` (stdlib only, binary32 RNE, BF16 RNE) checks:

- FP32 tie-to-even at 1.0 and BF16 tie-to-even at 1.0;
- one final BF16 projection for the sum;
- exact reciprocal-multiply mean resolves to the same final BF16 as divide on
  the tested values while differing by one F32 ulp before rounding;
- the T=5, T=64, and T=1021/C=16 structured witnesses;
- a sparse cancellation witness for **every requested T**
  (5,7,63,64,65,1020,1021,1024,2048,4096) where both sum and mean bits differ
  from sequential order;
- exhaustive positive-finite BF16 sum domain equivalence of reciprocal vs
  divide for all requested T.

Observed witnesses:

```text
T=5  [1, eps, -1, eps, eps]   ATen sum 0x3280 / mean 0x314d
                               sequential 0x3300 / mean 0x31cd
T=64 [1, -1, eps, eps, 0...]  ATen sum 0x0000 / mean 0x0000
                               sequential 0x3300 / mean 0x3000
T=1021 same 4 nonzero, C=16    ATen sum 0x0000 / mean 0x0000
                               sequential 0x3300 / mean 0x2e00
```

Run the fast suite:

```sh
taskset -c 152,153 python3 reference_vectors_v2.py
```

Reproduce the full T=1..4096 BF16-domain equivalence sweep (about 137 s,
two workers):

```sh
taskset -c 152,153 python3 reference_vectors_v2.py --full
```

## 6. Root-run GPU control

`NativeTorchCUDAcontrol.py` is for Root only, with the existing SDK:

```sh
CUDA_VISIBLE_DEVICES=0 /opt/torch-sdk/bin/python3.12 \
    NativeTorchCUDAcontrol.py --out ./native_cuda_control_out
```

Properties:

- refuses to run unless `CUDA_VISIBLE_DEVICES` is exactly `"0"`;
- refuses a Torch build whose `torch.version.git_version` does not start with
  the pinned commit;
- imports no NumPy; CPU bytes come from
  `ctypes.string_at(tensor.data_ptr(), nbytes)`;
- default `--suite first` is the bounded full-word control: one pattern per
  `(H,T)`, cycling sparse cancellation -> stress -> normal;
- `--suite all` runs sparse, normal, and stress at every `(H,T)` (about three
  times the binary volume);
- H = 3072/4096/5120 and T = 5/7/63/64/65/1020/1021/1024/2048/4096;
- the selected span starts at a nonzero row (`1..23`) and the surrounding
  prefix/tail rows are filled so a wrong offset would change the result;
- stores the full `input_storage_bf16.bin` (including the nonzero selected
  offset), `sum_bf16.bin`, and `mean_f32.bin` per case, plus a small
  `manifest.json` with SHA-256, byte count, dtype/shape, storage start and
  prefix/tail fill, `cu_seqlens`, `selected`, expected C, device properties,
  Torch git/version, CUDA/cuDNN versions;
- prints one line per case; the JSON has 27 entries in the default suite, not a
  huge dump.

Sparse witnesses are regime-correct: T<64 uses `[1, eps, -1, eps, eps]`;
T>=64 uses `[1, -1, eps, eps]`. Both are known to differ from sequential order
for every requested T. The default first suite selects sparse for T<=1021 and
uses stress/normal for the larger T; `--suite all` runs sparse at every T.

## 7. Licensing

PyTorch primary files, including the supplement `LICENSE`, are BSD-3-Clause.
`cuda-device` is Apache-2.0. The repository's covered Infinity V3 contributions
use Infinity Restricted MIT License 1.0 (custom, not standard MIT). The proposed
Rust source is an original order port and copies no upstream C++ text. See
`PROVENANCE_V2.md`.
