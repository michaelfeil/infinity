# Provenance and licensing

## Primary analysis source

Repository: `pytorch/pytorch`
Commit: `5c4886908584029761b579af026dcfb627c84070`

Fetched into `/tmp/oxide-mistral3-aten-5c488-mean-source-v1/` by the primary
source task; the blob IDs are recorded in that directory's `PROVENANCE.json`:

| File | Blob |
|---|---|
| `aten/src/ATen/native/cuda/Reduce.cuh` | `ab51f76613fd138293aa0c63faa5a991f17c24e9` |
| `aten/src/ATen/native/cuda/ReduceSumProdKernel.cu` | `36f0835890de87c2b85c3f6e38e2d39a8c52228d` |
| `aten/src/ATen/native/cuda/Reduce.cu` | `b32c55a10df6b3c640ed59bd373b2a893125c397` |

PyTorch is distributed under the BSD-3-Clause license. The files above retain
their original notices; the analysis and the original Rust port in this
directory are not a redistribution of PyTorch source text.

## Additional primary files needed to close non-sum gaps

These were not in the fetched set. They are named rather than guessed:

- `aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu` - exact true-divide path
  for `sum_bf16 / T` (and, if its CPU-scalar branch uses reciprocal multiply,
  `aten/src/ATen/native/cuda/BinaryMulKernel.cu`).
- `c10/util/BFloat16.h` - exact `float -> at::BFloat16` rounding contract.
- `aten/src/ATen/native/cuda/MemoryAccess.cuh` - `load_vector` /
  `aligned_vector` definition used by `thread_reduce_impl`.
- `aten/src/ATen/native/cuda/thread_constants.h` and `c10/macros/Macros.h` -
  `C10_WARP_SIZE` / `warp_size()`.

All would be taken at commit `5c4886908584029761b579af026dcfb627c84070`.

## Retained terms

- PyTorch primary source: BSD-3-Clause.
- `cuda-device` / cuda-oxide upstream intrinsics: Apache-2.0.
- Repository `infinity_v3` covered contributions: Infinity Restricted MIT
  License 1.0 (the custom license in `LICENSES/Infinity-Restricted-MIT-1.0.txt`,
  not standard MIT).
- Inherited Text Embeddings Inference and previously Apache-licensed material
  retain Apache-2.0; third-party components retain their own notices.

`mistral3_model_mean_aten.rs` is an original Rust expression of the reduction
order in the pinned PyTorch files. It reuses the repository's existing
`bf16_from_f32`, `f32_from_bf16`, and `read` helpers and the standard
`cuda_device` intrinsics; no upstream C++ text is copied. No model weights,
datasets, or generated model output are included.
