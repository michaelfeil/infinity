# Provenance and licensing (V2)

V1 remains at `/tmp/oxide-mistral3-aten-mean-analysis-v1/` and is unchanged.

## Pinned PyTorch primary source

Repository: `pytorch/pytorch`
Commit: `5c4886908584029761b579af026dcfb627c84070`

From the original fetch (`PROVENANCE.json`):

| File | Blob |
|---|---|
| `aten/src/ATen/native/cuda/Reduce.cuh` | `ab51f76613fd138293aa0c63faa5a991f17c24e9` |
| `aten/src/ATen/native/cuda/ReduceSumProdKernel.cu` | `36f0835890de87c2b85c3f6e38e2d39a8c52228d` |
| `aten/src/ATen/native/cuda/Reduce.cu` | `b32c55a10df6b3c640ed59bd373b2a893125c397` |

From the supplement (`PROVENANCE-supplement.json`):

| File | Blob |
|---|---|
| `aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu` | `c7345633edd15ba66d59a84d9991fb77bf7c7944` |
| `aten/src/ATen/native/cuda/BinaryMulKernel.cu` | `26c4b15a7c6f73136e47ee8a1c921eca9f35a317` |
| `c10/util/BFloat16.h` | `6d3510cd5be83614320044338c10ab470eae13d4` |
| `LICENSE` | `c23172f7aff0254e4f0f163fb2e6e355cbaec5f4` |

PyTorch is BSD-3-Clause. The fetched `c10/util/BFloat16.h` is a forwarding
header to `torch/headeronly/util/BFloat16.h`; that body was not in the
supplement. If a full audit of the conversion implementation is required, fetch
`torch/headeronly/util/BFloat16.h` at the same commit.

To pin the full division call path, the remaining exact paths are:

- `aten/src/ATen/native/cuda/BinaryInternal.h` - `BUnaryFunctor` stored `arg2_t`
  and return conversion.
- `aten/src/ATen/native/cuda/Loops.cuh` - `gpu_kernel` argument conversion.

These were cited rather than guessed. The branch in
`BinaryDivTrueKernel.cu` itself establishes reciprocal-multiply in `opmath_t`;
Root's GPU control confirms final bits.

## Retained terms

- PyTorch primary source: BSD-3-Clause.
- `cuda-device` / cuda-oxide upstream intrinsics: Apache-2.0.
- Repository `infinity_v3` covered contributions: Infinity Restricted MIT
  License 1.0 (custom, not standard MIT).
- Inherited Text Embeddings Inference and previously Apache-licensed material
  retain Apache-2.0; third-party components retain their notices.

`mistral3_model_mean_aten_v2.rs` is an original Rust expression of the pinned
reduction/division orders. It reuses the repository's existing
`bf16_from_f32`, `f32_from_bf16`, and `read` helpers and standard `cuda_device`
intrinsics; it copies no upstream C++ text. No model weights, datasets, or model
outputs are included.
