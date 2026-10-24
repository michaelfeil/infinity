// SPDX-License-Identifier: Apache-2.0
// Original exact-copy CUDA operator; no third-party kernel source copied.
#include "kernels/radix_qkv.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <cstdint>

namespace tei {
namespace {
__global__ void unfold_qkv_kernel(const uint4* q,
                                  const uint4* k,
                                  const uint4* v,
                                  uint4* qo,
                                  uint4* ko,
                                  uint4* vo,
                                  const int64_t* scatter,
                                  uint32_t qw,
                                  uint32_t kw,
                                  int64_t qs,
                                  int64_t ks,
                                  int64_t vs,
                                  uint32_t rows) {
  const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint32_t row = blockIdx.y; row < rows; row += gridDim.y) {
    const int64_t source = scatter[row];
    if (col < qw)
      qo[int64_t(row) * qw + col] = q[source * qs + col];
    else if (col < qw + kw)
      ko[int64_t(row) * kw + col - qw] = k[source * ks + col - qw];
    else if (col < qw + 2 * kw)
      vo[int64_t(row) * kw + col - qw - kw] = v[source * vs + col - qw - kw];
  }
}
}

bool radix_qkv_cuda_supported(const at::Tensor& q,
                              const at::Tensor& k,
                              const at::Tensor& v,
                              const at::Tensor& scatter) {
  if (!q.defined() || !k.defined() || !v.defined() || !scatter.defined() || !q.is_cuda() ||
      q.dim() != 3 || k.dim() != 3 || v.sizes() != k.sizes() || q.size(0) <= 0 ||
      q.size(0) != k.size(0) || q.size(2) <= 0 || q.size(2) != k.size(2) || q.size(1) <= 0 ||
      k.size(1) <= 0 || (q.scalar_type() != at::kHalf && q.scalar_type() != at::kBFloat16) ||
      scatter.device() != q.device() || scatter.scalar_type() != at::kLong || scatter.dim() != 1 ||
      !scatter.is_contiguous() || scatter.numel() <= 0 || scatter.numel() >= 65536)
    return false;
  if (q.size(1) > UINT32_MAX / q.size(2) || k.size(1) > UINT32_MAX / k.size(2))
    return false;
  const auto qw = q.size(1) * q.size(2), kw = k.size(1) * k.size(2);
  if (kw > (UINT32_MAX - qw) / 2 || qw % 8 || kw % 8)
    return false;
  for (const auto& x : {q, k, v})
    if (x.device() != q.device() || x.scalar_type() != q.scalar_type() || x.stride(2) != 1 ||
        x.stride(1) != x.size(2) || x.stride(0) % 8 ||
        reinterpret_cast<uintptr_t>(x.const_data_ptr()) % 16)
      return false;
  return true;
}

/// Allocate three logical-row outputs and copy Q/K/V bits
/// together; scatter values are prevalidated.
std::array<at::Tensor, 3> radix_unfold_qkv_cuda(const at::Tensor& q,
                                                const at::Tensor& k,
                                                const at::Tensor& v,
                                                const at::Tensor& scatter) {
  TORCH_CHECK(radix_qkv_cuda_supported(q, k, v, scatter), "Unsupported fused radix QKV geometry");
  // Host-validated scatter entries reference actual compact rows. Explicit
  // guarding matters when Candle and LibTorch alternate CUDA devices.
  c10::cuda::CUDAGuard guard(q.device());
  const auto rows = scatter.numel(), qw = q.size(1) * q.size(2) / 8, kw = k.size(1) * k.size(2) / 8;
  std::array<at::Tensor, 3> output = {at::empty({rows, q.size(1), q.size(2)}, q.options()),
                                      at::empty({rows, k.size(1), k.size(2)}, k.options()),
                                      at::empty({rows, v.size(1), v.size(2)}, v.options())};
  unfold_qkv_kernel<<<dim3((qw + 2 * kw + 255) / 256, std::min<int64_t>(rows, 256)),
                      256,
                      0,
                      at::cuda::getCurrentCUDAStream(q.device().index())>>>(
      reinterpret_cast<const uint4*>(q.const_data_ptr()),
      reinterpret_cast<const uint4*>(k.const_data_ptr()),
      reinterpret_cast<const uint4*>(v.const_data_ptr()),
      reinterpret_cast<uint4*>(output[0].mutable_data_ptr()),
      reinterpret_cast<uint4*>(output[1].mutable_data_ptr()),
      reinterpret_cast<uint4*>(output[2].mutable_data_ptr()),
      scatter.const_data_ptr<int64_t>(),
      qw,
      kw,
      q.stride(0) / 8,
      k.stride(0) / 8,
      v.stride(0) / 8,
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
}
