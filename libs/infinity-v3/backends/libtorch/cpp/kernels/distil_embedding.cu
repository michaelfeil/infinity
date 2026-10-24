// Adapted from the BSD-3-Clause Candle residual layer-norm kernel.
// The full notice is retained in distil_embedding_kernel.cuh and the installed
// candle-layer-norm license. Only input gathers and unused residual storage differ.
#include "kernels/distil_embedding.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "distil_embedding_kernel.cuh"

namespace tei {
at::Tensor distil_embedding_norm(const at::Tensor& words,
                                 const at::Tensor& positions,
                                 const at::Tensor& ids,
                                 const at::Tensor& pos,
                                 const at::Tensor& gamma,
                                 const at::Tensor& beta,
                                 double eps) {
  c10::cuda::CUDAGuard guard(words.device());
  TORCH_CHECK(words.is_cuda() && words.scalar_type() == at::kHalf && words.is_contiguous() &&
                  words.size(1) == 768,
              "word geometry");
  TORCH_CHECK(positions.device() == words.device() &&
                  positions.scalar_type() == words.scalar_type() && positions.is_contiguous() &&
                  positions.size(1) == 768,
              "position geometry");
  TORCH_CHECK(ids.device() == words.device() && pos.device() == words.device() &&
                  ids.scalar_type() == at::kLong && pos.scalar_type() == at::kLong &&
                  ids.is_contiguous() && pos.is_contiguous() && ids.numel() == pos.numel(),
              "indices");
  TORCH_CHECK(gamma.device() == words.device() && beta.device() == words.device() &&
                  gamma.is_contiguous() && beta.is_contiguous() && gamma.numel() == 768 &&
                  beta.numel() == 768 && gamma.scalar_type() == at::kHalf &&
                  beta.scalar_type() == at::kHalf,
              "norm geometry");
  auto out = at::empty({ids.numel(), 768}, words.options());
  using Traits = layer_norm::Kernel_traits<half, half, half, half, float, int, 768, 1, 4, 1, 16>;
  layer_norm::FwdParams p;
  p.x1 = nullptr;
  p.dmask = nullptr;
  p.dmask1 = nullptr;
  p.x0_subset = nullptr;
  p.z_subset = nullptr;
  p.rows = ids.numel();
  p.cols = 768;
  p.x0 = const_cast<void*>(words.const_data_ptr());
  p.residual = const_cast<void*>(positions.const_data_ptr());
  p.z = out.mutable_data_ptr();
  p.gamma = const_cast<void*>(gamma.const_data_ptr());
  p.beta = const_cast<void*>(beta.const_data_ptr());
  p.epsilon = eps;
  p.inverse_cols = 1.f / 768;
  p.rowscale_const = 1;
  p.round_residual = false;
  auto kernel = layer_norm::distil_embedding_norm_kernel<Traits, false, false, false, true>;
  static thread_local int cache[16] = {};
  const int dev = words.device().index();
  int blocks = 0;
  if (dev < 16)
    blocks = cache[dev];
  if (!blocks) {
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, kernel, Traits::THREADS_PER_CTA, Traits::SMEM_BYTES_FWD));
    if (dev < 16)
      cache[dev] = blocks;
  }
  p.ctas_per_col = blocks * at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  kernel<<<p.ctas_per_col,
           Traits::THREADS_PER_CTA,
           Traits::SMEM_BYTES_FWD,
           at::cuda::getCurrentCUDAStream(dev)>>>(p,
                                                  ids.const_data_ptr<int64_t>(),
                                                  pos.const_data_ptr<int64_t>(),
                                                  words.size(0),
                                                  positions.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
}
