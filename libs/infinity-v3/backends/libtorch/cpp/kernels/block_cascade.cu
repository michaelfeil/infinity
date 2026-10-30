// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
#include "block_cascade.h"
#include <ATen/ops/_flash_attention_forward.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <flashinfer/attention/cascade.cuh>

namespace infinity {
bool block_cascade_supported(const at::Tensor& q) {
  if (!q.is_cuda() || q.dim() != 3 || q.size(0) == 0 || q.stride(2) != 1 ||
      (q.scalar_type() != at::kHalf && q.scalar_type() != at::kBFloat16))
    return false;
  const auto dim = q.size(2);
  if (dim != 64 && dim != 128 && dim != 256)
    return false;
  const auto threads_per_head = dim / std::max<int64_t>(8, dim / 32);
  return q.size(1) * threads_per_head <= 1024 &&
         at::cuda::getDeviceProperties(q.get_device())->major >= 8;
}

at::Tensor block_cascade_attention(const at::Tensor& q,
                                   const at::Tensor& k,
                                   const at::Tensor& v,
                                   const at::Tensor& prefix_k,
                                   const at::Tensor& prefix_v,
                                   const at::Tensor& cumulative,
                                   const int32_t* offsets,
                                   const std::vector<int32_t>& prefix_offsets,
                                   const std::vector<int64_t>& sources,
                                   int64_t max_suffix,
                                   double scale) {
  TORCH_CHECK(block_cascade_supported(q), "Unsupported block cascade layout");
  c10::cuda::CUDAGuard guard(q.device());
  at::Tensor prefix_output, prefix_lse;
  // ATen returns natural-log LSE; FlashInfer MergeState requires base 2.
  constexpr double log2e = 1.4426950408889634;
  for (size_t first = 0; first < sources.size();) {
    size_t last = first + 1;
    while (last < sources.size() && sources[last] == sources[first])
      ++last;
    const auto source = sources[first];
    const auto start = prefix_offsets[source], length = prefix_offsets[source + 1] - start;
    const auto qstart = offsets[first], qlength = offsets[last] - qstart;
    // All these queries see the same immutable prefix. Fixed-batch attention
    // uses views: no prefix KV replication, padding, or device boundary uploads.
    auto result = at::_flash_attention_forward(q.narrow(0, qstart, qlength).unsqueeze(0),
                                               prefix_k.narrow(0, start, length).unsqueeze(0),
                                               prefix_v.narrow(0, start, length).unsqueeze(0),
                                               std::nullopt,
                                               std::nullopt,
                                               qlength,
                                               length,
                                               0.,
                                               false,
                                               false,
                                               scale);
    if (first == 0 && last == sources.size()) {
      // The common one-state fanout can merge the attention output directly.
      prefix_output = std::get<0>(result).squeeze(0);
      prefix_lse = std::get<1>(result).squeeze(0).transpose(0, 1).contiguous().mul_(log2e);
    } else {
      if (!prefix_output.defined()) {
        prefix_output = at::empty(q.sizes(), q.options());
        prefix_lse = at::empty({q.size(0), q.size(1)}, q.options().dtype(at::kFloat));
      }
      prefix_output.narrow(0, qstart, qlength).copy_(std::get<0>(result).squeeze(0));
      prefix_lse.narrow(0, qstart, qlength)
          .copy_(std::get<1>(result).squeeze(0).transpose(0, 1))
          .mul_(log2e);
    }
    first = last;
  }
  // Each suffix remains bidirectional within its own question. No question can
  // attend to another suffix. The FA4 export currently does not expose LSE,
  // so use ATen FlashAttention for both partial states rather than guessing it.
  auto suffix = at::_flash_attention_forward(
      q, k, v, cumulative, cumulative, max_suffix, max_suffix, 0., false, false, scale);
  auto suffix_output = std::get<0>(suffix).contiguous();
  auto suffix_lse = std::get<1>(suffix).transpose(0, 1).contiguous().mul_(log2e);
  auto output = at::empty(q.sizes(), q.options());
  auto stream = at::cuda::getCurrentCUDAStream(q.get_device()).stream();
  cudaError_t status;
  if (q.scalar_type() == at::kBFloat16)
    status = flashinfer::MergeState(reinterpret_cast<__nv_bfloat16*>(prefix_output.data_ptr()),
                                    prefix_lse.data_ptr<float>(),
                                    reinterpret_cast<__nv_bfloat16*>(suffix_output.data_ptr()),
                                    suffix_lse.data_ptr<float>(),
                                    reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                                    nullptr,
                                    q.size(0),
                                    q.size(1),
                                    q.size(2),
                                    stream);
  else
    status = flashinfer::MergeState(reinterpret_cast<half*>(prefix_output.data_ptr()),
                                    prefix_lse.data_ptr<float>(),
                                    reinterpret_cast<half*>(suffix_output.data_ptr()),
                                    suffix_lse.data_ptr<float>(),
                                    reinterpret_cast<half*>(output.data_ptr()),
                                    nullptr,
                                    q.size(0),
                                    q.size(1),
                                    q.size(2),
                                    stream);
  C10_CUDA_CHECK(status);
  return output;
}
}
