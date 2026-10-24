// Adapter for the pinned native Torch Flash source; vendor notices are retained.
#include "kernels/torch_flash_exact32.h"
#include <cuda_runtime_api.h>
#ifdef UNFUSE_FMA
#undef UNFUSE_FMA
#endif
#define FLASH_NAMESPACE tei_torch_flash_exact32
#include "vendor/torch_flash_attention/flash.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>
#include <limits>

namespace tei_torch_flash_exact32 {
void launch(Flash_fwd_params&, cudaStream_t);
}

namespace infinity {
/// Launch the twelve-head width-32 noncausal SM90 specialization over actual FP16/BF16 token rows.
at::Tensor torch_flash_exact32(const at::Tensor& q,
                               const at::Tensor& k,
                               const at::Tensor& v,
                               const at::Tensor& cumulative,
                               int64_t maximum,
                               double scale) {
  TORCH_CHECK(
      q.is_cuda() && q.dim() == 3 && k.sizes() == q.sizes() && v.sizes() == q.sizes() &&
          k.device() == q.device() && v.device() == q.device() &&
          (q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16) &&
          k.scalar_type() == q.scalar_type() && v.scalar_type() == q.scalar_type() &&
          q.size(2) == 32 && q.size(1) == 12 && q.stride(2) == 1 && k.stride(2) == 1 &&
          v.stride(2) == 1,
      "Exact head32 attention requires matching packed CUDA FP16/BF16 tensors with twelve heads");
  TORCH_CHECK(cumulative.device() == q.device() && cumulative.scalar_type() == at::kInt &&
                  cumulative.dim() == 1 && cumulative.is_contiguous() && cumulative.numel() > 1 &&
                  cumulative.numel() - 1 <= std::numeric_limits<int>::max() && q.size(0) > 0 &&
                  q.size(0) <= std::numeric_limits<int>::max() && maximum > 0 &&
                  maximum <= q.size(0) && maximum <= std::numeric_limits<int>::max() - 127 &&
                  std::isfinite(static_cast<float>(scale)),
              "Invalid exact head32 packed dimensions");
  c10::cuda::CUDAGuard guard(q.device());
  TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major == 9,
              "Exact head32 attention is qualified only for SM90");
  auto out = at::empty(q.sizes(), q.options());
  auto lse = at::empty({q.size(1), q.size(0)}, q.options().dtype(at::kFloat));
  tei_torch_flash_exact32::Flash_fwd_params p{};
  p.q_ptr = q.data_ptr();
  p.k_ptr = k.data_ptr();
  p.v_ptr = v.data_ptr();
  p.o_ptr = out.data_ptr();
  p.q_row_stride = q.stride(0);
  p.k_row_stride = k.stride(0);
  p.v_row_stride = v.stride(0);
  p.q_head_stride = q.stride(1);
  p.k_head_stride = k.stride(1);
  p.v_head_stride = v.stride(1);
  p.o_row_stride = out.stride(0);
  p.o_head_stride = out.stride(1);
  p.b = cumulative.numel() - 1;
  p.h = 12;
  p.h_k = 12;
  p.h_h_k_ratio = 1;
  p.seqlen_q = maximum;
  p.seqlen_k = maximum;
  p.seqlen_q_rounded = ((maximum + 127) / 128) * 128;
  p.seqlen_k_rounded = p.seqlen_q_rounded;
  p.d = 32;
  p.d_rounded = 32;
  p.total_q = q.size(0);
  p.cu_seqlens_q = cumulative.data_ptr<int>();
  p.cu_seqlens_k = cumulative.data_ptr<int>();
  p.softmax_lse_ptr = lse.data_ptr();
  p.unpadded_lse = true;
  p.is_seqlens_k_cumulative = true;
  p.scale_softmax = static_cast<float>(scale);
  p.scale_softmax_log2 = p.scale_softmax * M_LOG2E;
  p.p_dropout = 1.f;
  p.p_dropout_in_uint8_t = 255;
  p.rp_dropout = 1.f;
  p.scale_softmax_rp_dropout = p.scale_softmax;
  p.is_causal = false;
  p.window_size_left = -1;
  p.window_size_right = -1;
  p.is_bf16 = q.scalar_type() == at::kBFloat16;
  p.num_splits = 1;
  tei_torch_flash_exact32::launch(p, at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
}
