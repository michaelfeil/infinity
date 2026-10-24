// Adapter to the pinned Torch Flash implementation; vendor BSD notices retained.
#include "kernels/torch_flash_alibi64.h"
#include <cuda_runtime_api.h>
#define FLASH_NAMESPACE tei_torch_flash_alibi64
#include "vendor/torch_flash_attention/flash.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>
#include <limits>

namespace tei_torch_flash_alibi64 {
void launch_alibi64(Flash_fwd_params&, cudaStream_t);
}

namespace tei {
bool torch_flash_alibi64_supported(c10::Device device, at::ScalarType dtype, int64_t dim) {
  return device.is_cuda() && (dtype == at::kHalf || dtype == at::kBFloat16) && dim == 64 &&
         at::cuda::getDeviceProperties(device.index())->major >= 8;
}

/// Validate the qualified packed width-64 profile and pass one contiguous F32 ALiBi slope per head.
at::Tensor torch_flash_alibi64(const at::Tensor& q,
                               const at::Tensor& k,
                               const at::Tensor& v,
                               const at::Tensor& cumulative,
                               const at::Tensor& slopes,
                               int64_t max_sequence,
                               double scale) {
  TORCH_CHECK(q.dim() == 3 && torch_flash_alibi64_supported(q.device(), q.scalar_type(), q.size(2)),
              "Private ALiBi Flash requires CUDA Half/BF16 head64 on SM80 or newer");
  TORCH_CHECK(k.sizes() == q.sizes() && v.sizes() == q.sizes() && k.device() == q.device() &&
                  v.device() == q.device() && k.scalar_type() == q.scalar_type() &&
                  v.scalar_type() == q.scalar_type() && q.stride(2) == 1 && k.stride(2) == 1 &&
                  v.stride(2) == 1,
              "Private ALiBi Flash Q/K/V mismatch");
  TORCH_CHECK(slopes.device() == q.device() && slopes.scalar_type() == at::kFloat &&
                  slopes.dim() == 1 && slopes.numel() == q.size(1) && slopes.is_contiguous(),
              "Private ALiBi Flash requires one contiguous FP32 slope per head");
  TORCH_CHECK(cumulative.device() == q.device() && cumulative.scalar_type() == at::kInt &&
                  cumulative.dim() == 1 && cumulative.is_contiguous() && cumulative.numel() > 1,
              "Private ALiBi Flash requires packed int32 cumulative offsets");
  TORCH_CHECK(q.size(0) > 0 && q.size(1) > 0 && max_sequence > 0 && max_sequence <= q.size(0) &&
                  q.size(0) <= std::numeric_limits<int>::max() &&
                  q.size(1) <= std::numeric_limits<int>::max() &&
                  max_sequence <= std::numeric_limits<int>::max() - 127 &&
                  cumulative.numel() - 1 <= std::numeric_limits<int>::max() &&
                  std::isfinite(static_cast<float>(scale)) && scale > 0,
              "Invalid private ALiBi Flash dimensions or scale");
  c10::cuda::CUDAGuard guard(q.device());
  TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major >= 8,
              "Packed Torch FMA requires CUDA compute capability >=8.0");
  auto out = at::empty(q.sizes(), q.options());
  auto lse = at::empty({q.size(1), q.size(0)}, q.options().dtype(at::kFloat));
  tei_torch_flash_alibi64::Flash_fwd_params p{};
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
  p.h = q.size(1);
  p.h_k = k.size(1);
  p.h_h_k_ratio = p.h / p.h_k;
  p.seqlen_q = max_sequence;
  p.seqlen_k = max_sequence;
  p.seqlen_q_rounded = ((max_sequence + 127) / 128) * 128;
  p.seqlen_k_rounded = ((max_sequence + 127) / 128) * 128;
  p.d = q.size(2);
  p.d_rounded = ((p.d + 31) / 32) * 32;
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
  p.alibi_slopes_ptr = slopes.data_ptr<float>();
  p.alibi_slopes_batch_stride = 0;
  tei_torch_flash_alibi64::launch_alibi64(p, at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
}
