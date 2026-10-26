// Adapter for the pinned native Torch Flash source in vendor/torch_flash_attention.
// Upstream kernel source and its full BSD-3-Clause notices remain unmodified.
#include "kernels/torch_flash_fma.h"
#include <cuda_runtime_api.h>
// Preserve the fork's fused softmax arithmetic without altering upstream headers.
#ifdef UNFUSE_FMA
#undef UNFUSE_FMA
#endif
#define FLASH_NAMESPACE tei_torch_flash
#include "vendor/torch_flash_attention/flash.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>
#include <limits>

namespace tei_torch_flash {
void launch_alibi32(Flash_fwd_params&, cudaStream_t);
void launch_72(Flash_fwd_params&, cudaStream_t);
void launch_64(Flash_fwd_params&, cudaStream_t);
void launch_128(Flash_fwd_params&, cudaStream_t);
void launch_256(Flash_fwd_params&, cudaStream_t);
void launch_512(Flash_fwd_params&, cudaStream_t);
}

namespace tei_torch_flash_unfused {
void launch(const void*, size_t, cudaStream_t);
}

namespace infinity {
bool torch_flash_fma_supported(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v) {
  return torch_flash_fma_supported(q, k, v, false, -1, -1);
}

bool torch_flash_fma_supported(const at::Tensor& q,
                               const at::Tensor& k,
                               const at::Tensor& v,
                               bool causal,
                               int64_t window_left,
                               int64_t window_right) {
  if (!(q.is_cuda() && q.dim() == 3 && k.dim() == 3 && v.dim() == 3 &&
        (q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16) &&
        k.size(2) == q.size(2) && v.size(2) == q.size(2) && q.stride(2) == 1 && k.stride(2) == 1 &&
        v.stride(2) == 1))
    return false;
  // Head widths correspond to separately compiled launch units. The width-72 BF16/noncausal
  // profile has its own launch; advertising another dtype or mask also requires checking that
  // unit's template dispatch.
  if (q.size(2) == 72) {
    if (q.scalar_type() != at::kBFloat16 || causal || window_left != -1 || window_right != -1 ||
        k.device() != q.device() || v.device() != q.device() ||
        k.scalar_type() != q.scalar_type() || v.scalar_type() != q.scalar_type())
      return false;
    const auto* properties = at::cuda::getDeviceProperties(q.device().index());
    return properties->major >= 8 && properties->sharedMemPerBlockOptin >= 48 * 1024;
  }
  if (q.size(2) == 64 || q.size(2) == 128 || q.size(2) == 256)
    return at::cuda::getDeviceProperties(q.device().index())->major >= 8;
  // The wide-head branch is a separate exported specialization, not a generic extension of the
  // <=256-head path. Preserve its dtype/mask constraints and dynamic-shared-memory checks together.
  if (q.size(2) != 512 || window_left != -1 || window_right != -1)
    return false;
  // The wide-head FP16/BF16 launch requires 128 KiB of dynamic shared memory.
  // Devices with insufficient shared memory retain their efficient fallback.
  const auto* properties = at::cuda::getDeviceProperties(q.device().index());
  return properties->major >= 8 && properties->sharedMemPerBlockOptin >= 128 * 1024;
}

/// Translate packed Q/K/V metadata to the private Flash launch contract and allocate independent
/// output. Rounded launch dimensions describe tile bounds only; tensor
/// storage still contains exact token rows.
static at::Tensor torch_flash_fma_impl(const at::Tensor& q,
                                       const at::Tensor& k,
                                       const at::Tensor& v,
                                       const at::Tensor& cumulative_q,
                                       const at::Tensor& cumulative_k,
                                       int64_t max_queries,
                                       int64_t max_keys,
                                       double scale,
                                       bool causal,
                                       int64_t window_left,
                                       int64_t window_right,
                                       bool normalize_oversized_windows,
                                       bool unfused = false) {
  TORCH_CHECK(torch_flash_fma_supported(q, k, v, causal, window_left, window_right),
              "Unsupported packed Torch FMA attention shape or dtype");
  TORCH_CHECK(k.device() == q.device() && v.device() == q.device() &&
                  k.scalar_type() == q.scalar_type() && v.scalar_type() == q.scalar_type() &&
                  v.sizes() == k.sizes() && k.size(1) > 0 && q.size(1) > 0 &&
                  q.size(1) % k.size(1) == 0,
              "Packed Torch FMA Q/K/V mismatch");
  for (const auto& cumulative : {cumulative_q, cumulative_k})
    TORCH_CHECK(cumulative.device() == q.device() && cumulative.scalar_type() == at::kInt &&
                    cumulative.dim() == 1 && cumulative.is_contiguous() && cumulative.numel() > 1,
                "Packed Torch FMA requires contiguous int32 cumulative offsets");
  TORCH_CHECK(cumulative_q.numel() == cumulative_k.numel(),
              "Packed Torch FMA query/key batch mismatch");
  TORCH_CHECK(q.size(0) > 0 && k.size(0) > 0 && max_queries > 0 && max_keys > 0 &&
                  max_queries <= q.size(0) && max_keys <= k.size(0) &&
                  q.size(0) <= std::numeric_limits<int>::max() &&
                  k.size(0) <= std::numeric_limits<int>::max() &&
                  max_queries <= std::numeric_limits<int>::max() - 127 &&
                  max_keys <= std::numeric_limits<int>::max() - 127 &&
                  cumulative_q.numel() - 1 <= std::numeric_limits<int>::max() &&
                  q.size(1) <= std::numeric_limits<int>::max() && window_left >= -1 &&
                  window_right >= -1 && window_left <= std::numeric_limits<int>::max() &&
                  window_right <= std::numeric_limits<int>::max() &&
                  std::isfinite(static_cast<float>(scale)),
              "Invalid packed Torch FMA dimensions or attention scale");
  // Match the original variable-length API for image-query calls. Keep the
  // existing same-span wrapper's proven window metadata unchanged.
  if (normalize_oversized_windows) {
    if (window_left > max_keys)
      window_left = -1;
    if (window_right > max_keys)
      window_right = -1;
  }
  c10::cuda::CUDAGuard guard(q.device());
  TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major >= 8,
              "Packed Torch FMA requires CUDA compute capability >=8.0");
  auto out = at::empty(q.sizes(), q.options());
  auto lse = at::empty({q.size(1), q.size(0)}, q.options().dtype(at::kFloat));
  tei_torch_flash::Flash_fwd_params p{};
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
  p.b = cumulative_q.numel() - 1;
  p.h = q.size(1);
  p.h_k = k.size(1);
  p.h_h_k_ratio = p.h / p.h_k;
  p.seqlen_q = max_queries;
  p.seqlen_k = max_keys;
  p.seqlen_q_rounded = ((max_queries + 127) / 128) * 128;
  p.seqlen_k_rounded = ((max_keys + 127) / 128) * 128;
  p.d = q.size(2);
  p.d_rounded = ((p.d + 31) / 32) * 32;
  p.total_q = q.size(0);
  p.cu_seqlens_q = cumulative_q.data_ptr<int>();
  p.cu_seqlens_k = cumulative_k.data_ptr<int>();
  p.softmax_lse_ptr = lse.data_ptr();
  p.unpadded_lse = true;
  p.is_seqlens_k_cumulative = true;
  p.scale_softmax = static_cast<float>(scale);
  p.scale_softmax_log2 = p.scale_softmax * M_LOG2E;
  p.p_dropout = 1.f;
  p.p_dropout_in_uint8_t = 255;
  p.rp_dropout = 1.f;
  p.scale_softmax_rp_dropout = p.scale_softmax;
  if (causal)
    window_right = 0;
  p.is_causal = window_left < 0 && window_right == 0;
  if (window_left < 0 && window_right >= 0)
    window_left = max_keys;
  if (window_left >= 0 && window_right < 0)
    window_right = max_keys;
  p.window_size_left = window_left;
  p.window_size_right = window_right;
  p.is_bf16 = q.scalar_type() == at::kBFloat16;
  p.num_splits = 1;
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (unfused)
    tei_torch_flash_unfused::launch(&p, sizeof(p), stream);
  else if (p.d == 64)
    tei_torch_flash::launch_64(p, stream);
  else if (p.d == 72)
    tei_torch_flash::launch_72(p, stream);
  else if (p.d == 128)
    tei_torch_flash::launch_128(p, stream);
  else if (p.d == 256)
    tei_torch_flash::launch_256(p, stream);
  else
    tei_torch_flash::launch_512(p, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor torch_flash_fma(const at::Tensor& q,
                           const at::Tensor& k,
                           const at::Tensor& v,
                           const at::Tensor& cumulative,
                           int64_t max_sequence,
                           double scale,
                           bool causal,
                           int64_t window_left,
                           int64_t window_right) {
  TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && q.size(0) == k.size(0),
              "Packed Torch FMA same-span Q/K mismatch");
  return torch_flash_fma_impl(q,
                              k,
                              v,
                              cumulative,
                              cumulative,
                              max_sequence,
                              max_sequence,
                              scale,
                              causal,
                              window_left,
                              window_right,
                              false);
}

at::Tensor torch_flash_fma_varlen(const at::Tensor& q,
                                  const at::Tensor& k,
                                  const at::Tensor& v,
                                  const at::Tensor& cumulative_q,
                                  const at::Tensor& cumulative_k,
                                  int64_t max_queries,
                                  int64_t max_keys,
                                  double scale,
                                  bool causal,
                                  int64_t window_left,
                                  int64_t window_right) {
  return torch_flash_fma_impl(q,
                              k,
                              v,
                              cumulative_q,
                              cumulative_k,
                              max_queries,
                              max_keys,
                              scale,
                              causal,
                              window_left,
                              window_right,
                              true);
}

at::Tensor torch_flash_unfused_varlen(const at::Tensor& q,
                                      const at::Tensor& k,
                                      const at::Tensor& v,
                                      const at::Tensor& cumulative_q,
                                      const at::Tensor& cumulative_k,
                                      int64_t max_queries,
                                      int64_t max_keys,
                                      double scale) {
  TORCH_CHECK(q.dim() == 3 && (q.size(2) == 64 || q.size(2) == 128) && max_queries == 1,
              "Default Torch prefix attention requires one global head64/128 query per sequence");
  return torch_flash_fma_impl(
      q, k, v, cumulative_q, cumulative_k, max_queries, max_keys, scale, true, -1, -1, true, true);
}

bool torch_flash_fma_alibi32_supported(c10::Device device, at::ScalarType dtype, int64_t head_dim) {
  return device.is_cuda() && dtype == at::kHalf && head_dim == 32 &&
         at::cuda::getDeviceProperties(device.index())->major >= 8;
}

at::Tensor torch_flash_fma_alibi32(const at::Tensor& q,
                                   const at::Tensor& k,
                                   const at::Tensor& v,
                                   const at::Tensor& cumulative,
                                   const at::Tensor& slopes,
                                   int64_t max_sequence,
                                   double scale) {
  TORCH_CHECK(q.dim() == 3 &&
                  torch_flash_fma_alibi32_supported(q.device(), q.scalar_type(), q.size(2)),
              "Private ALiBi Flash requires CUDA FP16 head32 on SM80 or newer");
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
  tei_torch_flash::Flash_fwd_params p{};
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
  p.is_bf16 = false;
  p.num_splits = 1;
  p.alibi_slopes_ptr = slopes.data_ptr<float>();
  p.alibi_slopes_batch_stride = 0;
  tei_torch_flash::launch_alibi32(p, at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
}
