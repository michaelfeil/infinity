// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
// Translate native packed buffers to FlashInfer's fresh and cached GDN contract.
#include "flashinfer_gdn.h"
#include "models/model.h"
#ifdef INFINITY_FLASHINFER_GDN
#include "vendor/infinity_gdn/bridge.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#endif
namespace infinity {
bool flashinfer_gdn_available(const c10::Device& device) {
#ifdef INFINITY_FLASHINFER_GDN
  return device.is_cuda() &&
         infinity_gdn_supported_v1(at::cuda::getDeviceProperties(device.index())->major);
#else
  return false;
#endif
}

static bool run_packed(const std::array<at::Tensor, 5>& packed,
                       const PackedInput& input,
                       at::Tensor& output,
                       int64_t kh,
                       int64_t vh,
                       at::Tensor* final_state,
                       const at::Tensor& initial) {
#ifdef INFINITY_FLASHINFER_GDN
  const auto& q = packed[0];
  if (!q.is_cuda() || (q.scalar_type() != at::kHalf && q.scalar_type() != at::kBFloat16))
    return false;
  c10::cuda::CUDAGuard guard(q.device());
  const auto* props = at::cuda::getDeviceProperties(q.get_device());
  if (!infinity_gdn_supported_v1(props->major))
    return false;
  const int dtype = q.scalar_type() == at::kHalf ? 0 : 1;
  const auto tokens = q.size(0);
  auto start = at::Tensor();
  if (initial.defined()) {
    // Native kernels store [B,H,K,V]. A captured provider state is the transpose view of
    // provider [B,H,V,K] storage, so transposing it back already yields provider layout.
    auto provider = initial.dim() == 4 && initial.stride(2) == 1
                        ? initial.transpose(2, 3)
                        : initial.transpose(2, 3).contiguous();
    const bool reuse = input.block_cache_sources_identity;
    if (reuse) {
      TORCH_CHECK(input.block_cache_sources.defined() && input.block_cache_sources.dim() == 1 &&
                      input.block_cache_sources.scalar_type() == at::kLong &&
                      input.block_cache_sources.device() == provider.device() &&
                      input.block_cache_sources.numel() == input.batch,
                  "Invalid identity GDN cache sources");
      TORCH_CHECK(provider.size(0) >= input.batch, "Initial GDN state has too few prefix rows");
    }
    start = reuse && provider.is_contiguous() &&
                    reinterpret_cast<uintptr_t>(provider.const_data_ptr()) % 16 == 0
                ? provider
                : provider.index_select(0, input.block_cache_sources);
  }
  auto state = at::empty({input.batch, vh, 128, 128}, q.options().dtype(at::kFloat));
  // FlashInfer owns one TMA slot per SM (four descriptors on SM10x).
  auto workspace = at::empty({props->multiProcessorCount * (props->major == 10 ? 512 : 128)},
                             q.options().dtype(at::kChar));
  const int code = infinity_gdn_forward_v2(dtype,
                                           props->major,
                                           q.get_device(),
                                           at::cuda::getCurrentCUDAStream().stream(),
                                           tokens,
                                           input.batch,
                                           kh,
                                           vh,
                                           props->multiProcessorCount,
                                           packed[0].mutable_data_ptr(),
                                           packed[1].mutable_data_ptr(),
                                           packed[2].mutable_data_ptr(),
                                           packed[3].mutable_data_ptr(),
                                           packed[4].mutable_data_ptr(),
                                           output.mutable_data_ptr(),
                                           input.cumulative.data_ptr(),
                                           state.mutable_data_ptr(),
                                           workspace.mutable_data_ptr(),
                                           workspace.numel(),
                                           start.defined() ? start.mutable_data_ptr() : nullptr);
  TORCH_CHECK(code == 0, "FlashInfer GDN prefill failed: ", infinity_gdn_error_v1());
  // Native logical [B,H,K,V] over provider V/K storage; resume transposes it back without a copy.
  if (final_state)
    *final_state = state.transpose(2, 3);
  return true;
#else
  return false;
#endif
}

static bool run_gdn(const at::Tensor& values,
                    const at::Tensor& qk,
                    const at::Tensor& gates,
                    const PackedInput& input,
                    at::Tensor& output,
                    int64_t kh,
                    int64_t vh,
                    at::Tensor* final_state,
                    const at::Tensor& initial) {
#ifdef INFINITY_FLASHINFER_GDN
  if (!flashinfer_gdn_available(values.device()) ||
      (values.scalar_type() != at::kHalf && values.scalar_type() != at::kBFloat16))
    return false;
  return run_packed(flashinfer_gdn_pack_cuda(values, qk, gates, kh, vh),
                    input,
                    output,
                    kh,
                    vh,
                    final_state,
                    initial);
#else
  return false;
#endif
}

bool flashinfer_gdn_prefill_packed(const std::array<at::Tensor, 5>& packed,
                                   const PackedInput& input,
                                   at::Tensor& output,
                                   int64_t kh,
                                   int64_t vh,
                                   at::Tensor* final_state) {
  return run_packed(packed, input, output, kh, vh, final_state, {});
}

bool flashinfer_gdn_prefill(const at::Tensor& values,
                            const at::Tensor& qk,
                            const at::Tensor& gates,
                            const PackedInput& input,
                            at::Tensor& output,
                            int64_t kh,
                            int64_t vh,
                            at::Tensor* final_state) {
  return run_gdn(values, qk, gates, input, output, kh, vh, final_state, {});
}

bool flashinfer_gdn_resume(const at::Tensor& values,
                           const at::Tensor& qk,
                           const at::Tensor& gates,
                           const PackedInput& input,
                           at::Tensor& output,
                           int64_t kh,
                           int64_t vh,
                           at::Tensor* final_state,
                           const at::Tensor& initial) {
  return run_gdn(values, qk, gates, input, output, kh, vh, final_state, initial);
}

bool flashinfer_gdn_resume_packed(const std::array<at::Tensor, 5>& packed,
                                  const PackedInput& input,
                                  at::Tensor& output,
                                  int64_t kh,
                                  int64_t vh,
                                  at::Tensor* final_state,
                                  const at::Tensor& initial) {
  return run_packed(packed, input, output, kh, vh, final_state, initial);
}
}
