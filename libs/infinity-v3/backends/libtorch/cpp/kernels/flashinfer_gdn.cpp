// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
// Translate native packed buffers to FlashInfer's fresh and cached GDN contract.
#include "flashinfer_gdn.h"
#include "models/model.h"
#ifdef TEI_FLASHINFER_GDN
#include "vendor/infinity_gdn/bridge.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif
namespace infinity {
bool flashinfer_gdn_available(const c10::Device& device) {
#ifdef TEI_FLASHINFER_GDN
  return device.is_cuda() &&
         infinity_gdn_supported_v1(at::cuda::getDeviceProperties(device.index())->major);
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
#ifdef TEI_FLASHINFER_GDN
  if (!values.is_cuda() ||
      (values.scalar_type() != at::kHalf && values.scalar_type() != at::kBFloat16))
    return false;
  c10::cuda::CUDAGuard guard(values.device());
  const auto* props = at::cuda::getDeviceProperties(values.get_device());
  if (!infinity_gdn_supported_v1(props->major))
    return false;
  const int dtype = values.scalar_type() == at::kHalf ? 0 : 1;
  const auto tokens = values.size(0);
  auto start = initial.defined()
                   ? initial.transpose(2, 3).contiguous().index_select(0, input.block_cache_sources)
                   : at::Tensor();
  auto [q, k, v, alpha, beta] = flashinfer_gdn_pack_cuda(values, qk, gates, kh, vh);
  auto state = at::empty({input.batch, vh, 128, 128}, values.options().dtype(at::kFloat));
  // FlashInfer owns one TMA slot per SM (four descriptors on SM10x).
  auto workspace = at::empty({props->multiProcessorCount * (props->major == 10 ? 512 : 128)},
                             values.options().dtype(at::kChar));
  const int code = infinity_gdn_forward_v2(dtype,
                                           props->major,
                                           values.get_device(),
                                           at::cuda::getCurrentCUDAStream().stream(),
                                           tokens,
                                           input.batch,
                                           kh,
                                           vh,
                                           props->multiProcessorCount,
                                           q.mutable_data_ptr(),
                                           k.mutable_data_ptr(),
                                           v.mutable_data_ptr(),
                                           alpha.mutable_data_ptr(),
                                           beta.mutable_data_ptr(),
                                           output.mutable_data_ptr(),
                                           input.cumulative.data_ptr(),
                                           state.mutable_data_ptr(),
                                           workspace.mutable_data_ptr(),
                                           workspace.numel(),
                                           start.defined() ? start.mutable_data_ptr() : nullptr);
  TORCH_CHECK(code == 0, "FlashInfer GDN prefill failed: ", infinity_gdn_error_v1());
  if (final_state)
    *final_state = state.transpose(2, 3).contiguous();
  return true;
#else
  return false;
#endif
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
}
