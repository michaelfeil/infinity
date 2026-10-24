// Native adapter to FlashInfer 0.6.13; upstream kernel retains Apache-2.0.
/// \file
/// Hopper prefill adapter for the packed Qwen3.5 gated-delta path.

#include "kernels/qwen35_chunk.h"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <flashinfer/flat/prefill/prefill_kernel_delta_rule_sm90.cuh>

namespace tei {
/// Dispatch SM90 prefill into caller-owned output and assign FP32 state through final_state. The
/// caller supplies a non-null final_state pointer and validates the eligible profile.
void qwen35_prefill_cuda(const at::Tensor& values,
                         const at::Tensor& qk,
                         const at::Tensor& gates,
                         const PackedInput& input,
                         at::Tensor& output,
                         int64_t kh,
                         int64_t vh,
                         at::Tensor* final_state) {
  auto qk3 = qk.view({values.size(0), kh, 256});
  auto q = qk3.narrow(2, 0, 128).to(at::kBFloat16).contiguous(),
       k = qk3.narrow(2, 128, 128).to(at::kBFloat16).contiguous();
  auto v = values.narrow(1, 2 * kh * 128, vh * 128).contiguous();
  auto alpha = gates.narrow(1, 0, vh).contiguous(), beta = gates.narrow(1, vh, vh).contiguous();
  auto cu = input.cumulative.to(at::kLong);
  auto state = at::empty({input.batch, vh, 128, 128}, gates.options());
  int sm = at::cuda::getDeviceProperties(values.get_device())->multiProcessorCount;
  auto workspace = at::empty({sm * 128}, values.options().dtype(at::kByte));
  auto bf = [](const at::Tensor& t) {
    return reinterpret_cast<const __nv_bfloat16*>(t.const_data_ptr());
  };
  flat::launch_delta_rule_prefill_kernel_gbai<true,
                                              true,
                                              true,
                                              false,
                                              false,
                                              cutlass::arch::Sm90,
                                              __nv_bfloat16,
                                              __nv_bfloat16,
                                              float>(
      at::cuda::getCurrentCUDAStream(),
      reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr()),
      state.mutable_data_ptr<float>(),
      bf(q),
      bf(k),
      bf(v),
      nullptr,
      alpha.const_data_ptr<float>(),
      beta.const_data_ptr<float>(),
      cu.const_data_ptr<int64_t>(),
      workspace.mutable_data_ptr<uint8_t>(),
      input.batch,
      kh,
      kh,
      vh,
      vh,
      128,
      values.size(0),
      1.f,
      sm,
      nullptr,
      nullptr,
      0);
  *final_state = state.transpose(2, 3).contiguous();
}
}
