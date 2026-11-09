// SPDX-License-Identifier: MIT
// Reuse the Candle implementation with its original license notices intact.
/// \file
/// Native symbol-scoped instantiation of the portable grouped-expert kernels.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/_grouped_mm.h>

#include "kernels/moe_symbols.h"
#include "vendor/candle_kernels/moe/grouped_gemm.cu"

namespace infinity_moe {
/// Use the shared ATen SM90/SM100 grouped GEMM provider for the gate/up
/// projection when the Hopper-native path is unavailable. Down uses the SM100
/// FP32 provider when built, otherwise the portable FP32 grouped GEMM; both
/// keep the existing weighted BF16 readout and per-route rounding.
int gemma4_moe_forward_grouped_bf16(const at::Tensor& input,
                                    const at::Tensor& logits,
                                    const at::Tensor& gate_up,
                                    const at::Tensor& down,
                                    const at::Tensor& scales,
                                    at::Tensor& output,
                                    void* scratch,
                                    size_t scratch_bytes) {
  auto props = at::cuda::getCurrentDeviceProperties();
  if (!props || (props->major != 9 && props->major != 10))
    return -4;
  const int tokens = static_cast<int>(input.size(0));
  const int hidden = static_cast<int>(input.size(1));
  const int intermediate = static_cast<int>(down.size(2));
  const int experts = static_cast<int>(gate_up.size(0));
  const size_t required = gemma4_moe_workspace_bytes(tokens, hidden, intermediate);
#ifdef INFINITY_TORCH_MOE_SM100
  const bool sm100_down = sm100_down_available(props->major, props->minor);
  const size_t sm100_workspace = sm100_down ? sm100_down_workspace_bytes() : 0;
#else
  const bool sm100_down = false;
  const size_t sm100_workspace = 0;
#endif
  if (!required || scratch_bytes < required + sm100_workspace || !scratch)
    return -1;
  Workspace w{static_cast<char*>(scratch)};
  Buffers b(w, tokens, hidden, intermediate, experts);
  const int slots = tokens * 8;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device()).stream();
  if (cudaMemsetAsync(b.counts, 0, experts * sizeof(int), stream) != cudaSuccess ||
      cudaMemsetAsync(b.cursors, 0, experts * sizeof(int), stream) != cudaSuccess)
    return -2;
  gemma4_moe_route_128_8_f32<<<tokens, 128, 0, stream>>>(
      logits.data_ptr<float>(), scales.data_ptr<float>(), b.ids, b.weights);
  gemma4_moe_count<<<(slots + 255) / 256, 256, 0, stream>>>(b.ids, b.counts, slots);
  gemma4_moe_offsets<<<1, experts, 0, stream>>>(b.counts, b.offsets);
  gemma4_moe_assign<<<(slots + 255) / 256, 256, 0, stream>>>(
      b.ids, b.offsets, b.cursors, b.mapping, slots);
  if ((reinterpret_cast<uintptr_t>(input.data_ptr()) & 15) == 0) {
    gemma4_moe_pack_vec8<<<(uint64_t(slots) * (hidden / 8) + 255) / 256, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input.data_ptr()),
        b.mapping,
        reinterpret_cast<__nv_bfloat16*>(b.packed),
        slots,
        hidden);
  } else {
    gemma4_moe_pack<<<(uint64_t(slots) * hidden + 255) / 256, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input.data_ptr()),
        b.mapping,
        reinterpret_cast<__nv_bfloat16*>(b.packed),
        slots,
        hidden);
  }
  // The shared provider consumes the same expert-sorted packed rows and
  // cumulative ends as the native kernel. Gate/up stays [E,H,2I] via a
  // zero-copy transpose view of the contiguous checkpoint weights.
  auto packed = at::from_blob(b.packed, {slots, hidden}, input.options());
  auto ends = at::from_blob(b.offsets + 1, {experts}, input.options().dtype(at::kInt));
  auto gate_up_output = at::_grouped_mm(packed, gate_up.transpose(-2, -1), ends);
  gemma4_moe_gelu_mul<<<slots, 128, 0, stream>>>(
      static_cast<__nv_bfloat16*>(gate_up_output.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(b.activated),
      slots,
      intermediate);
  int status;
#ifdef INFINITY_TORCH_MOE_SM100
  if (sm100_down) {
    status = gemma4_moe_down_f32_sm100(b.activated,
                                       down.data_ptr(),
                                       b.counts,
                                       b.offsets,
                                       b.expert_output,
                                       experts,
                                       hidden,
                                       intermediate,
                                       static_cast<char*>(scratch) + required,
                                       sm100_workspace,
                                       stream);
  } else
#endif
  {
    setup_problems<<<1, experts, 0, stream>>>(b.counts,
                                              b.offsets,
                                              b.activated,
                                              static_cast<Element*>(down.data_ptr()),
                                              b.expert_output,
                                              intermediate,
                                              hidden,
                                              b.problems,
                                              b.a,
                                              b.b,
                                              b.c_f32,
                                              b.lda,
                                              b.ldb,
                                              b.ldc);
    status =
        grouped_gemm<float>(b.problems, experts, b.a, b.b, b.c_f32, b.lda, b.ldb, b.ldc, stream);
  }
  if (status)
    return status;
  if ((reinterpret_cast<uintptr_t>(output.data_ptr()) & 15) == 0) {
    gemma4_moe_combine_vec8<<<(uint64_t(tokens) * (hidden / 8) + 255) / 256, 256, 0, stream>>>(
        b.expert_output,
        b.mapping,
        b.weights,
        static_cast<__nv_bfloat16*>(output.data_ptr()),
        tokens,
        hidden);
  } else {
    gemma4_moe_unpermute_combine<<<(uint64_t(tokens) * hidden + 255) / 256, 256, 0, stream>>>(
        b.expert_output,
        b.mapping,
        b.weights,
        static_cast<__nv_bfloat16*>(output.data_ptr()),
        tokens,
        hidden);
  }
  return cudaGetLastError() == cudaSuccess ? 0 : -3;
}
} // namespace infinity_moe
