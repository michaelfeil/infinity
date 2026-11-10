// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
/// \file
/// Persistent SM100 FP8 routed-expert execution for Gemma4 MoE.

#include "kernels/moe_fp8.h"

#include "helpers/fp8_linear.h"
#include "kernels/gemma_moe_fp8_grouped_sm100.h"
#include "kernels/moe_symbols.h"
#include "vendor/candle_kernels/kernels/gemma4_moe_kernels.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <climits>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace infinity {
namespace {
constexpr int64_t kTopK = 8;

size_t checked_mul(size_t lhs, size_t rhs, const char* label) {
  TORCH_CHECK(rhs == 0 || lhs <= static_cast<size_t>(-1) / rhs, label);
  return lhs * rhs;
}

/// Device-side routing/packing buffers; every allocation stays in one
/// caller-owned byte workspace so capture does not need fresh host metadata.
struct Workspace {
  char* base;
  size_t offset = 0;

  template <typename T> T* take(size_t count) {
    offset = (offset + 255) & ~size_t(255);
    auto* result = base ? reinterpret_cast<T*>(base + offset) : nullptr;
    offset += sizeof(T) * count;
    return result;
  }

  void align() {
    offset = (offset + 255) & ~size_t(255);
  }
};

struct Buffers {
  int* counts;
  int* offsets;
  int* cursors;
  uint32_t* ids;
  uint32_t* mapping;
  float* weights;
  uint8_t* packed;
  float* packed_scales;
  __nv_bfloat16* gate_up;
  __nv_bfloat16* activated;
  float* expert_output;

  Buffers(Workspace& w, int64_t tokens, int64_t hidden, int64_t intermediate, int64_t experts) {
    const size_t slots = checked_mul(static_cast<size_t>(tokens), kTopK, "FP8 routing overflow");
    const size_t gate_up_width =
        checked_mul(static_cast<size_t>(intermediate), 2, "FP8 intermediate overflow");
    counts = w.take<int>(static_cast<size_t>(experts));
    offsets = w.take<int>(static_cast<size_t>(experts) + 1);
    cursors = w.take<int>(static_cast<size_t>(experts));
    ids = w.take<uint32_t>(slots);
    mapping = w.take<uint32_t>(slots);
    weights = w.take<float>(slots);
    packed =
        w.take<uint8_t>(checked_mul(slots, static_cast<size_t>(hidden), "FP8 packing overflow"));
    packed_scales = w.take<float>(slots);
    gate_up = w.take<__nv_bfloat16>(checked_mul(slots, gate_up_width, "FP8 gate/up overflow"));
    activated = w.take<__nv_bfloat16>(
        checked_mul(slots, static_cast<size_t>(intermediate), "FP8 activation overflow"));
    expert_output =
        w.take<float>(checked_mul(slots, static_cast<size_t>(hidden), "FP8 output overflow"));
  }
};

size_t provider_bytes() {
  return infinity_moe_fp8::grouped_workspace_bytes();
}

size_t required_bytes(int64_t tokens, int64_t hidden, int64_t intermediate, int64_t experts) {
  Workspace w{nullptr};
  Buffers b(w, tokens, hidden, intermediate, experts);
  (void)b;
  w.align();
  return w.offset + provider_bytes();
}

__global__ void pack_bytes_vec16(const uint8_t* __restrict__ source,
                                 const uint32_t* __restrict__ mapping,
                                 uint8_t* __restrict__ packed,
                                 uint64_t slots,
                                 uint32_t row_bytes) {
  const uint64_t vectors = row_bytes / 16;
  const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= slots * vectors)
    return;
  const uint64_t slot = i / vectors;
  const uint64_t vector = i % vectors;
  reinterpret_cast<uint4*>(packed)[uint64_t(mapping[slot]) * vectors + vector] =
      reinterpret_cast<const uint4*>(source)[(slot / kTopK) * vectors + vector];
}

__global__ void pack_bytes(const uint8_t* __restrict__ source,
                           const uint32_t* __restrict__ mapping,
                           uint8_t* __restrict__ packed,
                           uint64_t slots,
                           uint32_t row_bytes) {
  const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= slots * row_bytes)
    return;
  const uint64_t slot = i / row_bytes;
  const uint64_t byte = i % row_bytes;
  packed[uint64_t(mapping[slot]) * row_bytes + byte] = source[(slot / kTopK) * row_bytes + byte];
}

__global__ void copy_token_scales(const float* __restrict__ source,
                                  const uint32_t* __restrict__ mapping,
                                  float* __restrict__ packed,
                                  uint64_t slots) {
  const uint64_t slot = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (slot < slots)
    packed[mapping[slot]] = source[slot / kTopK];
}
} // namespace

GemmaMoeFp8::GemmaMoeFp8(const at::Tensor& gate_up, const at::Tensor& down) {
  TORCH_CHECK(gate_up.is_cuda() && gate_up.dim() == 3 && gate_up.is_contiguous() &&
                  gate_up.scalar_type() == at::kBFloat16,
              "Gemma MoE FP8 gate/up weights must be contiguous CUDA BF16 [E,2I,H]");
  TORCH_CHECK(down.is_cuda() && down.dim() == 3 && down.is_contiguous() &&
                  down.scalar_type() == at::kBFloat16,
              "Gemma MoE FP8 down weights must be contiguous CUDA BF16 [E,H,I]");
  TORCH_CHECK(gate_up.device() == down.device(),
              "Gemma MoE FP8 expert weights must share one device");
  experts_ = gate_up.size(0);
  const auto gate_up_width = gate_up.size(1);
  hidden_ = gate_up.size(2);
  intermediate_ = down.size(2);
  TORCH_CHECK(experts_ == 128 && hidden_ > 0 && intermediate_ > 0,
              "Gemma MoE FP8 requires the native 128-expert/top-8 routing geometry");
  TORCH_CHECK(gate_up_width == 2 * intermediate_,
              "Gemma MoE FP8 gate/up width must be twice the down input width");
  TORCH_CHECK(down.sizes() == at::IntArrayRef({experts_, hidden_, intermediate_}),
              "Gemma MoE FP8 down weight geometry does not match gate/up");
  TORCH_CHECK(hidden_ % 64 == 0 && intermediate_ % 64 == 0 && gate_up_width % 8 == 0,
              "Gemma MoE FP8 grouped kernels require hidden/intermediate multiples of 64");
  TORCH_CHECK(supported(gate_up.device()),
              "Gemma MoE FP8 requires the native SM100/SM103 grouped provider");

  c10::cuda::CUDAGuard guard(gate_up.device());
  auto gate = fp8_quantize_rows(gate_up.reshape({experts_ * gate_up_width, hidden_}));
  auto down_quant = fp8_quantize_rows(down.reshape({experts_ * hidden_, intermediate_}));
  gate_up_ = gate.first.reshape({experts_, gate_up_width, hidden_});
  gate_up_scales_ = gate.second.reshape({experts_, gate_up_width});
  down_ = down_quant.first.reshape({experts_, hidden_, intermediate_});
  down_scales_ = down_quant.second.reshape({experts_, hidden_});
  TORCH_CHECK(gate_up_.is_contiguous() && gate_up_scales_.is_contiguous() &&
                  down_.is_contiguous() && down_scales_.is_contiguous(),
              "Gemma MoE FP8 quantized expert storage must stay contiguous");
}

bool GemmaMoeFp8::defined() const {
  return experts_ > 0 && gate_up_.defined() && gate_up_scales_.defined() && down_.defined() &&
         down_scales_.defined();
}

bool GemmaMoeFp8::supported(const c10::Device& device) {
#if defined(INFINITY_TORCH_MOE_FP8)
  if (!device.is_cuda())
    return false;
  auto props = device.has_index() ? at::cuda::getDeviceProperties(device.index())
                                  : at::cuda::getCurrentDeviceProperties();
  return props && infinity_moe_fp8::grouped_available(props->major, props->minor);
#else
  (void)device;
  return false;
#endif
}

size_t GemmaMoeFp8::workspace_bytes(int64_t tokens) const {
  TORCH_CHECK(defined() && tokens >= 0 && tokens <= INT_MAX / kTopK,
              "Gemma MoE FP8 workspace requires prepared weights and a valid token count");
  return required_bytes(tokens, hidden_, intermediate_, experts_);
}

at::Tensor GemmaMoeFp8::forward(const at::Tensor& input,
                                const at::Tensor& logits,
                                const at::Tensor& expert_scales) const {
  TORCH_CHECK(defined() && input.is_cuda() && input.dim() == 2 && input.is_contiguous() &&
                  input.scalar_type() == at::kBFloat16 && input.size(1) == hidden_,
              "Gemma MoE FP8 input must be contiguous CUDA BF16 [tokens,hidden]");
  TORCH_CHECK(logits.is_cuda() && logits.dim() == 2 && logits.is_contiguous() &&
                  logits.scalar_type() == at::kFloat && logits.size(0) == input.size(0) &&
                  logits.size(1) == experts_ && logits.device() == input.device(),
              "Gemma MoE FP8 logits must be contiguous FP32 [tokens,experts]");
  TORCH_CHECK(expert_scales.is_cuda() && expert_scales.dim() == 1 &&
                  expert_scales.is_contiguous() && expert_scales.scalar_type() == at::kFloat &&
                  expert_scales.numel() == experts_ && expert_scales.device() == input.device(),
              "Gemma MoE FP8 learned expert scales must be contiguous FP32 [experts]");
  const auto tokens = input.size(0);
  if (tokens == 0)
    return at::empty_like(input);
  TORCH_CHECK(tokens <= INT_MAX / kTopK, "Gemma MoE FP8 token count exceeds the routing limit");
  const auto bytes = workspace_bytes(tokens);
  auto scratch = at::empty({static_cast<int64_t>(bytes)}, input.options().dtype(at::kByte));
  return forward(input, logits, expert_scales, scratch.data_ptr(), bytes);
}

at::Tensor GemmaMoeFp8::forward(const at::Tensor& input,
                                const at::Tensor& logits,
                                const at::Tensor& expert_scales,
                                void* scratch,
                                size_t scratch_bytes) const {
  TORCH_CHECK(defined() && input.is_cuda() && input.dim() == 2 && input.is_contiguous() &&
                  input.scalar_type() == at::kBFloat16 && input.size(1) == hidden_,
              "Gemma MoE FP8 input must be contiguous CUDA BF16 [tokens,hidden]");
  TORCH_CHECK(logits.is_cuda() && logits.dim() == 2 && logits.is_contiguous() &&
                  logits.scalar_type() == at::kFloat && logits.size(0) == input.size(0) &&
                  logits.size(1) == experts_ && logits.device() == input.device(),
              "Gemma MoE FP8 logits must be contiguous FP32 [tokens,experts]");
  TORCH_CHECK(expert_scales.is_cuda() && expert_scales.dim() == 1 &&
                  expert_scales.is_contiguous() && expert_scales.scalar_type() == at::kFloat &&
                  expert_scales.numel() == experts_ && expert_scales.device() == input.device(),
              "Gemma MoE FP8 learned expert scales must be contiguous FP32 [experts]");
  const auto tokens = input.size(0);
  if (tokens == 0)
    return at::empty_like(input);
  TORCH_CHECK(tokens <= INT_MAX / kTopK, "Gemma MoE FP8 token count exceeds the routing limit");

  c10::cuda::CUDAGuard guard(input.device());
  Workspace w{static_cast<char*>(scratch)};
  Buffers b(w, tokens, hidden_, intermediate_, experts_);
  w.align();
  const auto provider_workspace_bytes = provider_bytes();
  TORCH_CHECK(scratch && scratch_bytes >= w.offset + provider_workspace_bytes,
              "Gemma MoE FP8 scratch is smaller than the required workspace");
  auto* provider_workspace = static_cast<char*>(scratch) + w.offset;
  const auto slots = tokens * kTopK;
  const auto stream = at::cuda::getCurrentCUDAStream(input.get_device()).stream();

  TORCH_CHECK(cudaMemsetAsync(b.counts, 0, experts_ * sizeof(int), stream) == cudaSuccess &&
                  cudaMemsetAsync(b.cursors, 0, experts_ * sizeof(int), stream) == cudaSuccess,
              "Gemma MoE FP8 failed to clear device routing counters");

  gemma4_moe_route_128_8_f32<<<static_cast<unsigned>(tokens), 128, 0, stream>>>(
      logits.data_ptr<float>(), expert_scales.data_ptr<float>(), b.ids, b.weights);
  gemma4_moe_count<<<static_cast<unsigned>((slots + 255) / 256), 256, 0, stream>>>(
      b.ids, b.counts, static_cast<int>(slots));
  gemma4_moe_offsets<<<1, static_cast<unsigned>(experts_), 0, stream>>>(b.counts, b.offsets);
  gemma4_moe_assign<<<static_cast<unsigned>((slots + 255) / 256), 256, 0, stream>>>(
      b.ids, b.offsets, b.cursors, b.mapping, static_cast<int>(slots));

  const auto* input_ptr = reinterpret_cast<const __nv_bfloat16*>(input.const_data_ptr());
  at::Tensor aligned_input;
  if ((reinterpret_cast<uintptr_t>(input_ptr) & 15) != 0) {
    aligned_input = at::empty(input.sizes(), input.options());
    aligned_input.copy_(input);
  }
  auto input_fp8 = fp8_quantize_rows(aligned_input.defined() ? aligned_input : input);

  const auto row_bytes = static_cast<uint32_t>(hidden_);
  const auto* quantized_ptr = input_fp8.first.const_data_ptr();
  if (hidden_ % 16 == 0 && (reinterpret_cast<uintptr_t>(quantized_ptr) & 15) == 0) {
    const uint64_t vectors = uint64_t(slots) * (row_bytes / 16);
    pack_bytes_vec16<<<static_cast<unsigned>((vectors + 255) / 256), 256, 0, stream>>>(
        static_cast<const uint8_t*>(quantized_ptr), b.mapping, b.packed, slots, row_bytes);
  } else {
    const uint64_t bytes = uint64_t(slots) * row_bytes;
    pack_bytes<<<static_cast<unsigned>((bytes + 255) / 256), 256, 0, stream>>>(
        static_cast<const uint8_t*>(quantized_ptr), b.mapping, b.packed, slots, row_bytes);
  }
  copy_token_scales<<<static_cast<unsigned>((slots + 255) / 256), 256, 0, stream>>>(
      input_fp8.second.const_data_ptr<float>(), b.mapping, b.packed_scales, slots);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto status = infinity_moe_fp8::grouped_gemm_bf16(b.packed,
                                                    gate_up_.const_data_ptr(),
                                                    b.packed_scales,
                                                    gate_up_scales_.const_data_ptr<float>(),
                                                    b.counts,
                                                    b.offsets,
                                                    b.gate_up,
                                                    static_cast<int>(experts_),
                                                    static_cast<int>(hidden_),
                                                    static_cast<int>(2 * intermediate_),
                                                    provider_workspace,
                                                    provider_workspace_bytes,
                                                    stream);
  TORCH_CHECK(status == 0, "Gemma MoE FP8 gate/up grouped GEMM failed: ", status);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  gemma4_moe_gelu_mul<<<static_cast<unsigned>(slots), 128, 0, stream>>>(
      b.gate_up, b.activated, slots, static_cast<uint32_t>(intermediate_));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto activated_rows = at::from_blob(b.activated, {slots, intermediate_}, input.options());
  auto activated_fp8 = fp8_quantize_rows(activated_rows);
  status = infinity_moe_fp8::grouped_gemm_f32(activated_fp8.first.const_data_ptr(),
                                              down_.const_data_ptr(),
                                              activated_fp8.second.data_ptr<float>(),
                                              down_scales_.const_data_ptr<float>(),
                                              b.counts,
                                              b.offsets,
                                              b.expert_output,
                                              static_cast<int>(experts_),
                                              static_cast<int>(intermediate_),
                                              static_cast<int>(hidden_),
                                              provider_workspace,
                                              provider_workspace_bytes,
                                              stream);
  TORCH_CHECK(status == 0, "Gemma MoE FP8 down grouped GEMM failed: ", status);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto output = at::empty_like(input);
  if (hidden_ % 8 == 0 && (reinterpret_cast<uintptr_t>(output.const_data_ptr()) & 15) == 0) {
    gemma4_moe_combine_vec8<<<static_cast<unsigned>((tokens * (hidden_ / 8) + 255) / 256),
                              256,
                              0,
                              stream>>>(b.expert_output,
                                        b.mapping,
                                        b.weights,
                                        static_cast<__nv_bfloat16*>(output.mutable_data_ptr()),
                                        tokens,
                                        static_cast<uint32_t>(hidden_));
  } else {
    gemma4_moe_unpermute_combine<<<static_cast<unsigned>((tokens * hidden_ + 255) / 256),
                                   256,
                                   0,
                                   stream>>>(b.expert_output,
                                             b.mapping,
                                             b.weights,
                                             static_cast<__nv_bfloat16*>(output.mutable_data_ptr()),
                                             tokens,
                                             static_cast<uint32_t>(hidden_));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
} // namespace infinity
