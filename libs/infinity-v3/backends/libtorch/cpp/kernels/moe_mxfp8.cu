// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
/// \file
/// Persistent SM100 MXFP8 routed-expert execution for Gemma4 MoE.

#include "kernels/moe_mxfp8.h"

#include "kernels/gemma_moe_mxfp8_grouped_sm100.h"
#include "kernels/moe_symbols.h"
#include "kernels/mxfp8_fusions.h"
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
constexpr int64_t kPaddedExpertRows = 128;
constexpr int64_t kScaleVector = 32;
constexpr int64_t kScaleBlock = 4;
constexpr int64_t kScaleTile = 128;

size_t checked_mul(size_t lhs, size_t rhs, const char* label) {
  TORCH_CHECK(rhs == 0 || lhs <= static_cast<size_t>(-1) / rhs, label);
  return lhs * rhs;
}

__host__ __device__ inline int64_t mx_scale_offset(int64_t outer, int64_t block, int64_t inner) {
  const int64_t sf_outer = outer / kScaleTile;
  const int64_t inner_major = block & ~(kScaleBlock - 1);
  return (inner_major + sf_outer * inner) * kScaleTile + (outer % kScaleVector) * 16 +
         ((outer % kScaleTile) / kScaleVector) * kScaleBlock + (block % kScaleBlock);
}

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
  int* padded_offsets;
  int* cursors;
  uint32_t* ids;
  uint32_t* mapping;
  float* weights;
  uint8_t* packed;
  uint8_t* packed_scales;
  __nv_bfloat16* gate_up;
  float* expert_output;

  Buffers(Workspace& w,
          int64_t tokens,
          int64_t capacity,
          int64_t hidden,
          int64_t intermediate,
          int64_t experts) {
    const size_t slots = checked_mul(static_cast<size_t>(tokens), kTopK, "MXFP8 routing overflow");
    const size_t rows = static_cast<size_t>(capacity);
    const size_t gate_up_width =
        checked_mul(static_cast<size_t>(intermediate), 2, "MXFP8 intermediate overflow");
    counts = w.take<int>(static_cast<size_t>(experts));
    padded_offsets = w.take<int>(static_cast<size_t>(experts) + 1);
    cursors = w.take<int>(static_cast<size_t>(experts));
    ids = w.take<uint32_t>(slots);
    mapping = w.take<uint32_t>(slots);
    weights = w.take<float>(slots);
    packed =
        w.take<uint8_t>(checked_mul(rows, static_cast<size_t>(hidden), "MXFP8 packing overflow"));
    packed_scales = w.take<uint8_t>(infinity_moe_mxfp8::row_scale_elements(capacity, hidden));
    gate_up = w.take<__nv_bfloat16>(checked_mul(rows, gate_up_width, "MXFP8 gate/up overflow"));
    expert_output =
        w.take<float>(checked_mul(rows, static_cast<size_t>(hidden), "MXFP8 output overflow"));
  }
};

int64_t capacity_rows(int64_t tokens, int64_t experts) {
  const int64_t slots = tokens * kTopK;
  TORCH_CHECK(slots <= INT_MAX - experts * kPaddedExpertRows,
              "Gemma MoE MXFP8 routing capacity exceeds INT_MAX rows");
  return slots + experts * kPaddedExpertRows;
}

size_t provider_bytes() {
  return infinity_moe_mxfp8::grouped_workspace_bytes();
}

size_t required_bytes(
    int64_t tokens, int64_t capacity, int64_t hidden, int64_t intermediate, int64_t experts) {
  Workspace w{nullptr};
  Buffers b(w, tokens, capacity, hidden, intermediate, experts);
  (void)b;
  w.align();
  return w.offset + provider_bytes();
}

__global__ void padded_expert_offsets(const int* counts, int* offsets, int experts) {
  if (threadIdx.x != 0)
    return;
  int offset = 0;
  for (int e = 0; e < experts; ++e) {
    offsets[e] = offset;
    offset += ((counts[e] + kPaddedExpertRows - 1) / kPaddedExpertRows) *
              static_cast<int>(kPaddedExpertRows);
  }
  offsets[experts] = offset;
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

__global__ void pack_mx_scales(const uint8_t* __restrict__ source,
                               const uint32_t* __restrict__ mapping,
                               uint8_t* __restrict__ packed,
                               uint64_t slots,
                               int64_t blocks,
                               int64_t inner) {
  const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= slots * static_cast<uint64_t>(blocks))
    return;
  const uint64_t slot = i / static_cast<uint64_t>(blocks);
  const int64_t block = static_cast<int64_t>(i % static_cast<uint64_t>(blocks));
  const int64_t token = static_cast<int64_t>(slot / kTopK);
  packed[mx_scale_offset(static_cast<int64_t>(mapping[slot]), block, inner)] =
      source[mx_scale_offset(token, block, inner)];
}
} // namespace

GemmaMoeMxfp8::GemmaMoeMxfp8(const at::Tensor& gate_up, const at::Tensor& down) {
  TORCH_CHECK(gate_up.is_cuda() && gate_up.dim() == 3 && gate_up.is_contiguous() &&
                  gate_up.scalar_type() == at::kBFloat16,
              "Gemma MoE MXFP8 gate/up weights must be contiguous CUDA BF16 [E,2I,H]");
  TORCH_CHECK(down.is_cuda() && down.dim() == 3 && down.is_contiguous() &&
                  down.scalar_type() == at::kBFloat16,
              "Gemma MoE MXFP8 down weights must be contiguous CUDA BF16 [E,H,I]");
  TORCH_CHECK(gate_up.device() == down.device(),
              "Gemma MoE MXFP8 expert weights must share one device");
  experts_ = gate_up.size(0);
  const auto gate_up_width = gate_up.size(1);
  hidden_ = gate_up.size(2);
  intermediate_ = down.size(2);
  TORCH_CHECK(experts_ == 128 && hidden_ > 0 && intermediate_ > 0,
              "Gemma MoE MXFP8 requires the native 128-expert/top-8 routing geometry");
  TORCH_CHECK(gate_up_width == 2 * intermediate_,
              "Gemma MoE MXFP8 gate/up width must be twice the down input width");
  TORCH_CHECK(down.sizes() == at::IntArrayRef({experts_, hidden_, intermediate_}),
              "Gemma MoE MXFP8 down weight geometry does not match gate/up");
  TORCH_CHECK(hidden_ % 32 == 0 && intermediate_ % 32 == 0 && hidden_ % 128 == 0 &&
                  gate_up_width % 128 == 0,
              "Gemma MoE MXFP8 requires K multiples of 32 and N multiples of 128");
  TORCH_CHECK(supported(gate_up.device()),
              "Gemma MoE MXFP8 requires the native SM100/SM103 grouped provider");

  c10::cuda::CUDAGuard guard(gate_up.device());
  auto gate = mxfp8_quantize_rows(gate_up.reshape({experts_ * gate_up_width, hidden_}));
  auto down_quant = mxfp8_quantize_rows(down.reshape({experts_ * hidden_, intermediate_}));
  gate_up_ = gate.values.reshape({experts_, gate_up_width, hidden_});
  gate_up_scales_ = gate.scales;
  down_ = down_quant.values.reshape({experts_, hidden_, intermediate_});
  down_scales_ = down_quant.scales;
  TORCH_CHECK(gate_up_.is_contiguous() && gate_up_scales_.is_contiguous() &&
                  down_.is_contiguous() && down_scales_.is_contiguous(),
              "Gemma MoE MXFP8 quantized expert storage must stay contiguous");
  TORCH_CHECK(gate_up_scales_.numel() ==
                  experts_ * infinity_moe_mxfp8::weight_scale_expert_bytes(gate_up_width, hidden_),
              "Gemma MoE MXFP8 gate/up scale layout is not expert-sliceable");
  TORCH_CHECK(down_scales_.numel() ==
                  experts_ * infinity_moe_mxfp8::weight_scale_expert_bytes(hidden_, intermediate_),
              "Gemma MoE MXFP8 down scale layout is not expert-sliceable");
}

bool GemmaMoeMxfp8::defined() const {
  return experts_ > 0 && gate_up_.defined() && gate_up_scales_.defined() && down_.defined() &&
         down_scales_.defined();
}

bool GemmaMoeMxfp8::supported(const c10::Device& device) {
#if defined(INFINITY_TORCH_MOE_MXFP8)
  if (!device.is_cuda())
    return false;
  auto props = device.has_index() ? at::cuda::getDeviceProperties(device.index())
                                  : at::cuda::getCurrentDeviceProperties();
  return props && infinity_moe_mxfp8::grouped_available(props->major, props->minor);
#else
  (void)device;
  return false;
#endif
}

size_t GemmaMoeMxfp8::workspace_bytes(int64_t tokens) const {
  TORCH_CHECK(defined() && tokens >= 0 && tokens <= INT_MAX / kTopK,
              "Gemma MoE MXFP8 workspace requires prepared weights and a valid token count");
  return required_bytes(tokens, capacity_rows(tokens, experts_), hidden_, intermediate_, experts_);
}

at::Tensor GemmaMoeMxfp8::forward(const at::Tensor& input,
                                  const at::Tensor& logits,
                                  const at::Tensor& expert_scales) const {
  TORCH_CHECK(defined() && input.is_cuda() && input.dim() == 2 && input.is_contiguous() &&
                  input.scalar_type() == at::kBFloat16 && input.size(1) == hidden_,
              "Gemma MoE MXFP8 input must be contiguous CUDA BF16 [tokens,hidden]");
  TORCH_CHECK(logits.is_cuda() && logits.dim() == 2 && logits.is_contiguous() &&
                  logits.scalar_type() == at::kFloat && logits.size(0) == input.size(0) &&
                  logits.size(1) == experts_ && logits.device() == input.device(),
              "Gemma MoE MXFP8 logits must be contiguous FP32 [tokens,experts]");
  TORCH_CHECK(expert_scales.is_cuda() && expert_scales.dim() == 1 &&
                  expert_scales.is_contiguous() && expert_scales.scalar_type() == at::kFloat &&
                  expert_scales.numel() == experts_ && expert_scales.device() == input.device(),
              "Gemma MoE MXFP8 learned expert scales must be contiguous FP32 [experts]");
  const auto tokens = input.size(0);
  if (tokens == 0)
    return at::empty_like(input);
  TORCH_CHECK(tokens <= INT_MAX / kTopK, "Gemma MoE MXFP8 token count exceeds the routing limit");
  const auto bytes = workspace_bytes(tokens);
  auto scratch = at::empty({static_cast<int64_t>(bytes)}, input.options().dtype(at::kByte));
  return forward(input, logits, expert_scales, scratch.data_ptr(), bytes);
}

at::Tensor GemmaMoeMxfp8::forward(const at::Tensor& input,
                                  const at::Tensor& logits,
                                  const at::Tensor& expert_scales,
                                  void* scratch,
                                  size_t scratch_bytes) const {
  TORCH_CHECK(defined() && input.is_cuda() && input.dim() == 2 && input.is_contiguous() &&
                  input.scalar_type() == at::kBFloat16 && input.size(1) == hidden_,
              "Gemma MoE MXFP8 input must be contiguous CUDA BF16 [tokens,hidden]");
  TORCH_CHECK(logits.is_cuda() && logits.dim() == 2 && logits.is_contiguous() &&
                  logits.scalar_type() == at::kFloat && logits.size(0) == input.size(0) &&
                  logits.size(1) == experts_ && logits.device() == input.device(),
              "Gemma MoE MXFP8 logits must be contiguous FP32 [tokens,experts]");
  TORCH_CHECK(expert_scales.is_cuda() && expert_scales.dim() == 1 &&
                  expert_scales.is_contiguous() && expert_scales.scalar_type() == at::kFloat &&
                  expert_scales.numel() == experts_ && expert_scales.device() == input.device(),
              "Gemma MoE MXFP8 learned expert scales must be contiguous FP32 [experts]");
  const auto tokens = input.size(0);
  if (tokens == 0)
    return at::empty_like(input);
  TORCH_CHECK(tokens <= INT_MAX / kTopK, "Gemma MoE MXFP8 token count exceeds the routing limit");
  const auto capacity = capacity_rows(tokens, experts_);
  const auto required = required_bytes(tokens, capacity, hidden_, intermediate_, experts_);
  TORCH_CHECK(scratch && scratch_bytes >= required,
              "Gemma MoE MXFP8 scratch is smaller than the required workspace");

  c10::cuda::CUDAGuard guard(input.device());
  Workspace w{static_cast<char*>(scratch)};
  Buffers b(w, tokens, capacity, hidden_, intermediate_, experts_);
  w.align();
  const auto provider_workspace_bytes = provider_bytes();
  TORCH_CHECK(scratch_bytes >= w.offset + provider_workspace_bytes,
              "Gemma MoE MXFP8 scratch is smaller than the required workspace");
  auto* provider_workspace = static_cast<char*>(scratch) + w.offset;
  const auto slots = tokens * kTopK;
  const auto stream = at::cuda::getCurrentCUDAStream(input.get_device()).stream();

  TORCH_CHECK(cudaMemsetAsync(b.counts, 0, experts_ * sizeof(int), stream) == cudaSuccess &&
                  cudaMemsetAsync(b.cursors, 0, experts_ * sizeof(int), stream) == cudaSuccess,
              "Gemma MoE MXFP8 failed to clear device routing counters");

  gemma4_moe_route_128_8_f32<<<static_cast<unsigned>(tokens), 128, 0, stream>>>(
      logits.data_ptr<float>(), expert_scales.data_ptr<float>(), b.ids, b.weights);
  gemma4_moe_count<<<static_cast<unsigned>((slots + 255) / 256), 256, 0, stream>>>(
      b.ids, b.counts, static_cast<int>(slots));
  padded_expert_offsets<<<1, 1, 0, stream>>>(
      b.counts, b.padded_offsets, static_cast<int>(experts_));
  gemma4_moe_assign<<<static_cast<unsigned>((slots + 255) / 256), 256, 0, stream>>>(
      b.ids, b.padded_offsets, b.cursors, b.mapping, static_cast<int>(slots));

  const auto row_bytes = static_cast<uint32_t>(hidden_);
  const auto blocks = hidden_ / kScaleVector;
  const auto inner = static_cast<int64_t>(infinity_moe_mxfp8::row_scale_inner_blocks(hidden_));
  const auto scale_bytes = infinity_moe_mxfp8::row_scale_elements(capacity, hidden_);
  TORCH_CHECK(cudaMemsetAsync(b.packed_scales, 0, scale_bytes, stream) == cudaSuccess,
              "Gemma MoE MXFP8 failed to clear packed scale padding");

  auto input_mx = mxfp8_quantize_rows(input);
  const auto* quantized_ptr = input_mx.values.const_data_ptr();
  if (hidden_ % 16 == 0 && (reinterpret_cast<uintptr_t>(quantized_ptr) & 15) == 0) {
    const uint64_t vectors = uint64_t(slots) * (row_bytes / 16);
    pack_bytes_vec16<<<static_cast<unsigned>((vectors + 255) / 256), 256, 0, stream>>>(
        static_cast<const uint8_t*>(quantized_ptr), b.mapping, b.packed, slots, row_bytes);
  } else {
    const uint64_t bytes = uint64_t(slots) * row_bytes;
    pack_bytes<<<static_cast<unsigned>((bytes + 255) / 256), 256, 0, stream>>>(
        static_cast<const uint8_t*>(quantized_ptr), b.mapping, b.packed, slots, row_bytes);
  }
  const uint64_t scale_items = uint64_t(slots) * static_cast<uint64_t>(blocks);
  pack_mx_scales<<<static_cast<unsigned>((scale_items + 255) / 256), 256, 0, stream>>>(
      static_cast<const uint8_t*>(input_mx.scales.const_data_ptr()),
      b.mapping,
      b.packed_scales,
      slots,
      blocks,
      inner);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto status = infinity_moe_mxfp8::grouped_gemm_bf16(b.packed,
                                                      gate_up_.const_data_ptr(),
                                                      b.packed_scales,
                                                      gate_up_scales_.const_data_ptr(),
                                                      b.counts,
                                                      b.padded_offsets,
                                                      b.gate_up,
                                                      static_cast<int>(experts_),
                                                      static_cast<int>(hidden_),
                                                      static_cast<int>(2 * intermediate_),
                                                      provider_workspace,
                                                      provider_workspace_bytes,
                                                      stream);
  TORCH_CHECK(status == 0, "Gemma MoE MXFP8 gate/up grouped GEMM failed: ", status);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // Fuse the exact MoE GELU/multiply staging into MXFP8 quantization over only
  // the logical routed slots. This skips the BF16 activation matrix round trip
  // and never reads the uninitialized physical padding rows.
  auto gate_up_rows = at::from_blob(
      b.gate_up, {static_cast<int64_t>(capacity), 2 * intermediate_}, input.options());
  auto row_map =
      at::from_blob(b.mapping, {static_cast<int64_t>(slots)}, input.options().dtype(at::kInt));
  auto activated_fp8 = mxfp8_moe_gelu_quantize(gate_up_rows, row_map);
  status = infinity_moe_mxfp8::grouped_gemm_f32(activated_fp8.values.const_data_ptr(),
                                                down_.const_data_ptr(),
                                                activated_fp8.scales.const_data_ptr(),
                                                down_scales_.const_data_ptr(),
                                                b.counts,
                                                b.padded_offsets,
                                                b.expert_output,
                                                static_cast<int>(experts_),
                                                static_cast<int>(intermediate_),
                                                static_cast<int>(hidden_),
                                                provider_workspace,
                                                provider_workspace_bytes,
                                                stream);
  TORCH_CHECK(status == 0, "Gemma MoE MXFP8 down grouped GEMM failed: ", status);
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
