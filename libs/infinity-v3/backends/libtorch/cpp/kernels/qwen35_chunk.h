/// \file
/// Chunked and Hopper-prefill gated-delta interfaces for packed Qwen3.5 inputs.

#pragma once
#include "batching/packed_input.h"

namespace infinity {
// Packed BF16 values, normalized/scaled FP32 Q/K, and FP32 decay/beta.
// Prefix snapshots and initial states remain FP32. Chunk tails are exact-length.
/// Write real-token BF16 output with F32 intermediate state; optional final_state is allocated per
/// sequence. Initial states are selected through the input's block-cache source mapping.
void qwen35_chunk_delta_cuda(const at::Tensor& values,
                             const at::Tensor& qk,
                             const at::Tensor& gates,
                             const PackedInput& input,
                             at::Tensor& output,
                             int64_t key_heads,
                             int64_t value_heads,
                             const at::Tensor& initial = {},
                             at::Tensor* final_state = nullptr);
// Hopper-only BF16 prefill; optional final_state retains the FP32 terminal state.
// Defined when built for SM90.
void qwen35_prefill_cuda(const at::Tensor&,
                         const at::Tensor&,
                         const at::Tensor&,
                         const PackedInput&,
                         at::Tensor&,
                         int64_t,
                         int64_t,
                         at::Tensor*);
}
