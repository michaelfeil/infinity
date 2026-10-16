#pragma once
#include "batching/packed_input.h"
namespace tei {
// Packed BF16 values, normalized/scaled FP32 Q/K, and FP32 decay/beta.
// Prefix snapshots and initial states remain FP32. Chunk tails are exact-length.
void qwen35_chunk_delta_cuda(const at::Tensor& values,const at::Tensor& qk,
 const at::Tensor& gates,const PackedInput& input,at::Tensor& output,
 int64_t key_heads,int64_t value_heads,const at::Tensor& initial={},at::Tensor* final_state=nullptr);
}
