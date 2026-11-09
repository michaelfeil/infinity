// SPDX-License-Identifier: MIT
// Private names isolate the retained kernel implementations.
#pragma once

#include <cstddef>
#include <cuda_runtime.h>

namespace at {
class Tensor;
}

#define gemma4_grouped_gemm_bf16 infinity_gemma4_grouped_gemm_bf16
#define gemma4_moe_assign infinity_gemma4_moe_assign
#define gemma4_moe_combine_8_bf16 infinity_gemma4_moe_combine_8_bf16
#define gemma4_moe_combine_vec8 infinity_gemma4_moe_combine_vec8
#define gemma4_moe_count infinity_gemma4_moe_count
#define gemma4_moe_forward_bf16 infinity_gemma4_moe_forward_bf16
#define gemma4_moe_gelu_mul infinity_gemma4_moe_gelu_mul
#define gemma4_moe_offsets infinity_gemma4_moe_offsets
#define gemma4_moe_pack infinity_gemma4_moe_pack
#define gemma4_moe_pack_vec8 infinity_gemma4_moe_pack_vec8
#define gemma4_moe_route_128_8_f32 infinity_gemma4_moe_route_128_8_f32
#define gemma4_moe_unpermute_combine infinity_gemma4_moe_unpermute_combine
#define gemma4_moe_workspace_bytes infinity_gemma4_moe_workspace_bytes
#define hopper_gemma4_moe_forward_bf16 infinity_hopper_gemma4_moe_forward_bf16
#define hopper_gemma4_moe_workspace_bytes infinity_hopper_gemma4_moe_workspace_bytes
#define hopper_qwen35_moe_forward_bf16 infinity_hopper_qwen35_moe_forward_bf16
#define hopper_qwen35_moe_workspace_bytes infinity_hopper_qwen35_moe_workspace_bytes
#define hopper_qwen3_moe_forward_bf16 infinity_hopper_qwen3_moe_forward_bf16
#define hopper_qwen3_moe_workspace_bytes infinity_hopper_qwen3_moe_workspace_bytes
#define qwen35_moe_forward_bf16 infinity_qwen35_moe_forward_bf16
#define qwen35_moe_route_256_8_f32 infinity_qwen35_moe_route_256_8_f32
#define qwen35_moe_workspace_bytes infinity_qwen35_moe_workspace_bytes
#define qwen3_moe_forward_bf16 infinity_qwen3_moe_forward_bf16
#define qwen3_moe_route_128_8_f32 infinity_qwen3_moe_route_128_8_f32
#define qwen3_moe_silu_mul infinity_qwen3_moe_silu_mul
#define qwen3_moe_workspace_bytes infinity_qwen3_moe_workspace_bytes

namespace infinity_moe {
size_t gemma4_moe_grouped_workspace_bytes(int tokens, int hidden, int intermediate);
int gemma4_moe_forward_grouped_bf16(const at::Tensor& input,
                                    const at::Tensor& logits,
                                    const at::Tensor& gate_up,
                                    const at::Tensor& down,
                                    const at::Tensor& scales,
                                    at::Tensor& output,
                                    void* scratch,
                                    size_t scratch_bytes);
#if defined(INFINITY_TORCH_MOE_SM100)
bool sm100_down_available(int major, int minor);
size_t sm100_down_workspace_bytes();
int gemma4_moe_down_f32_sm100(const void* activated,
                              const void* down,
                              const int* counts,
                              const int* offsets,
                              float* output,
                              int experts,
                              int hidden,
                              int intermediate,
                              void* workspace,
                              size_t workspace_bytes,
                              cudaStream_t stream);
#endif
} // namespace infinity_moe
