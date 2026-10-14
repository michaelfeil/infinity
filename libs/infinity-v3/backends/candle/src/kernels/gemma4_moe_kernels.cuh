// SPDX-License-Identifier: MIT
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void gemma4_moe_route_128_8_f32(const float *logits,
                                                      const float *expert_scale, uint32_t *ids,
                                                      float *weights);

extern "C" __global__ void gemma4_moe_combine_8_bf16(const __nv_bfloat16 *expert_outputs,
                                                     const float *weights, __nv_bfloat16 *output,
                                                     uint64_t tokens, uint32_t hidden);

extern "C" __global__ void gemma4_moe_count(const uint32_t *ids, int *counts, int slots);

extern "C" __global__ void gemma4_moe_offsets(const int *counts, int *offsets);

extern "C" __global__ void gemma4_moe_assign(const uint32_t *ids, const int *offsets, int *cursors,
                                             uint32_t *mapping, int slots);

extern "C" __global__ void gemma4_moe_pack(const __nv_bfloat16 *input, const uint32_t *mapping,
                                           __nv_bfloat16 *packed, uint64_t slots, uint32_t hidden);

// Launch one block per expert row (slots); columns are traversed within the block.
extern "C" __global__ void gemma4_moe_gelu_mul(const __nv_bfloat16 *gate_up, __nv_bfloat16 *output,
                                               uint64_t slots, uint32_t width);

extern "C" __global__ void gemma4_moe_unpermute_combine(const float *expert_outputs,
                                                        const uint32_t *mapping,
                                                        const float *weights, __nv_bfloat16 *output,
                                                        uint64_t tokens, uint32_t hidden);

extern "C" __global__ void gemma4_moe_pack_vec8(const __nv_bfloat16 *input, const uint32_t *mapping,
                                                __nv_bfloat16 *packed, uint64_t slots,
                                                uint32_t hidden);

extern "C" __global__ void gemma4_moe_combine_vec8(const float *expert_outputs,
                                                   const uint32_t *mapping, const float *weights,
                                                   __nv_bfloat16 *output, uint64_t tokens,
                                                   uint32_t hidden);

extern "C" __global__ void qwen3_moe_route_128_8_f32(
    const float *logits, uint32_t *ids, float *weights, bool renormalize);
extern "C" __global__ void qwen3_moe_silu_mul(
    const __nv_bfloat16 *gate_up, __nv_bfloat16 *output, uint64_t slots, uint32_t width);

extern "C" __global__ void qwen35_moe_route_256_8_f32(const float *, uint32_t *, float *, bool);
