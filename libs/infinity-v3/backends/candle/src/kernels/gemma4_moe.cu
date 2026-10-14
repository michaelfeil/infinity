// SPDX-License-Identifier: MIT
// Shared 128-expert/top-8 routing. Gemma4 renormalizes and applies learned
// expert scales; Qwen3 optionally renormalizes and has no expert scales.
#include "gemma4_moe_kernels.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

template <bool Qwen, int Experts = 128>
__device__ __forceinline__ void route_128_8_f32(
    const float *logits, const float *expert_scale, uint32_t *ids, float *weights, bool renormalize) {
    const int lane = threadIdx.x;
    const int token = blockIdx.x;
    __shared__ float remaining[Experts];
    __shared__ float reduction[Experts];
    __shared__ uint32_t indices[Experts];
    __shared__ float chosen[8];
    float score = logits[uint64_t(token) * Experts + lane];
    remaining[lane] = score;
    reduction[lane] = score;
    __syncthreads();
    for (int stride = Experts / 2; stride; stride >>= 1) {
        if (lane < stride) reduction[lane] = fmaxf(reduction[lane], reduction[lane + stride]);
        __syncthreads();
    }
    float probability = expf(score - reduction[0]);
    __syncthreads();
    reduction[lane] = probability;
    __syncthreads();
    for (int stride = Experts / 2; stride; stride >>= 1) {
        if (lane < stride) reduction[lane] += reduction[lane + stride];
        __syncthreads();
    }
    probability /= reduction[0];
    __syncthreads();
    if constexpr (Qwen) remaining[lane] = probability;
    __syncthreads();
    for (int rank = 0; rank < 8; ++rank) {
        reduction[lane] = remaining[lane];
        indices[lane] = lane;
        __syncthreads();
        for (int stride = Experts / 2; stride; stride >>= 1) {
            if (lane < stride) {
                const float other = reduction[lane + stride];
                const uint32_t other_id = indices[lane + stride];
                const bool equal = other == reduction[lane];
                // Gemma4 orders signed zeros by its sortable-float key. Qwen3
                // selects softmax probabilities and breaks ties by expert ID.
                const uint32_t other_bits = __float_as_uint(other);
                const uint32_t current_bits = __float_as_uint(reduction[lane]);
                if (other > reduction[lane] || (equal &&
                    ((Qwen && other_id < indices[lane]) ||
                     (!Qwen && (other_bits < current_bits ||
                     (other_bits == current_bits && other_id < indices[lane])))))) {
                    reduction[lane] = other;
                    indices[lane] = other_id;
                }
            }
            __syncthreads();
        }
        const uint32_t winner = indices[0];
        if (lane == winner) {
            ids[token * 8 + rank] = winner;
            chosen[rank] = probability;
            remaining[lane] = -INFINITY;
        }
        __syncthreads();
    }
    if (lane < 8) {
        float sum = 0.f;
        for (int i = 0; i < 8; ++i) sum += chosen[i];
        if constexpr (Qwen) {
            weights[token * 8 + lane] = renormalize ? chosen[lane] / sum : chosen[lane];
        } else {
            weights[token * 8 + lane] = (chosen[lane] / sum) * expert_scale[ids[token * 8 + lane]];
        }
    }
}

extern "C" __global__ void gemma4_moe_route_128_8_f32(
    const float *logits, const float *expert_scale, uint32_t *ids, float *weights) {
    route_128_8_f32<false>(logits, expert_scale, ids, weights, true);
}
extern "C" __global__ void qwen3_moe_route_128_8_f32(
    const float *logits, uint32_t *ids, float *weights, bool renormalize) {
    route_128_8_f32<true>(logits, nullptr, ids, weights, renormalize);
}

extern "C" __global__ void qwen35_moe_route_256_8_f32(
    const float *logits, uint32_t *ids, float *weights, bool renormalize) {
    route_128_8_f32<true, 256>(logits, nullptr, ids, weights, renormalize);
}

// Input layout is [tokens, 8, hidden]. Combine in FP32 and round once.
extern "C" __global__ void gemma4_moe_combine_8_bf16(
    const __nv_bfloat16 *expert_outputs, const float *weights,
    __nv_bfloat16 *output, uint64_t tokens, uint32_t hidden) {
    const uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= tokens * hidden) return;
    const uint64_t token = index / hidden;
    const uint32_t column = index % hidden;
    float sum = 0.f;
    for (int rank = 0; rank < 8; ++rank) {
        sum += __bfloat162float(expert_outputs[(token * 8 + rank) * hidden + column])
            * weights[token * 8 + rank];
    }
    output[index] = __float2bfloat16_rn(sum);
}

extern "C" __global__ void gemma4_moe_count(const uint32_t *ids, int *counts, int slots) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < slots) atomicAdd(counts + ids[i], 1);
}
extern "C" __global__ void gemma4_moe_offsets(const int *counts, int *offsets) {
    const int expert = threadIdx.x;
    int offset = 0;
    for (int i = 0; i < expert; ++i) offset += counts[i];
    offsets[expert] = offset;
    if (expert == blockDim.x - 1) offsets[blockDim.x] = offset + counts[expert];
}
extern "C" __global__ void gemma4_moe_assign(
    const uint32_t *ids, const int *offsets, int *cursors, uint32_t *mapping, int slots) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < slots) {
        const int expert = ids[i];
        mapping[i] = offsets[expert] + atomicAdd(cursors + expert, 1);
    }
}
extern "C" __global__ void gemma4_moe_pack(
    const __nv_bfloat16 *input, const uint32_t *mapping,
    __nv_bfloat16 *packed, uint64_t slots, uint32_t hidden) {
    const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < slots * hidden) {
        const uint64_t slot = i / hidden;
        const uint32_t column = i % hidden;
        packed[uint64_t(mapping[slot]) * hidden + column] = input[(slot / 8) * hidden + column];
    }
}
extern "C" __global__ void gemma4_moe_gelu_mul(
    const __nv_bfloat16 *gate_up, __nv_bfloat16 *output, uint64_t slots, uint32_t width) {
    // One block per expert row avoids per-element division. Aligned rows
    // load eight BF16 values at a time without changing activation rounding.
    const uint64_t row = blockIdx.x;
    if (row >= slots) return;
    if (width % 8 == 0 && (reinterpret_cast<uintptr_t>(gate_up) & 15) == 0 &&
        (reinterpret_cast<uintptr_t>(output) & 15) == 0) {
        for (uint32_t column = threadIdx.x * 8; column < width; column += blockDim.x * 8) {
            __align__(16) __nv_bfloat16 gates[8], ups[8], result[8];
            *reinterpret_cast<uint4 *>(gates) = *reinterpret_cast<const uint4 *>(
                gate_up + row * (2 * width) + column);
            *reinterpret_cast<uint4 *>(ups) = *reinterpret_cast<const uint4 *>(
                gate_up + row * (2 * width) + width + column);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float gate = __bfloat162float(gates[j]);
                const float up = __bfloat162float(ups[j]);
                const float gelu = 0.5f * gate * (1.f + tanhf(0.7978845608028654f *
                    (gate + 0.044715f * gate * gate * gate)));
                const float rounded_gelu = __bfloat162float(__float2bfloat16_rn(gelu));
                result[j] = __float2bfloat16_rn(rounded_gelu * up);
            }
            *reinterpret_cast<uint4 *>(output + row * width + column) =
                *reinterpret_cast<const uint4 *>(result);
        }
        return;
    }
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
        const uint64_t i = row * width + column;
        const float gate = __bfloat162float(gate_up[row * (2 * width) + column]);
        const float up = __bfloat162float(gate_up[row * (2 * width) + width + column]);
        const float gelu = 0.5f * gate * (1.f + tanhf(0.7978845608028654f *
            (gate + 0.044715f * gate * gate * gate)));
        // Match the checkpoint's BF16 GELU followed by BF16 multiplication.
        // vLLM's gelu_tanh_and_mul preserves this intermediate rounding too.
        const float rounded_gelu = __bfloat162float(__float2bfloat16_rn(gelu));
        output[i] = __float2bfloat16_rn(rounded_gelu * up);
    }
}
extern "C" __global__ void qwen3_moe_silu_mul(
    const __nv_bfloat16 *gate_up, __nv_bfloat16 *output, uint64_t slots, uint32_t width) {
    // One block per expert row avoids per-element division. Aligned rows
    // load eight BF16 values at a time without changing activation rounding.
    const uint64_t row = blockIdx.x;
    if (row >= slots) return;
    if (width % 8 == 0 && (reinterpret_cast<uintptr_t>(gate_up) & 15) == 0 &&
        (reinterpret_cast<uintptr_t>(output) & 15) == 0) {
        for (uint32_t column = threadIdx.x * 8; column < width; column += blockDim.x * 8) {
            __align__(16) __nv_bfloat16 gates[8], ups[8], result[8];
            *reinterpret_cast<uint4 *>(gates) = *reinterpret_cast<const uint4 *>(
                gate_up + row * (2 * width) + column);
            *reinterpret_cast<uint4 *>(ups) = *reinterpret_cast<const uint4 *>(
                gate_up + row * (2 * width) + width + column);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float gate = __bfloat162float(gates[j]);
                const float up = __bfloat162float(ups[j]);
                const float silu = gate / (1.f + expf(-gate));
                const float rounded_silu = __bfloat162float(__float2bfloat16_rn(silu));
                result[j] = __float2bfloat16_rn(rounded_silu * up);
            }
            *reinterpret_cast<uint4 *>(output + row * width + column) =
                *reinterpret_cast<const uint4 *>(result);
        }
        return;
    }
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
        const uint64_t i = row * width + column;
        const float gate = __bfloat162float(gate_up[row * (2 * width) + column]);
        const float up = __bfloat162float(gate_up[row * (2 * width) + width + column]);
        const float silu = gate / (1.f + expf(-gate));
        // Match the checkpoint's BF16 SiLU followed by BF16 multiplication.
        // vLLM's silu_and_mul preserves this intermediate rounding too.
        const float rounded_silu = __bfloat162float(__float2bfloat16_rn(silu));
        output[i] = __float2bfloat16_rn(rounded_silu * up);
    }
}
extern "C" __global__ void gemma4_moe_unpermute_combine(
    const float *expert_outputs, const uint32_t *mapping, const float *weights,
    __nv_bfloat16 *output, uint64_t tokens, uint32_t hidden) {
    const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < tokens * hidden) {
        const uint64_t token = i / hidden;
        const uint32_t column = i % hidden;
        float sum = 0.f;
        for (int k = 0; k < 8; ++k) {
            const uint64_t slot = token * 8 + k;
            // vLLM weights the FP32 down-projection accumulator before the
            // per-expert BF16 cast, then sums the eight rounded values in FP32.
            const float weighted = expert_outputs[uint64_t(mapping[slot]) * hidden + column] * weights[slot];
            sum += __bfloat162float(__float2bfloat16_rn(weighted));
        }
        output[i] = __float2bfloat16_rn(sum);
    }
}

// Vectorized variants for the fixed Gemma4 MoE hidden width (2816).
// Inputs and workspace are 16-byte aligned; each row is a multiple of 8 BF16s.
// Preserve the scalar kernels for standalone diagnostics.
extern "C" __global__ void gemma4_moe_pack_vec8(
    const __nv_bfloat16 *input, const uint32_t *mapping,
    __nv_bfloat16 *packed, uint64_t slots, uint32_t hidden) {
    const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint32_t width = hidden / 8;
    if (i < slots * width) {
        const uint64_t slot = i / width;
        const uint32_t column = i % width;
        reinterpret_cast<uint4 *>(packed)[uint64_t(mapping[slot]) * width + column] =
            reinterpret_cast<const uint4 *>(input)[(slot / 8) * width + column];
    }
}

extern "C" __global__ void gemma4_moe_combine_vec8(
    const float *expert_outputs, const uint32_t *mapping, const float *weights,
    __nv_bfloat16 *output, uint64_t tokens, uint32_t hidden) {
    const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint32_t width = hidden / 8;
    if (i < tokens * width) {
        const uint64_t token = i / width;
        const uint32_t column = i % width;
        float sum[8] = {};
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            const uint64_t slot = token * 8 + k;
            const float *p = expert_outputs + uint64_t(mapping[slot]) * hidden + column * 8;
            const float4 a = *reinterpret_cast<const float4 *>(p);
            const float4 b = *reinterpret_cast<const float4 *>(p + 4);
            const float values[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
            const float weight = weights[slot];
            // Keep the same expert order and per-expert BF16 rounding as the
            // scalar path. Only the memory access width changes.
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                sum[j] += __bfloat162float(__float2bfloat16_rn(values[j] * weight));
            }
        }
        __align__(16) __nv_bfloat16 result[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) result[j] = __float2bfloat16_rn(sum[j]);
        reinterpret_cast<uint4 *>(output)[i] = *reinterpret_cast<const uint4 *>(result);
    }
}
