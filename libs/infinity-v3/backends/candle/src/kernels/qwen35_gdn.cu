// SPDX-License-Identifier: MIT
// Stateless varlen Qwen3.5 Gated DeltaNet prefill. Each sequence starts from zero.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
__device__ float gdn_sum(float x) {
    for (int d = 16; d; d >>= 1)
        x += __shfl_down_sync(0xffffffff, x, d);
    return __shfl_sync(0xffffffff, x, 0);
}
extern "C" __global__ void qwen35_conv_bf16(const __nv_bfloat16 *x, const __nv_bfloat16 *weight,
                                            const uint32_t *cu, __nv_bfloat16 *out, int tokens,
                                            int channels, int kernel, int sequences) {
    const uint64_t idx = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= uint64_t(tokens) * channels)
        return;
    int t = idx / channels, c = idx % channels, lo = 0, hi = sequences;
    while (lo + 1 < hi) {
        int m = (lo + hi) / 2;
        if (cu[m] <= t)
            lo = m;
        else
            hi = m;
    }
    float sum = 0;
    for (int j = 0; j < kernel; ++j) {
        int src = t + j - kernel + 1;
        if (src >= int(cu[lo]))
            sum += __bfloat162float(x[uint64_t(src) * channels + c]) *
                   __bfloat162float(weight[c * kernel + j]);
    }
    // Preserve the Transformers conv -> BF16 -> SiLU rounding contract.
    sum = __bfloat162float(__float2bfloat16_rn(sum));
    out[idx] = __float2bfloat16_rn(sum / (1.f + expf(-sum)));
}
extern "C" __global__ void qwen35_gdn_prepare(const __nv_bfloat16 *qkv, const __nv_bfloat16 *ab,
                                              const float *alog, const float *dt, float *qk,
                                              float *decay_beta, int tokens, int kh, int vh) {
    const int row = blockIdx.x, head = row % kh, t = row / kh, j = threadIdx.x;
    const int channels = (2 * kh + vh) * 128;
    float q = __bfloat162float(qkv[uint64_t(t) * channels + head * 128 + j]);
    float k = __bfloat162float(qkv[uint64_t(t) * channels + kh * 128 + head * 128 + j]);
    __shared__ float qs[4], ks[4];
    float qsum = gdn_sum(q * q), ksum = gdn_sum(k * k);
    if (j % 32 == 0) {
        qs[j / 32] = qsum;
        ks[j / 32] = ksum;
    }
    __syncthreads();
    qsum = qs[0] + qs[1] + qs[2] + qs[3];
    ksum = ks[0] + ks[1] + ks[2] + ks[3];
    qk[uint64_t(row) * 256 + j] = q * rsqrtf(qsum + 1e-6f) * rsqrtf(128.f);
    qk[uint64_t(row) * 256 + 128 + j] = k * rsqrtf(ksum + 1e-6f);
    if (head == 0 && j < vh) {
        float a = __bfloat162float(ab[uint64_t(t) * 2 * vh + j]) + dt[j];
        float softplus = a > 20.f ? a : log1pf(expf(a));
        float b = __bfloat162float(ab[uint64_t(t) * 2 * vh + vh + j]);
        decay_beta[uint64_t(t) * 2 * vh + j] = expf(-expf(alog[j]) * softplus);
        decay_beta[uint64_t(t) * 2 * vh + vh + j] =
            __bfloat162float(__float2bfloat16_rn(1.f / (1.f + expf(-b))));
    }
}
extern "C" __global__ void qwen35_gdn_recurrent(const __nv_bfloat16 *qkv, const float *qk,
                                                const float *decay_beta, const uint32_t *cu,
                                                __nv_bfloat16 *out, int kh, int vh) {
    const int seq = blockIdx.x, head = blockIdx.y, col = blockIdx.z * 8 + threadIdx.x / 32,
              lane = threadIdx.x % 32;
    const int qhead = head / (vh / kh), channels = (2 * kh + vh) * 128;
    // One warp owns one state column. Four FP32 entries per lane, no state in HBM.
    float state[4] = {0, 0, 0, 0};
    for (uint32_t t = cu[seq]; t < cu[seq + 1]; ++t) {
        float decay = decay_beta[uint64_t(t) * 2 * vh + head],
              beta = decay_beta[uint64_t(t) * 2 * vh + vh + head];
        float q[4], k[4], memory = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            q[i] = qk[(uint64_t(t) * kh + qhead) * 256 + lane + i * 32];
            k[i] = qk[(uint64_t(t) * kh + qhead) * 256 + 128 + lane + i * 32];
            state[i] *= decay;
            memory += state[i] * k[i];
        }
        memory = gdn_sum(memory);
        float v = __bfloat162float(qkv[uint64_t(t) * channels + 2 * kh * 128 + head * 128 + col]);
        float delta = (v - memory) * beta, sum = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            state[i] += k[i] * delta;
            sum += state[i] * q[i];
        }
        sum = gdn_sum(sum);
        if (lane == 0)
            out[(uint64_t(t) * vh + head) * 128 + col] = __float2bfloat16_rn(sum);
    }
}
extern "C" __global__ void qwen35_gdn_norm(const __nv_bfloat16 *x, const __nv_bfloat16 *z,
                                           const __nv_bfloat16 *weight, __nv_bfloat16 *out,
                                           float eps) {
    const uint64_t row = blockIdx.x;
    const int j = threadIdx.x;
    float v = __bfloat162float(x[row * 128 + j]);
    float sum = gdn_sum(v * v);
    __shared__ float sums[4];
    if (j % 32 == 0)
        sums[j / 32] = sum;
    __syncthreads();
    sum = (sums[0] + sums[1] + sums[2] + sums[3]) / 128.f;
    float normalized = __bfloat162float(__float2bfloat16_rn(v * rsqrtf(sum + eps)));
    normalized = __bfloat162float(__float2bfloat16_rn(normalized * __bfloat162float(weight[j])));
    float gate = __bfloat162float(z[row * 128 + j]);
    out[row * 128 + j] = __float2bfloat16_rn(normalized * (gate / (1.f + expf(-gate))));
}
