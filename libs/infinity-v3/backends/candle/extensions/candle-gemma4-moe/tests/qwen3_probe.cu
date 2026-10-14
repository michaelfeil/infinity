// Test-only launchers for Qwen3 routing and SiLU rounding.
#include "../../../src/kernels/gemma4_moe_kernels.cuh"
extern "C" int qwen3_test_route(const float *logits, uint32_t *ids, float *weights, int tokens,
                                int renormalize, cudaStream_t stream) {
    qwen3_moe_route_128_8_f32<<<tokens, 128, 0, stream>>>(logits, ids, weights, renormalize != 0);
    return cudaGetLastError();
}
extern "C" int qwen3_test_silu(const __nv_bfloat16 *input, __nv_bfloat16 *output, int tokens,
                               int width, cudaStream_t stream) {
    qwen3_moe_silu_mul<<<tokens, 128, 0, stream>>>(input, output, tokens, width);
    return cudaGetLastError();
}
