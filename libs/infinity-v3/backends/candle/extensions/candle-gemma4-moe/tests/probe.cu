// SPDX-License-Identifier: MIT
// Test-only entry points; the production library does not expose these.
#include "../kernels/grouped_gemm.cu"
extern "C" int gemma4_test_gelu(
    const __nv_bfloat16 *input, __nv_bfloat16 *output,
    int tokens, int width, cudaStream_t stream) {
    if (tokens <= 0 || width <= 0) return -1;
    gemma4_moe_gelu_mul<<<tokens,128,0,stream>>>(
        input,output,tokens,width);
    return cudaGetLastError();
}

// Scalar pre-vectorization arithmetic retained only as a numerical test oracle.
__global__ void scalar_gelu_reference(const __nv_bfloat16 *input, __nv_bfloat16 *output,
                                      uint64_t tokens, uint32_t width) {
    const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < tokens * width) {
        const uint64_t row = i / width;
        const uint32_t column = i % width;
        const float gate = __bfloat162float(input[row * (2 * width) + column]);
        const float up = __bfloat162float(input[row * (2 * width) + width + column]);
        const float gelu = 0.5f * gate * (1.f + tanhf(0.7978845608028654f *
            (gate + 0.044715f * gate * gate * gate)));
        const float rounded_gelu = __bfloat162float(__float2bfloat16_rn(gelu));
        output[i] = __float2bfloat16_rn(rounded_gelu * up);
    }
}
extern "C" int gemma4_test_gelu_reference(
    const __nv_bfloat16 *input, __nv_bfloat16 *output,
    int tokens, int width, cudaStream_t stream) {
    if (tokens <= 0 || width <= 0) return -1;
    scalar_gelu_reference<<<(uint64_t(tokens)*width+255)/256,256,0,stream>>>(
        input,output,tokens,width);
    return cudaGetLastError();
}
