#include "../../kernels/grouped_gemm.cu"
extern "C" int qwen35_test_route(const float *logits, uint32_t *ids, float *weights, int tokens,
                                 int renormalize, cudaStream_t stream) {
    qwen35_moe_route_256_8_f32<<<tokens, 256, 0, stream>>>(logits, ids, weights, renormalize);
    return cudaGetLastError();
}
