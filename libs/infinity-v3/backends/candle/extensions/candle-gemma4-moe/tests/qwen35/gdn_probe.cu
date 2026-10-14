#include "../../../../src/kernels/qwen35_gdn.cu"
extern "C" int gdn_forward(const __nv_bfloat16 *x, const __nv_bfloat16 *z, const __nv_bfloat16 *ab,
                           const __nv_bfloat16 *w, const float *a, const float *d,
                           const __nv_bfloat16 *n, const uint32_t *cu, __nv_bfloat16 *m, float *qk,
                           float *g, __nv_bfloat16 *r, __nv_bfloat16 *out, int t, int kh, int vh,
                           int kernel, int seq, float eps, cudaStream_t stream) {
    int ch = (2 * kh + vh) * 128;
    qwen35_conv_bf16<<<(t * ch + 255) / 256, 256, 0, stream>>>(x, w, cu, m, t, ch, kernel, seq);
    qwen35_gdn_prepare<<<t * kh, 128, 0, stream>>>(m, ab, a, d, qk, g, t, kh, vh);
    qwen35_gdn_recurrent<<<dim3(seq, vh, 16), 256, 0, stream>>>(m, qk, g, cu, r, kh, vh);
    qwen35_gdn_norm<<<t * vh, 128, 0, stream>>>(r, z, n, out, eps);
    return cudaGetLastError();
}
