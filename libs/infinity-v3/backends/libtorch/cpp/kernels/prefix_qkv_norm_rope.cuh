// SPDX-License-Identifier: MIT
// Derived from candle/src/kernels/qk_norm_rope.cu; retain its normalization
// reduction and rounding order. Output placement and raw V copying are added.
/*
 * Permission is hereby granted, free of charge, to any
 * person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the
 * Software without restriction, including without
 * limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software
 * is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice
 * shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
 * ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
 * TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
 * PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
 * SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
 * IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
template <class T,bool Indexed>
__global__ void prefix_qkv_norm_rope(const T *q, const T *k, const T *qw, const T *kw, const T *cos,
                                   const T *sin, const T *v, T *qout, T *kout, T *vout, int tokens, int qheads, int kheads, int qstride, int kstride, int vstride,
                                   float eps,const int32_t* output_rows,int output_tokens) {
    // Each 16-thread group handles one 128-element head (eight heads per block).
    const int lane = threadIdx.x % 16, group = threadIdx.x / 16;
    const int qrows = tokens * qheads, rows = qrows + tokens * kheads;
    using V = layer_norm::Vec<T, 8>;
    for (int row = blockIdx.x * 8 + group; row < rows; row += gridDim.x * 8) {
        bool isq = row < qrows;
        int r = isq ? row : row - qrows;
        int token = r / (isq ? qheads : kheads);
        const int output_token=Indexed&&!isq?output_rows[token]:token;
        if constexpr(Indexed)CUDA_KERNEL_ASSERT(output_token>=0&&output_token<output_tokens);
        const int output_row=isq?r:output_token*kheads+r%kheads;
        const T *in = isq ? q : k;
        const T *w = isq ? qw : kw;
        V x, g;
        float xf[8] = {};
        float mean = 0;

        const int heads = isq ? qheads : kheads;
        const size_t offset = size_t(token) * (isq ? qstride : kstride) + (r % heads) * 128;
        x.load_from(in, offset / 8 + lane);
        g.load_from(w, lane);
#pragma unroll
        for (int j = 0; j < 8; j++) {
            xf[j] = float(x.data.elt[j]);
            mean += xf[j];
        }

#pragma unroll
        for (int off = 1; off < 16; off *= 2)
            mean +=
                __shfl_xor_sync((threadIdx.x % 32) < 16 ? 0x0000ffff : 0xffff0000, mean, off, 16);
        // The original 32-lane reduction adds zero from the unused upper half warp.
        mean += 0.f;
        mean *= 1.f / 128;
        float variance = 0;

#pragma unroll
        for (int j = 0; j < 8; j++) {
            float d = xf[j] - mean;
            variance += d * d;
        }

#pragma unroll
        for (int off = 1; off < 16; off *= 2)
            variance += __shfl_xor_sync((threadIdx.x % 32) < 16 ? 0x0000ffff : 0xffff0000, variance,
                                        off, 16);
        // Match the baseline: round mean squared before adding it.
        variance += 0.f;
        float inv = rsqrtf(__fadd_rn(fmaf(variance, 1.f / 128, eps), __fmul_rn(mean, mean)));
        // Round normalized values to the model dtype before applying RoPE.
        T vals[8];
#pragma unroll
        for (int j = 0; j < 8; j++)
            vals[j] = T(float(g.data.elt[j]) * (inv * (xf[j] - 0.f)) + 0.f);

        // Lower eight lanes hold the first NeoX half; exchange with the second half.
        V c, s, a, b;
        if (lane < 8) {
            c.load_from(cos, token * 8 + lane);
            s.load_from(sin, token * 8 + lane);
        }
#pragma unroll
        for (int j = 0; j < 8; j++) {
            T peer = T(__shfl_xor_sync((threadIdx.x % 32) < 16 ? 0x0000ffff : 0xffff0000,
                                       float(vals[j]), 8, 16));
            if (lane < 8) {
                T v = vals[j];
                T cv = c.data.elt[j], sv = s.data.elt[j];
                a.data.elt[j] = v * cv - peer * sv;
                b.data.elt[j] = peer * cv + v * sv;
            }
        }
        if (lane < 8) {
            a.store_to(isq ? qout : kout, output_row * 16 + lane);
            b.store_to(isq ? qout : kout, output_row * 16 + 8 + lane);
        }
        if (!isq) {
            V value;
            value.load_from(v, size_t(token) * vstride / 8 + (r % kheads) * 16 + lane);
            value.store_to(vout, output_row * 16 + lane);
        }
    }
}
