/*
Independent test translation of candle-cublaslt/src/fp8.rs.
Original cuBLASLt layouts, scaling modes, fast/full tactic fallback and32MiB
workspace are preserved. No backend production helper is used for this GEMM.

Permission is hereby granted, free of charge, to any
person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the
Software without restriction, including without
limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice
shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
*/
#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>

namespace {
inline void fp8_reference_check(cublasStatus_t s) {
  TORCH_CHECK(s == CUBLAS_STATUS_SUCCESS, "Reference cuBLASLt status=", int(s));
}

inline at::Tensor fp8_original_recipe(const at::Tensor& x,
                                      const at::Tensor& w,
                                      const at::Tensor& sx,
                                      const at::Tensor& sw,
                                      at::ScalarType dtype) {
  const bool rowwise = at::cuda::getCurrentDeviceProperties()->major == 9;
  const auto gemm_dtype = rowwise ? dtype : at::kFloat;
  cublasLtHandle_t handle;
  fp8_reference_check(cublasLtCreate(&handle));
  cublasLtMatmulDesc_t desc;
  fp8_reference_check(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  auto trans = CUBLAS_OP_T;
  fp8_reference_check(
      cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)));
  int8_t fast = rowwise ? 1 : 0;
  fp8_reference_check(
      cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast, sizeof(fast)));
  if (rowwise) {
    auto mode = CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
    fp8_reference_check(cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode)));
    fp8_reference_check(cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode)));
    auto psw = sw.data_ptr<float>(), psx = sx.data_ptr<float>();
    fp8_reference_check(cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &psw, sizeof(psw)));
    fp8_reference_check(cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &psx, sizeof(psx)));
  }
  cublasLtMatrixLayout_t a, b, c;
  int64_t m = x.size(0), n = w.size(0), k = w.size(1);
  fp8_reference_check(cublasLtMatrixLayoutCreate(&a, CUDA_R_8F_E4M3, k, n, k));
  fp8_reference_check(cublasLtMatrixLayoutCreate(&b, CUDA_R_8F_E4M3, k, m, k));
  fp8_reference_check(cublasLtMatrixLayoutCreate(&c,
                                                 gemm_dtype == at::kHalf       ? CUDA_R_16F
                                                 : gemm_dtype == at::kBFloat16 ? CUDA_R_16BF
                                                                               : CUDA_R_32F,
                                                 n,
                                                 m,
                                                 n));
  cublasLtMatmulPreference_t pref;
  fp8_reference_check(cublasLtMatmulPreferenceCreate(&pref));
  size_t bytes = 32u << 20;
  auto scratch = at::empty({int64_t(bytes)}, x.options().dtype(at::kByte));
  fp8_reference_check(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &bytes, sizeof(bytes)));
  cublasLtMatmulHeuristicResult_t algo{};
  int count = 0;
  fp8_reference_check(
      cublasLtMatmulAlgoGetHeuristic(handle, desc, a, b, c, c, pref, 1, &algo, &count));
  if (!count) {
    fast = 0;
    fp8_reference_check(
        cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast, sizeof(fast)));
    fp8_reference_check(
        cublasLtMatmulAlgoGetHeuristic(handle, desc, a, b, c, c, pref, 1, &algo, &count));
  }
  TORCH_CHECK(count, "Reference cuBLASLt no algorithm");
  auto y = at::empty({m, n}, x.options().dtype(gemm_dtype));
  float alpha = 1, beta = 0;
  fp8_reference_check(cublasLtMatmul(handle,
                                     desc,
                                     &alpha,
                                     w.data_ptr(),
                                     a,
                                     x.data_ptr(),
                                     b,
                                     &beta,
                                     y.data_ptr(),
                                     c,
                                     y.data_ptr(),
                                     c,
                                     &algo.algo,
                                     scratch.data_ptr(),
                                     bytes,
                                     at::cuda::getCurrentCUDAStream()));
  fp8_reference_check(cublasLtMatmulPreferenceDestroy(pref));
  fp8_reference_check(cublasLtMatrixLayoutDestroy(a));
  fp8_reference_check(cublasLtMatrixLayoutDestroy(b));
  fp8_reference_check(cublasLtMatrixLayoutDestroy(c));
  fp8_reference_check(cublasLtMatmulDescDestroy(desc));
  fp8_reference_check(cublasLtDestroy(handle));
  return rowwise ? y : (y * sx * sw.t()).to(dtype);
}
}
