// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
#include "helpers/mxfp8_linear.h"
#include <ATen/ops/_scaled_mm.h>

#ifdef INFINITY_TORCH_CUDA_KERNELS
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cublasLt.h>
#endif

namespace infinity {
#ifdef INFINITY_TORCH_CUDA_KERNELS
namespace {
#if CUBLAS_VERSION >= 120800
const int32_t kBlockScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
const cublasLtEpilogue_t kEpilogue = CUBLASLT_EPILOGUE_DEFAULT;

struct Probe {
  cublasLtMatmulDesc_t operation{};
  cublasLtMatrixLayout_t a{}, b{}, c{}, d{};
  cublasLtMatmulPreference_t preference{};

  ~Probe() {
    if (preference)
      cublasLtMatmulPreferenceDestroy(preference);
    if (d)
      cublasLtMatrixLayoutDestroy(d);
    if (c)
      cublasLtMatrixLayoutDestroy(c);
    if (b)
      cublasLtMatrixLayoutDestroy(b);
    if (a)
      cublasLtMatrixLayoutDestroy(a);
    if (operation)
      cublasLtMatmulDescDestroy(operation);
  }
};

bool probe_geometry(cublasLtHandle_t handle, int64_t m, int64_t n, int64_t k) {
  if (m <= 0 || n <= 0 || k <= 0 || k % 32 != 0)
    return false;
  auto byte_options = at::TensorOptions().dtype(at::kByte).device(at::kCUDA);
  auto a_scales = at::zeros({mxfp8_scale_elements(n, k)}, byte_options);
  auto b_scales = at::zeros({mxfp8_scale_elements(m, k)}, byte_options);
  Probe probe;
  if (cublasLtMatmulDescCreate(&probe.operation, CUBLAS_COMPUTE_32F, CUDA_R_32F) !=
      CUBLAS_STATUS_SUCCESS)
    return false;
  auto set_i32 = [&](cublasLtMatmulDescAttributes_t attr, int32_t value) {
    return cublasLtMatmulDescSetAttribute(probe.operation, attr, &value, sizeof(value)) ==
           CUBLAS_STATUS_SUCCESS;
  };
  auto set_i8 = [&](cublasLtMatmulDescAttributes_t attr, int8_t value) {
    return cublasLtMatmulDescSetAttribute(probe.operation, attr, &value, sizeof(value)) ==
           CUBLAS_STATUS_SUCCESS;
  };
  auto set_ptr = [&](cublasLtMatmulDescAttributes_t attr, const void* value) {
    return cublasLtMatmulDescSetAttribute(probe.operation, attr, &value, sizeof(value)) ==
           CUBLAS_STATUS_SUCCESS;
  };
  const cublasOperation_t trans_a = CUBLAS_OP_T, trans_b = CUBLAS_OP_N;
  if (!set_i32(CUBLASLT_MATMUL_DESC_TRANSA, trans_a) ||
      !set_i32(CUBLASLT_MATMUL_DESC_TRANSB, trans_b) ||
      !set_i8(CUBLASLT_MATMUL_DESC_FAST_ACCUM, 0) ||
      !set_i32(CUBLASLT_MATMUL_DESC_EPILOGUE, kEpilogue) ||
      !set_i32(CUBLASLT_MATMUL_DESC_A_SCALE_MODE, kBlockScaleMode) ||
      !set_i32(CUBLASLT_MATMUL_DESC_B_SCALE_MODE, kBlockScaleMode) ||
      !set_ptr(CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, a_scales.const_data_ptr()) ||
      !set_ptr(CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, b_scales.const_data_ptr()))
    return false;
  if (cublasLtMatrixLayoutCreate(&probe.a, CUDA_R_8F_E4M3, k, n, k) != CUBLAS_STATUS_SUCCESS ||
      cublasLtMatrixLayoutCreate(&probe.b, CUDA_R_8F_E4M3, k, m, k) != CUBLAS_STATUS_SUCCESS ||
      cublasLtMatrixLayoutCreate(&probe.c, CUDA_R_16BF, n, m, n) != CUBLAS_STATUS_SUCCESS ||
      cublasLtMatrixLayoutCreate(&probe.d, CUDA_R_16BF, n, m, n) != CUBLAS_STATUS_SUCCESS)
    return false;
  if (cublasLtMatmulPreferenceCreate(&probe.preference) != CUBLAS_STATUS_SUCCESS)
    return false;
  const size_t workspace = at::cuda::getCUDABlasLtWorkspaceSize();
  if (cublasLtMatmulPreferenceSetAttribute(probe.preference,
                                           CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                           &workspace,
                                           sizeof(workspace)) != CUBLAS_STATUS_SUCCESS)
    return false;
  cublasLtMatmulHeuristicResult_t heuristic{};
  int returned = 0;
  const auto status = cublasLtMatmulAlgoGetHeuristic(handle,
                                                     probe.operation,
                                                     probe.a,
                                                     probe.b,
                                                     probe.c,
                                                     probe.d,
                                                     probe.preference,
                                                     1,
                                                     &heuristic,
                                                     &returned);
  return status == CUBLAS_STATUS_SUCCESS && returned == 1 &&
         heuristic.state == CUBLAS_STATUS_SUCCESS && heuristic.workspaceSize <= workspace;
}
#else
bool probe_geometry(cublasLtHandle_t, int64_t, int64_t, int64_t) {
  return false;
}
#endif

at::Tensor mxfp8_gemm_torch(const Mxfp8Quantized& lhs, const Mxfp8Quantized& rhs) {
  TORCH_CHECK(lhs.values.is_cuda() && rhs.values.is_cuda() && lhs.cols == rhs.cols,
              "MXFP8 GEMM operand mismatch");
  TORCH_CHECK(lhs.values.device() == rhs.values.device(),
              "MXFP8 GEMM operands must be on the same device");
  TORCH_CHECK(lhs.values.scalar_type() == at::kFloat8_e4m3fn &&
                  rhs.values.scalar_type() == at::kFloat8_e4m3fn,
              "MXFP8 GEMM requires E4M3 operands");
  TORCH_CHECK(lhs.scales.scalar_type() == at::kByte && rhs.scales.scalar_type() == at::kByte &&
                  lhs.scales.is_contiguous() && rhs.scales.is_contiguous(),
              "MXFP8 GEMM requires byte block-scale metadata");
  TORCH_CHECK(lhs.scales.device() == lhs.values.device() &&
                  rhs.scales.device() == rhs.values.device(),
              "MXFP8 GEMM block scales must be on the same device as their values");
  TORCH_CHECK(lhs.scales.numel() == mxfp8_scale_elements(lhs.rows, lhs.cols) &&
                  rhs.scales.numel() == mxfp8_scale_elements(rhs.rows, rhs.cols),
              "MXFP8 GEMM block-scale metadata has the wrong size");
  c10::cuda::CUDAGuard guard(lhs.values.device());
  return at::_scaled_mm(lhs.values,
                        rhs.values.t(),
                        lhs.scales.view(at::kFloat8_e8m0fnu),
                        rhs.scales.view(at::kFloat8_e8m0fnu),
                        {},
                        {},
                        at::kBFloat16,
                        false);
}

at::Tensor add_bias(at::Tensor output, const at::Tensor& bias, int64_t width) {
  if (!bias.defined())
    return output;
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == width && bias.is_cuda() &&
                  bias.device() == output.device() && bias.scalar_type() == at::kBFloat16 &&
                  bias.is_contiguous(),
              "Invalid MXFP8 MLP bias");
  return output + bias;
}
} // namespace
#endif

bool mxfp8_provider_supported(const c10::Device& device) {
#ifdef INFINITY_TORCH_CUDA_KERNELS
  if (!device.is_cuda() || CUBLAS_VERSION < 120800 || cublasLtGetVersion() < 120800)
    return false;
  c10::cuda::CUDAGuard guard(device);
  return probe_geometry(at::cuda::getCurrentCUDABlasLtHandle(), 1, 128, 128);
#else
  return false;
#endif
}

bool mxfp8_geometry_supported(const c10::Device& device, int64_t m, int64_t n, int64_t k) {
#ifdef INFINITY_TORCH_CUDA_KERNELS
  if (!mxfp8_provider_supported(device))
    return false;
  c10::cuda::CUDAGuard guard(device);
  return probe_geometry(at::cuda::getCurrentCUDABlasLtHandle(), m, n, k);
#else
  return false;
#endif
}

Mxfp8Linear::Mxfp8Linear(const at::Tensor& weight) {
#ifdef INFINITY_TORCH_CUDA_KERNELS
  TORCH_CHECK(weight.is_cuda() && weight.dim() == 2 && weight.size(0) > 0 &&
                  weight.size(0) % 16 == 0 && weight.size(1) % 32 == 0 &&
                  weight.scalar_type() == at::kBFloat16,
              "MXFP8 MLP weights require BF16 [N,K], N divisible by 16 and K by 32");
  TORCH_CHECK(mxfp8_provider_supported(weight.device()),
              "MXFP8 MLP requires a provider with native block-scaling support");
  c10::cuda::CUDAGuard guard(weight.device());
  matrix_ = mxfp8_quantize_rows(weight);
  dtype_ = weight.scalar_type();
#else
  TORCH_CHECK(false, "MXFP8 MLP requires a CUDA LibTorch build");
#endif
}

at::Tensor Mxfp8Linear::forward(const at::Tensor& input, const at::Tensor& bias) const {
#ifdef INFINITY_TORCH_CUDA_KERNELS
  TORCH_CHECK(input.is_cuda() && input.dim() == 2 && input.size(1) == input_width() &&
                  input.scalar_type() == dtype_,
              "Invalid MXFP8 MLP input");
  TORCH_CHECK(input.device() == matrix_.values.device(),
              "MXFP8 MLP input and weights must be on the same device");
  c10::cuda::CUDAGuard guard(input.device());
  if (input.size(0) == 0)
    return at::empty({0, output_width()}, input.options().dtype(dtype_));
  auto quantized = mxfp8_quantize_rows(input);
  return add_bias(mxfp8_gemm_torch(quantized, matrix_), bias, output_width());
#else
  TORCH_CHECK(false, "MXFP8 MLP requires a CUDA LibTorch build");
#endif
}

at::Tensor Mxfp8Linear::forward_quantized(const at::Tensor& input,
                                          const at::Tensor& scales,
                                          const at::Tensor& bias) const {
#ifdef INFINITY_TORCH_CUDA_KERNELS
  TORCH_CHECK(input.is_cuda() && input.dim() == 2 && input.size(1) == input_width() &&
                  input.scalar_type() == at::kFloat8_e4m3fn,
              "Invalid MXFP8 quantized MLP input");
  TORCH_CHECK(input.device() == matrix_.values.device(),
              "MXFP8 quantized input and weights must be on the same device");
  TORCH_CHECK(scales.scalar_type() == at::kByte,
              "MXFP8 MLP requires byte UE8M0 block-scale metadata; per-row FP32 RMS scales "
              "cannot feed an MXFP8 GEMM");
  TORCH_CHECK(scales.device() == input.device(), "MXFP8 block scales must be on the input device");
  c10::cuda::CUDAGuard guard(input.device());
  const int64_t rows = input.size(0);
  if (rows == 0) {
    TORCH_CHECK(scales.is_cuda() && scales.device() == input.device() && scales.numel() == 0,
                "Invalid MXFP8 empty block-scale metadata");
    return at::empty({0, output_width()}, input.options().dtype(at::kBFloat16));
  }
  auto values = mxfp8_aligned_tensor(input, at::kFloat8_e4m3fn, 2, "MXFP8 quantized input");
  auto aligned_scales = mxfp8_aligned_tensor(scales, at::kByte, 1, "MXFP8 block scales");
  TORCH_CHECK(aligned_scales.is_cuda() && aligned_scales.device() == values.device() &&
                  aligned_scales.numel() == mxfp8_scale_elements(rows, values.size(1)),
              "Invalid MXFP8 block-scale metadata shape or device");
  Mxfp8Quantized quantized{values, aligned_scales, rows, values.size(1)};
  return add_bias(mxfp8_gemm_torch(quantized, matrix_), bias, output_width());
#else
  TORCH_CHECK(false, "MXFP8 MLP requires a CUDA LibTorch build");
#endif
}

std::pair<at::Tensor, at::Tensor>
Mxfp8Linear::forward_gated_quantized(const at::Tensor& input,
                                     const at::Tensor& rowwise_scales,
                                     Fp8GatedActivation kind,
                                     bool gate_first) const {
#ifdef INFINITY_TORCH_CUDA_KERNELS
  TORCH_CHECK(!rowwise_scales.defined(),
              "MXFP8 cannot consume per-row FP8 RMS scales; provide model-dtype rows");
  TORCH_CHECK(kind == Fp8GatedActivation::RoundedSilu && gate_first,
              "MXFP8 fused gated path supports rounded SwiGLU gate-first only");
  TORCH_CHECK(input.is_cuda() && input.dim() == 2 && input.size(1) == input_width() &&
                  input.scalar_type() == dtype_,
              "Invalid MXFP8 gated MLP input");
  TORCH_CHECK(input.device() == matrix_.values.device(),
              "MXFP8 gated input and weights must be on the same device");
  c10::cuda::CUDAGuard guard(input.device());
  if (input.size(0) == 0) {
    const int64_t width = output_width() / 2;
    TORCH_CHECK(output_width() % 2 == 0 && width % 32 == 0, "Invalid MXFP8 gated output width");
    return {at::empty({0, width}, input.options().dtype(at::kFloat8_e4m3fn)),
            at::zeros({0}, input.options().dtype(at::kByte))};
  }
  auto quantized = mxfp8_quantize_rows(input);
  auto gate_up = mxfp8_gemm_torch(quantized, matrix_);
  auto activated = mxfp8_gated_silu_quantize(gate_up);
  return {activated.values, activated.scales};
#else
  TORCH_CHECK(false, "MXFP8 MLP requires a CUDA LibTorch build");
#endif
}
} // namespace infinity
