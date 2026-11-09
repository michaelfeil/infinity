// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
#include "helpers/nvfp4_linear.h"
#include <ATen/BlasBackend.h>
#include <ATen/ops/_scaled_mm_v2.h>
#include <array>
#include <optional>

#ifdef INFINITY_TORCH_CUDA_KERNELS
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

namespace infinity {
#ifdef INFINITY_TORCH_CUDA_KERNELS
namespace {
const std::array<int64_t, 2> kNvfp4Recipe{
    static_cast<int64_t>(at::blas::ScalingType::BlockWise1x16),
    static_cast<int64_t>(at::blas::ScalingType::TensorWise)};
const std::array<int64_t, 2> kNvfp4Swizzle{
    static_cast<int64_t>(at::blas::SwizzleType::SWIZZLE_32_4_4),
    static_cast<int64_t>(at::blas::SwizzleType::NO_SWIZZLE)};

at::Tensor add_bias(at::Tensor output, const at::Tensor& bias, int64_t width) {
  if (!bias.defined())
    return output;
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == width && bias.is_cuda() &&
                  bias.device() == output.device() && bias.scalar_type() == at::kBFloat16,
              "Invalid NVFP4 MLP bias");
  return output + bias;
}
} // namespace
#endif

bool nvfp4_provider_supported(const c10::Device& device) {
#ifdef INFINITY_TORCH_CUDA_KERNELS
  if (!device.is_cuda())
    return false;
  c10::cuda::CUDAGuard guard(device);
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  return properties != nullptr && properties->major >= 10;
#else
  return false;
#endif
}

bool nvfp4_geometry_supported(const c10::Device& device, int64_t m, int64_t n, int64_t k) {
  return nvfp4_provider_supported(device) && m > 0 && n > 0 && k > 0 && k % 32 == 0 && n % 16 == 0;
}

at::Tensor nvfp4_gemm(const Nvfp4Quantized& lhs, const Nvfp4Quantized& rhs, bool apply_row_scales) {
#ifdef INFINITY_TORCH_CUDA_KERNELS
  TORCH_CHECK(lhs.values.is_cuda() && rhs.values.is_cuda() && lhs.cols == rhs.cols,
              "NVFP4 GEMM operand mismatch");
  TORCH_CHECK(lhs.values.device() == rhs.values.device(),
              "NVFP4 GEMM operands must be on the same device");
  TORCH_CHECK(lhs.values.scalar_type() == at::kFloat4_e2m1fn_x2 &&
                  rhs.values.scalar_type() == at::kFloat4_e2m1fn_x2,
              "NVFP4 GEMM requires packed E2M1 operands");
  TORCH_CHECK(lhs.rows > 0 && lhs.cols > 0 && lhs.values.dim() == 2 &&
                  lhs.values.size(0) == lhs.rows && lhs.values.size(1) == lhs.cols / 2 &&
                  rhs.values.dim() == 2 && rhs.values.size(0) == rhs.rows &&
                  rhs.values.size(1) == rhs.cols / 2,
              "NVFP4 GEMM values shape does not match rows/cols");
  TORCH_CHECK(lhs.values.is_contiguous() && rhs.values.is_contiguous(),
              "NVFP4 GEMM values must be contiguous [rows, K/2]");
  TORCH_CHECK(lhs.block_scales.dim() == 1 && rhs.block_scales.dim() == 1,
              "NVFP4 GEMM block scales must be flat swizzled metadata");
  TORCH_CHECK(lhs.block_scales.scalar_type() == at::kFloat8_e4m3fn &&
                  rhs.block_scales.scalar_type() == at::kFloat8_e4m3fn &&
                  lhs.block_scales.is_contiguous() && rhs.block_scales.is_contiguous(),
              "NVFP4 GEMM requires E4M3 block-scale metadata");
  TORCH_CHECK(lhs.global_scale.scalar_type() == at::kFloat &&
                  rhs.global_scale.scalar_type() == at::kFloat && lhs.global_scale.numel() == 1 &&
                  rhs.global_scale.numel() == 1 && lhs.global_scale.is_contiguous() &&
                  rhs.global_scale.is_contiguous(),
              "NVFP4 GEMM requires one contiguous F32 global scale per operand");
  TORCH_CHECK(lhs.block_scales.device() == lhs.values.device() &&
                  rhs.block_scales.device() == rhs.values.device() &&
                  lhs.global_scale.device() == lhs.values.device() &&
                  rhs.global_scale.device() == rhs.values.device(),
              "NVFP4 GEMM metadata must share the value device");
  TORCH_CHECK(lhs.block_scales.numel() == nvfp4_scale_elements(lhs.rows, lhs.cols) &&
                  rhs.block_scales.numel() == nvfp4_scale_elements(rhs.rows, rhs.cols),
              "NVFP4 GEMM block-scale metadata has the wrong size");
  TORCH_CHECK(!rhs.row_scales.defined(),
              "NVFP4 GEMM right operand is a weight and cannot carry row scales");
  TORCH_CHECK(!lhs.row_scales.defined() ||
                  (lhs.row_scales.dim() == 1 && lhs.row_scales.size(0) == lhs.rows &&
                   lhs.row_scales.scalar_type() == at::kFloat && lhs.row_scales.is_contiguous() &&
                   lhs.row_scales.device() == lhs.values.device()),
              "NVFP4 GEMM requires contiguous F32 [rows] left-operand row scales");
  c10::cuda::CUDAGuard guard(lhs.values.device());
  auto output = at::_scaled_mm_v2(lhs.values,
                                  rhs.values.t(),
                                  {lhs.block_scales, lhs.global_scale},
                                  kNvfp4Recipe,
                                  kNvfp4Swizzle,
                                  {rhs.block_scales, rhs.global_scale},
                                  kNvfp4Recipe,
                                  kNvfp4Swizzle,
                                  std::nullopt,
                                  at::kBFloat16,
                                  {},
                                  false);
  if (apply_row_scales && lhs.row_scales.defined())
    nvfp4_apply_row_scales_(output, lhs.row_scales);
  return output;
#else
  TORCH_CHECK(false, "NVFP4 MLP requires a CUDA LibTorch build");
#endif
}

Nvfp4Linear::Nvfp4Linear(const at::Tensor& weight) {
#ifdef INFINITY_TORCH_CUDA_KERNELS
  TORCH_CHECK(weight.is_cuda() && weight.dim() == 2 && weight.size(0) > 0 &&
                  weight.size(0) % 16 == 0 && weight.size(1) % 32 == 0 &&
                  weight.scalar_type() == at::kBFloat16,
              "NVFP4 MLP weights require BF16 [N,K] with N divisible by 16 and K by 32");
  TORCH_CHECK(nvfp4_provider_supported(weight.device()),
              "NVFP4 MLP requires a Blackwell device with native block-scaling support");
  c10::cuda::CUDAGuard guard(weight.device());
  matrix_ = nvfp4_quantize_rows(weight);
  dtype_ = weight.scalar_type();
#else
  TORCH_CHECK(false, "NVFP4 MLP requires a CUDA LibTorch build");
#endif
}

at::Tensor Nvfp4Linear::forward(const at::Tensor& input, const at::Tensor& bias) const {
#ifdef INFINITY_TORCH_CUDA_KERNELS
  TORCH_CHECK(input.is_cuda() && input.dim() == 2 && input.size(1) == input_width() &&
                  input.scalar_type() == dtype_,
              "Invalid NVFP4 MLP input");
  TORCH_CHECK(input.device() == matrix_.values.device(),
              "NVFP4 MLP input and weights must be on the same device");
  c10::cuda::CUDAGuard guard(input.device());
  if (input.size(0) == 0)
    return at::empty({0, output_width()}, input.options().dtype(dtype_));
  const auto quantized = nvfp4_quantize_rows(input, true);
  return add_bias(nvfp4_gemm(quantized, matrix_), bias, output_width());
#else
  TORCH_CHECK(false, "NVFP4 MLP requires a CUDA LibTorch build");
#endif
}

at::Tensor Nvfp4Linear::forward_quantized(const Nvfp4Quantized& input,
                                          const at::Tensor& bias) const {
#ifdef INFINITY_TORCH_CUDA_KERNELS
  TORCH_CHECK(input.values.is_cuda() && input.cols == input_width() &&
                  input.values.scalar_type() == at::kFloat4_e2m1fn_x2,
              "Invalid NVFP4 quantized MLP input");
  TORCH_CHECK(input.values.device() == matrix_.values.device(),
              "NVFP4 quantized input and weights must be on the same device");
  c10::cuda::CUDAGuard guard(input.values.device());
  const int64_t rows = input.rows;
  if (rows == 0)
    return at::empty({0, output_width()}, matrix_.values.options().dtype(at::kBFloat16));
  return add_bias(nvfp4_gemm(input, matrix_), bias, output_width());
#else
  TORCH_CHECK(false, "NVFP4 MLP requires a CUDA LibTorch build");
#endif
}

Nvfp4Quantized Nvfp4Linear::forward_swiglu_quantized(const at::Tensor& input) const {
#ifdef INFINITY_TORCH_CUDA_KERNELS
  TORCH_CHECK(input.is_cuda() && input.dim() == 2 && input.size(1) == input_width() &&
                  input.scalar_type() == dtype_,
              "Invalid NVFP4 MLP input");
  TORCH_CHECK(input.device() == matrix_.values.device(),
              "NVFP4 MLP input and weights must be on the same device");
  c10::cuda::CUDAGuard guard(input.device());
  if (input.size(0) == 0)
    return nvfp4_swiglu_quantize(at::empty({0, output_width()}, input.options()));
  const auto quantized = nvfp4_quantize_rows(input, true);
  const auto raw = nvfp4_gemm(quantized, matrix_, false);
  return nvfp4_swiglu_quantize(raw, quantized.row_scales);
#else
  TORCH_CHECK(false, "NVFP4 MLP requires a CUDA LibTorch build");
#endif
}

Nvfp4Quantized Nvfp4Linear::swiglu_quantize(const at::Tensor& gate_up) const {
#ifdef INFINITY_TORCH_CUDA_KERNELS
  return nvfp4_swiglu_quantize(gate_up);
#else
  TORCH_CHECK(false, "NVFP4 MLP requires a CUDA LibTorch build");
#endif
}
} // namespace infinity
