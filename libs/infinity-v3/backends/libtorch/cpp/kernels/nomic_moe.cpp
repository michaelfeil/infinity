// SPDX-License-Identifier: Apache-2.0
/// \file
/// Qualified Nomic MoE routing-policy setup and grouped expert execution.
/// Combines native grouped GEMMs with the encoder activation contract.

#include "kernels/nomic_moe.h"
#include "kernels/encoder_activation.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#include <algorithm>
#include <vector>

namespace tei {
at::Tensor nomic_moe_up(const at::Tensor&, const at::Tensor&, const at::Tensor&);
at::Tensor
nomic_moe_down(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&);

namespace {
/// RAII owner of cuBLASLt descriptors used to qualify Nomic grouped-GEMM policy.
struct LtPolicy {
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  cublasLtMatrixLayout_t weight = nullptr, input = nullptr, output = nullptr;

  ~LtPolicy() {
    if (output)
      cublasLtMatrixLayoutDestroy(output);
    if (input)
      cublasLtMatrixLayoutDestroy(input);
    if (weight)
      cublasLtMatrixLayoutDestroy(weight);
    if (preference)
      cublasLtMatmulPreferenceDestroy(preference);
    if (operation)
      cublasLtMatmulDescDestroy(operation);
  }
};
}

/// Qualify the reference split-K choice for every bounded route count at model load. Return
/// undefined if device, library, precision, or any heuristic falls outside the proved profile.
at::Tensor nomic_moe_split_policy(const c10::Device& device,
                                  at::ScalarType dtype,
                                  int64_t hidden,
                                  int64_t width,
                                  int64_t experts,
                                  int64_t top_k) {
  if (!device.is_cuda() || dtype != at::kBFloat16 || hidden != 768 || width != 3072 ||
      experts != 8 || top_k != 2)
    return {};
  c10::cuda::CUDAGuard guard(device);
  auto* properties = at::cuda::getCurrentDeviceProperties();
  // The stored split policy reproduces this device/library/provider's reference heuristic. A new
  // library version needs per-route-count policy qualification, even when it
  // supports the same BF16 matrices.
  if (properties->major != 9 || properties->minor != 0 || cublasLtGetVersion() != 130200 ||
      at::globalContext().blasPreferredBackend() != at::BlasBackend::Cublaslt ||
      at::globalContext().allowBF16ReductionCuBLAS() !=
          at::CuBLASReductionOption::AllowReducedPrecisionWithSplitK)
    return {};
  // cuBLASLt may choose split-K=2 for small, uneven expert batches. Matching
  // its per-count choice preserves the BF16 result of the down projection.
  // Query once at model load; request counts and dispatch remain on the GPU.
  LtPolicy plan;
  if (cublasLtMatmulDescCreate(&plan.operation, CUBLAS_COMPUTE_32F, CUDA_R_32F) !=
          CUBLAS_STATUS_SUCCESS ||
      cublasLtMatmulPreferenceCreate(&plan.preference) != CUBLAS_STATUS_SUCCESS ||
      cublasLtMatrixLayoutCreate(&plan.weight, CUDA_R_16BF, hidden, width, hidden) !=
          CUBLAS_STATUS_SUCCESS)
    return {};
  const auto workspace = at::cuda::getCUDABlasLtWorkspaceSize();
  if (cublasLtMatmulPreferenceSetAttribute(plan.preference,
                                           CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                           &workspace,
                                           sizeof(workspace)) != CUBLAS_STATUS_SUCCESS)
    return {};
  std::vector<uint8_t> factors(kNomicGroupedMaxRoutes + 1, 1);
  const auto handle = at::cuda::getCurrentCUDABlasLtHandle();
  for (int64_t rows = 1; rows <= kNomicGroupedMaxRoutes; ++rows) {
    if (cublasLtMatrixLayoutCreate(&plan.input, CUDA_R_16BF, width, rows, width) !=
            CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&plan.output, CUDA_R_16BF, hidden, rows, hidden) !=
            CUBLAS_STATUS_SUCCESS)
      return {};
    cublasLtMatmulHeuristicResult_t algorithm{};
    int count = 0, factor = 0, reduction = 0;
    size_t written = 0;
    if (cublasLtMatmulAlgoGetHeuristic(handle,
                                       plan.operation,
                                       plan.weight,
                                       plan.input,
                                       plan.output,
                                       plan.output,
                                       plan.preference,
                                       1,
                                       &algorithm,
                                       &count) != CUBLAS_STATUS_SUCCESS ||
        count != 1 ||
        cublasLtMatmulAlgoConfigGetAttribute(
            &algorithm.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &factor, sizeof(factor), &written) !=
            CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulAlgoConfigGetAttribute(&algorithm.algo,
                                             CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                                             &reduction,
                                             sizeof(reduction),
                                             &written) != CUBLAS_STATUS_SUCCESS)
      return {};
    factor = std::max(1, factor);
    // The down kernel materializes at most two K partitions and reduces them in FP32. Larger
    // factors or another reduction scheme need an implementation change, not just a wider
    // device/version allowlist.
    if ((factor != 1 && factor != 2) ||
        (factor == 2 && reduction != CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE))
      return {};
    factors[rows] = factor;
    cublasLtMatrixLayoutDestroy(plan.input);
    plan.input = nullptr;
    cublasLtMatrixLayoutDestroy(plan.output);
    plan.output = nullptr;
  }
  return at::from_blob(
             factors.data(), {int64_t(factors.size())}, at::TensorOptions().dtype(at::kByte))
      .to(device);
}

/// Sort selected routes by expert, run grouped projections, then restore token/route order.
/// Weighted contributions accumulate in expert order with a model-dtype rounding at each addition.
at::Tensor nomic_grouped_experts(const at::Tensor& input,
                                 const at::Tensor& first,
                                 const at::Tensor& second,
                                 const at::Tensor& ids,
                                 const at::Tensor& scores,
                                 const at::Tensor& split_policy) {
  TORCH_CHECK(input.dim() == 2 && input.size(0) > 0 && input.size(0) <= kNomicGroupedMaxTokens &&
                  input.size(1) == 768 && input.is_cuda() && input.scalar_type() == at::kBFloat16 &&
                  input.is_contiguous(),
              "Invalid grouped Nomic input");
  TORCH_CHECK(first.sizes() == at::IntArrayRef({8, 3072, 768}) && second.sizes() == first.sizes() &&
                  first.is_contiguous() && second.is_contiguous() &&
                  first.scalar_type() == input.scalar_type() &&
                  second.scalar_type() == input.scalar_type() && first.device() == input.device() &&
                  second.device() == input.device(),
              "Invalid grouped Nomic expert weights");
  TORCH_CHECK(ids.sizes() == at::IntArrayRef({input.size(0), 2}) && scores.sizes() == ids.sizes() &&
                  ids.scalar_type() == at::kLong && scores.scalar_type() == input.scalar_type() &&
                  ids.device() == input.device() && scores.device() == input.device() &&
                  split_policy.device() == input.device() &&
                  split_policy.scalar_type() == at::kByte && split_policy.dim() == 1 &&
                  split_policy.is_contiguous() &&
                  split_policy.numel() == kNomicGroupedMaxRoutes + 1,
              "Invalid grouped Nomic routing/policy");
  c10::cuda::CUDAGuard guard(input.device());
  auto flat = ids.flatten(), order = at::argsort(flat, true, 0, false);
  auto tokens = at::floor_divide(order, 2);
  auto counts = at::zeros({8}, input.options().dtype(at::kInt));
  counts.scatter_add_(0, flat, at::ones_like(flat, flat.options().dtype(at::kInt)));
  auto ends = counts.cumsum(0, at::kInt);
  auto intermediate = nomic_moe_up(input.index_select(0, tokens), first, ends);
  auto contribution = nomic_moe_down(encoder_gelu(intermediate), second, ends, split_policy);
  // Preserve model-dtype multiplication and each expert-ordered addition.
  auto weighted = contribution * scores.flatten().index_select(0, order).unsqueeze(1);
  auto original = at::empty_like(weighted);
  original.index_copy_(0, order, weighted);
  auto slots = at::argsort(ids, true, -1, false);
  auto routed = original.view({input.size(0), 2, 768})
                    .gather(1, slots.unsqueeze(-1).expand({input.size(0), 2, 768}));
  return (at::zeros_like(input) + routed.select(1, 0)) + routed.select(1, 1);
}
} // namespace tei
