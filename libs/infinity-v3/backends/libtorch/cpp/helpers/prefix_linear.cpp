#include "helpers/prefix_linear.h"
#include <array>
#include <map>
#include <algorithm>
#ifdef TEI_TORCH_CUDA_KERNELS
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#endif

namespace tei {
struct PrefixLinearPlans::Impl {
#ifdef TEI_TORCH_CUDA_KERNELS
  struct Plan {
    cublasLtMatmulDesc_t operation{};
    cublasLtMatrixLayout_t weight{}, input{}, output{};
    cublasLtMatmulAlgo_t algorithm{};
    size_t workspace_bytes = 0;
    uint64_t used = 0;
    bool valid = false;

    ~Plan() {
      if (output)
        cublasLtMatrixLayoutDestroy(output);
      if (input)
        cublasLtMatrixLayoutDestroy(input);
      if (weight)
        cublasLtMatrixLayoutDestroy(weight);
      if (operation)
        cublasLtMatmulDescDestroy(operation);
    }
  };

  using Key = std::array<int64_t, 15>;
  std::map<Key, std::unique_ptr<Plan>> plans;
  uint64_t clock = 0;
#endif
};

PrefixLinearPlans::PrefixLinearPlans() : impl(std::make_unique<Impl>()) {}

PrefixLinearPlans::~PrefixLinearPlans() = default;

void PrefixLinearPlans::clear() {
  impl = std::make_unique<Impl>();
}

at::Tensor PrefixLinearPlans::forward(const at::Tensor& x,
                                      const at::Tensor& matrix,
                                      const at::Tensor& bias,
                                      int64_t logical_rows,
                                      bool float32) {
#ifdef TEI_TORCH_CUDA_KERNELS
  const bool eligible =
      x.is_cuda() && x.dim() == 2 && matrix.dim() == 2 && x.is_contiguous() &&
      matrix.is_contiguous() && x.size(0) > 0 && logical_rows > x.size(0) &&
      matrix.device() == x.device() && matrix.scalar_type() == x.scalar_type() &&
      matrix.size(1) == x.size(1) &&
      (x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16 ||
       (float32 && x.scalar_type() == at::kFloat && !at::globalContext().allowTF32CuBLAS() &&
        at::globalContext().float32MatmulPrecision() == at::Float32MatmulPrecision::HIGHEST)) &&
      (x.scalar_type() != at::kHalf || !at::globalContext().allowFP16AccumulationCuBLAS());
  if (eligible) {
    c10::cuda::CUDAGuard guard(x.device());
    auto alignment = [](const at::Tensor& t) {
      auto address = reinterpret_cast<uintptr_t>(t.const_data_ptr());
      uint32_t result = 1;
      while (result < 256 && address % (result * 2) == 0)
        result *= 2;
      return result;
    };
    const auto m = x.size(0), n = matrix.size(0), k = x.size(1);
    const auto limit = at::cuda::getCUDABlasLtWorkspaceSize();
    // cuBLAS and cuBLASLt have distinct ATen workspace budgets. Preserve the
    // original provider's planning budget. Some suffix recipes exceed the
    // default Lt scratch by only alignment padding; allow bounded temporary
    // ATen storage up to the original provider's workspace allowance.
    const auto logical_limit = at::globalContext().blasPreferredBackend() == at::BlasBackend::Cublas
                                   ? at::cuda::getChosenWorkspaceSize()
                                   : limit;
    const auto a_alignment = alignment(matrix), b_alignment = alignment(x);
    const auto data_type = x.scalar_type() == at::kHalf       ? CUDA_R_16F
                           : x.scalar_type() == at::kBFloat16 ? CUDA_R_16BF
                                                              : CUDA_R_32F;
    if (bias.defined())
      TORCH_CHECK(bias.device() == x.device() && bias.scalar_type() == x.scalar_type() &&
                      bias.is_contiguous() && bias.sizes() == at::IntArrayRef({n}),
                  "Invalid prefix linear bias");
    Impl::Key key = {x.get_device(),
                     logical_rows,
                     m,
                     n,
                     k,
                     int64_t(x.scalar_type()),
                     a_alignment,
                     b_alignment,
                     int64_t(bias.defined()),
                     int64_t(limit),
                     int64_t(at::globalContext().blasPreferredBackend()),
                     int64_t(at::globalContext().allowBF16ReductionCuBLAS()),
                     int64_t(at::globalContext().allowFP16ReductionCuBLAS()),
                     at::globalContext()._SMCarveout_EXPERIMENTAL().value_or(-1),
                     int64_t(logical_limit)};
    auto found = impl->plans.find(key);
    if (found == impl->plans.end()) {
      if (impl->plans.size() >= 64) {
        auto victim = impl->plans.begin();
        for (auto it = impl->plans.begin(); it != impl->plans.end(); ++it)
          if (it->second->used < victim->second->used)
            victim = it;
        impl->plans.erase(victim);
      }
      auto plan = std::make_unique<Impl::Plan>();

      struct Logical {
        cublasLtMatrixLayout_t input{}, output{};
        cublasLtMatmulPreference_t preference{};

        ~Logical() {
          if (preference)
            cublasLtMatmulPreferenceDestroy(preference);
          if (output)
            cublasLtMatrixLayoutDestroy(output);
          if (input)
            cublasLtMatrixLayoutDestroy(input);
        }
      } logical;

      auto transpose = CUBLAS_OP_T, no_transpose = CUBLAS_OP_N;
      bool valid = cublasLtMatmulDescCreate(&plan->operation, CUBLAS_COMPUTE_32F, CUDA_R_32F) ==
                   CUBLAS_STATUS_SUCCESS;
      if (valid)
        valid = cublasLtMatmulDescSetAttribute(
                    plan->operation, CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)) ==
                CUBLAS_STATUS_SUCCESS;
      if (valid)
        valid = cublasLtMatmulDescSetAttribute(plan->operation,
                                               CUBLASLT_MATMUL_DESC_TRANSB,
                                               &no_transpose,
                                               sizeof(no_transpose)) == CUBLAS_STATUS_SUCCESS;
      if (valid && at::globalContext()._SMCarveout_EXPERIMENTAL().has_value()) {
        int32_t target = at::cuda::getCurrentDeviceProperties()->multiProcessorCount -
                         at::globalContext()._SMCarveout_EXPERIMENTAL().value();
        valid =
            target > 0 && cublasLtMatmulDescSetAttribute(plan->operation,
                                                         CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET,
                                                         &target,
                                                         sizeof(target)) == CUBLAS_STATUS_SUCCESS;
      }
      if (valid && bias.defined()) {
        auto epilogue = CUBLASLT_EPILOGUE_BIAS;
        const auto* pointer = bias.const_data_ptr();
        valid = cublasLtMatmulDescSetAttribute(
                    plan->operation, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)) ==
                    CUBLAS_STATUS_SUCCESS &&
                cublasLtMatmulDescSetAttribute(plan->operation,
                                               CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                               &pointer,
                                               sizeof(pointer)) == CUBLAS_STATUS_SUCCESS &&
                cublasLtMatmulDescSetAttribute(plan->operation,
                                               CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                               &data_type,
                                               sizeof(data_type)) == CUBLAS_STATUS_SUCCESS;
      }
      if (valid)
        valid =
            cublasLtMatrixLayoutCreate(&plan->weight, data_type, k, n, k) == CUBLAS_STATUS_SUCCESS;
      if (valid)
        valid =
            cublasLtMatrixLayoutCreate(&plan->input, data_type, k, m, k) == CUBLAS_STATUS_SUCCESS;
      if (valid)
        valid =
            cublasLtMatrixLayoutCreate(&plan->output, data_type, n, m, n) == CUBLAS_STATUS_SUCCESS;
      if (valid)
        valid = cublasLtMatrixLayoutCreate(&logical.input, data_type, k, logical_rows, k) ==
                CUBLAS_STATUS_SUCCESS;
      if (valid)
        valid = cublasLtMatrixLayoutCreate(&logical.output, data_type, n, logical_rows, n) ==
                CUBLAS_STATUS_SUCCESS;
      if (valid)
        valid = cublasLtMatmulPreferenceCreate(&logical.preference) == CUBLAS_STATUS_SUCCESS;
      if (valid)
        valid =
            cublasLtMatmulPreferenceSetAttribute(logical.preference,
                                                 CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                 &logical_limit,
                                                 sizeof(logical_limit)) == CUBLAS_STATUS_SUCCESS;
      const auto reduction = x.scalar_type() == at::kHalf
                                 ? at::globalContext().allowFP16ReductionCuBLAS()
                                 : at::globalContext().allowBF16ReductionCuBLAS();
      if (valid && x.scalar_type() != at::kFloat &&
          reduction != at::CuBLASReductionOption::AllowReducedPrecisionWithSplitK) {
        uint32_t mask =
            reduction == at::CuBLASReductionOption::DisallowReducedPrecisionAllowSplitK
                ? CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE | CUBLASLT_REDUCTION_SCHEME_NONE
                : CUBLASLT_REDUCTION_SCHEME_NONE;
        valid = cublasLtMatmulPreferenceSetAttribute(logical.preference,
                                                     CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
                                                     &mask,
                                                     sizeof(mask)) == CUBLAS_STATUS_SUCCESS;
      }
      if (valid)
        valid =
            cublasLtMatmulPreferenceSetAttribute(logical.preference,
                                                 CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
                                                 &a_alignment,
                                                 sizeof(a_alignment)) == CUBLAS_STATUS_SUCCESS &&
            cublasLtMatmulPreferenceSetAttribute(logical.preference,
                                                 CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                                                 &b_alignment,
                                                 sizeof(b_alignment)) == CUBLAS_STATUS_SUCCESS;
      cublasLtMatmulHeuristicResult_t original{}, checked{};
      int count = 0;
      auto handle = at::cuda::getCurrentCUDABlasLtHandle();
      if (valid)
        valid = cublasLtMatmulAlgoGetHeuristic(handle,
                                               plan->operation,
                                               plan->weight,
                                               logical.input,
                                               logical.output,
                                               logical.output,
                                               logical.preference,
                                               1,
                                               &original,
                                               &count) == CUBLAS_STATUS_SUCCESS &&
                count == 1 && original.state == CUBLAS_STATUS_SUCCESS;
      if (valid) {
        plan->algorithm = original.algo;
        valid = cublasLtMatmulAlgoCheck(handle,
                                        plan->operation,
                                        plan->weight,
                                        plan->input,
                                        plan->output,
                                        plan->output,
                                        &plan->algorithm,
                                        &checked) == CUBLAS_STATUS_SUCCESS &&
                checked.state == CUBLAS_STATUS_SUCCESS &&
                checked.workspaceSize <= std::max(limit, logical_limit);
        plan->workspace_bytes = checked.workspaceSize;
      }
      plan->valid = valid;
      plan->used = ++impl->clock;
      found = impl->plans.emplace(key, std::move(plan)).first;
    }
    if (found != impl->plans.end() && found->second->valid) {
      auto& plan = *found->second;
      plan.used = ++impl->clock;
      if (bias.defined()) {
        const auto* pointer = bias.const_data_ptr();
        TORCH_CHECK(cublasLtMatmulDescSetAttribute(plan.operation,
                                                   CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                   &pointer,
                                                   sizeof(pointer)) == CUBLAS_STATUS_SUCCESS,
                    "Prefix GEMM bias descriptor update failed");
      }
      auto output = at::empty({m, n}, x.options());
      const float alpha = 1, beta = 0;
      auto extra_workspace =
          plan.workspace_bytes > limit
              ? at::empty({int64_t(plan.workspace_bytes)}, x.options().dtype(at::kByte))
              : at::Tensor();
      auto status = cublasLtMatmul(at::cuda::getCurrentCUDABlasLtHandle(),
                                   plan.operation,
                                   &alpha,
                                   matrix.const_data_ptr(),
                                   plan.weight,
                                   x.const_data_ptr(),
                                   plan.input,
                                   &beta,
                                   output.const_data_ptr(),
                                   plan.output,
                                   output.mutable_data_ptr(),
                                   plan.output,
                                   &plan.algorithm,
                                   extra_workspace.defined() ? extra_workspace.mutable_data_ptr()
                                   : plan.workspace_bytes    ? at::cuda::getCUDABlasLtWorkspace()
                                                             : nullptr,
                                   plan.workspace_bytes,
                                   at::cuda::getCurrentCUDAStream(x.get_device()).stream());
      TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "Prefix GEMM failed: ", int(status));
      return output;
    }
  }
#endif
  return at::linear(x, matrix, bias);
}
}
