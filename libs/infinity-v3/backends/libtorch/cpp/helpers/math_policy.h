// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <ATen/Context.h>
#include <c10/core/Device.h>
#include <array>
#ifdef TEI_TORCH_CUDA_KERNELS
#include <ATen/cuda/CUDAContext.h>
#endif

namespace tei {
using MathPolicy = std::array<int64_t, 22 + at::num_sdp_backends>;

// Both resident snapshots and captured shaders must use the same policy.
static inline MathPolicy execution_math_policy(const c10::Device& device) {
  auto& global = at::globalContext();
  MathPolicy policy = {int64_t(global.allowTF32CuBLAS()),
                       int64_t(global.allowFP16ReductionCuBLAS()),
                       int64_t(global.allowBF16ReductionCuBLAS()),
                       int64_t(global.allowFP16AccumulationCuBLAS()),
                       int64_t(global.float32MatmulPrecision()),
                       int64_t(global.blasPreferredBackend()),
                       int64_t(global.deterministicAlgorithms()),
                       int64_t(global.allowTF32CuDNN()),
                       int64_t(global.allowFP16ReductionCPU())};
#ifdef TEI_TORCH_CUDA_KERNELS
  if (device.is_cuda()) {
    policy[9] = global._SMCarveout_EXPERIMENTAL().value_or(-1);
    policy[10] = at::cuda::getChosenWorkspaceSize();
    policy[11] = at::cuda::getCUDABlasLtWorkspaceSize();
  }
#endif
  policy[12] = global.userEnabledFlashSDP();
  policy[13] = global.userEnabledFA3SDP();
  policy[14] = global.userEnabledMemEfficientSDP();
  policy[15] = global.userEnabledMathSDP();
  policy[16] = global.userEnabledCuDNNSDP();
  policy[17] = global.allowFP16BF16ReductionMathSDP();
  policy[18] = global.userEnabledOverrideableSDP();
  auto priority = at::globalContext().sDPPriorityOrder();
  for (size_t i = 0; i < priority.size(); ++i)
    policy[19 + i] = int64_t(priority[i]);
  constexpr auto cudnn_policy = 19 + at::num_sdp_backends;
  policy[cudnn_policy] = global.deterministicCuDNN();
  policy[cudnn_policy + 1] = global.benchmarkCuDNN();
  policy[cudnn_policy + 2] = global.benchmarkLimitCuDNN();
  return policy;
}
}
