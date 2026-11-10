// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
/// \file
/// CUTLASS 4.5 SM100 grouped FP8 expert GEMM provider API.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace infinity_moe_fp8 {

size_t grouped_workspace_bytes();

bool grouped_available(int major, int minor);

int grouped_gemm_bf16(const void* packed_a,
                      const void* expert_b,
                      const float* scale_a,
                      const float* scale_b,
                      const int* counts,
                      const int* offsets,
                      void* output,
                      int experts,
                      int hidden,
                      int width,
                      void* workspace,
                      size_t workspace_bytes,
                      cudaStream_t stream);

int grouped_gemm_f32(const void* packed_a,
                     const void* expert_b,
                     const float* scale_a,
                     const float* scale_b,
                     const int* counts,
                     const int* offsets,
                     void* output,
                     int experts,
                     int hidden,
                     int width,
                     void* workspace,
                     size_t workspace_bytes,
                     cudaStream_t stream);

} // namespace infinity_moe_fp8
