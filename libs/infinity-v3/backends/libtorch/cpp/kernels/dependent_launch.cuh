// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
#pragma once
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

namespace infinity {
/// Wait before dependent reads or writes, including writes to allocator-reused outputs.
__device__ inline void wait_for_producer() {
#if __CUDA_ARCH__ >= 900
  cudaGridDependencySynchronize();
#endif
}

/// Let a consumer start during the preceding kernel's completion tail on SM90+.
/// The producer need not explicitly trigger PDL; its completion satisfies the dependency.
template <auto Kernel, class... Args>
void launch_dependent(dim3 grid, dim3 block, cudaStream_t stream, Args... args) {
  // PDL requires the dependency-wait instruction compiled only into PTX90+ kernels.
  // This is an unsupported-capability fallback, including older PTX JITed on a newer GPU.
  // The loaded kernel target also establishes compatible hardware; no GPU-family policy is needed.
  const auto device = c10::cuda::current_device();
  static thread_local c10::DeviceIndex previous_device = -1;
  static thread_local bool supports_pdl = false;
  if (device != previous_device) {
    cudaFuncAttributes attributes{};
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, Kernel));
    supports_pdl = attributes.ptxVersion >= 90;
    previous_device = device;
  }
  if (supports_pdl) {
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = block;
    config.stream = stream;
    config.attrs = &attr;
    config.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&config, Kernel, args...));
  } else {
    Kernel<<<grid, block, 0, stream>>>(args...);
  }
}
}
