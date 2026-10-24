// Newly authored launch setup cache. Pinned Torch sources and notices stay intact.
#pragma once
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <array>
#include <cstddef>

namespace tei {
// Entries are thread-local and bounded. Context IDs are unique for the process,
// so a destroyed/reset CUDA context cannot reuse an old success entry.
template <typename Kernel>
inline cudaError_t flash_set_attribute(Kernel function, cudaFuncAttribute attribute, int value) {
  const void* kernel = reinterpret_cast<const void*>(function);
  if (attribute != cudaFuncAttributeMaxDynamicSharedMemorySize)
    return cudaFuncSetAttribute(kernel, attribute, value);
  // Resolve lazily through libcudart: loading the CUDA-enabled library on a
  // CPU-only host must not introduce a direct libcuda DT_NEEDED dependency.
  static const auto get_context_id = []() -> PFN_cuCtxGetId_v12000 {
    void* pointer = nullptr;
    cudaDriverEntryPointQueryResult query = cudaDriverEntryPointSymbolNotFound;
    const auto status =
        cudaGetDriverEntryPointByVersion("cuCtxGetId", &pointer, 12000, cudaEnableDefault, &query);
    if (status != cudaSuccess || query != cudaDriverEntryPointSuccess || !pointer)
      return nullptr;
    return reinterpret_cast<PFN_cuCtxGetId_v12000>(pointer);
  }();
  unsigned long long context_id = 0;
  if (!get_context_id || get_context_id(nullptr, &context_id) != CUDA_SUCCESS)
    return cudaFuncSetAttribute(kernel, attribute, value);

  struct Entry {
    unsigned long long context;
    const void* kernel;
    int value;
  };

  thread_local std::array<Entry, 16> entries{};
  thread_local std::size_t cursor = 0;
  Entry* existing = nullptr;
  for (auto& entry : entries) {
    if (entry.context == context_id && entry.kernel == kernel) {
      existing = &entry;
      if (entry.value == value)
        return cudaSuccess;
      break;
    }
  }
  const auto status = cudaFuncSetAttribute(kernel, attribute, value);
  if (status == cudaSuccess) {
    if (existing)
      *existing = {context_id, kernel, value};
    else {
      entries[cursor] = {context_id, kernel, value};
      cursor = (cursor + 1) % entries.size();
    }
  }
  return status;
}
} // namespace tei
