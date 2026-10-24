// SPDX-License-Identifier: Apache-2.0
/// \file
/// Native admission and allocation-failure signals shared with the C ABI bridge.
/// Separates configured budget rejection from allocator and nested-call failures.

#pragma once
#include <stdexcept>
#include <c10/util/Exception.h>

namespace infinity {
/// Signals rejection by a configured native memory-admission budget.
class MemoryAdmissionError final : public std::runtime_error {
public:
  explicit MemoryAdmissionError(const char* message) : std::runtime_error(message) {}
};

// A nested C ABI operation has already classified and captured its exception.
// Preserve that classification when the caller unwinds its pending lease.
struct NativeCallError final {
public:
  const int32_t kind;

  explicit NativeCallError(int32_t code) : kind(code) {}
};

/// Signal a configured-budget rejection separately from an allocator failure.
inline void require_memory_admission(bool accepted, const char* message) {
  if (!accepted)
    throw MemoryAdmissionError(message);
}

// This backend supports CPU and CUDA. CUDA resource APIs report allocation
// failure through AcceleratorError, whereas the caching allocator uses OOM.
/// Classify CUDA allocation failure as recoverable OOM (3); retain other device failures as 1.
inline int32_t accelerator_error_kind(const c10::AcceleratorError& error) {
  constexpr int32_t cuda_memory_allocation = 2;
  return error.get_error_code() == cuda_memory_allocation ? 3 : 1;
}
}
