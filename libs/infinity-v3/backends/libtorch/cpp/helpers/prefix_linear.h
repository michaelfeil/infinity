/// \file
/// Plan-cache interface for suffix-only linear execution.
/// Logical row counts retain the original projection recipe during prefix reuse.

#pragma once
#include <ATen/ATen.h>
#include <memory>

namespace infinity {
// Retain the original logical GEMM recipe while computing only suffix rows.
// Planning is bounded and workspace remains owned by ATen's current stream.
class PrefixLinearPlans {
public:
  PrefixLinearPlans();
  ~PrefixLinearPlans();
  void clear();
  /// Project real suffix rows using the full logical row count to select the GEMM recipe. Falls
  /// back to ordinary ATen linear when no qualified suffix plan is available.
  at::Tensor forward(const at::Tensor& input,
                     const at::Tensor& weight,
                     const at::Tensor& bias,
                     int64_t logical_rows,
                     bool float32 = false);

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};
}
