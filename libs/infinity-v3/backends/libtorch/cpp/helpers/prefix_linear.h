/// \file
/// Plan-cache interface for suffix-only linear execution.
/// Logical row counts retain the original projection recipe during prefix reuse.

#pragma once
#include <ATen/ATen.h>
#include <memory>

namespace tei {
// Retain the original logical GEMM recipe while computing only suffix rows.
// Planning is bounded and workspace remains owned by ATen's current stream.
class PrefixLinearPlans {
public:
  PrefixLinearPlans();
  ~PrefixLinearPlans();
  void clear();
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
