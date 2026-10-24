#pragma once
#include <ATen/ATen.h>
#include <utility>
#include <memory>

namespace tei {
// Real rows only: the first dimension is never aligned or padded.
std::pair<at::Tensor, at::Tensor> fp8_quantize_rows(const at::Tensor& input);

// Undefined output preserves the original ordinary ATen linear path.
// Bounded model-owned ordinary plans; graph capture never admits a new plan.
class OrdinaryLinearExecutor {
  struct Impl;
  std::shared_ptr<Impl> impl_;

public:
  class Session {
    struct State;
    std::shared_ptr<State> state_;
    friend class OrdinaryLinearExecutor;

  public:
    Session() = default;
  };

  OrdinaryLinearExecutor();
  // Request-local snapshot: retains its executor and captures policy, context,
  // stream and handle once. It stores no thread-local guard. Callers must keep
  // the same device, stream and policy while dispatching with it.
  Session prepare(const at::Tensor& input) const;
  at::Tensor
  forward(const at::Tensor& input, const at::Tensor& weight, const Session& session) const;
  at::Tensor forward(const at::Tensor& input, const at::Tensor& weight) const;
};

// Checkpoint weights are quantized once, activations once per invocation.
// Attention, normalization, bias and output heads retain their model dtype.
class Fp8Linear {
  at::Tensor matrix_, scales_, unit_;
  at::ScalarType dtype_;
  bool rowwise_;

public:
  explicit Fp8Linear(const at::Tensor& weight);
  at::Tensor forward(const at::Tensor& input, const at::Tensor& bias = {}) const;
  // Accept the identical rowwise recipe from a fused activation producer.
  at::Tensor forward_quantized(const at::Tensor& input,
                               const at::Tensor& scales,
                               const at::Tensor& bias = {}) const;

  int64_t input_width() const {
    return matrix_.size(1);
  }

  int64_t output_width() const {
    return matrix_.size(0);
  }

  const at::Tensor& matrix() const {
    return matrix_;
  }

  const at::Tensor& scales() const {
    return scales_;
  }
};
}
