#include "kernels/encoder_activation.h"
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
#include <stdexcept>

namespace {
template <class T> void exhaustive(at::ScalarType dtype) {
  std::vector<T> inputs, expected;
  for (unsigned bits = 0; bits < 65536; ++bits) {
    auto raw = static_cast<uint16_t>(bits);
    T x;
    std::memcpy(&x, &raw, sizeof(x));
    if (!std::isfinite(float(x)))
      continue;
    T square(float(x) * float(x)), cube(float(square) * float(x));
    // The alpha multiply-add rounds once to model dtype, matching CUDA HFMA.
    T alpha(double(x) + double(T(.044715)) * double(cube));
    T argument(float(T(.7978845608028654)) * float(alpha));
    T tangent(std::tanh(float(argument)));
    T scaled(float(T(.5)) * float(x));
    T result(float(scaled) * float(T(1.f + float(tangent))));
    inputs.push_back(x);
    expected.push_back(result);
  }
  auto options = at::TensorOptions().dtype(dtype);
  auto input = at::from_blob(inputs.data(), {static_cast<int64_t>(inputs.size())}, options).clone();
  auto oracle = at::from_blob(expected.data(), input.sizes(), options).to(at::kFloat);
  auto actual = infinity::encoder_gelu(input.to(at::Device(at::kCUDA, 0))).cpu().to(at::kFloat);
  if (!at::isfinite(actual).all().template item<bool>())
    throw std::runtime_error("Nonfinite GELU output for finite input");
  // CPU libm and CUDA libdevice tanhf may straddle one model-dtype rounding boundary.
  const float tolerance = dtype == at::kHalf ? .001f : .008f;
  auto error = (actual - oracle).abs().max().template item<float>();
  if (error > tolerance)
    throw std::runtime_error("Model-dtype GELU arithmetic differs from independent staged oracle");
  std::cout << dtype << ": all " << inputs.size() << " finite values, maximum error " << error
            << "\n";
}
}

int main() {
  at::set_num_threads(1);
  exhaustive<c10::Half>(at::kHalf);
  exhaustive<c10::BFloat16>(at::kBFloat16);
  for (auto dtype : {at::kHalf, at::kBFloat16, at::kFloat}) {
    auto options = at::TensorOptions().dtype(dtype).device(at::Device(at::kCUDA, 0));
    for (int64_t length : {0, 1, 7, 1025}) {
      auto x = at::linspace(-4, 4, length, options);
      auto result = infinity::encoder_gelu(x);
      if (result.sizes() != x.sizes() || result.scalar_type() != dtype)
        throw std::runtime_error("GELU changed shape or dtype");
      if (length && !at::isfinite(result).all().template item<bool>())
        throw std::runtime_error("GELU tail produced nonfinite values");
    }
    auto shifted = at::linspace(-4, 4, 1030, options).narrow(0, 1, 1025);
    if (!at::equal(infinity::encoder_gelu(shifted), infinity::encoder_gelu(shifted.clone())))
      throw std::runtime_error("GELU misaligned contiguous input differs");
    bool rejected = false;
    try {
      infinity::encoder_gelu(at::ones({3, 4}, options).transpose(0, 1));
    } catch (const c10::Error&) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error("GELU accepted strided input");
  }
  for (auto dtype : {at::kHalf, at::kBFloat16, at::kFloat})
    for (int64_t width : {31, 768, 3072}) {
      auto device = at::Device(at::kCUDA, c10::cuda::device_count() > 1 ? 1 : 0);
      auto options = at::TensorOptions().device(device).dtype(dtype);
      auto x = (at::randn({32, width}, options) * 20).contiguous(),
           bias = (at::randn({width}, options) * .25).contiguous();
      // Same-input original GELU already has an exhaustive independent oracle.
      // This checks the additional separate model-dtype bias-rounding boundary.
      auto reference = infinity::encoder_gelu(x + bias);
      c10::cuda::set_device(0);
      auto result = infinity::encoder_bias_gelu(x, bias);
      TORCH_CHECK(c10::cuda::current_device() == 0, "Fused bias/GELU changed caller device");
      TORCH_CHECK(at::equal(result.view(at::kByte), reference.view(at::kByte)),
                  "Fused bias/GELU changed model-dtype arithmetic");
    }
  std::cout << "Exact encoder GELU regression passed\n";
}
