#include "helpers/dense_mlp.h"
#include "helpers/fp8_linear.h"
#include "kernels/encoder_activation.h"
#include "kernels/fp8_fusions.h"
#include "kernels/fast_kernels.h"
#include "kernels/qwen35_kernels.h"
#include "helpers/workspace_admission.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Parallel.h>
#include <ATen/Context.h>
#include <torch/version.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/driver_api.h>
#include <thread>
#include <exception>
#include "tests/fp8_reference_test.h"
#include <cstdlib>
#include <array>
#include <cmath>
#include <limits>
#include <c10/core/InferenceMode.h>
#include <iostream>

namespace {
void check_same_or_nan(const at::Tensor& expected, const at::Tensor& actual, const char* label) {
  TORCH_CHECK(expected.sizes() == actual.sizes() && expected.scalar_type() == actual.scalar_type(),
              label,
              " changed shape or dtype");
  auto expected_nan = at::isnan(expected.to(at::kFloat));
  auto actual_nan = at::isnan(actual.to(at::kFloat));
  TORCH_CHECK(at::equal(expected_nan, actual_nan), label, " changed NaN classification");
  auto non_nan = at::logical_not(expected_nan).to(at::kCPU);
  const auto element_size = expected.element_size();
  at::Tensor expected_exact, actual_exact;
  if (element_size == 1) {
    expected_exact = expected.view(at::kByte);
    actual_exact = actual.view(at::kByte);
  } else if (element_size == 2) {
    expected_exact = expected.view(at::kShort);
    actual_exact = actual.view(at::kShort);
  } else if (element_size == 4) {
    expected_exact = expected.view(at::kInt);
    actual_exact = actual.view(at::kInt);
  } else {
    TORCH_CHECK(false, label, " unsupported element size ", element_size);
  }
  auto expected_cpu = expected_exact.to(at::kCPU).contiguous();
  auto actual_cpu = actual_exact.to(at::kCPU).contiguous();
  TORCH_CHECK(at::equal(expected_cpu.masked_select(non_nan), actual_cpu.masked_select(non_nan)),
              label,
              " changed non-NaN bytes");
}

bool blackwell_bf16(at::ScalarType dtype) {
#if ((TORCH_VERSION_MAJOR > 2) || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 9)) &&       \
    defined(CUDA_VERSION) && CUDA_VERSION >= 12080
  const auto* props = at::cuda::getCurrentDeviceProperties();
  const int required = props->major == 11 ? 13000 : props->minor == 0 ? 12080 : 12090;
  return dtype == at::kBFloat16 &&
         (props->major == 10 || props->major == 12 ||
          (props->major == 11 && (TORCH_VERSION_MAJOR > 2 || TORCH_VERSION_MINOR >= 10))) &&
         CUDA_VERSION >= required && at::Context::versionCUDART() >= required;
#else
  return false;
#endif
}

void check_projection_close(const at::Tensor& expected,
                            const at::Tensor& actual,
                            const char* label) {
  TORCH_CHECK(expected.sizes() == actual.sizes() && expected.scalar_type() == actual.scalar_type(),
              label,
              " changed shape or dtype");
  if (!blackwell_bf16(expected.scalar_type())) {
    TORCH_CHECK(at::equal(expected, actual), label, " differs from reference");
    return;
  }
  auto expected_f32 = expected.to(at::kFloat);
  auto positive = at::full_like(expected, std::numeric_limits<float>::infinity());
  auto negative = at::full_like(expected, -std::numeric_limits<float>::infinity());
  auto step_up = (at::nextafter(expected, positive).to(at::kFloat) - expected_f32).abs();
  auto step_down = (expected_f32 - at::nextafter(expected, negative).to(at::kFloat)).abs();
  auto error = (actual.to(at::kFloat) - expected_f32).abs();
  TORCH_CHECK((error <= at::maximum(step_up, step_down)).all().item<bool>(),
              label,
              " exceeded one BF16 ULP: max_error=",
              error.max().item<float>());
}

at::Tensor projection_reference(const at::Tensor& xq,
                                const at::Tensor& sx,
                                const infinity::Fp8Linear& linear,
                                at::ScalarType dtype) {
  return fp8_original_recipe(xq, linear.matrix(), sx, linear.scales(), dtype);
}

constexpr std::array<infinity::Fp8Activation, 3> activation_kinds{
    infinity::Fp8Activation::Relu,
    infinity::Fp8Activation::RoundedSilu,
    infinity::Fp8Activation::RoundedGelu};

at::Tensor reference_activation(const at::Tensor& input,
                                const at::Tensor& bias,
                                infinity::Fp8Activation kind) {
  switch (kind) {
  case infinity::Fp8Activation::Relu:
    return at::relu(bias.defined() ? input + bias : input);
  case infinity::Fp8Activation::RoundedSilu:
    return infinity::mlp_activation(bias.defined() ? input + bias : input,
                                    infinity::MlpActivation::RoundedSilu);
  case infinity::Fp8Activation::RoundedGelu:
    return bias.defined() ? infinity::encoder_bias_gelu(input, bias)
                          : infinity::encoder_gelu(input);
  }
  TORCH_CHECK(false, "Invalid reference activation");
  return {};
}

void check_activation_quantize(const at::Tensor& input,
                               const at::Tensor& bias,
                               infinity::Fp8Activation kind,
                               const char* label) {
  auto expected = infinity::fp8_quantize_rows(reference_activation(input, bias, kind));
  auto actual = infinity::fp8_activation_quantize(input, bias, kind);
  check_same_or_nan(expected.first, actual.first, label);
  check_same_or_nan(expected.second, actual.second, label);
}

void check_scaled_activation_quantize(const at::Tensor& values,
                                      const at::Tensor& row_scales,
                                      const at::Tensor& column_scales,
                                      at::ScalarType dtype,
                                      const at::Tensor& bias,
                                      infinity::Fp8Activation kind,
                                      const char* label) {
  auto scaled = infinity::fp8_scale_output(values, row_scales, column_scales, dtype);
  auto expected = infinity::fp8_quantize_rows(reference_activation(scaled, bias, kind));
  auto actual = infinity::fp8_scaled_activation_quantize(
      values, row_scales, column_scales, dtype, bias, kind);
  check_same_or_nan(expected.first, actual.first, label);
  check_same_or_nan(expected.second, actual.second, label);
}

void check_linear_activation(const at::Tensor& input,
                             const at::Tensor& bias,
                             infinity::Fp8Activation kind,
                             const infinity::Fp8Linear& linear,
                             const char* label) {
  auto quantized = infinity::fp8_quantize_rows(input);
  auto projected =
      projection_reference(quantized.first, quantized.second, linear, input.scalar_type());
  auto expected = infinity::fp8_quantize_rows(reference_activation(projected, bias, kind));
  auto actual = linear.forward_activation_quantized(input, {}, bias, kind);
  check_same_or_nan(expected.first, actual.first, label);
  check_same_or_nan(expected.second, actual.second, label);
  auto prequantized =
      linear.forward_activation_quantized(quantized.first, quantized.second, bias, kind);
  check_same_or_nan(expected.first, prequantized.first, label);
  check_same_or_nan(expected.second, prequantized.second, label);
}

constexpr std::array<infinity::Fp8GatedActivation, 2> gated_kinds{
    infinity::Fp8GatedActivation::RoundedSilu, infinity::Fp8GatedActivation::RoundedGelu};

int gated_activation_code(infinity::Fp8GatedActivation kind) {
  switch (kind) {
  case infinity::Fp8GatedActivation::RoundedSilu:
    return 5;
  case infinity::Fp8GatedActivation::RoundedGelu:
    return 4;
  }
  TORCH_CHECK(false, "Invalid gated activation reference");
  return 0;
}

void check_gated_quantize(const at::Tensor& gate_up,
                          infinity::Fp8GatedActivation kind,
                          bool gate_first,
                          const char* label) {
  auto activated =
      infinity::fused_gated_activation(gate_up, gated_activation_code(kind), gate_first);
  auto expected = infinity::fp8_quantize_rows(activated);
  auto actual = infinity::fp8_gated_quantize(gate_up, kind, gate_first);
  check_same_or_nan(expected.first, actual.first, label);
  check_same_or_nan(expected.second, actual.second, label);
}

void check_scaled_gated_quantize(const at::Tensor& values,
                                 const at::Tensor& row_scales,
                                 const at::Tensor& column_scales,
                                 at::ScalarType dtype,
                                 infinity::Fp8GatedActivation kind,
                                 bool gate_first,
                                 const char* label) {
  auto scaled = infinity::fp8_scale_output(values, row_scales, column_scales, dtype);
  auto expected = infinity::fp8_quantize_rows(
      infinity::fused_gated_activation(scaled, gated_activation_code(kind), gate_first));
  auto actual = infinity::fp8_scaled_gated_quantize(
      values, row_scales, column_scales, dtype, kind, gate_first);
  check_same_or_nan(expected.first, actual.first, label);
  check_same_or_nan(expected.second, actual.second, label);
}

void check_linear_gated_quantize(const at::Tensor& input,
                                 const infinity::Fp8Linear& linear,
                                 infinity::Fp8GatedActivation kind,
                                 bool gate_first,
                                 const char* label) {
  auto quantized = infinity::fp8_quantize_rows(input);
  auto projected =
      projection_reference(quantized.first, quantized.second, linear, input.scalar_type());
  auto expected = infinity::fp8_quantize_rows(
      infinity::fused_gated_activation(projected, gated_activation_code(kind), gate_first));
  auto actual = linear.forward_gated_quantized(input, {}, kind, gate_first);
  check_same_or_nan(expected.first, actual.first, label);
  check_same_or_nan(expected.second, actual.second, label);
  auto prequantized =
      linear.forward_gated_quantized(quantized.first, quantized.second, kind, gate_first);
  check_same_or_nan(expected.first, prequantized.first, label);
  check_same_or_nan(expected.second, prequantized.second, label);
}

// Gemma's precise GeGLU keeps separate model-dtype multiply/add staging, unlike the aligned
// half-pair RoundedGelu reference above. Compare the fused recipe against the actual Gemma kernel
// followed by the production row quantizer.
void check_precise_gated_quantize(const at::Tensor& gate_up, const char* label) {
  auto expected = infinity::fp8_quantize_rows(infinity::gemma_gated_cuda(gate_up, true));
  auto actual =
      infinity::fp8_gated_quantize(gate_up, infinity::Fp8GatedActivation::PreciseGelu, true);
  // Bounded first-mismatch diagnostic: report one element, never dump tensors.
  auto expected_bytes = expected.first.view(at::kByte).to(at::kCPU).contiguous().flatten();
  auto actual_bytes = actual.first.view(at::kByte).to(at::kCPU).contiguous().flatten();
  auto expected_nan = at::isnan(expected.first.to(at::kFloat)).to(at::kCPU).contiguous().flatten();
  auto mismatch =
      at::logical_and(at::ne(expected_bytes, actual_bytes), at::logical_not(expected_nan))
          .nonzero()
          .flatten();
  if (mismatch.numel()) {
    const auto width = expected.first.size(1);
    const auto index = mismatch[0].item<int64_t>();
    TORCH_CHECK(false,
                label,
                " first FP8 byte mismatch index=",
                index,
                " row=",
                index / width,
                " col=",
                index % width,
                " expected=",
                int(expected_bytes[index].item<uint8_t>()),
                " actual=",
                int(actual_bytes[index].item<uint8_t>()));
  }
  auto expected_scales = expected.second.view(at::kInt).to(at::kCPU).contiguous().flatten();
  auto actual_scales = actual.second.view(at::kInt).to(at::kCPU).contiguous().flatten();
  auto scale_mismatch = at::ne(expected_scales, actual_scales).nonzero().flatten();
  if (scale_mismatch.numel())
    TORCH_CHECK(false, label, " first scale mismatch row=", scale_mismatch[0].item<int64_t>());
  check_same_or_nan(expected.first, actual.first, label);
  check_same_or_nan(expected.second, actual.second, label);
}
}

int main() {
  // Match the original independently implemented recipe's workspace budget.
  setenv("CUBLASLT_WORKSPACE_SIZE", "32768", 1);
  c10::InferenceMode inference;
  at::manual_seed(7249);
  at::set_num_threads(4);
  at::globalContext().setAllowTF32CuBLAS(false);
  for (auto dtype : {at::kHalf, at::kBFloat16}) {
    auto options = at::TensorOptions().device(at::kCUDA, 0).dtype(dtype);
    c10::cuda::CUDAGuard device_guard(options.device());
    // Cover every input bit pattern against every row-maximum significand.
    // Include a maximum sentinel per row so every pattern sees the intended
    // scale, including FP8 midpoints and the BF16 scale floor.
    auto patterns = at::arange(65536, at::TensorOptions().dtype(at::kInt))
                        .to(at::kShort)
                        .view(dtype)
                        .to(at::kFloat);
    patterns.masked_fill_(at::logical_not(at::isfinite(patterns)), 0);
    auto padded = at::zeros({17 * 4095}, patterns.options());
    padded.narrow(0, 0, patterns.numel()).copy_(patterns);
    auto chunks = padded.view({1, 17, 4095});
    auto check_rounding = [&](const at::Tensor& maxima) {
      auto limits = maxima.view({-1, 1, 1});
      auto clipped = at::minimum(at::maximum(chunks, -limits), limits);
      auto values = at::cat({limits.expand({maxima.size(0), 17, 1}), clipped}, 2).view({-1, 4096});
      auto scales = at::repeat_interleave(maxima.clamp_min(1e-12) * (1.f / 448.f), 17, 0);
      auto expected = (values / scales).to(at::kFloat8_e4m3fn);
      auto actual = infinity::fp8_quantize_rows(values.to(dtype).to(at::kCUDA));
      TORCH_CHECK(
          at::equal(actual.second.to(at::kCPU), scales) &&
              at::equal(actual.first.to(at::kCPU).view(at::kByte), expected.view(at::kByte)),
          "FP8 conversion changed exhaustive input/significand rounding: ",
          dtype);
      // Exercise the hoisted gated division with varied numerators and a fixed row maximum.
      // A power-of-two gate preserves representable up values; the independent composition
      // also covers any model-dtype underflow before quantization.
      auto model_values = values.to(options);
      auto gate_up = at::cat({at::full_like(model_values, 64), model_values / 64}, 1);
      check_gated_quantize(gate_up,
                           infinity::Fp8GatedActivation::RoundedSilu,
                           true,
                           "Gated division changed exhaustive input/significand rounding");
    };
    const int significands = dtype == at::kHalf ? 1024 : 128;
    for (int first = 0; first < significands; first += 32)
      check_rounding(
          (1 + at::arange(first, first + 32, patterns.options()) / significands).unsqueeze(1));
    if (dtype == at::kBFloat16) {
      auto floor = at::full({1, 1}, 1e-12, at::TensorOptions().dtype(dtype));
      if (floor.item<float>() > 1e-12f)
        floor.view(at::kShort).sub_(1);
      check_rounding(floor.to(at::kFloat));
    }
    if (at::globalContext().blasPreferredBackend() == at::BlasBackend::Cublaslt) {
      infinity::OrdinaryLinearExecutor ordinary;
      for (const auto [input, output] : {std::pair{1024, 3072}, std::pair{1536, 1024}}) {
        auto values = at::randn({32, input}, options) * .3;
        auto matrix = at::randn({output, input}, options) * .05;
        auto projected = ordinary.forward(values, matrix);
        TORCH_CHECK(projected.defined(), "Ordinary GEMM declined a supported projection");
        TORCH_CHECK(at::allclose(projected, at::linear(values, matrix), .01, .01),
                    "Ordinary GEMM projection disagrees with ATen");
      }
      auto x = at::randn({32, 768}, options) * .3;
      auto w = at::randn({2304, 768}, options) * .05;
      auto expected = at::linear(x, w);
      c10::cuda::getCurrentCUDAStream().synchronize();
      {
        infinity::OrdinaryLinearExecutor retry;
        auto state = infinity::workspace_admission_state(0);
        size_t available;
        {
          std::lock_guard lock(state->mutex);
          available = (state->limit ? state->limit : size_t(512) << 20) - state->retained;
        }
        auto pressure = infinity::reserve_workspace_memory(0, available);
        TORCH_CHECK(pressure, "Workspace pressure fixture failed to reserve remaining allowance");
        auto session = retry.prepare(x);
        TORCH_CHECK(!retry.forward(x, w, session).defined(), "Workspace cap pressure was ignored");
        pressure.reset();
        auto recovered = retry.forward(x, w, session);
        TORCH_CHECK(recovered.defined() && at::equal(recovered, expected),
                    "Released workspace allowance did not restore the cached supported plan");
        c10::cuda::getCurrentCUDAStream().synchronize();
      }
      at::Tensor retained;
      auto streams = std::array<c10::cuda::CUDAStream, 2>{c10::cuda::getStreamFromPool(),
                                                          c10::cuda::getStreamFromPool()};
      for (auto stream : streams) {
        c10::cuda::CUDAStreamGuard stream_guard(stream);
        auto session = ordinary.prepare(x);
        auto first = ordinary.forward(x, w, session);
        TORCH_CHECK(first.defined(), "Supported ordinary request session declined");
        TORCH_CHECK(at::equal(first, expected), "Ordinary session changed original GEMM bytes");
        auto saved = first.clone();
        for (int i = 0; i < 4; ++i) {
          auto next = ordinary.forward(x, w, session);
          TORCH_CHECK(next.defined() && next.const_data_ptr() != first.const_data_ptr(),
                      "Request session reused a live output allocation");
          TORCH_CHECK(at::equal(next, expected) && at::equal(first, saved),
                      "Ordinary session changed or invalidated owned output");
        }
        {
          c10::cuda::CUDAStreamGuard other_guard(stream == streams[0] ? streams[1] : streams[0]);
          TORCH_CHECK(!ordinary.forward(x, w, session).defined(), "Cross-stream session admitted");
        }
        if (c10::cuda::device_count() > 1) {
          c10::cuda::CUDAGuard other_device(at::Device(at::kCUDA, 1));
          TORCH_CHECK(!ordinary.forward(x, w, session).defined(), "Cross-device session admitted");
          TORCH_CHECK(c10::cuda::current_device() == 1, "Rejected session changed caller device");
        }
        TORCH_CHECK(at::equal(ordinary.forward(x, w, session), expected),
                    "Restored stream lost session dispatch");
        retained = first;
        stream.synchronize();
      }
      TORCH_CHECK(at::equal(retained, expected), "Session destruction invalidated owned output");
      infinity::OrdinaryLinearExecutor::Session retired;
      at::Tensor lifetime_output;
      {
        infinity::OrdinaryLinearExecutor temporary;
        retired = temporary.prepare(x);
        lifetime_output = temporary.forward(x, w, retired);
        TORCH_CHECK(lifetime_output.defined(), "Lifetime fixture declined");
      }
      infinity::OrdinaryLinearExecutor different;
      TORCH_CHECK(!different.forward(x, w, retired).defined(),
                  "Session admitted a different executor");
      std::exception_ptr worker_failure;
      std::thread worker([session = std::move(retired), &worker_failure]() mutable {
        try {
          c10::InferenceMode worker_inference;
          const int device = c10::cuda::device_count() > 1 ? 1 : 0;
          c10::cuda::CUDAGuard ambient(at::Device(at::kCUDA, device));
          C10_CUDA_CHECK(cudaFree(nullptr));
          CUcontext before{}, after{};
          auto get = c10::cuda::DriverAPI::get()->cuCtxGetCurrent_;
          TORCH_CHECK(get(&before) == CUDA_SUCCESS, "Missing worker context");
          session = infinity::OrdinaryLinearExecutor::Session{};
          TORCH_CHECK(
              c10::cuda::current_device() == device && get(&after) == CUDA_SUCCESS &&
                  after == before,
              "Foreign-thread last session destruction changed ambient CUDA device/context");
        } catch (...) {
          worker_failure = std::current_exception();
        }
      });
      worker.join();
      if (worker_failure)
        std::rethrow_exception(worker_failure);
      TORCH_CHECK(at::equal(lifetime_output, expected),
                  "Executor/session destruction invalidated owned output");
      if (c10::cuda::device_count() > 1) {
        c10::cuda::CUDAGuard ambient(at::Device(at::kCUDA, 1));
        auto local = ordinary.prepare(x);
        TORCH_CHECK(c10::cuda::current_device() == 1, "Prepare failed to restore caller device");
      }
      at::globalContext().setDeterministicAlgorithms(true, false);
      {
        auto unsupported = ordinary.prepare(x);
        TORCH_CHECK(!ordinary.forward(x, w, unsupported).defined(), "Unsupported policy admitted");
      }
      at::globalContext().setDeterministicAlgorithms(false, false);
      {
        c10::cuda::CUDAStreamGuard guard(streams[0]);
        cudaGraph_t graph{};
        C10_CUDA_CHECK(cudaStreamBeginCapture(streams[0], cudaStreamCaptureModeThreadLocal));
        auto capturing = ordinary.prepare(x);
        TORCH_CHECK(!ordinary.forward(x, w, capturing).defined(),
                    "Capture admitted ordinary session");
        C10_CUDA_CHECK(cudaStreamEndCapture(streams[0], &graph));
        C10_CUDA_CHECK(cudaGraphDestroy(graph));
      }
      std::cout << "Ordinary request-local stream/policy/capture and owned output: " << dtype
                << " passed\n";
    }
    {
      // Odd width forces scalar conversion; known row maxima give an independent oracle.
      auto edges = at::zeros({5, 9729}, options);
      edges.select(0, 0).fill_(-0.f);
      edges.select(0, 1).fill_(std::ldexp(1.f, dtype == at::kHalf ? -24 : -133));
      edges.select(0, 2).fill_(std::numeric_limits<float>::infinity());
      edges.select(0, 2).select(0, 1).zero_();
      edges.select(0, 3).fill_(std::numeric_limits<float>::quiet_NaN());
      edges.select(0, 4).fill_(3.5f);
      auto scales = at::full({5, 1}, 1e-12f * (1.f / 448.f), at::TensorOptions().dtype(at::kFloat));
      if (dtype == at::kHalf)
        scales.select(0, 1).fill_(std::ldexp(1.f, -24) * (1.f / 448.f));
      scales.select(0, 2).fill_(std::numeric_limits<float>::infinity());
      scales.select(0, 4).fill_(3.5f * (1.f / 448.f));
      auto expected = (edges.to(at::kCPU).to(at::kFloat) / scales).to(at::kFloat8_e4m3fn);
      auto actual = infinity::fp8_quantize_rows(edges);
      TORCH_CHECK(at::equal(actual.second.to(at::kCPU), scales), "Scalar FP8 edge scales differ");
      check_same_or_nan(
          expected.to(options.device()), actual.first, "Scalar FP8 edge conversion differs");
    }
    auto weight = at::randn({512, 256}, options) * .05;
    infinity::Fp8Linear linear(weight);
    for (int64_t rows : {1, 3, 7, 17, 35, 128}) {
      auto input = at::randn({rows, 256}, options) * .3;
      // Exact zero and sub-floor rows exercise the scale clamp, including
      // fp16 underflow. Row count remains the actual number of tokens.
      input.select(0, 0).zero_();
      if (rows > 1)
        input.select(0, 1).fill_(1e-14);
      auto quantized = infinity::fp8_quantize_rows(input);
      auto cpu = input.to(at::kCPU).to(at::kFloat);
      auto scale = at::amax(cpu.abs(), {1}, true).clamp_min(1e-12) * (1.f / 448.f);
      auto reference = (cpu / scale).clamp(-448., 448.).to(at::kFloat8_e4m3fn);
      TORCH_CHECK(at::equal(quantized.second.to(at::kCPU), scale),
                  "FP8 row-scale arithmetic differs from independent CPU oracle");
      TORCH_CHECK(
          at::equal(quantized.first.to(at::kCPU).view(at::kByte), reference.view(at::kByte)),
          "FP8 quantized bytes differ from independent CPU oracle");
      TORCH_CHECK(quantized.first.sizes() == input.sizes() && quantized.second.size(0) == rows,
                  "FP8 quantization padded actual token rows");
      auto output = linear.forward(input);
      TORCH_CHECK(at::equal(linear.forward_quantized(quantized.first, quantized.second), output),
                  "Prequantized input changed FP8 GEMM bytes");
      auto expected = fp8_original_recipe(
          quantized.first, linear.matrix(), quantized.second, linear.scales(), dtype);
      check_projection_close(
          expected, output, "Native Torch FP8 GEMM differs from portable reference");
      TORCH_CHECK(output.sizes() == at::IntArrayRef({rows, 512}) && output.scalar_type() == dtype,
                  "FP8 GEMM changed actual row count or model dtype");
      auto bias = at::randn({512}, options) * .01;
      TORCH_CHECK(at::equal(linear.forward(input, bias), output + bias),
                  "FP8 bias lost separate model-dtype rounding");
    }
    // Reusing a geometry must rebind both activation and weight scale pointers.
    infinity::Fp8Linear other(at::randn({512, 256}, options) * 2);
    for (const auto* projection : {&linear, &other, &linear}) {
      auto values = at::randn({17, 256}, options) * 3;
      auto quantized = infinity::fp8_quantize_rows(values);
      auto expected = fp8_original_recipe(
          quantized.first, projection->matrix(), quantized.second, projection->scales(), dtype);
      check_projection_close(expected,
                             projection->forward(values),
                             "Reused FP8 geometry retained another projection's scale pointers");
    }
    auto empty = linear.forward(at::empty({0, 256}, options));
    TORCH_CHECK(empty.sizes() == at::IntArrayRef({0, 512}), "FP8 empty rows not preserved");

    // The fused unary activation quantizers must equal the original composed recipe:
    // model-dtype projection/bias rounding, the original activation helper, then row quantization.
    for (auto kind : activation_kinds) {
      for (int64_t rows : {0, 1, 17}) {
        auto input = at::randn({rows, 256}, options) * .5;
        if (rows) {
          input.select(0, 0).zero_();
          if (rows > 1)
            input.select(0, 1).fill_(1e-14);
        }
        auto bias = at::randn({256}, options) * .01;
        check_activation_quantize(input, {}, kind, "Unbiased fused activation differs");
        check_activation_quantize(input, bias, kind, "Biased fused activation differs");
      }
    }
    // Width 20480 is the vector boundary; wider or unaligned rows use the scalar fallback.
    for (const auto& [width, rows] : std::array<std::pair<int64_t, int64_t>, 10>{std::pair{513, 2},
                                                                                 {8193, 2},
                                                                                 {12288, 2},
                                                                                 {12289, 2},
                                                                                 {16384, 2},
                                                                                 {20480, 2},
                                                                                 {20481, 2},
                                                                                 {20488, 2},
                                                                                 {32768, 1},
                                                                                 {65536, 1}})
      for (bool aligned : {false, true}) {
        auto storage = at::randn({rows * width + 1}, options) * 3;
        auto input = storage.narrow(0, aligned ? 0 : 1, rows * width).view({rows, width});
        input.select(0, 0).zero_();
        if (rows > 1)
          input.select(0, 1).narrow(0, 0, 8).fill_(
              std::ldexp(1.f, dtype == at::kHalf ? -24 : -133));
        auto bias = at::randn({width}, options) * .01;
        for (auto kind : activation_kinds) {
          check_activation_quantize(input, {}, kind, "Wide/unaligned fused activation differs");
          check_activation_quantize(input, bias, kind, "Wide biased fused activation differs");
        }
      }
    // Every finite model-dtype bit pattern, in both the vector and scalar fallback layouts.
    {
      auto words = at::arange(65536, at::TensorOptions().dtype(at::kInt)).to(at::kShort);
      auto finite = words.view(dtype).to(at::kFloat);
      finite = finite.masked_select(at::isfinite(finite));
      for (int64_t width : {2048, 17}) {
        auto padded = at::zeros({(finite.numel() + width - 1) / width * width}, finite.options());
        padded.narrow(0, 0, finite.numel()).copy_(finite);
        auto rows = padded.view({-1, width}).to(options);
        for (auto kind : activation_kinds)
          check_activation_quantize(rows, {}, kind, "Exhaustive fused activation differs");
      }
    }
    // Portable row/column scaling must round once to model dtype before the same activation.
    const auto activation_f32 = at::TensorOptions().device(at::kCUDA, 0).dtype(at::kFloat);
    for (bool aligned : {false, true})
      for (const auto& [rows, width] : std::array<std::pair<int64_t, int64_t>, 5>{
               std::pair{7, 2048}, {5, 2056}, {2, 20480}, {2, 20481}, {2, 20488}}) {
        auto value_storage = at::randn({rows * width + 1}, activation_f32) * 8;
        auto column_storage = at::rand({width + 1}, activation_f32) + .5;
        auto values = value_storage.narrow(0, aligned ? 0 : 1, rows * width).view({rows, width});
        auto columns = column_storage.narrow(0, aligned ? 0 : 1, width).view({width, 1});
        auto row_scales = at::rand({rows, 1}, activation_f32) + .5;
        const std::array<float, 6> scales{.25f, 2.f, 1e-40f, 1e6f, 0.f, -1.5f};
        for (int64_t row = 0; row < rows && row < int64_t(scales.size()); ++row)
          row_scales.select(0, row).fill_(scales[row]);
        columns.select(0, 0).fill_(2.f);
        columns.select(0, 1).fill_(.25f);
        columns.select(0, 2).fill_(0.f);
        columns.select(0, 3).fill_(std::ldexp(1.f, -149));
        columns.select(0, 4).fill_(1e6f);
        columns.select(0, 5).fill_(-.5f);
        values.select(0, 0).zero_();
        if (rows > 1) {
          auto row = values.select(0, 1);
          row.narrow(0, 0, 8).fill_(std::ldexp(1.f, -149));
          row.narrow(0, 8, 8).fill_(448.f);
          row.narrow(0, 16, 8).fill_(448.5f);
          row.narrow(0, 24, 8).fill_(464.f);
        }
        if (rows > 2)
          values.select(0, 2).fill_(1e4f);
        auto bias = at::randn({width}, options) * .01;
        for (auto kind : activation_kinds) {
          check_scaled_activation_quantize(
              values, row_scales, columns, dtype, {}, kind, "Scaled fused activation differs");
          check_scaled_activation_quantize(values,
                                           row_scales,
                                           columns,
                                           dtype,
                                           bias,
                                           kind,
                                           "Scaled biased fused activation differs");
        }
      }
    for (auto kind : activation_kinds) {
      auto empty_values = at::empty({0, 2048}, activation_f32);
      auto empty_rows = at::empty({0, 1}, activation_f32);
      auto empty_columns = at::empty({2048, 1}, activation_f32);
      check_scaled_activation_quantize(empty_values,
                                       empty_rows,
                                       empty_columns,
                                       dtype,
                                       {},
                                       kind,
                                       "Scaled fused activation changed zero-row geometry");
      check_scaled_activation_quantize(empty_values,
                                       empty_rows,
                                       empty_columns,
                                       dtype,
                                       at::randn({2048}, options) * .01,
                                       kind,
                                       "Scaled biased activation changed zero-row geometry");
    }
    // NaN/Inf inputs and scales retain the composed classification; NaN payloads are not compared.
    for (bool aligned : {false, true}) {
      constexpr int64_t rows = 5, width = 2056;
      const float nan = std::numeric_limits<float>::quiet_NaN();
      const float inf = std::numeric_limits<float>::infinity();
      auto storage = at::randn({rows * width + 1}, options) * 3;
      auto input = storage.narrow(0, aligned ? 0 : 1, rows * width).view({rows, width});
      input.select(0, 0).fill_(nan);
      input.select(0, 1).fill_(inf);
      input.select(0, 2).fill_(-inf);
      auto mixed = input.select(0, 3);
      mixed.narrow(0, 0, 8).fill_(nan);
      mixed.narrow(0, 8, 8).fill_(inf);
      mixed.narrow(0, 16, 8).fill_(-inf);
      auto bias = at::randn({width}, options) * .01;
      for (auto kind : activation_kinds) {
        check_activation_quantize(input, {}, kind, "Nonfinite fused activation differs");
        check_activation_quantize(input, bias, kind, "Nonfinite biased fused activation differs");
      }
    }
    for (bool aligned : {false, true}) {
      constexpr int64_t rows = 5, width = 2048;
      const float nan = std::numeric_limits<float>::quiet_NaN();
      const float inf = std::numeric_limits<float>::infinity();
      auto value_storage = at::randn({rows * width + 1}, activation_f32) * 3;
      auto column_storage = at::rand({width + 1}, activation_f32) + .5;
      auto values = value_storage.narrow(0, aligned ? 0 : 1, rows * width).view({rows, width});
      auto columns = column_storage.narrow(0, aligned ? 0 : 1, width).view({width, 1});
      auto row_scales = at::rand({rows, 1}, activation_f32) + .5;
      values.select(0, 0).fill_(nan);
      values.select(0, 1).fill_(inf);
      values.select(0, 2).fill_(-inf);
      row_scales.select(0, 0).fill_(nan);
      row_scales.select(0, 1).fill_(inf);
      row_scales.select(0, 2).fill_(-inf);
      row_scales.select(0, 3).fill_(0.f);
      columns.select(0, 0).fill_(nan);
      columns.select(0, 1).fill_(inf);
      columns.select(0, 2).fill_(-inf);
      columns.select(0, 3).fill_(0.f);
      for (auto kind : activation_kinds) {
        check_scaled_activation_quantize(
            values, row_scales, columns, dtype, {}, kind, "Nonfinite scaled activation differs");
        check_scaled_activation_quantize(values,
                                         row_scales,
                                         columns,
                                         dtype,
                                         at::randn({width}, options) * .01,
                                         kind,
                                         "Nonfinite scaled biased activation differs");
      }
    }
    // The projection boundary must use the same GEMM, bias and activation rounding.
    for (auto kind : activation_kinds)
      for (bool use_bias : {false, true})
        for (int64_t rows : {1, 17}) {
          auto input = at::randn({rows, 256}, options) * .5;
          auto bias = use_bias ? at::randn({512}, options) * .01 : at::Tensor();
          check_linear_activation(input, bias, kind, linear, "Fused linear activation differs");
        }
    for (auto kind : activation_kinds) {
      auto bias = at::randn({512}, options) * .01;
      auto empty_input = at::empty({0, 256}, options);
      auto empty_activation = linear.forward_activation_quantized(empty_input, {}, bias, kind);
      auto quantized_empty = infinity::fp8_quantize_rows(empty_input);
      auto empty_prequantized = linear.forward_activation_quantized(
          quantized_empty.first, quantized_empty.second, bias, kind);
      for (const auto& result : {empty_activation, empty_prequantized})
        TORCH_CHECK(result.first.sizes() == at::IntArrayRef({0, 512}) &&
                        result.first.scalar_type() == at::kFloat8_e4m3fn &&
                        result.second.sizes() == at::IntArrayRef({0, 1}) &&
                        result.second.scalar_type() == at::kFloat,
                    "Fused linear activation changed zero-row geometry");
    }
    for (int64_t width : {1024, 1032, 2048, 2056, 4096, 4104, 5120, 6144, 6152})
      for (int64_t offset : {0, 1}) {
        auto storage = at::randn({17 * width + offset}, options) * 3;
        auto x = storage.narrow(0, offset, 17 * width).view({17, width});
        x.select(0, 0).zero_();
        x.select(0, 1).fill_(1e-14);
        auto q = infinity::fp8_quantize_rows(x);
        auto cpu = x.to(at::kCPU).to(at::kFloat);
        auto scale = at::amax(cpu.abs(), {1}, true).clamp_min(1e-12) * (1.f / 448.f);
        auto expected = (cpu / scale).clamp(-448., 448.).to(at::kFloat8_e4m3fn);
        TORCH_CHECK(
            at::equal(q.second.to(at::kCPU), scale) &&
                at::equal(q.first.to(at::kCPU).view(at::kByte), expected.view(at::kByte)),
            "Vector quantization or misaligned fallback changed CPU reference bytes/scales");
      }
    for (int64_t width :
         {256, 2048, 4096, 6144, 8192, 9728, 12288, 14336, 17408, 18432, 24576, 28672, 32768})
      for (bool aligned : {false, true}) {
        // A contiguous storage offset exercises the scalar fallback as well as vector PDL.
        auto storage = at::randn({17 * 2 * width + 1}, options) * 8;
        auto gate_up = storage.narrow(0, aligned ? 0 : 1, 17 * 2 * width).view({17, 2 * width});
        gate_up.select(0, 0).zero_();
        auto activated = infinity::fused_gated_activation(gate_up, 5);
        auto reference = infinity::fp8_quantize_rows(activated);
        auto fused = infinity::fp8_swiglu_quantize(gate_up);
        TORCH_CHECK(at::equal(fused.first.view(at::kByte), reference.first.view(at::kByte)) &&
                        at::equal(fused.second, reference.second),
                    "Fused SwiGLU changed FP8 bytes/scales");
        infinity::Fp8Linear down(at::randn({256, width}, options) * .05);
        auto bias = at::randn({256}, options) * .01;
        TORCH_CHECK(at::equal(down.forward_quantized(fused.first, fused.second, bias),
                              down.forward(activated, bias)),
                    "Fused MLP changed output or bias rounding");
      }
    // Check every gate encoding against frozen scalar staging on each GPU family.
    // Constant rows expose activation differences in the scale, even after FP8 rounding.
    auto gates = at::arange(65536, options.dtype(at::kInt)).to(at::kShort).view(dtype);
    auto signed_zeros = at::ones({2, 4096}, options);
    signed_zeros.select(0, 0).zero_();
    signed_zeros.select(0, 1).select(0, 2048).fill_(-0.f);
    check_gated_quantize(signed_zeros,
                         infinity::Fp8GatedActivation::RoundedSilu,
                         true,
                         "Hoisted gated division changed signed zero");
    auto exhaustive = at::empty({65536, 16}, options);
    exhaustive.narrow(1, 0, 8).copy_(gates.unsqueeze(1));
    auto unit_rows = at::ones({65536, 1}, options.dtype(at::kFloat));
    auto unit_columns = at::ones({16, 1}, options.dtype(at::kFloat));
    for (float up : {-1.f, .5f, 1.f, 3.f}) {
      exhaustive.narrow(1, 8, 8).fill_(up);
      for (bool scaled : {false, true}) {
        auto values = scaled ? exhaustive.to(at::kFloat) : exhaustive;
        auto reference = scaled
                             ? infinity::detail::fp8_scaled_gated_quantize_impl(
                                   values,
                                   unit_rows,
                                   unit_columns,
                                   dtype,
                                   infinity::Fp8GatedActivation::RoundedSilu,
                                   true,
                                   true)
                             : infinity::detail::fp8_gated_quantize_impl(
                                   values, infinity::Fp8GatedActivation::RoundedSilu, true, true);
        auto actual =
            scaled ? infinity::fp8_scaled_swiglu_quantize(values, unit_rows, unit_columns, dtype)
                   : infinity::fp8_swiglu_quantize(values);
        check_same_or_nan(reference.first, actual.first, "Paired SwiGLU changed scalar FP8 bytes");
        check_same_or_nan(reference.second, actual.second, "Paired SwiGLU changed scalar scales");
      }
    }
    // Large gates and dtype-subnormal operands, including the misaligned scalar fallback.
    auto check_swiglu_pattern = [&](int64_t width, float gate_value, float up_value, bool aligned) {
      auto storage = at::empty({17 * 2 * width + 1}, options);
      auto gate_up = storage.narrow(0, aligned ? 0 : 1, 17 * 2 * width).view({17, 2 * width});
      gate_up.narrow(1, 0, width).fill_(gate_value);
      gate_up.narrow(1, width, width).fill_(up_value);
      auto activated = infinity::fused_gated_activation(gate_up, 5);
      auto reference = infinity::fp8_quantize_rows(activated);
      auto fused = infinity::fp8_swiglu_quantize(gate_up);
      TORCH_CHECK(at::equal(fused.first.view(at::kByte), reference.first.view(at::kByte)) &&
                      at::equal(fused.second, reference.second),
                  "Fused SwiGLU changed large/subnormal FP8 bytes/scales: width=",
                  width,
                  " gate=",
                  gate_value,
                  " up=",
                  up_value,
                  " aligned=",
                  aligned);
    };
    const float subnormal = dtype == at::kHalf ? std::ldexp(1.f, -24) : std::ldexp(1.f, -133);
    for (int64_t width : {6144, 9728, 12288, 17408})
      for (bool aligned : {false, true}) {
        check_swiglu_pattern(width, 80.f, 100.f, aligned);
        check_swiglu_pattern(width, subnormal, 2.f, aligned);
      }
    // Fused FP32 row/column scaling plus rounded SwiGLU must equal the composed recipe.
    auto check_scaled_swiglu = [&](const at::Tensor& values,
                                   const at::Tensor& row_scales,
                                   const at::Tensor& column_scales) {
      // Independent scalar composition: round the scaled FP32 rows once to model dtype, apply the
      // original gated activation, then run row quantization. This covers the aligned-pair staging
      // selected for the production wrappers as well as the unaligned/FP16 scalar fallback.
      auto scaled = infinity::fp8_scale_output(values, row_scales, column_scales, dtype);
      auto expected = infinity::fp8_quantize_rows(infinity::fused_gated_activation(scaled, 5));
      auto actual = infinity::fp8_scaled_swiglu_quantize(values, row_scales, column_scales, dtype);
      TORCH_CHECK(actual.first.sizes() == expected.first.sizes() &&
                      actual.second.sizes() == expected.second.sizes() &&
                      at::equal(actual.first.view(at::kByte), expected.first.view(at::kByte)) &&
                      at::equal(actual.second, expected.second),
                  "Scaled fused SwiGLU differs from scalar scale/activation/quantize: rows=",
                  values.size(0),
                  " width=",
                  values.size(1) / 2,
                  " dtype=",
                  dtype);
    };
    const auto f32 = at::TensorOptions().device(at::kCUDA, 0).dtype(at::kFloat);
    for (int64_t width : {2048, 2056, 4096, 6144, 9728, 12288, 17408, 20480, 20481})
      for (int64_t rows : {1, 17, 257}) {
        // Bound rows for the max-shared and wide-fallback widths.
        if (width >= 20480 && rows > 17)
          continue;
        for (bool aligned : {false, true}) {
          auto raw_storage = at::randn({rows * 2 * width + 1}, f32) * 8;
          auto column_storage = at::rand({2 * width + 1}, f32) + .5;
          auto values =
              raw_storage.narrow(0, aligned ? 0 : 1, rows * 2 * width).view({rows, 2 * width});
          auto columns = column_storage.narrow(0, aligned ? 0 : 1, 2 * width).view({2 * width, 1});
          auto row_scales = at::rand({rows, 1}, f32) + .5;
          values.select(0, 0).zero_();
          row_scales.select(0, 0).fill_(.25f);
          if (rows > 1) {
            values.select(0, 1).fill_(std::ldexp(1.f, -149));
            row_scales.select(0, 1).fill_(2.f);
          }
          if (rows > 2)
            row_scales.select(0, 2).fill_(.5f);
          columns.select(0, 0).fill_(2.f);
          columns.select(0, 1).fill_(.25f);
          check_scaled_swiglu(values, row_scales, columns);
        }
      }
    // Wider rows retain the original composed scaling/quantization recipe.
    {
      auto values = at::randn({17, 2 * 32768}, f32) * 8;
      auto columns = at::rand({2 * 32768, 1}, f32) + .5;
      auto row_scales = at::rand({17, 1}, f32) + .5;
      check_scaled_swiglu(values, row_scales, columns);
    }
    {
      auto empty_values = at::empty({0, 2 * 6144}, f32);
      auto empty_rows = at::empty({0, 1}, f32);
      auto empty_columns = at::empty({2 * 6144, 1}, f32);
      auto empty =
          infinity::fp8_scaled_swiglu_quantize(empty_values, empty_rows, empty_columns, dtype);
      TORCH_CHECK(empty.first.sizes() == at::IntArrayRef({0, 6144}) &&
                      empty.second.sizes() == at::IntArrayRef({0, 1}),
                  "Scaled fused SwiGLU changed zero-row geometry");
    }
    auto check_nonfinite_scaled_swiglu = [&](int64_t width, int64_t rows, bool aligned) {
      auto raw_storage = at::randn({rows * 2 * width + 1}, f32) * 3;
      auto column_storage = at::rand({2 * width + 1}, f32) + .5;
      auto values =
          raw_storage.narrow(0, aligned ? 0 : 1, rows * 2 * width).view({rows, 2 * width});
      auto columns = column_storage.narrow(0, aligned ? 0 : 1, 2 * width).view({2 * width, 1});
      auto row_scales = at::rand({rows, 1}, f32) + .5;
      const float nan = std::numeric_limits<float>::quiet_NaN();
      const float inf = std::numeric_limits<float>::infinity();
      values.select(0, 0).fill_(nan);
      values.select(0, 1).fill_(inf);
      values.select(0, 2).fill_(-inf);
      values.select(0, 3).narrow(0, 0, 8).fill_(nan);
      values.select(0, 3).narrow(0, 8, 8).fill_(inf);
      values.select(0, 3).narrow(0, 16, 8).fill_(-inf);
      row_scales.select(0, 0).fill_(nan);
      row_scales.select(0, 1).fill_(inf);
      row_scales.select(0, 2).fill_(-inf);
      row_scales.select(0, 3).fill_(0.f);
      columns.select(0, 0).fill_(nan);
      columns.select(0, 1).fill_(inf);
      columns.select(0, 2).fill_(-inf);
      columns.select(0, 3).fill_(0.f);
      // Compare against the original separate ATen products, then composed SwiGLU.
      auto expected_scaled = (values * row_scales * columns.t()).to(dtype);
      TORCH_CHECK(at::isnan(expected_scaled).any().item<bool>(),
                  "nonfinite fixture produced no NaN");
      TORCH_CHECK(at::isinf(expected_scaled).any().item<bool>(),
                  "nonfinite fixture produced no Inf");
      auto actual_scaled = infinity::fp8_scale_output(values, row_scales, columns, dtype);
      check_same_or_nan(expected_scaled, actual_scaled, "nonfinite scale output");
      auto expected = infinity::fp8_swiglu_quantize(expected_scaled);
      auto actual = infinity::fp8_scaled_swiglu_quantize(values, row_scales, columns, dtype);
      check_same_or_nan(expected.first, actual.first, "nonfinite scaled SwiGLU bytes");
      check_same_or_nan(expected.second, actual.second, "nonfinite scaled SwiGLU scales");
    };
    for (bool aligned : {false, true}) {
      check_nonfinite_scaled_swiglu(2056, 5, aligned);
      check_nonfinite_scaled_swiglu(20480, 4, aligned);
    }
    if (dtype == at::kBFloat16) {
      for (int64_t width : {768, 896, 2560, 4096, 5120, 8192})
        for (int64_t rows : {1, 17})
          for (bool aligned_weight : {false, true}) {
            auto input = at::randn({rows, width}, options);
            auto storage = at::randn({width + 1}, options);
            auto weight = storage.narrow(0, aligned_weight ? 0 : 1, width);
            for (bool add : {false, true})
              for (double epsilon : {1e-6, 1e-5}) {
                auto residual = add ? at::randn_like(input) : at::Tensor();
                auto norm = infinity::fused_add_rms_norm(input, residual, weight.clone(), epsilon);
                auto reference = infinity::fp8_quantize_rows(norm.first);
                auto [quantized, summed, scales] =
                    infinity::fp8_add_rms_quantize(input, residual, weight, epsilon);
                TORCH_CHECK(at::equal(quantized.view(at::kByte), reference.first.view(at::kByte)) &&
                                at::equal(scales, reference.second) &&
                                at::equal(summed, norm.second),
                            "Fused RMS changed FP8 bytes, scales or residual rounding");
              }
          }
    }
    // The fused gated quantizer must equal the original gated helper followed by row quantization
    // for both orientations, exact scalar/vector group boundaries and borrowed unaligned storage.
    const float gated_subnormal = dtype == at::kHalf ? std::ldexp(1.f, -24) : std::ldexp(1.f, -133);
    for (auto kind : gated_kinds)
      for (bool gate_first : {false, true})
        for (int64_t rows : {0, 1, 3, 17}) {
          auto gate_up = at::randn({rows, 512}, options) * .5;
          if (rows) {
            gate_up.select(0, 0).zero_();
            if (rows > 1) {
              auto row = gate_up.select(0, 1);
              row.narrow(0, 0, 8).fill_(gated_subnormal);
              row.narrow(0, 256, 8).fill_(gated_subnormal);
            }
          }
          check_gated_quantize(gate_up, kind, gate_first, "Fused gated activation differs");
        }
    for (const auto& [width, rows] : std::array<std::pair<int64_t, int64_t>, 14>{std::pair{17, 3},
                                                                                 {128, 3},
                                                                                 {2048, 3},
                                                                                 {2056, 2},
                                                                                 {8192, 2},
                                                                                 {8193, 2},
                                                                                 {12288, 2},
                                                                                 {16384, 2},
                                                                                 {20480, 2},
                                                                                 {20481, 2},
                                                                                 {20488, 2},
                                                                                 {24576, 1},
                                                                                 {28672, 1},
                                                                                 {32768, 1}})
      for (bool aligned : {false, true}) {
        auto storage = at::randn({rows * 2 * width + (aligned ? 0 : 1)}, options) * 3;
        auto gate_up = storage.narrow(0, aligned ? 0 : 1, rows * 2 * width).view({rows, 2 * width});
        gate_up.select(0, 0).zero_();
        if (rows > 1) {
          auto row = gate_up.select(0, 1);
          row.narrow(0, 0, 8).fill_(gated_subnormal);
          row.narrow(0, width, 8).fill_(gated_subnormal);
        }
        for (auto kind : gated_kinds)
          for (bool gate_first : {false, true})
            check_gated_quantize(gate_up, kind, gate_first, "Gated width group/alignment differs");
      }
    // Every finite model-dtype gate pattern, ordered by magnitude so rows retain midrange
    // sensitivity, with constant up and with varying moderate positive/negative up values.
    {
      auto words = at::arange(65536, at::TensorOptions().dtype(at::kInt)).to(at::kShort);
      auto finite = words.view(dtype).to(at::kFloat);
      finite = finite.masked_select(at::isfinite(finite));
      finite = finite.index_select(0, at::argsort(finite.abs()));
      for (int64_t width : {17, 128, 2048}) {
        auto padded = at::zeros({(finite.numel() + width - 1) / width * width}, finite.options());
        padded.narrow(0, 0, finite.numel()).copy_(finite);
        auto gate_values = padded.view({-1, width});
        for (int up_mode : {0, 1}) {
          auto up = up_mode == 0 ? at::ones(gate_values.sizes(), gate_values.options())
                                 : at::arange(width, gate_values.options())
                                       .remainder(11)
                                       .sub(5)
                                       .mul(.5)
                                       .unsqueeze(0)
                                       .expand(gate_values.sizes());
          for (auto kind : gated_kinds)
            for (bool gate_first : {false, true}) {
              auto gate_up =
                  at::empty({gate_values.size(0), 2 * width}, gate_values.options().dtype(dtype));
              auto gate_half =
                  gate_first ? gate_up.narrow(1, 0, width) : gate_up.narrow(1, width, width);
              auto up_half =
                  gate_first ? gate_up.narrow(1, width, width) : gate_up.narrow(1, 0, width);
              gate_half.copy_(gate_values);
              up_half.copy_(up);
              check_gated_quantize(
                  gate_up.to(options), kind, gate_first, "Exhaustive gated activation differs");
            }
        }
      }
    }
    // Focused width-8 fixture for hazard H1: every finite model-dtype gate is the only
    // nonzero activation in its row, so the row maximum cannot hide a sub-ULP GELU staging
    // difference after quantization. Aligned rows exercise the native pair recipe; offset-1
    // storage exercises the separate scalar staging.
    {
      const int64_t isolated_width = 8;
      auto isolated_words = at::arange(65536, at::TensorOptions().dtype(at::kInt)).to(at::kShort);
      auto isolated_gates = isolated_words.view(dtype).to(at::kFloat);
      isolated_gates = isolated_gates.masked_select(at::isfinite(isolated_gates));
      const int64_t isolated_rows = isolated_gates.numel();
      auto isolated_up = at::arange(isolated_rows, f32).remainder(9).sub(4).mul_(.5);
      for (bool gate_first : {false, true})
        for (bool aligned : {false, true}) {
          auto isolated_storage =
              at::zeros({isolated_rows * 2 * isolated_width + (aligned ? 0 : 1)}, options);
          auto isolated_gate_up =
              isolated_storage.narrow(0, aligned ? 0 : 1, isolated_rows * 2 * isolated_width)
                  .view({isolated_rows, 2 * isolated_width});
          auto isolated_gate_half =
              gate_first ? isolated_gate_up.narrow(1, 0, isolated_width)
                         : isolated_gate_up.narrow(1, isolated_width, isolated_width);
          auto isolated_up_half = gate_first
                                      ? isolated_gate_up.narrow(1, isolated_width, isolated_width)
                                      : isolated_gate_up.narrow(1, 0, isolated_width);
          isolated_up_half.fill_(1);
          isolated_gate_half.select(1, 0).copy_(isolated_gates.to(options));
          check_gated_quantize(isolated_gate_up,
                               infinity::Fp8GatedActivation::RoundedGelu,
                               gate_first,
                               "Isolated aligned/misaligned GELU pattern differs");
          isolated_up_half.select(1, 0).copy_(isolated_up);
          check_gated_quantize(isolated_gate_up,
                               infinity::Fp8GatedActivation::RoundedGelu,
                               gate_first,
                               "Isolated varying-up GELU pattern differs");
        }
      // The scaled reference allocates an aligned model-dtype tensor, so its native pair
      // staging is selected from width alone even when the raw FP32 source is unaligned.
      auto isolated_row_scales = at::ones({isolated_rows, 1}, f32);
      auto isolated_columns = at::ones({2 * isolated_width, 1}, f32);
      for (bool gate_first : {false, true})
        for (bool aligned : {false, true}) {
          auto isolated_value_storage =
              at::zeros({isolated_rows * 2 * isolated_width + (aligned ? 0 : 1)}, f32);
          auto isolated_values =
              isolated_value_storage.narrow(0, aligned ? 0 : 1, isolated_rows * 2 * isolated_width)
                  .view({isolated_rows, 2 * isolated_width});
          auto isolated_gate_half = gate_first
                                        ? isolated_values.narrow(1, 0, isolated_width)
                                        : isolated_values.narrow(1, isolated_width, isolated_width);
          auto isolated_up_half = gate_first
                                      ? isolated_values.narrow(1, isolated_width, isolated_width)
                                      : isolated_values.narrow(1, 0, isolated_width);
          isolated_up_half.fill_(1);
          isolated_gate_half.select(1, 0).copy_(isolated_gates);
          check_scaled_gated_quantize(isolated_values,
                                      isolated_row_scales,
                                      isolated_columns,
                                      dtype,
                                      infinity::Fp8GatedActivation::RoundedGelu,
                                      gate_first,
                                      "Isolated scaled GELU pattern differs");
        }
    }
    // Practical near-zero curvature values with independent up operands.
    for (int64_t width : {17, 128, 2048}) {
      auto gate = at::rand({5, width}, options) * 16 - 8;
      auto up = at::rand({5, width}, options) * 8 - 4;
      gate.select(0, 0).zero_();
      gate.select(0, 1).narrow(0, 0, 8).fill_(std::ldexp(1.f, -10));
      gate.select(0, 2).narrow(0, 0, 8).fill_(.5f);
      up.select(0, 3).narrow(0, 0, 8).fill_(-2.f);
      for (auto kind : gated_kinds)
        for (bool gate_first : {false, true}) {
          auto gate_up = gate_first ? at::cat({gate, up}, 1) : at::cat({up, gate}, 1);
          check_gated_quantize(gate_up, kind, gate_first, "Near-zero gated curvature differs");
        }
    }
    // NaN/Inf gates and up operands retain the composed FP8 classification and zero rows.
    for (auto kind : gated_kinds)
      for (bool gate_first : {false, true}) {
        const int64_t rows = 6, width = 128;
        const float nan = std::numeric_limits<float>::quiet_NaN();
        const float inf = std::numeric_limits<float>::infinity();
        auto gate = at::randn({rows, width}, options) * 2;
        auto up = at::rand({rows, width}, options) * 4 - 2;
        gate.select(0, 0).fill_(nan);
        gate.select(0, 1).fill_(inf);
        gate.select(0, 2).fill_(-inf);
        gate.select(0, 3).fill_(gated_subnormal);
        gate.select(0, 3).narrow(0, 0, 8).fill_(nan);
        gate.select(0, 3).narrow(0, 8, 8).fill_(inf);
        up.select(0, 4).narrow(0, 0, 8).fill_(nan);
        up.select(0, 4).narrow(0, 8, 8).fill_(inf);
        up.select(0, 4).narrow(0, 16, 8).fill_(-inf);
        up.select(0, 5).zero_();
        auto gate_up = gate_first ? at::cat({gate, up}, 1) : at::cat({up, gate}, 1);
        check_gated_quantize(gate_up, kind, gate_first, "Nonfinite gated activation differs");
      }
    {
      auto gate_up = at::randn({17, 2 * 2048}, options) * 3;
      gate_up.select(0, 0).zero_();
      auto legacy = infinity::fp8_swiglu_quantize(gate_up);
      auto gated =
          infinity::fp8_gated_quantize(gate_up, infinity::Fp8GatedActivation::RoundedSilu, true);
      check_same_or_nan(gated.first, legacy.first, "Legacy SwiGLU changed FP8 bytes");
      check_same_or_nan(gated.second, legacy.second, "Legacy SwiGLU changed FP8 scales");
    }
    // Portable row/column scaling must round once to model dtype before the same gated activation.
    for (int64_t width : {17, 128, 2048, 2056, 8192, 12288, 16384, 20480, 20481, 20488, 32768})
      for (int64_t rows : {1, 5})
        for (bool aligned : {false, true}) {
          auto raw_storage = at::randn({rows * 2 * width + 1}, f32) * 8;
          auto column_storage = at::rand({2 * width + 1}, f32) + .5;
          auto values =
              raw_storage.narrow(0, aligned ? 0 : 1, rows * 2 * width).view({rows, 2 * width});
          auto columns = column_storage.narrow(0, aligned ? 0 : 1, 2 * width).view({2 * width, 1});
          auto row_scales = at::rand({rows, 1}, f32) + .5;
          values.select(0, 0).zero_();
          row_scales.select(0, 0).fill_(.25f);
          if (rows > 1) {
            values.select(0, 1).fill_(std::ldexp(1.f, -149));
            row_scales.select(0, 1).fill_(2.f);
          }
          if (rows > 2)
            row_scales.select(0, 2).fill_(1e-40f);
          if (rows > 3)
            row_scales.select(0, 3).fill_(1e6f);
          if (rows > 4)
            row_scales.select(0, 4).fill_(0.f);
          columns.select(0, 0).fill_(2.f);
          columns.select(0, 1).fill_(.25f);
          for (auto kind : gated_kinds)
            for (bool gate_first : {false, true})
              check_scaled_gated_quantize(values,
                                          row_scales,
                                          columns,
                                          dtype,
                                          kind,
                                          gate_first,
                                          "Scaled fused gated activation differs");
        }
    for (auto kind : gated_kinds)
      for (bool gate_first : {false, true}) {
        auto empty_values = at::empty({0, 2 * 6144}, f32);
        auto empty_rows = at::empty({0, 1}, f32);
        auto empty_columns = at::empty({2 * 6144, 1}, f32);
        auto empty = infinity::fp8_scaled_gated_quantize(
            empty_values, empty_rows, empty_columns, dtype, kind, gate_first);
        TORCH_CHECK(empty.first.sizes() == at::IntArrayRef({0, 6144}) &&
                        empty.first.scalar_type() == at::kFloat8_e4m3fn &&
                        empty.second.sizes() == at::IntArrayRef({0, 1}) &&
                        empty.second.scalar_type() == at::kFloat,
                    "Scaled fused gated activation changed zero-row geometry");
      }
    for (auto kind : gated_kinds)
      for (bool gate_first : {false, true})
        for (bool aligned : {false, true}) {
          const int64_t width = 2056, rows = 5;
          const float nan = std::numeric_limits<float>::quiet_NaN();
          const float inf = std::numeric_limits<float>::infinity();
          auto raw_storage = at::randn({rows * 2 * width + 1}, f32) * 3;
          auto column_storage = at::rand({2 * width + 1}, f32) + .5;
          auto values =
              raw_storage.narrow(0, aligned ? 0 : 1, rows * 2 * width).view({rows, 2 * width});
          auto columns = column_storage.narrow(0, aligned ? 0 : 1, 2 * width).view({2 * width, 1});
          auto row_scales = at::rand({rows, 1}, f32) + .5;
          values.select(0, 0).fill_(nan);
          values.select(0, 1).fill_(inf);
          values.select(0, 2).fill_(-inf);
          values.select(0, 3).narrow(0, 0, 8).fill_(nan);
          values.select(0, 3).narrow(0, 8, 8).fill_(inf);
          values.select(0, 3).narrow(0, 16, 8).fill_(-inf);
          values.select(0, 4).narrow(0, 0, 8).fill_(std::ldexp(1.f, -149));
          row_scales.select(0, 0).fill_(nan);
          row_scales.select(0, 1).fill_(inf);
          row_scales.select(0, 2).fill_(-inf);
          row_scales.select(0, 3).fill_(0.f);
          columns.select(0, 0).fill_(nan);
          columns.select(0, 1).fill_(inf);
          columns.select(0, 2).fill_(-inf);
          columns.select(0, 3).fill_(0.f);
          check_scaled_gated_quantize(values,
                                      row_scales,
                                      columns,
                                      dtype,
                                      kind,
                                      gate_first,
                                      "Nonfinite scaled gated activation differs");
        }
    {
      auto values = at::randn({17, 2 * 2056}, f32) * 8;
      auto columns = at::rand({2 * 2056, 1}, f32) + .5;
      auto row_scales = at::rand({17, 1}, f32) + .5;
      auto legacy = infinity::fp8_scaled_swiglu_quantize(values, row_scales, columns, dtype);
      auto gated = infinity::fp8_scaled_gated_quantize(
          values, row_scales, columns, dtype, infinity::Fp8GatedActivation::RoundedSilu, true);
      check_same_or_nan(gated.first, legacy.first, "Legacy scaled SwiGLU changed FP8 bytes");
      check_same_or_nan(gated.second, legacy.second, "Legacy scaled SwiGLU changed FP8 scales");
    }
    // The projection boundary must use the same GEMM, model-dtype rounding and gated quantization.
    for (auto kind : gated_kinds)
      for (bool gate_first : {false, true})
        for (int64_t rows : {1, 3, 17}) {
          auto input = at::randn({rows, 256}, options) * .5;
          input.select(0, 0).zero_();
          if (rows > 1) {
            auto row = input.select(0, 1);
            row.narrow(0, 0, 8).fill_(gated_subnormal);
          }
          check_linear_gated_quantize(
              input, linear, kind, gate_first, "Fused linear gated activation differs");
        }
    for (auto kind : gated_kinds)
      for (bool gate_first : {false, true}) {
        auto empty_input = at::empty({0, 256}, options);
        auto quantized_empty = infinity::fp8_quantize_rows(empty_input);
        auto empty_gated = linear.forward_gated_quantized(empty_input, {}, kind, gate_first);
        auto empty_prequantized = linear.forward_gated_quantized(
            quantized_empty.first, quantized_empty.second, kind, gate_first);
        for (const auto& result : {empty_gated, empty_prequantized})
          TORCH_CHECK(result.first.sizes() == at::IntArrayRef({0, 256}) &&
                          result.first.scalar_type() == at::kFloat8_e4m3fn &&
                          result.second.sizes() == at::IntArrayRef({0, 1}) &&
                          result.second.scalar_type() == at::kFloat,
                      "Fused linear gated activation changed zero-row geometry");
      }
    for (int64_t rows : {1, 17}) {
      auto input = at::randn({rows, 256}, options) * .5;
      auto quantized = infinity::fp8_quantize_rows(input);
      auto expected = linear.forward_gated_quantized(
          input, {}, infinity::Fp8GatedActivation::RoundedSilu, true);
      auto legacy = linear.forward_swiglu_quantized(input);
      check_same_or_nan(expected.first, legacy.first, "Legacy linear SwiGLU changed FP8 bytes");
      check_same_or_nan(expected.second, legacy.second, "Legacy linear SwiGLU changed FP8 scales");
      auto expected_prequantized = linear.forward_gated_quantized(
          quantized.first, quantized.second, infinity::Fp8GatedActivation::RoundedSilu, true);
      auto legacy_prequantized = linear.forward_swiglu_quantized(quantized.first, quantized.second);
      check_same_or_nan(expected_prequantized.first,
                        legacy_prequantized.first,
                        "Legacy prequantized linear SwiGLU changed FP8 bytes");
      check_same_or_nan(expected_prequantized.second,
                        legacy_prequantized.second,
                        "Legacy prequantized linear SwiGLU changed FP8 scales");
    }
    if (blackwell_bf16(dtype)) {
      // Exceed the fused gate-width limit with signed, exactly representable projections.
      auto columns = (at::arange(49152, options).remainder(5) - 2) * .0625;
      infinity::Fp8Linear wide(columns.unsqueeze(1).expand({49152, 64}).contiguous());
      auto input = at::full({3, 64}, .125, options);
      input.select(0, 1).neg_();
      check_linear_gated_quantize(input,
                                  wide,
                                  infinity::Fp8GatedActivation::RoundedSilu,
                                  true,
                                  "Wide gated qualified projection changed its activation recipe");
    }
    if (at::cuda::getCurrentDeviceProperties()->major != 9) {
      // Ragged rows and varied scales must match the per-architecture qualified
      // reference, while the direct helper coverage below stays portable.
      for (auto [rows, width] : {std::pair<int64_t, int64_t>{1, 1024},
                                 {17, 2560},
                                 {257, 8192},
                                 {1025, 2560},
                                 {4097, 256}}) {
        auto weight = at::randn({512, width}, options) * .05;
        weight.select(0, 0).zero_();
        weight.select(0, 511).mul_(1e4);
        infinity::Fp8Linear projection(weight);
        auto input = at::randn({rows, width}, options) * .3;
        auto quantized = infinity::fp8_quantize_rows(input);
        auto row_scales = at::rand({rows, 1}, options.dtype(at::kFloat)) + .5;
        row_scales.select(0, 0).fill_(1.0f);
        if (rows > 1)
          row_scales.select(0, 1).fill_(-1.0f);
        if (rows > 2)
          row_scales.select(0, 2).fill_(.5f);
        if (rows > 3)
          row_scales.select(0, 3).fill_(2.0f);
        if (rows > 4)
          row_scales.select(0, 4).fill_(1e-6f);
        if (rows > 5)
          row_scales.select(0, 5).fill_(1e6f);
        if (rows > 6)
          row_scales.select(0, 6).fill_(1e-40f);
        auto expected = fp8_original_recipe(
            quantized.first, projection.matrix(), row_scales, projection.scales(), dtype);
        auto actual = projection.forward_quantized(quantized.first, row_scales);
        check_projection_close(
            expected, actual, "Qualified FP8 projection differs from portable reference");
      }
      // Direct helper coverage: vector, ragged-width scalar and unaligned borrowed storage.
      for (auto shape : {std::array<int64_t, 4>{7, 256, 0, 0},
                         std::array<int64_t, 4>{65536, 16, 0, 0},
                         std::array<int64_t, 4>{7, 257, 0, 0},
                         std::array<int64_t, 4>{7, 256, 1, 0},
                         std::array<int64_t, 4>{7, 256, 0, 1}}) {
        const auto [rows, width, offset, scale_offset] = shape;
        auto value_storage = at::randn({rows * width + offset}, options.dtype(at::kFloat));
        auto values = value_storage.narrow(0, offset, rows * width).view({rows, width});
        auto row_scales = at::rand({rows, 1}, options.dtype(at::kFloat)) + .5;
        auto scale_storage = at::rand({width + scale_offset}, options.dtype(at::kFloat)) + .5;
        auto column_scales = scale_storage.narrow(0, scale_offset, width).view({width, 1});
        auto expected = (values * row_scales * column_scales.t()).to(dtype);
        auto actual = infinity::fp8_scale_output(values, row_scales, column_scales, dtype);
        TORCH_CHECK(values.is_contiguous() && column_scales.is_contiguous() &&
                        actual.sizes() == expected.sizes() &&
                        at::equal(actual.view(at::kByte), expected.view(at::kByte)),
                    "Portable FP8 output scaling failed ragged/unaligned storage: rows=",
                    rows,
                    " width=",
                    width,
                    " value offset=",
                    offset,
                    " scale offset=",
                    scale_offset,
                    " dtype=",
                    dtype);
      }
    }
    if (at::cuda::getCurrentDeviceProperties()->major >= 10) {
      // The Blackwell BF16 provider rounds once to BF16; the portable recipe keeps
      // an FP32 product until that same rounding. FP16 stays portable, so its
      // bytes must match the portable recipe exactly.
      for (const auto& shape : {std::array<int64_t, 3>{0, 2560, 512},
                                std::array<int64_t, 3>{8, 64, 48},
                                std::array<int64_t, 3>{17, 2560, 512},
                                std::array<int64_t, 3>{257, 8192, 256},
                                std::array<int64_t, 3>{1025, 9728, 512},
                                std::array<int64_t, 3>{4097, 256, 512},
                                std::array<int64_t, 3>{16384, 2560, 6144}}) {
        const auto [rows, width, out_width] = shape;
        auto weight = at::randn({out_width, width}, options) * .05;
        infinity::Fp8Linear projection(weight);
        if (rows == 0) {
          auto empty = projection.forward(at::empty({0, width}, options));
          TORCH_CHECK(empty.sizes() == at::IntArrayRef({0, out_width}) &&
                          empty.scalar_type() == dtype,
                      "Blackwell empty rows changed shape or dtype");
          continue;
        }
        auto input = at::randn({rows, width}, options) * .3;
        auto quantized = infinity::fp8_quantize_rows(input);
        auto actual = projection.forward_quantized(quantized.first, quantized.second);
        auto portable = fp8_portable_recipe(
            quantized.first, projection.matrix(), quantized.second, projection.scales(), dtype);
        if (dtype == at::kHalf) {
          TORCH_CHECK(at::equal(actual, portable),
                      "Blackwell FP16 left the portable recipe: rows=",
                      rows,
                      " width=",
                      width,
                      " out_width=",
                      out_width);
          continue;
        }
        check_projection_close(
            portable, actual, "Blackwell BF16 row-scaled drift exceeded one BF16 ULP");
      }
      if (blackwell_bf16(dtype)) {
        // Borrowed FP8 storage must be 16-byte aligned for the Blackwell TMA path.
        constexpr int64_t rows = 17, width = 2560, out_width = 512;
        auto weight = at::randn({out_width, width}, options) * .05;
        infinity::Fp8Linear projection(weight);
        auto input = at::randn({rows, width}, options) * .3;
        auto quantized = infinity::fp8_quantize_rows(input);
        auto expected = projection.forward_quantized(quantized.first, quantized.second);
        auto aligned_storage = at::empty({quantized.first.numel() + 16}, quantized.first.options());
        auto aligned = aligned_storage.narrow(0, 16, quantized.first.numel()).view({rows, width});
        aligned.copy_(quantized.first);
        TORCH_CHECK(at::equal(projection.forward_quantized(aligned, quantized.second), expected),
                    "Blackwell aligned borrowed FP8 storage failed");
        auto unaligned_storage =
            at::empty({quantized.first.numel() + 1}, quantized.first.options());
        auto unaligned =
            unaligned_storage.narrow(0, 1, quantized.first.numel()).view({rows, width});
        unaligned.copy_(quantized.first);
        bool rejected = false;
        try {
          projection.forward_quantized(unaligned, quantized.second);
        } catch (const c10::Error&) {
          rejected = true;
        }
        TORCH_CHECK(rejected, "Blackwell unaligned FP8 activation was not rejected");
      }
    }
    if (dtype == at::kBFloat16) {
      // Gemma's precise GeGLU fused with row FP8 quantization must stay byte-exact against
      // gemma_gated_cuda followed by the production quantizer. Width 21504 is the 31B activation
      // width beyond fp8_scaled_swiglu_max_width; 2817 is ragged and 1024 exercises the grouped
      // vector staging. Misaligned storage forces the scalar staging.
      for (int64_t width : {21504, 2817, 1024})
        for (bool aligned : {false, true}) {
          constexpr int64_t rows = 2;
          auto storage = at::randn({rows * 2 * width + (aligned ? 0 : 1)}, options) * 2;
          auto gate_up =
              storage.narrow(0, aligned ? 0 : 1, rows * 2 * width).view({rows, 2 * width});
          gate_up.select(0, 0).zero_();
          gate_up.select(0, 1).narrow(0, 0, 8).fill_(std::ldexp(1.f, -10));
          check_precise_gated_quantize(gate_up, "Precise Gemma gated activation differs");
        }
      // Every finite model-dtype gate pattern at an aligned vector width must keep the scalar
      // staging; selecting the half-pair HFMA recipe changes individual FP8 bytes here.
      {
        constexpr int64_t width = 1024;
        auto gates = at::arange(65536, at::TensorOptions().dtype(at::kInt))
                         .to(at::kShort)
                         .view(dtype)
                         .to(at::kFloat);
        gates = gates.masked_select(at::isfinite(gates));
        auto padded = at::zeros({(gates.numel() + width - 1) / width * width}, gates.options());
        padded.narrow(0, 0, gates.numel()).copy_(gates);
        auto gate_values = padded.view({-1, width});
        auto up = at::arange(width, gate_values.options())
                      .remainder(11)
                      .sub(5)
                      .mul(.5)
                      .unsqueeze(0)
                      .expand(gate_values.sizes());
        auto gate_up = at::empty({gate_values.size(0), 2 * width}, options);
        gate_up.narrow(1, 0, width).copy_(gate_values);
        gate_up.narrow(1, width, width).copy_(up);
        check_precise_gated_quantize(gate_up, "Exhaustive precise Gemma gated activation differs");
      }
      // The wide projection boundary must consume the exact BF16 forward_quantized result and
      // never the portable FP32 unscaled product. 31B's gate/up width is 2 * 21504.
      {
        constexpr int64_t rows = 2, input_width = 16, gate_width = 21504;
        auto weight = at::randn({2 * gate_width, input_width}, options) * .05;
        infinity::Fp8Linear projection(weight);
        auto input = at::randn({rows, input_width}, options) * .3;
        auto quantized = infinity::fp8_quantize_rows(input);
        auto middle = projection.forward_quantized(quantized.first, quantized.second);
        auto expected = infinity::fp8_quantize_rows(infinity::gemma_gated_cuda(middle, true));
        auto actual = projection.forward_gated_quantized(
            input, {}, infinity::Fp8GatedActivation::PreciseGelu, true);
        check_same_or_nan(expected.first, actual.first, "Precise wide projection differs");
        check_same_or_nan(expected.second, actual.second, "Precise wide projection scales differ");
        auto prequantized = projection.forward_gated_quantized(
            quantized.first, quantized.second, infinity::Fp8GatedActivation::PreciseGelu, true);
        check_same_or_nan(expected.first, prequantized.first, "Precise wide prequantized differs");
        check_same_or_nan(
            expected.second, prequantized.second, "Precise wide prequantized scales differ");
      }
    }
    std::cout << "FP8 row-scale/bytes, odd real rows, model-dtype GEMM and bias: " << dtype
              << " passed\n";
  }
  for (auto shape : {std::pair<int64_t, int64_t>{511, 256}, {512, 255}}) {
    bool rejected = false;
    try {
      infinity::Fp8Linear invalid(at::zeros(
          {shape.first, shape.second}, at::TensorOptions().device(at::kCUDA, 0).dtype(at::kHalf)));
    } catch (const c10::Error&) {
      rejected = true;
    }
    TORCH_CHECK(rejected, "Unsupported FP8 projection width silently padded");
  }
}
