#include "helpers/dense_mlp.h"
#include "helpers/fp8_linear.h"
#include "kernels/encoder_activation.h"
#include "kernels/fp8_fusions.h"
#include "kernels/fast_kernels.h"
#include "helpers/workspace_admission.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Parallel.h>
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
  auto projected = fp8_original_recipe(
      quantized.first, linear.matrix(), quantized.second, linear.scales(), input.scalar_type());
  auto expected = infinity::fp8_quantize_rows(reference_activation(projected, bias, kind));
  auto actual = linear.forward_activation_quantized(input, {}, bias, kind);
  check_same_or_nan(expected.first, actual.first, label);
  check_same_or_nan(expected.second, actual.second, label);
  auto prequantized =
      linear.forward_activation_quantized(quantized.first, quantized.second, bias, kind);
  check_same_or_nan(expected.first, prequantized.first, label);
  check_same_or_nan(expected.second, prequantized.second, label);
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
      TORCH_CHECK(at::equal(output, expected),
                  "Native Torch FP8 GEMM differs from original independent cuBLASLt recipe: rows=",
                  rows,
                  " dtype=",
                  dtype);
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
      TORCH_CHECK(at::equal(projection->forward(values), expected),
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
         {256, 2048, 4096, 6144, 8192, 12288, 14336, 17408, 18432, 24576, 28672, 32768})
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
    for (int64_t width : {6144, 12288, 17408})
      for (bool aligned : {false, true}) {
        check_swiglu_pattern(width, 80.f, 100.f, aligned);
        check_swiglu_pattern(width, subnormal, 2.f, aligned);
      }
    // Fused FP32 row/column scaling plus rounded SwiGLU must equal the composed recipe.
    auto check_scaled_swiglu = [&](const at::Tensor& values,
                                   const at::Tensor& row_scales,
                                   const at::Tensor& column_scales) {
      auto expected = infinity::fp8_swiglu_quantize(
          infinity::fp8_scale_output(values, row_scales, column_scales, dtype));
      auto actual = infinity::fp8_scaled_swiglu_quantize(values, row_scales, column_scales, dtype);
      TORCH_CHECK(actual.first.sizes() == expected.first.sizes() &&
                      actual.second.sizes() == expected.second.sizes() &&
                      at::equal(actual.first.view(at::kByte), expected.first.view(at::kByte)) &&
                      at::equal(actual.second, expected.second),
                  "Scaled fused SwiGLU differs from scale_output + swiglu: rows=",
                  values.size(0),
                  " width=",
                  values.size(1) / 2,
                  " dtype=",
                  dtype);
    };
    const auto f32 = at::TensorOptions().device(at::kCUDA, 0).dtype(at::kFloat);
    for (int64_t width : {2048, 2056, 4096, 6144, 12288, 17408, 20480, 20481})
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
    if (at::cuda::getCurrentDeviceProperties()->major != 9) {
      // Portable output scaling: ragged rows and varied scales must match the
      // reference's separate FP32 products and final model-dtype rounding.
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
        TORCH_CHECK(actual.sizes() == expected.sizes() &&
                        at::equal(actual.view(at::kByte), expected.view(at::kByte)),
                    "Portable FP8 output scaling differs from reference: rows=",
                    rows,
                    " width=",
                    width,
                    " dtype=",
                    dtype);
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
