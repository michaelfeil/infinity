#include "helpers/fp8_linear.h"
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
#include <c10/core/InferenceMode.h>
#include <iostream>

int main() {
  // Match the original independently implemented recipe's workspace budget.
  setenv("CUBLASLT_WORKSPACE_SIZE", "32768", 1);
  c10::InferenceMode inference;
  at::manual_seed(7249);
  at::set_num_threads(4);
  at::globalContext().setAllowTF32CuBLAS(false);
  for (auto dtype : {at::kHalf, at::kBFloat16}) {
    auto options = at::TensorOptions().device(at::kCUDA, 0).dtype(dtype);
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
