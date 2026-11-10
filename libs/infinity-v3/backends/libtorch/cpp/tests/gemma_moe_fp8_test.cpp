// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
/// \file
/// Focused GPU parity test for the persistent FP8 routed-expert helper.

#include "kernels/moe_fp8.h"

#ifdef INFINITY_TORCH_MOE_FP8
#include "gemma_moe_model_test.h"
#include "helpers/fp8_linear.h"
#include "kernels/moe_kernels.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <system_error>

namespace {
int64_t token_count_argument(int argc, char** argv) {
  constexpr int64_t kDefaultTokens = 19;
  constexpr int64_t kMaxTokens = 129;
  if (argc < 2)
    return kDefaultTokens;
  if (argc > 2)
    throw std::invalid_argument("usage: gemma_moe_fp8_test [tokens 1..129]");
  const std::string_view text(argv[1]);
  int64_t tokens = 0;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), tokens);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || tokens <= 0 ||
      tokens > kMaxTokens)
    throw std::invalid_argument("usage: gemma_moe_fp8_test [tokens 1..129]");
  return tokens;
}

struct DiffStats {
  float max_abs, cosine, rmse;
};

DiffStats diff_stats(const at::Tensor& observed, const at::Tensor& expected) {
  const auto a = observed.to(at::kFloat), b = expected.to(at::kFloat);
  return DiffStats{(a - b).abs().max().item<float>(),
                   at::cosine_similarity(a, b, -1).min().item<float>(),
                   (a - b).square().mean().sqrt().item<float>()};
}

/// Aggregate cosine over every element. The per-token minimum used above is not
/// a stable parity statistic: one route combination with a small output norm can
/// dominate it even when the total error is a fraction of a BF16 ulp.
float global_cosine(const at::Tensor& observed, const at::Tensor& expected) {
  return at::cosine_similarity(observed.to(at::kFloat).flatten().unsqueeze(0),
                               expected.to(at::kFloat).flatten().unsqueeze(0),
                               -1)
      .item<float>();
}

/// Exact gemma4_moe_gelu_mul staging. The native kernel contracts the last
/// cubic multiply and the add into one FFMA (verified in its SASS), while
/// separate ATen elementwise ops round twice; emulate the fused operation in
/// double so this oracle compares GEMM arithmetic, not activation staging.
at::Tensor gelu_mul(const at::Tensor& gate_up_bf16, int64_t width) {
  const auto gate = gate_up_bf16.narrow(1, 0, width).to(at::kFloat);
  const auto up = gate_up_bf16.narrow(1, width, width).to(at::kFloat);
  const auto square = (0.044715f * gate) * gate;
  const auto alpha =
      (square.to(at::kDouble) * gate.to(at::kDouble) + gate.to(at::kDouble)).to(at::kFloat);
  const auto tangent = at::tanh(0.7978845608028654f * alpha);
  const auto gelu = (0.5f * gate) * (1.f + tangent);
  const auto rounded_gelu = gelu.to(at::kBFloat16).to(at::kFloat);
  return (rounded_gelu * up).to(at::kBFloat16);
}
} // namespace
#endif

int main(int argc, char** argv) {
#ifdef INFINITY_TORCH_MOE_FP8
  try {
    const int64_t tokens = token_count_argument(argc, argv);
    if (!at::cuda::is_available()) {
      std::cout << "SKIP: CUDA is unavailable\n";
      return 77;
    }
    const c10::Device device(c10::kCUDA, 0);
    if (!infinity::GemmaMoeFp8::supported(device)) {
      std::cout << "SKIP: native SM100 FP8 grouped provider is unavailable\n";
      return 77;
    }
    c10::cuda::CUDAGuard guard(device);
    {
      at::manual_seed(664);
      // Production Gemma4 expert geometry, so the BF16 native reference selects
      // its supported learned-scale path instead of declining the fixture.
      const int64_t experts = 128, hidden = 2816, intermediate = 704;
      auto options = at::TensorOptions().device(device).dtype(at::kBFloat16);
      auto input = at::randn({tokens, hidden}, options) * .25;
      auto gate_up = at::randn({experts, 2 * intermediate, hidden}, options) * .04;
      auto down = at::randn({experts, hidden, intermediate}, options) * .04;
      auto logits = at::randn({tokens, experts}, options.dtype(at::kFloat));
      // Force a stable empty-expert tail. The route count scales with the token
      // argument, exercising zero-count grouped GEMM groups, uneven row counts,
      // and the lower-ID tie rule without host routing metadata.
      logits.narrow(1, experts - 32, 32).fill_(-1e4);
      logits[0].fill_(0);
      auto scales = at::linspace(.25, 1.5, experts, logits.options());

      infinity::GemmaMoeFp8 routed(gate_up, down);
      TORCH_CHECK(routed.defined() && routed.experts() == experts && routed.hidden() == hidden &&
                      routed.intermediate() == intermediate,
                  "FP8 routed helper lost its prepared geometry");
      auto actual = routed.forward(input, logits, scales);
      TORCH_CHECK(actual.defined() && actual.sizes() == input.sizes(),
                  "FP8 routed helper returned the wrong shape");
      std::cout << "Gemma FP8 routed helper prepared workspace=" << routed.workspace_bytes(tokens)
                << " bytes for " << tokens << " tokens\n";

      // Quantize the original rows exactly like the helper constructor and the
      // quantize-once-before-fanout front end. The oracle then only differs from
      // production in FP32 GEMM evaluation order.
      auto input_q = infinity::fp8_quantize_rows(input);
      auto input_fp8_values = input_q.first.to(at::kFloat);
      auto gate_up_q =
          infinity::fp8_quantize_rows(gate_up.reshape({experts * 2 * intermediate, hidden}));
      auto gate_up_fp8 = gate_up_q.first.reshape({experts, 2 * intermediate, hidden});
      auto gate_up_scales = gate_up_q.second.reshape({experts, 2 * intermediate, 1});
      auto down_q = infinity::fp8_quantize_rows(down.reshape({experts * hidden, intermediate}));
      auto down_fp8 = down_q.first.reshape({experts, hidden, intermediate});
      auto down_scales = down_q.second.reshape({experts, hidden, 1});

      // Reproduce the native 128-expert/top-8 stable ordering, renormalization,
      // and learned per-expert scales used by gemma4_moe_route_128_8_f32.
      auto probability = at::softmax(logits, -1);
      auto ids = at::argsort(probability, true, -1, true).narrow(-1, 0, 8);
      auto routes = probability.gather(-1, ids);
      routes = routes / routes.sum(-1, true);
      routes = routes * scales.index_select(0, ids.flatten()).view_as(ids);

      // Provider-order reference: the native epilogue accumulates raw E4M3
      // products, then applies `(accum * scale_a) * scale_b` in FP32. Gate/up
      // rounds to BF16 before the exact GeGLU staging and down stays FP32 for the
      // BF16 weighted combine.
      auto provider_output = at::empty({experts, tokens, hidden}, logits.options());
      for (int64_t expert = 0; expert < experts; ++expert) {
        const auto gate_weight = gate_up_fp8.select(0, expert).to(at::kFloat);
        const auto gate_scale = gate_up_scales.select(0, expert);
        const auto down_weight = down_fp8.select(0, expert).to(at::kFloat);
        const auto down_scale = down_scales.select(0, expert);

        const auto gate_accumulator = at::matmul(input_fp8_values, gate_weight.transpose(0, 1));
        const auto provider_gate_up =
            ((gate_accumulator * input_q.second) * gate_scale.transpose(0, 1)).to(at::kBFloat16);
        const auto provider_activated = gelu_mul(provider_gate_up, intermediate);
        const auto provider_activated_q = infinity::fp8_quantize_rows(provider_activated);
        const auto down_accumulator =
            at::matmul(provider_activated_q.first.to(at::kFloat), down_weight.transpose(0, 1));
        const auto provider_contribution =
            (down_accumulator * provider_activated_q.second) * down_scale.transpose(0, 1);

        provider_output.select(0, expert).copy_(provider_contribution);
      }

      const auto combine = [&](const at::Tensor& expert_output) {
        auto selected =
            expert_output.transpose(0, 1).gather(1, ids.unsqueeze(-1).expand({tokens, 8, hidden}));
        auto weighted = (selected * routes.unsqueeze(-1)).to(at::kBFloat16).to(at::kFloat);
        auto oracle = at::zeros_like(weighted.select(1, 0));
        for (int64_t rank = 0; rank < 8; ++rank)
          oracle = oracle + weighted.select(1, rank);
        return oracle.to(at::kBFloat16);
      };

      auto provider = combine(provider_output);

      const auto fp8_stats = diff_stats(actual, provider);
      const auto fp8_global_cosine = global_cosine(actual, provider);
      std::cout << "Native FP8 vs provider-order oracle: max_abs=" << fp8_stats.max_abs
                << " global_cosine=" << fp8_global_cosine
                << " min_token_cosine=" << fp8_stats.cosine << " rmse=" << fp8_stats.rmse << "\n";

      // The helper is validated against the provider's documented epilogue order.
      // A BF16 output can only differ at its own rounding granularity; allow two
      // BF16 ulps of the fixture peak (the observed difference is one ulp on a
      // handful of elements) and retain the original per-token cosine threshold.
      // Anything larger means routing, quantize-once packing, GeGLU staging, or
      // the combine moved rather than FP32 accumulation order.
      const float output_peak = provider.abs().max().item<float>();
      const float two_output_ulps = 2.f * std::ldexp(1.f, -7) * output_peak;
      TORCH_CHECK(fp8_stats.max_abs <= two_output_ulps && fp8_stats.max_abs < .005f &&
                      fp8_stats.cosine > .99999f,
                  "Native FP8 routed kernel diverged from provider-order FP8 oracle: max_abs=",
                  fp8_stats.max_abs,
                  " (two output ulps ",
                  two_output_ulps,
                  ") global_cosine=",
                  fp8_global_cosine,
                  " min_token_cosine=",
                  fp8_stats.cosine,
                  " rmse=",
                  fp8_stats.rmse);

      auto reference = infinity::routed_moe_cuda(input, logits, gate_up, down, true, scales);
      if (reference.defined()) {
        const auto bf16_stats = diff_stats(actual, reference);
        std::cout << "FP8 vs BF16 (intended quantization loss, not a bitwise ABI): max_abs="
                  << bf16_stats.max_abs << " cosine=" << bf16_stats.cosine
                  << " rmse=" << bf16_stats.rmse << "\n";
      } else
        std::cout << "BF16 routed reference unavailable; skipped model-quality comparison\n";

      // Cached expert weights/scales must produce identical replay output.
      auto replay = routed.forward(input, logits, scales);
      TORCH_CHECK(at::equal(actual, replay), "FP8 routed helper reused stale cached weights");

      if (tokens == 129) {
        // At the 128-row SF boundary, route every token through experts 0..7 so
        // each active expert owns a full padded tile. Reuse the provider table.
        auto compact_logits = at::full({tokens, experts}, -1e4f, logits.options());
        compact_logits.narrow(1, 0, 8).fill_(0);
        auto compact_actual = routed.forward(input, compact_logits, scales);
        auto compact_oracle = at::zeros_like(provider_output.select(0, 0));
        for (int64_t expert = 0; expert < 8; ++expert)
          compact_oracle =
              compact_oracle + (provider_output.select(0, expert) * (scales[expert] / 8.f))
                                   .to(at::kBFloat16)
                                   .to(at::kFloat);
        compact_oracle = compact_oracle.to(at::kBFloat16);
        const auto compact_stats = diff_stats(compact_actual, compact_oracle);
        const float compact_peak = compact_oracle.abs().max().item<float>();
        TORCH_CHECK(compact_stats.max_abs <= 2.f * std::ldexp(1.f, -7) * compact_peak &&
                        compact_stats.max_abs < .005f && compact_stats.cosine > .99999f,
                    "Native FP8 compact 128-row expert case diverged: max_abs=",
                    compact_stats.max_abs,
                    " min_token_cosine=",
                    compact_stats.cosine,
                    " rmse=",
                    compact_stats.rmse);
      }

      at::cuda::getCurrentCUDAStream(device.index()).synchronize();
      const auto graph_stream = c10::cuda::getStreamFromPool(false, device.index());
      {
        c10::cuda::CUDAStreamGuard graph_guard(graph_stream);
        const auto graph_scratch_bytes = routed.workspace_bytes(tokens);
        auto graph_scratch = at::empty({static_cast<int64_t>(graph_scratch_bytes)},
                                       input.options().dtype(at::kByte));
        for (int warm = 0; warm < 3; ++warm)
          (void)routed.forward(
              input, logits, scales, graph_scratch.data_ptr(), graph_scratch_bytes);
        graph_stream.synchronize();

        at::cuda::CUDAGraph graph;
        at::Tensor captured;
        bool capturing = false;
        try {
          graph.capture_begin();
          capturing = true;
          captured =
              routed.forward(input, logits, scales, graph_scratch.data_ptr(), graph_scratch_bytes);
          graph.capture_end();
          capturing = false;
        } catch (const std::exception& error) {
          if (capturing) {
            try {
              graph.capture_end();
            } catch (...) {
            }
          }
          TORCH_CHECK(
              false, "Gemma FP8 routed helper graph capture is unsupported: ", error.what());
        }
        graph.replay();
        graph_stream.synchronize();
        auto baseline = captured.clone();

        input.add_(.5f);
        logits.select(1, 0).fill_(-1e4f);
        logits.select(1, experts - 1).fill_(1e4f);
        graph.replay();
        graph_stream.synchronize();
        auto eager =
            routed.forward(input, logits, scales, graph_scratch.data_ptr(), graph_scratch_bytes);
        graph_stream.synchronize();
        TORCH_CHECK(!at::equal(baseline, captured),
                    "Gemma FP8 captured graph ignored changed input/routing tensors");
        TORCH_CHECK(at::equal(captured, eager),
                    "Gemma FP8 graph replay diverged from fresh eager helper output");
      }
    }
    gemma_quantized_moe_model_graph_test(device, "fp8");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
#else
  (void)argc;
  (void)argv;
  std::cout << "SKIP: FP8 routed helper is not built\n";
  return 77;
#endif
}
