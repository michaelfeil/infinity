// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
/// \file
/// Focused GPU parity test for the persistent MXFP8 routed-expert helper.

#include "kernels/moe_mxfp8.h"

#ifdef INFINITY_TORCH_MOE_MXFP8
#include "gemma_moe_model_test.h"
#include "kernels/gemma_moe_mxfp8_grouped_sm100.h"
#include "kernels/mxfp8_fusions.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <vector>

namespace {
int64_t token_count_argument(int argc, char** argv) {
  constexpr int64_t kDefaultTokens = 19;
  constexpr int64_t kMaxTokens = 129;
  if (argc < 2)
    return kDefaultTokens;
  if (argc > 2)
    throw std::invalid_argument("usage: gemma_moe_mxfp8_test [tokens 1..129]");
  const std::string_view text(argv[1]);
  int64_t tokens = 0;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), tokens);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || tokens <= 0 ||
      tokens > kMaxTokens)
    throw std::invalid_argument("usage: gemma_moe_mxfp8_test [tokens 1..129]");
  return tokens;
}

/// Exact gemma4_moe_gelu_mul staging. The native kernel contracts the last
/// cubic multiply and the add into one FMA, while separate ATen elementwise
/// ops round twice; emulate the fused operation in double, matching the FP8
/// oracle, so this reference compares quantization rather than staging.
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
#ifdef INFINITY_TORCH_MOE_MXFP8
  try {
    const int64_t tokens = token_count_argument(argc, argv);
    if (!at::cuda::is_available()) {
      std::cout << "SKIP: CUDA is unavailable\n";
      return 77;
    }
    const c10::Device device(c10::kCUDA, 0);
    if (!infinity::GemmaMoeMxfp8::supported(device)) {
      std::cout << "SKIP: native SM100 MXFP8 grouped provider is unavailable\n";
      return 77;
    }
    c10::cuda::CUDAGuard guard(device);
    {
      at::manual_seed(917);
      const int64_t experts = 128, hidden = 2816, intermediate = 704;

      // Layout eligibility: N=1408/2816 must be 128-aligned so the global
      // [experts*N,K] MX row quantizer yields contiguous per-expert SF slices.
      TORCH_CHECK(infinity::mxfp8_scale_elements(experts * 2 * intermediate, hidden) ==
                      experts *
                          infinity_moe_mxfp8::weight_scale_expert_bytes(2 * intermediate, hidden),
                  "MXFP8 gate/up SF layout is not expert-sliceable");
      TORCH_CHECK(infinity::mxfp8_scale_elements(experts * hidden, intermediate) ==
                      experts * infinity_moe_mxfp8::weight_scale_expert_bytes(hidden, intermediate),
                  "MXFP8 down SF layout is not expert-sliceable");
      TORCH_CHECK(infinity_moe_mxfp8::row_scale_inner_blocks(hidden) == 88 &&
                      infinity_moe_mxfp8::row_scale_inner_blocks(intermediate) == 24,
                  "MXFP8 K=704 must pad 22 scale blocks to 24");

      auto options = at::TensorOptions().device(device).dtype(at::kBFloat16);
      auto input = at::randn({tokens, hidden}, options) * .25;
      auto gate_up = at::randn({experts, 2 * intermediate, hidden}, options) * .04;
      auto down = at::randn({experts, hidden, intermediate}, options) * .04;
      auto logits = at::randn({tokens, experts}, options.dtype(at::kFloat));
      // Empty tail plus a multi-row/uneven distribution; first row exercises the
      // stable lower-ID tie rule.
      logits.narrow(1, experts - 32, 32).fill_(-1e4);
      logits[0].fill_(0);
      auto scales = at::linspace(.25, 1.5, experts, logits.options());

      infinity::GemmaMoeMxfp8 routed(gate_up, down);
      TORCH_CHECK(routed.defined() && routed.experts() == experts && routed.hidden() == hidden &&
                      routed.intermediate() == intermediate,
                  "MXFP8 routed helper lost its prepared geometry");
      auto actual = routed.forward(input, logits, scales);
      TORCH_CHECK(actual.defined() && actual.sizes() == input.sizes(),
                  "MXFP8 routed helper returned the wrong shape");
      std::cout << "Gemma MXFP8 routed helper workspace=" << routed.workspace_bytes(tokens)
                << " bytes for " << tokens << " tokens\n";

      const auto stats = [](const at::Tensor& observed, const at::Tensor& expected) {
        struct Result {
          float max_abs, cosine, rmse;
        };
        const auto a = observed.to(at::kFloat), b = expected.to(at::kFloat);
        return Result{(a - b).abs().max().item<float>(),
                      at::cosine_similarity(a, b, -1).min().item<float>(),
                      (a - b).square().mean().sqrt().item<float>()};
      };

      const auto inner_blocks = [](int64_t cols) { return ((cols / 32 + 3) / 4) * 4; };
      const auto scale_index = [](int64_t outer, int64_t block, int64_t inner) {
        const int64_t sf_outer = outer / 128;
        const int64_t inner_major = block & ~int64_t(3);
        return (inner_major + sf_outer * inner) * 128 + (outer % 32) * 16 +
               ((outer % 128) / 32) * 4 + (block % 4);
      };
      const auto dequantize = [&](const at::Tensor& values,
                                  const at::Tensor& scales,
                                  int64_t rows,
                                  int64_t cols) {
        const int64_t blocks = cols / 32, inner = inner_blocks(cols);
        std::vector<int64_t> index(static_cast<size_t>(rows * blocks));
        for (int64_t row = 0; row < rows; ++row)
          for (int64_t block = 0; block < blocks; ++block)
            index[static_cast<size_t>(row * blocks + block)] = scale_index(row, block, inner);
        auto gather = at::tensor(index, at::TensorOptions().dtype(at::kLong)).to(device);
        auto scale = at::exp2(
            scales.to(at::kLong).index_select(0, gather).to(at::kFloat).reshape({rows, blocks, 1}) -
            127.0f);
        return (values.to(at::kFloat).reshape({rows, blocks, 32}) * scale).reshape({rows, cols});
      };

      auto input_fp8 = infinity::mxfp8_quantize_rows(input);
      auto input_dequant = dequantize(input_fp8.values, input_fp8.scales, tokens, hidden);

      auto probability = at::softmax(logits, -1);
      auto ids = at::argsort(probability, true, -1, true).narrow(-1, 0, 8);
      auto routes = probability.gather(-1, ids);
      routes = routes / routes.sum(-1, true);
      routes = routes * scales.index_select(0, ids.flatten()).view_as(ids);

      auto expert_output = at::empty({experts, tokens, hidden}, logits.options());
      for (int64_t expert = 0; expert < experts; ++expert) {
        auto gate_weight = gate_up.select(0, expert);
        auto gate_fp8 = infinity::mxfp8_quantize_rows(gate_weight);
        auto gate_dequant = dequantize(gate_fp8.values, gate_fp8.scales, 2 * intermediate, hidden);
        auto gate_up_bf16 =
            at::matmul(input_dequant, gate_dequant.transpose(0, 1)).to(at::kBFloat16);

        auto activated = gelu_mul(gate_up_bf16, intermediate);

        auto activated_fp8 = infinity::mxfp8_quantize_rows(activated);
        if (tokens == 129 && expert == 0) {
          // The fused helper must reproduce the activation and block-quantization
          // bytes exactly. An identity row map keeps logical and physical rows aligned.
          auto row_map = at::arange(tokens, at::TensorOptions().dtype(at::kInt).device(device));
          auto fused = infinity::mxfp8_moe_gelu_quantize(gate_up_bf16, row_map);
          TORCH_CHECK(
              fused.values.sizes() == activated_fp8.values.sizes() &&
                  fused.scales.sizes() == activated_fp8.scales.sizes() &&
                  at::equal(fused.values.view(at::kByte), activated_fp8.values.view(at::kByte)) &&
                  at::equal(fused.scales, activated_fp8.scales),
              "Fused MXFP8 MoE GELU quantization diverged from activation + row quantizer");
        }
        auto activated_dequant =
            dequantize(activated_fp8.values, activated_fp8.scales, tokens, intermediate);
        auto down_weight = down.select(0, expert);
        auto down_fp8 = infinity::mxfp8_quantize_rows(down_weight);
        auto down_dequant = dequantize(down_fp8.values, down_fp8.scales, hidden, intermediate);
        expert_output.select(0, expert).copy_(
            at::matmul(activated_dequant, down_dequant.transpose(0, 1)));
      }
      auto selected =
          expert_output.transpose(0, 1).gather(1, ids.unsqueeze(-1).expand({tokens, 8, hidden}));
      auto weighted = (selected * routes.unsqueeze(-1)).to(at::kBFloat16).to(at::kFloat);
      auto oracle = at::zeros_like(weighted.select(1, 0));
      for (int64_t rank = 0; rank < 8; ++rank)
        oracle = oracle + weighted.select(1, rank);
      oracle = oracle.to(at::kBFloat16);

      const auto mx_stats = stats(actual, oracle);
      TORCH_CHECK(mx_stats.max_abs < .005f && mx_stats.cosine > .99999f,
                  "Native MXFP8 routed kernel diverged from dequantized-MXFP8 oracle: max_abs=",
                  mx_stats.max_abs,
                  " cosine=",
                  mx_stats.cosine,
                  " rmse=",
                  mx_stats.rmse);

      auto replay = routed.forward(input, logits, scales);
      TORCH_CHECK(at::equal(actual, replay), "MXFP8 routed helper reused stale cached weights");
      std::cout << "Native MXFP8 vs dequantized-MXFP8 oracle: max_abs=" << mx_stats.max_abs
                << " cosine=" << mx_stats.cosine << " rmse=" << mx_stats.rmse << "\n";

      if (tokens == 129) {
        // At the 128-row SF boundary, route every token through experts 0..7 so
        // each active expert owns a full padded tile. Reuse the dequant table.
        auto compact_logits = at::full({tokens, experts}, -1e4f, logits.options());
        compact_logits.narrow(1, 0, 8).fill_(0);
        auto compact_actual = routed.forward(input, compact_logits, scales);
        auto compact_oracle = at::zeros_like(expert_output.select(0, 0));
        for (int64_t expert = 0; expert < 8; ++expert)
          compact_oracle =
              compact_oracle + (expert_output.select(0, expert) * (scales[expert] / 8.f))
                                   .to(at::kBFloat16)
                                   .to(at::kFloat);
        compact_oracle = compact_oracle.to(at::kBFloat16);
        const auto compact_stats = stats(compact_actual, compact_oracle);
        TORCH_CHECK(compact_stats.max_abs < .005f && compact_stats.cosine > .99999f,
                    "Native MXFP8 compact 128-row expert case diverged: max_abs=",
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
              false, "Gemma MXFP8 routed helper graph capture is unsupported: ", error.what());
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
                    "Gemma MXFP8 captured graph ignored changed input/routing tensors");
        TORCH_CHECK(at::equal(captured, eager),
                    "Gemma MXFP8 graph replay diverged from fresh eager helper output");
      }
    }
    gemma_quantized_moe_model_graph_test(device, "mxfp8");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
#else
  (void)argc;
  (void)argv;
  std::cout << "SKIP: MXFP8 routed helper is not built\n";
  return 77;
#endif
}
