// SPDX-License-Identifier: MIT
#include "kernels/moe_kernels.h"
#include "kernels/fast_kernels.h"
#include "kernels/qwen35_kernels.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstring>
#include <iostream>

int main() {
  try {
    at::manual_seed(773);
    auto opts = at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16);
    for (int experts : {128, 256})
      for (bool renormalize : {false, true}) {
        const int tokens = 13, hidden = 128, intermediate = 128;
        auto x = at::randn({tokens, hidden}, opts) * .25;
        auto gu = at::randn({experts, 2 * intermediate, hidden}, opts) * .04;
        auto down = at::randn({experts, hidden, intermediate}, opts) * .04;
        auto logits = at::randn({tokens, experts}, opts.dtype(at::kFloat));
        // Stable ties and experts with no routed rows are both valid.
        logits[0].fill_(0);
        auto actual = infinity::routed_moe_cuda(x, logits, gu, down, renormalize);
        TORCH_CHECK(actual.defined(), "Native BF16 routed kernel unavailable");
        auto p = at::softmax(logits, -1);
        auto ids = at::argsort(p, true, -1, true).narrow(-1, 0, 8);
        auto routes = p.gather(-1, ids);
        if (renormalize)
          routes = routes / routes.sum(-1, true);
        auto both = at::bmm(x.to(at::kFloat).unsqueeze(0).expand({experts, tokens, hidden}),
                            gu.to(at::kFloat).transpose(-2, -1))
                        .to(at::kBFloat16);
        auto activation =
            infinity::fused_gated_activation(both.reshape({experts * tokens, 2 * intermediate}), 0)
                .reshape({experts, tokens, intermediate});
        // FP32 down accumulators are weighted before each route's BF16 store.
        auto acc = at::bmm(activation.to(at::kFloat), down.to(at::kFloat).transpose(-2, -1))
                       .transpose(0, 1);
        auto selected = acc.gather(1, ids.unsqueeze(-1).expand({tokens, 8, hidden}));
        auto reference = (selected * routes.unsqueeze(-1))
                             .to(at::kBFloat16)
                             .to(at::kFloat)
                             .sum(1)
                             .to(at::kBFloat16);
        auto delta = (actual.to(at::kFloat) - reference.to(at::kFloat)).abs().max().item<float>();
        TORCH_CHECK(delta < .002, "FP32 routed epilogue oracle mismatch: ", delta);
        auto cos = at::cosine_similarity(actual.to(at::kFloat), reference.to(at::kFloat), -1)
                       .min()
                       .item<float>();
        TORCH_CHECK(cos > .9999, "Routed oracle cosine mismatch: ", cos);
        std::cout << "experts=" << experts << " renormalize=" << renormalize << " max_abs=" << delta
                  << " cosine=" << cos << "\n";
      }
    {
      // Enough routes to exercise uneven multi-row expert groups, not just
      // empty or single-row groups.
      const int experts = 128, tokens = 17, hidden = 2816, intermediate = 704;
      auto x = at::randn({tokens, hidden}, opts) * .25;
      auto gu = at::randn({experts, 2 * intermediate, hidden}, opts) * .02;
      auto down = at::randn({experts, hidden, intermediate}, opts) * .02;
      auto logits = at::randn({tokens, experts}, opts.dtype(at::kFloat));
      auto scales = at::linspace(.25, 1.5, experts, logits.options());
      auto actual = infinity::routed_moe_cuda(x, logits, gu, down, true, scales);
      TORCH_CHECK(actual.defined(), "Native Gemma weighted epilogue unavailable");
      auto p = at::softmax(logits, -1);
      auto ids = at::argsort(p, true, -1, true).narrow(-1, 0, 8);
      auto routes = p.gather(-1, ids);
      routes = routes / routes.sum(-1, true);
      routes = routes * scales.index_select(0, ids.flatten()).view_as(routes);
      auto both = at::bmm(x.to(at::kFloat).unsqueeze(0).expand({experts, tokens, hidden}),
                          gu.to(at::kFloat).transpose(-2, -1))
                      .to(at::kBFloat16);
      auto activation =
          infinity::gemma_gated_cuda(both.reshape({experts * tokens, 2 * intermediate}), true)
              .reshape({experts, tokens, intermediate});
      auto acc =
          at::bmm(activation.to(at::kFloat), down.to(at::kFloat).transpose(-2, -1)).transpose(0, 1);
      auto selected = acc.gather(1, ids.unsqueeze(-1).expand({tokens, 8, hidden}));
      auto reference = (selected * routes.unsqueeze(-1))
                           .to(at::kBFloat16)
                           .to(at::kFloat)
                           .sum(1)
                           .to(at::kBFloat16);
      auto delta = (actual.to(at::kFloat) - reference.to(at::kFloat)).abs().max().item<float>();
      auto cos = at::cosine_similarity(actual.to(at::kFloat), reference.to(at::kFloat), -1)
                     .min()
                     .item<float>();
      TORCH_CHECK(delta < .002 && cos > .9999,
                  "Gemma FP32 weighted epilogue mismatch: ",
                  delta,
                  " / ",
                  cos);
      std::cout << "Gemma learned expert scales max_abs=" << delta << " cosine=" << cos << "\n";

      // Direct capture coverage for the routed call itself. Warm the selected
      // provider, capture on a pooled nondefault stream, then mutate the same
      // input/logit storage and replay. Production model graph policy is not
      // changed: bridge-native MoE graphs remain limited to validated Hopper.
      {
        c10::cuda::CUDAGuard device_guard(x.device());
        C10_CUDA_CHECK(cudaDeviceSynchronize());
        auto stream = at::cuda::getStreamFromPool(false, x.device().index());
        c10::cuda::CUDAStreamGuard stream_guard(stream);
        const auto bits_equal = [](const at::Tensor& a, const at::Tensor& b) {
          auto ac = a.cpu().contiguous(), bc = b.cpu().contiguous();
          return ac.sizes() == bc.sizes() && ac.scalar_type() == bc.scalar_type() &&
                 std::memcmp(ac.data_ptr(), bc.data_ptr(), ac.numel() * ac.element_size()) == 0;
        };
        for (int run = 0; run < 3; ++run) {
          auto warm = infinity::routed_moe_cuda(x, logits, gu, down, true, scales);
          TORCH_CHECK(warm.defined(),
                      "Native Gemma weighted epilogue unavailable for graph warmup");
        }
        stream.synchronize();
        auto eager = infinity::routed_moe_cuda(x, logits, gu, down, true, scales);
        TORCH_CHECK(eager.defined(), "Native Gemma weighted epilogue unavailable for graph eager");
        auto eager_bits = eager.clone();
        auto old_ids = at::argsort(at::softmax(logits, -1), true, -1, true).narrow(-1, 0, 8);
        auto changed_x = at::randn_like(x) * .25;
        auto changed_logits = at::roll(logits, {1}, {1});
        auto new_ids =
            at::argsort(at::softmax(changed_logits, -1), true, -1, true).narrow(-1, 0, 8);
        TORCH_CHECK(!at::equal(old_ids, new_ids), "Graph fixture did not change top-8 routing");
        const auto x_ptr = x.data_ptr();
        const auto logits_ptr = logits.data_ptr();
        stream.synchronize();

        at::cuda::CUDAGraph graph;
        graph.capture_begin();
        auto captured = infinity::routed_moe_cuda(x, logits, gu, down, true, scales);
        graph.capture_end();
        graph.replay();
        stream.synchronize();
        TORCH_CHECK(bits_equal(captured, eager_bits),
                    "Gemma graph replay changed eager routed bits");

        auto first = captured.clone();
        x.copy_(changed_x);
        logits.copy_(changed_logits);
        TORCH_CHECK(x.data_ptr() == x_ptr && logits.data_ptr() == logits_ptr,
                    "Graph replay fixture moved input/logit storage");
        auto eager_changed = infinity::routed_moe_cuda(x, logits, gu, down, true, scales);
        TORCH_CHECK(eager_changed.defined(),
                    "Native Gemma weighted epilogue unavailable after graph input change");
        auto eager_changed_bits = eager_changed.clone();
        stream.synchronize();
        graph.replay();
        stream.synchronize();
        TORCH_CHECK(bits_equal(captured, eager_changed_bits),
                    "Gemma graph replay reused stale input/logits");
        TORCH_CHECK(!bits_equal(first, captured),
                    "Gemma graph replay did not change output for different routing");
        std::cout << "Gemma graph replay bitwise parity passed\n";
      }
    }
    TORCH_CHECK(!infinity::routed_moe_cuda(at::zeros({2, 16}), {}, {}, {}, true).defined(),
                "CPU must retain reference path");
    std::cout << "Native routed MoE FP32 epilogue tests passed\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
