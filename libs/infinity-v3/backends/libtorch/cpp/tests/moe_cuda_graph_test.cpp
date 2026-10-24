// SPDX-License-Identifier: Apache-2.0
#include "batching/cuda_graph.h"
#include "models/decoder_models.h"
#include "kernels/moe_kernels.h"
#include <c10/core/InferenceMode.h>
#include <iostream>
using namespace infinity;

int main() {
  try {
    c10::InferenceMode inference;
    if (!native_hopper_moe_available(c10::Device("cuda:0"))) {
      std::cout << "Skipped routed graph fixture: native Hopper MoE on SM90 required\n";
      return 0;
    }
    at::manual_seed(971);
    Options cfg;
    cfg.values = {{"model_type", "qwen3_moe"},
                  {"hidden_size", "2048"},
                  {"num_attention_heads", "32"},
                  {"num_key_value_heads", "4"},
                  {"head_dim", "128"},
                  {"intermediate_size", "768"},
                  {"moe_intermediate_size", "768"},
                  {"num_hidden_layers", "1"},
                  {"vocab_size", "64"},
                  {"max_position_embeddings", "64"},
                  {"num_experts", "128"},
                  {"num_experts_per_tok", "8"},
                  {"norm_topk_prob", "true"}};
    Weights w;
    auto opts = at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16);
    auto add =
        [&](const std::string& name, std::initializer_list<int64_t> shape, bool norm = false) {
          w[name] = norm ? at::ones(at::IntArrayRef(shape), opts)
                         : at::randn(at::IntArrayRef(shape), opts) * .02;
        };
    add("model.embed_tokens.weight", {64, 2048});
    add("model.norm.weight", {2048}, true);
    const std::string p = "model.layers.0.";
    add(p + "input_layernorm.weight", {2048}, true);
    add(p + "post_attention_layernorm.weight", {2048}, true);
    add(p + "self_attn.q_proj.weight", {4096, 2048});
    add(p + "self_attn.k_proj.weight", {512, 2048});
    add(p + "self_attn.v_proj.weight", {512, 2048});
    add(p + "self_attn.o_proj.weight", {2048, 4096});
    add(p + "self_attn.q_norm.weight", {128}, true);
    add(p + "self_attn.k_norm.weight", {128}, true);
    add(p + "mlp.gate.weight", {128, 2048});
    add(p + "mlp.experts.gate_up_proj", {128, 1536, 2048});
    add(p + "mlp.experts.down_proj", {128, 2048, 768});
    auto decoder = create_decoder(cfg, w, c10::Device("cuda:0"), at::kBFloat16);
    decoder->ready();
    CudaGraphCache cache(2, 8, 1);
    auto longs = opts.dtype(at::kLong);
    for (auto offsets : std::vector<std::vector<int32_t>>{{0, 3}, {0, 1, 3}}) {
      auto ids = at::tensor({3, 4, 5}, longs), pos = at::tensor({0, 1, 2}, longs);
      auto cumulative = at::from_blob(offsets.data(),
                                      {int64_t(offsets.size())},
                                      at::TensorOptions().dtype(at::kInt))
                            .to(at::kCUDA);
      PackedInput input{ids,
                        at::zeros_like(ids),
                        pos,
                        cumulative,
                        offsets.data(),
                        int64_t(offsets.size() - 1),
                        3};
      for (int run = 0; run < 3; ++run) {
        input.ids.add_(7);
        input.positions.add_(1);
        auto actual = cache.forward(*decoder, input).clone(), expected = decoder->forward(input);
        TORCH_CHECK(at::equal(actual, expected),
                    "Routed decoder capture/replay changed eager outputs or used stale inputs");
      }
      TORCH_CHECK(cache.size() > 0, "Supported routed decoder did not capture");
    }
    TORCH_CHECK(cache.size() == 2, "Ragged routed decoder graph cache shape count mismatch");
    std::cout << "Native exact routed decoder graphs capture/replay bitwise parity passed; graphs="
              << cache.size() << "\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
