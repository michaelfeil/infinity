// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
#pragma once
#include "batching/cuda_graph.h"
#include "batching/packed_input.h"
#include "helpers/options.h"
#include "models/gemma_models.h"
#include <ATen/ATen.h>
#include <algorithm>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>
#ifdef INFINITY_TORCH_CUDA_KERNELS
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif

namespace infinity_gemma_moe_model_test_detail {
struct Fixture {
  infinity::Options cfg;
  infinity::Weights weights;
};

inline Fixture make_fixture(const c10::Device& device, const char* format) {
  constexpr int64_t hidden = 2816, experts = 128, routed = 704, dense = 2112, vocab = 64,
                    heads = 16, layers = 2;
  Fixture f;
  f.cfg.values = {{"model_type", "gemma4"},
                  {"architectures", "[\"Gemma4ForSequenceClassification\"]"},
                  {"architectures.0", "Gemma4ForSequenceClassification"},
                  {"auto_map.AutoModelForSequenceClassification",
                   "modeling_gemma4_sequence.Gemma4ForSequenceClassification"},
                  {"hidden_size", std::to_string(hidden)},
                  {"num_attention_heads", std::to_string(heads)},
                  {"num_key_value_heads", "8"},
                  {"num_global_key_value_heads", "2"},
                  {"head_dim", "256"},
                  {"global_head_dim", "512"},
                  {"num_hidden_layers", std::to_string(layers)},
                  {"max_position_embeddings", "64"},
                  {"sliding_window", "4"},
                  {"_sliding_window_pattern", "2"},
                  {"num_kv_shared_layers", "0"},
                  {"hidden_activation", "gelu_pytorch_tanh"},
                  {"hidden_size_per_layer_input", "0"},
                  {"intermediate_size", std::to_string(dense)},
                  {"moe_intermediate_size", std::to_string(routed)},
                  {"num_experts", std::to_string(experts)},
                  {"top_k_experts", "8"},
                  {"enable_moe_block", "true"},
                  {"attention_k_eq_v", "true"},
                  {"vocab_size", std::to_string(vocab)},
                  {"pad_token_id", "0"},
                  {"rms_norm_eps", "1e-6"},
                  {"layer_types.0", "sliding_attention"},
                  {"layer_types.1", "full_attention"},
                  {"per_layer_config.1.head_dim", "512"},
                  {"per_layer_config.1.num_key_value_heads", "2"},
                  {"rope_parameters.sliding_attention.rope_theta", "10000"},
                  {"rope_parameters.full_attention.rope_theta", "1000000"},
                  {"rope_parameters.full_attention.partial_rotary_factor", "0.25"},
                  {"_decision.kind", "option_tokens"},
                  {"_decision_attention_mode", "block_bidirectional"},
                  {"_dynamic_quantization", format},
                  {"_dynamic_quantization_scope", "mlp"}};
  const auto bf16 = at::TensorOptions().device(device).dtype(at::kBFloat16);
  auto add = [&](const std::string& name, std::initializer_list<int64_t> shape, float scale) {
    f.weights[name] = at::randn(at::IntArrayRef(shape), bf16).mul_(scale);
  };
  auto add_scaled = [&](const std::string& name, std::initializer_list<int64_t> shape) {
    f.weights[name] = at::ones(at::IntArrayRef(shape), bf16)
                          .add_(at::randn(at::IntArrayRef(shape), bf16).mul_(.1f));
  };
  add("embed_tokens.weight", {vocab, hidden}, .02f);
  add_scaled("norm.weight", {hidden});
  for (int64_t layer = 0; layer < layers; ++layer) {
    const bool full = layer + 1 == layers;
    const int64_t dim = full ? 512 : 256, kv = full ? 2 : 8;
    const auto p = "layers." + std::to_string(layer) + ".";
    for (const auto* name : {"input_layernorm",
                             "post_attention_layernorm",
                             "pre_feedforward_layernorm",
                             "post_feedforward_layernorm",
                             "post_feedforward_layernorm_1",
                             "pre_feedforward_layernorm_2",
                             "post_feedforward_layernorm_2"})
      add_scaled(p + name + ".weight", {hidden});
    add(p + "self_attn.q_proj.weight", {heads * dim, hidden}, .02f);
    add_scaled(p + "self_attn.q_norm.weight", {dim});
    add(p + "self_attn.k_proj.weight", {kv * dim, hidden}, .02f);
    add_scaled(p + "self_attn.k_norm.weight", {dim});
    if (!full)
      add(p + "self_attn.v_proj.weight", {kv * dim, hidden}, .02f);
    add(p + "self_attn.o_proj.weight", {hidden, heads * dim}, .02f);
    add(p + "mlp.gate_proj.weight", {dense, hidden}, .02f);
    add(p + "mlp.up_proj.weight", {dense, hidden}, .02f);
    add(p + "mlp.down_proj.weight", {hidden, dense}, .02f);
    add_scaled(p + "router.scale", {hidden});
    add(p + "router.proj.weight", {experts, hidden}, .02f);
    add_scaled(p + "router.per_expert_scale", {experts});
    add(p + "experts.gate_up_proj", {experts, 2 * routed, hidden}, .02f);
    add(p + "experts.down_proj", {experts, hidden, routed}, .02f);
    add_scaled(p + "layer_scalar", {1});
  }
  return f;
}

inline infinity::PackedInput make_input(const c10::Device& device,
                                        const std::vector<int64_t>& ids,
                                        const std::vector<int32_t>& offsets) {
  std::vector<int64_t> positions;
  int64_t maximum = 0;
  for (size_t row = 1; row < offsets.size(); ++row) {
    const auto length = offsets[row] - offsets[row - 1];
    maximum = std::max<int64_t>(maximum, length);
    for (int64_t position = 0; position < length; ++position)
      positions.push_back(position);
  }
  const auto longs = at::TensorOptions().dtype(at::kLong);
  infinity::PackedInput input{
      at::from_blob(const_cast<int64_t*>(ids.data()), {static_cast<int64_t>(ids.size())}, longs)
          .clone()
          .to(device),
      at::zeros({static_cast<int64_t>(ids.size())}, longs).to(device),
      at::from_blob(positions.data(), {static_cast<int64_t>(positions.size())}, longs)
          .clone()
          .to(device),
      at::from_blob(const_cast<int32_t*>(offsets.data()),
                    {static_cast<int64_t>(offsets.size())},
                    at::TensorOptions().dtype(at::kInt))
          .clone()
          .to(device),
      offsets.data(),
      static_cast<int64_t>(offsets.size() - 1),
      maximum};
  input.block_prefix_lengths = {3, 3};
  input.block_prefix_sources = {0, 0};
  return input;
}
} // namespace infinity_gemma_moe_model_test_detail

inline void gemma_quantized_moe_model_graph_test(const c10::Device& device, const char* format) {
#ifdef INFINITY_TORCH_CUDA_KERNELS
  using namespace infinity_gemma_moe_model_test_detail;
  TORCH_CHECK(device.is_cuda(), "Gemma quantized MoE model graph test requires CUDA");
  TORCH_CHECK(format && (std::string_view(format) == "fp8" || std::string_view(format) == "mxfp8"),
              "Gemma quantized MoE model graph test requires fp8 or mxfp8");
  at::manual_seed(20261109);
  auto fixture = make_fixture(device, format);
  auto model = infinity::create_gemma(fixture.cfg, fixture.weights, device, at::kBFloat16);
  TORCH_CHECK(model, "Gemma quantized MoE fixture was rejected");
  model->ready();
  for (int64_t layer = 0; layer < 2; ++layer) {
    const auto p = "layers." + std::to_string(layer) + ".experts.";
    TORCH_CHECK(!fixture.weights.count(p + "gate_up_proj") &&
                    !fixture.weights.count(p + "down_proj"),
                "Gemma native ready retained BF16 expert weights at layer ",
                layer);
  }

  const std::vector<int64_t> ids{
      7, 11, 13, 21, 22, 23, 24, 25, 26, 7, 11, 13, 31, 32, 33, 34, 35, 36, 37};
  const std::vector<int32_t> offsets{0, 9, 19};
  auto packed = make_input(device, ids, offsets);
  model->prepare_input(packed);
  const auto first = model->forward(packed);
  const auto second = model->forward(packed);
  TORCH_CHECK(at::isfinite(first).all().item<bool>() && at::isfinite(second).all().item<bool>() &&
                  at::equal(first, second),
              "Gemma quantized MoE eager repeat was non-finite or non-deterministic");
  const auto fresh = model->forward(packed);
  const auto fresh_pooled = model->forward_pooled(packed, true);
  const auto ends = at::tensor(std::vector<int64_t>{offsets[1] - 1, offsets[2] - 1},
                               at::TensorOptions().dtype(at::kLong).device(device));

  c10::cuda::getCurrentCUDAStream(device.index()).synchronize();
  const auto graph_stream = c10::cuda::getStreamFromPool(false, device.index());
  infinity::CudaGraphCache graphs;
  at::Tensor captured;
  {
    c10::cuda::CUDAStreamGuard guard(graph_stream);
    for (int run = 0; run < 3; ++run)
      captured = graphs.forward(*model, packed, 0).clone();
    graph_stream.synchronize();
  }
  auto stats = graphs.stats();
  TORCH_CHECK(graphs.size() == 1 && stats.captures == 1,
              "Gemma quantized MoE CUDA graph did not capture exactly once; captures=",
              stats.captures);
  TORCH_CHECK(stats.live_bytes > 0 && stats.live_bytes <= (512ULL << 20),
              "Gemma quantized MoE graph footprint exceeded the production 512 MiB budget: ",
              stats.live_bytes);
  TORCH_CHECK(at::equal(captured, fresh),
              "Gemma quantized MoE captured graph changed eager full outputs");
  TORCH_CHECK(at::allclose(captured.index_select(0, ends), fresh_pooled, .01, .02),
              "Gemma quantized MoE graph full rows differ from fresh eager pooled outputs");

  const auto* ids_pointer = packed.ids.const_data_ptr();
  packed.ids.narrow(0, 4, 2).copy_(at::tensor(std::vector<int64_t>{42, 43}, packed.ids.options()));
  TORCH_CHECK(packed.ids.const_data_ptr() == ids_pointer,
              "Gemma quantized MoE suffix mutation replaced ids storage");
  const auto mutated = model->forward(packed);
  const auto mutated_pooled = model->forward_pooled(packed, true);
  c10::cuda::getCurrentCUDAStream(device.index()).synchronize();
  at::Tensor replayed;
  {
    c10::cuda::CUDAStreamGuard guard(graph_stream);
    replayed = graphs.forward(*model, packed, 0).clone();
    graph_stream.synchronize();
  }
  stats = graphs.stats();
  TORCH_CHECK(stats.captures == 1,
              "Gemma quantized MoE replay silently fell back to eager; captures=",
              stats.captures);
  TORCH_CHECK(at::equal(replayed, mutated),
              "Gemma quantized MoE graph replay ignored in-place suffix ids");
  TORCH_CHECK(at::allclose(replayed.index_select(0, ends), mutated_pooled, .01, .02),
              "Gemma quantized MoE graph pooled rows differ from fresh eager outputs");
  TORCH_CHECK(!at::equal(replayed.narrow(0, 3, 6), captured.narrow(0, 3, 6)),
              "Gemma quantized MoE graph did not observe its own changed suffix");
  TORCH_CHECK(at::equal(replayed.narrow(0, 0, 3), captured.narrow(0, 0, 3)),
              "Gemma quantized MoE suffix mutation changed the shared prefix");
  TORCH_CHECK(at::equal(replayed.narrow(0, 9, 10), captured.narrow(0, 9, 10)),
              "Gemma quantized MoE suffix mutation leaked into the other sequence");
  const auto replayed_pooled = replayed.index_select(0, ends);
  const auto captured_pooled = captured.index_select(0, ends);
  TORCH_CHECK(!at::equal(replayed_pooled.narrow(0, 0, 1), captured_pooled.narrow(0, 0, 1)),
              "Gemma quantized MoE pooled output ignored its changed suffix");
  TORCH_CHECK(at::equal(replayed_pooled.narrow(0, 1, 1), captured_pooled.narrow(0, 1, 1)),
              "Gemma quantized MoE pooled output crossed sequence boundaries");
  std::cout << "Gemma " << format
            << " quantized MoE model graph capture/replay passed; captures=" << stats.captures
            << " graph_live_bytes=" << stats.live_bytes << "\n";
#else
  (void)device;
  (void)format;
  TORCH_CHECK(false, "Gemma quantized MoE model graph test requires CUDA kernels");
#endif
}
