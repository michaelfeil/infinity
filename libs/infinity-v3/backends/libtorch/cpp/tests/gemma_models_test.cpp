#include "prefix_cache_test.h"
#include "helpers/fp8_linear.h"
// Packed sequence isolation and geometry tests for native Gemma implementations.
#include "models/gemma_models.h"
#include "batching/radix.h"
#include "batching/cuda_graph.h"
#include "helpers/image_feature_cache.h"
#include <iostream>
#include <cstring>
#include <utility>
#ifdef INFINITY_TORCH_CUDA_KERNELS
#include "kernels/gemma_score.h"
#include "kernels/qwen35_kernels.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#endif

using namespace infinity;

namespace {
void batched_vision_test(Model& model, const PackedInput& input) {
  auto batch = input;
  auto original = input.images.front();
  batch.images = {original, original, original};
  batch.images[1].pixels = original.pixels.repeat({2, 1});
  batch.images[1].grid_thw = {1, 3, 6};
  batch.images[1].token_count = 2;
  batch.images[1].has_feature_key = false;
  batch.images[2].pixels = 1. - original.pixels;
  batch.images[2].has_feature_key = true;
  batch.images[2].feature_key[0] = 91;
  batch.images.push_back(batch.images[2]);
  model.prepare_input(batch);
  TORCH_CHECK(batch.images[2].prepared_features.const_data_ptr() ==
                  batch.images[3].prepared_features.const_data_ptr(),
              "Duplicate image keys were encoded twice in one batch");
  for (size_t i = 0; i < batch.images.size(); ++i) {
    auto single = input;
    single.images = {batch.images[i]};
    single.images[0].has_feature_key = false;
    model.prepare_input(single);
    TORCH_CHECK(
        at::allclose(
            batch.images[i].prepared_features, single.images[0].prepared_features, .015, .015),
        "Batched Gemma vision leaked between differently sized images or reused stale cache features");
  }
  batch.images[1].token_start = 3;
  batch.images[2].token_start = 5;
  batch.images[2].sequence_start = 5;
  batch.images[3].token_start = 6;
  batch.images[3].sequence_start = 5;
  auto expected = model.forward(batch);
  auto partial = batch;
  partial.images[1].prepared_features = at::Tensor();
  partial.images[3].prepared_features = at::Tensor();
  TORCH_CHECK(at::allclose(model.forward(partial), expected, .015, .015),
              "Direct Gemma image batching changed prepared/unprepared feature placement");
  auto again = batch;
  model.prepare_input(again);
  TORCH_CHECK(again.images[2].prepared_features.const_data_ptr() ==
                  batch.images[2].prepared_features.const_data_ptr(),
              "Batched Gemma feature cache hit recomputed features");
}

void feature_cache_bounds() {
  ImageFeatureCache cache(64, 2);
  ImageFeatureCache::Key a{}, b{}, c{};
  b[0] = 1;
  c[0] = 2;
  int calls = 0;
  auto large = at::arange(100, at::kFloat);
  auto first = cache.get_or_compute(a, [&] {
    ++calls;
    return large.narrow(0, 1, 4);
  });
  TORCH_CHECK(first.storage().nbytes() == 16 && first.storage_offset() == 0 &&
                  cache.usage() == std::make_pair(size_t(1), size_t(16)),
              "Cache budget retained oversized view storage");
  auto hit = cache.get_or_compute(a, [&] {
    ++calls;
    return at::zeros({4});
  });
  TORCH_CHECK(calls == 1 && at::equal(first, hit), "Image cache hit recomputed features");
  cache.get_or_compute(b, [&] {
    ++calls;
    return at::ones({4});
  });
  cache.get_or_compute(a, [&] {
    ++calls;
    return at::zeros({4});
  }); // a is most recent.
  cache.get_or_compute(c, [&] {
    ++calls;
    return at::ones({4});
  });
  cache.get_or_compute(a, [&] {
    ++calls;
    return at::zeros({4});
  });
  TORCH_CHECK(calls == 3, "LRU promotion evicted a recent entry");
  cache.get_or_compute(b, [&] {
    ++calls;
    return at::ones({4});
  });
  TORCH_CHECK(calls == 4 && cache.usage().first == 2 && cache.usage().second <= 64,
              "Bounded LRU did not evict the oldest entry");
  auto before = cache.usage();
  for (int i = 0; i < 2; ++i)
    cache.get_or_compute(c, [&] {
      ++calls;
      return at::zeros({100});
    });
  TORCH_CHECK(calls == 6 && cache.usage() == before, "Oversized features entered the cache");
  ImageFeatureCache entry_bound;
  for (int i = 0; i < 33; ++i) {
    ImageFeatureCache::Key key{};
    key[0] = i;
    entry_bound.get_or_compute(key, [] { return at::ones({1}, at::kByte); });
  }
  TORCH_CHECK(entry_bound.usage() == std::make_pair(size_t(32), size_t(32)),
              "Default image cache32entry bound failed");
}
#ifdef INFINITY_TORCH_CUDA_KERNELS
void publisher_bf16_score_readout(const c10::Device& device) {
  c10::cuda::CUDAGuard guard(device);
  auto exact = [](const at::Tensor& a, const at::Tensor& b) {
    auto x = a.cpu().contiguous(), y = b.cpu().contiguous();
    return x.sizes() == y.sizes() &&
           std::memcmp(x.data_ptr(), y.data_ptr(), x.numel() * sizeof(c10::BFloat16)) == 0;
  };
  // Cover every BF16 value, including signed zeros, infinities and NaN payloads.
  std::vector<uint16_t> bits(65536);
  for (int i = 0; i < 65536; ++i)
    bits[i] = i;
  auto values = at::from_blob(bits.data(), {65536, 1}, at::kBFloat16).clone().to(device);
  auto ends = at::arange(65537, at::TensorOptions().device(device).dtype(at::kInt));
  TORCH_CHECK(exact(values, gemma4_bf16_sequence_score_readout(values, ends, 65536, 0., false)),
              "Uncapped publisher gather changed BF16 bits");
  for (double cap : {.01, 1., 30., 50., -30.}) {
    auto reference = values / cap;
    reference = at::tanh(reference);
    reference = reference * cap;
    TORCH_CHECK(
        exact(reference, gemma4_bf16_sequence_score_readout(values, ends, 65536, cap, true)),
        "Publisher readout changed Torch BF16 scalar arithmetic");
  }
  // All BF16 operands, original learned layer scalars, zeros, infinities and
  // NaNs. Compare bytewise to the original two ATen operations, including both
  // intermediate rounding and non-destructive output ownership.
  std::vector<uint16_t> scalar_bits = {
      0,     15760, 15918, 15944, 15952, 16132, 16142, 16143, 16155, 16158, 16163,
      16167, 16173, 16177, 16180, 16182, 16184, 16186, 16187, 16188, 16189, 16192,
      16193, 16195, 16197, 16206, 16209, 16256, 32640, 32705, 32768, 49024, 65408};
  for (int shift : {0, 1, 32768, 12345}) {
    auto other = at::roll(values, {shift}, {0}).contiguous();
    auto before = values.clone(), other_before = other.clone();
    for (auto raw : scalar_bits) {
      auto scale = at::from_blob(&raw, {1}, at::kBFloat16).clone().to(device);
      auto reference = (values + other) * scale;
      auto observed = gemma4_bf16_residual_scale(values, other, scale);
      TORCH_CHECK(exact(reference, observed), "Gemma4 fused add/multiply changed BF16 bits");
      TORCH_CHECK(observed.const_data_ptr() != values.const_data_ptr() &&
                      observed.const_data_ptr() != other.const_data_ptr() &&
                      exact(values, before) && exact(other, other_before),
                  "Gemma4 residual scale overwrote retained input storage");
    }
  }
  // Compare the fused norm/add/scalar path to the actual deployed original
  // norm kernel plus ATen operations. Include true hidden width2816, original
  // learned scalar bits, finite data and every BF16 bit pattern across rows.
  for (int64_t width : {8, 256, 512, 2816, 8192}) {
    auto norm_scale = (at::randn({width}, values.options()) * .1 + 1.).to(at::kFloat);
    for (int pattern = 0; pattern < 3; ++pattern) {
      auto input = at::randn({32, width}, values.options());
      auto residual = at::randn_like(input);
      if (pattern == 1) {
        std::vector<uint16_t> words(input.numel());
        for (size_t i = 0; i < words.size(); ++i)
          words[i] = i % 65536;
        input = at::from_blob(words.data(), input.sizes(), at::kBFloat16).clone().to(device);
      }
      if (pattern == 2) {
        input.zero_();
        residual.zero_();
      }
      auto normalized = gemma_norm_cuda(input, norm_scale, 1e-6, true);
      auto added = residual + normalized;
      TORCH_CHECK(exact(added, gemma4_bf16_norm_residual(residual, input, norm_scale, 1e-6, {})),
                  "Gemma4 fused norm/add changed original reference tree or BF16 stages");
      for (auto raw : scalar_bits) {
        auto scale = at::from_blob(&raw, {1}, at::kBFloat16).clone().to(device);
        TORCH_CHECK(exact(added * scale,
                          gemma4_bf16_norm_residual(residual, input, norm_scale, 1e-6, scale)),
                    "Gemma4 fused norm/add/scalar changed original stages");
      }
    }
  }
  // The MoE's three pre-FF branches share RMS statistics, but retain their
  // distinct BF16 normalization/root/learned-scale stages and FP32 router.
  for (int64_t rows : {3, 129}) {
    auto x = at::randn({rows, 2816}, values.options());
    auto dense_scale = at::randn({2816}, x.options().dtype(at::kFloat)) * .1 + 1.;
    auto routed_scale = at::randn_like(dense_scale) * .1 + 1.;
    auto router_scale = at::randn({2816}, x.options());
    auto root = at::scalar_tensor(.3712345, x.options());
    auto ones = at::ones_like(dense_scale);
    auto same = [](const at::Tensor& a, const at::Tensor& b) {
      return a.sizes() == b.sizes() && a.scalar_type() == b.scalar_type() &&
             at::equal(a.contiguous().view(at::kByte), b.contiguous().view(at::kByte));
    };
    auto check = [&](const Gemma4PreFeedforwardNorm& actual) {
      TORCH_CHECK(
          same(actual.dense, gemma_norm_cuda(x, dense_scale, 1e-6, true)) &&
              same(actual.routed, gemma_norm_cuda(x, routed_scale, 1e-6, true)) &&
              same(actual.router,
                   ((gemma_norm_cuda(x, ones, 1e-6, true) * root) * router_scale).to(at::kFloat)),
          "Shared pre-FF statistics changed normalization or router stages");
    };
    auto norm_stream = at::cuda::getStreamFromPool(false, device.index());
    at::cuda::getCurrentCUDAStream(device.index()).synchronize();
    c10::cuda::CUDAStreamGuard norm_guard(norm_stream);
    auto execute = [&] {
      return gemma4_bf16_pre_feedforward_norm(
          x, dense_scale, routed_scale, router_scale, root, 1e-6);
    };
    check(execute());
    execute();
    execute();
    norm_stream.synchronize();
    at::cuda::CUDAGraph graph;
    graph.capture_begin();
    auto captured = execute();
    graph.capture_end();
    graph.replay();
    check(captured);
    x.add_(.25);
    root.fill_(.125);
    router_scale.mul_(.5);
    graph.replay();
    check(captured);
    TORCH_CHECK(!gemma4_bf16_pre_feedforward_norm(
                     x, dense_scale.to(at::kBFloat16), routed_scale, router_scale, root, 1e-6)
                     .dense.defined(),
                "Unsupported shared pre-FF normalization dtype did not decline");
  }
  auto stream = at::cuda::getStreamFromPool(false, device.index());
  c10::cuda::CUDAStreamGuard stream_guard(stream);
  auto options = at::TensorOptions().device(device).dtype(at::kFloat);
  auto scores = (at::arange(42, options).view({21, 2}) - 23.).to(at::kBFloat16);
  auto original = scores.clone();
  auto first_ends = at::tensor(std::vector<int32_t>{0, 7, 21}, at::kInt).to(device);
  auto changed_ends = at::tensor(std::vector<int32_t>{0, 9, 21}, at::kInt).to(device);
  auto current_ends = first_ends.clone();
  auto reference = [&]() {
    auto all = scores / 30.;
    all = at::tanh(all);
    all = all * 30.;
    return all.index_select(0, current_ends.narrow(0, 1, 2).to(at::kLong) - 1);
  };
  for (int i = 0; i < 20; ++i)
    gemma4_bf16_sequence_score_readout(scores, current_ends, 2, 30., true);
  stream.synchronize();
  at::cuda::CUDAGraph graph;
  graph.capture_begin();
  auto observed = gemma4_bf16_sequence_score_readout(scores, current_ends, 2, 30., true);
  graph.capture_end();
  graph.replay();
  TORCH_CHECK(exact(observed, reference()), "Publisher graph changed original score bits");
  auto first = observed.clone();
  current_ends.copy_(changed_ends);
  graph.replay();
  TORCH_CHECK(exact(observed, reference()) && !exact(first, observed),
              "Publisher graph used stale packed ends");
  scores.copy_(original + 1.);
  graph.replay();
  TORCH_CHECK(exact(observed, reference()), "Publisher graph used stale full-GEMM scores");
  scores.copy_(original);
  current_ends.copy_(first_ends);
  graph.replay();
  TORCH_CHECK(exact(first, observed), "Publisher graph restore changed score bits");
  auto residual = at::randn({3, 2816}, options).to(at::kBFloat16);
  auto normalized = at::randn({3, 2816}, options).to(at::kBFloat16);
  auto scalar = at::full({1}, .125, options).to(at::kBFloat16);
  auto saved = residual.clone();
  for (int i = 0; i < 20; ++i)
    gemma4_bf16_residual_scale(residual, normalized, scalar);
  stream.synchronize();
  at::cuda::CUDAGraph residual_graph;
  residual_graph.capture_begin();
  auto scaled = gemma4_bf16_residual_scale(residual, normalized, scalar);
  residual_graph.capture_end();
  residual_graph.replay();
  TORCH_CHECK(exact(scaled, (residual + normalized) * scalar),
              "Residual-scale graph changed original stages");
  auto owned = scaled.clone();
  residual.copy_(saved + 1.);
  scalar.fill_(.5);
  residual_graph.replay();
  TORCH_CHECK(exact(scaled, (residual + normalized) * scalar),
              "Residual-scale graph reused stale operands");
  residual.copy_(saved);
  scalar.fill_(.125);
  residual_graph.replay();
  TORCH_CHECK(exact(scaled, owned) && exact(residual, saved),
              "Residual-scale graph restore changed retained states");
  auto norm_scale = at::ones({2816}, options);
  for (int i = 0; i < 20; ++i)
    gemma4_bf16_norm_residual(residual, normalized, norm_scale, 1e-6, scalar);
  stream.synchronize();
  at::cuda::CUDAGraph norm_graph;
  norm_graph.capture_begin();
  auto norm_scaled = gemma4_bf16_norm_residual(residual, normalized, norm_scale, 1e-6, scalar);
  norm_graph.capture_end();
  norm_graph.replay();
  auto norm_reference = [&] {
    return (residual + gemma_norm_cuda(normalized, norm_scale, 1e-6, true)) * scalar;
  };
  TORCH_CHECK(exact(norm_scaled, norm_reference()),
              "Norm/residual/scalar graph changed original output");
  auto norm_owned = norm_scaled.clone();
  residual.copy_(saved + 1.);
  scalar.fill_(.5);
  norm_graph.replay();
  TORCH_CHECK(exact(norm_scaled, norm_reference()),
              "Norm/residual/scalar graph reused stale operands");
  residual.copy_(saved);
  scalar.fill_(.125);
  norm_graph.replay();
  TORCH_CHECK(exact(norm_owned, norm_scaled) && exact(residual, saved),
              "Norm/residual/scalar graph changed retained states");
  std::cout
      << "Gemma4 publisher BF16 scalar arithmetic and current graph score/end readout passed\n";
}

void feature_cache_stream_lifetime(const c10::Device& device) {
  c10::cuda::CUDAGuard device_guard(device);
  auto producer = at::cuda::getStreamFromPool(false, device.index());
  auto consumer = at::cuda::getStreamFromPool(false, device.index());
  constexpr int64_t count = 65536;
  ImageFeatureCache cache(count * sizeof(float), 1);
  ImageFeatureCache::Key first{}, replacement{};
  replacement[0] = 1;
  auto options = at::TensorOptions().device(device).dtype(at::kFloat);
  {
    c10::cuda::CUDAStreamGuard stream_guard(producer);
    cache.get_or_compute(first, [&] { return at::arange(count, options) * 3; });
  }
  at::Tensor observed, work;
  {
    c10::cuda::CUDAStreamGuard stream_guard(consumer);
    // Queue unrelated consumer work before reading a hit produced elsewhere.
    auto matrix = at::ones({1024, 1024}, options);
    work = at::mm(matrix, matrix);
    auto hit = cache.get_or_compute(first, [&]() -> at::Tensor {
      TORCH_CHECK(false, "Cross-stream hit unexpectedly recomputed features");
    });
    observed = hit + 7;
  }
  {
    c10::cuda::CUDAStreamGuard stream_guard(producer);
    // Drop the cache's last reference while consumer reads may still be queued,
    // then pressure this stream's allocator with identically sized allocations.
    cache.get_or_compute(replacement, [&] { return at::full({count}, -99, options); });
    for (int i = 0; i < 16; ++i)
      at::full({count}, 99 + i, options);
  }
  consumer.synchronize();
  producer.synchronize();
  TORCH_CHECK(at::equal(observed, at::arange(count, options) * 3 + 7),
              "Cross-stream cache hit changed after eviction");
  TORCH_CHECK(cache.usage() == std::make_pair(size_t(1), size_t(count * sizeof(float))),
              "CUDA eviction exceeded cache-owned storage budget");
  TORCH_CHECK(work[0][0].item<float>() == 1024, "Consumer work was corrupted during eviction");
}
#endif
struct Fixture {
  Options cfg;
  Weights weights;
  bool legacy, embedding2;

  Fixture(const std::string& family,
          int64_t head_dim = 8,
          int64_t layer_count = 4,
          int64_t shared_count = 2)
      : legacy(family == "gemma3" || family == "gemma3_text"),
        embedding2(family == "embedding_gemma2") {
    cfg.values = {{"model_type", family},
                  {"hidden_size", "16"},
                  {"num_attention_heads", "2"},
                  {"num_key_value_heads", "1"},
                  {"head_dim", std::to_string(head_dim)},
                  {"global_head_dim", std::to_string(head_dim)},
                  {"num_hidden_layers", std::to_string(layer_count)},
                  {"max_position_embeddings", "512"},
                  {"sliding_window", "2"},
                  {"query_pre_attn_scalar", "8"},
                  {"_sliding_window_pattern", "2"},
                  {"num_kv_shared_layers", legacy ? "0" : std::to_string(shared_count)},
                  {"hidden_activation", "gelu_pytorch_tanh"},
                  {"hidden_size_per_layer_input", legacy ? "0" : "4"}};
    cfg.values["embedding_dim"] = "12";
    auto put =
        [&](const std::string& name, std::initializer_list<int64_t> shape, bool norm = false) {
          weights.emplace(
              name, norm ? (legacy ? at::zeros(shape) : at::ones(shape)) : at::randn(shape) * .1);
        };
    put("embed_tokens.weight", {64, 16});
    put("norm.weight", {16}, true);
    if (!legacy) {
      auto stem = embedding2 ? std::string("ple.") : std::string();
      put(stem + "per_layer_model_projection.weight", {4 * layer_count, 16});
      put(stem + "per_layer_projection_norm.weight", {4}, true);
      if (!embedding2)
        put("embed_tokens_per_layer.weight", {64, 4 * layer_count});
      if (embedding2)
        put("embedding_projection.weight", {12, 16});
    }
    for (int64_t i = 0; i < layer_count; ++i) {
      cfg.values["layer_types." + std::to_string(i)] =
          (head_dim > 256 || i % 2) ? "full_attention" : "sliding_attention";
      auto p = "layers." + std::to_string(i) + ".";
      for (const auto& name : {"input_layernorm",
                               "post_attention_layernorm",
                               "pre_feedforward_layernorm",
                               "post_feedforward_layernorm"})
        put(p + name + ".weight", {16}, true);
      put(p + "self_attn.q_proj.weight", {2 * head_dim, 16});
      put(p + "self_attn.q_norm.weight", {head_dim}, true);
      if (legacy || i < layer_count - shared_count) {
        put(p + "self_attn.k_proj.weight", {head_dim, 16});
        put(p + "self_attn.v_proj.weight", {head_dim, 16});
        put(p + "self_attn.k_norm.weight", {head_dim}, true);
      }
      put(p + "self_attn.o_proj.weight", {16, 2 * head_dim});
      put(p + "mlp.gate_proj.weight", {24, 16});
      put(p + "mlp.up_proj.weight", {24, 16});
      put(p + "mlp.down_proj.weight", {16, 24});
      if (!legacy) {
        weights[p + "layer_scalar"] = at::ones({1});
        auto stem = p + (embedding2 ? "ple_block." : "");
        put(stem + "per_layer_input_gate.weight", {4, 16});
        put(stem + "per_layer_projection.weight", {16, 4});
        put(stem + "post_per_layer_input_norm.weight", {16}, true);
      }
    }
  }
};

PackedInput input(const std::vector<int64_t>& ids, const std::vector<int32_t>& offsets) {
  auto tokens = at::from_blob(const_cast<int64_t*>(ids.data()),
                              {static_cast<int64_t>(ids.size())},
                              at::TensorOptions().dtype(at::kLong))
                    .clone();
  std::vector<int64_t> positions;
  int64_t maximum = 0;
  for (size_t i = 1; i < offsets.size(); ++i) {
    auto len = offsets[i] - offsets[i - 1];
    maximum = std::max<int64_t>(maximum, len);
    for (int j = 0; j < len; ++j)
      positions.push_back(j);
  }
  auto pos = at::from_blob(positions.data(),
                           {static_cast<int64_t>(positions.size())},
                           at::TensorOptions().dtype(at::kLong))
                 .clone();
  return {tokens,
          at::zeros_like(tokens),
          pos,
          at::from_blob(const_cast<int32_t*>(offsets.data()),
                        {static_cast<int64_t>(offsets.size())},
                        at::TensorOptions().dtype(at::kInt))
              .clone(),
          offsets.data(),
          static_cast<int64_t>(offsets.size() - 1),
          maximum};
}
#ifdef INFINITY_TORCH_CUDA_KERNELS
void gemma4_moe_forward_parity(const c10::Device& device) {
  at::manual_seed(418);
  Fixture f("gemma4", 8, 2, 0);
  f.cfg.values["hidden_size_per_layer_input"] = "0";
  f.cfg.values["enable_moe_block"] = "true";
  f.cfg.values["num_experts"] = "3";
  // Select every expert so cross-device BF16 reduction order cannot flip a
  // top-k boundary; probabilities and per-expert scales still consume routing.
  f.cfg.values["top_k_experts"] = "3";
  // Non-proportional learned scales make the three fused pre-FF consumers
  // observable end to end: dense uses pre_feedforward_layernorm, routed uses
  // pre_feedforward_layernorm_2, and the FP32 router uses router.scale.
  const auto index = at::arange(16, at::kFloat);
  for (int64_t layer = 0; layer < 2; ++layer) {
    auto p = "layers." + std::to_string(layer) + ".";
    const auto gain = 1.f + .1f * layer;
    f.weights[p + "pre_feedforward_layernorm.weight"] = (.25f + index * .1f) * gain;
    f.weights[p + "pre_feedforward_layernorm_2.weight"] = (1.7f - index * .09f) * gain;
    f.weights[p + "router.scale"] = (.3f + index.remainder(3) * .6f) * gain;
    f.weights[p + "router.proj.weight"] = at::randn({3, 16}) * .1;
    f.weights[p + "router.per_expert_scale"] = at::tensor({.5f, 1.25f, .8f}) * gain;
    f.weights[p + "experts.gate_up_proj"] = at::randn({3, 12, 16}) * .1;
    f.weights[p + "experts.down_proj"] = at::randn({3, 16, 6}) * .1;
    for (const auto& name : {"post_feedforward_layernorm_1", "post_feedforward_layernorm_2"})
      f.weights[p + name + ".weight"] = at::ones({16});
  }
  Weights cpu_weights;
  for (const auto& [name, value] : f.weights)
    cpu_weights[name] = value.to(at::kBFloat16);
  auto cpu = create_gemma(f.cfg, cpu_weights, c10::Device("cpu"), at::kBFloat16);
  cpu->ready();
  Weights gpu_weights;
  for (const auto& [name, value] : cpu_weights)
    gpu_weights[name] = value.to(device);
  auto gpu = create_gemma(f.cfg, gpu_weights, device, at::kBFloat16);
  gpu->ready();
  // Mixed packed sequence lengths exercise every routed row without padding.
  const std::vector<int64_t> first = {1, 2, 3}, second = {4, 5, 6, 7, 8}, third = {9, 10};
  std::vector<int64_t> ids = first;
  ids.insert(ids.end(), second.begin(), second.end());
  ids.insert(ids.end(), third.begin(), third.end());
  const std::vector<int32_t> offsets = {0, 3, 8, 10};
  auto host = input(ids, offsets), dev = host;
  dev.ids = host.ids.to(device);
  dev.types = host.types.to(device);
  dev.positions = host.positions.to(device);
  dev.cumulative = host.cumulative.to(device);
  auto reference = cpu->forward(host).to(at::kFloat);
  auto actual = gpu->forward(dev).cpu().to(at::kFloat);
  // The CPU reference retains the original three separate norm calls. CUDA
  // only shares their RMS statistic, but BF16 attention/GEMM/expert reduction
  // order still differs across devices. Width 16 keeps the RMS sums well
  // conditioned, so use the same cross-device tolerance as the dense BF16
  // parity checks rather than pretending the reference is bit-exact.
  TORCH_CHECK(at::allclose(reference, actual, .04, .04),
              "Gemma4 BF16 MoE forward diverged from separate-normalization CPU reference: maxabs=",
              (reference - actual).abs().max().item<float>());
  std::cout << "gemma4: BF16 MoE forward matches separate-normalization CPU reference maxabs="
            << (reference - actual).abs().max().item<float>() << "\n";
}

void dynamic_fp8(const c10::Device& device, const std::string& family) {
  if (!fp8_device_supported(device))
    return;
  at::manual_seed(418);
  Fixture f(family, 8, 2, 0);
  f.cfg.values["_decision.kind"] = "option_tokens";
  for (int i = 0; i < 2; ++i) {
    const auto p = "layers." + std::to_string(i) + ".mlp.";
    f.weights[p + "gate_proj.weight"] = at::randn({32, 16}) * .1;
    f.weights[p + "up_proj.weight"] = at::randn({32, 16}) * .1;
    f.weights[p + "down_proj.weight"] = at::randn({16, 32}) * .1;
  }
  Weights reference_weights, quantized_weights;
  for (const auto& [name, value] : f.weights)
    reference_weights[name] = quantized_weights[name] = value.to(device, at::kBFloat16);
  auto reference = create_gemma(f.cfg, reference_weights, device, at::kBFloat16);
  reference->ready();
  f.cfg.values["_dynamic_quantization"] = "fp8";
  auto quantized = create_gemma(f.cfg, quantized_weights, device, at::kBFloat16);
  quantized->ready();
  TORCH_CHECK(!quantized_weights.count("layers.0.mlp.down_proj.weight"),
              "FP8 retained redundant dense weights");
  const std::vector<int32_t> offsets = {0, 5, 7};
  auto packed = input({1, 2, 3, 4, 5, 6, 7}, offsets);
  packed.ids = packed.ids.to(device);
  packed.types = packed.types.to(device);
  packed.positions = packed.positions.to(device);
  packed.cumulative = packed.cumulative.to(device);
  const auto expected = reference->forward(packed).to(at::kFloat);
  const auto actual = quantized->forward(packed).to(at::kFloat);
  // Two quantized projections per layer: bound aggregate error on the random
  // fixture; real-checkpoint task accuracy is validated separately.
  const auto relative_error =
      ((actual - expected).square().mean() / expected.square().mean()).sqrt().item<float>();
  TORCH_CHECK(at::isfinite(actual).all().item<bool>() && relative_error < .1,
              "Gemma dense FP8 numerical regression: maxabs=",
              (actual - expected).abs().max().item<float>(),
              " relative_rmse=",
              relative_error);
  check_prefix_cache(*quantized, packed, 0, .01, .02);
  std::cout << "Gemma dense FP8 numerical and prefix-cache checks passed\n";
}
#endif
void audio_operations(const c10::Device& device) {
  // Natural odd/even frame counts and feature widths: evaluate the scalar
  // convolution at each true output location, with only filter boundaries zero.
  for (int64_t height : {7, 8})
    for (int64_t width : {5, 6}) {
      auto input = (at::arange(2 * 3 * height * width, at::kFloat).remainder(7) - 3)
                       .reshape({2, 3, height, width});
      auto filter =
          (at::arange(4 * 3 * 3 * 3, at::kFloat).remainder(5) - 2).reshape({4, 3, 3, 3}) / 8.;
      const int64_t oh = (height + 1) / 2, ow = (width + 1) / 2;
      auto expected = at::zeros({2, 4, oh, ow}, at::kFloat);
      auto x = input.accessor<float, 4>(), w = filter.accessor<float, 4>(),
           y = expected.accessor<float, 4>();
      for (int64_t b = 0; b < 2; ++b)
        for (int64_t c = 0; c < 4; ++c)
          for (int64_t row = 0; row < oh; ++row)
            for (int64_t col = 0; col < ow; ++col)
              for (int64_t src = 0; src < 3; ++src)
                for (int64_t ky = 0; ky < 3; ++ky)
                  for (int64_t kx = 0; kx < 3; ++kx) {
                    auto iy = 2 * row + ky - 1, ix = 2 * col + kx - 1;
                    if (iy >= 0 && iy < height && ix >= 0 && ix < width)
                      y[b][c][row][col] += x[b][src][iy][ix] * w[c][src][ky][kx];
                  }
      auto actual = gemma_audio_conv2d(input.to(device), filter.to(device), 2);
      TORCH_CHECK(actual.sizes() == expected.sizes() && at::equal(actual.cpu(), expected),
                  "Audio convolution changed true frame geometry or scalar boundary semantics");
    }
  // Both sums land exactly on a BF16 midpoint. Rounding GEMM and bias separately
  // yields 1; a fused bias epilogue would yield 1 + 1/128 instead.
  auto input = at::ones({1, 2}, at::TensorOptions().device(device).dtype(at::kBFloat16));
  auto matrix = at::tensor({1.f, 1.f / 256}, at::kFloat).reshape({1, 2}).to(device, at::kBFloat16);
  auto bias = at::full({1}, 1. / 256, input.options());
  TORCH_CHECK(gemma_audio_linear(input, matrix, bias).item<float>() == 1.f,
              "Audio linear fused bias before BF16 GEMM rounding");
  std::cout << "Audio natural-frame convolution and staged BF16 bias passed on " << device << "\n";
}
}

void embedding_layer_key_aliases() {
  Fixture f("embedding_gemma2", 8, 6, 0);
  // Layer5 really has different Q/K/V geometry; ignoring the override cannot
  // pass merely because it happens to equal the backbone defaults.
  const std::string p = "layers.5.self_attn.";
  for (const auto* name : {"q_proj", "k_proj", "v_proj"})
    f.weights[p + name + ".weight"] = at::randn({32, 16}) * .1;
  f.weights[p + "q_norm.weight"] = at::ones({16});
  f.weights[p + "k_norm.weight"] = at::ones({16});
  f.weights[p + "o_proj.weight"] = at::randn({16, 32}) * .1;
  const std::vector<int64_t> ids = {1, 2, 3, 4, 5, 6, 7};
  const std::vector<int32_t> offsets = {0, 5, 7};
  auto packed = input(ids, offsets);
  auto options = [&](const std::string& spelling, const std::string& scope = "per_layer_config.") {
    auto cfg = f.cfg;
    cfg.values[scope + spelling + ".head_dim"] = "16";
    cfg.values[scope + spelling + ".num_key_value_heads"] = "2";
    return cfg;
  };
  auto run = [&](const Options& cfg) {
    auto model = create_gemma(cfg, f.weights, c10::Device("cpu"), at::kFloat);
    model->ready();
    return model->forward(packed);
  };
  const auto expected = run(options("05"));
  for (const auto* spelling : {"5", "0005", "+05", "+0005"})
    for (const auto* scope : {"per_layer_config.", "text_config.per_layer_config."})
      TORCH_CHECK(at::equal(expected, run(options(spelling, scope))),
                  "Numeric Gemma layer alias changed actual output: ",
                  spelling);
  auto duplicate = options("05");
  duplicate.values["per_layer_config.+0005.head_dim"] = "16";
  duplicate.values["per_layer_config.+0005.num_key_value_heads"] = "2";
  TORCH_CHECK(at::equal(expected, run(duplicate)), "Equivalent Gemma layer aliases rejected");
  duplicate.values["per_layer_config.+0005.head_dim"] = "8";
  bool rejected = false;
  try {
    run(duplicate);
  } catch (const c10::Error& error) {
    rejected = std::string(error.what()).find("Conflicting Gemma per_layer_config aliases") !=
               std::string::npos;
  }
  TORCH_CHECK(rejected, "Conflicting Gemma layer aliases must fail before inference");
  auto ignored = options("05");
  // Rust usize rejects signs other than +, spaces, nondigits and overflow.
  // Valid usize values beyond the real layer count are also unused by Candle.
  for (const auto* spelling :
       {"-0", " 5", "5 ", "+", "5x", "18446744073709551616", "18446744073709551615", "6"})
    ignored.values[std::string("per_layer_config.") + spelling + ".head_dim"] =
        "invalid-unused-value";
  TORCH_CHECK(at::equal(expected, run(ignored)),
              "Invalid or out-of-range Gemma layer key was used");
  auto inherited_root = options("05");
  inherited_root.values["text_config.per_layer_config.0005.head_dim"] = "16";
  inherited_root.values["text_config.per_layer_config.+05.head_dim"] = "16";
  inherited_root.values["text_config.per_layer_config.+05.num_key_value_heads"] = "2";
  TORCH_CHECK(at::equal(expected, run(inherited_root)),
              "Nested alias lost root KV override precedence");
  // Omitted Option<usize> values inherit the backbone; explicit defaults
  // therefore describe the same actual layer geometry, not a conflict.
  f.weights[p + "k_proj.weight"] = at::randn({16, 16}) * .1;
  f.weights[p + "v_proj.weight"] = at::randn({16, 16}) * .1;
  auto explicit_default = options("05");
  explicit_default.values["per_layer_config.05.num_key_value_heads"] = "1";
  const auto fallback_expected = run(explicit_default);
  auto partial = explicit_default;
  partial.values.erase("per_layer_config.05.num_key_value_heads");
  partial.values["per_layer_config.5.head_dim"] = "16";
  partial.values["per_layer_config.5.num_key_value_heads"] = "1";
  TORCH_CHECK(at::equal(fallback_expected, run(partial)),
              "Omitted KV default conflicts with explicit alias");
  auto nested = partial;
  nested.values["text_config.per_layer_config.0005.head_dim"] = "16";
  nested.values["text_config.per_layer_config.+05.head_dim"] = "16";
  nested.values["text_config.per_layer_config.+05.num_key_value_heads"] = "1";
  TORCH_CHECK(at::equal(fallback_expected, run(nested)),
              "Nested omitted KV default conflicts with explicit alias");
  auto omitted_head = options("05");
  omitted_head.values.erase("per_layer_config.05.head_dim");
  omitted_head.values["per_layer_config.05.num_key_value_heads"] = "1";
  omitted_head.values["per_layer_config.5.head_dim"] = "8";
  omitted_head.values["per_layer_config.5.num_key_value_heads"] = "1";
  f.weights[p + "q_proj.weight"] = at::randn({16, 16}) * .1;
  f.weights[p + "k_proj.weight"] = at::randn({8, 16}) * .1;
  f.weights[p + "v_proj.weight"] = at::randn({8, 16}) * .1;
  f.weights[p + "q_norm.weight"] = at::ones({8});
  f.weights[p + "k_norm.weight"] = at::ones({8});
  f.weights[p + "o_proj.weight"] = at::randn({16, 16}) * .1;
  auto original = f.cfg;
  TORCH_CHECK(at::equal(run(original), run(omitted_head)),
              "Omitted head default conflicts with explicit alias");
}

void gemma4_layer_head_dim_geometry() {
  Fixture f("gemma4", 8, 6, 0);
  for (int64_t layer = 0; layer < 6; ++layer)
    f.cfg.values["layer_types." + std::to_string(layer)] =
        layer == 5 ? "full_attention" : "sliding_attention";
  const std::string p = "layers.5.self_attn.";
  // Layer 5 is full attention; give it a per-layer width of 16 while the
  // sliding layers keep the root head_dim of 8.
  f.weights[p + "q_proj.weight"] = at::randn({32, 16}) * .1;
  f.weights[p + "k_proj.weight"] = at::randn({16, 16}) * .1;
  f.weights[p + "v_proj.weight"] = at::randn({16, 16}) * .1;
  f.weights[p + "q_norm.weight"] = at::ones({16});
  f.weights[p + "k_norm.weight"] = at::ones({16});
  f.weights[p + "o_proj.weight"] = at::randn({16, 32}) * .1;
  const std::vector<int64_t> ids = {1, 2, 3, 4, 5};
  const std::vector<int32_t> offsets = {0, 5};
  auto packed = input(ids, offsets);
  auto run = [&](const Options& cfg) {
    auto model = create_gemma(cfg, f.weights, c10::Device("cpu"), at::kFloat);
    model->ready();
    return model->forward(packed);
  };
  // Default full-layer geometry still falls back to global_head_dim.
  auto global = f.cfg;
  global.values["global_head_dim"] = "16";
  const auto expected = run(global);
  // Released Gemma4 text configs omit global_head_dim and store the full-layer
  // width under a padded per_layer_config index.
  auto padded = f.cfg;
  padded.values.erase("global_head_dim");
  padded.values["per_layer_config.05.head_dim"] = "16";
  TORCH_CHECK(at::equal(expected, run(padded)),
              "Gemma4 padded per-layer head_dim changed layer geometry");
  auto canonical = padded;
  canonical.values.erase("per_layer_config.05.head_dim");
  canonical.values["per_layer_config.5.head_dim"] = "16";
  TORCH_CHECK(at::equal(expected, run(canonical)),
              "Gemma4 canonical per-layer head_dim alias diverged");
  auto nested = padded;
  nested.values.erase("per_layer_config.05.head_dim");
  nested.values["text_config.per_layer_config.05.head_dim"] = "16";
  TORCH_CHECK(at::equal(expected, run(nested)), "Gemma4 text_config per-layer head_dim diverged");
}

#ifdef INFINITY_TORCH_CUDA_KERNELS
// Exercise the actual local/global head widths and strided packed QKV layouts.
// Unary/gated activation and rotary must retain their low-precision stages.
void gemma_low_precision_pointwise(c10::Device device) {
  at::manual_seed(418);
  for (auto dtype : {at::kHalf, at::kBFloat16}) {
    auto options = at::TensorOptions().device(device).dtype(dtype);
    for (int64_t width : {256, 512}) {
      auto packed = at::randn({37, 8, width}, options);
      auto x = packed.narrow(1, 1, 4);
      auto scale = at::randn({width}, options).to(at::kFloat);
      auto f = x.to(at::kFloat);
      auto variance = f.square().mean(-1, true) + 1e-6;
      for (bool reference : {false, true}) {
        auto expected =
            ((reference ? f / at::sqrt(variance) : f * at::rsqrt(variance)) * scale).to(dtype);
        // FP32 reduction trees may shift the final BF16 value by one ULP.
        TORCH_CHECK(at::allclose(gemma_norm_cuda(x, scale, 1e-6, reference),
                                 expected,
                                 dtype == at::kBFloat16 ? .008 : .002,
                                 .002),
                    "Gemma packed norm changed its reference arithmetic: ",
                    at::toString(dtype),
                    " width=",
                    width,
                    " reference=",
                    reference);
      }
      auto positions = at::arange(37, options.dtype(at::kLong)).remainder(19);
      auto angles = at::randn({19, width / 2}, options.dtype(at::kFloat));
      auto cos = angles.cos().to(dtype), sin = angles.sin().to(dtype);
      auto c = cos.index_select(0, positions).unsqueeze(1),
           sn = sin.index_select(0, positions).unsqueeze(1);
      auto a = x.narrow(-1, 0, width / 2), b = x.narrow(-1, width / 2, width / 2);
      auto expected = at::cat({a * c - b * sn, b * c + a * sn}, -1);
      TORCH_CHECK(at::equal(gemma_rotary_cuda(x, cos, sin, positions), expected),
                  "Gemma rotary changed rounded products or packed strides");
    }
    if (dtype == at::kHalf) {
      // Cover both launch layouts and a final partial group of four rows.
      for (int64_t rows : {1023, 1024, 1025}) {
        auto x = at::randn({rows, 512}, options);
        auto scale = at::randn({512}, options).to(at::kFloat);
        auto f = x.to(at::kFloat);
        auto expected = (f / at::sqrt(f.square().mean(-1, true) + 1e-6) * scale).to(dtype);
        TORCH_CHECK(at::allclose(gemma_norm_cuda(x, scale, 1e-6, true), expected, .002, .002),
                    "Gemma bulk norm changed normalization or crossed row boundaries");
        auto residual = at::randn_like(x);
        auto before_x = x.clone(), before_residual = residual.clone();
        for (bool reference : {false, true}) {
          auto expected = residual + gemma_norm_cuda(x, scale, 1e-6, reference);
          auto actual = gemma_half_residual_norm_cuda(residual, x, scale, 1e-6, reference);
          TORCH_CHECK(at::equal(actual, expected), "Fused FP16 residual changed rounding stages");
          TORCH_CHECK(actual.const_data_ptr() != x.const_data_ptr() &&
                          actual.const_data_ptr() != residual.const_data_ptr() &&
                          at::equal(x, before_x) && at::equal(residual, before_residual),
                      "Fused FP16 residual overwrote retained input storage");
        }
      }
    }
    for (int64_t columns : {6, 4096}) {
      auto storage = at::randn({37 * columns + 1}, options);
      for (int64_t offset : {0, 1}) {
        auto x = storage.narrow(0, offset, 37 * columns).view({37, columns});
        auto constant = [&](double value) { return at::scalar_tensor(value, options); };
        for (bool gelu : {false, true}) {
          at::Tensor expected;
          if (gelu) {
            auto cube = (x * x) * x;
            auto alpha = x + constant(.044715) * cube;
            auto value = at::tanh(constant(M_2_SQRTPI * M_SQRT1_2) * alpha);
            expected = (constant(.5) * x) * (constant(1.) + value);
          } else
            expected = x / (constant(1.) + at::exp(-x));
          TORCH_CHECK(at::equal(gemma_activation_cuda(x, gelu), expected),
                      "Gemma activation changed low-precision rounding");
          auto width = columns / 2;
          auto gated = expected.narrow(1, 0, width) * x.narrow(1, width, width);
          TORCH_CHECK(at::equal(gemma_gated_cuda(x, gelu), gated),
                      "Gemma gated activation changed its staged reference");
        }
      }
    }
  }
  std::cout << "Gemma FP16/BF16 packed norm, rotary and activation parity passed\n";
}

void gemma4_qkv_norm_rotary_test(c10::Device device) {
  at::manual_seed(917);
  c10::cuda::CUDAGuard device_guard(device);
  const auto options = at::TensorOptions().device(device).dtype(at::kBFloat16);
  const auto float_options = options.dtype(at::kFloat);
  const auto long_options = options.dtype(at::kLong);
  auto exact = [](const at::Tensor& a, const at::Tensor& b) {
    auto x = a.contiguous().cpu(), y = b.contiguous().cpu();
    return x.sizes() == y.sizes() && x.scalar_type() == y.scalar_type() &&
           std::memcmp(x.const_data_ptr(), y.const_data_ptr(), x.numel() * x.element_size()) == 0;
  };
  auto donor_norm = [&](const at::Tensor& x, const at::Tensor& scale) {
    auto source = x.stride(1) == x.size(2) ? x : x.contiguous();
    return gemma_norm_cuda(source, scale, 1e-6, true);
  };
  auto tables =
      [&](int64_t rows, int64_t width, bool partial, at::ScalarType type = at::kBFloat16) {
        auto angles = at::randn({rows, width / 2}, float_options);
        auto cos = angles.cos().to(type), sin = angles.sin().to(type);
        if (partial) {
          const auto keep = width / 4;
          if (keep < width / 2) {
            cos.narrow(1, keep, width / 2 - keep).fill_(1.f);
            sin.narrow(1, keep, width / 2 - keep).zero_();
          }
        }
        return std::make_pair(cos, sin);
      };
  auto make_positions = [&](std::vector<int64_t> values) {
    return at::tensor(std::move(values), long_options);
  };
  for (int64_t width : {32, 64, 256, 384, 512, 1024}) {
    const int64_t tokens = 7;
    const int64_t qheads = 6;
    const int64_t kheads = 2;
    const int64_t vheads = 3;
    auto [cos, sin] = tables(13, width, true);
    auto positions = make_positions({9, 0, 5, 2, 11, 1, 7});
    auto q_scale = at::randn({width}, float_options);
    auto k_scale = at::randn({width}, float_options);
    auto ones = at::ones({width}, float_options);
    auto q_storage = at::randn({tokens, qheads * width + 5}, options);
    auto q = q_storage.narrow(1, 0, qheads * width).view({tokens, qheads, width});
    auto k_storage = at::randn({tokens, kheads * width + 3}, options);
    auto k = k_storage.narrow(1, 0, kheads * width).view({tokens, kheads, width});
    auto v_storage = at::randn({tokens, vheads * width + 3}, options);
    auto v = v_storage.narrow(1, 0, vheads * width).view({tokens, vheads, width});
    auto q_before = q.clone(), k_before = k.clone(), v_before = v.clone();
    auto q_expected = gemma_rotary_cuda(donor_norm(q, q_scale), cos, sin, positions);
    auto k_expected = gemma_rotary_cuda(donor_norm(k, k_scale), cos, sin, positions);
    auto v_expected = donor_norm(v, ones);
    auto k_unit_expected = donor_norm(k, ones);
    auto fused =
        gemma4_qkv_norm_rotary_cuda(q, k, v, q_scale, k_scale, ones, cos, sin, positions, 1e-6);
    TORCH_CHECK(fused[0].defined() && fused[1].defined() && fused[2].defined(),
                "Gemma4 fused Q/K norm-rotary declined a supported layout");
    TORCH_CHECK(exact(fused[0], q_expected),
                "Gemma4 fused Q reference norm or staged rotary changed BF16 bits");
    TORCH_CHECK(exact(fused[1], k_expected),
                "Gemma4 fused K reference norm or staged rotary changed BF16 bits");
    TORCH_CHECK(exact(fused[2], v_expected),
                "Gemma4 fused V unit norm changed reference BF16 bits");
    TORCH_CHECK(exact(q, q_before) && exact(k, k_before) && exact(v, v_before),
                "Gemma4 fused norm/rotary overwrote retained Q/K/V inputs");
    auto q_only = gemma4_q_norm_rotary_cuda(q, q_scale, cos, sin, positions, 1e-6);
    TORCH_CHECK(q_only.defined() && exact(q_only, q_expected),
                "Gemma4 query-only shared-KV norm/rotary changed BF16 bits");
    auto tied =
        gemma4_qkv_norm_rotary_cuda(q, k, k, q_scale, k_scale, ones, cos, sin, positions, 1e-6);
    TORCH_CHECK(tied[0].defined() && tied[1].defined() && tied[2].defined() &&
                    exact(tied[1], k_expected) && exact(tied[2], k_unit_expected),
                "Gemma4 tied-KV fusion transformed V before or during key normalization");
    TORCH_CHECK(exact(k, k_before), "Gemma4 tied-KV fusion overwrote the shared projection");
  }

  {
    const int64_t width = 8192;
    auto [cos, sin] = tables(1, width, false);
    auto positions = make_positions({0});
    auto scale = at::ones({width}, float_options);
    auto q = at::randn({1, 1, width}, options);
    auto q_expected = gemma_rotary_cuda(donor_norm(q, scale), cos, sin, positions);
    auto v_expected = donor_norm(q, scale);
    auto q_only = gemma4_q_norm_rotary_cuda(q, scale, cos, sin, positions, 1e-6);
    auto qkv = gemma4_qkv_norm_rotary_cuda(q, q, q, scale, scale, scale, cos, sin, positions, 1e-6);
    TORCH_CHECK(q_only.defined() && exact(q_only, q_expected) && qkv[0].defined() &&
                    exact(qkv[0], q_expected) && exact(qkv[1], q_expected) &&
                    exact(qkv[2], v_expected),
                "Gemma4 composed norm/rotary fallback changed width 8192 BF16 bits");
  }
  {
    const int64_t width = 8194;
    auto [cos, sin] = tables(1, width, false);
    auto positions = make_positions({0});
    auto scale = at::ones({width}, float_options);
    auto q = at::randn({1, 1, width}, options);
    auto reference_norm = [&](const at::Tensor& x) {
      auto f = x.to(at::kFloat);
      const auto variance = f.square().mean(-1, true) + 1e-6;
      f = f / at::sqrt(variance);
      return (f * scale).to(x.scalar_type());
    };
    auto q_expected = gemma_rotary_cuda(reference_norm(q), cos, sin, positions);
    auto v_expected = reference_norm(q);
    auto q_only = gemma4_q_norm_rotary_cuda(q, scale, cos, sin, positions, 1e-6);
    auto qkv = gemma4_qkv_norm_rotary_cuda(q, q, q, scale, scale, scale, cos, sin, positions, 1e-6);
    TORCH_CHECK(q_only.defined() && exact(q_only, q_expected) && qkv[0].defined() &&
                    exact(qkv[0], q_expected) && exact(qkv[1], q_expected) &&
                    exact(qkv[2], v_expected),
                "Gemma4 composed norm/rotary fallback changed width above 8192 BF16 bits");
  }
  {
    const int64_t width = 64;
    auto [cos, sin] = tables(1, width, false, at::kHalf);
    auto positions = make_positions({0});
    auto scale = at::ones({width}, float_options);
    auto q = at::randn({1, 1, width}, options).to(at::kHalf);
    auto q_expected = gemma_rotary_cuda(donor_norm(q, scale), cos, sin, positions);
    auto v_expected = donor_norm(q, scale);
    auto q_only = gemma4_q_norm_rotary_cuda(q, scale, cos, sin, positions, 1e-6);
    auto qkv = gemma4_qkv_norm_rotary_cuda(q, q, q, scale, scale, scale, cos, sin, positions, 1e-6);
    TORCH_CHECK(q_only.defined() && exact(q_only, q_expected) && qkv[0].defined() &&
                    exact(qkv[0], q_expected) && exact(qkv[1], q_expected) &&
                    exact(qkv[2], v_expected),
                "Gemma4 composed norm/rotary fallback changed FP16 bits");
  }

  const int64_t width = 256, tokens = 5, qheads = 4, kheads = 2, vheads = 2;
  auto [cos, sin] = tables(9, width, true);
  auto positions = make_positions({8, 0, 3, 6, 1});
  auto q_scale = at::randn({width}, float_options);
  auto k_scale = at::randn({width}, float_options);
  auto ones = at::ones({width}, float_options);
  auto q_storage = at::randn({tokens, qheads * width + 1}, options);
  auto q = q_storage.narrow(1, 0, qheads * width).view({tokens, qheads, width});
  auto k_storage = at::randn({tokens, kheads * width + 1}, options);
  auto k = k_storage.narrow(1, 0, kheads * width).view({tokens, kheads, width});
  auto v_storage = at::randn({tokens, vheads * width + 1}, options);
  auto v = v_storage.narrow(1, 0, vheads * width).view({tokens, vheads, width});
  auto check = [&](const std::array<at::Tensor, 3>& actual) {
    TORCH_CHECK(
        exact(actual[0], gemma_rotary_cuda(donor_norm(q, q_scale), cos, sin, positions)) &&
            exact(actual[1], gemma_rotary_cuda(donor_norm(k, k_scale), cos, sin, positions)) &&
            exact(actual[2], donor_norm(v, ones)),
        "Gemma4 fused norm/rotary graph replay reused stale changed inputs");
  };
  auto stream = at::cuda::getStreamFromPool(false, device.index());
  at::cuda::getCurrentCUDAStream(device.index()).synchronize();
  c10::cuda::CUDAStreamGuard stream_guard(stream);
  auto execute = [&] {
    return gemma4_qkv_norm_rotary_cuda(q, k, v, q_scale, k_scale, ones, cos, sin, positions, 1e-6);
  };
  for (int i = 0; i < 3; ++i)
    execute();
  stream.synchronize();
  at::cuda::CUDAGraph graph;
  graph.capture_begin();
  auto captured = execute();
  graph.capture_end();
  graph.replay();
  check(captured);
  q.add_(.125);
  k.add_(-.25);
  v.add_(.5);
  q_scale.mul_(.75);
  k_scale.add_(.1);
  auto [changed_cos, changed_sin] = tables(9, width, true);
  cos.copy_(changed_cos);
  sin.copy_(changed_sin);
  positions.copy_(make_positions({1, 7, 2, 0, 4}));
  graph.replay();
  check(captured);
  std::cout << "Gemma4 common attention-transform norm/rotary and V unit norm parity passed\n";
}
#endif

void gemma_block_attention_test(const c10::Device& device, at::ScalarType dtype) {
  const bool cpu = device.is_cpu();
  auto load = [&](const Weights& host) {
    Weights weights;
    for (const auto& [name, value] : host)
      weights[name] = value.to(device, dtype);
    return weights;
  };
  auto on_device = [&](PackedInput packed) {
    if (!cpu) {
      packed.ids = packed.ids.to(device);
      packed.types = packed.types.to(device);
      packed.positions = packed.positions.to(device);
      packed.cumulative = packed.cumulative.to(device);
    }
    return packed;
  };
  const auto cross_shape = cpu ? 1e-5 : 1e-2;
  Fixture f("gemma4", 8, 4, 2);
  auto block_cfg = f.cfg;
  block_cfg.values["_decision.kind"] = "option_tokens";
  block_cfg.values["_decision_attention_mode"] = "block_bidirectional";
  Weights block_weights = load(f.weights);
  auto block = create_gemma(block_cfg, block_weights, device, dtype);
  block->ready();
  TORCH_CHECK(!block->supports_radix() && !block->supports_prefix_cache(0),
              "Block-bidirectional Gemma enabled causal radix/prefix state");
  // Row0: 6 tokens, 3-token prefix. Row1: 4 tokens, 2-token prefix.
  // Row2: 3 tokens, prefix-only row (no suffix).
  std::vector<int64_t> ids{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
  std::vector<int32_t> offsets{0, 6, 10, 13};
  auto packed = on_device(input(ids, offsets));
  packed.block_prefix_lengths = {3, 2, 3};
  packed.block_prefix_sources = {0, 1, 2};
  auto output = block->forward(packed);
  // The decision bridge reads pooled last rows; they must match the same rows
  // selected from the full forward on both CPU and CUDA.
  std::vector<int64_t> ends;
  for (size_t row = 0; row + 1 < offsets.size(); ++row)
    ends.push_back(offsets[row + 1] - 1);
  auto selected_ends = output.index_select(0, at::tensor(ends, output.options().dtype(at::kLong)));
  TORCH_CHECK(
      at::allclose(block->forward_pooled(packed, true), selected_ends, cross_shape, cross_shape),
      "Block Gemma pooled rows differed from selected forward rows");
  auto changed_future = packed;
  changed_future.ids = packed.ids.clone();
  changed_future.ids.index_put_({5}, 42);
  auto future = block->forward(changed_future);
  TORCH_CHECK(at::equal(output.narrow(0, 0, 3), future.narrow(0, 0, 3)),
              "Block Gemma prefix attended to its question suffix");
  TORCH_CHECK((output.narrow(0, 3, 2) - future.narrow(0, 3, 2)).abs().max().item<float>() > 1e-5,
              "Block Gemma suffix lost bidirectional attention");
  TORCH_CHECK(at::equal(output.narrow(0, 6, 7), future.narrow(0, 6, 7)),
              "Changing one block row leaked into another row or a prefix-only row");
  auto changed_prefix = packed;
  changed_prefix.ids = packed.ids.clone();
  changed_prefix.ids.index_put_({1}, 42);
  auto prefixed = block->forward(changed_prefix);
  TORCH_CHECK((output.narrow(0, 3, 3) - prefixed.narrow(0, 3, 3)).abs().max().item<float>() > 1e-5,
              "Block Gemma suffix ignored its prefix");
  // The prefix-only execution is the explicit reference for the block prefix mask.
  std::vector<int64_t> prefix_ids(ids.begin(), ids.begin() + 3);
  std::vector<int32_t> prefix_offsets{0, 3};
  auto prefix_only = on_device(input(prefix_ids, prefix_offsets));
  prefix_only.block_prefix_lengths = {3};
  prefix_only.block_prefix_sources = {0};
  TORCH_CHECK(
      at::allclose(output.narrow(0, 0, 3), block->forward(prefix_only), cross_shape, cross_shape),
      "Block Gemma prefix differs from an isolated prefix forward");
  // An all-prefix batch leaves the suffix gather empty; rows must stay isolated
  // and pooled rows must still match the full forward.
  std::vector<int64_t> prefix_batch_ids{1, 2, 3, 4, 5, 6};
  std::vector<int32_t> prefix_batch_offsets{0, 3, 6};
  auto prefix_batch = on_device(input(prefix_batch_ids, prefix_batch_offsets));
  prefix_batch.block_prefix_lengths = {3, 3};
  prefix_batch.block_prefix_sources = {0, 1};
  auto prefix_batch_output = block->forward(prefix_batch);
  for (size_t row = 0; row + 1 < prefix_batch_offsets.size(); ++row) {
    std::vector<int64_t> row_ids(prefix_batch_ids.begin() + prefix_batch_offsets[row],
                                 prefix_batch_ids.begin() + prefix_batch_offsets[row + 1]);
    std::vector<int32_t> row_offsets{0, 3};
    auto single = on_device(input(row_ids, row_offsets));
    single.block_prefix_lengths = {3};
    single.block_prefix_sources = {0};
    TORCH_CHECK(at::allclose(prefix_batch_output.narrow(0, prefix_batch_offsets[row], 3),
                             block->forward(single),
                             cross_shape,
                             cross_shape),
                "All-prefix block Gemma row crossed its batch boundary: row=",
                row);
  }
  auto prefix_batch_selected = prefix_batch_output.index_select(
      0,
      at::tensor(std::vector<int64_t>{prefix_batch_offsets[1] - 1, prefix_batch_offsets[2] - 1},
                 prefix_batch_output.options().dtype(at::kLong)));
  TORCH_CHECK(at::allclose(block->forward_pooled(prefix_batch, true),
                           prefix_batch_selected,
                           cross_shape,
                           cross_shape),
              "All-prefix block Gemma pooled rows differed from forward rows");
  for (size_t row = 0; row + 1 < offsets.size(); ++row) {
    const auto length = offsets[row + 1] - offsets[row];
    std::vector<int64_t> row_ids(ids.begin() + offsets[row], ids.begin() + offsets[row + 1]);
    std::vector<int32_t> row_offsets{0, length};
    auto single = on_device(input(row_ids, row_offsets));
    single.block_prefix_lengths = {packed.block_prefix_lengths[row]};
    single.block_prefix_sources = {0};
    TORCH_CHECK(at::allclose(output.narrow(0, offsets[row], length),
                             block->forward(single),
                             cross_shape,
                             cross_shape),
                "Mixed-length block Gemma row crossed its prefix or batch boundary: row=",
                row);
  }
  // Shared prefix witnesses must not change the result.
  std::vector<int64_t> shared_ids{1, 2, 3, 4, 5, 1, 2, 3, 8, 9};
  std::vector<int32_t> shared_offsets{0, 5, 10};
  auto shared = on_device(input(shared_ids, shared_offsets));
  shared.block_prefix_lengths = {3, 3};
  shared.block_prefix_sources = {0, 0};
  auto shared_output = block->forward(shared);
  shared.block_prefix_sources = {0, 1};
  TORCH_CHECK(at::equal(shared_output, block->forward(shared)),
              "Shared block-prefix witnesses changed Gemma outputs");

  // Shared-KV geometry with full-attention producers first: layers 2/3 reuse
  // producers 0/1. The block mask must not depend on the default producer order.
  Fixture sharing("gemma4", 8, 4, 2);
  auto sharing_cfg = sharing.cfg;
  sharing_cfg.values["layer_types.0"] = "full_attention";
  sharing_cfg.values["layer_types.1"] = "sliding_attention";
  sharing_cfg.values["layer_types.2"] = "full_attention";
  sharing_cfg.values["layer_types.3"] = "sliding_attention";
  sharing_cfg.values["_decision.kind"] = "option_tokens";
  sharing_cfg.values["_decision_attention_mode"] = "block_bidirectional";
  Weights sharing_weights = load(sharing.weights);
  TORCH_CHECK(sharing_weights.count("layers.1.self_attn.k_proj.weight") &&
                  !sharing_weights.count("layers.2.self_attn.k_proj.weight") &&
                  !sharing_weights.count("layers.3.self_attn.k_proj.weight"),
              "Shared-KV block fixture did not exercise producer reuse");
  auto sharing_block = create_gemma(sharing_cfg, sharing_weights, device, dtype);
  sharing_block->ready();
  auto sharing_input = on_device(input(ids, offsets));
  sharing_input.block_prefix_lengths = packed.block_prefix_lengths;
  sharing_input.block_prefix_sources = packed.block_prefix_sources;
  auto sharing_output = sharing_block->forward(sharing_input);
  auto sharing_changed = sharing_input;
  sharing_changed.ids = sharing_input.ids.clone();
  sharing_changed.ids.index_put_({4}, 42);
  TORCH_CHECK(at::equal(sharing_output.narrow(0, 0, 3),
                        sharing_block->forward(sharing_changed).narrow(0, 0, 3)),
              "Shared-KV block prefix observed its suffix");
  auto sharing_prefix = sharing_input;
  sharing_prefix.ids = sharing_input.ids.clone();
  sharing_prefix.ids.index_put_({1}, 42);
  TORCH_CHECK(
      (sharing_output.narrow(0, 3, 3) - sharing_block->forward(sharing_prefix).narrow(0, 3, 3))
              .abs()
              .max()
              .item<float>() > 1e-5,
      "Shared-KV block suffix ignored its prefix");

  // All-sliding block mode must stay exactly causal.
  Fixture sliding("gemma4", 8, 4, 2);
  auto sliding_cfg = sliding.cfg;
  for (int64_t layer = 0; layer < 4; ++layer)
    sliding_cfg.values["layer_types." + std::to_string(layer)] = "sliding_attention";
  auto sliding_block_cfg = sliding_cfg;
  sliding_block_cfg.values["_decision.kind"] = "option_tokens";
  sliding_block_cfg.values["_decision_attention_mode"] = "block_bidirectional";
  Weights sliding_weights = load(sliding.weights);
  auto sliding_block = create_gemma(sliding_block_cfg, sliding_weights, device, dtype);
  sliding_block->ready();
  auto causal_cfg = sliding_cfg;
  causal_cfg.values["_decision.kind"] = "option_tokens";
  auto causal = create_gemma(causal_cfg, sliding_weights, device, dtype);
  causal->ready();
  auto sliding_input = on_device(input(ids, offsets));
  sliding_input.block_prefix_lengths = packed.block_prefix_lengths;
  sliding_input.block_prefix_sources = packed.block_prefix_sources;
  TORCH_CHECK(at::equal(sliding_block->forward(sliding_input),
                        causal->forward(on_device(input(ids, offsets)))),
              "Block-mode sliding layers diverged from causal attention");

  // All-full block mode is bidirectional inside the question suffix.
  Fixture full("gemma4", 8, 4, 2);
  auto full_cfg = full.cfg;
  for (int64_t layer = 0; layer < 4; ++layer)
    full_cfg.values["layer_types." + std::to_string(layer)] = "full_attention";
  auto full_block_cfg = full_cfg;
  full_block_cfg.values["_decision.kind"] = "option_tokens";
  full_block_cfg.values["_decision_attention_mode"] = "block_bidirectional";
  Weights full_weights = load(full.weights);
  auto full_block = create_gemma(full_block_cfg, full_weights, device, dtype);
  full_block->ready();
  auto full_causal_cfg = full_cfg;
  full_causal_cfg.values["_decision.kind"] = "option_tokens";
  auto full_causal = create_gemma(full_causal_cfg, full_weights, device, dtype);
  full_causal->ready();
  std::vector<int64_t> full_ids{1, 2, 3, 4, 5};
  std::vector<int32_t> full_offsets{0, 5};
  auto full_input = on_device(input(full_ids, full_offsets));
  full_input.block_prefix_lengths = {2};
  full_input.block_prefix_sources = {0};
  auto bidirectional = full_block->forward(full_input);
  auto full_future = full_input;
  full_future.ids = full_input.ids.clone();
  full_future.ids.index_put_({4}, 42);
  auto future_bidirectional = full_block->forward(full_future);
  TORCH_CHECK(at::equal(bidirectional.narrow(0, 0, 2), future_bidirectional.narrow(0, 0, 2)),
              "Full-attention block prefix observed its suffix");
  TORCH_CHECK((bidirectional.narrow(0, 2, 2) - future_bidirectional.narrow(0, 2, 2))
                      .abs()
                      .max()
                      .item<float>() > 1e-5,
              "Full-attention block suffix was not bidirectional");
  auto causal_output = full_causal->forward(on_device(input(full_ids, full_offsets)));
  auto causal_changed = full_ids;
  causal_changed[4] = 42;
  auto causal_future = full_causal->forward(on_device(input(causal_changed, full_offsets)));
  TORCH_CHECK(at::equal(causal_output.narrow(0, 2, 2), causal_future.narrow(0, 2, 2)),
              "Causal reference observed a future token");
  // Prefix noncausality: an earlier prefix query must see a later prefix key in
  // the block mask, while the causal reference must not.
  auto full_prefix_future = full_input;
  full_prefix_future.ids = full_input.ids.clone();
  full_prefix_future.ids.index_put_({1}, 42);
  TORCH_CHECK(
      (bidirectional.narrow(0, 0, 1) - full_block->forward(full_prefix_future).narrow(0, 0, 1))
              .abs()
              .max()
              .item<float>() > 1e-5,
      "Full-attention block prefix was not bidirectional within its block");
  auto causal_prefix_ids = full_ids;
  causal_prefix_ids[1] = 42;
  auto causal_prefix = full_causal->forward(on_device(input(causal_prefix_ids, full_offsets)));
  TORCH_CHECK(at::equal(causal_output.narrow(0, 0, 1), causal_prefix.narrow(0, 0, 1)),
              "Causal reference observed a future prefix token");
  if (!cpu) {
    // Decision op3 pools last rows. The graph must replay mutated ids from the
    // same storage and keep distinct entries for distinct block boundaries.
    CudaGraphCache graphs(2, 4096, 1);
    const auto& graph_input = packed;
    const auto fresh = block->forward_pooled(graph_input, true);
    auto captured = graphs.forward(*block, graph_input, 3);
    TORCH_CHECK(at::allclose(captured, fresh, cross_shape, cross_shape),
                "Block Gemma graph capture changed pooled outputs");
    TORCH_CHECK(at::allclose(graphs.forward(*block, graph_input, 3),
                             block->forward_pooled(graph_input, true),
                             cross_shape,
                             cross_shape),
                "Block Gemma graph replay changed pooled outputs");
    TORCH_CHECK(graphs.size() == 1 && graphs.stats().captures == 1,
                "Block Gemma graph did not capture the block batch");
    const auto ids_pointer = graph_input.ids.const_data_ptr();
    graph_input.ids.add_(1).remainder_(64);
    TORCH_CHECK(graph_input.ids.const_data_ptr() == ids_pointer,
                "Block Gemma graph ids mutation replaced storage");
    auto mutated = block->forward_pooled(graph_input, true);
    TORCH_CHECK(
        at::allclose(graphs.forward(*block, graph_input, 3), mutated, cross_shape, cross_shape),
        "Block Gemma graph replay ignored in-place ids mutation");
    auto changed = graph_input;
    changed.block_prefix_lengths = {2, 3, 2};
    changed.block_prefix_sources = {0, 1, 2};
    auto changed_fresh = block->forward_pooled(changed, true);
    TORCH_CHECK(
        at::allclose(graphs.forward(*block, changed, 3), changed_fresh, cross_shape, cross_shape),
        "Block Gemma graph changed the new prefix-layout outputs");
    TORCH_CHECK(graphs.size() == 2 && graphs.stats().captures == 2,
                "Block Gemma graph cache did not separate block layout entries");
    TORCH_CHECK(at::allclose(graphs.forward(*block, graph_input, 3),
                             block->forward_pooled(graph_input, true),
                             cross_shape,
                             cross_shape),
                "Block Gemma graph replay reused the wrong block layout");
    TORCH_CHECK(at::allclose(graphs.forward(*block, changed, 3),
                             block->forward_pooled(changed, true),
                             cross_shape,
                             cross_shape),
                "Block Gemma changed-layout graph replay was invalid");
  }
  std::cout << "gemma4: block prefix isolation, suffix bidirectionality, causal sliding and mixed "
               "lengths passed on "
            << device << " " << at::toString(dtype) << "\n";
}

void add_tiny_vision(Options& cfg, Weights& weights) {
  cfg.values["vision_config.hidden_size"] = "8";
  cfg.values["vision_config.num_attention_heads"] = "2";
  cfg.values["vision_config.num_key_value_heads"] = "1";
  cfg.values["vision_config.head_dim"] = "4";
  cfg.values["vision_config.num_hidden_layers"] = "1";
  cfg.values["vision_config.intermediate_size"] = "12";
  auto put = [&](const std::string& name, std::initializer_list<int64_t> shape, bool norm = false) {
    weights[name] = norm ? at::ones(shape) : at::randn(shape) * .03;
  };
  put("vision_tower.patch_embedder.input_proj.weight", {8, 768});
  put("vision_tower.patch_embedder.position_embedding_table", {2, 16, 8});
  auto p = std::string("vision_tower.encoder.layers.0.");
  for (const auto& name : {"input_layernorm",
                           "post_attention_layernorm",
                           "pre_feedforward_layernorm",
                           "post_feedforward_layernorm"})
    put(p + name + ".weight", {8}, true);
  put(p + "self_attn.q_proj.linear.weight", {8, 8});
  put(p + "self_attn.k_proj.linear.weight", {4, 8});
  put(p + "self_attn.v_proj.linear.weight", {4, 8});
  put(p + "self_attn.o_proj.linear.weight", {8, 8});
  put(p + "self_attn.q_norm.weight", {4}, true);
  put(p + "self_attn.k_norm.weight", {4}, true);
  put(p + "mlp.gate_proj.linear.weight", {12, 8});
  put(p + "mlp.up_proj.linear.weight", {12, 8});
  put(p + "mlp.down_proj.linear.weight", {8, 12});
  put("embed_vision.embedding_projection.weight", {16, 8});
}

void add_vision_bounds(Weights& weights,
                       const std::string& base,
                       double input_min,
                       double input_max,
                       double output_min,
                       double output_max) {
  const auto options = at::TensorOptions().dtype(at::kFloat);
  weights[base + ".input_min"] = at::scalar_tensor(input_min, options);
  weights[base + ".input_max"] = at::scalar_tensor(input_max, options);
  weights[base + ".output_min"] = at::scalar_tensor(output_min, options);
  weights[base + ".output_max"] = at::scalar_tensor(output_max, options);
}

Weights clone_weights(const Weights& weights) {
  Weights copy;
  for (const auto& [name, value] : weights)
    copy[name] = value.clone();
  return copy;
}

at::Tensor prepare_vision_features(const Options& cfg,
                                   const Weights& weights,
                                   const at::Tensor& pixels,
                                   at::ScalarType dtype) {
  Weights typed;
  for (const auto& [name, value] : weights)
    typed[name] = value.to(dtype);
  auto model = create_gemma(cfg, typed, c10::Device("cpu"), dtype);
  model->ready();
  const std::vector<int32_t> offsets{0, 3};
  auto batch = input({1, 2, 3}, offsets);
  ImageInput image;
  image.pixels = pixels;
  image.grid_thw = {1, 3, 3};
  image.merge_size = 3;
  image.token_start = 0;
  image.token_count = 1;
  batch.images.push_back(image);
  model->prepare_input(batch);
  return batch.images[0].prepared_features.to(at::kFloat).clone();
}

void gemma4_vision_clipping_test() {
  at::manual_seed(418);
  Fixture f("gemma4", 8, 1, 0);
  add_tiny_vision(f.cfg, f.weights);
  f.cfg.values["vision_config.use_clipped_linears"] = "false";
  const auto pixels = at::arange(9 * 768, at::kFloat).reshape({9, 768}) / 1000.;
  const auto baseline = prepare_vision_features(f.cfg, f.weights, pixels, at::kFloat);
  const auto mlp = std::string("vision_tower.encoder.layers.0.mlp.");
  const std::vector<std::string> bases = {
      "vision_tower.encoder.layers.0.self_attn.q_proj",
      "vision_tower.encoder.layers.0.self_attn.k_proj",
      "vision_tower.encoder.layers.0.self_attn.v_proj",
      "vision_tower.encoder.layers.0.self_attn.o_proj",
      mlp + "gate_proj",
      mlp + "up_proj",
      mlp + "down_proj",
  };
  for (const auto& base : bases) {
    for (int mode = 0; mode < 2; ++mode) {
      auto cfg = f.cfg;
      cfg.values["vision_config.use_clipped_linears"] = "true";
      auto weights = clone_weights(f.weights);
      const double wide = 1e30;
      // Exercise both checkpoint clipping stages independently: zero the input
      // clamp with a wide output clamp, then the reverse.
      if (mode == 0)
        add_vision_bounds(weights, base, 0., 0., -wide, wide);
      else
        add_vision_bounds(weights, base, -wide, wide, 0., 0.);
      const auto clipped = prepare_vision_features(cfg, weights, pixels, at::kFloat);
      TORCH_CHECK(!at::allclose(baseline, clipped, 1e-6, 1e-6),
                  "Gemma4 ",
                  base,
                  mode == 0 ? " input" : " output",
                  " clamp was not applied");
    }
  }
  // Bounds present on an unclipped checkpoint must remain inert.
  auto inert_cfg = f.cfg;
  auto inert_weights = clone_weights(f.weights);
  add_vision_bounds(
      inert_weights, "vision_tower.encoder.layers.0.self_attn.q_proj", 0., 0., 0., 0.);
  TORCH_CHECK(at::allclose(baseline,
                           prepare_vision_features(inert_cfg, inert_weights, pixels, at::kFloat),
                           1e-6,
                           1e-6),
              "Gemma4 applied clip bounds while use_clipped_linears was false");
  // The BF16 path must also honor bounds. V and down give an observable
  // effect without relying on tiny softmax changes in a low-precision Q/K.
  auto bf16_cfg = f.cfg;
  bf16_cfg.values["vision_config.use_clipped_linears"] = "false";
  const auto bf16_baseline = prepare_vision_features(bf16_cfg, f.weights, pixels, at::kBFloat16);
  auto bf16_weights = clone_weights(f.weights);
  add_vision_bounds(
      bf16_weights, "vision_tower.encoder.layers.0.self_attn.v_proj", 0., 0., -1e30, 1e30);
  bf16_cfg.values["vision_config.use_clipped_linears"] = "true";
  TORCH_CHECK(!at::equal(bf16_baseline,
                         prepare_vision_features(bf16_cfg, bf16_weights, pixels, at::kBFloat16)),
              "Gemma4 BF16 vision input clamp was not applied");
  auto bf16_output_weights = clone_weights(f.weights);
  add_vision_bounds(
      bf16_output_weights, "vision_tower.encoder.layers.0.mlp.down_proj", -1e30, 1e30, 0., 0.);
  TORCH_CHECK(
      !at::equal(bf16_baseline,
                 prepare_vision_features(bf16_cfg, bf16_output_weights, pixels, at::kBFloat16)),
      "Gemma4 BF16 vision output clamp was not applied");
  std::cout << "gemma4: vision q/k/v/o and MLP input/output clips passed\n";
}

PackedInput image_input(const at::Tensor& features) {
  static const std::vector<int32_t> offsets{0, 4};
  auto batch = input({1, 63, 2, 3}, offsets);
  ImageInput image;
  image.pixels = at::zeros({9, 768});
  image.grid_thw = {1, 3, 3};
  image.merge_size = 3;
  image.token_start = 1;
  image.token_count = features.size(0);
  image.sequence_start = 0;
  image.prepared_features = features;
  batch.images.push_back(image);
  return batch;
}

void gemma4_image_ple_test() {
  at::manual_seed(418);
  // Four layers with two shared-KV layers keeps the E4B-like shared path active.
  Fixture f("gemma4", 8, 4, 2);
  add_tiny_vision(f.cfg, f.weights);
  f.cfg.values["rms_norm_eps"] = "1e6";
  f.cfg.values["pad_token_id"] = "0";
  for (int64_t layer = 0; layer < 4; ++layer) {
    const auto p = "layers." + std::to_string(layer) + ".";
    for (const auto* name : {"self_attn.q_proj.weight",
                             "self_attn.k_proj.weight",
                             "self_attn.v_proj.weight",
                             "self_attn.o_proj.weight",
                             "mlp.gate_proj.weight",
                             "mlp.up_proj.weight",
                             "mlp.down_proj.weight"}) {
      if (f.weights.count(p + name))
        f.weights[p + name].zero_();
    }
    f.weights[p + "per_layer_input_gate.weight"].zero_();
    f.weights[p + "per_layer_projection.weight"].zero_();
  }
  f.weights["per_layer_model_projection.weight"].zero_();
  f.weights["embed_tokens_per_layer.weight"].zero_();
  // A constant image-token gate and a visible layer-0 PLE path keep the
  // comparison linear enough to attribute differences to PLE alone.
  f.weights["layers.0.per_layer_input_gate.weight"].select(0, 0).select(0, 0).fill_(1000.);
  f.weights["layers.0.per_layer_projection.weight"].select(0, 0).select(0, 0).fill_(1000.);

  auto run = [&](Weights weights, const at::Tensor& pixels) {
    auto model = create_gemma(f.cfg, weights, c10::Device("cpu"), at::kFloat);
    model->ready();
    return model->forward(image_input(pixels));
  };
  auto constant_image = [](float value) {
    auto features = at::zeros({1, 16});
    features.select(0, 0).select(0, 0).fill_(1.);
    features.select(0, 0).select(0, 1).fill_(value);
    return features;
  };

  // Token identity must use the pad row on image slots, not the image token row.
  Weights pad = clone_weights(f.weights);
  pad["embed_tokens_per_layer.weight"].select(0, 0).select(0, 0).fill_(1.);
  auto image_row = clone_weights(pad);
  image_row["embed_tokens_per_layer.weight"].select(0, 63).select(0, 0).fill_(2.);
  const auto image = constant_image(.1);
  TORCH_CHECK(at::allclose(run(pad, image), run(image_row, image), 1e-6, 1e-6),
              "Gemma4 image PLE token identity used the image id instead of the pad id");

  // Context projection must use the injected image state, not the pre-image
  // token embedding. Both models share the same image injection, so A-B
  // isolates the PLE context contribution.
  auto context_weights = clone_weights(f.weights);
  context_weights["per_layer_model_projection.weight"].select(0, 0).select(0, 1).fill_(1.);
  auto no_context = clone_weights(f.weights);
  const auto first = constant_image(.1), second = constant_image(.3);
  const auto first_delta = run(context_weights, first) - run(no_context, first);
  const auto second_delta = run(context_weights, second) - run(no_context, second);
  TORCH_CHECK(!at::allclose(first_delta.narrow(0, 1, 1), second_delta.narrow(0, 1, 1), 1e-6, 1e-6),
              "Gemma4 image PLE context used pre-image token embeddings");

  // Radix compact rows must pad every aliased image row, not just the first
  // logical occurrence. Both sequences share the same image prefix.
  const std::vector<int32_t> offsets{0, 3, 6};
  auto logical = input({1, 63, 2, 1, 63, 3}, offsets);
  ImageInput first_image, second_image;
  first_image.pixels = at::zeros({9, 768});
  second_image.pixels = at::zeros({9, 768});
  first_image.grid_thw = {1, 3, 3};
  second_image.grid_thw = {1, 3, 3};
  first_image.merge_size = 3;
  second_image.merge_size = 3;
  first_image.token_start = 1;
  first_image.sequence_start = 0;
  second_image.token_start = 4;
  second_image.sequence_start = 3;
  first_image.token_count = second_image.token_count = 1;
  first_image.prepared_features = second_image.prepared_features = image;
  logical.images = {first_image, second_image};
  auto compact = logical;
  compact.compact_ids = at::tensor({1, 63, 2, 3}, at::kLong);
  compact.compact_positions = at::tensor({0, 1, 2, 2}, at::kLong);
  compact.compact_types = at::zeros({4}, at::kLong);
  compact.scatter_unfold = at::tensor({0, 1, 2, 0, 1, 3}, at::kLong);
  compact.fold_gather = at::tensor({0, 1, 2, 5}, at::kLong);
  validate_radix(compact, true);
  auto causal_cfg = f.cfg;
  causal_cfg.values["_decision.kind"] = "option_tokens";
  auto radix_model = create_gemma(causal_cfg, pad, c10::Device("cpu"), at::kFloat);
  radix_model->ready();
  TORCH_CHECK(
      at::allclose(radix_model->forward(logical), radix_model->forward(compact), 1e-6, 1e-6),
      "Gemma4 image PLE radix fold did not pad aliased image rows");
  std::cout << "gemma4: image PLE pad identity, post-image context and shared KV passed\n";
}

void gemma4_image_visibility_test() {
  at::manual_seed(418);
  Fixture f("gemma4", 8, 2, 0);
  add_tiny_vision(f.cfg, f.weights);
  for (int64_t layer = 0; layer < 2; ++layer) {
    const auto p = "layers." + std::to_string(layer) + ".";
    f.cfg.values["layer_types." + std::to_string(layer)] = "sliding_attention";
    // Uniform attention isolates visibility; identity V/O preserve direction,
    // unlike rank-one projections whose magnitude disappears in RMSNorm.
    for (const auto* name : {"self_attn.q_proj.weight", "self_attn.k_proj.weight"})
      f.weights[p + name].zero_();
    for (const auto* name : {"self_attn.v_proj.weight", "self_attn.o_proj.weight"}) {
      const auto& matrix = f.weights.at(p + name);
      f.weights[p + name] = at::eye(matrix.size(0), matrix.size(1), matrix.options());
    }
    for (const auto* name : {"mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"})
      f.weights[p + name].zero_();
  }
  f.cfg.values["_decision.kind"] = "option_tokens";
  auto features = at::randn({2, 16}) * .1;
  auto changed = features.clone();
  changed.select(0, 1).add_(1.);
  auto run = [&](bool vision, const at::Tensor& pixels) {
    auto cfg = f.cfg;
    if (vision)
      cfg.values["use_bidirectional_attention"] = "vision";
    auto model = create_gemma(cfg, f.weights, c10::Device("cpu"), at::kFloat);
    model->ready();
    return model->forward(image_input(pixels));
  };
  const auto causal = run(false, features), causal_changed = run(false, changed);
  TORCH_CHECK(at::allclose(causal.narrow(0, 1, 1), causal_changed.narrow(0, 1, 1), 1e-7, 1e-7),
              "Gemma4 use_bidirectional_attention=None applied the vision overlay");
  const auto vision = run(true, features), vision_changed = run(true, changed);
  TORCH_CHECK(!at::allclose(vision.narrow(0, 1, 1), vision_changed.narrow(0, 1, 1), 1e-6, 1e-6),
              "Gemma4 use_bidirectional_attention=vision did not make image tokens bidirectional");
  std::cout << "gemma4: use_bidirectional_attention vision overlay gate passed\n";
}

int main(int argc, char** argv) {
  embedding_layer_key_aliases();
  gemma4_image_ple_test();
  gemma4_image_visibility_test();
  gemma4_vision_clipping_test();
  gemma4_layer_head_dim_geometry();
  feature_cache_bounds();
  gemma_block_attention_test(c10::Device("cpu"), at::kFloat);
#ifdef INFINITY_TORCH_CUDA_KERNELS
  if (argc > 1) {
    gemma_block_attention_test(c10::Device(argv[1]), at::kBFloat16);
    gemma_low_precision_pointwise(c10::Device(argv[1]));
    gemma4_qkv_norm_rotary_test(c10::Device(argv[1]));
    for (const auto* family : {"gemma3", "gemma3_text", "gemma4", "embedding_gemma2"})
      dynamic_fp8(c10::Device(argv[1]), family);
    feature_cache_stream_lifetime(c10::Device(argv[1]));
    publisher_bf16_score_readout(c10::Device(argv[1]));
    gemma4_moe_forward_parity(c10::Device(argv[1]));
  }
#endif
  at::manual_seed(418);
  audio_operations(c10::Device("cpu"));
  if (argc > 1)
    audio_operations(c10::Device(argv[1]));
  for (const auto& family : {"gemma3", "gemma4", "embedding_gemma2"}) {
    Fixture f(family);
    auto model = create_gemma(f.cfg, f.weights, c10::Device("cpu"), at::kFloat);
    model->ready();
    const std::vector<int64_t> a = {1, 2, 3, 4, 5}, b = {6, 7};
    std::vector<int64_t> packed = a;
    packed.insert(packed.end(), b.begin(), b.end());
    const std::vector<int32_t> ao = {0, 5}, bo = {0, 2}, po = {0, 5, 7};
    auto dense_metadata = f.cfg;
    dense_metadata.values["enable_moe_block"] = "false";
    dense_metadata.values["num_experts"] = "3";
    dense_metadata.values["top_k_experts"] = "0";
    auto dense_model = create_gemma(dense_metadata, f.weights, c10::Device("cpu"), at::kFloat);
    dense_model->ready();
    auto output = model->forward(input(packed, po));
    TORCH_CHECK(at::equal(model->forward_pooled(input(packed, po), true),
                          output.index_select(0, at::tensor({4, 6}, at::kLong))),
                family,
                " last-token readout omitted a packed sequence");
    if (f.legacy) {
      auto with_vision = f.cfg;
      with_vision.values["vision_config.hidden_size"] = "8";
      with_vision.values["vision_config.num_attention_heads"] = "2";
      with_vision.values["vision_config.num_hidden_layers"] = "1";
      auto text_model = create_gemma(with_vision, f.weights, c10::Device("cpu"), at::kFloat);
      text_model->ready();
      TORCH_CHECK(!text_model->supports_images() &&
                      at::equal(output, text_model->forward(input(packed, po))),
                  "Unused Gemma3 vision metadata changed text startup or inference");
    }

    TORCH_CHECK(at::equal(output, dense_model->forward(input(packed, po))),
                "Disabled Gemma expert metadata changed full dense outputs");
    check_prefix_cache(*model, input(packed, po));
    auto independent = at::cat({model->forward(input(a, ao)), model->forward(input(b, bo))}, 0);
    TORCH_CHECK(
        at::allclose(output, independent, 1e-5, 1e-6), family, " packed cross-sequence leakage");
    auto changed = packed;
    changed[0] = 55;
    auto modified = model->forward(input(changed, po));
    TORCH_CHECK(at::allclose(output.narrow(0, 5, 2), modified.narrow(0, 5, 2), 1e-5, 1e-6),
                family,
                " cross-sequence attention leaked");
    TORCH_CHECK(output.size(0) == 7 && output.size(1) == (f.embedding2 ? 12 : 16),
                family,
                " unexpected packed output geometry");
    std::cout << family << ": packed/single parity and independent sequence isolation passed\n";
    if (std::string(family) == "gemma4") {
      auto decision_options = f.cfg;
      decision_options.values["_decision.kind"] = "option_tokens";
      auto decision = create_gemma(decision_options, f.weights, c10::Device("cpu"), at::kFloat);
      decision->ready();
      auto decision_output = decision->forward(input(packed, po));
      auto changed_future = packed;
      changed_future[4] = 55;
      auto decision_future = decision->forward(input(changed_future, po));
      TORCH_CHECK(at::allclose(
                      decision_output.narrow(0, 0, 4), decision_future.narrow(0, 0, 4), 1e-5, 1e-6),
                  "Gemma4 option-token decision observed future text tokens");
      TORCH_CHECK(!at::allclose(output.narrow(0, 0, 4),
                                model->forward(input(changed_future, po)).narrow(0, 0, 4),
                                1e-5,
                                1e-6),
                  "Gemma4 embedding unexpectedly lost bidirectional attention");
      auto decision_separate =
          at::cat({decision->forward(input(a, ao)), decision->forward(input(b, bo))}, 0);
      TORCH_CHECK(at::allclose(decision_output, decision_separate, 1e-5, 1e-6),
                  "Gemma4 causal decision leaked across packed sequences");
      TORCH_CHECK(
          at::allclose(
              decision_output.narrow(0, 0, 1), decision->forward(input({1}, {0, 1})), 1e-5, 1e-6),
          "Gemma4 causal decision prefix differs from independent one-token reference");
      TORCH_CHECK(decision->supports_radix() && !model->supports_radix(),
                  "Gemma folding capability must be causal");
      const std::vector<int32_t> fold_offsets = {0, 3, 7};
      auto logical = input({1, 2, 3, 1, 2, 4, 5}, fold_offsets), compact = logical;
      compact.compact_ids = at::tensor({1, 2, 3, 4, 5}, at::kLong);
      compact.compact_positions = at::tensor({0, 1, 2, 2, 3}, at::kLong);
      compact.compact_types = at::zeros({5}, at::kLong);
      compact.scatter_unfold = at::tensor({0, 1, 2, 0, 1, 3, 4}, at::kLong);
      compact.fold_gather = at::tensor({0, 1, 2, 5, 6}, at::kLong);
      validate_radix(compact, true);
      auto scored_weights = f.weights;
      scored_weights["score.weight"] = at::randn({2, 16}) * .03;
      auto scored_embedding_cfg = f.cfg;
      scored_embedding_cfg.values["model_type"] = "gemma4_text";
      auto scored_embedding =
          create_gemma(scored_embedding_cfg, scored_weights, c10::Device("cpu"), at::kFloat);
      scored_embedding->ready();
      TORCH_CHECK(!scored_embedding->supports_radix(),
                  "Unused score weights must not enable bidirectional embedding folding");
      auto scored_cfg = f.cfg;
      scored_cfg.values["_classifier"] = "true";
      auto scored = create_gemma(scored_cfg, scored_weights, c10::Device("cpu"), at::kFloat);
      scored->ready();
      TORCH_CHECK(scored->supports_radix(),
                  "Loaded Gemma sequence score head must support causal folding");
      auto full_score = scored->predict(logical, false);
      check_prefix_cache(*scored, logical, 1);
      check_complete_classifier_cache(*scored, logical);
      check_prefix_cache(*scored_embedding, logical);
      check_prefix_cache(*decision, logical);
      auto capped_cfg = scored_cfg;
      capped_cfg.values["final_logit_softcapping"] = "0.01";
      auto capped = create_gemma(capped_cfg, scored_weights, c10::Device("cpu"), at::kFloat);
      capped->ready();
      TORCH_CHECK(full_score.abs().max().item<float>() > 0.01f,
                  "Classifier fixture must expose accidental softcapping");
      TORCH_CHECK(at::equal(full_score, capped->predict(logical, false)),
                  "LM softcapping changed raw classifier logits");
      TORCH_CHECK(at::allclose(full_score, capped->predict(compact, false), 1e-5, 1e-6),
                  "LM softcapping changed folded classifier logits");
      TORCH_CHECK(at::allclose(full_score, scored->predict(compact, false), 1e-5, 1e-6),
                  "Gemma folded classifier changed sequence-end readout");
      auto publisher_cfg = capped_cfg;
      publisher_cfg.values["architectures"] = "[\"Gemma4ForSequenceClassification\"]";
      publisher_cfg.values["architectures.0"] = "Gemma4ForSequenceClassification";
      publisher_cfg.values["auto_map.AutoModelForSequenceClassification"] =
          "modeling_gemma4_sequence.Gemma4ForSequenceClassification";
      auto publisher = create_gemma(publisher_cfg, scored_weights, c10::Device("cpu"), at::kFloat);
      publisher->ready();
      check_complete_classifier_cache(*publisher, logical);
      auto publisher_hidden = decision->forward(logical);
      auto publisher_logits = at::linear(publisher_hidden, scored_weights.at("score.weight"));
      publisher_logits = publisher_logits / .01;
      publisher_logits = at::tanh(publisher_logits);
      publisher_logits = publisher_logits * .01;
      auto ends = logical.cumulative.narrow(0, 1, logical.batch).to(at::kLong) - 1;
      auto publisher_expected = publisher_logits.index_select(0, ends);
      TORCH_CHECK(at::equal(publisher_expected, publisher->predict(logical, false)),
                  "Declared Gemma4 sequence contract changed all-token readout");
      TORCH_CHECK(at::allclose(publisher_expected, publisher->predict(compact, false), 1e-5, 1e-6),
                  "Publisher sequence cap changed radix reconstruction");
      TORCH_CHECK(!at::allclose(full_score, publisher_expected, 1e-5, 1e-6),
                  "Publisher fixture must distinguish capped logits");
      auto unknown_cfg = publisher_cfg;
      unknown_cfg.values["auto_map.AutoModelForSequenceClassification"] =
          "other.Gemma4ForSequenceClassification";
      auto unknown = create_gemma(unknown_cfg, scored_weights, c10::Device("cpu"), at::kFloat);
      unknown->ready();
      TORCH_CHECK(at::equal(full_score, unknown->predict(logical, false)),
                  "Unrecognized sequence contract applied publisher cap");
      auto null_metadata_cfg = publisher_cfg;
      null_metadata_cfg.values["architectures"] = "null";
      auto null_metadata =
          create_gemma(null_metadata_cfg, scored_weights, c10::Device("cpu"), at::kFloat);
      null_metadata->ready();
      TORCH_CHECK(at::equal(full_score, null_metadata->predict(logical, false)),
                  "Nullable architecture metadata changed generic sequence logits");
      bool embedding_fold_rejected = false;
      try {
        scored->forward(compact);
      } catch (const c10::Error&) {
        embedding_fold_rejected = true;
      }
      TORCH_CHECK(embedding_fold_rejected,
                  "Loaded score head must not enable bidirectional embedding folding");
      auto full_hidden = decision->forward(logical), fold_hidden = decision->forward(compact);
      TORCH_CHECK(fold_hidden.sizes() == full_hidden.sizes() &&
                      at::allclose(full_hidden, fold_hidden, 1e-5, 1e-6),
                  "Gemma causal folding changed logical readout rows or shared KV/PLE");
      // Reordered representatives must reconstruct exactly the same logical rows.
      compact.compact_ids = compact.compact_ids.flip({0});
      compact.compact_positions = compact.compact_positions.flip({0});
      compact.fold_gather = compact.fold_gather.flip({0});
      compact.scatter_unfold = 4 - compact.scatter_unfold;
      validate_radix(compact, true);
      TORCH_CHECK(at::allclose(full_hidden, decision->forward(compact), 1e-5, 1e-6),
                  "Gemma folded output depends on representative order");
      TORCH_CHECK(at::allclose(full_score, scored->predict(compact, false), 1e-5, 1e-6),
                  "Gemma folded classifier depends on representative order");
      std::cout
          << "Gemma4 decisions causal; embeddings bidirectional; packed sequence isolation passed\n";
      Fixture mf(family);
      mf.cfg.values["enable_moe_block"] = "true";
      mf.cfg.values["num_experts"] = "3";
      mf.cfg.values["top_k_experts"] = "2";
      for (int layer = 0; layer < 4; ++layer) {
        auto p = "layers." + std::to_string(layer) + ".";
        mf.weights[p + "router.scale"] = at::ones({16});
        mf.weights[p + "router.proj.weight"] = at::zeros({3, 16});
        mf.weights[p + "router.per_expert_scale"] = at::ones({3});
        mf.weights[p + "experts.gate_up_proj"] = at::randn({3, 12, 16}) * .03;
        mf.weights[p + "experts.down_proj"] = at::randn({3, 16, 6}) * .03;
        for (const auto& name : {"post_feedforward_layernorm_1",
                                 "pre_feedforward_layernorm_2",
                                 "post_feedforward_layernorm_2"})
          mf.weights[p + name + ".weight"] = at::ones({16});
      }
      auto routed = create_gemma(mf.cfg, mf.weights, c10::Device("cpu"), at::kFloat);
      routed->ready();
      auto causal_moe_cfg = mf.cfg;
      causal_moe_cfg.values["_decision.kind"] = "option_tokens";
      auto causal_moe = create_gemma(causal_moe_cfg, mf.weights, c10::Device("cpu"), at::kFloat);
      causal_moe->ready();
      check_prefix_cache(*causal_moe, input(packed, po));
      auto mo = routed->forward(input(packed, po));
      auto mi = at::cat({routed->forward(input(a, ao)), routed->forward(input(b, bo))}, 0);
      TORCH_CHECK(at::allclose(mo, mi, 1e-5, 1e-6),
                  "Gemma4 MoE packed output differs from independent sequences");
      for (int layer = 0; layer < 4; ++layer)
        mf.weights["layers." + std::to_string(layer) + ".experts.down_proj"].select(0, 2).fill_(
            100.);
      TORCH_CHECK(at::allclose(mo, routed->forward(input(packed, po)), 1e-5, 1e-6),
                  "Gemma4 routing tie did not select lower expert IDs");
      for (int layer = 0; layer < 4; ++layer)
        mf.weights["layers." + std::to_string(layer) + ".router.per_expert_scale"] =
            at::tensor({0.f, 1.f, 1.f});
      TORCH_CHECK(at::allclose(mo, routed->forward(input(packed, po)), 1e-5, 1e-6),
                  "Gemma4 cached checkpoint scales changed after external map handle replacement");
      // Learned routing scales are cached by the checkpoint loader, like Candle.
      // Compare a separately loaded changed checkpoint, preserving sensitivity
      // to the same zero-vs-unit scales rather than replacing live map handles.
      auto rescaled = create_gemma(mf.cfg, mf.weights, c10::Device("cpu"), at::kFloat);
      rescaled->ready();
      auto scaled = rescaled->forward(input(packed, po));
      TORCH_CHECK(!at::allclose(mo, scaled, 1e-5, 1e-6),
                  "Gemma4 learned expert output scales were ignored");
      TORCH_CHECK(
          at::allclose(
              scaled,
              at::cat({rescaled->forward(input(a, ao)), rescaled->forward(input(b, bo))}, 0),
              1e-5,
              1e-6),
          "Gemma4 scaled expert outputs leaked across sequences");
      std::cout << "gemma4: native MoE packed isolation and stable routing ties passed\n";
    }
    if (!f.legacy) {
      f.cfg.values["vision_config.hidden_size"] = "8";
      f.cfg.values["vision_config.num_attention_heads"] = "2";
      f.cfg.values["vision_config.num_key_value_heads"] = "1";
      f.cfg.values["vision_config.head_dim"] = "4";
      f.cfg.values["vision_config.num_hidden_layers"] = "1";
      auto put =
          [&](const std::string& name, std::initializer_list<int64_t> shape, bool norm = false) {
            f.weights[name] = norm ? at::ones(shape) : at::randn(shape) * .03;
          };
      put("vision_tower.patch_embedder.input_proj.weight", {8, 768});
      put("vision_tower.patch_embedder.position_embedding_table", {2, 16, 8});
      auto p = std::string("vision_tower.encoder.layers.0.");
      for (const auto& name : {"input_layernorm",
                               "post_attention_layernorm",
                               "pre_feedforward_layernorm",
                               "post_feedforward_layernorm"})
        put(p + name + ".weight", {8}, true);
      put(p + "self_attn.q_proj.linear.weight", {8, 8});
      put(p + "self_attn.k_proj.linear.weight", {4, 8});
      put(p + "self_attn.v_proj.linear.weight", {4, 8});
      put(p + "self_attn.o_proj.linear.weight", {8, 8});
      put(p + "self_attn.q_norm.weight", {4}, true);
      put(p + "self_attn.k_norm.weight", {4}, true);
      put(p + "mlp.gate_proj.linear.weight", {12, 8});
      put(p + "mlp.up_proj.linear.weight", {12, 8});
      put(p + "mlp.down_proj.linear.weight", {8, 12});
      put("embed_vision.embedding_projection.weight", {16, 8});
      auto visual = create_gemma(f.cfg, f.weights, c10::Device("cpu"), at::kFloat);
      visual->ready();
      auto batch = input(packed, po);
      batch.images.push_back({at::rand({9, 768}), {1, 3, 3}, 3, 2, 1, 0});
      auto with_image = visual->forward(batch);
      auto prepared = batch;
      prepared.images[0].has_feature_key = true;
      prepared.images[0].feature_key[0] = 1;
      visual->prepare_input(prepared);
      TORCH_CHECK(at::equal(with_image, visual->forward(prepared)),
                  family,
                  " Feature preparation changed model outputs");
      batched_vision_test(*visual, prepared);
      auto features = prepared.images[0].prepared_features;
      auto moved = prepared;
      moved.images[0].token_start = 1;
      visual->prepare_input(moved);
      TORCH_CHECK(features.const_data_ptr() == moved.images[0].prepared_features.const_data_ptr(),
                  "Token position incorrectly entered feature cache key");
      auto changed = prepared;
      changed.images[0].pixels = 1. - batch.images[0].pixels;
      changed.images[0].feature_key[0] = 2;
      visual->prepare_input(changed);
      TORCH_CHECK(!at::equal(features, changed.images[0].prepared_features),
                  family,
                  " Changed image content reused stale cached features");
      Weights other_weights = f.weights;
      other_weights["embed_vision.embedding_projection.weight"] =
          at::zeros_like(other_weights.at("embed_vision.embedding_projection.weight"));
      auto other_model = create_gemma(f.cfg, other_weights, c10::Device("cpu"), at::kFloat);
      other_model->ready();
      auto other_input = prepared;
      other_model->prepare_input(other_input);
      TORCH_CHECK(!at::equal(features, other_input.images[0].prepared_features),
                  "Feature cache leaked across model instances");

      TORCH_CHECK(!at::allclose(output.narrow(0, 0, 5), with_image.narrow(0, 0, 5)),
                  family,
                  " vision features were ignored");
      TORCH_CHECK(at::allclose(output.narrow(0, 5, 2), with_image.narrow(0, 5, 2), 1e-5, 1e-6),
                  family,
                  " image injection leaked across sequence boundaries");
      std::cout << family << ": natural-resolution vision injection/isolation passed\n";

      if (argc > 1) {
        // A real CPU-host miss must create all image positions/offsets on the
        // model device. Preuploaded pixels or a warm hit would mask that bug.
        const c10::Device gpu(argv[1]);
        auto vc = f.cfg;
        Weights vw = f.weights;
        vc.values["vision_config.hidden_size"] = "128";
        vc.values["vision_config.head_dim"] = "64";
        auto replace =
            [&](const std::string& name, std::initializer_list<int64_t> shape, bool norm = false) {
              vw[name] = norm ? at::ones(shape) : at::randn(shape) * .03;
            };
        replace("vision_tower.patch_embedder.input_proj.weight", {128, 768});
        replace("vision_tower.patch_embedder.position_embedding_table", {2, 16, 128});
        for (const auto& name : {"input_layernorm",
                                 "post_attention_layernorm",
                                 "pre_feedforward_layernorm",
                                 "post_feedforward_layernorm"})
          replace(p + name + ".weight", {128}, true);
        replace(p + "self_attn.q_proj.linear.weight", {128, 128});
        replace(p + "self_attn.k_proj.linear.weight", {64, 128});
        replace(p + "self_attn.v_proj.linear.weight", {64, 128});
        replace(p + "self_attn.o_proj.linear.weight", {128, 128});
        replace(p + "self_attn.q_norm.weight", {64}, true);
        replace(p + "self_attn.k_norm.weight", {64}, true);
        replace(p + "mlp.gate_proj.linear.weight", {12, 128});
        replace(p + "mlp.up_proj.linear.weight", {12, 128});
        replace(p + "mlp.down_proj.linear.weight", {128, 12});
        replace("embed_vision.embedding_projection.weight", {16, 128});
        for (auto& [name, value] : vw)
          value = value.to(gpu, at::kBFloat16);
        auto on_gpu = create_gemma(vc, vw, gpu, at::kBFloat16);
        on_gpu->ready();
        auto host = prepared;
        host.ids = host.ids.to(gpu);
        host.types = host.types.to(gpu);
        host.positions = host.positions.to(gpu);
        host.cumulative = host.cumulative.to(gpu);
        host.images[0].prepared_features = at::Tensor();
        TORCH_CHECK(host.images[0].pixels.device().is_cpu(),
                    "CUDA miss regression already uploaded its pixels");
        on_gpu->prepare_input(host);
        auto cached = host.images[0].prepared_features;
        TORCH_CHECK(cached.device() == gpu, "Host pixel miss prepared features on wrong device");
        batched_vision_test(*on_gpu, host);
        auto uploaded = host;
        uploaded.images[0].pixels = host.images[0].pixels.to(gpu);
        uploaded.images[0].has_feature_key = false;
        uploaded.images[0].prepared_features = at::Tensor();
        on_gpu->prepare_input(uploaded);
        TORCH_CHECK(at::equal(cached, uploaded.images[0].prepared_features),
                    "Host pixel miss changed prepared GPU features");
        auto repeat = host;
        repeat.images[0].prepared_features = at::Tensor();
        on_gpu->prepare_input(repeat);
        TORCH_CHECK(cached.const_data_ptr() == repeat.images[0].prepared_features.const_data_ptr(),
                    "CUDA host image hit recomputed tower");
      }

      Weights bf16_weights;
      for (const auto& [name, value] : f.weights)
        bf16_weights[name] = value.to(at::kBFloat16);
      auto bf16_visual = create_gemma(f.cfg, bf16_weights, c10::Device("cpu"), at::kBFloat16);
      bf16_visual->ready();
      auto precise = bf16_visual->forward(batch);
      auto canonical = batch;
      // Distinct FP32 pixels whose normalized projection inputs are identical
      // after BF16 rounding must produce identical complete model outputs.
      canonical.images[0].pixels =
          ((batch.images[0].pixels - .5) * 2.).to(at::kBFloat16).to(at::kFloat) * .5 + .5;
      TORCH_CHECK(at::equal(precise, bf16_visual->forward(canonical)),
                  family,
                  " vision normalized source pixels after premature activation cast");
      auto rounded = batch;
      rounded.images[0].pixels = batch.images[0].pixels.to(at::kBFloat16).to(at::kFloat);
      TORCH_CHECK(
          !at::equal(precise, bf16_visual->forward(rounded)),
          family,
          " vision pixel precision regression fixture was insensitive to premature BF16 conversion");
      std::cout << family << ": FP32 image normalization before BF16 projection passed\n";
      // Preadding a constant row-position table to every column table is
      // equivalent only when the two position embeddings round before the
      // projected patches are added (the authoritative Candle grouping).
      const std::string table_key = "vision_tower.patch_embedder.position_embedding_table";
      Weights position_weights = bf16_weights;
      position_weights[table_key] = bf16_weights.at(table_key).clone();
      position_weights[table_key].select(0, 1).fill_(1. / 3.);
      Weights grouped_position_weights = position_weights;
      auto table = position_weights.at(table_key);
      grouped_position_weights[table_key] =
          at::stack({table.select(0, 0) + table.select(0, 1).select(0, 0),
                     at::zeros_like(table.select(0, 1))});
      auto original_positions =
          create_gemma(f.cfg, position_weights, c10::Device("cpu"), at::kBFloat16);
      auto preadded_positions =
          create_gemma(f.cfg, grouped_position_weights, c10::Device("cpu"), at::kBFloat16);
      original_positions->ready();
      preadded_positions->ready();
      TORCH_CHECK(
          at::equal(original_positions->forward(batch), preadded_positions->forward(batch)),
          family,
          " vision added position embeddings after projected patches rather than grouping the BF16 position sum first");
      std::cout << family << ": BF16 position-table addition grouping passed\n";
      if (f.embedding2) {
        f.cfg.values["audio_config.hidden_size"] = "8";
        f.cfg.values["audio_config.num_attention_heads"] = "2";
        f.cfg.values["audio_config.num_hidden_layers"] = "1";
        f.cfg.values["audio_config.attention_chunk_size"] = "2";
        f.cfg.values["audio_config.attention_context_left"] = "2";
        put("audio_tower.subsample_conv_projection.layer0.conv.weight", {2, 1, 3, 3});
        put("audio_tower.subsample_conv_projection.layer1.conv.weight", {2, 2, 3, 3});
        put("audio_tower.subsample_conv_projection.layer0.norm.weight", {2}, true);
        put("audio_tower.subsample_conv_projection.layer1.norm.weight", {2}, true);
        put("audio_tower.subsample_conv_projection.input_proj_linear.weight", {8, 64});
        auto p = std::string("audio_tower.layers.0.");
        for (const auto& stem : {"feed_forward1.", "feed_forward2."}) {
          put(p + stem + "pre_layer_norm.weight", {8}, true);
          put(p + stem + "post_layer_norm.weight", {8}, true);
          put(p + stem + "ffw_layer_1.linear.weight", {32, 8});
          put(p + stem + "ffw_layer_2.linear.weight", {8, 32});
        }
        put(p + "norm_pre_attn.weight", {8}, true);
        put(p + "norm_post_attn.weight", {8}, true);
        put(p + "norm_out.weight", {8}, true);
        for (const auto& stem : {"q_proj", "k_proj", "v_proj", "post", "relative_k_proj"})
          put(p + "self_attn." + stem + ".weight", {8, 8});
        put(p + "self_attn.per_dim_scale", {4});
        put(p + "lconv1d.pre_layer_norm.weight", {8}, true);
        put(p + "lconv1d.conv_norm.weight", {8}, true);
        put(p + "lconv1d.linear_start.linear.weight", {16, 8});
        put(p + "lconv1d.linear_end.linear.weight", {8, 8});
        put(p + "lconv1d.depthwise_conv1d.weight", {8, 1, 3});
        put("embed_audio.embedding_projection.weight", {16, 8});
        auto audio = create_gemma(f.cfg, f.weights, c10::Device("cpu"), at::kFloat);
        audio->ready();
        auto audio_batch = input(packed, po);
        audio_batch.audios.push_back(
            {at::randn({8, 128}), at::ones({8}, at::TensorOptions().dtype(at::kByte)), 1, 2});
        auto with_audio = audio->forward(audio_batch);
        auto prepared = audio_batch;
        audio->prepare_input(prepared);
        TORCH_CHECK(at::allclose(with_audio, audio->forward(prepared), 1e-5, 1e-6),
                    "Prepared Gemma audio indices changed output");
        auto masked = audio_batch;
        masked.audios[0].validity = masked.audios[0].validity.clone();
        masked.audios[0].validity.select(0, 0).zero_();
        masked.audios[0].token_count = 1;
        auto masked_reference = audio->forward(masked);
        audio->prepare_input(masked);
        TORCH_CHECK(masked.audios[0].selected_frames.numel() == 1 &&
                        masked.audios[0].selected_frames[0].item<int64_t>() == 1,
                    "Gemma prepared audio indices did not preserve exact SSCP validity");
        TORCH_CHECK(at::allclose(masked_reference, audio->forward(masked), 1e-5, 1e-6),
                    "Prepared masked Gemma audio changed selected feature");
        TORCH_CHECK(!at::allclose(output.narrow(0, 0, 5), with_audio.narrow(0, 0, 5)),
                    "Gemma audio features were ignored");
        TORCH_CHECK(at::allclose(output.narrow(0, 5, 2), with_audio.narrow(0, 5, 2), 1e-5, 1e-6),
                    "Gemma audio injection leaked across sequences");
        std::cout << family << ": actual-length Conformer audio injection/isolation passed\n";
      }
    }
  }
  if (argc > 1) {
    const c10::Device device(argv[1]);
    for (const auto& family : {"gemma3", "gemma4", "embedding_gemma2"}) {
      const int64_t head_dim = std::string(family) == "gemma3" ? 256 : 512;
      Fixture f(family, head_dim);
      f.cfg.values["num_hidden_layers"] = "1";
      f.cfg.values["num_kv_shared_layers"] = "0";
      f.cfg.values["hidden_size_per_layer_input"] = "0";
      if (f.legacy) {
        // Retain the original unit Q/K scales separately: the hidden width16
        // norm requires a full warp, and packed requests must remain isolated.
        Weights original_weights;
        for (const auto& [key, value] : f.weights)
          original_weights[key] = value.to(device, at::kBFloat16);
        auto original = create_gemma(f.cfg, original_weights, device, at::kBFloat16);
        original->ready();
        auto on_device = [&](PackedInput in) {
          in.ids = in.ids.to(device);
          in.types = in.types.to(device);
          in.positions = in.positions.to(device);
          in.cumulative = in.cumulative.to(device);
          return in;
        };
        for (int64_t length : {5, 129}) {
          std::vector<int64_t> first;
          for (int64_t i = 0; i < length; ++i)
            first.push_back(i % 63 + 1);
          const std::vector<int64_t> second = {17, 31};
          auto ids = first;
          ids.insert(ids.end(), second.begin(), second.end());
          const std::vector<int32_t> offsets = {0, int32_t(length), int32_t(length + 2)},
                                     one = {0, int32_t(length)}, two = {0, 2};
          auto packed = original->forward(on_device(input(ids, offsets)));
          auto separate = at::cat({original->forward(on_device(input(first, one))),
                                   original->forward(on_device(input(second, two)))},
                                  0);
          TORCH_CHECK(at::isfinite(packed).all().item<bool>(),
                      "Gemma3 unit-scale CUDA normalization produced nonfinite output");
          // Unit final RMSNorm bounds each component by sqrt(hidden_size),
          // allowing one BF16 rounding step. This catches invalid warp sums.
          TORCH_CHECK(packed.abs().max().item<float>() <= 4.03125f,
                      "Gemma3 unit-scale CUDA RMSNorm violated its component bound");
          TORCH_CHECK(at::allclose(packed, separate, .04, .04),
                      "Gemma3 unit-scale packed CUDA sequence isolation failed");
        }
        std::cout
            << "gemma3: original unit Q/K scales, tiny-width CUDA norm and packed isolation passed\n";
      }
      // Keep the synthetic attention well conditioned: untrained unit Q/K
      // norms at scale1 otherwise amplify BF16 reduction differences across4
      // random layers into unstable argmax decisions.
      for (auto& [key, value] : f.weights) {
        if (key.ends_with("q_norm.weight") || key.ends_with("k_norm.weight"))
          value = f.legacy ? at::full_like(value, -.95) : value * .05;
      }
      for (auto dtype : {at::kHalf, at::kBFloat16}) {
        Weights cpu_weights;
        for (const auto& [key, value] : f.weights)
          cpu_weights[key] = value.to(dtype);
        auto cpu = create_gemma(f.cfg, cpu_weights, c10::Device("cpu"), dtype);
        cpu->ready();
        Weights gpu_weights;
        for (const auto& [key, value] : cpu_weights)
          gpu_weights[key] = value.to(device);
        auto gpu = create_gemma(f.cfg, gpu_weights, device, dtype);
        gpu->ready();
        for (int64_t length : {5, 129}) {
          std::vector<int64_t> ids;
          for (int64_t i = 0; i < length + 2; ++i)
            ids.push_back(i % 63 + 1);
          const std::vector<int32_t> offsets = {
              0, static_cast<int32_t>(length), static_cast<int32_t>(length + 2)};
          auto host = input(ids, offsets), dev = host;
          dev.ids = host.ids.to(device);
          dev.types = host.types.to(device);
          dev.positions = host.positions.to(device);
          dev.cumulative = host.cumulative.to(device);
          auto reference = cpu->forward(host).to(at::kFloat),
               actual = gpu->forward(dev).cpu().to(at::kFloat);
          auto last = gpu->forward_pooled(dev, true).cpu().to(at::kFloat);
          TORCH_CHECK(
              at::allclose(last,
                           actual.index_select(0, at::tensor({length - 1, length + 1}, at::kLong)),
                           dtype == at::kHalf ? .005 : .04,
                           dtype == at::kHalf ? .005 : .04),
              family,
              " selected last-token rows changed");
          check_prefix_cache(*gpu, dev, 0, .01, .02);
          if (std::string(family) == "gemma4") {
            auto causal_cfg = f.cfg;
            causal_cfg.values["_decision.kind"] = "option_tokens";
            auto causal = create_gemma(causal_cfg, gpu_weights, device, dtype);
            causal->ready();
            check_prefix_cache(*causal, dev, 0, .01, .02);
          }
          TORCH_CHECK(at::allclose(reference,
                                   actual,
                                   dtype == at::kHalf ? .005 : .04,
                                   dtype == at::kHalf ? .005 : .04),
                      family,
                      " Torch varlen-vs-CPU mismatch max_length",
                      length,
                      " maxabs=",
                      (reference - actual).abs().max().item<float>());
          std::cout << family << " " << at::toString(dtype) << " Torch" << head_dim
                    << " GPU/CPU parity max_length=" << length
                    << " maxabs=" << (reference - actual).abs().max().item<float>() << "\n";
        }
      }
    }
  }
}
