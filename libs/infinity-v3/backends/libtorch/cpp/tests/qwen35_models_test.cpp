#include "env_config.h"
#include "prefix_cache_test.h"
#include "bridge.h"
#include "models/qwen35_models.h"
#include "models/decision_models.h"
#include "batching/radix.h"
#include "helpers/workspace_admission.h"
#include "batching/cuda_graph.h"
#include <ATen/ops/_flash_attention_forward.h>
#ifdef TEI_TORCH_CUDA_KERNELS
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#endif
#include <iostream>
using namespace infinity;

void dynamic_fp8_test(Options cfg, Weights weights, const c10::Device& device) {
#ifdef TEI_TORCH_CUDA_KERNELS
  const auto* properties = at::cuda::getDeviceProperties(device.index());
  const auto capability = properties->major * 10 + properties->minor;
  if (capability != 89 && capability != 90 && capability != 100 && capability != 120) {
    std::cout << "Qwen3.5 FP8 fixture skipped: requires SM89/90/100/120\n";
    return;
  }
#endif
  // Aligned projection widths; ragged token rows deliberately remain unaligned.
  cfg.values["head_dim"] = "16";
  weights["model.layers.1.self_attn.q_proj.weight"] = at::randn({64, 16}) * .05;
  for (auto name : {"k_proj", "v_proj"})
    weights[std::string("model.layers.1.self_attn.") + name + ".weight"] =
        at::randn({16, 16}) * .05;
  weights["model.layers.1.self_attn.o_proj.weight"] = at::randn({16, 32}) * .05;
  for (auto name : {"q_norm", "k_norm"})
    weights[std::string("model.layers.1.self_attn.") + name + ".weight"] = at::zeros({16});
  for (int i = 0; i < 2; ++i) {
    auto p = "model.layers." + std::to_string(i) + ".mlp.";
    for (auto name : {"gate_proj", "up_proj"})
      weights[p + name + ".weight"] = at::randn({32, 16}) * .05;
    weights[p + "down_proj.weight"] = at::randn({16, 32}) * .05;
  }
  for (auto& [name, value] : weights)
    value = value.to(device, at::kBFloat16);
  auto build = [&](const Options& options) {
    Weights loader;
    for (const auto& [name, value] : weights)
      loader[name] = value.clone();
    auto model = create_qwen35(options, loader, device, at::kBFloat16);
    model->ready();
    return model;
  };
  const std::vector<int32_t> ends{0, 5, 9};
  auto ids =
      at::tensor({1, 2, 3, 4, 5, 1, 2, 7, 8}, at::TensorOptions().device(device).dtype(at::kLong));
  auto positions = at::tensor({0, 1, 2, 3, 4, 0, 1, 2, 3}, ids.options());
  PackedInput input{ids,
                    at::zeros_like(ids),
                    positions,
                    at::tensor(ends, ids.options().dtype(at::kInt)),
                    ends.data(),
                    2,
                    5};
  auto close = [](const at::Tensor& expected, const at::Tensor& actual, double tolerance) {
    auto ref = expected.to(at::kFloat), got = actual.to(at::kFloat);
    TORCH_CHECK(at::isfinite(got).all().item<bool>(), "Qwen3.5 FP8 returned nonfinite outputs");
    auto error = ((got - ref).norm() / ref.norm()).item<double>();
    TORCH_CHECK(error < tolerance, "Qwen3.5 FP8 relative L2 error=", error);
  };
  for (const auto* family : {"qwen3_5", "qwen3_5_text"})
    for (const auto* scope : {"mlp", "all", "default"}) {
      cfg.values["model_type"] = family;
      auto enabled = cfg;
      enabled.values["_fp8_dynamic"] = "true";
      if (std::string(scope) != "default")
        enabled.values["_fp8_scope"] = scope;
      // Exercise the serving bridge's family gate, not only model construction.
      TeiConfig shape{16, 2, 2, 32, 32, 32, 1, 1e-6, 5};
      std::unique_ptr<void, decltype(&tei_destroy)> engine(
          tei_create(&shape, device.str().c_str(), 2), tei_destroy);
      TORCH_CHECK(engine, tei_error());
      for (const auto& [name, value] : enabled.values)
        TORCH_CHECK(tei_option(engine.get(), name.c_str(), value.c_str()) == 0, tei_error());
      for (const auto& [name, value] : weights) {
        auto host = value.cpu().contiguous();
        TORCH_CHECK(tei_weight(engine.get(),
                               name.c_str(),
                               static_cast<const uint8_t*>(host.const_data_ptr()),
                               host.sizes().data(),
                               host.dim(),
                               2) == 0,
                    tei_error());
      }
      TORCH_CHECK(tei_ready(engine.get()) == 0, tei_error());
      engine.reset();
      Weights loader;
      for (const auto& [name, value] : weights)
        loader[name] = value.clone();
      auto down = loader.at("model.layers.0.mlp.down_proj.weight").storage().getWeakStorageImpl();
      auto projection =
          loader.at("model.layers.0.linear_attn.out_proj.weight").storage().getWeakStorageImpl();
      auto quantized = create_qwen35(enabled, loader, device, at::kBFloat16);
      quantized->ready();
      TORCH_CHECK(loader.empty() && down.expired(), "Qwen3.5 retained consumed MLP weights");
      TORCH_CHECK(projection.expired() == (std::string(scope) != "mlp"),
                  "Qwen3.5 FP8 projection scope ignored");
      auto reference = build(cfg);
      auto actual = quantized->forward(input);
      close(reference->forward(input), actual, .08);
      // Alter one suffix; its causal prefix and the other sequence must survive.
      auto changed = input;
      changed.ids = input.ids.clone();
      changed.ids.index_put_({4}, 31);
      auto future = quantized->forward(changed);
      TORCH_CHECK(at::equal(actual.narrow(0, 0, 4), future.narrow(0, 0, 4)) &&
                      at::equal(actual.narrow(0, 5, 4), future.narrow(0, 5, 4)),
                  "Qwen3.5 FP8 causal/sequence isolation failed");
      auto folded = input;
      folded.compact_ids = ids.narrow(0, 0, 5);
      folded.compact_ids = at::cat({folded.compact_ids, ids.narrow(0, 7, 2)});
      folded.compact_positions = at::tensor({0, 1, 2, 3, 4, 2, 3}, ids.options());
      folded.compact_types = at::zeros_like(folded.compact_ids);
      folded.scatter_unfold = at::tensor({0, 1, 2, 3, 4, 0, 1, 5, 6}, ids.options());
      folded.fold_gather = at::tensor({0, 1, 2, 3, 4, 7, 8}, ids.options());
      close(actual, quantized->forward(folded), .02);
      check_prefix_cache(*quantized, input, 0, .02, .02);
      CudaGraphCache graphs(1, 4096, 1);
      for (int repeat = 0; repeat < 3; ++repeat)
        close(actual, graphs.forward(*quantized, input), .02);
      close(future, graphs.forward(*quantized, changed), .02);
      TORCH_CHECK(graphs.size() == 1, "Qwen3.5 FP8 did not capture a CUDA graph");
      enabled.values["_decision_attention_mode"] = "block_bidirectional";
      auto block = build(enabled);
      auto packed = input;
      packed.block_prefix_lengths = {2, 2};
      packed.block_prefix_sources = {0, 0};
      auto shared = block->forward(packed);
      packed.block_prefix_sources = {0, 1};
      auto independent = block->forward(packed);
      close(shared, independent, .02);
      changed = packed;
      changed.ids = input.ids.clone();
      changed.ids.index_put_({4}, 31);
      auto block_future = block->forward(changed);
      TORCH_CHECK(at::equal(independent.narrow(0, 0, 2), block_future.narrow(0, 0, 2)) &&
                      at::equal(independent.narrow(0, 5, 4), block_future.narrow(0, 5, 4)),
                  "Qwen3.5 FP8 block prefix or sibling isolation failed");
      std::cout << family << " FP8 " << scope
                << " drift, ownership, radix, prefix cache, graphs and block isolation passed\n";
    }
  for (const auto* invalid : {"cpu", "float16", "moe", "quantized", "qkv_shape"}) {
    auto enabled = cfg;
    enabled.values["_fp8_dynamic"] = "true";
    if (std::string(invalid) == "moe")
      enabled.values["num_experts"] = "2";
    if (std::string(invalid) == "quantized")
      enabled.values["text_config.quantization_config.quant_method"] = "fp8";
    auto loader = weights;
    if (std::string(invalid) == "qkv_shape")
      loader["model.layers.1.self_attn.q_proj.weight"] =
          at::zeros({80, 16}, weights.at("model.layers.1.self_attn.q_proj.weight").options());
    bool rejected = false;
    try {
      auto model = create_qwen35(enabled,
                                 loader,
                                 std::string(invalid) == "cpu" ? c10::Device("cpu") : device,
                                 std::string(invalid) == "float16" ? at::kHalf : at::kBFloat16);
      model->ready();
      if (std::string(invalid) == "qkv_shape")
        (void)model->forward(input);
    } catch (const c10::Error&) {
      rejected = true;
    }
    TORCH_CHECK(rejected, "Qwen3.5 FP8 accepted unsupported ", invalid);
  }
}

void radix_workspace_recovery_test(const c10::Device& device) {
#ifdef TEI_TORCH_CUDA_KERNELS
  const auto* properties = at::cuda::getDeviceProperties(device.index());
  if (properties->major != 9 || properties->minor != 0) {
    std::cout << "Qwen3.5 shared workspace fixture skipped: production path requires SM 9.0\n";
    return;
  }

  struct RestoreBlasBackend {
    at::BlasBackend prior = at::globalContext().blasPreferredBackend();

    ~RestoreBlasBackend() {
      at::globalContext().setBlasPreferredBackend(prior);
    }
  } restore;

  at::globalContext().setBlasPreferredBackend(at::BlasBackend::Cublaslt);
#else
  return;
#endif
  Options cfg;
  cfg.values = {{"model_type", "qwen3_5_moe"},
                {"hidden_size", "2048"},
                {"num_attention_heads", "32"},
                {"num_key_value_heads", "2"},
                {"head_dim", "128"},
                {"num_hidden_layers", "1"},
                {"max_position_embeddings", "1024"},
                {"rope_parameters.partial_rotary_factor", ".5"},
                {"layer_types.0", "full_attention"},
                {"num_experts", "128"},
                {"num_experts_per_tok", "8"}};
  auto options = at::TensorOptions().device(device).dtype(at::kBFloat16);
  Weights checkpoint;
  auto put =
      [&](const std::string& name, std::initializer_list<int64_t> dimensions, bool norm = false) {
        checkpoint["model." + name] =
            norm ? at::ones(dimensions, options)
                 : (at::randn(dimensions, options.dtype(at::kFloat)) * .003).to(at::kBFloat16);
      };
  put("embed_tokens.weight", {64, 2048});
  put("norm.weight", {2048}, true);
  const std::string layer = "layers.0.";
  put(layer + "input_layernorm.weight", {2048}, true);
  put(layer + "post_attention_layernorm.weight", {2048}, true);
  put(layer + "self_attn.q_proj.weight", {8192, 2048});
  put(layer + "self_attn.k_proj.weight", {256, 2048});
  put(layer + "self_attn.v_proj.weight", {256, 2048});
  put(layer + "self_attn.o_proj.weight", {2048, 4096});
  put(layer + "self_attn.q_norm.weight", {128}, true);
  put(layer + "self_attn.k_norm.weight", {128}, true);
  put(layer + "mlp.gate.weight", {128, 2048});
  put(layer + "mlp.experts.gate_up_proj", {128, 128, 2048});
  put(layer + "mlp.experts.down_proj", {128, 2048, 64});
  put(layer + "mlp.shared_expert.gate_proj.weight", {12, 2048});
  put(layer + "mlp.shared_expert.up_proj.weight", {12, 2048});
  put(layer + "mlp.shared_expert.down_proj.weight", {2048, 12});
  put(layer + "mlp.shared_expert_gate.weight", {1, 2048});
  auto build = [&] {
    auto weights = checkpoint;
    auto model = create_qwen35(cfg, weights, device, at::kBFloat16);
    model->ready();
    return model;
  };
  auto state = workspace_admission_state(device.index());
  auto retained = [&] {
    std::lock_guard lock(state->mutex);
    return state->retained;
  };
  const auto baseline = retained();
  bool exercised = false;
  for (int32_t length : {32, 128, 512}) {
    std::array<int32_t, 2> offsets{0, length};
    auto ids = at::arange(length, at::kLong).remainder(64);
    auto positions = at::arange(length, at::kLong);
    auto types = at::zeros_like(ids);
    auto cumulative = at::tensor({0, length}, at::kInt);
    PackedInput input{ids, types, positions, cumulative, offsets.data(), 1, length};
    input.compact_ids = ids;
    input.compact_positions = positions;
    input.compact_types = types;
    input.scatter_unfold = positions;
    input.fold_gather = positions;
    validate_radix(input, true);
    for (auto* tensor : {&input.ids,
                         &input.types,
                         &input.positions,
                         &input.cumulative,
                         &input.compact_ids,
                         &input.compact_positions,
                         &input.compact_types,
                         &input.scatter_unfold,
                         &input.fold_gather})
      *tensor = tensor->to(device);
    at::Tensor reference;
    {
      auto model = build();
      reference = model->forward(input).cpu();
      exercised = retained() > baseline;
    }
    TORCH_CHECK(retained() == baseline, "Qwen3.5 model destruction retained workspace credit");
    if (!exercised)
      continue;
    size_t available;
    {
      std::lock_guard lock(state->mutex);
      available = (state->limit ? state->limit : size_t(512) << 20) - state->retained;
    }
    auto pressure = reserve_workspace_memory(device.index(), available);
    TORCH_CHECK(pressure, "Unable to hold Qwen3.5 workspace credit");
    auto model = build();
    auto fallback = model->forward(input).cpu();
    TORCH_CHECK(at::isfinite(fallback).all().item<bool>() && retained() == baseline + available,
                "Qwen3.5 fallback allocated an uncredited custom workspace");
    pressure.reset();
    auto recovered = model->forward(input).cpu();
    TORCH_CHECK(retained() > baseline,
                "Released credit did not retry the cached Qwen3.5 supported plan");
    TORCH_CHECK(at::equal(reference, recovered) &&
                    at::equal(recovered, model->forward(input).cpu()),
                "Qwen3.5 workspace recovery changed full supported-plan output coordinates");
    {
      CudaGraphCache graphs(1, 4096, 1);
      for (int repeat = 0; repeat < 2; ++repeat)
        TORCH_CHECK(at::equal(graphs.forward(*model, input).cpu(), reference),
                    "Qwen3.5 captured workspace generation changed full outputs");
      TORCH_CHECK(graphs.size() == 1, "Qwen3.5 workspace recovery control did not capture");
    }
    model.reset();
    TORCH_CHECK(retained() == baseline, "Qwen3.5 graph/model teardown retained workspace credit");
    std::cout << "Qwen3.5 shared workspace fallback/release/full recovery tokens=" << length
              << " passed\n";
    break;
  }
  TORCH_CHECK(exercised,
              "Qwen3.5 workspace regression did not exercise a workspace-requiring supported plan");
}

void paged_block_test(const c10::Device& device) {
#ifdef TEI_TORCH_CUDA_KERNELS
  const std::vector<int32_t> prefixes{0, 1025, 1537, 1792}, suffixes{0, 3, 260, 261, 266, 283, 284};
  const std::vector<int64_t> sources{0, 1, 0, 2, 1, 0};
  std::vector<int32_t> kv{0};
  std::vector<int64_t> expanded;
  for (size_t row = 0; row < sources.size(); ++row) {
    for (int64_t i = prefixes[sources[row]]; i < prefixes[sources[row] + 1]; ++i)
      expanded.push_back(i);
    for (int64_t i = suffixes[row]; i < suffixes[row + 1]; ++i)
      expanded.push_back(prefixes.back() + i);
    kv.push_back(expanded.size());
  }
  auto integer = at::TensorOptions().device(device).dtype(at::kInt);
  auto pages = prepare_block_pages(prefixes, suffixes, sources, integer);
  TORCH_CHECK(pages.gather.defined() && pages.gather.numel() < kv.back(),
              "Paged block fixture did not share prefix storage");
  auto cuq = at::tensor(suffixes, integer), cuk = at::tensor(kv, integer);
  auto rows = at::tensor(expanded, integer.dtype(at::kLong));
  for (auto dtype : {at::kHalf, at::kBFloat16}) {
    for (auto [heads, dim] :
         std::vector<std::pair<int64_t, int64_t>>{{16, 256}, {24, 256}, {16, 64}, {24, 128}}) {
      auto options = integer.dtype(dtype);
      auto q = at::randn({suffixes.back(), heads, dim}, options);
      auto k = at::randn({prefixes.back() + suffixes.back(), 4, dim}, options);
      auto v = at::randn_like(k);
      const auto scale = 1. / std::sqrt(double(dim));
      auto run = [&] {
        return block_paged_attention(q,
                                     k.narrow(0, prefixes.back(), suffixes.back()),
                                     v.narrow(0, prefixes.back(), suffixes.back()),
                                     k.narrow(0, 0, prefixes.back()),
                                     v.narrow(0, 0, prefixes.back()),
                                     cuq,
                                     cuk,
                                     pages,
                                     257,
                                     scale);
      };
      auto reference = [&] {
        return std::get<0>(at::_flash_attention_forward(q,
                                                        k.index_select(0, rows),
                                                        v.index_select(0, rows),
                                                        cuq,
                                                        cuk,
                                                        257,
                                                        pages.max_k,
                                                        0.,
                                                        false,
                                                        false,
                                                        scale));
      };
      TORCH_CHECK(at::equal(reference(), run()),
                  "Paged prefix/tail/suffix attention differs from contiguous attention");
      auto stream = at::cuda::getStreamFromPool(false, device.index());
      at::cuda::CUDAStreamGuard guard(stream);
      c10::cuda::device_synchronize();
      at::Tensor captured;
      at::cuda::CUDAGraph graph;
      run();
      stream.synchronize();
      graph.capture_begin();
      captured = run();
      graph.capture_end();
      q.mul_(2.);
      k.add_(.125);
      v.mul_(.5);
      graph.replay();
      TORCH_CHECK(at::equal(reference(), captured), "Paged block graph replay used stale rows");
      std::cout << "Paged block exact parity and graph replay passed heads=" << heads
                << " dim=" << dim << " dtype=" << dtype << "\n";
    }
  }
#endif
}

void paged_model_test(Options cfg, Weights weights, const c10::Device& device) {
  cfg.values["head_dim"] = "64";
  cfg.values["max_position_embeddings"] = "1024";
  cfg.values["_decision_attention_mode"] = "block_bidirectional";
  const std::string attention = "model.layers.1.self_attn.";
  weights[attention + "q_proj.weight"] = at::randn({256, 16}) * .05;
  for (auto name : {"k_proj", "v_proj"})
    weights[attention + name + ".weight"] = at::randn({64, 16}) * .05;
  weights[attention + "o_proj.weight"] = at::randn({16, 128}) * .05;
  for (auto name : {"q_norm", "k_norm"})
    weights[attention + name + ".weight"] = at::zeros({64});
  auto build = [&] {
    Weights loader;
    for (const auto& [name, value] : weights)
      loader[name] = value.to(device, at::kBFloat16).clone();
    auto model = create_qwen35(cfg, loader, device, at::kBFloat16);
    model->ready();
    return model;
  };
  unsetenv(INFINITY_ENV("TORCH_BLOCK_PAGED"));
  unsetenv(INFINITY_ENV("TORCH_GDN_SUFFIX_PREFILL"));
  auto paged = build();
  setenv(INFINITY_ENV("TORCH_GDN_SUFFIX_PREFILL"), "1", 1);
  auto explicit_prefill = build();
  unsetenv(INFINITY_ENV("TORCH_GDN_SUFFIX_PREFILL"));
  setenv(INFINITY_ENV("TORCH_BLOCK_PAGED"), "0", 1);
  auto packed = build();
  unsetenv(INFINITY_ENV("TORCH_BLOCK_PAGED"));

  // Three ragged suffixes share two full pages and an incomplete prefix tail.
  const std::vector<int32_t> ends{0, 609, 1219, 1830};
  auto options = at::TensorOptions().device(device).dtype(at::kLong);
  auto positions =
      at::cat({at::arange(609, options), at::arange(610, options), at::arange(611, options)});
  auto ids = positions.remainder(32);
  PackedInput input{ids,
                    at::zeros_like(ids),
                    positions,
                    at::tensor(ends, options.dtype(at::kInt)),
                    ends.data(),
                    3,
                    611};
  input.block_prefix_lengths = {513, 513, 513};
  input.block_prefix_sources = {0, 0, 0};
  auto prepared = input, disabled = input;
  paged->prepare_input(prepared);
  packed->prepare_input(disabled);
  TORCH_CHECK(prepared.block_layout->pages.gather.defined() &&
                  !disabled.block_layout->pages.gather.defined(),
              "Default paging or its disable override did not select the expected model path");
  TORCH_CHECK(at::allclose(packed->forward(input), paged->forward(input), 1e-2, 1e-2),
              "Model paging changed ragged shared-prefix outputs");
  TORCH_CHECK(at::equal(paged->forward(input), explicit_prefill->forward(input)),
              "Default cached suffix prefill differs from explicit opt-in");
  CudaGraphCache graphs(2, 4096, 1);
  for (int operation : {0, 3})
    for (int iteration = 0; iteration < 3; ++iteration) {
      auto replay = input;
      replay.ids = (input.ids + iteration).remainder(32);
      auto expected = packed->forward(replay);
      if (operation == 3)
        expected = expected.index_select(0, pooled_token_rows(replay, true));
      TORCH_CHECK(at::allclose(expected, graphs.forward(*paged, replay, operation), 1e-2, 1e-2),
                  "Paged model graph replay changed full or pooled outputs");
    }
  TORCH_CHECK(graphs.stats().captures == 2 && graphs.stats().hits >= 4,
              "Paged model did not capture and replay full and pooled graphs");
  std::cout << "Qwen3.5 default/disabled model paging and full/pooled graph replay passed\n";
}

int main(int argc, char** argv) {
  std::vector<int32_t> fanout_suffixes{0};
  for (int i = 0; i < 50; ++i)
    fanout_suffixes.push_back(fanout_suffixes.back() + 185);
  auto pages = prepare_block_pages({0, 13045},
                                   fanout_suffixes,
                                   std::vector<int64_t>(50, 0),
                                   at::TensorOptions().dtype(at::kLong));
  TORCH_CHECK(pages.gather.numel() < 50000,
              "13K/50-question paged storage expanded the shared prefix");
  TORCH_CHECK(!prepare_block_pages({0, 2}, {0, 3}, {0}, at::TensorOptions()).gather.defined(),
              "Small unshared block should use packed attention");
  if (argc > 1)
    paged_block_test(c10::Device(argv[1]));
  at::manual_seed(1451);
  Options cfg;
  cfg.values = {{"model_type", "qwen3_5"},
                {"hidden_size", "16"},
                {"num_attention_heads", "2"},
                {"num_key_value_heads", "1"},
                {"head_dim", "8"},
                {"num_hidden_layers", "2"},
                {"max_position_embeddings", "32"},
                {"rope_parameters.partial_rotary_factor", ".5"},
                {"linear_num_key_heads", "1"},
                {"linear_num_value_heads", "1"},
                {"layer_types.0", "linear_attention"},
                {"layer_types.1", "full_attention"}};
  Weights weights;
  auto put = [&](const std::string& name, std::initializer_list<int64_t> shape, bool zero = false) {
    weights[name] = zero ? at::zeros(shape) : at::randn(shape) * .05;
  };
  put("model.embed_tokens.weight", {32, 16});
  put("model.norm.weight", {16}, true);
  for (int i = 0; i < 2; ++i) {
    auto p = "model.layers." + std::to_string(i) + ".";
    put(p + "input_layernorm.weight", {16}, true);
    put(p + "post_attention_layernorm.weight", {16}, true);
    put(p + "mlp.gate_proj.weight", {24, 16});
    put(p + "mlp.up_proj.weight", {24, 16});
    put(p + "mlp.down_proj.weight", {16, 24});
    if (i == 0) {
      p += "linear_attn.";
      put(p + "in_proj_qkv.weight", {384, 16});
      put(p + "in_proj_z.weight", {128, 16});
      put(p + "in_proj_a.weight", {1, 16});
      put(p + "in_proj_b.weight", {1, 16});
      put(p + "conv1d.weight", {384, 1, 4});
      put(p + "A_log", {1});
      put(p + "dt_bias", {1});
      weights[p + "norm.weight"] = at::ones({128});
      put(p + "out_proj.weight", {16, 128});
    } else {
      p += "self_attn.";
      put(p + "q_proj.weight", {32, 16});
      put(p + "k_proj.weight", {8, 16});
      put(p + "v_proj.weight", {8, 16});
      put(p + "o_proj.weight", {16, 16});
      put(p + "q_norm.weight", {8}, true);
      put(p + "k_norm.weight", {8}, true);
    }
  }
  auto build =
      [](const Options& options, const Weights& source, c10::Device device, at::ScalarType dtype) {
        auto loader = source;
        return create_qwen35(options, loader, device, dtype);
      };
  auto model = build(cfg, weights, c10::Device("cpu"), at::kFloat);
  model->ready();
  auto full_cfg = cfg;
  full_cfg.values["layer_types.0"] = "full_attention";
  auto full_weights = weights;
  for (const auto& [name, value] : weights) {
    const std::string from = "model.layers.1.self_attn.";
    if (name.rfind(from, 0) == 0)
      full_weights["model.layers.0.self_attn." + name.substr(from.size())] = value.clone();
  }
  auto full_model = build(full_cfg, full_weights, c10::Device("cpu"), at::kFloat);
  full_model->ready();
  std::vector<int32_t> offsets = {0, 3, 5}, one = {0, 3}, two = {0, 2};
  auto make = [](const std::vector<int64_t>& ids, const std::vector<int32_t>& ends) {
    auto tokens = at::from_blob(const_cast<int64_t*>(ids.data()),
                                {static_cast<int64_t>(ids.size())},
                                at::TensorOptions().dtype(at::kLong))
                      .clone();
    std::vector<int64_t> positions;
    int64_t max = 0;
    for (size_t i = 1; i < ends.size(); ++i) {
      auto n = ends[i] - ends[i - 1];
      max = std::max<int64_t>(max, n);
      for (int j = 0; j < n; ++j)
        positions.push_back(j);
    }
    auto pos = at::from_blob(positions.data(),
                             {static_cast<int64_t>(positions.size())},
                             at::TensorOptions().dtype(at::kLong))
                   .clone();
    auto cu = at::from_blob(const_cast<int32_t*>(ends.data()),
                            {static_cast<int64_t>(ends.size())},
                            at::TensorOptions().dtype(at::kInt))
                  .clone();
    return PackedInput{tokens,
                       at::zeros_like(tokens),
                       pos,
                       cu,
                       ends.data(),
                       static_cast<int64_t>(ends.size() - 1),
                       max};
  };
  TORCH_CHECK(at::isfinite(full_model->forward(make({1, 2, 3, 4, 5}, offsets))).all().item<bool>(),
              "Full-attention Qwen3.5 profile must preserve valid packed inference");
  auto output = model->forward(make({1, 2, 3, 4, 5}, offsets));
  TORCH_CHECK(at::equal(model->forward_pooled(make({1, 2, 3, 4, 5}, offsets), true),
                        output.index_select(0, at::tensor({2, 4}, at::kLong))),
              "Qwen3.5 last-token readout omitted a packed sequence");
  check_prefix_cache(*model, make({1, 2, 3, 4, 5}, offsets));
  auto independent =
      at::cat({model->forward(make({1, 2, 3}, one)), model->forward(make({4, 5}, two))}, 0);
  Weights owned_loader;
  for (const auto& [name, value] : weights)
    owned_loader[name] = value.clone();
  c10::weak_intrusive_ptr<c10::TensorImpl> original_gate(
      owned_loader.at("model.layers.0.mlp.gate_proj.weight").getIntrusivePtr());
  auto tied_cfg = cfg;
  tied_cfg.values["tie_word_embeddings"] = "true";
  auto tied = create_decision_head("option_tokens", {}, tied_cfg, {}, owned_loader);
  auto owned = create_qwen35(cfg, owned_loader, c10::Device("cpu"), at::kFloat);
  TORCH_CHECK(owned_loader.empty() && !original_gate.expired(),
              "Qwen3.5 must own loaded tensors before initialization");
  owned->ready();
  TORCH_CHECK(original_gate.expired(),
              "Consumed Qwen3.5 FFN projection still retained after initialization");
  auto owned_hidden = owned->forward(make({1, 2, 3, 4, 5}, offsets));
  TORCH_CHECK(at::equal(output, owned_hidden),
              "Qwen3.5 ownership/cache pruning changed full outputs");
  DecisionRequest options{DecisionRequest::Kind::OptionTokens};
  options.token_ids = {1, 2};
  auto baseline_head = create_decision_head("option_tokens", {}, tied_cfg, {}, weights);
  auto before = baseline_head->forward(output, make({1, 2, 3, 4, 5}, offsets), {options, options});
  auto after = tied->forward(owned_hidden, make({1, 2, 3, 4, 5}, offsets), {options, options});
  TORCH_CHECK(at::equal(before[0].logits, after[0].logits) &&
                  at::equal(before[1].logits, after[1].logits),
              "Tied option head lost owned embeddings");
  std::cout << "Qwen3.5 loader ownership, consumed FFN lifetime and tied decision outputs passed\n";

  TORCH_CHECK(at::allclose(output, independent, 1e-5, 1e-6),
              "DeltaNet packed recurrence leaked sequence state");
  auto changed = model->forward(make({31, 2, 3, 4, 5}, offsets));
  TORCH_CHECK(at::allclose(output.narrow(0, 3, 2), changed.narrow(0, 3, 2), 1e-5, 1e-6),
              "DeltaNet state not reset at sequence boundary");
  std::cout << "Qwen3.5 packed/single parity and recurrence reset passed\n";
  auto shared = make({1, 2, 3, 1, 2}, offsets), folded = shared;
  folded.compact_ids = at::tensor({1, 2, 3}, at::kLong);
  folded.compact_positions = at::tensor({0, 1, 2}, at::kLong);
  folded.compact_types = at::zeros({3}, at::kLong);
  folded.scatter_unfold = at::tensor({0, 1, 2, 0, 1}, at::kLong);
  folded.fold_gather = at::tensor({0, 1, 2}, at::kLong);
  validate_radix(folded, true);
  TORCH_CHECK(model->supports_radix(), "Causal Qwen3.5 folding capability missing");
  auto shared_output = model->forward(shared);
  TORCH_CHECK(at::allclose(shared_output, model->forward(folded), 1e-5, 1e-6),
              "Qwen3.5 folded Delta/full attention lost logical rows");
  auto reordered = folded;
  reordered.compact_ids = folded.compact_ids.flip({0});
  reordered.compact_positions = folded.compact_positions.flip({0});
  reordered.fold_gather = folded.fold_gather.flip({0});
  reordered.scatter_unfold = 2 - folded.scatter_unfold;
  validate_radix(reordered, true);
  TORCH_CHECK(at::allclose(shared_output, model->forward(reordered), 1e-5, 1e-6),
              "Qwen3.5 reordered compact rows changed readout");

  auto bidirectional_cfg = cfg;
  bidirectional_cfg.values["decision_attention_mode"] = "noncausal_full_attention";
  auto bidirectional = build(bidirectional_cfg, weights, c10::Device("cpu"), at::kFloat);
  bidirectional->ready();
  TORCH_CHECK(!bidirectional->supports_radix(), "Noncausal Qwen3.5 must reject prefix folding");
  bool rejected = false;
  try {
    bidirectional->forward(folded);
  } catch (const c10::Error&) {
    rejected = true;
  }
  TORCH_CHECK(rejected, "Noncausal Qwen3.5 accepted folded prefixes");
  auto bi = bidirectional->forward(make({1, 2, 3, 4, 5}, offsets));
  check_prefix_cache(*bidirectional, make({1, 2, 3, 4, 5}, offsets));
  auto future = bidirectional->forward(make({1, 2, 31, 4, 5}, offsets));
  TORCH_CHECK((bi.select(0, 0) - future.select(0, 0)).abs().max().item<float>() > 1e-5,
              "Qwen3.5 noncausal full attention ignored future tokens");
  TORCH_CHECK(at::allclose(bi.narrow(0, 3, 2), future.narrow(0, 3, 2), 1e-5, 1e-6),
              "Noncausal Qwen3.5 crossed packed sequence boundaries");
  std::cout << "Qwen3.5 noncausal-full attention decision mode and sequence isolation passed\n";
  {
    auto block_cfg = cfg;
    block_cfg.values["_decision_attention_mode"] = "block_bidirectional";
    block_cfg.values["num_hidden_layers"] = "4";
    block_cfg.values["layer_types.2"] = "linear_attention";
    block_cfg.values["layer_types.3"] = "full_attention";
    auto block_weights = weights;
    for (const auto& [name, value] : weights)
      for (int i = 0; i < 2; ++i) {
        auto prefix = "model.layers." + std::to_string(i) + ".";
        if (name.starts_with(prefix))
          block_weights["model.layers." + std::to_string(i + 2) + "." +
                        name.substr(prefix.size())] = value;
      }
    std::vector<c10::Device> block_devices{c10::Device("cpu")};
    if (argc > 1)
      block_devices.emplace_back(argv[1]);
    for (const auto& block_device : block_devices) {
      auto device_weights = block_weights;
      for (auto& [name, value] : device_weights)
        value = value.to(block_device, block_device.is_cpu() ? at::kFloat : at::kBFloat16);
      auto blocks = build(block_cfg,
                          device_weights,
                          block_device,
                          block_device.is_cpu() ? at::kFloat : at::kBFloat16);
      blocks->ready();
      auto to_device = [&](PackedInput& input) {
        for (auto* tensor : {&input.ids, &input.types, &input.positions, &input.cumulative})
          *tensor = tensor->to(block_device);
      };
      TORCH_CHECK(!blocks->supports_radix(), "Block cache policy missing");
      std::vector<int32_t> ends{0, 5, 9};
      auto packed = make({1, 2, 3, 4, 5, 1, 2, 7, 8}, ends);
      packed.block_prefix_lengths = {2, 2};
      packed.block_prefix_sources = {0, 0};
      to_device(packed);
      auto cached = blocks->forward(packed);
      packed.block_prefix_sources = {0, 1};
      TORCH_CHECK(at::allclose(cached, blocks->forward(packed), 1e-5, 1e-6),
                  "Whole-prefix state reuse changed outputs");
      auto changed = make({1, 2, 3, 4, 31, 1, 2, 7, 8}, ends);
      changed.block_prefix_lengths = {2, 2};
      changed.block_prefix_sources = {0, 0};
      to_device(changed);
      auto future = blocks->forward(changed);
      TORCH_CHECK(at::allclose(cached.narrow(0, 0, 2), future.narrow(0, 0, 2), 1e-5, 1e-6),
                  "Prefix attended to question suffix");
      TORCH_CHECK((cached.select(0, 2) - future.select(0, 2)).abs().max().item<float>() > 1e-5,
                  "Suffix lost bidirectional attention");
      TORCH_CHECK(at::allclose(cached.narrow(0, 5, 4), future.narrow(0, 5, 4), 1e-5, 1e-6),
                  "Question mutated shared GDN state");
      // Distinct ragged prefixes, shared siblings, a prefix shorter than the
      // convolution history, and a row with no suffix all coexist in one batch.
      std::vector<int32_t> mixed_ends{0, 5, 9, 11, 16, 18};
      std::vector<int64_t> mixed_ids{1, 2, 3, 4, 5, 1, 2, 7, 8, 9, 10, 6, 7, 8, 9, 10, 1, 2};
      auto mixed = make(mixed_ids, mixed_ends);
      mixed.block_prefix_lengths = {2, 2, 1, 3, 2};
      mixed.block_prefix_sources = {0, 0, 2, 3, 0};
      to_device(mixed);
      auto mixed_hidden = blocks->forward(mixed);
      for (bool last : {false, true})
        TORCH_CHECK(at::equal(blocks->forward_pooled(mixed, last),
                              mixed_hidden.index_select(0, pooled_token_rows(mixed, last))),
                    "Compact block readout changed shared, ragged or prefix-only rows");
      for (size_t row = 0; row + 1 < mixed_ends.size(); ++row) {
        std::vector<int32_t> single_ends{0, mixed_ends[row + 1] - mixed_ends[row]};
        auto single = make(std::vector<int64_t>(mixed_ids.begin() + mixed_ends[row],
                                                mixed_ids.begin() + mixed_ends[row + 1]),
                           single_ends);
        single.block_prefix_lengths = {mixed.block_prefix_lengths[row]};
        single.block_prefix_sources = {0};
        to_device(single);
        const auto tolerance = block_device.is_cpu() ? 1e-5 : 1e-2;
        TORCH_CHECK(at::allclose(mixed_hidden.narrow(0, mixed_ends[row], single_ends[1]),
                                 blocks->forward(single),
                                 tolerance,
                                 tolerance),
                    "Packed block rows crossed prefix, convolution, or recurrent boundaries on ",
                    block_device,
                    " row=",
                    row);
      }
      // Exercise both one-question chunks and a later multi-question chunk,
      // then replay graphs using the latter boundary layout.
      for (const auto* target : {"4", "7"}) {
        setenv(INFINITY_ENV("TORCH_BLOCK_KV_CHUNK_TOKENS"), target, 1);
        auto chunked = build(block_cfg,
                             device_weights,
                             block_device,
                             block_device.is_cpu() ? at::kFloat : at::kBFloat16);
        unsetenv(INFINITY_ENV("TORCH_BLOCK_KV_CHUNK_TOKENS"));
        chunked->ready();
        const auto chunk_tolerance = block_device.is_cpu() ? 1e-5 : 1e-2;
        TORCH_CHECK(
            at::allclose(mixed_hidden, chunked->forward(mixed), chunk_tolerance, chunk_tolerance),
            "Chunked block K/V changed ragged multi-prefix outputs");
        blocks = std::move(chunked);
      }
      if (block_device.is_cuda()) {
        CudaGraphCache graphs(8, 4096, 1);
        auto shared = packed;
        shared.block_prefix_sources = {0, 0};
        auto independent = packed;
        independent.block_prefix_sources = {0, 1};
        auto shorter = shared;
        shorter.block_prefix_lengths = {1, 1};
        // Equal packed lengths with different sharing/boundaries must not reuse
        // one another's graph. Changed token values must reach every replay.
        for (auto layout : {shared, independent, shorter, mixed}) {
          for (int iteration = 0; iteration < 3; ++iteration) {
            auto replay = layout;
            replay.ids = (layout.ids + iteration).remainder(32);
            auto expected = blocks->forward(replay);
            auto actual = graphs.forward(*blocks, replay);
            TORCH_CHECK(at::allclose(expected, actual, 1e-2, 1e-2),
                        "Block graph replay changed boundaries, sharing, or token values");
          }
        }
        TORCH_CHECK(graphs.stats().captures == 4 && graphs.stats().hits >= 8,
                    "Block execution did not capture and replay each distinct layout: captures=",
                    graphs.stats().captures,
                    " hits=",
                    graphs.stats().hits,
                    " attempts=",
                    graphs.stats().attempts);
        // Pooled and full outputs need distinct graph keys; replay must also
        // select fresh suffix rows and a shared prefix-only row correctly.
        for (int iteration = 0; iteration < 3; ++iteration) {
          auto replay = mixed;
          replay.ids = (mixed.ids + iteration).remainder(32);
          auto expected = blocks->forward(replay).index_select(0, pooled_token_rows(replay, true));
          TORCH_CHECK(at::allclose(expected, graphs.forward(*blocks, replay, 3), 1e-2, 1e-2),
                      "Compact block graph replay changed selected rows");
        }
        TORCH_CHECK(graphs.stats().captures == 5 && graphs.stats().hits >= 10,
                    "Compact block outputs did not use a separate captured graph");
      }
      bool missing = false;
      try {
        blocks->forward(make({1, 2, 3}, one));
      } catch (const c10::Error&) {
        missing = true;
      }
      TORCH_CHECK(missing, "Block mode silently accepted missing boundary metadata");
      std::cout
          << "Qwen3.5 block prefix isolation, bidirectional suffix and immutable GDN cache passed on "
          << block_device << "\n";
    }
  }
  auto moe_cfg = cfg;
  moe_cfg.values["num_experts"] = "3";
  moe_cfg.values["num_experts_per_tok"] = "2";
  auto moe_weights = weights;
  for (int i = 0; i < 2; ++i) {
    auto p = "model.layers." + std::to_string(i) + ".mlp.";
    moe_weights[p + "gate.weight"] = at::zeros({3, 16});
    auto gu = at::cat({weights.at(p + "gate_proj.weight"), weights.at(p + "up_proj.weight")}, 0);
    moe_weights[p + "experts.gate_up_proj"] = at::stack({gu, gu * 1.5, gu * 2.});
    auto down = weights.at(p + "down_proj.weight");
    moe_weights[p + "experts.down_proj"] = at::stack({down, down, down});
    for (auto name : {"gate_proj", "up_proj", "down_proj"})
      moe_weights[p + "shared_expert." + name + ".weight"] = weights.at(p + name + ".weight");
    moe_weights[p + "shared_expert_gate.weight"] = at::zeros({1, 16});
  }
  auto moe = build(moe_cfg, moe_weights, c10::Device("cpu"), at::kFloat);
  moe->ready();
  check_prefix_cache(*moe, make({1, 2, 3, 4, 5}, offsets));
  auto routed = moe->forward(make({1, 2, 3, 4, 5}, offsets));
  auto routed_single =
      at::cat({moe->forward(make({1, 2, 3}, one)), moe->forward(make({4, 5}, two))}, 0);
  TORCH_CHECK(at::allclose(routed, routed_single, 1e-5, 1e-6),
              "Qwen3.5 routed experts leak packed sequences");
  for (int i = 0; i < 2; ++i)
    moe_weights["model.layers." + std::to_string(i) + ".mlp.experts.gate_up_proj"]
        .select(0, 2)
        .fill_(100.);
  TORCH_CHECK(at::allclose(routed, moe->forward(make({1, 2, 3, 4, 5}, offsets)), 1e-5, 1e-6),
              "Qwen3.5 router ties must prefer lower expert IDs");
  std::cout << "Qwen3.5 MoE/shared expert, stable routing ties and packed isolation passed\n";
  if (argc > 1) {
    c10::Device device(argv[1]);
    paged_model_test(cfg, weights, device);
    dynamic_fp8_test(cfg, weights, device);
    for (auto& [key, value] : weights)
      value = value.to(at::kBFloat16);
    auto cpu = build(cfg, weights, c10::Device("cpu"), at::kBFloat16);
    cpu->ready();
    Weights cuda_weights;
    for (const auto& [key, value] : weights)
      cuda_weights[key] = value.to(device);
    c10::weak_intrusive_ptr<c10::TensorImpl> cuda_gate(
        cuda_weights.at("model.layers.0.mlp.gate_proj.weight").getIntrusivePtr());
    auto cuda = create_qwen35(cfg, cuda_weights, device, at::kBFloat16);
    cuda->ready();
    TORCH_CHECK(cuda_weights.empty() && cuda_gate.expired(),
                "CUDA Qwen3.5 retained consumed loader/FFN ownership");
    auto host = make({1, 2, 3, 4, 5}, offsets), dev = host;
    dev.ids = host.ids.to(device);
    dev.types = host.types.to(device);
    dev.positions = host.positions.to(device);
    dev.cumulative = host.cumulative.to(device);
    auto reference = cpu->forward(host).to(at::kFloat),
         actual = cuda->forward(dev).cpu().to(at::kFloat);
    TORCH_CHECK(at::allclose(cuda->forward_pooled(dev, true).cpu().to(at::kFloat),
                             actual.index_select(0, at::tensor({2, 4}, at::kLong)),
                             .02,
                             .02),
                "Qwen3.5 selected last-token rows changed");
    check_prefix_cache(*cuda, dev, 0, .01, .02);
    TORCH_CHECK(at::allclose(reference, actual, .025, .025),
                "Qwen3.5 native CUDA DeltaNet disagrees with CPU recurrence, maxabs=",
                (reference - actual).abs().max().item<float>());
    std::cout << "Qwen3.5 CUDA DeltaNet/CPU reference parity maxabs="
              << (reference - actual).abs().max().item<float>() << "\n";
    const auto prior_backend = at::globalContext().blasPreferredBackend();
    radix_workspace_recovery_test(device);
    TORCH_CHECK(at::globalContext().blasPreferredBackend() == prior_backend,
                "Qwen3.5 workspace fixture did not restore the prior BLAS backend");
  }
}
