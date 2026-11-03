// Packed batch invariants for native encoders. This fixture needs no Python or
// downloaded checkpoint. Run with --cuda to exercise Torch varlen FlashAttention.
#include "models/encoder_models.h"
#include "models/model.h"
#include "helpers/modern_cpu_math.h"
#include "helpers/fp8_linear.h"
#include "batching/cuda_graph.h"
#include "bridge.h"
#ifdef TEI_TORCH_CUDA_KERNELS
#include <ATen/cuda/CUDAContext.h>
#endif
#include <ATen/Parallel.h>
#include <c10/core/InferenceMode.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace {
constexpr int64_t H = 32, I = 48;

void add(infinity::Weights& w,
         const std::string& name,
         std::vector<int64_t> dims,
         c10::Device device,
         at::ScalarType dtype,
         bool identity = false) {
  auto value = identity ? at::ones(dims) : at::randn(dims) * .03;
  w[name] = value.to(device, dtype);
}

void linear(infinity::Weights& w,
            const std::string& n,
            int64_t output,
            int64_t input,
            c10::Device device,
            at::ScalarType dtype,
            bool bias = true) {
  add(w, n + ".weight", {output, input}, device, dtype);
  if (bias)
    add(w, n + ".bias", {output}, device, dtype);
}

void norm(infinity::Weights& w,
          const std::string& n,
          c10::Device device,
          at::ScalarType dtype,
          bool bias = true) {
  add(w, n + ".weight", {H}, device, dtype, true);
  if (bias)
    add(w, n + ".bias", {H}, device, dtype);
}

infinity::Options config(const std::string& type) {
  infinity::Options o;
  o.values = {{"model_type", type},
              {"hidden_size", "32"},
              {"dim", "32"},
              {"n_embd", "32"},
              {"num_attention_heads", "4"},
              {"n_heads", "4"},
              {"n_head", "4"},
              {"num_hidden_layers", "2"},
              {"n_layers", "2"},
              {"n_layer", "2"},
              {"max_position_embeddings", "32"},
              {"n_positions", "32"},
              {"global_attn_every_n_layers", "2"},
              {"local_attention", "4"},
              {"activation_function", "gelu"},
              {"type_vocab_size", "2"}};
  if (type == "jina" || type == "jina-code") {
    o.values["model_type"] = "bert";
    o.values["_name_or_path"] =
        type == "jina" ? "jinaai/jina-bert-implementation" : "jinaai/jina-bert-v2-qk-post-norm";
    o.values["position_embedding_type"] = "alibi";
  }
  return o;
}

infinity::Weights weights(const std::string& type, c10::Device d, at::ScalarType dt) {
  infinity::Weights w;
  add(w,
      type == "modernbert" ? "embeddings.tok_embeddings.weight"
                           : "embeddings.word_embeddings.weight",
      {64, H},
      d,
      dt);
  norm(w,
       type == "modernbert"   ? "embeddings.norm"
       : type == "nomic_bert" ? "emb_ln"
                              : "embeddings.LayerNorm",
       d,
       dt,
       type != "modernbert");
  if (type != "modernbert" && type != "distilbert")
    add(w, "embeddings.token_type_embeddings.weight", {2, H}, d, dt);
  if (type == "distilbert" || type == "mpnet" || type == "deberta-v2")
    add(w, "embeddings.position_embeddings.weight", {32, H}, d, dt);
  if (type == "mpnet")
    add(w, "encoder.relative_attention_bias.weight", {32, 4}, d, dt);
  for (int i = 0; i < 2; ++i) {
    std::string p = (type == "modernbert"   ? "layers."
                     : type == "nomic_bert" ? "encoder.layers."
                     : type == "distilbert" ? "transformer.layer."
                                            : "encoder.layer.") +
                    std::to_string(i) + ".";
    if (type == "deberta-v2") {
      for (auto q : {"query_proj", "key_proj", "value_proj"})
        linear(w, p + "attention.self." + q, H, H, d, dt);
      linear(w, p + "attention.output.dense", H, H, d, dt);
      norm(w, p + "attention.output.LayerNorm", d, dt);
      norm(w, p + "output.LayerNorm", d, dt);
      linear(w, p + "intermediate.dense", I, H, d, dt);
      linear(w, p + "output.dense", H, I, d, dt);
    } else if (type == "mpnet") {
      for (auto q : {"q", "k", "v", "o"})
        linear(w, p + "attention.attn." + q, H, H, d, dt);
      norm(w, p + "attention.LayerNorm", d, dt);
      norm(w, p + "output.LayerNorm", d, dt);
      linear(w, p + "intermediate.dense", I, H, d, dt);
      linear(w, p + "output.dense", H, I, d, dt);
    } else if (type == "distilbert") {
      for (auto q : {"q_lin", "k_lin", "v_lin", "out_lin"})
        linear(w, p + "attention." + q, H, H, d, dt);
      linear(w, p + "ffn.lin1", I, H, d, dt);
      linear(w, p + "ffn.lin2", H, I, d, dt);
      norm(w, p + "sa_layer_norm", d, dt);
      norm(w, p + "output_layer_norm", d, dt);
    } else if (type == "modernbert") {
      linear(w, p + "attn.Wqkv", 3 * H, H, d, dt, false);
      linear(w, p + "attn.Wo", H, H, d, dt, false);
      linear(w, p + "mlp.Wi", 2 * I, H, d, dt, false);
      linear(w, p + "mlp.Wo", H, I, d, dt, false);
      if (i)
        norm(w, p + "attn_norm", d, dt, false);
      norm(w, p + "mlp_norm", d, dt, false);
    } else if (type == "nomic_bert") {
      linear(w, p + "attn.Wqkv", 3 * H, H, d, dt);
      linear(w, p + "attn.out_proj", H, H, d, dt);
      linear(w, p + "mlp.fc1", I, H, d, dt);
      linear(w, p + "mlp.fc2", H, I, d, dt);
      norm(w, p + "norm1", d, dt);
      norm(w, p + "norm2", d, dt);
    } else if (type == "new") {
      linear(w, p + "attention.qkv_proj", 3 * H, H, d, dt);
      linear(w, p + "attention.o_proj", H, H, d, dt);
      linear(w, p + "mlp.up_gate_proj", 2 * I, H, d, dt, false);
      linear(w, p + "mlp.down_proj", H, I, d, dt);
      norm(w, p + "attn_ln", d, dt);
      norm(w, p + "mlp_ln", d, dt);
    } else {
      for (auto q : {"query", "key", "value"})
        linear(w, p + "attention.self." + q, H, H, d, dt);
      linear(w, p + "attention.output.dense", H, H, d, dt);
      norm(w, p + "attention.output.LayerNorm", d, dt);
      if (type == "jina-code") {
        norm(w, p + "attention.self.layer_norm_q", d, dt);
        norm(w, p + "attention.self.layer_norm_k", d, dt);
        norm(w, p + "layer_norm_1", d, dt);
        norm(w, p + "layer_norm_2", d, dt);
        linear(w, p + "mlp.up_gated_layer", 2 * I, H, d, dt, false);
        linear(w, p + "mlp.down_layer", H, I, d, dt);
      } else {
        linear(w, p + "mlp.gated_layers", 2 * I, H, d, dt, false);
        linear(w, p + "mlp.wo", H, I, d, dt);
        norm(w, p + "mlp.layernorm", d, dt);
      }
    }
  }
  if (type == "modernbert")
    norm(w, "final_norm", d, dt, false);
  return w;
}

void fp8_encoder(const infinity::Options& options,
                 const infinity::Weights& checkpoint,
                 const infinity::PackedInput& input) {
#ifdef TEI_TORCH_CUDA_KERNELS
  const auto device = input.ids.device();
  if (!infinity::fp8_device_supported(device))
    return;
  for (auto dtype : {at::kHalf, at::kBFloat16}) {
    infinity::Weights reference_weights, quantized_weights;
    for (const auto& [name, value] : checkpoint)
      reference_weights[name] = quantized_weights[name] = value.to(dtype);
    auto reference = infinity::create_encoder(options, reference_weights, device, dtype);
    reference->ready();
    auto quantized_options = options;
    quantized_options.values["_fp8_dynamic"] = "true";
    auto quantized = infinity::create_encoder(quantized_options, quantized_weights, device, dtype);
    quantized->ready();
    TORCH_CHECK(quantized_weights.size() < reference_weights.size(),
                "Encoder FP8 retained original MLP matrices");
    infinity::CudaGraphCache graphs(1, 4096, 1);
    for (int repeat = 0; repeat < 3; ++repeat) {
      auto replay = input;
      replay.ids = (input.ids + repeat).remainder(64);
      auto expected = reference->forward(replay).to(at::kFloat);
      auto actual = quantized->forward(replay).to(at::kFloat);
      const auto error = ((actual - expected).norm() / expected.norm()).item<double>();
      TORCH_CHECK(at::isfinite(actual).all().item<bool>() && error < .03,
                  "Encoder FP8 numerical regression: ",
                  options.string("model_type"),
                  " relative L2=",
                  error);
      TORCH_CHECK(at::allclose(actual, graphs.forward(*quantized, replay).to(at::kFloat), .01, .01),
                  "Encoder FP8 graph replay changed outputs");
    }
    TORCH_CHECK(graphs.stats().captures == 1 && graphs.stats().hits >= 2,
                "Encoder FP8 graph was not captured and replayed");
    std::cout << options.string("model_type") << " FP8 parity, released weights and graphs passed "
              << dtype << "\n";
  }
#endif
}

void nomic_expert_shape_validation(c10::Device device, at::ScalarType dtype) {
  constexpr int64_t experts = 3;
  auto o = config("nomic_bert");
  o.values["num_experts"] = std::to_string(experts);
  o.values["moe_top_k"] = "2";
  o.values["moe_every_n_layers"] = "2";
  o.values["n_inner"] = std::to_string(I);
  auto checkpoint = weights("nomic_bert", device, dtype);
  const std::string p = "encoder.layers.1.mlp.";
  linear(checkpoint, p + "router.layer", experts, H, device, dtype, false);
  for (auto name : {"w1", "w2"})
    add(checkpoint, p + "experts.mlp." + name, {experts * I, H}, device, dtype);
  add(checkpoint, p + "experts.bias", {H}, device, dtype);

  auto valid = infinity::create_encoder(o, checkpoint, device, dtype);
  valid->ready();
  auto opts = at::TensorOptions().device(device).dtype(at::kLong);
  auto ids = at::tensor({1, 2, 3}, opts), types = at::zeros_like(ids);
  auto positions = at::tensor({0, 1, 2}, opts),
       cumulative = at::tensor({0, 3}, opts.dtype(at::kInt));
  int32_t offsets[] = {0, 3};
  auto output = valid->forward({ids, types, positions, cumulative, offsets, 1, 3});
  TORCH_CHECK(output.sizes() == at::IntArrayRef({3, H}) && at::isfinite(output).all().item<bool>(),
              "Valid Nomic expert checkpoint failed inference");

  for (auto name : {"w1", "w2"}) {
    const auto key = p + "experts.mlp." + name;
    // Reject indivisible expert groups, indivisible hidden widths, and an
    // otherwise divisible tensor that disagrees with the configured FFN width.
    for (auto elements : {experts * I * H + 1, experts * (I * H + 1), experts * (I + 1) * H}) {
      auto malformed = checkpoint;
      malformed[key] = at::zeros({elements}, output.options());
      auto model = infinity::create_encoder(o, malformed, device, dtype);
      bool rejected = false;
      try {
        model->ready();
      } catch (const c10::Error& error) {
        rejected = std::string(error.what()).find("Invalid Nomic expert tensor dimensions") !=
                   std::string::npos;
      }
      TORCH_CHECK(
          rejected, "Malformed Nomic expert weights passed ready(): ", key, " elements=", elements);
    }
  }
  // Expert tensors in an entirely dense configuration do not participate in
  // inference, even if the checkpoint carries unused expert-count metadata.
  for (int every : {0, 1}) {
    auto dense = o;
    dense.values["moe_every_n_layers"] = std::to_string(every);
    auto unused = checkpoint;
    unused[p + "experts.mlp.w1"] = at::zeros({1}, output.options());
    auto model = infinity::create_encoder(dense, unused, device, dtype);
    model->ready();
    TORCH_CHECK(at::isfinite(model->forward({ids, types, positions, cumulative, offsets, 1, 3}))
                    .all()
                    .item<bool>(),
                "Unused Nomic expert tensors affected dense inference");
  }
}

struct AbiRequest {
  std::vector<int64_t> ids, types, positions;
  std::vector<int32_t> cumulative;
  int64_t batch = 0, max_sequence = 0;
  std::vector<int64_t> pooled, raw;
};

int32_t abi_scalar_type(at::ScalarType type) {
  if (type == at::kFloat)
    return 0;
  if (type == at::kHalf)
    return 1;
  if (type == at::kBFloat16)
    return 2;
  TORCH_CHECK(false, "Unsupported ABI fixture dtype");
  return 0;
}

AbiRequest make_abi_request(const std::vector<int64_t>& lengths, int64_t first_id) {
  AbiRequest request;
  request.batch = static_cast<int64_t>(lengths.size());
  request.cumulative.push_back(0);
  int64_t next = first_id;
  for (size_t row = 0; row < lengths.size(); ++row) {
    request.max_sequence = std::max(request.max_sequence, lengths[row]);
    for (int64_t index = 0; index < lengths[row]; ++index) {
      request.ids.push_back((next + index) % 64);
      request.types.push_back(static_cast<int64_t>(row % 2));
      request.positions.push_back(index);
    }
    next += lengths[row] + 7;
    request.cumulative.push_back(static_cast<int32_t>(request.ids.size()));
  }
  return request;
}

void* create_abi_encoder(const infinity::Options& options,
                         const infinity::Weights& checkpoint,
                         c10::Device device,
                         at::ScalarType dtype,
                         bool graphs) {
  TeiConfig shape{H, 4, 2, I, 64, 32, 2, 1e-12, 0};
  std::unique_ptr<void, decltype(&tei_destroy)> engine(
      tei_create(&shape, device.str().c_str(), abi_scalar_type(dtype)), tei_destroy);
  TORCH_CHECK(engine, tei_error());
  for (const auto& [key, value] : options.values)
    TORCH_CHECK(tei_option(engine.get(), key.c_str(), value.c_str()) == 0, tei_error());
  if (graphs) {
    TORCH_CHECK(tei_option(engine.get(), "_cuda_graphs", "true") == 0, tei_error());
    TORCH_CHECK(tei_option(engine.get(), "_cuda_graph_reuse", "1") == 0, tei_error());
    TORCH_CHECK(tei_option(engine.get(), "_cuda_graph_entries", "4") == 0, tei_error());
  }
  for (const auto& [name, value] : checkpoint) {
    auto host = value.to(at::kCPU, dtype).contiguous();
    TORCH_CHECK(tei_weight(engine.get(),
                           name.c_str(),
                           reinterpret_cast<const uint8_t*>(host.const_data_ptr()),
                           host.sizes().data(),
                           host.dim(),
                           abi_scalar_type(dtype)) == 0,
                tei_error());
  }
  TORCH_CHECK(tei_ready(engine.get()) == 0, tei_error());
  return engine.release();
}

size_t abi_output_count(void* engine, const AbiRequest& request) {
  const auto pooled_width = tei_pooled_width(engine), raw_width = tei_output_width(engine);
  TORCH_CHECK(pooled_width > 0 && raw_width > 0, "Invalid ABI output width");
  size_t count = request.pooled.size() * static_cast<size_t>(pooled_width);
  for (const auto index : request.raw) {
    TORCH_CHECK(index >= 0 && index < request.batch, "Invalid ABI raw index");
    count += static_cast<size_t>(request.cumulative[index + 1] - request.cumulative[index]) *
             static_cast<size_t>(raw_width);
  }
  return count;
}

std::vector<float> run_abi_sync(void* engine, const AbiRequest& request) {
  const auto capacity = abi_output_count(engine, request);
  std::vector<float> output(capacity);
  TORCH_CHECK(tei_forward(engine,
                          request.ids.data(),
                          request.types.data(),
                          request.positions.data(),
                          request.cumulative.data(),
                          request.batch,
                          request.max_sequence,
                          1,
                          request.pooled.data(),
                          request.pooled.size(),
                          request.raw.data(),
                          request.raw.size(),
                          output.data(),
                          capacity,
                          nullptr,
                          nullptr,
                          0,
                          nullptr,
                          0) == 0,
              tei_error());
  return output;
}

struct PendingAbiOutput {
  void* value = nullptr;
  size_t capacity = 0;
};

PendingAbiOutput submit_abi_async(void* engine, const AbiRequest& request) {
  auto ids = request.ids, types = request.types, positions = request.positions,
       pooled = request.pooled, raw = request.raw;
  auto cumulative = request.cumulative;
  const auto capacity = abi_output_count(engine, request);
  auto* output = tei_forward_submit(engine,
                                    ids.data(),
                                    types.data(),
                                    positions.data(),
                                    cumulative.data(),
                                    request.batch,
                                    request.max_sequence,
                                    1,
                                    pooled.data(),
                                    pooled.size(),
                                    raw.data(),
                                    raw.size(),
                                    capacity,
                                    nullptr,
                                    nullptr,
                                    0,
                                    nullptr,
                                    0,
                                    nullptr);
  TORCH_CHECK(output, tei_error());
  auto poison_and_release = [](auto& values) {
    using Value = typename std::decay_t<decltype(values)>::value_type;
    std::fill(values.begin(), values.end(), static_cast<Value>(-1));
    std::vector<Value>().swap(values);
  };
  poison_and_release(ids);
  poison_and_release(types);
  poison_and_release(positions);
  poison_and_release(cumulative);
  poison_and_release(pooled);
  poison_and_release(raw);
  return {output, capacity};
}

std::vector<float> read_abi_async(void* output, size_t capacity) {
  const float* values = tei_output_wait(output);
  TORCH_CHECK(values, tei_error());
  return std::vector<float>(values, values + capacity);
}

std::array<uint64_t, 7> abi_graph_stats(void* engine) {
  std::array<uint64_t, 7> counters{};
  TORCH_CHECK(tei_graph_stats(engine, counters.data(), counters.size()) == 0, tei_error());
  return counters;
}

bool same_float_bits(const std::vector<float>& left, const std::vector<float>& right) {
  return left.size() == right.size() &&
         (left.empty() || std::memcmp(left.data(), right.data(), left.size() * sizeof(float)) == 0);
}

void append_float_rows(std::vector<float>& output, const at::Tensor& rows) {
  auto flat = rows.to(at::kFloat).to(at::kCPU).contiguous().view({-1});
  const auto* values = flat.const_data_ptr<float>();
  output.insert(output.end(), values, values + flat.numel());
}

struct NativeEncoder {
  infinity::Options options;
  infinity::Weights weights;
  c10::Device device;
  std::unique_ptr<infinity::Model> model;

  NativeEncoder(const infinity::Options& config,
                const infinity::Weights& checkpoint,
                c10::Device value,
                at::ScalarType dtype)
      : options(config), weights(checkpoint), device(value) {
    for (auto& item : weights)
      item.second = item.second.to(device, dtype);
    model = infinity::create_encoder(options, weights, device, dtype);
    TORCH_CHECK(model, "Missing native encoder fixture for ", options.string("model_type"));
    model->ready();
  }
};

at::Tensor native_hidden(const NativeEncoder& native, const AbiRequest& request) {
  auto options = at::TensorOptions().device(native.device).dtype(at::kLong);
  auto ids = at::tensor(request.ids, options), types = at::tensor(request.types, options),
       positions = at::tensor(request.positions, options);
  auto cumulative = at::tensor(request.cumulative, options.dtype(at::kInt));
  return native.model->forward({ids,
                                types,
                                positions,
                                cumulative,
                                request.cumulative.data(),
                                request.batch,
                                request.max_sequence});
}

std::vector<float> native_raw(const AbiRequest& request, const at::Tensor& hidden) {
  std::vector<float> output;
  for (const auto index : request.raw) {
    const auto start = request.cumulative[index], length = request.cumulative[index + 1] - start;
    append_float_rows(output, hidden.narrow(0, start, length));
  }
  return output;
}

std::vector<float>
native_mean(const NativeEncoder& native, const AbiRequest& request, const at::Tensor& hidden) {
  std::vector<float> output;
  for (const auto index : request.pooled) {
    const auto start = request.cumulative[index], length = request.cumulative[index + 1] - start;
    append_float_rows(output, native.model->mean_pool(hidden.narrow(0, start, length)));
  }
  return output;
}

void check_abi_reference(const AbiRequest& request,
                         const std::vector<float>& actual,
                         const at::Tensor& hidden,
                         const NativeEncoder& native,
                         bool check_native_mean) {
  const auto width = hidden.size(1);
  size_t raw_elements = 0;
  for (const auto index : request.raw)
    raw_elements += static_cast<size_t>(request.cumulative[index + 1] - request.cumulative[index]) *
                    static_cast<size_t>(width);
  const auto pooled_elements = request.pooled.size() * static_cast<size_t>(width);
  TORCH_CHECK(actual.size() == pooled_elements + raw_elements,
              "C ABI output size mismatch for ",
              request.batch);
  const auto mean_end = actual.begin() + static_cast<std::ptrdiff_t>(pooled_elements);
  TORCH_CHECK(
      same_float_bits(std::vector<float>(mean_end, actual.end()), native_raw(request, hidden)),
      "C ABI raw output changed from native packed rows");
  if (check_native_mean)
    TORCH_CHECK(same_float_bits(std::vector<float>(actual.begin(), mean_end),
                                native_mean(native, request, hidden)),
                "C ABI mean output changed from native mean pooling");
}

void check_mean_selection(const std::vector<float>& full,
                          const std::vector<int64_t>& full_order,
                          const std::vector<float>& selected,
                          const std::vector<int64_t>& selected_order,
                          int64_t width,
                          const std::string& label) {
  TORCH_CHECK(full.size() >= full_order.size() * static_cast<size_t>(width) &&
                  selected.size() >= selected_order.size() * static_cast<size_t>(width),
              "Invalid mean-selection fixture ",
              label);
  for (size_t row = 0; row < selected_order.size(); ++row) {
    const auto found = std::find(full_order.begin(), full_order.end(), selected_order[row]);
    TORCH_CHECK(found != full_order.end(), "Missing pooled row in ", label);
    const auto source = static_cast<size_t>(found - full_order.begin()) * width;
    const auto target = row * static_cast<size_t>(width);
    TORCH_CHECK(std::memcmp(full.data() + source,
                            selected.data() + target,
                            static_cast<size_t>(width) * sizeof(float)) == 0,
                "Mean pooled selection order changed for ",
                label);
  }
}

std::vector<float> mean_rows(const std::vector<float>& output, size_t rows, int64_t width) {
  const auto end = output.begin() + static_cast<std::ptrdiff_t>(rows * static_cast<size_t>(width));
  return std::vector<float>(output.begin(), end);
}

void abi_encoder_case(const std::string& type, c10::Device device, at::ScalarType dtype) {
  auto options = config(type);
  auto checkpoint = weights(type, c10::Device(at::kCPU), dtype);
  NativeEncoder native(options, checkpoint, device, dtype);
  const auto width = native.model->output_width();
  TORCH_CHECK(width == H, "Unexpected native fixture width for ", type);

  std::unique_ptr<void, decltype(&tei_destroy)> reference(
      create_abi_encoder(options, checkpoint, device, dtype, false), tei_destroy);
  TORCH_CHECK(tei_output_width(reference.get()) == width &&
                  tei_pooled_width(reference.get()) == width,
              "C ABI width mismatch for ",
              type);

  const auto a = make_abi_request({2, 4, 3}, 2), b = make_abi_request({2, 4, 3}, 31),
             c = make_abi_request({1, 5}, 51);
  auto a_full = a, a_partial = a, a_sparse = a, b_full = b, c_full = c, c_partial = c;
  a_full.pooled = {0, 1, 2};
  a_full.raw = {2, 0, 2};
  a_partial.pooled = {2, 0, 2};
  a_partial.raw = {1, 0};
  a_sparse.pooled = {1};
  a_sparse.raw = {2};
  b_full.pooled = a_full.pooled;
  b_full.raw = a_full.raw;
  c_full.pooled = {0, 1};
  c_full.raw = {1, 0};
  c_partial.pooled = {1, 0, 1};
  c_partial.raw = {1, 0};

  const auto hidden_a = native_hidden(native, a), hidden_b = native_hidden(native, b),
             hidden_c = native_hidden(native, c);
  const auto expected_a_full = run_abi_sync(reference.get(), a_full),
             expected_a_partial = run_abi_sync(reference.get(), a_partial),
             expected_a_sparse = run_abi_sync(reference.get(), a_sparse),
             expected_b_full = run_abi_sync(reference.get(), b_full),
             expected_c_full = run_abi_sync(reference.get(), c_full),
             expected_c_partial = run_abi_sync(reference.get(), c_partial);
  const bool native_mean_available = !device.is_cuda();
  check_abi_reference(a_full, expected_a_full, hidden_a, native, native_mean_available);
  check_abi_reference(a_partial, expected_a_partial, hidden_a, native, native_mean_available);
  check_abi_reference(a_sparse, expected_a_sparse, hidden_a, native, native_mean_available);
  check_abi_reference(b_full, expected_b_full, hidden_b, native, native_mean_available);
  check_abi_reference(c_full, expected_c_full, hidden_c, native, native_mean_available);
  check_abi_reference(c_partial, expected_c_partial, hidden_c, native, native_mean_available);
  check_mean_selection(mean_rows(expected_a_full, a_full.pooled.size(), width),
                       a_full.pooled,
                       mean_rows(expected_a_partial, a_partial.pooled.size(), width),
                       a_partial.pooled,
                       width,
                       type + " A");
  check_mean_selection(mean_rows(expected_a_full, a_full.pooled.size(), width),
                       a_full.pooled,
                       mean_rows(expected_a_sparse, a_sparse.pooled.size(), width),
                       a_sparse.pooled,
                       width,
                       type + " A sparse");
  check_mean_selection(mean_rows(expected_c_full, c_full.pooled.size(), width),
                       c_full.pooled,
                       mean_rows(expected_c_partial, c_partial.pooled.size(), width),
                       c_partial.pooled,
                       width,
                       type + " C");
  TORCH_CHECK(!same_float_bits(expected_a_full, expected_b_full),
              "Changed token payload is indistinguishable for ",
              type);
  TORCH_CHECK(!same_float_bits(mean_rows(expected_a_full, a_full.pooled.size(), width),
                               mean_rows(expected_a_partial, a_partial.pooled.size(), width)),
              "Pooled selection is indistinguishable for ",
              type);

  if (!device.is_cuda()) {
    std::cout << type << " C ABI synchronous raw/mean selection passed\n";
    return;
  }

  const bool graph_supported = type != "new";
  std::unique_ptr<void, decltype(&tei_destroy)> subject(
      create_abi_encoder(options, checkpoint, device, dtype, graph_supported), tei_destroy);
  std::array<uint64_t, 7> replayed{};
  if (graph_supported) {
    TORCH_CHECK(same_float_bits(run_abi_sync(subject.get(), a_full), expected_a_full),
                "CUDA graph sync changed raw/mean output for ",
                type);
    const auto captured = abi_graph_stats(subject.get());
    TORCH_CHECK(tei_graph_count(subject.get()) == 1 && captured[2] >= 1 && captured[3] >= 1,
                "CUDA graph was not captured for ",
                type);
    TORCH_CHECK(same_float_bits(run_abi_sync(subject.get(), a_partial), expected_a_partial),
                "CUDA graph sync changed pooled selection for ",
                type);
    replayed = abi_graph_stats(subject.get());
    TORCH_CHECK(replayed[0] >= 1 && replayed[2] == captured[2] &&
                    tei_graph_count(subject.get()) == 1,
                "CUDA graph did not replay for same-geometry pooled selection ",
                type);
  } else {
    TORCH_CHECK(same_float_bits(run_abi_sync(subject.get(), a_full), expected_a_full),
                "CUDA sync changed raw/mean output for ",
                type);
    TORCH_CHECK(same_float_bits(run_abi_sync(subject.get(), a_partial), expected_a_partial),
                "CUDA sync changed pooled selection for ",
                type);
  }

  auto pending_b = submit_abi_async(subject.get(), b_full);
  auto pending_c = submit_abi_async(subject.get(), c_partial);
  std::unique_ptr<void, decltype(&tei_output_destroy)> lease_b(pending_b.value, tei_output_destroy);
  std::unique_ptr<void, decltype(&tei_output_destroy)> lease_c(pending_c.value, tei_output_destroy);
  const auto async_c = read_abi_async(lease_c.get(), pending_c.capacity);
  const auto async_b = read_abi_async(lease_b.get(), pending_b.capacity);
  lease_c.reset();
  lease_b.reset();
  TORCH_CHECK(same_float_bits(async_b, expected_b_full),
              "Deferred changed-payload replay changed output for ",
              type);
  TORCH_CHECK(same_float_bits(async_c, expected_c_partial),
              "Deferred geometry change changed output for ",
              type);
  const auto changed = abi_graph_stats(subject.get());
  if (graph_supported) {
    TORCH_CHECK(tei_graph_count(subject.get()) == 2,
                "Alternating geometry did not create a second graph for ",
                type);
    TORCH_CHECK(changed[0] >= replayed[0] + 1 && changed[2] == replayed[2] + 1,
                "Changed-payload graph replay counters are wrong for ",
                type);
  }

  auto pending_a = submit_abi_async(subject.get(), a_full);
  std::unique_ptr<void, decltype(&tei_output_destroy)> lease_a(pending_a.value, tei_output_destroy);
  const auto async_a = read_abi_async(lease_a.get(), pending_a.capacity);
  lease_a.reset();
  const auto reused = abi_graph_stats(subject.get());
  TORCH_CHECK(same_float_bits(async_a, expected_a_full),
              "Reused output slot changed deferred result for ",
              type);
  if (graph_supported)
    TORCH_CHECK(tei_graph_count(subject.get()) == 2 && reused[0] >= changed[0] + 1 &&
                    reused[2] == changed[2],
                "Alternating output slot did not reuse the graph for ",
                type);
  std::cout << type
            << (graph_supported ? " C ABI sync/deferred raw/mean graph replay passed\n"
                                : " C ABI sync/deferred raw/mean selection passed\n");
}

void abi_encoder_regression(c10::Device device) {
  const std::vector<std::string> families = {"modernbert", "nomic_bert", "new", "distilbert"};
  const std::vector<at::ScalarType> dtypes =
      device.is_cuda() ? std::vector<at::ScalarType>{at::kHalf, at::kBFloat16}
                       : std::vector<at::ScalarType>{at::kFloat};
  for (const auto dtype : dtypes)
    for (const auto& family : families)
      abi_encoder_case(family, device, dtype);
}
}

int main(int argc, char** argv) {
  c10::InferenceMode guard;
  // The Modern Half profile must not select serial attention on multicore.
  // Keep this in the existing encoder target; unsupported CPUs remain valid.
  const int saved_threads = at::get_num_threads();
  at::set_num_threads(1);
  const bool single_thread_profile = infinity::modern_cpu::supported_profile();
  for (int threads : {2, 4}) {
    at::set_num_threads(threads);
    TORCH_CHECK(!infinity::modern_cpu::supported_profile(),
                "Modern exact CPU Half profile must reject multicore intra-op threads");
  }
  at::set_num_threads(1);
  TORCH_CHECK(infinity::modern_cpu::supported_profile() == single_thread_profile,
              "Restoring one intra-op thread changed Modern CPU hardware eligibility");
  at::set_num_threads(saved_threads);
  {
    // Exercise the actual Modern override without running the large backbone.
    // The Half reduction distinguishes [2048, 1, -2048] from the F32 default.
    const int previous_threads = at::get_num_threads();
    auto cpu_half = at::TensorOptions().device(at::kCPU).dtype(at::kHalf);
    auto spans = at::tensor({2048.f, 1.f, -2048.f}, at::TensorOptions().dtype(at::kFloat))
                     .to(at::kHalf)
                     .view({3, 1})
                     .repeat({1, 768});
    auto half_mean = infinity::modern_cpu::mean(spans);
    auto float_mean = spans.mean(0, true, at::kFloat);
    TORCH_CHECK((half_mean.to(at::kFloat) - float_mean).abs().max().item<float>() > .3,
                "Mean pooling fixture must discriminate Half and F32 accumulation");
    auto modern = config("modernbert");
    modern.values["hidden_size"] = "768";
    modern.values["num_attention_heads"] = "12";
    modern.values["intermediate_size"] = "1152";
    modern.values["num_hidden_layers"] = "1";
    infinity::Weights modern_weights;
    modern_weights["embeddings.tok_embeddings.weight"] = at::zeros({64, 768}, cpu_half);
    modern_weights["embeddings.norm.weight"] = at::ones({768}, cpu_half);
    modern_weights["final_norm.weight"] = at::ones({768}, cpu_half);
    modern_weights["layers.0.mlp_norm.weight"] = at::ones({768}, cpu_half);
    for (const auto& item :
         std::vector<std::pair<std::string, std::vector<int64_t>>>{{"attn.Wqkv", {2304, 768}},
                                                                   {"attn.Wo", {768, 768}},
                                                                   {"mlp.Wi", {2304, 768}},
                                                                   {"mlp.Wo", {768, 1152}}})
      modern_weights["layers.0." + item.first + ".weight"] = at::zeros(item.second, cpu_half);
    for (int threads : {1, 2}) {
      at::set_num_threads(threads);
      auto modern_model =
          infinity::create_encoder(modern, modern_weights, c10::Device(at::kCPU), at::kHalf);
      modern_model->ready();
      auto pooled = modern_model->mean_pool(spans);
      const bool eligible = infinity::modern_cpu::supported_profile();
      TORCH_CHECK(pooled.scalar_type() == (eligible ? at::kHalf : at::kFloat),
                  "Modern mean pooling dtype did not follow ready-established profile");
      TORCH_CHECK(at::equal(pooled, eligible ? half_mean : float_mean),
                  "Modern mean pooling did not follow qualified profile arithmetic");
    }
    at::set_num_threads(previous_threads);
  }
  at::manual_seed(13);
  c10::Device device(argc > 1 ? "cuda:0" : "cpu");
  auto dtype = device.is_cuda() ? at::kHalf : at::kFloat;
  nomic_expert_shape_validation(device, dtype);
  auto opts = at::TensorOptions().device(device).dtype(at::kLong);
  int32_t offsets[] = {0, 3, 8}, short_offsets[] = {0, 3}, long_offsets[] = {0, 5};
  auto ids = at::tensor({2, 3, 4, 5, 6, 7, 8, 9}, opts), types = at::zeros({8}, opts),
       positions = at::tensor({0, 1, 2, 0, 1, 2, 3, 4}, opts);
  auto cu = at::tensor({0, 3, 8}, opts.dtype(at::kInt));
  if (device.is_cuda()) {
    // Optional Distil segment tables must remain observable at the optimized
    // Hopper geometry. Changing one request's types must not affect its peer.
    constexpr int64_t width = 768, inner = 16;
    auto o = config("distilbert");
    o.values["dim"] = o.values["hidden_size"] = "768";
    o.values["n_heads"] = o.values["num_attention_heads"] = "12";
    o.values["n_layers"] = o.values["num_hidden_layers"] = "1";
    infinity::Weights w;
    add(w, "embeddings.word_embeddings.weight", {64, width}, device, dtype);
    add(w, "embeddings.position_embeddings.weight", {32, width}, device, dtype);
    add(w, "embeddings.token_type_embeddings.weight", {2, width}, device, dtype);
    auto wide_norm = [&](const std::string& n) {
      add(w, n + ".weight", {width}, device, dtype, true);
      add(w, n + ".bias", {width}, device, dtype);
    };
    wide_norm("embeddings.LayerNorm");
    const std::string p = "transformer.layer.0.";
    for (auto q : {"q_lin", "k_lin", "v_lin", "out_lin"})
      linear(w, p + "attention." + q, width, width, device, dtype);
    linear(w, p + "ffn.lin1", inner, width, device, dtype);
    linear(w, p + "ffn.lin2", width, inner, device, dtype);
    wide_norm(p + "sa_layer_norm");
    wide_norm(p + "output_layer_norm");
    auto model = infinity::create_encoder(o, w, device, dtype);
    model->ready();
    auto baseline = model->forward({ids, types, positions, cu, offsets, 2, 5});
    auto changed_types = types.clone();
    changed_types.narrow(0, 0, 3).fill_(1);
    auto changed = model->forward({ids, changed_types, positions, cu, offsets, 2, 5});
    TORCH_CHECK(!at::equal(baseline.narrow(0, 0, 3), changed.narrow(0, 0, 3)),
                "Distil optional token-type embeddings were ignored");
    TORCH_CHECK(at::equal(baseline.narrow(0, 3, 5), changed.narrow(0, 3, 5)),
                "Distil token-type change contaminated another packed request");
  }
  for (auto type : {"distilbert",
                    "modernbert",
                    "nomic_bert",
                    "new",
                    "jina",
                    "jina-code",
                    "mpnet",
                    "deberta-v2"}) {
    auto o = config(type);
    auto w = weights(type, device, dtype);
    auto model = infinity::create_encoder(o, w, device, dtype);
    model->ready();
    auto packed = model->forward({ids, types, positions, cu, offsets, 2, 5});
    fp8_encoder(o, w, {ids, types, positions, cu, offsets, 2, 5});
    TORCH_CHECK(
        !model->classify_hidden({ids, types, positions, cu, offsets, 2, 5}, packed).defined(),
        "Embedding-only encoder must not expose a classifier readout");
    if (std::string(type) == "modernbert") {
      auto heads = w;
      linear(heads, "classifier", 3, H, device, dtype);
      linear(heads, "head.dense", H, H, device, dtype, false);
      norm(heads, "head.norm", device, dtype, false);
      for (const auto* pool : {"cls", "mean", "last_token"}) {
        auto options = o;
        options.values["classifier_pooling"] = pool;
        auto classifier = infinity::create_encoder(options, heads, device, dtype);
        classifier->ready();
        TORCH_CHECK(
            at::equal(
                classifier->predict({ids, types, positions, cu, offsets, 2, 5}, false),
                classifier->classify_hidden({ids, types, positions, cu, offsets, 2, 5}, packed)),
            "ModernBERT readout changed sequence logits for ",
            pool);
      }
    }
    auto selected_mean = model->mean_pool(packed.narrow(0, 3, 5));
    auto original_mean = packed.narrow(0, 3, 5).mean(0, true, at::kFloat);
    TORCH_CHECK(selected_mean.scalar_type() == at::kFloat &&
                    at::equal(selected_mean, original_mean),
                "Unqualified encoder geometry changed default embedding mean pooling for ",
                type);
    if (std::string(type) == "nomic_bert") {
      auto gated_options = o;
      gated_options.values["activation_function"] = "swiglu";
      auto gated_weights = w;
      for (int i = 0; i < 2; ++i) {
        const auto p = "encoder.layers." + std::to_string(i) + ".mlp.";
        for (const auto* name : {"fc11", "fc12"})
          linear(gated_weights, p + name, I, H, device, dtype);
        gated_weights.erase(p + "fc1.weight");
        gated_weights.erase(p + "fc1.bias");
      }
      fp8_encoder(gated_options, gated_weights, {ids, types, positions, cu, offsets, 2, 5});
      auto routed_options = o;
      routed_options.values["_fp8_dynamic"] = "true";
      routed_options.values["num_experts"] = "8";
      routed_options.values["moe_every_n_layers"] = "2";
      bool fp8_rejected = false;
      try {
        infinity::create_encoder(routed_options, w, device, dtype);
      } catch (const c10::Error& error) {
        fp8_rejected =
            std::string(error.what()).find("routed experts are unsupported") != std::string::npos;
      }
      TORCH_CHECK(fp8_rejected, "Nomic MoE silently accepted dense-only FP8");
      auto alias = o;
      alias.values["activation_function"] = "gelu_pytorch_tanh";
      auto alias_model = infinity::create_encoder(alias, w, device, dtype);
      alias_model->ready();
      auto alias_output = alias_model->forward({ids, types, positions, cu, offsets, 2, 5});
      TORCH_CHECK(at::equal(packed, alias_output),
                  "Nomic GELU alias changed checkpoint layout or outputs");
      for (int every : {0, 1}) {
        auto dense = o;
        dense.values["num_experts"] = "8";
        dense.values["n_inner"] = "48";
        dense.values["moe_every_n_layers"] = std::to_string(every);
        if (every)
          dense.values["moe_top_k"] = "2";
        auto dense_model = infinity::create_encoder(dense, w, device, dtype);
        dense_model->ready();
        auto actual = dense_model->forward({ids, types, positions, cu, offsets, 2, 5});
        TORCH_CHECK(actual.sizes() == packed.sizes() && at::isfinite(actual).all().item<bool>(),
                    "Dense Nomic expert metadata rejected valid inference");
        fp8_encoder(dense, w, {ids, types, positions, cu, offsets, 2, 5});
      }
    }
    if (std::string(type) == "new" || std::string(type) == "jina") {
      // The backbone has root keys. Classifier pooler selection is a separate
      // Candle contract: Jina uses bert.pooler; GTE prefers complete new.pooler.
      auto head_weights = w;
      linear(head_weights, "classifier", 3, H, device, dtype);
      linear(head_weights, "pooler.dense", H, H, device, dtype);
      const std::string preferred =
          std::string(type) == "new" ? "new.pooler.dense" : "bert.pooler.dense";
      linear(head_weights, preferred, H, H, device, dtype);
      auto head_model = infinity::create_encoder(o, head_weights, device, dtype);
      head_model->ready();
      auto cls = at::cat({packed.narrow(0, 0, 1), packed.narrow(0, 3, 1)}, 0);
      auto expected_head = [&](const std::string& key) {
        auto pooled = at::tanh(
            at::linear(cls, head_weights.at(key + ".weight"), head_weights.at(key + ".bias")));
        return at::linear(
            pooled, head_weights.at("classifier.weight"), head_weights.at("classifier.bias"));
      };
      auto scores = head_model->predict({ids, types, positions, cu, offsets, 2, 5}, false);
      TORCH_CHECK(at::equal(scores,
                            head_model->classify_hidden({ids, types, positions, cu, offsets, 2, 5},
                                                        packed)),
                  "Readout from existing hidden states changed sequence logits for ",
                  type);
      auto head_tolerance = device.is_cuda() ? .002 : 1e-6;
      TORCH_CHECK(at::allclose(scores, expected_head(preferred), 0, head_tolerance),
                  "Classifier pooler namespace/precedence mismatch for ",
                  type);
      head_weights.erase(preferred + ".bias");
      auto partial_model = infinity::create_encoder(o, head_weights, device, dtype);
      partial_model->ready();
      if (std::string(type) == "new") {
        auto fallback = partial_model->predict({ids, types, positions, cu, offsets, 2, 5}, false);
        TORCH_CHECK(at::allclose(fallback, expected_head("pooler.dense"), 0, head_tolerance),
                    "Incomplete GTE new.pooler must fall back to complete root pooler");
      } else {
        bool rejected = false;
        try {
          partial_model->predict({ids, types, positions, cu, offsets, 2, 5}, false);
        } catch (const c10::Error& error) {
          rejected = std::string(error.what()).find("Invalid Jina classifier pooler bias") !=
                     std::string::npos;
        }
        TORCH_CHECK(
            rejected,
            "Jina must reject missing bert.pooler bias instead of using unrelated root pooler");
        auto raw = head_model->predict({ids, types, positions, cu, offsets, 2, 5}, true);
        TORCH_CHECK(at::allclose(raw,
                                 at::linear(packed,
                                            head_weights.at("classifier.weight"),
                                            head_weights.at("classifier.bias")),
                                 0,
                                 head_tolerance),
                    "Jina token classifier must skip sequence pooler");
      }
    }
    std::vector<at::Tensor> separate;
    for (int b = 0; b < 2; ++b) {
      auto start = offsets[b], n = offsets[b + 1] - start;
      auto local_cu = at::tensor({0, n}, opts.dtype(at::kInt));
      separate.push_back(model->forward({ids.narrow(0, start, n),
                                         types.narrow(0, start, n),
                                         positions.narrow(0, start, n),
                                         local_cu,
                                         b ? long_offsets : short_offsets,
                                         1,
                                         n}));
    }
    auto error = (packed - at::cat(separate)).abs().max().item<float>();
    TORCH_CHECK(at::isfinite(packed).all().item<bool>(), "Nonfinite output for ", type);
    TORCH_CHECK(error < (device.is_cuda() ? .003 : 1e-5),
                "Packed sequence contamination for ",
                type,
                ": ",
                error);
    std::cout << type << " packed-vs-independent max error " << error << "\n";
  }
  abi_encoder_regression(device);
}
