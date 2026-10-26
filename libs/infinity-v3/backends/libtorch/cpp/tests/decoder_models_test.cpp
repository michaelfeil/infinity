#include "batching/prefix_cache.h"
#include "prefix_cache_test.h"
#include "helpers/prefix_linear.h"
#include "bridge.h"
// Native packed decoder regression tests; no Python runtime or padded tokens.
#include "models/decoder_models.h"
#include "batching/radix.h"
#include "batching/input_validation.h"
#include "helpers/workspace_admission.h"
#include "batching/cuda_graph.h"
#include <c10/core/InferenceMode.h>
#include <iostream>
#include <atomic>
#include <barrier>
#include <future>
#include <thread>
#include <ATen/Parallel.h>
#ifdef TEI_TORCH_CUDA_KERNELS
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include "kernels/cached_decoder_linear.h"
#include <c10/cuda/CUDAGuard.h>
#include <chrono>
#endif
using namespace infinity;

namespace {
void input_descriptor_test() {
  int64_t data[64]{};
  int32_t offsets[] = {0, 32, 64};
  float payload = 0;
  uint8_t mask = 1;
  TeiImage image{};
  image.pixels = &payload;
  image.rows = 32;
  image.patch_dim = 1536;
  image.grid[0] = 2;
  image.grid[1] = image.grid[2] = 4;
  image.merge_size = 2;
  image.token_start = 4;
  image.token_count = 8;
  TeiAudio audio{&payload, &mask, 4, 128, 2, 1};
  auto validate = [&](const TeiImage* images,
                      size_t ni,
                      const TeiAudio* audios,
                      size_t na,
                      int64_t maximum = 32) {
    return validate_input_descriptors(
        data, data, data, offsets, 2, maximum, nullptr, images, ni, audios, na);
  };
  TORCH_CHECK(validate(&image, 1, &audio, 1) == 64,
              "Valid temporal image/audio descriptors rejected");
  // max_sequence is an upper bound, not a padded input or required exact length.
  TORCH_CHECK(validate(nullptr, 0, nullptr, 0, 64) == 64,
              "Valid maximum sequence upper bound rejected");
  auto rejects = [&](auto call, const char* message) {
    bool failed = false;
    try {
      call();
    } catch (const c10::Error&) {
      failed = true;
    }
    TORCH_CHECK(failed, message);
  };
  rejects([&] { validate(nullptr, 1, nullptr, 0); }, "Null counted image array accepted");
  rejects([&] { validate(nullptr, 0, nullptr, 1); }, "Null counted audio array accepted");
  rejects([&] { validate(nullptr, 0, nullptr, 0, 31); }, "Undersized max_sequence accepted");
  rejects([&] { validate(nullptr, 0, nullptr, 0, 65); },
          "Maximum sequence beyond packed token count accepted");
  int64_t selection[] = {1, 0, 1};
  TORCH_CHECK(validate_output_indices(offsets, 2, selection, 3, selection, 3) == 96,
              "Repeated output selection rejected");
  TORCH_CHECK(expected_output_elements(3, 96, 16, 16) == 1584, "Output element count changed");
  rejects([&] { validate_output_indices(offsets, 2, nullptr, 1, nullptr, 0); },
          "Null pooled indices accepted");
  rejects([&] { validate_output_indices(offsets, 2, nullptr, 0, nullptr, 1); },
          "Null raw indices accepted");
  int64_t invalid = -1;
  rejects([&] { validate_output_indices(offsets, 2, &invalid, 1, nullptr, 0); },
          "Negative output index accepted");
  invalid = 2;
  rejects([&] { validate_output_indices(offsets, 2, nullptr, 0, &invalid, 1); },
          "Out-of-bounds output index accepted");
  rejects([&] { expected_output_elements(std::numeric_limits<size_t>::max(), 0, 16, 16); },
          "Overflowing output element count accepted");
  auto bad = image;
  bad.rows = std::numeric_limits<int64_t>::max();
  bad.grid[0] = bad.rows;
  rejects([&] { validate(&bad, 1, nullptr, 0); }, "Overflowing image grid accepted");
  bad = image;
  bad.patch_dim = std::numeric_limits<int64_t>::max();
  rejects([&] { validate(&bad, 1, nullptr, 0); }, "Overflowing image payload accepted");
  bad = image;
  bad.token_start = 30;
  rejects([&] { validate(&bad, 1, nullptr, 0); }, "Cross-sequence image span accepted");
  bad = image;
  bad.pixels = nullptr;
  rejects([&] { validate(&bad, 1, nullptr, 0); }, "Null image pixels accepted");
  auto bad_audio = audio;
  bad_audio.frames = -1;
  rejects([&] { validate(nullptr, 0, &bad_audio, 1); }, "Negative audio frames accepted");
  bad_audio = audio;
  bad_audio.mask = nullptr;
  rejects([&] { validate(nullptr, 0, &bad_audio, 1); }, "Null audio mask accepted");
  TeiImage gemma = image;
  gemma.rows = 36;
  gemma.patch_dim = 768;
  gemma.grid[0] = 1;
  gemma.grid[1] = gemma.grid[2] = 6;
  gemma.merge_size = 3;
  gemma.token_start = 40;
  gemma.token_count = 4;
  gemma.sequence_start = 32;
  TORCH_CHECK(validate_input_descriptors(
                  data, data, data, offsets, 2, 32, nullptr, &gemma, 1, nullptr, 0, true) == 64,
              "Valid Gemma pooled image rejected");
  auto wrong_sequence = gemma;
  wrong_sequence.sequence_start = 0;
  rejects(
      [&] {
        validate_input_descriptors(
            data, data, data, offsets, 2, 32, nullptr, &wrong_sequence, 1, nullptr, 0, true);
      },
      "Gemma image admitted keys from another packed sequence");
  wrong_sequence.sequence_start = 33;
  rejects(
      [&] {
        validate_input_descriptors(
            data, data, data, offsets, 2, 32, nullptr, &wrong_sequence, 1, nullptr, 0, true);
      },
      "Gemma image accepted a non-boundary sequence start");
  auto second_row = image;
  second_row.token_start = 40;
  second_row.sequence_start = 32;
  TORCH_CHECK(validate(&second_row, 1, nullptr, 0) == 64,
              "Valid second-row temporal image rejected");
  second_row.sequence_start = 0;
  rejects([&] { validate(&second_row, 1, nullptr, 0); },
          "Temporal image accepted another sequence start");
  rejects(
      [&] {
        validate_input_descriptors(
            data, data, data, offsets, 2, 32, nullptr, &image, 1, nullptr, 0, true);
      },
      "Multiple-frame Gemma image accepted");
  // Exercise the actual native bridge before model creation/any tensor upload.
  TeiConfig cfg{16, 4, 1, 24, 32, 64, 2, 1e-5, 0};
  auto handle = tei_create(&cfg, "cpu", 0);
  TORCH_CHECK(handle, tei_error());
  auto code = tei_forward(handle,
                          data,
                          data,
                          data,
                          offsets,
                          2,
                          32,
                          0,
                          nullptr,
                          0,
                          nullptr,
                          0,
                          nullptr,
                          0,
                          nullptr,
                          nullptr,
                          1,
                          nullptr,
                          0);
  TORCH_CHECK(code != 0 && std::string(tei_error()).find("counted image") != std::string::npos,
              "Native bridge did not reject malformed media before upload/model access");
  auto pending = tei_forward_submit(handle,
                                    data,
                                    data,
                                    data,
                                    offsets,
                                    2,
                                    32,
                                    0,
                                    nullptr,
                                    0,
                                    nullptr,
                                    0,
                                    std::numeric_limits<size_t>::max(),
                                    nullptr,
                                    nullptr,
                                    1,
                                    nullptr,
                                    0,
                                    nullptr);
  TORCH_CHECK(!pending && std::string(tei_error()).find("counted image") != std::string::npos,
              "Deferred bridge did not validate descriptors before output allocation");
  code = tei_forward(handle,
                     data,
                     data,
                     data,
                     offsets,
                     2,
                     32,
                     0,
                     nullptr,
                     1,
                     nullptr,
                     0,
                     nullptr,
                     0,
                     nullptr,
                     nullptr,
                     0,
                     nullptr,
                     0);
  TORCH_CHECK(code != 0 && std::string(tei_error()).find("counted pooled") != std::string::npos,
              "Native bridge did not reject missing output indices before model access");
  pending = tei_forward_submit(handle,
                               data,
                               data,
                               data,
                               offsets,
                               2,
                               32,
                               0,
                               nullptr,
                               0,
                               &invalid,
                               1,
                               std::numeric_limits<size_t>::max(),
                               nullptr,
                               nullptr,
                               0,
                               nullptr,
                               0,
                               nullptr);
  TORCH_CHECK(!pending && std::string(tei_error()).find("raw output index") != std::string::npos,
              "Deferred bridge did not reject output indices before output allocation");
  tei_destroy(handle);
}

Options config(const std::string& family, bool moe = false) {
  Options cfg;
  cfg.values = {{"model_type", family},
                {"hidden_size", "16"},
                {"num_attention_heads", "4"},
                {"num_key_value_heads", "2"},
                {"head_dim", "4"},
                {"intermediate_size", "24"},
                {"num_hidden_layers", "2"},
                {"vocab_size", "32"},
                {"max_position_embeddings", "32"},
                {"hidden_act", "silu"},
                {"rms_norm_eps", "0.000001"}};
  if (moe) {
    cfg.values["num_experts"] = "3";
    cfg.values["num_experts_per_tok"] = "2";
    cfg.values["moe_intermediate_size"] = "12";
  }
  return cfg;
}

Weights weights(const Options& cfg, bool fused = false) {
  Weights w;
  auto add = [&](const std::string& name, std::initializer_list<int64_t> shape, bool norm = false) {
    int64_t count = 1;
    for (auto n : shape)
      count *= n;
    float offset = 0;
    for (auto c : name)
      offset += c;
    auto data =
        at::sin(at::arange(count, at::TensorOptions().dtype(at::kFloat)) * .13 + offset * .01) *
        .15;
    if (norm)
      data += 1;
    w[name] = data.view(at::IntArrayRef(shape));
  };
  add("model.embed_tokens.weight", {32, 16});
  add("model.norm.weight", {16}, true);
  for (int layer = 0; layer < 2; ++layer) {
    auto p = "model.layers." + std::to_string(layer) + ".";
    add(p + "input_layernorm.weight", {16}, true);
    add(p + "post_attention_layernorm.weight", {16}, true);
    add(p + "self_attn.q_proj.weight", {16, 16});
    add(p + "self_attn.k_proj.weight", {8, 16});
    add(p + "self_attn.v_proj.weight", {8, 16});
    add(p + "self_attn.o_proj.weight", {16, 16});
    if (cfg.string("model_type") == "qwen2")
      for (auto name : {"q_proj", "k_proj", "v_proj"})
        add(p + "self_attn." + name + ".bias", {std::string(name) == "q_proj" ? 16 : 8});
    if (cfg.string("model_type") == "qwen3" || cfg.string("model_type") == "qwen3_moe") {
      add(p + "self_attn.q_norm.weight", {4}, true);
      add(p + "self_attn.k_norm.weight", {4}, true);
    }
    if (!cfg.integer("num_experts")) {
      const auto intermediate = cfg.integer("intermediate_size", 24);
      add(p + "mlp.gate_proj.weight", {intermediate, 16});
      add(p + "mlp.up_proj.weight", {intermediate, 16});
      add(p + "mlp.down_proj.weight", {16, intermediate});
    } else {
      add(p + "mlp.gate.weight", {3, 16});
      std::vector<at::Tensor> gu, down;
      for (int e = 0; e < 3; ++e) {
        auto ep = p + "mlp.experts." + std::to_string(e) + ".";
        add(ep + "gate_proj.weight", {12, 16});
        add(ep + "up_proj.weight", {12, 16});
        add(ep + "down_proj.weight", {16, 12});
        gu.push_back(at::cat({w.at(ep + "gate_proj.weight"), w.at(ep + "up_proj.weight")}, 0));
        down.push_back(w.at(ep + "down_proj.weight"));
      }
      if (fused) {
        w[p + "mlp.experts.gate_up_proj"] = at::stack(gu);
        w[p + "mlp.experts.down_proj"] = at::stack(down);
      }
    }
  }
  return w;
}

void output_capacity_bridge_test() {
  auto cfg = config("qwen3");
  auto w = weights(cfg);
  TeiConfig native_cfg{16, 4, 2, 24, 32, 32, 1, 1e-6, 0};
  auto handle = tei_create(&native_cfg, "cpu", 0);
  TORCH_CHECK(handle, tei_error());
  for (const auto& [key, value] : cfg.values)
    TORCH_CHECK(tei_option(handle, key.c_str(), value.c_str()) == 0, tei_error());
  TORCH_CHECK(tei_option(handle, "_decision.kind", "option_tokens") == 0 &&
                  tei_option(handle, "tie_word_embeddings", "true") == 0,
              tei_error());
  for (const auto& [key, value] : w)
    TORCH_CHECK(tei_weight(handle,
                           key.c_str(),
                           reinterpret_cast<const uint8_t*>(value.const_data_ptr()),
                           value.sizes().data(),
                           value.dim(),
                           0) == 0,
                tei_error());
  TORCH_CHECK(tei_option(handle, "_transient_max_bytes", "1") == 0 &&
                  tei_option(handle, "_device_memory_headroom_bytes", "9223372036854775807") == 0,
              tei_error());
  TORCH_CHECK(tei_ready(handle) == 0, tei_error());
  // Legacy estimate options must not reject valid requests or alter their results.
  // A one-byte cap rejected this same forward before the heuristic was removed.
  int64_t ids[] = {3, 4, 5, 6}, types[4]{}, positions[] = {0, 1, 0, 1}, pooled = 0;
  int32_t cumulative[] = {0, 2, 4};
  int64_t option_ids[] = {1, 2};
  TeiDecision valid_head[2]{};
  valid_head[0].kind = 1;
  valid_head[0].token_ids = option_ids;
  valid_head[0].token_count = 2;
  valid_head[1].kind = 3;
  float head_logits[3]{}, head_actions[2]{};
  TORCH_CHECK(tei_decide(handle,
                         ids,
                         types,
                         positions,
                         cumulative,
                         2,
                         2,
                         valid_head,
                         head_logits,
                         3,
                         head_actions,
                         nullptr,
                         nullptr,
                         0,
                         nullptr,
                         0) == 0,
              tei_error());
  for (auto value : head_logits)
    TORCH_CHECK(std::isfinite(value), "Typed response contains non-finite logits");
  for (const auto capacity : {size_t(17), std::numeric_limits<size_t>::max()}) {
    auto pending = tei_forward_submit(handle,
                                      ids,
                                      types,
                                      positions,
                                      cumulative,
                                      2,
                                      2,
                                      0,
                                      &pooled,
                                      1,
                                      nullptr,
                                      0,
                                      capacity,
                                      nullptr,
                                      nullptr,
                                      0,
                                      nullptr,
                                      0,
                                      nullptr);
    TORCH_CHECK(!pending &&
                    std::string(tei_error()).find("Deferred output capacity must exactly") !=
                        std::string::npos,
                "Deferred spare capacity reached CUDA/output allocation");
  }
  auto pending = tei_forward_submit(handle,
                                    ids,
                                    types,
                                    positions,
                                    cumulative,
                                    2,
                                    2,
                                    0,
                                    &pooled,
                                    1,
                                    nullptr,
                                    0,
                                    16,
                                    nullptr,
                                    nullptr,
                                    0,
                                    nullptr,
                                    0,
                                    nullptr);
  TORCH_CHECK(!pending && tei_last_error_kind() == 1 &&
                  std::string(tei_error()).find("Deferred output requires CUDA") !=
                      std::string::npos,
              "Exact deferred capacity failed metadata validation");
  float output[20];
  std::fill_n(output, 20, std::numeric_limits<float>::quiet_NaN());
  TORCH_CHECK(tei_forward(handle,
                          ids,
                          types,
                          positions,
                          cumulative,
                          2,
                          2,
                          0,
                          &pooled,
                          1,
                          nullptr,
                          0,
                          output,
                          20,
                          nullptr,
                          nullptr,
                          0,
                          nullptr,
                          0) == 0,
              tei_error());
  for (int i = 0; i < 16; ++i)
    TORCH_CHECK(std::isfinite(output[i]), "Valid response failed after capacity rejection");
  for (int i = 16; i < 20; ++i)
    TORCH_CHECK(std::isnan(output[i]), "Synchronous spare capacity was overwritten");
  // Oversized finite payload descriptors must be rejected without reading them.
  const auto fake_float = reinterpret_cast<const float*>(uintptr_t(1));
  const auto fake_mask = reinterpret_cast<const uint8_t*>(uintptr_t(1));
  TeiImage huge{};
  huge.pixels = fake_float;
  huge.rows = 1;
  huge.patch_dim = (512LL << 20) / sizeof(float) + 1;
  huge.grid[0] = huge.grid[1] = huge.grid[2] = huge.merge_size = huge.token_count = 1;
  auto budget_error = [&](int code) {
    TORCH_CHECK(code != 0 && tei_last_error_kind() == 2 &&
                    std::string(tei_error()).find("Native input upload exceeds") !=
                        std::string::npos,
                "Input budget did not reject before payload read/allocation");
  };
  budget_error(tei_forward(handle,
                           ids,
                           types,
                           positions,
                           cumulative,
                           2,
                           2,
                           0,
                           &pooled,
                           1,
                           nullptr,
                           0,
                           output,
                           20,
                           nullptr,
                           &huge,
                           1,
                           nullptr,
                           0));
  pending = tei_forward_submit(handle,
                               ids,
                               types,
                               positions,
                               cumulative,
                               2,
                               2,
                               0,
                               &pooled,
                               1,
                               nullptr,
                               0,
                               16,
                               nullptr,
                               &huge,
                               1,
                               nullptr,
                               0,
                               nullptr);
  budget_error(pending ? 0 : -1);
  budget_error(tei_decide(handle,
                          ids,
                          types,
                          positions,
                          cumulative,
                          2,
                          2,
                          nullptr,
                          nullptr,
                          0,
                          nullptr,
                          nullptr,
                          &huge,
                          1,
                          nullptr,
                          0));
  TeiAudio huge_audio{fake_float, fake_mask, 1, (512LL << 20) / sizeof(float) + 1, 0, 1};
  budget_error(tei_forward(handle,
                           ids,
                           types,
                           positions,
                           cumulative,
                           2,
                           2,
                           0,
                           &pooled,
                           1,
                           nullptr,
                           0,
                           output,
                           20,
                           nullptr,
                           nullptr,
                           0,
                           &huge_audio,
                           1));
  TORCH_CHECK(tei_option(handle, "_input_upload_max_bytes", "108") == 0, tei_error());
  auto fake_indices = reinterpret_cast<const int64_t*>(uintptr_t(1));
  TeiRadix fake_radix{fake_indices, fake_indices, fake_indices, fake_indices, fake_indices, 4};
  budget_error(tei_forward_radix(handle,
                                 ids,
                                 types,
                                 positions,
                                 cumulative,
                                 2,
                                 2,
                                 0,
                                 &pooled,
                                 1,
                                 nullptr,
                                 0,
                                 output,
                                 20,
                                 nullptr,
                                 nullptr,
                                 0,
                                 nullptr,
                                 0,
                                 &fake_radix));
  budget_error(tei_forward(handle,
                           ids,
                           types,
                           positions,
                           cumulative,
                           2,
                           2,
                           0,
                           &pooled,
                           1,
                           nullptr,
                           0,
                           output,
                           20,
                           fake_indices,
                           nullptr,
                           0,
                           nullptr,
                           0));
  std::fill_n(output, 20, std::numeric_limits<float>::quiet_NaN());
  TORCH_CHECK(tei_forward(handle,
                          ids,
                          types,
                          positions,
                          cumulative,
                          2,
                          2,
                          0,
                          &pooled,
                          1,
                          nullptr,
                          0,
                          output,
                          20,
                          nullptr,
                          nullptr,
                          0,
                          nullptr,
                          0) == 0,
              tei_error());
  TORCH_CHECK(tei_last_error_kind() == 0 && std::string(tei_error()).empty(),
              "Successful inference retained an admission error kind");
  for (int i = 0; i < 16; ++i)
    TORCH_CHECK(std::isfinite(output[i]), "Valid response failed after input budget rejection");
  uint64_t disabled_stats[7]{};
  TORCH_CHECK(tei_prefix_cache_stats(handle, disabled_stats, 7) == 0, tei_error());
  for (auto value : disabled_stats)
    TORCH_CHECK(value == 0, "Default configuration enabled the experimental prefix cache");
  TeiDecision huge_metadata{};
  huge_metadata.markers = fake_indices;
  huge_metadata.marker_count = std::numeric_limits<size_t>::max();
  bool metadata_rejected = false;
  try {
    validate_decision_metadata_budget(&huge_metadata, 1, 512 << 20);
  } catch (const MemoryAdmissionError&) {
    metadata_rejected = true;
  }
  TORCH_CHECK(metadata_rejected, "Decision metadata overflow was copied before admission");
  TORCH_CHECK(tei_option(handle, "_input_upload_max_bytes", "0") == 0, tei_error());
  TORCH_CHECK(tei_forward(handle,
                          ids,
                          types,
                          positions,
                          cumulative,
                          2,
                          2,
                          0,
                          &pooled,
                          1,
                          nullptr,
                          0,
                          output,
                          20,
                          nullptr,
                          nullptr,
                          0,
                          nullptr,
                          0) != 0,
              "Zero input budget accepted");
  tei_destroy(handle);
}

at::Tensor run(Model& model, std::vector<int64_t> ids, std::vector<int32_t> offsets) {
  std::vector<int64_t> positions;
  int64_t maximum = 0;
  for (size_t row = 0; row + 1 < offsets.size(); ++row) {
    auto len = offsets[row + 1] - offsets[row];
    maximum = std::max(maximum, int64_t(len));
    for (int n = 0; n < len; ++n)
      positions.push_back(n);
  }
  auto opts = at::TensorOptions().dtype(at::kLong);
  PackedInput input{at::from_blob(ids.data(), {int64_t(ids.size())}, opts),
                    at::zeros({int64_t(ids.size())}, opts),
                    at::from_blob(positions.data(), {int64_t(positions.size())}, opts),
                    at::from_blob(offsets.data(), {int64_t(offsets.size())}, opts.dtype(at::kInt)),
                    offsets.data(),
                    int64_t(offsets.size() - 1),
                    maximum};
  return model.forward(input);
}

void check_close(const at::Tensor& a, const at::Tensor& b) {
  auto error = (a - b).abs().max().item<float>();
  TORCH_CHECK(error < 2e-5, "Decoder parity error ", error);
}

void selected_classifier_rows_test() {
  for (auto family : {"qwen2", "qwen3"})
    for (bool padding : {false, true}) {
      auto cfg = config(family);
      cfg.values["architectures.0"] = std::string(family) == "qwen2"
                                          ? "Qwen2ForSequenceClassification"
                                          : "Qwen3ForSequenceClassification";
      cfg.values["use_bidirectional_attention"] = "true";
      cfg.values["num_labels"] = "3";
      cfg.values["num_attention_heads"] = "2";
      cfg.values["num_key_value_heads"] = "1";
      cfg.values["head_dim"] = "8";
      if (padding)
        cfg.values["pad_token_id"] = "0";
      auto tensors = weights(cfg);
      tensors["score.weight"] = at::arange(48, at::kFloat).view({3, 16}) * .013;
      tensors["score.bias"] = at::tensor({.1f, -.2f, .3f});
      if (std::string(family) == "qwen3")
        for (int layer = 0; layer < 2; ++layer)
          for (auto norm : {"q_norm", "k_norm"})
            tensors["model.layers." + std::to_string(layer) + ".self_attn." + norm + ".weight"] =
                at::ones({8});
      for (auto& [name, value] : tensors)
        value = value.to(at::kCUDA, at::kBFloat16);
      auto model = create_decoder(cfg, tensors, c10::Device("cuda:0"), at::kBFloat16);
      model->ready();
      // Ragged lengths, an all-padding sequence and trailing padding in the last row.
      std::vector<int32_t> offsets{0, 1, 4, 9, 16};
      auto opts = at::TensorOptions().device(at::kCUDA).dtype(at::kLong);
      PackedInput input{at::tensor({0, 3, 4, 5, 0, 0, 0, 0, 0, 3, 6, 7, 8, 0, 0, 0}, opts),
                        at::zeros({16}, opts),
                        at::tensor({0, 0, 1, 2, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6}, opts),
                        at::tensor(offsets, opts.dtype(at::kInt)),
                        offsets.data(),
                        4,
                        7};
      auto expected = model->classify_hidden(input, model->forward(input));
      auto actual = model->predict(input, false);
      check_close(actual, expected);
    }
}

void bidirectional_classifier_test(c10::Device device, at::ScalarType dtype) {
  auto cfg = config("llama");
  cfg.values["architectures.0"] = "LlamaBidirectionalForSequenceClassification";
  cfg.values["use_bidirectional_attention"] = "true";
  cfg.values["num_labels"] = "1";
  cfg.values["pooling"] = "avg";
  cfg.values["temperature"] = "0.7";
  if (device.is_cuda()) {
    cfg.values["num_attention_heads"] = "2";
    cfg.values["num_key_value_heads"] = "1";
    cfg.values["head_dim"] = "8";
  }
  auto tensors = weights(cfg);
  tensors["score.weight"] = at::arange(16, at::kFloat).view({1, 16}) * .013;
  for (auto& [name, value] : tensors)
    value = value.to(device, dtype);
  auto score = tensors.at("score.weight");
  auto model = create_decoder(cfg, tensors, device, dtype);
  model->ready();
  TORCH_CHECK(model->classification_width() == 1 && !model->supports_prefix_cache(1),
              "Bidirectional classifier head or cache eligibility changed");
  std::vector<int32_t> offsets{0, 3, 7};
  auto opts = at::TensorOptions().dtype(at::kLong).device(device);
  PackedInput input{at::tensor({3, 4, 5, 3, 6, 7, 8}, opts),
                    at::zeros({7}, opts),
                    at::tensor({0, 1, 2, 0, 1, 2, 3}, opts),
                    at::tensor(offsets, opts.dtype(at::kInt)),
                    offsets.data(),
                    2,
                    4};
  auto hidden = model->forward(input);
  std::vector<at::Tensor> pooled;
  for (int row = 0; row < 2; ++row) {
    auto tokens = hidden.narrow(0, offsets[row], offsets[row + 1] - offsets[row]);
    pooled.push_back(tokens.sum(0) / double(tokens.size(0)));
  }
  auto expected = at::linear(at::stack(pooled), score) / .7;
  TORCH_CHECK(at::allclose(model->predict(input, false), expected, .001, .001),
              "Bidirectional classifier pooling or temperature changed");
}

void decoder_token_classifier_test(c10::Device device, at::ScalarType dtype) {
  for (auto family : {"llama", "qwen2", "qwen3"})
    for (bool bias : {false, true})
      for (bool bidirectional : {false, true}) {
        auto cfg = config(family);
        cfg.values["architectures.0"] =
            std::string(family) == "llama"   ? "LlamaForTokenClassification"
            : std::string(family) == "qwen2" ? "Qwen2ForTokenClassification"
                                             : "Qwen3ForTokenClassification";
        cfg.values["num_labels"] = "3";
        if (!bias)
          cfg.values["token_classification_bias"] = "false";
        cfg.values["is_causal"] = bidirectional ? "false" : "true";
        for (int i = 0; i < 3; ++i)
          cfg.values["id2label." + std::to_string(i)] = std::to_string(i);
        if (device.is_cuda()) {
          cfg.values["num_attention_heads"] = "2";
          cfg.values["num_key_value_heads"] = "1";
          cfg.values["head_dim"] = "8";
        }
        auto w = weights(cfg);
        if (device.is_cuda() && std::string(family) == "qwen3")
          for (int i = 0; i < 2; ++i)
            for (auto norm : {"q_norm", "k_norm"})
              w["model.layers." + std::to_string(i) + ".self_attn." + norm + ".weight"] =
                  at::ones({8});
        w["score.weight"] = at::arange(48, at::kFloat).view({3, 16}) * .013;
        if (bias)
          w["score.bias"] = at::tensor({.1f, -.2f, .3f});
        for (auto& [name, value] : w)
          value = value.to(device, dtype);
        auto checkpoint = w, backbone_weights = w;
        auto backbone_cfg = cfg;
        backbone_cfg.values.erase("architectures.0");
        auto backbone = create_decoder(backbone_cfg, backbone_weights, device, dtype);
        backbone->ready();
        auto model = create_decoder(cfg, w, device, dtype);
        model->ready();
        TORCH_CHECK(model->classification_width() == 3 && !model->supports_prefix_cache(1),
                    "Token head advertised sequence cache readout");
        std::vector<int32_t> offsets{0, 3, 7};
        auto opts = at::TensorOptions().dtype(at::kLong).device(device);
        PackedInput input{at::tensor({3, 4, 5, 3, 6, 7, 8}, opts),
                          at::zeros({7}, opts),
                          at::tensor({0, 1, 2, 0, 1, 2, 3}, opts),
                          at::tensor(offsets, opts.dtype(at::kInt)),
                          offsets.data(),
                          2,
                          4};
        auto actual = model->predict(input, true);
        check_close(actual,
                    at::linear(backbone->forward(input),
                               checkpoint.at("score.weight"),
                               bias ? checkpoint.at("score.bias") : at::Tensor()));
        for (int row = 0; row < 2; ++row) {
          auto start = offsets[row], length = offsets[row + 1] - start;
          std::vector<int32_t> one_offsets{0, length};
          PackedInput one{input.ids.narrow(0, start, length),
                          input.types.narrow(0, start, length),
                          input.positions.narrow(0, start, length),
                          at::tensor(one_offsets, opts.dtype(at::kInt)),
                          one_offsets.data(),
                          1,
                          length};
          check_close(actual.narrow(0, start, length), model->predict(one, true));
        }
        TORCH_CHECK(model->predict(input, false).sizes() == at::IntArrayRef({2, 3}),
                    "Classifier warmup failed");
        if (!device.is_cpu())
          continue;
        auto rejects = [&](Options bad, Weights tensors) {
          bool failed = false;
          try {
            auto invalid = create_decoder(bad, tensors, device, dtype);
            invalid->ready();
          } catch (const c10::Error&) {
            failed = true;
          }
          TORCH_CHECK(failed, "Malformed decoder token classifier accepted");
        };
        auto missing = checkpoint;
        missing.erase("score.weight");
        rejects(cfg, missing);
        auto bad = cfg;
        bad.values["architectures.1"] = "Qwen2Model";
        rejects(bad, checkpoint);
        bad = cfg;
        bad.values["model_type"] = "mistral";
        rejects(bad, checkpoint);
        bad = cfg;
        bad.values["num_labels"] = "2";
        rejects(bad, checkpoint);
        bad = cfg;
        bad.values["token_classification_bias"] = bias ? "false" : "true";
        rejects(bad, checkpoint);
        auto wrong = checkpoint;
        wrong["score.weight"] = at::zeros({3, 8});
        rejects(cfg, wrong);
        if (bias) {
          wrong = checkpoint;
          wrong["score.bias"] = at::zeros({2});
          rejects(cfg, wrong);
        }
        bad = cfg;
        bad.values["use_linear_output_projection"] = "true";
        bad.values["linear_output_size"] = "16";
        wrong = checkpoint;
        wrong["linear_output_projection.weight"] = at::eye(16);
        rejects(bad, wrong);
      }
}

void radix_decoder_test(Model& model) {
  auto options = at::TensorOptions().dtype(at::kLong);
  std::vector<int32_t> offsets = {0, 3, 7, 10};
  PackedInput input{at::tensor({3, 4, 5, 3, 4, 6, 7, 3, 4, 5}, options),
                    at::zeros({10}, options),
                    at::tensor({0, 1, 2, 0, 1, 2, 3, 0, 1, 2}, options),
                    at::tensor({0, 3, 7, 10}, options.dtype(at::kInt)),
                    offsets.data(),
                    3,
                    4};
  auto expected = model.forward(input);
  input.compact_ids = at::tensor({3, 4, 5, 6, 7}, options);
  input.compact_positions = at::tensor({0, 1, 2, 2, 3}, options);
  input.compact_types = at::zeros({5}, options);
  input.scatter_unfold = at::tensor({0, 1, 2, 0, 1, 3, 4, 0, 1, 2}, options);
  input.fold_gather = at::tensor({0, 1, 2, 5, 6}, options);
  validate_radix(input, true);
  if (!model.supports_radix()) {
    bool rejected = false;
    try {
      model.forward(input);
    } catch (const c10::Error&) {
      rejected = true;
    }
    TORCH_CHECK(rejected, "Unsupported bidirectional folding was silently ignored");
    return;
  }
  check_close(model.forward(input), expected);
  check_prefix_cache(model, input);
  TORCH_CHECK(radix_ids(input).numel() == 5 && input.ids.numel() == 10,
              "Radix retained duplicate token rows");
  // Both tails contain token 9 at position 3, but only the first two requests
  // have the same complete causal history after resuming cached prefixes.
  std::vector<int32_t> tails = {0, 4, 8, 12};
  PackedInput branched{at::tensor({3, 4, 5, 9, 3, 4, 5, 9, 6, 4, 5, 9}, options),
                       at::zeros({12}, options),
                       at::tensor({0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3}, options),
                       at::tensor(tails, options.dtype(at::kInt)),
                       tails.data(),
                       3,
                       4};
  branched.compact_ids = at::tensor({3, 4, 5, 9, 6, 4, 5, 9}, options);
  branched.compact_positions = at::tensor({0, 1, 2, 3, 0, 1, 2, 3}, options);
  branched.compact_types = at::zeros({8}, options);
  branched.scatter_unfold = at::tensor({0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7}, options);
  branched.fold_gather = at::tensor({0, 1, 2, 3, 8, 9, 10, 11}, options);
  validate_radix(branched, true);
  check_prefix_cache(model, branched);
}

void cached_decoder_test(Model& model) {
  PrefixCache cache(1 << 20, 64, 8);
  auto cached = [&](std::vector<int64_t> ids, std::vector<int32_t> offsets) {
    auto options = at::TensorOptions().dtype(at::kLong);
    std::vector<int64_t> positions;
    for (size_t row = 0; row + 1 < offsets.size(); ++row)
      for (int64_t t = 0; t < offsets[row + 1] - offsets[row]; ++t)
        positions.push_back(t);
    auto tensor = at::tensor(ids, options);
    PackedInput input{tensor,
                      at::zeros_like(tensor),
                      at::tensor(positions, options),
                      at::tensor(offsets, at::TensorOptions().dtype(at::kInt)),
                      offsets.data(),
                      int64_t(offsets.size() - 1),
                      0};
    for (size_t row = 0; row + 1 < offsets.size(); ++row)
      input.max_sequence = std::max<int64_t>(input.max_sequence, offsets[row + 1] - offsets[row]);
    return cache.run(model, input, 0);
  };
  auto original = cached({3, 4, 5, 3, 4, 8, 9}, {0, 3, 7}).clone();
  check_close(original, run(model, {3, 4, 5, 3, 4, 8, 9}, {0, 3, 7}));
  check_close(cached({3, 4, 5, 3, 4, 8, 9}, {0, 3, 7}), original);
  check_close(cached({3, 4, 5, 9, 3, 4, 8}, {0, 4, 7}),
              run(model, {3, 4, 5, 9, 3, 4, 8}, {0, 4, 7}));
  // Equal suffixes/positions have different complete parent contexts.
  check_close(cached({6, 4, 5, 9, 3, 4, 5, 9}, {0, 4, 8}),
              run(model, {6, 4, 5, 9, 3, 4, 5, 9}, {0, 4, 8}));
  TORCH_CHECK(cache.stats().hits > 0 && cache.stats().reused_tokens > 0,
              "Prefix manager never reused actual causal state");
  TORCH_CHECK(cache.bytes() <= 1 << 20, "Prefix manager exceeded tensor budget");
  check_close(original, run(model, {3, 4, 5, 3, 4, 8, 9}, {0, 3, 7}));
  cache.clear();
  TORCH_CHECK(cache.bytes() == 0, "Cleared prefix entries retain resident tensors");
}

void cache_concurrency_test(Model& model, int64_t length = 12);

void cache_admission_test(Model& model) {
  struct Observed final : Model {
    Model& inner;
    mutable bool capturing = false;
    mutable size_t calls = 0;
    bool fail = false;

    explicit Observed(Model& model) : inner(model) {}

    void ready() override {}

    bool supports_prefix_cache(int operation) const override {
      return inner.supports_prefix_cache(operation);
    }

    int64_t prefix_layers() const override {
      return inner.prefix_layers();
    }

    uint64_t prefix_payload_bytes(int64_t tokens) const override {
      return inner.prefix_payload_bytes(tokens);
    }

    at::Tensor forward(const PackedInput& input) const override {
      ++calls;
      capturing = input.prefix && std::any_of(input.prefix->writers.begin(),
                                              input.prefix->writers.end(),
                                              [](const auto& writer) { return bool(writer); });
      auto result = inner.forward(input);
      TORCH_CHECK(!fail, "Injected failure after snapshot creation");
      return result;
    }
  } observed(model);

  auto options = at::TensorOptions().dtype(at::kLong);
  std::vector<int32_t> ends = {0, 4};
  PackedInput input{at::tensor({3, 4, 5, 6}, options),
                    at::zeros({4}, options),
                    at::arange(4, options),
                    at::tensor(ends, options.dtype(at::kInt)),
                    ends.data(),
                    1,
                    4};
  auto reference = model.forward(input);
  PrefixCache measure(1 << 20, 64, 1);
  check_close(measure.run(observed, input, 0), reference);
  const auto footprint = measure.bytes();
  TORCH_CHECK(footprint > 0 && observed.capturing, "Snapshot footprint was not measured");
  bool unsupported = false;
  try {
    measure.run(observed, input, 1);
  } catch (const c10::Error&) {
    unsupported = true;
  }
  TORCH_CHECK(unsupported && measure.bytes() == footprint,
              "Unsupported task discarded valid backbone state");
  PrefixCache too_small(footprint - 1, 64, 1);
  check_close(too_small.run(observed, input, 0), reference);
  TORCH_CHECK(!observed.capturing && too_small.bytes() == 0 && too_small.stats().admissions == 0,
              "Insufficient capacity still allocated partial snapshots");
  PrefixCache exact(footprint, 64, 1);
  check_close(exact.run(observed, input, 0), reference);
  TORCH_CHECK(observed.capturing && exact.bytes() == footprint,
              "Exact byte budget rejected a complete snapshot");
  check_close(exact.run(observed, input, 0), reference);
  TORCH_CHECK(exact.stats().hits == 1, "Exact budget prevented reuse");
  Observed other(model);
  auto hits = measure.stats().hits;
  check_close(measure.run(other, input, 0), reference);
  TORCH_CHECK(measure.stats().hits == hits,
              "A different loaded model instance reused old prefix state");
  check_close(measure.run(observed, input, 0), reference);
  const auto tf32 = at::globalContext().allowTF32CuBLAS();
  at::globalContext().setAllowTF32CuBLAS(!tf32);
  auto changed = measure.run(observed, input, 0);
  at::globalContext().setAllowTF32CuBLAS(tf32);
  check_close(changed, reference);
  TORCH_CHECK(measure.stats().hits == hits, "Arithmetic policy change retained old prefix state");
  check_close(measure.run(observed, input, 0), reference);
  hits = measure.stats().hits;
  auto flash = at::globalContext().userEnabledFlashSDP();
  at::globalContext().setSDPUseFlash(!flash);
  auto attention_changed = measure.run(observed, input, 0);
  at::globalContext().setSDPUseFlash(flash);
  check_close(attention_changed, reference);
  TORCH_CHECK(measure.stats().hits == hits, "Attention backend policy retained old prefix state");
  auto check_cudnn_policy = [&](auto change, auto restore) {
    check_close(measure.run(observed, input, 0), reference);
    const auto before = measure.stats().hits;
    change();
    at::Tensor changed;
    try {
      changed = measure.run(observed, input, 0);
    } catch (...) {
      restore();
      throw;
    }
    restore();
    check_close(changed, reference);
    TORCH_CHECK(measure.stats().hits == before, "cuDNN algorithm policy reused old prefix state");
    check_close(measure.run(observed, input, 0), reference);
    const auto restored = measure.stats().hits;
    check_close(measure.run(observed, input, 0), reference);
    TORCH_CHECK(measure.stats().hits == restored + 1,
                "Restored cuDNN policy prevented normal prefix reuse");
  };
  auto& global = at::globalContext();
  const auto deterministic = global.deterministicCuDNN();
  check_cudnn_policy([&] { global.setDeterministicCuDNN(!deterministic); },
                     [&] { global.setDeterministicCuDNN(deterministic); });
  const auto benchmark = global.benchmarkCuDNN();
  check_cudnn_policy([&] { global.setBenchmarkCuDNN(!benchmark); },
                     [&] { global.setBenchmarkCuDNN(benchmark); });
  const auto benchmark_limit = global.benchmarkLimitCuDNN();
  check_cudnn_policy([&] { global.setBenchmarkLimitCuDNN(benchmark_limit == 0 ? 1 : 0); },
                     [&] { global.setBenchmarkLimitCuDNN(benchmark_limit); });
  input.ids = at::tensor({8, 4, 5, 6}, options);
  observed.fail = true;
  bool failed = false;
  try {
    measure.run(observed, input, 0);
  } catch (const c10::Error&) {
    failed = true;
  }
  TORCH_CHECK(failed && observed.capturing && measure.bytes() == 0,
              "Failure retained resident or reserved cache memory");
  input.ids = at::tensor({3, 4, 5, 6}, options);
  observed.fail = false;
  check_close(measure.run(observed, input, 0), reference);
  // A pinned reader may lose its lookup slot while its immutable payload stays
  // alive. Capacity one must still admit a longer checkpoint after the run.
  auto extended = [&](int64_t tokens) {
    ends[1] = tokens;
    input.ids = at::arange(3, 3 + tokens, options);
    input.types = at::zeros_like(input.ids);
    input.positions = at::arange(tokens, options);
    input.cumulative = at::tensor(ends, options.dtype(at::kInt));
    input.max_sequence = tokens;
    check_close(measure.run(observed, input, 0), model.forward(input));
  };
  extended(5);
  auto reused = measure.stats().reused_tokens;
  extended(6);
  TORCH_CHECK(measure.stats().reused_tokens - reused == 5,
              "Capacity-one cache did not advance its retained prefix");
  // Production admission does not capture one-off inputs. Exact full results
  // avoid the backbone but still own writable output storage independently.
  ends[1] = 4;
  input.ids = at::tensor({3, 4, 5, 6}, options);
  input.types = at::zeros_like(input.ids);
  input.positions = at::arange(4, options);
  input.cumulative = at::tensor(ends, options.dtype(at::kInt));
  input.max_sequence = 4;
  PrefixCache delayed(1 << 20, 64, 8, 2);
  CudaGraphCache graphs;
  check_close(delayed.run(observed, input, 0, &graphs), reference);
  TORCH_CHECK(!observed.capturing && delayed.bytes() == 0,
              "First observation captured a one-off prefix");
  check_close(delayed.run(observed, input, 0, &graphs), reference);
  TORCH_CHECK(observed.capturing && delayed.bytes() > 0, "Repeated causal stem was never admitted");
  auto before_complete = observed.calls;
  auto owned = delayed.run(observed, input, 0, &graphs);
  check_close(owned, reference);
  owned.zero_();
  check_close(delayed.run(observed, input, 0, &graphs), reference);
  TORCH_CHECK(observed.calls == before_complete, "Exact complete input reran the backbone");
  input.max_sequence = 5;
  check_close(delayed.run(observed, input, 0, &graphs), reference);
  TORCH_CHECK(observed.calls == before_complete + 1,
              "Complete-result reuse ignored original attention geometry");
  delayed.clear();
  input.max_sequence = 4;
  for (int64_t token = 8; token < 24; ++token) {
    input.ids = at::tensor(std::vector<int64_t>{token, 4, 5, 6}, options);
    check_close(delayed.run(observed, input, 0, &graphs), model.forward(input));
  }
  TORCH_CHECK(delayed.bytes() == 0 && delayed.stats().admissions == 1,
              "One-off churn consumed cache tensor storage");
  // Mixed observation counts are per row: permitting capture of warmed A must
  // not observe first-seen B twice and admit both on this same request.
  PrefixCache mixed(1 << 20, 64, 8, 2);
  input.ids = at::tensor({3, 4, 5, 6}, options);
  mixed.run(observed, input, 0);
  std::vector<int32_t> mixed_ends = {0, 4, 8};
  PackedInput batch{at::tensor({3, 4, 5, 6, 9, 4, 5, 6}, options),
                    at::zeros({8}, options),
                    at::tensor({0, 1, 2, 3, 0, 1, 2, 3}, options),
                    at::tensor(mixed_ends, options.dtype(at::kInt)),
                    mixed_ends.data(),
                    2,
                    4};
  check_close(mixed.run(observed, batch, 0), model.forward(batch));
  TORCH_CHECK(mixed.stats().admissions == 1, "Mixed batch admitted a first-seen row");
  input.ids = at::tensor({9, 4, 5, 6}, options);
  mixed.run(observed, input, 0);
  TORCH_CHECK(mixed.stats().admissions == 2, "Mixed first observation was lost");
  // Changing documents admit their shared stem once; independent full-document
  // observations still promote complete repeats of interleaved documents.
  PrefixCache short_stem(1 << 20, 64, 8, 2);
  std::vector<int32_t> short_ends = {0, 12};
  PackedInput short_input{at::tensor({3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}, options),
                          at::zeros({12}, options),
                          at::arange(12, options),
                          at::tensor(short_ends, options.dtype(at::kInt)),
                          short_ends.data(),
                          1,
                          12};
  check_close(short_stem.run(observed, short_input, 0), model.forward(short_input));
  short_input.ids.index_put_({8}, 20);
  check_close(short_stem.run(observed, short_input, 0), model.forward(short_input));
  const auto short_bytes = short_stem.bytes();
  TORCH_CHECK(short_stem.stats().admissions == 1,
              "Changing document tail hid a repeated short stem");
  short_input.ids.index_put_({8}, 21);
  check_close(short_stem.run(observed, short_input, 0), model.forward(short_input));
  TORCH_CHECK(short_stem.stats().reused_tokens == 8 && short_stem.stats().admissions == 1 &&
                  short_stem.bytes() == short_bytes,
              "Short stem failed reuse or kept capturing changing tails");
  short_stem.clear();
  short_stem.run(observed, short_input, 0);
  TORCH_CHECK(short_stem.bytes() == 0 && short_stem.stats().admissions == 1,
              "Clear retained short-stem admission history");
  PrefixCache typed(1 << 20, 64, 8, 2);
  typed.run(observed, short_input, 0);
  auto changed_type = short_input;
  changed_type.types = at::ones_like(short_input.types);
  typed.run(observed, changed_type, 0);
  TORCH_CHECK(typed.bytes() == 0, "Short-stem admission ignored token types");
  typed.run(observed, changed_type, 0);
  TORCH_CHECK(typed.stats().admissions == 1, "Repeated typed document was never admitted");
  std::vector<int32_t> paired_ends = {0, 12, 24};
  auto pair = short_input;
  pair.ids = at::cat({short_input.ids, short_input.ids.clone()}, 0);
  pair.ids.index_put_({20}, 22);
  pair.types = at::zeros({24}, options);
  pair.positions = at::cat({short_input.positions, short_input.positions}, 0);
  pair.cumulative = at::tensor(paired_ends, options.dtype(at::kInt));
  pair.offsets = paired_ends.data();
  pair.batch = 2;
  PrefixCache paired(1 << 20, 64, 8, 2);
  check_close(paired.run(observed, pair, 0), model.forward(pair));
  TORCH_CHECK(paired.stats().admissions == 1,
              "Identical short stems in one batch duplicated snapshots");
  check_close(paired.run(observed, pair, 0), model.forward(pair));
  const auto paired_calls = observed.calls;
  check_close(paired.run(observed, pair, 0), model.forward(pair));
  TORCH_CHECK(observed.calls == paired_calls,
              "Short-stem history prevented complete repeats of interleaved documents");
  // Distinct questions may share the same role header. Interleaving them must
  // retain the longer question prefix, with its own required observation count.
  PrefixCache questions(1 << 20, 64, 8, 3);
  std::vector<int32_t> question_ends{0, 16};
  auto question = short_input;
  question.types = at::zeros({16}, options);
  question.positions = at::arange(16, options);
  question.cumulative = at::tensor(question_ends, options.dtype(at::kInt));
  question.offsets = question_ends.data();
  question.max_sequence = 16;
  auto ask = [&](int topic, int document) {
    question.ids = at::tensor({3,
                               4,
                               5,
                               6,
                               7,
                               8,
                               9,
                               10,
                               topic,
                               topic,
                               topic,
                               topic,
                               document,
                               document,
                               document,
                               document},
                              options);
    check_close(questions.run(observed, question, 0), model.forward(question));
  };
  ask(11, 16);
  ask(12, 17);
  ask(11, 18);
  TORCH_CHECK(questions.stats().admissions == 1,
              "Question prefix was admitted without three matching observations");
  ask(12, 19);
  ask(11, 20);
  TORCH_CHECK(questions.stats().admissions == 2,
              "Interleaved questions lost their longer shared prefix");
  auto question_reused = questions.stats().reused_tokens;
  ask(11, 21);
  TORCH_CHECK(questions.stats().reused_tokens - question_reused == 12,
              "Question reused only the common role header");
  // A sampled64-token stem must not promote every changing document into a
  // complete snapshot. Repeated complete documents must still bypass the model.
  auto long_config = config("qwen3");
  long_config.values["max_position_embeddings"] = "256";
  auto long_weights = weights(long_config);
  auto long_model = create_decoder(long_config, long_weights, c10::Device("cpu"), at::kFloat);
  long_model->ready();
  Observed long_observed(*long_model);
  std::vector<int32_t> long_ends = {0, 128};
  PackedInput long_input{at::arange(128, options).remainder(29) + 1,
                         at::zeros({128}, options),
                         at::arange(128, options),
                         at::tensor(long_ends, options.dtype(at::kInt)),
                         long_ends.data(),
                         1,
                         128};
  PrefixCache stable(1 << 20, 256, 4, 2);
  for (int i = 0; i < 2; ++i)
    check_close(stable.run(long_observed, long_input, 0), long_model->forward(long_input));
  const auto seeded = stable.stats().admissions;
  for (int document = 16; document < 26; ++document) {
    long_input.ids.narrow(0, 64, 64).fill_(document);
    check_close(stable.run(long_observed, long_input, 0), long_model->forward(long_input));
  }
  TORCH_CHECK(stable.stats().admissions == seeded && stable.stats().evictions == 0,
              "Changing documents kept admitting whole snapshots beyond the shared64-token stem");
  check_close(stable.run(long_observed, long_input, 0), long_model->forward(long_input));
  const auto complete_calls = long_observed.calls;
  check_close(stable.run(long_observed, long_input, 0), long_model->forward(long_input));
  TORCH_CHECK(long_observed.calls == complete_calls,
              "Repeated long document never promoted to complete readout");
  // Longer questions under a common64-token header count their own eight-token
  // descendants. The threshold and clear apply to those descendants too.
  PrefixCache descendants(1 << 20, 256, 8, 3);
  auto ask_long = [&](int topic, int document) {
    long_input.ids.narrow(0, 64, 32).fill_(topic);
    long_input.ids.narrow(0, 96, 32).fill_(document);
    check_close(descendants.run(long_observed, long_input, 0), long_model->forward(long_input));
  };
  for (int i = 0; i < 3; ++i)
    ask_long(11, 15);
  const auto root_admissions = descendants.stats().admissions;
  ask_long(12, 16);
  ask_long(12, 17);
  TORCH_CHECK(descendants.stats().admissions == root_admissions,
              "Longer question ignored its observation threshold");
  ask_long(12, 18);
  const auto prior_reused = descendants.stats().reused_tokens;
  ask_long(12, 19);
  TORCH_CHECK(descendants.stats().reused_tokens - prior_reused == 96,
              "Longer question reused only its common64-token header");
  descendants.clear();
  for (int i = 0; i < 3; ++i)
    ask_long(11, 15);
  const auto cleared_admissions = descendants.stats().admissions;
  ask_long(12, 20);
  TORCH_CHECK(descendants.stats().admissions == cleared_admissions,
              "Clear retained long-question observation counts");
  // Growing the longer-prefix table must preserve the first observation above.
  std::vector<int32_t> grown_ends = {0, 256};
  auto grown = long_input;
  grown.ids = at::cat({long_input.ids, at::full({128}, 20, options)}, 0);
  grown.types = at::zeros({256}, options);
  grown.positions = at::arange(256, options);
  grown.cumulative = at::tensor(grown_ends, options.dtype(at::kInt));
  grown.offsets = grown_ends.data();
  grown.max_sequence = 256;
  for (int document : {21, 22}) {
    grown.ids.narrow(0, 96, 160).fill_(document);
    check_close(descendants.run(long_observed, grown, 0), long_model->forward(grown));
  }
  const auto grown_reused = descendants.stats().reused_tokens;
  grown.ids.narrow(0, 96, 160).fill_(23);
  check_close(descendants.run(long_observed, grown, 0), long_model->forward(grown));
  TORCH_CHECK(descendants.stats().reused_tokens - grown_reused == 96,
              "Counter growth discarded a long-question observation");
  cache_concurrency_test(*long_model, 128);
  auto malformed = input;
  malformed.host_ids = input.ids.narrow(0, 0, 3);
  malformed.host_types = input.types.narrow(0, 0, 3);
  malformed.host_positions = input.positions.narrow(0, 0, 3);
  bool invalid_host = false;
  try {
    mixed.run(observed, malformed, 0);
  } catch (const c10::Error&) {
    invalid_host = true;
  }
  TORCH_CHECK(invalid_host && mixed.bytes() == 0,
              "Invalid borrowed host length reached prefix lookup");
}

void cache_concurrency_test(Model& model, int64_t length) {
  auto options = at::TensorOptions().dtype(at::kLong);
  std::vector<at::Tensor> references;
  auto tokens_for = [&](int topic) {
    std::vector<int64_t> tokens{3, 4, 5, 6, 7, 8, 9, 10, 3 + topic % 24, 3 + topic / 24, 13, 14};
    tokens.resize(length, 14);
    return tokens;
  };
  for (int64_t topic = 0; topic < 320; ++topic)
    references.push_back(run(model, tokens_for(topic), {0, int32_t(length)}));
  const auto budget = uint64_t(length <= 64 ? 2 : 32) << 20;
  PrefixCache cache(budget, std::max<int64_t>(64, length), 512, 2);
  std::vector<int32_t> ends = {0, int32_t(length)};
  auto input_for = [&](int topic) {
    return PackedInput{at::tensor(tokens_for(topic), options),
                       at::zeros({length}, options),
                       at::arange(length, options),
                       at::tensor(ends, options.dtype(at::kInt)),
                       ends.data(),
                       1,
                       length};
  };
  // Populate enough related questions to activate the resident index before
  // racing lookup, caller mutation and clear. Admission itself is also guarded.
  for (int topic = 0; topic < 320; ++topic)
    for (int repeat = 0; repeat < 3; ++repeat)
      check_close(cache.run(model, input_for(topic), 0), references[topic]);
  TORCH_CHECK(cache.stats().admissions >= 320,
              "Concurrency fixture never populated the resident position index");
  std::barrier start(3);
  std::exception_ptr failure;
  std::mutex errors;
  auto save_failure = [&] {
    std::lock_guard lock(errors);
    if (!failure)
      failure = std::current_exception();
  };
  auto worker = [&](int shift) {
    c10::InferenceMode inference;
    start.arrive_and_wait();
    try {
      for (int i = 0; i < 200; ++i) {
        const auto topic = (i / 3 + shift) % 320;
        auto value = cache.run(model, input_for(topic), 0);
        check_close(value, references[topic]);
        value.fill_(123); // Mutating caller storage must never change a cache reader.
      }
    } catch (...) {
      save_failure();
    }
  };
  std::jthread first(worker, 0), second(worker, 4);
  std::jthread maintenance([&] {
    c10::InferenceMode inference;
    start.arrive_and_wait();
    try {
      for (int i = 0; i < 600; ++i) {
        if (i % 7 == 0)
          cache.clear();
        TORCH_CHECK(cache.bytes() <= budget, "Concurrent clear/stats exceeded the cache budget");
        static_cast<void>(cache.stats());
        std::this_thread::yield();
      }
    } catch (...) {
      save_failure();
    }
  });
  first.join();
  second.join();
  maintenance.join();
  if (failure)
    std::rethrow_exception(failure);
  cache.clear();
  TORCH_CHECK(cache.bytes() == 0, "Concurrent readers retained cleared cache storage");
}
#ifdef TEI_TORCH_CUDA_KERNELS
void indexed_stream_test(const c10::Device& device) {
  struct Rows final : Model {
    void ready() override {}

    bool supports_prefix_cache(int operation) const override {
      return operation == 0;
    }

    int64_t prefix_layers() const override {
      return 1;
    }

    uint64_t prefix_payload_bytes(int64_t n) const override {
      return n * 3 * sizeof(float);
    }

    at::Tensor forward(const PackedInput& input) const override {
      auto rows = input.ids.to(at::kFloat).unsqueeze(1);
      if (input.prefix)
        input.prefix->attention(0, rows.unsqueeze(1), rows.unsqueeze(1));
      return prefix_restore(input, rows);
    }
  } model;

  const auto publisher = c10::cuda::getStreamFromPool(false, device.index());
  c10::cuda::CUDAStreamGuard publication(publisher);
  PrefixCache cache(4 << 20, 12, 512);
  std::vector<int32_t> ends = {0, 32};
  auto host = at::TensorOptions().dtype(at::kLong);
  auto input_for = [&](int topic, int document) {
    auto ids = at::arange(32, host);
    ids.index_put_({8}, 10000 + topic);
    ids.narrow(0, 12, 20).add_(document * 100);
    auto positions = at::arange(32, host), types = at::zeros({32}, host);
    PackedInput input{ids.to(device),
                      types.to(device),
                      positions.to(device),
                      at::tensor(ends, host.dtype(at::kInt)).to(device),
                      ends.data(),
                      1,
                      32};
    input.host_ids = ids;
    input.host_positions = positions;
    input.host_types = types;
    return input;
  };
  for (int topic = 0; topic < 320; ++topic)
    cache.run(model, input_for(topic, 0), 0);
  TORCH_CHECK(cache.stats().admissions == 320,
              "CUDA fixture never activated the short-question position index");
  at::cuda::getCurrentCUDAStream(device.index()).synchronize();
  {
    std::vector<at::Tensor> scratch;
    for (int i = 0; i < 1400; ++i)
      scratch.push_back(at::empty({12, 1, 1}, host.device(device).dtype(at::kFloat)).fill_(777));
    publisher.synchronize();
  }
  auto first = c10::cuda::getStreamFromPool(false, device.index()),
       second = c10::cuda::getStreamFromPool(false, device.index());
  std::vector<PackedInput> requests;
  for (int request = 0; request < 48; ++request) {
    c10::cuda::CUDAStreamGuard stream(request % 2 ? first : second);
    requests.push_back(input_for((request * 37) % 320, request + 1));
  }
  // Warm the consuming-stream allocator and boundary shape before deliberately
  // blocking one stream, so a fresh synchronizing allocation cannot open it.
  for (auto stream : {first, second}) {
    c10::cuda::CUDAStreamGuard guard(stream);
    std::vector<at::Tensor> warm_outputs;
    for (const auto& request : requests)
      warm_outputs.push_back(cache.run(model, request, 0).fill_(123));
    std::vector<at::Tensor> large;
    for (int i = 0; i < 32; ++i)
      large.push_back(at::empty({320, 12, 3}, host.device(device).dtype(at::kFloat)).fill_(777));
    std::vector<at::Tensor> scratch;
    for (int i = 0; i < 1000; ++i)
      scratch.push_back(at::empty({512}, host.device(device).dtype(at::kByte)));
    stream.synchronize();
  }
  const auto before = cache.stats().reused_tokens;

  struct Gate {
    std::atomic<bool> released = false;
    c10::cuda::CUDAStream stream;

    explicit Gate(c10::cuda::CUDAStream stream) : stream(stream) {}

    void open() {
      released.store(true);
      released.notify_all();
    }

    ~Gate() {
      open();
      stream.synchronize();
    }
  } gate(first);

  std::jthread watchdog([&](std::stop_token stop) {
    // Instrumented allocator churn can exceed the ordinary three-second gate.
    for (int i = 0; i < 3000 && !stop.stop_requested() && !gate.released.load(); ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    if (!stop.stop_requested() && !gate.released.load()) {
      std::cerr << "Indexed stream watchdog released pending work\n";
      gate.open();
    }
  });
  at::cuda::CUDAEvent proof;
  {
    c10::cuda::CUDAStreamGuard guard(first);
    C10_CUDA_CHECK(cudaLaunchHostFunc(
        first.stream(), [](void* p) { static_cast<Gate*>(p)->released.wait(false); }, &gate));
    proof.record(first);
  }
  std::vector<at::Tensor> outputs, expected;
  for (int request = 0; request < 48; ++request) {
    c10::cuda::CUDAStreamGuard stream(request % 2 ? first : second);
    const auto& input = requests[request];
    auto result = cache.run(model, input, 0);
    TORCH_CHECK(!proof.query(), "Indexed request ", request, " drained pending work");
    outputs.push_back(result);
    expected.push_back(input.host_ids.to(at::kFloat).unsqueeze(1));
    if (request == 0) {
      result.fill_(123);
      expected.back().fill_(123);
    }
  }
  TORCH_CHECK(cache.stats().reused_tokens - before == 48 * 12,
              "CUDA short questions fell back to their common role header");
  TORCH_CHECK(!proof.query(), "Indexed fixture lost its genuinely pending reader");
  // Pending readers must survive index destruction and allocator churn on both
  // consuming streams. The index itself must not retain the retired snapshots.
  cache.clear();
  TORCH_CHECK(cache.bytes() == 0, "Clearing the index retained charged snapshots");
  TORCH_CHECK(!proof.query(), "Clearing the index drained pending work");
  std::vector<at::Tensor> retired;
  for (int i = 0; i < 1400; ++i)
    retired.push_back(at::empty({12, 1, 1}, host.device(device).dtype(at::kFloat)).fill_(999));
  TORCH_CHECK(!proof.query(), "Publisher allocation drained pending work");
  at::cuda::CUDAEvent poisoned;
  poisoned.record(publisher);
  // Event polling verifies the poison writes without the sanitizer's
  // stream-synchronize flush waiting for unrelated deliberately blocked work.
  while (!poisoned.query() && !gate.released.load())
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  TORCH_CHECK(poisoned.query() && !proof.query(),
              "Publisher poison did not finish with the reader still pending");
  for (auto stream : {first, second}) {
    c10::cuda::CUDAStreamGuard guard(stream);
    for (int i = 0; i < 32; ++i)
      at::empty({320, 12, 3}, host.device(device).dtype(at::kFloat)).fill_(999);
  }
  TORCH_CHECK(!proof.query(), "Index clear or allocator churn drained the pending reader");
  gate.open();
  first.synchronize();
  second.synchronize();
  watchdog.request_stop();
  watchdog.join();
  for (size_t i = 0; i < outputs.size(); ++i)
    TORCH_CHECK(at::equal(outputs[i].cpu(), expected[i]),
                "Indexed reader changed after clear, mutation or stream retirement");
}

void cache_cold_maximum_test(const c10::Device& device) {
  const auto major = at::cuda::getDeviceProperties(device.index())->major;
  if ((major != 9 && major != 10) || at::detail::getCUDAHooks().versionRuntimeCuDNN() < 91800)
    return;
  auto cfg = config("qwen2");
  cfg.values["hidden_size"] = "128";
  cfg.values["num_attention_heads"] = "2";
  cfg.values["num_key_value_heads"] = "1";
  cfg.values["head_dim"] = "64";
  cfg.values["intermediate_size"] = "192";
  cfg.values["max_position_embeddings"] = "512";
  cfg.values["_cudnn_varlen"] = "true";
  auto generator = at::detail::createCPUGenerator(713);
  Weights tensors;
  auto add = [&](const std::string& name, std::initializer_list<int64_t> shape, bool norm = false) {
    auto value =
        norm ? at::ones(at::IntArrayRef(shape), at::TensorOptions().dtype(at::kFloat))
             : at::randn(at::IntArrayRef(shape), generator, at::TensorOptions().dtype(at::kFloat)) *
                   .03;
    tensors[name] = value.to(device, at::kBFloat16);
  };
  add("model.embed_tokens.weight", {32, 128});
  add("model.norm.weight", {128}, true);
  for (int64_t layer = 0; layer < 2; ++layer) {
    auto p = "model.layers." + std::to_string(layer) + ".";
    add(p + "input_layernorm.weight", {128}, true);
    add(p + "post_attention_layernorm.weight", {128}, true);
    for (auto projection : {"q_proj", "k_proj", "v_proj"}) {
      auto width = std::string(projection) == "q_proj" ? 128 : 64;
      add(p + "self_attn." + projection + ".weight", {width, 128});
      add(p + "self_attn." + projection + ".bias", {width});
    }
    add(p + "self_attn.o_proj.weight", {128, 128});
    add(p + "mlp.gate_proj.weight", {192, 128});
    add(p + "mlp.up_proj.weight", {192, 128});
    add(p + "mlp.down_proj.weight", {128, 192});
  }
  auto model = create_decoder(cfg, tensors, device, at::kBFloat16);
  model->ready();
  auto options = at::TensorOptions().device(device).dtype(at::kLong);
  std::vector<int32_t> offsets{0, 256, 512};
  PackedInput input{at::arange(512, options).remainder(31) + 1,
                    at::zeros({512}, options),
                    at::arange(256, options).repeat({2}),
                    at::tensor(offsets, options.dtype(at::kInt)),
                    offsets.data(),
                    2,
                    512};
  auto reference = model->forward(input);
  PrefixCache cache(2 << 20, 512, 4);
  auto captured = cache.run(*model, input, 0);
  TORCH_CHECK(at::equal(captured, reference),
              "Cold capture changed caller's attention bound/provider, maxabs=",
              (captured - reference).abs().max().item<float>());
  // A cached suffix must preserve packed cuDNN's geometry too. Both identical
  // rows hit; then one row changes its last 13 tokens, leaving a mixed suffix.
  auto warm = cache.run(*model, input, 0);
  TORCH_CHECK(at::equal(warm, reference),
              "Warm cuDNN changed outputs, maxabs=",
              (warm - reference).abs().max().item<float>());
  input.ids = input.ids.clone();
  input.ids.narrow(0, 499, 13).fill_(17);
  reference = model->forward(input);
  auto partial = cache.run(*model, input, 0);
  TORCH_CHECK(at::equal(partial, reference),
              "Mixed cuDNN suffix changed outputs, maxabs=",
              (partial - reference).abs().max().item<float>());
  TORCH_CHECK(cache.stats().hits >= 4, "cuDNN controls did not exercise prefix reuse");
}

void linear_workspace_test(const c10::Device& device) {
  // This SGEMM shape's valid suffix recipe needs 1 MiB plus alignment padding.
  // The original provider permits it, so the Lt default must not force GEMV.
  if (at::cuda::getDeviceProperties(device.index())->major != 9)
    return;
  c10::cuda::CUDAGuard guard(device);
  auto& global = at::globalContext();
  const auto old_precision = global.float32MatmulPrecision();
  const auto old_backend = global.blasPreferredBackend();
  const auto old_blas = at::cuda::getChosenWorkspaceSize(),
             old_lt = at::cuda::getCUDABlasLtWorkspaceSize();

  struct Restore {
    at::Context& context;
    at::Float32MatmulPrecision precision;
    at::BlasBackend backend;
    size_t blas, lt;

    ~Restore() {
      context.setFloat32MatmulPrecision(precision == at::Float32MatmulPrecision::HIGHEST ? "highest"
                                        : precision == at::Float32MatmulPrecision::HIGH  ? "high"
                                                                                        : "medium");
      context.setBlasPreferredBackend(backend);
      at::cuda::setChosenWorkspaceSize(blas);
      at::cuda::setCUDABlasLtWorkspaceSize(lt);
    }
  } restore{global, old_precision, old_backend, old_blas, old_lt};

  global.setFloat32MatmulPrecision("highest");
  global.setBlasPreferredBackend(at::BlasBackend::Cublas);
  at::cuda::setChosenWorkspaceSize(32 << 20);
  at::cuda::setCUDABlasLtWorkspaceSize(1 << 20);
  auto options = at::TensorOptions().device(device).dtype(at::kFloat);
  auto input = at::randn({2048, 2816}, options), weight = at::randn({128, 2816}, options);
  auto reference = at::linear(input, weight).narrow(0, 2047, 1);
  at::cuda::getCurrentCUDAStream(device.index()).synchronize();
  auto producer = c10::cuda::getStreamFromPool(false, device.index());
  at::Tensor actual;
  {
    c10::cuda::CUDAStreamGuard stream(producer);
    PrefixLinearPlans plans;
    actual = plans.forward(input.narrow(0, 2047, 1), weight, {}, 2048, true);
    plans.clear();
    // Scratch can be recycled after submission; all GPU uses remain ordered.
    for (int i = 0; i < 8; ++i)
      at::empty({(1 << 20) + 512}, options.dtype(at::kByte)).fill_(123);
  }
  producer.synchronize();
  TORCH_CHECK(at::equal(actual, reference),
              "Original SGEMM recipe fell back under the Lt workspace cutoff: ",
              (actual - reference).abs().max().item<float>());
  at::cuda::setCUDABlasLtWorkspaceSize(32 << 20);
  CachedNoBiasLt cached;
  auto bf16 = options.dtype(at::kBFloat16);
  auto mlp_weight = at::randn({12288, 2048}, bf16) * .02;
  for (int64_t rows : {2053, 8192}) {
    auto x = at::randn({rows, 2048}, bf16) * .1;
    auto expected = at::linear(x, mlp_weight);
    for (int repeat = 0; repeat < 2; ++repeat)
      TORCH_CHECK(at::equal(cached.forward(x, mlp_weight, true), expected),
                  "BF16 MLP prefill plan changed full or partial row tiles");
  }
}

void cumulative_stream_test(const c10::Device& device) {
  struct BoundaryReader final : Model {
    void ready() override {}

    bool supports_prefix_cache(int op) const override {
      return op == 0;
    }

    int64_t prefix_layers() const override {
      return 1;
    }

    uint64_t prefix_payload_bytes(int64_t n) const override {
      return n * 3 * sizeof(float);
    }

    at::Tensor forward(const PackedInput& input) const override {
      // A stale boundary becomes an observable value error rather than an
      // out-of-bounds access, allowing the pending-upload control to fail cleanly.
      auto delta = input.cumulative.select(0, input.batch).to(at::kFloat) - input.ids.numel();
      auto rows = input.ids.to(at::kFloat).unsqueeze(1) + delta;
      if (input.prefix)
        input.prefix->attention(0, rows.unsqueeze(1), rows.unsqueeze(1));
      return prefix_restore(input, rows);
    }
  } model;

  PrefixCache cache(1 << 20, 64, 32);
  std::vector<int32_t> ends{0, 32};
  auto options = at::TensorOptions().device(device).dtype(at::kLong);
  auto producer = c10::cuda::getStreamFromPool(false, device.index());
  auto consumer = c10::cuda::getStreamFromPool(false, device.index());
  auto request = [&](int cut, int tail) {
    std::vector<int64_t> ids(32);
    for (int i = 0; i < 32; ++i)
      ids[i] = i < cut ? i : tail;
    PackedInput input{at::Tensor(),
                      at::Tensor(),
                      at::Tensor(),
                      at::tensor(ends, options.dtype(at::kInt)),
                      ends.data(),
                      1,
                      32};
    input.host_ids = at::tensor(ids, at::TensorOptions().dtype(at::kLong));
    input.host_types = at::zeros({32}, at::TensorOptions().dtype(at::kLong));
    input.host_positions = at::arange(32, at::TensorOptions().dtype(at::kLong));
    input.ids = input.host_ids.to(device);
    input.types = input.host_types.to(device);
    input.positions = input.host_positions.to(device);
    return input;
  };
  // Warm both streams and fill five shapes, evicting the 24-row boundary from
  // the four-slot cache. Every source snapshot is complete before the gate.
  for (auto stream : {consumer, producer}) {
    c10::cuda::CUDAStreamGuard guard(stream);
    cache.run(model, request(32, 0), 0);
    int tail = stream == consumer ? 100 : 300;
    for (int cut : {8, 16, 24, 25, 26})
      cache.run(model, request(cut, ++tail), 0);
    stream.synchronize();
  }
  PackedInput first, second;
  at::Tensor expected, actual, queued;
  {
    c10::cuda::CUDAStreamGuard guard(producer);
    first = request(8, 201);
    producer.synchronize();
    // Seed a free allocation on the upload stream with invalid boundaries.
    auto poison = at::full({2}, -777, options.dtype(at::kInt));
    producer.synchronize();
  }
  {
    c10::cuda::CUDAStreamGuard guard(consumer);
    second = request(8, 202);
    expected = model.forward(second);
    consumer.synchronize();
  }
  // Preallocate and poison small blocks on both streams so the pending control
  // neither relies on uninitialized contents nor triggers synchronizing malloc.
  for (auto stream : {producer, consumer}) {
    c10::cuda::CUDAStreamGuard guard(stream);
    std::vector<at::Tensor> scratch;
    for (int i = 0; i < 1000; ++i)
      scratch.push_back(at::full({512}, 255, options.dtype(at::kByte)));
    stream.synchronize();
  }

  struct Gate {
    std::atomic<bool> released = false;
    c10::cuda::CUDAStream stream;

    explicit Gate(c10::cuda::CUDAStream s) : stream(s) {}

    void open() {
      released.store(true);
      released.notify_all();
    }

    ~Gate() {
      open();
      stream.synchronize();
    }
  } gate(producer);

  std::jthread watchdog([&](std::stop_token stop) {
    for (int i = 0; i < 300 && !stop.stop_requested() && !gate.released.load(); ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    if (!stop.stop_requested())
      gate.open();
  });
  at::cuda::CUDAEvent proof;
  {
    c10::cuda::CUDAStreamGuard guard(producer);
    C10_CUDA_CHECK(cudaLaunchHostFunc(
        producer.stream(), [](void* p) { static_cast<Gate*>(p)->released.wait(false); }, &gate));
    proof.record(producer);
    queued = cache.run(model, first, 0);
    TORCH_CHECK(!proof.query(), "Boundary fixture did not leave its upload pending");
  }
  {
    c10::cuda::CUDAStreamGuard guard(consumer);
    actual = cache.run(model, second, 0);
  }
  TORCH_CHECK(!proof.query(), "Boundary fixture lost its pending stream switch");
  cache.clear();
  TORCH_CHECK(cache.bytes() == 0, "Boundary clear retained charged snapshots");
  gate.open();
  consumer.synchronize();
  producer.synchronize();
  TORCH_CHECK(at::equal(actual, expected),
              "Pending boundary upload or clear changed cross-stream values");
  std::cout
      << "Immutable boundary reuse, four-shape eviction, pending upload, stream switch and clear passed\n";
}

void cache_stream_test(Model& model, PackedInput input) {
  input.host_ids = input.ids.cpu();
  input.host_types = input.types.cpu();
  input.host_positions = input.positions.cpu();
  auto expected = model.forward(input);
  at::cuda::getCurrentCUDAStream(input.ids.get_device()).synchronize();
  PrefixCache cache(1 << 20, 64, 8);
  auto producer = c10::cuda::getStreamFromPool(false, input.ids.get_device());
  auto consumer = c10::cuda::getStreamFromPool(false, input.ids.get_device());
  {
    c10::cuda::CUDAStreamGuard guard(producer);
    cache.run(model, input, 0);
    producer.synchronize();
  }
  at::Tensor actual;
  {
    c10::cuda::CUDAStreamGuard guard(consumer);
    actual = cache.run(model, input, 0);
    TORCH_CHECK(cache.stats().hits == input.batch,
                "Ready producer entries were not consumed across streams");
    cache.clear();
    TORCH_CHECK(cache.bytes() == 0, "Clear retained a cache entry after cross-stream submission");
  }
  {
    c10::cuda::CUDAStreamGuard guard(producer);
    // Recycle similarly sized allocations on the original allocation stream
    // while the consumer owns reads of the evicted immutable snapshots.
    for (int i = 0; i < 16; ++i)
      at::empty_like(expected).fill_(123);
  }
  consumer.synchronize();
  producer.synchronize();
  TORCH_CHECK(at::allclose(actual, expected, .01, .02),
              "Eviction corrupted cross-stream cached output");

  struct Gate {
    std::atomic<bool> released = false;
    c10::cuda::CUDAStream stream;

    explicit Gate(c10::cuda::CUDAStream value) : stream(value) {}

    void open() {
      released.store(true);
      released.notify_all();
    }

    ~Gate() {
      open();
      stream.synchronize();
    }
  } gate(producer);

  struct Delayed final : Model {
    Model& inner;
    Gate& gate;
    at::cuda::CUDAEvent& proof;
    bool delay = true;

    Delayed(Model& model, Gate& value, at::cuda::CUDAEvent& event)
        : inner(model), gate(value), proof(event) {}

    void ready() override {}

    bool supports_prefix_cache(int operation) const override {
      return inner.supports_prefix_cache(operation);
    }

    int64_t prefix_layers() const override {
      return inner.prefix_layers();
    }

    uint64_t prefix_payload_bytes(int64_t tokens) const override {
      return inner.prefix_payload_bytes(tokens);
    }

    at::Tensor forward(const PackedInput& input) const override {
      auto result = inner.forward(input);
      if (delay) {
        auto wait = [](void* data) { static_cast<Gate*>(data)->released.wait(false); };
        C10_CUDA_CHECK(cudaLaunchHostFunc(gate.stream.stream(), wait, &gate));
        proof.record(gate.stream);
      }
      return result;
    }
  };

  at::cuda::CUDAEvent proof;
  Delayed delayed(model, gate, proof);
  PrefixCache pending(1 << 20, 64, 8);
  {
    c10::cuda::CUDAStreamGuard guard(producer);
    pending.run(delayed, input, 0);
    // The proof event precedes cache publication's event. A closed gate keeps
    // both pending, so this tests actual queued state rather than a ready hit.
    const bool incomplete = !proof.query();
    std::jthread release([&] {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      gate.open();
    });
    TORCH_CHECK(incomplete, "Fixture did not retain pending producer work");
    delayed.delay = false;
    {
      c10::cuda::CUDAStreamGuard guard(consumer);
      auto unrelated = pending.run(delayed, input, 0);
      TORCH_CHECK(pending.stats().hits == 0,
                  "Independent consumer reused an unordered pending snapshot");
      consumer.synchronize();
      TORCH_CHECK(at::allclose(unrelated, expected, .01, .02),
                  "Independent pending fallback changed outputs");
    }
    auto resumed = pending.run(delayed, input, 0);
    TORCH_CHECK(pending.stats().hits == input.batch,
                "Pending producer-stream state was recomputed");
    producer.synchronize();
    TORCH_CHECK(at::allclose(resumed, expected, .01, .02),
                "Producer-stream pending reuse changed outputs");
  }
}

void native_engine_stream_test(const c10::Device& device) {
  auto producer = c10::cuda::getStreamFromPool(false, device.index());
  auto consumer = c10::cuda::getStreamFromPool(false, device.index());
  for (int pool : {2, 4})
    for (bool caching : {false, true}) {
      auto cfg = config("qwen3");
      cfg.values["num_attention_heads"] = "2";
      cfg.values["num_key_value_heads"] = "1";
      cfg.values["head_dim"] = "8";
      auto tensors = weights(cfg);
      const int width = pool == 4 ? 3 : 16;
      if (pool == 4) {
        cfg.values["architectures.0"] = "Qwen3ForSequenceClassification";
        cfg.values["num_labels"] = "3";
        cfg.values["pad_token_id"] = "0";
        tensors["score.weight"] = at::arange(48, at::kFloat).view({3, 16}) * .013;
        tensors["score.bias"] = at::tensor({.1f, -.2f, .3f});
      }
      for (auto& [name, value] : tensors)
        if (name.ends_with("q_norm.weight") || name.ends_with("k_norm.weight"))
          value = value.repeat({2});
      TeiConfig shape{16, 2, 2, 24, 32, 32, 1, 1e-6, 3};
      auto handle = tei_create(&shape, device.str().c_str(), 2);
      TORCH_CHECK(handle, tei_error());

      struct EngineOwner {
        void* value;

        ~EngineOwner() {
          if (value)
            tei_destroy(value);
        }
      } owner{handle};

      for (const auto& [key, value] : cfg.values)
        TORCH_CHECK(tei_option(handle, key.c_str(), value.c_str()) == 0, tei_error());
      TORCH_CHECK(tei_option(handle, "_cuda_graphs", "true") == 0, tei_error());
      TORCH_CHECK(tei_option(handle, "_cuda_graph_reuse", "1") == 0, tei_error());
      if (caching)
        TORCH_CHECK(tei_option(handle, "_prefix_cache_max_bytes", "1048576") == 0 &&
                        tei_option(handle, "_prefix_cache_entries", "1") == 0,
                    tei_error());
      for (const auto& [key, value] : tensors)
        TORCH_CHECK(tei_weight(handle,
                               key.c_str(),
                               reinterpret_cast<const uint8_t*>(value.const_data_ptr()),
                               value.sizes().data(),
                               value.dim(),
                               0) == 0,
                    tei_error());
      TORCH_CHECK(tei_ready(handle) == 0, tei_error());
      std::array<int64_t, 4> a = {3, 4, 5, 6}, b = {7, 4, 5, 8}, types{}, positions = {0, 1, 2, 3};
      int32_t ends[] = {0, 4};
      int64_t selected[] = {0};
      auto submit = [&](const auto& ids) {
        auto output = tei_forward_submit(handle,
                                         ids.data(),
                                         types.data(),
                                         positions.data(),
                                         ends,
                                         1,
                                         4,
                                         pool,
                                         selected,
                                         1,
                                         nullptr,
                                         0,
                                         width,
                                         nullptr,
                                         nullptr,
                                         0,
                                         nullptr,
                                         0,
                                         nullptr);
        TORCH_CHECK(output, tei_error());
        return output;
      };

      struct OutputOwner {
        void* value = nullptr;

        ~OutputOwner() {
          if (value)
            tei_output_destroy(value);
        }
      };

      std::vector<float> expected_a(width), expected_b(width);
      auto warm = [&](const auto& ids, auto& expected, c10::cuda::CUDAStream stream) {
        c10::cuda::CUDAStreamGuard guard(stream);
        for (int i = 0; i < 4; ++i) {
          OutputOwner output{submit(ids)};
          auto data = tei_output_wait(output.value);
          TORCH_CHECK(data, tei_error());
          std::copy(data, data + width, expected.begin());
        }
      };
      warm(a, expected_a, producer);
      warm(b, expected_b, consumer);
      TORCH_CHECK(expected_a != expected_b, "Stream fixture cannot distinguish its two requests");
      TORCH_CHECK(tei_graph_count(handle) == 1,
                  "Native stream fixture did not retain a shared graph");

      struct Gate {
        std::atomic<bool> open = false;
        std::mutex mutex;
        std::condition_variable changed;
        c10::cuda::CUDAStream stream;

        explicit Gate(c10::cuda::CUDAStream value) : stream(value) {}

        void release() {
          open.store(true);
          open.notify_all();
          changed.notify_all();
        }

        ~Gate() {
          release();
          stream.synchronize();
        }
      } gate(producer);

      C10_CUDA_CHECK(cudaLaunchHostFunc(
          producer.stream(),
          [](void* pointer) { static_cast<Gate*>(pointer)->open.wait(false); },
          &gate));
      at::cuda::CUDAEvent blocked;
      blocked.record(producer);
      std::jthread watchdog([&] {
        std::unique_lock lock(gate.mutex);
        gate.changed.wait_for(lock, std::chrono::seconds(2), [&] { return gate.open.load(); });
        gate.release();
      });
      OutputOwner first, second;
      {
        c10::cuda::CUDAStreamGuard guard(producer);
        first.value = submit(a);
      }
      at::cuda::CUDAEvent consumed;
      {
        c10::cuda::CUDAStreamGuard guard(consumer);
        second.value = submit(b);
        consumed.record(consumer);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      const bool pending = !blocked.query(), ordered = !consumed.query();
      gate.release();
      watchdog.join();
      std::cout << "submission proof: pool=" << pool << " cache=" << caching
                << " pending=" << pending << " ordered=" << ordered << std::endl;
      TORCH_CHECK(pending, "Native stream fixture did not retain queued producer work");
      TORCH_CHECK(ordered, "A different stream bypassed native submission ordering");
      for (auto pair :
           {std::pair{first.value, &expected_a}, std::pair{second.value, &expected_b}}) {
        auto data = tei_output_wait(pair.first);
        TORCH_CHECK(data, tei_error());
        TORCH_CHECK(std::equal(data, data + width, pair.second->begin()),
                    "Cross-stream replay overwrote a deferred native result");
      }
      tei_output_destroy(first.value);
      first.value = nullptr;
      tei_output_destroy(second.value);
      second.value = nullptr;
      {
        // Move completion to another CPU thread, alternate GPU streams and retire
        // readers only before reusing their A/B slot. Clear/one-entry eviction and
        // cancellation run while the other lease can still own queued output.
        std::array<OutputOwner, 2> pending;
        std::array<const std::vector<float>*, 2> expected{};
        auto collect = [&](int slot, bool cancel = false) {
          if (!pending[slot].value)
            return;
          if (!cancel) {
            auto value = pending[slot].value;
            auto data = std::async(std::launch::async, [value] {
                          auto result = tei_output_wait(value);
                          TORCH_CHECK(result, tei_error());
                          return result;
                        }).get();
            TORCH_CHECK(std::equal(data, data + width, expected[slot]->begin()),
                        "Native classifier/embedding stress changed a deferred result");
          }
          tei_output_destroy(pending[slot].value);
          pending[slot].value = nullptr;
        };
        for (int step = 0; step < 96; ++step) {
          const auto slot = step % 2;
          collect(slot, step % 13 == 0);
          if (step % 17 == 0)
            TORCH_CHECK(tei_prefix_cache_clear(handle) == 0, tei_error());
          const bool use_b = (step / 3) % 2;
          c10::cuda::CUDAStreamGuard guard(step % 2 ? consumer : producer);
          pending[slot].value = submit(use_b ? b : a);
          expected[slot] = use_b ? &expected_b : &expected_a;
        }
        collect(0);
        collect(1);
        if (caching) {
          uint64_t stats[7];
          TORCH_CHECK(tei_prefix_cache_stats(handle, stats, 7) == 0 && stats[4] > 0,
                      "One-entry native stress never evicted a snapshot");
        }
        {
          c10::cuda::CUDAStreamGuard guard(consumer);
          OutputOwner invalid{tei_forward_submit(handle,
                                                 a.data(),
                                                 types.data(),
                                                 positions.data(),
                                                 ends,
                                                 1,
                                                 4,
                                                 pool,
                                                 selected,
                                                 1,
                                                 nullptr,
                                                 0,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 0,
                                                 nullptr)};
          TORCH_CHECK(!invalid.value, "Undersized deferred output unexpectedly succeeded");
        }
        {
          c10::cuda::CUDAStreamGuard guard(producer);
          OutputOwner recovered{submit(a)};
          auto data = tei_output_wait(recovered.value);
          TORCH_CHECK(data && std::equal(data, data + width, expected_a.begin()),
                      "Recoverable submission error contaminated the next stream's result");
        }
      }
      if (caching) {
        TORCH_CHECK(tei_prefix_cache_clear(handle) == 0, tei_error());
        {
          c10::cuda::CUDAStreamGuard guard(producer);
          OutputOwner observe{submit(a)};
          TORCH_CHECK(tei_output_wait(observe.value), tei_error());
        }
        gate.open.store(false);
        C10_CUDA_CHECK(cudaLaunchHostFunc(
            producer.stream(),
            [](void* pointer) { static_cast<Gate*>(pointer)->open.wait(false); },
            &gate));
        at::cuda::CUDAEvent unpublished;
        unpublished.record(producer);
        std::jthread release_pending([&] {
          std::unique_lock lock(gate.mutex);
          gate.changed.wait_for(lock, std::chrono::seconds(2), [&] { return gate.open.load(); });
          gate.release();
        });
        OutputOwner produced, resumed;
        uint64_t before[7], after[7];
        {
          c10::cuda::CUDAStreamGuard guard(producer);
          produced.value = submit(a);
        }
        TORCH_CHECK(tei_prefix_cache_stats(handle, before, 7) == 0 && before[6] > 0,
                    "Pending native fixture did not admit a snapshot");
        {
          c10::cuda::CUDAStreamGuard guard(consumer);
          resumed.value = submit(a);
        }
        TORCH_CHECK(tei_prefix_cache_stats(handle, after, 7) == 0, tei_error());
        const bool was_pending = !unpublished.query();
        gate.release();
        release_pending.join();
        for (auto output : {produced.value, resumed.value}) {
          auto data = tei_output_wait(output);
          TORCH_CHECK(data && std::equal(data, data + width, expected_a.begin()),
                      "Ordered pending native reuse changed result bits");
        }
        TORCH_CHECK(was_pending, "Native snapshot fixture did not retain pending producer work");
        TORCH_CHECK(after[0] == before[0] + 1 && after[2] == before[2] + 4,
                    "Ordered native stream recomputed a pending snapshot");
      }
      // Stage replacement weights before blocking the producer. Ready must wait
      // for an old-model submission, then publish initialization on its new stream.
      for (const auto& [key, value] : tensors)
        TORCH_CHECK(tei_weight(handle,
                               key.c_str(),
                               reinterpret_cast<const uint8_t*>(value.const_data_ptr()),
                               value.sizes().data(),
                               value.dim(),
                               0) == 0,
                    tei_error());
      TORCH_CHECK(tei_option(handle, "_cuda_graphs", "false") == 0 &&
                      tei_option(handle, "_prefix_cache_max_bytes", "0") == 0,
                  tei_error());
      gate.open.store(false);
      C10_CUDA_CHECK(cudaLaunchHostFunc(
          producer.stream(),
          [](void* pointer) { static_cast<Gate*>(pointer)->open.wait(false); },
          &gate));
      at::cuda::CUDAEvent reloading;
      reloading.record(producer);
      std::jthread reload_watchdog([&] {
        std::unique_lock lock(gate.mutex);
        gate.changed.wait_for(lock, std::chrono::seconds(2), [&] { return gate.open.load(); });
        gate.release();
      });
      OutputOwner previous;
      {
        c10::cuda::CUDAStreamGuard guard(producer);
        previous.value = submit(a);
      }
      const bool reload_pending = !reloading.query();
      auto reload = std::async(std::launch::async, [handle] { return tei_ready(handle); });
      const auto reload_status = reload.wait_for(std::chrono::milliseconds(10));
      gate.release();
      reload_watchdog.join();
      const auto reload_result = reload.get();
      std::cout << "reload proof: pending=" << reload_pending
                << " wait_status=" << int(reload_status) << " result=" << reload_result
                << std::endl;
      TORCH_CHECK(reload_pending && reload_status == std::future_status::timeout &&
                      reload_result == 0,
                  "Ready did not drain pending old-model work");
      const auto prior = tei_output_wait(previous.value);
      TORCH_CHECK(prior && std::equal(prior, prior + width, expected_a.begin()),
                  "Ready invalidated a deferred result from the old model");
      TORCH_CHECK(tei_graph_count(handle) == 0, "Ready retained disabled CUDA graphs");
      {
        c10::cuda::CUDAStreamGuard guard(consumer);
        OutputOwner fresh{submit(b)};
        auto data = tei_output_wait(fresh.value);
        TORCH_CHECK(data && std::equal(data, data + width, expected_b.begin()),
                    "Ready did not order initialization before the next stream's inference");
      }
      TORCH_CHECK(std::equal(prior, prior + width, expected_a.begin()),
                  "New-model inference overwrote a retained old-model result");
      tei_output_destroy(previous.value);
      previous.value = nullptr;
      // Destruction must drain a genuinely pending submission, and the lease still
      // owns its host result afterward. No engine calls start after destruction.
      gate.open.store(false);
      C10_CUDA_CHECK(cudaLaunchHostFunc(
          producer.stream(),
          [](void* pointer) { static_cast<Gate*>(pointer)->open.wait(false); },
          &gate));
      at::cuda::CUDAEvent retiring;
      retiring.record(producer);
      std::jthread retire_watchdog([&] {
        std::unique_lock lock(gate.mutex);
        gate.changed.wait_for(lock, std::chrono::seconds(2), [&] { return gate.open.load(); });
        gate.release();
      });
      OutputOwner survivor;
      {
        c10::cuda::CUDAStreamGuard guard(producer);
        survivor.value = submit(a);
      }
      const bool retiring_pending = !retiring.query();
      auto destroyed = std::async(std::launch::async, [handle] { tei_destroy(handle); });
      const auto retiring_status = destroyed.wait_for(std::chrono::milliseconds(10));
      gate.release();
      retire_watchdog.join();
      destroyed.get();
      owner.value = nullptr;
      TORCH_CHECK(retiring_pending, "Destruction fixture did not retain queued GPU work");
      TORCH_CHECK(retiring_status == std::future_status::timeout,
                  "Engine destruction released a pending submission");
      const auto saved = tei_output_wait(survivor.value);
      TORCH_CHECK(saved && std::equal(saved, saved + width, expected_a.begin()),
                  "Engine destruction invalidated an owned output lease");
    }
  std::cout
      << "native stream ordering, pending engine retirement and independent deferred leases passed (cache off/on; embedding/classifier)\n";
}

void suffix_graph_test(Model& model, const PackedInput& original) {
  PrefixCache cache(16 << 20, 64, 8);
  CudaGraphCache graphs(4, 64, 1, 64 << 20, 8);
  auto a = original.ids.cpu().clone(), b = a.clone();
  b[0] = 6;
  b[original.offsets[1]] = 7;
  auto input = original;
  auto set = [&](const at::Tensor& ids) {
    input.host_ids = ids;
    input.host_types = original.types.cpu();
    input.host_positions = original.positions.cpu();
    input.ids = ids.to(original.ids.device());
  };
  set(a);
  cache.run(model, input, 0, &graphs);
  set(b);
  cache.run(model, input, 0, &graphs);
  // Repeat each parent before changing it: replay must preserve already-bound
  // immutable state, rebind another parent, and retire weak owners after clear.
  for (int step = 0; step < 8; ++step) {
    if (step == 4) {
      cache.clear();
      TORCH_CHECK(cache.bytes() == 0,
                  "Graph templates retained resident prefix owners after clear");
      set(a);
      cache.run(model, input, 0, &graphs);
      set(b);
      cache.run(model, input, 0, &graphs);
    }
    auto ids = ((step / 2) % 2 ? b : a).clone();
    for (int64_t row = 0; row < input.batch; ++row)
      ids[input.offsets[row + 1] - 1] = 10 + step;
    set(ids);
    auto expected = model.forward(input), actual = cache.run(model, input, 0, &graphs);
    TORCH_CHECK(at::allclose(actual, expected, .01, .02),
                "Suffix graph retained another parent's KV or output values");
    actual.zero_();
  }
  auto stats = graphs.stats();
  TORCH_CHECK(stats.prefix_captures == 1 && stats.prefix_hits >= 7,
              "Changing parents or cache clear invalidated a reusable suffix shader");
  TORCH_CHECK(cache.bytes() <= 16 << 20,
              "Suffix graph publication exceeded the resident snapshot budget");
  PrefixCache constrained(16 << 20, 64, 8);
  CudaGraphCache undersized(4, 64, 1, 1, 8);
  set(a);
  constrained.run(model, input, 0, &undersized);
  auto tail = a.clone();
  tail[input.offsets[1] - 1] = 18;
  set(tail);
  TORCH_CHECK(
      at::allclose(constrained.run(model, input, 0, &undersized), model.forward(input), .01, .02),
      "Known oversized suffix arena changed eager fallback results");
  TORCH_CHECK(undersized.stats().attempts == 0,
              "Known oversized suffix arena attempted GPU allocation/capture");
  std::cout
      << "suffix graph replay, parent rebinding, independent outputs and cache clear passed\n";
}
#endif
void multimodal_test() {
  auto cfg = config("qwen3");
  auto w = weights(cfg);
  auto model = create_decoder(cfg, w, c10::Device("cpu"), at::kFloat);
  model->ready();
  auto opts = at::TensorOptions().dtype(at::kLong);
  std::vector<int64_t> ids = {3, 4, 5, 6, 7}, pos = {0, 1, 2, 0, 1};
  std::vector<int32_t> offsets = {0, 3, 5};
  PackedInput input{at::from_blob(ids.data(), {5}, opts),
                    at::zeros({5}, opts),
                    at::from_blob(pos.data(), {5}, opts),
                    at::from_blob(offsets.data(), {3}, opts.dtype(at::kInt)),
                    offsets.data(),
                    2,
                    3};
  auto frequencies = at::tensor({1.f, .01f});
  auto angles = input.positions.to(at::kFloat).unsqueeze(1) * frequencies;
  auto cosine = at::cos(angles), sine = at::sin(angles);
  check_close(model->forward(input),
              decoder_multimodal_forward(*model, input, {}, {}, cosine, sine));
  auto indices = at::tensor({int64_t(1)}, opts), visual = at::ones({1, 16}) * .25;
  auto injected = decoder_multimodal_forward(*model, input, indices, visual, cosine, sine);
  auto zero = at::zeros_like(visual);
  check_close(
      injected,
      decoder_multimodal_forward(*model, input, indices, visual, cosine, sine, {zero, zero}));
  auto deep = decoder_multimodal_forward(*model,
                                         input,
                                         indices,
                                         visual,
                                         cosine,
                                         sine,
                                         {at::arange(16).to(at::kFloat).view({1, 16}) * .07});
  TORCH_CHECK((deep - injected).abs().max().item<float>() > 1e-4,
              "Deepstack feature injection did not affect text states");
  check_close(deep.narrow(0, 3, 2), model->forward(input).narrow(0, 3, 2));
  check_close(deep.narrow(0, 0, 1), injected.narrow(0, 0, 1));
  std::cout << "Qwen3 external embeddings, multimodal RoPE and deepstack packed isolation passed\n";
}

void native_reconfigure_test(const c10::Device& device) {
  auto cfg = config("qwen3");
  cfg.values["num_attention_heads"] = "2";
  cfg.values["num_key_value_heads"] = "1";
  cfg.values["head_dim"] = "8";
  cfg.values["tie_word_embeddings"] = "true";
  auto tensors = weights(cfg);
  for (auto& [name, value] : tensors)
    if (name.ends_with("q_norm.weight") || name.ends_with("k_norm.weight"))
      value = value.repeat({2});
  TeiConfig shape{16, 2, 2, 24, 32, 32, 1, 1e-6, 3};
  auto handle = tei_create(&shape, device.str().c_str(), device.is_cuda() ? 2 : 0);
  TORCH_CHECK(handle, tei_error());

  struct Owner {
    void* value;

    ~Owner() {
      tei_destroy(value);
    }
  } owner{handle};

  for (const auto& [key, value] : cfg.values)
    TORCH_CHECK(tei_option(handle, key.c_str(), value.c_str()) == 0, tei_error());
  auto upload = [&] {
    for (const auto& [name, value] : tensors)
      TORCH_CHECK(tei_weight(handle,
                             name.c_str(),
                             reinterpret_cast<const uint8_t*>(value.const_data_ptr()),
                             value.sizes().data(),
                             value.dim(),
                             0) == 0,
                  tei_error());
  };
  TORCH_CHECK(tei_option(handle, "_prefix_cache_max_bytes", "1048576") == 0, tei_error());
  TORCH_CHECK(tei_option(handle, "_cuda_graphs", "true") == 0, tei_error());
  TORCH_CHECK(tei_option(handle, "_cuda_graph_reuse", "1") == 0, tei_error());
  TORCH_CHECK(tei_option(handle, "_decision.kind", "option_tokens") == 0, tei_error());
  upload();
  TORCH_CHECK(tei_ready(handle) == 0, tei_error());
  int64_t ids[] = {3, 4, 5, 6}, types[] = {0, 0, 0, 0}, positions[] = {0, 1, 2, 3},
          selected[] = {0};
  int32_t ends[] = {0, 4};
  float output[16];
  auto forward = [&] {
    TORCH_CHECK(tei_forward(handle,
                            ids,
                            types,
                            positions,
                            ends,
                            1,
                            4,
                            2,
                            selected,
                            1,
                            nullptr,
                            0,
                            output,
                            16,
                            nullptr,
                            nullptr,
                            0,
                            nullptr,
                            0) == 0,
                tei_error());
  };
  forward();
  forward();
  forward();
  std::array<float, 16> expected;
  std::copy(std::begin(output), std::end(output), expected.begin());
  uint64_t stats[7];
  TORCH_CHECK(tei_prefix_cache_stats(handle, stats, 7) == 0 && stats[3] > 0 && stats[6] > 0,
              "Reconfiguration fixture did not fill cache");
  if (device.is_cuda())
    TORCH_CHECK(tei_graph_count(handle) > 0, "Reconfiguration fixture did not capture a graph");
  TeiDecision decision{};
  decision.kind = 1;
  decision.token_ids = selected;
  decision.token_count = 1;
  int64_t count;
  TORCH_CHECK(tei_decision_counts(handle, &decision, 1, &count) == 0 && count == 1,
              "Reconfiguration fixture did not load a decision head");
  TORCH_CHECK(tei_option(handle, "_prefix_cache_max_bytes", "0") == 0, tei_error());
  TORCH_CHECK(tei_option(handle, "_cuda_graphs", "false") == 0, tei_error());
  TORCH_CHECK(tei_option(handle, "_decision.kind", "") == 0, tei_error());
  upload();
  TORCH_CHECK(tei_ready(handle) == 0, tei_error());
  TORCH_CHECK(tei_prefix_cache_stats(handle, stats, 7) == 0, tei_error());
  const auto decision_status = tei_decision_counts(handle, &decision, 1, &count);
  const auto graphs = tei_graph_count(handle);
  std::cout << "reinitialized feature state: cache_bytes=" << stats[6] << " graphs=" << graphs
            << " decision_status=" << decision_status << std::endl;
  TORCH_CHECK(
      std::all_of(std::begin(stats), std::end(stats), [](auto value) { return value == 0; }) &&
          graphs == 0 && decision_status == -1,
      "Native reinitialization retained disabled prefix cache, graphs or decision head");
  forward();
  forward();
  TORCH_CHECK(tei_prefix_cache_stats(handle, stats, 7) == 0 && stats[3] == 0 &&
                  tei_graph_count(handle) == 0,
              "Disabled features reappeared after inference");
  TORCH_CHECK(std::equal(std::begin(output), std::end(output), expected.begin()),
              "Disabling optional features changed native inference results");
}

void probe_cache_test() {
  auto cfg = config("qwen3");
  auto source = weights(cfg);
  TeiConfig native{16, 4, 2, 24, 32, 32, 1, 1e-6, 3};
  auto handle = tei_create(&native, "cpu", 0);
  TORCH_CHECK(handle, tei_error());

  struct Cleanup {
    void* handle;

    ~Cleanup() {
      tei_destroy(handle);
    }
  } cleanup{handle};

  for (const auto& [key, value] : cfg.values)
    TORCH_CHECK(tei_option(handle, key.c_str(), value.c_str()) == 0, tei_error());
  TORCH_CHECK(tei_option(handle, "_prefix_cache_max_bytes", "1048576") == 0, tei_error());
  for (const auto& [name, tensor] : source)
    TORCH_CHECK(tei_weight(handle,
                           name.c_str(),
                           reinterpret_cast<const uint8_t*>(tensor.const_data_ptr()),
                           tensor.sizes().data(),
                           tensor.dim(),
                           0) == 0,
                tei_error());
  TORCH_CHECK(tei_ready(handle) == 0, tei_error());
  int64_t ids[] = {1, 2, 3}, types[] = {0, 0, 0}, positions[] = {0, 1, 2}, selection[] = {0};
  int32_t cumulative[] = {0, 3};
  float result[16];
  auto forward = [&] {
    TORCH_CHECK(tei_forward(handle,
                            ids,
                            types,
                            positions,
                            cumulative,
                            1,
                            3,
                            2,
                            selection,
                            1,
                            nullptr,
                            0,
                            result,
                            16,
                            nullptr,
                            nullptr,
                            0,
                            nullptr,
                            0) == 0,
                tei_error());
  };
  uint64_t before[7], after[7];
  forward();
  TORCH_CHECK(tei_prefix_cache_stats(handle, before, 7) == 0 && before[3] == 0,
              "One-off user input bypassed admission policy");
  forward();
  TORCH_CHECK(tei_prefix_cache_stats(handle, before, 7) == 0 && before[3] == 1,
              "User request did not fill prefix cache");
  TORCH_CHECK(tei_prefix_cache_set_probe(handle, 1) == 0, "Probe begin did not return prior flag");
  forward();
  TORCH_CHECK(tei_prefix_cache_stats(handle, after, 7) == 0, tei_error());
  TORCH_CHECK(std::equal(std::begin(before), std::end(before), std::begin(after)),
              "Probe consumed or populated user cache state");
  TORCH_CHECK(tei_prefix_cache_set_probe(handle, 0) == 1, "Probe end did not restore prior flag");
  forward();
  TORCH_CHECK(tei_prefix_cache_stats(handle, after, 7) == 0 && after[0] == before[0] + 1,
              "Probe bypass leaked into next user request");
  std::barrier start(2);
  std::exception_ptr failure;
  std::mutex errors;
  auto save_failure = [&] {
    std::lock_guard lock(errors);
    if (!failure)
      failure = std::current_exception();
  };
  std::jthread inference([&] {
    start.arrive_and_wait();
    try {
      for (int i = 0; i < 100; ++i) {
        float owned[16];
        TORCH_CHECK(tei_forward(handle,
                                ids,
                                types,
                                positions,
                                cumulative,
                                1,
                                3,
                                2,
                                selection,
                                1,
                                nullptr,
                                0,
                                owned,
                                16,
                                nullptr,
                                nullptr,
                                0,
                                nullptr,
                                0) == 0,
                    tei_error());
        TORCH_CHECK(std::equal(std::begin(owned), std::end(owned), std::begin(result)),
                    "Concurrent native probe changed user output");
      }
    } catch (...) {
      save_failure();
    }
  });
  std::jthread probes([&] {
    start.arrive_and_wait();
    try {
      for (int i = 0; i < 600; ++i) {
        TORCH_CHECK(tei_prefix_cache_set_probe(handle, i % 2) >= 0, tei_error());
        uint64_t current[7];
        TORCH_CHECK(tei_prefix_cache_stats(handle, current, 7) == 0, tei_error());
        if (i % 7 == 0)
          TORCH_CHECK(tei_prefix_cache_clear(handle) == 0, tei_error());
      }
    } catch (...) {
      save_failure();
    }
  });
  inference.join();
  probes.join();
  if (failure)
    std::rethrow_exception(failure);
  TORCH_CHECK(tei_prefix_cache_set_probe(handle, 0) >= 0, tei_error());
}
} // namespace

// Explicit scores/masks independently check the per-layer RoPE and local endpoints.
at::Tensor
ultrabert_reference(const Options& cfg, const Weights& weights, const std::vector<int64_t>& ids) {
  auto opts = at::TensorOptions().dtype(at::kFloat);
  auto x = at::embedding(weights.at("model.embed_tokens.weight"),
                         at::tensor(ids, at::TensorOptions().dtype(at::kLong)));
  const auto n = int64_t(ids.size()), heads = cfg.integer("num_attention_heads"),
             kv = cfg.integer("num_key_value_heads"), d = cfg.integer("head_dim");
  auto norm = [&](const at::Tensor& t, const std::string& key) {
    return t * at::rsqrt(t.square().mean(-1, true) + 1e-6) * weights.at(key);
  };
  for (int64_t layer = 0; layer < 2; ++layer) {
    auto prefix = "model.layers." + std::to_string(layer) + ".";
    auto h = norm(x, prefix + "input_layernorm.weight");
    auto inv = at::pow(layer ? 2560000. : 10000., -at::arange(0, d, 2, opts) / d);
    auto angle = at::arange(n, opts).unsqueeze(1) * inv.unsqueeze(0);
    auto cosine = at::cat({angle, angle}, -1).cos().unsqueeze(1),
         sine = at::cat({angle, angle}, -1).sin().unsqueeze(1);
    auto project = [&](const char* name, int64_t count) {
      return at::linear(h, weights.at(prefix + "self_attn." + name + ".weight"))
          .view({n, count, d});
    };
    auto rotate = [&](const at::Tensor& t) {
      return t * cosine + at::cat({-t.narrow(-1, d / 2, d / 2), t.narrow(-1, 0, d / 2)}, -1) * sine;
    };
    auto q = rotate(project("q_proj", heads)).transpose(0, 1);
    auto k = at::repeat_interleave(rotate(project("k_proj", kv)), heads / kv, 1).transpose(0, 1);
    auto v = at::repeat_interleave(project("v_proj", kv), heads / kv, 1).transpose(0, 1);
    auto scores = at::matmul(q, k.transpose(-1, -2)) / std::sqrt(double(d));
    if (layer == 0) {
      auto positions = at::arange(n, at::TensorOptions().dtype(at::kLong));
      scores = scores.masked_fill((positions.unsqueeze(1) - positions.unsqueeze(0)).abs() > 2,
                                  -INFINITY);
    }
    auto attended =
        at::matmul(scores.softmax(-1), v).transpose(0, 1).contiguous().view({n, heads * d});
    x = x + at::linear(attended, weights.at(prefix + "self_attn.o_proj.weight"));
    h = norm(x, prefix + "post_attention_layernorm.weight");
    x = x + at::linear(at::silu(at::linear(h, weights.at(prefix + "mlp.gate_proj.weight"))) *
                           at::linear(h, weights.at(prefix + "mlp.up_proj.weight")),
                       weights.at(prefix + "mlp.down_proj.weight"));
  }
  return norm(x, "model.norm.weight");
}

void ultrabert_test() {
  auto cfg = config("ultrabert");
  cfg.values["rope_theta"] = "[10000,2560000]";
  cfg.values["attention_windows"] = "[2,null]";
  for (bool independent : {false, true}) {
    if (independent) {
      cfg.values["num_attention_heads"] = "3";
      cfg.values["num_key_value_heads"] = "1";
    }
    auto original = weights(cfg);
    if (independent)
      for (int64_t i = 0; i < 2; ++i) {
        auto prefix = "model.layers." + std::to_string(i) + ".self_attn.";
        original[prefix + "q_proj.weight"] =
            original[prefix + "q_proj.weight"].narrow(0, 0, 12).contiguous();
        for (auto name : {"k_proj", "v_proj"})
          original[prefix + name + ".weight"] =
              original[prefix + name + ".weight"].narrow(0, 0, 4).contiguous();
        original[prefix + "o_proj.weight"] =
            original[prefix + "o_proj.weight"].narrow(1, 0, 12).contiguous();
      }
    for (const auto* arch : {"UltrabertModel",
                             "UltrabertForSequenceClassification",
                             "UltrabertForTokenClassification"}) {
      const bool classified = std::string(arch) != "UltrabertModel";
      auto c = cfg;
      c.values["architectures.0"] = arch;
      c.values["_classifier"] = classified ? "true" : "false";
      c.values["num_labels"] = "3";
      Weights loaded;
      for (const auto& [name, value] : original)
        loaded[(classified ? "ultrabert." : "") + name.substr(6)] = value;
      auto score = at::arange(48, at::TensorOptions().dtype(at::kFloat)).view({3, 16}) / 64.,
           bias = at::tensor({.1f, -.2f, .3f});
      if (classified) {
        loaded["classifier.weight"] = score;
        loaded["classifier.bias"] = bias;
      }
      auto model = create_decoder(c, loaded, c10::Device("cpu"), at::kFloat);
      model->ready();
      TORCH_CHECK(!model->supports_radix(), "UltraBERT advertised causal prefix reuse");
      check_close(run(*model, {3, 4, 5, 6, 7, 8, 9}, {0, 7}),
                  ultrabert_reference(cfg, original, {3, 4, 5, 6, 7, 8, 9}));
      auto expected = at::cat({ultrabert_reference(cfg, original, {3, 4, 5}),
                               ultrabert_reference(cfg, original, {6, 7, 8, 9})});
      check_close(run(*model, {3, 4, 5, 6, 7, 8, 9}, {0, 3, 7}), expected);
      TORCH_CHECK((run(*model, {3, 4}, {0, 2}) - run(*model, {3, 4, 5}, {0, 3}).narrow(0, 0, 2))
                          .abs()
                          .max()
                          .item<float>() > 1e-5,
                  "UltraBERT became causal");
      if (classified) {
        auto opts = at::TensorOptions().dtype(at::kLong);
        auto ids = at::tensor({3, 4, 5, 6, 7, 8, 9}, opts),
             positions = at::tensor({0, 1, 2, 0, 1, 2, 3}, opts);
        std::vector<int32_t> offsets{0, 3, 7};
        auto cumulative = at::tensor(offsets, opts.dtype(at::kInt));
        PackedInput input{ids, at::zeros({7}, opts), positions, cumulative, offsets.data(), 2, 4};
        auto starts = at::tensor({0, 3}, opts);
        check_close(model->predict(input, false),
                    at::linear(expected.index_select(0, starts), score, bias));
        if (std::string(arch) == "UltrabertForTokenClassification")
          check_close(model->predict(input, true), at::linear(expected, score, bias));
        else {
          bool rejected = false;
          try {
            model->predict(input, true);
          } catch (const c10::Error&) {
            rejected = true;
          }
          TORCH_CHECK(rejected, "Sequence head accepted token prediction");
        }
      }
    }
  }
  auto rejects = [&](Options bad, Weights w) {
    bool rejected = false;
    try {
      auto m = create_decoder(bad, w, c10::Device("cpu"), at::kFloat);
      m->ready();
    } catch (const std::exception&) {
      rejected = true;
    }
    TORCH_CHECK(rejected, "Invalid UltraBERT configuration accepted");
  };
  cfg.values["num_attention_heads"] = "4";
  cfg.values["num_key_value_heads"] = "2";
  for (const auto* key : {"rope_theta", "attention_windows"})
    for (const auto* value : {"[]", "[1]", "[1,2,3]", "[1,]", "[1,-2]"}) {
      auto bad = cfg;
      bad.values[key] = value;
      rejects(bad, weights(cfg));
    }
  for (const auto* key : {"attention_bias", "mlp_bias", "is_causal"}) {
    auto bad = cfg;
    bad.values[key] = "true";
    rejects(bad, weights(cfg));
  }
  for (const auto* value : {"[0,10000]", "[nan,10000]", "[inf,10000]"}) {
    auto bad = cfg;
    bad.values["rope_theta"] = value;
    rejects(bad, weights(cfg));
  }
  {
    auto bad = cfg;
    bad.values["_pooling"] = "splade";
    rejects(bad, weights(cfg));
  }
  {
    auto bad = cfg;
    bad.values["_classifier"] = "true";
    rejects(bad, weights(cfg));
  }
  std::cout
      << "UltraBERT rotary/window reference, ragged GQA, classification and validation passed\n";
}

int main(int argc, char** argv) {
  c10::InferenceMode inference;
  at::set_num_threads(1);
#ifdef TEI_TORCH_CUDA_KERNELS
  if (argc > 1 && std::string(argv[1]) == "native-stream-cuda") {
    native_engine_stream_test(c10::Device("cuda:0"));
    return 0;
  }
  if (argc > 1 && std::string(argv[1]) == "boundary-cuda") {
    cumulative_stream_test(c10::Device("cuda:0"));
    return 0;
  }
  if (argc > 1 && std::string(argv[1]) == "index-cuda") {
    indexed_stream_test(c10::Device("cuda:0"));
    return 0;
  }
  // Run the controlled allocator-poison fixture before unrelated model pools
  // can change its allocator warmup under CUDA instrumentation.
  if (argc > 1)
    indexed_stream_test(c10::Device(argv[1]));
#endif
  ultrabert_test();
  {
    auto surviving_policy = configure_workspace_admission(0, 256);
    auto policy = configure_workspace_admission(0, 128);
    auto state = workspace_admission_state(0);
    auto first = reserve_workspace_memory(0, 96);
    TORCH_CHECK(first && !reserve_workspace_memory(0, 40) && state->retained == 96,
                "Retained custom workspaces bypassed their shared per-device allowance");
    auto captured = first;
    first.reset();
    TORCH_CHECK(!reserve_workspace_memory(0, 40),
                "Captured workspace references released their allowance early");
    captured.reset();
    auto recovered = reserve_workspace_memory(0, 128);
    TORCH_CHECK(recovered && state->retained == 128,
                "Released workspace prevented valid fallback recovery");
    try {
      auto failed_engine_policy = configure_workspace_admission(0, 32);
      TORCH_CHECK(state->limit == 32, "Failed engine policy was not registered");
      throw std::runtime_error("Injected engine initialization failure");
    } catch (const std::runtime_error&) {
    }
    TORCH_CHECK(state->limit == 128 && state->retained == 128,
                "Engine initialization unwind leaked its workspace policy or permits");
    policy.reset();
    auto extended = reserve_workspace_memory(0, 128);
    TORCH_CHECK(extended && state->limit == 256 && state->retained == 256,
                "Destroyed restrictive engine policy prevented surviving workspace recovery");
    surviving_policy.reset();
    auto restrictive = configure_workspace_admission(0, 64);
    TORCH_CHECK(state->retained == 256 && !reserve_workspace_memory(0, 1),
                "Policy release erased outstanding workspace lifetime reservations");
    extended.reset();
    recovered.reset();
    restrictive.reset();
    TORCH_CHECK(state->retained == 0 && state->limit == 0 && state->policies.empty(),
                "Destroyed workspace policies were retained across engine lifetimes");
  }

  {
    c10::AcceleratorError allocation(
        {__func__, __FILE__, __LINE__}, 2, "Resource allocation failed");
    c10::AcceleratorError illegal({__func__, __FILE__, __LINE__}, 700, "Illegal device address");
    c10::AcceleratorError unknown({__func__, __FILE__, __LINE__}, 999, "Unknown device failure");
    TORCH_CHECK(accelerator_error_kind(allocation) == 3,
                "CUDA allocation failure must remain recoverable");
    TORCH_CHECK(accelerator_error_kind(illegal) == 1 && accelerator_error_kind(unknown) == 1,
                "Fatal accelerator failures must not be classified as overload");
    for (int32_t kind : {1, 2, 3}) {
      try {
        throw NativeCallError(kind);
      } catch (const NativeCallError& e) {
        TORCH_CHECK(e.kind == kind, "Nested native failure lost its structured classification");
      }
    }
  }
  bidirectional_classifier_test(c10::Device("cpu"), at::kFloat);
  decoder_token_classifier_test(c10::Device("cpu"), at::kFloat);
  if (argc > 1) {
    selected_classifier_rows_test();
    bidirectional_classifier_test(c10::Device(argv[1]), at::kBFloat16);
    decoder_token_classifier_test(c10::Device(argv[1]), at::kBFloat16);
  }
  output_capacity_bridge_test();
  input_descriptor_test();
  {
    // Preselected, non-normalized FP32 routes preserve learned expert scales
    // and their accumulation dtype, including a different activation callback.
    auto opts = at::TensorOptions().dtype(at::kBFloat16);
    auto x = at::arange(16, at::TensorOptions().dtype(at::kFloat)).view({2, 8}).to(opts);
    auto gate_up = at::zeros({3, 16, 8}, opts), down = at::zeros({3, 8, 8}, opts);
    for (int64_t e = 0; e < 3; ++e) {
      gate_up[e].narrow(0, 0, 8).copy_(at::eye(8, opts));
      gate_up[e].narrow(0, 8, 8).fill_((e + 1) / 8.);
      down[e].copy_(at::eye(8, opts));
    }
    auto ids = at::tensor({2, 0, 1, 2}, at::TensorOptions().dtype(at::kLong)).view({2, 2});
    auto routing = at::tensor({.123f, 1.501f, .379f, .941f}, at::TensorOptions().dtype(at::kFloat))
                       .view({2, 2});
    auto activation = [](const at::Tensor& both) {
      return at::relu(both.narrow(-1, 0, 8)) * both.narrow(-1, 8, 8);
    };
    auto actual = decoder_moe_dispatch(x, gate_up, down, ids, routing, activation);
    auto expected = at::zeros({2, 8}, at::TensorOptions().dtype(at::kFloat));
    for (int64_t row = 0; row < 2; ++row)
      for (int64_t slot = 0; slot < 2; ++slot) {
        auto e = ids[row][slot].item<int64_t>();
        auto y = at::linear(activation(at::linear(x[row].unsqueeze(0), gate_up[e])), down[e]);
        expected[row].add_(y.squeeze(0).to(at::kFloat) * routing[row][slot]);
      }
    TORCH_CHECK(actual.scalar_type() == at::kFloat, "Preselected MoE lost FP32 accumulation");
    check_close(actual, expected);
  }

  for (bool dense_list : {false, true}) {
    auto cfg = config("qwen3_moe", true);
    if (dense_list)
      cfg.values["mlp_only_layers"] = "[0,1]";
    else
      cfg.values["decoder_sparse_step"] = "3";
    auto dense_cfg = config("qwen3_moe");
    auto checkpoint = weights(dense_cfg);
    auto reference_weights = checkpoint;
    auto model = create_decoder(cfg, checkpoint, c10::Device("cpu"), at::kFloat);
    model->ready();
    auto reference = create_decoder(dense_cfg, reference_weights, c10::Device("cpu"), at::kFloat);
    reference->ready();
    TORCH_CHECK(at::equal(run(*model, {3, 4, 5, 6, 7, 8, 9}, {0, 3, 7}),
                          run(*reference, {3, 4, 5, 6, 7, 8, 9}, {0, 3, 7})),
                "Inactive Qwen3-MoE routing metadata changed dense outputs");
  }
  {
    auto cfg = config("qwen2");
    cfg.values["architectures.0"] = "Qwen2ForSequenceClassification";
    cfg.values["num_labels"] = "3";
    cfg.values["pad_token_id"] = "0";
    auto tensors = weights(cfg);
    tensors["score.weight"] = at::arange(48, at::kFloat).view({3, 16}) * .013;
    tensors["score.bias"] = at::tensor({.1f, -.2f, .3f});
    auto model = create_decoder(cfg, tensors, c10::Device("cpu"), at::kFloat);
    model->ready();
    TORCH_CHECK(model->supports_prefix_cache(1),
                "Loaded causal classifier did not advertise cache support");
    auto options = at::TensorOptions().dtype(at::kLong);
    std::vector<int32_t> offsets{0, 3, 7};
    PackedInput input{at::tensor({3, 4, 0, 3, 6, 7, 0}, options),
                      at::zeros({7}, options),
                      at::tensor({0, 1, 2, 0, 1, 2, 3}, options),
                      at::tensor(offsets, options.dtype(at::kInt)),
                      offsets.data(),
                      2,
                      4};
    check_prefix_cache(*model, input, 1);
    check_complete_classifier_cache(*model, input);
    auto all_pad = input;
    all_pad.ids = input.ids.clone();
    all_pad.ids.narrow(0, 0, 3).zero_();
    check_complete_classifier_cache(*model, all_pad);
    if (argc > 1) {
      auto gpu_cfg = cfg;
      gpu_cfg.values["num_attention_heads"] = "2";
      gpu_cfg.values["num_key_value_heads"] = "1";
      gpu_cfg.values["head_dim"] = "8";
      auto gpu_tensors = weights(gpu_cfg);
      gpu_tensors["score.weight"] = at::arange(48, at::kFloat).view({3, 16}) * .013;
      gpu_tensors["score.bias"] = at::tensor({.1f, -.2f, .3f});
      for (auto& [name, value] : gpu_tensors)
        value = value.to(c10::Device(argv[1]), at::kBFloat16);
      auto no_pad_tensors = gpu_tensors;
      auto gpu = create_decoder(gpu_cfg, gpu_tensors, c10::Device(argv[1]), at::kBFloat16);
      gpu->ready();
      auto device_input = input;
      device_input.ids = input.ids.to(c10::Device(argv[1]));
      device_input.types = input.types.to(c10::Device(argv[1]));
      device_input.positions = input.positions.to(c10::Device(argv[1]));
      device_input.cumulative = input.cumulative.to(c10::Device(argv[1]));
      check_complete_classifier_cache(*gpu, device_input);
      device_input.ids = all_pad.ids.to(c10::Device(argv[1]));
      check_complete_classifier_cache(*gpu, device_input);
      auto no_pad_cfg = gpu_cfg;
      no_pad_cfg.values.erase("pad_token_id");
      auto no_pad = create_decoder(no_pad_cfg, no_pad_tensors, c10::Device(argv[1]), at::kBFloat16);
      no_pad->ready();
      auto integer_ids = device_input;
      integer_ids.ids = device_input.ids.to(at::kInt);
      TORCH_CHECK(
          at::equal(no_pad->predict(device_input, false), no_pad->predict(integer_ids, false)),
          "Packed selector rejected or changed valid INT32 token IDs without PAD configuration");
    }
  }
  for (auto family : {"llama", "mistral", "qwen2", "qwen3", "qwen3_moe"}) {
    auto cfg = config(family, std::string(family) == "qwen3_moe");
    auto w = weights(cfg);
    auto initial = w;
    auto model = create_decoder(cfg, initial, c10::Device("cpu"), at::kFloat);
    model->ready();
    TORCH_CHECK(!model->supports_prefix_cache(1),
                "Backbone without a score head advertised classifier cache support");
    radix_decoder_test(*model);
    cached_decoder_test(*model);
    if (std::string(family) == "qwen3") {
      cache_admission_test(*model);
      cache_concurrency_test(*model);
    }
    if (argc > 1) {
      auto gpu_cfg = cfg;
      gpu_cfg.values["num_attention_heads"] = "2";
      gpu_cfg.values["num_key_value_heads"] = "1";
      gpu_cfg.values["head_dim"] = "8";
      auto gpu_weights = w;
      for (auto& [name, value] : gpu_weights)
        if (name.ends_with("q_norm.weight") || name.ends_with("k_norm.weight"))
          value = value.repeat({2});
      for (auto& [name, value] : gpu_weights)
        value = value.to(c10::Device(argv[1]), at::kBFloat16);
      auto gpu = create_decoder(gpu_cfg, gpu_weights, c10::Device(argv[1]), at::kBFloat16);
      gpu->ready();
      auto opts = at::TensorOptions().device(c10::Device(argv[1])).dtype(at::kLong);
      std::vector<int32_t> ends = {0, 3, 7};
      PackedInput input{at::tensor({3, 4, 5, 3, 4, 8, 9}, opts),
                        at::zeros({7}, opts),
                        at::tensor({0, 1, 2, 0, 1, 2, 3}, opts),
                        at::tensor(ends, opts.dtype(at::kInt)),
                        ends.data(),
                        2,
                        4};
      check_prefix_cache(*gpu, input, 0, .01, .02);
#ifdef TEI_TORCH_CUDA_KERNELS
      if (std::string(family) == "qwen3") {
        cumulative_stream_test(c10::Device(argv[1]));
        cache_stream_test(*gpu, input);
        suffix_graph_test(*gpu, input);
      }
      if (std::string(family) == "qwen2")
        cache_cold_maximum_test(c10::Device(argv[1]));
#endif
    }
    auto packed = run(*model, {3, 4, 5, 6, 7, 8, 9}, {0, 3, 7});
    check_close(packed,
                at::cat({run(*model, {3, 4, 5}, {0, 3}), run(*model, {6, 7, 8, 9}, {0, 4})}, 0));
    check_close(run(*model, {3, 4}, {0, 2}), run(*model, {3, 4, 5}, {0, 3}).narrow(0, 0, 2));
    if (std::string(family) == "qwen2" || std::string(family) == "qwen3") {
      auto scaledcfg = cfg;
      scaledcfg.values["rope_scaling.rope_type"] = "llama3";
      scaledcfg.values["rope_scaling.factor"] = "4";
      scaledcfg.values["rope_scaling.high_freq_factor"] = "4";
      auto scaledweights = weights(scaledcfg);
      auto scaled = create_decoder(scaledcfg, scaledweights, c10::Device("cpu"), at::kFloat);
      scaled->ready();
      check_close(packed, run(*scaled, {3, 4, 5, 6, 7, 8, 9}, {0, 3, 7}));
    }
    if (std::string(family) == "qwen3_moe") {
      auto unsupportedcfg = cfg;
      unsupportedcfg.values["rope_scaling.factor"] = "4";
      auto unsupportedweights = weights(unsupportedcfg);
      auto unsupported =
          create_decoder(unsupportedcfg, unsupportedweights, c10::Device("cpu"), at::kFloat);
      bool rejected = false;
      try {
        unsupported->ready();
      } catch (const c10::Error&) {
        rejected = true;
      }
      TORCH_CHECK(rejected, "Scaled Qwen3 MoE configuration was silently accepted");
    }
    if (std::string(family) == "qwen3_moe") {
      auto fused = weights(cfg, true);
      auto other = create_decoder(cfg, fused, c10::Device("cpu"), at::kFloat);
      other->ready();
      check_close(packed, run(*other, {3, 4, 5, 6, 7, 8, 9}, {0, 3, 7}));
    }
    cfg.values["use_bidirectional_attention"] = "true";
    if (std::string(family) == "qwen2")
      cfg.values["is_causal"] = "false";
    auto bidi = create_decoder(cfg, w, c10::Device("cpu"), at::kFloat);
    bidi->ready();
    auto b = run(*bidi, {3, 4, 5}, {0, 3});
    TORCH_CHECK((b.narrow(0, 0, 2) - run(*bidi, {3, 4}, {0, 2})).abs().max().item<float>() > 1e-5,
                "Bidirectional attention was accidentally causal");
    radix_decoder_test(*bidi);
    TORCH_CHECK(!bidi->supports_radix(),
                "Bidirectional decoder incorrectly advertises prefix folding");
    check_close(run(*bidi, {3, 4, 5, 6, 7}, {0, 3, 5}),
                at::cat({b, run(*bidi, {6, 7}, {0, 2})}, 0));
    std::cout << family << " packed boundaries, causality and bidirectional attention passed\n";
  }
  multimodal_test();
  probe_cache_test();
  native_reconfigure_test(c10::Device("cpu"));
#ifdef TEI_TORCH_CUDA_KERNELS
  if (argc > 1) {
    linear_workspace_test(c10::Device(argv[1]));
    native_engine_stream_test(c10::Device(argv[1]));
    native_reconfigure_test(c10::Device(argv[1]));
  }
#endif
}
