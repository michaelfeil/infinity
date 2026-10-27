// SPDX-License-Identifier: Apache-2.0
#include "models/decision_models.h"
#include "batching/input_validation.h"
#include "helpers/admission_error.h"
#include <c10/core/InferenceMode.h>
#include <ATen/Parallel.h>
#include <iostream>
using namespace infinity;

namespace {
void close(const at::Tensor& a, const at::Tensor& b, double tol = 1e-5) {
  TORCH_CHECK((a.to(at::kFloat) - b.to(at::kFloat)).abs().max().item<float>() < tol,
              "Decision expected result mismatch");
}

void reject(const std::function<void()>& run) {
  bool failed = false;
  try {
    run();
  } catch (const c10::Error&) {
    failed = true;
  }
  TORCH_CHECK(failed, "Invalid decision request was accepted");
}

void norm(Weights& w, const std::string& p, int64_t n) {
  w[p + ".weight"] = at::ones({n});
  w[p + ".bias"] = at::zeros({n});
}

void linear(Weights& w, const std::string& p, int64_t in, int64_t out, bool bias = true) {
  w[p + ".weight"] = at::zeros({out, in});
  if (bias)
    w[p + ".bias"] = at::zeros({out});
}

void attention(Weights& w, const std::string& p, int64_t n) {
  w[p + ".in_proj_weight"] = at::zeros({3 * n, n});
  w[p + ".in_proj_bias"] = at::zeros({3 * n});
  linear(w, p + ".out_proj", n, n);
}

struct Input {
  std::vector<int32_t> offsets{0, 3, 5};
  PackedInput input;

  Input() {
    auto ids = at::tensor({1, 2, 3, 4, 5}, at::TensorOptions().dtype(at::kLong));
    input = {ids,
             at::zeros_like(ids),
             at::tensor({0, 1, 2, 0, 1}, ids.options()),
             at::tensor({0, 3, 5}, ids.options().dtype(at::kInt)),
             offsets.data(),
             2,
             3};
  }
};

void options_test() {
  Input in;
  auto hidden = at::arange(80).view({5, 16}).to(at::kFloat) * .01;
  Weights backbone;
  backbone["lm_head.weight"] = at::arange(128).view({8, 16}).to(at::kFloat) * .001;
  Options cfg;
  cfg.values = {{"hidden_size", "16"}, {"model_type", "qwen3_5"}};
  auto head = create_decision_head("option_tokens", {}, cfg, {}, backbone);
  backbone.clear();
  DecisionRequest first{DecisionRequest::Kind::OptionTokens}, second{DecisionRequest::Kind::Warmup};
  first.token_ids = {4, 1};
  auto result = head->forward(hidden, in.input, {first, second});
  auto expectedweights = at::arange(128).view({8, 16}).to(at::kFloat) * .001;
  close(result[0].logits,
        at::matmul(expectedweights.index_select(0, at::tensor({4, 1}, in.input.ids.options())),
                   hidden[2]));
  close(result[1].logits, at::matmul(expectedweights[0], hidden[4]).view({1}));
  TORCH_CHECK(head->output_count(first) == 2 && head->output_count(second) == 1,
              "Incorrect option-token output count");
  close(result[0].action_probability, at::ones({}));
  first.token_ids = {8};
  reject([&] { head->output_count(first); });
  cfg.values["model_type"] = "gemma4";
  cfg.values["final_logit_softcapping"] = ".5";
  cfg.values["tie_word_embeddings"] = "false";
  cfg.values["text_config.tie_word_embeddings"] = "true";
  backbone["model.embed_tokens.weight"] = expectedweights;
  auto gemma = create_decision_head("option_tokens", {}, cfg, {}, backbone);
  first.token_ids = {4, 1};
  auto scores = gemma->forward(hidden, in.input, {first, second});
  close(scores[0].logits, at::tanh(result[0].logits / .5) * .5);
  Weights readout;
  readout["weight"] = at::arange(255 * 16).view({255, 16}).to(at::kBFloat16) * .001;
  auto pplx = create_decision_head("pplx", {}, cfg, readout, {});
  first.token_ids = {0, 1};
  auto values = pplx->forward(hidden, in.input, {first, second});
  close(values[0].logits,
        at::linear(hidden[2].to(at::kBFloat16).unsqueeze(0), readout["weight"].narrow(0, 0, 2))
            .flatten()
            .to(at::kFloat));
  DecisionRequest joint{DecisionRequest::Kind::InlineOptions};
  joint.markers = {0, 1};
  joint.token_ids = {2, 3};
  auto joint_logits = pplx->forward(hidden, in.input, {joint, second})[0].logits;
  auto all_logits =
      at::linear(hidden.narrow(0, 0, 2).to(at::kBFloat16), readout["weight"]).to(at::kFloat);
  close(joint_logits, at::cat({all_logits[0].narrow(0, 0, 2), all_logits[1].narrow(0, 0, 3)}));
  TORCH_CHECK(pplx->output_count(joint) == 5, "Wrong inline output count");
  joint.markers = {0, 3};
  reject([&] { pplx->forward(hidden, in.input, {joint, second}); });
  joint.markers = {1, 0};
  reject([&] { pplx->output_count(joint); });
  joint.markers = {0};
  reject([&] { pplx->output_count(joint); });
  joint.token_ids = {256};
  reject([&] { pplx->output_count(joint); });
  first.token_ids = {1, 0};
  reject([&] { pplx->output_count(first); });
  Options bad;
  bad.values["pooling"] = "mean";
  reject([&] { create_decision_head("pplx", bad, cfg, readout, {}); });
  std::cout
      << "Option tokens, tied embeddings, last-token selection, Gemma softcap and BF16 Pplx readout passed\n";
}

Weights laya_weights() {
  Weights w;
  w["type_emb.weight"] = at::zeros({3, 16});
  norm(w, "scorer.0", 16);
  linear(w, "scorer.1", 16, 16);
  linear(w, "scorer.3", 16, 1);
  linear(w, "act_head.0", 20, 256);
  linear(w, "act_head.2", 256, 2);
  w["act_head.0.weight"][0][16] = 1.;
  w["act_head.0.weight"][0][18] = 1.;
  w["act_head.2.weight"][0][0] = 1.;
  auto p = std::string("head.layers.0");
  norm(w, p + ".norm1", 16);
  norm(w, p + ".norm2", 16);
  attention(w, p + ".self_attn", 16);
  linear(w, p + ".linear1", 16, 64);
  linear(w, p + ".linear2", 64, 16);
  return w;
}

void laya_test() {
  Input in;
  Options cfg, backbone;
  cfg.values = {{"head_layers", "1"}, {"max_len", "32"}, {"head_max_len", "32"}};
  backbone.values["hidden_size"] = "16";
  auto weights = laya_weights();
  auto head = create_decision_head("laya", cfg, backbone, weights, {});
  weights.clear();
  DecisionRequest first{DecisionRequest::Kind::Laya}, second{DecisionRequest::Kind::Warmup};
  first.question_type = 2;
  first.markers = {0, 2};
  auto hidden = at::sin(at::arange(80).view({5, 16}).to(at::kFloat));
  auto out = head->forward(hidden, in.input, {first, second});
  close(out[0].logits, at::zeros({2}));
  close(out[1].logits, at::zeros({1}));
  close(out[0].action_probability, at::sigmoid(at::gelu(at::tensor(1.5), "none")));
  close(out[1].action_probability, at::sigmoid(at::gelu(at::tensor(1.), "none")));
  TORCH_CHECK(head->output_count(first) == 2 && head->output_count(second) == 1,
              "Wrong Laya output count");
  first.markers = {3};
  reject([&] { head->forward(hidden, in.input, {first, second}); });
  first.question_type = 3;
  reject([&] { head->output_count(first); });
  std::cout
      << "Laya head, learned action top1/entropy features, warmup and marker validation passed\n";
}

void laya_packed_cuda_test() {
  for (auto dtype : {at::kBFloat16, at::kHalf}) {
    Input in;
    Options cfg, backbone;
    cfg.values = {{"head_layers", "1"}, {"max_len", "32"}, {"head_max_len", "32"}};
    backbone.values["hidden_size"] = "16";
    auto weights = laya_weights();
    auto device = c10::Device("cuda:0");
    // Nonzero attention/FFN/scorer weights expose cross-sequence leakage. The
    // existing feature-dependent action weights keep probabilities nonsaturated.
    for (auto& [name, value] : weights) {
      if (name == "type_emb.weight" || name.starts_with("head.layers.0.self_attn") ||
          name.starts_with("head.layers.0.linear") || name == "scorer.1.weight" ||
          name == "scorer.3.weight")
        value =
            at::sin(at::arange(value.numel()).reshape(value.sizes()).to(at::kFloat) * .17) * .04;
      value = value.to(device, dtype);
    }
    auto head = create_decision_head("laya", cfg, backbone, weights, {});
    in.input.ids = in.input.ids.to(device);
    in.input.types = in.input.types.to(device);
    in.input.positions = in.input.positions.to(device);
    in.input.cumulative = in.input.cumulative.to(device);
    auto hidden = at::sin(at::arange(80).reshape({5, 16}).to(at::kFloat) * .13).to(device, dtype);
    DecisionRequest first{DecisionRequest::Kind::Laya}, second{DecisionRequest::Kind::Warmup};
    first.question_type = 2;
    first.markers = {2, 0, 2};
    auto output = head->forward(hidden, in.input, {first, second});
    auto metadata = head->prepare(in.input, {first, second});
    auto staged = head->forward_prepared(hidden, in.input, {first, second}, metadata);
    for (size_t i = 0; i < output.size(); ++i) {
      TORCH_CHECK(at::equal(staged[i].logits, output[i].logits), "Prepared logits changed");
      TORCH_CHECK(at::equal(staged[i].action_probability, output[i].action_probability),
                  "Prepared actions changed");
    }
    TORCH_CHECK(output[0].action_probability.item<float>() > .05 &&
                    output[0].action_probability.item<float>() < .95,
                "Action regression must be nonsaturated");
    for (int64_t i = 0; i < 2; ++i) {
      auto start = in.offsets[i], length = in.offsets[i + 1] - start;
      int32_t offsets[] = {0, static_cast<int32_t>(length)};
      PackedInput single{in.input.ids.narrow(0, start, length),
                         in.input.types.narrow(0, start, length),
                         in.input.positions.narrow(0, start, length),
                         at::tensor({0, length}, in.input.cumulative.options()),
                         offsets,
                         1,
                         length};
      auto reference = head->forward(hidden.narrow(0, start, length), single, {i ? second : first});
      close(output[i].logits, reference[0].logits, .006);
      close(output[i].action_probability, reference[0].action_probability, .001);
    }
    auto changed = hidden.clone();
    changed.narrow(0, 3, 2).fill_(17.);
    auto isolated = head->forward(changed, in.input, {first, second});
    close(output[0].logits, isolated[0].logits, .00001);
    close(output[0].action_probability, isolated[0].action_probability, .00001);
    first.question_type = 1;
    first.markers = {1};
    auto updated = head->forward(hidden, in.input, {first, second});
    auto changed_metadata = head->prepare(in.input, {first, second});
    auto prepared_update =
        head->forward_prepared(hidden, in.input, {first, second}, changed_metadata);
    TORCH_CHECK(at::equal(prepared_update[0].logits, updated[0].logits),
                "Updated prepared logits changed");
    TORCH_CHECK(at::equal(prepared_update[0].action_probability, updated[0].action_probability),
                "Updated prepared actions changed");
    TORCH_CHECK(updated[0].logits.numel() == 1 && updated[1].logits.numel() == 1,
                "Packed head retained old marker metadata");
    first.markers = {3};
    reject([&] { head->forward(hidden, in.input, {first, second}); });
    first.markers = {0};
    auto invalid = in.input;
    int32_t invalid_offsets[] = {-1, 3, 5};
    invalid.offsets = invalid_offsets;
    reject([&] { head->forward(hidden, invalid, {first, second}); });
  }
  std::cout
      << "Laya CUDA packed head FP16/BF16: nonsaturated action, repeated/unsorted markers, singleton warmup, changed metadata and sequence isolation passed\n";
}

Weights clef_weights() {
  Weights w;
  for (auto p : {"hidden_norm", "option_summary_norm", "field_norm", "option_norm"})
    norm(w, p, 16);
  for (auto p : {"memory_projection",
                 "question_projection",
                 "option_question_projection",
                 "global_projection",
                 "option_context_projection",
                 "option_lexical_projection"})
    linear(w, p, 16, 16, false);
  w["option_lexical_projection.weight"] = at::eye(16);
  w["type_embedding.weight"] = at::zeros({3, 16});
  linear(w, "residual_scorer.0", 64, 16);
  linear(w, "residual_scorer.3", 16, 1);
  w["prior_logit_scale"] = at::tensor(0.);
  w["joint_logit_scale"] = at::tensor(0.);
  w["residual_gate"] = at::tensor(-100.);
  auto p = std::string("evidence_layers.0");
  for (auto n : {"query_norm", "memory_norm", "feedforward_norm"})
    norm(w, p + "." + n, 16);
  attention(w, p + ".attention", 16);
  linear(w, p + ".feedforward.0", 16, 32);
  linear(w, p + ".feedforward.3", 32, 16);
  p = "layers.0";
  for (auto n : {"norm1", "norm2", "norm3"})
    norm(w, p + "." + n, 16);
  attention(w, p + ".self_attn", 16);
  attention(w, p + ".multihead_attn", 16);
  linear(w, p + ".linear1", 16, 32);
  linear(w, p + ".linear2", 32, 16);
  return w;
}

void clef_test() {
  Input in;
  Options cfg, backbone;
  cfg.values = {{"hidden_size", "16"},
                {"width", "16"},
                {"heads", "2"},
                {"feedforward", "32"},
                {"routing_layers", "1"},
                {"layers", "1"}};
  backbone.values = {{"hidden_size", "16"}, {"model_type", "qwen3_5"}};
  auto weights = clef_weights();
  Weights bw;
  bw["lm_head.weight"] = at::cos(at::arange(128).view({8, 16}).to(at::kFloat) * .13);
  auto lexical = bw["lm_head.weight"];
  auto head = create_decision_head("clef", cfg, backbone, weights, bw);
  weights.clear();
  bw.clear();
  DecisionRequest first{DecisionRequest::Kind::Clef}, second{DecisionRequest::Kind::Warmup};
  first.fields = {{0, {0, 1}, {{1, 2}, {2, 3}}}, {2, {1, 2}, {{0, 2}}}};
  auto hidden = at::sin(at::arange(80).view({5, 16}).to(at::kFloat) * .17);
  auto out = head->forward(hidden, in.input, {first, second});
  auto normalized =
      at::layer_norm(hidden.narrow(0, 0, 3), {16}, at::ones({16}), at::zeros({16}), 1e-5);
  std::vector<at::Tensor> expected;
  for (const auto& f : first.fields) {
    auto anchor =
        normalized.narrow(0, f.question.first, f.question.second - f.question.first).mean(0) +
        normalized[2];
    anchor = anchor / anchor.square().sum().sqrt();
    for (auto [a, b] : f.options) {
      auto word = lexical.index_select(0, in.input.ids.narrow(0, a, b - a)).mean(0);
      word = word / word.square().sum().sqrt();
      expected.push_back((word * anchor).sum());
    }
  }
  close(out[0].logits, at::stack(expected));
  TORCH_CHECK(head->output_count(first) == 3 && head->output_count(second) == 1,
              "Wrong Clef output counts");
  close(out[1].action_probability, at::ones({}));
  auto changed = hidden.clone();
  changed.narrow(0, 3, 2).fill_(17.);
  close(head->forward(changed, in.input, {first, second})[0].logits, out[0].logits);
  first.fields[0].options = {{2, 4}};
  reject([&] { head->forward(hidden, in.input, {first, second}); });
  std::cout
      << "Clef lexical prior/span means, routing/field decoder, variable option counts and sequence isolation passed\n";
}
}

void d3_test() {
  // Values straddling a BF16 midpoint expose FP32->FP16->BF16 double rounding.
  auto pixels =
      at::tensor({1.0038f, 1.00395f, -1.0038f, -1.00395f}, at::TensorOptions().dtype(at::kFloat));
  auto uploaded = upload_image_pixels(pixels, c10::Device(at::kCPU), at::kBFloat16, "qwen3_5");
  TORCH_CHECK(at::equal(uploaded, pixels.to(at::kBFloat16)),
              "D3 image upload must cast directly to BF16");
  TORCH_CHECK(!at::equal(uploaded, pixels.to(at::kHalf).to(at::kBFloat16)),
              "D3 fixture must distinguish double rounding");
  TORCH_CHECK(at::equal(upload_image_pixels(pixels, c10::Device(at::kCPU), at::kHalf, "qwen3_5"),
                        pixels.to(at::kHalf)),
              "Explicit FP16 image upload must retain its requested dtype");
  // One prepared image is shared by all questions, with one media reservation.
  RequestImageUploads uploads;
  int upload_calls = 0;
  auto upload = [&] {
    ++upload_calls;
    return upload_image_pixels(pixels, c10::Device(at::kCPU), at::kBFloat16, "qwen3_5");
  };
  auto first_upload = uploads.get(pixels, upload), second_upload = uploads.get(pixels, upload);
  TORCH_CHECK(upload_calls == 1 &&
                  first_upload.unsafeGetTensorImpl() == second_upload.unsafeGetTensorImpl(),
              "Shared D3 pixels uploaded once per request");
  auto reshaped = pixels.view({2, 2});
  uploads.get(reshaped, [&] {
    ++upload_calls;
    return reshaped.to(at::kBFloat16);
  });
  TORCH_CHECK(upload_calls == 2, "Distinct host layout incorrectly reused upload");
  uploads.get(reshaped, [&] {
    ++upload_calls;
    return reshaped.to(at::kBFloat16);
  });
  TORCH_CHECK(upload_calls == 2, "Matrix upload identity differs from admission identity");
  auto transposed = reshaped.transpose(0, 1);
  uploads.get(transposed, [&] {
    ++upload_calls;
    return transposed.to(at::kBFloat16);
  });
  TORCH_CHECK(upload_calls == 3, "Distinct matrix strides incorrectly reused upload");
  // Admission reads descriptors only, even before the caller's pixels are readable.
  TeiImage aliases[2]{};
  for (auto& image : aliases) {
    image.pixels = reinterpret_cast<const float*>(uintptr_t(4096));
    image.rows = 16;
    image.patch_dim = 16;
  }
  auto admitted = [&](size_t limit, bool dedup) {
    return validate_input_upload_budget(
        2, 2, false, aliases, 2, nullptr, 0, nullptr, 2, limit, false, dedup);
  };
  const auto shared_bytes = admitted(1 << 20, true);
  TORCH_CHECK(admitted(shared_bytes, true) == shared_bytes,
              "Shared uploads rejected at their exact byte budget");
  auto reject_budget = [&](bool dedup, size_t limit) {
    bool failed = false;
    try {
      admitted(limit, dedup);
    } catch (const MemoryAdmissionError&) {
      failed = true;
    }
    TORCH_CHECK(failed, "Over-budget upload did not reject as memory admission");
  };
  reject_budget(true, shared_bytes - 1);
  reject_budget(false, shared_bytes); // Non-D3 uploads remain distinct.
  aliases[1].pixels = reinterpret_cast<const float*>(uintptr_t(8192));
  reject_budget(true, shared_bytes);
  aliases[1].pixels = aliases[0].pixels;
  aliases[1].rows = 32;
  aliases[1].patch_dim = 8;
  reject_budget(true, shared_bytes); // Same pointer/count, different matrix layout.
  int vision_calls = 0;
  ImageInput first_image{first_upload, {1, 2, 2}, 2, 1, 1, 0};
  auto next_question = first_image;
  next_question.token_start = 9;
  next_question.sequence_start = 8;
  RequestImageFeatures features({first_image, next_question});
  auto compute = [&] {
    ++vision_calls;
    return at::ones({1, 16});
  };
  auto first_features = features.get(first_image, compute),
       second_features = features.get(next_question, compute);
  TORCH_CHECK(vision_calls == 1 &&
                  first_features.unsafeGetTensorImpl() == second_features.unsafeGetTensorImpl(),
              "Shared D3 vision computed once across question positions");
  auto other_grid = first_image;
  other_grid.grid_thw = {1, 1, 4};
  features.get(other_grid, compute);
  auto distinct_pixels = first_image;
  distinct_pixels.pixels = first_upload.clone();
  features.get(distinct_pixels, compute);
  TORCH_CHECK(vision_calls == 3, "Distinct pixels/grid incorrectly reused D3 vision");
  c10::weak_intrusive_ptr<at::TensorImpl> last_feature(first_features.getIntrusivePtr());
  first_features = at::Tensor();
  second_features = at::Tensor();
  TORCH_CHECK(last_feature.expired(), "D3 feature memo retained pixels after their final consumer");
  {
    RequestImageFeatures next_request({first_image});
    auto unique = next_request.get(first_image, compute);
    c10::weak_intrusive_ptr<at::TensorImpl> unique_feature(unique.getIntrusivePtr());
    unique = at::Tensor();
    TORCH_CHECK(unique_feature.expired(), "Unique D3 media feature should not enter memo");
  }
  TORCH_CHECK(vision_calls == 4, "D3 vision must not survive a native request");
  Input in;
  Options cfg, backbone;
  cfg.values = {{"format_id", "d3-code-readout-v1"},
                {"prompt", "d3"},
                {"pooling", "last"},
                {"attention_mode", "noncausal_full_attention"},
                {"readout_dtype", "float32"}};
  backbone.values = {{"model_type", "qwen3_5"}, {"hidden_size", "16"}};
  Weights readout;
  readout["weight"] = at::sin(at::arange(255 * 16).view({255, 16}).to(at::kFloat) * .017);
  auto hidden = (at::cos(at::arange(80).view({5, 16}).to(at::kFloat) * .13)).to(at::kBFloat16);
  auto block_cfg = cfg;
  block_cfg.values["attention_mode"] = "prefix_bidir";
  auto block_head = create_decision_head("d3", block_cfg, backbone, readout, {});
  auto head = create_decision_head("d3", cfg, backbone, readout, {});
  DecisionRequest first{DecisionRequest::Kind::OptionTokens}, second{DecisionRequest::Kind::Warmup};
  first.token_ids = {0, 1, 2};
  auto result = head->forward(hidden, in.input, {first, second});
  auto block_result = block_head->forward(hidden, in.input, {first, second});
  close(block_result[0].logits, result[0].logits);
  close(block_result[1].logits, result[1].logits);
  auto full = at::linear(hidden[2].to(at::kFloat).unsqueeze(0), readout["weight"]);
  close(result[0].logits, full.narrow(1, 0, 3).flatten());
  close(result[1].logits,
        at::linear(hidden[4].to(at::kFloat).unsqueeze(0), readout["weight"])
            .narrow(1, 0, 1)
            .flatten());
  auto bf16 = at::linear(hidden[2].unsqueeze(0), readout["weight"].to(at::kBFloat16))
                  .narrow(1, 0, 3)
                  .flatten();
  TORCH_CHECK((result[0].logits - bf16.to(at::kFloat)).abs().max().item<float>() > 1e-4,
              "D3 precision fixture must distinguish FP32 from BF16 readout");
  first.token_ids = {1, 0};
  reject([&] { head->output_count(first); });
  for (const auto& entry :
       std::vector<std::pair<std::string, std::string>>{{"pooling", "mean"},
                                                        {"attention_mode", "causal"},
                                                        {"readout_dtype", "bfloat16"},
                                                        {"prompt", "pplx"}}) {
    auto bad = cfg;
    bad.values[entry.first] = entry.second;
    reject([&] { create_decision_head("d3", bad, backbone, readout, {}); });
  }
  readout["weight"] = readout["weight"].to(at::kBFloat16);
  reject([&] { create_decision_head("d3", cfg, backbone, readout, {}); });
  std::cout
      << "D3 FP32 full readout, last-token selection, warmup and checkpoint validation passed\n";
}

int main(int argc, char** argv) {
  c10::InferenceMode inference;
  at::set_num_threads(1);
  options_test();
  d3_test();
  laya_test();
  clef_test();
  if (argc > 1 && std::string(argv[1]) == "--cuda")
    laya_packed_cuda_test();
}
