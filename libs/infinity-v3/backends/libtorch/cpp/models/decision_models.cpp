// SPDX-License-Identifier: Apache-2.0
// Ports the typed head contracts in Candle models/laya.rs and models/clef.rs.
#include "models/decision_models.h"
#ifdef TEI_TORCH_CUDA_KERNELS
#include "kernels/laya_features.h"
#endif
#include <ATen/ops/_flash_attention_forward.h>
#include <cmath>
#include <algorithm>

namespace tei {
namespace {
struct Linear {
  at::Tensor w, b;
  Linear() = default;

  Linear(
      const Weights& weights, const std::string& name, int64_t in, int64_t out, bool bias = true) {
    w = weight(weights, name + ".weight");
    TORCH_CHECK(w.sizes() == at::IntArrayRef({out, in}), "Invalid decision linear weight: ", name);
    if (bias) {
      b = weight(weights, name + ".bias");
      TORCH_CHECK(b.numel() == out, "Invalid decision linear bias: ", name);
    }
  }

  at::Tensor operator()(const at::Tensor& x) const {
    return at::linear(x, w, b);
  }
};

struct Norm {
  at::Tensor w, b;
  Norm() = default;

  Norm(const Weights& weights, const std::string& name, int64_t width) {
    w = weight(weights, name + ".weight");
    b = weight(weights, name + ".bias");
    TORCH_CHECK(w.sizes() == at::IntArrayRef({width}) && b.sizes() == w.sizes(),
                "Invalid decision norm: ",
                name);
  }

  at::Tensor operator()(const at::Tensor& x) const {
    return at::layer_norm(x, {w.numel()}, w, b, 1e-5);
  }
};

struct Attention {
  at::Tensor qkv, bias;
  Linear out;
  int64_t heads;
  bool fuse_self_qkv;

  Attention(const Weights& weights,
            const std::string& name,
            int64_t width,
            int64_t count,
            bool fuse = false)
      : qkv(weight(weights, name + ".in_proj_weight")),
        bias(weight(weights, name + ".in_proj_bias")),
        out(weights, name + ".out_proj", width, width), heads(count), fuse_self_qkv(fuse) {
    TORCH_CHECK(qkv.sizes() == at::IntArrayRef({3 * width, width}) && bias.numel() == 3 * width &&
                    count > 0 && width % count == 0,
                "Invalid decision attention geometry");
  }

  at::Tensor operator()(const at::Tensor& query, const at::Tensor& memory) const {
    auto n = query.size(0), m = memory.size(0), width = query.size(1), dim = width / heads;
    at::Tensor q, k, v;
    // Laya's published TransformerEncoderLayer projects Q/K/V together. Exact
    // tensor identity guarantees self-attention; Clef's separate projections
    // retain their existing contract, including cross-attention memories.
    if (fuse_self_qkv && query.unsafeGetTensorImpl() == memory.unsafeGetTensorImpl()) {
      auto projected = at::linear(query, qkv, bias).view({n, 3, heads, dim});
      q = projected.select(1, 0);
      k = projected.select(1, 1);
      v = projected.select(1, 2);
    } else {
      q = at::linear(query, qkv.narrow(0, 0, width), bias.narrow(0, 0, width))
              .view({n, heads, dim});
      k = at::linear(memory, qkv.narrow(0, width, width), bias.narrow(0, width, width))
              .view({m, heads, dim});
      v = at::linear(memory, qkv.narrow(0, 2 * width, width), bias.narrow(0, 2 * width, width))
              .view({m, heads, dim});
    }
    at::Tensor attended;
    if (query.is_cuda()) {
      TORCH_CHECK((query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16) &&
                      dim % 8 == 0 && dim <= 256,
                  "CUDA decision varlen attention requires FP16/BF16 and supported head dimension");
      auto opts = query.options().dtype(at::kInt);
      auto cuq = at::tensor({int64_t(0), n}, opts), cuk = at::tensor({int64_t(0), m}, opts);
      attended = std::get<0>(at::_flash_attention_forward(
          q, k, v, cuq, cuk, n, m, 0., false, false, 1. / std::sqrt(double(dim))));
    } else {
      auto transform = [](const at::Tensor& x) { return x.transpose(0, 1).unsqueeze(0); };
      attended = at::scaled_dot_product_attention(transform(q), transform(k), transform(v))
                     .squeeze(0)
                     .transpose(0, 1);
    }
    return out(attended.contiguous().view({n, width}));
  }

  at::Tensor packed_self(const at::Tensor& x, const PackedInput& input) const {
    auto tokens = x.size(0), width = x.size(1), dim = width / heads;
    TORCH_CHECK(x.is_cuda() && (x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16) &&
                    dim % 8 == 0 && dim <= 256,
                "Packed decision attention requires CUDA FP16/BF16");
    auto projected = at::linear(x, qkv, bias).view({tokens, 3, heads, dim});
    auto attended = std::get<0>(at::_flash_attention_forward(projected.select(1, 0),
                                                             projected.select(1, 1),
                                                             projected.select(1, 2),
                                                             input.cumulative,
                                                             input.cumulative,
                                                             input.max_sequence,
                                                             input.max_sequence,
                                                             0.,
                                                             false,
                                                             false,
                                                             1. / std::sqrt(double(dim))));
    return out(attended.contiguous().view({tokens, width}));
  }
};

at::Tensor span(const at::Tensor& x, std::pair<int64_t, int64_t> interval) {
  auto [a, b] = interval;
  TORCH_CHECK(a >= 0 && a < b && b <= x.size(0), "Invalid decision token span");
  return x.narrow(0, a, b - a).to(at::kFloat).mean(0).to(x.scalar_type());
}

at::Tensor normalize(const at::Tensor& x) {
  auto denominator =
      x.to(at::kFloat).square().sum(-1, true).sqrt().clamp_min(1e-12).to(x.scalar_type());
  return x / denominator;
}

at::Tensor lexical(const Options& cfg, const Weights& weights) {
  auto tied = cfg.boolean("tie_word_embeddings", false) ||
              cfg.boolean("text_config.tie_word_embeddings", false);
  if (!tied) {
    auto it = weights.find("lm_head.weight");
    TORCH_CHECK(it != weights.end(), "Decision checkpoint requires lm_head.weight");
    return it->second;
  }
  for (auto prefix : {"model.language_model.", "model.", "language_model.", ""}) {
    auto it = weights.find(std::string(prefix) + "embed_tokens.weight");
    if (it != weights.end())
      return it->second;
  }
  TORCH_CHECK(false, "Decision checkpoint has no tied token embedding weights");
}

void validate(const at::Tensor& hidden,
              const PackedInput& input,
              const std::vector<DecisionRequest>& requests) {
  TORCH_CHECK(hidden.dim() == 2 && hidden.size(0) == input.ids.numel() &&
                  int64_t(requests.size()) == input.batch,
              "Decision metadata count or hidden geometry does not match packed batch");
  for (int64_t i = 0; i < input.batch; ++i)
    TORCH_CHECK(input.offsets[i + 1] > input.offsets[i],
                "Typed decisions require nonempty sequences");
}

struct OptionHead final : DecisionHead {
  at::Tensor weights;
  bool external, qwen, d3;
  double cap;

  OptionHead(const std::string& kind,
             const Options& cfg,
             const Weights& head,
             const Weights& backbone)
      : weights((kind == "pplx" || kind == "d3") ? weight(head, "weight") : lexical(cfg, backbone)),
        external(kind == "pplx" || kind == "d3"),
        qwen(cfg.string("model_type").rfind("qwen", 0) == 0), d3(kind == "d3"),
        cap(cfg.real("final_logit_softcapping",
                     cfg.real("text_config.final_logit_softcapping", 0))) {
    TORCH_CHECK(weights.dim() == 2 && (!external || weights.size(0) == 255),
                "Invalid decision readout weight geometry");
    if (external)
      TORCH_CHECK(weights.scalar_type() == (d3 ? at::kFloat : at::kBFloat16),
                  "Decision readout precision does not match its checkpoint contract");
  }

  int64_t output_count(const DecisionRequest& request) const override {
    if (request.kind == DecisionRequest::Kind::Warmup)
      return 1;
    if (request.kind == DecisionRequest::Kind::InlineOptions) {
      TORCH_CHECK(external && !d3 && !request.markers.empty() &&
                      request.markers.size() == request.token_ids.size(),
                  "Invalid inline readout metadata");
      int64_t count = 0;
      for (size_t j = 0; j < request.markers.size(); ++j) {
        TORCH_CHECK(request.markers[j] >= 0 &&
                        (j == 0 || request.markers[j] > request.markers[j - 1]) &&
                        request.token_ids[j] > 0 && request.token_ids[j] <= weights.size(0),
                    "Invalid inline readout marker or option count");
        TORCH_CHECK(count <= INT64_MAX - request.token_ids[j], "Inline readout count overflow");
        count += request.token_ids[j];
      }
      return count;
    }
    TORCH_CHECK(request.kind == DecisionRequest::Kind::OptionTokens && !request.token_ids.empty(),
                "Expected nonempty option-token metadata");
    for (size_t j = 0; j < request.token_ids.size(); ++j)
      TORCH_CHECK(request.token_ids[j] >= 0 && request.token_ids[j] < weights.size(0) &&
                      (!external || request.token_ids[j] == int64_t(j)),
                  "Invalid option-token/readout-row metadata");
    return request.token_ids.size();
  }

  std::vector<DecisionResult> forward(const at::Tensor& hidden,
                                      const PackedInput& input,
                                      const std::vector<DecisionRequest>& requests) const override {
    validate(hidden, input, requests);
    TORCH_CHECK(hidden.size(1) == weights.size(1), "Decision hidden width mismatch");
    std::vector<DecisionResult> result;
    for (int64_t i = 0; i < input.batch; ++i) {
      const auto& request = requests[i];
      auto ids = request.token_ids;
      if (request.kind == DecisionRequest::Kind::InlineOptions) {
        output_count(request);
        TORCH_CHECK(request.markers.back() < input.offsets[i + 1] - input.offsets[i],
                    "Inline readout marker outside sequence");
        auto index =
            at::tensor(request.markers, hidden.options().dtype(at::kLong)) + input.offsets[i];
        auto logits =
            at::linear(hidden.index_select(0, index).to(at::kBFloat16), weights).to(at::kFloat);
        std::vector<at::Tensor> valid;
        for (size_t j = 0; j < request.markers.size(); ++j)
          valid.push_back(logits[j].narrow(0, 0, request.token_ids[j]));
        result.push_back({at::cat(valid, 0), at::ones({}, hidden.options().dtype(at::kFloat))});
        continue;
      }
      if (request.kind == DecisionRequest::Kind::Warmup)
        ids = {0};
      else
        TORCH_CHECK(request.kind == DecisionRequest::Kind::OptionTokens,
                    "Expected option-token decision metadata");
      TORCH_CHECK(!ids.empty(), "Option token metadata must not be empty");
      for (size_t j = 0; j < ids.size(); ++j)
        TORCH_CHECK(ids[j] >= 0 && ids[j] < weights.size(0) && (!external || ids[j] == int64_t(j)),
                    "Invalid option-token/readout-row metadata");
      auto last = hidden.select(0, input.offsets[i + 1] - 1).unsqueeze(0);
      auto index = at::tensor(ids, hidden.options().dtype(at::kLong));
      auto selected = weights.index_select(0, index);
      at::Tensor logits;
      if (d3)
        logits = at::linear(last.to(at::kFloat), weights).index_select(1, index);
      else if (external)
        logits = at::linear(last.to(at::kBFloat16), selected);
      else if (qwen)
        logits = at::linear(last.to(at::kFloat), selected.to(at::kFloat));
      else {
        logits = at::linear(last, selected);
        if (cap > 0)
          logits = at::tanh(logits / cap) * cap;
      }
      result.push_back(
          {logits.flatten().to(at::kFloat), at::ones({}, hidden.options().dtype(at::kFloat))});
    }
    return result;
  }
};

struct LayaLayer {
  Norm n1, n2;
  Attention attention;
  Linear ff1, ff2;

  LayaLayer(const Weights& w, const std::string& p, int64_t h)
      : n1(w, p + ".norm1", h), n2(w, p + ".norm2", h),
        attention(w, p + ".self_attn", h, std::max(int64_t(1), h / 64), true),
        ff1(w, p + ".linear1", h, 4 * h), ff2(w, p + ".linear2", 4 * h, h) {}

  at::Tensor operator()(const at::Tensor& x) const {
    auto n = n1(x);
    auto y = x + attention(n, n);
    return y + ff2(at::relu(ff1(n2(y))));
  }

  at::Tensor packed(const at::Tensor& x, const PackedInput& input) const {
    auto y = x + attention.packed_self(n1(x), input);
    return y + ff2(at::relu(ff1(n2(y))));
  }
};

struct LayaHead final : DecisionHead {
  at::Tensor types;
  std::vector<LayaLayer> layers;
  Norm scorer_norm;
  Linear scorer1, scorer2, action1, action2;
  int64_t width;

  LayaHead(const Options& cfg, const Weights& w, int64_t h)
      : types(weight(w, "type_emb.weight")), scorer_norm(w, "scorer.0", h),
        scorer1(w, "scorer.1", h, h), scorer2(w, "scorer.3", h, 1),
        action1(w, "act_head.0", h + 4, 256), action2(w, "act_head.2", 256, 2), width(h) {
    TORCH_CHECK(types.sizes() == at::IntArrayRef({3, h}) && cfg.integer("head_layers") >= 0 &&
                    cfg.integer("head_layers") <= 16 && cfg.integer("max_len") > 0 &&
                    cfg.integer("head_max_len") > 0,
                "Invalid Laya head configuration");
    for (int64_t i = 0; i < cfg.integer("head_layers"); ++i)
      layers.emplace_back(w, "head.layers." + std::to_string(i), h);
  }

  int64_t output_count(const DecisionRequest& request) const override {
    if (request.kind == DecisionRequest::Kind::Warmup)
      return 1;
    TORCH_CHECK(request.kind == DecisionRequest::Kind::Laya && request.question_type >= 0 &&
                    request.question_type < 3 && !request.markers.empty(),
                "Invalid Laya metadata");
    return request.markers.size();
  }

  struct Metadata {
    std::vector<int64_t> token_types, marker_rows, marker_offsets{0};
  };

  Metadata metadata(const PackedInput& input, const std::vector<DecisionRequest>& requests) const {
    TORCH_CHECK(input.batch > 0 && int64_t(requests.size()) == input.batch && input.offsets,
                "Invalid packed Laya metadata");
    Metadata value;
    value.token_types.resize(input.ids.numel());
    auto& token_types = value.token_types;
    auto& marker_rows = value.marker_rows;
    auto& marker_offsets = value.marker_offsets;
    for (int64_t i = 0; i < input.batch; ++i) {
      const auto& request = requests[i];
      const bool warmup = request.kind == DecisionRequest::Kind::Warmup;
      TORCH_CHECK(warmup || request.kind == DecisionRequest::Kind::Laya,
                  "Expected Laya marker metadata");
      auto type = warmup ? 0 : request.question_type;
      TORCH_CHECK(type >= 0 && type < 3 && (warmup || !request.markers.empty()),
                  "Invalid Laya question type or markers");
      auto start = input.offsets[i], end = input.offsets[i + 1];
      TORCH_CHECK(start >= 0 && end > start && end <= input.ids.numel(),
                  "Laya sequence outside packed hidden states");
      std::fill(token_types.begin() + start, token_types.begin() + end, type);
      if (warmup)
        marker_rows.push_back(start);
      else
        for (auto marker : request.markers) {
          TORCH_CHECK(marker >= 0 && marker < end - start, "Laya marker outside sequence");
          marker_rows.push_back(start + marker);
        }
      marker_offsets.push_back(marker_rows.size());
    }
    return value;
  }

  at::Tensor upload_metadata(const Metadata& value) const {
    auto combined = value.token_types;
    combined.insert(combined.end(), value.marker_rows.begin(), value.marker_rows.end());
    combined.insert(combined.end(), value.marker_offsets.begin(), value.marker_offsets.end());
    return at::tensor(combined, types.options().dtype(at::kLong));
  }

  at::Tensor prepare(const PackedInput& input,
                     const std::vector<DecisionRequest>& requests) const override {
    return types.is_cuda() ? upload_metadata(metadata(input, requests)) : at::Tensor();
  }

  std::vector<DecisionResult> forward_prepared(const at::Tensor& hidden,
                                               const PackedInput& input,
                                               const std::vector<DecisionRequest>& requests,
                                               const at::Tensor& prepared) const override {
    validate(hidden, input, requests);
    TORCH_CHECK(hidden.size(1) == width, "Laya hidden width mismatch");
    return hidden.is_cuda() ? forward_packed(hidden, input, requests, prepared)
                            : forward(hidden, input, requests);
  }

  std::vector<DecisionResult> forward_packed(const at::Tensor& hidden,
                                             const PackedInput& input,
                                             const std::vector<DecisionRequest>& requests,
                                             const at::Tensor& prepared = at::Tensor()) const {
    auto value = metadata(input, requests);
    const auto& marker_offsets = value.marker_offsets;
    auto indices = prepared.defined() ? prepared : upload_metadata(value);
    const int64_t tokens = value.token_types.size(), markers = value.marker_rows.size();
    TORCH_CHECK(indices.device() == hidden.device() && indices.scalar_type() == at::kLong &&
                    indices.is_contiguous() && indices.dim() == 1 &&
                    indices.numel() == tokens + markers + int64_t(marker_offsets.size()),
                "Invalid prepared Laya metadata");
    auto h = hidden + types.index_select(0, indices.narrow(0, 0, tokens));
    for (const auto& layer : layers)
      h = layer.packed(h, input);
    auto selected = h.index_select(0, indices.narrow(0, tokens, markers));
    auto all_logits =
        scorer2(at::gelu(scorer1(scorer_norm(selected)), "none")).flatten().to(at::kFloat);
#ifdef TEI_TORCH_CUDA_KERNELS
    auto features =
        laya_action_features(all_logits, indices.narrow(0, tokens + markers, marker_offsets.size()))
            .to(hidden.scalar_type());
    auto pooled = h.index_select(0, input.cumulative.narrow(0, 0, input.batch));
    auto actions =
        at::softmax(
            action2(at::gelu(action1(at::cat({pooled, features}, 1)), "none")).to(at::kFloat), -1)
            .select(1, 0);
    std::vector<DecisionResult> result;
    result.reserve(input.batch);
    for (int64_t i = 0; i < input.batch; ++i)
      result.push_back(
          {all_logits.narrow(0, marker_offsets[i], marker_offsets[i + 1] - marker_offsets[i]),
           actions.select(0, i)});
    return result;
#else
    std::vector<DecisionResult> result;
    result.reserve(input.batch);
    for (int64_t i = 0; i < input.batch; ++i) {
      auto count = marker_offsets[i + 1] - marker_offsets[i];
      auto logits = all_logits.narrow(0, marker_offsets[i], count);
      auto probabilities = at::softmax(logits, 0),
           sorted = std::get<0>(probabilities.sort(0, true));
      auto top1 = sorted.select(0, 0),
           top2 = count > 1 ? sorted.select(0, 1) : at::zeros_like(top1);
      auto entropy = -(probabilities * probabilities.clamp_min(1e-9).log()).sum() /
                     std::log(double(std::max(int64_t(2), count)));
      auto normalized_count =
          at::full({}, double(std::max(int64_t(2), count)) / 255., logits.options());
      auto features =
          at::stack({top1, top1 - top2, entropy, normalized_count}).to(hidden.scalar_type());
      auto action =
          at::softmax(
              action2(
                  at::gelu(action1(at::cat({h.select(0, input.offsets[i]), features}).unsqueeze(0)),
                           "none"))
                  .to(at::kFloat),
              -1)
              .select(0, 0)
              .select(0, 0);
      result.push_back({logits, action});
    }
    return result;
#endif
  }

  std::vector<DecisionResult> forward(const at::Tensor& hidden,
                                      const PackedInput& input,
                                      const std::vector<DecisionRequest>& requests) const override {
    validate(hidden, input, requests);
    TORCH_CHECK(hidden.size(1) == width, "Laya hidden width mismatch");
    std::vector<DecisionResult> result;
    if (hidden.is_cuda())
      return forward_packed(hidden, input, requests);
    for (int64_t i = 0; i < input.batch; ++i) {
      auto request = requests[i];
      if (request.kind == DecisionRequest::Kind::Warmup) {
        request.question_type = 0;
        request.markers = {0};
      } else
        TORCH_CHECK(request.kind == DecisionRequest::Kind::Laya, "Expected Laya marker metadata");
      auto length = input.offsets[i + 1] - input.offsets[i];
      TORCH_CHECK(request.question_type >= 0 && request.question_type < 3 &&
                      !request.markers.empty(),
                  "Invalid Laya question type or markers");
      for (auto marker : request.markers)
        TORCH_CHECK(marker >= 0 && marker < length, "Laya marker outside sequence");
      auto h = hidden.narrow(0, input.offsets[i], length) + types.select(0, request.question_type);
      for (const auto& layer : layers)
        h = layer(h);
      auto indices = at::tensor(request.markers, hidden.options().dtype(at::kLong));
      auto logits = scorer2(at::gelu(scorer1(scorer_norm(h.index_select(0, indices))), "none"))
                        .flatten()
                        .to(at::kFloat);
      auto probabilities = at::softmax(logits, 0),
           sorted = std::get<0>(probabilities.sort(0, true));
      auto top1 = sorted.select(0, 0),
           top2 = sorted.numel() > 1 ? sorted.select(0, 1) : at::zeros_like(top1);
      auto entropy = -(probabilities * probabilities.clamp_min(1e-9).log()).sum() /
                     std::log(double(std::max(size_t(2), request.markers.size())));
      auto count = at::full(
          {}, double(std::max(size_t(2), request.markers.size())) / 255., logits.options());
      auto features = at::stack({top1, top1 - top2, entropy, count}).to(hidden.scalar_type());
      auto action =
          at::softmax(
              action2(at::gelu(action1(at::cat({h.select(0, 0), features}).unsqueeze(0)), "none"))
                  .to(at::kFloat),
              -1)
              .select(0, 0)
              .select(0, 0);
      result.push_back({logits, action});
    }
    return result;
  }
};

struct Routing {
  Norm query, memory, feed;
  Attention attention;
  Linear ff1, ff2;

  Routing(const Weights& w, const std::string& p, int64_t width, int64_t heads, int64_t ff)
      : query(w, p + ".query_norm", width), memory(w, p + ".memory_norm", width),
        feed(w, p + ".feedforward_norm", width), attention(w, p + ".attention", width, heads),
        ff1(w, p + ".feedforward.0", width, ff), ff2(w, p + ".feedforward.3", ff, width) {}

  at::Tensor operator()(const at::Tensor& q, const at::Tensor& m) const {
    auto x = q + attention(query(q), memory(m));
    return x + ff2(at::gelu(ff1(feed(x)), "none"));
  }
};

struct FieldDecoder {
  Norm n1, n2, n3;
  Attention self, cross;
  Linear ff1, ff2;

  FieldDecoder(const Weights& w, const std::string& p, int64_t width, int64_t heads, int64_t ff)
      : n1(w, p + ".norm1", width), n2(w, p + ".norm2", width), n3(w, p + ".norm3", width),
        self(w, p + ".self_attn", width, heads), cross(w, p + ".multihead_attn", width, heads),
        ff1(w, p + ".linear1", width, ff), ff2(w, p + ".linear2", ff, width) {}

  at::Tensor operator()(const at::Tensor& x, const at::Tensor& m) const {
    auto n = n1(x), y = x + self(n, n);
    y = y + cross(n2(y), m);
    return y + ff2(at::gelu(ff1(n3(y)), "none"));
  }
};

struct ClefHead final : DecisionHead {
  Norm hidden_norm, summary_norm, field_norm, option_norm;
  std::vector<Linear> projections;
  at::Tensor types, words;
  std::vector<Routing> routing;
  std::vector<FieldDecoder> decoder;
  Linear scorer1, scorer2;
  double prior_scale, joint_scale, gate;
  int64_t width;

  ClefHead(
      const Options& cfg, const Options& backbone, const Weights& w, const Weights& bw, int64_t h)
      : hidden_norm(w, "hidden_norm", h),
        summary_norm(w, "option_summary_norm", cfg.integer("width")),
        field_norm(w, "field_norm", cfg.integer("width")),
        option_norm(w, "option_norm", cfg.integer("width")),
        types(weight(w, "type_embedding.weight")), words(lexical(backbone, bw)),
        scorer1(w, "residual_scorer.0", 4 * cfg.integer("width"), cfg.integer("width")),
        scorer2(w, "residual_scorer.3", cfg.integer("width"), 1), width(cfg.integer("width")) {
    auto heads = cfg.integer("heads"), ff = cfg.integer("feedforward");
    TORCH_CHECK(cfg.integer("hidden_size") == h && width > 0 && heads > 0 && width % heads == 0 &&
                    ff > 0 && types.sizes() == at::IntArrayRef({3, width}),
                "Invalid Clef configuration");
    for (auto name : {"memory_projection",
                      "question_projection",
                      "option_question_projection",
                      "global_projection",
                      "option_context_projection",
                      "option_lexical_projection"})
      projections.emplace_back(w, name, h, width, false);
    for (int64_t i = 0; i < cfg.integer("routing_layers"); ++i)
      routing.emplace_back(w, "evidence_layers." + std::to_string(i), width, heads, ff);
    for (int64_t i = 0; i < cfg.integer("layers"); ++i)
      decoder.emplace_back(w, "layers." + std::to_string(i), width, heads, ff);
    auto scale = [&](const std::string& name) {
      return weight(w, name).clamp_max(std::log(100.)).exp().to(at::kFloat).item<float>();
    };
    prior_scale = scale("prior_logit_scale");
    joint_scale = scale("joint_logit_scale");
    gate = at::sigmoid(weight(w, "residual_gate")).to(at::kFloat).item<float>();
  }

  int64_t output_count(const DecisionRequest& request) const override {
    if (request.kind == DecisionRequest::Kind::Warmup)
      return 1;
    TORCH_CHECK(request.kind == DecisionRequest::Kind::Clef && !request.fields.empty(),
                "Expected Clef fields");
    int64_t total = 0;
    for (const auto& field : request.fields) {
      TORCH_CHECK(field.kind >= 0 && field.kind <= 2 && !field.options.empty(),
                  "Invalid Clef schema");
      total += field.options.size();
    }
    return total;
  }

  at::Tensor score(const at::Tensor& raw,
                   const at::Tensor& ids,
                   const std::vector<DecisionField>& fields) const {
    TORCH_CHECK(!fields.empty(), "Clef requires at least one field");
    auto hidden = hidden_norm(raw), global = hidden.select(0, hidden.size(0) - 1),
         memory = projections[0](hidden);
    std::vector<at::Tensor> questions;
    for (const auto& f : fields) {
      TORCH_CHECK(f.kind >= 0 && f.kind <= 2 && !f.options.empty(), "Invalid Clef schema");
      questions.push_back(span(hidden, f.question));
    }
    auto question = at::stack(questions);
    std::vector<at::Tensor> lexical_rows, queries;
    for (size_t i = 0; i < fields.size(); ++i) {
      std::vector<at::Tensor> contexts, lexical_options;
      for (auto interval : fields[i].options) {
        contexts.push_back(span(hidden, interval));
        auto [a, b] = interval;
        TORCH_CHECK(a >= 0 && a < b && b <= ids.numel(), "Invalid Clef option span");
        auto selected = words.index_select(0, ids.narrow(0, a, b - a));
        lexical_options.push_back(span(selected, {0, b - a}));
      }
      auto lexical = at::stack(lexical_options);
      lexical_rows.push_back(lexical);
      queries.push_back(projections[4](at::stack(contexts)) + projections[5](lexical) +
                        projections[2](question.select(0, i).unsqueeze(0)));
    }
    auto routed = at::cat(queries);
    for (const auto& layer : routing)
      routed = layer(routed, memory);
    auto base = projections[1](question);
    std::vector<at::Tensor> options, summaries;
    int64_t offset = 0;
    for (size_t i = 0; i < fields.size(); ++i) {
      auto option = routed.narrow(0, offset, fields[i].options.size());
      offset += fields[i].options.size();
      auto weights = at::softmax(at::matmul(option, base.select(0, i).unsqueeze(1)).squeeze(1) /
                                     std::sqrt(double(width)),
                                 0);
      summaries.push_back((option * weights.unsqueeze(1)).sum(0));
      options.push_back(option);
    }
    std::vector<int64_t> kinds;
    for (const auto& field : fields)
      kinds.push_back(field.kind);
    auto field = base + summary_norm(at::stack(summaries)) + projections[3](global.unsqueeze(0)) +
                 at::embedding(types, at::tensor(kinds, ids.options()));
    for (const auto& layer : decoder)
      field = layer(field, memory);
    field = field_norm(field);
    std::vector<at::Tensor> logits;
    for (size_t i = 0; i < fields.size(); ++i) {
      auto anchor = normalize(question.select(0, i) + global);
      auto prior =
          at::matmul(normalize(lexical_rows[i]), anchor.unsqueeze(1)).squeeze(1) * prior_scale;
      auto option = option_norm(options[i]), f = field.select(0, i).expand_as(option).contiguous();
      auto cosine = (normalize(f) * normalize(option)).sum(-1);
      auto features = at::cat({f, option, f * option, (f - option).abs()}, 1);
      auto residual = scorer2(at::gelu(scorer1(features), "none")).squeeze(1);
      logits.push_back(prior + (cosine * joint_scale + residual) * gate);
    }
    return at::cat(logits).to(at::kFloat);
  }

  std::vector<DecisionResult> forward(const at::Tensor& hidden,
                                      const PackedInput& input,
                                      const std::vector<DecisionRequest>& requests) const override {
    validate(hidden, input, requests);
    std::vector<DecisionResult> result;
    for (int64_t i = 0; i < input.batch; ++i) {
      auto fields = requests[i].fields;
      if (requests[i].kind == DecisionRequest::Kind::Warmup)
        fields = {{0, {0, 1}, {{0, 1}}}};
      else
        TORCH_CHECK(requests[i].kind == DecisionRequest::Kind::Clef, "Expected Clef span metadata");
      auto start = input.offsets[i], length = input.offsets[i + 1] - start;
      result.push_back(
          {score(hidden.narrow(0, start, length), input.ids.narrow(0, start, length), fields),
           at::ones({}, hidden.options().dtype(at::kFloat))});
    }
    return result;
  }
};
} // namespace

std::unique_ptr<DecisionHead> create_decision_head(const std::string& kind,
                                                   const Options& head,
                                                   const Options& backbone,
                                                   const Weights& weights,
                                                   const Weights& backbone_weights) {
  auto hidden = backbone.integer("hidden_size", backbone.integer("text_config.hidden_size", 0));
  TORCH_CHECK(hidden > 0, "Decision backbone hidden_size is missing");
  if (kind == "laya")
    return std::make_unique<LayaHead>(head, weights, hidden);
  if (kind == "clef")
    return std::make_unique<ClefHead>(head, backbone, weights, backbone_weights, hidden);
  if (kind == "d3") {
    TORCH_CHECK(head.string("format_id") == "d3-code-readout-v1" && head.string("prompt") == "d3",
                "Unknown D3 readout format");
    TORCH_CHECK(head.string("pooling") == "last" &&
                    head.string("attention_mode") == "noncausal_full_attention" &&
                    head.string("readout_dtype") == "float32",
                "D3 requires last-token pooling, noncausal full attention and FP32 readout");
    TORCH_CHECK(backbone.string("model_type") == "qwen3_5" ||
                    backbone.string("model_type") == "qwen3_5_text",
                "D3 requires the dense Qwen3.5 backbone");
    return std::make_unique<OptionHead>(kind, backbone, weights, backbone_weights);
  }
  if (kind == "pplx") {
    TORCH_CHECK(head.string("pooling", "last") == "last", "Pplx requires last-token pooling");
    auto mode = head.string("attention_mode", "causal");
    TORCH_CHECK(mode == "causal" || mode == "noncausal_full_attention",
                "Unknown Pplx attention mode");
    return std::make_unique<OptionHead>(kind, backbone, weights, backbone_weights);
  }
  if (kind == "option_tokens")
    return std::make_unique<OptionHead>(kind, backbone, weights, backbone_weights);
  TORCH_CHECK(false, "Unsupported native decision head kind: ", kind);
}
}
