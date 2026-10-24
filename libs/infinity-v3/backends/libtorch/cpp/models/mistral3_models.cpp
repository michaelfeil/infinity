// Pixtral/Mistral3 adaptation of Hugging Face Transformers 5.12.1 (Apache-2.0).
// See vendor/transformers_pixtral/NOTICE; fork modifications follow repository LICENSE.
#include "models/mistral3_models.h"
#include "models/decoder_models.h"
#include "helpers/attention.h"
#include "helpers/vision_batch.h"
#ifdef TEI_TORCH_CUDA_KERNELS
#include "kernels/fast_kernels.h"
#include "kernels/transformers_norm.h"
#endif
#include <ATen/ops/scaled_dot_product_attention.h>
#include <cmath>
#include <algorithm>

namespace tei {
namespace {
/// Pixtral vision composition with a native text decoder and optional classification head.
class Mistral3 final : public Model {
  Options cfg;
  Weights weights;
  c10::Device device;
  at::ScalarType dtype;
  std::unique_ptr<Model> text;
  at::Tensor score;
  bool classifier = false, fused_vision = false;
  int64_t hidden, heads, dim, depth, patch, merge, feature_layer;

  const at::Tensor& w(const std::string& name) const {
    return weight(weights, name);
  }

  /// Compute RMS statistics in FP32, round normalized rows back to model dtype, then multiply
  /// affine weights. The fused vision path retains those Transformers boundaries.
  at::Tensor rms(const at::Tensor& x, const at::Tensor& scale, double eps) const {
#ifdef TEI_TORCH_CUDA_KERNELS
    if (fused_vision)
      return transformers_rms_norm(x, {}, scale, eps).first;
#endif
    auto f = x.to(at::kFloat);
    return (f * at::rsqrt(f.square().mean(-1, true) + eps)).to(x.scalar_type()) * scale;
  }

  /// Apply a Pixtral tower/projector matrix and include bias only when that
  /// checkpoint tensor exists.
  at::Tensor linear(const at::Tensor& x, const std::string& name) const {
    auto it = weights.find(name + ".bias");
    return at::linear(x, w(name + ".weight"), it == weights.end() ? at::Tensor() : it->second);
  }

  /// Validate one frame's patch/merge geometry against the Pixtral
  /// position grid and return patch count.
  int64_t validate_image(const ImageInput& image) const {
    const auto h = image.grid_thw[1], width = image.grid_thw[2];
    TORCH_CHECK(image.grid_thw[0] == 1 && h > 0 && width > 0 && h % merge == 0 &&
                    width % merge == 0 && image.merge_size == merge,
                "Invalid Pixtral patch grid");
    const auto maximum = cfg.integer("vision_config.image_size", 1540) / patch;
    TORCH_CHECK(h <= maximum && width <= maximum, "Pixtral image exceeds its position grid");
    TORCH_CHECK(image.pixels.sizes() == at::IntArrayRef({h * width, 3 * patch * patch}),
                "Invalid Pixtral pixel matrix");
    return h * width;
  }

  /// Pack independent image patch sequences for shared vision layers, then spatially merge/project
  /// them. Memory-aware subbatches preserve original image order and
  /// never permit cross-image attention.
  at::Tensor vision(const std::vector<ImageInput>& images) const {
    auto batches = vision_batches(
        images, device, hidden, cfg.integer("vision_config.intermediate_size", hidden * 4));
    if (batches.size() > 1) {
      std::vector<at::Tensor> features;
      for (auto [start, end] : batches)
        features.push_back(
            vision(std::vector<ImageInput>(images.begin() + start, images.begin() + end)));
      return at::cat(features, 0);
    }
    std::vector<at::Tensor> pixels, cosines, sines;
    std::vector<int32_t> offsets{0};
    int64_t maximum = 0;
    const auto theta = cfg.real("vision_config.rope_parameters.rope_theta",
                                cfg.real("vision_config.rope_theta", 10000.));
    for (const auto& image : images) {
      const auto n = validate_image(image), width = image.grid_thw[2];
      TORCH_CHECK(n <= INT32_MAX - offsets.back(), "Packed Pixtral patch count exceeds int32");
      offsets.push_back(offsets.back() + n);
      maximum = std::max(maximum, n);
      pixels.push_back(image.pixels.to(device, dtype));
      auto index = at::arange(n, pixels.back().options().dtype(at::kLong));
      auto rows = at::floor_divide(index, width).to(at::kFloat).unsqueeze(1);
      auto cols = index.remainder(width).to(at::kFloat).unsqueeze(1);
      auto freq = at::arange(dim / 4, pixels.back().options().dtype(at::kFloat));
      auto angles = at::cat(
          {rows * at::pow(theta, -4 * freq / dim), cols * at::pow(theta, -(4 * freq + 2) / dim)},
          -1);
      cosines.push_back(angles.cos().to(dtype));
      sines.push_back(angles.sin().to(dtype));
    }
    // Cumulative patch counts isolate images, even within one text sequence.
    PackedInput boundaries{};
    boundaries.offsets = offsets.data();
    boundaries.batch = images.size();
    boundaries.max_sequence = maximum;
    boundaries.cumulative = at::from_blob(offsets.data(),
                                          {int64_t(offsets.size())},
                                          at::TensorOptions().dtype(at::kInt))
                                .to(device);
    const int64_t n = offsets.back();
    auto x = at::linear(at::cat(pixels), w("vision_tower.patch_conv.weight").reshape({hidden, -1}));
    pixels.clear();
    x = rms(x, w("vision_tower.ln_pre.weight"), 1e-5);
    auto cos = at::cat(cosines).unsqueeze(1), sin = at::cat(sines).unsqueeze(1);
    cosines.clear();
    sines.clear();
    auto full_cos = fused_vision ? at::Tensor() : at::cat({cos, cos}, -1);
    auto full_sin = fused_vision ? at::Tensor() : at::cat({sin, sin}, -1);
    auto rotate = [&](const at::Tensor& t) {
      return t * full_cos +
             at::cat({-t.narrow(-1, dim / 2, dim / 2), t.narrow(-1, 0, dim / 2)}, -1) * full_sin;
    };
    at::Tensor selected;
    // Feature-layer zero selects patch embeddings after the initial norm; positive values select
    // the output of that numbered transformer block.
    if (feature_layer == 0)
      selected = x;
    for (int64_t i = 0; i < depth; ++i) {
      auto p = "vision_tower.transformer.layers." + std::to_string(i) + ".";
      auto norm = rms(x, w(p + "attention_norm.weight"), 1e-5);
      at::Tensor q, k, v, attended;
#ifdef TEI_TORCH_CUDA_KERNELS
      if (fused_vision) {
        auto qkv = linear(norm, p + "attention.qkv").view({n, 3, heads, dim});
        auto pair = fused_rotary(qkv.select(1, 0), qkv.select(1, 1), cos, sin, false);
        attended =
            packed_attention(
                pair.first, pair.second, qkv.select(1, 2), boundaries, 1. / std::sqrt(double(dim)))
                .reshape({n, hidden});
      } else
#endif
      {
        q = rotate(linear(norm, p + "attention.q_proj").view({n, heads, dim}));
        k = rotate(linear(norm, p + "attention.k_proj").view({n, heads, dim}));
        v = linear(norm, p + "attention.v_proj").view({n, heads, dim});
        std::vector<at::Tensor> outputs;
        for (size_t row = 0; row < images.size(); ++row) {
          auto slice = [&](const at::Tensor& t) {
            return t.narrow(0, offsets[row], offsets[row + 1] - offsets[row])
                .transpose(0, 1)
                .unsqueeze(0);
          };
          outputs.push_back(at::scaled_dot_product_attention(slice(q), slice(k), slice(v))
                                .squeeze(0)
                                .transpose(0, 1));
        }
        attended = at::cat(outputs).reshape({n, hidden});
      }
      auto projected = linear(attended, p + "attention.o_proj");
#ifdef TEI_TORCH_CUDA_KERNELS
      if (fused_vision) {
        auto pair = transformers_rms_norm(projected, x, w(p + "ffn_norm.weight"), 1e-5);
        x = pair.second;
        norm = pair.first;
        auto activated = fused_gated_activation(
            linear(norm, p + "feed_forward.gate_up"),
            cfg.string("vision_config.hidden_act", "silu") == "silu" ? 0 : 1);
        x = x + linear(activated, p + "feed_forward.down_proj");
      } else
#endif
      {
        x = x + projected;
        norm = rms(x, w(p + "ffn_norm.weight"), 1e-5);
        auto gate = linear(norm, p + "feed_forward.gate_proj");
        auto act = cfg.string("vision_config.hidden_act", "silu") == "silu" ? at::silu(gate)
                                                                            : at::gelu(gate);
        x = x +
            linear(act * linear(norm, p + "feed_forward.up_proj"), p + "feed_forward.down_proj");
      }
      if (feature_layer == i + 1)
        selected = x;
    }
    TORCH_CHECK(selected.defined(), "Invalid Pixtral feature layer");
    // The projector consumes the configured hidden-state snapshot rather than necessarily the last
    // tower layer. Keep this choice before spatial patch merging.
    x = rms(selected,
            w("multi_modal_projector.norm.weight"),
            cfg.real("text_config.rms_norm_eps", 1e-5));
    // Match F.unfold's channel, patch-row, patch-column order (not flattened pixels).
    std::vector<at::Tensor> merged;
    for (size_t i = 0; i < images.size(); ++i) {
      const auto h = images[i].grid_thw[1], width = images[i].grid_thw[2];
      // Reshape each image independently, then order concatenated features as channel, patch row,
      // patch column to match the checkpoint's unfold-based projector weights.
      merged.push_back(x.narrow(0, offsets[i], offsets[i + 1] - offsets[i])
                           .view({h / merge, merge, width / merge, merge, hidden})
                           .permute({0, 2, 4, 1, 3})
                           .contiguous()
                           .view({-1, hidden * merge * merge}));
    }
    x = at::cat(merged);
    x = linear(x, "multi_modal_projector.patch_merger.merging_layer");
    x = linear(x, "multi_modal_projector.linear_1");
    auto act = cfg.string("projector_hidden_act", "gelu");
    TORCH_CHECK(act == "gelu" || act == "gelu_pytorch_tanh" || act == "silu",
                "Unsupported Mistral3 projector activation");
    x = act == "silu" ? at::silu(x) : at::gelu(x, act == "gelu_pytorch_tanh" ? "tanh" : "none");
    return linear(x, "multi_modal_projector.linear_2");
  }

public:
  Mistral3(const Options& c, Weights& source, c10::Device dev, at::ScalarType dt)
      : cfg(c), device(dev), dtype(dt) {
    // Consume the loader map: retaining it would keep unfused text parameters
    // alive alongside the decoder's packed matrices.
    Weights tw;
    classifier = c.string("architectures.0", "") == "Mistral3ForSequenceClassification";
    for (auto& [name, value] : source) {
      auto key = name.starts_with("model.") ? name.substr(6) : name;
      if (classifier && (key == "score.weight" || key == "language_model.score.weight"))
        score = std::move(value);
      else if (key.starts_with("language_model."))
        tw[key.substr(15)] = std::move(value);
      else if (key.starts_with("vision_tower.") || key.starts_with("multi_modal_projector."))
        weights[key] = std::move(value);
    }
    source.clear();
    Options tc;
    for (const auto& [k, v] : c.values)
      if (k.starts_with("text_config."))
        tc.values[k.substr(12)] = v;
    TORCH_CHECK(tc.string("model_type") == "ministral3" || tc.string("model_type") == "mistral",
                "Mistral3 requires a Mistral/Ministral3 text model");
    // This wrapper follows Transformers rounding boundaries, independently
    // of the legacy Candle-compatible Mistral kernel contracts.
    tc.values["_transformers_math"] = "true";
    text = create_decoder(tc, tw, device, dtype);
    hidden = cfg.integer("vision_config.hidden_size");
    heads = cfg.integer("vision_config.num_attention_heads");
    dim = heads > 0 ? hidden / heads : 0;
    depth = cfg.integer("vision_config.num_hidden_layers");
    patch = cfg.integer("vision_config.patch_size");
    merge = cfg.integer("spatial_merge_size", 2);
    feature_layer = cfg.integer("vision_feature_layer", -1);
    if (feature_layer < 0)
      feature_layer += depth + 1;
  }

  void ready() override {
    TORCH_CHECK(text && hidden > 0 && heads > 0 && hidden % heads == 0 && dim % 4 == 0 &&
                    depth > 0 && patch > 0 && merge > 0,
                "Invalid Mistral3 architecture");
    TORCH_CHECK(cfg.string("vision_config.model_type") == "pixtral",
                "Mistral3 requires a Pixtral vision encoder");
    TORCH_CHECK(cfg.string("vision_config.rope_parameters.rope_type", "default") == "default",
                "Unsupported Pixtral rotary configuration");
    TORCH_CHECK(cfg.integer("vision_config.num_channels", 3) == 3 && feature_layer >= 0 &&
                    feature_layer <= depth,
                "Unsupported Pixtral feature configuration");
    TORCH_CHECK(!cfg.values.count("vision_feature_layer.0"),
                "Multiple Pixtral feature layers are not supported");
    TORCH_CHECK(patch <= 64 && merge <= 8 && hidden <= 32768 && depth <= 256 &&
                    cfg.integer("vision_config.image_size", 1540) > 0 &&
                    cfg.integer("vision_config.image_size", 1540) <= 4096,
                "Unsupported Pixtral dimensions");
    const auto intermediate = cfg.integer("vision_config.intermediate_size"),
               out = text->output_width();
    const auto theta = cfg.real("vision_config.rope_parameters.rope_theta",
                                cfg.real("vision_config.rope_theta", 10000.));
    TORCH_CHECK(intermediate > 0 && std::isfinite(theta) && theta > 1,
                "Invalid Pixtral feed-forward/rotary dimensions");
    TORCH_CHECK(cfg.string("vision_config.hidden_act", "silu") == "silu" ||
                    cfg.string("vision_config.hidden_act") == "gelu",
                "Unsupported Pixtral activation");
    const auto act = cfg.string("projector_hidden_act", "gelu");
    TORCH_CHECK(act == "gelu" || act == "gelu_pytorch_tanh" || act == "silu",
                "Unsupported Mistral3 projector activation");
    auto require = [&](const std::string& name, std::initializer_list<int64_t> shape) {
      TORCH_CHECK(
          w(name).sizes() == at::IntArrayRef(shape), "Invalid Mistral3 weight dimensions: ", name);
    };
    require("vision_tower.patch_conv.weight", {hidden, 3, patch, patch});
    require("vision_tower.ln_pre.weight", {hidden});
    for (int64_t i = 0; i < depth; ++i) {
      auto p = "vision_tower.transformer.layers." + std::to_string(i) + ".";
      require(p + "attention_norm.weight", {hidden});
      require(p + "ffn_norm.weight", {hidden});
      for (const auto* proj : {"q_proj", "k_proj", "v_proj", "o_proj"})
        require(p + "attention." + proj + ".weight", {hidden, hidden});
      require(p + "feed_forward.gate_proj.weight", {intermediate, hidden});
      require(p + "feed_forward.up_proj.weight", {intermediate, hidden});
      require(p + "feed_forward.down_proj.weight", {hidden, intermediate});
    }
    require("multi_modal_projector.norm.weight", {hidden});
    require("multi_modal_projector.patch_merger.merging_layer.weight",
            {hidden, hidden * merge * merge});
    require("multi_modal_projector.linear_1.weight", {out, hidden});
    require("multi_modal_projector.linear_2.weight", {out, out});
    for (const auto* name :
         {"multi_modal_projector.linear_1.bias", "multi_modal_projector.linear_2.bias"}) {
      if (cfg.boolean("multimodal_projector_bias", true))
        require(name, {out});
      else
        TORCH_CHECK(!weights.count(name), "Unexpected projector bias");
    }
    if (classifier)
      TORCH_CHECK(score.defined() && score.dim() == 2 && score.size(0) > 0 && score.size(1) == out,
                  "Mistral3 sequence classifier requires score.weight of shape [labels, ",
                  out,
                  "]");
    text->ready();
#ifdef TEI_TORCH_CUDA_KERNELS
    fused_vision = device.is_cuda() && (dtype == at::kHalf || dtype == at::kBFloat16) &&
                   dim % 8 == 0 && dim <= 256;
    if (fused_vision)
      for (int64_t i = 0; i < depth; ++i) {
        const auto p = "vision_tower.transformer.layers." + std::to_string(i) + ".";
        auto pack = [&](const std::string& target, std::initializer_list<std::string> names) {
          std::vector<at::Tensor> matrices, biases;
          bool has_bias = false;
          for (const auto& name : names) {
            matrices.push_back(w(name + ".weight"));
            auto it = weights.find(name + ".bias");
            has_bias |= it != weights.end();
            biases.push_back(it == weights.end()
                                 ? at::zeros({matrices.back().size(0)}, matrices.back().options())
                                 : it->second);
          }
          weights[target + ".weight"] = at::cat(matrices);
          if (has_bias)
            weights[target + ".bias"] = at::cat(biases);
          for (const auto& name : names) {
            weights.erase(name + ".weight");
            weights.erase(name + ".bias");
          }
        };
        pack(p + "attention.qkv",
             {p + "attention.q_proj", p + "attention.k_proj", p + "attention.v_proj"});
        pack(p + "feed_forward.gate_up",
             {p + "feed_forward.gate_proj", p + "feed_forward.up_proj"});
      }
#endif
  }

  bool supports_images() const override {
    return true;
  }

  int64_t classification_width() const override {
    return score.defined() ? score.size(0) : 0;
  }

  // Matches the checkpoint's reference head: masked mean in the hidden-state
  // dtype, then an FP32 linear over all real tokens, including image markers.
  at::Tensor predict(const PackedInput& input, bool tokenwise) const override {
    TORCH_CHECK(!tokenwise, "Mistral3 sequence classifier does not implement token classification");
    TORCH_CHECK(score.defined(), "Mistral3 checkpoint has no sequence classification score head");
    auto states = forward(input);
    std::vector<at::Tensor> pooled;
    pooled.reserve(input.batch);
    for (int64_t row = 0; row < input.batch; ++row) {
      auto start = input.offsets[row], length = input.offsets[row + 1] - start;
      TORCH_CHECK(length > 0, "Sequence classification requires nonempty inputs");
      pooled.push_back(mean_pool(states.narrow(0, start, length)));
    }
    return at::linear(at::cat(pooled), score.to(states.device(), at::kFloat));
  }

  bool model_dtype_mean_pool() const override {
    return true;
  }

  at::Tensor mean_pool(const at::Tensor& value) const override {
    // Match a masked sum followed by division in the hidden-state dtype.
    // Packed sequences contain only real tokens, including image markers.
    return (value.sum(0, true) / value.size(0)).to(at::kFloat);
  }

  int64_t output_width() const override {
    return text->output_width();
  }

  /// Replace validated image token spans with merged Pixtral
  /// features before invoking the text decoder.
  at::Tensor forward(const PackedInput& input) const override {
    if (input.images.empty())
      return text->forward(input);
    std::vector<at::Tensor> indices;
    for (const auto& image : input.images) {
      validate_image(image);
      auto nh = image.grid_thw[1] / merge, nw = image.grid_thw[2] / merge;
      TORCH_CHECK(image.token_count == nh * nw && image.token_start >= image.sequence_start,
                  "Invalid Pixtral image token count");
      auto id = at::arange(nh * nw, input.ids.options());
      id = image.token_start + at::floor_divide(id, nw) * (nw + 1) + id.remainder(nw);
      TORCH_CHECK(image.sequence_start >= 0 && image.sequence_start < input.ids.numel(),
                  "Invalid Pixtral sequence start");
      int64_t end = -1;
      for (int64_t r = 0; r < input.batch; ++r)
        if (input.offsets[r] == image.sequence_start)
          end = input.offsets[r + 1];
      TORCH_CHECK(end >= 0 && image.token_start < end && nh * (nw + 1) <= end - image.token_start,
                  "Pixtral image tokens cross a sequence boundary");
      indices.push_back(id);
    }
    auto all_indices = at::cat(indices);
    TORCH_CHECK(
        at::all(input.ids.index_select(0, all_indices) == cfg.integer("image_token_index", 10))
            .item<bool>(),
        "Pixtral image placeholders do not match");
    return decoder_visual_forward(*text, input, all_indices, vision(input.images));
  }
};
}

std::unique_ptr<Model>
create_mistral3(const Options& cfg, Weights& weights, c10::Device device, at::ScalarType dtype) {
  if (cfg.string("model_type") != "mistral3")
    return nullptr;
  return std::make_unique<Mistral3>(cfg, weights, device, dtype);
}
}
