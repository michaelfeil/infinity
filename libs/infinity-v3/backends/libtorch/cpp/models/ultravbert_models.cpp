// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
// UltraVBERT composition follows the published Apache-2.0 Transformers model:
// SigLIP vision, pixel shuffle, normalized connector, bidirectional UltraBERT.
#include "models/ultravbert_models.h"
#include "models/decoder_models.h"
#include <ATen/ops/scaled_dot_product_attention.h>
#include <cmath>

namespace infinity {
namespace {
class UltraVbert final : public Model {
  Options cfg;
  Weights weights;
  std::unique_ptr<Model> text;
  c10::Device device;
  at::ScalarType dtype;

  const at::Tensor& w(const std::string& name) const {
    return weight(weights, name);
  }

  at::Tensor linear(const at::Tensor& x, const std::string& name) const {
    return at::linear(x, w(name + ".weight"), w(name + ".bias"));
  }

  at::Tensor norm(const at::Tensor& x, const std::string& name) const {
    return at::layer_norm(x,
                          {x.size(-1)},
                          w(name + ".weight"),
                          w(name + ".bias"),
                          cfg.real("vision_config.layer_norm_eps", 1e-6));
  }

  at::Tensor vision(const ImageInput& image) const {
    const auto tiles = image.grid_thw[0], rows = image.grid_thw[1], cols = image.grid_thw[2];
    TORCH_CHECK(tiles > 0 && tiles <= 5 && rows == 32 && cols == 32 && image.merge_size == 4 &&
                    image.pixels.sizes() == at::IntArrayRef({tiles * 1024, 768}),
                "UltraVBERT expects 512px tiles as RGB 16px patches");
    auto x = at::linear(image.pixels,
                        w("vision_model.embeddings.patch_embedding.weight").flatten(1),
                        w("vision_model.embeddings.patch_embedding.bias"));
    auto positions = w("vision_model.embeddings.position_embedding.weight");
    x = x.view({tiles, 1024, 768}) + positions.unsqueeze(0);
    for (int64_t i = 0; i < cfg.integer("vision_config.num_hidden_layers"); ++i) {
      auto p = "vision_model.encoder.layers." + std::to_string(i) + ".";
      auto h = norm(x, p + "layer_norm1");
      auto projection = [&](const char* name) {
        return linear(h, p + "self_attn." + name).view({tiles, 1024, 12, 64}).transpose(1, 2);
      };
      auto attended = at::scaled_dot_product_attention(
                          projection("q_proj"), projection("k_proj"), projection("v_proj"))
                          .transpose(1, 2)
                          .contiguous()
                          .view({tiles, 1024, 768});
      x = x + linear(attended, p + "self_attn.out_proj");
      x = x + linear(at::gelu(linear(norm(x, p + "layer_norm2"), p + "mlp.fc1"), "tanh"),
                     p + "mlp.fc2");
    }
    x = norm(x, "vision_model.post_layernorm");
    // ModernVBERT pixel shuffle first groups columns, then rows. Preserve the
    // ordering expected by the trained modality_projection matrix.
    x = x.view({tiles, 32, 32, 768})
            .view({tiles, 32, 8, 3072})
            .permute({0, 2, 1, 3})
            .reshape({tiles, 8, 8, 12288})
            .permute({0, 2, 1, 3})
            .reshape({tiles * 64, 12288});
    x = at::linear(x, w("connector.modality_projection.weight"));
    auto f = x.to(at::kFloat);
    return (f * at::rsqrt((f * f).mean(-1, true) + cfg.real("text_config.rms_norm_eps", 1e-6)))
               .to(dtype) *
           w("connector.norm.weight");
  }

public:
  UltraVbert(const Options& config, Weights& tensors, c10::Device dev, at::ScalarType type)
      : cfg(config), weights(std::move(tensors)), device(dev), dtype(type) {}

  void ready() override {
    TORCH_CHECK(cfg.integer("vision_config.hidden_size") == 768 &&
                    cfg.integer("vision_config.num_attention_heads") == 12 &&
                    cfg.integer("vision_config.image_size") == 512 &&
                    cfg.integer("vision_config.patch_size") == 16 &&
                    cfg.integer("vision_config.num_hidden_layers") == 12 &&
                    cfg.integer("pixel_shuffle_factor") == 4 &&
                    cfg.boolean("connector_norm", false),
                "Unsupported UltraVBERT vision geometry");
    auto check = [&](const std::string& name, at::IntArrayRef shape) {
      TORCH_CHECK(w(name).sizes() == shape, "Invalid UltraVBERT weight: ", name);
    };
    check("vision_model.embeddings.patch_embedding.weight", {768, 3, 16, 16});
    check("vision_model.embeddings.patch_embedding.bias", {768});
    check("vision_model.embeddings.position_embedding.weight", {1024, 768});
    check("connector.modality_projection.weight", {768, 12288});
    check("connector.norm.weight", {768});
    for (int64_t i = 0; i < 12; ++i) {
      auto p = "vision_model.encoder.layers." + std::to_string(i) + ".";
      for (const auto* name : {"layer_norm1", "layer_norm2"}) {
        check(p + name + ".weight", {768});
        check(p + name + ".bias", {768});
      }
      for (const auto* name : {"q_proj", "k_proj", "v_proj", "out_proj"}) {
        check(p + "self_attn." + name + ".weight", {768, 768});
        check(p + "self_attn." + name + ".bias", {768});
      }
      check(p + "mlp.fc1.weight", {3072, 768});
      check(p + "mlp.fc1.bias", {3072});
      check(p + "mlp.fc2.weight", {768, 3072});
      check(p + "mlp.fc2.bias", {768});
    }
    check("vision_model.post_layernorm.weight", {768});
    check("vision_model.post_layernorm.bias", {768});
    Options options;
    for (const auto& [key, value] : cfg.values) {
      if (key.starts_with("text_config."))
        options.values[key.substr(12)] = value;
      if (key.starts_with("_"))
        options.values[key] = value;
    }
    options.values["model_type"] = "ultrabert";
    options.values["architectures"] = "[\"UltrabertModel\"]";
    std::string theta = "[", windows = "[";
    for (int64_t i = 0; i < options.integer("num_hidden_layers"); ++i) {
      if (i) {
        theta += ",";
        windows += ",";
      }
      bool full = options.string("layer_types." + std::to_string(i)) == "full_attention";
      theta += options.string(std::string("rope_parameters.") +
                              (full ? "full_attention" : "sliding_attention") + ".rope_theta");
      windows += full ? "null" : options.string("sliding_window");
    }
    options.values["rope_theta"] = theta + "]";
    options.values["attention_windows"] = windows + "]";
    Weights encoder;
    for (auto it = weights.begin(); it != weights.end();) {
      if (it->first.starts_with("text_model.")) {
        encoder.emplace(it->first.substr(11), it->second);
        it = weights.erase(it);
      } else
        ++it;
    }
    text = create_decoder(options, encoder, device, dtype);
    TORCH_CHECK(text, "Missing UltraVBERT text encoder");
    text->ready();
    w("connector.modality_projection.weight");
    w("connector.norm.weight");
  }

  int64_t output_width() const override {
    return text->output_width();
  }

  bool supports_images() const override {
    return true;
  }

  at::Tensor forward(const PackedInput& input) const override {
    if (input.images.empty())
      return text->forward(input);
    std::vector<at::Tensor> indices, features;
    for (const auto& image : input.images) {
      auto value = vision(image);
      TORCH_CHECK(image.token_count == value.size(0), "UltraVBERT image token count mismatch");
      indices.push_back(at::arange(
          image.token_start, image.token_start + image.token_count, input.ids.options()));
      features.push_back(value);
    }
    return decoder_visual_forward(*text, input, at::cat(indices), at::cat(features));
  }
};
}

std::unique_ptr<Model>
create_ultravbert(const Options& cfg, Weights& weights, c10::Device device, at::ScalarType dtype) {
  if (cfg.string("model_type") != "ultravbert")
    return nullptr;
  return std::make_unique<UltraVbert>(cfg, weights, device, dtype);
}
}
