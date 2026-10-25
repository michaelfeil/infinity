/// \file
/// Native Qwen3-VL vision encoding and multimodal text-backbone composition.
/// Builds spatial/temporal positions and injects image features into packed token rows.

#include "helpers/vision_batch.h"
#include "batching/prefix_cache.h"
#include "helpers/attention.h"
// Native adaptation of Candle Qwen3.5. Upstream notices retained in LICENSE/NOTICE.
#include "models/qwen3_vl_models.h"
#include "helpers/dense_mlp.h"
#include "models/decoder_models.h"
#include "kernels/qwen35_kernels.h"
#include "kernels/qwen35_reference.h"
#include "kernels/qwen35_vision.h"
#include "kernels/encoder_activation.h"
#include "kernels/torch_flash_fma.h"
#include "kernels/moe_kernels.h"
#include <ATen/ops/_efficient_attention_forward.h>
#include <ATen/ops/_flash_attention_forward.h>
#include <cmath>
#include <algorithm>
#include <map>

namespace infinity {
// Adapted from Candle qwen3_vl/vision.rs; upstream Apache-2.0 notices retained.
const at::Tensor& Qwen3VL::w(const std::string& name) const {
  return weight(visual_weights, "model.visual." + name);
}

/// Apply a vision projection with the profile's bias-rounding boundary. Exact vision rounds the
/// bias-free GEMM first, then adds bias separately.
at::Tensor Qwen3VL::linear(const at::Tensor& x, const std::string& name) const {
  // Original Linear rounds the model-dtype GEMM before adding its bias.
  if (exact_vision)
    return at::linear(x, w(name + ".weight")) + w(name + ".bias");
  return at::linear(x, w(name + ".weight"), w(name + ".bias"));
}

/// Normalize the final vision feature dimension with affine scale/bias and epsilon 1e-6; use the
/// qualified reference reduction for exact vision.
at::Tensor Qwen3VL::ln(const at::Tensor& x, const std::string& name) const {
#ifdef TEI_TORCH_CUDA_KERNELS
  if (exact_vision)
    return qwen35_vision_layer_norm(x.contiguous(), w(name + ".weight"), w(name + ".bias"), 1e-6f);
#endif
  return at::layer_norm(x, {x.size(-1)}, w(name + ".weight"), w(name + ".bias"), 1e-6);
}

/// Group merge-by-merge patches and project vision width to text width using
/// checkpoint-specific norm/GELU order.
at::Tensor Qwen3VL::merger(const at::Tensor& x, const std::string& p, bool post) const {
  // Preprocessing already places each spatial merge group in consecutive rows. Grouping therefore
  // changes only layout; the post flag selects normalization before or after
  // concatenating patch features.
  auto grouped = x.view({-1, hidden * merge * merge});
  auto normalized = post ? ln(grouped, p + ".norm") : ln(x, p + ".norm").view_as(grouped);
  // Retrieval checkpoints use Transformers' exact merger GELU; Candle uses tanh.
  return mlps.at(p).forward(normalized);
}

/// Batch patch projections and transformer layers while isolating individual frames in attention.
/// Return concatenated merged features and ordered deepstack
/// feature tensors for text-layer injection.
std::pair<at::Tensor, std::vector<at::Tensor>>
Qwen3VL::vision(const std::vector<ImageInput>& images) const {
  TORCH_CHECK(!images.empty(), "Empty Qwen3-VL image batch");
  auto batches = vision_batches(
      images, device, hidden, cfg.integer("vision_config.intermediate_size", hidden * 4));
  if (batches.size() > 1) {
    std::vector<at::Tensor> features;
    std::vector<std::vector<at::Tensor>> deep;
    for (auto [start, end] : batches) {
      auto [feat, ds] =
          vision(std::vector<ImageInput>(images.begin() + start, images.begin() + end));
      features.push_back(feat);
      if (deep.empty())
        deep.resize(ds.size());
      for (size_t i = 0; i < ds.size(); ++i)
        deep[i].push_back(ds[i]);
    }
    std::vector<at::Tensor> stacked;
    for (const auto& rows : deep)
      stacked.push_back(at::cat(rows, 0));
    return {at::cat(features, 0), std::move(stacked)};
  }
  const bool multi_vector = cfg.boolean("_multi_vector", false);
  std::vector<at::Tensor> patches, cosines, sines;
  std::vector<int32_t> offsets{0};
  int64_t max_frame = 0;
  for (const auto& image : images) {
    const auto [frames, height, width] = image.grid_thw;
    auto patch_dim = w("patch_embed.proj.weight").numel() / hidden;
    TORCH_CHECK(frames > 0 && height > 0 && width > 0 && height % merge == 0 &&
                    width % merge == 0 && image.merge_size == merge &&
                    image.pixels.sizes() == at::IntArrayRef({frames * height * width, patch_dim}),
                "Invalid Qwen3-VL packed image grid");
    // Each preprocessed row is one complete temporal/spatial patch; Conv3D's
    // nonoverlapping kernel is exactly this flattened projection.
    at::Tensor states;
    if (exact_vision) {
      // Candle temporal-2 projection: two actual K768 GEMMs, rounded FP16
      // sum, then separate FP16 bias. No temporal or token padding.
      auto patch = image.pixels.to(device, dtype).view({-1, 3, 2, 16, 16});
      auto first = at::linear(patch.select(2, 0).contiguous().view({-1, 768}), temporal_first);
      auto second = at::linear(patch.select(2, 1).contiguous().view({-1, 768}), temporal_second);
      states = (first + second) + w("patch_embed.proj.bias");
    } else if (multi_vector) {
      const auto& kernel = w("patch_embed.proj.weight");
      TORCH_CHECK(kernel.dim() == 5, "Invalid Qwen3-VL patch kernel");
      states = at::conv3d(
                   image.pixels.to(device, dtype)
                       .view({-1, kernel.size(1), kernel.size(2), kernel.size(3), kernel.size(4)}),
                   kernel,
                   w("patch_embed.proj.bias"),
                   {kernel.size(2), kernel.size(3), kernel.size(4)})
                   .view({-1, hidden});
    } else {
      states = at::linear(image.pixels.to(device, dtype),
                          w("patch_embed.proj.weight").reshape({hidden, patch_dim}),
                          w("patch_embed.proj.bias"));
    }
    auto side =
        static_cast<int64_t>(std::llround(std::sqrt(double(w("pos_embed.weight").size(0)))));
    TORCH_CHECK(side * side == w("pos_embed.weight").size(0),
                "Qwen3-VL position table must be square");
    auto ix = at::arange(frames * height * width, states.options().dtype(at::kLong));
    // Recover original grid coordinates from merge-group-major patch order before interpolating
    // learned positions and generating spatial rotary phases.
    auto c =
        ix.remainder(merge) + at::floor_divide(ix, merge * merge).remainder(width / merge) * merge;
    auto r = at::floor_divide(ix, merge).remainder(merge) +
             at::floor_divide(ix, merge * width).remainder(height / merge) * merge;
    auto rf = r.to(at::kFloat) * (height == 1 ? 0.f : float(side - 1) / float(height - 1));
    auto cf = c.to(at::kFloat) * (width == 1 ? 0.f : float(side - 1) / float(width - 1));
    auto r0 = rf.floor().to(at::kLong), r1 = rf.ceil().clamp_max(side - 1).to(at::kLong);
    auto c0 = cf.floor().to(at::kLong), c1 = cf.ceil().clamp_max(side - 1).to(at::kLong);
    auto dr = rf - r0.to(at::kFloat), dc = cf - c0.to(at::kFloat);
    auto table = w("pos_embed.weight");
    auto interpolate = [&](const at::Tensor& rr, const at::Tensor& cc, const at::Tensor& scale) {
      return table.index_select(0, rr * side + cc) * scale.to(dtype).unsqueeze(-1);
    };
    // Sum in table dtype, matching the four-corner interpolation stages.
    auto corners = at::stack({interpolate(r0, c0, (1 - dr) * (1 - dc)),
                              interpolate(r0, c1, (1 - dr) * dc),
                              interpolate(r1, c0, dr * (1 - dc)),
                              interpolate(r1, c1, dr * dc)},
                             0);
    auto positional = multi_vector
                          ? ((corners.select(0, 0) + corners.select(0, 1)) + corners.select(0, 2)) +
                                corners.select(0, 3)
                      : exact_vision ? (corners.select(0, 0) + corners.select(0, 2)) +
                                           (corners.select(0, 1) + corners.select(0, 3))
                                     : corners.sum(0);
    states = states + positional;

    auto angles = at::cat({r.to(at::kFloat).unsqueeze(1) * vision_frequency,
                           c.to(at::kFloat).unsqueeze(1) * vision_frequency},
                          -1);
    patches.push_back(states);
    cosines.push_back(angles.cos().unsqueeze(1));
    sines.push_back(angles.sin().unsqueeze(1));
    for (int64_t t = 0; t < frames; ++t) {
      TORCH_CHECK(int64_t(offsets.back()) + height * width <= INT32_MAX,
                  "Qwen3-VL image batch too large");
      offsets.push_back(offsets.back() + height * width);
    }
    max_frame = std::max(max_frame, height * width);
  }
  auto states = at::cat(patches, 0), cos = at::cat(cosines, 0), sin = at::cat(sines, 0);
  patches.clear();
  cosines.clear();
  sines.clear();
  auto cu = at::tensor(offsets, at::TensorOptions().dtype(at::kInt)).to(device);
  auto rotate = [&](const at::Tensor& value) {
    auto x = value.to(at::kFloat), a = x.narrow(-1, 0, dim / 2), b = x.narrow(-1, dim / 2, dim / 2);
    return at::cat({a * cos - b * sin, b * cos + a * sin}, -1);
  };
  std::vector<at::Tensor> deep;
  std::map<int64_t, std::vector<size_t>> frame_groups;
  if (multi_vector)
    for (size_t i = 0; i + 1 < offsets.size(); ++i)
      // Only equal-length frames share a dense SDPA batch in the retrieval path. Grouping preserves
      // the reference provider without padding shorter frames or merging their attention contexts.
      frame_groups[offsets[i + 1] - offsets[i]].push_back(i);
  for (int64_t layer = 0; layer < depth; ++layer) {
    auto p = "blocks." + std::to_string(layer) + ".";
    auto normalized = ln(states, p + "norm1");
    auto qkv = linear(normalized, p + "attn.qkv").view({states.size(0), 3, heads, dim});
    auto q = rotate(qkv.select(1, 0)), k = rotate(qkv.select(1, 1)),
         v = qkv.select(1, 2).to(at::kFloat);

    // Preserve the vision fallback's operand precision and frame grouping.
    auto aq = q, ak = k, av = v;
    if (device.is_cuda() && (multi_vector || dtype == at::kBFloat16)) {
      aq = q.to(dtype).contiguous();
      ak = k.to(dtype).contiguous();
      av = v.to(dtype).contiguous();
    }
    auto fallback = [&]() -> at::Tensor {
      if (multi_vector) {
        // Match SDPA's arithmetic while batching equally sized frames without padding.
        std::vector<at::Tensor> outputs(offsets.size() - 1);
        for (const auto& [length, frames] : frame_groups) {
          auto group = [&](const at::Tensor& x) {
            if (frames.size() == 1)
              return x.narrow(0, offsets[frames[0]], length).transpose(0, 1).unsqueeze(0).to(dtype);
            std::vector<at::Tensor> rows;
            for (auto frame : frames)
              rows.push_back(x.narrow(0, offsets[frame], length));
            return at::stack(rows, 0).transpose(1, 2).to(dtype);
          };
          auto result = at::scaled_dot_product_attention(group(aq),
                                                         group(ak),
                                                         group(av),
                                                         std::nullopt,
                                                         0.,
                                                         false,
                                                         1. / std::sqrt(double(dim)))
                            .transpose(1, 2);
          for (size_t i = 0; i < frames.size(); ++i)
            // Restore each frame to its original slot before concatenation, so merged image spans and
            // deepstack features retain the caller's ordering.
            outputs[frames[i]] = result.select(0, i);
        }
        return at::cat(outputs, 0);
      }
      if (device.is_cuda())
        return packed_aten_attention(dtype == at::kBFloat16 ? aq : q,
                                     dtype == at::kBFloat16 ? ak : k,
                                     dtype == at::kBFloat16 ? av : v,
                                     cu,
                                     cu,
                                     max_frame,
                                     max_frame,
                                     1. / std::sqrt(double(dim)),
                                     false);
      std::vector<at::Tensor> outputs;
      for (size_t t = 0; t + 1 < offsets.size(); ++t) {
        auto slice = [&](const at::Tensor& x) {
          return x.narrow(0, offsets[t], offsets[t + 1] - offsets[t]).transpose(0, 1).unsqueeze(0);
        };
        outputs.push_back(
            at::scaled_dot_product_attention(
                slice(q), slice(k), slice(v), std::nullopt, 0., false, 1. / std::sqrt(double(dim)))
                .squeeze(0)
                .transpose(0, 1));
      }
      return at::cat(outputs, 0);
    };
    auto attention = device.is_cuda() ? packed_cuda_attention(aq,
                                                              ak,
                                                              av,
                                                              cu,
                                                              cu,
                                                              max_frame,
                                                              max_frame,
                                                              1. / std::sqrt(double(dim)),
                                                              false,
                                                              -1,
                                                              -1,
                                                              false,
                                                              fallback,
                                                              dtype)
                                      : fallback();

    auto projected = linear(attention.reshape({states.size(0), hidden}).to(dtype), p + "attn.proj");
    states = states + projected;
    auto normalized2 = ln(states, p + "norm2");
    auto down = mlps.at(p + "mlp").forward(normalized2);
    states = states + down;
    for (int64_t i = 0;
         cfg.values.count("vision_config.deepstack_visual_indexes." + std::to_string(i));
         ++i)
      if (cfg.integer("vision_config.deepstack_visual_indexes." + std::to_string(i)) == layer)
        // Capture each configured intermediate tower layer through its own merger. Collect these
        // tensors in tower-layer execution order for early text-layer injection.
        deep.push_back(merger(states, "deepstack_merger_list." + std::to_string(i), true));
  }
  return {merger(states, "merger", false), std::move(deep)};
}

Qwen3VL::Qwen3VL(const Options& options,
                 Weights& weights,
                 c10::Device dev,
                 at::ScalarType type,
                 bool only_vision)
    : cfg(options), device(dev), dtype(type), tower_only(only_vision) {
  // Published full checkpoints use both model.visual.* and visual.*. Keep
  // one canonical local name and reject ambiguous towers instead of mixing
  // weights from two namespaces. Complete tower validation remains in ready.
  for (const auto& [name, value] : weights) {
    std::string canonical;
    if (name.starts_with("model.visual."))
      canonical = name;
    else if (name.starts_with("visual."))
      canonical = "model." + name;
    else
      continue;
    TORCH_CHECK(!visual_weights.count(canonical), "Duplicate Qwen visual weight: ", canonical);
    visual_weights.emplace(std::move(canonical), value.to(dtype));
  }
  Options tc = cfg;
  for (const auto& [name, value] : cfg.values)
    if (name.starts_with("text_config."))
      tc.values[name.substr(12)] = value;
  tc.values["model_type"] = "qwen3";
  tc.values["use_sliding_window"] = "false";
  text_dim =
      tc.integer("head_dim", tc.integer("hidden_size") / tc.integer("num_attention_heads", 1));
  hidden = cfg.integer("vision_config.hidden_size");
  heads = cfg.integer("vision_config.num_heads", 1);
  dim = hidden / heads;
  merge = cfg.integer("vision_config.spatial_merge_size", 2);
  depth = cfg.integer("vision_config.depth");
  if (tower_only)
    return;
  Weights text_weights;
  for (const auto& [name, value] : weights)
    if (name.starts_with("model.language_model."))
      text_weights[name.substr(21)] = value;
    else if (name.starts_with("language_model."))
      text_weights[name.substr(15)] = value;
  TORCH_CHECK(!text_weights.empty(), "Qwen3-VL language model weights missing");
  text = create_decoder(tc, text_weights, device, dtype);
}

/// Prepare vision frequencies and optional temporal projection slices, then validate interleaved
/// text mRoPE sections. Tower-only instances stop before text setup; full adapters
/// prepare their decoder last.
void Qwen3VL::ready() {
  TORCH_CHECK(hidden > 0 && heads > 0 && hidden % heads == 0 && dim % 4 == 0 && merge > 0 &&
                  depth > 0,
              "Invalid Qwen3-VL architecture");
  for (int64_t i = 0;
       cfg.values.count("vision_config.deepstack_visual_indexes." + std::to_string(i));
       ++i) {
    const auto layer = cfg.integer("vision_config.deepstack_visual_indexes." + std::to_string(i));
    TORCH_CHECK(layer >= 0 && layer < depth, "Invalid vision deepstack layer index");
  }
  const auto& patch_kernel = w("patch_embed.proj.weight");
  w("pos_embed.weight");
#ifdef TEI_TORCH_CUDA_KERNELS
  exact_vision =
      device.is_cuda() && dtype == at::kHalf && cfg.string("model_type").starts_with("qwen3_5") &&
      patch_kernel.dim() == 5 && patch_kernel.size(0) == hidden && patch_kernel.size(1) == 3 &&
      patch_kernel.size(2) == 2 && patch_kernel.size(3) == 16 && patch_kernel.size(4) == 16;
  if (exact_vision) {
    temporal_first = patch_kernel.select(2, 0).contiguous().view({hidden, 768});
    temporal_second = patch_kernel.select(2, 1).contiguous().view({hidden, 768});
  }
#endif
  std::vector<float> frequencies;
  for (int64_t j = 0; j < dim / 4; ++j)
    frequencies.push_back(1.f / std::pow(10000.f, float(2 * j) / float(dim / 2)));
  vision_frequency =
      at::from_blob(frequencies.data(), {dim / 4}, at::TensorOptions().dtype(at::kFloat))
          .clone()
          .to(device);
  auto projection = [&](const std::string& name) {
    ProjectionRecipe recipe;
    recipe.separate_bias = exact_vision;
    return MlpProjection{w(name + ".weight"), w(name + ".bias"), recipe, name};
  };
  auto bind_merger = [&](const std::string& p) {
    const bool exact = exact_vision, multi_vector = cfg.boolean("_multi_vector", false);
    auto activation = MlpActivationRecipe::numerical([exact, multi_vector](const at::Tensor& rows) {
#ifdef TEI_TORCH_CUDA_KERNELS
      if (exact)
        return encoder_gelu(rows);
#endif
      return at::gelu(rows, multi_vector ? "none" : "tanh");
    });
    mlps.insert_or_assign(p,
                          DenseMlp(projection(p + ".linear_fc1"),
                                   projection(p + ".linear_fc2"),
                                   std::move(activation)));
  };
  bind_merger("merger");
  for (int64_t i = 0;
       cfg.values.count("vision_config.deepstack_visual_indexes." + std::to_string(i));
       ++i)
    bind_merger("deepstack_merger_list." + std::to_string(i));
  for (int64_t i = 0; i < depth; ++i) {
    const auto p = "blocks." + std::to_string(i) + ".mlp";
    const auto act = cfg.string("vision_config.hidden_act", "gelu_pytorch_tanh");
    const bool exact = exact_vision;
    auto activation = MlpActivationRecipe::numerical([act, exact](const at::Tensor& rows) {
#ifdef TEI_TORCH_CUDA_KERNELS
      if (exact && act != "silu" && act != "gelu")
        return encoder_gelu(rows);
#endif
      return act == "silu" ? at::silu(rows) : at::gelu(rows, act == "gelu" ? "none" : "tanh");
    });
    mlps.insert_or_assign(p,
                          DenseMlp(projection(p + ".linear_fc1"),
                                   projection(p + ".linear_fc2"),
                                   std::move(activation)));
  }
  if (tower_only)
    return;
  TORCH_CHECK(text, "Qwen3-VL text decoder missing");
  TORCH_CHECK(cfg.boolean("text_config.rope_scaling.mrope_interleaved", false),
              "Only interleaved Qwen3-VL mRoPE is supported");
  int64_t sections = 0;
  for (int i = 0; i < 3; ++i)
    sections += cfg.integer("text_config.rope_scaling.mrope_section." + std::to_string(i));
  TORCH_CHECK(sections == text_dim / 2, "Invalid Qwen3-VL mRoPE sections");
  std::vector<float> freqs;
  std::vector<int64_t> axes;
  auto theta = cfg.real("text_config.rope_theta", 1000000.);
  auto s1 = cfg.integer("text_config.rope_scaling.mrope_section.1"),
       s2 = cfg.integer("text_config.rope_scaling.mrope_section.2");
  for (int64_t j = 0; j < text_dim / 2; ++j) {
    freqs.push_back(float(1. / std::pow(theta, double(2 * j) / text_dim)));
    // Assign each rotary frequency to one of the three supplied position axes, with section lengths
    // limiting the interleaved spatial axes and the remaining frequencies using axis zero.
    axes.push_back(j % 3 == 1 && j < s1 * 3 ? 1 : j % 3 == 2 && j < s2 * 3 ? 2 : 0);
  }
  text_frequency =
      at::from_blob(freqs.data(), {text_dim / 2}, at::TensorOptions().dtype(at::kFloat))
          .clone()
          .to(device);
  text_axes = at::from_blob(axes.data(), {text_dim / 2}, at::TensorOptions().dtype(at::kLong))
                  .clone()
                  .to(device);
  w("patch_embed.proj.weight");
  w("pos_embed.weight");
  text->ready();
}

bool Qwen3VL::supports_images() const {
  return true;
}

int64_t Qwen3VL::prefix_layers() const {
  return text ? text->prefix_layers() : 0;
}

uint64_t Qwen3VL::prefix_payload_bytes(int64_t tokens) const {
  return text ? text->prefix_payload_bytes(tokens) : 0;
}

bool Qwen3VL::supports_prefix_cache(int operation) const {
  return operation == 0 && text && text->supports_prefix_cache(0);
}

std::vector<at::Tensor> Qwen3VL::image_features(const std::vector<ImageInput>& images) const {
  if (images.empty())
    return {};
  auto features = vision(images).first;
  std::vector<at::Tensor> result;
  int64_t offset = 0;
  for (const auto& image : images) {
    auto count = image.grid_thw[0] * image.grid_thw[1] * image.grid_thw[2] / (merge * merge);
    result.push_back(features.narrow(0, offset, count));
    offset += count;
  }
  return result;
}

int64_t Qwen3VL::output_width() const {
  return text->output_width();
}

/// Build multimodal rotary phases and inject current vision/deepstack rows into the text decoder. A
/// cached visual prefix skips its tower work but retains the suffix's original deepstack stages.
at::Tensor Qwen3VL::forward(const PackedInput& input) const {
  auto positions = input.multimodal_positions.defined()
                       ? input.multimodal_positions
                       : input.positions.unsqueeze(0).expand({3, -1});
  auto angles =
      positions.index_select(0, text_axes).transpose(0, 1).to(at::kFloat) * text_frequency;
  std::vector<at::Tensor> indices, deep;
  at::Tensor features;
  for (const auto& image : input.images) {
    auto count = image.grid_thw[0] * image.grid_thw[1] * image.grid_thw[2] / (merge * merge);
    TORCH_CHECK(count == image.token_count, "Qwen3-VL merged visual token count mismatch");
    indices.push_back(
        at::arange(image.token_start, image.token_start + image.token_count, input.ids.options()));
  }
  if (!indices.empty())
    std::tie(features, deep) = vision(input.images);
  if (indices.empty() && input.prefix && !input.prefix->original.images.empty()) {
    // Cached visual rows need no feature computation, but the original
    // deepstack stages still consolidate residuals for the real text suffix.
    auto width = cfg.integer("text_config.hidden_size");
    auto empty = at::empty({0, width}, input.ids.options().dtype(dtype));
    indices.push_back(at::empty({0}, input.ids.options()));
    features = empty;
    for (int64_t i = 0;
         cfg.values.count("vision_config.deepstack_visual_indexes." + std::to_string(i));
         ++i) {
      auto layer = cfg.integer("vision_config.deepstack_visual_indexes." + std::to_string(i));
      if (layer >= 0 && layer < depth)
        deep.push_back(empty);
    }
  }
  return prefix_restore(
      input,
      decoder_multimodal_forward(*text,
                                 input,
                                 indices.empty() ? at::Tensor() : at::cat(indices, 0),
                                 features,
                                 angles.cos().to(dtype).contiguous(),
                                 angles.sin().to(dtype).contiguous(),
                                 deep));
}

std::unique_ptr<Model>
create_qwen3_vl(const Options& cfg, Weights& weights, c10::Device device, at::ScalarType dtype) {
  if (cfg.string("model_type") == "qwen3_vl")
    return std::make_unique<Qwen3VL>(cfg, weights, device, dtype);
  return nullptr;
}
}
