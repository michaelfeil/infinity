#include "prefix_cache_test.h"
// Native actual-length image/deepstack regression; no Python or token padding.
#include "models/qwen3_vl_models.h"
#include "helpers/vision_batch.h"
#include "helpers/fp8_linear.h"
#include "models/qwen35_models.h"
#include <c10/core/InferenceMode.h>
#include <ATen/Parallel.h>
#include <iostream>
#ifdef TEI_TORCH_CUDA_KERNELS
#include "kernels/qwen35_vision.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#endif
using namespace infinity;

#ifdef TEI_TORCH_CUDA_KERNELS
void vision_position_test(const c10::Device& device) {
  for (auto dtype : {at::kHalf, at::kBFloat16, at::kFloat})
    for (auto grid : {std::array<int64_t, 3>{4, 6, 2}, {6, 4, 2}, {1, 1, 1}}) {
      auto [height, width, merge] = grid;
      auto options = at::TensorOptions().device(device).dtype(dtype);
      auto table = at::randn({25, 24}, options);
      auto ix = at::arange(2 * height * width, options.dtype(at::kLong));
      auto c = ix.remainder(merge) +
               at::floor_divide(ix, merge * merge).remainder(width / merge) * merge;
      auto r = at::floor_divide(ix, merge).remainder(merge) +
               at::floor_divide(ix, merge * width).remainder(height / merge) * merge;
      auto rf = r.to(at::kFloat) * (height == 1 ? 0.f : 4.f / float(height - 1));
      auto cf = c.to(at::kFloat) * (width == 1 ? 0.f : 4.f / float(width - 1));
      auto r0 = rf.floor().to(at::kLong), r1 = rf.ceil().clamp_max(4).to(at::kLong);
      auto c0 = cf.floor().to(at::kLong), c1 = cf.ceil().clamp_max(4).to(at::kLong);
      auto dr = rf - r0.to(at::kFloat), dc = cf - c0.to(at::kFloat);
      auto corner = [&](const at::Tensor& rr, const at::Tensor& cc, const at::Tensor& scale) {
        return table.index_select(0, rr * 5 + cc) * scale.to(dtype).unsqueeze(-1);
      };
      auto expected = at::stack({corner(r0, c0, (1 - dr) * (1 - dc)),
                                 corner(r0, c1, (1 - dr) * dc),
                                 corner(r1, c0, dr * (1 - dc)),
                                 corner(r1, c1, dr * dc)});
      auto actual = vision_position_corners(table, 5, 2, height, width, merge);
      TORCH_CHECK(at::equal(actual.view(at::kByte), expected.view(at::kByte)),
                  "Vision interpolation changed corner order or model-dtype rounding");
    }
}

void vision_rotary_test(const c10::Device& device) {
  for (auto dtype : {at::kHalf, at::kBFloat16, at::kFloat})
    for (int64_t dim : {8, 64, 72, 128}) {
      auto options = at::TensorOptions().device(device).dtype(dtype);
      auto qkv = at::randn({7, 3, 3, dim}, options);
      auto phase = at::randn({7, 1, dim / 2}, options.dtype(at::kFloat));
      auto cosine = phase.cos(), sine = phase.sin();
      auto rotate = [&](const at::Tensor& x) {
        auto fp32 = x.to(at::kFloat);
        auto a = fp32.narrow(-1, 0, dim / 2), b = fp32.narrow(-1, dim / 2, dim / 2);
        return at::cat({a * cosine - b * sine, b * cosine + a * sine}, -1);
      };
      for (auto output_type : {dtype, at::kFloat}) {
        auto expected = at::stack({rotate(qkv.select(1, 0)).to(output_type),
                                   rotate(qkv.select(1, 1)).to(output_type),
                                   qkv.select(1, 2).to(output_type)});
        TORCH_CHECK(at::equal(vision_rotary_qkv(qkv, cosine, sine, output_type), expected),
                    "Fused vision rotary changed reference arithmetic or QKV layout");
      }
    }
}
#endif

void batched_tower_test(Qwen3VL& tower, const at::Tensor& pixels) {
  std::vector<ImageInput> images{{pixels.repeat({2, 1}), {1, 2, 4}, 2, 0, 2, 0},
                                 {pixels.repeat({3, 1}), {3, 2, 2}, 2, 0, 3, 0},
                                 {pixels * .2, {1, 2, 2}, 2, 0, 1, 0}};
  auto chunks = vision_batches(images, c10::Device("cpu"), 1LL << 30, 1LL << 30);
  TORCH_CHECK(chunks.size() == images.size() && chunks.front().first == 0 &&
                  chunks.back().second == images.size(),
              "Vision workspace bound lost images or joined oversized images");
  auto batch = tower.image_features(images);
  RequestImageFeatures cache({images[0], images[0]});
  auto retained = cache.get(images[0], [&] { return batch[0]; });
  TORCH_CHECK(retained.storage().nbytes() == retained.numel() * retained.element_size(),
              "Reused image retained the complete vision batch allocation");
  for (size_t i = 0; i < images.size(); ++i) {
    auto expected = tower.image_features(images[i]);
    TORCH_CHECK(at::allclose(batch[i], expected, .015, .015),
                "Batched Qwen image/frame attention leaked across grids");
  }
  std::reverse(images.begin(), images.end());
  auto reversed = tower.image_features(images);
  for (size_t i = 0; i < images.size(); ++i)
    TORCH_CHECK(at::allclose(reversed[i], batch[images.size() - 1 - i], .015, .015),
                "Qwen image descriptor order changed features");
}

void packed_images_test(Model& model, PackedInput input) {
  auto first = model.forward(input);
  auto second = input;
  second.images.clear();
  second.images.push_back({input.images[0].pixels.repeat({2, 1}), {1, 2, 4}, 2, 4, 2, 4});
  auto expected = model.forward(second);
  input.images.push_back(second.images[0]);
  auto mixed = model.forward(input);
  TORCH_CHECK(at::allclose(mixed.narrow(0, 0, 4), first.narrow(0, 0, 4), .015, .015) &&
                  at::allclose(mixed.narrow(0, 4, 2), expected.narrow(0, 4, 2), .015, .015),
              "Packed Qwen image/deepstack features leaked across sequences");
}

void rotary_presence_test(Model& model, PackedInput input) {
  input.images.clear();
  input.multimodal_positions = at::Tensor();
  input.host_multimodal_positions = at::Tensor();
  auto implicit = model.forward(input);
  auto explicit_zero = input;
  explicit_zero.multimodal_positions = at::zeros({3, input.ids.numel()}, input.ids.options());
  auto expected = model.forward(explicit_zero);
  TORCH_CHECK(!at::equal(implicit, expected),
              "Rotary presence control did not distinguish implicit positions from explicit zeros");
  PrefixCache cache(16 << 20, 32, 8);
  cache.run(model, input, 0);
  TORCH_CHECK(at::allclose(cache.run(model, explicit_zero, 0), expected, .003, .003) &&
                  cache.stats().hits == 0,
              "Explicit zero rotary axes reused implicit-position state");
  cache.clear();
  cache.run(model, explicit_zero, 0);
  TORCH_CHECK(at::allclose(cache.run(model, input, 0), implicit, .003, .003) &&
                  cache.stats().hits == 0,
              "Implicit rotary positions reused explicit-zero state");
}

// Exercise packed visual prefixes and suffixes against independently executed
// rows. Prefix representatives are request-local and may contain image spans.
void block_images_test(Options cfg, const Weights& checkpoint, const c10::Device& device) {
  cfg.values["_decision_attention_mode"] = "block_bidirectional";
  const auto dtype = device.is_cpu() ? at::kFloat : at::kBFloat16;
  auto weights = checkpoint;
  for (auto& [name, value] : weights)
    value = value.to(device, dtype);
  auto model = create_qwen35(cfg, weights, device, dtype);
  model->ready();
  const auto options = at::TensorOptions().device(device).dtype(at::kLong);
  std::vector<int32_t> offsets{0, 5, 9, 14, 18, 21};
  auto ids = at::tensor({1, 2, 3, 4, 5, 1, 2, 3, 7, 1, 2, 3, 8, 9, 1, 2, 3, 4, 1, 2, 3}, options);
  auto positions = at::cat({at::arange(5, options),
                            at::arange(4, options),
                            at::arange(5, options),
                            at::arange(4, options),
                            at::arange(3, options)});
  PackedInput input{ids,
                    at::zeros_like(ids),
                    positions,
                    at::tensor(offsets, options.dtype(at::kInt)),
                    offsets.data(),
                    5,
                    5};
  input.multimodal_positions = positions.unsqueeze(0).repeat({3, 1});
  input.block_prefix_lengths = {3, 3, 3, 3, 3};
  input.block_prefix_sources = {0, 0, 2, 3, 0};
  auto pixels = at::rand({4, 6}, options.dtype(dtype));
  for (int row : {0, 1, 2, 4})
    input.images.push_back({row == 2 ? at::zeros_like(pixels) : pixels,
                            {1, 2, 2},
                            2,
                            offsets[row] + 1,
                            1,
                            offsets[row]});
  auto actual = model->forward(input);
  auto separate = input;
  separate.block_prefix_sources = {0, 1, 2, 3, 4};
  const auto tolerance = device.is_cpu() ? 1e-5 : 1e-2;
  TORCH_CHECK(at::allclose(actual, model->forward(separate), tolerance, tolerance),
              "Image prefix sharing changed outputs");
  for (int row = 0; row < 5; ++row) {
    std::vector<int32_t> ends{0, offsets[row + 1] - offsets[row]};
    PackedInput single{ids.narrow(0, offsets[row], ends[1]),
                       input.types.narrow(0, offsets[row], ends[1]),
                       positions.narrow(0, offsets[row], ends[1]),
                       at::tensor(ends, options.dtype(at::kInt)),
                       ends.data(),
                       1,
                       ends[1]};
    single.multimodal_positions = input.multimodal_positions.narrow(1, offsets[row], ends[1]);
    single.block_prefix_lengths = {3};
    single.block_prefix_sources = {0};
    for (const auto& image : input.images)
      if (image.sequence_start == offsets[row]) {
        auto copy = image;
        copy.token_start -= offsets[row];
        copy.sequence_start = 0;
        single.images.push_back(copy);
      }
    TORCH_CHECK(
        at::allclose(
            actual.narrow(0, offsets[row], ends[1]), model->forward(single), tolerance, tolerance),
        "Packed image/text block rows leaked across sequences on ",
        device,
        " row=",
        row);
  }
  TORCH_CHECK(!at::allclose(actual.narrow(0, 0, 3), actual.narrow(0, 9, 3), 1e-5, 1e-6),
              "Distinct images did not change the prefix");
  auto changed = input;
  changed.ids = input.ids.clone();
  changed.ids.index_put_({4}, 12);
  auto future = model->forward(changed);
  TORCH_CHECK(at::allclose(actual.narrow(0, 0, 3), future.narrow(0, 0, 3), tolerance, tolerance),
              "Visual prefix attended to question suffix");
  auto suffix = input;
  suffix.block_prefix_lengths = {1, 1, 1, 1, 1};
  suffix.block_prefix_sources = {0, 0, 0, 0, 0};
  auto suffix_output = model->forward(suffix);
  suffix.images.clear();
  TORCH_CHECK(!at::allclose(suffix_output, model->forward(suffix), 1e-5, 1e-6),
              "Suffix image features were dropped");
  auto crossing = input;
  crossing.block_prefix_lengths = {1, 1, 1, 1, 1};
  crossing.block_prefix_sources = {0, 0, 0, 0, 0};
  crossing.images[0].token_start = 0;
  crossing.images[0].token_count = 2;
  bool rejected = false;
  try {
    model->forward(crossing);
  } catch (const c10::Error& e) {
    rejected = std::string(e.what()).find("crosses a block boundary") != std::string::npos;
  }
  TORCH_CHECK(rejected, "Image span crossing a block boundary was accepted");
  std::cout
      << "OneJev block images: shared/distinct prefixes, ragged mixed rows, suffix images and boundary rejection passed on "
      << device << "\n";
}

int main(int argc, char** argv) {
  c10::InferenceMode inference;
  at::set_num_threads(1);
  at::manual_seed(712);
  Options cfg;
  cfg.values = {{"model_type", "qwen3_vl"},
                {"text_config.hidden_size", "16"},
                {"text_config.num_attention_heads", "2"},
                {"text_config.num_key_value_heads", "1"},
                {"text_config.head_dim", "8"},
                {"text_config.num_hidden_layers", "2"},
                {"text_config.vocab_size", "32"},
                {"text_config.intermediate_size", "24"},
                {"text_config.max_position_embeddings", "32"},
                {"text_config.rope_theta", "10000"},
                {"text_config.rope_scaling.mrope_interleaved", "true"},
                {"text_config.rope_scaling.mrope_section.0", "2"},
                {"text_config.rope_scaling.mrope_section.1", "1"},
                {"text_config.rope_scaling.mrope_section.2", "1"},
                {"vision_config.hidden_size", "16"},
                {"vision_config.num_heads", "2"},
                {"vision_config.depth", "2"},
                {"vision_config.spatial_merge_size", "2"},
                {"vision_config.hidden_act", "gelu"},
                {"vision_config.deepstack_visual_indexes.0", "0"}};
  Weights weights;
  auto add = [&](const std::string& name, std::initializer_list<int64_t> shape, bool norm = false) {
    weights[name] = norm ? at::ones(shape) : at::randn(shape) * .1;
  };
  auto tp = std::string("model.language_model.");
  add(tp + "embed_tokens.weight", {32, 16});
  add(tp + "norm.weight", {16}, true);
  for (int i = 0; i < 2; ++i) {
    auto p = tp + "layers." + std::to_string(i) + ".";
    add(p + "input_layernorm.weight", {16}, true);
    add(p + "post_attention_layernorm.weight", {16}, true);
    add(p + "self_attn.q_proj.weight", {16, 16});
    add(p + "self_attn.k_proj.weight", {8, 16});
    add(p + "self_attn.v_proj.weight", {8, 16});
    add(p + "self_attn.o_proj.weight", {16, 16});
    add(p + "self_attn.q_norm.weight", {8}, true);
    add(p + "self_attn.k_norm.weight", {8}, true);
    add(p + "mlp.gate_proj.weight", {24, 16});
    add(p + "mlp.up_proj.weight", {24, 16});
    add(p + "mlp.down_proj.weight", {16, 24});
  }
  auto vp = std::string("model.visual.");
  auto ln = [&](const std::string& p, int64_t width) {
    add(vp + p + ".weight", {width}, true);
    weights[vp + p + ".bias"] = at::zeros({width});
  };
  auto fc = [&](const std::string& p, int64_t out, int64_t in) {
    add(vp + p + ".weight", {out, in});
    add(vp + p + ".bias", {out});
  };
  add(vp + "patch_embed.proj.weight", {16, 3, 2, 1, 1});
  add(vp + "patch_embed.proj.bias", {16});
  add(vp + "pos_embed.weight", {4, 16});
  for (int i = 0; i < 2; ++i) {
    auto p = "blocks." + std::to_string(i) + ".";
    ln(p + "norm1", 16);
    ln(p + "norm2", 16);
    fc(p + "attn.qkv", 48, 16);
    fc(p + "attn.proj", 16, 16);
    fc(p + "mlp.linear_fc1", 24, 16);
    fc(p + "mlp.linear_fc2", 16, 24);
  }
  for (const auto& p : {std::string("merger"), std::string("deepstack_merger_list.0")}) {
    ln(p + ".norm", p == "merger" ? 16 : 64);
    fc(p + ".linear_fc1", 64, 64);
    fc(p + ".linear_fc2", 16, 64);
  }
  auto model = create_qwen3_vl(cfg, weights, c10::Device("cpu"), at::kFloat);
  model->ready();
  batched_tower_test(*dynamic_cast<Qwen3VL*>(model.get()), at::rand({4, 6}));
  // Prepared numerical recipes must remain valid when the public tower object moves.
  auto move_cfg = cfg;
  move_cfg.values["_multi_vector"] = "true";
  move_cfg.values["vision_config.hidden_act"] = "silu";
  auto move_weights = weights;
  Qwen3VL original(move_cfg, move_weights, c10::Device("cpu"), at::kFloat, true);
  original.ready();
  ImageInput move_image{at::rand({4, 6}), {1, 2, 2}, 2, 0, 1, 0};
  auto before_move = original.image_features(move_image);
  Qwen3VL moved(std::move(original));
  TORCH_CHECK(at::equal(before_move, moved.image_features(move_image)),
              "Moving a prepared vision tower changed its MLP numerical recipe");
  TORCH_CHECK(model->supports_images(), "Qwen3-VL image capability unavailable");
  std::vector<int64_t> ids = {1, 2, 3, 4, 5, 6}, pos = {0, 1, 2, 3, 0, 1};
  std::vector<int32_t> offsets = {0, 4, 6};
  auto options = at::TensorOptions().dtype(at::kLong);
  PackedInput input{at::from_blob(ids.data(), {6}, options),
                    at::zeros({6}, options),
                    at::from_blob(pos.data(), {6}, options),
                    at::from_blob(offsets.data(), {3}, options.dtype(at::kInt)),
                    offsets.data(),
                    2,
                    4};
  auto plain = model->forward(input);
  rotary_presence_test(*model, input);
  input.images.push_back({at::rand({4, 6}), {1, 2, 2}, 2, 1, 1, 0});
  input.multimodal_positions = input.positions.unsqueeze(0).repeat({3, 1});
  auto visual = model->forward(input);
  packed_images_test(*model, input);
  check_prefix_cache(*model, input);
  PrefixCache media_cache(16 << 20, 32, 8);
  TORCH_CHECK(at::allclose(media_cache.run(*model, input, 0), visual, 1e-5, 2e-5),
              "Image cache fill changed outputs");
  TORCH_CHECK(at::allclose(media_cache.run(*model, input, 0), visual, 1e-5, 2e-5),
              "Image cache reuse changed outputs");
  TORCH_CHECK((visual - plain).abs().max().item<float>() > 1e-4, "Image features not injected");
  TORCH_CHECK(at::allclose(visual.narrow(0, 4, 2), plain.narrow(0, 4, 2), 1e-5, 1e-6),
              "Images leaked between packed sequences");
  TORCH_CHECK(at::allclose(visual.narrow(0, 0, 1), plain.narrow(0, 0, 1), 1e-5, 1e-6),
              "Image affected causal prefix");
  input.images[0].pixels = at::zeros({4, 6});
  auto changed = model->forward(input);
  TORCH_CHECK(at::allclose(media_cache.run(*model, input, 0), changed, 1e-5, 2e-5),
              "Different pixels reused an old image's prefix state");
  TORCH_CHECK((changed.narrow(0, 1, 3) - visual.narrow(0, 1, 3)).abs().max().item<float>() > 1e-4,
              "Image pixels ignored");
  TORCH_CHECK(at::allclose(changed.narrow(0, 4, 2), visual.narrow(0, 4, 2), 1e-5, 1e-6),
              "Changed image leaked between sequences");
  std::cout << "Qwen3-VL natural patches, image injection, deepstack and packed isolation passed\n";
  auto q35cfg = cfg;
  q35cfg.values["model_type"] = "qwen3_5";
  q35cfg.values.erase("vision_config.deepstack_visual_indexes.0");
  q35cfg.values["text_config.layer_types.0"] = "full_attention";
  q35cfg.values["text_config.layer_types.1"] = "full_attention";
  q35cfg.values["text_config.rope_parameters.mrope_interleaved"] = "true";
  q35cfg.values["text_config.rope_parameters.partial_rotary_factor"] = "1";
  for (int i = 0; i < 3; ++i)
    q35cfg.values["text_config.rope_parameters.mrope_section." + std::to_string(i)] =
        cfg.values.at("text_config.rope_scaling.mrope_section." + std::to_string(i));
  auto q35weights = weights;
  for (int i = 0; i < 2; ++i) {
    auto key = tp + "layers." + std::to_string(i) + ".self_attn.q_proj.weight";
    // Qwen3.5 interleaves each head's query with its attention output gate.
    auto query = q35weights.at(key).view({2, 8, 16});
    q35weights[key] = at::cat({query, at::zeros_like(query)}, 1).reshape({32, 16});
  }
  block_images_test(q35cfg, q35weights, c10::Device("cpu"));
  if (argc > 1)
    block_images_test(q35cfg, q35weights, c10::Device(argv[1]));
  auto q35loader = q35weights;
  auto q35 = create_qwen35(q35cfg, q35loader, c10::Device("cpu"), at::kFloat);
  q35->ready();
  auto q35plain = input;
  q35plain.images.clear();
  q35plain.multimodal_positions = at::Tensor();
  auto q35text = q35->forward(q35plain), q35image = q35->forward(input);
  TORCH_CHECK(q35->supports_images() && (q35image - q35text).abs().max().item<float>() > 1e-4,
              "Qwen3.5 vision features not injected");
  TORCH_CHECK(at::allclose(q35image.narrow(0, 4, 2), q35text.narrow(0, 4, 2), 1e-5, 1e-6),
              "Qwen3.5 vision leaked between packed sequences");
  Weights published;
  for (const auto& [name, value] : q35weights)
    published[name.starts_with("model.") ? name.substr(6) : name] = value;
  auto published_loader = published;
  auto unprefixed = create_qwen35(q35cfg, published_loader, c10::Device("cpu"), at::kFloat);
  unprefixed->ready();
  TORCH_CHECK(unprefixed->supports_images() && at::equal(q35text, unprefixed->forward(q35plain)) &&
                  at::equal(q35image, unprefixed->forward(input)),
              "Published Qwen3.5 visual/language prefix changed full text or image outputs");
  auto missing = published;
  missing.erase("visual.patch_embed.proj.weight");
  bool rejected = false;
  try {
    auto incomplete = create_qwen35(q35cfg, missing, c10::Device("cpu"), at::kFloat);
    incomplete->ready();
  } catch (const c10::Error& error) {
    rejected = std::string(error.what()).find("patch_embed.proj.weight") != std::string::npos;
  }
  TORCH_CHECK(rejected, "Incomplete published visual tower silently accepted");
  auto duplicate = published;
  duplicate["model.visual.patch_embed.proj.weight"] =
      published.at("visual.patch_embed.proj.weight");
  rejected = false;
  try {
    auto ambiguous = create_qwen35(q35cfg, duplicate, c10::Device("cpu"), at::kFloat);
    ambiguous->ready();
  } catch (const c10::Error& error) {
    rejected = std::string(error.what()).find("Duplicate Qwen visual weight") != std::string::npos;
  }
  TORCH_CHECK(rejected, "Duplicate canonical/published visual weight silently accepted");
  std::cout
      << "Qwen3.5 published prefixes preserve full outputs; missing/ambiguous tower rejected\n";
  std::cout << "Qwen3.5 FP16 vision, BF16-compatible injection and mRoPE packed isolation passed\n";
  if (argc > 1) {
    c10::Device device(argv[1]);
#ifdef TEI_TORCH_CUDA_KERNELS
    vision_position_test(device);
    vision_rotary_test(device);
#endif
    Weights hw, cw;
    for (const auto& [name, value] : weights) {
      hw[name] = value.to(at::kHalf);
      cw[name] = hw[name].to(device);
    }
    auto reference = create_qwen3_vl(cfg, hw, c10::Device("cpu"), at::kHalf);
    reference->ready();
    auto native = create_qwen3_vl(cfg, cw, device, at::kHalf);
    native->ready();
    auto host = input, dev = input;
    host.images[0].pixels = host.images[0].pixels.to(at::kHalf);
    dev.ids = host.ids.to(device);
    dev.types = host.types.to(device);
    dev.positions = host.positions.to(device);
    dev.cumulative = host.cumulative.to(device);
    dev.multimodal_positions = host.multimodal_positions.to(device);
    dev.images[0].pixels = host.images[0].pixels.to(device);
    rotary_presence_test(*native, dev);
    packed_images_test(*native, dev);
    auto expected = reference->forward(host).to(at::kFloat),
         actual = native->forward(dev).cpu().to(at::kFloat);
    if (fp8_device_supported(device)) {
      auto fp8_config = cfg;
      fp8_config.values["text_config.intermediate_size"] = "32";
      Weights checkpoint;
      for (const auto& [name, value] : weights)
        checkpoint[name] = value.to(device, at::kHalf);
      for (int i = 0; i < 2; ++i) {
        auto p = "model.language_model.layers." + std::to_string(i) + ".mlp.";
        for (const auto* name : {"gate_proj", "up_proj"})
          checkpoint[p + name + ".weight"] =
              at::randn({32, 16}, checkpoint.begin()->second.options()) * .05;
        checkpoint[p + "down_proj.weight"] =
            at::randn({16, 32}, checkpoint.begin()->second.options()) * .05;
      }
      auto reference_weights = checkpoint;
      auto baseline = create_qwen3_vl(fp8_config, reference_weights, device, at::kHalf);
      baseline->ready();
      fp8_config.values["_fp8_dynamic"] = "true";
      auto quantized = create_qwen3_vl(fp8_config, checkpoint, device, at::kHalf);
      quantized->ready();
      TORCH_CHECK(checkpoint.empty(), "Qwen3-VL retained redundant text checkpoint weights");
      auto expected_fp8 = baseline->forward(dev).to(at::kFloat);
      auto actual_fp8 = quantized->forward(dev).to(at::kFloat);
      TORCH_CHECK(at::isfinite(actual_fp8).all().item<bool>() &&
                      ((actual_fp8 - expected_fp8).norm() / expected_fp8.norm()).item<double>() <
                          .08,
                  "Qwen3-VL FP8 text MLPs changed image outputs beyond tolerance");
      check_prefix_cache(*quantized, dev, 0, .01, .02);
      std::cout << "Qwen3-VL FP8 image inference, released weights and prefix cache passed\n";
    }
    check_prefix_cache(*native, dev, 0, .003, .003);
    PrefixCache context_cache(1 << 20, 64, 8);
    context_cache.run(*native, dev, 0);
    auto plain = dev;
    plain.images.clear();
    auto plain_expected = native->forward(plain),
         plain_cached = context_cache.run(*native, plain, 0);
    TORCH_CHECK(context_cache.stats().hits == 0 &&
                    at::allclose(plain_expected, plain_cached, .003, .003),
                "Image-batch residual context leaked into plain-text prefix entries");
    TORCH_CHECK(at::allclose(expected, actual, .03, .03),
                "Qwen3-VL CUDA/CPU image parity mismatch: ",
                (expected - actual).abs().max().item<float>());
    std::cout << "Qwen3-VL CUDA/CPU float16 parity maxabs="
              << (expected - actual).abs().max().item<float>() << "\n";
#ifdef TEI_TORCH_CUDA_KERNELS
    // Actual 64x64 patch grid: two heads make each original F32 score
    // tensor 128 MiB, exercising production query chunking rather than the
    // unchanged small-image branch. These are synthetic functional weights,
    // not a replacement for the separate original learned-image oracle.
    auto bounded_cfg = q35cfg;
    bounded_cfg.values["text_config.max_position_embeddings"] = "2048";
    auto bounded_host_weights = q35weights;
    bounded_host_weights[vp + "patch_embed.proj.weight"] = at::randn({16, 3, 2, 16, 16}) * .002;
    Weights bounded_weights;
    for (const auto& [name, value] : bounded_host_weights)
      bounded_weights[name] = value.to(device, at::kBFloat16);
    auto bounded_loader = bounded_weights;
    auto bounded = create_qwen35(bounded_cfg, bounded_loader, device, at::kBFloat16);
    bounded->ready();
    constexpr int64_t patches = 4096, merged = 1024, total = 1027;
    std::vector<int32_t> large_offsets = {0, 1025, 1027};
    auto long_options = at::TensorOptions().device(device).dtype(at::kLong);
    auto positions = at::cat({at::arange(1025, long_options), at::arange(2, long_options)}, 0);
    PackedInput large{at::arange(total, long_options).remainder(30) + 1,
                      at::zeros({total}, long_options),
                      positions,
                      at::from_blob(large_offsets.data(), {3}, at::TensorOptions().dtype(at::kInt))
                          .clone()
                          .to(device),
                      large_offsets.data(),
                      2,
                      1025};
    large.multimodal_positions = positions.unsqueeze(0).repeat({3, 1});
    auto without_image = bounded->forward(large);
    large.images.push_back(
        {at::rand({patches, 1536}, long_options.dtype(at::kHalf)), {1, 64, 64}, 2, 1, merged, 0});
    Weights bounded_cpu_weights;
    for (const auto& [name, value] : bounded_host_weights)
      bounded_cpu_weights[name] = value.to(at::kBFloat16).to(at::kHalf);
    Qwen3VL cpu_vision(bounded_cfg, bounded_cpu_weights, c10::Device("cpu"), at::kHalf, true);
    cpu_vision.ready();
    auto cpu_image = large.images[0];
    cpu_image.pixels = cpu_image.pixels.cpu();
    auto original_cpu_features = cpu_vision.image_features(cpu_image).to(at::kFloat);
    // Initialize vision workspaces before recording live allocation peak.
    Qwen3VL vision_only(bounded_cfg, bounded_weights, device, at::kHalf, true);
    vision_only.ready();
    ImageInput warm_image{large.images[0].pixels.narrow(0, 0, 4), {1, 2, 2}, 2, 1, 1, 0};
    vision_only.image_features(warm_image);
    batched_tower_test(vision_only, warm_image.pixels);
    Weights bf16_vision_weights;
    for (const auto& [name, value] : bounded_weights)
      bf16_vision_weights[name] = value.to(at::kBFloat16);
    Qwen3VL bf16_tower(bounded_cfg, bf16_vision_weights, device, at::kBFloat16, true);
    bf16_tower.ready();
    batched_tower_test(bf16_tower, warm_image.pixels.to(at::kBFloat16));
    auto chunked_features = vision_only.image_features(large.images[0]).cpu().to(at::kFloat);
    TORCH_CHECK(at::allclose(original_cpu_features, chunked_features, .03, .03),
                "Large chunked Qwen3.5 vision differs from independent original CPU path: ",
                (original_cpu_features - chunked_features).abs().max().item<float>());
    at::cuda::getCurrentCUDAStream(device.index()).synchronize();
    const auto allocated_before =
        c10::cuda::CUDACachingAllocator::getDeviceStats(device.index()).allocated_bytes[0].current;
    c10::cuda::CUDACachingAllocator::resetPeakStats(device.index());
    auto large_image = bounded->forward(large);
    at::cuda::getCurrentCUDAStream(device.index()).synchronize();
    const auto peak_extra =
        c10::cuda::CUDACachingAllocator::getDeviceStats(device.index()).allocated_bytes[0].peak -
        allocated_before;
    TORCH_CHECK(peak_extra < 128LL * 1024 * 1024,
                "Large vision request recreated full square score/probability storage: ",
                peak_extra);
    TORCH_CHECK(at::isfinite(large_image).all().item<bool>(),
                "Chunked large-image output is nonfinite");
    TORCH_CHECK((large_image.narrow(0, 1, merged) - without_image.narrow(0, 1, merged))
                        .abs()
                        .max()
                        .item<float>() > 1e-5,
                "Large merged image span was ignored");
    TORCH_CHECK(at::equal(large_image.narrow(0, 1025, 2), without_image.narrow(0, 1025, 2)),
                "Large image leaked into another packed request");
    large.images[0].pixels = at::zeros_like(large.images[0].pixels);
    auto changed_image = bounded->forward(large);
    TORCH_CHECK(at::isfinite(changed_image).all().item<bool>() &&
                    (changed_image.narrow(0, 1, merged) - large_image.narrow(0, 1, merged))
                            .abs()
                            .max()
                            .item<float>() > 1e-5,
                "Large chunked image pixels ignored");
    TORCH_CHECK(at::equal(changed_image.narrow(0, 1025, 2), large_image.narrow(0, 1025, 2)),
                "Changed large image leaked between requests");
    std::cout
        << "Qwen3.5 actual 4096 patches / 1024 merged image tokens, chunked F32 attention, changed pixels and packed isolation passed; peak extra bytes="
        << peak_extra << "\n";
#endif
  }
}
