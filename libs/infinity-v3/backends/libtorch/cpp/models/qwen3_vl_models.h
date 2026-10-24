/// \file
/// Qwen3-VL model interface for vision features and packed multimodal execution.

#pragma once
#include <memory>
#include "models/model.h"
#include "helpers/dense_mlp.h"
#include <unordered_map>
#include "helpers/options.h"

namespace tei {
// Native Candle Qwen3-VL adaptation; upstream notices retained in LICENSE/NOTICE.
class Qwen3VL final : public Model {
  Options cfg;
  Weights visual_weights;
  std::unordered_map<std::string, DenseMlp> mlps;
  std::unique_ptr<Model> text;
  c10::Device device;
  at::ScalarType dtype;
  bool tower_only;
  bool exact_vision = false;
  at::Tensor temporal_first, temporal_second;
  int64_t hidden, heads, dim, merge, depth, text_dim;
  at::Tensor vision_frequency, text_frequency, text_axes;
  const at::Tensor& w(const std::string& name) const;
  at::Tensor linear(const at::Tensor& x, const std::string& name) const;
  at::Tensor ln(const at::Tensor& x, const std::string& name) const;
  at::Tensor merger(const at::Tensor& x, const std::string& p, bool post) const;
  /// Return concatenated merged image rows plus intermediate deepstack tensors in tower execution
  /// order. Frame boundaries isolate attention; merge groups reduce each image's
  /// patch count by merge squared.
  std::pair<at::Tensor, std::vector<at::Tensor>>
  vision(const std::vector<ImageInput>& images) const;

public:
  /// Retain vision and optional text parameter maps without consuming the caller map. only_vision
  /// builds a tower for image_features(); it omits the text decoder required by
  /// forward() and output_width().
  Qwen3VL(const Options& options,
          Weights& weights,
          c10::Device dev,
          at::ScalarType type,
          bool only_vision = false);
  /// Validate tower geometry and deepstack indices, prepare spatial/temporal frequency tables, and
  /// prepare the text decoder unless this instance is tower-only.
  void ready() override;
  bool supports_images() const override;
  bool supports_prefix_cache(int operation) const override;
  int64_t prefix_layers() const override;
  uint64_t prefix_payload_bytes(int64_t tokens) const override;
  /// Return one merged vision feature matrix per image, preserving
  /// image order and merged token counts.
  std::vector<at::Tensor> image_features(const std::vector<ImageInput>& images) const;

  /// Return the single image's merged feature view. Its tensor handle keeps backing storage alive
  /// independently of the temporary batch result.
  at::Tensor image_features(const ImageInput& image) const {
    return image_features(std::vector<ImageInput>{image}).front();
  }

  int64_t output_width() const override;
  /// Insert projected image/frame features at packed marker spans, then execute the text backbone.
  at::Tensor forward(const PackedInput& input) const override;
};

/// Construct the full vision/text adapter for qwen3_vl or return null. Parameter handles are
/// retained in model-owned maps; ready() remains the caller's responsibility.
std::unique_ptr<Model> create_qwen3_vl(const Options&, Weights&, c10::Device, at::ScalarType);
}
