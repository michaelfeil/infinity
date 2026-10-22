#pragma once
#include <memory>
#include "models/model.h"
#include "helpers/options.h"
namespace tei {
// Native Candle Qwen3-VL adaptation; upstream notices retained in LICENSE/NOTICE.
class Qwen3VL final : public Model {
  Options cfg;
  Weights visual_weights;
  std::unique_ptr<Model> text;
  c10::Device device;
  at::ScalarType dtype;
  bool tower_only;
  bool exact_vision=false;
  at::Tensor temporal_first,temporal_second;
  int64_t hidden,heads,dim,merge,depth,text_dim;
  at::Tensor vision_frequency,text_frequency,text_axes;
  const at::Tensor& w(const std::string& name) const;
  at::Tensor linear(const at::Tensor& x,const std::string& name) const;
  at::Tensor ln(const at::Tensor& x,const std::string& name) const;
  at::Tensor merger(const at::Tensor& x,const std::string& p,bool post) const;
  std::pair<at::Tensor,std::vector<at::Tensor>> vision(const std::vector<ImageInput>& images) const;
 public:
  Qwen3VL(const Options& options,Weights& weights,c10::Device dev,at::ScalarType type,bool only_vision=false);
  void ready() override;
  bool supports_images() const override;
  bool supports_prefix_cache(int operation) const override;
  int64_t prefix_layers() const override;
  uint64_t prefix_payload_bytes(int64_t tokens) const override;
  std::vector<at::Tensor> image_features(const std::vector<ImageInput>& images) const;
  at::Tensor image_features(const ImageInput& image) const {return image_features(std::vector<ImageInput>{image}).front();}
  int64_t output_width() const override;
  at::Tensor forward(const PackedInput& input) const override;
};
std::unique_ptr<Model> create_qwen3_vl(const Options&, Weights&, c10::Device, at::ScalarType);
}
