#pragma once
#include <ATen/ATen.h>
namespace tei {
// Original core warp reductions on actual tensor dimensions, without padding.
at::Tensor qwen35_vision_softmax(const at::Tensor& scores);
at::Tensor qwen35_vision_layer_norm(const at::Tensor& states,const at::Tensor& weight,const at::Tensor& bias,float epsilon);
}
