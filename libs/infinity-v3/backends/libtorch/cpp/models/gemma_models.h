// Native adaptations of the Candle Gemma encoders. Upstream notices are retained
// in the repository LICENSE and NOTICE files.
#pragma once
#include <memory>
#include "models/model.h"
#include "helpers/options.h"
namespace tei {
// Audio convolutions preserve actual frame dimensions; linear bias rounds after GEMM.
at::Tensor gemma_audio_conv2d(const at::Tensor&, const at::Tensor&, int64_t);
at::Tensor gemma_audio_linear(const at::Tensor&, const at::Tensor&, const at::Tensor& bias = {});
std::unique_ptr<Model> create_gemma(const Options&, Weights&, c10::Device, at::ScalarType);
}
