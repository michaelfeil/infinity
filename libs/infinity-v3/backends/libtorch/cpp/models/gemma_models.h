// Native adaptations of the Candle Gemma encoders. Upstream notices are retained
// in the repository LICENSE and NOTICE files.
#pragma once
#include <memory>
#include "models/model.h"
#include "helpers/options.h"

namespace tei {
// Audio convolutions preserve actual frame dimensions; linear bias rounds after GEMM.
/// Consume FP32 NCHW input and [out_channels,in_channels,kernel_h,kernel_w] weights on the same
/// device. The positive stride applies to both spatial axes; output is contiguous NCHW with
/// intrinsic filter padding.
at::Tensor gemma_audio_conv2d(const at::Tensor&, const at::Tensor&, int64_t);
/// Project the final feature dimension, rounding GEMM output to the input/weight dtype before an
/// optional separate bias addition.
at::Tensor gemma_audio_linear(const at::Tensor&, const at::Tensor&, const at::Tensor& bias = {});
/// Construct a supported Gemma adapter or return null. It borrows the loaded weight map for its
/// lifetime and requires ready() before inference.
std::unique_ptr<Model> create_gemma(const Options&, Weights&, c10::Device, at::ScalarType);
}
