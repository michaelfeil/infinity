/// \file
/// Native Qwen3.5 dense and MoE model construction interface.

#pragma once
#include <memory>
#include "models/model.h"
#include "helpers/options.h"

namespace tei {
/// Construct dense or MoE Qwen3.5 text/multimodal variants, moving the loaded weight map on a
/// match. Unsupported model types return null; call ready() before using the returned adapter.
std::unique_ptr<Model> create_qwen35(const Options&, Weights&, c10::Device, at::ScalarType);
}
