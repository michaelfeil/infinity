// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
#pragma once
#include "models/model.h"
#include "helpers/options.h"

namespace infinity {
std::unique_ptr<Model> create_ultravbert(const Options&, Weights&, c10::Device, at::ScalarType);
}
