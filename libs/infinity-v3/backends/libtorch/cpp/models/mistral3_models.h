// Copyright Infinity Core Contributors. See repository LICENSE and NOTICE.
#pragma once
#include "models/model.h"
#include "helpers/options.h"
namespace tei {
std::unique_ptr<Model> create_mistral3(const Options&, Weights&, c10::Device, at::ScalarType);
}
