// Copyright Infinity Core Contributors. See repository LICENSE and NOTICE.
/// \file
/// Native Mistral3/Pixtral model construction interface.

#pragma once
#include "models/model.h"
#include "helpers/options.h"

namespace tei {
/// Construct Mistral3/Pixtral by consuming and clearing the loader weight map, splitting
/// tower/projector and text parameters. Other model types return null; call
/// ready() before inference.
std::unique_ptr<Model> create_mistral3(const Options&, Weights&, c10::Device, at::ScalarType);
}
