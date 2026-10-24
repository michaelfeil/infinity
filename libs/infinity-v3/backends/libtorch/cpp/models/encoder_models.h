/// \file
/// Encoder model construction and qualified Nomic routing-policy query.

#pragma once
#include <memory>
#include "models/model.h"
#include "helpers/options.h"

namespace tei {
// Requires a loaded, qualified device routing policy and its current math settings.
/// Return false for other model classes or when current execution settings no longer match the
/// prepared Nomic routing policy. This is a capability query, not a request-time policy rebuild.
bool encoder_nomic_device_routing_ready(const Model&);
/// Select a supported encoder adapter or return null for another family. The loaded weight map is
/// borrowed and must outlive the model; call ready() to validate and prepare
/// parameters before inference.
std::unique_ptr<Model> create_encoder(const Options&, Weights&, c10::Device, at::ScalarType);
}
