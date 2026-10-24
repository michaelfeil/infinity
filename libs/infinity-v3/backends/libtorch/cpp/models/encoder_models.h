/// \file
/// Encoder model construction and qualified Nomic routing-policy query.

#pragma once
#include <memory>
#include "models/model.h"
#include "helpers/options.h"

namespace tei {
// Requires a loaded, qualified device routing policy and its current math settings.
bool encoder_nomic_device_routing_ready(const Model&);
std::unique_ptr<Model> create_encoder(const Options&, Weights&, c10::Device, at::ScalarType);
}
