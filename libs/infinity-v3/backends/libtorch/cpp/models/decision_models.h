// SPDX-License-Identifier: Apache-2.0
/// \file
/// Native decision-head request/result contracts and head construction interface.
/// Heads consume packed backbone hidden states and protocol-specific request metadata.

#pragma once
#include <memory>
#include "models/model.h"
#include "helpers/options.h"

namespace tei {
/// Question and option token spans for one field of a structured decision.
struct DecisionField {
  int64_t kind;
  std::pair<int64_t, int64_t> question;
  std::vector<std::pair<int64_t, int64_t>> options;
};

/// Protocol selector and request-owned token/marker metadata for a decision head.
struct DecisionRequest {
  enum class Kind { Laya, OptionTokens, Clef, Warmup, InlineOptions = 5 } kind;
  int64_t question_type = 0;
  std::vector<int64_t> markers, token_ids;
  std::vector<DecisionField> fields;
};

/// Option-logit and action-probability tensors produced by a decision readout.
struct DecisionResult {
  at::Tensor logits;
  at::Tensor action_probability;
};

/// Common interface for native readouts over packed backbone hidden states.
class DecisionHead {
public:
  virtual ~DecisionHead() = default;
  virtual int64_t output_count(const DecisionRequest&) const = 0;
  virtual std::vector<DecisionResult> forward(const at::Tensor& packed_hidden,
                                              const PackedInput&,
                                              const std::vector<DecisionRequest>&) const = 0;

  // Request-owned metadata may be uploaded before the encoder starts, avoiding
  // blocking host uploads between dependent GPU stages. Other heads need none.
  virtual at::Tensor prepare(const PackedInput&, const std::vector<DecisionRequest>&) const {
    return {};
  }

  virtual std::vector<DecisionResult> forward_prepared(const at::Tensor& hidden,
                                                       const PackedInput& input,
                                                       const std::vector<DecisionRequest>& requests,
                                                       const at::Tensor&) const {
    return forward(hidden, input, requests);
  }
};

// kind: laya, clef, pplx, option_tokens. Head config is the matching sidecar
// config, unprefixed. Weights are separate maps; tensors are cached, never moved.
// Laya head_weights: model.safetensors (encoder.* is ignored by the head).
// Clef head_weights: joint_head.safetensors; Pplx: readout.safetensors.
// Option tokens and Clef use lm_head.weight or tied embed_tokens.weight from
// backbone_weights, accepting the supported model/language_model prefixes.
std::unique_ptr<DecisionHead> create_decision_head(const std::string& kind,
                                                   const Options& head_config,
                                                   const Options& backbone_config,
                                                   const Weights& head_weights,
                                                   const Weights& backbone_weights);
}
