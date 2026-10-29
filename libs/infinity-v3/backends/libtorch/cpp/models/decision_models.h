// SPDX-License-Identifier: Apache-2.0
/// \file
/// Native decision-head request/result contracts and head construction interface.
/// Heads consume packed backbone hidden states and protocol-specific request metadata.

#pragma once
#include <memory>
#include "models/model.h"
#include "helpers/options.h"

namespace infinity {
/// Question and option token spans for one field of a structured decision.
struct DecisionField {
  /// Head-specific field kind (validated in the range 0..2); fields and options
  /// retain caller schema order.
  int64_t kind;
  /// Question and option intervals are nonempty [begin,end) token spans relative to this request's
  /// sequence, before packed-batch offsets are added.
  std::pair<int64_t, int64_t> question;
  std::vector<std::pair<int64_t, int64_t>> options;
};

/// Protocol selector and request-owned token/marker metadata for a decision head.
struct DecisionRequest {
  enum class Kind { Laya, OptionTokens, Clef, Warmup, InlineOptions = 5 } kind;
  int64_t question_type = 0;
  /// Laya uses sequence-relative marker positions. OptionTokens uses token/readout row IDs.
  /// InlineOptions pairs each marker with an option count in token_ids, rather than vocabulary IDs.
  std::vector<int64_t> markers, token_ids;
  /// Clef's per-field question/option spans; other head kinds use markers or token IDs instead.
  std::vector<DecisionField> fields;
};

/// Option-logit and action-probability tensors produced by a decision readout.
struct DecisionResult {
  /// Flattened F32 option logits; output_count() gives the required element count. Clef preserves
  /// field/option order and inline readouts preserve marker/option order.
  at::Tensor logits;
  /// Scalar F32 action probability. Heads without a learned action classifier return one
  /// independently of the option logits.
  at::Tensor action_probability;
};

/// Common interface for native readouts over packed backbone hidden states.
class DecisionHead {
public:
  virtual ~DecisionHead() = default;
  /// Validate head-specific request metadata and report the number of option logits to reserve.
  virtual int64_t output_count(const DecisionRequest&) const = 0;

  /// Whether every request consumes only its final token. Warmup stays full-token.
  virtual bool last_token_only(const PackedInput&, const std::vector<DecisionRequest>&) const {
    return false;
  }

  virtual std::vector<DecisionResult>
  forward_pooled(const at::Tensor&, const PackedInput&, const std::vector<DecisionRequest>&) const {
    TORCH_CHECK(false, "Decision head requires complete token states");
  }

  /// Read packed [tokens, hidden] states and return one result per decision request. Token spans
  /// and marker positions refer to each request's sequence, not the whole packed batch.
  virtual std::vector<DecisionResult> forward(const at::Tensor& packed_hidden,
                                              const PackedInput&,
                                              const std::vector<DecisionRequest>&) const = 0;

  // Request-owned metadata may be uploaded before the encoder starts, avoiding
  // blocking host uploads between dependent GPU stages. Other heads need none.
  virtual at::Tensor prepare(const PackedInput&, const std::vector<DecisionRequest>&) const {
    return {};
  }

  /// Consume optional pre-uploaded head metadata; the default delegates to ordinary forward().
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
/// Validate the selected protocol/configuration and retain tensor handles from the supplied maps.
/// Unknown kinds or incompatible contracts throw; construction does not
/// take ownership of either map.
std::unique_ptr<DecisionHead> create_decision_head(const std::string& kind,
                                                   const Options& head_config,
                                                   const Options& backbone_config,
                                                   const Weights& head_weights,
                                                   const Weights& backbone_weights);
}
