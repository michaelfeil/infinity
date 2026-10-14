// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <memory>
#include "models/model.h"
#include "helpers/options.h"
namespace tei {
struct DecisionField {
 int64_t kind;
 std::pair<int64_t,int64_t> question;
 std::vector<std::pair<int64_t,int64_t>> options;
};
struct DecisionRequest {
 enum class Kind { Laya, OptionTokens, Clef, Warmup } kind;
 int64_t question_type=0;
 std::vector<int64_t> markers,token_ids;
 std::vector<DecisionField> fields;
};
struct DecisionResult {
 at::Tensor logits;
 at::Tensor action_probability;
};
class DecisionHead {
public:
 virtual ~DecisionHead()=default;
 virtual int64_t output_count(const DecisionRequest&) const=0;
 virtual size_t transient_memory(const TeiDecision*,size_t,const int32_t*,size_t) const {
  TORCH_CHECK(false,"Unknown native typed decision transient tensor profile");
 }
 virtual std::vector<DecisionResult> forward(const at::Tensor& packed_hidden,
  const PackedInput&,const std::vector<DecisionRequest>&) const=0;
 // Request-owned metadata may be uploaded before the encoder starts, avoiding
 // blocking host uploads between dependent GPU stages. Other heads need none.
 virtual at::Tensor prepare(const PackedInput&,const std::vector<DecisionRequest>&) const {return {};}
 virtual std::vector<DecisionResult> forward_prepared(const at::Tensor& hidden,
  const PackedInput& input,const std::vector<DecisionRequest>& requests,const at::Tensor&) const {
  return forward(hidden,input,requests);
 }
};
// kind: laya, clef, pplx, option_tokens. Head config is the matching sidecar
// config, unprefixed. Weights are separate maps; tensors are cached, never moved.
// Laya head_weights: model.safetensors (encoder.* is ignored by the head).
// Clef head_weights: joint_head.safetensors; Pplx: readout.safetensors.
// Option tokens and Clef use lm_head.weight or tied embed_tokens.weight from
// backbone_weights, accepting the supported model/language_model prefixes.
std::unique_ptr<DecisionHead> create_decision_head(const std::string& kind,
 const Options& head_config,const Options& backbone_config,
 const Weights& head_weights,const Weights& backbone_weights);
}
