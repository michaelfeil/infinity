/// \file
/// Decoder construction and shared multimodal/expert execution interfaces.

#pragma once
#include <memory>
#include "models/model.h"
#include "helpers/options.h"
#include <functional>

namespace tei {
/// Replace selected embedding rows with vision features before the decoder backbone runs.
at::Tensor
decoder_visual_forward(const Model&, const PackedInput&, const at::Tensor&, const at::Tensor&);
std::unique_ptr<Model> create_decoder(const Options&, Weights&, c10::Device, at::ScalarType);
// Expert weights use checkpoint layouts [E,2I,H] and [E,H,I].
/// Compute router probabilities and select expert routes before dispatching [tokens, hidden] rows.
at::Tensor decoder_moe_forward(const at::Tensor& x,
                               const at::Tensor& router,
                               const at::Tensor& gate_up,
                               const at::Tensor& down,
                               int64_t top_k,
                               bool renormalize);
// Execute preselected routes without changing model-specific router semantics.
// Accumulation uses routing.dtype; the callback consumes concatenated gate/up.
at::Tensor
decoder_moe_dispatch(const at::Tensor& x,
                     const at::Tensor& gate_up,
                     const at::Tensor& down,
                     const at::Tensor& ids,
                     const at::Tensor& routing,
                     const std::function<at::Tensor(const at::Tensor&)>& gated_activation);
/// Run the decoder with visual row replacements, supplied rotary
/// phases, and optional deepstack features.
at::Tensor decoder_multimodal_forward(const Model&,
                                      const PackedInput&,
                                      const at::Tensor& visual_indices,
                                      const at::Tensor& visual_embeddings,
                                      const at::Tensor& cosine,
                                      const at::Tensor& sine,
                                      const std::vector<at::Tensor>& deepstack = {});
}
