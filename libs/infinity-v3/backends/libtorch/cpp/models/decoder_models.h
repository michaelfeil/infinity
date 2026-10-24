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
/// Construct a matching decoder without preparing it; call ready() before inference. A match moves
/// the weight map into the model. An unsupported model_type returns null without consuming the map.
std::unique_ptr<Model> create_decoder(const Options&, Weights&, c10::Device, at::ScalarType);
// Expert weights use checkpoint layouts [E,2I,H] and [E,H,I].
/// Compute router probabilities and select expert routes before dispatching [tokens, hidden] rows.
/// For N tokens, H hidden features, E experts, and I intermediate features: x is [N,H], router
/// [E,H], gate_up [E,2I,H], down [E,H,I], and output [N,H]. top_k selects experts per token;
/// renormalize rescales only their selected probabilities.
at::Tensor decoder_moe_forward(const at::Tensor& x,
                               const at::Tensor& router,
                               const at::Tensor& gate_up,
                               const at::Tensor& down,
                               int64_t top_k,
                               bool renormalize);
// Execute preselected routes without changing model-specific router semantics.
// Accumulation uses routing.dtype; the callback consumes concatenated gate/up.
/// ids is int64 [N,top_k] and routing has the same shape. The callback maps [routed_rows,2I]
/// gate/up activations to [routed_rows,I]. Output uses routing.dtype; caller-provided IDs
/// must reference valid experts.
at::Tensor
decoder_moe_dispatch(const at::Tensor& x,
                     const at::Tensor& gate_up,
                     const at::Tensor& down,
                     const at::Tensor& ids,
                     const at::Tensor& routing,
                     const std::function<at::Tensor(const at::Tensor&)>& gated_activation);
/// Run the decoder with visual row replacements, supplied rotary
/// phases, and optional deepstack features.
/// visual_indices is an on-device int64 vector of K packed row indices; visual_embeddings is
/// [K,hidden]. Each deepstack tensor uses that same layout. cosine/sine provide one half-head phase
/// vector per token on the decoder device/dtype.
at::Tensor decoder_multimodal_forward(const Model&,
                                      const PackedInput&,
                                      const at::Tensor& visual_indices,
                                      const at::Tensor& visual_embeddings,
                                      const at::Tensor& cosine,
                                      const at::Tensor& sine,
                                      const std::vector<at::Tensor>& deepstack = {});
}
