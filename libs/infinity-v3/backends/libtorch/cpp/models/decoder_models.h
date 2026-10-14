#pragma once
#include <memory>
#include "models/model.h"
#include "helpers/options.h"
#include <functional>
namespace tei {
std::unique_ptr<Model> create_decoder(const Options&, Weights&, c10::Device, at::ScalarType);
// Expert weights use checkpoint layouts [E,2I,H] and [E,H,I].
at::Tensor decoder_moe_forward(const at::Tensor& x,const at::Tensor& router,
 const at::Tensor& gate_up,const at::Tensor& down,int64_t top_k,bool renormalize);
// Execute preselected routes without changing model-specific router semantics.
// Accumulation uses routing.dtype; the callback consumes concatenated gate/up.
at::Tensor decoder_moe_dispatch(const at::Tensor& x,const at::Tensor& gate_up,
 const at::Tensor& down,const at::Tensor& ids,const at::Tensor& routing,
 const std::function<at::Tensor(const at::Tensor&)>& gated_activation);
at::Tensor decoder_multimodal_forward(const Model&, const PackedInput&,
 const at::Tensor& visual_indices, const at::Tensor& visual_embeddings,
 const at::Tensor& cosine, const at::Tensor& sine,
 const std::vector<at::Tensor>& deepstack = {});
}
