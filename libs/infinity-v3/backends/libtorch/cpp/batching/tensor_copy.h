/// \file
/// Batched copies for graph inputs and retained prefix state.
/// Groups compatible tensor layouts and uses the specialized copy path when eligible.

#pragma once
#include <ATen/ops/_foreach_copy.h>
#include <array>
#include <map>
#include <vector>
#ifdef TEI_TORCH_CUDA_KERNELS
#include "kernels/prefix_state_copy.h"
#endif

namespace infinity {
// Foreach's fast copy route requires a homogeneous tensor list. Graph inputs
// mix integer metadata with model values, so group them before dispatch.
/// Enqueue paired copies on the current stream; callers retain buffers through completion. With
/// leading_rows, each destination receives only its own leading-row extent.
inline void tensor_copies(const std::vector<at::Tensor>& targets,
                          const std::vector<at::Tensor>& sources,
                          bool leading_rows = false) {
  TORCH_CHECK(targets.size() == sources.size(), "Invalid tensor copy batch");
#ifdef TEI_TORCH_CUDA_KERNELS
  // K/V views can have gaps between rows, which makes foreach copy fall back
  // to separate launches. The raw batched route also accepts mixed dtypes.
  if (prefix_state_copies(targets, sources, leading_rows))
    return;
#endif
  using Group = std::pair<std::vector<at::Tensor>, std::vector<at::Tensor>>;
  std::map<std::array<int, 3>, Group> groups;
  for (size_t i = 0; i < targets.size(); ++i) {
    const auto device = targets[i].device();
    auto& group = groups[{int(device.type()), int(device.index()), int(targets[i].scalar_type())}];
    group.first.push_back(targets[i]);
    if (leading_rows && sources[i].sizes() != targets[i].sizes()) {
      TORCH_CHECK(sources[i].dim() > 0 && targets[i].dim() == sources[i].dim() &&
                      targets[i].size(0) <= sources[i].size(0) &&
                      targets[i].sizes().slice(1) == sources[i].sizes().slice(1),
                  "Invalid leading-row tensor copy");
      group.second.push_back(sources[i].narrow(0, 0, targets[i].size(0)));
    } else
      group.second.push_back(sources[i]);
  }
  for (auto& [key, group] : groups)
    at::_foreach_copy_(group.first, group.second, true);
}
}
