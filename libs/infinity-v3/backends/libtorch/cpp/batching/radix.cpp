// SPDX-License-Identifier: MIT
// Compact/unfold contract adapted from Candle's RadixMLP by Michael Feil.
// Copyright (c) 2025 michaelfeil
#include "batching/radix.h"
#ifdef TEI_TORCH_CUDA_KERNELS
#include "kernels/radix_qkv.h"
#endif
#include <vector>

namespace infinity {
// Full rows retain each request's sequence boundaries. Compact rows deduplicate
// tokens only when their attention context is identical. scatter_unfold maps
// full -> compact; fold_gather selects one full representative per compact row.
bool radix_enabled(const PackedInput& p) {
  return p.compact_ids.defined() || p.compact_positions.defined() || p.compact_types.defined() ||
         p.scatter_unfold.defined() || p.fold_gather.defined();
}

const at::Tensor& radix_ids(const PackedInput& p) {
  return radix_enabled(p) ? p.compact_ids : p.ids;
}

const at::Tensor& radix_positions(const PackedInput& p) {
  return radix_enabled(p) ? p.compact_positions : p.positions;
}

const at::Tensor& radix_types(const PackedInput& p) {
  return radix_enabled(p) ? p.compact_types : p.types;
}

at::Tensor radix_unfold(const at::Tensor& x, const PackedInput& p) {
  if (!radix_enabled(p))
    return x;
  TORCH_CHECK(x.dim() > 0 && x.size(0) == p.compact_ids.numel(),
              "Radix unfold requires compact token rows");
  TORCH_CHECK(p.scatter_unfold.defined() && p.scatter_unfold.scalar_type() == at::kLong &&
                  p.scatter_unfold.device() == x.device(),
              "Radix unfold requires matching on-device indices");
  return x.index_select(0, p.scatter_unfold);
}

// Attention still sees every actual token in each request. Only projections and
// token-local work use compact rows; the CUDA path gathers Q/K/V in one launch.
std::array<at::Tensor, 3> radix_unfold_qkv(const at::Tensor& q,
                                           const at::Tensor& k,
                                           const at::Tensor& v,
                                           const PackedInput& p) {
  if (!radix_enabled(p))
    return {q, k, v};
#ifdef TEI_TORCH_CUDA_KERNELS
  if (radix_qkv_cuda_supported(q, k, v, p.scatter_unfold)) {
    TORCH_CHECK(q.size(0) == p.compact_ids.numel(), "Radix QKV requires compact token rows");
    return radix_unfold_qkv_cuda(q, k, v, p.scatter_unfold);
  }
#endif
  return {radix_unfold(q, p), radix_unfold(k, p), radix_unfold(v, p)};
}

at::Tensor radix_fold(const at::Tensor& x, const PackedInput& p) {
  if (!radix_enabled(p))
    return x;
  TORCH_CHECK(x.dim() > 0 && x.size(0) == p.ids.numel(), "Radix gather requires full token rows");
  TORCH_CHECK(p.fold_gather.defined() && p.fold_gather.scalar_type() == at::kLong &&
                  p.fold_gather.device() == x.device(),
              "Radix gather requires matching on-device indices");
  return x.index_select(0, p.fold_gather);
}

// Validate host metadata before moving it to the inference device. Token
// equality alone cannot justify sharing: causal tokens must have the same past,
// and bidirectional tokens must belong to the same complete sequence.
/// Validate exact token/position/type and causal-parent equivalence using host metadata.
/// Bidirectional aliases must share complete sequences; the bridge
/// checks media equivalence separately.
void validate_radix(const PackedInput& p, bool causal) {
  if (!radix_enabled(p))
    return;
  auto host_vector = [](const at::Tensor& t, int64_t count, const char* name) {
    TORCH_CHECK(t.defined() && t.device().is_cpu() && t.scalar_type() == at::kLong &&
                    t.is_contiguous() && t.dim() == 1 && t.numel() == count,
                "Invalid host radix ",
                name);
  };
  TORCH_CHECK(p.ids.defined() && p.compact_ids.defined(),
              "Radix requires complete compact metadata");
  const auto n = p.ids.numel(), m = p.compact_ids.numel();
  TORCH_CHECK(n > 0 && m > 0 && m <= n && p.batch > 0 && p.offsets && p.offsets[0] == 0 &&
                  p.offsets[p.batch] == n,
              "Invalid radix token or sequence count");
  host_vector(p.ids, n, "full IDs");
  host_vector(p.positions, n, "full positions");
  host_vector(p.types, n, "full types");
  host_vector(p.compact_ids, m, "compact IDs");
  host_vector(p.compact_positions, m, "compact positions");
  host_vector(p.compact_types, m, "compact types");
  host_vector(p.scatter_unfold, n, "scatter");
  host_vector(p.fold_gather, m, "gather");
  const auto ids = p.ids.data_ptr<int64_t>(), pos = p.positions.data_ptr<int64_t>(),
             types = p.types.data_ptr<int64_t>();
  const auto ci = p.compact_ids.data_ptr<int64_t>(), cp = p.compact_positions.data_ptr<int64_t>(),
             ct = p.compact_types.data_ptr<int64_t>();
  const auto scatter = p.scatter_unfold.data_ptr<int64_t>(),
             gather = p.fold_gather.data_ptr<int64_t>();
  for (int64_t j = 0; j < m; ++j) {
    TORCH_CHECK(gather[j] >= 0 && gather[j] < n, "Radix representative index out of range");
    TORCH_CHECK(scatter[gather[j]] == j,
                "Radix compact rows must each represent an actual token (no padded rows)");
  }
  // -2 means unseen; -1 identifies a sequence root. A shared compact row must
  // have one parent, making prefix identity recursive across all aliased rows.
  // For bidirectional attention its terminal row must also match.
  std::vector<int64_t> parent(m, -2), terminal(m, -2);
  for (int64_t row = 0; row < p.batch; ++row) {
    const int64_t begin = p.offsets[row], end = p.offsets[row + 1];
    TORCH_CHECK(begin >= 0 && end > begin && end <= n, "Invalid radix sequence offsets");
    TORCH_CHECK(end - begin <= p.max_sequence, "Radix sequence exceeds maximum length");
    const auto last = scatter[end - 1];
    int64_t previous = -1;
    for (int64_t i = begin; i < end; ++i) {
      const auto j = scatter[i];
      TORCH_CHECK(j >= 0 && j < m, "Radix compact index out of range");
      TORCH_CHECK(ci[j] >= 0 && cp[j] >= 0 && ct[j] >= 0 && ids[i] == ci[j] && pos[i] == cp[j] &&
                      types[i] == ct[j],
                  "Radix aliases must preserve token IDs, positions and types");
      TORCH_CHECK(parent[j] == -2 || parent[j] == previous,
                  "Radix aliases require identical causal prefixes");
      parent[j] = previous;
      previous = j;
      TORCH_CHECK(causal || terminal[j] == -2 || terminal[j] == last,
                  "Bidirectional radix aliases require identical complete sequences");
      terminal[j] = last;
    }
  }
}
}
