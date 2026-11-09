// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
/// \file
/// Optional InfinityFA4 packed-attention and DeBERTa execution interfaces.

#pragma once
#include <ATen/ATen.h>
#include <optional>

namespace infinity {
/// Advertised attention-profile bits for this tensor's device and storage type.
int infinity_fa4_modes(const at::Tensor& reference);
/// Advertised paged-profile bits. Zero when the bundle predates paged support.
int infinity_fa4_paged_modes(const at::Tensor& reference);
/// Whether the loaded bundle exposes a paged profile for this Q geometry.
bool infinity_fa4_paged_supported(const at::Tensor& q, int64_t kv_heads);
// Undefined means this layout/mask has no exported kernel. A failed supported
// launch raises an error; it must never silently retry another implementation.
/// Consume [tokens, heads, head_dim] Q/K/V and device int32 cumulative Q/K boundaries. Unsupported
/// exports return undefined; errors after selecting a supported launch propagate.
/// An optional operand dtype defers conversion until a compatible export is selected.
at::Tensor infinity_fa4(const at::Tensor& q,
                        const at::Tensor& k,
                        const at::Tensor& v,
                        const at::Tensor& cu_q,
                        const at::Tensor& cu_k,
                        int64_t max_q,
                        double scale,
                        bool causal = false,
                        int64_t left = -1,
                        int64_t right = -1,
                        std::optional<at::ScalarType> operand_dtype = std::nullopt);
/// Consume [tokens, heads, head_dim] Q, [pages, page_size, kv_heads, head_dim] K/V,
/// int32 cumulative Q boundaries, per-sequence KV lengths, and an int32 block table.
/// K/V and the block table must be fully contiguous. Device-side value invariants
/// (seqused_k <= table capacity, cu_q[-1] <= total, page indices in range) are
/// trusted caller guarantees from prepare_block_pages; validating them here would
/// require a device sync. Unsupported exports return undefined; errors after
/// selecting a launch propagate.
at::Tensor infinity_fa4_paged(const at::Tensor& q,
                              const at::Tensor& k,
                              const at::Tensor& v,
                              const at::Tensor& cu_q,
                              const at::Tensor& seqused_k,
                              const at::Tensor& page_table,
                              int64_t max_q,
                              double scale);
bool infinity_fa4_deberta_supported(const at::Tensor& reference, int64_t head_dim = 64);
/// Evaluate packed disentangled attention using scaled keys,
/// relative projections, and signed buckets.
at::Tensor infinity_fa4_deberta(const at::Tensor& q,
                                const at::Tensor& scaled_k,
                                const at::Tensor& v,
                                const at::Tensor& cumulative,
                                const at::Tensor& c2p,
                                const at::Tensor& p2c,
                                const at::Tensor& buckets,
                                int64_t max_sequence);
}
