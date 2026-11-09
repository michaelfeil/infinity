// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
#pragma once
#include <ATen/ATen.h>
#include <vector>

namespace infinity {
// Whole prefix pages are shared. Each suffix owns the incomplete prefix tail
// and its own pages; unused cells are masked by the true KV sequence lengths.
struct BlockPages {
  static constexpr int64_t page_size = 256;
  // FA4 paged KV consumes 128-row pages; each 256-row page is two subpages of
  // the same gathered storage, so no copy is required.
  static constexpr int64_t fa4_page_size = 128;
  at::Tensor gather, table, subpage_table;
  int64_t max_k = 0;
};

BlockPages prepare_block_pages(const std::vector<int32_t>& prefixes,
                               const std::vector<int32_t>& suffixes,
                               const std::vector<int64_t>& sources,
                               const at::TensorOptions& options);
bool block_paged_supported(const at::Tensor& q);
at::Tensor block_paged_attention(const at::Tensor& q,
                                 const at::Tensor& k,
                                 const at::Tensor& v,
                                 const at::Tensor& prefix_k,
                                 const at::Tensor& prefix_v,
                                 const at::Tensor& cu_q,
                                 const at::Tensor& cu_k,
                                 const BlockPages& pages,
                                 int64_t max_q,
                                 double scale);
}
