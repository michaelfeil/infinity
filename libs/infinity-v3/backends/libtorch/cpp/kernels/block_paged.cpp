// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
#include "block_paged.h"
#include <algorithm>
#include <ATen/ops/_flash_attention_forward.h>
#ifdef TEI_TORCH_CUDA_KERNELS
#include <ATen/cuda/CUDAContext.h>
#endif

namespace infinity {
BlockPages prepare_block_pages(const std::vector<int32_t>& prefixes,
                               const std::vector<int32_t>& suffixes,
                               const std::vector<int64_t>& sources,
                               const at::TensorOptions& options) {
  TORCH_CHECK(prefixes.size() > 1 && prefixes.front() == 0 &&
                  suffixes.size() == sources.size() + 1 && suffixes.front() == 0,
              "Invalid block page boundaries");
  constexpr auto page = BlockPages::page_size;
  std::vector<int64_t> gather, first_page(prefixes.size() - 1, -1);
  std::vector<std::vector<int32_t>> rows;
  int64_t max_k = 0, expanded = 0;
  size_t columns = 0;
  for (size_t row = 0; row < sources.size(); ++row) {
    const auto source = sources[row];
    TORCH_CHECK(source >= 0 && size_t(source + 1) < prefixes.size() &&
                    prefixes[source + 1] > prefixes[source] && suffixes[row + 1] > suffixes[row],
                "Invalid block page source or length");
    const int64_t start = prefixes[source], length = prefixes[source + 1] - start;
    const auto full = length / page * page;
    if (first_page[source] < 0) {
      first_page[source] = gather.size() / page;
      for (int64_t i = 0; i < full; ++i)
        gather.push_back(start + i);
    }
    std::vector<int32_t> table;
    for (int64_t i = 0; i < full / page; ++i)
      table.push_back(first_page[source] + i);
    const auto first = gather.size() / page;
    for (int64_t i = full; i < length; ++i)
      gather.push_back(start + i);
    for (int64_t i = suffixes[row]; i < suffixes[row + 1]; ++i)
      gather.push_back(int64_t(prefixes.back()) + i);
    // Padding gathers a valid row but never contributes to the attention mask.
    gather.resize((gather.size() + page - 1) / page * page, 0);
    for (size_t i = first; i < gather.size() / page; ++i)
      table.push_back(i);
    columns = std::max(columns, table.size());
    rows.push_back(std::move(table));
    const auto kv = length + suffixes[row + 1] - suffixes[row];
    max_k = std::max(max_k, kv);
    expanded += kv;
  }
  // Small/unshared layouts are cheaper through the existing packed path.
  if (gather.size() >= size_t(expanded))
    return {};
  std::vector<int32_t> table(rows.size() * columns, 0);
  for (size_t row = 0; row < rows.size(); ++row)
    std::copy(rows[row].begin(), rows[row].end(), table.begin() + row * columns);
  return {at::tensor(gather, options.dtype(at::kLong)),
          at::tensor(table, options.dtype(at::kInt)).view({int64_t(rows.size()), int64_t(columns)}),
          max_k};
}

bool block_paged_supported(const at::Tensor& q) {
#ifdef TEI_TORCH_CUDA_KERNELS
  return q.is_cuda() && q.dim() == 3 && q.size(0) > 0 && q.stride(2) == 1 &&
         (q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16) &&
         (q.size(2) == 64 || q.size(2) == 128 || q.size(2) == 256) &&
         at::cuda::getDeviceProperties(q.get_device())->major >= 8;
#else
  return false;
#endif
}

at::Tensor block_paged_attention(const at::Tensor& q,
                                 const at::Tensor& k,
                                 const at::Tensor& v,
                                 const at::Tensor& prefix_k,
                                 const at::Tensor& prefix_v,
                                 const at::Tensor& cu_q,
                                 const at::Tensor& cu_k,
                                 const BlockPages& pages,
                                 int64_t max_q,
                                 double scale) {
  TORCH_CHECK(block_paged_supported(q) && pages.gather.defined(), "Unsupported block page layout");
  auto pack = [&](const at::Tensor& prefix, const at::Tensor& suffix) {
    return at::cat({prefix, suffix}, 0)
        .index_select(0, pages.gather)
        .view({-1, BlockPages::page_size, k.size(1), k.size(2)});
  };
  // One softmax over prefix + own suffix. Explicit num_splits=1 avoids the
  // provider's split-K workspace/merge and preserves its contiguous traversal.
  return std::get<0>(at::_flash_attention_forward(q,
                                                  pack(prefix_k, k),
                                                  pack(prefix_v, v),
                                                  cu_q,
                                                  cu_k,
                                                  max_q,
                                                  pages.max_k,
                                                  0.,
                                                  false,
                                                  false,
                                                  scale,
                                                  std::nullopt,
                                                  std::nullopt,
                                                  std::nullopt,
                                                  std::nullopt,
                                                  pages.table,
                                                  1));
}
}
