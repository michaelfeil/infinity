// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
/// \file
/// Packed host-to-device upload for native request token metadata.
/// Supports one blocking byte pack or a lease-owned direct pinned fill.

#pragma once
#include "batching/output.h"
#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>

namespace infinity {
/// Byte extents for three int64 token arrays followed by one int32 cumulative array.
struct TokenUploadLayout {
  int64_t tokens, batch;
  int64_t token_bytes, cumulative_bytes, total_bytes;
};

/// Validate packed token geometry and compute overflow-safe typed array extents.
inline TokenUploadLayout token_upload_layout(int64_t tokens, int64_t batch) {
  constexpr int64_t token_size = sizeof(int64_t), offset_size = sizeof(int32_t);
  constexpr auto maximum = std::numeric_limits<int64_t>::max();
  TORCH_CHECK(tokens > 0 && batch > 0, "Invalid packed token upload geometry");
  TORCH_CHECK(tokens <= maximum / token_size, "Overflowing packed token bytes");
  TORCH_CHECK(batch < maximum && batch + 1 <= maximum / offset_size,
              "Overflowing packed cumulative bytes");
  const auto token_bytes = tokens * token_size;
  const auto cumulative_bytes = (batch + 1) * offset_size;
  TORCH_CHECK(token_bytes <= (maximum - cumulative_bytes) / 3, "Overflowing packed token upload");
  const auto total_bytes = token_bytes * 3 + cumulative_bytes;
  TORCH_CHECK(static_cast<uint64_t>(total_bytes) <= std::numeric_limits<size_t>::max(),
              "Packed token upload does not fit host storage");
  return {tokens, batch, token_bytes, cumulative_bytes, total_bytes};
}

/// Device byte storage plus typed contiguous views for four packed token arrays.
struct TokenUpload {
  at::Tensor storage;
  at::Tensor ids, types, positions, cumulative;
};

/// Build typed views at fixed packed offsets. The views and returned storage own the same allocation.
inline TokenUpload token_upload_views(at::Tensor storage, const TokenUploadLayout& layout) {
  TORCH_CHECK(storage.defined() && storage.scalar_type() == at::kByte && storage.dim() == 1 &&
                  storage.is_contiguous() && storage.numel() == layout.total_bytes,
              "Invalid packed token storage");
  auto typed = [&](int64_t offset, int64_t bytes, at::ScalarType type, int64_t elements) {
    const auto element_size = static_cast<int64_t>(c10::elementSize(type));
    TORCH_CHECK(offset >= 0 && bytes >= 0 && bytes <= layout.total_bytes &&
                    offset <= layout.total_bytes - bytes && element_size > 0 &&
                    offset % element_size == 0 && bytes % element_size == 0,
                "Invalid packed token view");
    return storage.narrow(0, offset, bytes).view(type).view({elements});
  };
  auto ids = typed(0, layout.token_bytes, at::kLong, layout.tokens);
  auto types = typed(layout.token_bytes, layout.token_bytes, at::kLong, layout.tokens);
  auto positions = typed(2 * layout.token_bytes, layout.token_bytes, at::kLong, layout.tokens);
  auto cumulative =
      typed(3 * layout.token_bytes, layout.cumulative_bytes, at::kInt, layout.batch + 1);
  return {std::move(storage),
          std::move(ids),
          std::move(types),
          std::move(positions),
          std::move(cumulative)};
}

/// Copy three int64 token arrays and one int32 cumulative array into one CPU byte tensor,
/// then perform one blocking device upload. Callers validate descriptors before calling this.
inline TokenUpload upload_tokens(const int64_t* ids,
                                 const int64_t* types,
                                 const int64_t* positions,
                                 const int32_t* cumulative,
                                 int64_t tokens,
                                 int64_t batch,
                                 const c10::Device& device) {
  TORCH_CHECK(ids && types && positions && cumulative, "Missing native token buffer");
  const auto layout = token_upload_layout(tokens, batch);
  auto packed =
      at::empty({layout.total_bytes}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));
  auto* base = packed.mutable_data_ptr<uint8_t>();
  std::memcpy(base, ids, static_cast<size_t>(layout.token_bytes));
  std::memcpy(base + layout.token_bytes, types, static_cast<size_t>(layout.token_bytes));
  std::memcpy(base + 2 * layout.token_bytes, positions, static_cast<size_t>(layout.token_bytes));
  std::memcpy(
      base + 3 * layout.token_bytes, cumulative, static_cast<size_t>(layout.cumulative_bytes));
  return token_upload_views(packed.to(device), layout);
}

/// Fill one lease-owned pinned byte buffer directly from caller arrays and enqueue one
/// non-blocking device upload. The caller may release or overwrite its arrays on return.
inline TokenUpload upload_tokens(const int64_t* ids,
                                 const int64_t* types,
                                 const int64_t* positions,
                                 const int32_t* cumulative,
                                 int64_t tokens,
                                 int64_t batch,
                                 OutputLease& completion) {
  TORCH_CHECK(ids && types && positions && cumulative, "Missing native token buffer");
  const auto layout = token_upload_layout(tokens, batch);
  const std::array<HostBuffer, 4> sources{{
      {ids, static_cast<size_t>(layout.token_bytes)},
      {types, static_cast<size_t>(layout.token_bytes)},
      {positions, static_cast<size_t>(layout.token_bytes)},
      {cumulative, static_cast<size_t>(layout.cumulative_bytes)},
  }};
  auto storage = completion.upload_bytes(c10::ArrayRef<HostBuffer>(sources.data(), sources.size()));
  return token_upload_views(std::move(storage), layout);
}
}
