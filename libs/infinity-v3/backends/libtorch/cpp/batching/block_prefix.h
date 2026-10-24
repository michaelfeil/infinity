/// \file
/// Exact request-local prefix comparison over borrowed token and image descriptors.
/// Used to select representatives only when their attention inputs agree.

#pragma once
#include "bridge.h"
#include <algorithm>
#include <cstring>

namespace infinity {
// Compare the exact borrowed host inputs before selecting a request-local
// representative. Image marker IDs alone do not identify visual content.
inline bool same_block_prefix(int64_t left,
                              int64_t right,
                              int64_t length,
                              const int64_t* ids,
                              const int64_t* types,
                              const int64_t* positions,
                              const int64_t* axes,
                              int64_t tokens,
                              const int32_t* offsets,
                              const TeiImage* images,
                              size_t image_count) {
  const auto a = offsets[left], b = offsets[right];
  for (const auto* values : {ids, types, positions})
    if (!std::equal(values + a, values + a + length, values + b))
      return false;
  if (axes)
    for (int axis = 0; axis < 3; ++axis)
      if (!std::equal(axes + axis * tokens + a,
                      axes + axis * tokens + a + length,
                      axes + axis * tokens + b))
        return false;
  auto next = [&](size_t& index, int64_t start) -> const TeiImage* {
    while (index < image_count) {
      const auto& image = images[index++];
      if (image.sequence_start == start && image.token_start < start + length)
        return &image;
    }
    return nullptr;
  };
  size_t ai = 0, bi = 0;
  for (;;) {
    auto x = next(ai, a), y = next(bi, b);
    if (!x || !y)
      return x == y;
    if (x->token_start - a != y->token_start - b || x->token_count != y->token_count ||
        x->rows != y->rows || x->patch_dim != y->patch_dim || x->merge_size != y->merge_size ||
        !std::equal(x->grid, x->grid + 3, y->grid))
      return false;
    // Descriptor validation precedes this comparison, including checked byte
    // sizes. Caller-supplied feature hashes never authorize prefix sharing.
    if (x->pixels != y->pixels &&
        std::memcmp(x->pixels, y->pixels, size_t(x->rows) * size_t(x->patch_dim) * sizeof(float)))
      return false;
  }
}
}
