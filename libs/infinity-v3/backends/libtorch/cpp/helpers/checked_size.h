// SPDX-License-Identifier: Apache-2.0
/// \file
/// Checked nonnegative size arithmetic for native attention storage estimates.

#pragma once
#include <c10/util/Exception.h>
#include <cstdint>
#include <limits>

namespace tei {
inline int64_t checked_size_product(int64_t left, int64_t right) {
  TORCH_CHECK(left >= 0 && right >= 0 &&
                  (!right || left <= std::numeric_limits<int64_t>::max() / right),
              "Attention storage size overflow");
  return left * right;
}

inline int64_t checked_size_sum(int64_t left, int64_t right) {
  TORCH_CHECK(left >= 0 && right >= 0 && right <= std::numeric_limits<int64_t>::max() - left,
              "Attention storage size overflow");
  return left + right;
}
}
