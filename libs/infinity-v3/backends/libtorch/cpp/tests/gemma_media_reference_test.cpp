// CPU-only admission test for helpers/gemma_media_reference.h.
#include "helpers/gemma_media_reference.h"

#include <cassert>
#include <cmath>
#include <cstdio>

int main() {
  using namespace infinity::gemma_media;
  assert(bf16_round(1.0f) == 1.0f);
  assert(bf16_round(0.5f) == 0.5f);

  const float values[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  const float scale[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  float normed[4];
  norm_row(values, scale, normed, 4, 1e-6f);
  const float mean = 2.5f;
  const float variance = (2.25f + 0.25f + 0.25f + 2.25f) / 4.0f;
  const float denom = std::sqrt(variance + 1e-6f);
  for (int i = 0; i < 4; ++i) {
    assert(normed[i] == bf16_round((values[i] - mean) / denom));
  }

  float first, second;
  axial_pair(1.0f, 2.0f, 0.5f, 0.25f, &first, &second);
  assert(first == bf16_round(0.5f - 0.5f));
  assert(second == bf16_round(1.0f + 0.25f));

  const float pooled_in[36] = {
      1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6,
      1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6};
  float pooled[6];
  vision_pool_row(pooled_in, pooled, 2, 3, 1, 6, 0);
  assert(pooled[0] == bf16_round(1.0f * std::sqrt(6.0f)));

  const auto shift = relative_shift(0, 0, 0, 12, 12, 0, 13);
  assert(shift.row == 0);
  assert(shift.column == 12);
  assert(!shift.zero);
  const auto late = relative_shift(11, 11, 0, 12, 12, 0, 13);
  assert(late.row == 11);
  assert(late.zero || late.column < 13);

  std::puts("gemma_media_reference_test: pass");
  return 0;
}
