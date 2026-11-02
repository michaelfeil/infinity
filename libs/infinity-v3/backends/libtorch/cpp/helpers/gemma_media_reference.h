// SPDX-License-Identifier: MIT AND LicenseRef-Infinity-Restricted-MIT-1.0
// Scalar CPU reference for the EmbeddingGemma2 media primitives.
//
// This header intentionally depends on neither Torch nor CUDA. It mirrors the
// exact scalar order used by models/gemma_models.cpp so a CPU admission test can
// check the Oxide source contract without loading weights or creating a context.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace infinity::gemma_media {

inline float bf16_round(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffu + ((bits >> 16) & 1u);
  bits &= 0xffff0000u;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

/// Gemma DivSqrt norm: F32 statistics, F32 scale, one BF16 cast per output.
inline void norm_row(const float* input, const float* scale, float* output, int width,
                     float eps) {
  double sum = 0.0;
  for (int i = 0; i < width; ++i) sum += input[i];
  const float mean = static_cast<float>(sum / width);
  double variance = 0.0;
  for (int i = 0; i < width; ++i) {
    const float centered = input[i] - mean;
    variance += centered * centered;
  }
  const float denominator =
      std::sqrt(static_cast<float>(variance / width) + eps);
  for (int i = 0; i < width; ++i) {
    output[i] = bf16_round((input[i] - mean) / denominator * scale[i]);
  }
}

/// Unit RMS norm used by vision V and the final media projections.
inline void unit_norm_row(const float* input, float* output, int width, float eps) {
  double squares = 0.0;
  for (int i = 0; i < width; ++i) squares += input[i] * input[i];
  const float denominator =
      std::sqrt(static_cast<float>(squares / width) + eps);
  for (int i = 0; i < width; ++i) output[i] = bf16_round(input[i] / denominator);
}

/// Native axial rotate2d pair: `a*c - b*s`, `b*c + a*s` with per-op BF16 casts.
inline void axial_pair(float a, float b, float c, float s, float* first, float* second) {
  *first = bf16_round(bf16_round(a * c) - bf16_round(b * s));
  *second = bf16_round(bf16_round(b * c) + bf16_round(a * s));
}

/// Native vision pool: F32 mean over `pool x pool`, scaled by sqrt(hidden), one
/// BF16 cast. Input rows are spatial-major HWC.
inline void vision_pool_row(const float* input, float* output, int height, int width,
                            int pool, int hidden, int row) {
  const int out_w = width / pool;
  static_cast<void>(height);
  const int oy = row / out_w;
  const int ox = row % out_w;
  for (int column = 0; column < hidden; ++column) {
    double sum = 0.0;
    for (int y = 0; y < pool; ++y)
      for (int x = 0; x < pool; ++x)
        sum += input[((oy * pool + y) * width + ox * pool + x) * hidden + column];
    const float mean = static_cast<float>(sum / (pool * pool));
    output[row * hidden + column] =
        bf16_round(mean * std::sqrt(static_cast<float>(hidden)));
  }
}

struct RelativeShift {
  int row;
  int column;
  bool zero;
};

/// Native audio relative flatten/reshape/gather index, without the artificial
/// boundary rows. `key` is absolute and `start` is the chunk start.
inline RelativeShift relative_shift(int query_local, int key, int start, int past,
                                    int chunk, int future, int span) {
  const int context = chunk + past + future;
  const int shifted = query_local * context + (key - (start - past));
  const int row = shifted / (context + 1);
  const int column = shifted % (context + 1);
  return RelativeShift{row, std::min(column, span - 1), column >= span};
}

}  // namespace infinity::gemma_media
