/// \file
/// Packed attention composition and bounded sequence-local bias construction.
/// Provides native CUDA dispatch and independent per-sequence attention fallbacks.

#pragma once
#include <ATen/ATen.h>
#include "batching/packed_input.h"
#include "helpers/checked_size.h"
#include "helpers/admission_error.h"
#include "kernels/infinity_fa4.h"
#include <ATen/ops/_flash_attention_forward.h>
#include <ATen/ops/_efficient_attention_forward.h>

namespace tei {
// Per-sequence bias tiles for the native packed efficient attention operator.
// Its bias indexing uses local Q/K positions and stride(0) for each sequence,
// while public tensor metadata requires logical batch=1 and total-token axes.
// Allocate enough storage for BOTH the bounded logical view and actual tiles.
inline constexpr int64_t default_attention_bias_max_bytes = 512LL << 20;

inline void admit_attention_bias(int64_t bytes, int64_t limit) {
  TORCH_CHECK(limit > 0 && bytes >= 0, "Invalid packed attention bias memory budget");
  require_memory_admission(bytes <= limit, "Packed attention bias exceeds its memory budget");
}

inline at::Tensor packed_sequence_bias(const at::Tensor& q,
                                       const PackedInput& input,
                                       int64_t max_bytes = default_attention_bias_max_bytes) {
  const int64_t total = q.size(0), heads = q.size(1), nmax = input.max_sequence;
  TORCH_CHECK(total > 0 && heads > 0 && input.batch > 0 && nmax > 0 && nmax <= total,
              "Invalid packed sequence bias dimensions");
  const int64_t stride = checked_size_product(checked_size_sum(nmax, 7) / 8, 8);
  const int64_t tile = checked_size_product(nmax, stride),
                batch_stride = checked_size_product(heads, tile);
  const int64_t size =
      std::max(checked_size_product(input.batch, batch_stride),
               checked_size_sum(checked_size_sum(checked_size_product(heads - 1, tile),
                                                 checked_size_product(total - 1, stride)),
                                total));
  admit_attention_bias(checked_size_product(size, q.element_size()), max_bytes);
  checked_size_product(checked_size_product(heads, total), total);
  return at::empty({size}, q.options())
      .as_strided({1, heads, total, total}, {batch_stride, tile, stride, 1});
}

// PyTorch 2.14.1 validates a bias with logical total-token dimensions, while
// its varlen efficient-attention kernel reads sequence-local query/key offsets.
// Overlapping read-only strides express one shared ALiBi tile per head. The
// backing allocation covers every logical view address (including unused tails),
// and grows as total*max_sequence + heads*max_sequence^2, never total^2.
// Q/K/V retain precisely the real packed tokens; only bias storage is aligned.
inline at::Tensor packed_alibi_bias(const at::Tensor& q,
                                    const PackedInput& input,
                                    const at::Tensor& slopes,
                                    int64_t window_left = -1,
                                    int64_t window_right = -1,
                                    int64_t max_bytes = default_attention_bias_max_bytes) {
  const int64_t total = q.size(0), heads = q.size(1), length = input.max_sequence;
  TORCH_CHECK(total > 0 && length > 0 && length <= total && heads > 0,
              "Invalid packed ALiBi dimensions");
  const int64_t stride = checked_size_product(checked_size_sum(length, 7) / 8, 8),
                tile = checked_size_product(length, stride);
  const int64_t backing =
      checked_size_sum(checked_size_sum(checked_size_product(heads - 1, tile),
                                        checked_size_product(total - 1, stride)),
                       total);
  // Bound backing plus a conservative sum of construction temporaries:
  // index, signed/absolute deltas, broadcast multiply, dtype conversion and mask.
  const auto cells = checked_size_product(length, length);
  const auto head_cells = checked_size_product(heads, cells);
  auto bytes = checked_size_product(backing, q.element_size());
  bytes = checked_size_sum(bytes, checked_size_product(length, sizeof(int64_t)));
  bytes = checked_size_sum(bytes, checked_size_product(cells, 2 * sizeof(int64_t) + sizeof(bool)));
  bytes = checked_size_sum(
      bytes,
      checked_size_product(
          head_cells,
          checked_size_sum(std::max<int64_t>(slopes.element_size(), q.element_size()),
                           q.element_size())));
  admit_attention_bias(bytes, max_bytes);
  checked_size_product(checked_size_product(heads, total), total);
  auto storage = at::empty({backing}, q.options());
  auto local = storage.as_strided({heads, length, length}, {tile, stride, 1});
  auto idx = at::arange(length, q.options().dtype(at::kLong));
  auto delta = idx.unsqueeze(0) - idx.unsqueeze(1);
  local.copy_((-slopes.view({heads, 1, 1}) * delta.abs()).to(q.scalar_type()));
  if (window_left >= 0)
    local.masked_fill_(delta < -window_left, -INFINITY);
  if (window_right >= 0)
    local.masked_fill_(delta > window_right, -INFINITY);
  return storage.as_strided({1, heads, total, total}, {0, tile, stride, 1});
}

// Q/K/V are packed [total_tokens, heads, head_dim]; cumulative offsets delimit sequences.
inline at::Tensor packed_attention(const at::Tensor& q,
                                   const at::Tensor& k,
                                   const at::Tensor& v,
                                   const PackedInput& input,
                                   double scale,
                                   bool causal = false,
                                   int64_t window_left = -1,
                                   int64_t window_right = -1,
                                   const std::optional<at::Tensor>& alibi = std::nullopt,
                                   const std::optional<at::Tensor>& prepared_bias = std::nullopt) {
  if (q.is_cuda()) {
    if (!alibi && !prepared_bias) {
      auto fa4 = infinity_fa4(q,
                              k,
                              v,
                              input.cumulative,
                              input.cumulative,
                              input.max_sequence,
                              scale,
                              causal,
                              window_left,
                              window_right);
      if (fa4.defined())
        return fa4;
    }
    if (alibi) {
      // The prebuilt Torch Flash operator disables ALiBi. Its efficient
      // operator accepts actual packed Q/K/V and cumulative sequence offsets.
      auto bias = prepared_bias.value_or(at::Tensor());
      if (!bias.defined())
        bias = packed_alibi_bias(q, input, *alibi, window_left, window_right);
      return std::get<0>(at::_efficient_attention_forward(q.unsqueeze(0),
                                                          k.unsqueeze(0),
                                                          v.unsqueeze(0),
                                                          bias,
                                                          input.cumulative,
                                                          input.cumulative,
                                                          input.max_sequence,
                                                          input.max_sequence,
                                                          0.0,
                                                          causal ? 1 : 0,
                                                          false,
                                                          scale))
          .squeeze(0);
    }
    return std::get<0>(at::_flash_attention_forward(q,
                                                    k,
                                                    v,
                                                    input.cumulative,
                                                    input.cumulative,
                                                    input.max_sequence,
                                                    input.max_sequence,
                                                    0.0,
                                                    causal,
                                                    false,
                                                    scale,
                                                    window_left,
                                                    window_right,
                                                    std::nullopt,
                                                    alibi));
  }
  std::vector<at::Tensor> sequences;
  for (int64_t row = 0; row < input.batch; ++row) {
    const int64_t start = input.offsets[row], length = input.offsets[row + 1] - start;
    auto slice = [&](const at::Tensor& t) {
      return t.narrow(0, start, length).transpose(0, 1).unsqueeze(0);
    };
    std::optional<at::Tensor> mask;
    if (alibi || window_left >= 0 || window_right >= 0) {
      auto idx = at::arange(length, q.options().dtype(at::kLong));
      auto delta = idx.unsqueeze(0) - idx.unsqueeze(1);
      auto allowed = at::ones({length, length}, q.options().dtype(at::kBool));
      if (causal)
        allowed.logical_and_(delta <= 0);
      if (window_left >= 0)
        allowed.logical_and_(delta >= -window_left);
      if (window_right >= 0)
        allowed.logical_and_(delta <= window_right);
      auto bias = at::zeros({q.size(1), length, length}, q.options());
      if (alibi)
        bias -= alibi->view({q.size(1), 1, 1}) * delta.abs();
      mask = bias.masked_fill(allowed.logical_not(), -INFINITY).unsqueeze(0);
    }
    sequences.push_back(
        at::scaled_dot_product_attention(
            slice(q), slice(k), slice(v), mask, 0.0, causal && !mask, scale, q.size(1) != k.size(1))
            .squeeze(0)
            .transpose(0, 1));
  }
  return at::cat(sequences, 0);
}
}
