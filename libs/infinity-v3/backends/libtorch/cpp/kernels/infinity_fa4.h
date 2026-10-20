// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
#pragma once
#include <ATen/ATen.h>

namespace tei {
// Undefined means this layout/mask has no exported kernel. A failed supported
// launch raises an error; it must never silently retry another implementation.
at::Tensor infinity_fa4(const at::Tensor& q, const at::Tensor& k,
    const at::Tensor& v, const at::Tensor& cu_q, const at::Tensor& cu_k,
    int64_t max_q, double scale, bool causal=false,
    int64_t left=-1, int64_t right=-1);
bool infinity_fa4_deberta_supported(const at::Tensor& reference, int64_t head_dim=64);
at::Tensor infinity_fa4_deberta(const at::Tensor& q, const at::Tensor& scaled_k,
    const at::Tensor& v, const at::Tensor& cumulative,
    const at::Tensor& c2p, const at::Tensor& p2c, const at::Tensor& buckets,
    int64_t max_sequence);
}
