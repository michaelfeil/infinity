/// \file
/// Native packed Flash attention interface for the head-width-32 specialization.

#pragma once
#include <ATen/ATen.h>

namespace tei {
/// Execute the qualified width-32 packed attention specialization on real token rows.
at::Tensor torch_flash_exact32(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t, double);
}
