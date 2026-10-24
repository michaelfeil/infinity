#pragma once
#include <ATen/ATen.h>

namespace tei {
at::Tensor torch_flash_exact32(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t, double);
}
