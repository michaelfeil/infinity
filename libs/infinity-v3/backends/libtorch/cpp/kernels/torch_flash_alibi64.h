// Pinned Torch Flash implementation and its BSD notices remain in vendor/torch_flash_attention.
#pragma once
#include <ATen/ATen.h>

namespace tei {
bool torch_flash_alibi64_supported(c10::Device, at::ScalarType, int64_t);
at::Tensor torch_flash_alibi64(const at::Tensor&,
                               const at::Tensor&,
                               const at::Tensor&,
                               const at::Tensor&,
                               const at::Tensor&,
                               int64_t,
                               double);
}
