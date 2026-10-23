// Copyright 2026 Infinity Core Contributors.
// Covered by Infinity Restricted MIT License 1.0; see LICENSE.
#pragma once
#include <ATen/ATen.h>
namespace tei {
struct Nvfp4PrototypePacked {at::Tensor packed,scales,dequantized;};
Nvfp4PrototypePacked nvfp4_prototype_quantize_cuda(const at::Tensor& input,
                                                const at::Tensor& global,bool dequantize);
}
