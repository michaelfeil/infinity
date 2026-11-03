// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
#pragma once

namespace infinity {
// Explicit numerical recipes eligible for fused unary activation plus row FP8 quantization.
enum class Fp8Activation { Relu, RoundedSilu, RoundedGelu };
}
