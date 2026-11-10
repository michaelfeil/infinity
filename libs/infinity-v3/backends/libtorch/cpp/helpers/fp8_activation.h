// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
#pragma once

namespace infinity {
// Explicit numerical recipes eligible for fused unary activation plus row FP8 quantization.
enum class Fp8Activation { Relu, RoundedSilu, RoundedGelu };
// Explicit gated recipes. These retain the original native gated kernel's selected layout recipe:
// aligned vector rows may use the native half-pair HFMA staging, while ragged or unaligned rows
// retain its scalar separate staging. Both differ from the unary FP8 activation recipes above.
// PreciseGelu is the Gemma scalar-staged GeGLU: it always keeps the separate model-dtype multiply
// and add staging of gemma_precise_activation, so it matches gemma_gated_cuda exactly and never
// selects the aligned half-pair HFMA recipe.
enum class Fp8GatedActivation { RoundedSilu, RoundedGelu, PreciseGelu };
}
