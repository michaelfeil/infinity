// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Pure CPU semantic reference for the four pointwise Native activations.
//!
//! This is a semantic F32-opmath reference, not a claim of bit-exact CUDA
//! agreement. Arithmetic happens in F32; only storage reads and the final write
//! cross the BF16 boundary. Constants stay F32 literals and are never decoded
//! as BF16. The actual CUDA ATen oracle packet in `native/` is the runtime
//! qualification authority.

use crate::encoder_activation::Activation;
use crate::encoder_activation_bf16::{bits_from_f32, f32_from_bits};
use crate::encoder_activation_plan::BiasMode;
use crate::encoder_activation_plan::{ActivationPlan, PlanError};

/// Sentinel used outside active output rows in reference and root controls.
pub const OUTPUT_POISON: u16 = 0x7f7f;

/// `sqrt(2/pi)` widened from the Native double literal to one F32 constant.
pub const GELU_TANH_COEFFICIENT: f32 = f32::from_bits(0x3f4c_422a);
/// The Native tanh-GELU cubic coefficient as one F32 constant.
pub const GELU_TANH_CUBIC: f32 = 0.044_715_f32;

/// Native `at::gelu(x, "tanh")` semantic formula in F32 opmath.
#[inline]
pub fn gelu_tanh_f32(value: f32) -> f32 {
    let cube = value * value * value;
    let alpha = value + GELU_TANH_CUBIC * cube;
    let argument = GELU_TANH_COEFFICIENT * alpha;
    0.5 * value * (1.0 + libm::tanhf(argument))
}

/// Native `at::relu` scalar semantics: NaN propagates, negative values become
/// +0, and nonnegative values (including -0) are preserved.
#[inline]
pub fn relu_f32(value: f32) -> f32 {
    if value.is_nan() || value >= 0.0 {
        value
    } else {
        0.0
    }
}

/// Native `at::silu` semantic formula in F32 opmath.
#[inline]
pub fn silu_f32(value: f32) -> f32 {
    value / (1.0 + libm::expf(-value))
}

#[inline]
pub fn tanh_f32(value: f32) -> f32 {
    libm::tanhf(value)
}

#[inline]
pub fn activation_f32(activation: Activation, value: f32) -> f32 {
    match activation {
        Activation::GeluTanh => gelu_tanh_f32(value),
        Activation::Relu => relu_f32(value),
        Activation::Silu => silu_f32(value),
        Activation::Tanh => tanh_f32(value),
    }
}

/// Activate one already-widened value under the explicit contract.
#[inline]
pub fn activate_value(
    activation: Activation,
    bias_mode: BiasMode,
    value: f32,
    bias: Option<f32>,
) -> Result<f32, PlanError> {
    let operand = match (bias_mode, bias) {
        (BiasMode::None, _) => value,
        (BiasMode::SeparateBf16, Some(bias)) => f32_from_bits(bits_from_f32(value + bias)),
        (BiasMode::SeparateBf16, None) => return Err(PlanError::MissingBias),
    };
    Ok(activation_f32(activation, operand))
}

/// Activate one BF16 storage word and round once to BF16.
#[inline]
pub fn activate_bits(
    activation: Activation,
    bias_mode: BiasMode,
    value: u16,
    bias: Option<u16>,
) -> Result<u16, PlanError> {
    let value = f32_from_bits(value);
    let bias = bias.map(f32_from_bits);
    Ok(bits_from_f32(activate_value(
        activation, bias_mode, value, bias,
    )?))
}

/// Run one reference case into caller-owned padded output storage.
///
/// The output storage must cover the plan's output span and is initialized by
/// the caller (typically to [`OUTPUT_POISON`]). Only active `[rows, width]`
/// elements are written; row padding and any surrounding guard words are left
/// untouched. Input storage is read-only and is never modified.
pub fn run_into(
    plan: &ActivationPlan,
    input: &[u16],
    bias: Option<&[u16]>,
    output: &mut [u16],
) -> Result<(), PlanError> {
    let input_span = plan.input_layout().span_elements();
    let output_span = plan.output_layout().span_elements();
    if plan.elements() != 0 && input.len() < input_span {
        return Err(PlanError::Tensor(
            crate::tensor::TensorError::AddressOverflow,
        ));
    }
    if plan.elements() != 0 && output.len() < output_span {
        return Err(PlanError::Tensor(
            crate::tensor::TensorError::AddressOverflow,
        ));
    }
    if plan.bias().has_bias() && bias.is_none_or(|bias| bias.len() < plan.width()) {
        return Err(PlanError::MissingBias);
    }
    if !plan.bias().has_bias() && bias.is_some() {
        return Err(PlanError::UnexpectedBias);
    }
    let input_stride = plan.input_layout().strides()[0];
    let output_stride = plan.output_layout().strides()[0];
    for row in 0..plan.rows() {
        for column in 0..plan.width() {
            let source = input[row * input_stride + column];
            let bias = bias.map(|bias| bias[column]);
            let active = activate_bits(plan.activation(), plan.bias(), source, bias)?;
            output[row * output_stride + column] = active;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder_activation_bf16::Bf16;

    fn bits(value: f32) -> u16 {
        Bf16::from_f32(value).to_bits()
    }

    #[test]
    fn relu_preserves_signed_zero_and_propagates_nan() {
        assert_eq!(relu_f32(-1.0), 0.0);
        assert_eq!(bits_from_f32(relu_f32(-0.0)), 0x8000);
        assert_eq!(bits_from_f32(relu_f32(0.0)), 0x0000);
        assert_eq!(bits_from_f32(relu_f32(1.0)), bits(1.0));
        assert!(relu_f32(f32::NAN).is_nan());
        assert_eq!(bits_from_f32(relu_f32(f32::INFINITY)), 0x7f80);
        assert_eq!(bits_from_f32(relu_f32(f32::NEG_INFINITY)), 0x0000);
    }

    #[test]
    fn gelu_and_silu_keep_negative_zero() {
        assert_eq!(bits_from_f32(gelu_tanh_f32(-0.0)), 0x8000);
        assert_eq!(bits_from_f32(gelu_tanh_f32(0.0)), 0x0000);
        assert_eq!(bits_from_f32(silu_f32(-0.0)), 0x8000);
        assert_eq!(bits_from_f32(silu_f32(0.0)), 0x0000);
    }

    #[test]
    fn separate_bias_requires_an_explicit_bias_value() {
        assert_eq!(
            activate_value(Activation::Relu, BiasMode::SeparateBf16, 1.0, None),
            Err(PlanError::MissingBias)
        );
        assert_eq!(
            activate_bits(Activation::Relu, BiasMode::SeparateBf16, 0x3f80, None),
            Err(PlanError::MissingBias)
        );
    }

    #[test]
    fn separate_bias_rounds_once_before_activation() {
        let input = 1.0f32;
        let bias = Bf16::from_bits(0x3f81).to_f32();
        let separate =
            activate_value(Activation::Relu, BiasMode::SeparateBf16, input, Some(bias)).unwrap();
        assert_eq!(separate, 2.0);
        let none = activate_value(Activation::Relu, BiasMode::None, input, Some(bias)).unwrap();
        assert_eq!(none, 1.0);
    }

    #[test]
    fn run_into_ignores_padding_and_preserves_guards() {
        let input = [bits(1.0), bits(-1.0), 0xaaaa, 0xbbbb, bits(2.0)];
        let plan = ActivationPlan::activation_only(Activation::Relu, 2, 2, 3, 3).unwrap();
        let mut output = vec![OUTPUT_POISON; 5];
        run_into(&plan, &input, None, &mut output).unwrap();
        assert_eq!(output[0], bits(1.0));
        assert_eq!(output[1], 0);
        assert_eq!(output[2], OUTPUT_POISON);
        assert_eq!(output[3], 0);
        assert_eq!(output[4], bits(2.0));
        assert_eq!(input[2], 0xaaaa);
        assert_eq!(input[3], 0xbbbb);
    }

    #[test]
    fn bias_is_broadcast_per_column() {
        let input = [bits(1.0); 6];
        let bias = [bits(1.0), bits(-1.0)];
        let plan = ActivationPlan::separate_bf16_bias(Activation::Relu, 3, 2, 2, 2).unwrap();
        let mut output = vec![OUTPUT_POISON; 6];
        run_into(&plan, &input, Some(&bias), &mut output).unwrap();
        assert_eq!(output, [bits(2.0), 0, bits(2.0), 0, bits(2.0), 0]);
    }
}
