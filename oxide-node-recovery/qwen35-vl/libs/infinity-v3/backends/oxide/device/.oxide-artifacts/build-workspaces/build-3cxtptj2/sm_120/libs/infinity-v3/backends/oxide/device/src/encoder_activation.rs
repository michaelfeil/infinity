// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Optional BF16 pointwise encoder activations over an already-linear input.
//!
//! One thread owns one active `[row, column]` element. Row padding never enters
//! the active index range. A null bias pointer activates the BF16 input as-is.
//! A non-null bias pointer is an explicit separate BF16 bias add: the F32 sum is
//! rounded once to BF16 before activation. Activation arithmetic stays F32 and
//! only the written result crosses the BF16 boundary. This is not a fused GEMM
//! epilogue and makes no claim about `at::linear`.

use crate::{bf16_from_f32, f32_from_bf16};
use cuda_device::{float, thread};

/// Native `sqrt(2/pi)` widened to its exact F32 bit pattern.
const GELU_TANH_COEFFICIENT: f32 = f32::from_bits(0x3f4c_422a);
/// Native tanh-GELU cubic coefficient as one F32 constant.
const GELU_TANH_CUBIC: f32 = 0.044_715_f32;

#[inline(always)]
fn gelu_tanh(value: f32) -> f32 {
    let square = float::mul_rn_f32(value, value);
    let cube = float::mul_rn_f32(square, value);
    let cubic = float::mul_rn_f32(GELU_TANH_CUBIC, cube);
    let alpha = float::add_rn_f32(value, cubic);
    let argument = float::mul_rn_f32(GELU_TANH_COEFFICIENT, alpha);
    let tangent = libm::tanhf(argument);
    let plus = float::add_rn_f32(1.0, tangent);
    let half = float::mul_rn_f32(0.5, value);
    float::mul_rn_f32(half, plus)
}

#[inline(always)]
fn relu(value: f32) -> f32 {
    if value.is_nan() || value >= 0.0 {
        value
    } else {
        0.0
    }
}

#[inline(always)]
fn silu(value: f32) -> f32 {
    let exponential = libm::expf(-value);
    let denominator = float::add_rn_f32(1.0, exponential);
    float::div_rn_f32(value, denominator)
}

#[inline(always)]
fn activate_value<const KIND: u32>(value: f32) -> f32 {
    match KIND {
        0 => gelu_tanh(value),
        1 => relu(value),
        2 => silu(value),
        _ => libm::tanhf(value),
    }
}

/// Grid=ceil(rows*width/256), block=256; input and output may be row-padded.
/// Bias is optional (null) and broadcast across the active columns.
#[inline(always)]
pub(crate) unsafe fn activate<const KIND: u32>(
    input: *const u16,
    bias: *const u16,
    out: *mut u16,
    rows: u32,
    width: u32,
    input_stride: u32,
    output_stride: u32,
) {
    if rows == 0 || width == 0 {
        return;
    }
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if index >= rows as usize * width as usize {
        return;
    }
    let row = index / width as usize;
    let column = index % width as usize;
    let mut value = f32_from_bf16(unsafe { *input.add(row * input_stride as usize + column) });
    if !bias.is_null() {
        let bias_value = f32_from_bf16(unsafe { *bias.add(column) });
        value = f32_from_bf16(bf16_from_f32(float::add_rn_f32(value, bias_value)));
    }
    let active = activate_value::<KIND>(value);
    unsafe {
        *out.add(row * output_stride as usize + column) = bf16_from_f32(active);
    }
}
