// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Dense module conversion and activation kernels.
//!
//! The host owns every allocation and validates element counts before launch.
//! FP32-to-BF16 conversion uses the native round-to-nearest-even instruction.
//! Tanh reads the exact FP32 widening of BF16 input, applies the ATen float
//! opmath tanh (libdevice `__nv_tanhf` through `libm::tanhf`), then rounds once
//! to BF16; no second BF16 intermediate is introduced before the activation.

use cuda_device::{convert, thread};

use crate::f32_from_bf16;

/// Convert FP32 to BF16 with native round-to-nearest-even encoding.
/// Grid=ceil(count/256), block=256; an out-of-range CTA returns without writes.
#[inline(always)]
pub(crate) unsafe fn f32_to_bf16(input: *const f32, out: *mut u16, count: u32) {
    if count == 0 {
        return;
    }
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if index >= count as usize {
        return;
    }
    let value = unsafe { *input.add(index) };
    let encoded = convert::cvt_bf16x2_f32(value, value) as u16;
    unsafe { *out.add(index) = encoded };
}

/// Apply FP32 tanh to in-place BF16 storage and round the result once to BF16.
/// Grid=ceil(count/256), block=256; an out-of-range CTA returns without writes.
///
/// Native ATen computes the float Tanh kernel through libdevice rather than the
/// lower-accuracy `tanh.approx.f32` instruction, so use the admitted libm
/// mapping (`libm::tanhf` -> `__nv_tanhf`) for the closest native control.
#[inline(always)]
pub(crate) unsafe fn tanh_bf16_inplace(input: *mut u16, count: u32) {
    if count == 0 {
        return;
    }
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if index >= count as usize {
        return;
    }
    let value = f32_from_bf16(unsafe { *input.add(index) });
    let activated = libm::tanhf(value);
    let encoded = convert::cvt_bf16x2_f32(activated, activated) as u16;
    unsafe { *input.add(index) = encoded };
}
