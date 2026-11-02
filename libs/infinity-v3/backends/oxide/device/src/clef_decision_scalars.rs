// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Cold Native BF16 scalar preparation for the Clef joint head.
//!
//! `bridge.cpp::weight_dtype` first casts every Clef head tensor to the BF16
//! model dtype. Native `ClefHead` then evaluates, on those BF16 tensors:
//!
//! * `prior_logit_scale.clamp_max(log(100)).exp().to(F32).item<float>()`;
//! * `joint_logit_scale.clamp_max(log(100)).exp().to(F32).item<float>()`;
//! * `residual_gate.sigmoid().to(F32).item<float>()`.
//!
//! This source-only cold kernel preserves each BF16 boundary: the source scalar
//! is already BF16, the clamp bound is rounded to BF16, exp/sigmoid results are
//! rounded to BF16, and only then widened exactly to F32. It is launched once
//! at cold head load. Root must export it from `device/src/main.rs`; the ATen-
//! equivalent primitive is pending Root gating.

use cuda_device::{convert, thread};

#[inline(always)]
fn widen(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[inline(always)]
fn round(value: f32) -> u16 {
    convert::cvt_bf16x2_f32(value, value) as u16
}

/// Grid=(1,1,1), block=256. Input is three BF16 scalars in Native order:
/// prior scale, joint scale, residual gate. Output is three F32 values widened
/// from their final BF16 results. Only the first three threads write.
pub(crate) unsafe fn scalar_prep(input: *const u16, out: *mut f32, count: u32) {
    let index = thread::threadIdx_x() as usize;
    if index >= count as usize || index >= 3 {
        return;
    }
    let value = widen(unsafe { *input.add(index) });
    let prepared = if index < 2 {
        let bound = widen(unsafe { round(libm::logf(100.0f32)) });
        let clamped = if value > bound { bound } else { value };
        widen(unsafe { round(libm::expf(clamped)) })
    } else {
        let sigmoid = 1.0 / (1.0 + libm::expf(-value));
        widen(unsafe { round(sigmoid) })
    };
    unsafe { *out.add(index) = prepared };
}
