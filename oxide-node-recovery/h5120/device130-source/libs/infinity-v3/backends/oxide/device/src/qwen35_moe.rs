// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Native Qwen3.5-MoE shared-expert sigmoid gate.
//!
//! For each token/column the Native `feed_forward` path computes one BF16
//! sigmoid from the `[1, hidden]` shared gate logits and scales the shared
//! expert down projection before adding it to the routed FFN output:
//!
//! `out = bf16(bf16(out) + bf16(bf16(shared_down) * bf16(sigmoid(gate))))`
//!
//! Every stage is qwen35_moe_rounded to BF16. The sigmoid follows the Native reciprocal
//! chain (`1 / (1 + exp(-logit))`) with BF16 rounding after `exp`, after the
//! add and after the divide. One lane owns one BF16 element; `routed` may
//! alias the output because each lane reads its routed value before writing.
//! Grid=ceil(tokens*width/256), block=256.

use cuda_device::{float, thread};

use crate::{bf16_from_f32, f32_from_bf16};

#[inline(always)]
fn qwen35_moe_rounded(value: f32) -> f32 {
    f32_from_bf16(bf16_from_f32(value))
}

#[inline(always)]
pub(crate) unsafe fn qwen35_moe_shared_gate_bf16(
    routed: *mut u16,
    shared_down: *const u16,
    gate: *const u16,
    tokens: u32,
    width: u32,
) {
    if tokens == 0 || width == 0 {
        return;
    }
    let width = width as u64;
    let total = (tokens as u64) * width;
    let index = (thread::blockIdx_x() as u64) * (thread::blockDim_x() as u64)
        + (thread::threadIdx_x() as u64);
    if index >= total {
        return;
    }
    let token = (index / width) as usize;
    let logit = f32_from_bf16(unsafe { *gate.add(token) });
    let exponential = qwen35_moe_rounded(libm::expf(-logit));
    let denominator = qwen35_moe_rounded(float::add_rn_f32(1.0, exponential));
    let sigmoid = qwen35_moe_rounded(float::div_rn_f32(1.0, denominator));
    let down = f32_from_bf16(unsafe { *shared_down.add(index as usize) });
    let scaled = qwen35_moe_rounded(float::mul_rn_f32(down, sigmoid));
    let routed_value = f32_from_bf16(unsafe { *routed.add(index as usize) });
    let sum = qwen35_moe_rounded(float::add_rn_f32(routed_value, scaled));
    unsafe { *routed.add(index as usize) = bf16_from_f32(sum) };
}
