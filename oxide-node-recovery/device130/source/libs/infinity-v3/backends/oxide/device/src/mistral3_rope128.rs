// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Additive prefixed Pixtral/Mistral3 H128 rotary-table kernels.
//! Function bodies/constants are alpha-rename-identical to the qualified donor;
//! the donor's private round_bf16 helper is copied with a family prefix.

use cuda_device::{float, thread};

use crate::{bf16_from_f32, f32_from_bf16};

#[inline(always)]
fn mistral3_round_bf16(value: f32) -> f32 {
    f32_from_bf16(bf16_from_f32(value))
}

/// Native F32 phase -> CUDA libdevice trigonometry -> BF16 coefficients for
/// the H128 decoder rotary layout. This runs only during model construction,
/// never during request dispatch.
#[inline(always)]
pub(crate) unsafe fn mistral3_base_rope_tables_128_bf16(
    inverse: *const f32,
    cosine: *mut f32,
    sine: *mut f32,
    positions: u32,
) {
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if index >= positions as usize * 64 {
        return;
    }
    let phase = float::mul_rn_f32((index / 64) as f32, unsafe { *inverse.add(index % 64) });
    unsafe {
        *cosine.add(index) = mistral3_round_bf16(libm::cosf(phase));
        *sine.add(index) = mistral3_round_bf16(libm::sinf(phase));
    }
}

/// YaRN-scaled H128 decoder rotary coefficients. The host supplies the native
/// inverse frequencies and the YaRN attention scale; this kernel multiplies
/// cos/sin by that scale before BF16 rounding, exactly like the native decoder.
#[inline(always)]
pub(crate) unsafe fn mistral3_base_rope_tables_128_scaled_bf16(
    inverse: *const f32,
    cosine: *mut f32,
    sine: *mut f32,
    positions: u32,
    attention_scale: f32,
) {
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if index >= positions as usize * 64 {
        return;
    }
    let phase = float::mul_rn_f32((index / 64) as f32, unsafe { *inverse.add(index % 64) });
    unsafe {
        *cosine.add(index) = mistral3_round_bf16(float::mul_rn_f32(libm::cosf(phase), attention_scale));
        *sine.add(index) = mistral3_round_bf16(float::mul_rn_f32(libm::sinf(phase), attention_scale));
    }
}
