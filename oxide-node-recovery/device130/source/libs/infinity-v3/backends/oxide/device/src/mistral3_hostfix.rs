// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Additive Mistral3 shaped YaRN table kernel. Keeps the existing H128
//! base/scaled leaves untouched and accepts the host's generic pair count.

use cuda_device::{float, thread};
use crate::{bf16_from_f32, f32_from_bf16};

/// Native shaped scaled table initializer: phase = pos * inverse, libdevice
/// cos/sin multiplied by the YaRN attention scale, then one BF16 round-trip.
#[inline(always)]
pub(crate) unsafe fn rope_tables_scaled_bf16(
    inverse: *const f32,
    cosine: *mut f32,
    sine: *mut f32,
    positions: u32,
    pairs: u32,
    scale: f32,
) {
    if positions == 0 || pairs == 0 {
        return;
    }
    let pairs = pairs as u64;
    let total = positions as u64 * pairs;
    let index = thread::blockIdx_x() as u64 * 256 + thread::threadIdx_x() as u64;
    if index >= total {
        return;
    }
    let position = (index / pairs) as f32;
    let inverse = unsafe { *inverse.add((index % pairs) as usize) };
    let phase = float::mul_rn_f32(position, inverse);
    unsafe {
        *cosine.add(index as usize) =
            f32_from_bf16(bf16_from_f32(float::mul_rn_f32(libm::cosf(phase), scale)));
        *sine.add(index as usize) =
            f32_from_bf16(bf16_from_f32(float::mul_rn_f32(libm::sinf(phase), scale)));
    }
}
