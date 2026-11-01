// SPDX-License-Identifier: MIT AND LicenseRef-Infinity-Restricted-MIT-1.0
// Upstream kernel portions retain MIT; new original port contributions use
// Infinity Restricted MIT License 1.0. No inherited grant is changed.
// Fused Q/K RMSNorm and NeoX RoPE, retaining infinity_v3's intermediate rounding.
// Ported from the published tracked source
// libs/infinity-v3/backends/libtorch/cpp/vendor/candle_kernels/kernels/qk_norm_rope.cu
// (repository michaelfeil/infinity, commit fe6ddbb0443f5823f325fffb17cb29cd675a404e,
// SPDX MIT), adapted to the intended in-place 14-argument ABI with F32 tables.
// Original contributions Copyright Infinity Core Contributors,
// Licensed under the Infinity Restricted MIT License 1.0; see the repository LICENSE.
//! Fused in-place Q/K RMSNorm plus full NeoX rotary over head width 128.
//!
//! Each 16-lane group normalizes one 128-element head with eight adjacent BF16
//! values per lane. The first `tokens * qheads` logical rows cover Q and the
//! remaining `tokens * kvheads` rows cover K, each using its own token stride.
//! Statistics preserve the native sequential lane sum, the 16-lane XOR 1/2/4/8
//! reduction (segment clamp 0x101f, half-warp member mask), the separate
//! rounded mean square, and `rsqrt.approx`. Affine values round to BF16 before
//! the rotary products. The native SASS rounds the sine products and fuses each
//! cosine product into an `HFMA2.BF16_V2`; this port uses the matching packed
//! BF16 multiply/fused-multiply-add sequence. V storage is never touched.

use cuda_device::{device, float, ptx_asm, thread};

use crate::{bf16_from_f32, f32_from_bf16};

const HEAD_WIDTH: usize = 128;
const HALF: usize = HEAD_WIDTH / 2;
const VALUES: usize = 8;
const GROUPS_PER_CTA: usize = 8;
const INVERSE_COLS: f32 = 1.0 / HEAD_WIDTH as f32;
const VECTOR_BYTES: usize = VALUES * core::mem::size_of::<u16>();

#[inline(always)]
fn rounded(value: f32) -> f32 {
    f32_from_bf16(bf16_from_f32(value))
}

#[inline(always)]
fn bf16_mul(lhs: u16, rhs: u16) -> u16 {
    let result: u16;
    unsafe {
        ptx_asm!(
            "mul.bf16 %0, %1, %2;",
            out("=h") result,
            in("h") lhs,
            in("h") rhs,
        );
    }
    result
}

#[inline(always)]
fn bf16_fma(first: u16, second: u16, addend: u16) -> u16 {
    let result: u16;
    unsafe {
        ptx_asm!(
            "fma.rn.bf16 %0, %1, %2, %3;",
            out("=h") result,
            in("h") first,
            in("h") second,
            in("h") addend,
        );
    }
    result
}

#[inline(always)]
fn aligned(pointer: *const u16) -> bool {
    (pointer as usize) & (VECTOR_BYTES - 1) == 0
}

#[inline(always)]
unsafe fn load_vector(pointer: *const u16) -> [u16; VALUES] {
    let address = pointer as usize as u64;
    let first: u32;
    let second: u32;
    let third: u32;
    let fourth: u32;
    unsafe {
        ptx_asm!(
            "ld.global.v4.u32 {%0, %1, %2, %3}, [%4];",
            out("=r") first,
            out("=r") second,
            out("=r") third,
            out("=r") fourth,
            in("l") address,
            clobber("memory"),
        );
    }
    [
        first as u16,
        (first >> 16) as u16,
        second as u16,
        (second >> 16) as u16,
        third as u16,
        (third >> 16) as u16,
        fourth as u16,
        (fourth >> 16) as u16,
    ]
}

#[inline(always)]
unsafe fn store_vector(pointer: *mut u16, values: [u16; VALUES]) {
    let address = pointer as usize as u64;
    let first = values[0] as u32 | ((values[1] as u32) << 16);
    let second = values[2] as u32 | ((values[3] as u32) << 16);
    let third = values[4] as u32 | ((values[5] as u32) << 16);
    let fourth = values[6] as u32 | ((values[7] as u32) << 16);
    unsafe {
        ptx_asm!(
            "st.global.v4.u32 [%0], {%1, %2, %3, %4};",
            in("l") address,
            in("r") first,
            in("r") second,
            in("r") third,
            in("r") fourth,
            clobber("memory"),
        );
    }
}

/// Native `__shfl_xor_sync(mask, value, delta, 16)`: width-16 segment clamp
/// `0x101f` (derived from nvcc reference PTX) with a 16-lane member mask.
#[inline(always)]
fn shuffle_xor_16(value: f32, delta: u32, mask: u32) -> f32 {
    let source = value.to_bits();
    let result: u32;
    unsafe {
        ptx_asm!(
            "shfl.sync.bfly.b32 %0, %1, %2, 0x101f, %3;",
            out("=r") result,
            in("r") source,
            in("r") delta,
            in("r") mask,
        );
    }
    f32::from_bits(result)
}

#[inline(always)]
fn reduce_16(value: f32, mask: u32) -> f32 {
    let value = float::add_rn_f32(value, shuffle_xor_16(value, 1, mask));
    let value = float::add_rn_f32(value, shuffle_xor_16(value, 2, mask));
    let value = float::add_rn_f32(value, shuffle_xor_16(value, 4, mask));
    float::add_rn_f32(value, shuffle_xor_16(value, 8, mask))
}

#[inline(always)]
unsafe fn load_values<const VECTOR: bool>(pointer: *const u16) -> [f32; VALUES] {
    let mut values = [0.0f32; VALUES];
    let mut index = 0;
    if VECTOR {
        let raw = unsafe { load_vector(pointer) };
        while index < VALUES {
            values[index] = f32_from_bf16(raw[index]);
            index += 1;
        }
    } else {
        while index < VALUES {
            values[index] = f32_from_bf16(unsafe { *pointer.add(index) });
            index += 1;
        }
    }
    values
}

#[device]
unsafe fn norm_rope_head<const VECTOR: bool>(
    data: *mut u16,
    weight: *const u16,
    cosine: *const f32,
    sine: *const f32,
    offset: usize,
    lane: usize,
    mask: u32,
    table_row: usize,
    table_stride: usize,
    eps: f32,
) {
    let source = offset + lane * VALUES;
    let input_values = unsafe { load_values::<VECTOR>(data.add(source)) };
    let gamma_values = unsafe { load_values::<VECTOR>(weight.add(lane * VALUES)) };
    let mut mean = 0.0;
    let mut index = 0;
    #[unroll]
    while index < VALUES {
        mean = float::add_rn_f32(mean, input_values[index]);
        index += 1;
    }
    let mut mean = reduce_16(mean, mask);
    mean = float::add_rn_f32(mean, 0.0);
    let mean = float::mul_rn_f32(mean, INVERSE_COLS);
    let mut variance = 0.0;
    index = 0;
    #[unroll]
    while index < VALUES {
        let delta = float::add_rn_f32(input_values[index], -mean);
        variance = float::fma_rn_f32(delta, delta, variance);
        index += 1;
    }
    let mut variance = reduce_16(variance, mask);
    variance = float::add_rn_f32(variance, 0.0);
    let mean_squared = float::mul_rn_f32(mean, mean);
    let inverse = float::rsqrt_approx_f32(float::add_rn_f32(
        float::fma_rn_f32(variance, INVERSE_COLS, eps),
        mean_squared,
    ));
    let mut normalized = [0.0f32; VALUES];
    index = 0;
    #[unroll]
    while index < VALUES {
        let centered = float::add_rn_f32(input_values[index], -0.0);
        let scaled = float::mul_rn_f32(inverse, centered);
        let affine = float::add_rn_f32(float::mul_rn_f32(gamma_values[index], scaled), 0.0);
        normalized[index] = rounded(affine);
        index += 1;
    }
    let mut cos_values = [0.0f32; VALUES];
    let mut sin_values = [0.0f32; VALUES];
    if lane < 8 {
        index = 0;
        while index < VALUES {
            let phase = table_row * table_stride + lane * VALUES + index;
            cos_values[index] = rounded(unsafe { *cosine.add(phase) });
            sin_values[index] = rounded(unsafe { *sine.add(phase) });
            index += 1;
        }
    }
    let mut first = [0u16; VALUES];
    let mut second = [0u16; VALUES];
    index = 0;
    #[unroll]
    while index < VALUES {
        let peer = shuffle_xor_16(normalized[index], 8, mask);
        if lane < 8 {
            let value = bf16_from_f32(normalized[index]);
            let peer = bf16_from_f32(peer);
            let cos = bf16_from_f32(cos_values[index]);
            let sin = bf16_from_f32(sin_values[index]);
            let peer_sine = bf16_mul(peer, sin);
            let value_sine = bf16_mul(value, sin);
            first[index] = bf16_fma(value, cos, peer_sine ^ 0x8000);
            second[index] = bf16_fma(peer, cos, value_sine);
        }
        index += 1;
    }
    if lane < 8 {
        if VECTOR {
            unsafe { store_vector(data.add(source), first) };
            unsafe { store_vector(data.add(offset + HALF + lane * VALUES), second) };
        } else {
            index = 0;
            while index < VALUES {
                unsafe { *data.add(source + index) = first[index] };
                unsafe { *data.add(offset + HALF + lane * VALUES + index) = second[index] };
                index += 1;
            }
        }
    }
}

/// In-place Q/K RMSNorm and full NeoX rotary with the agreed 14-argument ABI.
/// Grid=(ceil(tokens*(qheads+kvheads)/8),1,1), block=(128,1,1).
#[inline(always)]
pub(crate) unsafe fn qk_norm_rope_128(
    q: *mut u16,
    k: *mut u16,
    qgamma: *const u16,
    kgamma: *const u16,
    cosine: *const f32,
    sine: *const f32,
    positions: *const u32,
    tokens: u32,
    qheads: u32,
    kvheads: u32,
    qstride: u64,
    kstride: u64,
    table_stride: u32,
    eps: f32,
) {
    let tid = thread::threadIdx_x() as usize;
    let lane = tid % 16;
    let group = tid / 16;
    if group >= GROUPS_PER_CTA {
        return;
    }
    let qrows = tokens as u64 * qheads as u64;
    let total_rows = qrows + tokens as u64 * kvheads as u64;
    let row = thread::blockIdx_x() as u64 * GROUPS_PER_CTA as u64 + group as u64;
    if row >= total_rows {
        return;
    }
    let is_q = row < qrows;
    let heads = (if is_q { qheads } else { kvheads }) as u64;
    if heads == 0 {
        return;
    }
    let r = if is_q { row } else { row - qrows };
    let token = (r / heads) as usize;
    if token >= tokens as usize {
        return;
    }
    let head = (r % heads) as usize;
    let (data, weight, stride) = if is_q {
        (q, qgamma, qstride)
    } else {
        (k, kgamma, kstride)
    };
    let offset = (token as u64 * stride + (head * HEAD_WIDTH) as u64) as usize;
    let table_row = unsafe { *positions.add(token) } as usize;
    let mask = if (tid & 31) < 16 {
        0x0000_ffffu32
    } else {
        0xffff_0000u32
    };
    let vector = aligned(unsafe { data.add(offset) } as *const u16) && aligned(weight);
    if vector {
        unsafe {
            norm_rope_head::<true>(
                data,
                weight,
                cosine,
                sine,
                offset,
                lane,
                mask,
                table_row,
                table_stride as usize,
                eps,
            )
        };
    } else {
        unsafe {
            norm_rope_head::<false>(
                data,
                weight,
                cosine,
                sine,
                offset,
                lane,
                mask,
                table_row,
                table_stride as usize,
                eps,
            )
        };
    }
}
