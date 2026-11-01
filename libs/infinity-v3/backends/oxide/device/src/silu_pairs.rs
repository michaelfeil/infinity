// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! BF16 paired rounded SiLU matching native DenseMlp activation code5.
//!
//! Each thread owns two adjacent columns in one row. Even pairs are read as
//! two 32-bit gate/up words and written as one 32-bit result word. The final
//! odd column, or a misaligned pointer, takes a scalar path without reading or
//! writing past the validated row width.
//! The native division specialization independently expresses CUDA's BF16
//! division recipe: non-FTZ approximate F32 division with a large-denominator
//! rescale, then BF16 rounding. The retained specialization uses RN division.

use cuda_device::{bf16x2, convert, float, ptx_asm, thread};

use crate::{bf16_from_f32, f32_from_bf16};

const LOG2E_UP: f32 = f32::from_bits(0x3fb8_aa3c);
const ONE_PACKED: u32 = 0x3f80_3f80;

#[inline(always)]
fn rounded(value: f32) -> f32 {
    f32_from_bf16(bf16_from_f32(value))
}

#[inline(always)]
fn aligned_pair(pointer: *const u16) -> bool {
    (pointer as usize) & 3 == 0
}

#[inline(always)]
unsafe fn load_word(pointer: *const u16) -> u32 {
    let address = pointer as usize as u64;
    let value: u32;
    unsafe {
        ptx_asm!(
            "ld.global.u32 %0, [%1];",
            out("=r") value,
            in("l") address,
            clobber("memory"),
        );
    }
    value
}

#[inline(always)]
unsafe fn store_word(pointer: *mut u16, value: u32) {
    let address = pointer as usize as u64;
    unsafe {
        ptx_asm!(
            "st.global.u32 [%0], %1;",
            in("l") address,
            in("r") value,
            clobber("memory"),
        );
    }
}

#[inline(always)]
unsafe fn load_pair(pointer: *const u16, column: usize, width: usize) -> (u16, u16, bool) {
    let first = unsafe { pointer.add(column) };
    let second = column + 1 < width;
    if !second {
        return (unsafe { *first }, 0, false);
    }
    if aligned_pair(first) {
        let packed = unsafe { load_word(first) };
        (packed as u16, (packed >> 16) as u16, true)
    } else {
        (unsafe { *first }, unsafe { *first.add(1) }, true)
    }
}

#[inline(always)]
unsafe fn divide<const NATIVE_DIVISION: bool>(numerator: f32, denominator: f32) -> f32 {
    if NATIVE_DIVISION {
        // BF16 hdiv rescales at |denominator| >= 2^126, including infinity.
        // Keep an ordered float comparison so NaN follows native fabs semantics.
        let magnitude = f32::from_bits(denominator.to_bits() & 0x7fff_ffff);
        let large = magnitude >= f32::from_bits(0x7e80_0000);
        let denominator = if large {
            float::mul_rn_f32(denominator, 0.25)
        } else {
            denominator
        };
        let quotient: f32;
        unsafe {
            ptx_asm!(
                "div.approx.f32 %0, %1, %2;",
                out("=f") quotient,
                in("f") numerator,
                in("f") denominator,
            );
        }
        if large {
            // Explicit IEEE FMA keeps subnormal results and negative zero.
            // Inline PTX prevents the importer folding the zero add into mul.
            let scaled: f32;
            unsafe {
                ptx_asm!(
                    "fma.rn.f32 %0, %1, 0f3E800000, 0f80000000;",
                    out("=f") scaled,
                    in("f") quotient,
                );
            }
            scaled
        } else {
            quotient
        }
    } else {
        float::div_rn_f32(numerator, denominator)
    }
}

#[inline(always)]
pub(super) unsafe fn pair_result<const NATIVE_DIVISION: bool>(
    gate: (u16, u16, bool),
    up: (u16, u16, bool),
) -> u32 {
    let first_gate = f32_from_bf16(gate.0);
    let first_exponential = rounded(float::ex2_approx_f32(float::mul_rn_f32(
        -first_gate,
        LOG2E_UP,
    )));
    let first_bits = bf16_from_f32(first_exponential) as u32;
    let mut exponential = first_bits | (first_bits << 16);
    let mut second_gate = first_gate;
    let mut up_bits = up.0 as u32 | ((up.0 as u32) << 16);
    if gate.2 && up.2 {
        second_gate = f32_from_bf16(gate.1);
        let second_exponential = rounded(float::ex2_approx_f32(float::mul_rn_f32(
            -second_gate,
            LOG2E_UP,
        )));
        exponential = first_bits | ((bf16_from_f32(second_exponential) as u32) << 16);
        up_bits = up.0 as u32 | ((up.1 as u32) << 16);
    }
    let denominator = bf16x2::add_bf16x2(ONE_PACKED, exponential);
    let (first_denominator, second_denominator) = convert::cvt_f32x2_bf16x2(denominator);
    let first_activated =
        rounded(unsafe { divide::<NATIVE_DIVISION>(first_gate, first_denominator) });
    let mut activated = bf16_from_f32(first_activated) as u32;
    if gate.2 && up.2 {
        let second_activated =
            rounded(unsafe { divide::<NATIVE_DIVISION>(second_gate, second_denominator) });
        activated |= (bf16_from_f32(second_activated) as u32) << 16;
    } else {
        activated |= activated << 16;
    }
    bf16x2::mul_bf16x2(activated, up_bits)
}

/// Rounded BF16 paired gate/up activation; grid=ceil(rows*ceil(width/2)/256).
#[inline(always)]
pub(crate) unsafe fn silu_mul_bf16_pairs<const NATIVE_DIVISION: bool>(
    input: *const u16,
    out: *mut u16,
    rows: u32,
    width: u32,
) {
    if width == 0 {
        return;
    }
    let width = width as usize;
    let pairs_per_row = width.div_ceil(2);
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    let row = index / pairs_per_row;
    if row >= rows as usize {
        return;
    }
    let column = (index % pairs_per_row) * 2;
    let row_input = unsafe { input.add(row * 2 * width) };
    let gate = unsafe { load_pair(row_input, column, width) };
    let up = unsafe { load_pair(row_input.add(width), column, width) };
    let result = unsafe { pair_result::<NATIVE_DIVISION>(gate, up) };
    let row_output = unsafe { out.add(row * width + column) };
    if gate.2 && up.2 && aligned_pair(row_output) {
        unsafe { store_word(row_output, result) };
    } else if gate.2 && up.2 {
        unsafe {
            *row_output = result as u16;
            *row_output.add(1) = (result >> 16) as u16;
        }
    } else {
        unsafe { *row_output = result as u16 };
    }
}
