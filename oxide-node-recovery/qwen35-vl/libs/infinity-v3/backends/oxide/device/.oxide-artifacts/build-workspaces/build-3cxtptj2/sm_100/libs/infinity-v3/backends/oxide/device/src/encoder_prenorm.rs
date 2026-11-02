/*
BSD 3-Clause License

Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
// Adaptation of the qualified width-768 geometry and arithmetic in
// backends/libtorch/cpp/vendor/candle_kernels/layer_norm.
// Original contributions Copyright Infinity Core Contributors,
// Licensed under the Infinity Restricted MIT License 1.0; see the repository LICENSE.
//! Biasless width-768 BF16 encoder normalization with rounded residual updates.
//!
//! One 32-lane warp owns each row and four rows share a 128-thread CTA. Each
//! lane keeps the qualified 24 values at `8 * lane + element + 256 * iteration`.
//! Summation, XOR butterfly reduction, fused variance, `rsqrt.approx`, and the
//! gamma-only FMA epilogue retain the exact native operation order. Optional
//! residuals are added with FTZ, rounded to BF16 before statistics, and written
//! to a retained model-dtype stream only after every row value is in registers.
//! The entry points are unqualified until SM compilation and exact GPU oracles pass.

use cuda_device::{device, float, ptx_asm, thread, warp};

use crate::{bf16_from_f32, f32_from_bf16};

const WIDTH: usize = 768;
const VALUES: usize = 24;
const LANE_VALUES: usize = 8;
const ITERATIONS: usize = VALUES / LANE_VALUES;
const ITERATION_STRIDE: usize = 256;
const ROWS_PER_CTA: usize = 4;
const INVERSE_COLS: f32 = 1.0 / WIDTH as f32;
const VECTOR_BYTES: usize = LANE_VALUES * core::mem::size_of::<u16>();

#[inline(always)]
fn round_bf16(value: f32) -> f32 {
    f32_from_bf16(bf16_from_f32(value))
}

#[inline(always)]
fn subtract_ftz(left: f32, right: f32) -> f32 {
    float::add_rn_ftz_f32(left, -right)
}

#[inline(always)]
fn reduce_sum_lane(value: f32) -> f32 {
    let value = float::add_rn_ftz_f32(value, warp::shuffle_xor_f32(value, 1));
    let value = float::add_rn_ftz_f32(value, warp::shuffle_xor_f32(value, 2));
    let value = float::add_rn_ftz_f32(value, warp::shuffle_xor_f32(value, 4));
    let value = float::add_rn_ftz_f32(value, warp::shuffle_xor_f32(value, 8));
    float::add_rn_ftz_f32(value, warp::shuffle_xor_f32(value, 16))
}

#[inline(always)]
fn column(lane: usize, index: usize) -> usize {
    lane * LANE_VALUES + index % LANE_VALUES + ITERATION_STRIDE * (index / LANE_VALUES)
}

#[inline(always)]
fn aligned(pointer: *const u16) -> bool {
    (pointer as usize) & (VECTOR_BYTES - 1) == 0
}

#[inline(always)]
unsafe fn load_vector(pointer: *const u16) -> [u16; LANE_VALUES] {
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
unsafe fn store_vector(pointer: *mut u16, values: [u16; LANE_VALUES]) {
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

#[inline(always)]
unsafe fn load_block(pointer: *const u16, vector: bool) -> [u16; LANE_VALUES] {
    if vector {
        unsafe { load_vector(pointer) }
    } else {
        let mut values = [0u16; LANE_VALUES];
        let mut element = 0;
        while element < LANE_VALUES {
            values[element] = unsafe { *pointer.add(element) };
            element += 1;
        }
        values
    }
}

#[inline(always)]
unsafe fn store_block(pointer: *mut u16, values: [u16; LANE_VALUES], vector: bool) {
    if vector {
        unsafe { store_vector(pointer, values) };
    } else {
        let mut element = 0;
        while element < LANE_VALUES {
            unsafe { *pointer.add(element) = values[element] };
            element += 1;
        }
    }
}

#[device]
unsafe fn modern_residual_values(
    input: *const u16,
    residual: *const u16,
    offset: usize,
    lane: usize,
    vector: bool,
) -> [f32; VALUES] {
    let mut values = [0.0f32; VALUES];
    let mut iteration = 0;
    #[unroll]
    while iteration < ITERATIONS {
        let source = column(lane, iteration * LANE_VALUES);
        let input_values = unsafe { load_block(input.add(offset + source), vector) };
        let modern_residual_values = if residual.is_null() {
            [0u16; LANE_VALUES]
        } else {
            unsafe { load_block(residual.add(offset + source), vector) }
        };
        let mut element = 0;
        #[unroll]
        while element < LANE_VALUES {
            let value = float::mul_rn_ftz_f32(f32_from_bf16(input_values[element]), 1.0);
            values[iteration * LANE_VALUES + element] = if residual.is_null() {
                value
            } else {
                round_bf16(float::add_rn_ftz_f32(
                    value,
                    f32_from_bf16(modern_residual_values[element]),
                ))
            };
            element += 1;
        }
        iteration += 1;
    }
    values
}

#[device]
unsafe fn modern_word_values(
    ids: *const u32,
    word: *const u16,
    row: usize,
    lane: usize,
    vector: bool,
) -> [f32; VALUES] {
    let token = unsafe { *ids.add(row) } as usize;
    let mut values = [0.0f32; VALUES];
    let mut iteration = 0;
    #[unroll]
    while iteration < ITERATIONS {
        let source = column(lane, iteration * LANE_VALUES);
        let modern_word_values = unsafe { load_block(word.add(token * WIDTH + source), vector) };
        let mut element = 0;
        #[unroll]
        while element < LANE_VALUES {
            values[iteration * LANE_VALUES + element] =
                float::mul_rn_ftz_f32(f32_from_bf16(modern_word_values[element]), 1.0);
            element += 1;
        }
        iteration += 1;
    }
    values
}

#[device]
unsafe fn modern_store_summed(
    summed: *mut u16,
    values: [f32; VALUES],
    offset: usize,
    lane: usize,
    vector: bool,
) {
    if summed.is_null() {
        return;
    }
    let mut iteration = 0;
    #[unroll]
    while iteration < ITERATIONS {
        let source = column(lane, iteration * LANE_VALUES);
        let mut block = [0u16; LANE_VALUES];
        let mut element = 0;
        #[unroll]
        while element < LANE_VALUES {
            block[element] = bf16_from_f32(values[iteration * LANE_VALUES + element]);
            element += 1;
        }
        unsafe { store_block(summed.add(offset + source), block, vector) };
        iteration += 1;
    }
}

#[device]
unsafe fn modern_normalize_values(
    values: [f32; VALUES],
    offset: usize,
    gamma: *const u16,
    out: *mut u16,
    eps: f32,
    lane: usize,
    vector: bool,
) {
    let mut sum = 0.0;
    let mut index = 0;
    #[unroll]
    while index < VALUES {
        sum = float::add_rn_ftz_f32(sum, values[index]);
        index += 1;
    }
    let mean = float::mul_rn_ftz_f32(reduce_sum_lane(sum), INVERSE_COLS);
    let mut variance = 0.0;
    index = 0;
    #[unroll]
    while index < VALUES {
        let centered = subtract_ftz(values[index], mean);
        variance = float::fma_rn_ftz_f32(centered, centered, variance);
        index += 1;
    }
    let inverse = float::rsqrt_approx_ftz_f32(float::fma_rn_ftz_f32(
        reduce_sum_lane(variance),
        INVERSE_COLS,
        eps,
    ));
    let mut iteration = 0;
    #[unroll]
    while iteration < ITERATIONS {
        let source = column(lane, iteration * LANE_VALUES);
        let gamma_values = unsafe { load_block(gamma.add(source), vector) };
        let mut result = [0u16; LANE_VALUES];
        let mut element = 0;
        #[unroll]
        while element < LANE_VALUES {
            let centered = subtract_ftz(values[iteration * LANE_VALUES + element], mean);
            let normalized = float::mul_rn_ftz_f32(centered, inverse);
            let value =
                float::fma_rn_ftz_f32(normalized, f32_from_bf16(gamma_values[element]), 0.0);
            result[element] = bf16_from_f32(value);
            element += 1;
        }
        unsafe { store_block(out.add(offset + source), result, vector) };
        iteration += 1;
    }
}

#[inline(always)]
pub(crate) unsafe fn embedding_word_norm_768(
    ids: *const u32,
    word: *const u16,
    gamma: *const u16,
    out: *mut u16,
    rows: u32,
    eps: f32,
) {
    let tid = thread::threadIdx_x() as usize;
    let lane = tid & 31;
    let row = thread::blockIdx_x() as usize * ROWS_PER_CTA + tid / 32;
    if row >= rows as usize {
        return;
    }
    let vector = aligned(word) && aligned(gamma) && aligned(out as *const u16);
    let values = unsafe { modern_word_values(ids, word, row, lane, vector) };
    unsafe { modern_normalize_values(values, row * WIDTH, gamma, out, eps, lane, vector) };
}

#[inline(always)]
pub(crate) unsafe fn rounded_residual_norm_768(
    input: *const u16,
    residual: *const u16,
    gamma: *const u16,
    normalized: *mut u16,
    summed: *mut u16,
    rows: u32,
    eps: f32,
) {
    let tid = thread::threadIdx_x() as usize;
    let lane = tid & 31;
    let row = thread::blockIdx_x() as usize * ROWS_PER_CTA + tid / 32;
    if row >= rows as usize {
        return;
    }
    let vector = aligned(input)
        && (residual.is_null() || aligned(residual))
        && aligned(gamma)
        && aligned(normalized as *const u16)
        && (summed.is_null() || aligned(summed as *const u16));
    let offset = row * WIDTH;
    let values = unsafe { modern_residual_values(input, residual, offset, lane, vector) };
    unsafe { modern_store_summed(summed, values, offset, lane, vector) };
    unsafe { modern_normalize_values(values, offset, gamma, normalized, eps, lane, vector) };
}

/// Native F32 phase -> CUDA libdevice trigonometry -> BF16 coefficients.
/// This runs only during model construction, never during request dispatch.
#[inline(always)]
pub(crate) unsafe fn base_rope_tables_bf16(
    inverse: *const f32,
    cosine: *mut f32,
    sine: *mut f32,
    positions: u32,
) {
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if index >= positions as usize * 32 {
        return;
    }
    let phase = float::mul_rn_f32((index / 32) as f32, unsafe { *inverse.add(index % 32) });
    unsafe {
        *cosine.add(index) = round_bf16(libm::cosf(phase));
        *sine.add(index) = round_bf16(libm::sinf(phase));
    }
}
