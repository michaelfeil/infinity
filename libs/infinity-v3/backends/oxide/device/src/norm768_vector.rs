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
// Vector-memory adaptation of the qualified width-768 geometry and arithmetic
// in backends/libtorch/cpp/vendor/candle_kernels/layer_norm.
// Original contributions Copyright Infinity Core Contributors,
// Licensed under the Infinity Restricted MIT License 1.0; see the repository LICENSE.
//! Exact width-768 BF16 LayerNorm with aligned 128-bit memory transactions.
//!
//! One warp owns each row and four rows share a 128-thread CTA. Each lane keeps
//! the scalar entry's 24 values and arithmetic order, moving eight adjacent
//! BF16 values at a time. Any misaligned required or non-null optional pointer
//! selects the unchanged scalar entry, including output aliases of an input.

use cuda_device::{device, float, ptx_asm, thread, warp};

use crate::{bf16_from_f32, f32_from_bf16, norm768};

const WIDTH: usize = 768;
const VALUES: usize = 24;
const LANE_VALUES: usize = 8;
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
pub(crate) unsafe fn load_vector(pointer: *const u16) -> [u16; LANE_VALUES] {
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

#[device]
pub(crate) unsafe fn normalize_vector(
    values: [f32; VALUES],
    offset: usize,
    gamma: *const u16,
    beta: *const u16,
    out: *mut u16,
    eps: f32,
) {
    let lane = thread::threadIdx_x() as usize & 31;
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
    index = 0;
    #[unroll]
    while index < VALUES {
        let source = column(lane, index);
        let gamma_values = unsafe { load_vector(gamma.add(source)) };
        let beta_values = unsafe { load_vector(beta.add(source)) };
        let mut result = [0u16; LANE_VALUES];
        let mut element = 0;
        #[unroll]
        while element < LANE_VALUES {
            let centered = subtract_ftz(values[index + element], mean);
            let normalized = float::mul_rn_ftz_f32(centered, inverse);
            let value = float::fma_rn_ftz_f32(
                normalized,
                f32_from_bf16(gamma_values[element]),
                f32_from_bf16(beta_values[element]),
            );
            result[element] = bf16_from_f32(value);
            element += 1;
        }
        unsafe { store_vector(out.add(offset + source), result) };
        index += LANE_VALUES;
    }
}

#[device]
unsafe fn embedding_values(
    word: *const u16,
    position: *const u16,
    token_type: *const u16,
    token: usize,
    position_row: usize,
    type_row: usize,
    lane: usize,
) -> [f32; VALUES] {
    let mut values = [0.0f32; VALUES];
    let mut index = 0;
    #[unroll]
    while index < VALUES {
        let source = column(lane, index);
        let word_values = unsafe { load_vector(word.add(token * WIDTH + source)) };
        let position_values = unsafe { load_vector(position.add(position_row * WIDTH + source)) };
        let type_values = unsafe { load_vector(token_type.add(type_row * WIDTH + source)) };
        let mut element = 0;
        #[unroll]
        while element < LANE_VALUES {
            let pair = float::add_rn_ftz_f32(
                f32_from_bf16(word_values[element]),
                f32_from_bf16(type_values[element]),
            );
            let rounded = float::mul_rn_ftz_f32(round_bf16(pair), 1.0);
            values[index + element] =
                float::add_rn_ftz_f32(rounded, f32_from_bf16(position_values[element]));
            element += 1;
        }
        index += LANE_VALUES;
    }
    values
}

#[device]
unsafe fn residual_values(
    input: *const u16,
    residual: *const u16,
    bias: *const u16,
    offset: usize,
    lane: usize,
) -> [f32; VALUES] {
    let mut values = [0.0f32; VALUES];
    let mut index = 0;
    #[unroll]
    while index < VALUES {
        let source = column(lane, index);
        let input_values = unsafe { load_vector(input.add(offset + source)) };
        let residual_values = if residual.is_null() {
            [0u16; LANE_VALUES]
        } else {
            unsafe { load_vector(residual.add(offset + source)) }
        };
        let bias_values = if bias.is_null() {
            [0u16; LANE_VALUES]
        } else {
            unsafe { load_vector(bias.add(source)) }
        };
        let mut element = 0;
        #[unroll]
        while element < LANE_VALUES {
            let mut value = f32_from_bf16(input_values[element]);
            if !bias.is_null() {
                value = round_bf16(float::add_rn_f32(
                    value,
                    f32_from_bf16(bias_values[element]),
                ));
            }
            let value = float::mul_rn_ftz_f32(value, 1.0);
            values[index + element] = if residual.is_null() {
                value
            } else {
                float::add_rn_ftz_f32(value, f32_from_bf16(residual_values[element]))
            };
            element += 1;
        }
        index += LANE_VALUES;
    }
    values
}

#[inline(always)]
pub(crate) unsafe fn embedding_norm_768_vector(
    ids: *const u32,
    positions: *const u32,
    types: *const u32,
    word: *const u16,
    position: *const u16,
    token_type: *const u16,
    gamma: *const u16,
    beta: *const u16,
    out: *mut u16,
    tokens: u32,
    hidden: u32,
    eps: f32,
) {
    if hidden as usize != WIDTH {
        return;
    }
    let tid = thread::threadIdx_x() as usize;
    let row = thread::blockIdx_x() as usize * ROWS_PER_CTA + tid / 32;
    if row >= tokens as usize {
        return;
    }
    if !(aligned(word)
        && aligned(position)
        && aligned(token_type)
        && aligned(gamma)
        && aligned(beta)
        && aligned(out))
    {
        unsafe {
            norm768::embedding_norm_768(
                ids, positions, types, word, position, token_type, gamma, beta, out, tokens,
                hidden, eps,
            )
        };
        return;
    }
    let values = unsafe {
        embedding_values(
            word,
            position,
            token_type,
            *ids.add(row) as usize,
            *positions.add(row) as usize,
            *types.add(row) as usize,
            tid & 31,
        )
    };
    unsafe { normalize_vector(values, row * WIDTH, gamma, beta, out, eps) };
}

#[inline(always)]
pub(crate) unsafe fn layer_norm_768_vector(
    input: *const u16,
    residual: *const u16,
    bias: *const u16,
    gamma: *const u16,
    beta: *const u16,
    out: *mut u16,
    rows: u32,
    width: u32,
    eps: f32,
) {
    if width as usize != WIDTH {
        return;
    }
    let tid = thread::threadIdx_x() as usize;
    let row = thread::blockIdx_x() as usize * ROWS_PER_CTA + tid / 32;
    if row >= rows as usize {
        return;
    }
    if !(aligned(input)
        && (residual.is_null() || aligned(residual))
        && (bias.is_null() || aligned(bias))
        && aligned(gamma)
        && aligned(beta)
        && aligned(out))
    {
        unsafe {
            norm768::layer_norm_768(input, residual, bias, gamma, beta, out, rows, width, eps)
        };
        return;
    }
    let offset = row * WIDTH;
    let values = unsafe { residual_values(input, residual, bias, offset, tid & 31) };
    unsafe { normalize_vector(values, offset, gamma, beta, out, eps) };
}
