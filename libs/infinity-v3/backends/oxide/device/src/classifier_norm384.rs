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
// Port of the width-384 (dispatched as Width512, non-even columns) geometry and
// arithmetic in backends/libtorch/cpp/vendor/candle_kernels/layer_norm, plus the
// embedding contract already used by the project width-768 module.
// Original contributions Copyright Infinity Core Contributors,
// Licensed under the Infinity Restricted MIT License 1.0; see the repository LICENSE.
//! Exact width-384 BF16 affine LayerNorm and fused word/type/position embedding
//! LayerNorm for the native encoder recipe.
//!
//! The native dispatch rounds 384 columns to a Width512 kernel with four rows
//! per CTA, one 32-lane warp per row, and two 8-element BF16 vectors per lane.
//! Arithmetic and statistics still use the actual 384 columns: the tail vectors
//! for lanes 16..31 in the second iteration are neither loaded, summed, nor
//! stored. One 32-lane warp normalizes each row; lane `lane` holds values at
//! columns `8 * lane + j + 256 * iteration`. Statistics use the native
//! sequential per-lane sum, an XOR reduction over 1, 2, 4, 8, 16, the actual
//! 384-column reciprocal, fused centered variance, `rsqrt.approx`, and the
//! native multiply-plus-FMA affine epilogue.

use cuda_device::{device, float, thread, warp};

use crate::{bf16_from_f32, f32_from_bf16, read, write};

const WIDTH: usize = 384;
const VALUES: usize = 16;
const LANE_VALUES: usize = 8;
const ITERATION_STRIDE: usize = 256;
const VEC_COLS_PER_LDG: usize = 32;
const ROWS_PER_CTA: usize = 4;
const INVERSE_COLS: f32 = 1.0 / WIDTH as f32;

#[inline(always)]
fn round_bf16(value: f32) -> f32 {
    f32_from_bf16(bf16_from_f32(value))
}

#[inline(always)]
fn subtract_ftz(left: f32, right: f32) -> f32 {
    float::add_rn_ftz_f32(left, -right)
}

/// Native lane-local butterfly reduction, in the upstream 1,2,4,8,16 order.
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

/// Native `num_valid_ldgs * NUM_ELTS` for the non-even-column tail. The actual
/// width is 384, so lanes 0..15 load two vectors and lanes 16..31 load one.
#[inline(always)]
fn valid_values(lane: usize) -> usize {
    let vectors = WIDTH / LANE_VALUES;
    ((vectors - 1 - lane + VEC_COLS_PER_LDG) / VEC_COLS_PER_LDG) * LANE_VALUES
}

/// Native embedding loader: round word+type to BF16, multiply by the unit row
/// scale (which flushes subnormals like the fast-math native kernel), then add
/// the unrounded FP32 position value.
#[inline(always)]
unsafe fn embedding_value(
    ids: *const u32,
    positions: *const u32,
    types: *const u32,
    word: *const u16,
    position: *const u16,
    token_type: *const u16,
    row: usize,
    column: usize,
) -> f32 {
    let token = unsafe { *ids.add(row) } as usize;
    let position_row = unsafe { *positions.add(row) } as usize;
    let type_row = unsafe { *types.add(row) } as usize;
    let pair = float::add_rn_ftz_f32(unsafe { read(word, token * WIDTH + column) }, unsafe {
        read(token_type, type_row * WIDTH + column)
    });
    let rounded = float::mul_rn_ftz_f32(round_bf16(pair), 1.0);
    float::add_rn_ftz_f32(rounded, unsafe {
        read(position, position_row * WIDTH + column)
    })
}

/// Native residual loader for already-BF16 projection output. An optional
/// non-native projection bias keeps the existing rounded-add contract first.
/// Residual addition remains FP32 for statistics; the unit row scale matches
/// the native kernel's fast-math multiply.
#[inline(always)]
unsafe fn residual_value(
    input: *const u16,
    residual: *const u16,
    bias: *const u16,
    index: usize,
    column: usize,
) -> f32 {
    let mut value = unsafe { read(input, index) };
    if !bias.is_null() {
        value = round_bf16(float::add_rn_f32(value, unsafe { read(bias, column) }));
    }
    let value = float::mul_rn_ftz_f32(value, 1.0);
    if residual.is_null() {
        value
    } else {
        float::add_rn_ftz_f32(value, unsafe { read(residual, index) })
    }
}

#[device]
unsafe fn normalize(
    values: [f32; VALUES],
    valid: usize,
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
        if index < valid {
            sum = float::add_rn_ftz_f32(sum, values[index]);
        }
        index += 1;
    }
    let mean = float::mul_rn_ftz_f32(reduce_sum_lane(sum), INVERSE_COLS);
    let mut variance = 0.0;
    index = 0;
    #[unroll]
    while index < VALUES {
        if index < valid {
            let centered = subtract_ftz(values[index], mean);
            variance = float::fma_rn_ftz_f32(centered, centered, variance);
        }
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
        if index < valid {
            let column = column(lane, index);
            let centered = subtract_ftz(values[index], mean);
            let normalized = float::mul_rn_ftz_f32(centered, inverse);
            let value = float::fma_rn_ftz_f32(normalized, unsafe { read(gamma, column) }, unsafe {
                read(beta, column)
            });
            unsafe { write(out, offset + column, value) };
        }
        index += 1;
    }
}

#[device]
pub(crate) unsafe fn embedding_norm_384_exact(
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
    let lane = tid & 31;
    let row = thread::blockIdx_x() as usize * ROWS_PER_CTA + tid / 32;
    if row >= tokens as usize {
        return;
    }
    let valid = valid_values(lane);
    let mut values = [0.0f32; VALUES];
    let mut index = 0;
    #[unroll]
    while index < VALUES {
        if index < valid {
            values[index] = unsafe {
                embedding_value(
                    ids,
                    positions,
                    types,
                    word,
                    position,
                    token_type,
                    row,
                    column(lane, index),
                )
            };
        }
        index += 1;
    }
    unsafe { normalize(values, valid, row * WIDTH, gamma, beta, out, eps) };
}

#[device]
pub(crate) unsafe fn layer_norm_384_exact(
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
    let lane = tid & 31;
    let row = thread::blockIdx_x() as usize * ROWS_PER_CTA + tid / 32;
    if row >= rows as usize {
        return;
    }
    let offset = row * WIDTH;
    let valid = valid_values(lane);
    let mut values = [0.0f32; VALUES];
    let mut index = 0;
    #[unroll]
    while index < VALUES {
        if index < valid {
            let column = column(lane, index);
            values[index] =
                unsafe { residual_value(input, residual, bias, offset + column, column) };
        }
        index += 1;
    }
    unsafe { normalize(values, valid, offset, gamma, beta, out, eps) };
}
