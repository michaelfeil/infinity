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
// The Candle-derived embedding geometry/arithmetic remains under the BSD-3-Clause
// notice above. The rounded residual recipe is derived from fast_kernels.cu,
// SPDX-License-Identifier: Apache-2.0.
// Original contributions Copyright Infinity Core Contributors,
// Licensed under the Infinity Restricted MIT License 1.0; see the repository LICENSE.
//! Two distinct width-512 BF16 encoder normalization primitives.
//!
//! `embedding_word_type_norm_512` follows the native Candle biased LayerNorm
//! reached by `encoder_layer_norm_only(word, gamma, eps, type, beta, false)`:
//! one warp per row, four rows per CTA, and unrounded FP32 word-plus-type values
//! through the FTZ statistics and FMA affine epilogue.
//!
//! `layer_norm_512_rounded` follows the native Jina residual recipe
//! `fused_add_layer_norm` / `add_norm_register_kernel<T, 2>`: one 256-thread CTA
//! per row, two values per thread, BF16 rounding after `x + residual`, precise
//! mean/variance divides, separate epsilon add, and a non-FTZ approximate rsqrt.

use cuda_device::{float, thread, warp, SharedArray};

use crate::{bf16_from_f32, block_sum, f32_from_bf16, norm768_vector, read, write};

const WIDTH: usize = 512;
const INVERSE_COLS: f32 = 1.0 / WIDTH as f32;
const LANE_VALUES: usize = 8;
const ITERATION_STRIDE: usize = 256;
const EMBED_VALUES: usize = 16;
const ROWS_PER_CTA: usize = 4;
const THREADS: usize = 256;
const ROUNDED_VALUES: usize = WIDTH / THREADS;

#[inline(always)]
fn round_bf16(value: f32) -> f32 {
    f32_from_bf16(bf16_from_f32(value))
}

#[inline(always)]
fn subtract_ftz(left: f32, right: f32) -> f32 {
    float::add_rn_ftz_f32(left, -right)
}

/// Candle lane-local butterfly reduction in the upstream 1,2,4,8,16 order.
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
    (pointer as usize) & (LANE_VALUES * core::mem::size_of::<u16>() - 1) == 0
}

/// Candle's unrounded word-plus-type loader. The unit row scale preserves the
/// native FP32 multiply and its subnormal flushing before the type addition.
#[inline(always)]
unsafe fn embedding_value(
    word: *const u16,
    type_table: *const u16,
    token: usize,
    type_row: usize,
    column: usize,
) -> f32 {
    let word_value = unsafe { read(word, token * WIDTH + column) };
    let type_value = unsafe { read(type_table, type_row * WIDTH + column) };
    float::add_rn_ftz_f32(float::mul_rn_ftz_f32(word_value, 1.0), type_value)
}

#[inline(always)]
unsafe fn embedding_values(
    word: *const u16,
    type_table: *const u16,
    token: usize,
    type_row: usize,
    lane: usize,
) -> [f32; EMBED_VALUES] {
    let mut values = [0.0f32; EMBED_VALUES];
    if aligned(word) && aligned(type_table) {
        let mut index = 0;
        while index < EMBED_VALUES {
            let source = column(lane, index);
            let word_values =
                unsafe { norm768_vector::load_vector(word.add(token * WIDTH + source)) };
            let type_values =
                unsafe { norm768_vector::load_vector(type_table.add(type_row * WIDTH + source)) };
            let mut element = 0;
            while element < LANE_VALUES {
                values[index + element] = float::add_rn_ftz_f32(
                    float::mul_rn_ftz_f32(f32_from_bf16(word_values[element]), 1.0),
                    f32_from_bf16(type_values[element]),
                );
                element += 1;
            }
            index += LANE_VALUES;
        }
    } else {
        let mut index = 0;
        while index < EMBED_VALUES {
            values[index] =
                unsafe { embedding_value(word, type_table, token, type_row, column(lane, index)) };
            index += 1;
        }
    }
    values
}

/// Candle biased LayerNorm statistics and affine epilogue for width 512.
#[inline(always)]
unsafe fn normalize_embedding(
    values: [f32; EMBED_VALUES],
    offset: usize,
    gamma: *const u16,
    beta: *const u16,
    out: *mut u16,
    eps: f32,
) {
    let lane = thread::threadIdx_x() as usize & 31;
    let mut sum = 0.0;
    let mut index = 0;
    while index < EMBED_VALUES {
        sum = float::add_rn_ftz_f32(sum, values[index]);
        index += 1;
    }
    let mean = float::mul_rn_ftz_f32(reduce_sum_lane(sum), INVERSE_COLS);
    let mut variance = 0.0;
    index = 0;
    while index < EMBED_VALUES {
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
    while index < EMBED_VALUES {
        let column = column(lane, index);
        let centered = subtract_ftz(values[index], mean);
        let normalized = float::mul_rn_ftz_f32(centered, inverse);
        let value = float::fma_rn_ftz_f32(normalized, unsafe { read(gamma, column) }, unsafe {
            read(beta, column)
        });
        unsafe { write(out, offset + column, value) };
        index += 1;
    }
}

#[inline(always)]
pub(crate) unsafe fn embedding_word_type_norm_512(
    ids: *const u32,
    types: *const u32,
    word: *const u16,
    type_table: *const u16,
    gamma: *const u16,
    beta: *const u16,
    out: *mut u16,
    tokens: u32,
    eps: f32,
) {
    let tid = thread::threadIdx_x() as usize;
    let lane = tid & 31;
    let row = thread::blockIdx_x() as usize * ROWS_PER_CTA + tid / 32;
    if row >= tokens as usize {
        return;
    }
    let token = unsafe { *ids.add(row) } as usize;
    let type_row = unsafe { *types.add(row) } as usize;
    let values = unsafe { embedding_values(word, type_table, token, type_row, lane) };
    unsafe { normalize_embedding(values, row * WIDTH, gamma, beta, out, eps) };
}

/// Native rounded residual loader. The projection output already contains its
/// bias for the Jina path; a non-null optional bias retains the existing
/// rounded-add contract. Each addition is rounded to BF16 before statistics.
#[inline(always)]
unsafe fn rounded_value(
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
    if residual.is_null() {
        value
    } else {
        round_bf16(float::add_rn_f32(value, unsafe { read(residual, index) }))
    }
}

/// Native `add_norm_register_kernel` statistics and affine epilogue, width 512.
#[inline(always)]
unsafe fn normalize_rounded(
    values: [f32; ROUNDED_VALUES],
    tid: usize,
    offset: usize,
    gamma: *const u16,
    beta: *const u16,
    out: *mut u16,
    eps: f32,
) {
    static mut SCRATCH: SharedArray<f32, 8> = SharedArray::UNINIT;
    let scratch = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCRATCH) };
    let mut sum = 0.0;
    let mut index = 0;
    while index < ROUNDED_VALUES {
        sum = float::add_rn_f32(sum, values[index]);
        index += 1;
    }
    let mean = float::div_rn_f32(unsafe { block_sum(sum, scratch) }, WIDTH as f32);
    let mut variance = 0.0;
    index = 0;
    while index < ROUNDED_VALUES {
        let centered = values[index] - mean;
        variance = float::fma_rn_f32(centered, centered, variance);
        index += 1;
    }
    let variance = float::div_rn_f32(unsafe { block_sum(variance, scratch) }, WIDTH as f32);
    let inverse = float::rsqrt_approx_f32(float::add_rn_f32(variance, eps));
    index = 0;
    while index < ROUNDED_VALUES {
        let column = tid + index * THREADS;
        let centered = values[index] - mean;
        let normalized = float::mul_rn_f32(inverse, centered);
        let scaled = float::mul_rn_f32(normalized, unsafe { read(gamma, column) });
        let value = float::add_rn_f32(scaled, unsafe { read(beta, column) });
        unsafe { write(out, offset + column, value) };
        index += 1;
    }
}

#[inline(always)]
pub(crate) unsafe fn layer_norm_512_rounded(
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
    let row = thread::blockIdx_x() as usize;
    if row >= rows as usize {
        return;
    }
    let tid = thread::threadIdx_x() as usize;
    let offset = row * WIDTH;
    let mut values = [0.0f32; ROUNDED_VALUES];
    let mut index = 0;
    while index < ROUNDED_VALUES {
        let column = tid + index * THREADS;
        values[index] = unsafe { rounded_value(input, residual, bias, offset + column, column) };
        index += 1;
    }
    unsafe { normalize_rounded(values, tid, offset, gamma, beta, out, eps) };
}
