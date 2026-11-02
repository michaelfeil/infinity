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
// Port of the qualified width-768 geometry and arithmetic in
// backends/libtorch/cpp/vendor/candle_kernels/layer_norm.
// Original contributions Copyright Infinity Core Contributors,
// Licensed under the Infinity Restricted MIT License 1.0; see the repository LICENSE.
//! Native unrounded word-plus-type width-768 embedding LayerNorm.
//!
//! One warp owns each row and four rows share a 128-thread CTA. Word and type
//! table rows are added in FP32 without the legacy intermediate BF16 rounding,
//! then the exact shared biased LayerNorm statistics and affine epilogue run.
//! Aligned word/type/gamma/beta/out pointers use 128-bit vector moves; any
//! misalignment falls back to bounded scalar word/type gathers with identical
//! column mapping and arithmetic.

use cuda_device::{float, thread};

use crate::{f32_from_bf16, norm768, norm768_vector, read};

const WIDTH: usize = 768;
const VALUES: usize = 24;
const LANE_VALUES: usize = 8;
const ITERATION_STRIDE: usize = 256;
const ROWS_PER_CTA: usize = 4;
const VECTOR_BYTES: usize = LANE_VALUES * core::mem::size_of::<u16>();

#[inline(always)]
fn column(lane: usize, index: usize) -> usize {
    lane * LANE_VALUES + index % LANE_VALUES + ITERATION_STRIDE * (index / LANE_VALUES)
}

#[inline(always)]
fn aligned(pointer: *const u16) -> bool {
    (pointer as usize) & (VECTOR_BYTES - 1) == 0
}

#[inline(always)]
unsafe fn vector_values(
    word: *const u16,
    token_type: *const u16,
    token: usize,
    type_row: usize,
    lane: usize,
) -> [f32; VALUES] {
    let mut values = [0.0f32; VALUES];
    let mut index = 0;
    while index < VALUES {
        let source = column(lane, index);
        let word_values = unsafe { norm768_vector::load_vector(word.add(token * WIDTH + source)) };
        let type_values =
            unsafe { norm768_vector::load_vector(token_type.add(type_row * WIDTH + source)) };
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
    values
}

#[inline(always)]
unsafe fn scalar_values(
    word: *const u16,
    token_type: *const u16,
    token: usize,
    type_row: usize,
    lane: usize,
) -> [f32; VALUES] {
    let mut values = [0.0f32; VALUES];
    let mut index = 0;
    while index < VALUES {
        let source = column(lane, index);
        let word_value = unsafe { read(word, token * WIDTH + source) };
        let type_value = unsafe { read(token_type, type_row * WIDTH + source) };
        values[index] = float::add_rn_ftz_f32(float::mul_rn_ftz_f32(word_value, 1.0), type_value);
        index += 1;
    }
    values
}

#[inline(always)]
pub(crate) unsafe fn word_type_norm_768(
    ids: *const u32,
    types: *const u32,
    word: *const u16,
    token_type: *const u16,
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
    let offset = row * WIDTH;
    if !(aligned(word)
        && aligned(token_type)
        && aligned(gamma)
        && aligned(beta)
        && aligned(out as *const u16))
    {
        let values = unsafe { scalar_values(word, token_type, token, type_row, lane) };
        unsafe { norm768::normalize(values, offset, gamma, beta, out, eps) };
        return;
    }
    let values = unsafe { vector_values(word, token_type, token, type_row, lane) };
    unsafe { norm768_vector::normalize_vector(values, offset, gamma, beta, out, eps) };
}
