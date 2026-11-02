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
// Port of the qualified width-2048 geometry and arithmetic in
// backends/libtorch/cpp/vendor/candle_kernels/layer_norm and the native
// decoder_norm.cu exact_norm<2048,4,1> instantiation.
// Original contributions Copyright Infinity Core Contributors,
// Licensed under the Infinity Restricted MIT License 1.0; see the repository LICENSE.
//! Exact width-2048 BF16 RMSNorm for the native Candle decoder recipe.
//!
//! One 32-lane warp normalizes each row; four warps (128 threads) cover four
//! rows per CTA. Lane `lane` holds 64 values at columns
//! `8 * lane + index % 8 + 256 * (index / 8)`. Statistics use the native sequential
//! per-lane sum, an XOR reduction over 1, 2, 4, 8, 16, centered variance, the
//! native `fma(m2, 2^-11, eps)` then separate `mu * mu` add, `rsqrt.approx`,
//! and a separate normalized multiply followed by the affine FMA.

use cuda_device::{device, float, thread, warp};

use crate::{read, write};

const WIDTH: usize = 2048;
const VALUES: usize = 64;
const LANE_VALUES: usize = 8;
const ITERATION_STRIDE: usize = 256;
const ROWS_PER_CTA: usize = 4;
const INVERSE_COLS: f32 = 1.0 / WIDTH as f32;

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

/// Native register loader: expand BF16 to F32, multiply by the unit row scale
/// with FTZ, then add an optional BF16 residual in unrounded F32.
#[inline(always)]
unsafe fn load_value<const ADD: bool>(
    input: *const u16,
    residual: *const u16,
    index: usize,
) -> f32 {
    let value = float::mul_rn_ftz_f32(unsafe { read(input, index) }, 1.0);
    if ADD {
        float::add_rn_ftz_f32(value, unsafe { read(residual, index) })
    } else {
        value
    }
}

#[device]
unsafe fn normalize<const ADD: bool>(
    input: *const u16,
    residual: *const u16,
    offset: usize,
    lane: usize,
    gamma: *const u16,
    out: *mut u16,
    summed: *mut u16,
    eps: f32,
) {
    let mut values = [0.0f32; VALUES];
    let mut index = 0;
    #[unroll]
    while index < VALUES {
        values[index] = unsafe { load_value::<ADD>(input, residual, offset + column(lane, index)) };
        index += 1;
    }
    let mut sum = 0.0;
    index = 0;
    #[unroll]
    while index < VALUES {
        let value = values[index];
        if ADD {
            unsafe { write(summed, offset + column(lane, index), value) };
        }
        sum = float::add_rn_ftz_f32(sum, value);
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
    let mean_squared = float::mul_rn_ftz_f32(mean, mean);
    let inverse = float::rsqrt_approx_ftz_f32(float::add_rn_ftz_f32(
        float::fma_rn_ftz_f32(reduce_sum_lane(variance), INVERSE_COLS, eps),
        mean_squared,
    ));
    index = 0;
    #[unroll]
    while index < VALUES {
        let column = column(lane, index);
        let centered = subtract_ftz(values[index], 0.0);
        let normalized = float::mul_rn_ftz_f32(inverse, centered);
        let value = float::fma_rn_ftz_f32(normalized, unsafe { read(gamma, column) }, 0.0);
        unsafe { write(out, offset + column, value) };
        index += 1;
    }
}

#[inline(always)]
unsafe fn rms_norm_2048<const ADD: bool>(
    input: *const u16,
    residual: *const u16,
    weight: *const u16,
    out: *mut u16,
    summed: *mut u16,
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
    unsafe { normalize::<ADD>(input, residual, offset, lane, weight, out, summed, eps) };
}

/// Exact width-2048 BF16 RMSNorm without a residual stream.
/// Grid=(ceil(rows/4),1,1), block=(128,1,1); one warp per row.
#[inline(always)]
pub(crate) unsafe fn exact_rms_norm_2048(
    input: *const u16,
    weight: *const u16,
    out: *mut u16,
    rows: u32,
    width: u32,
    eps: f32,
) {
    unsafe {
        rms_norm_2048::<false>(
            input,
            core::ptr::null(),
            weight,
            out,
            core::ptr::null_mut(),
            rows,
            width,
            eps,
        )
    };
}

/// Exact width-2048 BF16 residual RMSNorm with a separate rounded BF16 sum.
/// Statistics use the unrounded F32 input+residual staging; `summed` receives
/// the RNE BF16 sum and `out` the normalized affine result.
/// Grid=(ceil(rows/4),1,1), block=(128,1,1); one warp per row.
#[inline(always)]
pub(crate) unsafe fn exact_add_rms_norm_2048(
    input: *const u16,
    residual: *const u16,
    weight: *const u16,
    out: *mut u16,
    summed: *mut u16,
    rows: u32,
    width: u32,
    eps: f32,
) {
    unsafe { rms_norm_2048::<true>(input, residual, weight, out, summed, rows, width, eps) };
}
