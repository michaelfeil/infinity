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
// Exact width-1024 extension of the qualified width-2048 vector geometry in
// backends/libtorch/cpp/vendor/candle_kernels/layer_norm and the native
// decoder_norm.cu exact_norm<1024,4,1> instantiation.
// Original contributions Copyright Infinity Core Contributors,
// Licensed under the Infinity Restricted MIT License 1.0; see the repository LICENSE.
//! Exact width-1024 BF16 RMSNorm with aligned 128-bit memory transactions.
//!
//! Arithmetic follows the native Candle 4x1 geometry: one 32-lane warp per row,
//! 32 values per lane in four iterations of eight adjacent BF16 values at
//! columns `8 * lane + element + 256 * iteration`. Permitted pointers move as
//! 16-byte vectors; any misaligned required or non-null optional pointer takes
//! the width-1024 scalar fallback (never the width-2048 path).

use cuda_device::{device, float, ptx_asm, thread, warp};

use crate::{bf16_from_f32, f32_from_bf16, read, write};

const WIDTH: usize = 1024;
const VALUES: usize = 32;
const LANE_VALUES: usize = 8;
const ITERATION_STRIDE: usize = 256;
const ROWS_PER_CTA: usize = 4;
const INVERSE_COLS: f32 = 1.0 / WIDTH as f32;
const VECTOR_BYTES: usize = LANE_VALUES * core::mem::size_of::<u16>();

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

#[inline(always)]
fn aligned(pointer: *const u16) -> bool {
    (pointer as usize) & (VECTOR_BYTES - 1) == 0
}

#[inline(always)]
fn aligned_mut(pointer: *mut u16) -> bool {
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

/// Native register loader for the width-1024 scalar fallback: expand BF16 to
/// F32, multiply by the unit row scale with FTZ, then add an optional BF16
/// residual in unrounded F32.
#[inline(always)]
unsafe fn scalar_value<const ADD: bool>(
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
unsafe fn normalize_scalar_1024<const ADD: bool>(
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
        values[index] =
            unsafe { scalar_value::<ADD>(input, residual, offset + column(lane, index)) };
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

#[device]
unsafe fn normalize_vector_1024<const ADD: bool>(
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
        let source = column(lane, index);
        let input_values = unsafe { load_vector(input.add(offset + source)) };
        let residual_values = if ADD {
            unsafe { load_vector(residual.add(offset + source)) }
        } else {
            [0u16; LANE_VALUES]
        };
        let mut element = 0;
        #[unroll]
        while element < LANE_VALUES {
            let mut value = float::mul_rn_ftz_f32(f32_from_bf16(input_values[element]), 1.0);
            if ADD {
                value = float::add_rn_ftz_f32(value, f32_from_bf16(residual_values[element]));
            }
            values[index + element] = value;
            element += 1;
        }
        index += LANE_VALUES;
    }
    let mut sum = 0.0;
    index = 0;
    #[unroll]
    while index < VALUES {
        let source = column(lane, index);
        let mut stored = [0u16; LANE_VALUES];
        let mut element = 0;
        #[unroll]
        while element < LANE_VALUES {
            let value = values[index + element];
            stored[element] = bf16_from_f32(value);
            sum = float::add_rn_ftz_f32(sum, value);
            element += 1;
        }
        if ADD {
            unsafe { store_vector(summed.add(offset + source), stored) };
        }
        index += LANE_VALUES;
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
        let source = column(lane, index);
        let gamma_values = unsafe { load_vector(gamma.add(source)) };
        let mut result = [0u16; LANE_VALUES];
        let mut element = 0;
        #[unroll]
        while element < LANE_VALUES {
            let centered = subtract_ftz(values[index + element], 0.0);
            let normalized = float::mul_rn_ftz_f32(inverse, centered);
            let value =
                float::fma_rn_ftz_f32(normalized, f32_from_bf16(gamma_values[element]), 0.0);
            result[element] = bf16_from_f32(value);
            element += 1;
        }
        unsafe { store_vector(out.add(offset + source), result) };
        index += LANE_VALUES;
    }
}

#[inline(always)]
unsafe fn rms_norm_1024_vector<const ADD: bool>(
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
    let aligned_pointers = aligned(input)
        && aligned(weight)
        && aligned_mut(out)
        && (!ADD || (aligned(residual) && aligned_mut(summed)));
    if !aligned_pointers {
        unsafe {
            normalize_scalar_1024::<ADD>(input, residual, offset, lane, weight, out, summed, eps)
        };
        return;
    }
    unsafe {
        normalize_vector_1024::<ADD>(input, residual, offset, lane, weight, out, summed, eps)
    };
}

/// Exact width-1024 BF16 RMSNorm with aligned 128-bit memory transactions.
/// Grid=(ceil(rows/4),1,1), block=(128,1,1); one warp per row.
#[inline(always)]
pub(crate) unsafe fn exact_rms_norm_1024_vector(
    input: *const u16,
    weight: *const u16,
    out: *mut u16,
    rows: u32,
    width: u32,
    eps: f32,
) {
    unsafe {
        rms_norm_1024_vector::<false>(
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

/// Exact width-1024 BF16 residual RMSNorm with aligned 128-bit transactions.
/// Statistics use the unrounded F32 input+residual staging; `summed` receives
/// the RNE BF16 sum and `out` the normalized affine result.
/// Grid=(ceil(rows/4),1,1), block=(128,1,1); one warp per row.
#[inline(always)]
pub(crate) unsafe fn exact_add_rms_norm_1024_vector(
    input: *const u16,
    residual: *const u16,
    weight: *const u16,
    out: *mut u16,
    summed: *mut u16,
    rows: u32,
    width: u32,
    eps: f32,
) {
    unsafe { rms_norm_1024_vector::<true>(input, residual, weight, out, summed, rows, width, eps) };
}
