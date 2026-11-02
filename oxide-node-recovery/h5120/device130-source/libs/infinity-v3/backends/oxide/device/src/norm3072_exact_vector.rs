// SPDX-License-Identifier: Apache-2.0 AND LicenseRef-Infinity-Restricted-MIT-1.0
// Copyright Infinity Core Contributors.
//! Exact width-3072 BF16 RMSNorm for the native Candle 1x4 decoder recipe.
//!
//! One row per CTA and four warps cover 768 values each (32 lanes x 24 values).
//! Statistics use the Candle sequential per-lane sum, XOR warp reduction, the
//! Chan cross-warp merge for four 768-value groups, the native
//! `fma(m2, 1/3072, eps)` and separate `mean * mean` add, then `rsqrt.approx`
//! and the affine FMA. The 3072 registration differs from the width-2048/1024
//! four-row layout: `exact_norm<3072,1,4>` uses one row per CTA.

use cuda_device::{float, ptx_asm, thread, warp, SharedArray};

use crate::{bf16_from_f32, f32_from_bf16};

const WIDTH: usize = 3072;
const THREADS: usize = 128;
const VALUES: usize = 24;
const LANE_VALUES: usize = 8;
const ITERATION_STRIDE: usize = 1024;
const WARPS: usize = 4;
const WARP_VALUES: f32 = 768.0;
const INVERSE_COLS: f32 = 1.0 / WIDTH as f32;
const INVERSE_WARP: f32 = 1.0 / WARP_VALUES;
const VECTOR_BYTES: usize = LANE_VALUES * core::mem::size_of::<u16>();

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
fn column(tid: usize, index: usize) -> usize {
    tid * LANE_VALUES + index % LANE_VALUES + ITERATION_STRIDE * (index / LANE_VALUES)
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

#[inline(always)]
fn combine(mean_a: f32, m2_a: f32, mean_b: f32, m2_b: f32, count: f32) -> (f32, f32) {
    let total = float::mul_rn_ftz_f32(count, 2.0);
    let delta = subtract_ftz(mean_a, mean_b);
    let cross = float::mul_rn_ftz_f32(
        float::mul_rn_ftz_f32(float::mul_rn_ftz_f32(delta, delta), count),
        count,
    );
    let m2 = float::add_rn_ftz_f32(
        float::add_rn_ftz_f32(m2_a, m2_b),
        float::div_rn_f32(cross, total),
    );
    let mean = float::div_rn_f32(
        float::add_rn_ftz_f32(
            float::mul_rn_ftz_f32(count, mean_a),
            float::mul_rn_ftz_f32(count, mean_b),
        ),
        total,
    );
    (mean, m2)
}

#[inline(always)]
unsafe fn normalize<const ADD: bool>(
    input: *const u16,
    residual: *const u16,
    weight: *const u16,
    out: *mut u16,
    summed: *mut u16,
    offset: usize,
    eps: f32,
) {
    static mut STATS: SharedArray<f32, 8> = SharedArray::UNINIT;
    let stats = unsafe { SharedArray::as_raw_mut_ptr(&raw mut STATS) };
    let tid = thread::threadIdx_x() as usize;
    let lane = tid & 31;
    let warp = tid >> 5;
    let vector = aligned(input) && aligned(weight) && aligned_mut(out)
        && (!ADD || (aligned(residual) && aligned_mut(summed)));

    let mut values = [0.0f32; VALUES];
    let mut index = 0;
    while index < VALUES {
        let source = column(tid, index);
        if vector {
            let input_values = unsafe { load_vector(input.add(offset + source)) };
            let residual_values = if ADD {
                unsafe { load_vector(residual.add(offset + source)) }
            } else {
                [0u16; LANE_VALUES]
            };
            let mut element = 0;
            while element < LANE_VALUES {
                let mut value = float::mul_rn_ftz_f32(f32_from_bf16(input_values[element]), 1.0);
                if ADD {
                    value = float::add_rn_ftz_f32(value, f32_from_bf16(residual_values[element]));
                }
                values[index + element] = value;
                element += 1;
            }
        } else {
            let mut element = 0;
            while element < LANE_VALUES {
                let at = offset + source + element;
                let mut value = f32_from_bf16(unsafe { *input.add(at) });
                if ADD {
                    value = float::add_rn_ftz_f32(value, f32_from_bf16(unsafe { *residual.add(at) }));
                }
                values[index + element] = value;
                element += 1;
            }
        }
        index += LANE_VALUES;
    }

    // A summed output may alias input/residual; every source read above is in
    // registers before this barrier and the rounded residual writes.
    if ADD {
        thread::sync_threads();
        index = 0;
        while index < VALUES {
            let source = column(tid, index);
            let mut element = 0;
            while element < LANE_VALUES {
                let at = offset + source + element;
                unsafe { *summed.add(at) = bf16_from_f32(values[index + element]) };
                element += 1;
            }
            index += LANE_VALUES;
        }
    }

    let mut sum = 0.0;
    index = 0;
    while index < VALUES {
        sum = float::add_rn_ftz_f32(sum, values[index]);
        index += 1;
    }
    let warp_mean = float::mul_rn_ftz_f32(reduce_sum_lane(sum), INVERSE_WARP);
    let mut variance = 0.0;
    index = 0;
    while index < VALUES {
        let centered = subtract_ftz(values[index], warp_mean);
        variance = float::fma_rn_ftz_f32(centered, centered, variance);
        index += 1;
    }
    let warp_m2 = reduce_sum_lane(variance);
    if lane == 0 {
        unsafe {
            *stats.add(warp * 2) = warp_mean;
            *stats.add(warp * 2 + 1) = warp_m2;
        }
    }
    thread::sync_threads();
    let (even_mean, even_m2) = combine(
        unsafe { *stats.add(0) },
        unsafe { *stats.add(1) },
        unsafe { *stats.add(4) },
        unsafe { *stats.add(5) },
        WARP_VALUES,
    );
    let (odd_mean, odd_m2) = combine(
        unsafe { *stats.add(2) },
        unsafe { *stats.add(3) },
        unsafe { *stats.add(6) },
        unsafe { *stats.add(7) },
        WARP_VALUES,
    );
    let (mean, m2) = combine(even_mean, even_m2, odd_mean, odd_m2, WARP_VALUES * 2.0);
    let inverse = float::rsqrt_approx_ftz_f32(float::add_rn_ftz_f32(
        float::fma_rn_ftz_f32(m2, INVERSE_COLS, eps),
        float::mul_rn_ftz_f32(mean, mean),
    ));

    index = 0;
    while index < VALUES {
        let source = column(tid, index);
        if vector {
            let gamma_values = unsafe { load_vector(weight.add(source)) };
            let mut result = [0u16; LANE_VALUES];
            let mut element = 0;
            while element < LANE_VALUES {
                let centered = subtract_ftz(values[index + element], 0.0);
                let normalized = float::mul_rn_ftz_f32(inverse, centered);
                let value =
                    float::fma_rn_ftz_f32(normalized, f32_from_bf16(gamma_values[element]), 0.0);
                result[element] = bf16_from_f32(value);
                element += 1;
            }
            unsafe { store_vector(out.add(offset + source), result) };
        } else {
            let mut element = 0;
            while element < LANE_VALUES {
                let at = offset + source + element;
                let centered = subtract_ftz(values[index + element], 0.0);
                let normalized = float::mul_rn_ftz_f32(inverse, centered);
                let value = float::fma_rn_ftz_f32(
                    normalized,
                    f32_from_bf16(unsafe { *weight.add(source + element) }),
                    0.0,
                );
                unsafe { *out.add(at) = bf16_from_f32(value) };
                element += 1;
            }
        }
        index += LANE_VALUES;
    }
}

#[inline(always)]
unsafe fn rms_norm_3072<const ADD: bool>(
    input: *const u16,
    residual: *const u16,
    weight: *const u16,
    out: *mut u16,
    summed: *mut u16,
    rows: u32,
    width: u32,
    eps: f32,
) {
    let row = thread::blockIdx_x() as usize;
    let block = thread::blockDim_x() as usize;
    // Fail closed on any unsupported launch geometry. The generic decoder
    // norm body is a 256-thread contract (column += 256, eight warps), so it
    // must never be called from this 128-thread H3072 leaf. No read or write
    // occurs when the width, block size or row is outside the exact recipe;
    // the typed host H3072 route precludes such a launch before enqueue.
    if block != THREADS || width as usize != WIDTH || row >= rows as usize {
        return;
    }
    unsafe { normalize::<ADD>(input, residual, weight, out, summed, row * WIDTH, eps) };
}

/// Exact width-3072 BF16 RMSNorm. Grid=(rows,1,1), block=(128,1,1).
#[inline(always)]
pub(crate) unsafe fn exact_rms_norm_3072_vector(
    input: *const u16,
    weight: *const u16,
    out: *mut u16,
    rows: u32,
    width: u32,
    eps: f32,
) {
    unsafe {
        rms_norm_3072::<false>(
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

/// Exact width-3072 BF16 residual RMSNorm. Statistics use the unrounded F32
/// input+residual sum; `summed` receives the RNE BF16 sum.
#[inline(always)]
pub(crate) unsafe fn exact_add_rms_norm_3072_vector(
    input: *const u16,
    residual: *const u16,
    weight: *const u16,
    out: *mut u16,
    summed: *mut u16,
    rows: u32,
    width: u32,
    eps: f32,
) {
    unsafe { rms_norm_3072::<true>(input, residual, weight, out, summed, rows, width, eps) };
}
