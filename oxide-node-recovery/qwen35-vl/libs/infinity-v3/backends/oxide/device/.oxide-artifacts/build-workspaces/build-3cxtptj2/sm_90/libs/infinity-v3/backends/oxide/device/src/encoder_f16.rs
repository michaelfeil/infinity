// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Actual-native H384 FP16 encoder primitives.
//!
//! The embedding and residual LayerNorm entries follow the staged vendor
//! width-512 tail used for encoder width 384: one warp per row, four rows per
//! CTA, two 256-column iterations with lanes 0..15 valid in the second one.
//! Mean pooling follows the actual spans reduction tree and finite Neumaier
//! fallback. These are the exact F16 primitives qualified by the public H384
//! native controls; no arithmetic is shared with the BF16 specializations.
//!
//! Only the H384 MiniLM serving shape is admitted. Hosts still validate every
//! index, pointer, row count and width before launching these entries.

use cuda_device::{SharedArray, convert, device, f16x2, float, thread, warp};

const HIDDEN: usize = 384;
const VALUES: usize = 16;
const STRIDE: usize = 256;
const LANE_VALUES: usize = 8;
const ROWS_PER_CTA: usize = 4;
const INVERSE_COLS: f32 = 1.0 / HIDDEN as f32;

#[inline(always)]
fn pack(bits: u16) -> u32 {
    u32::from(bits) | (u32::from(bits) << 16)
}
#[inline(always)]
fn narrow(packed: u32) -> u16 {
    (packed & 0xffff) as u16
}
#[inline(always)]
fn f16_from_f32(value: f32) -> u16 {
    narrow(convert::cvt_f16x2_f32(value, value))
}
#[inline(always)]
fn f32_from_f16(bits: u16) -> f32 {
    convert::cvt_f32_f16x2_lo(u32::from(bits))
}
#[inline(always)]
fn add_f16(left: u16, right: u16) -> u16 {
    narrow(f16x2::add_f16x2(pack(left), pack(right)))
}
#[inline(always)]
fn mul_f16(left: u16, right: u16) -> u16 {
    narrow(f16x2::mul_f16x2(pack(left), pack(right)))
}

/// Deterministic round-to-nearest-even f64 to binary16 conversion for the mean
/// reciprocal and finite fallback. The device has no direct f64 narrowing
/// intrinsic, and the qualified native control uses the same explicit path.
fn f64_to_f16_bits(value: f64) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 48) & 0x8000) as u16;
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & 0x000f_ffff_ffff_ffff;
    if exponent == 0x7ff {
        return if fraction == 0 { sign | 0x7c00 } else { 0x7fff };
    }
    if exponent == 0 {
        return sign;
    }
    let unbiased = exponent - 1023;
    if unbiased > 15 {
        return sign | 0x7c00;
    }
    let significand = (1u64 << 52) | fraction;
    fn round_shift(value: u64, shift: u32) -> u64 {
        if shift == 0 {
            return value;
        }
        if shift >= 64 {
            return 0;
        }
        let truncated = value >> shift;
        let remainder = value & ((1u64 << shift) - 1);
        let halfway = 1u64 << (shift - 1);
        if remainder > halfway || (remainder == halfway && (truncated & 1) != 0) {
            truncated + 1
        } else {
            truncated
        }
    }
    if unbiased >= -14 {
        let rounded = round_shift(significand, 42);
        if rounded == 2048 {
            let carried = unbiased + 1;
            return if carried > 15 {
                sign | 0x7c00
            } else {
                sign | (((carried + 15) as u16) << 10)
            };
        }
        return sign | (((unbiased + 15) as u16) << 10) | ((rounded - 1024) as u16);
    }
    let shift = 28 - unbiased;
    let rounded = round_shift(significand, shift as u32);
    if rounded >= 1024 {
        return sign | 0x0400;
    }
    sign | (rounded as u16)
}

#[inline(always)]
fn column(lane: usize, index: usize) -> usize {
    lane * LANE_VALUES + index % LANE_VALUES + STRIDE * (index / LANE_VALUES)
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

#[device]
unsafe fn normalize_384(
    lane: usize,
    values: [f32; VALUES],
    offset: usize,
    gamma: *const u16,
    beta: *const u16,
    out: *mut u16,
    eps: f32,
) {
    let mut sum = 0.0f32;
    let mut index = 0usize;
    #[unroll]
    while index < VALUES {
        if column(lane, index) < HIDDEN {
            sum = float::add_rn_ftz_f32(sum, values[index]);
        }
        index += 1;
    }
    let mean = float::mul_rn_ftz_f32(reduce_sum_lane(sum), INVERSE_COLS);
    let mut variance = 0.0f32;
    index = 0;
    #[unroll]
    while index < VALUES {
        if column(lane, index) < HIDDEN {
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
        let col = column(lane, index);
        if col < HIDDEN {
            let centered = subtract_ftz(values[index], mean);
            let normalized = float::mul_rn_ftz_f32(centered, inverse);
            let value = float::fma_rn_ftz_f32(
                normalized,
                f32_from_f16(unsafe { *gamma.add(col) }),
                f32_from_f16(unsafe { *beta.add(col) }),
            );
            unsafe { *out.add(offset + col) = f16_from_f32(value) };
        }
        index += 1;
    }
}

#[inline(always)]
unsafe fn embedding_value(
    ids: *const u32,
    positions: *const u32,
    types: *const u32,
    word: *const u16,
    position: *const u16,
    token_type: *const u16,
    row: usize,
    col: usize,
) -> f32 {
    let token = unsafe { *ids.add(row) } as usize;
    let position_row = unsafe { *positions.add(row) } as usize;
    let type_row = unsafe { *types.add(row) } as usize;
    let pair = add_f16(unsafe { *word.add(token * HIDDEN + col) }, unsafe {
        *token_type.add(type_row * HIDDEN + col)
    });
    let rounded = float::mul_rn_ftz_f32(f32_from_f16(pair), 1.0);
    float::add_rn_ftz_f32(
        rounded,
        f32_from_f16(unsafe { *position.add(position_row * HIDDEN + col) }),
    )
}

#[inline(always)]
unsafe fn residual_value(
    input: *const u16,
    residual: *const u16,
    bias: *const u16,
    index: usize,
    col: usize,
) -> f32 {
    let mut value = float::mul_rn_ftz_f32(f32_from_f16(unsafe { *input.add(index) }), 1.0);
    if !residual.is_null() {
        value = float::add_rn_ftz_f32(value, f32_from_f16(unsafe { *residual.add(index) }));
    }
    if !bias.is_null() {
        value = float::add_rn_ftz_f32(value, f32_from_f16(unsafe { *bias.add(col) }));
    }
    value
}

#[inline(always)]
unsafe fn plain_value(input: *const u16, index: usize) -> f32 {
    float::mul_rn_ftz_f32(f32_from_f16(unsafe { *input.add(index) }), 1.0)
}

/// Embedding gather plus affine LayerNorm, H384, grid=ceil(tokens/4), block=128.
#[device]
pub(crate) unsafe fn embedding_norm_384_f16(
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
    eps: f32,
) {
    let tid = thread::threadIdx_x() as usize;
    let lane = tid & 31;
    let row = thread::blockIdx_x() as usize * ROWS_PER_CTA + tid / 32;
    if row >= tokens as usize {
        return;
    }
    let mut values = [0.0f32; VALUES];
    let mut index = 0;
    #[unroll]
    while index < VALUES {
        let col = column(lane, index);
        if col < HIDDEN {
            values[index] = unsafe {
                embedding_value(ids, positions, types, word, position, token_type, row, col)
            };
        }
        index += 1;
    }
    unsafe { normalize_384(lane, values, row * HIDDEN, gamma, beta, out, eps) };
}

/// Residual plus affine LayerNorm, H384, grid=ceil(rows/4), block=128.
/// The admitted MiniLM path passes a null bias because Lt folds it earlier.
#[device]
pub(crate) unsafe fn layer_norm_384_f16(
    input: *const u16,
    residual: *const u16,
    bias: *const u16,
    gamma: *const u16,
    beta: *const u16,
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
    let offset = row * HIDDEN;
    let mut values = [0.0f32; VALUES];
    let mut index = 0;
    #[unroll]
    while index < VALUES {
        let col = column(lane, index);
        if col < HIDDEN {
            values[index] = unsafe { residual_value(input, residual, bias, offset + col, col) };
        }
        index += 1;
    }
    unsafe { normalize_384(lane, values, offset, gamma, beta, out, eps) };
}

/// Plain affine LayerNorm without a residual, H384, grid=ceil(rows/4), block=128.
#[device]
pub(crate) unsafe fn plain_norm_384_f16(
    input: *const u16,
    gamma: *const u16,
    beta: *const u16,
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
    let offset = row * HIDDEN;
    let mut values = [0.0f32; VALUES];
    let mut index = 0;
    #[unroll]
    while index < VALUES {
        let col = column(lane, index);
        if col < HIDDEN {
            values[index] = unsafe { plain_value(input, offset + col) };
        }
        index += 1;
    }
    unsafe { normalize_384(lane, values, offset, gamma, beta, out, eps) };
}

#[inline(always)]
unsafe fn finite_fallback(
    input: *const u16,
    start: usize,
    length: usize,
    width: usize,
    feature: usize,
    original: u16,
) -> u16 {
    if f32_from_f16(original).is_finite() {
        return original;
    }
    let mut sum = 0.0f64;
    let mut correction = 0.0f64;
    let mut row = 0usize;
    while row < length {
        let value = f64::from(f32_from_f16(unsafe {
            *input.add((start + row) * width + feature)
        }));
        if !value.is_finite() {
            return original;
        }
        let next = sum + value;
        let error = if sum.abs() >= value.abs() {
            (sum - next) + value
        } else {
            (value - next) + sum
        };
        correction += error;
        sum = next;
        row += 1;
    }
    f64_to_f16_bits((sum + correction) / (length as f64))
}

/// Actual spans mean pooling for H384 F16 storage. Grid=(count,ceil(width/8)),
/// block=256; each block reduces one eight-feature tile with the vendor tree and
/// widens the exact F16 result to the FP32 public embedding.
#[inline(always)]
pub(crate) unsafe fn mean_pool_384_f16(
    input: *const u16,
    cu_seqlens: *const u32,
    selected: *const u32,
    out: *mut f32,
    count: u32,
    hidden: u32,
) {
    let sequence = thread::blockIdx_x() as usize;
    if sequence >= count as usize || hidden == 0 {
        return;
    }
    let base = thread::blockIdx_y() as usize * 8;
    let index = unsafe { *selected.add(sequence) } as usize;
    let start = unsafe { *cu_seqlens.add(index) } as usize;
    let end = unsafe { *cu_seqlens.add(index + 1) } as usize;
    if end <= start {
        return;
    }
    let length = end - start;
    let mut threads = 1usize;
    while threads < length && threads < 1024 {
        threads *= 2;
    }
    static mut PARTIAL: SharedArray<u16, 8192> = SharedArray::UNINIT;
    let scratch = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PARTIAL) };
    let tid = thread::threadIdx_x() as usize;
    let stride = thread::blockDim_x() as usize;
    let mut i = tid;
    while i < threads * 8 {
        let feature = base + i % 8;
        let mut sum = 0u16;
        if feature < hidden as usize {
            let mut row = i / 8;
            while row < length {
                sum = add_f16(sum, unsafe {
                    *input.add((start + row) * hidden as usize + feature)
                });
                row += threads;
            }
        }
        unsafe { *scratch.add(i) = sum };
        i += stride;
    }
    thread::sync_threads();
    let mut step = threads / 2;
    while step > 0 {
        let mut i = tid;
        while i < step * 8 {
            let left = unsafe { *scratch.add(i) };
            let right = unsafe { *scratch.add(i + step * 8) };
            unsafe { *scratch.add(i) = add_f16(left, right) };
            i += stride;
        }
        thread::sync_threads();
        step /= 2;
    }
    let reciprocal = f64_to_f16_bits(1.0 / (length as f64));
    if tid < 8 && base + tid < hidden as usize {
        let partial = unsafe { *scratch.add(tid) };
        let original = add_f16(mul_f16(partial, reciprocal), 0);
        let value =
            unsafe { finite_fallback(input, start, length, hidden as usize, base + tid, original) };
        unsafe { *out.add(sequence * hidden as usize + base + tid) = f32_from_f16(value) };
    }
}

/// Public CLS, mean or last-token selection over F16 rows. Modes 0 and 2 widen
/// one exact row; mode 1 is an F32 accumulation fallback used only by callers
/// that deliberately bypass the qualified mean entry.
#[inline(always)]
pub(crate) unsafe fn pool_f16(
    input: *const u16,
    cu_seqlens: *const u32,
    selected: *const u32,
    out: *mut f32,
    count: u32,
    hidden: u32,
    mode: u32,
) {
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if hidden == 0 || index >= count as usize * hidden as usize || mode > 2 {
        return;
    }
    let row = index / hidden as usize;
    let column = index % hidden as usize;
    let sequence = unsafe { *selected.add(row) } as usize;
    let start = unsafe { *cu_seqlens.add(sequence) } as usize;
    let end = unsafe { *cu_seqlens.add(sequence + 1) } as usize;
    if end <= start {
        return;
    }
    let value = if mode == 1 {
        let mut total = 0.0;
        let mut token = start;
        while token < end {
            total += f32_from_f16(unsafe { *input.add(token * hidden as usize + column) });
            token += 1;
        }
        total / (end - start) as f32
    } else {
        let token = if mode == 2 { end - 1 } else { start };
        f32_from_f16(unsafe { *input.add(token * hidden as usize + column) })
    };
    unsafe { *out.add(index) = value };
}

/// Split one fused F16 QKV projection and add its packed bias.
#[inline(always)]
pub(crate) unsafe fn qkv_f16(
    input: *const u16,
    bias: *const u16,
    q: *mut u16,
    k: *mut u16,
    v: *mut u16,
    elements: u32,
    width: u32,
) {
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if index >= elements as usize || width == 0 {
        return;
    }
    let hidden = width as usize;
    let row = index / hidden;
    let column = index % hidden;
    let offset = row * hidden * 3 + column;
    unsafe {
        *q.add(index) = add_f16(*input.add(offset), *bias.add(column));
        *k.add(index) = add_f16(*input.add(offset + hidden), *bias.add(column + hidden));
        *v.add(index) = add_f16(
            *input.add(offset + hidden * 2),
            *bias.add(column + hidden * 2),
        );
    }
}

/// Widen selected F16 rows to FP32 without arithmetic or rounding.
#[inline(always)]
pub(crate) unsafe fn f16_to_f32(
    input: *const u16,
    out: *mut f32,
    selected: *const u32,
    count: u32,
    width: u32,
) {
    if count == 0 || width == 0 {
        return;
    }
    let elements = (count as usize) * (width as usize);
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if index >= elements {
        return;
    }
    let row = index / width as usize;
    let column = index % width as usize;
    let sequence = if selected.is_null() {
        row
    } else {
        (unsafe { *selected.add(row) }) as usize
    };
    unsafe { *out.add(index) = f32_from_f16(*input.add(sequence * width as usize + column)) };
}
