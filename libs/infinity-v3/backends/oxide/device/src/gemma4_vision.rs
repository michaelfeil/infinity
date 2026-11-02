// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Exact Gemma4 vision elementwise/spatial primitives.
//!
//! Arithmetic mirrors `kernels/qwen35_kernels.cu::exact_vision_norm` family
//! sources: FP32 statistics/angles, explicit rounding at the BF16 activation
//! boundary, clamp semantics matching `torch.clamp` on BF16 tensors, and a
//! row-major patch gather without synthetic padding.

use crate::{bf16_from_f32, read, write};

#[inline(always)]
fn gemma4_vision_sub_rn_f32(a: f32, b: f32) -> f32 {
    float::add_rn_f32(a, -b)
}

use cuda_device::{float, thread};

#[inline(always)]
fn bf16(value: f32) -> f32 {
    f32::from_bits(u32::from(bf16_from_f32(value)) << 16)
}

/// Clamp only the real `width` columns of each row. Padding columns remain
/// untouched so padded attention heads stay exactly zero.
#[inline(always)]
pub(crate) unsafe fn clip_rows(
    input: *const u16,
    out: *mut u16,
    rows: u32,
    width: u32,
    stride: u32,
    min: f32,
    max: f32,
) {
    if rows == 0 || width == 0 || stride < width {
        return;
    }
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if index >= rows as usize * width as usize {
        return;
    }
    let row = index / width as usize;
    let column = index % width as usize;
    let value = unsafe { read(input, row * stride as usize + column) };
    // `torch.clamp` keeps NaN and selects either bound exactly.
    let clamped = if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    };
    unsafe { write(out, row * stride as usize + column, clamped) };
}

/// `hidden + bf16(bf16(x_embedding) + bf16(y_embedding))` per patch.
#[inline(always)]
pub(crate) unsafe fn position_add(
    hidden: *const u16,
    table: *const u16,
    columns: *const u32,
    rows: *const u32,
    out: *mut u16,
    tokens: u32,
    width: u32,
    table_size: u32,
) {
    if tokens == 0 || width == 0 || table_size == 0 {
        return;
    }
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if index >= tokens as usize * width as usize {
        return;
    }
    let token = index / width as usize;
    let column = index % width as usize;
    let x = unsafe { *columns.add(token) } as usize;
    let y = unsafe { *rows.add(token) } as usize;
    let x_index = x.min(table_size as usize - 1);
    let y_index = y.min(table_size as usize - 1);
    let x_value = unsafe { read(table, x_index * width as usize + column) };
    let y_value = unsafe {
        read(
            table,
            table_size as usize * width as usize + y_index * width as usize + column,
        )
    };
    let position = bf16(float::add_rn_f32(x_value, y_value));
    let value = unsafe { read(hidden, index) };
    unsafe { write(out, index, float::add_rn_f32(value, position)) };
}

/// Per-token two-axis BF16 rotary table. Axis 0 uses columns, axis 1 rows.
#[inline(always)]
pub(crate) unsafe fn rope_tables(
    columns: *const u32,
    rows: *const u32,
    inverse: *const f32,
    cosine: *mut u16,
    sine: *mut u16,
    tokens: u32,
    pairs: u32,
) {
    if tokens == 0 || pairs == 0 {
        return;
    }
    let index = thread::blockIdx_x() as u64 * 256 + thread::threadIdx_x() as u64;
    if index >= tokens as u64 * pairs as u64 {
        return;
    }
    let token = (index / pairs as u64) as usize;
    let pair = (index % pairs as u64) as usize;
    let inverse = unsafe { *inverse.add(pair) };
    let x_phase = float::mul_rn_f32(unsafe { *columns.add(token) } as f32, inverse);
    let y_phase = float::mul_rn_f32(unsafe { *rows.add(token) } as f32, inverse);
    let table = token * 2 * pairs as usize;
    unsafe {
        *cosine.add(table + pair) = bf16_from_f32(libm::cosf(x_phase));
        *sine.add(table + pair) = bf16_from_f32(libm::sinf(x_phase));
        *cosine.add(table + pairs as usize + pair) = bf16_from_f32(libm::cosf(y_phase));
        *sine.add(table + pairs as usize + pair) = bf16_from_f32(libm::sinf(y_phase));
    }
}

/// Exact `rope2d` port from the Native vision kernel.
#[inline(always)]
pub(crate) unsafe fn rope2d(
    input: *const u16,
    cosine: *const u16,
    sine: *const u16,
    out: *mut u16,
    count: u32,
    heads: u32,
    dim: u32,
    token_stride: u64,
    head_stride: u64,
) {
    if count == 0 || heads == 0 || dim == 0 || dim % 4 != 0 {
        return;
    }
    let index = thread::blockIdx_x() as u64 * 256 + thread::threadIdx_x() as u64;
    if index >= count as u64 {
        return;
    }
    let dim = dim as usize;
    let heads = heads as usize;
    let d = index as usize % dim;
    let head = (index as usize / dim) % heads;
    let token = index as usize / (dim * heads);
    let part = d % (dim / 2);
    let quarter = dim / 4;
    let partner = if part < quarter {
        d + quarter
    } else {
        d - quarter
    };
    let base = token as u64 * token_stride + head as u64 * head_stride;
    let a = unsafe { read(input, base as usize + d) };
    let mut b = unsafe { read(input, base as usize + partner) };
    if part < quarter {
        b = -b;
    }
    let frequency = token * (dim / 2) + (d / (dim / 2)) * quarter + part % quarter;
    let p = bf16(float::mul_rn_f32(a, unsafe { read(cosine, frequency) }));
    let q = bf16(float::mul_rn_f32(b, unsafe { read(sine, frequency) }));
    unsafe { write(out, index as usize, float::add_rn_f32(p, q)) };
}

/// Row-major 2-D spatial average, rounded to BF16, then scaled by `sqrt(H)`.
#[inline(always)]
pub(crate) unsafe fn pool(
    input: *const u16,
    out: *mut u16,
    height: u32,
    width: u32,
    hidden: u32,
    kernel: u32,
    inverse: f32,
    scale: f32,
) {
    if height == 0 || width == 0 || hidden == 0 || kernel == 0 {
        return;
    }
    let bins = (height / kernel) as usize * (width / kernel) as usize;
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if index >= bins * hidden as usize {
        return;
    }
    let bin = index / hidden as usize;
    let column = index % hidden as usize;
    let columns = (width / kernel) as usize;
    let by = bin / columns;
    let bx = bin % columns;
    let mut sum = 0.0f32;
    for y in 0..kernel as usize {
        for x in 0..kernel as usize {
            let patch = (by * kernel as usize + y) * width as usize + bx * kernel as usize + x;
            let value = unsafe { read(input, patch * hidden as usize + column) };
            sum = float::add_rn_f32(sum, float::mul_rn_f32(value, inverse));
        }
    }
    let rounded = bf16(sum);
    unsafe { write(out, index, float::mul_rn_f32(rounded, scale)) };
}

/// Combined Native spatial pool and post-pool standardize. The sum is rounded
/// to BF16 once, then `sqrt(hidden)`, bias and scale stay in F32 until the final
/// BF16 activation write.
#[inline(always)]
pub(crate) unsafe fn pool_standardize_rows(
    input: *const u16,
    bias: *const f32,
    scale: *const f32,
    out: *mut u16,
    height: u32,
    width: u32,
    hidden: u32,
    kernel: u32,
    inverse: f32,
    sqrt_scale: f32,
) {
    if height == 0 || width == 0 || hidden == 0 || kernel == 0 {
        return;
    }
    let bins = (height / kernel) as usize * (width / kernel) as usize;
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if index >= bins * hidden as usize {
        return;
    }
    let bin = index / hidden as usize;
    let column = index % hidden as usize;
    let columns = (width / kernel) as usize;
    let by = bin / columns;
    let bx = bin % columns;
    let mut sum = 0.0f32;
    for y in 0..kernel as usize {
        for x in 0..kernel as usize {
            let patch = (by * kernel as usize + y) * width as usize + bx * kernel as usize + x;
            let value = unsafe { read(input, patch * hidden as usize + column) };
            sum = float::add_rn_f32(sum, float::mul_rn_f32(value, inverse));
        }
    }
    let pooled = float::mul_rn_f32(bf16(sum), sqrt_scale);
    let centered = gemma4_vision_sub_rn_f32(pooled, unsafe { *bias.add(column) });
    let value = float::mul_rn_f32(centered, unsafe { *scale.add(column) });
    unsafe { write(out, index, value) };
}

/// Replace ordered embedding rows by BF16 image feature rows.
#[inline(always)]
pub(crate) unsafe fn scatter_rows(
    destination: *mut u16,
    features: *const u16,
    rows: *const u32,
    row_count: u32,
    width: u32,
) {
    if row_count == 0 || width == 0 {
        return;
    }
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if index >= row_count as usize * width as usize {
        return;
    }
    let row = index / width as usize;
    let column = index % width as usize;
    let target = unsafe { *rows.add(row) } as usize * width as usize + column;
    let value = unsafe { *features.add(index) };
    unsafe { *destination.add(target) = value };
}
