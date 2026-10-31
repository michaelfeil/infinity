// Copyright Infinity Core Contributors.
// Licensed under the Infinity Restricted MIT License 1.0; see the repository LICENSE.
//! Register-cached specializations for the 384-wide BERT LayerNorm layers.
//!
//! Each 256-thread block handles one row. Values are loaded once into registers
//! before reduction, permitting input/output aliasing. The original reduction
//! tree and BF16 rounding boundaries are preserved.

use cuda_device::{SharedArray, float, thread};

use crate::{block_sum, embedding_value, norm_value, read, write};

const WIDTH: usize = 384;
const LANES: usize = 256;
const PER_LANE: usize = 2;

/// Normalize the cached values, using the generic kernel's reduction tree.
#[inline(always)]
unsafe fn normalize_384(
    values: [f32; PER_LANE],
    offset: usize,
    gamma: *const u16,
    beta: *const u16,
    out: *mut u16,
    eps: f32,
) {
    let lane = thread::threadIdx_x() as usize;
    static mut SCRATCH: SharedArray<f32, 8> = SharedArray::UNINIT;
    let scratch = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCRATCH) };
    let mut i = 0;
    let mut sum = 0.0;
    while i < PER_LANE {
        sum += values[i];
        i += 1;
    }
    let mean = unsafe { block_sum(sum, scratch) } / WIDTH as f32;
    i = 0;
    let mut variance = 0.0;
    while i < PER_LANE {
        if lane + i * LANES < WIDTH {
            let centered = values[i] - mean;
            variance = float::fma_rn_f32(centered, centered, variance);
        }
        i += 1;
    }
    let inv =
        1.0 / float::sqrt_rn_f32(unsafe { block_sum(variance, scratch) } / WIDTH as f32 + eps);
    i = 0;
    while i < PER_LANE {
        let column = lane + i * LANES;
        if column < WIDTH {
            let value = (values[i] - mean) * inv * unsafe { read(gamma, column) }
                + unsafe { read(beta, column) };
            unsafe { write(out, offset + column, value) };
        }
        i += 1;
    }
}

/// Affine LayerNorm over optional projection-bias/residual additions.
/// Grid=(rows,1,1), block=(256,1,1); the width is fixed at 384.
#[inline(always)]
pub(crate) unsafe fn layer_norm_384(
    input: *const u16,
    residual: *const u16,
    bias: *const u16,
    gamma: *const u16,
    beta: *const u16,
    out: *mut u16,
    rows: u32,
    eps: f32,
) {
    let row = thread::blockIdx_x() as usize;
    if row >= rows as usize {
        return;
    }
    let lane = thread::threadIdx_x() as usize;
    let offset = row * WIDTH;
    let mut values = [0.0f32; PER_LANE];
    let mut i = 0;
    while i < PER_LANE {
        let column = lane + i * LANES;
        if column < WIDTH {
            values[i] = unsafe { norm_value(input, residual, bias, offset + column, column) };
        }
        i += 1;
    }
    unsafe { normalize_384(values, offset, gamma, beta, out, eps) };
}

/// Fuse embedding gathers with affine LayerNorm, retaining FP32 position statistics.
/// Grid=(tokens,1,1), block=(256,1,1); the width is fixed at 384.
#[inline(always)]
pub(crate) unsafe fn embedding_norm_384(
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
    let row = thread::blockIdx_x() as usize;
    if row >= tokens as usize {
        return;
    }
    let lane = thread::threadIdx_x() as usize;
    let mut values = [0.0f32; PER_LANE];
    let mut i = 0;
    while i < PER_LANE {
        let column = lane + i * LANES;
        if column < WIDTH {
            values[i] = unsafe {
                embedding_value(
                    ids, positions, types, word, position, token_type, row, column, WIDTH,
                )
            };
        }
        i += 1;
    }
    unsafe { normalize_384(values, row * WIDTH, gamma, beta, out, eps) };
}
