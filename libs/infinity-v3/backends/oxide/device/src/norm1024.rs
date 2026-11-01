// SPDX-License-Identifier: Apache-2.0
// Port of the qualified width-1024 rounded normalization arithmetic in
// backends/libtorch/cpp/kernels/fast_kernels.cu (add_norm_register_kernel /
// run_norm and the fused embedding gather path).
// Original contributions Copyright Infinity Core Contributors,
// Licensed under the Infinity Restricted MIT License 1.0; see the repository LICENSE.
//! Width-1024 rounded BF16 encoder normalization with native statistics.
//!
//! One 256-thread CTA owns each row. Thread `tid` holds four values at columns
//! `tid + j * 256`. The global mean is reduced before the second pass computes
//! centered variance. Unlike the H768 Candle path, the native fast kernel uses
//! a precise divide for both mean and variance, a separate epsilon add, a
//! non-FTZ approximate reciprocal square root, and two multiplies plus an add
//! for the affine epilogue.

use cuda_device::{SharedArray, float, thread};

use crate::{bf16_from_f32, block_sum, f32_from_bf16, read, write};

const WIDTH: usize = 1024;
const THREADS: usize = 256;
const VALUES: usize = WIDTH / THREADS;

#[inline(always)]
fn round_bf16(value: f32) -> f32 {
    f32_from_bf16(bf16_from_f32(value))
}

/// Native rounded embedding loader: round word+type to BF16, then round the
/// position addition to BF16 before collecting statistics.
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
    let pair = float::add_rn_f32(unsafe { read(word, token * WIDTH + column) }, unsafe {
        read(token_type, type_row * WIDTH + column)
    });
    round_bf16(float::add_rn_f32(round_bf16(pair), unsafe {
        read(position, position_row * WIDTH + column)
    }))
}

/// Native rounded residual loader. The projection output is already BF16 and
/// includes its projection bias. A non-null optional bias keeps the existing
/// rounded-add contract; the residual addition is rounded before statistics.
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
    if residual.is_null() {
        value
    } else {
        round_bf16(float::add_rn_f32(value, unsafe { read(residual, index) }))
    }
}

#[inline(always)]
unsafe fn normalize(
    values: [f32; VALUES],
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
    while index < VALUES {
        sum = float::add_rn_f32(sum, values[index]);
        index += 1;
    }
    let mean = float::div_rn_f32(unsafe { block_sum(sum, scratch) }, WIDTH as f32);
    let mut variance = 0.0;
    index = 0;
    while index < VALUES {
        let centered = values[index] - mean;
        variance = float::fma_rn_f32(centered, centered, variance);
        index += 1;
    }
    let variance = float::div_rn_f32(unsafe { block_sum(variance, scratch) }, WIDTH as f32);
    let inverse = float::rsqrt_approx_f32(float::add_rn_f32(variance, eps));
    index = 0;
    while index < VALUES {
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
pub(crate) unsafe fn embedding_norm_1024_rounded(
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
    let row = thread::blockIdx_x() as usize;
    if row >= tokens as usize {
        return;
    }
    let tid = thread::threadIdx_x() as usize;
    let mut values = [0.0f32; VALUES];
    let mut index = 0;
    while index < VALUES {
        values[index] = unsafe {
            embedding_value(
                ids,
                positions,
                types,
                word,
                position,
                token_type,
                row,
                tid + index * THREADS,
            )
        };
        index += 1;
    }
    unsafe { normalize(values, tid, row * WIDTH, gamma, beta, out, eps) };
}

#[inline(always)]
pub(crate) unsafe fn layer_norm_1024_rounded(
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
    let mut values = [0.0f32; VALUES];
    let mut index = 0;
    while index < VALUES {
        let column = tid + index * THREADS;
        values[index] = unsafe { residual_value(input, residual, bias, offset + column, column) };
        index += 1;
    }
    unsafe { normalize(values, tid, offset, gamma, beta, out, eps) };
}
