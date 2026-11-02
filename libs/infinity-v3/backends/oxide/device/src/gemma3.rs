// SPDX-License-Identifier: MIT AND LicenseRef-Infinity-Restricted-MIT-1.0
// Upstream Candle kernel portions retain MIT; new original port contributions
// use the Infinity Restricted MIT License 1.0. No inherited grant is changed.
//
// Adapted from the published tracked sources
//   libs/infinity-v3/backends/libtorch/cpp/vendor/candle_kernels/kernels/gemma_rms_norm.cu
//   libs/infinity-v3/backends/libtorch/cpp/kernels/qwen35_kernels.cu
//   libs/infinity-v3/backends/oxide/device/src/qk_norm_rope_128.rs
// (repository michaelfeil/infinity, commit c4c90b961c73ca2a3f669383559fcfaf99b4344f).
// Original contributions Copyright Infinity Core Contributors,
// Licensed under the Infinity Restricted MIT License 1.0; see the repository
// LICENSE, NOTICE and THIRD_PARTY_NOTICES.md. The Infinity Restricted MIT
// License 1.0 is not the standard MIT License.
//
// Candle MIT notice retained:
// Copyright The Hugging Face Team. All rights reserved.
// Permission is hereby granted, free of charge, to any
// person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the
// Software without restriction, including without
// limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software
// is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice
// shall be included in all copies or substantial portions
// of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
// ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
// TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
// PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
// SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
// IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//! Exact BF16 Gemma3 device primitives for qualified 128/256 head widths and
//! residual widths up to 8192.
//!
//! The four exports are source-only device entry points; the host owns launch
//! geometry, positive widths/strides, valid position ranges, and even head
//! widths no greater than 256. No input is clamped and no dynamic
//! shared-memory metadata is queried here.
//!
//! * `rms_norm_bf16_f32scale` keeps Candle's optimized legacy 256-lane
//!   reduction, its contracted `sum += v * v`, full-mask shuffle-down tree,
//!   eight-warp shared partials, `rsqrt.approx`, and separately rounded FP32
//!   affine products.
//! * `residual_norm_bf16_f32scale` computes statistics from `input` only and
//!   rounds the normalized affine result to BF16 before the BF16 residual add.
//! * `qk_norm_rope_bf16` reproduces the native Candle norm tree in a BF16
//!   shared row for any positive even head width up to 256, then applies the
//!   native `fma.rn.bf16`/`mul.bf16` rotary with in-place packed-QKV writes.

use cuda_device::{float, ptx_asm, thread, SharedArray};

use crate::{bf16_from_f32, f32_from_bf16};

const WARPS: usize = 8;
const QK_MAX: usize = 256;

/// `__shfl_down_sync(0xffffffff, value, delta)` over a full warp.
#[inline(always)]
fn shuffle_down(value: f32, delta: u32) -> f32 {
    let source = value.to_bits();
    let mask: u32 = 0xffff_ffff;
    let result: u32;
    unsafe {
        ptx_asm!(
            "shfl.sync.down.b32 %0, %1, %2, 0x1f, %3;",
            out("=r") result,
            in("r") source,
            in("r") delta,
            in("r") mask,
        );
    }
    f32::from_bits(result)
}

/// Native full-mask shuffle-down reduction, offsets 16, 8, 4, 2, 1.
/// The unguarded adds match the native source; lane 0 observes the exact tree.
#[inline(always)]
fn warp_down_sum(value: f32) -> f32 {
    let value = float::add_rn_f32(value, shuffle_down(value, 16));
    let value = float::add_rn_f32(value, shuffle_down(value, 8));
    let value = float::add_rn_f32(value, shuffle_down(value, 4));
    let value = float::add_rn_f32(value, shuffle_down(value, 2));
    float::add_rn_f32(value, shuffle_down(value, 1))
}

/// Candle legacy inverse RMS for one row-wide CTA; the reduction width is the launched block.
///
/// Native source: `gemma_rms_norm_bf16` in `gemma_rms_norm.cu` (and the same
/// tree inline in `gemma3_qk_norm_rotary`). `sum += v * v` is expected to
/// contract to an FP32 FMA under native NVCC default `fmad`; the root GPU
/// golden result remains the authority for that contract. The partial buffer
/// has eight FP32 slots, one per warp. All threads execute both barriers.
#[inline(always)]
unsafe fn legacy_inverse(
    input: *const u16,
    offset: usize,
    width: u32,
    eps: f32,
    partial: *mut f32,
) -> f32 {
    let tid = thread::threadIdx_x() as usize;
    let threads = thread::blockDim_x() as usize;
    let warps = threads / 32;
    let width = width as usize;
    let mut sum = 0.0f32;
    let mut column = tid;
    while column < width {
        let value = f32_from_bf16(unsafe { *input.add(offset + column) });
        sum = float::fma_rn_f32(value, value, sum);
        column += threads;
    }
    let mut sum = warp_down_sum(sum);
    if (tid & 31) == 0 {
        unsafe {
            *partial.add(tid / 32) = sum;
        }
    }
    thread::sync_threads();
    if tid < 32 {
        sum = if tid < warps {
            unsafe { *partial.add(tid) }
        } else {
            0.0f32
        };
        sum = warp_down_sum(sum);
        if tid == 0 {
            let mean = float::div_rn_f32(sum, width as f32);
            let variance = float::add_rn_f32(mean, eps);
            unsafe {
                *partial = float::rsqrt_approx_f32(variance);
            }
        }
    }
    thread::sync_threads();
    unsafe { *partial }
}

/// One CTA per row, looping any hidden width up to 8192 at the launched width.
/// `scale` is FP32 host data (BF16 weight expanded to FP32, plus one).
/// Native source: `gemma_rms_norm_bf16` in `gemma_rms_norm.cu`.
/// `out` may alias `input`: all statistics reads finish before the barrier
/// that precedes any output write.
#[inline(always)]
pub(crate) unsafe fn rms_norm_bf16_f32scale(
    input: *const u16,
    scale: *const f32,
    out: *mut u16,
    rows: u32,
    width: u32,
    eps: f32,
) {
    let row = thread::blockIdx_x() as usize;
    if row >= rows as usize || width == 0 {
        return;
    }
    static mut RMS_PARTIAL: SharedArray<f32, WARPS> = SharedArray::UNINIT;
    let partial = unsafe { SharedArray::as_raw_mut_ptr(&raw mut RMS_PARTIAL) };
    let threads = thread::blockDim_x() as usize;
    let width = width as usize;
    let offset = row * width;
    let inverse = unsafe { legacy_inverse(input, offset, width as u32, eps, partial) };
    let tid = thread::threadIdx_x() as usize;
    let mut column = tid;
    while column < width {
        let value = f32_from_bf16(unsafe { *input.add(offset + column) });
        let normalized = float::mul_rn_f32(value, inverse);
        let scaled = float::mul_rn_f32(normalized, unsafe { *scale.add(column) });
        unsafe {
            *out.add(offset + column) = bf16_from_f32(scaled);
        }
        column += threads;
    }
}

/// Native source: `gemma3_legacy_residual_norm_bf16` in `qwen35_kernels.cu`,
/// which follows `gemma_rms_norm_bf16` and adds the residual with a BF16 add.
/// One CTA per row, looping any hidden width up to 8192 at the launched width.
/// Statistics are taken from `input` only. The normalized affine value rounds
/// to BF16 before the residual addition, and that sum rounds to BF16 again.
/// `out` may alias `input` or `residual`: each lane reads both operands before
/// writing its column, and all statistics reads precede the barriers.
#[inline(always)]
pub(crate) unsafe fn residual_norm_bf16_f32scale(
    input: *const u16,
    residual: *const u16,
    scale: *const f32,
    out: *mut u16,
    rows: u32,
    width: u32,
    eps: f32,
) {
    let row = thread::blockIdx_x() as usize;
    if row >= rows as usize || width == 0 {
        return;
    }
    static mut RESIDUAL_PARTIAL: SharedArray<f32, WARPS> = SharedArray::UNINIT;
    let partial = unsafe { SharedArray::as_raw_mut_ptr(&raw mut RESIDUAL_PARTIAL) };
    let threads = thread::blockDim_x() as usize;
    let width = width as usize;
    let offset = row * width;
    let inverse = unsafe { legacy_inverse(input, offset, width as u32, eps, partial) };
    let tid = thread::threadIdx_x() as usize;
    let mut column = tid;
    while column < width {
        let value = f32_from_bf16(unsafe { *input.add(offset + column) });
        let normalized = float::mul_rn_f32(value, inverse);
        let scaled = float::mul_rn_f32(normalized, unsafe { *scale.add(column) });
        let affine = bf16_from_f32(scaled);
        let residual_value = f32_from_bf16(unsafe { *residual.add(offset + column) });
        let summed = float::add_rn_f32(residual_value, f32_from_bf16(affine));
        unsafe {
            *out.add(offset + column) = bf16_from_f32(summed);
        }
        column += threads;
    }
}

/// Native BF16 multiply, `mul.bf16`, as used by the published device port.
#[inline(always)]
fn bf16_mul(lhs: u16, rhs: u16) -> u16 {
    let result: u16;
    unsafe {
        ptx_asm!(
            "mul.bf16 %0, %1, %2;",
            out("=h") result,
            in("h") lhs,
            in("h") rhs,
        );
    }
    result
}

/// Native BF16 fused multiply-add, `fma.rn.bf16`.
#[inline(always)]
fn bf16_fma(first: u16, second: u16, addend: u16) -> u16 {
    let result: u16;
    unsafe {
        ptx_asm!(
            "fma.rn.bf16 %0, %1, %2, %3;",
            out("=h") result,
            in("h") first,
            in("h") second,
            in("h") addend,
        );
    }
    result
}

/// Exact in-place Q/K norm plus rotary for even head widths up to 256.
///
/// Native source: `gemma3_qk_norm_rotary` in `qwen35_kernels.cu` (Candle
/// legacy norm tree at the Native power-of-two width). The first `tokens * qheads` CTAs process Q
/// and the remaining `tokens * kvheads` CTAs process K. `head_dim` is the
/// positive even per-head width (the qualified model uses 128 and 256); the
/// paired-rotary frequency offset and cosine/sine phase span are
/// `head_dim / 2`. `q` and `k` are mutable views at their original packed-QKV
/// offsets, with `qstride` and `kstride` counting BF16 elements per original
/// token row. V is never touched.
///
/// Argument order: `q, k, qscale, kscale, cosine, sine, positions, tokens,
/// qheads, kvheads, head_dim, qstride, kstride, table_stride, eps`.
///
/// The cosine/sine tables are FP32 values that were already BF16-rounded before
/// expansion, so converting them back to BF16 reproduces the native table.
/// The sine products round to BF16 separately; the cosine products are native
/// `fma.rn.bf16` operations with the negated/rounded addend. The normalized
/// values are staged in a static BF16 shared row of 256 elements before any
/// in-place write.
#[inline(always)]
pub(crate) unsafe fn qk_norm_rope_bf16(
    q: *mut u16,
    k: *mut u16,
    qscale: *const f32,
    kscale: *const f32,
    cosine: *const f32,
    sine: *const f32,
    positions: *const u32,
    tokens: u32,
    qheads: u32,
    kvheads: u32,
    head_dim: u32,
    qstride: u64,
    kstride: u64,
    table_stride: u32,
    eps: f32,
) {
    let dimension = head_dim as usize;
    if dimension == 0 || dimension > QK_MAX || (dimension & 1) != 0 {
        return;
    }
    let half = dimension / 2;
    let block = thread::blockIdx_x() as u64;
    let qrows = (tokens as u64) * (qheads as u64);
    let total = qrows + (tokens as u64) * (kvheads as u64);
    if block >= total {
        return;
    }
    let query = block < qrows;
    let row = if query { block } else { block - qrows };
    let heads = if query { qheads as u64 } else { kvheads as u64 };
    if heads == 0 {
        return;
    }
    let token = (row / heads) as usize;
    if token >= tokens as usize {
        return;
    }
    let head = (row % heads) as usize;
    let (data, scale, stride): (*mut u16, *const f32, u64) = if query {
        (q, qscale, qstride)
    } else {
        (k, kscale, kstride)
    };
    let base = (token as u64 * stride + (head * dimension) as u64) as usize;
    let tid = thread::threadIdx_x() as usize;
    let threads = thread::blockDim_x() as usize;

    static mut QK_NORMALIZED: SharedArray<u16, QK_MAX> = SharedArray::UNINIT;
    static mut QK_PARTIAL: SharedArray<f32, WARPS> = SharedArray::UNINIT;
    let normalized = unsafe { SharedArray::as_raw_mut_ptr(&raw mut QK_NORMALIZED) };
    let partial = unsafe { SharedArray::as_raw_mut_ptr(&raw mut QK_PARTIAL) };

    let inverse =
        unsafe { legacy_inverse(data as *const u16, base, dimension as u32, eps, partial) };
    let mut column = tid;
    while column < dimension {
        let value = f32_from_bf16(unsafe { *data.add(base + column) });
        let normalized_value = float::mul_rn_f32(value, inverse);
        let scaled = float::mul_rn_f32(normalized_value, unsafe { *scale.add(column) });
        unsafe {
            *normalized.add(column) = bf16_from_f32(scaled);
        }
        column += threads;
    }
    thread::sync_threads();

    let table_row = unsafe { *positions.add(token) } as usize;
    let mut column = tid;
    while column < half {
        let a = unsafe { *normalized.add(column) };
        let b = unsafe { *normalized.add(column + half) };
        let phase = table_row * table_stride as usize + column;
        let cosine_value = bf16_from_f32(unsafe { *cosine.add(phase) });
        let sine_value = bf16_from_f32(unsafe { *sine.add(phase) });
        let b_sine = bf16_mul(b, sine_value);
        let a_sine = bf16_mul(a, sine_value);
        unsafe {
            *data.add(base + column) = bf16_fma(a, cosine_value, b_sine ^ 0x8000);
        }
        unsafe {
            *data.add(base + column + half) = bf16_fma(b, cosine_value, a_sine);
        }
        column += threads;
    }
}
