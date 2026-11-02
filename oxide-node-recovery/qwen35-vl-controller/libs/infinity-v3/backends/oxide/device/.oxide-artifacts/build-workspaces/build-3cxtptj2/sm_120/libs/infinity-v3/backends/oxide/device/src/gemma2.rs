// SPDX-License-Identifier: MIT AND LicenseRef-Infinity-Restricted-MIT-1.0
// Upstream Candle MIT portions retain their MIT grant. Original port
// contributions use the Infinity Restricted MIT License 1.0.
//
// Exact arithmetic sources:
//   libtorch/cpp/kernels/qwen35_kernels.cu (Gemma2 DivSqrt norm and precise GELU)
//   libtorch/cpp/vendor/candle_kernels/pooling/mean_pool.cu (BF16 mean tree)
//   libtorch/cpp/models/gemma_models.cpp (EmbeddingGemma2 composition)
// Repository LICENSE, NOTICE and THIRD_PARTY_NOTICES remain authoritative.
//!
//! EmbeddingGemma2-only device primitives. The host validates width, row,
//! stride and position bounds before launch. Norm rows use exactly `width`
//! threads (256 or 512), matching the Native width-lane reduction tree.

use cuda_device::bf16x2::{add_bf16x2, mul_bf16x2};
use cuda_device::{SharedArray, device, float, ptx_asm, thread};

use crate::{bf16_from_f32, f32_from_bf16, read, write};

const MAX_THREADS: usize = 512;
const POOL_TILE: usize = 8;
const POOL_MAX_THREADS: usize = 1024;
const POOL_PARTIALS: usize = POOL_MAX_THREADS * POOL_TILE;

#[inline(always)]
fn rounded(value: f32) -> f32 {
    f32_from_bf16(bf16_from_f32(value))
}

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

#[inline(always)]
fn warp_down_sum(value: f32) -> f32 {
    let value = float::add_rn_f32(value, shuffle_down(value, 16));
    let value = float::add_rn_f32(value, shuffle_down(value, 8));
    let value = float::add_rn_f32(value, shuffle_down(value, 4));
    let value = float::add_rn_f32(value, shuffle_down(value, 2));
    float::add_rn_f32(value, shuffle_down(value, 1))
}

#[inline(always)]
unsafe fn inverse_sqrt_mean(
    input: *const u16,
    base: usize,
    width: usize,
    eps: f32,
    sums: *mut f32,
) -> f32 {
    let tid = thread::threadIdx_x() as usize;
    let mut sum = 0.0f32;
    let mut column = tid;
    while column < width {
        let value = unsafe { read(input, base + column) };
        sum = float::add_rn_f32(sum, float::mul_rn_f32(value, value));
        column += MAX_THREADS;
    }
    unsafe { *sums.add(tid) = sum };
    let mut stride = (thread::blockDim_x() as usize) / 2;
    while stride >= 32 {
        thread::sync_threads();
        if tid < stride {
            let left = unsafe { *sums.add(tid) };
            let right = unsafe { *sums.add(tid + stride) };
            unsafe { *sums.add(tid) = float::add_rn_f32(left, right) };
        }
        stride >>= 1;
    }
    if tid < 32 {
        let mut value = unsafe { *sums.add(tid) };
        let mut stride = 16;
        while stride > 0 {
            let other = shuffle_down(value, stride as u32);
            if tid < stride {
                value = float::add_rn_f32(value, other);
            }
            stride >>= 1;
        }
        if tid == 0 {
            let reciprocal = (1.0f64 / width as f64) as f32;
            let mean = float::mul_rn_f32(value, reciprocal);
            let variance = float::add_rn_f32(mean, eps);
            unsafe { *sums = float::sqrt_rn_f32(variance) };
        }
    }
    thread::sync_threads();
    unsafe { *sums }
}

/// Native `gemma2_norm_cuda` reference tree: DivSqrt, F32 scale, no +1.
#[inline(always)]
pub(crate) unsafe fn norm_bf16_f32scale(
    input: *const u16,
    scale: *const f32,
    out: *mut u16,
    rows: u32,
    width: u32,
    heads: u32,
    token_stride: u64,
    eps: f32,
) {
    if rows == 0 || width == 0 || heads == 0 {
        return;
    }
    static mut SUMS: SharedArray<f32, MAX_THREADS> = SharedArray::UNINIT;
    let sums = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SUMS) };
    let row = thread::blockIdx_x() as usize;
    if row >= rows as usize {
        return;
    }
    let width = width as usize;
    let base = (row / heads as usize) * token_stride as usize + (row % heads as usize) * width;
    let denominator = unsafe { inverse_sqrt_mean(input, base, width, eps, sums) };
    let out_base = row * width;
    let tid = thread::threadIdx_x() as usize;
    let mut column = tid;
    while column < width {
        let value = unsafe { read(input, base + column) };
        let normalized = float::div_rn_f32(value, denominator);
        let scaled = float::mul_rn_f32(normalized, unsafe { *scale.add(column) });
        unsafe { write(out, out_base + column, scaled) };
        column += MAX_THREADS;
    }
}

/// Native `gemma2_residual_norm_cuda`: statistics from `input` only, rounded
/// affine multiplied by the F32 scale, then one BF16 residual add.
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
    if rows == 0 || width == 0 {
        return;
    }
    static mut SUMS: SharedArray<f32, MAX_THREADS> = SharedArray::UNINIT;
    let sums = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SUMS) };
    let row = thread::blockIdx_x() as usize;
    if row >= rows as usize {
        return;
    }
    let width = width as usize;
    let base = row * width;
    let denominator = unsafe { inverse_sqrt_mean(input, base, width, eps, sums) };
    let tid = thread::threadIdx_x() as usize;
    let mut column = tid;
    while column < width {
        let value = unsafe { read(input, base + column) };
        let normalized = float::div_rn_f32(value, denominator);
        let scaled = rounded(float::mul_rn_f32(normalized, unsafe { *scale.add(column) }));
        let residual_value = unsafe { read(residual, base + column) };
        unsafe {
            write(
                out,
                base + column,
                float::add_rn_f32(residual_value, scaled),
            )
        };
        column += MAX_THREADS;
    }
}

/// Unit V RMS norm: same DivSqrt tree with the Native F32 ones scale.
#[inline(always)]
pub(crate) unsafe fn unit_norm_bf16(
    input: *const u16,
    out: *mut u16,
    rows: u32,
    width: u32,
    heads: u32,
    token_stride: u64,
    eps: f32,
) {
    if rows == 0 || width == 0 || heads == 0 {
        return;
    }
    static mut SUMS: SharedArray<f32, MAX_THREADS> = SharedArray::UNINIT;
    let sums = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SUMS) };
    let row = thread::blockIdx_x() as usize;
    if row >= rows as usize {
        return;
    }
    let width = width as usize;
    let base = (row / heads as usize) * token_stride as usize + (row % heads as usize) * width;
    let denominator = unsafe { inverse_sqrt_mean(input, base, width, eps, sums) };
    let out_base = row * width;
    let tid = thread::threadIdx_x() as usize;
    let mut column = tid;
    while column < width {
        let value = unsafe { read(input, base + column) };
        let normalized = float::div_rn_f32(value, denominator);
        unsafe { write(out, out_base + column, normalized) };
        column += MAX_THREADS;
    }
}

#[inline(always)]
fn precise_gelu(value: f32) -> f32 {
    let square = rounded(float::mul_rn_f32(value, value));
    let cube = rounded(float::mul_rn_f32(square, value));
    let product = rounded(float::mul_rn_f32(rounded(0.044715f32), cube));
    let alpha = rounded(float::add_rn_f32(value, product));
    let angle = rounded(float::mul_rn_f32(
        rounded(0.797_884_560_802_865_4f32),
        alpha,
    ));
    let tangent = rounded(libm::tanhf(angle));
    let half = rounded(float::mul_rn_f32(0.5f32, value));
    let plus = rounded(float::add_rn_f32(1.0f32, tangent));
    rounded(float::mul_rn_f32(half, plus))
}

/// Native gate-first packed MLP activation. Input rows are `[gate, up]`.
#[inline(always)]
pub(crate) unsafe fn gated_gelu_bf16(input: *const u16, out: *mut u16, rows: u32, width: u32) {
    if rows == 0 || width == 0 {
        return;
    }
    let width = width as usize;
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if index >= rows as usize * width {
        return;
    }
    let row = index / width;
    let column = index % width;
    let base = row * 2 * width + column;
    let gate = unsafe { read(input, base) };
    let up = unsafe { read(input, base + width) };
    let active = precise_gelu(gate);
    unsafe { write(out, index, float::mul_rn_f32(active, up)) };
}

/// Apply precise GELU to the per-layer gate and multiply the PLE row.
#[inline(always)]
pub(crate) unsafe fn ple_gate_bf16(
    gate: *const u16,
    ple: *const u16,
    out: *mut u16,
    rows: u32,
    width: u32,
    layers: u32,
    layer: u32,
) {
    if rows == 0 || width == 0 || layers == 0 || layer >= layers {
        return;
    }
    let width = width as usize;
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if index >= rows as usize * width {
        return;
    }
    let row = index / width;
    let column = index % width;
    let ple_index = (row * layers as usize + layer as usize) * width + column;
    let active = precise_gelu(unsafe { read(gate, index) });
    let value = unsafe { read(ple, ple_index) };
    unsafe { write(out, index, float::mul_rn_f32(active, value)) };
}

/// BF16 multiply by one host-prepared BF16 scalar widened to F32.
#[inline(always)]
pub(crate) unsafe fn scale_bf16(input: *const u16, out: *mut u16, count: u32, scale: f32) {
    if count == 0 {
        return;
    }
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if index >= count as usize {
        return;
    }
    let value = unsafe { read(input, index) };
    unsafe { write(out, index, float::mul_rn_f32(value, scale)) };
}

/// Native cold table initializer: phase = pos * inverse, libdevice cos/sin,
/// then one BF16 round-trip stored as F32.
#[inline(always)]
pub(crate) unsafe fn rope_tables_bf16(
    inverse: *const f32,
    cosine: *mut f32,
    sine: *mut f32,
    positions: u32,
    pairs: u32,
) {
    if positions == 0 || pairs == 0 {
        return;
    }
    let pairs = pairs as u64;
    let total = positions as u64 * pairs;
    let index = thread::blockIdx_x() as u64 * 256 + thread::threadIdx_x() as u64;
    if index >= total {
        return;
    }
    let position = (index / pairs) as f32;
    let inverse = unsafe { *inverse.add((index % pairs) as usize) };
    let phase = float::mul_rn_f32(position, inverse);
    unsafe {
        *cosine.add(index as usize) = rounded(libm::cosf(phase));
        *sine.add(index as usize) = rounded(libm::sinf(phase));
    }
}

#[device]
#[inline(always)]
fn pool_bf16_pair(bits: u16) -> u32 {
    u32::from(bits) | (u32::from(bits) << 16)
}

#[device]
#[inline(always)]
fn pool_bf16_add(left: u16, right: u16) -> u16 {
    (add_bf16x2(pool_bf16_pair(left), pool_bf16_pair(right)) & 0xffff) as u16
}

#[device]
#[inline(always)]
fn pool_bf16_mul(left: u16, right: u16) -> u16 {
    (mul_bf16x2(pool_bf16_pair(left), pool_bf16_pair(right)) & 0xffff) as u16
}

#[device]
#[inline(always)]
fn bf16_from_f64(value: f64) -> u16 {
    let bits: u16;
    unsafe {
        ptx_asm!("cvt.rn.bf16.f64 %0, %1;", out("=h") bits, in("d") value);
    }
    bits
}

/// Native packed BF16 mean over selected spans; output is widened to F32.
#[device]
#[inline(always)]
pub(crate) unsafe fn mean_pool_bf16(
    input: *const u16,
    cu_seqlens: *const u32,
    selected: *const u32,
    out: *mut f32,
    count: u32,
    width: u32,
) {
    let sequence = thread::blockIdx_x() as usize;
    if sequence >= count as usize || width == 0 {
        return;
    }
    let width = width as usize;
    let base = thread::blockIdx_y() as usize * POOL_TILE;
    let index = unsafe { *selected.add(sequence) } as usize;
    let start = unsafe { *cu_seqlens.add(index) } as usize;
    let end = unsafe { *cu_seqlens.add(index + 1) } as usize;
    if end <= start {
        return;
    }
    let length = end - start;
    let mut threads = 1usize;
    while threads < length && threads < POOL_MAX_THREADS {
        threads *= 2;
    }
    static mut PARTIAL: SharedArray<u16, POOL_PARTIALS> = SharedArray::UNINIT;
    let partial = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PARTIAL) };
    let tid = thread::threadIdx_x() as usize;
    let stride = thread::blockDim_x() as usize;
    let mut i = tid;
    while i < threads * POOL_TILE {
        let feature = base + i % POOL_TILE;
        let mut sum = 0u16;
        if feature < width {
            let mut row = i / POOL_TILE;
            while row < length {
                sum = pool_bf16_add(sum, unsafe { *input.add((start + row) * width + feature) });
                row += threads;
            }
        }
        unsafe { *partial.add(i) = sum };
        i += stride;
    }
    let mut step = threads / 2;
    while step > 0 {
        thread::sync_threads();
        let mut i = tid;
        while i < step * POOL_TILE {
            let left = unsafe { *partial.add(i) };
            let right = unsafe { *partial.add(i + step * POOL_TILE) };
            unsafe { *partial.add(i) = pool_bf16_add(left, right) };
            i += stride;
        }
        step /= 2;
    }
    thread::sync_threads();
    if tid < POOL_TILE && base + tid < width {
        let reciprocal = bf16_from_f64(float::div_rn_f64(1.0f64, length as f64));
        let scaled = pool_bf16_mul(unsafe { *partial.add(tid) }, reciprocal);
        let rounded = pool_bf16_add(scaled, 0);
        unsafe {
            *out.add(sequence * width + base + tid) = f32_from_bf16(rounded);
        }
    }
}
