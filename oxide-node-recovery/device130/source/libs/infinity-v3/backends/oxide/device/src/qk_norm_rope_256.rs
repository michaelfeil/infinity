// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Fused in-place Q/K RMSNorm plus full NeoX rotary over head width 256.
//!
//! This mirrors the ordinary Native `fast_kernels.cu` branch used when the head
//! width is not 128: an ordinary contiguous RMSNorm followed by separate BF16
//! `a*cos - b*sin` and `b*cos + a*sin`. It deliberately does not reproduce, call,
//! or extend the width-128 Candle HFMA kernel.
//!
//! One CTA owns one Q or K head and launches 256 threads. Thread `column` reads
//! one BF16 value, accumulates `value * value` in FP32, participates in the same
//! eight-warp block reduction as the ordinary decoder norm, and rounds the affine
//! result to BF16 into shared scratch. After the CTA barrier, each of the first
//! 128 lanes writes both separated-BF16 rotary members of its pair. Q/K token
//! strides are BF16-element strides; heads are adjacent 256-wide
//! rows. V, Q/K padding, and every other head are not touched. The caller owns
//! validation of allocations, alias disjointness, positions bounds, and the exact
//! launch grid `tokens * (qheads + kvheads)` with 256 threads.

use cuda_device::{SharedArray, float, ptx_asm, thread};

use crate::{bf16_from_f32, block_sum, read};

const QK256_HEAD_WIDTH: usize = 256;
const QK256_HALF: usize = QK256_HEAD_WIDTH / 2;
const QK256_THREADS: usize = 256;

#[inline(always)]
fn qk256_bf16_mul(lhs: u16, rhs: u16) -> u16 {
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

#[inline(always)]
fn qk256_bf16_add(lhs: u16, rhs: u16) -> u16 {
    let result: u16;
    unsafe {
        ptx_asm!(
            "add.rn.bf16 %0, %1, %2;",
            out("=h") result,
            in("h") lhs,
            in("h") rhs,
        );
    }
    result
}

#[inline(always)]
fn qk256_bf16_sub(lhs: u16, rhs: u16) -> u16 {
    let result: u16;
    unsafe {
        ptx_asm!(
            "sub.rn.bf16 %0, %1, %2;",
            out("=h") result,
            in("h") lhs,
            in("h") rhs,
        );
    }
    result
}

/// In-place Q/K RMSNorm and full NeoX rotary with the 14-argument ABI shared
/// with `infinity_qk_norm_rope_128_bf16`.
///
/// `qstride` and `kstride` are token strides in BF16 elements and must be at
/// least `qheads * 256` and `kvheads * 256`. `table_stride` is the F32 cosine/sine
/// row stride and must be at least 128. The row selected for each token is
/// `positions[token]`. Each head's first 128 columns use the same angles as its
/// second 128 columns. `eps` is the RMS epsilon added to the mean square.
#[inline(always)]
pub(crate) unsafe fn qk256_qk_norm_rope_256(
    q: *mut u16,
    k: *mut u16,
    qgamma: *const u16,
    kgamma: *const u16,
    cosine: *const f32,
    sine: *const f32,
    positions: *const u32,
    tokens: u32,
    qheads: u32,
    kvheads: u32,
    qstride: u64,
    kstride: u64,
    table_stride: u32,
    eps: f32,
) {
    let tid = thread::threadIdx_x() as usize;
    if tid >= QK256_THREADS {
        return;
    }
    let qrows = tokens as u64 * qheads as u64;
    let total_rows = qrows + tokens as u64 * kvheads as u64;
    let row = thread::blockIdx_x() as u64;
    if row >= total_rows {
        return;
    }
    let is_q = row < qrows;
    let heads = if is_q { qheads as u64 } else { kvheads as u64 };
    if heads == 0 {
        return;
    }
    let (data, weight, stride, minimum_stride) = if is_q {
        (q, qgamma, qstride, qheads as u64 * QK256_HEAD_WIDTH as u64)
    } else {
        (k, kgamma, kstride, kvheads as u64 * QK256_HEAD_WIDTH as u64)
    };
    if stride < minimum_stride || table_stride < QK256_HALF as u32 {
        return;
    }
    let r = if is_q { row } else { row - qrows };
    let token = (r / heads) as usize;
    if token >= tokens as usize {
        return;
    }
    let head = (r % heads) as usize;
    let offset = (token as u64 * stride + (head * QK256_HEAD_WIDTH) as u64) as usize;

    let value = unsafe { read(data, offset + tid) };
    let sum = float::fma_rn_f32(value, value, 0.0);
    static mut QK256_REDUCTION: SharedArray<f32, 8> = SharedArray::UNINIT;
    static mut QK256_NORMALIZED: SharedArray<u16, QK256_HEAD_WIDTH> = SharedArray::UNINIT;
    let reduction = unsafe { SharedArray::as_raw_mut_ptr(&raw mut QK256_REDUCTION) };
    let normalized = unsafe { SharedArray::as_raw_mut_ptr(&raw mut QK256_NORMALIZED) };
    let inverse =
        float::rsqrt_approx_f32(unsafe { block_sum(sum, reduction) } / QK256_HEAD_WIDTH as f32 + eps);
    let affine = float::mul_rn_f32(float::mul_rn_f32(value, inverse), unsafe {
        read(weight, tid)
    });
    unsafe { *normalized.add(tid) = bf16_from_f32(affine) };
    thread::sync_threads();

    if tid >= QK256_HALF {
        return;
    }
    let phase = unsafe { *positions.add(token) } as usize * table_stride as usize + tid;
    let cos_bits = bf16_from_f32(unsafe { *cosine.add(phase) });
    let sin_bits = bf16_from_f32(unsafe { *sine.add(phase) });
    let a = unsafe { *normalized.add(tid) };
    let b = unsafe { *normalized.add(tid + QK256_HALF) };
    let first = qk256_bf16_sub(qk256_bf16_mul(a, cos_bits), qk256_bf16_mul(b, sin_bits));
    let second = qk256_bf16_add(qk256_bf16_mul(b, cos_bits), qk256_bf16_mul(a, sin_bits));
    unsafe {
        *data.add(offset + tid) = first;
        *data.add(offset + QK256_HALF + tid) = second;
    }
}
