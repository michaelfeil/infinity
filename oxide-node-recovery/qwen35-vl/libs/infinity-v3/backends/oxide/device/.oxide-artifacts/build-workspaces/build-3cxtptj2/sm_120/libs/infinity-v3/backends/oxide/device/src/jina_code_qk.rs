// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Full-hidden Q/K LayerNorm over packed `[Q768 | K768 | V768]` BF16 rows.
//! The prepared caller owns aligned buffers and launches four warps per CTA.

use cuda_device::{device, thread};

use crate::f32_from_bf16;
use crate::norm768_vector::{load_vector, normalize_vector};

const HIDDEN: usize = 768;
const PACKED: usize = 3 * HIDDEN;
const VALUES: usize = 24;
const VECTOR: usize = 8;

/// Normalize one packed slice using the unchanged native Candle arithmetic.
/// One warp reads its complete row before any lane overwrites its Q/K slice.
#[device]
unsafe fn normalize_slice(
    qkv: *mut u16,
    gamma: *const u16,
    beta: *const u16,
    offset: usize,
    lane: usize,
    eps: f32,
) {
    let mut values = [0.0f32; VALUES];
    let mut group = 0;
    #[unroll]
    while group < VALUES / VECTOR {
        let column = group * 256 + lane * VECTOR;
        let words = unsafe { load_vector(qkv.add(offset + column)) };
        let mut element = 0;
        #[unroll]
        while element < VECTOR {
            values[group * VECTOR + element] = f32_from_bf16(words[element]);
            element += 1;
        }
        group += 1;
    }
    unsafe { normalize_vector(values, offset, gamma, beta, qkv, eps) };
}

/// # Safety
/// `qkv` owns `rows * 2304` BF16 values; each affine owns 768 values. All five
/// bases are 16-byte aligned, and affines do not overlap mutable QKV storage.
/// The host validates extents/aliases before launch. V is never read or written.
#[inline(always)]
pub(crate) unsafe fn qk_norm_768(
    qkv: *mut u16,
    q_gamma: *const u16,
    q_beta: *const u16,
    k_gamma: *const u16,
    k_beta: *const u16,
    rows: u32,
    eps: f32,
) {
    let tid = thread::threadIdx_x() as usize;
    let row = thread::blockIdx_x() as usize * 4 + tid / 32;
    if row >= rows as usize {
        return;
    }
    let offset = row * PACKED;
    let lane = tid & 31;
    unsafe {
        normalize_slice(qkv, q_gamma, q_beta, offset, lane, eps);
        normalize_slice(qkv, k_gamma, k_beta, offset + HIDDEN, lane, eps);
    }
}
