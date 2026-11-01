/*
 * MPNet H=768 additive embedding/residual BF16-rounded LayerNorm.
 *
 * Derived from the public PyTorch CUDA implementation in
 * aten/src/ATen/native/cuda/layer_norm_kernel.cu at commit
 * 5c4886908584029761b579af026dcfb627c84070 (vectorized_layer_norm_kernel /
 * compute_stats / cuWelfordOnlineSum / cuWelfordCombine). The full upstream
 * BSD-3-Clause notice is preserved in aten_embedding.rs and in
 * LICENSE-PyTorch-BSD3.
 *
 * Both kernels reuse the qualified word-only Welford helpers and reduction
 * order: vector width 4, 128 threads (four warps), six shared floats.
 */

use cuda_device::{SharedArray, float, thread, warp};

use crate::aten_embedding::{WelfordData, welford_combine, welford_online_sum};
use crate::{bf16_from_f32, f32_from_bf16, read, write};

const HIDDEN: usize = 768;
const VEC_SIZE: usize = 4;
const N_VECTORS: usize = HIDDEN / VEC_SIZE;
const BLOCK_X: usize = 32;
const BLOCK_Y: usize = 4;
const NUM_THREADS: usize = BLOCK_X * BLOCK_Y;
const SMEM_FLOATS: usize = BLOCK_Y * 3 / 2;

/// MPNet adds the already-offset position embedding to the word row in FP32
/// and rounds the sum to BF16 exactly once before collecting statistics.
#[inline(always)]
unsafe fn embedding_value(
    ids: *const u32,
    positions: *const u32,
    word: *const u16,
    position: *const u16,
    row: usize,
    column: usize,
) -> f32 {
    let token = unsafe { *ids.add(row) } as usize;
    let pos = unsafe { *positions.add(row) } as usize;
    let value =
        unsafe { read(word, token * HIDDEN + column) + read(position, pos * HIDDEN + column) };
    f32_from_bf16(bf16_from_f32(value))
}

/// Projected input plus optional bias, then optional residual. Each present
/// addition is rounded to BF16 before the next addition and the statistics.
#[inline(always)]
unsafe fn norm_value(
    input: *const u16,
    residual: *const u16,
    bias: *const u16,
    index: usize,
    column: usize,
) -> f32 {
    let mut value = unsafe { read(input, index) };
    if !bias.is_null() {
        value = f32_from_bf16(bf16_from_f32(value + unsafe { read(bias, column) }));
    }
    if !residual.is_null() {
        value = f32_from_bf16(bf16_from_f32(value + unsafe { read(residual, index) }));
    }
    value
}

#[inline(always)]
unsafe fn reduce_welford(wd: WelfordData, smem: *mut f32, lane: usize, warp: usize) -> WelfordData {
    let mut wd = wd;
    let mut offset = BLOCK_X / 2;
    while offset > 0 {
        let shuffled = WelfordData {
            mean: warp::shuffle_down_f32(wd.mean, offset as u32),
            sigma2: warp::shuffle_down_f32(wd.sigma2, offset as u32),
            count: warp::shuffle_down_f32(wd.count, offset as u32),
        };
        wd = welford_combine(wd, shuffled);
        offset /= 2;
    }

    let mut offset = BLOCK_Y / 2;
    while offset > 0 {
        if lane == 0 && warp >= offset && warp < 2 * offset {
            let write_y = warp - offset;
            unsafe { *smem.add(2 * write_y) = wd.mean };
            unsafe { *smem.add(2 * write_y + 1) = wd.sigma2 };
            unsafe { *smem.add(BLOCK_Y + write_y) = wd.count };
        }
        thread::sync_threads();

        if lane == 0 && warp < offset {
            let shuffled = WelfordData {
                mean: unsafe { *smem.add(2 * warp) },
                sigma2: unsafe { *smem.add(2 * warp + 1) },
                count: unsafe { *smem.add(BLOCK_Y + warp) },
            };
            wd = welford_combine(wd, shuffled);
        }
        thread::sync_threads();

        offset /= 2;
    }
    wd
}

#[inline(always)]
unsafe fn affine_write(
    out: *mut u16,
    index: usize,
    column: usize,
    x: f32,
    mean: f32,
    rstd: f32,
    gamma: *const u16,
    beta: *const u16,
) {
    let gamma_value = unsafe { read(gamma, column) };
    let beta_value = unsafe { read(beta, column) };
    let normalized = rstd * (x - mean);
    let result = float::fma_rn_f32(gamma_value, normalized, beta_value);
    unsafe { write(out, index, result) };
}

#[inline(always)]
pub(crate) unsafe fn embedding_norm_mpnet_768_aten(
    ids: *const u32,
    positions: *const u32,
    _types: *const u32,
    word: *const u16,
    position: *const u16,
    _token_type: *const u16,
    gamma: *const u16,
    beta: *const u16,
    out: *mut u16,
    rows: u32,
    width: u32,
    eps: f32,
) {
    if width != HIDDEN as u32 || rows == 0 {
        return;
    }

    let row = thread::blockIdx_x() as usize;
    if row >= rows as usize {
        return;
    }

    let tid = thread::threadIdx_x() as usize;
    let lane = tid % BLOCK_X;
    let warp = tid / BLOCK_X;

    static mut SCRATCH: SharedArray<f32, SMEM_FLOATS> = SharedArray::UNINIT;
    let smem = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCRATCH) };
    let out_row = row * HIDDEN;

    let mut wd = WelfordData::zero();
    let mut vector_index = tid;
    while vector_index < N_VECTORS {
        let base = vector_index * VEC_SIZE;
        let mut element = 0;
        while element < VEC_SIZE {
            let value =
                unsafe { embedding_value(ids, positions, word, position, row, base + element) };
            wd = welford_online_sum(value, wd);
            element += 1;
        }
        vector_index += NUM_THREADS;
    }

    let wd = unsafe { reduce_welford(wd, smem, lane, warp) };

    if lane == 0 && warp == 0 {
        unsafe { *smem = wd.mean };
        unsafe { *smem.add(1) = float::div_rn_f32(wd.sigma2, HIDDEN as f32) };
    }
    thread::sync_threads();

    let mean = unsafe { *smem };
    let variance = unsafe { *smem.add(1) };
    let rstd = float::rsqrt_approx_f32(variance + eps);

    let mut vector_index = tid;
    while vector_index < N_VECTORS {
        let column = vector_index * VEC_SIZE;
        let mut element = 0;
        while element < VEC_SIZE {
            let index = column + element;
            let x = unsafe { embedding_value(ids, positions, word, position, row, index) };
            unsafe {
                affine_write(out, out_row + index, index, x, mean, rstd, gamma, beta);
            }
            element += 1;
        }
        vector_index += NUM_THREADS;
    }
}

#[inline(always)]
pub(crate) unsafe fn layer_norm_mpnet_768_aten(
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
    if width != HIDDEN as u32 || rows == 0 {
        return;
    }

    let row = thread::blockIdx_x() as usize;
    if row >= rows as usize {
        return;
    }

    let tid = thread::threadIdx_x() as usize;
    let lane = tid % BLOCK_X;
    let warp = tid / BLOCK_X;

    static mut SCRATCH: SharedArray<f32, SMEM_FLOATS> = SharedArray::UNINIT;
    let smem = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCRATCH) };
    let offset = row * HIDDEN;

    let mut wd = WelfordData::zero();
    let mut vector_index = tid;
    while vector_index < N_VECTORS {
        let base = vector_index * VEC_SIZE;
        let mut element = 0;
        while element < VEC_SIZE {
            let column = base + element;
            let value = unsafe { norm_value(input, residual, bias, offset + column, column) };
            wd = welford_online_sum(value, wd);
            element += 1;
        }
        vector_index += NUM_THREADS;
    }

    let wd = unsafe { reduce_welford(wd, smem, lane, warp) };

    if lane == 0 && warp == 0 {
        unsafe { *smem = wd.mean };
        unsafe { *smem.add(1) = float::div_rn_f32(wd.sigma2, HIDDEN as f32) };
    }
    thread::sync_threads();

    let mean = unsafe { *smem };
    let variance = unsafe { *smem.add(1) };
    let rstd = float::rsqrt_approx_f32(variance + eps);

    let mut vector_index = tid;
    while vector_index < N_VECTORS {
        let column = vector_index * VEC_SIZE;
        let mut element = 0;
        while element < VEC_SIZE {
            let index = column + element;
            let x = unsafe { norm_value(input, residual, bias, offset + index, index) };
            unsafe {
                affine_write(out, offset + index, index, x, mean, rstd, gamma, beta);
            }
            element += 1;
        }
        vector_index += NUM_THREADS;
    }
}
