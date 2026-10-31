// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Actual cuda-oxide SIMT kernels for the native BERT executor.
//!
//! Tensor storage is BF16 (u16 bits); reductions and arithmetic use FP32.
//! The host owns every allocation and validates shapes before raw launches.
//! Row kernels launch exactly 256 threads; scalar kernels use a 1D grid.

use cuda_device::{DynamicSharedArray, SharedArray, convert, cuda_module, kernel, thread, warp};
mod norm384;

/// Interpret one BF16 storage element without numerical conversion error.
/// The upper16 bits are the exponent, sign and significant FP32 bits.
#[inline(always)]
fn f32_from_bf16(value: u16) -> f32 {
    f32::from_bits((value as u32) << 16)
}

/// Round FP32 to BF16 using the compiler's admitted native CUDA intrinsic.
/// Both halves receive the same value; the low half is its BF16 representation.
#[inline(always)]
fn bf16_from_f32(value: f32) -> u16 {
    convert::cvt_bf16x2_f32(value, value) as u16
}

/// Read one scalar from host-validated contiguous BF16 storage.
/// Kernels preserve ownership externally and never free these pointers.
#[inline(always)]
unsafe fn read(input: *const u16, index: usize) -> f32 {
    f32_from_bf16(unsafe { *input.add(index) })
}

/// Store one scalar, rounding only at the BF16 activation boundary.
/// Each kernel assigns disjoint output indices to different threads.
#[inline(always)]
unsafe fn write(output: *mut u16, index: usize, value: f32) {
    unsafe { *output.add(index) = bf16_from_f32(value) };
}

/// Sum all256 lanes using warp shuffles and eight shared warp totals.
/// Two CTA barriers keep shared scratch reuse safe across successive reductions.
#[inline(always)]
unsafe fn block_sum(value: f32, scratch: *mut f32) -> f32 {
    let tid = thread::threadIdx_x() as usize;
    let lane = tid & 31;
    let value = warp::reduce_sum_f32(value);
    if lane == 0 {
        unsafe { *scratch.add(tid >> 5) = value };
    }
    thread::sync_threads();
    let subtotal = if lane < 8 {
        unsafe { *scratch.add(lane) }
    } else {
        0.0
    };
    let total = warp::reduce_sum_f32(subtotal);
    thread::sync_threads();
    total
}

/// Maximum across all256 lanes, sharing the row kernel's eight-slot scratch.
/// All warps participate, including lanes whose local value is negative infinity.
#[inline(always)]
unsafe fn block_max(value: f32, scratch: *mut f32) -> f32 {
    let tid = thread::threadIdx_x() as usize;
    let lane = tid & 31;
    let value = warp::reduce_max_f32(value);
    if lane == 0 {
        unsafe { *scratch.add(tid >> 5) = value };
    }
    thread::sync_threads();
    let subtotal = if lane < 8 {
        unsafe { *scratch.add(lane) }
    } else {
        f32::NEG_INFINITY
    };
    let total = warp::reduce_max_f32(subtotal);
    thread::sync_threads();
    total
}

/// Reproduce a projected BF16 row plus optional projection bias and residual.
/// Linear+bias rounds to BF16 before the residual is added in FP32 for statistics.
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
        value += unsafe { read(residual, index) };
    }
    value
}

/// BERT's embedding recipe rounds word+type to BF16 first.
/// Position is then added in FP32 for the qualified biased LayerNorm statistics.
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
    hidden: usize,
) -> f32 {
    let token = unsafe { *ids.add(row) } as usize;
    let typ = unsafe { *types.add(row) } as usize;
    let pos = unsafe { *positions.add(row) } as usize;
    let pair =
        unsafe { read(word, token * hidden + column) + read(token_type, typ * hidden + column) };
    f32_from_bf16(bf16_from_f32(pair)) + unsafe { read(position, pos * hidden + column) }
}

#[cuda_module]
mod kernels {
    use super::*;

    /// Register-cached specialization, grid=(tokens,1,1), block=256.
    #[kernel]
    pub unsafe fn infinity_embedding_norm_384(
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
        unsafe {
            crate::norm384::embedding_norm_384(
                ids, positions, types, word, position, token_type, gamma, beta, out, tokens, eps,
            )
        };
    }

    /// Register-cached specialization, grid=(rows,1,1), block=256.
    #[kernel]
    pub unsafe fn infinity_layer_norm_384(
        input: *const u16,
        residual: *const u16,
        bias: *const u16,
        gamma: *const u16,
        beta: *const u16,
        out: *mut u16,
        rows: u32,
        eps: f32,
    ) {
        unsafe {
            crate::norm384::layer_norm_384(input, residual, bias, gamma, beta, out, rows, eps)
        };
    }

    /// Eight warps split token reduction for each 32-column output tile.
    /// Grid=(selected sequences,ceil(hidden/32),1), block=256.
    #[kernel]
    pub unsafe fn infinity_mean_pool(
        input: *const u16,
        cu_seqlens: *const u32,
        selected: *const u32,
        out: *mut f32,
        count: u32,
        hidden: u32,
    ) {
        let row = thread::blockIdx_x() as usize;
        if row >= count as usize {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let lane = tid & 31;
        let column = thread::blockIdx_y() as usize * 32 + lane;
        let sequence = unsafe { *selected.add(row) } as usize;
        let start = unsafe { *cu_seqlens.add(sequence) } as usize;
        let end = unsafe { *cu_seqlens.add(sequence + 1) } as usize;
        if end <= start {
            return;
        }
        let mut token = start + (tid >> 5);
        let mut total = 0.0;
        if column < hidden as usize {
            while token < end {
                total += unsafe { read(input, token * hidden as usize + column) };
                token += 8;
            }
        }
        static mut PARTIALS: SharedArray<f32, 256> = SharedArray::UNINIT;
        let scratch = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PARTIALS) };
        unsafe { *scratch.add(tid) = total };
        thread::sync_threads();
        if tid < 32 && column < hidden as usize {
            total = 0.0;
            let mut warp_index = 0;
            while warp_index < 8 {
                total += unsafe { *scratch.add(warp_index * 32 + lane) };
                warp_index += 1;
            }
            unsafe { *out.add(row * hidden as usize + column) = total / (end - start) as f32 };
        }
    }

    /// Fuse embedding gathers with affine LayerNorm, retaining FP32 position statistics.
    /// Grid=(tokens,1,1), block=(256,1,1); no dynamic shared memory is required.
    #[kernel]
    pub unsafe fn infinity_embedding_norm(
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
        let row = thread::blockIdx_x() as usize;
        if row >= tokens as usize || hidden == 0 {
            return;
        }
        static mut SCRATCH: SharedArray<f32, 8> = SharedArray::UNINIT;
        let scratch = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCRATCH) };
        let tid = thread::threadIdx_x() as usize;
        let width = hidden as usize;
        let mut column = tid;
        let mut sum = 0.0;
        while column < width {
            sum += unsafe {
                embedding_value(
                    ids, positions, types, word, position, token_type, row, column, width,
                )
            };
            column += 256;
        }
        let mean = unsafe { block_sum(sum, scratch) } / hidden as f32;
        let mut variance = 0.0;
        column = tid;
        while column < width {
            let centered = unsafe {
                embedding_value(
                    ids, positions, types, word, position, token_type, row, column, width,
                )
            } - mean;
            variance = cuda_device::float::fma_rn_f32(centered, centered, variance);
            column += 256;
        }
        let inv = 1.0
            / cuda_device::float::sqrt_rn_f32(
                unsafe { block_sum(variance, scratch) } / hidden as f32 + eps,
            );
        column = tid;
        while column < width {
            let centered = unsafe {
                embedding_value(
                    ids, positions, types, word, position, token_type, row, column, width,
                )
            } - mean;
            let value =
                centered * inv * unsafe { read(gamma, column) } + unsafe { read(beta, column) };
            unsafe { write(out, row * width + column, value) };
            column += 256;
        }
    }

    /// Affine LayerNorm over optional projection-bias/residual additions.
    /// Centered population variance and CTA barriers permit input/output aliasing.
    #[kernel]
    pub unsafe fn infinity_layer_norm(
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
        let row = thread::blockIdx_x() as usize;
        if row >= rows as usize || width == 0 {
            return;
        }
        static mut SCRATCH: SharedArray<f32, 8> = SharedArray::UNINIT;
        let scratch = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCRATCH) };
        let tid = thread::threadIdx_x() as usize;
        let offset = row * width as usize;
        let mut column = tid;
        let mut sum = 0.0;
        while column < width as usize {
            sum += unsafe { norm_value(input, residual, bias, offset + column, column) };
            column += 256;
        }
        let mean = unsafe { block_sum(sum, scratch) } / width as f32;
        column = tid;
        let mut variance = 0.0;
        while column < width as usize {
            let centered =
                unsafe { norm_value(input, residual, bias, offset + column, column) } - mean;
            variance = cuda_device::float::fma_rn_f32(centered, centered, variance);
            column += 256;
        }
        let inv = 1.0
            / cuda_device::float::sqrt_rn_f32(
                unsafe { block_sum(variance, scratch) } / width as f32 + eps,
            );
        column = tid;
        while column < width as usize {
            let centered =
                unsafe { norm_value(input, residual, bias, offset + column, column) } - mean;
            let value =
                centered * inv * unsafe { read(gamma, column) } + unsafe { read(beta, column) };
            unsafe { write(out, offset + column, value) };
            column += 256;
        }
    }

    /// Add FFN projection bias, apply exact erf GELU in FP32, and write BF16.
    /// One scalar per thread; in-place input/output is permitted.
    // Match the tanh GELU epilogue selected by the native LibTorch BERT embedding path.
    #[kernel]
    pub unsafe fn infinity_gelu_bias(
        input: *const u16,
        bias: *const u16,
        out: *mut u16,
        elements: u32,
        width: u32,
    ) {
        let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
        if index >= elements as usize || width == 0 {
            return;
        }
        let value = unsafe { read(input, index) + read(bias, index % width as usize) };
        unsafe {
            write(
                out,
                index,
                0.5 * value
                    * (1.0
                        + cuda_device::float::tanh_approx_f32(
                            0.7978845608028654 * value * (1.0 + 0.044715 * value * value),
                        )),
            )
        };
    }

    /// Split one fused [tokens,3*hidden] QKV projection and add its packed bias.
    /// Each thread writes corresponding disjoint scalars in the three [tokens,hidden] outputs.
    #[kernel]
    pub unsafe fn infinity_qkv(
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
            write(q, index, read(input, offset) + read(bias, column));
            write(
                k,
                index,
                read(input, offset + hidden) + read(bias, column + hidden),
            );
            write(
                v,
                index,
                read(input, offset + hidden * 2) + read(bias, column + hidden * 2),
            );
        }
    }

    /// Bidirectional ragged attention with FP32 scores and BF16 activation storage.
    /// Grid=(max_length,heads,batch), block256; dynamic shared bytes=max_length*4.
    #[kernel]
    pub unsafe fn infinity_attention(
        q: *const u16,
        k: *const u16,
        v: *const u16,
        cu_seqlens: *const u32,
        out: *mut u16,
        batch: u32,
        heads: u32,
        head_dim: u32,
        hidden: u32,
    ) {
        let sequence = thread::blockIdx_z() as usize;
        let head = thread::blockIdx_y() as usize;
        let query = thread::blockIdx_x() as usize;
        if sequence >= batch as usize || head >= heads as usize || head_dim == 0 || hidden == 0 {
            return;
        }
        let start = unsafe { *cu_seqlens.add(sequence) } as usize;
        let end = unsafe { *cu_seqlens.add(sequence + 1) } as usize;
        if end <= start || query >= end - start {
            return;
        }
        let length = end - start;
        let width = hidden as usize;
        let head_offset = head * head_dim as usize;
        let q_offset = (start + query) * width + head_offset;
        let scores = DynamicSharedArray::<f32>::get();
        static mut SCRATCH: SharedArray<f32, 8> = SharedArray::UNINIT;
        let scratch = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCRATCH) };
        let tid = thread::threadIdx_x() as usize;
        let scale = 1.0 / cuda_device::float::sqrt_rn_f32(head_dim as f32);
        let lane = tid & 31;
        let mut key = tid >> 5;
        let mut maximum = f32::NEG_INFINITY;
        while key < length {
            let k_offset = (start + key) * width + head_offset;
            let mut column = lane;
            let mut value = 0.0;
            while column < head_dim as usize {
                value = cuda_device::float::fma_rn_f32(
                    unsafe { read(q, q_offset + column) },
                    unsafe { read(k, k_offset + column) },
                    value,
                );
                column += 32;
            }
            value = warp::reduce_sum_f32(value) * scale;
            if lane == 0 {
                unsafe { *scores.add(key) = value };
                maximum = f32::max(maximum, value);
            }
            key += 8;
        }
        maximum = unsafe { block_max(maximum, scratch) };
        key = tid;
        let mut total = 0.0;
        while key < length {
            let value = libm::expf(unsafe { *scores.add(key) } - maximum);
            unsafe { *scores.add(key) = value };
            total += value;
            key += 256;
        }
        let inverse_sum = 1.0 / unsafe { block_sum(total, scratch) };
        key = tid;
        while key < length {
            unsafe { *scores.add(key) *= inverse_sum };
            key += 256;
        }
        thread::sync_threads();
        let mut column = tid;
        while column < head_dim as usize {
            let mut value = 0.0;
            key = 0;
            while key < length {
                value = cuda_device::float::fma_rn_f32(
                    unsafe { *scores.add(key) },
                    unsafe { read(v, (start + key) * width + head_offset + column) },
                    value,
                );
                key += 1;
            }
            unsafe { write(out, q_offset + column, value) };
            column += 256;
        }
    }

    /// Convert selected BF16 rows to public FP32 CLS, mean or last-token vectors.
    /// Mode0=CLS,1=mean,2=last; selected entries index validated nonempty sequences.
    #[kernel]
    pub unsafe fn infinity_pool(
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
                total += unsafe { read(input, token * hidden as usize + column) };
                token += 1;
            }
            total / (end - start) as f32
        } else {
            let token = if mode == 2 { end - 1 } else { start };
            unsafe { read(input, token * hidden as usize + column) }
        };
        unsafe { *out.add(index) = value };
    }
}

fn main() {}
