// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Actual cuda-oxide SIMT kernels for the native BERT executor.
//!
//! Tensor storage is BF16 (u16 bits); reductions and arithmetic use FP32.
//! The host owns every allocation and validates shapes before raw launches.
//! Row kernels launch exactly 256 threads; scalar kernels use a 1D grid.

use cuda_device::{convert, cuda_module, kernel, thread, warp, DynamicSharedArray, SharedArray};
mod aten_embedding;
mod classifier;
mod classifier_norm384;
mod deberta_math;
mod deberta_norm;
mod decoder;
mod dense;
mod encoder_activation;
mod encoder_f16;
mod encoder_prenorm;
mod gated;
mod gelu_exact;
mod gemma2;
mod jina_code_qk;
mod mpnet_norm;
mod nomic_embedding;
mod norm1024;
mod norm1024_exact_vector;
mod norm2048;
mod norm2048_vector;
mod qwen35_norm;
mod qwen35_norm_vector_memory;
mod qwen35_attention;
mod qwen35_copy;
mod packed_mean_bf16;
mod norm384;
mod norm512;
mod norm768;
mod norm768_vector;
mod qk_norm_rope_128;
mod mistral3;
mod mistral3_hostfix;
mod norm3072_exact_vector;
mod mistral3_rope128;
mod qk_norm_rope_256;
mod qwen35_moe;
mod qwen35vl;
mod qwen3vl;
mod readout;
mod rms1536;
mod rms1536_vector;
mod rms4096_vector;
mod rms5120_aten;
mod silu_pairs;
mod silu_vector8;

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
/// ROUND_RESIDUAL also rounds that residual sum before collecting statistics.
#[inline(always)]
unsafe fn norm_value<const ROUND_RESIDUAL: bool>(
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
        if ROUND_RESIDUAL {
            value = f32_from_bf16(bf16_from_f32(value));
        }
    }
    value
}

/// BERT's embedding recipe rounds word+type to BF16 first.
/// Position is then added in FP32 for the qualified biased LayerNorm statistics.
/// ROUND_POSITION rounds the position sum for materialized-add LayerNorm recipes.
#[inline(always)]
unsafe fn embedding_value<const ROUND_POSITION: bool>(
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
    let value =
        f32_from_bf16(bf16_from_f32(pair)) + unsafe { read(position, pos * hidden + column) };
    if ROUND_POSITION {
        f32_from_bf16(bf16_from_f32(value))
    } else {
        value
    }
}

#[inline(always)]
unsafe fn embedding_norm<const ROUND_POSITION: bool>(
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
            embedding_value::<ROUND_POSITION>(
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
            embedding_value::<ROUND_POSITION>(
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
            embedding_value::<ROUND_POSITION>(
                ids, positions, types, word, position, token_type, row, column, width,
            )
        } - mean;
        let value = centered * inv * unsafe { read(gamma, column) } + unsafe { read(beta, column) };
        unsafe { write(out, row * width + column, value) };
        column += 256;
    }
}

#[inline(always)]
unsafe fn layer_norm<const ROUND_RESIDUAL: bool>(
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
        sum +=
            unsafe { norm_value::<ROUND_RESIDUAL>(input, residual, bias, offset + column, column) };
        column += 256;
    }
    let mean = unsafe { block_sum(sum, scratch) } / width as f32;
    column = tid;
    let mut variance = 0.0;
    while column < width as usize {
        let centered =
            unsafe { norm_value::<ROUND_RESIDUAL>(input, residual, bias, offset + column, column) }
                - mean;
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
            unsafe { norm_value::<ROUND_RESIDUAL>(input, residual, bias, offset + column, column) }
                - mean;
        let value = centered * inv * unsafe { read(gamma, column) } + unsafe { read(beta, column) };
        unsafe { write(out, offset + column, value) };
        column += 256;
    }
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub unsafe fn infinity_deberta_unpack_scaled(
        packed: *const u16,
        q: *mut u16,
        k: *mut u16,
        v: *mut u16,
        elements: u32,
        width: u32,
        scale: f32,
    ) {
        unsafe { crate::deberta_math::unpack_scaled(packed, q, k, v, elements, width, scale) };
    }
    #[kernel]
    pub unsafe fn infinity_deberta_relative_transpose(
        input: *const u16,
        out: *mut u16,
        rows: u32,
        width: u32,
    ) {
        unsafe { crate::deberta_math::relative_transpose(input, out, rows, width) };
    }


    #[kernel]
    pub unsafe fn infinity_deberta_divide(input: *mut u16, elements: u32, scale: f32) {
        unsafe { crate::deberta_math::divide(input, elements, scale) };
    }
    #[kernel]
    pub unsafe fn infinity_deberta_signed_buckets(
        out: *mut i32,
        length: u32,
        span: u32,
        logarithmic: u32,
        denominator: f32,
    ) {
        unsafe { crate::deberta_math::signed_buckets(out, length, span, logarithmic, denominator) };
    }
    #[kernel]
    pub unsafe fn infinity_deberta_embedding_word_norm_768(
        ids: *const u32,
        word: *const u16,
        gamma: *const u16,
        beta: *const u16,
        out: *mut u16,
        rows: u32,
        _width: u32,
        eps: f32,
    ) {
        unsafe { crate::deberta_norm::word(ids, word, gamma, beta, out, rows, eps) };
    }
    #[kernel]
    pub unsafe fn infinity_deberta_layer_norm_768(
        input: *const u16,
        residual: *const u16,
        bias: *const u16,
        gamma: *const u16,
        beta: *const u16,
        out: *mut u16,
        rows: u32,
        _width: u32,
        eps: f32,
    ) {
        unsafe { crate::deberta_norm::rounded(input, residual, bias, gamma, beta, out, rows, eps) };
    }

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

    /// Native width-768 embedding plus affine LayerNorm.
    /// Grid=(ceil(tokens/4),1,1), block=(128,1,1); one warp per row.
    #[kernel]
    pub unsafe fn infinity_embedding_norm_768_exact(
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
        unsafe {
            crate::norm768::embedding_norm_768(
                ids, positions, types, word, position, token_type, gamma, beta, out, tokens,
                hidden, eps,
            )
        };
    }

    /// Native width-768 residual plus affine LayerNorm.
    /// Grid=(ceil(rows/4),1,1), block=(128,1,1); one warp per row.
    #[kernel]
    pub unsafe fn infinity_layer_norm_768_exact(
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
        unsafe {
            crate::norm768::layer_norm_768(
                input, residual, bias, gamma, beta, out, rows, width, eps,
            )
        };
    }

    /// Word/type unrounded F32 statistics, four rows per128-thread CTA.
    #[kernel]
    pub unsafe fn infinity_embedding_word_type_norm_768_exact(
        ids: *const u32,
        types: *const u32,
        word: *const u16,
        token_type: *const u16,
        gamma: *const u16,
        beta: *const u16,
        out: *mut u16,
        tokens: u32,
        eps: f32,
    ) {
        unsafe {
            crate::nomic_embedding::word_type_norm_768(
                ids, types, word, token_type, gamma, beta, out, tokens, eps,
            )
        };
    }

    /// Native width512 word/type Candle norm, grid ceil(T/4), block128.
    #[kernel]
    pub unsafe fn infinity_embedding_word_type_norm_512_exact(
        ids: *const u32,
        types: *const u32,
        word: *const u16,
        token_type: *const u16,
        gamma: *const u16,
        beta: *const u16,
        out: *mut u16,
        tokens: u32,
        eps: f32,
    ) {
        unsafe {
            crate::norm512::embedding_word_type_norm_512(
                ids, types, word, token_type, gamma, beta, out, tokens, eps,
            )
        };
    }

    /// Native width512 rounded residual norm, grid rows, block256.
    #[kernel]
    pub unsafe fn infinity_layer_norm_512_rounded_native(
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
        unsafe {
            crate::norm512::layer_norm_512_rounded(
                input, residual, bias, gamma, beta, out, rows, width, eps,
            )
        };
    }

    /// Vector-memory native width-768 embedding plus affine LayerNorm.
    /// Grid=(ceil(tokens/4),1,1), block=(128,1,1); one warp per row.
    #[kernel]
    pub unsafe fn infinity_embedding_norm_768_exact_vector(
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
        unsafe {
            crate::norm768_vector::embedding_norm_768_vector(
                ids, positions, types, word, position, token_type, gamma, beta, out, tokens,
                hidden, eps,
            )
        };
    }

    /// Vector-memory native width-768 residual plus affine LayerNorm.
    /// Grid=(ceil(rows/4),1,1), block=(128,1,1); one warp per row.
    #[kernel]
    pub unsafe fn infinity_layer_norm_768_exact_vector(
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
        unsafe {
            crate::norm768_vector::layer_norm_768_vector(
                input, residual, bias, gamma, beta, out, rows, width, eps,
            )
        };
    }

    /// JinaCode full-hidden Q/K affine norm in packed BF16 QKV storage.
    /// Grid=ceil(rows/4), block128; Q/K slices normalize independently, V stays.
    #[kernel]
    pub unsafe fn infinity_jina_code_qk_norm_768(
        qkv: *mut u16,
        q_gamma: *const u16,
        q_beta: *const u16,
        k_gamma: *const u16,
        k_beta: *const u16,
        rows: u32,
        eps: f32,
    ) {
        unsafe {
            crate::jina_code_qk::qk_norm_768(qkv, q_gamma, q_beta, k_gamma, k_beta, rows, eps)
        };
    }

    /// Native width-1024 rounded embedding plus affine LayerNorm.
    /// Grid=(tokens,1,1), block=(256,1,1); one row per CTA.
    #[kernel]
    pub unsafe fn infinity_embedding_norm_1024_rounded_native(
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
        unsafe {
            crate::norm1024::embedding_norm_1024_rounded(
                ids, positions, types, word, position, token_type, gamma, beta, out, tokens,
                hidden, eps,
            )
        };
    }

    /// Native width-1024 rounded residual plus affine LayerNorm.
    /// Grid=(rows,1,1), block=(256,1,1); one row per CTA.
    #[kernel]
    pub unsafe fn infinity_layer_norm_1024_rounded_native(
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
        unsafe {
            crate::norm1024::layer_norm_1024_rounded(
                input, residual, bias, gamma, beta, out, rows, width, eps,
            )
        };
    }

    /// Exact width-2048 BF16 RMSNorm.
    /// Grid=(ceil(rows/4),1,1), block=(128,1,1); one warp per row.
    #[kernel]
    pub unsafe fn infinity_rms_norm_2048_exact(
        input: *const u16,
        weight: *const u16,
        out: *mut u16,
        rows: u32,
        width: u32,
        eps: f32,
    ) {
        unsafe { crate::norm2048::exact_rms_norm_2048(input, weight, out, rows, width, eps) };
    }

    /// Exact width-2048 BF16 residual RMSNorm with a separate rounded BF16 sum.
    /// Grid=(ceil(rows/4),1,1), block=(128,1,1); one warp per row.
    #[kernel]
    pub unsafe fn infinity_add_rms_norm_2048_exact(
        input: *const u16,
        residual: *const u16,
        weight: *const u16,
        out: *mut u16,
        summed: *mut u16,
        rows: u32,
        width: u32,
        eps: f32,
    ) {
        unsafe {
            crate::norm2048::exact_add_rms_norm_2048(
                input, residual, weight, out, summed, rows, width, eps,
            )
        };
    }

    /// Exact width-2048 BF16 RMSNorm with aligned 128-bit transactions.
    /// Grid=(ceil(rows/4),1,1), block=(128,1,1); one warp per row.
    #[kernel]
    pub unsafe fn infinity_rms_norm_2048_vector(
        input: *const u16,
        weight: *const u16,
        out: *mut u16,
        rows: u32,
        width: u32,
        eps: f32,
    ) {
        unsafe {
            crate::norm2048_vector::exact_rms_norm_2048_vector(input, weight, out, rows, width, eps)
        };
    }

    /// Exact width-2048 BF16 residual RMSNorm with aligned 128-bit transactions.
    /// Grid=(ceil(rows/4),1,1), block=(128,1,1); one warp per row.
    #[kernel]
    pub unsafe fn infinity_add_rms_norm_2048_vector(
        input: *const u16,
        residual: *const u16,
        weight: *const u16,
        out: *mut u16,
        summed: *mut u16,
        rows: u32,
        width: u32,
        eps: f32,
    ) {
        unsafe {
            crate::norm2048_vector::exact_add_rms_norm_2048_vector(
                input, residual, weight, out, summed, rows, width, eps,
            )
        };
    }

    /// Exact width-1024 BF16 RMSNorm with aligned 128-bit transactions.
    /// Grid=(ceil(rows/4),1,1), block=(128,1,1); one warp per row.
    #[kernel]
    pub unsafe fn infinity_rms_norm_1024_exact_vector(
        input: *const u16,
        weight: *const u16,
        out: *mut u16,
        rows: u32,
        width: u32,
        eps: f32,
    ) {
        unsafe {
            crate::norm1024_exact_vector::exact_rms_norm_1024_vector(
                input, weight, out, rows, width, eps,
            )
        };
    }

    /// Exact width-1024 BF16 residual RMSNorm with aligned 128-bit transactions.
    /// Grid=(ceil(rows/4),1,1), block=(128,1,1); one warp per row.
    #[kernel]
    pub unsafe fn infinity_add_rms_norm_1024_exact_vector(
        input: *const u16,
        residual: *const u16,
        weight: *const u16,
        out: *mut u16,
        summed: *mut u16,
        rows: u32,
        width: u32,
        eps: f32,
    ) {
        unsafe {
            crate::norm1024_exact_vector::exact_add_rms_norm_1024_vector(
                input, residual, weight, out, summed, rows, width, eps,
            )
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
        unsafe {
            super::embedding_norm::<false>(
                ids, positions, types, word, position, token_type, gamma, beta, out, tokens,
                hidden, eps,
            )
        };
    }

    /// Round word+type, then +position to BF16 before LayerNorm statistics.
    /// A missing type table is represented by a host-owned zero table.
    /// Grid=(tokens,1,1), block=(256,1,1); ABI matches infinity_embedding_norm.
    #[kernel]
    pub unsafe fn infinity_embedding_norm_rounded(
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
        unsafe {
            super::embedding_norm::<true>(
                ids, positions, types, word, position, token_type, gamma, beta, out, tokens,
                hidden, eps,
            )
        };
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
        unsafe {
            super::layer_norm::<false>(input, residual, bias, gamma, beta, out, rows, width, eps)
        };
    }

    /// Round projection+bias, then +residual to BF16 before LayerNorm statistics.
    /// Grid=(rows,1,1), block=(256,1,1); ABI matches infinity_layer_norm.
    #[kernel]
    pub unsafe fn infinity_layer_norm_rounded(
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
        unsafe {
            super::layer_norm::<true>(input, residual, bias, gamma, beta, out, rows, width, eps)
        };
    }

    /// Prepared native phase tables; construction only, block256.
    #[kernel]
    pub unsafe fn infinity_base_rope_tables_64_bf16(
        inverse: *const f32,
        cosine: *mut f32,
        sine: *mut f32,
        positions: u32,
    ) {
        unsafe { crate::encoder_prenorm::base_rope_tables_bf16(inverse, cosine, sine, positions) };
    }
    /// Word-only biasless Candle width768; four rows per128-thread CTA.
    #[kernel]
    pub unsafe fn infinity_embedding_word_norm_768_candle(
        ids: *const u32,
        word: *const u16,
        gamma: *const u16,
        out: *mut u16,
        rows: u32,
        eps: f32,
    ) {
        unsafe {
            crate::encoder_prenorm::embedding_word_norm_768(ids, word, gamma, out, rows, eps)
        };
    }
    /// Rounded residual statistics; optional retained BF16 sum destination.
    #[kernel]
    pub unsafe fn infinity_rounded_residual_norm_768_candle(
        input: *const u16,
        residual: *const u16,
        gamma: *const u16,
        normalized: *mut u16,
        summed: *mut u16,
        rows: u32,
        eps: f32,
    ) {
        unsafe {
            crate::encoder_prenorm::rounded_residual_norm_768(
                input, residual, gamma, normalized, summed, rows, eps,
            )
        };
    }
    /// Rounded BF16 gated GELU with gate in the second packed half.
    /// Grid=ceil(rows*width/256), block=256; input and output must not alias.
    #[kernel]
    pub unsafe fn infinity_gated_gelu_up_first(
        input: *const u16,
        out: *mut u16,
        rows: u32,
        width: u32,
    ) {
        unsafe { crate::gated::gated_gelu_up_first(input, out, rows, width) };
    }

    // Derived from public PyTorch ATen layer_norm_kernel.cu at commit
    // 5c4886908584029761b579af026dcfb627c84070. The full BSD-3-Clause
    // notice is preserved in aten_embedding.rs.

    #[kernel]
    pub unsafe fn infinity_embedding_norm_word_768_aten(
        ids: *const u32,
        word: *const u16,
        gamma: *const u16,
        beta: *const u16,
        out: *mut u16,
        rows: u32,
        width: u32,
        eps: f32,
    ) {
        unsafe {
            crate::aten_embedding::embedding_norm_word_768_aten(
                ids, word, gamma, beta, out, rows, width, eps,
            )
        };
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

    /// BF16 RMSNorm; grid=(rows,1,1), block=256. See decoder.rs for rounding.
    #[kernel]
    pub unsafe fn infinity_rms_norm(
        input: *const u16,
        weight: *const u16,
        out: *mut u16,
        rows: u32,
        width: u32,
        eps: f32,
    ) {
        unsafe {
            crate::decoder::rms_norm::<false, false>(
                input,
                core::ptr::null(),
                weight,
                out,
                core::ptr::null_mut(),
                rows,
                width,
                eps,
            )
        };
    }

    /// BF16 RMSNorm; grid=(rows,1,1), block=256. See decoder.rs for rounding.
    #[kernel]
    pub unsafe fn infinity_rms_norm_rounded(
        input: *const u16,
        weight: *const u16,
        out: *mut u16,
        rows: u32,
        width: u32,
        eps: f32,
    ) {
        unsafe {
            crate::decoder::rms_norm::<false, true>(
                input,
                core::ptr::null(),
                weight,
                out,
                core::ptr::null_mut(),
                rows,
                width,
                eps,
            )
        };
    }

    /// BF16 RMSNorm; grid=(rows,1,1), block=256. See decoder.rs for rounding.
    #[kernel]
    pub unsafe fn infinity_add_rms_norm(
        input: *const u16,
        residual: *const u16,
        weight: *const u16,
        out: *mut u16,
        summed: *mut u16,
        rows: u32,
        width: u32,
        eps: f32,
    ) {
        unsafe {
            crate::decoder::rms_norm::<true, false>(
                input, residual, weight, out, summed, rows, width, eps,
            )
        };
    }

    /// Ordinary fixed-H1536 RMSNorm; grid=(rows,1,1), block=256.
    #[kernel]
    pub unsafe fn infinity_rms_norm_1536_cached(
        input: *const u16,
        weight: *const u16,
        out: *mut u16,
        rows: u32,
        width: u32,
        eps: f32,
    ) {
        unsafe {
            crate::rms1536::rms_norm_1536::<false>(
                input,
                core::ptr::null(),
                weight,
                out,
                core::ptr::null_mut(),
                rows,
                width,
                eps,
            )
        };
    }

    /// Ordinary residual fixed-H1536 RMSNorm; grid=(rows,1,1), block=256.
    #[kernel]
    pub unsafe fn infinity_add_rms_norm_1536_cached(
        input: *const u16,
        residual: *const u16,
        weight: *const u16,
        out: *mut u16,
        summed: *mut u16,
        rows: u32,
        width: u32,
        eps: f32,
    ) {
        unsafe {
            crate::rms1536::rms_norm_1536::<true>(
                input, residual, weight, out, summed, rows, width, eps,
            )
        };
    }

    /// Ordinary fixed-H1536 RMSNorm with aligned 128-bit vector I/O.
    /// Grid=(rows,1,1), block=256; arithmetic matches the cached scalar export.
    #[kernel]
    pub unsafe fn infinity_rms_norm_1536_vector(
        input: *const u16,
        weight: *const u16,
        out: *mut u16,
        rows: u32,
        width: u32,
        eps: f32,
    ) {
        unsafe {
            crate::rms1536_vector::rms_norm_1536_vector::<false>(
                input,
                core::ptr::null(),
                weight,
                out,
                core::ptr::null_mut(),
                rows,
                width,
                eps,
            )
        };
    }

    /// Ordinary residual fixed-H1536 RMSNorm with aligned 128-bit vector I/O.
    /// Grid=(rows,1,1), block=256; arithmetic matches the cached scalar export.
    #[kernel]
    pub unsafe fn infinity_add_rms_norm_1536_vector(
        input: *const u16,
        residual: *const u16,
        weight: *const u16,
        out: *mut u16,
        summed: *mut u16,
        rows: u32,
        width: u32,
        eps: f32,
    ) {
        unsafe {
            crate::rms1536_vector::rms_norm_1536_vector::<true>(
                input, residual, weight, out, summed, rows, width, eps,
            )
        };
    }

    /// Ordinary fixed-H4096 RMSNorm with aligned 128-bit vector I/O.
    /// Grid=(rows,1,1), block=256; arithmetic matches the generic ordinary export.
    #[kernel]
    pub unsafe fn infinity_rms_norm_4096_vector(
        input: *const u16,
        weight: *const u16,
        out: *mut u16,
        rows: u32,
        width: u32,
        eps: f32,
    ) {
        unsafe {
            crate::rms4096_vector::rms_norm_4096_vector::<false>(
                input,
                core::ptr::null(),
                weight,
                out,
                core::ptr::null_mut(),
                rows,
                width,
                eps,
            )
        };
    }

    /// Ordinary exact-H5120 ATen RMSNorm. Grid=(rows,1,1), block=128 (or 32x4).
    /// One CTA per row; this export is separate from the generic and add paths.
    #[kernel]
    pub unsafe fn infinity_rms_norm_5120_aten(
        input: *const u16,
        weight: *const u16,
        out: *mut u16,
        rows: u32,
        width: u32,
        eps: f32,
    ) {
        unsafe {
            crate::rms5120_aten::rms_norm_5120_aten(input, weight, out, rows, width, eps)
        }
    }

    /// Ordinary residual fixed-H4096 RMSNorm with aligned 128-bit vector I/O.
    /// Grid=(rows,1,1), block=256; arithmetic matches the generic ordinary export.
    #[kernel]
    pub unsafe fn infinity_add_rms_norm_4096_vector(
        input: *const u16,
        residual: *const u16,
        weight: *const u16,
        out: *mut u16,
        summed: *mut u16,
        rows: u32,
        width: u32,
        eps: f32,
    ) {
        unsafe {
            crate::rms4096_vector::rms_norm_4096_vector::<true>(
                input, residual, weight, out, summed, rows, width, eps,
            )
        };
    }

    /// BF16 RMSNorm; grid=(rows,1,1), block=256. See decoder.rs for rounding.
    #[kernel]
    pub unsafe fn infinity_add_rms_norm_rounded(
        input: *const u16,
        residual: *const u16,
        weight: *const u16,
        out: *mut u16,
        summed: *mut u16,
        rows: u32,
        width: u32,
        eps: f32,
    ) {
        unsafe {
            crate::decoder::rms_norm::<true, true>(
                input, residual, weight, out, summed, rows, width, eps,
            )
        };
    }

    /// In-place Q/K rotary pairs; block=256, one scalar thread per pair.
    #[kernel]
    pub unsafe fn infinity_rope_neox(
        q: *mut u16,
        k: *mut u16,
        cosine: *const f32,
        sine: *const f32,
        positions: *const u32,
        tokens: u32,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        rotary_dim: u32,
        q_stride: u64,
        k_stride: u64,
        table_stride: u32,
    ) {
        unsafe {
            crate::decoder::rope::<false, false>(
                q,
                k,
                cosine,
                sine,
                positions,
                tokens,
                q_heads,
                kv_heads,
                head_dim,
                rotary_dim,
                q_stride,
                k_stride,
                table_stride,
            )
        };
    }

    /// In-place Q/K rotary pairs; block=256, one scalar thread per pair.
    #[kernel]
    pub unsafe fn infinity_rope_neox_hfma(
        q: *mut u16,
        k: *mut u16,
        cosine: *const f32,
        sine: *const f32,
        positions: *const u32,
        tokens: u32,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        rotary_dim: u32,
        q_stride: u64,
        k_stride: u64,
        table_stride: u32,
    ) {
        unsafe {
            crate::decoder::rope::<false, true>(
                q,
                k,
                cosine,
                sine,
                positions,
                tokens,
                q_heads,
                kv_heads,
                head_dim,
                rotary_dim,
                q_stride,
                k_stride,
                table_stride,
            )
        };
    }

    /// In-place Q/K rotary pairs; block=256, one scalar thread per pair.
    #[kernel]
    pub unsafe fn infinity_rope_interleaved(
        q: *mut u16,
        k: *mut u16,
        cosine: *const f32,
        sine: *const f32,
        positions: *const u32,
        tokens: u32,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        rotary_dim: u32,
        q_stride: u64,
        k_stride: u64,
        table_stride: u32,
    ) {
        unsafe {
            crate::decoder::rope::<true, false>(
                q,
                k,
                cosine,
                sine,
                positions,
                tokens,
                q_heads,
                kv_heads,
                head_dim,
                rotary_dim,
                q_stride,
                k_stride,
                table_stride,
            )
        };
    }

    /// In-place Q/K rotary pairs; block=256, one scalar thread per pair.
    #[kernel]
    pub unsafe fn infinity_rope_interleaved_hfma(
        q: *mut u16,
        k: *mut u16,
        cosine: *const f32,
        sine: *const f32,
        positions: *const u32,
        tokens: u32,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        rotary_dim: u32,
        q_stride: u64,
        k_stride: u64,
        table_stride: u32,
    ) {
        unsafe {
            crate::decoder::rope::<true, true>(
                q,
                k,
                cosine,
                sine,
                positions,
                tokens,
                q_heads,
                kv_heads,
                head_dim,
                rotary_dim,
                q_stride,
                k_stride,
                table_stride,
            )
        };
    }

    /// In-place fused Q/K RMSNorm and NeoX rotary for head width 128.
    /// Grid=(ceil(tokens*(qheads+kvheads)/8),1,1), block=(128,1,1).
    #[kernel]
    pub unsafe fn infinity_qk_norm_rope_128_bf16(
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
        unsafe {
            crate::qk_norm_rope_128::qk_norm_rope_128(
                q,
                k,
                qgamma,
                kgamma,
                cosine,
                sine,
                positions,
                tokens,
                qheads,
                kvheads,
                qstride,
                kstride,
                table_stride,
                eps,
            )
        };
    }

    /// Gate/up activation; grid=ceil(rows*width/256), block=256.
    #[kernel]
    pub unsafe fn infinity_silu_mul(input: *const u16, out: *mut u16, rows: u32, width: u32) {
        unsafe { crate::decoder::silu_mul::<true>(input, out, rows, width) };
    }

    /// Gate/up activation; grid=ceil(rows*width/256), block=256.
    #[kernel]
    pub unsafe fn infinity_silu_mul_f32(input: *const u16, out: *mut u16, rows: u32, width: u32) {
        unsafe { crate::decoder::silu_mul::<false>(input, out, rows, width) };
    }

    /// Rounded BF16 paired gate/up activation; grid=ceil(rows*ceil(width/2)/256).
    #[kernel]
    pub unsafe fn infinity_silu_mul_bf16_pairs(
        input: *const u16,
        out: *mut u16,
        rows: u32,
        width: u32,
    ) {
        unsafe { crate::silu_pairs::silu_mul_bf16_pairs::<false>(input, out, rows, width) };
    }

    /// Native BF16 division recipe; ABI/grid match infinity_silu_mul_bf16_pairs.
    #[kernel]
    pub unsafe fn infinity_silu_mul_bf16_pairs_native(
        input: *const u16,
        out: *mut u16,
        rows: u32,
        width: u32,
    ) {
        unsafe { crate::silu_pairs::silu_mul_bf16_pairs::<true>(input, out, rows, width) };
    }

    /// Native BF16 SiLU; eight values/thread, grid=ceil(rows*ceil(width/8)/256).
    #[kernel]
    pub unsafe fn infinity_silu_mul_bf16_vector8_native(
        input: *const u16,
        out: *mut u16,
        rows: u32,
        width: u32,
    ) {
        unsafe { crate::silu_vector8::silu_mul_bf16_vector8_native(input, out, rows, width) };
    }

    /// Exact native classifier H384 embedding norm; 4 rows/128-thread CTA.
    #[kernel]
    pub unsafe fn infinity_classifier_embedding_norm_384_exact(
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
        unsafe {
            crate::classifier_norm384::embedding_norm_384_exact(
                ids, positions, types, word, position, token_type, gamma, beta, out, tokens,
                hidden, eps,
            )
        };
    }

    /// Exact native classifier H384 residual norm; 4 rows/128-thread CTA.
    #[kernel]
    pub unsafe fn infinity_classifier_layer_norm_384_exact(
        input: *const u16,
        residual: *const u16,
        bias: *const u16,
        gamma: *const u16,
        beta: *const u16,
        out: *mut u16,
        rows: u32,
        hidden: u32,
        eps: f32,
    ) {
        unsafe {
            crate::classifier_norm384::layer_norm_384_exact(
                input, residual, bias, gamma, beta, out, rows, hidden, eps,
            )
        };
    }

    /// Word-only BF16 gather; grid=ceil(tokens*width/256), block=256.
    #[kernel]
    pub unsafe fn infinity_embedding_gather(
        ids: *const u32,
        table: *const u16,
        out: *mut u16,
        tokens: u32,
        width: u32,
    ) {
        unsafe { crate::decoder::embedding_gather::<false>(ids, table, out, tokens, width, 1.0) };
    }

    /// Word gather with one FP32 scale multiply and BF16 output; no normalization.
    #[kernel]
    pub unsafe fn infinity_embedding_gather_scaled(
        ids: *const u32,
        table: *const u16,
        out: *mut u16,
        tokens: u32,
        width: u32,
        scale: f32,
    ) {
        unsafe { crate::decoder::embedding_gather::<true>(ids, table, out, tokens, width, scale) };
    }

    /// Gather the same CLS or last rows from two independent-width BF16 tensors.
    /// Grid=ceil(count*(first_width+second_width)/256), block=256; bit copies.
    #[kernel]
    pub unsafe fn infinity_selected_rows(
        first: *const u16,
        second: *const u16,
        cu_seqlens: *const u32,
        selected: *const u32,
        first_out: *mut u16,
        second_out: *mut u16,
        count: u32,
        first_width: u32,
        second_width: u32,
        mode: u32,
    ) {
        unsafe {
            crate::readout::selected_rows(
                first,
                second,
                cu_seqlens,
                selected,
                first_out,
                second_out,
                count,
                first_width,
                second_width,
                mode,
            )
        };
    }

    /// Convert contiguous FP32 to BF16; grid=ceil(count/256), block=256.
    #[kernel]
    pub unsafe fn infinity_f32_to_bf16(input: *const f32, out: *mut u16, count: u32) {
        unsafe { crate::dense::f32_to_bf16(input, out, count) };
    }

    /// Apply float Tanh to BF16 storage in place; grid=ceil(count/256), block=256.
    #[kernel]
    pub unsafe fn infinity_dense_tanh_bf16_inplace(input: *mut u16, count: u32) {
        unsafe { crate::dense::tanh_bf16_inplace(input, count) };
    }

    /// Seed every logical BF16 output element from the bias vector.
    /// Grid=ceil(rows*width/256), block=256.
    #[kernel]
    pub unsafe fn infinity_classifier_bias_bf16(
        bias: *const u16,
        out: *mut u16,
        rows: u32,
        width: u32,
    ) {
        unsafe { crate::classifier::bias_seed_bf16(bias, out, rows, width) };
    }

    /// Copy each sequence's first packed BF16 row into contiguous output rows.
    /// Grid=ceil(rows*width/256), block=256; exact 16-bit copies only.
    #[kernel]
    pub unsafe fn infinity_classifier_gather_first_bf16(
        source: *const u16,
        bounds: *const u32,
        out: *mut u16,
        rows: u32,
        width: u32,
    ) {
        unsafe { crate::classifier::gather_first_rows_bf16(source, bounds, out, rows, width) };
    }

    /// Widen selected BF16 rows to FP32; grid=ceil(count*width/256).
    /// A null selection keeps canonical row order.
    #[kernel]
    pub unsafe fn infinity_bf16_to_f32(
        input: *const u16,
        out: *mut f32,
        selected: *const u32,
        count: u32,
        width: u32,
    ) {
        unsafe { crate::readout::bf16_to_f32(input, out, selected, count, width) };
    }

    /// Native H384 F16 embedding gather plus affine LayerNorm.
    /// Grid=(ceil(tokens/4),1,1), block=(128,1,1); one warp per row.
    #[kernel]
    pub unsafe fn infinity_embedding_norm_384_f16(
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
            crate::encoder_f16::embedding_norm_384_f16(
                ids, positions, types, word, position, token_type, gamma, beta, out, tokens, eps,
            )
        };
    }

    /// Native H384 F16 residual plus affine LayerNorm.
    /// Grid=(ceil(rows/4),1,1), block=(128,1,1); one warp per row.
    #[kernel]
    pub unsafe fn infinity_layer_norm_384_f16(
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
            crate::encoder_f16::layer_norm_384_f16(
                input, residual, bias, gamma, beta, out, rows, eps,
            )
        };
    }

    /// Native H384 F16 plain affine LayerNorm without a residual.
    /// Grid=(ceil(rows/4),1,1), block=(128,1,1); one warp per row.
    #[kernel]
    pub unsafe fn infinity_plain_norm_384_f16(
        input: *const u16,
        gamma: *const u16,
        beta: *const u16,
        out: *mut u16,
        rows: u32,
        eps: f32,
    ) {
        unsafe { crate::encoder_f16::plain_norm_384_f16(input, gamma, beta, out, rows, eps) };
    }

    /// Actual spans F16 mean pooling; grid=(selected,ceil(hidden/8)), block=256.
    #[kernel]
    pub unsafe fn infinity_mean_pool_f16(
        input: *const u16,
        cu_seqlens: *const u32,
        selected: *const u32,
        out: *mut f32,
        count: u32,
        hidden: u32,
    ) {
        unsafe {
            crate::encoder_f16::mean_pool_384_f16(input, cu_seqlens, selected, out, count, hidden)
        };
    }

    /// F16 CLS, mean or last-token selection widened to FP32.
    #[kernel]
    pub unsafe fn infinity_pool_f16(
        input: *const u16,
        cu_seqlens: *const u32,
        selected: *const u32,
        out: *mut f32,
        count: u32,
        hidden: u32,
        mode: u32,
    ) {
        unsafe {
            crate::encoder_f16::pool_f16(input, cu_seqlens, selected, out, count, hidden, mode)
        };
    }

    /// Split one fused F16 QKV projection and add its packed bias.
    #[kernel]
    pub unsafe fn infinity_qkv_f16(
        input: *const u16,
        bias: *const u16,
        q: *mut u16,
        k: *mut u16,
        v: *mut u16,
        elements: u32,
        width: u32,
    ) {
        unsafe { crate::encoder_f16::qkv_f16(input, bias, q, k, v, elements, width) };
    }

    /// Widen selected F16 rows to FP32; grid=ceil(count*width/256).
    #[kernel]
    pub unsafe fn infinity_f16_to_f32(
        input: *const u16,
        out: *mut f32,
        selected: *const u32,
        count: u32,
        width: u32,
    ) {
        unsafe { crate::encoder_f16::f16_to_f32(input, out, selected, count, width) };
    }
    #[kernel]
    pub unsafe fn infinity_embedding_norm_mpnet_768_aten(
        ids: *const u32,
        positions: *const u32,
        types: *const u32,
        word: *const u16,
        position: *const u16,
        token_type: *const u16,
        gamma: *const u16,
        beta: *const u16,
        out: *mut u16,
        rows: u32,
        width: u32,
        eps: f32,
    ) {
        unsafe {
            crate::mpnet_norm::embedding_norm_mpnet_768_aten(
                ids, positions, types, word, position, token_type, gamma, beta, out, rows, width,
                eps,
            )
        };
    }

    #[kernel]
    pub unsafe fn infinity_layer_norm_mpnet_768_aten(
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
        unsafe {
            crate::mpnet_norm::layer_norm_mpnet_768_aten(
                input, residual, bias, gamma, beta, out, rows, width, eps,
            )
        };
    }

    #[kernel]
    pub unsafe fn infinity_gelu_exact(input: *mut u16, elements: u32) {
        unsafe { crate::gelu_exact::gelu_exact(input, elements) };
    }

    /// Optional BF16 pointwise tanh-GELU; null bias activates the input as-is.
    #[kernel]
    pub unsafe fn infinity_encoder_activation_gelu_tanh_bf16(
        input: *const u16,
        bias: *const u16,
        out: *mut u16,
        rows: u32,
        width: u32,
        input_stride: u32,
        output_stride: u32,
    ) {
        unsafe {
            crate::encoder_activation::activate::<0>(
                input,
                bias,
                out,
                rows,
                width,
                input_stride,
                output_stride,
            )
        };
    }

    /// Optional BF16 pointwise ReLU; null bias activates the input as-is.
    #[kernel]
    pub unsafe fn infinity_encoder_activation_relu_bf16(
        input: *const u16,
        bias: *const u16,
        out: *mut u16,
        rows: u32,
        width: u32,
        input_stride: u32,
        output_stride: u32,
    ) {
        unsafe {
            crate::encoder_activation::activate::<1>(
                input,
                bias,
                out,
                rows,
                width,
                input_stride,
                output_stride,
            )
        };
    }

    /// Optional BF16 pointwise SiLU; null bias activates the input as-is.
    #[kernel]
    pub unsafe fn infinity_encoder_activation_silu_bf16(
        input: *const u16,
        bias: *const u16,
        out: *mut u16,
        rows: u32,
        width: u32,
        input_stride: u32,
        output_stride: u32,
    ) {
        unsafe {
            crate::encoder_activation::activate::<2>(
                input,
                bias,
                out,
                rows,
                width,
                input_stride,
                output_stride,
            )
        };
    }

    /// Optional BF16 pointwise tanh; null bias activates the input as-is.
    #[kernel]
    pub unsafe fn infinity_encoder_activation_tanh_bf16(
        input: *const u16,
        bias: *const u16,
        out: *mut u16,
        rows: u32,
        width: u32,
        input_stride: u32,
        output_stride: u32,
    ) {
        unsafe {
            crate::encoder_activation::activate::<3>(
                input,
                bias,
                out,
                rows,
                width,
                input_stride,
                output_stride,
            )
        };
    }

    /// EmbeddingGemma2 DivSqrt norm with F32 scale and no +1.
    #[kernel]
    pub unsafe fn infinity_gemma2_norm_bf16_f32scale(
        input: *const u16,
        scale: *const f32,
        out: *mut u16,
        rows: u32,
        width: u32,
        heads: u32,
        token_stride: u64,
        eps: f32,
    ) {
        unsafe {
            crate::gemma2::norm_bf16_f32scale(
                input,
                scale,
                out,
                rows,
                width,
                heads,
                token_stride,
                eps,
            )
        };
    }

    /// EmbeddingGemma2 residual DivSqrt norm; statistics from input only.
    #[kernel]
    pub unsafe fn infinity_gemma2_residual_norm_bf16_f32scale(
        input: *const u16,
        residual: *const u16,
        scale: *const f32,
        out: *mut u16,
        rows: u32,
        width: u32,
        eps: f32,
    ) {
        unsafe {
            crate::gemma2::residual_norm_bf16_f32scale(
                input, residual, scale, out, rows, width, eps,
            )
        };
    }

    /// EmbeddingGemma2 unit V norm (F32 ones scale).
    #[kernel]
    pub unsafe fn infinity_gemma2_unit_norm_bf16(
        input: *const u16,
        out: *mut u16,
        rows: u32,
        width: u32,
        heads: u32,
        token_stride: u64,
        eps: f32,
    ) {
        unsafe { crate::gemma2::unit_norm_bf16(input, out, rows, width, heads, token_stride, eps) };
    }

    /// Native gate-first packed BF16 GELU-tanh.
    #[kernel]
    pub unsafe fn infinity_gemma2_gated_gelu_bf16(
        input: *const u16,
        out: *mut u16,
        rows: u32,
        width: u32,
    ) {
        unsafe { crate::gemma2::gated_gelu_bf16(input, out, rows, width) };
    }

    /// Precise GELU on a PLE gate row multiplied by the token's PLE slice.
    #[kernel]
    pub unsafe fn infinity_gemma2_ple_gate_bf16(
        gate: *const u16,
        ple: *const u16,
        out: *mut u16,
        rows: u32,
        width: u32,
        layers: u32,
        layer: u32,
    ) {
        unsafe { crate::gemma2::ple_gate_bf16(gate, ple, out, rows, width, layers, layer) };
    }

    /// BF16 pointwise multiply by one F32 value holding an exact BF16 scalar.
    #[kernel]
    pub unsafe fn infinity_gemma2_scale_bf16(
        input: *const u16,
        out: *mut u16,
        count: u32,
        scale: f32,
    ) {
        unsafe { crate::gemma2::scale_bf16(input, out, count, scale) };
    }

    /// Cold BF16-rounded rotary table initializer.
    #[kernel]
    pub unsafe fn infinity_gemma2_rope_tables_bf16(
        inverse: *const f32,
        cosine: *mut f32,
        sine: *mut f32,
        positions: u32,
        pairs: u32,
    ) {
        unsafe { crate::gemma2::rope_tables_bf16(inverse, cosine, sine, positions, pairs) };
    }

    /// Native packed BF16 mean pooling widened to F32.
    #[kernel]
    pub unsafe fn infinity_gemma2_mean_pool_bf16(
        input: *const u16,
        cu_seqlens: *const u32,
        selected: *const u32,
        out: *mut f32,
        count: u32,
        width: u32,
    ) {
        unsafe { crate::gemma2::mean_pool_bf16(input, cu_seqlens, selected, out, count, width) };
    }
    /// Native model-dtype BF16 mean, widened to F32.
    /// Grid=(count,ceil(hidden/8),1), block=256.
    #[kernel]
    pub unsafe fn infinity_mean_pool_bf16_native(
        input: *const u16,
        cu_seqlens: *const u32,
        selected: *const u32,
        out: *mut f32,
        count: u32,
        hidden: u32,
    ) {
        unsafe {
            crate::packed_mean_bf16::packed_mean_pool_bf16(
                input, cu_seqlens, selected, out, count, hidden,
            )
        };
    }
    /// Exact Qwen3.5 offset RMS, one CTA per flattened head row.
    #[kernel]
    pub unsafe fn infinity_qwen35_norm_vector_bf16(
        x: *const u16,
        gamma: *const u16,
        out: *mut u16,
        rows: u32,
        width: u32,
        heads: u32,
        stride0: u64,
        stride1: u64,
        eps: f32,
    ) {
        unsafe {
            crate::qwen35_norm::norm_vector(
                x, gamma, out, rows, width, heads, stride0, stride1, eps,
            )
        };
    }

    /// Exact rounded-residual Qwen3.5 offset RMS, one CTA per row.
    #[kernel]
    pub unsafe fn infinity_qwen35_residual_norm_vector_bf16(
        x: *const u16,
        addition: *const u16,
        gamma: *const u16,
        out: *mut u16,
        updated: *mut u16,
        rows: u32,
        width: u32,
        eps: f32,
    ) {
        unsafe {
            crate::qwen35_norm::residual_norm_vector(
                x, addition, gamma, out, updated, rows, width, eps,
            )
        };
    }

    /// Exact Qwen3.5 offset RMS, one CTA per flattened head row.

    #[kernel]
    pub unsafe fn infinity_qwen35_norm_vector_memory_bf16(
        x: *const u16,
        gamma: *const u16,
        out: *mut u16,
        rows: u32,
        width: u32,
        heads: u32,
        stride0: u64,
        stride1: u64,
        eps: f32,
    ) {
        unsafe {
            crate::qwen35_norm_vector_memory::norm_vector_memory(
                x, gamma, out, rows, width, heads, stride0, stride1, eps,
            )
        };
    }

    #[kernel]
    pub unsafe fn infinity_qwen35_residual_norm_vector_memory_bf16(
        x: *const u16,
        addition: *const u16,
        gamma: *const u16,
        out: *mut u16,
        updated: *mut u16,
        rows: u32,
        width: u32,
        eps: f32,
    ) {
        unsafe {
            crate::qwen35_norm_vector_memory::residual_norm_vector_memory(
                x, addition, gamma, out, updated, rows, width, eps,
            )
        };
    }

    #[kernel]
    pub unsafe fn infinity_qwen35_norm_scalar_bf16(
        x: *const u16,
        gamma: *const u16,
        out: *mut u16,
        rows: u32,
        width: u32,
        heads: u32,
        stride0: u64,
        stride1: u64,
        eps: f32,
    ) {
        unsafe {
            crate::qwen35_norm::norm_scalar(
                x, gamma, out, rows, width, heads, stride0, stride1, eps,
            )
        };
    }

    /// Exact rounded-residual Qwen3.5 offset RMS, one CTA per row.
    #[kernel]
    pub unsafe fn infinity_qwen35_residual_norm_scalar_bf16(
        x: *const u16,
        addition: *const u16,
        gamma: *const u16,
        out: *mut u16,
        updated: *mut u16,
        rows: u32,
        width: u32,
        eps: f32,
    ) {
        unsafe {
            crate::qwen35_norm::residual_norm_scalar(
                x, addition, gamma, out, updated, rows, width, eps,
            )
        };
    }

    /// Qwen3.5 staged partial RoPE; four head rows per 128-thread CTA.
    #[kernel]
    pub unsafe fn infinity_qwen35_partial_rope_256_bf16(
        q: *mut u16,
        k: *mut u16,
        cosine: *const f32,
        sine: *const f32,
        positions: *const u32,
        tokens: u32,
        qheads: u32,
        kvheads: u32,
        rotary: u32,
        qstride: u64,
        kstride: u64,
        table_stride: u32,
    ) {
        unsafe {
            crate::qwen35_attention::partial_rope_256_bf16(
                q,
                k,
                cosine,
                sine,
                positions,
                tokens,
                qheads,
                kvheads,
                rotary,
                qstride,
                kstride,
                table_stride,
            )
        };
    }

    /// Dense BF16 output gate; grid=ceil(tokens*heads*256/256), block256.
    #[kernel]
    pub unsafe fn infinity_qwen35_attention_gate_bf16(
        attention: *const u16,
        query_gate: *const u16,
        out: *mut u16,
        tokens: u32,
        heads: u32,
        gate_stride: u64,
    ) {
        unsafe {
            crate::qwen35_attention::attention_gate_bf16(
                attention,
                query_gate,
                out,
                tokens,
                heads,
                gate_stride,
            )
        };
    }

    /// Cold BF16-rounded RoPE coefficients; grid=ceil(positions*pairs/256), block256.
    #[kernel]
    pub unsafe fn infinity_qwen35_rope_tables_bf16(
        inverse: *const f32,
        cosine: *mut f32,
        sine: *mut f32,
        positions: u32,
        pairs: u32,
    ) {
        unsafe {
            crate::qwen35_attention::rope_tables_bf16(inverse, cosine, sine, positions, pairs)
        };
    }
    /// Exact bit-copy concatenation, grid=ceil(T*2*heads/256), block256.
    #[kernel]
    pub unsafe fn infinity_qwen35_join_ab_bf16(a: *const u16, b: *const u16, out: *mut u16, tokens: u32, heads: u32) {
        unsafe { crate::qwen35_copy::join_ab(a, b, out, tokens, heads) };
    }

    /// In-place fused Q/K RMSNorm and NeoX rotary for head width 256.
    /// This is the ordinary Native d!=128 recipe: ordinary contiguous RMSNorm
    /// followed by separate BF16 rotary products/adds. It is not the width-128
    /// Candle HFMA path.
    /// Grid=(tokens*(qheads+kvheads),1,1), block=(256,1,1).
    /// qstride/kstride are BF16-element token strides at least heads*256.
    /// table_stride is the F32 cosine/sine row stride and must be at least 128.
    /// V is untouched. Caller validates allocation extent, alias disjointness,
    /// positions bounds, and the exact geometry.
    #[kernel]
    pub unsafe fn infinity_qk_norm_rope_256_bf16(
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
        unsafe {
            crate::qk_norm_rope_256::qk256_qk_norm_rope_256(
                q,
                k,
                qgamma,
                kgamma,
                cosine,
                sine,
                positions,
                tokens,
                qheads,
                kvheads,
                qstride,
                kstride,
                table_stride,
                eps,
            )
        };
    }
    /// Copy or add BF16 rows by u32 index; grid=ceil(count*width/256), block=256.
    #[kernel]
    pub unsafe fn infinity_decoder_rows_bf16(
        dst: *mut u16,
        src: *const u16,
        indices: *const u32,
        count: u32,
        width: u32,
        add: u32,
    ) {
        unsafe { crate::qwen3vl::qwen3vl_rows_bf16(dst, src, indices, count, width, add) };
    }
    /// Elementwise BF16 add; grid=ceil(count/256), block=256.
    #[kernel]
    pub unsafe fn infinity_qwen3vl_add_bf16(
        a: *const u16,
        b: *const u16,
        out: *mut u16,
        count: u32,
    ) {
        unsafe { crate::qwen3vl::qwen3vl_add_bf16(a, b, out, count) };
    }
    /// Tanh GELU in place; grid=ceil(count/256), block=256.
    #[kernel]
    pub unsafe fn infinity_qwen3vl_gelu_tanh_bf16(inout: *mut u16, count: u32) {
        unsafe { crate::qwen3vl::qwen3vl_gelu_tanh_bf16(inout, count) };
    }
    /// Four-corner learned position interpolation plus add; grid=ceil(tokens*hidden/256).
    #[kernel]
    pub unsafe fn infinity_qwen3vl_pos_interp(
        pos_embed: *const u16,
        states: *mut u16,
        tokens: u32,
        hidden: u32,
        side: u32,
        height: u32,
        width: u32,
        merge: u32,
    ) {
        unsafe {
            crate::qwen3vl::qwen3vl_pos_interp(
                pos_embed, states, tokens, hidden, side, height, width, merge,
            )
        };
    }
    /// 2-D split-half vision rotary; grid=ceil(tokens*heads*(head_dim/2)/256).
    #[kernel]
    pub unsafe fn infinity_qwen3vl_rope2d_bf16(
        q: *mut u16,
        k: *mut u16,
        heads: u32,
        head_dim: u32,
        tokens: u32,
        height: u32,
        width: u32,
        merge: u32,
        q_stride: u64,
        k_stride: u64,
    ) {
        unsafe {
            crate::qwen3vl::qwen3vl_rope2d_bf16(
                q, k, heads, head_dim, tokens, height, width, merge, q_stride, k_stride,
            )
        };
    }
    /// SiLU in place; grid=ceil(count/256), block=256.
    #[kernel]
    pub unsafe fn infinity_qwen3vl_silu_bf16(inout: *mut u16, count: u32) {
        unsafe { crate::qwen3vl::qwen3vl_silu_bf16(inout, count) };
    }
    /// Additive Qwen3.5 vision-language BF16 add; grid=ceil(count/256).
    #[kernel]
    pub unsafe fn infinity_qwen35vl_add_bf16(a: *const u16, b: *const u16, out: *mut u16, count: u32) {
        unsafe { crate::qwen35vl::qwen35vl_add_bf16(a, b, out, count) };
    }
    /// Add one bias value per row column; grid=ceil(count/256).
    #[kernel]
    pub unsafe fn infinity_qwen35vl_bias_bf16(states: *mut u16, bias: *const u16, count: u32, width: u32) {
        unsafe { crate::qwen35vl::qwen35vl_bias_bf16(states, bias, count, width) };
    }
    /// Tanh-GELU in place on BF16 storage; grid=ceil(count/256).
    #[kernel]
    pub unsafe fn infinity_qwen35vl_gelu_tanh_bf16(values: *mut u16, count: u32) {
        unsafe { crate::qwen35vl::qwen35vl_gelu_tanh_bf16(values, count) };
    }
    /// Four-corner learned-position interpolation plus residual add.
    #[kernel]
    pub unsafe fn infinity_qwen35vl_pos_interp(
        table: *const u16,
        states: *mut u16,
        tokens: u32,
        hidden: u32,
        side: u32,
        height: u32,
        width: u32,
        merge: u32,
    ) {
        unsafe {
            crate::qwen35vl::qwen35vl_pos_interp(table, states, tokens, hidden, side, height, width, merge)
        };
    }
    /// Split-half 2-D vision rotary over packed Q/K rows.
    #[kernel]
    pub unsafe fn infinity_qwen35vl_rope2d_bf16(
        q: *mut u16,
        k: *mut u16,
        heads: u32,
        head_dim: u32,
        tokens: u32,
        height: u32,
        width: u32,
        merge: u32,
        q_stride: u64,
        k_stride: u64,
    ) {
        unsafe {
            crate::qwen35vl::qwen35vl_rope2d_bf16(
                q, k, heads, head_dim, tokens, height, width, merge, q_stride, k_stride,
            )
        };
    }
    /// SiLU in place on BF16 storage; grid=ceil(count/256).
    #[kernel]
    pub unsafe fn infinity_qwen35vl_silu_bf16(values: *mut u16, count: u32) {
        unsafe { crate::qwen35vl::qwen35vl_silu_bf16(values, count) };
    }
    /// Extract one temporal slice from packed `[C, T, P, P]` patch rows.
    #[kernel]
    pub unsafe fn infinity_qwen35vl_temporal_slice_bf16(
        source: *const u16,
        destination: *mut u16,
        rows: u32,
        channels: u32,
        temporal: u32,
        patch: u32,
        slice: u32,
    ) {
        unsafe {
            crate::qwen35vl::qwen35vl_temporal_slice_bf16(
                source, destination, rows, channels, temporal, patch, slice,
            )
        };
    }
    /// Native H128 rotary coefficients prepared once during decoder construction.
    #[kernel]
    pub unsafe fn infinity_base_rope_tables_128_bf16(
        inverse: *const f32,
        cosine: *mut f32,
        sine: *mut f32,
        positions: u32,
    ) {
        unsafe {
            crate::mistral3_rope128::mistral3_base_rope_tables_128_bf16(inverse, cosine, sine, positions)
        };
    }
    /// YaRN-scaled H128 phase tables; construction only, block256.
    #[kernel]
    pub unsafe fn infinity_base_rope_tables_128_scaled_bf16(
        inverse: *const f32,
        cosine: *mut f32,
        sine: *mut f32,
        positions: u32,
        attention_scale: f32,
    ) {
        unsafe {
            crate::mistral3_rope128::mistral3_base_rope_tables_128_scaled_bf16(
                inverse,
                cosine,
                sine,
                positions,
                attention_scale,
            )
        };
    }
    /// Elementwise BF16 residual sum into a distinct output.
    #[kernel]
    pub unsafe fn infinity_mistral3_add_bf16(
        left: *const u16,
        right: *const u16,
        out: *mut u16,
        elements: u32,
    ) {
        unsafe { crate::mistral3::mistral3_add_bf16(left, right, out, elements) };
    }
    /// BF16 activation: 0 exact GELU, 1 tanh GELU, 2 SiLU.
    #[kernel]
    pub unsafe fn infinity_mistral3_gelu_bf16(
        input: *const u16,
        out: *mut u16,
        elements: u32,
        mode: u32,
    ) {
        unsafe { crate::mistral3::mistral3_gelu_bf16(input, out, elements, mode) };
    }
    /// Apply the Native llama_4_scaling per-position query multiplier.
    #[kernel]
    pub unsafe fn infinity_mistral3_llama4_query_scale_bf16(
        q: *mut u16,
        scales: *const u16,
        positions: *const u32,
        tokens: u32,
        heads: u32,
        head_dim: u32,
        q_stride: u64,
    ) {
        unsafe {
            crate::mistral3::mistral3_llama4_query_scale_bf16(
                q, scales, positions, tokens, heads, head_dim, q_stride,
            )
        };
    }
    /// Pixtral patch-merge gather in checkpoint channel-major order.
    #[kernel]
    pub unsafe fn infinity_mistral3_merge_gather_bf16(
        input: *const u16,
        source_rows: *const u32,
        out: *mut u16,
        token_rows: u32,
        hidden: u32,
        merge: u32,
    ) {
        unsafe {
            crate::mistral3::mistral3_merge_gather_bf16(input, source_rows, out, token_rows, hidden, merge)
        };
    }
    /// Convert F32 patch pixels to BF16 tower input.
    #[kernel]
    pub unsafe fn infinity_mistral3_pixels_to_bf16(
        input: *const f32,
        out: *mut u16,
        elements: u32,
    ) {
        unsafe { crate::mistral3::mistral3_pixels_to_bf16(input, out, elements) };
    }
    /// Pixtral 2-D split-half rotary over packed per-image Q/K rows.
    #[kernel]
    pub unsafe fn infinity_mistral3_rope2d_bf16(
        q: *mut u16,
        k: *mut u16,
        grids: *const u32,
        offsets: *const u32,
        image_count: u32,
        tokens: u32,
        heads: u32,
        head_dim: u32,
        theta: f32,
    ) {
        unsafe {
            crate::mistral3::mistral3_rope2d_bf16(
                q,
                k,
                grids,
                offsets,
                image_count,
                tokens,
                heads,
                head_dim,
                theta,
            )
        };
    }
    /// Scatter projected image rows into gathered token embeddings.
    #[kernel]
    pub unsafe fn infinity_mistral3_scatter_features_bf16(
        embeddings: *mut u16,
        features: *const u16,
        indices: *const u32,
        rows: u32,
        width: u32,
    ) {
        unsafe { crate::mistral3::mistral3_scatter_features_bf16(embeddings, features, indices, rows, width) };
    }
    /// BF16 SiLU gate times a separate up matrix.
    #[kernel]
    pub unsafe fn infinity_mistral3_silu_mul_bf16(
        gate: *const u16,
        up: *const u16,
        out: *mut u16,
        elements: u32,
    ) {
        unsafe { crate::mistral3::mistral3_silu_mul_bf16(gate, up, out, elements) };
    }
    /// BF16 shared-expert gate scaling and add, one element per lane.
    /// Grid=ceil(tokens*width/256), block256.
    #[kernel]
    pub unsafe fn infinity_qwen35_moe_shared_gate_bf16(
        routed: *mut u16,
        shared_down: *const u16,
        gate: *const u16,
        tokens: u32,
        width: u32,
    ) {
        unsafe { crate::qwen35_moe::qwen35_moe_shared_gate_bf16(routed, shared_down, gate, tokens, width) };
    }
    /// Additive generic-pair Mistral3 YaRN scaled table export.
    #[kernel]
    pub unsafe fn infinity_mistral3_rope_tables_scaled_bf16(
        inverse: *const f32,
        cosine: *mut f32,
        sine: *mut f32,
        positions: u32,
        pairs: u32,
        scale: f32,
    ) {
        unsafe {
            crate::mistral3_hostfix::rope_tables_scaled_bf16(
                inverse, cosine, sine, positions, pairs, scale,
            )
        };
    }

    /// Exact width-3072 BF16 RMSNorm for the Native Candle 1x4 recipe.
    /// Grid=(rows,1,1), block=(128,1,1); one row per CTA.
    #[kernel]
    pub unsafe fn infinity_rms_norm_3072_exact_vector(
        input: *const u16,
        weight: *const u16,
        out: *mut u16,
        rows: u32,
        width: u32,
        eps: f32,
    ) {
        unsafe {
            crate::norm3072_exact_vector::exact_rms_norm_3072_vector(
                input, weight, out, rows, width, eps,
            )
        };
    }

    /// Exact width-3072 BF16 residual RMSNorm for the Native Candle 1x4 recipe.
    #[kernel]
    pub unsafe fn infinity_add_rms_norm_3072_exact_vector(
        input: *const u16,
        residual: *const u16,
        weight: *const u16,
        out: *mut u16,
        summed: *mut u16,
        rows: u32,
        width: u32,
        eps: f32,
    ) {
        unsafe {
            crate::norm3072_exact_vector::exact_add_rms_norm_3072_vector(
                input, residual, weight, out, summed, rows, width, eps,
            )
        };
    }

}

fn main() {}
