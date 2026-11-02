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
mod colbert;
mod deberta_embedding;
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
mod gemma4_vision;
mod gemma4;
mod gemma2_media;
mod gemma3;
mod jina_code_qk;
mod mpnet_norm;
mod nomic_embedding;
mod norm1024;
mod gte_norm1024;
mod modern_norm1024;
mod plain_layer_norm_aten;
mod norm1024_exact_vector;
mod norm2048;
mod norm2048_vector;
mod qwen35_norm;
mod qwen35_attention;
mod qwen35_copy;
mod rms5120_aten;
mod packed_mean_bf16;
mod norm384;
mod norm512;
mod norm768;
mod norm768_vector;
mod plain_layer_norm_768;
mod qk_norm_rope_128;
mod readout;
mod rms1536;
mod rms1536_vector;
mod rms4096_vector;
mod rms896_exact_vector;
mod silu_pairs;
mod silu_vector8;
mod splade_pool;

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

    /// Gather selected BF16 K/V rows for ColBERT key masking.
    #[kernel]
    pub unsafe fn infinity_attention_row_gather_bf16(
        source: *const u16,
        indices: *const u32,
        out: *mut u16,
        rows: u32,
        width: u32,
        source_stride: u32,
    ) {
        unsafe {
            crate::colbert::attention_row_gather_bf16(
                source,
                indices,
                out,
                rows,
                width,
                source_stride,
            )
        };
    }

    /// F32 per-token L2 normalization over BF16 ColBERT projection rows.
    #[kernel]
    pub unsafe fn infinity_colbert_l2_bf16_f32(
        input: *const u16,
        output: *mut f32,
        rows: u32,
        width: u32,
        eps: f32,
    ) {
        unsafe { crate::colbert::colbert_l2_bf16_f32(input, output, rows, width, eps) };
    }

    /// Plain model-storage ATen Welford LN; grid rows, block128.
    #[kernel]
    pub unsafe fn infinity_plain_layer_norm_768_bf16_aten(
        input: *const u16,
        gamma: *const u16,
        beta: *const u16,
        output: *mut u16,
        rows: u32,
        width: u32,
        eps: f32,
    ) {
        unsafe {
            crate::plain_layer_norm_768::plain_layer_norm_768_bf16_aten(
                input, gamma, beta, output, rows, width, eps,
            )
        };
    }

    /// BF16 SPLADE log1p/max with one final BF16 rounding and F32 widening.
    /// Host validates nonempty spans; grid ceil(width/32)*selected, block128.
    #[kernel]
    pub unsafe fn infinity_splade_pool_bf16_f32(
        input: *const u16,
        spans: *const u32,
        output: *mut f32,
        width: u32,
        selected: u32,
    ) {
        unsafe { crate::splade_pool::splade_pool_bf16_f32(input, spans, output, width, selected) };
    }

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
    pub unsafe fn infinity_plain_layer_norm_bf16_aten(
        input: *const u16, gamma: *const u16, beta: *const u16,
        out: *mut u16, rows: u32, width: u32, eps: f32,
    ) {
        unsafe { crate::plain_layer_norm_aten::plain_layer_norm_bf16_aten(input,gamma,beta,out,rows,width,eps) };
    }

    /// GTE H1024 Candle affine norm with unrounded residual statistics.
    #[kernel]
    pub unsafe fn infinity_gte_layer_norm_1024_candle(input:*const u16,residual:*const u16,bias:*const u16,gamma:*const u16,beta:*const u16,out:*mut u16,rows:u32,width:u32,eps:f32) {
        unsafe{crate::gte_norm1024::gte1024_layer_norm_1024(input,residual,bias,gamma,beta,out,rows,width,eps)};
    }

    #[kernel]
    pub unsafe fn infinity_deberta_divide(input: *mut u16, elements: u32, scale: f32) {
        unsafe { crate::deberta_math::divide(input, elements, scale) };
    }

    #[kernel]
    pub unsafe fn infinity_deberta_plain_norm_bf16_aten(
        input: *const u16, residual: *const u16, bias: *const u16,
        gamma: *const u16, beta: *const u16, out: *mut u16,
        rows: u32, width: u32, eps: f32,
    ) {
        unsafe { crate::plain_layer_norm_aten::deberta_plain_norm_bf16_aten(input,residual,bias,gamma,beta,out,rows,width,eps) };
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
    /// DeBERTa word + optional absolute-position + optional token-type compose.
    /// Grid=(tokens,1,1), block=(256,1,1).
    #[kernel]
    pub unsafe fn infinity_deberta_embedding_compose(
        ids: *const u32,
        positions: *const u32,
        types: *const u32,
        word: *const u16,
        position: *const u16,
        token_type: *const u16,
        out: *mut u16,
        tokens: u32,
        width: u32,
        position_biased: u32,
        token_types: u32,
    ) {
        unsafe {
            crate::deberta_embedding::embedding_compose(
                ids,
                positions,
                types,
                word,
                position,
                token_type,
                out,
                tokens,
                width,
                position_biased,
                token_types,
            )
        };
    }

    /// Grouped Conv1d over packed sequences plus the configured activation.
    /// Grid=(tokens,1,1), block=(256,1,1); the host validates the geometry.
    #[kernel]
    pub unsafe fn infinity_deberta_conv_768(
        input: *const u16,
        offsets: *const u32,
        out: *mut u16,
        weight: *const u16,
        bias: *const u16,
        tokens: u32,
        batch: u32,
        hidden: u32,
        kernel: u32,
        groups: u32,
        padding: u32,
        activation: u32,
    ) {
        unsafe {
            crate::deberta_embedding::grouped_conv(
                input, offsets, out, weight, bias, tokens, batch, hidden, kernel, groups, padding,
                activation,
            )
        };
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

    /// Modern width-1024 word-only rounded embedding norm.
    /// Grid=(rows,1,1), block=(256,1,1); one row per CTA.
    #[kernel]
    pub unsafe fn infinity_embedding_word_norm_1024_candle(
        ids: *const u32,
        word: *const u16,
        gamma: *const u16,
        out: *mut u16,
        rows: u32,
        eps: f32,
    ) {
        unsafe { crate::modern_norm1024::embedding_word_norm_1024(ids, word, gamma, out, rows, eps) }
    }

    /// Modern width-1024 rounded residual norm retaining the BF16 summed stream.
    /// Grid=(rows,1,1), block=(256,1,1); one row per CTA.
    #[kernel]
    pub unsafe fn infinity_rounded_residual_norm_1024_candle(
        input: *const u16,
        residual: *const u16,
        gamma: *const u16,
        normalized: *mut u16,
        summed: *mut u16,
        rows: u32,
        eps: f32,
    ) {
        unsafe {
            crate::modern_norm1024::rounded_residual_norm_1024(
                input, residual, gamma, normalized, summed, rows, eps,
            )
        }
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

    /// Exact masked width-896 BF16 RMSNorm with aligned 128-bit transactions.
    /// Grid=(ceil(rows/4),1,1), block=(128,1,1); one warp per row. The kernel
    /// masks 896 real columns inside the registered 1024 lane map.
    #[kernel]
    pub unsafe fn infinity_rms_norm_896_exact_masked_vector(
        input: *const u16,
        weight: *const u16,
        out: *mut u16,
        rows: u32,
        width: u32,
        eps: f32,
    ) {
        unsafe {
            crate::rms896_exact_vector::exact_rms_norm_896_vector(
                input, weight, out, rows, width, eps,
            )
        };
    }

    /// Exact masked width-896 BF16 residual RMSNorm with aligned 128-bit
    /// transactions. Grid=(ceil(rows/4),1,1), block=(128,1,1).
    #[kernel]
    pub unsafe fn infinity_add_rms_norm_896_exact_masked_vector(
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
            crate::rms896_exact_vector::exact_add_rms_norm_896_vector(
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
    /// Gemma3 exact RMS norm with F32 scale, one CTA per row.
    #[kernel]
    pub unsafe fn infinity_gemma3_rms_norm_bf16_f32scale(
        input: *const u16,
        scale: *const f32,
        out: *mut u16,
        rows: u32,
        width: u32,
        eps: f32,
    ) {
        unsafe { crate::gemma3::rms_norm_bf16_f32scale(input, scale, out, rows, width, eps) };
    }

    /// Gemma3 exact residual RMS norm; statistics from input only.
    #[kernel]
    pub unsafe fn infinity_gemma3_residual_norm_bf16_f32scale(
        input: *const u16,
        residual: *const u16,
        scale: *const f32,
        out: *mut u16,
        rows: u32,
        width: u32,
        eps: f32,
    ) {
        unsafe {
            crate::gemma3::residual_norm_bf16_f32scale(
                input, residual, scale, out, rows, width, eps,
            )
        };
    }

    /// Gemma3 in-place Q/K norm plus rotary for even head widths up to 256.
    #[kernel]
    pub unsafe fn infinity_gemma3_qk_norm_rope_bf16(
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
        unsafe {
            crate::gemma3::qk_norm_rope_bf16(
                q,
                k,
                qscale,
                kscale,
                cosine,
                sine,
                positions,
                tokens,
                qheads,
                kvheads,
                head_dim,
                qstride,
                kstride,
                table_stride,
                eps,
            )
        };
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

    /// Ordinary H5120 BF16 ATen reduction. One row per128-thread CTA.
    /// The separate generic add-RMS ABI remains256 threads.
    #[kernel]
    pub unsafe fn infinity_rms_norm_5120_aten(
        input: *const u16,
        weight: *const u16,
        out: *mut u16,
        rows: u32,
        width: u32,
        eps: f32,
    ) {
        unsafe { crate::rms5120_aten::rms_norm_5120_aten(input, weight, out, rows, width, eps) };
    }

}

fn main() {

    #[kernel]
    pub unsafe fn infinity_gemma4_norm_bf16_f32scale(
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
            crate::gemma4::norm_bf16_f32scale(
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

    #[kernel]
    pub unsafe fn infinity_gemma4_router_scale_bf16_to_f32(
        unit: *const u16,
        scale: *const u16,
        out: *mut f32,
        count: u32,
        root: f32,
    ) {
        unsafe { crate::gemma4::router_scale_bf16_to_f32(unit, scale, out, count, root) };
    }

    #[kernel]
    pub unsafe fn infinity_gemma4_residual_norm_bf16_f32scale(
        input: *const u16,
        residual: *const u16,
        scale: *const f32,
        out: *mut u16,
        rows: u32,
        width: u32,
        eps: f32,
    ) {
        unsafe {
            crate::gemma4::residual_norm_bf16_f32scale(
                input, residual, scale, out, rows, width, eps,
            )
        };
    }

    #[kernel]
    pub unsafe fn infinity_gemma2_media_im2col_f32(
        input: *const f32,
        out: *mut f32,
        channels: u32,
        height: u32,
        width: u32,
        kh: u32,
        kw: u32,
        stride: u32,
        pad_h: u32,
        pad_w: u32,
        out_h: u32,
        out_w: u32,
    ) {
        unsafe {
            crate::gemma2_media::im2col_f32(
                input, out, channels, height, width, kh, kw, stride, pad_h, pad_w, out_h,
                out_w,
            )
        };
    }

    #[kernel]
    pub unsafe fn infinity_gemma2_media_mask_f32(
        input: *const f32,
        valid: *const u8,
        out: *mut f32,
        count: u32,
    ) {
        unsafe { crate::gemma2_media::mask_f32(input, valid, out, count) };
    }

    #[kernel]
    pub unsafe fn infinity_gemma2_media_channel_norm_relu_bf16(
        input: *const f32,
        scale: *const u16,
        out: *mut u16,
        rows: u32,
        width: u32,
        eps: f32,
    ) {
        unsafe { crate::gemma2_media::channel_norm_relu_bf16(input, scale, out, rows, width, eps) };
    }

    #[kernel]
    pub unsafe fn infinity_gemma2_media_transpose_btwc_to_bctw_bf16(
        input: *const u16,
        out: *mut u16,
        rows: u32,
        width: u32,
    ) {
        unsafe { crate::gemma2_media::transpose_btwc_to_bctw_bf16(input, out, rows, width) };
    }

    #[kernel]
    pub unsafe fn infinity_gemma2_media_conv_rows_to_btcw_bf16(
        input: *const u16,
        out: *mut u16,
        time: u32,
        spatial: u32,
        channels: u32,
    ) {
        unsafe {
            crate::gemma2_media::conv_rows_to_btcw_bf16(input, out, time, spatial, channels)
        };
    }

    #[kernel]
    pub unsafe fn infinity_gemma2_media_center_pixels_bf16(
        input: *const f32,
        out: *mut u16,
        count: u32,
    ) {
        unsafe { crate::gemma2_media::center_pixels_bf16(input, out, count) };
    }

    #[kernel]
    pub unsafe fn infinity_gemma2_media_patch_add_position_bf16(
        input: *const u16,
        columns: *const u32,
        rows: *const u32,
        table: *const u16,
        out: *mut u16,
        count: u32,
        width: u32,
        positions: u32,
    ) {
        unsafe {
            crate::gemma2_media::patch_add_position_bf16(
                input, columns, rows, table, out, count, width, positions,
            )
        };
    }

    #[kernel]
    pub unsafe fn infinity_gemma2_media_axial_tables_bf16(
        columns: *const u32,
        rows: *const u32,
        inverse: *const f32,
        cosine: *mut u16,
        sine: *mut u16,
        count: u32,
        pairs: u32,
    ) {
        unsafe {
            crate::gemma2_media::axial_tables_bf16(
                columns, rows, inverse, cosine, sine, count, pairs,
            )
        };
    }

    #[kernel]
    pub unsafe fn infinity_gemma2_media_axial_rope_bf16(
        qk: *mut u16,
        cosine: *const u16,
        sine: *const u16,
        tokens: u32,
        heads: u32,
        dim: u32,
    ) {
        unsafe { crate::gemma2_media::axial_rope_bf16(qk, cosine, sine, tokens, heads, dim) };
    }

    #[kernel]
    pub unsafe fn infinity_gemma2_media_attention_bf16(
        q: *const u16,
        k: *const u16,
        v: *const u16,
        out: *mut u16,
        cu_seqlens: *const u32,
        total: u32,
        batch: u32,
        heads: u32,
        kv_heads: u32,
        dim: u32,
        max_q: u32,
        scale: f32,
    ) {
        unsafe {
            crate::gemma2_media::attention_bf16(
                q, k, v, out, cu_seqlens, total, batch, heads, kv_heads, dim, max_q, scale,
            )
        };
    }

    #[kernel]
    pub unsafe fn infinity_gemma2_media_gated_gelu_split_bf16(
        gate: *const u16,
        up: *const u16,
        out: *mut u16,
        rows: u32,
        width: u32,
    ) {
        unsafe { crate::gemma2_media::gated_gelu_split_bf16(gate, up, out, rows, width) };
    }

    #[kernel]
    pub unsafe fn infinity_gemma2_media_vision_pool_bf16(
        input: *const u16,
        out: *mut u16,
        height: u32,
        width: u32,
        pool: u32,
        hidden: u32,
    ) {
        unsafe { crate::gemma2_media::vision_pool_bf16(input, out, height, width, pool, hidden) };
    }

    #[kernel]
    pub unsafe fn infinity_gemma2_media_clamp_bf16(
        input: *const u16,
        out: *mut u16,
        count: u32,
        minimum: f32,
        maximum: f32,
    ) {
        unsafe { crate::gemma2_media::clamp_bf16(input, out, count, minimum, maximum) };
    }

    #[kernel]
    pub unsafe fn infinity_gemma2_media_residual_scale_add_bf16(
        destination: *mut u16,
        source: *const u16,
        count: u32,
        scale: f32,
    ) {
        unsafe { crate::gemma2_media::residual_scale_add_bf16(destination, source, count, scale) };
    }

    #[kernel]
    pub unsafe fn infinity_gemma2_media_add_bf16(
        destination: *mut u16,
        source: *const u16,
        count: u32,
    ) {
        unsafe { crate::gemma2_media::add_bf16(destination, source, count) };
    }

    #[kernel]
    pub unsafe fn infinity_gemma2_media_bias_add_bf16(
        rows: u32,
        width: u32,
        bias: *const u16,
        data: *mut u16,
    ) {
        unsafe { crate::gemma2_media::bias_add_bf16(rows, width, bias, data) };
    }

    #[kernel]
    pub unsafe fn infinity_gemma2_media_audio_scaled_qk_f32(
        q: *const u16,
        k: *const u16,
        per_dim_scale: *const u16,
        q_out: *mut f32,
        k_out: *mut f32,
        tokens: u32,
        heads: u32,
        dim: u32,
    ) {
        unsafe {
            crate::gemma2_media::audio_scaled_qk_f32(
                q, k, per_dim_scale, q_out, k_out, tokens, heads, dim,
            )
        };
    }

    #[kernel]
    pub unsafe fn infinity_gemma2_media_audio_attention_bf16(
        q: *const f32,
        k: *const f32,
        v: *const u16,
        relative: *const u16,
        valid: *const u8,
        out: *mut u16,
        tokens: u32,
        heads: u32,
        dim: u32,
        chunk: u32,
        past: u32,
        future: u32,
        span: u32,
        cap: f32,
        invalid: f32,
    ) {
        unsafe {
            crate::gemma2_media::audio_attention_bf16(
                q, k, v, relative, valid, out, tokens, heads, dim, chunk, past, future, span,
                cap, invalid,
            )
        };
    }

    #[kernel]
    pub unsafe fn infinity_gemma2_media_glu_bf16(
        input: *const u16,
        out: *mut u16,
        rows: u32,
        width: u32,
    ) {
        unsafe { crate::gemma2_media::glu_bf16(input, out, rows, width) };
    }

    #[kernel]
    pub unsafe fn infinity_gemma2_media_depthwise_conv1d_causal_bf16(
        input: *const u16,
        weight: *const u16,
        out: *mut u16,
        tokens: u32,
        channels: u32,
        kernel: u32,
        clip: f32,
    ) {
        unsafe {
            crate::gemma2_media::depthwise_conv1d_causal_bf16(
                input, weight, out, tokens, channels, kernel, clip,
            )
        };
    }

    #[kernel]
    pub unsafe fn infinity_gemma2_media_silu_bf16(
        input: *const u16,
        out: *mut u16,
        count: u32,
    ) {
        unsafe { crate::gemma2_media::silu_bf16(input, out, count) };
    }

    #[kernel]
    pub unsafe fn infinity_gemma2_media_select_rows_bf16(
        input: *const u16,
        out: *mut u16,
        selected: *const u32,
        rows: u32,
        width: u32,
    ) {
        unsafe { crate::gemma2_media::select_rows_bf16(input, out, selected, rows, width) };
    }

    #[kernel]
    pub unsafe fn infinity_gemma2_media_scatter_rows_bf16(
        state: *mut u16,
        rows: *const u16,
        indices: *const u32,
        count: u32,
        width: u32,
    ) {
        unsafe { crate::gemma2_media::scatter_rows_bf16(state, rows, indices, count, width) };
    }

    #[kernel]
    pub unsafe fn infinity_gemma2_media_audio_norm_bf16_f32scale(
        input: *const u16,
        scale: *const f32,
        out: *mut u16,
        rows: u32,
        width: u32,
        eps: f32,
    ) {
        unsafe {
            crate::gemma2_media::audio_norm_bf16_f32scale(input, scale, out, rows, width, eps)
        };
    }

    #[kernel]
    pub unsafe fn infinity_gemma4_unit_norm_bf16(
        input: *const u16,
        out: *mut u16,
        rows: u32,
        width: u32,
        heads: u32,
        token_stride: u64,
        eps: f32,
    ) {
        unsafe { crate::gemma4::unit_norm_bf16(input, out, rows, width, heads, token_stride, eps) };
    }

    #[kernel]
    pub unsafe fn infinity_gemma4_gated_gelu_bf16(
        input: *const u16,
        out: *mut u16,
        rows: u32,
        width: u32,
    ) {
        unsafe { crate::gemma4::gated_gelu_bf16(input, out, rows, width) };
    }

    #[kernel]
    pub unsafe fn infinity_gemma4_ple_gate_bf16(
        gate: *const u16,
        ple: *const u16,
        out: *mut u16,
        rows: u32,
        width: u32,
        layers: u32,
        layer: u32,
    ) {
        unsafe { crate::gemma4::ple_gate_bf16(gate, ple, out, rows, width, layers, layer) };
    }

    #[kernel]
    pub unsafe fn infinity_gemma4_ple_combine_bf16(
        projected: *const u16,
        token: *const u16,
        out: *mut u16,
        count: u32,
        scale: f32,
    ) {
        unsafe { crate::gemma4::ple_combine_bf16(projected, token, out, count, scale) };
    }

    #[kernel]
    pub unsafe fn infinity_gemma4_scale_bf16(
        input: *const u16,
        out: *mut u16,
        count: u32,
        scale: f32,
    ) {
        unsafe { crate::gemma4::scale_bf16(input, out, count, scale) };
    }

    #[kernel]
    pub unsafe fn infinity_gemma4_rope_tables_bf16(
        inverse: *const f32,
        cosine: *mut f32,
        sine: *mut f32,
        positions: u32,
        pairs: u32,
    ) {
        unsafe { crate::gemma4::rope_tables_bf16(inverse, cosine, sine, positions, pairs) };
    }

    #[kernel]
    pub unsafe fn infinity_gemma4_rope_q_bf16(
        q: *mut u16,
        cosine: *const f32,
        sine: *const f32,
        positions: *const u32,
        tokens: u32,
        q_heads: u32,
        head_dim: u32,
        rotary_dim: u32,
        q_stride: u64,
        table_stride: u32,
    ) {
        unsafe {
            crate::gemma4::rope_q_bf16(
                q,
                cosine,
                sine,
                positions,
                tokens,
                q_heads,
                head_dim,
                rotary_dim,
                q_stride,
                table_stride,
            )
        };
    }

    #[kernel]
    pub unsafe fn infinity_gemma4_attention_256_bf16(
        q: *const u16,
        k: *const u16,
        v: *const u16,
        cu_q: *const u32,
        cu_k: *const u32,
        out: *mut u16,
        batch: u32,
        heads: u32,
        kv_heads: u32,
        head_dim: u32,
        q_stride: u64,
        kv_stride: u64,
        max_q: u32,
        max_support: u32,
        scale: f32,
        causal: u32,
        window_left: i32,
        window_right: i32,
    ) {
        unsafe {
            crate::gemma4::attention_256_bf16(
                q,
                k,
                v,
                cu_q,
                cu_k,
                out,
                batch,
                heads,
                kv_heads,
                head_dim,
                q_stride,
                kv_stride,
                max_q,
                max_support,
                scale,
                causal,
                window_left,
                window_right,
            )
        };
    }

    #[kernel]
    pub unsafe fn infinity_gemma4_option_softcap_bf16_to_f32(
        input: *const u16,
        out: *mut f32,
        count: u32,
        cap: f32,
    ) {
        unsafe { crate::gemma4::option_softcap_bf16_to_f32(input, out, count, cap) };
    }

    #[kernel]
    pub unsafe fn infinity_gemma4_select_rows_bf16(
        input: *const u16,
        cu_seqlens: *const u32,
        out: *mut u16,
        rows: u32,
        width: u32,
        last: u32,
    ) {
        unsafe { crate::gemma4::select_rows_bf16(input, cu_seqlens, out, rows, width, last) };
    }

    #[kernel]
    pub unsafe fn infinity_gemma4_vision_clip_rows_bf16(
        input: *const u16,
        out: *mut u16,
        rows: u32,
        width: u32,
        stride: u32,
        min: f32,
        max: f32,
    ) {
        unsafe { crate::gemma4_vision::clip_rows(input, out, rows, width, stride, min, max) };
    }

    #[kernel]
    pub unsafe fn infinity_gemma4_vision_position_add_bf16(
        hidden: *const u16,
        table: *const u16,
        columns: *const u32,
        rows: *const u32,
        out: *mut u16,
        tokens: u32,
        width: u32,
        table_size: u32,
    ) {
        unsafe { crate::gemma4_vision::position_add(hidden, table, columns, rows, out, tokens, width, table_size) };
    }

    #[kernel]
    pub unsafe fn infinity_gemma4_vision_rope_tables_bf16(
        columns: *const u32,
        rows: *const u32,
        inverse: *const f32,
        cosine: *mut u16,
        sine: *mut u16,
        tokens: u32,
        pairs: u32,
    ) {
        unsafe { crate::gemma4_vision::rope_tables(columns, rows, inverse, cosine, sine, tokens, pairs) };
    }

    #[kernel]
    pub unsafe fn infinity_gemma4_vision_rope2d_bf16(
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
        unsafe { crate::gemma4_vision::rope2d(input, cosine, sine, out, count, heads, dim, token_stride, head_stride) };
    }

    #[kernel]
    pub unsafe fn infinity_gemma4_vision_pool_bf16(
        input: *const u16,
        out: *mut u16,
        height: u32,
        width: u32,
        hidden: u32,
        kernel: u32,
        inverse: f32,
        scale: f32,
    ) {
        unsafe { crate::gemma4_vision::pool(input, out, height, width, hidden, kernel, inverse, scale) };
    }

    #[kernel]
    pub unsafe fn infinity_gemma4_vision_pool_standardize_rows_bf16(
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
        unsafe {
            crate::gemma4_vision::pool_standardize_rows(
                input, bias, scale, out, height, width, hidden, kernel, inverse, sqrt_scale,
            )
        };
    }

    #[kernel]
    pub unsafe fn infinity_gemma4_vision_scatter_rows_bf16(
        destination: *mut u16,
        features: *const u16,
        rows: *const u32,
        row_count: u32,
        width: u32,
    ) {
        unsafe { crate::gemma4_vision::scatter_rows(destination, features, rows, row_count, width) };
    }
}
