// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! BF16 decoder primitives with explicit arithmetic recipes selected by exports.
//!
//! The caller validates lengths, index tables, nonoverlapping Q/K head views and
//! element strides, and retains all storage through stream completion.
//! Generic RMS reductions do not reproduce Candle's width-registered Chan layouts.

use cuda_device::{SharedArray, float, thread};

use crate::{bf16_from_f32, block_sum, f32_from_bf16, read, write};

#[inline(always)]
fn rounded(value: f32) -> f32 {
    f32_from_bf16(bf16_from_f32(value))
}

/// Word-only gather from a contiguous [vocabulary,width] table. IDs are validated
/// by the host. The unscaled export copies BF16 bits; SCALED applies one FP32
/// scalar multiply and rounds its result to BF16, without any normalization.
/// Output must not overlap the table or IDs.
#[inline(always)]
pub(crate) unsafe fn embedding_gather<const SCALED: bool>(
    ids: *const u32,
    table: *const u16,
    out: *mut u16,
    tokens: u32,
    width: u32,
    scale: f32,
) {
    if width == 0 {
        return;
    }
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    let row = index / width as usize;
    if row >= tokens as usize {
        return;
    }
    let token = unsafe { *ids.add(row) } as usize;
    let source = token * width as usize + index % width as usize;
    if SCALED {
        let value = float::mul_rn_f32(unsafe { read(table, source) }, scale);
        unsafe { write(out, index, value) };
    } else {
        unsafe { *out.add(index) = *table.add(source) };
    }
}

#[inline(always)]
unsafe fn rms_value<const ADD: bool, const ROUNDED: bool>(
    input: *const u16,
    residual: *const u16,
    index: usize,
) -> f32 {
    let value = unsafe { read(input, index) };
    if ADD {
        let sum = value + unsafe { read(residual, index) };
        if ROUNDED { rounded(sum) } else { sum }
    } else {
        value
    }
}

/// One row per 256-thread block. ADD writes a separate rounded residual stream.
/// Ordinary statistics use the unrounded FP32 sum and one final affine cast.
/// ROUNDED matches Transformers' sum and pre-affine activation rounding.
/// Outputs must not overlap one another; each may alias an input of equal size.
#[inline(always)]
pub(crate) unsafe fn rms_norm<const ADD: bool, const ROUNDED: bool>(
    input: *const u16,
    residual: *const u16,
    weight: *const u16,
    out: *mut u16,
    summed: *mut u16,
    rows: u32,
    width: u32,
    eps: f32,
) {
    let row = thread::blockIdx_x() as usize;
    if row >= rows as usize || width == 0 {
        return;
    }
    let offset = row * width as usize;
    let tid = thread::threadIdx_x() as usize;
    let mut column = tid;
    let mut sum = 0.0;
    while column < width as usize {
        let value = unsafe { rms_value::<ADD, ROUNDED>(input, residual, offset + column) };
        if ROUNDED {
            sum = float::add_rn_f32(sum, float::mul_rn_f32(value, value));
        } else {
            sum = float::fma_rn_f32(value, value, sum);
        }
        column += 256;
    }
    static mut SCRATCH: SharedArray<f32, 8> = SharedArray::UNINIT;
    let scratch = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCRATCH) };
    let inverse = float::rsqrt_approx_f32(unsafe { block_sum(sum, scratch) } / width as f32 + eps);
    column = tid;
    while column < width as usize {
        let index = offset + column;
        let value = unsafe { rms_value::<ADD, ROUNDED>(input, residual, index) };
        let normalized = float::mul_rn_f32(value, inverse);
        let normalized = if ROUNDED {
            rounded(normalized)
        } else {
            normalized
        };
        let result = float::mul_rn_f32(normalized, unsafe { read(weight, column) });
        if ADD {
            unsafe { write(summed, index, value) };
        }
        unsafe { write(out, index, result) };
        column += 256;
    }
}

/// One thread owns both members of a disjoint rotary pair in Q or K.
/// Strides count BF16 elements; head rows are contiguous within each token.
/// F32 tables have table_stride columns, indexed by positions[token], and their
/// coefficients round to BF16 before multiplication, like native decoder tables.
/// No V, unrotated tail, token padding or table element is written.
#[inline(always)]
pub(crate) unsafe fn rope<const INTERLEAVED: bool, const HFMA: bool>(
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
    if q_heads == 0
        || kv_heads == 0
        || rotary_dim == 0
        || rotary_dim > head_dim
        || rotary_dim % 2 != 0
        || table_stride < rotary_dim / 2
    {
        return;
    }
    let half = (rotary_dim / 2) as usize;
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    let heads = q_heads as usize + kv_heads as usize;
    let pair_row = index / half;
    let token = pair_row / heads;
    if token >= tokens as usize {
        return;
    }
    let head = pair_row % heads;
    let lane = index % half;
    let (data, head, stride) = if head < q_heads as usize {
        (q, head, q_stride)
    } else {
        (k, head - q_heads as usize, k_stride)
    };
    let base = token * stride as usize + head * head_dim as usize;
    let first = if INTERLEAVED { lane * 2 } else { lane };
    let second = if INTERLEAVED { first + 1 } else { lane + half };
    let a = unsafe { read(data, base + first) };
    let b = unsafe { read(data, base + second) };
    let phase = unsafe { *positions.add(token) } as usize * table_stride as usize + lane;
    let cosine = rounded(unsafe { *cosine.add(phase) });
    let sine = rounded(unsafe { *sine.add(phase) });
    let cross_a = rounded(float::mul_rn_f32(a, sine));
    let cross_b = rounded(float::mul_rn_f32(b, sine));
    let (first_value, second_value) = if HFMA {
        (
            float::fma_rn_f32(a, cosine, -cross_b),
            float::fma_rn_f32(b, cosine, cross_a),
        )
    } else {
        (
            rounded(float::mul_rn_f32(a, cosine)) - cross_b,
            rounded(float::mul_rn_f32(b, cosine)) + cross_a,
        )
    };
    unsafe {
        write(data, base + first, first_value);
        write(data, base + second, second_value);
    }
}

/// Packed input rows are [gate(width), up(width)], with contiguous output rows.
/// Both recipes round SiLU to BF16 before the up-product. MODEL_INTERMEDIATES
/// additionally rounds exp(-gate) and its denominator, matching activation code5.
/// Input and output storage must not overlap because their row widths differ.
#[inline(always)]
pub(crate) unsafe fn silu_mul<const MODEL_INTERMEDIATES: bool>(
    input: *const u16,
    out: *mut u16,
    rows: u32,
    width: u32,
) {
    if width == 0 {
        return;
    }
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    let row = index / width as usize;
    if row >= rows as usize {
        return;
    }
    let column = index % width as usize;
    let base = row * 2 * width as usize + column;
    let gate = unsafe { read(input, base) };
    let up = unsafe { read(input, base + width as usize) };
    let exponential = if MODEL_INTERMEDIATES {
        // CUDA 13.1 hexp/h2exp use this upward-rounded log2(e) and ex2.approx.
        let exponent = float::mul_rn_f32(-gate, f32::from_bits(0x3fb8_aa3c));
        rounded(float::ex2_approx_f32(exponent))
    } else {
        // libm::expf is recognized by the pinned cuda-oxide math importer.
        libm::expf(-gate)
    };
    let denominator = 1.0 + exponential;
    let denominator = if MODEL_INTERMEDIATES {
        rounded(denominator)
    } else {
        denominator
    };
    let activated = rounded(float::div_rn_f32(gate, denominator));
    unsafe { write(out, index, float::mul_rn_f32(activated, up)) };
}
