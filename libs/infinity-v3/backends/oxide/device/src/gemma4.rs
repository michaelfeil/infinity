// SPDX-License-Identifier: MIT AND LicenseRef-Infinity-Restricted-MIT-1.0
// Upstream Candle MIT portions retain their MIT grant. Original port
// contributions use the Infinity Restricted MIT License 1.0.
//
// Exact arithmetic sources:
//   libtorch/cpp/kernels/qwen35_kernels.cu (Gemma2 DivSqrt norm and precise GELU)
//   libtorch/cpp/models/gemma_models.cpp (dense Gemma4 composition)
// Repository LICENSE, NOTICE and THIRD_PARTY_NOTICES remain authoritative.
//!
//! Dense Gemma4 text primitives. The host validates width, row, stride and
//! position bounds before launch. Norm rows use up to 512 FP32 statistics
//! lanes, matching the Native width-lane reduction tree.

use cuda_device::{DynamicSharedArray, SharedArray, float, ptx_asm, thread, warp};

use crate::{bf16_from_f32, f32_from_bf16, read, write};

const REFERENCE_MAX_THREADS: usize = 1024;

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

/// Modeled `gemma_norm_cuda(..., reference=true)` reduction tree. The host
/// launches min(next_pow2(width),1024) threads; all widths use the same
/// 1024-slot shared sums, including H1536, PLE256 and Q/K256/512.
///
/// Native H1536 source is `sum = v*v + sum`; the compiler may contract that
/// into FMA. This routine does not assert Native FMA/rounding equivalence.
/// The Root b66 Norm gold is the arbiter before any exactness claim.
#[inline(always)]
unsafe fn reference_denominator(
    input: *const u16,
    base: usize,
    width: usize,
    eps: f32,
    sums: *mut f32,
) -> f32 {
    let tid = thread::threadIdx_x() as usize;
    let block = thread::blockDim_x() as usize;
    let mut sum = 0.0f32;
    let mut column = tid;
    while column < width {
        let value = unsafe { read(input, base + column) };
        // Deliberately modeled as separate mul/add pending the Root b66 gold;
        // Native source contraction is not claimed here.
        sum = float::add_rn_f32(sum, float::mul_rn_f32(value, value));
        column += block;
    }
    unsafe { *sums.add(tid) = sum };
    let mut stride = block / 2;
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
        let mut stride = if block < 32 { block / 2 } else { 16 };
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

/// Modeled Native generic BF16 reference norm: FP32 statistics, sqrt/add
/// epsilon, divide, one BF16 scale rounding and no +1 offset. Exact Native
/// reduction/FMA parity is not claimed until the b66 Norm gold.
/// Native `(unit_norm(x) * root) * router_scale`, rounding to BF16 after each
/// scalar multiply, then widening to F32 for the FP32 router projection.
#[inline(always)]
pub unsafe fn router_scale_bf16_to_f32(
    unit: *const u16,
    scale: *const u16,
    out: *mut f32,
    count: u32,
    root: f32,
) {
    let index = (thread::blockIdx_x() as usize)
        .checked_mul(thread::blockDim_x() as usize)
        .and_then(|base| base.checked_add(thread::threadIdx_x() as usize))
        .unwrap_or(usize::MAX);
    if index >= count as usize {
        return;
    }
    let first = rounded(unsafe { read(unit, index) } * root);
    let second = rounded(first * unsafe { read(scale, index) });
    unsafe { *out.add(index) = second };
}

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
    static mut SUMS: SharedArray<f32, REFERENCE_MAX_THREADS> = SharedArray::UNINIT;
    let sums = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SUMS) };
    let row = thread::blockIdx_x() as usize;
    if row >= rows as usize {
        return;
    }
    let width = width as usize;
    let base = (row / heads as usize) * token_stride as usize + (row % heads as usize) * width;
    let denominator = unsafe { reference_denominator(input, base, width, eps, sums) };
    let out_base = row * width;
    let tid = thread::threadIdx_x() as usize;
    let block = thread::blockDim_x() as usize;
    let mut column = tid;
    while column < width {
        let value = unsafe { read(input, base + column) };
        let normalized = float::div_rn_f32(value, denominator);
        let scaled = float::mul_rn_f32(normalized, unsafe { *scale.add(column) });
        unsafe { write(out, out_base + column, scaled) };
        column += block;
    }
}

/// Native generic BF16 reference residual norm: statistics from `input` only,
/// then one BF16 rounded scale value and one BF16 residual add.
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
    static mut SUMS: SharedArray<f32, REFERENCE_MAX_THREADS> = SharedArray::UNINIT;
    let sums = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SUMS) };
    let row = thread::blockIdx_x() as usize;
    if row >= rows as usize {
        return;
    }
    let width = width as usize;
    let base = row * width;
    let denominator = unsafe { reference_denominator(input, base, width, eps, sums) };
    let tid = thread::threadIdx_x() as usize;
    let block = thread::blockDim_x() as usize;
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
        column += block;
    }
}

/// Native generic BF16 unit RMS norm (F32 ones scale) with the same reference
/// reduction tree used by Q/K/V 256/512.
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
    static mut SUMS: SharedArray<f32, REFERENCE_MAX_THREADS> = SharedArray::UNINIT;
    let sums = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SUMS) };
    let row = thread::blockIdx_x() as usize;
    if row >= rows as usize {
        return;
    }
    let width = width as usize;
    let base = (row / heads as usize) * token_stride as usize + (row % heads as usize) * width;
    let denominator = unsafe { reference_denominator(input, base, width, eps, sums) };
    let out_base = row * width;
    let tid = thread::threadIdx_x() as usize;
    let block = thread::blockDim_x() as usize;
    let mut column = tid;
    while column < width {
        let value = unsafe { read(input, base + column) };
        let normalized = float::div_rn_f32(value, denominator);
        unsafe { write(out, out_base + column, normalized) };
        column += block;
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

/// Combine Gemma4 PLE projected and token rows exactly as the Native composed
/// expression: BF16 rounded add followed by one BF16 rounded scalar multiply.
#[inline(always)]
pub(crate) unsafe fn ple_combine_bf16(
    projected: *const u16,
    token: *const u16,
    out: *mut u16,
    count: u32,
    scale: f32,
) {
    if count == 0 {
        return;
    }
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if index >= count as usize {
        return;
    }
    let sum = rounded(float::add_rn_f32(
        unsafe { read(projected, index) },
        unsafe { read(token, index) },
    ));
    unsafe { write(out, index, rounded(float::mul_rn_f32(sum, scale))) };
}

/// Sliding-window GQA attention for Gemma4 d256 layers. This is an isolated
/// semantic bridge, not the Native FlashAttention online-softmax tree: scores,
/// max, sum and output accumulation are FP32, with one BF16 output rounding.
/// Grid=(max_q, heads, batch), block=256, dynamic shared=max_support*4 bytes.
pub unsafe fn attention_256_bf16(
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
    let sequence = thread::blockIdx_z() as usize;
    let head = thread::blockIdx_y() as usize;
    let query = thread::blockIdx_x() as usize;
    if sequence >= batch as usize
        || head >= heads as usize
        || kv_heads == 0
        || head_dim != 256
        || max_q == 0
        || max_support == 0
    {
        return;
    }
    let q_start = unsafe { *cu_q.add(sequence) } as usize;
    let q_end = unsafe { *cu_q.add(sequence + 1) } as usize;
    let k_start = unsafe { *cu_k.add(sequence) } as usize;
    let k_end = unsafe { *cu_k.add(sequence + 1) } as usize;
    if q_end <= q_start || k_end <= k_start || query >= q_end - q_start {
        return;
    }
    let k_len = k_end - k_start;
    let start = if window_left < 0 {
        0
    } else {
        query.saturating_sub(window_left as usize)
    };
    let mut end = if window_right < 0 {
        k_len - 1
    } else {
        core::cmp::min(k_len - 1, query.saturating_add(window_right as usize))
    };
    if causal != 0 {
        end = core::cmp::min(end, query);
    }
    if start >= k_len || start > end {
        return;
    }
    let support = end - start + 1;
    if support > max_support as usize {
        return;
    }
    let group = heads as usize / kv_heads as usize;
    let kv_head = head / group;
    let scores = DynamicSharedArray::<f32>::get();
    static mut SCRATCH: SharedArray<f32, 8> = SharedArray::UNINIT;
    let scratch = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCRATCH) };
    let tid = thread::threadIdx_x() as usize;
    let lane = tid & 31;
    let warp = tid >> 5;
    let warps = (thread::blockDim_x() as usize) >> 5;
    let q_offset = (q_start + query) * q_stride as usize + head * head_dim as usize;
    let mut key = warp;
    let mut maximum = f32::NEG_INFINITY;
    while key < support {
        let k_offset = (k_start + start + key) * kv_stride as usize + kv_head * head_dim as usize;
        let mut value = 0.0f32;
        let mut column = lane;
        while column < head_dim as usize {
            value = float::fma_rn_f32(
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
        key += warps;
    }
    maximum = unsafe { super::block_max(maximum, scratch) };
    key = tid;
    let mut total = 0.0f32;
    while key < support {
        let value = libm::expf(unsafe { *scores.add(key) } - maximum);
        unsafe { *scores.add(key) = value };
        total += value;
        key += 256;
    }
    let inverse = 1.0f32 / unsafe { super::block_sum(total, scratch) };
    key = tid;
    while key < support {
        unsafe { *scores.add(key) *= inverse };
        key += 256;
    }
    thread::sync_threads();
    let mut column = tid;
    let out_offset = (q_start + query) * q_stride as usize + head * head_dim as usize;
    while column < head_dim as usize {
        let mut value = 0.0f32;
        let mut key = 0usize;
        while key < support {
            value = float::fma_rn_f32(
                unsafe { *scores.add(key) },
                unsafe {
                    read(
                        v,
                        (k_start + start + key) * kv_stride as usize
                            + kv_head * head_dim as usize
                            + column,
                    )
                },
                value,
            );
            key += 1;
        }
        unsafe { write(out, out_offset + column, value) };
        column += 256;
    }
}

/// In-place Q-only NeoX rotary for shared-KV Gemma4 layers. Uses the same
/// BF16 rounding sequence as the Native composed reference.
#[inline(always)]
pub(crate) unsafe fn rope_q_bf16(
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
    if q_heads == 0
        || rotary_dim == 0
        || rotary_dim > head_dim
        || rotary_dim % 2 != 0
        || table_stride < rotary_dim / 2
    {
        return;
    }
    let half = (rotary_dim / 2) as usize;
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    let pair_row = index / half;
    let token = pair_row / q_heads as usize;
    if token >= tokens as usize {
        return;
    }
    let head = pair_row % q_heads as usize;
    let lane = index % half;
    let base = token * q_stride as usize + head * head_dim as usize;
    let a = unsafe { read(q, base + lane) };
    let b = unsafe { read(q, base + lane + half) };
    let phase = unsafe { *positions.add(token) } as usize * table_stride as usize + lane;
    let cosine = rounded(unsafe { *cosine.add(phase) });
    let sine = rounded(unsafe { *sine.add(phase) });
    let cross_a = rounded(float::mul_rn_f32(a, sine));
    let cross_b = rounded(float::mul_rn_f32(b, sine));
    unsafe {
        write(
            q,
            base + lane,
            rounded(float::mul_rn_f32(a, cosine)) - cross_b,
        );
        write(
            q,
            base + lane + half,
            rounded(float::mul_rn_f32(b, cosine)) + cross_a,
        );
    }
}

/// Gather one first/last row per packed sequence into a compact BF16 buffer.
/// Used only by the Native pooled fast path before the final O/MLP/PLE GEMMs.
pub unsafe fn select_rows_bf16(
    input: *const u16,
    cu_seqlens: *const u32,
    out: *mut u16,
    rows: u32,
    width: u32,
    last: u32,
) {
    let row = thread::blockIdx_x() as usize;
    if row >= rows as usize || width == 0 {
        return;
    }
    let source_row = if last != 0 {
        unsafe { *cu_seqlens.add(row + 1) }
    } else {
        unsafe { *cu_seqlens.add(row) }
    };
    let source_row = if last != 0 {
        source_row.saturating_sub(1)
    } else {
        source_row
    } as usize;
    let column = thread::blockIdx_y() as usize * 256 + thread::threadIdx_x() as usize;
    if column < width as usize {
        let value = unsafe { *input.add(source_row * width as usize + column) };
        unsafe { *out.add(row * width as usize + column) = value };
    }
}

/// Native OptionHead softcap staging for selected BF16 score rows.
///
/// Native `decision_models.cpp` applies three Torch scalar operations to the
/// BF16 projection result before widening the flattened logits to F32:
/// divide by `cap`, `tanh`, then multiply by `cap`. Each Torch scalar result is
/// rounded to BF16 before the next operation. This kernel follows that source
/// explicitly with `div.rn.f32`; it deliberately does not copy the separate
/// `gemma_score.cu` sequence helper's reciprocal-multiply form. Whether ATen
/// scalar division lowers identically at every BF16 tie remains an exhaustive
/// primitive-oracle gate. The admitted libdevice `tanhf` mapping is used
/// (never tanh.approx). Input and output must be disjoint.
/// Grid=ceil(count/256), block=256; an out-of-range CTA returns without writes.
#[inline(always)]
pub unsafe fn option_softcap_bf16_to_f32(
    input: *const u16,
    out: *mut f32,
    count: u32,
    cap: f32,
) {
    if count == 0 || !cap.is_finite() || cap <= 0.0 {
        return;
    }
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if index >= count as usize {
        return;
    }
    let divided = float::div_rn_f32(unsafe { read(input, index) }, cap);
    let divided = f32_from_bf16(bf16_from_f32(divided));
    let tangent = bf16_from_f32(libm::tanhf(divided));
    let scaled = float::mul_rn_f32(f32_from_bf16(tangent), cap);
    let rounded = bf16_from_f32(scaled);
    unsafe { *out.add(index) = f32_from_bf16(rounded) };
}
