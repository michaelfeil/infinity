// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Laya packed decision refinement and readout elementwise kernels.
//!
//! These entries are deliberately pointwise or per-question reductions. The
//! existing shared `infinity_layer_norm`, `infinity_attention`, `infinity_qkv`
//! and encoder activation entries own the heavy matrix/row work. Laya needs
//! only the operations that do not already have a source-identical Oxide
//! kernel:
//!
//! * add the per-token question-type embedding before refinement;
//! * add two BF16 residual streams with one BF16 rounding;
//! * turn packed F32 option logits into the four published action features;
//! * concatenate the first refined token with those features;
//! * add a BF16 linear bias and widen to F32 for logits/probabilities;
//! * softmax the two learned action classes and select class zero.
//!
//! Every row is isolated by explicit per-question offsets. No kernel reads a
//! neighbouring question's logits, and every output element is assigned once.

use crate::{bf16_from_f32, read, write};
use cuda_device::{float, thread, SharedArray};

/// Add `type_emb[types[row]]` to each packed BF16 hidden row.
/// Grid=ceil(tokens*hidden/256), block=256.
#[inline(always)]
pub(crate) unsafe fn type_add(
    hidden: *const u16,
    types: *const u32,
    type_emb: *const u16,
    out: *mut u16,
    tokens: u32,
    width: u32,
) {
    if tokens == 0 || width == 0 {
        return;
    }
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if index >= tokens as usize * width as usize {
        return;
    }
    let row = index / width as usize;
    let column = index % width as usize;
    let kind = unsafe { *types.add(row) } as usize;
    let value =
        unsafe { read(hidden, index) } + unsafe { read(type_emb, kind * width as usize + column) };
    unsafe { write(out, index, value) };
}

/// Elementwise BF16 residual add. Grid=ceil(elements/256), block=256.
#[inline(always)]
pub(crate) unsafe fn add(a: *const u16, b: *const u16, out: *mut u16, elements: u32) {
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if index >= elements as usize {
        return;
    }
    let value = unsafe { read(a, index) } + unsafe { read(b, index) };
    unsafe { write(out, index, value) };
}

/// Four published action features for each packed question.
///
/// This is the Native `laya_features.cu` tree: one 128-thread block per
/// question, three `float[128]` shared arrays, the same max/runner-up update
/// and butterfly reductions, per-element RN division for probabilities/top2,
/// and the same f64 log/count conversions. Empty questions are rejected by the
/// host metadata validator before this entry can be reached.
#[inline(always)]
pub(crate) unsafe fn action_features(
    logits: *const f32,
    counts: *const u32,
    features: *mut f32,
    questions: u32,
) {
    const THREADS: usize = 128;
    static mut FIRST: SharedArray<f32, THREADS> = SharedArray::UNINIT;
    static mut SECOND: SharedArray<f32, THREADS> = SharedArray::UNINIT;
    static mut SUM: SharedArray<f32, THREADS> = SharedArray::UNINIT;
    let row = thread::blockIdx_x() as usize;
    if row >= questions as usize {
        return;
    }
    let start = unsafe { *counts.add(row) } as usize;
    let end = unsafe { *counts.add(row + 1) } as usize;
    let count = end.saturating_sub(start);
    if count == 0 {
        return;
    }
    let first = unsafe { SharedArray::as_raw_mut_ptr(&raw mut FIRST) };
    let second = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SECOND) };
    let sum = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SUM) };
    let tid = thread::threadIdx_x() as usize;
    let mut a = f32::NEG_INFINITY;
    let mut b = f32::NEG_INFINITY;
    let mut index = start + tid;
    while index < end {
        let value = unsafe { *logits.add(index) };
        if value >= a {
            b = a;
            a = value;
        } else {
            b = libm::fmaxf(b, value);
        }
        index += THREADS;
    }
    unsafe { *first.add(tid) = a };
    unsafe { *second.add(tid) = b };
    thread::sync_threads();
    let mut stride = THREADS / 2;
    while stride > 0 {
        if tid < stride {
            let other = tid + stride;
            let other_first = unsafe { *first.add(other) };
            let other_second = unsafe { *second.add(other) };
            let current_first = unsafe { *first.add(tid) };
            let current_second = unsafe { *second.add(tid) };
            unsafe {
                *second.add(tid) = libm::fmaxf(
                    libm::fminf(current_first, other_first),
                    libm::fmaxf(current_second, other_second),
                );
                *first.add(tid) = libm::fmaxf(current_first, other_first);
            }
        }
        thread::sync_threads();
        stride /= 2;
    }
    let maximum = unsafe { *first.add(0) };
    let runner_up = unsafe { *second.add(0) };
    let mut total = 0.0f32;
    index = start + tid;
    while index < end {
        total = float::add_rn_f32(
            total,
            libm::expf(float::add_rn_f32(unsafe { *logits.add(index) }, -maximum)),
        );
        index += THREADS;
    }
    unsafe { *sum.add(tid) = total };
    thread::sync_threads();
    stride = THREADS / 2;
    while stride > 0 {
        if tid < stride {
            let other = unsafe { *sum.add(tid + stride) };
            unsafe { *sum.add(tid) = float::add_rn_f32(*sum.add(tid), other) };
        }
        thread::sync_threads();
        stride /= 2;
    }
    let denominator = unsafe { *sum.add(0) };
    let mut entropy = 0.0f32;
    index = start + tid;
    while index < end {
        let probability = float::div_rn_f32(
            libm::expf(float::add_rn_f32(unsafe { *logits.add(index) }, -maximum)),
            denominator,
        );
        let logarithm = libm::logf(libm::fmaxf(probability, 1e-9f32));
        entropy = float::add_rn_f32(entropy, -float::mul_rn_f32(probability, logarithm));
        index += THREADS;
    }
    thread::sync_threads();
    unsafe { *sum.add(tid) = entropy };
    thread::sync_threads();
    stride = THREADS / 2;
    while stride > 0 {
        if tid < stride {
            let other = unsafe { *sum.add(tid + stride) };
            unsafe { *sum.add(tid) = float::add_rn_f32(*sum.add(tid), other) };
        }
        thread::sync_threads();
        stride /= 2;
    }
    if tid == 0 {
        let top1 = float::div_rn_f32(1.0f32, denominator);
        let top2 = if count > 1 {
            float::div_rn_f32(
                libm::expf(float::add_rn_f32(runner_up, -maximum)),
                denominator,
            )
        } else {
            0.0f32
        };
        let normalized = if count > 2 { count } else { 2 };
        let entropy_denominator = libm::log(normalized as f64) as f32;
        let count_feature = float::div_rn_f64(normalized as f64, 255.0) as f32;
        unsafe {
            *features.add(row * 4) = top1;
            *features.add(row * 4 + 1) = float::add_rn_f32(top1, -top2);
            *features.add(row * 4 + 2) =
                float::div_rn_f32(unsafe { *sum.add(0) }, entropy_denominator);
            *features.add(row * 4 + 3) = count_feature;
        }
    }
}

/// Add an optional BF16 bias, apply exact erf GELU in FP32, then write BF16.
/// This matches Native `MlpActivation::Gelu` (`at::gelu(x, "none")`) used by
/// Laya's option and action MLPs. Grid=ceil(rows*width/256), block=256.
#[inline(always)]
pub(crate) unsafe fn gelu_exact_bias(
    input: *const u16,
    bias: *const u16,
    out: *mut u16,
    rows: u32,
    width: u32,
    input_stride: u32,
    output_stride: u32,
) {
    if rows == 0 || width == 0 {
        return;
    }
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if index >= rows as usize * width as usize {
        return;
    }
    let row = index / width as usize;
    let column = index % width as usize;
    let mut value = unsafe { read(input, row * input_stride as usize + column) };
    if !bias.is_null() {
        value = f32::from_bits((bf16_from_f32(value + unsafe { read(bias, column) }) as u32) << 16);
    }
    let active = 0.5 * value * (1.0 + libm::erff(value * 0.707_106_77));
    unsafe { write(out, row * output_stride as usize + column, active) };
}

/// Concatenate one F32 pooled first token and four F32 action features into a
/// BF16 `[questions, hidden + 4]` row. Grid=ceil(questions*(hidden+4)/256).
#[inline(always)]
pub(crate) unsafe fn action_input(
    pooled: *const f32,
    features: *const f32,
    out: *mut u16,
    questions: u32,
    hidden: u32,
) {
    if questions == 0 || hidden == 0 {
        return;
    }
    let width = hidden as usize + 4;
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if index >= questions as usize * width {
        return;
    }
    let row = index / width;
    let column = index % width;
    let value = if column < hidden as usize {
        unsafe { *pooled.add(row * hidden as usize + column) }
    } else {
        unsafe { *features.add(row * 4 + (column - hidden as usize)) }
    };
    unsafe { *out.add(index) = bf16_from_f32(value) };
}

/// Softmax two F32 action classes and return class zero. Grid=ceil(rows/256).
#[inline(always)]
pub(crate) unsafe fn softmax_two(input: *const f32, out: *mut f32, rows: u32) {
    let row = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if row >= rows as usize {
        return;
    }
    let first = unsafe { *input.add(row * 2) };
    let second = unsafe { *input.add(row * 2 + 1) };
    let maximum = f32::max(first, second);
    let a = libm::expf(first - maximum);
    let b = libm::expf(second - maximum);
    unsafe { *out.add(row) = float::div_rn_f32(a, float::add_rn_f32(a, b)) };
}
