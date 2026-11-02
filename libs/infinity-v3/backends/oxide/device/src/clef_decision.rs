// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Source-only Clef joint-head device primitives.
//!
//! These are the small operations that have no exact existing oxide entry:
//! FP32-mean/BF16 span pooling, BF16 residual/bias additions, normalized
//! anchor/prior/cosine features and the final gated F32 logits. They preserve
//! Native `decision_models.cpp` operation order and BF16 rounding boundaries.
//! Root must export the `infinity_clef_*` wrappers from `device/src/main.rs`;
//! until that native primitive bundle is composed this source is unqualified.

use cuda_device::{convert, thread, warp, SharedArray};

#[inline(always)]
pub(crate) fn read_bf16(pointer: *const u16, index: usize) -> f32 {
    f32::from_bits((unsafe { *pointer.add(index) } as u32) << 16)
}

#[inline(always)]
pub(crate) fn round_bf16(value: f32) -> u16 {
    convert::cvt_bf16x2_f32(value, value) as u16
}

#[inline(always)]
pub(crate) fn write_bf16(pointer: *mut u16, index: usize, value: f32) {
    unsafe { *pointer.add(index) = round_bf16(value) };
}

#[inline(always)]
pub(crate) unsafe fn block_sum(value: f32, scratch: *mut f32) -> f32 {
    let tid = thread::threadIdx_x() as usize;
    let lane = tid & 31;
    let value = warp::reduce_sum_f32(value);
    if lane == 0 {
        unsafe { *scratch.add(tid >> 5) = value };
    }
    thread::sync_threads();
    let subtotal = if lane < 8 { unsafe { *scratch.add(lane) } } else { 0.0 };
    let total = warp::reduce_sum_f32(subtotal);
    thread::sync_threads();
    total
}

#[inline(always)]
pub(crate) unsafe fn block_max(value: f32, scratch: *mut f32) -> f32 {
    let tid = thread::threadIdx_x() as usize;
    let lane = tid & 31;
    let value = warp::reduce_max_f32(value);
    if lane == 0 {
        unsafe { *scratch.add(tid >> 5) = value };
    }
    thread::sync_threads();
    let subtotal = if lane < 8 { unsafe { *scratch.add(lane) } } else { f32::NEG_INFINITY };
    let total = warp::reduce_max_f32(subtotal);
    thread::sync_threads();
    total
}

#[inline(always)]
pub(crate) unsafe fn bf16_add(left: f32, right: f32) -> f32 {
    f32::from_bits((unsafe { round_bf16(left + right) } as u32) << 16)
}

#[inline(always)]
pub(crate) unsafe fn bf16_mul(left: f32, right: f32) -> f32 {
    f32::from_bits((unsafe { round_bf16(left * right) } as u32) << 16)
}

/// FP32 mean over one selected span, cast once to BF16. Grid=(count,1,1).
pub(crate) unsafe fn span_mean(
    input: *const u16,
    boundaries: *const u32,
    selected: *const u32,
    out: *mut u16,
    count: u32,
    width: u32,
) {
    let row = thread::blockIdx_x() as usize;
    if row >= count as usize || width == 0 {
        return;
    }
    let span = unsafe { *selected.add(row) } as usize;
    let start = unsafe { *boundaries.add(span * 2) } as usize;
    let end = unsafe { *boundaries.add(span * 2 + 1) } as usize;
    if end <= start {
        return;
    }
    let width = width as usize;
    let tid = thread::threadIdx_x() as usize;
    let inv = 1.0 / (end - start) as f32;
    let mut column = tid;
    while column < width {
        let mut total = 0.0f32;
        let mut token = start;
        while token < end {
            total += unsafe { read_bf16(input, token * width + column) };
            token += 1;
        }
        unsafe { write_bf16(out, row * width + column, total * inv) };
        column += 256;
    }
}

/// Elementwise BF16 residual add.
pub(crate) unsafe fn residual_add(a: *const u16, b: *const u16, out: *mut u16, elements: u32) {
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if index >= elements as usize {
        return;
    }
    let value = unsafe { bf16_add(read_bf16(a, index), read_bf16(b, index)) };
    unsafe { write_bf16(out, index, value) };
}

/// `(context + lexical) + question[field]` in Native order.
pub(crate) unsafe fn query_compose(
    question: *const u16,
    context: *const u16,
    lexical: *const u16,
    field_of_option: *const u32,
    out: *mut u16,
    options: u32,
    width: u32,
) {
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if width == 0 || index >= options as usize * width as usize {
        return;
    }
    let option = index / width as usize;
    let field = unsafe { *field_of_option.add(option) } as usize;
    let column = index % width as usize;
    let value = unsafe {
        bf16_add(
            bf16_add(read_bf16(context, index), read_bf16(lexical, index)),
            read_bf16(question, field * width as usize + column),
        )
    };
    unsafe { write_bf16(out, index, value) };
}

/// `matmul(option, base[field]) / sqrt(width)` in BF16. Grid=(options,1,1).
pub(crate) unsafe fn route_scores(
    option: *const u16,
    base: *const u16,
    field_of_option: *const u32,
    out: *mut u16,
    options: u32,
    width: u32,
    _fields: u32,
    scale: f32,
) {
    let row = thread::blockIdx_x() as usize;
    if row >= options as usize || width == 0 {
        return;
    }
    let field = unsafe { *field_of_option.add(row) } as usize;
    let width = width as usize;
    let tid = thread::threadIdx_x() as usize;
    let mut dot = 0.0f32;
    let mut column = tid;
    while column < width {
        dot = cuda_device::float::fma_rn_f32(
            unsafe { read_bf16(option, row * width + column) },
            unsafe { read_bf16(base, field * width + column) },
            dot,
        );
        column += 256;
    }
    static mut SCRATCH: SharedArray<f32, 8> = SharedArray::UNINIT;
    let scratch = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCRATCH) };
    let dot = unsafe { block_sum(dot, scratch) };
    let rounded = f32::from_bits((unsafe { round_bf16(dot) } as u32) << 16);
    unsafe { write_bf16(out, row, rounded * scale) };
}

/// Softmax each field's option scores, then sum weighted option rows.
pub(crate) unsafe fn softmax_sum(
    option: *const u16,
    scores: *const u16,
    field_option_offsets: *const u32,
    out: *mut u16,
    options: u32,
    width: u32,
    fields: u32,
) {
    let field = thread::blockIdx_x() as usize;
    if field >= fields as usize || width == 0 {
        return;
    }
    let begin = unsafe { *field_option_offsets.add(field) } as usize;
    let end = unsafe { *field_option_offsets.add(field + 1) } as usize;
    if end <= begin || end > options as usize {
        return;
    }
    let width = width as usize;
    let tid = thread::threadIdx_x() as usize;
    static mut SCRATCH: SharedArray<f32, 8> = SharedArray::UNINIT;
    let scratch = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCRATCH) };
    let mut maximum = f32::NEG_INFINITY;
    let mut row = begin + tid;
    while row < end {
        maximum = f32::max(maximum, unsafe { read_bf16(scores, row) });
        row += 256;
    }
    maximum = unsafe { block_max(maximum, scratch) };
    let mut total = 0.0f32;
    row = begin + tid;
    while row < end {
        let value = libm::expf(unsafe { read_bf16(scores, row) } - maximum);
        total += value;
        row += 256;
    }
    let total = unsafe { block_sum(total, scratch) };
    let inverse = 1.0 / total;
    let mut column = tid;
    while column < width {
        let mut sum = 0.0f32;
        row = begin;
        while row < end {
            let weight = f32::from_bits(
                (unsafe { round_bf16(libm::expf(read_bf16(scores, row) - maximum) * inverse) }
                    as u32)
                    << 16,
            );
            sum += unsafe { bf16_mul(read_bf16(option, row * width + column), weight) };
            row += 1;
        }
        unsafe { write_bf16(out, field * width + column, sum) };
        column += 256;
    }
}
