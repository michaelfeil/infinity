// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Clef field composition, normalized prior/cosine features and final logits.

use crate::clef_decision::{
    bf16_add, bf16_mul, block_max, block_sum, read_bf16, round_bf16, write_bf16,
};
use cuda_device::{thread, SharedArray};

/// `base + summary_norm + global_projection + type_embedding[kind]`.
pub(crate) unsafe fn field_compose(
    base: *const u16,
    summary: *const u16,
    global: *const u16,
    types: *const u16,
    field_request: *const u32,
    field_kind: *const u32,
    out: *mut u16,
    fields: u32,
    width: u32,
) {
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if width == 0 || index >= fields as usize * width as usize {
        return;
    }
    let field = index / width as usize;
    let column = index % width as usize;
    let request = unsafe { *field_request.add(field) } as usize;
    let kind = unsafe { *field_kind.add(field) } as usize;
    let mut value = unsafe { bf16_add(read_bf16(base, index), read_bf16(summary, index)) };
    value = unsafe { bf16_add(value, read_bf16(global, request * width as usize + column)) };
    value = unsafe { bf16_add(value, read_bf16(types, kind * width as usize + column)) };
    unsafe { write_bf16(out, index, value) };
}

/// `normalize(question[field] + global[request])`, one CTA per field.
pub(crate) unsafe fn anchor(
    question: *const u16,
    global: *const u16,
    field_request: *const u32,
    out: *mut u16,
    fields: u32,
    hidden: u32,
) {
    let field = thread::blockIdx_x() as usize;
    if hidden == 0 || field >= fields as usize {
        return;
    }
    let hidden = hidden as usize;
    let request = unsafe { *field_request.add(field) } as usize;
    let tid = thread::threadIdx_x() as usize;
    static mut SCRATCH: SharedArray<f32, 8> = SharedArray::UNINIT;
    let scratch = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCRATCH) };
    let mut norm = 0.0f32;
    let mut column = tid;
    while column < hidden {
        let value = unsafe {
            bf16_add(
                read_bf16(question, field * hidden + column),
                read_bf16(global, request * hidden + column),
            )
        };
        norm = cuda_device::float::fma_rn_f32(value, value, norm);
        column += 256;
    }
    let norm = cuda_device::float::sqrt_rn_f32(unsafe { block_sum(norm, scratch) });
    let denominator = f32::from_bits((unsafe { round_bf16(f32::max(norm, 1e-12)) } as u32) << 16);
    column = tid;
    while column < hidden {
        let value = unsafe {
            bf16_add(
                read_bf16(question, field * hidden + column),
                read_bf16(global, request * hidden + column),
            )
        };
        unsafe { write_bf16(out, field * hidden + column, value / denominator) };
        column += 256;
    }
}

/// `dot(normalize(lexical), normalize(anchor[field])) * scale` in BF16.
pub(crate) unsafe fn prior(
    lexical: *const u16,
    anchors: *const u16,
    field_of_option: *const u32,
    out: *mut u16,
    options: u32,
    hidden: u32,
    scale: f32,
) {
    let row = thread::blockIdx_x() as usize;
    if hidden == 0 || row >= options as usize {
        return;
    }
    let hidden = hidden as usize;
    let field = unsafe { *field_of_option.add(row) } as usize;
    let tid = thread::threadIdx_x() as usize;
    static mut SCRATCH: SharedArray<f32, 8> = SharedArray::UNINIT;
    let scratch = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCRATCH) };
    let mut lexical_norm = 0.0f32;
    let mut anchor_norm = 0.0f32;
    let mut column = tid;
    while column < hidden {
        let left = unsafe { read_bf16(lexical, row * hidden + column) };
        let right = unsafe { read_bf16(anchors, field * hidden + column) };
        lexical_norm = cuda_device::float::fma_rn_f32(left, left, lexical_norm);
        anchor_norm = cuda_device::float::fma_rn_f32(right, right, anchor_norm);
        column += 256;
    }
    let lexical_norm = f32::from_bits((unsafe { round_bf16(f32::max(cuda_device::float::sqrt_rn_f32(unsafe { block_sum(lexical_norm, scratch) }), 1e-12)) } as u32) << 16);
    let anchor_norm = f32::from_bits((unsafe { round_bf16(f32::max(cuda_device::float::sqrt_rn_f32(unsafe { block_sum(anchor_norm, scratch) }), 1e-12)) } as u32) << 16);
    let mut dot = 0.0f32;
    column = tid;
    while column < hidden {
        let left = unsafe { bf16_mul(read_bf16(lexical, row * hidden + column), 1.0 / lexical_norm) };
        let right = unsafe { bf16_mul(read_bf16(anchors, field * hidden + column), 1.0 / anchor_norm) };
        dot = cuda_device::float::fma_rn_f32(left, right, dot);
        column += 256;
    }
    let dot = f32::from_bits((unsafe { round_bf16(unsafe { block_sum(dot, scratch) }) } as u32) << 16);
    unsafe { write_bf16(out, row, dot * scale) };
}

/// Normalized cosine plus `[field, option, field*option, |field-option|]`.
pub(crate) unsafe fn features(
    field_rows: *const u16,
    option_rows: *const u16,
    field_of_option: *const u32,
    out: *mut u16,
    cosine: *mut u16,
    options: u32,
    width: u32,
) {
    let row = thread::blockIdx_x() as usize;
    if width == 0 || row >= options as usize {
        return;
    }
    let width = width as usize;
    let field = unsafe { *field_of_option.add(row) } as usize;
    let tid = thread::threadIdx_x() as usize;
    static mut SCRATCH: SharedArray<f32, 8> = SharedArray::UNINIT;
    let scratch = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCRATCH) };
    let mut field_norm = 0.0f32;
    let mut option_norm = 0.0f32;
    let mut column = tid;
    while column < width {
        let left = unsafe { read_bf16(field_rows, field * width + column) };
        let right = unsafe { read_bf16(option_rows, row * width + column) };
        field_norm = cuda_device::float::fma_rn_f32(left, left, field_norm);
        option_norm = cuda_device::float::fma_rn_f32(right, right, option_norm);
        column += 256;
    }
    let field_norm = f32::from_bits((unsafe { round_bf16(f32::max(cuda_device::float::sqrt_rn_f32(unsafe { block_sum(field_norm, scratch) }), 1e-12)) } as u32) << 16);
    let option_norm = f32::from_bits((unsafe { round_bf16(f32::max(cuda_device::float::sqrt_rn_f32(unsafe { block_sum(option_norm, scratch) }), 1e-12)) } as u32) << 16);
    let mut dot = 0.0f32;
    column = tid;
    while column < width {
        let left = unsafe { read_bf16(field_rows, field * width + column) };
        let right = unsafe { read_bf16(option_rows, row * width + column) };
        let normalized = unsafe { bf16_mul(left / field_norm, right / option_norm) };
        dot = cuda_device::float::fma_rn_f32(normalized, 1.0, dot);
        let base = row * width * 4 + column;
        unsafe {
            write_bf16(out, base, left);
            write_bf16(out, base + width, right);
            write_bf16(out, base + width * 2, bf16_mul(left, right));
            write_bf16(out, base + width * 3, (left - right).abs());
        }
        column += 256;
    }
    unsafe { write_bf16(cosine, row, unsafe { block_sum(dot, scratch) }) };
}

/// Final `prior + (cosine * joint_scale + residual) * gate`, widened to F32.
pub(crate) unsafe fn logits(
    prior: *const u16,
    cosine: *const u16,
    residual: *const u16,
    out: *mut f32,
    options: u32,
    joint_scale: f32,
    gate: f32,
) {
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if index >= options as usize {
        return;
    }
    let scaled = unsafe { bf16_mul(read_bf16(cosine, index), joint_scale) };
    let sum = unsafe { bf16_add(scaled, read_bf16(residual, index)) };
    let gated = unsafe { bf16_mul(sum, gate) };
    let value = unsafe { bf16_add(read_bf16(prior, index), gated) };
    unsafe { *out.add(index) = value };
}
