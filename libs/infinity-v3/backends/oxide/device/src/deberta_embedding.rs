// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! DeBERTa embedding composition and grouped post-layer convolution.
//!
//! Composition follows the Native member order: word tensor, then the optional
//! absolute-position add, then the optional token-type add. Every tensor add is
//! rounded once to BF16. The convolution accumulates products in F32, adds the
//! Conv1d bias once, rounds that Conv1d result to BF16 exactly as the Native
//! model-dtype tensor boundary does, then applies the configured Native
//! activation and rounds the activation result to BF16. Packed sequence
//! boundaries replace Native's padded zeros, so a convolution never crosses a
//! packed sequence.

use crate::{bf16_from_f32, f32_from_bf16, read, write};
use cuda_device::{SharedArray, thread};

const INV_SQRT_TWO: f32 = 0.707_106_77;

#[inline(always)]
unsafe fn compose(
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
    let row = thread::blockIdx_x() as usize;
    if row >= tokens as usize || width == 0 {
        return;
    }
    let width = width as usize;
    let tid = thread::threadIdx_x() as usize;
    let stride = thread::blockDim_x() as usize;
    let token = unsafe { *ids.add(row) } as usize;
    let mut column = tid;
    while column < width {
        let mut value = unsafe { read(word, token * width + column) };
        if position_biased != 0 {
            let position_row = unsafe { *positions.add(row) } as usize;
            value = f32_from_bf16(bf16_from_f32(
                value + unsafe { read(position, position_row * width + column) },
            ));
        }
        if token_types != 0 {
            let type_row = unsafe { *types.add(row) } as usize;
            value = f32_from_bf16(bf16_from_f32(
                value + unsafe { read(token_type, type_row * width + column) },
            ));
        }
        unsafe { write(out, row * width + column, value) };
        column += stride;
    }
}

#[inline(always)]
fn activate(value: f32, mode: u32) -> f32 {
    match mode {
        // Native activation("gelu") is at::gelu(x, "none").
        0 => 0.5 * value * (1.0 + libm::erff(value * INV_SQRT_TWO)),
        // Native activation("tanh") is at::tanh.
        1 => libm::tanhf(value),
        // Native activation("relu") is at::relu, which propagates NaN.
        2 => {
            if value.is_nan() || value >= 0.0 {
                value
            } else {
                0.0
            }
        }
        // Native activation("silu") is at::silu.
        3 => {
            let exponential = libm::expf(-value);
            value / (1.0 + exponential)
        }
        // Native activation("gelu_new"/"gelu_pytorch_tanh") is
        // at::gelu(x, "tanh") with the same tanh approximation.
        4 => {
            let cubic = value * value * value;
            let inner = 0.797_884_6 * (value + 0.044_715 * cubic);
            0.5 * value * (1.0 + libm::tanhf(inner))
        }
        _ => value,
    }
}

#[inline(always)]
unsafe fn conv(
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
    let row = thread::blockIdx_x() as usize;
    if row >= tokens as usize || hidden == 0 || groups == 0 || kernel == 0 {
        return;
    }
    static mut SEGMENT: SharedArray<u32, 2> = SharedArray::UNINIT;
    let segment = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SEGMENT) };
    let tid = thread::threadIdx_x() as usize;
    if tid == 0 {
        let mut index = 0usize;
        while index + 1 < batch as usize && unsafe { *offsets.add(index + 1) } as usize <= row {
            index += 1;
        }
        unsafe {
            *segment.add(0) = *offsets.add(index);
            *segment.add(1) = *offsets.add(index + 1);
        }
    }
    thread::sync_threads();
    let start = unsafe { *segment.add(0) } as isize;
    let end = unsafe { *segment.add(1) } as isize;
    let hidden = hidden as usize;
    let kernel = kernel as usize;
    let groups = groups as usize;
    let padding = padding as isize;
    let group_width = hidden / groups;
    let stride = thread::blockDim_x() as usize;
    let mut column = tid;
    while column < hidden {
        let group = column / group_width;
        let group_start = group * group_width;
        let mut sum = 0.0;
        let mut in_group = 0usize;
        while in_group < group_width {
            let in_channel = group_start + in_group;
            let mut tap = 0usize;
            while tap < kernel {
                let source = row as isize + tap as isize - padding;
                if source >= start && source < end {
                    let weight_index = (column * group_width + in_group) * kernel + tap;
                    sum += unsafe { read(input, source as usize * hidden + in_channel) }
                        * unsafe { read(weight, weight_index) };
                }
                tap += 1;
            }
            in_group += 1;
        }
        // Native at::conv1d returns the model dtype (BF16 here), so the bias
        // addition is rounded before activation consumes the Conv1d result.
        // The activation itself returns BF16, and write rounds that result.
        let convolved = f32_from_bf16(bf16_from_f32(sum + unsafe { read(bias, column) }));
        let value = activate(convolved, activation);
        unsafe { write(out, row * hidden + column, value) };
        column += stride;
    }
}

pub(crate) unsafe fn embedding_compose(
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
        compose(
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

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn grouped_conv(
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
        conv(
            input, offsets, out, weight, bias, tokens, batch, hidden, kernel, groups, padding,
            activation,
        )
    };
}
