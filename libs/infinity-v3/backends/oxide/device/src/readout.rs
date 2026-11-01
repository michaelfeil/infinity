// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Final-layer selected-row readout primitives.
//!
//! The host validates every index and nonoverlapping output before launch.
//! These entries copy BF16 storage bit-for-bit or widen it without arithmetic;
//! neither reduction, normalization nor tensor-core work belongs here.

use cuda_device::thread;

use crate::f32_from_bf16;

/// Gather the same CLS or last-token rows from two BF16 tensors with independent
/// widths, using one packed cumulative table and an optional selected sequence
/// table. A null `selected` maps each output row to its canonical sequence row,
/// which is what the final-layer bridge uses for all B sequences.
/// `mode` 0 selects the first token of each sequence, 1 selects the last token.
/// The two outputs must not overlap either input or one another, and every
/// output row is written exactly once. Grid=ceil(rows*(first+second)/256).
#[inline(always)]
pub(crate) unsafe fn selected_rows(
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
    if mode > 1 || count == 0 || (first_width == 0 && second_width == 0) {
        return;
    }
    let first_elements = (count as usize) * (first_width as usize);
    let second_elements = (count as usize) * (second_width as usize);
    let total = first_elements + second_elements;
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if index >= total {
        return;
    }
    let (row, column, width, source, out) = if index < first_elements {
        let row = index / first_width as usize;
        (
            row,
            index % first_width as usize,
            first_width,
            first,
            first_out,
        )
    } else {
        let local = index - first_elements;
        let row = local / second_width as usize;
        (
            row,
            local % second_width as usize,
            second_width,
            second,
            second_out,
        )
    };
    if row >= count as usize {
        return;
    }
    let sequence = if selected.is_null() {
        row
    } else {
        (unsafe { *selected.add(row) }) as usize
    };
    let start = unsafe { *cu_seqlens.add(sequence) } as usize;
    let end = unsafe { *cu_seqlens.add(sequence + 1) } as usize;
    if end <= start {
        return;
    }
    let token = if mode == 0 { start } else { end - 1 };
    let source_index = token * width as usize + column;
    unsafe { *out.add(row * width as usize + column) = *source.add(source_index) };
}

/// Widen selected BF16 rows to FP32 without arithmetic or rounding.
/// `selected` may be null for the identity row order; otherwise it holds
/// `count` row indices into `input`. Grid=ceil(count*width/256); input and
/// output storage must not overlap.
#[inline(always)]
pub(crate) unsafe fn bf16_to_f32(
    input: *const u16,
    out: *mut f32,
    selected: *const u32,
    count: u32,
    width: u32,
) {
    if count == 0 || width == 0 {
        return;
    }
    let elements = (count as usize) * (width as usize);
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if index >= elements {
        return;
    }
    let row = index / width as usize;
    let column = index % width as usize;
    let sequence = if selected.is_null() {
        row
    } else {
        (unsafe { *selected.add(row) }) as usize
    };
    unsafe { *out.add(index) = f32_from_bf16(*input.add(sequence * width as usize + column)) };
}
