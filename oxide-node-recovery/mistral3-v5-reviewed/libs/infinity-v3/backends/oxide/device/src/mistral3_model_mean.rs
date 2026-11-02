// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Native Mistral3 model-dtype masked mean.
//!
//! `mistral3_models.cpp::mean_pool` computes `value.sum(0, true) / size(0)` in
//! the hidden-state BF16 dtype and only then widens to F32. This is not the
//! Candle packed_mean BF16 partial tree used by the generic V6 route: the sum
//! is rounded to BF16 once, then divided by the span length in BF16.

use cuda_device::thread;
use crate::{bf16_from_f32, f32_from_bf16};

#[inline(always)]
pub(crate) unsafe fn model_dtype_mean_bf16(
    input: *const u16,
    cu_seqlens: *const u32,
    selected: *const u32,
    out: *mut f32,
    count: u32,
    width: u32,
) {
    if count == 0 || width == 0 {
        return;
    }
    let sequence = thread::blockIdx_x() as usize;
    let feature = thread::blockIdx_y() as usize * 256 + thread::threadIdx_x() as usize;
    if sequence >= count as usize || feature >= width as usize {
        return;
    }
    let slot = unsafe { *selected.add(sequence) } as usize;
    let start = unsafe { *cu_seqlens.add(slot) } as usize;
    let end = unsafe { *cu_seqlens.add(slot + 1) } as usize;
    let destination = sequence * width as usize + feature;
    if end <= start {
        unsafe { *out.add(destination) = 0.0 };
        return;
    }
    let mut sum = 0.0f32;
    let mut row = start;
    while row < end {
        sum += f32_from_bf16(unsafe { *input.add(row * width as usize + feature) });
        row += 1;
    }
    // ATen BF16 sum rounds once, then BF16 divide, then widens to F32.
    let rounded_sum = bf16_from_f32(sum);
    let mean = bf16_from_f32(f32_from_bf16(rounded_sum) / (end - start) as f32);
    unsafe { *out.add(destination) = f32_from_bf16(mean) };
}
