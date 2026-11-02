// SPDX-License-Identifier: LicenseRef-Infinity-Restricted-MIT-1.0
//
// New ColBERT token-matrix primitives for the Infinity V3 Oxide backend,
// offered under the repository's Infinity Restricted MIT License 1.0.
// Public contract derived from michaelfeil/infinity commit
// 524235296c1863daebd630ccb8ff5dd6af62399d:
//   libs/infinity-v3/backends/libtorch/cpp/models/multivector_model.h
//   libs/infinity-v3/backends/libtorch/cpp/models/bert_model.h
// Retain the repository LICENSE, NOTICE and upstream Hugging Face attribution.

use cuda_device::{SharedArray, device, float, thread};

#[device]
fn colbert_bf16_to_f32(raw: u16) -> f32 {
    f32::from_bits(u32::from(raw) << 16)
}

#[device]
fn colbert_round_bf16(value: f32) -> f32 {
    let word = cuda_device::convert::cvt_bf16x2_f32(value, value) as u16;
    colbert_bf16_to_f32(word)
}

/// Gather BF16 rows selected by u32 token indices. `source` is a
/// `[tokens, source_stride]` view; `out` is contiguous `[rows, width]`.
/// Grid = ceil(rows * width / 256), block = 256.
#[device]
pub unsafe fn attention_row_gather_bf16(
    source: *const u16,
    indices: *const u32,
    out: *mut u16,
    rows: u32,
    width: u32,
    source_stride: u32,
) {
    if rows == 0 || width == 0 || source_stride < width {
        return;
    }
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    let row = index / width as usize;
    if row >= rows as usize {
        return;
    }
    let token = unsafe { indices.add(row).read() } as usize;
    let column = index % width as usize;
    let source_offset = token * source_stride as usize + column;
    let output_offset = row * width as usize + column;
    unsafe { out.add(output_offset).write(source.add(source_offset).read()) };
}

/// Model-dtype L2 normalization, widened only after BF16 division.
/// Native MultiVectorModel materializes its norm, clamp and quotient in BF16.
/// The sum uses F32 accumulation; its exact ATen reduction is GPU-gated.
/// Grid = rows, block = 256.
#[device]
pub unsafe fn colbert_l2_bf16_f32(
    input: *const u16,
    output: *mut f32,
    rows: u32,
    width: u32,
    eps: f32,
) {
    let row = thread::blockIdx_x() as usize;
    if row >= rows as usize || width == 0 || !(eps > 0.0) {
        return;
    }
    let tid = thread::threadIdx_x() as usize;
    let offset = row * width as usize;
    let mut sum = 0.0f32;
    let mut column = tid;
    while column < width as usize {
        let value = colbert_bf16_to_f32(unsafe { input.add(offset + column).read() });
        sum = float::fma_rn_f32(value, value, sum);
        column += 256;
    }
    static mut SCRATCH: SharedArray<f32, 8> = SharedArray::UNINIT;
    let scratch = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCRATCH) };
    let norm = colbert_round_bf16(float::sqrt_rn_f32(unsafe { crate::block_sum(sum, scratch) }));
    let epsilon = colbert_round_bf16(eps);
    let divisor = if norm < epsilon { epsilon } else { norm };
    column = tid;
    while column < width as usize {
        let value = colbert_bf16_to_f32(unsafe { input.add(offset + column).read() });
        unsafe { output.add(offset + column).write(colbert_round_bf16(value / divisor)) };
        column += 256;
    }
}
