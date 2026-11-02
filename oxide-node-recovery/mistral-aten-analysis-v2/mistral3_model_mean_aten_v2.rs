// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Exact ATen-5c488 BF16 `sum(dim=0)` order and CPU-scalar true-divide mean.
//!
//! One CUDA thread owns one output feature. It replays ATen's per-feature F32
//! add order sequentially: the same four-accumulator thread fold, the same
//! block-y halving tree, and the same CTA staging fold. No global atomics,
//! semaphores, staging storage, or cross-thread communication are needed, because
//! each feature's reduction is independent and the order is a pure function of
//! T, H, and the captured device properties. The parallel mapping is therefore
//! free to differ from ATen's.
//!
//! Primary source: `pytorch/pytorch@5c4886908584029761b579af026dcfb627c84070`
//! `aten/src/ATen/native/cuda/{Reduce.cuh,ReduceSumProdKernel.cu,Reduce.cu,
//! BinaryDivTrueKernel.cu,BinaryMulKernel.cu}` (BSD-3-Clause; see PROVENANCE).
//! Original Rust port; no upstream C++ text is copied. `cuda-device` intrinsics
//! retain Apache-2.0. This file is a covered Infinity V3 contribution.
//!
//! Admitted serving shapes: contiguous row-major BF16 input, H in
//! {3072,4096,5120}, spans T in 1..=4096, H divisible by 4 with 8-byte row
//! starts. ATen then has output_vec_size=4, block (32,4) for T>=4, and the C
//! formula below. `num_sm` and `max_threads_per_sm` must be captured from the
//! real device (multiProcessorCount / maxThreadsPerMultiProcessor).
//!
//! Launch: block = 256, grid = (ceil(width/256), sequences). Output is a disjoint
//! F32 row-major `[sequences, width]`. `cu_seqlens` and `selected` follow the
//! existing Mistral3 pooling convention: start=cu_seqlens[selected[s]],
//! end=cu_seqlens[selected[s]+1]. Empty spans write 0.0; Mistral3 production
//! validation rejects them before launch. All inputs and outputs must remain
//! alive until stream completion and must not overlap.

use cuda_device::{float, thread};

use crate::{bf16_from_f32, f32_from_bf16, read};


/// `Reduce.cuh:1162-1183` global-reduction C for the admitted H/T layout.
/// Returns 1 when ATen does not enter the global-reduction branch.
#[inline(always)]
fn aten_ctas_per_output(rows: u32, width: u32, num_sm: u32, max_threads_per_sm: u32) -> u32 {
    if rows < 1021 || max_threads_per_sm < 128 {
        return 1;
    }
    let blocks_per_sm = (max_threads_per_sm / 128) as u64;
    let target = (num_sm as u64) * blocks_per_sm;
    let grid = ((width as u64) + 127) / 128;
    if grid == 0 || grid > target {
        return 1;
    }
    let values_per_thread = ((rows as u64) + 3) / 4;
    let ctas1 = (target + grid - 1) / grid;
    let ctas2 = (values_per_thread + 15) / 16;
    let ctas3 = (values_per_thread + 255) / 256;
    let mut ctas = if ctas1 > ctas3 { ctas1 } else { ctas3 };
    if ctas > ctas2 {
        ctas = ctas2;
    }
    if ctas > 1 { ctas as u32 } else { 1 }
}

/// `Reduce.cuh::thread_reduce_impl`, vt0=4, for a single output feature.
#[inline(always)]
unsafe fn reduce_feature(
    input: *const u16,
    width: usize,
    start: usize,
    rows: usize,
    feature: usize,
    first_row: usize,
    row_stride: usize,
) -> f32 {
    let mut acc = [0.0f32; 4];
    let mut row = first_row;
    while row + 3 * row_stride < rows {
        let base = (start + row) * width + feature;
        acc[0] = float::add_rn_f32(acc[0], unsafe { read(input, base) });
        acc[1] = float::add_rn_f32(acc[1], unsafe { read(input, base + row_stride * width) });
        acc[2] = float::add_rn_f32(acc[2], unsafe { read(input, base + 2 * row_stride * width) });
        acc[3] = float::add_rn_f32(acc[3], unsafe { read(input, base + 3 * row_stride * width) });
        row += 4 * row_stride;
    }
    let mut i = 0;
    while i < 4 && row < rows {
        let index = (start + row) * width + feature;
        acc[i] = float::add_rn_f32(acc[i], unsafe { read(input, index) });
        row += row_stride;
        i += 1;
    }
    acc[0] = float::add_rn_f32(acc[0], acc[1]);
    acc[0] = float::add_rn_f32(acc[0], acc[2]);
    acc[0] = float::add_rn_f32(acc[0], acc[3]);
    acc[0]
}

/// Exact per-feature ATen order for one ragged span.
#[inline(always)]
unsafe fn aten_sum_feature(
    input: *const u16,
    width: usize,
    start: usize,
    rows: usize,
    feature: usize,
    ctas: u32,
) -> f32 {
    if rows < 64 {
        return unsafe { reduce_feature(input, width, start, rows, feature, 0, 1) };
    }
    if ctas <= 1 {
        let p0 = unsafe { reduce_feature(input, width, start, rows, feature, 0, 4) };
        let p1 = unsafe { reduce_feature(input, width, start, rows, feature, 1, 4) };
        let p2 = unsafe { reduce_feature(input, width, start, rows, feature, 2, 4) };
        let p3 = unsafe { reduce_feature(input, width, start, rows, feature, 3, 4) };
        return float::add_rn_f32(
            float::add_rn_f32(p0, p2),
            float::add_rn_f32(p1, p3),
        );
    }

    // T >= 1021 with C CTAs: q_b is the block-y tree inside CTA b. The last CTA
    // folds q_b into u_{b mod 4} in increasing b, then repeats the block-y tree.
    let stride = 4 * ctas as usize;
    let mut u = [0.0f32; 4];
    let mut cta = 0;
    while cta < ctas as usize {
        let base = 4 * cta;
        let p0 = unsafe { reduce_feature(input, width, start, rows, feature, base, stride) };
        let p1 = unsafe { reduce_feature(input, width, start, rows, feature, base + 1, stride) };
        let p2 = unsafe { reduce_feature(input, width, start, rows, feature, base + 2, stride) };
        let p3 = unsafe { reduce_feature(input, width, start, rows, feature, base + 3, stride) };
        let q = float::add_rn_f32(
            float::add_rn_f32(p0, p2),
            float::add_rn_f32(p1, p3),
        );
        let y = cta % 4;
        u[y] = float::add_rn_f32(u[y], q);
        cta += 1;
    }
    float::add_rn_f32(
        float::add_rn_f32(u[0], u[2]),
        float::add_rn_f32(u[1], u[3]),
    )
}

#[inline(always)]
pub(crate) unsafe fn mistral3_model_mean_aten_bf16_v2(
    input: *const u16,
    cu_seqlens: *const u32,
    selected: *const u32,
    out: *mut f32,
    sequences: u32,
    width: u32,
    num_sm: u32,
    max_threads_per_sm: u32,
) {
    let sequence = thread::blockIdx_y() as usize;
    if sequence >= sequences as usize || width == 0 {
        return;
    }
    let w = width as usize;
    let feature = thread::blockIdx_x() as usize * thread::blockDim_x() as usize
        + thread::threadIdx_x() as usize;
    if feature >= w {
        return;
    }

    let destination = sequence * w + feature;
    let slot = unsafe { *selected.add(sequence) } as usize;
    let start = unsafe { *cu_seqlens.add(slot) } as usize;
    let end = unsafe { *cu_seqlens.add(slot + 1) } as usize;
    if end <= start {
        unsafe { *out.add(destination) = 0.0 };
        return;
    }
    let rows = end - start;
    let ctas = aten_ctas_per_output(rows as u32, width, num_sm, max_threads_per_sm);

    let sum = unsafe { aten_sum_feature(input, w, start, rows, feature, ctas) };

    // Reduce.cuh: one F32 -> BF16 RNE projection after the complete sum.
    let rounded_sum = bf16_from_f32(sum);
    // BinaryDivTrueKernel.cu CPU-scalar branch:
    // inv_b = opmath_t(high_prec_t(1.0) / scalar<high_prec_t>(2)),
    // out = BF16(float(sum_bf16) * inv_b).
    let inverse = float::div_rn_f64(1.0f64, rows as f64) as f32;
    let mean = bf16_from_f32(float::mul_rn_f32(
        f32_from_bf16(rounded_sum),
        inverse,
    ));
    unsafe { *out.add(destination) = f32_from_bf16(mean) };
}
