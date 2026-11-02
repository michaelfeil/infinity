// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Proposed exact ATen-5c488 BF16 `sum(dim=0)` order for contiguous `[T,H]`.
//!
//! Provenance: the schedule is an original Rust port of the reduction order in
//! `pytorch/pytorch@5c4886908584029761b579af026dcfb627c84070`
//! `aten/src/ATen/native/cuda/{Reduce.cuh,ReduceSumProdKernel.cu,Reduce.cu}`
//! (BSD-3-Clause). No upstream C++ text is copied. The surrounding `cuda-device`
//! intrinsics retain their Apache-2.0 terms; this file is a covered Infinity V3
//! contribution under the repository LICENSE.
//!
//! Admitted launch shapes, all with `H` in {3072, 4096, 5120}, `T` in 1..=4096:
//!   * T = 1:          block = (128, 1), grid.y = 1
//!   * T = 2, 3:       block = (64, 2),  grid.y = 1
//!   * T >= 4:         block = (32, 4)
//!   * 4 <= T < 64:    grid.x = ceil((H/4)/128), grid.y = 1
//!   * 64 <= T < 1021: grid.x = ceil((H/4)/32),  grid.y = 1
//!   * T >= 1021:      grid.x = ceil((H/4)/32),  grid.y = C
//! where C is ATen's `ctas_per_output`. C is device dependent; pass the value
//! captured from the real ReduceConfig. C == 1 means "no global reduction".
//! Every admitted H tiles grid.x exactly, so no thread returns before a barrier.
//!
//! `partials` must hold at least `C * H` F32 values and `semaphores` at least
//! `grid.x` zeroed U32 values. Both are unused when C <= 1. Input is row-major
//! contiguous BF16; output is a disjoint F32 `[H]` row. All storage must remain
//! alive through stream completion and no output element may alias the input.
//!
//! Division: this keeps the candidate's FP32 divide then BF16 round. The exact
//! true-divide path of the mean is not in the pinned source set; pin
//! `aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu` (and, if its CPU-scalar
//! branch is reciprocal-multiply, `BinaryMulKernel.cu`) before freezing division.

use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
use cuda_device::{SharedArray, fence, float, thread};

use crate::{bf16_from_f32, f32_from_bf16, read};

const OUT_VEC: usize = 4;
const VT0: usize = 4;
const THREADS: usize = 128;
const SHARED_F32: usize = THREADS * OUT_VEC;

#[inline(always)]
unsafe fn load_vec4(input: *const u16, row: usize, feature: usize, width: usize) -> [f32; OUT_VEC] {
    let base = row * width + feature;
    [
        unsafe { read(input, base) },
        unsafe { read(input, base + 1) },
        unsafe { read(input, base + 2) },
        unsafe { read(input, base + 3) },
    ]
}

/// `Reduce.cuh::thread_reduce_impl`, vt0=4, output_vec_size=4.
#[inline(always)]
unsafe fn thread_reduce4(
    input: *const u16,
    width: usize,
    rows: usize,
    feature: usize,
    first_row: usize,
    row_stride: usize,
) -> [f32; OUT_VEC] {
    let mut acc = [[0.0f32; OUT_VEC]; VT0];
    let mut row = first_row;
    while row + (VT0 - 1) * row_stride < rows {
        let mut i = 0;
        while i < VT0 {
            let values = unsafe { load_vec4(input, row + i * row_stride, feature, width) };
            let mut c = 0;
            while c < OUT_VEC {
                acc[i][c] = float::add_rn_f32(acc[i][c], values[c]);
                c += 1;
            }
            i += 1;
        }
        row += VT0 * row_stride;
    }
    let mut i = 0;
    while i < VT0 && row < rows {
        let values = unsafe { load_vec4(input, row, feature, width) };
        let mut c = 0;
        while c < OUT_VEC {
            acc[i][c] = float::add_rn_f32(acc[i][c], values[c]);
            c += 1;
        }
        row += row_stride;
        i += 1;
    }
    let mut c = 0;
    while c < OUT_VEC {
        let mut i = 1;
        while i < VT0 {
            acc[0][c] = float::add_rn_f32(acc[0][c], acc[i][c]);
            i += 1;
        }
        c += 1;
    }
    acc[0]
}

/// `Reduce.cuh::block_y_reduce`: shared[tx + ty*blockDim.x], halving offsets.
#[inline(always)]
unsafe fn block_y_reduce(mut value: [f32; OUT_VEC], shared: *mut f32) -> [f32; OUT_VEC] {
    let tx = thread::threadIdx_x() as usize;
    let ty = thread::threadIdx_y() as usize;
    let width = thread::blockDim_x() as usize;
    let height = thread::blockDim_y() as usize;
    let base = tx + ty * width;
    let mut c = 0;
    while c < OUT_VEC {
        unsafe { *shared.add(base * OUT_VEC + c) = value[c] };
        c += 1;
    }
    let mut offset = height / 2;
    while offset > 0 {
        thread::sync_threads();
        if ty < offset && ty + offset < height {
            let other = tx + (ty + offset) * width;
            let mut c = 0;
            while c < OUT_VEC {
                value[c] = float::add_rn_f32(value[c], unsafe {
                    *shared.add(other * OUT_VEC + c)
                });
                c += 1;
            }
            let mut c = 0;
            while c < OUT_VEC {
                unsafe { *shared.add(base * OUT_VEC + c) = value[c] };
                c += 1;
            }
        }
        offset /= 2;
    }
    value
}

/// Exact ATen order plus the candidate BF16 mean projection.
#[inline(always)]
pub(crate) unsafe fn mistral3_model_mean_aten_bf16(
    input: *const u16,
    out: *mut f32,
    partials: *mut f32,
    semaphores: *mut u32,
    rows: u32,
    width: u32,
    ctas_per_output: u32,
) {
    if rows == 0 || width == 0 {
        return;
    }
    let t = rows as usize;
    let w = width as usize;
    let ctas = ctas_per_output as usize;
    let tx = thread::threadIdx_x() as usize;
    let ty = thread::threadIdx_y() as usize;
    let bw = thread::blockDim_x() as usize;
    let bh = thread::blockDim_y() as usize;
    let bx = thread::blockIdx_x() as usize;
    let by = thread::blockIdx_y() as usize;

    // 64 <= T turns off output_mult.y and turns on input_mult.y.
    let warp_split = t >= 64;
    let vector = if warp_split {
        tx + bx * bw
    } else {
        tx + ty * bw + bx * bw * bh
    };
    let feature = vector * OUT_VEC;
    if feature >= w {
        return;
    }

    let (first_row, row_stride) = if warp_split {
        (ty + 4 * by, if ctas > 1 { 4 * ctas } else { 4 })
    } else {
        (0, 1)
    };

    let mut value = unsafe { thread_reduce4(input, w, t, feature, first_row, row_stride) };

    static mut SHARED: SharedArray<f32, SHARED_F32> = SharedArray::UNINIT;
    static mut LAST_CTA: SharedArray<u32, 1> = SharedArray::UNINIT;
    let shared = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SHARED) };
    let last = unsafe { SharedArray::as_raw_mut_ptr(&raw mut LAST_CTA) };

    if warp_split {
        value = unsafe { block_y_reduce(value, shared) };
    }

    if ctas > 1 {
        // Reduce.cuh::global_reduce staging, semaphore and last-CTA fold.
        if ty == 0 {
            let staging = by * w + feature;
            let mut c = 0;
            while c < OUT_VEC {
                unsafe { *partials.add(staging + c) = value[c] };
                c += 1;
            }
        }
        fence::threadfence();
        thread::sync_threads();
        if tx == 0 && ty == 0 {
            let counter = unsafe { DeviceAtomicU32::from_ptr(semaphores.add(bx)) };
            let previous = counter.fetch_add(1, AtomicOrdering::Relaxed) as usize;
            unsafe { *last = (previous + 1 == ctas) as u32 };
        }
        thread::sync_threads();
        if unsafe { *last } == 0 {
            return;
        }
        fence::threadfence();

        let mut c = 0;
        while c < OUT_VEC {
            value[c] = 0.0;
            c += 1;
        }
        let mut source = ty;
        while source < ctas {
            let staging = source * w + feature;
            let mut c = 0;
            while c < OUT_VEC {
                value[c] = float::add_rn_f32(value[c], unsafe {
                    *partials.add(staging + c)
                });
                c += 1;
            }
            source += bh;
        }
        value = unsafe { block_y_reduce(value, shared) };
    }

    let store = !warp_split || ty == 0;
    if store {
        let mut c = 0;
        while c < OUT_VEC {
            let rounded_sum = bf16_from_f32(value[c]);
            let sum = f32_from_bf16(rounded_sum);
            let mean = bf16_from_f32(float::div_rn_f32(sum, t as f32));
            unsafe { *out.add(feature + c) = f32_from_bf16(mean) };
            c += 1;
        }
    }
}
