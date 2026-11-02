/*
Derived from public PyTorch aten/src/ATen/native/cuda/layer_norm_kernel.cu at
commit 5c4886908584029761b579af026dcfb627c84070. The complete BSD-3-Clause
notice from the matched PyTorch LICENSE follows and is preserved verbatim.

From PyTorch:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

From Caffe2:

Copyright (c) 2016-present, Facebook Inc. All rights reserved.

All contributions by Facebook:
Copyright (c) 2016 Facebook Inc.

All contributions by Google:
Copyright (c) 2015 Google Inc.
All rights reserved.

All contributions by Yangqing Jia:
Copyright (c) 2015 Yangqing Jia
All rights reserved.

All contributions by Kakao Brain:
Copyright 2019-2020 Kakao Brain

All contributions by Cruise LLC:
Copyright (c) 2022 Cruise LLC.
All rights reserved.

All contributions by Tri Dao:
Copyright (c) 2024 Tri Dao.
All rights reserved.

All contributions by Arm:
Copyright (c) 2021, 2023-2025 Arm Limited and/or its affiliates

All contributions from Caffe:
Copyright(c) 2013, 2014, 2015, the respective contributors
All rights reserved.

All other contributions:
Copyright(c) 2015, 2016 the respective contributors
All rights reserved.

Caffe2 uses a copyright model similar to Caffe: each contributor holds
copyright over their contributions to Caffe2. The project versioning records
all such contribution and copyright details. If a contributor wants to further
mark their specific copyright on a particular contribution, they should
indicate their copyright solely in the commit message of the change when it is
committed.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/
// Original contributions Copyright Infinity Core Contributors.
// Licensed under the repository LICENSE.
//! Source-only candidate for BF16 width-5120 ordinary RMSNorm matching the
//! ATen `_fused_rms_norm` vectorized CUDA path at the pinned source commit.
//!
//! The target is `RmsNormKernelImpl -> LayerNormKernelImplInternal<..., true>
//! -> vectorized_layer_norm_kernel -> compute_stats`. For BF16 H5120 the
//! vectorized eligibility test passes when N is a multiple of four, N is at
//! most 2^24, and input/output/gamma are 8-byte aligned (4 BF16 values).
//! Native `num_threads()` is C10_WARP_SIZE * 4 = 128 on CUDA, not 512. The
//! 512-thread `kCUDABlockReduceNumThreads` launch is the non-vector fallback
//! and is not this recipe.
//!
//! This leaf intentionally contains only the ordinary, non-residual export.
//! It does not call or alter `decoder::rms_norm`, the add-RMS recipe, QK128, or
//! SiLU. Misaligned pointers or a non-5120 width return without writing; the
//! host must bind this export only for the exact vectorized geometry.

use cuda_device::{SharedArray, float, thread, warp};

use crate::{f32_from_bf16, read, write};

const WIDTH: usize = 5120;
const VEC_SIZE: usize = 4;
const GROUPS: usize = WIDTH / VEC_SIZE;
const ATEN_BLOCK_X: usize = 32;
const ATEN_THREADS: usize = 128;
const ATEN_WARPS: usize = ATEN_THREADS / ATEN_BLOCK_X;
const SMEM_FLOATS: usize = ATEN_WARPS * 3 / 2;
const VECTOR_BYTES: usize = VEC_SIZE * core::mem::size_of::<u16>();

#[inline(always)]
fn aligned(pointer: *const u16) -> bool {
    (pointer as usize) & (VECTOR_BYTES - 1) == 0
}

#[inline(always)]
fn aligned_mut(pointer: *mut u16) -> bool {
    (pointer as usize) & (VECTOR_BYTES - 1) == 0
}

/// Width-5120 ordinary BF16 RMSNorm using the ATen vectorized reduction order.
///
/// Required launch equivalence: 128 flattened threads per row. A one-dimensional
/// 128-thread block and the native two-dimensional (32, 4) block both produce
/// the same flattened `tid`; one CTA owns exactly one row. `threadIdx.y` is
/// folded with `blockDim.x`, so the same code accepts either launch form.
///
/// All arithmetic is non-FTZ FP32. The per-element `sigma2 + value * value`
/// update is emitted as one RN FMA under CUDA's default `fmad=true` policy.
/// The warp tree is shuffle-down 16, 8, 4, 2, 1 with current + shuffled in the
/// native `cuWelfordCombine(current, shuffled)` order. The inter-warp tree is
/// offset 2 then 1 using the native six-float shared overlay. The final RMS
/// variance is a precise RN division by 5120, never a reciprocal multiply.
/// Output is `gamma * (rsqrt.approx(variance + eps) * value)` followed by one
/// BF16 round-to-nearest-even store.
#[inline(always)]
pub(crate) unsafe fn rms_norm_5120_aten(
    input: *const u16,
    weight: *const u16,
    out: *mut u16,
    rows: u32,
    width: u32,
    eps: f32,
) {
    let row = thread::blockIdx_x() as usize;
    let block_threads =
        thread::blockDim_x() as usize * thread::blockDim_y() as usize;
    if width as usize != WIDTH
        || row >= rows as usize
        || block_threads != ATEN_THREADS
        || !aligned(input)
        || !aligned(weight)
        || !aligned_mut(out)
    {
        return;
    }

    let tid = thread::threadIdx_x() as usize
        + thread::threadIdx_y() as usize * thread::blockDim_x() as usize;
    let lane = tid & (ATEN_BLOCK_X - 1);
    let warp = tid / ATEN_BLOCK_X;
    let row_offset = row * WIDTH;

    let mut sigma2 = 0.0f32;
    let mut group = tid;
    while group < GROUPS {
        let base = row_offset + group * VEC_SIZE;
        let mut element = 0;
        while element < VEC_SIZE {
            let value = unsafe { read(input, base + element) };
            sigma2 = float::fma_rn_f32(value, value, sigma2);
            element += 1;
        }
        group += ATEN_THREADS;
    }

    let mut offset = ATEN_BLOCK_X / 2;
    while offset > 0 {
        let shuffled = warp::shuffle_down_f32(sigma2, offset as u32);
        sigma2 = float::add_rn_f32(sigma2, shuffled);
        offset /= 2;
    }

    static mut SHARED: SharedArray<f32, SMEM_FLOATS> = SharedArray::UNINIT;
    let shared = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SHARED) };

    let mut merge = ATEN_WARPS / 2;
    while merge > 0 {
        if lane == 0 && warp >= merge && warp < 2 * merge {
            let write_y = warp - merge;
            unsafe {
                *shared.add(2 * write_y) = 0.0;
                *shared.add(2 * write_y + 1) = sigma2;
                *shared.add(ATEN_WARPS + write_y) = 0.0;
            }
        }
        thread::sync_threads();

        if lane == 0 && warp < merge {
            let shuffled = unsafe { *shared.add(2 * warp + 1) };
            sigma2 = float::add_rn_f32(sigma2, shuffled);
        }
        thread::sync_threads();

        merge /= 2;
    }

    if lane == 0 && warp == 0 {
        unsafe {
            *shared = 0.0;
            *shared.add(1) = float::div_rn_f32(sigma2, WIDTH as f32);
        }
    }
    thread::sync_threads();

    let variance = unsafe { *shared.add(1) };
    let inverse = float::rsqrt_approx_f32(float::add_rn_f32(variance, eps));

    group = tid;
    while group < GROUPS {
        let column = group * VEC_SIZE;
        let base = row_offset + column;
        let mut element = 0;
        while element < VEC_SIZE {
            let value = unsafe { read(input, base + element) };
            let normalized = float::mul_rn_f32(inverse, value);
            let gamma = f32_from_bf16(unsafe { *weight.add(column + element) });
            let result = float::mul_rn_f32(gamma, normalized);
            unsafe { write(out, base + element, result) };
            element += 1;
        }
        group += ATEN_THREADS;
    }
}
