/*
 * Fused word-only embedding gather + H=768 BF16 affine LayerNorm.
 *
 * Derived from the public PyTorch CUDA implementation in
 * aten/src/ATen/native/cuda/layer_norm_kernel.cu at commit
 * 5c4886908584029761b579af026dcfb627c84070, in particular
 * vectorized_layer_norm_kernel / compute_stats / cuWelfordOnlineSum /
 * cuWelfordCombine, with the CUDA launch geometry produced by
 * launch_vectorized_layer_norm_kernel for T=BFloat16, T_ACC=float.
 *
 * The upstream BSD-3-Clause notice is preserved in full below.
 *
 * From PyTorch:
 *
 * Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
 * Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
 * Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
 * Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
 * Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
 * Copyright (c) 2011-2013 NYU                      (Clement Farabet)
 * Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
 * Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
 * Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
 *
 * From Caffe2:
 *
 * Copyright (c) 2016-present, Facebook Inc. All rights reserved.
 *
 * All contributions by Facebook:
 * Copyright (c) 2016 Facebook Inc.
 *
 * All contributions by Google:
 * Copyright (c) 2015 Google Inc.
 * All rights reserved.
 *
 * All contributions by Yangqing Jia:
 * Copyright (c) 2015 Yangqing Jia
 * All rights reserved.
 *
 * All contributions by Kakao Brain:
 * Copyright 2019-2020 Kakao Brain
 *
 * All contributions by Cruise LLC:
 * Copyright (c) 2022 Cruise LLC.
 * All rights reserved.
 *
 * All contributions by Tri Dao:
 * Copyright (c) 2024 Tri Dao.
 * All rights reserved.
 *
 * All contributions by Arm:
 * Copyright (c) 2021, 2023-2025 Arm Limited and/or its affiliates
 *
 * All contributions from Caffe:
 * Copyright(c) 2013, 2014, 2015, the respective contributors
 * All rights reserved.
 *
 * All other contributions:
 * Copyright(c) 2015, 2016 the respective contributors
 * All rights reserved.
 *
 * Caffe2 uses a copyright model similar to Caffe: each contributor holds
 * copyright over their contributions to Caffe2. The project versioning records
 * all such contribution and copyright details. If a contributor wants to further
 * mark their specific copyright on a particular contribution, they should
 * indicate their copyright solely in the commit message of the change when it is
 * committed.
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
 *    and IDIAP Research Institute nor the names of its contributors may be
 *    used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

use cuda_device::{SharedArray, float, thread, warp};

use crate::{read, write};

const HIDDEN: usize = 768;
const VEC_SIZE: usize = 4;
const N_VECTORS: usize = HIDDEN / VEC_SIZE;
const BLOCK_X: usize = 32;
const BLOCK_Y: usize = 4;
const NUM_THREADS: usize = BLOCK_X * BLOCK_Y;
const SMEM_FLOATS: usize = BLOCK_Y * 3 / 2;

#[derive(Copy, Clone)]
pub(crate) struct WelfordData {
    pub(crate) mean: f32,
    pub(crate) sigma2: f32,
    pub(crate) count: f32,
}

impl WelfordData {
    #[inline(always)]
    pub(crate) fn zero() -> Self {
        Self {
            mean: 0.0,
            sigma2: 0.0,
            count: 0.0,
        }
    }
}

#[inline(always)]
pub(crate) fn welford_online_sum(value: f32, current: WelfordData) -> WelfordData {
    let delta = value - current.mean;
    let new_count = current.count + 1.0;
    let reciprocal = float::div_rn_f32(1.0, new_count);
    let new_mean = float::fma_rn_f32(delta, reciprocal, current.mean);
    let new_sigma2 = float::fma_rn_f32(delta, value - new_mean, current.sigma2);
    WelfordData {
        mean: new_mean,
        sigma2: new_sigma2,
        count: new_count,
    }
}

#[inline(always)]
pub(crate) fn welford_combine(data_b: WelfordData, data_a: WelfordData) -> WelfordData {
    let delta = data_b.mean - data_a.mean;
    let count = data_a.count + data_b.count;
    if count > 0.0 {
        let coefficient = float::div_rn_f32(1.0, count);
        let n_a = data_a.count * coefficient;
        let n_b = data_b.count * coefficient;
        let mean = float::fma_rn_f32(n_a, data_a.mean, n_b * data_b.mean);
        let sigma2 = float::fma_rn_f32(
            delta * delta * data_a.count,
            n_b,
            data_a.sigma2 + data_b.sigma2,
        );
        WelfordData {
            mean,
            sigma2,
            count,
        }
    } else {
        WelfordData {
            mean: 0.0,
            sigma2: 0.0,
            count,
        }
    }
}

#[inline(always)]
pub(crate) unsafe fn embedding_norm_word_768_aten(
    ids: *const u32,
    word: *const u16,
    gamma: *const u16,
    beta: *const u16,
    out: *mut u16,
    rows: u32,
    width: u32,
    eps: f32,
) {
    if width != HIDDEN as u32 || rows == 0 {
        return;
    }

    let row = thread::blockIdx_x() as usize;
    if row >= rows as usize {
        return;
    }

    let tid = thread::threadIdx_x() as usize;
    let lane = tid % BLOCK_X;
    let warp = tid / BLOCK_X;

    static mut SCRATCH: SharedArray<f32, SMEM_FLOATS> = SharedArray::UNINIT;
    let smem = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCRATCH) };
    let table_row = unsafe { *ids.add(row) } as usize * HIDDEN;
    let out_row = row * HIDDEN;

    let mut wd = WelfordData::zero();
    let mut vector_index = tid;
    while vector_index < N_VECTORS {
        let base = table_row + vector_index * VEC_SIZE;
        let mut element = 0;
        while element < VEC_SIZE {
            let value = unsafe { read(word, base + element) };
            wd = welford_online_sum(value, wd);
            element += 1;
        }
        vector_index += NUM_THREADS;
    }

    let mut offset = BLOCK_X / 2;
    while offset > 0 {
        let shuffled = WelfordData {
            mean: warp::shuffle_down_f32(wd.mean, offset as u32),
            sigma2: warp::shuffle_down_f32(wd.sigma2, offset as u32),
            count: warp::shuffle_down_f32(wd.count, offset as u32),
        };
        wd = welford_combine(wd, shuffled);
        offset /= 2;
    }

    let mut offset = BLOCK_Y / 2;
    while offset > 0 {
        if lane == 0 && warp >= offset && warp < 2 * offset {
            let write_y = warp - offset;
            unsafe { *smem.add(2 * write_y) = wd.mean };
            unsafe { *smem.add(2 * write_y + 1) = wd.sigma2 };
            unsafe { *smem.add(BLOCK_Y + write_y) = wd.count };
        }
        thread::sync_threads();

        if lane == 0 && warp < offset {
            let shuffled = WelfordData {
                mean: unsafe { *smem.add(2 * warp) },
                sigma2: unsafe { *smem.add(2 * warp + 1) },
                count: unsafe { *smem.add(BLOCK_Y + warp) },
            };
            wd = welford_combine(wd, shuffled);
        }
        thread::sync_threads();

        offset /= 2;
    }

    if lane == 0 && warp == 0 {
        unsafe { *smem = wd.mean };
        unsafe { *smem.add(1) = float::div_rn_f32(wd.sigma2, HIDDEN as f32) };
    }
    thread::sync_threads();

    let mean = unsafe { *smem };
    let variance = unsafe { *smem.add(1) };
    let rstd = float::rsqrt_approx_f32(variance + eps);

    let mut vector_index = tid;
    while vector_index < N_VECTORS {
        let column = vector_index * VEC_SIZE;
        let mut element = 0;
        while element < VEC_SIZE {
            let index = column + element;
            let x = unsafe { read(word, table_row + index) };
            let gamma_value = unsafe { read(gamma, index) };
            let beta_value = unsafe { read(beta, index) };
            let normalized = rstd * (x - mean);
            let result = float::fma_rn_f32(gamma_value, normalized, beta_value);
            unsafe { write(out, out_row + index, result) };
            element += 1;
        }
        vector_index += NUM_THREADS;
    }
}
