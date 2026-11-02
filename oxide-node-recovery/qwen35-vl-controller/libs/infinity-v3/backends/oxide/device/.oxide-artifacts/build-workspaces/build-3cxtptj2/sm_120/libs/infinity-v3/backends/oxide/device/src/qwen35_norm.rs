// SPDX-License-Identifier: MIT AND LicenseRef-Infinity-Restricted-MIT-1.0
// Upstream Candle kernel portions retain MIT; new original port contributions
// use the Infinity Restricted MIT License 1.0. No inherited grant is changed.
//
// Adapted from the published tracked source
//   libs/infinity-v3/backends/libtorch/cpp/kernels/qwen35_reference.cu
//   (`norm_tree<V>` and the reference qwen35 norm launches), itself adapted
//   from the original Candle qwen35_gdn.cu and Candle fast_sum at
//   036f39e44015e78ba18d716d73dfdc2e78730c44. Original licenses retained.
// This translation unit alone used the original --fmad=false contract; the
// port below reproduces that contract with explicit round-to-nearest FP32
// intrinsics and no contractions or flush-to-zero.
// Original contributions Copyright Infinity Core Contributors,
// Licensed under the Infinity Restricted MIT License 1.0; see the repository
// LICENSE, NOTICE and THIRD_PARTY_NOTICES.md. The Infinity Restricted MIT
// License 1.0 is not the standard MIT License.
//
// Candle MIT notice retained:
// Copyright The Hugging Face Team. All rights reserved.
// Permission is hereby granted, free of charge, to any
// person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the
// Software without restriction, including without
// limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software
// is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice
// shall be included in all copies or substantial portions
// of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
// ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
// TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
// PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
// SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
// IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//! Exact Qwen3.5 BF16 norm device primitives ported from native
//! `norm_tree<V>` in `qwen35_reference.cu`.
//!
//! The four exports are source-only device entry points. The host owns the
//! positive geometry, the one-CTA-per-row launch, and the logical-to-physical
//! thread split described in `HANDOFF.md`. Vector entry points retain the
//! native V=4 logical lane grouping; this first correct port performs plain
//! 16-bit BF16 element accesses, so `V=4` changes lane ownership and the
//! reduction tree, not the width of each memory instruction.
//!
//! Arithmetic follows the native source order with explicit FP32
//! round-to-nearest and no FMA contraction or FTZ:
//!
//! * per logical lane `sum = add_rn(sum, mul_rn(z, z))`, where `z` is the BF16
//!   value widened to FP32;
//! * the 1024-slot FP32 shared tree runs while `s >= 32 * V`, then the
//!   remaining logical lanes are reduced in warp registers with
//!   `shfl.sync.down` over `s / V` lanes and folded `V/2`, `V/4`, ... in
//!   index order;
//! * `mean = mul_rn(sum, div_rn(1, width))` and
//!   `inv = div_rn(1, sqrt_rn(add_rn(mean, eps)))`;
//! * output is `bf16_rn(mul_rn(mul_rn(z, inv), add_rn(gamma, 1)))`.
//!
//! Residual entry points first store `bf16_rn(add_rn(x, addition))` to
//! `updated`, take every statistic from those rounded BF16 cells, and
//! normalize the same `updated` cells. `x` may alias `out` or `updated`, and
//! `addition` may alias the independent cell input, because every cell is
//! read before its own write and all statistic reads complete before the final
//! barrier releases the output pass. Plain norm permits input/output aliasing
//! only for identical contiguous row layouts; strided compaction must use a
//! disjoint output because separate CTAs have no cross-row barrier.

use cuda_device::{float, thread, warp, SharedArray};

use crate::{bf16_from_f32, f32_from_bf16};

/// Native `__shared__ float sums[1024]`.
const SHARED_LANES: usize = 1024;

/// Shared row reduction followed by normalization, generic over the native
/// adjacent-lane width `V` and the residual staging flag.
///
/// `blockIdx.x` is the row and `blockDim.x * V` is the logical lane count.
/// Every barrier is reached by every thread of the CTA; the only early return
/// depends on the CTA-uniform row index. The caller must launch this as the
/// body of a real device kernel.
#[inline(always)]
unsafe fn norm_tree<const V: usize, const RESIDUAL: bool>(
    x: *const u16,
    gamma: *const u16,
    out: *mut u16,
    rows: u32,
    width: u32,
    heads: u32,
    stride0: u64,
    stride1: u64,
    eps: f32,
    addition: *const u16,
    updated: *mut u16,
    sums: *mut f32,
) {
    let row = thread::blockIdx_x() as usize;
    if row >= rows as usize || width == 0 || heads == 0 {
        return;
    }

    let heads = heads as usize;
    let width = width as usize;
    let logical = thread::threadIdx_x() as usize * V;
    let threads = thread::blockDim_x() as usize * V;
    let base = (row / heads) * stride0 as usize + (row % heads) * stride1 as usize;

    let mut sum = [0.0f32; V];
    let mut d = logical;
    while d < width {
        let mut values = [0u16; V];
        let mut j = 0;
        while j < V {
            let value = unsafe { *x.add(base + d + j) };
            values[j] = if RESIDUAL {
                let delta = unsafe { *addition.add(base + d + j) };
                bf16_from_f32(float::add_rn_f32(
                    f32_from_bf16(value),
                    f32_from_bf16(delta),
                ))
            } else {
                value
            };
            j += 1;
        }
        if RESIDUAL {
            let mut j = 0;
            while j < V {
                unsafe { *updated.add(base + d + j) = values[j] };
                j += 1;
            }
        }
        let mut j = 0;
        while j < V {
            let z = f32_from_bf16(values[j]);
            sum[j] = float::add_rn_f32(sum[j], float::mul_rn_f32(z, z));
            j += 1;
        }
        d += threads;
    }

    let mut j = 0;
    while j < V {
        unsafe { *sums.add(logical + j) = sum[j] };
        j += 1;
    }

    let mut s = threads / 2;
    while s >= 32 * V {
        thread::sync_threads();
        let mut j = 0;
        while j < V {
            if logical + j < s {
                unsafe {
                    let left = *sums.add(logical + j);
                    let right = *sums.add(logical + j + s);
                    *sums.add(logical + j) = float::add_rn_f32(left, right);
                }
            }
            j += 1;
        }
        s >>= 1;
    }

    if logical < 32 * V {
        let mut value = [0.0f32; V];
        let mut j = 0;
        while j < V {
            value[j] = unsafe { *sums.add(logical + j) };
            j += 1;
        }
        let mask = warp::active_mask();
        let mut s = core::cmp::min(threads / 2, 16 * V);
        while s >= V {
            let mut j = 0;
            while j < V {
                value[j] = float::add_rn_f32(
                    value[j],
                    warp::shuffle_down_f32_sync(mask, value[j], (s / V) as u32),
                );
                j += 1;
            }
            s >>= 1;
        }
        let mut s = V / 2;
        while s != 0 {
            let mut j = 0;
            while j < s {
                value[j] = float::add_rn_f32(value[j], value[j + s]);
                j += 1;
            }
            s >>= 1;
        }
        if logical == 0 {
            let mean = float::mul_rn_f32(value[0], float::div_rn_f32(1.0, width as f32));
            unsafe {
                *sums = float::div_rn_f32(1.0, float::sqrt_rn_f32(float::add_rn_f32(mean, eps)));
            }
        }
    }
    thread::sync_threads();

    let inverse = unsafe { *sums };
    let source: *const u16 = if RESIDUAL { updated } else { x };
    let out_base = row * width;
    let mut d = logical;
    while d < width {
        let mut z = [0.0f32; V];
        let mut weight = [0.0f32; V];
        let mut j = 0;
        while j < V {
            z[j] = f32_from_bf16(unsafe { *source.add(base + d + j) });
            weight[j] = float::add_rn_f32(f32_from_bf16(unsafe { *gamma.add(d + j) }), 1.0);
            j += 1;
        }
        let mut result = [0u16; V];
        j = 0;
        while j < V {
            result[j] = bf16_from_f32(float::mul_rn_f32(
                float::mul_rn_f32(z[j], inverse),
                weight[j],
            ));
            j += 1;
        }
        j = 0;
        while j < V {
            unsafe { *out.add(out_base + d + j) = result[j] };
            j += 1;
        }
        d += threads;
    }
}

/// Native `norm_tree<4>` for a positive strided BF16 row.
///
/// One CTA per row; `blockDim.x` is the logical thread count divided by four.
/// `width` must be a multiple of four and the x, gamma and out row bases must
/// be eight-byte aligned so the native V=4 cell mapping is well defined.
#[inline(always)]
pub(crate) unsafe fn norm_vector(
    x: *const u16,
    gamma: *const u16,
    out: *mut u16,
    rows: u32,
    width: u32,
    heads: u32,
    stride0: u64,
    stride1: u64,
    eps: f32,
) {
    if width % 4 != 0 {
        return;
    }
    static mut SUMS: SharedArray<f32, SHARED_LANES> = SharedArray::UNINIT;
    let sums = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SUMS) };
    unsafe {
        norm_tree::<4, false>(
            x,
            gamma,
            out,
            rows,
            width,
            heads,
            stride0,
            stride1,
            eps,
            core::ptr::null(),
            core::ptr::null_mut(),
            sums,
        )
    };
}

/// Native `norm_tree<1>` for a positive strided BF16 row.
///
/// One CTA per row with one logical lane per physical thread. This is the
/// meaningful fallback for odd widths such as 255 or 257 and for any geometry
/// that does not satisfy the vector preconditions.
#[inline(always)]
pub(crate) unsafe fn norm_scalar(
    x: *const u16,
    gamma: *const u16,
    out: *mut u16,
    rows: u32,
    width: u32,
    heads: u32,
    stride0: u64,
    stride1: u64,
    eps: f32,
) {
    static mut SUMS: SharedArray<f32, SHARED_LANES> = SharedArray::UNINIT;
    let sums = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SUMS) };
    unsafe {
        norm_tree::<1, false>(
            x,
            gamma,
            out,
            rows,
            width,
            heads,
            stride0,
            stride1,
            eps,
            core::ptr::null(),
            core::ptr::null_mut(),
            sums,
        )
    };
}

/// Native `norm_tree<4, true>` for contiguous `[rows, width]` BF16 storage.
///
/// `updated` receives `bf16_rn(x + addition)` and every statistic is taken
/// from that rounded cell. `out` receives the normalized affine value.
#[inline(always)]
pub(crate) unsafe fn residual_norm_vector(
    x: *const u16,
    addition: *const u16,
    gamma: *const u16,
    out: *mut u16,
    updated: *mut u16,
    rows: u32,
    width: u32,
    eps: f32,
) {
    if width % 4 != 0 {
        return;
    }
    static mut SUMS: SharedArray<f32, SHARED_LANES> = SharedArray::UNINIT;
    let sums = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SUMS) };
    unsafe {
        norm_tree::<4, true>(
            x,
            gamma,
            out,
            rows,
            width,
            1,
            width as u64,
            0,
            eps,
            addition,
            updated,
            sums,
        )
    };
}

/// Native `norm_tree<1, true>` for contiguous `[rows, width]` BF16 storage.
///
/// The scalar fallback matches the vector residual entry point arithmetic and
/// the native width-rounded-to-a-power-of-two logical lane count.
#[inline(always)]
pub(crate) unsafe fn residual_norm_scalar(
    x: *const u16,
    addition: *const u16,
    gamma: *const u16,
    out: *mut u16,
    updated: *mut u16,
    rows: u32,
    width: u32,
    eps: f32,
) {
    static mut SUMS: SharedArray<f32, SHARED_LANES> = SharedArray::UNINIT;
    let sums = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SUMS) };
    unsafe {
        norm_tree::<1, true>(
            x,
            gamma,
            out,
            rows,
            width,
            1,
            width as u64,
            0,
            eps,
            addition,
            updated,
            sums,
        )
    };
}
