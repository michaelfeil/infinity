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
//! `norm_tree<V>` in `qwen35_reference.cu`, with an eight-byte memory
//! transaction for each adjacent group of four BF16 cells.
//!
//! The arithmetic, the V=4 logical lane mapping, the shared reduction and
//! its barriers are the inherited `norm_tree<4, _>` source order. Only the
//! memory access width changes: an aligned four-u16 group uses one 64-bit
//! `ld.global.v2.u32` or `st.global.v2.u32`, while each pointer independently
//! falls back to four scalar u16 accesses when that particular group is not
//! eight-byte aligned.
//!
//! Arithmetic follows the native source order with explicit FP32
//! round-to-nearest and no FMA contraction or flush-to-zero:
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
//! `addition` may alias the independent cell input, because every four-cell
//! group is loaded before its own cells are written and all statistic reads
//! complete before the final barrier releases the output pass. Plain norm
//! permits input/output aliasing only for identical contiguous row layouts;
//! strided compaction must use a disjoint output because separate CTAs have no
//! cross-row barrier.
//!
//! No row-base alignment is required. Each x, addition, updated, gamma,
//! source and out group makes its own eight-byte alignment decision; a
//! width that is a positive multiple of four is the only geometric contract.

use cuda_device::{float, ptx_asm, thread, warp, SharedArray};

use crate::{bf16_from_f32, f32_from_bf16};

/// Native `__shared__ float sums[1024]`.
const SHARED_LANES: usize = 1024;

/// True when one four-u16 group can use a single eight-byte transaction.
#[inline(always)]
fn qwen35_norm_mem_aligned(pointer: *const u16) -> bool {
    (pointer as usize) & 7 == 0
}

/// True when one four-u16 output group can use a single eight-byte transaction.
#[inline(always)]
fn qwen35_norm_mem_aligned_mut(pointer: *mut u16) -> bool {
    (pointer as usize) & 7 == 0
}

/// Load four adjacent BF16 cells as one 64-bit transaction when this exact
/// pointer is eight-byte aligned, otherwise use four scalar u16 loads.
#[inline(always)]
unsafe fn qwen35_norm_mem_load4(pointer: *const u16) -> [u16; 4] {
    if qwen35_norm_mem_aligned(pointer) {
        let address = pointer as usize as u64;
        let first: u32;
        let second: u32;
        unsafe {
            ptx_asm!(
                "ld.global.v2.u32 {%0, %1}, [%2];",
                out("=r") first,
                out("=r") second,
                in("l") address,
                clobber("memory"),
            );
        }
        [
            first as u16,
            (first >> 16) as u16,
            second as u16,
            (second >> 16) as u16,
        ]
    } else {
        let mut values = [0u16; 4];
        let mut j = 0;
        while j < 4 {
            values[j] = unsafe { *pointer.add(j) };
            j += 1;
        }
        values
    }
}

/// Store four adjacent BF16 cells as one 64-bit transaction when this exact
/// pointer is eight-byte aligned, otherwise use four scalar u16 stores.
#[inline(always)]
unsafe fn qwen35_norm_mem_store4(pointer: *mut u16, values: [u16; 4]) {
    if qwen35_norm_mem_aligned_mut(pointer) {
        let address = pointer as usize as u64;
        let first = values[0] as u32 | ((values[1] as u32) << 16);
        let second = values[2] as u32 | ((values[3] as u32) << 16);
        unsafe {
            ptx_asm!(
                "st.global.v2.u32 [%0], {%1, %2};",
                in("l") address,
                in("r") first,
                in("r") second,
                clobber("memory"),
            );
        }
    } else {
        let mut j = 0;
        while j < 4 {
            unsafe { *pointer.add(j) = values[j] };
            j += 1;
        }
    }
}

/// Shared row reduction followed by normalization with the inherited native
/// `norm_tree<V=4, RESIDUAL>` arithmetic and lane mapping.
///
/// `blockIdx.x` is the row and `blockDim.x * 4` is the logical lane count.
/// Every barrier is reached by every thread of the CTA; the only early return
/// depends on the CTA-uniform row index. The caller must launch this as the
/// body of a real device kernel.
#[inline(always)]
unsafe fn qwen35_norm_mem_tree<const RESIDUAL: bool>(
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
    const V: usize = 4;

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
        let input = unsafe { qwen35_norm_mem_load4(x.add(base + d)) };
        let mut values = [0u16; V];
        if RESIDUAL {
            let delta = unsafe { qwen35_norm_mem_load4(addition.add(base + d)) };
            let mut j = 0;
            while j < V {
                values[j] = bf16_from_f32(float::add_rn_f32(
                    f32_from_bf16(input[j]),
                    f32_from_bf16(delta[j]),
                ));
                j += 1;
            }
            unsafe { qwen35_norm_mem_store4(updated.add(base + d), values) };
        } else {
            let mut j = 0;
            while j < V {
                values[j] = input[j];
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
        let source_values = unsafe { qwen35_norm_mem_load4(source.add(base + d)) };
        let gamma_values = unsafe { qwen35_norm_mem_load4(gamma.add(d)) };
        let mut z = [0.0f32; V];
        let mut weight = [0.0f32; V];
        let mut j = 0;
        while j < V {
            z[j] = f32_from_bf16(source_values[j]);
            weight[j] = float::add_rn_f32(f32_from_bf16(gamma_values[j]), 1.0);
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
        unsafe { qwen35_norm_mem_store4(out.add(out_base + d), result) };
        d += threads;
    }
}

/// Native `norm_tree<4>` arithmetic with four-cell vector memory transactions
/// for a positive strided BF16 row.
///
/// One CTA per row; `blockDim.x` is the logical thread count divided by four.
/// `width` must be a positive multiple of four. No pointer alignment is
/// required: each group of four adjacent cells uses the 64-bit transaction
/// only when that group's own pointer is eight-byte aligned.
#[inline(always)]
pub(crate) unsafe fn norm_vector_memory(
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
        qwen35_norm_mem_tree::<false>(
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

/// Native `norm_tree<4, true>` arithmetic with four-cell vector memory
/// transactions for contiguous `[rows, width]` BF16 storage.
///
/// `updated` receives `bf16_rn(x + addition)` and every statistic is taken
/// from that rounded cell. `out` receives the normalized affine value. No
/// pointer alignment is required; every group of four cells makes its own
/// eight-byte alignment decision.
#[inline(always)]
pub(crate) unsafe fn residual_norm_vector_memory(
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
        qwen35_norm_mem_tree::<true>(
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
