// SPDX-License-Identifier: Apache-2.0 AND LicenseRef-Infinity-Restricted-MIT-1.0
// Copyright Infinity Core Contributors.
// Original contributions licensed under the Infinity Restricted MIT License 1.0;
// see the repository LICENSE.
//! Fixed-width H4096 BF16 RMSNorm with aligned 128-bit vector I/O.
//!
//! Arithmetic follows `decoder::rms_norm::<ADD, false>` exactly: 256 lanes
//! reduce sixteen strided FP32 values in `tid + j * 256` order, ADD residual
//! sums happen before statistics, and the FP32 value sum is reduced through
//! the unchanged `block_sum`. The native vector-transfer layout comes from
//! `backends/libtorch/cpp/kernels/fast_kernels.cu`
//! (`add_rms_bf16_vector_kernel`, Apache-2.0), whose grant remains unchanged;
//! this kernel additionally defers every summed write until after the CTA
//! barrier so an output may alias its equal-sized input/residual.
//! Unaligned required pointers take the unchanged generic scalar export so the
//! ABI always works.

use cuda_device::{SharedArray, float, ptx_asm, thread};

use crate::{bf16_from_f32, block_sum, f32_from_bf16};

const WIDTH: usize = 4096;
const THREADS: usize = 256;
const VALUES: usize = WIDTH / THREADS;
const LANE_VALUES: usize = 8;
const IO_VECTORS: usize = WIDTH / LANE_VALUES;
const VECTOR_BYTES: usize = LANE_VALUES * core::mem::size_of::<u16>();

#[inline(always)]
fn aligned(pointer: *const u16) -> bool {
    (pointer as usize) & (VECTOR_BYTES - 1) == 0
}

#[inline(always)]
fn aligned_mut(pointer: *mut u16) -> bool {
    (pointer as usize) & (VECTOR_BYTES - 1) == 0
}

#[inline(always)]
unsafe fn load_vector(pointer: *const u16) -> [u16; LANE_VALUES] {
    let address = pointer as usize as u64;
    let first: u32;
    let second: u32;
    let third: u32;
    let fourth: u32;
    unsafe {
        ptx_asm!(
            "ld.global.v4.u32 {%0, %1, %2, %3}, [%4];",
            out("=r") first,
            out("=r") second,
            out("=r") third,
            out("=r") fourth,
            in("l") address,
            clobber("memory"),
        );
    }
    [
        first as u16,
        (first >> 16) as u16,
        second as u16,
        (second >> 16) as u16,
        third as u16,
        (third >> 16) as u16,
        fourth as u16,
        (fourth >> 16) as u16,
    ]
}

#[inline(always)]
unsafe fn store_vector(pointer: *mut u16, values: [u16; LANE_VALUES]) {
    let address = pointer as usize as u64;
    let first = values[0] as u32 | ((values[1] as u32) << 16);
    let second = values[2] as u32 | ((values[3] as u32) << 16);
    let third = values[4] as u32 | ((values[5] as u32) << 16);
    let fourth = values[6] as u32 | ((values[7] as u32) << 16);
    unsafe {
        ptx_asm!(
            "st.global.v4.u32 [%0], {%1, %2, %3, %4};",
            in("l") address,
            in("r") first,
            in("r") second,
            in("r") third,
            in("r") fourth,
            clobber("memory"),
        );
    }
}

#[inline(always)]
unsafe fn normalize_vector<const ADD: bool>(
    input: *const u16,
    residual: *const u16,
    weight: *const u16,
    out: *mut u16,
    summed: *mut u16,
    offset: usize,
    eps: f32,
) {
    static mut SHARED_VALUES: SharedArray<f32, WIDTH> = SharedArray::UNINIT;
    static mut SCRATCH: SharedArray<f32, 8> = SharedArray::UNINIT;
    let values = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SHARED_VALUES) };
    let scratch = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCRATCH) };
    let tid = thread::threadIdx_x() as usize;

    let mut group = tid;
    while group < IO_VECTORS {
        let source = group * LANE_VALUES;
        let input_values = unsafe { load_vector(input.add(offset + source)) };
        let residual_values = if ADD {
            unsafe { load_vector(residual.add(offset + source)) }
        } else {
            [0u16; LANE_VALUES]
        };
        let mut element = 0;
        while element < LANE_VALUES {
            let mut value = f32_from_bf16(input_values[element]);
            if ADD {
                value = float::add_rn_f32(value, f32_from_bf16(residual_values[element]));
            }
            unsafe { *values.add(source + element) = value };
            element += 1;
        }
        group += THREADS;
    }
    thread::sync_threads();
    if ADD {
        let mut group = tid;
        while group < IO_VECTORS {
            let source = group * LANE_VALUES;
            let mut stored = [0u16; LANE_VALUES];
            let mut element = 0;
            while element < LANE_VALUES {
                stored[element] = bf16_from_f32(unsafe { *values.add(source + element) });
                element += 1;
            }
            unsafe { store_vector(summed.add(offset + source), stored) };
            group += THREADS;
        }
    }
    let mut sum = 0.0;
    let mut item = 0;
    while item < VALUES {
        let value = unsafe { *values.add(tid + item * THREADS) };
        sum = float::fma_rn_f32(value, value, sum);
        item += 1;
    }
    let mean = float::div_rn_f32(unsafe { block_sum(sum, scratch) }, WIDTH as f32);
    let inverse = float::rsqrt_approx_f32(float::add_rn_f32(mean, eps));
    let mut group = tid;
    while group < IO_VECTORS {
        let source = group * LANE_VALUES;
        let weight_values = unsafe { load_vector(weight.add(source)) };
        let mut result = [0u16; LANE_VALUES];
        let mut element = 0;
        while element < LANE_VALUES {
            let value = unsafe { *values.add(source + element) };
            let normalized = float::mul_rn_f32(value, inverse);
            let scaled = float::mul_rn_f32(normalized, f32_from_bf16(weight_values[element]));
            result[element] = bf16_from_f32(scaled);
            element += 1;
        }
        unsafe { store_vector(out.add(offset + source), result) };
        group += THREADS;
    }
}

/// Alias contract: `out` and `summed` must be distinct. Either may alias an
/// equal-sized `input`/`residual` buffer; every input/residual read for the
/// entire row completes before the CTA barrier that precedes any summed write,
/// and the final output follows `block_sum`. Weight aliasing is not part of the
/// contract. Absent ADD residual/summed pointers are not read or written.
#[inline(always)]
pub(crate) unsafe fn rms_norm_4096_vector<const ADD: bool>(
    input: *const u16,
    residual: *const u16,
    weight: *const u16,
    out: *mut u16,
    summed: *mut u16,
    rows: u32,
    width: u32,
    eps: f32,
) {
    let row = thread::blockIdx_x() as usize;
    if width as usize != WIDTH || row >= rows as usize {
        unsafe {
            crate::decoder::rms_norm::<ADD, false>(
                input, residual, weight, out, summed, rows, width, eps,
            )
        };
        return;
    }
    let aligned_io = aligned(input)
        && aligned(weight)
        && aligned_mut(out)
        && (!ADD || (aligned(residual) && aligned_mut(summed)));
    if !aligned_io {
        unsafe {
            crate::decoder::rms_norm::<ADD, false>(
                input, residual, weight, out, summed, rows, width, eps,
            )
        };
        return;
    }
    unsafe { normalize_vector::<ADD>(input, residual, weight, out, summed, row * WIDTH, eps) };
}
