// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Fixed-width ordinary RMSNorm retaining the generic 256-lane reduction.
//!
//! Each lane keeps its six strided values through the reduction, avoiding a
//! second input/residual read. Arithmetic order, FP32 residual statistics and
//! BF16 output boundaries match `decoder::rms_norm::<ADD, false>`.

use cuda_device::{SharedArray, float, thread};

use crate::{block_sum, read, write};

const WIDTH: usize = 1536;
const THREADS: usize = 256;
const VALUES: usize = WIDTH / THREADS;

/// Grid=(rows,1,1), block=256. Outputs must not overlap each other; either
/// output may alias an equal-sized input. All lane values are loaded before
/// the shared reduction barriers permit any output writes.
#[inline(always)]
pub(crate) unsafe fn rms_norm_1536<const ADD: bool>(
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
    if row >= rows as usize || width as usize != WIDTH {
        return;
    }
    let offset = row * WIDTH;
    let tid = thread::threadIdx_x() as usize;
    let mut values = [0.0f32; VALUES];
    let mut sum = 0.0;
    let mut item = 0;
    while item < VALUES {
        let index = offset + tid + item * THREADS;
        let value = unsafe { read(input, index) };
        let value = if ADD {
            float::add_rn_f32(value, unsafe { read(residual, index) })
        } else {
            value
        };
        values[item] = value;
        sum = float::fma_rn_f32(value, value, sum);
        item += 1;
    }
    static mut SCRATCH: SharedArray<f32, 8> = SharedArray::UNINIT;
    let scratch = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCRATCH) };
    // Explicit rounded division retains the generic runtime-width operation;
    // multiplying by a rounded reciprocal would change some BF16 boundaries.
    let mean = float::div_rn_f32(unsafe { block_sum(sum, scratch) }, WIDTH as f32);
    let inverse = float::rsqrt_approx_f32(float::add_rn_f32(mean, eps));
    item = 0;
    while item < VALUES {
        let column = tid + item * THREADS;
        let index = offset + column;
        let value = values[item];
        let normalized = float::mul_rn_f32(value, inverse);
        let result = float::mul_rn_f32(normalized, unsafe { read(weight, column) });
        if ADD {
            unsafe { write(summed, index, value) };
        }
        unsafe { write(out, index, result) };
        item += 1;
    }
}
