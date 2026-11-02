// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Exact native `cat(A, B, -1)` after separate contiguous projection GEMMs.
use cuda_device::{device, thread};

/// No arithmetic: each destination receives exactly one original BF16 bit pattern.
#[device]
pub unsafe fn join_ab(a: *const u16, b: *const u16, out: *mut u16, tokens: u32, heads: u32) {
    let index = thread::blockIdx_x() as usize * thread::blockDim_x() as usize
        + thread::threadIdx_x() as usize;
    let width = heads as usize;
    let total = tokens as usize * width * 2;
    if index < total {
        let row = index / (width * 2);
        let column = index % (width * 2);
        let value = if column < width {
            unsafe { *a.add(row * width + column) }
        } else {
            unsafe { *b.add(row * width + column - width) }
        };
        unsafe { *out.add(index) = value };
    }
}
