// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Exact erf GELU for the MPNet feed-forward epilogue.
//!
//! Native MPNet maps `hidden_act = "gelu"` to `at::gelu(x, "none")`, which is
//! `0.5*x*(1+erf(x/sqrt(2)))` in the model dtype's accumulation type. The host
//! validates the row geometry; this kernel performs one FP32 activation and
//! rounds only when writing BF16.

use crate::{read, write};
use cuda_device::thread;

const INV_SQRT_TWO: f32 = 0.707_106_77;

/// Grid=ceil(elements/256), block=256. In-place update of contiguous BF16 rows.
#[inline(always)]
pub(crate) unsafe fn gelu_exact(input: *mut u16, elements: u32) {
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if index >= elements as usize {
        return;
    }
    let value = unsafe { read(input, index) };
    let active = 0.5 * value * (1.0 + libm::erff(value * INV_SQRT_TWO));
    unsafe { write(input, index, active) };
}
