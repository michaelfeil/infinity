// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Classifier output preseed kernel for the prepared biased-linear candidate.
//!
//! The host owns every allocation and validates logical extents before launch.
//! Each logical `C[row,label]` receives the BF16 bias bit pattern unchanged; the
//! following FP32-accumulating cuBLAS `beta=1` operation adds it before the one
//! BF16 output rounding. Storage tails outside the logical extent stay untouched.

use cuda_device::thread;

/// Seed every logical BF16 output element from the bias vector.
/// Grid=ceil(rows*width/256), block=256; out-of-range threads return.
#[inline(always)]
pub(crate) unsafe fn bias_seed_bf16(bias: *const u16, out: *mut u16, rows: u32, width: u32) {
    if rows == 0 || width == 0 {
        return;
    }
    let index = thread::blockIdx_x() as u64 * 256 + thread::threadIdx_x() as u64;
    let total = rows as u64 * width as u64;
    if index >= total {
        return;
    }
    let column = (index % width as u64) as usize;
    unsafe { *out.add(index as usize) = *bias.add(column) };
}

/// Gather each sequence's first packed BF16 row into contiguous output rows.
/// `bounds` holds `rows + 1` cumulative token offsets. Grid=ceil(rows*width/256),
/// block=256; the copy is an exact 16-bit move with no numerical conversion.
#[inline(always)]
pub(crate) unsafe fn gather_first_rows_bf16(
    source: *const u16,
    bounds: *const u32,
    out: *mut u16,
    rows: u32,
    width: u32,
) {
    if rows == 0 || width == 0 {
        return;
    }
    let index = thread::blockIdx_x() as u64 * 256 + thread::threadIdx_x() as u64;
    let total = rows as u64 * width as u64;
    if index >= total {
        return;
    }
    let row = (index / width as u64) as usize;
    let column = (index % width as u64) as usize;
    let start = unsafe { *bounds.add(row) } as usize;
    unsafe { *out.add(index as usize) = *source.add(start * width as usize + column) };
}
