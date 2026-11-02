// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! DeBERTa stage boundaries around the existing native attention API.
//! Every arithmetic write is BF16-rounded just as the separate ATen operation.
use crate::{read, write};
use cuda_device::thread;

#[inline(always)]
pub(crate) unsafe fn unpack_scaled(
    packed: *const u16,
    query: *mut u16,
    key: *mut u16,
    value: *mut u16,
    elements: u32,
    width: u32,
    scale: f32,
) {
    let i = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if i >= elements as usize {
        return;
    }
    let h = width as usize;
    let row = i / h;
    let col = i % h;
    let base = row * (3 * h) + col;
    unsafe {
        *query.add(i) = *packed.add(base);
        write(key, i, read(packed, base + h) / scale);
        *value.add(i) = *packed.add(base + 2 * h);
    }
}

#[inline(always)]
pub(crate) unsafe fn relative_transpose(input: *const u16, out: *mut u16, rows: u32, width: u32) {
    let i = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    let rows = rows as usize;
    let width = width as usize;
    if i >= rows * width {
        return;
    }
    // [2*span,heads,dim] -> [heads,dim,2*span], preserving BF16 bits.
    let relative = i % rows;
    let feature = i / rows;
    unsafe { *out.add(i) = *input.add(relative * width + feature) };
}

#[inline(always)]
pub(crate) unsafe fn divide(input: *mut u16, elements: u32, scale: f32) {
    let i = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if i < elements as usize {
        unsafe { write(input, i, read(input, i) / scale) };
    }
}

#[inline(always)]
pub(crate) unsafe fn signed_buckets(
    out: *mut i32,
    length: u32,
    span: u32,
    logarithmic: u32,
    denominator: f32,
) {
    let i = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    let count = 2 * length as usize - 1;
    if i >= count {
        return;
    }
    let mut delta = i as i32 - (length as i32 - 1);
    if logarithmic != 0 {
        let distance = delta.unsigned_abs() as f32;
        let mid = span as f32 * 0.5;
        if distance > mid {
            let logged =
                libm::ceilf(libm::logf(distance.max(1.0) / mid) / denominator * (mid - 1.0)) + mid;
            let bucket = logged as i32;
            delta = if delta < 0 { -bucket } else { bucket };
        }
    }
    let index = (delta + span as i32).clamp(0, 2 * span as i32 - 1);
    unsafe { *out.add(i) = index };
}
