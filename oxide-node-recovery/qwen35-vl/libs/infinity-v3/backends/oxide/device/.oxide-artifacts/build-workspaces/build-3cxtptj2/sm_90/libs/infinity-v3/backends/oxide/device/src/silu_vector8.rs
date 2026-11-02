// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Native-recipe BF16 SiLU with eight output values per thread.
//!
//! Complete aligned chunks move gate, up and output as 16-byte vectors. The
//! shared paired arithmetic retains every BF16 stage and native division.
//! Partial chunks and unaligned rows use bounded scalar loads and stores.
//! The caller validates packed [rows,2*width] storage and distinct output.

use cuda_device::{ptx_asm, thread};

use crate::silu_pairs::pair_result;

const CHUNK: usize = 8;

#[inline(always)]
fn aligned(pointer: *const u16) -> bool {
    (pointer as usize) & 15 == 0
}

#[inline(always)]
unsafe fn load_vector(pointer: *const u16) -> [u32; 4] {
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
    [first, second, third, fourth]
}

#[inline(always)]
unsafe fn store_vector(pointer: *mut u16, words: [u32; 4]) {
    let address = pointer as usize as u64;
    let [first, second, third, fourth] = words;
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
unsafe fn full_pair(gate: u32, up: u32) -> u32 {
    unsafe {
        pair_result::<true>(
            (gate as u16, (gate >> 16) as u16, true),
            (up as u16, (up >> 16) as u16, true),
        )
    }
}

/// Grid=ceil(rows*ceil(width/8)/256), block=256; no padded row access.
#[inline(always)]
pub(crate) unsafe fn silu_mul_bf16_vector8_native(
    input: *const u16,
    out: *mut u16,
    rows: u32,
    width: u32,
) {
    if width == 0 {
        return;
    }
    let width = width as usize;
    let chunks = width.div_ceil(CHUNK);
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    let row = index / chunks;
    if row >= rows as usize {
        return;
    }
    let column = (index % chunks) * CHUNK;
    let remaining = width - column;
    let gate = unsafe { input.add(row * 2 * width + column) };
    let up = unsafe { gate.add(width) };
    let output = unsafe { out.add(row * width + column) };
    if remaining >= CHUNK && aligned(gate) && aligned(up) && aligned(output.cast_const()) {
        let [g0, g1, g2, g3] = unsafe { load_vector(gate) };
        let [u0, u1, u2, u3] = unsafe { load_vector(up) };
        let words = unsafe {
            [
                full_pair(g0, u0),
                full_pair(g1, u1),
                full_pair(g2, u2),
                full_pair(g3, u3),
            ]
        };
        unsafe { store_vector(output, words) };
    } else {
        let mut offset = 0;
        while offset < CHUNK && offset < remaining {
            let second = offset + 1 < remaining;
            let first_gate = unsafe { *gate.add(offset) };
            let first_up = unsafe { *up.add(offset) };
            let second_gate = if second {
                unsafe { *gate.add(offset + 1) }
            } else {
                0
            };
            let second_up = if second {
                unsafe { *up.add(offset + 1) }
            } else {
                0
            };
            let result = unsafe {
                pair_result::<true>(
                    (first_gate, second_gate, second),
                    (first_up, second_up, second),
                )
            };
            unsafe { *output.add(offset) = result as u16 };
            if second {
                unsafe { *output.add(offset + 1) = (result >> 16) as u16 };
            }
            offset += 2;
        }
    }
}
