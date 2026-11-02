// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Affine DeBERTa adapters to the already qualified Candle H768 reduction.
//! Original reduction/affine order and BSD notice remain in norm768.rs.
use crate::{bf16_from_f32, f32_from_bf16, read};
use cuda_device::{float, thread};
const WIDTH: usize = 768;
const VALUES: usize = 24;

#[inline(always)]
fn column(lane: usize, i: usize) -> usize {
    8 * lane + i % 8 + 256 * (i / 8)
}

#[inline(always)]
pub(crate) unsafe fn word(
    ids: *const u32,
    weights: *const u16,
    gamma: *const u16,
    beta: *const u16,
    out: *mut u16,
    rows: u32,
    eps: f32,
) {
    let tid = thread::threadIdx_x() as usize;
    let row = thread::blockIdx_x() as usize * 4 + tid / 32;
    if row >= rows as usize {
        return;
    }
    let token = unsafe { *ids.add(row) } as usize;
    let mut values = [0.0; VALUES];
    let mut i = 0;
    while i < VALUES {
        values[i] = float::mul_rn_ftz_f32(
            unsafe { read(weights, token * WIDTH + column(tid & 31, i)) },
            1.0,
        );
        i += 1;
    }
    unsafe { crate::norm768::normalize(values, row * WIDTH, gamma, beta, out, eps) };
}

#[inline(always)]
pub(crate) unsafe fn rounded(
    input: *const u16,
    residual: *const u16,
    bias: *const u16,
    gamma: *const u16,
    beta: *const u16,
    out: *mut u16,
    rows: u32,
    eps: f32,
) {
    let tid = thread::threadIdx_x() as usize;
    let row = thread::blockIdx_x() as usize * 4 + tid / 32;
    if row >= rows as usize {
        return;
    }
    let mut values = [0.0; VALUES];
    let mut i = 0;
    while i < VALUES {
        let col = column(tid & 31, i);
        let index = row * WIDTH + col;
        let mut value = unsafe { read(input, index) };
        if !bias.is_null() {
            value = f32_from_bf16(bf16_from_f32(float::add_rn_f32(value, unsafe {
                read(bias, col)
            })));
        }
        value = float::mul_rn_ftz_f32(value, 1.0);
        if !residual.is_null() {
            value = f32_from_bf16(bf16_from_f32(float::add_rn_ftz_f32(value, unsafe {
                read(residual, index)
            })));
        }
        values[i] = value;
        i += 1;
    }
    unsafe { crate::norm768::normalize(values, row * WIDTH, gamma, beta, out, eps) };
}
