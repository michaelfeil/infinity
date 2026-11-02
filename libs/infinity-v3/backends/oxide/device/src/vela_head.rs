// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! F32 Vela3 encoder/vision decision heads.
//!
//! The host keeps the trained weights in F32 and passes the exact packed
//! request markers. One CTA owns one request so option order, duplicate spans
//! and per-request option counts stay request-local.

use crate::{bf16_from_f32, f32_from_bf16};
use cuda_device::{SharedArray, float, thread};

const MAX_HIDDEN: usize = 1024;
const MAX_FIRST: usize = 2048;
const MAX_PROJECTION: usize = 256;
const LAYER_NORM_EPS: f32 = 1e-5;
const NORMALIZE_EPS: f32 = 1e-12;

static mut NORM: SharedArray<f32, MAX_HIDDEN> = SharedArray::UNINIT;
static mut AUX: SharedArray<f32, MAX_FIRST> = SharedArray::UNINIT;
static mut V: SharedArray<f32, MAX_PROJECTION> = SharedArray::UNINIT;
static mut Q: SharedArray<f32, MAX_PROJECTION> = SharedArray::UNINIT;
static mut U: SharedArray<f32, MAX_PROJECTION> = SharedArray::UNINIT;
static mut UO: SharedArray<f32, MAX_PROJECTION> = SharedArray::UNINIT;
static mut SCRATCH: SharedArray<f32, 8> = SharedArray::UNINIT;

#[inline(always)]
unsafe fn block_sum(value: f32, scratch: *mut f32) -> f32 {
    let tid = thread::threadIdx_x() as usize;
    let lane = tid & 31;
    let value = cuda_device::warp::reduce_sum_f32(value);
    if lane == 0 {
        unsafe { *scratch.add(tid >> 5) = value };
    }
    thread::sync_threads();
    let subtotal = if lane < 8 {
        unsafe { *scratch.add(lane) }
    } else {
        0.0
    };
    let total = cuda_device::warp::reduce_sum_f32(subtotal);
    thread::sync_threads();
    total
}

#[inline(always)]
unsafe fn gelu_exact(value: f32) -> f32 {
    let inv_sqrt_two = 0.707_106_77;
    0.5 * value * (1.0 + libm::erff(value * inv_sqrt_two))
}

#[inline(always)]
unsafe fn relu(value: f32) -> f32 {
    if value.is_nan() || value >= 0.0 {
        value
    } else {
        0.0
    }
}

#[inline(always)]
unsafe fn hidden_at(hidden: *const u16, index: usize) -> f32 {
    f32_from_bf16(unsafe { *hidden.add(index) })
}

#[inline(always)]
unsafe fn layernorm_hidden_row(
    hidden: *const u16,
    row: usize,
    gamma: *const f32,
    beta: *const f32,
    out: *mut f32,
    width: usize,
    scratch: *mut f32,
) {
    let tid = thread::threadIdx_x() as usize;
    let base = row * width;
    let mut sum = 0.0;
    let mut column = tid;
    while column < width {
        sum = float::add_rn_f32(sum, unsafe { hidden_at(hidden, base + column) });
        column += 256;
    }
    let mean = float::div_rn_f32(unsafe { block_sum(sum, scratch) }, width as f32);
    let mut variance = 0.0;
    column = tid;
    while column < width {
        let centered = float::add_rn_f32(unsafe { hidden_at(hidden, base + column) }, -mean);
        variance = float::fma_rn_f32(centered, centered, variance);
        column += 256;
    }
    let variance = float::div_rn_f32(unsafe { block_sum(variance, scratch) }, width as f32);
    let inverse = float::div_rn_f32(
        1.0,
        float::sqrt_rn_f32(float::add_rn_f32(variance, LAYER_NORM_EPS)),
    );
    column = tid;
    while column < width {
        let centered = float::add_rn_f32(unsafe { hidden_at(hidden, base + column) }, -mean);
        let scaled = float::mul_rn_f32(centered, inverse);
        let affine = float::fma_rn_f32(scaled, unsafe { *gamma.add(column) }, unsafe {
            *beta.add(column)
        });
        unsafe { *out.add(column) = affine };
        column += 256;
    }
    thread::sync_threads();
}

#[inline(always)]
unsafe fn project_hidden_row(
    hidden: *const u16,
    row: usize,
    weight: *const f32,
    bias: *const f32,
    out: *mut f32,
    outputs: usize,
    width: usize,
) {
    let tid = thread::threadIdx_x() as usize;
    let base = row * width;
    let mut output = tid;
    while output < outputs {
        let mut value = 0.0;
        let mut column = 0;
        while column < width {
            value = float::fma_rn_f32(
                unsafe { hidden_at(hidden, base + column) },
                unsafe { *weight.add(output * width + column) },
                value,
            );
            column += 1;
        }
        unsafe { *out.add(output) = float::add_rn_f32(value, *bias.add(output)) };
        output += 256;
    }
    thread::sync_threads();
}

#[inline(always)]
unsafe fn layernorm_row(
    input: *const f32,
    gamma: *const f32,
    beta: *const f32,
    out: *mut f32,
    width: usize,
    scratch: *mut f32,
) {
    let tid = thread::threadIdx_x() as usize;
    let mut sum = 0.0;
    let mut column = tid;
    while column < width {
        sum = float::add_rn_f32(sum, unsafe { *input.add(column) });
        column += 256;
    }
    let mean = float::div_rn_f32(unsafe { block_sum(sum, scratch) }, width as f32);
    let mut variance = 0.0;
    column = tid;
    while column < width {
        let centered = float::add_rn_f32(unsafe { *input.add(column) }, -mean);
        variance = float::fma_rn_f32(centered, centered, variance);
        column += 256;
    }
    let variance = float::div_rn_f32(unsafe { block_sum(variance, scratch) }, width as f32);
    let inverse = float::div_rn_f32(
        1.0,
        float::sqrt_rn_f32(float::add_rn_f32(variance, LAYER_NORM_EPS)),
    );
    column = tid;
    while column < width {
        let centered = float::add_rn_f32(unsafe { *input.add(column) }, -mean);
        let scaled = float::mul_rn_f32(centered, inverse);
        let affine = float::fma_rn_f32(scaled, unsafe { *gamma.add(column) }, unsafe {
            *beta.add(column)
        });
        unsafe { *out.add(column) = affine };
        column += 256;
    }
    thread::sync_threads();
}

#[inline(always)]
unsafe fn project(
    input: *const f32,
    weight: *const f32,
    bias: *const f32,
    out: *mut f32,
    rows: usize,
    width: usize,
) {
    let tid = thread::threadIdx_x() as usize;
    let mut row = tid;
    while row < rows {
        let mut value = 0.0;
        let mut column = 0;
        while column < width {
            value = float::fma_rn_f32(
                unsafe { *input.add(column) },
                unsafe { *weight.add(row * width + column) },
                value,
            );
            column += 1;
        }
        unsafe { *out.add(row) = float::add_rn_f32(value, *bias.add(row)) };
        row += 256;
    }
    thread::sync_threads();
}

#[inline(always)]
unsafe fn normalize(vector: *mut f32, width: usize, scratch: *mut f32) {
    let tid = thread::threadIdx_x() as usize;
    let mut sum = 0.0;
    let mut column = tid;
    while column < width {
        let value = unsafe { *vector.add(column) };
        sum = float::fma_rn_f32(value, value, sum);
        column += 256;
    }
    let norm = float::sqrt_rn_f32(unsafe { block_sum(sum, scratch) });
    let divisor = if norm < NORMALIZE_EPS {
        NORMALIZE_EPS
    } else {
        norm
    };
    column = tid;
    while column < width {
        unsafe { *vector.add(column) = float::div_rn_f32(*vector.add(column), divisor) };
        column += 256;
    }
    thread::sync_threads();
}

#[inline(always)]
unsafe fn dot(left: *const f32, right: *const f32, width: usize, scratch: *mut f32) -> f32 {
    let tid = thread::threadIdx_x() as usize;
    let mut sum = 0.0;
    let mut column = tid;
    while column < width {
        sum = float::fma_rn_f32(
            unsafe { *left.add(column) },
            unsafe { *right.add(column) },
            sum,
        );
        column += 256;
    }
    unsafe { block_sum(sum, scratch) }
}

#[inline(always)]
unsafe fn mean_span(
    hidden: *const u16,
    start: usize,
    end: usize,
    width: usize,
    out: *mut f32,
    round_bf16: bool,
) {
    let tid = thread::threadIdx_x() as usize;
    let count = end.saturating_sub(start);
    if count == 0 {
        return;
    }
    let mut column = tid;
    while column < width {
        let mut sum = 0.0;
        let mut row = start;
        while row < end {
            sum = float::add_rn_f32(sum, unsafe { hidden_at(hidden, row * width + column) });
            row += 1;
        }
        let mean = float::div_rn_f32(sum, count as f32);
        let value = if round_bf16 {
            f32_from_bf16(bf16_from_f32(mean))
        } else {
            mean
        };
        unsafe { *out.add(column) = value };
        column += 256;
    }
    thread::sync_threads();
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn forward(
    hidden: *const u16,
    boundaries: *const u32,
    markers: *const u32,
    marker_offsets: *const u32,
    weights: *const f32,
    out: *mut f32,
    counts: *mut u32,
    hidden_width: u32,
    projection: u32,
    max_options: u32,
    mode: u32,
) {
    let request = thread::blockIdx_x() as usize;
    let tid = thread::threadIdx_x() as usize;
    let width = hidden_width as usize;
    let proj = projection as usize;
    let start = unsafe { *boundaries.add(request) } as usize;
    let end = unsafe { *boundaries.add(request + 1) } as usize;
    let marker_start = unsafe { *marker_offsets.add(request) } as usize;
    let marker_end = unsafe { *marker_offsets.add(request + 1) } as usize;
    if end <= start || marker_end <= marker_start || width == 0 || width > MAX_HIDDEN {
        return;
    }
    let length = end - start;
    let norm = unsafe { SharedArray::as_raw_mut_ptr(&raw mut NORM) };
    let aux = unsafe { SharedArray::as_raw_mut_ptr(&raw mut AUX) };
    let v = unsafe { SharedArray::as_raw_mut_ptr(&raw mut V) };
    let q = unsafe { SharedArray::as_raw_mut_ptr(&raw mut Q) };
    let u = unsafe { SharedArray::as_raw_mut_ptr(&raw mut U) };
    let uo = unsafe { SharedArray::as_raw_mut_ptr(&raw mut UO) };
    let scratch = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCRATCH) };
    if mode == 0 {
        if proj == 0 || proj > MAX_PROJECTION || marker_end - marker_start < 4 {
            return;
        }
        let marker = |index: usize| unsafe { *markers.add(marker_start + index) as usize };
        let question = marker(0);
        let state_start = marker(1);
        let state_end = marker(2);
        if question >= length || state_start > state_end || state_end > length {
            return;
        }
        let options = marker_end - marker_start - 3;
        let norm_weight = weights;
        let norm_bias = weights.add(width);
        let option_weight = weights.add(2 * width);
        let option_bias = option_weight.add(proj * width);
        let question_weight = option_bias.add(proj);
        let question_bias = question_weight.add(proj * width);
        let state_weight = question_bias.add(proj);
        let state_bias = state_weight.add(proj * width);
        let first_weight = state_bias.add(proj);
        let first_bias = first_weight.add(2 * width * width);
        let last_weight = first_bias.add(2 * width);
        let last_bias = last_weight.add(2 * width);
        let log_tau = unsafe { *last_bias.add(1) };
        let tau = libm::expf(log_tau);

        unsafe {
            layernorm_hidden_row(
                hidden,
                start + question,
                norm_weight,
                norm_bias,
                norm,
                width,
                scratch,
            );
            project(norm, question_weight, question_bias, q, proj, width);
        }
        unsafe {
            mean_span(
                hidden,
                start + state_start,
                start + state_end,
                width,
                aux,
                false,
            );
            layernorm_row(aux, norm_weight, norm_bias, norm, width, scratch);
            project(norm, state_weight, state_bias, v, proj, width);
            normalize(v, proj, scratch);
        }
        let mut option = 0usize;
        while option < options {
            let marker_row = marker(3 + option);
            if marker_row >= length {
                option += 1;
                continue;
            }
            let row = start + marker_row;
            unsafe {
                layernorm_hidden_row(hidden, row, norm_weight, norm_bias, norm, width, scratch);
                project(norm, option_weight, option_bias, uo, proj, width);
                let mut column = tid;
                while column < proj {
                    *u.add(column) = float::add_rn_f32(*uo.add(column), *q.add(column));
                    column += 256;
                }
                thread::sync_threads();
                normalize(u, proj, scratch);
                let cosine = dot(u, v, proj, scratch);
                project_hidden_row(hidden, row, first_weight, first_bias, aux, 2 * width, width);
                let mut column = tid;
                while column < 2 * width {
                    *aux.add(column) = relu(*aux.add(column));
                    column += 256;
                }
                thread::sync_threads();
                let residual = dot(aux, last_weight, 2 * width, scratch);
                let value = float::add_rn_f32(
                    float::div_rn_f32(cosine, tau),
                    float::add_rn_f32(residual, *last_bias),
                );
                if tid == 0 {
                    *out.add(request * max_options as usize + option) = value;
                }
            }
            option += 1;
        }
        if tid == 0 {
            unsafe { *counts.add(request) = options as u32 };
        }
    } else {
        let first_weight = weights;
        let first_bias = first_weight.add(width * width);
        let norm_weight = first_bias.add(width);
        let norm_bias = norm_weight.add(width);
        let last_weight = norm_bias.add(width);
        let last_bias = last_weight.add(width);
        let options = (marker_end - marker_start) / 2;
        let mut option = 0usize;
        while option < options {
            let span_start = unsafe { *markers.add(marker_start + option * 2) as usize };
            let span_end = unsafe { *markers.add(marker_start + option * 2 + 1) as usize };
            if span_start >= span_end || span_end > length {
                option += 1;
                continue;
            }
            unsafe {
                mean_span(
                    hidden,
                    start + span_start,
                    start + span_end,
                    width,
                    aux,
                    true,
                );
                project(aux, first_weight, first_bias, norm, width, width);
                let mut column = tid;
                while column < width {
                    *norm.add(column) = gelu_exact(*norm.add(column));
                    column += 256;
                }
                thread::sync_threads();
                layernorm_row(norm, norm_weight, norm_bias, aux, width, scratch);
                let value = float::add_rn_f32(dot(aux, last_weight, width, scratch), *last_bias);
                if tid == 0 {
                    *out.add(request * max_options as usize + option) = value;
                }
            }
            option += 1;
        }
        if tid == 0 {
            unsafe { *counts.add(request) = options as u32 };
        }
    }
    thread::sync_threads();
}

/// Trained UltraVBERT pixel shuffle: [tiles,32,32,768] -> [tiles*64,12288]
/// for one tile. The host validates the fixed geometry.
#[inline(always)]
pub(crate) unsafe fn pixel_shuffle(input: *const u16, out: *mut u16) {
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if index >= 64 * 12288 {
        return;
    }
    let row = index / 12288;
    let within = index % 12288;
    let group = within / 3072;
    let element = within % 3072;
    let outer = row / 8;
    let column = row % 8;
    let source = ((outer * 4 + group) * 8 + column) * 3072 + element;
    unsafe { *out.add(index) = *input.add(source) };
}
