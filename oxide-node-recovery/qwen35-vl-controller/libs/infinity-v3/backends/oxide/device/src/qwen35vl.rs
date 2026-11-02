// SPDX-License-Identifier: MIT AND Apache-2.0 AND LicenseRef-Infinity-Restricted-MIT-1.0
// Qwen3.5-VL vision arithmetic leaf.
// Adapted from the Candle Qwen3-VL vision.rs and the public Native
// qwen3_vl_models.cpp position-interpolation/rotary recipes. Candle portions are
// MIT (Copyright The Hugging Face Team); Qwen portions are Apache-2.0
// (Copyright Alibaba Cloud). Original contributions are Infinity Restricted MIT
// 1.0. Full inherited texts: device/LICENSE-Candle-BSD3,
// device/LICENSE-PyTorch-BSD3, and device/NOTICE.qwen35vl in this repository.
// The Native recipe rounds the interpolation scale to BF16 BEFORE multiply and
// groups exact corners as (c00+c10)+(c01+c11); this leaf matches that order.

//! Qwen3.5 vision-language tower kernels.
//!
//! Additive source-only exports for Root's SM90/SM100/SM120 build. Existing
//! exports and their ABI are untouched. Arithmetic rounds at the same BF16
//! points as the Native tower: FP32 accumulation, `f32 -> bf16` before an
//! elementwise add or residual store.

use cuda_device::thread;

#[inline(always)]
fn rounded(value: f32) -> f32 {
    super::f32_from_bf16(super::bf16_from_f32(value))
}

#[inline(always)]
fn bf16(value: f32) -> u16 {
    super::bf16_from_f32(value)
}

/// Elementwise BF16 add: out = round(a + b). Grid=ceil(count/256), block=256.
pub unsafe fn add_bf16(a: *const u16, b: *const u16, out: *mut u16, count: u32) {
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if index >= count as usize {
        return;
    }
    unsafe {
        let sum = super::f32_from_bf16(*a.add(index)) + super::f32_from_bf16(*b.add(index));
        *out.add(index) = bf16(rounded(sum));
    }
}

/// Add one bias value per row column to contiguous BF16 rows.
/// Grid=ceil(count/256), block=256.
pub unsafe fn bias_bf16(states: *mut u16, bias: *const u16, count: u32, width: u32) {
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if index >= count as usize || width == 0 {
        return;
    }
    let column = index % width as usize;
    unsafe {
        let value = super::f32_from_bf16(*states.add(index))
            + super::f32_from_bf16(*bias.add(column));
        *states.add(index) = bf16(rounded(value));
    }
}

/// Tanh-approximation GELU in place on BF16 storage.
/// Grid=ceil(count/256), block=256.
pub unsafe fn gelu_tanh_bf16(values: *mut u16, count: u32) {
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if index >= count as usize {
        return;
    }
    unsafe {
        let value = super::f32_from_bf16(*values.add(index));
        let inner = 0.797_884_6f32 * (value + 0.044_715f32 * value * value * value);
        let result = 0.5f32 * value * (1.0 + libm::tanhf(inner));
        *values.add(index) = bf16(rounded(result));
    }
}

/// SiLU in place on BF16 storage. Grid=ceil(count/256), block=256.
pub unsafe fn silu_bf16(values: *mut u16, count: u32) {
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if index >= count as usize {
        return;
    }
    unsafe {
        let value = super::f32_from_bf16(*values.add(index));
        let result = value / (1.0 + libm::expf(-value));
        *values.add(index) = bf16(rounded(result));
    }
}

/// Four-corner learned-position interpolation followed by addition into states.
/// Grid=ceil(tokens*hidden/256), block=256.
pub unsafe fn pos_interp(
    table: *const u16,
    states: *mut u16,
    tokens: u32,
    hidden: u32,
    side: u32,
    height: u32,
    width: u32,
    merge: u32,
) {
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    let total = tokens as usize * hidden as usize;
    if index >= total || merge == 0 || side == 0 || width == 0 {
        return;
    }
    let token = index / hidden as usize;
    let hidden_column = index % hidden as usize;
    let frame_patches = height as usize * width as usize;
    let spatial = token % frame_patches;
    let merge_area = merge as usize * merge as usize;
    let width_blocks = width as usize / merge as usize;
    let height_blocks = height as usize / merge as usize;
    let column_index = spatial % merge as usize
        + ((spatial / merge_area) % width_blocks) * merge as usize;
    let row_index = (spatial / merge as usize) % merge as usize
        + ((spatial / (merge as usize * width as usize)) % height_blocks) * merge as usize;
    let row_scale = if height == 1 {
        0.0
    } else {
        (side - 1) as f32 / (height - 1) as f32
    };
    let column_scale = if width == 1 {
        0.0
    } else {
        (side - 1) as f32 / (width - 1) as f32
    };
    let row = row_index as f32 * row_scale;
    let column = column_index as f32 * column_scale;
    let r0 = row.floor() as usize;
    let c0 = column.floor() as usize;
    let r1 = (row.ceil() as usize).min(side as usize - 1);
    let c1 = (column.ceil() as usize).min(side as usize - 1);
    let dr = row - r0 as f32;
    let dc = column - c0 as f32;
    let corner = |r: usize, c: usize, scale: f32| -> f32 {
        let value = unsafe { *table.add((r * side as usize + c) * hidden as usize + hidden_column) };
        let scale = rounded(scale);
        rounded(super::f32_from_bf16(value) * scale)
    };
    let first = rounded(corner(r0, c0, (1.0 - dr) * (1.0 - dc)) + corner(r1, c0, dr * (1.0 - dc)));
    let second = rounded(corner(r0, c1, (1.0 - dr) * dc) + corner(r1, c1, dr * dc));
    let positional = rounded(first + second);
    unsafe {
        let state = super::f32_from_bf16(*states.add(index));
        *states.add(index) = bf16(rounded(state + positional));
    }
}

/// Split-half 2-D vision rotary over packed Q/K rows.
/// Grid=ceil(tokens*heads*(head_dim/2)/256), block=256.
pub unsafe fn rope2d_bf16(
    q: *mut u16,
    k: *mut u16,
    heads: u32,
    head_dim: u32,
    tokens: u32,
    height: u32,
    width: u32,
    merge: u32,
    q_stride: u64,
    k_stride: u64,
) {
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    let half = head_dim as usize / 2;
    let total = tokens as usize * heads as usize * half;
    if index >= total || merge == 0 || width == 0 || half == 0 || head_dim % 4 != 0 {
        return;
    }
    let token = index / (heads as usize * half);
    let head = (index / half) % heads as usize;
    let phase = index % half;
    let frame_patches = height as usize * width as usize;
    let spatial = token % frame_patches;
    let merge_area = merge as usize * merge as usize;
    let width_blocks = width as usize / merge as usize;
    let height_blocks = height as usize / merge as usize;
    let column_index = spatial % merge as usize
        + ((spatial / merge_area) % width_blocks) * merge as usize;
    let row_index = (spatial / merge as usize) % merge as usize
        + ((spatial / (merge as usize * width as usize)) % height_blocks) * merge as usize;
    let quarter = half / 2;
    let frequency_index = if phase < quarter { phase } else { phase - quarter };
    let frequency = 1.0f32 / 10_000f32.powf(2.0 * frequency_index as f32 / (head_dim as f32 / 2.0));
    let angle = if phase < quarter {
        row_index as f32 * frequency
    } else {
        column_index as f32 * frequency
    };
    let sine = libm::sinf(angle);
    let cosine = libm::cosf(angle);
    let base_q = token as u64 * q_stride + (head * head_dim as usize + phase) as u64;
    let base_q_pair = token as u64 * q_stride + (head * head_dim as usize + phase + half) as u64;
    let base_k = token as u64 * k_stride + (head * head_dim as usize + phase) as u64;
    let base_k_pair = token as u64 * k_stride + (head * head_dim as usize + phase + half) as u64;
    unsafe {
        let aq = super::f32_from_bf16(*q.add(base_q as usize));
        let bq = super::f32_from_bf16(*q.add(base_q_pair as usize));
        *q.add(base_q as usize) = bf16(rounded(aq * cosine - bq * sine));
        *q.add(base_q_pair as usize) = bf16(rounded(bq * cosine + aq * sine));
        let ak = super::f32_from_bf16(*k.add(base_k as usize));
        let bk = super::f32_from_bf16(*k.add(base_k_pair as usize));
        *k.add(base_k as usize) = bf16(rounded(ak * cosine - bk * sine));
        *k.add(base_k_pair as usize) = bf16(rounded(bk * cosine + ak * sine));
    }
}

/// Extract one temporal slice of packed `[C, T, P, P]` patch rows into a
/// contiguous `[C, P, P]` projection row. Grid=ceil(rows*C*P*P/256).
pub unsafe fn temporal_slice_bf16(
    source: *const u16,
    destination: *mut u16,
    rows: u32,
    channels: u32,
    temporal: u32,
    patch: u32,
    slice: u32,
) {
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    let area = patch as usize * patch as usize;
    let compact = channels as usize * area;
    let total = rows as usize * compact;
    if index >= total || slice >= temporal || temporal == 0 || compact == 0 {
        return;
    }
    let row = index / compact;
    let element = index % compact;
    let channel = element / area;
    let within = element % area;
    let source_index = row * compact * temporal as usize
        + (channel * temporal as usize + slice as usize) * area
        + within;
    unsafe {
        *destination.add(index) = *source.add(source_index);
    }
}
