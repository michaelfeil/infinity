// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Source-only Pixtral/Mistral3 media kernels.
//!
//! These operators are additive: they do not alter the existing 86 exported
//! kernel ABI. Root compiles them with the rest of the oxide device crate for
//! sm_90/sm_100/sm_120 before a GPU route may be admitted.

use cuda_device::{float, thread};
use crate::{bf16_from_f32, f32_from_bf16};

#[inline(always)]
fn read_bf16(pointer: *const u16, index: usize) -> f32 {
    f32_from_bf16(unsafe { *pointer.add(index) })
}

#[inline(always)]
fn write_bf16(pointer: *mut u16, index: usize, value: f32) {
    unsafe { *pointer.add(index) = bf16_from_f32(value) };
}

/// Apply Pixtral 2-D split-half rotary to packed per-image Q/K rows.
///
/// Layout: separate contiguous `[tokens, heads, head_dim]` BF16 buffers. The
/// first half of each head rotates by a row frequency, the second half by a
/// column frequency, exactly matching the native `cat({rows, cols}, -1)`
/// duplicate of the reference tower. One thread owns one rotary pair.
#[inline(always)]
pub(crate) unsafe fn rope2d_bf16(
    q: *mut u16,
    k: *mut u16,
    grids: *const u32,
    offsets: *const u32,
    image_count: u32,
    tokens: u32,
    heads: u32,
    head_dim: u32,
    theta: f32,
) {
    if heads == 0 || head_dim == 0 || !head_dim.is_multiple_of(4) {
        return;
    }
    let pairs = (head_dim / 2) as usize;
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    let total = tokens as usize * heads as usize * pairs;
    if index >= total {
        return;
    }
    let pair = index % pairs;
    let head = (index / pairs) % heads as usize;
    let token = index / (pairs * heads as usize);
    // Image boundaries are short (normally one frame); a linear scan avoids
    // per-thread binary-search divergence and extra workspace.
    let mut image = 0usize;
    while image + 1 < image_count as usize && token >= unsafe { *offsets.add(image + 1) } as usize {
        image += 1;
    }
    let start = unsafe { *offsets.add(image) } as usize;
    let width = unsafe { *grids.add(image * 3 + 2) } as usize;
    if width == 0 {
        return;
    }
    let local = token - start;
    let row = local / width;
    let column = local % width;
    let half = (head_dim / 2) as usize;
    let phase = if pair < half / 2 {
        row as f32 * libm::powf(theta, -(4.0 * pair as f32) / head_dim as f32)
    } else {
        let frequency = pair - half / 2;
        column as f32 * libm::powf(theta, -(4.0 * frequency as f32 + 2.0) / head_dim as f32)
    };
    // The native tower rounds the coefficient tables to the model dtype
    // before the rotary multiply, not only the final product.
    let cosine = f32_from_bf16(bf16_from_f32(libm::cosf(phase)));
    let sine = f32_from_bf16(bf16_from_f32(libm::sinf(phase)));
    let base = (token * heads as usize + head) * head_dim as usize;
    // Native rotary composes BF16 tensors: each product and the final sum are
    // rounded at their own boundary.
    let round = |value: f32| f32_from_bf16(bf16_from_f32(value));
    for buffer in [q, k] {
        let first = read_bf16(buffer, base + pair);
        let second = read_bf16(buffer, base + half + pair);
        let first_cos = round(first * cosine);
        let second_sin = round(second * sine);
        let second_cos = round(second * cosine);
        let first_sin = round(first * sine);
        write_bf16(buffer, base + pair, round(first_cos - second_sin));
        write_bf16(buffer, base + half + pair, round(second_cos + first_sin));
    }
}

/// Add two BF16 matrices elementwise into a distinct output.
#[inline(always)]
pub(crate) unsafe fn add_bf16(
    left: *const u16,
    right: *const u16,
    out: *mut u16,
    elements: u32,
) {
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if index >= elements as usize {
        return;
    }
    write_bf16(out, index, read_bf16(left, index) + read_bf16(right, index));
}

/// Scatter projected image rows into gathered token embeddings.
/// One thread copies one feature column; row indices are device int32.
#[inline(always)]
pub(crate) unsafe fn scatter_features_bf16(
    embeddings: *mut u16,
    features: *const u16,
    indices: *const u32,
    rows: u32,
    width: u32,
) {
    if width == 0 {
        return;
    }
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if index >= rows as usize * width as usize {
        return;
    }
    let source_row = index / width as usize;
    let column = index % width as usize;
    let destination_row = unsafe { *indices.add(source_row) } as usize;
    unsafe {
        *embeddings.add(destination_row * width as usize + column) =
            *features.add(source_row * width as usize + column);
    }
}

/// Convert F32 patch pixels to the BF16 tower input without a host round-trip.
#[inline(always)]
pub(crate) unsafe fn pixels_to_bf16(input: *const f32, out: *mut u16, elements: u32) {
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if index >= elements as usize {
        return;
    }
    write_bf16(out, index, unsafe { *input.add(index) });
}

/// Spatial merge gather matching `F.unfold` channel-major order.
///
/// Input is `[tokens, hidden]` in spatial patch-row order. Output is
/// `[token_rows, hidden * merge * merge]` with `(hidden, merge_row,
/// merge_column)` as the flattened inner order. One thread owns one output
/// element and reads its validated source row from `source_rows` (the media
/// foundation computes those exact indices; this kernel never guesses them).
#[inline(always)]
pub(crate) unsafe fn merge_gather_bf16(
    input: *const u16,
    source_rows: *const u32,
    out: *mut u16,
    token_rows: u32,
    hidden: u32,
    merge: u32,
) {
    if hidden == 0 || merge == 0 {
        return;
    }
    let group = (merge * merge) as usize;
    let width = hidden as usize * group;
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if index >= token_rows as usize * width {
        return;
    }
    let output_row = index / width;
    let inner = index % width;
    let channel = inner / group;
    let position = inner % group;
    // `source_rows` already contains absolute patch rows from the media
    // foundation, so no image-boundary inference is needed here.
    let source = unsafe { *source_rows.add(output_row * group + position) } as usize;
    unsafe {
        *out.add(output_row * width + inner) = *input.add(source * hidden as usize + channel);
    }
}

/// BF16 SiLU gate multiplied by a separate up matrix.
#[inline(always)]
pub(crate) unsafe fn silu_mul_bf16(
    gate: *const u16,
    up: *const u16,
    out: *mut u16,
    elements: u32,
) {
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if index >= elements as usize {
        return;
    }
    let value = read_bf16(gate, index);
    let activated = value / (1.0 + libm::expf(-value));
    // Native activation code 0 rounds the activation to model dtype before
    // multiplying the up matrix, then rounds the product.
    let activated = f32_from_bf16(bf16_from_f32(activated));
    write_bf16(out, index, activated * read_bf16(up, index));
}

/// BF16 GELU gate multiplied by a separate up matrix. `mode` follows the
/// Native activation code: 0 exact, 1 PyTorch tanh, 2 SiLU.
#[inline(always)]
pub(crate) unsafe fn gelu_mul_bf16(
    gate: *const u16,
    up: *const u16,
    out: *mut u16,
    elements: u32,
    mode: u32,
) {
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if index >= elements as usize {
        return;
    }
    let value = read_bf16(gate, index);
    let activated = match mode {
        2 => value / (1.0 + libm::expf(-value)),
        1 => {
            0.5 * value
                * (1.0
                    + libm::tanhf(0.7978845608028654 * (value + 0.044715 * value * value * value)))
        }
        _ => 0.5 * value * (1.0 + libm::erff(value * 0.7071067811865476)),
    };
    let activated = f32_from_bf16(bf16_from_f32(activated));
    write_bf16(out, index, activated * read_bf16(up, index));
}

/// BF16 GELU (`tanh` selects the PyTorch approximation).
#[inline(always)]
pub(crate) unsafe fn gelu_bf16(
    input: *const u16,
    out: *mut u16,
    elements: u32,
    mode: u32,
) {
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if index >= elements as usize {
        return;
    }
    let value = read_bf16(input, index);
    let activated = match mode {
        1 => 0.5
            * value
            * (1.0
                + libm::tanhf(0.7978845608028654 * (value + 0.044715 * value * value * value))),
        2 => value / (1.0 + libm::expf(-value)),
        _ => 0.5 * value * (1.0 + libm::erff(value * 0.7071067811865476)),
    };
    write_bf16(out, index, activated);
}

/// Apply the Native `llama_4_scaling` per-position query multiplier. `scales`
/// is a BF16 table indexed by the token position; `q` is packed QKV storage.
#[inline(always)]
pub(crate) unsafe fn llama4_query_scale_bf16(
    q: *mut u16,
    scales: *const u16,
    positions: *const u32,
    tokens: u32,
    heads: u32,
    head_dim: u32,
    q_stride: u64,
) {
    if heads == 0 || head_dim == 0 {
        return;
    }
    let width = heads as usize * head_dim as usize;
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if index >= tokens as usize * width {
        return;
    }
    let token = index / width;
    let inner = index % width;
    let position = unsafe { *positions.add(token) } as usize;
    let scale = read_bf16(scales, position);
    let address = (token as u64 * q_stride) as usize + inner;
    let value = read_bf16(q, address);
    write_bf16(q, address, value * scale);
}

/// Native shaped scaled table initializer: phase = pos * inverse, libdevice
/// cos/sin multiplied by the YaRN attention scale, then one BF16 round-trip.
#[inline(always)]
pub(crate) unsafe fn rope_tables_scaled_bf16(
    inverse: *const f32,
    cosine: *mut f32,
    sine: *mut f32,
    positions: u32,
    pairs: u32,
    scale: f32,
) {
    if positions == 0 || pairs == 0 {
        return;
    }
    let pairs = pairs as u64;
    let total = positions as u64 * pairs;
    let index = thread::blockIdx_x() as u64 * 256 + thread::threadIdx_x() as u64;
    if index >= total {
        return;
    }
    let position = (index / pairs) as f32;
    let inverse = unsafe { *inverse.add((index % pairs) as usize) };
    let phase = float::mul_rn_f32(position, inverse);
    unsafe {
        *cosine.add(index as usize) =
            f32_from_bf16(bf16_from_f32(float::mul_rn_f32(libm::cosf(phase), scale)));
        *sine.add(index as usize) =
            f32_from_bf16(bf16_from_f32(float::mul_rn_f32(libm::sinf(phase), scale)));
    }
}
