// SPDX-License-Identifier: MIT AND LicenseRef-Infinity-Restricted-MIT-1.0
// Upstream Candle kernel portions retain MIT; new original port contributions
// use the Infinity Restricted MIT License 1.0. No inherited grant is changed.
//
// Adapted from the published tracked sources
//   libs/infinity-v3/backends/libtorch/cpp/models/qwen35_models.cpp
//     (`rotate` and `full_attention`, including the dense post-attention gate)
//   libs/infinity-v3/backends/libtorch/cpp/kernels/qwen35_kernels.cu
//     (`gemma_rotary_cuda` / `staged_rotary` and `gemma3_qk_norm_rotary`)
//   libs/infinity-v3/backends/oxide/device/src/gemma3.rs
//     (`qk_norm_rope_256_bf16`, the qualified BF16 mul/HFMA orientation)
//   libs/infinity-v3/backends/oxide/device/src/gemma3_table_initializer.rs
//     (cold BF16 rotary tables)
// Original contributions Copyright Infinity Core Contributors,
// Licensed under the Infinity Restricted MIT License 1.0; see the repository
// LICENSE, NOTICE and THIRD_PARTY_NOTICES.md. The Infinity Restricted MIT
// License 1.0 is not the standard MIT License.
//
// Candle MIT notice retained:
// Copyright The Hugging Face Team. All rights reserved.
// Permission is hereby granted, free of charge, to any
// person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the
// Software without restriction, including without
// limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software
// is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice
// shall be included in all copies or substantial portions
// of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
// ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
// TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
// PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
// SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
// IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//! Exact Qwen3.5 BF16 attention-side device helpers.
//!
//! The three exports are source-only device entry points. The host owns the
//! positive geometry, position/table bounds, pointer validity, and the launch
//! shapes documented in `HANDOFF.md`.
//!
//! * `partial_rope_256_bf16` rotates only the leading `rotary` channels of an
//!   already normalized 256-wide Q/K head in place. The first `rotary / 2`
//!   channels pair with the next `rotary / 2` channels; the tail stays
//!   untouched. It preserves native `staged_rotary`: four separately BF16-
//!   rounded products, followed by a separately rounded FP32 add/sub and BF16 cast.
//! * `attention_gate_bf16` applies the dense `at::sigmoid` gate to the packed
//!   `[query | gate]` projection and multiplies it by attention with a native
//!   BF16 product. The sigmoid keeps the ATen FP32 opmath order.
//! * `rope_tables_bf16` fills the cold rotary tables from host-computed
//!   inverse frequencies with libdevice cosine/sine rounded back to BF16.

use cuda_device::{float, ptx_asm, thread};

use crate::{bf16_from_f32, f32_from_bf16};

const HEAD_WIDTH: usize = 256;
const WARP_SIZE: usize = 32;
const HEADS_PER_CTA: usize = 4;
const TABLE_THREADS: usize = 256;

/// Exact native `__fsub_rn`; the pinned SDK has no named subtraction wrapper.
#[inline(always)]
fn qwen35_attention_sub_rn_f32(lhs: f32, rhs: f32) -> f32 {
    let result: f32;
    unsafe {
        ptx_asm!("sub.rn.f32 %0, %1, %2;", out("=f") result, in("f") lhs, in("f") rhs);
    }
    result
}

/// Native BF16 multiply, `mul.bf16`.
#[inline(always)]
fn qwen35_attention_bf16_mul(lhs: u16, rhs: u16) -> u16 {
    let result: u16;
    unsafe {
        ptx_asm!(
            "mul.bf16 %0, %1, %2;",
            out("=h") result,
            in("h") lhs,
            in("h") rhs,
        );
    }
    result
}

/// Round an FP32 value through native BF16 conversion and back.
#[inline(always)]
fn qwen35_attention_round_bf16(value: f32) -> f32 {
    f32_from_bf16(bf16_from_f32(value))
}

/// In-place partial rotary over `[token, head]` rows of width 256.
///
/// `q` and `k` are already the exact native offset-norm BF16 values. Rows
/// `[0, tokens * qheads)` address Q and the remaining
/// `[tokens * qheads, tokens * (qheads + kvheads))` rows address K. A row base
/// is `token * stride + head * 256` for the selected tensor. `positions` and
/// the coefficient tables are host-validated; `table_stride` is the number of
/// F32 coefficients per position row (32 for the initial rotary of 64).
///
/// Native source: `rotate` in `qwen35_models.cpp` calls `gemma_rotary_cuda`,
/// whose `staged_rotary` explicitly rounds all four products to BF16 before
/// RN FP32 add/sub and the final BF16 casts. This is distinct from Gemma3's
/// fused QK-norm rotary kernel. Coefficients are already BF16-rounded F32.
///
/// Launch: grid `ceil(tokens * (qheads + kvheads) / 4)`, block 128. Warp
/// `threadIdx.x / 32` owns one head; each of its 32 lanes owns one pair column
/// and loops in steps of 32 when `rotary / 2` exceeds one warp. Invalid warps
/// in a partial final CTA return immediately. There are no barriers.
#[inline(always)]
pub(crate) unsafe fn partial_rope_256_bf16(
    q: *mut u16,
    k: *mut u16,
    cosine: *const f32,
    sine: *const f32,
    positions: *const u32,
    tokens: u32,
    qheads: u32,
    kvheads: u32,
    rotary: u32,
    qstride: u64,
    kstride: u64,
    table_stride: u32,
) {
    if rotary == 0 || rotary > HEAD_WIDTH as u32 || (rotary & 1) != 0 || table_stride < rotary / 2 {
        return;
    }
    let qrows = (tokens as u64) * (qheads as u64);
    let total = qrows + (tokens as u64) * (kvheads as u64);
    if total == 0 {
        return;
    }
    let tid = thread::threadIdx_x() as usize;
    let lane = tid & (WARP_SIZE - 1);
    let warp = tid / WARP_SIZE;
    if warp >= HEADS_PER_CTA {
        return;
    }
    let row = (thread::blockIdx_x() as u64) * (HEADS_PER_CTA as u64) + (warp as u64);
    if row >= total {
        return;
    }
    let query = row < qrows;
    let local = if query { row } else { row - qrows };
    let heads = (if query { qheads } else { kvheads }) as u64;
    if heads == 0 {
        return;
    }
    let token = local / heads;
    let head = local % heads;
    let (data, stride) = if query { (q, qstride) } else { (k, kstride) };
    let base = (token as usize) * (stride as usize) + (head as usize) * HEAD_WIDTH;
    let table_row = unsafe { *positions.add(token as usize) } as usize;
    let table_base = table_row * (table_stride as usize);
    let half = rotary as usize / 2;

    let mut column = lane;
    while column < half {
        let a = unsafe { *data.add(base + column) };
        let b = unsafe { *data.add(base + column + half) };
        let c = bf16_from_f32(unsafe { *cosine.add(table_base + column) });
        let s = bf16_from_f32(unsafe { *sine.add(table_base + column) });
        // Native Qwen3.5 uses staged_rotary, not Gemma3's fused QK kernel:
        // each of all four products rounds to BF16 before the separate add/sub.
        let a = f32_from_bf16(a);
        let b = f32_from_bf16(b);
        let c = f32_from_bf16(c);
        let s = f32_from_bf16(s);
        let ac = qwen35_attention_round_bf16(float::mul_rn_f32(a, c));
        let bs = qwen35_attention_round_bf16(float::mul_rn_f32(b, s));
        let bc = qwen35_attention_round_bf16(float::mul_rn_f32(b, c));
        let asine = qwen35_attention_round_bf16(float::mul_rn_f32(a, s));
        unsafe {
            *data.add(base + column) = bf16_from_f32(qwen35_attention_sub_rn_f32(ac, bs));
            *data.add(base + column + half) = bf16_from_f32(float::add_rn_f32(bc, asine));
        }
        column += WARP_SIZE;
    }
}

/// Dense post-attention sigmoid gate over contiguous `[tokens, heads, 256]`.
///
/// `query_gate` is the address of the first gate half, i.e. element 256 of the
/// first packed `[query(256) | gate(256)]` head. Each following head's gate
/// starts 512 BF16 elements later, and each following token's gate starts
/// `gate_stride` BF16 elements later, so a merged `[querygate, K, V]` token row
/// is supported. The helper does not add the initial +256; the caller must pass
/// the gate-half address. `attention` is the contiguous attention output.
///
/// Native source: the dense branch of `full_attention` in
/// `qwen35_models.cpp`, where the gate is `at::sigmoid(logits)` (not the MoE
/// reciprocal chain) and the product is a BF16 elementwise multiply. For BF16
/// ATen uses float opmath: widen the BF16 gate, evaluate `libm::expf(-gate)`
/// (libdevice `__nv_expf`, not a fast approximation), add one, divide one by
/// the sum, then round that FP32 sigmoid result to BF16. The final multiply is
/// native `mul.bf16`, not an FP32 product rounded once.
///
/// Launch: grid `ceil(tokens * heads * 256 / 256)`, block 256, one element per
/// lane. The flat index maps directly to the contiguous attention/output
/// element. `out` may alias `attention`; each lane reads both operands before
/// its own disjoint write. All accesses are scalar 16-bit BF16 accesses.
#[inline(always)]
pub(crate) unsafe fn attention_gate_bf16(
    attention: *const u16,
    query_gate: *const u16,
    out: *mut u16,
    tokens: u32,
    heads: u32,
    gate_stride: u64,
) {
    let total = (tokens as u64) * (heads as u64) * (HEAD_WIDTH as u64);
    let index = (thread::blockIdx_x() as u64) * (thread::blockDim_x() as u64)
        + (thread::threadIdx_x() as u64);
    if index >= total {
        return;
    }
    let row = index / (HEAD_WIDTH as u64);
    let column = index % (HEAD_WIDTH as u64);
    let token = row / (heads as u64);
    let head = row % (heads as u64);
    let gate_index = token * gate_stride + head * ((HEAD_WIDTH as u64) * 2) + column;

    let gate = f32_from_bf16(unsafe { *query_gate.add(gate_index as usize) });
    let exponential = libm::expf(-gate);
    let denominator = float::add_rn_f32(1.0, exponential);
    let sigmoid = bf16_from_f32(float::div_rn_f32(1.0, denominator));
    let attention_value = unsafe { *attention.add(index as usize) };
    unsafe {
        *out.add(index as usize) = qwen35_attention_bf16_mul(attention_value, sigmoid);
    }
}

/// Cold BF16 rotary table initializer.
///
/// `inverse` is host-provided FP32 data, 32 entries initially. Native host
/// floats follow `1 / pow(theta, 2 * i / rotary)`; this helper never computes
/// or approximates frequencies on the device. For every
/// `(position, pair)` the phase is the rounded FP32 product of the position and
/// inverse frequency, followed by libdevice cosine/sine and a BF16 round trip
/// back to F32.
///
/// Launch: grid `ceil(positions * pairs / 256)`, block 256. `cosine` and
/// `sine` each receive `positions * pairs` contiguous F32 values, row-major by
/// position then pair.
#[inline(always)]
pub(crate) unsafe fn rope_tables_bf16(
    inverse: *const f32,
    cosine: *mut f32,
    sine: *mut f32,
    positions: u32,
    pairs: u32,
) {
    if positions == 0 || pairs == 0 {
        return;
    }
    let pairs = pairs as u64;
    let total = (positions as u64) * pairs;
    let index =
        (thread::blockIdx_x() as u64) * (TABLE_THREADS as u64) + (thread::threadIdx_x() as u64);
    if index >= total {
        return;
    }
    let phase = float::mul_rn_f32((index / pairs) as f32, unsafe {
        *inverse.add((index % pairs) as usize)
    });
    unsafe {
        *cosine.add(index as usize) = qwen35_attention_round_bf16(libm::cosf(phase));
        *sine.add(index as usize) = qwen35_attention_round_bf16(libm::sinf(phase));
    }
}
