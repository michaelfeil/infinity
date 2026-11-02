// SPDX-License-Identifier: MIT AND LicenseRef-Infinity-Restricted-MIT-1.0
// Upstream Candle MIT portions retain their MIT grant. Original port
// contributions use the Infinity Restricted MIT License 1.0.
//
// Exact arithmetic sources:
//   libtorch/cpp/models/gemma_models.cpp (vision()/audio() composition)
//   libtorch/cpp/kernels/qwen35_kernels.cu (Gemma4 norm/pool/rotary kernels)
//   embeddinggemma2 Candle gemma4_vision/audio references
//
// These kernels are the source-only media leaf. They are intentionally scalar
// and deterministic: one thread owns one output row/query, and every reduction
// walks the source in native row-major order. The host validates all extents
// before launch; no kernel allocates, synchronizes, or queries the driver.
//! EmbeddingGemma2 vision/audio device primitives.

use cuda_device::{float, thread};
use libm;

use crate::{bf16_from_f32, f32_from_bf16, read, write};

#[inline(always)]
fn gemma2_media_sub_rn_f32(a: f32, b: f32) -> f32 {
    float::add_rn_f32(a, -b)
}


const THREADS: usize = 256;

#[inline(always)]
fn index() -> usize {
    thread::blockIdx_x() as usize * THREADS + thread::threadIdx_x() as usize
}

#[inline(always)]
fn rounded(value: f32) -> f32 {
    f32_from_bf16(bf16_from_f32(value))
}

#[inline(always)]
fn store_bf16(out: *mut u16, index: usize, value: f32) {
    unsafe { write(out, index, value) };
}

/// NCHW F32 im2col with the Native `gemma_audio_conv2d` geometry.
/// Input is `[channels, height, width]` (batch one), output rows are
/// `(out_h, out_w)` and columns are `(channel, kh, kw)` row-major.
#[inline(always)]
pub(crate) unsafe fn im2col_f32(
    input: *const f32,
    out: *mut f32,
    channels: u32,
    height: u32,
    width: u32,
    kh: u32,
    kw: u32,
    stride: u32,
    pad_h: u32,
    pad_w: u32,
    out_h: u32,
    out_w: u32,
) {
    let rows = out_h as usize * out_w as usize;
    let columns = channels as usize * kh as usize * kw as usize;
    let i = index();
    if i >= rows * columns {
        return;
    }
    let row = i / columns;
    let column = i % columns;
    let hon = row / out_w as usize;
    let won = row % out_w as usize;
    let c = column / (kh as usize * kw as usize);
    let kernel = column % (kh as usize * kw as usize);
    let ky = kernel / kw as usize;
    let kx = kernel % kw as usize;
    let y = hon * stride as usize + ky;
    let x = won * stride as usize + kx;
    let y = y.wrapping_sub(pad_h as usize);
    let x = x.wrapping_sub(pad_w as usize);
    let value = if y < height as usize && x < width as usize {
        unsafe { *input.add((c * height as usize + y) * width as usize + x) }
    } else {
        0.0
    };
    unsafe { *out.add(i) = value };
}

/// Apply the native validity mask (`states * valid`) to F32 feature rows.
#[inline(always)]
pub(crate) unsafe fn mask_f32(input: *const f32, valid: *const u8, out: *mut f32, count: u32) {
    let i = index();
    if i >= count as usize {
        return;
    }
    let mask = unsafe { *valid.add(i) };
    let value = if mask != 0 {
        unsafe { *input.add(i) }
    } else {
        0.0
    };
    unsafe { *out.add(i) = value };
}

/// Native per-row channel normalization followed by the BF16 cast and ReLU.
#[inline(always)]
pub(crate) unsafe fn channel_norm_relu_bf16(
    input: *const f32,
    scale: *const u16,
    out: *mut u16,
    rows: u32,
    width: u32,
    eps: f32,
) {
    if rows == 0 || width == 0 {
        return;
    }
    let row = thread::blockIdx_x() as usize;
    if row >= rows as usize {
        return;
    }
    let width = width as usize;
    let tid = thread::threadIdx_x() as usize;
    static mut SUM: cuda_device::SharedArray<f32, 256> = cuda_device::SharedArray::UNINIT;
    let shared = unsafe { cuda_device::SharedArray::as_raw_mut_ptr(&raw mut SUM) };
    let mut sum = 0.0f32;
    let mut column = tid;
    while column < width {
        sum = float::add_rn_f32(sum, unsafe { *input.add(row * width + column) });
        column += THREADS;
    }
    unsafe { *shared.add(tid) = sum };
    let mut stride = THREADS / 2;
    while stride >= 1 {
        thread::sync_threads();
        if tid < stride {
            let left = unsafe { *shared.add(tid) };
            let right = unsafe { *shared.add(tid + stride) };
            unsafe { *shared.add(tid) = float::add_rn_f32(left, right) };
        }
        stride >>= 1;
    }
    thread::sync_threads();
    let mean = unsafe { *shared.add(0) } / width as f32;
    let mut variance = 0.0f32;
    column = tid;
    while column < width {
        let centered = gemma2_media_sub_rn_f32(unsafe { *input.add(row * width + column) }, mean);
        variance = float::add_rn_f32(variance, float::mul_rn_f32(centered, centered));
        column += THREADS;
    }
    unsafe { *shared.add(tid) = variance };
    let mut stride = THREADS / 2;
    while stride >= 1 {
        thread::sync_threads();
        if tid < stride {
            let left = unsafe { *shared.add(tid) };
            let right = unsafe { *shared.add(tid + stride) };
            unsafe { *shared.add(tid) = float::add_rn_f32(left, right) };
        }
        stride >>= 1;
    }
    thread::sync_threads();
    let denominator = float::sqrt_rn_f32(unsafe { *shared.add(0) } / width as f32 + eps);
    column = tid;
    while column < width {
        let centered = gemma2_media_sub_rn_f32(unsafe { *input.add(row * width + column) }, mean);
        let normalized = float::div_rn_f32(centered, denominator);
        let gamma = f32_from_bf16(unsafe { *scale.add(column) });
        let value = rounded(float::mul_rn_f32(normalized, gamma));
        let value = if value > 0.0 { value } else { 0.0 };
        store_bf16(out, row * width + column, value);
        column += THREADS;
    }
}

/// NHWC `[rows, width]` -> NCHW `[width, rows]` BF16 transpose.
#[inline(always)]
pub(crate) unsafe fn transpose_btwc_to_bctw_bf16(
    input: *const u16,
    out: *mut u16,
    rows: u32,
    width: u32,
) {
    let i = index();
    if i >= rows as usize * width as usize {
        return;
    }
    let row = i / width as usize;
    let column = i % width as usize;
    store_bf16(
        out,
        column * rows as usize + row,
        f32_from_bf16(unsafe { *input.add(i) }),
    );
}

/// Native second-block reshape `[t, w, c] -> [t, c, w]`.
#[inline(always)]
pub(crate) unsafe fn conv_rows_to_btcw_bf16(
    input: *const u16,
    out: *mut u16,
    time: u32,
    spatial: u32,
    channels: u32,
) {
    let i = index();
    if i >= time as usize * spatial as usize * channels as usize {
        return;
    }
    let t = i / (spatial as usize * channels as usize);
    let rest = i % (spatial as usize * channels as usize);
    let w = rest / channels as usize;
    let c = rest % channels as usize;
    let source = (t * spatial as usize + w) * channels as usize + c;
    store_bf16(out, i, f32_from_bf16(unsafe { *input.add(source) }));
}

/// Center `((pixel - .5) * 2)` in F32 before the patch projection.
#[inline(always)]
pub(crate) unsafe fn center_pixels_bf16(input: *const f32, out: *mut u16, count: u32) {
    let i = index();
    if i >= count as usize {
        return;
    }
    let value = float::mul_rn_f32(gemma2_media_sub_rn_f32(unsafe { *input.add(i) }, 0.5), 2.0);
    store_bf16(out, i, value);
}

/// BF16 patch rows plus the two row-major position-table rows.
#[inline(always)]
pub(crate) unsafe fn patch_add_position_bf16(
    input: *const u16,
    columns: *const u32,
    rows: *const u32,
    table: *const u16,
    out: *mut u16,
    count: u32,
    width: u32,
    positions: u32,
) {
    let i = index();
    if i >= count as usize * width as usize {
        return;
    }
    let row = i / width as usize;
    let column = i % width as usize;
    let position_column = unsafe { *columns.add(row) } as usize;
    let position_row = unsafe { *rows.add(row) } as usize;
    if position_column >= positions as usize || position_row >= positions as usize {
        return;
    }
    let table_row = width as usize * positions as usize;
    let position = rounded(float::add_rn_f32(
        f32_from_bf16(unsafe { *table.add(position_column * width as usize + column) }),
        f32_from_bf16(unsafe { *table.add(table_row + position_row * width as usize + column) }),
    ));
    let value = float::add_rn_f32(f32_from_bf16(unsafe { *input.add(i) }), position);
    store_bf16(out, i, value);
}

#[inline(always)]
fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        libm::log1pf(libm::expf(value))
    }
}

/// Axial cos/sin tables. `pairs = head_dim / 4`; output is `[rows, 2*pairs]`
/// with column angles first and row angles second, rounded to BF16.
#[inline(always)]
pub(crate) unsafe fn axial_tables_bf16(
    columns: *const u32,
    rows: *const u32,
    inverse: *const f32,
    cosine: *mut u16,
    sine: *mut u16,
    count: u32,
    pairs: u32,
) {
    let i = index();
    if i >= count as usize * pairs as usize {
        return;
    }
    let row = i / pairs as usize;
    let pair = i % pairs as usize;
    let frequency = unsafe { *inverse.add(pair) };
    let column_angle = float::mul_rn_f32(unsafe { *columns.add(row) } as f32, frequency);
    let row_angle = float::mul_rn_f32(unsafe { *rows.add(row) } as f32, frequency);
    let out = row * 2 * pairs as usize + pair;
    store_bf16(cosine, out, rounded(libm::cosf(column_angle)));
    store_bf16(sine, out, rounded(libm::sinf(column_angle)));
    store_bf16(cosine, out + pairs as usize, rounded(libm::cosf(row_angle)));
    store_bf16(sine, out + pairs as usize, rounded(libm::sinf(row_angle)));
}

/// Native axial `rotate2d` on `[tokens, heads, head_dim]` in place.
/// Each BF16 multiply/subtract is rounded exactly like the reference tensor ops.
#[inline(always)]
pub(crate) unsafe fn axial_rope_bf16(
    qk: *mut u16,
    cosine: *const u16,
    sine: *const u16,
    tokens: u32,
    heads: u32,
    dim: u32,
) {
    let half = dim as usize / 2;
    let quarter = dim as usize / 4;
    let pairs = dim as usize / 4;
    let i = index();
    if i >= tokens as usize * heads as usize * (dim as usize / 2) {
        return;
    }
    let axis = (i / (tokens as usize * heads as usize * quarter)).min(1);
    let rem = i % (tokens as usize * heads as usize * quarter);
    let token = rem / (heads as usize * quarter);
    let rem = rem % (heads as usize * quarter);
    let head = rem / quarter;
    let offset = rem % quarter;
    let base = (token * heads as usize + head) * dim as usize + axis * half;
    let a = f32_from_bf16(unsafe { *qk.add(base + offset) });
    let b = f32_from_bf16(unsafe { *qk.add(base + quarter + offset) });
    let c = f32_from_bf16(unsafe { *cosine.add(token * 2 * pairs + axis * pairs + offset) });
    let s = f32_from_bf16(unsafe { *sine.add(token * 2 * pairs + axis * pairs + offset) });
    let first = rounded(gemma2_media_sub_rn_f32(
        rounded(float::mul_rn_f32(a, c)),
        rounded(float::mul_rn_f32(b, s)),
    ));
    let second = rounded(float::add_rn_f32(
        rounded(float::mul_rn_f32(b, c)),
        rounded(float::mul_rn_f32(a, s)),
    ));
    store_bf16(qk, base + offset, first);
    store_bf16(qk, base + quarter + offset, second);
}

/// Packed noncausal BF16 attention with GQA and per-image cu_seqlens.
#[inline(always)]
pub(crate) unsafe fn attention_bf16(
    q: *const u16,
    k: *const u16,
    v: *const u16,
    out: *mut u16,
    cu_seqlens: *const u32,
    total: u32,
    batch: u32,
    heads: u32,
    kv_heads: u32,
    dim: u32,
    max_q: u32,
    scale: f32,
) {
    let _ = max_q;
    let dim = dim as usize;
    let group = (heads / kv_heads.max(1)) as usize;
    let i = index();
    if i >= total as usize * heads as usize {
        return;
    }
    let token = i / heads as usize;
    let head = i % heads as usize;
    let mut sequence = 0usize;
    while sequence + 1 < batch as usize
        && unsafe { *cu_seqlens.add(sequence + 1) } as usize <= token
    {
        sequence += 1;
    }
    let start = unsafe { *cu_seqlens.add(sequence) } as usize;
    let end = unsafe { *cu_seqlens.add(sequence + 1) } as usize;
    let kv_head = head / group;
    // Online softmax in F32 over the source row order.
    let mut maximum = f32::NEG_INFINITY;
    let row = |index: usize, head: usize| unsafe { q.add((index * heads as usize + head) * dim) };
    let krow =
        |index: usize, head: usize| unsafe { k.add((index * kv_heads as usize + head) * dim) };
    let mut key = start;
    while key < end {
        let mut dot = 0.0f32;
        let mut column = 0usize;
        while column < dim {
            dot = float::add_rn_f32(
                dot,
                float::mul_rn_f32(
                    f32_from_bf16(unsafe { *row(token, head).add(column) }),
                    f32_from_bf16(unsafe { *krow(key, kv_head).add(column) }),
                ),
            );
            column += 1;
        }
        dot = float::mul_rn_f32(dot, scale);
        if dot > maximum {
            maximum = dot;
        }
        key += 1;
    }
    let mut denominator = 0.0f32;
    let mut accumulator = [0.0f32; 256];
    debug_assert!(dim <= 256);
    let mut key = start;
    while key < end {
        let mut dot = 0.0f32;
        let mut column = 0usize;
        while column < dim {
            dot = float::add_rn_f32(
                dot,
                float::mul_rn_f32(
                    f32_from_bf16(unsafe { *row(token, head).add(column) }),
                    f32_from_bf16(unsafe { *krow(key, kv_head).add(column) }),
                ),
            );
            column += 1;
        }
        let probability = libm::expf(gemma2_media_sub_rn_f32(float::mul_rn_f32(dot, scale), maximum));
        denominator = float::add_rn_f32(denominator, probability);
        column = 0;
        while column < dim {
            accumulator[column] = float::add_rn_f32(
                accumulator[column],
                float::mul_rn_f32(
                    probability,
                    f32_from_bf16(unsafe {
                        *v.add((key * kv_heads as usize + kv_head) * dim + column)
                    }),
                ),
            );
            column += 1;
        }
        key += 1;
    }
    let mut column = 0usize;
    while column < dim {
        let value = if denominator != 0.0 {
            float::div_rn_f32(accumulator[column], denominator)
        } else {
            0.0
        };
        store_bf16(out, (token * heads as usize + head) * dim + column, value);
        column += 1;
    }
}

/// Gate/up GELU product with separate source matrices.
#[inline(always)]
pub(crate) unsafe fn gated_gelu_split_bf16(
    gate: *const u16,
    up: *const u16,
    out: *mut u16,
    rows: u32,
    width: u32,
) {
    let i = index();
    if i >= rows as usize * width as usize {
        return;
    }
    let value = f32_from_bf16(unsafe { *gate.add(i) });
    let square = rounded(float::mul_rn_f32(value, value));
    let cube = rounded(float::mul_rn_f32(square, value));
    let product = rounded(float::mul_rn_f32(rounded(0.044715f32), cube));
    let alpha = rounded(float::add_rn_f32(value, product));
    let angle = rounded(float::mul_rn_f32(
        rounded(0.797_884_560_802_865_4f32),
        alpha,
    ));
    let tangent = rounded(libm::tanhf(angle));
    let half = rounded(float::mul_rn_f32(0.5f32, value));
    let plus = rounded(float::add_rn_f32(1.0f32, tangent));
    let active = rounded(float::mul_rn_f32(half, plus));
    store_bf16(
        out,
        i,
        float::mul_rn_f32(active, f32_from_bf16(unsafe { *up.add(i) })),
    );
}

/// Native Gemma vision pooling: F32 mean of each `pool x pool` bin, scaled by
/// `sqrt(hidden)`, then one BF16 cast.
#[inline(always)]
pub(crate) unsafe fn vision_pool_bf16(
    input: *const u16,
    out: *mut u16,
    height: u32,
    width: u32,
    pool: u32,
    hidden: u32,
) {
    let out_w = width / pool;
    let out_h = height / pool;
    let i = index();
    if i >= out_h as usize * out_w as usize * hidden as usize {
        return;
    }
    let row = i / hidden as usize;
    let column = i % hidden as usize;
    let oy = row / out_w as usize;
    let ox = row % out_w as usize;
    let mut sum = 0.0f32;
    let inverse = 1.0 / (pool as usize * pool as usize) as f32;
    let mut y = 0u32;
    while y < pool {
        let mut x = 0u32;
        while x < pool {
            let source = ((oy * pool as usize + y as usize) * width as usize
                + ox * pool as usize
                + x as usize)
                * hidden as usize
                + column;
            sum = float::add_rn_f32(
                sum,
                float::mul_rn_f32(f32_from_bf16(unsafe { *input.add(source) }), inverse),
            );
            x += 1;
        }
        y += 1;
    }
    let rounded_mean = rounded(sum);
    let scaled = float::mul_rn_f32(rounded_mean, float::sqrt_rn_f32(hidden as f32));
    store_bf16(out, i, scaled);
}

/// In-place/external BF16 clamp against two exact BF16 bounds widened to F32.
#[inline(always)]
pub(crate) unsafe fn clamp_bf16(
    input: *const u16,
    out: *mut u16,
    count: u32,
    minimum: f32,
    maximum: f32,
) {
    let i = index();
    if i >= count as usize {
        return;
    }
    let value = f32_from_bf16(unsafe { *input.add(i) });
    let value = if value < minimum {
        minimum
    } else if value > maximum {
        maximum
    } else {
        value
    };
    store_bf16(out, i, value);
}

/// BF16 residual add with a BF16-rounded scale on the source first.
#[inline(always)]
pub(crate) unsafe fn residual_scale_add_bf16(
    destination: *mut u16,
    source: *const u16,
    count: u32,
    scale: f32,
) {
    let i = index();
    if i >= count as usize {
        return;
    }
    let source = rounded(float::mul_rn_f32(
        f32_from_bf16(unsafe { *source.add(i) }),
        scale,
    ));
    let value = float::add_rn_f32(f32_from_bf16(unsafe { *destination.add(i) }), source);
    store_bf16(destination, i, value);
}

/// Plain BF16 residual add (ATen BF16 add rounding).
#[inline(always)]
pub(crate) unsafe fn add_bf16(destination: *mut u16, source: *const u16, count: u32) {
    let i = index();
    if i >= count as usize {
        return;
    }
    let value = float::add_rn_f32(
        f32_from_bf16(unsafe { *destination.add(i) }),
        f32_from_bf16(unsafe { *source.add(i) }),
    );
    store_bf16(destination, i, value);
}

/// Row-wise BF16 bias add.
#[inline(always)]
pub(crate) unsafe fn bias_add_bf16(rows: u32, width: u32, bias: *const u16, data: *mut u16) {
    let i = index();
    if i >= rows as usize * width as usize {
        return;
    }
    let column = i % width as usize;
    let value = float::add_rn_f32(
        f32_from_bf16(unsafe { *data.add(i) }),
        f32_from_bf16(unsafe { *bias.add(column) }),
    );
    store_bf16(data, i, value);
}

/// Native audio Q/K scaling before the relative attention logits.
#[inline(always)]
pub(crate) unsafe fn audio_scaled_qk_f32(
    q: *const u16,
    k: *const u16,
    per_dim_scale: *const u16,
    q_out: *mut f32,
    k_out: *mut f32,
    tokens: u32,
    heads: u32,
    dim: u32,
) {
    let i = index();
    if i >= tokens as usize * heads as usize * dim as usize {
        return;
    }
    let column = i % dim as usize;
    let scale = rounded(softplus(f32_from_bf16(unsafe {
        *per_dim_scale.add(column)
    })));
    let q_scale = float::mul_rn_f32(
        float::div_rn_f32(1.0, float::sqrt_rn_f32(dim as f32)),
        float::div_rn_f32(1.0, libm::logf(2.0)),
    );
    let q_scale = float::mul_rn_f32(q_scale, scale);
    let k_scale = float::div_rn_f32(libm::log1pf(libm::expf(1.0)), libm::logf(2.0));
    unsafe { *q_out.add(i) = float::mul_rn_f32(f32_from_bf16(unsafe { *q.add(i) }), q_scale) };
    unsafe { *k_out.add(i) = float::mul_rn_f32(f32_from_bf16(unsafe { *k.add(i) }), k_scale) };
}

/// Native conformer chunked relative attention. One thread owns one query head.
#[inline(always)]
pub(crate) unsafe fn audio_attention_bf16(
    q: *const f32,
    k: *const f32,
    v: *const u16,
    relative: *const u16,
    valid: *const u8,
    out: *mut u16,
    tokens: u32,
    heads: u32,
    dim: u32,
    chunk: u32,
    past: u32,
    future: u32,
    span: u32,
    cap: f32,
    invalid: f32,
) {
    let tokens = tokens as usize;
    let heads = heads as usize;
    let dim = dim as usize;
    let chunk = chunk as usize;
    let past = past as usize;
    let future = future as usize;
    let span = span as usize;
    let chunks = tokens.div_ceil(chunk);
    let i = index();
    if i >= chunks * chunk * heads {
        return;
    }
    let chunk_index = i / (chunk * heads);
    let rem = i % (chunk * heads);
    let query_local = rem / heads;
    let head = rem % heads;
    let start = chunk_index * chunk;
    let count = chunk.min(tokens - start);
    if query_local >= count {
        return;
    }
    let query = start + query_local;
    let key_start = start.saturating_sub(past);
    let key_end = (start + count + future).min(tokens);
    let context = chunk + past + future;
    let mut maximum = f32::NEG_INFINITY;
    let mut logits = [0.0f32; 64];
    debug_assert!(key_end - key_start <= 64);
    // The Native reshape/gather can select a different query row's relative
    // logits. Compute the whole chunk's `[count, span]` table, not just this
    // query's row.
    let mut relative_logits = [0.0f32; 64 * 16];
    debug_assert!(chunk <= 64 && span <= 16);
    let mut relative_query = 0usize;
    while relative_query < count {
        let mut source = 0usize;
        while source < span {
            let mut dot = 0.0f32;
            let mut column = 0usize;
            while column < dim {
                dot = float::add_rn_f32(
                    dot,
                    float::mul_rn_f32(
                        unsafe { *q.add(((start + relative_query) * heads + head) * dim + column) },
                        f32_from_bf16(unsafe {
                            *relative.add((source * heads + head) * dim + column)
                        }),
                    ),
                );
                column += 1;
            }
            relative_logits[relative_query * span + source] = dot;
            source += 1;
        }
        relative_query += 1;
    }
    let mut key = key_start;
    while key < key_end {
        let shifted = query_local * context + (key + past - start);
        let source_row = shifted / (context + 1);
        let source_column = shifted % (context + 1);
        let relative_term = if source_column < span && source_row < count {
            relative_logits[source_row * span + source_column]
        } else {
            0.0
        };
        let mut dot = 0.0f32;
        let mut column = 0usize;
        while column < dim {
            dot = float::add_rn_f32(
                dot,
                float::mul_rn_f32(
                    unsafe { *q.add((query * heads + head) * dim + column) },
                    unsafe { *k.add((key * heads + head) * dim + column) },
                ),
            );
            column += 1;
        }
        let mut value = float::add_rn_f32(dot, relative_term);
        value = float::mul_rn_f32(libm::tanhf(float::div_rn_f32(value, cap)), cap);
        let delta = key as i64 - query as i64;
        let allowed = ((delta <= 0 && delta > -(past as i64))
            || (delta > 0 && delta < future as i64))
            && unsafe { *valid.add(key) } != 0;
        let value = if allowed { value } else { invalid };
        logits[(key - key_start) as usize] = value;
        if value > maximum {
            maximum = value;
        }
        key += 1;
    }
    let mut denominator = 0.0f32;
    let mut accumulator = [0.0f32; 256];
    debug_assert!(dim <= 256);
    let mut key = key_start;
    while key < key_end {
        let probability = libm::expf(gemma2_media_sub_rn_f32(
            logits[(key - key_start) as usize],
            maximum,
        ));
        denominator = float::add_rn_f32(denominator, probability);
        let mut column = 0usize;
        while column < dim {
            accumulator[column] = float::add_rn_f32(
                accumulator[column],
                float::mul_rn_f32(
                    probability,
                    f32_from_bf16(unsafe { *v.add((key * heads + head) * dim + column) }),
                ),
            );
            column += 1;
        }
        key += 1;
    }
    let mut column = 0usize;
    while column < dim {
        let value = if denominator != 0.0 {
            float::div_rn_f32(accumulator[column], denominator)
        } else {
            0.0
        };
        store_bf16(out, (query * heads + head) * dim + column, value);
        column += 1;
    }
}

/// Conformer fused GLU: `first * sigmoid(second)` with one BF16 rounding.
#[inline(always)]
pub(crate) unsafe fn glu_bf16(input: *const u16, out: *mut u16, rows: u32, width: u32) {
    let i = index();
    if i >= rows as usize * width as usize {
        return;
    }
    let row = i / width as usize;
    let column = i % width as usize;
    let first = f32_from_bf16(unsafe { *input.add(row * 2 * width as usize + column) });
    let second =
        f32_from_bf16(unsafe { *input.add(row * 2 * width as usize + width as usize + column) });
    let value = float::mul_rn_f32(first, 1.0 / (1.0 + libm::expf(-second)));
    store_bf16(out, i, value);
}

/// Causal depthwise Conv1d with left padding `kernel - 1`, then BF16 and clamp.
#[inline(always)]
pub(crate) unsafe fn depthwise_conv1d_causal_bf16(
    input: *const u16,
    weight: *const u16,
    out: *mut u16,
    tokens: u32,
    channels: u32,
    kernel: u32,
    clip: f32,
) {
    let i = index();
    if i >= tokens as usize * channels as usize {
        return;
    }
    let token = i / channels as usize;
    let channel = i % channels as usize;
    let mut sum = 0.0f32;
    let mut tap = 0u32;
    while tap < kernel {
        let source = token as i64 + tap as i64 - (kernel as i64 - 1);
        if source >= 0 && source < tokens as i64 {
            let value =
                f32_from_bf16(unsafe { *input.add(source as usize * channels as usize + channel) });
            let filter =
                f32_from_bf16(unsafe { *weight.add((channel * kernel as usize) + tap as usize) });
            sum = float::add_rn_f32(sum, float::mul_rn_f32(value, filter));
        }
        tap += 1;
    }
    let value = rounded(sum);
    let value = if value < -clip {
        -clip
    } else if value > clip {
        clip
    } else {
        value
    };
    store_bf16(out, i, value);
}

/// Native F32 SiLU followed by BF16 rounding.
#[inline(always)]
pub(crate) unsafe fn silu_bf16(input: *const u16, out: *mut u16, count: u32) {
    let i = index();
    if i >= count as usize {
        return;
    }
    let value = f32_from_bf16(unsafe { *input.add(i) });
    let active = float::mul_rn_f32(value, 1.0 / (1.0 + libm::expf(-value)));
    store_bf16(out, i, active);
}

/// Copy selected BF16 rows into the packed media-row destination.
#[inline(always)]
pub(crate) unsafe fn select_rows_bf16(
    input: *const u16,
    out: *mut u16,
    selected: *const u32,
    rows: u32,
    width: u32,
) {
    let i = index();
    if i >= rows as usize * width as usize {
        return;
    }
    let row = i / width as usize;
    let column = i % width as usize;
    let source = unsafe { *selected.add(row) } as usize;
    store_bf16(
        out,
        i,
        f32_from_bf16(unsafe { *input.add(source * width as usize + column) }),
    );
}

/// Scatter prepared media rows into the text token state.
#[inline(always)]
pub(crate) unsafe fn scatter_rows_bf16(
    state: *mut u16,
    rows: *const u16,
    indices: *const u32,
    count: u32,
    width: u32,
) {
    let i = index();
    if i >= count as usize * width as usize {
        return;
    }
    let row = i / width as usize;
    let column = i % width as usize;
    let destination = unsafe { *indices.add(row) } as usize;
    store_bf16(
        state,
        destination * width as usize + column,
        f32_from_bf16(unsafe { *rows.add(i) }),
    );
}
/// Native `gemma_audio_norm_cuda`: F32 variance, runtime `powf(-0.5)`, F32
/// scale, one BF16 cast. Threads follow the reference power-of-two rule.
#[inline(always)]
pub(crate) unsafe fn audio_norm_bf16_f32scale(
    input: *const u16,
    scale: *const f32,
    out: *mut u16,
    rows: u32,
    width: u32,
    eps: f32,
) {
    if rows == 0 || width == 0 {
        return;
    }
    static mut SUMS: cuda_device::SharedArray<f32, 1024> = cuda_device::SharedArray::UNINIT;
    let sums = unsafe { cuda_device::SharedArray::as_raw_mut_ptr(&raw mut SUMS) };
    let row = thread::blockIdx_x() as usize;
    if row >= rows as usize {
        return;
    }
    let width = width as usize;
    let tid = thread::threadIdx_x() as usize;
    let mut sum = 0.0f32;
    let mut column = tid;
    while column < width {
        let value = f32_from_bf16(unsafe { *input.add(row * width + column) });
        sum = float::add_rn_f32(sum, float::mul_rn_f32(value, value));
        column += thread::blockDim_x() as usize;
    }
    unsafe { *sums.add(tid) = sum };
    let mut stride = thread::blockDim_x() as usize / 2;
    while stride > 0 {
        thread::sync_threads();
        if tid < stride {
            let left = unsafe { *sums.add(tid) };
            let right = unsafe { *sums.add(tid + stride) };
            unsafe { *sums.add(tid) = float::add_rn_f32(left, right) };
        }
        stride >>= 1;
    }
    thread::sync_threads();
    let divisor = (1.0f64 / width as f64) as f32;
    let variance = float::add_rn_f32(float::mul_rn_f32(unsafe { *sums.add(0) }, divisor), eps);
    let inverse = libm::powf(variance, -0.5);
    column = tid;
    while column < width {
        let normalized = float::mul_rn_f32(
            f32_from_bf16(unsafe { *input.add(row * width + column) }),
            inverse,
        );
        let scaled = float::mul_rn_f32(normalized, unsafe { *scale.add(column) });
        store_bf16(out, row * width + column, scaled);
        column += thread::blockDim_x() as usize;
    }
}
