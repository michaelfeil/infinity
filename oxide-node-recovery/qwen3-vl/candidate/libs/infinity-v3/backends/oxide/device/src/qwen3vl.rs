//! Qwen3-VL vision arithmetic kernels.
//!
//! These are additive source-only exports for Root's SM90/SM100/SM120 build.
//! Existing exports and their ABI are untouched. All pointers are host-validated
//! contiguous buffers; arithmetic uses FP32 internally and rounds to BF16 at the
//! documented Native rounding points.

use cuda_device::thread;

#[inline(always)]
fn rounded(value: f32) -> f32 {
    super::f32_from_bf16(super::bf16_from_f32(value))
}

#[inline(always)]
fn bf16(value: f32) -> u16 {
    super::bf16_from_f32(value)
}

/// Copy or add BF16 rows selected by a u32 row-index list.
/// Grid=ceil(rows*width/256), block=256. `add=0` copies, `add=1` adds.
pub unsafe fn rows_bf16(
    dst: *mut u16,
    src: *const u16,
    indices: *const u32,
    count: u32,
    width: u32,
    add: u32,
) {
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    let total = count as usize * width as usize;
    if index >= total {
        return;
    }
    let row = index / width as usize;
    let column = index % width as usize;
    let destination = unsafe { *indices.add(row) } as usize * width as usize + column;
    let source = index;
    unsafe {
        let value = *src.add(source);
        if add == 0 {
            *dst.add(destination) = value;
        } else {
            let sum = super::f32_from_bf16(*dst.add(destination)) + super::f32_from_bf16(value);
            *dst.add(destination) = bf16(rounded(sum));
        }
    }
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

/// Four-corner learned-position interpolation followed by positional add.
/// Grid=ceil(tokens*hidden/256), block=256.
pub unsafe fn pos_interp(
    pos_embed: *const u16,
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
    if index >= total {
        return;
    }
    let token = index / hidden as usize;
    let hidden_column = index % hidden as usize;
    let merge = merge as usize;
    let merge_area = merge * merge;
    let width_usize = width as usize;
    let height_usize = height as usize;
    let side = side as usize;
    let c = (token % merge) + ((token / merge_area) % (width_usize / merge)) * merge;
    let r = ((token / merge) % merge)
        + ((token / (merge * width_usize)) % (height_usize / merge)) * merge;
    let row_scale = if height_usize == 1 {
        0.0
    } else {
        (side - 1) as f32 / (height_usize - 1) as f32
    };
    let column_scale = if width_usize == 1 {
        0.0
    } else {
        (side - 1) as f32 / (width_usize - 1) as f32
    };
    let rf = r as f32 * row_scale;
    let cf = c as f32 * column_scale;
    let r0 = libm::floorf(rf) as usize;
    let c0 = libm::floorf(cf) as usize;
    let r1 = (libm::ceilf(rf) as usize).min(side - 1);
    let c1 = (libm::ceilf(cf) as usize).min(side - 1);
    let dr = rf - r0 as f32;
    let dc = cf - c0 as f32;
    let corner = |row: usize, grid_column: usize| -> f32 {
        unsafe {
            super::f32_from_bf16(*pos_embed.add((row * side + grid_column) * hidden as usize + hidden_column))
        }
    };
    let c00 = rounded(corner(r0, c0) * rounded((1.0 - dr) * (1.0 - dc)));
    let c01 = rounded(corner(r0, c1) * rounded((1.0 - dr) * dc));
    let c10 = rounded(corner(r1, c0) * rounded(dr * (1.0 - dc)));
    let c11 = rounded(corner(r1, c1) * rounded(dr * dc));
    let positional = rounded(c00 + c01 + c10 + c11);
    unsafe {
        let state = super::f32_from_bf16(*states.add(index));
        *states.add(index) = bf16(rounded(state + positional));
    }
}

/// 2-D split-half vision rotary on packed Q/K rows.
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
    let half = (head_dim / 2) as usize;
    let quarter = (head_dim / 4) as usize;
    let total = tokens as usize * heads as usize * half;
    if index >= total {
        return;
    }
    let pair = index % half;
    let head = (index / half) % heads as usize;
    let token = index / (half * heads as usize);
    let merge = merge as usize;
    let merge_area = merge * merge;
    let width_usize = width as usize;
    let height_usize = height as usize;
    let c = (token % merge) + ((token / merge_area) % (width_usize / merge)) * merge;
    let r = ((token / merge) % merge)
        + ((token / (merge * width_usize)) % (height_usize / merge)) * merge;
    let j = if pair < quarter { pair } else { pair - quarter };
    let frequency = libm::expf(
        -9.210_340_371_976_184f32 * (2.0 * j as f32) / (half as f32),
    );
    let angle = if pair < quarter {
        r as f32 * frequency
    } else {
        c as f32 * frequency
    };
    let cosine = libm::cosf(angle);
    let sine = libm::sinf(angle);
    unsafe {
        let q_base = token * q_stride as usize + head * head_dim as usize;
        let a = super::f32_from_bf16(*q.add(q_base + pair));
        let b = super::f32_from_bf16(*q.add(q_base + pair + half));
        *q.add(q_base + pair) = bf16(rounded(a * cosine - b * sine));
        *q.add(q_base + pair + half) = bf16(rounded(b * cosine + a * sine));
        let k_base = token * k_stride as usize + head * head_dim as usize;
        let a = super::f32_from_bf16(*k.add(k_base + pair));
        let b = super::f32_from_bf16(*k.add(k_base + pair + half));
        *k.add(k_base + pair) = bf16(rounded(a * cosine - b * sine));
        *k.add(k_base + pair + half) = bf16(rounded(b * cosine + a * sine));
    }
}

/// Tanh-approximate GELU in place, matching at::gelu(..., "tanh").
/// Grid=ceil(count/256), block=256.
pub unsafe fn gelu_tanh_bf16(inout: *mut u16, count: u32) {
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if index >= count as usize {
        return;
    }
    unsafe {
        let x = super::f32_from_bf16(*inout.add(index));
        let cube = x * x * x;
        let tanh = libm::tanhf(0.797_884_560_802_865_4f32 * (x + 0.044_715 * cube));
        let y = 0.5 * x * (1.0 + tanh);
        *inout.add(index) = bf16(rounded(y));
    }
}

/// SiLU in place. Grid=ceil(count/256), block=256.
pub unsafe fn silu_bf16(inout: *mut u16, count: u32) {
    let index = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
    if index >= count as usize {
        return;
    }
    unsafe {
        let x = super::f32_from_bf16(*inout.add(index));
        let y = x / (1.0 + libm::expf(-x));
        *inout.add(index) = bf16(rounded(y));
    }
}
