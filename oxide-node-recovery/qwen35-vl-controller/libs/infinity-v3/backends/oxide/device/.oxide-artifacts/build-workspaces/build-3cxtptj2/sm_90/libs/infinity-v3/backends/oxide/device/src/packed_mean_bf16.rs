// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Standalone native BF16 packed mean pooling for selected ragged spans.
//!
//! The reduction is a direct port of `packed_mean_pool<__nv_bfloat16>` from the
//! retained `backends/libtorch/cpp/vendor/candle_kernels/pooling/mean_pool.cu`
//! at commit `ccb36282eaa8a776f527b609337b958e772d2b5a`. Native relocation
//! provenance is retained in `vendor/candle_kernels/PROVENANCE.md`; inherited
//! rights and notices follow the workspace LICENSE and THIRD_PARTY_NOTICES.
//! Span selection, cumulative offsets and the CTA-uniform barrier shape follow
//! the supplied `encoder_f16.rs` mean-pool implementation. Every addition, the
//! halving tree, the reciprocal scaling and the trailing add are native BF16
//! round-to-nearest operations; only the final read is widened to F32.

use cuda_device::bf16x2::{add_bf16x2, mul_bf16x2};
use cuda_device::{device, thread, SharedArray};

const TILE: usize = 8;
const MAX_THREADS: usize = 1024;
const PARTIALS: usize = MAX_THREADS * TILE;

/// Read BF16 storage as the exactly equal F32 value by widening its bits.
#[device]
#[inline(always)]
fn qwen35_packed_mean_f32_from_bf16(value: u16) -> f32 {
    f32::from_bits((value as u32) << 16)
}

#[device]
#[inline(always)]
fn qwen35_packed_mean_bf16_pair(bits: u16) -> u32 {
    u32::from(bits) | (u32::from(bits) << 16)
}

#[device]
#[inline(always)]
fn qwen35_packed_mean_bf16_narrow(packed: u32) -> u16 {
    (packed & 0xffff) as u16
}

/// Native BF16 round-to-nearest addition. Both halves of the packed operand
/// hold the same value, so the low half is the scalar result.
#[device]
#[inline(always)]
fn qwen35_packed_mean_bf16_add(left: u16, right: u16) -> u16 {
    qwen35_packed_mean_bf16_narrow(add_bf16x2(qwen35_packed_mean_bf16_pair(left), qwen35_packed_mean_bf16_pair(right)))
}

/// Native BF16 round-to-nearest multiplication, using the same packed trick.
#[device]
#[inline(always)]
fn qwen35_packed_mean_bf16_mul(left: u16, right: u16) -> u16 {
    qwen35_packed_mean_bf16_narrow(mul_bf16x2(qwen35_packed_mean_bf16_pair(left), qwen35_packed_mean_bf16_pair(right)))
}

/// Convert F64 to BF16 with one round-to-nearest step through the pinned
/// `cuda_device::ptx_asm!` surface. `cvt.rn.bf16.f64` is a native
/// SM90/SM100/SM120 instruction; going through F32 would double-round.
#[device]
#[inline(always)]
fn qwen35_packed_mean_bf16_from_f64(value: f64) -> u16 {
    let bits: u16;
    unsafe {
        cuda_device::ptx_asm!(
            "cvt.rn.bf16.f64 %0, %1;",
            out("=h") bits,
            in("d") value,
        );
    }
    bits
}

/// Actual spans packed mean for BF16 storage, grid=(count,ceil(hidden/8),1),
/// block=256. One CTA reduces eight adjacent features. The shared partials
/// layout, strided summation and halving tree match the native control exactly.
///
/// # Safety
/// Launch exactly 256 threads per CTA with grid `(count, ceil(hidden / 8), 1)`.
/// Input is contiguous row-major BF16; output is a disjoint F32 `[count, hidden]`.
/// Selected indices and their two cumulative boundaries must be valid, each
/// selected span must be nonempty, and every span must lie within input storage.
/// Input, boundaries and selections stay initialized/alive until stream completion;
/// output must provide writable storage for every selected row and feature.
#[device]
#[inline(always)]
pub(crate) unsafe fn packed_mean_pool_bf16(
    input: *const u16,
    cu_seqlens: *const u32,
    selected: *const u32,
    out: *mut f32,
    count: u32,
    hidden: u32,
) {
    let sequence = thread::blockIdx_x() as usize;
    if sequence >= count as usize || hidden == 0 {
        return;
    }
    let width = hidden as usize;
    let base = thread::blockIdx_y() as usize * TILE;
    let index = unsafe { *selected.add(sequence) } as usize;
    let start = unsafe { *cu_seqlens.add(index) } as usize;
    let end = unsafe { *cu_seqlens.add(index + 1) } as usize;
    if end <= start {
        return;
    }
    let length = end - start;
    let mut threads = 1usize;
    while threads < length && threads < MAX_THREADS {
        threads *= 2;
    }
    static mut PARTIAL: SharedArray<u16, PARTIALS> = SharedArray::UNINIT;
    let partial = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PARTIAL) };
    let tid = thread::threadIdx_x() as usize;
    let stride = thread::blockDim_x() as usize;
    let mut i = tid;
    while i < threads * TILE {
        let feature = base + i % TILE;
        let mut sum = 0u16;
        if feature < width {
            let mut row = i / TILE;
            while row < length {
                sum = qwen35_packed_mean_bf16_add(sum, unsafe { *input.add((start + row) * width + feature) });
                row += threads;
            }
        }
        unsafe { *partial.add(i) = sum };
        i += stride;
    }
    let mut step = threads / 2;
    while step > 0 {
        thread::sync_threads();
        let mut i = tid;
        while i < step * TILE {
            let left = unsafe { *partial.add(i) };
            let right = unsafe { *partial.add(i + step * TILE) };
            unsafe { *partial.add(i) = qwen35_packed_mean_bf16_add(left, right) };
            i += stride;
        }
        step /= 2;
    }
    thread::sync_threads();
    if tid < TILE && base + tid < width {
        let reciprocal = qwen35_packed_mean_bf16_from_f64(cuda_device::float::div_rn_f64(1.0f64, length as f64));
        let scaled = qwen35_packed_mean_bf16_mul(unsafe { *partial.add(tid) }, reciprocal);
        let rounded = qwen35_packed_mean_bf16_add(scaled, 0);
        unsafe {
            *out.add(sequence * width + base + tid) = qwen35_packed_mean_f32_from_bf16(rounded);
        }
    }
}
