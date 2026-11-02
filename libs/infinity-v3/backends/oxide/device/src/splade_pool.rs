// SPDX-License-Identifier: LicenseRef-Infinity-Restricted-MIT-1.0
//
// New SPLADE reduction candidate for the Infinity V3 Oxide backend, offered
// under the repository's Infinity Restricted MIT License 1.0. Public contract
// derived from michaelfeil/infinity commit
// 524235296c1863daebd630ccb8ff5dd6af62399d:
//   libs/infinity-v3/backends/libtorch/cpp/models/bert_model.h
//   libs/infinity-v3/backends/libtorch/cpp/bridge.cpp
// Retain the repository LICENSE, NOTICE and upstream Hugging Face attribution.

use cuda_device::{SharedArray, device, thread};

const FEATURE_TILE: u32 = 32;
const TOKEN_STRIDE: u32 = 4;
const EMPTY_INDEX: u32 = u32::MAX;

#[device]
fn splade_pool_consider(value: f32, index: u32, mut best: f32, mut best_index: u32) -> (f32, u32) {
    if best_index == EMPTY_INDEX {
        return (value, index);
    }

    let value_nan = value != value;
    let best_nan = best != best;
    if value_nan && best_nan {
        if index < best_index {
            best = value;
            best_index = index;
        }
        return (best, best_index);
    }
    if value_nan {
        return (value, index);
    }
    if best_nan {
        return (best, best_index);
    }

    if value > best || (value == best && index < best_index) {
        best = value;
        best_index = index;
    }
    (best, best_index)
}

#[device]
fn splade_pool_bf16_to_f32(raw: u16) -> f32 {
    f32::from_bits(u32::from(raw) << 16)
}

#[device]
fn splade_pool_f32_to_bf16(value: f32) -> u16 {
    cuda_device::convert::cvt_bf16x2_f32(value, value) as u16
}

#[device]
pub unsafe fn splade_pool_bf16_f32(
    input: *const u16,
    spans: *const u32,
    output: *mut f32,
    width: u32,
    selected: u32,
) {
    if selected == 0 || width == 0 {
        return;
    }

    let feature_tiles = (width as u64 + (FEATURE_TILE as u64 - 1)) / FEATURE_TILE as u64;
    let block = thread::blockIdx_x() as u64;
    if block >= feature_tiles * selected as u64 {
        return;
    }
    let span = (block / feature_tiles) as u32;
    let tile = (block % feature_tiles) as u32;

    let tx = thread::threadIdx_x();
    let x = tx % FEATURE_TILE;
    let y = tx / FEATURE_TILE;
    let feature = tile * FEATURE_TILE + x;
    let active = feature < width;

    let span_offset = span as u64 * 2;
    let span_start = unsafe { spans.add(span_offset as usize).read() };
    let span_length = unsafe { spans.add(span_offset as usize + 1).read() };

    static mut VALUES: SharedArray<f32, 128> = SharedArray::UNINIT;
    let values = unsafe { SharedArray::as_raw_mut_ptr(&raw mut VALUES) };
    static mut INDICES: SharedArray<u32, 128> = SharedArray::UNINIT;
    let indices = unsafe { SharedArray::as_raw_mut_ptr(&raw mut INDICES) };

    let mut best = 0.0f32;
    let mut best_index = EMPTY_INDEX;
    if active {
        let mut token = y as u64;
        while token < span_length as u64 {
            let row = span_start as u64 + token;
            let offset = row * width as u64 + feature as u64;
            let value = splade_pool_bf16_to_f32(unsafe { input.add(offset as usize).read() });
            let merged = splade_pool_consider(value, token as u32, best, best_index);
            best = merged.0;
            best_index = merged.1;
            token += TOKEN_STRIDE as u64;
        }
    }

    let shared_index = (y * FEATURE_TILE + x) as usize;
    unsafe { values.add(shared_index).write(best) };
    unsafe { indices.add(shared_index).write(best_index) };
    thread::sync_threads();

    if y == 0 && active {
        let mut merged_value = 0.0f32;
        let mut merged_index = EMPTY_INDEX;
        let mut lane = 0u32;
        while lane < TOKEN_STRIDE {
            let lane_index = unsafe { indices.add((lane * FEATURE_TILE + x) as usize).read() };
            if lane_index != EMPTY_INDEX {
                let merged = splade_pool_consider(
                    unsafe { values.add((lane * FEATURE_TILE + x) as usize).read() },
                    lane_index,
                    merged_value,
                    merged_index,
                );
                merged_value = merged.0;
                merged_index = merged.1;
            }
            lane += 1;
        }

        let output_offset = span as u64 * width as u64 + feature as u64;
        let result = if merged_index == EMPTY_INDEX {
            0.0f32
        } else {
            libm::log1pf(merged_value)
        };
        let rounded = splade_pool_f32_to_bf16(result);
        unsafe {
            output
                .add(output_offset as usize)
                .write(splade_pool_bf16_to_f32(rounded))
        };
    }
}
