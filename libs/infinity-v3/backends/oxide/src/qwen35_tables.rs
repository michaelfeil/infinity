// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Cold native Qwen3.5 partial-64 BF16 rotary tables.
//!
//! The host derives only the 32 native inverse frequencies. The existing device
//! initializer evaluates the FP32 phase and libdevice cosine/sine, rounds both
//! back to BF16, and stores them as F32 coefficients. Construction drains the
//! stream before publication and never runs per request.
use crate::{arg, launch_threads, product, start, u32_size};
use cuda_core::{DeviceBuffer, Function, Stream};
use infinity_v3_backend_core::BackendError;
use std::sync::Arc;

const ROTARY: usize = 64;
const PAIRS: usize = ROTARY / 2;
const THREADS: usize = 256;
const MAX_TABLE_BYTES: usize = 128 * 1024 * 1024;

fn validated_theta(theta: f32) -> Result<f32, BackendError> {
    if !theta.is_finite() || theta <= 0.0 {
        return Err(start("Qwen3.5 rotary theta must be finite and positive"));
    }
    Ok(theta)
}

fn validated_rotary(rotary: usize) -> Result<(), BackendError> {
    if rotary != ROTARY {
        return Err(start("Qwen3.5 rotary must be the native partial width 64"));
    }
    Ok(())
}

/// Native expression order:
/// `1.f / std::pow(float(theta), float(2 * i) / float(rotary))`.
pub(crate) fn inverse_frequencies(theta: f32, rotary: usize) -> Result<Vec<f32>, BackendError> {
    let theta = validated_theta(theta)?;
    validated_rotary(rotary)?;
    let mut result = Vec::new();
    result.try_reserve_exact(PAIRS).map_err(start)?;
    for index in 0..PAIRS {
        let exponent = (index * 2) as f32 / rotary as f32;
        let frequency = 1.0f32 / theta.powf(exponent);
        if !frequency.is_finite() {
            return Err(start("Qwen3.5 rotary inverse frequency must be finite"));
        }
        result.push(frequency);
    }
    Ok(result)
}

struct TablesPlan {
    positions: u32,
    elements: usize,
    total_bytes: usize,
    inverse: Vec<f32>,
}

fn tables_plan(
    theta: f32,
    rotary: usize,
    positions: usize,
    byte_limit: usize,
) -> Result<TablesPlan, BackendError> {
    let inverse = inverse_frequencies(theta, rotary)?;
    if positions == 0 {
        return Err(start("Qwen3.5 rotary positions must be positive"));
    }
    let positions = u32_size(positions)?;
    let elements = product(positions as usize, PAIRS)?;
    let table_bytes = product(elements, core::mem::size_of::<f32>())?;
    let tables_bytes = table_bytes
        .checked_mul(2)
        .ok_or_else(|| start("Qwen3.5 dual-table byte size overflowed usize"))?;
    let inverse_bytes = product(PAIRS, core::mem::size_of::<f32>())?;
    let total_bytes = tables_bytes
        .checked_add(inverse_bytes)
        .ok_or_else(|| start("Qwen3.5 rotary byte budget overflowed usize"))?;
    if total_bytes > byte_limit {
        return Err(start("Qwen3.5 rotary tables exceed the byte budget"));
    }
    Ok(TablesPlan {
        positions,
        elements,
        total_bytes,
        inverse,
    })
}

/// Pure CPU planning for both output tables and the temporary inverse upload.
pub(crate) fn required_bytes(
    theta: f32,
    rotary: usize,
    positions: usize,
) -> Result<usize, BackendError> {
    Ok(tables_plan(theta, rotary, positions, usize::MAX)?.total_bytes)
}

fn launch_blocks(elements: usize) -> Result<u32, BackendError> {
    let rounded = elements
        .checked_add(THREADS - 1)
        .ok_or_else(|| start("Qwen3.5 rotary launch grid overflowed usize"))?;
    let blocks = rounded / THREADS;
    if blocks > i32::MAX as usize {
        return Err(start("Qwen3.5 rotary launch grid exceeds CUDA x dimension"));
    }
    u32::try_from(blocks).map_err(|_| start("Qwen3.5 rotary launch grid exceeds CUDA limits"))
}

pub(crate) struct Qwen35Tables {
    pub(crate) cosine: DeviceBuffer<f32>,
    pub(crate) sine: DeviceBuffer<f32>,
    pub(crate) positions: usize,
    pub(crate) pairs: usize,
}

struct PendingTables {
    stream: Arc<Stream>,
    initializer: Option<Arc<Function>>,
    inverse: Option<DeviceBuffer<f32>>,
    tables: Option<Qwen35Tables>,
    enqueued: bool,
}

impl PendingTables {
    fn retain_after_failed_drain(&mut self) {
        if self.tables.is_none() {
            return;
        }
        let tables = self.tables.take();
        let inverse = self.inverse.take();
        let initializer = self.initializer.take();
        let stream = Arc::clone(&self.stream);
        self.enqueued = false;
        std::mem::forget((tables, inverse, initializer, stream));
    }
}

impl Drop for PendingTables {
    fn drop(&mut self) {
        if !self.enqueued || self.tables.is_none() {
            return;
        }
        if self.stream.synchronize().is_err() {
            self.retain_after_failed_drain();
        }
    }
}

impl Qwen35Tables {
    pub(crate) fn new(
        stream: &Arc<Stream>,
        initializer: Arc<Function>,
        theta: f32,
        rotary: usize,
        positions: usize,
    ) -> Result<Self, BackendError> {
        let plan = tables_plan(theta, rotary, positions, MAX_TABLE_BYTES)?;
        debug_assert!(plan.total_bytes <= MAX_TABLE_BYTES);
        let grid = launch_blocks(plan.elements)?;
        let cosine = unsafe { DeviceBuffer::<f32>::uninitialized_async(stream, plan.elements) }
            .map_err(start)?;
        let sine = unsafe { DeviceBuffer::<f32>::uninitialized_async(stream, plan.elements) }
            .map_err(start)?;
        let inverse = DeviceBuffer::from_host(stream, &plan.inverse).map_err(start)?;
        let mut pending = PendingTables {
            stream: Arc::clone(stream),
            initializer: Some(initializer),
            inverse: Some(inverse),
            tables: Some(Self {
                cosine,
                sine,
                positions: plan.positions as usize,
                pairs: PAIRS,
            }),
            enqueued: false,
        };
        let (mut inv, mut cos, mut sin) = {
            let tables = pending
                .tables
                .as_ref()
                .ok_or_else(|| start("Missing Qwen3.5 table constructor owner"))?;
            let inverse = pending
                .inverse
                .as_ref()
                .ok_or_else(|| start("Missing Qwen3.5 inverse upload owner"))?;
            (
                inverse.cu_deviceptr(),
                tables.cosine.cu_deviceptr(),
                tables.sine.cu_deviceptr(),
            )
        };
        let mut positions = plan.positions;
        let mut pairs = u32_size(PAIRS)?;
        pending.enqueued = true;
        let initializer = pending
            .initializer
            .as_ref()
            .ok_or_else(|| start("Missing Qwen3.5 table initializer owner"))?;
        launch_threads(
            initializer,
            stream,
            (grid, 1, 1),
            THREADS as u32,
            0,
            &mut [
                arg(&mut inv),
                arg(&mut cos),
                arg(&mut sin),
                arg(&mut positions),
                arg(&mut pairs),
            ],
        )
        .map_err(start)?;
        if let Err(error) = stream.synchronize() {
            pending.retain_after_failed_drain();
            return Err(start(error));
        }
        pending
            .tables
            .take()
            .ok_or_else(|| start("Missing initialized Qwen3.5 rotary tables"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "CPU-only comparison with actual Native constructor frequencies; set INFINITY_OXIDE_TEST_QWEN35_INVERSE"]
    fn actual_native_inverse_frequency_bits() {
        let path = std::env::var_os("INFINITY_OXIDE_TEST_QWEN35_INVERSE")
            .expect("actual Native frequency fixture path");
        let bytes = std::fs::read(path).expect("actual Native frequency fixture");
        let actual = inverse_frequencies(10_000_000.0, 64).unwrap();
        assert_eq!(bytes.len(), actual.len() * 4);
        for (index, (chunk, value)) in bytes.chunks_exact(4).zip(actual).enumerate() {
            let expected = u32::from_le_bytes(chunk.try_into().unwrap());
            assert_eq!(
                value.to_bits(),
                expected,
                "Native inverse frequency {index}"
            );
        }
    }

    #[test]
    fn native_frequency_theta_one_is_exactly_one() {
        let inverse = inverse_frequencies(1.0, ROTARY).unwrap();
        assert_eq!(inverse.len(), PAIRS);
        assert!(inverse.iter().all(|&value| value == 1.0));
    }

    #[test]
    fn native_exact_power_of_two_frequencies_preserve_order() {
        let inverse = inverse_frequencies(16.0, ROTARY).unwrap();
        assert_eq!(inverse[0].to_bits(), 1.0f32.to_bits());
        assert_eq!(inverse[8].to_bits(), 0.5f32.to_bits());
        assert_eq!(inverse[16].to_bits(), 0.25f32.to_bits());
        assert_eq!(inverse[24].to_bits(), 0.125f32.to_bits());
    }

    #[test]
    fn inverse_frequencies_are_finite_and_ordered() {
        for theta in [1.0f32, 10_000.0, 1e30] {
            let inverse = inverse_frequencies(theta, ROTARY).unwrap();
            assert_eq!(inverse.len(), PAIRS);
            assert!(inverse.iter().all(|value| value.is_finite()));
            assert!(inverse.windows(2).all(|pair| pair[0] >= pair[1]));
        }
        let below_one = inverse_frequencies(0.5, ROTARY).unwrap();
        assert_eq!(below_one[0], 1.0);
        assert!(below_one.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn invalid_theta_is_rejected() {
        for theta in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -1.0, 1e-44] {
            assert!(inverse_frequencies(theta, ROTARY).is_err(), "theta {theta}");
        }
    }

    #[test]
    fn rotary_must_be_native_partial_64() {
        for rotary in [0, 1, 32, 63, 65, 128] {
            assert!(inverse_frequencies(1.0, rotary).is_err(), "rotary {rotary}");
        }
    }

    #[test]
    fn positions_must_be_positive_and_fit_u32() {
        assert!(tables_plan(1.0, ROTARY, 0, usize::MAX).is_err());
        if let Ok(too_many) = usize::try_from(u64::from(u32::MAX) + 1) {
            assert!(tables_plan(1.0, ROTARY, too_many, usize::MAX).is_err());
        }
        #[cfg(target_pointer_width = "64")]
        assert!(tables_plan(1.0, ROTARY, u32::MAX as usize, usize::MAX).is_ok());
    }

    #[test]
    fn required_bytes_includes_temporary_inverse() {
        let expected =
            7 * PAIRS * 2 * core::mem::size_of::<f32>() + PAIRS * core::mem::size_of::<f32>();
        assert_eq!(required_bytes(1.0, ROTARY, 7).unwrap(), expected);
        assert_eq!(expected, 1_920);
    }

    #[test]
    fn byte_budget_is_inclusive_and_bounded_to_128_mib() {
        let limit = required_bytes(1.0, ROTARY, 1).unwrap();
        assert!(tables_plan(1.0, ROTARY, 1, limit).is_ok());
        assert!(tables_plan(1.0, ROTARY, 1, limit - 1).is_err());
        assert!(tables_plan(1.0, ROTARY, 524_287, MAX_TABLE_BYTES).is_ok());
        assert!(tables_plan(1.0, ROTARY, 524_288, MAX_TABLE_BYTES).is_err());
    }

    #[test]
    fn actual_32768_position_budget() {
        let expected = 32768usize * PAIRS * 2 * core::mem::size_of::<f32>()
            + PAIRS * core::mem::size_of::<f32>();
        assert_eq!(expected, 8_388_736);
        assert_eq!(required_bytes(1.0, ROTARY, 32768).unwrap(), expected);
        assert!(tables_plan(1.0, ROTARY, 32768, expected).is_ok());
        assert!(tables_plan(1.0, ROTARY, 32768, expected - 1).is_err());
    }

    #[test]
    fn launch_blocks_round_up_and_reject_invalid_cuda_grid() {
        assert_eq!(launch_blocks(1).unwrap(), 1);
        assert_eq!(launch_blocks(THREADS).unwrap(), 1);
        assert_eq!(launch_blocks(THREADS + 1).unwrap(), 2);
        assert_eq!(launch_blocks(32768 * PAIRS).unwrap(), 4096);
        assert_eq!(
            launch_blocks(i32::MAX as usize * THREADS).unwrap(),
            i32::MAX as u32
        );
        assert!(launch_blocks(i32::MAX as usize * THREADS + 1).is_err());
        assert!(launch_blocks(usize::MAX).is_err());
    }
}
