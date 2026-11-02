// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Prepared native BF16 base-RoPE tables; construction drains before publication.
use crate::{arg, blocks, gemma2_math::RopePlan, launch, product, start, u32_size};
use cuda_core::{DeviceBuffer, Function, Stream};
use infinity_v3_backend_core::BackendError;
use std::sync::Arc;

pub(crate) fn inverse_frequencies(theta: f64) -> Result<Vec<f32>, BackendError> {
    let base = theta as f32;
    if !theta.is_finite() || theta <= 1.0 || !base.is_finite() || base <= 1.0 {
        return Err(start("Modern rotary base must be finite and exceed one"));
    }
    let mut result = Vec::new();
    result.try_reserve_exact(32).map_err(start)?;
    for index in 0..32 {
        result.push(1.0f32 / base.powf((index * 2) as f32 / 64.0));
    }
    Ok(result)
}
pub(crate) struct Tables {
    pub(crate) cosine: DeviceBuffer<f32>,
    pub(crate) sine: DeviceBuffer<f32>,
    pub(crate) stride: u32,
    _inverse: DeviceBuffer<f32>,
    _initializer: Arc<Function>,
}
struct PendingTables {
    stream: Arc<Stream>,
    tables: Option<Tables>,
}
impl Drop for PendingTables {
    fn drop(&mut self) {
        if self.tables.is_some() && self.stream.synchronize().is_err() {
            if let Some(tables) = self.tables.take() {
                std::mem::forget(tables);
            }
        }
    }
}
impl Tables {
    pub(crate) fn new(
        theta: f64,
        positions: usize,
        stream: &Arc<Stream>,
        initializer: Arc<Function>,
    ) -> Result<Self, BackendError> {
        if positions == 0 || positions > 8192 {
            return Err(start("Modern rotary capacity is outside1..8192"));
        }
        let inverse = inverse_frequencies(theta)?;
        let elements = product(positions, 32)?;
        let cosine = DeviceBuffer::<f32>::zeroed(stream, elements).map_err(start)?;
        let sine = DeviceBuffer::<f32>::zeroed(stream, elements).map_err(start)?;
        let inverse = DeviceBuffer::from_host(stream, &inverse).map_err(start)?;
        let mut pending = PendingTables {
            stream: stream.clone(),
            tables: Some(Self {
                cosine,
                sine,
                stride: 32,
                _inverse: inverse,
                _initializer: initializer,
            }),
        };
        let table = pending
            .tables
            .as_ref()
            .ok_or_else(|| start("Missing Modern table constructor owner"))?;
        let mut inv = table._inverse.cu_deviceptr();
        let mut cos = table.cosine.cu_deviceptr();
        let mut sin = table.sine.cu_deviceptr();
        let mut positions = u32_size(positions)?;
        launch(
            &table._initializer,
            stream,
            blocks(elements)?,
            0,
            &mut [
                arg(&mut inv),
                arg(&mut cos),
                arg(&mut sin),
                arg(&mut positions),
            ],
        )
        .map_err(start)?;
        stream.synchronize().map_err(start)?;
        pending
            .tables
            .take()
            .ok_or_else(|| start("Missing initialized Modern rotary tables"))
    }
}

struct DecoderTablesPlan {
    plan: RopePlan,
    positions: u32,
    elements: usize,
}

fn decoder_128_plan(
    theta: f64,
    positions: usize,
    byte_limit: usize,
) -> Result<DecoderTablesPlan, BackendError> {
    let base = theta as f32;
    if !theta.is_finite() || theta <= 1.0 || !base.is_finite() {
        return Err(start("Decoder rotary base must be finite and exceed one"));
    }
    let plan = RopePlan::new(positions, 64)
        .map_err(|error| start(format!("Decoder rotary plan: {error}")))?;
    let elements = plan.full_table_elements();
    let table_bytes = product(product(elements, 2)?, core::mem::size_of::<f32>())?;
    let inverse_bytes = product(64, core::mem::size_of::<f32>())?;
    let total_bytes = table_bytes
        .checked_add(inverse_bytes)
        .ok_or_else(|| start("Decoder rotary byte budget overflowed usize"))?;
    if total_bytes > byte_limit {
        return Err(start(
            "Decoder rotary tables exceed the configured byte budget",
        ));
    }
    Ok(DecoderTablesPlan {
        plan,
        positions: plan.positions(),
        elements,
    })
}

fn decoder_128_inverse_frequencies(theta: f64) -> Result<Vec<f32>, BackendError> {
    let base = theta as f32;
    if !theta.is_finite() || theta <= 1.0 || !base.is_finite() {
        return Err(start("Decoder rotary base must be finite and exceed one"));
    }
    let mut result = Vec::new();
    result.try_reserve_exact(64).map_err(start)?;
    for index in 0..64 {
        result.push(1.0f32 / base.powf((index * 2) as f32 / 128.0));
    }
    Ok(result)
}

pub(crate) fn decoder_128_tables(
    theta: f64,
    positions: usize,
    byte_limit: usize,
    stream: &Arc<Stream>,
    initializer: Arc<Function>,
) -> Result<(DeviceBuffer<f32>, DeviceBuffer<f32>), BackendError> {
    let plan = decoder_128_plan(theta, positions, byte_limit)?;
    let geometry = plan
        .plan
        .geometry()
        .map_err(|error| start(format!("Decoder rotary grid: {error}")))?;
    let inverse = decoder_128_inverse_frequencies(theta)?;
    let cosine = DeviceBuffer::<f32>::zeroed(stream, plan.elements).map_err(start)?;
    let sine = DeviceBuffer::<f32>::zeroed(stream, plan.elements).map_err(start)?;
    let inverse = DeviceBuffer::from_host(stream, &inverse).map_err(start)?;
    let mut pending = PendingTables {
        stream: stream.clone(),
        tables: Some(Tables {
            cosine,
            sine,
            stride: 64,
            _inverse: inverse,
            _initializer: initializer,
        }),
    };
    let table = pending
        .tables
        .as_ref()
        .ok_or_else(|| start("Missing decoder table constructor owner"))?;
    let mut inv = table._inverse.cu_deviceptr();
    let mut cos = table.cosine.cu_deviceptr();
    let mut sin = table.sine.cu_deviceptr();
    let mut positions = plan.positions;
    let mut pairs = 64u32;
    launch(
        &table._initializer,
        stream,
        (geometry.grid()[0], geometry.grid()[1], geometry.grid()[2]),
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
    stream.synchronize().map_err(start)?;
    let tables = pending
        .tables
        .take()
        .ok_or_else(|| start("Missing initialized decoder rotary tables"))?;
    Ok((tables.cosine, tables.sine))
}

#[cfg(test)]
mod decoder_128_tests {
    use super::{decoder_128_inverse_frequencies, decoder_128_plan};

    #[test]
    fn decoder_128_plan_rejects_zero_overflow_and_invalid_theta() {
        assert!(decoder_128_plan(10000.0, 0, usize::MAX).is_err());
        assert!(decoder_128_plan(10000.0, usize::MAX, usize::MAX).is_err());
        for theta in [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            1.0,
            0.5,
            -2.0,
            1.0e40,
        ] {
            assert!(decoder_128_plan(theta, 4, usize::MAX).is_err());
        }
    }

    #[test]
    fn decoder_128_budget_is_inclusive_and_capacity_is_not_modern_head64() {
        let plan = decoder_128_plan(10000.0, 4, usize::MAX).unwrap();
        assert_eq!(plan.elements, 4 * 64);
        assert_eq!(
            plan.plan.full_table_elements(),
            4 * 64,
            "one cosine/sine pair per position"
        );
        let total = 4 * 64 * 2 * core::mem::size_of::<f32>() + 64 * core::mem::size_of::<f32>();
        assert!(decoder_128_plan(10000.0, 4, total).is_ok());
        assert!(decoder_128_plan(10000.0, 4, total - 1).is_err());
        assert!(decoder_128_plan(10000.0, 32768, 128 * 1024 * 1024).is_ok());
        let inverse = decoder_128_inverse_frequencies(10000.0).unwrap();
        assert_eq!(inverse.len(), 64);
        assert!(inverse
            .iter()
            .all(|value| value.is_finite() && *value > 0.0));
    }
}
