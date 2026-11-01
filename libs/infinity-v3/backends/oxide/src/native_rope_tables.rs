// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Prepared native BF16 base-RoPE tables; construction drains before publication.
use crate::{arg, blocks, launch, product, start, u32_size};
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
