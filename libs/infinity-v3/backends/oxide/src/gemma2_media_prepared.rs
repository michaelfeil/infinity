// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Cold norm staging shared by the media tower executors.
//!
//! Norm scales are read as BF16 and widened to F32 with no +1 offset, matching
//! the Native Gemma4 tower math. Unlike the text executor, media does not build
//! rotary tables here: the axial tables are initialized on device.
#![allow(dead_code)]

use crate::model::WeightSource;
use cuda_core::{DeviceBuffer, Stream};
use infinity_v3_backend_core::BackendError;
use std::sync::Arc;

fn start(error: impl std::fmt::Display) -> BackendError {
    BackendError::Start(format!("Oxide gemma2 media norm: {error}"))
}

/// Expand a BF16 checkpoint norm scale to F32 with no offset.
pub(crate) fn norm_values(source: &WeightSource, width: usize) -> Result<Vec<f32>, BackendError> {
    if width == 0 || source.shape() != [width].as_slice() {
        return Err(start("norm source shape disagrees with width"));
    }
    if source.dtype() != safetensors::Dtype::BF16 {
        return Err(start(
            "media norm scales must be BF16 in the released checkpoint",
        ));
    }
    let mut bits = Vec::new();
    bits.try_reserve_exact(width)
        .map_err(|_| start("norm BF16 staging allocation failed"))?;
    bits.resize(width, 0u16);
    source
        .read_bf16(0, &mut bits)
        .map_err(|error| start(format!("norm source read failed: {error}")))?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(width)
        .map_err(|_| start("norm F32 allocation failed"))?;
    for bits in bits {
        let value = f32::from_bits(u32::from(bits) << 16);
        if !value.is_finite() {
            return Err(start("norm source contains nonfinite values"));
        }
        values.push(value);
    }
    Ok(values)
}

pub(crate) fn upload_norm(
    stream: &Arc<Stream>,
    source: &WeightSource,
    width: usize,
) -> Result<DeviceBuffer<f32>, BackendError> {
    let values = norm_values(source, width)?;
    DeviceBuffer::from_host(stream, &values).map_err(start)
}
