// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Prepared native BF16 base-RoPE tables; construction drains before publication.
use crate::{
    arg, blocks, decoder_model::YarnRope, gemma2_math::RopePlan, launch, product, start,
    tensor::Bf16, u32_size,
};
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

#[allow(dead_code)]
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

fn decoder_plan(
    head_dim: usize,
    positions: usize,
    byte_limit: usize,
) -> Result<DecoderTablesPlan, BackendError> {
    if head_dim == 0 || !head_dim.is_multiple_of(2) || head_dim > 512 {
        return Err(start("Decoder rotary head_dim is outside 2..=512"));
    }
    let pairs = head_dim / 2;
    let plan = RopePlan::new(positions, pairs)
        .map_err(|error| start(format!("Decoder rotary plan: {error}")))?;
    let elements = plan.full_table_elements();
    let table_bytes = product(product(elements, 2)?, core::mem::size_of::<f32>())?;
    let inverse_bytes = product(pairs, core::mem::size_of::<f32>())?;
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

fn decoder_inverse_frequencies(head_dim: usize, theta: f64) -> Result<Vec<f32>, BackendError> {
    let base = theta as f32;
    if head_dim == 0
        || !head_dim.is_multiple_of(2)
        || !theta.is_finite()
        || theta <= 1.0
        || !base.is_finite()
    {
        return Err(start("Decoder rotary base must be finite and exceed one"));
    }
    let pairs = head_dim / 2;
    let mut result = Vec::new();
    result.try_reserve_exact(pairs).map_err(start)?;
    for index in 0..pairs {
        result.push(1.0f32 / base.powf((index * 2) as f32 / head_dim as f32));
    }
    Ok(result)
}

/// Shaped BF16 base tables for any even head dimension. The generic public
/// Gemma2 initializer receives the exact pair count, so one implementation
/// serves Pixtral d64, Mistral d96/d128 and wider native limits.
pub(crate) fn decoder_tables(
    head_dim: usize,
    theta: f64,
    positions: usize,
    byte_limit: usize,
    stream: &Arc<Stream>,
    initializer: Arc<Function>,
) -> Result<(DeviceBuffer<f32>, DeviceBuffer<f32>), BackendError> {
    let plan = decoder_plan(head_dim, positions, byte_limit)?;
    let pairs = u32::try_from(head_dim / 2).map_err(start)?;
    let geometry = plan
        .plan
        .geometry()
        .map_err(|error| start(format!("Decoder rotary grid: {error}")))?;
    let inverse = decoder_inverse_frequencies(head_dim, theta)?;
    let cosine = DeviceBuffer::<f32>::zeroed(stream, plan.elements).map_err(start)?;
    let sine = DeviceBuffer::<f32>::zeroed(stream, plan.elements).map_err(start)?;
    let inverse = DeviceBuffer::from_host(stream, &inverse).map_err(start)?;
    let mut pending = PendingTables {
        stream: stream.clone(),
        tables: Some(Tables {
            cosine,
            sine,
            stride: pairs,
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
    let mut pairs = pairs;
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

/// Native YaRN inverse frequencies and attention scale for any even head
/// dimension. F32 base inverse, F64 correction bounds and blend, F32 output.
pub(crate) fn yarn_inverse_frequencies(
    head_dim: usize,
    theta: f64,
    yarn: &YarnRope,
) -> Result<(Vec<f32>, f32), BackendError> {
    yarn.validate()?;
    let base = theta as f32;
    if head_dim == 0
        || !head_dim.is_multiple_of(4)
        || !theta.is_finite()
        || theta <= 1.0
        || !base.is_finite()
        || base <= 1.0
    {
        return Err(start("Invalid YaRN rotary geometry"));
    }
    let factor = yarn.factor;
    let original = yarn.original_max_position_embeddings;
    let fast = yarn.beta_fast;
    let slow = yarn.beta_slow;
    let correction = |r: f64| {
        head_dim as f64 * (original / (r * 2.0 * std::f64::consts::PI)).ln() / (2.0 * theta.ln())
    };
    let low = correction(fast).floor().max(0.0);
    let mut high = correction(slow).ceil().min((head_dim - 1) as f64);
    if low == high {
        high += 0.001;
    }
    if !(low.is_finite() && high.is_finite() && high > low) {
        return Err(start("Invalid YaRN correction bounds"));
    }
    let mscale = |m: f64| 1.0 + 0.1 * m * factor.ln();
    let attention_scale = if yarn.mscale != 0.0 && yarn.mscale_all_dim != 0.0 {
        mscale(yarn.mscale) / mscale(yarn.mscale_all_dim)
    } else {
        mscale(1.0)
    };
    let attention_scale = attention_scale as f32;
    if !attention_scale.is_finite() || attention_scale == 0.0 {
        return Err(start("Invalid YaRN attention scale"));
    }
    let mut inverse = Vec::new();
    inverse.try_reserve_exact(head_dim / 2).map_err(start)?;
    for index in 0..head_dim / 2 {
        let base_inverse = 1.0f32 / base.powf((index * 2) as f32 / head_dim as f32);
        let extrapolation = 1.0 - (((index as f64 - low) / (high - low)).clamp(0.0, 1.0));
        let scaled = base_inverse as f64 * ((1.0 - extrapolation) / factor + extrapolation);
        inverse.push(scaled as f32);
    }
    Ok((inverse, attention_scale))
}

/// Shaped YaRN-scaled tables; the attention scale multiplies cos/sin before the
/// BF16 round-trip, matching Native.
pub(crate) fn decoder_scaled_tables(
    head_dim: usize,
    theta: f64,
    positions: usize,
    byte_limit: usize,
    stream: &Arc<Stream>,
    initializer: Arc<Function>,
    yarn: &YarnRope,
) -> Result<(DeviceBuffer<f32>, DeviceBuffer<f32>), BackendError> {
    let plan = decoder_plan(head_dim, positions, byte_limit)?;
    let pairs = u32::try_from(head_dim / 2).map_err(start)?;
    let (inverse, attention_scale) = yarn_inverse_frequencies(head_dim, theta, yarn)?;
    if inverse.len() != head_dim / 2 {
        return Err(start("YaRN inverse frequency geometry mismatch"));
    }
    let cosine = DeviceBuffer::<f32>::zeroed(stream, plan.elements).map_err(start)?;
    let sine = DeviceBuffer::<f32>::zeroed(stream, plan.elements).map_err(start)?;
    let inverse = DeviceBuffer::from_host(stream, &inverse).map_err(start)?;
    let mut pending = PendingTables {
        stream: stream.clone(),
        tables: Some(Tables {
            cosine,
            sine,
            stride: pairs,
            _inverse: inverse,
            _initializer: initializer,
        }),
    };
    let table = pending
        .tables
        .as_ref()
        .ok_or_else(|| start("Missing scaled decoder table constructor owner"))?;
    let mut inv = table._inverse.cu_deviceptr();
    let mut cos = table.cosine.cu_deviceptr();
    let mut sin = table.sine.cu_deviceptr();
    let mut positions = plan.positions;
    let mut pairs = pairs;
    let mut scale = attention_scale;
    launch(
        &table._initializer,
        stream,
        blocks(plan.elements)?,
        0,
        &mut [
            arg(&mut inv),
            arg(&mut cos),
            arg(&mut sin),
            arg(&mut positions),
            arg(&mut pairs),
            arg(&mut scale),
        ],
    )
    .map_err(start)?;
    stream.synchronize().map_err(start)?;
    let tables = pending
        .tables
        .take()
        .ok_or_else(|| start("Missing initialized scaled decoder rotary tables"))?;
    Ok((tables.cosine, tables.sine))
}

/// Native llama_4_scaling query table, rounded to the model dtype.
pub(crate) fn llama4_query_scales(
    positions: usize,
    beta: f64,
    original_max_position_embeddings: f64,
) -> Result<Vec<Bf16>, BackendError> {
    if positions == 0
        || !beta.is_finite()
        || !original_max_position_embeddings.is_finite()
        || original_max_position_embeddings <= 0.0
    {
        return Err(start("Invalid llama_4_scaling query table geometry"));
    }
    let mut result = Vec::new();
    result.try_reserve_exact(positions).map_err(start)?;
    let beta = beta as f32;
    let original = original_max_position_embeddings as f32;
    for position in 0..positions {
        let quotient = (position as f32 / original).floor();
        let value = 1.0f32 + beta * quotient.ln_1p();
        result.push(Bf16::from_bits(half::bf16::from_f32(value).to_bits()));
    }
    Ok(result)
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

    #[test]
    fn shaped_decoder_plan_covers_mistral_head_dims() {
        for (head_dim, pairs) in [(64, 32), (96, 48), (128, 64), (256, 128)] {
            let plan = super::decoder_plan(head_dim, 32, usize::MAX).unwrap();
            assert_eq!(plan.elements, 32 * pairs);
            assert_eq!(plan.plan.full_table_elements(), 32 * pairs);
            let inverse = super::decoder_inverse_frequencies(head_dim, 1_000_000.0).unwrap();
            assert_eq!(inverse.len(), pairs);
            assert!(inverse
                .iter()
                .all(|value| value.is_finite() && *value > 0.0));
        }
        assert!(super::decoder_plan(95, 4, usize::MAX).is_err());
        let budget_256 = 262144usize * 128 * 2 * 4 + 128 * 4;
        assert!(super::decoder_plan(256, 262144, budget_256).is_ok());
        assert!(super::decoder_plan(256, 262144, budget_256 - 1).is_err());
    }

    #[test]
    fn yarn_inverse_frequencies_are_shaped_and_preserve_native_conventions() {
        use crate::decoder_model::YarnRope;
        let yarn = YarnRope {
            factor: 16.0,
            original_max_position_embeddings: 16384.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            mscale: 1.0,
            mscale_all_dim: 1.0,
            llama_4_scaling_beta: 0.1,
        };
        for head_dim in [96usize, 128, 256] {
            let (inverse, scale) =
                super::yarn_inverse_frequencies(head_dim, 1_000_000.0, &yarn).unwrap();
            assert_eq!(inverse.len(), head_dim / 2);
            assert!(inverse
                .iter()
                .all(|value| value.is_finite() && *value > 0.0));
            assert!(scale.is_finite() && scale > 0.0);
        }
        assert!(super::yarn_inverse_frequencies(95, 1_000_000.0, &yarn).is_err());
        assert_eq!(super::llama4_query_scales(4, 0.1, 2.0).unwrap().len(), 4);
    }
}
