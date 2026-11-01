// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Load-time base NeoX RoPE tables for the native decoder arithmetic recipe.
//!
//! This module implements unscaled RoPE and GTE's native NTK-scaled variant.
//! Other scaling recipes such as Llama3 and YaRN are not implemented here.
//! Native construction casts theta to F32, evaluates inverse frequencies in F32,
//! multiplies F32 positions by those frequencies, then casts sine/cosine to BF16.
//! Tables retain those BF16 values expanded to F32 for the device RoPE ABI.
//!
//! Host libm and CUDA libdevice can differ near rounding boundaries. Matching
//! these stages is not a claim of bitwise CUDA phase-table equivalence.

use half::bf16;
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) enum RotaryError {
    InvalidHeadDimension,
    InvalidTheta,
    InvalidScale,
    InvalidCapacity,
    SizeOverflow,
    ByteBudgetExceeded { required: usize, limit: usize },
    AllocationFailed,
    PositionOutOfRange { position: u32, capacity: usize },
}

impl fmt::Display for RotaryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidHeadDimension => {
                formatter.write_str("RoPE head dimension must be positive, even and fit u32")
            }
            Self::InvalidTheta => formatter
                .write_str("RoPE theta must be finite, greater than one and representable in F32"),
            Self::InvalidScale => formatter.write_str(
                "RoPE NTK factor and scaled frequencies must be finite, positive and representable in F32",
            ),
            Self::InvalidCapacity => {
                formatter.write_str("RoPE position capacity must be positive and fit u32")
            }
            Self::SizeOverflow => formatter.write_str("RoPE table size overflow"),
            Self::ByteBudgetExceeded { required, limit } => write!(
                formatter,
                "RoPE construction requires {required} bytes, exceeding its {limit}-byte budget"
            ),
            Self::AllocationFailed => formatter.write_str("RoPE table allocation failed"),
            Self::PositionOutOfRange { position, capacity } => write!(
                formatter,
                "RoPE position {position} exceeds table capacity {capacity}"
            ),
        }
    }
}

impl std::error::Error for RotaryError {}

/// Immutable tables built once before publishing a model to requests.
/// Capacity is explicit and never grows during inference. The byte budget covers
/// both F32 table payloads and inverse-frequency scratch during construction;
/// allocator bookkeeping is separate. Device uploads need their own accounting.
#[derive(Debug)]
pub(crate) struct BaseRopeTables {
    head_dim: usize,
    positions: usize,
    table_bytes: usize,
    cosine: Vec<f32>,
    sine: Vec<f32>,
}

fn zeroed(elements: usize) -> Result<Vec<f32>, RotaryError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| RotaryError::AllocationFailed)?;
    values.resize(elements, 0.0);
    Ok(values)
}

impl BaseRopeTables {
    pub(crate) fn new(
        head_dim: usize,
        theta: f64,
        position_capacity: usize,
        byte_limit: usize,
    ) -> Result<Self, RotaryError> {
        if head_dim == 0 || !head_dim.is_multiple_of(2) || u32::try_from(head_dim).is_err() {
            return Err(RotaryError::InvalidHeadDimension);
        }
        let theta_f32 = theta as f32;
        if !theta.is_finite() || theta <= 1.0 || !theta_f32.is_finite() {
            return Err(RotaryError::InvalidTheta);
        }
        if position_capacity == 0 || u32::try_from(position_capacity).is_err() {
            return Err(RotaryError::InvalidCapacity);
        }
        let half = head_dim / 2;
        let elements = position_capacity
            .checked_mul(half)
            .ok_or(RotaryError::SizeOverflow)?;
        let table_bytes = elements
            .checked_mul(2)
            .and_then(|n| n.checked_mul(size_of::<f32>()))
            .ok_or(RotaryError::SizeOverflow)?;
        let required = half
            .checked_mul(size_of::<f32>())
            .and_then(|n| n.checked_add(table_bytes))
            .ok_or(RotaryError::SizeOverflow)?;
        if required > byte_limit {
            return Err(RotaryError::ByteBudgetExceeded {
                required,
                limit: byte_limit,
            });
        }

        let mut inverse = zeroed(half)?;
        for (index, frequency) in inverse.iter_mut().enumerate() {
            let numerator = index.checked_mul(2).ok_or(RotaryError::SizeOverflow)?;
            let exponent = numerator as f32 / head_dim as f32;
            *frequency = 1.0f32 / theta_f32.powf(exponent);
        }
        let mut cosine = zeroed(elements)?;
        let mut sine = zeroed(elements)?;
        for (position, (cosine_row, sine_row)) in cosine
            .chunks_exact_mut(half)
            .zip(sine.chunks_exact_mut(half))
            .enumerate()
        {
            for ((cosine_value, sine_value), frequency) in cosine_row
                .iter_mut()
                .zip(sine_row.iter_mut())
                .zip(inverse.iter())
            {
                let angle = position as f32 * frequency;
                *cosine_value = bf16::from_f32(angle.cos()).to_f32();
                *sine_value = bf16::from_f32(angle.sin()).to_f32();
            }
        }
        Ok(Self {
            head_dim,
            positions: position_capacity,
            table_bytes,
            cosine,
            sine,
        })
    }

    /// Build GTE's NTK-scaled rotary tables using the native F32 stage order.
    ///
    /// The native encoder computes `scaled_base = f32(theta) * f32(factor)`,
    /// `scale = factor^(2/dimension)`, unscaled inverse powers from
    /// `scaled_base`, then multiplies every frequency by the F32 reciprocal of
    /// that scale. Phases are F32 positions times those F32 frequencies before
    /// BF16 sine/cosine conversion, matching the unscaled table ABI.
    pub(crate) fn new_ntk(
        head_dim: usize,
        theta: f64,
        factor: f64,
        position_capacity: usize,
        byte_limit: usize,
    ) -> Result<Self, RotaryError> {
        if head_dim == 0 || !head_dim.is_multiple_of(2) || u32::try_from(head_dim).is_err() {
            return Err(RotaryError::InvalidHeadDimension);
        }
        let base = theta as f32;
        if !theta.is_finite() || theta <= 1.0 || !base.is_finite() || base <= 1.0 {
            return Err(RotaryError::InvalidTheta);
        }
        let factor_f32 = factor as f32;
        if !factor.is_finite()
            || factor <= 0.0
            || !factor_f32.is_finite()
            || !factor_f32.is_normal()
        {
            return Err(RotaryError::InvalidScale);
        }
        if position_capacity == 0 || u32::try_from(position_capacity).is_err() {
            return Err(RotaryError::InvalidCapacity);
        }
        let scaled_base = base * factor_f32;
        let scale = factor_f32.powf(2.0f32 / head_dim as f32);
        let multiplier = (1.0f64 / scale as f64) as f32;
        if !scaled_base.is_normal() || !scale.is_normal() || !multiplier.is_normal() {
            return Err(RotaryError::InvalidScale);
        }
        let half = head_dim / 2;
        let elements = position_capacity
            .checked_mul(half)
            .ok_or(RotaryError::SizeOverflow)?;
        let table_bytes = elements
            .checked_mul(2)
            .and_then(|n| n.checked_mul(size_of::<f32>()))
            .ok_or(RotaryError::SizeOverflow)?;
        let required = half
            .checked_mul(size_of::<f32>())
            .and_then(|n| n.checked_add(table_bytes))
            .ok_or(RotaryError::SizeOverflow)?;
        if required > byte_limit {
            return Err(RotaryError::ByteBudgetExceeded {
                required,
                limit: byte_limit,
            });
        }

        let mut inverse = zeroed(half)?;
        for (lane, frequency) in inverse.iter_mut().enumerate() {
            let numerator = lane.checked_mul(2).ok_or(RotaryError::SizeOverflow)?;
            let exponent = numerator as f32 / head_dim as f32;
            let power = scaled_base.powf(exponent);
            let unscaled = 1.0f32 / power;
            let scaled = unscaled * multiplier;
            if !power.is_normal() || !unscaled.is_normal() || !scaled.is_normal() {
                return Err(RotaryError::InvalidScale);
            }
            *frequency = scaled;
        }
        let mut cosine = zeroed(elements)?;
        let mut sine = zeroed(elements)?;
        for (position, (cosine_row, sine_row)) in cosine
            .chunks_exact_mut(half)
            .zip(sine.chunks_exact_mut(half))
            .enumerate()
        {
            for ((cosine_value, sine_value), frequency) in cosine_row
                .iter_mut()
                .zip(sine_row.iter_mut())
                .zip(inverse.iter())
            {
                let angle = position as f32 * frequency;
                if !angle.is_finite() {
                    return Err(RotaryError::InvalidScale);
                }
                *cosine_value = bf16::from_f32(angle.cos()).to_f32();
                *sine_value = bf16::from_f32(angle.sin()).to_f32();
            }
        }
        Ok(Self {
            head_dim,
            positions: position_capacity,
            table_bytes,
            cosine,
            sine,
        })
    }

    pub(crate) const fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub(crate) const fn position_capacity(&self) -> usize {
        self.positions
    }

    pub(crate) const fn table_stride(&self) -> usize {
        self.head_dim / 2
    }

    pub(crate) const fn table_bytes(&self) -> usize {
        self.table_bytes
    }

    pub(crate) fn cosine(&self) -> &[f32] {
        &self.cosine
    }

    pub(crate) fn sine(&self) -> &[f32] {
        &self.sine
    }

    /// Validate the supplied logical position IDs, not the token count or batch
    /// capacity. Repeated/nonmonotonic positions are valid. This allocates nothing.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn validate_positions(&self, positions: &[u32]) -> Result<(), RotaryError> {
        if let Some(position) = positions
            .iter()
            .find(|position| **position as usize >= self.positions)
        {
            return Err(RotaryError::PositionOutOfRange {
                position: *position,
                capacity: self.positions,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_power_of_two_phases_match_independent_f64_values() {
        let tables = BaseRopeTables::new(8, 16.0, 64, 4096).expect("known geometry");
        // 16^(-2*i/8) gives these exact binary inverse frequencies.
        let frequencies = [1.0f64, 0.5, 0.25, 0.125];
        for position in [0usize, 1, 7, 63] {
            for (lane, inverse) in frequencies.iter().enumerate() {
                let angle = position as f64 * inverse;
                let offset = position * frequencies.len() + lane;
                assert_eq!(
                    tables.cosine()[offset],
                    bf16::from_f64(angle.cos()).to_f32()
                );
                assert_eq!(tables.sine()[offset], bf16::from_f64(angle.sin()).to_f32());
            }
        }
        assert_eq!(tables.head_dim(), 8);
        assert_eq!(tables.position_capacity(), 64);
        assert_eq!(tables.table_stride(), 4);
        assert_eq!(tables.table_bytes(), 64 * 4 * 2 * 4);
    }

    #[test]
    fn gte_base_phases_follow_f32_stages_with_bf16_coefficients() {
        let tables = BaseRopeTables::new(128, 1_000_000.0, 1025, 1 << 20).expect("GTE geometry");
        for position in [0usize, 1, 31, 1024] {
            for lane in [0usize, 1, 17, 63] {
                let inverse = 1_000_000.0f64.powf(-((2 * lane) as f64) / 128.0);
                let angle = position as f64 * inverse;
                let offset = position * 64 + lane;
                assert!((tables.cosine()[offset] as f64 - angle.cos()).abs() <= 0.004);
                assert!((tables.sine()[offset] as f64 - angle.sin()).abs() <= 0.004);
            }
        }
        assert!(tables
            .cosine()
            .iter()
            .chain(tables.sine())
            .all(|value| value.is_finite() && value.to_bits() & 0xffff == 0));
        let first_cosine = tables.cosine().first().expect("nonempty cosine");
        let first_sine = tables.sine().first().expect("nonempty sine");
        assert_eq!(first_cosine.to_bits(), 1.0f32.to_bits());
        assert_eq!(first_sine.to_bits(), 0.0f32.to_bits());
        assert!(tables.sine().iter().any(|value| value.is_sign_negative()));
        // This phase distinguishes the native F32 stages from a seemingly more
        // accurate F64 table builder: their BF16 sine values are adjacent.
        let native_phase = f32::from_bits(0x4197_537e) as f64;
        let native_sine = bf16::from_f64(native_phase.sin());
        let all_f64 = bf16::from_f64((132.0 * 1_000_000.0f64.powf(-18.0 / 128.0)).sin());
        assert_ne!(native_sine.to_bits(), all_f64.to_bits());
        assert_eq!(tables.sine()[132 * 64 + 9], native_sine.to_f32());
    }

    #[test]
    fn gte_ntk_stages_match_reference_and_differ_from_unscaled_tables() {
        let dimension = 128usize;
        let theta = 10_000.0f64;
        let factor = 2.0f64;
        let capacity = 17usize;
        let tables = BaseRopeTables::new_ntk(dimension, theta, factor, capacity, 1 << 20)
            .expect("GTE NTK geometry");
        let base = theta as f32;
        let factor_f32 = factor as f32;
        let scaled_base = base * factor_f32;
        let scale = factor_f32.powf(2.0f32 / dimension as f32);
        let multiplier = (1.0f64 / scale as f64) as f32;
        for position in [0usize, 1, 15] {
            for lane in 0..dimension / 2 {
                let exponent = (2 * lane) as f32 / dimension as f32;
                let frequency = (1.0f32 / scaled_base.powf(exponent)) * multiplier;
                let angle = position as f32 * frequency;
                let offset = position * (dimension / 2) + lane;
                assert_eq!(
                    tables.cosine()[offset],
                    bf16::from_f32(angle.cos()).to_f32()
                );
                assert_eq!(tables.sine()[offset], bf16::from_f32(angle.sin()).to_f32());
            }
        }
        let unscaled =
            BaseRopeTables::new(dimension, theta, capacity, 1 << 20).expect("unscaled geometry");
        assert_ne!(tables.cosine(), unscaled.cosine());
        assert_ne!(tables.sine(), unscaled.sine());
    }

    #[test]
    fn rejects_invalid_gte_ntk_numbers_and_exhausted_budgets() {
        for factor in [
            0.0,
            -2.0,
            f64::NAN,
            f64::INFINITY,
            f64::MIN_POSITIVE,
            f64::MAX,
        ] {
            assert!(matches!(
                BaseRopeTables::new_ntk(8, 10_000.0, factor, 8, 1 << 20),
                Err(RotaryError::InvalidScale)
            ));
        }
        for theta in [0.0, 1.0, f64::NAN, f64::INFINITY, f64::MAX] {
            assert!(matches!(
                BaseRopeTables::new_ntk(8, theta, 2.0, 8, 1 << 20),
                Err(RotaryError::InvalidTheta)
            ));
        }
        let required = (10 * 4 * 2 + 4) * 4;
        assert!(matches!(
            BaseRopeTables::new_ntk(8, 10_000.0, 2.0, 10, required - 1),
            Err(RotaryError::ByteBudgetExceeded {
                required: 336,
                limit: 335
            })
        ));
        assert!(BaseRopeTables::new_ntk(8, 10_000.0, 2.0, 10, required).is_ok());
        if usize::BITS >= 64 {
            assert!(matches!(
                BaseRopeTables::new_ntk(
                    u32::MAX as usize - 1,
                    10_000.0,
                    2.0,
                    u32::MAX as usize,
                    usize::MAX
                ),
                Err(RotaryError::SizeOverflow)
            ));
        }
    }

    #[test]
    fn validates_actual_position_ids_without_reordering_or_allocation() {
        let tables = BaseRopeTables::new(8, 16.0, 8, 1024).expect("bounded table");
        let cosine = tables.cosine().as_ptr();
        assert_eq!(tables.validate_positions(&[7, 0, 7, 1]), Ok(()));
        assert_eq!(tables.validate_positions(&[]), Ok(()));
        assert_eq!(tables.cosine().as_ptr(), cosine);
        assert_eq!(
            tables.validate_positions(&[8]),
            Err(RotaryError::PositionOutOfRange {
                position: 8,
                capacity: 8
            })
        );
        assert_eq!(
            tables.validate_positions(&[u32::MAX]),
            Err(RotaryError::PositionOutOfRange {
                position: u32::MAX,
                capacity: 8
            })
        );
    }

    #[test]
    fn rejects_malformed_geometry_and_nonfinite_theta() {
        for dimension in [0usize, 1, 3, usize::MAX] {
            assert!(matches!(
                BaseRopeTables::new(dimension, 16.0, 8, 1024),
                Err(RotaryError::InvalidHeadDimension)
            ));
        }
        for theta in [0.0, 1.0, -2.0, f64::NAN, f64::INFINITY, f64::MAX] {
            assert!(matches!(
                BaseRopeTables::new(8, theta, 8, 1024),
                Err(RotaryError::InvalidTheta)
            ));
        }
        assert!(matches!(
            BaseRopeTables::new(8, 16.0, 0, 1024),
            Err(RotaryError::InvalidCapacity)
        ));
    }

    #[test]
    fn checks_peak_payload_budget_and_overflow_before_allocating() {
        let required = (10 * 4 * 2 + 4) * 4;
        assert!(matches!(
            BaseRopeTables::new(8, 16.0, 10, required - 1),
            Err(RotaryError::ByteBudgetExceeded {
                required: 336,
                limit: 335
            })
        ));
        let tables = BaseRopeTables::new(8, 16.0, 10, required).expect("exact payload budget");
        assert_eq!(tables.table_bytes(), 320);
        assert!(matches!(
            BaseRopeTables::new(8, 16.0, 10, 0),
            Err(RotaryError::ByteBudgetExceeded { .. })
        ));
        if usize::BITS >= 64 {
            assert!(matches!(
                BaseRopeTables::new(u32::MAX as usize - 1, 16.0, u32::MAX as usize, usize::MAX),
                Err(RotaryError::SizeOverflow)
            ));
            assert!(matches!(
                BaseRopeTables::new(8, 16.0, u32::MAX as usize + 1, usize::MAX),
                Err(RotaryError::InvalidCapacity)
            ));
        }
    }
}
