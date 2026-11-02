// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! BF16 bit helpers with one explicit rounding boundary.
//!
//! Finite values use round-to-nearest, ties-to-even. Nonfinite values keep
//! their classification; NaN payload bits are deliberately not promised.
//! Arithmetic helpers never decode a BF16 literal and re-round it as an F32
//! constant: only the values read from storage and the final written result
//! cross the BF16 rounding boundary.

/// One BF16 storage word.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Bf16(u16);

impl Bf16 {
    #[inline]
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    #[inline]
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    /// Widen without numerical conversion. The low 16 bits are zero.
    #[inline]
    pub fn to_f32(self) -> f32 {
        f32::from_bits(u32::from(self.0) << 16)
    }

    /// Round one F32 value to BF16 using the native RNE conversion.
    #[inline]
    pub fn from_f32(value: f32) -> Self {
        Self(bits_from_f32(value))
    }
}

/// Round one F32 value to its BF16 bit pattern.
#[inline]
pub fn bits_from_f32(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        // No payload contract. Force a quiet NaN rather than allowing the
        // finite rounding add below to cross a NaN/Inf boundary.
        return ((bits >> 16) as u16) | 0x0040;
    }
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7fff + lsb);
    (rounded >> 16) as u16
}

/// Widen one BF16 bit pattern to F32 without numerical conversion.
#[inline]
pub const fn f32_from_bits(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Add two BF16 values and round the model-dtype sum once. This is the explicit
/// separate-bias contract; it is not a claim about `at::linear` or a fused GEMM
/// epilogue.
#[inline]
pub fn add_model_dtype(left: Bf16, right: Bf16) -> Bf16 {
    Bf16::from_f32(left.to_f32() + right.to_f32())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finite(bits: u16) -> bool {
        let exponent = (bits >> 7) & 0xff;
        exponent != 0xff
    }

    #[test]
    fn widening_is_exact() {
        for bits in 0..=u16::MAX {
            let widened = f32_from_bits(bits);
            assert_eq!(widened.to_bits() >> 16, u32::from(bits));
            assert_eq!(widened.to_bits() & 0xffff, 0);
        }
    }

    #[test]
    fn finite_round_trip_is_identity() {
        for bits in 0..=u16::MAX {
            if finite(bits) {
                assert_eq!(bits_from_f32(f32_from_bits(bits)), bits, "{bits:04x}");
            }
        }
    }

    #[test]
    fn nonfinite_classification_survives() {
        for bits in [0x7f80u16, 0xff80, 0x7fc0, 0xffc0, 0x7f81, 0xff81] {
            let widened = f32_from_bits(bits);
            let rounded = f32_from_bits(bits_from_f32(widened));
            assert_eq!(widened.is_nan(), rounded.is_nan());
            assert_eq!(widened.is_infinite(), rounded.is_infinite());
            if widened.is_infinite() {
                assert_eq!(widened.is_sign_negative(), rounded.is_sign_negative());
            }
        }
    }

    #[test]
    fn signed_zero_and_rne_ties_are_preserved() {
        assert_eq!(bits_from_f32(0.0), 0x0000);
        assert_eq!(bits_from_f32(-0.0), 0x8000);
        let tie = f32::from_bits(0x3f80_8000);
        assert_eq!(bits_from_f32(tie), 0x3f80);
        let up = f32::from_bits(0x3f80_8001);
        assert_eq!(bits_from_f32(up), 0x3f81);
    }
}
