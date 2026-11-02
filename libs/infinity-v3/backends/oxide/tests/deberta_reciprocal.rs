// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! The two materialized BF16 divisors permit exact fixed-reciprocal results
//! after the required BF16 RNE boundary. This is a CPU arithmetic regression;
//! GPU instructions and whole-model equivalence require separate qualification.

fn bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    if bits & 0x7fff_ffff > 0x7f80_0000 {
        return ((bits >> 16) as u16) | 0x0040;
    }
    ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16
}

#[test]
fn fixed_reciprocals_preserve_all_bf16_division_results() {
    for (divisor, reciprocal_bits) in [(13.875_f32, 0x3d93_9a86_u32), (11.3125, 0x3db5_09e7)] {
        let reciprocal = f32::from_bits(reciprocal_bits);
        assert_eq!((1.0 / divisor).to_bits(), reciprocal_bits);
        for pattern in 0..=u16::MAX {
            let input = std::hint::black_box(f32::from_bits(u32::from(pattern) << 16));
            let original = input / std::hint::black_box(divisor);
            let fused = input * std::hint::black_box(reciprocal);
            assert_eq!(
                bf16_rne(original),
                bf16_rne(fused),
                "BF16 pattern {pattern:#06x}, materialized divisor {divisor}"
            );
        }
    }
}
