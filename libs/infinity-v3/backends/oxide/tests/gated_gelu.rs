// SPDX-License-Identifier: Apache-2.0
//
// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Host-side CPU reference for the rounded BF16 gated GELU kernel.
//!
//! This module is intended to be included under `#[cfg(test)]` so that the
//! `half` dependency is only needed for host unit tests.

use half::bf16;

#[inline(always)]
fn rounded(value: f32) -> f32 {
    bf16::from_f32(value).to_f32()
}

#[inline(always)]
fn rounded_bits(value: f32) -> u16 {
    bf16::from_f32(value).to_bits()
}

#[inline(always)]
fn bf16_const(value: f32) -> f32 {
    bf16::from_f32(value).to_f32()
}

/// Reproduce the device recipe for packed `[up(width), gate(width)]` rows.
pub fn gated_gelu_up_first_reference(input: &[u16], rows: usize, width: usize) -> Vec<u16> {
    assert!(width > 0, "reference width must be nonzero");
    assert_eq!(input.len(), rows * width * 2);
    let mut output = vec![0u16; rows * width];
    let alpha_scale = bf16_const(0.044715);
    let argument_scale = bf16_const(0.797_884_6);
    for row in 0..rows {
        for column in 0..width {
            let base = row * width * 2 + column;
            let up = bf16::from_bits(input[base]).to_f32();
            let gate = bf16::from_bits(input[base + width]).to_f32();

            let square = rounded(gate * gate);
            let cube = rounded(square * gate);
            let alpha = rounded(alpha_scale.mul_add(cube, gate));
            let argument = rounded(argument_scale * alpha);
            let tangent = rounded(argument.tanh());
            let half_gate = rounded(0.5 * gate);
            let plus = rounded(1.0 + tangent);
            let active = rounded(half_gate * plus);
            output[row * width + column] = rounded_bits(active * up);
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(value: f32) -> u16 {
        rounded_bits(value)
    }

    fn one_row(up: &[f32], gate: &[f32]) -> Vec<u16> {
        assert_eq!(up.len(), gate.len());
        let mut input = Vec::with_capacity(up.len() * 2);
        input.extend(up.iter().copied().map(bits));
        input.extend(gate.iter().copied().map(bits));
        input
    }

    fn unfused_output(up_bits: u16, gate_bits: u16) -> u16 {
        let up = bf16::from_bits(up_bits).to_f32();
        let gate = bf16::from_bits(gate_bits).to_f32();
        let square = rounded(gate * gate);
        let cube = rounded(square * gate);
        let alpha = rounded(rounded(bf16_const(0.044715) * cube) + gate);
        let argument = rounded(bf16_const(0.797_884_6) * alpha);
        let tangent = rounded(argument.tanh());
        let half_gate = rounded(0.5 * gate);
        let plus = rounded(1.0 + tangent);
        let active = rounded(half_gate * plus);
        rounded_bits(active * up)
    }

    fn high_width_input(rows: usize, width: usize) -> Vec<u16> {
        let mut input = vec![0u16; rows * width * 2];
        for row in 0..rows {
            for column in 0..width {
                let up = (((row * 17 + column * 5) % 31) as f32) / 8.0 - 1.0;
                let gate = (((row * 13 + column * 7) % 37) as f32) / 8.0 - 2.0;
                let base = row * width * 2 + column;
                input[base] = bits(up);
                input[base + width] = bits(gate);
            }
        }
        input
    }

    #[test]
    fn gate_is_read_from_the_second_packed_half() {
        let up = [1.0, -2.0, 0.5, 3.0, -0.75, 2.5, -1.5, 0.25];
        let gate = [0.0, 0.25, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0];
        let output = gated_gelu_up_first_reference(&one_row(&up, &gate), 1, 8);
        assert_eq!(
            output.as_slice(),
            &[0x0000, 0xbe99, 0xbd9e, 0x4022, 0x3df3, 0x409c, 0x3d90, 0x3f80]
        );

        let swapped = gated_gelu_up_first_reference(&one_row(&gate, &up), 1, 8);
        assert_ne!(output, swapped);
    }

    #[test]
    fn fused_alpha_is_not_rounded_multiply_then_add() {
        let up = bits(1.0);
        let gate = bits(1.6484375);
        let output = gated_gelu_up_first_reference(&one_row(&[1.0; 8], &[1.6484375; 8]), 1, 8);
        assert_eq!(output.as_slice(), &[0x3fc9; 8]);

        let unfused = unfused_output(up, gate);
        assert_eq!(unfused, 0x3fc8);
        assert_ne!(output[0], unfused);
    }

    #[test]
    fn disjoint_rows_and_device_tail_match_fixed_bits() {
        const ROWS: usize = 3;
        const WIDTH: usize = 264;
        let input = high_width_input(ROWS, WIDTH);
        let output = gated_gelu_up_first_reference(&input, ROWS, WIDTH);
        assert_eq!(output.len(), ROWS * WIDTH);

        let expected = [
            (0, 0x3d40),
            (255, 0x3dac),
            (256, 0x0000),
            (263, 0x3f5e),
            (264, 0xbe19),
            (519, 0x3f70),
            (520, 0x405e),
            (527, 0xbe90),
            (528, 0xbf32),
            (767, 0x4055),
            (768, 0xbe5f),
            (791, 0x3d8d),
        ];
        for (index, expected_bits) in expected {
            assert_eq!(output[index], expected_bits, "index {index}");
        }
    }

    #[test]
    fn finite_edge_inputs_match_fixed_bits() {
        let up = [0x3f80u16; 8];
        let gate = [
            0x7f7f, 0xff7f, 0x5f80, 0xdf80, 0x0001, 0x8001, 0x0000, 0x8000,
        ];
        let mut input = Vec::with_capacity(16);
        input.extend(up);
        input.extend(gate);

        let output = gated_gelu_up_first_reference(&input, 1, 8);
        assert_eq!(
            output.as_slice(),
            &[0x7f7f, 0x8000, 0x5f80, 0x8000, 0x0000, 0x8000, 0x0000, 0x8000]
        );
        for &value in &output {
            assert!(bf16::from_bits(value).to_f32().is_finite());
        }
    }
}

#[test]
#[ignore = "requires the GTE kernel bundle and INFINITY_OXIDE_GATED_GELU_REFERENCE from the native CUDA pairs oracle"]
fn gpu_gated_gelu_up_first_preserves_native_stages_and_row_bounds() {
    use cuda_core::{launch_kernel, Context, DeviceBuffer};
    use std::{ffi::c_void, path::PathBuf};

    fn arg<T>(value: &mut T) -> *mut c_void {
        (value as *mut T).cast()
    }
    #[derive(serde::Deserialize)]
    struct Oracle {
        schema_version: u32,
        cuda_runtime_version: u32,
        device_capability: [i32; 2],
        cases: Vec<Case>,
    }
    #[derive(serde::Deserialize)]
    struct Case {
        rows: usize,
        width: usize,
        input_bits: Vec<u16>,
        expected_bits: Vec<u16>,
    }
    let path = std::env::var_os("INFINITY_OXIDE_GATED_GELU_REFERENCE").expect(
        "run native-gated-reference on the owned CUDA GPU to generate the exact pair oracle",
    );
    let oracle: Oracle = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    assert_eq!(oracle.schema_version, 1);
    assert_eq!(oracle.cuda_runtime_version, 13010);
    assert_eq!(
        oracle
            .cases
            .iter()
            .map(|case| (case.rows, case.width))
            .collect::<Vec<_>>(),
        [(1, 8), (3, 264), (5, 3072)]
    );
    let context = Context::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let (major, minor) = context.compute_capability().unwrap();
    assert_eq!(oracle.device_capability, [major, minor]);
    let root = PathBuf::from(std::env::var_os("INFINITY_OXIDE_KERNEL_MODULE").unwrap());
    let image = if root.is_dir() {
        let (major, minor) = context.compute_capability().unwrap();
        root.join(format!("sm_{major}{minor}"))
            .join("kernels.cubin")
    } else {
        root
    };
    let module = context
        .load_module_from_image(&std::fs::read(image).unwrap())
        .unwrap();
    let kernel = module
        .load_function("infinity_gated_gelu_up_first")
        .unwrap();
    for case in &oracle.cases {
        let (rows, width) = (case.rows, case.width);
        let input = &case.input_bits;
        let expected = &case.expected_bits;
        assert_eq!(input.len(), rows * width * 2);
        assert_eq!(expected.len(), rows * width);
        let device_input = DeviceBuffer::from_host(&stream, input).unwrap();
        const GUARD: usize = 16;
        const SENTINEL: u16 = 0x4242;
        let device_output =
            DeviceBuffer::from_host(&stream, &vec![SENTINEL; expected.len() + 2 * GUARD]).unwrap();
        let mut source = device_input.cu_deviceptr();
        let mut destination = device_output.cu_deviceptr() + (GUARD * 2) as u64;
        let mut rows_arg = rows as u32;
        let mut width_arg = width as u32;
        // One extra CTA exercises the device guard without touching either sentinel.
        unsafe {
            launch_kernel(
                kernel.cu_function(),
                ((expected.len().div_ceil(256) + 1) as u32, 1, 1),
                (256, 1, 1),
                0,
                stream.cu_stream(),
                &mut [
                    arg(&mut source),
                    arg(&mut destination),
                    arg(&mut rows_arg),
                    arg(&mut width_arg),
                ],
            )
        }
        .unwrap();
        let mut actual = vec![0u16; device_output.len()];
        device_output.copy_to_host(&stream, &mut actual).unwrap();
        assert!(actual[..GUARD].iter().all(|value| *value == SENTINEL));
        assert!(actual[GUARD + expected.len()..]
            .iter()
            .all(|value| *value == SENTINEL));
        assert_eq!(
            &actual[GUARD..GUARD + expected.len()],
            expected.as_slice(),
            "rows={rows}, width={width}"
        );
    }
}
