// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Controls for the retained and native-division paired rounded BF16 SiLU exports.

use cuda_core::{launch_kernel, Context, DeviceBuffer, Function, Module, Stream};
use half::bf16;
use std::{ffi::c_void, path::PathBuf, sync::Arc};

const GUARD: usize = 16;
const SENTINEL: u16 = 0x4242;
const NATIVE_EXPORT: &str = "infinity_silu_mul_bf16_pairs_native";
const PAIRED_EXPORTS: [&str; 2] = ["infinity_silu_mul_bf16_pairs", NATIVE_EXPORT];

fn paired_exports(module: &Arc<Module>) -> [(&'static str, Function); 2] {
    PAIRED_EXPORTS.map(|name| {
        (
            name,
            module
                .load_function(name)
                .unwrap_or_else(|error| panic!("bundle must export {name}: {error}")),
        )
    })
}

fn setup() -> (Arc<Module>, Arc<Stream>) {
    let root = std::env::var_os("INFINITY_OXIDE_KERNEL_MODULE")
        .filter(|path| !path.is_empty())
        .expect("set INFINITY_OXIDE_KERNEL_MODULE to the decoder-primitives kernel bundle");
    let context = Context::new(0).expect("create CUDA context on GPU0");
    let stream = context.new_stream().expect("create CUDA stream");
    let path = PathBuf::from(root);
    let path = if path.is_dir() {
        let (major, minor) = context
            .compute_capability()
            .expect("query GPU architecture");
        path.join(format!("sm_{major}{minor}"))
            .join("kernels.cubin")
    } else {
        path
    };
    let image = std::fs::read(&path)
        .unwrap_or_else(|error| panic!("read kernel bundle {}: {error}", path.display()));
    let module = context
        .load_module_from_image(&image)
        .expect("load decoder-primitives kernel bundle");
    (module, stream)
}

fn arg<T>(value: &mut T) -> *mut c_void {
    (value as *mut T).cast()
}

fn launch(function: &Function, stream: &Stream, blocks: usize, args: &mut [*mut c_void]) {
    unsafe {
        launch_kernel(
            function.cu_function(),
            (u32::try_from(blocks).expect("fixture grid fits u32"), 1, 1),
            (256, 1, 1),
            0,
            stream.cu_stream(),
            args,
        )
    }
    .expect("launch decoder primitive");
}

fn round(value: f32) -> f32 {
    bf16::from_f32(value).to_f32()
}

fn upload(stream: &Stream, values: &[bf16]) -> DeviceBuffer<bf16> {
    DeviceBuffer::from_host(stream, values).expect("upload BF16 fixture")
}

fn guarded(stream: &Stream, elements: usize) -> DeviceBuffer<bf16> {
    DeviceBuffer::from_host(
        stream,
        &vec![bf16::from_bits(SENTINEL); elements + 2 * GUARD],
    )
    .expect("allocate guarded output")
}

fn download(stream: &Stream, buffer: &DeviceBuffer<bf16>) -> Vec<bf16> {
    let mut values = vec![bf16::ZERO; buffer.len()];
    buffer
        .copy_to_host(stream, &mut values)
        .expect("download and drain stream");
    values
}

fn guard_at(values: &[bf16], start: usize, elements: usize) {
    assert!(values[..start].iter().all(|v| v.to_bits() == SENTINEL));
    assert!(values[start + elements..]
        .iter()
        .all(|v| v.to_bits() == SENTINEL));
}

fn dump_raw_case(export: &str, rows: usize, width: usize, input: &[bf16], output: &[bf16]) {
    let Some(directory) = std::env::var_os("INFINITY_OXIDE_SILU_PAIRS_RAW_DIR") else {
        return;
    };
    assert_eq!(input.len(), rows * 2 * width);
    assert_eq!(output.len(), rows * width);
    let directory = std::path::PathBuf::from(directory);
    // Keep the existing root-level fixture contract for the retained export.
    // The native division results must never overwrite those baseline outputs.
    let directory = if export == NATIVE_EXPORT {
        directory.join("native")
    } else {
        directory
    };
    std::fs::create_dir_all(&directory).expect("create native SiLU fixture directory");
    let label = format!("rows{rows}-width{width}");
    let bytes = |values: &[bf16]| {
        values
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect::<Vec<_>>()
    };
    for (suffix, data) in [
        ("rows.bin", (rows as u32).to_le_bytes().to_vec()),
        ("width.bin", (width as u32).to_le_bytes().to_vec()),
        ("input.bf16", bytes(input)),
        ("out.bf16", bytes(output)),
    ] {
        std::fs::write(directory.join(format!("{label}.{suffix}")), data)
            .expect("write native SiLU fixture");
    }
}

fn fixture(rows: usize, width: usize) -> Vec<bf16> {
    let gate_bits = [
        0xc2c8u16, 0xc1a0, 0xc000, 0xbf80, 0xbe80, 0xbc80, 0x8000, 0x0000, 0x0001, 0x8001, 0x3c80,
        0x3f80, 0x4000, 0x41a0, 0x42c8,
        // -87 through -89 span large-denominator rescaling and exp overflow.
        0xc2ae, 0xc2af, 0xc2b0, 0xc2b1, 0xc2b2,
    ];
    let up_bits = [
        0x42c8u16, 0xc2c8, 0x0001, 0x8001, 0x3f80, 0xbf80, 0x8000, 0x0000,
    ];
    let mut values = Vec::with_capacity(rows * 2 * width);
    for row in 0..rows {
        let mut gate = Vec::with_capacity(width);
        let mut up = Vec::with_capacity(width);
        for column in 0..width {
            let gate_value = if column < gate_bits.len() {
                bf16::from_bits(gate_bits[column])
            } else {
                bf16::from_f32(round(((column * 37 + row * 17) % 257) as f32 / 32.0 - 4.0))
            };
            let up_value = if (15..20).contains(&column) {
                // Unit up isolates the native division at these boundary gates.
                bf16::from_f32(1.0)
            } else if column < up_bits.len() {
                bf16::from_bits(up_bits[column])
            } else {
                bf16::from_f32(round(((column * 13 + row * 7) % 31) as f32 / 8.0 - 1.875))
            };
            gate.push(gate_value);
            up.push(up_value);
        }
        values.extend_from_slice(&gate);
        values.extend_from_slice(&up);
    }
    values
}

fn reference(input: &[bf16], width: usize, staged: bool) -> Vec<f32> {
    input
        .chunks_exact(2 * width)
        .flat_map(|row| {
            row[..width]
                .iter()
                .zip(&row[width..])
                .map(move |(gate, up)| {
                    let gate = gate.to_f32();
                    let up = up.to_f32();
                    let exponential = (-(gate as f64)).exp() as f32;
                    let denominator = if staged {
                        round(1.0 + round(exponential))
                    } else {
                        1.0 + exponential
                    };
                    round(round(gate / denominator) * up)
                })
        })
        .collect()
}

fn check_staging(got: &[f32], expected: &[f32], alternate: &[f32], label: &str) {
    assert_eq!(got.len(), expected.len());
    let mut error = 0.0f64;
    let mut distinction = 0.0f64;
    for (index, ((got, expected), alternate)) in got.iter().zip(expected).zip(alternate).enumerate()
    {
        assert!(got.is_finite(), "{label}: nonfinite value at {index}");
        assert!(
            (*got - *expected).abs() <= 0.02 * (1.0 + expected.abs()),
            "{label}: index{index}: got {got}, expected {expected}"
        );
        error += (*got as f64 - *expected as f64).abs();
        distinction += (*expected as f64 - *alternate as f64).abs();
    }
    assert!(
        distinction > 0.01,
        "{label}: fixture must distinguish staged from unstaged rounding"
    );
    assert!(
        error < 0.35 * distinction,
        "{label}: total error {error} must be much smaller than alternate distance {distinction}"
    );
}

fn run_pair(
    function: &Function,
    stream: &Stream,
    input: &DeviceBuffer<bf16>,
    out: &DeviceBuffer<bf16>,
    rows: usize,
    width: usize,
    extra_blocks: usize,
) {
    let pairs = rows * width.div_ceil(2);
    let mut input_ptr = input.cu_deviceptr();
    let mut out_ptr = out.cu_deviceptr() + (2 * GUARD) as u64;
    let mut count = rows as u32;
    let mut columns = width as u32;
    launch(
        function,
        stream,
        pairs.div_ceil(256) + extra_blocks,
        &mut [
            arg(&mut input_ptr),
            arg(&mut out_ptr),
            arg(&mut count),
            arg(&mut columns),
        ],
    );
}

#[test]
#[ignore = "requires INFINITY_OXIDE_KERNEL_MODULE with decoder exports and a matching CUDA device"]
fn paired_silu_matches_scalar_export_and_independent_staging() {
    let (module, stream) = setup();
    let scalar = module
        .load_function("infinity_silu_mul")
        .expect("bundle must export the retained scalar SiLU");
    let paired = paired_exports(&module);
    for (rows, width) in [(1usize, 17usize), (3, 31), (2, 5632), (2, 8960)] {
        let input = fixture(rows, width);
        let input_device = upload(&stream, &input);
        let expected = reference(&input, width, true);
        let alternate = reference(&input, width, false);
        let scalar_out = guarded(&stream, rows * width);
        let mut input_ptr = input_device.cu_deviceptr();
        let mut out_ptr = scalar_out.cu_deviceptr() + (2 * GUARD) as u64;
        let mut count = rows as u32;
        let mut columns = width as u32;
        launch(
            &scalar,
            &stream,
            (rows * width).div_ceil(256) + 1,
            &mut [
                arg(&mut input_ptr),
                arg(&mut out_ptr),
                arg(&mut count),
                arg(&mut columns),
            ],
        );
        let scalar_values = download(&stream, &scalar_out);
        guard_at(&scalar_values, GUARD, rows * width);
        let scalar_bits = scalar_values[GUARD..GUARD + rows * width]
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        for (name, function) in &paired {
            let paired_out = guarded(&stream, rows * width);
            run_pair(
                function,
                &stream,
                &input_device,
                &paired_out,
                rows,
                width,
                1,
            );
            let paired_values = download(&stream, &paired_out);
            guard_at(&paired_values, GUARD, rows * width);
            let paired_bits = paired_values[GUARD..GUARD + rows * width]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>();
            assert_eq!(
                paired_bits, scalar_bits,
                "{name} rows{rows} width{width}: paired export must match scalar bits"
            );
            dump_raw_case(
                name,
                rows,
                width,
                &input,
                &paired_values[GUARD..GUARD + rows * width],
            );
            let got = paired_values[GUARD..GUARD + rows * width]
                .iter()
                .map(|value| value.to_f32())
                .collect::<Vec<_>>();
            check_staging(
                &got,
                &expected,
                &alternate,
                &format!("{name} rows{rows} width{width}"),
            );
        }
    }
}

#[test]
#[ignore = "requires INFINITY_OXIDE_KERNEL_MODULE with decoder exports and a matching CUDA device"]
fn paired_silu_matches_scalar_for_every_finite_bf16_gate_with_unit_up() {
    let (module, stream) = setup();
    let scalar = module
        .load_function("infinity_silu_mul")
        .expect("bundle must export the retained scalar SiLU");
    let paired = paired_exports(&module);
    // Enumerate every finite encoding, including both signed zeros and all
    // subnormals. Unit up-values isolate activation/division rounding without
    // introducing output overflow or NaN payload comparisons.
    let gates = (0..=u16::MAX)
        .map(bf16::from_bits)
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    assert_eq!(gates.len(), 65_280, "all finite BF16 gate encodings");
    let rows = 1usize;
    let width = gates.len();
    let mut input = gates;
    input.extend(std::iter::repeat_n(bf16::from_f32(1.0), width));
    let input_device = upload(&stream, &input);
    let scalar_out = guarded(&stream, width);
    let mut input_ptr = input_device.cu_deviceptr();
    let mut out_ptr = scalar_out.cu_deviceptr() + (2 * GUARD) as u64;
    let mut count = rows as u32;
    let mut columns = width as u32;
    launch(
        &scalar,
        &stream,
        width.div_ceil(256) + 1,
        &mut [
            arg(&mut input_ptr),
            arg(&mut out_ptr),
            arg(&mut count),
            arg(&mut columns),
        ],
    );
    let scalar_values = download(&stream, &scalar_out);
    guard_at(&scalar_values, GUARD, width);
    for (name, function) in &paired {
        let paired_out = guarded(&stream, width);
        run_pair(
            function,
            &stream,
            &input_device,
            &paired_out,
            rows,
            width,
            1,
        );
        let paired_values = download(&stream, &paired_out);
        guard_at(&paired_values, GUARD, width);
        for (index, (scalar, paired)) in scalar_values[GUARD..GUARD + width]
            .iter()
            .zip(&paired_values[GUARD..GUARD + width])
            .enumerate()
        {
            assert!(
                scalar.is_finite() && paired.is_finite(),
                "{name}: finite gate {:#06x} must produce finite SiLU with unit up",
                input[index].to_bits()
            );
            assert_eq!(
                paired.to_bits(),
                scalar.to_bits(),
                "{name}: paired/scalar mismatch for gate {:#06x}",
                input[index].to_bits()
            );
        }
        dump_raw_case(
            name,
            rows,
            width,
            &input,
            &paired_values[GUARD..GUARD + width],
        );
    }
}

#[test]
#[ignore = "requires INFINITY_OXIDE_KERNEL_MODULE with decoder exports and a matching CUDA device"]
fn paired_silu_unaligned_odd_widths_fall_back_without_oob() {
    let (module, stream) = setup();
    let scalar = module
        .load_function("infinity_silu_mul")
        .expect("bundle must export the retained scalar SiLU");
    let paired = paired_exports(&module);
    let offset = 1 + GUARD;
    for width in [1usize, 2, 17, 18, 31] {
        let rows = 3usize;
        let elements = rows * width;
        let input = fixture(rows, width);
        let mut unaligned_input = vec![bf16::from_bits(SENTINEL); offset + input.len() + GUARD];
        unaligned_input[offset..offset + input.len()].copy_from_slice(&input);
        let input_device = upload(&stream, &unaligned_input);
        let scalar_out = upload(
            &stream,
            &vec![bf16::from_bits(SENTINEL); offset + elements + GUARD],
        );
        let mut input_ptr = input_device.cu_deviceptr() + offset as u64 * 2;
        let mut scalar_ptr = scalar_out.cu_deviceptr() + offset as u64 * 2;
        let mut count = rows as u32;
        let mut columns = width as u32;
        launch(
            &scalar,
            &stream,
            (rows * width).div_ceil(256) + 1,
            &mut [
                arg(&mut input_ptr),
                arg(&mut scalar_ptr),
                arg(&mut count),
                arg(&mut columns),
            ],
        );
        let scalar_values = download(&stream, &scalar_out);
        guard_at(&scalar_values, offset, elements);
        for (name, function) in &paired {
            let paired_out = upload(
                &stream,
                &vec![bf16::from_bits(SENTINEL); offset + elements + GUARD],
            );
            let mut input_ptr = input_device.cu_deviceptr() + offset as u64 * 2;
            let mut paired_ptr = paired_out.cu_deviceptr() + offset as u64 * 2;
            let pairs = rows * width.div_ceil(2);
            launch(
                function,
                &stream,
                pairs.div_ceil(256) + 1,
                &mut [
                    arg(&mut input_ptr),
                    arg(&mut paired_ptr),
                    arg(&mut count),
                    arg(&mut columns),
                ],
            );
            let paired_values = download(&stream, &paired_out);
            guard_at(&paired_values, offset, elements);
            assert_eq!(
                paired_values[offset..offset + elements]
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                scalar_values[offset..offset + elements]
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "{name} width{width}: unaligned paired export must match scalar fallback"
            );
        }
    }
}

#[test]
#[ignore = "requires INFINITY_OXIDE_KERNEL_MODULE with decoder exports and a matching CUDA device"]
fn paired_silu_guards_zero_rows_zero_width_and_partial_ctas() {
    let (module, stream) = setup();
    let paired = paired_exports(&module);
    let input = upload(&stream, &[bf16::from_f32(1.0); 64]);
    for (rows, width) in [(0usize, 31usize), (1, 0)] {
        let elements = 32usize;
        for (name, function) in &paired {
            let out = guarded(&stream, elements);
            let mut input_ptr = input.cu_deviceptr();
            let mut out_ptr = out.cu_deviceptr() + (2 * GUARD) as u64;
            let mut count = rows as u32;
            let mut columns = width as u32;
            launch(
                function,
                &stream,
                1,
                &mut [
                    arg(&mut input_ptr),
                    arg(&mut out_ptr),
                    arg(&mut count),
                    arg(&mut columns),
                ],
            );
            let values = download(&stream, &out);
            assert!(
                values.iter().all(|value| value.to_bits() == SENTINEL),
                "{name} rows{rows} width{width}: guards must stay sentinel"
            );
        }
    }

    // 258 pairs span a full CTA plus two live lanes in a second CTA. Extra
    // CTAs must remain idle, including the odd tail column of each row.
    let (rows, width) = (3usize, 171usize);
    let elements = rows * width;
    let input = fixture(rows, width);
    let input_device = upload(&stream, &input);
    let scalar = module
        .load_function("infinity_silu_mul")
        .expect("bundle must export the retained scalar SiLU");
    let scalar_out = guarded(&stream, elements);
    let mut input_ptr = input_device.cu_deviceptr();
    let mut out_ptr = scalar_out.cu_deviceptr() + (2 * GUARD) as u64;
    let mut count = rows as u32;
    let mut columns = width as u32;
    launch(
        &scalar,
        &stream,
        elements.div_ceil(256) + 2,
        &mut [
            arg(&mut input_ptr),
            arg(&mut out_ptr),
            arg(&mut count),
            arg(&mut columns),
        ],
    );
    let scalar_values = download(&stream, &scalar_out);
    guard_at(&scalar_values, GUARD, elements);
    for (name, function) in &paired {
        let out = guarded(&stream, elements);
        run_pair(function, &stream, &input_device, &out, rows, width, 2);
        let values = download(&stream, &out);
        guard_at(&values, GUARD, elements);
        assert_eq!(
            values[GUARD..GUARD + elements]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            scalar_values[GUARD..GUARD + elements]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "{name}: partial active CTA and excess blocks must match scalar output"
        );
    }
}
