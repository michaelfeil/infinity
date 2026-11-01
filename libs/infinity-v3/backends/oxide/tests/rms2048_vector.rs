// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Same-input controls for the aligned width-2048 vector RMS exports.

use cuda_core::{launch_kernel, Context, DeviceBuffer, Function, Module, Stream};
use half::bf16;
use std::{
    ffi::c_void,
    path::{Path, PathBuf},
    sync::Arc,
};

const WIDTH: usize = 2048;
const GUARD: usize = 16;
const SENTINEL: u16 = 0x4242;

fn setup() -> (Arc<Module>, Arc<Stream>) {
    let root = std::env::var_os("INFINITY_OXIDE_KERNEL_MODULE")
        .filter(|path| !path.is_empty())
        .expect("set INFINITY_OXIDE_KERNEL_MODULE to the vector RMS bundle");
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
        .expect("load vector RMS bundle");
    (module, stream)
}

fn arg<T>(value: &mut T) -> *mut c_void {
    (value as *mut T).cast()
}

fn launch(function: &Function, stream: &Stream, rows: usize, args: &mut [*mut c_void]) {
    unsafe {
        launch_kernel(
            function.cu_function(),
            (
                u32::try_from(rows.div_ceil(4)).expect("fixture grid fits u32"),
                1,
                1,
            ),
            (128, 1, 1),
            0,
            stream.cu_stream(),
            args,
        )
    }
    .expect("launch exact width-2048 normalization");
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

fn output_values(stream: &Stream, buffer: &DeviceBuffer<bf16>, elements: usize) -> Vec<u16> {
    let values = download(stream, buffer);
    assert!(values[..GUARD].iter().all(|v| v.to_bits() == SENTINEL));
    assert!(values[GUARD + elements..]
        .iter()
        .all(|v| v.to_bits() == SENTINEL));
    values[GUARD..GUARD + elements]
        .iter()
        .map(|value| value.to_bits())
        .collect()
}

fn raw_u16(path: &Path) -> Vec<u16> {
    std::fs::read(path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
        .chunks_exact(2)
        .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
        .collect()
}

fn raw_u32(path: &Path) -> u32 {
    let bytes =
        std::fs::read(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

#[derive(Clone, Copy)]
struct FixtureKind {
    add: bool,
}

fn upload_fixture(stream: &Stream, values: &[u16]) -> DeviceBuffer<bf16> {
    upload(
        stream,
        &values
            .iter()
            .copied()
            .map(bf16::from_bits)
            .collect::<Vec<_>>(),
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_plain(
    function: &Function,
    stream: &Stream,
    input: &DeviceBuffer<bf16>,
    weight: &DeviceBuffer<bf16>,
    out: &DeviceBuffer<bf16>,
    rows: usize,
    width: usize,
    eps: f32,
) {
    let mut input_ptr = input.cu_deviceptr();
    let mut weight_ptr = weight.cu_deviceptr();
    let mut out_ptr = out.cu_deviceptr() + (2 * GUARD) as u64;
    let mut count = u32::try_from(rows).expect("fixture rows fit u32");
    let mut columns = u32::try_from(width).expect("fixture width fits u32");
    let mut epsilon = eps;
    launch(
        function,
        stream,
        rows,
        &mut [
            arg(&mut input_ptr),
            arg(&mut weight_ptr),
            arg(&mut out_ptr),
            arg(&mut count),
            arg(&mut columns),
            arg(&mut epsilon),
        ],
    );
}

#[allow(clippy::too_many_arguments)]
fn launch_add(
    function: &Function,
    stream: &Stream,
    input: &DeviceBuffer<bf16>,
    residual: &DeviceBuffer<bf16>,
    weight: &DeviceBuffer<bf16>,
    out: &DeviceBuffer<bf16>,
    summed: &DeviceBuffer<bf16>,
    rows: usize,
    width: usize,
    eps: f32,
) {
    let mut input_ptr = input.cu_deviceptr();
    let mut residual_ptr = residual.cu_deviceptr();
    let mut weight_ptr = weight.cu_deviceptr();
    let mut out_ptr = out.cu_deviceptr() + (2 * GUARD) as u64;
    let mut summed_ptr = summed.cu_deviceptr() + (2 * GUARD) as u64;
    let mut count = u32::try_from(rows).expect("fixture rows fit u32");
    let mut columns = u32::try_from(width).expect("fixture width fits u32");
    let mut epsilon = eps;
    launch(
        function,
        stream,
        rows,
        &mut [
            arg(&mut input_ptr),
            arg(&mut residual_ptr),
            arg(&mut weight_ptr),
            arg(&mut out_ptr),
            arg(&mut summed_ptr),
            arg(&mut count),
            arg(&mut columns),
            arg(&mut epsilon),
        ],
    );
}

fn fixture_labels(directory: &Path) -> Vec<String> {
    let mut labels = std::fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("read {}: {error}", directory.display()))
        .filter_map(Result::ok)
        .filter_map(|entry| entry.file_name().into_string().ok())
        .filter_map(|name| name.strip_suffix(".rows.bin").map(str::to_owned))
        .collect::<Vec<_>>();
    labels.sort();
    assert_eq!(
        labels.len(),
        16,
        "expected all 16 native raw RMS fixtures in {}: {labels:?}",
        directory.display()
    );
    labels
}

#[test]
#[ignore = "requires INFINITY_OXIDE_KERNEL_MODULE and INFINITY_OXIDE_RMS2048_RAW_DIR on a matching CUDA device"]
fn vector_rms_matches_all_native_raw_fixtures_and_scalar_export() {
    let (module, stream) = setup();
    let raw = PathBuf::from(
        std::env::var_os("INFINITY_OXIDE_RMS2048_RAW_DIR")
            .expect("set INFINITY_OXIDE_RMS2048_RAW_DIR to the native raw fixture directory"),
    );
    let scalar_plain = module
        .load_function("infinity_rms_norm_2048_exact")
        .expect("bundle must export the retained scalar plain RMS");
    let scalar_add = module
        .load_function("infinity_add_rms_norm_2048_exact")
        .expect("bundle must export the retained scalar add RMS");
    let vector_plain = module
        .load_function("infinity_rms_norm_2048_vector")
        .expect("bundle must export the vector plain RMS");
    let vector_add = module
        .load_function("infinity_add_rms_norm_2048_vector")
        .expect("bundle must export the vector add RMS");

    for label in fixture_labels(&raw) {
        let kind = FixtureKind {
            add: label.starts_with("add-"),
        };
        let rows = raw_u32(&raw.join(format!("{label}.rows.bin"))) as usize;
        let width = raw_u32(&raw.join(format!("{label}.width.bin"))) as usize;
        let eps_bytes = std::fs::read(raw.join(format!("{label}.eps.bin")))
            .unwrap_or_else(|error| panic!("read {label} eps: {error}"));
        let eps = f32::from_le_bytes([eps_bytes[0], eps_bytes[1], eps_bytes[2], eps_bytes[3]]);
        let input = upload_fixture(&stream, &raw_u16(&raw.join(format!("{label}.input.bf16"))));
        let weight = upload_fixture(&stream, &raw_u16(&raw.join(format!("{label}.weight.bf16"))));
        let expected = raw_u16(&raw.join(format!("{label}.out.bf16")));
        let elements = rows * width;
        let scalar_out = guarded(&stream, elements);
        let vector_out = guarded(&stream, elements);
        if kind.add {
            let expected_summed = raw_u16(&raw.join(format!("{label}.summed.bf16")));
            let residual = upload_fixture(
                &stream,
                &raw_u16(&raw.join(format!("{label}.residual.bf16"))),
            );
            let scalar_summed = guarded(&stream, elements);
            let vector_summed = guarded(&stream, elements);
            launch_add(
                &scalar_add,
                &stream,
                &input,
                &residual,
                &weight,
                &scalar_out,
                &scalar_summed,
                rows,
                width,
                eps,
            );
            launch_add(
                &vector_add,
                &stream,
                &input,
                &residual,
                &weight,
                &vector_out,
                &vector_summed,
                rows,
                width,
                eps,
            );
            let scalar = output_values(&stream, &scalar_out, elements);
            let vector = output_values(&stream, &vector_out, elements);
            let scalar_sum = output_values(&stream, &scalar_summed, elements);
            let vector_sum = output_values(&stream, &vector_summed, elements);
            assert_eq!(scalar, expected, "{label}: scalar normalized");
            assert_eq!(vector, expected, "{label}: vector normalized");
            assert_eq!(scalar_sum, expected_summed, "{label}: scalar summed");
            assert_eq!(vector_sum, expected_summed, "{label}: vector summed");
            assert_eq!(vector, scalar, "{label}: vector must match scalar bits");
            assert_eq!(
                vector_sum, scalar_sum,
                "{label}: vector summed must match scalar bits"
            );
        } else {
            launch_plain(
                &scalar_plain,
                &stream,
                &input,
                &weight,
                &scalar_out,
                rows,
                width,
                eps,
            );
            launch_plain(
                &vector_plain,
                &stream,
                &input,
                &weight,
                &vector_out,
                rows,
                width,
                eps,
            );
            let scalar = output_values(&stream, &scalar_out, elements);
            let vector = output_values(&stream, &vector_out, elements);
            assert_eq!(scalar, expected, "{label}: scalar normalized");
            assert_eq!(vector, expected, "{label}: vector normalized");
            assert_eq!(vector, scalar, "{label}: vector must match scalar bits");
        }
    }
}

#[test]
#[ignore = "requires INFINITY_OXIDE_KERNEL_MODULE on a matching CUDA device"]
fn vector_rms_unaligned_two_byte_pointers_take_scalar_fallback() {
    let (module, stream) = setup();
    let scalar_plain = module
        .load_function("infinity_rms_norm_2048_exact")
        .expect("bundle must export the retained scalar plain RMS");
    let scalar_add = module
        .load_function("infinity_add_rms_norm_2048_exact")
        .expect("bundle must export the retained scalar add RMS");
    let vector_plain = module
        .load_function("infinity_rms_norm_2048_vector")
        .expect("bundle must export the vector plain RMS");
    let vector_add = module
        .load_function("infinity_add_rms_norm_2048_vector")
        .expect("bundle must export the vector add RMS");
    let rows = 5usize;
    let elements = rows * WIDTH;
    let input = (0..elements)
        .map(|index| bf16::from_f32(1.25 + (((index * 37) % 251) as f32 - 125.0) / 128.0))
        .collect::<Vec<_>>();
    let residual = (0..elements)
        .map(|index| bf16::from_f32((((index * 19) % 255) as f32 - 127.0) / 256.0))
        .collect::<Vec<_>>();
    let weight = (0..WIDTH)
        .map(|index| bf16::from_f32(0.5 + ((index * 11) % 23) as f32 / 16.0))
        .collect::<Vec<_>>();

    let plain_input = upload(&stream, &input);
    let plain_weight = upload(&stream, &weight);
    let plain_out = guarded(&stream, elements);
    launch_plain(
        &scalar_plain,
        &stream,
        &plain_input,
        &plain_weight,
        &plain_out,
        rows,
        WIDTH,
        1e-5,
    );
    let expected_plain = output_values(&stream, &plain_out, elements);

    let offset = 1 + GUARD;
    let mut unaligned_input_host = vec![bf16::from_bits(SENTINEL); offset + elements + GUARD];
    unaligned_input_host[offset..offset + elements].copy_from_slice(&input);
    let mut unaligned_weight_host = vec![bf16::from_bits(SENTINEL); offset + WIDTH + GUARD];
    unaligned_weight_host[offset..offset + WIDTH].copy_from_slice(&weight);
    let unaligned_input = upload(&stream, &unaligned_input_host);
    let unaligned_weight = upload(&stream, &unaligned_weight_host);
    let unaligned_out = upload(
        &stream,
        &vec![bf16::from_bits(SENTINEL); offset + elements + GUARD],
    );
    let mut input_ptr = unaligned_input.cu_deviceptr() + offset as u64 * 2;
    let mut weight_ptr = unaligned_weight.cu_deviceptr() + offset as u64 * 2;
    let mut out_ptr = unaligned_out.cu_deviceptr() + offset as u64 * 2;
    let mut count = rows as u32;
    let mut width = WIDTH as u32;
    let mut eps = 1e-5f32;
    launch(
        &vector_plain,
        &stream,
        rows,
        &mut [
            arg(&mut input_ptr),
            arg(&mut weight_ptr),
            arg(&mut out_ptr),
            arg(&mut count),
            arg(&mut width),
            arg(&mut eps),
        ],
    );
    let values = download(&stream, &unaligned_out);
    assert!(values[..offset].iter().all(|v| v.to_bits() == SENTINEL));
    assert!(values[offset + elements..]
        .iter()
        .all(|v| v.to_bits() == SENTINEL));
    assert_eq!(
        values[offset..offset + elements]
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected_plain,
        "unaligned plain vector export must use the scalar fallback"
    );

    let add_input = upload(&stream, &input);
    let add_residual = upload(&stream, &residual);
    let add_weight = upload(&stream, &weight);
    let add_out = guarded(&stream, elements);
    let add_summed = guarded(&stream, elements);
    launch_add(
        &scalar_add,
        &stream,
        &add_input,
        &add_residual,
        &add_weight,
        &add_out,
        &add_summed,
        rows,
        WIDTH,
        1e-5,
    );
    let expected_out = output_values(&stream, &add_out, elements);
    let expected_summed = output_values(&stream, &add_summed, elements);

    let mut unaligned_residual_host = vec![bf16::from_bits(SENTINEL); offset + elements + GUARD];
    unaligned_residual_host[offset..offset + elements].copy_from_slice(&residual);
    let unaligned_input = upload(&stream, &unaligned_input_host);
    let unaligned_residual = upload(&stream, &unaligned_residual_host);
    let unaligned_weight = upload(&stream, &unaligned_weight_host);
    let unaligned_out = upload(
        &stream,
        &vec![bf16::from_bits(SENTINEL); offset + elements + GUARD],
    );
    let unaligned_summed = upload(
        &stream,
        &vec![bf16::from_bits(SENTINEL); offset + elements + GUARD],
    );
    let mut input_ptr = unaligned_input.cu_deviceptr() + offset as u64 * 2;
    let mut residual_ptr = unaligned_residual.cu_deviceptr() + offset as u64 * 2;
    let mut weight_ptr = unaligned_weight.cu_deviceptr() + offset as u64 * 2;
    let mut out_ptr = unaligned_out.cu_deviceptr() + offset as u64 * 2;
    let mut summed_ptr = unaligned_summed.cu_deviceptr() + offset as u64 * 2;
    let mut count = rows as u32;
    let mut width = WIDTH as u32;
    let mut eps = 1e-5f32;
    launch(
        &vector_add,
        &stream,
        rows,
        &mut [
            arg(&mut input_ptr),
            arg(&mut residual_ptr),
            arg(&mut weight_ptr),
            arg(&mut out_ptr),
            arg(&mut summed_ptr),
            arg(&mut count),
            arg(&mut width),
            arg(&mut eps),
        ],
    );
    let out_values = download(&stream, &unaligned_out);
    let summed_values = download(&stream, &unaligned_summed);
    assert!(out_values[..offset].iter().all(|v| v.to_bits() == SENTINEL));
    assert!(out_values[offset + elements..]
        .iter()
        .all(|v| v.to_bits() == SENTINEL));
    assert!(summed_values[..offset]
        .iter()
        .all(|v| v.to_bits() == SENTINEL));
    assert!(summed_values[offset + elements..]
        .iter()
        .all(|v| v.to_bits() == SENTINEL));
    assert_eq!(
        out_values[offset..offset + elements]
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected_out,
        "unaligned add vector export must use the scalar fallback"
    );
    assert_eq!(
        summed_values[offset..offset + elements]
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected_summed,
        "unaligned add summed stream must use the scalar fallback"
    );
}

#[test]
#[ignore = "requires INFINITY_OXIDE_KERNEL_MODULE on a matching CUDA device"]
fn vector_rms_in_place_aliases_match_scalar_for_aligned_and_unaligned_partial_ctas() {
    let (module, stream) = setup();
    let scalar_plain = module
        .load_function("infinity_rms_norm_2048_exact")
        .expect("bundle must export the retained scalar plain RMS");
    let scalar_add = module
        .load_function("infinity_add_rms_norm_2048_exact")
        .expect("bundle must export the retained scalar add RMS");
    let vector_plain = module
        .load_function("infinity_rms_norm_2048_vector")
        .expect("bundle must export the vector plain RMS");
    let vector_add = module
        .load_function("infinity_add_rms_norm_2048_vector")
        .expect("bundle must export the vector add RMS");
    let rows = 5usize;
    let elements = rows * WIDTH;
    let input = (0..elements)
        .map(|index| bf16::from_f32(1.25 + (((index * 37) % 251) as f32 - 125.0) / 128.0))
        .collect::<Vec<_>>();
    let residual = (0..elements)
        .map(|index| bf16::from_f32((((index * 19) % 255) as f32 - 127.0) / 256.0))
        .collect::<Vec<_>>();
    let weight = (0..WIDTH)
        .map(|index| bf16::from_f32(0.5 + ((index * 11) % 23) as f32 / 16.0))
        .collect::<Vec<_>>();
    let weight_device = upload(&stream, &weight);

    // Obtain both references from untouched inputs and disjoint scalar outputs
    // before any aliasing launch can overwrite either source allocation.
    let reference_input = upload(&stream, &input);
    let reference_residual = upload(&stream, &residual);
    let plain_out = guarded(&stream, elements);
    launch_plain(
        &scalar_plain,
        &stream,
        &reference_input,
        &weight_device,
        &plain_out,
        rows,
        WIDTH,
        1e-5,
    );
    let expected_plain = output_values(&stream, &plain_out, elements);
    let add_out = guarded(&stream, elements);
    let add_summed = guarded(&stream, elements);
    launch_add(
        &scalar_add,
        &stream,
        &reference_input,
        &reference_residual,
        &weight_device,
        &add_out,
        &add_summed,
        rows,
        WIDTH,
        1e-5,
    );
    let expected_out = output_values(&stream, &add_out, elements);
    let expected_summed = output_values(&stream, &add_summed, elements);

    for extra in [0usize, 1] {
        let offset = GUARD + extra;
        let guarded_source = |source: &[bf16]| {
            let mut values = vec![bf16::from_bits(SENTINEL); offset + elements + GUARD];
            values[offset..offset + elements].copy_from_slice(source);
            upload(&stream, &values)
        };
        let check_alias = |buffer: &DeviceBuffer<bf16>, expected: &[u16], label: &str| {
            let values = download(&stream, buffer);
            assert!(
                values[..offset].iter().all(|v| v.to_bits() == SENTINEL),
                "offset{offset} {label}: prefix guard"
            );
            assert!(
                values[offset + elements..]
                    .iter()
                    .all(|v| v.to_bits() == SENTINEL),
                "offset{offset} {label}: suffix guard"
            );
            assert_eq!(expected.len(), elements);
            for (index, (got, expected)) in values[offset..offset + elements]
                .iter()
                .zip(expected)
                .enumerate()
            {
                assert_eq!(
                    got.to_bits(),
                    *expected,
                    "offset{offset} {label}: element{index}"
                );
            }
        };

        let plain_input = guarded_source(&input);
        let mut input_ptr = plain_input.cu_deviceptr() + (offset * 2) as u64;
        assert_eq!(input_ptr % 16, (extra * 2) as u64);
        let mut out_ptr = input_ptr;
        let mut weight_ptr = weight_device.cu_deviceptr();
        let mut count = rows as u32;
        let mut width = WIDTH as u32;
        let mut eps = 1e-5f32;
        // SAFETY: exact-base input/output aliasing is intentional; each lane
        // caches its complete source before overwriting those same columns.
        launch(
            &vector_plain,
            &stream,
            rows,
            &mut [
                arg(&mut input_ptr),
                arg(&mut weight_ptr),
                arg(&mut out_ptr),
                arg(&mut count),
                arg(&mut width),
                arg(&mut eps),
            ],
        );
        check_alias(&plain_input, &expected_plain, "plain out=input");

        for swapped in [false, true] {
            let add_input = guarded_source(&input);
            let add_residual = guarded_source(&residual);
            let mut input_ptr = add_input.cu_deviceptr() + (offset * 2) as u64;
            let mut residual_ptr = add_residual.cu_deviceptr() + (offset * 2) as u64;
            let (mut out_ptr, mut summed_ptr) = if swapped {
                (residual_ptr, input_ptr)
            } else {
                (input_ptr, residual_ptr)
            };
            assert_ne!(out_ptr, summed_ptr, "the two outputs must remain disjoint");
            let mut weight_ptr = weight_device.cu_deviceptr();
            let mut count = rows as u32;
            let mut width = WIDTH as u32;
            let mut eps = 1e-5f32;
            // Both sources are fully cached before either output is stored;
            // only exact-base aliases are exercised, never overlapping outputs.
            launch(
                &vector_add,
                &stream,
                rows,
                &mut [
                    arg(&mut input_ptr),
                    arg(&mut residual_ptr),
                    arg(&mut weight_ptr),
                    arg(&mut out_ptr),
                    arg(&mut summed_ptr),
                    arg(&mut count),
                    arg(&mut width),
                    arg(&mut eps),
                ],
            );
            let (normalized, summed) = if swapped {
                (&add_residual, &add_input)
            } else {
                (&add_input, &add_residual)
            };
            check_alias(normalized, &expected_out, "add normalized");
            check_alias(summed, &expected_summed, "add summed");
        }
    }
}

#[test]
#[ignore = "requires INFINITY_OXIDE_KERNEL_MODULE on a matching CUDA device"]
fn vector_rms_guards_rows_zero_and_wrong_width_without_touching_output() {
    let (module, stream) = setup();
    let input = upload(&stream, &[bf16::from_f32(1.0); WIDTH]);
    let weight = upload(&stream, &[bf16::from_f32(1.0); WIDTH]);
    let residual = upload(&stream, &[bf16::from_f32(1.0); WIDTH]);
    let plain = module
        .load_function("infinity_rms_norm_2048_vector")
        .expect("bundle must export the vector plain RMS");
    let add = module
        .load_function("infinity_add_rms_norm_2048_vector")
        .expect("bundle must export the vector add RMS");
    for (rows, width, wrong_width) in [(0usize, WIDTH, false), (1, WIDTH - 1, true)] {
        let out = guarded(&stream, WIDTH);
        let summed = guarded(&stream, WIDTH);
        let mut input_ptr = input.cu_deviceptr();
        let mut residual_ptr = residual.cu_deviceptr();
        let mut weight_ptr = weight.cu_deviceptr();
        let mut out_ptr = out.cu_deviceptr() + (2 * GUARD) as u64;
        let mut summed_ptr = summed.cu_deviceptr() + (2 * GUARD) as u64;
        let mut count = rows as u32;
        let mut columns = width as u32;
        let mut eps = 1e-5f32;
        let function = if wrong_width { &add } else { &plain };
        launch(
            function,
            &stream,
            rows.max(1),
            &mut if wrong_width {
                vec![
                    arg(&mut input_ptr),
                    arg(&mut residual_ptr),
                    arg(&mut weight_ptr),
                    arg(&mut out_ptr),
                    arg(&mut summed_ptr),
                    arg(&mut count),
                    arg(&mut columns),
                    arg(&mut eps),
                ]
            } else {
                vec![
                    arg(&mut input_ptr),
                    arg(&mut weight_ptr),
                    arg(&mut out_ptr),
                    arg(&mut count),
                    arg(&mut columns),
                    arg(&mut eps),
                ]
            },
        );
        let out_values = download(&stream, &out);
        assert!(out_values.iter().all(|value| value.to_bits() == SENTINEL));
        if wrong_width {
            let summed_values = download(&stream, &summed);
            assert!(summed_values
                .iter()
                .all(|value| value.to_bits() == SENTINEL));
        }
    }
}
