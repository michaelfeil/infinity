// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Numerical controls for the Dense FP32 conversion and Tanh kernels.

use cuda_core::{launch_kernel, Context, DeviceBuffer, Function, Module, Stream};
use half::bf16;
use serde::Deserialize;
use std::{env, ffi::c_void, path::PathBuf, sync::Arc};

const GUARD: usize = 16;
const SENTINEL_U16: u16 = 0x4242;
const SENTINEL_F32: f32 = f32::from_bits(0x4242_4242);
const F32_GOLDEN_ENV: &str = "INFINITY_OXIDE_DENSE_F32_GOLDEN";
const TANH_GOLDEN_ENV: &str = "INFINITY_OXIDE_DENSE_TANH_GOLDEN";

#[derive(Deserialize)]
struct Golden {
    values_f32: Option<Vec<f32>>,
    values_bf16_bits: Option<Vec<u16>>,
    f32_to_bf16_bits: Option<Vec<u16>>,
    tanh_bf16_bits: Option<Vec<u16>>,
}

fn setup() -> (Arc<Module>, Arc<Stream>) {
    let root = env::var_os("INFINITY_OXIDE_KERNEL_MODULE")
        .filter(|path| !path.is_empty())
        .expect("set INFINITY_OXIDE_KERNEL_MODULE to a Dense kernel bundle");
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
        .expect("load Dense kernel bundle");
    (module, stream)
}

fn arg<T>(value: &mut T) -> *mut c_void {
    (value as *mut T).cast()
}

fn launch(function: &Function, stream: &Stream, blocks: usize, args: &mut [*mut c_void]) {
    // SAFETY: callers provide the exact exported argument ABI, retain all
    // allocations through the blocking download, and use the module context.
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
    .expect("launch Dense kernel");
}

fn f32_to_bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        return ((bits >> 16) as u16) | 0x0040;
    }
    let lsb = (bits >> 16) & 1;
    (bits.wrapping_add(0x7fff + lsb) >> 16) as u16
}

fn bf16_tanh_reference(bits: u16) -> u16 {
    f32_to_bf16_rne(bf16::from_bits(bits).to_f32().tanh())
}

fn ordered_bf16(bits: u16) -> u32 {
    if bits & 0x8000 != 0 {
        (!bits) as u32 & 0xffff
    } else {
        bits as u32 | 0x8000
    }
}

fn bf16_distance(first: u16, second: u16) -> u32 {
    ordered_bf16(first).abs_diff(ordered_bf16(second))
}

fn assert_bf16_exact_or_nan(got: u16, expected: u16, label: &str, index: usize) {
    let got_value = bf16::from_bits(got);
    let expected_value = bf16::from_bits(expected);
    if expected_value.is_nan() {
        assert!(
            got_value.is_nan(),
            "{label}: index {index}: expected NaN, got {got:#06x}"
        );
    } else {
        assert_eq!(
            got, expected,
            "{label}: index {index}: expected {expected:#06x}, got {got:#06x}"
        );
    }
}

fn assert_bf16_tanh_bound(got: u16, expected: u16, input: u16, label: &str, index: usize) {
    let got_value = bf16::from_bits(got);
    let expected_value = bf16::from_bits(expected);
    if expected_value.is_nan() {
        assert!(
            got_value.is_nan(),
            "{label}: index {index}: input {input:#06x} must stay NaN, got {got:#06x}"
        );
        return;
    }
    if expected == 0x0000 || expected == 0x8000 || expected == 0x7f80 || expected == 0xff80 {
        assert_eq!(
            got, expected,
            "{label}: index {index}: input {input:#06x} must preserve signed zero or saturation"
        );
        return;
    }
    assert!(
        bf16_distance(got, expected) <= 1,
        "{label}: index {index}: input {input:#06x}, expected {expected:#06x}, got {got:#06x}"
    );
}

fn guarded_f32(stream: &Stream, values: &[f32]) -> DeviceBuffer<f32> {
    let mut storage = vec![SENTINEL_F32; values.len() + 2 * GUARD];
    storage[GUARD..GUARD + values.len()].copy_from_slice(values);
    DeviceBuffer::from_host(stream, &storage).expect("upload guarded FP32 storage")
}

fn guarded_u16(stream: &Stream, values: &[u16]) -> DeviceBuffer<u16> {
    let mut storage = vec![SENTINEL_U16; values.len() + 2 * GUARD];
    storage[GUARD..GUARD + values.len()].copy_from_slice(values);
    DeviceBuffer::from_host(stream, &storage).expect("upload guarded BF16 storage")
}

fn download_f32(stream: &Stream, buffer: &DeviceBuffer<f32>, elements: usize) -> Vec<f32> {
    let mut storage = vec![SENTINEL_F32; buffer.len()];
    buffer
        .copy_to_host(stream, &mut storage)
        .expect("download and drain FP32 storage");
    assert!(storage[..GUARD]
        .iter()
        .all(|value| value.to_bits() == SENTINEL_F32.to_bits()));
    assert!(storage[GUARD + elements..]
        .iter()
        .all(|value| value.to_bits() == SENTINEL_F32.to_bits()));
    storage[GUARD..GUARD + elements].to_vec()
}

fn download_u16(stream: &Stream, buffer: &DeviceBuffer<u16>, elements: usize) -> Vec<u16> {
    let mut storage = vec![SENTINEL_U16; buffer.len()];
    buffer
        .copy_to_host(stream, &mut storage)
        .expect("download and drain BF16 storage");
    assert!(storage[..GUARD].iter().all(|&value| value == SENTINEL_U16));
    assert!(storage[GUARD + elements..]
        .iter()
        .all(|&value| value == SENTINEL_U16));
    storage[GUARD..GUARD + elements].to_vec()
}

fn run_f32_to_bf16(
    function: &Function,
    stream: &Stream,
    values: &[f32],
    extra_blocks: usize,
) -> Vec<u16> {
    let input = guarded_f32(stream, values);
    let output = guarded_u16(stream, &[]);
    let output = if values.is_empty() {
        output
    } else {
        guarded_u16(stream, &vec![SENTINEL_U16; values.len()])
    };
    let mut input_ptr = input.cu_deviceptr() + (4 * GUARD) as u64;
    let mut output_ptr = output.cu_deviceptr() + (2 * GUARD) as u64;
    let mut count = u32::try_from(values.len()).expect("fixture count fits u32");
    let blocks = if values.is_empty() {
        1
    } else {
        values.len().div_ceil(256) + extra_blocks
    };
    launch(
        function,
        stream,
        blocks,
        &mut [arg(&mut input_ptr), arg(&mut output_ptr), arg(&mut count)],
    );
    let converted = download_u16(stream, &output, values.len());
    let unchanged = download_f32(stream, &input, values.len());
    assert_eq!(
        unchanged
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "FP32 source must remain unchanged"
    );
    converted
}

fn run_tanh(
    function: &Function,
    stream: &Stream,
    input_bits: &[u16],
    extra_blocks: usize,
) -> Vec<u16> {
    let input = guarded_u16(stream, input_bits);
    let mut input_ptr = input.cu_deviceptr() + (2 * GUARD) as u64;
    let mut count = u32::try_from(input_bits.len()).expect("fixture count fits u32");
    let blocks = if input_bits.is_empty() {
        1
    } else {
        input_bits.len().div_ceil(256) + extra_blocks
    };
    launch(
        function,
        stream,
        blocks,
        &mut [arg(&mut input_ptr), arg(&mut count)],
    );
    download_u16(stream, &input, input_bits.len())
}

fn f32_fixture() -> Vec<f32> {
    let mut values = vec![
        0.0,
        -0.0,
        f32::from_bits(0x0000_0001),
        f32::from_bits(0x0000_8001),
        f32::from_bits(0x007f_ffff),
        f32::from_bits(0x0080_0000),
        f32::from_bits(0x3f80_0000),
        f32::from_bits(0xbf80_0000),
        f32::MAX,
        f32::MIN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::from_bits(0x7fc0_0001),
    ];
    for base in [
        0x0000u16, 0x0001, 0x007f, 0x0080, 0x3f7f, 0x3f80, 0x3f81, 0x7f7e, 0x7f7f, 0x8000, 0x8001,
        0x807f, 0xbf80, 0xbf81, 0xff7f,
    ] {
        let upper = (base as u32) << 16;
        values.push(f32::from_bits(upper));
        values.push(f32::from_bits(upper | 0x7fff));
        values.push(f32::from_bits(upper | 0x8000));
        values.push(f32::from_bits(upper | 0x8001));
    }
    let mut state = 0x1234_5678u32;
    while values.len() < 1536 {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let bits = state & 0x7fff_ffff;
        if bits < 0x7f80_0000 {
            values.push(f32::from_bits(bits));
        }
    }
    values
}

fn bf16_fixture() -> Vec<u16> {
    let mut values = vec![
        0x0000u16, 0x8000, 0x0001, 0x8001, 0x007f, 0x807f, 0x0080, 0x8080, 0x3f80, 0xbf80, 0x3f81,
        0xbf81, 0x4000, 0xc000, 0x7f7f, 0xff7f, 0x7f80, 0xff80, 0x7fc0, 0xffc0, 0x7fff, 0xffff,
    ];
    values.extend((0..=u16::MAX).filter(|&bits| bf16::from_bits(bits).is_finite()));
    values
}

fn load_golden(path: &std::ffi::OsStr) -> Golden {
    let bytes = std::fs::read(PathBuf::from(path)).expect("read native Torch golden JSON");
    serde_json::from_slice(&bytes).expect("parse native Torch golden JSON")
}

#[test]
fn cpu_oracle_rounds_f32_to_bf16_round_to_nearest_even() {
    let cases = [
        (0x3f80_0000u32, 0x3f80u16),
        (0x3f80_8000, 0x3f80),
        (0x3f81_8000, 0x3f82),
        (0x3f81_7fff, 0x3f81),
        (0x3f81_8001, 0x3f82),
        (0xbf80_8000, 0xbf80),
        (0xbf81_8000, 0xbf82),
        (0x0000_8000, 0x0000),
        (0x0001_8000, 0x0002),
        (0x8000_8000, 0x8000),
        (0x8001_8000, 0x8002),
        (0x0000_0001, 0x0000),
        (0x7f7f_ffff, 0x7f80),
        (0x7f7f_0000, 0x7f7f),
        (0x7f80_0000, 0x7f80),
        (0xff80_0000, 0xff80),
    ];
    for (bits, expected) in cases {
        assert_eq!(
            f32_to_bf16_rne(f32::from_bits(bits)),
            expected,
            "input {bits:#010x}"
        );
    }
    assert_eq!(f32_to_bf16_rne(0.0), 0x0000);
    assert_eq!(f32_to_bf16_rne(-0.0), 0x8000);
    assert!(bf16::from_bits(f32_to_bf16_rne(f32::NAN)).is_nan());
}

#[test]
fn cpu_oracle_tanh_preserves_signed_zero_and_saturates() {
    for (input, expected) in [
        (0x0000u16, 0x0000u16),
        (0x8000, 0x8000),
        (0x7f80, 0x3f80),
        (0xff80, 0xbf80),
        (0x7f7f, 0x3f80),
        (0xff7f, 0xbf80),
    ] {
        assert_eq!(
            bf16_tanh_reference(input),
            expected,
            "BF16 tanh input {input:#06x}"
        );
    }
    for input in [0x7fc0u16, 0xffc0, 0x7fff, 0xffff] {
        assert!(bf16::from_bits(bf16_tanh_reference(input)).is_nan());
    }
    let one = bf16::from_f32(1.0f32.tanh()).to_bits();
    assert_eq!(bf16_tanh_reference(0x3f80), one);
    assert_eq!(bf16_tanh_reference(0xbf80), one | 0x8000);
}

#[test]
#[ignore = "requires INFINITY_OXIDE_KERNEL_MODULE with Dense exports and a matching CUDA device"]
fn f32_to_bf16_matches_cpu_rounding_oracle_and_guards() {
    let (module, stream) = setup();
    let function = module
        .load_function("infinity_f32_to_bf16")
        .expect("bundle must export infinity_f32_to_bf16");
    let fixture = f32_fixture();
    let expected = fixture
        .iter()
        .map(|value| f32_to_bf16_rne(*value))
        .collect::<Vec<_>>();
    for count in [0usize, 1, 31, 32, 255, 256, 257] {
        let converted = run_f32_to_bf16(&function, &stream, &fixture[..count], 1);
        assert_eq!(converted.len(), count);
        for (index, (got, expected)) in converted.iter().zip(&expected).enumerate() {
            assert_bf16_exact_or_nan(*got, *expected, "f32_to_bf16", index);
        }
    }
    let converted = run_f32_to_bf16(&function, &stream, &fixture, 3);
    for (index, (got, expected)) in converted.iter().zip(&expected).enumerate() {
        assert_bf16_exact_or_nan(*got, *expected, "f32_to_bf16 full", index);
    }
    if let Some(path) = env::var_os(F32_GOLDEN_ENV) {
        let golden = load_golden(&path);
        let values = golden
            .values_f32
            .expect("golden needs values_f32 for FP32 conversion");
        let expected = golden
            .f32_to_bf16_bits
            .expect("golden needs f32_to_bf16_bits for FP32 conversion");
        assert_eq!(values.len(), expected.len(), "golden lengths differ");
        let converted = run_f32_to_bf16(&function, &stream, &values, 1);
        for (index, (got, expected)) in converted.iter().zip(&expected).enumerate() {
            assert_bf16_exact_or_nan(*got, *expected, "native Torch f32_to_bf16 golden", index);
        }
    }
}

#[test]
#[ignore = "requires INFINITY_OXIDE_KERNEL_MODULE with Dense exports and a matching CUDA device"]
fn dense_tanh_bf16_inplace_matches_cpu_reference_and_guards() {
    let (module, stream) = setup();
    let function = module
        .load_function("infinity_dense_tanh_bf16_inplace")
        .expect("bundle must export infinity_dense_tanh_bf16_inplace");
    let fixture = bf16_fixture();
    let expected = fixture
        .iter()
        .map(|bits| bf16_tanh_reference(*bits))
        .collect::<Vec<_>>();
    for count in [0usize, 1, 31, 32, 255, 256, 257] {
        let got = run_tanh(&function, &stream, &fixture[..count], 1);
        assert_eq!(got.len(), count);
        for (index, (got, expected)) in got.iter().zip(&expected).enumerate() {
            assert_bf16_tanh_bound(*got, *expected, fixture[index], "dense_tanh tail", index);
        }
    }
    let got = run_tanh(&function, &stream, &fixture, 2);
    assert_eq!(got.len(), fixture.len());
    for (index, (got, expected)) in got.iter().zip(&expected).enumerate() {
        assert_bf16_tanh_bound(
            *got,
            *expected,
            fixture[index],
            "dense_tanh all BF16 inputs",
            index,
        );
    }
    if let Some(path) = env::var_os(TANH_GOLDEN_ENV) {
        let golden = load_golden(&path);
        let values = golden
            .values_bf16_bits
            .expect("golden needs values_bf16_bits for Tanh");
        let expected = golden
            .tanh_bf16_bits
            .expect("golden needs tanh_bf16_bits for Tanh");
        assert_eq!(values.len(), expected.len(), "golden lengths differ");
        let got = run_tanh(&function, &stream, &values, 1);
        for (index, (got, expected)) in got.iter().zip(&expected).enumerate() {
            assert_bf16_exact_or_nan(*got, *expected, "native Torch Tanh golden", index);
        }
    }
}
