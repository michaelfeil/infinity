// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Direct numerical controls for the materialized BF16 add/LayerNorm recipes.

use cuda_core::{launch_kernel, Context, DeviceBuffer, Function, Module, Stream};
use half::bf16;
use std::{ffi::c_void, path::PathBuf, sync::Arc};

const ROWS: usize = 3;
const GUARD: usize = 16;

fn setup() -> (Arc<Module>, Arc<Stream>) {
    let root = std::env::var_os("INFINITY_OXIDE_KERNEL_MODULE")
        .filter(|path| !path.is_empty())
        .expect("set INFINITY_OXIDE_KERNEL_MODULE to a bundle with the rounded norm exports");
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
        .expect("load CUDA kernel bundle");
    (module, stream)
}

fn arg<T>(value: &mut T) -> *mut c_void {
    (value as *mut T).cast()
}

fn launch(function: &Function, stream: &Stream, args: &mut [*mut c_void]) {
    // SAFETY: callers provide the exact exported argument ABI, retain all CUDA
    // allocations through the blocking download, and use the module's context.
    // The extra block must return at the row bound without writing the guards.
    unsafe {
        launch_kernel(
            function.cu_function(),
            (ROWS as u32 + 1, 1, 1),
            (256, 1, 1),
            0,
            stream.cu_stream(),
            args,
        )
    }
    .expect("launch rounded LayerNorm");
}

fn rounded(value: f32) -> f32 {
    bf16::from_f32(value).to_f32()
}

fn upload(stream: &Stream, values: &[f32]) -> DeviceBuffer<bf16> {
    let encoded = values
        .iter()
        .copied()
        .map(bf16::from_f32)
        .collect::<Vec<_>>();
    DeviceBuffer::from_host(stream, &encoded).expect("upload BF16 fixture")
}

fn affine(width: usize) -> (Vec<f32>, Vec<f32>) {
    let gamma = (0..width).map(|c| 0.5 + (c % 5) as f32 * 0.125).collect();
    let beta = (0..width)
        .map(|c| ((c % 7) as f32 - 3.0) * 0.015625)
        .collect();
    (gamma, beta)
}

/// Independent scalar population statistics over the already rounded add results.
/// FP64 accumulation avoids reproducing the device reduction implementation.
fn reference(values: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
    let width = gamma.len();
    values
        .chunks_exact(width)
        .flat_map(|row| {
            let mean = row.iter().map(|v| *v as f64).sum::<f64>() / width as f64;
            let variance =
                row.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / width as f64;
            let inverse = (variance + eps as f64).sqrt().recip();
            row.iter().enumerate().map(move |(column, value)| {
                rounded(
                    ((*value as f64 - mean) * inverse * gamma[column] as f64 + beta[column] as f64)
                        as f32,
                )
            })
        })
        .collect()
}

fn output(stream: &Stream, elements: usize) -> DeviceBuffer<bf16> {
    DeviceBuffer::from_host(stream, &vec![bf16::from_bits(0x4242); elements + 2 * GUARD])
        .expect("allocate guarded output")
}

fn check_output(
    stream: &Stream,
    output: &DeviceBuffer<bf16>,
    expected: &[f32],
    legacy: &[f32],
    label: &str,
) {
    let (distinguishing, gap) = expected
        .iter()
        .zip(legacy)
        .enumerate()
        .map(|(index, (a, b))| (index, (a - b).abs()))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .expect("nonempty numerical fixture");
    assert!(
        gap > 0.1,
        "{label}: fixture must distinguish legacy FP32 sums"
    );
    let mut host = vec![bf16::ZERO; output.len()];
    output
        .copy_to_host(stream, &mut host)
        .expect("download output and drain stream");
    assert!(host[..GUARD].iter().all(|v| v.to_bits() == 0x4242));
    assert!(host[GUARD + expected.len()..]
        .iter()
        .all(|v| v.to_bits() == 0x4242));
    let got = &host[GUARD..GUARD + expected.len()];
    for (index, (got, expected)) in got.iter().zip(expected).enumerate() {
        let got = got.to_f32();
        assert!(got.is_finite(), "{label}: nonfinite output at {index}");
        assert!(
            (got - expected).abs() <= 0.015 * (1.0 + expected.abs()),
            "{label}: mismatch at {index}: got {got}, expected {expected}"
        );
    }
    assert!(
        (got[distinguishing].to_f32() - expected[distinguishing]).abs() < gap / 4.0,
        "{label}: output must follow rounded sums at the distinguishing column"
    );
}

#[test]
#[ignore = "requires INFINITY_OXIDE_KERNEL_MODULE with rounded exports and a matching CUDA device"]
fn rounded_embedding_matches_independent_reference() {
    let (module, stream) = setup();
    let function = module
        .load_function("infinity_embedding_norm_rounded")
        .expect("bundle must export infinity_embedding_norm_rounded");
    let ids = [2u32, 0, 1];
    let positions = [3u32, 1, 2];
    let types = [1u32, 0, 1];
    let ids_device = DeviceBuffer::from_host(&stream, &ids).expect("upload token IDs");
    let positions_device = DeviceBuffer::from_host(&stream, &positions).expect("upload positions");
    let types_device = DeviceBuffer::from_host(&stream, &types).expect("upload token types");
    for (width, mut eps) in [(768usize, 1e-12f32), (1024, 1e-5)] {
        let word = (0..4 * width)
            .map(|i| rounded(1.0 + (((i / width) * 7 + (i % width) * 3) % 9) as f32 / 128.0))
            .collect::<Vec<_>>();
        let token_type = (0..2 * width)
            .map(|i| rounded((1 + ((i / width) * 3 + i % width) % 5) as f32 / 256.0))
            .collect::<Vec<_>>();
        let position = (0..5 * width)
            .map(|i| rounded(1.0 + (((i / width) * 5 + (i % width) * 2) % 7) as f32 / 128.0))
            .collect::<Vec<_>>();
        let mut values = Vec::with_capacity(ROWS * width);
        let mut unrounded = Vec::with_capacity(ROWS * width);
        for row in 0..ROWS {
            for column in 0..width {
                let pair = rounded(
                    word[ids[row] as usize * width + column]
                        + token_type[types[row] as usize * width + column],
                );
                let sum = pair + position[positions[row] as usize * width + column];
                values.push(rounded(sum));
                unrounded.push(sum);
            }
        }
        let (gamma, beta) = affine(width);
        let expected = reference(&values, &gamma, &beta, eps);
        let legacy = reference(&unrounded, &gamma, &beta, eps);
        let word_device = upload(&stream, &word);
        let type_device = upload(&stream, &token_type);
        let position_device = upload(&stream, &position);
        let gamma_device = upload(&stream, &gamma);
        let beta_device = upload(&stream, &beta);
        let out = output(&stream, ROWS * width);
        let mut ids_ptr = ids_device.cu_deviceptr();
        let mut positions_ptr = positions_device.cu_deviceptr();
        let mut types_ptr = types_device.cu_deviceptr();
        let mut word_ptr = word_device.cu_deviceptr();
        let mut position_ptr = position_device.cu_deviceptr();
        let mut type_ptr = type_device.cu_deviceptr();
        let mut gamma_ptr = gamma_device.cu_deviceptr();
        let mut beta_ptr = beta_device.cu_deviceptr();
        let mut out_ptr = out.cu_deviceptr() + (2 * GUARD) as u64;
        let mut rows = ROWS as u32;
        let mut hidden = width as u32;
        launch(
            &function,
            &stream,
            &mut [
                arg(&mut ids_ptr),
                arg(&mut positions_ptr),
                arg(&mut types_ptr),
                arg(&mut word_ptr),
                arg(&mut position_ptr),
                arg(&mut type_ptr),
                arg(&mut gamma_ptr),
                arg(&mut beta_ptr),
                arg(&mut out_ptr),
                arg(&mut rows),
                arg(&mut hidden),
                arg(&mut eps),
            ],
        );
        check_output(
            &stream,
            &out,
            &expected,
            &legacy,
            &format!("embedding width{width}"),
        );
    }
}

#[test]
#[ignore = "requires INFINITY_OXIDE_KERNEL_MODULE with rounded exports and a matching CUDA device"]
fn rounded_residual_matches_independent_reference() {
    let (module, stream) = setup();
    let function = module
        .load_function("infinity_layer_norm_rounded")
        .expect("bundle must export infinity_layer_norm_rounded");
    for (width, mut eps) in [(768usize, 1e-12f32), (1024, 1e-5)] {
        let input = (0..ROWS * width)
            .map(|i| rounded(1.0 + (((i / width) * 7 + (i % width) * 3) % 9) as f32 / 128.0))
            .collect::<Vec<_>>();
        let bias = (0..width)
            .map(|c| rounded((1 + c % 5) as f32 / 256.0))
            .collect::<Vec<_>>();
        let residual = (0..ROWS * width)
            .map(|i| rounded(1.0 + (((i / width) * 5 + (i % width) * 2) % 7) as f32 / 128.0))
            .collect::<Vec<_>>();
        let unrounded = input
            .iter()
            .zip(&residual)
            .enumerate()
            .map(|(i, (a, b))| rounded(a + bias[i % width]) + b)
            .collect::<Vec<_>>();
        let values = unrounded.iter().copied().map(rounded).collect::<Vec<_>>();
        let (gamma, beta) = affine(width);
        let expected = reference(&values, &gamma, &beta, eps);
        let legacy = reference(&unrounded, &gamma, &beta, eps);
        let input_device = upload(&stream, &input);
        let residual_device = upload(&stream, &residual);
        let bias_device = upload(&stream, &bias);
        let gamma_device = upload(&stream, &gamma);
        let beta_device = upload(&stream, &beta);
        let out = output(&stream, ROWS * width);
        let mut input_ptr = input_device.cu_deviceptr();
        let mut residual_ptr = residual_device.cu_deviceptr();
        let mut bias_ptr = bias_device.cu_deviceptr();
        let mut gamma_ptr = gamma_device.cu_deviceptr();
        let mut beta_ptr = beta_device.cu_deviceptr();
        let mut out_ptr = out.cu_deviceptr() + (2 * GUARD) as u64;
        let mut rows = ROWS as u32;
        let mut hidden = width as u32;
        launch(
            &function,
            &stream,
            &mut [
                arg(&mut input_ptr),
                arg(&mut residual_ptr),
                arg(&mut bias_ptr),
                arg(&mut gamma_ptr),
                arg(&mut beta_ptr),
                arg(&mut out_ptr),
                arg(&mut rows),
                arg(&mut hidden),
                arg(&mut eps),
            ],
        );
        check_output(
            &stream,
            &out,
            &expected,
            &legacy,
            &format!("residual width{width}"),
        );
    }
}
