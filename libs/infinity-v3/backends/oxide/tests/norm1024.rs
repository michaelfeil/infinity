// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Numerical controls for the native-statistics width-1024 rounded normalized pair.

use cuda_core::{launch_kernel, Context, DeviceBuffer, Function, Module, Stream};
use half::bf16;
use std::{ffi::c_void, path::PathBuf, sync::Arc};

const WIDTH: usize = 1024;
const THREADS: usize = 256;
const VALUES: usize = WIDTH / THREADS;
const GUARD: usize = 16;
const SENTINEL: u16 = 0x4242;
const ROWS: [usize; 3] = [1, 3, 5];
const MAX_ROWS: usize = 5;

fn setup() -> (Arc<Module>, Arc<Stream>) {
    let root = std::env::var_os("INFINITY_OXIDE_KERNEL_MODULE")
        .filter(|path| !path.is_empty())
        .expect("set INFINITY_OXIDE_KERNEL_MODULE to the native 1024 norm bundle");
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

fn launch(function: &Function, stream: &Stream, rows: usize, args: &mut [*mut c_void]) {
    // SAFETY: callers supply the exported ABI, retain every CUDA allocation
    // through the blocking download, and use the module context. Native
    // geometry is one row per CTA and 256 threads.
    unsafe {
        launch_kernel(
            function.cu_function(),
            (u32::try_from(rows).expect("fixture rows fit u32"), 1, 1),
            (256, 1, 1),
            0,
            stream.cu_stream(),
            args,
        )
    }
    .expect("launch native-statistics width-1024 normalization");
}

fn round(value: f32) -> f32 {
    bf16::from_f32(value).to_f32()
}

fn upload(stream: &Stream, values: &[f32]) -> DeviceBuffer<bf16> {
    let values = values
        .iter()
        .copied()
        .map(bf16::from_f32)
        .collect::<Vec<_>>();
    DeviceBuffer::from_host(stream, &values).expect("upload BF16 fixture")
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

fn guarded_values(stream: &Stream, buffer: &DeviceBuffer<bf16>, elements: usize) -> Vec<f32> {
    let values = download(stream, buffer);
    assert!(values[..GUARD].iter().all(|v| v.to_bits() == SENTINEL));
    assert!(values[GUARD + elements..]
        .iter()
        .all(|v| v.to_bits() == SENTINEL));
    values[GUARD..GUARD + elements]
        .iter()
        .map(|v| v.to_f32())
        .collect()
}

fn affine() -> (Vec<f32>, Vec<f32>) {
    (
        (0..WIDTH).map(|c| 0.5 + (c % 5) as f32 * 0.125).collect(),
        (0..WIDTH)
            .map(|c| ((c % 7) as f32 - 3.0) * 0.015625)
            .collect(),
    )
}

fn warp_sum(values: [f32; 32]) -> f32 {
    let mut values = values;
    for step in [16usize, 8, 4, 2, 1] {
        let previous = values;
        for lane in 0..32 {
            values[lane] += previous[lane ^ step];
        }
    }
    values[0]
}

fn block_sum(values: [f32; THREADS]) -> f32 {
    let mut warps = [0.0f32; 32];
    for warp in 0..8 {
        let mut lanes = [0.0f32; 32];
        lanes.copy_from_slice(&values[warp * 32..warp * 32 + 32]);
        warps[warp] = warp_sum(lanes);
    }
    warp_sum(warps)
}

/// Independent CPU oracle for the native two-pass global-mean statistics.
/// Device grouping is four values per thread at `tid + j*256`; the global mean
/// is reduced before the centered variance pass.
fn reference_row(values: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(values.len(), WIDTH);
    let mut sums = [0.0f32; THREADS];
    for tid in 0..THREADS {
        let mut sum = 0.0f32;
        for index in 0..VALUES {
            sum += values[tid + index * THREADS];
        }
        sums[tid] = sum;
    }
    let mean = block_sum(sums) / WIDTH as f32;
    let mut variances = [0.0f32; THREADS];
    for tid in 0..THREADS {
        let mut variance = 0.0f32;
        for index in 0..VALUES {
            let centered = values[tid + index * THREADS] - mean;
            variance = centered.mul_add(centered, variance);
        }
        variances[tid] = variance;
    }
    let variance = block_sum(variances) / WIDTH as f32;
    let inverse = 1.0 / (variance + eps).sqrt();
    (0..WIDTH)
        .map(|column| {
            let centered = values[column] - mean;
            let normalized = inverse * centered;
            let scaled = normalized * gamma[column];
            round(scaled + beta[column])
        })
        .collect()
}

/// Same global statistics, but evaluated independently in FP64.
fn fp64_reference_row(values: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
    let mean = values.iter().map(|v| *v as f64).sum::<f64>() / WIDTH as f64;
    let variance = values
        .iter()
        .map(|v| (*v as f64 - mean).powi(2))
        .sum::<f64>()
        / WIDTH as f64;
    let inverse = 1.0 / (variance + eps as f64).sqrt();
    (0..WIDTH)
        .map(|column| {
            round(
                ((values[column] as f64 - mean) * inverse * gamma[column] as f64
                    + beta[column] as f64) as f32,
            )
        })
        .collect()
}

fn check_close(got: &[f32], expected: &[f32], label: &str) {
    assert_eq!(got.len(), expected.len());
    for (index, (got, expected)) in got.iter().zip(expected).enumerate() {
        assert!(got.is_finite(), "{label}: nonfinite output at {index}");
        assert!(
            (got - expected).abs() <= 0.02 * (1.0 + expected.abs()),
            "{label}: mismatch at {index}: got {got}, expected {expected}"
        );
    }
}

#[test]
fn rounded_1024_oracle_uses_global_mean_before_variance() {
    let values = (0..WIDTH)
        .map(|column| if column % THREADS == 0 { 0.0 } else { 100.0 })
        .collect::<Vec<_>>();
    let (gamma, beta) = affine();
    let eps = 1e-5f32;
    let got = reference_row(&values, &gamma, &beta, eps);
    let expected = fp64_reference_row(&values, &gamma, &beta, eps);
    check_close(&got, &expected, "CPU oracle global two-pass statistics");

    let local_mean = values[..VALUES].iter().copied().sum::<f32>() / VALUES as f32;
    let local = values[..VALUES]
        .iter()
        .map(|value| {
            let centered = *value - local_mean;
            round(centered * gamma[0] + beta[0])
        })
        .collect::<Vec<_>>();
    assert!(
        (0..VALUES)
            .map(|index| (got[index] - local[index]).abs())
            .fold(0.0f32, f32::max)
            > 0.1,
        "fixture must distinguish a local mean from the global mean"
    );
}

#[test]
#[ignore = "requires INFINITY_OXIDE_KERNEL_MODULE with native 1024 exports and a matching CUDA device"]
fn rounded_1024_embedding_matches_native_statistics_and_guards() {
    let (module, stream) = setup();
    let function = module
        .load_function("infinity_embedding_norm_1024_rounded_native")
        .expect("bundle must export infinity_embedding_norm_1024_rounded_native");
    let ids = [3u32, 0, 2, 4, 1];
    let positions = [4u32, 1, 3, 0, 2];
    let types = [1u32, 0, 1, 0, 1];
    assert_eq!(ids.len(), MAX_ROWS);
    let word = (0..MAX_ROWS * WIDTH)
        .map(|i| round(1.0 + (((i / WIDTH) * 7 + (i % WIDTH) * 3) % 9) as f32 / 128.0))
        .collect::<Vec<_>>();
    let token_type = (0..2 * WIDTH)
        .map(|i| round((1 + ((i / WIDTH) * 3 + i % WIDTH) % 5) as f32 / 256.0))
        .collect::<Vec<_>>();
    let position = (0..MAX_ROWS * WIDTH)
        .map(|i| round(1.0 + (((i / WIDTH) * 5 + (i % WIDTH) * 2) % 7) as f32 / 128.0))
        .collect::<Vec<_>>();
    let (gamma, beta) = affine();
    let ids_device = DeviceBuffer::from_host(&stream, &ids).expect("upload token IDs");
    let positions_device = DeviceBuffer::from_host(&stream, &positions).expect("upload positions");
    let types_device = DeviceBuffer::from_host(&stream, &types).expect("upload token types");
    let word_device = upload(&stream, &word);
    let position_device = upload(&stream, &position);
    let type_device = upload(&stream, &token_type);
    let gamma_device = upload(&stream, &gamma);
    let beta_device = upload(&stream, &beta);
    for rows in ROWS {
        let mut expected = Vec::with_capacity(rows * WIDTH);
        for row in 0..rows {
            let token = ids[row] as usize;
            let typ = types[row] as usize;
            let pos = positions[row] as usize;
            let values = (0..WIDTH)
                .map(|column| {
                    round(
                        round(word[token * WIDTH + column] + token_type[typ * WIDTH + column])
                            + position[pos * WIDTH + column],
                    )
                })
                .collect::<Vec<_>>();
            expected.extend(reference_row(&values, &gamma, &beta, 1e-5));
        }
        let out = guarded(&stream, rows * WIDTH);
        let mut ids_ptr = ids_device.cu_deviceptr();
        let mut positions_ptr = positions_device.cu_deviceptr();
        let mut types_ptr = types_device.cu_deviceptr();
        let mut word_ptr = word_device.cu_deviceptr();
        let mut position_ptr = position_device.cu_deviceptr();
        let mut type_ptr = type_device.cu_deviceptr();
        let mut gamma_ptr = gamma_device.cu_deviceptr();
        let mut beta_ptr = beta_device.cu_deviceptr();
        let mut out_ptr = out.cu_deviceptr() + (2 * GUARD) as u64;
        let mut tokens = u32::try_from(rows).expect("fixture rows fit u32");
        let mut hidden = WIDTH as u32;
        let mut eps = 1e-5f32;
        launch(
            &function,
            &stream,
            rows,
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
                arg(&mut tokens),
                arg(&mut hidden),
                arg(&mut eps),
            ],
        );
        let got = guarded_values(&stream, &out, rows * WIDTH);
        check_close(&got, &expected, &format!("rounded embedding rows{rows}"));
    }
}

#[test]
#[ignore = "requires INFINITY_OXIDE_KERNEL_MODULE with native 1024 exports and a matching CUDA device"]
fn rounded_1024_residual_matches_native_statistics_and_guards() {
    let (module, stream) = setup();
    let function = module
        .load_function("infinity_layer_norm_1024_rounded_native")
        .expect("bundle must export infinity_layer_norm_1024_rounded_native");
    let (gamma, beta) = affine();
    let input = (0..MAX_ROWS * WIDTH)
        .map(|i| round(1.0 + (((i / WIDTH) * 7 + (i % WIDTH) * 3) % 9) as f32 / 128.0))
        .collect::<Vec<_>>();
    let bias = (0..WIDTH)
        .map(|c| round((1 + c % 5) as f32 / 256.0))
        .collect::<Vec<_>>();
    let residual = (0..MAX_ROWS * WIDTH)
        .map(|i| round(1.0 + (((i / WIDTH) * 5 + (i % WIDTH) * 2) % 7) as f32 / 128.0))
        .collect::<Vec<_>>();
    let input_device = upload(&stream, &input);
    let bias_device = upload(&stream, &bias);
    let residual_device = upload(&stream, &residual);
    let gamma_device = upload(&stream, &gamma);
    let beta_device = upload(&stream, &beta);
    for rows in ROWS {
        for use_bias in [false, true] {
            for use_residual in [false, true] {
                let mut expected = Vec::with_capacity(rows * WIDTH);
                for row in 0..rows {
                    let values = (0..WIDTH)
                        .map(|column| {
                            let index = row * WIDTH + column;
                            let value = if use_bias {
                                round(input[index] + bias[column])
                            } else {
                                input[index]
                            };
                            if use_residual {
                                round(value + residual[index])
                            } else {
                                value
                            }
                        })
                        .collect::<Vec<_>>();
                    expected.extend(reference_row(&values, &gamma, &beta, 1e-5));
                }
                let out = guarded(&stream, rows * WIDTH);
                let mut input_ptr = input_device.cu_deviceptr();
                let mut residual_ptr = if use_residual {
                    residual_device.cu_deviceptr()
                } else {
                    0
                };
                let mut bias_ptr = if use_bias {
                    bias_device.cu_deviceptr()
                } else {
                    0
                };
                let mut gamma_ptr = gamma_device.cu_deviceptr();
                let mut beta_ptr = beta_device.cu_deviceptr();
                let mut out_ptr = out.cu_deviceptr() + (2 * GUARD) as u64;
                let mut count = u32::try_from(rows).expect("fixture rows fit u32");
                let mut width = WIDTH as u32;
                let mut eps = 1e-5f32;
                launch(
                    &function,
                    &stream,
                    rows,
                    &mut [
                        arg(&mut input_ptr),
                        arg(&mut residual_ptr),
                        arg(&mut bias_ptr),
                        arg(&mut gamma_ptr),
                        arg(&mut beta_ptr),
                        arg(&mut out_ptr),
                        arg(&mut count),
                        arg(&mut width),
                        arg(&mut eps),
                    ],
                );
                let got = guarded_values(&stream, &out, rows * WIDTH);
                check_close(
                    &got,
                    &expected,
                    &format!("rounded residual rows{rows} bias{use_bias} residual{use_residual}"),
                );
            }
        }
    }
}
