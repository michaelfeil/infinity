// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Numerical controls for the exact width-768 BF16 encoder normalization exports.

use cuda_core::{launch_kernel, Context, DeviceBuffer, Function, Module, Stream};
use half::bf16;
use std::{ffi::c_void, path::PathBuf, sync::Arc};

const WIDTH: usize = 768;
const VALUES: usize = 24;
const GUARD: usize = 16;
const SENTINEL: u16 = 0x4242;
const ROWS: [usize; 4] = [1, 4, 5, 8];

fn setup() -> (Arc<Module>, Arc<Stream>) {
    let root = std::env::var_os("INFINITY_OXIDE_KERNEL_MODULE")
        .filter(|path| !path.is_empty())
        .expect("set INFINITY_OXIDE_KERNEL_MODULE to the exact norm bundle");
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
    // SAFETY: the callers supply the exported argument ABI, retain every CUDA
    // allocation through the blocking download, and use the module context.
    // The native geometry is one warp per row and four rows per 128-thread CTA.
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
    .expect("launch exact width-768 normalization");
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

fn lane_column(lane: usize, index: usize) -> usize {
    lane * 8 + index % 8 + 256 * (index / 8)
}

fn xor_sum(mut values: [f32; 32]) -> f32 {
    for step in [1usize, 2, 4, 8, 16] {
        let previous = values;
        for lane in 0..32 {
            values[lane] += previous[lane ^ step];
        }
    }
    values[0]
}

fn normalize_row(values: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
    let inverse_cols = 1.0f32 / WIDTH as f32;
    let mut means = [0.0f32; 32];
    let mut variances = [0.0f32; 32];
    for lane in 0..32 {
        let mut sum = 0.0f32;
        for index in 0..VALUES {
            sum += values[lane_column(lane, index)];
        }
        means[lane] = sum;
    }
    let mean = xor_sum(means) * inverse_cols;
    for lane in 0..32 {
        let mut variance = 0.0f32;
        for index in 0..VALUES {
            let centered = values[lane_column(lane, index)] - mean;
            variance = centered.mul_add(centered, variance);
        }
        variances[lane] = variance;
    }
    let variance = xor_sum(variances) * inverse_cols;
    let inverse = 1.0 / (variance + eps).sqrt();
    (0..WIDTH)
        .map(|column| {
            let normalized = (values[column] - mean) * inverse;
            round(normalized.mul_add(gamma[column], beta[column]))
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
fn reference_uses_global_mean_before_centered_variance() {
    let values = (0..WIDTH)
        .map(|i| if i < WIDTH / 2 { 1.0 } else { 3.0 })
        .collect::<Vec<_>>();
    let got = normalize_row(&values, &vec![1.0; WIDTH], &vec![0.0; WIDTH], 1e-6);
    for (i, value) in got.iter().enumerate() {
        assert_eq!(*value, if i < WIDTH / 2 { -1.0 } else { 1.0 });
    }
}

#[test]
#[ignore = "requires INFINITY_OXIDE_KERNEL_MODULE with exact 768 exports and a matching CUDA device"]
fn exact_embedding_768_matches_native_recipe_for_all_cta_shapes() {
    let (module, stream) = setup();
    let function = module
        .load_function("infinity_embedding_norm_768_exact")
        .expect("bundle must export infinity_embedding_norm_768_exact");
    let ids = [2u32, 0, 1, 3, 2, 0, 1, 3];
    let positions = [3u32, 1, 2, 0, 3, 1, 2, 0];
    let types = [1u32, 0, 1, 1, 1, 0, 1, 1];
    let word = (0..4 * WIDTH)
        .map(|i| round(1.0 + (((i / WIDTH) * 7 + (i % WIDTH) * 3) % 9) as f32 / 128.0))
        .collect::<Vec<_>>();
    let token_type = (0..2 * WIDTH)
        .map(|i| round((1 + ((i / WIDTH) * 3 + i % WIDTH) % 5) as f32 / 256.0))
        .collect::<Vec<_>>();
    let position = (0..4 * WIDTH)
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
                    let pair = word[token * WIDTH + column] + token_type[typ * WIDTH + column];
                    round(pair) + position[pos * WIDTH + column]
                })
                .collect::<Vec<_>>();
            expected.extend(normalize_row(&values, &gamma, &beta, 1e-12));
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
        let mut eps = 1e-12f32;
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
        check_close(&got, &expected, &format!("embedding rows{rows}"));
    }
}

#[test]
#[ignore = "requires INFINITY_OXIDE_KERNEL_MODULE with exact 768 exports and a matching CUDA device"]
fn exact_residual_768_matches_native_recipe_with_bias_and_residual() {
    let (module, stream) = setup();
    let function = module
        .load_function("infinity_layer_norm_768_exact")
        .expect("bundle must export infinity_layer_norm_768_exact");
    let (gamma, beta) = affine();
    let input = (0..ROWS[3] * WIDTH)
        .map(|i| round(1.0 + (((i / WIDTH) * 7 + (i % WIDTH) * 3) % 9) as f32 / 128.0))
        .collect::<Vec<_>>();
    let bias = (0..WIDTH)
        .map(|c| round((1 + c % 5) as f32 / 256.0))
        .collect::<Vec<_>>();
    let residual = (0..ROWS[3] * WIDTH)
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
                                value + residual[index]
                            } else {
                                value
                            }
                        })
                        .collect::<Vec<_>>();
                    expected.extend(normalize_row(&values, &gamma, &beta, 1e-12));
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
                let mut eps = 1e-12f32;
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
                    &format!("residual rows{rows} bias{use_bias} residual{use_residual}"),
                );
            }
        }
    }
}
