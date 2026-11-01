// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Independent numerical controls for the BF16 decoder device primitives.

use cuda_core::{launch_kernel, Context, DeviceBuffer, Function, Module, Stream};
use half::bf16;
use std::{ffi::c_void, path::PathBuf, sync::Arc};

const GUARD: usize = 16;
const SENTINEL: u16 = 0x4242;
const ROWS: usize = 3;

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
    // SAFETY: each caller supplies the documented export ABI and retains all
    // buffers/module/context until blocking download completes. An extra block
    // exercises the entry point's bounds check.
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

fn check_recipe(got: &[f32], expected: &[f32], other_recipe: &[f32], label: &str) {
    assert_eq!(got.len(), expected.len());
    let mut error = 0.0;
    let mut distinction = 0.0;
    for (index, ((got, expected), other)) in got.iter().zip(expected).zip(other_recipe).enumerate()
    {
        assert!(got.is_finite(), "{label}: nonfinite value at {index}");
        assert!(
            (got - expected).abs() <= 0.015 * (1.0 + expected.abs()),
            "{label}: index{index}: got {got}, expected {expected}"
        );
        error += (got - expected).abs();
        distinction += (expected - other).abs();
    }
    assert!(
        distinction > 0.01,
        "{label}: fixture must distinguish rounding order"
    );
    assert!(
        error < 0.3 * distinction,
        "{label}: total error {error} must distinguish the alternate recipe distance {distinction}"
    );
}

/// Scalar FP64 mean-square accumulation deliberately differs from the CUDA
/// block reduction. BF16 casts express the externally visible model stages.
fn rms_reference(input: &[f32], weight: &[f32], eps: f32, rounded_affine: bool) -> Vec<f32> {
    input
        .chunks_exact(weight.len())
        .flat_map(|row| {
            let mean_square =
                row.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / row.len() as f64;
            let inverse = (mean_square + eps as f64).sqrt().recip();
            row.iter().zip(weight).map(move |(value, weight)| {
                let normalized = (*value as f64 * inverse) as f32;
                round(
                    if rounded_affine {
                        round(normalized)
                    } else {
                        normalized
                    } * weight,
                )
            })
        })
        .collect()
}

#[test]
#[ignore = "requires INFINITY_OXIDE_KERNEL_MODULE with decoder exports and a matching CUDA device"]
fn rms_norm_and_residual_rounding_match_independent_reference() {
    let (module, stream) = setup();
    for width in [37usize, 896, 1536, 2048] {
        let input = (0..ROWS * width)
            .map(|i| {
                if i / width == 0 {
                    0.0
                } else {
                    round(((i * 37 + 13) % 127) as f32 / 32.0 - 1.96875)
                }
            })
            .collect::<Vec<_>>();
        let residual = (0..ROWS * width)
            .map(|i| {
                if i / width == 0 {
                    0.0
                } else {
                    round(1.0 + ((i * 7 + 3) % 19) as f32 / 256.0)
                }
            })
            .collect::<Vec<_>>();
        let weight = (0..width)
            .map(|c| round(((c * 11) % 23 + 3) as f32 / 16.0))
            .collect::<Vec<_>>();
        let input_device = upload(&stream, &input);
        let residual_device = upload(&stream, &residual);
        let weight_device = upload(&stream, &weight);
        for rounded_recipe in [false, true] {
            for add in [false, true] {
                let name = match (add, rounded_recipe) {
                    (false, false) => "infinity_rms_norm",
                    (false, true) => "infinity_rms_norm_rounded",
                    (true, false) => "infinity_add_rms_norm",
                    (true, true) => "infinity_add_rms_norm_rounded",
                };
                let function = module
                    .load_function(name)
                    .unwrap_or_else(|error| panic!("load {name}: {error}"));
                let unrounded = input
                    .iter()
                    .zip(&residual)
                    .map(|(x, r)| if add { x + r } else { *x })
                    .collect::<Vec<_>>();
                let values = unrounded
                    .iter()
                    .map(|v| if rounded_recipe && add { round(*v) } else { *v })
                    .collect::<Vec<_>>();
                let mut eps = 1e-6f32;
                let expected = rms_reference(&values, &weight, eps, rounded_recipe);
                let other = if add {
                    let alternate = unrounded
                        .iter()
                        .map(|v| if rounded_recipe { *v } else { round(*v) })
                        .collect::<Vec<_>>();
                    rms_reference(&alternate, &weight, eps, rounded_recipe)
                } else {
                    rms_reference(&values, &weight, eps, !rounded_recipe)
                };
                let out = guarded(&stream, input.len());
                let summed = guarded(&stream, input.len());
                let mut input_ptr = input_device.cu_deviceptr();
                let mut residual_ptr = residual_device.cu_deviceptr();
                let mut weight_ptr = weight_device.cu_deviceptr();
                let mut out_ptr = out.cu_deviceptr() + (2 * GUARD) as u64;
                let mut summed_ptr = summed.cu_deviceptr() + (2 * GUARD) as u64;
                let mut rows = ROWS as u32;
                let mut columns = width as u32;
                if add {
                    launch(
                        &function,
                        &stream,
                        ROWS + 1,
                        &mut [
                            arg(&mut input_ptr),
                            arg(&mut residual_ptr),
                            arg(&mut weight_ptr),
                            arg(&mut out_ptr),
                            arg(&mut summed_ptr),
                            arg(&mut rows),
                            arg(&mut columns),
                            arg(&mut eps),
                        ],
                    );
                } else {
                    launch(
                        &function,
                        &stream,
                        ROWS + 1,
                        &mut [
                            arg(&mut input_ptr),
                            arg(&mut weight_ptr),
                            arg(&mut out_ptr),
                            arg(&mut rows),
                            arg(&mut columns),
                            arg(&mut eps),
                        ],
                    );
                }
                let got = guarded_values(&stream, &out, input.len());
                check_recipe(&got, &expected, &other, &format!("{name} width{width}"));
                if add {
                    let actual_sum = guarded_values(&stream, &summed, input.len());
                    for (got, expected) in actual_sum.iter().zip(&unrounded) {
                        assert_eq!(
                            bf16::from_f32(*got).to_bits(),
                            bf16::from_f32(*expected).to_bits(),
                            "{name}: rounded residual output"
                        );
                    }
                }
            }
        }
    }
}

fn rope_reference(
    storage: &mut [bf16],
    positions: &[u32],
    cosine: &[f32],
    sine: &[f32],
    stride: usize,
    table_stride: usize,
    rotary: usize,
    interleaved: bool,
    hfma: bool,
) {
    const Q_HEADS: usize = 4;
    const K_HEADS: usize = 2;
    const DIM: usize = 8;
    let half = rotary / 2;
    for (token, position) in positions.iter().enumerate() {
        for (base, heads) in [(0, Q_HEADS), (Q_HEADS * DIM, K_HEADS)] {
            for head in 0..heads {
                for pair in 0..half {
                    let first = if interleaved { 2 * pair } else { pair };
                    let second = if interleaved { first + 1 } else { pair + half };
                    let offset = GUARD + token * stride + base + head * DIM;
                    let a = storage[offset + first].to_f32();
                    let b = storage[offset + second].to_f32();
                    let c = round(cosine[*position as usize * table_stride + pair]);
                    let s = round(sine[*position as usize * table_stride + pair]);
                    let (low, high) = if hfma {
                        (
                            (a as f64 * c as f64 - round(b * s) as f64) as f32,
                            (b as f64 * c as f64 + round(a * s) as f64) as f32,
                        )
                    } else {
                        (round(a * c) - round(b * s), round(b * c) + round(a * s))
                    };
                    storage[offset + first] = bf16::from_f32(low);
                    storage[offset + second] = bf16::from_f32(high);
                }
            }
        }
    }
}

#[test]
#[ignore = "requires INFINITY_OXIDE_KERNEL_MODULE with decoder exports and a matching CUDA device"]
fn rope_preserves_strided_gqa_values_tails_and_rounding() {
    let (module, stream) = setup();
    let positions = [4u32, 1, 4, 0];
    let positions_device = DeviceBuffer::from_host(&stream, &positions).expect("upload positions");
    let stride = (4 + 2 * 2) * 8 + 7;
    let table_stride = 6;
    let cosine = (0..5 * table_stride)
        .map(|i| ((i * 13 + 3) as f64 * 0.17).cos() as f32)
        .collect::<Vec<_>>();
    let sine = (0..5 * table_stride)
        .map(|i| ((i * 13 + 3) as f64 * 0.17).sin() as f32)
        .collect::<Vec<_>>();
    let cosine_device =
        DeviceBuffer::from_host(&stream, &cosine).expect("upload FP32 cosine table");
    let sine_device = DeviceBuffer::from_host(&stream, &sine).expect("upload FP32 sine table");
    let mut original = vec![bf16::from_bits(SENTINEL); positions.len() * stride + 2 * GUARD];
    for token in 0..positions.len() {
        for column in 0..64 {
            original[GUARD + token * stride + column] =
                bf16::from_f32(((token * 79 + column * 31) % 101) as f32 / 19.0 - 2.6315789);
        }
    }
    for rotary in [6usize, 8] {
        for (name, interleaved, hfma) in [
            ("infinity_rope_neox", false, false),
            ("infinity_rope_interleaved", true, false),
            ("infinity_rope_neox_hfma", false, true),
            ("infinity_rope_interleaved_hfma", true, true),
        ] {
            let mut expected = original.clone();
            rope_reference(
                &mut expected,
                &positions,
                &cosine,
                &sine,
                stride,
                table_stride,
                rotary,
                interleaved,
                hfma,
            );
            let mut alternate = original.clone();
            rope_reference(
                &mut alternate,
                &positions,
                &cosine,
                &sine,
                stride,
                table_stride,
                rotary,
                interleaved,
                !hfma,
            );
            assert!(
                expected
                    .iter()
                    .zip(&alternate)
                    .any(|(a, b)| a.to_bits() != b.to_bits()),
                "RoPE fixture must distinguish product contraction"
            );
            let buffer = DeviceBuffer::from_host(&stream, &original).expect("upload packed QKV");
            let function = module
                .load_function(name)
                .unwrap_or_else(|error| panic!("load {name}: {error}"));
            let mut q = buffer.cu_deviceptr() + (2 * GUARD) as u64;
            let mut k = q + (2 * 4 * 8) as u64;
            let mut c = cosine_device.cu_deviceptr();
            let mut s = sine_device.cu_deviceptr();
            let mut p = positions_device.cu_deviceptr();
            let mut tokens = positions.len() as u32;
            let mut q_heads = 4u32;
            let mut k_heads = 2u32;
            let mut head_dim = 8u32;
            let mut rotary_dim = rotary as u32;
            let mut q_stride = stride as u64;
            let mut k_stride = stride as u64;
            let mut phase_stride = table_stride as u32;
            let count = positions.len() * 6 * (rotary / 2);
            launch(
                &function,
                &stream,
                count.div_ceil(256) + 1,
                &mut [
                    arg(&mut q),
                    arg(&mut k),
                    arg(&mut c),
                    arg(&mut s),
                    arg(&mut p),
                    arg(&mut tokens),
                    arg(&mut q_heads),
                    arg(&mut k_heads),
                    arg(&mut head_dim),
                    arg(&mut rotary_dim),
                    arg(&mut q_stride),
                    arg(&mut k_stride),
                    arg(&mut phase_stride),
                ],
            );
            let got = download(&stream, &buffer);
            for (index, (got, expected)) in got.iter().zip(&expected).enumerate() {
                assert_eq!(got.to_bits(), expected.to_bits(), "{name} rotary{rotary}: packed element{index}; V, unrotated tails, padding and guards must remain exact");
            }
        }
    }
}

fn silu_reference(input: &[f32], width: usize, staged: bool) -> Vec<f32> {
    input
        .chunks_exact(2 * width)
        .flat_map(|row| {
            row[..width]
                .iter()
                .zip(&row[width..])
                .map(move |(gate, up)| {
                    let exponential = (-(*gate as f64)).exp() as f32;
                    let denominator = if staged {
                        round(1.0 + round(exponential))
                    } else {
                        1.0 + exponential
                    };
                    round(round(*gate / denominator) * up)
                })
        })
        .collect()
}

#[test]
#[ignore = "requires INFINITY_OXIDE_KERNEL_MODULE with decoder exports and a matching CUDA device"]
fn silu_mul_preserves_packed_order_and_model_dtype_stages() {
    let (module, stream) = setup();
    let width = 259usize;
    let extremes = [
        -90.0, -80.0, -20.0, -8.0, -2.0, -0.5, -0.015625, 0.0, 0.015625, 0.5, 2.0, 8.0, 20.0, 80.0,
        90.0,
    ];
    let mut input = Vec::with_capacity(ROWS * 2 * width);
    for row in 0..ROWS {
        for column in 0..width {
            input.push(round(if column < extremes.len() {
                extremes[column]
            } else {
                ((column * 37 + row * 17) % 257) as f32 / 32.0 - 4.0
            }));
        }
        input.extend(
            (0..width).map(|column| round(((column * 13 + row * 7) % 31) as f32 / 8.0 - 1.875)),
        );
    }
    let input_device = upload(&stream, &input);
    for (name, staged) in [
        ("infinity_silu_mul", true),
        ("infinity_silu_mul_f32", false),
    ] {
        let function = module
            .load_function(name)
            .unwrap_or_else(|error| panic!("load {name}: {error}"));
        let expected = silu_reference(&input, width, staged);
        let alternate = silu_reference(&input, width, !staged);
        let out = guarded(&stream, ROWS * width);
        let mut input_ptr = input_device.cu_deviceptr();
        let mut out_ptr = out.cu_deviceptr() + (2 * GUARD) as u64;
        let mut rows = ROWS as u32;
        let mut columns = width as u32;
        launch(
            &function,
            &stream,
            (ROWS * width).div_ceil(256) + 1,
            &mut [
                arg(&mut input_ptr),
                arg(&mut out_ptr),
                arg(&mut rows),
                arg(&mut columns),
            ],
        );
        let got = guarded_values(&stream, &out, ROWS * width);
        check_recipe(&got, &expected, &alternate, name);
    }
}

#[test]
#[ignore = "requires INFINITY_OXIDE_KERNEL_MODULE with gather exports and a matching CUDA device"]
fn embedding_gather_preserves_ids_bits_scale_and_guards() {
    let (module, stream) = setup();
    let ids = [3u32, 0, 2, 2, 1];
    let ids_device = DeviceBuffer::from_host(&stream, &ids).expect("upload gather IDs");
    for width in [1usize, 37, 1536, 2048] {
        let mut table = (0..4 * width)
            .map(|i| bf16::from_f32(((i * 37 + 19) % 257) as f32 / 16.0 - 8.0))
            .collect::<Vec<_>>();
        // A bitwise gather must retain signed zero without arithmetic.
        table[0] = bf16::from_bits(0x8000);
        table[3 * width] = bf16::from_f32(1024.0);
        let table_device = DeviceBuffer::from_host(&stream, &table).expect("upload word table");
        let elements = ids.len() * width;
        for scale in [None, Some(0.0f32), Some(0.5), Some(-1.25), Some(1.00390625)] {
            let name = if scale.is_some() {
                "infinity_embedding_gather_scaled"
            } else {
                "infinity_embedding_gather"
            };
            let function = module
                .load_function(name)
                .unwrap_or_else(|error| panic!("load {name}: {error}"));
            let out = guarded(&stream, elements);
            let mut ids_ptr = ids_device.cu_deviceptr();
            let mut table_ptr = table_device.cu_deviceptr();
            let mut out_ptr = out.cu_deviceptr() + (2 * GUARD) as u64;
            let mut tokens = ids.len() as u32;
            let mut columns = width as u32;
            let mut args = vec![
                arg(&mut ids_ptr),
                arg(&mut table_ptr),
                arg(&mut out_ptr),
                arg(&mut tokens),
                arg(&mut columns),
            ];
            let mut multiplier = scale.unwrap_or(1.0);
            if scale.is_some() {
                args.push(arg(&mut multiplier));
            }
            launch(&function, &stream, elements.div_ceil(256) + 1, &mut args);
            let got = download(&stream, &out);
            assert!(got[..GUARD].iter().all(|v| v.to_bits() == SENTINEL));
            assert!(got[GUARD + elements..]
                .iter()
                .all(|v| v.to_bits() == SENTINEL));
            for (row, id) in ids.iter().enumerate() {
                for column in 0..width {
                    let value = table[*id as usize * width + column];
                    let expected = scale
                        .map(|scale| bf16::from_f32(value.to_f32() * scale))
                        .unwrap_or(value);
                    assert_eq!(
                        got[GUARD + row * width + column].to_bits(),
                        expected.to_bits(),
                        "{name} width{width} row{row} column{column} scale{scale:?}"
                    );
                }
            }
        }
        assert_eq!(
            download(&stream, &table_device),
            table,
            "gather mutated its word table"
        );
    }
}
