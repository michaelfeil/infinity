// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Numerical controls for the exact width-2048 BF16 RMSNorm exports.

use cuda_core::{launch_kernel, Context, DeviceBuffer, Function, Module, Stream};
use half::bf16;
use std::{ffi::c_void, path::PathBuf, sync::Arc};

const WIDTH: usize = 2048;
const VALUES: usize = 64;
const LANE_VALUES: usize = 8;
const ITERATION_STRIDE: usize = 256;
const GUARD: usize = 16;
const SENTINEL: u16 = 0x4242;
const ROWS: [usize; 7] = [1, 3, 4, 5, 127, 128, 257];
const MAX_ROWS: usize = 257;

fn setup() -> (Arc<Module>, Arc<Stream>) {
    let root = std::env::var_os("INFINITY_OXIDE_KERNEL_MODULE")
        .filter(|path| !path.is_empty())
        .expect("set INFINITY_OXIDE_KERNEL_MODULE to the exact RMS 2048 bundle");
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
    // SAFETY: the callers supply the documented export ABI, retain every CUDA
    // allocation through blocking download, and use the module context.
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
    .expect("launch exact width-2048 normalization");
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
    upload_bf16(stream, &values)
}

fn upload_bf16(stream: &Stream, values: &[bf16]) -> DeviceBuffer<bf16> {
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

fn ftz(value: f32) -> f32 {
    if value.is_subnormal() {
        0.0f32.copysign(value)
    } else {
        value
    }
}

fn add_ftz(left: f32, right: f32) -> f32 {
    ftz(left + right)
}

fn subtract_ftz(left: f32, right: f32) -> f32 {
    add_ftz(left, -right)
}

fn multiply_ftz(left: f32, right: f32) -> f32 {
    ftz(left * right)
}

fn fma_ftz(left: f32, right: f32, addend: f32) -> f32 {
    ftz(left.mul_add(right, addend))
}

fn lane_column(lane: usize, index: usize) -> usize {
    lane * LANE_VALUES + index % LANE_VALUES + ITERATION_STRIDE * (index / LANE_VALUES)
}

fn reduce_sum(mut lanes: [f32; 32]) -> f32 {
    for step in [1usize, 2, 4, 8, 16] {
        let previous = lanes;
        for lane in 0..32 {
            lanes[lane] = add_ftz(lanes[lane], previous[lane ^ step]);
        }
    }
    lanes[0]
}

#[derive(Clone, Copy)]
struct RecipeStats {
    mean: f32,
    variance_sum: f32,
    inverse: f32,
}

/// Explicit F32 operations with FTZ at each source boundary. The CPU square
/// root models the native expression only; CUDA rsqrt.approx bits are not
/// modelled here.
fn recipe_row(
    input: &[f32],
    residual: Option<&[f32]>,
    gamma: &[f32],
    eps: f32,
    rounded_staging: bool,
) -> (Vec<f32>, RecipeStats) {
    assert_eq!(input.len(), WIDTH);
    assert_eq!(gamma.len(), WIDTH);
    if let Some(residual) = residual {
        assert_eq!(residual.len(), WIDTH);
    }
    let mut values = vec![0.0f32; WIDTH];
    for (index, value) in values.iter_mut().enumerate() {
        let mut loaded = multiply_ftz(input[index], 1.0);
        if let Some(residual) = residual {
            loaded = add_ftz(loaded, residual[index]);
        }
        *value = if rounded_staging {
            round(loaded)
        } else {
            loaded
        };
    }
    let mut lane_sums = [0.0f32; 32];
    for (lane, sum) in lane_sums.iter_mut().enumerate() {
        let mut total = 0.0;
        for index in 0..VALUES {
            total = add_ftz(total, values[lane_column(lane, index)]);
        }
        *sum = total;
    }
    let mean = multiply_ftz(reduce_sum(lane_sums), 1.0 / WIDTH as f32);
    let mut lane_variances = [0.0f32; 32];
    for (lane, variance) in lane_variances.iter_mut().enumerate() {
        let mut total = 0.0;
        for index in 0..VALUES {
            let centered = subtract_ftz(values[lane_column(lane, index)], mean);
            total = fma_ftz(centered, centered, total);
        }
        *variance = total;
    }
    let variance_sum = reduce_sum(lane_variances);
    let mean_squared = multiply_ftz(mean, mean);
    let denominator = add_ftz(fma_ftz(variance_sum, 1.0 / WIDTH as f32, eps), mean_squared);
    let inverse = ftz(denominator.sqrt().recip());
    let out = values
        .iter()
        .zip(gamma)
        .map(|(value, gamma)| {
            let centered = subtract_ftz(*value, 0.0);
            let normalized = multiply_ftz(inverse, centered);
            round(fma_ftz(*gamma, normalized, 0.0))
        })
        .collect();
    (
        out,
        RecipeStats {
            mean,
            variance_sum,
            inverse,
        },
    )
}

/// Independent FP64 mean-square math with only the externally visible BF16
/// staging at the affine output. This deliberately avoids the native centered
/// variance bookkeeping and is not a bit-exact CUDA claim.
fn mathematical_row(input: &[f32], residual: Option<&[f32]>, gamma: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(input.len(), WIDTH);
    assert_eq!(gamma.len(), WIDTH);
    if let Some(residual) = residual {
        assert_eq!(residual.len(), WIDTH);
    }
    let values = input
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let value = *value as f64;
            match residual {
                Some(residual) => value + residual[index] as f64,
                None => value,
            }
        })
        .collect::<Vec<_>>();
    let mean_square = values.iter().map(|value| value * value).sum::<f64>() / WIDTH as f64;
    let inverse = (mean_square + eps as f64).sqrt().recip();
    values
        .iter()
        .zip(gamma)
        .map(|(value, gamma)| round((*value * inverse * (*gamma as f64)) as f32))
        .collect()
}

fn check_close(got: &[f32], expected: &[f32], tolerance: f64, label: &str) {
    assert_eq!(got.len(), expected.len());
    for (index, (got, expected)) in got.iter().zip(expected).enumerate() {
        assert!(got.is_finite(), "{label}: nonfinite output at {index}");
        let error = (*got as f64 - *expected as f64).abs();
        let limit = tolerance * (1.0 + expected.abs() as f64);
        assert!(
            error <= limit,
            "{label}: index{index}: got {got}, expected {expected}, error {error} > {limit}"
        );
    }
}

fn learned_gamma() -> Vec<f32> {
    (0..WIDTH)
        .map(|column| round(0.5 + ((column * 11 + 3) % 23) as f32 / 16.0))
        .collect()
}

fn nonzero_mean_input(rows: usize) -> Vec<f32> {
    (0..rows * WIDTH)
        .map(|index| {
            let row = index / WIDTH;
            let column = index % WIDTH;
            round(1.5 + (((column * 37 + row * 13) % 127) as f32 - 63.0) / 64.0)
        })
        .collect()
}

fn halfway_inputs(rows: usize) -> (Vec<f32>, Vec<f32>) {
    let mut input = Vec::with_capacity(rows * WIDTH);
    let mut residual = Vec::with_capacity(rows * WIDTH);
    for row in 0..rows {
        for column in 0..WIDTH {
            let mantissa = (column * 37 + row * 11 + 13) % 128;
            input.push(round(1.0 + mantissa as f32 / 128.0));
            let sign = if (column * 7 + row * 5 + 3) % 2 == 0 {
                1.0
            } else {
                -1.0
            };
            residual.push(sign * 2.0f32.powi(-8));
        }
    }
    (input, residual)
}

fn subnormal_inputs(rows: usize) -> (Vec<bf16>, Vec<bf16>) {
    let positive = bf16::from_bits(0x0001);
    let negative = bf16::from_bits(0x8001);
    let mut input = Vec::with_capacity(rows * WIDTH);
    let mut residual = Vec::with_capacity(rows * WIDTH);
    for row in 0..rows {
        for column in 0..WIDTH {
            let value = if (column + row) % 2 == 0 {
                positive
            } else {
                negative
            };
            input.push(value);
            residual.push(value);
        }
    }
    (input, residual)
}

/// Test-only bridge for the separate native adapter: when
/// `INFINITY_OXIDE_RMS2048_OUTPUT_DIR` is set, write the exact BF16 inputs,
/// weights, and downloaded BF16 outputs for a same-input comparison.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn dump_raw(
    label: &str,
    rows: usize,
    eps: f32,
    input: &[f32],
    residual: Option<&[f32]>,
    weight: &[f32],
    out: &[f32],
    summed: Option<&[f32]>,
) {
    let Some(output_dir) = std::env::var_os("INFINITY_OXIDE_RMS2048_OUTPUT_DIR") else {
        return;
    };
    let directory = PathBuf::from(output_dir);
    std::fs::create_dir_all(&directory).expect("create raw fixture directory");
    let bf16_bytes = |values: &[f32]| {
        values
            .iter()
            .flat_map(|value| bf16::from_f32(*value).to_bits().to_le_bytes())
            .collect::<Vec<_>>()
    };
    let write = |name: &str, bytes: Vec<u8>| {
        std::fs::write(directory.join(name), bytes).expect("write raw fixture");
    };
    write(
        &format!("{label}.rows.bin"),
        (rows as u32).to_le_bytes().to_vec(),
    );
    write(
        &format!("{label}.width.bin"),
        (WIDTH as u32).to_le_bytes().to_vec(),
    );
    write(&format!("{label}.eps.bin"), eps.to_le_bytes().to_vec());
    write(&format!("{label}.input.bf16"), bf16_bytes(input));
    if let Some(residual) = residual {
        write(&format!("{label}.residual.bf16"), bf16_bytes(residual));
    }
    write(&format!("{label}.weight.bf16"), bf16_bytes(weight));
    write(&format!("{label}.out.bf16"), bf16_bytes(out));
    if let Some(summed) = summed {
        write(&format!("{label}.summed.bf16"), bf16_bytes(summed));
    }
}

#[test]
fn recipe_uses_global_mean_then_restores_mean_square() {
    let input = nonzero_mean_input(1);
    let gamma = learned_gamma();
    let (out, stats) = recipe_row(&input, None, &gamma, 1e-5, false);
    let global_mean = input.iter().map(|value| *value as f64).sum::<f64>() / WIDTH as f64;
    assert!(
        (stats.mean as f64 - global_mean).abs() <= 1e-5 * (1.0 + global_mean.abs()),
        "recipe mean {} must use the global row mean {global_mean}",
        stats.mean
    );
    let lane_means = (0..32)
        .map(|lane| {
            (0..VALUES)
                .map(|index| input[lane_column(lane, index)] as f64)
                .sum::<f64>()
                / VALUES as f64
        })
        .collect::<Vec<_>>();
    assert!(
        lane_means
            .iter()
            .any(|mean| (*mean - global_mean).abs() > 1e-3),
        "fixture must distinguish a global mean from lane-local means"
    );
    let mean_square = input
        .iter()
        .map(|value| (*value as f64) * (*value as f64))
        .sum::<f64>()
        / WIDTH as f64;
    let restored = stats.variance_sum as f64 / WIDTH as f64 + (stats.mean as f64).powi(2);
    assert!(
        (restored - mean_square).abs() <= 1e-5 * (1.0 + mean_square.abs()),
        "centered variance plus mean squared must restore the global mean square"
    );
    let expected = mathematical_row(&input, None, &gamma, 1e-5);
    check_close(&out, &expected, 0.01, "global mean recipe");
    assert!(stats.inverse.is_finite());
}

#[test]
fn bf16_halfway_residual_fixture_distinguishes_staging() {
    let (input, residual) = halfway_inputs(1);
    let gamma = learned_gamma();
    let (unrounded, _) = recipe_row(&input, Some(&residual), &gamma, 1e-5, false);
    let (rounded, _) = recipe_row(&input, Some(&residual), &gamma, 1e-5, true);
    let mut differences = 0usize;
    let mut staged_differences = 0usize;
    for ((input, residual), (unrounded, rounded)) in input
        .iter()
        .zip(&residual)
        .zip(unrounded.iter().zip(&rounded))
    {
        let staged = add_ftz(multiply_ftz(*input, 1.0), *residual);
        if bf16::from_f32(staged).to_f32().to_bits() != staged.to_bits() {
            staged_differences += 1;
        }
        if bf16::from_f32(*unrounded).to_bits() != bf16::from_f32(*rounded).to_bits() {
            differences += 1;
        }
    }
    assert!(
        staged_differences > 0,
        "halfway residual fixture must exercise BF16 staging"
    );
    assert!(
        differences > 100,
        "halfway residual fixture must distinguish rounded from unrounded statistics; got {differences}"
    );
}

#[test]
fn ftz_staging_flushes_subnormal_inputs_to_signed_zero() {
    let positive = f32::from_bits(0x0001_0000);
    let negative = -positive;
    let mut input = Vec::with_capacity(WIDTH);
    let mut residual = Vec::with_capacity(WIDTH);
    for column in 0..WIDTH {
        let value = if column % 2 == 0 { positive } else { negative };
        input.push(value);
        residual.push(value);
    }
    let gamma = learned_gamma();
    let (out, stats) = recipe_row(&input, Some(&residual), &gamma, 1e-5, false);
    assert_eq!(stats.mean.to_bits(), 0, "global mean of flushed zeros");
    for (column, (input, residual)) in input.iter().zip(&residual).enumerate() {
        let staged = add_ftz(multiply_ftz(*input, 1.0), *residual);
        let expected = if *input < 0.0 { 0x8000 } else { 0x0000 };
        assert_eq!(
            bf16::from_f32(staged).to_bits(),
            expected,
            "staged signed zero at {column}"
        );
        assert_eq!(
            bf16::from_f32(out[column]).to_bits(),
            0,
            "native affine adds +0 after a zero normalized value"
        );
    }
}

#[test]
#[ignore = "requires INFINITY_OXIDE_KERNEL_MODULE with exact 2048 exports and a matching CUDA device"]
fn exact_rms_2048_matches_f64_and_f32_recipe_across_cta_shapes() {
    let (module, stream) = setup();
    let function = module
        .load_function("infinity_rms_norm_2048_exact")
        .expect("bundle must export infinity_rms_norm_2048_exact");
    let input = nonzero_mean_input(MAX_ROWS);
    let gamma = learned_gamma();
    let gamma_device = upload(&stream, &gamma);
    for rows in ROWS {
        let input_device = upload(&stream, &input[..rows * WIDTH]);
        let mut expected = Vec::with_capacity(rows * WIDTH);
        let mut mathematical = Vec::with_capacity(rows * WIDTH);
        for row in 0..rows {
            let values = &input[row * WIDTH..(row + 1) * WIDTH];
            let (recipe, stats) = recipe_row(values, None, &gamma, 1e-5, false);
            assert!(stats.mean.is_finite() && stats.inverse.is_finite());
            expected.extend_from_slice(&recipe);
            mathematical.extend(mathematical_row(values, None, &gamma, 1e-5));
        }
        let out = guarded(&stream, rows * WIDTH);
        let mut input_ptr = input_device.cu_deviceptr();
        let mut gamma_ptr = gamma_device.cu_deviceptr();
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
                arg(&mut gamma_ptr),
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
            0.01,
            &format!("exact RMS 2048 rows{rows} F32 recipe"),
        );
        check_close(
            &got,
            &mathematical,
            0.02,
            &format!("exact RMS 2048 rows{rows} F64 math"),
        );
        dump_raw(
            &format!("plain-rows{rows}"),
            rows,
            1e-5,
            &input[..rows * WIDTH],
            None,
            &gamma,
            &got,
            None,
        );
    }

    // The width guard must leave every output sentinel untouched.
    let input_device = upload(&stream, &input[..WIDTH]);
    let out = guarded(&stream, WIDTH);
    let mut input_ptr = input_device.cu_deviceptr();
    let mut gamma_ptr = gamma_device.cu_deviceptr();
    let mut out_ptr = out.cu_deviceptr() + (2 * GUARD) as u64;
    let mut count = 1u32;
    let mut width = (WIDTH - 1) as u32;
    let mut eps = 1e-5f32;
    launch(
        &function,
        &stream,
        1,
        &mut [
            arg(&mut input_ptr),
            arg(&mut gamma_ptr),
            arg(&mut out_ptr),
            arg(&mut count),
            arg(&mut width),
            arg(&mut eps),
        ],
    );
    let values = download(&stream, &out);
    assert!(values.iter().all(|value| value.to_bits() == SENTINEL));
}

#[test]
#[ignore = "requires INFINITY_OXIDE_KERNEL_MODULE with exact 2048 exports and a matching CUDA device"]
fn exact_add_rms_2048_keeps_unrounded_statistics_and_bf16_halfway_sums() {
    let (module, stream) = setup();
    let function = module
        .load_function("infinity_add_rms_norm_2048_exact")
        .expect("bundle must export infinity_add_rms_norm_2048_exact");
    let (input, residual) = halfway_inputs(MAX_ROWS);
    let gamma = learned_gamma();
    let gamma_device = upload(&stream, &gamma);
    let mut distinction = 0.0f64;
    let mut error = 0.0f64;
    for rows in ROWS {
        let input_device = upload(&stream, &input[..rows * WIDTH]);
        let residual_device = upload(&stream, &residual[..rows * WIDTH]);
        let mut expected = Vec::with_capacity(rows * WIDTH);
        let mut mathematical = Vec::with_capacity(rows * WIDTH);
        let mut rounded = Vec::with_capacity(rows * WIDTH);
        let mut staged = Vec::with_capacity(rows * WIDTH);
        for row in 0..rows {
            let row_input = &input[row * WIDTH..(row + 1) * WIDTH];
            let row_residual = &residual[row * WIDTH..(row + 1) * WIDTH];
            let (recipe, _) = recipe_row(row_input, Some(row_residual), &gamma, 1e-5, false);
            let (alternate, _) = recipe_row(row_input, Some(row_residual), &gamma, 1e-5, true);
            expected.extend_from_slice(&recipe);
            mathematical.extend(mathematical_row(
                row_input,
                Some(row_residual),
                &gamma,
                1e-5,
            ));
            rounded.extend_from_slice(&alternate);
            staged.extend(
                row_input
                    .iter()
                    .zip(row_residual)
                    .map(|(input, residual)| add_ftz(multiply_ftz(*input, 1.0), *residual)),
            );
        }
        let out = guarded(&stream, rows * WIDTH);
        let summed = guarded(&stream, rows * WIDTH);
        let mut input_ptr = input_device.cu_deviceptr();
        let mut residual_ptr = residual_device.cu_deviceptr();
        let mut gamma_ptr = gamma_device.cu_deviceptr();
        let mut out_ptr = out.cu_deviceptr() + (2 * GUARD) as u64;
        let mut summed_ptr = summed.cu_deviceptr() + (2 * GUARD) as u64;
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
                arg(&mut gamma_ptr),
                arg(&mut out_ptr),
                arg(&mut summed_ptr),
                arg(&mut count),
                arg(&mut width),
                arg(&mut eps),
            ],
        );
        let got = guarded_values(&stream, &out, rows * WIDTH);
        let got_summed = guarded_values(&stream, &summed, rows * WIDTH);
        for (index, (got, expected)) in got_summed.iter().zip(&staged).enumerate() {
            assert_eq!(
                bf16::from_f32(*got).to_bits(),
                bf16::from_f32(*expected).to_bits(),
                "rows{rows}: rounded summed stream at {index}"
            );
        }
        for (recipe, alternate) in expected.iter().zip(&rounded) {
            distinction += (*recipe as f64 - *alternate as f64).abs();
        }
        for (got, recipe) in got.iter().zip(&expected) {
            error += (*got as f64 - *recipe as f64).abs();
        }
        check_close(
            &got,
            &expected,
            0.01,
            &format!("exact add RMS 2048 rows{rows} F32 recipe"),
        );
        check_close(
            &got,
            &mathematical,
            0.02,
            &format!("exact add RMS 2048 rows{rows} F64 math"),
        );
        dump_raw(
            &format!("add-halfway-rows{rows}"),
            rows,
            1e-5,
            &input[..rows * WIDTH],
            Some(&residual[..rows * WIDTH]),
            &gamma,
            &got,
            Some(&got_summed),
        );
    }
    assert!(
        distinction > 1.0,
        "halfway fixture must distinguish rounded from unrounded staging"
    );
    assert!(
        error < 0.3 * distinction,
        "unrounded recipe error {error} must be much smaller than alternate-recipe distance {distinction}"
    );
}

#[test]
#[ignore = "requires INFINITY_OXIDE_KERNEL_MODULE with exact 2048 exports and a matching CUDA device"]
fn exact_add_rms_2048_flushes_subnormals_and_preserves_signed_zero_sums() {
    let (module, stream) = setup();
    let function = module
        .load_function("infinity_add_rms_norm_2048_exact")
        .expect("bundle must export infinity_add_rms_norm_2048_exact");
    let gamma = vec![1.0f32; WIDTH];
    let gamma_device = upload(&stream, &gamma);
    for rows in [1usize, 5] {
        let (input, residual) = subnormal_inputs(rows);
        let input_device = upload_bf16(&stream, &input);
        let residual_device = upload_bf16(&stream, &residual);
        let out = guarded(&stream, rows * WIDTH);
        let summed = guarded(&stream, rows * WIDTH);
        let mut input_ptr = input_device.cu_deviceptr();
        let mut residual_ptr = residual_device.cu_deviceptr();
        let mut gamma_ptr = gamma_device.cu_deviceptr();
        let mut out_ptr = out.cu_deviceptr() + (2 * GUARD) as u64;
        let mut summed_ptr = summed.cu_deviceptr() + (2 * GUARD) as u64;
        let mut count = rows as u32;
        let mut width = WIDTH as u32;
        let mut eps = 1e-5f32;
        launch(
            &function,
            &stream,
            rows,
            &mut [
                arg(&mut input_ptr),
                arg(&mut residual_ptr),
                arg(&mut gamma_ptr),
                arg(&mut out_ptr),
                arg(&mut summed_ptr),
                arg(&mut count),
                arg(&mut width),
                arg(&mut eps),
            ],
        );
        let got = guarded_values(&stream, &out, rows * WIDTH);
        let got_summed = guarded_values(&stream, &summed, rows * WIDTH);
        for (index, value) in got.iter().enumerate() {
            assert_eq!(
                bf16::from_f32(*value).to_bits(),
                0,
                "rows{rows}: affine output at {index} must be +0"
            );
            let expected = if input[index].to_bits() & 0x8000 != 0 {
                0x8000
            } else {
                0x0000
            };
            assert_eq!(
                bf16::from_f32(got_summed[index]).to_bits(),
                expected,
                "rows{rows}: signed-zero summed stream at {index}"
            );
        }
        let input_f32 = input.iter().map(|value| value.to_f32()).collect::<Vec<_>>();
        let residual_f32 = residual
            .iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        dump_raw(
            &format!("add-subnormal-rows{rows}"),
            rows,
            1e-5,
            &input_f32,
            Some(&residual_f32),
            &gamma,
            &got,
            Some(&got_summed),
        );
    }
}
