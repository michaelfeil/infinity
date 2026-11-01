// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Same-input bit controls for the cached H1536 ordinary RMS recipe.

use cuda_core::{launch_kernel, Context, DeviceBuffer, Function, Module, Stream};
use half::bf16;
use std::{ffi::c_void, path::PathBuf, sync::Arc};

const WIDTH: usize = 1536;
const GUARD: usize = 16;
const SENTINEL: u16 = 0x4242;

fn setup() -> (Arc<Module>, Arc<Stream>) {
    let root = PathBuf::from(
        std::env::var_os("INFINITY_OXIDE_KERNEL_MODULE")
            .expect("set INFINITY_OXIDE_KERNEL_MODULE to the H1536 bundle"),
    );
    let context = Context::new(0).expect("create GPU0 context");
    let stream = context.new_stream().expect("create stream");
    let path = if root.is_dir() {
        let (major, minor) = context.compute_capability().expect("query architecture");
        root.join(format!("sm_{major}{minor}/kernels.cubin"))
    } else {
        root
    };
    let module = context
        .load_module_from_image(&std::fs::read(path).expect("read cubin"))
        .expect("load cubin");
    (module, stream)
}

fn arg<T>(value: &mut T) -> *mut c_void {
    (value as *mut T).cast()
}

fn guarded(stream: &Stream, data: &[u16]) -> DeviceBuffer<u16> {
    let mut values = vec![SENTINEL; data.len() + 2 * GUARD];
    values[GUARD..GUARD + data.len()].copy_from_slice(data);
    DeviceBuffer::from_host(stream, &values).expect("upload guarded storage")
}

fn pointer(buffer: &DeviceBuffer<u16>) -> u64 {
    buffer.cu_deviceptr() + (2 * GUARD) as u64
}

fn download(stream: &Stream, buffer: &DeviceBuffer<u16>, len: usize) -> Vec<u16> {
    let mut data = vec![0; buffer.len()];
    buffer
        .copy_to_host(stream, &mut data)
        .expect("drain stream");
    assert!(data[..GUARD].iter().all(|&x| x == SENTINEL));
    assert!(data[GUARD + len..].iter().all(|&x| x == SENTINEL));
    data[GUARD..GUARD + len].to_vec()
}

#[allow(clippy::too_many_arguments)]
fn run(
    function: &Function,
    stream: &Stream,
    input: &[u16],
    residual: &[u16],
    weight: &[u16],
    rows: usize,
    width: usize,
    add: bool,
    alias: usize,
    eps: f32,
) -> (Vec<u16>, Vec<u16>) {
    let x = guarded(stream, input);
    let r = guarded(stream, residual);
    let w = guarded(stream, weight);
    let out = guarded(stream, &vec![SENTINEL; input.len()]);
    let summed = guarded(stream, &vec![SENTINEL; input.len()]);
    let (out_buffer, summed_buffer) = match alias {
        0 => (&out, &summed),
        1 => (&x, if add { &r } else { &summed }),
        2 if add => (&r, &x),
        _ => panic!("invalid test alias"),
    };
    let mut x_ptr = pointer(&x);
    let mut r_ptr = pointer(&r);
    let mut w_ptr = pointer(&w);
    let mut out_ptr = pointer(out_buffer);
    let mut summed_ptr = pointer(summed_buffer);
    let mut count = rows as u32;
    let mut columns = width as u32;
    let mut epsilon = eps;
    let mut args = if add {
        vec![
            arg(&mut x_ptr),
            arg(&mut r_ptr),
            arg(&mut w_ptr),
            arg(&mut out_ptr),
            arg(&mut summed_ptr),
            arg(&mut count),
            arg(&mut columns),
            arg(&mut epsilon),
        ]
    } else {
        vec![
            arg(&mut x_ptr),
            arg(&mut w_ptr),
            arg(&mut out_ptr),
            arg(&mut count),
            arg(&mut columns),
            arg(&mut epsilon),
        ]
    };
    // SAFETY: both exports use the same documented six/eight-argument ABI,
    // one row per 256-thread CTA, valid equal-sized alias targets and retained
    // buffers. An extra CTA exercises the row bounds check.
    unsafe {
        launch_kernel(
            function.cu_function(),
            ((rows + 1) as u32, 1, 1),
            (256, 1, 1),
            0,
            stream.cu_stream(),
            &mut args,
        )
    }
    .expect("launch RMS");
    let result = download(stream, out_buffer, input.len());
    let sum = download(stream, summed_buffer, input.len());
    for buffer in [&x, &r, &w, &out, &summed] {
        let len = if std::ptr::eq(buffer, &w) {
            weight.len()
        } else {
            input.len()
        };
        let _ = download(stream, buffer, len);
    }
    (result, sum)
}

fn fixtures(rows: usize, kind: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let input = (0..rows * WIDTH)
        .map(|i| {
            bf16::from_f32(match kind {
                0 => ((i * 37 + 13) % 127) as f32 / 32.0 - 1.96875,
                1 => {
                    let exponent = ((i * 11) % 41) as i32 - 20;
                    let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                    sign * 2.0f32.powi(exponent) * (1.0 + (i % 7) as f32 / 16.0)
                }
                2 => f32::from_bits(([0, 0x8000, 1, 0x8001, 0x3f80, 0xbf80][i % 6] as u32) << 16),
                _ => unreachable!(),
            })
            .to_bits()
        })
        .collect::<Vec<_>>();
    let residual = input
        .iter()
        .enumerate()
        .map(|(i, &bits)| {
            bf16::from_f32(if kind == 1 && i % 3 != 0 {
                -bf16::from_bits(bits).to_f32()
            } else {
                ((i * 7 + 3) % 19) as f32 / 256.0 + 1.0
            })
            .to_bits()
        })
        .collect();
    let weight = (0..WIDTH)
        .map(|i| bf16::from_f32(((i * 11) % 23 + 3) as f32 / 16.0).to_bits())
        .collect();
    (input, residual, weight)
}

#[test]
#[ignore = "requires H1536 cubins and a matching GPU"]
fn cached_rms_matches_generic_bits_with_residuals_aliases_and_changing_rows() {
    let (module, stream) = setup();
    for add in [false, true] {
        let generic = module
            .load_function(if add {
                "infinity_add_rms_norm"
            } else {
                "infinity_rms_norm"
            })
            .expect("load generic control");
        let cached = module
            .load_function(if add {
                "infinity_add_rms_norm_1536_cached"
            } else {
                "infinity_rms_norm_1536_cached"
            })
            .expect("load cached RMS");
        for rows in [1, 3, 7, 129] {
            for kind in 0..3 {
                let (input, residual, weight) = fixtures(rows, kind);
                for eps in [1e-6, 1e-5] {
                    let expected = run(
                        &generic, &stream, &input, &residual, &weight, rows, WIDTH, add, 0, eps,
                    );
                    assert!(expected.0.iter().all(|&v| bf16::from_bits(v).is_finite()));
                    for alias in 0..if add { 3 } else { 2 } {
                        let actual = run(
                            &cached, &stream, &input, &residual, &weight, rows, WIDTH, add, alias,
                            eps,
                        );
                        assert_eq!(
                            actual.0, expected.0,
                            "rows={rows}, kind={kind}, add={add}, alias={alias}, eps={eps}"
                        );
                        if add {
                            assert_eq!(
                                actual.1, expected.1,
                                "residual stream, rows={rows}, kind={kind}, alias={alias}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires H1536 cubins and a matching GPU"]
fn cached_rms_rejects_other_widths_and_zero_rows_without_writes() {
    let (module, stream) = setup();
    let (input, residual, weight) = fixtures(1, 0);
    for add in [false, true] {
        let cached = module
            .load_function(if add {
                "infinity_add_rms_norm_1536_cached"
            } else {
                "infinity_rms_norm_1536_cached"
            })
            .expect("load cached RMS");
        for (rows, width) in [(1, 0), (1, WIDTH - 1), (1, WIDTH + 1), (0, WIDTH)] {
            let actual = run(
                &cached, &stream, &input, &residual, &weight, rows, width, add, 0, 1e-6,
            );
            assert!(actual.0.iter().all(|&v| v == SENTINEL));
            assert!(actual.1.iter().all(|&v| v == SENTINEL));
        }
    }
}
