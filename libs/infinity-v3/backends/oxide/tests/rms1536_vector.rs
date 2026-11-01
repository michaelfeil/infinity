// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! GPU-only parity controls for the H1536 BF16 vector RMSNorm exports.
//!
//! The vector exports are compared bit-for-bit with the preserved cached scalar
//! exports, including guards, alias contracts, unaligned fallback paths, and an
//! independent native NVCC reference.

use cuda_core::{launch_kernel, Context, DeviceBuffer, Function, Module, Stream};
use half::bf16;
use std::{ffi::c_void, path::PathBuf, sync::Arc};

const WIDTH: usize = 1536;
const GUARD: usize = 16;
const SENTINEL: u16 = 0x4242;
const VECTOR_BYTES: u64 = 16;

#[derive(Clone, Copy, PartialEq, Eq)]
struct Offsets {
    input: usize,
    residual: usize,
    weight: usize,
    out: usize,
    summed: usize,
}

impl Offsets {
    const ZERO: Self = Self {
        input: 0,
        residual: 0,
        weight: 0,
        out: 0,
        summed: 0,
    };
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Target {
    Input,
    Residual,
    Out,
    Summed,
}

fn setup_impl(native: bool) -> (Arc<Module>, Option<Arc<Module>>, Arc<Stream>) {
    let root = PathBuf::from(
        std::env::var_os("INFINITY_OXIDE_KERNEL_MODULE")
            .expect("set INFINITY_OXIDE_KERNEL_MODULE to the H1536 bundle"),
    );
    let context = Context::new(0).expect("create GPU0 context");
    let stream = context.new_stream().expect("create stream");
    let (major, minor) = context.compute_capability().expect("query architecture");
    let path = if root.is_dir() {
        root.join(format!("sm_{major}{minor}/kernels.cubin"))
    } else {
        root
    };
    let module = context
        .load_module_from_image(&std::fs::read(&path).expect("read cubin"))
        .expect("load cubin");
    let native_module = if native {
        let native_root = PathBuf::from(
            std::env::var_os("INFINITY_OXIDE_RMS1536_NATIVE_REFERENCE").expect(
                "set INFINITY_OXIDE_RMS1536_NATIVE_REFERENCE to the native reference directory",
            ),
        );
        let native_path = native_root.join(format!("sm_{major}{minor}/native-reference.cubin"));
        assert!(
            native_path.is_file(),
            "missing native reference cubin {native_path:?}"
        );
        Some(
            context
                .load_module_from_image(
                    &std::fs::read(&native_path).expect("read native reference cubin"),
                )
                .expect("load native reference cubin"),
        )
    } else {
        None
    };
    (module, native_module, stream)
}

fn setup() -> (Arc<Module>, Arc<Stream>) {
    let (module, _, stream) = setup_impl(false);
    (module, stream)
}

fn setup_native() -> (Arc<Module>, Arc<Module>, Arc<Stream>) {
    let (module, native, stream) = setup_impl(true);
    (module, native.expect("native module loaded"), stream)
}

fn load_rms(module: &Arc<Module>, add: bool) -> (Function, Function) {
    let suffix = if add {
        "add_rms_norm_1536"
    } else {
        "rms_norm_1536"
    };
    (
        module
            .load_function(&format!("infinity_{suffix}_vector"))
            .expect("load vector RMS"),
        module
            .load_function(&format!("infinity_{suffix}_cached"))
            .expect("load cached RMS"),
    )
}

fn arg<T>(value: &mut T) -> *mut c_void {
    (value as *mut T).cast()
}

fn guarded_at(stream: &Stream, data: &[u16], extra: usize) -> DeviceBuffer<u16> {
    let pad = GUARD + extra;
    let mut values = vec![SENTINEL; data.len() + pad + GUARD];
    values[pad..pad + data.len()].copy_from_slice(data);
    DeviceBuffer::from_host(stream, &values).expect("upload guarded storage")
}

fn pointer_at(buffer: &DeviceBuffer<u16>, extra: usize) -> u64 {
    buffer.cu_deviceptr() + (2 * (GUARD + extra)) as u64
}

#[rustfmt::skip]
fn download_padded(stream: &Stream, buffer: &DeviceBuffer<u16>, extra: usize, len: usize) -> Vec<u16> {
    let mut data = vec![0; buffer.len()];
    buffer
        .copy_to_host(stream, &mut data)
        .expect("drain stream");
    let pad = GUARD + extra;
    assert!(
        data[..pad].iter().all(|&x| x == SENTINEL),
        "leading guard modified"
    );
    assert!(
        data[pad + len..].iter().all(|&x| x == SENTINEL),
        "trailing guard modified"
    );
    data[pad..pad + len].to_vec()
}

fn check_bits(label: &str, actual: &[u16], expected: &[u16], context: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label} length mismatch ({context})"
    );
    let mut count = 0usize;
    let mut first = Vec::new();
    for (index, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        if got != want {
            count += 1;
            if first.len() < 8 {
                first.push(format!("#{index}: got={got:#06x} want={want:#06x}"));
            }
        }
    }
    assert!(
        count == 0,
        "{label} bit mismatch ({context}): {count} total, first: {}",
        first.join(", ")
    );
}

#[rustfmt::skip]
fn required_aligned(offsets: Offsets, add: bool) -> bool {
    offsets.input == 0 && offsets.weight == 0 && offsets.out == 0 && (!add || (offsets.residual == 0 && offsets.summed == 0))
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
    offsets: Offsets,
) -> (Vec<u16>, Vec<u16>) {
    let x = guarded_at(stream, input, offsets.input);
    let r = guarded_at(stream, residual, offsets.residual);
    let w = guarded_at(stream, weight, offsets.weight);
    let out = guarded_at(stream, &vec![SENTINEL; input.len()], offsets.out);
    let summed = guarded_at(stream, &vec![SENTINEL; input.len()], offsets.summed);

    let (out_target, summed_target) = match alias {
        0 => (Target::Out, Target::Summed),
        1 => (
            Target::Input,
            if add {
                Target::Residual
            } else {
                Target::Summed
            },
        ),
        2 if add => (Target::Residual, Target::Input),
        _ => panic!("invalid alias {alias}"),
    };

    let mut out_ptr = match out_target {
        Target::Input => pointer_at(&x, offsets.input),
        Target::Residual => pointer_at(&r, offsets.residual),
        Target::Out => pointer_at(&out, offsets.out),
        Target::Summed => pointer_at(&summed, offsets.summed),
    };
    let mut summed_ptr = match summed_target {
        Target::Input => pointer_at(&x, offsets.input),
        Target::Residual => pointer_at(&r, offsets.residual),
        Target::Out => pointer_at(&out, offsets.out),
        Target::Summed => pointer_at(&summed, offsets.summed),
    };

    if offsets == Offsets::ZERO {
        for pointer in [
            pointer_at(&x, 0),
            pointer_at(&r, 0),
            pointer_at(&w, 0),
            pointer_at(&out, 0),
            pointer_at(&summed, 0),
        ] {
            assert_eq!(
                pointer % VECTOR_BYTES,
                0,
                "aligned baseline must be 16-byte aligned"
            );
        }
    }

    let mut x_ptr = pointer_at(&x, offsets.input);
    let mut r_ptr = pointer_at(&r, offsets.residual);
    let mut w_ptr = pointer_at(&w, offsets.weight);
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
    // SAFETY: six/eight-argument ABI, block 256, one row per CTA plus a guard CTA.
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

    let x_data = download_padded(stream, &x, offsets.input, input.len());
    let r_data = download_padded(stream, &r, offsets.residual, input.len());
    let w_data = download_padded(stream, &w, offsets.weight, weight.len());
    let out_data = download_padded(stream, &out, offsets.out, input.len());
    let summed_data = download_padded(stream, &summed, offsets.summed, input.len());

    if out_target != Target::Input && summed_target != Target::Input {
        check_bits("input", &x_data, input, "input altered");
    }
    if out_target != Target::Residual && summed_target != Target::Residual {
        check_bits("residual", &r_data, residual, "residual altered");
    }
    check_bits("weight", &w_data, weight, "weight altered");
    if !add {
        assert!(
            summed_data.iter().all(|&v| v == SENTINEL),
            "plain summed must stay sentinel"
        );
    }

    let result = match out_target {
        Target::Input => x_data.clone(),
        Target::Residual => r_data.clone(),
        Target::Out => out_data.clone(),
        Target::Summed => summed_data.clone(),
    };
    let sum = match summed_target {
        Target::Input => x_data,
        Target::Residual => r_data,
        Target::Out => out_data,
        Target::Summed => summed_data,
    };
    (result, sum)
}

#[allow(clippy::too_many_arguments)]
fn run_fixture(
    function: &Function,
    stream: &Stream,
    fixture: &(Vec<u16>, Vec<u16>, Vec<u16>),
    rows: usize,
    width: usize,
    add: bool,
    alias: usize,
    eps: f32,
    offsets: Offsets,
) -> (Vec<u16>, Vec<u16>) {
    run(
        function, stream, &fixture.0, &fixture.1, &fixture.2, rows, width, add, alias, eps, offsets,
    )
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
                2 => f32::from_bits(([0u32, 0x8000, 1, 0x8001, 0x3f80, 0xbf80][i % 6]) << 16),
                3 => ((i * 17 + 5) % 13) as f32 / 8.0 - 0.75,
                _ => unreachable!(),
            })
            .to_bits()
        })
        .collect::<Vec<_>>();
    let residual = input
        .iter()
        .enumerate()
        .map(|(i, &bits)| {
            bf16::from_f32(match kind {
                1 if i % 3 != 0 => -bf16::from_bits(bits).to_f32(),
                3 if i % 5 != 0 => -bf16::from_bits(bits).to_f32(),
                3 => bf16::from_bits(bits).to_f32(),
                _ => ((i * 7 + 3) % 19) as f32 / 256.0 + 1.0,
            })
            .to_bits()
        })
        .collect();
    let weight = (0..WIDTH)
        .map(|i| bf16::from_f32(((i * 11) % 23 + 3) as f32 / 16.0).to_bits())
        .collect();
    (input, residual, weight)
}

#[allow(clippy::too_many_arguments)]
fn run_native(
    function: &Function,
    stream: &Stream,
    input: &[u16],
    residual: &[u16],
    weight: &[u16],
    rows: usize,
    add: bool,
    eps: f32,
) -> (Vec<u16>, Vec<u16>) {
    let x = guarded_at(stream, input, 0);
    let r = guarded_at(stream, residual, 0);
    let w = guarded_at(stream, weight, 0);
    let out = guarded_at(stream, &vec![SENTINEL; input.len()], 0);
    let summed = guarded_at(stream, &vec![SENTINEL; input.len()], 0);
    let mut x_ptr = pointer_at(&x, 0);
    let mut r_ptr = if add { pointer_at(&r, 0) } else { 0 };
    let mut w_ptr = pointer_at(&w, 0);
    let mut out_ptr = pointer_at(&out, 0);
    let mut summed_ptr = pointer_at(&summed, 0);
    for pointer in [x_ptr, w_ptr, out_ptr, summed_ptr] {
        assert_eq!(
            pointer % VECTOR_BYTES,
            0,
            "native pointers must be 16-byte aligned"
        );
    }
    if add {
        assert_eq!(
            r_ptr % VECTOR_BYTES,
            0,
            "native residual must be 16-byte aligned"
        );
    }
    let mut width = WIDTH as i64;
    let mut epsilon = eps;
    let mut args = vec![
        arg(&mut x_ptr),
        arg(&mut r_ptr),
        arg(&mut w_ptr),
        arg(&mut out_ptr),
        arg(&mut summed_ptr),
        arg(&mut width),
        arg(&mut epsilon),
    ];
    // Native ABI has no rows argument; launch exactly one CTA per row.
    unsafe {
        launch_kernel(
            function.cu_function(),
            (rows as u32, 1, 1),
            (256, 1, 1),
            0,
            stream.cu_stream(),
            &mut args,
        )
    }
    .expect("launch native RMS");
    let x_data = download_padded(stream, &x, 0, input.len());
    let r_data = download_padded(stream, &r, 0, input.len());
    let w_data = download_padded(stream, &w, 0, weight.len());
    let out_data = download_padded(stream, &out, 0, input.len());
    let summed_data = download_padded(stream, &summed, 0, input.len());
    check_bits("native input", &x_data, input, "input altered");
    check_bits("native residual", &r_data, residual, "residual altered");
    check_bits("native weight", &w_data, weight, "weight altered");
    (out_data, summed_data)
}

#[test]
#[ignore = "requires H1536 cubins and a matching GPU"]
fn vector_matches_cached_bits_across_rows_fixtures_eps_and_aliases() {
    let (module, stream) = setup();
    for add in [false, true] {
        let (vector, cached) = load_rms(&module, add);
        for rows in [1, 3, 7, 129] {
            for kind in 0..4 {
                let fixture = fixtures(rows, kind);
                for eps in [1e-6, 1e-5] {
                    for alias in 0..if add { 3 } else { 2 } {
                        let context =
                            format!("rows={rows} kind={kind} add={add} eps={eps} alias={alias}");
                        let expected = run_fixture(
                            &cached,
                            &stream,
                            &fixture,
                            rows,
                            WIDTH,
                            add,
                            alias,
                            eps,
                            Offsets::ZERO,
                        );
                        let actual = run_fixture(
                            &vector,
                            &stream,
                            &fixture,
                            rows,
                            WIDTH,
                            add,
                            alias,
                            eps,
                            Offsets::ZERO,
                        );
                        check_bits("out", &actual.0, &expected.0, &context);
                        check_bits("summed", &actual.1, &expected.1, &context);
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires H1536 cubins and a matching GPU"]
fn vector_unaligned_fallbacks_match_cached_and_preserve_guards() {
    let (module, stream) = setup();
    #[rustfmt::skip]
    let cases = [
        ("input", Offsets { input: 1, ..Offsets::ZERO }),
        ("residual", Offsets { residual: 1, ..Offsets::ZERO }),
        ("weight", Offsets { weight: 1, ..Offsets::ZERO }),
        ("out", Offsets { out: 1, ..Offsets::ZERO }),
        ("summed", Offsets { summed: 1, ..Offsets::ZERO }),
        ("all", Offsets { input: 1, residual: 1, weight: 1, out: 1, summed: 1 }),
    ];
    for add in [false, true] {
        let (vector, cached) = load_rms(&module, add);
        for rows in [1, 3] {
            for kind in [1, 3] {
                let fixture = fixtures(rows, kind);
                for eps in [1e-6, 1e-5] {
                    for &(name, offsets) in &cases {
                        if add {
                            assert!(
                                !required_aligned(offsets, add),
                                "offset case {name} must force fallback"
                            );
                        }
                        let context =
                            format!("offset={name} rows={rows} kind={kind} add={add} eps={eps}");
                        let expected = run_fixture(
                            &cached, &stream, &fixture, rows, WIDTH, add, 0, eps, offsets,
                        );
                        let actual = run_fixture(
                            &vector, &stream, &fixture, rows, WIDTH, add, 0, eps, offsets,
                        );
                        check_bits("out", &actual.0, &expected.0, &context);
                        check_bits("summed", &actual.1, &expected.1, &context);
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires H1536 cubins and a matching GPU"]
fn vector_rejects_zero_and_invalid_width_without_writes() {
    let (module, stream) = setup();
    let fixture = fixtures(1, 0);
    for add in [false, true] {
        let (vector, _) = load_rms(&module, add);
        for (rows, width) in [(1, 0), (1, WIDTH - 1), (1, WIDTH + 1), (0, WIDTH)] {
            let (out, summed) = run_fixture(
                &vector,
                &stream,
                &fixture,
                rows,
                width,
                add,
                0,
                1e-6,
                Offsets::ZERO,
            );
            let context = format!("rows={rows} width={width} add={add}");
            assert!(
                out.iter().all(|&v| v == SENTINEL),
                "invalid shape wrote out ({context})"
            );
            assert!(
                summed.iter().all(|&v| v == SENTINEL),
                "invalid shape wrote summed ({context})"
            );
        }
    }
}

#[test]
#[ignore = "requires H1536 cubins, native reference cubin, and a matching GPU"]
fn vector_matches_native_reference_on_aligned_ordinary_and_adversarial_controls() {
    let (module, native_module, stream) = setup_native();
    let native = native_module
        .load_function("infinity_native_add_rms_bf16_vector")
        .expect("load native reference kernel");
    for add in [false, true] {
        let (vector, cached) = load_rms(&module, add);
        for rows in [1, 3, 7, 129] {
            for kind in 0..4 {
                let fixture = fixtures(rows, kind);
                for eps in [1e-6, 1e-5] {
                    let context = format!("rows={rows} kind={kind} add={add} eps={eps}");
                    let expected = run_native(
                        &native, &stream, &fixture.0, &fixture.1, &fixture.2, rows, add, eps,
                    );
                    let actual = run_fixture(
                        &vector,
                        &stream,
                        &fixture,
                        rows,
                        WIDTH,
                        add,
                        0,
                        eps,
                        Offsets::ZERO,
                    );
                    let old = run_fixture(
                        &cached,
                        &stream,
                        &fixture,
                        rows,
                        WIDTH,
                        add,
                        0,
                        eps,
                        Offsets::ZERO,
                    );
                    check_bits("vector out vs native", &actual.0, &expected.0, &context);
                    check_bits("cached out vs native", &old.0, &expected.0, &context);
                    if add {
                        check_bits("vector summed vs native", &actual.1, &expected.1, &context);
                        check_bits("cached summed vs native", &old.1, &expected.1, &context);
                    }
                }
            }
        }
    }
}
