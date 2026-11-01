// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Independent native-C++ controls and bounded GPU tests for the three new
//! Qwen3 BF16 primitives. The native Q/K reference is the ORIGINAL published
//! qk_norm_rope.cu compiled unchanged; the width-1024 reference instantiates
//! the ORIGINAL public Candle ln_fwd_kernel<1024, 4x1>. Every GPU test is
//! ignored and only creates a Context on GPU0 inside its ignored body.

use cuda_core::{launch_kernel, Context, DeviceBuffer, Function, Module, Stream};
use half::bf16;
use libloading::{Library, Symbol};
use serde_json::Value;
use std::{ffi::c_void, fs, path::PathBuf, sync::Arc};

const WIDTH: usize = 1024;
const HEAD: usize = 128;
const ROT: usize = 64;
const GUARD: usize = 16;
const SENTINEL: u16 = 0x4242;
const RMS_ROWS: [usize; 7] = [0, 1, 3, 4, 5, 17, 128];
const EPSILONS: [f32; 2] = [1e-5, 1e-12];

fn env_path(name: &str) -> PathBuf {
    let value = std::env::var_os(name)
        .filter(|path| !path.is_empty())
        .unwrap_or_else(|| panic!("set {name}"));
    PathBuf::from(value)
}

fn setup() -> (Arc<Module>, Arc<Stream>) {
    let root = env_path("INFINITY_OXIDE_KERNEL_MODULE");
    let context = Context::new(0).expect("create CUDA context on GPU0");
    let stream = context.new_stream().expect("create CUDA stream");
    let path = if root.is_dir() {
        let (major, minor) = context
            .compute_capability()
            .expect("query GPU architecture");
        root.join(format!("sm_{major}{minor}"))
            .join("kernels.cubin")
    } else {
        root
    };
    let image = fs::read(&path)
        .unwrap_or_else(|error| panic!("read kernel bundle {}: {error}", path.display()));
    let module = context
        .load_module_from_image(&image)
        .expect("load Qwen3 kernel bundle");
    (module, stream)
}

fn setup_with_native_qk() -> (Arc<Module>, Arc<Module>, Arc<Stream>) {
    let root = env_path("INFINITY_OXIDE_KERNEL_MODULE");
    let context = Context::new(0).expect("create CUDA context on GPU0");
    let stream = context.new_stream().expect("create CUDA stream");
    let (major, minor) = context
        .compute_capability()
        .expect("query GPU architecture");
    let arch = format!("sm_{major}{minor}");
    let path = if root.is_dir() {
        root.join(&arch).join("kernels.cubin")
    } else {
        root
    };
    let image = fs::read(&path)
        .unwrap_or_else(|error| panic!("read kernel bundle {}: {error}", path.display()));
    let module = context
        .load_module_from_image(&image)
        .expect("load Qwen3 kernel bundle");
    let native_path = native_qk_cubin(&arch);
    let native_image = fs::read(&native_path)
        .unwrap_or_else(|error| panic!("read native QK cubin {}: {error}", native_path.display()));
    let native = context
        .load_module_from_image(&native_image)
        .expect("load original native QK reference");
    (module, native, stream)
}

fn native_report() -> Value {
    let path = env_path("INFINITY_OXIDE_QWEN3_NATIVE_REFERENCE");
    let bytes = fs::read(&path)
        .unwrap_or_else(|error| panic!("read native report {}: {error}", path.display()));
    serde_json::from_slice(&bytes).expect("parse native reference report")
}

fn report_path(report: &Value, section: &str, key: &str) -> PathBuf {
    let value = report
        .get(section)
        .and_then(|entry| entry.get(key))
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("native report lacks {section}.{key}"));
    PathBuf::from(value)
}

fn native_qk_cubin(arch: &str) -> PathBuf {
    let report = native_report();
    let entries = report
        .get("qk_reference")
        .and_then(|entry| entry.get("architectures"))
        .and_then(Value::as_array)
        .expect("native report qk_reference.architectures");
    for entry in entries {
        if entry.get("architecture").and_then(Value::as_str) == Some(arch) {
            let path = entry
                .get("cubin")
                .and_then(Value::as_str)
                .expect("native QK cubin path");
            return PathBuf::from(path);
        }
    }
    panic!("native report lacks QK cubin for {arch}");
}

fn native_rms_library_path() -> PathBuf {
    let report = native_report();
    report_path(&report, "rms1024_reference", "shared_library")
}

fn arg<T>(value: &mut T) -> *mut c_void {
    (value as *mut T).cast()
}

fn upload(stream: &Stream, values: &[bf16]) -> DeviceBuffer<bf16> {
    DeviceBuffer::from_host(stream, values).expect("upload BF16 fixture")
}

fn download(stream: &Stream, buffer: &DeviceBuffer<bf16>) -> Vec<bf16> {
    let mut values = vec![bf16::ZERO; buffer.len()];
    buffer
        .copy_to_host(stream, &mut values)
        .expect("download and drain stream");
    values
}

struct Guarded {
    buffer: DeviceBuffer<bf16>,
    offset: usize,
    len: usize,
}

impl Guarded {
    fn with_extra(stream: &Stream, values: &[bf16], extra: usize) -> Self {
        let offset = GUARD + extra;
        let mut data = vec![bf16::from_bits(SENTINEL); offset + values.len() + GUARD];
        data[offset..offset + values.len()].copy_from_slice(values);
        Self {
            buffer: upload(stream, &data),
            offset,
            len: values.len(),
        }
    }

    fn empty(stream: &Stream, len: usize, extra: usize) -> Self {
        Self::with_extra(stream, &vec![bf16::from_bits(SENTINEL); len], extra)
    }

    fn ptr(&self) -> u64 {
        self.buffer.cu_deviceptr() + (2 * self.offset) as u64
    }

    fn bits(&self, stream: &Stream) -> Vec<u16> {
        let values = download(stream, &self.buffer);
        assert!(values[..self.offset]
            .iter()
            .all(|value| value.to_bits() == SENTINEL));
        assert!(values[self.offset + self.len..]
            .iter()
            .all(|value| value.to_bits() == SENTINEL));
        values[self.offset..self.offset + self.len]
            .iter()
            .map(|value| value.to_bits())
            .collect()
    }
}

fn launch(
    function: &Function,
    stream: &Stream,
    blocks: usize,
    threads: usize,
    args: &mut [*mut c_void],
) {
    unsafe {
        launch_kernel(
            function.cu_function(),
            (
                u32::try_from(blocks.max(1)).expect("fixture grid fits u32"),
                1,
                1,
            ),
            (
                u32::try_from(threads).expect("fixture block fits u32"),
                1,
                1,
            ),
            0,
            stream.cu_stream(),
            args,
        )
    }
    .expect("launch Qwen3 primitive");
}

struct NativeRms {
    library: Library,
}

impl NativeRms {
    fn new() -> Self {
        let path = native_rms_library_path();
        let library = unsafe { Library::new(&path) }
            .unwrap_or_else(|error| panic!("load native RMS {}: {error}", path.display()));
        Self { library }
    }

    fn plain(
        &self,
        input: u64,
        weight: u64,
        out: u64,
        rows: u32,
        eps: f32,
        stream: &Stream,
    ) -> i32 {
        type Entry = unsafe extern "C" fn(
            *const c_void,
            *const c_void,
            *mut c_void,
            u32,
            u32,
            f32,
            *mut c_void,
        ) -> i32;
        let entry: Symbol<Entry> = unsafe {
            self.library
                .get(b"qwen3_native_rms1024\0")
                .expect("native RMS symbol")
        };
        unsafe {
            entry(
                input as *const c_void,
                weight as *const c_void,
                out as *mut c_void,
                rows,
                WIDTH as u32,
                eps,
                stream.cu_stream() as *mut c_void,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn add(
        &self,
        input: u64,
        residual: u64,
        weight: u64,
        out: u64,
        summed: u64,
        rows: u32,
        eps: f32,
        stream: &Stream,
    ) -> i32 {
        type Entry = unsafe extern "C" fn(
            *const c_void,
            *const c_void,
            *const c_void,
            *mut c_void,
            *mut c_void,
            u32,
            u32,
            f32,
            *mut c_void,
        ) -> i32;
        let entry: Symbol<Entry> = unsafe {
            self.library
                .get(b"qwen3_native_add_rms1024\0")
                .expect("native add RMS symbol")
        };
        unsafe {
            entry(
                input as *const c_void,
                residual as *const c_void,
                weight as *const c_void,
                out as *mut c_void,
                summed as *mut c_void,
                rows,
                WIDTH as u32,
                eps,
                stream.cu_stream() as *mut c_void,
            )
        }
    }
}

fn align_up(value: usize, align: usize) -> usize {
    value.div_ceil(align) * align
}

#[derive(Clone, Copy, Debug)]
struct QkLayout {
    qh: usize,
    kh: usize,
    tokens: usize,
    q_off: usize,
    k_off: usize,
    v_off: usize,
    q_span: usize,
    k_span: usize,
    total: usize,
}

fn qk_layout(
    qh: usize,
    kh: usize,
    tokens: usize,
    q_pad: usize,
    k_pad: usize,
    q_unalign: usize,
    k_unalign: usize,
) -> QkLayout {
    let q_off = GUARD + q_unalign;
    let q_span = qh * HEAD + q_pad;
    let q_end = q_off + tokens * q_span;
    let k_off = align_up(q_end + 5, 8) + k_unalign;
    let k_span = kh * HEAD + k_pad;
    let k_end = k_off + tokens * k_span;
    let v_off = align_up(k_end + 5, 8);
    let v_span = kh * HEAD;
    let total = v_off + tokens * v_span + GUARD;
    QkLayout {
        qh,
        kh,
        tokens,
        q_off,
        k_off,
        v_off,
        q_span,
        k_span,
        total,
    }
}

fn qk_value(token: usize, head: usize, column: usize, salt: usize) -> bf16 {
    let raw = ((token * 131 + head * 29 + column * 17 + salt * 73) % 509) as f32 / 64.0 - 3.953_125;
    bf16::from_f32(raw)
}

fn fill_qk(layout: &QkLayout) -> Vec<bf16> {
    let mut values = vec![bf16::from_bits(SENTINEL); layout.total];
    for token in 0..layout.tokens {
        for head in 0..layout.qh {
            for column in 0..HEAD {
                let index = layout.q_off + token * layout.q_span + head * HEAD + column;
                values[index] = qk_value(token, head, column, 1);
            }
        }
        for head in 0..layout.kh {
            for column in 0..HEAD {
                let index = layout.k_off + token * layout.k_span + head * HEAD + column;
                values[index] = qk_value(token, head, column, 7);
            }
        }
        for head in 0..layout.kh {
            for column in 0..HEAD {
                let index = layout.v_off + token * layout.kh * HEAD + head * HEAD + column;
                values[index] = qk_value(token, head, column, 13);
            }
        }
    }
    values
}

// Tiny-only heads expose non-FTZ arithmetic; the other heads exercise
// cancellation and BF16 rounding boundaries without overflowing FP32 variance.
fn fill_adversarial_qk(layout: &QkLayout, host: &mut [bf16]) {
    const TINY: [u16; 8] = [0, 0x8000, 1, 0x8001, 0x007f, 0x807f, 0x0080, 0x8080];
    const LARGE: [u16; 8] = [0x4cbe, 0xccbe, 0x4780, 0xc780, 0x3f80, 0xbf80, 0, 0x8000];
    const NEAR: [u16; 8] = [
        0x3f7f, 0x3f80, 0x3f81, 0x3f82, 0x3f80, 0x3f7f, 0x3f81, 0x3f80,
    ];
    for (kind, offset, span, heads) in [
        (0usize, layout.q_off, layout.q_span, layout.qh),
        (1usize, layout.k_off, layout.k_span, layout.kh),
    ] {
        for token in 0..layout.tokens {
            for head in 0..heads {
                let pattern = match (token + head + kind) % 3 {
                    0 => &TINY,
                    1 => &LARGE,
                    _ => &NEAR,
                };
                for column in 0..HEAD {
                    let index = offset + token * span + head * HEAD + column;
                    host[index] = bf16::from_bits(pattern[(column * 3 + token + head) % 8]);
                }
            }
        }
    }
}

fn qk_rows(layout: &QkLayout, host: &[bf16], is_q: bool) -> Vec<bf16> {
    let (offset, span, heads) = if is_q {
        (layout.q_off, layout.q_span, layout.qh)
    } else {
        (layout.k_off, layout.k_span, layout.kh)
    };
    let mut rows = Vec::with_capacity(layout.tokens * heads * HEAD);
    for token in 0..layout.tokens {
        for head in 0..heads {
            let start = offset + token * span + head * HEAD;
            rows.extend_from_slice(&host[start..start + HEAD]);
        }
    }
    rows
}

fn native_qk_strides(qh: usize, kh: usize) -> (i32, i32) {
    (
        i32::try_from(qh * HEAD).expect("Q token stride fits i32"),
        i32::try_from(kh * HEAD).expect("K token stride fits i32"),
    )
}

fn assert_storage_bits(label: &str, got: &[u16], expected: &[u16]) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{label}: storage lengths got {} expected {}",
        got.len(),
        expected.len()
    );
    let mismatches = got
        .iter()
        .zip(expected)
        .filter(|(got, expected)| got != expected)
        .count();
    if mismatches == 0 {
        return;
    }
    let mut message = format!(
        "{label}: {mismatches}/{} storage elements differ",
        got.len()
    );
    for (index, (got, expected)) in got
        .iter()
        .zip(expected)
        .enumerate()
        .filter(|(_, (got, expected))| got != expected)
        .take(8)
    {
        message.push_str(&format!(
            "; [{index}] got={got}(0x{got:04x},{}) expected={expected}(0x{expected:04x},{})",
            bf16::from_bits(*got).to_f32(),
            bf16::from_bits(*expected).to_f32()
        ));
    }
    panic!("{message}");
}

fn gamma_values(seed: usize) -> Vec<bf16> {
    (0..HEAD)
        .map(|index| {
            let raw = 0.25 + ((index * seed + seed * 11) % 37) as f32 / 16.0 - 1.156_25;
            bf16::from_f32(raw)
        })
        .collect()
}

fn gather_bf16_table(table: &[f32], positions: &[u32], table_stride: usize) -> Vec<bf16> {
    let mut gathered = Vec::with_capacity(positions.len() * ROT);
    for &position in positions {
        let start = position as usize * table_stride;
        assert!(start + ROT <= table.len(), "position/table bound");
        gathered.extend(
            table[start..start + ROT]
                .iter()
                .map(|value| bf16::from_f32(*value)),
        );
    }
    gathered
}

fn scatter_native_qk(expected: &mut [bf16], native: &[u16], layout: &QkLayout) {
    let qrows = layout.tokens * layout.qh;
    let rows = qrows + layout.tokens * layout.kh;
    assert_eq!(native.len(), rows * HEAD);
    for row in 0..rows {
        let (base, span, heads, r) = if row < qrows {
            (layout.q_off, layout.q_span, layout.qh, row)
        } else {
            (layout.k_off, layout.k_span, layout.kh, row - qrows)
        };
        let token = r / heads;
        let head = r % heads;
        let start = base + token * span + head * HEAD;
        for column in 0..HEAD {
            expected[start + column] = bf16::from_bits(native[row * HEAD + column]);
        }
    }
}

fn rms_values(elements: usize, seed: usize, adversarial: bool) -> Vec<bf16> {
    const BITS: [u16; 16] = [
        0x0000, 0x8000, 0x0001, 0x8001, 0x007f, 0x807f, 0x0080, 0x8080, 0x3f7f, 0x3f80, 0x3f81,
        0xbf80, 0x4780, 0xc780, 0x4cbe, 0xccbe,
    ];
    (0..elements)
        .map(|index| {
            if adversarial {
                bf16::from_bits(BITS[(index * 7 + seed * 3 + index / WIDTH) % BITS.len()])
            } else {
                bf16::from_f32((((index * 37 + seed * 19) % 511) as f32 - 255.0) / 128.0)
            }
        })
        .collect()
}

fn rms_weight(seed: usize) -> Vec<bf16> {
    (0..WIDTH)
        .map(|column| {
            let raw = 0.5 + ((column * 11 + seed * 7) % 23) as f32 / 16.0;
            bf16::from_f32(raw)
        })
        .collect()
}

#[derive(Clone, Copy)]
struct PlainOffsets {
    input: usize,
    weight: usize,
    out: usize,
}

#[allow(clippy::too_many_arguments)]
fn compare_plain(
    module: &Arc<Module>,
    native: &NativeRms,
    stream: &Stream,
    rows: usize,
    eps: f32,
    adversarial: bool,
    offsets: PlainOffsets,
    alias_input: bool,
) {
    let elements = rows * WIDTH;
    let input = rms_values(elements, 1, adversarial);
    let weight = rms_weight(3);
    let native_input = upload(stream, &input);
    let native_weight = upload(stream, &weight);
    let native_out = Guarded::empty(stream, elements, 0);
    let status = native.plain(
        native_input.cu_deviceptr(),
        native_weight.cu_deviceptr(),
        native_out.ptr(),
        rows as u32,
        eps,
        stream,
    );
    assert_eq!(status, 0, "native plain RMS status rows{rows} eps{eps}");
    let expected = native_out.bits(stream);

    let input_guarded = Guarded::with_extra(stream, &input, offsets.input);
    let weight_guarded = Guarded::with_extra(stream, &weight, offsets.weight);
    let out_guarded = Guarded::empty(stream, elements, offsets.out);
    let function = module
        .load_function("infinity_rms_norm_1024_exact_vector")
        .expect("load infinity_rms_norm_1024_exact_vector");
    let mut input_ptr = input_guarded.ptr();
    let mut weight_ptr = weight_guarded.ptr();
    let mut out_ptr = if alias_input {
        input_ptr
    } else {
        out_guarded.ptr()
    };
    let mut count = u32::try_from(rows).expect("rows fit u32");
    let mut width = WIDTH as u32;
    let mut epsilon = eps;
    launch(
        &function,
        stream,
        rows.div_ceil(4),
        128,
        &mut [
            arg(&mut input_ptr),
            arg(&mut weight_ptr),
            arg(&mut out_ptr),
            arg(&mut count),
            arg(&mut width),
            arg(&mut epsilon),
        ],
    );
    let got = if alias_input {
        input_guarded.bits(stream)
    } else {
        out_guarded.bits(stream)
    };
    assert_eq!(
        got,
        expected,
        "plain RMS rows{rows} eps{eps} adversarial{adversarial} offsets{:?} alias{alias_input}",
        offsets.input + offsets.weight * 10 + offsets.out * 100
    );
}

#[derive(Clone, Copy)]
struct AddOffsets {
    input: usize,
    residual: usize,
    weight: usize,
    out: usize,
    summed: usize,
}

#[allow(clippy::too_many_arguments)]
fn compare_add(
    module: &Arc<Module>,
    native: &NativeRms,
    stream: &Stream,
    rows: usize,
    eps: f32,
    adversarial: bool,
    offsets: AddOffsets,
    swapped: bool,
) {
    let elements = rows * WIDTH;
    let input = rms_values(elements, 2, adversarial);
    let residual = rms_values(elements, 5, adversarial);
    let weight = rms_weight(9);
    let native_input = upload(stream, &input);
    let native_residual = upload(stream, &residual);
    let native_weight = upload(stream, &weight);
    let native_out = Guarded::empty(stream, elements, 0);
    let native_summed = Guarded::empty(stream, elements, 0);
    let status = native.add(
        native_input.cu_deviceptr(),
        native_residual.cu_deviceptr(),
        native_weight.cu_deviceptr(),
        native_out.ptr(),
        native_summed.ptr(),
        rows as u32,
        eps,
        stream,
    );
    assert_eq!(status, 0, "native add RMS status rows{rows} eps{eps}");
    let expected_out = native_out.bits(stream);
    let expected_summed = native_summed.bits(stream);

    let input_guarded = Guarded::with_extra(stream, &input, offsets.input);
    let residual_guarded = Guarded::with_extra(stream, &residual, offsets.residual);
    let weight_guarded = Guarded::with_extra(stream, &weight, offsets.weight);
    let out_guarded = Guarded::empty(stream, elements, offsets.out);
    let summed_guarded = Guarded::empty(stream, elements, offsets.summed);
    let function = module
        .load_function("infinity_add_rms_norm_1024_exact_vector")
        .expect("load infinity_add_rms_norm_1024_exact_vector");
    let mut input_ptr = input_guarded.ptr();
    let mut residual_ptr = residual_guarded.ptr();
    let mut weight_ptr = weight_guarded.ptr();
    let (mut out_ptr, mut summed_ptr) = if swapped {
        (residual_ptr, input_ptr)
    } else {
        (out_guarded.ptr(), summed_guarded.ptr())
    };
    let mut count = u32::try_from(rows).expect("rows fit u32");
    let mut width = WIDTH as u32;
    let mut epsilon = eps;
    launch(
        &function,
        stream,
        rows.div_ceil(4),
        128,
        &mut [
            arg(&mut input_ptr),
            arg(&mut residual_ptr),
            arg(&mut weight_ptr),
            arg(&mut out_ptr),
            arg(&mut summed_ptr),
            arg(&mut count),
            arg(&mut width),
            arg(&mut epsilon),
        ],
    );
    let (got_out, got_summed) = if swapped {
        (residual_guarded.bits(stream), input_guarded.bits(stream))
    } else {
        (out_guarded.bits(stream), summed_guarded.bits(stream))
    };
    assert_eq!(
        got_out, expected_out,
        "add RMS normalized rows{rows} eps{eps} adversarial{adversarial} swapped{swapped}"
    );
    assert_eq!(
        got_summed, expected_summed,
        "add RMS summed rows{rows} eps{eps} adversarial{adversarial} swapped{swapped}"
    );
}

struct QkCase {
    qh: usize,
    kh: usize,
    tokens: usize,
    q_pad: usize,
    k_pad: usize,
    table_stride: usize,
    table_seed: usize,
    unaligned: usize,
    adversarial: bool,
}

fn run_qk_case(module: &Arc<Module>, native: &Arc<Module>, stream: &Stream, case: &QkCase) {
    let q_unalign = usize::from(case.unaligned == 0);
    let k_unalign = usize::from(case.unaligned == 1);
    let layout = qk_layout(
        case.qh,
        case.kh,
        case.tokens,
        case.q_pad,
        case.k_pad,
        q_unalign,
        k_unalign,
    );
    let mut host = fill_qk(&layout);
    if case.adversarial {
        fill_adversarial_qk(&layout, &mut host);
    }
    let q_rows = qk_rows(&layout, &host, true);
    let k_rows = qk_rows(&layout, &host, false);
    let qgamma = gamma_values(3);
    let kgamma = gamma_values(5);
    let table_rows = case.tokens * 2 + 3;
    let positions = (0..case.tokens)
        .map(|token| ((token * 7 + case.table_seed * 3 + 3) % table_rows) as u32)
        .collect::<Vec<_>>();
    let cos_table = (0..table_rows * case.table_stride)
        .map(|index| (((index * 37 + case.table_seed * 11) as f64) * 0.017).cos() as f32)
        .collect::<Vec<_>>();
    let sin_table = (0..table_rows * case.table_stride)
        .map(|index| (((index * 29 + case.table_seed * 5) as f64) * 0.013).sin() as f32)
        .collect::<Vec<_>>();
    let cos_bf16 = gather_bf16_table(&cos_table, &positions, case.table_stride);
    let sin_bf16 = gather_bf16_table(&sin_table, &positions, case.table_stride);

    let native_q = upload(stream, &q_rows);
    let native_k = upload(stream, &k_rows);
    let native_qg = upload(stream, &qgamma);
    let native_kg = upload(stream, &kgamma);
    let native_cos = upload(stream, &cos_bf16);
    let native_sin = upload(stream, &sin_bf16);
    let rows = case.tokens * (case.qh + case.kh);
    let native_out = Guarded::empty(stream, rows * HEAD, 0);
    let native_function = native
        .load_function("qk_norm_rope_bf16")
        .expect("load original native qk_norm_rope_bf16");
    let mut nq = native_q.cu_deviceptr();
    let mut nk = native_k.cu_deviceptr();
    let mut nqg = native_qg.cu_deviceptr();
    let mut nkg = native_kg.cu_deviceptr();
    let mut ncos = native_cos.cu_deviceptr();
    let mut nsin = native_sin.cu_deviceptr();
    let mut nout = native_out.ptr();
    let mut ntokens = i32::try_from(case.tokens).expect("tokens fit i32");
    let mut nqh = i32::try_from(case.qh).expect("q heads fit i32");
    let mut nkh = i32::try_from(case.kh).expect("k heads fit i32");
    let (mut nqs, mut nks) = native_qk_strides(case.qh, case.kh);
    let mut neps = 1e-5f32;
    launch(
        &native_function,
        stream,
        rows.div_ceil(8),
        128,
        &mut [
            arg(&mut nq),
            arg(&mut nk),
            arg(&mut nqg),
            arg(&mut nkg),
            arg(&mut ncos),
            arg(&mut nsin),
            arg(&mut nout),
            arg(&mut ntokens),
            arg(&mut nqh),
            arg(&mut nkh),
            arg(&mut nqs),
            arg(&mut nks),
            arg(&mut neps),
        ],
    );
    let native_bits = native_out.bits(stream);
    let mut expected = host.clone();
    scatter_native_qk(&mut expected, &native_bits, &layout);
    if rows > 0 {
        assert_ne!(expected, host, "Q/K fixture must change logical values");
    }

    let buffer = upload(stream, &host);
    let qgamma_guarded = Guarded::with_extra(stream, &qgamma, usize::from(case.unaligned == 2));
    let kgamma_guarded = Guarded::with_extra(stream, &kgamma, usize::from(case.unaligned == 3));
    let cos_device = DeviceBuffer::from_host(stream, &cos_table).expect("upload FP32 cosine table");
    let sin_device = DeviceBuffer::from_host(stream, &sin_table).expect("upload FP32 sine table");
    let positions_device = DeviceBuffer::from_host(stream, &positions).expect("upload positions");
    let mut q_ptr = buffer.cu_deviceptr() + (2 * layout.q_off) as u64;
    let mut k_ptr = buffer.cu_deviceptr() + (2 * layout.k_off) as u64;
    let mut qg_ptr = qgamma_guarded.ptr();
    let mut kg_ptr = kgamma_guarded.ptr();
    let mut cos_ptr = cos_device.cu_deviceptr();
    let mut sin_ptr = sin_device.cu_deviceptr();
    let mut pos_ptr = positions_device.cu_deviceptr();
    let mut tokens = u32::try_from(case.tokens).expect("tokens fit u32");
    let mut qh = u32::try_from(case.qh).expect("q heads fit u32");
    let mut kh = u32::try_from(case.kh).expect("k heads fit u32");
    let mut qstride = layout.q_span as u64;
    let mut kstride = layout.k_span as u64;
    let mut table_stride = case.table_stride as u32;
    let mut eps = 1e-5f32;
    let function = module
        .load_function("infinity_qk_norm_rope_128_bf16")
        .expect("load infinity_qk_norm_rope_128_bf16");
    launch(
        &function,
        stream,
        rows.div_ceil(8),
        128,
        &mut [
            arg(&mut q_ptr),
            arg(&mut k_ptr),
            arg(&mut qg_ptr),
            arg(&mut kg_ptr),
            arg(&mut cos_ptr),
            arg(&mut sin_ptr),
            arg(&mut pos_ptr),
            arg(&mut tokens),
            arg(&mut qh),
            arg(&mut kh),
            arg(&mut qstride),
            arg(&mut kstride),
            arg(&mut table_stride),
            arg(&mut eps),
        ],
    );
    let got = download(stream, &buffer);
    let got = got.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
    let expected = expected
        .iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>();
    assert_storage_bits(
        &format!(
            "QK norm+RoPE qh{} kh{} tokens{} qpad{} kpad{} stride{} table{} unaligned{}",
            case.qh,
            case.kh,
            case.tokens,
            case.q_pad,
            case.k_pad,
            case.table_stride,
            case.table_seed,
            case.unaligned
        ),
        &got,
        &expected,
    );
}

#[test]
fn qk_fixture_layout_retains_guards_padding_v_and_scatter_targets() {
    let layout = qk_layout(4, 1, 3, 3, 7, 1, 1);
    assert_eq!(layout.q_off, GUARD + 1);
    assert_eq!(layout.k_off % 2, 1, "k pointer is 2-byte unaligned");
    assert_eq!(layout.q_span, 4 * HEAD + 3);
    assert_eq!(layout.k_span, HEAD + 7);
    assert!(layout.k_off >= layout.q_off + layout.tokens * layout.q_span);
    assert!(layout.v_off >= layout.k_off + layout.tokens * layout.k_span);
    let host = fill_qk(&layout);
    assert_eq!(host.len(), layout.total);
    assert!(host[..GUARD].iter().all(|v| v.to_bits() == SENTINEL));
    assert!(host[layout.total - GUARD..]
        .iter()
        .all(|v| v.to_bits() == SENTINEL));
    for token in 0..layout.tokens {
        for column in layout.qh * HEAD..layout.q_span {
            let index = layout.q_off + token * layout.q_span + column;
            assert_eq!(host[index].to_bits(), SENTINEL, "Q row padding");
        }
        for column in layout.kh * HEAD..layout.k_span {
            let index = layout.k_off + token * layout.k_span + column;
            assert_eq!(host[index].to_bits(), SENTINEL, "K row padding");
        }
    }
    let rows = layout.tokens * (layout.qh + layout.kh);
    let native = (0..rows * HEAD)
        .map(|index| 0x3f80u16.wrapping_add(index as u16))
        .collect::<Vec<_>>();
    let mut expected = host.clone();
    scatter_native_qk(&mut expected, &native, &layout);
    for row in 0..rows {
        let (base, span, heads) = if row < layout.tokens * layout.qh {
            (layout.q_off, layout.q_span, layout.qh)
        } else {
            (layout.k_off, layout.k_span, layout.kh)
        };
        let r = if row < layout.tokens * layout.qh {
            row
        } else {
            row - layout.tokens * layout.qh
        };
        let token = r / heads;
        let head = r % heads;
        let start = base + token * span + head * HEAD;
        for column in 0..HEAD {
            assert_eq!(
                expected[start + column].to_bits(),
                native[row * HEAD + column]
            );
        }
    }
    for token in 0..layout.tokens {
        for head in 0..layout.kh {
            let start = layout.v_off + (token * layout.kh + head) * HEAD;
            assert_eq!(&expected[start..start + HEAD], &host[start..start + HEAD]);
        }
    }
}

#[test]
fn native_packed_qk_rows_use_head_scaled_token_stride() {
    let layout = qk_layout(4, 2, 3, 3, 7, 1, 1);
    let host = fill_qk(&layout);
    let q_rows = qk_rows(&layout, &host, true);
    let k_rows = qk_rows(&layout, &host, false);
    assert_eq!(q_rows.len(), layout.tokens * layout.qh * HEAD);
    assert_eq!(k_rows.len(), layout.tokens * layout.kh * HEAD);
    let (qstride, kstride) = native_qk_strides(layout.qh, layout.kh);
    assert_eq!(qstride as usize, layout.qh * HEAD);
    assert_eq!(kstride as usize, layout.kh * HEAD);
    assert_ne!(qstride as usize, HEAD);
    assert_ne!(kstride as usize, HEAD);
    for token in 0..layout.tokens {
        for head in 0..layout.qh {
            for column in 0..HEAD {
                let packed = (token * layout.qh + head) * HEAD + column;
                let padded = layout.q_off + token * layout.q_span + head * HEAD + column;
                assert_eq!(q_rows[packed].to_bits(), host[padded].to_bits());
                assert_eq!(token * qstride as usize + head * HEAD + column, packed);
            }
        }
        for head in 0..layout.kh {
            for column in 0..HEAD {
                let packed = (token * layout.kh + head) * HEAD + column;
                let padded = layout.k_off + token * layout.k_span + head * HEAD + column;
                assert_eq!(k_rows[packed].to_bits(), host[padded].to_bits());
                assert_eq!(token * kstride as usize + head * HEAD + column, packed);
            }
        }
    }
}

#[test]
fn table_gather_uses_positions_nonmonotonic_and_bf16_rounding() {
    let table_stride = 71usize;
    let table = (0..7 * table_stride)
        .map(|index| ((index * 37 + 11) as f32 * 0.000_1).sin() + 0.062_5)
        .collect::<Vec<f32>>();
    let positions = [4u32, 0, 6, 4, 1];
    let gathered = gather_bf16_table(&table, &positions, table_stride);
    assert!(table_stride > ROT);
    assert_eq!(gathered.len(), positions.len() * ROT);
    for (token, &position) in positions.iter().enumerate() {
        for column in 0..ROT {
            let expected = bf16::from_f32(table[position as usize * table_stride + column]);
            assert_eq!(gathered[token * ROT + column].to_bits(), expected.to_bits());
        }
    }
    let adversarial = rms_values(32, 1, true);
    assert!(adversarial.iter().all(|value| value.is_finite()));
    assert!(adversarial
        .iter()
        .any(|value| value.to_f32().abs() < f32::MIN_POSITIVE && value.to_bits() != 0));
    assert!(adversarial
        .iter()
        .any(|value| value.to_bits() == 0x0000 || value.to_bits() == 0x8000));
}

#[test]
#[ignore = "requires CUDA; loads INFINITY_OXIDE_KERNEL_MODULE and native report"]
fn rms1024_exact_vector_matches_original_candle_native_bits() {
    let (module, stream) = setup();
    let native = NativeRms::new();
    for &rows in &RMS_ROWS {
        for &eps in &EPSILONS {
            for adversarial in [false, true] {
                compare_plain(
                    &module,
                    &native,
                    &stream,
                    rows,
                    eps,
                    adversarial,
                    PlainOffsets {
                        input: 0,
                        weight: 0,
                        out: 0,
                    },
                    false,
                );
            }
        }
    }
}

#[test]
#[ignore = "requires CUDA; loads INFINITY_OXIDE_KERNEL_MODULE and native report"]
fn add_rms1024_exact_vector_keeps_unrounded_stats_and_bf16_summed_bits() {
    let (module, stream) = setup();
    let native = NativeRms::new();
    for &rows in &RMS_ROWS {
        for &eps in &EPSILONS {
            for adversarial in [false, true] {
                compare_add(
                    &module,
                    &native,
                    &stream,
                    rows,
                    eps,
                    adversarial,
                    AddOffsets {
                        input: 0,
                        residual: 0,
                        weight: 0,
                        out: 0,
                        summed: 0,
                    },
                    false,
                );
            }
        }
    }
}

#[test]
#[ignore = "requires CUDA; loads INFINITY_OXIDE_KERNEL_MODULE and native report"]
fn rms1024_aliases_guards_and_two_byte_unaligned_pointers_match_native() {
    let (module, stream) = setup();
    let native = NativeRms::new();
    let rows = 5usize;
    let eps = 1e-5f32;
    for adversarial in [false, true] {
        for offsets in [
            PlainOffsets {
                input: 1,
                weight: 0,
                out: 0,
            },
            PlainOffsets {
                input: 0,
                weight: 1,
                out: 0,
            },
            PlainOffsets {
                input: 0,
                weight: 0,
                out: 1,
            },
        ] {
            compare_plain(
                &module,
                &native,
                &stream,
                rows,
                eps,
                adversarial,
                offsets,
                false,
            );
        }
        compare_plain(
            &module,
            &native,
            &stream,
            rows,
            eps,
            adversarial,
            PlainOffsets {
                input: 1,
                weight: 0,
                out: 0,
            },
            true,
        );
        for slot in 0..5 {
            let mut offsets = AddOffsets {
                input: 0,
                residual: 0,
                weight: 0,
                out: 0,
                summed: 0,
            };
            match slot {
                0 => offsets.input = 1,
                1 => offsets.residual = 1,
                2 => offsets.weight = 1,
                3 => offsets.out = 1,
                _ => offsets.summed = 1,
            }
            compare_add(
                &module,
                &native,
                &stream,
                rows,
                eps,
                adversarial,
                offsets,
                false,
            );
        }
        for swapped in [false, true] {
            for offsets in [
                AddOffsets {
                    input: 0,
                    residual: 0,
                    weight: 0,
                    out: 0,
                    summed: 0,
                },
                AddOffsets {
                    input: 1,
                    residual: 1,
                    weight: 0,
                    out: 0,
                    summed: 0,
                },
            ] {
                compare_add(
                    &module,
                    &native,
                    &stream,
                    rows,
                    eps,
                    adversarial,
                    offsets,
                    swapped,
                );
            }
        }
    }
}

#[test]
#[ignore = "requires CUDA; loads INFINITY_OXIDE_KERNEL_MODULE"]
fn rms1024_invalid_width_and_zero_rows_leave_outputs_untouched() {
    let (module, stream) = setup();
    let plain = module
        .load_function("infinity_rms_norm_1024_exact_vector")
        .unwrap();
    let add = module
        .load_function("infinity_add_rms_norm_1024_exact_vector")
        .unwrap();
    let input = Guarded::empty(&stream, WIDTH, 0);
    let residual = Guarded::empty(&stream, WIDTH, 0);
    let weight = Guarded::empty(&stream, WIDTH, 0);
    let out = Guarded::empty(&stream, WIDTH, 0);
    let summed = Guarded::empty(&stream, WIDTH, 0);
    let mut input_ptr = input.ptr();
    let mut residual_ptr = residual.ptr();
    let mut weight_ptr = weight.ptr();
    let mut out_ptr = out.ptr();
    let mut summed_ptr = summed.ptr();
    let mut rows = 3u32;
    let mut width = (WIDTH - 1) as u32;
    let mut eps = 1e-5f32;
    launch(
        &plain,
        &stream,
        1,
        128,
        &mut [
            arg(&mut input_ptr),
            arg(&mut weight_ptr),
            arg(&mut out_ptr),
            arg(&mut rows),
            arg(&mut width),
            arg(&mut eps),
        ],
    );
    assert!(out.bits(&stream).iter().all(|bits| *bits == SENTINEL));
    launch(
        &add,
        &stream,
        1,
        128,
        &mut [
            arg(&mut input_ptr),
            arg(&mut residual_ptr),
            arg(&mut weight_ptr),
            arg(&mut out_ptr),
            arg(&mut summed_ptr),
            arg(&mut rows),
            arg(&mut width),
            arg(&mut eps),
        ],
    );
    assert!(out.bits(&stream).iter().all(|bits| *bits == SENTINEL));
    assert!(summed.bits(&stream).iter().all(|bits| *bits == SENTINEL));

    let mut zero_rows = 0u32;
    let mut full_width = WIDTH as u32;
    launch(
        &plain,
        &stream,
        0,
        128,
        &mut [
            arg(&mut input_ptr),
            arg(&mut weight_ptr),
            arg(&mut out_ptr),
            arg(&mut zero_rows),
            arg(&mut full_width),
            arg(&mut eps),
        ],
    );
    assert!(out.bits(&stream).iter().all(|bits| *bits == SENTINEL));
}

#[test]
#[ignore = "requires CUDA; loads INFINITY_OXIDE_KERNEL_MODULE and native report"]
fn qk_norm_rope_matches_original_public_native_for_gqa_strides_and_guards() {
    let (module, native, stream) = setup_with_native_qk();
    let function = module
        .load_function("infinity_qk_norm_rope_128_bf16")
        .unwrap();
    let layout = qk_layout(1, 1, 0, 0, 0, 0, 0);
    let host = fill_qk(&layout);
    let buffer = upload(&stream, &host);
    let qgamma = gamma_values(3);
    let kgamma = gamma_values(5);
    let qgamma_device = upload(&stream, &qgamma);
    let kgamma_device = upload(&stream, &kgamma);
    let cos_table = vec![0.5f32; 65];
    let sin_table = vec![0.25f32; 65];
    let cos_device = DeviceBuffer::from_host(&stream, &cos_table).unwrap();
    let sin_device = DeviceBuffer::from_host(&stream, &sin_table).unwrap();
    let positions_device = DeviceBuffer::from_host(&stream, &[0u32]).unwrap();
    let mut q_ptr = buffer.cu_deviceptr() + (2 * layout.q_off) as u64;
    let mut k_ptr = buffer.cu_deviceptr() + (2 * layout.k_off) as u64;
    let mut qg_ptr = qgamma_device.cu_deviceptr();
    let mut kg_ptr = kgamma_device.cu_deviceptr();
    let mut cos_ptr = cos_device.cu_deviceptr();
    let mut sin_ptr = sin_device.cu_deviceptr();
    let mut pos_ptr = positions_device.cu_deviceptr();
    let mut tokens = 0u32;
    let mut qh = 1u32;
    let mut kh = 1u32;
    let mut qstride = layout.q_span as u64;
    let mut kstride = layout.k_span as u64;
    let mut table_stride = 65u32;
    let mut eps = 1e-5f32;
    launch(
        &function,
        &stream,
        0,
        128,
        &mut [
            arg(&mut q_ptr),
            arg(&mut k_ptr),
            arg(&mut qg_ptr),
            arg(&mut kg_ptr),
            arg(&mut cos_ptr),
            arg(&mut sin_ptr),
            arg(&mut pos_ptr),
            arg(&mut tokens),
            arg(&mut qh),
            arg(&mut kh),
            arg(&mut qstride),
            arg(&mut kstride),
            arg(&mut table_stride),
            arg(&mut eps),
        ],
    );
    let zero_got = download(&stream, &buffer);
    assert_eq!(
        zero_got, host,
        "zero-token launch must not touch Q/K/V/padding"
    );

    let cases = [
        QkCase {
            qh: 8,
            kh: 1,
            tokens: 1,
            q_pad: 3,
            k_pad: 7,
            table_stride: 65,
            table_seed: 1,
            unaligned: 99,
            adversarial: false,
        },
        QkCase {
            qh: 4,
            kh: 1,
            tokens: 3,
            q_pad: 0,
            k_pad: 3,
            table_stride: 71,
            table_seed: 2,
            unaligned: 99,
            adversarial: false,
        },
        QkCase {
            qh: 6,
            kh: 3,
            tokens: 1,
            q_pad: 7,
            k_pad: 0,
            table_stride: 96,
            table_seed: 3,
            unaligned: 99,
            adversarial: false,
        },
        QkCase {
            qh: 1,
            kh: 1,
            tokens: 1,
            q_pad: 0,
            k_pad: 0,
            table_stride: 65,
            table_seed: 4,
            unaligned: 99,
            adversarial: false,
        },
        QkCase {
            qh: 16,
            kh: 4,
            tokens: 7,
            q_pad: 3,
            k_pad: 7,
            table_stride: 71,
            table_seed: 5,
            unaligned: 99,
            adversarial: false,
        },
        QkCase {
            qh: 7,
            kh: 1,
            tokens: 1,
            q_pad: 0,
            k_pad: 0,
            table_stride: 96,
            table_seed: 6,
            unaligned: 99,
            adversarial: false,
        },
        QkCase {
            qh: 3,
            kh: 1,
            tokens: 2,
            q_pad: 3,
            k_pad: 7,
            table_stride: 65,
            table_seed: 7,
            unaligned: 99,
            adversarial: false,
        },
    ];
    for case in &cases {
        run_qk_case(&module, &native, &stream, case);
    }
    let mut base = QkCase {
        qh: 4,
        kh: 1,
        tokens: 3,
        q_pad: 3,
        k_pad: 7,
        table_stride: 71,
        table_seed: 8,
        unaligned: 0,
        adversarial: false,
    };
    for slot in 0..4 {
        base.unaligned = slot;
        run_qk_case(&module, &native, &stream, &base);
    }
    base.adversarial = true;
    base.qh = 8;
    base.kh = 1;
    base.tokens = 3;
    for slot in [99, 0, 1, 2, 3] {
        base.unaligned = slot;
        run_qk_case(&module, &native, &stream, &base);
    }
}
