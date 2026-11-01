// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Same-input bit controls for exact H768 vector-memory normalization.

use cuda_core::{launch_kernel, Context, DeviceBuffer, Function, Module, Stream};
use half::bf16;
use std::{ffi::c_void, path::PathBuf, sync::Arc};

const WIDTH: usize = 768;
const GUARD: usize = 16;
const SENTINEL: u16 = 0x4242;
const ROWS: [usize; 7] = [0, 1, 3, 4, 5, 17, 128];

fn setup() -> (Arc<Module>, Arc<Stream>) {
    let root = std::env::var_os("INFINITY_OXIDE_KERNEL_MODULE")
        .filter(|value| !value.is_empty())
        .expect("set INFINITY_OXIDE_KERNEL_MODULE to the H768 vector bundle");
    let context = Context::new(0).expect("create CUDA context on GPU0");
    let stream = context.new_stream().expect("create CUDA stream");
    let root = PathBuf::from(root);
    let path = if root.is_dir() {
        let (major, minor) = context
            .compute_capability()
            .expect("query GPU architecture");
        root.join(format!("sm_{major}{minor}/kernels.cubin"))
    } else {
        root
    };
    let image = std::fs::read(path).expect("read H768 vector kernel bundle");
    let module = context
        .load_module_from_image(&image)
        .expect("load H768 vector kernel bundle");
    (module, stream)
}

fn arg<T>(value: &mut T) -> *mut c_void {
    (value as *mut T).cast()
}

fn launch(function: &Function, stream: &Stream, rows: usize, args: &mut [*mut c_void]) {
    // SAFETY: every caller uses the fixed export ABI, retains its allocations
    // through download, and submits only onto this module's ordered stream.
    unsafe {
        launch_kernel(
            function.cu_function(),
            (u32::try_from(rows.max(1).div_ceil(4)).unwrap(), 1, 1),
            (128, 1, 1),
            0,
            stream.cu_stream(),
            args,
        )
    }
    .expect("launch H768 normalization");
}

struct Guarded {
    buffer: DeviceBuffer<bf16>,
    offset: usize,
    len: usize,
}

impl Guarded {
    fn new(stream: &Stream, values: &[bf16], unaligned: bool) -> Self {
        let offset = GUARD + usize::from(unaligned);
        let mut data = vec![bf16::from_bits(SENTINEL); offset + values.len() + GUARD];
        data[offset..offset + values.len()].copy_from_slice(values);
        let buffer = DeviceBuffer::from_host(stream, &data).expect("upload guarded fixture");
        Self {
            buffer,
            offset,
            len: values.len(),
        }
    }

    fn output(stream: &Stream, len: usize, unaligned: bool) -> Self {
        Self::new(stream, &vec![bf16::from_bits(SENTINEL); len], unaligned)
    }

    fn ptr(&self) -> u64 {
        self.buffer.cu_deviceptr() + (2 * self.offset) as u64
    }

    fn bits(&self, stream: &Stream) -> Vec<u16> {
        let mut data = vec![bf16::ZERO; self.buffer.len()];
        self.buffer
            .copy_to_host(stream, &mut data)
            .expect("download and drain fixture");
        assert!(data[..self.offset]
            .iter()
            .all(|value| value.to_bits() == SENTINEL));
        assert!(data[self.offset + self.len..]
            .iter()
            .all(|value| value.to_bits() == SENTINEL));
        data[self.offset..self.offset + self.len]
            .iter()
            .map(|value| value.to_bits())
            .collect()
    }
}

fn values(len: usize, seed: usize, adversarial: bool) -> Vec<bf16> {
    // Finite extremes exercise FTZ, signed zero, rounding boundaries, and large
    // cancellation without overflowing the FP32 variance accumulator.
    const BITS: [u16; 16] = [
        0x0000, 0x8000, 0x0001, 0x8001, 0x007f, 0x807f, 0x0080, 0x8080, 0x3f7f, 0x3f80, 0x3f81,
        0xbf80, 0x4780, 0xc780, 0x4cbe, 0xccbe,
    ];
    (0..len)
        .map(|index| {
            if adversarial {
                bf16::from_bits(BITS[(index * 7 + seed * 3 + index / WIDTH) % BITS.len()])
            } else {
                bf16::from_f32((((index * 37 + seed * 19) % 511) as f32 - 255.0) / 128.0)
            }
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn run_layer(
    module: &Arc<Module>,
    stream: &Stream,
    rows: usize,
    adversarial: bool,
    residual: bool,
    bias: bool,
    unaligned: Option<usize>,
    alias: usize,
) {
    let scalar = module
        .load_function("infinity_layer_norm_768_exact")
        .unwrap();
    let vector = module
        .load_function("infinity_layer_norm_768_exact_vector")
        .unwrap();
    let elements = rows * WIDTH;
    let input = Guarded::new(
        stream,
        &values(elements, 1, adversarial),
        unaligned == Some(0),
    );
    let residual_buffer = Guarded::new(
        stream,
        &values(elements, 2, adversarial),
        unaligned == Some(1),
    );
    let bias_values = values(WIDTH, 3, false)
        .iter()
        .map(|value| bf16::from_f32(value.to_f32() / 64.0))
        .collect::<Vec<_>>();
    let bias_buffer = Guarded::new(stream, &bias_values, unaligned == Some(2));
    let gamma = Guarded::new(stream, &values(WIDTH, 4, false), unaligned == Some(3));
    let beta = Guarded::new(stream, &values(WIDTH, 5, false), unaligned == Some(4));
    let scalar_out = Guarded::output(stream, elements, false);
    let vector_out = Guarded::output(stream, elements, unaligned == Some(5));
    let mut input_ptr = input.ptr();
    let mut residual_ptr = if residual { residual_buffer.ptr() } else { 0 };
    let mut bias_ptr = if bias { bias_buffer.ptr() } else { 0 };
    let mut gamma_ptr = gamma.ptr();
    let mut beta_ptr = beta.ptr();
    let mut count = rows as u32;
    let mut width = WIDTH as u32;
    let mut eps = if adversarial { 1e-12f32 } else { 1e-5 };
    let mut out_ptr = scalar_out.ptr();
    launch(
        &scalar,
        stream,
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
    let expected = scalar_out.bits(stream);
    assert!(expected
        .iter()
        .all(|bits| bf16::from_bits(*bits).is_finite()));
    let output = match alias {
        0 => &vector_out,
        1 => &input,
        2 => {
            assert!(residual);
            &residual_buffer
        }
        _ => unreachable!(),
    };
    out_ptr = output.ptr();
    // Exact-base in-place writes are intentional. Each lane reads its complete
    // row fragment before writing those same columns; output aliases no weights.
    launch(
        &vector,
        stream,
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
    assert_eq!(
        output.bits(stream),
        expected,
        "rows{rows} adversarial{adversarial} residual{residual} bias{bias} unaligned{unaligned:?} alias{alias}"
    );
}

#[test]
#[ignore = "requires INFINITY_OXIDE_KERNEL_MODULE and a matching CUDA device"]
fn vector_layer_norm_768_matches_scalar_for_shapes_and_recipes() {
    let (module, stream) = setup();
    for rows in ROWS {
        for adversarial in [false, true] {
            for residual in [false, true] {
                for bias in [false, true] {
                    run_layer(&module, &stream, rows, adversarial, residual, bias, None, 0);
                }
            }
        }
    }
}

#[test]
#[ignore = "requires INFINITY_OXIDE_KERNEL_MODULE and a matching CUDA device"]
fn vector_layer_norm_768_preserves_individual_fallbacks_and_in_place_aliases() {
    let (module, stream) = setup();
    for adversarial in [false, true] {
        for slot in 0..6 {
            run_layer(&module, &stream, 5, adversarial, true, true, Some(slot), 0);
        }
        for residual in [false, true] {
            for bias in [false, true] {
                for unaligned in [None, Some(0)] {
                    run_layer(
                        &module,
                        &stream,
                        5,
                        adversarial,
                        residual,
                        bias,
                        unaligned,
                        1,
                    );
                }
                if residual {
                    for unaligned in [None, Some(1)] {
                        run_layer(&module, &stream, 5, adversarial, true, bias, unaligned, 2);
                    }
                }
            }
        }
    }
}

fn run_embedding(
    module: &Arc<Module>,
    stream: &Stream,
    rows: usize,
    adversarial: bool,
    unaligned: Option<usize>,
) {
    let scalar = module
        .load_function("infinity_embedding_norm_768_exact")
        .unwrap();
    let vector = module
        .load_function("infinity_embedding_norm_768_exact_vector")
        .unwrap();
    let ids = (0..rows)
        .map(|index| ((index * 3 + 1) % 7) as u32)
        .collect::<Vec<_>>();
    let positions = (0..rows)
        .map(|index| ((index * 5 + 3) % 11) as u32)
        .collect::<Vec<_>>();
    let types = (0..rows)
        .map(|index| (index % 2) as u32)
        .collect::<Vec<_>>();
    let indices = [&ids, &positions, &types].map(|source| {
        // Index tables need only natural u32 alignment, not vector alignment.
        let mut data = vec![0xeeee_eeeeu32];
        data.extend_from_slice(source);
        data.push(0xdddd_dddd);
        DeviceBuffer::from_host(stream, &data).unwrap()
    });
    let word = Guarded::new(
        stream,
        &values(7 * WIDTH, 1, adversarial),
        unaligned == Some(0),
    );
    let position = Guarded::new(
        stream,
        &values(11 * WIDTH, 2, adversarial),
        unaligned == Some(1),
    );
    let token_type = Guarded::new(
        stream,
        &values(2 * WIDTH, 3, adversarial),
        unaligned == Some(2),
    );
    let gamma = Guarded::new(stream, &values(WIDTH, 4, false), unaligned == Some(3));
    let beta = Guarded::new(stream, &values(WIDTH, 5, false), unaligned == Some(4));
    let scalar_out = Guarded::output(stream, rows * WIDTH, false);
    let vector_out = Guarded::output(stream, rows * WIDTH, unaligned == Some(5));
    let mut ids_ptr = indices[0].cu_deviceptr() + 4;
    let mut positions_ptr = indices[1].cu_deviceptr() + 4;
    let mut types_ptr = indices[2].cu_deviceptr() + 4;
    let mut word_ptr = word.ptr();
    let mut position_ptr = position.ptr();
    let mut token_ptr = token_type.ptr();
    let mut gamma_ptr = gamma.ptr();
    let mut beta_ptr = beta.ptr();
    let mut count = rows as u32;
    let mut width = WIDTH as u32;
    let mut eps = if adversarial { 1e-12f32 } else { 1e-5 };
    for (function, output) in [(&scalar, &scalar_out), (&vector, &vector_out)] {
        let mut out_ptr = output.ptr();
        launch(
            function,
            stream,
            rows,
            &mut [
                arg(&mut ids_ptr),
                arg(&mut positions_ptr),
                arg(&mut types_ptr),
                arg(&mut word_ptr),
                arg(&mut position_ptr),
                arg(&mut token_ptr),
                arg(&mut gamma_ptr),
                arg(&mut beta_ptr),
                arg(&mut out_ptr),
                arg(&mut count),
                arg(&mut width),
                arg(&mut eps),
            ],
        );
    }
    let expected = scalar_out.bits(stream);
    assert!(expected
        .iter()
        .all(|bits| bf16::from_bits(*bits).is_finite()));
    assert_eq!(
        vector_out.bits(stream),
        expected,
        "embedding rows{rows} adversarial{adversarial} unaligned{unaligned:?}"
    );
}

#[test]
#[ignore = "requires INFINITY_OXIDE_KERNEL_MODULE and a matching CUDA device"]
fn vector_embedding_norm_768_matches_scalar_for_shapes_and_individual_fallbacks() {
    let (module, stream) = setup();
    for adversarial in [false, true] {
        for rows in ROWS {
            run_embedding(&module, &stream, rows, adversarial, None);
        }
        for slot in 0..6 {
            run_embedding(&module, &stream, 5, adversarial, Some(slot));
        }
    }
}
