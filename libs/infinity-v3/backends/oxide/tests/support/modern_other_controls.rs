// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
use super::primitive_schema::*;
use super::*;
use half::bf16;
use std::path::PathBuf;

#[test]
#[ignore = "Root GPU0 and exact native Modern reference required"]
fn gpu_modern_rope_tables_and_hfma_match_actual_native_bits() {
    let (oracle, _context, stream, kernels) = setup();
    assert_eq!(oracle.rope_cases.len(), 2);
    for (case, theta) in oracle.rope_cases.iter().zip([160000.0, 10000.0]) {
        assert_eq!(case.theta, theta);
        assert_eq!(case.capacity, 8192);
        assert_eq!(case.rows, 5);
        assert_eq!(case.positions, [0, 1, 63, 64, 8191]);
        assert_eq!(case.cosine_bits.len(), 8192 * 32);
        assert_eq!(case.sine_bits.len(), 8192 * 32);
        assert_eq!(case.qkv_input_bits.len(), 5 * 2304);
        assert_eq!(case.expected_qkv_bits.len(), 5 * 2304);
        let tables = Tables::new(theta, 8192, &stream, kernels.table_init.clone()).unwrap();
        for (buffer, expected) in [
            (&tables.cosine, &case.cosine_bits),
            (&tables.sine, &case.sine_bits),
        ] {
            let mut actual = vec![0f32; buffer.len()];
            buffer.copy_to_host(&stream, &mut actual).unwrap();
            for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
                assert_eq!(
                    bf16::from_f32(a).to_bits(),
                    e,
                    "theta={theta} coefficient={index}"
                );
                assert_eq!(a.to_bits(), bf16::from_bits(e).to_f32().to_bits());
            }
        }
        let padding = 16;
        let before = guarded(&case.qkv_input_bits, padding);
        let packed = DeviceBuffer::from_host(&stream, &before).unwrap();
        let positions = DeviceBuffer::from_host(&stream, &case.positions).unwrap();
        let mut q = packed.cu_deviceptr() + (padding * 2) as u64;
        let mut k = q + (WIDTH * 2) as u64;
        let mut cosine = tables.cosine.cu_deviceptr();
        let mut sine = tables.sine.cu_deviceptr();
        let mut pos = positions.cu_deviceptr();
        let mut rows = 5u32;
        let mut qheads = 12u32;
        let mut kheads = 12u32;
        let mut dim = 64u32;
        let mut rotary = 64u32;
        let mut qs = 2304u64;
        let mut ks = 2304u64;
        let mut stride = 32u32;
        launch(
            &kernels.rope,
            &stream,
            blocks(5 * 24 * 32).unwrap(),
            0,
            &mut [
                arg(&mut q),
                arg(&mut k),
                arg(&mut cosine),
                arg(&mut sine),
                arg(&mut pos),
                arg(&mut rows),
                arg(&mut qheads),
                arg(&mut kheads),
                arg(&mut dim),
                arg(&mut rotary),
                arg(&mut qs),
                arg(&mut ks),
                arg(&mut stride),
            ],
        )
        .unwrap();
        let actual = download(&packed, &stream);
        assert_guard(&actual, padding, 5 * 2304);
        assert_eq!(
            &actual[padding..padding + 5 * 2304],
            case.expected_qkv_bits.as_slice()
        );
        for row in 0..5 {
            assert_eq!(
                &actual[padding + row * 2304 + 2 * WIDTH..padding + (row + 1) * 2304],
                &case.qkv_input_bits[row * 2304 + 2 * WIDTH..(row + 1) * 2304]
            );
        }
    }
}

fn gate(kernels: &Kernels, stream: &Stream, input: &[u16], rows: usize) -> Vec<u16> {
    let padding = 16;
    let before = guarded(input, padding);
    let source = DeviceBuffer::from_host(stream, &before).unwrap();
    let target = DeviceBuffer::from_host(
        stream,
        &guarded(&vec![SENTINEL; rows * INTERMEDIATE], padding),
    )
    .unwrap();
    let mut input = source.cu_deviceptr() + (padding * 2) as u64;
    let mut output = target.cu_deviceptr() + (padding * 2) as u64;
    let mut rows_u32 = rows as u32;
    let mut width = INTERMEDIATE as u32;
    launch(
        &kernels.gate,
        stream,
        blocks(rows * INTERMEDIATE).unwrap(),
        0,
        &mut [
            arg(&mut input),
            arg(&mut output),
            arg(&mut rows_u32),
            arg(&mut width),
        ],
    )
    .unwrap();
    let actual = download(&target, stream);
    assert_guard(&actual, padding, rows * INTERMEDIATE);
    assert_eq!(
        download(&source, stream),
        before.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    actual[padding..padding + rows * INTERMEDIATE].to_vec()
}
fn swap_halves(input: &[u16], rows: usize) -> Vec<u16> {
    assert_eq!(input.len(), rows * INTERMEDIATE * 2);
    let mut output = Vec::with_capacity(input.len());
    for row in input.chunks_exact(INTERMEDIATE * 2) {
        output.extend_from_slice(&row[INTERMEDIATE..]);
        output.extend_from_slice(&row[..INTERMEDIATE]);
    }
    output
}
struct FixtureDir(PathBuf);
impl Drop for FixtureDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
#[ignore = "Root GPU0 and exact native Modern GEMM/gate reference required"]
fn gpu_modern_weight_canonicalization_and_gate_match_actual_native() {
    use safetensors::tensor::{serialize, TensorView};
    use safetensors::Dtype;
    let (oracle, context, stream, kernels) = setup();
    assert_eq!(
        oracle
            .gemm_gate_cases
            .iter()
            .map(|c| c.rows)
            .collect::<Vec<_>>(),
        [1, 3, 5, 128]
    );
    for case in &oracle.gemm_gate_cases {
        let native_up_first = swap_halves(&case.expected_packed_bits, case.rows);
        assert_eq!(
            gate(&kernels, &stream, &native_up_first, case.rows),
            case.expected_gate_bits,
            "native packed tensor must establish gate-order equivalence independently"
        );
    }
    let directory = FixtureDir(
        std::env::temp_dir().join(format!("oxide-modern-gate-control-{}", std::process::id())),
    );
    std::fs::create_dir(&directory.0).unwrap();
    let weight = (0..2 * INTERMEDIATE * WIDTH)
        .map(|i| {
            (((i * 17 + 11) % 127) as i32 - 63) as f32 / 1024.0
                + (((i * 7) % 5) as i32 - 2) as f32 / 262144.0
        })
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let view = TensorView::new(Dtype::F32, vec![2 * INTERMEDIATE, WIDTH], &weight).unwrap();
    std::fs::write(
        directory.0.join("model.safetensors"),
        serialize([("wi", view)], None).unwrap(),
    )
    .unwrap();
    let checkpoint = crate::model::Checkpoint::open(&directory.0).unwrap();
    let source = checkpoint.source("wi", &[2 * INTERMEDIATE, WIDTH]).unwrap();
    let [up, gate_source] =
        crate::modern_encoder_model::canonical_up_gate(&source, WIDTH, INTERMEDIATE).unwrap();
    let mut uploader = WeightUploader::new(stream.clone()).unwrap();
    let weights = uploader.upload(&[&up, &gate_source]).unwrap();
    uploader.finish().unwrap();
    let blas = Blas::new(context.clone(), stream.clone()).unwrap();
    context.bind_to_thread().unwrap();
    for case in &oracle.gemm_gate_cases {
        assert_eq!(
            case.weight_recipe,
            "f32_base127_over1024_plus_jitter5_over262144_v1"
        );
        assert_eq!(case.weight_shape, [2304, 768]);
        assert_eq!(case.input_bits.len(), case.rows * WIDTH);
        assert_eq!(case.expected_packed_bits.len(), case.rows * 2304);
        assert_eq!(case.expected_gate_bits.len(), case.rows * INTERMEDIATE);
        let input_host = case
            .input_bits
            .iter()
            .copied()
            .map(Bf16::from_bits)
            .collect::<Vec<_>>();
        let input = DeviceBuffer::from_host(&stream, &input_host).unwrap();
        let mut projected = DeviceBuffer::<Bf16>::zeroed(&stream, case.rows * 2304).unwrap();
        let plan = math::LinearPlan::<Bf16>::contiguous(case.rows, WIDTH, 2304).unwrap();
        unsafe {
            blas.linear(&plan, &input, &weights, &mut projected)
                .unwrap();
        }
        let actual = download(&projected, &stream);
        let original = swap_halves(&actual, case.rows);
        for (index, (&a, &e)) in original.iter().zip(&case.expected_packed_bits).enumerate() {
            assert_eq!(
                a, e,
                "rows={} packed={index} nativeordinary={}",
                case.rows, case.used_ordinary
            );
        }
        assert_eq!(
            gate(&kernels, &stream, &actual, case.rows),
            case.expected_gate_bits
        );
        assert_eq!(download(&input, &stream), case.input_bits);
    }
}

fn value(bits: u16) -> f32 {
    bf16::from_bits(bits).to_f32()
}
fn within_bf16_ulp(actual: f32, expected: f32) {
    let rounded = bf16::from_f32(expected);
    let bits = rounded.to_bits();
    let ulp = bf16::from_bits(bits + 1).to_f32() - rounded.to_f32();
    assert!(
        (actual - expected).abs() <= ulp,
        "actual={actual}, expected={expected}, ulp={ulp}"
    );
}

#[test]
#[ignore = "Root GPU0 and FA4 global/local provider required"]
fn gpu_modern_fa4_local_boundary_and_batch_isolation() {
    let context = Context::new(0).unwrap();
    context.bind_to_thread().unwrap();
    let stream = context.new_stream().unwrap();
    let root = PathBuf::from(
        std::env::var_os(infinity_v3_config::infinity_env!("FA4_ROOT")).expect("INFINITY_FA4_ROOT"),
    );
    let spec = fa4::AttentionSpec {
        heads: 12,
        kv_heads: 12,
        head_dim: 64,
        causal: false,
        dtype: fa4::DType::Bf16,
        scale: fa4::Scale::RsqrtHeadDim,
        window: fa4::Window::Global,
    };
    let mut packed = vec![0u16; 133 * 2304];
    let v_value = |row: usize| match row {
        0 | 130 => 8.0,
        1 | 129 => 1.0,
        131 => 17.0,
        132 => 19.0,
        _ => 0.0,
    };
    for row in 0usize..133 {
        for col in 0..WIDTH {
            packed[row * 2304 + 2 * WIDTH + col] = bf16::from_f32(v_value(row)).to_bits();
        }
    }
    let padding = 16;
    let before = guarded(&packed, padding);
    let source = DeviceBuffer::from_host(&stream, &before).unwrap();
    let boundaries = DeviceBuffer::from_host(&stream, &[0u32, 131, 133]).unwrap();
    for local in [false, true] {
        let provider = fa4::Fa4::new(
            context.clone(),
            &root,
            fa4::AttentionSpec {
                window: if local {
                    fa4::Window::Local {
                        left: 64,
                        right: 64,
                    }
                } else {
                    fa4::Window::Global
                },
                ..spec
            },
        )
        .unwrap();
        let output =
            DeviceBuffer::from_host(&stream, &guarded(&vec![SENTINEL; 133 * WIDTH], padding))
                .unwrap();
        let q = source.cu_deviceptr() + (padding * 2) as u64;
        let batch = fa4::AttentionBatch {
            q,
            k: q + 1536,
            v: q + 3072,
            out: output.cu_deviceptr() + (padding * 2) as u64,
            q_offsets: boundaries.cu_deviceptr(),
            k_offsets: boundaries.cu_deviceptr(),
            q_tokens: 133,
            k_tokens: 133,
            batch: 2,
            max_q: 131,
            strides: [2304, 64, 1, 2304, 64, 1, 2304, 64, 1, 768, 64, 1],
        };
        unsafe {
            provider.forward_batch_bound(&stream, &batch).unwrap();
        }
        let actual = download(&output, &stream);
        assert_guard(&actual, padding, 133 * WIDTH);
        let mut reference = Vec::with_capacity(133 * WIDTH);
        let mut candidate = Vec::with_capacity(133 * WIDTH);
        for row in 0usize..133 {
            let (start, end) = if row >= 131 {
                (131, 133)
            } else if local {
                (row.saturating_sub(64), (row + 65).min(131))
            } else {
                (0, 131)
            };
            let expected = (start..end).map(v_value).sum::<f32>() / (end - start) as f32;
            for col in 0..WIDTH {
                let a = value(actual[padding + row * WIDTH + col]);
                assert!(a.is_finite());
                reference.push(expected as f64);
                candidate.push(a as f64);
                if row == 65 {
                    within_bf16_ulp(a, if local { 2.0 / 129.0 } else { 18.0 / 131.0 });
                }
                if row >= 131 {
                    assert_eq!(a, 18.0, "packed batch leaked");
                }
            }
        }
        let dot = reference
            .iter()
            .zip(&candidate)
            .map(|(a, b)| a * b)
            .sum::<f64>();
        let rr = reference.iter().map(|v| v * v).sum::<f64>();
        let cc = candidate.iter().map(|v| v * v).sum::<f64>();
        let error = reference
            .iter()
            .zip(&candidate)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>();
        let max_abs = reference
            .iter()
            .zip(&candidate)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(dot / (rr * cc).sqrt() >= 0.999 && max_abs <= 0.25 && (error / rr).sqrt() <= 0.02);
        assert_eq!(
            download(&source, &stream),
            before.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
}
