// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
// Test-only actual-native Jina numerical GPU controls.
use crate::jina_primitive_schema::*;
use crate::{arg, cublas::Blas, launch_threads, math, tensor::Bf16, upload::WeightUploader};
use cuda_core::{DeviceBuffer, Stream};
use half::{bf16, f16};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

fn upload_bf16(stream: &Stream, bits: &[u16], padding: usize) -> (DeviceBuffer<Bf16>, Vec<u16>) {
    let host = guarded(bits, padding);
    let before = host.iter().copied().map(Bf16::to_bits).collect::<Vec<_>>();
    (DeviceBuffer::from_host(stream, &host).unwrap(), before)
}

fn offset(buffer: &DeviceBuffer<Bf16>, padding: usize) -> u64 {
    buffer.cu_deviceptr() + u64::try_from(padding * 2).unwrap()
}

fn assert_unchanged(
    stream: &Stream,
    buffer: &DeviceBuffer<Bf16>,
    before: &[u16],
    padding: usize,
    elements: usize,
    label: &str,
) {
    let after = download(buffer, stream);
    assert_guard(&after, padding, elements);
    assert_eq!(after.as_slice(), before, "{label} input changed");
}

fn canonical_bf16_bits_from_f16(value: f32) -> u16 {
    bf16::from_f32(f16::from_f32(value).to_f32()).to_bits()
}

fn weight_value(index: usize) -> f32 {
    ((((index * 17 + 11) % 127) as i32 - 63) as f32) / 1024.0
        + ((((index * 7) % 5) as i32 - 2) as f32) / 262144.0
}

fn packed_source_index(index: usize) -> usize {
    let chunk = INTERMEDIATE * WIDTH;
    if index < chunk {
        chunk + index
    } else {
        index - chunk
    }
}

fn packed_gate_up_bf16_bits() -> Vec<u16> {
    (0..PACKED_MLP * WIDTH)
        .map(|index| canonical_bf16_bits_from_f16(weight_value(packed_source_index(index))))
        .collect()
}

#[test]
fn jina_f16_to_bf16_canonicalization_rounds_f16_before_bf16() {
    let value = 1.0 + 2f32.powi(-8) + 2f32.powi(-12);
    assert_eq!(
        f16::from_f32(value).to_f32().to_bits(),
        (1.0f32 + 1.0 / 256.0).to_bits()
    );
    assert_eq!(
        canonical_bf16_bits_from_f16(value),
        bf16::from_f32(1.0).to_bits()
    );
    assert_eq!(
        bf16::from_f32(value).to_bits(),
        bf16::from_f32(1.0078125).to_bits()
    );
    assert_ne!(
        canonical_bf16_bits_from_f16(value),
        bf16::from_f32(value).to_bits()
    );
}

#[test]
fn jina_packed_gate_weights_are_canonical_up_then_gate_rows() {
    let chunk = INTERMEDIATE * WIDTH;
    assert_eq!(packed_source_index(0), chunk);
    assert_eq!(packed_source_index(chunk - 1), PACKED_MLP * WIDTH - 1);
    assert_eq!(packed_source_index(chunk), 0);
    assert_eq!(packed_source_index(PACKED_MLP * WIDTH - 1), chunk - 1);

    let packed = packed_gate_up_bf16_bits();
    assert_eq!(packed.len(), PACKED_MLP * WIDTH);
    for probe in [
        0usize,
        1,
        WIDTH - 1,
        chunk - 1,
        chunk,
        PACKED_MLP * WIDTH - 1,
    ] {
        assert_eq!(
            packed[probe],
            canonical_bf16_bits_from_f16(weight_value(packed_source_index(probe))),
            "packed weight probe {probe}"
        );
    }
}

#[test]
#[ignore = "Root GPU0 and actual matched Jina reference required"]
fn gpu_jina_word_type_norm_512_matches_actual_native_bits() {
    let (oracle, _context, stream, kernels) = setup();
    let word = &oracle.word;
    assert_eq!(word.width, WIDTH);
    assert!(word.bias);
    assert_eq!(word.cases.len(), 24);
    let type_source = word.type_bits.concat();

    for padding in [16usize, 17] {
        for case in &word.cases {
            let elements = case.rows * WIDTH;
            let (word_buffer, word_before) = upload_bf16(&stream, &word.word_bits, padding);
            let (type_buffer, type_before) = upload_bf16(&stream, &type_source, padding);
            let (gamma, gamma_before) = upload_bf16(&stream, &word.gamma_bits, padding);
            let (beta, beta_before) = upload_bf16(&stream, &word.beta_bits, padding);
            let (out, _) = upload_bf16(&stream, &vec![SENTINEL; elements], padding);
            let ids = DeviceBuffer::from_host(&stream, &case.ids).unwrap();
            let type_ids = DeviceBuffer::from_host(&stream, &case.type_ids).unwrap();
            let mut ids_ptr = ids.cu_deviceptr();
            let mut type_ids_ptr = type_ids.cu_deviceptr();
            let mut word_ptr = offset(&word_buffer, padding);
            let mut type_ptr = offset(&type_buffer, padding);
            let mut gamma_ptr = offset(&gamma, padding);
            let mut beta_ptr = offset(&beta, padding);
            let mut out_ptr = offset(&out, padding);
            let mut rows = u32::try_from(case.rows).unwrap();
            let mut eps = case.eps;
            launch_threads(
                &kernels.word_type_norm,
                &stream,
                ((case.rows.div_ceil(4) + 1) as u32, 1, 1),
                128,
                0,
                &mut [
                    arg(&mut ids_ptr),
                    arg(&mut type_ids_ptr),
                    arg(&mut word_ptr),
                    arg(&mut type_ptr),
                    arg(&mut gamma_ptr),
                    arg(&mut beta_ptr),
                    arg(&mut out_ptr),
                    arg(&mut rows),
                    arg(&mut eps),
                ],
            )
            .unwrap();
            stream.synchronize().unwrap();

            assert_unchanged(
                &stream,
                &word_buffer,
                &word_before,
                padding,
                word.word_bits.len(),
                "word",
            );
            assert_unchanged(
                &stream,
                &type_buffer,
                &type_before,
                padding,
                type_source.len(),
                "type table",
            );
            assert_unchanged(
                &stream,
                &gamma,
                &gamma_before,
                padding,
                word.gamma_bits.len(),
                "gamma",
            );
            assert_unchanged(
                &stream,
                &beta,
                &beta_before,
                padding,
                word.beta_bits.len(),
                "beta",
            );

            let mut ids_after = vec![0u32; case.ids.len()];
            ids.copy_to_host(&stream, &mut ids_after).unwrap();
            assert_eq!(ids_after, case.ids, "ids changed at padding {padding}");
            let mut type_ids_after = vec![0u32; case.type_ids.len()];
            type_ids.copy_to_host(&stream, &mut type_ids_after).unwrap();
            assert_eq!(
                type_ids_after, case.type_ids,
                "type ids changed at padding {padding}"
            );

            let actual = download(&out, &stream);
            assert_guard(&actual, padding, elements);
            assert_eq!(
                &actual[padding..padding + elements],
                case.expected_bits.as_slice(),
                "word rows={} eps={} variant={} padding={padding}",
                case.rows,
                case.eps,
                case.variant
            );
        }
    }
}

#[test]
#[ignore = "Root GPU0 and actual matched Jina reference required"]
fn gpu_jina_rounded_residual_512_matches_actual_native_bits() {
    let (oracle, _context, stream, kernels) = setup();
    let residual = &oracle.residual;
    assert_eq!(residual.width, WIDTH);
    assert_eq!(residual.cases.len(), 8);

    for padding in [16usize, 17] {
        for case in &residual.cases {
            let elements = case.rows * WIDTH;
            for alias in ["separate", "input", "residual"] {
                let (input, input_before) = upload_bf16(&stream, &case.input_bits, padding);
                let (residual_input, residual_before) =
                    upload_bf16(&stream, &case.residual_bits, padding);
                let (gamma, gamma_before) = upload_bf16(&stream, &case.gamma_bits, padding);
                let (beta, beta_before) = upload_bf16(&stream, &case.beta_bits, padding);
                let (out, _) = upload_bf16(&stream, &vec![SENTINEL; elements], padding);

                let mut input_ptr = offset(&input, padding);
                let mut residual_ptr = offset(&residual_input, padding);
                let mut bias_ptr: u64 = 0;
                let mut gamma_ptr = offset(&gamma, padding);
                let mut beta_ptr = offset(&beta, padding);
                let mut out_ptr = match alias {
                    "separate" => offset(&out, padding),
                    "input" => input_ptr,
                    "residual" => residual_ptr,
                    _ => unreachable!(),
                };
                let mut rows = u32::try_from(case.rows).unwrap();
                let mut width = WIDTH as u32;
                let mut eps = case.eps;
                launch_threads(
                    &kernels.rounded_residual,
                    &stream,
                    (u32::try_from(case.rows).unwrap(), 1, 1),
                    256,
                    0,
                    &mut [
                        arg(&mut input_ptr),
                        arg(&mut residual_ptr),
                        arg(&mut bias_ptr),
                        arg(&mut gamma_ptr),
                        arg(&mut beta_ptr),
                        arg(&mut out_ptr),
                        arg(&mut rows),
                        arg(&mut width),
                        arg(&mut eps),
                    ],
                )
                .unwrap();
                stream.synchronize().unwrap();

                assert_unchanged(
                    &stream,
                    &gamma,
                    &gamma_before,
                    padding,
                    case.gamma_bits.len(),
                    "gamma",
                );
                assert_unchanged(
                    &stream,
                    &beta,
                    &beta_before,
                    padding,
                    case.beta_bits.len(),
                    "beta",
                );
                if alias != "input" {
                    assert_unchanged(&stream, &input, &input_before, padding, elements, "input");
                }
                if alias != "residual" {
                    assert_unchanged(
                        &stream,
                        &residual_input,
                        &residual_before,
                        padding,
                        elements,
                        "residual",
                    );
                }

                let actual = match alias {
                    "separate" => download(&out, &stream),
                    "input" => download(&input, &stream),
                    "residual" => download(&residual_input, &stream),
                    _ => unreachable!(),
                };
                assert_guard(&actual, padding, elements);
                assert_eq!(
                    &actual[padding..padding + elements],
                    case.expected_bits.as_slice(),
                    "residual rows={} eps={} alias={alias} padding={padding}",
                    case.rows,
                    case.eps
                );
            }
        }
    }
}

// Native Jina records [gate, up]; the prepared weight policy exposes [up, gate].
// Canonicalize within each token row, never across the complete packed matrix.
fn canonical_preactivation(packed: &[u16]) -> Vec<u16> {
    assert!(packed.len().is_multiple_of(PACKED_MLP));
    packed
        .chunks_exact(PACKED_MLP)
        .flat_map(|row| {
            row[INTERMEDIATE..]
                .iter()
                .chain(row[..INTERMEDIATE].iter())
                .copied()
        })
        .collect()
}

#[test]
fn jina_native_preactivation_canonicalization_keeps_token_rows_separate() {
    let mut packed = vec![0u16; 2 * PACKED_MLP];
    packed[..INTERMEDIATE].fill(11);
    packed[INTERMEDIATE..PACKED_MLP].fill(22);
    packed[PACKED_MLP..PACKED_MLP + INTERMEDIATE].fill(33);
    packed[PACKED_MLP + INTERMEDIATE..].fill(44);
    let actual = canonical_preactivation(&packed);
    assert!(actual[..INTERMEDIATE].iter().all(|&v| v == 22));
    assert!(actual[INTERMEDIATE..PACKED_MLP].iter().all(|&v| v == 11));
    assert!(actual[PACKED_MLP..PACKED_MLP + INTERMEDIATE]
        .iter()
        .all(|&v| v == 44));
    assert!(actual[PACKED_MLP + INTERMEDIATE..].iter().all(|&v| v == 33));
}

fn gate_native(
    kernels: &PrimitiveKernels,
    stream: &Stream,
    packed: &[u16],
    rows: usize,
) -> Vec<u16> {
    let padding = 16usize;
    let before = guarded(packed, padding);
    let source = DeviceBuffer::from_host(stream, &before).unwrap();
    let target = DeviceBuffer::from_host(
        stream,
        &guarded(&vec![SENTINEL; rows * INTERMEDIATE], padding),
    )
    .unwrap();
    let mut input_ptr = offset(&source, padding);
    let mut output_ptr = offset(&target, padding);
    let mut rows_u32 = u32::try_from(rows).unwrap();
    let mut width = INTERMEDIATE as u32;
    let grid = u32::try_from((rows * INTERMEDIATE).div_ceil(256) + 1).unwrap();
    launch_threads(
        &kernels.gated_gelu,
        stream,
        (grid, 1, 1),
        256,
        0,
        &mut [
            arg(&mut input_ptr),
            arg(&mut output_ptr),
            arg(&mut rows_u32),
            arg(&mut width),
        ],
    )
    .unwrap();
    let actual = download(&target, stream);
    assert_guard(&actual, padding, rows * INTERMEDIATE);
    assert_eq!(
        download(&source, stream),
        before
            .iter()
            .copied()
            .map(Bf16::to_bits)
            .collect::<Vec<_>>(),
        "gated GELU input changed"
    );
    actual[padding..padding + rows * INTERMEDIATE].to_vec()
}

struct FixtureDir(PathBuf);
impl Drop for FixtureDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
static DIRECTORY_COUNTER: AtomicUsize = AtomicUsize::new(0);

#[test]
#[ignore = "Root GPU0 and actual matched Jina reference required"]
fn gpu_jina_gate_first_weights_and_gelu_match_actual_native_bits() {
    use safetensors::tensor::{serialize, TensorView};
    use safetensors::Dtype;

    let (oracle, context, stream, kernels) = setup();
    assert_eq!(
        oracle
            .gemm_gate_cases
            .iter()
            .map(|case| case.rows)
            .collect::<Vec<_>>(),
        ROWS.to_vec()
    );

    let counter = DIRECTORY_COUNTER.fetch_add(1, Ordering::Relaxed);
    let directory = FixtureDir(std::env::temp_dir().join(format!(
        "oxide-jina-gate-control-{}-{counter}",
        std::process::id()
    )));
    std::fs::create_dir(&directory.0).unwrap();
    let packed_bits = packed_gate_up_bf16_bits();
    let packed_bytes = packed_bits
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<u8>>();
    let packed_view = TensorView::new(Dtype::BF16, vec![PACKED_MLP, WIDTH], &packed_bytes).unwrap();
    std::fs::write(
        directory.0.join("model.safetensors"),
        serialize([("packed", packed_view)], None).unwrap(),
    )
    .unwrap();
    let checkpoint = crate::model::Checkpoint::open(&directory.0).unwrap();
    let packed_source = checkpoint.source("packed", &[PACKED_MLP, WIDTH]).unwrap();
    let mut uploader = WeightUploader::new(stream.clone()).unwrap();
    let weights = uploader.upload(&[&packed_source]).unwrap();
    uploader.finish().unwrap();
    let blas = Blas::new(context.clone(), stream.clone()).unwrap();
    context.bind_to_thread().unwrap();

    for case in &oracle.gemm_gate_cases {
        let expected_canonical = canonical_preactivation(&case.expected_packed_bits);
        assert_eq!(
            gate_native(&kernels, &stream, &expected_canonical, case.rows),
            case.expected_gate_bits,
            "native GELU rows={} ordinary={}",
            case.rows,
            case.used_ordinary
        );

        let input_host = case
            .input_bits
            .iter()
            .copied()
            .map(Bf16::from_bits)
            .collect::<Vec<_>>();
        let input = DeviceBuffer::from_host(&stream, &input_host).unwrap();
        let mut projected = DeviceBuffer::<Bf16>::zeroed(&stream, case.rows * PACKED_MLP).unwrap();
        let plan = math::LinearPlan::<Bf16>::contiguous(case.rows, WIDTH, PACKED_MLP).unwrap();
        unsafe {
            blas.linear(&plan, &input, &weights, &mut projected)
                .unwrap();
        }
        let actual = download(&projected, &stream);
        assert_eq!(
            actual.len(),
            case.rows * PACKED_MLP,
            "packed shape rows={} ordinary={}",
            case.rows,
            case.used_ordinary
        );
        assert_eq!(
            actual.as_slice(),
            expected_canonical.as_slice(),
            "canonical packed bits rows={} ordinary={}",
            case.rows,
            case.used_ordinary
        );
        assert_eq!(
            gate_native(&kernels, &stream, &actual, case.rows),
            case.expected_gate_bits,
            "synthetic GELU rows={} ordinary={}",
            case.rows,
            case.used_ordinary
        );
        assert_eq!(
            download(&input, &stream).as_slice(),
            case.input_bits.as_slice(),
            "gemm input changed rows={}",
            case.rows
        );
    }

    stream.synchronize().unwrap();
    drop(weights);
    drop(uploader);
    drop(blas);
    drop(checkpoint);
    drop(directory);
}
