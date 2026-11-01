// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
// Test-only actual-native Nomic numerical GPU controls.
use crate::nomic_primitive_schema::*;
use crate::{
    arg, blocks, cublas::Blas, launch, launch_threads, math, native_rope_tables::Tables,
    tensor::Bf16, upload::WeightUploader,
};
use cuda_core::{DeviceBuffer, Stream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

fn upload_bf16(stream: &Stream, bits: &[u16], padding: usize) -> (DeviceBuffer<Bf16>, Vec<u16>) {
    let host = guarded(bits, padding);
    let before = host.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
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

#[test]
#[ignore = "Root GPU0 and actual matched Nomic reference required"]
fn gpu_nomic_word_type_norm_matches_actual_native_bits() {
    let (oracle, _context, stream, kernels) = setup();
    assert_eq!(oracle.word.width, WIDTH);
    assert!(oracle.word.bias);
    let type_source = oracle.word.type_bits.concat();
    for padding in [16usize, 17] {
        for case in &oracle.word.cases {
            let elements = case.rows * WIDTH;
            let (word, word_before) = upload_bf16(&stream, &oracle.word.word_bits, padding);
            let (types, types_before) = upload_bf16(&stream, &type_source, padding);
            let (gamma, gamma_before) = upload_bf16(&stream, &oracle.word.gamma_bits, padding);
            let (beta, beta_before) = upload_bf16(&stream, &oracle.word.beta_bits, padding);
            let (out, _) = upload_bf16(&stream, &vec![SENTINEL; elements], padding);
            let ids = DeviceBuffer::from_host(&stream, &case.ids).unwrap();
            let type_ids = DeviceBuffer::from_host(&stream, &case.type_ids).unwrap();
            let mut ids_ptr = ids.cu_deviceptr();
            let mut type_ids_ptr = type_ids.cu_deviceptr();
            let mut word_ptr = offset(&word, padding);
            let mut table_ptr = offset(&types, padding);
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
                    arg(&mut table_ptr),
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
                &word,
                &word_before,
                padding,
                oracle.word.word_bits.len(),
                "word",
            );
            assert_unchanged(
                &stream,
                &types,
                &types_before,
                padding,
                type_source.len(),
                "type table",
            );
            assert_unchanged(
                &stream,
                &gamma,
                &gamma_before,
                padding,
                oracle.word.gamma_bits.len(),
                "gamma",
            );
            assert_unchanged(
                &stream,
                &beta,
                &beta_before,
                padding,
                oracle.word.beta_bits.len(),
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
#[ignore = "Root GPU0 and actual matched Nomic reference required"]
fn gpu_nomic_theta1000_tables_and_rope_match_actual_native_bits() {
    let (oracle, _context, stream, kernels) = setup();
    let case = &oracle.rope;
    assert_eq!(case.theta, 1000.0);
    assert_eq!(case.capacity, ROPE_CAPACITY);
    let tables = Tables::new(
        case.theta,
        case.capacity,
        &stream,
        kernels.table_init.clone(),
    )
    .unwrap();
    for (buffer, expected, label) in [
        (&tables.cosine, &case.cosine_bits, "cosine"),
        (&tables.sine, &case.sine_bits, "sine"),
    ] {
        let mut actual = vec![0f32; buffer.len()];
        buffer.copy_to_host(&stream, &mut actual).unwrap();
        assert_eq!(actual.len(), case.capacity * (HEAD_DIM / 2));
        assert_eq!(expected.len(), actual.len());
        for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                half::bf16::from_f32(a).to_bits(),
                e,
                "theta={} {label} coefficient={index}",
                case.theta
            );
            assert_eq!(
                a.to_bits(),
                half::bf16::from_bits(e).to_f32().to_bits(),
                "theta={} {label} f32 coefficient={index}",
                case.theta
            );
        }
    }
    let padding = 16usize;
    let before = guarded(&case.source_packed_bits, padding);
    let packed = DeviceBuffer::from_host(&stream, &before).unwrap();
    let positions = DeviceBuffer::from_host(&stream, &case.positions).unwrap();
    let mut q = offset(&packed, padding);
    let mut k = q + (WIDTH * 2) as u64;
    let mut cosine = tables.cosine.cu_deviceptr();
    let mut sine = tables.sine.cu_deviceptr();
    let mut pos = positions.cu_deviceptr();
    let mut rows = u32::try_from(case.rows).unwrap();
    let mut qheads = 12u32;
    let mut kheads = 12u32;
    let mut dim = HEAD_DIM as u32;
    let mut rotary = HEAD_DIM as u32;
    let mut qs = PACKED_QKV as u64;
    let mut ks = PACKED_QKV as u64;
    let mut stride = (HEAD_DIM / 2) as u32;
    launch(
        &kernels.rope,
        &stream,
        blocks(case.rows * 24 * 32).unwrap(),
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
    assert_guard(&actual, padding, case.rows * PACKED_QKV);
    assert_eq!(
        &actual[padding..padding + case.rows * PACKED_QKV],
        case.expected_rotated_bits.as_slice()
    );
    for row in 0..case.rows {
        assert_eq!(
            &actual[padding + row * PACKED_QKV + 2 * WIDTH..padding + (row + 1) * PACKED_QKV],
            &case.source_packed_bits[row * PACKED_QKV + 2 * WIDTH..(row + 1) * PACKED_QKV],
            "V chunk row={row} changed"
        );
    }
    stream.synchronize().unwrap();
    drop(tables);
}

fn gate_native(
    kernels: &PrimitiveKernels,
    stream: &Stream,
    input: &[u16],
    rows: usize,
) -> Vec<u16> {
    let padding = 16usize;
    let before = guarded(input, padding);
    let source = DeviceBuffer::from_host(stream, &before).unwrap();
    let target = DeviceBuffer::from_host(
        stream,
        &guarded(&vec![SENTINEL; rows * INTERMEDIATE], padding),
    )
    .unwrap();
    let mut input_ptr = offset(&source, padding);
    let mut output_ptr = offset(&target, padding);
    let mut rows_u32 = rows as u32;
    let mut width = INTERMEDIATE as u32;
    launch(
        &kernels.silu_mul,
        stream,
        blocks(rows * INTERMEDIATE / 8).unwrap(),
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
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "silu input changed"
    );
    actual[padding..padding + rows * INTERMEDIATE].to_vec()
}
fn weight_value(index: usize) -> f32 {
    (((index * 17 + 11) % 127) as i32 - 63) as f32 / 1024.0
        + (((index * 7) % 5) as i32 - 2) as f32 / 262144.0
}
struct FixtureDir(PathBuf);
impl Drop for FixtureDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
static DIRECTORY_COUNTER: AtomicUsize = AtomicUsize::new(0);

#[test]
#[ignore = "Root GPU0 and actual matched Nomic reference required"]
fn gpu_nomic_gate_first_weights_and_silu_match_actual_native_bits() {
    use safetensors::tensor::{serialize, TensorView};
    use safetensors::Dtype;
    let (oracle, context, stream, kernels) = setup();
    assert_eq!(
        oracle
            .gemm_gate_cases
            .iter()
            .map(|c| c.rows)
            .collect::<Vec<_>>(),
        ROWS.to_vec()
    );
    for case in &oracle.gemm_gate_cases {
        assert_eq!(
            gate_native(&kernels, &stream, &case.expected_packed_bits, case.rows),
            case.expected_gate_bits,
            "native activation rows={} ordinary={}",
            case.rows,
            case.used_ordinary
        );
    }
    let counter = DIRECTORY_COUNTER.fetch_add(1, Ordering::Relaxed);
    let directory = FixtureDir(std::env::temp_dir().join(format!(
        "oxide-nomic-gate-control-{}-{counter}",
        std::process::id()
    )));
    std::fs::create_dir(&directory.0).unwrap();
    let gate_bytes = (0..INTERMEDIATE * WIDTH)
        .map(weight_value)
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<u8>>();
    let up_bytes = (0..INTERMEDIATE * WIDTH)
        .map(|i| weight_value(INTERMEDIATE * WIDTH + i))
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<u8>>();
    let gate_view = TensorView::new(Dtype::F32, vec![INTERMEDIATE, WIDTH], &gate_bytes).unwrap();
    let up_view = TensorView::new(Dtype::F32, vec![INTERMEDIATE, WIDTH], &up_bytes).unwrap();
    std::fs::write(
        directory.0.join("model.safetensors"),
        serialize([("fc12", gate_view), ("fc11", up_view)], None).unwrap(),
    )
    .unwrap();
    let checkpoint = crate::model::Checkpoint::open(&directory.0).unwrap();
    let gate_source = checkpoint.source("fc12", &[INTERMEDIATE, WIDTH]).unwrap();
    let up_source = checkpoint.source("fc11", &[INTERMEDIATE, WIDTH]).unwrap();
    let mut uploader = WeightUploader::new(stream.clone()).unwrap();
    let weights = uploader.upload(&[&gate_source, &up_source]).unwrap();
    uploader.finish().unwrap();
    let blas = Blas::new(context.clone(), stream.clone()).unwrap();
    context.bind_to_thread().unwrap();
    for case in &oracle.gemm_gate_cases {
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
        for (index, (&a, &e)) in actual.iter().zip(&case.expected_packed_bits).enumerate() {
            assert_eq!(
                a, e,
                "rows={} packed={index} ordinary={}",
                case.rows, case.used_ordinary
            );
        }
        assert_eq!(
            gate_native(&kernels, &stream, &actual, case.rows),
            case.expected_gate_bits,
            "synthetic activation rows={} ordinary={}",
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
