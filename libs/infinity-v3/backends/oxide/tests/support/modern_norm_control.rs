// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
use super::primitive_schema::*;
use super::*;

fn bits(data: &[Bf16]) -> Vec<u16> {
    data.iter().copied().map(Bf16::to_bits).collect()
}

fn upload(stream: &Stream, data: &[u16], padding: usize) -> (DeviceBuffer<Bf16>, Vec<u16>) {
    let host = guarded(data, padding);
    let before = bits(&host);
    (DeviceBuffer::from_host(stream, &host).unwrap(), before)
}

fn offset(buffer: &DeviceBuffer<Bf16>, padding: usize) -> u64 {
    buffer.cu_deviceptr() + u64::try_from(padding * 2).unwrap()
}

fn assert_expected(label: &str, data: &[u16], padding: usize, elements: usize, expected: &[u16]) {
    assert_guard(data, padding, elements);
    assert_eq!(&data[padding..padding + elements], expected, "{label}");
}

#[test]
#[ignore = "requires CUDA 13 GPU runtime"]
fn gpu_modern_biasless_norm_matches_actual_native_bits() {
    let (oracle, _ctx, stream, kernels) = setup();
    assert_eq!(oracle.word.width, WIDTH);
    assert_eq!(oracle.word.eps.to_bits(), 1e-5f32.to_bits());
    assert!(!oracle.word.bias);
    assert_eq!(oracle.word.word_bits.len(), 8 * WIDTH);
    assert_eq!(oracle.word.gamma_bits.len(), WIDTH);
    let word_rows = oracle
        .word
        .cases
        .iter()
        .map(|case| case.rows)
        .collect::<Vec<_>>();
    assert_eq!(word_rows, vec![1, 3, 5, 128]);
    assert_eq!(oracle.norm_cases.len(), 24);
    for rows in [1usize, 3, 5, 128] {
        for eps in [1e-5f32, 1e-12] {
            for mode in ["plain", "add_store", "final"] {
                assert!(oracle.norm_cases.iter().any(|case| {
                    case.rows == rows && case.eps.to_bits() == eps.to_bits() && case.mode == mode
                }));
            }
        }
    }
    for padding in [16usize, 17] {
        for case in &oracle.word.cases {
            let elements = case.rows * WIDTH;
            let (word, word_before) = upload(&stream, &oracle.word.word_bits, padding);
            let (gamma, gamma_before) = upload(&stream, &oracle.word.gamma_bits, padding);
            let (out, _) = upload(&stream, &vec![SENTINEL; elements], padding);
            let ids = DeviceBuffer::from_host(&stream, &case.ids).unwrap();
            let mut ids_ptr = ids.cu_deviceptr();
            let mut word_ptr = offset(&word, padding);
            let mut gamma_ptr = offset(&gamma, padding);
            let mut out_ptr = offset(&out, padding);
            let mut rows = u32::try_from(case.rows).unwrap();
            let mut eps = oracle.word.eps;
            launch_threads(
                &kernels.word_norm,
                &stream,
                ((case.rows.div_ceil(4) + 1) as u32, 1, 1),
                128,
                0,
                &mut [
                    arg(&mut ids_ptr),
                    arg(&mut word_ptr),
                    arg(&mut gamma_ptr),
                    arg(&mut out_ptr),
                    arg(&mut rows),
                    arg(&mut eps),
                ],
            )
            .unwrap();
            stream.synchronize().unwrap();
            assert_eq!(
                download(&word, &stream),
                word_before,
                "word input changed at {padding}"
            );
            assert_eq!(
                download(&gamma, &stream),
                gamma_before,
                "gamma input changed at {padding}"
            );
            let mut ids_after = vec![0u32; case.rows];
            ids.copy_to_host(&stream, &mut ids_after).unwrap();
            assert_eq!(ids_after, case.ids, "ids changed at {padding}");
            let out_after = download(&out, &stream);
            assert_expected(
                &format!("word rows {} offset {padding}", case.rows),
                &out_after,
                padding,
                elements,
                &case.expected_bits,
            );
        }
        for case in &oracle.norm_cases {
            let elements = case.rows * WIDTH;
            let variants = if case.mode == "add_store" { 2 } else { 1 };
            for variant in 0..variants {
                let (input, input_before) = upload(&stream, &case.input_bits, padding);
                let (gamma, gamma_before) = upload(&stream, &case.gamma_bits, padding);
                let (normalized, _) = upload(&stream, &vec![SENTINEL; elements], padding);
                let (residual, residual_before) = if case.residual_bits.is_empty() {
                    (None, None)
                } else {
                    let (buffer, host) = upload(&stream, &case.residual_bits, padding);
                    (Some(buffer), Some(host))
                };
                let mut residual_ptr = residual
                    .as_ref()
                    .map(|buffer| offset(buffer, padding))
                    .unwrap_or(0);
                let (sum, _sum_before) = if case.mode == "add_store" && variant == 1 {
                    let (buffer, host) = upload(&stream, &vec![SENTINEL; elements], padding);
                    (Some(buffer), Some(host))
                } else {
                    (None, None)
                };
                let mut sum_ptr = if case.mode == "add_store" {
                    if variant == 0 {
                        residual_ptr
                    } else {
                        offset(sum.as_ref().unwrap(), padding)
                    }
                } else {
                    0
                };
                let mut input_ptr = offset(&input, padding);
                let mut gamma_ptr = offset(&gamma, padding);
                let mut normalized_ptr = offset(&normalized, padding);
                let mut rows = u32::try_from(case.rows).unwrap();
                let mut eps = case.eps;
                launch_threads(
                    &kernels.prenorm,
                    &stream,
                    ((case.rows.div_ceil(4) + 1) as u32, 1, 1),
                    128,
                    0,
                    &mut [
                        arg(&mut input_ptr),
                        arg(&mut residual_ptr),
                        arg(&mut gamma_ptr),
                        arg(&mut normalized_ptr),
                        arg(&mut sum_ptr),
                        arg(&mut rows),
                        arg(&mut eps),
                    ],
                )
                .unwrap();
                stream.synchronize().unwrap();
                assert_eq!(
                    download(&input, &stream),
                    input_before,
                    "norm input changed for {} rows {} eps {} variant {variant}",
                    case.mode,
                    case.rows,
                    case.eps,
                );
                assert_eq!(
                    download(&gamma, &stream),
                    gamma_before,
                    "norm gamma changed for {} rows {} variant {variant}",
                    case.mode,
                    case.rows,
                );
                let normalized_after = download(&normalized, &stream);
                assert_expected(
                    &format!(
                        "norm {} rows {} eps {} variant {variant}",
                        case.mode, case.rows, case.eps
                    ),
                    &normalized_after,
                    padding,
                    elements,
                    &case.expected_bits,
                );
                match case.mode.as_str() {
                    "add_store" => {
                        let sum_after = if variant == 0 {
                            download(residual.as_ref().unwrap(), &stream)
                        } else {
                            download(sum.as_ref().unwrap(), &stream)
                        };
                        assert_expected(
                            &format!(
                                "sum {} rows {} eps {} variant {variant}",
                                case.mode, case.rows, case.eps
                            ),
                            &sum_after,
                            padding,
                            elements,
                            &case.expected_sum_bits,
                        );
                        if variant == 1 {
                            assert_eq!(
                                download(residual.as_ref().unwrap(), &stream),
                                *residual_before.as_ref().unwrap(),
                                "separate sum changed residual for {} rows {}",
                                case.rows,
                                case.eps,
                            );
                        }
                    }
                    "final" => {
                        assert_eq!(
                            download(residual.as_ref().unwrap(), &stream),
                            *residual_before.as_ref().unwrap(),
                            "final changed residual for {} rows {}",
                            case.rows,
                            case.eps,
                        );
                    }
                    "plain" => assert!(residual.is_none()),
                    other => panic!("unexpected norm mode {other}"),
                }
            }
        }
    }
}
