// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
use crate::fa4::{AttentionBatch, AttentionSpec, DType, Fa4, Scale, Window};
use cuda_core::{Context, DeviceBuffer, Stream};
use half::{bf16, f16};
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

const FIXTURE_ENV: &str = "INFINITY_OXIDE_JINA_REFERENCE";
const ROOT_ENV: &str = "INFINITY_FA4_ROOT";
const MAX_FIXTURE_BYTES: u64 = 64 * 1024 * 1024;

const HEADS: usize = 8;
const HEAD_DIM: usize = 64;
const WIDTH: usize = 512;
const PACKED_QKV: usize = 1536;
const DISTANCE_COUNT: usize = 8192;
const LONG_LENGTH: usize = 8192;
const GUARD: usize = 16;
const SENTINEL: u16 = 0x4242;
const SCALE_VALUE: f32 = 0.125;
const SLOPES: [f32; HEADS] = [
    0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625,
];
const LONG_SELECTED_ROWS: [usize; 10] = [0, 1, 255, 256, 257, 511, 512, 513, 8190, 8191];
const NORMAL_CASE_NAMES: [&str; 6] = [
    "single_1",
    "single_3",
    "ragged_3_1_5",
    "single_128",
    "boundary_514",
    "unequal_257_513",
];
const NORMAL_CASE_LENGTHS: [&[usize]; 6] = [&[1], &[3], &[3, 1, 5], &[128], &[514], &[257, 513]];
const NORMAL_CASE_KINDS: [&str; 6] = [
    "standard",
    "standard",
    "standard",
    "standard",
    "boundary_alibi",
    "sequence_leakage",
];
const STRIDES: [i64; 12] = [
    PACKED_QKV as i64,
    HEAD_DIM as i64,
    1,
    PACKED_QKV as i64,
    HEAD_DIM as i64,
    1,
    PACKED_QKV as i64,
    HEAD_DIM as i64,
    1,
    WIDTH as i64,
    HEAD_DIM as i64,
    1,
];

#[derive(Clone, Deserialize)]
struct AlibiBiasCase {
    dtype_code: u32,
    dtype: String,
    shape: [usize; 2],
    expected_bits: Vec<u16>,
}

#[derive(Clone, Deserialize)]
struct AlibiBias {
    slopes: Vec<f32>,
    distances: Vec<i64>,
    cases: Vec<AlibiBiasCase>,
}

#[derive(Clone, Deserialize)]
struct AlibiNormalCase {
    name: String,
    dtype_code: u32,
    dtype: String,
    case_kind: String,
    lengths: Vec<usize>,
    cumulative: Vec<i32>,
    max_sequence: usize,
    scale: f32,
    source_packed_bits: Vec<u16>,
    expected_bits: Vec<u16>,
}

#[derive(Clone, Deserialize)]
struct AlibiLongCase {
    name: String,
    dtype_code: u32,
    dtype: String,
    case_kind: String,
    lengths: Vec<usize>,
    cumulative: Vec<i32>,
    max_sequence: usize,
    scale: f32,
    source_recipe: String,
    selected_rows: Vec<usize>,
    expected_selected_bits: Vec<u16>,
}

struct AlibiFixture {
    capability: [i64; 2],
    bias: AlibiBias,
    normal: Vec<AlibiNormalCase>,
    long: Vec<AlibiLongCase>,
}

#[derive(Clone, Copy)]
struct RowMetrics {
    max_abs: f64,
    max_rel: f64,
    cosine: f64,
    exact: usize,
}

fn ensure(ok: bool, message: &str) -> Result<(), String> {
    ok.then_some(()).ok_or_else(|| message.to_string())
}

fn dtype_name(code: u32) -> Option<&'static str> {
    match code {
        0 => Some("float16"),
        1 => Some("bfloat16"),
        _ => None,
    }
}

fn validate_case_dtype(code: u32, name: &str) -> Result<(), String> {
    let expected = dtype_name(code).ok_or_else(|| format!("invalid dtype code {code}"))?;
    ensure(
        name == expected,
        "fixture dtype name does not match dtype code",
    )
}

fn bits_finite(bits: u16, code: u32) -> bool {
    match code {
        0 => bits & 0x7c00 != 0x7c00,
        1 => bits & 0x7f80 != 0x7f80,
        _ => false,
    }
}

fn all_bits_finite(bits: &[u16], code: u32) -> bool {
    bits.iter().all(|&value| bits_finite(value, code))
}

fn round_to_bits(value: f32, code: u32) -> u16 {
    match code {
        0 => f16::from_f32(value).to_bits(),
        1 => bf16::from_f32(value).to_bits(),
        _ => panic!("invalid ALiBi dtype code {code}"),
    }
}

fn bits_to_f32(bits: u16, code: u32) -> f32 {
    match code {
        0 => f16::from_bits(bits).to_f32(),
        1 => bf16::from_bits(bits).to_f32(),
        _ => panic!("invalid ALiBi dtype code {code}"),
    }
}

fn normal_case_definition(name: &str) -> Option<(Vec<usize>, &'static str)> {
    let index = NORMAL_CASE_NAMES
        .iter()
        .position(|candidate| *candidate == name)?;
    Some((
        NORMAL_CASE_LENGTHS[index].to_vec(),
        NORMAL_CASE_KINDS[index],
    ))
}

fn expected_cumulative(lengths: &[usize]) -> Result<Vec<i32>, String> {
    let mut cursor = 0usize;
    let mut cumulative = Vec::with_capacity(lengths.len() + 1);
    cumulative.push(0);
    for &length in lengths {
        cursor = cursor
            .checked_add(length)
            .ok_or_else(|| "attention token count overflow".to_string())?;
        cumulative.push(
            i32::try_from(cursor).map_err(|_| "attention token count exceeds i32".to_string())?,
        );
    }
    Ok(cumulative)
}

fn checked_packed_len(lengths: &[usize]) -> Result<usize, String> {
    let total = lengths.iter().try_fold(0usize, |sum, &length| {
        sum.checked_add(length)
            .ok_or_else(|| "attention token count overflow".to_string())
    })?;
    total
        .checked_mul(PACKED_QKV)
        .ok_or_else(|| "packed source length overflow".to_string())
}

fn checked_output_len(lengths: &[usize]) -> Result<usize, String> {
    let total = lengths.iter().try_fold(0usize, |sum, &length| {
        sum.checked_add(length)
            .ok_or_else(|| "attention token count overflow".to_string())
    })?;
    total
        .checked_mul(WIDTH)
        .ok_or_else(|| "attention output length overflow".to_string())
}

fn validate_scale(scale: f32) -> Result<(), String> {
    ensure(
        scale.to_bits() == SCALE_VALUE.to_bits(),
        "attention scale is not 1/sqrt(64)",
    )
}

fn validate_distinct_sequence_v(
    source: &[u16],
    lengths: &[usize],
    cumulative: &[i32],
) -> Result<(), String> {
    let mut seen: Vec<Vec<u16>> = Vec::with_capacity(lengths.len());
    for &start in &cumulative[..cumulative.len().saturating_sub(1)] {
        let start =
            usize::try_from(start).map_err(|_| "negative attention boundary".to_string())?;
        let offset = start
            .checked_mul(PACKED_QKV)
            .and_then(|value| value.checked_add(2 * WIDTH))
            .ok_or_else(|| "packed V offset overflow".to_string())?;
        let end = offset
            .checked_add(HEAD_DIM)
            .ok_or_else(|| "packed V slice overflow".to_string())?;
        ensure(end <= source.len(), "packed V slice is out of bounds")?;
        let row = source[offset..end].to_vec();
        ensure(
            !seen.iter().any(|previous| previous == &row),
            "packed V rows are not distinct per sequence",
        )?;
        seen.push(row);
    }
    Ok(())
}

fn validate_normal_case(case: &AlibiNormalCase) -> Result<(), String> {
    let (expected_lengths, expected_kind) = normal_case_definition(&case.name)
        .ok_or_else(|| format!("unknown normal ALiBi case {}", case.name))?;
    ensure(
        case.lengths == expected_lengths,
        "normal case geometry mismatch",
    )?;
    ensure(case.case_kind == expected_kind, "normal case kind mismatch")?;
    validate_case_dtype(case.dtype_code, &case.dtype)?;
    validate_scale(case.scale)?;
    let cumulative = expected_cumulative(&case.lengths)?;
    ensure(
        case.cumulative == cumulative,
        "normal case boundaries mismatch",
    )?;
    let maximum = case.lengths.iter().copied().max().unwrap_or(0);
    ensure(
        case.max_sequence == maximum,
        "normal case max sequence mismatch",
    )?;
    let source_len = checked_packed_len(&case.lengths)?;
    let output_len = checked_output_len(&case.lengths)?;
    ensure(
        case.source_packed_bits.len() == source_len,
        "normal case packed source length mismatch",
    )?;
    ensure(
        case.expected_bits.len() == output_len,
        "normal case attention output length mismatch",
    )?;
    ensure(
        all_bits_finite(&case.source_packed_bits, case.dtype_code),
        "normal case packed source is not finite for its dtype",
    )?;
    ensure(
        all_bits_finite(&case.expected_bits, case.dtype_code),
        "normal case expected output is not finite for its dtype",
    )?;
    validate_distinct_sequence_v(&case.source_packed_bits, &case.lengths, &case.cumulative)
}

fn validate_long_case(case: &AlibiLongCase) -> Result<(), String> {
    ensure(
        case.name == "long_8192_alibi",
        "long ALiBi case name mismatch",
    )?;
    ensure(
        case.case_kind == "long_alibi",
        "long ALiBi case kind mismatch",
    )?;
    validate_case_dtype(case.dtype_code, &case.dtype)?;
    ensure(
        case.lengths == vec![LONG_LENGTH],
        "long ALiBi geometry mismatch",
    )?;
    ensure(
        case.cumulative == vec![0, LONG_LENGTH as i32],
        "long ALiBi boundaries mismatch",
    )?;
    ensure(
        case.max_sequence == LONG_LENGTH,
        "long ALiBi max sequence mismatch",
    )?;
    validate_scale(case.scale)?;
    ensure(
        case.source_recipe
            == "q_token0_head_dim0_8_k_head_dim0_round_model_dtype_slope_x_token_\
v_all_dims_round_model_dtype_unrounded_slope_residual_over16_v1",
        "long ALiBi source recipe mismatch",
    )?;
    ensure(
        case.selected_rows == LONG_SELECTED_ROWS,
        "long ALiBi selected row set mismatch",
    )?;
    ensure(
        case.expected_selected_bits.len() == LONG_SELECTED_ROWS.len() * WIDTH,
        "long ALiBi selected output length mismatch",
    )?;
    ensure(
        all_bits_finite(&case.expected_selected_bits, case.dtype_code),
        "long ALiBi selected output is not finite for its dtype",
    )
}

fn validate_bias_section(bias: &AlibiBias) -> Result<(), String> {
    ensure(bias.slopes.len() == HEADS, "bias slope count mismatch")?;
    ensure(
        bias.slopes
            .iter()
            .zip(SLOPES)
            .all(|(actual, expected)| actual.to_bits() == expected.to_bits()),
        "bias slopes mismatch",
    )?;
    ensure(
        bias.distances.len() == DISTANCE_COUNT
            && bias
                .distances
                .iter()
                .enumerate()
                .all(|(index, &distance)| distance == index as i64),
        "bias distances mismatch",
    )?;
    ensure(bias.cases.len() == 2, "bias dtype case count mismatch")?;
    for code in [0u32, 1u32] {
        let case = bias
            .cases
            .iter()
            .find(|case| case.dtype_code == code)
            .ok_or_else(|| format!("missing bias dtype case {code}"))?;
        validate_case_dtype(case.dtype_code, &case.dtype)?;
        ensure(
            case.shape == [HEADS, DISTANCE_COUNT],
            "bias coefficient shape mismatch",
        )?;
        ensure(
            case.expected_bits.len() == HEADS * DISTANCE_COUNT,
            "bias coefficient length mismatch",
        )?;
        ensure(
            all_bits_finite(&case.expected_bits, case.dtype_code),
            "bias coefficient output is not finite for its dtype",
        )?;
    }
    ensure(
        bias.cases
            .iter()
            .filter(|case| case.dtype_code == 0)
            .count()
            == 1
            && bias
                .cases
                .iter()
                .filter(|case| case.dtype_code == 1)
                .count()
                == 1,
        "duplicate bias dtype case",
    )
}

fn validate_long_case_serialization(value: &Value) -> Result<(), String> {
    if let Some(cases) = value.get("attention_long_cases").and_then(Value::as_array) {
        for case in cases {
            ensure(
                case.get("source_packed_bits").is_none(),
                "long ALiBi fixture must not serialize packed source",
            )?;
            ensure(
                case.get("expected_bits").is_none(),
                "long ALiBi fixture must not serialize full expected output",
            )?;
        }
    }
    Ok(())
}

fn parse_attention_sections(
    value: &Value,
) -> Result<(AlibiBias, Vec<AlibiNormalCase>, Vec<AlibiLongCase>), String> {
    validate_long_case_serialization(value)?;
    let bias_value = value
        .get("bias_coefficients")
        .ok_or_else(|| "bias_coefficients missing".to_string())?
        .clone();
    let bias: AlibiBias = serde_json::from_value(bias_value)
        .map_err(|error| format!("invalid bias_coefficients: {error}"))?;
    let normal_value = value
        .get("attention_cases")
        .ok_or_else(|| "attention_cases missing".to_string())?
        .clone();
    let normal: Vec<AlibiNormalCase> = serde_json::from_value(normal_value)
        .map_err(|error| format!("invalid attention_cases: {error}"))?;
    let long_value = value
        .get("attention_long_cases")
        .ok_or_else(|| "attention_long_cases missing".to_string())?
        .clone();
    let long: Vec<AlibiLongCase> = serde_json::from_value(long_value)
        .map_err(|error| format!("invalid attention_long_cases: {error}"))?;

    validate_bias_section(&bias)?;
    ensure(
        normal.len() == 2 * NORMAL_CASE_NAMES.len(),
        "attention case count mismatch",
    )?;
    ensure(long.len() == 2, "attention long case count mismatch")?;

    for code in [0u32, 1u32] {
        for name in NORMAL_CASE_NAMES {
            let count = normal
                .iter()
                .filter(|case| case.dtype_code == code && case.name == name)
                .count();
            ensure(count == 1, "attention case multiplicity mismatch")?;
        }
        ensure(
            long.iter().filter(|case| case.dtype_code == code).count() == 1,
            "attention long case multiplicity mismatch",
        )?;
    }
    for case in &normal {
        validate_normal_case(case)?;
    }
    for case in &long {
        validate_long_case(case)?;
    }
    Ok((bias, normal, long))
}

fn parse_capability(value: &Value) -> Result<[i64; 2], String> {
    serde_json::from_value(
        value
            .get("device_capability")
            .and_then(|capability| capability.get("capability"))
            .ok_or_else(|| "device capability missing".to_string())?
            .clone(),
    )
    .map_err(|error| format!("invalid device capability: {error}"))
}

fn read_alibi_fixture() -> AlibiFixture {
    let path = PathBuf::from(
        std::env::var_os(FIXTURE_ENV)
            .expect("Root must provide the matched actual-native Jina ALiBi fixture"),
    );
    let metadata = std::fs::metadata(&path).expect("Jina ALiBi fixture metadata failed");
    assert!(
        metadata.len() <= MAX_FIXTURE_BYTES,
        "native Jina ALiBi fixture exceeds 64MiB"
    );
    let value: Value =
        serde_json::from_slice(&std::fs::read(&path).expect("Jina ALiBi fixture read failed"))
            .expect("Jina ALiBi fixture JSON is invalid");
    crate::jina_primitive_schema::validate_metadata(&value)
        .expect("native Jina ALiBi fixture metadata mismatch");
    let capability = parse_capability(&value).expect("fixture device capability is invalid");
    let (bias, normal, long) =
        parse_attention_sections(&value).expect("native Jina ALiBi fixture sections are invalid");
    AlibiFixture {
        capability,
        bias,
        normal,
        long,
    }
}

fn check_bias_coefficients(bias: &AlibiBias) {
    let mut compared = 0usize;
    for case in &bias.cases {
        for (head, &slope) in SLOPES.iter().enumerate() {
            for distance in 0..DISTANCE_COUNT {
                let value = -slope * distance as f32;
                let actual = round_to_bits(value, case.dtype_code);
                let expected = case.expected_bits[head * DISTANCE_COUNT + distance];
                assert_eq!(
                    actual, expected,
                    "native ALiBi bias coefficient mismatch for dtype {} head {} distance {}",
                    case.dtype, head, distance
                );
                compared += 1;
            }
        }
    }
    assert_eq!(compared, 2 * HEADS * DISTANCE_COUNT);
    println!(
        "jina alibi bias coefficients exact: {compared} bits (both dtype full {}x{})",
        HEADS, DISTANCE_COUNT
    );
}

fn guarded(bits: &[u16]) -> Vec<u16> {
    let mut guarded = vec![SENTINEL; GUARD];
    guarded.extend_from_slice(bits);
    guarded.extend(std::iter::repeat_n(SENTINEL, GUARD));
    guarded
}

fn assert_guards(data: &[u16], elements: usize) {
    assert!(data.len() >= elements + 2 * GUARD);
    assert!(data[..GUARD].iter().all(|&value| value == SENTINEL));
    assert!(data[GUARD + elements..]
        .iter()
        .all(|&value| value == SENTINEL));
}

fn upload_u16(stream: &Stream, host: &[u16]) -> DeviceBuffer<u16> {
    DeviceBuffer::from_host(stream, host).expect("source upload failed")
}

fn upload_i32(stream: &Stream, host: &[i32]) -> DeviceBuffer<i32> {
    DeviceBuffer::from_host(stream, host).expect("cumulative upload failed")
}

fn download_u16(buffer: &DeviceBuffer<u16>, stream: &Stream) -> Vec<u16> {
    let mut data = vec![0u16; buffer.len()];
    buffer
        .copy_to_host(stream, &mut data)
        .expect("u16 download failed");
    data
}

fn download_i32(buffer: &DeviceBuffer<i32>, stream: &Stream) -> Vec<i32> {
    let mut data = vec![0i32; buffer.len()];
    buffer
        .copy_to_host(stream, &mut data)
        .expect("i32 download failed");
    data
}

fn device_address<T: cuda_core::DeviceCopy>(buffer: &DeviceBuffer<T>) -> u64 {
    buffer.cu_deviceptr()
}

fn attention_spec(dtype: DType) -> AttentionSpec {
    AttentionSpec {
        heads: HEADS,
        kv_heads: HEADS,
        head_dim: HEAD_DIM,
        causal: false,
        window: Window::Global,
        dtype,
        scale: Scale::RsqrtHeadDim,
    }
}

fn attention_batch(
    source: &DeviceBuffer<u16>,
    output: &DeviceBuffer<u16>,
    offsets: &DeviceBuffer<i32>,
    tokens: usize,
    batch: usize,
    max_q: usize,
) -> AttentionBatch {
    let q = device_address(source) + (GUARD as u64) * 2;
    let k = q + (WIDTH as u64) * 2;
    let v = q + (2 * WIDTH as u64) * 2;
    let out = device_address(output) + (GUARD as u64) * 2;
    let offsets = device_address(offsets);
    AttentionBatch {
        q,
        k,
        v,
        out,
        q_offsets: offsets,
        k_offsets: offsets,
        q_tokens: tokens,
        k_tokens: tokens,
        batch,
        max_q,
        strides: STRIDES,
    }
}

fn run_attention(
    fa4: &Fa4,
    stream: &Stream,
    source_bits: &[u16],
    cumulative: &[i32],
    tokens: usize,
    batch: usize,
    max_q: usize,
) -> Vec<u16> {
    let source_host = guarded(source_bits);
    let source = upload_u16(stream, &source_host);
    let output_elements = tokens * WIDTH;
    let output_host = guarded(&vec![SENTINEL; output_elements]);
    let output = upload_u16(stream, &output_host);
    let offsets = upload_i32(stream, cumulative);
    let batch_spec = attention_batch(&source, &output, &offsets, tokens, batch, max_q);
    unsafe {
        fa4.forward_batch(stream, &batch_spec)
            .expect("Jina ALiBi forward_batch failed");
    }
    let downloaded_source = download_u16(&source, stream);
    assert_eq!(
        downloaded_source, source_host,
        "Jina ALiBi forward modified its packed source"
    );
    assert_guards(&downloaded_source, source_bits.len());
    let downloaded_output = download_u16(&output, stream);
    assert_guards(&downloaded_output, output_elements);
    let downloaded_offsets = download_i32(&offsets, stream);
    assert_eq!(
        downloaded_offsets, cumulative,
        "Jina ALiBi forward modified cumulative boundaries"
    );
    downloaded_output[GUARD..GUARD + output_elements].to_vec()
}

fn compare_row(code: u32, actual: &[u16], expected: &[u16], label: &str) -> RowMetrics {
    assert_eq!(actual.len(), expected.len(), "{label}: row width mismatch");
    let mut dot = 0.0f64;
    let mut actual_norm = 0.0f64;
    let mut expected_norm = 0.0f64;
    let mut difference_norm = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut exact = 0usize;
    for (index, (&actual_bits, &expected_bits)) in actual.iter().zip(expected).enumerate() {
        assert!(
            bits_finite(actual_bits, code),
            "{label}: nonfinite actual output at column {index}"
        );
        if actual_bits == expected_bits {
            exact += 1;
        }
        let actual_value = bits_to_f32(actual_bits, code) as f64;
        let expected_value = bits_to_f32(expected_bits, code) as f64;
        let difference = actual_value - expected_value;
        dot += actual_value * expected_value;
        actual_norm += actual_value * actual_value;
        expected_norm += expected_value * expected_value;
        difference_norm += difference * difference;
        max_abs = max_abs.max(difference.abs());
    }
    if expected_norm == 0.0 {
        assert!(
            actual_norm == 0.0,
            "{label}: reference row is exactly zero but actual row is not"
        );
        return RowMetrics {
            max_abs: 0.0,
            max_rel: 0.0,
            cosine: 1.0,
            exact,
        };
    }
    let cosine = dot / (actual_norm.sqrt() * expected_norm.sqrt());
    let max_rel = (difference_norm / expected_norm).sqrt();
    let (max_abs_limit, max_rel_limit) = match code {
        0 => (0.03f64, 0.02f64),
        1 => (0.08f64, 0.02f64),
        _ => panic!("invalid ALiBi dtype code {code}"),
    };
    assert!(
        max_abs <= max_abs_limit,
        "{label}: max abs {max_abs} exceeds {max_abs_limit}"
    );
    assert!(cosine >= 0.999, "{label}: cosine {cosine} is below 0.999");
    assert!(
        max_rel <= max_rel_limit,
        "{label}: relative L2 {max_rel} exceeds {max_rel_limit}"
    );
    RowMetrics {
        max_abs,
        max_rel,
        cosine,
        exact,
    }
}

fn log_metrics(dtype: &str, name: &str, rows: usize, elements: usize, metrics: &[RowMetrics]) {
    let mut min_cosine = f64::INFINITY;
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    let mut exact = 0usize;
    for row in metrics {
        min_cosine = min_cosine.min(row.cosine);
        max_abs = max_abs.max(row.max_abs);
        max_rel = max_rel.max(row.max_rel);
        exact += row.exact;
    }
    println!(
        "jina alibi {dtype} {name}: rows={rows} min_cos={min_cosine:.9} \
max_abs={max_abs:.9} max_rel={max_rel:.9} bit_exact={exact}/{elements}"
    );
}

fn run_normal_case(fa4: &Fa4, stream: &Stream, case: &AlibiNormalCase) {
    let tokens = case.lengths.iter().sum::<usize>();
    let actual = run_attention(
        fa4,
        stream,
        &case.source_packed_bits,
        &case.cumulative,
        tokens,
        case.lengths.len(),
        case.max_sequence,
    );
    let mut metrics = Vec::with_capacity(tokens);
    for token in 0..tokens {
        let actual_row = &actual[token * WIDTH..(token + 1) * WIDTH];
        let expected_row = &case.expected_bits[token * WIDTH..(token + 1) * WIDTH];
        let label = format!("{} {} token {}", case.name, case.dtype, token);
        metrics.push(compare_row(
            case.dtype_code,
            actual_row,
            expected_row,
            &label,
        ));
    }
    log_metrics(&case.dtype, &case.name, tokens, actual.len(), &metrics);
}

fn long_source(dtype_code: u32) -> Vec<u16> {
    let mut source = vec![0u16; LONG_LENGTH * PACKED_QKV];
    for token in 0..LONG_LENGTH {
        let token_base = token * PACKED_QKV;
        for (head, &slope) in SLOPES.iter().enumerate() {
            let head_base = head * HEAD_DIM;
            if token == 0 {
                source[token_base + head_base] = round_to_bits(8.0, dtype_code);
            }
            let product = slope * token as f32;
            let key_bits = round_to_bits(product, dtype_code);
            source[token_base + WIDTH + head_base] = key_bits;
            let key_value = bits_to_f32(key_bits, dtype_code);
            let value_bits = round_to_bits((key_value - product) / 16.0, dtype_code);
            let value_base = token_base + 2 * WIDTH + head_base;
            source[value_base..value_base + HEAD_DIM].fill(value_bits);
        }
    }
    source
}

fn run_long_case(fa4: &Fa4, stream: &Stream, case: &AlibiLongCase) {
    let source = long_source(case.dtype_code);
    let actual = run_attention(
        fa4,
        stream,
        &source,
        &case.cumulative,
        LONG_LENGTH,
        1,
        LONG_LENGTH,
    );
    for (index, &bits) in actual.iter().enumerate() {
        assert!(
            bits_finite(bits, case.dtype_code),
            "{} {} long actual output nonfinite at element {index}",
            case.name,
            case.dtype
        );
    }
    let mut metrics = Vec::with_capacity(case.selected_rows.len());
    for (slot, &row) in case.selected_rows.iter().enumerate() {
        let actual_row = &actual[row * WIDTH..(row + 1) * WIDTH];
        let expected_row = &case.expected_selected_bits[slot * WIDTH..(slot + 1) * WIDTH];
        let label = format!("{} {} selected row {}", case.name, case.dtype, row);
        metrics.push(compare_row(
            case.dtype_code,
            actual_row,
            expected_row,
            &label,
        ));
    }
    let selected_elements = case.selected_rows.len() * WIDTH;
    log_metrics(
        &case.dtype,
        &case.name,
        case.selected_rows.len(),
        selected_elements,
        &metrics,
    );
}

#[test]
#[ignore = "requires actual GPU plus INFINITY_OXIDE_JINA_REFERENCE and INFINITY_FA4_ROOT"]
fn gpu_jina_alibi_model_bias_matches_actual_native() {
    let fixture = read_alibi_fixture();
    check_bias_coefficients(&fixture.bias);

    let context = Context::new(0).expect("CUDA context 0 failed");
    context
        .bind_to_thread()
        .expect("CUDA bind_to_thread failed");
    let (major, minor) = context
        .compute_capability()
        .expect("CUDA capability query failed");
    assert_eq!(
        fixture.capability,
        [major as i64, minor as i64],
        "fixture and actual device capability differ"
    );
    let stream = context.new_stream().expect("CUDA stream creation failed");
    let root = PathBuf::from(
        std::env::var_os(ROOT_ENV).expect("INFINITY_FA4_ROOT must point at the FA4 package"),
    );
    let engines = [
        Fa4::new_alibi(
            context.clone(),
            stream.clone(),
            Path::new(&root),
            attention_spec(DType::F16),
        )
        .expect("Jina ALiBi F16 engine creation failed"),
        Fa4::new_alibi(
            context.clone(),
            stream.clone(),
            Path::new(&root),
            attention_spec(DType::Bf16),
        )
        .expect("Jina ALiBi BF16 engine creation failed"),
    ];

    for (dtype_code, engine) in [(0u32, &engines[0]), (1u32, &engines[1])] {
        for case in fixture
            .normal
            .iter()
            .filter(|case| case.dtype_code == dtype_code)
        {
            run_normal_case(engine, &stream, case);
        }
        let long = fixture
            .long
            .iter()
            .find(|case| case.dtype_code == dtype_code)
            .expect("validated long ALiBi case is missing");
        run_long_case(engine, &stream, long);
    }
}

fn valid_cpu_normal_case() -> AlibiNormalCase {
    AlibiNormalCase {
        name: "single_1".to_string(),
        dtype_code: 0,
        dtype: "float16".to_string(),
        case_kind: "standard".to_string(),
        lengths: vec![1],
        cumulative: vec![0, 1],
        max_sequence: 1,
        scale: SCALE_VALUE,
        source_packed_bits: vec![0; PACKED_QKV],
        expected_bits: vec![0; WIDTH],
    }
}

#[test]
fn jina_alibi_fixture_rejects_malformed_without_cuda() {
    let mut case = valid_cpu_normal_case();
    assert!(validate_normal_case(&case).is_ok());

    case.cumulative = vec![0, 2];
    assert!(validate_normal_case(&case).is_err());

    case = valid_cpu_normal_case();
    case.dtype = "bfloat16".to_string();
    assert!(validate_normal_case(&case).is_err());

    case = valid_cpu_normal_case();
    case.source_packed_bits.push(0);
    assert!(validate_normal_case(&case).is_err());

    assert!(checked_packed_len(&[usize::MAX]).is_err());

    let mut bias = AlibiBias {
        slopes: SLOPES.to_vec(),
        distances: (0..DISTANCE_COUNT as i64).collect(),
        cases: [0u32, 1u32]
            .into_iter()
            .map(|code| AlibiBiasCase {
                dtype_code: code,
                dtype: dtype_name(code).unwrap().to_string(),
                shape: [HEADS, DISTANCE_COUNT],
                expected_bits: vec![0; HEADS * DISTANCE_COUNT],
            })
            .collect(),
    };
    assert!(validate_bias_section(&bias).is_ok());
    bias.cases[0].shape = [HEADS, DISTANCE_COUNT - 1];
    assert!(validate_bias_section(&bias).is_err());

    let serialized_long = json!({
        "attention_long_cases": [{"source_packed_bits": []}]
    });
    assert!(validate_long_case_serialization(&serialized_long).is_err());
}
