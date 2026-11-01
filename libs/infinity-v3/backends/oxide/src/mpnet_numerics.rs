// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Public-source MPNet native primitive numerics.
//!
//! The CPU tests build and validate packed records without CUDA or Root gold.
//! The three `#[ignore]` tests consume Root-local oracle artifacts and provider
//! bundles supplied through environment variables; no gold data, checkpoint
//! payload, private kernel, or outside path is embedded in this source.

use super::*;
use serde_json::{Map, Value};

const GUARD: usize = 16;
const SENTINEL: u16 = 0x4242;
const U32_SENTINEL: u32 = 0x4242_4242;
const NORM_WIDTH: usize = 768;
const NORM_CASES: usize = 16;
const ATTENTION_CASES: usize = 14;
const FINITE_BF16_WORDS: usize = 65_280;
const REFERENCE_ENV: &str = "INFINITY_OXIDE_MPNET_REFERENCE";
const MODEL_ENV: &str = "INFINITY_MPNET_MODEL";
const NATIVE_ROOT_ENV: &str = "INFINITY_OXIDE_MPNET_ATTENTION_ROOT";
const KERNEL_ENV: &str = "INFINITY_OXIDE_KERNEL_MODULE";
const SDK_TORCH_VERSION: &str = "2.14.1+cu130";
const SDK_TORCH_GIT: &str = "5c4886908584029761b579af026dcfb627c84070";
const SDK_CUDA_VERSION: &str = "13.0";
const ARTIFACT_SCHEMA: &str = "mpnet-native-primitive-oracle-artifact/v1";
const PACKED_MAX_BYTES: u128 = (1u128 << 63) - 1;

fn sha256(data: &[u8]) -> [u8; 32] {
    use sha2::Digest;
    sha2::Sha256::digest(data).into()
}

fn sha256_hex(data: &[u8]) -> String {
    let mut text = String::with_capacity(64);
    for byte in sha256(data) {
        text.push_str(&format!("{byte:02x}"));
    }
    text
}

fn base64_encode(data: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(data)
}
fn base64_decode(encoded: &str) -> Result<Vec<u8>, String> {
    use base64::Engine;
    let data = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .map_err(|e| format!("packed data_base64 is not canonical or malformed: {e}"))?;
    if base64_encode(&data) != encoded {
        return Err("packed data_base64 is not canonical".into());
    }
    Ok(data)
}

fn packed_item_size(dtype: &str) -> Result<usize, String> {
    Ok(match dtype {
        "bfloat16" | "float16" | "int16" | "uint16" => 2,
        "float32" | "int32" | "uint32" => 4,
        "float64" | "int64" | "uint64" => 8,
        "int8" | "uint8" | "bool" => 1,
        _ => return Err(format!("unsupported packed dtype: {dtype}")),
    })
}

fn is_float_dtype(dtype: &str) -> bool {
    matches!(dtype, "bfloat16" | "float16" | "float32" | "float64")
}

fn packed_is_finite(dtype: &str, data: &[u8]) -> Result<bool, String> {
    let finite = match dtype {
        "bfloat16" => data.chunks_exact(2).all(|chunk| {
            let word = u16::from_le_bytes([chunk[0], chunk[1]]);
            (word >> 7) & 0xff != 0xff
        }),
        "float16" => data.chunks_exact(2).all(|chunk| {
            let word = u16::from_le_bytes([chunk[0], chunk[1]]);
            (word >> 10) & 0x1f != 0x1f
        }),
        "float32" => data.chunks_exact(4).all(|chunk| {
            let word = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            (word >> 23) & 0xff != 0xff
        }),
        "float64" => data.chunks_exact(8).all(|chunk| {
            let word = u64::from_le_bytes(chunk.try_into().expect("eight-byte float64 chunk"));
            (word >> 52) & 0x7ff != 0x7ff
        }),
        _ => true,
    };
    Ok(finite)
}

#[derive(Debug, Clone)]
struct PackedRecord {
    dtype: String,
    shape: Vec<usize>,
    bytes: usize,
    data: Vec<u8>,
    finite: bool,
}

impl PackedRecord {
    fn from_value(value: &Value) -> Result<Self, String> {
        let object = value
            .as_object()
            .ok_or_else(|| "packed record must be an object".to_string())?;
        for key in [
            "dtype",
            "shape",
            "endian",
            "bytes",
            "sha256",
            "data_base64",
            "finite",
        ] {
            if !object.contains_key(key) {
                return Err(format!("packed record missing key: {key}"));
            }
        }
        let dtype = object
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| "packed dtype must be a string".to_string())?
            .to_string();
        let item_size = packed_item_size(&dtype)?;
        let shape_values = object
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| "packed shape must be a list".to_string())?;
        let mut shape = Vec::with_capacity(shape_values.len());
        let mut elements = 1u128;
        for dimension in shape_values {
            let dimension = dimension
                .as_u64()
                .ok_or_else(|| "packed shape entries must be non-negative integers".to_string())?
                as u128;
            if dimension > PACKED_MAX_BYTES {
                return Err("packed shape entry exceeds the supported span".into());
            }
            elements = elements
                .checked_mul(dimension)
                .ok_or_else(|| "packed shape overflows the supported byte length".to_string())?;
            if elements > PACKED_MAX_BYTES {
                return Err("packed shape overflows the supported byte length".into());
            }
            shape.push(
                usize::try_from(dimension)
                    .map_err(|_| "packed shape entry does not fit usize".to_string())?,
            );
        }
        let expected_bytes = elements
            .checked_mul(item_size as u128)
            .ok_or_else(|| "packed byte length overflows the supported span".to_string())?;
        if expected_bytes > PACKED_MAX_BYTES {
            return Err("packed byte length overflows the supported span".into());
        }
        let bytes = usize::try_from(expected_bytes)
            .map_err(|_| "packed byte length does not fit usize".to_string())?;
        let declared = object
            .get("bytes")
            .and_then(Value::as_u64)
            .ok_or_else(|| "packed bytes must be a non-negative integer".to_string())?;
        if declared != bytes as u64 {
            return Err("packed bytes do not match dtype and shape".into());
        }
        let endian = object
            .get("endian")
            .and_then(Value::as_str)
            .ok_or_else(|| "packed endian must be a string".to_string())?;
        if endian != "little" {
            return Err("packed record must be little-endian".into());
        }
        let encoded = object
            .get("data_base64")
            .and_then(Value::as_str)
            .ok_or_else(|| "packed data_base64 must be a string".to_string())?;
        let data = base64_decode(encoded)?;
        if data.len() != bytes {
            return Err("packed decoded length does not match dtype and shape".into());
        }
        let digest = object
            .get("sha256")
            .and_then(Value::as_str)
            .ok_or_else(|| "packed sha256 must be lowercase hexadecimal".to_string())?;
        if digest.len() != 64
            || !digest
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            return Err("packed sha256 must be lowercase hexadecimal".into());
        }
        if digest != sha256_hex(&data) {
            return Err("packed sha256 does not match payload".into());
        }
        let finite = object
            .get("finite")
            .and_then(Value::as_bool)
            .ok_or_else(|| "packed finite must be a boolean".to_string())?;
        if is_float_dtype(&dtype) {
            if finite != packed_is_finite(&dtype, &data)? {
                return Err("packed finite flag does not match payload".into());
            }
        } else if !finite {
            return Err("integer packed records must be finite".into());
        }
        Ok(Self {
            dtype,
            shape,
            bytes,
            data,
            finite,
        })
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn require_dtype(&self, dtype: &str) -> Result<(), String> {
        if self.dtype == dtype {
            Ok(())
        } else {
            Err(format!("expected packed dtype {dtype}, got {}", self.dtype))
        }
    }

    fn require_shape(&self, shape: &[usize]) -> Result<(), String> {
        if self.shape == shape {
            Ok(())
        } else {
            Err(format!(
                "expected packed shape {shape:?}, got {:?}",
                self.shape
            ))
        }
    }

    fn u16_words(&self) -> Result<Vec<u16>, String> {
        if !matches!(self.dtype.as_str(), "bfloat16" | "uint16") {
            return Err(format!("packed dtype {} is not 16-bit words", self.dtype));
        }
        Ok(self
            .data
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
            .collect())
    }

    fn bf16_bits(&self) -> Result<Vec<u16>, String> {
        self.require_dtype("bfloat16")?;
        self.u16_words()
    }

    fn i64_values(&self) -> Result<Vec<i64>, String> {
        self.require_dtype("int64")?;
        Ok(self
            .data
            .chunks_exact(8)
            .map(|chunk| i64::from_le_bytes(chunk.try_into().expect("eight-byte int64 chunk")))
            .collect())
    }
}

fn bf16_record(value: &Value, shape: &[usize], label: &str) -> Result<Vec<u16>, String> {
    let record = PackedRecord::from_value(value).map_err(|error| format!("{label}: {error}"))?;
    record
        .require_dtype("bfloat16")
        .map_err(|error| format!("{label}: {error}"))?;
    record
        .require_shape(shape)
        .map_err(|error| format!("{label}: {error}"))?;
    record
        .bf16_bits()
        .map_err(|error| format!("{label}: {error}"))
}

fn int64_record(value: &Value, shape: &[usize], label: &str) -> Result<Vec<i64>, String> {
    let record = PackedRecord::from_value(value).map_err(|error| format!("{label}: {error}"))?;
    record
        .require_shape(shape)
        .map_err(|error| format!("{label}: {error}"))?;
    record
        .i64_values()
        .map_err(|error| format!("{label}: {error}"))
}

fn require_object<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))
}

fn require_array<'a>(value: &'a Value, label: &str) -> Result<&'a Vec<Value>, String> {
    value
        .as_array()
        .ok_or_else(|| format!("{label} must be an array"))
}

fn require_str<'a>(value: &'a Value, label: &str) -> Result<&'a str, String> {
    value
        .as_str()
        .ok_or_else(|| format!("{label} must be a string"))
}

fn require_usize(value: &Value, label: &str) -> Result<usize, String> {
    let number = value
        .as_u64()
        .ok_or_else(|| format!("{label} must be a non-negative integer"))?;
    usize::try_from(number).map_err(|_| format!("{label} does not fit usize"))
}

fn require_f64(value: &Value, label: &str) -> Result<f64, String> {
    value
        .as_f64()
        .ok_or_else(|| format!("{label} must be a number"))
}

fn require_bool(value: &Value, label: &str) -> Result<bool, String> {
    value
        .as_bool()
        .ok_or_else(|| format!("{label} must be a boolean"))
}

fn field<'a>(object: &'a Map<String, Value>, key: &str) -> Result<&'a Value, String> {
    object
        .get(key)
        .ok_or_else(|| format!("missing field: {key}"))
}

fn load_reference_from(path: Option<&std::path::Path>) -> Result<Value, String> {
    let path =
        path.ok_or_else(|| format!("Set {REFERENCE_ENV} to the Root-local MPNet oracle artifact"))?;
    let bytes =
        std::fs::read(path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    if let Some(expected) = std::env::var_os("INFINITY_OXIDE_MPNET_REFERENCE_SHA256") {
        if expected.to_str() != Some(sha256_hex(&bytes).as_str()) {
            return Err("Root-local native reference file hash changed".into());
        }
    }
    serde_json::from_slice(&bytes)
        .map_err(|error| format!("cannot parse {}: {error}", path.display()))
}

fn load_reference() -> Result<Value, String> {
    let path = std::env::var_os(REFERENCE_ENV)
        .map(std::path::PathBuf::from)
        .ok_or_else(|| format!("Set {REFERENCE_ENV} to the Root-local MPNet oracle artifact"))?;
    load_reference_from(Some(&path))
}

fn is_lower_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn validate_reference(reference: &Value) -> Result<&Map<String, Value>, String> {
    let object = require_object(reference, "reference")?;
    let schema = require_str(field(object, "schema")?, "schema")?;
    if schema != ARTIFACT_SCHEMA {
        return Err(format!("unexpected reference schema: {schema}"));
    }
    if require_str(field(object, "mode")?, "mode")? != "actual-weights" {
        return Err("reference must be the actual-weights oracle artifact".into());
    }
    if require_str(field(object, "device")?, "device")? != "cuda:0" {
        return Err("reference was not recorded on a CUDA device".into());
    }
    let torch = require_str(field(object, "torch_version")?, "torch_version")?;
    if torch != SDK_TORCH_VERSION {
        return Err(format!(
            "reference Torch {torch} does not match {SDK_TORCH_VERSION}"
        ));
    }
    if require_str(field(object, "torch_git_version")?, "torch_git_version")? != SDK_TORCH_GIT {
        return Err("reference Torch git identity does not match the SDK".into());
    }
    if require_str(field(object, "torch_cuda_version")?, "torch_cuda_version")? != SDK_CUDA_VERSION
    {
        return Err("reference CUDA version does not match CUDA 13.0".into());
    }
    if require_str(field(object, "endian")?, "endian")? != "little" {
        return Err("reference must be little-endian".into());
    }
    for key in ["metadata_sha256", "oracle_sha256", "artifact_sha256"] {
        let digest = require_str(field(object, key)?, key)?;
        if !is_lower_hex(digest) {
            return Err(format!("{key} must be lowercase SHA-256 hexadecimal"));
        }
    }
    let sources = require_object(field(object, "source_sha256")?, "source_sha256")?;
    let expected = [
        "mpnet_oracle_spec.py",
        "mpnet_oracle.py",
        "mpnet_primitive_oracle.cpp",
        "mpnet_metadata.py",
        "native-source/models/encoder_models.cpp",
        "native-source/helpers/dense_mlp.cpp",
        "native-source/helpers/dense_mlp.h",
    ];
    if sources.len() != expected.len()
        || expected.iter().any(|key| {
            sources
                .get(*key)
                .and_then(Value::as_str)
                .is_none_or(|v| !is_lower_hex(v))
        })
    {
        return Err("source_sha256 must bind all seven exact public oracle/native sources".into());
    }
    let checks = require_object(field(object, "checks")?, "checks")?;
    for flag in [
        "finite",
        "strict_preativation_bits",
        "native_guards",
        "masking_witness",
        "weights_read",
        "cuda_initialized",
    ] {
        if !require_bool(field(checks, flag)?, flag)? {
            return Err(format!("native reference check did not pass: {flag}"));
        }
    }
    if !require_bool(field(object, "cuda_initialized")?, "cuda_initialized")? {
        return Err("native reference did not initialize CUDA".into());
    }
    require_object(field(object, "records")?, "records")
}

fn guarded_u16(stream: &Stream, words: &[u16], padding: usize) -> Result<GuardedU16, String> {
    let mut host = vec![SENTINEL; padding];
    host.extend_from_slice(words);
    host.extend(std::iter::repeat_n(SENTINEL, GUARD));
    let buffer = DeviceBuffer::from_host(stream, &host)
        .map_err(|error| format!("uploading u16 guard buffer: {error:?}"))?;
    Ok(GuardedU16 {
        buffer,
        padding,
        host,
    })
}

struct GuardedU16 {
    buffer: DeviceBuffer<u16>,
    padding: usize,
    host: Vec<u16>,
}

impl GuardedU16 {
    fn ptr(&self) -> u64 {
        self.buffer.cu_deviceptr() + (self.padding * 2) as u64
    }

    fn read(&self, stream: &Stream, length: usize, label: &str) -> Result<Vec<u16>, String> {
        let mut actual = vec![0u16; self.buffer.len()];
        self.buffer
            .copy_to_host(stream, &mut actual)
            .map_err(|error| format!("{label}: reading device buffer: {error:?}"))?;
        if actual[..self.padding].iter().any(|word| *word != SENTINEL) {
            return Err(format!("{label}: prefix guard was modified"));
        }
        let end = self
            .padding
            .checked_add(length)
            .ok_or_else(|| format!("{label}: guard length overflow"))?;
        if end > actual.len() || actual[end..].iter().any(|word| *word != SENTINEL) {
            return Err(format!("{label}: suffix guard was modified"));
        }
        Ok(actual[self.padding..end].to_vec())
    }

    fn assert_unchanged(&self, stream: &Stream, label: &str) -> Result<(), String> {
        let mut actual = vec![0u16; self.buffer.len()];
        self.buffer
            .copy_to_host(stream, &mut actual)
            .map_err(|error| format!("{label}: reading device buffer: {error:?}"))?;
        if actual != self.host {
            return Err(format!("{label}: source or guard bits were modified"));
        }
        Ok(())
    }
}

fn guarded_u32(stream: &Stream, words: &[u32], padding: usize) -> Result<GuardedU32, String> {
    let mut host = vec![U32_SENTINEL; padding];
    host.extend_from_slice(words);
    host.extend(std::iter::repeat_n(U32_SENTINEL, GUARD));
    let buffer = DeviceBuffer::from_host(stream, &host)
        .map_err(|error| format!("uploading u32 guard buffer: {error:?}"))?;
    Ok(GuardedU32 {
        buffer,
        padding,
        host,
    })
}

struct GuardedU32 {
    buffer: DeviceBuffer<u32>,
    padding: usize,
    host: Vec<u32>,
}

impl GuardedU32 {
    fn ptr(&self) -> u64 {
        self.buffer.cu_deviceptr() + (self.padding * 4) as u64
    }

    fn assert_unchanged(&self, stream: &Stream, label: &str) -> Result<(), String> {
        let mut actual = vec![0u32; self.buffer.len()];
        self.buffer
            .copy_to_host(stream, &mut actual)
            .map_err(|error| format!("{label}: reading device buffer: {error:?}"))?;
        if actual != self.host {
            return Err(format!("{label}: source or guard words were modified"));
        }
        Ok(())
    }
}

fn launch_and_sync(
    kernel: &Function,
    stream: &Stream,
    grid: (u32, u32, u32),
    threads: u32,
    args: &mut [*mut c_void],
    label: &str,
) -> Result<(), String> {
    let launch = launch_threads(kernel, stream, grid, threads, 0, args)
        .map_err(|error| format!("{label}: {error}"));
    let sync = stream
        .synchronize()
        .map_err(|error| format!("{label}: stream synchronize failed: {error}"));
    launch?;
    sync
}

fn cumulative(lengths: &[usize]) -> Result<Vec<u32>, String> {
    let mut total = 0usize;
    let mut values = Vec::with_capacity(lengths.len() + 1);
    values.push(0u32);
    for length in lengths {
        total = total
            .checked_add(*length)
            .ok_or_else(|| "cumulative length overflow".to_string())?;
        values.push(u32::try_from(total).map_err(|_| "cumulative length exceeds u32".to_string())?);
    }
    Ok(values)
}

fn u32_indices(values: &[i64], add: u64) -> Result<Vec<u32>, String> {
    values
        .iter()
        .map(|value| {
            let value = u64::try_from(*value)
                .map_err(|_| "negative index in packed int64 record".to_string())?;
            let value = value
                .checked_add(add)
                .ok_or_else(|| "index overflow while applying position offset".to_string())?;
            u32::try_from(value).map_err(|_| "index exceeds u32".to_string())
        })
        .collect()
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    (rounded >> 16) as u16
}

fn bf16_add(left: u16, right: u16) -> u16 {
    f32_to_bf16_bits(bf16_to_f32(left) + bf16_to_f32(right))
}

fn bf16_finite(bits: u16) -> bool {
    (bits >> 7) & 0xff != 0xff
}

fn deterministic_bf16(count: usize, modulus: usize, offset: usize, scale: f32) -> Vec<u16> {
    (0..count)
        .map(|index| {
            let value = ((index % modulus) as i32 - offset as i32) as f32 * scale;
            f32_to_bf16_bits(value)
        })
        .collect()
}

fn finite_bf16_patterns() -> Vec<u16> {
    (0..=u16::MAX)
        .filter(|word| bf16_finite(*word))
        .collect::<Vec<_>>()
}

fn shared_record<'a>(shared: &'a Map<String, Value>, name: &str) -> Result<&'a Value, String> {
    shared
        .get(name)
        .ok_or_else(|| format!("missing shared norm input: {name}"))
}

fn assert_attention_row(actual: &[u16], expected: &[u16], label: &str) {
    assert_eq!(
        actual, expected,
        "{label}: native bucket attention must match every BF16 bit"
    );
    assert_eq!(actual.len(), NORM_WIDTH, "{label}: row width");
    assert_eq!(expected.len(), NORM_WIDTH, "{label}: reference row width");
    let mut dot = 0.0f64;
    let mut actual_sq = 0.0f64;
    let mut expected_sq = 0.0f64;
    let mut difference_sq = 0.0f64;
    let mut max_abs = 0.0f64;
    for (actual, expected) in actual.iter().zip(expected) {
        let actual = bf16_to_f32(*actual);
        let expected = bf16_to_f32(*expected);
        assert!(actual.is_finite(), "{label}: non-finite actual BF16");
        assert!(expected.is_finite(), "{label}: non-finite reference BF16");
        let difference = (actual - expected).abs();
        dot += actual as f64 * expected as f64;
        actual_sq += actual as f64 * actual as f64;
        expected_sq += expected as f64 * expected as f64;
        difference_sq += difference as f64 * difference as f64;
        max_abs = max_abs.max(difference as f64);
    }
    let cosine = if actual_sq == 0.0 && expected_sq == 0.0 {
        1.0
    } else if actual_sq == 0.0 || expected_sq == 0.0 {
        0.0
    } else {
        dot / (actual_sq.sqrt() * expected_sq.sqrt())
    };
    let relative_l2 = if expected_sq == 0.0 {
        0.0
    } else {
        (difference_sq / expected_sq).sqrt()
    };
    assert!(
        cosine >= 0.999,
        "{label}: cosine {cosine} is below the qualified BF16 guard"
    );
    assert!(
        max_abs <= 0.08,
        "{label}: max absolute error {max_abs} exceeds 0.08"
    );
    assert!(
        relative_l2 <= 0.02,
        "{label}: relative L2 error {relative_l2} exceeds 0.02"
    );
}

#[test]
fn sha256_and_base64_known_vectors() {
    assert_eq!(
        sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    for payload in [
        b"".as_slice(),
        b"f".as_slice(),
        b"fo".as_slice(),
        b"foo".as_slice(),
        b"hello world".as_slice(),
    ] {
        let encoded = base64_encode(payload);
        assert_eq!(base64_decode(&encoded).unwrap(), payload);
    }
    assert_eq!(base64_encode(b"hello"), "aGVsbG8=");
    assert_eq!(base64_encode(b"hello world"), "aGVsbG8gd29ybGQ=");
}

fn make_record(dtype: &str, shape: &[usize], payload: &[u8], finite: bool) -> Value {
    serde_json::json!({
        "dtype": dtype,
        "shape": shape,
        "endian": "little",
        "bytes": payload.len(),
        "sha256": sha256_hex(payload),
        "data_base64": base64_encode(payload),
        "finite": finite,
    })
}

fn words_payload(words: &[u16]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

fn values_payload(values: &[i64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

#[test]
fn packed_reader_accepts_canonical_records() {
    let words = [0x3f80u16, 0xbf80, 0x0000, 0x7f7f];
    let record = PackedRecord::from_value(&make_record(
        "bfloat16",
        &[2, 2],
        &words_payload(&words),
        true,
    ))
    .unwrap();
    assert_eq!(record.dtype, "bfloat16");
    assert_eq!(record.shape(), &[2, 2]);
    assert_eq!(record.bytes, 8);
    assert!(record.finite);
    assert_eq!(record.bf16_bits().unwrap(), words.to_vec());

    let integers = [0i64, 7, -9, i64::MAX, i64::MIN];
    let record = PackedRecord::from_value(&make_record(
        "int64",
        &[5],
        &values_payload(&integers),
        true,
    ))
    .unwrap();
    assert_eq!(record.i64_values().unwrap(), integers);
    assert!(record.bf16_bits().is_err());

    let record = PackedRecord::from_value(&make_record(
        "uint16",
        &[3],
        &words_payload(&[1, 2, 3]),
        true,
    ))
    .unwrap();
    assert_eq!(record.u16_words().unwrap(), vec![1, 2, 3]);
}

#[test]
fn packed_reader_rejects_corrupt_base64() {
    let mut record = make_record("bfloat16", &[1], &words_payload(&[0x3f80]), true);
    record["data_base64"] = Value::String("@@@=".into());
    assert!(PackedRecord::from_value(&record)
        .unwrap_err()
        .contains("malformed"));

    let mut record = make_record("uint8", &[1], &[0], true);
    record["data_base64"] = Value::String("AB==".into());
    assert!(PackedRecord::from_value(&record)
        .unwrap_err()
        .contains("not canonical"));

    let mut record = make_record("bfloat16", &[1], &words_payload(&[0x3f80]), true);
    record["data_base64"] = Value::String("gA".into());
    assert!(PackedRecord::from_value(&record)
        .unwrap_err()
        .contains("malformed"));

    let mut record = make_record("bfloat16", &[2], &words_payload(&[0x3f80, 0xbf80]), true);
    record["data_base64"] = Value::String("gA==".into());
    assert!(PackedRecord::from_value(&record)
        .unwrap_err()
        .contains("decoded length"));
}

#[test]
fn packed_reader_rejects_endian_shape_and_overflow() {
    let mut record = make_record("bfloat16", &[1], &words_payload(&[0x3f80]), true);
    record["endian"] = Value::String("big".into());
    assert!(PackedRecord::from_value(&record)
        .unwrap_err()
        .contains("little-endian"));

    let mut record = make_record("bfloat16", &[1], &words_payload(&[0x3f80]), true);
    record["shape"] = serde_json::json!(["1"]);
    assert!(PackedRecord::from_value(&record)
        .unwrap_err()
        .contains("non-negative integers"));

    let mut record = make_record("bfloat16", &[1], &words_payload(&[0x3f80]), true);
    record["shape"] = serde_json::json!([u64::MAX]);
    assert!(PackedRecord::from_value(&record)
        .unwrap_err()
        .contains("supported span"));

    let mut record = make_record("bfloat16", &[1], &words_payload(&[0x3f80]), true);
    record["shape"] = serde_json::json!([4_294_967_296u64, 4_294_967_296u64]);
    assert!(PackedRecord::from_value(&record)
        .unwrap_err()
        .contains("overflows"));

    let mut record = make_record("bfloat16", &[1], &words_payload(&[0x3f80]), true);
    record["bytes"] = Value::from(4);
    assert!(PackedRecord::from_value(&record)
        .unwrap_err()
        .contains("do not match dtype and shape"));
}

#[test]
fn packed_reader_rejects_hash_finiteness_and_missing_fields() {
    let mut record = make_record("bfloat16", &[1], &words_payload(&[0x3f80]), true);
    record["sha256"] = Value::String("0".repeat(64));
    assert!(PackedRecord::from_value(&record)
        .unwrap_err()
        .contains("does not match payload"));

    let mut record = make_record("bfloat16", &[1], &words_payload(&[0x7f80]), true);
    record["finite"] = Value::Bool(true);
    assert!(PackedRecord::from_value(&record)
        .unwrap_err()
        .contains("finite flag"));

    let record = make_record("int64", &[1], &values_payload(&[1]), false);
    assert!(PackedRecord::from_value(&record)
        .unwrap_err()
        .contains("integer packed records must be finite"));

    let mut record = make_record("bfloat16", &[1], &words_payload(&[0x3f80]), true);
    record.as_object_mut().unwrap().remove("sha256");
    assert!(PackedRecord::from_value(&record)
        .unwrap_err()
        .contains("missing key: sha256"));

    let record = make_record("float8_e4m3fn", &[1], &[0], true);
    assert!(PackedRecord::from_value(&record)
        .unwrap_err()
        .contains("unsupported packed dtype"));
}

#[test]
fn reference_loader_rejects_missing_environment_or_file() {
    let error = load_reference_from(None).unwrap_err();
    assert!(error.contains(REFERENCE_ENV));

    let missing = std::env::temp_dir().join(format!(
        "mpnet-reference-not-present-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let error = load_reference_from(Some(&missing)).unwrap_err();
    assert!(error.contains("cannot read"));
}

#[test]
fn reference_envelope_binds_public_sdk_and_source_hashes() {
    let digest = "0".repeat(64);
    let mut reference = serde_json::json!({
        "schema": ARTIFACT_SCHEMA,
        "mode": "actual-weights",
        "device": "cuda:0",
        "endian": "little",
        "torch_version": "2.14.1+cu130",
        "torch_git_version": SDK_TORCH_GIT,
        "torch_cuda_version": SDK_CUDA_VERSION,
        "metadata_sha256": digest.as_str(),
        "source_sha256": {
            "mpnet_oracle_spec.py": digest.as_str(), "mpnet_oracle.py": digest.as_str(),
            "mpnet_primitive_oracle.cpp": digest.as_str(), "mpnet_metadata.py": digest.as_str(),
            "native-source/models/encoder_models.cpp": digest.as_str(),
            "native-source/helpers/dense_mlp.cpp": digest.as_str(),
            "native-source/helpers/dense_mlp.h": digest.as_str()
        },
        "oracle_sha256": digest.as_str(),
        "artifact_sha256": digest.as_str(),
        "checks": {"finite":true, "strict_preativation_bits":true,"native_guards":true,
            "masking_witness":true,"weights_read":true,"cuda_initialized":true},
        "cuda_initialized":true,
        "records": {},
    });
    validate_reference(&reference).unwrap();
    reference["torch_git_version"] = Value::String("0".repeat(40));
    assert!(validate_reference(&reference)
        .unwrap_err()
        .contains("git identity"));
    reference["torch_git_version"] = Value::String(SDK_TORCH_GIT.into());
    reference["source_sha256"] = Value::String("not-a-hash".into());
    assert!(validate_reference(&reference)
        .unwrap_err()
        .contains("source_sha256"));
    reference["source_sha256"] = serde_json::json!({
"mpnet_oracle_spec.py": digest.as_str(),"mpnet_oracle.py": digest.as_str(),"mpnet_primitive_oracle.cpp": digest.as_str(),"mpnet_metadata.py": digest.as_str(),"native-source/models/encoder_models.cpp": digest.as_str(),"native-source/helpers/dense_mlp.cpp": digest.as_str(),"native-source/helpers/dense_mlp.h": digest.as_str()});
    reference["mode"] = Value::String("cpu-selfcheck".into());
    assert!(validate_reference(&reference)
        .unwrap_err()
        .contains("actual-weights"));
}

#[test]
#[ignore = "requires CUDA, INFINITY_OXIDE_KERNEL_MODULE, and INFINITY_OXIDE_MPNET_REFERENCE"]
fn gpu_mpnet_native_aten_norms() {
    let reference = load_reference().unwrap();
    let records = validate_reference(&reference).unwrap();
    let norm = require_object(
        field(records, "synthetic_norm_cases").unwrap(),
        "synthetic_norm_cases",
    )
    .unwrap();
    let shared = require_object(field(norm, "shared_inputs").unwrap(), "shared_inputs").unwrap();
    let cases = require_array(field(norm, "cases").unwrap(), "cases").unwrap();
    assert_eq!(cases.len(), NORM_CASES, "synthetic norm case count");
    assert_eq!(
        require_usize(field(norm, "case_count").unwrap(), "case_count").unwrap(),
        NORM_CASES
    );

    let context = Context::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let root = std::path::PathBuf::from(
        std::env::var_os(KERNEL_ENV).expect("Set INFINITY_OXIDE_KERNEL_MODULE"),
    );
    let image = if root.is_dir() {
        let (major, minor) = context.compute_capability().unwrap();
        root.join(format!("sm_{major}{minor}"))
            .join("kernels.cubin")
    } else {
        root
    };
    let module = context
        .load_module_from_image(&std::fs::read(&image).unwrap())
        .unwrap();
    let embedding = module
        .load_function("infinity_embedding_norm_mpnet_768_aten")
        .unwrap();
    let layer_norm = module
        .load_function("infinity_layer_norm_mpnet_768_aten")
        .unwrap();

    for unaligned in [false, true] {
        let padding = GUARD + usize::from(unaligned);
        let mut word_tables = std::collections::BTreeMap::new();
        let mut position_tables = std::collections::BTreeMap::new();
        for pattern in ["near_constant", "cancellation"] {
            let word = bf16_record(
                shared_record(shared, &format!("word_{pattern}")).unwrap(),
                &[7, NORM_WIDTH],
                "word table",
            )
            .unwrap();
            let position = bf16_record(
                shared_record(shared, &format!("position_{pattern}")).unwrap(),
                &[514, NORM_WIDTH],
                "position table",
            )
            .unwrap();
            word_tables.insert(pattern, guarded_u16(&stream, &word, padding).unwrap());
            position_tables.insert(pattern, guarded_u16(&stream, &position, padding).unwrap());
        }
        let gamma = bf16_record(
            shared_record(shared, "gamma").unwrap(),
            &[NORM_WIDTH],
            "gamma",
        )
        .unwrap();
        let beta = bf16_record(
            shared_record(shared, "beta").unwrap(),
            &[NORM_WIDTH],
            "beta",
        )
        .unwrap();
        let gamma = guarded_u16(&stream, &gamma, padding).unwrap();
        let beta = guarded_u16(&stream, &beta, padding).unwrap();
        let zero_words = vec![0u16; 128 * NORM_WIDTH];
        let zeros = guarded_u16(&stream, &zero_words, padding).unwrap();

        for case in cases {
            let case = require_object(case, "norm case").unwrap();
            let rows = require_usize(field(case, "rows").unwrap(), "rows").unwrap();
            assert!(rows > 0 && rows <= 128, "norm rows must be bounded");
            let epsilon = require_f64(field(case, "epsilon").unwrap(), "epsilon").unwrap() as f32;
            assert!(epsilon == 1e-5f32 || epsilon == 1e-12f32);
            let pattern = require_str(field(case, "pattern").unwrap(), "pattern").unwrap();
            assert!(matches!(pattern, "near_constant" | "cancellation"));
            let inputs = require_object(field(case, "inputs").unwrap(), "inputs").unwrap();
            let word_ref = require_str(field(inputs, "word").unwrap(), "word").unwrap();
            let position_ref = require_str(field(inputs, "position").unwrap(), "position").unwrap();
            assert_eq!(word_ref, format!("word_{pattern}").as_str());
            assert_eq!(position_ref, format!("position_{pattern}").as_str());
            assert_eq!(
                require_str(field(inputs, "gamma").unwrap(), "gamma").unwrap(),
                "gamma"
            );
            assert_eq!(
                require_str(field(inputs, "beta").unwrap(), "beta").unwrap(),
                "beta"
            );

            let ids = int64_record(
                shared_record(
                    shared,
                    require_str(field(inputs, "ids").unwrap(), "ids").unwrap(),
                )
                .unwrap(),
                &[rows],
                "ids",
            )
            .unwrap();
            let positions = int64_record(
                shared_record(
                    shared,
                    require_str(field(inputs, "positions").unwrap(), "positions").unwrap(),
                )
                .unwrap(),
                &[rows],
                "positions",
            )
            .unwrap();
            let ids = u32_indices(&ids, 0).unwrap();
            let positions = u32_indices(&positions, 2).unwrap();
            let word_bits = word_tables.get(pattern).unwrap();
            let position_bits = position_tables.get(pattern).unwrap();
            let word_words = bf16_record(
                shared_record(shared, &format!("word_{pattern}")).unwrap(),
                &[7, NORM_WIDTH],
                "word table",
            )
            .unwrap();
            let position_words = bf16_record(
                shared_record(shared, &format!("position_{pattern}")).unwrap(),
                &[514, NORM_WIDTH],
                "position table",
            )
            .unwrap();
            let residual = bf16_record(
                shared_record(
                    shared,
                    require_str(field(inputs, "residual").unwrap(), "residual").unwrap(),
                )
                .unwrap(),
                &[rows, NORM_WIDTH],
                "residual",
            )
            .unwrap();
            let update = bf16_record(
                shared_record(
                    shared,
                    require_str(field(inputs, "update").unwrap(), "update").unwrap(),
                )
                .unwrap(),
                &[rows, NORM_WIDTH],
                "update",
            )
            .unwrap();

            let embedding_norm =
                require_object(field(case, "embedding_norm").unwrap(), "embedding_norm").unwrap();
            let expected_prenorm = bf16_record(
                field(embedding_norm, "prenorm").unwrap(),
                &[rows, NORM_WIDTH],
                "embedding prenorm",
            )
            .unwrap();
            let expected_normed = bf16_record(
                field(embedding_norm, "normed").unwrap(),
                &[rows, NORM_WIDTH],
                "embedding normed",
            )
            .unwrap();
            let residual_norm =
                require_object(field(case, "residual_norm").unwrap(), "residual_norm").unwrap();
            let expected_sum = bf16_record(
                field(residual_norm, "sum").unwrap(),
                &[rows, NORM_WIDTH],
                "residual sum",
            )
            .unwrap();
            let expected_residual_normed = bf16_record(
                field(residual_norm, "normed").unwrap(),
                &[rows, NORM_WIDTH],
                "residual normed",
            )
            .unwrap();

            for row in 0..rows {
                let id = ids[row] as usize;
                let position = positions[row] as usize;
                assert!(id < 7, "norm id index must stay inside the word table");
                assert!(
                    position < 514,
                    "norm position index must stay inside the position table"
                );
                for column in 0..NORM_WIDTH {
                    let index = row * NORM_WIDTH + column;
                    assert_eq!(
                        expected_prenorm[index],
                        bf16_add(
                            word_words[id * NORM_WIDTH + column],
                            position_words[position * NORM_WIDTH + column]
                        ),
                        "gold embedding prenorm add row={row} column={column}"
                    );
                    assert_eq!(
                        expected_sum[index],
                        bf16_add(residual[index], update[index]),
                        "gold residual sum row={row} column={column}"
                    );
                }
            }

            let ids_device = DeviceBuffer::from_host(&stream, &ids).unwrap();
            let positions_device = DeviceBuffer::from_host(&stream, &positions).unwrap();
            let output = guarded_u16(&stream, &vec![SENTINEL; rows * NORM_WIDTH], padding).unwrap();
            let mut ids_ptr = ids_device.cu_deviceptr();
            let mut positions_ptr = positions_device.cu_deviceptr();
            let mut types_ptr = 0u64;
            let mut word_ptr = word_bits.ptr();
            let mut position_ptr = position_bits.ptr();
            let mut type_weight_ptr = 0u64;
            let mut gamma_ptr = gamma.ptr();
            let mut beta_ptr = beta.ptr();
            let mut out_ptr = output.ptr();
            let mut rows_u32 = u32::try_from(rows).unwrap();
            let mut width = NORM_WIDTH as u32;
            let mut eps = epsilon;
            launch_and_sync(
                &embedding,
                &stream,
                (rows_u32, 1, 1),
                128,
                &mut [
                    arg(&mut ids_ptr),
                    arg(&mut positions_ptr),
                    arg(&mut types_ptr),
                    arg(&mut word_ptr),
                    arg(&mut position_ptr),
                    arg(&mut type_weight_ptr),
                    arg(&mut gamma_ptr),
                    arg(&mut beta_ptr),
                    arg(&mut out_ptr),
                    arg(&mut rows_u32),
                    arg(&mut width),
                    arg(&mut eps),
                ],
                "embedding norm",
            )
            .unwrap();
            let actual_normed = output
                .read(&stream, rows * NORM_WIDTH, "embedding output")
                .unwrap();
            assert_eq!(
                actual_normed, expected_normed,
                "native embedding norm bits row={rows} pattern={pattern} eps={epsilon}"
            );

            let norm_input = guarded_u16(&stream, &expected_prenorm, padding).unwrap();
            let norm_output =
                guarded_u16(&stream, &vec![SENTINEL; rows * NORM_WIDTH], padding).unwrap();
            let mut input_ptr = norm_input.ptr();
            let mut residual_ptr = zeros.ptr();
            let mut bias_ptr = 0u64;
            let mut gamma_ptr = gamma.ptr();
            let mut beta_ptr = beta.ptr();
            let mut out_ptr = norm_output.ptr();
            let mut rows_u32 = u32::try_from(rows).unwrap();
            let mut width = NORM_WIDTH as u32;
            let mut eps = epsilon;
            launch_and_sync(
                &layer_norm,
                &stream,
                (rows_u32, 1, 1),
                128,
                &mut [
                    arg(&mut input_ptr),
                    arg(&mut residual_ptr),
                    arg(&mut bias_ptr),
                    arg(&mut gamma_ptr),
                    arg(&mut beta_ptr),
                    arg(&mut out_ptr),
                    arg(&mut rows_u32),
                    arg(&mut width),
                    arg(&mut eps),
                ],
                "layer norm from prenorm",
            )
            .unwrap();
            let actual_normed = norm_output
                .read(&stream, rows * NORM_WIDTH, "layer norm output")
                .unwrap();
            assert_eq!(
                actual_normed, expected_normed,
                "native layer norm bits row={rows} pattern={pattern} eps={epsilon}"
            );
            norm_input
                .assert_unchanged(&stream, "prenorm input")
                .unwrap();

            for alias in ["residual", "input"] {
                let update_guard = guarded_u16(&stream, &update, padding).unwrap();
                let residual_guard = guarded_u16(&stream, &residual, padding).unwrap();
                let mut input_ptr = update_guard.ptr();
                let mut residual_ptr = residual_guard.ptr();
                let mut bias_ptr = 0u64;
                let mut gamma_ptr = gamma.ptr();
                let mut beta_ptr = beta.ptr();
                let mut out_ptr = if alias == "residual" {
                    residual_guard.ptr()
                } else {
                    update_guard.ptr()
                };
                let mut rows_u32 = u32::try_from(rows).unwrap();
                let mut width = NORM_WIDTH as u32;
                let mut eps = epsilon;
                launch_and_sync(
                    &layer_norm,
                    &stream,
                    (rows_u32, 1, 1),
                    128,
                    &mut [
                        arg(&mut input_ptr),
                        arg(&mut residual_ptr),
                        arg(&mut bias_ptr),
                        arg(&mut gamma_ptr),
                        arg(&mut beta_ptr),
                        arg(&mut out_ptr),
                        arg(&mut rows_u32),
                        arg(&mut width),
                        arg(&mut eps),
                    ],
                    "residual layer norm aliased",
                )
                .unwrap();
                let target = if alias == "residual" {
                    &residual_guard
                } else {
                    &update_guard
                };
                let actual = target
                    .read(&stream, rows * NORM_WIDTH, "aliased residual norm output")
                    .unwrap();
                assert_eq!(
                    actual, expected_residual_normed,
                    "native residual norm alias={alias} row={rows} pattern={pattern} eps={epsilon}"
                );
                if alias == "residual" {
                    update_guard
                        .assert_unchanged(&stream, "residual norm update input")
                        .unwrap();
                } else {
                    residual_guard
                        .assert_unchanged(&stream, "residual norm residual input")
                        .unwrap();
                }
            }
        }
        for table in word_tables.values() {
            table.assert_unchanged(&stream, "word table input").unwrap();
        }
        for table in position_tables.values() {
            table
                .assert_unchanged(&stream, "position table input")
                .unwrap();
        }
        gamma.assert_unchanged(&stream, "gamma input").unwrap();
        beta.assert_unchanged(&stream, "beta input").unwrap();
    }
    stream.synchronize().unwrap();
}

#[test]
#[ignore = "requires CUDA, INFINITY_OXIDE_KERNEL_MODULE, and INFINITY_OXIDE_MPNET_REFERENCE"]
fn gpu_mpnet_exhaustive_gelu_none() {
    let reference = load_reference().unwrap();
    let records = validate_reference(&reference).unwrap();
    let extended = require_object(
        field(records, "extended_native").unwrap(),
        "extended_native",
    )
    .unwrap();
    let input_words = bf16_record(
        field(extended, "input").unwrap(),
        &[FINITE_BF16_WORDS],
        "gelu input",
    )
    .unwrap();
    let expected = bf16_record(
        field(extended, "output").unwrap(),
        &[FINITE_BF16_WORDS],
        "gelu output",
    )
    .unwrap();
    let patterns = finite_bf16_patterns();
    assert_eq!(patterns.len(), FINITE_BF16_WORDS);
    assert_eq!(
        input_words, patterns,
        "gold GELU input must be every finite BF16 word"
    );
    assert!(input_words.iter().all(|word| bf16_finite(*word)));
    assert!(expected.iter().all(|word| bf16_finite(*word)));
    let recipe =
        require_object(field(extended, "pattern_recipe").unwrap(), "pattern_recipe").unwrap();
    assert_eq!(
        require_usize(field(recipe, "word_count").unwrap(), "word_count").unwrap(),
        FINITE_BF16_WORDS
    );
    assert_eq!(
        require_usize(field(recipe, "excluded_count").unwrap(), "excluded_count").unwrap(),
        65_536 - FINITE_BF16_WORDS
    );
    let word_sha = require_str(field(recipe, "word_sha256").unwrap(), "word_sha256").unwrap();
    assert_eq!(word_sha, sha256_hex(&words_payload(&input_words)));

    let context = Context::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let root = std::path::PathBuf::from(
        std::env::var_os(KERNEL_ENV).expect("Set INFINITY_OXIDE_KERNEL_MODULE"),
    );
    let image = if root.is_dir() {
        let (major, minor) = context.compute_capability().unwrap();
        root.join(format!("sm_{major}{minor}"))
            .join("kernels.cubin")
    } else {
        root
    };
    let module = context
        .load_module_from_image(&std::fs::read(&image).unwrap())
        .unwrap();
    let gelu = module.load_function("infinity_gelu_exact").unwrap();

    for unaligned in [false, true] {
        let padding = GUARD + usize::from(unaligned);
        for count in [1usize, 3, 5, 128, FINITE_BF16_WORDS] {
            let buffer = guarded_u16(&stream, &input_words[..count], padding).unwrap();
            let mut input = buffer.ptr();
            let mut elements = u32::try_from(count).unwrap();
            launch_and_sync(
                &gelu,
                &stream,
                (elements.div_ceil(256), 1, 1),
                256,
                &mut [arg(&mut input), arg(&mut elements)],
                "exact GELU",
            )
            .unwrap();
            let actual = buffer.read(&stream, count, "GELU output").unwrap();
            assert_eq!(
                actual,
                expected[..count].to_vec(),
                "exact GELU bits differ for count={count} unaligned={unaligned}"
            );
            assert!(
                actual.iter().all(|word| bf16_finite(*word)),
                "GELU output must stay finite for count={count}"
            );
        }
    }
    stream.synchronize().unwrap();
}

#[test]
#[ignore = "requires CUDA, INFINITY_OXIDE_MPNET_ATTENTION_ROOT, INFINITY_MPNET_MODEL, and INFINITY_OXIDE_MPNET_REFERENCE"]
fn gpu_mpnet_native_bucket_attention() {
    let reference = load_reference().unwrap();
    let records = validate_reference(&reference).unwrap();
    let cases = require_array(
        field(records, "attention_cases").unwrap(),
        "attention_cases",
    )
    .unwrap();
    assert_eq!(cases.len(), ATTENTION_CASES, "attention case count");
    let mut saw_512 = false;
    let mut saw_unequal = false;
    for case in cases {
        let case = require_object(case, "attention case").unwrap();
        let lengths = require_array(field(case, "lengths").unwrap(), "lengths").unwrap();
        let lengths = lengths
            .iter()
            .map(|length| require_usize(length, "length").unwrap())
            .collect::<Vec<_>>();
        assert!(!lengths.is_empty());
        assert!(lengths.iter().all(|length| *length > 0 && *length <= 512));
        saw_512 |= lengths.contains(&512);
        saw_unequal |= lengths.windows(2).any(|window| window[0] != window[1]);
        let total = require_usize(field(case, "total_tokens").unwrap(), "total_tokens").unwrap();
        assert_eq!(total, lengths.iter().sum::<usize>());
        bf16_record(
            field(case, "output").unwrap(),
            &[total, 12, 64],
            "attention output",
        )
        .unwrap();
    }
    assert!(saw_512, "attention suite must include a length-512 case");
    assert!(
        saw_unequal,
        "attention suite must include unequal packed sequences"
    );

    let model =
        std::path::PathBuf::from(std::env::var_os(MODEL_ENV).expect("Set INFINITY_MPNET_MODEL"));
    let fa4_root = std::path::PathBuf::from(
        std::env::var_os(NATIVE_ROOT_ENV).expect("Set INFINITY_OXIDE_MPNET_ATTENTION_ROOT"),
    );
    let context = Context::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let host = BertWeights::load(Path::new(&model)).unwrap();
    let source = host
        .relative_bias
        .as_ref()
        .expect("MPNet checkpoint must provide encoder.relative_attention_bias.weight");
    assert_eq!(source.shape(), &[32, 12]);
    let mut uploader = WeightUploader::new(stream.clone()).unwrap();
    let table = uploader.upload(&[source]).unwrap();
    uploader.finish().unwrap();
    let provider = mpnet_attention::NativeMpnetAttention::new(
        context.clone(),
        &fa4_root,
        crate::fa4::DType::Bf16,
    )
    .unwrap();

    for case in cases {
        let case = require_object(case, "attention case").unwrap();
        let name = require_str(field(case, "name").unwrap(), "name").unwrap();
        let lengths = require_array(field(case, "lengths").unwrap(), "lengths").unwrap();
        let lengths = lengths
            .iter()
            .map(|length| require_usize(length, "length").unwrap())
            .collect::<Vec<_>>();
        let total = require_usize(field(case, "total_tokens").unwrap(), "total_tokens").unwrap();
        let expected = bf16_record(
            field(case, "output").unwrap(),
            &[total, 12, 64],
            "attention output",
        )
        .unwrap();
        let q_words = deterministic_bf16(total * 12 * 64, 43, 21, 0.0625);
        let k_words = deterministic_bf16(total * 12 * 64, 47, 23, 0.0625);
        let v_words = deterministic_bf16(total * 12 * 64, 53, 26, 0.0625);
        let packed = packed_qkv(&q_words, &k_words, &v_words).unwrap();
        let qkv = guarded_u16(&stream, &packed, GUARD).unwrap();
        let output = guarded_u16(&stream, &vec![SENTINEL; total * 12 * 64], GUARD).unwrap();
        let offsets = guarded_u32(&stream, &cumulative(&lengths).unwrap(), GUARD).unwrap();
        let q_ptr = qkv.ptr();
        let k_ptr = q_ptr + 1536;
        let v_ptr = q_ptr + 3072;
        let out_ptr = output.ptr();
        let offsets_ptr = offsets.ptr();
        let table_ptr = table.cu_deviceptr();
        let tokens = total;
        let batch = lengths.len();
        let max_q = *lengths.iter().max().unwrap();
        let strides = [2304i64, 64, 1, 2304, 64, 1, 2304, 64, 1, 768, 64, 1];
        let attention = mpnet_attention::RelativeAttentionBatch {
            q: q_ptr,
            k: k_ptr,
            v: v_ptr,
            out: out_ptr,
            offsets: offsets_ptr,
            table: table_ptr,
            tokens,
            batch,
            max_q,
            strides,
        };
        let result = unsafe { provider.forward_batch_bound(&stream, &attention) }
            .map_err(|error| format!("{name}: {error}"));
        let sync = stream
            .synchronize()
            .map_err(|error| format!("{name}: stream synchronize failed: {error}"));
        result.unwrap();
        sync.unwrap();
        let actual = output
            .read(&stream, total * 12 * 64, "attention output")
            .unwrap();
        for row in 0..total {
            assert_attention_row(
                &actual[row * 768..(row + 1) * 768],
                &expected[row * 768..(row + 1) * 768],
                &format!("{name} row {row}"),
            );
        }
        qkv.assert_unchanged(&stream, "attention packed QKV source")
            .unwrap();
        offsets
            .assert_unchanged(&stream, "attention offsets")
            .unwrap();

        if lengths.len() > 1 {
            let first_end = lengths[0];
            let mut perturbed_words = q_words.clone();
            let tail = deterministic_bf16((total - first_end) * 12 * 64, 59, 29, 0.0625);
            perturbed_words[first_end * 12 * 64..].copy_from_slice(&tail);
            let perturbed_packed = packed_qkv(&perturbed_words, &k_words, &v_words).unwrap();
            let perturbed_q = guarded_u16(&stream, &perturbed_packed, GUARD).unwrap();
            let perturbed_output =
                guarded_u16(&stream, &vec![SENTINEL; total * 12 * 64], GUARD).unwrap();
            let q_ptr = perturbed_q.ptr();
            let k_ptr = q_ptr + 1536;
            let v_ptr = q_ptr + 3072;
            let out_ptr = perturbed_output.ptr();
            let offsets_ptr = offsets.ptr();
            let table_ptr = table.cu_deviceptr();
            let tokens = total;
            let batch = lengths.len();
            let max_q = *lengths.iter().max().unwrap();
            let strides = [2304i64, 64, 1, 2304, 64, 1, 2304, 64, 1, 768, 64, 1];
            let witness = mpnet_attention::RelativeAttentionBatch {
                q: q_ptr,
                k: k_ptr,
                v: v_ptr,
                out: out_ptr,
                offsets: offsets_ptr,
                table: table_ptr,
                tokens,
                batch,
                max_q,
                strides,
            };
            let result = unsafe { provider.forward_batch_bound(&stream, &witness) }
                .map_err(|error| format!("{name}: perturbed witness failed: {error}"));
            let sync = stream
                .synchronize()
                .map_err(|error| format!("{name}: witness synchronize failed: {error}"));
            result.unwrap();
            sync.unwrap();
            let changed = perturbed_output
                .read(&stream, total * 12 * 64, "perturbed attention output")
                .unwrap();
            assert_eq!(
                changed[..first_end * 768],
                actual[..first_end * 768],
                "{name}: perturbing later Q must not change the first sequence"
            );
            assert_ne!(
                changed[first_end * 768..],
                actual[first_end * 768..],
                "{name}: perturbing later Q must change the later sequence"
            );
            perturbed_q
                .assert_unchanged(&stream, "perturbed Q source")
                .unwrap();
            qkv.assert_unchanged(&stream, "original packed QKV witness source")
                .unwrap();
            offsets
                .assert_unchanged(&stream, "witness offsets")
                .unwrap();
        }
    }
    stream.synchronize().unwrap();
}

fn packed_qkv(q: &[u16], k: &[u16], v: &[u16]) -> Result<Vec<u16>, String> {
    if q.len() != k.len() || q.len() != v.len() || !q.len().is_multiple_of(768) {
        return Err("Invalid packed QKV rows".into());
    }
    let size = q
        .len()
        .checked_mul(3)
        .ok_or_else(|| "Packed QKV overflow".to_string())?;
    let mut out = Vec::new();
    out.try_reserve_exact(size).map_err(|e| e.to_string())?;
    for ((q, k), v) in q
        .chunks_exact(768)
        .zip(k.chunks_exact(768))
        .zip(v.chunks_exact(768))
    {
        out.extend_from_slice(q);
        out.extend_from_slice(k);
        out.extend_from_slice(v);
    }
    Ok(out)
}
#[test]
fn packed_native_qkv_preserves_rows_and_alias_free_offsets() {
    let q = (0..1536).map(|x| x as u16).collect::<Vec<_>>();
    let k = vec![5000u16; 1536];
    let v = vec![9000u16; 1536];
    let packed = packed_qkv(&q, &k, &v).unwrap();
    assert_eq!(&packed[..768], &q[..768]);
    assert_eq!(&packed[768..1536], &k[..768]);
    assert_eq!(&packed[1536..2304], &v[..768]);
    assert_eq!(&packed[2304..3072], &q[768..]);
    assert_eq!(&packed[3072..3840], &k[768..]);
    assert_eq!(&packed[3840..], &v[768..]);
}
#[test]
fn packed_native_qkv_rejects_ragged_and_misaligned_rows() {
    assert!(packed_qkv(&[1; 768], &[1; 767], &[1; 768]).is_err());
    assert!(packed_qkv(&[1; 769], &[1; 769], &[1; 769]).is_err());
    assert!(packed_qkv(&[], &[], &[]).unwrap().is_empty());
}
