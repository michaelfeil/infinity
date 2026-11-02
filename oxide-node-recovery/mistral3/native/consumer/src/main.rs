// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//
// SOURCE-ONLY Rust consumer for the frozen b66 libtei_torch.so.
//
// NON-CLAIM: this crate was not compiled or run during the CPU-only control.
// Root builds it later with the offline Cargo cache. It uses std and libloading
// only; there are no network dependencies. Production paths return typed
// errors instead of unwrap/expect/panic. The process refuses to start unless
// CUDA_VISIBLE_DEVICES is explicitly set by Root.

use libloading::Library;

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{c_char, c_void, CStr, CString};
use std::fmt;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::time::Instant;

const DEFAULT_LIBRARY: &str = concat!(
    "/tmp/infinity-fast-weight-reader/libs/infinity-v3/target/debug/build/",
    "infinity-backend-libtorch-0ee14ee2449a0322/out/build/libtei_torch.so"
);
const DEFAULT_CHECKPOINT: &str = concat!(
    "/workspace/model-performance/michaelfeil/benchmark-overview/models/",
    "nvidia--Nemotron-3.5-Rerank-8B-BF16"
);

// ---------------------------------------------------------------------------
// Typed errors
// ---------------------------------------------------------------------------
#[derive(Debug)]
enum Error {
    Env(String),
    Io(String),
    Json(String),
    Config(String),
    Abi(String),
    Native {
        code: i32,
        kind: i32,
        message: String,
    },
    Layout(String),
    Bounds(String),
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Env(message) => write!(formatter, "environment error: {message}"),
            Error::Io(message) => write!(formatter, "I/O error: {message}"),
            Error::Json(message) => write!(formatter, "JSON error: {message}"),
            Error::Config(message) => write!(formatter, "config error: {message}"),
            Error::Abi(message) => write!(formatter, "ABI error: {message}"),
            Error::Native {
                code,
                kind,
                message,
            } => write!(
                formatter,
                "native error code={code} kind={kind}: {message}"
            ),
            Error::Layout(message) => write!(formatter, "layout error: {message}"),
            Error::Bounds(message) => write!(formatter, "bounds error: {message}"),
        }
    }
}

impl std::error::Error for Error {}

type Result<T> = std::result::Result<T, Error>;

fn io_error(context: &str, error: std::io::Error) -> Error {
    Error::Io(format!("{context}: {error}"))
}

// ---------------------------------------------------------------------------
// Minimal JSON reader (config, index, safetensors headers only)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
enum Json {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Json>),
    Object(BTreeMap<String, Json>),
}

impl Json {
    fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(map) => map.get(key),
            _ => None,
        }
    }

    fn as_object(&self) -> Option<&BTreeMap<String, Json>> {
        match self {
            Json::Object(map) => Some(map),
            _ => None,
        }
    }

    fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Array(values) => Some(values),
            _ => None,
        }
    }

    fn as_str(&self) -> Option<&str> {
        match self {
            Json::String(value) => Some(value),
            _ => None,
        }
    }

    fn as_i64(&self) -> Option<i64> {
        match self {
            Json::Number(value) if value.is_finite() && *value >= i64::MIN as f64
                && *value <= i64::MAX as f64 =>
            {
                Some(*value as i64)
            }
            _ => None,
        }
    }

    fn as_u64(&self) -> Option<u64> {
        match self {
            Json::Number(value) if value.is_finite() && *value >= 0.0
                && *value <= u64::MAX as f64 =>
            {
                Some(*value as u64)
            }
            _ => None,
        }
    }

    fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Number(value) => Some(*value),
            _ => None,
        }
    }
}

struct JsonParser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            bytes: text.as_bytes(),
            pos: 0,
        }
    }

    fn parse(mut self) -> Result<Json> {
        self.skip_space();
        let value = self.value()?;
        self.skip_space();
        if self.pos != self.bytes.len() {
            return Err(Error::Json("trailing data after JSON value".into()));
        }
        Ok(value)
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<u8> {
        let value = self.peek()?;
        self.pos += 1;
        Some(value)
    }

    fn skip_space(&mut self) {
        while let Some(byte) = self.peek() {
            if byte == b' ' || byte == b'\t' || byte == b'\n' || byte == b'\r' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn literal(&mut self, expected: &[u8]) -> Result<()> {
        let end = self
            .pos
            .checked_add(expected.len())
            .ok_or_else(|| Error::Json("literal offset overflow".into()))?;
        let actual = self
            .bytes
            .get(self.pos..end)
            .ok_or_else(|| Error::Json("unexpected end of JSON literal".into()))?;
        if actual != expected {
            return Err(Error::Json("invalid JSON literal".into()));
        }
        self.pos = end;
        Ok(())
    }

    fn value(&mut self) -> Result<Json> {
        self.skip_space();
        match self.peek() {
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Json::String(self.string()?)),
            Some(b't') => {
                self.literal(b"true")?;
                Ok(Json::Bool(true))
            }
            Some(b'f') => {
                self.literal(b"false")?;
                Ok(Json::Bool(false))
            }
            Some(b'n') => {
                self.literal(b"null")?;
                Ok(Json::Null)
            }
            Some(_) => self.number(),
            None => Err(Error::Json("unexpected end of JSON".into())),
        }
    }

    fn object(&mut self) -> Result<Json> {
        if self.advance() != Some(b'{') {
            return Err(Error::Json("expected object".into()));
        }
        let mut map = BTreeMap::new();
        self.skip_space();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Json::Object(map));
        }
        loop {
            self.skip_space();
            if self.peek() != Some(b'"') {
                return Err(Error::Json("expected object key".into()));
            }
            let key = self.string()?;
            self.skip_space();
            if self.advance() != Some(b':') {
                return Err(Error::Json("expected ':'".into()));
            }
            let value = self.value()?;
            map.insert(key, value);
            self.skip_space();
            match self.advance() {
                Some(b'}') => return Ok(Json::Object(map)),
                Some(b',') => continue,
                _ => return Err(Error::Json("expected ',' or '}'".into())),
            }
        }
    }

    fn array(&mut self) -> Result<Json> {
        if self.advance() != Some(b'[') {
            return Err(Error::Json("expected array".into()));
        }
        let mut values = Vec::new();
        self.skip_space();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Json::Array(values));
        }
        loop {
            values.push(self.value()?);
            self.skip_space();
            match self.advance() {
                Some(b']') => return Ok(Json::Array(values)),
                Some(b',') => continue,
                _ => return Err(Error::Json("expected ',' or ']'".into())),
            }
        }
    }

    fn hex4(&mut self) -> Result<u32> {
        let mut code = 0u32;
        for _ in 0..4 {
            let byte = self
                .advance()
                .ok_or_else(|| Error::Json("short \\u escape".into()))?;
            let digit = match byte {
                b'0'..=b'9' => u32::from(byte - b'0'),
                b'a'..=b'f' => u32::from(byte - b'a' + 10),
                b'A'..=b'F' => u32::from(byte - b'A' + 10),
                _ => return Err(Error::Json("invalid \\u escape".into())),
            };
            code = code
                .checked_mul(16)
                .and_then(|value| value.checked_add(digit))
                .ok_or_else(|| Error::Bounds("unicode escape overflow".into()))?;
        }
        Ok(code)
    }

    fn string(&mut self) -> Result<String> {
        if self.advance() != Some(b'"') {
            return Err(Error::Json("expected string".into()));
        }
        let mut out = String::new();
        loop {
            let byte = self
                .advance()
                .ok_or_else(|| Error::Json("unterminated string".into()))?;
            match byte {
                b'"' => return Ok(out),
                b'\\' => {
                    let escape = self
                        .advance()
                        .ok_or_else(|| Error::Json("unterminated escape".into()))?;
                    match escape {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{0008}'),
                        b'f' => out.push('\u{000c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let code = self.hex4()?;
                            let character = char::from_u32(code).ok_or_else(|| {
                                Error::Json("invalid unicode scalar".into())
                            })?;
                            out.push(character);
                        }
                        _ => return Err(Error::Json("invalid escape".into())),
                    }
                }
                0x00..=0x1f => return Err(Error::Json("control character".into())),
                value => out.push(char::from(value)),
            }
        }
    }

    fn number(&mut self) -> Result<Json> {
        let begin = self.pos;
        if matches!(self.peek(), Some(b'-') | Some(b'+')) {
            self.pos += 1;
        }
        let mut digits = false;
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            digits = true;
            self.pos += 1;
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                digits = true;
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'-') | Some(b'+')) {
                self.pos += 1;
            }
            let mut exponent_digits = false;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                exponent_digits = true;
                self.pos += 1;
            }
            if !exponent_digits {
                return Err(Error::Json("invalid exponent".into()));
            }
        }
        if !digits {
            return Err(Error::Json("invalid number".into()));
        }
        let token = self
            .bytes
            .get(begin..self.pos)
            .ok_or_else(|| Error::Json("number slice out of range".into()))?;
        let text = std::str::from_utf8(token)
            .map_err(|error| Error::Json(format!("invalid number bytes: {error}")))?;
        let value = text
            .parse::<f64>()
            .map_err(|error| Error::Json(format!("invalid number: {error}")))?;
        if !value.is_finite() {
            return Err(Error::Json("non-finite number".into()));
        }
        Ok(Json::Number(value))
    }
}

fn parse_json(text: &str) -> Result<Json> {
    JsonParser::new(text).parse()
}

fn read_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path).map_err(|error| io_error(&path.display().to_string(), error))?;
    String::from_utf8(bytes)
        .map_err(|error| Error::Io(format!("{} is not UTF-8: {error}", path.display())))
}

fn flatten_json(value: &Json, prefix: &str, out: &mut BTreeMap<String, String>) {
    match value {
        Json::Object(map) => {
            for (key, child) in map {
                let path = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                flatten_json(child, &path, out);
            }
        }
        Json::Array(values) => {
            for (index, child) in values.iter().enumerate() {
                let path = format!("{prefix}.{index}");
                flatten_json(child, &path, out);
            }
        }
        Json::String(text) => {
            if !prefix.is_empty() {
                out.insert(prefix.to_string(), text.clone());
            }
        }
        Json::Number(number) => {
            if !prefix.is_empty() {
                out.insert(prefix.to_string(), format!("{number}"));
            }
        }
        Json::Bool(flag) => {
            if !prefix.is_empty() {
                out.insert(prefix.to_string(), flag.to_string());
            }
        }
        Json::Null => {}
    }
}

fn json_path<'a>(config: &'a Json, path: &[&str]) -> Result<&'a Json> {
    let mut current = config;
    for segment in path {
        let next = current.get(segment).or_else(|| {
            if let Json::Array(values) = current {
                segment
                    .parse::<usize>()
                    .ok()
                    .and_then(|index| values.get(index))
            } else {
                None
            }
        });
        current = next.ok_or_else(|| Error::Config(format!("missing config key {segment}")))?;
    }
    Ok(current)
}

fn config_i64(config: &Json, path: &[&str]) -> Result<i64> {
    json_path(config, path)?
        .as_i64()
        .ok_or_else(|| Error::Config(format!("config {} is not an integer", path.join("."))))
}

fn config_str<'a>(config: &'a Json, path: &[&str]) -> Result<&'a str> {
    json_path(config, path)?
        .as_str()
        .ok_or_else(|| Error::Config(format!("config {} is not a string", path.join("."))))
}

fn json_escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 8);
    for character in value.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            value if (value as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", value as u32));
            }
            value => out.push(value),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Frozen b66 C ABI
// ---------------------------------------------------------------------------
#[repr(C)]
struct NativeConfig {
    hidden: i64,
    heads: i64,
    layers: i64,
    intermediate: i64,
    vocab: i64,
    positions: i64,
    types: i64,
    epsilon: f64,
    activation: i32,
}

#[repr(C)]
struct NativeImage {
    pixels: *const f32,
    rows: i64,
    patch_dim: i64,
    grid: [i64; 3],
    merge_size: i64,
    token_start: i64,
    token_count: i64,
    sequence_start: i64,
    feature_key: [u8; 32],
    has_feature_key: u8,
}

#[repr(C)]
struct NativeAudio {
    values: *const f32,
    mask: *const u8,
    frames: i64,
    feature_size: i64,
    token_start: i64,
    token_count: i64,
}

#[repr(C)]
struct NativeWeight {
    name: *const c_char,
    shape: *const i64,
    rank: usize,
    dtype: i32,
    offset: u64,
    length: u64,
}

type TeiError = unsafe extern "C" fn() -> *const c_char;
type TeiLastErrorKind = unsafe extern "C" fn() -> i32;
type TeiCreate = unsafe extern "C" fn(*const NativeConfig, *const c_char, i32) -> *mut c_void;
type TeiOption = unsafe extern "C" fn(*mut c_void, *const c_char, *const c_char) -> i32;
type TeiLoadSafetensors = unsafe extern "C" fn(
    *mut c_void,
    i32,
    u64,
    u64,
    *const NativeWeight,
    usize,
    usize,
    usize,
    i32,
) -> i32;
type TeiReady = unsafe extern "C" fn(*mut c_void) -> i32;
#[allow(clippy::type_complexity)]
type TeiForward = unsafe extern "C" fn(
    *mut c_void,
    *const i64,
    *const i64,
    *const i64,
    *const i32,
    i64,
    i64,
    i32,
    *const i64,
    usize,
    *const i64,
    usize,
    *mut f32,
    usize,
    *const i64,
    *const NativeImage,
    usize,
    *const NativeAudio,
    usize,
) -> i32;
#[allow(clippy::type_complexity)]
type TeiForwardSubmit = unsafe extern "C" fn(
    *mut c_void,
    *const i64,
    *const i64,
    *const i64,
    *const i32,
    i64,
    i64,
    i32,
    *const i64,
    usize,
    *const i64,
    usize,
    usize,
    *const i64,
    *const NativeImage,
    usize,
    *const NativeAudio,
    usize,
    *const c_void,
) -> *mut c_void;
type TeiOutputWait = unsafe extern "C" fn(*mut c_void) -> *const f32;
type TeiOutputDestroy = unsafe extern "C" fn(*mut c_void);
type TeiDestroy = unsafe extern "C" fn(*mut c_void);

struct Functions {
    error: TeiError,
    last_error_kind: TeiLastErrorKind,
    create: TeiCreate,
    option: TeiOption,
    load_safetensors: TeiLoadSafetensors,
    ready: TeiReady,
    forward: TeiForward,
    forward_submit: TeiForwardSubmit,
    output_wait: TeiOutputWait,
    output_destroy: TeiOutputDestroy,
    destroy: TeiDestroy,
}

fn symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T> {
    let loaded = unsafe { library.get::<T>(name) }
        .map_err(|error| Error::Abi(format!("missing {}: {error}", String::from_utf8_lossy(name))))?;
    Ok(*loaded)
}

fn load_functions(library: &Library) -> Result<Functions> {
    Ok(Functions {
        error: symbol::<TeiError>(library, b"tei_error\0")?,
        last_error_kind: symbol::<TeiLastErrorKind>(library, b"tei_last_error_kind\0")?,
        create: symbol::<TeiCreate>(library, b"tei_create\0")?,
        option: symbol::<TeiOption>(library, b"tei_option\0")?,
        load_safetensors: symbol::<TeiLoadSafetensors>(library, b"tei_load_safetensors\0")?,
        ready: symbol::<TeiReady>(library, b"tei_ready\0")?,
        forward: symbol::<TeiForward>(library, b"tei_forward\0")?,
        forward_submit: symbol::<TeiForwardSubmit>(library, b"tei_forward_submit\0")?,
        output_wait: symbol::<TeiOutputWait>(library, b"tei_output_wait\0")?,
        output_destroy: symbol::<TeiOutputDestroy>(library, b"tei_output_destroy\0")?,
        destroy: symbol::<TeiDestroy>(library, b"tei_destroy\0")?,
    })
}

fn native_text(functions: &Functions) -> String {
    let pointer = unsafe { (functions.error)() };
    if pointer.is_null() {
        "native call failed without an error string".into()
    } else {
        unsafe { CStr::from_ptr(pointer) }
            .to_string_lossy()
            .into_owned()
    }
}

fn native_status(functions: &Functions, status: i32) -> Error {
    Error::Native {
        code: status,
        kind: unsafe { (functions.last_error_kind)() },
        message: native_text(functions),
    }
}

// ---------------------------------------------------------------------------
// Engine lifecycle
// ---------------------------------------------------------------------------
struct Engine {
    _library: Library,
    handle: *mut c_void,
    functions: Functions,
    checkpoint: PathBuf,
    text_hidden: i64,
    patch_size: i64,
    merge_size: i64,
    image_token_index: i64,
    tensor_count: usize,
    shard_count: usize,
}

impl Engine {
    fn open(library_path: &Path, checkpoint: &Path, device: &str) -> Result<Self> {
        let config_text = read_file(&checkpoint.join("config.json"))?;
        let config = parse_json(&config_text)?;
        if config_str(&config, &["model_type"])? != "mistral3" {
            return Err(Error::Config("expected model_type mistral3".into()));
        }
        let architecture = config_str(&config, &["architectures", "0"])?;
        if architecture != "Mistral3ForSequenceClassification" {
            return Err(Error::Config(format!(
                "expected Mistral3ForSequenceClassification, got {architecture}"
            )));
        }
        let native = NativeConfig {
            hidden: config_i64(&config, &["text_config", "hidden_size"])?,
            heads: config_i64(&config, &["text_config", "num_attention_heads"])?,
            layers: config_i64(&config, &["text_config", "num_hidden_layers"])?,
            intermediate: config_i64(&config, &["text_config", "intermediate_size"])?,
            vocab: config_i64(&config, &["text_config", "vocab_size"])?,
            positions: config_i64(&config, &["text_config", "max_position_embeddings"])?,
            types: 0,
            epsilon: json_path(&config, &["text_config", "rms_norm_eps"])?
                .as_f64()
                .ok_or_else(|| Error::Config("rms_norm_eps is not numeric".into()))?,
            activation: 0,
        };
        let text_hidden = native.hidden;
        let patch_size = config_i64(&config, &["vision_config", "patch_size"])?;
        let merge_size = config_i64(&config, &["spatial_merge_size"])?;
        let image_token_index = config_i64(&config, &["image_token_index"])?;
        let library = unsafe { Library::new(library_path) }.map_err(|error| {
            Error::Io(format!(
                "could not load {}: {error}",
                library_path.display()
            ))
        })?;
        let functions = load_functions(&library)?;
        let device_c = CString::new(device)
            .map_err(|error| Error::Config(format!("device contains NUL: {error}")))?;
        let handle = unsafe { (functions.create)(&native, device_c.as_ptr(), 2) };
        if handle.is_null() {
            return Err(native_status(&functions, -1));
        }
        let engine = Self {
            _library: library,
            handle,
            functions,
            checkpoint: checkpoint.to_path_buf(),
            text_hidden,
            patch_size,
            merge_size,
            image_token_index,
            tensor_count: 0,
            shard_count: 0,
        };
        engine.apply_options(&config)?;
        Ok(engine)
    }

    fn apply_options(&self, config: &Json) -> Result<()> {
        let mut options = BTreeMap::new();
        flatten_json(config, "", &mut options);
        for (key, value) in options {
            let key_c = CString::new(key.clone())
                .map_err(|error| Error::Config(format!("option key contains NUL: {error}")))?;
            let value_c = CString::new(value)
                .map_err(|error| Error::Config(format!("option value contains NUL: {error}")))?;
            let status =
                unsafe { (self.functions.option)(self.handle, key_c.as_ptr(), value_c.as_ptr()) };
            if status != 0 {
                return Err(native_status(&self.functions, status));
            }
        }
        Ok(())
    }

    fn check(&self, status: i32) -> Result<()> {
        if status == 0 {
            Ok(())
        } else {
            Err(native_status(&self.functions, status))
        }
    }

    fn load_checkpoint(&mut self) -> Result<()> {
        let index_text = read_file(&self.checkpoint.join("model.safetensors.index.json"))?;
        let index = parse_json(&index_text)?;
        let weight_map = index
            .get("weight_map")
            .and_then(Json::as_object)
            .ok_or_else(|| Error::Config("index lacks weight_map".into()))?;
        let indexed_total = index
            .get("metadata")
            .and_then(|metadata| metadata.get("total_size"))
            .and_then(Json::as_u64)
            .ok_or_else(|| Error::Config("index lacks metadata.total_size".into()))?;
        let mut shards = BTreeSet::new();
        for (name, owner) in weight_map {
            if name.is_empty() {
                return Err(Error::Config("empty tensor name in index".into()));
            }
            let owner = owner
                .as_str()
                .ok_or_else(|| Error::Config("index shard owner is not a string".into()))?;
            let path = Path::new(owner);
            if path.file_name().and_then(|value| value.to_str()) != Some(owner)
                || !owner.ends_with(".safetensors")
            {
                return Err(Error::Config(format!("unsafe shard name {owner}")));
            }
            shards.insert(owner.to_string());
        }
        let mut loaded_tensors = 0usize;
        let mut loaded_bytes = 0u64;
        for shard in shards {
            let shard_path = self.checkpoint.join(&shard);
            let mut file = File::open(&shard_path)
                .map_err(|error| io_error(&shard_path.display().to_string(), error))?;
            let file_size = file
                .metadata()
                .map_err(|error| io_error(&shard_path.display().to_string(), error))?
                .len();
            let mut length_bytes = [0u8; 8];
            file.read_exact(&mut length_bytes)
                .map_err(|error| io_error("safetensors length prefix", error))?;
            let header_bytes = u64::from_le_bytes(length_bytes);
            if header_bytes == 0 || header_bytes > 100_000_000 {
                return Err(Error::Layout(format!(
                    "invalid safetensors header size in {shard}"
                )));
            }
            let header_length = usize::try_from(header_bytes)
                .map_err(|_| Error::Bounds("header length does not fit usize".into()))?;
            let mut header = vec![0u8; header_length];
            file.read_exact(&mut header)
                .map_err(|error| io_error("safetensors header", error))?;
            let header_text = std::str::from_utf8(&header)
                .map_err(|error| Error::Json(format!("header is not UTF-8: {error}")))?;
            let header_json = parse_json(header_text)?;
            let header_object = header_json
                .as_object()
                .ok_or_else(|| Error::Json("safetensors header is not an object".into()))?;
            let data_start = header_bytes
                .checked_add(8)
                .ok_or_else(|| Error::Bounds("safetensors data start overflow".into()))?;
            if data_start > file_size {
                return Err(Error::Layout(format!("header exceeds file size in {shard}")));
            }
            let data_length = file_size - data_start;
            let expected_names: BTreeSet<&str> = weight_map
                .iter()
                .filter_map(|(name, owner)| {
                    owner
                        .as_str()
                        .filter(|value| *value == shard.as_str())
                        .map(|_| name.as_str())
                })
                .collect();
            let mut owned = Vec::new();
            for (name, info) in header_object {
                if name == "__metadata__" {
                    continue;
                }
                let dtype_name = info
                    .get("dtype")
                    .and_then(Json::as_str)
                    .ok_or_else(|| Error::Layout(format!("missing dtype for {name}")))?;
                let dtype = match dtype_name {
                    "F32" => 0,
                    "F16" => 1,
                    "BF16" => 2,
                    other => {
                        return Err(Error::Layout(format!(
                            "unsupported dtype {other} for {name}"
                        )))
                    }
                };
                let element_bytes: u64 = match dtype {
                    0 => 4,
                    1 | 2 => 2,
                    _ => return Err(Error::Layout(format!("invalid dtype for {name}"))),
                };
                let shape_values = info
                    .get("shape")
                    .and_then(Json::as_array)
                    .ok_or_else(|| Error::Layout(format!("missing shape for {name}")))?;
                let mut shape = Vec::with_capacity(shape_values.len());
                let mut expected_bytes = element_bytes;
                for dimension in shape_values {
                    let value = dimension
                        .as_i64()
                        .ok_or_else(|| Error::Layout(format!("invalid shape for {name}")))?;
                    if value < 0 {
                        return Err(Error::Layout(format!("negative shape for {name}")));
                    }
                    let value = value as u64;
                    shape.push(value as i64);
                    expected_bytes = expected_bytes
                        .checked_mul(value)
                        .ok_or_else(|| Error::Bounds(format!("shape overflow for {name}")))?;
                }
                let offsets = info
                    .get("data_offsets")
                    .and_then(Json::as_array)
                    .ok_or_else(|| Error::Layout(format!("missing offsets for {name}")))?;
                if offsets.len() != 2 {
                    return Err(Error::Layout(format!("invalid offsets for {name}")));
                }
                let begin = offsets
                    .first()
                    .and_then(Json::as_u64)
                    .ok_or_else(|| Error::Layout(format!("invalid offset for {name}")))?;
                let end = offsets
                    .get(1)
                    .and_then(Json::as_u64)
                    .ok_or_else(|| Error::Layout(format!("invalid offset for {name}")))?;
                if end < begin || end - begin != expected_bytes {
                    return Err(Error::Layout(format!("offset/length mismatch for {name}")));
                }
                owned.push(OwnedWeight {
                    name: name.clone(),
                    shape,
                    dtype,
                    offset: begin,
                    length: end - begin,
                });
            }
            owned.sort_by_key(|weight| weight.offset);
            let mut previous_end = 0u64;
            for weight in &owned {
                if weight.offset < previous_end
                    || weight.offset > data_length
                    || weight.length > data_length - weight.offset
                {
                    return Err(Error::Layout(format!(
                        "invalid range for {}",
                        weight.name
                    )));
                }
                previous_end = weight
                    .offset
                    .checked_add(weight.length)
                    .ok_or_else(|| Error::Bounds("weight end overflow".into()))?;
            }
            if previous_end != data_length {
                return Err(Error::Layout(format!(
                    "payload does not fill the declared file: {shard}"
                )));
            }
            let header_names: BTreeSet<&str> =
                owned.iter().map(|weight| weight.name.as_str()).collect();
            if header_names != expected_names {
                return Err(Error::Layout(format!(
                    "index/header tensor set mismatch in {shard}"
                )));
            }
            let names: Vec<CString> = owned
                .iter()
                .map(|weight| {
                    CString::new(weight.name.clone()).map_err(|error| {
                        Error::Layout(format!("tensor name contains NUL: {error}"))
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let shapes: Vec<Vec<i64>> = owned.iter().map(|weight| weight.shape.clone()).collect();
            let mut weights = Vec::with_capacity(owned.len());
            for (index, weight) in owned.iter().enumerate() {
                let name = names
                    .get(index)
                    .ok_or_else(|| Error::Bounds("name index out of range".into()))?;
                let shape = shapes
                    .get(index)
                    .ok_or_else(|| Error::Bounds("shape index out of range".into()))?;
                weights.push(NativeWeight {
                    name: name.as_ptr(),
                    shape: if shape.is_empty() {
                        std::ptr::null()
                    } else {
                        shape.as_ptr()
                    },
                    rank: shape.len(),
                    dtype: weight.dtype,
                    offset: weight.offset,
                    length: weight.length,
                });
            }
            file.seek(SeekFrom::Start(0))
                .map_err(|error| io_error("seek shard", error))?;
            let status = unsafe {
                (self.functions.load_safetensors)(
                    self.handle,
                    file.as_raw_fd(),
                    data_start,
                    data_length,
                    weights.as_ptr(),
                    weights.len(),
                    8,
                    256 << 20,
                    0,
                )
            };
            self.check(status)?;
            loaded_tensors = loaded_tensors
                .checked_add(owned.len())
                .ok_or_else(|| Error::Bounds("tensor count overflow".into()))?;
            loaded_bytes = loaded_bytes
                .checked_add(data_length)
                .ok_or_else(|| Error::Bounds("payload byte count overflow".into()))?;
            self.shard_count += 1;
        }
        if loaded_tensors != weight_map.len() {
            return Err(Error::Layout(format!(
                "loaded {loaded_tensors} tensors but the index lists {}",
                weight_map.len()
            )));
        }
        if loaded_bytes != indexed_total {
            return Err(Error::Layout(format!(
                "loaded {loaded_bytes} bytes but metadata.total_size is {indexed_total}"
            )));
        }
        self.tensor_count = loaded_tensors;
        Ok(())
    }

    fn ready(&self) -> Result<()> {
        let status = unsafe { (self.functions.ready)(self.handle) };
        self.check(status)
    }

    fn forward_sync(&self, request: &HostRequest, images: &[NativeImage]) -> Result<Vec<f32>> {
        let mut output = vec![0.0f32; request.expected_output];
        let status = unsafe {
            (self.functions.forward)(
                self.handle,
                request.ids.as_ptr(),
                request.types.as_ptr(),
                request.positions.as_ptr(),
                request.cumulative.as_ptr(),
                request.batch,
                request.max_sequence,
                request.pool,
                if request.pooled.is_empty() {
                    std::ptr::null()
                } else {
                    request.pooled.as_ptr()
                },
                request.pooled.len(),
                if request.raw.is_empty() {
                    std::ptr::null()
                } else {
                    request.raw.as_ptr()
                },
                request.raw.len(),
                output.as_mut_ptr(),
                output.len(),
                std::ptr::null(),
                if images.is_empty() {
                    std::ptr::null()
                } else {
                    images.as_ptr()
                },
                images.len(),
                std::ptr::null(),
                0,
            )
        };
        self.check(status)?;
        Ok(output)
    }

    fn forward_deferred<'a>(
        &'a self,
        request: &HostRequest,
        images: &[NativeImage],
    ) -> Result<OutputLease<'a>> {
        let handle = unsafe {
            (self.functions.forward_submit)(
                self.handle,
                request.ids.as_ptr(),
                request.types.as_ptr(),
                request.positions.as_ptr(),
                request.cumulative.as_ptr(),
                request.batch,
                request.max_sequence,
                request.pool,
                if request.pooled.is_empty() {
                    std::ptr::null()
                } else {
                    request.pooled.as_ptr()
                },
                request.pooled.len(),
                if request.raw.is_empty() {
                    std::ptr::null()
                } else {
                    request.raw.as_ptr()
                },
                request.raw.len(),
                request.expected_output,
                std::ptr::null(),
                if images.is_empty() {
                    std::ptr::null()
                } else {
                    images.as_ptr()
                },
                images.len(),
                std::ptr::null(),
                0,
                std::ptr::null(),
            )
        };
        if handle.is_null() {
            return Err(native_status(&self.functions, -1));
        }
        Ok(OutputLease {
            engine: self,
            handle,
            elements: request.expected_output,
        })
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { (self.functions.destroy)(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

struct OutputLease<'a> {
    engine: &'a Engine,
    handle: *mut c_void,
    elements: usize,
}

impl<'a> OutputLease<'a> {
    fn wait(&self) -> Result<&[f32]> {
        let pointer = unsafe { (self.engine.functions.output_wait)(self.handle) };
        if pointer.is_null() {
            return Err(native_status(&self.engine.functions, -1));
        }
        if self.elements == 0 {
            return Err(Error::Layout("empty output lease".into()));
        }
        unsafe { Ok(std::slice::from_raw_parts(pointer, self.elements)) }
    }
}

impl<'a> Drop for OutputLease<'a> {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { (self.engine.functions.output_destroy)(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

struct OwnedWeight {
    name: String,
    shape: Vec<i64>,
    dtype: i32,
    offset: u64,
    length: u64,
}

// ---------------------------------------------------------------------------
// Reference requests
// ---------------------------------------------------------------------------
struct HostRequest {
    ids: Vec<i64>,
    types: Vec<i64>,
    positions: Vec<i64>,
    cumulative: Vec<i32>,
    pooled: Vec<i64>,
    raw: Vec<i64>,
    batch: i64,
    max_sequence: i64,
    pool: i32,
    expected_output: usize,
}

fn seeded_id(seed: u64, index: u64) -> i64 {
    const VOCAB_LIMIT: u64 = 1000;
    let mixed = seed.wrapping_add(index.wrapping_mul(37));
    (1 + mixed % VOCAB_LIMIT) as i64
}

fn output_width(pool: i32, text_hidden: i64) -> Result<usize> {
    let width = if pool == 4 { 1 } else { text_hidden };
    if width <= 0 {
        return Err(Error::Config("non-positive output width".into()));
    }
    usize::try_from(width).map_err(|_| Error::Bounds("output width does not fit usize".into()))
}

fn make_text_request(tokens: usize, pool: i32, text_hidden: i64) -> Result<HostRequest> {
    if tokens == 0 {
        return Err(Error::Layout("text case needs at least one token".into()));
    }
    let token_count = i32::try_from(tokens)
        .map_err(|_| Error::Bounds("token count does not fit int32".into()))?;
    let expected = output_width(pool, text_hidden)?
        .checked_mul(1)
        .ok_or_else(|| Error::Bounds("output size overflow".into()))?;
    let mut ids = Vec::with_capacity(tokens);
    let mut positions = Vec::with_capacity(tokens);
    for index in 0..tokens {
        let index_u64 = index as u64;
        ids.push(seeded_id(0x4d49_5354, index_u64));
        positions.push(
            i64::try_from(index)
                .map_err(|_| Error::Bounds("position does not fit int64".into()))?,
        );
    }
    Ok(HostRequest {
        ids,
        types: vec![0; tokens],
        positions,
        cumulative: vec![0, token_count],
        pooled: vec![0],
        raw: Vec::new(),
        batch: 1,
        max_sequence: i64::from(token_count),
        pool,
        expected_output: expected,
    })
}

struct HostImage {
    pixels: Vec<f32>,
    grid: [i64; 3],
    patch_dim: i64,
    rows: i64,
    merge_size: i64,
    token_start: i64,
    token_count: i64,
}

impl HostImage {
    fn native(&self) -> NativeImage {
        NativeImage {
            pixels: self.pixels.as_ptr(),
            rows: self.rows,
            patch_dim: self.patch_dim,
            grid: self.grid,
            merge_size: self.merge_size,
            token_start: self.token_start,
            token_count: self.token_count,
            sequence_start: 0,
            feature_key: [0u8; 32],
            has_feature_key: 0,
        }
    }
}

fn make_image_request(
    height: i64,
    width: i64,
    sequence_tokens: usize,
    token_start: i64,
    pool: i32,
    text_hidden: i64,
    patch_size: i64,
    merge_size: i64,
    image_token_index: i64,
) -> Result<(HostRequest, HostImage)> {
    if height <= 0 || width <= 0 || merge_size <= 0 || patch_size <= 0 {
        return Err(Error::Layout("invalid image geometry".into()));
    }
    if height % merge_size != 0 || width % merge_size != 0 {
        return Err(Error::Layout("image grid is not divisible by merge size".into()));
    }
    let rows = height
        .checked_mul(width)
        .ok_or_else(|| Error::Bounds("image patch count overflow".into()))?;
    let patch_dim = patch_size
        .checked_mul(patch_size)
        .and_then(|value| value.checked_mul(3))
        .ok_or_else(|| Error::Bounds("patch dimension overflow".into()))?;
    let merge_tokens = (height / merge_size)
        .checked_mul(width / merge_size)
        .ok_or_else(|| Error::Bounds("merge token count overflow".into()))?;
    let sequence_end = token_start
        .checked_add(merge_tokens)
        .ok_or_else(|| Error::Bounds("image token span overflow".into()))?;
    let sequence_tokens_i64 = i64::try_from(sequence_tokens)
        .map_err(|_| Error::Bounds("sequence length does not fit int64".into()))?;
    if token_start < 0 || sequence_end > sequence_tokens_i64 {
        return Err(Error::Layout("image token span crosses the sequence".into()));
    }
    let token_count_i32 = i32::try_from(sequence_tokens)
        .map_err(|_| Error::Bounds("sequence length does not fit int32".into()))?;
    let expected = output_width(pool, text_hidden)?;
    let mut ids = Vec::with_capacity(sequence_tokens);
    let mut positions = Vec::with_capacity(sequence_tokens);
    for index in 0..sequence_tokens {
        let index_u64 = index as u64;
        ids.push(seeded_id(0x4d49_5354, index_u64));
        positions.push(
            i64::try_from(index)
                .map_err(|_| Error::Bounds("position does not fit int64".into()))?,
        );
    }
    let merge_width = width / merge_size;
    for index in 0..merge_tokens {
        let row = index / merge_width;
        let column = index % merge_width;
        let position = token_start
            .checked_add(row)
            .and_then(|value| value.checked_mul(merge_width + 1))
            .and_then(|value| value.checked_add(column))
            .ok_or_else(|| Error::Bounds("image token index overflow".into()))?;
        if position < 0 || position >= sequence_tokens_i64 {
            return Err(Error::Layout("image token index outside sequence".into()));
        }
        let slot = usize::try_from(position)
            .map_err(|_| Error::Bounds("image token index does not fit usize".into()))?;
        if let Some(value) = ids.get_mut(slot) {
            *value = image_token_index;
        }
    }
    let pixel_count = usize::try_from(
        rows.checked_mul(patch_dim)
            .ok_or_else(|| Error::Bounds("pixel count overflow".into()))?,
    )
    .map_err(|_| Error::Bounds("pixel count does not fit usize".into()))?;
    let mut pixels = Vec::with_capacity(pixel_count);
    for index in 0..pixel_count {
        let sample = (index as u32).wrapping_mul(2_654_435_761) ^ 0x9e37_79b9;
        pixels.push((sample & 0xffff) as f32 / 65535.0);
    }
    let request = HostRequest {
        ids,
        types: vec![0; sequence_tokens],
        positions,
        cumulative: vec![0, token_count_i32],
        pooled: vec![0],
        raw: Vec::new(),
        batch: 1,
        max_sequence: sequence_tokens_i64,
        pool,
        expected_output: expected,
    };
    let image = HostImage {
        pixels,
        grid: [1, height, width],
        patch_dim,
        rows,
        merge_size,
        token_start,
        token_count: merge_tokens,
    };
    Ok((request, image))
}

fn checksum(values: &[f32]) -> f64 {
    values.iter().map(|value| f64::from(*value)).sum()
}

struct CaseResult {
    name: String,
    elements: usize,
    checksum: f64,
    elapsed_ns: u64,
}

struct PerfResult {
    iterations: usize,
    min_ns: u64,
    max_ns: u64,
    mean_ns: f64,
    p50_ns: u64,
    checksum: f64,
}

fn measure_perf(
    engine: &Engine,
    request: &HostRequest,
    images: &[NativeImage],
    warmup: usize,
    iterations: usize,
) -> Result<PerfResult> {
    if iterations == 0 {
        return Err(Error::Config("perf needs at least one iteration".into()));
    }
    for _ in 0..warmup {
        let _ = engine.forward_sync(request, images)?;
    }
    let mut samples = Vec::with_capacity(iterations);
    let mut last_checksum = 0.0;
    for _ in 0..iterations {
        let start = Instant::now();
        let output = engine.forward_sync(request, images)?;
        let elapsed = start.elapsed().as_nanos();
        samples.push(u64::try_from(elapsed).unwrap_or(u64::MAX));
        last_checksum = checksum(&output);
    }
    samples.sort_unstable();
    let min_ns = samples.first().copied().unwrap_or(0);
    let max_ns = samples.last().copied().unwrap_or(0);
    let total: u128 = samples.iter().map(|value| u128::from(*value)).sum();
    let mean_ns = if samples.is_empty() {
        0.0
    } else {
        total as f64 / samples.len() as f64
    };
    let middle = samples.len() / 2;
    let p50_ns = samples.get(middle).copied().unwrap_or(0);
    Ok(PerfResult {
        iterations: samples.len(),
        min_ns,
        max_ns,
        mean_ns,
        p50_ns,
        checksum: last_checksum,
    })
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
struct Options {
    library: PathBuf,
    checkpoint: PathBuf,
    device: String,
    iterations: usize,
    warmup: usize,
}

fn parse_options() -> Result<Options> {
    let mut library = PathBuf::from(DEFAULT_LIBRARY);
    let mut checkpoint = PathBuf::from(DEFAULT_CHECKPOINT);
    let mut device = "cuda:0".to_string();
    let mut iterations = 20usize;
    let mut warmup = 3usize;
    let mut arguments = std::env::args_os().skip(1);
    while let Some(argument) = arguments.next() {
        let text = argument
            .to_str()
            .ok_or_else(|| Error::Config("non-UTF8 argument".into()))?;
        match text {
            "--library" => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::Config("--library needs a path".into()))?;
                library = PathBuf::from(value);
            }
            "--checkpoint" => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::Config("--checkpoint needs a path".into()))?;
                checkpoint = PathBuf::from(value);
            }
            "--device" => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::Config("--device needs a value".into()))?;
                device = value
                    .to_str()
                    .ok_or_else(|| Error::Config("non-UTF8 device".into()))?
                    .to_string();
            }
            "--iterations" => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::Config("--iterations needs a value".into()))?;
                let text = value
                    .to_str()
                    .ok_or_else(|| Error::Config("non-UTF8 iteration count".into()))?;
                iterations = text
                    .parse::<usize>()
                    .map_err(|error| Error::Config(format!("invalid --iterations: {error}")))?;
                if iterations == 0 {
                    return Err(Error::Config("--iterations must be positive".into()));
                }
            }
            "--warmup" => {
                let value = arguments
                    .next()
                    .ok_or_else(|| Error::Config("--warmup needs a value".into()))?;
                let text = value
                    .to_str()
                    .ok_or_else(|| Error::Config("non-UTF8 warmup count".into()))?;
                warmup = text
                    .parse::<usize>()
                    .map_err(|error| Error::Config(format!("invalid --warmup: {error}")))?;
            }
            "--help" | "-h" => {
                println!(
                    "usage: mistral3-b66-consumer --library SO --checkpoint DIR \
                     [--device cuda:0] [--iterations N] [--warmup N]"
                );
                std::process::exit(0);
            }
            other => {
                return Err(Error::Config(format!("unknown argument {other}")));
            }
        }
    }
    Ok(Options {
        library,
        checkpoint,
        device,
        iterations,
        warmup,
    })
}

fn run() -> Result<()> {
    let visible = std::env::var("CUDA_VISIBLE_DEVICES")
        .map_err(|error| Error::Env(format!("CUDA_VISIBLE_DEVICES: {error}")))?;
    if visible.is_empty() {
        return Err(Error::Env(
            "refusing to run with an empty CUDA_VISIBLE_DEVICES; Root must set it explicitly"
                .into(),
        ));
    }
    let options = parse_options()?;
    let mut engine = Engine::open(&options.library, &options.checkpoint, &options.device)?;
    engine.load_checkpoint()?;
    engine.ready()?;

    let text_16 = make_text_request(16, 1, engine.text_hidden)?;
    let text_128 = make_text_request(128, 1, engine.text_hidden)?;
    let text_score = make_text_request(32, 4, engine.text_hidden)?;
    let (image_mean, host_image_mean) = make_image_request(
        4,
        6,
        12,
        2,
        1,
        engine.text_hidden,
        engine.patch_size,
        engine.merge_size,
        engine.image_token_index,
    )?;
    let (image_score, host_image_score) = make_image_request(
        8,
        8,
        24,
        2,
        4,
        engine.text_hidden,
        engine.patch_size,
        engine.merge_size,
        engine.image_token_index,
    )?;

    let mut results = Vec::new();
    for (name, request) in [
        ("text-mean-16", &text_16),
        ("text-mean-128", &text_128),
        ("text-score-32", &text_score),
    ] {
        let start = Instant::now();
        let output = engine.forward_sync(request, &[])?;
        let elapsed_ns = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        results.push(CaseResult {
            name: name.to_string(),
            elements: output.len(),
            checksum: checksum(&output),
            elapsed_ns,
        });
    }
    for (name, request, image) in [
        ("image-mean-1x4x6", &image_mean, &host_image_mean),
        ("image-score-1x8x8", &image_score, &host_image_score),
    ] {
        let native = image.native();
        let start = Instant::now();
        let output = engine.forward_sync(request, std::slice::from_ref(&native))?;
        let elapsed_ns = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        results.push(CaseResult {
            name: name.to_string(),
            elements: output.len(),
            checksum: checksum(&output),
            elapsed_ns,
        });
    }
    let deferred_native = host_image_mean.native();
    let deferred_start = Instant::now();
    let deferred_lease = engine.forward_deferred(&image_mean, std::slice::from_ref(&deferred_native))?;
    let deferred_values = deferred_lease.wait()?;
    let deferred_checksum = checksum(deferred_values);
    let deferred_elements = deferred_values.len();
    drop(deferred_lease);
    let deferred_ns = u64::try_from(deferred_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    results.push(CaseResult {
        name: "image-mean-1x4x6-deferred".to_string(),
        elements: deferred_elements,
        checksum: deferred_checksum,
        elapsed_ns: deferred_ns,
    });

    let perf_native = host_image_mean.native();
    let perf = measure_perf(
        &engine,
        &image_mean,
        std::slice::from_ref(&perf_native),
        options.warmup,
        options.iterations,
    )?;

    println!("{{");
    println!("  \"consumer\": \"mistral3-b66-consumer\",");
    println!("  \"source_only_non_claim\": \"not compiled or run during the CPU-only control\",");
    println!("  \"library\": \"{}\",", json_escape(&options.library.display().to_string()));
    println!("  \"checkpoint\": \"{}\",", json_escape(&options.checkpoint.display().to_string()));
    println!("  \"device\": \"{}\",", json_escape(&options.device));
    println!("  \"cuda_visible_devices\": \"{}\",", json_escape(&visible));
    println!("  \"tensors\": {},", engine.tensor_count);
    println!("  \"shards\": {},", engine.shard_count);
    println!("  \"reference_cases\": [");
    for (index, result) in results.iter().enumerate() {
        let comma = if index + 1 == results.len() { "" } else { "," };
        println!(
            "    {{\"name\":\"{}\",\"elements\":{},\"checksum\":{},\"elapsed_ns\":{}}}{}",
            json_escape(&result.name),
            result.elements,
            result.checksum,
            result.elapsed_ns,
            comma
        );
    }
    println!("  ],");
    println!(
        "  \"perf\": {{\"iterations\":{},\"min_ns\":{},\"max_ns\":{},\"mean_ns\":{},\"p50_ns\":{},\"checksum\":{}}}",
        perf.iterations, perf.min_ns, perf.max_ns, perf.mean_ns, perf.p50_ns, perf.checksum
    );
    println!("}}");
    Ok(())
}

fn main() {
    match run() {
        Ok(()) => {}
        Err(error) => {
            eprintln!("mistral3-b66-consumer: {error}");
            std::process::exit(1);
        }
    }
}
