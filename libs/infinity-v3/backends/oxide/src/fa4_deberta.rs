// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Thin BF16 DeBERTa binding for the public InfinityFA4 dedicated kernel.

use std::ffi::{c_char, c_int, c_void, CStr};
use std::fs::File;
use std::io::Read;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};

use cuda_core::{Context, Stream};
use libloading::Library;

use crate::{infer, start};
use infinity_v3_backend_core::BackendError;

const ABI_VERSION: u64 = 7;
const NATIVE_ABI_VERSION: u64 = 1;
const MAX_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;
const HEADS: usize = 12;
const HEAD_DIM: usize = 64;
const DTYPE_BF16: c_int = 1;
const DEBERTA_SYMBOL: &str = "fa4_deberta_bf16";
pub(crate) const EXACT_MUL_BOTH_SYMBOL: &str = "fa4_deberta_exact_mul_both_bf16";
pub(crate) const EXACT_MUL_SINGLE_SYMBOL: &str = "fa4_deberta_exact_mul_single_bf16";
const EXACT_MUL_SYMBOL_PREFIX: &str = "fa4_deberta_exact_mul_";
const EXACT_MUL_SCORE_INPUT: &str = "raw_bf16_matmul";
const EXACT_MUL_SCORE_TRANSFORM: &str = "mul.rn.f32+cvt.rn.bf16x2.f32+bit_widen_f32";
const EXACT_MUL_DIVISOR_BOTH: f32 = 13.875;
const EXACT_MUL_DIVISOR_SINGLE: f32 = 11.3125;
const EXACT_MUL_RECIPROCAL_BOTH_BITS: u64 = 0x3d93_9a86;
const EXACT_MUL_RECIPROCAL_SINGLE_BITS: u64 = 0x3db5_09e7;
const BF16_NAME: &str = "bf16";

const PORTABLE_SYMBOL: &[u8] = b"infinity_fa4_portable_host_v1\0";
const CAPABILITY_SYMBOL: &[u8] = b"infinity_fa4_compute_capability_v1\0";
const DEBERTA_ENTRY: &[u8] = b"infinity_fa4_deberta_v1\0";
const EXACT_MUL_DEBERTA_ENTRY: &[u8] = b"infinity_fa4_deberta_exact_mul_v1\0";
const ERROR_ENTRY: &[u8] = b"infinity_fa4_error_v1\0";

type PortableFn = unsafe extern "C" fn() -> c_int;
type CapabilityFn = unsafe extern "C" fn() -> c_int;
type ErrorFn = unsafe extern "C" fn() -> *const c_char;
type DebertaKernel = unsafe extern "C" fn(
    c_int,
    c_int,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    i64,
    i64,
    i64,
    i64,
    i64,
    c_int,
) -> c_int;
type ExactMulDebertaKernel = unsafe extern "C" fn(
    c_int,
    c_int,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    i64,
    i64,
    i64,
    i64,
    i64,
    c_int,
    f32,
) -> c_int;

#[derive(Clone, Copy)]
enum Kernel {
    Normalized(DebertaKernel),
    ExactMul(ExactMulDebertaKernel, f32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ColdScoreMode {
    Normalized,
    ExactMul,
}

/// Caller-owned device buffers for one packed DeBERTa launch.
pub(crate) struct DebertaBatch {
    pub(crate) q: u64,
    pub(crate) k: u64,
    pub(crate) v: u64,
    pub(crate) out: u64,
    pub(crate) offsets: u64,
    pub(crate) c2p: u64,
    pub(crate) p2c: u64,
    pub(crate) buckets: u64,
    pub(crate) tokens: usize,
    pub(crate) batch: usize,
    pub(crate) max_sequence: usize,
    pub(crate) bucket_count: usize,
}

struct Engine {
    library: ManuallyDrop<Library>,
    context: ManuallyDrop<Arc<Context>>,
    stream: ManuallyDrop<Arc<Stream>>,
    kernel: Kernel,
    error: ErrorFn,
    device: c_int,
}

impl Drop for Engine {
    fn drop(&mut self) {
        // Safety: release resident code only after a confirmed stream drain.
        // If the drain is uncertain, every resource is deliberately retained.
        if self.stream.synchronize().is_ok() {
            unsafe {
                ManuallyDrop::drop(&mut self.stream);
                ManuallyDrop::drop(&mut self.context);
                ManuallyDrop::drop(&mut self.library);
            }
        }
    }
}

struct EngineMutex(Mutex<Engine>, PhantomData<*const ()>);

// Safety: the engine is reachable only through its mutex. Moving the owner
// between threads is sound; shared cross-thread access remains unavailable.
unsafe impl Send for EngineMutex {}

impl EngineMutex {
    fn new(engine: Engine) -> Self {
        Self(Mutex::new(engine), PhantomData)
    }

    fn lock(&self) -> Result<MutexGuard<'_, Engine>, BackendError> {
        self.0
            .lock()
            .map_err(|_| bad("InfinityFA4 DeBERTa engine lock is poisoned"))
    }
}

/// A validated, resident BF16 DeBERTa bundle.
pub struct Fa4Deberta {
    engine: EngineMutex,
    span: usize,
}

// Safety: device work is serialized by the engine mutex, and the resident
// library/context/stream are retained until the stream has drained.
unsafe impl Send for Fa4Deberta {}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ManifestEntry {
    symbol: String,
    dtype: String,
    heads: u64,
    kv_heads: u64,
    head_dim: u64,
    causal: bool,
    window: [Option<i64>; 2],
}

fn bad(message: impl Into<String>) -> BackendError {
    infer(message.into())
}

fn select_deberta_entry(entries: &[ManifestEntry]) -> Result<&ManifestEntry, String> {
    let mut found = None;
    for entry in entries {
        if entry.symbol != DEBERTA_SYMBOL || entry.dtype != BF16_NAME {
            continue;
        }
        if found.is_some() {
            return Err("duplicate bf16 fa4_deberta_bf16 manifest entry".to_string());
        }
        found = Some(entry);
    }
    let entry = found.ok_or_else(|| "missing bf16 fa4_deberta_bf16 manifest entry".to_string())?;
    if entry.heads != HEADS as u64
        || entry.kv_heads != HEADS as u64
        || entry.head_dim != HEAD_DIM as u64
    {
        return Err("tampered fa4_deberta_bf16 manifest geometry".to_string());
    }
    if entry.causal || entry.window != [None, None] {
        return Err("tampered fa4_deberta_bf16 manifest window".to_string());
    }
    Ok(entry)
}

fn str_field<'a>(value: &'a serde_json::Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(serde_json::Value::as_str)
}

fn u64_field(value: &serde_json::Value, key: &str) -> Option<u64> {
    value.get(key).and_then(serde_json::Value::as_u64)
}

fn parse_window(value: Option<&serde_json::Value>) -> Option<[Option<i64>; 2]> {
    let array = value?.as_array()?;
    if array.len() != 2 {
        return None;
    }
    let mut window = [None, None];
    for (slot, item) in window.iter_mut().zip(array.iter()) {
        *slot = if item.is_null() {
            None
        } else {
            Some(item.as_i64()?)
        };
    }
    Some(window)
}

fn valid_exact_mul_scale(scale: f32) -> bool {
    scale == EXACT_MUL_DIVISOR_BOTH || scale == EXACT_MUL_DIVISOR_SINGLE
}

pub(crate) fn is_exact_mul_symbol(symbol: &str) -> bool {
    symbol == EXACT_MUL_BOTH_SYMBOL || symbol == EXACT_MUL_SINGLE_SYMBOL
}

fn validate_exact_mul_entry(entry: &serde_json::Value, symbol: &str) -> Result<(), BackendError> {
    let detail = |message: &str| bad(format!("{symbol} {message}"));
    if str_field(entry, "symbol") != Some(symbol) {
        return Err(detail("has an unexpected symbol"));
    }
    if str_field(entry, "dtype") != Some(BF16_NAME) {
        return Err(detail("must be bf16"));
    }
    let heads = u64_field(entry, "heads").ok_or_else(|| detail("is missing heads"))?;
    let kv_heads = u64_field(entry, "kv_heads").ok_or_else(|| detail("is missing kv_heads"))?;
    let head_dim = u64_field(entry, "d")
        .or_else(|| u64_field(entry, "head_dim"))
        .ok_or_else(|| detail("is missing d"))?;
    if heads != HEADS as u64 || kv_heads != HEADS as u64 || head_dim != HEAD_DIM as u64 {
        return Err(detail("has tampered geometry"));
    }
    let causal = entry
        .get("causal")
        .and_then(serde_json::Value::as_bool)
        .ok_or_else(|| detail("is missing causal"))?;
    let window =
        parse_window(entry.get("window")).ok_or_else(|| detail("has an invalid window"))?;
    if causal || window != [None, None] {
        return Err(detail("has a tampered window"));
    }
    if str_field(entry, "relative_score_input") != Some(EXACT_MUL_SCORE_INPUT) {
        return Err(detail("has an invalid relative_score_input"));
    }
    let (expected_divisor, expected_reciprocal) = if symbol == EXACT_MUL_BOTH_SYMBOL {
        (EXACT_MUL_DIVISOR_BOTH, EXACT_MUL_RECIPROCAL_BOTH_BITS)
    } else {
        (EXACT_MUL_DIVISOR_SINGLE, EXACT_MUL_RECIPROCAL_SINGLE_BITS)
    };
    let divisor = entry
        .get("relative_divisor_f32")
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| detail("is missing relative_divisor_f32"))?;
    if divisor != f64::from(expected_divisor) {
        return Err(detail("has an invalid relative_divisor_f32"));
    }
    let reciprocal = u64_field(entry, "relative_reciprocal_f32_bits")
        .ok_or_else(|| detail("is missing relative_reciprocal_f32_bits"))?;
    if reciprocal != expected_reciprocal {
        return Err(detail("has an invalid relative_reciprocal_f32_bits"));
    }
    if str_field(entry, "relative_score_transform") != Some(EXACT_MUL_SCORE_TRANSFORM) {
        return Err(detail("has an invalid relative_score_transform"));
    }
    Ok(())
}

fn exact_mul_exports(
    root: &serde_json::Value,
) -> Result<Option<(&serde_json::Value, &serde_json::Value)>, BackendError> {
    let exports = root
        .get("kernels")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| bad("InfinityFA4 manifest is missing kernels"))?;
    let mut both = None;
    let mut single = None;
    for export in exports {
        let Some(symbol) = str_field(export, "symbol") else {
            continue;
        };
        if !symbol.starts_with(EXACT_MUL_SYMBOL_PREFIX) {
            continue;
        }
        let slot = if symbol == EXACT_MUL_BOTH_SYMBOL {
            &mut both
        } else if symbol == EXACT_MUL_SINGLE_SYMBOL {
            &mut single
        } else {
            return Err(bad(format!(
                "unknown exact-mul DeBERTa export symbol {symbol}"
            )));
        };
        if slot.is_some() {
            return Err(bad(format!("duplicate exact-mul DeBERTa export {symbol}")));
        }
        *slot = Some(export);
    }
    match (both, single) {
        (None, None) => Ok(None),
        (Some(both), Some(single)) => {
            validate_exact_mul_entry(both, EXACT_MUL_BOTH_SYMBOL)?;
            validate_exact_mul_entry(single, EXACT_MUL_SINGLE_SYMBOL)?;
            Ok(Some((both, single)))
        }
        _ => Err(bad(
            "incomplete exact-mul DeBERTa manifest pair: both and single are required",
        )),
    }
}

pub(crate) fn validate_exact_mul_exports(root: &serde_json::Value) -> Result<(), BackendError> {
    exact_mul_exports(root).map(|_| ())
}

fn parse_manifest_with_mode(bytes: &[u8]) -> Result<ColdScoreMode, BackendError> {
    let root: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|error| bad(format!("invalid InfinityFA4 manifest: {error}")))?;
    if u64_field(&root, "abi_version") != Some(ABI_VERSION) {
        return Err(bad("unsupported InfinityFA4 manifest abi_version"));
    }
    if u64_field(&root, "native_abi_version") != Some(NATIVE_ABI_VERSION) {
        return Err(bad("unsupported InfinityFA4 manifest native_abi_version"));
    }
    let exact_mul = exact_mul_exports(&root)?.is_some();
    let exports = root
        .get("kernels")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| bad("InfinityFA4 manifest is missing kernels"))?;
    let mut entries = Vec::new();
    entries.try_reserve_exact(exports.len()).map_err(start)?;
    for export in exports {
        if str_field(export, "symbol") != Some(DEBERTA_SYMBOL)
            || str_field(export, "dtype") != Some(BF16_NAME)
        {
            continue;
        }
        entries.push(ManifestEntry {
            symbol: DEBERTA_SYMBOL.to_string(),
            dtype: BF16_NAME.to_string(),
            heads: u64_field(export, "heads")
                .ok_or_else(|| bad("fa4_deberta_bf16 entry is missing heads"))?,
            kv_heads: u64_field(export, "kv_heads")
                .ok_or_else(|| bad("fa4_deberta_bf16 entry is missing kv_heads"))?,
            head_dim: u64_field(export, "d")
                .or_else(|| u64_field(export, "head_dim"))
                .ok_or_else(|| bad("fa4_deberta_bf16 entry is missing d"))?,
            causal: export
                .get("causal")
                .and_then(serde_json::Value::as_bool)
                .ok_or_else(|| bad("fa4_deberta_bf16 entry is missing causal"))?,
            window: parse_window(export.get("window"))
                .ok_or_else(|| bad("fa4_deberta_bf16 entry has an invalid window"))?,
        });
    }
    select_deberta_entry(&entries).map_err(bad)?;
    Ok(if exact_mul {
        ColdScoreMode::ExactMul
    } else {
        ColdScoreMode::Normalized
    })
}

#[cfg(test)]
fn parse_manifest(bytes: &[u8]) -> Result<(), BackendError> {
    parse_manifest_with_mode(bytes).map(|_| ())
}

fn load_manifest_with_mode(path: &Path) -> Result<ColdScoreMode, BackendError> {
    let file = File::open(path)
        .map_err(|error| bad(format!("cannot open {}: {error}", path.display())))?;
    let limit = MAX_MANIFEST_BYTES
        .checked_add(1)
        .ok_or_else(|| bad("InfinityFA4 manifest size limit overflow"))?;
    let mut bytes = Vec::new();
    file.take(limit)
        .read_to_end(&mut bytes)
        .map_err(|error| bad(format!("cannot read {}: {error}", path.display())))?;
    if u64::try_from(bytes.len()).map_or(true, |length| length > MAX_MANIFEST_BYTES) {
        return Err(bad("InfinityFA4 manifest exceeds 16 MiB"));
    }
    parse_manifest_with_mode(&bytes)
}

fn checked_element_count(values: &[usize]) -> Result<usize, String> {
    let mut total = 1usize;
    for value in values {
        total = total
            .checked_mul(*value)
            .ok_or_else(|| "DeBERTa element count overflows usize".to_string())?;
    }
    if total > i32::MAX as usize {
        return Err("DeBERTa element count exceeds the kernel i32 contract".to_string());
    }
    Ok(total)
}

fn checked_extent(elements: usize, width: u64) -> Result<u64, String> {
    let bytes = u64::try_from(elements)
        .map_err(|_| "DeBERTa element count does not fit u64".to_string())?
        .checked_mul(width)
        .ok_or_else(|| "DeBERTa byte extent overflows u64".to_string())?;
    i64::try_from(bytes).map_err(|_| "DeBERTa byte extent does not fit i64".to_string())?;
    Ok(bytes)
}

fn validate_batch(batch: &DebertaBatch, span: usize) -> Result<(), String> {
    if !(1..=4096).contains(&span) {
        return Err("DeBERTa span must be between 1 and 4096".to_string());
    }
    if batch.tokens == 0 || batch.batch == 0 || batch.max_sequence == 0 {
        return Err("DeBERTa tokens, batch and max_sequence must be positive".to_string());
    }
    if batch.batch > batch.tokens || batch.max_sequence > batch.tokens {
        return Err("DeBERTa batch and max_sequence cannot exceed tokens".to_string());
    }
    let expected_buckets = batch
        .max_sequence
        .checked_mul(2)
        .and_then(|value| value.checked_sub(1))
        .ok_or_else(|| "DeBERTa bucket count overflows usize".to_string())?;
    if batch.bucket_count != expected_buckets {
        return Err("DeBERTa bucket_count must equal 2 * max_sequence - 1".to_string());
    }
    if [
        batch.q,
        batch.k,
        batch.v,
        batch.out,
        batch.offsets,
        batch.c2p,
        batch.p2c,
        batch.buckets,
    ]
    .contains(&0)
    {
        return Err("DeBERTa buffer pointer is null".to_string());
    }
    if [batch.q, batch.k, batch.v, batch.out, batch.c2p, batch.p2c]
        .iter()
        .any(|pointer| pointer % 2 != 0)
    {
        return Err("DeBERTa BF16 buffer is not 2-byte aligned".to_string());
    }
    if !batch.offsets.is_multiple_of(4) || !batch.buckets.is_multiple_of(4) {
        return Err("DeBERTa metadata buffer is not 4-byte aligned".to_string());
    }
    let two_span = span
        .checked_mul(2)
        .ok_or_else(|| "DeBERTa span extent overflows usize".to_string())?;
    let qkv_extent = checked_extent(checked_element_count(&[batch.tokens, HEADS, HEAD_DIM])?, 2)?;
    let table_extent = checked_extent(checked_element_count(&[HEADS, batch.tokens, two_span])?, 2)?;
    let offset_extent = checked_extent(
        checked_element_count(&[batch
            .batch
            .checked_add(1)
            .ok_or_else(|| "DeBERTa offset count overflows usize".to_string())?])?,
        4,
    )?;
    let bucket_extent = checked_extent(checked_element_count(&[batch.bucket_count])?, 4)?;
    let out_end = batch
        .out
        .checked_add(qkv_extent)
        .ok_or_else(|| "DeBERTa output extent overflows address space".to_string())?;
    for (name, start, extent) in [
        ("q", batch.q, qkv_extent),
        ("k", batch.k, qkv_extent),
        ("v", batch.v, qkv_extent),
        ("offsets", batch.offsets, offset_extent),
        ("c2p", batch.c2p, table_extent),
        ("p2c", batch.p2c, table_extent),
        ("buckets", batch.buckets, bucket_extent),
    ] {
        let end = start
            .checked_add(extent)
            .ok_or_else(|| format!("DeBERTa {name} extent overflows address space"))?;
        if batch.out < end && start < out_end {
            return Err(format!("DeBERTa output aliases {name}"));
        }
    }
    Ok(())
}

fn status_error(error: ErrorFn, status: c_int) -> BackendError {
    // Safety: the bundle contract guarantees a null or valid resident C string.
    let detail = unsafe {
        let pointer = error();
        if pointer.is_null() {
            None
        } else {
            Some(CStr::from_ptr(pointer).to_string_lossy().into_owned())
        }
    };
    match detail {
        Some(message) if !message.is_empty() => {
            bad(format!("InfinityFA4 DeBERTa failed: {message}"))
        }
        _ => bad(format!("InfinityFA4 DeBERTa failed with status {status}")),
    }
}

impl Fa4Deberta {
    pub fn new(
        context: Arc<Context>,
        stream: Arc<Stream>,
        root: &Path,
        span: usize,
        relative_scale: f32,
    ) -> Result<(Self, ColdScoreMode), BackendError> {
        if !valid_exact_mul_scale(relative_scale) {
            return Err(bad(
                "InfinityFA4 exact-mul DeBERTa scale must be exactly 13.875 or 11.3125",
            ));
        }
        if !(1..=4096).contains(&span) {
            return Err(bad("DeBERTa span must be between 1 and 4096"));
        }
        if !Arc::ptr_eq(stream.context(), &context) {
            return Err(bad("InfinityFA4 DeBERTa context/stream mismatch"));
        }
        context.bind_to_thread().map_err(start)?;
        let (major, minor) = context.compute_capability().map_err(start)?;
        let capability = major
            .checked_mul(10)
            .and_then(|value| value.checked_add(minor))
            .ok_or_else(|| bad("invalid InfinityFA4 compute capability"))?;
        if !matches!(capability, 90 | 100 | 120) {
            return Err(bad(format!(
                "unsupported InfinityFA4 capability sm_{capability}"
            )));
        }
        let device = i32::try_from(context.ordinal())
            .map_err(|_| bad("CUDA device ordinal does not fit the kernel ABI"))?;
        let directory = root.join(format!("sm_{capability}"));
        let manifest_path = directory.join("manifest.json");
        let score_mode = load_manifest_with_mode(&manifest_path)?;
        let library_path = directory.join("libinfinityfa4.so");
        let library = unsafe { Library::new(&library_path) }
            .map_err(|error| bad(format!("cannot load {}: {error}", library_path.display())))?;
        let portable = *unsafe { library.get::<PortableFn>(PORTABLE_SYMBOL) }
            .map_err(|error| bad(format!("missing portable host entry: {error}")))?;
        let capability_fn = *unsafe { library.get::<CapabilityFn>(CAPABILITY_SYMBOL) }
            .map_err(|error| bad(format!("missing capability entry: {error}")))?;
        let kernel = match score_mode {
            ColdScoreMode::ExactMul => {
                let exact =
                    *unsafe { library.get::<ExactMulDebertaKernel>(EXACT_MUL_DEBERTA_ENTRY) }
                        .map_err(|error| {
                            bad(format!(
                                "missing dedicated exact-mul DeBERTa entry: {error}"
                            ))
                        })?;
                Kernel::ExactMul(exact, relative_scale)
            }
            ColdScoreMode::Normalized => {
                let normalized = *unsafe { library.get::<DebertaKernel>(DEBERTA_ENTRY) }
                    .map_err(|error| bad(format!("missing dedicated DeBERTa entry: {error}")))?;
                Kernel::Normalized(normalized)
            }
        };
        let error = *unsafe { library.get::<ErrorFn>(ERROR_ENTRY) }
            .map_err(|error| bad(format!("missing error entry: {error}")))?;
        if unsafe { portable() } != 1 {
            return Err(bad("InfinityFA4 bundle is not portable host code"));
        }
        if unsafe { capability_fn() } != capability {
            return Err(bad("InfinityFA4 bundle capability mismatch"));
        }
        Ok((
            Self {
                engine: EngineMutex::new(Engine {
                    library: ManuallyDrop::new(library),
                    context: ManuallyDrop::new(context),
                    stream: ManuallyDrop::new(stream),
                    kernel,
                    error,
                    device,
                }),
                span,
            },
            score_mode,
        ))
    }

    pub fn check_stream(&self, stream: &Stream) -> Result<(), BackendError> {
        let engine = self.engine.lock()?;
        if Arc::ptr_eq(stream.context(), &*engine.context)
            && stream.cu_stream() == engine.stream.cu_stream()
        {
            Ok(())
        } else {
            Err(bad(
                "InfinityFA4 DeBERTa launch must use its prepared context and ordered stream",
            ))
        }
    }

    pub fn token_limit(&self) -> usize {
        i32::MAX as usize / (HEADS * (2 * self.span).max(HEAD_DIM))
    }

    /// # Safety
    /// All `batch` pointers must be valid, live, caller-owned device allocations
    /// in this engine's context. QKV/output are contiguous `[tokens,12,64]` BF16;
    /// relative tables are contiguous `[12,tokens,2*span]` BF16; offsets and
    /// buckets are contiguous int32. Offset contents must be validated on the
    /// host by the caller. The engine remains resident until the stream drains.
    pub unsafe fn launch_bound(
        &self,
        stream: &Stream,
        batch: &DebertaBatch,
    ) -> Result<(), BackendError> {
        let engine = self.engine.lock()?;
        if !Arc::ptr_eq(stream.context(), &*engine.context)
            || stream.cu_stream() != engine.stream.cu_stream()
        {
            return Err(bad(
                "InfinityFA4 DeBERTa launch stream belongs to another context",
            ));
        }
        validate_batch(batch, self.span).map_err(bad)?;
        let status = match engine.kernel {
            Kernel::Normalized(deberta) => unsafe {
                (deberta)(
                    DTYPE_BF16,
                    engine.device,
                    stream.cu_stream().cast(),
                    batch.q as *mut c_void,
                    batch.k as *mut c_void,
                    batch.v as *mut c_void,
                    batch.out as *mut c_void,
                    batch.offsets as *mut c_void,
                    batch.c2p as *mut c_void,
                    batch.p2c as *mut c_void,
                    batch.buckets as *mut c_void,
                    batch.tokens as i64,
                    batch.batch as i64,
                    HEADS as i64,
                    self.span as i64,
                    batch.bucket_count as i64,
                    batch.max_sequence as c_int,
                )
            },
            Kernel::ExactMul(exact, divisor) => unsafe {
                (exact)(
                    DTYPE_BF16,
                    engine.device,
                    stream.cu_stream().cast(),
                    batch.q as *mut c_void,
                    batch.k as *mut c_void,
                    batch.v as *mut c_void,
                    batch.out as *mut c_void,
                    batch.offsets as *mut c_void,
                    batch.c2p as *mut c_void,
                    batch.p2c as *mut c_void,
                    batch.buckets as *mut c_void,
                    batch.tokens as i64,
                    batch.batch as i64,
                    HEADS as i64,
                    self.span as i64,
                    batch.bucket_count as i64,
                    batch.max_sequence as c_int,
                    divisor,
                )
            },
        };
        if status == 0 {
            Ok(())
        } else {
            Err(status_error(engine.error, status))
        }
    }
}

#[cfg(test)]
pub(crate) fn test_exact_mul_kernel(symbol: &str) -> serde_json::Value {
    let both = symbol == EXACT_MUL_BOTH_SYMBOL;
    serde_json::json!({
        "symbol": symbol,
        "dtype": BF16_NAME,
        "heads": HEADS,
        "kv_heads": HEADS,
        "d": HEAD_DIM,
        "causal": false,
        "window": [null, null],
        "relative_score_input": EXACT_MUL_SCORE_INPUT,
        "relative_divisor_f32": if both { EXACT_MUL_DIVISOR_BOTH } else { EXACT_MUL_DIVISOR_SINGLE },
        "relative_reciprocal_f32_bits": if both { EXACT_MUL_RECIPROCAL_BOTH_BITS } else { EXACT_MUL_RECIPROCAL_SINGLE_BITS },
        "relative_score_transform": EXACT_MUL_SCORE_TRANSFORM,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry() -> ManifestEntry {
        ManifestEntry {
            symbol: DEBERTA_SYMBOL.to_string(),
            dtype: BF16_NAME.to_string(),
            heads: HEADS as u64,
            kv_heads: HEADS as u64,
            head_dim: HEAD_DIM as u64,
            causal: false,
            window: [None, None],
        }
    }

    fn batch() -> DebertaBatch {
        DebertaBatch {
            q: 0x1_0000,
            k: 0x2_0000,
            v: 0x3_0000,
            out: 0x4_0000,
            offsets: 0x5_0000,
            c2p: 0x6_0000,
            p2c: 0x7_0000,
            buckets: 0x8_0000,
            tokens: 8,
            batch: 2,
            max_sequence: 4,
            bucket_count: 7,
        }
    }

    #[test]
    fn manifest_parser_requires_native_kernel_shape_and_single_entry() {
        use serde_json::json;
        let entry = json!({"symbol":"fa4_deberta_bf16","dtype":"bf16","heads":12,"kv_heads":12,"d":64,"causal":false,"window":[null,null]});
        let mut manifest =
            json!({"abi_version":7,"native_abi_version":1,"kernels":[entry.clone()]});
        let parse = |v: &serde_json::Value| parse_manifest(&serde_json::to_vec(v).unwrap());
        assert!(parse(&manifest).is_ok());
        manifest["kernels"] = json!([entry.clone(), entry.clone()]);
        assert!(parse(&manifest).is_err());
        manifest["kernels"] = json!([entry]);
        for (key, value) in [
            ("heads", json!(11)),
            ("kv_heads", json!(8)),
            ("d", json!(128)),
            ("causal", json!(true)),
            ("window", json!([-1, null])),
        ] {
            let mut bad = manifest.clone();
            bad["kernels"][0][key] = value;
            assert!(parse(&bad).is_err());
        }
        let mut wrong = manifest.clone();
        wrong["native_abi_version"] = json!(2);
        assert!(parse(&wrong).is_err());
        let kernels = manifest.as_object_mut().unwrap().remove("kernels").unwrap();
        manifest["exports"] = kernels;
        assert!(parse(&manifest).is_err());
    }

    #[test]
    fn manifest_requires_exactly_one_matching_entry() {
        let mut wrong_dtype = entry();
        wrong_dtype.dtype = "fp16".to_string();
        assert!(select_deberta_entry(&[wrong_dtype]).is_err());
        assert!(select_deberta_entry(&[entry(), entry()]).is_err());
        assert!(select_deberta_entry(&[entry()]).is_ok());
    }

    #[test]
    fn manifest_rejects_tampered_entries() {
        let mut heads = entry();
        heads.heads = 11;
        assert!(select_deberta_entry(&[heads]).is_err());
        let mut causal = entry();
        causal.causal = true;
        assert!(select_deberta_entry(&[causal]).is_err());
        let mut window = entry();
        window.window = [Some(-1), None];
        assert!(select_deberta_entry(&[window]).is_err());
    }

    #[test]
    fn batch_rejects_bad_geometry_alias_and_overflow() {
        let mut buckets = batch();
        buckets.bucket_count = 6;
        assert!(validate_batch(&buckets, 4).is_err());
        let mut alias = batch();
        alias.out = alias.q;
        assert!(validate_batch(&alias, 4).is_err());
        let mut overflow = batch();
        overflow.tokens = usize::MAX;
        overflow.batch = 1;
        overflow.max_sequence = 1;
        overflow.bucket_count = 1;
        assert!(validate_batch(&overflow, 4).is_err());
        assert!(validate_batch(&batch(), 0).is_err());
    }

    fn exact_mul_manifest() -> serde_json::Value {
        let mut kernels = vec![serde_json::json!({
            "symbol": "fa4_deberta_bf16",
            "dtype": "bf16",
            "heads": 12,
            "kv_heads": 12,
            "d": 64,
            "causal": false,
            "window": [null, null]
        })];
        kernels.push(test_exact_mul_kernel(EXACT_MUL_BOTH_SYMBOL));
        kernels.push(test_exact_mul_kernel(EXACT_MUL_SINGLE_SYMBOL));
        serde_json::json!({
            "abi_version": 7,
            "native_abi_version": 1,
            "kernels": kernels,
        })
    }

    #[test]
    fn exact_mul_exports_require_one_complete_valid_pair() {
        let both = test_exact_mul_kernel(EXACT_MUL_BOTH_SYMBOL);
        let single = test_exact_mul_kernel(EXACT_MUL_SINGLE_SYMBOL);
        let valid = serde_json::json!({"kernels": [both.clone(), single.clone()]});
        assert!(validate_exact_mul_exports(&valid).is_ok());

        let absent = serde_json::json!({"kernels": []});
        assert!(validate_exact_mul_exports(&absent).is_ok());

        for partial in [
            serde_json::json!({"kernels": [both.clone()]}),
            serde_json::json!({"kernels": [single.clone()]}),
        ] {
            assert!(validate_exact_mul_exports(&partial).is_err());
        }

        let duplicate =
            serde_json::json!({"kernels": [both.clone(), both.clone(), single.clone()]});
        assert!(validate_exact_mul_exports(&duplicate).is_err());
        let duplicate_single = serde_json::json!({
            "kernels": [both.clone(), single.clone(), single.clone()]
        });
        assert!(validate_exact_mul_exports(&duplicate_single).is_err());

        let unknown = serde_json::json!({
            "kernels": [
                both.clone(),
                single.clone(),
                test_exact_mul_kernel("fa4_deberta_exact_mul_extra_bf16")
            ]
        });
        assert!(validate_exact_mul_exports(&unknown).is_err());

        let mut wrong_dtype = both;
        wrong_dtype["dtype"] = serde_json::json!("fp16");
        let wrong_dtype = serde_json::json!({"kernels": [wrong_dtype, single]});
        assert!(validate_exact_mul_exports(&wrong_dtype).is_err());
    }

    #[test]
    fn exact_mul_divisor_and_reciprocal_are_bound_to_export_symbol() {
        for symbol in [EXACT_MUL_BOTH_SYMBOL, EXACT_MUL_SINGLE_SYMBOL] {
            let both_symbol = symbol == EXACT_MUL_BOTH_SYMBOL;
            let (other_divisor, other_bits, expected_bits) = if both_symbol {
                (
                    EXACT_MUL_DIVISOR_SINGLE,
                    EXACT_MUL_RECIPROCAL_SINGLE_BITS,
                    EXACT_MUL_RECIPROCAL_BOTH_BITS,
                )
            } else {
                (
                    EXACT_MUL_DIVISOR_BOTH,
                    EXACT_MUL_RECIPROCAL_BOTH_BITS,
                    EXACT_MUL_RECIPROCAL_SINGLE_BITS,
                )
            };
            let set = |field: &str, value: serde_json::Value| {
                let mut manifest = exact_mul_manifest();
                let export = manifest["kernels"]
                    .as_array_mut()
                    .unwrap()
                    .iter_mut()
                    .find(|entry| entry["symbol"] == symbol)
                    .unwrap();
                export[field] = value;
                validate_exact_mul_exports(&manifest)
            };
            assert!(set("relative_divisor_f32", serde_json::json!(other_divisor)).is_err());
            assert!(set(
                "relative_reciprocal_f32_bits",
                serde_json::json!(other_bits)
            )
            .is_err());
            assert!(set(
                "relative_reciprocal_f32_bits",
                serde_json::json!(expected_bits + 1)
            )
            .is_err());
        }
        let mut swapped = exact_mul_manifest();
        for export in swapped["kernels"].as_array_mut().unwrap() {
            if export["symbol"] == EXACT_MUL_BOTH_SYMBOL {
                export["relative_divisor_f32"] = serde_json::json!(EXACT_MUL_DIVISOR_SINGLE);
                export["relative_reciprocal_f32_bits"] =
                    serde_json::json!(EXACT_MUL_RECIPROCAL_SINGLE_BITS);
            } else if export["symbol"] == EXACT_MUL_SINGLE_SYMBOL {
                export["relative_divisor_f32"] = serde_json::json!(EXACT_MUL_DIVISOR_BOTH);
                export["relative_reciprocal_f32_bits"] =
                    serde_json::json!(EXACT_MUL_RECIPROCAL_BOTH_BITS);
            }
        }
        assert!(validate_exact_mul_exports(&swapped).is_err());
    }

    #[test]
    fn exact_mul_exports_reject_missing_or_tampered_fields() {
        let single = test_exact_mul_kernel(EXACT_MUL_SINGLE_SYMBOL);
        for (field, value) in [
            ("heads", serde_json::json!(11)),
            ("kv_heads", serde_json::json!(8)),
            ("d", serde_json::json!(128)),
            ("causal", serde_json::json!(true)),
            ("window", serde_json::json!([-1, null])),
            ("relative_score_input", serde_json::json!("bf16_matmul")),
            ("relative_divisor_f32", serde_json::json!(13.8)),
            ("relative_divisor_f32", serde_json::json!("13.875")),
            (
                "relative_reciprocal_f32_bits",
                serde_json::json!(EXACT_MUL_RECIPROCAL_SINGLE_BITS),
            ),
            (
                "relative_reciprocal_f32_bits",
                serde_json::json!(format!("{EXACT_MUL_RECIPROCAL_BOTH_BITS}")),
            ),
            ("relative_score_transform", serde_json::json!("div.rn.f32")),
        ] {
            let mut wrong = test_exact_mul_kernel(EXACT_MUL_BOTH_SYMBOL);
            wrong[field] = value;
            let manifest = serde_json::json!({"kernels": [wrong, single.clone()]});
            assert!(
                validate_exact_mul_exports(&manifest).is_err(),
                "accepted corrupt exact-mul {field}"
            );
        }
        for field in [
            "symbol",
            "dtype",
            "heads",
            "kv_heads",
            "d",
            "causal",
            "window",
            "relative_score_input",
            "relative_divisor_f32",
            "relative_reciprocal_f32_bits",
            "relative_score_transform",
        ] {
            let mut wrong = test_exact_mul_kernel(EXACT_MUL_BOTH_SYMBOL);
            wrong.as_object_mut().unwrap().remove(field);
            let manifest = serde_json::json!({"kernels": [wrong, single.clone()]});
            assert!(
                validate_exact_mul_exports(&manifest).is_err(),
                "accepted exact-mul missing {field}"
            );
        }
    }

    #[test]
    fn manifest_parse_prefers_complete_exact_mul_pair() {
        let exact = exact_mul_manifest();
        let bytes = serde_json::to_vec(&exact).unwrap();
        assert_eq!(
            parse_manifest_with_mode(&bytes).unwrap(),
            ColdScoreMode::ExactMul
        );
    }

    #[test]
    fn manifest_parse_falls_back_to_normalized_without_exact_pair() {
        let mut normalized = exact_mul_manifest();
        normalized["kernels"].as_array_mut().unwrap().truncate(1);
        let bytes = serde_json::to_vec(&normalized).unwrap();
        assert_eq!(
            parse_manifest_with_mode(&bytes).unwrap(),
            ColdScoreMode::Normalized
        );
    }

    #[test]
    fn manifest_parse_rejects_malformed_exact_pairs() {
        let mut incomplete = exact_mul_manifest();
        incomplete["kernels"].as_array_mut().unwrap().pop();
        let bytes = serde_json::to_vec(&incomplete).unwrap();
        assert!(parse_manifest_with_mode(&bytes).is_err());

        let mut corrupt = exact_mul_manifest();
        let export = corrupt["kernels"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|entry| entry["symbol"] == EXACT_MUL_BOTH_SYMBOL)
            .unwrap();
        export["relative_reciprocal_f32_bits"] =
            serde_json::json!(EXACT_MUL_RECIPROCAL_BOTH_BITS + 1);
        let bytes = serde_json::to_vec(&corrupt).unwrap();
        assert!(parse_manifest_with_mode(&bytes).is_err());

        let mut missing_normalized = exact_mul_manifest();
        missing_normalized["kernels"]
            .as_array_mut()
            .unwrap()
            .remove(0);
        let bytes = serde_json::to_vec(&missing_normalized).unwrap();
        assert!(parse_manifest_with_mode(&bytes).is_err());
    }

    #[test]
    fn exact_mul_scale_is_exactly_the_two_supported_values() {
        assert!(valid_exact_mul_scale(13.875));
        assert!(valid_exact_mul_scale(11.3125));
        for invalid in [0.0, 11.3124, 13.875_001, f32::NAN, f32::INFINITY] {
            assert!(!valid_exact_mul_scale(invalid));
        }
    }
}
