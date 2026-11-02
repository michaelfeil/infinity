// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Torch-free ABI2 loader and typed launcher for the dense Qwen3.5 GDN AOT provider.
//!
//! `libinfinitygdn.so` is the CuTe/TVM-FFI host wrapper around the unmodified
//! FlashInfer fresh and initial-state prefill kernels. This module validates the
//! installed bundle on the CPU, binds it to one CUDA device exactly once, and
//! launches the exact twenty-argument ABI2 call with BF16 inputs. It does not
//! link or load LibTorch.
//!
//! One `manifest.json` descriptor is accepted in two explicitly distinguished
//! forms. The normal producer form carries ABI2, both required capabilities,
//! architecture exports and library digests. A bundle without the original
//! producer manifest can instead use `kind = "infinity.gdn.aot.derived-donor.v1"`.
//! That form is self-derived admission metadata, not producer provenance: it
//! records the source SDK digest separately from the exact provider ELF
//! digest, and its exports use `digest_kind = "containing-library"`.
//! Source-SDK proof remains outside this loader.
//!
//! The cold path owns an `Arc<libloading::Library>`; cloning the provider keeps
//! that library resident for every launch owner. Launch errors permanently
//! poison the provider because the native bridge retains a process-static
//! failure once a launch throws. The hot path performs only pure shape,
//! alignment, overlap and overflow checks and never queries CUDA or allocates
//! on success.

use crate::{infer, start};
use cuda_core::Context;
use infinity_v3_backend_core::BackendError;
use libloading::Library;
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    ffi::{c_char, c_void, CStr},
    fs::{self, File},
    io::Read,
    path::{Component, Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

const ABI_VERSION: u32 = 2;
const PRODUCER_KIND: &str = "infinity.gdn.aot.producer.v1";
const DERIVED_KIND: &str = "infinity.gdn.aot.derived-donor.v1";
const MANIFEST_FILE: &str = "manifest.json";
const MAIN_LIBRARY: &str = "libinfinitygdn.so";
const REQUIRED_LIBRARIES: [&str; 3] = [MAIN_LIBRARY, "libtvm_ffi.so", "libcute_dsl_runtime.so"];
const REQUIRED_CAPABILITIES: [&str; 2] = ["fresh_prefill", "initial_state"];
const FLASHINFER_VERSION: &str = "0.7.2";
const BF16_DTYPE: i32 = 1;
const HEAD_DIM: u64 = 128;
const MAX_VALUE_HEADS: u64 = 128;
const DATA_ALIGNMENT: u64 = 16;
const CUMULATIVE_ALIGNMENT: u64 = 4;
const WORKSPACE_ALIGNMENT: u64 = 128;
const MAX_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;

type Supported = unsafe extern "C" fn(i32) -> i32;
type Error = unsafe extern "C" fn() -> *const c_char;
// Exact ABI2 export from vendor/infinity_gdn/bridge.h.
type Forward = unsafe extern "C" fn(
    i32,
    i32,
    i32,
    *mut c_void,
    i64,
    i64,
    i64,
    i64,
    i32,
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
    *mut c_void,
) -> i32;

fn start_error(error: impl std::fmt::Display) -> BackendError {
    start(format!("Oxide GDN AOT: {error}"))
}

fn infer_error(error: impl std::fmt::Display) -> BackendError {
    infer(format!("Oxide GDN AOT: {error}"))
}

/// Exact device facts sampled once while the provider is bound cold.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DeviceFacts {
    pub(crate) ordinal: i32,
    pub(crate) major: i32,
    pub(crate) sm_count: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ManifestKind {
    Producer,
    Derived,
}

#[derive(Clone, Debug)]
struct ExpectedFile {
    path: String,
    sha256: String,
    size: Option<u64>,
}

#[derive(Deserialize)]
struct Manifest {
    #[serde(default)]
    kind: Option<String>,
    abi_version: u32,
    capabilities: Vec<String>,
    flashinfer: Option<String>,
    architectures: Vec<ArchitectureRecord>,
    #[serde(default)]
    objects: Option<BTreeMap<String, String>>,
    libraries: BTreeMap<String, LibraryExpectation>,
    #[serde(default)]
    source_sdk_sha256: Option<String>,
    #[serde(default)]
    source_provider_sha256: Option<String>,
    #[serde(flatten)]
    #[allow(dead_code)]
    metadata: BTreeMap<String, Value>,
}

impl Manifest {
    fn kind(&self) -> Result<ManifestKind, BackendError> {
        match self.kind.as_deref() {
            None | Some(PRODUCER_KIND) => Ok(ManifestKind::Producer),
            Some(DERIVED_KIND) => Ok(ManifestKind::Derived),
            Some(other) => Err(start_error(format!(
                "manifest kind {other:?} is neither a producer nor a derived donor tag"
            ))),
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ArchitectureRecord {
    architecture: String,
    exports: Vec<ExportRecord>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
enum DigestKind {
    #[default]
    Object,
    ContainingLibrary,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExportRecord {
    symbol: String,
    initial_state: bool,
    sha256: String,
    #[serde(default)]
    digest_kind: DigestKind,
    #[serde(default)]
    object: Option<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum LibraryExpectation {
    Digest(String),
    File(LibraryFile),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LibraryFile {
    path: String,
    sha256: String,
    size: u64,
}

/// Cold manifest, architecture and device decision. Holds no CUDA or dlopen
/// state, so every field can be validated by a CPU-only test.
#[derive(Debug)]
pub(crate) struct GdnAotPlan {
    library: PathBuf,
    major: i32,
    device: i32,
    sm_count: i32,
    workspace_bytes: i64,
}

impl GdnAotPlan {
    /// Read and validate one installed provider bundle without loading it.
    pub(crate) fn load(root: &Path, facts: DeviceFacts) -> Result<Self, BackendError> {
        if facts.ordinal < 0 || facts.sm_count <= 0 {
            return Err(start_error("invalid cold device facts"));
        }
        let root = root
            .canonicalize()
            .map_err(|error| start_error(format!("{}: {error}", root.display())))?;
        let manifest_path = resolve_expected_path(&root, MANIFEST_FILE)?;
        let manifest_bytes = read_bounded_file(&manifest_path, MAX_MANIFEST_BYTES)?;
        let manifest = parse_manifest(&manifest_bytes)?;
        let kind = manifest.kind()?;
        let expected = library_expectations(&manifest, kind)?;

        let mut main_library = None;
        for (name, record) in REQUIRED_LIBRARIES.iter().zip(&expected) {
            let path = verify_expected_file(&root, record)?;
            if *name == MAIN_LIBRARY {
                main_library = Some((path, record.sha256.clone()));
            }
        }
        let (library, main_sha256) =
            main_library.ok_or_else(|| start_error("missing main GDN library"))?;
        validate_manifest(&manifest, facts.major, kind, &main_sha256)?;
        let workspace_bytes = workspace_bytes(facts.major, facts.sm_count)?;
        Ok(Self {
            library,
            major: facts.major,
            device: facts.ordinal,
            sm_count: facts.sm_count,
            workspace_bytes,
        })
    }

    fn workspace_bytes(&self) -> i64 {
        self.workspace_bytes
    }
}

/// Clonable cold owner. Every clone shares the library, poison state and
/// context, so an invocation owner can retain the provider through GPU
/// completion without reloading or revalidating the bundle.
#[derive(Clone)]
pub(crate) struct GdnAotProvider {
    inner: Arc<ProviderInner>,
}

struct ProviderInner {
    _library: Arc<Library>,
    _context: Arc<Context>,
    forward: Forward,
    error: Error,
    poisoned: AtomicBool,
    plan: GdnAotPlan,
}

impl GdnAotProvider {
    /// Bind the provider context, sample ordinal/major/SM count exactly once,
    /// validate the installed bundle, then load the host library.
    pub(crate) fn load(root: &Path, context: Arc<Context>) -> Result<Self, BackendError> {
        context
            .bind_to_thread()
            .map_err(|error| start_error(format!("context bind failed: {error}")))?;
        let ordinal = i32::try_from(context.ordinal())
            .map_err(|_| start_error("device ordinal exceeds the native int range"))?;
        let (major, _minor) = context
            .compute_capability()
            .map_err(|error| start_error(format!("compute capability query failed: {error}")))?;
        let sm_count = i32::try_from(
            context
                .multiprocessor_count()
                .map_err(|error| start_error(format!("SM count query failed: {error}")))?,
        )
        .map_err(|_| start_error("SM count exceeds the native int range"))?;
        let facts = DeviceFacts {
            ordinal,
            major,
            sm_count,
        };
        let plan = GdnAotPlan::load(root, facts)?;
        // SAFETY: the canonical path was fixed by the validated bundle and the
        // library is retained for at least as long as the provider. The host
        // wrapper's RUNPATH loads its TVM-FFI and CuTe dependencies from the
        // same directory; no LibTorch dependency is present in the ABI2 build.
        let library = Arc::new(
            unsafe { Library::new(&plan.library) }
                .map_err(|error| start_error(format!("{}: {error}", plan.library.display())))?,
        );
        // SAFETY: every symbol has the exact ABI2 scalar signature and the
        // library outlives the copied function pointers through `_library`.
        let (supported, forward, error) = unsafe {
            (
                *library
                    .get::<Supported>(b"infinity_gdn_supported_v1\0")
                    .map_err(|error| start_error(format!("supported symbol: {error}")))?,
                *library
                    .get::<Forward>(b"infinity_gdn_forward_v2\0")
                    .map_err(|error| start_error(format!("forward symbol: {error}")))?,
                *library
                    .get::<Error>(b"infinity_gdn_error_v1\0")
                    .map_err(|error| start_error(format!("error symbol: {error}")))?,
            )
        };
        // SAFETY: the scalar query has no borrowed arguments and the library is
        // retained by the provider under construction.
        if unsafe { supported(plan.major) } == 0 {
            return Err(start_error(format!(
                "native GDN wrapper does not support SM{}",
                plan.major
            )));
        }
        Ok(Self {
            inner: Arc::new(ProviderInner {
                _library: library,
                _context: context,
                forward,
                error,
                poisoned: AtomicBool::new(false),
                plan,
            }),
        })
    }

    pub(crate) fn workspace_bytes(&self) -> i64 {
        self.inner.plan.workspace_bytes()
    }

    pub(crate) fn is_healthy(&self) -> bool {
        !self.inner.poisoned.load(Ordering::Acquire)
    }

    /// Enqueue one packed GDN ABI2 launch. No CUDA query, allocation, symbol
    /// lookup or policy selection occurs here. The provider retains the library
    /// and context; the caller retains the stream and every device buffer until
    /// the stream work completes.
    ///
    /// # Safety
    ///
    /// `stream` must be a live stream for this provider's context and every
    /// nonzero device pointer must remain valid, writable where required, and
    /// unmodified until the launch completes. All pointers must belong to the
    /// provider's device. `cumulative` must contain `batch + 1` little-endian
    /// int32 boundaries starting at zero, strictly increasing, and ending at
    /// `tokens`. `state` is contiguous FP32 `[batch, value_heads, 128, 128]`;
    /// `initial`, when nonzero, is contiguous FP32
    /// `[batch, value_heads, 128, 128]` and must not overlap `state`. `output`
    /// is contiguous BF16 `[tokens, value_heads, 128]`; Q/K are BF16
    /// `[tokens, key_heads, 128]`; V is BF16 `[tokens, value_heads, 128]`;
    /// gates are contiguous FP32 `[tokens, value_heads]`; and `workspace` is
    /// exactly `workspace_bytes()` bytes. The caller must keep all of those
    /// allocations alive past stream completion.
    pub(crate) unsafe fn forward(&self, launch: &GdnAotLaunch) -> Result<(), BackendError> {
        if self.inner.poisoned.load(Ordering::Acquire) {
            return Err(infer_error(
                "provider is permanently poisoned after a failed native launch",
            ));
        }
        let validated = validate_launch(
            self.inner.plan.major,
            self.inner.plan.sm_count,
            self.inner.plan.workspace_bytes,
            launch,
        )?;
        // SAFETY: the caller upholds stream/context and buffer lifetime; the
        // pure checks above establish the exact ABI2 shape, alignment, extent
        // and non-overlap contract. The native wrapper does not retain Rust
        // stack data.
        let status = unsafe {
            (self.inner.forward)(
                BF16_DTYPE,
                self.inner.plan.major,
                self.inner.plan.device,
                validated.stream,
                validated.tokens,
                validated.batch,
                validated.key_heads,
                validated.value_heads,
                self.inner.plan.sm_count,
                validated.q,
                validated.k,
                validated.v,
                validated.alpha,
                validated.beta,
                validated.output,
                validated.cumulative,
                validated.state,
                validated.workspace,
                validated.workspace_bytes,
                validated.initial,
            )
        };
        if status == 0 {
            return Ok(());
        }
        // The native error is thread-local, so copy it on this thread before
        // any other call could replace it, and make the static bridge failure
        // permanent for this provider.
        let text = unsafe {
            let pointer = (self.inner.error)();
            if pointer.is_null() {
                format!("native status {status}")
            } else {
                CStr::from_ptr(pointer).to_string_lossy().into_owned()
            }
        };
        self.inner.poisoned.store(true, Ordering::SeqCst);
        Err(infer_error(format!("native launch failed: {text}")))
    }
}

/// One typed ABI2 launch. A zero `initial` selects the fresh zero-state kernel;
/// a nonzero `initial` selects the immutable initial-state kernel. Addresses
/// are raw device addresses because this layer deliberately performs no CUDA
/// handle lookup or context query on the hot path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GdnAotLaunch {
    pub(crate) stream: u64,
    pub(crate) tokens: u64,
    pub(crate) batch: u64,
    pub(crate) key_heads: u64,
    pub(crate) value_heads: u64,
    pub(crate) q: u64,
    pub(crate) k: u64,
    pub(crate) v: u64,
    pub(crate) alpha: u64,
    pub(crate) beta: u64,
    pub(crate) output: u64,
    pub(crate) cumulative: u64,
    pub(crate) state: u64,
    pub(crate) workspace: u64,
    pub(crate) workspace_bytes: u64,
    pub(crate) initial: u64,
}

struct ValidatedLaunch {
    stream: *mut c_void,
    tokens: i64,
    batch: i64,
    key_heads: i64,
    value_heads: i64,
    q: *mut c_void,
    k: *mut c_void,
    v: *mut c_void,
    alpha: *mut c_void,
    beta: *mut c_void,
    output: *mut c_void,
    cumulative: *mut c_void,
    state: *mut c_void,
    workspace: *mut c_void,
    workspace_bytes: i64,
    initial: *mut c_void,
}

fn parse_manifest(bytes: &[u8]) -> Result<Manifest, BackendError> {
    serde_json::from_slice(bytes)
        .map_err(|error| start_error(format!("invalid provider manifest: {error}")))
}

fn library_expectations(
    manifest: &Manifest,
    kind: ManifestKind,
) -> Result<Vec<ExpectedFile>, BackendError> {
    validate_library_names(manifest.libraries.keys())?;
    let mut expected = Vec::new();
    expected
        .try_reserve_exact(REQUIRED_LIBRARIES.len())
        .map_err(|_| start_error("provider library allocation failed"))?;
    for name in REQUIRED_LIBRARIES {
        let record = manifest
            .libraries
            .get(name)
            .ok_or_else(|| start_error(format!("manifest lacks {name}")))?;
        let (path, sha256, size) = match record {
            LibraryExpectation::Digest(sha256) => {
                if kind == ManifestKind::Derived {
                    return Err(start_error(format!(
                        "derived donor {name} must record an explicit path and size"
                    )));
                }
                (name.to_string(), sha256.clone(), None)
            }
            LibraryExpectation::File(file) => {
                (file.path.clone(), file.sha256.clone(), Some(file.size))
            }
        };
        if !safe_file_name(&path) {
            return Err(start_error(format!(
                "provider library {name} path {path:?} is unsafe"
            )));
        }
        if !is_sha256(&sha256) {
            return Err(start_error(format!(
                "provider library {name} has a malformed SHA-256"
            )));
        }
        if size == Some(0) {
            return Err(start_error(format!(
                "provider library {name} has zero size"
            )));
        }
        expected.push(ExpectedFile { path, sha256, size });
    }
    Ok(expected)
}

fn validate_manifest(
    manifest: &Manifest,
    major: i32,
    kind: ManifestKind,
    main_sha256: &str,
) -> Result<(), BackendError> {
    if manifest.abi_version != ABI_VERSION {
        return Err(start_error(format!(
            "manifest ABI {} does not match ABI {ABI_VERSION}",
            manifest.abi_version
        )));
    }
    if manifest.capabilities.len() != REQUIRED_CAPABILITIES.len()
        || !manifest
            .capabilities
            .iter()
            .map(String::as_str)
            .eq(REQUIRED_CAPABILITIES)
    {
        return Err(start_error(
            "manifest must advertise fresh_prefill and initial_state only",
        ));
    }
    if manifest.flashinfer.as_deref() != Some(FLASHINFER_VERSION) {
        return Err(start_error(format!(
            "manifest FlashInfer version must be {FLASHINFER_VERSION}"
        )));
    }
    validate_architectures(
        &manifest.architectures,
        major,
        kind,
        main_sha256,
        manifest.objects.as_ref(),
    )?;
    match kind {
        ManifestKind::Producer => {
            if manifest.source_sdk_sha256.is_some() || manifest.source_provider_sha256.is_some() {
                return Err(start_error(
                    "producer manifest must not claim derived-donor source digests",
                ));
            }
        }
        ManifestKind::Derived => {
            let sdk = manifest
                .source_sdk_sha256
                .as_deref()
                .ok_or_else(|| start_error("derived donor lacks source_sdk_sha256"))?;
            let provider = manifest
                .source_provider_sha256
                .as_deref()
                .ok_or_else(|| start_error("derived donor lacks source_provider_sha256"))?;
            // The SDK is the separate libtei_torch.so which depends on this
            // provider. Its provenance digest is not the provider ELF digest.
            if !is_sha256(sdk) {
                return Err(start_error("derived donor source SDK SHA-256 is malformed"));
            }
            if !provider.eq_ignore_ascii_case(main_sha256) {
                return Err(start_error(
                    "derived donor source provider SHA-256 does not match the main library",
                ));
            }
        }
    }
    Ok(())
}

fn validate_library_names<'a>(names: impl Iterator<Item = &'a String>) -> Result<(), BackendError> {
    let names: Vec<&str> = names.map(String::as_str).collect();
    if names.len() != REQUIRED_LIBRARIES.len()
        || REQUIRED_LIBRARIES
            .iter()
            .any(|required| !names.contains(required))
    {
        return Err(start_error(
            "manifest must contain exactly the GDN, TVM-FFI and CuTe libraries",
        ));
    }
    Ok(())
}

fn validate_architectures(
    architectures: &[ArchitectureRecord],
    major: i32,
    kind: ManifestKind,
    main_sha256: &str,
    objects: Option<&BTreeMap<String, String>>,
) -> Result<(), BackendError> {
    let mut seen = [false; 3];
    let mut selected = false;
    for record in architectures {
        let (index, record_major) = architecture_major(&record.architecture).ok_or_else(|| {
            start_error(format!(
                "unknown provider architecture {:?}",
                record.architecture
            ))
        })?;
        let slot = seen
            .get_mut(index)
            .ok_or_else(|| start_error("architecture index overflow"))?;
        if *slot {
            return Err(start_error(format!(
                "duplicate provider architecture {:?}",
                record.architecture
            )));
        }
        *slot = true;
        validate_architecture_exports(record, record_major, kind, main_sha256, objects)?;
        if record_major == major {
            selected = true;
        }
    }
    if !selected {
        return Err(start_error(format!("manifest has no export for SM{major}")));
    }
    Ok(())
}

fn validate_architecture_exports(
    record: &ArchitectureRecord,
    major: i32,
    kind: ManifestKind,
    main_sha256: &str,
    objects: Option<&BTreeMap<String, String>>,
) -> Result<(), BackendError> {
    let expected = [
        (format!("gdn_{major}_fp16"), false),
        (format!("gdn_{major}_bf16"), false),
        (format!("gdn_{major}_fp16_initial"), true),
        (format!("gdn_{major}_bf16_initial"), true),
    ];
    let mut seen = [false; 4];
    for export in &record.exports {
        if !is_sha256(&export.sha256) {
            return Err(start_error(format!(
                "export {:?} has a malformed SHA-256",
                export.symbol
            )));
        }
        let index = expected
            .iter()
            .position(|(symbol, _)| symbol.as_str() == export.symbol.as_str())
            .ok_or_else(|| {
                start_error(format!(
                    "unexpected {:?} export {:?}",
                    record.architecture, export.symbol
                ))
            })?;
        let slot = seen
            .get_mut(index)
            .ok_or_else(|| start_error("export index overflow"))?;
        if *slot {
            return Err(start_error(format!(
                "duplicate {:?} export {:?}",
                record.architecture, export.symbol
            )));
        }
        let (_, initial_state) = expected
            .get(index)
            .ok_or_else(|| start_error("export index overflow"))?;
        if export.initial_state != *initial_state {
            return Err(start_error(format!(
                "export {:?} has the wrong initial-state tag",
                export.symbol
            )));
        }
        match kind {
            ManifestKind::Producer => {
                if export.digest_kind != DigestKind::Object {
                    return Err(start_error(format!(
                        "producer export {:?} must keep its object digest kind",
                        export.symbol
                    )));
                }
                if let Some(objects) = objects {
                    let key = export
                        .object
                        .clone()
                        .unwrap_or_else(|| format!("{}.o", export.symbol));
                    if !safe_file_name(&key) {
                        return Err(start_error(format!(
                            "producer export {:?} object name is unsafe",
                            export.symbol
                        )));
                    }
                    let digest = objects.get(&key).ok_or_else(|| {
                        start_error(format!(
                            "producer export {:?} object {key:?} is absent from objects",
                            export.symbol
                        ))
                    })?;
                    if !digest.eq_ignore_ascii_case(&export.sha256) {
                        return Err(start_error(format!(
                            "producer export {:?} digest differs from its object",
                            export.symbol
                        )));
                    }
                }
            }
            ManifestKind::Derived => {
                if export.digest_kind != DigestKind::ContainingLibrary {
                    return Err(start_error(format!(
                        "derived export {:?} must use the containing-library digest kind",
                        export.symbol
                    )));
                }
                if export.object.is_some() {
                    return Err(start_error(format!(
                        "derived export {:?} must not claim an original object",
                        export.symbol
                    )));
                }
                if !export.sha256.eq_ignore_ascii_case(main_sha256) {
                    return Err(start_error(format!(
                        "derived export {:?} digest does not match the main library",
                        export.symbol
                    )));
                }
            }
        }
        *slot = true;
    }
    if seen.iter().any(|value| !value) {
        return Err(start_error(format!(
            "{} must export fp16/bf16 fresh and initial-state kernels",
            record.architecture
        )));
    }
    Ok(())
}

fn architecture_major(name: &str) -> Option<(usize, i32)> {
    match name {
        "sm_90a" => Some((0, 9)),
        "sm_100f" => Some((1, 10)),
        "sm_120f" => Some((2, 12)),
        _ => None,
    }
}

fn resolve_expected_path(root: &Path, path: &str) -> Result<PathBuf, BackendError> {
    if !safe_file_name(path) {
        return Err(start_error(format!("unsafe provider file path {path:?}")));
    }
    let joined = root.join(path);
    let canonical = joined
        .canonicalize()
        .map_err(|error| start_error(format!("{}: {error}", joined.display())))?;
    if !canonical.starts_with(root) {
        return Err(start_error(format!(
            "provider file {} escapes the bundle directory",
            joined.display()
        )));
    }
    let metadata = fs::metadata(&canonical)
        .map_err(|error| start_error(format!("{}: {error}", canonical.display())))?;
    if !metadata.is_file() {
        return Err(start_error(format!(
            "{} is not a regular file",
            canonical.display()
        )));
    }
    Ok(canonical)
}

fn verify_expected_file(root: &Path, expected: &ExpectedFile) -> Result<PathBuf, BackendError> {
    let path = resolve_expected_path(root, &expected.path)?;
    let metadata =
        fs::metadata(&path).map_err(|error| start_error(format!("{}: {error}", path.display())))?;
    validate_file_size(expected, metadata.len())?;
    let actual = sha256_file(&path)?;
    if !actual.eq_ignore_ascii_case(&expected.sha256) {
        return Err(start_error(format!(
            "{} does not match the declared SHA-256",
            path.display()
        )));
    }
    Ok(path)
}

fn validate_file_size(expected: &ExpectedFile, actual: u64) -> Result<(), BackendError> {
    if actual == 0 {
        return Err(start_error(format!("{} is empty", expected.path)));
    }
    if let Some(size) = expected.size {
        if actual != size {
            return Err(start_error(format!(
                "{} has {actual} bytes, expected {size}",
                expected.path
            )));
        }
    }
    Ok(())
}

fn read_bounded_file(path: &Path, maximum: u64) -> Result<Vec<u8>, BackendError> {
    let metadata =
        fs::metadata(path).map_err(|error| start_error(format!("{}: {error}", path.display())))?;
    if !metadata.is_file() {
        return Err(start_error(format!(
            "{} is not a regular file",
            path.display()
        )));
    }
    if metadata.len() > maximum {
        return Err(start_error(format!(
            "{} is larger than the {maximum}-byte admission limit",
            path.display()
        )));
    }
    let bytes =
        fs::read(path).map_err(|error| start_error(format!("{}: {error}", path.display())))?;
    if u64::try_from(bytes.len()).ok() != Some(metadata.len()) {
        return Err(start_error(format!(
            "{} changed size while being read",
            path.display()
        )));
    }
    Ok(bytes)
}

fn sha256_file(path: &Path) -> Result<String, BackendError> {
    let mut file =
        File::open(path).map_err(|error| start_error(format!("{}: {error}", path.display())))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| start_error(format!("{}: {error}", path.display())))?;
        if read == 0 {
            break;
        }
        let chunk = buffer
            .get(..read)
            .ok_or_else(|| start_error("file read exceeded its buffer"))?;
        hasher.update(chunk);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn safe_file_name(value: &str) -> bool {
    if value.is_empty() || value.len() > 255 || value.as_bytes().contains(&0) {
        return false;
    }
    if value.contains('/') || value.contains('\\') {
        return false;
    }
    let mut components = Path::new(value).components();
    matches!(
        (components.next(), components.next()),
        (Some(Component::Normal(_)), None)
    )
}

fn workspace_bytes(major: i32, sm_count: i32) -> Result<i64, BackendError> {
    let factor = match major {
        9 | 12 => 128i64,
        10 => 512i64,
        _ => {
            return Err(start_error(format!(
                "SM{major} has no FlashInfer GDN AOT export"
            )))
        }
    };
    if sm_count <= 0 {
        return Err(start_error("SM count must be positive"));
    }
    i64::from(sm_count)
        .checked_mul(factor)
        .ok_or_else(|| start_error("GDN workspace byte extent overflows"))
}

fn validate_launch(
    major: i32,
    sm_count: i32,
    provider_workspace_bytes: i64,
    launch: &GdnAotLaunch,
) -> Result<ValidatedLaunch, BackendError> {
    let expected_workspace = workspace_bytes(major, sm_count)?;
    if expected_workspace != provider_workspace_bytes {
        return Err(infer_error("provider workspace plan is inconsistent"));
    }
    let expected_workspace = u64::try_from(expected_workspace)
        .map_err(|_| infer_error("workspace byte extent exceeds u64"))?;
    if launch.workspace_bytes != expected_workspace {
        return Err(infer_error(format!(
            "workspace has {} bytes, SM{major} requires {expected_workspace}",
            launch.workspace_bytes
        )));
    }
    if launch.stream == 0 {
        return Err(infer_error("stream must be nonzero"));
    }
    let tokens = positive_value(launch.tokens, i32::MAX as u64, "tokens")?;
    let batch = positive_value(launch.batch, i32::MAX as u64, "batch")?;
    let key_heads = positive_value(launch.key_heads, MAX_VALUE_HEADS, "key heads")?;
    let value_heads = positive_value(launch.value_heads, MAX_VALUE_HEADS, "value heads")?;
    if value_heads < key_heads || value_heads.checked_rem(key_heads) != Some(0) {
        return Err(infer_error(
            "value heads must be a positive multiple of key heads",
        ));
    }
    if batch > tokens {
        return Err(infer_error("batch exceeds packed tokens"));
    }
    let tokens_u64 = u64::try_from(tokens).map_err(|_| infer_error("token overflow"))?;
    let batch_u64 = u64::try_from(batch).map_err(|_| infer_error("batch overflow"))?;
    let key_heads_u64 = u64::try_from(key_heads).map_err(|_| infer_error("key-head overflow"))?;
    let value_heads_u64 =
        u64::try_from(value_heads).map_err(|_| infer_error("value-head overflow"))?;

    let q_bytes = byte_extent(extent(&[tokens_u64, key_heads_u64, HEAD_DIM], "Q")?, 2, "Q")?;
    let k_bytes = q_bytes;
    let v_bytes = byte_extent(
        extent(&[tokens_u64, value_heads_u64, HEAD_DIM], "V")?,
        2,
        "V",
    )?;
    let gate_bytes = byte_extent(extent(&[tokens_u64, value_heads_u64], "gates")?, 4, "gates")?;
    let state_elements = extent(&[batch_u64, value_heads_u64, HEAD_DIM, HEAD_DIM], "state")?;
    let state_bytes = byte_extent(state_elements, 4, "state")?;
    let cumulative_elements = batch_u64
        .checked_add(1)
        .ok_or_else(|| infer_error("cumulative extent overflows"))?;
    let cumulative_bytes = byte_extent(cumulative_elements, 4, "cumulative")?;

    require_alignment(launch.q, DATA_ALIGNMENT, "Q")?;
    require_alignment(launch.k, DATA_ALIGNMENT, "K")?;
    require_alignment(launch.v, DATA_ALIGNMENT, "V")?;
    require_alignment(launch.alpha, DATA_ALIGNMENT, "alpha")?;
    require_alignment(launch.beta, DATA_ALIGNMENT, "beta")?;
    require_alignment(launch.output, DATA_ALIGNMENT, "output")?;
    require_alignment(launch.cumulative, CUMULATIVE_ALIGNMENT, "cumulative")?;
    require_alignment(launch.state, DATA_ALIGNMENT, "state")?;
    require_alignment(launch.workspace, WORKSPACE_ALIGNMENT, "workspace")?;
    if launch.initial != 0 {
        require_alignment(launch.initial, DATA_ALIGNMENT, "initial")?;
    }

    let ranges = [
        checked_range(launch.q, q_bytes, "Q")?,
        checked_range(launch.k, k_bytes, "K")?,
        checked_range(launch.v, v_bytes, "V")?,
        checked_range(launch.alpha, gate_bytes, "alpha")?,
        checked_range(launch.beta, gate_bytes, "beta")?,
        checked_range(launch.output, v_bytes, "output")?,
        checked_range(launch.cumulative, cumulative_bytes, "cumulative")?,
        checked_range(launch.state, state_bytes, "state")?,
        checked_range(launch.workspace, expected_workspace, "workspace")?,
    ];
    for (index, range) in ranges.iter().enumerate() {
        for other in ranges.iter().skip(index + 1) {
            if ranges_overlap(*range, *other) {
                return Err(infer_error("GDN device buffer extents overlap"));
            }
        }
    }
    if launch.initial != 0 {
        let initial = checked_range(launch.initial, state_bytes, "initial")?;
        if ranges.iter().any(|range| ranges_overlap(*range, initial)) {
            return Err(infer_error("initial state overlaps another buffer"));
        }
    }

    Ok(ValidatedLaunch {
        stream: pointer(launch.stream),
        tokens,
        batch,
        key_heads,
        value_heads,
        q: pointer(launch.q),
        k: pointer(launch.k),
        v: pointer(launch.v),
        alpha: pointer(launch.alpha),
        beta: pointer(launch.beta),
        output: pointer(launch.output),
        cumulative: pointer(launch.cumulative),
        state: pointer(launch.state),
        workspace: pointer(launch.workspace),
        workspace_bytes: i64::try_from(expected_workspace)
            .map_err(|_| infer_error("workspace byte extent exceeds int64"))?,
        initial: if launch.initial == 0 {
            std::ptr::null_mut()
        } else {
            pointer(launch.initial)
        },
    })
}

fn positive_value(value: u64, maximum: u64, name: &str) -> Result<i64, BackendError> {
    if value == 0 || value > maximum {
        return Err(infer_error(format!("{name} is outside the native range")));
    }
    i64::try_from(value).map_err(|_| infer_error(format!("{name} exceeds int64")))
}

fn extent(parts: &[u64], name: &str) -> Result<u64, BackendError> {
    parts
        .iter()
        .try_fold(1u64, |product, value| product.checked_mul(*value))
        .ok_or_else(|| infer_error(format!("{name} element extent overflows")))
}

fn byte_extent(elements: u64, width: u64, name: &str) -> Result<u64, BackendError> {
    elements
        .checked_mul(width)
        .filter(|bytes| *bytes <= i64::MAX as u64)
        .ok_or_else(|| infer_error(format!("{name} byte extent overflows")))
}

fn checked_range(start: u64, bytes: u64, name: &str) -> Result<(u64, u64), BackendError> {
    let end = start
        .checked_add(bytes)
        .ok_or_else(|| infer_error(format!("{name} address range overflows")))?;
    Ok((start, end))
}

fn ranges_overlap(left: (u64, u64), right: (u64, u64)) -> bool {
    left.0 < right.1 && right.0 < left.1
}

fn require_alignment(value: u64, alignment: u64, name: &str) -> Result<(), BackendError> {
    if value == 0 {
        return Err(infer_error(format!("{name} pointer is null")));
    }
    if !value.is_multiple_of(alignment) {
        return Err(infer_error(format!(
            "{name} pointer is not {alignment}-byte aligned"
        )));
    }
    Ok(())
}

fn pointer(value: u64) -> *mut c_void {
    value as usize as *mut c_void
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Map};
    use std::sync::atomic::{AtomicUsize, Ordering};

    const BRIDGE_HEADER: &str = include_str!("../../libtorch/cpp/vendor/infinity_gdn/bridge.h");

    static TEMP_COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn facts(major: i32) -> DeviceFacts {
        DeviceFacts {
            ordinal: 0,
            major,
            sm_count: 4,
        }
    }

    fn digest(label: &str) -> String {
        sha256_hex(label.as_bytes())
    }

    fn temp_root(label: &str) -> PathBuf {
        let counter = TEMP_COUNTER.fetch_add(1, Ordering::SeqCst);
        let path = std::env::temp_dir().join(format!(
            "oxide-gdn-aot-{}-{label}-{counter}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).expect("create synthetic bundle");
        path
    }

    fn write_manifest(root: &Path, value: &Value) {
        fs::write(
            root.join(MANIFEST_FILE),
            serde_json::to_vec(value).expect("serialize manifest"),
        )
        .expect("write synthetic manifest");
    }

    fn libraries(root: &Path, records: bool) -> Value {
        let mut map = Map::new();
        for name in REQUIRED_LIBRARIES {
            let payload = format!("{name}-synthetic-payload");
            fs::write(root.join(name), payload.as_bytes()).expect("write synthetic library");
            let sha256 = sha256_hex(payload.as_bytes());
            map.insert(
                name.to_string(),
                if records {
                    json!({"path": name, "sha256": sha256, "size": payload.len()})
                } else {
                    json!(sha256)
                },
            );
        }
        Value::Object(map)
    }

    fn export(symbol: &str, initial_state: bool, sha256: &str) -> Value {
        json!({"symbol": symbol, "initial_state": initial_state, "sha256": sha256})
    }

    fn producer_architecture(family: i32, architecture: &str) -> Value {
        let symbols = [
            (format!("gdn_{family}_fp16"), false),
            (format!("gdn_{family}_bf16"), false),
            (format!("gdn_{family}_fp16_initial"), true),
            (format!("gdn_{family}_bf16_initial"), true),
        ];
        let exports: Vec<Value> = symbols
            .iter()
            .map(|(symbol, initial)| export(symbol, *initial, &digest(&format!("{symbol}.o"))))
            .collect();
        json!({"architecture": architecture, "exports": exports})
    }

    fn producer_objects() -> Value {
        let mut map = Map::new();
        for family in [9, 10, 12] {
            for label in ["fp16", "bf16", "fp16_initial", "bf16_initial"] {
                let name = format!("gdn_{family}_{label}.o");
                map.insert(name.clone(), json!(digest(&name)));
            }
        }
        Value::Object(map)
    }

    fn producer_manifest(libraries: Value) -> Value {
        json!({
            "abi_version": 2,
            "capabilities": ["fresh_prefill", "initial_state"],
            "flashinfer": "0.7.2",
            "flashinfer_source_commit": "d84335d8c54e0c8f9f7e22faa15c7e34a58abf94",
            "compile_only": true,
            "executed_during_export": false,
            "architectures": [
                producer_architecture(9, "sm_90a"),
                producer_architecture(10, "sm_100f"),
                producer_architecture(12, "sm_120f"),
            ],
            "objects": producer_objects(),
            "libraries": libraries,
            "licenses": {"flashinfer-python/LICENSE": digest("license")},
            "provenance": {
                "source": "/vendor/infinity_gdn/provenance.json",
                "destination": "provenance.json",
                "sha256": digest("provenance"),
            },
            "source_pins": {
                "infinity_vendor": {"bridge.h": digest("bridge")},
                "flashinfer_version": "0.7.2",
            },
            "python_metadata": {"torch": null},
            "observer": {
                "so": "libinfinity_public_pdl_observer.so",
                "sha256": digest("observer"),
                "freeze_sha256": digest("freeze"),
                "families": [],
            },
            "bridge": {"library": MAIN_LIBRARY, "library_sha256": "not-the-package-digest"},
            "gpu_visible_devices": "0",
            "runtime_qualified": false,
        })
    }

    fn derived_manifest(libraries: Value) -> Value {
        let main_sha256 = libraries[MAIN_LIBRARY]["sha256"]
            .as_str()
            .expect("main library digest");
        let mut exports = Vec::new();
        for (symbol, initial) in [
            ("gdn_9_fp16", false),
            ("gdn_9_bf16", false),
            ("gdn_9_fp16_initial", true),
            ("gdn_9_bf16_initial", true),
        ] {
            let mut record = export(symbol, initial, main_sha256);
            record["digest_kind"] = json!("containing-library");
            exports.push(record);
        }
        json!({
            "kind": DERIVED_KIND,
            "abi_version": 2,
            "capabilities": ["fresh_prefill", "initial_state"],
            "flashinfer": "0.7.2",
            "source_sdk_sha256": digest("separate-native-sdk"),
            "source_provider_sha256": main_sha256,
            "architectures": [{"architecture": "sm_90a", "exports": exports}],
            "libraries": libraries,
        })
    }

    fn producer_bundle(label: &str) -> (PathBuf, Value) {
        let root = temp_root(label);
        let manifest = producer_manifest(libraries(&root, false));
        write_manifest(&root, &manifest);
        (root, manifest)
    }

    fn derived_bundle(label: &str) -> (PathBuf, Value) {
        let root = temp_root(label);
        let manifest = derived_manifest(libraries(&root, true));
        write_manifest(&root, &manifest);
        (root, manifest)
    }

    fn rejects(root: &Path, manifest: &Value, edit: impl FnOnce(&mut Value)) {
        let mut value = manifest.clone();
        edit(&mut value);
        write_manifest(root, &value);
        assert!(GdnAotPlan::load(root, facts(9)).is_err());
    }

    fn valid_launch(major: i32, sm_count: i32, value_heads: u64) -> GdnAotLaunch {
        let workspace = workspace_bytes(major, sm_count).expect("workspace") as u64;
        GdnAotLaunch {
            stream: 1,
            tokens: 8,
            batch: 2,
            key_heads: 16,
            value_heads,
            q: 0x1000000,
            k: 0x2000000,
            v: 0x3000000,
            alpha: 0x4000000,
            beta: 0x5000000,
            output: 0x6000000,
            cumulative: 0x7000000,
            state: 0x8000000,
            workspace: 0x9000000,
            workspace_bytes: workspace,
            initial: 0xa000000,
        }
    }

    #[test]
    fn source_abi2_header_has_the_exact_twenty_arguments_in_order() {
        let after = BRIDGE_HEADER
            .split("int infinity_gdn_forward_v2(")
            .nth(1)
            .expect("bridge.h must declare infinity_gdn_forward_v2");
        let parameters = after
            .split(");")
            .next()
            .expect("forward declaration must terminate");
        let actual: Vec<&str> = parameters
            .split(',')
            .map(str::trim)
            .filter(|parameter| !parameter.is_empty())
            .collect();
        assert_eq!(
            actual,
            [
                "int dtype",
                "int major",
                "int device",
                "void* stream",
                "int64_t tokens",
                "int64_t batch",
                "int64_t kh",
                "int64_t vh",
                "int sm_count",
                "void* q",
                "void* k",
                "void* v",
                "void* alpha",
                "void* beta",
                "void* output",
                "void* cumulative",
                "void* state",
                "void* workspace",
                "int64_t workspace_bytes",
                "void* initial",
            ]
        );
    }

    #[test]
    fn actual_producer_manifest_shape_is_admitted() {
        let (root, manifest) = producer_bundle("producer");
        let plan = GdnAotPlan::load(&root, facts(9)).expect("actual producer shape");
        assert_eq!(plan.major, 9);
        assert_eq!(plan.workspace_bytes(), 512);
        assert_eq!(plan.library.file_name(), Some(MAIN_LIBRARY.as_ref()));
        let mut tagged = manifest.clone();
        tagged["kind"] = json!(PRODUCER_KIND);
        write_manifest(&root, &tagged);
        assert!(GdnAotPlan::load(&root, facts(9)).is_ok());

        rejects(&root, &manifest, |value| value["abi_version"] = json!(1));
        rejects(&root, &manifest, |value| {
            value["capabilities"] = json!(["fresh_prefill"])
        });
        rejects(&root, &manifest, |value| {
            value["flashinfer"] = json!("0.7.1")
        });
        rejects(&root, &manifest, |value| {
            value["architectures"][0]["architecture"] = json!("sm_80a")
        });
        rejects(&root, &manifest, |value| {
            value["architectures"][0]["exports"][0]["initial_state"] = json!(true)
        });
        rejects(&root, &manifest, |value| {
            value["architectures"][0]["exports"]
                .as_array_mut()
                .expect("exports")
                .pop();
        });
        rejects(&root, &manifest, |value| {
            value["architectures"][0]["exports"][0]["digest_kind"] = json!("containing-library")
        });
        rejects(&root, &manifest, |value| {
            value["objects"]["gdn_9_fp16.o"] = json!(digest("different"))
        });
        rejects(&root, &manifest, |value| value["kind"] = json!("unknown"));
        rejects(&root, &manifest, |value| {
            value["abi_versoin"] = value["abi_version"].clone();
            value.as_object_mut().expect("object").remove("abi_version");
        });
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn derived_donor_needs_no_original_producer_manifest() {
        let (root, manifest) = derived_bundle("derived");
        assert!(!root.join("derived-donor.json").exists());
        let plan = GdnAotPlan::load(&root, facts(9)).expect("derived donor without producer");
        assert_eq!(plan.workspace_bytes(), 512);

        rejects(&root, &manifest, |value| {
            value["source_sdk_sha256"] = json!("not-a-digest")
        });
        rejects(&root, &manifest, |value| {
            value["source_provider_sha256"] = json!(digest("different"))
        });
        rejects(&root, &manifest, |value| {
            value
                .as_object_mut()
                .expect("object")
                .remove("source_provider_sha256");
        });
        rejects(&root, &manifest, |value| {
            value["architectures"][0]["exports"][0]["digest_kind"] = json!("object")
        });
        rejects(&root, &manifest, |value| {
            value["architectures"][0]["exports"][0]["object"] = json!("gdn_9_fp16.o")
        });
        rejects(&root, &manifest, |value| {
            value["architectures"][0]["exports"][0]["sha256"] = json!(digest("different"))
        });
        rejects(&root, &manifest, |value| {
            value["libraries"][MAIN_LIBRARY] = json!(digest("main"))
        });
        rejects(&root, &manifest, |value| {
            value["libraries"][MAIN_LIBRARY]["path"] = json!("../libinfinitygdn.so")
        });
        rejects(&root, &manifest, |value| value["abi_version"] = json!(1));
        rejects(&root, &manifest, |value| {
            value["capabilities"] = json!(["fresh_prefill"])
        });
        rejects(&root, &manifest, |value| {
            value["architectures"][0]["architecture"] = json!("sm_100f")
        });
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn library_digests_sizes_and_regular_files_are_enforced() {
        let (root, manifest) = producer_bundle("library-files");
        rejects(&root, &manifest, |value| {
            value["libraries"][MAIN_LIBRARY] = json!(digest("different"))
        });
        rejects(&root, &manifest, |value| {
            value["libraries"]["libtvm_ffi.so"] = json!("short")
        });
        rejects(&root, &manifest, |value| {
            value["libraries"]["libextra.so"] = json!(digest("extra"))
        });
        rejects(&root, &manifest, |value| {
            value["libraries"][MAIN_LIBRARY] = json!({
                "path": MAIN_LIBRARY,
                "sha256": digest("libinfinitygdn.so-synthetic-payload"),
                "size": 1,
            })
        });

        write_manifest(&root, &manifest);
        fs::write(root.join(MAIN_LIBRARY), b"mutated").expect("mutate library");
        assert!(GdnAotPlan::load(&root, facts(9)).is_err());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn safe_paths_reject_traversal_absolute_and_library_names() {
        for bad in [
            "",
            ".",
            "..",
            "../manifest.json",
            "a/b",
            "/tmp/x",
            "x\\y",
            "a\0b",
        ] {
            assert!(!safe_file_name(bad), "{bad:?} must be rejected");
        }
        for good in [MANIFEST_FILE, MAIN_LIBRARY, "sm_90a"] {
            assert!(safe_file_name(good), "{good:?} must be accepted");
        }
    }

    #[test]
    fn workspace_is_selected_by_major_and_overflow_is_rejected() {
        assert_eq!(workspace_bytes(9, 4).expect("sm90"), 512);
        assert_eq!(workspace_bytes(10, 4).expect("sm100"), 2048);
        assert_eq!(workspace_bytes(12, 4).expect("sm120"), 512);
        assert!(workspace_bytes(8, 4).is_err());
        assert!(workspace_bytes(9, 0).is_err());
    }

    #[test]
    fn gva_shapes_vh32_and_vh48_are_pure_and_extents_must_not_overlap() {
        for value_heads in [32, 48] {
            let launch = valid_launch(10, 4, value_heads);
            assert!(validate_launch(10, 4, 2048, &launch).is_ok());
            let fresh = GdnAotLaunch {
                initial: 0,
                ..launch
            };
            assert!(validate_launch(10, 4, 2048, &fresh).is_ok());
        }

        let mut wrong_workspace = valid_launch(10, 4, 32);
        wrong_workspace.workspace_bytes = 1024;
        assert!(validate_launch(10, 4, 2048, &wrong_workspace).is_err());

        let mut wrong_major_workspace = valid_launch(9, 4, 32);
        assert!(validate_launch(10, 4, 2048, &wrong_major_workspace).is_err());
        wrong_major_workspace.workspace_bytes = 2048;
        assert!(validate_launch(10, 4, 2048, &wrong_major_workspace).is_ok());

        for field in 0u8..9 {
            let mut misaligned = valid_launch(9, 4, 32);
            match field {
                0 => misaligned.q += 1,
                1 => misaligned.k += 1,
                2 => misaligned.v += 1,
                3 => misaligned.alpha += 1,
                4 => misaligned.beta += 1,
                5 => misaligned.output += 1,
                6 => misaligned.state += 1,
                7 => misaligned.workspace += 1,
                _ => misaligned.initial += 1,
            }
            assert!(
                validate_launch(9, 4, 512, &misaligned).is_err(),
                "misaligned field must be rejected"
            );
        }
        let mut misaligned_cu = valid_launch(10, 4, 32);
        misaligned_cu.cumulative += 1;
        assert!(validate_launch(10, 4, 2048, &misaligned_cu).is_err());

        let mut aliased = valid_launch(10, 4, 32);
        aliased.output = aliased.q;
        assert!(validate_launch(10, 4, 2048, &aliased).is_err());
        let mut partial_overlap = valid_launch(10, 4, 32);
        partial_overlap.v = partial_overlap.q + 0x10;
        assert!(validate_launch(10, 4, 2048, &partial_overlap).is_err());
        let mut aliased_initial = valid_launch(10, 4, 32);
        aliased_initial.initial = aliased_initial.state + 0x10;
        assert!(validate_launch(10, 4, 2048, &aliased_initial).is_err());
        let mut adjacent = valid_launch(10, 4, 32);
        adjacent.v = adjacent.q + 8 * 16 * HEAD_DIM * 2;
        assert!(validate_launch(10, 4, 2048, &adjacent).is_ok());
    }

    #[test]
    fn invalid_geometry_and_overflow_are_rejected_before_launch() {
        let base = valid_launch(9, 4, 32);
        for (field, value) in [
            ("tokens", 0u64),
            ("batch", 0),
            ("key_heads", 0),
            ("value_heads", 0),
            ("tokens", u64::from(u32::MAX) + 1),
        ] {
            let mut launch = base;
            match field {
                "tokens" => launch.tokens = value,
                "batch" => launch.batch = value,
                "key_heads" => launch.key_heads = value,
                "value_heads" => launch.value_heads = value,
                _ => unreachable!(),
            }
            assert!(
                validate_launch(9, 4, 512, &launch).is_err(),
                "{field}={value} must be rejected"
            );
        }
        let mut not_multiple = base;
        not_multiple.value_heads = 31;
        assert!(validate_launch(9, 4, 512, &not_multiple).is_err());
        let mut too_wide = base;
        too_wide.key_heads = 13;
        too_wide.value_heads = 130;
        assert!(validate_launch(9, 4, 512, &too_wide).is_err());
        let mut batch_too_large = base;
        batch_too_large.tokens = 1;
        batch_too_large.batch = 2;
        assert!(validate_launch(9, 4, 512, &batch_too_large).is_err());
        let mut zero_stream = base;
        zero_stream.stream = 0;
        assert!(validate_launch(9, 4, 512, &zero_stream).is_err());
        let mut overflow = base;
        overflow.tokens = u64::MAX;
        assert!(validate_launch(9, 4, 512, &overflow).is_err());
        let mut overflow_state = base;
        overflow_state.batch = u64::from(u32::MAX);
        overflow_state.tokens = u64::from(u32::MAX);
        assert!(validate_launch(9, 4, 512, &overflow_state).is_err());
    }
}
