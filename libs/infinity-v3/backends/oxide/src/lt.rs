// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Prepared cuBLASLt fused projection plans.
//!
//! A plan owns a complete opaque algorithm selected by the current GPU for one
//! `(K, N, epilogue)` tuple at a seed row count of 128. For rows above 32,
//! logical row changes only resize the B/D descriptors and call
//! `cublasLtMatmulAlgoCheck`; they never run the heuristic again or replace the
//! algorithm. Encoder plans additionally cache one heuristic-selected algorithm
//! per actual row count 1 through 32, prepared under the same 64 KiB provider
//! workspace. The provider retains the loaded
//! library, context, stream, and cuBLASLt handle, while each plan retains the API
//! vtable and drains its stream before destroying host descriptors. One fixed
//! provider workspace is reused in stream order, including when a zero-workspace
//! seed algorithm requires a small auxiliary buffer at a different row count.
//!
//! The GTE down projection uses a separate native-default plan. Its provider
//! owns a 32 MiB zeroed backing allocation, keeps the audited 1 MiB Lt
//! workspace preference, and caches one heuristic result per actual
//! `(M, K, N, epilogue, alignment)` key in at most 64 entries. Repeated keys
//! perform no heuristic query or allocation.
//!
//! ABI provenance (CUDA 13.1 primary header):
//! `/usr/local/cuda-13.1/include/cublasLt.h`; enum values, descriptor sizes, and
//! `cublasLtMatmulHeuristicResult_t` offsets were checked with the compiled probe
//! `/tmp/infinity-oxide-next-20261031/lt-prototype/abi_probe.cpp` (output in
//! `abi_probe.json` beside it). The probe reports a 96-byte, 8-byte-aligned
//! heuristic result; 64-byte/8-byte `cublasLtMatmulAlgo_t`; and offsets 0, 64,
//! 72, 76, 80 for algo, workspaceSize, state, wavesCount, and reserved.
//!
//! Ordinary plans admit only BF16 and F16. The E4M3FN tensor storage marker
//! cannot reach an ordinary plan; F32 is also rejected rather than silently
//! selecting an unqualified method.

mod fp8;
pub(crate) use fp8::{Fp8Geometry, Fp8Plan, Fp8Recipe};

use crate::{
    math::{LinearPlan, MathError},
    tensor::{Element, Layout, ScalarFormat, TensorMut, TensorRef},
};
use cuda_core::{Context, DeviceBuffer, Stream};
use infinity_v3_backend_core::BackendError;
use libloading::Library;
#[cfg(test)]
use std::cell::Cell;
use std::{
    ffi::c_void,
    marker::PhantomData,
    mem::{align_of, size_of, ManuallyDrop},
    ptr,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

/// A rejected shape has not submitted GPU work and leaves every changed
/// descriptor restored or covered by an invalidated aggregate row cache.
#[derive(Debug)]
pub(crate) enum ShapeError {
    Rejected(BackendError),
    Fatal(BackendError),
}
impl ShapeError {
    pub(crate) fn into_backend(self) -> BackendError {
        match self {
            Self::Rejected(error) | Self::Fatal(error) => error,
        }
    }
}
impl From<BackendError> for ShapeError {
    fn from(error: BackendError) -> Self {
        Self::Fatal(error)
    }
}

#[cfg(test)]
#[derive(Clone, Copy)]
struct ShapeRejection {
    rows: i32,
    successful_checks_left: usize,
}

#[cfg(test)]
fn inject_shape_rejection(hook: &Cell<Option<ShapeRejection>>, rows: i32) -> bool {
    let Some(mut rejection) = hook.get().filter(|rejection| rejection.rows == rows) else {
        return false;
    };
    if rejection.successful_checks_left == 0 {
        hook.set(None);
        return true;
    }
    rejection.successful_checks_left -= 1;
    hook.set(Some(rejection));
    false
}

type Handle = *mut c_void;
type Descriptor = *mut c_void;
type Create = unsafe extern "C" fn(*mut Handle) -> i32;
type Destroy = unsafe extern "C" fn(Descriptor) -> i32;
type DescCreate = unsafe extern "C" fn(*mut Descriptor, i32, i32) -> i32;
type LayoutCreate = unsafe extern "C" fn(*mut Descriptor, i32, u64, u64, i64) -> i32;
type SetAttribute = unsafe extern "C" fn(Descriptor, i32, *const c_void, usize) -> i32;
type Heuristic = unsafe extern "C" fn(
    Handle,
    Descriptor,
    Descriptor,
    Descriptor,
    Descriptor,
    Descriptor,
    Descriptor,
    i32,
    *mut HeuristicResult,
    *mut i32,
) -> i32;
type AlgoCheck = unsafe extern "C" fn(
    Handle,
    Descriptor,
    Descriptor,
    Descriptor,
    Descriptor,
    Descriptor,
    *const Algo,
    *mut HeuristicResult,
) -> i32;
type Matmul = unsafe extern "C" fn(
    Handle,
    Descriptor,
    *const f32,
    *const c_void,
    Descriptor,
    *const c_void,
    Descriptor,
    *const f32,
    *const c_void,
    Descriptor,
    *mut c_void,
    Descriptor,
    *const Algo,
    *mut c_void,
    usize,
    *mut c_void,
) -> i32;

const STATUS_SUCCESS: i32 = 0;
const STATUS_NOT_SUPPORTED: i32 = 15;
const CUBLAS_OP_T: i32 = 1;
const CUBLAS_COMPUTE_32F: i32 = 68;
const CUDA_R_32F: i32 = 0;
const CUDA_R_16F: i32 = 2;
const CUDA_R_16BF: i32 = 14;
const MATRIX_LAYOUT_COLS: i32 = 3;
const MATMUL_DESC_POINTER_MODE: i32 = 2;
const MATMUL_DESC_TRANSA: i32 = 3;
const MATMUL_DESC_TRANSB: i32 = 4;
const MATMUL_DESC_EPILOGUE: i32 = 7;
const MATMUL_DESC_BIAS_POINTER: i32 = 8;
const MATMUL_DESC_BIAS_DATA_TYPE: i32 = 26;
const PREFERENCE_MAX_WORKSPACE_BYTES: i32 = 1;
const PREFERENCE_MIN_ALIGNMENT_A_BYTES: i32 = 5;
const PREFERENCE_MIN_ALIGNMENT_B_BYTES: i32 = 6;
const PREFERENCE_MIN_ALIGNMENT_C_BYTES: i32 = 7;
const PREFERENCE_MIN_ALIGNMENT_D_BYTES: i32 = 8;
const EPILOGUE_BIAS: i32 = 4;
const EPILOGUE_GELU_BIAS: i32 = 36;

const SEED_ROWS: usize = 128;
const SEED_WORKSPACE_BYTES: usize = 0;
const SMALL_ROWS: usize = 32;
const WORKSPACE_BYTES: usize = 64 * 1024;
pub(crate) const NATIVE_DEFAULT_WORKSPACE_BYTES: usize = 32 * 1024 * 1024;
pub(crate) const NATIVE_DEFAULT_SEED_ROWS: usize = SEED_ROWS;
const NATIVE_DEFAULT_WORKSPACE_PREFERENCE_BYTES: usize = 1024 * 1024;
const NATIVE_DEFAULT_CACHE_ENTRIES: usize = 64;
const HEURISTIC_CHOICES: usize = 8;
// cuBLASLt never dereferences the bias pointer during heuristic selection, but
// uses its value to filter algorithms by expected alignment. A non-null,
// 256-byte-aligned seed matches normal device allocations; every enqueue
// replaces it with the caller's real bias pointer before the matmul.
const SEED_BIAS_POINTER: *const c_void = 0x1000usize as *const c_void;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Algo {
    data: [u64; 8],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct HeuristicResult {
    algo: Algo,
    workspace_size: usize,
    state: i32,
    waves_count: f32,
    reserved: [i32; 4],
}

const _: () = {
    assert!(size_of::<Algo>() == 64);
    assert!(align_of::<Algo>() == 8);
    assert!(size_of::<HeuristicResult>() == 96);
    assert!(align_of::<HeuristicResult>() == 8);
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PreparedAlgorithm {
    algo: Algo,
    workspace: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct NativeDefaultAlignment {
    a: u32,
    b: u32,
    c: u32,
    d: u32,
}

const NATIVE_DEFAULT_ALIGNMENT: NativeDefaultAlignment = NativeDefaultAlignment {
    a: 256,
    b: 256,
    c: 256,
    d: 256,
};

#[derive(Clone, Copy)]
struct NativeDefaultSnapshot {
    rows: i32,
    algo: Algo,
    workspace: usize,
    bias_pointer: *const c_void,
    alignment: NativeDefaultAlignment,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct NativeDefaultCacheKey {
    rows: i32,
    input: i32,
    width: i32,
    x_stride: i32,
    weight_stride: i32,
    output_stride: i32,
    element_type: i32,
    epilogue: Epilogue,
    workspace_preference: usize,
    alignment: NativeDefaultAlignment,
}

#[derive(Clone, Copy, Debug)]
struct NativeDefaultCacheEntry {
    key: NativeDefaultCacheKey,
    prepared: PreparedAlgorithm,
}

struct NativeDefaultCache {
    entries: [Option<NativeDefaultCacheEntry>; NATIVE_DEFAULT_CACHE_ENTRIES],
    next: usize,
    #[cfg(test)]
    queries: usize,
}

impl Default for NativeDefaultCache {
    fn default() -> Self {
        Self {
            entries: [None; NATIVE_DEFAULT_CACHE_ENTRIES],
            next: 0,
            #[cfg(test)]
            queries: 0,
        }
    }
}

impl NativeDefaultCache {
    fn get(&self, key: &NativeDefaultCacheKey) -> Option<PreparedAlgorithm> {
        self.entries
            .iter()
            .flatten()
            .find(|entry| entry.key == *key)
            .map(|entry| entry.prepared)
    }

    fn insert(&mut self, key: NativeDefaultCacheKey, prepared: PreparedAlgorithm) {
        let index = self
            .entries
            .iter()
            .position(|entry| entry.as_ref().is_some_and(|entry| entry.key == key))
            .unwrap_or_else(|| {
                let index = self.next;
                self.next = (self.next + 1) % NATIVE_DEFAULT_CACHE_ENTRIES;
                index
            });
        self.entries[index] = Some(NativeDefaultCacheEntry { key, prepared });
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries.iter().flatten().count()
    }
}

/// Request-scoped cold results that are not visible to later requests until
/// their corresponding matmul launch succeeds. The store is fixed-size and
/// evicts like the persistent cache, so a full store never rejects a legal
/// shape; each invocation already owns the prepared `Copy` data it needs.
pub(crate) struct NativeDefaultPending {
    entries: [Option<NativeDefaultCacheEntry>; NATIVE_DEFAULT_CACHE_ENTRIES],
    next: usize,
}

impl Default for NativeDefaultPending {
    fn default() -> Self {
        Self {
            entries: [None; NATIVE_DEFAULT_CACHE_ENTRIES],
            next: 0,
        }
    }
}

impl NativeDefaultPending {
    fn get(&self, key: &NativeDefaultCacheKey) -> Option<PreparedAlgorithm> {
        self.entries
            .iter()
            .flatten()
            .find(|entry| entry.key == *key)
            .map(|entry| entry.prepared)
    }

    fn insert(&mut self, key: NativeDefaultCacheKey, prepared: PreparedAlgorithm) {
        let index = self
            .entries
            .iter()
            .position(|entry| entry.as_ref().is_some_and(|entry| entry.key == key))
            .unwrap_or_else(|| {
                let index = self.next;
                self.next = (self.next + 1) % NATIVE_DEFAULT_CACHE_ENTRIES;
                index
            });
        self.entries[index] = Some(NativeDefaultCacheEntry { key, prepared });
    }

    pub(crate) fn clear(&mut self) {
        self.entries = [None; NATIVE_DEFAULT_CACHE_ENTRIES];
        self.next = 0;
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries.iter().flatten().count()
    }
}

/// The bounded metadata selected for one already-validated down invocation.
/// It is produced before any request kernel and consumed by the launch site.
#[derive(Clone, Copy, Debug)]
pub(crate) struct NativeDefaultPrepared {
    rows: i32,
    key: NativeDefaultCacheKey,
    prepared: PreparedAlgorithm,
    bias_pointer: u64,
    alignment: NativeDefaultAlignment,
    x_pointer: u64,
    weight_pointer: u64,
    output_pointer: u64,
    pending: bool,
}

fn get_alignment(address: u64) -> u32 {
    let mut alignment = 256u32;
    while alignment > 1 && !address.is_multiple_of(u64::from(alignment)) {
        alignment /= 2;
    }
    alignment
}

fn prepared_for_rows(
    base: PreparedAlgorithm,
    small: &[Option<PreparedAlgorithm>; SMALL_ROWS],
    small_prepared: bool,
    rows: usize,
) -> Result<PreparedAlgorithm, BackendError> {
    let Some(index) = rows.checked_sub(1) else {
        return Ok(base);
    };
    let Some(prepared) = small.get(index) else {
        return Ok(base);
    };
    match *prepared {
        Some(prepared) => Ok(prepared),
        None if small_prepared => Err(super::infer(format!(
            "qualified small-row algorithm is missing for M={rows}"
        ))),
        None => Ok(base),
    }
}

fn check_algorithm(
    algo: Algo,
    check: impl FnOnce(&Algo) -> Result<usize, ShapeError>,
) -> Result<PreparedAlgorithm, ShapeError> {
    let workspace = check(&algo)?;
    Ok(PreparedAlgorithm { algo, workspace })
}

fn restore_prepared(previous: PreparedAlgorithm, algo: &mut Algo, workspace: &mut usize) {
    *algo = previous.algo;
    *workspace = previous.workspace;
}

#[derive(Clone, Copy)]
struct DescriptorDestructors {
    matmul: Destroy,
    layout: Destroy,
    preference: Destroy,
}

struct Api {
    create: Create,
    handle_destroy: Destroy,
    descriptor_destroy: DescriptorDestructors,
    desc_create: DescCreate,
    desc_set: SetAttribute,
    layout_create: LayoutCreate,
    layout_set: SetAttribute,
    preference_create: Create,
    preference_set: SetAttribute,
    heuristic: Heuristic,
    algo_check: AlgoCheck,
    matmul: Matmul,
    _library: Library,
}

impl Api {
    fn load() -> Result<Self, BackendError> {
        // SAFETY: symbol signatures below are the installed CUDA 13.1 ABI.
        let library = unsafe { Library::new("libcublasLt.so.13") }.map_err(super::start)?;
        let create = unsafe {
            *library
                .get::<Create>(b"cublasLtCreate\0")
                .map_err(super::start)?
        };
        let handle_destroy = unsafe {
            *library
                .get::<Destroy>(b"cublasLtDestroy\0")
                .map_err(super::start)?
        };
        let descriptor_destroy = DescriptorDestructors {
            matmul: unsafe {
                *library
                    .get::<Destroy>(b"cublasLtMatmulDescDestroy\0")
                    .map_err(super::start)?
            },
            layout: unsafe {
                *library
                    .get::<Destroy>(b"cublasLtMatrixLayoutDestroy\0")
                    .map_err(super::start)?
            },
            preference: unsafe {
                *library
                    .get::<Destroy>(b"cublasLtMatmulPreferenceDestroy\0")
                    .map_err(super::start)?
            },
        };
        let desc_create = unsafe {
            *library
                .get::<DescCreate>(b"cublasLtMatmulDescCreate\0")
                .map_err(super::start)?
        };
        let desc_set = unsafe {
            *library
                .get::<SetAttribute>(b"cublasLtMatmulDescSetAttribute\0")
                .map_err(super::start)?
        };
        let layout_create = unsafe {
            *library
                .get::<LayoutCreate>(b"cublasLtMatrixLayoutCreate\0")
                .map_err(super::start)?
        };
        let layout_set = unsafe {
            *library
                .get::<SetAttribute>(b"cublasLtMatrixLayoutSetAttribute\0")
                .map_err(super::start)?
        };
        let preference_create = unsafe {
            *library
                .get::<Create>(b"cublasLtMatmulPreferenceCreate\0")
                .map_err(super::start)?
        };
        let preference_set = unsafe {
            *library
                .get::<SetAttribute>(b"cublasLtMatmulPreferenceSetAttribute\0")
                .map_err(super::start)?
        };
        let heuristic = unsafe {
            *library
                .get::<Heuristic>(b"cublasLtMatmulAlgoGetHeuristic\0")
                .map_err(super::start)?
        };
        let algo_check = unsafe {
            *library
                .get::<AlgoCheck>(b"cublasLtMatmulAlgoCheck\0")
                .map_err(super::start)?
        };
        let matmul = unsafe {
            *library
                .get::<Matmul>(b"cublasLtMatmul\0")
                .map_err(super::start)?
        };
        Ok(Self {
            create,
            handle_destroy,
            descriptor_destroy,
            desc_create,
            desc_set,
            layout_create,
            layout_set,
            preference_create,
            preference_set,
            heuristic,
            algo_check,
            matmul,
            _library: library,
        })
    }
}

// This synchronous allocation is freed explicitly so the live-byte gauge is
// decremented only after physical deallocation, rather than a deferred free.
// It is a fixed provider cost charged once alongside arenas, not per plan or
// per logical row count. The current cuBLASLt split-K path reads scratch bytes
// that algorithms do not promise to initialize, so construction zeros the whole
// buffer once on the owned stream before the provider is published.
struct Workspace {
    pointer: u64,
    bytes: usize,
    context: Arc<Context>,
}

impl Workspace {
    fn new(context: Arc<Context>, bytes: usize) -> Result<Self, BackendError> {
        if bytes == 0 {
            return Err(super::start("cuBLASLt workspace must be nonzero"));
        }
        context.bind_to_thread().map_err(super::start)?;
        // SAFETY: the allocation's context is bound, and Drop uses free_sync.
        let pointer = unsafe { cuda_core::memory::malloc_sync(bytes) }.map_err(super::start)?;
        super::LIVE_ARENA_BYTES.fetch_add(bytes, Ordering::Relaxed);
        super::record_live_arena_bytes();
        Ok(Self {
            pointer,
            bytes,
            context,
        })
    }

    fn initialize(&self, stream: &Arc<Stream>) -> Result<(), BackendError> {
        // One-time stream-ordered zero on the provider's own stream. The
        // existing teardown drain makes the initialized contents deterministic
        // before any plan is published or any provider resource is released.
        // SAFETY: the allocation covers self.bytes in the bound context, and
        // stream is the provider's valid stream for that context.
        unsafe {
            cuda_core::memory::memset_d8_async(self.pointer, 0, self.bytes, stream.cu_stream())
        }
        .map_err(super::start)
    }

    fn record_retained(bytes: usize) {
        metrics::counter!("oxide_arena_retained_bytes_total").increment(bytes as u64);
    }
}

impl Drop for Workspace {
    fn drop(&mut self) {
        if self.context.bind_to_thread().is_err()
            // SAFETY: Core drops this allocation only after draining its stream;
            // construction failures have not submitted any users of the pointer.
            || unsafe { cuda_core::memory::free_sync(self.pointer) }.is_err()
        {
            Self::record_retained(self.bytes);
            std::mem::forget(self.context.clone());
            return;
        }
        super::LIVE_ARENA_BYTES.fetch_sub(self.bytes, Ordering::Relaxed);
        super::record_live_arena_bytes();
    }
}

struct Core {
    api: Arc<Api>,
    handle: Handle,
    context: Arc<Context>,
    stream: Arc<Stream>,
    descriptors_safe: Arc<AtomicBool>,
    owner_token: Arc<()>,
    workspace: ManuallyDrop<Workspace>,
    #[cfg(test)]
    shape_rejection: Cell<Option<ShapeRejection>>,
    #[cfg(test)]
    down_shape_rejection: Cell<Option<ShapeRejection>>,
}

impl Drop for Core {
    fn drop(&mut self) {
        if self.context.bind_to_thread().is_err() || self.stream.synchronize().is_err() {
            self.descriptors_safe.store(false, Ordering::Release);
            // Retain the handle, library, context, and stream after an uncertain
            // completion instead of releasing resources still used by the GPU.
            retain_failed_resources(&self.api, &self.context, &self.stream);
            Workspace::record_retained(self.workspace.bytes);
            return;
        }
        if unsafe { (self.api.handle_destroy)(self.handle) } != STATUS_SUCCESS {
            self.descriptors_safe.store(false, Ordering::Release);
            retain_failed_resources(&self.api, &self.context, &self.stream);
            Workspace::record_retained(self.workspace.bytes);
            return;
        }
        // SAFETY: the stream has drained and the handle has been destroyed.
        // Failed drains/destruction above deliberately retain this allocation.
        unsafe { ManuallyDrop::drop(&mut self.workspace) };
    }
}

pub(crate) struct LtProvider {
    core: Box<Core>,
}

// SAFETY: the provider is the sole strong owner of its handle and is movable
// between threads, but it is not `Sync`. Every handle use therefore remains
// externally serialized. Moving a plan also transfers its exclusive host
// descriptors; plan Drop only binds and drains its retained stream.
unsafe impl Send for LtProvider {}

impl LtProvider {
    pub(crate) fn new(context: Arc<Context>, stream: Arc<Stream>) -> Result<Self, BackendError> {
        Self::with_workspace_bytes(context, stream, WORKSPACE_BYTES)
    }

    /// Native-default provider with the dedicated 32 MiB backing allocation.
    pub(crate) fn new_native_default(
        context: Arc<Context>,
        stream: Arc<Stream>,
    ) -> Result<Self, BackendError> {
        Self::new_native_default_workspace_bytes(context, stream, NATIVE_DEFAULT_WORKSPACE_BYTES)
    }

    /// Native-default-only provider constructor. The caller owns the actual backing
    /// byte count, which is zeroed, accounted, and released at this exact size.
    pub(crate) fn new_native_default_workspace_bytes(
        context: Arc<Context>,
        stream: Arc<Stream>,
        workspace_bytes: usize,
    ) -> Result<Self, BackendError> {
        if workspace_bytes < NATIVE_DEFAULT_WORKSPACE_PREFERENCE_BYTES {
            return Err(super::start(format!(
                "native-default cuBLASLt workspace {workspace_bytes} is below the 1 MiB preference"
            )));
        }
        Self::with_workspace_bytes(context, stream, workspace_bytes)
    }

    fn with_workspace_bytes(
        context: Arc<Context>,
        stream: Arc<Stream>,
        workspace_bytes: usize,
    ) -> Result<Self, BackendError> {
        context.bind_to_thread().map_err(super::start)?;
        if !Arc::ptr_eq(stream.context(), &context) {
            return Err(super::start(
                "cuBLASLt stream belongs to a different CUDA context",
            ));
        }
        let api = Arc::new(Api::load()?);
        let workspace = Workspace::new(context.clone(), workspace_bytes)?;
        let mut handle = ptr::null_mut();
        let status = unsafe { (api.create)(&mut handle) };
        if status != STATUS_SUCCESS {
            return Err(super::start(format!(
                "cuBLASLt handle create status {status}"
            )));
        }
        if handle.is_null() {
            return Err(super::start("cuBLASLt returned a null handle"));
        }
        let core = Box::new(Core {
            api,
            handle,
            context,
            stream,
            descriptors_safe: Arc::new(AtomicBool::new(true)),
            owner_token: Arc::new(()),
            workspace: ManuallyDrop::new(workspace),
            #[cfg(test)]
            shape_rejection: Cell::new(None),
            #[cfg(test)]
            down_shape_rejection: Cell::new(None),
        });
        core.workspace.initialize(&core.stream)?;
        Ok(Self { core })
    }

    /// FP8 reuses this provider core with one shared initialized 32 MiB workspace.
    pub(crate) fn new_fp8(
        context: Arc<Context>,
        stream: Arc<Stream>,
    ) -> Result<Self, BackendError> {
        Self::with_workspace_bytes(context, stream, NATIVE_DEFAULT_WORKSPACE_BYTES)
    }

    /// Bind this provider's CUDA context once for a safe standalone operation.
    ///
    /// Engine-driven requests bind once before their body and use the bound
    /// prepare/launch variants directly; this helper is the boundary for the
    /// safe convenience wrapper and standalone primitive callers.
    pub(crate) fn bind_context(&self) -> Result<(), BackendError> {
        #[cfg(test)]
        super::bump_provider_context_query();
        self.core.context.bind_to_thread().map_err(super::infer)
    }

    pub(crate) fn stream(&self) -> &Stream {
        &self.core.stream
    }

    pub(crate) fn context(&self) -> &Arc<Context> {
        &self.core.context
    }

    // Strided entry point retained for operator qualification and future callers.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn prepare_bias<E: Element>(
        &self,
        plan: &LinearPlan<E>,
    ) -> Result<LtPlan<E>, BackendError> {
        self.prepare(Geometry::from_linear(plan)?, Epilogue::Bias)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn prepare_gelu_bias<E: Element>(
        &self,
        plan: &LinearPlan<E>,
    ) -> Result<LtPlan<E>, BackendError> {
        self.prepare(Geometry::from_linear(plan)?, Epilogue::GeluBias)
    }

    #[allow(dead_code)]
    pub(crate) fn prepare_bias_shape<E: Element>(
        &self,
        input: usize,
        output: usize,
    ) -> Result<LtPlan<E>, BackendError> {
        self.prepare(Geometry::contiguous::<E>(input, output)?, Epilogue::Bias)
    }

    #[allow(dead_code)]
    pub(crate) fn prepare_gelu_bias_shape<E: Element>(
        &self,
        input: usize,
        output: usize,
    ) -> Result<LtPlan<E>, BackendError> {
        self.prepare(
            Geometry::contiguous::<E>(input, output)?,
            Epilogue::GeluBias,
        )
    }

    pub(crate) fn prepare_native_small_bias_shape<E: Element>(
        &self,
        input: usize,
        output: usize,
    ) -> Result<LtPlan<E>, BackendError> {
        self.prepare_encoder_small(Geometry::contiguous::<E>(input, output)?, Epilogue::Bias)
    }

    pub(crate) fn prepare_native_small_gelu_bias_shape<E: Element>(
        &self,
        input: usize,
        output: usize,
    ) -> Result<LtPlan<E>, BackendError> {
        self.prepare_encoder_small(
            Geometry::contiguous::<E>(input, output)?,
            Epilogue::GeluBias,
        )
    }

    fn prepare_encoder_small<E: Element>(
        &self,
        geometry: Geometry,
        epilogue: Epilogue,
    ) -> Result<LtPlan<E>, BackendError> {
        self.prepare_inner(geometry, epilogue, true)
            .map_err(|error| super::start(format!("native small-row preparation failed: {error}")))
    }

    fn prepare<E: Element>(
        &self,
        geometry: Geometry,
        epilogue: Epilogue,
    ) -> Result<LtPlan<E>, BackendError> {
        self.prepare_inner(geometry, epilogue, false)
    }

    fn allocate_plan<E: Element>(
        &self,
        geometry: Geometry,
        epilogue: Epilogue,
    ) -> Result<LtPlan<E>, BackendError> {
        let element_type = element_type::<E>()?;
        self.core.context.bind_to_thread().map_err(super::infer)?;
        let api = self.core.api.clone();
        let mut plan = LtPlan::<E> {
            api: api.clone(),
            context: self.core.context.clone(),
            stream: self.core.stream.clone(),
            owner: self.core.owner_token.clone(),
            descriptors_safe: self.core.descriptors_safe.clone(),
            op: ptr::null_mut(),
            a: ptr::null_mut(),
            b: ptr::null_mut(),
            y: ptr::null_mut(),
            pref: ptr::null_mut(),
            algo: Algo::default(),
            base: PreparedAlgorithm::default(),
            small: [None; SMALL_ROWS],
            small_prepared: false,
            native_default_cache: None,
            native_default_bias_pointer: ptr::null(),
            native_default_alignment: NATIVE_DEFAULT_ALIGNMENT,
            geometry,
            epilogue,
            rows: geometry.seed_rows,
            workspace: SEED_WORKSPACE_BYTES,
            usable: true,
            _element: PhantomData,
        };

        let mut op = ptr::null_mut();
        let status = unsafe { (api.desc_create)(&mut op, CUBLAS_COMPUTE_32F, CUDA_R_32F) };
        plan.op = op;
        check(status, "matmul descriptor create")?;
        if plan.op.is_null() {
            return Err(super::infer("cuBLASLt returned a null matmul descriptor"));
        }
        let pointer_mode: i32 = 0;
        let transa = CUBLAS_OP_T;
        let transb: i32 = 0;
        let epilogue_attribute = epilogue.attribute();
        set_desc(
            &api,
            plan.op,
            MATMUL_DESC_POINTER_MODE,
            &pointer_mode,
            "pointer mode",
        )?;
        set_desc(&api, plan.op, MATMUL_DESC_TRANSA, &transa, "transpose A")?;
        set_desc(&api, plan.op, MATMUL_DESC_TRANSB, &transb, "transpose B")?;
        set_desc(
            &api,
            plan.op,
            MATMUL_DESC_EPILOGUE,
            &epilogue_attribute,
            "epilogue",
        )?;
        set_desc(
            &api,
            plan.op,
            MATMUL_DESC_BIAS_DATA_TYPE,
            &element_type,
            "bias data type",
        )?;

        plan.a = create_layout(
            &api,
            element_type,
            geometry.input,
            geometry.width,
            geometry.weight_stride,
            "layout A",
        )?;
        plan.b = create_layout(
            &api,
            element_type,
            geometry.input,
            geometry.seed_rows,
            geometry.x_stride,
            "layout B",
        )?;
        plan.y = create_layout(
            &api,
            element_type,
            geometry.width,
            geometry.seed_rows,
            geometry.output_stride,
            "layout Y",
        )?;

        let mut pref = ptr::null_mut();
        let status = unsafe { (api.preference_create)(&mut pref) };
        plan.pref = pref;
        check(status, "preference create")?;
        if plan.pref.is_null() {
            return Err(super::infer(
                "cuBLASLt returned a null preference descriptor",
            ));
        }
        Ok(plan)
    }

    fn prepare_inner<E: Element>(
        &self,
        geometry: Geometry,
        epilogue: Epilogue,
        small: bool,
    ) -> Result<LtPlan<E>, BackendError> {
        let mut plan = self.allocate_plan(geometry, epilogue)?;
        set_desc(
            &plan.api,
            plan.op,
            MATMUL_DESC_BIAS_POINTER,
            &SEED_BIAS_POINTER,
            "seed bias pointer",
        )?;
        let workspace_limit = SEED_WORKSPACE_BYTES as u64;
        set_preference(
            &plan.api,
            plan.pref,
            PREFERENCE_MAX_WORKSPACE_BYTES,
            &workspace_limit,
            "workspace preference",
        )?;

        let (choices, count) = self.heuristic(&plan)?;
        let algo = select_zero_workspace_algorithm(&choices[..count])?;
        let base = check_algorithm(algo, |algo| {
            plan.algo = *algo;
            self.verify(&plan, plan.rows, WORKSPACE_BYTES)
        })
        .map_err(ShapeError::into_backend)?;
        plan.base = base;
        plan.algo = base.algo;
        plan.workspace = base.workspace;
        if small {
            self.prepare_small_rows(&mut plan)?;
        }
        self.set_logical_rows(&mut plan, geometry.initial_rows as usize)
            .map_err(ShapeError::into_backend)?;
        Ok(plan)
    }

    pub(crate) fn prepare_native_default_bias_shape<E: Element>(
        &self,
        input: usize,
        output: usize,
        rows: usize,
    ) -> Result<LtPlan<E>, BackendError> {
        if self.core.workspace.bytes < NATIVE_DEFAULT_WORKSPACE_PREFERENCE_BYTES {
            return Err(super::infer(
                "native-default cuBLASLt provider does not own the required backing workspace",
            ));
        }
        let geometry = Geometry::contiguous_rows::<E>(input, output, rows)?;
        let mut plan = self.allocate_plan(geometry, Epilogue::Bias)?;
        let workspace_limit = NATIVE_DEFAULT_WORKSPACE_PREFERENCE_BYTES as u64;
        set_preference(
            &plan.api,
            plan.pref,
            PREFERENCE_MAX_WORKSPACE_BYTES,
            &workspace_limit,
            "native-default workspace preference",
        )?;
        set_native_default_alignment(&plan.api, plan.pref, NATIVE_DEFAULT_ALIGNMENT)?;
        plan.native_default_alignment = NATIVE_DEFAULT_ALIGNMENT;
        plan.native_default_cache = Some(Box::<NativeDefaultCache>::default());
        Ok(plan)
    }

    fn heuristic<E: Element>(
        &self,
        plan: &LtPlan<E>,
    ) -> Result<([HeuristicResult; HEURISTIC_CHOICES], usize), BackendError> {
        let mut choices = [HeuristicResult::default(); HEURISTIC_CHOICES];
        let mut count = 0i32;
        let status = unsafe {
            (self.core.api.heuristic)(
                self.core.handle,
                plan.op,
                plan.a,
                plan.b,
                plan.y,
                plan.y,
                plan.pref,
                HEURISTIC_CHOICES as i32,
                choices.as_mut_ptr(),
                &mut count,
            )
        };
        if status == STATUS_NOT_SUPPORTED {
            return Err(super::infer(
                "cuBLASLt has no fused bias/GELU heuristic for this layout",
            ));
        }
        check(status, "algorithm heuristic")?;
        let count = usize::try_from(count)
            .ok()
            .filter(|count| *count > 0 && *count <= choices.len())
            .ok_or_else(|| super::infer("cuBLASLt returned an invalid heuristic count"))?;
        Ok((choices, count))
    }

    fn prepare_small_rows<E: Element>(&self, plan: &mut LtPlan<E>) -> Result<(), BackendError> {
        let workspace_limit = WORKSPACE_BYTES as u64;
        set_preference(
            &plan.api,
            plan.pref,
            PREFERENCE_MAX_WORKSPACE_BYTES,
            &workspace_limit,
            "small-row workspace preference",
        )?;
        let mut small = [None; SMALL_ROWS];
        for (index, slot) in small.iter_mut().enumerate() {
            let rows = dimension(index + 1)?;
            set_columns(&plan.api, plan, rows)?;
            let (choices, count) = self.heuristic(plan)?;
            let algo = select_small_algorithm(&choices[..count], WORKSPACE_BYTES)?;
            let prepared = check_algorithm(algo, |algo| {
                plan.algo = *algo;
                self.verify(plan, rows, WORKSPACE_BYTES)
            })
            .map_err(ShapeError::into_backend)?;
            *slot = Some(prepared);
        }
        set_columns(&plan.api, plan, SEED_ROWS as i32)?;
        plan.algo = plan.base.algo;
        plan.workspace = plan.base.workspace;
        plan.small = small;
        plan.small_prepared = true;
        Ok(())
    }

    pub(crate) fn set_logical_rows<E: Element>(
        &self,
        plan: &mut LtPlan<E>,
        rows: usize,
    ) -> Result<(), ShapeError> {
        self.belongs(plan)?;
        if !plan.usable {
            return Err(ShapeError::Fatal(super::infer(
                "fused plan is unusable after a failed descriptor rollback",
            )));
        }
        // Pure validation precedes context binding and descriptor mutation.
        let rows = plan.geometry.rows(rows).map_err(ShapeError::Rejected)?;
        if rows == plan.rows {
            return Ok(());
        }
        let target = prepared_for_rows(plan.base, &plan.small, plan.small_prepared, rows as usize)
            .map_err(ShapeError::Fatal)?;
        self.bind_context()?;
        let previous_rows = plan.rows;
        let previous = PreparedAlgorithm {
            algo: plan.algo,
            workspace: plan.workspace,
        };
        if let Err(error) = set_columns(&plan.api, plan, rows) {
            let restored = set_columns(&plan.api, plan, previous_rows);
            return Err(after_rollback(error.into(), restored, &mut plan.usable));
        }
        plan.algo = target.algo;
        match self.verify(plan, rows, self.core.workspace.bytes) {
            Ok(workspace) => {
                plan.rows = rows;
                plan.workspace = workspace;
                Ok(())
            }
            Err(error) => {
                restore_prepared(previous, &mut plan.algo, &mut plan.workspace);
                let restored = set_columns(&plan.api, plan, previous_rows);
                Err(after_rollback(error, restored, &mut plan.usable))
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn reject_shape_once_after(&self, rows: usize, successful_checks: usize) {
        self.core.shape_rejection.set(Some(ShapeRejection {
            rows: i32::try_from(rows).expect("test shape must fit cuBLAS row count"),
            successful_checks_left: successful_checks,
        }));
    }

    #[cfg(test)]
    pub(crate) fn reject_down_shape_once_after(&self, rows: usize, successful_checks: usize) {
        self.core.down_shape_rejection.set(Some(ShapeRejection {
            rows: i32::try_from(rows).expect("test shape must fit cuBLAS row count"),
            successful_checks_left: successful_checks,
        }));
    }

    fn verify<E: Element>(
        &self,
        plan: &LtPlan<E>,
        rows: i32,
        workspace_limit: usize,
    ) -> Result<usize, ShapeError> {
        self.verify_with(plan, rows, workspace_limit, |status, checked| {
            // Test-only rejection occurs after a real successful check, while
            // descriptors contain the requested rows. Normal rollback handles it.
            #[cfg(test)]
            if status == STATUS_SUCCESS
                && checked.state == STATUS_SUCCESS
                && checked.workspace_size <= workspace_limit
                && inject_shape_rejection(&self.core.shape_rejection, rows)
            {
                return STATUS_NOT_SUPPORTED;
            }
            let _ = (status, checked);
            status
        })
    }

    #[cfg(test)]
    fn verify_down<E: Element>(
        &self,
        plan: &LtPlan<E>,
        rows: i32,
        workspace_limit: usize,
    ) -> Result<usize, ShapeError> {
        self.verify_with(plan, rows, workspace_limit, |status, checked| {
            if status == STATUS_SUCCESS
                && checked.state == STATUS_SUCCESS
                && checked.workspace_size <= workspace_limit
                && inject_shape_rejection(&self.core.down_shape_rejection, rows)
            {
                return STATUS_NOT_SUPPORTED;
            }
            status
        })
    }

    #[cfg(not(test))]
    fn verify_down<E: Element>(
        &self,
        plan: &LtPlan<E>,
        rows: i32,
        workspace_limit: usize,
    ) -> Result<usize, ShapeError> {
        self.verify(plan, rows, workspace_limit)
    }

    fn verify_with<E: Element>(
        &self,
        plan: &LtPlan<E>,
        rows: i32,
        workspace_limit: usize,
        inject: impl FnOnce(i32, &HeuristicResult) -> i32,
    ) -> Result<usize, ShapeError> {
        let geometry = || {
            format!(
                "M={rows} K={} N={} epilogue={:?}",
                plan.geometry.input, plan.geometry.width, plan.epilogue
            )
        };
        let mut checked = HeuristicResult::default();
        let status = unsafe {
            (self.core.api.algo_check)(
                self.core.handle,
                plan.op,
                plan.a,
                plan.b,
                plan.y,
                plan.y,
                &plan.algo,
                &mut checked,
            )
        };
        let status = inject(status, &checked);
        checked_workspace_with_limit(status, &checked, workspace_limit, geometry)
    }

    fn belongs<E: Element>(&self, plan: &LtPlan<E>) -> Result<(), BackendError> {
        if !Arc::ptr_eq(&plan.owner, &self.core.owner_token) {
            return Err(super::infer(
                "fused plan belongs to a different cuBLASLt provider",
            ));
        }
        Ok(())
    }

    /// Enqueue the prepared fused projection with borrowed, checked buffers.
    ///
    /// # Safety
    /// The caller must bind this provider's context on the current thread and
    /// keep all buffers and the output free of conflicting access until the
    /// stream drains. This method neither binds the context nor synchronizes.
    pub(crate) unsafe fn enqueue<E: Element>(
        &self,
        plan: &LtPlan<E>,
        x: &DeviceBuffer<E::Storage>,
        weight: &DeviceBuffer<E::Storage>,
        bias: &DeviceBuffer<E::Storage>,
        output: &mut DeviceBuffer<E::Storage>,
    ) -> Result<(), BackendError> {
        self.belongs(plan)?;
        if !plan.usable {
            return Err(super::infer(
                "fused plan is unusable after a failed descriptor rollback",
            ));
        }
        if plan.workspace > self.core.workspace.bytes {
            return Err(super::infer(format!(
                "fused plan requires {} workspace bytes, provider limit {} (M={} K={} N={} epilogue={:?})",
                plan.workspace,
                self.core.workspace.bytes,
                plan.rows,
                plan.geometry.input,
                plan.geometry.width,
                plan.epilogue
            )));
        }
        plan.geometry.check_bias(bias.len())?;
        for context in [
            x.context(),
            weight.context(),
            bias.context(),
            output.context(),
        ] {
            if !Arc::ptr_eq(context, &self.core.context) {
                return Err(super::infer(
                    "fused projection buffers belong to a different CUDA context",
                ));
            }
        }

        let (x_layout, weight_layout, output_layout) = plan.geometry.layouts(plan.rows)?;
        let x_view = TensorRef::<E, 2>::new(x, x_layout).map_err(super::infer)?;
        let weight_view = TensorRef::<E, 2>::new(weight, weight_layout).map_err(super::infer)?;
        let bias_layout =
            Layout::<1>::contiguous([plan.geometry.width as usize]).map_err(super::infer)?;
        let bias_view = TensorRef::<E, 1>::new(bias, bias_layout).map_err(super::infer)?;
        let output_view = TensorMut::<E, 2>::new(output, output_layout).map_err(super::infer)?;

        let bias_pointer = bias_view.device_ptr() as *const c_void;
        set_desc(
            &self.core.api,
            plan.op,
            MATMUL_DESC_BIAS_POINTER,
            &bias_pointer,
            "bias pointer",
        )?;

        self.matmul_prepared(
            plan,
            weight_view.device_ptr(),
            x_view.device_ptr(),
            output_view.device_ptr(),
        )
    }

    unsafe fn matmul_prepared<E: Element>(
        &self,
        plan: &LtPlan<E>,
        weight: u64,
        x: u64,
        output: u64,
    ) -> Result<(), BackendError> {
        let alpha = 1.0f32;
        let beta = 0.0f32;
        // All submissions use the same ordered stream and are externally
        // serialized, so this one provider allocation can serve every plan.
        let workspace = self.core.workspace.pointer as *mut c_void;
        let stream = self.core.stream.cu_stream().cast();
        // SAFETY: callers validated every buffer view and retained its owner.
        let status = unsafe {
            (self.core.api.matmul)(
                self.core.handle,
                plan.op,
                &alpha,
                weight as *const c_void,
                plan.a,
                x as *const c_void,
                plan.b,
                &beta,
                output as *const c_void,
                plan.y,
                output as *mut c_void,
                plan.y,
                &plan.algo,
                workspace,
                plan.workspace,
                stream,
            )
        };
        check(status, "fused matmul")
    }

    fn heuristic_one<E: Element>(&self, plan: &LtPlan<E>) -> Result<HeuristicResult, ShapeError> {
        let mut choice = HeuristicResult::default();
        let mut count = 0i32;
        let status = unsafe {
            (self.core.api.heuristic)(
                self.core.handle,
                plan.op,
                plan.a,
                plan.b,
                plan.y,
                plan.y,
                plan.pref,
                1,
                &mut choice,
                &mut count,
            )
        };
        if status == STATUS_NOT_SUPPORTED {
            return Err(ShapeError::Rejected(super::infer(
                "cuBLASLt has no fused bias heuristic for this native-default shape",
            )));
        }
        if status != STATUS_SUCCESS {
            return Err(ShapeError::Fatal(super::infer(format!(
                "cuBLASLt native-default heuristic status {status}"
            ))));
        }
        if count != 1 || choice.state != STATUS_SUCCESS {
            return Err(ShapeError::Rejected(super::infer(
                "cuBLASLt returned no successful native-default heuristic",
            )));
        }
        Ok(choice)
    }

    /// Prepare the native-default policy with an actual-M/alignment cache.
    ///
    /// This is the pure planning half of the shared executor: it validates all
    /// real arguments, reuses the persistent cache or the bounded request-scoped
    /// pending store, and runs the existing heuristic plus `AlgoCheck` only on a
    /// cold key. It submits no GPU work and allocates nothing. A successful cold
    /// result remains pending until [`Self::launch_native_default_bound`] proves
    /// the vendor matmul launched.
    ///
    /// # Safety
    /// The caller must have bound this provider's CUDA context on the current
    /// thread and keep it current for the subsequent bound launch. This method
    /// performs no context bind or `cuCtxGetCurrent` query.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn prepare_native_default_bound<E: Element>(
        &self,
        plan: &mut LtPlan<E>,
        pending: &mut NativeDefaultPending,
        rows: usize,
        x: &DeviceBuffer<E::Storage>,
        weight: &DeviceBuffer<E::Storage>,
        bias: &DeviceBuffer<E::Storage>,
        output: &mut DeviceBuffer<E::Storage>,
    ) -> Result<NativeDefaultPrepared, ShapeError> {
        self.belongs(plan).map_err(ShapeError::Fatal)?;
        if !plan.usable {
            return Err(ShapeError::Fatal(super::infer(
                "native-default plan is unusable after a failed descriptor rollback",
            )));
        }
        if plan.native_default_cache.is_none() {
            return Err(ShapeError::Fatal(super::infer(
                "native-default plan was not prepared with the default ATen policy",
            )));
        }
        let rows = plan.geometry.rows(rows).map_err(ShapeError::Rejected)?;
        plan.geometry
            .check_bias(bias.len())
            .map_err(ShapeError::Fatal)?;
        for context in [
            x.context(),
            weight.context(),
            bias.context(),
            output.context(),
        ] {
            if !Arc::ptr_eq(context, &self.core.context) {
                return Err(ShapeError::Fatal(super::infer(
                    "native-default projection buffers belong to a different CUDA context",
                )));
            }
        }

        let (x_layout, weight_layout, output_layout) =
            plan.geometry.layouts(rows).map_err(ShapeError::Fatal)?;
        let x_view = TensorRef::<E, 2>::new(x, x_layout)
            .map_err(|error| ShapeError::Fatal(super::infer(error)))?;
        let weight_view = TensorRef::<E, 2>::new(weight, weight_layout)
            .map_err(|error| ShapeError::Fatal(super::infer(error)))?;
        let bias_layout = Layout::<1>::contiguous([plan.geometry.width as usize])
            .map_err(|error| ShapeError::Fatal(super::infer(error)))?;
        let bias_view = TensorRef::<E, 1>::new(bias, bias_layout)
            .map_err(|error| ShapeError::Fatal(super::infer(error)))?;
        let output_view = TensorMut::<E, 2>::new(output, output_layout)
            .map_err(|error| ShapeError::Fatal(super::infer(error)))?;

        let alignment = NativeDefaultAlignment {
            a: get_alignment(weight_view.device_ptr()),
            b: get_alignment(x_view.device_ptr()),
            c: get_alignment(output_view.device_ptr()),
            d: get_alignment(output_view.device_ptr()),
        };
        let key = NativeDefaultCacheKey {
            rows,
            input: plan.geometry.input,
            width: plan.geometry.width,
            x_stride: plan.geometry.x_stride,
            weight_stride: plan.geometry.weight_stride,
            output_stride: plan.geometry.output_stride,
            element_type: element_type::<E>().map_err(ShapeError::Fatal)?,
            epilogue: plan.epilogue,
            workspace_preference: NATIVE_DEFAULT_WORKSPACE_PREFERENCE_BYTES,
            alignment,
        };
        let bias_pointer = bias_view.device_ptr();
        let invocation = |prepared, pending| NativeDefaultPrepared {
            rows,
            key,
            prepared,
            bias_pointer,
            alignment,
            x_pointer: x_view.device_ptr(),
            weight_pointer: weight_view.device_ptr(),
            output_pointer: output_view.device_ptr(),
            pending,
        };

        if let Some(prepared) = plan
            .native_default_cache
            .as_ref()
            .and_then(|cache| cache.get(&key))
        {
            return Ok(invocation(prepared, false));
        }
        if let Some(prepared) = pending.get(&key) {
            return Ok(invocation(prepared, true));
        }

        let snapshot = snapshot_native_default(plan);
        if rows != snapshot.rows {
            if let Err(error) = set_columns(&plan.api, plan, rows) {
                return Err(fail_native_default(
                    &self.core.api,
                    plan,
                    snapshot,
                    error.into(),
                ));
            }
        }
        let bias_descriptor = bias_pointer as *const c_void;
        if let Err(error) = set_desc(
            &self.core.api,
            plan.op,
            MATMUL_DESC_BIAS_POINTER,
            &bias_descriptor,
            "native-default bias pointer",
        ) {
            return Err(fail_native_default(
                &self.core.api,
                plan,
                snapshot,
                error.into(),
            ));
        }
        if let Err(error) = set_native_default_alignment(&self.core.api, plan.pref, alignment) {
            return Err(fail_native_default(
                &self.core.api,
                plan,
                snapshot,
                error.into(),
            ));
        }
        #[cfg(test)]
        if let Some(cache) = plan.native_default_cache.as_mut() {
            cache.queries += 1;
        }
        let choice = match self.heuristic_one(plan) {
            Ok(choice) => choice,
            Err(error) => return Err(fail_native_default(&self.core.api, plan, snapshot, error)),
        };
        plan.algo = choice.algo;
        let workspace_limit = self
            .core
            .workspace
            .bytes
            .min(NATIVE_DEFAULT_WORKSPACE_PREFERENCE_BYTES);
        let prepared = match self.verify_down(plan, rows, workspace_limit) {
            Ok(workspace) => PreparedAlgorithm {
                algo: choice.algo,
                workspace,
            },
            Err(error) => return Err(fail_native_default(&self.core.api, plan, snapshot, error)),
        };
        // Keep the descriptor fields consistent with the validated pending
        // metadata. Publication to the persistent cache still waits for launch.
        plan.rows = rows;
        plan.workspace = prepared.workspace;
        plan.native_default_bias_pointer = bias_descriptor;
        plan.native_default_alignment = alignment;
        pending.insert(key, prepared);
        Ok(invocation(prepared, true))
    }

    /// Launch one prepared native-default invocation. No heuristic or
    /// `AlgoCheck` runs here, and a cold pending entry is published only after
    /// the vendor matmul itself succeeds.
    ///
    /// # Safety
    /// The caller must have bound this provider's CUDA context on the current
    /// thread, keep it current through launch, and keep every buffer and the
    /// output free of conflicting access until this provider's stream drains.
    /// `prepared` must have been produced by
    /// [`Self::prepare_native_default_bound`] for the same buffers. This method
    /// performs no context bind or `cuCtxGetCurrent` query.
    pub(crate) unsafe fn launch_native_default_bound<E: Element>(
        &self,
        plan: &mut LtPlan<E>,
        prepared: &NativeDefaultPrepared,
        x: &DeviceBuffer<E::Storage>,
        weight: &DeviceBuffer<E::Storage>,
        output: &mut DeviceBuffer<E::Storage>,
    ) -> Result<(), ShapeError> {
        self.belongs(plan).map_err(ShapeError::Fatal)?;
        if !plan.usable {
            return Err(ShapeError::Fatal(super::infer(
                "native-default plan is unusable after a failed descriptor rollback",
            )));
        }
        if plan.native_default_cache.is_none() {
            return Err(ShapeError::Fatal(super::infer(
                "native-default plan was not prepared with the default ATen policy",
            )));
        }
        if x.cu_deviceptr() != prepared.x_pointer
            || weight.cu_deviceptr() != prepared.weight_pointer
            || output.cu_deviceptr() != prepared.output_pointer
        {
            return Err(ShapeError::Fatal(super::infer(
                "native-default launch buffers differ from prepared metadata",
            )));
        }
        let snapshot = snapshot_native_default(plan);
        if prepared.rows != snapshot.rows {
            if let Err(error) = set_columns(&plan.api, plan, prepared.rows) {
                return Err(fail_native_default(
                    &self.core.api,
                    plan,
                    snapshot,
                    error.into(),
                ));
            }
        }
        let bias_descriptor = prepared.bias_pointer as *const c_void;
        if let Err(error) = set_desc(
            &self.core.api,
            plan.op,
            MATMUL_DESC_BIAS_POINTER,
            &bias_descriptor,
            "native-default bias pointer",
        ) {
            return Err(fail_native_default(
                &self.core.api,
                plan,
                snapshot,
                error.into(),
            ));
        }
        if snapshot.alignment != prepared.alignment {
            if let Err(error) =
                set_native_default_alignment(&self.core.api, plan.pref, prepared.alignment)
            {
                return Err(fail_native_default(
                    &self.core.api,
                    plan,
                    snapshot,
                    error.into(),
                ));
            }
        }
        plan.rows = prepared.rows;
        plan.algo = prepared.prepared.algo;
        plan.workspace = prepared.prepared.workspace;
        plan.native_default_bias_pointer = bias_descriptor;
        plan.native_default_alignment = prepared.alignment;
        // SAFETY: the caller's retained-buffer contract covers these pointers,
        // which were validated to match the prepared metadata.
        if let Err(error) = unsafe {
            self.matmul_prepared(
                plan,
                prepared.weight_pointer,
                prepared.x_pointer,
                prepared.output_pointer,
            )
        } {
            return Err(fail_native_default(
                &self.core.api,
                plan,
                snapshot,
                ShapeError::Fatal(error),
            ));
        }
        if prepared.pending {
            let Some(cache) = plan.native_default_cache.as_mut() else {
                return Err(fail_native_default(
                    &self.core.api,
                    plan,
                    snapshot,
                    ShapeError::Fatal(super::infer("native-default cache disappeared")),
                ));
            };
            cache.insert(prepared.key, prepared.prepared);
        }
        Ok(())
    }

    /// Unbound standalone native-default convenience entry point: bind once,
    /// then reuse the same bound prepare/launch implementation used by the
    /// Engine. Only the usual asynchronous buffer-lifetime contract is unsafe.
    ///
    /// # Safety
    /// The caller must keep every buffer and the output free of conflicting
    /// access until this provider's stream drains.
    #[allow(dead_code, clippy::too_many_arguments)]
    pub(crate) unsafe fn enqueue_native_default<E: Element>(
        &self,
        plan: &mut LtPlan<E>,
        rows: usize,
        x: &DeviceBuffer<E::Storage>,
        weight: &DeviceBuffer<E::Storage>,
        bias: &DeviceBuffer<E::Storage>,
        output: &mut DeviceBuffer<E::Storage>,
    ) -> Result<(), ShapeError> {
        self.bind_context().map_err(ShapeError::Fatal)?;
        let mut pending = NativeDefaultPending::default();
        // SAFETY: this thread bound the provider context above, and the caller's
        // retained-buffer contract covers both prepare and launch.
        let prepared = unsafe {
            self.prepare_native_default_bound(plan, &mut pending, rows, x, weight, bias, output)?
        };
        // SAFETY: the context remains bound; the prepared invocation was built
        // from these exact buffers.
        unsafe { self.launch_native_default_bound(plan, &prepared, x, weight, output) }
    }

    #[cfg(test)]
    pub(crate) fn native_default_queries<E: Element>(&self, plan: &LtPlan<E>) -> usize {
        let _ = self;
        plan.native_default_cache
            .as_ref()
            .map_or(0, |cache| cache.queries)
    }

    #[cfg(test)]
    pub(crate) fn native_default_cache_len<E: Element>(&self, plan: &LtPlan<E>) -> usize {
        let _ = self;
        plan.native_default_cache
            .as_ref()
            .map_or(0, |cache| cache.len())
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn checked_workspace(
    status: i32,
    checked: &HeuristicResult,
    geometry: impl FnOnce() -> String,
) -> Result<usize, ShapeError> {
    checked_workspace_with_limit(status, checked, WORKSPACE_BYTES, geometry)
}

fn checked_workspace_with_limit(
    status: i32,
    checked: &HeuristicResult,
    workspace_limit: usize,
    geometry: impl FnOnce() -> String,
) -> Result<usize, ShapeError> {
    if status != STATUS_SUCCESS {
        let error = super::infer(format!(
            "cuBLASLt algo check status {status} ({})",
            geometry()
        ));
        return Err(if status == STATUS_NOT_SUPPORTED {
            ShapeError::Rejected(error)
        } else {
            ShapeError::Fatal(error)
        });
    }
    if checked.state != STATUS_SUCCESS {
        let error = super::infer(format!(
            "cuBLASLt algo result state {} ({})",
            checked.state,
            geometry()
        ));
        return Err(if checked.state == STATUS_NOT_SUPPORTED {
            ShapeError::Rejected(error)
        } else {
            ShapeError::Fatal(error)
        });
    }
    if checked.workspace_size > workspace_limit {
        return Err(ShapeError::Rejected(super::infer(format!(
            "fused algorithm requires {} workspace bytes, provider limit {workspace_limit} ({})",
            checked.workspace_size,
            geometry()
        ))));
    }
    Ok(checked.workspace_size)
}

fn set_native_default_alignment(
    api: &Api,
    preference: Descriptor,
    alignment: NativeDefaultAlignment,
) -> Result<(), BackendError> {
    for (attribute, value) in [
        (PREFERENCE_MIN_ALIGNMENT_A_BYTES, alignment.a),
        (PREFERENCE_MIN_ALIGNMENT_B_BYTES, alignment.b),
        (PREFERENCE_MIN_ALIGNMENT_C_BYTES, alignment.c),
        (PREFERENCE_MIN_ALIGNMENT_D_BYTES, alignment.d),
    ] {
        set_preference(
            api,
            preference,
            attribute,
            &value,
            "native-default alignment preference",
        )?;
    }
    Ok(())
}

fn snapshot_native_default<E: Element>(plan: &LtPlan<E>) -> NativeDefaultSnapshot {
    NativeDefaultSnapshot {
        rows: plan.rows,
        algo: plan.algo,
        workspace: plan.workspace,
        bias_pointer: plan.native_default_bias_pointer,
        alignment: plan.native_default_alignment,
    }
}

#[derive(Clone, Copy)]
struct NativeDefaultMutableState {
    rows: i32,
    algo: Algo,
    workspace: usize,
    bias_pointer: *const c_void,
    alignment: NativeDefaultAlignment,
}

fn native_default_state<E: Element>(plan: &LtPlan<E>) -> NativeDefaultMutableState {
    NativeDefaultMutableState {
        rows: plan.rows,
        algo: plan.algo,
        workspace: plan.workspace,
        bias_pointer: plan.native_default_bias_pointer,
        alignment: plan.native_default_alignment,
    }
}

fn apply_native_default_snapshot(
    state: &mut NativeDefaultMutableState,
    snapshot: NativeDefaultSnapshot,
) {
    state.rows = snapshot.rows;
    state.algo = snapshot.algo;
    state.workspace = snapshot.workspace;
    state.bias_pointer = snapshot.bias_pointer;
    state.alignment = snapshot.alignment;
}

fn write_native_default_state<E: Element>(plan: &mut LtPlan<E>, state: NativeDefaultMutableState) {
    plan.rows = state.rows;
    plan.algo = state.algo;
    plan.workspace = state.workspace;
    plan.native_default_bias_pointer = state.bias_pointer;
    plan.native_default_alignment = state.alignment;
}

fn run_native_default_rollback(
    state: &mut NativeDefaultMutableState,
    snapshot: NativeDefaultSnapshot,
    mut restore_columns: impl FnMut() -> Result<(), BackendError>,
    mut restore_bias: impl FnMut() -> Result<(), BackendError>,
    mut restore_alignment: impl FnMut() -> Result<(), BackendError>,
) -> Result<(), BackendError> {
    let mut result = Ok(());
    if let Err(error) = restore_columns() {
        result = Err(error);
    }
    apply_native_default_snapshot(state, snapshot);
    if let Err(error) = restore_bias() {
        result = Err(error);
    }
    if let Err(error) = restore_alignment() {
        result = Err(error);
    }
    result
}

fn restore_native_default<E: Element>(
    api: &Api,
    plan: &mut LtPlan<E>,
    snapshot: NativeDefaultSnapshot,
) -> Result<(), BackendError> {
    let mut state = native_default_state(plan);
    let result = run_native_default_rollback(
        &mut state,
        snapshot,
        || set_columns(api, plan, snapshot.rows),
        || {
            set_desc(
                api,
                plan.op,
                MATMUL_DESC_BIAS_POINTER,
                &snapshot.bias_pointer,
                "native-default bias pointer rollback",
            )
        },
        || set_native_default_alignment(api, plan.pref, snapshot.alignment),
    );
    write_native_default_state(plan, state);
    result
}

fn native_default_failure(
    error: ShapeError,
    rollback: Result<(), BackendError>,
    usable: &mut bool,
) -> ShapeError {
    match rollback {
        Ok(()) => error,
        Err(rollback) => {
            *usable = false;
            ShapeError::Fatal(super::infer(format!(
                "{}; failed to restore native-default transaction: {rollback}",
                error.into_backend()
            )))
        }
    }
}

fn fail_native_default<E: Element>(
    api: &Api,
    plan: &mut LtPlan<E>,
    snapshot: NativeDefaultSnapshot,
    error: ShapeError,
) -> ShapeError {
    let rollback = restore_native_default(api, plan, snapshot);
    native_default_failure(error, rollback, &mut plan.usable)
}

fn after_rollback(
    error: ShapeError,
    restored: Result<(), BackendError>,
    usable: &mut bool,
) -> ShapeError {
    match restored {
        Ok(()) => error,
        Err(rollback) => {
            *usable = false;
            ShapeError::Fatal(super::infer(format!(
                "{}; failed to restore logical rows: {rollback}",
                error.into_backend()
            )))
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Epilogue {
    Bias,
    GeluBias,
}

impl Epilogue {
    const fn attribute(self) -> i32 {
        match self {
            Self::Bias => EPILOGUE_BIAS,
            Self::GeluBias => EPILOGUE_GELU_BIAS,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Geometry {
    input: i32,
    width: i32,
    seed_rows: i32,
    initial_rows: i32,
    x_stride: i32,
    weight_stride: i32,
    output_stride: i32,
}

impl Geometry {
    fn contiguous<E: Element>(input: usize, width: usize) -> Result<Self, BackendError> {
        // The format check is explicit even when no LinearPlan describes storage.
        element_type::<E>()?;
        let input = dimension(input)?;
        let width = dimension(width)?;
        let seed_rows = dimension(SEED_ROWS)?;
        Self::checked(input, width, seed_rows, input, input, width)
    }

    fn contiguous_rows<E: Element>(
        input: usize,
        width: usize,
        rows: usize,
    ) -> Result<Self, BackendError> {
        element_type::<E>()?;
        let input = dimension(input)?;
        let width = dimension(width)?;
        let rows = dimension(rows)?;
        Self::checked(input, width, rows, input, input, width)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn from_linear<E: Element>(plan: &LinearPlan<E>) -> Result<Self, BackendError> {
        element_type::<E>()?;
        let x = plan.input_layout();
        let weight = plan.weight_layout();
        let output = plan.output_layout();
        let [rows, input] = *x.shape();
        let [width, weight_input] = *weight.shape();
        let [output_rows, output_width] = *output.shape();
        if input != weight_input || rows != output_rows || width != output_width {
            return Err(math_error(MathError::ShapeMismatch));
        }
        let [x_stride, x_inner] = *x.strides();
        let [weight_stride, weight_inner] = *weight.strides();
        let [output_stride, output_inner] = *output.strides();
        if x_inner != 1 || weight_inner != 1 || output_inner != 1 {
            return Err(math_error(MathError::UnsupportedLayout));
        }
        let initial_rows = dimension(rows)?;
        let mut geometry = Self::checked(
            dimension(input)?,
            dimension(width)?,
            dimension(SEED_ROWS)?,
            dimension(x_stride)?,
            dimension(weight_stride)?,
            dimension(output_stride)?,
        )?;
        geometry.initial_rows = initial_rows;
        Ok(geometry)
    }

    fn checked(
        input: i32,
        width: i32,
        seed_rows: i32,
        x_stride: i32,
        weight_stride: i32,
        output_stride: i32,
    ) -> Result<Self, BackendError> {
        if input == 0 || width == 0 || seed_rows == 0 {
            return Err(math_error(MathError::EmptyMatrix));
        }
        if input < 0 || width < 0 || seed_rows < 0 {
            return Err(math_error(MathError::DimensionOverflow));
        }
        if x_stride < input || weight_stride < input || output_stride < width {
            return Err(math_error(MathError::UnsupportedLayout));
        }
        Ok(Self {
            input,
            width,
            seed_rows,
            initial_rows: seed_rows,
            x_stride,
            weight_stride,
            output_stride,
        })
    }

    fn rows(&self, rows: usize) -> Result<i32, BackendError> {
        if rows == 0 {
            return Err(math_error(MathError::EmptyMatrix));
        }
        dimension(rows)
    }

    fn check_bias(&self, length: usize) -> Result<(), BackendError> {
        if length < self.width as usize {
            return Err(super::infer(format!(
                "bias length {length} is shorter than fused output width {}",
                self.width
            )));
        }
        Ok(())
    }

    fn layouts(&self, rows: i32) -> Result<(Layout<2>, Layout<2>, Layout<2>), BackendError> {
        let rows = rows as usize;
        let input = self.input as usize;
        let width = self.width as usize;
        let x =
            Layout::strided([rows, input], [self.x_stride as usize, 1]).map_err(super::infer)?;
        let weight = Layout::strided([width, input], [self.weight_stride as usize, 1])
            .map_err(super::infer)?;
        let output = Layout::strided([rows, width], [self.output_stride as usize, 1])
            .map_err(super::infer)?;
        Ok((x, weight, output))
    }
}

pub(crate) struct LtPlan<E: Element> {
    api: Arc<Api>,
    context: Arc<Context>,
    stream: Arc<Stream>,
    owner: Arc<()>,
    descriptors_safe: Arc<AtomicBool>,
    op: Descriptor,
    a: Descriptor,
    b: Descriptor,
    y: Descriptor,
    pref: Descriptor,
    algo: Algo,
    base: PreparedAlgorithm,
    small: [Option<PreparedAlgorithm>; SMALL_ROWS],
    small_prepared: bool,
    native_default_cache: Option<Box<NativeDefaultCache>>,
    native_default_bias_pointer: *const c_void,
    native_default_alignment: NativeDefaultAlignment,
    geometry: Geometry,
    epilogue: Epilogue,
    rows: i32,
    workspace: usize,
    usable: bool,
    _element: PhantomData<E>,
}

#[cfg(test)]
impl<E: Element> LtPlan<E> {
    pub(crate) fn logical_rows(&self) -> i32 {
        self.rows
    }

    pub(crate) fn prepared_small_rows(&self) -> usize {
        self.small.iter().flatten().count()
    }
}

// SAFETY: moving a plan transfers exclusive ownership of its host descriptors.
// No descriptor is accessed through a shared cross-thread reference: raw
// pointers keep LtPlan !Sync, and provider use is externally serialized.
// The retained API/context/stream outlive every descriptor; Drop binds that
// context and drains the stream even when it runs on a completion thread.
unsafe impl<E: Element + Send> Send for LtPlan<E> {}

impl<E: Element> Drop for LtPlan<E> {
    fn drop(&mut self) {
        let _ = self.context.bind_to_thread();
        let mut safe = self.descriptors_safe.load(Ordering::Acquire);
        if self.stream.synchronize().is_err() {
            self.descriptors_safe.store(false, Ordering::Release);
            safe = false;
        }
        if !safe {
            // Keep the library and descriptors alive after uncertain completion.
            retain_failed_resources(&self.api, &self.context, &self.stream);
            return;
        }
        let destroyed = destroy_descriptors(
            &self.api.descriptor_destroy,
            &mut self.op,
            &mut self.a,
            &mut self.b,
            &mut self.y,
            &mut self.pref,
        );
        if !destroyed {
            self.descriptors_safe.store(false, Ordering::Release);
            retain_failed_resources(&self.api, &self.context, &self.stream);
        }
    }
}

fn retain_failed_resources(api: &Arc<Api>, context: &Arc<Context>, stream: &Arc<Stream>) {
    std::mem::forget(api.clone());
    std::mem::forget(context.clone());
    std::mem::forget(stream.clone());
}

fn destroy_descriptors(
    destroy: &DescriptorDestructors,
    op: &mut Descriptor,
    a: &mut Descriptor,
    b: &mut Descriptor,
    y: &mut Descriptor,
    pref: &mut Descriptor,
) -> bool {
    let mut succeeded = true;
    for (function, descriptor) in [
        (destroy.preference, pref),
        (destroy.layout, y),
        (destroy.layout, b),
        (destroy.layout, a),
        (destroy.matmul, op),
    ] {
        if !descriptor.is_null() {
            // SAFETY: each pointer was created by the matching descriptor API.
            if unsafe { function(*descriptor) } == STATUS_SUCCESS {
                *descriptor = ptr::null_mut();
            } else {
                succeeded = false;
            }
        }
    }
    succeeded
}

fn select_zero_workspace_algorithm(choices: &[HeuristicResult]) -> Result<Algo, BackendError> {
    choices
        .iter()
        .find(|choice| {
            choice.state == STATUS_SUCCESS && choice.workspace_size == SEED_WORKSPACE_BYTES
        })
        .map(|choice| choice.algo)
        .ok_or_else(|| {
            super::infer("cuBLASLt returned no successful zero-workspace fused algorithm")
        })
}

fn select_small_algorithm(
    choices: &[HeuristicResult],
    workspace_limit: usize,
) -> Result<Algo, BackendError> {
    choices
        .iter()
        .find(|choice| {
            choice.state == STATUS_SUCCESS && choice.workspace_size <= workspace_limit
        })
        .map(|choice| choice.algo)
        .ok_or_else(|| {
            super::infer(format!(
                "cuBLASLt returned no successful small-row algorithm within {workspace_limit} workspace bytes"
            ))
        })
}

fn create_layout(
    api: &Arc<Api>,
    element_type: i32,
    rows: i32,
    columns: i32,
    leading: i32,
    operation: &str,
) -> Result<Descriptor, BackendError> {
    let mut layout = ptr::null_mut();
    let status = unsafe {
        (api.layout_create)(
            &mut layout,
            element_type,
            rows as u64,
            columns as u64,
            leading as i64,
        )
    };
    if status != STATUS_SUCCESS || layout.is_null() {
        if !layout.is_null() {
            unsafe {
                if (api.descriptor_destroy.layout)(layout) != STATUS_SUCCESS {
                    std::mem::forget(api.clone());
                }
            }
        }
        return Err(super::infer(format!(
            "cuBLASLt {operation} status {status}"
        )));
    }
    Ok(layout)
}

fn set_columns<E: Element>(api: &Api, plan: &LtPlan<E>, rows: i32) -> Result<(), BackendError> {
    let columns = rows as u64;
    for (descriptor, operation) in [(plan.b, "logical rows B"), (plan.y, "logical rows Y")] {
        let status = unsafe {
            (api.layout_set)(
                descriptor,
                MATRIX_LAYOUT_COLS,
                &columns as *const u64 as *const c_void,
                size_of::<u64>(),
            )
        };
        if status != STATUS_SUCCESS {
            return Err(super::infer(format!(
                "cuBLASLt {operation} status {status}"
            )));
        }
    }
    Ok(())
}

fn set_desc<T>(
    api: &Api,
    descriptor: Descriptor,
    attribute: i32,
    value: &T,
    operation: &str,
) -> Result<(), BackendError> {
    let status = unsafe {
        (api.desc_set)(
            descriptor,
            attribute,
            value as *const T as *const c_void,
            size_of::<T>(),
        )
    };
    check(status, operation)
}

fn set_preference<T>(
    api: &Api,
    descriptor: Descriptor,
    attribute: i32,
    value: &T,
    operation: &str,
) -> Result<(), BackendError> {
    let status = unsafe {
        (api.preference_set)(
            descriptor,
            attribute,
            value as *const T as *const c_void,
            size_of::<T>(),
        )
    };
    check(status, operation)
}

fn check(status: i32, operation: &str) -> Result<(), BackendError> {
    if status == STATUS_SUCCESS {
        Ok(())
    } else {
        Err(super::infer(format!(
            "cuBLASLt {operation} status {status}"
        )))
    }
}

fn dimension(value: usize) -> Result<i32, BackendError> {
    i32::try_from(value).map_err(|_| math_error(MathError::DimensionOverflow))
}

fn element_type<E: Element>() -> Result<i32, BackendError> {
    match E::FORMAT {
        ScalarFormat::Bf16 => Ok(CUDA_R_16BF),
        ScalarFormat::F16 => Ok(CUDA_R_16F),
        _ => Err(math_error(MathError::UnsupportedFormat)),
    }
}

fn math_error(error: MathError) -> BackendError {
    super::infer(error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_check_classification_preserves_fatal_status_precedence() {
        for (status, state, workspace, rejected) in [
            (STATUS_NOT_SUPPORTED, STATUS_SUCCESS, 0, true),
            (STATUS_SUCCESS, STATUS_NOT_SUPPORTED, 0, true),
            (STATUS_SUCCESS, STATUS_SUCCESS, WORKSPACE_BYTES + 1, true),
            (7, STATUS_NOT_SUPPORTED, WORKSPACE_BYTES + 1, false),
            (STATUS_SUCCESS, 7, WORKSPACE_BYTES + 1, false),
        ] {
            let checked = HeuristicResult {
                state,
                workspace_size: workspace,
                ..HeuristicResult::default()
            };
            let error =
                checked_workspace(status, &checked, || "M=65 K=384 N=384".into()).unwrap_err();
            assert_eq!(matches!(error, ShapeError::Rejected(_)), rejected);
        }
        let checked = HeuristicResult {
            workspace_size: WORKSPACE_BYTES,
            ..HeuristicResult::default()
        };
        let formatted = Cell::new(false);
        assert_eq!(
            checked_workspace(STATUS_SUCCESS, &checked, || {
                formatted.set(true);
                "unused geometry".into()
            })
            .unwrap(),
            WORKSPACE_BYTES
        );
        assert!(
            !formatted.get(),
            "valid shapes must not allocate an error string"
        );
    }

    #[test]
    fn recoverable_shape_rejection_requires_successful_descriptor_rollback() {
        let rejected = || {
            checked_workspace(STATUS_NOT_SUPPORTED, &HeuristicResult::default(), || {
                "injected shape".into()
            })
            .unwrap_err()
        };
        let mut usable = true;
        assert!(matches!(
            after_rollback(rejected(), Ok(()), &mut usable),
            ShapeError::Rejected(_)
        ));
        assert!(usable);
        let failed = after_rollback(
            rejected(),
            Err(super::super::infer("injected descriptor restore failure")),
            &mut usable,
        );
        assert!(matches!(failed, ShapeError::Fatal(_)));
        assert!(!usable);

        // The same numeric status from a descriptor API is never a shape
        // rejection, even if restoring that descriptor succeeds.
        let descriptor_error: ShapeError = super::super::infer("descriptor API status 15").into();
        let mut usable = true;
        assert!(matches!(
            after_rollback(descriptor_error, Ok(()), &mut usable),
            ShapeError::Fatal(_)
        ));
        assert!(usable);
    }

    #[test]
    fn rejected_partial_plan_refresh_recovers_failed_and_previous_shapes() {
        for recovered_rows in [65usize, 3] {
            let mut cached = Some(3);
            let mut descriptor_rows = [3usize; 4];
            let mut usable = true;
            let failed = crate::math::refresh_plan_rows(&mut cached, 65, || {
                for (index, rows) in descriptor_rows.iter_mut().enumerate() {
                    let previous = *rows;
                    *rows = 65;
                    let status = if index == 1 {
                        STATUS_NOT_SUPPORTED
                    } else {
                        STATUS_SUCCESS
                    };
                    if let Err(error) =
                        checked_workspace(status, &HeuristicResult::default(), || {
                            "injected partial refresh".into()
                        })
                    {
                        *rows = previous;
                        return Err(after_rollback(error, Ok(()), &mut usable));
                    }
                }
                Ok(())
            });
            assert!(matches!(failed, Err(ShapeError::Rejected(_))));
            assert_eq!(descriptor_rows, [65, 3, 3, 3]);
            assert_eq!(cached, None);
            assert!(usable);
            crate::math::refresh_plan_rows(&mut cached, recovered_rows, || {
                descriptor_rows.fill(recovered_rows);
                Ok::<_, ShapeError>(())
            })
            .unwrap();
            assert_eq!(descriptor_rows, [recovered_rows; 4]);
            assert_eq!(cached, Some(recovered_rows));
        }
    }

    #[test]
    fn provider_rejection_hook_is_shape_specific_and_consumed_once() {
        let hook = Cell::new(Some(ShapeRejection {
            rows: 65,
            successful_checks_left: 1,
        }));
        let other_provider = Cell::new(None);
        assert!(!inject_shape_rejection(&hook, 3));
        assert!(!inject_shape_rejection(&hook, 65));
        assert!(!inject_shape_rejection(&other_provider, 65));
        assert!(inject_shape_rejection(&hook, 65));
        assert!(!inject_shape_rejection(&hook, 65));
    }
    use crate::tensor::{Bf16, F16, F32};
    use std::{
        cell::RefCell,
        mem::{align_of, offset_of, size_of},
    };

    thread_local! {
        static DESTROYED: RefCell<Vec<(u8, usize)>> = const { RefCell::new(Vec::new()) };
    }

    fn record_destroy(kind: u8, descriptor: Descriptor) -> i32 {
        DESTROYED.with(|calls| calls.borrow_mut().push((kind, descriptor as usize)));
        STATUS_SUCCESS
    }
    unsafe extern "C" fn mock_matmul_destroy(descriptor: Descriptor) -> i32 {
        record_destroy(1, descriptor)
    }
    unsafe extern "C" fn mock_layout_destroy(descriptor: Descriptor) -> i32 {
        record_destroy(2, descriptor)
    }
    unsafe extern "C" fn mock_preference_destroy(descriptor: Descriptor) -> i32 {
        record_destroy(3, descriptor)
    }
    unsafe extern "C" fn mock_failed_layout_destroy(descriptor: Descriptor) -> i32 {
        record_destroy(2, descriptor);
        STATUS_NOT_SUPPORTED
    }

    #[test]
    fn normal_and_partial_plans_use_the_matching_descriptor_destructors() {
        let destroy = DescriptorDestructors {
            matmul: mock_matmul_destroy,
            layout: mock_layout_destroy,
            preference: mock_preference_destroy,
        };
        // Creation order is op, A, B, Y, preference; every prefix can be owned
        // when preparation fails. Mock pointers are identity tags, never read.
        let tags = [
            (1, 0x1000usize),
            (2, 0x1100),
            (2, 0x1200),
            (2, 0x1300),
            (3, 0x1400),
        ];
        for owned in 0..=tags.len() {
            DESTROYED.with(|calls| calls.borrow_mut().clear());
            let mut descriptors = [ptr::null_mut(); 5];
            for (slot, (_, address)) in descriptors.iter_mut().zip(tags).take(owned) {
                *slot = address as Descriptor;
            }
            let [op, a, b, y, pref] = &mut descriptors;
            assert!(destroy_descriptors(&destroy, op, a, b, y, pref));
            let expected = tags[..owned].iter().copied().rev().collect::<Vec<_>>();
            DESTROYED.with(|calls| assert_eq!(*calls.borrow(), expected));
            assert!(descriptors.iter().all(|descriptor| descriptor.is_null()));
        }
    }

    #[test]
    fn failed_descriptor_cleanup_reports_failure_and_still_releases_other_kinds() {
        let destroy = DescriptorDestructors {
            matmul: mock_matmul_destroy,
            layout: mock_failed_layout_destroy,
            preference: mock_preference_destroy,
        };
        DESTROYED.with(|calls| calls.borrow_mut().clear());
        let (mut op, mut a, mut b, mut y, mut pref) = (
            0x1000usize as Descriptor,
            ptr::null_mut(),
            0x1200usize as Descriptor,
            ptr::null_mut(),
            0x1400usize as Descriptor,
        );
        assert!(!destroy_descriptors(
            &destroy, &mut op, &mut a, &mut b, &mut y, &mut pref
        ));
        assert!(op.is_null() && a.is_null() && y.is_null() && pref.is_null());
        assert_eq!(b as usize, 0x1200);
        DESTROYED
            .with(|calls| assert_eq!(*calls.borrow(), vec![(3, 0x1400), (2, 0x1200), (1, 0x1000)]));
    }

    #[test]
    fn native_default_cache_reuses_exact_keys_and_stays_bounded() {
        let key = |rows: i32| NativeDefaultCacheKey {
            rows,
            input: 768,
            width: 9,
            x_stride: 768,
            weight_stride: 768,
            output_stride: 9,
            element_type: CUDA_R_16BF,
            epilogue: Epilogue::Bias,
            workspace_preference: NATIVE_DEFAULT_WORKSPACE_PREFERENCE_BYTES,
            alignment: NativeDefaultAlignment {
                a: 256,
                b: 128,
                c: 64,
                d: 64,
            },
        };
        let prepared = |value: u64| PreparedAlgorithm {
            algo: Algo { data: [value; 8] },
            workspace: value as usize,
        };
        let mut cache = NativeDefaultCache::default();
        cache.insert(key(1), prepared(11));
        assert_eq!(cache.get(&key(1)).unwrap().algo.data, [11; 8]);
        cache.insert(key(1), prepared(12));
        assert_eq!(cache.get(&key(1)).unwrap().algo.data, [12; 8]);
        assert_eq!(cache.len(), 1);
        for rows in 2..=NATIVE_DEFAULT_CACHE_ENTRIES as i32 + 5 {
            cache.insert(key(rows), prepared(rows as u64));
        }
        assert_eq!(cache.len(), NATIVE_DEFAULT_CACHE_ENTRIES);
        assert_eq!(
            cache
                .get(&key(NATIVE_DEFAULT_CACHE_ENTRIES as i32 + 5))
                .unwrap()
                .algo
                .data,
            [(NATIVE_DEFAULT_CACHE_ENTRIES as u64 + 5); 8]
        );
        assert!(cache.get(&key(1)).is_none(), "oldest metadata was evicted");
        cache.insert(key(1), prepared(13));
        assert_eq!(cache.len(), NATIVE_DEFAULT_CACHE_ENTRIES);
        assert_eq!(cache.get(&key(1)).unwrap().algo.data, [13; 8]);
    }

    #[test]
    fn native_default_pending_reuses_cold_keys_without_publishing_and_stays_bounded() {
        let key = |rows: i32| NativeDefaultCacheKey {
            rows,
            input: 768,
            width: 9,
            x_stride: 768,
            weight_stride: 768,
            output_stride: 9,
            element_type: CUDA_R_16BF,
            epilogue: Epilogue::Bias,
            workspace_preference: NATIVE_DEFAULT_WORKSPACE_PREFERENCE_BYTES,
            alignment: NativeDefaultAlignment {
                a: 256,
                b: 128,
                c: 64,
                d: 64,
            },
        };
        let prepared = |value: u64| PreparedAlgorithm {
            algo: Algo { data: [value; 8] },
            workspace: value as usize,
        };
        let mut pending = NativeDefaultPending::default();
        assert!(pending.get(&key(3)).is_none());
        pending.insert(key(3), prepared(33));
        assert_eq!(pending.get(&key(3)).unwrap().algo.data, [33; 8]);
        pending.insert(key(3), prepared(34));
        assert_eq!(pending.get(&key(3)).unwrap().algo.data, [34; 8]);
        assert_eq!(pending.len(), 1);
        for rows in 4..=NATIVE_DEFAULT_CACHE_ENTRIES as i32 + 8 {
            pending.insert(key(rows), prepared(rows as u64));
        }
        assert_eq!(pending.len(), NATIVE_DEFAULT_CACHE_ENTRIES);
        assert_eq!(
            pending
                .get(&key(NATIVE_DEFAULT_CACHE_ENTRIES as i32 + 8))
                .unwrap()
                .algo
                .data,
            [(NATIVE_DEFAULT_CACHE_ENTRIES as u64 + 8); 8]
        );
        pending.clear();
        assert_eq!(pending.len(), 0);
        assert!(pending.get(&key(3)).is_none());
    }

    #[test]
    fn native_default_cache_does_not_publish_rejected_pending_entry_or_evict_existing() {
        let key = |rows: i32, alignment: NativeDefaultAlignment| NativeDefaultCacheKey {
            rows,
            input: 768,
            width: 9,
            x_stride: 768,
            weight_stride: 768,
            output_stride: 9,
            element_type: CUDA_R_16BF,
            epilogue: Epilogue::Bias,
            workspace_preference: NATIVE_DEFAULT_WORKSPACE_PREFERENCE_BYTES,
            alignment,
        };
        let prepared = |value: u64| PreparedAlgorithm {
            algo: Algo { data: [value; 8] },
            workspace: value as usize,
        };
        let first_alignment = NativeDefaultAlignment {
            a: 256,
            b: 256,
            c: 256,
            d: 256,
        };
        let second_alignment = NativeDefaultAlignment {
            a: 128,
            b: 64,
            c: 64,
            d: 64,
        };
        let third_alignment = NativeDefaultAlignment {
            a: 64,
            b: 64,
            c: 32,
            d: 32,
        };
        let first = key(1, first_alignment);
        let second = key(3, second_alignment);
        let rejected = key(5, third_alignment);
        let mut cache = NativeDefaultCache::default();
        cache.insert(first, prepared(11));
        cache.insert(second, prepared(22));
        let before_first = cache.get(&first).unwrap();
        let before_second = cache.get(&second).unwrap();

        // A rejected cold miss never publishes its pending entry and must not
        // disturb either previously cached shape/alignment tuple.
        assert!(cache.get(&rejected).is_none());
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get(&first).unwrap(), before_first);
        assert_eq!(cache.get(&second).unwrap(), before_second);

        cache.insert(rejected, prepared(55));
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.get(&first).unwrap(), before_first);
        assert_eq!(cache.get(&second).unwrap(), before_second);
        assert_eq!(cache.get(&rejected).unwrap().algo.data, [55; 8]);
    }

    #[test]
    fn native_default_rollback_restores_every_mutable_stage_and_fails_closed() {
        let snapshot = NativeDefaultSnapshot {
            rows: 3,
            algo: Algo { data: [7; 8] },
            workspace: 123,
            bias_pointer: 0x2000usize as *const c_void,
            alignment: NativeDefaultAlignment {
                a: 256,
                b: 128,
                c: 64,
                d: 64,
            },
        };
        let dirty = NativeDefaultMutableState {
            rows: 5,
            algo: Algo { data: [9; 8] },
            workspace: 999,
            bias_pointer: 0x3000usize as *const c_void,
            alignment: NativeDefaultAlignment {
                a: 64,
                b: 32,
                c: 16,
                d: 8,
            },
        };
        for failing_stage in 0..3 {
            let mut state = dirty;
            let result = run_native_default_rollback(
                &mut state,
                snapshot,
                || {
                    if failing_stage == 0 {
                        Err(super::super::infer("injected column rollback failure"))
                    } else {
                        Ok(())
                    }
                },
                || {
                    if failing_stage == 1 {
                        Err(super::super::infer("injected bias rollback failure"))
                    } else {
                        Ok(())
                    }
                },
                || {
                    if failing_stage == 2 {
                        Err(super::super::infer("injected alignment rollback failure"))
                    } else {
                        Ok(())
                    }
                },
            );
            assert!(result.is_err());
            assert_eq!(state.rows, snapshot.rows);
            assert_eq!(state.algo, snapshot.algo);
            assert_eq!(state.workspace, snapshot.workspace);
            assert_eq!(state.bias_pointer, snapshot.bias_pointer);
            assert_eq!(state.alignment, snapshot.alignment);

            let mut usable = true;
            let failure = native_default_failure(
                ShapeError::Rejected(super::super::infer("injected cold miss")),
                result,
                &mut usable,
            );
            assert!(matches!(failure, ShapeError::Fatal(_)));
            assert!(!usable);
        }

        let mut state = dirty;
        let result =
            run_native_default_rollback(&mut state, snapshot, || Ok(()), || Ok(()), || Ok(()));
        assert!(result.is_ok());
        assert_eq!(state.rows, snapshot.rows);
        assert_eq!(state.algo, snapshot.algo);
        assert_eq!(state.workspace, snapshot.workspace);
        assert_eq!(state.bias_pointer, snapshot.bias_pointer);
        assert_eq!(state.alignment, snapshot.alignment);

        let mut usable = true;
        let failure = native_default_failure(
            ShapeError::Rejected(super::super::infer("injected cold miss")),
            result,
            &mut usable,
        );
        assert!(matches!(failure, ShapeError::Rejected(_)));
        assert!(usable);
    }

    #[test]
    fn native_default_alignment_matches_the_upstream_256_to_one_halving_rule() {
        assert_eq!(get_alignment(0), 256);
        assert_eq!(get_alignment(4096), 256);
        assert_eq!(get_alignment(257), 1);
        assert_eq!(get_alignment(2 * 256 + 128), 128);
        assert_eq!(get_alignment(64 * 3), 64);
        assert_eq!(get_alignment(31), 1);
    }

    #[test]
    fn native_default_backing_and_preference_are_distinct_and_seeded() {
        assert_eq!(NATIVE_DEFAULT_WORKSPACE_BYTES, 32 * 1024 * 1024);
        assert_eq!(NATIVE_DEFAULT_WORKSPACE_PREFERENCE_BYTES, 1024 * 1024);
        assert_eq!(NATIVE_DEFAULT_SEED_ROWS, SEED_ROWS);
        const {
            assert!(NATIVE_DEFAULT_WORKSPACE_BYTES > NATIVE_DEFAULT_WORKSPACE_PREFERENCE_BYTES);
            assert!(WORKSPACE_BYTES < NATIVE_DEFAULT_WORKSPACE_PREFERENCE_BYTES);
        }
    }

    #[test]
    fn selects_a_successful_zero_workspace_candidate_after_ineligible_results() {
        let choices = [
            HeuristicResult {
                algo: Algo { data: [1; 8] },
                workspace_size: 4096,
                ..HeuristicResult::default()
            },
            HeuristicResult {
                algo: Algo { data: [2; 8] },
                state: STATUS_NOT_SUPPORTED,
                ..HeuristicResult::default()
            },
            HeuristicResult {
                algo: Algo { data: [3; 8] },
                ..HeuristicResult::default()
            },
        ];
        assert_eq!(
            select_zero_workspace_algorithm(&choices).unwrap(),
            choices[2].algo
        );
        assert!(select_zero_workspace_algorithm(&choices[..2]).is_err());
    }

    #[test]
    fn small_row_algorithm_is_first_success_within_the_workspace_bound() {
        let choices = [
            HeuristicResult {
                algo: Algo { data: [1; 8] },
                workspace_size: WORKSPACE_BYTES + 1,
                ..HeuristicResult::default()
            },
            HeuristicResult {
                algo: Algo { data: [2; 8] },
                workspace_size: WORKSPACE_BYTES + 8,
                ..HeuristicResult::default()
            },
            HeuristicResult {
                algo: Algo { data: [3; 8] },
                state: STATUS_NOT_SUPPORTED,
                ..HeuristicResult::default()
            },
            HeuristicResult {
                algo: Algo { data: [4; 8] },
                workspace_size: 30_736,
                ..HeuristicResult::default()
            },
        ];
        assert_eq!(
            select_small_algorithm(&choices, WORKSPACE_BYTES).unwrap(),
            choices[3].algo
        );
        assert!(select_small_algorithm(&choices[..2], WORKSPACE_BYTES).is_err());
        assert!(select_small_algorithm(&[], WORKSPACE_BYTES).is_err());
    }

    #[test]
    fn small_row_cache_covers_only_actual_rows_one_through_thirty_two() {
        let base = PreparedAlgorithm {
            algo: Algo { data: [99; 8] },
            workspace: 0,
        };
        let mut small = [None; SMALL_ROWS];
        for (index, prepared) in small.iter_mut().enumerate() {
            *prepared = Some(PreparedAlgorithm {
                algo: Algo {
                    data: [index as u64 + 1; 8],
                },
                workspace: index + 1,
            });
        }
        assert_eq!(prepared_for_rows(base, &small, true, 0).unwrap(), base);
        for rows in 1..=SMALL_ROWS {
            let actual = prepared_for_rows(base, &small, true, rows).unwrap();
            assert_eq!(actual.algo.data, [rows as u64; 8]);
            assert_eq!(actual.workspace, rows);
        }
        assert_eq!(
            prepared_for_rows(base, &small, true, SMALL_ROWS + 1).unwrap(),
            base
        );
        assert_eq!(
            prepared_for_rows(base, &small, true, usize::MAX).unwrap(),
            base
        );
        let mut hole = small;
        hole[4] = None;
        assert!(prepared_for_rows(base, &hole, true, 5).is_err());
        assert_eq!(prepared_for_rows(base, &hole, false, 5).unwrap(), base);
    }

    #[test]
    fn prepared_algorithm_check_receives_the_selected_algorithm() {
        let selected = Algo { data: [7; 8] };
        let seen = Cell::new(None);
        let prepared = check_algorithm(selected, |algorithm| {
            seen.set(Some(*algorithm));
            Ok(30_736)
        })
        .unwrap();
        assert_eq!(seen.get(), Some(selected));
        assert_eq!(prepared.algo, selected);
        assert_eq!(prepared.workspace, 30_736);
        assert!(check_algorithm(selected, |_| {
            Err(ShapeError::Rejected(super::super::infer(
                "mock provider rejected the selected algorithm",
            )))
        })
        .is_err());
    }

    #[test]
    fn rejected_algorithm_refresh_restores_algorithm_and_workspace() {
        let previous = PreparedAlgorithm {
            algo: Algo { data: [7; 8] },
            workspace: 12,
        };
        let target = PreparedAlgorithm {
            algo: Algo { data: [8; 8] },
            workspace: 30_736,
        };
        let mut algo = target.algo;
        let mut workspace = target.workspace;
        restore_prepared(previous, &mut algo, &mut workspace);
        assert_eq!(algo, previous.algo);
        assert_eq!(workspace, previous.workspace);
    }

    #[test]
    fn abi_matches_the_cuda_13_1_header_probe() {
        assert_eq!(STATUS_SUCCESS, 0);
        assert_eq!(STATUS_NOT_SUPPORTED, 15);
        assert_eq!(CUBLAS_OP_T, 1);
        assert_eq!(CUBLAS_COMPUTE_32F, 68);
        assert_eq!(CUDA_R_32F, 0);
        assert_eq!(CUDA_R_16F, 2);
        assert_eq!(CUDA_R_16BF, 14);
        assert_eq!(MATRIX_LAYOUT_COLS, 3);
        assert_eq!(MATMUL_DESC_POINTER_MODE, 2);
        assert_eq!(MATMUL_DESC_TRANSA, 3);
        assert_eq!(MATMUL_DESC_TRANSB, 4);
        assert_eq!(MATMUL_DESC_EPILOGUE, 7);
        assert_eq!(MATMUL_DESC_BIAS_POINTER, 8);
        assert_eq!(MATMUL_DESC_BIAS_DATA_TYPE, 26);
        assert_eq!(PREFERENCE_MAX_WORKSPACE_BYTES, 1);
        assert_eq!(EPILOGUE_BIAS, 4);
        assert_eq!(EPILOGUE_GELU_BIAS, 36);
        assert_eq!((size_of::<Algo>(), align_of::<Algo>()), (64, 8));
        assert_eq!(
            (size_of::<HeuristicResult>(), align_of::<HeuristicResult>()),
            (96, 8)
        );
        assert_eq!(offset_of!(HeuristicResult, algo), 0);
        assert_eq!(offset_of!(HeuristicResult, workspace_size), 64);
        assert_eq!(offset_of!(HeuristicResult, state), 72);
        assert_eq!(offset_of!(HeuristicResult, waves_count), 76);
        assert_eq!(offset_of!(HeuristicResult, reserved), 80);
    }

    #[test]
    fn geometry_preserves_linear_padding_and_rejects_bad_geometry() {
        let linear = LinearPlan::<Bf16>::new(
            Layout::strided([7, 13], [17, 1]).unwrap(),
            Layout::strided([5, 13], [19, 1]).unwrap(),
            Layout::strided([7, 5], [9, 1]).unwrap(),
        )
        .unwrap();
        let geometry = Geometry::from_linear(&linear).unwrap();
        assert_eq!(geometry.initial_rows, 7);
        assert_eq!(
            (
                geometry.input,
                geometry.width,
                geometry.seed_rows,
                geometry.x_stride,
                geometry.weight_stride,
                geometry.output_stride,
            ),
            (13, 5, 128, 17, 19, 9)
        );
        let (x, weight, output) = geometry.layouts(5).unwrap();
        assert_eq!((x.shape(), x.strides()), (&[5, 13], &[17, 1]));
        assert_eq!((weight.shape(), weight.strides()), (&[5, 13], &[19, 1]));
        assert_eq!((output.shape(), output.strides()), (&[5, 5], &[9, 1]));

        let contiguous = Geometry::contiguous::<F16>(13, 5).unwrap();
        assert_eq!(contiguous.initial_rows, SEED_ROWS as i32);
        assert_eq!(
            (
                contiguous.input,
                contiguous.width,
                contiguous.seed_rows,
                contiguous.x_stride,
                contiguous.weight_stride,
                contiguous.output_stride,
            ),
            (13, 5, 128, 13, 13, 5)
        );
        assert!(Geometry::contiguous::<Bf16>(0, 5).is_err());
        assert!(Geometry::contiguous::<Bf16>(i32::MAX as usize + 1, 5).is_err());
        assert!(geometry.rows(0).is_err());
        assert!(geometry.rows(i32::MAX as usize + 1).is_err());
        assert!(geometry.check_bias(5).is_ok());
        assert!(geometry.check_bias(4).is_err());
    }

    #[test]
    fn only_bf16_and_f16_claim_a_cublaslt_element_type() {
        assert_eq!(element_type::<Bf16>().unwrap(), CUDA_R_16BF);
        assert_eq!(element_type::<F16>().unwrap(), CUDA_R_16F);
        assert!(matches!(
            element_type::<F32>(),
            Err(BackendError::Inference(_))
        ));
    }

    #[test]
    fn provider_and_exclusive_plans_can_move_to_model_and_completion_threads() {
        fn assert_send<T: Send>() {}
        assert_send::<LtProvider>();
        assert_send::<LtPlan<Bf16>>();
        assert_send::<LtPlan<F16>>();
    }

    trait FixtureElement: Element {
        fn encode(value: f32) -> Self::Storage;
        fn decode(value: Self::Storage) -> f32;
        const TOLERANCE: f32;
    }
    impl FixtureElement for Bf16 {
        fn encode(value: f32) -> Bf16 {
            Bf16::from_bits(half::bf16::from_f32(value).to_bits())
        }
        fn decode(value: Bf16) -> f32 {
            half::bf16::from_bits(value.to_bits()).to_f32()
        }
        const TOLERANCE: f32 = 0.012;
    }
    impl FixtureElement for F16 {
        fn encode(value: f32) -> F16 {
            F16::from_bits(half::f16::from_f32(value).to_bits())
        }
        fn decode(value: F16) -> f32 {
            half::f16::from_bits(value.to_bits()).to_f32()
        }
        const TOLERANCE: f32 = 0.0025;
    }

    const GPU_K: usize = 19;
    const GPU_N: usize = 13;
    const GPU_X_STRIDE: usize = GPU_K + 3;
    const GPU_W_STRIDE: usize = GPU_K + 5;
    const GPU_Y_STRIDE: usize = GPU_N + 4;
    const GPU_MAX_ROWS: usize = 512;
    const GPU_ROWS: [usize; 5] = [1, 7, 128, 129, 512];

    fn values<E: FixtureElement>(count: usize, seed: usize, scale: f32) -> Vec<E::Storage> {
        (0..count)
            .map(|i| E::encode((((i * 37 + seed) % 101) as f32 / 50.0 - 1.0) * scale))
            .collect()
    }

    fn reference<E: FixtureElement>(
        epilogue: Epilogue,
        x: &[E::Storage],
        weight: &[E::Storage],
        bias: &[E::Storage],
        row: usize,
        column: usize,
    ) -> f32 {
        let mut sum = 0.0f32;
        for input in 0..GPU_K {
            sum += E::decode(x[row * GPU_X_STRIDE + input])
                * E::decode(weight[column * GPU_W_STRIDE + input]);
        }
        let value = sum + E::decode(bias[column]);
        match epilogue {
            Epilogue::Bias => value,
            Epilogue::GeluBias => {
                let inner = 0.797_884_6 * value * (1.0 + 0.044715 * value * value);
                0.5 * value * (1.0 + inner.tanh())
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_gpu_epilogue<E: FixtureElement>(
        provider: &LtProvider,
        stream: &Arc<Stream>,
        x: &DeviceBuffer<E::Storage>,
        weight: &DeviceBuffer<E::Storage>,
        bias_a: &DeviceBuffer<E::Storage>,
        bias_b: &DeviceBuffer<E::Storage>,
        plan: &mut LtPlan<E>,
        epilogue: Epilogue,
        host_x: &[E::Storage],
        host_weight: &[E::Storage],
        host_bias_a: &[E::Storage],
        host_bias_b: &[E::Storage],
    ) {
        let seed_algorithm = plan.algo;
        let sentinel = E::encode(-99.0);
        let sentinel_value = E::decode(sentinel);
        for &rows in &GPU_ROWS {
            provider.set_logical_rows(plan, rows).unwrap();
            assert_eq!(
                plan.algo, seed_algorithm,
                "row update replaced opaque algorithm"
            );
            let mut output_a =
                DeviceBuffer::from_host(stream, &[sentinel; GPU_MAX_ROWS * GPU_Y_STRIDE]).unwrap();
            let mut output_b =
                DeviceBuffer::from_host(stream, &[sentinel; GPU_MAX_ROWS * GPU_Y_STRIDE]).unwrap();
            // SAFETY: the fixture binds no other context, the buffers stay alive,
            // and both host copies drain the stream before assertions.
            unsafe {
                provider
                    .enqueue(plan, x, weight, bias_a, &mut output_a)
                    .unwrap();
                provider
                    .enqueue(plan, x, weight, bias_b, &mut output_b)
                    .unwrap();
            }
            assert_eq!(
                plan.algo, seed_algorithm,
                "matmul replaced opaque algorithm"
            );
            let output_a = output_a.to_host_vec(stream).unwrap();
            let output_b = output_b.to_host_vec(stream).unwrap();
            let mut bias_changed_output = false;
            for row in 0..rows {
                for column in 0..GPU_N {
                    let expected_a =
                        reference::<E>(epilogue, host_x, host_weight, host_bias_a, row, column);
                    let expected_b =
                        reference::<E>(epilogue, host_x, host_weight, host_bias_b, row, column);
                    let actual_a = E::decode(output_a[row * GPU_Y_STRIDE + column]);
                    let actual_b = E::decode(output_b[row * GPU_Y_STRIDE + column]);
                    assert!(actual_a.is_finite() && actual_b.is_finite());
                    assert!(
                        (actual_a - expected_a).abs() <= E::TOLERANCE * expected_a.abs().max(1.0),
                        "bias A M={rows} {row},{column}: {actual_a} vs {expected_a}"
                    );
                    assert!(
                        (actual_b - expected_b).abs() <= E::TOLERANCE * expected_b.abs().max(1.0),
                        "bias B M={rows} {row},{column}: {actual_b} vs {expected_b}"
                    );
                    bias_changed_output |= (actual_a - actual_b).abs() > E::TOLERANCE;
                }
                for column in GPU_N..GPU_Y_STRIDE {
                    assert_eq!(
                        E::decode(output_a[row * GPU_Y_STRIDE + column]),
                        sentinel_value,
                        "bias A wrote padding at M={rows} {row},{column}"
                    );
                    assert_eq!(
                        E::decode(output_b[row * GPU_Y_STRIDE + column]),
                        sentinel_value,
                        "bias B wrote padding at M={rows} {row},{column}"
                    );
                }
            }
            assert!(
                bias_changed_output,
                "different bias pointers produced identical outputs at M={rows}"
            );
        }

        let previous_rows = plan.rows;
        assert!(provider.set_logical_rows(plan, 0).is_err());
        assert_eq!(plan.rows, previous_rows);

        let short_bias = DeviceBuffer::from_host(stream, &[E::encode(1.0); GPU_N - 1]).unwrap();
        let mut untouched =
            DeviceBuffer::from_host(stream, &[sentinel; GPU_MAX_ROWS * GPU_Y_STRIDE]).unwrap();
        // SAFETY: all views are valid and the rejected bias length returns before
        // any descriptor or launch effect.
        unsafe {
            assert!(provider
                .enqueue(plan, x, weight, &short_bias, &mut untouched)
                .is_err());
        }
        let untouched = untouched.to_host_vec(stream).unwrap();
        assert!(untouched
            .iter()
            .all(|value| E::decode(*value) == sentinel_value));
    }

    fn gpu_fixture<E: FixtureElement>() {
        let context = Context::new(0).unwrap();
        let stream = context.new_stream().unwrap();
        let provider = LtProvider::new(context.clone(), stream.clone()).unwrap();
        let linear = LinearPlan::<E>::new(
            Layout::strided([SEED_ROWS, GPU_K], [GPU_X_STRIDE, 1]).unwrap(),
            Layout::strided([GPU_N, GPU_K], [GPU_W_STRIDE, 1]).unwrap(),
            Layout::strided([SEED_ROWS, GPU_N], [GPU_Y_STRIDE, 1]).unwrap(),
        )
        .unwrap();
        let mut bias_plan = provider.prepare_bias(&linear).unwrap();
        let mut gelu_plan = provider.prepare_gelu_bias(&linear).unwrap();
        let host_x = values::<E>(GPU_MAX_ROWS * GPU_X_STRIDE, 17, 0.5);
        let host_weight = values::<E>(GPU_N * GPU_W_STRIDE, 911, 0.0625);
        let host_bias_a = values::<E>(GPU_N, 1919, 0.125);
        let host_bias_b = values::<E>(GPU_N, 2027, 0.25);
        let x = DeviceBuffer::from_host(&stream, &host_x).unwrap();
        let weight = DeviceBuffer::from_host(&stream, &host_weight).unwrap();
        let bias_a = DeviceBuffer::from_host(&stream, &host_bias_a).unwrap();
        let bias_b = DeviceBuffer::from_host(&stream, &host_bias_b).unwrap();
        run_gpu_epilogue(
            &provider,
            &stream,
            &x,
            &weight,
            &bias_a,
            &bias_b,
            &mut bias_plan,
            Epilogue::Bias,
            &host_x,
            &host_weight,
            &host_bias_a,
            &host_bias_b,
        );
        run_gpu_epilogue(
            &provider,
            &stream,
            &x,
            &weight,
            &bias_a,
            &bias_b,
            &mut gelu_plan,
            Epilogue::GeluBias,
            &host_x,
            &host_weight,
            &host_bias_a,
            &host_bias_b,
        );
    }

    #[test]
    #[ignore = "requires CUDA 13 GPU runtime"]
    fn gpu_fused_bf16_changing_rows_same_algo_and_bias_pointers() {
        gpu_fixture::<Bf16>();
    }

    #[test]
    #[ignore = "requires CUDA 13 GPU runtime"]
    fn gpu_fused_f16_changing_rows_same_algo_and_bias_pointers() {
        gpu_fixture::<F16>();
    }

    #[test]
    #[ignore = "requires CUDA 13 GPU runtime"]
    fn gpu_native_small_rows_select_prepared_algorithms_and_share_workspace() {
        const INPUT: usize = 3072;
        const WIDTH: usize = 768;
        const MAX_ROWS: usize = 128;
        let context = Context::new(0).unwrap();
        let stream = context.new_stream().unwrap();
        let provider = LtProvider::new(context.clone(), stream.clone()).unwrap();
        let workspace_pointer = provider.core.workspace.pointer;
        stream.synchronize().unwrap();
        let mut workspace = vec![0xffu8; WORKSPACE_BYTES];
        // SAFETY: the provider stream is drained, the allocation covers the full
        // workspace, and the destination is a live host vector of that length.
        unsafe {
            cuda_core::memory::memcpy_dtoh_async(
                workspace.as_mut_ptr(),
                workspace_pointer,
                WORKSPACE_BYTES,
                stream.cu_stream(),
            )
        }
        .unwrap();
        stream.synchronize().unwrap();
        let first_nonzero = workspace.iter().position(|byte| *byte != 0);
        assert!(
            first_nonzero.is_none(),
            "workspace {workspace_pointer:#x} was not zeroed at construction: {first_nonzero:?}"
        );
        let mut bias_plan = provider
            .prepare_native_small_bias_shape::<Bf16>(INPUT, WIDTH)
            .unwrap();
        let mut gelu_plan = provider
            .prepare_native_small_gelu_bias_shape::<Bf16>(INPUT, WIDTH)
            .unwrap();
        for plan in [&bias_plan, &gelu_plan] {
            assert_eq!(plan.prepared_small_rows(), SMALL_ROWS);
        }
        let host_x = values::<Bf16>(MAX_ROWS * INPUT, 17, 0.5);
        let host_weight = values::<Bf16>(WIDTH * INPUT, 911, 0.0625);
        let host_bias_a = values::<Bf16>(WIDTH, 1919, 0.125);
        let host_bias_b = values::<Bf16>(WIDTH, 2027, 0.25);
        let x = DeviceBuffer::from_host(&stream, &host_x).unwrap();
        let weight = DeviceBuffer::from_host(&stream, &host_weight).unwrap();
        let bias_a = DeviceBuffer::from_host(&stream, &host_bias_a).unwrap();
        let bias_b = DeviceBuffer::from_host(&stream, &host_bias_b).unwrap();
        let sentinel = Bf16::encode(-99.0);
        let mut saw_small_workspace = false;
        let mut saw_base_algorithm = false;
        for &rows in &[5usize, 1, SMALL_ROWS, 33, 128, 5, SMALL_ROWS] {
            let base = bias_plan.base;
            let expected =
                prepared_for_rows(base, &bias_plan.small, bias_plan.small_prepared, rows).unwrap();
            provider.set_logical_rows(&mut bias_plan, rows).unwrap();
            assert_eq!(bias_plan.logical_rows(), rows as i32);
            assert_eq!(bias_plan.algo, expected.algo);
            assert_eq!(provider.core.workspace.pointer, workspace_pointer);
            assert!(bias_plan.workspace <= WORKSPACE_BYTES);
            saw_small_workspace |= rows <= SMALL_ROWS && bias_plan.workspace != 0;
            saw_base_algorithm |= rows > SMALL_ROWS && bias_plan.algo == base.algo;
            let mut output_a =
                DeviceBuffer::from_host(&stream, &[sentinel; MAX_ROWS * WIDTH]).unwrap();
            let mut output_b =
                DeviceBuffer::from_host(&stream, &[sentinel; MAX_ROWS * WIDTH]).unwrap();
            // SAFETY: buffers stay alive, bias pointer replacement is checked
            // through distinct outputs, and the host copies drain the stream.
            unsafe {
                provider
                    .enqueue(&bias_plan, &x, &weight, &bias_a, &mut output_a)
                    .unwrap();
                provider
                    .enqueue(&bias_plan, &x, &weight, &bias_b, &mut output_b)
                    .unwrap();
            }
            let output_a = output_a.to_host_vec(&stream).unwrap();
            let output_b = output_b.to_host_vec(&stream).unwrap();
            let mut bias_changed_output = false;
            for row in 0..rows {
                for column in 0..WIDTH {
                    let actual_a = Bf16::decode(output_a[row * WIDTH + column]);
                    let actual_b = Bf16::decode(output_b[row * WIDTH + column]);
                    assert!(actual_a.is_finite() && actual_b.is_finite());
                    bias_changed_output |= actual_a != actual_b;
                }
            }
            assert!(
                bias_changed_output,
                "bias alias was not applied at M={rows}"
            );
            let expected_gelu = prepared_for_rows(
                gelu_plan.base,
                &gelu_plan.small,
                gelu_plan.small_prepared,
                rows,
            )
            .unwrap();
            provider.set_logical_rows(&mut gelu_plan, rows).unwrap();
            assert_eq!(gelu_plan.algo, expected_gelu.algo);
        }
        assert!(saw_small_workspace);
        assert!(saw_base_algorithm);
    }

    // Real encoder widths exercise the shape-dependent auxiliary workspace that
    // the small padded fixtures do not need. Keep the oracle O(samples * K),
    // rather than multiplying every element on the CPU at the largest M.
    #[test]
    #[ignore = "requires CUDA 13 GPU runtime"]
    fn gpu_fused_model_widths_large_rows_share_workspace_and_shrink() {
        const MAX_ROWS: usize = 32768;
        const GUARD: usize = 32;
        let context = Context::new(0).unwrap();
        let stream = context.new_stream().unwrap();
        let provider = LtProvider::new(context.clone(), stream.clone()).unwrap();
        let workspace_pointer = provider.core.workspace.pointer;
        for (input, width, epilogue) in [
            (768, 2304, Epilogue::Bias),
            (768, 768, Epilogue::Bias),
            (768, 3072, Epilogue::GeluBias),
            (3072, 768, Epilogue::Bias),
            (1024, 4096, Epilogue::GeluBias),
        ] {
            let mut plan = match epilogue {
                Epilogue::Bias => provider.prepare_bias_shape::<Bf16>(input, width),
                Epilogue::GeluBias => provider.prepare_gelu_bias_shape::<Bf16>(input, width),
            }
            .unwrap();
            let seed_algorithm = plan.algo;
            let host_x = values::<Bf16>(MAX_ROWS * input, 17, 0.5);
            let host_weight = values::<Bf16>(width * input, 911, 0.0625);
            let host_bias = values::<Bf16>(width, 1919, 0.125);
            let x = DeviceBuffer::from_host(&stream, &host_x).unwrap();
            let weight = DeviceBuffer::from_host(&stream, &host_weight).unwrap();
            let bias = DeviceBuffer::from_host(&stream, &host_bias).unwrap();
            let mut output =
                DeviceBuffer::<Bf16>::zeroed(&stream, MAX_ROWS * width + GUARD).unwrap();
            let sentinel = [Bf16::encode(-99.0); GUARD];
            stream.synchronize().unwrap();
            let mut saw_workspace = false;
            for rows in [1, 128, 512, 16384, MAX_ROWS, 7] {
                provider.set_logical_rows(&mut plan, rows).unwrap();
                assert_eq!(plan.algo, seed_algorithm);
                assert_eq!(provider.core.workspace.pointer, workspace_pointer);
                assert!(plan.workspace <= WORKSPACE_BYTES);
                saw_workspace |= plan.workspace != 0;
                for guard_start in [rows * width, MAX_ROWS * width] {
                    // SAFETY: the previous iteration drained; both guard ranges
                    // fit this allocation, and the current context owns it.
                    unsafe {
                        cuda_core::memory::memcpy_htod_sync(
                            output.cu_deviceptr() + (guard_start * size_of::<Bf16>()) as u64,
                            sentinel.as_ptr(),
                            size_of_val(&sentinel),
                        )
                        .unwrap();
                    }
                }
                // SAFETY: all input/output allocations remain alive and this
                // same stream is drained before raw sampling or guard reads.
                unsafe { provider.enqueue(&plan, &x, &weight, &bias, &mut output) }.unwrap();
                stream.synchronize().unwrap();
                for sample in 0..64 {
                    let row = if sample == 63 {
                        rows - 1
                    } else {
                        (sample * 977) % rows
                    };
                    let column = if sample == 63 {
                        width - 1
                    } else {
                        (sample * 97) % width
                    };
                    let mut sum = 0.0f32;
                    for k in 0..input {
                        sum += Bf16::decode(host_x[row * input + k])
                            * Bf16::decode(host_weight[column * input + k]);
                    }
                    let value = sum + Bf16::decode(host_bias[column]);
                    let expected = match epilogue {
                        Epilogue::Bias => value,
                        Epilogue::GeluBias => {
                            let inner = 0.797_884_6 * value * (1.0 + 0.044715 * value * value);
                            0.5 * value * (1.0 + inner.tanh())
                        }
                    };
                    let mut actual = Bf16::encode(0.0);
                    // SAFETY: this initialized element is in the logical output
                    // and its producing stream has completed.
                    unsafe {
                        cuda_core::memory::memcpy_dtoh_async(
                            &mut actual,
                            output.cu_deviceptr()
                                + ((row * width + column) * size_of::<Bf16>()) as u64,
                            size_of::<Bf16>(),
                            stream.cu_stream(),
                        )
                        .unwrap();
                    }
                    stream.synchronize().unwrap();
                    let actual = Bf16::decode(actual);
                    assert!(
                        actual.is_finite()
                            && (actual - expected).abs() <= Bf16::TOLERANCE * expected.abs().max(1.0),
                        "M={rows} K={input} N={width} {epilogue:?} {row},{column}: {actual} vs {expected}"
                    );
                }
                for guard_start in [rows * width, MAX_ROWS * width] {
                    let mut actual = [Bf16::encode(0.0); GUARD];
                    // SAFETY: the guard is in the allocation and the stream is
                    // drained, so no foreign accesses can conflict with this read.
                    unsafe {
                        cuda_core::memory::memcpy_dtoh_async(
                            actual.as_mut_ptr(),
                            output.cu_deviceptr() + (guard_start * size_of::<Bf16>()) as u64,
                            size_of_val(&actual),
                            stream.cu_stream(),
                        )
                        .unwrap();
                    }
                    stream.synchronize().unwrap();
                    assert!(
                        actual.iter().all(|value| *value == sentinel[0]),
                        "projection wrote past M={rows} K={input} N={width} {epilogue:?}"
                    );
                }
            }
            assert!(
                saw_workspace,
                "fixture did not exercise nonzero workspace for K={input} N={width}"
            );
        }
    }
}
