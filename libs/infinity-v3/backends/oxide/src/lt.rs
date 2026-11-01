// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Prepared cuBLASLt fused projection plans.
//!
//! A plan owns a complete opaque algorithm selected by the current GPU for one
//! `(K, N, epilogue)` tuple at a seed row count of 128. Logical row changes only
//! resize the B/D descriptors and call `cublasLtMatmulAlgoCheck`; they never run
//! the heuristic again or replace the algorithm. The provider retains the loaded
//! library, context, stream, and cuBLASLt handle, while each plan retains the API
//! vtable and drains its stream before destroying host descriptors. One fixed
//! provider workspace is reused in stream order, including when a zero-workspace
//! seed algorithm requires a small auxiliary buffer at a different row count.
//!
//! ABI provenance (CUDA 13.1 primary header):
//! `/usr/local/cuda-13.1/include/cublasLt.h`; enum values, descriptor sizes, and
//! `cublasLtMatmulHeuristicResult_t` offsets were checked with the compiled probe
//! `/tmp/infinity-oxide-next-20261031/lt-prototype/abi_probe.cpp` (output in
//! `abi_probe.json` beside it). The probe reports a 96-byte, 8-byte-aligned
//! heuristic result; 64-byte/8-byte `cublasLtMatmulAlgo_t`; and offsets 0, 64,
//! 72, 76, 80 for algo, workspaceSize, state, wavesCount, and reserved.
//!
//! Only BF16 and F16 storage are claimed. FP8/FP4 `ScalarFormat` values have no
//! `Element` marker and therefore cannot reach this module; F32 is also rejected
//! rather than silently selecting an unqualified method.

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
const EPILOGUE_BIAS: i32 = 4;
const EPILOGUE_GELU_BIAS: i32 = 36;

const SEED_ROWS: usize = 128;
const SEED_WORKSPACE_BYTES: usize = 0;
const WORKSPACE_BYTES: usize = 64 * 1024;
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
// per logical row count. No initialization is needed by cuBLASLt's workspace ABI.
struct Workspace {
    pointer: u64,
    context: Arc<Context>,
}

impl Workspace {
    fn new(context: Arc<Context>) -> Result<Self, BackendError> {
        context.bind_to_thread().map_err(super::start)?;
        // SAFETY: the allocation's context is bound, and Drop uses free_sync.
        let pointer =
            unsafe { cuda_core::memory::malloc_sync(WORKSPACE_BYTES) }.map_err(super::start)?;
        super::LIVE_ARENA_BYTES.fetch_add(WORKSPACE_BYTES, Ordering::Relaxed);
        super::record_live_arena_bytes();
        Ok(Self { pointer, context })
    }

    fn record_retained() {
        metrics::counter!("oxide_arena_retained_bytes_total").increment(WORKSPACE_BYTES as u64);
    }
}

impl Drop for Workspace {
    fn drop(&mut self) {
        if self.context.bind_to_thread().is_err()
            // SAFETY: Core drops this allocation only after draining its stream;
            // construction failures have not submitted any users of the pointer.
            || unsafe { cuda_core::memory::free_sync(self.pointer) }.is_err()
        {
            Self::record_retained();
            std::mem::forget(self.context.clone());
            return;
        }
        super::LIVE_ARENA_BYTES.fetch_sub(WORKSPACE_BYTES, Ordering::Relaxed);
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
}

impl Drop for Core {
    fn drop(&mut self) {
        if self.context.bind_to_thread().is_err() || self.stream.synchronize().is_err() {
            self.descriptors_safe.store(false, Ordering::Release);
            // Retain the handle, library, context, and stream after an uncertain
            // completion instead of releasing resources still used by the GPU.
            retain_failed_resources(&self.api, &self.context, &self.stream);
            Workspace::record_retained();
            return;
        }
        if unsafe { (self.api.handle_destroy)(self.handle) } != STATUS_SUCCESS {
            self.descriptors_safe.store(false, Ordering::Release);
            retain_failed_resources(&self.api, &self.context, &self.stream);
            Workspace::record_retained();
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
        context.bind_to_thread().map_err(super::start)?;
        if !Arc::ptr_eq(stream.context(), &context) {
            return Err(super::start(
                "cuBLASLt stream belongs to a different CUDA context",
            ));
        }
        let api = Arc::new(Api::load()?);
        let workspace = Workspace::new(context.clone())?;
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
        Ok(Self {
            core: Box::new(Core {
                api,
                handle,
                context,
                stream,
                descriptors_safe: Arc::new(AtomicBool::new(true)),
                owner_token: Arc::new(()),
                workspace: ManuallyDrop::new(workspace),
                #[cfg(test)]
                shape_rejection: Cell::new(None),
            }),
        })
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

    fn prepare<E: Element>(
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
        set_desc(
            &api,
            plan.op,
            MATMUL_DESC_BIAS_POINTER,
            &SEED_BIAS_POINTER,
            "seed bias pointer",
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
        let workspace_limit = SEED_WORKSPACE_BYTES as u64;
        set_preference(
            &api,
            plan.pref,
            PREFERENCE_MAX_WORKSPACE_BYTES,
            &workspace_limit,
            "workspace preference",
        )?;

        let mut choices = [HeuristicResult::default(); HEURISTIC_CHOICES];
        let mut count = 0i32;
        let status = unsafe {
            (api.heuristic)(
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
        plan.algo = select_zero_workspace_algorithm(&choices[..count])?;
        plan.workspace = self
            .verify(&plan, plan.rows)
            .map_err(ShapeError::into_backend)?;
        self.set_logical_rows(&mut plan, geometry.initial_rows as usize)
            .map_err(ShapeError::into_backend)?;
        Ok(plan)
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
        self.core.context.bind_to_thread().map_err(super::infer)?;
        let previous = plan.rows;
        if let Err(error) = set_columns(&plan.api, plan, rows) {
            let restored = set_columns(&plan.api, plan, previous);
            return Err(after_rollback(error.into(), restored, &mut plan.usable));
        }
        match self.verify(plan, rows) {
            Ok(workspace) => {
                plan.rows = rows;
                plan.workspace = workspace;
                Ok(())
            }
            Err(error) => {
                let restored = set_columns(&plan.api, plan, previous);
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

    fn verify<E: Element>(&self, plan: &LtPlan<E>, rows: i32) -> Result<usize, ShapeError> {
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
        // Test-only rejection occurs after a real successful check, while
        // descriptors contain the requested rows. Normal rollback handles it.
        #[cfg(test)]
        let status = if status == STATUS_SUCCESS
            && checked.state == STATUS_SUCCESS
            && checked.workspace_size <= WORKSPACE_BYTES
            && inject_shape_rejection(&self.core.shape_rejection, rows)
        {
            STATUS_NOT_SUPPORTED
        } else {
            status
        };
        checked_workspace(status, &checked, geometry)
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
        if plan.workspace > WORKSPACE_BYTES {
            return Err(super::infer(format!(
                "fused plan requires {} workspace bytes, provider limit {WORKSPACE_BYTES} (M={} K={} N={} epilogue={:?})",
                plan.workspace, plan.rows, plan.geometry.input, plan.geometry.width, plan.epilogue
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

        let alpha = 1.0f32;
        let beta = 0.0f32;
        // All submissions use the same ordered stream and are externally
        // serialized, so this one provider allocation can serve every plan.
        let workspace = self.core.workspace.pointer as *mut c_void;
        let stream = self.core.stream.cu_stream().cast();
        let status = (self.core.api.matmul)(
            self.core.handle,
            plan.op,
            &alpha,
            weight_view.device_ptr() as *const c_void,
            plan.a,
            x_view.device_ptr() as *const c_void,
            plan.b,
            &beta,
            output_view.device_ptr() as *const c_void,
            plan.y,
            output_view.device_ptr() as *mut c_void,
            plan.y,
            &plan.algo,
            workspace,
            plan.workspace,
            stream,
        );
        check(status, "fused matmul")
    }
}

fn checked_workspace(
    status: i32,
    checked: &HeuristicResult,
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
    if checked.workspace_size > WORKSPACE_BYTES {
        return Err(ShapeError::Rejected(super::infer(format!(
            "fused algorithm requires {} workspace bytes, provider limit {WORKSPACE_BYTES} ({})",
            checked.workspace_size,
            geometry()
        ))));
    }
    Ok(checked.workspace_size)
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
