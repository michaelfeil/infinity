// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! The small cuBLAS ABI boundary; tensor ownership stays in cuda-core.
use crate::{
    lt::ShapeError,
    math::{GemmArguments, LinearPlan, MathError},
    tensor::Element,
};
use cuda_core::{Context, Stream};
use infinity_v3_backend_core::BackendError;
use libloading::Library;
use std::{ffi::c_void, mem::ManuallyDrop, ptr, sync::Arc};

type Handle = *mut c_void;
type Create = unsafe extern "C" fn(*mut Handle) -> i32;
type Destroy = unsafe extern "C" fn(Handle) -> i32;
type SetStream = unsafe extern "C" fn(Handle, *mut c_void) -> i32;
type GemmEx = unsafe extern "C" fn(
    Handle,
    i32,
    i32,
    i32,
    i32,
    i32,
    *const f32,
    *const c_void,
    i32,
    i32,
    *const c_void,
    i32,
    i32,
    *const f32,
    *mut c_void,
    i32,
    i32,
    i32,
    i32,
) -> i32;
type SetWorkspace = unsafe extern "C" fn(Handle, *mut c_void, usize) -> i32;
type GemmStridedBatchedEx = unsafe extern "C" fn(
    Handle,
    i32,
    i32,
    i32,
    i32,
    i32,
    *const f32,
    *const c_void,
    i32,
    i32,
    i64,
    *const c_void,
    i32,
    i32,
    i64,
    *const f32,
    *mut c_void,
    i32,
    i32,
    i64,
    i32,
    i32,
    i32,
) -> i32;
const CUBLAS_GEMM_DEFAULT_TENSOR_OP: i32 = 99;

// CUDA 13.1 recommends 32 MiB for SM90/SM10x/SM12x split-K workspaces, which
// matches the cuBLAS default capacity. cuMemAlloc_v2 returns 256-byte-aligned
// device storage, satisfying cublasSetWorkspace_v2's alignment requirement.
const WORKSPACE_BYTES: usize = 32 * 1024 * 1024;

// This synchronous allocation is freed explicitly so the live-byte gauge is
// decremented only after physical deallocation, rather than a deferred free.
// Its one-time zero is stream-ordered, and every teardown path either proves
// the stream drained or retains the allocation because device use is uncertain.
struct Workspace {
    pointer: u64,
    context: Arc<Context>,
}

pub(crate) struct Blas {
    handle: Handle,
    destroy: Destroy,
    gemm: GemmEx,
    gemm_strided_batched: GemmStridedBatchedEx,
    stream: Arc<Stream>,
    context: Arc<Context>,
    workspace: ManuallyDrop<Workspace>,
    _library: ManuallyDrop<Library>,
}
// SAFETY: the backend serializes this handle behind a Mutex and binds its
// retained primary context on every call. No handle is shared concurrently.
unsafe impl Send for Blas {}

fn check(status: i32) -> Result<(), BackendError> {
    if status == 0 {
        Ok(())
    } else {
        Err(BackendError::Inference(format!("cuBLAS status {status}")))
    }
}

impl Workspace {
    fn new(context: Arc<Context>) -> Result<Self, BackendError> {
        context.bind_to_thread().map_err(super::start)?;
        // SAFETY: the allocation's context is bound, and free always runs
        // synchronously after the owning stream has drained.
        let pointer =
            unsafe { cuda_core::memory::malloc_sync(WORKSPACE_BYTES) }.map_err(super::start)?;
        super::LIVE_ARENA_BYTES.fetch_add(WORKSPACE_BYTES, std::sync::atomic::Ordering::Relaxed);
        super::record_live_arena_bytes();
        Ok(Self { pointer, context })
    }

    fn initialize(&self, stream: &Stream) -> Result<(), BackendError> {
        // One-time stream-ordered zero; the startup drain below makes the
        // workspace deterministic before cuBLAS binds it.
        // SAFETY: the allocation covers WORKSPACE_BYTES in the bound context,
        // and stream is a valid stream for that context.
        unsafe {
            cuda_core::memory::memset_d8_async(self.pointer, 0, WORKSPACE_BYTES, stream.cu_stream())
        }
        .map_err(super::start)?;
        stream.synchronize().map_err(super::start)
    }

    fn record_retained() {
        metrics::counter!("oxide_arena_retained_bytes_total").increment(WORKSPACE_BYTES as u64);
    }

    /// # Safety
    /// Every stream that could reference this workspace must have drained.
    unsafe fn free(&mut self) -> Result<(), BackendError> {
        self.context.bind_to_thread().map_err(super::start)?;
        // SAFETY: the caller drained every stream that could reference this
        // workspace, and this is the only free path for the pointer.
        unsafe { cuda_core::memory::free_sync(self.pointer) }.map_err(super::start)?;
        super::LIVE_ARENA_BYTES.fetch_sub(WORKSPACE_BYTES, std::sync::atomic::Ordering::Relaxed);
        super::record_live_arena_bytes();
        Ok(())
    }
}

fn retain_start_failure(
    context: &Arc<Context>,
    stream: &Arc<Stream>,
    workspace: Option<Workspace>,
    library: Library,
) {
    if let Some(workspace) = workspace {
        Workspace::record_retained();
        std::mem::forget(workspace);
    }
    std::mem::forget(library);
    std::mem::forget(stream.clone());
    std::mem::forget(context.clone());
}

fn cleanup_failed_start(
    context: &Arc<Context>,
    stream: &Arc<Stream>,
    destroy: Destroy,
    handle: Handle,
    workspace: Option<Workspace>,
    library: Library,
    cause: BackendError,
) -> BackendError {
    if context.bind_to_thread().is_err() || stream.synchronize().is_err() {
        retain_start_failure(context, stream, workspace, library);
        return cause;
    }
    // SAFETY: the stream drained, so this failed-start handle is idle.
    if check(unsafe { destroy(handle) }).is_err() {
        retain_start_failure(context, stream, workspace, library);
        return cause;
    }
    if let Some(mut workspace) = workspace {
        // SAFETY: the stream drained above, so no device work can reference it.
        if unsafe { workspace.free() }.is_err() {
            // The handle is gone, but the allocation, library, and context may
            // still be needed if the free left device state uncertain.
            retain_start_failure(context, stream, Some(workspace), library);
            return cause;
        }
    }
    cause
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct NativeM1Arguments {
    pub arguments: GemmArguments,
    pub bias: u64,
    pub bytes: usize,
    pub stream: usize,
}

fn classify_m1_preflight_math(error: MathError) -> ShapeError {
    match error {
        MathError::ContextMismatch | MathError::UnsupportedFormat => {
            ShapeError::Fatal(super::infer(error))
        }
        MathError::ShapeMismatch
        | MathError::EmptyMatrix
        | MathError::UnsupportedLayout
        | MathError::DimensionOverflow
        | MathError::Tensor(_) => ShapeError::Rejected(super::infer(error)),
    }
}

fn check_native_m1_prepared(
    prepared: &NativeM1Arguments,
    stream: usize,
    x: u64,
    weight: u64,
    bias: u64,
    output: u64,
) -> Result<(), ShapeError> {
    if stream != prepared.stream {
        return Err(ShapeError::Fatal(super::infer(
            "GTE down M1 launch stream differs from prepared metadata",
        )));
    }
    if x != prepared.arguments.x
        || weight != prepared.arguments.weight
        || bias != prepared.bias
        || output != prepared.arguments.output
    {
        return Err(ShapeError::Fatal(super::infer(
            "GTE down M1 launch buffers differ from prepared metadata",
        )));
    }
    Ok(())
}

impl Blas {
    pub(crate) fn stream(&self) -> &Stream {
        &self.stream
    }

    pub(crate) fn context(&self) -> &Arc<Context> {
        &self.context
    }

    pub(crate) fn new(context: Arc<Context>, stream: Arc<Stream>) -> Result<Self, BackendError> {
        context.bind_to_thread().map_err(super::start)?;
        // CUDA 13 is required by the pinned cuda-core headers/runtime.
        let library = unsafe { Library::new("libcublas.so.13") }.map_err(super::start)?;
        let (create, destroy, set_stream, set_workspace, gemm, gemm_strided_batched) = unsafe {
            (
                *library
                    .get::<Create>(b"cublasCreate_v2\0")
                    .map_err(super::start)?,
                *library
                    .get::<Destroy>(b"cublasDestroy_v2\0")
                    .map_err(super::start)?,
                *library
                    .get::<SetStream>(b"cublasSetStream_v2\0")
                    .map_err(super::start)?,
                *library
                    .get::<SetWorkspace>(b"cublasSetWorkspace_v2\0")
                    .map_err(super::start)?,
                *library
                    .get::<GemmEx>(b"cublasGemmEx\0")
                    .map_err(super::start)?,
                *library
                    .get::<GemmStridedBatchedEx>(b"cublasGemmStridedBatchedEx\0")
                    .map_err(super::start)?,
            )
        };
        let mut handle = ptr::null_mut();
        check(unsafe { create(&mut handle) }).map_err(super::start)?;
        if let Err(error) = check(unsafe { set_stream(handle, stream.cu_stream().cast()) }) {
            return Err(cleanup_failed_start(
                &context,
                &stream,
                destroy,
                handle,
                None,
                library,
                super::start(error),
            ));
        }
        let workspace = match Workspace::new(context.clone()) {
            Ok(workspace) => workspace,
            Err(error) => {
                return Err(cleanup_failed_start(
                    &context, &stream, destroy, handle, None, library, error,
                ));
            }
        };
        // Keep ownership here during initialization so a failed zero/drain can
        // use the same cleanup path as a failed binding. A successful cleanup
        // drain frees it; an uncertain drain retains every owner.
        let setup = workspace.initialize(&stream).and_then(|()| {
            // The stream is set and the workspace initialized before binding.
            // SAFETY: handle/storage share the context and storage covers the
            // requested capacity for every use on this stream.
            check(unsafe {
                set_workspace(handle, workspace.pointer as *mut c_void, WORKSPACE_BYTES)
            })
            .map_err(super::start)
        });
        if let Err(error) = setup {
            return Err(cleanup_failed_start(
                &context,
                &stream,
                destroy,
                handle,
                Some(workspace),
                library,
                error,
            ));
        }
        Ok(Self {
            handle,
            destroy,
            gemm,
            gemm_strided_batched,
            stream,
            context,
            workspace: ManuallyDrop::new(workspace),
            _library: ManuallyDrop::new(library),
        })
    }

    /// BF16 strided head-wise product for the native disentangled score tables.
    /// No transpose is applied: the caller supplies checked column-major views.
    ///
    /// # Safety
    /// This handle's context must be current. All pointers must name initialized
    /// compatible device views on its stream and survive completion; output may
    /// not alias an input. The caller preflights every extent and stride.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn strided_batched_bf16(
        &self,
        a: u64,
        b: u64,
        c: u64,
        m: i32,
        n: i32,
        k: i32,
        lda: i32,
        ldb: i32,
        ldc: i32,
        stride_a: i64,
        stride_b: i64,
        stride_c: i64,
        batches: i32,
    ) -> Result<(), BackendError> {
        if a == 0
            || b == 0
            || c == 0
            || m <= 0
            || n <= 0
            || k <= 0
            || lda < m
            || ldb < k
            || ldc < m
            || stride_a <= 0
            || stride_b <= 0
            || stride_c <= 0
            || batches <= 0
        {
            return Err(super::infer("Invalid BF16 strided relative projection"));
        }
        let (alpha, beta) = (1.0f32, 0.0f32);
        // SAFETY: checked geometry and the caller's retained borrowed-storage contract.
        check(unsafe {
            (self.gemm_strided_batched)(
                self.handle,
                0,
                0,
                m,
                n,
                k,
                &alpha,
                a as *const c_void,
                14,
                lda,
                stride_a,
                b as *const c_void,
                14,
                ldb,
                stride_b,
                &beta,
                c as *mut c_void,
                14,
                ldc,
                stride_c,
                batches,
                68,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )
        })
    }

    /// Enqueue one prepared X W^T operation with borrowed, checked storage.
    ///
    /// # Safety
    /// The caller must bind this context on the current thread and keep every
    /// allocation alive, without conflicting access, until this stream drains.
    /// A Rust borrow alone does not cover deferred GPU use.
    pub(crate) unsafe fn linear<E: Element>(
        &self,
        plan: &LinearPlan<E>,
        x: &cuda_core::DeviceBuffer<E::Storage>,
        w: &cuda_core::DeviceBuffer<E::Storage>,
        y: &mut cuda_core::DeviceBuffer<E::Storage>,
    ) -> Result<(), BackendError> {
        // Context identity checks do not call the driver or clone an Arc.
        if !Arc::ptr_eq(x.context(), &self.context) {
            return Err(super::infer(
                "Linear buffers belong to a different CUDA context",
            ));
        }
        let a = plan.bind(x, w, y).map_err(super::infer)?;
        let (alpha, beta) = (1.0f32, 0.0f32);
        // cuBLAS column-major view computes Y^T = W × X^T.
        check(unsafe {
            (self.gemm)(
                self.handle,
                1,
                0,
                a.width,
                a.rows,
                a.input,
                &alpha,
                a.weight as *const c_void,
                a.cuda_type,
                a.weight_stride,
                a.x as *const c_void,
                a.cuda_type,
                a.x_stride,
                &beta,
                a.output as *mut c_void,
                a.cuda_type,
                a.output_stride,
                68, // CUBLAS_COMPUTE_32F
                -1, // CUBLAS_GEMM_DEFAULT
            )
        })
    }

    /// Cheap metadata preflight for the native-biased beta=1 path.
    ///
    /// This checks the handle context and the prepared plan's storage without
    /// binding the context, allocating, creating descriptors, or enqueueing.
    /// The caller can therefore reject cross-context/cross-stream use before a
    /// bias-preseed copy and then call [`Self::enqueue_native_n1`] once.
    #[allow(dead_code)]
    pub(crate) fn preflight_beta_one<E: Element>(
        &self,
        plan: &LinearPlan<E>,
        x: &cuda_core::DeviceBuffer<E::Storage>,
        w: &cuda_core::DeviceBuffer<E::Storage>,
        y: &mut cuda_core::DeviceBuffer<E::Storage>,
    ) -> Result<GemmArguments, BackendError> {
        if !Arc::ptr_eq(x.context(), &self.context) {
            return Err(super::infer(
                "Linear input belongs to a different CUDA context",
            ));
        }
        if !Arc::ptr_eq(w.context(), &self.context) {
            return Err(super::infer(
                "Linear weight belongs to a different CUDA context",
            ));
        }
        if !Arc::ptr_eq(y.context(), &self.context) {
            return Err(super::infer(
                "Linear output belongs to a different CUDA context",
            ));
        }
        plan.bind(x, w, y).map_err(super::infer)
    }

    /// Pure preflight for the native non-Lt N=1 biased GEMM.
    ///
    /// Validates the real GEMM arguments plus the bias-copy shape, context and
    /// stream before any request work is submitted. Context/ownership/format
    /// invariants stay fatal; only genuine bias/geometry failures are rejected.
    /// The returned data contains no ownership; the caller retains every
    /// allocation until launch drains.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn preflight_native_m1<E: Element>(
        &self,
        plan: &LinearPlan<E>,
        x: &cuda_core::DeviceBuffer<E::Storage>,
        weight: &cuda_core::DeviceBuffer<E::Storage>,
        bias: &cuda_core::DeviceBuffer<E::Storage>,
        output: &mut cuda_core::DeviceBuffer<E::Storage>,
        stream: &Stream,
        width: usize,
    ) -> Result<NativeM1Arguments, ShapeError> {
        if width == 0 || bias.len() < width {
            return Err(ShapeError::Rejected(super::infer(format!(
                "GTE down M1 bias length {} is shorter than output width {width}",
                bias.len()
            ))));
        }
        let bytes = width
            .checked_mul(std::mem::size_of::<E::Storage>())
            .ok_or_else(|| {
                ShapeError::Fatal(super::infer("GTE down M1 bias copy size overflow"))
            })?;
        for context in [
            x.context(),
            weight.context(),
            bias.context(),
            output.context(),
        ] {
            if !Arc::ptr_eq(context, &self.context) {
                return Err(ShapeError::Fatal(super::infer(
                    "GTE down M1 buffers belong to a different CUDA context",
                )));
            }
        }
        if !std::ptr::eq(self.stream.as_ref(), stream) {
            return Err(ShapeError::Fatal(super::infer(
                "GTE down M1 requires the plain cuBLAS stream",
            )));
        }
        let arguments = plan
            .bind(x, weight, output)
            .map_err(classify_m1_preflight_math)?;
        Ok(NativeM1Arguments {
            arguments,
            bias: bias.cu_deviceptr(),
            bytes,
            stream: stream.cu_stream() as usize,
        })
    }

    /// Enqueue the actual bias D2D preseed and native `GemmEx(99)` selected by
    /// [`Self::preflight_native_m1`]. No metadata query or allocation runs here.
    ///
    /// # Safety
    /// `prepared` must come from this handle's preflight, the context must be
    /// bound on this thread, every allocation must remain alive without
    /// conflicting access until this stream drains, and the caller must pass the
    /// same current buffers and stream used for preflight.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn launch_native_m1<E: Element>(
        &self,
        plan: &LinearPlan<E>,
        prepared: &NativeM1Arguments,
        x: &cuda_core::DeviceBuffer<E::Storage>,
        weight: &cuda_core::DeviceBuffer<E::Storage>,
        bias: &cuda_core::DeviceBuffer<E::Storage>,
        output: &mut cuda_core::DeviceBuffer<E::Storage>,
        stream: &Stream,
    ) -> Result<(), ShapeError> {
        if !std::ptr::eq(self.stream.as_ref(), stream) {
            return Err(ShapeError::Fatal(super::infer(
                "GTE down M1 requires the plain cuBLAS stream",
            )));
        }
        for context in [
            x.context(),
            weight.context(),
            bias.context(),
            output.context(),
        ] {
            if !Arc::ptr_eq(context, &self.context) {
                return Err(ShapeError::Fatal(super::infer(
                    "GTE down M1 buffers belong to a different CUDA context",
                )));
            }
        }
        check_native_m1_prepared(
            prepared,
            stream.cu_stream() as usize,
            x.cu_deviceptr(),
            weight.cu_deviceptr(),
            bias.cu_deviceptr(),
            output.cu_deviceptr(),
        )?;
        let arguments = plan
            .bind(x, weight, output)
            .map_err(|error| ShapeError::Fatal(super::infer(error)))?;
        if arguments != prepared.arguments {
            return Err(ShapeError::Fatal(super::infer(
                "GTE down M1 plan arguments changed after preflight",
            )));
        }
        let width = usize::try_from(arguments.width)
            .map_err(|error| ShapeError::Fatal(super::infer(error)))?;
        if width == 0 || bias.len() < width {
            return Err(ShapeError::Fatal(super::infer(format!(
                "GTE down M1 bias length {} is shorter than output width {width}",
                bias.len()
            ))));
        }
        let bytes = width
            .checked_mul(std::mem::size_of::<E::Storage>())
            .ok_or_else(|| {
                ShapeError::Fatal(super::infer("GTE down M1 bias copy size overflow"))
            })?;
        if bytes != prepared.bytes {
            return Err(ShapeError::Fatal(super::infer(
                "GTE down M1 bias copy size differs from prepared metadata",
            )));
        }
        // SAFETY: preflight and the checks above validated the current buffers,
        // stream and bias-copy range; the copy and GEMM are ordered on the same
        // plain cuBLAS stream.
        unsafe {
            cuda_core::memory::memcpy_dtod_async(
                arguments.output,
                bias.cu_deviceptr(),
                bytes,
                stream.cu_stream(),
            )
        }
        .map_err(|error| ShapeError::Fatal(super::infer(error)))?;
        // SAFETY: the caller's retained-buffer contract covers the current
        // arguments, and the private handle remains serialized in default math.
        unsafe { self.enqueue_native_n1(&arguments) }
            .map_err(|error| ShapeError::Fatal(super::infer(error)))
    }

    /// Enqueue the native non-Lt N=1 biased GEMM.
    ///
    /// BF16 storage, FP32 accumulation, beta=1 over a pre-seeded bias output,
    /// `CUBLAS_GEMM_DEFAULT_TENSOR_OP`, and the default math mode.
    ///
    /// The handle is private to this `Blas`, is created by `cublasCreate_v2`
    /// with the primary CUDA default (`CUBLAS_DEFAULT_MATH`), and is never
    /// exposed or passed to another method that changes math mode. This method
    /// therefore uses the known handle invariant directly and performs no
    /// per-request `cublasGetMathMode`/`cublasSetMathMode` metadata call.
    ///
    /// # Safety
    /// `arguments` must come from [`Self::preflight_beta_one`], the context
    /// must be bound on this thread, and every allocation must remain alive
    /// without conflicting access until this stream drains.
    pub(crate) unsafe fn enqueue_native_n1(
        &self,
        arguments: &GemmArguments,
    ) -> Result<(), BackendError> {
        let (alpha, beta) = (1.0f32, 1.0f32);
        // SAFETY: the caller's contract covers the prepared arguments and owner
        // lifetimes; the private handle remains exclusively owned and
        // serialized in its default math mode.
        check(unsafe {
            (self.gemm)(
                self.handle,
                1,
                0,
                arguments.width,
                arguments.rows,
                arguments.input,
                &alpha,
                arguments.weight as *const c_void,
                arguments.cuda_type,
                arguments.weight_stride,
                arguments.x as *const c_void,
                arguments.cuda_type,
                arguments.x_stride,
                &beta,
                arguments.output as *mut c_void,
                arguments.cuda_type,
                arguments.output_stride,
                68, // CUBLAS_COMPUTE_32F
                CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )
        })
    }

    fn retain_owners(&self) {
        Workspace::record_retained();
        std::mem::forget(self.stream.clone());
        std::mem::forget(self.context.clone());
    }
}
impl Drop for Blas {
    fn drop(&mut self) {
        if self.context.bind_to_thread().is_err() || self.stream.synchronize().is_err() {
            // Retain the handle, workspace, library, stream, and context.
            self.retain_owners();
            return;
        }
        // SAFETY: the stream drained, so the handle is idle.
        if check(unsafe { (self.destroy)(self.handle) }).is_err() {
            // The handle may still be valid; retain everything that could be
            // needed to keep its storage and code loaded.
            self.retain_owners();
            return;
        }
        // SAFETY: the handle is gone and the stream drained, so no device work
        // can reference the workspace when it is freed synchronously.
        if unsafe { self.workspace.free() }.is_err() {
            self.retain_owners();
            return;
        }
        // SAFETY: the workspace field was already freed above; dropping it now
        // only releases its context clone. Unload the library last.
        unsafe {
            ManuallyDrop::drop(&mut self.workspace);
            ManuallyDrop::drop(&mut self._library);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        math::LinearPlan,
        tensor::{Bf16, Layout, F16, F32},
    };
    use cuda_core::DeviceBuffer;

    trait FixtureElement: Element {
        fn encode(value: f32) -> Self::Storage;
        fn decode(value: Self::Storage) -> f32;
        const TOLERANCE: f32;
    }
    impl FixtureElement for Bf16 {
        fn encode(v: f32) -> Bf16 {
            Bf16::from_bits(half::bf16::from_f32(v).to_bits())
        }
        fn decode(v: Bf16) -> f32 {
            half::bf16::from_bits(v.to_bits()).to_f32()
        }
        const TOLERANCE: f32 = 0.008;
    }
    impl FixtureElement for F16 {
        fn encode(v: f32) -> F16 {
            F16::from_bits(half::f16::from_f32(v).to_bits())
        }
        fn decode(v: F16) -> f32 {
            half::f16::from_bits(v.to_bits()).to_f32()
        }
        const TOLERANCE: f32 = 0.001;
    }
    impl FixtureElement for F32 {
        fn encode(v: f32) -> f32 {
            v
        }
        fn decode(v: f32) -> f32 {
            v
        }
        const TOLERANCE: f32 = 1e-5;
    }
    fn fixture<E: FixtureElement>(context: &Arc<Context>, stream: &Arc<Stream>, padded: bool) {
        let (m, k, n) = (7usize, 13usize, 5usize);
        let extra = if padded { 4 } else { 0 };
        let (sx, sw, sy) = (k + extra, k + extra, n + extra);
        let x = (0..m * sx)
            .map(|i| E::encode((i % 19) as f32 / 8.0 - 1.0))
            .collect::<Vec<_>>();
        let w = (0..n * sw)
            .map(|i| E::encode((i % 13) as f32 / 8.0 - 0.5))
            .collect::<Vec<_>>();
        let xb = DeviceBuffer::from_host(stream, &x).unwrap();
        let wb = DeviceBuffer::from_host(stream, &w).unwrap();
        let mut yb = DeviceBuffer::from_host(stream, &vec![E::encode(-99.0); m * sy]).unwrap();
        let plan = LinearPlan::<E>::new(
            Layout::strided([m, k], [sx, 1]).unwrap(),
            Layout::strided([n, k], [sw, 1]).unwrap(),
            Layout::strided([m, n], [sy, 1]).unwrap(),
        )
        .unwrap();
        let blas = Blas::new(context.clone(), stream.clone()).unwrap();
        // SAFETY: one bound context, disjoint buffers, and to_host_vec drains
        // all uses before the allocations/handle leave scope.
        unsafe {
            blas.linear(&plan, &xb, &wb, &mut yb).unwrap();
        }
        let y = yb.to_host_vec(stream).unwrap();
        for row in 0..m {
            for column in 0..n {
                let expected = (0..k)
                    .map(|i| E::decode(x[row * sx + i]) * E::decode(w[column * sw + i]))
                    .sum::<f32>();
                let actual = E::decode(y[row * sy + column]);
                assert!(
                    actual.is_finite()
                        && (actual - expected).abs() <= E::TOLERANCE * expected.abs().max(1.0),
                    "{:?} {row},{column}: {actual} vs {expected}",
                    E::FORMAT
                );
            }
            for column in n..sy {
                assert_eq!(E::decode(y[row * sy + column]), -99.0);
            }
        }
        let mut short = DeviceBuffer::zeroed(stream, 1).unwrap();
        assert!(plan.bind(&xb, &wb, &mut short).is_err());
    }

    #[test]
    #[ignore = "requires CUDA 13 GPU runtime"]
    fn gpu_linear_precisions_and_padded_strides() {
        let context = Context::new(0).unwrap();
        let stream = context.new_stream().unwrap();
        fixture::<Bf16>(&context, &stream, false);
        fixture::<F16>(&context, &stream, false);
        fixture::<F32>(&context, &stream, false);
        fixture::<Bf16>(&context, &stream, true);
        fixture::<F16>(&context, &stream, true);
        fixture::<F32>(&context, &stream, true);
    }

    #[test]
    #[ignore = "requires CUDA 13 GPU runtime and a device that selects split-K"]
    fn gpu_blas_splitk_scratch_is_initialized() {
        const M: usize = 10;
        const N: usize = 1536;
        const K: usize = 8960;
        const SENTINEL: f32 = -1337.0;
        let context = Context::new(0).unwrap();
        let stream = context.new_stream().unwrap();
        let one = <Bf16 as FixtureElement>::encode(1.0);
        let x = DeviceBuffer::from_host(&stream, &vec![one; M * K]).unwrap();
        let w = DeviceBuffer::from_host(&stream, &vec![one; N * K]).unwrap();
        let sentinel = <Bf16 as FixtureElement>::encode(SENTINEL);
        let mut y = DeviceBuffer::from_host(&stream, &vec![sentinel; M * N]).unwrap();
        let plan = LinearPlan::<Bf16>::contiguous(M, K, N).unwrap();
        let blas = Blas::new(context.clone(), stream.clone()).unwrap();
        // SAFETY: one bound context, disjoint buffers, and to_host_vec drains
        // all uses before the allocations/handle leave scope.
        unsafe {
            blas.linear(&plan, &x, &w, &mut y).unwrap();
        }
        let downloaded = y.to_host_vec(&stream).unwrap();
        let expected = <Bf16 as FixtureElement>::encode(8960.0);
        assert_eq!(<Bf16 as FixtureElement>::decode(expected), 8960.0);
        for (index, value) in downloaded.iter().enumerate() {
            assert_eq!(
                value.to_bits(),
                expected.to_bits(),
                "Y[{index}] is {:?}, not exactly BF16(8960.0)",
                <Bf16 as FixtureElement>::decode(*value)
            );
        }
    }
    #[test]
    fn native_m1_preflight_classifies_shape_rejected_and_invariants_fatal() {
        for error in [
            MathError::ShapeMismatch,
            MathError::EmptyMatrix,
            MathError::UnsupportedLayout,
            MathError::DimensionOverflow,
            MathError::Tensor(crate::tensor::TensorError::BufferTooSmall),
        ] {
            assert!(
                matches!(classify_m1_preflight_math(error), ShapeError::Rejected(_)),
                "{error:?} must remain a recoverable shape rejection"
            );
        }
        for error in [MathError::ContextMismatch, MathError::UnsupportedFormat] {
            assert!(
                matches!(classify_m1_preflight_math(error), ShapeError::Fatal(_)),
                "{error:?} must stay fatal"
            );
        }
    }

    #[test]
    fn native_m1_prepared_identity_rejects_stream_and_buffer_mismatches() {
        let prepared = NativeM1Arguments {
            arguments: GemmArguments {
                x: 0x1000,
                weight: 0x2000,
                output: 0x3000,
                rows: 1,
                input: 3072,
                width: 768,
                x_stride: 3072,
                weight_stride: 3072,
                output_stride: 768,
                cuda_type: 14,
            },
            bias: 0x4000,
            bytes: 1536,
            stream: 0x5000,
        };
        check_native_m1_prepared(&prepared, 0x5000, 0x1000, 0x2000, 0x4000, 0x3000).unwrap();
        assert!(matches!(
            check_native_m1_prepared(&prepared, 0x5001, 0x1000, 0x2000, 0x4000, 0x3000),
            Err(ShapeError::Fatal(_))
        ));
        assert!(matches!(
            check_native_m1_prepared(&prepared, 0x5000, 0x1001, 0x2000, 0x4000, 0x3000),
            Err(ShapeError::Fatal(_))
        ));
        assert!(matches!(
            check_native_m1_prepared(&prepared, 0x5000, 0x1000, 0x2000, 0x4001, 0x3000),
            Err(ShapeError::Fatal(_))
        ));
        assert!(matches!(
            check_native_m1_prepared(&prepared, 0x5000, 0x1000, 0x2000, 0x4000, 0x3001),
            Err(ShapeError::Fatal(_))
        ));
    }
}
