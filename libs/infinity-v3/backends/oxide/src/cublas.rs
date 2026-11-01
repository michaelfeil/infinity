// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! The small cuBLAS ABI boundary; tensor ownership stays in cuda-core.
use crate::{math::LinearPlan, tensor::Element};
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

impl Blas {
    pub(crate) fn new(context: Arc<Context>, stream: Arc<Stream>) -> Result<Self, BackendError> {
        context.bind_to_thread().map_err(super::start)?;
        // CUDA 13 is required by the pinned cuda-core headers/runtime.
        let library = unsafe { Library::new("libcublas.so.13") }.map_err(super::start)?;
        let (create, destroy, set_stream, set_workspace, gemm) = unsafe {
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
            stream,
            context,
            workspace: ManuallyDrop::new(workspace),
            _library: ManuallyDrop::new(library),
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
}
