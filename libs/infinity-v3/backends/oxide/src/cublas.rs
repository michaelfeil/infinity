// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! The small cuBLAS ABI boundary; tensor ownership stays in cuda-core.
use cuda_core::{Context, Stream};
use infinity_v3_backend_core::BackendError;
use libloading::Library;
use std::{ffi::c_void, ptr, sync::Arc};

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

pub(crate) struct Blas {
    handle: Handle,
    destroy: Destroy,
    gemm: GemmEx,
    stream: Arc<Stream>,
    context: Arc<Context>,
    _library: Library,
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

impl Blas {
    pub(crate) fn new(context: Arc<Context>, stream: Arc<Stream>) -> Result<Self, BackendError> {
        context.bind_to_thread().map_err(super::start)?;
        // CUDA 13 is required by the pinned cuda-core headers/runtime.
        let library = unsafe { Library::new("libcublas.so.13") }.map_err(super::start)?;
        let (create, destroy, set_stream, gemm) = unsafe {
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
                    .get::<GemmEx>(b"cublasGemmEx\0")
                    .map_err(super::start)?,
            )
        };
        let mut handle = ptr::null_mut();
        check(unsafe { create(&mut handle) }).map_err(super::start)?;
        if let Err(error) = check(unsafe { set_stream(handle, stream.cu_stream().cast()) }) {
            unsafe { destroy(handle) };
            return Err(super::start(error));
        }
        Ok(Self {
            handle,
            destroy,
            gemm,
            stream,
            context,
            _library: library,
        })
    }

    /// Row-major X[rows,input] × W[output,input]^T → Y[rows,output].
    /// CUDA pointers and storage dimensions are validated by the caller.
    pub(crate) fn linear(
        &self,
        x: u64,
        w: u64,
        y: u64,
        rows: usize,
        input: usize,
        output: usize,
    ) -> Result<(), BackendError> {
        let rows = i32::try_from(rows).map_err(super::infer)?;
        let input = i32::try_from(input).map_err(super::infer)?;
        let output = i32::try_from(output).map_err(super::infer)?;
        self.context.bind_to_thread().map_err(super::infer)?;
        let (alpha, beta) = (1.0f32, 0.0f32);
        // cuBLAS column-major view computes Y^T = W × X^T.
        check(unsafe {
            (self.gemm)(
                self.handle,
                1,
                0,
                output,
                rows,
                input,
                &alpha,
                w as *const c_void,
                14, // CUDA_R_16BF
                input,
                x as *const c_void,
                14,
                input,
                &beta,
                y as *mut c_void,
                14,
                output,
                68, // CUBLAS_COMPUTE_32F
                -1, // CUBLAS_GEMM_DEFAULT
            )
        })
    }
}
impl Drop for Blas {
    fn drop(&mut self) {
        let _ = self.context.bind_to_thread();
        let _ = self.stream.synchronize();
        unsafe { (self.destroy)(self.handle) };
    }
}
