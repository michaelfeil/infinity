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

pub(crate) struct Blas {
    handle: Handle,
    destroy: Destroy,
    gemm: GemmEx,
    stream: Arc<Stream>,
    context: Arc<Context>,
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
}
impl Drop for Blas {
    fn drop(&mut self) {
        let _ = self.context.bind_to_thread();
        if self.stream.synchronize().is_ok() {
            // SAFETY: the handle is idle; unload only after destroying it.
            unsafe {
                (self.destroy)(self.handle);
                ManuallyDrop::drop(&mut self._library);
            }
        } else {
            // Retain the stream/context and loaded library on uncertain use.
            std::mem::forget(self.stream.clone());
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
}
