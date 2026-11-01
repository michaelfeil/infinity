// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Prepared borrowed-F32 ALiBi launch and its stream-bound storage owner.
use crate::{infer, start};
use cuda_core::{Context, DeviceBuffer, Stream};
use infinity_v3_backend_core::BackendError;
use libloading::Library;
use std::{ffi::c_void, mem::ManuallyDrop, path::Path, sync::Arc};

type Forward = unsafe extern "C" fn(
    i32,
    i32,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *const f32,
    i64,
    i64,
    i64,
    *const i64,
    i32,
) -> i32;
const SLOPES: [f32; 8] = [
    0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625,
];

pub(crate) fn kernel_matches(kernel: &serde_json::Value, dtype: &str) -> bool {
    kernel.get("symbol").and_then(serde_json::Value::as_str)
        == Some(format!("fa4_alibi_model_bias_{dtype}").as_str())
        && kernel.get("dtype").and_then(serde_json::Value::as_str) == Some(dtype)
        && kernel.get("kind").and_then(serde_json::Value::as_str) == Some("alibi_model_bias")
        && kernel
            .get("score_source")
            .and_then(serde_json::Value::as_str)
            == Some("alibi_score.py")
        && kernel.get("heads").and_then(serde_json::Value::as_u64) == Some(8)
        && kernel.get("kv_heads").and_then(serde_json::Value::as_u64) == Some(8)
        && kernel.get("d").and_then(serde_json::Value::as_u64) == Some(64)
        && kernel.get("causal").and_then(serde_json::Value::as_bool) == Some(false)
        && kernel.get("window") == Some(&serde_json::json!([null, null]))
        && kernel.get("score_abi").and_then(serde_json::Value::as_u64) == Some(1)
        && kernel.get("qk_scale").and_then(serde_json::Value::as_f64) == Some(0.125)
        && kernel
            .get("external_softmax_scale")
            .and_then(serde_json::Value::as_f64)
            == Some(1.0)
        && kernel
            .get("bias_rounding")
            .and_then(serde_json::Value::as_str)
            == Some("model_dtype")
}

pub(crate) struct AlibiLaunch {
    forward: Forward,
    slopes: ManuallyDrop<DeviceBuffer<f32>>,
    stream: ManuallyDrop<Arc<Stream>>,
    // A separate dlopen reference retains code even if the enclosing ordinary
    // provider drops after an uncertain drain. All these fields intentionally
    // remain live on that error path, matching the engine's cleanup contract.
    library: ManuallyDrop<Library>,
    context: ManuallyDrop<Arc<Context>>,
}
impl AlibiLaunch {
    pub(crate) fn new(
        context: Arc<Context>,
        stream: Arc<Stream>,
        path: &Path,
    ) -> Result<Self, BackendError> {
        if context.cu_ctx() != stream.context().cu_ctx() {
            return Err(start("ALiBi stream belongs to another CUDA context"));
        }
        // SAFETY: trusted provider path and additive versioned ABI; the owning
        // library reference is retained through the final ordered stream use.
        let library = unsafe { Library::new(path) }.map_err(start)?;
        let forward = unsafe {
            *library
                .get::<Forward>(b"infinity_fa4_alibi_model_bias_v1\0")
                .map_err(start)?
        };
        // SAFETY: the eight slopes are fully initialized before publication.
        let slopes =
            unsafe { DeviceBuffer::uninitialized_async(&stream, SLOPES.len()) }.map_err(start)?;
        let mut result = Self {
            forward,
            slopes: ManuallyDrop::new(slopes),
            stream: ManuallyDrop::new(stream),
            library: ManuallyDrop::new(library),
            context: ManuallyDrop::new(context),
        };
        result
            .slopes
            .copy_from_host(&result.stream, &SLOPES)
            .map_err(start)?;
        Ok(result)
    }
    pub(crate) fn check_stream(&self, stream: &Stream) -> Result<(), BackendError> {
        if self.stream.cu_stream() != stream.cu_stream() {
            return Err(infer("ALiBi launch must use its prepared ordered stream"));
        }
        Ok(())
    }
    /// Caller validates all geometry and keeps the borrowed launch storage live.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn launch(
        &self,
        dtype: i32,
        device: i32,
        stream: &Stream,
        q: u64,
        k: u64,
        v: u64,
        out: u64,
        offsets: u64,
        total: i64,
        batch: i64,
        strides: &[i64; 12],
        max_q: i32,
    ) -> i32 {
        // SAFETY: caller's checked packed self-attention contract; immutable
        // F32 slopes remain retained by this owner until the stream drains.
        unsafe {
            (self.forward)(
                dtype,
                device,
                stream.cu_stream().cast(),
                q as *mut c_void,
                k as *mut c_void,
                v as *mut c_void,
                out as *mut c_void,
                offsets as *mut c_void,
                self.slopes.cu_deviceptr() as *const f32,
                total,
                batch,
                8,
                strides.as_ptr(),
                max_q,
            )
        }
    }
}
impl Drop for AlibiLaunch {
    fn drop(&mut self) {
        if self.stream.synchronize().is_ok() {
            // SAFETY: all uses have drained; each field is dropped once.
            unsafe {
                ManuallyDrop::drop(&mut self.slopes);
                ManuallyDrop::drop(&mut self.library);
                ManuallyDrop::drop(&mut self.stream);
                ManuallyDrop::drop(&mut self.context);
            }
        }
    }
}
