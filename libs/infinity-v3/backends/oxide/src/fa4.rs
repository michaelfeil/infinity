// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Torch-free entry to the explicitly selected InfinityFA4 AOT provider.
use cuda_core::{Context, Stream};
use infinity_v3_backend_core::BackendError;
use libloading::Library;
use std::{
    ffi::{c_char, c_void, CStr},
    path::Path,
    sync::Arc,
};

const MODE: usize = 7;
type Query = unsafe extern "C" fn() -> i32;
type Error = unsafe extern "C" fn() -> *const c_char;
// Exact native1/export7 ABI from vendor/infinity_fa4/bridge.cpp.
type Forward = unsafe extern "C" fn(
    i32,
    i32,
    i32,
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
    f64,
    i32,
    i32,
    *const i64,
    i32,
) -> i32;
fn start(e: impl std::fmt::Display) -> BackendError {
    BackendError::Start(format!("Oxide FA4: {e}"))
}
fn infer(e: impl std::fmt::Display) -> BackendError {
    BackendError::Inference(format!("Oxide FA4: {e}"))
}
fn is_mha32(kernel: &serde_json::Value, dtype: &str) -> bool {
    let symbol = format!("fa4_mha32_{dtype}");
    kernel.get("symbol").and_then(serde_json::Value::as_str) == Some(symbol.as_str())
        && kernel.get("heads").and_then(serde_json::Value::as_u64) == Some(12)
        && kernel.get("kv_heads").and_then(serde_json::Value::as_u64) == Some(12)
        && kernel.get("d").and_then(serde_json::Value::as_u64) == Some(32)
        && kernel.get("causal").and_then(serde_json::Value::as_bool) == Some(false)
        && kernel
            .get("window")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|window| {
                window.len() == 2 && window.iter().all(serde_json::Value::is_null)
            })
}
fn mode_profile_matches(manifest: &serde_json::Value, dtype: &str) -> bool {
    manifest
        .get("kernels")
        .and_then(serde_json::Value::as_array)
        .and_then(|kernels| {
            kernels
                .iter()
                .filter(|kernel| {
                    kernel.get("dtype").and_then(serde_json::Value::as_str) == Some(dtype)
                })
                .nth(MODE)
        })
        .is_some_and(|kernel| is_mha32(kernel, dtype))
}
fn profiles_match(manifest: &serde_json::Value) -> bool {
    ["fp16", "bf16"]
        .into_iter()
        .all(|dtype| mode_profile_matches(manifest, dtype))
}

pub(crate) struct Fa4 {
    forward: Forward,
    error: Error,
    context: Arc<Context>,
    device: i32,
    _library: Library,
}
// SAFETY: the engine Mutex serializes calls and teardown, binds the retained
// primary context and drains pending work before releasing buffers/library.
unsafe impl Send for Fa4 {}
impl Fa4 {
    pub(crate) fn new(
        context: Arc<Context>,
        root: &Path,
        heads: usize,
        head_dim: usize,
    ) -> Result<Self, BackendError> {
        if (heads, head_dim) != (12, 32) {
            return Err(start("initial AOT profile requires12heads and width32"));
        }
        context.bind_to_thread().map_err(start)?;
        let (major, minor) = context.compute_capability().map_err(start)?;
        let capability = major
            .checked_mul(10)
            .and_then(|n| n.checked_add(minor))
            .ok_or_else(|| start("device capability overflow"))?;
        let path = root.join(format!("sm_{capability}"));
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path.join("manifest.json")).map_err(start)?)
                .map_err(start)?;
        if manifest
            .get("abi_version")
            .and_then(serde_json::Value::as_u64)
            != Some(7)
            || manifest
                .get("native_abi_version")
                .and_then(serde_json::Value::as_u64)
                != Some(1)
        {
            return Err(start("expected native ABI1/export ABI7"));
        }
        if !profiles_match(&manifest) {
            return Err(start(
                "bundle lacks noncausal mha32 mode 7 profiles for FP16 and BF16",
            ));
        }
        // SAFETY: root is a trusted provider selection; symbol signatures and
        // versioned export order match the repository's installed AOT provider.
        let library = unsafe { Library::new(path.join("libinfinityfa4.so")) }.map_err(start)?;
        let (portable, architecture, modes, forward, error) = unsafe {
            (
                *library
                    .get::<Query>(b"infinity_fa4_portable_host_v1\0")
                    .map_err(start)?,
                *library
                    .get::<Query>(b"infinity_fa4_compute_capability_v1\0")
                    .map_err(start)?,
                *library
                    .get::<Query>(b"infinity_fa4_modes_v1\0")
                    .map_err(start)?,
                *library
                    .get::<Forward>(b"infinity_fa4_forward_v1\0")
                    .map_err(start)?,
                *library
                    .get::<Error>(b"infinity_fa4_error_v1\0")
                    .map_err(start)?,
            )
        };
        // SAFETY: scalar queries have no borrowed arguments; library retained.
        if unsafe { portable() } != 1
            || unsafe { architecture() } != capability
            || unsafe { modes() } & (1 << MODE) == 0
        {
            return Err(start(
                "incompatible portable-host marker, architecture or attention modes",
            ));
        }
        Ok(Self {
            forward,
            error,
            device: i32::try_from(context.ordinal()).map_err(start)?,
            context,
            _library: library,
        })
    }
    /// The engine retains validated contiguous BF16 Q/K/V/output allocations
    /// and packed int32 boundaries through stream completion. No launch fallback.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward(
        &self,
        stream: &Stream,
        q: u64,
        k: u64,
        v: u64,
        out: u64,
        cumulative: u64,
        tokens: usize,
        batch: usize,
        max_len: usize,
    ) -> Result<(), BackendError> {
        if self.context.cu_ctx() != stream.context().cu_ctx() {
            return Err(infer("stream belongs to another CUDA context"));
        }
        if tokens == 0
            || batch == 0
            || batch > tokens
            || max_len == 0
            || max_len > tokens
            || [q, k, v, out]
                .iter()
                .any(|p| *p == 0 || !p.is_multiple_of(16))
            || cumulative == 0
            || !cumulative.is_multiple_of(4)
        {
            return Err(infer("invalid attention geometry or pointer alignment"));
        }
        let total = i64::from(i32::try_from(tokens).map_err(infer)?);
        let count = i32::try_from(batch).map_err(infer)?;
        count
            .checked_add(1)
            .ok_or_else(|| infer("boundary count overflow"))?;
        let max_q = i32::try_from(max_len).map_err(infer)?;
        self.context.bind_to_thread().map_err(infer)?;
        // Elements, not bytes: four contiguous[token,12,32] BF16 arrays.
        let strides = [384i64, 32, 1, 384, 32, 1, 384, 32, 1, 384, 32, 1];
        // SAFETY: profile7/dtype1 is noncausal BF16 MHA12x32. Device pointers
        // and library survive execution; stack descriptors are copied at launch.
        let status = unsafe {
            (self.forward)(
                MODE as i32,
                1,
                self.device,
                stream.cu_stream().cast(),
                q as *mut c_void,
                k as *mut c_void,
                v as *mut c_void,
                out as *mut c_void,
                cumulative as *mut c_void,
                cumulative as *mut c_void,
                total,
                total,
                i64::from(count),
                12,
                12,
                1.0 / 32f64.sqrt(),
                -1,
                -1,
                strides.as_ptr(),
                max_q,
            )
        };
        if status != 0 {
            // SAFETY: native error is thread-local, NUL-terminated and consumed
            // on this same host thread while the library remains loaded.
            let ptr = unsafe { (self.error)() };
            let detail = if ptr.is_null() {
                format!("provider status {status}")
            } else {
                unsafe { CStr::from_ptr(ptr) }
                    .to_string_lossy()
                    .into_owned()
            };
            return Err(infer(detail));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn kernel(dtype: &str, symbol: &str, heads: u64, d: u64, causal: bool, window: Value) -> Value {
        json!({
            "symbol": symbol,
            "heads": heads,
            "kv_heads": heads,
            "d": d,
            "causal": causal,
            "window": window,
            "dtype": dtype,
        })
    }
    fn kernels(dtype: &str) -> Vec<Value> {
        (0..=MODE)
            .map(|mode| {
                if mode == MODE {
                    kernel(
                        dtype,
                        &format!("fa4_mha32_{dtype}"),
                        12,
                        32,
                        false,
                        json!([null, null]),
                    )
                } else {
                    kernel(
                        dtype,
                        &format!("fa4_profile{mode}_{dtype}"),
                        16,
                        64,
                        false,
                        json!([null, null]),
                    )
                }
            })
            .collect()
    }
    fn manifest(fp16: Vec<Value>, bf16: Vec<Value>) -> Value {
        json!({"kernels": fp16.into_iter().chain(bf16).collect::<Vec<_>>()})
    }
    fn replace(kernels: &mut [Value], mode: usize, value: Value) {
        if let Some(slot) = kernels.get_mut(mode) {
            *slot = value;
        }
    }

    #[test]
    fn accepts_mha32_at_mode_seven_for_both_dtypes() {
        assert!(profiles_match(&manifest(kernels("fp16"), kernels("bf16"))));
    }

    #[test]
    fn rejects_reordered_wrong_width_profile_at_mode_seven() {
        for (position, dtype) in ["fp16", "bf16"].into_iter().enumerate() {
            let (mut fp16, mut bf16) = (kernels("fp16"), kernels("bf16"));
            let target = if position == 0 { &mut fp16 } else { &mut bf16 };
            replace(
                target,
                MODE,
                kernel(
                    dtype,
                    &format!("fa4_media72_{dtype}"),
                    16,
                    72,
                    false,
                    json!([null, null]),
                ),
            );
            replace(
                target,
                MODE - 1,
                kernel(
                    dtype,
                    &format!("fa4_mha32_{dtype}"),
                    12,
                    32,
                    false,
                    json!([null, null]),
                ),
            );
            assert!(
                !profiles_match(&manifest(fp16, bf16)),
                "{dtype} mode 7 is not mha32"
            );
        }
    }

    #[test]
    fn rejects_local_window_or_causal_mha32_at_mode_seven() {
        let mut bf16 = kernels("bf16");
        replace(
            &mut bf16,
            MODE,
            kernel("bf16", "fa4_mha32_bf16", 12, 32, false, json!([64, 64])),
        );
        assert!(!profiles_match(&manifest(kernels("fp16"), bf16)));

        let mut bf16 = kernels("bf16");
        replace(
            &mut bf16,
            MODE,
            kernel("bf16", "fa4_mha32_bf16", 12, 32, true, json!([null, null])),
        );
        assert!(!profiles_match(&manifest(kernels("fp16"), bf16)));
    }
}
