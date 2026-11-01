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

/// Legacy 8-head slopes retained byte-for-byte.
const SLOPES_8: [f32; 8] = [
    0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625,
];
/// Native twelve-head extension. The first eight slopes match the legacy
/// power-of-two block; the remaining four use the native `extra` recurrence
/// `2^(-2^(-(log2(16)-3)))` raised to odd powers.
const SLOPES_12: [f32; 12] = [
    0.5,
    0.25,
    0.125,
    0.0625,
    0.03125,
    0.015625,
    0.0078125,
    0.00390625,
    f32::from_bits(0x3f35_04f3),
    f32::from_bits(0x3eb5_04f3),
    f32::from_bits(0x3e35_04f3),
    f32::from_bits(0x3db5_04f3),
];

/// Thin typed prepared ALiBi profile. It owns only immutable launch metadata;
/// device pointers remain in the ordinary launch owner below.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AlibiProfile {
    Eight,
    TwelveNativeFlash,
}

impl AlibiProfile {
    pub(crate) const fn for_heads(heads: usize) -> Option<Self> {
        match heads {
            8 => Some(Self::Eight),
            _ => None,
        }
    }

    pub(crate) const fn heads(self) -> usize {
        match self {
            Self::Eight => 8,
            Self::TwelveNativeFlash => 12,
        }
    }

    pub(crate) const fn token_limit(self) -> usize {
        i32::MAX as usize / (self.heads() * 64)
    }

    pub(crate) const fn slopes(self) -> &'static [f32] {
        match self {
            Self::Eight => &SLOPES_8,
            Self::TwelveNativeFlash => &SLOPES_12,
        }
    }

    /// Additive host ABI. The legacy symbol and argument list stay unchanged.
    pub(crate) const fn host_symbol(self) -> &'static [u8] {
        match self {
            Self::Eight => b"infinity_fa4_alibi_model_bias_v1\0",
            Self::TwelveNativeFlash => b"infinity_fa4_jinacode_flash_alibi_v1\0",
        }
    }

    fn kernel_symbol(self, dtype: &str) -> String {
        match self {
            Self::Eight => format!("fa4_alibi_model_bias_{dtype}"),
            Self::TwelveNativeFlash => format!("fa4_jinacode_flash_alibi_{dtype}"),
        }
    }
}

pub(crate) fn profile_for_symbol(symbol: &str, dtype: &str) -> Option<AlibiProfile> {
    [AlibiProfile::Eight, AlibiProfile::TwelveNativeFlash]
        .into_iter()
        .find(|profile| symbol == profile.kernel_symbol(dtype))
}

pub(crate) fn kernel_matches(
    kernel: &serde_json::Value,
    dtype: &str,
    profile: AlibiProfile,
) -> bool {
    let heads = u64::try_from(profile.heads()).ok();
    let slope_metadata_matches = match profile {
        AlibiProfile::Eight => true,
        AlibiProfile::TwelveNativeFlash => {
            kernel
                .get("alibi_heads")
                .and_then(serde_json::Value::as_u64)
                == Some(12)
                && kernel
                    .get("alibi_slope_count")
                    .and_then(serde_json::Value::as_u64)
                    == Some(12)
        }
    };
    slope_metadata_matches
        && kernel.get("symbol").and_then(serde_json::Value::as_str)
            == Some(profile.kernel_symbol(dtype).as_str())
        && kernel.get("dtype").and_then(serde_json::Value::as_str) == Some(dtype)
        && kernel.get("kind").and_then(serde_json::Value::as_str)
            == Some(match profile {
                AlibiProfile::Eight => "alibi_model_bias",
                AlibiProfile::TwelveNativeFlash => "alibi12_native_flash",
            })
        && kernel
            .get("score_source")
            .and_then(serde_json::Value::as_str)
            == Some(if profile == AlibiProfile::TwelveNativeFlash {
                "alibi12_native_score.py"
            } else {
                "alibi_score.py"
            })
        && kernel.get("heads").and_then(serde_json::Value::as_u64) == heads
        && kernel.get("kv_heads").and_then(serde_json::Value::as_u64) == heads
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
            == Some(if profile == AlibiProfile::TwelveNativeFlash {
                "float32_fma_unscaled_then_pow2_scale"
            } else {
                "model_dtype"
            })
}

pub(crate) struct AlibiLaunch {
    forward: Forward,
    slopes: ManuallyDrop<DeviceBuffer<f32>>,
    heads: i64,
    token_limit: usize,
    stream: ManuallyDrop<Arc<Stream>>,
    // A separate dlopen reference retains code even if the enclosing ordinary
    // provider drops after an uncertain drain. All these fields intentionally
    // remain live on that error path, matching the engine's cleanup contract.
    library: ManuallyDrop<Library>,
    context: ManuallyDrop<Arc<Context>>,
}
impl AlibiLaunch {
    pub(crate) fn token_limit(&self) -> usize {
        self.token_limit
    }

    pub(crate) fn new(
        context: Arc<Context>,
        stream: Arc<Stream>,
        path: &Path,
        profile: AlibiProfile,
    ) -> Result<Self, BackendError> {
        if context.cu_ctx() != stream.context().cu_ctx() {
            return Err(start("ALiBi stream belongs to another CUDA context"));
        }
        let heads = i64::try_from(profile.heads()).map_err(start)?;
        // SAFETY: trusted provider path and additive versioned ABI; the owning
        // library reference is retained through the final ordered stream use.
        let library = unsafe { Library::new(path) }.map_err(start)?;
        let forward = unsafe {
            *library
                .get::<Forward>(profile.host_symbol())
                .map_err(start)?
        };
        let slopes_values = profile.slopes();
        // SAFETY: the native slope vector is fully initialized before publication.
        let slopes = unsafe { DeviceBuffer::uninitialized_async(&stream, slopes_values.len()) }
            .map_err(start)?;
        let mut result = Self {
            forward,
            slopes: ManuallyDrop::new(slopes),
            heads,
            token_limit: profile.token_limit(),
            stream: ManuallyDrop::new(stream),
            library: ManuallyDrop::new(library),
            context: ManuallyDrop::new(context),
        };
        result
            .slopes
            .copy_from_host(&result.stream, slopes_values)
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
                self.heads,
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

#[cfg(test)]
mod native_flash_recipe_tests {
    use super::*;

    #[test]
    fn native_flash_has_distinct_abi_and_preserves_all12_slopes_and_bounds() {
        let native = AlibiProfile::TwelveNativeFlash;
        assert_eq!(native.heads(), 12);
        assert_eq!(native.slopes(), &SLOPES_12);
        assert_eq!(native.token_limit(), i32::MAX as usize / 768);
        assert_ne!(native.host_symbol(), AlibiProfile::Eight.host_symbol());
        assert_ne!(
            native.kernel_symbol("bf16"),
            AlibiProfile::Eight.kernel_symbol("bf16")
        );
        assert_eq!(
            AlibiProfile::Eight.host_symbol(),
            b"infinity_fa4_alibi_model_bias_v1\0"
        );
    }

    #[test]
    fn native_flash_descriptor_rejects_fallback_bias_rounding_and_wrong_scale() {
        let native = AlibiProfile::TwelveNativeFlash;
        let metadata = serde_json::json!({
            "symbol":"fa4_jinacode_flash_alibi_bf16", "dtype":"bf16",
            "kind":"alibi12_native_flash", "heads":12, "kv_heads":12,
            "d":64, "causal":false, "window":[null,null],
            "score_source":"alibi12_native_score.py", "score_abi":1,
            "qk_scale":0.125, "external_softmax_scale":1.0,
            "bias_rounding":"float32_fma_unscaled_then_pow2_scale",
            "alibi_heads":12, "alibi_slope_count":12
        });
        assert!(kernel_matches(&metadata, "bf16", native));
        assert!(!kernel_matches(&metadata, "bf16", AlibiProfile::Eight));
        for (field, value) in [
            ("bias_rounding", serde_json::json!("model_dtype")),
            ("external_softmax_scale", serde_json::json!(0.125)),
            ("score_source", serde_json::json!("alibi_score.py")),
            ("alibi_heads", serde_json::json!(8)),
            ("score_abi", serde_json::json!(2)),
        ] {
            let mut bad = metadata.clone();
            bad[field] = value;
            assert!(!kernel_matches(&bad, "bf16", native), "{field}");
        }
    }
}
