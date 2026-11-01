// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Torch-free entry to the explicitly selected InfinityFA4 AOT provider.
//!
//! Profile order and exported windows mirror the repository's ABI7
//! `vendor/infinity_fa4/profiles.inc` and `build.py`. A bundle is accepted only
//! when its manifest exports the full legacy-19 or appended-20 mode prefix in
//! that known order for the selected dtype. Absolute head counts are runtime
//! tensor dimensions; selection uses
//! the exported Q/KV grouping with the exact dimension and mask.
//!
//! Softcap, relative-bias and DeBERTa score modifiers are not represented by
//! [`AttentionSpec`], so they fail selection at load instead of falling back to
//! another attention implementation.
use cuda_core::{Context, Stream};
use infinity_v3_backend_core::BackendError;
use libloading::Library;
use std::{
    ffi::{c_char, c_void, CStr},
    path::Path,
    sync::Arc,
};

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DType {
    #[allow(dead_code)] // GPU-qualified operator; BERT serving currently uses BF16.
    F16,
    Bf16,
}

impl DType {
    const fn name(self) -> &'static str {
        match self {
            Self::F16 => "fp16",
            Self::Bf16 => "bf16",
        }
    }

    const fn code(self) -> i32 {
        match self {
            Self::F16 => 0,
            Self::Bf16 => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Window {
    Global,
    Local { left: i32, right: i32 },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Scale {
    RsqrtHeadDim,
    #[allow(dead_code)] // Needed by model-specific temperatures, tested independently.
    Explicit(f64),
}

/// Provider geometry and scale convention selected once at load. `window` is
/// the exact exported mask, not an inferred approximation;
/// `Scale::RsqrtHeadDim` resolves to `1 / sqrt(head_dim)` at load.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct AttentionSpec {
    pub(crate) heads: usize,
    pub(crate) kv_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) causal: bool,
    pub(crate) window: Window,
    pub(crate) dtype: DType,
    pub(crate) scale: Scale,
}

/// One packed attention launch. All pointers are CUDA addresses in the
/// selected provider context and all storage uses the selected spec's dtype.
/// Strides are element strides for `[tokens, heads, head_dim]`; `q_offsets`
/// and `k_offsets` name device int32 arrays of `batch + 1` cumulative lengths.
/// `max_q` is the longest query sequence. Output storage must not alias an
/// input or offsets. Offsets must start at zero, increase monotonically,
/// end at the corresponding token count, and describe queries no longer
/// than `max_q`. All buffers must remain live until stream completion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AttentionBatch {
    pub(crate) q: u64,
    pub(crate) k: u64,
    pub(crate) v: u64,
    pub(crate) out: u64,
    pub(crate) q_offsets: u64,
    pub(crate) k_offsets: u64,
    pub(crate) q_tokens: usize,
    pub(crate) k_tokens: usize,
    pub(crate) batch: usize,
    pub(crate) max_q: usize,
    pub(crate) strides: [i64; 12],
}

#[derive(Clone, Copy, Debug)]
struct Profile {
    name: &'static str,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    causal: bool,
    window: Window,
}

impl Profile {
    const fn new(
        name: &'static str,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        causal: bool,
        window: Window,
    ) -> Self {
        Self {
            name,
            heads,
            kv_heads,
            head_dim,
            causal,
            window,
        }
    }

    fn matches(&self, spec: &AttentionSpec) -> bool {
        if spec.heads == 0 || spec.kv_heads == 0 {
            return false;
        }
        let (Some(export_group), Some(request_group)) = (
            self.heads.checked_mul(spec.kv_heads),
            spec.heads.checked_mul(self.kv_heads),
        ) else {
            return false;
        };
        export_group == request_group
            && self.head_dim == spec.head_dim
            && self.causal == spec.causal
            && self.window == spec.window
    }
}

// Exact ABI7 export order. Local windows follow build.py: noncausal locals are
// `[64, 64]` and causal locals are `[64, 0]`.
const LEGACY_PROFILE_COUNT: usize = 19;
const PROFILES: [Profile; 20] = [
    Profile::new("bert", 16, 16, 64, false, Window::Global),
    Profile::new(
        "modern_local",
        12,
        12,
        64,
        false,
        Window::Local {
            left: 64,
            right: 64,
        },
    ),
    Profile::new("qwen", 32, 8, 128, true, Window::Global),
    Profile::new("voyage", 16, 8, 128, false, Window::Global),
    Profile::new("gqa6", 12, 2, 128, false, Window::Global),
    Profile::new("gqa8", 32, 4, 128, false, Window::Global),
    Profile::new("qwen35", 16, 4, 256, false, Window::Global),
    Profile::new("mha32", 12, 12, 32, false, Window::Global),
    Profile::new("media72", 16, 16, 72, false, Window::Global),
    Profile::new("mha128", 6, 6, 128, false, Window::Global),
    Profile::new(
        "local128",
        6,
        6,
        128,
        false,
        Window::Local {
            left: 64,
            right: 64,
        },
    ),
    Profile::new("causal_gqa2", 16, 8, 128, true, Window::Global),
    Profile::new("gqa3", 24, 8, 128, false, Window::Global),
    Profile::new("gqa4", 32, 8, 128, false, Window::Global),
    Profile::new("gqa7", 28, 4, 128, false, Window::Global),
    Profile::new(
        "causal_local128",
        32,
        8,
        128,
        true,
        Window::Local { left: 64, right: 0 },
    ),
    Profile::new("wide_gqa6", 24, 4, 256, false, Window::Global),
    Profile::new("causal_wide6", 24, 4, 256, true, Window::Global),
    Profile::new("causal_wide4", 16, 4, 256, true, Window::Global),
    Profile::new("causal_gqa8_d64", 32, 4, 64, true, Window::Global),
];

fn start(e: impl std::fmt::Display) -> BackendError {
    BackendError::Start(format!("Oxide FA4: {e}"))
}
fn infer(e: impl std::fmt::Display) -> BackendError {
    BackendError::Inference(format!("Oxide FA4: {e}"))
}

fn profile_index(spec: &AttentionSpec) -> Option<usize> {
    PROFILES.iter().position(|profile| profile.matches(spec))
}

fn parse_window(value: Option<&serde_json::Value>) -> Option<Window> {
    let items = value?.as_array()?;
    if items.len() != 2 {
        return None;
    }
    match (items.first()?, items.get(1)?) {
        (serde_json::Value::Null, serde_json::Value::Null) => Some(Window::Global),
        (serde_json::Value::Number(left), serde_json::Value::Number(right)) => {
            let left = i32::try_from(left.as_i64()?).ok()?;
            let right = i32::try_from(right.as_i64()?).ok()?;
            if left < 0 || right < 0 {
                return None;
            }
            Some(Window::Local { left, right })
        }
        _ => None,
    }
}

fn kernel_matches(kernel: &serde_json::Value, profile: &Profile, dtype: DType) -> bool {
    let symbol = format!("fa4_{}_{}", profile.name, dtype.name());
    kernel.get("symbol").and_then(serde_json::Value::as_str) == Some(symbol.as_str())
        && kernel.get("heads").and_then(serde_json::Value::as_u64)
            == u64::try_from(profile.heads).ok()
        && kernel.get("kv_heads").and_then(serde_json::Value::as_u64)
            == u64::try_from(profile.kv_heads).ok()
        && kernel.get("d").and_then(serde_json::Value::as_u64)
            == u64::try_from(profile.head_dim).ok()
        && kernel.get("causal").and_then(serde_json::Value::as_bool) == Some(profile.causal)
        && parse_window(kernel.get("window")) == Some(profile.window)
        && kernel.get("dtype").and_then(serde_json::Value::as_str) == Some(dtype.name())
}

fn validate_profile_order(
    manifest: &serde_json::Value,
    dtype: DType,
) -> Result<usize, BackendError> {
    let kernels = manifest
        .get("kernels")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| start("manifest lacks a kernels array"))?;
    let deberta_symbol = format!("fa4_deberta_{}", dtype.name());
    let mut deberta_seen = false;
    let mut seen = 0usize;
    for kernel in kernels {
        if kernel.get("dtype").and_then(serde_json::Value::as_str) != Some(dtype.name()) {
            continue;
        }
        if kernel.get("symbol").and_then(serde_json::Value::as_str) == Some(deberta_symbol.as_str())
        {
            if deberta_seen {
                return Err(start(format!(
                    "manifest has duplicate DeBERTa {} exports",
                    dtype.name()
                )));
            }
            deberta_seen = true;
            continue;
        }
        if seen >= PROFILES.len() {
            return Err(start(format!(
                "manifest has unknown extra {} exports",
                dtype.name()
            )));
        }
        let profile = PROFILES
            .get(seen)
            .ok_or_else(|| start("manifest export count mismatch"))?;
        if !kernel_matches(kernel, profile, dtype) {
            return Err(start(format!(
                "manifest mode {seen} does not match exported {} {}",
                dtype.name(),
                profile.name
            )));
        }
        seen += 1;
    }
    if seen < LEGACY_PROFILE_COUNT {
        return Err(start(format!(
            "manifest lacks {} declared {} exports",
            LEGACY_PROFILE_COUNT,
            dtype.name()
        )));
    }
    Ok(seen)
}

fn mode_mask(declared: usize) -> Result<i32, BackendError> {
    let bits = u32::try_from(declared).map_err(start)?;
    let mask = 1i64
        .checked_shl(bits)
        .and_then(|value| value.checked_sub(1))
        .ok_or_else(|| start("provider mode count overflow"))?;
    i32::try_from(mask).map_err(start)
}

fn validate_mode_mask(declared: usize, modes: i32) -> Result<(), BackendError> {
    if modes != mode_mask(declared)? {
        return Err(start("provider mode mask does not match manifest exports"));
    }
    Ok(())
}

fn profile_selection(
    manifest: &serde_json::Value,
    spec: &AttentionSpec,
) -> Result<(usize, usize), BackendError> {
    let mode = profile_index(spec).ok_or_else(|| {
        start("selected attention grouping, dimension or mask has no ABI7 export")
    })?;
    let declared = validate_profile_order(manifest, spec.dtype)?;
    if mode >= declared {
        return Err(start(format!(
            "manifest lacks the selected {} export",
            spec.dtype.name()
        )));
    }
    Ok((mode, declared))
}

#[cfg(test)]
fn select_profile(
    manifest: &serde_json::Value,
    spec: &AttentionSpec,
) -> Result<usize, BackendError> {
    profile_selection(manifest, spec).map(|(mode, _)| mode)
}

fn resolve_scale(spec: &AttentionSpec) -> Result<f64, BackendError> {
    match spec.scale {
        Scale::RsqrtHeadDim => {
            if spec.head_dim == 0 {
                return Err(start("head dimension must be nonzero"));
            }
            Ok(1.0 / (spec.head_dim as f64).sqrt())
        }
        Scale::Explicit(scale) if scale.is_finite() && scale > 0.0 => Ok(scale),
        Scale::Explicit(_) => Err(start("attention scale must be finite and positive")),
    }
}

fn contiguous_strides(spec: &AttentionSpec) -> Option<[i64; 12]> {
    let head_dim = i64::try_from(spec.head_dim).ok()?;
    let q_width = i64::try_from(spec.heads.checked_mul(spec.head_dim)?).ok()?;
    let kv_width = i64::try_from(spec.kv_heads.checked_mul(spec.head_dim)?).ok()?;
    Some([
        q_width, head_dim, 1, kv_width, head_dim, 1, kv_width, head_dim, 1, q_width, head_dim, 1,
    ])
}

fn separate_qk_forbidden(
    capability: i32,
    head_dim: usize,
    q_tokens: usize,
    k_tokens: usize,
    q_offsets: u64,
    k_offsets: u64,
) -> bool {
    capability / 10 == 10 && head_dim == 256 && (q_tokens != k_tokens || q_offsets != k_offsets)
}

fn valid_stride(stride: &[i64], width: i64, head_dim: i64) -> bool {
    stride.len() == 3
        && stride[2] == 1
        && stride[1] == head_dim
        && stride[0] >= width
        && stride[0] % 8 == 0
}

fn check_batch(
    spec: &AttentionSpec,
    capability: i32,
    batch: &AttentionBatch,
) -> Result<(), BackendError> {
    if batch.q_tokens == 0
        || batch.k_tokens == 0
        || batch.batch == 0
        || batch.max_q == 0
        || batch.max_q > batch.q_tokens
    {
        return Err(infer("invalid attention geometry"));
    }
    if batch.batch > batch.q_tokens || batch.batch > batch.k_tokens {
        return Err(infer("batch exceeds packed tokens"));
    }
    if [batch.q, batch.k, batch.v, batch.out]
        .iter()
        .any(|pointer| *pointer == 0 || !pointer.is_multiple_of(16))
    {
        return Err(infer("invalid attention pointer alignment"));
    }
    if batch.q_offsets == 0
        || batch.k_offsets == 0
        || !batch.q_offsets.is_multiple_of(4)
        || !batch.k_offsets.is_multiple_of(4)
    {
        return Err(infer("invalid cumulative-length pointer alignment"));
    }
    let head_dim = i64::try_from(spec.head_dim).map_err(infer)?;
    let q_width = i64::try_from(
        spec.heads
            .checked_mul(spec.head_dim)
            .ok_or_else(|| infer("attention width overflow"))?,
    )
    .map_err(infer)?;
    let kv_width = i64::try_from(
        spec.kv_heads
            .checked_mul(spec.head_dim)
            .ok_or_else(|| infer("attention width overflow"))?,
    )
    .map_err(infer)?;
    let strides = &batch.strides;
    if !valid_stride(&strides[0..3], q_width, head_dim)
        || !valid_stride(&strides[3..6], kv_width, head_dim)
        || !valid_stride(&strides[6..9], kv_width, head_dim)
        || !valid_stride(&strides[9..12], q_width, head_dim)
    {
        return Err(infer("invalid attention element strides"));
    }
    if separate_qk_forbidden(
        capability,
        spec.head_dim,
        batch.q_tokens,
        batch.k_tokens,
        batch.q_offsets,
        batch.k_offsets,
    ) {
        return Err(infer("SM100 d256 attention requires shared Q/K boundaries"));
    }
    Ok(())
}

fn provider_error(error: Error, status: i32) -> BackendError {
    // SAFETY: native error is thread-local, NUL-terminated and consumed on this
    // same host thread while the selected library remains loaded.
    let pointer = unsafe { error() };
    if pointer.is_null() {
        infer(format!("provider status {status}"))
    } else {
        // SAFETY: the provider contract returns a valid NUL-terminated string.
        infer(unsafe { CStr::from_ptr(pointer) }.to_string_lossy())
    }
}

pub(crate) struct Fa4 {
    forward: Forward,
    error: Error,
    context: Arc<Context>,
    device: i32,
    capability: i32,
    mode: i32,
    dtype: i32,
    heads: i64,
    kv_heads: i64,
    scale: f64,
    left: i32,
    right: i32,
    spec: AttentionSpec,
    contiguous_strides: [i64; 12],
    _library: Library,
}
// SAFETY: the engine Mutex serializes calls and teardown, binds the retained
// primary context and drains pending work before releasing buffers/library.
unsafe impl Send for Fa4 {}
impl Fa4 {
    pub(crate) fn supports(spec: &AttentionSpec) -> bool {
        profile_index(spec).is_some() && resolve_scale(spec).is_ok()
    }
    pub(crate) fn new(
        context: Arc<Context>,
        root: &Path,
        spec: AttentionSpec,
    ) -> Result<Self, BackendError> {
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
        let (mode, declared) = profile_selection(&manifest, &spec)?;
        let mode = i32::try_from(mode).map_err(start)?;
        let scale = resolve_scale(&spec)?;
        let contiguous_strides =
            contiguous_strides(&spec).ok_or_else(|| start("attention strides overflow"))?;
        let (left, right) = match spec.window {
            Window::Global => (-1, -1),
            Window::Local { left, right } => (left, right),
        };
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
        if unsafe { portable() } != 1 {
            return Err(start("incompatible portable-host marker"));
        }
        if unsafe { architecture() } != capability {
            return Err(start("incompatible native compute capability"));
        }
        let modes = unsafe { modes() };
        validate_mode_mask(declared, modes)?;
        let selected = 1i32
            .checked_shl(u32::try_from(mode).map_err(start)?)
            .ok_or_else(|| start("provider mode index overflow"))?;
        if modes & selected == 0 {
            return Err(start("selected attention mode is not exported"));
        }
        Ok(Self {
            forward,
            error,
            device: i32::try_from(context.ordinal()).map_err(start)?,
            context,
            capability,
            mode,
            dtype: spec.dtype.code(),
            heads: i64::try_from(spec.heads).map_err(start)?,
            kv_heads: i64::try_from(spec.kv_heads).map_err(start)?,
            scale,
            left,
            right,
            spec,
            contiguous_strides,
            _library: library,
        })
    }

    /// Contiguous packed self-attention using the selected spec. The engine
    /// retains Q/K/V/output allocations and the int32 boundaries through stream
    /// completion. This preserves the existing BERT call shape.
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
        let batch = AttentionBatch {
            q,
            k,
            v,
            out,
            q_offsets: cumulative,
            k_offsets: cumulative,
            q_tokens: tokens,
            k_tokens: tokens,
            batch,
            max_q: max_len,
            strides: self.contiguous_strides,
        };
        // SAFETY: this wrapper applies the engine's contiguous self-attention
        // contract: same-context, non-aliased buffers retained through stream
        // completion, and packed int32 boundaries of `batch + 1` entries.
        unsafe { self.forward_batch(stream, &batch) }
    }

    /// Enqueue one generalized packed attention launch. No manifest or symbol
    /// lookup occurs here; mode, dtype, scale, mask and immutable geometry were
    /// selected and validated at load.
    ///
    /// # Safety
    ///
    /// `stream` must belong to this provider's context. The caller must keep
    /// all Q/K/V/output and cumulative-length device allocations alive and
    /// unmodified until this stream work completes, must provide `batch + 1`
    /// int32 entries at each offsets pointer, and must ensure output storage
    /// does not alias an input. All pointers must remain valid for the launch.
    pub(crate) unsafe fn forward_batch(
        &self,
        stream: &Stream,
        batch: &AttentionBatch,
    ) -> Result<(), BackendError> {
        if self.context.cu_ctx() != stream.context().cu_ctx() {
            return Err(infer("stream belongs to another CUDA context"));
        }
        check_batch(&self.spec, self.capability, batch)?;
        let total = i64::from(i32::try_from(batch.q_tokens).map_err(infer)?);
        let total_k = i64::from(i32::try_from(batch.k_tokens).map_err(infer)?);
        let count = i32::try_from(batch.batch).map_err(infer)?;
        count
            .checked_add(1)
            .ok_or_else(|| infer("boundary count overflow"))?;
        let max_q = i32::try_from(batch.max_q).map_err(infer)?;
        self.context.bind_to_thread().map_err(infer)?;
        // SAFETY: check_batch validated geometry, pointer alignment and the
        // element strides. The selected profile is immutable, and the caller
        // upholds the lifetime/context contract documented above. Stack
        // descriptors are copied by the synchronous native launch call.
        let status = unsafe {
            (self.forward)(
                self.mode,
                self.dtype,
                self.device,
                stream.cu_stream().cast(),
                batch.q as *mut c_void,
                batch.k as *mut c_void,
                batch.v as *mut c_void,
                batch.out as *mut c_void,
                batch.q_offsets as *mut c_void,
                batch.k_offsets as *mut c_void,
                total,
                total_k,
                i64::from(count),
                self.heads,
                self.kv_heads,
                self.scale,
                self.left,
                self.right,
                batch.strides.as_ptr(),
                max_q,
            )
        };
        if status != 0 {
            return Err(provider_error(self.error, status));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuda_core::{DeviceBuffer, DeviceCopy};
    use half::{bf16, f16};
    use serde_json::{json, Value};
    use std::path::PathBuf;

    fn spec_for(profile: &Profile, dtype: DType) -> AttentionSpec {
        AttentionSpec {
            heads: profile.heads,
            kv_heads: profile.kv_heads,
            head_dim: profile.head_dim,
            causal: profile.causal,
            window: profile.window,
            dtype,
            scale: Scale::RsqrtHeadDim,
        }
    }

    fn kernel(profile: &Profile, dtype: DType) -> Value {
        let window = match profile.window {
            Window::Global => json!([null, null]),
            Window::Local { left, right } => json!([left, right]),
        };
        json!({
            "symbol": format!("fa4_{}_{}", profile.name, dtype.name()),
            "heads": profile.heads,
            "kv_heads": profile.kv_heads,
            "d": profile.head_dim,
            "causal": profile.causal,
            "window": window,
            "dtype": dtype.name(),
        })
    }

    fn manifest_for(dtypes: &[DType]) -> Value {
        manifest_prefix(dtypes, PROFILES.len())
    }

    fn manifest_prefix(dtypes: &[DType], count: usize) -> Value {
        let mut kernels = Vec::new();
        for dtype in dtypes {
            for profile in PROFILES.iter().take(count) {
                kernels.push(kernel(profile, *dtype));
            }
        }
        json!({"kernels": kernels})
    }

    fn mha32() -> Profile {
        PROFILES[7]
    }

    fn appended_profile() -> Profile {
        PROFILES[LEGACY_PROFILE_COUNT]
    }

    fn deberta_kernel(dtype: DType) -> Value {
        json!({
            "symbol": format!("fa4_deberta_{}", dtype.name()),
            "heads": 12,
            "kv_heads": 12,
            "d": 64,
            "causal": false,
            "window": json!([null, null]),
            "dtype": dtype.name(),
        })
    }

    fn fake_batch(spec: &AttentionSpec, q_tokens: usize, k_tokens: usize) -> AttentionBatch {
        AttentionBatch {
            q: 0x1000,
            k: 0x2000,
            v: 0x3000,
            out: 0x4000,
            q_offsets: 0x5000,
            k_offsets: if q_tokens == k_tokens { 0x5000 } else { 0x6000 },
            q_tokens,
            k_tokens,
            batch: 1,
            max_q: q_tokens,
            strides: contiguous_strides(spec).unwrap_or([0; 12]),
        }
    }

    #[test]
    fn matches_every_declared_profile_and_dtype() {
        let manifest = manifest_for(&[DType::F16, DType::Bf16]);
        for dtype in [DType::F16, DType::Bf16] {
            for (mode, profile) in PROFILES.iter().enumerate() {
                let spec = spec_for(profile, dtype);
                assert_eq!(select_profile(&manifest, &spec).ok(), Some(mode));
            }
        }
    }

    #[test]
    fn accepts_full_legacy_19_and_appended_20_export_prefixes() {
        let mut legacy = manifest_prefix(&[DType::F16, DType::Bf16], LEGACY_PROFILE_COUNT);
        let mut appended = manifest_for(&[DType::F16, DType::Bf16]);
        for dtype in [DType::F16, DType::Bf16] {
            legacy["kernels"]
                .as_array_mut()
                .expect("test manifest has kernels")
                .push(deberta_kernel(dtype));
            appended["kernels"]
                .as_array_mut()
                .expect("test manifest has kernels")
                .push(deberta_kernel(dtype));
        }
        for dtype in [DType::F16, DType::Bf16] {
            for (mode, profile) in PROFILES.iter().take(LEGACY_PROFILE_COUNT).enumerate() {
                let spec = spec_for(profile, dtype);
                assert_eq!(select_profile(&legacy, &spec).ok(), Some(mode));
                assert_eq!(select_profile(&appended, &spec).ok(), Some(mode));
            }
            assert!(select_profile(&legacy, &spec_for(&appended_profile(), dtype)).is_err());
            assert_eq!(
                select_profile(&appended, &spec_for(&appended_profile(), dtype)).ok(),
                Some(LEGACY_PROFILE_COUNT)
            );
        }
        assert!(validate_mode_mask(
            LEGACY_PROFILE_COUNT,
            mode_mask(LEGACY_PROFILE_COUNT).expect("legacy mode mask")
        )
        .is_ok());
        assert!(validate_mode_mask(
            PROFILES.len(),
            mode_mask(PROFILES.len()).expect("appended mode mask")
        )
        .is_ok());
        assert!(validate_mode_mask(
            LEGACY_PROFILE_COUNT,
            mode_mask(PROFILES.len()).expect("appended mode mask")
        )
        .is_err());
        assert!(validate_mode_mask(
            PROFILES.len(),
            mode_mask(LEGACY_PROFILE_COUNT).expect("legacy mode mask")
        )
        .is_err());
        assert!(validate_mode_mask(
            PROFILES.len(),
            mode_mask(PROFILES.len()).expect("appended mode mask") | (1 << 25)
        )
        .is_err());
    }

    #[test]
    fn selects_causal_d64_gqa8_only_for_matching_ratio_dimension_and_mask() {
        let manifest = manifest_for(&[DType::F16, DType::Bf16]);
        let profile = appended_profile();
        for dtype in [DType::F16, DType::Bf16] {
            let mode = Some(LEGACY_PROFILE_COUNT);
            assert_eq!(
                select_profile(&manifest, &spec_for(&profile, dtype)).ok(),
                mode
            );
            for (heads, kv_heads) in [(32, 4), (16, 2), (8, 1), (64, 8)] {
                let mut spec = spec_for(&profile, dtype);
                spec.heads = heads;
                spec.kv_heads = kv_heads;
                assert_eq!(select_profile(&manifest, &spec).ok(), mode);
            }
            for (heads, kv_heads) in [(32, 8), (24, 4), (16, 4)] {
                let mut spec = spec_for(&profile, dtype);
                spec.heads = heads;
                spec.kv_heads = kv_heads;
                assert!(select_profile(&manifest, &spec).is_err());
            }
            let mut wrong_dim = spec_for(&profile, dtype);
            wrong_dim.head_dim = 128;
            assert!(select_profile(&manifest, &wrong_dim).is_err());
            let mut noncausal = spec_for(&profile, dtype);
            noncausal.causal = false;
            assert!(select_profile(&manifest, &noncausal).is_err());
            let mut local = spec_for(&profile, dtype);
            local.window = Window::Local { left: 64, right: 0 };
            assert!(select_profile(&manifest, &local).is_err());
        }
    }

    #[test]
    fn rejects_truncated_reordered_tampered_and_unknown_exports() {
        let truncated = manifest_prefix(&[DType::F16, DType::Bf16], LEGACY_PROFILE_COUNT - 1);
        assert!(select_profile(&truncated, &spec_for(&mha32(), DType::F16)).is_err());

        let mut reordered = manifest_for(&[DType::F16, DType::Bf16]);
        reordered["kernels"]
            .as_array_mut()
            .expect("test manifest has kernels")
            .swap(18, 19);
        assert!(select_profile(&reordered, &spec_for(&mha32(), DType::F16)).is_err());
        assert!(select_profile(&reordered, &spec_for(&appended_profile(), DType::F16)).is_err());
        assert!(select_profile(&reordered, &spec_for(&mha32(), DType::Bf16)).is_ok());
        assert!(select_profile(&reordered, &spec_for(&appended_profile(), DType::Bf16)).is_ok());

        for (field, value) in [
            ("symbol", json!("fa4_causal_gqa8_d64_extra")),
            ("heads", json!(16)),
            ("kv_heads", json!(8)),
            ("d", json!(128)),
            ("causal", json!(false)),
            ("window", json!([64, 0])),
        ] {
            let mut manifest = manifest_for(&[DType::F16, DType::Bf16]);
            manifest["kernels"]
                .as_array_mut()
                .expect("test manifest has kernels")[19][field] = value;
            assert!(
                select_profile(&manifest, &spec_for(&appended_profile(), DType::F16)).is_err(),
                "accepted corrupt {field}"
            );
        }

        let mut extra = manifest_for(&[DType::F16, DType::Bf16]);
        extra["kernels"]
            .as_array_mut()
            .expect("test manifest has kernels")
            .push(kernel(&appended_profile(), DType::F16));
        assert!(select_profile(&extra, &spec_for(&appended_profile(), DType::F16)).is_err());

        let mut unknown = manifest_for(&[DType::F16, DType::Bf16]);
        unknown["kernels"]
            .as_array_mut()
            .expect("test manifest has kernels")[19] = kernel(&mha32(), DType::F16);
        assert!(select_profile(&unknown, &spec_for(&appended_profile(), DType::F16)).is_err());

        let mut duplicate_deberta = manifest_for(&[DType::F16, DType::Bf16]);
        let kernels = duplicate_deberta["kernels"]
            .as_array_mut()
            .expect("test manifest has kernels");
        kernels.push(deberta_kernel(DType::F16));
        kernels.push(deberta_kernel(DType::F16));
        assert!(select_profile(&duplicate_deberta, &spec_for(&mha32(), DType::F16)).is_err());
    }

    #[test]
    fn selects_runtime_head_counts_with_original_manifest_exemplars() {
        let manifest = manifest_for(&[DType::F16, DType::Bf16]);
        for dtype in [DType::F16, DType::Bf16] {
            for (mode, heads, kv_heads) in [
                (0, 12, 12),
                (0, 24, 24),
                (9, 32, 32),
                (3, 24, 12),
                (2, 16, 4),
                (4, 6, 1),
                (19, 32, 4),
                (19, 16, 2),
                (19, 8, 1),
            ] {
                let mut spec = spec_for(&PROFILES[mode], dtype);
                spec.heads = heads;
                spec.kv_heads = kv_heads;
                assert_eq!(select_profile(&manifest, &spec).ok(), Some(mode));
            }
        }
    }

    #[test]
    fn rejects_invalid_runtime_head_grouping() {
        let manifest = manifest_for(&[DType::F16, DType::Bf16]);
        for dtype in [DType::F16, DType::Bf16] {
            let base = spec_for(&mha32(), dtype);
            for (heads, kv_heads) in [
                (0, 0),
                (0, 12),
                (12, 0),
                (13, 12),
                (12, 13),
                (24, 12),
                (usize::MAX, usize::MAX),
            ] {
                let mut spec = base;
                spec.heads = heads;
                spec.kv_heads = kv_heads;
                assert!(select_profile(&manifest, &spec).is_err());
            }
        }
    }

    #[test]
    fn rejects_reordered_export_order() {
        let mut manifest = manifest_for(&[DType::F16, DType::Bf16]);
        let kernels = manifest["kernels"]
            .as_array_mut()
            .expect("test manifest has kernels");
        kernels.swap(7, 8);
        assert!(select_profile(&manifest, &spec_for(&mha32(), DType::F16)).is_err());
        assert!(select_profile(&manifest, &spec_for(&mha32(), DType::Bf16)).is_ok());
    }

    #[test]
    fn rejects_corrupt_export_metadata() {
        for (field, value) in [
            ("symbol", json!("fa4_mha32_fp16_extra")),
            ("heads", json!(16)),
            ("kv_heads", json!(6)),
            ("d", json!(64)),
            ("causal", json!(true)),
            ("window", json!([64, 64])),
            ("dtype", json!("bf16")),
        ] {
            let mut manifest = manifest_for(&[DType::F16, DType::Bf16]);
            let kernel = &mut manifest["kernels"]
                .as_array_mut()
                .expect("test manifest has kernels")[7];
            kernel[field] = value;
            assert!(
                select_profile(&manifest, &spec_for(&mha32(), DType::F16)).is_err(),
                "accepted corrupt {field}"
            );
        }
    }

    #[test]
    fn rejects_unsupported_geometry_and_dtype_at_load() {
        let manifest = manifest_for(&[DType::F16, DType::Bf16]);
        let base = spec_for(&mha32(), DType::F16);

        let mut wrong_heads = base;
        wrong_heads.heads = 16;
        assert!(select_profile(&manifest, &wrong_heads).is_err());

        let mut wrong_kv = base;
        wrong_kv.kv_heads = 4;
        assert!(select_profile(&manifest, &wrong_kv).is_err());

        let mut wrong_window = base;
        wrong_window.window = Window::Local {
            left: 64,
            right: 64,
        };
        assert!(select_profile(&manifest, &wrong_window).is_err());

        let mut wrong_causal = base;
        wrong_causal.causal = true;
        assert!(select_profile(&manifest, &wrong_causal).is_err());

        assert!(select_profile(
            &manifest_for(&[DType::F16]),
            &spec_for(&mha32(), DType::Bf16)
        )
        .is_err());
    }

    #[test]
    fn rejects_sm100_d256_separate_boundaries_but_allows_sm120() {
        let profile = PROFILES[6];
        let spec = spec_for(&profile, DType::Bf16);
        let shared = fake_batch(&spec, 8, 8);
        let separate = fake_batch(&spec, 8, 16);

        assert!(check_batch(&spec, 100, &separate).is_err());
        assert!(check_batch(&spec, 100, &shared).is_ok());
        assert!(check_batch(&spec, 120, &separate).is_ok());

        let mut split = shared;
        split.k_offsets = 0x6000;
        assert!(check_batch(&spec, 100, &split).is_err());

        let mha32 = spec_for(&mha32(), DType::Bf16);
        let separate = fake_batch(&mha32, 8, 16);
        assert!(check_batch(&mha32, 100, &separate).is_ok());
    }

    #[test]
    fn validates_scale_convention_at_load() {
        let mut spec = spec_for(&mha32(), DType::Bf16);
        assert!(resolve_scale(&spec).is_ok());

        spec.scale = Scale::Explicit(0.25);
        assert_eq!(resolve_scale(&spec).ok(), Some(0.25));
        for invalid in [0.0, -1.0, f64::INFINITY, f64::NAN] {
            spec.scale = Scale::Explicit(invalid);
            assert!(resolve_scale(&spec).is_err());
        }
    }

    trait TestElem: DeviceCopy + Copy + Default {
        const DTYPE: DType;
        const TOLERANCE: f32;
        fn encode(value: f32) -> Self;
        fn decode(self) -> f32;
    }

    impl TestElem for f16 {
        const DTYPE: DType = DType::F16;
        const TOLERANCE: f32 = 3.0e-2;

        fn encode(value: f32) -> Self {
            f16::from_f32(value)
        }

        fn decode(self) -> f32 {
            self.to_f32()
        }
    }

    impl TestElem for bf16 {
        const DTYPE: DType = DType::Bf16;
        const TOLERANCE: f32 = 8.0e-2;

        fn encode(value: f32) -> Self {
            bf16::from_f32(value)
        }

        fn decode(self) -> f32 {
            self.to_f32()
        }
    }

    fn provider_root() -> PathBuf {
        match std::env::var_os("INFINITY_FA4_ROOT") {
            Some(path) if !path.is_empty() => PathBuf::from(path),
            _ => panic!("set INFINITY_FA4_ROOT to an ABI7 native provider bundle"),
        }
    }

    fn pattern(seed: usize, len: usize) -> Vec<f32> {
        (0..len)
            .map(|index| (((index * 37 + seed * 53) % 61) as f32 - 30.0) / 32.0)
            .collect()
    }

    fn upload<T: TestElem>(stream: &Stream, values: &[f32]) -> DeviceBuffer<T> {
        let encoded = values.iter().copied().map(T::encode).collect::<Vec<_>>();
        DeviceBuffer::from_host(stream, &encoded).expect("upload test attention operand")
    }

    fn download<T: TestElem>(stream: &Stream, buffer: &DeviceBuffer<T>) -> Vec<f32> {
        let mut host = vec![T::default(); buffer.len()];
        buffer
            .copy_to_host(stream, &mut host)
            .expect("download test attention output");
        host.into_iter().map(T::decode).collect()
    }

    fn cpu_attention(
        spec: &AttentionSpec,
        scale: f64,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        q_offsets: &[i32],
        k_offsets: &[i32],
    ) -> Vec<f32> {
        let d = spec.head_dim;
        let heads = spec.heads;
        let kv_heads = spec.kv_heads;
        let group = heads / kv_heads;
        let mut output = vec![0.0f32; q.len()];
        for sequence in 0..q_offsets.len().saturating_sub(1) {
            let q_start = q_offsets[sequence] as usize;
            let k_start = k_offsets[sequence] as usize;
            let q_len = q_offsets[sequence + 1] as usize - q_start;
            let k_len = k_offsets[sequence + 1] as usize - k_start;
            let offset = k_len as isize - q_len as isize;
            for row in 0..q_len {
                for head in 0..heads {
                    let kv_head = head / group;
                    let mut scores = Vec::with_capacity(k_len);
                    for col in 0..k_len {
                        let col_position = col as isize;
                        let masked = match spec.window {
                            Window::Global => false,
                            Window::Local { left, right } => {
                                let lower = row as isize + offset - left as isize;
                                let upper =
                                    (row as isize + offset + right as isize).min(k_len as isize);
                                col_position < lower || col_position > upper
                            }
                        };
                        if masked || (spec.causal && col_position > row as isize + offset) {
                            continue;
                        }
                        let q_base = (q_start + row) * heads * d + head * d;
                        let k_base = (k_start + col) * kv_heads * d + kv_head * d;
                        let mut dot = 0.0f64;
                        for dim in 0..d {
                            dot += q[q_base + dim] as f64 * k[k_base + dim] as f64;
                        }
                        scores.push((col, dot * scale));
                    }
                    let out_base = (q_start + row) * heads * d + head * d;
                    let Some(maximum) = scores.iter().map(|(_, score)| *score).reduce(f64::max)
                    else {
                        continue;
                    };
                    let mut denominator = 0.0f64;
                    let mut accumulator = vec![0.0f64; d];
                    for (col, score) in scores {
                        let weight = (score - maximum).exp();
                        denominator += weight;
                        let v_base = (k_start + col) * kv_heads * d + kv_head * d;
                        for dim in 0..d {
                            accumulator[dim] += weight * v[v_base + dim] as f64;
                        }
                    }
                    for dim in 0..d {
                        output[out_base + dim] = (accumulator[dim] / denominator) as f32;
                    }
                }
            }
        }
        output
    }

    fn cpu_scale(spec: &AttentionSpec) -> f64 {
        resolve_scale(spec).expect("test scale")
    }

    #[allow(clippy::too_many_arguments)]
    fn run_case<T: TestElem>(
        context: &Arc<Context>,
        root: &std::path::Path,
        stream: &Stream,
        spec: AttentionSpec,
        q_offsets: &[i32],
        k_offsets: &[i32],
        max_q: usize,
        wrapper: bool,
    ) {
        assert_eq!(spec.dtype, T::DTYPE);
        let q_tokens = *q_offsets.last().expect("test query boundaries") as usize;
        let k_tokens = *k_offsets.last().expect("test key boundaries") as usize;
        let batch = q_offsets.len() - 1;
        let d = spec.head_dim;
        let heads = spec.heads;
        let kv_heads = spec.kv_heads;
        let q_host = pattern(1, q_tokens * heads * d);
        let k_host = pattern(2, k_tokens * kv_heads * d);
        let v_host = pattern(3, k_tokens * kv_heads * d);
        let padding = if wrapper { 0 } else { 8 };
        let padded = |values: &[f32], width: usize| {
            values
                .chunks(width)
                .flat_map(|row| row.iter().copied().chain(std::iter::repeat_n(0.0, padding)))
                .collect::<Vec<_>>()
        };
        let q = upload::<T>(stream, &padded(&q_host, heads * d));
        let k = upload::<T>(stream, &padded(&k_host, kv_heads * d));
        let v = upload::<T>(stream, &padded(&v_host, kv_heads * d));
        let out = DeviceBuffer::<T>::zeroed(stream, q_tokens * (heads * d + padding))
            .expect("allocate test attention output");
        let q_offsets_device =
            DeviceBuffer::from_host(stream, q_offsets).expect("upload query boundaries");
        let k_offsets_device =
            DeviceBuffer::from_host(stream, k_offsets).expect("upload key boundaries");
        let fa4 = Fa4::new(context.clone(), root, spec)
            .unwrap_or_else(|error| panic!("load selected FA4 provider: {error}"));
        if wrapper {
            assert_eq!(q_offsets, k_offsets);
            fa4.forward(
                stream,
                q.cu_deviceptr(),
                k.cu_deviceptr(),
                v.cu_deviceptr(),
                out.cu_deviceptr(),
                q_offsets_device.cu_deviceptr(),
                q_tokens,
                batch,
                max_q,
            )
            .expect("wrapper attention launch");
        } else {
            let q_width = (heads * d + padding) as i64;
            let kv_width = (kv_heads * d + padding) as i64;
            let launch = AttentionBatch {
                q: q.cu_deviceptr(),
                k: k.cu_deviceptr(),
                v: v.cu_deviceptr(),
                out: out.cu_deviceptr(),
                q_offsets: q_offsets_device.cu_deviceptr(),
                k_offsets: k_offsets_device.cu_deviceptr(),
                q_tokens,
                k_tokens,
                batch,
                max_q,
                strides: [
                    q_width, d as i64, 1, kv_width, d as i64, 1, kv_width, d as i64, 1, q_width,
                    d as i64, 1,
                ],
            };
            // SAFETY: all four buffers and both boundary arrays are retained
            // until the synchronization below; output does not alias inputs.
            unsafe { fa4.forward_batch(stream, &launch) }.expect("generalized attention launch");
        }
        let raw = download(stream, &out);
        let got = raw
            .chunks(heads * d + padding)
            .flat_map(|row| row[..heads * d].iter().copied())
            .collect::<Vec<_>>();
        assert!(raw
            .chunks(heads * d + padding)
            .all(|row| row[heads * d..].iter().all(|v| *v == 0.0)));
        let want = cpu_attention(
            &spec,
            cpu_scale(&spec),
            &q_host,
            &k_host,
            &v_host,
            q_offsets,
            k_offsets,
        );
        assert_eq!(got.len(), want.len());
        for (index, (got, want)) in got.iter().zip(want.iter()).enumerate() {
            let tolerance = T::TOLERANCE * (1.0 + want.abs().max(got.abs()));
            assert!(
                (got - want).abs() <= tolerance,
                "{} mismatch at {index}: got {got}, want {want}",
                spec.dtype.name()
            );
        }
    }

    fn gpu_setup() -> (PathBuf, Arc<Context>, Arc<Stream>) {
        let root = provider_root();
        let context = Context::new(0).expect("create CUDA context on GPU0");
        let stream = context.new_stream().expect("create CUDA stream");
        (root, context, stream)
    }

    #[test]
    #[ignore = "requires INFINITY_FA4_ROOT and a matching CUDA device"]
    fn gpu_loads_every_declared_export() {
        let (root, context, _stream) = gpu_setup();
        for dtype in [DType::F16, DType::Bf16] {
            for profile in &PROFILES {
                Fa4::new(context.clone(), &root, spec_for(profile, dtype)).unwrap_or_else(
                    |error| panic!("load {} {}: {error}", profile.name, dtype.name()),
                );
            }
        }
    }

    #[test]
    #[ignore = "requires INFINITY_FA4_ROOT and a matching CUDA device"]
    fn gpu_every_export_computes_small_ragged_attention() {
        let (root, context, stream) = gpu_setup();
        let offsets = [0i32, 3, 7];
        for profile in &PROFILES {
            run_case::<f16>(
                &context,
                &root,
                &stream,
                spec_for(profile, DType::F16),
                &offsets,
                &offsets,
                4,
                false,
            );
            run_case::<bf16>(
                &context,
                &root,
                &stream,
                spec_for(profile, DType::Bf16),
                &offsets,
                &offsets,
                4,
                false,
            );
        }
    }

    #[test]
    #[ignore = "requires INFINITY_FA4_ROOT and a matching CUDA device"]
    fn gpu_runtime_head_counts_match_cpu_reference() {
        let (root, context, stream) = gpu_setup();
        let offsets = [0i32, 3, 133];
        for (mode, heads, kv_heads) in [
            (0, 12, 12),
            (0, 24, 24),
            (9, 32, 32),
            (3, 24, 12),
            (2, 16, 4),
            (4, 6, 1),
        ] {
            for wrapper in [true, false] {
                let mut spec = spec_for(&PROFILES[mode], DType::F16);
                spec.heads = heads;
                spec.kv_heads = kv_heads;
                run_case::<f16>(
                    &context, &root, &stream, spec, &offsets, &offsets, 130, wrapper,
                );
                spec.dtype = DType::Bf16;
                run_case::<bf16>(
                    &context, &root, &stream, spec, &offsets, &offsets, 130, wrapper,
                );
            }
        }
    }

    #[test]
    #[ignore = "requires INFINITY_FA4_ROOT and a matching CUDA device"]
    fn gpu_causal_d64_gqa8_matches_cpu_reference_across_tile_tails() {
        let (root, context, stream) = gpu_setup();
        let ragged = [0i32, 3, 133];
        let longer = [0i32, 130, 390];
        let tile_tail = [0i32, 1, 257];
        for dtype in [DType::F16, DType::Bf16] {
            let spec = spec_for(&appended_profile(), dtype);
            for (offsets, max_q) in [
                (&ragged[..], 130),
                (&longer[..], 260),
                (&tile_tail[..], 256),
            ] {
                for wrapper in [true, false] {
                    match dtype {
                        DType::F16 => run_case::<f16>(
                            &context, &root, &stream, spec, offsets, offsets, max_q, wrapper,
                        ),
                        DType::Bf16 => run_case::<bf16>(
                            &context, &root, &stream, spec, offsets, offsets, max_q, wrapper,
                        ),
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "requires INFINITY_FA4_ROOT and a matching CUDA device"]
    fn gpu_self_attention_matches_cpu_reference() {
        let (root, context, stream) = gpu_setup();
        let offsets = [0i32, 70, 135];
        let cases = [
            (PROFILES[7], DType::Bf16, true),
            (PROFILES[0], DType::F16, true),
            (PROFILES[1], DType::Bf16, false),
            (PROFILES[2], DType::F16, false),
            (PROFILES[4], DType::Bf16, false),
            (PROFILES[15], DType::Bf16, false),
        ];
        for (profile, dtype, wrapper) in cases {
            let spec = spec_for(&profile, dtype);
            match dtype {
                DType::F16 => run_case::<f16>(
                    &context, &root, &stream, spec, &offsets, &offsets, 70, wrapper,
                ),
                DType::Bf16 => run_case::<bf16>(
                    &context, &root, &stream, spec, &offsets, &offsets, 70, wrapper,
                ),
            }
        }
    }

    #[test]
    #[ignore = "requires INFINITY_FA4_ROOT and a matching CUDA device"]
    fn gpu_separate_qk_boundaries_match_cpu_reference() {
        let (root, context, stream) = gpu_setup();
        let q_offsets = [0i32, 40, 96];
        let k_offsets = [0i32, 70, 160];
        let spec = spec_for(&PROFILES[2], DType::F16);
        run_case::<f16>(
            &context, &root, &stream, spec, &q_offsets, &k_offsets, 56, false,
        );

        let q_offsets = [0i32, 37, 91];
        let k_offsets = [0i32, 55, 140];
        let spec = spec_for(&PROFILES[4], DType::Bf16);
        run_case::<bf16>(
            &context, &root, &stream, spec, &q_offsets, &k_offsets, 54, false,
        );

        let q_offsets = [0i32, 50, 90];
        let k_offsets = [0i32, 70, 150];
        let spec = spec_for(&PROFILES[1], DType::F16);
        run_case::<f16>(
            &context, &root, &stream, spec, &q_offsets, &k_offsets, 50, false,
        );
    }
}
