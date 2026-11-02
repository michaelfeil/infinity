// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Torch-free entry to the explicitly selected InfinityFA4 AOT provider.
//!
//! Profile order and exported windows mirror the repository's ABI7
//! `vendor/infinity_fa4/profiles.inc` and `build.py`. A bundle is accepted only
//! when its manifest exports a declared prefix of the canonical append registry
//! (legacy19, published20, private23, registry24, or canonical26) in that known
//! order for the selected dtype. Absolute head counts are runtime tensor
//! dimensions; selection uses the exported Q/KV grouping with the exact ratio,
//! dimension and mask kind. Global masks stay exact. Local profiles may serve
//! safe runtime extents of the same causal/local kind because FA4 passes the
//! left/right bounds as runtime scalars.
//!
//! Selection is loaded-prefix aware: it considers only the manifest's declared
//! ordinary prefix. Exact extents are preferred inside that prefix, then a
//! compatible local profile from the same prefix, otherwise a typed
//! missing-profile error. An appended exact window must never hide a compatible
//! older local export.
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

impl Window {
    /// Validate a local window as FA4 normalizes it for the requested semantic
    /// causal bit, and return its runtime bounds.
    ///
    /// FA4's `_resolve_causal_local_window` forces `window_size_right = 0` for a
    /// causal request, then selects the local path when a left bound is present.
    /// The kernel therefore sees its own `causal = false, local = true,
    /// right = 0` form for a semantic causal-local profile. Keep the provider's
    /// causal bit separate from those normalized kernel flags: a causal local
    /// request must already carry `right = 0`, while a noncausal local request
    /// must not use that causal-local form (`right > 0`).
    fn canonical_local_extents(self, causal: bool) -> Option<(i32, i32)> {
        let Self::Local { left, right } = self else {
            return None;
        };
        if left < 0 || right < 0 {
            return None;
        }
        left.checked_add(right)?.checked_add(1)?;
        if causal {
            (right == 0).then_some((left, right))
        } else {
            (right > 0).then_some((left, right))
        }
    }

    /// Global masks are exact. Two local masks describe the same cold kernel
    /// when both validate for the same causal kind, regardless of extents.
    fn matches_mask(self, requested: Self, causal: bool) -> bool {
        match (self, requested) {
            (Self::Global, Self::Global) => true,
            (Self::Local { .. }, Self::Local { .. }) => {
                self.canonical_local_extents(causal).is_some()
                    && requested.canonical_local_extents(causal).is_some()
            }
            _ => false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Scale {
    RsqrtHeadDim,
    #[allow(dead_code)] // Needed by model-specific temperatures, tested independently.
    Explicit(f64),
}

/// Provider geometry and scale convention selected once at load. `window`
/// distinguishes global and local masks. Global masks are exact; local requests
/// may use safe runtime extents of the same causal/local kind. A causal local
/// request must use `right = 0`; a noncausal local request must use `right > 0`.
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
            && self.window.matches_mask(spec.window, spec.causal)
    }
}

// Exact ABI7 export order. Local windows follow build.py: noncausal locals are
// `[64, 64]` and causal locals are `[64, 0]`, except Mistral `[4095, 0]`,
// Gemma3 `[511, 0]`, and EmbeddingGemma2 `[512, 512]`.
const LEGACY_PROFILE_COUNT: usize = 19;
const PUBLISHED_PROFILE_COUNT: usize = 20;
const PRIVATE23_PROFILE_COUNT: usize = 23;
const REGISTRY24_PROFILE_COUNT: usize = 24;
const CANONICAL26_PROFILE_COUNT: usize = 26;
const PROFILES: [Profile; CANONICAL26_PROFILE_COUNT] = [
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
    // Private23 prefix, preserved as-is. These are the canonical mode indices
    // shared by every descendant packet; standalone packets that put other
    // profiles here are rejected by strict export-order validation.
    Profile::new(
        "mistral_local4096",
        32,
        8,
        128,
        true,
        Window::Local {
            left: 4095,
            right: 0,
        },
    ),
    Profile::new(
        "gemma3_local512",
        4,
        1,
        256,
        true,
        Window::Local {
            left: 511,
            right: 0,
        },
    ),
    Profile::new("causal_gqa8", 32, 4, 128, true, Window::Global),
    // Registry24: EmbeddingGemma2 d256 noncausal local GQA2 at +/-512.
    Profile::new(
        "embedding_gemma2_local256",
        4,
        2,
        256,
        false,
        Window::Local {
            left: 512,
            right: 512,
        },
    ),
    // Canonical26: Gemma4 E2B d256 local Q8/KV1. The published profile and
    // manifest carry the exact canonical extents, not an internal compile hint:
    // causal [511, 0] and noncausal [256, 256]. FA4 passes these left/right
    // values as runtime scalars, so exact preference and provenance agree.
    // Scale is explicit 1.0 at the model call site; provider scale remains a
    // runtime argument.
    Profile::new(
        "gemma4_local256_causal",
        8,
        1,
        256,
        true,
        Window::Local {
            left: 511,
            right: 0,
        },
    ),
    Profile::new(
        "gemma4_local256_noncausal",
        8,
        1,
        256,
        false,
        Window::Local {
            left: 256,
            right: 256,
        },
    ),
];
const _: () = assert!(LEGACY_PROFILE_COUNT < PUBLISHED_PROFILE_COUNT);
const _: () = assert!(PUBLISHED_PROFILE_COUNT < PRIVATE23_PROFILE_COUNT);
const _: () = assert!(PRIVATE23_PROFILE_COUNT < REGISTRY24_PROFILE_COUNT);
const _: () = assert!(REGISTRY24_PROFILE_COUNT < CANONICAL26_PROFILE_COUNT);
const _: () = assert!(PROFILES.len() == CANONICAL26_PROFILE_COUNT);

// Exact manifest registry identity for every legal declared ordinary prefix.
// The fingerprint is the SHA-256 of the canonical profile sequence
// "mode:name:heads:kv_heads:head_dim:causal:local" joined by '|'. A manifest
// that chooses to publish registry metadata must match both the namespace and
// this prefix-specific fingerprint; geometry itself is already checked by the
// strict ordered `kernel_matches` scan above.
const REGISTRY_PREFIX_IDENTITIES: &[(usize, &str, &str)] = &[
    (
        19,
        "legacy19",
        "064e20f61405380889e1d88511d3f52d01caf654a2df983aa407f4bc5e737ea3",
    ),
    (
        20,
        "published20",
        "827958cc0e82a0bccf0cf2ac96d25492a9fb40e7e36c54843618eaaf26807ebd",
    ),
    (
        21,
        "private23",
        "b7d9e06b0fa290812b4e0018d8ff919af29762a47dd68ea525799a86b81ec0fa",
    ),
    (
        22,
        "private23",
        "7ca9a15773f4aae33b55bb17619afa65edb9bf2c14b25d7e89338c7d8907e490",
    ),
    (
        23,
        "private23",
        "268db2182d75fcda41f7d6e52b35dc967c1e9652ac5ad86e543ceed922fe7f80",
    ),
    (
        24,
        "registry24",
        "99c39dc70c2cd22532df08d4edbb6a3f62cfd10a910c1da66321cbb434a233b6",
    ),
    (
        25,
        "canonical26",
        "8443cd2fefe5c627da27c305442f5aff1a1d7cdfb1ffc1cbbd19149f6cde4245",
    ),
    (
        26,
        "canonical26",
        "3b214841c169c3bcaad0574ee9df917c9fc8a05be1e64602cf5b91d99a67ff1e",
    ),
];

fn start(e: impl std::fmt::Display) -> BackendError {
    BackendError::Start(format!("Oxide FA4: {e}"))
}
fn infer(e: impl std::fmt::Display) -> BackendError {
    BackendError::Inference(format!("Oxide FA4: {e}"))
}

fn profile_index(spec: &AttentionSpec) -> Option<usize> {
    select_profile_index(&PROFILES, spec)
}

/// Select deterministically among equivalent local kernels: prefer the profile
/// whose exported extents match the request exactly, then the first declared
/// profile. This keeps original profile order authoritative while allowing an
/// older local export to serve a new runtime extent.
fn select_profile_index(profiles: &[Profile], spec: &AttentionSpec) -> Option<usize> {
    let mut selected: Option<(usize, bool)> = None;
    for (mode, profile) in profiles.iter().enumerate() {
        if !profile.matches(spec) {
            continue;
        }
        let exact = profile.window == spec.window;
        if let Some((_, best_exact)) = selected {
            if best_exact || !exact {
                continue;
            }
        }
        selected = Some((mode, exact));
    }
    selected.map(|(mode, _)| mode)
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
    crate::fa4_deberta::validate_exact_mul_exports(manifest)?;
    let deberta_symbol = format!("fa4_deberta_{}", dtype.name());
    let mut deberta_seen = false;
    let mut alibi_eight_seen = false;
    let mut alibi_native_twelve_seen = false;
    let mut seen = 0usize;
    for kernel in kernels {
        if kernel.get("dtype").and_then(serde_json::Value::as_str) != Some(dtype.name()) {
            continue;
        }
        if let Some(symbol) = kernel.get("symbol").and_then(serde_json::Value::as_str) {
            if crate::fa4_deberta::is_exact_mul_symbol(symbol) {
                continue;
            }
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
        if let Some(profile) = kernel
            .get("symbol")
            .and_then(serde_json::Value::as_str)
            .and_then(|symbol| crate::fa4_alibi::profile_for_symbol(symbol, dtype.name()))
        {
            let seen = match profile {
                crate::fa4_alibi::AlibiProfile::Eight => &mut alibi_eight_seen,
                crate::fa4_alibi::AlibiProfile::TwelveNativeFlash => &mut alibi_native_twelve_seen,
            };
            if *seen || !crate::fa4_alibi::kernel_matches(kernel, dtype.name(), profile) {
                return Err(start("invalid or duplicate ALiBi model-bias export"));
            }
            *seen = true;
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
    validate_registry_metadata(manifest, seen)?;
    Ok(seen)
}

/// Optional canonical registry metadata. Old bundles may omit it; when present
/// it must match the exact namespace and prefix-specific sequence fingerprint
/// for the declared ordinary prefix. Geometry was already verified by the
/// strict ordered `kernel_matches` scan, so this closes the remaining namespace
/// collision without pretending to hash at runtime.
fn validate_registry_metadata(
    manifest: &serde_json::Value,
    declared: usize,
) -> Result<(), BackendError> {
    let Some(registry) = manifest.get("registry") else {
        return Ok(());
    };
    let Some(&(_, expected_namespace, expected_fingerprint)) = REGISTRY_PREFIX_IDENTITIES
        .iter()
        .find(|(count, _, _)| *count == declared)
    else {
        return Err(start("registry metadata for an unknown prefix length"));
    };
    let namespace = registry
        .get("namespace")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| start("registry metadata lacks a namespace"))?;
    if namespace != expected_namespace {
        return Err(start(format!(
            "registry namespace {namespace} != {expected_namespace} for declared prefix {declared}"
        )));
    }
    let mode_count = registry
        .get("mode_count")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| start("registry metadata lacks mode_count"))?;
    if usize::try_from(mode_count).ok() != Some(declared) {
        return Err(start("registry mode_count differs from declared exports"));
    }
    if let Some(mask) = registry
        .get("mode_mask")
        .and_then(serde_json::Value::as_str)
    {
        let expected = mode_mask(declared)?;
        if mask != format!("{expected:#x}") {
            return Err(start("registry mode_mask differs from declared exports"));
        }
    }
    let fingerprint = registry
        .get("profile_sequence_sha256")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| start("registry metadata lacks profile_sequence_sha256"))?;
    if fingerprint != expected_fingerprint {
        return Err(start(
            "registry profile_sequence_sha256 differs from the canonical prefix",
        ));
    }
    Ok(())
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

fn missing_profile(dtype: DType, declared: usize, spec: &AttentionSpec) -> BackendError {
    start(format!(
        "missing-profile: no compatible {} export in declared prefix {declared} for Q{}xKV{} d{} causal={} window={:?}",
        dtype.name(),
        spec.heads,
        spec.kv_heads,
        spec.head_dim,
        spec.causal,
        spec.window
    ))
}

fn profile_selection(
    manifest: &serde_json::Value,
    spec: &AttentionSpec,
) -> Result<(usize, usize), BackendError> {
    // The manifest's declared ordinary prefix is authoritative. Selecting from
    // the full registry first would let an appended exact profile hide an older
    // compatible export in a truncated bundle.
    let declared = validate_profile_order(manifest, spec.dtype)?;
    let mode = select_profile_index(&PROFILES[..declared], spec)
        .ok_or_else(|| missing_profile(spec.dtype, declared, spec))?;
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

#[cfg(test)]
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
    local: bool,
    q_tokens: usize,
    k_tokens: usize,
    q_offsets: u64,
    k_offsets: u64,
) -> bool {
    // The dedicated SM100 d256 two-CTA global kernel requires shared Q/K
    // boundaries. Local d256 is compiled through the generic one-CTA path,
    // which carries independent offset_q/offset_k.
    capability / 10 == 10
        && head_dim == 256
        && !local
        && (q_tokens != k_tokens || q_offsets != k_offsets)
}

fn valid_stride(stride: &[i64], width: i64, head_dim: i64) -> bool {
    stride.len() == 3
        && stride[2] == 1
        && stride[1] == head_dim
        && stride[0] >= width
        && stride[0] % 8 == 0
}

/// Keep the runtime mask arithmetic inside signed int32. The longest per-batch
/// Q length and the packed K extent bound `causal_row_offset`; a local window
/// larger than the remaining int32 headroom could overflow the native mask.
fn validate_local_window_bounds(
    spec: &AttentionSpec,
    batch: &AttentionBatch,
) -> Result<(), BackendError> {
    let Window::Local { .. } = spec.window else {
        return Ok(());
    };
    let Some((left, right)) = spec.window.canonical_local_extents(spec.causal) else {
        return Err(infer("invalid local attention window"));
    };
    let max_sequence = batch.max_q.max(batch.k_tokens);
    let max_sequence = i64::try_from(max_sequence).map_err(infer)?;
    let limit = i64::from(i32::MAX) - max_sequence;
    if i64::from(left) > limit || i64::from(right) > limit {
        return Err(infer("local attention window exceeds safe int32 bounds"));
    }
    Ok(())
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
    validate_local_window_bounds(spec, batch)?;
    if separate_qk_forbidden(
        capability,
        spec.head_dim,
        matches!(spec.window, Window::Local { .. }),
        batch.q_tokens,
        batch.k_tokens,
        batch.q_offsets,
        batch.k_offsets,
    ) {
        return Err(infer(
            "SM100 d256 global attention requires shared Q/K boundaries",
        ));
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
    alibi: Option<crate::fa4_alibi::AlibiLaunch>,
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
    #[cfg(test)]
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

    /// Global mode selected by the published FA4 ABI: absolute Q/KV counts are
    /// runtime DLTensor shapes, while the compiled specialization is the exact
    /// Q/KV ratio plus head_dim, dtype, causal bit and global mask. Runtime
    /// q/kv shape and stride validation remains in the launch path unchanged.
    pub(crate) fn ratio_global_mode_for(spec: &AttentionSpec) -> Option<usize> {
        select_profile_index(
            &PROFILES,
            &AttentionSpec {
                window: Window::Global,
                ..*spec
            },
        )
    }

    /// Conservative guard for published auto routing. Appended canonical26
    /// profiles remain available to explicit constructors and to the
    /// manifest-driven loader, but old auto paths keep the published prefix.
    pub(crate) fn supports_published(spec: &AttentionSpec) -> bool {
        Self::supports(spec)
            && select_profile_index(&PROFILES[..PUBLISHED_PROFILE_COUNT], spec).is_some()
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
        #[cfg(test)]
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
            alibi: None,
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
            #[cfg(test)]
            contiguous_strides,
            _library: library,
        })
    }

    /// Construct the fixed native Jina-small ALiBi recipe. The ordinary ABI
    /// and its mode mask stay unchanged; the additive export owns F32 slopes.
    pub(crate) fn new_alibi(
        context: Arc<Context>,
        stream: Arc<Stream>,
        root: &Path,
        spec: AttentionSpec,
    ) -> Result<Self, BackendError> {
        let profile = crate::fa4_alibi::AlibiProfile::for_heads(spec.heads)
            .ok_or_else(|| start("Legacy ALiBi model-bias requires exactly8 attention heads"))?;
        Self::new_alibi_profile(context, stream, root, spec, profile)
    }

    pub(crate) fn new_alibi_native12(
        context: Arc<Context>,
        stream: Arc<Stream>,
        root: &Path,
        spec: AttentionSpec,
    ) -> Result<Self, BackendError> {
        Self::new_alibi_profile(
            context,
            stream,
            root,
            spec,
            crate::fa4_alibi::AlibiProfile::TwelveNativeFlash,
        )
    }

    fn new_alibi_profile(
        context: Arc<Context>,
        stream: Arc<Stream>,
        root: &Path,
        spec: AttentionSpec,
        profile: crate::fa4_alibi::AlibiProfile,
    ) -> Result<Self, BackendError> {
        if profile.heads() != spec.heads {
            return Err(start("ALiBi recipe/head geometry differs"));
        }
        if spec.kv_heads != spec.heads
            || spec.head_dim != 64
            || spec.causal
            || spec.window != Window::Global
            || resolve_scale(&spec)? != 0.125
        {
            return Err(start(
                "ALiBi model-bias requires noncausal 8/12x64 self-attention at scale0.125",
            ));
        }
        let mut result = Self::new(context.clone(), root, spec)?;
        let path = root.join(format!("sm_{}", result.capability));
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path.join("manifest.json")).map_err(start)?)
                .map_err(start)?;
        let kernels = manifest
            .get("kernels")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| start("manifest lacks kernels"))?;
        if kernels
            .iter()
            .filter(|k| crate::fa4_alibi::kernel_matches(k, spec.dtype.name(), profile))
            .count()
            != 1
        {
            return Err(start(
                "manifest lacks exactly one selected ALiBi model-bias export",
            ));
        }
        result.alibi = Some(crate::fa4_alibi::AlibiLaunch::new(
            context,
            stream,
            &path.join("libinfinityfa4.so"),
            profile,
        )?);
        Ok(result)
    }

    /// Contiguous packed self-attention using the selected spec. The engine
    /// retains Q/K/V/output allocations and the int32 boundaries through stream
    /// completion. This preserves the existing BERT call shape.
    #[cfg(test)]
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
    #[allow(dead_code)] // Safe binding entry retained for tests and non-engine callers.
    pub(crate) unsafe fn forward_batch(
        &self,
        stream: &Stream,
        batch: &AttentionBatch,
    ) -> Result<(), BackendError> {
        let args = self.validate_forward(stream, batch)?;
        self.context.bind_to_thread().map_err(infer)?;
        // SAFETY: validation passed and the context is current on this thread;
        // no context switch occurs below.
        unsafe { self.launch_forward(stream, batch, args) }
    }

    /// Enqueue one generalized packed attention launch on an already-bound
    /// context.
    ///
    /// # Safety
    ///
    /// The provider's context MUST be current on the calling thread before
    /// entry and MUST remain current for the entire call. The caller must keep
    /// all Q/K/V/output and cumulative-length device allocations alive and
    /// unmodified until this stream work completes, must provide `batch + 1`
    /// int32 entries at each offsets pointer, and must ensure output storage
    /// does not alias an input. All pointers must remain valid for the launch.
    pub(crate) unsafe fn forward_batch_bound(
        &self,
        stream: &Stream,
        batch: &AttentionBatch,
    ) -> Result<(), BackendError> {
        let args = self.validate_forward(stream, batch)?;
        // SAFETY: the caller guarantees the context remains current; validation
        // above upholds the remaining launch contract.
        unsafe { self.launch_forward(stream, batch, args) }
    }

    fn validate_forward(
        &self,
        stream: &Stream,
        batch: &AttentionBatch,
    ) -> Result<(i64, i64, i32, i32), BackendError> {
        if self.context.cu_ctx() != stream.context().cu_ctx() {
            return Err(infer("stream belongs to another CUDA context"));
        }
        check_batch(&self.spec, self.capability, batch)?;
        if let Some(alibi) = &self.alibi {
            alibi.check_stream(stream)?;
            if batch.q_offsets != batch.k_offsets
                || batch.q_tokens != batch.k_tokens
                || batch.q_tokens > alibi.token_limit()
            {
                return Err(infer(
                    "ALiBi requires shared packed self-attention boundaries within int32 geometry",
                ));
            }
        }
        let total = i64::from(i32::try_from(batch.q_tokens).map_err(infer)?);
        let total_k = i64::from(i32::try_from(batch.k_tokens).map_err(infer)?);
        let count = i32::try_from(batch.batch).map_err(infer)?;
        count
            .checked_add(1)
            .ok_or_else(|| infer("boundary count overflow"))?;
        let max_q = i32::try_from(batch.max_q).map_err(infer)?;
        Ok((total, total_k, count, max_q))
    }

    unsafe fn launch_forward(
        &self,
        stream: &Stream,
        batch: &AttentionBatch,
        (total, total_k, count, max_q): (i64, i64, i32, i32),
    ) -> Result<(), BackendError> {
        if let Some(alibi) = &self.alibi {
            // SAFETY: common validation plus fixed ALiBi self-attention contract.
            let status = unsafe {
                alibi.launch(
                    self.dtype,
                    self.device,
                    stream,
                    batch.q,
                    batch.k,
                    batch.v,
                    batch.out,
                    batch.q_offsets,
                    total,
                    i64::from(count),
                    &batch.strides,
                    max_q,
                )
            };
            if status != 0 {
                return Err(provider_error(self.error, status));
            }
            return Ok(());
        }
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

    fn exact_mul_deberta_kernel(symbol: &str) -> Value {
        crate::fa4_deberta::test_exact_mul_kernel(symbol)
    }

    fn push_exact_mul_deberta_pair(manifest: &mut Value) {
        let kernels = manifest["kernels"]
            .as_array_mut()
            .expect("test manifest has kernels");
        kernels.push(exact_mul_deberta_kernel(
            crate::fa4_deberta::EXACT_MUL_BOTH_SYMBOL,
        ));
        kernels.push(exact_mul_deberta_kernel(
            crate::fa4_deberta::EXACT_MUL_SINGLE_SYMBOL,
        ));
    }

    fn exact_mul_deberta_position(manifest: &Value, symbol: &str) -> usize {
        manifest["kernels"]
            .as_array()
            .expect("test manifest has kernels")
            .iter()
            .position(|kernel| kernel.get("symbol").and_then(Value::as_str) == Some(symbol))
            .expect("exact-mul DeBERTa export is present")
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
    fn published_auto_guard_keeps_old_prefix_and_falls_back_for_appended_geometry() {
        for dtype in [DType::F16, DType::Bf16] {
            for profile in PROFILES.iter().take(PUBLISHED_PROFILE_COUNT) {
                let spec = spec_for(profile, dtype);
                assert!(Fa4::supports(&spec));
                assert!(Fa4::supports_published(&spec));
            }
            // Mistral's exact 4095 window remains serviceable by the older
            // compatible local prefix, matching the previous auto behavior.
            let mistral = spec_for(&PROFILES[20], dtype);
            assert!(Fa4::supports(&mistral));
            assert!(Fa4::supports_published(&mistral));
            assert_eq!(
                select_profile_index(&PROFILES[..PUBLISHED_PROFILE_COUNT], &mistral),
                Some(15)
            );
            assert_eq!(select_profile_index(&PROFILES, &mistral), Some(20));
            // Appended geometries with no published-prefix match must stay on
            // the old auto fallback path.
            for mode in [21, 22, 23, 24, 25] {
                let spec = spec_for(&PROFILES[mode], dtype);
                assert!(Fa4::supports(&spec));
                assert!(!Fa4::supports_published(&spec));
            }
        }
    }

    #[test]
    fn accepts_runtime_local_extents_of_the_existing_mask_kind() {
        let manifest = manifest_for(&[DType::F16, DType::Bf16]);
        for dtype in [DType::F16, DType::Bf16] {
            // UltraBERT: 6/6 d128 noncausal local, radius 128.
            let mut ultrabert = spec_for(&PROFILES[10], dtype);
            ultrabert.heads = 6;
            ultrabert.kv_heads = 6;
            ultrabert.window = Window::Local {
                left: 128,
                right: 128,
            };
            assert_eq!(select_profile(&manifest, &ultrabert).ok(), Some(10));

            // Absolute heads remain runtime aliases of the ratio-1 local kernel.
            let mut ultrabert_alias = ultrabert;
            ultrabert_alias.heads = 12;
            ultrabert_alias.kv_heads = 12;
            assert_eq!(select_profile(&manifest, &ultrabert_alias).ok(), Some(10));

            // Mistral: 32/8 d128 causal local, radius 4095. The full
            // canonical26 prefix contains the exact mode20; the published20
            // prefix can only offer the compatible mode15 fallback.
            let mut mistral = spec_for(&PROFILES[15], dtype);
            mistral.heads = 32;
            mistral.kv_heads = 8;
            mistral.window = Window::Local {
                left: 4095,
                right: 0,
            };
            assert_eq!(select_profile(&manifest, &mistral).ok(), Some(20));
            let published = manifest_prefix(&[dtype], PUBLISHED_PROFILE_COUNT);
            assert_eq!(select_profile(&published, &mistral).ok(), Some(15));

            let mut mistral_alias = mistral;
            mistral_alias.heads = 16;
            mistral_alias.kv_heads = 4;
            assert_eq!(select_profile(&manifest, &mistral_alias).ok(), Some(20));
            assert_eq!(select_profile(&published, &mistral_alias).ok(), Some(15));

            // Asymmetric noncausal windows stay local and noncausal.
            let mut asymmetric = spec_for(&PROFILES[10], dtype);
            asymmetric.window = Window::Local {
                left: 0,
                right: 128,
            };
            assert_eq!(select_profile(&manifest, &asymmetric).ok(), Some(10));
        }
    }

    #[test]
    fn rejects_mismatched_mask_kinds_and_unsafe_local_extents() {
        let manifest = manifest_for(&[DType::F16, DType::Bf16]);
        let reject = |spec: AttentionSpec, label: &str| {
            assert!(
                select_profile(&manifest, &spec).is_err(),
                "accepted {label}"
            );
        };

        // A global request keeps the global export even when the same geometry
        // has a local export: local128 must not absorb mha128's global mask.
        let mut global_for_local = spec_for(&PROFILES[10], DType::F16);
        global_for_local.window = Window::Global;
        assert_eq!(select_profile(&manifest, &global_for_local).ok(), Some(9));

        let mut global_for_causal_local = spec_for(&PROFILES[15], DType::F16);
        global_for_causal_local.window = Window::Global;
        assert_eq!(
            select_profile(&manifest, &global_for_causal_local).ok(),
            Some(2)
        );

        // The local request selects the local kernel, not the global mha128
        // export with the same ratio and width.
        let mut local_for_global = spec_for(&PROFILES[9], DType::F16);
        local_for_global.window = Window::Local {
            left: 128,
            right: 128,
        };
        assert_eq!(select_profile(&manifest, &local_for_global).ok(), Some(10));

        let mut unsupported_local = spec_for(&PROFILES[7], DType::F16);
        unsupported_local.window = Window::Local {
            left: 128,
            right: 128,
        };
        reject(unsupported_local, "local request without a local export");

        let mut causal_for_noncausal = spec_for(&PROFILES[10], DType::F16);
        causal_for_noncausal.causal = true;
        causal_for_noncausal.window = Window::Local {
            left: 128,
            right: 0,
        };
        reject(
            causal_for_noncausal,
            "causal request for a noncausal export",
        );

        let mut noncausal_for_causal = spec_for(&PROFILES[15], DType::F16);
        noncausal_for_causal.causal = false;
        noncausal_for_causal.window = Window::Local {
            left: 4095,
            right: 0,
        };
        reject(
            noncausal_for_causal,
            "noncausal request for a causal export",
        );

        let mut causal_right_nonzero = spec_for(&PROFILES[15], DType::F16);
        causal_right_nonzero.window = Window::Local {
            left: 4095,
            right: 1,
        };
        reject(causal_right_nonzero, "causal local right != 0");

        let mut noncausal_right_zero = spec_for(&PROFILES[10], DType::F16);
        noncausal_right_zero.window = Window::Local {
            left: 128,
            right: 0,
        };
        reject(noncausal_right_zero, "noncausal local right == 0");

        for (left, right, label) in [
            (-1, 128, "negative left"),
            (128, -1, "negative right"),
            (0, 0, "noncausal zero extent"),
            (i32::MAX, i32::MAX, "overflowing extent"),
        ] {
            let mut spec = spec_for(&PROFILES[10], DType::F16);
            spec.window = Window::Local { left, right };
            reject(spec, label);
        }

        let mut causal_overflow = spec_for(&PROFILES[15], DType::F16);
        causal_overflow.window = Window::Local {
            left: i32::MAX,
            right: 0,
        };
        reject(causal_overflow, "causal overflow");
    }

    #[test]
    fn rejects_unsafe_local_window_bounds_before_enqueue() {
        let mut spec = spec_for(&PROFILES[10], DType::F16);
        spec.window = Window::Local {
            left: 128,
            right: 128,
        };
        let safe = fake_batch(&spec, 256, 256);
        assert!(check_batch(&spec, 90, &safe).is_ok());

        let huge = fake_batch(&spec, i32::MAX as usize, i32::MAX as usize);
        assert!(check_batch(&spec, 90, &huge).is_err());

        let mut negative = spec;
        negative.window = Window::Local {
            left: -1,
            right: 128,
        };
        let batch = fake_batch(&negative, 256, 256);
        assert!(check_batch(&negative, 90, &batch).is_err());
    }

    #[test]
    fn local_manifest_identity_remains_exact_while_requests_broaden() {
        let manifest = manifest_for(&[DType::F16, DType::Bf16]);
        let mut ultrabert = spec_for(&PROFILES[10], DType::F16);
        ultrabert.window = Window::Local {
            left: 128,
            right: 128,
        };
        assert_eq!(select_profile(&manifest, &ultrabert).ok(), Some(10));
        assert_eq!(
            select_profile(&manifest, &spec_for(&PROFILES[10], DType::F16)).ok(),
            Some(10)
        );

        for (field, value) in [
            ("window", json!([128, 128])),
            ("causal", json!(true)),
            ("heads", json!(12)),
            ("kv_heads", json!(12)),
        ] {
            let mut corrupt = manifest_for(&[DType::F16, DType::Bf16]);
            corrupt["kernels"]
                .as_array_mut()
                .expect("test manifest has kernels")[10][field] = value;
            assert!(
                select_profile(&corrupt, &ultrabert).is_err(),
                "accepted corrupt local {field}"
            );
            assert!(
                select_profile(&corrupt, &spec_for(&PROFILES[10], DType::F16)).is_err(),
                "canonical request accepted corrupt local {field}"
            );
        }
    }

    #[test]
    fn chooses_exact_then_first_declared_duplicate_deterministically() {
        let canonical = Profile::new(
            "canonical",
            6,
            6,
            128,
            false,
            Window::Local {
                left: 64,
                right: 64,
            },
        );
        let wider = Profile::new(
            "wider",
            6,
            6,
            128,
            false,
            Window::Local {
                left: 128,
                right: 128,
            },
        );
        let profiles = [canonical, wider];
        let mut spec = spec_for(&canonical, DType::F16);
        spec.window = Window::Local {
            left: 128,
            right: 128,
        };
        assert_eq!(select_profile_index(&profiles, &spec), Some(1));

        spec.window = Window::Local {
            left: 64,
            right: 64,
        };
        assert_eq!(select_profile_index(&profiles, &spec), Some(0));

        spec.window = Window::Local {
            left: 96,
            right: 96,
        };
        assert_eq!(select_profile_index(&profiles, &spec), Some(0));

        let duplicate = [canonical, canonical];
        spec.window = Window::Local {
            left: 64,
            right: 64,
        };
        assert_eq!(select_profile_index(&duplicate, &spec), Some(0));
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
            mode_mask(PROFILES.len()).expect("appended mode mask")
                | (1 << CANONICAL26_PROFILE_COUNT)
        )
        .is_err());
    }

    #[test]
    fn canonical26_modes_are_appended_in_order_and_mode_mask_is_exact() {
        assert_eq!(PROFILES.len(), 26);
        assert_eq!(PUBLISHED_PROFILE_COUNT, 20);
        assert_eq!(PRIVATE23_PROFILE_COUNT, 23);
        assert_eq!(REGISTRY24_PROFILE_COUNT, 24);
        assert_eq!(CANONICAL26_PROFILE_COUNT, 26);
        assert_eq!(PROFILES[20].name, "mistral_local4096");
        assert_eq!(PROFILES[21].name, "gemma3_local512");
        assert_eq!(PROFILES[22].name, "causal_gqa8");
        assert_eq!(PROFILES[23].name, "embedding_gemma2_local256");
        assert_eq!(PROFILES[24].name, "gemma4_local256_causal");
        assert_eq!(PROFILES[25].name, "gemma4_local256_noncausal");
        assert_eq!(
            PROFILES[23].window,
            Window::Local {
                left: 512,
                right: 512
            }
        );
        assert_eq!(
            PROFILES[24].window,
            Window::Local {
                left: 511,
                right: 0
            }
        );
        assert_eq!(
            PROFILES[25].window,
            Window::Local {
                left: 256,
                right: 256
            }
        );

        let mask = mode_mask(CANONICAL26_PROFILE_COUNT).expect("canonical26 mask");
        assert_eq!(mask, 0x03ff_ffff);
        assert_eq!(mask.count_ones(), 26);
        assert!(validate_mode_mask(PROFILES.len(), mask).is_ok());
        assert!(validate_mode_mask(
            PROFILES.len(),
            mode_mask(REGISTRY24_PROFILE_COUNT).expect("registry24 mask")
        )
        .is_err());
    }

    #[test]
    fn canonical26_selection_is_loaded_prefix_aware_and_typed() {
        let full = manifest_for(&[DType::F16, DType::Bf16]);
        for dtype in [DType::F16, DType::Bf16] {
            let mistral = spec_for(&PROFILES[20], dtype);
            let gemma3 = spec_for(&PROFILES[21], dtype);
            let eg2 = spec_for(&PROFILES[23], dtype);
            let gemma4_causal = spec_for(&PROFILES[24], dtype);
            let gemma4_noncausal = spec_for(&PROFILES[25], dtype);

            // Published20 cannot see any appended d256 profile. Mistral's exact
            // extents fall back to the declared causal-local d128 kernel.
            let p20 = manifest_prefix(&[dtype], PUBLISHED_PROFILE_COUNT);
            assert_eq!(select_profile(&p20, &mistral).ok(), Some(15));
            assert!(select_profile(&p20, &gemma3).is_err());
            assert!(select_profile(&p20, &eg2).is_err());
            assert!(select_profile(&p20, &gemma4_causal).is_err());
            assert!(select_profile(&p20, &gemma4_noncausal).is_err());
            assert!(
                format!("{}", select_profile(&p20, &gemma4_causal).unwrap_err())
                    .contains("missing-profile")
            );

            // Private23 sees Mistral/Gemma3/causal_gqa8 but not d256 Q8/KV1
            // local or EG2, and must reject instead of approximating.
            let p23 = manifest_prefix(&[dtype], PRIVATE23_PROFILE_COUNT);
            assert_eq!(select_profile(&p23, &mistral).ok(), Some(20));
            assert_eq!(select_profile(&p23, &gemma3).ok(), Some(21));
            assert!(select_profile(&p23, &eg2).is_err());
            assert!(select_profile(&p23, &gemma4_causal).is_err());
            assert!(select_profile(&p23, &gemma4_noncausal).is_err());

            // Registry24 adds EG2 (noncausal d256 GQA2 local) but still has no
            // Gemma4 Q8/KV1 d256 local profile.
            let p24 = manifest_prefix(&[dtype], REGISTRY24_PROFILE_COUNT);
            assert_eq!(select_profile(&p24, &eg2).ok(), Some(23));
            assert!(select_profile(&p24, &gemma4_causal).is_err());
            assert!(select_profile(&p24, &gemma4_noncausal).is_err());

            // Prefix25 exposes the causal Gemma4 mode only.
            let p25 = manifest_prefix(&[dtype], CANONICAL26_PROFILE_COUNT - 1);
            assert_eq!(select_profile(&p25, &gemma4_causal).ok(), Some(24));
            assert!(select_profile(&p25, &gemma4_noncausal).is_err());

            // Full canonical26 exposes both exact modes.
            assert_eq!(select_profile(&full, &gemma4_causal).ok(), Some(24));
            assert_eq!(select_profile(&full, &gemma4_noncausal).ok(), Some(25));
        }
    }

    #[test]
    fn canonical26_gemma4_ratio_is_not_broadened_and_global_d256_is_exact() {
        let manifest = manifest_for(&[DType::F16, DType::Bf16]);
        for dtype in [DType::F16, DType::Bf16] {
            let profile = PROFILES[24];
            // Absolute heads are runtime for the same exported ratio 8.
            for (heads, kv_heads) in [(8, 1), (16, 2), (32, 4), (64, 8)] {
                let mut spec = spec_for(&profile, dtype);
                spec.heads = heads;
                spec.kv_heads = kv_heads;
                assert_eq!(select_profile(&manifest, &spec).ok(), Some(24));
            }
            // A different ratio must not alias mode24. Ratio 4 d256 causal
            // local is the separate Gemma3 mode21, not a broadened Gemma4 hit.
            for (heads, kv_heads) in [(12, 2), (6, 1), (24, 4)] {
                let mut spec = spec_for(&profile, dtype);
                spec.heads = heads;
                spec.kv_heads = kv_heads;
                assert!(select_profile(&manifest, &spec).is_err());
            }
            let mut ratio4 = spec_for(&profile, dtype);
            ratio4.heads = 16;
            ratio4.kv_heads = 4;
            assert_eq!(select_profile(&manifest, &ratio4).ok(), Some(21));
            // Global d256 masks stay exact: ratio-4 noncausal is mode6, while a
            // ratio-8 global d256 request has no compiled profile.
            let mut global_ratio4 = spec_for(&profile, dtype);
            global_ratio4.heads = 16;
            global_ratio4.kv_heads = 4;
            global_ratio4.causal = false;
            global_ratio4.window = Window::Global;
            assert_eq!(select_profile(&manifest, &global_ratio4).ok(), Some(6));
            let mut global_ratio8 = global_ratio4;
            global_ratio8.heads = 16;
            global_ratio8.kv_heads = 2;
            assert!(select_profile(&manifest, &global_ratio8).is_err());
        }
    }

    #[test]
    fn canonical26_complete_bundle_with_exact_mul_validates_26_modes() {
        let mut manifest = manifest_for(&[DType::F16, DType::Bf16]);
        for dtype in [DType::F16, DType::Bf16] {
            manifest["kernels"]
                .as_array_mut()
                .expect("test manifest has kernels")
                .push(deberta_kernel(dtype));
        }
        push_exact_mul_deberta_pair(&mut manifest);
        for dtype in [DType::F16, DType::Bf16] {
            assert_eq!(
                validate_profile_order(&manifest, dtype).ok(),
                Some(CANONICAL26_PROFILE_COUNT)
            );
            assert_eq!(
                select_profile(&manifest, &spec_for(&PROFILES[25], dtype)).ok(),
                Some(25)
            );
        }
    }

    #[test]
    fn canonical26_registry_metadata_is_validated_when_present() {
        let baseline = manifest_for(&[DType::F16, DType::Bf16]);
        let mut manifest = baseline.clone();
        manifest["registry"] = json!({
            "namespace": "canonical26",
            "mode_count": 26,
            "mode_mask": "0x3ffffff",
            "profile_sequence_sha256": "3b214841c169c3bcaad0574ee9df917c9fc8a05be1e64602cf5b91d99a67ff1e",
        });
        for dtype in [DType::F16, DType::Bf16] {
            assert_eq!(validate_profile_order(&manifest, dtype).ok(), Some(26));
        }
        for (field, value) in [
            ("mode_count", json!(25)),
            ("mode_mask", json!("0x1ffffff")),
            ("namespace", json!("standalone")),
            ("namespace", json!("published20")),
            ("profile_sequence_sha256", json!("not-a-hash")),
            // A well-formed 64-hex fingerprint from a different prefix.
            (
                "profile_sequence_sha256",
                json!("827958cc0e82a0bccf0cf2ac96d25492a9fb40e7e36c54843618eaaf26807ebd"),
            ),
        ] {
            let mut wrong = manifest.clone();
            wrong["registry"][field] = value;
            assert!(
                validate_profile_order(&wrong, DType::F16).is_err(),
                "{field}"
            );
        }

        // Prefix-specific metadata is accepted for the matching prefix.
        let mut prefix24 = manifest_prefix(&[DType::F16, DType::Bf16], REGISTRY24_PROFILE_COUNT);
        prefix24["registry"] = json!({
            "namespace": "registry24",
            "mode_count": 24,
            "mode_mask": "0xffffff",
            "profile_sequence_sha256": "99c39dc70c2cd22532df08d4edbb6a3f62cfd10a910c1da66321cbb434a233b6",
        });
        assert_eq!(
            validate_profile_order(&prefix24, DType::F16).ok(),
            Some(REGISTRY24_PROFILE_COUNT)
        );
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
            // d128 ratio8 causal global is now the canonical mode22 append.
            let mut wider_dim = spec_for(&profile, dtype);
            wider_dim.head_dim = 128;
            assert_eq!(select_profile(&manifest, &wider_dim).ok(), Some(22));
            let mut wrong_dim = spec_for(&profile, dtype);
            wrong_dim.head_dim = 32;
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

        // SM100 d256 local uses the generic one-CTA kernel and carries separate
        // Q/K offsets; only the dedicated global two-CTA path is restricted.
        let local = spec_for(&PROFILES[24], DType::Bf16);
        let separate_local = fake_batch(&local, 8, 16);
        assert!(check_batch(&local, 100, &separate_local).is_ok());
        let mut split_local = separate_local;
        split_local.k_offsets = 0x6000;
        assert!(check_batch(&local, 100, &split_local).is_ok());
        let global = spec_for(&PROFILES[6], DType::Bf16);
        assert!(check_batch(&global, 100, &fake_batch(&global, 8, 16)).is_err());
    }

    #[test]
    fn alibi_exports_are_additive_and_keep_ordinary_modes() {
        for dtype in [DType::F16, DType::Bf16] {
            let mut manifest = manifest_for(&[dtype]);
            manifest["kernels"].as_array_mut().unwrap().push(json!({
                "symbol": format!("fa4_alibi_model_bias_{}", dtype.name()),
                "dtype": dtype.name(), "kind":"alibi_model_bias", "score_source":"alibi_score.py", "score_abi":1,"qk_scale":0.125,"external_softmax_scale":1.0,"bias_rounding":"model_dtype",
                "heads":8,"kv_heads":8,"d":64,"causal":false,"window":[null,null]
            }));
            assert_eq!(
                select_profile(&manifest, &spec_for(&PROFILES[0], dtype)).unwrap(),
                0
            );
            assert_eq!(
                validate_profile_order(&manifest, dtype).unwrap(),
                PROFILES.len()
            );
            let extra = manifest["kernels"]
                .as_array()
                .unwrap()
                .last()
                .unwrap()
                .clone();
            manifest["kernels"].as_array_mut().unwrap().push(extra);
            assert!(validate_profile_order(&manifest, dtype).is_err());
        }
    }

    #[test]
    fn malformed_alibi_manifest_cannot_change_the_score_recipe() {
        let dtype = DType::Bf16;
        let mut manifest = manifest_for(&[dtype]);
        let alibi = json!({"symbol":"fa4_alibi_model_bias_bf16", "dtype":"bf16", "kind":"alibi_model_bias", "score_source":"alibi_score.py", "score_abi":1,"qk_scale":0.125,"external_softmax_scale":1.0,"bias_rounding":"model_dtype", "heads":8,"kv_heads":8,"d":64,"causal":false,"window":[null,null]});
        for (field, value) in [
            ("heads", json!(12)),
            ("kv_heads", json!(4)),
            ("d", json!(32)),
            ("causal", json!(true)),
            ("window", json!([64, 64])),
            ("kind", json!("alibi")),
            ("score_source", json!("different.py")),
            ("score_abi", json!(2)),
            ("qk_scale", json!(1.0)),
            ("external_softmax_scale", json!(0.125)),
            ("bias_rounding", json!("f32")),
        ] {
            let mut wrong = alibi.clone();
            wrong[field] = value;
            manifest["kernels"].as_array_mut().unwrap().push(wrong);
            assert!(validate_profile_order(&manifest, dtype).is_err(), "{field}");
            manifest["kernels"].as_array_mut().unwrap().pop();
        }
    }

    #[test]
    fn exact_mul_deberta_exports_are_additive_and_keep_ordinary_modes() {
        for dtype in [DType::F16, DType::Bf16] {
            let mut manifest = manifest_for(&[dtype]);
            let before = validate_profile_order(&manifest, dtype).unwrap();
            assert_eq!(before, PROFILES.len());
            push_exact_mul_deberta_pair(&mut manifest);
            assert_eq!(validate_profile_order(&manifest, dtype).unwrap(), before);
            assert_eq!(
                select_profile(&manifest, &spec_for(&PROFILES[0], dtype)).unwrap(),
                0
            );
            assert_eq!(
                select_profile(&manifest, &spec_for(&appended_profile(), dtype)).unwrap(),
                LEGACY_PROFILE_COUNT
            );
        }
    }

    #[test]
    fn exact_mul_deberta_exports_reject_partial_duplicate_and_unknown_entries() {
        let dtype = DType::Bf16;
        let base = manifest_for(&[dtype]);

        let mut partial = base.clone();
        partial["kernels"]
            .as_array_mut()
            .unwrap()
            .push(exact_mul_deberta_kernel(
                crate::fa4_deberta::EXACT_MUL_BOTH_SYMBOL,
            ));
        assert!(validate_profile_order(&partial, dtype).is_err());

        let mut duplicate = base.clone();
        push_exact_mul_deberta_pair(&mut duplicate);
        duplicate["kernels"]
            .as_array_mut()
            .unwrap()
            .push(exact_mul_deberta_kernel(
                crate::fa4_deberta::EXACT_MUL_BOTH_SYMBOL,
            ));
        assert!(validate_profile_order(&duplicate, dtype).is_err());

        let mut unknown = base;
        push_exact_mul_deberta_pair(&mut unknown);
        unknown["kernels"]
            .as_array_mut()
            .unwrap()
            .push(exact_mul_deberta_kernel("fa4_deberta_exact_mul_other_bf16"));
        assert!(validate_profile_order(&unknown, dtype).is_err());
    }

    #[test]
    fn exact_mul_deberta_exports_validate_before_dtype_filtering() {
        let dtype = DType::F16;
        let mut manifest = manifest_for(&[dtype]);
        push_exact_mul_deberta_pair(&mut manifest);
        let index =
            exact_mul_deberta_position(&manifest, crate::fa4_deberta::EXACT_MUL_BOTH_SYMBOL);
        for (field, value) in [
            ("relative_reciprocal_f32_bits", json!(0x3d939a85u64)),
            ("relative_divisor_f32", json!(11.3125)),
            ("window", json!([-1, null])),
            ("dtype", json!("fp16")),
        ] {
            let mut wrong = manifest.clone();
            wrong["kernels"][index][field] = value;
            assert!(
                validate_profile_order(&wrong, dtype).is_err(),
                "accepted corrupt exact-mul {field}"
            );
        }
        assert_eq!(
            validate_profile_order(&manifest, dtype).unwrap(),
            PROFILES.len()
        );
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
    fn gpu_local_radius128_boundary_and_changed_heads_match_cpu_reference() {
        let (root, context, stream) = gpu_setup();
        // One 260-token sequence crosses the radius-128 boundary at rows
        // 128..131 and at the tail. run_case compares every row with the CPU
        // reference, so an off-by-one local bound fails.
        let offsets = [0i32, 260];
        for (heads, kv_heads) in [(6, 6), (12, 12)] {
            for wrapper in [true, false] {
                let mut spec = spec_for(&PROFILES[10], DType::Bf16);
                spec.heads = heads;
                spec.kv_heads = kv_heads;
                spec.window = Window::Local {
                    left: 128,
                    right: 128,
                };
                run_case::<bf16>(
                    &context, &root, &stream, spec, &offsets, &offsets, 260, wrapper,
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
