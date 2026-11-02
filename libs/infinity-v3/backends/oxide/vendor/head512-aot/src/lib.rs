// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Cold-loaded Rust interface for the isolated Torch-free head_dim=512
//! attention provider.
//!
//! The provider is a plain CUDA AOT library derived from the isolated
//! FlashAttention 2.8.3 Native512 bridge. It is not a general FlashAttention-2
//! backend: it exists only to bridge d512 Gemma4/EmbeddingGemma2 attention
//! until the preferred FA4 path can express this geometry, and it leaves the
//! normal FA4 path unchanged. The crate performs no CUDA context query,
//! allocation, synchronization, graph capture or JIT work.
//!
//! Cold preparation is mandatory. After the caller binds its one owned context,
//! [`Provider::prepare_current_context`] applies the fixed kernel shared-memory
//! attributes exactly once. Warm [`Provider::enqueue`] then only performs
//! checked host validation plus one direct launch on the caller's stream.

use libloading::Library;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::ffi::{c_char, c_void, CStr};
use std::fmt;
use std::fs;
use std::path::Path;

pub const HEAD_DIM: usize = 512;
pub const ABI_VERSION: i32 = 2;
pub const NATIVE_DYNAMIC_SHARED_MEMORY_BYTES: u64 = 128 * 1024;
pub const COMPACT_DYNAMIC_SHARED_MEMORY_BYTES: u64 = 96 * 1024;
pub const LSE_LAYOUT: &str = "[heads, total_q] f32";
const SUPPORTED_ARCHES: [&str; 3] = ["sm_90", "sm_100", "sm_120"];
const MANIFEST_SCHEMA: &str = "infinity-head512-aot/v2";
const VECTOR_BASE_ALIGNMENT_BYTES: u64 = 16;
const VECTOR_STRIDE_ALIGNMENT_ELEMENTS: i64 = 8;

/// Explicitly compiled wide-head kernel shapes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KernelVariant {
    /// Native M64/N32/W4 tile requiring 128 KiB opt-in shared memory.
    Native,
    /// Compact M32/N32/W2 tile requiring 96 KiB opt-in shared memory.
    Compact,
}

impl KernelVariant {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::Compact => "compact",
        }
    }

    pub const fn code(self) -> i32 {
        match self {
            Self::Native => 1,
            Self::Compact => 2,
        }
    }

    pub const fn block_m(self) -> usize {
        match self {
            Self::Native => 64,
            Self::Compact => 32,
        }
    }

    pub const fn block_n(self) -> usize {
        32
    }

    pub const fn warps(self) -> usize {
        match self {
            Self::Native => 4,
            Self::Compact => 2,
        }
    }

    pub const fn dynamic_shared_memory_bytes(self) -> u64 {
        match self {
            Self::Native => NATIVE_DYNAMIC_SHARED_MEMORY_BYTES,
            Self::Compact => COMPACT_DYNAMIC_SHARED_MEMORY_BYTES,
        }
    }

    /// Standard cold selection for an architecture directory. Qualification
    /// builds of the compact shape are loaded explicitly instead.
    pub fn for_architecture(architecture: &str) -> Result<Self, Error> {
        match architecture {
            "sm_90" | "sm_100" => Ok(Self::Native),
            "sm_120" => Ok(Self::Compact),
            other => Err(Error::Invalid(format!(
                "unsupported head512 architecture {other}"
            ))),
        }
    }

    fn from_name(name: &str) -> Result<Self, Error> {
        match name {
            "native" => Ok(Self::Native),
            "compact" => Ok(Self::Compact),
            other => Err(Error::Invalid(format!(
                "unsupported head512 kernel variant {other}"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DType {
    F16,
    Bf16,
}

impl DType {
    pub const fn code(self) -> i32 {
        match self {
            Self::F16 => 0,
            Self::Bf16 => 1,
        }
    }

    pub const fn bytes(self) -> usize {
        2
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::F16 => "fp16",
            Self::Bf16 => "bf16",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Window {
    Global,
    Local { left: i32, right: i32 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Head512Spec {
    pub dtype: DType,
    pub causal: bool,
    pub window: Window,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Capabilities {
    pub abi_version: i32,
    pub head_dim: usize,
    pub fp16: bool,
    pub bf16: bool,
    pub causal: bool,
    pub noncausal: bool,
    pub global: bool,
    pub local: bool,
    pub ragged_qk: bool,
    pub runtime_gqa: bool,
    pub dynamic_shared_memory_bytes: u64,
    pub kernel_variant: KernelVariant,
    pub max_optin_shared_memory_bytes: u64,
    pub local_experimental: bool,
}

impl Capabilities {
    pub fn supports(&self, spec: Head512Spec) -> bool {
        self.head_dim == HEAD_DIM
            && self.dynamic_shared_memory_bytes <= self.max_optin_shared_memory_bytes
            && match spec.dtype {
                DType::F16 => self.fp16,
                DType::Bf16 => self.bf16,
            }
            && if spec.causal {
                self.causal
            } else {
                self.noncausal
            }
            && match spec.window {
                Window::Global => self.global,
                Window::Local { left, right } => self.local && left >= 0 && right >= 0,
            }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResolvedMask {
    pub is_causal: bool,
    pub left: i32,
    pub right: i32,
}

/// Reproduce the exact Native packed-FMA mask normalization for a known
/// `max_k`. This is pure host policy; no device values are inspected.
pub fn resolve_mask(causal: bool, window: Window, max_k: usize) -> Result<ResolvedMask, Error> {
    if max_k == 0 || max_k > i32::MAX as usize {
        return Err(Error::Invalid("max_k must be in 1..=i32::MAX".into()));
    }
    let max_k = max_k as i64;
    let (mut left, mut right) = match window {
        Window::Global => (-1i64, -1i64),
        Window::Local { left, right } => {
            if left < 0 || right < 0 {
                return Err(Error::Invalid("local windows must be nonnegative".into()));
            }
            (left as i64, right as i64)
        }
    };
    if left > max_k {
        left = -1;
    }
    if right > max_k {
        right = -1;
    }
    if causal {
        right = 0;
    }
    let is_causal = left < 0 && right == 0;
    if left < 0 && right >= 0 {
        left = max_k;
    }
    if left >= 0 && right < 0 {
        right = max_k;
    }
    Ok(ResolvedMask {
        is_causal,
        left: i32::try_from(left).map_err(|_| Error::Invalid("left window overflow".into()))?,
        right: i32::try_from(right).map_err(|_| Error::Invalid("right window overflow".into()))?,
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum PrepareState {
    Cold,
    Ready,
    Poisoned(String),
}

impl PrepareState {
    fn require_ready(&self) -> Result<(), Error> {
        match self {
            Self::Ready => Ok(()),
            Self::Cold => Err(Error::Unprepared),
            Self::Poisoned(message) => Err(Error::Provider(message.clone())),
        }
    }

    fn is_ready(&self) -> bool {
        matches!(self, Self::Ready)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeviceRegion {
    pub ptr: u64,
    pub bytes: u64,
}

impl DeviceRegion {
    pub fn new(ptr: u64, bytes: u64) -> Result<Self, Error> {
        if ptr == 0 || bytes == 0 {
            return Err(Error::Invalid(
                "device regions need a non-null pointer and positive size".into(),
            ));
        }
        Ok(Self { ptr, bytes })
    }
}

/// Outer Work-owned output storage. Construction and enqueue never allocate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Head512Work {
    pub out: DeviceRegion,
    pub lse: DeviceRegion,
}

impl Head512Work {
    pub fn new(out: DeviceRegion, lse: DeviceRegion) -> Self {
        Self { out, lse }
    }

    pub fn layout(&self, batch: &Head512Batch) -> Result<WorkLayout, Error> {
        batch.layout(self)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorkLayout {
    /// Logical output count: total_q * heads * HEAD_DIM.
    pub out_elements: usize,
    /// Physical output span implied by the caller's O row/head strides.
    pub out_physical_elements: u64,
    /// Physical output bytes that `work.out` must own.
    pub out_bytes: u64,
    pub lse_elements: usize,
    pub lse_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Head512Batch {
    pub q: u64,
    pub k: u64,
    pub v: u64,
    pub q_offsets: u64,
    pub q_offsets_bytes: usize,
    pub k_offsets: u64,
    pub k_offsets_bytes: usize,
    pub total_q: usize,
    pub total_k: usize,
    pub batch: usize,
    pub max_q: usize,
    pub max_k: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub scale: f64,
    pub dtype: DType,
    pub causal: bool,
    pub window: Window,
    /// q_row,q_head,q_d, k_row,k_head,k_d, v_row,v_head,v_d,
    /// o_row,o_head,o_d. Element strides; the three d strides are one.
    pub strides: [i64; 12],
}

impl Head512Batch {
    pub fn spec(&self) -> Head512Spec {
        Head512Spec {
            dtype: self.dtype,
            causal: self.causal,
            window: self.window,
        }
    }

    pub fn validate(&self, capabilities: &Capabilities) -> Result<(), Error> {
        if !capabilities.supports(self.spec()) {
            return Err(Error::Unsupported(
                "provider does not declare this d512 dtype/mask".into(),
            ));
        }
        if self.q == 0 || self.k == 0 || self.v == 0 || self.q_offsets == 0 || self.k_offsets == 0 {
            return Err(Error::Invalid("missing head512 device pointer".into()));
        }
        if self.total_q == 0
            || self.total_k == 0
            || self.batch == 0
            || self.max_q == 0
            || self.max_k == 0
            || self.heads == 0
            || self.kv_heads == 0
        {
            return Err(Error::Invalid("head512 geometry must be positive".into()));
        }
        if self.batch > self.total_q
            || self.batch > self.total_k
            || self.max_q > self.total_q
            || self.max_k > self.total_k
        {
            return Err(Error::Invalid(
                "head512 token counts are inconsistent".into(),
            ));
        }
        if !self.heads.is_multiple_of(self.kv_heads) {
            return Err(Error::Invalid(
                "head512 heads must be divisible by kv_heads".into(),
            ));
        }
        for (name, value) in [
            ("total_q", self.total_q),
            ("total_k", self.total_k),
            ("batch", self.batch),
            ("heads", self.heads),
            ("kv_heads", self.kv_heads),
        ] {
            if value > i32::MAX as usize {
                return Err(Error::Invalid(format!(
                    "{name} exceeds int32 kernel bounds"
                )));
            }
        }
        if self.max_q > (i32::MAX as usize - 127) || self.max_k > (i32::MAX as usize - 127) {
            return Err(Error::Invalid(
                "max sequence exceeds rounded int32 kernel bounds".into(),
            ));
        }
        if !self.scale.is_finite()
            || self.scale <= 0.0
            || !(self.scale as f32).is_finite()
            || (self.scale as f32) <= 0.0
        {
            return Err(Error::Invalid(
                "head512 scale must be finite and positive as f32".into(),
            ));
        }
        let offset_bytes = self
            .batch
            .checked_add(1)
            .and_then(|count| count.checked_mul(4))
            .ok_or_else(|| Error::Invalid("offset byte count overflow".into()))?;
        if !self.q_offsets.is_multiple_of(4) || !self.k_offsets.is_multiple_of(4) {
            return Err(Error::Invalid(
                "cumulative offset pointers require four-byte alignment".into(),
            ));
        }
        for pointer in [self.q_offsets, self.k_offsets] {
            pointer
                .checked_add(offset_bytes as u64)
                .ok_or_else(|| Error::Invalid("cumulative offset pointer span overflows".into()))?;
        }
        if self.q_offsets_bytes < offset_bytes || self.k_offsets_bytes < offset_bytes {
            return Err(Error::Invalid(
                "cumulative offset buffers must hold batch+1 int32 values".into(),
            ));
        }
        resolve_mask(self.causal, self.window, self.max_k)?;
        let element_bytes = self.dtype.bytes();
        checked_input_span(
            "q",
            self.total_q,
            self.heads,
            &self.strides[0..3],
            self.q,
            element_bytes,
        )?;
        checked_input_span(
            "k",
            self.total_k,
            self.kv_heads,
            &self.strides[3..6],
            self.k,
            element_bytes,
        )?;
        checked_input_span(
            "v",
            self.total_k,
            self.kv_heads,
            &self.strides[6..9],
            self.v,
            element_bytes,
        )?;
        checked_output_layout(
            self.total_q,
            self.heads,
            &self.strides[9..12],
            self.dtype.bytes(),
        )?;
        Ok(())
    }

    pub fn layout(&self, work: &Head512Work) -> Result<WorkLayout, Error> {
        if self.total_q == 0
            || self.total_k == 0
            || self.batch == 0
            || self.max_q == 0
            || self.max_k == 0
            || self.heads == 0
            || self.kv_heads == 0
        {
            return Err(Error::Invalid("head512 geometry must be positive".into()));
        }
        let out_elements = checked_product(self.total_q, self.heads, "output elements")?
            .checked_mul(HEAD_DIM)
            .ok_or_else(|| Error::Invalid("output element overflow".into()))?;
        let (out_physical_elements, out_bytes) = checked_output_span(
            self.total_q,
            self.heads,
            &self.strides[9..12],
            work.out.ptr,
            self.dtype.bytes(),
        )?;
        let lse_elements = checked_product(self.total_q, self.heads, "lse elements")?;
        let lse_bytes = checked_product(lse_elements, 4, "lse bytes")? as u64;
        if work.lse.ptr == 0 || !work.lse.ptr.is_multiple_of(4) {
            return Err(Error::Invalid(
                "LSE pointer requires four-byte alignment".into(),
            ));
        }
        work.lse
            .ptr
            .checked_add(lse_bytes)
            .ok_or_else(|| Error::Invalid("LSE pointer span overflows".into()))?;
        if work.out.bytes < out_bytes {
            return Err(Error::Invalid(format!(
                "outer Work output capacity is too small: need {out_bytes} physical bytes"
            )));
        }
        if work.lse.bytes < lse_bytes {
            return Err(Error::Invalid(
                "outer Work lse capacity is too small".into(),
            ));
        }
        Ok(WorkLayout {
            out_elements,
            out_physical_elements,
            out_bytes,
            lse_elements,
            lse_bytes,
        })
    }
}

fn checked_product(left: usize, right: usize, what: &str) -> Result<usize, Error> {
    left.checked_mul(right)
        .ok_or_else(|| Error::Invalid(format!("{what} overflow")))
}

fn extent_i128(tokens: usize, heads: usize, strides: &[i64], what: &str) -> Result<i128, Error> {
    if strides.len() != 3 || strides[2] != 1 {
        return Err(Error::Invalid(format!("{what} d stride must be one")));
    }
    if strides[0] <= 0 || strides[1] < HEAD_DIM as i64 {
        return Err(Error::Invalid(format!(
            "{what} row/head strides are invalid"
        )));
    }
    Ok((tokens as i128 - 1) * strides[0] as i128
        + (heads as i128 - 1) * strides[1] as i128
        + (HEAD_DIM as i128 - 1))
}

fn checked_pointer_span(
    base: u64,
    last_element: u64,
    element_bytes: usize,
    what: &str,
) -> Result<(), Error> {
    if base == 0 {
        return Err(Error::Invalid(format!("{what} device pointer is null")));
    }
    let elements = last_element
        .checked_add(1)
        .ok_or_else(|| Error::Invalid(format!("{what} element span overflow")))?;
    let bytes = (elements as u128)
        .checked_mul(element_bytes as u128)
        .filter(|bytes| *bytes <= u64::MAX as u128)
        .ok_or_else(|| Error::Invalid(format!("{what} byte span overflow")))?;
    base.checked_add(bytes as u64)
        .ok_or_else(|| Error::Invalid(format!("{what} pointer addition overflow")))?;
    Ok(())
}

fn checked_input_span(
    name: &str,
    tokens: usize,
    heads: usize,
    strides: &[i64],
    base: u64,
    element_bytes: usize,
) -> Result<(), Error> {
    if !base.is_multiple_of(VECTOR_BASE_ALIGNMENT_BYTES) {
        return Err(Error::Invalid(format!(
            "{name} base pointer must be at least 16-byte aligned for 128-bit vector loads"
        )));
    }
    if strides.len() != 3 || strides[0] <= 0 || strides[1] < HEAD_DIM as i64 {
        return Err(Error::Invalid(format!(
            "{name} row/head strides must be positive multiples of 8 elements"
        )));
    }
    let stride_alignment = VECTOR_STRIDE_ALIGNMENT_ELEMENTS as u64;
    if !(strides[0] as u64).is_multiple_of(stride_alignment)
        || !(strides[1] as u64).is_multiple_of(stride_alignment)
    {
        return Err(Error::Invalid(format!(
            "{name} row/head strides must be positive multiples of 8 elements"
        )));
    }
    let extent = extent_i128(tokens, heads, strides, name)?;
    if extent < 0 || extent > i64::MAX as i128 {
        return Err(Error::Invalid(format!(
            "{name} stride extent overflows int64"
        )));
    }
    checked_pointer_span(base, extent as u64, element_bytes, name)
}

fn checked_output_layout(
    tokens: usize,
    heads: usize,
    strides: &[i64],
    element_bytes: usize,
) -> Result<(u64, u64), Error> {
    if strides.len() != 3 {
        return Err(Error::Invalid("output strides are incomplete".into()));
    }
    let extent = extent_i128(tokens, heads, strides, "output")?;
    if !(strides[0] as u64).is_multiple_of(8) || !(strides[1] as u64).is_multiple_of(8) {
        return Err(Error::Invalid(
            "output row/head strides require eight-element alignment".into(),
        ));
    }
    let minimum_row_stride = (heads as i128 - 1) * strides[1] as i128 + HEAD_DIM as i128;
    if (strides[0] as i128) < minimum_row_stride {
        return Err(Error::Invalid(
            "output row/head layout overlaps; use contiguous or positive padded row-major heads"
                .into(),
        ));
    }
    if extent < 0 || extent > i64::MAX as i128 {
        return Err(Error::Invalid(
            "output stride extent overflows int64".into(),
        ));
    }
    let last_element = extent as u64;
    let elements = last_element
        .checked_add(1)
        .ok_or_else(|| Error::Invalid("output physical element span overflow".into()))?;
    let bytes = (elements as u128)
        .checked_mul(element_bytes as u128)
        .filter(|bytes| *bytes <= u64::MAX as u128)
        .ok_or_else(|| Error::Invalid("output physical byte span overflow".into()))?;
    Ok((elements, bytes as u64))
}

fn checked_output_span(
    tokens: usize,
    heads: usize,
    strides: &[i64],
    base: u64,
    element_bytes: usize,
) -> Result<(u64, u64), Error> {
    if base == 0 || !base.is_multiple_of(16) {
        return Err(Error::Invalid(
            "output pointer requires sixteen-byte alignment".into(),
        ));
    }
    let (elements, _) = checked_output_layout(tokens, heads, strides, element_bytes)?;
    let last_element = elements - 1;
    checked_pointer_span(base, last_element, element_bytes, "output")?;
    let bytes = (elements as u128)
        .checked_mul(element_bytes as u128)
        .filter(|bytes| *bytes <= u64::MAX as u128)
        .ok_or_else(|| Error::Invalid("output physical byte span overflow".into()))?;
    Ok((elements, bytes as u64))
}

#[derive(Debug, Deserialize, Clone)]
#[allow(dead_code)]
struct Manifest {
    schema: String,
    abi_version: i32,
    cold_prepare_required: bool,
    warm_launch_only: bool,
    architecture: String,
    compute_capability: u32,
    library: String,
    library_sha256: String,
    head_dim: usize,
    dtypes: Vec<String>,
    causal: bool,
    noncausal: bool,
    windows: Vec<String>,
    ragged_qk: bool,
    runtime_gqa: bool,
    stride_d: usize,
    kernel_variant: String,
    qualification: bool,
    block_m: usize,
    block_n: usize,
    warps: usize,
    dynamic_shared_memory_bytes: u64,
    min_optin_shared_memory_bytes: u64,
    local_experimental: bool,
    torch_free: bool,
    requires_cudart: bool,
    dt_needed: Vec<String>,
    cuda_version: String,
    cutlass_include: String,
    cutlass_cutlass_h_sha256: String,
    nvcc: String,
    cxx: String,
    compile_definitions: Vec<String>,
    source_sha256: BTreeMap<String, String>,
}

impl Manifest {
    fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let manifest: Self = serde_json::from_slice(bytes).map_err(Error::Json)?;
        manifest.validate_contract()?;
        Ok(manifest)
    }

    fn validate_contract(&self) -> Result<(), Error> {
        if self.schema != MANIFEST_SCHEMA {
            return Err(Error::Invalid(format!(
                "unsupported head512 manifest schema {}",
                self.schema
            )));
        }
        if self.abi_version != ABI_VERSION {
            return Err(Error::Invalid("head512 manifest ABI mismatch".into()));
        }
        if !self.cold_prepare_required || !self.warm_launch_only {
            return Err(Error::Invalid(
                "head512 manifest must require cold prepare and warm-only launch".into(),
            ));
        }
        if !SUPPORTED_ARCHES.contains(&self.architecture.as_str()) {
            return Err(Error::Invalid(format!(
                "unsupported head512 architecture {}",
                self.architecture
            )));
        }
        let expected_cc = match self.architecture.as_str() {
            "sm_90" => 90,
            "sm_100" => 100,
            "sm_120" => 120,
            other => {
                return Err(Error::Invalid(format!(
                    "unsupported head512 architecture {other}"
                )))
            }
        };
        if self.compute_capability != expected_cc {
            return Err(Error::Invalid(
                "head512 manifest architecture/capability disagree".into(),
            ));
        }
        if self.head_dim != HEAD_DIM {
            return Err(Error::Invalid("head512 manifest head_dim mismatch".into()));
        }
        let dtypes = self
            .dtypes
            .iter()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>();
        if dtypes != ["bf16", "fp16"].into_iter().collect() {
            return Err(Error::Invalid(
                "head512 manifest must export fp16 and bf16".into(),
            ));
        }
        let windows = self
            .windows
            .iter()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>();
        if windows != ["global", "local"].into_iter().collect() {
            return Err(Error::Invalid(
                "head512 manifest must export global and local masks".into(),
            ));
        }
        if !self.causal || !self.noncausal || !self.ragged_qk || !self.runtime_gqa {
            return Err(Error::Invalid(
                "head512 manifest lacks required causal/ragged/GQA capability".into(),
            ));
        }
        if self.stride_d != 1 {
            return Err(Error::Invalid(
                "head512 manifest stride_d must be one".into(),
            ));
        }
        let variant = KernelVariant::from_name(&self.kernel_variant)?;
        if self.block_m != variant.block_m()
            || self.block_n != variant.block_n()
            || self.warps != variant.warps()
        {
            return Err(Error::Invalid(
                "head512 manifest kernel tile does not match its variant".into(),
            ));
        }
        let expected_smem = variant.dynamic_shared_memory_bytes();
        if self.dynamic_shared_memory_bytes != expected_smem
            || self.min_optin_shared_memory_bytes != expected_smem
        {
            return Err(Error::Invalid(
                "head512 manifest shared-memory contract mismatch".into(),
            ));
        }
        if !self.local_experimental {
            return Err(Error::Invalid(
                "head512 manifest must mark local windows experimental".into(),
            ));
        }
        if self.qualification {
            if self.architecture != "sm_90" || variant != KernelVariant::Compact {
                return Err(Error::Invalid(
                    "head512 qualification builds are compact sm_90 only".into(),
                ));
            }
        } else if variant != KernelVariant::for_architecture(&self.architecture)? {
            return Err(Error::Invalid(
                "head512 manifest variant does not match its architecture".into(),
            ));
        }
        if !self.torch_free || !self.requires_cudart {
            return Err(Error::Invalid(
                "head512 manifest must be Torch-free and declare exact cudart".into(),
            ));
        }
        if self.dt_needed.is_empty()
            || !self
                .dt_needed
                .iter()
                .any(|name| name.starts_with("libcudart.so"))
        {
            return Err(Error::Invalid(
                "head512 manifest missing exact cudart dependency".into(),
            ));
        }
        if self.dt_needed.iter().any(|name| {
            let lower = name.to_ascii_lowercase();
            lower.contains("torch")
                || lower.contains("c10")
                || lower.contains("aten")
                || lower.contains("python")
        }) {
            return Err(Error::Invalid(
                "head512 manifest declares a forbidden Torch dependency".into(),
            ));
        }
        if self.library.is_empty()
            || self.library.contains('/')
            || self.library.contains('\\')
            || !self.library.ends_with(".so")
        {
            return Err(Error::Invalid(
                "head512 manifest library name is invalid".into(),
            ));
        }
        if self.library_sha256.len() != 64
            || !self
                .library_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(Error::Invalid(
                "head512 manifest library hash is invalid".into(),
            ));
        }
        if self.source_sha256.is_empty()
            || !self.source_sha256.contains_key("native/head512_launch.cu")
            || !self.source_sha256.contains_key("native/head512_bridge.cpp")
        {
            return Err(Error::Invalid(
                "head512 manifest is missing kernel source identity".into(),
            ));
        }
        Ok(())
    }

    fn capabilities(&self, max_optin_shared_memory_bytes: u64) -> Result<Capabilities, Error> {
        Ok(Capabilities {
            abi_version: self.abi_version,
            head_dim: self.head_dim,
            fp16: true,
            bf16: true,
            causal: self.causal,
            noncausal: self.noncausal,
            global: self.windows.iter().any(|window| window == "global"),
            local: self.windows.iter().any(|window| window == "local"),
            ragged_qk: self.ragged_qk,
            runtime_gqa: self.runtime_gqa,
            dynamic_shared_memory_bytes: self.dynamic_shared_memory_bytes,
            kernel_variant: KernelVariant::from_name(&self.kernel_variant)?,
            max_optin_shared_memory_bytes,
            local_experimental: self.local_experimental,
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Head512Info {
    abi_version: i32,
    head_dim: i32,
    dtype_mask: u32,
    window_mask: u32,
    dynamic_shared_memory_bytes: i64,
    min_optin_shared_memory_bytes: i64,
    flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct FfiBatch {
    q: u64,
    k: u64,
    v: u64,
    q_offsets: u64,
    k_offsets: u64,
    total_q: i64,
    total_k: i64,
    batch: i64,
    max_q: i64,
    max_k: i64,
    heads: i64,
    kv_heads: i64,
    scale: f64,
    strides: [i64; 12],
    dtype: i32,
    causal: i32,
    window_left: i32,
    window_right: i32,
}

type AbiFn = unsafe extern "C" fn() -> i32;
type VariantFn = unsafe extern "C" fn() -> i32;
type PrepareFn = unsafe extern "C" fn() -> i32;
type InfoFn = unsafe extern "C" fn(*mut Head512Info) -> i32;
type ErrorFn = unsafe extern "C" fn() -> *const c_char;
type EnqueueFn =
    unsafe extern "C" fn(*const FfiBatch, *mut c_void, u64, *mut c_void, u64, *mut c_void) -> i32;

pub struct Provider {
    _library: Library,
    manifest: Manifest,
    capabilities: Capabilities,
    prepare_fn: PrepareFn,
    enqueue_fn: EnqueueFn,
    error: ErrorFn,
    prepare_state: PrepareState,
}

impl fmt::Debug for Provider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Provider")
            .field("architecture", &self.manifest.architecture)
            .field("capabilities", &self.capabilities)
            .field("prepared", &self.prepare_state.is_ready())
            .finish()
    }
}

impl Provider {
    /// Load the standard cold variant for an architecture without a hardware
    /// shared-memory fact. Prefer [`Provider::load_for_device`] when the
    /// caller knows the actual device limit.
    pub fn load_for_arch(root: &Path, architecture: &str) -> Result<Self, Error> {
        let variant = KernelVariant::for_architecture(architecture)?;
        Self::load_directory(root, architecture, architecture, variant, false, u64::MAX)
    }

    /// Select the cold artifact from caller-supplied hardware facts. The
    /// provider never queries a device or context. Callers bind the returned
    /// provider to a module compiled for this exact architecture and stream
    /// before enqueueing.
    pub fn load_for_device(
        root: &Path,
        architecture: &str,
        max_optin_shared_memory_bytes: u64,
    ) -> Result<Self, Error> {
        let variant = KernelVariant::for_architecture(architecture)?;
        require_shared_memory(variant, max_optin_shared_memory_bytes)?;
        Self::load_directory(
            root,
            architecture,
            architecture,
            variant,
            false,
            max_optin_shared_memory_bytes,
        )
    }

    /// Load the compact sm_90 qualification artifact used to compare compact
    /// arithmetic against the actual Native M64 kernel on H100-class devices.
    /// This is not evidence that SM120 has run.
    pub fn load_compact_qualification(
        root: &Path,
        max_optin_shared_memory_bytes: u64,
    ) -> Result<Self, Error> {
        require_shared_memory(KernelVariant::Compact, max_optin_shared_memory_bytes)?;
        Self::load_directory(
            root,
            "sm_90_compact",
            "sm_90",
            KernelVariant::Compact,
            true,
            max_optin_shared_memory_bytes,
        )
    }

    pub fn load(manifest_path: &Path, library_path: &Path) -> Result<Self, Error> {
        let bytes = fs::read(manifest_path).map_err(Error::Io)?;
        let manifest = Manifest::from_bytes(&bytes)?;
        Self::load_manifest(manifest, library_path, u64::MAX)
    }

    fn load_directory(
        root: &Path,
        directory: &str,
        architecture: &str,
        variant: KernelVariant,
        qualification: bool,
        max_optin_shared_memory_bytes: u64,
    ) -> Result<Self, Error> {
        let dir = root.join(directory);
        let manifest_path = dir.join("manifest.json");
        let bytes = fs::read(&manifest_path).map_err(Error::Io)?;
        let manifest = Manifest::from_bytes(&bytes)?;
        if manifest.architecture != architecture {
            return Err(Error::Invalid(
                "head512 manifest architecture does not match its directory".into(),
            ));
        }
        if manifest.qualification != qualification {
            return Err(Error::Invalid(
                "head512 manifest qualification flag does not match its directory".into(),
            ));
        }
        if KernelVariant::from_name(&manifest.kernel_variant)? != variant {
            return Err(Error::Unsupported(format!(
                "head512 artifact in {directory} does not provide the {} variant",
                variant.name()
            )));
        }
        if manifest.dynamic_shared_memory_bytes > max_optin_shared_memory_bytes {
            return Err(Error::Unsupported(format!(
                "head512 {} variant needs {} bytes of opt-in shared memory; hardware provides {}",
                variant.name(),
                manifest.dynamic_shared_memory_bytes,
                max_optin_shared_memory_bytes
            )));
        }
        let library_path = dir.join(&manifest.library);
        Self::load_manifest(manifest, &library_path, max_optin_shared_memory_bytes)
    }

    fn load_manifest(
        manifest: Manifest,
        library_path: &Path,
        max_optin_shared_memory_bytes: u64,
    ) -> Result<Self, Error> {
        if manifest.dynamic_shared_memory_bytes > max_optin_shared_memory_bytes {
            return Err(Error::Unsupported(format!(
                "head512 artifact needs {} bytes of opt-in shared memory; hardware provides {}",
                manifest.dynamic_shared_memory_bytes, max_optin_shared_memory_bytes
            )));
        }
        let library_bytes = fs::read(library_path).map_err(Error::Io)?;
        let actual = hex_sha256(&library_bytes);
        if actual != manifest.library_sha256 {
            return Err(Error::Invalid(
                "head512 library hash does not match manifest".into(),
            ));
        }
        let library = unsafe { Library::new(library_path) }.map_err(Error::Library)?;
        let abi: libloading::Symbol<AbiFn> =
            unsafe { library.get(b"infinity_head512_abi_v2\0") }.map_err(Error::Library)?;
        let variant_fn: libloading::Symbol<VariantFn> =
            unsafe { library.get(b"infinity_head512_variant_v2\0") }.map_err(Error::Library)?;
        let info: libloading::Symbol<InfoFn> =
            unsafe { library.get(b"infinity_head512_info_v2\0") }.map_err(Error::Library)?;
        let error: libloading::Symbol<ErrorFn> =
            unsafe { library.get(b"infinity_head512_error_v2\0") }.map_err(Error::Library)?;
        let enqueue: libloading::Symbol<EnqueueFn> =
            unsafe { library.get(b"infinity_head512_enqueue_v2\0") }.map_err(Error::Library)?;
        // Mandatory ABI v2 symbol: an old bundle that lacks it is rejected
        // here, before any capability inference or enqueue.
        let prepare: libloading::Symbol<PrepareFn> =
            unsafe { library.get(b"infinity_head512_prepare_v2\0") }.map_err(Error::Library)?;
        let abi_value = unsafe { abi() };
        if abi_value != ABI_VERSION {
            return Err(Error::Invalid(format!(
                "head512 library ABI {abi_value} does not match {ABI_VERSION}"
            )));
        }
        let manifest_variant = KernelVariant::from_name(&manifest.kernel_variant)?;
        let library_variant = unsafe { variant_fn() };
        if library_variant != manifest_variant.code() {
            return Err(Error::Invalid(format!(
                "head512 C ABI variant {library_variant} does not match manifest variant {}",
                manifest_variant.name()
            )));
        }
        let mut ffi_info = Head512Info {
            abi_version: 0,
            head_dim: 0,
            dtype_mask: 0,
            window_mask: 0,
            dynamic_shared_memory_bytes: 0,
            min_optin_shared_memory_bytes: 0,
            flags: 0,
        };
        if unsafe { info(&mut ffi_info) } != 0 {
            return Err(Error::Provider(c_error_string(*error)));
        }
        let capabilities = manifest.capabilities(max_optin_shared_memory_bytes)?;
        if ffi_info.abi_version != ABI_VERSION
            || ffi_info.head_dim != HEAD_DIM as i32
            || ffi_info.dtype_mask != 3
            || ffi_info.window_mask != 3
            || ffi_info.dynamic_shared_memory_bytes
                != manifest_variant.dynamic_shared_memory_bytes() as i64
            || ffi_info.min_optin_shared_memory_bytes
                != manifest_variant.dynamic_shared_memory_bytes() as i64
            || ffi_info.flags != 0x0f
        {
            return Err(Error::Invalid(
                "head512 C ABI capabilities do not match the manifest".into(),
            ));
        }
        Ok(Self {
            prepare_fn: *prepare,
            enqueue_fn: *enqueue,
            error: *error,
            prepare_state: PrepareState::Cold,
            _library: library,
            manifest,
            capabilities,
        })
    }

    pub fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }

    pub fn architecture(&self) -> &str {
        &self.manifest.architecture
    }

    pub fn kernel_variant(&self) -> KernelVariant {
        self.capabilities.kernel_variant
    }

    pub fn library_name(&self) -> &str {
        &self.manifest.library
    }

    /// Declared capability check. Requires cold preparation because the
    /// provider is only usable on the context it was prepared for. Pure
    /// geometry validation remains available through [`Capabilities::supports`].
    pub fn supports(&self, spec: Head512Spec) -> Result<bool, Error> {
        self.prepare_state.require_ready()?;
        Ok(self.capabilities.supports(spec))
    }

    /// Apply the fixed kernel shared-memory attributes for the one context the
    /// caller has already bound to this provider. This is cold work: it must be
    /// called before any [`Provider::enqueue`] and must not be called in a hot
    /// path. It performs no device query, allocation, launch or synchronization.
    ///
    /// # Safety
    /// The caller must own exactly one current CUDA context, must have bound it
    /// before this call, and must keep the library, that context, every stream
    /// and all work alive through GPU completion and provider error drain. The
    /// context must not be reset or migrated afterwards. The caller engine owns
    /// the `Arc<Provider>` and must call this exactly once during cold setup.
    pub unsafe fn prepare_current_context(&mut self) -> Result<(), Error> {
        match &self.prepare_state {
            PrepareState::Ready => return Ok(()),
            PrepareState::Poisoned(message) => return Err(Error::Provider(message.clone())),
            PrepareState::Cold => {}
        }
        let status = (self.prepare_fn)();
        if status != 0 {
            let message = c_error_string(self.error);
            self.prepare_state = PrepareState::Poisoned(message.clone());
            return Err(Error::Provider(message));
        }
        self.prepare_state = PrepareState::Ready;
        Ok(())
    }

    /// Validate outer Work + batch and enqueue exactly one launch.
    ///
    /// # Safety
    /// The caller's Provider library, context, stream and every input must stay
    /// live and unchanged until the GPU has completed the enqueued work. Q, K
    /// and V must name valid non-overlapping device storage that is at least
    /// 16-byte aligned with row/head strides in whole 8-element vectors.
    /// `q_offsets` and `k_offsets` must be device int32 cumulative arrays with
    /// `batch + 1` entries that are nondecreasing and bounded by total_q/total_k.
    /// Output and lse must be caller-owned storage with the physical capacity
    /// reported by [`Provider::enqueue`]'s returned layout, and must not alias
    /// inputs, offsets or each other. The function only validates host facts and
    /// launches once; it performs no allocation, synchronization, device/context
    /// query, graph capture or JIT work.
    pub unsafe fn enqueue(
        &self,
        batch: &Head512Batch,
        work: &mut Head512Work,
        stream: u64,
    ) -> Result<WorkLayout, Error> {
        self.prepare_state.require_ready()?;
        batch.validate(&self.capabilities)?;
        let layout = batch.layout(work)?;
        resolve_mask(batch.causal, batch.window, batch.max_k)?;
        let (window_left, window_right) = match batch.window {
            Window::Global => (-1, -1),
            Window::Local { left, right } => (left, right),
        };
        let ffi = FfiBatch {
            q: batch.q,
            k: batch.k,
            v: batch.v,
            q_offsets: batch.q_offsets,
            k_offsets: batch.k_offsets,
            total_q: batch.total_q as i64,
            total_k: batch.total_k as i64,
            batch: batch.batch as i64,
            max_q: batch.max_q as i64,
            max_k: batch.max_k as i64,
            heads: batch.heads as i64,
            kv_heads: batch.kv_heads as i64,
            scale: batch.scale,
            strides: batch.strides,
            dtype: batch.dtype.code(),
            causal: i32::from(batch.causal),
            window_left,
            window_right,
        };
        let status = (self.enqueue_fn)(
            &ffi,
            work.out.ptr as *mut c_void,
            work.out.bytes,
            work.lse.ptr as *mut c_void,
            work.lse.bytes,
            stream as *mut c_void,
        );
        if status != 0 {
            return Err(Error::Provider(c_error_string(self.error)));
        }
        Ok(layout)
    }
}

fn require_shared_memory(
    variant: KernelVariant,
    max_optin_shared_memory_bytes: u64,
) -> Result<(), Error> {
    let required = variant.dynamic_shared_memory_bytes();
    if max_optin_shared_memory_bytes < required {
        return Err(Error::Unsupported(format!(
            "head512 {} variant needs {required} bytes of opt-in shared memory; hardware provides {max_optin_shared_memory_bytes}",
            variant.name()
        )));
    }
    Ok(())
}

fn hex_sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

fn c_error_string(error: ErrorFn) -> String {
    let pointer = unsafe { error() };
    if pointer.is_null() {
        return "head512 provider error".into();
    }
    unsafe { CStr::from_ptr(pointer) }
        .to_string_lossy()
        .into_owned()
}

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Json(serde_json::Error),
    Library(libloading::Error),
    Invalid(String),
    Unsupported(String),
    Unprepared,
    Provider(String),
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "head512 host I/O: {error}"),
            Self::Json(error) => write!(formatter, "head512 manifest JSON: {error}"),
            Self::Library(error) => write!(formatter, "head512 library load: {error}"),
            Self::Invalid(error) => write!(formatter, "head512 invalid: {error}"),
            Self::Unsupported(error) => write!(formatter, "head512 unsupported: {error}"),
            Self::Unprepared => write!(
                formatter,
                "head512 provider: cold prepare_current_context has not completed"
            ),
            Self::Provider(error) => write!(formatter, "head512 provider: {error}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for Error {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

impl From<libloading::Error> for Error {
    fn from(error: libloading::Error) -> Self {
        Self::Library(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capabilities() -> Capabilities {
        Capabilities {
            abi_version: ABI_VERSION,
            head_dim: HEAD_DIM,
            fp16: true,
            bf16: true,
            causal: true,
            noncausal: true,
            global: true,
            local: true,
            ragged_qk: true,
            runtime_gqa: true,
            dynamic_shared_memory_bytes: NATIVE_DYNAMIC_SHARED_MEMORY_BYTES,
            kernel_variant: KernelVariant::Native,
            max_optin_shared_memory_bytes: NATIVE_DYNAMIC_SHARED_MEMORY_BYTES,
            local_experimental: true,
        }
    }

    fn batch() -> Head512Batch {
        Head512Batch {
            q: 0x1000,
            k: 0x2000,
            v: 0x3000,
            q_offsets: 0x4000,
            q_offsets_bytes: 16,
            k_offsets: 0x5000,
            k_offsets_bytes: 16,
            total_q: 7,
            total_k: 9,
            batch: 3,
            max_q: 5,
            max_k: 9,
            heads: 8,
            kv_heads: 2,
            scale: 1.0,
            dtype: DType::Bf16,
            causal: true,
            window: Window::Global,
            strides: [
                8 * 512,
                512,
                1,
                2 * 544,
                544,
                1,
                2 * 544,
                544,
                1,
                8 * 512,
                512,
                1,
            ],
        }
    }

    #[test]
    fn resolves_native_global_and_local_masks() {
        assert_eq!(
            resolve_mask(false, Window::Global, 9).unwrap(),
            ResolvedMask {
                is_causal: false,
                left: -1,
                right: -1
            }
        );
        assert_eq!(
            resolve_mask(true, Window::Global, 9).unwrap(),
            ResolvedMask {
                is_causal: true,
                left: 9,
                right: 0
            }
        );
        assert_eq!(
            resolve_mask(true, Window::Local { left: 64, right: 0 }, 128).unwrap(),
            ResolvedMask {
                is_causal: false,
                left: 64,
                right: 0
            }
        );
        assert_eq!(
            resolve_mask(
                false,
                Window::Local {
                    left: 64,
                    right: 64
                },
                128
            )
            .unwrap(),
            ResolvedMask {
                is_causal: false,
                left: 64,
                right: 64
            }
        );
        let oversized = resolve_mask(
            false,
            Window::Local {
                left: 256,
                right: 256,
            },
            128,
        )
        .unwrap();
        assert_eq!((oversized.left, oversized.right), (-1, -1));
        assert!(resolve_mask(false, Window::Local { left: -1, right: 0 }, 128).is_err());
    }

    #[test]
    fn validates_ragged_gqa_batch_and_work_capacity() {
        let batch = batch();
        batch.validate(&capabilities()).unwrap();
        let work = Head512Work::new(
            DeviceRegion::new(0x6000, 7 * 8 * 512 * 2).unwrap(),
            DeviceRegion::new(0x7000, 7 * 8 * 4).unwrap(),
        );
        let layout = work.layout(&batch).unwrap();
        assert_eq!(layout.out_elements, 7 * 8 * 512);
        assert_eq!(layout.lse_elements, 7 * 8);

        let short = Head512Work::new(
            DeviceRegion::new(0x6000, 7 * 8 * 512 * 2 - 1).unwrap(),
            DeviceRegion::new(0x7000, 7 * 8 * 4).unwrap(),
        );
        assert!(short.layout(&batch).is_err());
        let mut invalid = batch;
        invalid.heads = 7;
        assert!(invalid.validate(&capabilities()).is_err());
        let mut invalid = batch;
        invalid.q_offsets_bytes = 12;
        assert!(invalid.validate(&capabilities()).is_err());
        let mut invalid = batch;
        invalid.strides[2] = 2;
        assert!(invalid.validate(&capabilities()).is_err());
        let mut invalid = batch;
        invalid.strides[0] = i64::MAX;
        assert!(invalid.validate(&capabilities()).is_err());
    }

    #[test]
    fn rejects_overflowed_layout_and_unsupported_specs() {
        let mut huge = batch();
        huge.total_q = i32::MAX as usize;
        huge.max_q = 1;
        huge.heads = i32::MAX as usize;
        huge.kv_heads = 1;
        let work = Head512Work::new(
            DeviceRegion::new(1, u64::MAX).unwrap(),
            DeviceRegion::new(2, u64::MAX).unwrap(),
        );
        assert!(work.layout(&huge).is_err());

        let caps = capabilities();
        assert!(!caps.supports(Head512Spec {
            dtype: DType::F16,
            causal: true,
            window: Window::Local { left: -1, right: 0 }
        }));
        let mut no_local = caps;
        no_local.local = false;
        assert!(!no_local.supports(Head512Spec {
            dtype: DType::Bf16,
            causal: false,
            window: Window::Local {
                left: 64,
                right: 64
            }
        }));
    }

    #[test]
    fn physical_output_capacity_accounts_for_padded_strides() {
        let mut batch = batch();
        batch.strides[9] = 8 * 544 + 16;
        batch.strides[10] = 544;
        let dense_bytes = 7 * 8 * 512 * 2;
        let dense = Head512Work::new(
            DeviceRegion::new(0x6000, dense_bytes).unwrap(),
            DeviceRegion::new(0x7000, 7 * 8 * 4).unwrap(),
        );
        let error = batch.layout(&dense).unwrap_err();
        assert!(error.to_string().contains("physical bytes"));

        let physical_elements = 6 * (8 * 544 + 16) + 7 * 544 + 512;
        let physical_bytes = physical_elements * 2;
        assert!(physical_bytes > dense_bytes);
        let work = Head512Work::new(
            DeviceRegion::new(0x6000, physical_bytes).unwrap(),
            DeviceRegion::new(0x7000, 7 * 8 * 4).unwrap(),
        );
        let layout = batch.layout(&work).unwrap();
        assert_eq!(layout.out_elements, 7 * 8 * 512);
        assert_eq!(layout.out_physical_elements, physical_elements);
        assert_eq!(layout.out_bytes, physical_bytes);
        batch.validate(&capabilities()).unwrap();
    }

    #[test]
    fn rejects_overlapping_output_head_and_row_layouts() {
        let mut overlapping_heads = batch();
        overlapping_heads.strides[10] = 511;
        assert!(overlapping_heads.validate(&capabilities()).is_err());

        let mut overlapping_rows = batch();
        overlapping_rows.strides[9] = 8 * 512 - 1;
        assert!(overlapping_rows.validate(&capabilities()).is_err());

        let mut tight_rows = batch();
        tight_rows.strides[9] = 7 * 512 + 512;
        assert!(tight_rows.validate(&capabilities()).is_ok());
    }

    #[test]
    fn rejects_wrapping_output_extent_and_pointer_addition() {
        let mut wrapping = batch();
        wrapping.total_q = 4;
        wrapping.strides[9] = i64::MAX / 2;
        let huge = Head512Work::new(
            DeviceRegion::new(0x1000, u64::MAX).unwrap(),
            DeviceRegion::new(0x2000, u64::MAX).unwrap(),
        );
        assert!(wrapping.layout(&huge).is_err());

        let batch = batch();
        let wrapping_pointer = Head512Work::new(
            DeviceRegion::new(u64::MAX - 8, 7 * 8 * 512 * 2).unwrap(),
            DeviceRegion::new(0x7000, 7 * 8 * 4).unwrap(),
        );
        assert!(batch.layout(&wrapping_pointer).is_err());
    }

    #[test]
    fn rejects_misaligned_qkv_vector_storage() {
        let mut misaligned_base = batch();
        misaligned_base.q = 0x1008;
        let error = misaligned_base.validate(&capabilities()).unwrap_err();
        assert!(error.to_string().contains("16-byte aligned"));

        let mut misaligned_head = batch();
        misaligned_head.strides[1] = 513;
        assert!(misaligned_head.validate(&capabilities()).is_err());

        let mut misaligned_row = batch();
        misaligned_row.strides[3] = 2 * 544 + 1;
        assert!(misaligned_row.validate(&capabilities()).is_err());
    }

    #[test]
    fn output_vector_alignment_rejects_base_and_padded_stride_misalignment() {
        let valid = batch();
        let enough = 1 << 20;
        let mut work = Head512Work::new(
            DeviceRegion::new(0x10000, enough).unwrap(),
            DeviceRegion::new(0x20000, enough).unwrap(),
        );
        assert!(valid.layout(&work).is_ok());
        work.out.ptr += 2;
        assert!(valid.layout(&work).is_err());
        work.out.ptr -= 2;
        let mut bad_head = valid;
        bad_head.strides[10] += 1;
        assert!(bad_head.validate(&capabilities()).is_err());
        assert!(bad_head.layout(&work).is_err());
        let mut bad_row = valid;
        bad_row.strides[9] += 2;
        assert!(bad_row.validate(&capabilities()).is_err());
        assert!(bad_row.layout(&work).is_err());
    }

    #[test]
    fn scalar_metadata_and_lse_alignment_and_address_wrap_are_rejected() {
        let valid = batch();
        for key in 0..2 {
            let mut bad = valid;
            if key == 0 {
                bad.q_offsets += 2;
            } else {
                bad.k_offsets += 2;
            }
            assert!(bad.validate(&capabilities()).is_err());
        }
        let mut bad = valid;
        bad.q_offsets = u64::MAX - 3;
        assert!(bad.validate(&capabilities()).is_err());
        let enough = 1 << 20;
        let mut work = Head512Work::new(
            DeviceRegion::new(0x10000, enough).unwrap(),
            DeviceRegion::new(0x20000, enough).unwrap(),
        );
        work.lse.ptr += 2;
        assert!(valid.layout(&work).is_err());
        work.lse.ptr = u64::MAX - 3;
        assert!(valid.layout(&work).is_err());
    }

    #[test]
    fn selects_cold_variant_from_architecture_and_shared_memory() {
        assert_eq!(
            KernelVariant::for_architecture("sm_90").unwrap(),
            KernelVariant::Native
        );
        assert_eq!(
            KernelVariant::for_architecture("sm_100").unwrap(),
            KernelVariant::Native
        );
        assert_eq!(
            KernelVariant::for_architecture("sm_120").unwrap(),
            KernelVariant::Compact
        );
        assert_eq!(
            (
                KernelVariant::Native.block_m(),
                KernelVariant::Native.block_n(),
                KernelVariant::Native.warps(),
                KernelVariant::Native.dynamic_shared_memory_bytes(),
            ),
            (64, 32, 4, NATIVE_DYNAMIC_SHARED_MEMORY_BYTES)
        );
        assert_eq!(
            (
                KernelVariant::Compact.block_m(),
                KernelVariant::Compact.block_n(),
                KernelVariant::Compact.warps(),
                KernelVariant::Compact.dynamic_shared_memory_bytes(),
            ),
            (32, 32, 2, COMPACT_DYNAMIC_SHARED_MEMORY_BYTES)
        );

        let root = tempfile::tempdir().unwrap();
        assert!(matches!(
            Provider::load_for_device(root.path(), "sm_90", NATIVE_DYNAMIC_SHARED_MEMORY_BYTES - 1),
            Err(Error::Unsupported(_))
        ));
        assert!(matches!(
            Provider::load_for_device(
                root.path(),
                "sm_120",
                COMPACT_DYNAMIC_SHARED_MEMORY_BYTES - 1
            ),
            Err(Error::Unsupported(_))
        ));
        assert!(matches!(
            Provider::load_for_device(root.path(), "sm_120", COMPACT_DYNAMIC_SHARED_MEMORY_BYTES),
            Err(Error::Io(_))
        ));
        assert!(matches!(
            Provider::load_compact_qualification(
                root.path(),
                COMPACT_DYNAMIC_SHARED_MEMORY_BYTES - 1
            ),
            Err(Error::Unsupported(_))
        ));
    }

    #[test]
    fn supports_rejects_insufficient_shared_memory() {
        let mut caps = capabilities();
        caps.max_optin_shared_memory_bytes = NATIVE_DYNAMIC_SHARED_MEMORY_BYTES - 1;
        assert!(!caps.supports(Head512Spec {
            dtype: DType::Bf16,
            causal: true,
            window: Window::Global
        }));
    }

    #[test]
    fn prepare_state_is_cold_or_poisoned_and_never_implicitly_ready() {
        assert!(matches!(
            PrepareState::Cold.require_ready(),
            Err(Error::Unprepared)
        ));
        assert!(PrepareState::Ready.require_ready().is_ok());
        assert!(matches!(
            PrepareState::Poisoned("prepare failed".into()).require_ready(),
            Err(Error::Provider(_))
        ));
        assert!(!PrepareState::Cold.is_ready());
        assert!(PrepareState::Ready.is_ready());
    }

    #[test]
    fn rejects_old_abi_and_missing_cold_prepare_contract() {
        let mut manifest = valid_manifest();
        manifest["schema"] = "infinity-head512-aot/v1".into();
        assert!(Manifest::from_bytes(manifest.to_string().as_bytes()).is_err());

        manifest = valid_manifest();
        manifest["abi_version"] = 1.into();
        assert!(Manifest::from_bytes(manifest.to_string().as_bytes()).is_err());

        manifest = valid_manifest();
        manifest["cold_prepare_required"] = false.into();
        assert!(Manifest::from_bytes(manifest.to_string().as_bytes()).is_err());

        manifest = valid_manifest();
        manifest["warm_launch_only"] = false.into();
        assert!(Manifest::from_bytes(manifest.to_string().as_bytes()).is_err());
    }

    #[test]
    fn parses_and_rejects_manifests_without_torch_or_wrong_contract() {
        let mut manifest = valid_manifest();
        let parsed = Manifest::from_bytes(manifest.to_string().as_bytes()).unwrap();
        assert_eq!(parsed.architecture, "sm_90");
        let parsed_capabilities = parsed.capabilities(u64::MAX).unwrap();
        assert_eq!(parsed_capabilities.head_dim, HEAD_DIM);
        assert_eq!(parsed_capabilities.kernel_variant, KernelVariant::Native);
        assert!(parsed_capabilities.local_experimental);

        manifest["torch_free"] = false.into();
        assert!(Manifest::from_bytes(manifest.to_string().as_bytes()).is_err());
        manifest = valid_manifest();
        manifest["dt_needed"] = serde_json::json!(["libtorch.so", "libcudart.so.12"]);
        assert!(Manifest::from_bytes(manifest.to_string().as_bytes()).is_err());
        manifest = valid_manifest();
        manifest["head_dim"] = 256.into();
        assert!(Manifest::from_bytes(manifest.to_string().as_bytes()).is_err());
        manifest = valid_manifest();
        manifest["dynamic_shared_memory_bytes"] = 65536.into();
        assert!(Manifest::from_bytes(manifest.to_string().as_bytes()).is_err());
        manifest = valid_manifest();
        manifest["library_sha256"] = "not-a-hash".into();
        assert!(Manifest::from_bytes(manifest.to_string().as_bytes()).is_err());
        manifest = valid_manifest();
        manifest["kernel_variant"] = "compact".into();
        assert!(Manifest::from_bytes(manifest.to_string().as_bytes()).is_err());
        manifest = valid_manifest();
        manifest["warps"] = 8.into();
        assert!(Manifest::from_bytes(manifest.to_string().as_bytes()).is_err());
        manifest = valid_manifest();
        manifest["local_experimental"] = false.into();
        assert!(Manifest::from_bytes(manifest.to_string().as_bytes()).is_err());

        manifest = valid_manifest();
        manifest["kernel_variant"] = "compact".into();
        manifest["qualification"] = true.into();
        manifest["block_m"] = 32.into();
        manifest["warps"] = 2.into();
        manifest["dynamic_shared_memory_bytes"] = 98304.into();
        manifest["min_optin_shared_memory_bytes"] = 98304.into();
        let parsed = Manifest::from_bytes(manifest.to_string().as_bytes()).unwrap();
        assert_eq!(
            parsed.capabilities(u64::MAX).unwrap().kernel_variant,
            KernelVariant::Compact
        );
    }

    #[test]
    #[ignore = "set HEAD512_TEST_MANIFEST to a manifest generated by scripts/build_aot.sh"]
    fn parses_build_generated_manifest() {
        let path = std::env::var("HEAD512_TEST_MANIFEST").expect("HEAD512_TEST_MANIFEST");
        let bytes = std::fs::read(path).unwrap();
        let manifest = Manifest::from_bytes(&bytes).unwrap();
        assert_eq!(manifest.head_dim, HEAD_DIM);
        assert!(manifest
            .source_sha256
            .contains_key("native/head512_launch.cu"));
        assert!(manifest
            .source_sha256
            .contains_key("native/head512_bridge.cpp"));
        assert!(!manifest.cutlass_cutlass_h_sha256.is_empty());
    }

    fn valid_manifest() -> serde_json::Value {
        serde_json::json!({
            "schema": "infinity-head512-aot/v2",
            "abi_version": 2,
            "cold_prepare_required": true,
            "warm_launch_only": true,
            "architecture": "sm_90",
            "compute_capability": 90,
            "library": "libinfinity_head512_sm_90.so",
            "library_sha256": "0".repeat(64),
            "head_dim": 512,
            "dtypes": ["fp16", "bf16"],
            "causal": true,
            "noncausal": true,
            "windows": ["global", "local"],
            "ragged_qk": true,
            "runtime_gqa": true,
            "stride_d": 1,
            "kernel_variant": "native",
            "qualification": false,
            "block_m": 64,
            "block_n": 32,
            "warps": 4,
            "dynamic_shared_memory_bytes": 131072,
            "min_optin_shared_memory_bytes": 131072,
            "local_experimental": true,
            "torch_free": true,
            "requires_cudart": true,
            "dt_needed": ["libcudart.so.12", "libstdc++.so.6", "libc.so.6"],
            "cuda_version": "12.9",
            "cutlass_include": "/opt/cutlass/include",
            "cutlass_cutlass_h_sha256": "1".repeat(64),
            "nvcc": "/usr/local/cuda/bin/nvcc",
            "cxx": "g++",
            "compile_definitions": ["FLASHATTENTION_DISABLE_DROPOUT"],
            "source_sha256": {
                "native/head512_launch.cu": "2".repeat(64),
                "native/head512_bridge.cpp": "3".repeat(64)
            }
        })
    }
}
