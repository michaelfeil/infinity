// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Explicit experimental owned Qwen3.5-MoE engine.
//!
//! This module is the public, opt-in route for the hybrid 40-layer
//! Qwen3.5-35B-A3B text backbone. It owns the cold weights, provider, kernel
//! bindings, stream, exactly two host-I/O leases and one request Work arena.
//! It is deliberately not selected by the serving factory and never claims
//! model admission.
//!
//! The actual checkpoint is Q16/KV2/d256 causal global BF16. Canonical27
//! appends that exact geometry as mode 26. A legacy canonical26 manifest/DSO
//! still has no such profile, so cold admission returns a typed missing-profile
//! gap plus an append-only proposal. It never borrows the existing
//! `fa4_qwen35_*` Q16/KV4/d256 noncausal profile or a ratio-4 causal kernel.

use crate::cublas::Blas;
use crate::fa4::{AttentionSpec, DType, Fa4, Scale, Window};
use crate::gdn_prepared::{GdnBufferView, GdnCapacity, GdnKernelSet, GdnPrepared};
use crate::qwen35_backbone::{Qwen35BackboneKind, Qwen35BackboneWeights};
use crate::qwen35_execute::{Qwen35AuxKernels, Slabs};
use crate::qwen35_moe_config::Qwen35MoeConfig;
use crate::qwen35_moe_full::{
    allocate_scratch, scratch_allocation_bytes, Qwen35MoeFullDeviceWeights, Qwen35MoeFullHostPlan,
    Qwen35MoeFullKernels, Qwen35MoeFullPlan, Qwen35MoeFullScratch,
};
use crate::qwen35_moe_layout::Qwen35MoeAttentionLayout;
use crate::qwen35_moe_provider::Qwen35MoeProvider;
use crate::qwen35_prepared::Qwen35Kernels;
use crate::qwen35_tables::Qwen35Tables;
use crate::staging::{HostIo, SlotPool};
use crate::tensor::Bf16;
use crate::work_retirement::{CompletedTicket, CompletionTicket};
use crate::{arg, blocks, infer, launch, product, u32_size, upload_input, InputLayout};
use cuda_core::{Context, DeviceBuffer, Function, Stream};
use infinity_v3_backend_core::{BackendError, Batch, Pool};
use serde_json::Value;
use std::fmt;
use std::mem::ManuallyDrop;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

const QWEN35_HIDDEN: usize = 2048;
const QWEN35_LAYERS: usize = 40;
const QWEN35_FULL_LAYERS: usize = 10;
const QWEN35_LINEAR_LAYERS: usize = 30;
const QWEN35_HEADS: usize = 16;
const QWEN35_KV_HEADS: usize = 2;
const QWEN35_HEAD_DIM: usize = 256;
const QWEN35_EXPERTS: usize = 256;
const QWEN35_TOP_K: usize = 8;
const QWEN35_ROUTED_INTERMEDIATE: usize = 512;
const QWEN35_SHARED_INTERMEDIATE: usize = 512;
const QWEN35_TENSORS: usize = 1811;
const CANONICAL26_BASE_MODES: usize = 26;
const CANONICAL26_SEQUENCE: &str =
    "3b214841c169c3bcaad0574ee9df917c9fc8a05be1e64602cf5b91d99a67ff1e";
const APPENDED_PREFIXES: &[(usize, &str, &str)] = &[
    (
        27,
        "canonical27",
        "7cb60381f0383fdcdd8fd4da8b0dbe0f31252e2236de40aa78ebd54987376114",
    ),
    (
        28,
        "canonical28",
        "e8cf5c714796adedf2562b4d0cd2070af237a4d5b2df39a869b2ca9607bc5e7e",
    ),
    (
        29,
        "canonical29",
        "8d08ad78acc3a31e3b8714503c0c399fa2934f2245534ef2cc59765a017d6a79",
    ),
    (
        30,
        "canonical30",
        "4647be9d34a82927cdaf37387f5314372c7cac9213f2d52a317a9f2ed16a2d26",
    ),
    (
        31,
        "canonical31",
        "187defaacc9f9b9b326b392fbeda675545a73660f44f69e21743743f782a87f1",
    ),
];
const PROPOSED_SYMBOL: &str = "fa4_qwen35_moe_gqa8_bf16";

/// One request's explicit packed-input word geometry. This is the public
/// efbd735 `InputLayout` with the model's exact capacity checks.
pub type ExperimentalMoEInputLayout = InputLayout;

/// Public completion clock/ticket/queue API. The clock is still advanced only
/// by an unsafe caller that observed the real same-stream event or fence.
#[allow(unused_imports)]
pub mod completion {
    #[allow(unused_imports)]
    use super::*;
    pub use crate::work_retirement::{
        CompletedTicket, CompletionClock, CompletionTicket, RetirementQueue,
    };
}

/// One exact FA4 geometry observed from the actual checkpoint config.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExperimentalFa4Geometry {
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub causal: bool,
    pub dtype: &'static str,
    pub window: &'static str,
}

impl ExperimentalFa4Geometry {
    pub const fn actual_qwen35_moe() -> Self {
        Self {
            heads: QWEN35_HEADS,
            kv_heads: QWEN35_KV_HEADS,
            head_dim: QWEN35_HEAD_DIM,
            causal: true,
            dtype: "bf16",
            window: "global",
        }
    }

    pub fn symbol(&self) -> String {
        format!(
            "fa4_qwen35_moe_gqa{}_{}",
            self.heads / self.kv_heads.max(1),
            self.dtype
        )
    }

    pub fn matches_kernel(&self, kernel: &Value) -> bool {
        kernel.get("heads").and_then(Value::as_u64) == u64::try_from(self.heads).ok()
            && kernel.get("kv_heads").and_then(Value::as_u64) == u64::try_from(self.kv_heads).ok()
            && kernel.get("d").and_then(Value::as_u64) == u64::try_from(self.head_dim).ok()
            && kernel.get("causal").and_then(Value::as_bool) == Some(self.causal)
            && kernel.get("dtype").and_then(Value::as_str) == Some(self.dtype)
            && matches!(
                kernel.get("window").and_then(Value::as_array),
                Some(window) if window.len() == 2 && window.iter().all(Value::is_null)
            )
    }

    pub(crate) fn attention_spec(&self) -> AttentionSpec {
        AttentionSpec {
            heads: self.heads,
            kv_heads: self.kv_heads,
            head_dim: self.head_dim,
            causal: self.causal,
            window: Window::Global,
            dtype: DType::Bf16,
            scale: Scale::RsqrtHeadDim,
        }
    }
}

/// Append-only FA4 proposal for the actual MoE geometry. The proposal allocates
/// a new mode index after the canonical26 prefix; it does not reuse or remap an
/// existing profile id.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExperimentalFa4Proposal {
    pub namespace: String,
    pub base_mode_count: usize,
    pub append_mode_index: usize,
    pub append_mode_mask: u64,
    pub symbol: String,
    pub profile: String,
    pub geometry: ExperimentalFa4Geometry,
    pub borrowed_profile_id: Option<usize>,
    pub borrowed_symbol: Option<String>,
    pub existing_modes_unchanged: bool,
    pub new_dso_required: bool,
    pub source_registry_append_required: bool,
}

/// Cold result of comparing the actual checkpoint geometry with one real
/// canonical26 manifest and its DSO. A missing profile is data, not a fallback.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExperimentalFa4Check {
    pub root: PathBuf,
    pub architecture: String,
    pub registry_namespace: String,
    pub mode_count: usize,
    pub profile_sequence_sha256: String,
    pub dso_path: PathBuf,
    pub actual: ExperimentalFa4Geometry,
    pub exact_symbol: Option<String>,
    pub exact_symbol_in_dso: bool,
    pub proposal: Option<ExperimentalFa4Proposal>,
}

impl ExperimentalFa4Check {
    pub fn admitted(&self) -> bool {
        self.exact_symbol_in_dso
    }

    pub fn typed_gap(&self) -> Option<&ExperimentalFa4Proposal> {
        self.proposal.as_ref()
    }
}

/// Cold, payload-free admission report for the actual checkpoint. This opens
/// safetensors headers/index only; it never reads tensor payload, creates a CUDA
/// context, or queries a device.
#[derive(Clone, Debug)]
pub struct ExperimentalMoeColdPlan {
    pub model: PathBuf,
    pub tensor_count: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub full_attention_layers: usize,
    pub linear_attention_layers: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub rope_theta: f32,
    pub rotary_dim: usize,
    pub max_position_embeddings: usize,
    pub layer_types: Vec<String>,
    pub fa4: ExperimentalFa4Check,
}

impl ExperimentalMoeColdPlan {
    /// Validate the real checkpoint and the real canonical26 bundle on the CPU.
    pub fn validate(model: &Path, fa4_root: &Path) -> Result<Self, ExperimentalMoeError> {
        let config = Qwen35MoeConfig::load(model)?;
        let backbone = Qwen35BackboneWeights::load_moe(model)?;
        if backbone.kind != Qwen35BackboneKind::Moe {
            return Err(ExperimentalMoeError::Cold(
                "checkpoint did not bind as a routed MoE backbone".into(),
            ));
        }
        backbone.validate()?;
        let host_plan = Qwen35MoeFullHostPlan::from_backbone(&backbone)?;
        let index: Value = serde_json::from_slice(
            &std::fs::read(model.join("model.safetensors.index.json")).map_err(|error| {
                ExperimentalMoeError::Cold(format!("read checkpoint index: {error}"))
            })?,
        )
        .map_err(|error| ExperimentalMoeError::Cold(format!("parse checkpoint index: {error}")))?;
        let tensor_count = index
            .get("weight_map")
            .and_then(Value::as_object)
            .map_or(0, serde_json::Map::len);
        let t = &config.text;
        let m = &config.moe;
        let expected = [
            (t.hidden_size == QWEN35_HIDDEN, "hidden_size 2048"),
            (t.num_hidden_layers == QWEN35_LAYERS, "40 layers"),
            (t.num_attention_heads == QWEN35_HEADS, "16 query heads"),
            (t.num_key_value_heads == QWEN35_KV_HEADS, "2 KV heads"),
            (t.head_dim == QWEN35_HEAD_DIM, "head_dim 256"),
            (m.num_experts == QWEN35_EXPERTS, "256 experts"),
            (m.num_experts_per_tok == QWEN35_TOP_K, "top-8"),
            (
                m.moe_intermediate_size == QWEN35_ROUTED_INTERMEDIATE,
                "routed I512",
            ),
            (
                m.shared_expert_intermediate_size == QWEN35_SHARED_INTERMEDIATE,
                "shared I512",
            ),
            (
                host_plan.full_attention_layers() == QWEN35_FULL_LAYERS,
                "10 full layers",
            ),
            (
                host_plan.linear_attention_layers() == QWEN35_LINEAR_LAYERS,
                "30 GDN layers",
            ),
            (tensor_count == QWEN35_TENSORS, "1811 tensors"),
        ];
        if let Some((_, reason)) = expected.iter().find(|(ok, _)| !ok) {
            return Err(ExperimentalMoeError::Cold(format!(
                "actual Qwen3.5-MoE geometry mismatch: {reason}"
            )));
        }
        let fa4 = check_fa4(fa4_root, ExperimentalFa4Geometry::actual_qwen35_moe())?;
        Ok(Self {
            model: model.to_path_buf(),
            tensor_count,
            hidden_size: t.hidden_size,
            num_hidden_layers: t.num_hidden_layers,
            full_attention_layers: host_plan.full_attention_layers(),
            linear_attention_layers: host_plan.linear_attention_layers(),
            num_experts: m.num_experts,
            top_k: m.num_experts_per_tok,
            moe_intermediate_size: m.moe_intermediate_size,
            shared_expert_intermediate_size: m.shared_expert_intermediate_size,
            rope_theta: t.rope_theta,
            rotary_dim: t.rotary_dim,
            max_position_embeddings: t.max_position_embeddings,
            layer_types: t
                .layer_types
                .iter()
                .map(|kind| match kind {
                    crate::qwen35_config::Qwen35LayerType::Full => "full_attention".into(),
                    crate::qwen35_config::Qwen35LayerType::Linear => "linear_attention".into(),
                })
                .collect(),
            fa4,
        })
    }
}

/// Typed engine error. The missing-FA4 variant is the expected cold result for
/// the shipped canonical26 DSO and is not converted into a fallback kernel.
#[derive(Debug)]
pub enum ExperimentalMoeError {
    Cold(String),
    Backend(BackendError),
    MissingFa4Profile(Box<ExperimentalFa4Proposal>),
    Admission(String),
}

impl fmt::Display for ExperimentalMoeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cold(message) => write!(f, "Qwen3.5-MoE cold admission: {message}"),
            Self::Backend(error) => write!(f, "Qwen3.5-MoE backend: {error}"),
            Self::MissingFa4Profile(proposal) => write!(
                f,
                "missing-profile: actual {} has no exact canonical26 FA4 export; append {} at mode {} (mask {:#x}) without reusing {} or a ratio-4/wrong-GQA mode",
                proposal.geometry.symbol(),
                proposal.symbol,
                proposal.append_mode_index,
                proposal.append_mode_mask,
                proposal.borrowed_symbol.as_deref().unwrap_or("an existing id"),
            ),
            Self::Admission(message) => write!(f, "Qwen3.5-MoE admission: {message}"),
        }
    }
}

impl std::error::Error for ExperimentalMoeError {}

impl From<BackendError> for ExperimentalMoeError {
    fn from(error: BackendError) -> Self {
        Self::Backend(error)
    }
}

impl From<cuda_core::DriverError> for ExperimentalMoeError {
    fn from(error: cuda_core::DriverError) -> Self {
        Self::Cold(format!("CUDA driver: {error}"))
    }
}

impl From<std::io::Error> for ExperimentalMoeError {
    fn from(error: std::io::Error) -> Self {
        Self::Cold(format!("I/O: {error}"))
    }
}

/// Explicit options for the public MoE route. All paths are supplied by the
/// caller; there is no implicit serving default and no network lookup.
#[derive(Clone, Debug)]
pub struct ExperimentalMoEOptions {
    pub fa4_root: PathBuf,
    pub kernel_module: PathBuf,
    pub moe_aot_root: PathBuf,
    pub gdn_raw_kernels: PathBuf,
    pub gdn_provider: PathBuf,
    pub gdn_aux_kernels: PathBuf,
}

impl ExperimentalMoEOptions {
    pub fn from_environment() -> Result<Self, ExperimentalMoeError> {
        let get = |name: &str| {
            std::env::var_os(name)
                .map(PathBuf::from)
                .ok_or_else(|| ExperimentalMoeError::Cold(format!("{name} is required")))
        };
        Ok(Self {
            fa4_root: get("INFINITY_FA4_ROOT")?,
            kernel_module: get("INFINITY_OXIDE_KERNEL_MODULE")?,
            moe_aot_root: get("INFINITY_MOE_AOT_ROOT")?,
            gdn_raw_kernels: get("QWEN35_GDN_KERNELS")?,
            gdn_provider: get("QWEN35_GDN_PROVIDER")?,
            gdn_aux_kernels: get("QWEN35_GDN_AUX")?,
        })
    }
}

/// A worked public host-I/O lease. The engine owns exactly two pooled slots and
/// a lease is returned to the pool only after its event/fence is drained.
fn fence_once<T>(
    failed: &mut bool,
    fence: impl FnOnce() -> Result<T, BackendError>,
) -> Result<T, BackendError> {
    if *failed {
        return Err(infer("experimental MoE completion fence previously failed"));
    }
    match fence() {
        Ok(value) => Ok(value),
        Err(error) => {
            *failed = true;
            Err(error)
        }
    }
}

pub struct ExperimentalMoELease {
    engine: Arc<()>,
    slot: Option<HostIo>,
    pool: Arc<SlotPool<HostIo>>,
    stream: Arc<Stream>,
    poison: Arc<std::sync::atomic::AtomicBool>,
    started: bool,
    recorded: bool,
    drained: bool,
    failed_fence: bool,
    completion: Option<CompletionTicket>,
}

impl ExperimentalMoELease {
    fn new(
        engine: Arc<()>,
        slot: HostIo,
        pool: Arc<SlotPool<HostIo>>,
        stream: Arc<Stream>,
        poison: Arc<std::sync::atomic::AtomicBool>,
    ) -> Self {
        Self {
            engine,
            slot: Some(slot),
            pool,
            stream,
            poison,
            started: false,
            recorded: false,
            drained: false,
            failed_fence: false,
            completion: None,
        }
    }

    pub fn input_words(&self) -> Result<usize, ExperimentalMoeError> {
        Ok(self
            .slot
            .as_ref()
            .ok_or_else(|| ExperimentalMoeError::Cold("host lease was already released".into()))?
            .input
            .len())
    }

    pub fn output_elements(&self) -> Result<usize, ExperimentalMoeError> {
        Ok(self
            .slot
            .as_ref()
            .ok_or_else(|| ExperimentalMoeError::Cold("host lease was already released".into()))?
            .output
            .len())
    }

    pub fn begin_tracked(&mut self) -> Result<CompletionTicket, BackendError> {
        if let Some(ticket) = self.completion.as_ref() {
            return Ok(ticket.clone());
        }
        let slot = self
            .slot
            .as_mut()
            .ok_or_else(|| infer("Missing experimental MoE host lease"))?;
        let stream_identity = Arc::as_ptr(&self.stream) as usize;
        let ticket = Arc::clone(&slot.clock).ticket(stream_identity)?;
        self.started = true;
        self.completion = Some(ticket.clone());
        Ok(ticket)
    }

    pub fn record(&mut self) -> Result<(), BackendError> {
        self.started = true;
        let slot = self
            .slot
            .as_ref()
            .ok_or_else(|| infer("Missing experimental MoE host lease"))?;
        slot.event.record(&self.stream).map_err(infer)?;
        self.recorded = true;
        Ok(())
    }

    pub fn started(&self) -> bool {
        self.started
    }

    pub fn wait(&mut self) -> Result<(), BackendError> {
        if self.failed_fence {
            return Err(infer("experimental MoE completion fence previously failed"));
        }
        if self.drained {
            return Ok(());
        }
        let complete = fence_once(&mut self.failed_fence, || match self.slot.as_ref() {
            Some(slot) if self.recorded => slot.event.synchronize().map_err(infer),
            Some(_) if self.started => self.stream.synchronize().map_err(infer),
            Some(_) => Ok(()),
            None => Err(infer("Missing experimental MoE host lease")),
        });
        if let Err(error) = complete {
            self.latch_failed_fence();
            return Err(error);
        }
        if let Some(ticket) = self.completion.as_ref() {
            // SAFETY: the same stream event or whole-stream fence just returned.
            unsafe { ticket.mark_completed_after_fence() };
        }
        self.drained = true;
        Ok(())
    }

    fn latch_failed_fence(&mut self) {
        self.failed_fence = true;
        self.drained = true;
        self.poison
            .store(true, std::sync::atomic::Ordering::Release);
        self.pool.close();
        if let Some(slot) = self.slot.take() {
            std::mem::forget(slot);
        }
    }

    pub fn output(&self, elements: usize) -> Result<&[f32], BackendError> {
        if !self.drained {
            return Err(infer("Experimental MoE output has not completed"));
        }
        self.slot
            .as_ref()
            .ok_or_else(|| infer("Missing experimental MoE host lease"))?
            .output
            .as_slice()
            .get(..elements)
            .ok_or_else(|| infer("Experimental MoE output exceeds the leased arena"))
    }

    fn prepare(&mut self, input_words: usize, output_elements: usize) -> Result<(), BackendError> {
        prepare_healthy(
            self.pool.poisoned(),
            self.poison.load(std::sync::atomic::Ordering::Acquire),
        )?;
        let slot = self
            .slot
            .as_mut()
            .ok_or_else(|| infer("Missing experimental MoE host lease"))?;
        let key = (input_words, output_elements, 0);
        if slot.input.len() >= input_words && slot.output.len() >= output_elements {
            slot.key = key;
            return Ok(());
        }
        let old = slot.bytes();
        let input = cuda_core::PinnedHostBuffer::zeroed(self.stream.context(), input_words)
            .map_err(infer)?;
        let output = cuda_core::PinnedHostBuffer::zeroed(self.stream.context(), output_elements)
            .map_err(infer)?;
        let new = input.num_bytes().saturating_add(output.num_bytes());
        self.pool.grow(old, new, false)?;
        slot.key = key;
        slot.input = input;
        slot.output = output;
        Ok(())
    }
}

impl Drop for ExperimentalMoELease {
    fn drop(&mut self) {
        if self.failed_fence {
            return;
        }
        if self.slot.is_none() {
            return;
        }
        if !self.drained {
            let complete = fence_once(&mut self.failed_fence, || match self.slot.as_ref() {
                Some(slot) if self.recorded => slot.event.synchronize().map_err(infer),
                Some(_) if self.started => self.stream.synchronize().map_err(infer),
                Some(_) => Ok(()),
                None => Err(infer("Missing experimental MoE host lease")),
            });
            if complete.is_err() {
                self.latch_failed_fence();
                return;
            }
            if let Some(ticket) = self.completion.as_ref() {
                // SAFETY: the actual same-stream fence returned above.
                unsafe { ticket.mark_completed_after_fence() };
            }
        }
        if let Some(slot) = self.slot.take() {
            self.pool.release(slot);
        }
    }
}

/// The public explicit engine. It owns the cold model and exactly two host-I/O
/// leases; a Work owns the input/attention slabs and separate FFN scratch.
pub struct ExperimentalMoEEngine {
    inner: Mutex<Inner>,
}

struct Inner {
    identity: Arc<()>,
    poison: Arc<std::sync::atomic::AtomicBool>,
    context: Arc<Context>,
    stream: Arc<Stream>,
    blas: Blas,
    math: Qwen35Kernels,
    aux: Qwen35AuxKernels,
    fa4: Fa4,
    gather: Function,
    pool: Function,
    mean: Function,
    raw: Function,
    tables: Qwen35Tables,
    weights: Qwen35MoeFullDeviceWeights,
    provider: Qwen35MoeProvider,
    kernels: Qwen35MoeFullKernels,
    options: ExperimentalMoEOptions,
    io: Arc<SlotPool<HostIo>>,
    gdn_major: i32,
    gdn_sm_count: i32,
}

fn engine_unhealthy(io_poisoned: bool, work_fence_poisoned: bool) -> bool {
    io_poisoned || work_fence_poisoned
}

fn prepare_healthy(io_poisoned: bool, work_fence_poisoned: bool) -> Result<(), BackendError> {
    if engine_unhealthy(io_poisoned, work_fence_poisoned) {
        Err(BackendError::Unhealthy)
    } else {
        Ok(())
    }
}

fn admit_healthy(io_poisoned: bool, work_fence_poisoned: bool) -> Result<(), ExperimentalMoeError> {
    if engine_unhealthy(io_poisoned, work_fence_poisoned) {
        Err(ExperimentalMoeError::Backend(BackendError::Unhealthy))
    } else {
        Ok(())
    }
}

fn validate_moe_batch(
    batch: &Batch,
    config: &crate::qwen35_moe_config::Qwen35MoeConfig,
) -> Result<usize, ExperimentalMoeError> {
    let text = &config.text;
    crate::validate_qwen35_batch_geometry(
        batch,
        text.max_position_embeddings,
        text.vocab_size,
        text.q_width()
            .map_err(|error| ExperimentalMoeError::Admission(error.to_string()))?,
        text.kv_width()
            .map_err(|error| ExperimentalMoeError::Admission(error.to_string()))?,
        config
            .moe
            .moe_intermediate_size
            .max(config.moe.shared_expert_intermediate_size),
        text.hidden_size,
    )
    .map_err(|error| ExperimentalMoeError::Admission(error.to_string()))
}

impl ExperimentalMoEEngine {
    /// Cold payload-free check. This is the Root preflight entry.
    pub fn cold_plan(
        model: &Path,
        fa4_root: &Path,
    ) -> Result<ExperimentalMoeColdPlan, ExperimentalMoeError> {
        ExperimentalMoeColdPlan::validate(model, fa4_root)
    }

    /// Explicit experimental constructor. It fails before any CUDA allocation
    /// when the actual geometry has no exact FA4 profile.
    pub fn load(
        model: &Path,
        device_id: usize,
        options: ExperimentalMoEOptions,
    ) -> Result<Self, ExperimentalMoeError> {
        let plan = Self::cold_plan(model, &options.fa4_root)?;
        if let Some(proposal) = plan.fa4.typed_gap() {
            return Err(ExperimentalMoeError::MissingFa4Profile(Box::new(
                proposal.clone(),
            )));
        }
        let spec = plan.fa4.actual.attention_spec();
        if !Fa4::supports(&spec) {
            return Err(ExperimentalMoeError::Admission(
                "FA4 manifest has the exact symbol but this source registry has no matching profile; append the proposal first".into(),
            ));
        }
        if device_id >= crate::device_count()? {
            return Err(ExperimentalMoeError::Admission(
                "CUDA device ordinal is not visible".into(),
            ));
        }
        let context = Context::new(device_id)?;
        let stream = context.new_stream()?;
        let (major, minor) = context.compute_capability()?;
        let capability = major
            .checked_mul(10)
            .and_then(|value| value.checked_add(minor))
            .ok_or_else(|| ExperimentalMoeError::Admission("compute capability overflow".into()))?;
        let gdn_major = major;
        let gdn_sm_count = i32::try_from(
            context
                .multiprocessor_count()
                .map_err(|error| ExperimentalMoeError::Cold(error.to_string()))?,
        )
        .map_err(|_| ExperimentalMoeError::Admission("SM count overflow".into()))?;
        let module_path = if options.kernel_module.is_dir() {
            options
                .kernel_module
                .join(format!("sm_{capability}"))
                .join("kernels.cubin")
        } else {
            options.kernel_module.clone()
        };
        let module = context.load_module_from_image(&std::fs::read(&module_path)?)?;
        let math = Qwen35Kernels::from_module(&module)?;
        let aux = Qwen35AuxKernels {
            join: module.load_function("infinity_qwen35_join_ab_bf16")?,
            silu: module.load_function("infinity_silu_mul_bf16_vector8_native")?,
        };
        let gather = module.load_function("infinity_embedding_gather")?;
        let pool = module.load_function("infinity_pool")?;
        let mean = module.load_function("infinity_mean_pool_bf16_native")?;
        let raw = module.load_function("infinity_bf16_to_f32")?;
        let fa4 = Fa4::new(Arc::clone(&context), &options.fa4_root, spec)?;
        let tables = Qwen35Tables::new(
            &stream,
            math.tables_initializer(),
            plan.rope_theta,
            plan.rotary_dim,
            plan.max_position_embeddings,
        )?;
        let weights = Qwen35MoeFullDeviceWeights::load(model, &stream)?;
        let provider_path = options.moe_aot_root.join(format!(
            "liboxide_moe_sm{}.so",
            match capability {
                90 => 90,
                100 => 100,
                120 => 120,
                other => {
                    return Err(ExperimentalMoeError::Admission(format!(
                        "no Qwen3.5-MoE provider mapping for SM{other}"
                    )));
                }
            }
        ));
        // SAFETY: the caller supplied the independently reviewed AOT provider
        // root; no untrusted path is loaded here.
        let provider = unsafe {
            Qwen35MoeProvider::load(
                &provider_path,
                match capability {
                    90 => crate::moe_abi::ARCH_SM90,
                    100 => crate::moe_abi::ARCH_SM100,
                    120 => crate::moe_abi::ARCH_SM120,
                    _ => unreachable!(),
                },
            )?
        };
        let kernels = Qwen35MoeFullKernels::from_module(&module)
            .map_err(|error| ExperimentalMoeError::Admission(error.to_string()))?;
        let blas = Blas::new(Arc::clone(&context), Arc::clone(&stream))?;
        let io = HostIo::pool(&context)?;
        Ok(Self {
            inner: Mutex::new(Inner {
                identity: Arc::new(()),
                poison: Arc::new(std::sync::atomic::AtomicBool::new(false)),
                context,
                stream,
                blas,
                math,
                aux,
                fa4,
                gather,
                pool,
                mean,
                raw,
                tables,
                weights,
                provider,
                kernels,
                options,
                io,
                gdn_major,
                gdn_sm_count,
            }),
        })
    }

    fn lock(&self) -> Result<MutexGuard<'_, Inner>, ExperimentalMoeError> {
        self.inner
            .lock()
            .map_err(|_| ExperimentalMoeError::Admission("engine mutex is poisoned".into()))
    }

    /// Create one owned request Work. Attention slabs are allocated only for
    /// embedding/attention/GDN widths; routed and shared FFN scratch is a
    /// separate allocation.
    pub fn work(
        &self,
        tokens: usize,
        sequences: usize,
        max_len: usize,
    ) -> Result<ExperimentalMoEWork, ExperimentalMoeError> {
        let inner = self.lock()?;
        admit_healthy(
            inner.io.poisoned(),
            inner.poison.load(std::sync::atomic::Ordering::Acquire),
        )?;
        ExperimentalMoEWork::new(
            Arc::clone(&inner.identity),
            Arc::clone(&inner.poison),
            &inner.context,
            &inner.stream,
            &inner.weights,
            &inner.provider,
            &inner.options,
            inner.gdn_major,
            inner.gdn_sm_count,
            tokens,
            sequences,
            max_len,
        )
    }

    pub fn work_bytes(
        &self,
        tokens: usize,
        sequences: usize,
        max_len: usize,
    ) -> Result<usize, ExperimentalMoeError> {
        let inner = self.lock()?;
        admit_healthy(
            inner.io.poisoned(),
            inner.poison.load(std::sync::atomic::Ordering::Acquire),
        )?;
        ExperimentalMoEWork::allocation_bytes(
            &inner.weights,
            &inner.provider,
            inner.gdn_major,
            inner.gdn_sm_count,
            tokens,
            sequences,
            max_len,
        )
    }

    /// Acquire one of exactly two pooled host-I/O leases.
    pub fn lease(&self) -> Result<ExperimentalMoELease, ExperimentalMoeError> {
        let (identity, io, stream, poison) = {
            let inner = self.lock()?;
            admit_healthy(
                inner.io.poisoned(),
                inner.poison.load(std::sync::atomic::Ordering::Acquire),
            )?;
            (
                Arc::clone(&inner.identity),
                Arc::clone(&inner.io),
                Arc::clone(&inner.stream),
                Arc::clone(&inner.poison),
            )
        };
        let slot = io.acquire()?;
        Ok(ExperimentalMoELease::new(
            identity, slot, io, stream, poison,
        ))
    }

    pub fn active_leases(&self) -> Result<usize, ExperimentalMoeError> {
        Ok(self.lock()?.io.active_leases())
    }

    pub fn host_slots(&self) -> Result<usize, ExperimentalMoeError> {
        Ok(2)
    }

    /// Forward the whole 40-layer hybrid text backbone and return the final
    /// normalized hidden states as F32 rows. No lm_head, logits, vision or MTP
    /// path is claimed.
    pub fn enqueue(
        &self,
        work: &mut ExperimentalMoEWork,
        batch: &Batch,
        lease: &mut ExperimentalMoELease,
        pool: Pool,
    ) -> Result<usize, ExperimentalMoeError> {
        let inner = self.lock()?;
        admit_healthy(
            inner.io.poisoned(),
            inner.poison.load(std::sync::atomic::Ordering::Acquire),
        )?;
        if work.payload_released {
            return Err(ExperimentalMoeError::Admission(
                "Work payload was released".into(),
            ));
        }
        if !Arc::ptr_eq(&work.stream, &inner.stream) {
            return Err(ExperimentalMoeError::Admission(
                "Work belongs to a different engine stream".into(),
            ));
        }
        if lease.started || lease.drained || lease.completion.is_some() {
            return Err(ExperimentalMoeError::Admission(
                "host lease is one-shot; acquire a fresh lease for each submit".into(),
            ));
        }
        if !Arc::ptr_eq(&lease.engine, &inner.identity) {
            return Err(ExperimentalMoeError::Admission(
                "host lease belongs to a different engine".into(),
            ));
        }
        if !Arc::ptr_eq(&work.engine, &inner.identity) {
            return Err(ExperimentalMoeError::Admission(
                "Work belongs to a different engine".into(),
            ));
        }
        validate_moe_batch(batch, &inner.weights.config)?;
        if batch.input_ids.is_empty() {
            return Err(ExperimentalMoeError::Admission("empty batch".into()));
        }
        if batch.input_ids.len() != work.tokens {
            return Err(ExperimentalMoeError::Admission(format!(
                "batch has {} ids but Work owns {}",
                batch.input_ids.len(),
                work.tokens
            )));
        }
        let sequences = batch
            .cumulative_seq_lengths
            .len()
            .checked_sub(1)
            .filter(|value| *value > 0)
            .ok_or_else(|| ExperimentalMoeError::Admission("missing sequence boundaries".into()))?;
        if sequences > work.sequences {
            return Err(ExperimentalMoeError::Admission(
                "batch has more sequences than the Work capacity".into(),
            ));
        }
        let layout = InputLayout::new(
            work.tokens,
            batch.cumulative_seq_lengths.len(),
            batch.pooled_indices.len(),
        )?
        .with_trailer(crate::output::raw_rows(batch)?)?;
        if layout.words() > work.input_words {
            return Err(ExperimentalMoeError::Admission(
                "packed input exceeds the Work input capacity".into(),
            ));
        }
        let raw_rows = crate::output::raw_rows(batch)?;
        let selected_count =
            selected_output_elements(batch.pooled_indices.len(), raw_rows, work.hidden)?;
        if selected_count > selected_capacity_elements(work.sequences, work.tokens, work.hidden)? {
            return Err(ExperimentalMoeError::Admission(
                "selected outputs exceed Work selected capacity".into(),
            ));
        }
        lease.prepare(layout.words(), selected_count)?;
        let ticket = lease.begin_tracked()?;
        work.mark_last_use(ticket);
        let data = &mut *work.data;
        // Keep the packed request in the lease-owned pinned input buffer for
        // the entire async enqueue/D2H/completion boundary. A local pageable
        // Vec would be freed before the upload consumes it.
        {
            let slot = lease
                .slot
                .as_mut()
                .ok_or_else(|| ExperimentalMoeError::Admission("host lease missing".into()))?;
            let host = slot
                .input
                .as_mut_slice()
                .get_mut(..layout.words())
                .ok_or_else(|| {
                    ExperimentalMoeError::Admission("packed input exceeds retained pin".into())
                })?;
            stage_moe_packed_input(host, &layout, batch)?;
            upload_input(&work.stream, &data.input, host, layout.words())?;
        }
        let base = data.input.cu_deviceptr();
        let positions = layout.positions().address(base)?;
        let cumulative = layout.boundaries().address(base)?;
        let uploaded_cumulative =
            GdnBufferView::from_buffer_range(&data.input, layout.boundaries_range())?;
        let slabs = Slabs {
            residual: &data.residual,
            projection: &data.projection,
            activation: &data.activation,
        };
        let mut plan = Qwen35MoeFullPlan::new(
            &inner.weights,
            &inner.provider,
            &work.layout,
            data.scratch.as_ref().ok_or_else(|| {
                ExperimentalMoeError::Admission("Work scratch was released".into())
            })?,
            &slabs,
            work.tokens,
            positions,
            cumulative,
            sequences,
            work.max_len,
            &work.stream,
        )?;
        // SAFETY: this engine owns every cold weight, provider, kernel, table,
        // GDN owner, Work slab/scratch and the stream until the lease fence.
        unsafe {
            plan.enqueue(
                &inner.weights,
                &inner.provider,
                &inner.kernels,
                &inner.blas,
                &inner.math,
                &inner.aux,
                &inner.fa4,
                &data.input,
                &inner.gather,
                &work.stream,
                &slabs,
                data.scratch.as_mut().ok_or_else(|| {
                    ExperimentalMoeError::Admission("Work scratch was released".into())
                })?,
                data.gdn.as_mut().ok_or_else(|| {
                    ExperimentalMoeError::Admission("Work GDN was released".into())
                })?,
                &batch.cumulative_seq_lengths,
                uploaded_cumulative,
                inner.tables.cosine.cu_deviceptr(),
                inner.tables.sine.cu_deviceptr(),
                u32_size(inner.tables.pairs)?,
            )?;
            let raw_rows = crate::output::raw_rows(batch)?;
            let selected_count = product(
                batch
                    .pooled_indices
                    .len()
                    .checked_add(raw_rows)
                    .ok_or_else(|| {
                        ExperimentalMoeError::Admission("selected output count overflow".into())
                    })?,
                work.hidden,
            )?;
            let mut normalized = data.activation.cu_deviceptr();
            let mut bounds = cumulative;
            let mut indices = layout.selections().address(base)?;
            let mut out = data.selected.cu_deviceptr();
            let mut count = u32_size(batch.pooled_indices.len())?;
            let mut pool_width = u32_size(work.hidden)?;
            if count > 0 {
                if pool == Pool::Mean && batch.pooled_indices.len() <= u16::MAX as usize {
                    launch(
                        &inner.mean,
                        &work.stream,
                        (count, u32_size(work.hidden.div_ceil(8))?, 1),
                        0,
                        &mut [
                            arg(&mut normalized),
                            arg(&mut bounds),
                            arg(&mut indices),
                            arg(&mut out),
                            arg(&mut count),
                            arg(&mut pool_width),
                        ],
                    )?;
                } else {
                    let mut mode = match pool {
                        Pool::Cls => 0u32,
                        Pool::Mean => 1u32,
                        Pool::LastToken => 2u32,
                        Pool::Splade => {
                            return Err(ExperimentalMoeError::Admission(
                                "SPLADE is unsupported for Qwen3.5-MoE".into(),
                            ))
                        }
                    };
                    launch(
                        &inner.pool,
                        &work.stream,
                        blocks(product(count as usize, work.hidden)?)?,
                        0,
                        &mut [
                            arg(&mut normalized),
                            arg(&mut bounds),
                            arg(&mut indices),
                            arg(&mut out),
                            arg(&mut count),
                            arg(&mut pool_width),
                            arg(&mut mode),
                        ],
                    )?;
                }
            }
            if !batch.raw_indices.is_empty() {
                crate::enqueue_raw_output(
                    &inner.raw,
                    &work.stream,
                    normalized,
                    data.selected.cu_deviceptr(),
                    layout.trailer().address(base)?,
                    batch,
                    work.hidden,
                )?;
            }
            let capacity = lease
                .slot
                .as_mut()
                .ok_or_else(|| ExperimentalMoeError::Admission("host lease missing".into()))?
                .output
                .as_mut_slice();
            if capacity.len() < selected_count {
                return Err(ExperimentalMoeError::Admission(
                    "selected host output exceeds lease capacity".into(),
                ));
            }
            crate::download_output_range(
                &work.stream,
                &data.selected,
                0,
                capacity,
                0,
                selected_count,
            )?;
            lease.record()?;
            Ok(selected_count)
        }
    }

    pub fn finish(
        &self,
        _work: &ExperimentalMoEWork,
        lease: &mut ExperimentalMoELease,
        selected_count: usize,
    ) -> Result<Vec<f32>, ExperimentalMoeError> {
        lease.wait()?;
        Ok(lease.output(selected_count)?.to_vec())
    }

    pub fn wait(&self, lease: &mut ExperimentalMoELease) -> Result<(), ExperimentalMoeError> {
        lease.wait().map_err(ExperimentalMoeError::Backend)
    }

    pub fn submit(
        &self,
        work: &mut ExperimentalMoEWork,
        batch: &Batch,
        lease: &mut ExperimentalMoELease,
        pool: Pool,
    ) -> Result<Vec<f32>, ExperimentalMoeError> {
        let selected = self.enqueue(work, batch, lease, pool)?;
        self.finish(work, lease, selected)
    }

    pub fn validate_batch(&self, batch: &Batch) -> Result<usize, ExperimentalMoeError> {
        let inner = self.lock()?;
        admit_healthy(
            inner.io.poisoned(),
            inner.poison.load(std::sync::atomic::Ordering::Acquire),
        )?;
        validate_moe_batch(batch, &inner.weights.config)
    }

    pub fn hidden_size(&self) -> Result<usize, ExperimentalMoeError> {
        Ok(self.lock()?.weights.config.text.hidden_size)
    }

    pub fn completion_proof(
        &self,
        lease: &ExperimentalMoELease,
    ) -> Option<crate::work_retirement::CompletedTicket> {
        lease
            .completion
            .as_ref()
            .and_then(|ticket| ticket.completed())
    }

    pub fn poison_after_failed_fence(&self) {
        if let Ok(inner) = self.inner.lock() {
            inner
                .poison
                .store(true, std::sync::atomic::Ordering::Release);
        }
    }

    pub fn health_check(&self) -> Result<(), ExperimentalMoeError> {
        let inner = self.lock()?;
        admit_healthy(
            inner.io.poisoned(),
            inner.poison.load(std::sync::atomic::Ordering::Acquire),
        )
    }
}

fn stage_moe_packed_input(
    host: &mut [u32],
    layout: &InputLayout,
    batch: &Batch,
) -> Result<(), ExperimentalMoeError> {
    if host.len() != layout.words() {
        return Err(ExperimentalMoeError::Admission(
            "packed host length differs from InputLayout.words".into(),
        ));
    }
    let (packed, trailer) = host
        .split_at_mut_checked(layout.packed_words())
        .ok_or_else(|| ExperimentalMoeError::Admission("packed/trailer split".into()))?;
    crate::pack_input_with_position_offset(
        packed,
        batch.input_ids.len(),
        &batch.input_ids,
        &batch.position_ids,
        &batch.token_type_ids,
        &batch.cumulative_seq_lengths,
        &batch.pooled_indices,
        0,
    )
    .map_err(|error| ExperimentalMoeError::Admission(error.to_string()))?;
    let raw_rows = crate::output::raw_rows(batch)
        .map_err(|error| ExperimentalMoeError::Admission(error.to_string()))?;
    if trailer.len() != raw_rows {
        return Err(ExperimentalMoeError::Admission(
            "raw trailer length differs from raw_rows".into(),
        ));
    }
    if raw_rows > 0 {
        crate::output::write_raw_indices(batch, trailer)
            .map_err(|error| ExperimentalMoeError::Admission(error.to_string()))?;
    }
    Ok(())
}

fn selected_capacity_elements(
    sequences: usize,
    tokens: usize,
    hidden: usize,
) -> Result<usize, ExperimentalMoeError> {
    product(
        sequences.checked_add(tokens).ok_or_else(|| {
            ExperimentalMoeError::Admission("selected output capacity overflow".into())
        })?,
        hidden,
    )
    .map_err(|error| ExperimentalMoeError::Admission(error.to_string()))
}

fn selected_output_elements(
    pooled_rows: usize,
    raw_rows: usize,
    hidden: usize,
) -> Result<usize, ExperimentalMoeError> {
    product(
        pooled_rows.checked_add(raw_rows).ok_or_else(|| {
            ExperimentalMoeError::Admission("selected output count overflow".into())
        })?,
        hidden,
    )
    .map_err(|error| ExperimentalMoeError::Admission(error.to_string()))
}

/// One owned request arena. The three BF16 attention slabs are sized by the
/// MoE attention-only layout; routed/shared FFN scratch is a distinct owner.
struct ExperimentalMoEWorkData {
    input: DeviceBuffer<u32>,
    residual: DeviceBuffer<Bf16>,
    projection: DeviceBuffer<Bf16>,
    activation: DeviceBuffer<Bf16>,
    scratch: Option<Qwen35MoeFullScratch>,
    output_f32: DeviceBuffer<f32>,
    selected: DeviceBuffer<f32>,
    gdn: Option<GdnPrepared>,
}

pub struct ExperimentalMoEWork {
    engine: Arc<()>,
    poison: Arc<std::sync::atomic::AtomicBool>,
    stream: Arc<Stream>,
    tokens: usize,
    sequences: usize,
    max_len: usize,
    hidden: usize,
    input_words: usize,
    bytes: usize,
    layout: Qwen35MoeAttentionLayout,
    data: ManuallyDrop<ExperimentalMoEWorkData>,
    last_completion: Option<CompletionTicket>,
    payload_released: bool,
}

#[allow(clippy::too_many_arguments)]
fn work_allocation_bytes(
    slab_bytes: usize,
    input_words: usize,
    tokens: usize,
    hidden: usize,
    selected_elements: usize,
    ffn_scratch_bytes: usize,
    gdn_device_bytes: usize,
) -> Result<usize, ExperimentalMoeError> {
    slab_bytes
        .checked_add(product(input_words, 4)?)
        .and_then(|value| value.checked_add(product(product(tokens, hidden).ok()?, 4).ok()?))
        .and_then(|value| value.checked_add(product(selected_elements, 4).ok()?))
        .and_then(|value| value.checked_add(ffn_scratch_bytes))
        .and_then(|value| value.checked_add(gdn_device_bytes))
        .ok_or_else(|| ExperimentalMoeError::Admission("Work byte count overflow".into()))
}

impl ExperimentalMoEWork {
    fn allocation_bytes(
        weights: &Qwen35MoeFullDeviceWeights,
        provider: &Qwen35MoeProvider,
        gdn_major: i32,
        gdn_sm_count: i32,
        tokens: usize,
        sequences: usize,
        max_len: usize,
    ) -> Result<usize, ExperimentalMoeError> {
        if tokens == 0 || sequences == 0 || max_len == 0 || max_len > tokens {
            return Err(ExperimentalMoeError::Admission(
                "invalid Work token/sequence/context geometry".into(),
            ));
        }
        let config = &weights.config;
        let layout = Qwen35MoeAttentionLayout::new(config, tokens)?;
        let input_words = product(tokens, 3)?
            .checked_add(sequences + 1)
            .and_then(|value| value.checked_add(tokens))
            .and_then(|value| value.checked_add(tokens))
            .ok_or_else(|| ExperimentalMoeError::Admission("input capacity overflow".into()))?;
        let hidden = config.text.hidden_size;
        let selected_elements = selected_capacity_elements(sequences, tokens, hidden)?;
        let first = weights
            .layers
            .iter()
            .find_map(|layer| match &layer.attention {
                crate::qwen35_weights::Qwen35DeviceAttention::Linear { weights, .. } => {
                    Some(Arc::clone(weights))
                }
                _ => None,
            })
            .ok_or_else(|| {
                ExperimentalMoeError::Admission(
                    "actual MoE geometry requires at least one GDN layer".into(),
                )
            })?;
        let gdn_bytes = GdnPrepared::scratch_bytes_default_dense(
            first.geometry(),
            GdnCapacity::new(tokens, sequences, sequences)?,
            gdn_major,
            gdn_sm_count,
        )?
        .device_bytes;
        let ffn_scratch_bytes = scratch_allocation_bytes(
            tokens,
            hidden,
            config.moe.moe_intermediate_size,
            config.moe.shared_expert_intermediate_size,
            product(config.moe.shared_expert_intermediate_size, 2)?,
            provider,
            config.moe.norm_topk_prob,
        )
        .map_err(ExperimentalMoeError::Backend)?;
        work_allocation_bytes(
            layout.slab_bytes,
            input_words,
            tokens,
            hidden,
            selected_elements,
            ffn_scratch_bytes,
            gdn_bytes,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new(
        engine: Arc<()>,
        poison: Arc<std::sync::atomic::AtomicBool>,
        context: &Arc<Context>,
        stream: &Arc<Stream>,
        weights: &Qwen35MoeFullDeviceWeights,
        provider: &Qwen35MoeProvider,
        options: &ExperimentalMoEOptions,
        gdn_major: i32,
        gdn_sm_count: i32,
        tokens: usize,
        sequences: usize,
        max_len: usize,
    ) -> Result<Self, ExperimentalMoeError> {
        if tokens == 0 || sequences == 0 || max_len == 0 || max_len > tokens {
            return Err(ExperimentalMoeError::Admission(
                "invalid Work token/sequence/context geometry".into(),
            ));
        }
        let config = &weights.config;
        let layout = Qwen35MoeAttentionLayout::new(config, tokens)?;
        let input_words = product(tokens, 3)?
            .checked_add(sequences + 1)
            .and_then(|value| value.checked_add(tokens))
            .and_then(|value| value.checked_add(tokens))
            .ok_or_else(|| ExperimentalMoeError::Admission("input capacity overflow".into()))?;
        let hidden = config.text.hidden_size;
        let bytes = Self::allocation_bytes(
            weights,
            provider,
            gdn_major,
            gdn_sm_count,
            tokens,
            sequences,
            max_len,
        )?;
        let input = DeviceBuffer::zeroed(stream, input_words)?;
        let residual = DeviceBuffer::zeroed(stream, layout.hidden_elements)?;
        let projection = DeviceBuffer::zeroed(stream, layout.projection_elements)?;
        let activation = DeviceBuffer::zeroed(stream, layout.activation_elements)?;
        let scratch = allocate_scratch(
            stream,
            tokens,
            hidden,
            config.moe.moe_intermediate_size,
            config.moe.shared_expert_intermediate_size,
            product(config.moe.shared_expert_intermediate_size, 2)?,
            provider,
            config.moe.norm_topk_prob,
        )?;
        let output_f32 = DeviceBuffer::zeroed(stream, product(tokens, hidden)?)?;
        let selected_elements = selected_capacity_elements(sequences, tokens, hidden)?;
        let selected = DeviceBuffer::zeroed(stream, selected_elements)?;
        let first = weights
            .layers
            .iter()
            .find_map(|layer| match &layer.attention {
                crate::qwen35_weights::Qwen35DeviceAttention::Linear { weights, .. } => {
                    Some(Arc::clone(weights))
                }
                _ => None,
            })
            .ok_or_else(|| {
                ExperimentalMoeError::Admission(
                    "actual MoE geometry requires at least one GDN layer".into(),
                )
            })?;
        let gdn_capacity = GdnCapacity::new(tokens, sequences, sequences)?;
        let kernels = GdnKernelSet::load_bundle(context, &options.gdn_raw_kernels)?;
        let gdn = GdnPrepared::new_default_dense_with_device_weights(
            stream,
            kernels,
            gdn_capacity,
            first,
            config.text.rms_norm_eps,
            &options.gdn_provider,
            &options.gdn_aux_kernels,
        )?;
        stream.synchronize().map_err(|error| {
            ExperimentalMoeError::Cold(format!("Work construction fence failed: {error}"))
        })?;
        Ok(Self {
            engine,
            poison,
            stream: Arc::clone(stream),
            tokens,
            sequences,
            max_len,
            hidden,
            input_words,
            bytes,
            layout,
            data: ManuallyDrop::new(ExperimentalMoEWorkData {
                input,
                residual,
                projection,
                activation,
                scratch: Some(scratch),
                output_f32,
                selected,
                gdn: Some(gdn),
            }),
            last_completion: None,
            payload_released: false,
        })
    }

    pub fn layout(&self) -> &Qwen35MoeAttentionLayout {
        &self.layout
    }
    pub fn tokens(&self) -> usize {
        self.tokens
    }
    pub fn hidden(&self) -> usize {
        self.hidden
    }
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    pub fn input_words(&self) -> usize {
        self.input_words
    }
    pub fn mark_last_use(&mut self, ticket: CompletionTicket) {
        self.last_completion = Some(ticket);
    }
    pub fn completion_ticket(&self) -> Option<CompletionTicket> {
        self.last_completion.clone()
    }
    /// Release the Work payload only when its last use has an actual proof.
    ///
    /// # Safety
    /// The caller must own the bound context and must not call this while any
    /// unproved use of this Work is still in flight.
    pub unsafe fn release_if_idle(&mut self) -> Result<bool, BackendError> {
        match self.last_completion.as_ref() {
            None => {
                self.release_payload();
                Ok(true)
            }
            Some(ticket) => match ticket.completed() {
                Some(proof) => {
                    // SAFETY: exact last-use proof was rechecked.
                    unsafe { self.release_after_completion(&proof)? };
                    Ok(true)
                }
                None => Ok(false),
            },
        }
    }
    pub fn validate_completion(&self, proof: &CompletedTicket) -> bool {
        self.last_completion.as_ref().is_some_and(|ticket| {
            proof.matches(ticket) && proof.stream_identity() == Arc::as_ptr(&self.stream) as usize
        })
    }
    /// Release this Work after an exact same-stream completion proof.
    ///
    /// # Safety
    /// `proof` must come from the actual fence for this Work's tracked ticket.
    pub unsafe fn release_after_completion(
        &mut self,
        proof: &CompletedTicket,
    ) -> Result<(), BackendError> {
        if !self.validate_completion(proof) {
            return Err(infer("completion proof does not match this MoE Work"));
        }
        if !self.payload_released {
            self.release_payload();
        }
        Ok(())
    }
    /// Release this Work after the caller drained the whole producing stream.
    ///
    /// # Safety
    /// The caller must have observed a successful whole-stream drain covering
    /// every use of this Work. A failed drain must not call this method.
    pub unsafe fn release_after_drain(&mut self) {
        if !self.payload_released {
            self.release_payload();
        }
    }
    fn release_payload(&mut self) {
        if self.payload_released {
            return;
        }
        self.payload_released = true;
        // SAFETY: every field is dropped exactly once here; no method may touch
        // `data` after this flag is set.
        unsafe { ManuallyDrop::drop(&mut self.data) };
    }
}

impl Drop for ExperimentalMoEWork {
    fn drop(&mut self) {
        if self.payload_released {
            return;
        }
        if self.stream.synchronize().is_ok() {
            self.payload_released = true;
            // SAFETY: the whole producing stream drained and this is the only
            // owner of the payload.
            unsafe { ManuallyDrop::drop(&mut self.data) };
        } else {
            // The ManuallyDrop payload is intentionally retained/quarantined;
            // ordinary field drop must not free device state on an unproven
            // stream.
            self.poison
                .store(true, std::sync::atomic::Ordering::Release);
            metrics::counter!("oxide_arena_retained_bytes_total").increment(self.bytes as u64);
        }
    }
}

fn check_fa4(
    root: &Path,
    actual: ExperimentalFa4Geometry,
) -> Result<ExperimentalFa4Check, ExperimentalMoeError> {
    let architecture = ["sm_90", "sm_100", "sm_120"]
        .iter()
        .find(|arch| root.join(arch).join("manifest.json").is_file())
        .copied()
        .ok_or_else(|| {
            ExperimentalMoeError::Cold(format!(
                "FA4 root {} has no sm_90/sm_100/sm_120 manifest",
                root.display()
            ))
        })?;
    let dir = root.join(architecture);
    let manifest: Value = serde_json::from_slice(
        &std::fs::read(dir.join("manifest.json"))
            .map_err(|error| ExperimentalMoeError::Cold(format!("read FA4 manifest: {error}")))?,
    )
    .map_err(|error| ExperimentalMoeError::Cold(format!("parse FA4 manifest: {error}")))?;
    let registry = manifest
        .get("registry")
        .ok_or_else(|| ExperimentalMoeError::Cold("FA4 manifest lacks registry metadata".into()))?;
    let namespace = registry
        .get("namespace")
        .and_then(Value::as_str)
        .ok_or_else(|| ExperimentalMoeError::Cold("FA4 registry lacks namespace".into()))?
        .to_owned();
    let mode_count = registry
        .get("mode_count")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| ExperimentalMoeError::Cold("FA4 registry lacks mode_count".into()))?;
    let appended = APPENDED_PREFIXES
        .iter()
        .find(|(count, name, _)| *count == mode_count && *name == namespace);
    let expected_sequence = if namespace == "canonical26" && mode_count == CANONICAL26_BASE_MODES {
        CANONICAL26_SEQUENCE
    } else if let Some((_, _, sequence)) = appended {
        *sequence
    } else {
        return Err(ExperimentalMoeError::Cold(format!(
            "expected canonical26/{CANONICAL26_BASE_MODES} or an exact canonical27..canonical33 prefix, found {namespace}/{mode_count}"
        )));
    };
    let fingerprint = registry
        .get("profile_sequence_sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            ExperimentalMoeError::Cold("FA4 registry lacks profile_sequence_sha256".into())
        })?
        .to_owned();
    if fingerprint != expected_sequence {
        return Err(ExperimentalMoeError::Cold(format!(
            "FA4 {namespace} profile_sequence_sha256 {fingerprint} != {expected_sequence}"
        )));
    }
    let kernels = manifest
        .get("kernels")
        .and_then(Value::as_array)
        .ok_or_else(|| ExperimentalMoeError::Cold("FA4 manifest lacks kernels".into()))?;
    let exact = kernels.iter().find(|kernel| actual.matches_kernel(kernel));
    let exact_symbol = exact
        .and_then(|kernel| kernel.get("symbol"))
        .and_then(Value::as_str)
        .map(str::to_owned);
    let dso_path = dir.join("libinfinityfa4.so");
    let exact_symbol_in_dso = exact_symbol
        .as_ref()
        .map(|symbol| dso_has_kernel(&dso_path, symbol))
        .transpose()?
        .unwrap_or(false);
    let proposal = if exact_symbol_in_dso {
        None
    } else if appended.is_some() {
        // Any appended source prefix must carry the exact manifest symbol and
        // its linked DSO export; a missing one is a real bundle failure, not
        // another append request.
        return Err(ExperimentalMoeError::Cold(match exact_symbol {
            Some(symbol) => {
                format!("{namespace} manifest declares {symbol} but its DSO lacks the symbol")
            }
            None => format!("{namespace} manifest lacks the exact {PROPOSED_SYMBOL} geometry"),
        }));
    } else {
        Some(ExperimentalFa4Proposal {
            namespace: namespace.clone(),
            base_mode_count: mode_count,
            append_mode_index: mode_count,
            append_mode_mask: 1u64
                .checked_shl(u32::try_from(mode_count).map_err(|_| {
                    ExperimentalMoeError::Cold("FA4 mode index does not fit u32".into())
                })?)
                .ok_or_else(|| ExperimentalMoeError::Cold("FA4 mode mask overflow".into()))?,
            symbol: PROPOSED_SYMBOL.into(),
            profile: "qwen35_moe_gqa8".into(),
            geometry: actual,
            borrowed_profile_id: None,
            borrowed_symbol: None,
            existing_modes_unchanged: true,
            new_dso_required: true,
            source_registry_append_required: true,
        })
    };
    Ok(ExperimentalFa4Check {
        root: root.to_path_buf(),
        architecture: architecture.into(),
        registry_namespace: namespace,
        mode_count,
        profile_sequence_sha256: fingerprint,
        dso_path,
        actual,
        exact_symbol,
        exact_symbol_in_dso,
        proposal,
    })
}

fn dso_has_kernel(path: &Path, symbol: &str) -> Result<bool, ExperimentalMoeError> {
    if dso_has_symbol(path, symbol)? {
        return Ok(true);
    }
    dso_has_symbol(path, &format!("__tvm_ffi_{symbol}"))
}

fn dso_has_symbol(path: &Path, needle: &str) -> Result<bool, ExperimentalMoeError> {
    let bytes = std::fs::read(path)
        .map_err(|error| ExperimentalMoeError::Cold(format!("read FA4 DSO: {error}")))?;
    if bytes.len() < 64 || &bytes[..4] != b"\x7fELF" || bytes[4] != 2 || bytes[5] != 1 {
        return Err(ExperimentalMoeError::Cold(
            "FA4 DSO is not a little-endian ELF64 image".into(),
        ));
    }
    let u16_at = |offset: usize| -> Option<u16> {
        bytes
            .get(offset..offset + 2)
            .map(|s| u16::from_le_bytes([s[0], s[1]]))
    };
    let u32_at = |offset: usize| -> Option<u32> {
        bytes
            .get(offset..offset + 4)
            .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    };
    let u64_at = |offset: usize| -> Option<u64> {
        bytes
            .get(offset..offset + 8)
            .map(|s| u64::from_le_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]))
    };
    let shoff = usize::try_from(u64_at(0x28).ok_or_else(|| {
        ExperimentalMoeError::Cold("ELF64 section header offset is missing".into())
    })?)
    .map_err(|_| ExperimentalMoeError::Cold("ELF64 section header offset overflow".into()))?;
    let shentsize = usize::from(
        u16_at(0x3a)
            .ok_or_else(|| ExperimentalMoeError::Cold("ELF64 section size missing".into()))?,
    );
    let shnum = usize::from(
        u16_at(0x3c)
            .ok_or_else(|| ExperimentalMoeError::Cold("ELF64 section count missing".into()))?,
    );
    if shentsize < 64 || shnum == 0 {
        return Err(ExperimentalMoeError::Cold(
            "ELF64 DSO has no usable section table".into(),
        ));
    }
    for index in 0..shnum {
        let offset =
            shoff
                .checked_add(index.checked_mul(shentsize).ok_or_else(|| {
                    ExperimentalMoeError::Cold("ELF64 section table overflow".into())
                })?)
                .ok_or_else(|| ExperimentalMoeError::Cold("ELF64 section table overflow".into()))?;
        let section_type = u32_at(offset + 4)
            .ok_or_else(|| ExperimentalMoeError::Cold("ELF64 section type missing".into()))?;
        if section_type != 11 {
            continue;
        }
        let section_offset =
            usize::try_from(u64_at(offset + 24).ok_or_else(|| {
                ExperimentalMoeError::Cold("ELF64 section offset missing".into())
            })?)
            .map_err(|_| ExperimentalMoeError::Cold("ELF64 section offset overflow".into()))?;
        let section_size = usize::try_from(
            u64_at(offset + 32)
                .ok_or_else(|| ExperimentalMoeError::Cold("ELF64 section size missing".into()))?,
        )
        .map_err(|_| ExperimentalMoeError::Cold("ELF64 section size overflow".into()))?;
        let link = usize::try_from(
            u32_at(offset + 40)
                .ok_or_else(|| ExperimentalMoeError::Cold("ELF64 section link missing".into()))?,
        )
        .map_err(|_| ExperimentalMoeError::Cold("ELF64 section link overflow".into()))?;
        let entry_size = usize::try_from(
            u64_at(offset + 56)
                .ok_or_else(|| ExperimentalMoeError::Cold("ELF64 entry size missing".into()))?,
        )
        .map_err(|_| ExperimentalMoeError::Cold("ELF64 entry size overflow".into()))?;
        if entry_size < 24 || link >= shnum {
            continue;
        }
        let string_header = shoff
            .checked_add(link.checked_mul(shentsize).ok_or_else(|| {
                ExperimentalMoeError::Cold("ELF64 string section overflow".into())
            })?)
            .ok_or_else(|| ExperimentalMoeError::Cold("ELF64 string section overflow".into()))?;
        let strings_offset = usize::try_from(u64_at(string_header + 24).ok_or_else(|| {
            ExperimentalMoeError::Cold("ELF64 string section offset missing".into())
        })?)
        .map_err(|_| ExperimentalMoeError::Cold("ELF64 string offset overflow".into()))?;
        let strings_size = usize::try_from(u64_at(string_header + 32).ok_or_else(|| {
            ExperimentalMoeError::Cold("ELF64 string section size missing".into())
        })?)
        .map_err(|_| ExperimentalMoeError::Cold("ELF64 string size overflow".into()))?;
        let strings = bytes
            .get(strings_offset..strings_offset.saturating_add(strings_size))
            .ok_or_else(|| ExperimentalMoeError::Cold("ELF64 string table out of range".into()))?;
        let entries = section_size / entry_size;
        for entry in 0..entries {
            let name_offset = u32_at(section_offset + entry * entry_size).ok_or_else(|| {
                ExperimentalMoeError::Cold("ELF64 symbol name offset missing".into())
            })? as usize;
            let Some(rest) = strings.get(name_offset..) else {
                continue;
            };
            let end = rest
                .iter()
                .position(|byte| *byte == 0)
                .unwrap_or(rest.len());
            if rest.get(..end) == Some(needle.as_bytes()) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CACHED_35B: &str =
        "/workspace/model-performance/michaelfeil/onejev-perf/models/qwen35-35b-a3b";
    const CANONICAL26: &str = "/tmp/oxide-fa4-canonical26-derived-v4/bundle";

    #[test]
    fn public_input_layout_matches_actual_packed_words() {
        let layout = ExperimentalMoEInputLayout::new(7, 3, 2)
            .unwrap()
            .with_trailer(4)
            .unwrap();
        assert_eq!(layout.packed_words(), 7 * 3 + 3 + 2);
        assert_eq!(layout.words(), 7 * 3 + 3 + 2 + 4);
        assert_eq!(layout.ids_range(), 0..7);
        assert_eq!(layout.positions_range(), 7..14);
        assert_eq!(layout.types_range(), 14..21);
    }

    #[test]
    fn moe_attention_layout_is_attention_only_and_never_dense_ffn_sized() {
        let config = Qwen35MoeConfig::from_json(&serde_json::json!({
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "vocab_size": 32,
            "layer_types": ["linear_attention", "full_attention"],
            "attn_output_gate": true,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512
        }))
        .unwrap();
        let layout = Qwen35MoeAttentionLayout::new(&config, 7).unwrap();
        assert!(!layout.has_dense_ffn_span());
        assert_eq!(layout.capacity_tokens_value(), 7);
        assert_eq!(layout.hidden_elements_value(), 7 * 16);
        assert_eq!(
            layout.slab_bytes(),
            (layout.hidden_elements_value()
                + layout.projection_elements_value()
                + layout.activation_elements_value())
                * 2
        );
    }

    #[test]
    fn actual_moe_attention_slab_is_exactly_the_attention_widths_only() {
        let model = Path::new(CACHED_35B);
        if !model.join("config.json").is_file() {
            return;
        }
        let config = Qwen35MoeConfig::load(model).unwrap();
        let layout = Qwen35MoeAttentionLayout::new(&config, 2048).unwrap();
        // Actual full attention projection Q16*2*256 + KV2*2*256 = 9216;
        // activation Q16*2*256 + KV2*256 = 8704. No 512-wide dense FFN span.
        assert_eq!(layout.hidden_elements_value(), 2048 * 2048);
        assert_eq!(layout.projection_elements_value(), 2048 * 9216);
        assert_eq!(layout.activation_elements_value(), 2048 * 8704);
        assert_eq!(layout.slab_bytes(), (2048 * (2048 + 9216 + 8704)) * 2);
        assert!(!layout.has_dense_ffn_span());
    }

    #[test]
    fn public_completion_clock_issues_and_proves_exact_ticket() {
        use crate::completion::{CompletionClock, RetirementQueue};
        let clock = Arc::new(CompletionClock::new());
        let ticket = clock.ticket(0x1000).unwrap();
        assert!(ticket.completed().is_none());
        let mut queue = RetirementQueue::new();
        queue.try_push(7u32, 4, Some(ticket.clone())).unwrap();
        assert!(queue.pop_completed().is_none());
        // SAFETY: the test supplies an actual completed same-stream proof.
        unsafe { ticket.mark_completed_after_fence() };
        let (value, proof) = queue.pop_completed().unwrap();
        assert_eq!(value, 7);
        assert!(proof.matches(&ticket));
    }

    #[test]
    fn canonical27_geometry_matches_only_the_appended_mode26_kernel() {
        let actual = ExperimentalFa4Geometry::actual_qwen35_moe();
        let appended = serde_json::json!({
            "symbol": PROPOSED_SYMBOL,
            "heads": 16,
            "kv_heads": 2,
            "d": 256,
            "causal": true,
            "window": [null, null],
            "dtype": "bf16",
        });
        assert!(actual.matches_kernel(&appended));
        assert_eq!(actual.symbol(), PROPOSED_SYMBOL);
        let old_qwen35 = serde_json::json!({
            "symbol": "fa4_qwen35_bf16",
            "heads": 16,
            "kv_heads": 4,
            "d": 256,
            "causal": false,
            "window": [null, null],
            "dtype": "bf16",
        });
        assert!(!actual.matches_kernel(&old_qwen35));
        for (symbol, heads, kv_heads) in [
            ("fa4_causal_wide4_bf16", 16, 4),
            ("fa4_causal_gqa8_bf16", 32, 4),
            ("fa4_gemma4_local256_causal_bf16", 8, 1),
        ] {
            let wrong_mode = serde_json::json!({
                "symbol": symbol,
                "heads": heads,
                "kv_heads": kv_heads,
                "d": 256,
                "causal": true,
                "window": [null, null],
                "dtype": "bf16",
            });
            assert!(!actual.matches_kernel(&wrong_mode), "{symbol}");
        }
    }

    #[test]
    fn canonical27_declared_symbol_without_dso_export_is_a_cold_error() {
        let fa4 = Path::new(CANONICAL26);
        let source_manifest = fa4.join("sm_90/manifest.json");
        let source_dso = fa4.join("sm_90/libinfinityfa4.so");
        if !source_manifest.is_file() || !source_dso.is_file() {
            return;
        }
        let root = std::env::temp_dir().join(format!(
            "qwen35-moe-v6-canonical27-check-{}",
            std::process::id()
        ));
        let dir = root.join("sm_90");
        std::fs::create_dir_all(&dir).unwrap();
        let mut manifest: Value =
            serde_json::from_slice(&std::fs::read(&source_manifest).unwrap()).unwrap();
        manifest["registry"] = serde_json::json!({
            "namespace": "canonical27",
            "mode_count": 27,
            "mode_mask": "0x7ffffff",
            "profile_sequence_sha256": APPENDED_PREFIXES[0].2,
        });
        manifest["kernels"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({
                "symbol": PROPOSED_SYMBOL,
                "heads": 16,
                "kv_heads": 2,
                "d": 256,
                "causal": true,
                "window": [null, null],
                "dtype": "bf16",
            }));
        std::fs::write(
            dir.join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        std::fs::copy(&source_dso, dir.join("libinfinityfa4.so")).unwrap();
        let error = check_fa4(&root, ExperimentalFa4Geometry::actual_qwen35_moe()).unwrap_err();
        assert!(
            error.to_string().contains("DSO lacks the symbol"),
            "{error}"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn dso_scanner_reads_real_canonical26_symbols_and_not_the_absent_profile() {
        let path = Path::new(CANONICAL26)
            .join("sm_90")
            .join("libinfinityfa4.so");
        if !path.is_file() {
            return;
        }
        assert!(dso_has_kernel(&path, "fa4_qwen35_bf16").unwrap());
        assert!(!dso_has_kernel(&path, PROPOSED_SYMBOL).unwrap());
    }

    #[test]
    fn engine_load_returns_the_typed_gap_before_any_gpu_work() {
        let model = Path::new(CACHED_35B);
        let fa4 = Path::new(CANONICAL26);
        if !model.join("model.safetensors.index.json").is_file()
            || !fa4.join("sm_90/manifest.json").is_file()
        {
            return;
        }
        let options = ExperimentalMoEOptions {
            fa4_root: fa4.to_path_buf(),
            kernel_module: PathBuf::from("/definitely/missing/kernels.cubin"),
            moe_aot_root: PathBuf::from("/definitely/missing/moe"),
            gdn_raw_kernels: PathBuf::from("/definitely/missing/gdn"),
            gdn_provider: PathBuf::from("/definitely/missing/provider"),
            gdn_aux_kernels: PathBuf::from("/definitely/missing/aux"),
        };
        match ExperimentalMoEEngine::load(model, 0, options) {
            Err(ExperimentalMoeError::MissingFa4Profile(proposal)) => {
                assert_eq!(proposal.append_mode_index, 26);
                assert_eq!(proposal.borrowed_profile_id, None);
            }
            Ok(_) => panic!("expected typed FA4 gap, got an engine"),
            Err(other) => panic!("expected typed FA4 gap, got {other}"),
        }
    }

    #[test]
    fn actual_cold_plan_reports_typed_append_only_fa4_gap() {
        let model = Path::new(CACHED_35B);
        let fa4 = Path::new(CANONICAL26);
        if !model.join("model.safetensors.index.json").is_file()
            || !fa4.join("sm_90/manifest.json").is_file()
        {
            if std::env::var_os("QWEN35_MOE_REQUIRE_CACHED").is_some() {
                panic!("actual Qwen3.5-MoE checkpoint and canonical26 bundle are required");
            }
            return;
        }
        let plan = ExperimentalMoeColdPlan::validate(model, fa4).unwrap();
        assert_eq!(plan.tensor_count, 1811);
        assert_eq!(plan.num_hidden_layers, 40);
        assert_eq!(plan.full_attention_layers, 10);
        assert_eq!(plan.linear_attention_layers, 30);
        assert_eq!(plan.num_experts, 256);
        assert_eq!(plan.top_k, 8);
        assert_eq!(plan.moe_intermediate_size, 512);
        assert_eq!(plan.shared_expert_intermediate_size, 512);
        assert_eq!(plan.fa4.registry_namespace, "canonical26");
        assert_eq!(plan.fa4.mode_count, 26);
        assert!(!plan.fa4.admitted());
        assert!(plan.fa4.exact_symbol.is_none());
        let proposal = plan.fa4.typed_gap().expect("actual geometry is absent");
        assert_eq!(proposal.append_mode_index, 26);
        assert_eq!(proposal.append_mode_mask, 1 << 26);
        assert_eq!(proposal.symbol, PROPOSED_SYMBOL);
        assert_eq!(proposal.symbol, proposal.geometry.symbol());
        assert_eq!(proposal.borrowed_profile_id, None);
        assert_eq!(proposal.borrowed_symbol, None);
        assert!(proposal.existing_modes_unchanged);
        assert!(proposal.new_dso_required);
        assert_eq!(
            proposal.geometry,
            ExperimentalFa4Geometry::actual_qwen35_moe()
        );
    }
    #[test]
    fn pooled_only_result_is_smaller_than_device_selected_capacity() {
        let capacity = selected_capacity_elements(2, 7, 16).unwrap();
        let pooled_only = selected_output_elements(2, 0, 16).unwrap();
        assert_eq!(capacity, (2 + 7) * 16);
        assert_eq!(pooled_only, 2 * 16);
        assert!(pooled_only < capacity);
    }

    fn raw_batch(tokens: usize, raw: Vec<u32>, pooled: Vec<u32>) -> Batch {
        Batch {
            multimodal: Vec::new(),
            input_ids: (0..tokens as u32).collect(),
            token_type_ids: vec![0; tokens],
            position_ids: (0..tokens as u32).collect(),
            cumulative_seq_lengths: vec![0, tokens as u32],
            max_length: tokens as u32,
            pooled_indices: pooled,
            raw_indices: raw,
            compact_input_ids: None,
            compact_position_ids: None,
            scatter_unfold: None,
            fold_gather: None,
            tokens: Vec::new(),
            offsets: Vec::new(),
        }
    }

    #[test]
    fn production_packing_helper_accepts_raw_duplicate_and_mixed_selectors() {
        for (pooled, raw, raw_rows) in [
            (Vec::new(), vec![0u32], 3usize),
            (Vec::new(), vec![0u32, 0u32], 6usize),
            (vec![0u32], vec![0u32, 0u32], 6usize),
        ] {
            let batch = raw_batch(3, raw.clone(), pooled.clone());
            let layout = InputLayout::new(3, 2, pooled.len())
                .unwrap()
                .with_trailer(raw_rows)
                .unwrap();
            let mut host = vec![0u32; layout.words()];
            stage_moe_packed_input(&mut host, &layout, &batch).unwrap();
            let (_, trailer) = host.split_at(layout.packed_words());
            let mut expected = Vec::new();
            for index in &raw {
                let _ = index;
                expected.extend_from_slice(&[0, 1, 2]);
            }
            assert_eq!(trailer, expected);
        }
    }

    #[test]
    fn production_packing_helper_rejects_trailerless_raw_layout() {
        let batch = raw_batch(3, vec![0], Vec::new());
        let bad = InputLayout::new(3, 2, 0).unwrap();
        let mut host = vec![0u32; bad.words()];
        assert!(stage_moe_packed_input(&mut host, &bad, &batch).is_err());
    }
    #[test]
    fn failed_work_fence_poison_is_admitted_as_unhealthy() {
        assert!(!engine_unhealthy(false, false));
        assert!(engine_unhealthy(true, false));
        assert!(engine_unhealthy(false, true));
        assert!(engine_unhealthy(true, true));
    }
    #[test]
    fn fence_latch_attempts_exactly_once_after_first_failure() {
        let mut failed = false;
        let mut attempts = 0;
        let first: Result<(), BackendError> = fence_once(&mut failed, || {
            attempts += 1;
            Err(BackendError::Unhealthy)
        });
        assert!(first.is_err());
        let second: Result<(), BackendError> = fence_once(&mut failed, || {
            attempts += 1;
            Ok(())
        });
        assert!(second.is_err());
        assert_eq!(attempts, 1);
        assert!(failed);
    }
    #[test]
    fn io_or_engine_poison_maps_to_typed_unhealthy() {
        for (io, work) in [(true, false), (false, true), (true, true)] {
            let error = admit_healthy(io, work).unwrap_err();
            assert!(matches!(
                error,
                ExperimentalMoeError::Backend(BackendError::Unhealthy)
            ));
        }
        assert!(admit_healthy(false, false).is_ok());
        assert!(matches!(
            prepare_healthy(true, false),
            Err(BackendError::Unhealthy)
        ));
        assert!(matches!(
            prepare_healthy(false, true),
            Err(BackendError::Unhealthy)
        ));
        assert!(prepare_healthy(false, false).is_ok());
    }
    #[test]
    fn work_byte_plan_includes_exact_gdn_and_ffn_scratch_against_independent_formula() {
        let tokens = 7usize;
        let hidden = 2048usize;
        let selected = 8usize;
        let slab = 1000usize;
        let input_words = 27usize;
        let workspace = 12_345usize;
        let ffn = crate::qwen35_moe_full::scratch_allocation_bytes_for_shape(
            tokens, hidden, 512, 512, 1024, workspace,
        )
        .unwrap();
        let ffn_expected = tokens * 1024 * 2
            + tokens * 512 * 2
            + tokens * hidden * 2
            + tokens * 2
            + tokens * 256 * 2
            + tokens * 256 * 4
            + workspace
            + tokens * hidden * 2;
        assert_eq!(ffn, ffn_expected);

        let geometry = crate::gdn_prepared::GdnGeometry::new(16, 48, 4).unwrap();
        let capacity = GdnCapacity::new(tokens, 1, 1).unwrap();
        let gdn = GdnPrepared::scratch_bytes_default_dense(geometry, capacity, 9, 132)
            .unwrap()
            .device_bytes;
        assert!(gdn > 0);
        let total =
            work_allocation_bytes(slab, input_words, tokens, hidden, selected, ffn, gdn).unwrap();
        let independent = slab + input_words * 4 + tokens * hidden * 4 + selected * 4 + ffn + gdn;
        assert_eq!(total, independent);
        assert!(total > slab + input_words * 4 + tokens * hidden * 4 + selected * 4 + ffn);
    }
}
