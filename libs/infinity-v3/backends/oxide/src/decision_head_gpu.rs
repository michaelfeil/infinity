// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Prepared GPU OneJev option-token readout.
//!
//! Native Qwen OneJev resolves the lexical `lm_head.weight` as BF16
//! `[vocab_size, hidden_size]`, widens only the answer rows and the sequence's
//! last real hidden row to FP32, and calls `at::linear` in FP32
//! (`decision_models.cpp`). This module keeps that contract on the device:
//!
//! * the canonical BF16 vocabulary matrix is uploaded exactly once during cold
//!   load and is never read from a checkpoint again;
//! * the selected answer rows and the hidden row are gathered and widened by
//!   the existing shared `infinity_bf16_to_f32` device entry;
//! * one prepared F32 SGEMM per request projects `[1, hidden]` against
//!   `[count, hidden]`, preserving request order and duplicate option ids;
//! * only the small F32 logit result is copied back to the host.
//!
//! The head does not create a CUDA context, stream, cuBLAS handle, workspace
//! or graph. It uses the caller's existing context, stream, Blas handle and
//! uploaded/source infrastructure. Device allocation happens only in cold
//! `prepare`; `enqueue` performs no device allocation, plan construction or
//! heuristic. It registers one cuda-core read lease on the hidden allocation so
//! the source owner and its cross-stream ordering are retained through proven
//! completion.

use crate::{
    cublas::Blas,
    decision_head::{
        output_count_for, plan_decision_requests, requests_are_last_token_only, DecisionHeadError,
        OneJevHead, PackedDecisionBatch, RequestPlan,
    },
    math::{GemmArguments, LinearPlan},
    tensor::{Bf16, Layout, F32},
    upload::WeightUploader,
};
use cuda_core::{
    AllocationLease, Context, DeviceBuffer, DriverError, Function, PinnedHostBuffer, Stream,
};
use infinity_v3_backend_core::{DecisionInput, DecisionOutput};
use std::{ffi::c_void, fmt, mem::ManuallyDrop, path::Path, sync::Arc};

const CAST_ELEMENTS_PER_BLOCK: usize = 256;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum DecisionGpuError {
    Head(DecisionHeadError),
    ContextMismatch(&'static str),
    StreamMismatch(&'static str),
    KernelModule(String),
    Driver(String),
    KernelLaunch(String),
    Geometry(&'static str),
    Hidden(&'static str),
    HiddenBusy(String),
    PreparedMismatch(&'static str),
    PreparedInUse(&'static str),
    OutputAllocation,
}

impl fmt::Display for DecisionGpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Head(error) => error.fmt(f),
            Self::ContextMismatch(what) => write!(f, "{what} belongs to another CUDA context"),
            Self::StreamMismatch(what) => write!(f, "{what} does not use the prepared stream"),
            Self::KernelModule(error) => write!(f, "OneJev readout kernel module failed: {error}"),
            Self::Driver(error) => write!(f, "OneJev readout driver operation failed: {error}"),
            Self::KernelLaunch(error) => write!(f, "OneJev readout kernel launch failed: {error}"),
            Self::Geometry(error) => write!(f, "invalid OneJev GPU readout geometry: {error}"),
            Self::Hidden(error) => write!(f, "invalid OneJev GPU hidden states: {error}"),
            Self::HiddenBusy(error) => {
                write!(
                    f,
                    "hidden OneJev source is not ready for the prepared stream: {error}"
                )
            }
            Self::PreparedMismatch(error) => {
                write!(f, "prepared OneJev GPU readout mismatch: {error}")
            }
            Self::PreparedInUse(error) => {
                write!(f, "prepared OneJev GPU readout is not reusable: {error}")
            }
            Self::OutputAllocation => f.write_str("OneJev GPU logits allocation failed"),
        }
    }
}

impl std::error::Error for DecisionGpuError {}

impl From<DecisionHeadError> for DecisionGpuError {
    fn from(error: DecisionHeadError) -> Self {
        Self::Head(error)
    }
}

impl From<DriverError> for DecisionGpuError {
    fn from(error: DriverError) -> Self {
        Self::Driver(error.to_string())
    }
}

impl From<infinity_v3_backend_core::BackendError> for DecisionGpuError {
    fn from(error: infinity_v3_backend_core::BackendError) -> Self {
        Self::Driver(error.to_string())
    }
}

fn driver(what: &'static str, error: DriverError) -> DecisionGpuError {
    DecisionGpuError::Driver(format!("{what}: {error}"))
}

/// The existing shared-module entry needed by the prepared head.
///
/// Loading a function from the already shared oxide module is not a new
/// context, stream, cuBLAS handle or workspace. The module path is the same
/// `INFINITY_OXIDE_KERNEL_MODULE` artifact used by the rest of the backend.
pub(crate) struct DecisionReadoutKernels {
    cast: Function,
}

impl DecisionReadoutKernels {
    pub(crate) fn cast_kernel(&self) -> &Function {
        &self.cast
    }
}

impl DecisionReadoutKernels {
    pub(crate) fn load(context: &Arc<Context>) -> Result<Self, DecisionGpuError> {
        let path = std::env::var_os(infinity_v3_config::infinity_env!("OXIDE_KERNEL_MODULE"))
            .ok_or_else(|| {
                DecisionGpuError::KernelModule(
                    "set INFINITY_OXIDE_KERNEL_MODULE to the oxide kernel artifact".into(),
                )
            })?;
        let path = std::path::PathBuf::from(path);
        let path = if path.is_dir() {
            let (major, minor) = context
                .compute_capability()
                .map_err(|error| driver("compute capability", error))?;
            path.join(format!("sm_{major}{minor}"))
                .join("kernels.cubin")
        } else {
            path
        };
        let data = std::fs::read(&path).map_err(|error| {
            DecisionGpuError::KernelModule(format!("{}: {error}", path.display()))
        })?;
        let module = context
            .load_module_from_image(&data)
            .map_err(|error| driver("load oxide kernel module", error))?;
        let cast = module
            .load_function("infinity_bf16_to_f32")
            .map_err(|error| driver("load infinity_bf16_to_f32", error))?;
        Ok(Self { cast })
    }
}

/// Borrowed packed BF16 hidden states. The head consumes only selected rows.
///
/// Safe `forward`/`forward_pooled` acquire a cuda-core read lease on the
/// prepared stream and keep it in the prepared state until a proven drain.
/// Safe producers must use cuda-core mutation APIs so the access tracker can
/// reject a read while a tracked cross-stream write is still active. An unsafe
/// raw producer's ordering is the engine-bound caller's outer Work obligation.
#[derive(Clone, Copy)]
pub(crate) struct GpuHiddenRows<'a> {
    buffer: &'a DeviceBuffer<Bf16>,
    rows: usize,
    width: usize,
}

impl<'a> GpuHiddenRows<'a> {
    pub(crate) fn new(
        buffer: &'a DeviceBuffer<Bf16>,
        rows: usize,
        width: usize,
    ) -> Result<Self, DecisionGpuError> {
        if rows == 0 {
            return Err(DecisionGpuError::Hidden(
                "at least one hidden row is required",
            ));
        }
        if width == 0 {
            return Err(DecisionGpuError::Hidden("hidden width must be positive"));
        }
        let elements = rows
            .checked_mul(width)
            .ok_or(DecisionGpuError::Hidden("hidden element count overflow"))?;
        if buffer.len() < elements {
            return Err(DecisionGpuError::Hidden(
                "hidden storage is shorter than its logical rows",
            ));
        }
        Ok(Self {
            buffer,
            rows,
            width,
        })
    }

    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    pub(crate) fn width(&self) -> usize {
        self.width
    }

    pub(crate) fn buffer(&self) -> &'a DeviceBuffer<Bf16> {
        self.buffer
    }
}

/// One cold-prepared F32 projection. `arguments` already contains the exact
/// per-request buffer offsets and cuBLAS dimensions.
#[derive(Debug)]
pub(crate) struct DecisionProjection {
    arguments: GemmArguments,
    count: usize,
}

impl DecisionProjection {
    pub(crate) fn arguments(&self) -> &GemmArguments {
        &self.arguments
    }

    pub(crate) fn count(&self) -> usize {
        self.count
    }
}

struct GpuPreparedOwners {
    hidden: DeviceBuffer<f32>,
    selected: DeviceBuffer<f32>,
    row_ids: DeviceBuffer<u32>,
    last_rows: DeviceBuffer<u32>,
    logits: DeviceBuffer<f32>,
    host_logits: PinnedHostBuffer<f32>,
}

impl GpuPreparedOwners {
    fn bytes(&self) -> usize {
        self.hidden
            .num_bytes()
            .saturating_add(self.selected.num_bytes())
            .saturating_add(self.row_ids.num_bytes())
            .saturating_add(self.last_rows.num_bytes())
            .saturating_add(self.logits.num_bytes())
            .saturating_add(self.host_logits.num_bytes())
    }
}

/// A completion-retained read lease. Production uses cuda-core's real
/// `AllocationLease`; CPU tests use a counting fake because no CUDA context is
/// permitted in the CPU gate.
pub(crate) trait RetainedLease: Send + 'static {
    fn release(self);
}

impl RetainedLease for AllocationLease {
    fn release(self) {
        drop(self);
    }
}

/// Owned hidden-source retention. The lease holds the allocation and its
/// access-tracker entry, so the caller may drop its `DeviceBuffer` as soon as
/// `forward`/`forward_pooled` returns, including on every error path.
pub(crate) struct HiddenLeaseSlot<L: RetainedLease> {
    lease: Option<L>,
    bytes: usize,
}

impl<L: RetainedLease> HiddenLeaseSlot<L> {
    pub(crate) fn empty() -> Self {
        Self {
            lease: None,
            bytes: 0,
        }
    }

    pub(crate) fn is_held(&self) -> bool {
        self.lease.is_some()
    }

    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }

    pub(crate) fn install(&mut self, lease: L, bytes: usize) -> Result<(), &'static str> {
        if self.lease.is_some() {
            lease.release();
            return Err("hidden source lease is already held");
        }
        self.lease = Some(lease);
        self.bytes = bytes;
        Ok(())
    }

    /// The prepared stream has drained successfully, so the allocation and its
    /// read access may be released.
    pub(crate) fn release_after_drain(&mut self) {
        if let Some(lease) = self.lease.take() {
            lease.release();
        }
        self.bytes = 0;
    }

    /// Completion is uncertain: leak the lease (which owns the allocation)
    /// instead of freeing storage that a failed stream may still read.
    pub(crate) fn retain_after_uncertain_drain(&mut self) {
        if let Some(lease) = self.lease.take() {
            std::mem::forget(lease);
        }
    }

    pub(crate) fn drop_with_drain(&mut self, drained: bool) {
        if drained {
            self.release_after_drain();
        } else {
            self.retain_after_uncertain_drain();
        }
    }
}

/// Request-owned prepared GPU readout.
///
/// The selected rows, hidden-row selection, logits and pinned readback host
/// buffer are allocated once here. `enqueue` only launches the prebuilt work.
pub(crate) struct GpuPreparedDecision {
    head_id: u64,
    hidden: usize,
    vocab_rows: usize,
    boundaries: Vec<u32>,
    counts: Vec<usize>,
    row_offsets: Vec<usize>,
    row_ids: Vec<u32>,
    last_rows: Vec<u32>,
    total: usize,
    projections: Vec<DecisionProjection>,
    stream: Arc<Stream>,
    owners: ManuallyDrop<GpuPreparedOwners>,
    hidden_lease: HiddenLeaseSlot<AllocationLease>,
    pending: bool,
    drained: bool,
    poisoned: bool,
}

impl GpuPreparedDecision {
    pub(crate) fn output_counts(&self) -> &[usize] {
        &self.counts
    }

    pub(crate) fn selected_rows(&self) -> usize {
        self.total
    }

    pub(crate) fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    pub(crate) fn projections(&self) -> &[DecisionProjection] {
        &self.projections
    }

    fn retained_bytes(&self) -> usize {
        self.owners
            .bytes()
            .saturating_add(self.hidden_lease.bytes())
    }
}

impl Drop for GpuPreparedDecision {
    fn drop(&mut self) {
        let drained = self.drained || self.stream.synchronize().is_ok();
        if drained {
            // SAFETY: no stream work can reference these owners any more.
            unsafe { ManuallyDrop::drop(&mut self.owners) };
        } else {
            // Leave device and pinned-host storage allocated, retain the hidden
            // read lease, and retain one stream reference, because completion
            // is uncertain.
            crate::record_uncharged_retained_bytes(self.retained_bytes());
            std::mem::forget(self.stream.clone());
        }
        self.hidden_lease.drop_with_drain(drained);
    }
}

/// Cold OneJev GPU head owning the canonical BF16 vocabulary upload.
pub(crate) struct OneJevGpuHead {
    id: u64,
    weight: ManuallyDrop<DeviceBuffer<Bf16>>,
    kernels: ManuallyDrop<DecisionReadoutKernels>,
    context: Arc<Context>,
    stream: Arc<Stream>,
    hidden: usize,
    vocab_rows: usize,
    block_attention: bool,
    namespace: &'static str,
}

impl OneJevGpuHead {
    /// Validate cold metadata and upload the canonical BF16 head once.
    ///
    /// The checkpoint payload is read only here. Later forwards use the device
    /// copy and never reopen or re-read checkpoint tensors.
    pub(crate) fn load(
        path: &Path,
        context: &Arc<Context>,
        stream: &Arc<Stream>,
    ) -> Result<Self, DecisionGpuError> {
        if !Arc::ptr_eq(context, stream.context()) {
            return Err(DecisionGpuError::ContextMismatch(
                "OneJev stream and context",
            ));
        }
        context
            .bind_to_thread()
            .map_err(|error| driver("bind OneJev load context", error))?;
        let metadata = OneJevHead::load(path)?;
        let id = metadata.id();
        let hidden = metadata.hidden_size();
        let vocab_rows = metadata.vocab_rows();
        let block_attention = metadata.block_attention();
        let namespace = metadata.namespace();
        let source = metadata.into_weight_source();
        let kernels = DecisionReadoutKernels::load(context)?;
        let mut uploader = WeightUploader::new(stream.clone())?;
        let weight = uploader.upload(&[&source])?;
        uploader.finish()?;
        Ok(Self {
            id,
            weight: ManuallyDrop::new(weight),
            kernels: ManuallyDrop::new(kernels),
            context: context.clone(),
            stream: stream.clone(),
            hidden,
            vocab_rows,
            block_attention,
            namespace,
        })
    }

    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    pub(crate) fn hidden_size(&self) -> usize {
        self.hidden
    }

    pub(crate) fn vocab_rows(&self) -> usize {
        self.vocab_rows
    }

    pub(crate) fn block_attention(&self) -> bool {
        self.block_attention
    }

    pub(crate) fn namespace(&self) -> &'static str {
        self.namespace
    }

    /// Canonical BF16 `[vocab_rows, hidden]` upload. The engine-bound warm
    /// readout borrows this owner; it never creates a second vocabulary copy.
    pub(crate) fn weight(&self) -> &DeviceBuffer<Bf16> {
        &self.weight
    }

    /// Shared `infinity_bf16_to_f32` entry loaded from the existing oxide
    /// module. Borrowed so the warm readout adds no module or context.
    pub(crate) fn cast_kernel(&self) -> &Function {
        &self.kernels.cast
    }

    pub(crate) fn last_token_only(
        &self,
        batch: &PackedDecisionBatch<'_>,
        requests: &[DecisionInput],
    ) -> bool {
        requests_are_last_token_only(batch, requests)
    }

    /// Allocate and plan the whole request batch before the backbone runs.
    ///
    /// This performs all device allocations and builds every F32 projection
    /// descriptor. It does not run a heuristic, query cuBLAS, or read a
    /// checkpoint tensor.
    pub(crate) fn prepare(
        &self,
        batch: &PackedDecisionBatch<'_>,
        requests: &[DecisionInput],
    ) -> Result<GpuPreparedDecision, DecisionGpuError> {
        let plan = plan_decision_requests(self.vocab_rows, batch, requests)?;
        let requests = plan.counts.len();
        if requests == 0 || plan.total == 0 {
            return Err(DecisionGpuError::Geometry(
                "OneJev request plan must reserve at least one logit",
            ));
        }
        let hidden_elements = requests
            .checked_mul(self.hidden)
            .ok_or(DecisionGpuError::Geometry("hidden buffer size overflow"))?;
        let selected_elements =
            plan.total
                .checked_mul(self.hidden)
                .ok_or(DecisionGpuError::Geometry(
                    "selected row buffer size overflow",
                ))?;
        for value in [requests, plan.total, self.hidden] {
            u32::try_from(value)
                .map_err(|_| DecisionGpuError::Geometry("OneJev dimension exceeds u32"))?;
        }
        cast_grid(hidden_elements)?;
        cast_grid(selected_elements)?;
        // SAFETY: every element is written by the gather kernel before the
        // projection or readback that consumes it.
        let hidden =
            unsafe { DeviceBuffer::<f32>::uninitialized_async(&self.stream, hidden_elements) }
                .map_err(|error| driver("allocate OneJev hidden rows", error))?;
        // SAFETY: every selected row element is written before projection.
        let selected =
            unsafe { DeviceBuffer::<f32>::uninitialized_async(&self.stream, selected_elements) }
                .map_err(|error| driver("allocate OneJev selected rows", error))?;
        let row_ids = DeviceBuffer::<u32>::from_host(&self.stream, &plan.row_ids)
            .map_err(|error| driver("upload OneJev answer row ids", error))?;
        let last_rows = DeviceBuffer::<u32>::from_host(&self.stream, &plan.last_rows)
            .map_err(|error| driver("upload OneJev last-token row ids", error))?;
        // SAFETY: each request projection writes its whole logical logit range.
        let logits = unsafe { DeviceBuffer::<f32>::uninitialized_async(&self.stream, plan.total) }
            .map_err(|error| driver("allocate OneJev logits", error))?;
        let host_logits = PinnedHostBuffer::<f32>::zeroed(&self.context, plan.total)
            .map_err(|error| driver("allocate OneJev pinned logits", error))?;
        let projections = build_projections(
            self.hidden,
            &plan,
            hidden.cu_deviceptr(),
            selected.cu_deviceptr(),
            logits.cu_deviceptr(),
        )?;
        Ok(GpuPreparedDecision {
            head_id: self.id,
            hidden: self.hidden,
            vocab_rows: self.vocab_rows,
            boundaries: plan.boundaries,
            counts: plan.counts,
            row_offsets: plan.row_offsets,
            row_ids: plan.row_ids,
            last_rows: plan.last_rows,
            total: plan.total,
            projections,
            stream: self.stream.clone(),
            owners: ManuallyDrop::new(GpuPreparedOwners {
                hidden,
                selected,
                row_ids,
                last_rows,
                logits,
                host_logits,
            }),
            hidden_lease: HiddenLeaseSlot::empty(),
            pending: false,
            drained: true,
            poisoned: false,
        })
    }

    /// Enqueue hidden/selected-row gather, every F32 projection and the single
    /// logit readback. No device allocation, heuristic or plan construction
    /// occurs here; the only new bookkeeping is one cuda-core read lease.
    ///
    /// The read lease is registered on `stream` through cuda-core's allocation
    /// access tracker. A tracked write on another stream that has not completed
    /// is rejected with `CUDA_ERROR_NOT_READY` before any launch; completed
    /// entries are purged by their completion event. Raw unsafe producers are
    /// outside this safe contract, so the engine-bound caller's outer Work
    /// owner must order them on this stream or prove completion.
    ///
    /// # Safety
    /// The caller retains the hidden source, the prepared owners and the head
    /// upload until this stream drains. Safe `forward`/`forward_pooled` acquire
    /// the read lease themselves and retain it in the prepared state.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn enqueue(
        &self,
        hidden: GpuHiddenRows<'_>,
        batch: &PackedDecisionBatch<'_>,
        requests: &[DecisionInput],
        prepared: &mut GpuPreparedDecision,
        blas: &Blas,
        stream: &Stream,
        pooled: bool,
    ) -> Result<(), DecisionGpuError> {
        self.validate_prepared(batch, requests, prepared, pooled)?;
        if prepared.pending || prepared.poisoned {
            return Err(DecisionGpuError::PreparedInUse(
                "drain or recover the previous submission first",
            ));
        }
        if !Arc::ptr_eq(stream.context(), &self.context)
            || !Arc::ptr_eq(blas.context(), &self.context)
        {
            return Err(DecisionGpuError::ContextMismatch(
                "OneJev hidden source, stream or cuBLAS handle",
            ));
        }
        if !std::ptr::eq(stream, self.stream.as_ref())
            || !std::ptr::eq(blas.stream(), self.stream.as_ref())
        {
            return Err(DecisionGpuError::StreamMismatch(
                "OneJev stream or cuBLAS handle",
            ));
        }
        if !Arc::ptr_eq(hidden.buffer.context(), &self.context) {
            return Err(DecisionGpuError::ContextMismatch("OneJev hidden source"));
        }
        let expected_rows = if pooled { batch.len() } else { batch.tokens() };
        if hidden.rows() != expected_rows {
            return Err(DecisionGpuError::Hidden(
                "hidden row count does not match the packed or pooled request shape",
            ));
        }
        if hidden.width() != self.hidden {
            return Err(DecisionGpuError::Hidden(
                "hidden width does not match the OneJev readout",
            ));
        }
        if prepared.hidden_lease.is_held() {
            return Err(DecisionGpuError::PreparedInUse(
                "hidden source lease is still held",
            ));
        }
        // Actual cuda-core access mechanism: register a read on the prepared
        // stream. A tracked write on another stream that has not completed is
        // rejected with CUDA_ERROR_NOT_READY before any launch; completed
        // entries are purged by their completion event. Raw unsafe producers
        // are outside this safe contract: the engine-bound caller's outer Work
        // owner must order them (same stream or completed).
        let lease = hidden
            .buffer
            .storage()
            .acquire(stream, false, false)
            .map_err(|error| {
                DecisionGpuError::HiddenBusy(format!(
                    "hidden source is not ordered after its producer: {error}"
                ))
            })?;
        prepared
            .hidden_lease
            .install(lease, hidden.buffer.num_bytes())
            .map_err(DecisionGpuError::PreparedInUse)?;
        prepared.poisoned = true;
        prepared.drained = false;
        prepared.pending = false;
        let request_count = prepared.counts.len();
        let total = prepared.total;
        let hidden_width = self.hidden;
        let selection = if pooled {
            0
        } else {
            prepared.owners.last_rows.cu_deviceptr()
        };
        launch_cast(
            &self.kernels.cast,
            stream,
            hidden.buffer,
            &prepared.owners.hidden,
            selection,
            request_count,
            hidden_width,
        )?;
        launch_cast(
            &self.kernels.cast,
            stream,
            &self.weight,
            &prepared.owners.selected,
            prepared.owners.row_ids.cu_deviceptr(),
            total,
            hidden_width,
        )?;
        for projection in &prepared.projections {
            // SAFETY: the prepared owner contract covers the F32 buffers and
            // the caller keeps them alive through stream completion.
            if let Err(error) = unsafe { blas.enqueue_f32_projection(projection.arguments()) } {
                return Err(DecisionGpuError::Driver(format!(
                    "F32 decision projection: {error}"
                )));
            }
        }
        let device_logits = prepared.owners.logits.cu_deviceptr();
        let host_logits = prepared.owners.host_logits.as_mut_ptr();
        let bytes = total
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(DecisionGpuError::Geometry("logit readback size overflow"))?;
        // SAFETY: the destination pinned buffer and source device buffer remain
        // owned by `prepared`; both are retained until `finish` or a drain.
        if let Err(error) = unsafe {
            cuda_core::memory::memcpy_dtoh_async(
                host_logits,
                device_logits,
                bytes,
                stream.cu_stream(),
            )
        } {
            return Err(driver("enqueue OneJev logit readback", error));
        }
        prepared.pending = true;
        prepared.poisoned = false;
        Ok(())
    }

    /// Wait for the prepared submission and materialize the small host logits.
    pub(crate) fn finish(
        &self,
        prepared: &mut GpuPreparedDecision,
    ) -> Result<Vec<DecisionOutput>, DecisionGpuError> {
        if !prepared.pending {
            return Err(DecisionGpuError::PreparedMismatch(
                "one submission has not been enqueued",
            ));
        }
        if let Err(error) = prepared.stream.synchronize() {
            prepared.poisoned = true;
            return Err(driver("drain OneJev submission", error));
        }
        prepared.pending = false;
        prepared.drained = true;
        prepared.poisoned = false;
        prepared.hidden_lease.release_after_drain();
        let host = prepared.owners.host_logits.as_slice();
        let mut results = Vec::new();
        results
            .try_reserve_exact(prepared.counts.len())
            .map_err(|_| DecisionGpuError::OutputAllocation)?;
        for (index, count) in prepared.counts.iter().enumerate() {
            let begin = prepared.row_offsets.get(index).copied().ok_or(
                DecisionGpuError::PreparedMismatch("missing prepared row offset"),
            )?;
            let end = begin
                .checked_add(*count)
                .ok_or(DecisionGpuError::Geometry("logit range overflow"))?;
            let values = host
                .get(begin..end)
                .ok_or(DecisionGpuError::PreparedMismatch(
                    "prepared logits are incomplete",
                ))?;
            let mut logits = Vec::new();
            logits
                .try_reserve_exact(values.len())
                .map_err(|_| DecisionGpuError::OutputAllocation)?;
            logits.extend_from_slice(values);
            results.push(DecisionOutput {
                logits,
                action_probability: 1.0,
                memory: None,
            });
        }
        Ok(results)
    }

    /// Enqueue and complete one full packed last-real-token readout.
    ///
    /// The hidden source is retained by an owned cuda-core read lease from
    /// before the first launch until a proven drain. Every partial enqueue and
    /// failed-drain path therefore remains safe even if the caller drops its
    /// `DeviceBuffer` immediately after this returns.
    pub(crate) fn forward(
        &self,
        hidden: GpuHiddenRows<'_>,
        batch: &PackedDecisionBatch<'_>,
        requests: &[DecisionInput],
        prepared: &mut GpuPreparedDecision,
        blas: &Blas,
        stream: &Stream,
    ) -> Result<Vec<DecisionOutput>, DecisionGpuError> {
        // SAFETY: this method owns the complete enqueue/finish lifetime.
        unsafe { self.enqueue(hidden, batch, requests, prepared, blas, stream, false)? };
        self.finish(prepared)
    }

    /// Enqueue and complete one pooled one-row-per-request readout.
    ///
    /// Same owned hidden read-lease retention as `forward`.
    pub(crate) fn forward_pooled(
        &self,
        hidden: GpuHiddenRows<'_>,
        batch: &PackedDecisionBatch<'_>,
        requests: &[DecisionInput],
        prepared: &mut GpuPreparedDecision,
        blas: &Blas,
        stream: &Stream,
    ) -> Result<Vec<DecisionOutput>, DecisionGpuError> {
        if !requests_are_last_token_only(batch, requests) {
            return Err(DecisionGpuError::Head(
                DecisionHeadError::UnsupportedRequest(
                    "pooled decisions require last-token-only requests",
                ),
            ));
        }
        // SAFETY: this method owns the complete enqueue/finish lifetime.
        unsafe { self.enqueue(hidden, batch, requests, prepared, blas, stream, true)? };
        self.finish(prepared)
    }

    /// Re-establish a safe stream after a failed enqueue/drain.
    ///
    /// Only a successful drain makes the prepared owners reusable and
    /// releasable again. A failed drain leaves them retained.
    pub(crate) fn recover_after_failed_drain(
        &self,
        prepared: &mut GpuPreparedDecision,
    ) -> Result<(), DecisionGpuError> {
        if !prepared.poisoned && !prepared.pending {
            return Ok(());
        }
        prepared
            .stream
            .synchronize()
            .map_err(|error| DecisionGpuError::Driver(format!("recover OneJev stream: {error}")))?;
        prepared.pending = false;
        prepared.drained = true;
        prepared.poisoned = false;
        prepared.hidden_lease.release_after_drain();
        Ok(())
    }

    fn validate_prepared(
        &self,
        batch: &PackedDecisionBatch<'_>,
        requests: &[DecisionInput],
        prepared: &GpuPreparedDecision,
        pooled: bool,
    ) -> Result<(), DecisionGpuError> {
        if prepared.head_id != self.id
            || prepared.hidden != self.hidden
            || prepared.vocab_rows != self.vocab_rows
        {
            return Err(DecisionGpuError::PreparedMismatch(
                "prepared metadata belongs to another head",
            ));
        }
        if prepared.boundaries.as_slice() != batch.boundaries() {
            return Err(DecisionGpuError::PreparedMismatch(
                "prepared boundaries belong to another packed batch",
            ));
        }
        if prepared.counts.len() != requests.len()
            || prepared.row_offsets.len() != requests.len().saturating_add(1)
            || prepared.projections.len() != requests.len()
        {
            return Err(DecisionGpuError::PreparedMismatch(
                "prepared request count does not match the decision batch",
            ));
        }
        for (index, request) in requests.iter().enumerate() {
            let expected = output_count_for(self.vocab_rows, index, request)?;
            if prepared.counts.get(index) != Some(&expected) {
                return Err(DecisionGpuError::PreparedMismatch(
                    "prepared output count does not match the request",
                ));
            }
            let begin = prepared.row_offsets.get(index).copied().ok_or(
                DecisionGpuError::PreparedMismatch("missing prepared row offset"),
            )?;
            let end = prepared
                .row_offsets
                .get(index.saturating_add(1))
                .copied()
                .ok_or(DecisionGpuError::PreparedMismatch(
                    "missing prepared row offset",
                ))?;
            if end.checked_sub(begin) != Some(expected) {
                return Err(DecisionGpuError::PreparedMismatch(
                    "prepared row range does not match the output count",
                ));
            }
            let ids =
                prepared
                    .row_ids
                    .get(begin..end)
                    .ok_or(DecisionGpuError::PreparedMismatch(
                        "prepared answer row ids are incomplete",
                    ))?;
            let same_ids = match request {
                DecisionInput::Warmup => ids == [0],
                DecisionInput::OptionTokens { token_ids }
                | DecisionInput::BlockOptionTokens { token_ids, .. } => ids == token_ids.as_slice(),
                DecisionInput::InlineOptions { .. }
                | DecisionInput::Laya { .. }
                | DecisionInput::Clef { .. } => false,
            };
            if !same_ids {
                return Err(DecisionGpuError::PreparedMismatch(
                    "prepared answer rows do not match the request",
                ));
            }
            let (_, sequence_end) = batch.sequence(index)?;
            let last = sequence_end
                .checked_sub(1)
                .and_then(|row| u32::try_from(row).ok())
                .ok_or(DecisionGpuError::PreparedMismatch(
                    "prepared last-token row is invalid",
                ))?;
            if prepared.last_rows.get(index) != Some(&last) {
                return Err(DecisionGpuError::PreparedMismatch(
                    "prepared last-token row does not match the packed batch",
                ));
            }
            if prepared
                .projections
                .get(index)
                .map(DecisionProjection::count)
                != Some(expected)
            {
                return Err(DecisionGpuError::PreparedMismatch(
                    "prepared projection width does not match the request",
                ));
            }
        }
        if pooled && !requests_are_last_token_only(batch, requests) {
            return Err(DecisionGpuError::Head(
                DecisionHeadError::UnsupportedRequest(
                    "pooled decisions require last-token-only requests",
                ),
            ));
        }
        Ok(())
    }
}

impl Drop for OneJevGpuHead {
    fn drop(&mut self) {
        if self.stream.synchronize().is_ok() {
            // SAFETY: the upload and shared module are no longer referenced by
            // stream work.
            unsafe {
                ManuallyDrop::drop(&mut self.weight);
                ManuallyDrop::drop(&mut self.kernels);
            }
        } else {
            crate::record_uncharged_retained_bytes(self.weight.num_bytes());
            std::mem::forget(self.stream.clone());
        }
    }
}

fn build_projections(
    hidden: usize,
    plan: &RequestPlan,
    hidden_base: u64,
    selected_base: u64,
    logits_base: u64,
) -> Result<Vec<DecisionProjection>, DecisionGpuError> {
    if plan.row_offsets.len() != plan.counts.len().saturating_add(1) {
        return Err(DecisionGpuError::Geometry(
            "request row offsets do not match the request count",
        ));
    }
    if plan.row_offsets.last().copied() != Some(plan.total)
        || plan.row_ids.len() != plan.total
        || plan.last_rows.len() != plan.counts.len()
    {
        return Err(DecisionGpuError::Geometry(
            "request plan row counts are inconsistent",
        ));
    }
    let mut projections = Vec::new();
    projections
        .try_reserve_exact(plan.counts.len())
        .map_err(|_| DecisionGpuError::OutputAllocation)?;
    for (index, (&count, &row_offset)) in
        plan.counts.iter().zip(plan.row_offsets.iter()).enumerate()
    {
        if count == 0 {
            return Err(DecisionGpuError::Geometry(
                "a prepared projection cannot have zero options",
            ));
        }
        let x = element_pointer(hidden_base, index, hidden)?;
        let weight = element_pointer(selected_base, row_offset, hidden)?;
        let output = element_pointer(logits_base, row_offset, 1)?;
        let plan = LinearPlan::<F32>::new(
            Layout::contiguous([1, hidden])
                .map_err(|_| DecisionGpuError::Geometry("invalid hidden layout"))?,
            Layout::contiguous([count, hidden])
                .map_err(|_| DecisionGpuError::Geometry("invalid weight layout"))?,
            Layout::contiguous([1, count])
                .map_err(|_| DecisionGpuError::Geometry("invalid output layout"))?,
        )
        .map_err(|_| DecisionGpuError::Geometry("invalid F32 projection geometry"))?;
        if plan.input_layout().shape() != &[1, hidden]
            || plan.weight_layout().shape() != &[count, hidden]
            || plan.output_layout().shape() != &[1, count]
        {
            return Err(DecisionGpuError::Geometry(
                "prepared F32 projection shape differs from its request",
            ));
        }
        projections.push(DecisionProjection {
            arguments: plan.arguments(x, weight, output),
            count,
        });
    }
    Ok(projections)
}

fn element_pointer(base: u64, row: usize, stride: usize) -> Result<u64, DecisionGpuError> {
    let elements = row
        .checked_mul(stride)
        .ok_or(DecisionGpuError::Geometry("device element offset overflow"))?;
    let bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(DecisionGpuError::Geometry("device byte offset overflow"))?;
    base.checked_add(bytes)
        .ok_or(DecisionGpuError::Geometry("device address overflow"))
}

fn cast_grid(elements: usize) -> Result<u32, DecisionGpuError> {
    u32::try_from(elements.div_ceil(CAST_ELEMENTS_PER_BLOCK))
        .map_err(|_| DecisionGpuError::Geometry("BF16 widen grid exceeds u32"))
}

fn arg<T>(value: &mut T) -> *mut c_void {
    (value as *mut T).cast()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_cast(
    kernel: &Function,
    stream: &Stream,
    input: &DeviceBuffer<Bf16>,
    output: &DeviceBuffer<f32>,
    selected: u64,
    count: usize,
    width: usize,
) -> Result<(), DecisionGpuError> {
    if count == 0 || width == 0 {
        return Err(DecisionGpuError::Geometry(
            "BF16 widen requires nonzero rows and width",
        ));
    }
    let elements = count.checked_mul(width).ok_or(DecisionGpuError::Geometry(
        "BF16 widen element count overflow",
    ))?;
    if output.len() < elements {
        return Err(DecisionGpuError::Geometry(
            "BF16 widen output is shorter than its logical rows",
        ));
    }
    let grid = cast_grid(elements)?;
    let mut input = input.cu_deviceptr();
    let mut output = output.cu_deviceptr();
    let mut selected = selected;
    let mut count = u32::try_from(count)
        .map_err(|_| DecisionGpuError::Geometry("BF16 widen rows exceed u32"))?;
    let mut width = u32::try_from(width)
        .map_err(|_| DecisionGpuError::Geometry("BF16 widen width exceeds u32"))?;
    // SAFETY: fixed kernel ABI, checked element count and output capacity, and
    // the caller retains every buffer until this stream drains.
    unsafe {
        cuda_core::launch_kernel(
            kernel.cu_function(),
            (grid, 1, 1),
            (CAST_ELEMENTS_PER_BLOCK as u32, 1, 1),
            0,
            stream.cu_stream(),
            &mut [
                arg(&mut input),
                arg(&mut output),
                arg(&mut selected),
                arg(&mut count),
                arg(&mut width),
            ],
        )
    }
    .map_err(|error| DecisionGpuError::KernelLaunch(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cublas::{F32_PROJECTION_ALGORITHM, F32_PROJECTION_COMPUTE};
    use crate::decision_head::plan_decision_requests;
    use crate::decision_head::PackedDecisionBatch;
    use crate::decision_head::RequestPlan;
    use infinity_v3_backend_core::DecisionInput;

    const HIDDEN: usize = 4;
    const ROWS: usize = 8;

    fn option(token_ids: &[u32]) -> DecisionInput {
        DecisionInput::OptionTokens {
            token_ids: token_ids.to_vec(),
        }
    }

    #[test]
    fn f32_projection_uses_native_sgemm_not_bf16_tensor_op() {
        assert_eq!(F32_PROJECTION_COMPUTE, 68);
        assert_eq!(F32_PROJECTION_ALGORITHM, -1);
        assert_ne!(F32_PROJECTION_ALGORITHM, 99);
    }

    #[test]
    fn heterogeneous_request_plans_preserve_order_and_duplicates() {
        let batch = PackedDecisionBatch::new(&[0, 3, 5]).unwrap();
        let requests = [option(&[4, 1]), DecisionInput::Warmup];
        let plan = plan_decision_requests(ROWS, &batch, &requests).unwrap();
        assert_eq!(plan.counts, vec![2, 1]);
        assert_eq!(plan.row_offsets, vec![0, 2, 3]);
        assert_eq!(plan.row_ids, vec![4, 1, 0]);
        assert_eq!(plan.last_rows, vec![2, 4]);
        assert_eq!(plan.boundaries, vec![0, 3, 5]);
        assert_eq!(plan.total, 3);

        let duplicates = [option(&[5, 0, 5])];
        let batch = PackedDecisionBatch::new(&[0, 3]).unwrap();
        let plan = plan_decision_requests(ROWS, &batch, &duplicates).unwrap();
        assert_eq!(plan.row_ids, vec![5, 0, 5]);
        assert_eq!(plan.counts, vec![3]);
    }

    #[test]
    fn protocol_sized_batch_keeps_per_request_boundaries() {
        let batch = PackedDecisionBatch::new(&[0, 4, 8, 10]).unwrap();
        let first = (0..255).collect::<Vec<u32>>();
        let second = (0..255).collect::<Vec<u32>>();
        let requests = [option(&first), option(&second), option(&[0, 1])];
        let plan = plan_decision_requests(256, &batch, &requests).unwrap();
        assert_eq!(plan.counts, vec![255, 255, 2]);
        assert_eq!(plan.row_offsets, vec![0, 255, 510, 512]);
        assert_eq!(plan.total, 512);
        assert_eq!(plan.row_ids.len(), 512);
        assert_eq!(plan.last_rows, vec![3, 7, 9]);
        assert_eq!(plan.row_ids.get(254), Some(&254));
        assert_eq!(plan.row_ids.get(255), Some(&0));
        assert_eq!(plan.row_ids.get(509), Some(&254));
        assert_eq!(plan.row_ids.get(510), Some(&0));
        assert_eq!(plan.row_ids.get(511), Some(&1));

        let projections = build_projections(HIDDEN, &plan, 0x1000, 0x2000, 0x3000).unwrap();
        assert_eq!(projections.len(), 3);
        assert_eq!(projections[2].arguments().width, 2);
        assert_eq!(
            projections[2].arguments().weight,
            0x2000 + 510 * HIDDEN as u64 * 4
        );
        assert_eq!(projections[2].arguments().output, 0x3000 + 510 * 4);
    }

    #[test]
    fn projection_arguments_are_per_request_and_f32() {
        let batch = PackedDecisionBatch::new(&[0, 2, 3, 5]).unwrap();
        let requests = [option(&[4, 1]), option(&[2]), option(&[7, 6, 5])];
        let plan = plan_decision_requests(ROWS, &batch, &requests).unwrap();
        let (hidden_base, selected_base, logits_base) = (0x1000u64, 0x2000u64, 0x3000u64);
        let projections =
            build_projections(HIDDEN, &plan, hidden_base, selected_base, logits_base).unwrap();
        assert_eq!(projections.len(), 3);
        assert_eq!(projections[0].count(), 2);
        assert_eq!(projections[1].count(), 1);
        assert_eq!(projections[2].count(), 3);
        for (index, projection) in projections.iter().enumerate() {
            let arguments = projection.arguments();
            assert_eq!(arguments.cuda_type, 0);
            assert_eq!(arguments.rows, 1);
            assert_eq!(arguments.input, HIDDEN as i32);
            assert_eq!(arguments.width, plan.counts[index] as i32);
            assert_eq!(arguments.x_stride, HIDDEN as i32);
            assert_eq!(arguments.weight_stride, HIDDEN as i32);
            assert_eq!(arguments.output_stride, plan.counts[index] as i32);
            assert_eq!(
                arguments.x,
                hidden_base + (index * HIDDEN * std::mem::size_of::<f32>()) as u64
            );
            assert_eq!(
                arguments.weight,
                selected_base
                    + (plan.row_offsets[index] * HIDDEN * std::mem::size_of::<f32>()) as u64
            );
            assert_eq!(
                arguments.output,
                logits_base + (plan.row_offsets[index] * std::mem::size_of::<f32>()) as u64
            );
        }
    }

    #[test]
    fn malformed_request_plan_geometry_is_rejected() {
        let plan = RequestPlan {
            counts: vec![1],
            row_offsets: vec![0],
            row_ids: vec![0],
            boundaries: vec![0, 1],
            last_rows: vec![0],
            total: 1,
        };
        assert!(matches!(
            build_projections(HIDDEN, &plan, 0x1000, 0x2000, 0x3000),
            Err(DecisionGpuError::Geometry(_))
        ));

        let plan = RequestPlan {
            counts: vec![0],
            row_offsets: vec![0, 0],
            row_ids: vec![],
            boundaries: vec![0, 1],
            last_rows: vec![0],
            total: 0,
        };
        assert!(matches!(
            build_projections(HIDDEN, &plan, 0x1000, 0x2000, 0x3000),
            Err(DecisionGpuError::Geometry(_))
        ));
    }

    #[test]
    fn pooled_eligibility_rejects_warmup_and_other_heads() {
        let batch = PackedDecisionBatch::new(&[0, 1, 2]).unwrap();
        assert!(requests_are_last_token_only(
            &batch,
            &[option(&[0]), option(&[1])]
        ));
        assert!(!requests_are_last_token_only(
            &batch,
            &[option(&[0]), DecisionInput::Warmup]
        ));
        assert!(!requests_are_last_token_only(
            &batch,
            &[DecisionInput::InlineOptions {
                markers: vec![0],
                option_counts: vec![1],
            }]
        ));
    }

    fn assert_readout_matches(actual: &[DecisionOutput], expected: &[DecisionOutput]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(actual.logits.len(), expected.logits.len());
            assert_eq!(actual.action_probability, 1.0);
            assert!(actual.memory.is_none());
            for (option, (actual, expected)) in
                actual.logits.iter().zip(&expected.logits).enumerate()
            {
                let scale = expected.abs().max(1.0);
                assert!(
                    (actual - expected).abs() <= 5e-3 * scale,
                    "request {index} option {option}: gpu={actual}, cpu={expected}"
                );
            }
        }
    }

    #[test]
    #[ignore = "Requires CUDA 0, INFINITY_OXIDE_DECISION_MODEL and INFINITY_OXIDE_KERNEL_MODULE"]
    fn gpu_prepared_head_matches_cpu_selected_row_reference() {
        use crate::decision_head::HiddenRows;
        let model = std::env::var_os(infinity_v3_config::infinity_env!("OXIDE_DECISION_MODEL"))
            .expect("set INFINITY_OXIDE_DECISION_MODEL to a OneJev checkpoint");
        let model = std::path::PathBuf::from(model);
        assert!(
            Context::device_count().unwrap() > 0,
            "GPU control must execute"
        );
        let context = Context::new(0).unwrap();
        let stream = context.new_stream().unwrap();
        let blas = Blas::new(context.clone(), stream.clone()).unwrap();
        let gpu = OneJevGpuHead::load(&model, &context, &stream).unwrap();
        let cpu = OneJevHead::load(&model).unwrap();
        let hidden = gpu.hidden_size();
        assert_eq!(gpu.hidden_size(), cpu.hidden_size());

        let batch = PackedDecisionBatch::new(&[0, 3, 5]).unwrap();
        let requests = [option(&[32, 33, 32]), DecisionInput::Warmup];
        let bits = (0..batch.tokens() * hidden)
            .map(|index| {
                let value = ((index % 113) as f32 - 56.0) * 0.017;
                half::bf16::from_f32(value).to_bits()
            })
            .collect::<Vec<_>>();
        let device = bits
            .iter()
            .map(|&bits| Bf16::from_bits(bits))
            .collect::<Vec<_>>();
        let hidden_buffer = DeviceBuffer::from_host(&stream, &device).unwrap();
        let hidden_rows = GpuHiddenRows::new(&hidden_buffer, batch.tokens(), hidden).unwrap();
        let mut prepared = gpu.prepare(&batch, &requests).unwrap();
        let cpu_prepared = cpu.prepare(&batch, &requests).unwrap();
        let actual = gpu
            .forward(
                hidden_rows,
                &batch,
                &requests,
                &mut prepared,
                &blas,
                &stream,
            )
            .unwrap();
        let expected = cpu
            .forward(
                HiddenRows::from_bf16(&bits, hidden).unwrap(),
                &batch,
                &requests,
                &cpu_prepared,
            )
            .unwrap();
        assert_readout_matches(&actual, &expected);

        let pooled_batch = PackedDecisionBatch::new(&[0, 2, 4]).unwrap();
        let pooled_requests = [option(&[32]), option(&[33, 34])];
        let pooled_bits = [1usize, 3]
            .into_iter()
            .flat_map(|row| bits[row * hidden..(row + 1) * hidden].iter().copied())
            .collect::<Vec<_>>();
        let pooled_device = pooled_bits
            .iter()
            .map(|&bits| Bf16::from_bits(bits))
            .collect::<Vec<_>>();
        let pooled_buffer = DeviceBuffer::from_host(&stream, &pooled_device).unwrap();
        let pooled_rows = GpuHiddenRows::new(&pooled_buffer, 2, hidden).unwrap();
        let mut pooled_prepared = gpu.prepare(&pooled_batch, &pooled_requests).unwrap();
        let cpu_pooled_prepared = cpu.prepare(&pooled_batch, &pooled_requests).unwrap();
        let actual_pooled = gpu
            .forward_pooled(
                pooled_rows,
                &pooled_batch,
                &pooled_requests,
                &mut pooled_prepared,
                &blas,
                &stream,
            )
            .unwrap();
        let expected_pooled = cpu
            .forward_pooled(
                HiddenRows::from_bf16(&pooled_bits, hidden).unwrap(),
                &pooled_batch,
                &pooled_requests,
                &cpu_pooled_prepared,
            )
            .unwrap();
        assert_readout_matches(&actual_pooled, &expected_pooled);
    }

    struct FakeLease {
        drops: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl RetainedLease for FakeLease {
        fn release(self) {
            self.drops.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }

    fn fake_lease(drops: &std::sync::Arc<std::sync::atomic::AtomicUsize>) -> FakeLease {
        FakeLease {
            drops: drops.clone(),
        }
    }

    fn drop_count(drops: &std::sync::Arc<std::sync::atomic::AtomicUsize>) -> usize {
        drops.load(std::sync::atomic::Ordering::SeqCst)
    }

    #[test]
    fn production_hidden_lease_uses_real_cuda_allocation_lease() {
        fn assert_retained<T: RetainedLease>() {}
        assert_retained::<AllocationLease>();
    }

    #[test]
    fn real_allocation_lease_orders_cross_stream_read_after_write() {
        // Real cuda-core ownership/access types with deliberately non-null fake
        // stream handles. acquire/register/lease-drop never touch the driver;
        // Stream::drop does, so the borrowed fake streams are forgotten.
        let context = unsafe { Context::borrow_raw(std::ptr::null_mut(), 0, 0) };
        let stream_a = unsafe { Stream::borrow_raw(std::ptr::without_provenance_mut(1), &context) };
        let stream_b = unsafe { Stream::borrow_raw(std::ptr::without_provenance_mut(2), &context) };
        let raw = unsafe {
            cuda_core::RawDeviceBuffer::borrowed_from_raw_parts(0x1000, 4096, context.clone())
        };
        let write = raw.acquire(&stream_a, true, false).unwrap();
        let error = raw.acquire(&stream_b, false, false).unwrap_err();
        let text = format!("{error}").to_ascii_uppercase();
        assert!(
            text.contains("NOT READY") || text.contains("NOT_READY"),
            "{error}"
        );
        drop(write);
        let read = raw.acquire(&stream_b, false, false).unwrap();
        drop(read);
        drop(raw);
        std::mem::forget(stream_a);
        std::mem::forget(stream_b);
        drop(context);
    }

    #[test]
    fn hidden_lease_is_held_after_partial_enqueue_and_released_after_recovery() {
        let drops = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut slot = HiddenLeaseSlot::empty();
        slot.install(fake_lease(&drops), 4096).unwrap();
        assert!(slot.is_held());
        assert_eq!(slot.bytes(), 4096);
        // A partial enqueue error leaves the lease installed; it is not
        // released just because the error path returned.
        assert_eq!(drop_count(&drops), 0);
        slot.release_after_drain();
        assert!(!slot.is_held());
        assert_eq!(slot.bytes(), 0);
        assert_eq!(drop_count(&drops), 1);
    }

    #[test]
    fn hidden_lease_is_leaked_when_drain_completion_is_uncertain() {
        let drops = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut slot = HiddenLeaseSlot::empty();
        slot.install(fake_lease(&drops), 8192).unwrap();
        slot.drop_with_drain(false);
        assert!(!slot.is_held());
        assert_eq!(slot.bytes(), 8192);
        assert_eq!(drop_count(&drops), 0);
    }

    #[test]
    fn hidden_lease_is_released_only_after_proven_drain_drop() {
        let drops = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut slot = HiddenLeaseSlot::empty();
        slot.install(fake_lease(&drops), 2048).unwrap();
        slot.drop_with_drain(true);
        assert!(!slot.is_held());
        assert_eq!(slot.bytes(), 0);
        assert_eq!(drop_count(&drops), 1);
    }

    #[test]
    fn hidden_lease_rejects_a_second_install_without_losing_the_first() {
        let drops = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut slot = HiddenLeaseSlot::empty();
        slot.install(fake_lease(&drops), 1).unwrap();
        assert!(slot.install(fake_lease(&drops), 2).is_err());
        assert_eq!(drop_count(&drops), 1);
        assert_eq!(slot.bytes(), 1);
        slot.release_after_drain();
        assert_eq!(drop_count(&drops), 2);
    }

    #[test]
    fn hidden_lease_can_be_reinstalled_after_a_proven_drain() {
        let drops = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut slot = HiddenLeaseSlot::empty();
        slot.install(fake_lease(&drops), 1).unwrap();
        slot.release_after_drain();
        slot.install(fake_lease(&drops), 3).unwrap();
        slot.release_after_drain();
        assert_eq!(drop_count(&drops), 2);
    }

    #[test]
    fn hidden_busy_error_is_typed() {
        let error = DecisionGpuError::HiddenBusy("CUDA_ERROR_NOT_READY".into());
        assert!(format!("{error}").contains("not ready"));
    }

    #[test]
    fn request_plan_rejects_typed_metadata_errors_before_device_use() {
        let batch = PackedDecisionBatch::new(&[0, 2, 4]).unwrap();
        assert!(matches!(
            plan_decision_requests(ROWS, &batch, &[option(&[]), option(&[1])]),
            Err(DecisionHeadError::EmptyOptions)
        ));
        assert!(matches!(
            plan_decision_requests(ROWS, &batch, &[option(&[ROWS as u32]), option(&[1])]),
            Err(DecisionHeadError::TokenIdOutOfRange { .. })
        ));
        assert!(matches!(
            plan_decision_requests(
                ROWS,
                &batch,
                &[
                    DecisionInput::BlockOptionTokens {
                        token_ids: vec![1],
                        prefix_length: 0,
                    },
                    option(&[1])
                ]
            ),
            Err(DecisionHeadError::PrefixOutOfRange { .. })
        ));
        assert!(matches!(
            plan_decision_requests(ROWS, &batch, &[option(&[1])]),
            Err(DecisionHeadError::InvalidBatch(_))
        ));
    }
}
