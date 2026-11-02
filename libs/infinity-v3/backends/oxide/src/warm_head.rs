// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Engine-bound warm OneJev readout.
//!
//! This is the private integration leaf between an already prepared Qwen3.5
//! `Work` and the reviewed standalone `OneJevGpuHead`. It deliberately keeps
//! every request-dependent object bounded and reusable:
//!
//! * the canonical BF16 vocabulary matrix is uploaded once by
//!   `OneJevGpuHead::load` and never duplicated as a full F32 table;
//! * the F32 hidden/selected/logit scratch is capacity-backed and owned by the
//!   existing `Qwen35Work`, so changing option ids, option counts and packed
//!   batch boundaries reuses it without device allocation;
//! * dynamic row ids, last-real-token rows and request row offsets are appended
//!   to the single existing `Work.input` staging region and uploaded once;
//! * the two existing `infinity_bf16_to_f32` gather/cast launches and the plain
//!   F32 cuBLAS compute-32F/default-math projection are reused, with no new
//!   device kernel, heuristic, context query, event or allocation lease.
//!
//! The unsafe enqueue is crate-private because the outer `Work`/`Engine`
//! owner is the completion authority. The safe standalone `forward` and
//! `forward_pooled` wrappers in `decision_head_gpu` remain unchanged and keep
//! their own read-lease retention.

use crate::{
    cublas::Blas,
    decision_head::{output_count_for, plan_decision_requests, PackedDecisionBatch, RequestPlan},
    decision_head_gpu::{launch_cast, DecisionGpuError, GpuHiddenRows, OneJevGpuHead},
    infer,
    math::{GemmArguments, LinearPlan},
    tensor::{Bf16, Layout, F32},
    WorkKey,
};
use cuda_core::{DeviceBuffer, Stream};
use infinity_v3_backend_core::{BackendError, DecisionInput, DecisionOutput};
use std::{mem::ManuallyDrop, sync::Arc};

/// Router-permitted option count for one OneJev question.
pub(crate) const MAX_OPTIONS_PER_REQUEST: usize = 255;
/// Router-permitted selected option rows for one OneJev request.
pub(crate) const MAX_TOTAL_ROWS: usize = 512;

fn geometry(what: &'static str) -> DecisionGpuError {
    DecisionGpuError::Geometry(what)
}

fn checked_product(
    left: usize,
    right: usize,
    what: &'static str,
) -> Result<usize, DecisionGpuError> {
    left.checked_mul(right).ok_or_else(|| geometry(what))
}

fn bytes_of_elements(elements: usize, what: &'static str) -> Result<usize, DecisionGpuError> {
    elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| geometry(what))
}

fn device_offset(base: u64, elements: usize, what: &'static str) -> Result<u64, DecisionGpuError> {
    let bytes = u64::try_from(bytes_of_elements(elements, what)?).map_err(|_| geometry(what))?;
    base.checked_add(bytes).ok_or_else(|| geometry(what))
}

/// Fixed-capacity F32 readout scratch owned by one reusable `Qwen35Work`.
///
/// `requests` follows the work's sequence capacity; `rows` is the fixed router
/// total cap. No request shape participates in allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct WarmHeadCapacity {
    requests: usize,
    rows: usize,
    hidden: usize,
}

impl WarmHeadCapacity {
    pub(crate) fn new(requests: usize, hidden: usize) -> Result<Self, BackendError> {
        if requests == 0 {
            return Err(infer("Warm OneJev capacity requires at least one request"));
        }
        if hidden == 0 {
            return Err(infer(
                "Warm OneJev capacity requires a nonzero hidden width",
            ));
        }
        Ok(Self {
            requests,
            rows: MAX_TOTAL_ROWS,
            hidden,
        })
    }

    pub(crate) fn requests(&self) -> usize {
        self.requests
    }

    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    pub(crate) fn hidden(&self) -> usize {
        self.hidden
    }

    fn hidden_elements(&self) -> Result<usize, BackendError> {
        checked_product(self.requests, self.hidden, "warm hidden element overflow")
            .map_err(|error| infer(error.to_string()))
    }

    fn selected_elements(&self) -> Result<usize, BackendError> {
        checked_product(self.rows, self.hidden, "warm selected element overflow")
            .map_err(|error| infer(error.to_string()))
    }

    /// Device F32 storage: hidden rows, selected weight rows, logits.
    pub(crate) fn device_bytes(&self) -> Result<usize, BackendError> {
        let elements = self
            .hidden_elements()?
            .checked_add(self.selected_elements()?)
            .and_then(|n| n.checked_add(self.rows))
            .ok_or_else(|| infer("Warm OneJev device scratch overflow"))?;
        elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| infer("Warm OneJev device byte overflow"))
    }
}

/// Borrowed device bases for one rebinding. Addresses convey no ownership.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct WarmScratchBases {
    pub(crate) hidden: u64,
    pub(crate) selected: u64,
    pub(crate) logits: u64,
}

struct ScratchBuilding<'a> {
    stream: &'a Arc<Stream>,
    hidden: Option<DeviceBuffer<f32>>,
    selected: Option<DeviceBuffer<f32>>,
    logits: Option<DeviceBuffer<f32>>,
}

impl<'a> ScratchBuilding<'a> {
    fn new(stream: &'a Arc<Stream>) -> Self {
        Self {
            stream,
            hidden: None,
            selected: None,
            logits: None,
        }
    }

    fn take(&mut self) -> Option<WarmHeadScratch> {
        Some(WarmHeadScratch {
            hidden: ManuallyDrop::new(self.hidden.take()?),
            selected: ManuallyDrop::new(self.selected.take()?),
            logits: ManuallyDrop::new(self.logits.take()?),
        })
    }
}

impl Drop for ScratchBuilding<'_> {
    fn drop(&mut self) {
        let allocated = self.hidden.is_some() || self.selected.is_some() || self.logits.is_some();
        if !allocated {
            return;
        }
        if self.stream.synchronize().is_ok() {
            self.hidden.take();
            self.selected.take();
            self.logits.take();
        } else {
            std::mem::forget((
                self.hidden.take(),
                self.selected.take(),
                self.logits.take(),
                Arc::clone(self.stream),
            ));
        }
    }
}

/// Capacity-backed F32 readout scratch. The outer `Work` drop proves the
/// stream drained before these owners leave the arena.
pub(crate) struct WarmHeadScratch {
    hidden: ManuallyDrop<DeviceBuffer<f32>>,
    selected: ManuallyDrop<DeviceBuffer<f32>>,
    logits: ManuallyDrop<DeviceBuffer<f32>>,
}

impl WarmHeadScratch {
    pub(crate) fn new(
        stream: &Arc<Stream>,
        capacity: WarmHeadCapacity,
    ) -> Result<Self, BackendError> {
        let mut pending = ScratchBuilding::new(stream);
        pending.hidden =
            Some(DeviceBuffer::zeroed(stream, capacity.hidden_elements()?).map_err(infer)?);
        pending.selected =
            Some(DeviceBuffer::zeroed(stream, capacity.selected_elements()?).map_err(infer)?);
        pending.logits = Some(DeviceBuffer::zeroed(stream, capacity.rows).map_err(infer)?);
        stream.synchronize().map_err(infer)?;
        pending
            .take()
            .ok_or_else(|| infer("Warm OneJev scratch constructor lost owners"))
    }

    pub(crate) fn bases(&self) -> WarmScratchBases {
        WarmScratchBases {
            hidden: self.hidden.cu_deviceptr(),
            selected: self.selected.cu_deviceptr(),
            logits: self.logits.cu_deviceptr(),
        }
    }

    pub(crate) fn hidden_len(&self) -> usize {
        self.hidden.len()
    }

    pub(crate) fn selected_len(&self) -> usize {
        self.selected.len()
    }

    pub(crate) fn logits_len(&self) -> usize {
        self.logits.len()
    }
}

impl Drop for WarmHeadScratch {
    fn drop(&mut self) {
        // SAFETY: the owning Work drains the stream before dropping Payload.
        unsafe {
            ManuallyDrop::drop(&mut self.hidden);
            ManuallyDrop::drop(&mut self.selected);
            ManuallyDrop::drop(&mut self.logits);
        }
    }
}

/// Fixed metadata slot appended after the existing packed input layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct WarmInputLayout {
    metadata_base: usize,
    row_ids: usize,
    last_rows: usize,
    row_offsets: usize,
    end: usize,
}

impl WarmInputLayout {
    pub(crate) fn new(key: WorkKey) -> Result<Self, BackendError> {
        Self::with_reserved_prefix(key, 0)
    }

    pub(crate) fn with_reserved_prefix(
        key: WorkKey,
        reserved_words: usize,
    ) -> Result<Self, BackendError> {
        let metadata_base = crate::staging::input_capacity(key)?
            .checked_add(reserved_words)
            .ok_or_else(|| infer("Warm OneJev metadata base overflow"))?;
        let row_ids = metadata_base;
        let last_rows = row_ids
            .checked_add(MAX_TOTAL_ROWS)
            .ok_or_else(|| infer("Warm OneJev row id slot overflow"))?;
        let row_offsets = last_rows
            .checked_add(key.1)
            .ok_or_else(|| infer("Warm OneJev last-row slot overflow"))?;
        let end = row_offsets
            .checked_add(key.1)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| infer("Warm OneJev request offset slot overflow"))?;
        Ok(Self {
            metadata_base,
            row_ids,
            last_rows,
            row_offsets,
            end,
        })
    }

    pub(crate) fn metadata_base(&self) -> usize {
        self.metadata_base
    }

    pub(crate) fn row_ids(&self) -> usize {
        self.row_ids
    }

    pub(crate) fn last_rows(&self) -> usize {
        self.last_rows
    }

    pub(crate) fn row_offsets(&self) -> usize {
        self.row_offsets
    }

    pub(crate) fn end(&self) -> usize {
        self.end
    }

    pub(crate) fn words(&self) -> usize {
        self.end.saturating_sub(self.metadata_base)
    }
}

/// Metadata capacity appended to every existing input staging slot. It is a
/// small fixed row table plus one last-row word and one offset word per
/// sequence capacity.
pub(crate) fn metadata_words(sequences_capacity: usize) -> Result<usize, BackendError> {
    MAX_TOTAL_ROWS
        .checked_add(
            sequences_capacity
                .checked_mul(2)
                .ok_or_else(|| infer("Warm OneJev metadata capacity overflow"))?,
        )
        .and_then(|n| n.checked_add(1))
        .ok_or_else(|| infer("Warm OneJev metadata capacity overflow"))
}

/// Fully validated CPU metadata for one warm readout, before any device base
/// is bound. The single `RequestPlan` is retained and moved into the bound
/// plan, so preparing and binding never validate or allocate it twice.
pub(crate) struct WarmDecisionMetadata {
    layout: WarmInputLayout,
    key: WorkKey,
    tokens: usize,
    hidden: usize,
    plan: RequestPlan,
}

impl WarmDecisionMetadata {
    pub(crate) fn layout(&self) -> WarmInputLayout {
        self.layout
    }

    pub(crate) fn total_rows(&self) -> usize {
        self.plan.total
    }

    pub(crate) fn key(&self) -> WorkKey {
        self.key
    }
}

/// One request-rebound readout plan. Every field is CPU metadata; the binding
/// table contains only checked F32 GEMM descriptors.
pub(crate) struct WarmDecisionPlan {
    layout: WarmInputLayout,
    tokens: usize,
    hidden: usize,
    request_count: usize,
    total_rows: usize,
    counts: Vec<usize>,
    row_offsets: Vec<usize>,
    row_ids: Vec<u32>,
    last_rows: Vec<u32>,
    projections: Vec<GemmArguments>,
}

/// Rebind the exact per-request F32 `[1, hidden] x [count, hidden]^T`
/// descriptors against capacity-backed scratch. No allocation, heuristic,
/// context bind or device query occurs here.
pub(crate) fn bind_projections(
    hidden: usize,
    plan: &RequestPlan,
    bases: WarmScratchBases,
) -> Result<Vec<GemmArguments>, DecisionGpuError> {
    if hidden == 0 {
        return Err(geometry("warm F32 projection width must be nonzero"));
    }
    if plan.row_offsets.len() != plan.counts.len().saturating_add(1) {
        return Err(geometry(
            "warm request row offsets do not match the request count",
        ));
    }
    if plan.row_offsets.last().copied() != Some(plan.total)
        || plan.row_ids.len() != plan.total
        || plan.last_rows.len() != plan.counts.len()
    {
        return Err(geometry("warm request plan row counts are inconsistent"));
    }
    let mut projections = Vec::new();
    projections
        .try_reserve_exact(plan.counts.len())
        .map_err(|_| DecisionGpuError::OutputAllocation)?;
    for (index, (&count, &row_offset)) in
        plan.counts.iter().zip(plan.row_offsets.iter()).enumerate()
    {
        if count == 0 || count > MAX_OPTIONS_PER_REQUEST {
            return Err(geometry(
                "warm request option count is outside the 255-option cap",
            ));
        }
        let x = device_offset(
            bases.hidden,
            checked_product(index, hidden, "warm hidden row offset overflow")?,
            "warm hidden row offset overflow",
        )?;
        let weight = device_offset(
            bases.selected,
            checked_product(row_offset, hidden, "warm selected row offset overflow")?,
            "warm selected row offset overflow",
        )?;
        let output = device_offset(bases.logits, row_offset, "warm logit row offset overflow")?;
        let linear = LinearPlan::<F32>::new(
            Layout::contiguous([1, hidden]).map_err(|_| geometry("invalid hidden layout"))?,
            Layout::contiguous([count, hidden])
                .map_err(|_| geometry("invalid selected row layout"))?,
            Layout::contiguous([1, count]).map_err(|_| geometry("invalid logit layout"))?,
        )
        .map_err(|_| geometry("invalid F32 projection geometry"))?;
        if linear.input_layout().shape() != &[1, hidden]
            || linear.weight_layout().shape() != &[count, hidden]
            || linear.output_layout().shape() != &[1, count]
        {
            return Err(geometry(
                "warm F32 projection shape differs from its request",
            ));
        }
        projections.push(linear.arguments(x, weight, output));
    }
    Ok(projections)
}

/// Validate the whole packed request batch and rebind it to reusable scratch.
#[allow(clippy::too_many_arguments)]
pub(crate) fn plan_warm_decision(
    head: &OneJevGpuHead,
    capacity: WarmHeadCapacity,
    key: WorkKey,
    tokens: usize,
    batch: &PackedDecisionBatch<'_>,
    requests: &[DecisionInput],
    bases: WarmScratchBases,
) -> Result<WarmDecisionPlan, DecisionGpuError> {
    plan_warm_decision_with_reserved_prefix(head, capacity, key, tokens, batch, requests, bases, 0)
}

/// Rebind the warm readout after an existing staging prefix of
/// `reserved_words`. Ordinary workers pass zero; a block-plus-head worker passes
/// only the block metadata word count so the warm tail starts after it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn plan_warm_decision_with_reserved_prefix(
    head: &OneJevGpuHead,
    capacity: WarmHeadCapacity,
    key: WorkKey,
    tokens: usize,
    batch: &PackedDecisionBatch<'_>,
    requests: &[DecisionInput],
    bases: WarmScratchBases,
    reserved_words: usize,
) -> Result<WarmDecisionPlan, DecisionGpuError> {
    plan_warm_decision_parts_with_reserved_prefix(
        head.vocab_rows(),
        head.hidden_size(),
        capacity,
        key,
        tokens,
        batch,
        requests,
        bases,
        reserved_words,
    )
}

/// Pure metadata/binding half of [`plan_warm_decision`]. Kept separate so the
/// CPU gate exercises the exact production plan and GEMM records without a
/// CUDA context or a device head object.
#[allow(clippy::too_many_arguments)]
pub(crate) fn plan_warm_decision_parts(
    vocab_rows: usize,
    hidden: usize,
    capacity: WarmHeadCapacity,
    key: WorkKey,
    tokens: usize,
    batch: &PackedDecisionBatch<'_>,
    requests: &[DecisionInput],
    bases: WarmScratchBases,
) -> Result<WarmDecisionPlan, DecisionGpuError> {
    plan_warm_decision_parts_with_reserved_prefix(
        vocab_rows, hidden, capacity, key, tokens, batch, requests, bases, 0,
    )
}

/// Validate and plan the whole packed request batch without touching any
/// device base, Work arena or context. The returned metadata owns the exact
/// validated `RequestPlan` that [`bind_warm_decision_metadata`] moves into the
/// final plan.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_warm_decision_metadata(
    vocab_rows: usize,
    hidden: usize,
    capacity: WarmHeadCapacity,
    key: WorkKey,
    tokens: usize,
    batch: &PackedDecisionBatch<'_>,
    requests: &[DecisionInput],
    reserved_words: usize,
) -> Result<WarmDecisionMetadata, DecisionGpuError> {
    if hidden != capacity.hidden {
        return Err(DecisionGpuError::PreparedMismatch(
            "warm head hidden width differs from the Work scratch capacity",
        ));
    }
    if tokens == 0 || batch.tokens() != tokens {
        return Err(geometry(
            "warm activation rows do not match the packed decision batch",
        ));
    }
    let plan = plan_decision_requests(vocab_rows, batch, requests)?;
    if plan.counts.is_empty() {
        return Err(geometry("warm decision batch must contain a request"));
    }
    if plan.counts.len() > capacity.requests {
        return Err(geometry(
            "warm request count exceeds the retained Work sequence capacity",
        ));
    }
    if plan.total == 0 || plan.total > capacity.rows {
        return Err(geometry(
            "warm selected row count exceeds the retained 512-row capacity",
        ));
    }
    for (index, (&count, &last)) in plan.counts.iter().zip(plan.last_rows.iter()).enumerate() {
        if count == 0 || count > MAX_OPTIONS_PER_REQUEST {
            return Err(geometry(
                "warm request option count is outside the 255-option cap",
            ));
        }
        if usize::try_from(last).map_or(true, |row| row >= tokens) {
            return Err(geometry(
                "warm last real token row does not address the activation span",
            ));
        }
        let request = requests
            .get(index)
            .ok_or(DecisionGpuError::PreparedMismatch(
                "warm plan lost a request while rebinding",
            ))?;
        let expected = output_count_for(vocab_rows, index, request)?;
        if expected != count {
            return Err(DecisionGpuError::PreparedMismatch(
                "warm rebinding changed a request output count",
            ));
        }
    }
    let layout = WarmInputLayout::with_reserved_prefix(key, reserved_words)?;
    Ok(WarmDecisionMetadata {
        layout,
        key,
        tokens,
        hidden,
        plan,
    })
}

/// Bind checked F32 addresses to already validated metadata. This moves the
/// retained `RequestPlan` vectors without reparsing or cloning them.
pub(crate) fn bind_warm_decision_metadata(
    metadata: WarmDecisionMetadata,
    bases: WarmScratchBases,
) -> Result<WarmDecisionPlan, DecisionGpuError> {
    let WarmDecisionMetadata {
        layout,
        key: _,
        tokens,
        hidden,
        plan,
    } = metadata;
    let projections = bind_projections(hidden, &plan, bases)?;
    let request_count = plan.counts.len();
    let total_rows = plan.total;
    Ok(WarmDecisionPlan {
        layout,
        tokens,
        hidden,
        request_count,
        total_rows,
        counts: plan.counts,
        row_offsets: plan.row_offsets,
        row_ids: plan.row_ids,
        last_rows: plan.last_rows,
        projections,
    })
}

/// Pure counterpart of [`plan_warm_decision_with_reserved_prefix`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn plan_warm_decision_parts_with_reserved_prefix(
    vocab_rows: usize,
    hidden: usize,
    capacity: WarmHeadCapacity,
    key: WorkKey,
    tokens: usize,
    batch: &PackedDecisionBatch<'_>,
    requests: &[DecisionInput],
    bases: WarmScratchBases,
    reserved_words: usize,
) -> Result<WarmDecisionPlan, DecisionGpuError> {
    let metadata = prepare_warm_decision_metadata(
        vocab_rows,
        hidden,
        capacity,
        key,
        tokens,
        batch,
        requests,
        reserved_words,
    )?;
    bind_warm_decision_metadata(metadata, bases)
}

impl WarmDecisionPlan {
    pub(crate) fn layout(&self) -> WarmInputLayout {
        self.layout
    }

    pub(crate) fn request_count(&self) -> usize {
        self.request_count
    }

    pub(crate) fn total_rows(&self) -> usize {
        self.total_rows
    }

    pub(crate) fn hidden(&self) -> usize {
        self.hidden
    }

    pub(crate) fn tokens(&self) -> usize {
        self.tokens
    }

    pub(crate) fn counts(&self) -> &[usize] {
        &self.counts
    }

    pub(crate) fn row_offsets(&self) -> &[usize] {
        &self.row_offsets
    }

    pub(crate) fn row_ids(&self) -> &[u32] {
        &self.row_ids
    }

    pub(crate) fn last_rows(&self) -> &[u32] {
        &self.last_rows
    }

    pub(crate) fn projections(&self) -> &[GemmArguments] {
        &self.projections
    }

    /// Append the dynamic row ids, last-real-token rows and request row
    /// offsets to the already packed single input staging slice. The returned
    /// element count covers the ordinary prefix padding plus the fixed metadata
    /// tail, so the caller performs exactly one H2D upload.
    pub(crate) fn pack_metadata(&self, input: &mut [u32]) -> Result<usize, BackendError> {
        let layout = self.layout;
        let needed = layout.end();
        if input.len() < needed {
            return Err(infer(
                "Warm OneJev metadata exceeds the retained input staging capacity",
            ));
        }
        let row = layout.row_ids();
        let tail = row
            .checked_add(self.row_ids.len())
            .ok_or_else(|| infer("Warm OneJev row id staging overflow"))?;
        if tail > layout.last_rows() {
            return Err(infer("Warm OneJev row ids exceed their staging slot"));
        }
        let last = layout.last_rows();
        let last_tail = last
            .checked_add(self.last_rows.len())
            .ok_or_else(|| infer("Warm OneJev last-row staging overflow"))?;
        if last_tail > layout.row_offsets() {
            return Err(infer("Warm OneJev last rows exceed their staging slot"));
        }
        let offsets = layout.row_offsets();
        let offsets_tail = offsets
            .checked_add(self.row_offsets.len())
            .ok_or_else(|| infer("Warm OneJev request offset staging overflow"))?;
        if offsets_tail > layout.end() {
            return Err(infer(
                "Warm OneJev request offsets exceed their staging slot",
            ));
        }
        let slot = input
            .get_mut(row..tail)
            .ok_or_else(|| infer("Warm OneJev row id slot is unavailable"))?;
        slot.copy_from_slice(&self.row_ids);
        let slot = input
            .get_mut(last..last_tail)
            .ok_or_else(|| infer("Warm OneJev last-row slot is unavailable"))?;
        slot.copy_from_slice(&self.last_rows);
        for (slot, value) in input
            .get_mut(offsets..offsets_tail)
            .ok_or_else(|| infer("Warm OneJev offset slot is unavailable"))?
            .iter_mut()
            .zip(&self.row_offsets)
        {
            *slot = u32::try_from(*value)
                .map_err(|_| infer("Warm OneJev request offset exceeds u32"))?;
        }
        Ok(layout.end())
    }
}

fn metadata_pointer(
    input: &DeviceBuffer<u32>,
    element: usize,
    elements: usize,
    what: &'static str,
) -> Result<u64, DecisionGpuError> {
    let end = element
        .checked_add(elements)
        .ok_or_else(|| geometry(what))?;
    if elements == 0 || end > input.len() {
        return Err(geometry(what));
    }
    let bytes = u64::try_from(
        element
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| geometry(what))?,
    )
    .map_err(|_| geometry(what))?;
    input
        .cu_deviceptr()
        .checked_add(bytes)
        .ok_or_else(|| geometry(what))
}

/// Engine-bound enqueue of the two existing BF16-widen gathers, every checked
/// F32 projection and one logit D2H copy.
///
/// No allocation, context bind, capability query, event, lease or synchronize
/// occurs here. The outer `Work`/`Engine` owns `activation`, `input`, `scratch`,
/// the canonical head weight and the stream until its completion boundary
/// drains or quarantines the whole request.
///
/// # Safety
/// The calling engine MUST own the bound context/stream and keep every input,
/// scratch and head owner alive without conflicting access until the stream
/// drains. The provided `activation` MUST be the actual normalized BF16 Work
/// activation span for `plan.tokens()` rows, and `input` MUST be the same Work
/// input buffer that received `plan.pack_metadata`. On any error the caller
/// MUST drain or quarantine the whole Work.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn enqueue_warm_decision(
    head: &OneJevGpuHead,
    blas: &Blas,
    stream: &Stream,
    plan: &WarmDecisionPlan,
    scratch: &mut WarmHeadScratch,
    activation: &DeviceBuffer<Bf16>,
    input: &DeviceBuffer<u32>,
    host_logits: &mut [f32],
) -> Result<(), DecisionGpuError> {
    if plan.hidden != head.hidden_size() {
        return Err(DecisionGpuError::PreparedMismatch(
            "warm plan hidden width differs from the head",
        ));
    }
    let hidden_rows = plan
        .request_count
        .checked_mul(plan.hidden)
        .ok_or_else(|| geometry("warm hidden scratch size overflow"))?;
    let selected_rows = plan
        .total_rows
        .checked_mul(plan.hidden)
        .ok_or_else(|| geometry("warm selected scratch size overflow"))?;
    if scratch.hidden_len() < hidden_rows
        || scratch.selected_len() < selected_rows
        || scratch.logits_len() < plan.total_rows
        || host_logits.len() < plan.total_rows
    {
        return Err(DecisionGpuError::PreparedMismatch(
            "warm plan exceeds the retained Work scratch capacity",
        ));
    }
    if plan.projections.len() != plan.request_count
        || plan.counts.len() != plan.request_count
        || plan.row_offsets.len() != plan.request_count.saturating_add(1)
        || plan.row_ids.len() != plan.total_rows
        || plan.last_rows.len() != plan.request_count
    {
        return Err(DecisionGpuError::PreparedMismatch(
            "warm plan metadata is internally inconsistent",
        ));
    }
    let _hidden_rows = GpuHiddenRows::new(activation, plan.tokens, plan.hidden)?;
    let row_ids = metadata_pointer(
        input,
        plan.layout.row_ids(),
        plan.total_rows,
        "warm row id metadata exceeds the retained input",
    )?;
    let last_rows = metadata_pointer(
        input,
        plan.layout.last_rows(),
        plan.request_count,
        "warm last-row metadata exceeds the retained input",
    )?;
    // The caller's retained ownership contract covers both launches.
    launch_cast(
        head.cast_kernel(),
        stream,
        activation,
        &scratch.hidden,
        last_rows,
        plan.request_count,
        plan.hidden,
    )?;
    launch_cast(
        head.cast_kernel(),
        stream,
        head.weight(),
        &scratch.selected,
        row_ids,
        plan.total_rows,
        plan.hidden,
    )?;
    for arguments in &plan.projections {
        // SAFETY: the caller keeps the checked F32 buffers and head weight
        // alive through the outer completion boundary.
        unsafe { blas.enqueue_f32_projection(arguments) }.map_err(|error| {
            DecisionGpuError::Driver(format!("warm F32 decision projection: {error}"))
        })?;
    }
    let bytes = plan
        .total_rows
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| geometry("warm logit readback size overflow"))?;
    let host_logits = host_logits.as_mut_ptr();
    // SAFETY: the request-specific host output slot and device logits remain
    // owned through the caller's completion boundary.
    unsafe {
        cuda_core::memory::memcpy_dtoh_async(
            host_logits,
            scratch.logits.cu_deviceptr(),
            bytes,
            stream.cu_stream(),
        )
    }
    .map_err(|error| DecisionGpuError::Driver(format!("warm logit readback: {error}")))
}

/// Materialize the small host logits after the caller has proven the stream
/// drained. Order, duplicates, warmup row zero and per-request boundaries are
/// preserved exactly as planned.
pub(crate) fn finish_warm_decision(
    plan: &WarmDecisionPlan,
    host_logits: &[f32],
) -> Result<Vec<DecisionOutput>, DecisionGpuError> {
    let host = host_logits;
    let mut results = Vec::new();
    results
        .try_reserve_exact(plan.counts.len())
        .map_err(|_| DecisionGpuError::OutputAllocation)?;
    for (index, count) in plan.counts.iter().enumerate() {
        let begin =
            plan.row_offsets
                .get(index)
                .copied()
                .ok_or(DecisionGpuError::PreparedMismatch(
                    "warm plan lost a row offset",
                ))?;
        let end = begin
            .checked_add(*count)
            .ok_or_else(|| geometry("warm logit range overflow"))?;
        let values = host
            .get(begin..end)
            .ok_or(DecisionGpuError::PreparedMismatch(
                "warm host logits are incomplete",
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decision_head::{HiddenRows, OneJevHead};
    use cuda_core::Context;

    const HIDDEN: usize = 4;
    const VOCAB: usize = 8;

    fn option(token_ids: &[u32]) -> DecisionInput {
        DecisionInput::OptionTokens {
            token_ids: token_ids.to_vec(),
        }
    }

    fn bases() -> WarmScratchBases {
        WarmScratchBases {
            hidden: 0x1000,
            selected: 0x2000,
            logits: 0x3000,
        }
    }

    fn capacity(requests: usize) -> WarmHeadCapacity {
        WarmHeadCapacity::new(requests, HIDDEN).unwrap()
    }

    #[test]
    fn changing_ids_counts_and_boundaries_rebind_checked_records_in_place() {
        let key = (32, 4, 2);
        let first_batch = PackedDecisionBatch::new(&[0, 3, 5]).unwrap();
        let first_requests = [option(&[4, 1]), DecisionInput::Warmup];
        let first = plan_warm_decision_parts(
            VOCAB,
            HIDDEN,
            capacity(4),
            key,
            first_batch.tokens(),
            &first_batch,
            &first_requests,
            bases(),
        )
        .unwrap();

        let second_batch = PackedDecisionBatch::new(&[0, 2, 4, 6]).unwrap();
        let second_requests = [option(&[5, 0, 5]), option(&[1]), option(&[2, 3])];
        let second = plan_warm_decision_parts(
            VOCAB,
            HIDDEN,
            capacity(4),
            key,
            second_batch.tokens(),
            &second_batch,
            &second_requests,
            bases(),
        )
        .unwrap();

        assert_eq!(first.counts(), &[2, 1]);
        assert_eq!(first.row_offsets(), &[0, 2, 3]);
        assert_eq!(first.row_ids(), &[4, 1, 0]);
        assert_eq!(first.last_rows(), &[2, 4]);
        assert_eq!(second.counts(), &[3, 1, 2]);
        assert_eq!(second.row_offsets(), &[0, 3, 4, 6]);
        assert_eq!(second.row_ids(), &[5, 0, 5, 1, 2, 3]);
        assert_eq!(second.last_rows(), &[1, 3, 5]);
        for (plan, counts, offsets) in [
            (&first, first.counts(), first.row_offsets()),
            (&second, second.counts(), second.row_offsets()),
        ] {
            for (index, projection) in plan.projections().iter().enumerate() {
                assert_eq!(projection.cuda_type, 0);
                assert_eq!(projection.rows, 1);
                assert_eq!(projection.input, HIDDEN as i32);
                assert_eq!(projection.width, counts[index] as i32);
                assert_eq!(projection.x_stride, HIDDEN as i32);
                assert_eq!(projection.weight_stride, HIDDEN as i32);
                assert_eq!(projection.output_stride, counts[index] as i32);
                assert_eq!(
                    projection.x,
                    bases().hidden + (index * HIDDEN * std::mem::size_of::<f32>()) as u64
                );
                assert_eq!(
                    projection.weight,
                    bases().selected
                        + (offsets[index] * HIDDEN * std::mem::size_of::<f32>()) as u64
                );
                assert_eq!(
                    projection.output,
                    bases().logits + (offsets[index] * std::mem::size_of::<f32>()) as u64
                );
            }
        }
    }

    #[test]
    fn prepared_metadata_binds_the_exact_legacy_plan_outputs() {
        let key = (32, 4, 2);
        let batch = PackedDecisionBatch::new(&[0, 3, 5]).unwrap();
        let requests = [option(&[4, 1]), DecisionInput::Warmup];
        let legacy = plan_warm_decision_parts(
            VOCAB,
            HIDDEN,
            capacity(4),
            key,
            batch.tokens(),
            &batch,
            &requests,
            bases(),
        )
        .unwrap();
        let metadata = prepare_warm_decision_metadata(
            VOCAB,
            HIDDEN,
            capacity(4),
            key,
            batch.tokens(),
            &batch,
            &requests,
            0,
        )
        .unwrap();
        assert_eq!(metadata.layout(), legacy.layout());
        assert_eq!(metadata.total_rows(), legacy.total_rows());
        assert_eq!(metadata.key(), key);
        let bound = bind_warm_decision_metadata(metadata, bases()).unwrap();
        assert_eq!(bound.layout(), legacy.layout());
        assert_eq!(bound.tokens(), legacy.tokens());
        assert_eq!(bound.hidden(), legacy.hidden());
        assert_eq!(bound.request_count(), legacy.request_count());
        assert_eq!(bound.total_rows(), legacy.total_rows());
        assert_eq!(bound.counts(), legacy.counts());
        assert_eq!(bound.row_offsets(), legacy.row_offsets());
        assert_eq!(bound.row_ids(), legacy.row_ids());
        assert_eq!(bound.last_rows(), legacy.last_rows());
        assert_eq!(bound.projections(), legacy.projections());
    }

    #[test]
    fn invalid_metadata_is_rejected_before_any_binding() {
        let key = (32, 4, 2);
        let batch = PackedDecisionBatch::new(&[0, 3, 5]).unwrap();
        let tokens = batch.tokens();
        assert!(prepare_warm_decision_metadata(
            VOCAB,
            HIDDEN,
            capacity(4),
            key,
            tokens,
            &batch,
            &[option(&[VOCAB as u32]), DecisionInput::Warmup],
            0,
        )
        .is_err());
        assert!(prepare_warm_decision_metadata(
            VOCAB,
            HIDDEN,
            capacity(4),
            key,
            tokens,
            &batch,
            &[option(&[4, 1])],
            0,
        )
        .is_err());
        assert!(prepare_warm_decision_metadata(
            VOCAB,
            HIDDEN,
            capacity(4),
            key,
            tokens,
            &batch,
            &[
                DecisionInput::BlockOptionTokens {
                    token_ids: vec![1],
                    prefix_length: tokens + 1,
                },
                DecisionInput::Warmup,
            ],
            0,
        )
        .is_err());

        let batch = PackedDecisionBatch::new(&[0, 1, 2]).unwrap();
        let requests = [option(&[1]), option(&[2])];
        assert!(prepare_warm_decision_metadata(
            VOCAB,
            HIDDEN,
            capacity(1),
            key,
            batch.tokens(),
            &batch,
            &requests,
            0,
        )
        .is_err());

        let too_many = (0..=MAX_OPTIONS_PER_REQUEST as u32).collect::<Vec<_>>();
        let batch = PackedDecisionBatch::new(&[0, 1]).unwrap();
        assert!(prepare_warm_decision_metadata(
            MAX_OPTIONS_PER_REQUEST + 1,
            HIDDEN,
            capacity(4),
            key,
            batch.tokens(),
            &batch,
            &[option(&too_many)],
            0,
        )
        .is_err());

        let batch = PackedDecisionBatch::new(&[0, 1, 2, 3]).unwrap();
        let over_total = [
            option(&(0..200).collect::<Vec<_>>()),
            option(&(0..200).collect::<Vec<_>>()),
            option(&(0..200).collect::<Vec<_>>()),
        ];
        assert!(prepare_warm_decision_metadata(
            VOCAB + 300,
            HIDDEN,
            capacity(4),
            key,
            batch.tokens(),
            &batch,
            &over_total,
            0,
        )
        .is_err());

        let batch = PackedDecisionBatch::new(&[0, 3, 5]).unwrap();
        let requests = [option(&[4, 1]), DecisionInput::Warmup];
        assert!(prepare_warm_decision_metadata(
            VOCAB,
            HIDDEN,
            capacity(4),
            key,
            batch.tokens(),
            &batch,
            &requests,
            usize::MAX,
        )
        .is_err());
        assert!(prepare_warm_decision_metadata(
            VOCAB,
            HIDDEN,
            capacity(4),
            (usize::MAX, 2, 1),
            batch.tokens(),
            &batch,
            &requests,
            0,
        )
        .is_err());
    }

    #[test]
    fn metadata_tail_is_opt_in_inside_the_existing_input_capacity() {
        let key = (32, 4, 2);
        let layout = WarmInputLayout::new(key).unwrap();
        let base = crate::staging::input_capacity(key).unwrap();
        assert_eq!(crate::staging::input_capacity(key).unwrap(), base);
        let input_capacity =
            crate::staging::input_capacity_with_tail(key, metadata_words(key.1).unwrap()).unwrap();
        assert_eq!(layout.metadata_base(), base);
        assert_eq!(input_capacity, base + metadata_words(key.1).unwrap());
        assert_eq!(layout.words(), metadata_words(key.1).unwrap());
        assert_eq!(layout.end(), input_capacity);

        let batch = PackedDecisionBatch::new(&[0, 3, 5]).unwrap();
        let requests = [option(&[4, 1]), DecisionInput::Warmup];
        let plan = plan_warm_decision_parts(
            VOCAB,
            HIDDEN,
            capacity(4),
            key,
            batch.tokens(),
            &batch,
            &requests,
            bases(),
        )
        .unwrap();
        let mut host = vec![0u32; input_capacity];
        let uploaded = plan.pack_metadata(&mut host).unwrap();
        assert_eq!(uploaded, input_capacity);
        assert_eq!(
            &host[layout.row_ids()..layout.row_ids() + plan.row_ids().len()],
            plan.row_ids()
        );
        assert_eq!(
            &host[layout.last_rows()..layout.last_rows() + plan.last_rows().len()],
            plan.last_rows()
        );
        assert_eq!(
            &host[layout.row_offsets()..layout.row_offsets() + plan.row_offsets().len()],
            &[0, 2, 3]
        );
        assert!(pack_metadata_short_buffer_rejected(&plan, &host));
    }

    fn pack_metadata_short_buffer_rejected(plan: &WarmDecisionPlan, host: &[u32]) -> bool {
        let mut short = host.to_vec();
        short.truncate(plan.layout().end().saturating_sub(1));
        plan.pack_metadata(&mut short).is_err()
    }

    #[test]
    fn reserved_zero_planning_parts_match_the_ordinary_records() {
        let key = (32, 4, 2);
        let batch = PackedDecisionBatch::new(&[0, 3, 5]).unwrap();
        let requests = [option(&[4, 1]), DecisionInput::Warmup];
        let ordinary = plan_warm_decision_parts(
            VOCAB,
            HIDDEN,
            capacity(4),
            key,
            batch.tokens(),
            &batch,
            &requests,
            bases(),
        )
        .unwrap();
        let zero = plan_warm_decision_parts_with_reserved_prefix(
            VOCAB,
            HIDDEN,
            capacity(4),
            key,
            batch.tokens(),
            &batch,
            &requests,
            bases(),
            0,
        )
        .unwrap();
        assert_eq!(zero.layout(), ordinary.layout());
        assert_eq!(zero.tokens(), ordinary.tokens());
        assert_eq!(zero.hidden(), ordinary.hidden());
        assert_eq!(zero.request_count(), ordinary.request_count());
        assert_eq!(zero.total_rows(), ordinary.total_rows());
        assert_eq!(zero.counts(), ordinary.counts());
        assert_eq!(zero.row_offsets(), ordinary.row_offsets());
        assert_eq!(zero.row_ids(), ordinary.row_ids());
        assert_eq!(zero.last_rows(), ordinary.last_rows());
        assert_eq!(zero.projections(), ordinary.projections());
    }

    #[test]
    fn reserved_block_prefix_moves_only_metadata_addresses_and_end() {
        let key = (32, 4, 2);
        let reserved = 9;
        let batch = PackedDecisionBatch::new(&[0, 3, 5]).unwrap();
        let requests = [option(&[4, 1]), DecisionInput::Warmup];
        let ordinary = plan_warm_decision_parts(
            VOCAB,
            HIDDEN,
            capacity(4),
            key,
            batch.tokens(),
            &batch,
            &requests,
            bases(),
        )
        .unwrap();
        let shifted = plan_warm_decision_parts_with_reserved_prefix(
            VOCAB,
            HIDDEN,
            capacity(4),
            key,
            batch.tokens(),
            &batch,
            &requests,
            bases(),
            reserved,
        )
        .unwrap();
        let old = ordinary.layout();
        let new = shifted.layout();
        assert_eq!(
            new,
            WarmInputLayout::with_reserved_prefix(key, reserved).unwrap()
        );
        assert_eq!(new.metadata_base(), old.metadata_base() + reserved);
        assert_eq!(new.row_ids(), old.row_ids() + reserved);
        assert_eq!(new.last_rows(), old.last_rows() + reserved);
        assert_eq!(new.row_offsets(), old.row_offsets() + reserved);
        assert_eq!(new.end(), old.end() + reserved);
        assert_eq!(new.words(), old.words());
        assert_eq!(shifted.counts(), ordinary.counts());
        assert_eq!(shifted.row_offsets(), ordinary.row_offsets());
        assert_eq!(shifted.row_ids(), ordinary.row_ids());
        assert_eq!(shifted.last_rows(), ordinary.last_rows());
        assert_eq!(shifted.projections(), ordinary.projections());
        let host = (0..ordinary.total_rows())
            .map(|row| 100.0 + row as f32)
            .collect::<Vec<_>>();
        let ordinary_output = finish_warm_decision(&ordinary, &host).unwrap();
        let shifted_output = finish_warm_decision(&shifted, &host).unwrap();
        assert_eq!(shifted_output.len(), ordinary_output.len());
        for (shifted, ordinary) in shifted_output.iter().zip(&ordinary_output) {
            assert_eq!(shifted.logits, ordinary.logits);
        }
    }

    #[test]
    fn reserved_prefix_pack_keeps_every_old_tail_canary_before_the_shift() {
        let key = (32, 4, 2);
        let reserved = 9;
        let batch = PackedDecisionBatch::new(&[0, 3, 5]).unwrap();
        let requests = [option(&[4, 1]), DecisionInput::Warmup];
        let ordinary = plan_warm_decision_parts(
            VOCAB,
            HIDDEN,
            capacity(4),
            key,
            batch.tokens(),
            &batch,
            &requests,
            bases(),
        )
        .unwrap();
        let shifted = plan_warm_decision_parts_with_reserved_prefix(
            VOCAB,
            HIDDEN,
            capacity(4),
            key,
            batch.tokens(),
            &batch,
            &requests,
            bases(),
            reserved,
        )
        .unwrap();
        let old = ordinary.layout();
        let new = shifted.layout();
        let mut host = (0..new.end() + 4)
            .map(|word| 0xCAFE_0000u32 ^ word as u32)
            .collect::<Vec<_>>();
        let canary = host.clone();
        let uploaded = shifted.pack_metadata(&mut host).unwrap();
        assert_eq!(uploaded, new.end());
        for word in old.metadata_base()..new.metadata_base() {
            assert_eq!(host[word], canary[word], "old tail word {word} changed");
        }
        for word in new.end()..host.len() {
            assert_eq!(host[word], canary[word], "post-tail word {word} changed");
        }
        assert_eq!(
            &host[new.row_ids()..new.row_ids() + shifted.row_ids().len()],
            shifted.row_ids()
        );
        assert_eq!(
            &host[new.last_rows()..new.last_rows() + shifted.last_rows().len()],
            shifted.last_rows()
        );
        assert_eq!(
            &host[new.row_offsets()..new.row_offsets() + shifted.row_offsets().len()],
            &[0, 2, 3]
        );
    }

    #[test]
    fn reserved_prefix_overflow_is_rejected_before_any_write() {
        let key = (32, 4, 2);
        assert!(WarmInputLayout::with_reserved_prefix(key, usize::MAX).is_err());
        assert!(WarmInputLayout::with_reserved_prefix((usize::MAX, 2, 1), 9).is_err());
        let batch = PackedDecisionBatch::new(&[0, 3, 5]).unwrap();
        let requests = [option(&[4, 1]), DecisionInput::Warmup];
        let untouched = vec![0xA5A5_5A5Au32; 16];
        let before = untouched.clone();
        assert!(plan_warm_decision_parts_with_reserved_prefix(
            VOCAB,
            HIDDEN,
            capacity(4),
            key,
            batch.tokens(),
            &batch,
            &requests,
            bases(),
            usize::MAX,
        )
        .is_err());
        assert_eq!(untouched, before);
    }

    #[test]
    fn capacity_accounting_is_fixed_by_router_caps_not_request_shape() {
        let capacity = capacity(4);
        assert_eq!(capacity.requests(), 4);
        assert_eq!(capacity.rows(), MAX_TOTAL_ROWS);
        assert_eq!(capacity.hidden(), HIDDEN);
        assert_eq!(
            capacity.device_bytes().unwrap(),
            (capacity.requests() * HIDDEN + 512 * HIDDEN + 512) * 4
        );
        assert_eq!(metadata_words(4).unwrap(), 512 + 9);
        assert!(WarmHeadCapacity::new(0, HIDDEN).is_err());
        assert!(WarmHeadCapacity::new(4, 0).is_err());
        assert!(metadata_words(usize::MAX).is_err());
        assert!(WarmInputLayout::new((usize::MAX, 2, 1)).is_err());
    }

    #[test]
    fn checked_offsets_reject_overflow_before_device_use() {
        assert!(device_offset(u64::MAX, 1, "overflow").is_err());
        assert!(checked_product(usize::MAX, 2, "overflow").is_err());
        assert!(bytes_of_elements(usize::MAX, "overflow").is_err());
        let malformed = RequestPlan {
            counts: vec![1],
            row_offsets: vec![0],
            row_ids: vec![0],
            boundaries: vec![0, 1],
            last_rows: vec![0],
            total: 1,
        };
        assert!(matches!(
            bind_projections(HIDDEN, &malformed, bases()),
            Err(DecisionGpuError::Geometry(_))
        ));
    }

    #[test]
    fn request_specific_host_outputs_materialize_a_and_b_without_sharing() {
        let key = (32, 4, 2);
        let first_batch = PackedDecisionBatch::new(&[0, 3, 5]).unwrap();
        let first_requests = [option(&[4, 1]), DecisionInput::Warmup];
        let first = plan_warm_decision_parts(
            VOCAB,
            HIDDEN,
            capacity(4),
            key,
            first_batch.tokens(),
            &first_batch,
            &first_requests,
            bases(),
        )
        .unwrap();
        let second_batch = PackedDecisionBatch::new(&[0, 2, 4, 6]).unwrap();
        let second_requests = [option(&[5, 0, 5]), option(&[1]), option(&[2, 3])];
        let second = plan_warm_decision_parts(
            VOCAB,
            HIDDEN,
            capacity(4),
            key,
            second_batch.tokens(),
            &second_batch,
            &second_requests,
            bases(),
        )
        .unwrap();
        let mut host_a = (0..first.total_rows())
            .map(|n| n as f32)
            .collect::<Vec<_>>();
        let mut host_b = (0..second.total_rows())
            .map(|n| 100.0 + n as f32)
            .collect::<Vec<_>>();
        // Complete B first; A must still materialize from its own slot.
        let output_b = finish_warm_decision(&second, &host_b).unwrap();
        let output_a = finish_warm_decision(&first, &host_a).unwrap();
        assert_eq!(
            output_a
                .iter()
                .map(|output| output.logits.clone())
                .collect::<Vec<_>>(),
            vec![vec![0.0, 1.0], vec![2.0]]
        );
        assert_eq!(
            output_b
                .iter()
                .map(|output| output.logits.clone())
                .collect::<Vec<_>>(),
            vec![vec![100.0, 101.0, 102.0], vec![103.0], vec![104.0, 105.0]]
        );
        // Reusing an output slot for a later request cannot change a completed
        // result, because the result owns copied logits rather than a view.
        host_a.fill(-1.0);
        host_b.fill(-2.0);
        assert_eq!(output_a[1].logits, vec![2.0]);
        assert_eq!(output_b[0].logits, vec![100.0, 101.0, 102.0]);
    }

    #[test]
    fn caps_and_last_real_token_are_enforced_with_production_types() {
        let too_many_per_request = (0..=MAX_OPTIONS_PER_REQUEST as u32).collect::<Vec<_>>();
        let batch = PackedDecisionBatch::new(&[0, 1]).unwrap();
        assert!(matches!(
            plan_warm_decision_parts(
                MAX_OPTIONS_PER_REQUEST + 1,
                HIDDEN,
                capacity(4),
                (32, 4, 2),
                1,
                &batch,
                &[option(&too_many_per_request)],
                bases(),
            ),
            Err(DecisionGpuError::Geometry(_))
        ));

        let batch = PackedDecisionBatch::new(&[0, 1, 2, 3]).unwrap();
        let over_total = [
            option(&(0..200).collect::<Vec<_>>()),
            option(&(0..200).collect::<Vec<_>>()),
            option(&(0..200).collect::<Vec<_>>()),
        ];
        assert!(matches!(
            plan_warm_decision_parts(
                VOCAB + 300,
                HIDDEN,
                capacity(4),
                (32, 4, 2),
                3,
                &batch,
                &over_total,
                bases(),
            ),
            Err(DecisionGpuError::Geometry(_))
        ));

        let batch = PackedDecisionBatch::new(&[0, 1, 2, 3]).unwrap();
        let requests = [option(&[1]), option(&[2]), option(&[3])];
        assert!(matches!(
            plan_warm_decision_parts(
                VOCAB,
                HIDDEN,
                capacity(2),
                (32, 4, 2),
                3,
                &batch,
                &requests,
                bases(),
            ),
            Err(DecisionGpuError::Geometry(_))
        ));

        let batch = PackedDecisionBatch::new(&[0, 2]).unwrap();
        assert!(matches!(
            plan_warm_decision_parts(
                VOCAB,
                HIDDEN,
                capacity(4),
                (32, 4, 2),
                1,
                &batch,
                &[option(&[1])],
                bases(),
            ),
            Err(DecisionGpuError::Geometry(_))
        ));

        let batch = PackedDecisionBatch::new(&[0, 1]).unwrap();
        assert!(matches!(
            plan_warm_decision_parts(
                VOCAB,
                HIDDEN,
                capacity(4),
                (32, 4, 2),
                1,
                &batch,
                &[DecisionInput::InlineOptions {
                    markers: vec![0],
                    option_counts: vec![1],
                }],
                bases(),
            ),
            Err(DecisionGpuError::Head(_))
        ));
    }

    #[test]
    #[ignore = "Requires CUDA 0, INFINITY_OXIDE_DECISION_MODEL and INFINITY_OXIDE_KERNEL_MODULE"]
    fn gpu_warm_readout_gate_matches_cpu_on_actual_kernels_and_cublas() {
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
        assert_eq!(hidden, cpu.hidden_size());
        let batch = PackedDecisionBatch::new(&[0, 3, 5]).unwrap();
        let requests = [option(&[2, 1, 2]), DecisionInput::Warmup];
        let key = (
            batch.tokens().max(32).next_power_of_two(),
            (batch.len() + 1).max(2).next_power_of_two(),
            1,
        );
        let capacity = WarmHeadCapacity::new(key.1, hidden).unwrap();
        let mut scratch = WarmHeadScratch::new(&stream, capacity).unwrap();
        let plan = plan_warm_decision_parts(
            gpu.vocab_rows(),
            hidden,
            capacity,
            key,
            batch.tokens(),
            &batch,
            &requests,
            scratch.bases(),
        )
        .unwrap();
        let mut host_input =
            vec![
                0u32;
                crate::staging::input_capacity_with_tail(key, metadata_words(key.1).unwrap(),)
                    .unwrap()
            ];
        plan.pack_metadata(&mut host_input).unwrap();
        let input = DeviceBuffer::from_host(&stream, &host_input).unwrap();
        let bits = (0..batch.tokens() * hidden)
            .map(|index| {
                let value = ((index % 113) as f32 - 56.0) * 0.017;
                half::bf16::from_f32(value).to_bits()
            })
            .collect::<Vec<_>>();
        let activation = bits
            .iter()
            .map(|&bits| Bf16::from_bits(bits))
            .collect::<Vec<_>>();
        let activation = DeviceBuffer::from_host(&stream, &activation).unwrap();
        let mut host_logits = vec![0f32; plan.total_rows()];
        // SAFETY: this test owns the context, stream, head, scratch, activation
        // and input through the explicit synchronize below.
        unsafe {
            enqueue_warm_decision(
                &gpu,
                &blas,
                &stream,
                &plan,
                &mut scratch,
                &activation,
                &input,
                &mut host_logits,
            )
            .unwrap()
        };
        stream.synchronize().unwrap();
        let actual = finish_warm_decision(&plan, &host_logits).unwrap();
        let cpu_plan = cpu.prepare(&batch, &requests).unwrap();
        let expected = cpu
            .forward(
                HiddenRows::from_bf16(&bits, hidden).unwrap(),
                &batch,
                &requests,
                &cpu_plan,
            )
            .unwrap();
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(&expected) {
            assert_eq!(actual.logits.len(), expected.logits.len());
            for (actual, expected) in actual.logits.iter().zip(&expected.logits) {
                let scale = expected.abs().max(1.0);
                assert!((actual - expected).abs() <= 5e-3 * scale);
            }
        }
    }
}
