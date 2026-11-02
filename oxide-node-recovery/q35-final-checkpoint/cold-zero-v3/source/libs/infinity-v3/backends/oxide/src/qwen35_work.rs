// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! One arena owns three phase slabs and one recurrent scratch pipeline.
use crate::decision_head_gpu::{DecisionGpuError, OneJevGpuHead};
use crate::gdn_prepared::{GdnCapacity, GdnKernelSet, GdnPrepared, GdnPreparedBuilding};
use crate::qwen35_block_work::{BlockWork, BlockWorkBuilding};
use crate::qwen35_execute::{PreparedWork, Slabs};
use crate::qwen35_layout::Qwen35Layout;
use crate::qwen35_runtime::Qwen35Model;
use crate::tensor::Bf16;
use crate::warm_head::{WarmDecisionPlan, WarmHeadBuilding, WarmHeadScratch};
use crate::work_retirement::{CompletedTicket, CompletionTicket};
use crate::{infer, product, HostIoLease, WorkKey};
use cuda_core::{DeviceBuffer, DeviceCopy, Stream};
use infinity_v3_backend_core::{BackendError, Batch};
use std::{
    mem::ManuallyDrop,
    sync::{atomic::Ordering, Arc},
};

/// Owners whose initialization was enqueued but whose actual stream fence has
/// not yet been observed. Drop is deliberately conservative: an owner that was
/// never settled is leaked rather than freed while queued work may still touch
/// it. Only [`ColdPending::settle`] may release owners after a successful fence.
pub(crate) struct ColdPending<T> {
    owners: Option<T>,
    charged_bytes: Option<usize>,
}

pub(crate) enum ColdFenceOutcome<T> {
    Fenced(T),
    Quarantined { bytes: Option<usize> },
}

impl<T> ColdPending<T> {
    pub(crate) fn new(owners: T, charged_bytes: Option<usize>) -> Self {
        Self {
            owners: Some(owners),
            charged_bytes,
        }
    }

    pub(crate) fn owners_ref(&self) -> Option<&T> {
        self.owners.as_ref()
    }

    pub(crate) fn owners_mut(&mut self) -> Option<&mut T> {
        self.owners.as_mut()
    }

    /// Extracts owners only after the caller has already observed a successful
    /// actual fence, and therefore performs no second fence.
    pub(crate) fn into_fenced(mut self) -> Result<T, BackendError> {
        self.owners
            .take()
            .ok_or_else(|| infer("cold pending owners are missing after the fence"))
    }

    pub(crate) fn settle<F>(mut self, fence: F) -> Result<ColdFenceOutcome<T>, BackendError>
    where
        F: FnOnce() -> Result<(), BackendError>,
    {
        match fence() {
            Ok(()) => {
                let owners = self
                    .owners
                    .take()
                    .ok_or_else(|| infer("cold pending owners are missing after the fence"))?;
                Ok(ColdFenceOutcome::Fenced(owners))
            }
            Err(_) => {
                if let Some(owners) = self.owners.take() {
                    std::mem::forget(owners);
                }
                Ok(ColdFenceOutcome::Quarantined {
                    bytes: self.charged_bytes,
                })
            }
        }
    }

    /// Quarantines owners after an actual fence has already failed. This does
    /// not retry the fence and does not free any owner.
    pub(crate) fn quarantine(mut self) -> ColdFenceOutcome<T> {
        if let Some(owners) = self.owners.take() {
            std::mem::forget(owners);
        }
        ColdFenceOutcome::Quarantined {
            bytes: self.charged_bytes,
        }
    }
}

impl<T> Drop for ColdPending<T> {
    fn drop(&mut self) {
        if let Some(owners) = self.owners.take() {
            std::mem::forget(owners);
        }
    }
}

/// Allocates a stream-ordered zero buffer and stores its typed owner in `slot`
/// before the fallible zero enqueue. Callers hold the enclosing construction
/// guard until one actual fence decides release versus quarantine.
pub(crate) fn allocate_zero_ordered<T: DeviceCopy>(
    stream: &Arc<Stream>,
    len: usize,
    slot: &mut Option<DeviceBuffer<T>>,
) -> Result<(), BackendError> {
    if slot.is_some() {
        return Err(infer("cold zero buffer slot is already occupied"));
    }
    // SAFETY: the private allocation is initialized before any use on this
    // stream; no other owner can observe the undefined contents.
    let buffer = unsafe { DeviceBuffer::<T>::uninitialized_async(stream, len) }.map_err(infer)?;
    *slot = Some(buffer);
    slot.as_mut()
        .ok_or_else(|| infer("cold zero buffer lost its owner"))?
        .zero_async(stream)
        .map_err(infer)
}

pub(crate) struct Payload {
    pub input: DeviceBuffer<u32>,
    pub residual: DeviceBuffer<Bf16>,
    pub projection: DeviceBuffer<Bf16>,
    pub activation: DeviceBuffer<Bf16>,
    pub pooled: DeviceBuffer<f32>,
    pub gdn: GdnPrepared,
    pub block: Option<BlockWork>,
    pub head: Option<WarmHeadScratch>,
    pub prepared: Option<PreparedWork>,
    pub prepared_shape: Option<(usize, usize, usize)>,
    pub layout: Qwen35Layout,
}
pub(crate) struct Qwen35Work {
    stream: Arc<Stream>,
    key: WorkKey,
    bytes: usize,
    last_completion: Option<CompletionTicket>,
    payload_released: bool,
    pub data: ManuallyDrop<Payload>,
}

#[derive(Default)]
struct BuildingData {
    input: Option<DeviceBuffer<u32>>,
    residual: Option<DeviceBuffer<Bf16>>,
    projection: Option<DeviceBuffer<Bf16>>,
    activation: Option<DeviceBuffer<Bf16>>,
    pooled: Option<DeviceBuffer<f32>>,
    gdn: Option<GdnPreparedBuilding>,
    block: Option<BlockWorkBuilding>,
    head: Option<WarmHeadBuilding>,
}
impl BuildingData {
    fn has_owners(&self) -> bool {
        self.input.is_some()
            || self.residual.is_some()
            || self.projection.is_some()
            || self.activation.is_some()
            || self.pooled.is_some()
            || self.gdn.is_some()
            || self.block.is_some()
            || self.head.is_some()
    }
}
struct Building {
    stream: Arc<Stream>,
    pending: ColdPending<BuildingData>,
}
impl Building {
    fn new(stream: Arc<Stream>, reserved_bytes: usize) -> Self {
        Self {
            stream,
            // On an uncertain cold fence, charge the full reserved upper bound
            // even when construction only reached a subset of these owners.
            pending: ColdPending::new(BuildingData::default(), Some(reserved_bytes)),
        }
    }

    fn data_mut(&mut self) -> Result<&mut BuildingData, BackendError> {
        self.pending
            .owners_mut()
            .ok_or_else(|| infer("Missing Qwen3.5 arena constructor"))
    }

    /// On a construction error, one actual fence decides whether the frozen
    /// owners may be released. A failed fence quarantines every owner, charges
    /// the reserved upper bound once, and preserves the original error text.
    /// Authoritative health is the existing parent Engine path: this Err reaches
    /// ExecutableModel::workspace, classify_backend selects Overloaded/Admission
    /// or Fatal, and a Fatal enqueue makes fail_closed set failed + io.close
    /// before the fallback drain. A per-constructor provider poison would be
    /// unobservable once this private builder is quarantined.
    fn settle_error(self, original: BackendError) -> BackendError {
        let Building { stream, pending } = self;
        match stream.synchronize() {
            Ok(()) => match pending.into_fenced() {
                Ok(data) => {
                    drop(data);
                    original
                }
                Err(invariant) => infer(format!("{original}; {invariant}")),
            },
            Err(fence_error) => {
                let allocated = pending.owners_ref().is_some_and(BuildingData::has_owners);
                let bytes = match pending.quarantine() {
                    ColdFenceOutcome::Quarantined { bytes } => bytes,
                    ColdFenceOutcome::Fenced(_) => None,
                };
                if allocated {
                    if let Some(bytes) = bytes {
                        crate::record_uncharged_retained_bytes(bytes);
                    }
                }
                infer(format!(
                    "{original}; Qwen3.5 cold initialization fence failed: {fence_error}"
                ))
            }
        }
    }

    /// The single successful publication fence for a whole new cold Work.
    fn publish(self) -> Result<BuildingData, BackendError> {
        let Building { stream, pending } = self;
        match stream.synchronize() {
            Ok(()) => pending.into_fenced(),
            Err(fence_error) => {
                let bytes = match pending.quarantine() {
                    ColdFenceOutcome::Quarantined { bytes } => bytes,
                    ColdFenceOutcome::Fenced(_) => None,
                };
                if let Some(bytes) = bytes {
                    crate::record_uncharged_retained_bytes(bytes);
                }
                Err(infer(format!(
                    "Qwen3.5 cold publication fence failed: {fence_error}"
                )))
            }
        }
    }
}

impl Qwen35Work {
    fn layout_for(key: WorkKey, model: &Qwen35Model) -> Result<Qwen35Layout, BackendError> {
        if model.block_mode().is_some() {
            Qwen35Layout::new_block(&model.weights.config, key.0)
        } else {
            Qwen35Layout::new(&model.weights.config, key.0)
        }
    }

    pub(crate) fn allocation_bytes(
        key: WorkKey,
        model: &Qwen35Model,
    ) -> Result<usize, BackendError> {
        let layout = Self::layout_for(key, model)?;
        let input = model.input_capacity(key)?;
        let block = BlockWork::allocation_bytes(key, model)?;
        let capacity = GdnCapacity::new(
            key.0,
            key.1
                .checked_sub(1)
                .ok_or_else(|| infer("Missing GDN sequence capacity"))?,
            key.1 - 1,
        )?;
        let scratch = GdnPrepared::scratch_bytes_default_dense(
            model.gdn_geometry,
            capacity,
            model.major,
            model.sm_count,
        )?;
        let head_bytes = model
            .decision_capacity(key)?
            .map_or(Ok(0), |capacity| capacity.device_bytes())?;
        layout
            .slab_bytes
            .checked_add(product(input, 4)?)
            .and_then(|n| {
                n.checked_add(
                    product(product(key.2, model.weights.config.hidden_size).ok()?, 4).ok()?,
                )
            })
            .and_then(|n| n.checked_add(scratch.device_bytes))
            .and_then(|n| n.checked_add(block))
            .and_then(|n| n.checked_add(head_bytes))
            .ok_or_else(|| infer("Qwen3.5 complete arena byte count overflow"))
    }
    fn build_cold(
        building: &mut Building,
        stream: &Arc<Stream>,
        model: &Qwen35Model,
        key: WorkKey,
        layout: Qwen35Layout,
    ) -> Result<(), BackendError> {
        {
            let d = building.data_mut()?;
            allocate_zero_ordered(stream, model.input_capacity(key)?, &mut d.input)?;
            allocate_zero_ordered(stream, layout.hidden_elements, &mut d.residual)?;
            allocate_zero_ordered(stream, layout.projection_elements, &mut d.projection)?;
            allocate_zero_ordered(stream, layout.activation_elements, &mut d.activation)?;
            allocate_zero_ordered(
                stream,
                product(key.2, model.weights.config.hidden_size)?,
                &mut d.pooled,
            )?;
        }
        if model.block_mode().is_some() {
            let block = BlockWorkBuilding::new(model, key)?
                .ok_or_else(|| infer("Qwen3.5 block builder is unavailable"))?;
            building.data_mut()?.block = Some(block);
            BlockWork::build_into(
                building
                    .data_mut()?
                    .block
                    .as_mut()
                    .ok_or_else(|| infer("Qwen3.5 block builder is unavailable"))?,
                stream,
            )?;
        }
        let first = model
            .weights
            .layers
            .iter()
            .find_map(|layer| match &layer.attention {
                crate::qwen35_weights::Qwen35DeviceAttention::Linear { weights, .. } => {
                    Some(Arc::clone(weights))
                }
                _ => None,
            })
            .ok_or_else(|| infer("Dense Qwen3.5 requires recurrent layers"))?;
        let sequences = key
            .1
            .checked_sub(1)
            .ok_or_else(|| infer("Missing Qwen3.5 sequence capacity"))?;
        let capacity = GdnCapacity::new(key.0, sequences, sequences)?;
        let kernels = GdnKernelSet::load_bundle(stream.context(), &model.options.gdn_raw_kernels)?;
        let prepared = GdnPreparedBuilding::new(
            stream,
            model.gdn_geometry,
            capacity,
            model.weights.config.rms_norm_eps,
        );
        building.data_mut()?.gdn = Some(prepared);
        GdnPrepared::build_default_dense_into(
            building
                .data_mut()?
                .gdn
                .as_mut()
                .ok_or_else(|| infer("Qwen3.5 GDN builder is unavailable"))?,
            stream,
            kernels,
            capacity,
            first,
            model.weights.config.rms_norm_eps,
            &model.options.gdn_provider,
            &model.options.gdn_aux_kernels,
        )?;
        if let Some(capacity) = model.decision_capacity(key)? {
            building.data_mut()?.head = Some(WarmHeadBuilding::new(capacity));
            WarmHeadScratch::build_into(
                building
                    .data_mut()?
                    .head
                    .as_mut()
                    .ok_or_else(|| infer("Warm OneJev builder is unavailable"))?,
                stream,
            )?;
        }
        Ok(())
    }

    pub(crate) fn new(
        stream: Arc<Stream>,
        model: &Qwen35Model,
        key: WorkKey,
    ) -> Result<Self, BackendError> {
        let bytes = Self::allocation_bytes(key, model)?;
        let layout = Self::layout_for(key, model)?;
        let mut pending = Building::new(Arc::clone(&stream), bytes);
        if let Err(error) = Self::build_cold(&mut pending, &stream, model, key, layout) {
            return Err(pending.settle_error(error));
        }
        // Cold construction publishes no owner until the one shared fence below.
        let mut d = pending.publish()?;
        let missing = || infer("Qwen3.5 incomplete arena constructor");
        let data = Payload {
            input: d.input.take().ok_or_else(missing)?,
            residual: d.residual.take().ok_or_else(missing)?,
            projection: d.projection.take().ok_or_else(missing)?,
            activation: d.activation.take().ok_or_else(missing)?,
            pooled: d.pooled.take().ok_or_else(missing)?,
            gdn: d.gdn.take().ok_or_else(missing)?.finish()?,
            block: match d.block.take() {
                Some(block) => Some(block.finish()?),
                None => None,
            },
            head: match d.head.take() {
                Some(head) => Some(head.finish()?),
                None => None,
            },
            prepared: None,
            prepared_shape: None,
            layout,
        };
        crate::LIVE_ARENA_BYTES.fetch_add(bytes, Ordering::Relaxed);
        crate::record_live_arena_bytes();
        metrics::counter!("oxide_arena_allocations_total").increment(1);
        Ok(Self {
            stream,
            key,
            bytes,
            last_completion: None,
            payload_released: false,
            data: ManuallyDrop::new(data),
        })
    }
    /// Records the tracked completion generation for this Work's last use. The
    /// engine calls this before the first Qwen3.5 GPU upload.
    pub(crate) fn mark_last_use(&mut self, ticket: CompletionTicket) {
        self.last_completion = Some(ticket);
    }
    pub(crate) fn completion_ticket(&self) -> Option<CompletionTicket> {
        self.last_completion.clone()
    }
    /// Releases an owner without a drain only when its last use is proven done.
    /// A ticket-less Work is still covered by `new`'s successful initialization
    /// fence: every request marks a ticket before its first upload or kernel.
    ///
    /// # Safety
    /// The engine context must already be current. Every request use of this
    /// Work must have been marked before any GPU submission.
    pub(crate) unsafe fn release_if_idle(&mut self) -> Result<bool, BackendError> {
        match self.last_completion.as_ref() {
            None => {
                // SAFETY: construction fenced initialization; no request has
                // ever submitted work against this payload.
                unsafe { self.drop_payload_once() };
                Ok(true)
            }
            Some(ticket) => match ticket.completed() {
                Some(proof) => {
                    // SAFETY: the exact last-use proof is rechecked below.
                    unsafe { self.release_payload_after_completion(&proof)? };
                    Ok(true)
                }
                None => Ok(false),
            },
        }
    }
    /// Releases this Work's owned payload after an exact completion proof.
    ///
    /// # Safety
    /// The proof MUST come from the actual same-stream fence for this Work's
    /// tracked ticket. On any identity mismatch the payload stays owned.
    pub(crate) unsafe fn release_payload_after_completion(
        &mut self,
        proof: &CompletedTicket,
    ) -> Result<(), BackendError> {
        let ticket = self
            .last_completion
            .as_ref()
            .ok_or_else(|| infer("Qwen3.5 Work has no completion ticket"))?;
        let stream_identity = Arc::as_ptr(&self.stream) as usize;
        if !proof.matches(ticket)
            || proof.stream_identity() != stream_identity
            || ticket.stream_identity() != stream_identity
        {
            return Err(infer(
                "Completion proof does not match this Qwen3.5 Work request",
            ));
        }
        if !self.payload_released {
            // SAFETY: the caller proved the actual same-stream fence for this
            // exact ticket, so no borrowed owner is still in use.
            unsafe { self.drop_payload_once() };
        }
        Ok(())
    }
    /// Releases this Work's owned payload after the caller drained the whole
    /// producing stream. A failed drain MUST NOT call this method.
    ///
    /// # Safety
    /// The caller MUST have observed a successful whole-stream drain covering
    /// every use of this Work.
    pub(crate) unsafe fn release_payload_after_drain(&mut self) {
        if !self.payload_released {
            // SAFETY: the caller drained the producing stream for this owner.
            unsafe { self.drop_payload_once() };
        }
    }
    unsafe fn drop_payload_once(&mut self) {
        if self.payload_released {
            return;
        }
        self.payload_released = true;
        // SAFETY: exactly one owner drops the payload, and this method is
        // called only after a completion proof or an actual stream drain.
        unsafe { ManuallyDrop::drop(&mut self.data) };
        crate::LIVE_ARENA_BYTES.fetch_sub(self.bytes, Ordering::Relaxed);
        crate::record_live_arena_bytes();
    }
    pub(crate) fn key(&self) -> WorkKey {
        self.key
    }
    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }
    pub(crate) fn fits(&self, key: WorkKey) -> bool {
        crate::capacity_fits(self.key, key, self.bytes)
    }
    /// Exact input allocation this Work exposes to the outer host-I/O lease.
    pub(crate) fn input_capacity(&self) -> usize {
        self.data.input.len()
    }
    pub(crate) fn block(&self) -> Option<&BlockWork> {
        self.data.block.as_ref()
    }
    pub(crate) fn block_mut(&mut self) -> Option<&mut BlockWork> {
        self.data.block.as_mut()
    }
    pub(crate) fn layout(&self) -> &Qwen35Layout {
        &self.data.layout
    }
    pub(crate) fn stage(
        &self,
        batch: &Batch,
        geometry: crate::output::OutputGeometry,
        lease: &mut HostIoLease,
    ) -> Result<(usize, usize), BackendError> {
        self.stage_with_prefix(batch, geometry, lease, None)
    }
    fn stage_with_prefix(
        &self,
        batch: &Batch,
        geometry: crate::output::OutputGeometry,
        lease: &mut HostIoLease,
        request_prefix: Option<usize>,
    ) -> Result<(usize, usize), BackendError> {
        if geometry.backbone_elements(batch)? > self.data.pooled.len() {
            return Err(infer("Qwen3.5 readout exceeds retained arena"));
        }
        let (input, output) = crate::stage_batch(batch, geometry, lease)?;
        let Some(block) = self.data.block.as_ref() else {
            return Ok((input, output));
        };
        // The outer engine already prepared this lease with the block input
        // capacity; append the checked tail and return the single upload end.
        let elements = block.pack_metadata(lease, batch.input_ids.len(), request_prefix)?;
        Ok((elements, output))
    }
    pub(crate) fn upload(&self, host: &[u32], elements: usize) -> Result<(), BackendError> {
        crate::upload_input(&self.stream, &self.data.input, host, elements)
    }
    pub(crate) fn head(&self) -> Option<&WarmHeadScratch> {
        self.data.head.as_ref()
    }
    pub(crate) fn activation(&self) -> &DeviceBuffer<Bf16> {
        &self.data.activation
    }
    /// Stage the ordinary prefix, the checked block tail for
    /// `request_prefix`, and the warm OneJev metadata tail into the same pinned
    /// input slot. The returned count feeds the single upload.
    pub(crate) fn stage_warm(
        &self,
        batch: &Batch,
        geometry: crate::output::OutputGeometry,
        plan: &WarmDecisionPlan,
        lease: &mut HostIoLease,
        request_prefix: Option<usize>,
    ) -> Result<(usize, usize), BackendError> {
        let (_, output) = self.stage_with_prefix(batch, geometry, lease, request_prefix)?;
        let host = lease.slot_mut()?.input.as_mut_slice();
        let elements = plan.pack_metadata(host)?;
        Ok((elements, output))
    }
    /// Enqueue the warm readout from this Work's actual normalized BF16
    /// activation and the same input buffer that received the metadata tail.
    ///
    /// # Safety
    /// The enclosing engine MUST keep the bound context/stream, this Work, the
    /// model head and every upload alive until its completion boundary drains
    /// or the whole Work is quarantined.
    pub(crate) unsafe fn enqueue_warm(
        &mut self,
        head: &OneJevGpuHead,
        blas: &crate::Blas,
        stream: &Stream,
        plan: &WarmDecisionPlan,
        host_logits: &mut [f32],
    ) -> Result<(), DecisionGpuError> {
        let data = &mut *self.data;
        let scratch = data
            .head
            .as_mut()
            .ok_or(DecisionGpuError::PreparedMismatch(
                "this Work has no warm OneJev scratch",
            ))?;
        // SAFETY: the caller's outer Work/model ownership covers completion.
        unsafe {
            crate::warm_head::enqueue_warm_decision(
                head,
                blas,
                stream,
                plan,
                scratch,
                &data.activation,
                &data.input,
                host_logits,
            )
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn finish(
        &mut self,
        dense: Option<(&mut crate::DenseHead, &mut crate::DenseWork)>,
        lt: &crate::LtProvider,
        blas: Option<&crate::Blas>,
        pooled_rows: usize,
        raw_rows: usize,
        width: usize,
        lease: &mut HostIoLease,
    ) -> Result<(), BackendError> {
        crate::finish_pooled(
            dense,
            lt,
            blas,
            &self.stream,
            &self.data.pooled,
            pooled_rows,
            raw_rows,
            width,
            lease,
        )
    }
}
impl Payload {
    pub(crate) fn slabs(&self) -> Slabs<'_> {
        Slabs {
            residual: &self.residual,
            projection: &self.projection,
            activation: &self.activation,
        }
    }
}
impl Drop for Qwen35Work {
    fn drop(&mut self) {
        if self.payload_released {
            return;
        }
        #[cfg(test)]
        crate::note_qwen35_work_default_drop_drain();
        if self.stream.synchronize().is_ok() {
            // SAFETY: the entire request and its borrowed layer uses drained.
            unsafe { self.drop_payload_once() };
        } else {
            metrics::counter!("oxide_arena_retained_bytes_total").increment(self.bytes as u64);
        }
    }
}

#[cfg(test)]
mod cold_zero_tests {
    use super::*;
    use std::cell::Cell;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    struct FakeOwner {
        drops: Arc<AtomicUsize>,
    }

    impl FakeOwner {
        fn new(drops: Arc<AtomicUsize>) -> Self {
            Self { drops }
        }
    }

    impl Drop for FakeOwner {
        fn drop(&mut self) {
            self.drops.fetch_add(1, AtomicOrdering::SeqCst);
        }
    }

    fn counted_fence(
        calls: &Cell<usize>,
        fail: bool,
    ) -> impl FnOnce() -> Result<(), BackendError> + '_ {
        move || {
            calls.set(calls.get() + 1);
            if fail {
                Err(infer("injected fence failure"))
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn partial_zero_failure_releases_only_after_a_successful_fence() {
        let drops = Arc::new(AtomicUsize::new(0));
        let fence_calls = Cell::new(0);
        // The first owner represents buffers already allocated and zeroed; the
        // second zero enqueue fails and keeps the whole owner set pending.
        let owner = FakeOwner::new(Arc::clone(&drops));
        let failed_zero: Result<(), BackendError> =
            Err(infer("injected partial zero enqueue failure"));
        assert!(failed_zero.is_err());
        let outcome =
            match ColdPending::new(owner, Some(4_096)).settle(counted_fence(&fence_calls, false)) {
                Ok(outcome) => outcome,
                Err(error) => panic!("cold settlement lost its owners: {error}"),
            };
        let ColdFenceOutcome::Fenced(owner) = outcome else {
            panic!("a successful fence must release owners");
        };
        assert_eq!(fence_calls.get(), 1);
        assert_eq!(drops.load(AtomicOrdering::SeqCst), 0);
        drop(owner);
        assert_eq!(drops.load(AtomicOrdering::SeqCst), 1);
    }

    #[test]
    fn failed_fence_retains_every_owner_and_reports_reserved_bytes() {
        let drops = Arc::new(AtomicUsize::new(0));
        let fence_calls = Cell::new(0);
        let outcome = match ColdPending::new(
            (
                FakeOwner::new(Arc::clone(&drops)),
                FakeOwner::new(Arc::clone(&drops)),
            ),
            Some(12_345),
        )
        .settle(counted_fence(&fence_calls, true))
        {
            Ok(outcome) => outcome,
            Err(error) => panic!("cold settlement lost its owners: {error}"),
        };

        match outcome {
            ColdFenceOutcome::Quarantined { bytes } => assert_eq!(bytes, Some(12_345)),
            ColdFenceOutcome::Fenced(_) => panic!("a failed fence must not release owners"),
        }
        assert_eq!(fence_calls.get(), 1, "settlement fences exactly once");
        assert_eq!(
            drops.load(AtomicOrdering::SeqCst),
            0,
            "uncertain owners must not be freed"
        );
    }

    #[test]
    fn already_failed_fence_is_never_retried_and_does_not_double_charge() {
        let drops = Arc::new(AtomicUsize::new(0));
        let fence_calls = Cell::new(0);
        let pending = ColdPending::new(FakeOwner::new(Arc::clone(&drops)), Some(777));
        // Simulate the caller having observed a failed fence already.
        let outcome = pending.quarantine();
        assert_eq!(fence_calls.get(), 0, "quarantine must not retry the fence");
        match outcome {
            ColdFenceOutcome::Quarantined { bytes } => assert_eq!(bytes, Some(777)),
            ColdFenceOutcome::Fenced(_) => panic!("quarantine cannot release owners"),
        }
        assert_eq!(drops.load(AtomicOrdering::SeqCst), 0);
    }

    #[test]
    fn outer_owner_extraction_does_not_add_another_fence_or_drop() {
        let drops = Arc::new(AtomicUsize::new(0));
        let pending = ColdPending::new(FakeOwner::new(Arc::clone(&drops)), Some(1));
        let owner = match pending.into_fenced() {
            Ok(owner) => owner,
            Err(error) => panic!("cold owners vanished after the fence: {error}"),
        };
        assert_eq!(drops.load(AtomicOrdering::SeqCst), 0);
        drop(owner);
        assert_eq!(drops.load(AtomicOrdering::SeqCst), 1);
    }

    #[test]
    fn unsettled_pending_drop_quarantines_instead_of_freeing() {
        let drops = Arc::new(AtomicUsize::new(0));
        {
            let _pending = ColdPending::new(FakeOwner::new(Arc::clone(&drops)), Some(9));
        }
        assert_eq!(
            drops.load(AtomicOrdering::SeqCst),
            0,
            "an unfenced owner must never be freed by Drop"
        );
    }
}
