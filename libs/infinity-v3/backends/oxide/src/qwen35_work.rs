// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! One arena owns three phase slabs and one recurrent scratch pipeline.
use crate::decision_head_gpu::{DecisionGpuError, OneJevGpuHead};
use crate::gdn_prepared::{GdnCapacity, GdnKernelSet, GdnPrepared};
use crate::qwen35_block_work::BlockWork;
use crate::qwen35_execute::{PreparedWork, Slabs};
use crate::qwen35_layout::Qwen35Layout;
use crate::qwen35_runtime::Qwen35Model;
use crate::tensor::Bf16;
use crate::warm_head::{WarmDecisionPlan, WarmHeadScratch};
use crate::work_retirement::{CompletedTicket, CompletionTicket};
use crate::{infer, product, HostIoLease, WorkKey};
use cuda_core::{DeviceBuffer, Stream};
use infinity_v3_backend_core::{BackendError, Batch};
use std::{
    mem::ManuallyDrop,
    sync::{atomic::Ordering, Arc},
};

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
    gdn: Option<GdnPrepared>,
    block: Option<BlockWork>,
    head: Option<WarmHeadScratch>,
}
struct Building {
    stream: Arc<Stream>,
    data: Option<BuildingData>,
    // On an uncertain cold fence, charge the full reserved upper bound even
    // when construction only reached a subset of these allocations.
    reserved_bytes: usize,
}
impl Drop for Building {
    fn drop(&mut self) {
        if let Some(data) = self.data.take() {
            if self.stream.synchronize().is_err() {
                let allocated = data.input.is_some()
                    || data.residual.is_some()
                    || data.projection.is_some()
                    || data.activation.is_some()
                    || data.pooled.is_some()
                    || data.gdn.is_some()
                    || data.block.is_some()
                    || data.head.is_some();
                if allocated {
                    crate::record_uncharged_retained_bytes(self.reserved_bytes);
                }
                std::mem::forget((data, Arc::clone(&self.stream)));
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
    pub(crate) fn new(
        stream: Arc<Stream>,
        model: &Qwen35Model,
        key: WorkKey,
    ) -> Result<Self, BackendError> {
        let bytes = Self::allocation_bytes(key, model)?;
        let layout = Self::layout_for(key, model)?;
        let mut pending = Building {
            stream: Arc::clone(&stream),
            data: Some(BuildingData::default()),
            reserved_bytes: bytes,
        };
        let d = pending
            .data
            .as_mut()
            .ok_or_else(|| infer("Missing Qwen3.5 arena constructor"))?;
        d.input = Some(DeviceBuffer::zeroed(&stream, model.input_capacity(key)?).map_err(infer)?);
        d.residual = Some(DeviceBuffer::zeroed(&stream, layout.hidden_elements).map_err(infer)?);
        d.projection =
            Some(DeviceBuffer::zeroed(&stream, layout.projection_elements).map_err(infer)?);
        d.activation =
            Some(DeviceBuffer::zeroed(&stream, layout.activation_elements).map_err(infer)?);
        d.pooled = Some(
            DeviceBuffer::zeroed(&stream, product(key.2, model.weights.config.hidden_size)?)
                .map_err(infer)?,
        );
        d.block = BlockWork::new(&stream, model, key)?;
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
        let kernels = GdnKernelSet::load_bundle(stream.context(), &model.options.gdn_raw_kernels)?;
        let sequences = key
            .1
            .checked_sub(1)
            .ok_or_else(|| infer("Missing Qwen3.5 sequence capacity"))?;
        d.gdn = Some(GdnPrepared::new_default_dense_with_device_weights(
            &stream,
            kernels,
            GdnCapacity::new(key.0, sequences, sequences)?,
            first,
            model.weights.config.rms_norm_eps,
            &model.options.gdn_provider,
            &model.options.gdn_aux_kernels,
        )?);
        d.head = match model.decision_capacity(key)? {
            Some(capacity) => Some(WarmHeadScratch::new(&stream, capacity)?),
            None => None,
        };
        // Cold construction publishes no owner until every initialization finished.
        stream.synchronize().map_err(infer)?;
        let mut d = pending
            .data
            .take()
            .ok_or_else(|| infer("Qwen3.5 constructor lost owners"))?;
        let missing = || infer("Qwen3.5 incomplete arena constructor");
        let data = Payload {
            input: d.input.take().ok_or_else(missing)?,
            residual: d.residual.take().ok_or_else(missing)?,
            projection: d.projection.take().ok_or_else(missing)?,
            activation: d.activation.take().ok_or_else(missing)?,
            pooled: d.pooled.take().ok_or_else(missing)?,
            gdn: d.gdn.take().ok_or_else(missing)?,
            block: d.block.take(),
            head: d.head.take(),
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
