// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Engine-bound D3 / Perplexity external readout over the shared Work arena.
//!
//! The standalone `ExternalGpuHead` owns the cold sidecar and weight upload.
//! This module is the actual served path: it reuses one cached `Qwen35Work` for
//! every request, binds the exact normalized BF16 activation span into a
//! capacity-backed external scratch, appends row ids and counts to the single
//! HostIo input tail, and enqueues the Native projection/selection recipe.

use crate::{
    cublas::Blas,
    decision_external::{
        plan_external_requests, request_output_count, ExternalFamily, ExternalPlan,
        ExternalRequestPlan,
    },
    decision_external_gpu::{
        build_work, element_pointer, launch_cast_rows, ExternalGpuError, ExternalGpuHead,
        Selection, WorkPlan,
    },
    decision_head::PackedDecisionBatch,
    decision_head_gpu::GpuHiddenRows,
    infer,
    tensor::Bf16,
    WorkKey,
};
use cuda_core::{DeviceBuffer, Stream};
use infinity_v3_backend_core::{BackendError, DecisionInput, DecisionOutput};
use std::{mem::ManuallyDrop, sync::Arc};

/// Router-permitted external logits per request and total.
pub(crate) const MAX_EXTERNAL_ROWS: usize = 512;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ExternalHeadCapacity {
    family: ExternalFamily,
    requests: usize,
    rows: usize,
    hidden: usize,
    readout_rows: usize,
}

impl ExternalHeadCapacity {
    pub(crate) fn new(
        family: ExternalFamily,
        requests: usize,
        hidden: usize,
        readout_rows: usize,
    ) -> Result<Self, BackendError> {
        if requests == 0 {
            return Err(infer(
                "Warm external capacity requires at least one request",
            ));
        }
        if hidden == 0 || readout_rows == 0 {
            return Err(infer("Warm external capacity requires nonzero widths"));
        }
        Ok(Self {
            family,
            requests,
            rows: MAX_EXTERNAL_ROWS,
            hidden,
            readout_rows,
        })
    }

    pub(crate) fn family(&self) -> ExternalFamily {
        self.family
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

    pub(crate) fn readout_rows(&self) -> usize {
        self.readout_rows
    }

    fn hidden_elements(&self) -> Result<usize, BackendError> {
        if self.family == ExternalFamily::D3F32 {
            self.requests
                .checked_mul(self.hidden)
                .ok_or_else(|| infer("Warm external hidden scratch overflow"))
        } else {
            Ok(0)
        }
    }

    fn marker_elements(&self) -> Result<usize, BackendError> {
        if self.family == ExternalFamily::PplxBf16 {
            self.rows
                .checked_mul(self.hidden)
                .ok_or_else(|| infer("Warm external marker scratch overflow"))
        } else {
            Ok(0)
        }
    }

    fn full_elements(&self) -> Result<usize, BackendError> {
        self.rows
            .checked_mul(self.readout_rows)
            .ok_or_else(|| infer("Warm external full-logit scratch overflow"))
    }

    pub(crate) fn bf16_full_elements(&self) -> Result<usize, BackendError> {
        if self.family == ExternalFamily::PplxBf16 {
            self.full_elements()
        } else {
            Ok(0)
        }
    }

    pub(crate) fn bf16_logit_elements(&self) -> Result<usize, BackendError> {
        if self.family == ExternalFamily::PplxBf16 {
            Ok(self.rows)
        } else {
            Ok(0)
        }
    }

    pub(crate) fn device_bytes(&self) -> Result<usize, BackendError> {
        self.hidden_elements()?
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|bytes| {
                self.marker_elements()
                    .ok()?
                    .checked_mul(std::mem::size_of::<Bf16>())
                    .and_then(|markers| bytes.checked_add(markers))
            })
            .and_then(|bytes| {
                self.full_elements()
                    .ok()?
                    .checked_mul(std::mem::size_of::<f32>())
                    .and_then(|full| bytes.checked_add(full))
            })
            .and_then(|bytes| {
                self.rows
                    .checked_mul(std::mem::size_of::<f32>())
                    .and_then(|logits| bytes.checked_add(logits))
            })
            .and_then(|bytes| {
                self.bf16_full_elements()
                    .ok()?
                    .checked_mul(std::mem::size_of::<Bf16>())
                    .and_then(|full| bytes.checked_add(full))
            })
            .and_then(|bytes| {
                self.bf16_logit_elements()
                    .ok()?
                    .checked_mul(std::mem::size_of::<Bf16>())
                    .and_then(|logits| bytes.checked_add(logits))
            })
            .ok_or_else(|| infer("Warm external scratch byte count overflow"))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ExternalScratchBases {
    pub(crate) hidden: u64,
    pub(crate) markers: u64,
    pub(crate) full_logits: u64,
    pub(crate) full_logits_bf16: u64,
    pub(crate) logits: u64,
    pub(crate) logits_bf16: u64,
}

struct ScratchBuilding<'a> {
    stream: &'a Arc<Stream>,
    hidden: Option<DeviceBuffer<f32>>,
    markers: Option<DeviceBuffer<Bf16>>,
    full_logits: Option<DeviceBuffer<f32>>,
    full_logits_bf16: Option<DeviceBuffer<Bf16>>,
    logits: Option<DeviceBuffer<f32>>,
    logits_bf16: Option<DeviceBuffer<Bf16>>,
}

impl<'a> ScratchBuilding<'a> {
    fn new(stream: &'a Arc<Stream>) -> Self {
        Self {
            stream,
            hidden: None,
            markers: None,
            full_logits: None,
            full_logits_bf16: None,
            logits: None,
            logits_bf16: None,
        }
    }

    fn take(&mut self) -> Option<ExternalScratch> {
        Some(ExternalScratch {
            hidden: self.hidden.take().map(ManuallyDrop::new),
            markers: self.markers.take().map(ManuallyDrop::new),
            full_logits: ManuallyDrop::new(self.full_logits.take()?),
            full_logits_bf16: self.full_logits_bf16.take().map(ManuallyDrop::new),
            logits: ManuallyDrop::new(self.logits.take()?),
            logits_bf16: self.logits_bf16.take().map(ManuallyDrop::new),
        })
    }
}

impl Drop for ScratchBuilding<'_> {
    fn drop(&mut self) {
        let allocated = self.hidden.is_some()
            || self.markers.is_some()
            || self.full_logits.is_some()
            || self.full_logits_bf16.is_some()
            || self.logits.is_some()
            || self.logits_bf16.is_some();
        if !allocated {
            return;
        }
        if self.stream.synchronize().is_ok() {
            self.hidden.take();
            self.markers.take();
            self.full_logits.take();
            self.full_logits_bf16.take();
            self.logits.take();
            self.logits_bf16.take();
        } else {
            std::mem::forget((
                self.hidden.take(),
                self.markers.take(),
                self.full_logits.take(),
                self.full_logits_bf16.take(),
                self.logits.take(),
                self.logits_bf16.take(),
                Arc::clone(self.stream),
            ));
        }
    }
}

/// Capacity-backed external scratch owned by one reusable `Qwen35Work`.
pub(crate) struct ExternalScratch {
    hidden: Option<ManuallyDrop<DeviceBuffer<f32>>>,
    markers: Option<ManuallyDrop<DeviceBuffer<Bf16>>>,
    full_logits: ManuallyDrop<DeviceBuffer<f32>>,
    full_logits_bf16: Option<ManuallyDrop<DeviceBuffer<Bf16>>>,
    logits: ManuallyDrop<DeviceBuffer<f32>>,
    logits_bf16: Option<ManuallyDrop<DeviceBuffer<Bf16>>>,
}

impl ExternalScratch {
    pub(crate) fn new(
        stream: &Arc<Stream>,
        capacity: ExternalHeadCapacity,
    ) -> Result<Self, BackendError> {
        let mut pending = ScratchBuilding::new(stream);
        if capacity.hidden_elements()? > 0 {
            pending.hidden =
                Some(DeviceBuffer::zeroed(stream, capacity.hidden_elements()?).map_err(infer)?);
        }
        if capacity.marker_elements()? > 0 {
            pending.markers =
                Some(DeviceBuffer::zeroed(stream, capacity.marker_elements()?).map_err(infer)?);
        }
        pending.full_logits =
            Some(DeviceBuffer::zeroed(stream, capacity.full_elements()?).map_err(infer)?);
        if capacity.bf16_full_elements()? > 0 {
            pending.full_logits_bf16 =
                Some(DeviceBuffer::zeroed(stream, capacity.bf16_full_elements()?).map_err(infer)?);
        }
        pending.logits = Some(DeviceBuffer::zeroed(stream, capacity.rows).map_err(infer)?);
        if capacity.bf16_logit_elements()? > 0 {
            pending.logits_bf16 =
                Some(DeviceBuffer::zeroed(stream, capacity.bf16_logit_elements()?).map_err(infer)?);
        }
        stream.synchronize().map_err(infer)?;
        pending
            .take()
            .ok_or_else(|| infer("Warm external scratch constructor lost owners"))
    }

    pub(crate) fn bases(&self) -> ExternalScratchBases {
        ExternalScratchBases {
            hidden: self
                .hidden
                .as_ref()
                .map_or(0, |buffer| buffer.cu_deviceptr()),
            markers: self
                .markers
                .as_ref()
                .map_or(0, |buffer| buffer.cu_deviceptr()),
            full_logits: self.full_logits.cu_deviceptr(),
            full_logits_bf16: self
                .full_logits_bf16
                .as_ref()
                .map_or(0, |buffer| buffer.cu_deviceptr()),
            logits: self.logits.cu_deviceptr(),
            logits_bf16: self
                .logits_bf16
                .as_ref()
                .map_or(0, |buffer| buffer.cu_deviceptr()),
        }
    }

    pub(crate) fn hidden_len(&self) -> usize {
        self.hidden.as_ref().map_or(0, |buffer| buffer.len())
    }

    pub(crate) fn marker_len(&self) -> usize {
        self.markers.as_ref().map_or(0, |buffer| buffer.len())
    }

    pub(crate) fn full_len(&self) -> usize {
        self.full_logits.len()
    }

    pub(crate) fn full_bf16_len(&self) -> usize {
        self.full_logits_bf16
            .as_ref()
            .map_or(0, |buffer| buffer.len())
    }

    pub(crate) fn logits_len(&self) -> usize {
        self.logits.len()
    }

    pub(crate) fn logits_bf16_len(&self) -> usize {
        self.logits_bf16.as_ref().map_or(0, |buffer| buffer.len())
    }
}

impl Drop for ExternalScratch {
    fn drop(&mut self) {
        // SAFETY: the owning Work drains the stream before dropping Payload.
        unsafe {
            if let Some(mut buffer) = self.hidden.take() {
                ManuallyDrop::drop(&mut buffer);
            }
            if let Some(mut buffer) = self.markers.take() {
                ManuallyDrop::drop(&mut buffer);
            }
            ManuallyDrop::drop(&mut self.full_logits);
            if let Some(mut buffer) = self.full_logits_bf16.take() {
                ManuallyDrop::drop(&mut buffer);
            }
            ManuallyDrop::drop(&mut self.logits);
            if let Some(mut buffer) = self.logits_bf16.take() {
                ManuallyDrop::drop(&mut buffer);
            }
        }
    }
}

/// Fixed external metadata tail after the existing packed input layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ExternalInputLayout {
    metadata_base: usize,
    device_rows: usize,
    counts: usize,
    end: usize,
}

impl ExternalInputLayout {
    pub(crate) fn with_reserved_prefix(
        key: WorkKey,
        reserved_words: usize,
    ) -> Result<Self, BackendError> {
        let metadata_base = crate::staging::input_capacity(key)?
            .checked_add(reserved_words)
            .ok_or_else(|| infer("Warm external metadata base overflow"))?;
        let device_rows = metadata_base;
        let counts = device_rows
            .checked_add(MAX_EXTERNAL_ROWS)
            .ok_or_else(|| infer("Warm external row slot overflow"))?;
        let end = counts
            .checked_add(MAX_EXTERNAL_ROWS)
            .ok_or_else(|| infer("Warm external count slot overflow"))?;
        Ok(Self {
            metadata_base,
            device_rows,
            counts,
            end,
        })
    }

    pub(crate) fn metadata_base(&self) -> usize {
        self.metadata_base
    }

    pub(crate) fn device_rows(&self) -> usize {
        self.device_rows
    }

    pub(crate) fn counts(&self) -> usize {
        self.counts
    }

    pub(crate) fn end(&self) -> usize {
        self.end
    }
}

pub(crate) fn external_metadata_words(_sequences_capacity: usize) -> Result<usize, BackendError> {
    MAX_EXTERNAL_ROWS
        .checked_mul(2)
        .ok_or_else(|| infer("Warm external metadata capacity overflow"))
}

pub(crate) struct ExternalWarmMetadata {
    family: ExternalFamily,
    layout: ExternalInputLayout,
    key: WorkKey,
    tokens: usize,
    capacity: ExternalHeadCapacity,
    plan: ExternalPlan,
}

impl ExternalWarmMetadata {
    pub(crate) fn layout(&self) -> ExternalInputLayout {
        self.layout
    }

    pub(crate) fn key(&self) -> WorkKey {
        self.key
    }

    pub(crate) fn total_rows(&self) -> usize {
        self.plan.total
    }
}

pub(crate) struct ExternalWarmPlan {
    family: ExternalFamily,
    layout: ExternalInputLayout,
    tokens: usize,
    hidden: usize,
    rows: usize,
    request_count: usize,
    plan: ExternalPlan,
    work: WorkPlan,
    tail_rows: Vec<u32>,
    tail_counts: Vec<u32>,
}

impl ExternalWarmPlan {
    pub(crate) fn layout(&self) -> ExternalInputLayout {
        self.layout
    }

    pub(crate) fn request_count(&self) -> usize {
        self.request_count
    }

    pub(crate) fn total_rows(&self) -> usize {
        self.plan.total
    }

    pub(crate) fn hidden(&self) -> usize {
        self.hidden
    }

    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    pub(crate) fn family(&self) -> ExternalFamily {
        self.family
    }

    pub(crate) fn counts(&self) -> &[usize] {
        &self.plan.counts
    }

    pub(crate) fn row_offsets(&self) -> &[usize] {
        &self.plan.row_offsets
    }

    pub(crate) fn tokens(&self) -> usize {
        self.tokens
    }

    /// Append the selected/marker row ids and their counts to the already packed
    /// single HostIo input slice. GPU enqueue still uses the CPU plan records;
    /// the device tail is the shared request metadata surface and keeps the
    /// external path on the same single-upload contract as OneJev.
    pub(crate) fn pack_metadata(&self, input: &mut [u32]) -> Result<usize, BackendError> {
        let layout = self.layout;
        if input.len() < layout.end() {
            return Err(infer(
                "Warm external metadata exceeds the retained input staging capacity",
            ));
        }
        let row_end = layout
            .device_rows()
            .checked_add(self.tail_rows.len())
            .ok_or_else(|| infer("Warm external row staging overflow"))?;
        if row_end > layout.counts() {
            return Err(infer("Warm external rows exceed their staging slot"));
        }
        input
            .get_mut(layout.device_rows()..row_end)
            .ok_or_else(|| infer("Warm external row slot is unavailable"))?
            .copy_from_slice(&self.tail_rows);
        let count_end = layout
            .counts()
            .checked_add(self.tail_counts.len())
            .ok_or_else(|| infer("Warm external count staging overflow"))?;
        if count_end > layout.end() {
            return Err(infer("Warm external counts exceed their staging slot"));
        }
        input
            .get_mut(layout.counts()..count_end)
            .ok_or_else(|| infer("Warm external count slot is unavailable"))?
            .copy_from_slice(&self.tail_counts);
        Ok(layout.end())
    }
}

fn tail_rows(family: ExternalFamily, plan: &ExternalPlan) -> Vec<u32> {
    if family == ExternalFamily::D3F32 {
        plan.requests
            .iter()
            .filter_map(|request| match request {
                ExternalRequestPlan::LastToken { row, .. } => Some(*row),
                ExternalRequestPlan::Inline { .. } => None,
            })
            .collect()
    } else {
        plan.marker_rows.clone()
    }
}

fn tail_counts(family: ExternalFamily, plan: &ExternalPlan) -> Vec<u32> {
    if family == ExternalFamily::PplxBf16
        && plan
            .requests
            .iter()
            .any(|request| matches!(request, ExternalRequestPlan::Inline { .. }))
    {
        plan.requests
            .iter()
            .flat_map(|request| match request {
                ExternalRequestPlan::Inline { counts, .. } => counts
                    .iter()
                    .map(|&count| u32::try_from(count).unwrap_or(u32::MAX))
                    .collect::<Vec<_>>(),
                ExternalRequestPlan::LastToken { count, .. } => {
                    vec![u32::try_from(*count).unwrap_or(u32::MAX)]
                }
            })
            .collect()
    } else {
        plan.counts
            .iter()
            .map(|&count| u32::try_from(count).unwrap_or(u32::MAX))
            .collect()
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_external_warm_metadata(
    family: ExternalFamily,
    capacity: ExternalHeadCapacity,
    key: WorkKey,
    tokens: usize,
    batch: &PackedDecisionBatch<'_>,
    requests: &[DecisionInput],
    reserved_words: usize,
) -> Result<ExternalWarmMetadata, BackendError> {
    if family != capacity.family() {
        return Err(infer(
            "Warm external head family differs from the Work scratch capacity",
        ));
    }
    if tokens == 0 || batch.tokens() != tokens {
        return Err(infer(
            "Warm external activation rows do not match the packed decision batch",
        ));
    }
    let plan = plan_external_requests(family, capacity.readout_rows(), batch, requests)
        .map_err(|error| infer(error.to_string()))?;
    if plan.counts.is_empty() {
        return Err(infer("Warm external batch must contain a request"));
    }
    if plan.counts.len() > capacity.requests() {
        return Err(infer(
            "Warm external request count exceeds the retained Work sequence capacity",
        ));
    }
    if plan.total == 0 || plan.total > capacity.rows() {
        return Err(infer(
            "Warm external output count exceeds the retained row capacity",
        ));
    }
    if plan.marker_rows.len() > capacity.rows() || plan.full_rows > capacity.rows() {
        return Err(infer(
            "Warm external marker count exceeds the retained row capacity",
        ));
    }
    for (index, request) in requests.iter().enumerate() {
        let expected = request_output_count(family, capacity.readout_rows(), index, request)
            .map_err(|error| infer(error.to_string()))?;
        if plan.counts.get(index) != Some(&expected) {
            return Err(infer(
                "Warm external rebinding changed a request output count",
            ));
        }
    }
    let layout = ExternalInputLayout::with_reserved_prefix(key, reserved_words)?;
    Ok(ExternalWarmMetadata {
        family,
        layout,
        key,
        tokens,
        capacity,
        plan,
    })
}

pub(crate) fn bind_external_warm_metadata(
    metadata: ExternalWarmMetadata,
) -> Result<ExternalWarmPlan, BackendError> {
    let ExternalWarmMetadata {
        family,
        layout,
        key: _,
        tokens,
        capacity,
        plan,
    } = metadata;
    let work = build_work(family, capacity.hidden(), capacity.readout_rows(), &plan)
        .map_err(|error| infer(error.to_string()))?;
    let request_count = plan.counts.len();
    let tail_rows = tail_rows(family, &plan);
    let tail_counts = tail_counts(family, &plan);
    if tail_rows.len() > MAX_EXTERNAL_ROWS || tail_counts.len() > MAX_EXTERNAL_ROWS {
        return Err(infer("Warm external metadata tail exceeds its fixed slot"));
    }
    Ok(ExternalWarmPlan {
        family,
        layout,
        tokens,
        hidden: capacity.hidden(),
        rows: capacity.readout_rows(),
        request_count,
        plan,
        work,
        tail_rows,
        tail_counts,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn plan_external_warm(
    family: ExternalFamily,
    capacity: ExternalHeadCapacity,
    key: WorkKey,
    tokens: usize,
    batch: &PackedDecisionBatch<'_>,
    requests: &[DecisionInput],
    reserved_words: usize,
) -> Result<ExternalWarmPlan, BackendError> {
    let metadata = prepare_external_warm_metadata(
        family,
        capacity,
        key,
        tokens,
        batch,
        requests,
        reserved_words,
    )?;
    bind_external_warm_metadata(metadata)
}

fn metadata_pointer(
    input: &DeviceBuffer<u32>,
    element: usize,
    elements: usize,
    what: &'static str,
) -> Result<u64, ExternalGpuError> {
    if elements == 0 {
        return Err(ExternalGpuError::Geometry(what));
    }
    let end = element
        .checked_add(elements)
        .ok_or(ExternalGpuError::Geometry(what))?;
    if end > input.len() {
        return Err(ExternalGpuError::Geometry(what));
    }
    let bytes = u64::try_from(
        element
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or(ExternalGpuError::Geometry(what))?,
    )
    .map_err(|_| ExternalGpuError::Geometry(what))?;
    input
        .cu_deviceptr()
        .checked_add(bytes)
        .ok_or(ExternalGpuError::Geometry(what))
}

/// Enqueue the prepared external readout from the Work's actual normalized
/// BF16 activation span. No allocation, context bind or checkpoint read occurs.
///
/// # Safety
/// The outer engine keeps the model, Work, scratch, input upload, head weight
/// and stream alive until its completion boundary drains or quarantines them.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn enqueue_external_warm(
    head: &ExternalGpuHead,
    blas: &Blas,
    stream: &Stream,
    plan: &ExternalWarmPlan,
    scratch: &mut ExternalScratch,
    activation: &DeviceBuffer<Bf16>,
    input: &DeviceBuffer<u32>,
    host_logits: &mut [f32],
) -> Result<(), ExternalGpuError> {
    if plan.family != head.family() || plan.hidden != head.hidden_size() || plan.rows != head.rows()
    {
        return Err(ExternalGpuError::PreparedMismatch(
            "warm external plan differs from the attached head",
        ));
    }
    let hidden =
        plan.request_count()
            .checked_mul(plan.hidden)
            .ok_or(ExternalGpuError::Geometry(
                "external hidden scratch overflow",
            ))?;
    let markers =
        plan.work
            .marker_rows
            .len()
            .checked_mul(plan.hidden)
            .ok_or(ExternalGpuError::Geometry(
                "external marker scratch overflow",
            ))?;
    let full = plan
        .plan
        .full_rows
        .checked_mul(plan.rows)
        .ok_or(ExternalGpuError::Geometry(
            "external full-logit scratch overflow",
        ))?;
    let typed = plan.family == ExternalFamily::PplxBf16;
    if scratch.hidden_len() < hidden
        || scratch.marker_len() < markers
        || scratch.full_len() < full
        || scratch.logits_len() < plan.total_rows()
        || (typed && scratch.full_bf16_len() < full)
        || (typed && scratch.logits_bf16_len() < plan.total_rows())
        || host_logits.len() < plan.total_rows()
    {
        return Err(ExternalGpuError::PreparedMismatch(
            "warm external plan exceeds the retained Work scratch capacity",
        ));
    }
    if plan.work.gemms.is_empty() || plan.request_count != plan.plan.counts.len() {
        return Err(ExternalGpuError::PreparedMismatch(
            "warm external plan metadata is internally inconsistent",
        ));
    }
    let _rows = GpuHiddenRows::new(activation, plan.tokens, plan.hidden)?;
    let bases = scratch.bases();
    if plan.family == ExternalFamily::D3F32 {
        if plan.work.cast_rows.is_empty() || scratch.hidden_len() < hidden {
            return Err(ExternalGpuError::PreparedMismatch(
                "warm D3 plan lost its cast scratch",
            ));
        }
        let rows = metadata_pointer(
            input,
            plan.layout.device_rows(),
            plan.work.cast_rows.len(),
            "warm D3 cast rows exceed the retained input",
        )?;
        launch_cast_rows(
            head.cast_kernel(),
            stream,
            activation.cu_deviceptr(),
            bases.hidden,
            rows,
            plan.work.cast_rows.len(),
            plan.hidden,
        )?;
    } else if !plan.work.marker_rows.is_empty() {
        if scratch.marker_len() < markers {
            return Err(ExternalGpuError::PreparedMismatch(
                "warm Pplx plan lost its marker scratch",
            ));
        }
        let bytes = plan
            .hidden
            .checked_mul(std::mem::size_of::<Bf16>())
            .ok_or(ExternalGpuError::Geometry("external marker copy overflow"))?;
        for (destination, &source_row) in plan.work.marker_rows.iter().enumerate() {
            let source = element_pointer(
                activation.cu_deviceptr(),
                usize::try_from(source_row)
                    .map_err(|_| ExternalGpuError::Geometry("external marker row overflow"))?,
                plan.hidden,
                std::mem::size_of::<Bf16>(),
            )?;
            let destination = element_pointer(
                bases.markers,
                destination,
                plan.hidden,
                std::mem::size_of::<Bf16>(),
            )?;
            // SAFETY: both rows are owned by retained buffers.
            cuda_core::memory::memcpy_dtod_async(destination, source, bytes, stream.cu_stream())
                .map_err(|error| ExternalGpuError::Driver(error.to_string()))?;
        }
    }
    let f32_weight = head.f32_weight_pointer();
    let bf16_weight = head.bf16_weight_pointer();
    for record in &plan.work.gemms {
        match record {
            crate::decision_external_gpu::GemmRecord::F32Full {
                plan: linear,
                source_row,
                output_row,
                rows,
                width,
            } => {
                let weight = f32_weight.ok_or(ExternalGpuError::Geometry(
                    "warm D3 head lost its FP32 readout",
                ))?;
                let x = element_pointer(bases.hidden, *source_row, plan.hidden, 4)?;
                let y = element_pointer(bases.full_logits, *output_row, *width, 4)?;
                let arguments = linear.arguments(x, weight, y);
                if arguments.rows != i32::try_from(*rows).unwrap_or(-1)
                    || arguments.width != i32::try_from(*width).unwrap_or(-1)
                {
                    return Err(ExternalGpuError::Geometry(
                        "warm D3 projection shape changed",
                    ));
                }
                // SAFETY: the outer Work retains every owner through completion.
                if let Err(error) = unsafe { blas.enqueue_f32_projection(&arguments) } {
                    return Err(ExternalGpuError::Driver(format!(
                        "warm D3 projection: {error}"
                    )));
                }
            }
            crate::decision_external_gpu::GemmRecord::Bf16Selected {
                plan: linear,
                source_row,
                output_offset,
                count,
            } => {
                let weight = bf16_weight.ok_or(ExternalGpuError::Geometry(
                    "warm Pplx head lost its BF16 readout",
                ))?;
                let x = element_pointer(
                    activation.cu_deviceptr(),
                    usize::try_from(*source_row)
                        .map_err(|_| ExternalGpuError::Geometry("external row overflow"))?,
                    plan.hidden,
                    std::mem::size_of::<Bf16>(),
                )?;
                let y = element_pointer(
                    bases.logits_bf16,
                    *output_offset,
                    1,
                    std::mem::size_of::<Bf16>(),
                )?;
                let arguments = linear.arguments(x, weight, y);
                if arguments.cuda_type != 14
                    || arguments.rows != 1
                    || arguments.width != i32::try_from(*count).unwrap_or(-1)
                {
                    return Err(ExternalGpuError::Geometry(
                        "warm Pplx projection shape changed",
                    ));
                }
                // SAFETY: the outer Work retains every owner through completion.
                if let Err(error) = unsafe { blas.linear_bound(&arguments) } {
                    return Err(ExternalGpuError::Driver(format!(
                        "warm Pplx projection: {error}"
                    )));
                }
            }
            crate::decision_external_gpu::GemmRecord::Bf16Full {
                plan: linear,
                marker_row,
                output_row,
                markers,
                width,
            } => {
                let weight = bf16_weight.ok_or(ExternalGpuError::Geometry(
                    "warm Pplx head lost its BF16 readout",
                ))?;
                let x = element_pointer(
                    bases.markers,
                    *marker_row,
                    plan.hidden,
                    std::mem::size_of::<Bf16>(),
                )?;
                let y = element_pointer(
                    bases.full_logits_bf16,
                    *output_row,
                    *width,
                    std::mem::size_of::<Bf16>(),
                )?;
                let arguments = linear.arguments(x, weight, y);
                if arguments.cuda_type != 14
                    || arguments.rows != i32::try_from(*markers).unwrap_or(-1)
                    || arguments.width != i32::try_from(*width).unwrap_or(-1)
                {
                    return Err(ExternalGpuError::Geometry(
                        "warm joint projection shape changed",
                    ));
                }
                // SAFETY: the outer Work retains every owner through completion.
                if let Err(error) = unsafe { blas.linear_bound(&arguments) } {
                    return Err(ExternalGpuError::Driver(format!(
                        "warm joint projection: {error}"
                    )));
                }
            }
        }
    }
    if typed {
        launch_cast_rows(
            head.cast_kernel(),
            stream,
            bases.logits_bf16,
            bases.logits,
            0,
            plan.total_rows(),
            1,
        )?;
        if plan.plan.full_rows > 0 {
            launch_cast_rows(
                head.cast_kernel(),
                stream,
                bases.full_logits_bf16,
                bases.full_logits,
                0,
                plan.plan.full_rows,
                plan.rows,
            )?;
        }
    }
    for selection in &plan.work.selections {
        select_logits(selection, bases, plan, stream)?;
    }
    let bytes = plan
        .total_rows()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(ExternalGpuError::Geometry(
            "external readback size overflow",
        ))?;
    // SAFETY: the request host slot and device logits remain owned through the
    // outer completion boundary.
    unsafe {
        cuda_core::memory::memcpy_dtoh_async(
            host_logits.as_mut_ptr(),
            bases.logits,
            bytes,
            stream.cu_stream(),
        )
    }
    .map_err(|error| ExternalGpuError::Driver(error.to_string()))
}

fn select_logits(
    selection: &Selection,
    bases: ExternalScratchBases,
    plan: &ExternalWarmPlan,
    stream: &Stream,
) -> Result<(), ExternalGpuError> {
    let source = element_pointer(bases.full_logits, selection.source_full_row, plan.rows, 4)?;
    let destination = element_pointer(bases.logits, selection.dest_offset, 1, 4)?;
    let bytes = selection
        .count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(ExternalGpuError::Geometry(
            "external selection size overflow",
        ))?;
    // SAFETY: both ranges are owned by the retained Work scratch.
    unsafe { cuda_core::memory::memcpy_dtod_async(destination, source, bytes, stream.cu_stream()) }
        .map_err(|error| ExternalGpuError::Driver(error.to_string()))
}

pub(crate) fn finish_external_warm(
    plan: &ExternalWarmPlan,
    host_logits: &[f32],
) -> Result<Vec<DecisionOutput>, ExternalGpuError> {
    let mut results = Vec::new();
    results
        .try_reserve_exact(plan.plan.counts.len())
        .map_err(|_| ExternalGpuError::OutputAllocation)?;
    for (index, count) in plan.plan.counts.iter().enumerate() {
        let begin = plan
            .plan
            .row_offsets
            .get(index)
            .copied()
            .ok_or(ExternalGpuError::PreparedMismatch("missing row offset"))?;
        let end = begin
            .checked_add(*count)
            .ok_or(ExternalGpuError::Geometry("external logit range overflow"))?;
        let values = host_logits
            .get(begin..end)
            .ok_or(ExternalGpuError::PreparedMismatch(
                "external logits incomplete",
            ))?;
        let mut logits = Vec::new();
        logits
            .try_reserve_exact(values.len())
            .map_err(|_| ExternalGpuError::OutputAllocation)?;
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
    use infinity_v3_backend_core::DecisionInput;

    const HIDDEN: usize = 4;
    const KEY: WorkKey = (32, 4, 2);

    fn option(count: usize) -> DecisionInput {
        DecisionInput::OptionTokens {
            token_ids: (0..count as u32).collect(),
        }
    }

    fn capacity(family: ExternalFamily) -> ExternalHeadCapacity {
        ExternalHeadCapacity::new(family, 4, HIDDEN, 255).unwrap()
    }

    #[test]
    fn external_metadata_tail_packs_rows_counts_and_preserves_request_order() {
        let batch = PackedDecisionBatch::new(&[0, 3, 5]).unwrap();
        let requests = [
            DecisionInput::OptionTokens {
                token_ids: vec![0, 1, 2],
            },
            DecisionInput::Warmup,
        ];
        let metadata = prepare_external_warm_metadata(
            ExternalFamily::D3F32,
            capacity(ExternalFamily::D3F32),
            KEY,
            batch.tokens(),
            &batch,
            &requests,
            0,
        )
        .unwrap();
        let plan = bind_external_warm_metadata(metadata).unwrap();
        assert_eq!(plan.counts(), &[3, 1]);
        assert_eq!(plan.row_offsets(), &[0, 3, 4]);
        assert_eq!(plan.total_rows(), 4);
        let mut input = vec![0u32; plan.layout().end()];
        assert_eq!(plan.pack_metadata(&mut input).unwrap(), plan.layout().end());
        assert_eq!(
            &input[plan.layout().device_rows()..plan.layout().device_rows() + 2],
            &[2, 4]
        );
        assert_eq!(
            &input[plan.layout().counts()..plan.layout().counts() + 2],
            &[3, 1]
        );
    }

    #[test]
    fn pplx_joint_tail_packs_markers_and_each_marker_count() {
        let batch = PackedDecisionBatch::new(&[0, 3, 5]).unwrap();
        let requests = [
            DecisionInput::InlineOptions {
                markers: vec![0, 1],
                option_counts: vec![2, 3],
            },
            DecisionInput::Warmup,
        ];
        let metadata = prepare_external_warm_metadata(
            ExternalFamily::PplxBf16,
            capacity(ExternalFamily::PplxBf16),
            KEY,
            batch.tokens(),
            &batch,
            &requests,
            0,
        )
        .unwrap();
        let plan = bind_external_warm_metadata(metadata).unwrap();
        assert_eq!(plan.counts(), &[5, 1]);
        assert_eq!(plan.row_offsets(), &[0, 5, 6]);
        let mut input = vec![0u32; plan.layout().end()];
        plan.pack_metadata(&mut input).unwrap();
        assert_eq!(
            &input[plan.layout().device_rows()..plan.layout().device_rows() + 2],
            &[0, 1]
        );
        assert_eq!(
            &input[plan.layout().counts()..plan.layout().counts() + 3],
            &[2, 3, 1]
        );
    }

    #[test]
    fn pplx_capacity_and_gemm_records_use_typed_bf16_destinations() {
        let pplx = capacity(ExternalFamily::PplxBf16);
        assert!(pplx.bf16_full_elements().unwrap() > 0);
        assert_eq!(pplx.bf16_logit_elements().unwrap(), MAX_EXTERNAL_ROWS);
        let d3 = capacity(ExternalFamily::D3F32);
        assert_eq!(d3.bf16_full_elements().unwrap(), 0);
        assert_eq!(d3.bf16_logit_elements().unwrap(), 0);

        let batch = PackedDecisionBatch::new(&[0, 3, 5]).unwrap();
        let requests = [
            DecisionInput::OptionTokens {
                token_ids: vec![0, 1],
            },
            DecisionInput::Warmup,
        ];
        let metadata = prepare_external_warm_metadata(
            ExternalFamily::PplxBf16,
            pplx,
            KEY,
            batch.tokens(),
            &batch,
            &requests,
            0,
        )
        .unwrap();
        let plan = bind_external_warm_metadata(metadata).unwrap();
        for record in &plan.work.gemms {
            match record {
                crate::decision_external_gpu::GemmRecord::Bf16Selected {
                    plan: linear,
                    count,
                    ..
                } => {
                    let arguments = linear.arguments(0, 0, 0);
                    assert_eq!(arguments.cuda_type, 14);
                    assert_eq!(arguments.rows, 1);
                    assert_eq!(arguments.width, i32::try_from(*count).unwrap());
                    assert_eq!(arguments.rows as usize * arguments.width as usize, *count);
                }
                other => panic!("unexpected Pplx ordinary record: {other:?}"),
            }
        }

        let joint = [
            DecisionInput::InlineOptions {
                markers: vec![0, 1],
                option_counts: vec![2, 3],
            },
            DecisionInput::Warmup,
        ];
        let metadata = prepare_external_warm_metadata(
            ExternalFamily::PplxBf16,
            pplx,
            KEY,
            batch.tokens(),
            &batch,
            &joint,
            0,
        )
        .unwrap();
        let plan = bind_external_warm_metadata(metadata).unwrap();
        let mut full = 0;
        let mut selected = 0;
        for record in &plan.work.gemms {
            match record {
                crate::decision_external_gpu::GemmRecord::Bf16Full {
                    plan: linear,
                    markers,
                    width,
                    ..
                } => {
                    full += 1;
                    let arguments = linear.arguments(0, 0, 0);
                    assert_eq!(arguments.cuda_type, 14);
                    assert_eq!(arguments.rows, i32::try_from(*markers).unwrap());
                    assert_eq!(arguments.width, i32::try_from(*width).unwrap());
                    assert_eq!(
                        arguments.rows as usize * arguments.width as usize,
                        markers * width
                    );
                }
                crate::decision_external_gpu::GemmRecord::Bf16Selected {
                    plan: linear,
                    count,
                    ..
                } => {
                    selected += 1;
                    let arguments = linear.arguments(0, 0, 0);
                    assert_eq!(arguments.cuda_type, 14);
                    assert_eq!(arguments.width, i32::try_from(*count).unwrap());
                }
                other => panic!("unexpected Pplx joint record: {other:?}"),
            }
        }
        assert_eq!((full, selected), (1, 1));
    }

    #[test]
    fn external_warm_rejects_capacity_and_wrong_family_before_enqueue() {
        let batch = PackedDecisionBatch::new(&[0, 3, 5]).unwrap();
        let requests = [option(255), option(255), option(255)];
        assert!(prepare_external_warm_metadata(
            ExternalFamily::PplxBf16,
            capacity(ExternalFamily::D3F32),
            KEY,
            batch.tokens(),
            &batch,
            &requests,
            0,
        )
        .is_err());
        let narrow = ExternalHeadCapacity::new(ExternalFamily::PplxBf16, 1, HIDDEN, 255).unwrap();
        assert!(prepare_external_warm_metadata(
            ExternalFamily::PplxBf16,
            narrow,
            KEY,
            batch.tokens(),
            &batch,
            &requests[..1],
            0,
        )
        .is_err());
    }
}
