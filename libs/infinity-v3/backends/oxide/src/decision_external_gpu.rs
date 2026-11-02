// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Prepared GPU D3 / Perplexity external readouts.
//!
//! The cold `prepare` pass validates the sidecar and request plan, allocates the
//! request-independent scratch once, and records every GEMM and selection. The
//! `enqueue` body only launches borrowed views from those owners; it performs no
//! device allocation, plan construction, cuBLAS query or checkpoint read.
//!
//! D3 keeps the Native order: one FP32 SGEMM per request over all 255 codes,
//! then a device-to-device copy of the requested leading logits. Perplexity
//! keeps its separate BF16 recipe: selected rows for ordinary option-token
//! requests, and one full 255-row BF16 projection per joint-marker group.

use crate::{
    cublas::Blas,
    decision_external::{
        external_last_token_only, plan_external_requests, request_output_count, ExternalConfig,
        ExternalDecisionError, ExternalFamily, ExternalHead, ExternalPlan, ExternalRequestPlan,
    },
    decision_head::PackedDecisionBatch,
    decision_head_gpu::{DecisionGpuError, DecisionReadoutKernels, GpuHiddenRows, HiddenLeaseSlot},
    math::LinearPlan,
    tensor::{Bf16, F32},
    upload::WeightUploader,
};
use cuda_core::{
    AllocationLease, Context, DeviceBuffer, DriverError, Function, PinnedHostBuffer, Stream,
};
use infinity_v3_backend_core::{DecisionInput, DecisionOutput};
use std::{ffi::c_void, fmt, mem::ManuallyDrop, path::Path, sync::Arc};

const CAST_ELEMENTS_PER_BLOCK: usize = 256;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ExternalGpuError {
    External(ExternalDecisionError),
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

impl fmt::Display for ExternalGpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::External(error) => error.fmt(f),
            Self::ContextMismatch(what) => write!(f, "{what} belongs to another CUDA context"),
            Self::StreamMismatch(what) => write!(f, "{what} does not use the prepared stream"),
            Self::KernelModule(error) => {
                write!(f, "external decision kernel module failed: {error}")
            }
            Self::Driver(error) => write!(f, "external decision driver operation failed: {error}"),
            Self::KernelLaunch(error) => {
                write!(f, "external decision kernel launch failed: {error}")
            }
            Self::Geometry(error) => write!(f, "invalid external decision geometry: {error}"),
            Self::Hidden(error) => write!(f, "invalid external decision hidden states: {error}"),
            Self::HiddenBusy(error) => {
                write!(
                    f,
                    "hidden external decision source is not ready for the prepared stream: {error}"
                )
            }
            Self::PreparedMismatch(error) => {
                write!(f, "prepared external decision mismatch: {error}")
            }
            Self::PreparedInUse(error) => {
                write!(f, "prepared external decision is not reusable: {error}")
            }
            Self::OutputAllocation => f.write_str("external decision logits allocation failed"),
        }
    }
}

impl std::error::Error for ExternalGpuError {}

impl From<ExternalDecisionError> for ExternalGpuError {
    fn from(error: ExternalDecisionError) -> Self {
        Self::External(error)
    }
}

impl From<DecisionGpuError> for ExternalGpuError {
    fn from(error: DecisionGpuError) -> Self {
        Self::Driver(error.to_string())
    }
}

impl From<infinity_v3_backend_core::BackendError> for ExternalGpuError {
    fn from(error: infinity_v3_backend_core::BackendError) -> Self {
        Self::Driver(error.to_string())
    }
}

fn driver(what: &'static str, error: DriverError) -> ExternalGpuError {
    ExternalGpuError::Driver(format!("{what}: {error}"))
}

fn geometry(what: &'static str) -> ExternalGpuError {
    ExternalGpuError::Geometry(what)
}

fn checked_product(
    left: usize,
    right: usize,
    what: &'static str,
) -> Result<usize, ExternalGpuError> {
    left.checked_mul(right).ok_or_else(|| geometry(what))
}

fn cast_grid(elements: usize) -> Result<u32, ExternalGpuError> {
    u32::try_from(elements.div_ceil(CAST_ELEMENTS_PER_BLOCK))
        .map_err(|_| geometry("BF16 widen grid exceeds u32"))
}

fn arg<T>(value: &mut T) -> *mut c_void {
    (value as *mut T).cast()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_cast_rows(
    kernel: &Function,
    stream: &Stream,
    input: u64,
    output: u64,
    selected: u64,
    count: usize,
    width: usize,
) -> Result<(), ExternalGpuError> {
    if count == 0 || width == 0 {
        return Err(geometry("BF16 widen requires nonzero rows and width"));
    }
    let elements = checked_product(count, width, "BF16 widen element count overflow")?;
    let grid = cast_grid(elements)?;
    let mut input = input;
    let mut output = output;
    let mut selected = selected;
    let mut count = u32::try_from(count).map_err(|_| geometry("BF16 widen rows exceed u32"))?;
    let mut width = u32::try_from(width).map_err(|_| geometry("BF16 widen width exceeds u32"))?;
    // SAFETY: fixed kernel ABI, checked element count and the caller retains
    // every buffer and the module until this stream drains.
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
    .map_err(|error| ExternalGpuError::KernelLaunch(error.to_string()))
}

enum DeviceWeight {
    F32(DeviceBuffer<f32>),
    Bf16(DeviceBuffer<Bf16>),
}

impl DeviceWeight {
    fn num_bytes(&self) -> usize {
        match self {
            Self::F32(buffer) => buffer.num_bytes(),
            Self::Bf16(buffer) => buffer.num_bytes(),
        }
    }
}

#[derive(Debug)]
pub(crate) enum GemmRecord {
    F32Full {
        plan: LinearPlan<F32>,
        source_row: usize,
        output_row: usize,
        rows: usize,
        width: usize,
    },
    Bf16Selected {
        plan: LinearPlan<Bf16>,
        source_row: u32,
        output_offset: usize,
        count: usize,
    },
    Bf16Full {
        plan: LinearPlan<Bf16>,
        marker_row: usize,
        output_row: usize,
        markers: usize,
        width: usize,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Selection {
    pub(crate) source_full_row: usize,
    pub(crate) dest_offset: usize,
    pub(crate) count: usize,
}

#[derive(Debug, Default)]
pub(crate) struct WorkPlan {
    pub(crate) gemms: Vec<GemmRecord>,
    pub(crate) selections: Vec<Selection>,
    pub(crate) cast_rows: Vec<u32>,
    pub(crate) marker_rows: Vec<u32>,
}

pub(crate) fn build_work(
    family: ExternalFamily,
    hidden: usize,
    rows: usize,
    plan: &ExternalPlan,
) -> Result<WorkPlan, ExternalGpuError> {
    if hidden == 0 || rows == 0 {
        return Err(geometry("external readout dimensions must be nonzero"));
    }
    let mut work = WorkPlan::default();
    work.gemms
        .try_reserve_exact(plan.requests.len())
        .map_err(|_| ExternalGpuError::OutputAllocation)?;
    work.selections
        .try_reserve_exact(plan.full_rows)
        .map_err(|_| ExternalGpuError::OutputAllocation)?;
    let mut marker_offset = 0usize;
    for (index, request) in plan.requests.iter().enumerate() {
        let output_offset = *plan
            .row_offsets
            .get(index)
            .ok_or_else(|| geometry("missing request output offset"))?;
        match request {
            ExternalRequestPlan::LastToken {
                row,
                count,
                full,
                full_row,
            } => match family {
                ExternalFamily::D3F32 => {
                    let full_row = full_row.ok_or_else(|| {
                        geometry("D3 last-token request has no full projection row")
                    })?;
                    if !*full || *count == 0 || *count > rows {
                        return Err(geometry("invalid D3 full projection selection"));
                    }
                    let source_row = work.cast_rows.len();
                    work.cast_rows.push(*row);
                    work.gemms.push(GemmRecord::F32Full {
                        plan: LinearPlan::<F32>::contiguous(1, hidden, rows)
                            .map_err(|_| geometry("invalid D3 projection geometry"))?,
                        source_row,
                        output_row: full_row,
                        rows: 1,
                        width: rows,
                    });
                    work.selections.push(Selection {
                        source_full_row: full_row,
                        dest_offset: output_offset,
                        count: *count,
                    });
                }
                ExternalFamily::PplxBf16 => {
                    if *full || *count == 0 || *count > rows {
                        return Err(geometry("invalid Pplx selected-row projection"));
                    }
                    work.gemms.push(GemmRecord::Bf16Selected {
                        plan: LinearPlan::<Bf16>::contiguous(1, hidden, *count)
                            .map_err(|_| geometry("invalid Pplx projection geometry"))?,
                        source_row: *row,
                        output_offset,
                        count: *count,
                    });
                }
            },
            ExternalRequestPlan::Inline {
                markers,
                counts,
                full_row_start,
            } => {
                if family != ExternalFamily::PplxBf16
                    || markers.is_empty()
                    || markers.len() != counts.len()
                {
                    return Err(geometry("invalid joint inline projection"));
                }
                work.gemms.push(GemmRecord::Bf16Full {
                    plan: LinearPlan::<Bf16>::contiguous(markers.len(), hidden, rows)
                        .map_err(|_| geometry("invalid joint projection geometry"))?,
                    marker_row: marker_offset,
                    output_row: *full_row_start,
                    markers: markers.len(),
                    width: rows,
                });
                let mut dest = output_offset;
                for (local, (&marker, &count)) in markers.iter().zip(counts).enumerate() {
                    if count == 0 || count > rows {
                        return Err(geometry("invalid joint marker option count"));
                    }
                    work.marker_rows.push(marker);
                    work.selections.push(Selection {
                        source_full_row: full_row_start + local,
                        dest_offset: dest,
                        count,
                    });
                    dest = dest
                        .checked_add(count)
                        .ok_or_else(|| geometry("joint selection range overflow"))?;
                }
                marker_offset = marker_offset
                    .checked_add(markers.len())
                    .ok_or_else(|| geometry("joint marker offset overflow"))?;
            }
        }
    }
    if work.cast_rows.len() != plan.full_rows && family == ExternalFamily::D3F32 {
        return Err(geometry("D3 projection rows do not match the request plan"));
    }
    Ok(work)
}

struct Owners {
    hidden_f32: Option<DeviceBuffer<f32>>,
    markers: Option<DeviceBuffer<Bf16>>,
    cast_rows: Option<DeviceBuffer<u32>>,
    full_logits: Option<DeviceBuffer<f32>>,
    full_logits_bf16: Option<DeviceBuffer<Bf16>>,
    logits: DeviceBuffer<f32>,
    logits_bf16: Option<DeviceBuffer<Bf16>>,
    host_logits: PinnedHostBuffer<f32>,
}

impl Owners {
    fn bytes(&self) -> usize {
        let mut bytes = self
            .logits
            .num_bytes()
            .saturating_add(self.host_logits.num_bytes());
        for buffer in [
            self.hidden_f32.as_ref().map(DeviceBuffer::num_bytes),
            self.markers.as_ref().map(DeviceBuffer::num_bytes),
            self.cast_rows.as_ref().map(DeviceBuffer::num_bytes),
            self.full_logits.as_ref().map(DeviceBuffer::num_bytes),
            self.full_logits_bf16.as_ref().map(DeviceBuffer::num_bytes),
            self.logits_bf16.as_ref().map(DeviceBuffer::num_bytes),
        ]
        .into_iter()
        .flatten()
        {
            bytes = bytes.saturating_add(buffer);
        }
        bytes
    }
}

pub(crate) struct ExternalGpuPrepared {
    head_id: u64,
    family: ExternalFamily,
    hidden: usize,
    rows: usize,
    boundaries: Vec<u32>,
    counts: Vec<usize>,
    row_offsets: Vec<usize>,
    total: usize,
    full_rows: usize,
    work: WorkPlan,
    stream: Arc<Stream>,
    owners: ManuallyDrop<Owners>,
    hidden_lease: HiddenLeaseSlot<AllocationLease>,
    pending: bool,
    drained: bool,
    poisoned: bool,
}

impl ExternalGpuPrepared {
    pub(crate) fn output_counts(&self) -> &[usize] {
        &self.counts
    }

    pub(crate) fn selected_rows(&self) -> usize {
        self.total
    }

    pub(crate) fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    fn retained_bytes(&self) -> usize {
        self.owners
            .bytes()
            .saturating_add(self.hidden_lease.bytes())
    }
}

impl Drop for ExternalGpuPrepared {
    fn drop(&mut self) {
        let drained = self.drained || self.stream.synchronize().is_ok();
        if drained {
            // SAFETY: no stream work can reference these owners any more.
            unsafe { ManuallyDrop::drop(&mut self.owners) };
        } else {
            crate::record_uncharged_retained_bytes(self.retained_bytes());
            std::mem::forget(self.stream.clone());
        }
        self.hidden_lease.drop_with_drain(drained);
    }
}

pub(crate) struct ExternalGpuHead {
    id: u64,
    config: ExternalConfig,
    weight: ManuallyDrop<DeviceWeight>,
    kernels: ManuallyDrop<DecisionReadoutKernels>,
    context: Arc<Context>,
    stream: Arc<Stream>,
}

impl ExternalGpuHead {
    /// Validate cold metadata and upload the exact sidecar once.
    pub(crate) fn load(
        path: &Path,
        context: &Arc<Context>,
        stream: &Arc<Stream>,
    ) -> Result<Self, ExternalGpuError> {
        if !Arc::ptr_eq(context, stream.context()) {
            return Err(ExternalGpuError::ContextMismatch(
                "external decision stream and context",
            ));
        }
        context
            .bind_to_thread()
            .map_err(|error| driver("bind external decision load context", error))?;
        let metadata = ExternalHead::load(path)?;
        let id = metadata.id();
        let config = metadata.config().clone();
        let source = metadata.into_weight_source();
        let kernels = DecisionReadoutKernels::load(context)?;
        let weight = match config.family {
            ExternalFamily::D3F32 => {
                let mut host = Vec::new();
                let elements = checked_product(
                    config.rows,
                    config.hidden,
                    "D3 readout element count overflow",
                )?;
                host.try_reserve_exact(elements)
                    .map_err(|_| ExternalGpuError::OutputAllocation)?;
                host.resize(elements, 0f32);
                source
                    .read_f32(0, &mut host)
                    .map_err(|error| ExternalGpuError::Driver(error.to_string()))?;
                let buffer = DeviceBuffer::from_host(stream, &host)
                    .map_err(|error| driver("upload D3 readout", error))?;
                stream
                    .synchronize()
                    .map_err(|error| driver("publish D3 readout", error))?;
                DeviceWeight::F32(buffer)
            }
            ExternalFamily::PplxBf16 => {
                let mut uploader = WeightUploader::new(stream.clone())?;
                let buffer = uploader.upload_typed::<Bf16>(&[&source])?;
                uploader.finish()?;
                DeviceWeight::Bf16(buffer)
            }
        };
        Ok(Self {
            id,
            config,
            weight: ManuallyDrop::new(weight),
            kernels: ManuallyDrop::new(kernels),
            context: context.clone(),
            stream: stream.clone(),
        })
    }

    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    pub(crate) fn family(&self) -> ExternalFamily {
        self.config.family
    }

    pub(crate) fn hidden_size(&self) -> usize {
        self.config.hidden
    }

    pub(crate) fn rows(&self) -> usize {
        self.config.rows
    }

    pub(crate) fn config(&self) -> &ExternalConfig {
        &self.config
    }

    pub(crate) fn cast_kernel(&self) -> &Function {
        self.kernels.cast_kernel()
    }

    pub(crate) fn f32_weight_pointer(&self) -> Option<u64> {
        match &*self.weight {
            DeviceWeight::F32(weight) => Some(weight.cu_deviceptr()),
            DeviceWeight::Bf16(_) => None,
        }
    }

    pub(crate) fn bf16_weight_pointer(&self) -> Option<u64> {
        match &*self.weight {
            DeviceWeight::Bf16(weight) => Some(weight.cu_deviceptr()),
            DeviceWeight::F32(_) => None,
        }
    }

    pub(crate) fn last_token_only(
        &self,
        batch: &PackedDecisionBatch<'_>,
        requests: &[DecisionInput],
    ) -> bool {
        external_last_token_only(self.config.family, batch, requests)
    }

    /// Allocate and record the whole request batch before the backbone runs.
    pub(crate) fn prepare(
        &self,
        batch: &PackedDecisionBatch<'_>,
        requests: &[DecisionInput],
    ) -> Result<ExternalGpuPrepared, ExternalGpuError> {
        let plan = plan_external_requests(self.config.family, self.config.rows, batch, requests)?;
        if plan.counts.is_empty() || plan.total == 0 {
            return Err(geometry(
                "external readout request plan must reserve at least one logit",
            ));
        }
        let work = build_work(
            self.config.family,
            self.config.hidden,
            self.config.rows,
            &plan,
        )?;
        let hidden_elements = checked_product(
            work.cast_rows.len(),
            self.config.hidden,
            "external hidden buffer size overflow",
        )?;
        let marker_elements = checked_product(
            work.marker_rows.len(),
            self.config.hidden,
            "external marker buffer size overflow",
        )?;
        let full_elements = checked_product(
            plan.full_rows,
            self.config.rows,
            "external full logits size overflow",
        )?;
        let hidden_f32 = if hidden_elements == 0 {
            None
        } else {
            // SAFETY: the cast kernel writes every element before projection.
            Some(
                unsafe { DeviceBuffer::<f32>::uninitialized_async(&self.stream, hidden_elements) }
                    .map_err(|error| driver("allocate external hidden rows", error))?,
            )
        };
        let markers = if marker_elements == 0 {
            None
        } else {
            // SAFETY: each marker row is copied before the projection.
            Some(
                unsafe { DeviceBuffer::<Bf16>::uninitialized_async(&self.stream, marker_elements) }
                    .map_err(|error| driver("allocate external marker rows", error))?,
            )
        };
        let cast_rows = if work.cast_rows.is_empty() {
            None
        } else {
            Some(
                DeviceBuffer::<u32>::from_host(&self.stream, &work.cast_rows)
                    .map_err(|error| driver("upload external cast rows", error))?,
            )
        };
        let full_logits = if full_elements == 0 {
            None
        } else {
            // SAFETY: every full projection row is written before selection.
            Some(
                unsafe { DeviceBuffer::<f32>::uninitialized_async(&self.stream, full_elements) }
                    .map_err(|error| driver("allocate external full logits", error))?,
            )
        };
        // SAFETY: every selected range is written by a projection or copy.
        let logits = unsafe { DeviceBuffer::<f32>::uninitialized_async(&self.stream, plan.total) }
            .map_err(|error| driver("allocate external logits", error))?;
        let typed = self.config.family == ExternalFamily::PplxBf16;
        let full_logits_bf16 = if typed && full_elements > 0 {
            // SAFETY: the BF16 GEMM writes every element before the cast.
            Some(
                unsafe { DeviceBuffer::<Bf16>::uninitialized_async(&self.stream, full_elements) }
                    .map_err(|error| driver("allocate external BF16 full logits", error))?,
            )
        } else {
            None
        };
        let logits_bf16 = if typed {
            // Inline selections overwrite their ranges after the typed cast;
            // zero initialization keeps untouched ranges deterministic.
            Some(
                DeviceBuffer::<Bf16>::zeroed(&self.stream, plan.total)
                    .map_err(|error| driver("allocate external BF16 logits", error))?,
            )
        } else {
            None
        };
        let host_logits = PinnedHostBuffer::<f32>::zeroed(&self.context, plan.total)
            .map_err(|error| driver("allocate external pinned logits", error))?;
        Ok(ExternalGpuPrepared {
            head_id: self.id,
            family: self.config.family,
            hidden: self.config.hidden,
            rows: self.config.rows,
            boundaries: plan.boundaries,
            counts: plan.counts,
            row_offsets: plan.row_offsets,
            total: plan.total,
            full_rows: plan.full_rows,
            work,
            stream: self.stream.clone(),
            owners: ManuallyDrop::new(Owners {
                hidden_f32,
                markers,
                cast_rows,
                full_logits,
                full_logits_bf16,
                logits,
                logits_bf16,
                host_logits,
            }),
            hidden_lease: HiddenLeaseSlot::empty(),
            pending: false,
            drained: true,
            poisoned: false,
        })
    }

    /// Enqueue the prepared gather/projection/selection/readback body.
    ///
    /// No allocation, plan construction, query or checkpoint read occurs here.
    ///
    /// # Safety
    /// The caller retains the hidden source, prepared owners and head upload
    /// until this stream drains. Safe `forward`/`forward_pooled` acquire the
    /// read lease themselves and retain it in the prepared state.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn enqueue(
        &self,
        hidden: GpuHiddenRows<'_>,
        batch: &PackedDecisionBatch<'_>,
        requests: &[DecisionInput],
        prepared: &mut ExternalGpuPrepared,
        blas: &Blas,
        stream: &Stream,
        pooled: bool,
    ) -> Result<(), ExternalGpuError> {
        self.validate_prepared(batch, requests, prepared, pooled)?;
        if prepared.pending || prepared.poisoned {
            return Err(ExternalGpuError::PreparedInUse(
                "drain or recover the previous submission first",
            ));
        }
        if !Arc::ptr_eq(stream.context(), &self.context)
            || !Arc::ptr_eq(blas.context(), &self.context)
        {
            return Err(ExternalGpuError::ContextMismatch(
                "external decision hidden source, stream or cuBLAS handle",
            ));
        }
        if !std::ptr::eq(stream, self.stream.as_ref())
            || !std::ptr::eq(blas.stream(), self.stream.as_ref())
        {
            return Err(ExternalGpuError::StreamMismatch(
                "external decision stream or cuBLAS handle",
            ));
        }
        if !Arc::ptr_eq(hidden.buffer().context(), &self.context) {
            return Err(ExternalGpuError::ContextMismatch(
                "external decision hidden source",
            ));
        }
        let expected_rows = if pooled { batch.len() } else { batch.tokens() };
        if hidden.rows() != expected_rows {
            return Err(ExternalGpuError::Hidden(
                "hidden row count does not match the packed or pooled request shape",
            ));
        }
        if hidden.width() != self.config.hidden {
            return Err(ExternalGpuError::Hidden(
                "hidden width does not match the external readout",
            ));
        }
        if prepared.hidden_lease.is_held() {
            return Err(ExternalGpuError::PreparedInUse(
                "hidden source lease is still held",
            ));
        }
        let lease = hidden
            .buffer()
            .storage()
            .acquire(stream, false, false)
            .map_err(|error| {
                ExternalGpuError::HiddenBusy(format!(
                    "hidden source is not ordered after its producer: {error}"
                ))
            })?;
        prepared
            .hidden_lease
            .install(lease, hidden.buffer().num_bytes())
            .map_err(ExternalGpuError::PreparedInUse)?;
        prepared.poisoned = true;
        prepared.drained = false;
        prepared.pending = false;

        if let (Some(cast_rows), Some(hidden_f32)) = (
            prepared.owners.cast_rows.as_ref(),
            prepared.owners.hidden_f32.as_ref(),
        ) {
            launch_cast_rows(
                self.kernels.cast_kernel(),
                stream,
                hidden.buffer().cu_deviceptr(),
                hidden_f32.cu_deviceptr(),
                cast_rows.cu_deviceptr(),
                prepared.work.cast_rows.len(),
                self.config.hidden,
            )?;
        }
        if let Some(markers) = prepared.owners.markers.as_ref() {
            let bytes = self
                .config
                .hidden
                .checked_mul(std::mem::size_of::<Bf16>())
                .ok_or_else(|| geometry("external marker copy size overflow"))?;
            for (destination, &source_row) in prepared.work.marker_rows.iter().enumerate() {
                let source = hidden
                    .buffer()
                    .cu_deviceptr()
                    .checked_add(
                        u64::try_from(source_row as usize)
                            .map_err(|_| geometry("external source row overflow"))?
                            .checked_mul(
                                u64::try_from(bytes)
                                    .map_err(|_| geometry("external marker byte size overflow"))?,
                            )
                            .ok_or_else(|| geometry("external source byte offset overflow"))?,
                    )
                    .ok_or_else(|| geometry("external source address overflow"))?;
                let destination = markers
                    .cu_deviceptr()
                    .checked_add(
                        u64::try_from(destination)
                            .map_err(|_| geometry("external marker row overflow"))?
                            .checked_mul(
                                u64::try_from(bytes)
                                    .map_err(|_| geometry("external marker byte size overflow"))?,
                            )
                            .ok_or_else(|| geometry("external marker byte offset overflow"))?,
                    )
                    .ok_or_else(|| geometry("external marker address overflow"))?;
                // SAFETY: both rows are owned by retained buffers and the
                // ranges are one checked hidden row.
                cuda_core::memory::memcpy_dtod_async(
                    destination,
                    source,
                    bytes,
                    stream.cu_stream(),
                )
                .map_err(|error| driver("copy external marker row", error))?;
            }
        }
        for record in &prepared.work.gemms {
            match record {
                GemmRecord::F32Full {
                    plan,
                    source_row,
                    output_row,
                    rows,
                    width,
                } => {
                    let hidden_f32 = prepared
                        .owners
                        .hidden_f32
                        .as_ref()
                        .ok_or_else(|| geometry("missing D3 hidden projection buffer"))?;
                    let full_logits = prepared
                        .owners
                        .full_logits
                        .as_ref()
                        .ok_or_else(|| geometry("missing D3 full logit buffer"))?;
                    let weight = match &*self.weight {
                        DeviceWeight::F32(weight) => weight,
                        DeviceWeight::Bf16(_) => {
                            return Err(geometry("D3 head lost its FP32 readout"))
                        }
                    };
                    let x = element_pointer(
                        hidden_f32.cu_deviceptr(),
                        *source_row,
                        self.config.hidden,
                        std::mem::size_of::<f32>(),
                    )?;
                    let y = element_pointer(
                        full_logits.cu_deviceptr(),
                        *output_row,
                        *width,
                        std::mem::size_of::<f32>(),
                    )?;
                    let arguments = plan.arguments(x, weight.cu_deviceptr(), y);
                    if arguments.rows != i32::try_from(*rows).unwrap_or(-1)
                        || arguments.width != i32::try_from(*width).unwrap_or(-1)
                    {
                        return Err(geometry("D3 prepared GEMM shape changed"));
                    }
                    // SAFETY: prepared owners and the borrowed hidden source
                    // remain alive until the caller drains the stream.
                    if let Err(error) = unsafe { blas.enqueue_f32_projection(&arguments) } {
                        return Err(ExternalGpuError::Driver(format!(
                            "F32 external decision projection: {error}"
                        )));
                    }
                }
                GemmRecord::Bf16Selected {
                    plan,
                    source_row,
                    output_offset,
                    count,
                } => {
                    let weight = match &*self.weight {
                        DeviceWeight::Bf16(weight) => weight,
                        DeviceWeight::F32(_) => {
                            return Err(geometry("Pplx head lost its BF16 readout"))
                        }
                    };
                    let x = element_pointer(
                        hidden.buffer().cu_deviceptr(),
                        usize::try_from(*source_row)
                            .map_err(|_| geometry("external source row overflow"))?,
                        self.config.hidden,
                        std::mem::size_of::<Bf16>(),
                    )?;
                    let logits = prepared
                        .owners
                        .logits_bf16
                        .as_ref()
                        .ok_or_else(|| geometry("missing BF16 selected logit buffer"))?;
                    let y = element_pointer(
                        logits.cu_deviceptr(),
                        *output_offset,
                        1,
                        std::mem::size_of::<Bf16>(),
                    )?;
                    let arguments = plan.arguments(x, weight.cu_deviceptr(), y);
                    if arguments.cuda_type != 14
                        || arguments.rows != 1
                        || arguments.width != i32::try_from(*count).unwrap_or(-1)
                    {
                        return Err(geometry("Pplx prepared GEMM shape changed"));
                    }
                    // SAFETY: same retained-owner contract as above.
                    if let Err(error) = unsafe { blas.linear_bound(&arguments) } {
                        return Err(ExternalGpuError::Driver(format!(
                            "BF16 external decision projection: {error}"
                        )));
                    }
                }
                GemmRecord::Bf16Full {
                    plan,
                    marker_row,
                    output_row,
                    markers,
                    width,
                } => {
                    let weight = match &*self.weight {
                        DeviceWeight::Bf16(weight) => weight,
                        DeviceWeight::F32(_) => {
                            return Err(geometry("Pplx head lost its BF16 readout"))
                        }
                    };
                    let markers_buffer = prepared
                        .owners
                        .markers
                        .as_ref()
                        .ok_or_else(|| geometry("missing joint marker projection buffer"))?;
                    let x = element_pointer(
                        markers_buffer.cu_deviceptr(),
                        *marker_row,
                        self.config.hidden,
                        std::mem::size_of::<Bf16>(),
                    )?;
                    let full_logits_bf16 = prepared
                        .owners
                        .full_logits_bf16
                        .as_ref()
                        .ok_or_else(|| geometry("missing BF16 joint logit buffer"))?;
                    let y = element_pointer(
                        full_logits_bf16.cu_deviceptr(),
                        *output_row,
                        *width,
                        std::mem::size_of::<Bf16>(),
                    )?;
                    let arguments = plan.arguments(x, weight.cu_deviceptr(), y);
                    if arguments.cuda_type != 14
                        || arguments.rows != i32::try_from(*markers).unwrap_or(-1)
                        || arguments.width != i32::try_from(*width).unwrap_or(-1)
                    {
                        return Err(geometry("joint prepared GEMM shape changed"));
                    }
                    // SAFETY: same retained-owner contract as above.
                    if let Err(error) = unsafe { blas.linear_bound(&arguments) } {
                        return Err(ExternalGpuError::Driver(format!(
                            "joint external decision projection: {error}"
                        )));
                    }
                }
            }
        }
        if let Some(logits_bf16) = prepared.owners.logits_bf16.as_ref() {
            launch_cast_rows(
                self.kernels.cast_kernel(),
                stream,
                logits_bf16.cu_deviceptr(),
                prepared.owners.logits.cu_deviceptr(),
                0,
                prepared.total,
                1,
            )?;
        }
        if let Some(full_logits_bf16) = prepared.owners.full_logits_bf16.as_ref() {
            let full_logits = prepared
                .owners
                .full_logits
                .as_ref()
                .ok_or_else(|| geometry("missing external full logit buffer"))?;
            launch_cast_rows(
                self.kernels.cast_kernel(),
                stream,
                full_logits_bf16.cu_deviceptr(),
                full_logits.cu_deviceptr(),
                0,
                prepared.full_rows,
                self.config.rows,
            )?;
        }
        if !prepared.work.selections.is_empty() {
            let full_logits = prepared
                .owners
                .full_logits
                .as_ref()
                .ok_or_else(|| geometry("missing external full logit buffer"))?;
            for selection in &prepared.work.selections {
                let source = element_pointer(
                    full_logits.cu_deviceptr(),
                    selection.source_full_row,
                    self.config.rows,
                    std::mem::size_of::<f32>(),
                )?;
                let destination = element_pointer(
                    prepared.owners.logits.cu_deviceptr(),
                    selection.dest_offset,
                    1,
                    std::mem::size_of::<f32>(),
                )?;
                let bytes = selection
                    .count
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| geometry("external selection size overflow"))?;
                // SAFETY: both ranges are owned by prepared buffers.
                cuda_core::memory::memcpy_dtod_async(
                    destination,
                    source,
                    bytes,
                    stream.cu_stream(),
                )
                .map_err(|error| driver("select external decision logits", error))?;
            }
        }
        let device_logits = prepared.owners.logits.cu_deviceptr();
        let host_logits = prepared.owners.host_logits.as_mut_ptr();
        let bytes = prepared
            .total
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| geometry("external logit readback size overflow"))?;
        // SAFETY: both buffers are owned by prepared until finish/drain.
        if let Err(error) = unsafe {
            cuda_core::memory::memcpy_dtoh_async(
                host_logits,
                device_logits,
                bytes,
                stream.cu_stream(),
            )
        } {
            return Err(driver("enqueue external logit readback", error));
        }
        prepared.pending = true;
        prepared.poisoned = false;
        Ok(())
    }

    pub(crate) fn finish(
        &self,
        prepared: &mut ExternalGpuPrepared,
    ) -> Result<Vec<DecisionOutput>, ExternalGpuError> {
        if !prepared.pending {
            return Err(ExternalGpuError::PreparedMismatch(
                "one submission has not been enqueued",
            ));
        }
        if let Err(error) = prepared.stream.synchronize() {
            prepared.poisoned = true;
            return Err(driver("drain external decision submission", error));
        }
        prepared.pending = false;
        prepared.drained = true;
        prepared.poisoned = false;
        prepared.hidden_lease.release_after_drain();
        let host = prepared.owners.host_logits.as_slice();
        let mut results = Vec::new();
        results
            .try_reserve_exact(prepared.counts.len())
            .map_err(|_| ExternalGpuError::OutputAllocation)?;
        for (index, count) in prepared.counts.iter().enumerate() {
            let begin = prepared
                .row_offsets
                .get(index)
                .copied()
                .ok_or_else(|| geometry("missing prepared logit offset"))?;
            let end = begin
                .checked_add(*count)
                .ok_or_else(|| geometry("prepared logit range overflow"))?;
            let values = host
                .get(begin..end)
                .ok_or_else(|| geometry("prepared logits are incomplete"))?;
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

    pub(crate) fn forward(
        &self,
        hidden: GpuHiddenRows<'_>,
        batch: &PackedDecisionBatch<'_>,
        requests: &[DecisionInput],
        prepared: &mut ExternalGpuPrepared,
        blas: &Blas,
        stream: &Stream,
    ) -> Result<Vec<DecisionOutput>, ExternalGpuError> {
        // SAFETY: this method owns the complete enqueue/finish lifetime.
        unsafe { self.enqueue(hidden, batch, requests, prepared, blas, stream, false)? };
        self.finish(prepared)
    }

    pub(crate) fn forward_pooled(
        &self,
        hidden: GpuHiddenRows<'_>,
        batch: &PackedDecisionBatch<'_>,
        requests: &[DecisionInput],
        prepared: &mut ExternalGpuPrepared,
        blas: &Blas,
        stream: &Stream,
    ) -> Result<Vec<DecisionOutput>, ExternalGpuError> {
        if !external_last_token_only(self.config.family, batch, requests) {
            return Err(ExternalGpuError::External(
                ExternalDecisionError::UnsupportedRequest(
                    "pooled external decisions require last-token-only requests",
                ),
            ));
        }
        // SAFETY: this method owns the complete enqueue/finish lifetime.
        unsafe { self.enqueue(hidden, batch, requests, prepared, blas, stream, true)? };
        self.finish(prepared)
    }

    pub(crate) fn recover_after_failed_drain(
        &self,
        prepared: &mut ExternalGpuPrepared,
    ) -> Result<(), ExternalGpuError> {
        if !prepared.poisoned && !prepared.pending {
            return Ok(());
        }
        prepared.stream.synchronize().map_err(|error| {
            ExternalGpuError::Driver(format!("recover external stream: {error}"))
        })?;
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
        prepared: &ExternalGpuPrepared,
        pooled: bool,
    ) -> Result<(), ExternalGpuError> {
        if prepared.head_id != self.id
            || prepared.family != self.config.family
            || prepared.hidden != self.config.hidden
            || prepared.rows != self.config.rows
        {
            return Err(ExternalGpuError::PreparedMismatch(
                "prepared metadata belongs to another head",
            ));
        }
        if prepared.boundaries.as_slice() != batch.boundaries() {
            return Err(ExternalGpuError::PreparedMismatch(
                "prepared boundaries belong to another packed batch",
            ));
        }
        if prepared.counts.len() != requests.len()
            || prepared.row_offsets.len() != requests.len().saturating_add(1)
        {
            return Err(ExternalGpuError::PreparedMismatch(
                "prepared request count does not match the decision batch",
            ));
        }
        for (index, request) in requests.iter().enumerate() {
            let expected =
                request_output_count(self.config.family, self.config.rows, index, request)?;
            if prepared.counts.get(index) != Some(&expected) {
                return Err(ExternalGpuError::PreparedMismatch(
                    "prepared output count does not match the request",
                ));
            }
            let begin = prepared
                .row_offsets
                .get(index)
                .copied()
                .ok_or_else(|| geometry("missing prepared logit offset"))?;
            let end = prepared
                .row_offsets
                .get(index.saturating_add(1))
                .copied()
                .ok_or_else(|| geometry("missing prepared logit offset"))?;
            if end.checked_sub(begin) != Some(expected) {
                return Err(ExternalGpuError::PreparedMismatch(
                    "prepared logit range does not match the output count",
                ));
            }
        }
        if pooled && !external_last_token_only(self.config.family, batch, requests) {
            return Err(ExternalGpuError::External(
                ExternalDecisionError::UnsupportedRequest(
                    "pooled external decisions require last-token-only requests",
                ),
            ));
        }
        Ok(())
    }
}

impl Drop for ExternalGpuHead {
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

pub(crate) fn element_pointer(
    base: u64,
    row: usize,
    stride: usize,
    element_bytes: usize,
) -> Result<u64, ExternalGpuError> {
    let elements = row
        .checked_mul(stride)
        .ok_or_else(|| geometry("external device element offset overflow"))?;
    let bytes = elements
        .checked_mul(element_bytes)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| geometry("external device byte offset overflow"))?;
    base.checked_add(bytes)
        .ok_or_else(|| geometry("external device address overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decision_external::plan_external_requests;
    use crate::decision_head::PackedDecisionBatch;
    use infinity_v3_backend_core::DecisionInput;

    const HIDDEN: usize = 4;
    const ROWS: usize = 255;

    fn option(count: usize) -> DecisionInput {
        DecisionInput::OptionTokens {
            token_ids: (0..count as u32).collect(),
        }
    }

    #[test]
    fn d3_work_keeps_full_projection_and_selected_prefix() {
        let batch = PackedDecisionBatch::new(&[0, 3, 5]).unwrap();
        let requests = [option(3), DecisionInput::Warmup];
        let plan = plan_external_requests(ExternalFamily::D3F32, ROWS, &batch, &requests).unwrap();
        let work = build_work(ExternalFamily::D3F32, HIDDEN, ROWS, &plan).unwrap();
        assert_eq!(work.cast_rows, vec![2, 4]);
        assert_eq!(work.gemms.len(), 2);
        assert_eq!(work.selections.len(), 2);
        match &work.gemms[0] {
            GemmRecord::F32Full {
                source_row,
                output_row,
                rows,
                width,
                plan,
            } => {
                assert_eq!((*source_row, *output_row, *rows, *width), (0, 0, 1, ROWS));
                assert_eq!(plan.weight_layout().shape(), &[ROWS, HIDDEN]);
                assert_eq!(plan.output_layout().shape(), &[1, ROWS]);
            }
            _ => panic!("D3 must use one full FP32 projection per request"),
        }
        assert_eq!(work.selections[0].count, 3);
        assert_eq!(work.selections[1].count, 1);
    }

    #[test]
    fn pplx_work_uses_selected_rows_and_one_full_joint_projection() {
        let batch = PackedDecisionBatch::new(&[0, 3, 5]).unwrap();
        let requests = [option(2), DecisionInput::Warmup];
        let plan =
            plan_external_requests(ExternalFamily::PplxBf16, ROWS, &batch, &requests).unwrap();
        let work = build_work(ExternalFamily::PplxBf16, HIDDEN, ROWS, &plan).unwrap();
        assert!(work.cast_rows.is_empty());
        assert!(work.selections.is_empty());
        assert_eq!(work.gemms.len(), 2);
        match &work.gemms[0] {
            GemmRecord::Bf16Selected {
                source_row,
                output_offset,
                count,
                plan,
            } => {
                assert_eq!((*source_row, *output_offset, *count), (2, 0, 2));
                assert_eq!(plan.weight_layout().shape(), &[2, HIDDEN]);
            }
            _ => panic!("Pplx selected rows must use the BF16 recipe"),
        }

        let joint = [
            DecisionInput::InlineOptions {
                markers: vec![0, 1],
                option_counts: vec![2, 3],
            },
            DecisionInput::Warmup,
        ];
        let plan = plan_external_requests(ExternalFamily::PplxBf16, ROWS, &batch, &joint).unwrap();
        let work = build_work(ExternalFamily::PplxBf16, HIDDEN, ROWS, &plan).unwrap();
        assert_eq!(work.marker_rows, vec![0, 1]);
        assert_eq!(work.gemms.len(), 2);
        match &work.gemms[0] {
            GemmRecord::Bf16Full {
                marker_row,
                output_row,
                markers,
                width,
                plan,
            } => {
                assert_eq!(
                    (*marker_row, *output_row, *markers, *width),
                    (0, 0, 2, ROWS)
                );
                assert_eq!(plan.weight_layout().shape(), &[ROWS, HIDDEN]);
                assert_eq!(plan.output_layout().shape(), &[2, ROWS]);
            }
            _ => panic!("Joint markers must share one full BF16 projection"),
        }
        assert_eq!(work.selections.len(), 2);
        assert_eq!(work.selections[0].source_full_row, 0);
        assert_eq!(work.selections[1].source_full_row, 1);
        assert_eq!(work.selections[0].count, 2);
        assert_eq!(work.selections[1].count, 3);
    }

    #[test]
    fn malformed_work_geometry_is_rejected_before_any_device_use() {
        let plan = ExternalPlan {
            counts: vec![1],
            row_offsets: vec![0, 1],
            boundaries: vec![0, 1],
            requests: vec![ExternalRequestPlan::LastToken {
                row: 0,
                count: 0,
                full: true,
                full_row: Some(0),
            }],
            marker_rows: vec![],
            full_rows: 1,
            total: 1,
        };
        assert!(build_work(ExternalFamily::D3F32, HIDDEN, ROWS, &plan).is_err());
    }
}
