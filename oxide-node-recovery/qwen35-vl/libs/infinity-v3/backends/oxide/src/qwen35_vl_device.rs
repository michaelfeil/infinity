// Concrete device execution for the Qwen3.5 vision tower.
//!
//! The executor performs real cold BF16 upload, real cuBLASLt projections and
//! the qualified FA4 non-causal vision profile. It reuses the same generic
//! `LtProvider`, `Fa4`, typed `Bf16` storage and `Checkpoint` metadata reader as
//! the rest of the Oxide backend; it does not introduce a second tensor or
//! GEMM stack.
//!
//! Qwen3.5-specific projection: the Conv3D patch kernel `[H, C, 2, P, P]` is
//! deinterleaved at cold load into two compact `[H, C*P*P]` matrices. At forward
//! time a device slice kernel extracts temporal slice 0/1 rows from the original
//! packed `[C, T, P, P]` input, two GEMMs run, their BF16 sums are added, and the
//! single patch bias is applied. This reproduces the Native temporal-2 path
//! without padding or duplicated rows.
//!
//! Media batching and preprocessing geometry come from `qwen35_vl_media`; the
//! device path consumes the same `HostImage` rows the CPU reference test uses.

use crate::fa4::{AttentionBatch, AttentionSpec, DType, Fa4, Scale, Window};
use crate::lt::{LtPlan, LtProvider};
use crate::model::Checkpoint;
use crate::qwen35_vl_config::{Dtype, Qwen35VlConfig};
use crate::qwen35_vl_inventory::expected_vision_tensors;
use crate::qwen35_vl_vision::{HostImage, VisionOutput};
use crate::tensor::Bf16;
use crate::upload::WeightUploader;
use cuda_core::{launch_kernel, Context, DeviceBuffer, PinnedHostBuffer, Stream};
use infinity_v3_backend_core::BackendError;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::ffi::c_void;
use std::path::Path;
use std::sync::Arc;

fn start(error: impl std::fmt::Display) -> BackendError {
    BackendError::Start(format!("Oxide qwen3.5-vl device: {error}"))
}

fn infer(error: impl std::fmt::Display) -> BackendError {
    BackendError::Inference(format!("Oxide qwen3.5-vl device: {error}"))
}

fn product(left: usize, right: usize) -> Result<usize, BackendError> {
    left.checked_mul(right)
        .ok_or_else(|| infer("dimension overflow"))
}

fn u32_size(value: usize) -> Result<u32, BackendError> {
    u32::try_from(value).map_err(|_| infer("dimension exceeds u32"))
}

fn arg<T>(value: &mut T) -> *mut c_void {
    value as *mut T as *mut c_void
}

fn blocks(elements: usize) -> Result<(u32, u32, u32), BackendError> {
    let blocks = elements
        .checked_add(255)
        .ok_or_else(|| infer("launch grid overflow"))?
        / 256;
    Ok((u32_size(blocks.max(1))?, 1, 1))
}

fn launch(
    function: &cuda_core::Function,
    stream: &Stream,
    grid: (u32, u32, u32),
    args: &mut [*mut c_void],
) -> Result<(), BackendError> {
    // SAFETY: callers retain every buffer through workspace drain and pass the
    // exact kernel ABI scalar/pointer arguments.
    unsafe {
        launch_kernel(
            function.cu_function(),
            grid,
            (256, 1, 1),
            0,
            stream.cu_stream(),
            args,
        )
    }
    .map_err(infer)
}

fn launch_threads(
    function: &cuda_core::Function,
    stream: &Stream,
    grid: (u32, u32, u32),
    threads: u32,
    args: &mut [*mut c_void],
) -> Result<(), BackendError> {
    unsafe {
        launch_kernel(
            function.cu_function(),
            grid,
            (threads, 1, 1),
            0,
            stream.cu_stream(),
            args,
        )
    }
    .map_err(infer)
}

fn zeroed<T: cuda_core::DeviceCopy>(
    stream: &Arc<Stream>,
    len: usize,
) -> Result<DeviceBuffer<T>, BackendError> {
    // SAFETY: the complete allocation is zeroed on the ordered stream first.
    let buffer = unsafe { DeviceBuffer::<T>::uninitialized_async(stream, len) }.map_err(infer)?;
    unsafe {
        cuda_core::memory::memset_d8_async(
            buffer.cu_deviceptr(),
            0,
            buffer.num_bytes(),
            stream.cu_stream(),
        )
    }
    .map_err(infer)?;
    Ok(buffer)
}

struct VisionKernels {
    layer_norm: cuda_core::Function,
    add: cuda_core::Function,
    bias: cuda_core::Function,
    f32_to_bf16: cuda_core::Function,
    gelu_tanh: cuda_core::Function,
    silu: cuda_core::Function,
    pos_interp: cuda_core::Function,
    rope2d: cuda_core::Function,
    temporal_slice: cuda_core::Function,
}

impl VisionKernels {
    fn load(context: &Arc<Context>) -> Result<Self, BackendError> {
        let path = std::env::var_os(infinity_v3_config::infinity_env!("OXIDE_KERNEL_MODULE"))
            .ok_or_else(|| start("Set INFINITY_OXIDE_KERNEL_MODULE"))?;
        let path = std::path::PathBuf::from(path);
        let path = if path.is_dir() {
            let (major, minor) = context.compute_capability().map_err(start)?;
            path.join(format!("sm_{major}{minor}"))
                .join("kernels.cubin")
        } else {
            path
        };
        let data =
            std::fs::read(&path).map_err(|error| start(format!("{}: {error}", path.display())))?;
        let module = context.load_module_from_image(&data).map_err(start)?;
        Ok(Self {
            layer_norm: module.load_function("infinity_layer_norm").map_err(start)?,
            add: module
                .load_function("infinity_qwen35vl_add_bf16")
                .map_err(start)?,
            bias: module
                .load_function("infinity_qwen35vl_bias_bf16")
                .map_err(start)?,
            f32_to_bf16: module
                .load_function("infinity_f32_to_bf16")
                .map_err(start)?,
            gelu_tanh: module
                .load_function("infinity_qwen35vl_gelu_tanh_bf16")
                .map_err(start)?,
            silu: module
                .load_function("infinity_qwen35vl_silu_bf16")
                .map_err(start)?,
            pos_interp: module
                .load_function("infinity_qwen35vl_pos_interp")
                .map_err(start)?,
            rope2d: module
                .load_function("infinity_qwen35vl_rope2d_bf16")
                .map_err(start)?,
            temporal_slice: module
                .load_function("infinity_qwen35vl_temporal_slice_bf16")
                .map_err(start)?,
        })
    }
}

struct VisionPlans {
    patch_flat: LtPlan<Bf16>,
    patch_t0: LtPlan<Bf16>,
    patch_t1: LtPlan<Bf16>,
    qkv: LtPlan<Bf16>,
    projection: LtPlan<Bf16>,
    mlp_fc1: LtPlan<Bf16>,
    mlp_fc2: LtPlan<Bf16>,
    merger_fc1: Vec<LtPlan<Bf16>>,
    merger_fc2: Vec<LtPlan<Bf16>>,
}

pub struct Qwen35VlVisionDevice {
    config: Qwen35VlConfig,
    weights: BTreeMap<String, DeviceBuffer<Bf16>>,
    patch_t0: Option<DeviceBuffer<Bf16>>,
    patch_t1: Option<DeviceBuffer<Bf16>>,
    zero_patch_bias: Option<DeviceBuffer<Bf16>>,
    plans: RefCell<VisionPlans>,
    kernels: VisionKernels,
    fa4: Fa4,
    side: usize,
    stream: Arc<Stream>,
}

fn upload(
    uploader: &mut WeightUploader,
    checkpoint: &Checkpoint,
    name: &str,
    shape: &[usize],
) -> Result<DeviceBuffer<Bf16>, BackendError> {
    let source = checkpoint.source(name, shape)?;
    uploader.upload(&[&source])
}

fn read_deinterleaved_patch(
    checkpoint: &Checkpoint,
    config: &Qwen35VlConfig,
) -> Result<(Vec<Bf16>, Vec<Bf16>), BackendError> {
    let vision = &config.vision;
    let hidden = vision.hidden_size;
    let channels = vision.in_channels;
    let temporal = vision.temporal_patch_size;
    let patch = vision.patch_size;
    let patch_area = product(patch, patch)?;
    let total = product(product(hidden, channels)?, product(temporal, patch_area)?)?;
    let source = checkpoint.source(
        "model.visual.patch_embed.proj.weight",
        &[hidden, channels, temporal, patch, patch],
    )?;
    let mut packed = vec![0u16; total];
    // One bounded read of the actual patch kernel. This tensor is a few MiB
    // even for the largest published tower, and deinterleaving is a cold-load
    // transform that never mutates the checkpoint.
    source.read_bf16(0, &mut packed)?;
    let compact = product(channels, patch_area)?;
    let mut first = vec![Bf16::from_bits(0); product(hidden, compact)?];
    let mut second = first.clone();
    for unit in 0..hidden {
        for channel in 0..channels {
            for slice in 0..temporal {
                for y in 0..patch {
                    for x in 0..patch {
                        let source_index =
                            ((((unit * channels + channel) * temporal + slice) * patch + y)
                                * patch)
                                + x;
                        let destination = ((unit * channels + channel) * patch + y) * patch + x;
                        if slice == 0 {
                            first[destination] = Bf16::from_bits(packed[source_index]);
                        } else {
                            second[destination] = Bf16::from_bits(packed[source_index]);
                        }
                    }
                }
            }
        }
    }
    Ok((first, second))
}

enum PinBuffer {
    F32(PinnedHostBuffer<f32>),
    U32(PinnedHostBuffer<u32>),
}

/// Pinned host owner retained until the ordered H2D copy completes. Drop fences
/// the stream; an uncertain fence leaks the pinned allocation rather than
/// calling `cuMemFreeHost` while CUDA may still read it.
pub(crate) struct HostPin {
    buffer: std::mem::ManuallyDrop<PinBuffer>,
    stream: Arc<Stream>,
    armed: bool,
}

impl HostPin {
    pub(crate) fn f32(stream: &Arc<Stream>, data: &[f32]) -> Result<Self, BackendError> {
        let buffer = PinnedHostBuffer::from_slice(stream.context(), data).map_err(infer)?;
        Ok(Self {
            buffer: std::mem::ManuallyDrop::new(PinBuffer::F32(buffer)),
            stream: stream.clone(),
            armed: true,
        })
    }
    pub(crate) fn u32(stream: &Arc<Stream>, data: &[u32]) -> Result<Self, BackendError> {
        let buffer = PinnedHostBuffer::from_slice(stream.context(), data).map_err(infer)?;
        Ok(Self {
            buffer: std::mem::ManuallyDrop::new(PinBuffer::U32(buffer)),
            stream: stream.clone(),
            armed: true,
        })
    }
    pub(crate) fn as_f32(&self) -> &PinnedHostBuffer<f32> {
        match &*self.buffer {
            PinBuffer::F32(buffer) => buffer,
            _ => unreachable!("f32 pin"),
        }
    }
    pub(crate) fn as_u32(&self) -> &PinnedHostBuffer<u32> {
        match &*self.buffer {
            PinBuffer::U32(buffer) => buffer,
            _ => unreachable!("u32 pin"),
        }
    }
    pub(crate) fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for HostPin {
    fn drop(&mut self) {
        if self.armed {
            if self.stream.synchronize().is_ok() {
                unsafe { std::mem::ManuallyDrop::drop(&mut self.buffer) };
            } else {
                let buffer = unsafe { std::mem::ManuallyDrop::take(&mut self.buffer) };
                std::mem::forget(buffer);
            }
        } else {
            unsafe { std::mem::ManuallyDrop::drop(&mut self.buffer) };
        }
    }
}

struct VisionDeviceInner {
    workspace: VisionWorkspace,
    merged_rows: usize,
    out_hidden: usize,
    host_pins: Vec<HostPin>,
    pixels: Option<DeviceBuffer<f32>>,
    frame_offsets: Option<DeviceBuffer<u32>>,
}

/// RAII owner for device-resident merged rows, workspace and pinned H2D sources.
/// Drop fences the stream; an uncertain fence leaks/quarantines the whole inner
/// owner rather than freeing unproven GPU/pinned storage.
pub(crate) struct VisionDeviceOutput {
    inner: std::mem::ManuallyDrop<VisionDeviceInner>,
    stream: Arc<Stream>,
    completed: bool,
}

impl VisionDeviceOutput {
    fn new(workspace: VisionWorkspace, stream: Arc<Stream>) -> Self {
        Self {
            inner: std::mem::ManuallyDrop::new(VisionDeviceInner {
                workspace,
                merged_rows: 0,
                out_hidden: 0,
                host_pins: Vec::new(),
                pixels: None,
                frame_offsets: None,
            }),
            stream,
            completed: false,
        }
    }
    fn set_geometry(&mut self, merged_rows: usize, out_hidden: usize) {
        self.inner.merged_rows = merged_rows;
        self.inner.out_hidden = out_hidden;
    }
    fn workspace_ptr(&mut self) -> *mut VisionWorkspace {
        &mut self.inner.workspace
    }
    fn push_pin(&mut self, pin: HostPin) {
        self.inner.host_pins.push(pin);
    }
    fn set_pixels(&mut self, pixels: DeviceBuffer<f32>) {
        self.inner.pixels = Some(pixels);
    }
    fn pixels_ptr(&self) -> u64 {
        self.inner
            .pixels
            .as_ref()
            .map_or(0, DeviceBuffer::cu_deviceptr)
    }
    fn set_frame_offsets(&mut self, offsets: DeviceBuffer<u32>) {
        self.inner.frame_offsets = Some(offsets);
    }
    fn frame_offsets_ptr(&self) -> u64 {
        self.inner
            .frame_offsets
            .as_ref()
            .map_or(0, DeviceBuffer::cu_deviceptr)
    }
    pub(crate) fn workspace(&self) -> &VisionWorkspace {
        &self.inner.workspace
    }
    pub(crate) fn workspace_mut(&mut self) -> &mut VisionWorkspace {
        &mut self.inner.workspace
    }
    pub(crate) fn merged_rows(&self) -> usize {
        self.inner.merged_rows
    }
    pub(crate) fn out_hidden(&self) -> usize {
        self.inner.out_hidden
    }
    pub(crate) fn mark_completed(&mut self) {
        for pin in &mut self.inner.host_pins {
            pin.disarm();
        }
        self.inner.workspace.mark_completed();
        self.completed = true;
    }
}

impl Drop for VisionDeviceOutput {
    fn drop(&mut self) {
        if self.completed || self.stream.synchronize().is_ok() {
            unsafe { std::mem::ManuallyDrop::drop(&mut self.inner) };
        } else {
            let inner = unsafe { std::mem::ManuallyDrop::take(&mut self.inner) };
            std::mem::forget(inner);
        }
    }
}

impl Qwen35VlVisionDevice {
    pub fn new(
        config: &Qwen35VlConfig,
        checkpoint: &Checkpoint,
        context: &Arc<Context>,
        stream: &Arc<Stream>,
        lt: &LtProvider,
    ) -> Result<Self, BackendError> {
        if config.text.dtype != Dtype::Bf16 {
            return Err(start(
                "device vision supports BF16 only; F16/F32 are not attached as device recipes",
            ));
        }
        if !config.vision.deepstack_visual_indexes.is_empty() {
            return Err(start(
                "published Qwen3.5 roots use no deepstack; non-empty deepstack device execution is not implemented",
            ));
        }
        let expected = expected_vision_tensors(config)?;
        let mut uploader = WeightUploader::new(stream.clone())?;
        let mut weights = BTreeMap::new();
        for tensor in &expected {
            let buffer = upload(&mut uploader, checkpoint, &tensor.name, &tensor.shape)?;
            if weights.insert(tensor.name.clone(), buffer).is_some() {
                return Err(start(format!("duplicate vision tensor {}", tensor.name)));
            }
        }
        uploader.finish()?;
        let exact = config.text.dtype == Dtype::F16 && config.model_type.starts_with("qwen3_5");
        let (patch_t0, patch_t1, zero_patch_bias) = if exact {
            let (first, second) = read_deinterleaved_patch(checkpoint, config)?;
            let hidden = config.vision.hidden_size;
            (
                Some(DeviceBuffer::from_host(stream, &first).map_err(infer)?),
                Some(DeviceBuffer::from_host(stream, &second).map_err(infer)?),
                Some(
                    DeviceBuffer::from_host(stream, &vec![Bf16::from_bits(0); hidden])
                        .map_err(infer)?,
                ),
            )
        } else {
            (None, None, None)
        };
        let hidden = config.vision.hidden_size;
        let patch_dim = config.vision.patch_dim()?;
        let temporal_width = config.vision.temporal_patch_dim()?;
        let grouped = product(hidden, config.vision.merge_area()?)?;
        let out_hidden = config.vision.out_hidden_size;
        let mlp_fc1 = lt.prepare_bias_shape::<Bf16>(hidden, config.vision.intermediate_size)?;
        let mut merger_fc1 = Vec::with_capacity(1 + config.vision.deepstack_visual_indexes.len());
        merger_fc1.push(lt.prepare_bias_shape::<Bf16>(grouped, grouped)?);
        let mut merger_fc2 = Vec::with_capacity(1 + config.vision.deepstack_visual_indexes.len());
        merger_fc2.push(lt.prepare_bias_shape::<Bf16>(grouped, out_hidden)?);
        for _ in 0..config.vision.deepstack_visual_indexes.len() {
            merger_fc1.push(lt.prepare_bias_shape::<Bf16>(grouped, grouped)?);
            merger_fc2.push(lt.prepare_bias_shape::<Bf16>(grouped, out_hidden)?);
        }
        let spec = AttentionSpec {
            heads: config.vision.num_heads,
            kv_heads: config.vision.num_heads,
            head_dim: config.vision.head_dim(),
            causal: false,
            window: Window::Global,
            dtype: DType::Bf16,
            scale: Scale::RsqrtHeadDim,
        };
        if !Fa4::supports(&spec) {
            return Err(start(format!(
                "vision attention H{} d{} has no qualified non-causal FA4 export",
                config.vision.num_heads,
                config.vision.head_dim()
            )));
        }
        let fa4_root = std::env::var_os(infinity_v3_config::infinity_env!("FA4_ROOT"))
            .ok_or_else(|| start("vision attention requires INFINITY_FA4_ROOT"))?;
        let fa4 = Fa4::new(context.clone(), Path::new(&fa4_root), spec)?;
        Ok(Self {
            config: config.clone(),
            weights,
            patch_t0,
            patch_t1,
            zero_patch_bias,
            plans: RefCell::new(VisionPlans {
                patch_flat: lt.prepare_bias_shape::<Bf16>(patch_dim, hidden)?,
                patch_t0: lt.prepare_bias_shape::<Bf16>(temporal_width, hidden)?,
                patch_t1: lt.prepare_bias_shape::<Bf16>(temporal_width, hidden)?,
                qkv: lt.prepare_bias_shape::<Bf16>(hidden, product(hidden, 3)?)?,
                projection: lt.prepare_bias_shape::<Bf16>(hidden, hidden)?,
                mlp_fc1,
                mlp_fc2: lt.prepare_bias_shape::<Bf16>(config.vision.intermediate_size, hidden)?,
                merger_fc1,
                merger_fc2,
            }),
            kernels: VisionKernels::load(context)?,
            fa4,
            side: config.vision.position_side(),
            stream: stream.clone(),
        })
    }

    fn weight(&self, suffix: &str) -> Result<&DeviceBuffer<Bf16>, BackendError> {
        self.weights
            .get(&format!("model.visual.{suffix}"))
            .ok_or_else(|| start(format!("missing uploaded vision tensor {suffix}")))
    }

    fn norm(
        &self,
        input: &DeviceBuffer<Bf16>,
        gamma: &DeviceBuffer<Bf16>,
        beta: &DeviceBuffer<Bf16>,
        output: &mut DeviceBuffer<Bf16>,
        rows: usize,
        width: usize,
    ) -> Result<(), BackendError> {
        let mut input = input.cu_deviceptr();
        let mut residual = 0u64;
        let mut bias = 0u64;
        let mut gamma = gamma.cu_deviceptr();
        let mut beta = beta.cu_deviceptr();
        let mut output = output.cu_deviceptr();
        let mut rows = u32_size(rows)?;
        let mut width = u32_size(width)?;
        let mut eps = 1e-6f32;
        launch_threads(
            &self.kernels.layer_norm,
            &self.stream,
            (rows, 1, 1),
            256,
            &mut [
                arg(&mut input),
                arg(&mut residual),
                arg(&mut bias),
                arg(&mut gamma),
                arg(&mut beta),
                arg(&mut output),
                arg(&mut rows),
                arg(&mut width),
                arg(&mut eps),
            ],
        )
    }

    fn add(
        &self,
        left: &DeviceBuffer<Bf16>,
        right: &DeviceBuffer<Bf16>,
        output: &mut DeviceBuffer<Bf16>,
        elements: usize,
    ) -> Result<(), BackendError> {
        let mut left = left.cu_deviceptr();
        let mut right = right.cu_deviceptr();
        let mut output = output.cu_deviceptr();
        let mut count = u32_size(elements)?;
        launch(
            &self.kernels.add,
            &self.stream,
            blocks(elements)?,
            &mut [
                arg(&mut left),
                arg(&mut right),
                arg(&mut output),
                arg(&mut count),
            ],
        )
    }

    fn add_assign(
        &self,
        states: &mut DeviceBuffer<Bf16>,
        right: &DeviceBuffer<Bf16>,
        elements: usize,
    ) -> Result<(), BackendError> {
        let mut states_ptr = states.cu_deviceptr();
        let mut right_ptr = right.cu_deviceptr();
        let mut count = u32_size(elements)?;
        launch(
            &self.kernels.add,
            &self.stream,
            blocks(elements)?,
            &mut [
                arg(&mut states_ptr),
                arg(&mut right_ptr),
                arg(&mut states_ptr),
                arg(&mut count),
            ],
        )
    }

    fn add_bias(
        &self,
        states: &mut DeviceBuffer<Bf16>,
        bias: &DeviceBuffer<Bf16>,
        rows: usize,
        width: usize,
    ) -> Result<(), BackendError> {
        let mut states = states.cu_deviceptr();
        let mut bias = bias.cu_deviceptr();
        let mut count = u32_size(product(rows, width)?)?;
        let mut width = u32_size(width)?;
        launch(
            &self.kernels.bias,
            &self.stream,
            blocks(product(rows, width as usize)?)?,
            &mut [
                arg(&mut states),
                arg(&mut bias),
                arg(&mut count),
                arg(&mut width),
            ],
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn linear(
        &self,
        lt: &LtProvider,
        plan: &mut LtPlan<Bf16>,
        rows: usize,
        input: &DeviceBuffer<Bf16>,
        weight: &DeviceBuffer<Bf16>,
        bias: &DeviceBuffer<Bf16>,
        output: &mut DeviceBuffer<Bf16>,
    ) -> Result<(), BackendError> {
        lt.set_logical_rows(plan, rows)
            .map_err(|error| start(format!("vision plan rows: {error:?}")))?;
        // SAFETY: the caller retains every buffer until the workspace drains.
        unsafe { lt.enqueue(plan, input, weight, bias, output) }
    }

    fn activate(
        &self,
        buffer: &mut DeviceBuffer<Bf16>,
        elements: usize,
        silu: bool,
    ) -> Result<(), BackendError> {
        let mut pointer = buffer.cu_deviceptr();
        let mut count = u32_size(elements)?;
        let function = if silu {
            &self.kernels.silu
        } else {
            &self.kernels.gelu_tanh
        };
        launch(
            function,
            &self.stream,
            blocks(elements)?,
            &mut [arg(&mut pointer), arg(&mut count)],
        )
    }

    /// Execute one image worth of packed patches and return device-resident
    /// merged rows. No host round trip or stream synchronization occurs.
    pub(crate) fn forward_image_device(
        &self,
        lt: &LtProvider,
        image: &HostImage,
    ) -> Result<VisionDeviceOutput, BackendError> {
        let config = &self.config;
        let vision = &config.vision;
        let hidden = vision.hidden_size;
        let patch_dim = vision.patch_dim()?;
        let temporal_width = vision.temporal_patch_dim()?;
        let head_dim = vision.head_dim();
        let heads = vision.num_heads;
        let rows = image.patch_rows()?;
        let merged = image.merged_rows()?;
        let grouped_width = product(hidden, vision.merge_area()?)?;
        let out_hidden = vision.out_hidden_size;
        if image.pixels.len() != product(rows, patch_dim)? {
            return Err(start("packed pixel payload length mismatch"));
        }
        let workspace = VisionWorkspace::new(
            &self.stream,
            rows,
            merged,
            hidden,
            vision.intermediate_size,
            patch_dim,
            temporal_width,
            grouped_width,
            out_hidden,
            vision.deepstack_visual_indexes.len(),
        )?;
        let mut output = VisionDeviceOutput::new(workspace, self.stream.clone());
        output.set_geometry(merged, out_hidden);
        // SAFETY: `output` is the sole owner of this workspace and outlives the
        // raw borrow; pins/workspace are fenced by VisionDeviceOutput::Drop.
        let ws = unsafe { &mut *output.workspace_ptr() };
        let pixels_pin = HostPin::f32(&self.stream, &image.pixels)?;
        // SAFETY: `pixels_pin` is retained in VisionDeviceOutput until the
        // outer request fence completes.
        let pixels = unsafe { DeviceBuffer::from_pinned_host(&self.stream, pixels_pin.as_f32()) }
            .map_err(infer)?;
        output.push_pin(pixels_pin);
        output.set_pixels(pixels);
        {
            let mut source = output.pixels_ptr();
            let mut destination = ws.pixels_bf16.cu_deviceptr();
            let mut count = u32_size(product(rows, patch_dim)?)?;
            launch(
                &self.kernels.f32_to_bf16,
                &self.stream,
                blocks(product(rows, patch_dim)?)?,
                &mut [arg(&mut source), arg(&mut destination), arg(&mut count)],
            )?;
        }
        let mut plans = self.plans.borrow_mut();
        if let (Some(patch_t0), Some(patch_t1), Some(zero_patch_bias)) = (
            self.patch_t0.as_ref(),
            self.patch_t1.as_ref(),
            self.zero_patch_bias.as_ref(),
        ) {
            for slice in [0usize, 1usize] {
                let plan = if slice == 0 {
                    &mut plans.patch_t0
                } else {
                    &mut plans.patch_t1
                };
                {
                    let mut source = ws.pixels_bf16.cu_deviceptr();
                    let mut destination = ws.patch_slice.cu_deviceptr();
                    let mut rows_u32 = u32_size(rows)?;
                    let mut channels = u32_size(vision.in_channels)?;
                    let mut temporal = u32_size(vision.temporal_patch_size)?;
                    let mut patch = u32_size(vision.patch_size)?;
                    let mut slice_u32 = u32_size(slice)?;
                    launch(
                        &self.kernels.temporal_slice,
                        &self.stream,
                        blocks(product(rows, temporal_width)?)?,
                        &mut [
                            arg(&mut source),
                            arg(&mut destination),
                            arg(&mut rows_u32),
                            arg(&mut channels),
                            arg(&mut temporal),
                            arg(&mut patch),
                            arg(&mut slice_u32),
                        ],
                    )?;
                }
                let weight = if slice == 0 { patch_t0 } else { patch_t1 };
                let output = if slice == 0 {
                    &mut ws.states
                } else {
                    &mut ws.branch
                };
                lt.set_logical_rows(plan, rows)
                    .map_err(|error| start(format!("vision patch plan: {error:?}")))?;
                // SAFETY: both outputs and inputs remain owned by `workspace`.
                unsafe {
                    lt.enqueue(plan, &ws.patch_slice, weight, zero_patch_bias, output)?;
                }
            }
            self.add_assign(&mut ws.states, &ws.branch, product(rows, hidden)?)?;
            self.add_bias(
                &mut ws.states,
                self.weight("patch_embed.proj.bias")?,
                rows,
                hidden,
            )?;
        } else {
            lt.set_logical_rows(&mut plans.patch_flat, rows)
                .map_err(|error| start(format!("vision patch plan: {error:?}")))?;
            // SAFETY: the flattened Conv3D weight and pixel rows are retained.
            unsafe {
                lt.enqueue(
                    &plans.patch_flat,
                    &ws.pixels_bf16,
                    self.weight("patch_embed.proj.weight")?,
                    self.weight("patch_embed.proj.bias")?,
                    &mut ws.states,
                )?;
            }
        }
        {
            let mut table = self.weight("pos_embed.weight")?.cu_deviceptr();
            let mut states = ws.states.cu_deviceptr();
            let mut tokens = u32_size(rows)?;
            let mut hidden_u32 = u32_size(hidden)?;
            let mut side = u32_size(self.side)?;
            let mut height = u32_size(image.grid.height)?;
            let mut width = u32_size(image.grid.width)?;
            let mut merge = u32_size(image.grid.merge)?;
            launch(
                &self.kernels.pos_interp,
                &self.stream,
                blocks(product(rows, hidden)?)?,
                &mut [
                    arg(&mut table),
                    arg(&mut states),
                    arg(&mut tokens),
                    arg(&mut hidden_u32),
                    arg(&mut side),
                    arg(&mut height),
                    arg(&mut width),
                    arg(&mut merge),
                ],
            )?;
        }
        let frame_len = product(image.grid.height, image.grid.width)?;
        let mut frames = Vec::with_capacity(image.grid.temporal + 1);
        frames.push(0u32);
        for frame in 0..image.grid.temporal {
            frames.push(u32_size(product(frame + 1, frame_len)?)?);
        }
        let frame_pin = HostPin::u32(&self.stream, &frames)?;
        // SAFETY: `frame_pin` is retained in VisionDeviceOutput until the
        // outer request fence completes.
        let frame_offsets =
            unsafe { DeviceBuffer::from_pinned_host(&self.stream, frame_pin.as_u32()) }
                .map_err(infer)?;
        output.push_pin(frame_pin);
        output.set_frame_offsets(frame_offsets);
        let qkv_stride = u64::try_from(product(hidden, 3)?).map_err(infer)?;
        let hidden_bytes = u64::try_from(product(hidden, 2)?).map_err(infer)?;
        let strides = [
            qkv_stride as i64,
            head_dim as i64,
            1,
            qkv_stride as i64,
            head_dim as i64,
            1,
            qkv_stride as i64,
            head_dim as i64,
            1,
            hidden as i64,
            head_dim as i64,
            1,
        ];
        for layer in 0..vision.depth {
            let prefix = format!("blocks.{layer}");
            self.norm(
                &ws.states,
                self.weight(&format!("{prefix}.norm1.weight"))?,
                self.weight(&format!("{prefix}.norm1.bias"))?,
                &mut ws.norm,
                rows,
                hidden,
            )?;
            self.linear(
                lt,
                &mut plans.qkv,
                rows,
                &ws.norm,
                self.weight(&format!("{prefix}.attn.qkv.weight"))?,
                self.weight(&format!("{prefix}.attn.qkv.bias"))?,
                &mut ws.qkv,
            )?;
            {
                let qkv = ws.qkv.cu_deviceptr();
                let mut q = qkv;
                let mut k = qkv
                    .checked_add(hidden_bytes)
                    .ok_or_else(|| infer("vision K address overflow"))?;
                let mut heads = u32_size(heads)?;
                let mut head_dim = u32_size(head_dim)?;
                let mut tokens = u32_size(rows)?;
                let mut height = u32_size(image.grid.height)?;
                let mut width = u32_size(image.grid.width)?;
                let mut merge = u32_size(image.grid.merge)?;
                let mut q_stride = qkv_stride;
                let mut k_stride = qkv_stride;
                launch(
                    &self.kernels.rope2d,
                    &self.stream,
                    blocks(product(
                        rows,
                        product(heads as usize, head_dim as usize / 2)?,
                    )?)?,
                    &mut [
                        arg(&mut q),
                        arg(&mut k),
                        arg(&mut heads),
                        arg(&mut head_dim),
                        arg(&mut tokens),
                        arg(&mut height),
                        arg(&mut width),
                        arg(&mut merge),
                        arg(&mut q_stride),
                        arg(&mut k_stride),
                    ],
                )?;
                let v = qkv
                    .checked_add(hidden_bytes * 2)
                    .ok_or_else(|| infer("vision V address overflow"))?;
                let attention = AttentionBatch {
                    q: qkv,
                    k: qkv
                        .checked_add(hidden_bytes)
                        .ok_or_else(|| infer("vision K address overflow"))?,
                    v,
                    out: ws.attention.cu_deviceptr(),
                    q_offsets: output.frame_offsets_ptr(),
                    k_offsets: output.frame_offsets_ptr(),
                    q_tokens: rows,
                    k_tokens: rows,
                    batch: image.grid.temporal,
                    max_q: frame_len,
                    strides,
                };
                // SAFETY: packed QKV/out/offsets remain owned through drain.
                unsafe { self.fa4.forward_batch_bound(&self.stream, &attention)? };
            }
            self.linear(
                lt,
                &mut plans.projection,
                rows,
                &ws.attention,
                self.weight(&format!("{prefix}.attn.proj.weight"))?,
                self.weight(&format!("{prefix}.attn.proj.bias"))?,
                &mut ws.branch,
            )?;
            self.add_assign(&mut ws.states, &ws.branch, product(rows, hidden)?)?;
            self.norm(
                &ws.states,
                self.weight(&format!("{prefix}.norm2.weight"))?,
                self.weight(&format!("{prefix}.norm2.bias"))?,
                &mut ws.norm,
                rows,
                hidden,
            )?;
            self.linear(
                lt,
                &mut plans.mlp_fc1,
                rows,
                &ws.norm,
                self.weight(&format!("{prefix}.mlp.linear_fc1.weight"))?,
                self.weight(&format!("{prefix}.mlp.linear_fc1.bias"))?,
                &mut ws.mlp,
            )?;
            self.activate(
                &mut ws.mlp,
                product(rows, vision.intermediate_size)?,
                vision.hidden_act.as_str() == "silu",
            )?;
            self.linear(
                lt,
                &mut plans.mlp_fc2,
                rows,
                &ws.mlp,
                self.weight(&format!("{prefix}.mlp.linear_fc2.weight"))?,
                self.weight(&format!("{prefix}.mlp.linear_fc2.bias"))?,
                &mut ws.branch,
            )?;
            self.add_assign(&mut ws.states, &ws.branch, product(rows, hidden)?)?;
        }
        // Final merger: normalize each hidden row, then reinterpret the
        // merge-major contiguous rows as `[merged, hidden * merge^2]`.
        self.norm(
            &ws.states,
            self.weight("merger.norm.weight")?,
            self.weight("merger.norm.bias")?,
            &mut ws.norm,
            rows,
            hidden,
        )?;
        self.linear(
            lt,
            &mut plans.merger_fc1[0],
            merged,
            &ws.norm,
            self.weight("merger.linear_fc1.weight")?,
            self.weight("merger.linear_fc1.bias")?,
            &mut ws.merger_tmp,
        )?;
        self.activate(&mut ws.merger_tmp, product(merged, grouped_width)?, false)?;
        self.linear(
            lt,
            &mut plans.merger_fc2[0],
            merged,
            &ws.merger_tmp,
            self.weight("merger.linear_fc2.weight")?,
            self.weight("merger.linear_fc2.bias")?,
            &mut ws.image_merged,
        )?;
        Ok(output)
    }

    /// Host-visible reference path used by CPU tests and `tower-cpu`. The
    /// serving path uses `forward_image_device` and never synchronizes here.
    pub fn forward_image(
        &self,
        lt: &LtProvider,
        image: &HostImage,
    ) -> Result<VisionOutput, BackendError> {
        let mut output = self.forward_image_device(lt, image)?;
        let merged_rows = output
            .workspace()
            .image_merged
            .to_host_vec(&self.stream)
            .map_err(infer)?;
        let mut merged_host = Vec::with_capacity(merged_rows.len());
        for value in merged_rows {
            merged_host.push(half::bf16::from_bits(value.to_bits()).to_f32());
        }
        let mut deepstack = Vec::with_capacity(output.workspace().image_deepstack.len());
        for buffer in &output.workspace().image_deepstack {
            let values = buffer.to_host_vec(&self.stream).map_err(infer)?;
            deepstack.push(
                values
                    .into_iter()
                    .map(|value| half::bf16::from_bits(value.to_bits()).to_f32())
                    .collect(),
            );
        }
        output.workspace_mut().drain()?;
        Ok(VisionOutput {
            merged: merged_host,
            merged_rows: output.merged_rows(),
            out_hidden: output.out_hidden(),
            deepstack,
        })
    }
}

pub(crate) struct VisionWorkspace {
    stream: Arc<Stream>,
    pixels_bf16: DeviceBuffer<Bf16>,
    patch_slice: DeviceBuffer<Bf16>,
    states: DeviceBuffer<Bf16>,
    norm: DeviceBuffer<Bf16>,
    qkv: DeviceBuffer<Bf16>,
    attention: DeviceBuffer<Bf16>,
    branch: DeviceBuffer<Bf16>,
    mlp: DeviceBuffer<Bf16>,
    merger_tmp: DeviceBuffer<Bf16>,
    pub(crate) image_merged: DeviceBuffer<Bf16>,
    image_deepstack: Vec<DeviceBuffer<Bf16>>,
    drained: bool,
}

impl VisionWorkspace {
    #[allow(clippy::too_many_arguments)]
    fn new(
        stream: &Arc<Stream>,
        rows: usize,
        merged: usize,
        hidden: usize,
        intermediate: usize,
        patch_dim: usize,
        temporal_width: usize,
        grouped: usize,
        out_hidden: usize,
        deepstack: usize,
    ) -> Result<Self, BackendError> {
        let bf16 = |len: usize| zeroed::<Bf16>(stream, len);
        let mut image_deepstack = Vec::with_capacity(deepstack);
        for _ in 0..deepstack {
            image_deepstack.push(bf16(product(merged, out_hidden)?)?);
        }
        Ok(Self {
            stream: stream.clone(),
            pixels_bf16: bf16(product(rows, patch_dim)?)?,
            patch_slice: bf16(product(rows, temporal_width)?)?,
            states: bf16(product(rows, hidden)?)?,
            norm: bf16(product(rows, hidden.max(grouped))?)?,
            qkv: bf16(product(rows, product(hidden, 3)?)?)?,
            attention: bf16(product(rows, hidden)?)?,
            branch: bf16(product(rows, hidden)?)?,
            mlp: bf16(product(rows, intermediate)?)?,
            merger_tmp: bf16(product(merged, grouped)?)?,
            image_merged: bf16(product(merged, out_hidden)?)?,
            image_deepstack,
            drained: false,
        })
    }

    fn drain(&mut self) -> Result<(), BackendError> {
        if !self.drained {
            self.stream.synchronize().map_err(infer)?;
            self.drained = true;
        }
        Ok(())
    }

    /// Record that the outer HostIo lease already observed stream completion.
    /// Drop then releases without a second synchronization.
    pub(crate) fn mark_completed(&mut self) {
        self.drained = true;
    }
}

impl Drop for VisionWorkspace {
    fn drop(&mut self) {
        if !self.drained {
            let _ = self.stream.synchronize();
            self.drained = true;
        }
    }
}
