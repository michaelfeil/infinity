//! Concrete Qwen3-VL device executor.
//!
//! This module performs real cold BF16 weight upload, owns reusable device
//! workspaces, prepares cuBLASLt plans and FA4 attention, and executes the
//! vision tower plus the shared dense decoder loop. It does not use a test sink
//! or a no-op fallback: the only execution path launches the real primitives.

use crate::cublas::Blas;
use crate::decoder::{DecoderModel, DecoderMultimodal, DecoderWork};
use crate::decoder_model::{DecoderConfig, DecoderWeights};
use crate::fa4::{AttentionBatch, AttentionSpec, DType, Fa4, Scale, Window};
use crate::lt::{LtPlan, LtProvider};
use crate::model::Checkpoint;
use crate::qwen3_vl_config::{Dtype, Qwen3VlConfig};
use crate::qwen3_vl_inventory::{expected_tensors, InventoryScope};
use crate::qwen3_vl_text::{image_position_ids, mrope_axis_map, mrope_frequencies, mrope_phases};
use crate::qwen3_vl_vision::{deepstack_capture_order, ImageGrid};
use crate::staging::HostIoLease;
use crate::tensor::Bf16;
use crate::upload::WeightUploader;
use crate::WorkKey;
use cuda_core::{
    launch_kernel, Context, DeviceBuffer, DeviceCopy, Function, PinnedHostBuffer, Stream,
};
use infinity_v3_backend_core::{BackendError, Batch, Pool};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::ffi::c_void;
use std::mem::ManuallyDrop;
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;

fn start(error: impl std::fmt::Display) -> BackendError {
    BackendError::Start(format!("Oxide Qwen3-VL device: {error}"))
}

fn infer(error: impl std::fmt::Display) -> BackendError {
    BackendError::Inference(format!("Oxide Qwen3-VL device: {error}"))
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
    function: &Function,
    stream: &Stream,
    grid: (u32, u32, u32),
    shared: u32,
    args: &mut [*mut c_void],
) -> Result<(), BackendError> {
    // SAFETY: callers retain every buffer through stream completion and pass
    // exactly the kernel ABI's scalar/pointer arguments.
    unsafe {
        launch_kernel(
            function.cu_function(),
            grid,
            (256, 1, 1),
            shared,
            stream.cu_stream(),
            args,
        )
    }
    .map_err(infer)
}

fn launch_threads(
    function: &Function,
    stream: &Stream,
    grid: (u32, u32, u32),
    threads: u32,
    shared: u32,
    args: &mut [*mut c_void],
) -> Result<(), BackendError> {
    unsafe {
        launch_kernel(
            function.cu_function(),
            grid,
            (threads, 1, 1),
            shared,
            stream.cu_stream(),
            args,
        )
    }
    .map_err(infer)
}

fn zero_alloc<T: DeviceCopy>(
    stream: &Arc<Stream>,
    len: usize,
) -> Result<DeviceBuffer<T>, BackendError> {
    // SAFETY: the complete buffer is zeroed on the ordered stream before use.
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

/// Copy the active prefix of a host slice into a larger reusable device buffer.
/// `src` must outlive the ordered copy. This does not assert exact capacity.
unsafe fn copy_prefix<T>(
    destination: &DeviceBuffer<T>,
    stream: &Stream,
    source: &[T],
) -> Result<(), BackendError> {
    if source.is_empty() {
        return Ok(());
    }
    let bytes = source
        .len()
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| infer("host copy size overflow"))?;
    // SAFETY: destination capacity is checked by the workspace and `source`
    // outlives the ordered copy contract at every call site.
    unsafe {
        cuda_core::memory::memcpy_htod_async(
            destination.cu_deviceptr(),
            source.as_ptr(),
            bytes,
            stream.cu_stream(),
        )
    }
    .map_err(infer)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct VisionWorkKey {
    patches: usize,
    merged: usize,
    frames: usize,
    tokens: usize,
    staged_pixels: usize,
    staged_frames: usize,
}

/// Pinned host arena capacities are request totals, not worst-single-image
/// maxima. Device pixels/frame_offsets stay max-sized and stream-ordered.
fn pinned_staging_capacities(key: &VisionWorkKey) -> (usize, usize) {
    (key.staged_pixels, key.staged_frames)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PinnedStagingPlan {
    pub(crate) pixel_ranges: Vec<(usize, usize)>,
    pub(crate) frame_ranges: Vec<(usize, usize)>,
    pub(crate) total_pixels: usize,
    pub(crate) total_frames: usize,
}

/// Compute disjoint immutable pinned source ranges for every image before any
/// async H2D is enqueued.
pub(crate) fn plan_pinned_staging(
    images: &[VisionImage],
) -> Result<PinnedStagingPlan, BackendError> {
    let mut pixel_ranges = Vec::with_capacity(images.len());
    let mut frame_ranges = Vec::with_capacity(images.len());
    let mut total_pixels = 0usize;
    let mut total_frames = 0usize;
    for image in images {
        let patch_rows = image.patch_rows()?;
        let patch_dim = image
            .pixels
            .len()
            .checked_div(patch_rows)
            .filter(|dim| *dim > 0 && patch_rows.checked_mul(*dim) == Some(image.pixels.len()))
            .ok_or_else(|| start(format!("{} has an invalid pixel payload", image.name)))?;
        let pixels = patch_rows
            .checked_mul(patch_dim)
            .ok_or_else(|| start("pinned pixel range overflow"))?;
        let frames = image.grid_thw[0]
            .checked_add(1)
            .ok_or_else(|| start("pinned frame count overflow"))?;
        pixel_ranges.push((total_pixels, pixels));
        frame_ranges.push((total_frames, frames));
        total_pixels = total_pixels
            .checked_add(pixels)
            .ok_or_else(|| start("pinned pixel total overflow"))?;
        total_frames = total_frames
            .checked_add(frames)
            .ok_or_else(|| start("pinned frame total overflow"))?;
    }
    Ok(PinnedStagingPlan {
        pixel_ranges,
        frame_ranges,
        total_pixels,
        total_frames,
    })
}

struct VisionWork {
    stream: Arc<Stream>,
    key: VisionWorkKey,
    bytes: usize,
    pixels: ManuallyDrop<DeviceBuffer<f32>>,
    pixels_bf16: ManuallyDrop<DeviceBuffer<Bf16>>,
    states: ManuallyDrop<DeviceBuffer<Bf16>>,
    norm: ManuallyDrop<DeviceBuffer<Bf16>>,
    qkv: ManuallyDrop<DeviceBuffer<Bf16>>,
    attention: ManuallyDrop<DeviceBuffer<Bf16>>,
    branch: ManuallyDrop<DeviceBuffer<Bf16>>,
    mlp: ManuallyDrop<DeviceBuffer<Bf16>>,
    grouped_norm: ManuallyDrop<DeviceBuffer<Bf16>>,
    merger_tmp: ManuallyDrop<DeviceBuffer<Bf16>>,
    image_merged: ManuallyDrop<DeviceBuffer<Bf16>>,
    image_deepstack: ManuallyDrop<Vec<DeviceBuffer<Bf16>>>,
    merged: ManuallyDrop<DeviceBuffer<Bf16>>,
    deepstack: ManuallyDrop<Vec<DeviceBuffer<Bf16>>>,
    frame_offsets: ManuallyDrop<DeviceBuffer<u32>>,
    visual_rows: ManuallyDrop<DeviceBuffer<u32>>,
    rope_cos: ManuallyDrop<DeviceBuffer<f32>>,
    rope_sin: ManuallyDrop<DeviceBuffer<f32>>,
    rope_positions: ManuallyDrop<DeviceBuffer<u32>>,
    host_pixels: ManuallyDrop<PinnedHostBuffer<f32>>,
    host_frames: ManuallyDrop<PinnedHostBuffer<u32>>,
    host_rows: ManuallyDrop<PinnedHostBuffer<u32>>,
    host_cos: ManuallyDrop<PinnedHostBuffer<f32>>,
    host_sin: ManuallyDrop<PinnedHostBuffer<f32>>,
    host_positions: ManuallyDrop<PinnedHostBuffer<u32>>,
}

impl VisionWork {
    fn new(
        stream: Arc<Stream>,
        config: &Qwen3VlConfig,
        key: VisionWorkKey,
        deepstack: usize,
    ) -> Result<Self, BackendError> {
        let hidden = config.vision.hidden_size;
        let patch_dim = config.vision.patch_dim()?;
        let grouped = product(hidden, config.vision.merge_area()?)?;
        let out_hidden = config.vision.out_hidden_size;
        let intermediate = config.vision.intermediate_size;
        let half = config.text.text_dim() / 2;
        let mut bytes = 0usize;
        let alloc_bf16 =
            |bytes: &mut usize, len: usize| -> Result<DeviceBuffer<Bf16>, BackendError> {
                *bytes = bytes
                    .checked_add(product(len, 2)?)
                    .ok_or_else(|| infer("workspace byte overflow"))?;
                zero_alloc::<Bf16>(&stream, len)
            };
        let pixels_len = product(key.patches, patch_dim)?;
        let hidden_len = product(key.patches, hidden)?;
        let qkv_len = product(key.patches, product(hidden, 3)?)?;
        let mlp_len = product(key.patches, intermediate)?;
        let grouped_len = product(key.merged, grouped)?;
        let out_len = product(key.merged, out_hidden)?;
        let pixels = {
            bytes = bytes
                .checked_add(product(pixels_len, 4)?)
                .ok_or_else(|| infer("pixel byte overflow"))?;
            zero_alloc::<f32>(&stream, pixels_len)?
        };
        let pixels_bf16 = alloc_bf16(&mut bytes, pixels_len)?;
        let states = alloc_bf16(&mut bytes, hidden_len)?;
        let norm = alloc_bf16(&mut bytes, hidden_len)?;
        let qkv = alloc_bf16(&mut bytes, qkv_len)?;
        let attention = alloc_bf16(&mut bytes, hidden_len)?;
        let branch = alloc_bf16(&mut bytes, hidden_len)?;
        let mlp = alloc_bf16(&mut bytes, mlp_len)?;
        let grouped_norm = alloc_bf16(&mut bytes, grouped_len)?;
        let merger_tmp = alloc_bf16(&mut bytes, grouped_len)?;
        let image_merged = alloc_bf16(&mut bytes, out_len)?;
        let mut image_deepstack = Vec::with_capacity(deepstack);
        for _ in 0..deepstack {
            image_deepstack.push(alloc_bf16(&mut bytes, out_len)?);
        }
        let total_out = product(key.merged, out_hidden)?;
        let merged = alloc_bf16(&mut bytes, total_out)?;
        let mut deepstack_total = Vec::with_capacity(deepstack);
        for _ in 0..deepstack {
            deepstack_total.push(alloc_bf16(&mut bytes, total_out)?);
        }
        let frame_offsets = {
            bytes = bytes
                .checked_add(product(key.frames + 1, 4)?)
                .ok_or_else(|| infer("frame byte overflow"))?;
            zero_alloc::<u32>(&stream, key.frames + 1)?
        };
        let visual_rows = {
            bytes = bytes
                .checked_add(product(key.merged, 4)?)
                .ok_or_else(|| infer("row byte overflow"))?;
            zero_alloc::<u32>(&stream, key.merged)?
        };
        let rope_len = product(key.tokens, half)?;
        let rope_cos = {
            bytes = bytes
                .checked_add(product(rope_len, 4)?)
                .ok_or_else(|| infer("mrope byte overflow"))?;
            zero_alloc::<f32>(&stream, rope_len)?
        };
        let rope_sin = {
            bytes = bytes
                .checked_add(product(rope_len, 4)?)
                .ok_or_else(|| infer("mrope byte overflow"))?;
            zero_alloc::<f32>(&stream, rope_len)?
        };
        let rope_positions = {
            bytes = bytes
                .checked_add(product(key.tokens, 4)?)
                .ok_or_else(|| infer("position byte overflow"))?;
            zero_alloc::<u32>(&stream, key.tokens)?
        };
        let pinned = |bytes: &mut usize, len: usize, width: usize| -> Result<usize, BackendError> {
            *bytes = bytes
                .checked_add(product(len, width)?)
                .ok_or_else(|| infer("pinned staging byte overflow"))?;
            Ok(len)
        };
        let (staged_pixels, staged_frames) = pinned_staging_capacities(&key);
        let host_pixels =
            PinnedHostBuffer::zeroed(stream.context(), pinned(&mut bytes, staged_pixels, 4)?)
                .map_err(infer)?;
        let host_frames =
            PinnedHostBuffer::zeroed(stream.context(), pinned(&mut bytes, staged_frames, 4)?)
                .map_err(infer)?;
        let host_rows =
            PinnedHostBuffer::zeroed(stream.context(), pinned(&mut bytes, key.merged, 4)?)
                .map_err(infer)?;
        let host_cos = PinnedHostBuffer::zeroed(stream.context(), pinned(&mut bytes, rope_len, 4)?)
            .map_err(infer)?;
        let host_sin = PinnedHostBuffer::zeroed(stream.context(), pinned(&mut bytes, rope_len, 4)?)
            .map_err(infer)?;
        let host_positions =
            PinnedHostBuffer::zeroed(stream.context(), pinned(&mut bytes, key.tokens, 4)?)
                .map_err(infer)?;
        crate::LIVE_ARENA_BYTES.fetch_add(bytes, Ordering::Relaxed);
        crate::record_live_arena_bytes();
        Ok(Self {
            stream,
            key,
            bytes,
            pixels: ManuallyDrop::new(pixels),
            pixels_bf16: ManuallyDrop::new(pixels_bf16),
            states: ManuallyDrop::new(states),
            norm: ManuallyDrop::new(norm),
            qkv: ManuallyDrop::new(qkv),
            attention: ManuallyDrop::new(attention),
            branch: ManuallyDrop::new(branch),
            mlp: ManuallyDrop::new(mlp),
            grouped_norm: ManuallyDrop::new(grouped_norm),
            merger_tmp: ManuallyDrop::new(merger_tmp),
            image_merged: ManuallyDrop::new(image_merged),
            image_deepstack: ManuallyDrop::new(image_deepstack),
            merged: ManuallyDrop::new(merged),
            deepstack: ManuallyDrop::new(deepstack_total),
            frame_offsets: ManuallyDrop::new(frame_offsets),
            visual_rows: ManuallyDrop::new(visual_rows),
            rope_cos: ManuallyDrop::new(rope_cos),
            rope_sin: ManuallyDrop::new(rope_sin),
            rope_positions: ManuallyDrop::new(rope_positions),
            host_pixels: ManuallyDrop::new(host_pixels),
            host_frames: ManuallyDrop::new(host_frames),
            host_rows: ManuallyDrop::new(host_rows),
            host_cos: ManuallyDrop::new(host_cos),
            host_sin: ManuallyDrop::new(host_sin),
            host_positions: ManuallyDrop::new(host_positions),
        })
    }

    fn fits(&self, key: VisionWorkKey, deepstack: usize) -> bool {
        // Exact geometry: decoder hooks use DeviceBuffer::len() as the active
        // row/token count, so capacity-only reuse would expose stale rows.
        self.key == key && self.deepstack.len() == deepstack
    }
}

impl Drop for VisionWork {
    fn drop(&mut self) {
        if self.stream.synchronize().is_ok() {
            unsafe {
                ManuallyDrop::drop(&mut self.pixels);
                ManuallyDrop::drop(&mut self.pixels_bf16);
                ManuallyDrop::drop(&mut self.states);
                ManuallyDrop::drop(&mut self.norm);
                ManuallyDrop::drop(&mut self.qkv);
                ManuallyDrop::drop(&mut self.attention);
                ManuallyDrop::drop(&mut self.branch);
                ManuallyDrop::drop(&mut self.mlp);
                ManuallyDrop::drop(&mut self.grouped_norm);
                ManuallyDrop::drop(&mut self.merger_tmp);
                ManuallyDrop::drop(&mut self.image_merged);
                ManuallyDrop::drop(&mut self.image_deepstack);
                ManuallyDrop::drop(&mut self.merged);
                ManuallyDrop::drop(&mut self.deepstack);
                ManuallyDrop::drop(&mut self.frame_offsets);
                ManuallyDrop::drop(&mut self.visual_rows);
                ManuallyDrop::drop(&mut self.rope_cos);
                ManuallyDrop::drop(&mut self.rope_sin);
                ManuallyDrop::drop(&mut self.rope_positions);
                ManuallyDrop::drop(&mut self.host_pixels);
                ManuallyDrop::drop(&mut self.host_frames);
                ManuallyDrop::drop(&mut self.host_rows);
                ManuallyDrop::drop(&mut self.host_cos);
                ManuallyDrop::drop(&mut self.host_sin);
                ManuallyDrop::drop(&mut self.host_positions);
            }
            crate::LIVE_ARENA_BYTES.fetch_sub(self.bytes, Ordering::Relaxed);
            crate::record_live_arena_bytes();
        }
    }
}

/// Retained external-mRoPE tables for a text-only request (no visual rows).
/// `rows`/`visual` are one-element dummies so the decoder ABI can still carry
/// a zero logical row count; the text suffix keeps its real phases.
struct PositionWork {
    stream: Arc<Stream>,
    tokens: usize,
    cos: DeviceBuffer<f32>,
    sin: DeviceBuffer<f32>,
    positions: DeviceBuffer<u32>,
    rows: DeviceBuffer<u32>,
    visual: DeviceBuffer<Bf16>,
    host_cos: PinnedHostBuffer<f32>,
    host_sin: PinnedHostBuffer<f32>,
    host_positions: PinnedHostBuffer<u32>,
}

impl PositionWork {
    fn new(stream: Arc<Stream>, tokens: usize, half: usize) -> Result<Self, BackendError> {
        let count = product(tokens, half)?;
        let cos = zero_alloc::<f32>(&stream, count)?;
        let sin = zero_alloc::<f32>(&stream, count)?;
        let positions = zero_alloc::<u32>(&stream, tokens)?;
        let rows = zero_alloc::<u32>(&stream, 1)?;
        let visual = zero_alloc::<Bf16>(&stream, 1)?;
        let host_cos = PinnedHostBuffer::zeroed(stream.context(), count).map_err(infer)?;
        let host_sin = PinnedHostBuffer::zeroed(stream.context(), count).map_err(infer)?;
        let host_positions = PinnedHostBuffer::zeroed(stream.context(), tokens).map_err(infer)?;
        Ok(Self {
            stream,
            tokens,
            cos,
            sin,
            positions,
            rows,
            visual,
            host_cos,
            host_sin,
            host_positions,
        })
    }

    fn fits(&self, tokens: usize) -> bool {
        tokens <= self.tokens
    }

    fn bytes(&self) -> usize {
        self.cos.num_bytes()
            + self.sin.num_bytes()
            + self.positions.num_bytes()
            + self.rows.num_bytes()
            + self.visual.num_bytes()
            + self.host_cos.num_bytes()
            + self.host_sin.num_bytes()
            + self.host_positions.num_bytes()
    }
}

/// Per-request device workspaces. Two may be in flight (A/B) and a larger
/// request may allocate a new one; completed workspaces are returned to the
/// engine's free list only after the same-stream fence.
#[derive(Default)]
pub(crate) struct Qwen3VlWorkspaces {
    vision_work: Option<VisionWork>,
    text_media: Option<PositionWork>,
    decoder_work: Option<DecoderWork>,
}

impl Qwen3VlWorkspaces {
    pub(crate) fn bytes(&self) -> usize {
        self.vision_work.as_ref().map_or(0, |work| work.bytes)
            + self.text_media.as_ref().map_or(0, PositionWork::bytes)
            + self.decoder_work.as_ref().map_or(0, DecoderWork::bytes)
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Qwen3VlPending {
    pub(crate) elements: usize,
    pub(crate) tokens: usize,
    pub(crate) sequences: usize,
    pub(crate) hidden: usize,
}

pub(crate) enum Qwen3VlForwardResult {
    Ready(Qwen3VlOutput),
    Launched(Qwen3VlPending),
}

/// Pack ids/positions/types/boundaries/selected into a caller-owned pinned
/// input slab using the same `InputLayout` as the decoder. Returns word count.
pub(crate) fn pack_decoder_metadata(
    out: &mut [u32],
    ids: &[u32],
    positions: &[u32],
    types: &[u32],
    boundaries: &[u32],
    selected: &[u32],
) -> Result<usize, BackendError> {
    if positions.len() != ids.len() || types.len() != ids.len() {
        return Err(start("decoder metadata token arrays disagree"));
    }
    let layout = crate::InputLayout::new(ids.len(), boundaries.len(), selected.len())?;
    let elements = layout.words();
    if out.len() < elements {
        return Err(start("pinned decoder metadata slab is too small"));
    }
    let mut place = |offset: usize, values: &[u32]| -> Result<(), BackendError> {
        let end = offset
            .checked_add(values.len())
            .ok_or_else(|| start("decoder metadata offset overflow"))?;
        out.get_mut(offset..end)
            .ok_or_else(|| start("decoder metadata exceeds pinned slab"))?
            .copy_from_slice(values);
        Ok(())
    };
    place(layout.ids().get(), ids)?;
    place(layout.positions().get(), positions)?;
    place(layout.types().get(), types)?;
    place(layout.boundaries().get(), boundaries)?;
    place(layout.selections().get(), selected)?;
    Ok(elements)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum VisionActivation {
    Tanh,
    Exact,
    Silu,
}

struct VisionKernels {
    layer_norm: Function,
    add: Function,
    f32_to_bf16: Function,
    gelu_exact: Function,
    gelu_tanh: Function,
    silu: Function,
    pos_interp: Function,
    rope2d: Function,
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
                .load_function("infinity_qwen3vl_add_bf16")
                .map_err(start)?,
            f32_to_bf16: module
                .load_function("infinity_f32_to_bf16")
                .map_err(start)?,
            gelu_exact: module.load_function("infinity_gelu_exact").map_err(start)?,
            gelu_tanh: module
                .load_function("infinity_qwen3vl_gelu_tanh_bf16")
                .map_err(start)?,
            silu: module
                .load_function("infinity_qwen3vl_silu_bf16")
                .map_err(start)?,
            pos_interp: module
                .load_function("infinity_qwen3vl_pos_interp")
                .map_err(start)?,
            rope2d: module
                .load_function("infinity_qwen3vl_rope2d_bf16")
                .map_err(start)?,
        })
    }
}

struct VisionLinearPlans {
    patch: LtPlan<Bf16>,
    qkv: LtPlan<Bf16>,
    projection: LtPlan<Bf16>,
    mlp_fc1: LtPlan<Bf16>,
    mlp_fc2: LtPlan<Bf16>,
    merger_fc1: Vec<LtPlan<Bf16>>,
    merger_fc2: Vec<LtPlan<Bf16>>,
}

struct Qwen3VlVision {
    config: Qwen3VlConfig,
    weights: BTreeMap<String, DeviceBuffer<Bf16>>,
    plans: RefCell<VisionLinearPlans>,
    kernels: VisionKernels,
    fa4: Fa4,
    block_activation: VisionActivation,
    merger_activation: VisionActivation,
    deepstack_capture: Vec<(usize, usize)>,
    side: usize,
    stream: Arc<Stream>,
}

fn upload_tensor(
    uploader: &mut WeightUploader,
    checkpoint: &Checkpoint,
    name: &str,
    shape: &[usize],
) -> Result<DeviceBuffer<Bf16>, BackendError> {
    let source = checkpoint.source_alias(name, shape)?;
    uploader.upload(&[&source])
}

impl Qwen3VlVision {
    fn new(
        config: &Qwen3VlConfig,
        checkpoint: &Checkpoint,
        context: &Arc<Context>,
        stream: &Arc<Stream>,
        lt: &LtProvider,
    ) -> Result<Self, BackendError> {
        let expected = expected_tensors(config, InventoryScope::TowerOnly)?;
        let mut uploader = WeightUploader::new(stream.clone())?;
        let mut weights = BTreeMap::new();
        for tensor in &expected {
            let buffer = upload_tensor(&mut uploader, checkpoint, &tensor.name, &tensor.shape)?;
            if weights.insert(tensor.name.clone(), buffer).is_some() {
                return Err(start(format!("duplicate vision tensor {}", tensor.name)));
            }
        }
        uploader.finish()?;
        let hidden = config.vision.hidden_size;
        let grouped = product(hidden, config.vision.merge_area()?)?;
        let out_hidden = config.vision.out_hidden_size;
        let block_activation = match config.vision.hidden_act.as_str() {
            "gelu" => VisionActivation::Exact,
            "silu" => VisionActivation::Silu,
            _ => VisionActivation::Tanh,
        };
        // Default16 final/deepstack mergers use tanh GELU; multi-vector and
        // exact-FP16 retrieval checkpoints use Transformers' exact-erf GELU.
        let merger_activation = match config.vision_profile(false) {
            crate::qwen3_vl_contract::VisionProfile::MultiVector
            | crate::qwen3_vl_contract::VisionProfile::ExactFp16 => VisionActivation::Exact,
            crate::qwen3_vl_contract::VisionProfile::Default16 => VisionActivation::Tanh,
        };
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
            plans: RefCell::new(VisionLinearPlans {
                patch: lt.prepare_bias_shape::<Bf16>(config.vision.patch_dim()?, hidden)?,
                qkv: lt.prepare_bias_shape::<Bf16>(hidden, product(hidden, 3)?)?,
                projection: lt.prepare_bias_shape::<Bf16>(hidden, hidden)?,
                mlp_fc1,
                mlp_fc2: lt.prepare_bias_shape::<Bf16>(config.vision.intermediate_size, hidden)?,
                merger_fc1,
                merger_fc2,
            }),
            kernels: VisionKernels::load(context)?,
            fa4,
            block_activation,
            merger_activation,
            deepstack_capture: deepstack_capture_order(
                &config.vision.deepstack_visual_indexes,
                config.vision.depth,
            )?,
            side: config.vision.position_side(),
            stream: stream.clone(),
        })
    }

    fn weight(&self, suffix: &str) -> Result<&DeviceBuffer<Bf16>, BackendError> {
        self.weights
            .get(&format!("model.visual.{suffix}"))
            .ok_or_else(|| start(format!("missing uploaded vision tensor {suffix}")))
    }

    #[allow(clippy::too_many_arguments)]
    fn norm(
        &self,
        stream: &Stream,
        input: u64,
        gamma: u64,
        beta: u64,
        output: u64,
        rows: usize,
        width: usize,
    ) -> Result<(), BackendError> {
        let mut input = input;
        let mut residual = 0u64;
        let mut bias = 0u64;
        let mut gamma = gamma;
        let mut beta = beta;
        let mut output = output;
        let mut rows = u32_size(rows)?;
        let mut width = u32_size(width)?;
        let mut eps = 1e-6f32;
        launch_threads(
            &self.kernels.layer_norm,
            stream,
            (rows, 1, 1),
            256,
            0,
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
        stream: &Stream,
        a: u64,
        b: u64,
        out: u64,
        elements: usize,
    ) -> Result<(), BackendError> {
        let mut a = a;
        let mut b = b;
        let mut out = out;
        let mut count = u32_size(elements)?;
        launch(
            &self.kernels.add,
            stream,
            blocks(elements)?,
            0,
            &mut [arg(&mut a), arg(&mut b), arg(&mut out), arg(&mut count)],
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
        // SAFETY: caller retains every buffer and stream until completion.
        unsafe { lt.enqueue(plan, input, weight, bias, output) }
    }

    fn activate(
        &self,
        stream: &Stream,
        buffer: &DeviceBuffer<Bf16>,
        elements: usize,
        activation: VisionActivation,
    ) -> Result<(), BackendError> {
        let mut pointer = buffer.cu_deviceptr();
        let mut count = u32_size(elements)?;
        let function = match activation {
            VisionActivation::Tanh => &self.kernels.gelu_tanh,
            VisionActivation::Exact => &self.kernels.gelu_exact,
            VisionActivation::Silu => &self.kernels.silu,
        };
        launch(
            function,
            stream,
            blocks(elements)?,
            0,
            &mut [arg(&mut pointer), arg(&mut count)],
        )
    }

    fn run(
        &self,
        lt: &LtProvider,
        work: &mut VisionWork,
        images: &[VisionImage],
        staging: &PinnedStagingPlan,
        merged_total: usize,
        deepstack_count: usize,
    ) -> Result<(), BackendError> {
        if staging.pixel_ranges.len() != images.len() || staging.frame_ranges.len() != images.len()
        {
            return Err(start("pinned staging plan does not cover every image"));
        }
        // Stage every image and frame list into disjoint pinned regions before
        // the first async H2D. No source region is rewritten later in the loop.
        for (index, image) in images.iter().enumerate() {
            let (pixel_offset, pixel_count) = staging.pixel_ranges[index];
            let (frame_offset, frame_count) = staging.frame_ranges[index];
            if pixel_count != image.pixels.len()
                || frame_count != image.grid_thw[0] + 1
                || pixel_offset
                    .checked_add(pixel_count)
                    .is_none_or(|end| end > work.host_pixels.as_slice().len())
                || frame_offset
                    .checked_add(frame_count)
                    .is_none_or(|end| end > work.host_frames.as_slice().len())
            {
                return Err(start("pinned staging capacity mismatch"));
            }
            let frame_len = product(image.grid_thw[1], image.grid_thw[2])?;
            let mut frames = Vec::with_capacity(frame_count);
            frames.push(0u32);
            for frame in 0..image.grid_thw[0] {
                let next = (frame + 1)
                    .checked_mul(frame_len)
                    .ok_or_else(|| start("frame offset overflow"))?;
                frames.push(u32_size(next)?);
            }
            work.host_pixels.as_mut_slice()[pixel_offset..pixel_offset + pixel_count]
                .copy_from_slice(&image.pixels);
            work.host_frames.as_mut_slice()[frame_offset..frame_offset + frame_count]
                .copy_from_slice(&frames);
        }
        let hidden = self.config.vision.hidden_size;
        let patch_dim = self.config.vision.patch_dim()?;
        let mut merged_offset = 0usize;
        for (image_index, image) in images.iter().enumerate() {
            let patch_rows = image.patch_rows()?;
            let merged = image.merged_rows()?;
            if image.pixels.len() != product(patch_rows, patch_dim)? {
                return Err(start(format!(
                    "{} pixel payload has {} elements, expected {}",
                    image.name,
                    image.pixels.len(),
                    product(patch_rows, patch_dim)?
                )));
            }
            if image.merge_size != self.config.vision.spatial_merge_size {
                return Err(start("image merge size does not match config"));
            }
            // Upload this image's immutable disjoint pinned region.
            let (pixel_offset, pixel_count) = staging.pixel_ranges[image_index];
            if pixel_count != product(patch_rows, patch_dim)? {
                return Err(start("pinned pixel range does not match image"));
            }
            unsafe {
                copy_prefix(
                    &work.pixels,
                    &self.stream,
                    &work.host_pixels.as_slice()[pixel_offset..pixel_offset + pixel_count],
                )?
            };
            let mut pixels = work.pixels.cu_deviceptr();
            let mut pixels_bf16 = work.pixels_bf16.cu_deviceptr();
            let mut count = u32_size(product(patch_rows, patch_dim)?)?;
            launch(
                &self.kernels.f32_to_bf16,
                &self.stream,
                blocks(product(patch_rows, patch_dim)?)?,
                0,
                &mut [arg(&mut pixels), arg(&mut pixels_bf16), arg(&mut count)],
            )?;
            let patch_weight = self.weight("patch_embed.proj.weight")?;
            let patch_bias = self.weight("patch_embed.proj.bias")?;
            let mut plans = self.plans.borrow_mut();
            self.linear(
                lt,
                &mut plans.patch,
                patch_rows,
                &work.pixels_bf16,
                patch_weight,
                patch_bias,
                &mut work.states,
            )?;
            let pos = self.weight("pos_embed.weight")?;
            let mut pos_ptr = pos.cu_deviceptr();
            let mut states = work.states.cu_deviceptr();
            let mut tokens = u32_size(patch_rows)?;
            let mut hidden_u32 = u32_size(hidden)?;
            let mut side = u32_size(self.side)?;
            let mut height = u32_size(image.grid_thw[1])?;
            let mut width = u32_size(image.grid_thw[2])?;
            let mut merge = u32_size(image.merge_size)?;
            launch(
                &self.kernels.pos_interp,
                &self.stream,
                blocks(product(patch_rows, hidden)?)?,
                0,
                &mut [
                    arg(&mut pos_ptr),
                    arg(&mut states),
                    arg(&mut tokens),
                    arg(&mut hidden_u32),
                    arg(&mut side),
                    arg(&mut height),
                    arg(&mut width),
                    arg(&mut merge),
                ],
            )?;
            let frame_len = product(image.grid_thw[1], image.grid_thw[2])?;
            let (frame_offset, frame_count) = staging.frame_ranges[image_index];
            // SAFETY: the disjoint pinned frame region is immutable for the
            // whole request and retained by the workspace until the fence.
            unsafe {
                copy_prefix(
                    &work.frame_offsets,
                    &self.stream,
                    &work.host_frames.as_slice()[frame_offset..frame_offset + frame_count],
                )?
            };
            let max_frame = u32_size(frame_len)?;
            let qkv_stride = u64::try_from(product(hidden, 3)?).map_err(infer)?;
            let hidden_bytes = u64::try_from(product(hidden, 2)?).map_err(infer)?;
            let mut head_dim = u32_size(self.config.vision.head_dim())?;
            let mut heads = u32_size(self.config.vision.num_heads)?;
            let mut qkv_ptr = work.qkv.cu_deviceptr();
            let v_ptr = qkv_ptr
                .checked_add(hidden_bytes * 2)
                .ok_or_else(|| infer("vision V address overflow"))?;
            let frame_ptr = work.frame_offsets.cu_deviceptr();
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
            for layer in 0..self.config.vision.depth {
                let prefix = format!("blocks.{layer}");
                let norm1_w = self.weight(&format!("{prefix}.norm1.weight"))?;
                let norm1_b = self.weight(&format!("{prefix}.norm1.bias"))?;
                self.norm(
                    &self.stream,
                    work.states.cu_deviceptr(),
                    norm1_w.cu_deviceptr(),
                    norm1_b.cu_deviceptr(),
                    work.norm.cu_deviceptr(),
                    patch_rows,
                    hidden,
                )?;
                let qkv_w = self.weight(&format!("{prefix}.attn.qkv.weight"))?;
                let qkv_b = self.weight(&format!("{prefix}.attn.qkv.bias"))?;
                self.linear(
                    lt,
                    &mut plans.qkv,
                    patch_rows,
                    &work.norm,
                    qkv_w,
                    qkv_b,
                    &mut work.qkv,
                )?;
                // 2-D rotary on packed Q/K.
                qkv_ptr = work.qkv.cu_deviceptr();
                let k_ptr = qkv_ptr
                    .checked_add(hidden_bytes)
                    .ok_or_else(|| infer("vision K address overflow"))?;
                let mut q = qkv_ptr;
                let mut k = k_ptr;
                let mut tokens = u32_size(patch_rows)?;
                let mut height = u32_size(image.grid_thw[1])?;
                let mut width = u32_size(image.grid_thw[2])?;
                let mut merge = u32_size(image.merge_size)?;
                let mut q_stride = u64::try_from(product(hidden, 3)?).map_err(infer)?;
                let mut k_stride = q_stride;
                launch(
                    &self.kernels.rope2d,
                    &self.stream,
                    blocks(product(
                        patch_rows,
                        product(
                            self.config.vision.num_heads,
                            self.config.vision.head_dim() / 2,
                        )?,
                    )?)?,
                    0,
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
                let attention = AttentionBatch {
                    q: qkv_ptr,
                    k: k_ptr,
                    v: v_ptr,
                    out: work.attention.cu_deviceptr(),
                    q_offsets: frame_ptr,
                    k_offsets: frame_ptr,
                    q_tokens: patch_rows,
                    k_tokens: patch_rows,
                    batch: image.grid_thw[0],
                    max_q: max_frame as usize,
                    strides,
                };
                // SAFETY: packed QKV/out/offsets and FA4 provider remain owned.
                unsafe { self.fa4.forward_batch_bound(&self.stream, &attention)? };
                let proj_w = self.weight(&format!("{prefix}.attn.proj.weight"))?;
                let proj_b = self.weight(&format!("{prefix}.attn.proj.bias"))?;
                self.linear(
                    lt,
                    &mut plans.projection,
                    patch_rows,
                    &work.attention,
                    proj_w,
                    proj_b,
                    &mut work.branch,
                )?;
                self.add(
                    &self.stream,
                    work.states.cu_deviceptr(),
                    work.branch.cu_deviceptr(),
                    work.states.cu_deviceptr(),
                    product(patch_rows, hidden)?,
                )?;
                let norm2_w = self.weight(&format!("{prefix}.norm2.weight"))?;
                let norm2_b = self.weight(&format!("{prefix}.norm2.bias"))?;
                self.norm(
                    &self.stream,
                    work.states.cu_deviceptr(),
                    norm2_w.cu_deviceptr(),
                    norm2_b.cu_deviceptr(),
                    work.norm.cu_deviceptr(),
                    patch_rows,
                    hidden,
                )?;
                let fc1_w = self.weight(&format!("{prefix}.mlp.linear_fc1.weight"))?;
                let fc1_b = self.weight(&format!("{prefix}.mlp.linear_fc1.bias"))?;
                self.linear(
                    lt,
                    &mut plans.mlp_fc1,
                    patch_rows,
                    &work.norm,
                    fc1_w,
                    fc1_b,
                    &mut work.mlp,
                )?;
                self.activate(
                    &self.stream,
                    &work.mlp,
                    product(patch_rows, self.config.vision.intermediate_size)?,
                    self.block_activation,
                )?;
                let fc2_w = self.weight(&format!("{prefix}.mlp.linear_fc2.weight"))?;
                let fc2_b = self.weight(&format!("{prefix}.mlp.linear_fc2.bias"))?;
                self.linear(
                    lt,
                    &mut plans.mlp_fc2,
                    patch_rows,
                    &work.mlp,
                    fc2_w,
                    fc2_b,
                    &mut work.branch,
                )?;
                self.add(
                    &self.stream,
                    work.states.cu_deviceptr(),
                    work.branch.cu_deviceptr(),
                    work.states.cu_deviceptr(),
                    product(patch_rows, hidden)?,
                )?;
                if let Some(capture_slot) = self
                    .deepstack_capture
                    .iter()
                    .position(|(capture_layer, _)| *capture_layer == layer)
                {
                    let name_index = self.deepstack_capture[capture_slot].1;
                    self.merger(
                        lt,
                        work,
                        images,
                        image,
                        merged,
                        name_index,
                        true,
                        capture_slot,
                        merged_offset,
                        &mut plans,
                    )?;
                }
            }
            self.merger(
                lt,
                work,
                images,
                image,
                merged,
                0,
                false,
                0,
                merged_offset,
                &mut plans,
            )?;
            merged_offset = merged_offset
                .checked_add(merged)
                .ok_or_else(|| infer("merged row offset overflow"))?;
        }
        if merged_offset != merged_total {
            return Err(start("merged row accounting mismatch"));
        }
        if deepstack_count != self.config.vision.deepstack_visual_indexes.len() {
            return Err(start("deepstack count mismatch"));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn merger(
        &self,
        lt: &LtProvider,
        work: &mut VisionWork,
        images: &[VisionImage],
        image: &VisionImage,
        merged: usize,
        deepstack_name_index: usize,
        deepstack: bool,
        output_index: usize,
        offset: usize,
        plans: &mut VisionLinearPlans,
    ) -> Result<(), BackendError> {
        let _ = images;
        let hidden = self.config.vision.hidden_size;
        let grouped = product(hidden, self.config.vision.merge_area()?)?;
        let out_hidden = self.config.vision.out_hidden_size;
        let patch_rows = image.patch_rows()?;
        let (prefix, plan_index) = if deepstack {
            (
                format!("deepstack_merger_list.{deepstack_name_index}"),
                deepstack_name_index + 1,
            )
        } else {
            ("merger".to_owned(), 0)
        };
        let norm_w = self.weight(&format!("{prefix}.norm.weight"))?;
        let norm_b = self.weight(&format!("{prefix}.norm.bias"))?;
        if deepstack {
            self.norm(
                &self.stream,
                work.states.cu_deviceptr(),
                norm_w.cu_deviceptr(),
                norm_b.cu_deviceptr(),
                work.grouped_norm.cu_deviceptr(),
                merged,
                grouped,
            )?;
        } else {
            self.norm(
                &self.stream,
                work.states.cu_deviceptr(),
                norm_w.cu_deviceptr(),
                norm_b.cu_deviceptr(),
                work.norm.cu_deviceptr(),
                patch_rows,
                hidden,
            )?;
        }
        let input = if deepstack {
            &work.grouped_norm
        } else {
            &work.norm
        };
        let fc1_w = self.weight(&format!("{prefix}.linear_fc1.weight"))?;
        let fc1_b = self.weight(&format!("{prefix}.linear_fc1.bias"))?;
        self.linear(
            lt,
            &mut plans.merger_fc1[plan_index],
            merged,
            input,
            fc1_w,
            fc1_b,
            &mut work.merger_tmp,
        )?;
        self.activate(
            &self.stream,
            &work.merger_tmp,
            product(merged, grouped)?,
            self.merger_activation,
        )?;
        let fc2_w = self.weight(&format!("{prefix}.linear_fc2.weight"))?;
        let fc2_b = self.weight(&format!("{prefix}.linear_fc2.bias"))?;
        if deepstack {
            self.linear(
                lt,
                &mut plans.merger_fc2[plan_index],
                merged,
                &work.merger_tmp,
                fc2_w,
                fc2_b,
                &mut work.image_deepstack[output_index],
            )?;
        } else {
            self.linear(
                lt,
                &mut plans.merger_fc2[plan_index],
                merged,
                &work.merger_tmp,
                fc2_w,
                fc2_b,
                &mut work.image_merged,
            )?;
        }
        let destination_ptr = if deepstack {
            work.image_deepstack[output_index].cu_deviceptr()
        } else {
            work.image_merged.cu_deviceptr()
        };
        let target_ptr = if deepstack {
            work.deepstack[output_index].cu_deviceptr()
        } else {
            work.merged.cu_deviceptr()
        };
        let bytes = u64::try_from(product(product(merged, out_hidden)?, 2)?).map_err(infer)?;
        let offset_bytes =
            u64::try_from(product(product(offset, out_hidden)?, 2)?).map_err(infer)?;
        // SAFETY: disjoint merged output regions in one retained allocation.
        unsafe {
            cuda_core::memory::memcpy_dtod_async(
                target_ptr
                    .checked_add(offset_bytes)
                    .ok_or_else(|| infer("merged destination overflow"))?,
                destination_ptr,
                usize::try_from(bytes).map_err(infer)?,
                self.stream.cu_stream(),
            )
        }
        .map_err(infer)?;
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct VisionImage {
    pub name: String,
    pub pixels: Vec<f32>,
    pub grid_thw: [usize; 3],
    pub merge_size: usize,
    pub token_start: usize,
}

impl VisionImage {
    pub fn patch_rows(&self) -> Result<usize, BackendError> {
        if self.grid_thw.contains(&0) || self.merge_size == 0 {
            return Err(start(format!("{} has an empty grid or merge", self.name)));
        }
        if !self.grid_thw[1].is_multiple_of(self.merge_size)
            || !self.grid_thw[2].is_multiple_of(self.merge_size)
        {
            return Err(start(format!(
                "{} grid is not divisible by merge",
                self.name
            )));
        }
        self.grid_thw
            .iter()
            .try_fold(1usize, |value, dim| product(value, *dim))
    }

    pub fn merged_rows(&self) -> Result<usize, BackendError> {
        let area = product(self.merge_size, self.merge_size)?;
        self.patch_rows()?
            .checked_div(area)
            .filter(|rows| *rows > 0)
            .ok_or_else(|| start("zero merged rows"))
    }
}

#[derive(Clone, Debug)]
pub struct Qwen3VlForward {
    pub ids: Vec<u32>,
    pub cumulative: Vec<u32>,
    pub images: Vec<VisionImage>,
    pub media_positions: Option<[Vec<u32>; 3]>,
    pub pool: Pool,
    /// Return row-major `[tokens, hidden]` states instead of pooled vectors.
    /// Required for raw and multi-vector wrappers.
    pub raw_tokens: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Qwen3VlOutput {
    pub pooled: Option<Vec<f32>>,
    pub raw: Option<Vec<f32>>,
    pub sequences: usize,
    pub hidden: usize,
}

pub struct Qwen3VlExecutor {
    config: Qwen3VlConfig,
    vision: Qwen3VlVision,
    decoder: DecoderModel,
    decoder_config: DecoderConfig,
    lt: LtProvider,
    plain_blas: Option<Blas>,
    stream: Arc<Stream>,
    workspaces: Qwen3VlWorkspaces,
    host_meta: Vec<u32>,
}

impl Qwen3VlExecutor {
    pub fn new(
        checkpoint_path: &Path,
        context: &Arc<Context>,
        stream: &Arc<Stream>,
    ) -> Result<Self, BackendError> {
        let config = Qwen3VlConfig::from_path(checkpoint_path)?;
        if config.vision.dtype != Dtype::Bf16 || config.text.dtype != Dtype::Bf16 {
            return Err(start(
                "Qwen3-VL device executor requires the reviewed BF16 checkpoint",
            ));
        }
        let checkpoint = Checkpoint::open(checkpoint_path)?;
        let mut text_config = DecoderConfig::from_text_config(
            config
                .raw
                .get("text_config")
                .ok_or_else(|| start("missing text_config"))?,
        )?;
        // Native wrapper options that are not stored in config.json.
        // Nemotron ColEmbed is bidirectional and extracts pre-final-norm token
        // states; ColQwen3 stays causal.
        let nemotron = config.readout.kind == crate::qwen3_vl_config::ReadoutKind::NemotronColEmbed;
        text_config = text_config.with_readout_modifiers(!nemotron, nemotron);
        // ColQwen3 stores the VLM under `vlm.model.*`; Native strips the
        // `vlm.` namespace before constructing the same Qwen3-VL adapter.
        let text_prefix = if checkpoint.contains("vlm.model.language_model.embed_tokens.weight") {
            "vlm.model.language_model."
        } else {
            "model.language_model."
        };
        let text_weights =
            DecoderWeights::load_with_checkpoint(text_config.clone(), &checkpoint, text_prefix)?;
        let lt = LtProvider::new_native_default(context.clone(), stream.clone())?;
        let decoder = DecoderModel::new(text_weights, context, stream, &lt)?;
        let vision = Qwen3VlVision::new(&config, &checkpoint, context, stream, &lt)?;
        let plain_blas = if decoder.needs_plain() {
            Some(Blas::new(context.clone(), stream.clone())?)
        } else {
            None
        };
        if plain_blas.is_none() && decoder.needs_plain() {
            return Err(start("decoder requires plain cuBLAS"));
        }
        // Bind the provider context once for standalone plan preparation.
        lt.bind_context()?;
        Ok(Self {
            config,
            vision,
            decoder,
            decoder_config: text_config,
            lt,
            plain_blas,
            stream: stream.clone(),
            workspaces: Qwen3VlWorkspaces::default(),
            host_meta: Vec::new(),
        })
    }

    pub fn forward(&mut self, input: &Qwen3VlForward) -> Result<Vec<f32>, BackendError> {
        let output = self.forward_outputs(input)?;
        output
            .pooled
            .ok_or_else(|| start("forward() produced token rows, not a pooled vector"))
    }

    /// Concrete output of one packed Qwen3-VL request. `raw` is row-major
    /// `[tokens, hidden]` for raw/multi-vector requests; `pooled` is row-major
    /// `[sequences, hidden]` for embedding requests.
    pub fn forward_outputs(
        &mut self,
        input: &Qwen3VlForward,
    ) -> Result<Qwen3VlOutput, BackendError> {
        self.stream.context().bind_to_thread().map_err(infer)?;
        let mut workspaces = std::mem::take(&mut self.workspaces);
        let result = self.forward_outputs_with(input, &mut workspaces, None);
        self.workspaces = workspaces;
        match result? {
            Qwen3VlForwardResult::Ready(output) => Ok(output),
            Qwen3VlForwardResult::Launched(_) => Err(start(
                "internal VL launch returned without a completion lease",
            )),
        }
    }

    pub(crate) fn forward_outputs_with(
        &mut self,
        input: &Qwen3VlForward,
        workspaces: &mut Qwen3VlWorkspaces,
        mut lease: Option<&mut HostIoLease>,
    ) -> Result<Qwen3VlForwardResult, BackendError> {
        let tokens = input.ids.len();
        if tokens == 0 {
            return Err(start("empty Qwen3-VL input"));
        }
        let sequences = input
            .cumulative
            .len()
            .checked_sub(1)
            .ok_or_else(|| start("missing cumulative sequence boundaries"))?;
        if sequences == 0
            || input.cumulative.first().copied() != Some(0)
            || input.cumulative.last().copied() != Some(u32_size(tokens)?)
        {
            return Err(start("invalid cumulative sequence boundaries"));
        }
        let mut max_length = 0usize;
        for pair in input.cumulative.windows(2) {
            if pair[0] > pair[1] {
                return Err(start("non-monotonic cumulative boundaries"));
            }
            max_length = max_length.max((pair[1] - pair[0]) as usize);
        }
        let mut spans = Vec::with_capacity(input.images.len());
        for image in &input.images {
            let merged = image.merged_rows()?;
            let end = image
                .token_start
                .checked_add(merged)
                .ok_or_else(|| start("image token span overflow"))?;
            if end > tokens {
                return Err(start(format!(
                    "{} image span exceeds the packed token count",
                    image.name
                )));
            }
            if !input
                .cumulative
                .windows(2)
                .any(|pair| pair[0] as usize <= image.token_start && end <= pair[1] as usize)
            {
                return Err(start(format!(
                    "{} image span crosses a packed sequence boundary",
                    image.name
                )));
            }
            spans.push((image.token_start, end));
        }
        spans.sort_unstable();
        if spans.windows(2).any(|pair| pair[0].1 > pair[1].0) {
            return Err(start("overlapping image token spans"));
        }
        if let Some(positions) = input.media_positions.as_ref() {
            for axis in positions {
                if axis.len() != tokens {
                    return Err(start("media position axis length mismatch"));
                }
            }
        } else if !input.images.is_empty() && sequences > 1 {
            return Err(start(
                "multi-sequence images require explicit media_positions",
            ));
        }
        let merged_total = input.images.iter().try_fold(0usize, |total, image| {
            total
                .checked_add(image.merged_rows()?)
                .ok_or_else(|| start("merged row total overflow"))
        })?;
        let max_patches = input
            .images
            .iter()
            .map(|image| image.patch_rows())
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .max()
            .unwrap_or(0);
        let max_frames = input
            .images
            .iter()
            .map(|image| image.grid_thw[0])
            .max()
            .unwrap_or(0);
        let deepstack_count = self.config.vision.deepstack_visual_indexes.len();
        let staging = plan_pinned_staging(&input.images)?;
        let vision_key = VisionWorkKey {
            patches: max_patches,
            merged: merged_total.max(1),
            frames: max_frames,
            tokens,
            staged_pixels: staging.total_pixels,
            staged_frames: staging.total_frames,
        };
        let need_vision = input
            .images
            .iter()
            .any(|image| image.patch_rows().unwrap_or(0) > 0);
        let need_media = need_vision || input.media_positions.is_some();
        if need_vision {
            let fits = workspaces
                .vision_work
                .as_ref()
                .is_some_and(|work| work.fits(vision_key, deepstack_count));
            if !fits {
                if let Some(old) = workspaces.vision_work.take() {
                    drop(old);
                }
                workspaces.vision_work = Some(VisionWork::new(
                    self.stream.clone(),
                    &self.config,
                    vision_key,
                    deepstack_count,
                )?);
            }
        }
        if need_vision {
            let work = workspaces
                .vision_work
                .as_mut()
                .ok_or_else(|| start("vision workspace missing"))?;
            self.vision.run(
                &self.lt,
                work,
                &input.images,
                &staging,
                merged_total,
                deepstack_count,
            )?;
            // Flatten visual rows into the owned pinned staging buffer.
            let mut rows = Vec::with_capacity(merged_total);
            for image in &input.images {
                for index in 0..image.merged_rows()? {
                    rows.push(u32_size(image.token_start + index)?);
                }
            }
            if rows.len() > work.host_rows.as_slice().len() {
                return Err(start("visual rows exceed pinned vision staging"));
            }
            work.host_rows.as_mut_slice()[..rows.len()].copy_from_slice(&rows);
            // SAFETY: pinned rows are retained on the work owner until the stream drains.
            unsafe {
                copy_prefix(
                    &work.visual_rows,
                    &self.stream,
                    &work.host_rows.as_slice()[..rows.len()],
                )?
            };
        }
        if need_media {
            let positions = match input.media_positions.clone() {
                Some(positions) => positions,
                None => {
                    let grids: Vec<ImageGrid> = input
                        .images
                        .iter()
                        .map(|image| {
                            ImageGrid::new(
                                image.grid_thw[0],
                                image.grid_thw[1],
                                image.grid_thw[2],
                                image.merge_size,
                                image.token_start,
                                image.merged_rows()?,
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    image_position_ids(tokens, &grids)?
                }
            };
            let axes = mrope_axis_map(self.config.text.text_dim(), self.config.text.mrope_section)?;
            let frequency =
                mrope_frequencies(self.config.text.text_dim(), self.config.text.rope_theta)?;
            let phases = mrope_phases(&positions, &axes, &frequency)?;
            let mut identity = Vec::with_capacity(tokens);
            for token in 0..tokens {
                identity.push(u32_size(token)?);
            }
            if need_vision {
                let work = workspaces
                    .vision_work
                    .as_mut()
                    .ok_or_else(|| start("vision workspace missing"))?;
                if phases.cos.len() > work.host_cos.as_slice().len()
                    || phases.sin.len() > work.host_sin.as_slice().len()
                    || identity.len() > work.host_positions.as_slice().len()
                {
                    return Err(start("mRoPE tables exceed pinned vision staging"));
                }
                work.host_cos.as_mut_slice()[..phases.cos.len()].copy_from_slice(&phases.cos);
                work.host_sin.as_mut_slice()[..phases.sin.len()].copy_from_slice(&phases.sin);
                work.host_positions.as_mut_slice()[..identity.len()].copy_from_slice(&identity);
                // SAFETY: pinned phase tables are retained on the work owner.
                unsafe {
                    copy_prefix(
                        &work.rope_cos,
                        &self.stream,
                        &work.host_cos.as_slice()[..phases.cos.len()],
                    )?;
                    copy_prefix(
                        &work.rope_sin,
                        &self.stream,
                        &work.host_sin.as_slice()[..phases.sin.len()],
                    )?;
                    copy_prefix(
                        &work.rope_positions,
                        &self.stream,
                        &work.host_positions.as_slice()[..identity.len()],
                    )?;
                }
            } else {
                let fits = workspaces
                    .text_media
                    .as_ref()
                    .is_some_and(|work| work.fits(tokens));
                if !fits {
                    workspaces.text_media = Some(PositionWork::new(
                        self.stream.clone(),
                        tokens,
                        self.config.text.text_dim() / 2,
                    )?);
                }
                let work = workspaces
                    .text_media
                    .as_mut()
                    .ok_or_else(|| start("text mRoPE workspace missing"))?;
                if phases.cos.len() > work.host_cos.as_slice().len()
                    || phases.sin.len() > work.host_sin.as_slice().len()
                    || identity.len() > work.host_positions.as_slice().len()
                {
                    return Err(start("mRoPE tables exceed pinned text staging"));
                }
                work.host_cos.as_mut_slice()[..phases.cos.len()].copy_from_slice(&phases.cos);
                work.host_sin.as_mut_slice()[..phases.sin.len()].copy_from_slice(&phases.sin);
                work.host_positions.as_mut_slice()[..identity.len()].copy_from_slice(&identity);
                // SAFETY: pinned phase tables are retained on the work owner.
                unsafe {
                    copy_prefix(
                        &work.cos,
                        &self.stream,
                        &work.host_cos.as_slice()[..phases.cos.len()],
                    )?;
                    copy_prefix(
                        &work.sin,
                        &self.stream,
                        &work.host_sin.as_slice()[..phases.sin.len()],
                    )?;
                    copy_prefix(
                        &work.positions,
                        &self.stream,
                        &work.host_positions.as_slice()[..identity.len()],
                    )?;
                }
            }
        }
        let decoder_key: WorkKey = (tokens, sequences + 1, tokens);
        let fits = workspaces
            .decoder_work
            .as_ref()
            .is_some_and(|work| work.fits(decoder_key));
        if !fits {
            if let Some(old) = workspaces.decoder_work.take() {
                drop(old);
            }
            workspaces.decoder_work = Some(DecoderWork::new(
                self.stream.clone(),
                &self.decoder_config,
                decoder_key,
            )?);
        }
        let decoder_work = workspaces
            .decoder_work
            .as_mut()
            .ok_or_else(|| start("decoder workspace missing"))?;
        let mut positions = Vec::with_capacity(tokens);
        for pair in input.cumulative.windows(2) {
            for local in 0..(pair[1] - pair[0]) {
                positions.push(local);
            }
        }
        let types = vec![0u32; tokens];
        let selected: Vec<u32> = (0..tokens).map(u32_size).collect::<Result<Vec<_>, _>>()?;
        if let Some(lease) = lease.as_mut() {
            // Request-specific pinned host input. The lease stays alive until
            // the deferred completion fence; the device copy is ordered on the
            // same stream as every subsequent launch.
            let elements = {
                let host = lease.slot_mut()?.input.as_mut_slice();
                pack_decoder_metadata(
                    host,
                    &input.ids,
                    &positions,
                    &types,
                    &input.cumulative,
                    &selected,
                )?
            };
            let host = lease.slot()?.input.as_slice();
            decoder_work.upload(&host[..elements], elements)?;
        } else {
            decoder_work.stage_raw(
                &mut self.host_meta,
                &input.ids,
                &positions,
                &types,
                &input.cumulative,
                &selected,
            )?;
        }
        let batch = Batch {
            multimodal: Vec::new(),
            input_ids: input.ids.clone(),
            token_type_ids: types,
            position_ids: positions,
            cumulative_seq_lengths: input.cumulative.clone(),
            max_length: u32_size(max_length)?,
            pooled_indices: Vec::new(),
            raw_indices: (0..sequences)
                .map(u32_size)
                .collect::<Result<Vec<_>, _>>()?,
            compact_input_ids: None,
            compact_position_ids: None,
            scatter_unfold: None,
            fold_gather: None,
            tokens: Vec::new(),
            offsets: Vec::new(),
        };
        if need_vision {
            let work = workspaces
                .vision_work
                .as_ref()
                .ok_or_else(|| start("vision workspace missing"))?;
            let media = DecoderMultimodal {
                rows: &work.visual_rows,
                row_count: merged_total,
                visual: &work.merged,
                deepstack: work.deepstack.as_slice(),
                cosine: &work.rope_cos,
                sine: &work.rope_sin,
                rope_positions: &work.rope_positions,
            };
            self.decoder.enqueue_with(
                &self.lt,
                self.plain_blas.as_ref(),
                &self.stream,
                decoder_work,
                &batch,
                &input.pool,
                max_length,
                tokens,
                Some(&media),
            )?;
        } else if let Some(work) = workspaces.text_media.as_ref() {
            let media = DecoderMultimodal {
                rows: &work.rows,
                row_count: 0,
                visual: &work.visual,
                deepstack: &[],
                cosine: &work.cos,
                sine: &work.sin,
                rope_positions: &work.positions,
            };
            self.decoder.enqueue_with(
                &self.lt,
                self.plain_blas.as_ref(),
                &self.stream,
                decoder_work,
                &batch,
                &input.pool,
                max_length,
                tokens,
                Some(&media),
            )?;
        } else {
            self.decoder.enqueue_with(
                &self.lt,
                self.plain_blas.as_ref(),
                &self.stream,
                decoder_work,
                &batch,
                &input.pool,
                max_length,
                tokens,
                None,
            )?;
        }
        let hidden = self.decoder_config.hidden_size;
        let elements = product(tokens, hidden)?;
        if let Some(lease) = lease.as_mut() {
            decoder_work.download_output_into(&self.stream, &lease.slot()?.output, elements)?;
            return Ok(Qwen3VlForwardResult::Launched(Qwen3VlPending {
                elements,
                tokens,
                sequences,
                hidden,
            }));
        }
        let raw = decoder_work.download_output(&self.stream, elements)?;
        if input.raw_tokens {
            return Ok(Qwen3VlForwardResult::Ready(Qwen3VlOutput {
                pooled: None,
                raw: Some(raw),
                sequences,
                hidden,
            }));
        }
        let pooled = match input.pool {
            Pool::Cls => {
                let mut pooled = Vec::with_capacity(sequences * hidden);
                for pair in input.cumulative.windows(2) {
                    let start = pair[0] as usize;
                    pooled.extend_from_slice(&raw[start * hidden..(start + 1) * hidden]);
                }
                pooled
            }
            Pool::LastToken => {
                let mut pooled = Vec::with_capacity(sequences * hidden);
                for pair in input.cumulative.windows(2) {
                    let last = pair[1] as usize - 1;
                    pooled.extend_from_slice(&raw[last * hidden..(last + 1) * hidden]);
                }
                pooled
            }
            Pool::Mean => {
                let mut pooled = Vec::with_capacity(sequences * hidden);
                for pair in input.cumulative.windows(2) {
                    let start = pair[0] as usize;
                    let end = pair[1] as usize;
                    let count = (end - start) as f32;
                    for column in 0..hidden {
                        let mut sum = 0f32;
                        for row in start..end {
                            sum += raw[row * hidden + column];
                        }
                        pooled.push(sum / count);
                    }
                }
                pooled
            }
            Pool::Splade => return Err(start("Oxide Qwen3-VL SPLADE pooling is unsupported")),
        };
        Ok(Qwen3VlForwardResult::Ready(Qwen3VlOutput {
            pooled: Some(pooled),
            raw: None,
            sequences,
            hidden,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(
        name: &str,
        frames: usize,
        height: usize,
        width: usize,
        patch_dim: usize,
    ) -> VisionImage {
        let rows = frames * height * width;
        VisionImage {
            name: name.to_owned(),
            pixels: vec![0.0; rows * patch_dim],
            grid_thw: [frames, height, width],
            merge_size: 2,
            token_start: 0,
        }
    }

    #[test]
    fn pinned_staging_plan_is_disjoint_and_covers_every_image() {
        let images = vec![
            image("a", 1, 2, 4, 4),
            image("b", 1, 4, 2, 4),
            image("c", 3, 2, 2, 4),
        ];
        let plan = plan_pinned_staging(&images).unwrap();
        assert_eq!(plan.pixel_ranges.len(), 3);
        assert_eq!(plan.frame_ranges.len(), 3);
        let mut pixel_end = 0;
        let mut frame_end = 0;
        for (index, image) in images.iter().enumerate() {
            let (pixel_start, pixels) = plan.pixel_ranges[index];
            let (frame_start, frames) = plan.frame_ranges[index];
            assert_eq!(pixel_start, pixel_end);
            assert_eq!(frame_start, frame_end);
            assert_eq!(pixels, image.pixels.len());
            assert_eq!(frames, image.grid_thw[0] + 1);
            pixel_end = pixel_end.checked_add(pixels).unwrap();
            frame_end = frame_end.checked_add(frames).unwrap();
        }
        assert_eq!(pixel_end, plan.total_pixels);
        assert_eq!(frame_end, plan.total_frames);
        assert!(plan.total_pixels >= images.iter().map(|i| i.pixels.len()).sum::<usize>());
        assert!(plan.total_frames >= images.len());
    }

    #[test]
    fn pinned_arena_capacities_use_request_totals_not_max_image() {
        let images = vec![
            image("a", 1, 2, 4, 4),
            image("b", 1, 4, 2, 4),
            image("c", 3, 2, 2, 4),
        ];
        let plan = plan_pinned_staging(&images).unwrap();
        let key = VisionWorkKey {
            patches: images
                .iter()
                .map(|image| image.patch_rows().unwrap())
                .max()
                .unwrap(),
            merged: images
                .iter()
                .map(|image| image.merged_rows().unwrap())
                .sum(),
            frames: images.iter().map(|image| image.grid_thw[0]).max().unwrap(),
            tokens: 8,
            staged_pixels: plan.total_pixels,
            staged_frames: plan.total_frames,
        };
        let (pixels, frames) = pinned_staging_capacities(&key);
        assert_eq!(pixels, plan.total_pixels);
        assert_eq!(frames, plan.total_frames);
        assert!(pixels > key.patches * 4);
        assert!(frames > key.frames + 1);
    }

    #[test]
    fn pinned_staging_plan_rejects_bad_payloads_and_overflow() {
        let mut bad = image("bad", 1, 2, 2, 4);
        bad.pixels.truncate(1);
        assert!(plan_pinned_staging(&[bad]).is_err());

        let mut huge = image("huge", 1, 2, 2, 4);
        huge.grid_thw = [usize::MAX, 2, 2];
        assert!(plan_pinned_staging(&[huge]).is_err());
    }
}
