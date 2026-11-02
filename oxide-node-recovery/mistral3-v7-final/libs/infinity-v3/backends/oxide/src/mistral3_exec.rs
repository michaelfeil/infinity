// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Owned Pixtral tower preparation and enqueue.
//!
//! [`Mistral3Prepared`] owns every uploaded BF16 matrix, the FA4 provider, the
//! plain cuBLAS handle and one request arena sized at construction. Enqueue
//! performs no allocation, device query, synchronization or CPU fallback. The
//! shared text decoder consumes the emitted projector rows through
//! `crate::decoder::DecoderModel::enqueue_visual`.
#![allow(dead_code)] // Bound by the Root GPU route once its AOT bundle exists.

use crate::mistral3::{
    Mistral3MediaPlan, Mistral3Pending, Mistral3Plan, Mistral3VisionProfile, ProjectorActivation,
};
use crate::{
    arg, blocks, cublas::Blas, decoder::DecoderVisual, fa4::AttentionBatch, infer, launch,
    math::LinearPlan, product, start, tensor::Bf16, u32_size, upload::WeightUploader,
};
use cuda_core::{Context, DeviceBuffer, Function, PinnedHostBuffer, Stream};
use std::{collections::BTreeMap, path::Path, sync::Arc};

struct Kernels {
    rms: Function,
    rope2d: Function,
    add: Function,
    silu: Function,
    gelu: Function,
    gelu_mul: Option<Function>,
    merge: Function,
    pixels: Function,
}

impl Kernels {
    fn load(context: &Arc<Context>, require_gelu_mul: bool) -> Result<Self, crate::BackendError> {
        let path = std::env::var_os(infinity_v3_config::infinity_env!("OXIDE_KERNEL_MODULE"))
            .ok_or_else(|| start("Mistral3 requires INFINITY_OXIDE_KERNEL_MODULE"))?;
        let path = Path::new(&path);
        let path = if path.is_dir() {
            let (major, minor) = context.compute_capability().map_err(start)?;
            path.join(format!("sm_{major}{minor}"))
                .join("kernels.cubin")
        } else {
            path.to_path_buf()
        };
        let bytes = std::fs::read(&path).map_err(start)?;
        let module = context.load_module_from_image(&bytes).map_err(start)?;
        let load = |name: &str| module.load_function(name).map_err(start);
        Ok(Self {
            rms: load("infinity_rms_norm_rounded")?,
            rope2d: load("infinity_mistral3_rope2d_bf16")?,
            add: load("infinity_mistral3_add_bf16")?,
            silu: load("infinity_mistral3_silu_mul_bf16")?,
            gelu: load("infinity_mistral3_gelu_bf16")?,
            gelu_mul: if require_gelu_mul {
                Some(load("infinity_mistral3_gelu_mul_bf16").map_err(|error| {
                    start(format!(
                        "Mistral3 selected a GELU vision activation but the AOT bundle lacks infinity_mistral3_gelu_mul_bf16: {error}"
                    ))
                })?)
            } else {
                load("infinity_mistral3_gelu_mul_bf16").ok()
            },
            merge: load("infinity_mistral3_merge_gather_bf16")?,
            pixels: load("infinity_mistral3_pixels_to_bf16")?,
        })
    }
}

/// One owned cache of vision and projector matrices. Names stay explicit so a
/// missing tensor fails at load rather than during a request.
pub(crate) struct VisionWeights {
    pub(crate) tensors: BTreeMap<String, DeviceBuffer<Bf16>>,
}

impl VisionWeights {
    fn get(&self, name: &str) -> Result<&DeviceBuffer<Bf16>, crate::BackendError> {
        self.tensors
            .get(name)
            .ok_or_else(|| start(format!("Mistral3 prepared weights lack {name}")))
    }
}

/// One preallocated request arena. `max_patches` bounds the tower token count
/// and `max_tokens` the post-merge projector rows.
struct Workspace {
    pixels_f32: DeviceBuffer<f32>,
    pixels_bf16: DeviceBuffer<Bf16>,
    hidden: DeviceBuffer<Bf16>,
    normalized: DeviceBuffer<Bf16>,
    q: DeviceBuffer<Bf16>,
    k: DeviceBuffer<Bf16>,
    v: DeviceBuffer<Bf16>,
    attended: DeviceBuffer<Bf16>,
    projection: DeviceBuffer<Bf16>,
    gate: DeviceBuffer<Bf16>,
    up: DeviceBuffer<Bf16>,
    selected: DeviceBuffer<Bf16>,
    merged: DeviceBuffer<Bf16>,
    merged_hidden: DeviceBuffer<Bf16>,
    projector_hidden: DeviceBuffer<Bf16>,
    features: DeviceBuffer<Bf16>,
    grids: DeviceBuffer<u32>,
    offsets: DeviceBuffer<u32>,
    merge_rows: DeviceBuffer<u32>,
    indices: DeviceBuffer<u32>,
    max_patches: usize,
    max_tokens: usize,
    max_images: usize,
}

impl Workspace {
    #[allow(clippy::too_many_arguments)]
    fn new(
        stream: &Arc<Stream>,
        vision_hidden: usize,
        intermediate: usize,
        patch_dim: usize,
        text_hidden: usize,
        merge: usize,
        max_patches: usize,
        max_tokens: usize,
        max_images: usize,
    ) -> Result<Self, crate::BackendError> {
        let merged_width = vision_hidden
            .checked_mul(merge)
            .and_then(|value| value.checked_mul(merge))
            .ok_or_else(|| start("Mistral3 merge width overflow"))?;
        let alloc = |elements: usize| {
            if elements == 0 {
                return Err(start("Mistral3 workspace requires nonzero extents"));
            }
            // SAFETY: every buffer is fully written by the tower before it is read.
            unsafe { DeviceBuffer::<Bf16>::uninitialized_async(stream, elements) }.map_err(infer)
        };
        let pixels_f32 = unsafe {
            DeviceBuffer::<f32>::uninitialized_async(stream, product(max_patches, patch_dim)?)
        }
        .map_err(infer)?;
        Ok(Self {
            pixels_f32,
            pixels_bf16: alloc(product(max_patches, patch_dim)?)?,
            hidden: alloc(product(max_patches, vision_hidden)?)?,
            normalized: alloc(product(max_patches, vision_hidden)?)?,
            q: alloc(product(max_patches, vision_hidden)?)?,
            k: alloc(product(max_patches, vision_hidden)?)?,
            v: alloc(product(max_patches, vision_hidden)?)?,
            attended: alloc(product(max_patches, vision_hidden)?)?,
            projection: alloc(product(max_patches, intermediate.max(vision_hidden))?)?,
            gate: alloc(product(max_patches, intermediate)?)?,
            up: alloc(product(max_patches, intermediate)?)?,
            selected: alloc(product(max_patches, vision_hidden)?)?,
            merged: alloc(product(max_tokens, merged_width)?)?,
            merged_hidden: alloc(product(max_tokens, vision_hidden)?)?,
            projector_hidden: alloc(product(max_tokens, text_hidden)?)?,
            features: alloc(product(max_tokens, text_hidden)?)?,
            grids: unsafe { DeviceBuffer::<u32>::uninitialized_async(stream, max_images * 3) }
                .map_err(infer)?,
            offsets: unsafe { DeviceBuffer::<u32>::uninitialized_async(stream, max_images + 1) }
                .map_err(infer)?,
            merge_rows: unsafe {
                DeviceBuffer::<u32>::uninitialized_async(stream, max_tokens * merge * merge)
            }
            .map_err(infer)?,
            indices: unsafe { DeviceBuffer::<u32>::uninitialized_async(stream, max_tokens) }
                .map_err(infer)?,
            max_patches,
            max_tokens,
            max_images,
        })
    }

    fn ptr(&self, name: &str) -> Result<u64, crate::BackendError> {
        Ok(self.buffer(name)?.cu_deviceptr())
    }

    fn buffer(&self, name: &str) -> Result<&DeviceBuffer<Bf16>, crate::BackendError> {
        match name {
            "hidden" => Ok(&self.hidden),
            "normalized" => Ok(&self.normalized),
            "q" => Ok(&self.q),
            "k" => Ok(&self.k),
            "v" => Ok(&self.v),
            "attended" => Ok(&self.attended),
            "projection" => Ok(&self.projection),
            "gate" => Ok(&self.gate),
            "up" => Ok(&self.up),
            "selected" => Ok(&self.selected),
            "merged" => Ok(&self.merged),
            "merged_hidden" => Ok(&self.merged_hidden),
            "projector_hidden" => Ok(&self.projector_hidden),
            "features" => Ok(&self.features),
            "pixels" => Ok(&self.pixels_bf16),
            other => Err(infer(format!("unknown Mistral3 arena {other}"))),
        }
    }

    fn buffer_mut(&mut self, name: &str) -> Result<&mut DeviceBuffer<Bf16>, crate::BackendError> {
        match name {
            "hidden" => Ok(&mut self.hidden),
            "normalized" => Ok(&mut self.normalized),
            "q" => Ok(&mut self.q),
            "k" => Ok(&mut self.k),
            "v" => Ok(&mut self.v),
            "attended" => Ok(&mut self.attended),
            "projection" => Ok(&mut self.projection),
            "gate" => Ok(&mut self.gate),
            "up" => Ok(&mut self.up),
            "selected" => Ok(&mut self.selected),
            "merged" => Ok(&mut self.merged),
            "merged_hidden" => Ok(&mut self.merged_hidden),
            "projector_hidden" => Ok(&mut self.projector_hidden),
            "features" => Ok(&mut self.features),
            "pixels" => Ok(&mut self.pixels_bf16),
            other => Err(infer(format!("unknown Mistral3 arena {other}"))),
        }
    }
}

/// Per-request pinned host staging for the Pixtral pixel payload and the
/// metadata tail. The owner is moved into the pending request and must outlive
/// the recorded completion event; no raw pageable H2D source is used.
pub(crate) struct Mistral3MediaHost {
    metadata: PinnedHostBuffer<u32>,
    pixels: Option<PinnedHostBuffer<f32>>,
}

fn media_metadata_elements(
    images: usize,
    total_image_tokens: usize,
    merge_area: usize,
) -> Result<usize, crate::BackendError> {
    images
        .checked_mul(4)
        .and_then(|value| value.checked_add(1))
        .and_then(|value| value.checked_add(total_image_tokens.checked_mul(merge_area)?))
        .and_then(|value| value.checked_add(total_image_tokens))
        .ok_or_else(|| infer("Mistral3 metadata staging overflow"))
}

impl Mistral3MediaHost {
    fn new(
        stream: &Arc<Stream>,
        plan: &Mistral3MediaPlan,
        patch_dim: usize,
        merge_area: usize,
    ) -> Result<Self, crate::BackendError> {
        let images = plan.images.len();
        let metadata_elements =
            media_metadata_elements(images, plan.total_image_tokens, merge_area)?;
        let metadata =
            PinnedHostBuffer::<u32>::zeroed(stream.context(), metadata_elements).map_err(infer)?;
        let pixels = if images == 0 {
            None
        } else {
            let elements = plan
                .total_patches
                .checked_mul(patch_dim)
                .ok_or_else(|| infer("Mistral3 pixel staging overflow"))?;
            Some(PinnedHostBuffer::<f32>::zeroed(stream.context(), elements).map_err(infer)?)
        };
        Ok(Self { metadata, pixels })
    }
}

fn compute_plan(
    plans: &BTreeMap<(usize, usize), LinearPlan<Bf16>>,
    input: usize,
    output: usize,
    rows: usize,
) -> Result<LinearPlan<Bf16>, crate::BackendError> {
    let mut plan = plans
        .get(&(input, output))
        .cloned()
        .ok_or_else(|| infer(format!("Mistral3 lacks a {input}->{output} linear plan")))?;
    plan.set_rows(rows).map_err(infer)?;
    Ok(plan)
}

/// Prepared Pixtral tower and the fusion handle consumed by the shared text
/// decoder. Construction requires a real CUDA context; no CPU fallback exists.
pub(crate) struct Mistral3Prepared {
    pub(crate) config: crate::mistral3::Mistral3Config,
    pub(crate) weights: VisionWeights,
    pub(crate) profile: Mistral3VisionProfile,
    stream: Arc<Stream>,
    attention: crate::fa4::Fa4,
    blas: Blas,
    kernels: Kernels,
    workspace: Workspace,
    plans: BTreeMap<(usize, usize), LinearPlan<Bf16>>,
    projector_activation: ProjectorActivation,
}

/// Borrow of the prepared arena that produced a fusion handle. The lease keeps
/// the workspace alive while the caller enqueues the shared text decoder.
pub(crate) struct Mistral3Fusion<'a> {
    visual: DecoderVisual,
    _prepared: &'a Mistral3Prepared,
}

impl Mistral3Fusion<'_> {
    pub(crate) fn visual(&self) -> DecoderVisual {
        self.visual
    }
}

impl Mistral3Prepared {
    pub(crate) fn pending_reason(plan: &Mistral3Plan) -> Option<String> {
        if let Some(pending) = &plan.projector_pending {
            return Some(pending.reason().to_owned());
        }
        match (&plan.vision_profile, &plan.text_profile) {
            (Ok(_), Ok(())) => None,
            (Err(pending), _) => Some(pending.reason().to_owned()),
            (_, Err(pending)) => Some(pending.reason().to_owned()),
        }
    }

    pub(crate) fn new(
        plan: Mistral3Plan,
        context: Arc<Context>,
        stream: Arc<Stream>,
        fa4_root: &Path,
        max_patches: usize,
        max_image_tokens: usize,
        max_images: usize,
    ) -> Result<Self, crate::BackendError> {
        if let Some(reason) = Self::pending_reason(&plan) {
            return Err(start(format!(
                "Mistral3 execution is cold-pending: {reason}"
            )));
        }
        let profile = plan
            .vision_profile
            .clone()
            .map_err(|pending: Mistral3Pending| start(pending.reason().to_owned()))?;
        if max_patches == 0 || max_image_tokens == 0 || max_images == 0 {
            return Err(start("Mistral3 preparation bounds must be nonzero"));
        }
        let mut uploader = WeightUploader::new(stream.clone())?;
        let mut tensors = BTreeMap::new();
        for tensor in plan.weights.vision.iter().chain(&plan.weights.projector) {
            tensors.insert(tensor.name.clone(), uploader.upload(&[&tensor.source])?);
        }
        uploader.finish()?;
        let attention = crate::fa4::Fa4::new(
            context.clone(),
            fa4_root,
            plan.config.vision.attention_spec(),
        )?;
        let blas = Blas::new(context.clone(), stream.clone())?;
        let require_gelu_mul = plan.config.vision.hidden_activation != ProjectorActivation::Silu;
        let kernels = Kernels::load(&context, require_gelu_mul)?;
        let workspace = Workspace::new(
            &stream,
            plan.config.vision.hidden_size,
            plan.config.vision.intermediate_size,
            plan.config.vision.patch_dim(),
            plan.config.text.hidden_size,
            plan.config.spatial_merge_size,
            max_patches,
            max_image_tokens,
            max_images,
        )?;
        let merge_width = plan.config.vision.hidden_size
            * plan.config.spatial_merge_size
            * plan.config.spatial_merge_size;
        let mut plans = BTreeMap::new();
        for (input, output) in [
            (
                plan.config.vision.patch_dim(),
                plan.config.vision.hidden_size,
            ),
            (
                plan.config.vision.hidden_size,
                plan.config.vision.hidden_size,
            ),
            (
                plan.config.vision.hidden_size,
                plan.config.vision.intermediate_size,
            ),
            (
                plan.config.vision.intermediate_size,
                plan.config.vision.hidden_size,
            ),
            (merge_width, plan.config.vision.hidden_size),
            (plan.config.vision.hidden_size, plan.config.text.hidden_size),
            (plan.config.text.hidden_size, plan.config.text.hidden_size),
        ] {
            plans.insert(
                (input, output),
                LinearPlan::contiguous(max_patches.max(max_image_tokens), input, output)
                    .map_err(infer)?,
            );
        }
        let projector_activation = plan.config.projector_activation;
        Ok(Self {
            config: plan.config,
            weights: VisionWeights { tensors },
            profile,
            stream: stream.clone(),
            attention,
            blas,
            kernels,
            workspace,
            plans,
            projector_activation,
        })
    }

    /// Enqueue the complete Pixtral tower plus merge/projector. The returned
    /// fusion handle borrows the prepared arena until the stream drains.
    pub(crate) fn enqueue_vision(
        &mut self,
        plan: &Mistral3MediaPlan,
        stream: &Stream,
        ids: &[u32],
        upload: &mut Mistral3MediaHost,
    ) -> Result<Mistral3Fusion<'_>, crate::BackendError> {
        if !std::ptr::eq(self.stream.as_ref(), stream) {
            return Err(infer(
                "Mistral3 enqueue stream differs from the prepared cuBLAS stream",
            ));
        }
        plan.validate_markers(ids, self.config.image_token_index)
            .map_err(|error| infer(format!("Mistral3 image markers rejected: {error}")))?;
        if plan.total_patches == 0
            || plan.total_patches > self.workspace.max_patches
            || plan.total_image_tokens > self.workspace.max_tokens
            || plan.images.len() > self.workspace.max_images
        {
            return Err(infer("Mistral3 request exceeds the retained tower arena"));
        }
        if plan.images.is_empty() {
            return Err(infer("Mistral3 vision enqueue requires at least one image"));
        }
        self.upload_metadata(plan, stream, upload)?;
        let patch_dim = self.config.vision.patch_dim();
        if plan.media.images_len() != plan.images.len() {
            return Err(infer("Mistral3 media view disagrees with the plan"));
        }
        let pixels = upload
            .pixels
            .as_mut()
            .ok_or_else(|| infer("Mistral3 vision upload lacks pinned pixel staging"))?
            .as_mut_slice();
        for (image, planned) in plan.media.view().images().zip(&plan.images) {
            let elements = planned
                .patch_rows
                .checked_mul(patch_dim)
                .ok_or_else(|| infer("Mistral3 pixel element overflow"))?;
            if image.data().len() != elements {
                return Err(infer(
                    "Mistral3 pixel payload changed after media validation",
                ));
            }
            let start = planned
                .patch_offset
                .checked_mul(patch_dim)
                .ok_or_else(|| infer("Mistral3 pixel offset overflow"))?;
            let end = start
                .checked_add(elements)
                .ok_or_else(|| infer("Mistral3 pixel end overflow"))?;
            let host = pixels
                .get_mut(start..end)
                .ok_or_else(|| infer("Mistral3 pinned pixel staging overflow"))?;
            host.copy_from_slice(image.data());
            let byte_offset = start
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| infer("Mistral3 pixel byte offset overflow"))?;
            let destination = self
                .workspace
                .pixels_f32
                .cu_deviceptr()
                .checked_add(u64::try_from(byte_offset).map_err(infer)?)
                .ok_or_else(|| infer("Mistral3 pixel address overflow"))?;
            unsafe {
                cuda_core::memory::memcpy_htod_async(
                    destination,
                    host.as_ptr(),
                    elements * std::mem::size_of::<f32>(),
                    stream.cu_stream(),
                )
            }
            .map_err(infer)?;
        }
        let pixel_elements = plan
            .total_patches
            .checked_mul(patch_dim)
            .ok_or_else(|| infer("Mistral3 pixel element overflow"))?;
        let mut pixels_in = self.workspace.pixels_f32.cu_deviceptr();
        let mut pixels_out = self.workspace.pixels_bf16.cu_deviceptr();
        let mut pixel_count = u32_size(pixel_elements)?;
        launch(
            &self.kernels.pixels,
            stream,
            blocks(pixel_elements)?,
            0,
            &mut [
                arg(&mut pixels_in),
                arg(&mut pixels_out),
                arg(&mut pixel_count),
            ],
        )?;

        let h = self.config.vision.hidden_size;
        let i = self.config.vision.intermediate_size;
        self.linear(
            patch_dim,
            h,
            plan.total_patches,
            "pixels",
            "model.vision_tower.patch_conv.weight",
            "hidden",
        )?;
        self.rms(
            stream,
            "hidden",
            "normalized",
            plan.total_patches,
            h,
            1e-5,
            "model.vision_tower.ln_pre.weight",
        )?;
        self.copy(stream, "normalized", "hidden", plan.total_patches, h)?;
        // Feature index zero is the post-ln_pre snapshot. The loop selects
        // `layer + 1`, so this branch must not be overwritten by a post-loop
        // fallback.
        let mut selected_set = self.config.selects_feature_snapshot(0);
        if selected_set {
            self.copy(stream, "hidden", "selected", plan.total_patches, h)?;
        }
        for layer in 0..self.config.vision.num_hidden_layers {
            let p = format!("model.vision_tower.transformer.layers.{layer}");
            self.rms(
                stream,
                "hidden",
                "normalized",
                plan.total_patches,
                h,
                1e-5,
                &format!("{p}.attention_norm.weight"),
            )?;
            self.linear(
                h,
                h,
                plan.total_patches,
                "normalized",
                &format!("{p}.attention.q_proj.weight"),
                "q",
            )?;
            self.linear(
                h,
                h,
                plan.total_patches,
                "normalized",
                &format!("{p}.attention.k_proj.weight"),
                "k",
            )?;
            self.linear(
                h,
                h,
                plan.total_patches,
                "normalized",
                &format!("{p}.attention.v_proj.weight"),
                "v",
            )?;
            self.rope2d(stream, plan)?;
            let offsets = self.workspace.offsets.cu_deviceptr();
            let head_dim = self.config.vision.head_dim;
            let strides = [
                i64::try_from(h).map_err(infer)?,
                i64::try_from(head_dim).map_err(infer)?,
                1,
                i64::try_from(h).map_err(infer)?,
                i64::try_from(head_dim).map_err(infer)?,
                1,
                i64::try_from(h).map_err(infer)?,
                i64::try_from(head_dim).map_err(infer)?,
                1,
                i64::try_from(h).map_err(infer)?,
                i64::try_from(head_dim).map_err(infer)?,
                1,
            ];
            let batch = AttentionBatch {
                q: self.workspace.q.cu_deviceptr(),
                k: self.workspace.k.cu_deviceptr(),
                v: self.workspace.v.cu_deviceptr(),
                out: self.workspace.attended.cu_deviceptr(),
                q_offsets: offsets,
                k_offsets: offsets,
                q_tokens: plan.total_patches,
                k_tokens: plan.total_patches,
                batch: plan.images.len(),
                max_q: plan
                    .images
                    .iter()
                    .map(|image| image.patch_rows)
                    .max()
                    .unwrap_or(0),
                strides,
            };
            // SAFETY: the same context/stream created the provider and arena and
            // every allocation outlives the current stream work.
            unsafe { self.attention.forward_batch(stream, &batch)? };
            self.linear(
                h,
                h,
                plan.total_patches,
                "attended",
                &format!("{p}.attention.o_proj.weight"),
                "projection",
            )?;
            self.add(
                stream,
                "hidden",
                "projection",
                "hidden",
                plan.total_patches,
                h,
            )?;
            self.rms(
                stream,
                "hidden",
                "normalized",
                plan.total_patches,
                h,
                1e-5,
                &format!("{p}.ffn_norm.weight"),
            )?;
            self.linear(
                h,
                i,
                plan.total_patches,
                "normalized",
                &format!("{p}.feed_forward.gate_proj.weight"),
                "gate",
            )?;
            self.linear(
                h,
                i,
                plan.total_patches,
                "normalized",
                &format!("{p}.feed_forward.up_proj.weight"),
                "up",
            )?;
            self.vision_activation(stream, plan.total_patches, i)?;
            self.linear(
                i,
                h,
                plan.total_patches,
                "projection",
                &format!("{p}.feed_forward.down_proj.weight"),
                "attended",
            )?;
            self.add(
                stream,
                "hidden",
                "attended",
                "hidden",
                plan.total_patches,
                h,
            )?;
            if self.config.selects_feature_snapshot(layer + 1) {
                self.copy(stream, "hidden", "selected", plan.total_patches, h)?;
                selected_set = true;
            }
        }
        if !selected_set {
            return Err(infer(
                "Mistral3 feature layer did not match any tower snapshot",
            ));
        }
        let eps = self.config.text.rms_norm_eps as f32;
        self.rms(
            stream,
            "selected",
            "normalized",
            plan.total_patches,
            h,
            eps,
            "model.multi_modal_projector.norm.weight",
        )?;
        self.merge(stream, plan)?;
        let merge_width = h * self.config.spatial_merge_size * self.config.spatial_merge_size;
        self.linear(
            merge_width,
            h,
            plan.total_image_tokens,
            "merged",
            "model.multi_modal_projector.patch_merger.merging_layer.weight",
            "merged_hidden",
        )?;
        self.linear(
            h,
            self.config.text.hidden_size,
            plan.total_image_tokens,
            "merged_hidden",
            "model.multi_modal_projector.linear_1.weight",
            "projector_hidden",
        )?;
        self.activate(
            stream,
            "projector_hidden",
            plan.total_image_tokens,
            self.config.text.hidden_size,
        )?;
        self.linear(
            self.config.text.hidden_size,
            self.config.text.hidden_size,
            plan.total_image_tokens,
            "projector_hidden",
            "model.multi_modal_projector.linear_2.weight",
            "features",
        )?;
        Ok(Mistral3Fusion {
            visual: DecoderVisual {
                features: self.workspace.features.cu_deviceptr(),
                indices: self.workspace.indices.cu_deviceptr(),
                rows: plan.total_image_tokens,
                width: self.config.text.hidden_size,
            },
            _prepared: self,
        })
    }

    fn upload_metadata(
        &mut self,
        plan: &Mistral3MediaPlan,
        stream: &Stream,
        upload: &mut Mistral3MediaHost,
    ) -> Result<(), crate::BackendError> {
        let mut host_meta = Vec::new();
        let image_count = plan.images.len();
        for image in &plan.images {
            for value in image.grid_thw {
                host_meta
                    .push(u32::try_from(value).map_err(|_| infer("Mistral3 grid exceeds u32"))?);
            }
        }
        let grids_len = host_meta.len();
        for image in &plan.images {
            host_meta.push(
                u32::try_from(image.patch_offset)
                    .map_err(|_| infer("Mistral3 patch offset exceeds u32"))?,
            );
        }
        host_meta.push(
            u32::try_from(plan.total_patches)
                .map_err(|_| infer("Mistral3 patch count exceeds u32"))?,
        );
        for image in &plan.images {
            for rows in &image.merge_source_rows {
                for &row in rows {
                    let absolute = image
                        .patch_offset
                        .checked_add(row)
                        .ok_or_else(|| infer("Mistral3 merge source overflow"))?;
                    host_meta.push(
                        u32::try_from(absolute)
                            .map_err(|_| infer("Mistral3 merge source exceeds u32"))?,
                    );
                }
            }
        }
        for image in &plan.images {
            for &index in &image.marker_indices {
                host_meta
                    .push(u32::try_from(index).map_err(|_| infer("Mistral3 marker exceeds u32"))?);
            }
        }
        let pinned = upload.metadata.as_mut_slice();
        let destination = pinned
            .get_mut(..host_meta.len())
            .ok_or_else(|| infer("Mistral3 pinned metadata staging overflow"))?;
        destination.copy_from_slice(&host_meta);
        let copy = |destination: u64, values: &[u32]| {
            if values.is_empty() {
                return Ok(());
            }
            unsafe {
                cuda_core::memory::memcpy_htod_async(
                    destination,
                    values.as_ptr(),
                    std::mem::size_of_val(values),
                    stream.cu_stream(),
                )
            }
            .map_err(infer)
        };
        copy(
            self.workspace.grids.cu_deviceptr(),
            &destination[..grids_len],
        )?;
        copy(
            self.workspace.offsets.cu_deviceptr(),
            &destination[grids_len..grids_len + image_count + 1],
        )?;
        let merge_start = grids_len + image_count + 1;
        let merge_len = plan
            .total_image_tokens
            .checked_mul(self.config.spatial_merge_size * self.config.spatial_merge_size)
            .ok_or_else(|| infer("Mistral3 merge metadata overflow"))?;
        copy(
            self.workspace.merge_rows.cu_deviceptr(),
            &destination[merge_start..merge_start + merge_len],
        )?;
        copy(
            self.workspace.indices.cu_deviceptr(),
            &destination
                [merge_start + merge_len..merge_start + merge_len + plan.total_image_tokens],
        )?;
        Ok(())
    }

    fn linear(
        &mut self,
        input: usize,
        output: usize,
        rows: usize,
        from: &str,
        weight: &str,
        to: &str,
    ) -> Result<(), crate::BackendError> {
        let plan = compute_plan(&self.plans, input, output, rows)?;
        let weight: *const DeviceBuffer<Bf16> = self.weights.get(weight)?;
        let x: *const DeviceBuffer<Bf16> = self.workspace.buffer(from)?;
        let y: *mut DeviceBuffer<Bf16> = self.workspace.buffer_mut(to)?;
        // SAFETY: x and y are distinct arena fields, all three buffers belong to
        // the prepared context, and the caller retains them through completion.
        unsafe { self.blas.linear(&plan, &*x, &*weight, &mut *y) }
    }

    #[allow(clippy::too_many_arguments)]
    fn rms(
        &self,
        stream: &Stream,
        input: &str,
        output: &str,
        rows: usize,
        width: usize,
        eps: f32,
        weight: &str,
    ) -> Result<(), crate::BackendError> {
        let mut source = self.workspace.ptr(input)?;
        let mut scale = self.weights.get(weight)?.cu_deviceptr();
        let mut out = self.workspace.ptr(output)?;
        let mut r = u32_size(rows)?;
        let mut w = u32_size(width)?;
        let mut e = eps;
        launch(
            &self.kernels.rms,
            stream,
            (r, 1, 1),
            0,
            &mut [
                arg(&mut source),
                arg(&mut scale),
                arg(&mut out),
                arg(&mut r),
                arg(&mut w),
                arg(&mut e),
            ],
        )
    }

    fn rope2d(&self, stream: &Stream, plan: &Mistral3MediaPlan) -> Result<(), crate::BackendError> {
        let head_dim = self.config.vision.head_dim;
        let heads = self.config.vision.num_attention_heads;
        let elements = plan
            .total_patches
            .checked_mul(heads)
            .and_then(|value| value.checked_mul(head_dim / 2))
            .ok_or_else(|| infer("Mistral3 rotary element overflow"))?;
        let mut q = self.workspace.q.cu_deviceptr();
        let mut k = self.workspace.k.cu_deviceptr();
        let mut grids = self.workspace.grids.cu_deviceptr();
        let mut offsets = self.workspace.offsets.cu_deviceptr();
        let mut image_count = u32_size(plan.images.len())?;
        let mut tokens = u32_size(plan.total_patches)?;
        let mut head_count = u32_size(heads)?;
        let mut dim = u32_size(head_dim)?;
        let mut theta = self.config.vision.rope_theta as f32;
        launch(
            &self.kernels.rope2d,
            stream,
            blocks(elements)?,
            0,
            &mut [
                arg(&mut q),
                arg(&mut k),
                arg(&mut grids),
                arg(&mut offsets),
                arg(&mut image_count),
                arg(&mut tokens),
                arg(&mut head_count),
                arg(&mut dim),
                arg(&mut theta),
            ],
        )
    }

    fn add(
        &self,
        stream: &Stream,
        left: &str,
        right: &str,
        out: &str,
        rows: usize,
        width: usize,
    ) -> Result<(), crate::BackendError> {
        let elements = product(rows, width)?;
        let mut a = self.workspace.ptr(left)?;
        let mut b = self.workspace.ptr(right)?;
        let mut destination = self.workspace.ptr(out)?;
        let mut count = u32_size(elements)?;
        launch(
            &self.kernels.add,
            stream,
            blocks(elements)?,
            0,
            &mut [
                arg(&mut a),
                arg(&mut b),
                arg(&mut destination),
                arg(&mut count),
            ],
        )
    }

    fn silu(&self, stream: &Stream, rows: usize, width: usize) -> Result<(), crate::BackendError> {
        let elements = product(rows, width)?;
        let mut gate = self.workspace.gate.cu_deviceptr();
        let mut up = self.workspace.up.cu_deviceptr();
        let mut out = self.workspace.projection.cu_deviceptr();
        let mut count = u32_size(elements)?;
        launch(
            &self.kernels.silu,
            stream,
            blocks(elements)?,
            0,
            &mut [arg(&mut gate), arg(&mut up), arg(&mut out), arg(&mut count)],
        )
    }

    fn vision_activation(
        &self,
        stream: &Stream,
        rows: usize,
        width: usize,
    ) -> Result<(), crate::BackendError> {
        match self.config.vision.hidden_activation {
            ProjectorActivation::Silu => self.silu(stream, rows, width),
            ProjectorActivation::Gelu => self.gelu_mul(stream, rows, width, 0),
            ProjectorActivation::GeluTanh => self.gelu_mul(stream, rows, width, 1),
        }
    }

    fn gelu_mul(
        &self,
        stream: &Stream,
        rows: usize,
        width: usize,
        mode: u32,
    ) -> Result<(), crate::BackendError> {
        let elements = product(rows, width)?;
        let mut gate = self.workspace.gate.cu_deviceptr();
        let mut up = self.workspace.up.cu_deviceptr();
        let mut out = self.workspace.projection.cu_deviceptr();
        let mut count = u32_size(elements)?;
        let mut mode = mode;
        let function = self.kernels.gelu_mul.as_ref().ok_or_else(|| {
            start("Mistral3 GELU vision activation requires the additive gelu_mul export")
        })?;
        launch(
            function,
            stream,
            blocks(elements)?,
            0,
            &mut [
                arg(&mut gate),
                arg(&mut up),
                arg(&mut out),
                arg(&mut count),
                arg(&mut mode),
            ],
        )
    }

    fn activate(
        &self,
        stream: &Stream,
        name: &str,
        rows: usize,
        width: usize,
    ) -> Result<(), crate::BackendError> {
        let elements = product(rows, width)?;
        let mut input = self.workspace.ptr(name)?;
        let mut out = input;
        let mut count = u32_size(elements)?;
        let mut mode = match self.projector_activation {
            ProjectorActivation::Gelu => 0u32,
            ProjectorActivation::GeluTanh => 1,
            ProjectorActivation::Silu => 2,
        };
        launch(
            &self.kernels.gelu,
            stream,
            blocks(elements)?,
            0,
            &mut [
                arg(&mut input),
                arg(&mut out),
                arg(&mut count),
                arg(&mut mode),
            ],
        )
    }

    fn merge(&self, stream: &Stream, plan: &Mistral3MediaPlan) -> Result<(), crate::BackendError> {
        let merge = self.config.spatial_merge_size;
        let elements = plan
            .total_image_tokens
            .checked_mul(self.config.vision.hidden_size)
            .and_then(|value| value.checked_mul(merge * merge))
            .ok_or_else(|| infer("Mistral3 merge element overflow"))?;
        let mut input = self.workspace.normalized.cu_deviceptr();
        let mut rows = self.workspace.merge_rows.cu_deviceptr();
        let mut out = self.workspace.merged.cu_deviceptr();
        let mut tokens = u32_size(plan.total_image_tokens)?;
        let mut hidden = u32_size(self.config.vision.hidden_size)?;
        let mut size = u32_size(merge)?;
        launch(
            &self.kernels.merge,
            stream,
            blocks(elements)?,
            0,
            &mut [
                arg(&mut input),
                arg(&mut rows),
                arg(&mut out),
                arg(&mut tokens),
                arg(&mut hidden),
                arg(&mut size),
            ],
        )
    }

    fn copy(
        &self,
        stream: &Stream,
        from: &str,
        to: &str,
        rows: usize,
        width: usize,
    ) -> Result<(), crate::BackendError> {
        let bytes = product(product(rows, width)?, std::mem::size_of::<Bf16>())?;
        unsafe {
            cuda_core::memory::memcpy_dtod_async(
                self.workspace.ptr(to)?,
                self.workspace.ptr(from)?,
                bytes,
                stream.cu_stream(),
            )
        }
        .map_err(infer)
    }
}

/// Production-module qualification consumer for the cached checkpoint.
///
/// The engine binds the shared [`crate::decoder::DecoderModel`] to the prepared
/// Pixtral tower and an optional Native score head. The qualification `run`
/// path retains one decoder/host/output workspace and waits so Root can compare
/// full vectors directly. The production `enqueue` path lazily retains up to
/// two request works, stages into the caller's pinned lease and records one
/// completion fence per request without synchronising the stream.
pub(crate) struct Mistral3Engine {
    prepared: Mistral3Prepared,
    text: crate::decoder::DecoderModel,
    lt: crate::lt::LtProvider,
    blas: Blas,
    stream: Arc<Stream>,
    context: Arc<Context>,
    work: crate::decoder::DecoderWork,
    host: Vec<u32>,
    output: Vec<f32>,
    logits: DeviceBuffer<f32>,
    logit_host: Vec<f32>,
    score: Option<(DeviceBuffer<f32>, LinearPlan<crate::tensor::F32>)>,
    max_tokens: usize,
    max_sequences: usize,
    hidden: usize,
    labels: usize,
    async_works: Vec<Mistral3AsyncWork>,
}

/// Per-request workspace for the production enqueue path. Unlike the single
/// qualification workspace, two of these may be in flight at once; each owns
/// its decoder arena, logits slab and completion state.
struct Mistral3AsyncWork {
    work: crate::decoder::DecoderWork,
    logits: DeviceBuffer<f32>,
    in_flight: bool,
    quarantined: bool,
}

/// Result of successfully enqueueing one Mistral3 request. The caller retains
/// the `HostIoLease`; this only identifies the workspace that must be released
/// after the lease completion fence.
pub(crate) struct Mistral3Enqueued {
    pub(crate) work: usize,
    pub(crate) pooled_rows: usize,
    pub(crate) raw_rows: usize,
    pub(crate) output_elements: usize,
    pub(crate) media: Option<Mistral3MediaHost>,
}

/// Host result slices produced by one qualification run.
pub(crate) struct Mistral3Run<'a> {
    pub(crate) pooled: &'a [f32],
    pub(crate) raw: &'a [f32],
    pub(crate) logits: &'a [f32],
}

impl Mistral3Engine {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        plan: Mistral3Plan,
        context: Arc<Context>,
        stream: Arc<Stream>,
        lt: crate::lt::LtProvider,
        fa4_root: &Path,
        max_tokens: usize,
        max_sequences: usize,
        max_images: usize,
    ) -> Result<Self, crate::BackendError> {
        if max_tokens == 0 || max_sequences == 0 {
            return Err(start("Mistral3 engine bounds must be nonzero"));
        }
        let text_host = plan.weights.text.clone();
        let score_source = plan
            .weights
            .score
            .as_ref()
            .map(|score| score.source.clone());
        let hidden = plan.config.text.hidden_size;
        let labels = plan.weights.score.as_ref().map_or(0, |score| score.rows);
        let max_image_tokens = max_tokens;
        // Every merged image token is sourced from merge^2 pre-merge patch
        // rows, so a token budget alone would undersize the Pixtral arena for
        // a real long-edge image. The media planner rejects any plan larger
        // than this bound before enqueue.
        let merge_area = plan
            .config
            .spatial_merge_size
            .checked_mul(plan.config.spatial_merge_size)
            .ok_or_else(|| start("Mistral3 merge area overflow"))?;
        let max_patches = max_tokens
            .checked_mul(merge_area)
            .ok_or_else(|| start("Mistral3 patch arena overflow"))?;
        let prepared = Mistral3Prepared::new(
            plan,
            context.clone(),
            stream.clone(),
            fa4_root,
            max_patches,
            max_image_tokens,
            max_images,
        )?;
        let text = crate::decoder::DecoderModel::new(text_host, &context, &stream, &lt)?;
        let boundary_capacity = max_sequences
            .checked_add(1)
            .ok_or_else(|| start("Mistral3 boundary capacity overflow"))?;
        let selection_capacity = max_tokens
            .checked_add(max_sequences)
            .ok_or_else(|| start("Mistral3 selection capacity overflow"))?;
        // Raw token outputs add up to `max_tokens` rows on top of the pooled
        // selections, so both the selection metadata and host output reserve
        // the full token capacity.
        let work = crate::decoder::DecoderWork::new(
            stream.clone(),
            text.config(),
            (max_tokens, boundary_capacity, selection_capacity),
        )?;
        let host = vec![
            0u32;
            crate::staging::input_capacity((
                max_tokens,
                boundary_capacity,
                selection_capacity,
            ))?
        ];
        let output = vec![
            0f32;
            hidden
                .checked_mul(
                    max_sequences
                        .checked_add(max_tokens)
                        .ok_or_else(|| start("Mistral3 host output overflow"))?,
                )
                .ok_or_else(|| start("Mistral3 host output overflow"))?
        ];
        let logit_elements = max_sequences
            .checked_mul(labels.max(1))
            .ok_or_else(|| start("Mistral3 logit arena overflow"))?;
        let logits = unsafe { DeviceBuffer::<f32>::uninitialized_async(&stream, logit_elements) }
            .map_err(infer)?;
        let logit_host = vec![0f32; logit_elements];
        let score = match score_source {
            None => None,
            Some(source) => {
                let bits = source.len();
                if labels == 0
                    || bits
                        != labels
                            .checked_mul(hidden)
                            .ok_or_else(|| start("Mistral3 score head size overflow"))?
                {
                    return Err(start(format!(
                        "Mistral3 score head has {bits} values, expected {labels}x{hidden}"
                    )));
                }
                let mut encoded = vec![0u16; bits];
                source.read_bf16(0, &mut encoded)?;
                let mut values = Vec::new();
                values
                    .try_reserve_exact(bits)
                    .map_err(|_| start("score allocation"))?;
                for value in encoded {
                    values.push(half::bf16::from_bits(value).to_f32());
                }
                let weight = DeviceBuffer::<f32>::from_host(&stream, &values).map_err(infer)?;
                let plan =
                    LinearPlan::<crate::tensor::F32>::contiguous(max_sequences, hidden, labels)
                        .map_err(infer)?;
                Some((weight, plan))
            }
        };
        let blas = Blas::new(context.clone(), stream.clone())?;
        Ok(Self {
            prepared,
            text,
            lt,
            blas,
            stream,
            context,
            work,
            host,
            output,
            logits,
            logit_host,
            score,
            max_tokens,
            max_sequences,
            hidden,
            labels,
            async_works: Vec::new(),
        })
    }

    pub(crate) fn hidden_size(&self) -> usize {
        self.hidden
    }

    pub(crate) fn empty_media_plan(
        &self,
        batch: &infinity_v3_backend_core::Batch,
    ) -> Result<Mistral3MediaPlan, crate::BackendError> {
        Mistral3MediaPlan::build(&self.prepared.config, &batch.cumulative_seq_lengths, &[])
    }

    /// Production enqueue path: stage one request into the lease, enqueue the
    /// Pixtral tower (when present), the shared text decoder and the optional
    /// score head, then record the single host completion fence. No stream
    /// synchronisation happens here; the caller owns the returned lease.
    pub(crate) fn enqueue(
        &mut self,
        batch: &infinity_v3_backend_core::Batch,
        pool: &infinity_v3_backend_core::Pool,
        images: &[(usize, Arc<infinity_v3_backend_core::ImagePatches>)],
        lease: &mut crate::staging::HostIoLease,
        apply_score: bool,
    ) -> Result<Mistral3Enqueued, crate::BackendError> {
        if let Err(error) = self.context.bind_to_thread() {
            lease.poison();
            return Err(infer(error));
        }
        let max_length = crate::validate_mistral3(batch, &self.prepared.config)?;
        let tokens = batch.input_ids.len();
        let sequences = batch
            .cumulative_seq_lengths
            .len()
            .checked_sub(1)
            .ok_or_else(|| infer("Missing Mistral3 sequence boundaries"))?;
        if tokens > self.max_tokens || sequences > self.max_sequences {
            return Err(infer("Mistral3 request exceeds the retained engine arena"));
        }
        let media =
            Mistral3MediaPlan::build(&self.prepared.config, &batch.cumulative_seq_lengths, images)?;
        self.text
            .preflight(&self.lt, batch, pool)
            .map_err(|error| infer(format!("Mistral3 text preflight: {error:?}")))?;
        let rows = crate::output::rows(batch)?;
        let selection_capacity = self
            .max_tokens
            .checked_add(self.max_sequences)
            .ok_or_else(|| infer("Mistral3 selection capacity overflow"))?;
        if rows > selection_capacity {
            return Err(infer(
                "Mistral3 pooled plus raw rows exceed the retained selection capacity",
            ));
        }
        let elements = crate::InputLayout::new(
            tokens,
            sequences
                .checked_add(1)
                .ok_or_else(|| infer("Mistral3 boundary overflow"))?,
            rows,
        )?
        .words();
        let pooled_rows = batch.pooled_indices.len();
        let raw_rows = crate::output::raw_rows(batch)?;
        let output_elements = if apply_score {
            if self.score.is_none() || self.labels == 0 {
                return Err(infer("Mistral3 classifier enqueue lacks a score head"));
            }
            pooled_rows
                .checked_mul(self.labels)
                .ok_or_else(|| infer("Mistral3 logit element overflow"))?
        } else {
            pooled_rows
                .checked_add(raw_rows)
                .and_then(|count| count.checked_mul(self.hidden))
                .ok_or_else(|| infer("Mistral3 output element overflow"))?
        };
        let patch_dim = self.prepared.config.vision.patch_dim();
        let merge_area = self
            .prepared
            .config
            .spatial_merge_size
            .checked_mul(self.prepared.config.spatial_merge_size)
            .ok_or_else(|| infer("Mistral3 merge area overflow"))?;
        let mut media_owner = if media.images.is_empty() {
            None
        } else {
            Some(Mistral3MediaHost::new(
                &self.stream,
                &media,
                patch_dim,
                merge_area,
            )?)
        };
        if self.async_works.len() < 2 {
            let boundary_capacity = self
                .max_sequences
                .checked_add(1)
                .ok_or_else(|| infer("Mistral3 boundary capacity overflow"))?;
            let selection_capacity = self
                .max_tokens
                .checked_add(self.max_sequences)
                .ok_or_else(|| infer("Mistral3 selection capacity overflow"))?;
            let work = crate::decoder::DecoderWork::new(
                self.stream.clone(),
                self.text.config(),
                (self.max_tokens, boundary_capacity, selection_capacity),
            )?;
            let logit_elements = self
                .max_sequences
                .checked_mul(self.labels.max(1))
                .ok_or_else(|| infer("Mistral3 logit arena overflow"))?;
            let logits =
                unsafe { DeviceBuffer::<f32>::uninitialized_async(&self.stream, logit_elements) }
                    .map_err(infer)?;
            self.async_works.push(Mistral3AsyncWork {
                work,
                logits,
                in_flight: false,
                quarantined: false,
            });
        }
        let index = self
            .async_works
            .iter()
            .position(|work| !work.in_flight && !work.quarantined)
            .ok_or_else(|| {
                crate::BackendError::Overloaded(
                    "Mistral3 retains at most two requests in flight".into(),
                )
            })?;
        let key = (
            tokens,
            sequences
                .checked_add(1)
                .ok_or_else(|| infer("Mistral3 boundary overflow"))?,
            rows,
        );
        let exclusive = lease
            .required_bytes_with_input(key, elements, output_elements)?
            .is_some_and(|bytes| bytes > crate::cache::BYTE_BUDGET);
        lease.prepare_with_input(key, elements, output_elements, exclusive)?;
        let ticket = lease.begin_tracked()?;
        self.async_works[index].work.mark_last_use(ticket);
        let staged = (|| -> Result<(), crate::BackendError> {
            let host = lease
                .slot_mut()?
                .input
                .as_mut_slice()
                .get_mut(..elements)
                .ok_or_else(|| infer("Mistral3 host staging overflow"))?;
            let pooled_metadata =
                crate::InputLayout::new(tokens, sequences + 1, pooled_rows)?.words();
            let (pooled_input, raw_input) = host
                .split_at_mut_checked(pooled_metadata)
                .ok_or_else(|| infer("Mistral3 raw staging layout"))?;
            crate::pack_input_with_position_offset(
                pooled_input,
                tokens,
                &batch.input_ids,
                &batch.position_ids,
                &batch.token_type_ids,
                &batch.cumulative_seq_lengths,
                &batch.pooled_indices,
                0,
            )?;
            crate::output::write_raw_indices(batch, raw_input)?;
            self.async_works[index].work.upload(host, elements)
        })();
        if let Err(error) = staged {
            self.async_works[index].quarantined = true;
            lease.poison();
            return Err(error);
        }
        let result = {
            let Self {
                prepared,
                text,
                lt,
                blas,
                stream,
                score,
                hidden,
                labels: _,
                async_works,
                ..
            } = self;
            let work = &mut async_works[index];
            (|| -> Result<(), crate::BackendError> {
                if media.images.is_empty() {
                    text.enqueue(
                        lt,
                        Some(blas),
                        stream,
                        &mut work.work,
                        batch,
                        pool,
                        max_length,
                        tokens,
                        None,
                    )?;
                } else {
                    let upload = media_owner.as_mut().ok_or_else(|| {
                        infer("Mistral3 vision request lacks pinned media staging")
                    })?;
                    let fusion =
                        prepared.enqueue_vision(&media, stream, &batch.input_ids, upload)?;
                    text.enqueue_visual(
                        lt,
                        Some(blas),
                        stream,
                        &mut work.work,
                        batch,
                        pool,
                        max_length,
                        tokens,
                        fusion.visual(),
                    )?;
                }
                if apply_score {
                    let (weight, plan) = score
                        .as_ref()
                        .ok_or_else(|| infer("Mistral3 classifier enqueue lacks a score head"))?;
                    let mut plan = plan.clone();
                    plan.set_rows(pooled_rows).map_err(infer)?;
                    let pooled = work.work.pooled()?;
                    // SAFETY: the pooled slab, logits slab and score weight all
                    // belong to this engine context and outlive the stream work.
                    unsafe { blas.linear(&plan, pooled, weight, &mut work.logits)? };
                    let host = lease.slot_mut()?.output.as_mut_slice();
                    crate::download_output_range(
                        stream,
                        &work.logits,
                        0,
                        host,
                        0,
                        output_elements,
                    )?;
                    lease.record()?;
                } else {
                    work.work.finish(
                        None,
                        lt,
                        Some(blas),
                        pooled_rows,
                        raw_rows,
                        *hidden,
                        lease,
                    )?;
                }
                Ok(())
            })()
        };
        if let Err(error) = result {
            self.async_works[index].quarantined = true;
            lease.poison();
            if let Some(owner) = media_owner.take() {
                // The stream may already have consumed part of this request's
                // pageable-to-pinned upload. A poisoned engine is never reused,
                // so retaining the pinned owner is the only safe action.
                std::mem::forget(owner);
            }
            return Err(error);
        }
        self.async_works[index].in_flight = true;
        Ok(Mistral3Enqueued {
            work: index,
            pooled_rows,
            raw_rows,
            output_elements,
            media: media_owner,
        })
    }

    /// Release a completed async workspace. A failed fence quarantines it and
    /// poisons the engine so uncertain device state is never reused.
    pub(crate) fn complete_work(&mut self, index: usize, healthy: bool) {
        if let Some(work) = self.async_works.get_mut(index) {
            if healthy {
                work.in_flight = false;
            } else {
                work.quarantined = true;
            }
        }
    }

    /// Build an owned media plan for this engine's actual config, then run.
    pub(crate) fn run_images(
        &mut self,
        batch: &infinity_v3_backend_core::Batch,
        pool: &infinity_v3_backend_core::Pool,
        images: &[(usize, Arc<infinity_v3_backend_core::ImagePatches>)],
    ) -> Result<Mistral3Run<'_>, crate::BackendError> {
        let media =
            Mistral3MediaPlan::build(&self.prepared.config, &batch.cumulative_seq_lengths, images)?;
        self.run(batch, pool, &media)
    }

    /// Execute one packed text/media request through the real shared decoder
    /// and the owned Pixtral tower. `pool` controls the text pooling mode;
    /// `raw_indices` emit token states in caller order. The returned slices are
    /// valid until the next call.
    pub(crate) fn run(
        &mut self,
        batch: &infinity_v3_backend_core::Batch,
        pool: &infinity_v3_backend_core::Pool,
        media: &Mistral3MediaPlan,
    ) -> Result<Mistral3Run<'_>, crate::BackendError> {
        self.context.bind_to_thread().map_err(infer)?;
        let max_length = crate::validate_mistral3(batch, &self.prepared.config)?;
        let tokens = batch.input_ids.len();
        let sequences = batch
            .cumulative_seq_lengths
            .len()
            .checked_sub(1)
            .ok_or_else(|| infer("Missing Mistral3 sequence boundaries"))?;
        if tokens > self.max_tokens || sequences > self.max_sequences {
            return Err(infer("Mistral3 request exceeds the retained engine arena"));
        }
        self.text
            .preflight(&self.lt, batch, pool)
            .map_err(|error| infer(format!("Mistral3 text preflight: {error:?}")))?;
        let rows = crate::output::rows(batch)?;
        let elements = crate::InputLayout::new(
            tokens,
            sequences
                .checked_add(1)
                .ok_or_else(|| infer("Mistral3 boundary overflow"))?,
            rows,
        )?
        .words();
        let host = self
            .host
            .get_mut(..elements)
            .ok_or_else(|| infer("Mistral3 host staging overflow"))?;
        let pooled_metadata =
            crate::InputLayout::new(tokens, sequences + 1, batch.pooled_indices.len())?.words();
        let (pooled_input, raw_input) = host
            .split_at_mut_checked(pooled_metadata)
            .ok_or_else(|| infer("Mistral3 raw staging layout"))?;
        crate::pack_input_with_position_offset(
            pooled_input,
            tokens,
            &batch.input_ids,
            &batch.position_ids,
            &batch.token_type_ids,
            &batch.cumulative_seq_lengths,
            &batch.pooled_indices,
            0,
        )?;
        crate::output::write_raw_indices(batch, raw_input)?;
        self.work.upload(host, elements)?;
        let patch_dim = self.prepared.config.vision.patch_dim();
        let merge_area = self
            .prepared
            .config
            .spatial_merge_size
            .checked_mul(self.prepared.config.spatial_merge_size)
            .ok_or_else(|| infer("Mistral3 merge area overflow"))?;
        let mut upload_owner = if media.images.is_empty() {
            None
        } else {
            Some(Mistral3MediaHost::new(
                &self.stream,
                media,
                patch_dim,
                merge_area,
            )?)
        };
        let enqueued = (|| -> Result<(), crate::BackendError> {
            if media.images.is_empty() {
                self.text.enqueue(
                    &self.lt,
                    Some(&self.blas),
                    &self.stream,
                    &mut self.work,
                    batch,
                    pool,
                    max_length,
                    tokens,
                    None,
                )?;
            } else {
                let upload = upload_owner
                    .as_mut()
                    .ok_or_else(|| infer("Mistral3 vision request lacks pinned media staging"))?;
                let fusion =
                    self.prepared
                        .enqueue_vision(media, &self.stream, &batch.input_ids, upload)?;
                self.text.enqueue_visual(
                    &self.lt,
                    Some(&self.blas),
                    &self.stream,
                    &mut self.work,
                    batch,
                    pool,
                    max_length,
                    tokens,
                    fusion.visual(),
                )?;
            }
            Ok(())
        })();
        if let Err(error) = enqueued {
            if let Some(owner) = upload_owner.take() {
                std::mem::forget(owner);
            }
            return Err(error);
        }
        let pooled_rows = batch.pooled_indices.len();
        let raw_rows = crate::output::raw_rows(batch)?;
        let total = pooled_rows
            .checked_add(raw_rows)
            .and_then(|value| value.checked_mul(self.hidden))
            .ok_or_else(|| infer("Mistral3 output element overflow"))?;
        let output = self
            .output
            .get_mut(..total)
            .ok_or_else(|| infer("Mistral3 host output overflow"))?;
        self.work.download(&self.stream, total, output)?;
        let pooled_end = pooled_rows
            .checked_mul(self.hidden)
            .ok_or_else(|| infer("Mistral3 pooled output overflow"))?;
        let (pooled, raw) = output.split_at_mut(pooled_end);
        let mut logits = &[][..];
        if let Some((weight, plan)) = self.score.as_mut() {
            let mut plan = plan.clone();
            plan.set_rows(pooled_rows).map_err(infer)?;
            // SAFETY: retained pooled/logit buffers belong to the engine
            // context; the score weight is uploaded once at construction.
            let pooled = self.work.pooled()?;
            unsafe {
                self.blas.linear(&plan, pooled, weight, &mut self.logits)?;
            }
            let logit_elements = pooled_rows
                .checked_mul(self.labels)
                .ok_or_else(|| infer("Mistral3 logit element overflow"))?;
            let bytes = logit_elements
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| infer("Mistral3 logit byte overflow"))?;
            if bytes > 0 {
                unsafe {
                    cuda_core::memory::memcpy_dtoh_async(
                        self.logit_host.as_mut_ptr(),
                        self.logits.cu_deviceptr(),
                        bytes,
                        self.stream.cu_stream(),
                    )
                }
                .map_err(infer)?;
                self.stream.synchronize().map_err(infer)?;
            }
            logits = &self.logit_host[..logit_elements];
        }
        Ok(Mistral3Run {
            pooled: &pooled[..],
            raw: &raw[..],
            logits,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::media_metadata_elements;

    #[test]
    fn media_metadata_capacity_matches_the_pinned_layout_and_rejects_overflow() {
        assert_eq!(
            media_metadata_elements(2, 10, 4).unwrap(),
            2 * 4 + 1 + 40 + 10
        );
        assert_eq!(media_metadata_elements(0, 0, 1).unwrap(), 1);
        assert!(media_metadata_elements(usize::MAX, 1, 1).is_err());
        assert!(media_metadata_elements(1, usize::MAX, 2).is_err());
    }
}
