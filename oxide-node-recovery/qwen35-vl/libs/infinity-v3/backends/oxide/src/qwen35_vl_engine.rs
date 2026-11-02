// Owning Qwen3.5-VL engine leaf.
//!
//! This module is additive: it composes the stable public `OxideBackend` text
//! facade with the isolated `qwen35_vl_*` vision/tower leaf. It does not
//! overwrite any text-worker file. The only shared hooks are:
//!
//! * `Qwen35Model::enqueue(..., vl)` copies merged visual rows into the gathered
//!   embedding slab before the layer loop and selects the per-request external
//!   partial-mRoPE tables for full-attention layers;
//! * `PreparedWork::enqueue`/`enqueue_block` accept an optional rope view.
//!
//! Gated-delta layers never consume the external positions; only full-attention
//! layers do, matching the Native `rotate` call site.
//!
//! Inputs are the original packed processor payloads (`[rows, patch_dim]` F32).
//! Visual features are produced by the same FA4/LtProvider tower path and are
//! retained with the request until the engine's completion boundary drains.

use crate::fa4::{AttentionSpec, DType as Fa4Dtype, Fa4, Scale, Window};
use crate::model::Checkpoint;
use crate::qwen35_config::Qwen35Config;
use crate::qwen35_execute::Qwen35RopeOverride;
use crate::qwen35_vl_config::Dtype;
use crate::qwen35_vl_config::Qwen35VlConfig;
use crate::qwen35_vl_device::HostPin;
use crate::qwen35_vl_inventory::validate_vision_inventory_path;
use crate::qwen35_vl_media::{MediaGrid, MediaKind, MediaSpan};
use crate::qwen35_vl_text::prepare_text_fusion;
use crate::qwen35_vl_vision::HostImage;
use crate::staging::{HostIo, SlotPool};
use crate::tensor::Bf16;
use crate::{HostIoLease, OxideBackend, Qwen35TextOptions};
use cuda_core::{Context, DeviceBuffer, Stream};
use infinity_v3_backend_core::{
    BackendError, Batch, DecisionInput, DecisionOutput, Embedding, Embeddings, Inference,
    MultimodalEncoding,
};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

fn start(error: impl std::fmt::Display) -> BackendError {
    BackendError::Start(format!("Oxide qwen3.5-vl engine: {error}"))
}

fn infer(error: impl std::fmt::Display) -> BackendError {
    BackendError::Inference(format!("Oxide qwen3.5-vl engine: {error}"))
}

fn product(left: usize, right: usize) -> Result<usize, BackendError> {
    left.checked_mul(right)
        .ok_or_else(|| start("dimension overflow"))
}

/// Cold limits for one VL request. The defaults match the original 262144
/// position budget and reject accidental token-count growth before staging.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen35VlOptions {
    pub max_tokens: usize,
}

impl Default for Qwen35VlOptions {
    fn default() -> Self {
        Self {
            max_tokens: 262_144,
        }
    }
}

/// One original media item with already packed processor rows.
#[derive(Clone, Debug)]
pub struct Qwen35VlRequest {
    pub name: String,
    pub kind: MediaKind,
    pub pixels: Vec<f32>,
    pub grid: MediaGrid,
    pub token_start: usize,
}

impl Qwen35VlRequest {
    fn host_image(&self) -> HostImage {
        HostImage {
            name: self.name.clone(),
            pixels: self.pixels.clone(),
            grid: self.grid,
            token_start: self.token_start,
        }
    }

    fn span(&self) -> Result<MediaSpan, BackendError> {
        Ok(MediaSpan {
            kind: self.kind,
            token_start: self.token_start,
            token_count: self.grid.merged_tokens()?,
            grid: self.grid,
        })
    }
}

pub(crate) trait Fence: Send + Sync {
    fn proven(&self) -> bool;
}

impl Fence for Stream {
    fn proven(&self) -> bool {
        self.synchronize().is_ok()
    }
}

pub(crate) struct Qwen35VlSpan {
    pub(crate) token_start: usize,
    pub(crate) token_count: usize,
    pub(crate) workspace: crate::qwen35_vl_device::VisionDeviceOutput,
}

pub(crate) struct Qwen35VlRope {
    pub(crate) cosine: DeviceBuffer<f32>,
    pub(crate) sine: DeviceBuffer<f32>,
    pub(crate) positions: DeviceBuffer<u32>,
    pub(crate) pairs: usize,
}

pub(crate) struct Qwen35VlFusion {
    pub(crate) spans: Vec<Qwen35VlSpan>,
    pub(crate) rope: Option<Qwen35VlRope>,
    pub(crate) host_pins: Vec<HostPin>,
    runtime: Option<Arc<Mutex<Qwen35VlRuntime>>>,
    failure: Option<Qwen35VlFailure>,
    stream: Arc<dyn Fence>,
    completed: bool,
    #[cfg(test)]
    retained_token: Option<Arc<()>>,
}

impl Qwen35VlFusion {
    pub(crate) fn rope_override(&self) -> Option<Qwen35RopeOverride> {
        self.rope.as_ref().map(|rope| Qwen35RopeOverride {
            cosine: rope.cosine.cu_deviceptr(),
            sine: rope.sine.cu_deviceptr(),
            positions: rope.positions.cu_deviceptr(),
            table_stride: u32::try_from(rope.pairs).unwrap_or(u32::MAX),
        })
    }

    /// Copy each merged visual row span over the gathered embedding rows.
    ///
    /// # Safety
    /// The caller retains every feature buffer until the same stream drains and
    /// guarantees `residual` covers `tokens * hidden` BF16 elements.
    pub(crate) unsafe fn copy_rows(
        &self,
        stream: &Stream,
        residual: &DeviceBuffer<Bf16>,
        tokens: usize,
        hidden: usize,
    ) -> Result<(), BackendError> {
        for span in &self.spans {
            let end = span
                .token_start
                .checked_add(span.token_count)
                .ok_or_else(|| infer("visual span overflow"))?;
            let features = &span.workspace.workspace().image_merged;
            let feature_elements =
                product(span.workspace.merged_rows(), span.workspace.out_hidden())?;
            if end > tokens || feature_elements != product(span.token_count, hidden)? {
                return Err(infer("visual span exceeds the text embedding rows"));
            }
            let bytes = product(product(span.token_count, hidden)?, 2)?;
            let offset = product(product(span.token_start, hidden)?, 2)?;
            let destination = residual
                .cu_deviceptr()
                .checked_add(u64::try_from(offset).map_err(infer)?)
                .ok_or_else(|| infer("visual destination pointer overflow"))?;
            // SAFETY: both allocations are BF16 and retained through drain.
            unsafe {
                cuda_core::memory::memcpy_dtod_async(
                    destination,
                    features.cu_deviceptr(),
                    bytes,
                    stream.cu_stream(),
                )
            }
            .map_err(infer)?;
        }
        Ok(())
    }

    pub(crate) fn set_runtime(&mut self, runtime: Arc<Mutex<Qwen35VlRuntime>>) {
        self.runtime = Some(runtime);
    }

    pub(crate) fn set_failure(&mut self, failure: Qwen35VlFailure) {
        self.failure = Some(failure);
    }

    pub(crate) fn mark_completed(&mut self) {
        for span in &mut self.spans {
            span.workspace.mark_completed();
        }
        for pin in &mut self.host_pins {
            pin.disarm();
        }
        self.completed = true;
    }

    #[cfg(test)]
    pub(crate) fn test_guard(
        failure: Qwen35VlFailure,
        stream: Arc<dyn Fence>,
        completed: bool,
        retained_token: Option<Arc<()>>,
    ) -> Self {
        Self {
            spans: Vec::new(),
            rope: None,
            host_pins: Vec::new(),
            runtime: None,
            failure: Some(failure),
            stream,
            completed,
            #[cfg(test)]
            retained_token,
        }
    }
}

impl Drop for Qwen35VlFusion {
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        if self.stream.proven() {
            return;
        }
        if let Some(rope) = self.rope.take() {
            std::mem::forget(rope);
        }
        if !self.host_pins.is_empty() {
            std::mem::forget(std::mem::take(&mut self.host_pins));
        }
        if !self.spans.is_empty() {
            std::mem::forget(std::mem::take(&mut self.spans));
        }
        if let Some(failure) = self.failure.take() {
            failure.fail_closed();
        }
        if let Some(runtime) = self.runtime.take() {
            std::mem::forget(runtime);
        }
    }
}

/// Pure CPU media preflight shared by the leaf and normal-factory routes.
pub(crate) fn validate_media_requests(
    config: &Qwen35VlConfig,
    tokens: usize,
    max_tokens: usize,
    requests: &[Qwen35VlRequest],
) -> Result<(), BackendError> {
    if tokens == 0 || tokens > max_tokens {
        return Err(infer(format!(
            "VL token count {tokens} is outside 1..={max_tokens}"
        )));
    }
    let patch_dim = config.vision.patch_dim()?;
    let merge = config.vision.spatial_merge_size;
    let mut spans = Vec::with_capacity(requests.len());
    for request in requests {
        if request.grid.merge != merge {
            return Err(infer("request merge size differs from the vision config"));
        }
        let rows = request.grid.patch_rows()?;
        if request.pixels.len() != product(rows, patch_dim)? {
            return Err(infer("visual request pixel payload length mismatch"));
        }
        let count = request.grid.merged_tokens()?;
        request
            .token_start
            .checked_add(count)
            .filter(|end| *end <= tokens)
            .ok_or_else(|| infer("visual token span exceeds the text token count"))?;
        spans.push(request.span()?);
    }
    prepare_text_fusion(config, tokens, &spans)?;
    Ok(())
}

/// Cold tower owner for one admitted checkpoints. The tower uses the text
/// engine's context and stream so the visual-row copies are same-stream.
pub struct Qwen35VlRuntime {
    config: Qwen35VlConfig,
    tower: Option<crate::Qwen35VlTower>,
    context: Arc<Context>,
    stream: Arc<Stream>,
    options: Qwen35VlOptions,
    failed: Arc<AtomicBool>,
}

impl Drop for Qwen35VlRuntime {
    fn drop(&mut self) {
        if self.failed.load(Ordering::Acquire) {
            if let Some(tower) = self.tower.take() {
                // Uncertain stream: retain the tower weights/modules/provider
                // rather than unloading after an unproven fence.
                std::mem::forget(tower);
            }
        }
    }
}

impl Qwen35VlRuntime {
    pub(crate) fn new(
        path: &Path,
        config: Qwen35VlConfig,
        context: &Arc<Context>,
        stream: &Arc<Stream>,
        options: Qwen35VlOptions,
    ) -> Result<Self, BackendError> {
        let checkpoint = Checkpoint::open(path)?;
        let tower = crate::Qwen35VlTower::attach(config.clone(), &checkpoint, context, stream)?;
        Ok(Self {
            config,
            tower: Some(tower),
            context: context.clone(),
            stream: stream.clone(),
            options,
            failed: Arc::new(AtomicBool::new(false)),
        })
    }

    pub(crate) fn build_fusion(
        &self,
        tokens: usize,
        requests: &[Qwen35VlRequest],
    ) -> Result<Qwen35VlFusion, BackendError> {
        if tokens == 0 || tokens > self.options.max_tokens {
            return Err(infer(format!(
                "VL token count {tokens} is outside 1..={}",
                self.options.max_tokens
            )));
        }
        let images: Vec<HostImage> = requests.iter().map(Qwen35VlRequest::host_image).collect();
        let outputs = self
            .tower
            .as_ref()
            .ok_or_else(|| infer("VL runtime tower is quarantined"))?
            .image_features_device(&images)?;
        if outputs.len() != requests.len() {
            return Err(infer("tower output count differs from request count"));
        }
        let mut spans = requests
            .iter()
            .map(Qwen35VlRequest::span)
            .collect::<Result<Vec<_>, _>>()?;
        spans.sort_unstable_by_key(|span| span.token_start);
        let mut fusion_spans = Vec::with_capacity(requests.len());
        for (request, output) in requests.iter().zip(outputs) {
            let token_count = request.grid.merged_tokens()?;
            if output.merged_rows() != token_count
                || output.out_hidden() != self.config.text.hidden_size
                || product(output.merged_rows(), output.out_hidden())?
                    != product(token_count, output.out_hidden())?
            {
                return Err(infer(
                    "tower merged geometry differs from the text hidden size",
                ));
            }
            fusion_spans.push(Qwen35VlSpan {
                token_start: request.token_start,
                token_count,
                workspace: output,
            });
        }
        let plan = prepare_text_fusion(&self.config, tokens, &spans)?;
        let half_dim = plan.phases.half_dim;
        let pairs = u32::try_from(half_dim).map_err(|_| infer("mRoPE pair width exceeds u32"))?;
        let cosine_pin = HostPin::f32(&self.stream, &plan.phases.cos)?;
        // SAFETY: the pin is retained in the fusion until the request fence.
        let cosine = unsafe { DeviceBuffer::from_pinned_host(&self.stream, cosine_pin.as_f32()) }
            .map_err(infer)?;
        let sine_pin = HostPin::f32(&self.stream, &plan.phases.sin)?;
        // SAFETY: the pin is retained in the fusion until the request fence.
        let sine = unsafe { DeviceBuffer::from_pinned_host(&self.stream, sine_pin.as_f32()) }
            .map_err(infer)?;
        let identity: Vec<u32> = (0..tokens as u64)
            .map(|index| u32::try_from(index).map_err(|_| infer("token index exceeds u32")))
            .collect::<Result<_, _>>()?;
        let positions_pin = HostPin::u32(&self.stream, &identity)?;
        // SAFETY: the pin is retained in the fusion until the request fence.
        let positions =
            unsafe { DeviceBuffer::from_pinned_host(&self.stream, positions_pin.as_u32()) }
                .map_err(infer)?;
        Ok(Qwen35VlFusion {
            spans: fusion_spans,
            rope: Some(Qwen35VlRope {
                cosine,
                sine,
                positions,
                pairs: pairs as usize,
            }),
            host_pins: vec![cosine_pin, sine_pin, positions_pin],
            runtime: None,
            failure: None,
            stream: self.stream.clone(),
            completed: false,
            #[cfg(test)]
            retained_token: None,
        })
    }

    /// Convert the ordinary server Batch.multimodal image entries into one
    /// request fusion. Audio is rejected because the Native Qwen3.5 VL family
    /// in scope is image/video only.
    fn requests_from_batch(
        &self,
        cumulative: &[u32],
        media: &[Option<Arc<MultimodalEncoding>>],
    ) -> Result<Vec<Qwen35VlRequest>, BackendError> {
        if media.is_empty() {
            return Ok(Vec::new());
        }
        if cumulative.len() != media.len() + 1 {
            return Err(infer(
                "media sequence count differs from cumulative boundaries",
            ));
        }
        let mut requests = Vec::new();
        for (sequence, entry) in media.iter().enumerate() {
            let Some(entry) = entry else { continue };
            if !entry.audios.is_empty() {
                return Err(infer(
                    "Qwen3.5-VL audio is outside the image/video contract",
                ));
            }
            let base = usize::try_from(cumulative[sequence])
                .map_err(|_| infer("sequence token offset exceeds usize"))?;
            for (image_index, (start, patches)) in entry.images.iter().enumerate() {
                let grid = MediaGrid::new(
                    patches.grid_thw[0],
                    patches.grid_thw[1],
                    patches.grid_thw[2],
                    patches.merge_size,
                )?;
                let token_start = base
                    .checked_add(*start)
                    .ok_or_else(|| infer("image token start overflow"))?;
                requests.push(Qwen35VlRequest {
                    name: format!("sequence-{sequence}-image-{image_index}"),
                    kind: MediaKind::Image,
                    pixels: patches.pixels.clone(),
                    grid,
                    token_start,
                });
            }
        }
        Ok(requests)
    }

    /// Pure CPU preflight for a leaf request slice. It validates grid/pixel/span
    /// arithmetic and the external mRoPE plan before any empty-selector return
    /// or tower work.
    pub(crate) fn validate_requests(
        &self,
        tokens: usize,
        requests: &[Qwen35VlRequest],
    ) -> Result<(), BackendError> {
        validate_media_requests(&self.config, tokens, self.options.max_tokens, requests)
    }

    pub(crate) fn validate_batch_media(
        &self,
        tokens: usize,
        cumulative: &[u32],
        media: &[Option<Arc<MultimodalEncoding>>],
    ) -> Result<(), BackendError> {
        let requests = self.requests_from_batch(cumulative, media)?;
        if requests.is_empty() {
            Ok(())
        } else {
            self.validate_requests(tokens, &requests)
        }
    }

    pub(crate) fn mark_failed(&mut self) {
        self.failed.store(true, Ordering::Release);
    }

    pub(crate) fn failed_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.failed)
    }

    pub(crate) fn failure_handle(
        &self,
        engine: Arc<Mutex<crate::Engine>>,
        io: Arc<SlotPool<HostIo>>,
    ) -> Qwen35VlFailure {
        Qwen35VlFailure::backend(Arc::clone(&self.failed), engine, io)
    }

    pub(crate) fn build_fusion_from_batch(
        &self,
        tokens: usize,
        cumulative: &[u32],
        media: &[Option<Arc<MultimodalEncoding>>],
    ) -> Result<Option<Qwen35VlFusion>, BackendError> {
        let requests = self.requests_from_batch(cumulative, media)?;
        if requests.is_empty() {
            Ok(None)
        } else {
            self.validate_requests(tokens, &requests)?;
            self.build_fusion(tokens, &requests).map(Some)
        }
    }
}

/// Owning Qwen3.5-VL engine. It keeps the admitted text backend and the tower
/// on one context/stream and exposes the ordinary pooled/raw embedding result.
pub struct Qwen35VlEngine {
    backend: OxideBackend,
    runtime: Arc<Mutex<Qwen35VlRuntime>>,
    config: Qwen35VlConfig,
}

impl Qwen35VlEngine {
    /// Attach the VL leaf to an already admitted public text backend. The
    /// backend's config must be a dense Qwen3.5 candidate (Elephant/Mammoth
    /// full checkpoints) or any future full-family text admission.
    pub fn attach(
        mut backend: OxideBackend,
        path: &Path,
        options: Qwen35VlOptions,
    ) -> Result<Self, BackendError> {
        let plan = Self::run_plan(path)?;
        if !plan.normal_factory_ready {
            return Err(start(format!(
                "VL attach requires a ready normal-factory plan: {}",
                plan.gaps.join("; ")
            )));
        }
        let (context, stream) = {
            let engine = backend.engine.lock().map_err(|_| BackendError::Unhealthy)?;
            (engine.context.clone(), backend.stream.clone())
        };
        let config =
            Qwen35VlConfig::from_path(path).or_else(|_| Qwen35VlConfig::from_config_path(path))?;
        if config.text.dtype != crate::qwen35_vl_config::Dtype::Bf16 {
            return Err(start(
                "the owning VL engine is BF16-only; F16/F32 device recipes are not attached",
            ));
        }
        if config.text.hidden_size != backend.config.hidden_size() {
            return Err(start(
                "vision out_hidden/text hidden size differs from the admitted text backend",
            ));
        }
        let runtime = match backend.qwen35_vl.take() {
            Some(runtime) => runtime,
            None => Arc::new(Mutex::new(Qwen35VlRuntime::new(
                path,
                config.clone(),
                &context,
                &stream,
                options,
            )?)),
        };
        Ok(Self {
            backend,
            runtime,
            config,
        })
    }

    /// Convenience constructor for the dense full Elephant/Mammoth checkpoints.
    pub fn new(
        path: &Path,
        pool: infinity_v3_backend_core::Pool,
        device_id: usize,
        text: Qwen35TextOptions,
        options: Qwen35VlOptions,
    ) -> Result<Self, BackendError> {
        let attention = crate::qwen35_serving::decision_attention(path)?;
        let backend = if attention == crate::qwen35_serving::DecisionAttention::BlockBidirectional {
            let mut text = text;
            text.causal = false;
            OxideBackend::new_qwen35_dynamic_block_decision_candidate(path, device_id, text)?
        } else {
            OxideBackend::new_qwen35_text_candidate(path, pool, device_id, text)?
        };
        Self::attach(backend, path, options)
    }

    pub fn config(&self) -> &Qwen35VlConfig {
        &self.config
    }

    pub fn backend(&self) -> &OxideBackend {
        &self.backend
    }

    /// Submit one fused prefill. `batch` carries the original text ids, packed
    /// positions/boundaries and pooled/raw selectors; `requests` carry the
    /// original media spans. Visual rows replace the image-token embeddings in
    /// place and full-attention layers consume the per-token 3-axis mRoPE.
    pub fn submit_embed(
        &self,
        batch: Batch,
        requests: &[Qwen35VlRequest],
    ) -> Result<Inference<Embeddings>, BackendError> {
        if self.backend.readout.is_some() {
            return Err(infer(
                "VL embedding does not use the classifier readout route",
            ));
        }
        if matches!(
            &self.backend.config,
            crate::LoadedConfig::Qwen35DynamicBlock(_)
        ) {
            return Err(infer(
                "VL dynamic block Decision is not admitted on the embedding route",
            ));
        }
        let tokens = batch.input_ids.len();
        self.backend.ensure_open()?;
        if batch.multimodal.iter().any(Option::is_some) {
            return Err(infer(
                "leaf VL embedding takes media via requests, not Batch.multimodal",
            ));
        }
        let max_length = crate::validate_loaded(&batch, &self.backend.config)?;
        let runtime = Arc::clone(&self.runtime);
        runtime
            .lock()
            .map_err(|_| infer("VL runtime lock is poisoned"))?
            .validate_requests(tokens, requests)?;
        if batch.pooled_indices.is_empty() && batch.raw_indices.is_empty() {
            return Ok(Inference::Ready(Embeddings::default()));
        }
        if !batch.raw_indices.is_empty() && !self.backend.raw_output_supported {
            return Err(infer(
                "This oxide kernel bundle does not support token outputs",
            ));
        }
        let hidden = self.backend.config.hidden_size();
        let output_width = self.backend.output_width;
        let geometry = crate::output::OutputGeometry::new(output_width, hidden)?;
        let elements = geometry.host_elements(&batch)?;
        crate::u32_size(elements)?;
        let failure = runtime
            .lock()
            .map_err(|_| infer("VL runtime lock is poisoned"))?
            .failure_handle(self.backend.engine.clone(), self.backend.io.clone());
        let fusion_failure = failure.clone();
        let owner_failure = failure.clone();
        let mut prep =
            Qwen35VlPrepGuard::new(Arc::clone(&runtime), failure, self.backend.stream.clone());
        let mut fusion = runtime
            .lock()
            .map_err(|_| infer("VL runtime lock is poisoned"))?
            .build_fusion(tokens, requests)?;
        fusion.set_runtime(Arc::clone(&runtime));
        fusion.set_failure(fusion_failure);
        prep.disarm();
        let memory = crate::result_memory::global_pool()?.reserve(crate::output::result_bytes(
            &batch,
            output_width,
            hidden,
        )?)?;
        let slot = self.backend.io.acquire()?;
        let mut lease = HostIoLease::new(
            slot,
            self.backend.io.clone(),
            self.backend.engine.clone(),
            self.backend.stream.clone(),
        );
        {
            let mut engine = self
                .backend
                .engine
                .lock()
                .map_err(|_| BackendError::Unhealthy)?;
            if engine.failed || self.backend.io.poisoned() {
                self.backend.io.close();
                return Err(BackendError::Unhealthy);
            }
            engine.enqueue(
                &batch,
                &self.backend.pool,
                max_length,
                None,
                &mut lease,
                Some(&fusion),
            )?;
        }
        let inner = if !batch.raw_indices.is_empty() {
            let pooled = batch.pooled_indices;
            let raw = batch.raw_indices;
            let boundaries = batch.cumulative_seq_lengths;
            Inference::Deferred(Box::new(move || {
                lease.wait()?;
                let output = lease.output(elements)?;
                if output.iter().any(|value| !value.is_finite()) {
                    lease.poison();
                    return Err(infer("Invalid oxide token output"));
                }
                let mut result =
                    crate::output::pack(output, output_width, hidden, &pooled, &raw, &boundaries)?;
                result.memory = Some(memory);
                Ok(result)
            }))
        } else {
            let pooled = batch.pooled_indices;
            Inference::Deferred(Box::new(move || {
                lease.wait()?;
                let output = lease.output(elements)?;
                if output.iter().any(|value| !value.is_finite()) {
                    lease.poison();
                    return Err(infer("Invalid oxide embedding output"));
                }
                let mut result = Embeddings::default();
                result.memory = Some(memory);
                for (&index, row) in pooled.iter().zip(output.chunks_exact(output_width)) {
                    result.insert(index as usize, Embedding::Pooled(row.to_vec()));
                }
                Ok(result)
            }))
        };
        let mut owner = Qwen35VlInference::new(
            inner,
            Some(fusion),
            Some(Arc::clone(&runtime)),
            Some(owner_failure),
            Some(self.backend.stream.clone()),
        );
        Ok(Inference::Deferred(Box::new(move || owner.complete())))
    }

    pub fn embed(
        &self,
        batch: Batch,
        requests: &[Qwen35VlRequest],
    ) -> Result<Embeddings, BackendError> {
        self.submit_embed(batch, requests)?.complete()
    }
}

impl Qwen35VlEngine {
    /// Causal Decision submission with the same fused prefill and external
    /// mRoPE. The original text engine keeps the readout and retirement path;
    /// this leaf only owns the feature/mRoPE request until completion.
    pub fn submit_decision(
        &self,
        batch: Batch,
        requests: &[Qwen35VlRequest],
        decisions: Vec<DecisionInput>,
    ) -> Result<Inference<Vec<DecisionOutput>>, BackendError> {
        let tokens = batch.input_ids.len();
        self.backend.ensure_open()?;
        if batch.multimodal.iter().any(Option::is_some) {
            return Err(infer(
                "leaf VL Decision takes media via requests, not Batch.multimodal",
            ));
        }
        self.backend.preflight_warm_decision(&batch, &decisions)?;
        let runtime = Arc::clone(&self.runtime);
        let (fusion, owner_failure) = {
            let guard = runtime
                .lock()
                .map_err(|_| infer("VL runtime lock is poisoned"))?;
            guard.validate_requests(tokens, requests)?;
            let failure =
                guard.failure_handle(self.backend.engine.clone(), self.backend.io.clone());
            let fusion_failure = failure.clone();
            let owner_failure = failure.clone();
            let mut prep =
                Qwen35VlPrepGuard::new(Arc::clone(&runtime), failure, self.backend.stream.clone());
            let mut fusion = guard.build_fusion(tokens, requests)?;
            fusion.set_runtime(Arc::clone(&runtime));
            fusion.set_failure(fusion_failure);
            prep.disarm();
            drop(guard);
            (fusion, owner_failure)
        };
        let inner = self
            .backend
            .submit_warm_decision_inner(batch, decisions, Some(&fusion))?;
        let mut owner = Qwen35VlInference::new(
            inner,
            Some(fusion),
            Some(Arc::clone(&runtime)),
            Some(owner_failure),
            Some(self.backend.stream.clone()),
        );
        Ok(Inference::Deferred(Box::new(move || owner.complete())))
    }

    pub fn decide(
        &self,
        batch: Batch,
        requests: &[Qwen35VlRequest],
        decisions: Vec<DecisionInput>,
    ) -> Result<Vec<DecisionOutput>, BackendError> {
        self.submit_decision(batch, requests, decisions)?.complete()
    }
}

/// Explicit dtype coverage, never inferred from a source-code branch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DtypeCoverage {
    pub reference_recipe: &'static str,
    pub device_recipe: &'static str,
    pub text_device_recipe: &'static str,
    pub f16_patch_stem: &'static str,
    pub f32_decay_parameters: &'static str,
}

pub const fn dtype_coverage() -> DtypeCoverage {
    DtypeCoverage {
        reference_recipe: "BF16-rounded CPU reference",
        device_recipe: "BF16 device only (LtPlan/FA4/device kernels)",
        text_device_recipe: "BF16 only",
        f16_patch_stem: "source structure only; not an F16 device or F16 reference recipe",
        f32_decay_parameters:
            "A_log/dt_bias preserve stored F32; all other F32 weights convert to BF16",
    }
}

/// One attention geometry required by a Qwen3.5-VL checkpoint.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen35VlModePlan {
    pub component: String,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub causal: bool,
    /// Central-registry mode requested by the actual Native geometry.
    pub expected_mode: Option<usize>,
    /// Mode actually selected from this branch's compiled registry.
    pub registry_mode: Option<usize>,
    pub note: String,
}

impl Qwen35VlModePlan {
    pub fn present(&self) -> bool {
        self.registry_mode.is_some()
    }
}

/// Cold whole-model run plan. It names the real text/vision/projection pieces
/// and every actual attention geometry; it never substitutes a private text
/// stand-in or borrows a different-head FA4 profile.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen35VlRunPlan {
    pub path: String,
    pub family: &'static str,
    pub dtype: String,
    pub vision_tensors: usize,
    pub text_hidden: usize,
    pub text_layers: usize,
    pub text_causal: bool,
    pub modes: Vec<Qwen35VlModePlan>,
    pub text_admission: String,
    pub moe_owner_required: bool,
    pub fp16_supported: bool,
    pub projection_score_route: String,
    pub gaps: Vec<String>,
    pub normal_factory_ready: bool,
}

pub(crate) fn expected_central_mode(
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    causal: bool,
    vision: bool,
) -> Option<usize> {
    if head_dim == 64 && heads == kv_heads && !causal && vision {
        return Some(0);
    }
    if head_dim == 72 && heads == kv_heads && !causal && vision {
        return Some(8);
    }
    if head_dim != 256 || kv_heads == 0 || !heads.is_multiple_of(kv_heads) {
        return None;
    }
    match (heads / kv_heads, causal) {
        (4, true) => Some(18),
        (4, false) => Some(6),
        (6, true) => Some(17),
        (6, false) => Some(16),
        (8, true) => Some(26),
        _ => None,
    }
}

fn mode_plan(
    component: &str,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    causal: bool,
    vision: bool,
) -> Qwen35VlModePlan {
    let spec = AttentionSpec {
        heads,
        kv_heads,
        head_dim,
        causal,
        window: Window::Global,
        dtype: Fa4Dtype::Bf16,
        scale: Scale::RsqrtHeadDim,
    };
    let registry_mode = Fa4::ratio_global_mode_for(&spec);
    let expected_mode = expected_central_mode(heads, kv_heads, head_dim, causal, vision);
    let note = match (expected_mode, registry_mode) {
        (Some(expected), Some(actual)) if expected == actual => "central mode present".to_owned(),
        (Some(expected), Some(actual)) => {
            format!("registry selected mode {actual} but central geometry is mode {expected}")
        }
        (Some(expected), None) => {
            format!("central append-only mode {expected} is not in this branch registry")
        }
        (None, Some(actual)) => format!(
            "no central mapping for real geometry; branch registry would select mode {actual}"
        ),
        (None, None) => "no central mapping and no branch profile for real geometry".to_owned(),
    };
    Qwen35VlModePlan {
        component: component.to_owned(),
        heads,
        kv_heads,
        head_dim,
        causal,
        expected_mode,
        registry_mode,
        note,
    }
}

impl Qwen35VlEngine {
    /// Pure cold plan over the actual config and checkpoint metadata. No CUDA
    /// context, weight payload, Torch, or device query is used.
    pub fn run_plan(path: &Path) -> Result<Qwen35VlRunPlan, BackendError> {
        let config = match Qwen35VlConfig::from_path(path) {
            Ok(config) => config,
            Err(error) if error.to_string().contains("processor_config") => {
                Qwen35VlConfig::from_config_path(path)?
            }
            Err(error) => return Err(error),
        };
        let report = validate_vision_inventory_path(&config, path)?;
        let text = &config.text;
        let family = if text.is_moe() { "moe" } else { "dense" };
        let block_bidirectional = crate::qwen35_serving::decision_attention(path)?
            == crate::qwen35_serving::DecisionAttention::BlockBidirectional;
        let text_causal = if block_bidirectional {
            false
        } else {
            crate::qwen35_serving::text_causal(path)?
        };
        let modes = vec![
            mode_plan(
                "vision",
                config.vision.num_heads,
                config.vision.num_heads,
                config.vision.head_dim(),
                false,
                true,
            ),
            mode_plan(
                if text_causal {
                    "text.full.causal"
                } else {
                    "text.full.noncausal"
                },
                text.num_attention_heads,
                text.num_key_value_heads,
                text.head_dim,
                text_causal,
                false,
            ),
        ];
        let mut gaps = Vec::new();
        for mode in &modes {
            let exact_central = match (mode.expected_mode, mode.registry_mode) {
                (Some(expected), Some(actual)) => expected == actual,
                _ => false,
            };
            if !exact_central {
                gaps.push(format!("{}: {}", mode.component, mode.note));
            }
        }
        let moe_owner_required = text.is_moe();
        let text_admission = if moe_owner_required {
            gaps.push(
                "MoE Qwen3.5 text execution must be supplied by the owning MoE engine".to_owned(),
            );
            "MoE owner required".to_owned()
        } else {
            match Qwen35Config::load(path).and_then(|config| {
                config.validate_candidate()?;
                Ok(())
            }) {
                Ok(()) => "dense text candidate admitted".to_owned(),
                Err(error) => {
                    let text = format!("full dense text admission pending: {error}");
                    gaps.push(text.clone());
                    text
                }
            }
        };
        let fp16_supported = false;
        if config.text.dtype != Dtype::Bf16 {
            gaps.push(format!(
                "dtype {} is not attached; BF16 device only until the Native exact FP16 recipe",
                config.text.dtype.name()
            ));
        }
        let normal_factory_ready = !moe_owner_required
            && config.text.dtype == Dtype::Bf16
            && text_admission == "dense text candidate admitted"
            && modes.iter().all(|mode| {
                mode.expected_mode.is_some() && mode.expected_mode == mode.registry_mode
            })
            && gaps.is_empty();
        Ok(Qwen35VlRunPlan {
            path: path.display().to_string(),
            family,
            dtype: config.text.dtype.name().to_owned(),
            vision_tensors: report.vision_tensors,
            text_hidden: text.hidden_size,
            text_layers: text.num_hidden_layers,
            text_causal,
            modes,
            text_admission,
            moe_owner_required,
            fp16_supported,
            projection_score_route:
                "OxideBackend pooled/raw output + warm OneJev Decision score; no Dense text projection stand-in"
                    .to_owned(),
            gaps,
            normal_factory_ready,
        })
    }

    /// Normal factory entry: `OxideBackend::new` selects the text route from
    /// the actual checkpoint, then the VL tower attaches on the same context.
    /// It refuses unless the cold run plan is fully ready.
    pub fn from_normal_factory(
        path: &Path,
        dtype: &str,
        model_type: infinity_v3_backend_core::ModelType,
        device_id: usize,
        options: Qwen35VlOptions,
    ) -> Result<Self, BackendError> {
        if dtype != "bfloat16" {
            return Err(start(
                "normal Qwen3.5-VL factory is BF16 only until the Native exact FP16 recipe",
            ));
        }
        let plan = Self::run_plan(path)?;
        if !plan.normal_factory_ready {
            return Err(start(format!(
                "normal Qwen3.5-VL factory plan is not ready: {}",
                plan.gaps.join("; ")
            )));
        }
        let backend = OxideBackend::new(path, dtype, model_type, device_id)?;
        Self::attach(backend, path, options)
    }

    /// Projection/pool readout alias for the normal embedding route.
    pub fn project(
        &self,
        batch: Batch,
        requests: &[Qwen35VlRequest],
    ) -> Result<Embeddings, BackendError> {
        self.embed(batch, requests)
    }

    /// Decision score alias for the same owning engine path.
    pub fn score(
        &self,
        batch: Batch,
        requests: &[Qwen35VlRequest],
        decisions: Vec<DecisionInput>,
    ) -> Result<Vec<DecisionOutput>, BackendError> {
        self.decide(batch, requests, decisions)
    }
}

/// Proof that the normal public factory (not a private constructor) accepts the
/// actual checkpoint and selects the expected text route before the VL tower
/// attaches. This is pure metadata and does not bind CUDA.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen35VlFactoryProof {
    pub is_qwen35_checkpoint: bool,
    pub serving_route: String,
    pub dtype_accepted: bool,
    pub run_plan_ready: bool,
    pub gaps: Vec<String>,
}

pub fn normal_factory_proof(
    path: &Path,
    dtype: &str,
    model_type: &infinity_v3_backend_core::ModelType,
) -> Result<Qwen35VlFactoryProof, BackendError> {
    let plan = Qwen35VlEngine::run_plan(path)?;
    let is_qwen35_checkpoint = crate::qwen35_serving::is_qwen35_checkpoint(path)?;
    let attention = crate::qwen35_serving::decision_attention(path)?;
    let route = crate::qwen35_serving::route(dtype, model_type, None, attention)?;
    let serving_route = match route {
        crate::qwen35_serving::ServingRoute::Text { causal, .. } => {
            format!("Text(causal={causal})")
        }
        crate::qwen35_serving::ServingRoute::TextDecision => "TextDecision".to_owned(),
        crate::qwen35_serving::ServingRoute::DynamicBlockDecision => {
            "DynamicBlockDecision".to_owned()
        }
    };
    let mut gaps = plan.gaps.clone();
    if !is_qwen35_checkpoint {
        gaps.push("normal factory does not detect a Qwen3.5 checkpoint".to_owned());
    }
    if dtype != "bfloat16" {
        gaps.push("normal factory VL route is BF16 only".to_owned());
    }
    Ok(Qwen35VlFactoryProof {
        is_qwen35_checkpoint,
        serving_route,
        dtype_accepted: dtype == "bfloat16",
        run_plan_ready: plan.normal_factory_ready,
        gaps,
    })
}

/// RAII owner for one deferred VL request. It is constructed BEFORE the outer
/// `Inference::Deferred` closure is returned. If the closure is never called,
/// `Drop` still drives the inner lease/fence to completion before the fusion
/// workspace is released. An uncertain fence quarantines the fusion buffers
/// instead of freeing unproven GPU state.
pub(crate) struct Qwen35VlInference<T: Send + 'static> {
    inner: Option<Inference<T>>,
    fusion: Option<Qwen35VlFusion>,
    runtime: Option<Arc<Mutex<Qwen35VlRuntime>>>,
    failure: Option<Qwen35VlFailure>,
    stream: Option<Arc<dyn Fence>>,
    completed: bool,
}

impl<T: Send + 'static> Qwen35VlInference<T> {
    pub(crate) fn new(
        inner: Inference<T>,
        fusion: Option<Qwen35VlFusion>,
        runtime: Option<Arc<Mutex<Qwen35VlRuntime>>>,
        failure: Option<Qwen35VlFailure>,
        stream: Option<Arc<dyn Fence>>,
    ) -> Self {
        Self {
            inner: Some(inner),
            fusion,
            runtime,
            failure,
            stream,
            completed: false,
        }
    }

    pub(crate) fn complete(&mut self) -> Result<T, BackendError> {
        if self.completed {
            return Err(infer("VL deferred request was already completed"));
        }
        let inner = self
            .inner
            .take()
            .ok_or_else(|| infer("VL deferred request has no inner"))?;
        let result = inner.complete();
        match result {
            Ok(value) => {
                if let Some(fusion) = self.fusion.as_mut() {
                    fusion.mark_completed();
                }
                self.runtime = None;
                self.completed = true;
                Ok(value)
            }
            Err(error) => {
                let fence_proven = self.stream.as_ref().is_some_and(|stream| stream.proven());
                if let Some(fusion) = self.fusion.take() {
                    if fence_proven {
                        drop(fusion);
                    } else {
                        // The lease/fence failed. Retain the feature/rope buffers
                        // rather than freeing GPU storage with unproven completion.
                        std::mem::forget(fusion);
                    }
                }
                if !fence_proven {
                    if let Some(failure) = self.failure.take() {
                        failure.fail_closed();
                    }
                }
                if let Some(runtime) = self.runtime.take() {
                    if fence_proven {
                        drop(runtime);
                    } else {
                        std::mem::forget(runtime);
                    }
                }
                self.completed = true;
                Err(error)
            }
        }
    }
}

impl<T: Send + 'static> Drop for Qwen35VlInference<T> {
    fn drop(&mut self) {
        if !self.completed {
            let _ = self.complete();
        }
    }
}

/// Shared failure handle. It never locks the model/runtime mutex, so destructors
/// can fail closed while a runtime guard is still held.
pub(crate) trait FailureSink: Send + Sync {
    fn close(&self);
}

struct BackendFailureSink {
    engine: Arc<Mutex<crate::Engine>>,
    io: Arc<SlotPool<HostIo>>,
}

impl FailureSink for BackendFailureSink {
    fn close(&self) {
        self.io.close();
        if let Ok(mut engine) = self.engine.lock() {
            engine.fail_closed();
        }
    }
}

#[derive(Clone)]
pub(crate) struct Qwen35VlFailure {
    failed: Arc<AtomicBool>,
    sink: Arc<dyn FailureSink>,
}

impl Qwen35VlFailure {
    pub(crate) fn new(failed: Arc<AtomicBool>, sink: Arc<dyn FailureSink>) -> Self {
        Self { failed, sink }
    }
    pub(crate) fn backend(
        failed: Arc<AtomicBool>,
        engine: Arc<Mutex<crate::Engine>>,
        io: Arc<SlotPool<HostIo>>,
    ) -> Self {
        Self::new(failed, Arc::new(BackendFailureSink { engine, io }))
    }
    pub(crate) fn fail_closed(&self) {
        self.failed.store(true, Ordering::Release);
        self.sink.close();
    }
}

/// Pre-owner guard constructed BEFORE any tower/rope H2D. It owns the shared
/// runtime/failure state, so a partial first-enqueue error with an unproven fence
/// quarantines the runtime and closes the pool without re-locking the runtime.
pub(crate) struct Qwen35VlPrepGuard {
    runtime: Option<Arc<Mutex<Qwen35VlRuntime>>>,
    failure: Qwen35VlFailure,
    stream: Arc<dyn Fence>,
    armed: bool,
}

impl Qwen35VlPrepGuard {
    pub(crate) fn new(
        runtime: Arc<Mutex<Qwen35VlRuntime>>,
        failure: Qwen35VlFailure,
        stream: Arc<dyn Fence>,
    ) -> Self {
        Self {
            runtime: Some(runtime),
            failure,
            stream,
            armed: true,
        }
    }
    pub(crate) fn disarm(&mut self) {
        self.armed = false;
    }

    #[cfg(test)]
    pub(crate) fn test_guard(failure: Qwen35VlFailure, stream: Arc<dyn Fence>) -> Self {
        Self {
            runtime: None,
            failure,
            stream,
            armed: true,
        }
    }
}

impl Drop for Qwen35VlPrepGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        if self.stream.proven() {
            return;
        }
        self.failure.fail_closed();
        if let Some(runtime) = self.runtime.take() {
            std::mem::forget(runtime);
        }
    }
}
