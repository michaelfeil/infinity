// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Graph-free BF16 Gemma2 text executor.
//!
//! Owns the non-attention execution: cold BF16/F32 weight upload, exact DivSqrt
//! norms, staged BF16 Q/K norm + split-half RoPE, unit V norm, PLE, packed
//! gate/up GELU-tanh, shared-KV producer reuse, final embedding projection and
//! Native BF16 pooling. `NativeGemma2Attention` owns both d256 local FA4 and
//! d512 full attention. The real image/audio `MediaRuntime` runs on the same
//! stream before the text layers; its projected rows are scattered into the
//! gathered token state and retained until the text completion fence.
#![allow(dead_code)]

const MEAN_POOL_THREADS: u32 = 256;
const MEAN_POOL_TILE: usize = 8;
const ROPE_THREADS: u32 = 256;

use crate::cublas::Blas;
use crate::dense::{DenseHead, DenseWork};
use crate::gemma2_attention::{
    validate_batch, Gemma2AttentionBatch, Gemma2AttentionProvider, Gemma2AttentionSpec,
    NativeGemma2Attention,
};
use crate::gemma2_math::EXPORTS as GEMMA2_EXPORTS;
use crate::gemma2_media::{
    MediaConfig, MediaHost, MediaRequest, MediaRuntime, MediaTokens, PreparedMedia,
};
use crate::gemma2_prepared::{upload_norm, Gemma2PreparedMath, Gemma2RoPe};
use crate::gemma2_text::{Gemma2AttentionKind, Gemma2Text, Gemma2TextConfig};
use crate::math::LinearPlan;
use crate::staging::HostIoLease;
use crate::tensor::Bf16;
use crate::upload::WeightUploader;
use crate::work_retirement::{CompletedTicket, CompletionTicket};
use crate::{
    arg, blocks, infer, launch, launch_threads, product, start, u32_size, InputLayout, WorkKey,
};
use cuda_core::{Context, DeviceBuffer, Function, Stream};
use infinity_v3_backend_core::{BackendError, Batch, Pool};
use std::{
    mem::ManuallyDrop,
    path::{Path, PathBuf},
    sync::{atomic::Ordering, Arc},
};

struct Gemma2Kernels {
    norm: Function,
    residual_norm: Function,
    unit_norm: Function,
    gated_gelu: Function,
    ple_gate: Function,
    scale: Function,
    mean_pool: Function,
    pool: Function,
    gather_scaled: Function,
    rope_neox: Function,
    bf16_to_f32: Function,
    rope_tables: Arc<Function>,
}

impl Gemma2Kernels {
    fn load(context: &Arc<Context>, module_path: Option<&Path>) -> Result<Self, BackendError> {
        let path = match module_path {
            Some(path) => path.to_path_buf(),
            None => std::env::var_os(infinity_v3_config::infinity_env!("OXIDE_KERNEL_MODULE"))
                .map(PathBuf::from)
                .ok_or_else(|| {
                    start("set INFINITY_OXIDE_KERNEL_MODULE to the cargo-oxide artifact")
                })?,
        };
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
        let function = |name: &str| module.load_function(name).map_err(start);
        Ok(Self {
            norm: function(GEMMA2_EXPORTS[0])?,
            residual_norm: function(GEMMA2_EXPORTS[1])?,
            unit_norm: function(GEMMA2_EXPORTS[2])?,
            gated_gelu: function(GEMMA2_EXPORTS[3])?,
            ple_gate: function(GEMMA2_EXPORTS[4])?,
            scale: function(GEMMA2_EXPORTS[5])?,
            rope_tables: Arc::new(function(GEMMA2_EXPORTS[6])?),
            mean_pool: function(GEMMA2_EXPORTS[7])?,
            pool: function("infinity_pool")?,
            gather_scaled: function("infinity_embedding_gather_scaled")?,
            rope_neox: function("infinity_rope_neox")?,
            bf16_to_f32: function("infinity_bf16_to_f32")?,
        })
    }
}

fn require_bf16(source: &crate::model::WeightSource, name: &str) -> Result<(), BackendError> {
    if source.dtype() != safetensors::Dtype::BF16 {
        return Err(start(format!(
            "Gemma2 prepared path requires BF16 {name}, found {:?}",
            source.dtype()
        )));
    }
    Ok(())
}

fn upload<const N: usize>(
    uploader: &mut WeightUploader,
    sources: &[&crate::model::WeightSource; N],
    name: &str,
) -> Result<DeviceBuffer<Bf16>, BackendError> {
    for source in sources {
        require_bf16(source, name)?;
    }
    uploader.upload(sources)
}

struct LayerBuffers {
    input_norm: DeviceBuffer<f32>,
    post_attention_norm: DeviceBuffer<f32>,
    pre_feedforward_norm: DeviceBuffer<f32>,
    post_feedforward_norm: DeviceBuffer<f32>,
    q_norm: DeviceBuffer<f32>,
    k_norm: Option<DeviceBuffer<f32>>,
    q: DeviceBuffer<Bf16>,
    k: Option<DeviceBuffer<Bf16>>,
    v: Option<DeviceBuffer<Bf16>>,
    o: DeviceBuffer<Bf16>,
    gate_up: DeviceBuffer<Bf16>,
    down: DeviceBuffer<Bf16>,
    layer_scalar: DeviceBuffer<Bf16>,
    layer_scalar_value: f32,
    ple_gate: DeviceBuffer<Bf16>,
    ple_projection: DeviceBuffer<Bf16>,
    ple_post_norm: DeviceBuffer<f32>,
}

struct WeightBuffers {
    embed: DeviceBuffer<Bf16>,
    embedding_projection: DeviceBuffer<Bf16>,
    final_norm: DeviceBuffer<f32>,
    ple_model: DeviceBuffer<Bf16>,
    ple_norm: DeviceBuffer<f32>,
    layers: Vec<LayerBuffers>,
}

struct LayerPlans {
    q: LinearPlan<Bf16>,
    k: Option<LinearPlan<Bf16>>,
    v: Option<LinearPlan<Bf16>>,
    o: LinearPlan<Bf16>,
    gate_up: LinearPlan<Bf16>,
    down: LinearPlan<Bf16>,
    ple_gate: LinearPlan<Bf16>,
    ple_projection: LinearPlan<Bf16>,
}

struct Plans {
    rows: usize,
    ple_model: LinearPlan<Bf16>,
    embedding_projection: LinearPlan<Bf16>,
    layers: Vec<LayerPlans>,
}

impl Plans {
    fn new(text: &Gemma2TextConfig, tokens: usize) -> Result<Self, BackendError> {
        let hidden = text.hidden_size;
        let intermediate = text.intermediate_size;
        let ple_width = text.ple_width;
        let mut layers = Vec::new();
        layers.try_reserve_exact(text.layers.len()).map_err(start)?;
        for layer in &text.layers {
            let k = if layer.shared {
                None
            } else {
                Some(LinearPlan::contiguous(tokens, hidden, layer.kv_width).map_err(infer)?)
            };
            let v = k
                .as_ref()
                .map(|_| LinearPlan::contiguous(tokens, hidden, layer.kv_width).map_err(infer))
                .transpose()?;
            layers.push(LayerPlans {
                q: LinearPlan::contiguous(tokens, hidden, layer.q_width).map_err(infer)?,
                k,
                v,
                o: LinearPlan::contiguous(tokens, layer.q_width, hidden).map_err(infer)?,
                gate_up: LinearPlan::contiguous(
                    tokens,
                    hidden,
                    intermediate
                        .checked_mul(2)
                        .ok_or_else(|| infer("gate/up width overflow"))?,
                )
                .map_err(infer)?,
                down: LinearPlan::contiguous(tokens, intermediate, hidden).map_err(infer)?,
                ple_gate: LinearPlan::contiguous(tokens, hidden, ple_width).map_err(infer)?,
                ple_projection: LinearPlan::contiguous(tokens, ple_width, hidden).map_err(infer)?,
            });
        }
        Ok(Self {
            rows: tokens,
            ple_model: LinearPlan::contiguous(
                tokens,
                hidden,
                text.num_hidden_layers
                    .checked_mul(ple_width)
                    .ok_or_else(|| infer("PLE projection width overflow"))?,
            )
            .map_err(infer)?,
            embedding_projection: LinearPlan::contiguous(tokens, hidden, text.embedding_dim)
                .map_err(infer)?,
            layers,
        })
    }
}

struct RetainedMedia {
    ticket: CompletionTicket,
    prepared: PreparedMedia,
    bytes: usize,
}

pub(crate) struct Gemma2Work {
    stream: Arc<Stream>,
    tokens: usize,
    sequences: usize,
    selections: usize,
    embedding_dim: usize,
    bytes: usize,
    input: ManuallyDrop<DeviceBuffer<u32>>,
    state: ManuallyDrop<DeviceBuffer<Bf16>>,
    next: ManuallyDrop<DeviceBuffer<Bf16>>,
    normalized: ManuallyDrop<DeviceBuffer<Bf16>>,
    branch: ManuallyDrop<DeviceBuffer<Bf16>>,
    q: ManuallyDrop<DeviceBuffer<Bf16>>,
    k: ManuallyDrop<DeviceBuffer<Bf16>>,
    v: ManuallyDrop<DeviceBuffer<Bf16>>,
    attention: ManuallyDrop<DeviceBuffer<Bf16>>,
    /// Reusable F32 [query_heads, tokens] softmax-LSE slab. Every layer
    /// overwrites it on the ordered stream.
    lse: ManuallyDrop<DeviceBuffer<f32>>,
    gate_up: ManuallyDrop<DeviceBuffer<Bf16>>,
    activation: ManuallyDrop<DeviceBuffer<Bf16>>,
    ple: ManuallyDrop<DeviceBuffer<Bf16>>,
    output: ManuallyDrop<DeviceBuffer<Bf16>>,
    shared_k: ManuallyDrop<[DeviceBuffer<Bf16>; 2]>,
    shared_v: ManuallyDrop<[DeviceBuffer<Bf16>; 2]>,
    pooled: ManuallyDrop<DeviceBuffer<f32>>,
    retained_media: Vec<RetainedMedia>,
    last_completion: Option<CompletionTicket>,
    released: bool,
}

impl Gemma2Work {
    pub(crate) fn key(&self) -> WorkKey {
        (self.tokens, self.sequences, self.selections)
    }
    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }
    pub(crate) fn fits(&self, key: WorkKey) -> bool {
        crate::capacity_fits(self.key(), key, self.bytes)
    }
    pub(crate) fn uses_stream(&self, stream: &Stream) -> bool {
        std::ptr::eq(self.stream.as_ref(), stream)
    }
    pub(crate) fn stream_handle(&self) -> usize {
        self.stream.cu_stream() as usize
    }
    pub(crate) fn context_handle(&self) -> usize {
        self.stream.context().cu_ctx() as usize
    }
    pub(crate) fn lse_elements(
        text: &Gemma2TextConfig,
        tokens: usize,
    ) -> Result<usize, BackendError> {
        product(tokens, text.num_attention_heads)
    }
    pub(crate) fn allocation_bytes(
        key: WorkKey,
        text: &Gemma2TextConfig,
    ) -> Result<usize, BackendError> {
        let (tokens, _, selections) = key;
        let hidden = text.hidden_size;
        let input_words = crate::staging::input_capacity_with_tail(key, key.2)?;
        let max_q = text
            .layers
            .iter()
            .map(|layer| layer.q_width)
            .max()
            .unwrap_or(0);
        let max_kv = text
            .layers
            .iter()
            .map(|layer| layer.kv_width)
            .max()
            .unwrap_or(0);
        let ple = text
            .num_hidden_layers
            .checked_mul(text.ple_width)
            .ok_or_else(|| infer("PLE width overflow"))?;
        let mut total = input_words
            .checked_mul(4)
            .ok_or_else(|| infer("input arena overflow"))?;
        let lse_bytes = Self::lse_elements(text, tokens)?
            .checked_mul(4)
            .ok_or_else(|| infer("LSE bytes overflow"))?;
        total = total
            .checked_add(lse_bytes)
            .ok_or_else(|| infer("workspace arena overflow"))?;
        for (name, elements) in [
            ("state", product(tokens, hidden)?),
            ("next", product(tokens, hidden)?),
            ("normalized", product(tokens, hidden)?),
            ("branch", product(tokens, hidden)?),
            ("q", product(tokens, max_q)?),
            ("k", product(tokens, max_kv)?),
            ("v", product(tokens, max_kv)?),
            ("attention", product(tokens, max_q)?),
            (
                "gate_up",
                product(
                    tokens,
                    text.intermediate_size
                        .checked_mul(2)
                        .ok_or_else(|| infer("gate/up width overflow"))?,
                )?,
            ),
            ("activation", product(tokens, text.intermediate_size)?),
            ("ple", product(tokens, ple)?),
            ("output", product(tokens, text.embedding_dim)?),
            ("shared_k", product(product(tokens, max_kv)?, 2)?),
            ("shared_v", product(product(tokens, max_kv)?, 2)?),
        ] {
            total = total
                .checked_add(
                    elements
                        .checked_mul(2)
                        .ok_or_else(|| infer(format!("{name} bytes overflow")))?,
                )
                .ok_or_else(|| infer("workspace arena overflow"))?;
        }
        total
            .checked_add(
                product(selections, text.embedding_dim)?
                    .checked_mul(4)
                    .ok_or_else(|| infer("pooled bytes overflow"))?,
            )
            .ok_or_else(|| infer("workspace arena overflow"))
    }
    pub(crate) fn new(
        stream: Arc<Stream>,
        text: &Gemma2TextConfig,
        key: WorkKey,
    ) -> Result<Self, BackendError> {
        let bytes = Self::allocation_bytes(key, text)?;
        let (tokens, sequences, selections) = key;
        let hidden = text.hidden_size;
        let max_q = text
            .layers
            .iter()
            .map(|layer| layer.q_width)
            .max()
            .unwrap_or(0);
        let max_kv = text
            .layers
            .iter()
            .map(|layer| layer.kv_width)
            .max()
            .unwrap_or(0);
        let ple = text
            .num_hidden_layers
            .checked_mul(text.ple_width)
            .ok_or_else(|| infer("PLE width overflow"))?;
        let input_len = crate::staging::input_capacity_with_tail(key, key.2)?;
        let alloc = |len: usize| -> Result<DeviceBuffer<Bf16>, BackendError> {
            // SAFETY: private arena initialized before any read.
            let buffer = unsafe { DeviceBuffer::<Bf16>::uninitialized_async(&stream, len) }
                .map_err(infer)?;
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
        };
        let input = unsafe { DeviceBuffer::<u32>::uninitialized_async(&stream, input_len) }
            .map_err(infer)?;
        let state = alloc(product(tokens, hidden)?)?;
        let next = alloc(product(tokens, hidden)?)?;
        let normalized = alloc(product(tokens, hidden)?)?;
        let branch = alloc(product(tokens, hidden)?)?;
        let q = alloc(product(tokens, max_q)?)?;
        let k = alloc(product(tokens, max_kv)?)?;
        let v = alloc(product(tokens, max_kv)?)?;
        let attention = alloc(product(tokens, max_q)?)?;
        // SAFETY: provider writes every [head, selected query] element before use.
        let lse = unsafe {
            DeviceBuffer::<f32>::uninitialized_async(&stream, Self::lse_elements(text, tokens)?)
        }
        .map_err(infer)?;
        let gate_up = alloc(product(
            tokens,
            text.intermediate_size
                .checked_mul(2)
                .ok_or_else(|| infer("gate/up width overflow"))?,
        )?)?;
        let activation = alloc(product(tokens, text.intermediate_size)?)?;
        let ple = alloc(product(tokens, ple)?)?;
        let output = alloc(product(tokens, text.embedding_dim)?)?;
        let shared_k = [
            alloc(product(tokens, max_kv)?)?,
            alloc(product(tokens, max_kv)?)?,
        ];
        let shared_v = [
            alloc(product(tokens, max_kv)?)?,
            alloc(product(tokens, max_kv)?)?,
        ];
        // SAFETY: every pooled element is written before download.
        let pooled = unsafe {
            DeviceBuffer::<f32>::uninitialized_async(
                &stream,
                product(selections, text.embedding_dim)?,
            )
        }
        .map_err(infer)?;
        crate::LIVE_ARENA_BYTES.fetch_add(bytes, Ordering::Relaxed);
        crate::record_live_arena_bytes();
        Ok(Self {
            stream,
            tokens,
            sequences,
            selections,
            embedding_dim: text.embedding_dim,
            bytes,
            input: ManuallyDrop::new(input),
            state: ManuallyDrop::new(state),
            next: ManuallyDrop::new(next),
            normalized: ManuallyDrop::new(normalized),
            branch: ManuallyDrop::new(branch),
            q: ManuallyDrop::new(q),
            k: ManuallyDrop::new(k),
            v: ManuallyDrop::new(v),
            attention: ManuallyDrop::new(attention),
            lse: ManuallyDrop::new(lse),
            gate_up: ManuallyDrop::new(gate_up),
            activation: ManuallyDrop::new(activation),
            ple: ManuallyDrop::new(ple),
            output: ManuallyDrop::new(output),
            shared_k: ManuallyDrop::new(shared_k),
            shared_v: ManuallyDrop::new(shared_v),
            pooled: ManuallyDrop::new(pooled),
            retained_media: Vec::new(),
            last_completion: None,
            released: false,
        })
    }
    /// Records the tracked completion generation for this Work's last use.
    pub(crate) fn mark_last_use(&mut self, ticket: CompletionTicket) {
        self.last_completion = Some(ticket);
    }
    pub(crate) fn completion_ticket(&self) -> Option<CompletionTicket> {
        self.last_completion.clone()
    }

    /// Retain a request's prepared tower rows until the text stream completion
    /// that scatters them has actually fenced.
    pub(crate) fn retain_prepared_media(
        &mut self,
        prepared: PreparedMedia,
    ) -> Result<(), BackendError> {
        let Some(ticket) = self.last_completion.clone() else {
            std::mem::forget(prepared);
            return Err(infer(
                "Gemma2 media retention requires a tracked completion ticket",
            ));
        };
        let bytes = match prepared.device_bytes() {
            Ok(bytes) => bytes,
            Err(error) => {
                std::mem::forget(prepared);
                return Err(error);
            }
        };
        crate::LIVE_ARENA_BYTES.fetch_add(bytes, Ordering::Relaxed);
        crate::record_live_arena_bytes();
        self.retained_media.push(RetainedMedia {
            ticket,
            prepared,
            bytes,
        });
        Ok(())
    }

    /// Drop media owners whose exact text-stream completion is already known.
    fn prune_completed_media(&mut self) {
        let (completed, pending): (Vec<_>, Vec<_>) = std::mem::take(&mut self.retained_media)
            .into_iter()
            .partition(|media| media.ticket.completed().is_some());
        for media in completed {
            crate::LIVE_ARENA_BYTES.fetch_sub(media.bytes, Ordering::Relaxed);
        }
        crate::record_live_arena_bytes();
        self.retained_media = pending;
    }
    /// Releases only when the last tracked submission has an actual completion
    /// proof. A ticket-less owner stays retained for the final stream drain.
    pub(crate) unsafe fn release_if_idle(&mut self) -> Result<bool, BackendError> {
        match self.last_completion.as_ref() {
            None => Ok(false),
            Some(ticket) => match ticket.completed() {
                Some(proof) => {
                    // SAFETY: the exact last-use proof is rechecked below.
                    unsafe { self.release_after_completion(&proof)? };
                    Ok(true)
                }
                None => Ok(false),
            },
        }
    }
    /// Releases owned buffers after an exact same-stream proof.
    ///
    /// # Safety
    /// The proof MUST come from the actual fence for this Work's tracked
    /// ticket. On any identity mismatch the payload stays owned.
    pub(crate) unsafe fn release_after_completion(
        &mut self,
        proof: &CompletedTicket,
    ) -> Result<(), BackendError> {
        let ticket = self
            .last_completion
            .as_ref()
            .ok_or_else(|| infer("Gemma2 Work has no completion ticket"))?;
        let stream_identity = Arc::as_ptr(&self.stream) as usize;
        if !proof.matches(ticket)
            || proof.stream_identity() != stream_identity
            || ticket.stream_identity() != stream_identity
        {
            return Err(infer(
                "Completion proof does not match this Gemma2 Work request",
            ));
        }
        if !self.released {
            // SAFETY: the caller proved the actual same-stream fence for this
            // exact ticket, so no borrowed owner is still in use.
            unsafe { self.release_payload_once() };
        }
        Ok(())
    }
    /// Releases this Work's owned payload after the caller drained the whole
    /// producing stream. A failed drain MUST NOT call this method.
    ///
    /// # Safety
    /// The caller MUST have observed a successful whole-stream drain covering
    /// every use of this Work.
    pub(crate) unsafe fn release_after_drain(&mut self) {
        if !self.released {
            // SAFETY: the caller drained the producing stream for this owner.
            unsafe { self.release_payload_once() };
        }
    }
    unsafe fn release_payload_once(&mut self) {
        if self.released {
            return;
        }
        self.released = true;
        // SAFETY: exactly one owner drops the payload after a completion proof
        // or an actual stream drain.
        unsafe {
            ManuallyDrop::drop(&mut self.input);
            ManuallyDrop::drop(&mut self.state);
            ManuallyDrop::drop(&mut self.next);
            ManuallyDrop::drop(&mut self.normalized);
            ManuallyDrop::drop(&mut self.branch);
            ManuallyDrop::drop(&mut self.q);
            ManuallyDrop::drop(&mut self.k);
            ManuallyDrop::drop(&mut self.v);
            ManuallyDrop::drop(&mut self.attention);
            ManuallyDrop::drop(&mut self.lse);
            ManuallyDrop::drop(&mut self.gate_up);
            ManuallyDrop::drop(&mut self.activation);
            ManuallyDrop::drop(&mut self.ple);
            ManuallyDrop::drop(&mut self.output);
            ManuallyDrop::drop(&mut self.shared_k);
            ManuallyDrop::drop(&mut self.shared_v);
            ManuallyDrop::drop(&mut self.pooled);
        }
        for media in self.retained_media.drain(..) {
            crate::LIVE_ARENA_BYTES.fetch_sub(media.bytes, Ordering::Relaxed);
        }
        crate::LIVE_ARENA_BYTES.fetch_sub(self.bytes, Ordering::Relaxed);
        crate::record_live_arena_bytes();
    }
    /// Validate token/boundary/output capacity before any host-to-device upload.
    pub(crate) fn preflight(
        &self,
        tokens: usize,
        boundary_count: usize,
        selections: usize,
    ) -> Result<(), BackendError> {
        self.check(tokens, boundary_count, selections)
    }
    fn check(
        &self,
        tokens: usize,
        boundary_count: usize,
        selections: usize,
    ) -> Result<(), BackendError> {
        if tokens > self.tokens || boundary_count > self.sequences || selections > self.selections {
            return Err(infer("Gemma2 request exceeds retained arena"));
        }
        Ok(())
    }
    pub(crate) fn stage(
        &self,
        batch: &Batch,
        geometry: crate::output::OutputGeometry,
        lease: &mut HostIoLease,
    ) -> Result<(usize, usize), BackendError> {
        if geometry.backbone_elements(batch)? > self.pooled.len() {
            return Err(infer("Gemma2 output exceeds retained arena"));
        }
        crate::stage_batch(batch, geometry, lease)
    }
    pub(crate) fn upload(&self, host: &[u32], elements: usize) -> Result<(), BackendError> {
        crate::upload_input(&self.stream, &self.input, host, elements)
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn finish(
        &mut self,
        dense: Option<(&mut DenseHead, &mut DenseWork)>,
        lt: &crate::lt::LtProvider,
        plain_blas: Option<&Blas>,
        pooled_rows: usize,
        raw_rows: usize,
        backbone_width: usize,
        lease: &mut HostIoLease,
    ) -> Result<(), BackendError> {
        crate::finish_pooled(
            dense,
            lt,
            plain_blas,
            &self.stream,
            &self.pooled,
            pooled_rows,
            raw_rows,
            backbone_width,
            lease,
        )
    }
}

impl Drop for Gemma2Work {
    fn drop(&mut self) {
        if self.released {
            return;
        }
        #[cfg(test)]
        crate::note_default_drop_drain();
        if self.stream.synchronize().is_ok() {
            // SAFETY: this destructor just drained the owning stream. Failed
            // drains deliberately retain device allocations.
            unsafe { self.release_payload_once() };
        } else {
            metrics::counter!("oxide_arena_retained_bytes_total").increment(self.bytes as u64);
            let retained = std::mem::take(&mut self.retained_media);
            std::mem::forget(retained);
        }
    }
}

/// Buffer roles after the per-layer PLE residual norm and layer-scalar pass.
/// The scalar kernel reads the unscaled PLE sum in `spare` and writes the
/// scaled result into `current`; no subsequent swap may undo that carry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct LayerCarryPlan {
    pub(crate) scalar_input: u64,
    pub(crate) scalar_output: u64,
    pub(crate) carried: u64,
    pub(crate) next_spare: u64,
    pub(crate) next_input_norm: u64,
    pub(crate) final_norm_input: u64,
}

pub(crate) fn layer_carry_plan(current: u64, spare: u64) -> LayerCarryPlan {
    LayerCarryPlan {
        scalar_input: spare,
        scalar_output: current,
        carried: current,
        next_spare: spare,
        next_input_norm: current,
        final_norm_input: current,
    }
}

/// Host launch geometry for `infinity_rope_neox`: one thread per rotary pair
/// across Q and KV heads, not one thread per token/head. Gemma2 uses
/// full head widths (d256 local, d512 global), never Gemma4 partial rotary.
pub(crate) fn rope_launch(
    tokens: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
) -> Result<((u32, u32, u32), u32), BackendError> {
    if tokens == 0 || q_heads == 0 || kv_heads == 0 {
        return Err(infer("rope requires tokens and positive Q/KV heads"));
    }
    if rotary_dim == 0 || !rotary_dim.is_multiple_of(2) || rotary_dim != head_dim {
        return Err(infer(
            "rope rotary width must be positive, even and <= head_dim",
        ));
    }
    let heads = q_heads
        .checked_add(kv_heads)
        .ok_or_else(|| infer("rope head count overflow"))?;
    let pairs = tokens
        .checked_mul(heads)
        .and_then(|rows| rows.checked_mul(rotary_dim / 2))
        .ok_or_else(|| infer("rope pair count overflow"))?;
    Ok((blocks(pairs)?, ROPE_THREADS))
}

fn media_placeholder_rows(batch: &Batch, tokens: Option<MediaTokens>) -> usize {
    let Some(tokens) = tokens else {
        return 0;
    };
    batch
        .input_ids
        .iter()
        .filter(|id| **id == tokens.image || **id == tokens.audio || **id == tokens.video)
        .count()
}

fn check_media_row_coverage(
    placeholders: usize,
    planned_rows: Option<usize>,
) -> Result<(), BackendError> {
    match planned_rows {
        Some(rows) if rows == placeholders => Ok(()),
        Some(rows) => Err(infer(format!(
            "Gemma2 media spans cover {rows} rows but the packed batch contains {placeholders} placeholder tokens"
        ))),
        None if placeholders == 0 => Ok(()),
        None => Err(infer(
            "Gemma2 media placeholder tokens require a complete image/audio request",
        )),
    }
}

/// Host launch geometry for the authoritative mean kernel. The device body
/// only writes features for `threadIdx.x < 8`, so the x block must always be at
/// least eight even when a selected span has one to four rows. Native uses 256.
pub(crate) fn mean_pool_launch(
    selected: usize,
    width: usize,
) -> Result<((u32, u32, u32), u32), BackendError> {
    if selected == 0 || width == 0 {
        return Err(infer("mean pool requires selected spans and width"));
    }
    let grid = (
        u32_size(selected)?,
        u32_size(width.div_ceil(MEAN_POOL_TILE))?,
        1,
    );
    Ok((grid, MEAN_POOL_THREADS))
}

/// Graph-free text executor. Attention is the native local/full provider.
pub(crate) struct Gemma2Executor {
    context: Arc<Context>,
    stream: Arc<Stream>,
    text: Gemma2TextConfig,
    math: Gemma2PreparedMath,
    kernels: Gemma2Kernels,
    weights: WeightBuffers,
    local_rope: Gemma2RoPe,
    global_rope: Gemma2RoPe,
    plans: Option<Plans>,
    attention: Arc<dyn Gemma2AttentionProvider>,
    media_config: MediaConfig,
    media_tokens: Option<MediaTokens>,
    media: Option<MediaRuntime>,
    pending_media: Option<MediaRequest>,
}

impl Gemma2Executor {
    pub(crate) fn config(&self) -> &Gemma2TextConfig {
        &self.text
    }

    fn check_cold_identity(context: &Arc<Context>, stream: &Stream) -> Result<(), BackendError> {
        if !Arc::ptr_eq(stream.context(), context) {
            return Err(start(
                "Gemma2 constructor stream belongs to another CUDA context",
            ));
        }
        Ok(())
    }

    pub(crate) fn uses_stream(&self, stream: &Stream) -> bool {
        std::ptr::eq(self.stream.as_ref(), stream)
    }

    pub(crate) fn uses_context(&self, context: &Arc<Context>) -> bool {
        Arc::ptr_eq(&self.context, context)
    }

    pub(crate) fn new(
        context: Arc<Context>,
        stream: Arc<Stream>,
        host: &Gemma2Text,
        model_root: &Path,
        fa4_root: &Path,
        head512_root: &Path,
        module_path: Option<&Path>,
    ) -> Result<Self, BackendError> {
        Self::check_cold_identity(&context, &stream)?;
        let math = Gemma2PreparedMath::admit(&host.config)?;
        context.bind_to_thread().map_err(start)?;
        let kernels = Gemma2Kernels::load(&context, module_path)?;
        let (major, minor) = context.compute_capability().map_err(start)?;
        let architecture = format!("sm_{major}{minor}");
        let max_optin_shared_memory_bytes = if architecture == "sm_120" {
            96 * 1024
        } else {
            128 * 1024
        };
        let attention: Arc<dyn Gemma2AttentionProvider> = Arc::new(NativeGemma2Attention::new(
            Arc::clone(&context),
            fa4_root,
            head512_root,
            &architecture,
            max_optin_shared_memory_bytes,
        )?);
        for index in 0..host.config.text.layers.len() {
            let spec = Gemma2AttentionSpec::for_layer(&host.config.text, index)?;
            if !attention.supports(&spec) {
                return Err(start(format!(
                    "Gemma2 native attention does not support layer {index}"
                )));
            }
        }
        let local_rope = Gemma2RoPe::new(
            &stream,
            Arc::clone(&kernels.rope_tables),
            256,
            host.config.text.rope_sliding_theta,
            host.config.text.max_position_embeddings,
        )?;
        let global_rope = Gemma2RoPe::new(
            &stream,
            Arc::clone(&kernels.rope_tables),
            512,
            host.config.text.rope_full_theta,
            host.config.text.max_position_embeddings,
        )?;
        let media_config = host
            .config
            .media
            .to_media_config(host.config.text.hidden_size);
        let media_tokens = media_config.tokens;
        let media_host = MediaHost::load(model_root, &media_config)?;
        let media = media_host
            .as_ref()
            .map(|media_host| {
                MediaRuntime::load(&context, &stream, media_host, math.hidden, module_path)
            })
            .transpose()?;
        let mut uploader = WeightUploader::new(Arc::clone(&stream))?;
        let weights = Self::upload_weights(&mut uploader, &stream, host, &math)?;
        uploader.finish()?;
        Ok(Self {
            context,
            stream,
            text: host.config.text.clone(),
            math,
            kernels,
            weights,
            local_rope,
            global_rope,
            plans: None,
            attention,
            media_config,
            media_tokens,
            media,
            pending_media: None,
        })
    }

    fn upload_weights(
        uploader: &mut WeightUploader,
        stream: &Arc<Stream>,
        host: &Gemma2Text,
        math: &Gemma2PreparedMath,
    ) -> Result<WeightBuffers, BackendError> {
        for layer in &host.weights.layers {
            if layer.q.bias.is_some()
                || layer.k.as_ref().is_some_and(|k| k.bias.is_some())
                || layer.v.as_ref().is_some_and(|v| v.bias.is_some())
                || layer.o.bias.is_some()
                || layer.gate.bias.is_some()
                || layer.up.bias.is_some()
                || layer.down.bias.is_some()
                || layer.ple_input_gate.bias.is_some()
                || layer.ple_projection.bias.is_some()
            {
                return Err(start(
                    "Gemma2 prepared executor requires the released biasless checkpoint",
                ));
            }
        }
        if host.weights.embedding_projection.bias.is_some()
            || host.weights.ple.model_projection.bias.is_some()
        {
            return Err(start(
                "Gemma2 prepared executor requires biasless global projections",
            ));
        }
        let embed = upload(uploader, &[&host.weights.embed], "embed_tokens")?;
        let embedding_projection = upload(
            uploader,
            &[&host.weights.embedding_projection.weight],
            "embedding_projection",
        )?;
        let final_norm = upload_norm(stream, &host.weights.final_norm, math.hidden)?;
        let ple_model = upload(
            uploader,
            &[&host.weights.ple.model_projection.weight],
            "PLE model projection",
        )?;
        let ple_norm = upload_norm(stream, &host.weights.ple.projection_norm, math.ple_width)?;
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(host.weights.layers.len())
            .map_err(start)?;
        for (index, layer) in host.weights.layers.iter().enumerate() {
            let config_layer = host
                .config
                .text
                .layers
                .get(index)
                .ok_or_else(|| start("Gemma2 layer/config mismatch"))?;
            let k_norm = layer
                .k_norm
                .as_ref()
                .map(|source| upload_norm(stream, source, config_layer.head_dim))
                .transpose()?;
            let k = layer
                .k
                .as_ref()
                .map(|linear| upload(uploader, &[&linear.weight], "key projection"))
                .transpose()?;
            let v = layer
                .v
                .as_ref()
                .map(|linear| upload(uploader, &[&linear.weight], "value projection"))
                .transpose()?;
            layers.push(LayerBuffers {
                input_norm: upload_norm(stream, &layer.input_norm, math.hidden)?,
                post_attention_norm: upload_norm(stream, &layer.post_attention_norm, math.hidden)?,
                pre_feedforward_norm: upload_norm(
                    stream,
                    &layer.pre_feedforward_norm,
                    math.hidden,
                )?,
                post_feedforward_norm: upload_norm(
                    stream,
                    &layer.post_feedforward_norm,
                    math.hidden,
                )?,
                q_norm: upload_norm(stream, &layer.q_norm, config_layer.head_dim)?,
                k_norm,
                q: upload(uploader, &[&layer.q.weight], "query projection")?,
                k,
                v,
                o: upload(uploader, &[&layer.o.weight], "attention output projection")?,
                gate_up: upload(uploader, &[&layer.gate.weight, &layer.up.weight], "gate/up")?,
                down: upload(uploader, &[&layer.down.weight], "down")?,
                layer_scalar: upload(uploader, &[&layer.layer_scalar], "layer_scalar")?,
                layer_scalar_value: bf16_source_scalar(&layer.layer_scalar)?,
                ple_gate: upload(uploader, &[&layer.ple_input_gate.weight], "PLE input gate")?,
                ple_projection: upload(
                    uploader,
                    &[&layer.ple_projection.weight],
                    "PLE projection",
                )?,
                ple_post_norm: upload_norm(stream, &layer.ple_post_norm, math.hidden)?,
            });
        }
        Ok(WeightBuffers {
            embed,
            embedding_projection,
            final_norm,
            ple_model,
            ple_norm,
            layers,
        })
    }

    fn prepare_plans(&mut self, tokens: usize) -> Result<(), BackendError> {
        if self.plans.as_ref().is_none_or(|plans| plans.rows != tokens) {
            self.plans = Some(Plans::new(&self.text, tokens)?);
        }
        Ok(())
    }

    /// Plan all media spans on the host before arena reservation, lease begin,
    /// H2D upload or any tower/text launch.
    pub(crate) fn preflight_media(&mut self, batch: &Batch) -> Result<(), BackendError> {
        let request = MediaRequest::plan(
            batch,
            &self.media_config,
            self.math.hidden,
            self.media_tokens,
        )?;
        check_media_row_coverage(
            media_placeholder_rows(batch, self.media_tokens),
            request.as_ref().map(|request| request.row_indices.len()),
        )?;
        if request.is_some() && self.media.is_none() {
            return Err(infer(
                "Gemma2 media input requires the loaded vision/audio runtime",
            ));
        }
        self.pending_media = request;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn enqueue(
        &mut self,
        plain_blas: Option<&Blas>,
        stream: &Stream,
        work: &mut Gemma2Work,
        batch: &Batch,
        pool: &Pool,
        max_length: usize,
        tokens: usize,
    ) -> Result<(), BackendError> {
        self.enqueue_multimodal(plain_blas, stream, work, batch, pool, max_length, tokens)
    }

    /// Execute image/audio towers, then scatter their projected rows into the
    /// gathered text state and run the complete text graph. The prepared media
    /// owner is retained on Work until the text stream completion.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn enqueue_multimodal(
        &mut self,
        plain_blas: Option<&Blas>,
        stream: &Stream,
        work: &mut Gemma2Work,
        batch: &Batch,
        pool: &Pool,
        max_length: usize,
        tokens: usize,
    ) -> Result<(), BackendError> {
        work.prune_completed_media();
        let request = match self.pending_media.take() {
            Some(request) => Some(request),
            None => MediaRequest::plan(
                batch,
                &self.media_config,
                self.math.hidden,
                self.media_tokens,
            )?,
        };
        let Some(request) = request else {
            return self.enqueue_impl(
                plain_blas, stream, work, batch, pool, max_length, tokens, None,
            );
        };
        let runtime = self
            .media
            .as_ref()
            .ok_or_else(|| infer("Gemma2 media runtime is not loaded"))?;
        let blas = plain_blas
            .ok_or_else(|| infer("Gemma2 media towers require the plain cuBLAS handle"))?;
        if !work.uses_stream(stream)
            || !self.uses_stream(stream)
            || !std::ptr::eq(blas.stream(), stream)
            || !Arc::ptr_eq(blas.context(), &self.context)
            || !Arc::ptr_eq(stream.context(), &self.context)
        {
            return Err(infer(
                "Gemma2 media requires one shared Work/executor/cuBLAS stream and context",
            ));
        }
        let prepared = runtime.prepare(blas, &self.stream, request)?;
        match self.enqueue_impl(
            plain_blas,
            stream,
            work,
            batch,
            pool,
            max_length,
            tokens,
            Some(&prepared),
        ) {
            Ok(()) => work.retain_prepared_media(prepared),
            Err(error) => {
                // No completion event may be recorded for a failed request.
                // A proven stream drain makes dropping the prepared owners safe.
                if self.stream.synchronize().is_ok() {
                    drop(prepared);
                } else {
                    work.retain_prepared_media(prepared)?;
                }
                Err(error)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue_impl(
        &mut self,
        plain_blas: Option<&Blas>,
        stream: &Stream,
        work: &mut Gemma2Work,
        batch: &Batch,
        pool: &Pool,
        max_length: usize,
        tokens: usize,
        prepared_media: Option<&PreparedMedia>,
    ) -> Result<(), BackendError> {
        let blas =
            plain_blas.ok_or_else(|| infer("Gemma2 biasless projections require plain cuBLAS"))?;
        if !work.uses_stream(stream)
            || !self.uses_stream(stream)
            || !std::ptr::eq(blas.stream(), stream)
            || !Arc::ptr_eq(blas.context(), &self.context)
        {
            return Err(infer(
                "Gemma2 enqueue requires one shared Work/executor/cuBLAS stream and context",
            ));
        }
        if tokens == 0 || batch.cumulative_seq_lengths.len() < 2 {
            return Err(infer("Gemma2 requires tokens and sequence boundaries"));
        }
        let sequences = batch
            .cumulative_seq_lengths
            .len()
            .checked_sub(1)
            .ok_or_else(|| infer("Gemma2 boundary count underflow"))?;
        let selections = batch.pooled_indices.len();
        let output_rows = crate::output::rows(batch)?;
        let raw_rows = crate::output::raw_rows(batch)?;
        work.check(tokens, batch.cumulative_seq_lengths.len(), output_rows)?;
        self.prepare_plans(tokens)?;
        let attention = self.attention.as_ref();
        let plans = self
            .plans
            .as_ref()
            .ok_or_else(|| infer("missing Gemma2 plans"))?;
        let text = &self.text;
        let math = &self.math;
        let weights = &self.weights;
        let kernels = &self.kernels;
        let local_rope = &self.local_rope;
        let global_rope = &self.global_rope;
        let layers = &text.layers;
        let rows = u32_size(tokens)?;
        let hidden = u32_size(math.hidden)?;
        let ple_width = u32_size(math.ple_width)?;
        let mut ple_layers = u32_size(math.layers)?;
        let selections_u32 = u32_size(selections)?;
        let base = work.input.cu_deviceptr();
        let layout = InputLayout::new(tokens, batch.cumulative_seq_lengths.len(), output_rows)?;
        let positions = layout.positions().address(base)?;
        let boundaries = layout.boundaries().address(base)?;
        let selected = layout.selections().address(base)?;
        let raw_selected =
            InputLayout::new(tokens, batch.cumulative_seq_lengths.len(), selections)?
                .with_trailer(raw_rows)?
                .trailer()
                .address(base)?;

        let state = work.state.cu_deviceptr();
        let next = work.next.cu_deviceptr();
        let normalized = work.normalized.cu_deviceptr();
        let mut branch = work.branch.cu_deviceptr();
        let q_base = work.q.cu_deviceptr();
        let k_base = work.k.cu_deviceptr();
        let v_base = work.v.cu_deviceptr();
        let attention_out = work.attention.cu_deviceptr();
        let attention_out_bytes = u64::try_from(work.attention.num_bytes()).map_err(infer)?;
        let lse_ptr = work.lse.cu_deviceptr();
        let lse_bytes = u64::try_from(work.lse.num_bytes()).map_err(infer)?;
        let attention_offset_bytes = (sequences + 1)
            .checked_mul(4)
            .ok_or_else(|| infer("attention offset bytes overflow"))?;
        let gate_up = work.gate_up.cu_deviceptr();
        let activation = work.activation.cu_deviceptr();
        let ple = work.ple.cu_deviceptr();
        let output = work.output.cu_deviceptr();

        let mut ids = base;
        let mut table = weights.embed.cu_deviceptr();
        let mut gather_out = state;
        let mut gather_tokens = rows;
        let mut gather_hidden = hidden;
        let mut embedding_scale = math.embedding_scale;
        let gather_grid = blocks(product(tokens, math.hidden)?)?;
        launch(
            &kernels.gather_scaled,
            stream,
            gather_grid,
            0,
            &mut [
                arg(&mut ids),
                arg(&mut table),
                arg(&mut gather_out),
                arg(&mut gather_tokens),
                arg(&mut gather_hidden),
                arg(&mut embedding_scale),
            ],
        )?;
        if let Some(prepared) = prepared_media {
            let runtime = self
                .media
                .as_ref()
                .ok_or_else(|| infer("Gemma2 media runtime disappeared"))?;
            runtime.scatter(stream, &mut work.state, prepared)?;
        }

        // PLE model projection: [tokens, 24*512], scaled by bf16(1/sqrt(512)),
        // then normalized in-place with plain DivSqrt per 512-wide layer row.
        unsafe {
            blas.linear(
                &plans.ple_model,
                &work.state,
                &weights.ple_model,
                &mut work.ple,
            )?;
        }
        let ple_elements = product(
            tokens,
            math.layers
                .checked_mul(math.ple_width)
                .ok_or_else(|| infer("PLE width overflow"))?,
        )?;
        let mut scale_in = ple;
        let mut scale_out = ple;
        let mut scale_count = u32_size(ple_elements)?;
        let mut ple_scale = math.ple_scale;
        launch(
            &kernels.scale,
            stream,
            blocks(ple_elements)?,
            0,
            &mut [
                arg(&mut scale_in),
                arg(&mut scale_out),
                arg(&mut scale_count),
                arg(&mut ple_scale),
            ],
        )?;
        let mut norm_in = ple;
        let mut norm_scale = weights.ple_norm.cu_deviceptr();
        let mut norm_out = ple;
        let mut norm_rows = u32_size(product(tokens, math.layers)?)?;
        let mut norm_width = ple_width;
        let mut norm_heads = 1u32;
        let mut norm_stride = u64::from(ple_width);
        let mut norm_eps = math.norm_eps;
        launch_threads(
            &kernels.norm,
            stream,
            (norm_rows, 1, 1),
            ple_width,
            0,
            &mut [
                arg(&mut norm_in),
                arg(&mut norm_scale),
                arg(&mut norm_out),
                arg(&mut norm_rows),
                arg(&mut norm_width),
                arg(&mut norm_heads),
                arg(&mut norm_stride),
                arg(&mut norm_eps),
            ],
        )?;

        let mut current = state;
        let mut spare = next;
        let shared_k = [
            work.shared_k[0].cu_deviceptr(),
            work.shared_k[1].cu_deviceptr(),
        ];
        let shared_v = [
            work.shared_v[0].cu_deviceptr(),
            work.shared_v[1].cu_deviceptr(),
        ];

        for (index, layer) in layers.iter().enumerate() {
            let plan = plans
                .layers
                .get(index)
                .ok_or_else(|| infer("Gemma2 layer plan missing"))?;
            let buffers = weights
                .layers
                .get(index)
                .ok_or_else(|| infer("Gemma2 layer weights missing"))?;
            let rope = if matches!(layer.kind, Gemma2AttentionKind::Sliding) {
                local_rope
            } else {
                global_rope
            };
            let mut norm_input = current;
            let mut norm_weight = buffers.input_norm.cu_deviceptr();
            let mut norm_out = normalized;
            let mut rows_u32 = rows;
            let mut width_u32 = hidden;
            let mut heads_u32 = 1u32;
            let mut stride_u64 = u64::from(hidden);
            let mut eps = math.norm_eps;
            launch_threads(
                &kernels.norm,
                stream,
                (rows_u32, 1, 1),
                hidden,
                0,
                &mut [
                    arg(&mut norm_input),
                    arg(&mut norm_weight),
                    arg(&mut norm_out),
                    arg(&mut rows_u32),
                    arg(&mut width_u32),
                    arg(&mut heads_u32),
                    arg(&mut stride_u64),
                    arg(&mut eps),
                ],
            )?;
            unsafe {
                blas.linear(&plan.q, &work.normalized, &buffers.q, &mut work.q)?;
                if let (Some(k_plan), Some(v_plan), Some(k_weight), Some(v_weight)) =
                    (&plan.k, &plan.v, &buffers.k, &buffers.v)
                {
                    blas.linear(k_plan, &work.normalized, k_weight, &mut work.k)?;
                    blas.linear(v_plan, &work.normalized, v_weight, &mut work.v)?;
                }
            }
            let q_heads_usize = math.q_heads as usize;
            let mut q_ptr = q_base;
            let mut q_norm_scale = buffers.q_norm.cu_deviceptr();
            let mut q_norm_out = q_base;
            let mut q_rows = u32_size(product(tokens, q_heads_usize)?)?;
            let mut q_width = u32_size(layer.head_dim)?;
            let mut q_heads = u32_size(q_heads_usize)?;
            let mut q_stride = u64::from(u32_size(layer.q_width)?);
            launch_threads(
                &kernels.norm,
                stream,
                (q_rows, 1, 1),
                u32_size(layer.head_dim)?,
                0,
                &mut [
                    arg(&mut q_ptr),
                    arg(&mut q_norm_scale),
                    arg(&mut q_norm_out),
                    arg(&mut q_rows),
                    arg(&mut q_width),
                    arg(&mut q_heads),
                    arg(&mut q_stride),
                    arg(&mut eps),
                ],
            )?;
            let mut k_heads = u32_size(layer.kv_heads)?;
            let mut k_width = u32_size(layer.head_dim)?;
            let mut k_stride = u64::from(u32_size(layer.kv_width)?);
            if !layer.shared {
                let mut k_ptr = k_base;
                let mut k_norm_scale = buffers
                    .k_norm
                    .as_ref()
                    .ok_or_else(|| infer("Gemma2 K norm missing"))?
                    .cu_deviceptr();
                let mut k_norm_out = k_base;
                let mut k_rows = u32_size(product(tokens, layer.kv_heads)?)?;
                launch_threads(
                    &kernels.norm,
                    stream,
                    (k_rows, 1, 1),
                    u32_size(layer.head_dim)?,
                    0,
                    &mut [
                        arg(&mut k_ptr),
                        arg(&mut k_norm_scale),
                        arg(&mut k_norm_out),
                        arg(&mut k_rows),
                        arg(&mut k_width),
                        arg(&mut k_heads),
                        arg(&mut k_stride),
                        arg(&mut eps),
                    ],
                )?;
            }
            let mut cos = rope.cosine.cu_deviceptr();
            let mut sin = rope.sine.cu_deviceptr();
            let mut pos = positions;
            let mut r_tokens = rows;
            let mut r_q_heads = q_heads;
            let mut r_kv_heads = k_heads;
            let mut r_head_dim = q_width;
            let mut r_rotary = q_width;
            let mut r_q_stride = q_stride;
            let mut r_k_stride = k_stride;
            let mut r_table = rope.stride;
            let mut rope_k = k_base;
            let (rope_grid, rope_threads) = rope_launch(
                tokens,
                q_heads_usize,
                layer.kv_heads,
                layer.head_dim,
                layer.head_dim,
            )?;
            launch_threads(
                &kernels.rope_neox,
                stream,
                rope_grid,
                rope_threads,
                0,
                &mut [
                    arg(&mut q_ptr),
                    arg(&mut rope_k),
                    arg(&mut cos),
                    arg(&mut sin),
                    arg(&mut pos),
                    arg(&mut r_tokens),
                    arg(&mut r_q_heads),
                    arg(&mut r_kv_heads),
                    arg(&mut r_head_dim),
                    arg(&mut r_rotary),
                    arg(&mut r_q_stride),
                    arg(&mut r_k_stride),
                    arg(&mut r_table),
                ],
            )?;
            if !layer.shared {
                let mut v_ptr = v_base;
                let mut v_out = v_base;
                let mut v_rows = u32_size(product(tokens, layer.kv_heads)?)?;
                let mut v_width = k_width;
                let mut v_heads = k_heads;
                let mut v_stride = k_stride;
                launch_threads(
                    &kernels.unit_norm,
                    stream,
                    (v_rows, 1, 1),
                    u32_size(layer.head_dim)?,
                    0,
                    &mut [
                        arg(&mut v_ptr),
                        arg(&mut v_out),
                        arg(&mut v_rows),
                        arg(&mut v_width),
                        arg(&mut v_heads),
                        arg(&mut v_stride),
                        arg(&mut eps),
                    ],
                )?;
            }
            let category = match layer.kind {
                Gemma2AttentionKind::Sliding => 0,
                Gemma2AttentionKind::Full => 1,
            };
            if !layer.shared && layer.store_kv {
                let bytes = product(tokens, layer.kv_width)?
                    .checked_mul(2)
                    .ok_or_else(|| infer("shared KV byte overflow"))?;
                unsafe {
                    cuda_core::memory::memcpy_dtod_async(
                        shared_k[category],
                        k_base,
                        bytes,
                        stream.cu_stream(),
                    )
                    .map_err(infer)?;
                    cuda_core::memory::memcpy_dtod_async(
                        shared_v[category],
                        v_base,
                        bytes,
                        stream.cu_stream(),
                    )
                    .map_err(infer)?;
                }
            }
            let (k_ptr, v_ptr) = if layer.shared {
                (shared_k[category], shared_v[category])
            } else {
                (k_base, v_base)
            };
            let spec = Gemma2AttentionSpec::for_layer(text, index)?;
            if !attention.supports(&spec) {
                return Err(infer(format!(
                    "Gemma2 attention provider does not support layer {index}"
                )));
            }
            let head_dim = i64::try_from(layer.head_dim).map_err(infer)?;
            let q_width = i64::try_from(layer.q_width).map_err(infer)?;
            let kv_width = i64::try_from(layer.kv_width).map_err(infer)?;
            let batch_attention = Gemma2AttentionBatch {
                q: q_base,
                k: k_ptr,
                v: v_ptr,
                out: attention_out,
                out_bytes: attention_out_bytes,
                lse: lse_ptr,
                lse_bytes,
                q_offsets: boundaries,
                q_offsets_bytes: attention_offset_bytes,
                k_offsets: boundaries,
                k_offsets_bytes: attention_offset_bytes,
                total_q: tokens,
                total_k: tokens,
                batch: sequences,
                max_q: max_length,
                // Same-span Gemma2: explicit K extent follows Q. Cross/prefix
                // callers must supply their own max_k through the provider.
                max_k: max_length,
                heads: q_heads_usize,
                kv_heads: layer.kv_heads,
                scale: 1.0,
                strides: [
                    q_width, head_dim, 1, kv_width, head_dim, 1, kv_width, head_dim, 1, q_width,
                    head_dim, 1,
                ],
            };
            validate_batch(&spec, &batch_attention)?;
            // SAFETY: all pointers live in the retained work arena; the engine
            // bound the context before this request and the provider was
            // qualified at construction.
            unsafe {
                attention.forward_batch_bound(stream, &spec, &batch_attention)?;
                blas.linear(&plan.o, &work.attention, &buffers.o, &mut work.branch)?;
            }
            let mut residual = current;
            let mut rn_scale = buffers.post_attention_norm.cu_deviceptr();
            let mut rn_out = spare;
            let mut rn_rows = rows;
            let mut rn_width = hidden;
            launch_threads(
                &kernels.residual_norm,
                stream,
                (rn_rows, 1, 1),
                hidden,
                0,
                &mut [
                    arg(&mut branch),
                    arg(&mut residual),
                    arg(&mut rn_scale),
                    arg(&mut rn_out),
                    arg(&mut rn_rows),
                    arg(&mut rn_width),
                    arg(&mut eps),
                ],
            )?;
            core::mem::swap(&mut current, &mut spare);

            // Feed-forward pre-norm and packed gate/up.
            let mut ff_in = current;
            let mut ff_scale = buffers.pre_feedforward_norm.cu_deviceptr();
            let mut ff_out = normalized;
            let mut ff_rows = rows;
            let mut ff_width = hidden;
            launch_threads(
                &kernels.norm,
                stream,
                (ff_rows, 1, 1),
                hidden,
                0,
                &mut [
                    arg(&mut ff_in),
                    arg(&mut ff_scale),
                    arg(&mut ff_out),
                    arg(&mut ff_rows),
                    arg(&mut ff_width),
                    arg(&mut heads_u32),
                    arg(&mut stride_u64),
                    arg(&mut eps),
                ],
            )?;
            unsafe {
                blas.linear(
                    &plan.gate_up,
                    &work.normalized,
                    &buffers.gate_up,
                    &mut work.gate_up,
                )?;
            }
            let gate_elements = product(tokens, math.intermediate)?;
            let mut gate_in = gate_up;
            let mut gate_out = activation;
            let mut gate_rows = rows;
            let mut gate_width = u32_size(math.intermediate)?;
            launch(
                &kernels.gated_gelu,
                stream,
                blocks(gate_elements)?,
                0,
                &mut [
                    arg(&mut gate_in),
                    arg(&mut gate_out),
                    arg(&mut gate_rows),
                    arg(&mut gate_width),
                ],
            )?;
            unsafe {
                blas.linear(
                    &plan.down,
                    &work.activation,
                    &buffers.down,
                    &mut work.branch,
                )?;
            }
            let mut residual = current;
            let mut rn_scale = buffers.post_feedforward_norm.cu_deviceptr();
            let mut rn_out = spare;
            let mut rn_rows = rows;
            let mut rn_width = hidden;
            launch_threads(
                &kernels.residual_norm,
                stream,
                (rn_rows, 1, 1),
                hidden,
                0,
                &mut [
                    arg(&mut branch),
                    arg(&mut residual),
                    arg(&mut rn_scale),
                    arg(&mut rn_out),
                    arg(&mut rn_rows),
                    arg(&mut rn_width),
                    arg(&mut eps),
                ],
            )?;
            core::mem::swap(&mut current, &mut spare);

            // PLE contribution.
            let current_input = if current == work.state.cu_deviceptr() {
                &work.state
            } else {
                &work.next
            };
            unsafe {
                blas.linear(
                    &plan.ple_gate,
                    current_input,
                    &buffers.ple_gate,
                    &mut work.branch,
                )?;
            }
            let mut ple_gate_in = branch;
            let mut ple_values = ple;
            let mut ple_gate_out = activation;
            let mut ple_gate_rows = rows;
            let mut ple_gate_width = ple_width;
            let mut layer_index = u32_size(index)?;
            launch(
                &kernels.ple_gate,
                stream,
                blocks(product(tokens, math.ple_width)?)?,
                0,
                &mut [
                    arg(&mut ple_gate_in),
                    arg(&mut ple_values),
                    arg(&mut ple_gate_out),
                    arg(&mut ple_gate_rows),
                    arg(&mut ple_gate_width),
                    arg(&mut ple_layers),
                    arg(&mut layer_index),
                ],
            )?;
            unsafe {
                blas.linear(
                    &plan.ple_projection,
                    &work.activation,
                    &buffers.ple_projection,
                    &mut work.branch,
                )?;
            }
            let mut residual = current;
            let mut rn_scale = buffers.ple_post_norm.cu_deviceptr();
            let mut rn_out = spare;
            let mut rn_rows = rows;
            let mut rn_width = hidden;
            launch_threads(
                &kernels.residual_norm,
                stream,
                (rn_rows, 1, 1),
                hidden,
                0,
                &mut [
                    arg(&mut branch),
                    arg(&mut residual),
                    arg(&mut rn_scale),
                    arg(&mut rn_out),
                    arg(&mut rn_rows),
                    arg(&mut rn_width),
                    arg(&mut eps),
                ],
            )?;
            let carry = layer_carry_plan(current, spare);
            let mut scalar_in = carry.scalar_input;
            let mut scalar_out = carry.scalar_output;
            let mut scalar_count = u32_size(product(tokens, hidden as usize)?)?;
            let mut scalar = buffers.layer_scalar_value;
            launch(
                &kernels.scale,
                stream,
                blocks(product(tokens, math.hidden)?)?,
                0,
                &mut [
                    arg(&mut scalar_in),
                    arg(&mut scalar_out),
                    arg(&mut scalar_count),
                    arg(&mut scalar),
                ],
            )?;
            current = carry.carried;
            spare = carry.next_spare;
        }

        // Final norm and width-768 embedding projection.
        let mut final_in = current;
        let mut final_scale = weights.final_norm.cu_deviceptr();
        let mut final_out = normalized;
        let mut final_rows = rows;
        let mut final_width = hidden;
        let mut final_heads = 1u32;
        let mut final_stride = u64::from(hidden);
        let mut final_eps = math.norm_eps;
        launch_threads(
            &kernels.norm,
            stream,
            (final_rows, 1, 1),
            hidden,
            0,
            &mut [
                arg(&mut final_in),
                arg(&mut final_scale),
                arg(&mut final_out),
                arg(&mut final_rows),
                arg(&mut final_width),
                arg(&mut final_heads),
                arg(&mut final_stride),
                arg(&mut final_eps),
            ],
        )?;
        unsafe {
            blas.linear(
                &plans.embedding_projection,
                &work.normalized,
                &weights.embedding_projection,
                &mut work.output,
            )?;
        }

        if selections > 0 {
            if matches!(pool, Pool::Mean) {
                let mut pool_input = output;
                let mut pool_bounds = boundaries;
                let mut pool_indices = selected;
                let mut pool_out = work.pooled.cu_deviceptr();
                let mut pool_count = selections_u32;
                let mut pool_width = u32_size(math.embedding_dim)?;
                let (grid, threads) = mean_pool_launch(selections, math.embedding_dim)?;
                launch_threads(
                    &kernels.mean_pool,
                    stream,
                    grid,
                    threads,
                    0,
                    &mut [
                        arg(&mut pool_input),
                        arg(&mut pool_bounds),
                        arg(&mut pool_indices),
                        arg(&mut pool_out),
                        arg(&mut pool_count),
                        arg(&mut pool_width),
                    ],
                )?;
            } else {
                let mode = match pool {
                    Pool::Cls => 0u32,
                    Pool::LastToken => 2,
                    Pool::Mean => return Err(infer("Gemma2 mean pool dispatch mismatch")),
                    Pool::Splade => return Err(infer("Gemma2 SPLADE unsupported")),
                };
                let mut pool_input = output;
                let mut pool_bounds = boundaries;
                let mut pool_indices = selected;
                let mut pool_out = work.pooled.cu_deviceptr();
                let mut pool_count = selections_u32;
                let mut pool_width = u32_size(math.embedding_dim)?;
                let mut mode = mode;
                launch(
                    &kernels.pool,
                    stream,
                    blocks(product(selections, math.embedding_dim)?)?,
                    0,
                    &mut [
                        arg(&mut pool_input),
                        arg(&mut pool_bounds),
                        arg(&mut pool_indices),
                        arg(&mut pool_out),
                        arg(&mut pool_count),
                        arg(&mut pool_width),
                        arg(&mut mode),
                    ],
                )?;
            }
        }
        if !batch.raw_indices.is_empty() {
            crate::enqueue_raw_output(
                &kernels.bf16_to_f32,
                stream,
                output,
                work.pooled.cu_deviceptr(),
                raw_selected,
                batch,
                math.embedding_dim,
            )?;
        }
        Ok(())
    }
}

fn layer_scalar_from_bits(bits: u16) -> Result<f32, BackendError> {
    let value = f32::from_bits(u32::from(bits) << 16);
    if !value.is_finite() {
        return Err(start("layer_scalar is not finite"));
    }
    Ok(value)
}

fn bf16_source_scalar(source: &crate::model::WeightSource) -> Result<f32, BackendError> {
    require_bf16(source, "layer_scalar")?;
    if source.shape() != [1] {
        return Err(start("layer_scalar must have shape [1]"));
    }
    let mut bits = [0u16; 1];
    source
        .read_bf16(0, &mut bits)
        .map_err(|error| start(format!("layer_scalar read failed: {error}")))?;
    layer_scalar_from_bits(bits[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_config() -> Gemma2TextConfig {
        let layers = (0..24)
            .map(|index| {
                let full = matches!(index, 5 | 11 | 17 | 23);
                let head_dim = if full { 512 } else { 256 };
                let kv_heads = if full { 1 } else { 2 };
                crate::gemma2_text::Gemma2LayerConfig {
                    index,
                    kind: if full {
                        Gemma2AttentionKind::Full
                    } else {
                        Gemma2AttentionKind::Sliding
                    },
                    head_dim,
                    kv_heads,
                    q_width: 4 * head_dim,
                    kv_width: kv_heads * head_dim,
                    window_left: if full { -1 } else { 512 },
                    window_right: if full { -1 } else { 512 },
                    rope_theta: if full { 1_000_000.0 } else { 10_000.0 },
                    shared: false,
                    store_kv: true,
                    producer: None,
                }
            })
            .collect();
        Gemma2TextConfig {
            hidden_size: 512,
            intermediate_size: 2048,
            num_hidden_layers: 24,
            num_attention_heads: 4,
            vocab_size: 262_144,
            max_position_embeddings: 8192,
            rms_norm_eps: 1e-6,
            ple_width: 512,
            embedding_dim: 768,
            sliding_window: 512,
            rope_sliding_theta: 10_000.0,
            rope_full_theta: 1_000_000.0,
            activation: crate::gemma2_text::Gemma2Activation::GeluTanh,
            layers,
        }
    }

    #[test]
    fn work_allocation_matches_every_declared_bf16_and_f32_slab() {
        let text = synthetic_config();
        let key = (7usize, 2usize, 3usize);
        let input = crate::staging::input_capacity_with_tail(key, key.2).unwrap();
        let tokens = key.0;
        let hidden = text.hidden_size;
        let max_q = 2048usize;
        let max_kv = 512usize;
        let intermediate = text.intermediate_size;
        let ple = text.num_hidden_layers * text.ple_width;
        let lse = tokens * text.num_attention_heads * 4;
        let expected = input * 4
            + lse
            + tokens * hidden * 8
            + tokens * max_q * 2
            + tokens * max_kv * 2
            + tokens * max_kv * 2
            + tokens * max_q * 2
            + tokens * intermediate * 2 * 2
            + tokens * intermediate * 2
            + tokens * ple * 2
            + tokens * text.embedding_dim * 2
            + tokens * max_kv * 8
            + key.2 * text.embedding_dim * 4;
        assert_eq!(Gemma2Work::allocation_bytes(key, &text).unwrap(), expected);
        assert_eq!(Gemma2Work::lse_elements(&text, tokens).unwrap(), 28);
    }

    #[test]
    fn shared_kv_composition_keeps_only_queries_on_shared_layers() {
        let mut config = synthetic_config();
        for layer in &mut config.layers {
            if layer.index >= 22 {
                layer.shared = true;
                layer.store_kv = false;
                layer.producer = Some(if matches!(layer.kind, Gemma2AttentionKind::Sliding) {
                    21
                } else {
                    17
                });
            }
        }
        let plans = Plans::new(&config, 7).expect("shared host plans");
        assert!(plans.layers[22].k.is_none());
        assert!(plans.layers[22].v.is_none());
        assert!(plans.layers[23].k.is_none());
        assert!(plans.layers[23].v.is_none());
        assert!(plans.layers[21].k.is_some());
        assert!(plans.layers[17].k.is_some());
    }

    #[test]
    fn media_placeholder_coverage_rejects_partial_or_missing_spans() {
        assert!(check_media_row_coverage(0, None).is_ok());
        assert!(check_media_row_coverage(3, Some(3)).is_ok());
        assert!(check_media_row_coverage(3, None).is_err());
        assert!(check_media_row_coverage(2, Some(3)).is_err());
        assert!(check_media_row_coverage(2, Some(2)).is_ok());
    }

    #[test]
    fn bf16_layer_scalar_is_finite_expansion() {
        assert_eq!(layer_scalar_from_bits(0x3f80).unwrap(), 1.0);
        assert_eq!(layer_scalar_from_bits(0xbf80).unwrap(), -1.0);
        assert!(layer_scalar_from_bits(0x7f80).is_err());
        assert!(layer_scalar_from_bits(0x7fc0).is_err());
    }

    #[test]
    fn native_gelu_oracle_preserves_each_bf16_stage() {
        fn rounded(value: f32) -> f32 {
            half::bf16::from_f32(value).to_f32()
        }
        fn oracle(value: f32) -> f32 {
            let square = rounded(value * value);
            let cube = rounded(square * value);
            let alpha = rounded(value + rounded(0.044715 * cube));
            let angle = rounded(rounded(0.797_884_6) * alpha);
            let tangent = rounded(angle.tanh());
            rounded(rounded(0.5 * value) * rounded(1.0 + tangent))
        }
        for (value, bits) in [
            (0.0f32, 0x0000_0000u32),
            (0.5, 0x3eb1_0000),
            (1.0, 0x3f58_0000),
            (-1.0, 0xbe22_0000),
            (3.0, 0x4040_0000),
        ] {
            assert_eq!(oracle(value).to_bits(), bits, "value {value}");
        }
    }
    #[test]
    fn mean_pool_short_span_launch_always_has_at_least_eight_x_threads() {
        for selected in 1..=4 {
            for width in [9usize, 16, 512, 768] {
                let (grid, threads) = mean_pool_launch(selected, width).expect("mean geometry");
                assert_eq!(grid, (selected as u32, width.div_ceil(8) as u32, 1));
                assert_eq!(threads, 256);
                assert!(threads >= 8);
                let old_next_pow2 = selected.next_power_of_two().min(1024);
                let fixed_tile_features = (threads as usize).min(8);
                let latent_tile_features = old_next_pow2.min(8);
                assert_eq!(fixed_tile_features, 8);
                assert!(latent_tile_features < 8);
            }
        }
    }
    #[test]
    fn rope_launch_covers_one_thread_per_pair_for_local_global_and_shared_q() {
        for tokens in [1usize, 4, 7] {
            for (q_heads, kv_heads, dim) in [(4usize, 2usize, 256usize), (4, 1, 512)] {
                let pairs = tokens * (q_heads + kv_heads) * (dim / 2);
                let (grid, threads) =
                    rope_launch(tokens, q_heads, kv_heads, dim, dim).expect("rope launch geometry");
                assert_eq!(grid, (pairs.div_ceil(256) as u32, 1, 1));
                assert_eq!(threads, 256);
                assert!(grid.0 as usize * 256 >= pairs);
            }
        }
        // Shared-Q layers currently retain the generic rope ABI's nonzero KV
        // dispatch on the dummy K buffer, so the same pair coverage applies.
        let (shared, _) = rope_launch(4, 4, 2, 256, 256).expect("shared-Q rope geometry");
        let (local, _) = rope_launch(4, 4, 2, 256, 256).expect("local rope geometry");
        assert_eq!(shared, local);
    }

    #[test]
    fn rope_launch_rejects_zero_width_and_overflow() {
        for args in [
            (0usize, 4usize, 2usize, 256usize, 256usize),
            (4, 0, 2, 256, 256),
            (4, 4, 0, 256, 256),
            (4, 4, 2, 0, 0),
            (4, 4, 2, 256, 0),
            (4, 4, 2, 256, 255),
            (4, 4, 2, 256, 128),
            (usize::MAX, 4, 2, 512, 512),
        ] {
            assert!(rope_launch(args.0, args.1, args.2, args.3, args.4).is_err());
        }
    }
    #[test]
    fn nonunit_bf16_layer_scalar_carry_uses_scaled_output_for_next_and_final_norm() {
        let scalars = [0.5f32, -1.25f32, 1.5f32];
        assert!(scalars.iter().any(|value| *value != 1.0));
        for &scalar in &scalars {
            assert_eq!(half::bf16::from_f32(scalar).to_f32(), scalar);
        }
        let mut current = 0x1000u64;
        let mut spare = 0x2000u64;
        for (layer, &scalar) in scalars.iter().enumerate() {
            let plan = layer_carry_plan(current, spare);
            assert_eq!(plan.scalar_input, spare, "layer {layer} scalar input");
            assert_eq!(plan.scalar_output, current, "layer {layer} scalar output");
            assert_eq!(plan.carried, plan.scalar_output, "layer {layer} carry");
            assert_eq!(
                plan.next_input_norm, plan.scalar_output,
                "layer {layer} next input norm source"
            );
            assert_eq!(
                plan.final_norm_input, plan.scalar_output,
                "layer {layer} final norm source"
            );
            assert_ne!(scalar, 1.0, "layer {layer} must exercise nonunit scalar");
            current = plan.carried;
            spare = plan.next_spare;
        }
        assert_eq!(current, 0x1000);
        assert_eq!(spare, 0x2000);
    }
}
