// Concrete Qwen3-VL composition: config -> vision tower -> dense text decoder
// -> trained readout -> engine `Backend`.
//
// The family is deliberately narrow: `qwen3_vl`, `colqwen3` and
// `qwen3_vl_nemotron_embed`. Qwen3.5-VL is a different text foundation and is
// handled by its own author through the shared contract in
// `qwen-vl-shared-plan-v1`.

use crate::checkpoint::start;
use crate::qwen3_vl_batch::{pack_batch, PackedVlRequest};
use crate::qwen3_vl_config::{Qwen3VlConfig, ReadoutKind};
use crate::qwen3_vl_device::{
    Qwen3VlExecutor, Qwen3VlForwardResult, Qwen3VlPending, Qwen3VlWorkspaces,
};
use crate::qwen3_vl_readout::Qwen3VlReadout;
use crate::staging::HostIoLease;
use cuda_core::{Context, Stream};
use infinity_v3_backend_core::{
    Backend, BackendError, Batch, Embedding, Embeddings, Inference, Pool, Predictions,
    TokenPredictions,
};
use std::path::Path;
use std::sync::Arc;

/// Config-aware media preflight used by every public route before an empty
/// selector shortcut or any GPU work. The descriptor's own patch_dim is not
/// trusted: the loaded tower's patch dimension and merge size are authoritative.
pub(crate) fn validate_media_against_config(
    config: &Qwen3VlConfig,
    request: &PackedVlRequest,
) -> Result<(), BackendError> {
    let patch_dim = config.vision.patch_dim()?;
    if patch_dim == 0 {
        return Err(start("vision patch dimension is zero"));
    }
    for image in &request.forward.images {
        let patch_rows = image.patch_rows()?;
        let expected = patch_rows
            .checked_mul(patch_dim)
            .ok_or_else(|| start("configured image payload overflow"))?;
        if image.pixels.len() != expected {
            return Err(start(format!(
                "{} pixel payload {} does not match configured patch_dim {} x {} rows",
                image.name,
                image.pixels.len(),
                patch_dim,
                patch_rows
            )));
        }
        if image.merge_size != config.vision.spatial_merge_size {
            return Err(start(format!(
                "{} merge size {} does not match configured {}",
                image.name, image.merge_size, config.vision.spatial_merge_size
            )));
        }
    }
    Ok(())
}

/// Validate selection bounds and duplicates before any lease/ticket. Mixed
/// pooled+raw selection is rejected because this engine's readout emits one
/// result kind per request.
pub(crate) fn validate_selectors(batch: &Batch, sequences: usize) -> Result<(), BackendError> {
    if !batch.pooled_indices.is_empty() && !batch.raw_indices.is_empty() {
        return Err(start(
            "Qwen3-VL does not serve mixed pooled and raw selections",
        ));
    }
    for (label, indices) in [
        ("pooled", batch.pooled_indices.as_slice()),
        ("raw", batch.raw_indices.as_slice()),
    ] {
        let mut seen = std::collections::BTreeSet::new();
        for &index in indices {
            if index as usize >= sequences {
                return Err(start(format!(
                    "{label} selection {index} exceeds {sequences} sequences"
                )));
            }
            if !seen.insert(index) {
                return Err(start(format!("duplicate {label} selection {index}")));
            }
        }
    }
    Ok(())
}

/// One CPU preflight used by the async Backend submit and the public sync
/// Batch route. Media, config consistency and selector bounds all run before
/// an empty-selector success or any GPU work.
pub(crate) fn prepare_batch(
    config: &Qwen3VlConfig,
    batch: &Batch,
    pool: Pool,
    raw: bool,
) -> Result<PackedVlRequest, BackendError> {
    let request = pack_batch(batch, pool, raw)?;
    validate_media_against_config(config, &request)?;
    validate_selectors(batch, request.sequences)?;
    Ok(request)
}

fn product(left: usize, right: usize) -> Result<usize, BackendError> {
    left.checked_mul(right)
        .ok_or_else(|| start("dimension overflow"))
}

#[derive(Clone, Debug)]
pub struct Qwen3VlBatchOutput {
    pub pooled: Option<Vec<f32>>,
    pub tokens: Option<Vec<f32>>,
    pub width: usize,
    pub sequences: usize,
    pub cumulative: Vec<u32>,
}

pub struct Qwen3VlModel {
    config: Qwen3VlConfig,
    executor: Qwen3VlExecutor,
    readout: Option<Qwen3VlReadout>,
    stream: Arc<Stream>,
}

impl Qwen3VlModel {
    pub fn open(
        checkpoint: &Path,
        context: &Arc<Context>,
        stream: &Arc<Stream>,
    ) -> Result<Self, BackendError> {
        Self::open_with_readout(checkpoint, context, stream, None)
    }

    /// Open with an explicit trained readout override. No Native Qwen3-VL
    /// `model_type` selects `ReadoutConfig::rerank()`; a caller must opt in
    /// only after verifying the checkpoint's `score.weight`.
    pub fn open_with_readout(
        checkpoint: &Path,
        context: &Arc<Context>,
        stream: &Arc<Stream>,
        override_readout: Option<crate::qwen3_vl_config::ReadoutConfig>,
    ) -> Result<Self, BackendError> {
        let mut config = Qwen3VlConfig::from_path(checkpoint)?;
        if let Some(readout) = override_readout {
            config.readout = readout;
            config.validate()?;
        }
        let executor = Qwen3VlExecutor::new(checkpoint, context, stream)?;
        let readout =
            if config.readout.kind == ReadoutKind::Embedding && !config.readout.multi_vector {
                None
            } else {
                Some(Qwen3VlReadout::load(
                    checkpoint,
                    &config.readout,
                    config.text.hidden_size,
                )?)
            };
        if config.readout.multi_vector && !config.readout.normalize {
            return Err(start(
                "multi-vector Qwen3-VL requires per-row normalization",
            ));
        }
        Ok(Self {
            config,
            executor,
            readout,
            stream: stream.clone(),
        })
    }

    pub fn config(&self) -> &Qwen3VlConfig {
        &self.config
    }

    pub fn synchronize(&self) -> Result<(), BackendError> {
        self.stream.synchronize().map_err(start)
    }

    pub fn supports_images(&self) -> bool {
        true
    }

    pub fn supports_video(&self) -> bool {
        true
    }

    pub fn supports_raw(&self) -> bool {
        true
    }

    pub fn supports_multi_vector(&self) -> bool {
        self.config.readout.multi_vector
    }

    pub fn output_width(&self) -> usize {
        self.readout
            .as_ref()
            .map_or(self.config.text.hidden_size, Qwen3VlReadout::output_width)
    }

    pub(crate) fn hidden(&self) -> usize {
        self.config.text.hidden_size
    }

    pub(crate) fn stream(&self) -> Arc<Stream> {
        self.stream.clone()
    }

    pub(crate) fn bind_context(&self) -> Result<(), BackendError> {
        self.stream.context().bind_to_thread().map_err(start)
    }

    pub(crate) fn workspace() -> Qwen3VlWorkspaces {
        Qwen3VlWorkspaces::default()
    }

    pub(crate) fn launch_packed(
        &mut self,
        request: &PackedVlRequest,
        workspaces: &mut Qwen3VlWorkspaces,
        lease: &mut HostIoLease,
    ) -> Result<Qwen3VlPending, BackendError> {
        match self
            .executor
            .forward_outputs_with(&request.forward, workspaces, Some(lease))?
        {
            Qwen3VlForwardResult::Launched(pending) => Ok(pending),
            Qwen3VlForwardResult::Ready(_) => Err(start(
                "VL async launch unexpectedly completed without a host lease",
            )),
        }
    }

    /// Pack raw token rows from the request's pinned host output after the
    /// ordered fence. The readout projection is the checkpoint's trained one.
    /// Actual raw-output contract: plain Qwen3-VL returns the final hidden rows
    /// unchanged (Native MultiVectorModel is not applied); trained multi-vector
    /// wrappers apply their projection/normalization; Nemotron applies row L2.
    pub(crate) fn raw_output_rows(
        readout: Option<&Qwen3VlReadout>,
        raw: &[f32],
        tokens: usize,
        hidden: usize,
        multi_vector: bool,
    ) -> Result<(Vec<f32>, usize), BackendError> {
        match readout {
            Some(readout) => Ok((readout.apply_rows(raw, tokens)?, readout.output_width())),
            None if !multi_vector => Ok((raw.to_vec(), hidden)),
            None => Err(start(
                "multi-vector raw output requires a trained projection/normalization readout",
            )),
        }
    }

    pub(crate) fn pack_raw(
        &self,
        batch: &Batch,
        pool: Pool,
        raw_tokens: bool,
        raw: &[f32],
    ) -> Result<Embeddings, BackendError> {
        let hidden = self.config.text.hidden_size;
        let tokens = batch.input_ids.len();
        let sequences = batch
            .cumulative_seq_lengths
            .len()
            .checked_sub(1)
            .ok_or_else(|| start("missing cumulative boundaries"))?;
        if raw.len() != product(tokens, hidden)? {
            return Err(start("pinned VL output row count mismatch"));
        }
        let mut embeddings = Embeddings::default();
        if raw_tokens {
            let (rows, width) = Self::raw_output_rows(
                self.readout.as_ref(),
                raw,
                tokens,
                hidden,
                self.config.readout.multi_vector,
            )?;
            for &index in &batch.raw_indices {
                let row = index as usize;
                let begin = *batch
                    .cumulative_seq_lengths
                    .get(row)
                    .ok_or_else(|| start("raw sequence index out of range"))?
                    as usize;
                let end = *batch
                    .cumulative_seq_lengths
                    .get(row + 1)
                    .ok_or_else(|| start("raw sequence end out of range"))?
                    as usize;
                let vectors = rows
                    .get(begin * width..end * width)
                    .ok_or_else(|| start("raw token rows missing"))?
                    .chunks_exact(width)
                    .map(|row| row.to_vec())
                    .collect();
                embeddings.insert(row, Embedding::All(vectors));
            }
            return Ok(embeddings);
        }
        let mut pooled = Vec::with_capacity(sequences * hidden);
        for pair in batch.cumulative_seq_lengths.windows(2) {
            let begin = pair[0] as usize;
            let end = pair[1] as usize;
            match pool {
                Pool::Cls => pooled.extend_from_slice(&raw[begin * hidden..(begin + 1) * hidden]),
                Pool::LastToken => {
                    let last = end - 1;
                    pooled.extend_from_slice(&raw[last * hidden..(last + 1) * hidden]);
                }
                Pool::Mean => {
                    let count = (end - begin) as f32;
                    for column in 0..hidden {
                        let mut sum = 0f32;
                        for row in begin..end {
                            sum += raw[row * hidden + column];
                        }
                        pooled.push(sum / count);
                    }
                }
                Pool::Splade => return Err(start("Oxide Qwen3-VL SPLADE pooling is unsupported")),
            }
        }
        if let Some(readout) = self.readout.as_ref() {
            if readout.config.kind == ReadoutKind::Rerank {
                let scores = readout.apply_scores(&pooled, sequences)?;
                for &index in &batch.pooled_indices {
                    let row = index as usize;
                    embeddings.insert(row, Embedding::Pooled(scores[row..row + 1].to_vec()));
                }
                return Ok(embeddings);
            }
        }
        for &index in &batch.pooled_indices {
            let row = index as usize;
            let begin = row
                .checked_mul(hidden)
                .ok_or_else(|| start("pooled row offset overflow"))?;
            embeddings.insert(
                row,
                Embedding::Pooled(pooled[begin..begin + hidden].to_vec()),
            );
        }
        Ok(embeddings)
    }

    pub fn forward_batch(
        &mut self,
        batch: &Batch,
        pool: Pool,
    ) -> Result<Qwen3VlBatchOutput, BackendError> {
        let multi_vector = self.config.readout.multi_vector;
        if multi_vector && !batch.pooled_indices.is_empty() {
            return Err(start(
                "Qwen3-VL multi-vector wrappers return token rows and reject pooling",
            ));
        }
        let raw_tokens = multi_vector || !batch.raw_indices.is_empty();
        // Validate media, config consistency and selectors before the empty shortcut.
        let request = prepare_batch(&self.config, batch, pool, raw_tokens)?;
        let pooled_request = batch.pooled_indices.len();
        let raw_request = batch.raw_indices.len();
        if pooled_request == 0 && raw_request == 0 {
            return Ok(Qwen3VlBatchOutput {
                pooled: Some(Vec::new()),
                tokens: None,
                width: self.output_width(),
                sequences: request.sequences,
                cumulative: batch.cumulative_seq_lengths.clone(),
            });
        }
        let output = self.executor.forward_outputs(&request.forward)?;
        if let Some(tokens) = output.raw {
            let readout = self
                .readout
                .as_ref()
                .ok_or_else(|| start("raw Qwen3-VL output requires a configured readout"))?;
            let rows = readout.apply_rows(&tokens, request.tokens)?;
            return Ok(Qwen3VlBatchOutput {
                pooled: None,
                tokens: Some(rows),
                width: readout.output_width(),
                sequences: request.sequences,
                cumulative: batch.cumulative_seq_lengths.clone(),
            });
        }
        let pooled = output
            .pooled
            .ok_or_else(|| start("Qwen3-VL request produced no output"))?;
        if let Some(readout) = self.readout.as_ref() {
            if readout.config.kind == ReadoutKind::Rerank {
                return Ok(Qwen3VlBatchOutput {
                    pooled: Some(readout.apply_scores(&pooled, request.sequences)?),
                    tokens: None,
                    width: 1,
                    sequences: request.sequences,
                    cumulative: batch.cumulative_seq_lengths.clone(),
                });
            }
        }
        Ok(Qwen3VlBatchOutput {
            pooled: Some(pooled),
            tokens: None,
            width: self.output_width(),
            sequences: request.sequences,
            cumulative: batch.cumulative_seq_lengths.clone(),
        })
    }
}

pub struct Qwen3VlBackend {
    engine: crate::qwen3_vl_engine::Qwen3VlEngine,
}

impl Qwen3VlBackend {
    pub fn new(checkpoint: &Path, dtype: &str, device_id: usize) -> Result<Self, BackendError> {
        Self::new_with_pool(checkpoint, dtype, device_id, Pool::LastToken)
    }

    pub fn new_with_pool(
        checkpoint: &Path,
        dtype: &str,
        device_id: usize,
        pool: Pool,
    ) -> Result<Self, BackendError> {
        if dtype != "bfloat16" {
            return Err(start(
                "Qwen3-VL Oxide concrete execution currently requires bfloat16",
            ));
        }
        let context = Context::new(device_id).map_err(start)?;
        let stream = context.new_stream().map_err(start)?;
        let model = Qwen3VlModel::open(checkpoint, &context, &stream)?;
        let engine = crate::qwen3_vl_engine::Qwen3VlEngine::new(model, context, pool)?;
        Ok(Self { engine })
    }

    pub fn config(&self) -> Result<Qwen3VlConfig, BackendError> {
        self.engine.config()
    }

    pub fn output_width(&self) -> Result<usize, BackendError> {
        self.engine.output_width()
    }

    /// Explicit-pool submission on the same owning Backend route. Used by the
    /// quality consumer to emit Native-comparable Cls/Mean/LastToken/raw rows
    /// without constructing additional model instances.
    pub fn submit_with_pool(
        &self,
        batch: Batch,
        pool: Pool,
    ) -> Result<Inference<Embeddings>, BackendError> {
        self.engine.submit_with_pool(batch, pool)
    }
}

impl Backend for Qwen3VlBackend {
    fn health(&self) -> Result<(), BackendError> {
        self.engine.health()
    }

    fn max_batch_size(&self) -> Option<usize> {
        None
    }

    fn embed(&self, batch: Batch) -> Result<Embeddings, BackendError> {
        self.submit_embed(batch)?.complete()
    }

    fn submit_embed(&self, batch: Batch) -> Result<Inference<Embeddings>, BackendError> {
        self.engine.submit(batch)
    }

    fn predict(&self, _batch: Batch) -> Result<Predictions, BackendError> {
        Err(start(
            "Qwen3-VL does not select a Native sequence-classification wrapper",
        ))
    }

    fn predict_tokens(&self, _batch: Batch) -> Result<TokenPredictions, BackendError> {
        Err(start(
            "Qwen3-VL does not select a Native token-classification wrapper",
        ))
    }
}
