// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Concrete Gemma4 text/vision/Rune runtime composition.
//!
//! The dense text executor follows Native `gemma_models.cpp`: learned embedding
//! scale, PLE model/token projections, shared KV producer layers, local d256
//! and global d512 attention, per-layer input gates, layer scalars, and the
//! final unit RMS norm. The optional vision tower projects every prepared image
//! into text-hidden rows before the first text layer. Rune uses the same
//! backbone in causal `option_tokens` mode and the trained BF16 tied option
//! head with its publisher softcap.

use crate::cublas::Blas;
use crate::gemma4_e2b_model::{Gemma4E2bExecutor, Gemma4E2bWork, Head512Aot};
use crate::gemma4_option_head::{Gemma4OptionWork, PreparedGemma4OptionHead};
use crate::gemma4_text_model::{Gemma4Config, Gemma4Contract, Gemma4Weights};
use crate::gemma4_vision_model::{
    Gemma4VisionEngine, Gemma4VisionImage, Gemma4VisionWeights, Gemma4VisionWork,
};
use crate::lt::ShapeError;
use crate::staging::HostIoLease;
use crate::{infer, start, WorkKey};
use cuda_core::{Context, Stream};
use infinity_v3_backend_core::{BackendError, Batch, DecisionInput, Pool};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const DEFAULT_HEAD512_MAX_OPTIN: u64 = 128 * 1024;

fn admitted_arch(context: &Arc<Context>) -> Result<String, BackendError> {
    let (major, minor) = context.compute_capability().map_err(start)?;
    match (major, minor) {
        (9, 0) => Ok("sm_90".to_owned()),
        (10, 0) => Ok("sm_100".to_owned()),
        (12, 0) => Ok("sm_120".to_owned()),
        _ => Err(start(format!(
            "Gemma4 requires sm90/sm100/sm120 attention artifacts, found sm{major}{minor}"
        ))),
    }
}

/// One loaded Gemma4 model. Device weights, FA4/Head512 providers and the
/// vision tower are cold-owned; request arenas are created by the Engine.
pub(crate) struct Gemma4Model {
    pub(crate) weights: Gemma4Weights,
    pub(crate) config: Gemma4Config,
    executor: Gemma4E2bExecutor,
    vision: Option<Gemma4VisionEngine>,
}

impl Gemma4Model {
    pub(crate) fn load(
        path: &Path,
        context: Arc<Context>,
        stream: Arc<Stream>,
        contract: Gemma4Contract,
    ) -> Result<Self, BackendError> {
        let weights = Gemma4Weights::load(path, contract)?;
        let config = weights.config.clone();
        let module_path =
            std::env::var_os(infinity_v3_config::infinity_env!("OXIDE_KERNEL_MODULE"))
                .map(PathBuf::from);
        let fa4_root = std::env::var_os(infinity_v3_config::infinity_env!("FA4_ROOT"))
            .map(PathBuf::from)
            .ok_or_else(|| start("Gemma4 requires INFINITY_FA4_ROOT for FA4 attention"))?;
        let head512_root = std::env::var_os(infinity_v3_config::infinity_env!("HEAD512_ROOT"))
            .map(PathBuf::from)
            .ok_or_else(|| start("Gemma4 requires INFINITY_HEAD512_ROOT for d512 attention"))?;
        let architecture = std::env::var_os(infinity_v3_config::infinity_env!("HEAD512_ARCH"))
            .map(PathBuf::from)
            .and_then(|value| value.to_str().map(str::to_owned))
            .unwrap_or(admitted_arch(&context)?);
        let max_optin = std::env::var("INFINITY_HEAD512_MAX_OPTIN")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(DEFAULT_HEAD512_MAX_OPTIN);
        let attention = Arc::new(Head512Aot::load(
            &context,
            &head512_root,
            &architecture,
            max_optin,
        )?);
        let executor = Gemma4E2bExecutor::new_with_fa4_local(
            context.clone(),
            stream.clone(),
            &weights,
            Some(attention),
            &fa4_root,
            module_path.as_deref(),
        )
        .map_err(|error| start(error.to_string()))?;
        let vision = vision_present(path)?
            .then(|| {
                let weights = Gemma4VisionWeights::load(path)?;
                Gemma4VisionEngine::new(
                    context.clone(),
                    stream.clone(),
                    &weights,
                    module_path.as_deref(),
                    &fa4_root,
                )
            })
            .transpose()?;
        Ok(Self {
            weights,
            config,
            executor,
            vision,
        })
    }

    pub(crate) fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    pub(crate) fn workspace_bytes(&self, key: WorkKey) -> Result<usize, BackendError> {
        Gemma4E2bWork::allocation_bytes_moe(key, &self.weights, key.0, self.executor.moe_provider())
    }

    pub(crate) fn workspace(
        &self,
        stream: Arc<Stream>,
        key: WorkKey,
    ) -> Result<Gemma4E2bWork, BackendError> {
        Gemma4E2bWork::new_with_raw_moe(
            stream,
            &self.weights,
            key,
            key.0,
            self.executor.moe_provider(),
        )
    }

    pub(crate) fn preflight(&self, batch: &Batch, pool: &Pool) -> Result<(), ShapeError> {
        let tokens = batch.input_ids.len();
        if tokens == 0 || batch.cumulative_seq_lengths.len() < 2 {
            return Err(ShapeError::Rejected(infer(
                "Gemma4 requires at least one packed token and sequence",
            )));
        }
        if batch.compact_input_ids.is_some()
            || batch.scatter_unfold.is_some()
            || batch.fold_gather.is_some()
        {
            return Err(ShapeError::Rejected(infer(
                "Gemma4 compact/radix batches are not admitted; full logical rows are required",
            )));
        }
        if batch.input_ids.len() != batch.position_ids.len()
            || batch.input_ids.len() != batch.token_type_ids.len()
        {
            return Err(ShapeError::Rejected(infer(
                "Gemma4 packed input arrays disagree in length",
            )));
        }
        if let Some(&max) = batch.position_ids.iter().max() {
            if usize::try_from(max)
                .map_err(|_| ShapeError::Rejected(infer("Gemma4 position overflow")))?
                >= self.config.max_position_embeddings
            {
                return Err(ShapeError::Rejected(infer(
                    "Gemma4 position exceeds max_position_embeddings",
                )));
            }
        }
        if batch.pooled_indices.len() > tokens || batch.raw_indices.len() > tokens {
            return Err(ShapeError::Rejected(infer(
                "Gemma4 selection count exceeds the packed token count",
            )));
        }
        if *pool == Pool::Splade {
            return Err(ShapeError::Rejected(infer(
                "Gemma4 SPLADE pooling is unsupported",
            )));
        }
        if let Some(media) = batch.multimodal.iter().flatten().next() {
            if !media.audios.is_empty() {
                return Err(ShapeError::Rejected(infer(
                    "Gemma4 official checkpoints have no audio tower",
                )));
            }
            if !media.images.is_empty() && self.vision.is_none() {
                return Err(ShapeError::Rejected(infer(
                    "Gemma4 image request requires an admitted vision tower",
                )));
            }
        }
        Ok(())
    }

    pub(crate) fn encode_vision(
        &self,
        blas: &Blas,
        batch: &Batch,
        lease: &HostIoLease,
    ) -> Result<Option<(Gemma4VisionWork, Vec<u32>)>, BackendError> {
        let Some(engine) = self.vision.as_ref() else {
            return Ok(None);
        };
        let mut images = Vec::new();
        let mut rows = Vec::new();
        for media in batch.multimodal.iter().flatten() {
            for (start, image) in &media.images {
                let config = engine.executor().config();
                let soft = config.soft_tokens(image.grid_thw)?;
                let end = start
                    .checked_add(soft)
                    .ok_or_else(|| infer("Gemma4 image token span overflow"))?;
                if end > batch.input_ids.len() {
                    return Err(infer(
                        "Gemma4 image span exceeds the packed text token count",
                    ));
                }
                images.push(Gemma4VisionImage {
                    pixels: &image.pixels,
                    grid_thw: image.grid_thw,
                });
                rows.extend(
                    (*start..end)
                        .map(|row| {
                            u32::try_from(row).map_err(|_| infer("Gemma4 image row overflow"))
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                );
            }
        }
        if images.is_empty() {
            return Ok(None);
        }
        let mut work = engine.encode_request(blas, &images)?;
        work.set_scatter_rows(&rows)?;
        let _ = lease;
        Ok(Some((work, rows)))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn enqueue_text(
        &mut self,
        blas: Option<&Blas>,
        stream: &Stream,
        work: &mut Gemma4E2bWork,
        batch: &Batch,
        pool: &Pool,
        max_length: usize,
        tokens: usize,
        lease: &HostIoLease,
        vision: Option<(&Gemma4VisionWork, &[u32])>,
    ) -> Result<(), BackendError> {
        self.executor.enqueue_text_with_vision(
            blas,
            stream,
            work,
            batch,
            pool,
            max_length,
            tokens,
            lease,
            vision.map(|(work, rows)| crate::gemma4_e2b_model::VisionInjection { work, rows }),
        )
    }

    pub(crate) fn finish_text(
        &self,
        stream: &Stream,
        work: &mut Gemma4E2bWork,
        batch: &Batch,
        pooled_rows: usize,
        raw_rows: usize,
        lease: &mut HostIoLease,
    ) -> Result<(), BackendError> {
        self.executor
            .finish_text(stream, work, batch, pooled_rows, raw_rows, lease)
    }

    pub(crate) fn prepare_rune(
        &mut self,
        batch: &Batch,
        requests: &[DecisionInput],
    ) -> Result<PreparedGemma4OptionHead, BackendError> {
        self.executor.prepare_rune(batch, requests)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn enqueue_rune(
        &mut self,
        blas: Option<&Blas>,
        stream: &Stream,
        work: &mut Gemma4E2bWork,
        option_work: &mut Gemma4OptionWork,
        batch: &Batch,
        max_length: usize,
        tokens: usize,
        requests: &[DecisionInput],
        lease: &mut HostIoLease,
        prepared: &PreparedGemma4OptionHead,
        vision: Option<(&Gemma4VisionWork, &[u32])>,
    ) -> Result<(), BackendError> {
        unsafe {
            self.executor.enqueue_rune(
                blas,
                stream,
                work,
                option_work,
                batch,
                max_length,
                tokens,
                requests,
                lease,
                prepared,
                vision.map(|(work, rows)| crate::gemma4_e2b_model::VisionInjection { work, rows }),
            )
        }
    }
}

fn vision_present(path: &Path) -> Result<bool, BackendError> {
    let raw = std::fs::read(path.join("config.json")).map_err(start)?;
    let value: serde_json::Value = serde_json::from_slice(&raw).map_err(start)?;
    Ok(value
        .get("vision_config")
        .and_then(serde_json::Value::as_object)
        .is_some())
}
