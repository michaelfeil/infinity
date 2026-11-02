// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Experimental dense BF16 text route over the shared engine/I/O completion.
use crate::clef_warm::{ClefQ35Head, ClefWarmPrepared};
use crate::cublas::Blas;
use crate::decision_external_gpu::ExternalGpuHead;
use crate::decision_head::PackedDecisionBatch;
use crate::decision_head_gpu::OneJevGpuHead;
use crate::external_warm::{ExternalHeadCapacity, ExternalWarmMetadata, ExternalWarmPlan};
use crate::fa4::{AttentionSpec, DType, Fa4, Scale, Window};
use crate::gdn_prepared::GdnGeometry;
use crate::lt::ShapeError;
use crate::qwen35_block_attention::BlockArchitecture;
use crate::qwen35_block_layout::Qwen35BlockMode;
use crate::qwen35_execute::{PreparedWork, Qwen35AuxKernels};
use crate::qwen35_model::Qwen35Weights;
use crate::qwen35_prepared::Qwen35Kernels;
use crate::qwen35_tables::Qwen35Tables;
use crate::qwen35_weights::Qwen35DeviceWeights;
use crate::qwen35_work::{DecisionCapacity, Payload, Qwen35Work};
use crate::warm_head::{WarmDecisionMetadata, WarmDecisionPlan, WarmHeadCapacity};
use crate::{arg, blocks, infer, launch, product, staging, start, u32_size, WorkKey};
use cuda_core::{Context, Function, Stream};
use infinity_v3_backend_core::{BackendError, Batch, DecisionInput, Pool};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use crate::qwen35_block_layout::{
    validate_block_token_count, BLOCK_TAIL_WORDS as BLOCK_METADATA_WORDS,
};

/// Explicit unadmitted text experiment. This does not expose prefix/cache/media
/// or the Decision head, and never enables CUDA graphs or runtime quantization.
#[derive(Clone, Debug)]
pub struct Qwen35TextOptions {
    pub causal: bool,
    pub fa4_root: PathBuf,
    pub gdn_raw_kernels: PathBuf,
    pub gdn_provider: PathBuf,
    pub gdn_aux_kernels: PathBuf,
}
pub(crate) struct Qwen35Model {
    pub weights: Qwen35DeviceWeights,
    pub options: Qwen35TextOptions,
    pub gdn_geometry: GdnGeometry,
    pub major: i32,
    pub sm_count: i32,
    capability: i32,
    block_mode: Option<Qwen35BlockMode>,
    math: Qwen35Kernels,
    aux: Qwen35AuxKernels,
    fa4: Fa4,
    tables: Qwen35Tables,
    gather: Function,
    pool: Function,
    mean: Function,
    raw: Function,
    context: Arc<Context>,
    stream: Arc<Stream>,
    decision: Option<Qwen35DecisionHead>,
}

pub(crate) enum Qwen35DecisionHead {
    OneJev(OneJevGpuHead),
    External(ExternalGpuHead),
    Clef(Box<ClefQ35Head>),
}

impl Qwen35DecisionHead {
    fn hidden_size(&self) -> usize {
        match self {
            Self::OneJev(head) => head.hidden_size(),
            Self::External(head) => head.hidden_size(),
            Self::Clef(head) => head.config().hidden,
        }
    }
}

pub(crate) enum WarmDecisionMetadataRoute {
    OneJev(WarmDecisionMetadata),
    External(ExternalWarmMetadata),
    Clef {
        prepared: ClefWarmPrepared,
        layout_end: usize,
    },
}

impl WarmDecisionMetadataRoute {
    pub(crate) fn layout_end(&self) -> usize {
        match self {
            Self::OneJev(metadata) => metadata.layout().end(),
            Self::External(metadata) => metadata.layout().end(),
            Self::Clef { layout_end, .. } => *layout_end,
        }
    }

    pub(crate) fn total_rows(&self) -> usize {
        match self {
            Self::OneJev(metadata) => metadata.total_rows(),
            Self::External(metadata) => metadata.total_rows(),
            Self::Clef { prepared, .. } => prepared.output_elements().unwrap_or(0),
        }
    }

    fn key(&self) -> WorkKey {
        match self {
            Self::OneJev(metadata) => metadata.key(),
            Self::External(metadata) => metadata.key(),
            Self::Clef { .. } => (0, 0, 0),
        }
    }
}

pub(crate) enum WarmDecisionRoute {
    OneJev(WarmDecisionPlan),
    External(ExternalWarmPlan),
    Clef(Box<ClefWarmPrepared>),
}

impl WarmDecisionRoute {
    pub(crate) fn total_rows(&self) -> usize {
        match self {
            Self::OneJev(plan) => plan.total_rows(),
            Self::External(plan) => plan.total_rows(),
            Self::Clef(prepared) => prepared.output_elements().unwrap_or(0),
        }
    }

    pub(crate) fn output_elements(&self) -> Result<usize, BackendError> {
        match self {
            Self::OneJev(plan) => Ok(plan.total_rows()),
            Self::External(plan) => Ok(plan.total_rows()),
            Self::Clef(prepared) => prepared.output_elements(),
        }
    }
}
fn ensure_warm_metadata_key(
    metadata: &WarmDecisionMetadata,
    key: WorkKey,
) -> Result<(), BackendError> {
    if metadata.key() == key {
        Ok(())
    } else {
        Err(start(
            "Qwen3.5 warm metadata was prepared for a different Work key",
        ))
    }
}
fn ensure_warm_metadata_key_external(
    metadata: &ExternalWarmMetadata,
    key: WorkKey,
) -> Result<(), BackendError> {
    if metadata.key() == key {
        Ok(())
    } else {
        Err(start(
            "Qwen3.5 external metadata was prepared for a different Work key",
        ))
    }
}

fn validate_external_attention(
    attention: crate::decision_external::ExternalAttention,
    block: bool,
) -> Result<(), BackendError> {
    use crate::decision_external::ExternalAttention;
    match (attention, block) {
        (ExternalAttention::PrefixBidirectional, true)
        | (ExternalAttention::Causal, false)
        | (ExternalAttention::NoncausalFull, false) => Ok(()),
        _ => Err(start(
            "External decision attention_mode differs from the actual backbone mode",
        )),
    }
}

/// Native sidecar precedence for the Qwen3.5 Decision route. A decision
/// config selects the external D3/Pplx head; the joint-head sidecar is Clef.
/// Otherwise the checkpoint is OneJev.
fn load_decision_head(
    path: &Path,
    context: &Arc<Context>,
    stream: &Arc<Stream>,
    block_mode: Option<Qwen35BlockMode>,
    weights: &crate::qwen35_weights::Qwen35DeviceWeights,
    fa4_root: &Path,
) -> Result<Qwen35DecisionHead, BackendError> {
    let decision = path.join("decision_config.json");
    let joint = path.join("joint_head_config.json");
    if crate::model::path_present(&decision).map_err(start)? {
        let head = ExternalGpuHead::load(path, context, stream)
            .map_err(|error| start(error.to_string()))?;
        validate_external_attention(head.config().attention, block_mode.is_some())?;
        return Ok(Qwen35DecisionHead::External(head));
    }
    if crate::model::path_present(&joint).map_err(start)? {
        if block_mode.is_some() {
            return Err(start("Clef does not support a block attention route"));
        }
        let head = ClefQ35Head::load(
            path,
            weights.config.hidden_size,
            context,
            stream,
            fa4_root,
            weights.lm_head.as_ref(),
        )?;
        return Ok(Qwen35DecisionHead::Clef(Box::new(head)));
    }
    let head =
        OneJevGpuHead::load(path, context, stream).map_err(|error| start(error.to_string()))?;
    crate::validate_decision_attention_mode(head.block_attention(), block_mode.is_some())?;
    Ok(Qwen35DecisionHead::OneJev(head))
}

impl Qwen35Model {
    pub(crate) fn new(
        host: Qwen35Weights,
        context: &Arc<Context>,
        stream: &Arc<Stream>,
        options: Qwen35TextOptions,
    ) -> Result<Self, BackendError> {
        Self::new_inner(host, context, stream, options, None, None)
    }

    pub(crate) fn new_with_block(
        host: Qwen35Weights,
        context: &Arc<Context>,
        stream: &Arc<Stream>,
        options: Qwen35TextOptions,
        prefix: usize,
    ) -> Result<Self, BackendError> {
        if prefix == 0 || options.causal {
            return Err(start(
                "Qwen3.5 block requires a nonzero prefix and noncausal attention",
            ));
        }
        Self::new_inner(
            host,
            context,
            stream,
            options,
            Some(Qwen35BlockMode::Fixed(prefix)),
            None,
        )
    }
    /// Private constructor integration point for the future Root whole-model /
    /// Decision gate. Ordinary serving construction passes `None` and keeps the
    /// existing text-only execution path; only the small fixed input metadata
    /// tail is reserved by staging for both paths.
    pub(crate) fn new_with_decision_head(
        host: Qwen35Weights,
        context: &Arc<Context>,
        stream: &Arc<Stream>,
        options: Qwen35TextOptions,
        decision_path: Option<&Path>,
    ) -> Result<Self, BackendError> {
        Self::new_inner(host, context, stream, options, None, decision_path)
    }

    /// Explicit B1 fixed-prefix block plus attached warm OneJev head. This is a
    /// private trial constructor: it is not reachable through the ordinary
    /// factory and enables no dynamic prefix, cross-request cache or admission.
    pub(crate) fn new_with_block_decision_head(
        host: Qwen35Weights,
        context: &Arc<Context>,
        stream: &Arc<Stream>,
        options: Qwen35TextOptions,
        prefix: usize,
        path: &Path,
    ) -> Result<Self, BackendError> {
        if prefix == 0 || options.causal {
            return Err(start(
                "Qwen3.5 block requires a nonzero prefix and noncausal attention",
            ));
        }
        Self::new_inner(
            host,
            context,
            stream,
            options,
            Some(Qwen35BlockMode::Fixed(prefix)),
            Some(path),
        )
    }

    /// Explicit B1 block plus attached warm OneJev head whose prefix is
    /// resolved per request against the same actual token count and retained
    /// allocation. This is a private trial constructor: no dynamic prefix is
    /// reachable through the ordinary factory and nothing is cached across
    /// requests beyond the Work's reused prefix-state scratch.
    pub(crate) fn new_with_dynamic_block_decision_head(
        host: Qwen35Weights,
        context: &Arc<Context>,
        stream: &Arc<Stream>,
        options: Qwen35TextOptions,
        path: &Path,
    ) -> Result<Self, BackendError> {
        if options.causal {
            return Err(start("Qwen3.5 dynamic block requires noncausal attention"));
        }
        Self::new_inner(
            host,
            context,
            stream,
            options,
            Some(Qwen35BlockMode::Dynamic),
            Some(path),
        )
    }

    fn new_inner(
        host: Qwen35Weights,
        context: &Arc<Context>,
        stream: &Arc<Stream>,
        options: Qwen35TextOptions,
        block_mode: Option<Qwen35BlockMode>,
        decision_path: Option<&Path>,
    ) -> Result<Self, BackendError> {
        host.config.validate_candidate()?;
        let spec = AttentionSpec {
            heads: host.config.num_attention_heads,
            kv_heads: host.config.num_key_value_heads,
            head_dim: host.config.head_dim,
            causal: options.causal,
            window: Window::Global,
            dtype: DType::Bf16,
            scale: Scale::RsqrtHeadDim,
        };
        if !Fa4::supports(&spec) {
            return Err(start(
                "Qwen3.5 actual attention geometry has no qualified FA4 export",
            ));
        }
        let (major, minor) = context.compute_capability().map_err(start)?;
        let capability = major
            .checked_mul(10)
            .and_then(|value| value.checked_add(minor))
            .ok_or_else(|| start("Qwen3.5 device capability overflow"))?;
        if block_mode.is_some() {
            BlockArchitecture::from_capability(capability)?;
        }
        let sm_count =
            i32::try_from(context.multiprocessor_count().map_err(start)?).map_err(start)?;
        let path = std::env::var_os(infinity_v3_config::infinity_env!("OXIDE_KERNEL_MODULE"))
            .map(PathBuf::from)
            .ok_or_else(|| start("Qwen3.5 requires a precompiled kernel module"))?;
        let path = if path.is_dir() {
            path.join(format!("sm_{major}{minor}"))
                .join("kernels.cubin")
        } else {
            path
        };
        let module = context
            .load_module_from_image(&std::fs::read(path).map_err(start)?)
            .map_err(start)?;
        let math = Qwen35Kernels::from_module(&module)?;
        // Bind every required name before reading the checkpoint payload.
        let aux = Qwen35AuxKernels {
            join: module
                .load_function("infinity_qwen35_join_ab_bf16")
                .map_err(start)?,
            silu: module
                .load_function("infinity_silu_mul_bf16_vector8_native")
                .map_err(start)?,
        };
        let gather = module
            .load_function("infinity_embedding_gather")
            .map_err(start)?;
        let pool = module.load_function("infinity_pool").map_err(start)?;
        let mean = module
            .load_function("infinity_mean_pool_bf16_native")
            .map_err(start)?;
        let raw = module
            .load_function("infinity_bf16_to_f32")
            .map_err(start)?;
        let fa4 = Fa4::new(Arc::clone(context), &options.fa4_root, spec)?;
        let tables = Qwen35Tables::new(
            stream,
            math.tables_initializer(),
            host.config.rope_theta,
            host.config.rotary_dim,
            host.config.max_position_embeddings,
        )?;
        let gdn_geometry = GdnGeometry::new(
            host.config.linear_num_key_heads,
            host.config.linear_num_value_heads,
            host.config.linear_conv_kernel_dim,
        )?;
        let weights = Qwen35DeviceWeights::upload(host, stream)?;
        let decision = decision_path
            .map(|path| {
                load_decision_head(
                    path,
                    context,
                    stream,
                    block_mode,
                    &weights,
                    &options.fa4_root,
                )
            })
            .transpose()?;
        Ok(Self {
            weights,
            options,
            gdn_geometry,
            major,
            sm_count,
            capability,
            block_mode,
            math,
            aux,
            fa4,
            tables,
            gather,
            pool,
            mean,
            raw,
            context: Arc::clone(context),
            stream: Arc::clone(stream),
            decision,
        })
    }
    pub(crate) const fn block_mode(&self) -> Option<Qwen35BlockMode> {
        self.block_mode
    }
    /// Legacy fixed-prefix accessor. Dynamic mode reports `None`; every block
    /// enabled predicate must use [`Self::block_mode`] instead.
    pub(crate) const fn block_prefix(&self) -> Option<usize> {
        match self.block_mode {
            Some(mode) => mode.fixed_prefix(),
            None => None,
        }
    }
    /// Exact GPU/host input word count for one Work allocation. It is the
    /// ordinary backbone layout plus the block tail (fixed or dynamic) and the
    /// warm head metadata tail that are actually attached to this model.
    pub(crate) fn input_capacity(&self, key: WorkKey) -> Result<usize, BackendError> {
        let mut words = if self.block_mode.is_some() {
            BLOCK_METADATA_WORDS
        } else {
            0
        };
        match self.decision.as_ref() {
            Some(Qwen35DecisionHead::OneJev(_)) => {
                words = words
                    .checked_add(crate::warm_head::metadata_words(key.1)?)
                    .ok_or_else(|| infer("Qwen3.5 input staging overflow"))?;
            }
            Some(Qwen35DecisionHead::External(_)) => {
                words = words
                    .checked_add(crate::external_warm::external_metadata_words(key.1)?)
                    .ok_or_else(|| infer("Qwen3.5 input staging overflow"))?;
            }
            Some(Qwen35DecisionHead::Clef(head)) => {
                words = words
                    .checked_add(crate::clef_warm::metadata_words(head.config(), key)?)
                    .ok_or_else(|| infer("Qwen3.5 input staging overflow"))?;
            }
            None => {}
        }
        staging::input_capacity_with_tail(key, words)
    }
    /// Pure CPU block admission before any request context bind or GPU work.
    /// Fixed mode keeps the exact prefix check; dynamic mode can only check
    /// the B1 single-sequence envelope because the prefix arrives per request.
    pub(crate) fn preflight(&self, batch: &Batch) -> Result<(), ShapeError> {
        let Some(mode) = self.block_mode else {
            return Ok(());
        };
        let tokens = batch.input_ids.len();
        validate_block_token_count(tokens).map_err(ShapeError::Rejected)?;
        if batch.cumulative_seq_lengths.len() != 2 {
            return Err(ShapeError::Rejected(infer(
                "Qwen3.5 block admits exactly one packed sequence",
            )));
        }
        if let Qwen35BlockMode::Fixed(prefix) = mode {
            if prefix == 0 || prefix > tokens {
                return Err(ShapeError::Rejected(infer(
                    "Qwen3.5 block prefix must be nonzero and at most the actual token count",
                )));
            }
            self.check_partial_prefix(prefix, tokens)?;
        }
        Ok(())
    }
    /// Resolve and check the actual request prefix against this backend's block
    /// mode before any request context is bound. Dynamic mode requires an
    /// explicit prefix; fixed mode accepts it absent or exact. SM100 keeps the
    /// existing partial-prefix rejection with no relaxed architecture policy.
    pub(crate) fn preflight_request_prefix(
        &self,
        batch: &Batch,
        prefix: Option<usize>,
    ) -> Result<(), ShapeError> {
        let Some(mode) = self.block_mode else {
            return Ok(());
        };
        let tokens = batch.input_ids.len();
        validate_block_token_count(tokens).map_err(ShapeError::Rejected)?;
        if batch.cumulative_seq_lengths.len() != 2 {
            return Err(ShapeError::Rejected(infer(
                "Qwen3.5 block admits exactly one packed sequence",
            )));
        }
        let prefix = mode
            .resolve_prefix(prefix, tokens)
            .map_err(ShapeError::Rejected)?;
        self.check_partial_prefix(prefix, tokens)
    }
    fn check_partial_prefix(&self, prefix: usize, tokens: usize) -> Result<(), ShapeError> {
        let architecture =
            BlockArchitecture::from_capability(self.capability).map_err(ShapeError::Rejected)?;
        if architecture == BlockArchitecture::Sm100 && prefix < tokens {
            return Err(ShapeError::Rejected(infer(
                "SM100 d256 FA4 does not admit a partial block prefix",
            )));
        }
        Ok(())
    }
    pub(crate) fn decision_capacity(
        &self,
        key: WorkKey,
    ) -> Result<Option<DecisionCapacity>, BackendError> {
        match self.decision.as_ref() {
            Some(Qwen35DecisionHead::OneJev(head)) => Ok(Some(DecisionCapacity::OneJev(
                WarmHeadCapacity::new(key.1, head.hidden_size())?,
            ))),
            Some(Qwen35DecisionHead::External(head)) => Ok(Some(DecisionCapacity::External(
                ExternalHeadCapacity::new(head.family(), key.1, head.hidden_size(), head.rows())?,
            ))),
            Some(Qwen35DecisionHead::Clef(head)) => Ok(Some(DecisionCapacity::Clef(
                crate::clef_warm::capacity(key, head.config())?,
            ))),
            None => Ok(None),
        }
    }
    pub(crate) fn onejev_head(&self) -> Option<&OneJevGpuHead> {
        match self.decision.as_ref() {
            Some(Qwen35DecisionHead::OneJev(head)) => Some(head),
            _ => None,
        }
    }
    pub(crate) fn external_head(&self) -> Option<&ExternalGpuHead> {
        match self.decision.as_ref() {
            Some(Qwen35DecisionHead::External(head)) => Some(head),
            _ => None,
        }
    }
    pub(crate) fn clef_head(&self) -> Option<&ClefQ35Head> {
        match self.decision.as_ref() {
            Some(Qwen35DecisionHead::Clef(head)) => Some(head.as_ref()),
            _ => None,
        }
    }
    /// Validate and plan one whole packed request against the actual attached
    /// head, the capacity for `key`, and the central block metadata reservation.
    /// No Work arena, cache slot, context bind or device address is touched.
    pub(crate) fn prepare_warm_metadata(
        &self,
        key: WorkKey,
        tokens: usize,
        batch: &PackedDecisionBatch<'_>,
        ids: &[u32],
        requests: &[DecisionInput],
    ) -> Result<WarmDecisionMetadataRoute, BackendError> {
        let reserved_words = if self.block_mode.is_some() {
            BLOCK_METADATA_WORDS
        } else {
            0
        };
        match self.decision.as_ref() {
            Some(Qwen35DecisionHead::OneJev(head)) => {
                let capacity = match self.decision_capacity(key)? {
                    Some(DecisionCapacity::OneJev(capacity)) => capacity,
                    _ => return Err(start("Qwen3.5 warm OneJev capacity is unavailable")),
                };
                crate::warm_head::prepare_warm_decision_metadata(
                    head.vocab_rows(),
                    head.hidden_size(),
                    capacity,
                    key,
                    tokens,
                    batch,
                    requests,
                    reserved_words,
                )
                .map(WarmDecisionMetadataRoute::OneJev)
                .map_err(|error| start(error.to_string()))
            }
            Some(Qwen35DecisionHead::External(head)) => {
                let capacity = match self.decision_capacity(key)? {
                    Some(DecisionCapacity::External(capacity)) => capacity,
                    _ => return Err(start("Qwen3.5 warm external capacity is unavailable")),
                };
                crate::external_warm::prepare_external_warm_metadata(
                    head.family(),
                    capacity,
                    key,
                    tokens,
                    batch,
                    requests,
                    reserved_words,
                )
                .map(WarmDecisionMetadataRoute::External)
            }
            Some(Qwen35DecisionHead::Clef(head)) => {
                let capacity = match self.decision_capacity(key)? {
                    Some(DecisionCapacity::Clef(capacity)) => capacity,
                    _ => return Err(start("Qwen3.5 warm Clef capacity is unavailable")),
                };
                let prepared = head.prepare(batch, ids, requests, &self.stream)?;
                if prepared.metadata_words() > capacity.metadata_words() {
                    return Err(infer(
                        "Qwen3.5 Clef metadata exceeds the retained Work capacity",
                    ));
                }
                let layout_end = crate::staging::input_capacity(key)?
                    .checked_add(reserved_words)
                    .and_then(|base| base.checked_add(capacity.metadata_words()))
                    .ok_or_else(|| infer("Qwen3.5 Clef metadata tail overflow"))?;
                Ok(WarmDecisionMetadataRoute::Clef {
                    prepared,
                    layout_end,
                })
            }
            None => Err(start("Qwen3.5 warm decision head is not attached")),
        }
    }
    /// Bind already validated metadata to this Work's actual retained scratch.
    /// Every owner and address is read from the Work; no cache, arena or
    /// context state is changed.
    pub(crate) fn bind_warm_metadata(
        &self,
        metadata: WarmDecisionMetadataRoute,
        work: &Qwen35Work,
    ) -> Result<WarmDecisionRoute, BackendError> {
        match metadata {
            WarmDecisionMetadataRoute::OneJev(metadata) => {
                let head = self
                    .onejev_head()
                    .ok_or_else(|| start("Qwen3.5 warm OneJev head is not attached"))?;
                ensure_warm_metadata_key(&metadata, work.key())?;
                let capacity = match self.decision_capacity(metadata.key())? {
                    Some(DecisionCapacity::OneJev(capacity)) => capacity,
                    _ => return Err(start("Qwen3.5 warm OneJev capacity is unavailable")),
                };
                if metadata.total_rows() > capacity.rows() {
                    return Err(infer(
                        "Qwen3.5 warm metadata exceeds the actual Work row capacity",
                    ));
                }
                let scratch = work
                    .onejev_head()
                    .ok_or_else(|| infer("Qwen3.5 Work has no warm OneJev scratch"))?;
                let selected_rows = metadata
                    .total_rows()
                    .checked_mul(head.hidden_size())
                    .ok_or_else(|| infer("Qwen3.5 warm selected scratch overflow"))?;
                if scratch.logits_len() < metadata.total_rows()
                    || scratch.selected_len() < selected_rows
                {
                    return Err(infer(
                        "Qwen3.5 warm metadata exceeds the actual Work scratch",
                    ));
                }
                crate::warm_head::bind_warm_decision_metadata(metadata, scratch.bases())
                    .map(WarmDecisionRoute::OneJev)
                    .map_err(|error| start(error.to_string()))
            }
            WarmDecisionMetadataRoute::External(metadata) => {
                ensure_warm_metadata_key_external(&metadata, work.key())?;
                if work.external_head().is_none() {
                    return Err(infer("Qwen3.5 Work has no warm external scratch"));
                }
                crate::external_warm::bind_external_warm_metadata(metadata)
                    .map(WarmDecisionRoute::External)
            }
            WarmDecisionMetadataRoute::Clef { prepared, .. } => {
                if work.clef_head().is_none() {
                    return Err(infer("Qwen3.5 Work has no warm Clef workspace"));
                }
                Ok(WarmDecisionRoute::Clef(Box::new(prepared)))
            }
        }
    }
    pub(crate) fn context(&self) -> &Arc<Context> {
        &self.context
    }
    pub(crate) fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }
    /// Rebind one whole packed request against this Work's reusable scratch.
    /// This allocates nothing and queries no device.
    pub(crate) fn plan_warm(
        &self,
        _key: WorkKey,
        tokens: usize,
        batch: &PackedDecisionBatch<'_>,
        ids: &[u32],
        requests: &[DecisionInput],
        work: &Qwen35Work,
    ) -> Result<WarmDecisionRoute, BackendError> {
        let metadata = self.prepare_warm_metadata(work.key(), tokens, batch, ids, requests)?;
        self.bind_warm_metadata(metadata, work)
    }
    /// Crate-private enqueue from this Work's actual normalized BF16 activation.
    ///
    /// # Safety
    /// The enclosing engine owns the bound context/stream and keeps this model,
    /// Work, head and uploads alive through its completion/quarantine boundary.
    pub(crate) unsafe fn enqueue_warm(
        &self,
        work: &mut Qwen35Work,
        blas: &Blas,
        stream: &Stream,
        plan: &mut WarmDecisionRoute,
        host_logits: &mut [f32],
    ) -> Result<(), BackendError> {
        match (plan, self.decision.as_ref()) {
            (WarmDecisionRoute::OneJev(plan), Some(Qwen35DecisionHead::OneJev(head))) => {
                // SAFETY: the caller's outer engine ownership covers completion.
                unsafe { work.enqueue_warm(head, blas, stream, plan, host_logits) }
                    .map_err(|error| start(error.to_string()))
            }
            (WarmDecisionRoute::External(plan), Some(Qwen35DecisionHead::External(head))) => {
                // SAFETY: the caller's outer engine ownership covers completion.
                unsafe { work.enqueue_external(head, blas, stream, plan, host_logits) }
                    .map_err(|error| start(error.to_string()))
            }
            (WarmDecisionRoute::Clef(prepared), Some(Qwen35DecisionHead::Clef(head))) => {
                // SAFETY: the caller's outer engine ownership covers completion.
                unsafe {
                    work.enqueue_clef(
                        head,
                        &self.weights.embed,
                        blas,
                        stream,
                        prepared,
                        host_logits,
                    )
                }
                .map_err(|error| start(error.to_string()))
            }
            _ => Err(start(
                "Qwen3.5 warm plan and attached decision head do not agree",
            )),
        }
    }
    pub(crate) fn workspace_bytes(&self, key: WorkKey) -> Result<usize, BackendError> {
        Qwen35Work::allocation_bytes(key, self)
    }
    pub(crate) fn workspace(
        &self,
        stream: Arc<Stream>,
        key: WorkKey,
    ) -> Result<Qwen35Work, BackendError> {
        Qwen35Work::new(stream, self, key)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn enqueue(
        &self,
        blas: &Blas,
        stream: &Stream,
        work: &mut Qwen35Work,
        batch: &Batch,
        pool: &Pool,
        max_len: usize,
        tokens: usize,
    ) -> Result<(), BackendError> {
        let sequences = batch
            .cumulative_seq_lengths
            .len()
            .checked_sub(1)
            .ok_or_else(|| infer("Qwen3.5 missing sequence boundaries"))?;
        let c = &self.weights.config;
        let h = c.hidden_size;
        let rows = u32_size(tokens)?;
        let width = u32_size(h)?;
        // Use the layout already staged into this Work for the actual (T, P).
        // Never rebuild it from a model-wide prefix: the same Work can stage a
        // different prefix for the same actual token count.
        let block_layout = match (self.block_mode, work.block()) {
            (Some(mode), Some(block)) => {
                let layout = block.layout()?;
                if layout.tokens() != tokens {
                    return Err(infer(
                        "Qwen3.5 staged block layout token count differs from the request",
                    ));
                }
                if let Qwen35BlockMode::Fixed(prefix) = mode {
                    if layout.prefix_tokens() != prefix {
                        return Err(infer(
                            "Qwen3.5 staged block layout prefix differs from the fixed model prefix",
                        ));
                    }
                }
                Some((layout, block.metadata_base()))
            }
            (Some(_), None) => {
                return Err(infer("Qwen3.5 block Work is missing prefix metadata state"))
            }
            (None, _) => None,
        };
        let data = &mut *work.data;
        let base = data.input.cu_deviceptr();
        let input_layout = uploaded_input_layout(tokens, batch)?;
        let positions = input_layout.positions().address(base)?;
        let boundaries = input_layout.boundaries().address(base)?;
        let selected = input_layout.selections().address(base)?;
        // A block plan is a property of the actual shape, actual prefix and the
        // Work-capacity-based metadata tail addresses it was built against.
        let shape = block_layout
            .as_ref()
            .map_or((tokens, sequences, max_len), |(layout, metadata_base)| {
                (tokens, layout.prefix_tokens(), *metadata_base)
            });
        if data.prepared_shape != Some(shape) {
            data.prepared_shape = None;
            data.prepared = Some(match block_layout.as_ref() {
                Some((layout, metadata_base)) => {
                    let word_address = |word: usize| {
                        base.checked_add(product(word, 4)? as u64)
                            .ok_or_else(|| infer("Qwen3.5 block metadata pointer overflow"))
                    };
                    let prefix = layout.prefix_cu_range(*metadata_base)?;
                    let suffix = layout.suffix_cu_range(*metadata_base)?;
                    let architecture = BlockArchitecture::from_capability(self.capability)?;
                    PreparedWork::new_block(
                        &self.weights,
                        &data.layout,
                        &data.slabs(),
                        tokens,
                        positions,
                        boundaries,
                        sequences,
                        max_len,
                        layout.prefix_tokens(),
                        architecture,
                        word_address(prefix.start)?,
                        word_address(suffix.start)?,
                        boundaries,
                    )?
                }
                None => PreparedWork::new(
                    &self.weights,
                    &data.layout,
                    &data.slabs(),
                    tokens,
                    positions,
                    boundaries,
                    sequences,
                    max_len,
                )?,
            });
            data.prepared_shape = Some(shape);
        }
        let mut ids = base;
        let mut table = self.weights.embed.cu_deviceptr();
        let mut residual = data.residual.cu_deviceptr();
        let mut r = rows;
        let mut w = width;
        launch(
            &self.gather,
            stream,
            blocks(product(tokens, h)?)?,
            0,
            &mut [
                arg(&mut ids),
                arg(&mut table),
                arg(&mut residual),
                arg(&mut r),
                arg(&mut w),
            ],
        )?;
        let Payload {
            input,
            residual,
            projection,
            activation,
            gdn,
            block,
            prepared,
            layout: slab_layout,
            ..
        } = data;
        let uploaded_cumulative = crate::gdn_prepared::GdnBufferView::from_buffer_range(
            input,
            product(tokens, 3)?
                ..product(tokens, 3)?
                    .checked_add(batch.cumulative_seq_lengths.len())
                    .ok_or_else(|| infer("Qwen3.5 cumulative view overflow"))?,
        )?;
        let slabs = crate::qwen35_execute::Slabs {
            residual,
            projection,
            activation,
        };
        // SAFETY: the common Engine owns this Work/model/provider/table through
        // its one completion boundary and quarantines the whole Work on errors.
        unsafe {
            let prepared = prepared
                .as_ref()
                .ok_or_else(|| infer("Qwen3.5 missing prepared frame"))?;
            if let Some(block) = block.as_ref() {
                let staged = block.layout()?;
                let output = slab_layout
                    .block_linear_output()
                    .ok_or_else(|| infer("Qwen3.5 block activation has no combined output"))?
                    .row_span(tokens)?;
                let combined = output.offset()..output.offset() + output.elements();
                prepared.enqueue_block(
                    &self.weights,
                    blas,
                    &self.math,
                    &self.aux,
                    &self.fa4,
                    stream,
                    &slabs,
                    gdn,
                    &batch.cumulative_seq_lengths,
                    uploaded_cumulative,
                    self.tables.cosine.cu_deviceptr(),
                    self.tables.sine.cu_deviceptr(),
                    u32_size(self.tables.pairs)?,
                    input,
                    block.prefix_state(),
                    &staged,
                    block.metadata_base(),
                    combined,
                )?;
            } else {
                prepared.enqueue(
                    &self.weights,
                    blas,
                    &self.math,
                    &self.aux,
                    &self.fa4,
                    stream,
                    &slabs,
                    gdn,
                    &batch.cumulative_seq_lengths,
                    uploaded_cumulative,
                    self.tables.cosine.cu_deviceptr(),
                    self.tables.sine.cu_deviceptr(),
                    u32_size(self.tables.pairs)?,
                )?;
            }
        }
        let mut normalized = data.activation.cu_deviceptr();
        let mut bounds = boundaries;
        let mut indices = selected;
        let mut out = data.pooled.cu_deviceptr();
        let mut count = u32_size(batch.pooled_indices.len())?;
        let mut pool_width = width;
        if count > 0 {
            if *pool == Pool::Mean && batch.pooled_indices.len() <= u16::MAX as usize {
                launch(
                    &self.mean,
                    stream,
                    (count, u32_size(h.div_ceil(8))?, 1),
                    0,
                    &mut [
                        arg(&mut normalized),
                        arg(&mut bounds),
                        arg(&mut indices),
                        arg(&mut out),
                        arg(&mut count),
                        arg(&mut pool_width),
                    ],
                )?;
            } else {
                let mut mode = match pool {
                    Pool::Cls => 0u32,
                    Pool::Mean => 1,
                    Pool::LastToken => 2,
                    Pool::Splade => return Err(infer("Qwen3.5 SPLADE unsupported")),
                };
                launch(
                    &self.pool,
                    stream,
                    blocks(product(count as usize, h)?)?,
                    0,
                    &mut [
                        arg(&mut normalized),
                        arg(&mut bounds),
                        arg(&mut indices),
                        arg(&mut out),
                        arg(&mut count),
                        arg(&mut pool_width),
                        arg(&mut mode),
                    ],
                )?;
            }
        }
        if !batch.raw_indices.is_empty() {
            crate::enqueue_raw_output(
                &self.raw,
                stream,
                normalized,
                data.pooled.cu_deviceptr(),
                input_layout.trailer().address(base)?,
                batch,
                h,
            )?;
        }
        Ok(())
    }
}

fn uploaded_input_layout(
    tokens: usize,
    batch: &Batch,
) -> Result<crate::input_layout::InputLayout, BackendError> {
    crate::input_layout::InputLayout::new(
        tokens,
        batch.cumulative_seq_lengths.len(),
        batch.pooled_indices.len(),
    )?
    .with_trailer(crate::output::raw_rows(batch)?)
}

#[cfg(test)]
mod warm_root_tests {
    use super::*;
    use crate::decision_head::PackedDecisionBatch;
    use crate::decision_head_gpu::GpuHiddenRows;
    use infinity_v3_backend_core::{DecisionInput, Pool};
    use std::path::PathBuf;

    #[test]
    fn mixed_raw_selection_pointer_uses_staged_token_trailer_after_pooled_ids() {
        let mut batch = decision_batch(&[2, 3]);
        batch.pooled_indices = vec![1, 0, 1];
        batch.raw_indices = vec![1, 0, 1];
        let layout = uploaded_input_layout(5, &batch).unwrap();
        let mut words = vec![0; layout.words()];
        let (pooled, raw) = words.split_at_mut(layout.packed_words());
        crate::pack_input_with_position_offset(
            pooled,
            5,
            &batch.input_ids,
            &batch.position_ids,
            &batch.token_type_ids,
            &batch.cumulative_seq_lengths,
            &batch.pooled_indices,
            0,
        )
        .unwrap();
        crate::output::write_raw_indices(&batch, raw).unwrap();
        assert_eq!(
            &words[layout.selections().get()..layout.trailer().get()],
            &[1, 0, 1]
        );
        assert_eq!(&words[layout.trailer().get()..], &[2, 3, 4, 0, 1, 2, 3, 4]);
        assert_eq!(
            layout.trailer().address(0x1000).unwrap()
                - layout.selections().address(0x1000).unwrap(),
            12
        );
    }

    #[test]
    fn warm_metadata_key_mismatch_is_rejected_before_binding() {
        let key = (32, 4, 2);
        let batch = PackedDecisionBatch::new(&[0, 3, 5]).unwrap();
        let requests = [option(&[4, 1]), DecisionInput::Warmup];
        let metadata = crate::warm_head::prepare_warm_decision_metadata(
            8,
            4,
            WarmHeadCapacity::new(4, 4).unwrap(),
            key,
            batch.tokens(),
            &batch,
            &requests,
            0,
        )
        .unwrap();
        assert!(ensure_warm_metadata_key(&metadata, key).is_ok());
        assert!(ensure_warm_metadata_key(&metadata, (32, 4, 3)).is_err());
        assert!(ensure_warm_metadata_key(&metadata, (32, 8, 2)).is_err());
    }

    fn source(name: &str) -> PathBuf {
        let path = std::env::var_os(name).unwrap_or_else(|| panic!("missing {name}"));
        let path = PathBuf::from(path);
        assert!(path.exists(), "missing {name}: {}", path.display());
        path
    }

    fn option(token_ids: &[u32]) -> DecisionInput {
        DecisionInput::OptionTokens {
            token_ids: token_ids.to_vec(),
        }
    }

    fn decision_batch(lengths: &[u32]) -> Batch {
        let mut cumulative = vec![0u32];
        let mut input_ids = Vec::new();
        let mut position_ids = Vec::new();
        for &length in lengths {
            for position in 0..length {
                input_ids.push(20 + (input_ids.len() as u32 * 13) % 20_000);
                position_ids.push(position);
            }
            cumulative.push(input_ids.len() as u32);
        }
        Batch {
            multimodal: Vec::new(),
            token_type_ids: vec![0; input_ids.len()],
            input_ids,
            position_ids,
            cumulative_seq_lengths: cumulative,
            max_length: *lengths.iter().max().unwrap(),
            pooled_indices: Vec::new(),
            raw_indices: Vec::new(),
            compact_input_ids: None,
            compact_position_ids: None,
            scatter_unfold: None,
            fold_gather: None,
            tokens: Vec::new(),
            offsets: Vec::new(),
        }
    }

    fn assert_decision_counts(
        outputs: &[infinity_v3_backend_core::DecisionOutput],
        counts: &[usize],
    ) {
        assert_eq!(outputs.len(), counts.len());
        for (output, count) in outputs.iter().zip(counts) {
            assert_eq!(output.logits.len(), *count);
            assert_eq!(output.action_probability, 1.0);
            assert!(output.memory.is_none());
            assert!(output.logits.iter().all(|value| value.is_finite()));
        }
    }

    /// Root-only bounded async decision lifecycle: two held requests must use
    /// the two existing host slots, complete out of order, release on drop, and
    /// survive a Work shape change without per-request GPU allocation.
    #[test]
    #[ignore = "Requires CUDA Qwen3.5 kernels/FA4, INFINITY_QWEN35_WARM_MODEL and QWEN35_CONTROL_* sources"]
    fn root_warm_decision_holds_a_b_drop_and_shape_lifecycle() {
        let model_path = source("INFINITY_QWEN35_WARM_MODEL");
        let options = Qwen35TextOptions {
            causal: false,
            fa4_root: source("QWEN35_CONTROL_FA4_ROOT"),
            gdn_raw_kernels: source("QWEN35_CONTROL_GDN_RAW_KERNELS"),
            gdn_provider: source("QWEN35_CONTROL_GDN_PROVIDER"),
            gdn_aux_kernels: source("QWEN35_CONTROL_GDN_AUX_KERNELS"),
        };
        let backend =
            crate::OxideBackend::new_qwen35_text_decision_candidate(&model_path, 0, options)
                .unwrap();
        let a = backend
            .submit_warm_decision(
                decision_batch(&[3, 5]),
                vec![option(&[2, 1, 2]), option(&[3, 4])],
            )
            .unwrap();
        let b = backend
            .submit_warm_decision(
                decision_batch(&[2, 4, 3]),
                vec![option(&[5, 6]), option(&[7]), option(&[8, 9, 10])],
            )
            .unwrap();
        assert!(matches!(&a, crate::Inference::Deferred(_)));
        assert!(matches!(&b, crate::Inference::Deferred(_)));
        assert_eq!(backend.io.active_leases(), 2);
        let b = b.complete().unwrap();
        assert_decision_counts(&b, &[2, 1, 3]);

        // Shape growth is submitted while A is still held in the other slot.
        // Any Work replacement must preserve A's request-specific host owner.
        let large = backend
            .submit_warm_decision(
                decision_batch(&[40, 20]),
                vec![option(&[4, 3, 2]), option(&[1])],
            )
            .unwrap();
        assert!(matches!(&large, crate::Inference::Deferred(_)));
        assert_eq!(backend.io.active_leases(), 2);
        let large = large.complete().unwrap();
        assert_decision_counts(&large, &[3, 1]);
        let a = a.complete().unwrap();
        assert_decision_counts(&a, &[3, 2]);
        assert_eq!(backend.io.active_leases(), 0);

        // A dropped deferred request returns its slot without materializing.
        let dropped = backend
            .submit_warm_decision(decision_batch(&[3]), vec![option(&[1, 2])])
            .unwrap();
        assert!(matches!(&dropped, crate::Inference::Deferred(_)));
        assert_eq!(backend.io.active_leases(), 1);
        drop(dropped);
        assert_eq!(backend.io.active_leases(), 0);

        // Return to the smaller shape class and complete with exact counts.
        let small = backend
            .submit_warm_decision(decision_batch(&[2]), vec![option(&[9, 8])])
            .unwrap()
            .complete()
            .unwrap();
        assert_decision_counts(&small, &[2]);
        assert_eq!(backend.io.active_leases(), 0);
    }

    /// Root-only whole-model readout gate: run the real Qwen3.5 backbone into a
    /// Work arena, then compare the engine-bound warm path against the isolated
    /// reviewed prepared head on the same normalized activation span.
    #[test]
    #[ignore = "Requires CUDA Qwen3.5 kernels/FA4, INFINITY_QWEN35_WARM_MODEL and QWEN35_CONTROL_* sources"]
    fn root_whole_model_warm_readout_matches_isolated_prepared_head() {
        let model_path = source("INFINITY_QWEN35_WARM_MODEL");
        let options = Qwen35TextOptions {
            causal: false,
            fa4_root: source("QWEN35_CONTROL_FA4_ROOT"),
            gdn_raw_kernels: source("QWEN35_CONTROL_GDN_RAW_KERNELS"),
            gdn_provider: source("QWEN35_CONTROL_GDN_PROVIDER"),
            gdn_aux_kernels: source("QWEN35_CONTROL_GDN_AUX_KERNELS"),
        };
        let host = Qwen35Weights::load(&model_path).unwrap();
        let hidden = host.config.hidden_size;
        let context = Context::new(0).unwrap();
        let stream = context.new_stream().unwrap();
        let blas = Blas::new(context.clone(), stream.clone()).unwrap();
        let model = Qwen35Model::new_with_decision_head(
            host,
            &context,
            &stream,
            options,
            Some(&model_path),
        )
        .unwrap();
        let lengths = [3u32, 5u32];
        let mut cumulative = vec![0u32];
        let mut input_ids = Vec::new();
        let mut position_ids = Vec::new();
        for &length in &lengths {
            for position in 0..length {
                input_ids.push(20 + (input_ids.len() as u32 * 13) % 20_000);
                position_ids.push(position);
            }
            cumulative.push(input_ids.len() as u32);
        }
        let tokens = input_ids.len();
        let batch = Batch {
            multimodal: Vec::new(),
            token_type_ids: vec![0; tokens],
            input_ids: input_ids.clone(),
            position_ids: position_ids.clone(),
            cumulative_seq_lengths: cumulative.clone(),
            max_length: *lengths.iter().max().unwrap(),
            pooled_indices: Vec::new(),
            raw_indices: Vec::new(),
            compact_input_ids: None,
            compact_position_ids: None,
            scatter_unfold: None,
            fold_gather: None,
            tokens: Vec::new(),
            offsets: Vec::new(),
        };
        let key = crate::work_key(&batch).unwrap();
        let mut work = model.workspace(stream.clone(), key).unwrap();
        let packed = PackedDecisionBatch::new(&cumulative).unwrap();
        let requests = [option(&[2, 1, 2]), option(&[3, 4])];
        let mut route = model
            .plan_warm(key, tokens, &packed, &input_ids, &requests, &work)
            .unwrap();
        let mut host_input = vec![
            0u32;
            crate::staging::input_capacity_with_tail(
                key,
                crate::warm_head::metadata_words(key.1).unwrap(),
            )
            .unwrap()
        ];
        let layout = crate::input_layout::InputLayout::new(tokens, cumulative.len(), 0)
            .unwrap()
            .packed_words();
        crate::pack_input_with_position_offset(
            &mut host_input[..layout],
            tokens,
            &input_ids,
            &position_ids,
            &batch.token_type_ids,
            &cumulative,
            &[],
            0,
        )
        .unwrap();
        let upload = match &route {
            WarmDecisionRoute::OneJev(plan) => plan.pack_metadata(&mut host_input).unwrap(),
            _ => panic!("fixture must attach OneJev"),
        };
        work.upload(&host_input, upload).unwrap();
        model
            .enqueue(
                &blas,
                &stream,
                &mut work,
                &batch,
                &Pool::Mean,
                batch.max_length as usize,
                tokens,
            )
            .unwrap();
        let mut host_logits = vec![0f32; route.total_rows()];
        // SAFETY: this test owns the model, Work, head, uploads and stream.
        unsafe { model.enqueue_warm(&mut work, &blas, &stream, &mut route, &mut host_logits) }
            .unwrap();
        stream.synchronize().unwrap();
        let warm = match &route {
            WarmDecisionRoute::OneJev(plan) => {
                crate::warm_head::finish_warm_decision(plan, &host_logits).unwrap()
            }
            _ => panic!("fixture must attach OneJev"),
        };
        let isolated = model.onejev_head().unwrap();
        let mut prepared = isolated.prepare(&packed, &requests).unwrap();
        let reference = isolated
            .forward(
                GpuHiddenRows::new(work.activation(), tokens, hidden).unwrap(),
                &packed,
                &requests,
                &mut prepared,
                &blas,
                &stream,
            )
            .unwrap();
        assert_eq!(warm.len(), reference.len());
        for (warm, reference) in warm.iter().zip(&reference) {
            assert_eq!(warm.logits.len(), reference.logits.len());
            for (warm, reference) in warm.logits.iter().zip(&reference.logits) {
                assert!((warm - reference).abs() <= 1e-5 * reference.abs().max(1.0));
            }
        }
    }
}
