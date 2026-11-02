// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//!
//! Native Gemma4 OptionHead planning and warm enqueue.
//!
//! The source contract is `libtorch/cpp/models/decision_models.cpp:205-330`.
//! For a tied lexical checkpoint the head reads the last real BF16 hidden row
//! of each packed sequence, gathers the requested vocabulary rows in request
//! order (duplicates included), runs one M=1, N=option_count BF16 projection
//! per sequence, and only then applies the publisher softcap staging:
//!
//! `tanh(BF16(logits / cap)) * cap` with a BF16 rounding after each Torch
//! scalar operation, followed by a final BF16-to-F32 widening.
//!
//! This module owns cold request metadata, host-only GEMM plans and bounded
//! per-request device scratch. It never reads checkpoint payloads, never
//! uploads a second vocabulary matrix, and never queries device state from the
//! warm path. The canonical BF16 embedding buffer remains owned by the E2B
//! executor's existing `WeightUploader` result.
//!
//! The candidate is crate-private and has no public Engine route yet; accessor
//! and lifecycle helpers are exercised by host tests and the ignored GPU
//! readiness cases before that integration is admitted.
#![allow(dead_code)]

use crate::cublas::Blas;
use crate::gemma4_text_model::{Gemma4Contract, Gemma4LmHeadPlan, Gemma4Weights};
use crate::math::LinearPlan;
use crate::staging::HostIoLease;
use crate::tensor::Bf16;
use crate::{arg, blocks, download_output_range, infer, launch, product, start, u32_size};
use cuda_core::{DeviceBuffer, Function, PinnedHostBuffer, Stream};
use infinity_v3_backend_core::{result_memory, BackendError, Batch, DecisionInput, DecisionOutput};
use safetensors::Dtype;
use std::fmt;
use std::mem::ManuallyDrop;
use std::sync::{atomic::Ordering, Arc};

/// The option-row gather uses the published unscaled BF16 bit-copy export.
/// Its layout is the contiguous `[vocab_size, hidden]` canonical embedding.
/// Published dense E2B `final_logit_softcapping` value admitted by this
/// candidate. The Native OptionHead source uses ATen scalar divide/tanh/mul;
/// the separate Native CUDA sequence helper uses a reciprocal multiply. Those
/// recipes are not claimed equivalent at BF16 ties until the Root oracle proves
/// it, so admission is deliberately limited to the actual E2B cap.
pub(crate) const E2B_OPTION_SOFCAP: f32 = 30.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct OptionSoftcap {
    cap: f32,
}

impl OptionSoftcap {
    /// Admit the typed finite-positive E2B 30 contract.
    pub(crate) fn e2b(cap: f64) -> Result<Self, BackendError> {
        if !cap.is_finite() || cap <= 0.0 {
            return Err(start("Gemma4 option softcap must be finite and positive"));
        }
        if cap != f64::from(E2B_OPTION_SOFCAP) {
            return Err(start(format!(
                "Gemma4 E2B option softcap must be {E2B_OPTION_SOFCAP}, found {cap:?}"
            )));
        }
        Ok(Self { cap: cap as f32 })
    }

    pub(crate) fn value(self) -> f32 {
        self.cap
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum OptionPlanError {
    InvalidCumulative(&'static str),
    RequestCountMismatch {
        requests: usize,
        sequences: usize,
    },
    UnsupportedRequest(&'static str),
    EmptyOptions {
        request: usize,
    },
    TokenIdOutOfRange {
        request: usize,
        token_id: u32,
        vocab_rows: usize,
    },
    CountOverflow,
    LayoutOverflow,
    DimensionOverflow(&'static str),
}

impl fmt::Display for OptionPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCumulative(reason) => {
                write!(formatter, "invalid packed option boundaries: {reason}")
            }
            Self::RequestCountMismatch {
                requests,
                sequences,
            } => write!(
                formatter,
                "option request count {requests} does not match {sequences} packed sequences"
            ),
            Self::UnsupportedRequest(reason) => {
                write!(formatter, "unsupported Gemma4 option request: {reason}")
            }
            Self::EmptyOptions { request } => {
                write!(
                    formatter,
                    "Gemma4 option request {request} has no option tokens"
                )
            }
            Self::TokenIdOutOfRange {
                request,
                token_id,
                vocab_rows,
            } => write!(
                formatter,
                "Gemma4 option request {request} token id {token_id} is outside 0..{vocab_rows}"
            ),
            Self::CountOverflow => formatter.write_str("Gemma4 option count overflow"),
            Self::LayoutOverflow => formatter.write_str("Gemma4 option layout size overflow"),
            Self::DimensionOverflow(what) => {
                write!(
                    formatter,
                    "Gemma4 option {what} exceeds the prepared plan range"
                )
            }
        }
    }
}

impl std::error::Error for OptionPlanError {}

impl From<OptionPlanError> for BackendError {
    fn from(error: OptionPlanError) -> Self {
        start(error)
    }
}

/// Physical hidden-row contract for the per-sequence M=1 projection.
///
/// Gemma4 text/Rune production uses `RawPacked`: hidden rows are the full packed
/// token stream and each sequence reads its last row `offsets[i+1]-1`.
/// `Pooled` binds a real pre-selected `[batch, hidden]` matrix and reads row `i`
/// for sequence `i`, matching Native `forward_pooled` without host expansion or
/// test-only field mutation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OptionInputLayout {
    RawPacked,
    Pooled,
}

/// Cold, immutable row plan shared by CPU admission tests and the warm device
/// enqueue. `row_ids` preserves request order and duplicates exactly.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct OptionRequestPlan {
    tokens: usize,
    hidden: usize,
    hidden_rows: usize,
    layout: OptionInputLayout,
    vocab_rows: usize,
    counts: Vec<usize>,
    row_offsets: Vec<usize>,
    row_ids: Vec<u32>,
    boundaries: Vec<u32>,
    last_rows: Vec<u32>,
    total_options: usize,
    max_options: usize,
}

impl OptionRequestPlan {
    pub(crate) fn tokens(&self) -> usize {
        self.tokens
    }

    pub(crate) fn hidden(&self) -> usize {
        self.hidden
    }

    pub(crate) fn hidden_rows(&self) -> usize {
        self.hidden_rows
    }

    pub(crate) fn layout(&self) -> OptionInputLayout {
        self.layout
    }

    pub(crate) fn vocab_rows(&self) -> usize {
        self.vocab_rows
    }

    pub(crate) fn counts(&self) -> &[usize] {
        &self.counts
    }

    pub(crate) fn row_offsets(&self) -> &[usize] {
        &self.row_offsets
    }

    pub(crate) fn row_ids(&self) -> &[u32] {
        &self.row_ids
    }

    pub(crate) fn boundaries(&self) -> &[u32] {
        &self.boundaries
    }

    pub(crate) fn last_rows(&self) -> &[u32] {
        &self.last_rows
    }

    pub(crate) fn total_options(&self) -> usize {
        self.total_options
    }

    pub(crate) fn max_options(&self) -> usize {
        self.max_options
    }
}

fn checked_add(left: usize, right: usize) -> Result<usize, OptionPlanError> {
    left.checked_add(right)
        .ok_or(OptionPlanError::CountOverflow)
}

fn checked_product(
    left: usize,
    right: usize,
    what: &'static str,
) -> Result<usize, OptionPlanError> {
    left.checked_mul(right)
        .ok_or(OptionPlanError::DimensionOverflow(what))
}

/// Validate the exact Native `OptionHead` request contract and build a cold
/// per-sequence M=1 row plan. Warmup reads vocabulary row zero exactly like
/// Native; ordinary options remain in caller order with duplicates preserved.
pub(crate) fn plan_option_requests(
    vocab_rows: usize,
    hidden: usize,
    cumulative_seq_lengths: &[u32],
    tokens: usize,
    requests: &[DecisionInput],
) -> Result<OptionRequestPlan, OptionPlanError> {
    plan_option_requests_with_layout(
        vocab_rows,
        hidden,
        cumulative_seq_lengths,
        tokens,
        requests,
        OptionInputLayout::RawPacked,
    )
}

pub(crate) fn plan_option_requests_with_layout(
    vocab_rows: usize,
    hidden: usize,
    cumulative_seq_lengths: &[u32],
    tokens: usize,
    requests: &[DecisionInput],
    layout: OptionInputLayout,
) -> Result<OptionRequestPlan, OptionPlanError> {
    if vocab_rows == 0 || hidden == 0 {
        return Err(OptionPlanError::DimensionOverflow(
            "vocabulary or hidden width",
        ));
    }
    if cumulative_seq_lengths.len() < 2 {
        return Err(OptionPlanError::InvalidCumulative(
            "at least one sequence is required",
        ));
    }
    if cumulative_seq_lengths[0] != 0 {
        return Err(OptionPlanError::InvalidCumulative(
            "the first boundary must be zero",
        ));
    }
    for pair in cumulative_seq_lengths.windows(2) {
        if pair[1] <= pair[0] {
            return Err(OptionPlanError::InvalidCumulative(
                "every sequence must be nonempty and strictly increasing",
            ));
        }
    }
    let last = usize::try_from(
        *cumulative_seq_lengths
            .last()
            .ok_or(OptionPlanError::InvalidCumulative("missing final boundary"))?,
    )
    .map_err(|_| OptionPlanError::InvalidCumulative("boundary exceeds usize"))?;
    if last != tokens {
        return Err(OptionPlanError::InvalidCumulative(
            "the final boundary must equal the hidden token-row count",
        ));
    }
    let sequences = cumulative_seq_lengths.len() - 1;
    if requests.len() != sequences {
        return Err(OptionPlanError::RequestCountMismatch {
            requests: requests.len(),
            sequences,
        });
    }
    if hidden > i32::MAX as usize {
        return Err(OptionPlanError::DimensionOverflow("hidden width"));
    }

    let mut counts = Vec::new();
    counts
        .try_reserve_exact(requests.len())
        .map_err(|_| OptionPlanError::LayoutOverflow)?;
    let mut row_offsets = Vec::new();
    row_offsets
        .try_reserve_exact(requests.len() + 1)
        .map_err(|_| OptionPlanError::LayoutOverflow)?;
    row_offsets.push(0usize);
    let mut row_ids = Vec::new();
    let mut last_rows = Vec::new();
    last_rows
        .try_reserve_exact(requests.len())
        .map_err(|_| OptionPlanError::LayoutOverflow)?;
    let mut total_options = 0usize;
    let mut max_options = 0usize;

    for (request_index, request) in requests.iter().enumerate() {
        let count = match request {
            DecisionInput::Warmup => {
                row_ids
                    .try_reserve(1)
                    .map_err(|_| OptionPlanError::LayoutOverflow)?;
                row_ids.push(0);
                1usize
            }
            DecisionInput::OptionTokens { token_ids } => {
                if token_ids.is_empty() {
                    return Err(OptionPlanError::EmptyOptions {
                        request: request_index,
                    });
                }
                for &token_id in token_ids {
                    if usize::try_from(token_id).map_or(true, |id| id >= vocab_rows) {
                        return Err(OptionPlanError::TokenIdOutOfRange {
                            request: request_index,
                            token_id,
                            vocab_rows,
                        });
                    }
                }
                let count = token_ids.len();
                if count > i32::MAX as usize {
                    return Err(OptionPlanError::DimensionOverflow(
                        "option count for one sequence",
                    ));
                }
                row_ids
                    .try_reserve_exact(count)
                    .map_err(|_| OptionPlanError::LayoutOverflow)?;
                row_ids.extend_from_slice(token_ids);
                count
            }
            DecisionInput::BlockOptionTokens { .. } => {
                return Err(OptionPlanError::UnsupportedRequest(
                    "BlockOptionTokens is not the Gemma4 option_tokens contract",
                ))
            }
            DecisionInput::InlineOptions { .. } => {
                return Err(OptionPlanError::UnsupportedRequest(
                    "InlineOptions is not the Gemma4 option_tokens contract",
                ))
            }
            DecisionInput::Laya { .. } => {
                return Err(OptionPlanError::UnsupportedRequest(
                    "Laya is not the Gemma4 option_tokens contract",
                ))
            }
            DecisionInput::Clef { .. } => {
                return Err(OptionPlanError::UnsupportedRequest(
                    "Clef is not the Gemma4 option_tokens contract",
                ))
            }
        };
        counts.push(count);
        max_options = max_options.max(count);
        total_options = checked_add(total_options, count)?;
        row_offsets.push(total_options);
        let sequence_end = usize::try_from(cumulative_seq_lengths[request_index + 1])
            .map_err(|_| OptionPlanError::InvalidCumulative("boundary exceeds usize"))?;
        let last_row = match layout {
            OptionInputLayout::RawPacked => {
                sequence_end
                    .checked_sub(1)
                    .ok_or(OptionPlanError::InvalidCumulative(
                        "a sequence has no last row",
                    ))?
            }
            OptionInputLayout::Pooled => request_index,
        };
        last_rows.push(
            u32::try_from(last_row)
                .map_err(|_| OptionPlanError::InvalidCumulative("last row exceeds u32"))?,
        );
    }

    let boundaries = {
        let mut bound = Vec::new();
        bound
            .try_reserve_exact(cumulative_seq_lengths.len())
            .map_err(|_| OptionPlanError::LayoutOverflow)?;
        bound.extend_from_slice(cumulative_seq_lengths);
        bound
    };
    let hidden_rows = match layout {
        OptionInputLayout::RawPacked => tokens,
        OptionInputLayout::Pooled => sequences,
    };
    Ok(OptionRequestPlan {
        tokens,
        hidden,
        hidden_rows,
        layout,
        vocab_rows,
        counts,
        row_offsets,
        row_ids,
        boundaries,
        last_rows,
        total_options,
        max_options,
    })
}

/// Bounded per-request scratch. The gather buffer holds at most the largest
/// single-sequence option list because gather and GEMM are stream-ordered per
/// sequence. The BF16 logits, F32 output and device ids cover the exact flat
/// request total.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct OptionScratchLayout {
    pub(crate) gather_rows: usize,
    pub(crate) total_options: usize,
}

impl OptionScratchLayout {
    pub(crate) fn for_plan(plan: &OptionRequestPlan) -> Result<Self, OptionPlanError> {
        let gather_rows = checked_product(
            plan.max_options,
            plan.hidden,
            "selected vocabulary row extent",
        )?;
        if plan.total_options == 0 || gather_rows == 0 {
            return Err(OptionPlanError::DimensionOverflow("option scratch"));
        }
        Ok(Self {
            gather_rows,
            total_options: plan.total_options,
        })
    }

    pub(crate) fn bytes(self) -> Result<usize, BackendError> {
        let gather = product(self.gather_rows, std::mem::size_of::<Bf16>())?;
        let logits = product(self.total_options, std::mem::size_of::<Bf16>())?;
        let output = product(self.total_options, std::mem::size_of::<f32>())?;
        let ids = product(self.total_options, std::mem::size_of::<u32>())?;
        gather
            .checked_add(logits)
            .and_then(|bytes| bytes.checked_add(output))
            .and_then(|bytes| bytes.checked_add(ids))
            .ok_or_else(|| infer("Gemma4 option scratch byte size overflow"))
    }
}

/// Cold prepared option head. It consumes only the canonical tied embedding
/// uploaded by the E2B executor and never owns or reloads vocabulary rows.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Gemma4OptionHead {
    hidden: usize,
    vocab_rows: usize,
    cap: Option<OptionSoftcap>,
}

impl Gemma4OptionHead {
    /// Admit the tied lexical BF16 head for the Rune option contract.
    pub(crate) fn from_weights(weights: &Gemma4Weights) -> Result<Self, BackendError> {
        if weights.config.contract != Gemma4Contract::RuneOptionTokens {
            return Err(start(
                "Gemma4 option head requires the RuneOptionTokens contract",
            ));
        }
        if !matches!(weights.lm_head_plan, Gemma4LmHeadPlan::TiedEmbedding { .. }) {
            return Err(start(
                "Gemma4 option head requires tied BF16 embed_tokens weights",
            ));
        }
        if weights.dtype != Dtype::BF16 {
            return Err(start(format!(
                "Gemma4 option head requires BF16 embed_tokens, found {:?}",
                weights.dtype
            )));
        }
        let hidden = weights.config.hidden_size;
        let vocab_rows = weights.config.vocab_size;
        if hidden == 0 || vocab_rows == 0 {
            return Err(start(
                "Gemma4 option head requires a nonzero vocabulary and hidden width",
            ));
        }
        let expected = vocab_rows
            .checked_mul(hidden)
            .ok_or_else(|| start("Gemma4 option embedding size overflow"))?;
        if weights.embed.len() != expected {
            return Err(start(
                "Gemma4 option head embedding size disagrees with its vocabulary geometry",
            ));
        }
        let cap = match weights.plan.readout.final_logit_softcap {
            Some(cap) => Some(OptionSoftcap::e2b(cap)?),
            None => None,
        };
        Ok(Self {
            hidden,
            vocab_rows,
            cap,
        })
    }

    pub(crate) fn hidden_size(&self) -> usize {
        self.hidden
    }

    pub(crate) fn vocab_rows(&self) -> usize {
        self.vocab_rows
    }

    pub(crate) fn cap(&self) -> Option<OptionSoftcap> {
        self.cap
    }

    /// Build all cold metadata and host-only GEMM plans for one request batch.
    /// This performs no device allocation, upload, query, sync or heuristic.
    pub(crate) fn prepare(
        &self,
        batch: &Batch,
        requests: &[DecisionInput],
    ) -> Result<PreparedGemma4OptionHead, BackendError> {
        self.prepare_with_layout(batch, requests, OptionInputLayout::RawPacked)
    }

    /// Prepare against an explicit physical hidden layout. Production Gemma4
    /// uses `RawPacked`; `Pooled` is for a real cold `[batch, hidden]` input.
    pub(crate) fn prepare_with_layout(
        &self,
        batch: &Batch,
        requests: &[DecisionInput],
        layout: OptionInputLayout,
    ) -> Result<PreparedGemma4OptionHead, BackendError> {
        let plan = plan_option_requests_with_layout(
            self.vocab_rows,
            self.hidden,
            &batch.cumulative_seq_lengths,
            batch.input_ids.len(),
            requests,
            layout,
        )?;
        let layout = OptionScratchLayout::for_plan(&plan)?;
        let mut plans = Vec::new();
        plans
            .try_reserve_exact(plan.counts.len())
            .map_err(|error| infer(format!("Gemma4 option plan allocation: {error}")))?;
        for &count in &plan.counts {
            plans.push(
                LinearPlan::<Bf16>::contiguous(1, self.hidden, count)
                    .map_err(|error| infer(format!("Gemma4 option M1 plan: {error}")))?,
            );
        }
        Ok(PreparedGemma4OptionHead {
            plan,
            plans,
            layout,
            cap: self.cap,
        })
    }
}

/// Kernel handles needed by the warm option enqueue. The executor supplies
/// references to the already-loaded canonical module functions.
pub(crate) struct OptionKernels<'a> {
    pub(crate) gather_bf16: &'a Function,
    pub(crate) widen_bf16: &'a Function,
    pub(crate) softcap_bf16_to_f32: Option<&'a Function>,
}

/// One admitted option batch ready for warm enqueue.
pub(crate) struct PreparedGemma4OptionHead {
    plan: OptionRequestPlan,
    plans: Vec<LinearPlan<Bf16>>,
    layout: OptionScratchLayout,
    cap: Option<OptionSoftcap>,
}

impl PreparedGemma4OptionHead {
    pub(crate) fn tokens(&self) -> usize {
        self.plan.tokens
    }

    pub(crate) fn hidden_rows(&self) -> usize {
        self.plan.hidden_rows
    }

    pub(crate) fn input_layout(&self) -> OptionInputLayout {
        self.plan.layout
    }

    pub(crate) fn counts(&self) -> &[usize] {
        &self.plan.counts
    }

    pub(crate) fn total_options(&self) -> usize {
        self.plan.total_options
    }

    pub(crate) fn last_rows(&self) -> &[u32] {
        &self.plan.last_rows
    }

    pub(crate) fn row_ids(&self) -> &[u32] {
        &self.plan.row_ids
    }

    pub(crate) fn layout(&self) -> OptionScratchLayout {
        self.layout
    }

    pub(crate) fn fits(&self, needed: OptionScratchLayout) -> bool {
        self.layout.gather_rows >= needed.gather_rows
            && self.layout.total_options >= needed.total_options
    }

    /// True when this prepared head was built for the exact packed batch.
    pub(crate) fn matches_batch(&self, batch: &Batch) -> bool {
        self.plan.tokens == batch.input_ids.len()
            && self.plan.boundaries.as_slice() == batch.cumulative_seq_lengths
            && self.plan.counts.len() == batch.cumulative_seq_lengths.len().saturating_sub(1)
    }

    /// True when the leased host tail already holds this prepared row plan.
    pub(crate) fn host_ids_match(
        &self,
        host: &PinnedHostBuffer<u32>,
        offset: usize,
    ) -> Result<bool, BackendError> {
        let end = offset
            .checked_add(self.plan.row_ids.len())
            .ok_or_else(|| infer("Gemma4 option host id range overflow"))?;
        Ok(host.as_slice().get(offset..end) == Some(self.plan.row_ids.as_slice()))
    }

    /// True when every request's exact option IDs/order/duplicates match the
    /// prepared flat row plan. No allocation or cloning is performed.
    pub(crate) fn matches_requests(&self, requests: &[DecisionInput]) -> bool {
        if requests.len() != self.plan.counts.len() {
            return false;
        }
        for (index, request) in requests.iter().enumerate() {
            let (Some(&start), Some(&end)) = (
                self.plan.row_offsets.get(index),
                self.plan.row_offsets.get(index + 1),
            ) else {
                return false;
            };
            let expected = match request {
                DecisionInput::Warmup => &[0u32][..],
                DecisionInput::OptionTokens { token_ids } => token_ids.as_slice(),
                _ => return false,
            };
            if self.plan.row_ids.get(start..end) != Some(expected) {
                return false;
            }
        }
        true
    }

    pub(crate) fn cap(&self) -> Option<OptionSoftcap> {
        self.cap
    }

    /// Validate every request, buffer extent, context/stream identity and
    /// kernel prerequisite before the first H2D or backbone launch. This is
    /// metadata-only and makes no allocation, query, launch or synchronization.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn preflight(
        &self,
        blas: &Blas,
        stream: &Stream,
        hidden: &DeviceBuffer<Bf16>,
        embed: &DeviceBuffer<Bf16>,
        work: &Gemma4OptionWork,
        ids_host: &PinnedHostBuffer<u32>,
        ids_offset: usize,
        kernels: &OptionKernels<'_>,
    ) -> Result<(), BackendError> {
        if !blas.uses_stream(stream) {
            return Err(infer(
                "Gemma4 option enqueue requires the plain cuBLAS stream",
            ));
        }
        if !work.uses_stream(stream) {
            return Err(infer("Gemma4 option Work belongs to a different stream"));
        }
        for (name, context) in [
            ("hidden", hidden.context()),
            ("embedding", embed.context()),
            ("work", work.context()),
        ] {
            if !blas.uses_context(context) {
                return Err(infer(format!(
                    "Gemma4 option {name} belongs to a different CUDA context"
                )));
            }
        }
        if !work.fits(self.layout) {
            return Err(infer(
                "Gemma4 option work scratch does not match the prepared request",
            ));
        }
        work.can_begin()?;
        if !Arc::ptr_eq(ids_host.context(), stream.context()) {
            return Err(infer(
                "Gemma4 option id staging belongs to a different CUDA context",
            ));
        }
        if ids_offset
            .checked_add(self.plan.total_options)
            .is_none_or(|end| end > ids_host.len())
        {
            return Err(infer(
                "Gemma4 option id staging exceeds the leased HostIo input arena",
            ));
        }
        let hidden_elements = product(self.plan.hidden_rows, self.plan.hidden)?;
        if hidden.len() < hidden_elements {
            return Err(infer(
                "Gemma4 option hidden source is shorter than the packed request",
            ));
        }
        let embedding_elements = product(self.plan.vocab_rows, self.plan.hidden)?;
        if embed.len() < embedding_elements {
            return Err(infer(
                "Gemma4 canonical tied embedding is shorter than the vocabulary",
            ));
        }
        if self.cap.is_some() && kernels.softcap_bf16_to_f32.is_none() {
            return Err(start(
                "Gemma4 capped option head requires the additive softcap export",
            ));
        }
        if self.plans.len() != self.plan.counts.len()
            || self.plan.row_offsets.len() != self.plan.counts.len() + 1
            || self.plan.last_rows.len() != self.plan.counts.len()
        {
            return Err(infer(
                "Gemma4 option prepared metadata disagrees with its request count",
            ));
        }
        for &count in &self.plan.counts {
            u32_size(count)?;
        }
        u32_size(self.plan.hidden)?;
        u32_size(self.plan.total_options)?;
        Ok(())
    }

    /// Enqueue one ordered gather + M=1 GEMM per sequence and one aggregate
    /// flat softcap/widen launch. The caller retains `hidden` (the final
    /// normalized BF16 rows), `embed` (the canonical tied embedding) and
    /// `work` until the stream event/drain. No synchronization or allocation
    /// occurs here.
    ///
    /// # Safety
    /// `hidden`, `embed` and `work` must belong to the same bound CUDA context
    /// and ordered stream as `blas`; `hidden` must already contain the final
    /// normalized BF16 rows produced by the same stream, and every owner must
    /// stay alive until the completion event or drain.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn enqueue(
        &self,
        blas: &Blas,
        stream: &Stream,
        hidden: &DeviceBuffer<Bf16>,
        embed: &DeviceBuffer<Bf16>,
        work: &mut Gemma4OptionWork,
        ids_host: &PinnedHostBuffer<u32>,
        ids_offset: usize,
        kernels: &OptionKernels<'_>,
    ) -> Result<(), BackendError> {
        self.preflight(
            blas, stream, hidden, embed, work, ids_host, ids_offset, kernels,
        )?;
        work.upload_ids_from(ids_host, ids_offset, self.plan.total_options)?;
        let hidden_width = u32_size(self.plan.hidden)?;
        for (index, &count) in self.plan.counts.iter().enumerate() {
            let count_u32 = u32_size(count)?;
            let row_offset = *self
                .plan
                .row_offsets
                .get(index)
                .ok_or_else(|| infer("Gemma4 option row offset missing"))?;
            let mut ids = work
                .ids
                .cu_deviceptr()
                .checked_add(
                    u64::try_from(product(row_offset, std::mem::size_of::<u32>())?)
                        .map_err(infer)?,
                )
                .ok_or_else(|| infer("Gemma4 option id address overflow"))?;
            let mut table = embed.cu_deviceptr();
            let mut gathered = work.rows.cu_deviceptr();
            let mut tokens = count_u32;
            let mut width = hidden_width;
            launch(
                kernels.gather_bf16,
                stream,
                blocks(product(count, self.plan.hidden)?)?,
                0,
                &mut [
                    arg(&mut ids),
                    arg(&mut table),
                    arg(&mut gathered),
                    arg(&mut tokens),
                    arg(&mut width),
                ],
            )?;
            let last_row = *self
                .plan
                .last_rows
                .get(index)
                .ok_or_else(|| infer("Gemma4 option last row missing"))?;
            let hidden_offset =
                product(usize::try_from(last_row).map_err(infer)?, self.plan.hidden)?;
            let plan = self
                .plans
                .get(index)
                .ok_or_else(|| infer("Gemma4 option M1 plan missing"))?;
            // SAFETY: stream order makes each row gather visible to its
            // following GEMM. The canonical owners and bounded Work scratch
            // stay alive through the caller's event/drain contract.
            unsafe {
                blas.linear_subrange(
                    plan,
                    hidden,
                    hidden_offset,
                    &work.rows,
                    0,
                    &mut work.logits,
                    row_offset,
                )?;
            }
        }
        let total_options = self.plan.total_options;
        let mut count = u32_size(total_options)?;
        let mut input = work.logits.cu_deviceptr();
        let mut output = work.output.cu_deviceptr();
        match (self.cap, kernels.softcap_bf16_to_f32) {
            (Some(cap), Some(kernel)) => {
                let mut cap = cap.value();
                launch(
                    kernel,
                    stream,
                    blocks(total_options)?,
                    0,
                    &mut [
                        arg(&mut input),
                        arg(&mut output),
                        arg(&mut count),
                        arg(&mut cap),
                    ],
                )?;
            }
            (None, _) => {
                let mut selected = 0u64;
                let mut rows = 1u32;
                let mut width = count;
                launch(
                    kernels.widen_bf16,
                    stream,
                    blocks(total_options)?,
                    0,
                    &mut [
                        arg(&mut input),
                        arg(&mut output),
                        arg(&mut selected),
                        arg(&mut rows),
                        arg(&mut width),
                    ],
                )?;
            }
            (Some(_), None) => {
                return Err(start(
                    "Gemma4 capped option head requires the additive softcap export",
                ))
            }
        }
        Ok(())
    }

    /// Bound production path: validate the prepared IDs already staged in the
    /// request's HostIoLease input tail, begin lease retention before the first
    /// H2D, and then call the raw enqueue. The lease remains the pinned source
    /// owner until its completion event/drain.
    ///
    /// # Safety
    /// Same owner-lifetime contract as [`Self::enqueue`]. The caller must not
    /// return the lease slot before the recorded completion event drains.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn enqueue_from_lease(
        &self,
        blas: &Blas,
        stream: &Stream,
        hidden: &DeviceBuffer<Bf16>,
        embed: &DeviceBuffer<Bf16>,
        work: &mut Gemma4OptionWork,
        lease: &mut HostIoLease,
        ids_offset: usize,
        kernels: &OptionKernels<'_>,
    ) -> Result<(), BackendError> {
        if !lease.uses_stream(stream) {
            return Err(infer(
                "Gemma4 option HostIo lease belongs to a different stream",
            ));
        }
        {
            let ids_host = &lease.slot()?.input;
            if !self.host_ids_match(ids_host, ids_offset)? {
                return Err(infer(
                    "Gemma4 option leased host tail does not match the prepared IDs",
                ));
            }
            self.preflight(
                blas, stream, hidden, embed, work, ids_host, ids_offset, kernels,
            )?;
        }
        lease.begin();
        let ids_host = &lease.slot()?.input;
        // SAFETY: the caller's unsafe owner contract and the now-begun lease
        // retain the exact pinned source through the ordered H2D and completion.
        unsafe {
            self.enqueue(
                blas, stream, hidden, embed, work, ids_host, ids_offset, kernels,
            )
        }
    }
}

/// Copy the exact flat per-request logit payload into Native-shaped
/// `DecisionOutput` values. Each output count comes from the admitted request
/// plan; duplicates and request order are already represented in the flat
/// buffer. Native OptionHead returns action probability one.
pub(crate) fn pack_option_outputs(
    output: &[f32],
    counts: &[usize],
) -> Result<Vec<DecisionOutput>, BackendError> {
    let total = counts.iter().try_fold(0usize, |total, &count| {
        total
            .checked_add(count)
            .ok_or_else(|| infer("Gemma4 option output count overflow"))
    })?;
    if output.len() != total {
        return Err(infer(
            "Gemma4 option flat output length disagrees with request counts",
        ));
    }
    let bytes = total
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| infer("Gemma4 option result memory size overflow"))?;
    let memory = result_memory::global_pool()?.reserve(bytes)?;
    let mut results = Vec::new();
    results
        .try_reserve_exact(counts.len())
        .map_err(|error| infer(format!("Gemma4 option output allocation: {error}")))?;
    let mut rest = output;
    for &count in counts {
        let (values, tail) = rest
            .split_at_checked(count)
            .ok_or_else(|| infer("Gemma4 option output row exceeds the flat payload"))?;
        let mut logits = Vec::new();
        logits
            .try_reserve_exact(count)
            .map_err(|error| infer(format!("Gemma4 option logit allocation: {error}")))?;
        logits.extend_from_slice(values);
        results.push(DecisionOutput {
            logits,
            action_probability: 1.0,
            memory: Some(memory.clone()),
        });
        rest = tail;
    }
    if !rest.is_empty() {
        return Err(infer("Gemma4 option output has trailing logit data"));
    }
    Ok(results)
}

impl PreparedGemma4OptionHead {
    /// Record one D2H of the exact flat logit payload followed by the completion
    /// event. The caller must not reuse `work` or `lease` until that event (or
    /// the Work drop drain) completes. No per-sequence synchronization occurs.
    pub(crate) fn finish_rune(
        &self,
        stream: &Stream,
        work: &mut Gemma4OptionWork,
        lease: &mut HostIoLease,
    ) -> Result<(), BackendError> {
        if !work.fits(self.layout) {
            return Err(infer(
                "Gemma4 option work output does not match the prepared request",
            ));
        }
        if !work.uses_stream(stream) {
            return Err(infer(
                "Gemma4 option finish requires the Work allocation stream",
            ));
        }
        if !lease.uses_stream(stream) {
            return Err(infer(
                "Gemma4 option finish requires the HostIo lease stream",
            ));
        }
        if !work.state.needs_output_copy() {
            return Err(infer(
                "Gemma4 option finish requires a freshly enqueued Work",
            ));
        }
        let host = lease.slot_mut()?.output.as_mut_slice();
        download_output_range(stream, &work.output, 0, host, 0, self.plan.total_options)?;
        lease.record()?;
        work.state.mark_output_copied().map_err(infer)
    }

    /// Pack the drained flat logit payload into Native-shaped outputs.
    pub(crate) fn outputs(&self, lease: &HostIoLease) -> Result<Vec<DecisionOutput>, BackendError> {
        let output = lease.output(self.plan.total_options)?;
        pack_option_outputs(output, &self.plan.counts)
    }
}

/// Explicit reuse state for one bounded Work. A drain proves the previous
/// stream uses finished; the next upload/launch must clear it before any GPU
/// operation, otherwise Drop could free storage still referenced by CUDA.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct RetentionState {
    used: bool,
    output_copied: bool,
    drained: bool,
}

impl RetentionState {
    /// Pure readiness check shared by cold preflight and the actual enqueue.
    fn can_begin(self) -> Result<(), &'static str> {
        if self.used && !self.drained && !self.output_copied {
            return Err("Gemma4 option Work output D2H is not yet enqueued");
        }
        Ok(())
    }

    /// A shared Work may be reused only after the previous request's output D2H
    /// was enqueued (the host lease remains live) or after the stream drained.
    fn begin_use(&mut self) -> Result<(), &'static str> {
        self.can_begin()?;
        self.used = true;
        self.output_copied = false;
        self.drained = false;
        Ok(())
    }

    fn mark_output_copied(&mut self) -> Result<(), &'static str> {
        if !self.used || self.output_copied {
            return Err("Gemma4 option Work has no pending output D2H");
        }
        self.output_copied = true;
        Ok(())
    }

    fn mark_drained(&mut self) {
        self.drained = true;
    }

    fn is_drained(self) -> bool {
        self.drained
    }

    fn needs_output_copy(self) -> bool {
        self.used && !self.output_copied
    }
}

/// Checked tail offset inside an existing HostIoLease input arena. The layout
/// is the canonical packed input prefix (tokens*3 + sequences) followed by the
/// request's own option-ID staging words. No GPU geometry is inflated by O.
pub(crate) fn option_id_staging_offset(
    tokens: usize,
    boundaries: usize,
    option_ids: usize,
) -> Result<(usize, usize), BackendError> {
    let prefix = tokens
        .checked_mul(3)
        .and_then(|value| value.checked_add(boundaries))
        .ok_or_else(|| infer("Gemma4 option id staging offset overflow"))?;
    let end = prefix
        .checked_add(option_ids)
        .ok_or_else(|| infer("Gemma4 option id staging end overflow"))?;
    Ok((prefix, end))
}

/// Stage the prepared option IDs into this request's retained HostIoLease input
/// tail. The host key must reserve exactly `option_ids.len()` tail words
/// (`key.2`); the GPU Work remains keyed independently and allocates no
/// per-option pooled buffer. The lease must be retained until its request's
/// completion event/drain, which is what allows two ordered A/B submissions.
pub(crate) fn write_option_id_tail(
    host: &mut [u32],
    key: (usize, usize, usize),
    option_ids: &[u32],
) -> Result<usize, BackendError> {
    let (tokens, boundaries, reserved) = key;
    if reserved != option_ids.len() {
        return Err(infer(
            "Gemma4 option host key tail reservation disagrees with the prepared IDs",
        ));
    }
    let capacity = crate::staging::input_capacity(key)?;
    let (offset, end) = option_id_staging_offset(tokens, boundaries, option_ids.len())?;
    if end != capacity {
        return Err(infer(
            "Gemma4 option host tail does not end at the reserved input capacity",
        ));
    }
    let tail = host
        .get_mut(offset..end)
        .ok_or_else(|| infer("Gemma4 option host id tail exceeds the leased input arena"))?;
    tail.copy_from_slice(option_ids);
    Ok(offset)
}

pub(crate) fn stage_option_ids(
    lease: &mut HostIoLease,
    key: (usize, usize, usize),
    option_ids: &[u32],
) -> Result<usize, BackendError> {
    write_option_id_tail(lease.slot_mut()?.input.as_mut_slice(), key, option_ids)
}

struct OptionOwners {
    rows: Option<DeviceBuffer<Bf16>>,
    logits: Option<DeviceBuffer<Bf16>>,
    output: Option<DeviceBuffer<f32>>,
    ids: Option<DeviceBuffer<u32>>,
}

impl OptionOwners {
    fn retain_if_uncertain(mut self, stream: &Stream) {
        if stream.synchronize().is_ok() {
            return;
        }
        let mut bytes = 0usize;
        if let Some(rows) = self.rows.take() {
            bytes = bytes.saturating_add(rows.num_bytes());
            std::mem::forget(rows);
        }
        if let Some(logits) = self.logits.take() {
            bytes = bytes.saturating_add(logits.num_bytes());
            std::mem::forget(logits);
        }
        if let Some(output) = self.output.take() {
            bytes = bytes.saturating_add(output.num_bytes());
            std::mem::forget(output);
        }
        if let Some(ids) = self.ids.take() {
            bytes = bytes.saturating_add(ids.num_bytes());
            std::mem::forget(ids);
        }
        metrics::counter!("oxide_arena_retained_bytes_total").increment(bytes as u64);
    }
}

/// Bounded option scratch owned by one request. The canonical embedding is
/// intentionally absent: it stays owned by the E2B executor's WeightUploader
/// result and is borrowed only for the ordered gather launches.
pub(crate) struct Gemma4OptionWork {
    stream: Arc<Stream>,
    layout: OptionScratchLayout,
    bytes: usize,
    rows: ManuallyDrop<DeviceBuffer<Bf16>>,
    logits: ManuallyDrop<DeviceBuffer<Bf16>>,
    output: ManuallyDrop<DeviceBuffer<f32>>,
    ids: ManuallyDrop<DeviceBuffer<u32>>,
    state: RetentionState,
}

impl Gemma4OptionWork {
    pub(crate) fn new(
        stream: Arc<Stream>,
        layout: OptionScratchLayout,
    ) -> Result<Self, BackendError> {
        let bytes = layout.bytes()?;
        let mut owners = OptionOwners {
            rows: None,
            logits: None,
            output: None,
            ids: None,
        };
        let mut allocate = || -> Result<(), BackendError> {
            // SAFETY: every buffer is fully written before any read on the same
            // ordered stream. The caller retains this Work through completion.
            owners.rows = Some(
                unsafe { DeviceBuffer::<Bf16>::uninitialized_async(&stream, layout.gather_rows) }
                    .map_err(infer)?,
            );
            owners.logits = Some(
                unsafe { DeviceBuffer::<Bf16>::uninitialized_async(&stream, layout.total_options) }
                    .map_err(infer)?,
            );
            owners.output = Some(
                unsafe { DeviceBuffer::<f32>::uninitialized_async(&stream, layout.total_options) }
                    .map_err(infer)?,
            );
            owners.ids = Some(
                unsafe { DeviceBuffer::<u32>::uninitialized_async(&stream, layout.total_options) }
                    .map_err(infer)?,
            );
            Ok(())
        };
        if let Err(error) = allocate() {
            owners.retain_if_uncertain(&stream);
            return Err(error);
        }
        let rows = owners
            .rows
            .ok_or_else(|| infer("Gemma4 option row scratch missing"))?;
        let logits = owners
            .logits
            .ok_or_else(|| infer("Gemma4 option logit scratch missing"))?;
        let output = owners
            .output
            .ok_or_else(|| infer("Gemma4 option output scratch missing"))?;
        let ids = owners
            .ids
            .ok_or_else(|| infer("Gemma4 option id scratch missing"))?;
        crate::LIVE_ARENA_BYTES.fetch_add(bytes, Ordering::Relaxed);
        crate::record_live_arena_bytes();
        Ok(Self {
            stream,
            layout,
            bytes,
            rows: ManuallyDrop::new(rows),
            logits: ManuallyDrop::new(logits),
            output: ManuallyDrop::new(output),
            ids: ManuallyDrop::new(ids),
            state: RetentionState::default(),
        })
    }

    pub(crate) fn layout(&self) -> OptionScratchLayout {
        self.layout
    }

    pub(crate) fn fits(&self, needed: OptionScratchLayout) -> bool {
        self.layout.gather_rows >= needed.gather_rows
            && self.layout.total_options >= needed.total_options
    }

    pub(crate) fn can_begin(&self) -> Result<(), BackendError> {
        self.state.can_begin().map_err(infer)
    }

    pub(crate) fn output(&self) -> &DeviceBuffer<f32> {
        &self.output
    }

    pub(crate) fn context(&self) -> &Arc<cuda_core::Context> {
        self.rows.context()
    }

    pub(crate) fn uses_stream(&self, stream: &Stream) -> bool {
        std::ptr::eq(self.stream.as_ref(), stream)
    }

    fn upload_ids_from(
        &mut self,
        host: &PinnedHostBuffer<u32>,
        offset: usize,
        total_options: usize,
    ) -> Result<(), BackendError> {
        // Any new upload/launch invalidates a previous drain proof before the
        // first GPU-visible byte of this use. A later request may reuse this
        // shared GPU Work only after the prior request's gather is stream-ordered
        // ahead of the next H2D; each request still retains its own host lease.
        self.state.begin_use().map_err(infer)?;
        if total_options > self.layout.total_options {
            return Err(infer("Gemma4 option ids exceed the retained Work capacity"));
        }
        if !Arc::ptr_eq(host.context(), self.stream.context()) {
            return Err(infer(
                "Gemma4 option id staging belongs to a different CUDA context",
            ));
        }
        let end = offset
            .checked_add(total_options)
            .ok_or_else(|| infer("Gemma4 option id staging range overflow"))?;
        let ids = host
            .as_slice()
            .get(offset..end)
            .ok_or_else(|| infer("Gemma4 option id staging exceeds the leased input arena"))?;
        // SAFETY: the request's own pinned lease slot stays owned and
        // unmodified until that request's event/drain; the H2D copy precedes
        // every gather on the ordered stream, and a later request's H2D is
        // ordered after this request's gather.
        unsafe {
            cuda_core::memory::memcpy_htod_async(
                self.ids.cu_deviceptr(),
                ids.as_ptr(),
                product(ids.len(), std::mem::size_of::<u32>())?,
                self.stream.cu_stream(),
            )
        }
        .map_err(infer)
    }

    /// # Safety
    /// The owning stream must have completed every launch using this scratch.
    pub(crate) unsafe fn mark_stream_drained(&mut self) {
        self.state.mark_drained();
    }
}

impl Drop for Gemma4OptionWork {
    fn drop(&mut self) {
        if self.state.is_drained() || self.stream.synchronize().is_ok() {
            unsafe {
                ManuallyDrop::drop(&mut self.rows);
                ManuallyDrop::drop(&mut self.logits);
                ManuallyDrop::drop(&mut self.output);
                ManuallyDrop::drop(&mut self.ids);
            }
            crate::LIVE_ARENA_BYTES.fetch_sub(self.bytes, Ordering::Relaxed);
            crate::record_live_arena_bytes();
        } else {
            metrics::counter!("oxide_arena_retained_bytes_total").increment(self.bytes as u64);
            let rows = unsafe { ManuallyDrop::take(&mut self.rows) };
            let logits = unsafe { ManuallyDrop::take(&mut self.logits) };
            let output = unsafe { ManuallyDrop::take(&mut self.output) };
            let ids = unsafe { ManuallyDrop::take(&mut self.ids) };
            std::mem::forget(rows);
            std::mem::forget(logits);
            std::mem::forget(output);
            std::mem::forget(ids);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use infinity_v3_backend_core::DecisionInput;
    use sha2::Digest;

    type PairPlan = (Vec<u32>, Vec<usize>);

    fn actual_path() -> &'static str {
        "/workspace/model-performance/michaelfeil/onejev-perf/backbone-comparison-20261023/models/gemma-4-E2B"
    }

    fn batch(boundaries: Vec<u32>, tokens: usize) -> Batch {
        Batch {
            multimodal: Vec::new(),
            input_ids: vec![1; tokens],
            token_type_ids: vec![0; tokens],
            position_ids: (0..tokens as u32).collect(),
            cumulative_seq_lengths: boundaries,
            max_length: 1,
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

    fn options(ids: &[u32]) -> DecisionInput {
        DecisionInput::OptionTokens {
            token_ids: ids.to_vec(),
        }
    }

    #[test]
    fn warmup_reads_row_zero_and_counts_one_like_native() {
        let plan =
            plan_option_requests(8, 3, &[0, 2], 2, &[DecisionInput::Warmup]).expect("warmup plan");
        assert_eq!(plan.counts(), &[1]);
        assert_eq!(plan.row_ids(), &[0]);
        assert_eq!(plan.row_offsets(), &[0, 1]);
        assert_eq!(plan.last_rows(), &[1]);
        assert_eq!(plan.total_options(), 1);
        assert_eq!(plan.max_options(), 1);
    }

    #[test]
    fn duplicate_and_ordered_option_rows_are_preserved_per_sequence() {
        let plan = plan_option_requests(
            16,
            3,
            &[0, 2, 5, 6],
            6,
            &[options(&[4, 1, 4]), options(&[0]), options(&[7, 7])],
        )
        .expect("ordered option plan");
        assert_eq!(plan.counts(), &[3, 1, 2]);
        assert_eq!(plan.row_offsets(), &[0, 3, 4, 6]);
        assert_eq!(plan.row_ids(), &[4, 1, 4, 0, 7, 7]);
        assert_eq!(plan.last_rows(), &[1, 4, 5]);
        assert_eq!(plan.boundaries(), &[0, 2, 5, 6]);
        assert_eq!(plan.total_options(), 6);
        assert_eq!(plan.max_options(), 3);
        let layout = OptionScratchLayout::for_plan(&plan).expect("layout");
        assert_eq!(
            layout,
            OptionScratchLayout {
                gather_rows: 9,
                total_options: 6,
            }
        );
        assert_eq!(
            layout.bytes().unwrap(),
            9 * 2 + 6 * 2 + 6 * 4 + 6 * 4,
            "leased HostIo ID staging is not charged to the shared GPU Work"
        );
    }

    #[test]
    fn prepares_one_m1_n_option_count_plan_per_sequence_with_exact_boundaries() {
        let head = Gemma4OptionHead {
            hidden: 4,
            vocab_rows: 32,
            cap: Some(OptionSoftcap {
                cap: E2B_OPTION_SOFCAP,
            }),
        };
        let prepared = head
            .prepare(
                &batch(vec![0, 3, 4], 4),
                &[options(&[1, 2, 1]), options(&[7])],
            )
            .expect("prepared option head");
        assert_eq!(prepared.tokens(), 4);
        assert_eq!(prepared.counts(), &[3, 1]);
        assert_eq!(prepared.total_options(), 4);
        assert_eq!(
            prepared.cap().map(OptionSoftcap::value),
            Some(E2B_OPTION_SOFCAP)
        );
        assert_eq!(prepared.layout().gather_rows, 3 * 4);
        assert_eq!(prepared.plans.len(), 2);
        assert_eq!(prepared.plans[0].input_layout().shape(), &[1, 4]);
        assert_eq!(prepared.plans[0].weight_layout().shape(), &[3, 4]);
        assert_eq!(prepared.plans[0].output_layout().shape(), &[1, 3]);
        assert_eq!(prepared.plans[1].weight_layout().shape(), &[1, 4]);
        assert_eq!(prepared.plans[1].output_layout().shape(), &[1, 1]);
    }

    #[test]
    fn empty_invalid_and_unsupported_requests_are_rejected_before_work() {
        assert_eq!(
            plan_option_requests(8, 2, &[0, 1], 1, &[options(&[])]).unwrap_err(),
            OptionPlanError::EmptyOptions { request: 0 }
        );
        assert_eq!(
            plan_option_requests(8, 2, &[0, 1], 1, &[options(&[8])]).unwrap_err(),
            OptionPlanError::TokenIdOutOfRange {
                request: 0,
                token_id: 8,
                vocab_rows: 8
            }
        );
        assert_eq!(
            plan_option_requests(8, 2, &[0, 1], 1, &[options(&[u32::MAX])]).unwrap_err(),
            OptionPlanError::TokenIdOutOfRange {
                request: 0,
                token_id: u32::MAX,
                vocab_rows: 8
            }
        );
        for request in [
            DecisionInput::InlineOptions {
                markers: vec![0],
                option_counts: vec![1],
            },
            DecisionInput::BlockOptionTokens {
                token_ids: vec![0],
                prefix_length: 1,
            },
            DecisionInput::Laya {
                question_type: 0,
                markers: vec![0],
            },
            DecisionInput::Clef { fields: Vec::new() },
        ] {
            assert!(matches!(
                plan_option_requests(8, 2, &[0, 1], 1, &[request]),
                Err(OptionPlanError::UnsupportedRequest(_))
            ));
        }
    }

    #[test]
    fn invalid_cumulative_boundaries_and_request_counts_are_rejected() {
        for boundaries in [vec![], vec![1, 2], vec![0, 2, 2], vec![0, 3, 2]] {
            let tokens = *boundaries.last().unwrap_or(&0) as usize;
            let requests = vec![options(&[0]); boundaries.len().saturating_sub(1)];
            assert!(matches!(
                plan_option_requests(8, 2, &boundaries, tokens, &requests),
                Err(OptionPlanError::InvalidCumulative(_))
            ));
        }
        assert_eq!(
            plan_option_requests(8, 2, &[0, 1, 2], 2, &[options(&[0])]).unwrap_err(),
            OptionPlanError::RequestCountMismatch {
                requests: 1,
                sequences: 2
            }
        );
    }

    #[test]
    fn work_retention_state_is_cleared_before_every_new_use() {
        let mut state = RetentionState::default();
        assert!(!state.is_drained());
        assert!(!state.needs_output_copy());
        state.mark_drained();
        assert!(state.is_drained());
        state.begin_use().unwrap();
        assert!(!state.is_drained());
        assert!(state.needs_output_copy());
        assert!(state.can_begin().is_err(), "B-before-A preflight must fail");
        assert!(state.begin_use().is_err(), "B before A D2H must fail");
        state.mark_output_copied().unwrap();
        assert!(!state.needs_output_copy());
        state.begin_use().unwrap();
        assert!(state.needs_output_copy());
        state.mark_output_copied().unwrap();
        state.mark_drained();
        assert!(state.is_drained());
    }

    #[test]
    fn typed_softcap_admits_only_finite_positive_e2b_thirty() {
        assert_eq!(
            OptionSoftcap::e2b(30.0).expect("E2B cap").value(),
            E2B_OPTION_SOFCAP
        );
        for cap in [
            0.0,
            -30.0,
            f64::NAN,
            f64::INFINITY,
            29.0,
            31.0,
            29.999_999_999_9,
            30.000_000_000_1,
        ] {
            assert!(OptionSoftcap::e2b(cap).is_err(), "cap {cap}");
        }
    }

    #[test]
    fn raw_and_pooled_layouts_share_rows_but_use_physical_hidden_rows() {
        let requests = [options(&[4, 1, 4]), options(&[7])];
        let raw = plan_option_requests_with_layout(
            16,
            3,
            &[0, 2, 5],
            5,
            &requests,
            OptionInputLayout::RawPacked,
        )
        .expect("raw packed plan");
        let pooled = plan_option_requests_with_layout(
            16,
            3,
            &[0, 2, 5],
            5,
            &requests,
            OptionInputLayout::Pooled,
        )
        .expect("pooled plan");
        assert_eq!(raw.row_ids(), pooled.row_ids());
        assert_eq!(raw.counts(), pooled.counts());
        assert_eq!(raw.last_rows(), &[1, 4]);
        assert_eq!(pooled.last_rows(), &[0, 1]);
        assert_eq!(raw.hidden_rows(), 5);
        assert_eq!(pooled.hidden_rows(), 2);
        assert_eq!(raw.layout(), OptionInputLayout::RawPacked);
        assert_eq!(pooled.layout(), OptionInputLayout::Pooled);
    }

    #[test]
    fn single_sequence_boundary_is_valid() {
        let plan = plan_option_requests(8, 2, &[0, 2], 2, &[options(&[1])]).expect("single row");
        assert_eq!(plan.counts(), &[1]);
        assert_eq!(plan.last_rows(), &[1]);
        assert_eq!(plan.boundaries(), &[0, 2]);
    }

    #[test]
    fn host_id_tail_offsets_are_checked_and_requests_do_not_alias() {
        let key = (7usize, 2usize, 5usize);
        let (offset, end) = option_id_staging_offset(key.0, key.1, key.2).unwrap();
        assert_eq!((offset, end), (23, 28));
        assert_eq!(crate::staging::input_capacity(key).unwrap(), end);
        let mut first = vec![0u32; end];
        let mut second = vec![0u32; end];
        let first_ids = [11, 12, 13, 14, 15];
        let second_ids = [21, 22, 23, 24, 25];
        assert_eq!(
            write_option_id_tail(&mut first, key, &first_ids).unwrap(),
            offset
        );
        assert_eq!(
            write_option_id_tail(&mut second, key, &second_ids).unwrap(),
            offset
        );
        assert_eq!(&first[offset..], &first_ids);
        assert_eq!(&second[offset..], &second_ids);
        assert!(first[..offset].iter().all(|value| *value == 0));
        assert!(second[..offset].iter().all(|value| *value == 0));
        assert!(write_option_id_tail(&mut first, key, &first_ids[..4]).is_err());
        assert!(write_option_id_tail(&mut first[..end - 1], key, &first_ids).is_err());
        assert!(option_id_staging_offset(usize::MAX, 0, 1).is_err());
    }

    #[test]
    fn packed_input_prefix_offsets_and_option_tail_do_not_overlap() {
        for (tokens, lengths, selections) in [
            (1usize, vec![1usize], 0usize),
            (7usize, vec![7usize], 3usize),
            (6usize, vec![1usize, 2usize, 3usize], 0usize),
        ] {
            let boundaries = lengths.len() + 1;
            let mut cumulative = vec![0u32];
            for length in &lengths {
                cumulative.push(cumulative.last().copied().unwrap() + *length as u32);
            }
            let ids: Vec<u32> = (0..tokens as u32).collect();
            let positions: Vec<u32> = lengths
                .iter()
                .flat_map(|&length| (0..length as u32).collect::<Vec<_>>())
                .collect();
            let types = vec![0u32; tokens];
            let host_key = (tokens, boundaries, 5usize);
            let capacity = crate::staging::input_capacity(host_key).unwrap();
            let prefix_words = tokens * 3 + boundaries;
            let (position_bytes, boundary_bytes, selection_bytes) =
                crate::input_section_offsets(tokens, boundaries, selections).unwrap();
            assert_eq!(position_bytes, tokens * 4);
            assert_eq!(boundary_bytes, tokens * 12);
            assert_eq!(selection_bytes, tokens * 12 + boundaries * 4);
            assert_eq!(
                selection_bytes + selections * 4,
                crate::staging::input_capacity((tokens, boundaries, selections)).unwrap() * 4
            );
            let mut host = vec![0u32; capacity];
            crate::pack_input(
                &mut host[..prefix_words],
                tokens,
                &ids,
                &positions,
                &types,
                &cumulative,
                &[],
            )
            .expect("pack actual token and boundary prefix");
            assert_eq!(&host[..tokens], ids.as_slice());
            assert_eq!(&host[tokens..tokens * 2], positions.as_slice());
            assert_eq!(&host[tokens * 2..tokens * 3], types.as_slice());
            assert_eq!(&host[tokens * 3..prefix_words], cumulative.as_slice());
            let option_ids = [9u32, 8, 7, 6, 5];
            let (tail, end) =
                option_id_staging_offset(tokens, boundaries, option_ids.len()).unwrap();
            assert_eq!((tail, end), (prefix_words, capacity));
            write_option_id_tail(&mut host, host_key, &option_ids)
                .expect("stage option tail after boundaries");
            assert_eq!(&host[tokens * 3..prefix_words], cumulative.as_slice());
            assert_eq!(&host[tail..end], option_ids.as_slice());
        }
    }

    #[test]
    fn layout_helpers_reject_checked_overflow() {
        assert!(OptionScratchLayout {
            gather_rows: usize::MAX,
            total_options: 1,
        }
        .bytes()
        .is_err());
        assert!(OptionScratchLayout {
            gather_rows: 1,
            total_options: usize::MAX,
        }
        .bytes()
        .is_err());
        assert_eq!(
            pack_option_outputs(&[], &[0, 0]).unwrap().len(),
            2,
            "zero-length outputs are admitted only when the plan itself required them"
        );
        assert!(pack_option_outputs(&[1.0], &[2]).is_err());
        assert!(pack_option_outputs(&[1.0, 2.0], &[1, 0, 2]).is_err());
    }

    #[test]
    fn flat_output_packing_keeps_request_order_duplicates_and_action_one() {
        let output = [10.0, 11.0, 12.0, 20.0, 30.0];
        let packed = pack_option_outputs(&output, &[2, 1, 2]).expect("packed outputs");
        assert_eq!(packed.len(), 3);
        assert_eq!(packed[0].logits, vec![10.0, 11.0]);
        assert_eq!(packed[1].logits, vec![12.0]);
        assert_eq!(packed[2].logits, vec![20.0, 30.0]);
        assert!(packed
            .iter()
            .all(|output| { output.action_probability == 1.0 && output.memory.is_some() }));
    }

    #[test]
    fn actual_rune_option_plan_admits_duplicate_and_warmup_rows() {
        let path =
            std::env::var("INFINITY_GEMMA4_E2B_ROOT").unwrap_or_else(|_| actual_path().to_owned());
        let weights =
            Gemma4Weights::load_rune(std::path::Path::new(&path)).expect("actual Rune E2B headers");
        let head = Gemma4OptionHead::from_weights(&weights).expect("tied BF16 option head");
        let tokens = 3usize;
        let batch = batch(vec![0, 2, 3], tokens);
        let prepared = head
            .prepare(&batch, &[options(&[1, 1, 2]), DecisionInput::Warmup])
            .expect("actual option plan");
        assert_eq!(prepared.counts(), &[3, 1]);
        assert_eq!(prepared.total_options(), 4);
        assert_eq!(prepared.layout().gather_rows, 3 * head.hidden_size());
    }

    #[test]
    #[ignore = "requires INFINITY_GEMMA4_OPTION_SOFTCAP_ORACLE, the oxide kernel module with the additive softcap export, and CUDA"]
    fn actual_option_softcap_operator_matches_bound_native_reference() {
        let oracle_path = std::env::var("INFINITY_GEMMA4_OPTION_SOFTCAP_ORACLE")
            .expect("set INFINITY_GEMMA4_OPTION_SOFTCAP_ORACLE to the tiny operator oracle JSON");
        let oracle: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&oracle_path).expect("read softcap oracle"))
                .expect("parse softcap oracle");
        assert_eq!(
            oracle["schema"], "gemma4-option-softcap-oracle/v1",
            "unexpected softcap oracle schema"
        );
        let cap = oracle["cap"].as_f64().expect("oracle cap");
        let softcap = OptionSoftcap::e2b(cap).expect("oracle cap is the admitted E2B 30");
        let context = cuda_core::Context::new(0).expect("CUDA context");
        let stream = context.new_stream().expect("CUDA stream");
        let path = std::env::var_os(infinity_v3_config::infinity_env!("OXIDE_KERNEL_MODULE"))
            .map(std::path::PathBuf::from)
            .expect("set INFINITY_OXIDE_KERNEL_MODULE to the oxide kernel artifact");
        let path = if path.is_dir() {
            let (major, minor) = context.compute_capability().expect("compute capability");
            path.join(format!("sm_{major}{minor}"))
                .join("kernels.cubin")
        } else {
            path
        };
        let data = std::fs::read(&path).expect("read oxide kernel artifact");
        let module = context
            .load_module_from_image(&data)
            .expect("load oxide kernel module");
        let kernel = module
            .load_function("infinity_gemma4_option_softcap_bf16_to_f32")
            .expect("additive softcap export");
        for case in oracle["cases"].as_array().expect("oracle cases") {
            let input_bits: Vec<u16> = case["input_bf16_bits"]
                .as_array()
                .expect("input_bf16_bits")
                .iter()
                .map(|value| u16::try_from(value.as_u64().expect("bf16 bits")).expect("u16 bits"))
                .collect();
            let expected: Vec<u32> = case["expected_f32_bits"]
                .as_array()
                .expect("expected_f32_bits")
                .iter()
                .map(|value| u32::try_from(value.as_u64().expect("f32 bits")).expect("u32 bits"))
                .collect();
            assert_eq!(input_bits.len(), expected.len());
            assert!(!input_bits.is_empty());
            let input: DeviceBuffer<Bf16> = DeviceBuffer::from_host(
                &stream,
                &input_bits
                    .iter()
                    .copied()
                    .map(Bf16::from_bits)
                    .collect::<Vec<_>>(),
            )
            .expect("upload softcap input");
            let output = unsafe {
                DeviceBuffer::<f32>::uninitialized_async(&stream, expected.len())
                    .expect("softcap output allocation")
            };
            let mut input_ptr = input.cu_deviceptr();
            let mut output_ptr = output.cu_deviceptr();
            let mut count = u32_size(expected.len()).expect("softcap count");
            let mut cap = softcap.value();
            launch(
                &kernel,
                &stream,
                blocks(expected.len()).expect("softcap grid"),
                0,
                &mut [
                    arg(&mut input_ptr),
                    arg(&mut output_ptr),
                    arg(&mut count),
                    arg(&mut cap),
                ],
            )
            .expect("softcap launch");
            let actual = output.to_host_vec(&stream).expect("read softcap output");
            let actual: Vec<u32> = actual.iter().map(|value| value.to_bits()).collect();
            assert_eq!(actual, expected, "softcap primitive oracle case");
        }
    }

    fn consumer_usize_list(value: &serde_json::Value, field: &str) -> Result<Vec<usize>, String> {
        value[field]
            .as_array()
            .ok_or_else(|| format!("missing {field}"))?
            .iter()
            .map(|entry| {
                entry
                    .as_u64()
                    .and_then(|value| usize::try_from(value).ok())
                    .ok_or_else(|| format!("invalid {field} entry"))
            })
            .collect()
    }

    fn consumer_u32_list(value: &serde_json::Value, field: &str) -> Result<Vec<u32>, String> {
        value[field]
            .as_array()
            .ok_or_else(|| format!("missing {field}"))?
            .iter()
            .map(|entry| {
                entry
                    .as_u64()
                    .and_then(|value| u32::try_from(value).ok())
                    .ok_or_else(|| format!("invalid {field} entry"))
            })
            .collect()
    }

    fn consumer_i32_list(value: &serde_json::Value, field: &str) -> Result<Vec<i32>, String> {
        value[field]
            .as_array()
            .ok_or_else(|| format!("missing {field}"))?
            .iter()
            .map(|entry| {
                entry
                    .as_i64()
                    .and_then(|value| i32::try_from(value).ok())
                    .ok_or_else(|| format!("invalid {field} entry"))
            })
            .collect()
    }

    fn consumer_read(
        base: &std::path::Path,
        descriptor: &serde_json::Value,
    ) -> Result<Vec<u8>, String> {
        let file = descriptor["file"]
            .as_str()
            .ok_or_else(|| "missing file name".to_owned())?;
        let path = base.join(file);
        let bytes = std::fs::read(&path).map_err(|error| format!("{}: {error}", path.display()))?;
        let expected = descriptor["bytes"]
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| format!("missing byte extent for {file}"))?;
        if bytes.len() != expected {
            return Err(format!(
                "{file}: read {} bytes, expected {expected}",
                bytes.len()
            ));
        }
        let digest = sha2::Sha256::digest(&bytes);
        let hash = format!("{digest:x}");
        if descriptor["sha256"].as_str() != Some(hash.as_str()) {
            return Err(format!("{file}: sha256 mismatch"));
        }
        if descriptor["byte_order"].as_str() != Some("little") {
            return Err(format!("{file}: expected little-endian payload"));
        }
        if descriptor["normalized_cpu_contiguous"].as_bool() != Some(true)
            || descriptor["payload_contiguous"].as_bool() != Some(true)
        {
            return Err(format!(
                "{file}: payload is not canonical contiguous CPU storage"
            ));
        }
        Ok(bytes)
    }

    fn consumer_bf16_words(bytes: &[u8], name: &str) -> Result<Vec<u16>, String> {
        if !bytes.len().is_multiple_of(2) {
            return Err(format!("{name}: BF16 payload has odd byte length"));
        }
        Ok(bytes
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect())
    }

    fn consumer_u32_words(bytes: &[u8], name: &str) -> Result<Vec<u32>, String> {
        if !bytes.len().is_multiple_of(4) {
            return Err(format!("{name}: uint32 payload has partial word"));
        }
        Ok(bytes
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect())
    }

    fn consumer_i32_words(bytes: &[u8], name: &str) -> Result<Vec<i32>, String> {
        if !bytes.len().is_multiple_of(4) {
            return Err(format!("{name}: int32 payload has partial word"));
        }
        Ok(bytes
            .chunks_exact(4)
            .map(|word| i32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect())
    }

    fn consumer_tensor_shape(
        descriptor: &serde_json::Value,
        name: &str,
        dtype: &str,
        shape: &[usize],
        stride: &[usize],
    ) -> Result<(), String> {
        if descriptor["dtype"].as_str() != Some(dtype) {
            return Err(format!("{name}: dtype does not match {dtype}"));
        }
        if consumer_usize_list(descriptor, "shape")? != shape {
            return Err(format!("{name}: physical shape does not match {shape:?}"));
        }
        if consumer_usize_list(descriptor, "stride")? != stride {
            return Err(format!("{name}: physical stride does not match {stride:?}"));
        }
        let numel = shape.iter().try_fold(1usize, |value, &extent| {
            value
                .checked_mul(extent)
                .ok_or_else(|| format!("{name}: extent overflow"))
        })?;
        let itemsize = if dtype == "torch.bfloat16" { 2 } else { 4 };
        let bytes = numel
            .checked_mul(itemsize)
            .ok_or_else(|| format!("{name}: byte extent overflow"))?;
        if descriptor["numel"].as_u64() != Some(numel as u64)
            || descriptor["bytes"].as_u64() != Some(bytes as u64)
        {
            return Err(format!(
                "{name}: logical extent does not match physical bytes"
            ));
        }
        Ok(())
    }

    fn run_cpu_consumer(
        input: &std::path::Path,
        output: &std::path::Path,
    ) -> Result<serde_json::Value, String> {
        let manifest_path = input.join("manifest.json");
        let manifest: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&manifest_path)
                .map_err(|error| format!("{}: {error}", manifest_path.display()))?,
        )
        .map_err(|error| format!("manifest JSON: {error}"))?;
        if manifest["schema"] != "oxide-gemma4-rune-head-native-b66-gold/v1" {
            return Err("unexpected native source input schema".into());
        }
        if manifest["case_count"].as_u64() != Some(9) {
            return Err("native source packet must contain exactly 9 cases".into());
        }
        if manifest["weights_are_synthetic_not_checkpoint"].as_bool() != Some(true)
            || manifest["actual_checkpoint_read"].as_bool() != Some(false)
        {
            return Err(
                "native source packet is not the synthetic non-checkpoint inventory".into(),
            );
        }
        let weight_descriptor = &manifest["weights"];
        consumer_tensor_shape(
            weight_descriptor,
            "tied embedding",
            "torch.bfloat16",
            &[257, 1536],
            &[1536, 1],
        )?;
        if weight_descriptor["tied_embedding_key"].as_str()
            != Some("model.language_model.embed_tokens.weight")
        {
            return Err("tied embedding key differs from Native lexical lookup".into());
        }
        if manifest["payloads"]["weights/model.language_model.embed_tokens.weight.bin"].as_str()
            != weight_descriptor["sha256"].as_str()
        {
            return Err("tied embedding payload manifest hash differs".into());
        }
        let weight_bytes = consumer_read(&input.join("weights"), weight_descriptor)?;
        let weight_words = consumer_bf16_words(&weight_bytes, "tied embedding")?;
        if weight_words.len() != 257 * 1536 {
            return Err("tied embedding word count differs".into());
        }

        let cases = manifest["cases"]
            .as_array()
            .ok_or_else(|| "missing cases".to_owned())?;
        let mut seen = std::collections::BTreeSet::new();
        let mut paired: std::collections::BTreeMap<String, PairPlan> =
            std::collections::BTreeMap::new();
        let mut reports = Vec::new();
        for case in cases {
            let id = case["id"].as_str().ok_or_else(|| "case id".to_owned())?;
            if !seen.insert(id.to_owned()) {
                return Err(format!("duplicate case id {id}"));
            }
            if case["weight_id"].as_str() != manifest["weight_id"].as_str()
                || case["native_tag"].as_str()
                    != Some("actual_native_create_decision_head_optionhead")
                || case["stage_tag"].as_str() != Some("SOURCE-EXACT ATen stage decomposition")
            {
                return Err(format!("{id}: case contract tag differs"));
            }
            let lengths = consumer_u32_list(case, "sequence_lengths")?;
            let offsets = consumer_i32_list(case, "offsets")?;
            let option_ids = case["option_ids"]
                .as_array()
                .ok_or_else(|| format!("{id}: option_ids"))?
                .iter()
                .map(|ids| {
                    ids.as_array()
                        .ok_or_else(|| format!("{id}: option id list"))?
                        .iter()
                        .map(|value| {
                            value
                                .as_u64()
                                .and_then(|value| u32::try_from(value).ok())
                                .ok_or_else(|| format!("{id}: option id range"))
                        })
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<Vec<_>, _>>()?;
            let flat_option_ids: Vec<u32> = option_ids.iter().flatten().copied().collect();
            let option_counts = consumer_usize_list(case, "option_counts")?;
            if offsets.len() != lengths.len() + 1
                || offsets.first() != Some(&0)
                || offsets.windows(2).any(|pair| pair[1] <= pair[0])
            {
                return Err(format!("{id}: offsets are not a packed cumulative table"));
            }
            let token_rows = lengths.iter().try_fold(0usize, |total, &length| {
                total
                    .checked_add(length as usize)
                    .ok_or_else(|| format!("{id}: length overflow"))
            })?;
            let offsets_usize: Vec<usize> = offsets
                .iter()
                .map(|&value| usize::try_from(value).map_err(|_| format!("{id}: negative offset")))
                .collect::<Result<_, _>>()?;
            if offsets_usize.last().copied() != Some(token_rows) {
                return Err(format!("{id}: final offset differs from token rows"));
            }
            if option_counts.iter().sum::<usize>() != flat_option_ids.len() {
                return Err(format!("{id}: option counts differ from ids"));
            }
            let pooled = case["pooled"]
                .as_bool()
                .ok_or_else(|| "pooled".to_owned())?;
            let physical_hidden_rows = if pooled { lengths.len() } else { token_rows };
            let hidden = &case["hidden"];
            consumer_tensor_shape(
                hidden,
                &format!("{id}/hidden"),
                "torch.bfloat16",
                &[physical_hidden_rows, 1536],
                &[1536, 1],
            )?;
            let hidden_bytes = consumer_read(&input.join(id), hidden)?;
            let hidden_words = consumer_bf16_words(&hidden_bytes, "hidden")?;
            if hidden_words.len() != physical_hidden_rows * 1536 {
                return Err(format!("{id}: hidden word count differs"));
            }
            if format!(
                "{:x}",
                sha2::Sha256::digest(hidden_words_bytes(&hidden_words))
            ) != case["hidden_physical_sha256"]
                .as_str()
                .ok_or_else(|| format!("{id}: hidden_physical_sha256"))?
            {
                return Err(format!("{id}: physical hidden word hash differs"));
            }
            let ids_file = &case["option_ids_file"];
            consumer_tensor_shape(
                ids_file,
                &format!("{id}/option_ids"),
                "uint32",
                &[flat_option_ids.len()],
                &[1],
            )?;
            let ids_bytes = consumer_read(&input.join(id), ids_file)?;
            if consumer_u32_words(&ids_bytes, "option_ids")? != flat_option_ids {
                return Err(format!(
                    "{id}: physical option id file differs from JSON metadata"
                ));
            }
            let lengths_file = &case["sequence_lengths_file"];
            consumer_tensor_shape(
                lengths_file,
                &format!("{id}/sequence_lengths"),
                "uint32",
                &[lengths.len()],
                &[1],
            )?;
            let lengths_bytes = consumer_read(&input.join(id), lengths_file)?;
            if consumer_u32_words(&lengths_bytes, "sequence_lengths")? != lengths {
                return Err(format!("{id}: physical sequence lengths differ"));
            }
            let offsets_file = &case["offsets_file"];
            consumer_tensor_shape(
                offsets_file,
                &format!("{id}/offsets"),
                "int32",
                &[offsets.len()],
                &[1],
            )?;
            let offsets_bytes = consumer_read(&input.join(id), offsets_file)?;
            if consumer_i32_words(&offsets_bytes, "offsets")? != offsets {
                return Err(format!("{id}: physical offsets differ"));
            }
            let requests: Vec<DecisionInput> = option_ids
                .iter()
                .map(|ids| DecisionInput::OptionTokens {
                    token_ids: ids.clone(),
                })
                .collect();
            let offsets_u32: Vec<u32> = offsets
                .iter()
                .map(|&value| u32::try_from(value).map_err(|_| format!("{id}: offset range")))
                .collect::<Result<_, _>>()?;
            let input_layout = if pooled {
                OptionInputLayout::Pooled
            } else {
                OptionInputLayout::RawPacked
            };
            let plan = plan_option_requests_with_layout(
                257,
                1536,
                &offsets_u32,
                token_rows,
                &requests,
                input_layout,
            )
            .map_err(|error| format!("{id}: {error}"))?;
            let expected_last_rows: Vec<u32> = match input_layout {
                OptionInputLayout::RawPacked => offsets_usize[1..]
                    .iter()
                    .map(|&end| {
                        end.checked_sub(1)
                            .and_then(|row| u32::try_from(row).ok())
                            .ok_or_else(|| format!("{id}: empty sequence boundary"))
                    })
                    .collect::<Result<_, _>>()?,
                OptionInputLayout::Pooled => (0..lengths.len())
                    .map(|row| u32::try_from(row).map_err(|_| format!("{id}: pooled row range")))
                    .collect::<Result<_, _>>()?,
            };
            if plan.counts() != option_counts.as_slice()
                || plan.row_ids() != flat_option_ids.as_slice()
                || plan.last_rows() != expected_last_rows.as_slice()
                || plan.hidden_rows() != physical_hidden_rows
                || plan.layout() != input_layout
            {
                return Err(format!(
                    "{id}: prepared plan differs from Native row metadata"
                ));
            }
            let layout =
                OptionScratchLayout::for_plan(&plan).map_err(|error| format!("{id}: {error}"))?;
            let signature = (plan.row_ids().to_vec(), plan.counts().to_vec());
            if let Some(pair_id) = case["pair_id"].as_str() {
                if let Some(previous) = paired.get(pair_id) {
                    if previous != &signature {
                        return Err(format!("{id}: fullT/pooled pair plan differs"));
                    }
                } else {
                    paired.insert(pair_id.to_owned(), signature.clone());
                }
            }
            let case_payloads = manifest["payloads"]
                .as_object()
                .ok_or_else(|| "payloads".to_owned())?;
            for (suffix, descriptor) in [
                ("hidden.bin", hidden),
                ("option_ids.bin", ids_file),
                ("sequence_lengths.bin", lengths_file),
                ("offsets.bin", offsets_file),
            ] {
                let name = format!("{id}/{suffix}");
                if case_payloads.get(&name).and_then(|value| value.as_str())
                    != descriptor["sha256"].as_str()
                {
                    return Err(format!("{name}: payload manifest hash differs"));
                }
            }
            // Repeat the same physical parse and plan; the identity result is
            // the CPU canary for stable ordered metadata.
            let repeat = plan_option_requests_with_layout(
                257,
                1536,
                &offsets_u32,
                token_rows,
                &requests,
                input_layout,
            )
            .map_err(|error| format!("{id}: repeat {error}"))?;
            if repeat.row_ids() != plan.row_ids()
                || repeat.last_rows() != plan.last_rows()
                || repeat.counts() != plan.counts()
            {
                return Err(format!("{id}: repeated plan differs"));
            }
            reports.push(serde_json::json!({
                "id": id,
                "pooled": pooled,
                "input_layout": format!("{input_layout:?}"),
                "pair_id": case["pair_id"],
                "token_rows": token_rows,
                "physical_hidden_rows": physical_hidden_rows,
                "sequence_lengths": lengths,
                "offsets": offsets,
                "option_counts": option_counts,
                "row_ids": plan.row_ids(),
                "last_rows": plan.last_rows(),
                "hidden_physical_sha256": case["hidden_physical_sha256"],
                "physical_sha256": case["physical_sha256"],
                "layout_bytes": layout.bytes().map_err(|error| error.to_string())?,
                "files": {
                    "hidden_sha256": hidden["sha256"],
                    "option_ids_sha256": ids_file["sha256"],
                    "sequence_lengths_sha256": lengths_file["sha256"],
                    "offsets_sha256": offsets_file["sha256"],
                },
            }));
        }
        if reports.len() != 9 || seen.len() != 9 {
            return Err("native source packet did not expose all 9 cases".into());
        }
        let exhaustive: Vec<u16> = (0..=u16::MAX).collect();
        let exhaustive_bytes = exhaustive
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<u8>>();
        let exhaustive_sha = format!("{:x}", sha2::Sha256::digest(&exhaustive_bytes));
        std::fs::create_dir_all(output)
            .map_err(|error| format!("create {}: {error}", output.display()))?;
        std::fs::write(output.join("softcap-input-65536.bin"), &exhaustive_bytes)
            .map_err(|error| format!("write softcap input: {error}"))?;
        // Re-read every physical file after the full parse to prove the reader
        // neither rewrote nor mutated the source packet.
        for case in cases {
            let id = case["id"].as_str().ok_or_else(|| "case id".to_owned())?;
            for (directory, descriptor) in [
                (input.join(id), &case["hidden"]),
                (input.join(id), &case["option_ids_file"]),
                (input.join(id), &case["sequence_lengths_file"]),
                (input.join(id), &case["offsets_file"]),
            ] {
                consumer_read(&directory, descriptor)?;
            }
        }
        consumer_read(&input.join("weights"), weight_descriptor)?;
        let report = serde_json::json!({
            "schema": "oxide-gemma4-rune-head-cpu-consumer/v1",
            "status": "passed",
            "input_dir": input.display().to_string(),
            "output_dir": output.display().to_string(),
            "source_schema": manifest["schema"],
            "case_count": reports.len(),
            "weights": {
                "sha256": weight_descriptor["sha256"],
                "shape": [257, 1536],
                "dtype": "bfloat16",
                "synthetic": true,
            },
            "cases": reports,
            "exhaustive_softcap": {
                "count": exhaustive.len(),
                "sha256": exhaustive_sha,
                "file": "softcap-input-65536.bin",
            },
            "canaries": {
                "input_immutable_after_full_parse": true,
                "repeat_plan_identical": true,
                "unique_case_ids": true,
                "payload_manifest_matches": true,
            },
            "not_claims": [
                "no GPU launch",
                "no Native logits/gold read",
                "no bit-exactness claim",
                "source inputs are synthetic and not a checkpoint",
            ],
        });
        std::fs::write(
            output.join("candidate-input-report.json"),
            serde_json::to_vec_pretty(&report).map_err(|error| error.to_string())?,
        )
        .map_err(|error| format!("write report: {error}"))?;
        Ok(report)
    }

    fn hidden_words_bytes(words: &[u16]) -> Vec<u8> {
        words.iter().flat_map(|word| word.to_le_bytes()).collect()
    }

    #[test]
    fn native_source_cpu_consumer_reads_all_nine_physical_inputs() {
        let (Ok(input), Ok(output)) = (
            std::env::var("INFINITY_GEMMA4_RUNE_CONSUMER_INPUT"),
            std::env::var("INFINITY_GEMMA4_RUNE_CONSUMER_OUTPUT"),
        ) else {
            return;
        };
        let report = run_cpu_consumer(std::path::Path::new(&input), std::path::Path::new(&output))
            .expect("CPU native source consumer");
        assert_eq!(report["schema"], "oxide-gemma4-rune-head-cpu-consumer/v1");
        assert_eq!(report["status"], "passed");
        assert_eq!(report["case_count"], 9);
        assert_eq!(report["exhaustive_softcap"]["count"], 65536);
    }

    fn gpu_read_descriptor(
        base: &std::path::Path,
        descriptor: &serde_json::Value,
    ) -> Result<Vec<u8>, String> {
        consumer_read(base, descriptor)
    }

    fn gpu_manifest_case<'a>(
        manifest: &'a serde_json::Value,
        id: &str,
    ) -> Result<&'a serde_json::Value, String> {
        manifest["cases"]
            .as_array()
            .ok_or_else(|| "reference cases".to_owned())?
            .iter()
            .find(|case| case["id"].as_str() == Some(id))
            .ok_or_else(|| format!("reference case {id} missing"))
    }

    fn gpu_f32_bits(bytes: &[u8], name: &str) -> Result<Vec<u32>, String> {
        consumer_u32_words(bytes, name)
    }

    struct NativeReferenceCase {
        id: String,
        counts: Vec<usize>,
        logits: Vec<Vec<u32>>,
        actions: Vec<u32>,
        stage_final: Vec<Vec<u32>>,
        repeat: bool,
        input_immutable: bool,
    }

    fn load_native_reference(
        reference_manifest: &std::path::Path,
        source_input: &std::path::Path,
    ) -> Result<Vec<NativeReferenceCase>, String> {
        let reference: serde_json::Value = serde_json::from_slice(
            &std::fs::read(reference_manifest)
                .map_err(|error| format!("{}: {error}", reference_manifest.display()))?,
        )
        .map_err(|error| format!("reference JSON: {error}"))?;
        if reference["schema"] != "oxide-gemma4-rune-head-native-b66-gold/v1"
            || reference["status"] != "passed-native-gpu"
            || reference["GPU_state"] != "passed"
        {
            return Err("reference is not a passed Native-source GPU run".into());
        }
        if reference["case_count"].as_u64() != Some(9) {
            return Err("reference must contain all 9 Native source cases".into());
        }
        let source: serde_json::Value = serde_json::from_slice(
            &std::fs::read(source_input.join("manifest.json"))
                .map_err(|error| format!("source manifest: {error}"))?,
        )
        .map_err(|error| format!("source JSON: {error}"))?;
        if source["schema"] != "oxide-gemma4-rune-head-native-b66-gold/v1"
            || source["weights_are_synthetic_not_checkpoint"].as_bool() != Some(true)
            || source["actual_checkpoint_read"].as_bool() != Some(false)
            || source["case_count"].as_u64() != Some(9)
        {
            return Err("source input is not the synthetic Native source packet".into());
        }
        let source_cases = source["cases"]
            .as_array()
            .ok_or_else(|| "source cases".to_owned())?;
        let reference_cases = reference["cases"]
            .as_array()
            .ok_or_else(|| "reference cases".to_owned())?;
        if source_cases.len() != 9 || reference_cases.len() != 9 {
            return Err("source/reference case count is not 9".into());
        }
        let base = reference_manifest
            .parent()
            .ok_or_else(|| "reference manifest has no parent".to_owned())?;
        let mut result = Vec::new();
        for (source_case, reference_case) in source_cases.iter().zip(reference_cases) {
            let id = source_case["id"]
                .as_str()
                .ok_or_else(|| "source case id".to_owned())?;
            if reference_case["id"].as_str() != Some(id) {
                return Err(format!("{id}: reference case order/id differs"));
            }
            if reference_case["native_tag"].as_str()
                != Some("actual_native_create_decision_head_optionhead")
                || reference_case["stage_tag"].as_str()
                    != Some("SOURCE-EXACT ATen stage decomposition")
            {
                return Err(format!("{id}: reference tag differs"));
            }
            if source_case["native_tag"].as_str()
                != Some("actual_native_create_decision_head_optionhead")
                || source_case["stage_tag"].as_str()
                    != Some("SOURCE-EXACT ATen stage decomposition")
            {
                return Err(format!("{id}: source tag differs"));
            }
            let source_counts = consumer_usize_list(source_case, "option_counts")?;
            let reference_counts = consumer_usize_list(reference_case, "option_counts")?;
            if source_counts != reference_counts {
                return Err(format!("{id}: source/reference counts differ"));
            }
            if reference_case["native_matches_stage_final_fullbits"].as_bool() != Some(true)
                || reference_case["repeat_bitidentity"].as_bool() != Some(true)
                || reference_case["input_immutability"]["hidden"].as_bool() != Some(true)
                || reference_case["input_immutability"]["weight"].as_bool() != Some(true)
            {
                return Err(format!(
                    "{id}: reference repeat/immutability/stage guard differs"
                ));
            }
            let case_dir = base.join(id);
            let mut logits = Vec::new();
            let mut actions = Vec::new();
            let mut stage_final = Vec::new();
            for (sequence, &count) in reference_counts.iter().enumerate() {
                let logit = &reference_case["outputs"][format!("logits_{sequence}")];
                consumer_tensor_shape(
                    logit,
                    &format!("{id}/logits_{sequence}"),
                    "torch.float32",
                    &[count],
                    &[1],
                )?;
                let logit_bytes = consumer_read(&case_dir, logit)?;
                let logit_bits = gpu_f32_bits(&logit_bytes, "reference logits")?;
                if logit_bits.iter().any(|value| {
                    let value = f32::from_bits(*value);
                    !value.is_finite()
                }) {
                    return Err(format!("{id}/{sequence}: nonfinite reference logits"));
                }
                let action = &reference_case["outputs"][format!("action_{sequence}")];
                consumer_tensor_shape(
                    action,
                    &format!("{id}/action_{sequence}"),
                    "torch.float32",
                    &[],
                    &[],
                )?;
                let action_bytes = consumer_read(&case_dir, action)?;
                let action_bits = gpu_f32_bits(&action_bytes, "reference action")?;
                if action_bits != vec![1.0f32.to_bits()] {
                    return Err(format!("{id}/{sequence}: reference action is not one"));
                }
                let stage = &reference_case["stages"][sequence.to_string()];
                let stage_shapes: [(&str, &str, Vec<usize>, Vec<usize>); 7] = [
                    ("last_bf16", "torch.bfloat16", vec![1, 1536], vec![1536, 1]),
                    (
                        "selected_weights_bf16",
                        "torch.bfloat16",
                        vec![count, 1536],
                        vec![1536, 1],
                    ),
                    (
                        "linear_bf16",
                        "torch.bfloat16",
                        vec![1, count],
                        vec![count, 1],
                    ),
                    (
                        "softcap_div_bf16",
                        "torch.bfloat16",
                        vec![1, count],
                        vec![count, 1],
                    ),
                    (
                        "tanh_bf16",
                        "torch.bfloat16",
                        vec![1, count],
                        vec![count, 1],
                    ),
                    (
                        "softcap_mul_bf16",
                        "torch.bfloat16",
                        vec![1, count],
                        vec![count, 1],
                    ),
                    ("final_f32", "torch.float32", vec![count], vec![1]),
                ];
                let mut final_bits = Vec::new();
                for (name, dtype, shape, stride) in stage_shapes {
                    let descriptor = stage
                        .get(name)
                        .ok_or_else(|| format!("{id}/{sequence}: missing stage {name}"))?;
                    consumer_tensor_shape(
                        descriptor,
                        &format!("{id}/stage_{sequence}_{name}"),
                        dtype,
                        &shape,
                        &stride,
                    )?;
                    let bytes = consumer_read(&case_dir, descriptor)?;
                    if name == "final_f32" {
                        final_bits = gpu_f32_bits(&bytes, "reference stage final")?;
                    }
                }
                if final_bits != logit_bits {
                    return Err(format!(
                        "{id}/{sequence}: Native output differs from stage final"
                    ));
                }
                logits.push(logit_bits);
                actions.push(action_bits[0]);
                stage_final.push(final_bits);
            }
            result.push(NativeReferenceCase {
                id: id.to_owned(),
                counts: reference_counts,
                logits,
                actions,
                stage_final,
                repeat: true,
                input_immutable: true,
            });
        }
        Ok(result)
    }

    #[test]
    fn actual_native_reference_cpu_parser_reads_all_nine_cases_without_cuda() {
        let (Ok(reference), Ok(source)) = (
            std::env::var("INFINITY_GEMMA4_RUNE_HEAD_NATIVE_REFERENCE"),
            std::env::var("INFINITY_GEMMA4_RUNE_HEAD_SOURCE_INPUT"),
        ) else {
            return;
        };
        let cases = load_native_reference(
            std::path::Path::new(&reference),
            std::path::Path::new(&source),
        )
        .expect("CPU-only Native reference parse");
        assert_eq!(cases.len(), 9);
        assert!(cases.iter().all(|case| case.repeat && case.input_immutable));
        for case in &cases {
            assert_eq!(case.logits.len(), case.counts.len());
            assert_eq!(case.stage_final.len(), case.counts.len());
            assert_eq!(case.actions, vec![1.0f32.to_bits(); case.counts.len()]);
            for (bits, &count) in case.logits.iter().zip(&case.counts) {
                assert_eq!(bits.len(), count);
            }
        }
    }

    #[test]
    #[ignore = "requires the 9 synthetic Native source inputs, a Native-source GPU reference manifest, the oxide kernel module with the additive softcap export, and CUDA"]
    fn actual_fullhead_gpu_consumer_matches_native_source_outputs_and_stages() {
        let input = std::path::PathBuf::from(
            std::env::var("INFINITY_GEMMA4_RUNE_HEAD_SOURCE_INPUT")
                .expect("set INFINITY_GEMMA4_RUNE_HEAD_SOURCE_INPUT"),
        );
        let reference = std::path::PathBuf::from(
            std::env::var("INFINITY_GEMMA4_RUNE_HEAD_NATIVE_REFERENCE").expect(
                "set INFINITY_GEMMA4_RUNE_HEAD_NATIVE_REFERENCE to the source-run manifest",
            ),
        );
        let reference_dir = reference
            .parent()
            .ok_or_else(|| "reference manifest has no parent".to_owned())
            .expect("reference parent");
        let input_manifest: serde_json::Value = serde_json::from_slice(
            &std::fs::read(input.join("manifest.json")).expect("read source inputs"),
        )
        .expect("parse source inputs");
        let reference_manifest: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&reference).expect("read Native source reference"),
        )
        .expect("parse Native source reference");
        let reference_cases =
            load_native_reference(&reference, &input).expect("CPU reference parser before GPU");
        assert_eq!(reference_cases.len(), 9);
        assert_eq!(
            reference_manifest["schema"],
            "oxide-gemma4-rune-head-native-b66-gold/v1"
        );
        assert_eq!(reference_manifest["status"], "passed-native-gpu");
        assert_eq!(reference_manifest["GPU_state"], "passed");
        let module_path =
            std::env::var_os(infinity_v3_config::infinity_env!("OXIDE_KERNEL_MODULE"))
                .map(std::path::PathBuf::from)
                .expect("set INFINITY_OXIDE_KERNEL_MODULE");
        let context = cuda_core::Context::new(0).expect("CUDA context");
        let stream = context.new_stream().expect("CUDA stream");
        let second_stream = context.new_stream().expect("second CUDA stream");
        let blas = Blas::new(context.clone(), stream.clone()).expect("plain cuBLAS");
        let path = if module_path.is_dir() {
            let (major, minor) = context.compute_capability().expect("compute capability");
            module_path
                .join(format!("sm_{major}{minor}"))
                .join("kernels.cubin")
        } else {
            module_path
        };
        let data = std::fs::read(&path).expect("read oxide kernel module");
        let module = context
            .load_module_from_image(&data)
            .expect("load oxide kernel module");
        let kernels = OptionKernels {
            gather_bf16: &module
                .load_function("infinity_embedding_gather")
                .expect("bit-copy gather export"),
            widen_bf16: &module
                .load_function("infinity_bf16_to_f32")
                .expect("BF16 widen export"),
            softcap_bf16_to_f32: Some(
                &module
                    .load_function("infinity_gemma4_option_softcap_bf16_to_f32")
                    .expect("additive softcap export"),
            ),
        };
        let weight_descriptor = &input_manifest["weights"];
        let weight_bytes = consumer_read(&input.join("weights"), weight_descriptor)
            .expect("read synthetic tied embedding");
        let weight_words = consumer_bf16_words(&weight_bytes, "weight").expect("weight words");
        let weight_gpu: DeviceBuffer<Bf16> = DeviceBuffer::from_host(
            &stream,
            &weight_words
                .iter()
                .copied()
                .map(Bf16::from_bits)
                .collect::<Vec<_>>(),
        )
        .expect("upload synthetic tied embedding");
        let weight_before = weight_gpu.to_host_vec(&stream).expect("read weight canary");
        let stage_names = [
            "last_bf16",
            "selected_weights_bf16",
            "linear_bf16",
            "softcap_div_bf16",
            "tanh_bf16",
            "softcap_mul_bf16",
            "final_f32",
        ];
        let mut checked_cases = 0usize;
        for case in input_manifest["cases"]
            .as_array()
            .expect("source input cases")
        {
            let id = case["id"].as_str().expect("case id");
            checked_cases += 1;
            let hidden = &case["hidden"];
            let hidden_rows = consumer_usize_list(hidden, "shape").expect("hidden shape")[0];
            let hidden_bytes = consumer_read(&input.join(id), hidden).expect("read hidden");
            let hidden_words = consumer_bf16_words(&hidden_bytes, "hidden").expect("hidden words");
            let hidden_gpu: DeviceBuffer<Bf16> = DeviceBuffer::from_host(
                &stream,
                &hidden_words
                    .iter()
                    .copied()
                    .map(Bf16::from_bits)
                    .collect::<Vec<_>>(),
            )
            .expect("upload synthetic hidden");
            let hidden_before = hidden_gpu.to_host_vec(&stream).expect("read hidden canary");
            let ids_bytes =
                consumer_read(&input.join(id), &case["option_ids_file"]).expect("read option ids");
            let flat_ids = consumer_u32_words(&ids_bytes, "option ids").expect("option ids");
            let lengths = consumer_u32_list(case, "sequence_lengths").expect("lengths");
            let offsets = consumer_i32_list(case, "offsets").expect("offsets");
            let offsets_u32: Vec<u32> = offsets
                .iter()
                .map(|&value| u32::try_from(value).expect("offsets fit u32"))
                .collect();
            let option_counts = consumer_usize_list(case, "option_counts").expect("counts");
            let mut cursor = 0usize;
            let requests: Vec<DecisionInput> = option_counts
                .iter()
                .map(|&count| {
                    let ids = flat_ids[cursor..cursor + count].to_vec();
                    cursor += count;
                    DecisionInput::OptionTokens { token_ids: ids }
                })
                .collect();
            let pooled = case["pooled"].as_bool().expect("pooled");
            let layout = if pooled {
                OptionInputLayout::Pooled
            } else {
                OptionInputLayout::RawPacked
            };
            let head = Gemma4OptionHead {
                hidden: 1536,
                vocab_rows: 257,
                cap: Some(OptionSoftcap::e2b(30.0).expect("cap30")),
            };
            let token_rows = lengths.iter().map(|&length| length as usize).sum::<usize>();
            let batch = batch(offsets_u32, token_rows);
            let prepared = head
                .prepare_with_layout(&batch, &requests, layout)
                .expect("production layout preparation");
            assert_eq!(prepared.hidden_rows(), hidden_rows);
            assert_eq!(prepared.counts(), option_counts.as_slice());
            let mut work = Gemma4OptionWork::new(stream.clone(), prepared.layout())
                .expect("production option Work");
            let ids_host =
                PinnedHostBuffer::from_slice(&context, &flat_ids).expect("pinned option ids");
            unsafe {
                prepared
                    .enqueue(
                        &blas,
                        &stream,
                        &hidden_gpu,
                        &weight_gpu,
                        &mut work,
                        &ids_host,
                        0,
                        &kernels,
                    )
                    .expect("production fullhead enqueue");
            }
            stream.synchronize().expect("fullhead completion");
            let actual = work
                .output()
                .to_host_vec(&stream)
                .expect("read fullhead logits");
            let actual_bits: Vec<u32> = actual.iter().map(|value| value.to_bits()).collect();
            // The first use has been proven complete by synchronize + host read.
            unsafe { work.mark_stream_drained() };
            let reference_case =
                gpu_manifest_case(&reference_manifest, id).expect("reference case");
            assert_eq!(
                reference_case["native_matches_stage_final_fullbits"], true,
                "{id}: Native/reference stage final differs"
            );
            assert_eq!(
                reference_case["repeat_bitidentity"], true,
                "{id}: Native repeat bit identity differs"
            );
            assert_eq!(
                reference_case["input_immutability"]["hidden"], true,
                "{id}: Native hidden mutation"
            );
            assert_eq!(
                reference_case["input_immutability"]["weight"], true,
                "{id}: Native weight mutation"
            );
            let reference_counts =
                consumer_usize_list(reference_case, "option_counts").expect("reference counts");
            assert_eq!(reference_counts, option_counts);
            let mut offset = 0usize;
            for (sequence, &count) in option_counts.iter().enumerate() {
                let logits_descriptor = &reference_case["outputs"][format!("logits_{sequence}")];
                let logits_bytes = gpu_read_descriptor(&reference_dir.join(id), logits_descriptor)
                    .expect("read Native logits");
                let expected = gpu_f32_bits(&logits_bytes, "Native logits").expect("logit bits");
                assert_eq!(expected.len(), count);
                assert_eq!(
                    &actual_bits[offset..offset + count],
                    expected.as_slice(),
                    "{id}/{sequence}: full F32 logit words differ"
                );
                let action_descriptor = &reference_case["outputs"][format!("action_{sequence}")];
                let action_bytes = gpu_read_descriptor(&reference_dir.join(id), action_descriptor)
                    .expect("read Native action");
                let action = gpu_f32_bits(&action_bytes, "Native action").expect("action bits");
                assert_eq!(action, vec![1.0f32.to_bits()]);
                let stage = &reference_case["stages"][sequence.to_string()];
                for name in stage_names {
                    assert!(
                        stage.get(name).is_some(),
                        "{id}/{sequence}: missing stage {name}"
                    );
                }
                let final_bytes = gpu_read_descriptor(&reference_dir.join(id), &stage["final_f32"])
                    .expect("read stage final");
                let final_bits =
                    gpu_f32_bits(&final_bytes, "stage final").expect("stage final bits");
                assert_eq!(
                    &actual_bits[offset..offset + count],
                    final_bits.as_slice(),
                    "{id}/{sequence}: stage final differs"
                );
                offset += count;
            }
            assert_eq!(offset, actual_bits.len());
            // Re-run the exact same production path and require bit identity.
            unsafe {
                prepared
                    .enqueue(
                        &blas,
                        &stream,
                        &hidden_gpu,
                        &weight_gpu,
                        &mut work,
                        &ids_host,
                        0,
                        &kernels,
                    )
                    .expect("repeat fullhead enqueue");
            }
            stream.synchronize().expect("repeat completion");
            let repeat = work
                .output()
                .to_host_vec(&stream)
                .expect("read repeat logits");
            assert_eq!(
                repeat
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                actual_bits
            );
            // Production path must not mutate either physical input.
            assert_eq!(
                hidden_gpu
                    .to_host_vec(&stream)
                    .expect("read hidden after")
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                hidden_before
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                weight_gpu
                    .to_host_vec(&stream)
                    .expect("read weight after")
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                weight_before
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
            // The same production preflight rejects a different same-context
            // stream before any H2D/launch.
            assert!(prepared
                .preflight(
                    &blas,
                    &second_stream,
                    &hidden_gpu,
                    &weight_gpu,
                    &work,
                    &ids_host,
                    0,
                    &kernels,
                )
                .is_err());
            unsafe { work.mark_stream_drained() };
        }
        assert_eq!(checked_cases, 9);
        assert_eq!(input_manifest["case_count"], 9);
    }

    #[test]
    #[ignore = "requires the synthetic Native source input, the oxide kernel module with the additive softcap export, two HostIo leases, and CUDA"]
    fn actual_two_hostio_leases_reuse_one_gpu_work_and_reject_stream_context() {
        let input = std::path::PathBuf::from(
            std::env::var("INFINITY_GEMMA4_RUNE_HEAD_SOURCE_INPUT")
                .expect("set INFINITY_GEMMA4_RUNE_HEAD_SOURCE_INPUT"),
        );
        let manifest: serde_json::Value = serde_json::from_slice(
            &std::fs::read(input.join("manifest.json")).expect("read source inputs"),
        )
        .expect("parse source inputs");
        let case = &manifest["cases"][0];
        let id = case["id"].as_str().expect("case id");
        let hidden = &case["hidden"];
        let hidden_bytes = consumer_read(&input.join(id), hidden).expect("read hidden");
        let hidden_words = consumer_bf16_words(&hidden_bytes, "hidden").expect("hidden words");
        let weight_descriptor = &manifest["weights"];
        let weight_bytes = consumer_read(&input.join("weights"), weight_descriptor)
            .expect("read synthetic embedding");
        let weight_words = consumer_bf16_words(&weight_bytes, "weight").expect("weight words");
        let ids_bytes =
            consumer_read(&input.join(id), &case["option_ids_file"]).expect("read option ids");
        let flat_ids = consumer_u32_words(&ids_bytes, "option ids").expect("option ids");
        let offsets = consumer_i32_list(case, "offsets").expect("offsets");
        let offsets_u32: Vec<u32> = offsets
            .iter()
            .map(|&value| u32::try_from(value).expect("offsets fit u32"))
            .collect();
        let lengths = consumer_u32_list(case, "sequence_lengths").expect("lengths");
        let token_rows = lengths.iter().map(|&length| length as usize).sum::<usize>();
        let pooled = case["pooled"].as_bool().expect("pooled");
        assert!(!pooled, "A/B gate uses the RawPacked production layout");
        let layout = OptionInputLayout::RawPacked;
        let first = flat_ids.clone();
        let mut second = flat_ids.clone();
        second.reverse();
        let first_request = [DecisionInput::OptionTokens {
            token_ids: first.clone(),
        }];
        let second_request = [DecisionInput::OptionTokens {
            token_ids: second.clone(),
        }];
        let batch = batch(offsets_u32, token_rows);
        let head = Gemma4OptionHead {
            hidden: 1536,
            vocab_rows: 257,
            cap: Some(OptionSoftcap::e2b(30.0).expect("cap30")),
        };
        let prepared_first = head
            .prepare_with_layout(&batch, &first_request, layout)
            .expect("first prepared head");
        let prepared_second = head
            .prepare_with_layout(&batch, &second_request, layout)
            .expect("second prepared head");
        let capacity = OptionScratchLayout {
            gather_rows: prepared_first
                .layout()
                .gather_rows
                .max(prepared_second.layout().gather_rows),
            total_options: prepared_first
                .layout()
                .total_options
                .max(prepared_second.layout().total_options),
        };
        let context = cuda_core::Context::new(0).expect("CUDA context");
        let stream = context.new_stream().expect("CUDA stream");
        let second_stream = context.new_stream().expect("second CUDA stream");
        let other_context = cuda_core::Context::new(0).expect("second CUDA context");
        let other_stream = other_context.new_stream().expect("other context stream");
        let blas = Blas::new(context.clone(), stream.clone()).expect("plain cuBLAS");
        let module_path =
            std::env::var_os(infinity_v3_config::infinity_env!("OXIDE_KERNEL_MODULE"))
                .map(std::path::PathBuf::from)
                .expect("set INFINITY_OXIDE_KERNEL_MODULE");
        let path = if module_path.is_dir() {
            let (major, minor) = context.compute_capability().expect("compute capability");
            module_path
                .join(format!("sm_{major}{minor}"))
                .join("kernels.cubin")
        } else {
            module_path
        };
        let data = std::fs::read(&path).expect("read oxide kernel module");
        let module = context
            .load_module_from_image(&data)
            .expect("load oxide kernel module");
        let kernels = OptionKernels {
            gather_bf16: &module
                .load_function("infinity_embedding_gather")
                .expect("bit-copy gather"),
            widen_bf16: &module.load_function("infinity_bf16_to_f32").expect("widen"),
            softcap_bf16_to_f32: Some(
                &module
                    .load_function("infinity_gemma4_option_softcap_bf16_to_f32")
                    .expect("softcap"),
            ),
        };
        let hidden_gpu: DeviceBuffer<Bf16> = DeviceBuffer::from_host(
            &stream,
            &hidden_words
                .iter()
                .copied()
                .map(Bf16::from_bits)
                .collect::<Vec<_>>(),
        )
        .expect("upload hidden");
        let weight_gpu: DeviceBuffer<Bf16> = DeviceBuffer::from_host(
            &stream,
            &weight_words
                .iter()
                .copied()
                .map(Bf16::from_bits)
                .collect::<Vec<_>>(),
        )
        .expect("upload weight");
        let run_expected = |prepared: &PreparedGemma4OptionHead, ids: &[u32]| -> Vec<u32> {
            let mut work = Gemma4OptionWork::new(stream.clone(), capacity).expect("expected Work");
            let ids_host =
                PinnedHostBuffer::from_slice(&context, ids).expect("expected pinned ids");
            unsafe {
                prepared
                    .enqueue(
                        &blas,
                        &stream,
                        &hidden_gpu,
                        &weight_gpu,
                        &mut work,
                        &ids_host,
                        0,
                        &kernels,
                    )
                    .expect("expected enqueue");
            }
            stream.synchronize().expect("expected sync");
            let bits = work
                .output()
                .to_host_vec(&stream)
                .expect("expected read")
                .into_iter()
                .map(|value| value.to_bits())
                .collect();
            unsafe { work.mark_stream_drained() };
            bits
        };
        let expected_first = run_expected(&prepared_first, &first);
        let expected_second = run_expected(&prepared_second, &second);
        let pool = crate::staging::HostIo::pool(&context).expect("HostIo pool");
        let first_slot = pool.acquire().expect("first lease slot");
        let second_slot = pool.acquire().expect("second lease slot");
        let mut first_lease = HostIoLease::detached(first_slot, pool.clone(), stream.clone());
        let mut second_lease = HostIoLease::detached(second_slot, pool.clone(), stream.clone());
        let boundaries = batch.cumulative_seq_lengths.len();
        let mut packed = vec![
            0u32;
            crate::staging::input_capacity((token_rows, boundaries, first.len()))
                .expect("A/B host layout")
        ];
        crate::pack_input(
            &mut packed[..token_rows * 3 + boundaries],
            token_rows,
            &batch.input_ids,
            &batch.position_ids,
            &batch.token_type_ids,
            &batch.cumulative_seq_lengths,
            &[],
        )
        .expect("A/B canonical prefix");
        let boundary_snapshot = packed[token_rows * 3..token_rows * 3 + boundaries].to_vec();
        let first_key = (token_rows, boundaries, first.len());
        let second_key = (token_rows, boundaries, second.len());
        first_lease
            .prepare(first_key, first.len(), false)
            .expect("prepare first lease");
        second_lease
            .prepare(second_key, second.len(), false)
            .expect("prepare second lease");
        let first_offset =
            stage_option_ids(&mut first_lease, first_key, &first).expect("stage first ids");
        let second_offset =
            stage_option_ids(&mut second_lease, second_key, &second).expect("stage second ids");
        write_option_id_tail(&mut packed, first_key, &first).expect("A/B tail witness");
        assert_eq!(
            &packed[token_rows * 3..token_rows * 3 + boundaries],
            boundary_snapshot.as_slice(),
            "option tail must not overwrite packed boundaries"
        );
        let mut work = Gemma4OptionWork::new(stream.clone(), capacity).expect("shared Work");
        unsafe {
            prepared_first
                .enqueue_from_lease(
                    &blas,
                    &stream,
                    &hidden_gpu,
                    &weight_gpu,
                    &mut work,
                    &mut first_lease,
                    first_offset,
                    &kernels,
                )
                .expect("A bound lease enqueue");
        }
        // B must be rejected by real state preflight before A's output D2H.
        let early = unsafe {
            prepared_second.enqueue_from_lease(
                &blas,
                &stream,
                &hidden_gpu,
                &weight_gpu,
                &mut work,
                &mut second_lease,
                second_offset,
                &kernels,
            )
        };
        assert!(
            matches!(&early, Err(BackendError::Inference(message)) if message.contains("output D2H is not yet enqueued")),
            "B before A finish/D2H must be the Work-output state error: {early:?}"
        );
        prepared_first
            .finish_rune(&stream, &mut work, &mut first_lease)
            .expect("A D2H event");
        unsafe {
            prepared_second
                .enqueue_from_lease(
                    &blas,
                    &stream,
                    &hidden_gpu,
                    &weight_gpu,
                    &mut work,
                    &mut second_lease,
                    second_offset,
                    &kernels,
                )
                .expect("B bound lease enqueue");
        }
        prepared_second
            .finish_rune(&stream, &mut work, &mut second_lease)
            .expect("B D2H event");
        second_lease.wait().expect("B wait");
        first_lease.wait().expect("A wait");
        let actual_first: Vec<u32> = first_lease
            .output(first.len())
            .expect("A output")
            .iter()
            .map(|value| value.to_bits())
            .collect();
        let actual_second: Vec<u32> = second_lease
            .output(second.len())
            .expect("B output")
            .iter()
            .map(|value| value.to_bits())
            .collect();
        assert_eq!(actual_first, expected_first, "A result was overwritten");
        assert_eq!(actual_second, expected_second, "B result differs");
        assert!(prepared_first
            .preflight(
                &blas,
                &second_stream,
                &hidden_gpu,
                &weight_gpu,
                &work,
                &first_lease.slot().expect("first slot").input,
                first_offset,
                &kernels,
            )
            .is_err());
        let other_work =
            Gemma4OptionWork::new(other_stream.clone(), capacity).expect("other context Work");
        assert!(prepared_first
            .preflight(
                &blas,
                &stream,
                &hidden_gpu,
                &weight_gpu,
                &other_work,
                &first_lease.slot().expect("first slot").input,
                first_offset,
                &kernels,
            )
            .is_err());
        drop(other_work);
        unsafe { work.mark_stream_drained() };
        // Early lease drop after enqueue: Drop retains/synchronizes the pinned
        // source, and the shared Work still requires an explicit drain proof
        // before its next use.
        drop(first_lease);
        let early_slot = pool.acquire().expect("early-drop lease slot");
        let mut early_lease = HostIoLease::detached(early_slot, pool.clone(), stream.clone());
        early_lease
            .prepare((token_rows, boundaries, first.len()), first.len(), false)
            .expect("prepare early lease");
        let early_offset = stage_option_ids(
            &mut early_lease,
            (token_rows, boundaries, first.len()),
            &first,
        )
        .expect("stage early ids");
        unsafe {
            prepared_first
                .enqueue_from_lease(
                    &blas,
                    &stream,
                    &hidden_gpu,
                    &weight_gpu,
                    &mut work,
                    &mut early_lease,
                    early_offset,
                    &kernels,
                )
                .expect("early-drop enqueue");
        }
        drop(early_lease);
        stream.synchronize().expect("early-drop completion");
        unsafe { work.mark_stream_drained() };
    }
}
