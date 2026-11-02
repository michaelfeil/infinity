// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Standalone OneJev option-token readout foundation.
//!
//! The native LibTorch contract in `decision_models.cpp` resolves OneJev as
//! follows:
//!
//! * A checkpoint without `decision_config.json` selects the `option_tokens`
//!   head, not the external 255-row `pplx`/`d3` readout.
//! * `OptionHead::lexical` returns the checkpoint's tied token embedding or its
//!   explicit `lm_head.weight`. Elephant and Mammoth set
//!   `tie_word_embeddings=false` and store `lm_head.weight` as BF16
//!   `[vocab_size, hidden_size]`.
//! * The Qwen branch selects answer-token rows, casts the last real token to
//!   FP32, and projects with FP32 math. It does not project the full vocabulary
//!   and does not apply `final_logit_softcapping`.
//! * `Warmup` reads readout row zero from each sequence's last real token.
//!   Pooled readout is only valid when every request consumes its final token.
//!
//! This module is a cold metadata/weight loader plus an owned CPU readout
//! reference. It does not admit `ModelType::Decision` into Oxide, does not
//! create CUDA state, and does not implement the Qwen3.5 backbone or any other
//! decision head. Other decision protocols fail explicitly.

use crate::model::{path_present, read_json, Checkpoint, WeightSource};
use half::bf16;
use infinity_v3_backend_core::{block_attention_config, DecisionInput, DecisionOutput};
use safetensors::Dtype;
use serde::Deserialize;
use std::{
    fmt,
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
};

const ONEJEV_FAMILIES: [&str; 4] = ["qwen3_5", "qwen3_5_moe", "qwen3_5_text", "qwen3_5_moe_text"];
/// Native `lexical` fallback order for tied token embeddings.
const TIED_NAMESPACES: [&str; 4] = [
    "model.language_model.embed_tokens.weight",
    "model.embed_tokens.weight",
    "language_model.embed_tokens.weight",
    "embed_tokens.weight",
];
const LM_HEAD: &str = "lm_head.weight";
const DECISION_CONFIG: &str = "decision_config.json";
const D3_FORMAT: &str = "d3-code-readout-v1";
static NEXT_HEAD_ID: AtomicU64 = AtomicU64::new(0);

/// Readout contract selected by the checkpoint's cold metadata.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReadoutFamily {
    /// OneJev's Qwen lexical vocabulary projection over BF16 checkpoint rows.
    OneJevLexical,
    /// Perplexity's external 255-row BF16 readout. Not a OneJev head.
    ExternalPerplexityBf16,
    /// D3's external 255-row FP32 readout. Not a OneJev head.
    ExternalD3F32,
}

impl ReadoutFamily {
    fn label(self) -> &'static str {
        match self {
            Self::OneJevLexical => "OneJev Qwen lexical FP32 projection",
            Self::ExternalPerplexityBf16 => "external 255-row BF16 pplx readout",
            Self::ExternalD3F32 => "external 255-row FP32 d3 readout",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum DecisionHeadError {
    UnsupportedReadout(ReadoutFamily),
    InvalidExternalMetadata(String),
    InvalidConfig(String),
    MissingWeight(&'static str),
    InvalidWeightGeometry {
        name: &'static str,
        shape: Vec<usize>,
    },
    WrongWeightDtype {
        name: &'static str,
        found: String,
    },
    ConfigWidthMismatch {
        configured: usize,
        weights: usize,
    },
    InvalidBatch(&'static str),
    UnsupportedRequest(&'static str),
    EmptyOptions,
    TokenIdOutOfRange {
        request: usize,
        token_id: u32,
        rows: usize,
    },
    PrefixOutOfRange {
        request: usize,
        prefix_length: usize,
        length: usize,
    },
    OutputCountOverflow,
    Allocation,
    WeightRead(String),
    PreparedMismatch(&'static str),
    HiddenShape(&'static str),
}

impl fmt::Display for DecisionHeadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedReadout(family) => {
                write!(f, "{} is not the OneJev option-token head", family.label())
            }
            Self::InvalidExternalMetadata(error) => {
                write!(f, "invalid decision_config.json: {error}")
            }
            Self::InvalidConfig(error) => write!(f, "invalid OneJev config: {error}"),
            Self::MissingWeight(name) => write!(f, "missing OneJev readout weight {name}"),
            Self::InvalidWeightGeometry { name, shape } => {
                write!(f, "invalid OneJev readout geometry for {name}: {shape:?}")
            }
            Self::WrongWeightDtype { name, found } => {
                write!(f, "OneJev readout {name} must be BF16, found {found}")
            }
            Self::ConfigWidthMismatch {
                configured,
                weights,
            } => write!(
                f,
                "OneJev hidden width mismatch: config={configured}, readout={weights}"
            ),
            Self::InvalidBatch(reason) => write!(f, "invalid packed decision batch: {reason}"),
            Self::UnsupportedRequest(reason) => {
                write!(f, "unsupported OneJev decision request: {reason}")
            }
            Self::EmptyOptions => f.write_str("OneJev option-token metadata must not be empty"),
            Self::TokenIdOutOfRange {
                request,
                token_id,
                rows,
            } => write!(
                f,
                "OneJev request {request} token id {token_id} is outside 0..{rows}"
            ),
            Self::PrefixOutOfRange {
                request,
                prefix_length,
                length,
            } => write!(
                f,
                "OneJev request {request} prefix length {prefix_length} is outside 1..={length}"
            ),
            Self::OutputCountOverflow => f.write_str("OneJev option count overflow"),
            Self::Allocation => f.write_str("OneJev readout allocation failed"),
            Self::WeightRead(error) => write!(f, "OneJev readout weight read failed: {error}"),
            Self::PreparedMismatch(reason) => {
                write!(
                    f,
                    "prepared OneJev readout does not match request: {reason}"
                )
            }
            Self::HiddenShape(reason) => write!(f, "invalid OneJev hidden states: {reason}"),
        }
    }
}

impl std::error::Error for DecisionHeadError {}

/// Resolve the cold readout family exactly as the native loader does.
///
/// `decision_config.json` selects the external pplx/d3 readout. Its
/// `format_id` distinguishes D3's FP32 readout from Perplexity's BF16
/// readout. OneJev checkpoints have no decision config and use the lexical
/// token-embedding projection instead.
pub(crate) fn classify_readout(path: &Path) -> Result<ReadoutFamily, DecisionHeadError> {
    let metadata = path.join(DECISION_CONFIG);
    if !path_present(&metadata)
        .map_err(|error| DecisionHeadError::InvalidExternalMetadata(error.to_string()))?
    {
        return Ok(ReadoutFamily::OneJevLexical);
    }
    let config: serde_json::Value = read_json(&metadata)
        .map_err(|error| DecisionHeadError::InvalidExternalMetadata(error.to_string()))?;
    let format = config.get("format_id").and_then(serde_json::Value::as_str);
    Ok(if format == Some(D3_FORMAT) {
        ReadoutFamily::ExternalD3F32
    } else {
        ReadoutFamily::ExternalPerplexityBf16
    })
}

#[derive(Debug, Deserialize)]
struct RawTextConfig {
    hidden_size: Option<usize>,
    tie_word_embeddings: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct RawOneJevConfig {
    model_type: String,
    hidden_size: Option<usize>,
    tie_word_embeddings: Option<bool>,
    text_config: Option<RawTextConfig>,
}

impl RawOneJevConfig {
    fn load(path: &Path) -> Result<Self, DecisionHeadError> {
        read_json(&path.join("config.json"))
            .map_err(|error| DecisionHeadError::InvalidConfig(error.to_string()))
    }

    fn hidden_size(&self) -> Result<usize, DecisionHeadError> {
        self.text_config
            .as_ref()
            .and_then(|config| config.hidden_size)
            .or(self.hidden_size)
            .filter(|&hidden| hidden > 0)
            .ok_or_else(|| {
                DecisionHeadError::InvalidConfig(
                    "missing positive hidden_size/text_config.hidden_size".into(),
                )
            })
    }

    fn tied(&self) -> bool {
        self.tie_word_embeddings.unwrap_or(false)
            || self
                .text_config
                .as_ref()
                .and_then(|config| config.tie_word_embeddings)
                .unwrap_or(false)
    }
}

/// Validated packed boundaries for one decision batch.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PackedDecisionBatch<'a> {
    cumulative_seq_lengths: &'a [u32],
    tokens: usize,
}

impl<'a> PackedDecisionBatch<'a> {
    pub(crate) fn new(cumulative_seq_lengths: &'a [u32]) -> Result<Self, DecisionHeadError> {
        if cumulative_seq_lengths.len() < 2 {
            return Err(DecisionHeadError::InvalidBatch(
                "at least one nonempty sequence is required",
            ));
        }
        if cumulative_seq_lengths.first() != Some(&0) {
            return Err(DecisionHeadError::InvalidBatch(
                "cumulative boundaries must start at zero",
            ));
        }
        let mut previous = 0usize;
        for &boundary in cumulative_seq_lengths.iter().skip(1) {
            let boundary = usize::try_from(boundary)
                .map_err(|_| DecisionHeadError::InvalidBatch("boundary exceeds usize"))?;
            if boundary <= previous {
                return Err(DecisionHeadError::InvalidBatch(
                    "decision sequences must be nonempty",
                ));
            }
            previous = boundary;
        }
        if previous == 0 {
            return Err(DecisionHeadError::InvalidBatch(
                "packed decision batch must contain tokens",
            ));
        }
        Ok(Self {
            cumulative_seq_lengths,
            tokens: previous,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.cumulative_seq_lengths.len() - 1
    }

    pub(crate) fn tokens(&self) -> usize {
        self.tokens
    }

    pub(crate) fn boundaries(&self) -> &[u32] {
        self.cumulative_seq_lengths
    }

    pub(crate) fn sequence(&self, row: usize) -> Result<(usize, usize), DecisionHeadError> {
        let next = row
            .checked_add(1)
            .ok_or(DecisionHeadError::InvalidBatch("sequence index overflow"))?;
        let start = self.cumulative_seq_lengths.get(row).copied().ok_or(
            DecisionHeadError::InvalidBatch("request index exceeds packed batch"),
        )?;
        let end = self.cumulative_seq_lengths.get(next).copied().ok_or(
            DecisionHeadError::InvalidBatch("request index exceeds packed batch"),
        )?;
        Ok((
            usize::try_from(start)
                .map_err(|_| DecisionHeadError::InvalidBatch("sequence start exceeds usize"))?,
            usize::try_from(end)
                .map_err(|_| DecisionHeadError::InvalidBatch("sequence end exceeds usize"))?,
        ))
    }
}

/// Cold request plan shared by the CPU reference and the prepared GPU head.
///
/// `row_ids` already contains the exact request order, including duplicate
/// option ids. `last_rows` names the last real packed token of each request.
/// Building this plan reads no checkpoint payload and allocates no device
/// storage, so callers can reuse it for either execution backend.
#[derive(Clone, Debug)]
pub(crate) struct RequestPlan {
    pub(crate) counts: Vec<usize>,
    pub(crate) row_offsets: Vec<usize>,
    pub(crate) row_ids: Vec<u32>,
    pub(crate) boundaries: Vec<u32>,
    pub(crate) last_rows: Vec<u32>,
    pub(crate) total: usize,
}

/// Validate one request and return its logical option count.
pub(crate) fn output_count_for(
    vocab_rows: usize,
    request: usize,
    input: &DecisionInput,
) -> Result<usize, DecisionHeadError> {
    match input {
        DecisionInput::Warmup => Ok(1),
        DecisionInput::OptionTokens { token_ids }
        | DecisionInput::BlockOptionTokens { token_ids, .. } => {
            if token_ids.is_empty() {
                return Err(DecisionHeadError::EmptyOptions);
            }
            for &token_id in token_ids {
                if usize::try_from(token_id).map_or(true, |id| id >= vocab_rows) {
                    return Err(DecisionHeadError::TokenIdOutOfRange {
                        request,
                        token_id,
                        rows: vocab_rows,
                    });
                }
            }
            Ok(token_ids.len())
        }
        DecisionInput::InlineOptions { .. } => Err(DecisionHeadError::UnsupportedRequest(
            "InlineOptions belongs to the external pplx/d3 readout",
        )),
        DecisionInput::Laya { .. } => Err(DecisionHeadError::UnsupportedRequest(
            "Laya is not the OneJev option-token head",
        )),
        DecisionInput::Clef { .. } => Err(DecisionHeadError::UnsupportedRequest(
            "Clef is not the OneJev option-token head",
        )),
    }
}

/// Native `OptionHead::last_token_only` eligibility.
///
/// Warmup deliberately stays full-token, so it disqualifies pooling even when
/// every other request reads only its last token.
pub(crate) fn requests_are_last_token_only(
    batch: &PackedDecisionBatch<'_>,
    requests: &[DecisionInput],
) -> bool {
    requests.len() == batch.len()
        && requests.iter().all(|request| {
            matches!(
                request,
                DecisionInput::OptionTokens { .. } | DecisionInput::BlockOptionTokens { .. }
            )
        })
}

/// Validate a whole packed request batch and build its immutable row plan.
pub(crate) fn plan_decision_requests(
    vocab_rows: usize,
    batch: &PackedDecisionBatch<'_>,
    requests: &[DecisionInput],
) -> Result<RequestPlan, DecisionHeadError> {
    if requests.len() != batch.len() {
        return Err(DecisionHeadError::InvalidBatch(
            "request count does not match packed sequences",
        ));
    }
    let mut counts = Vec::new();
    counts
        .try_reserve_exact(requests.len())
        .map_err(|_| DecisionHeadError::Allocation)?;
    let offsets = requests
        .len()
        .checked_add(1)
        .ok_or(DecisionHeadError::OutputCountOverflow)?;
    let mut row_offsets = Vec::new();
    row_offsets
        .try_reserve_exact(offsets)
        .map_err(|_| DecisionHeadError::Allocation)?;
    row_offsets.push(0usize);
    let mut total = 0usize;
    for (index, request) in requests.iter().enumerate() {
        let count = output_count_for(vocab_rows, index, request)?;
        if let DecisionInput::BlockOptionTokens { prefix_length, .. } = request {
            let (start, end) = batch.sequence(index)?;
            let length = end
                .checked_sub(start)
                .ok_or(DecisionHeadError::InvalidBatch("sequence length overflow"))?;
            if *prefix_length == 0 || *prefix_length > length {
                return Err(DecisionHeadError::PrefixOutOfRange {
                    request: index,
                    prefix_length: *prefix_length,
                    length,
                });
            }
        }
        counts.push(count);
        total = total
            .checked_add(count)
            .ok_or(DecisionHeadError::OutputCountOverflow)?;
        row_offsets.push(total);
    }
    let mut row_ids = Vec::new();
    row_ids
        .try_reserve_exact(total)
        .map_err(|_| DecisionHeadError::Allocation)?;
    let mut last_rows = Vec::new();
    last_rows
        .try_reserve_exact(requests.len())
        .map_err(|_| DecisionHeadError::Allocation)?;
    for (index, request) in requests.iter().enumerate() {
        match request {
            DecisionInput::Warmup => row_ids.push(0),
            DecisionInput::OptionTokens { token_ids }
            | DecisionInput::BlockOptionTokens { token_ids, .. } => {
                row_ids.extend_from_slice(token_ids);
            }
            DecisionInput::InlineOptions { .. }
            | DecisionInput::Laya { .. }
            | DecisionInput::Clef { .. } => {
                return Err(DecisionHeadError::UnsupportedRequest(
                    "only OptionTokens, BlockOptionTokens and Warmup are OneJev requests",
                ));
            }
        }
        let (_, end) = batch.sequence(index)?;
        let last = end.checked_sub(1).ok_or(DecisionHeadError::InvalidBatch(
            "decision sequences must be nonempty",
        ))?;
        last_rows.push(
            u32::try_from(last)
                .map_err(|_| DecisionHeadError::InvalidBatch("hidden row exceeds u32"))?,
        );
    }
    let mut boundaries = Vec::new();
    boundaries
        .try_reserve_exact(batch.cumulative_seq_lengths.len())
        .map_err(|_| DecisionHeadError::Allocation)?;
    boundaries.extend_from_slice(batch.cumulative_seq_lengths);
    Ok(RequestPlan {
        counts,
        row_offsets,
        row_ids,
        boundaries,
        last_rows,
        total,
    })
}

#[derive(Clone, Copy)]
enum HiddenData<'a> {
    F32(&'a [f32]),
    Bf16(&'a [u16]),
}

/// Borrowed packed hidden rows, either native FP32 or BF16 bit patterns.
#[derive(Clone, Copy)]
pub(crate) struct HiddenRows<'a> {
    data: HiddenData<'a>,
    rows: usize,
    width: usize,
}

impl<'a> HiddenRows<'a> {
    fn checked(data: HiddenData<'a>, width: usize) -> Result<Self, DecisionHeadError> {
        if width == 0 {
            return Err(DecisionHeadError::HiddenShape(
                "hidden width must be positive",
            ));
        }
        let len = match data {
            HiddenData::F32(values) => values.len(),
            HiddenData::Bf16(values) => values.len(),
        };
        if len % width != 0 {
            return Err(DecisionHeadError::HiddenShape(
                "hidden storage is not a whole number of rows",
            ));
        }
        Ok(Self {
            data,
            rows: len / width,
            width,
        })
    }

    pub(crate) fn from_f32(values: &'a [f32], width: usize) -> Result<Self, DecisionHeadError> {
        Self::checked(HiddenData::F32(values), width)
    }

    pub(crate) fn from_bf16(values: &'a [u16], width: usize) -> Result<Self, DecisionHeadError> {
        Self::checked(HiddenData::Bf16(values), width)
    }

    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    pub(crate) fn width(&self) -> usize {
        self.width
    }

    pub(crate) fn dot(&self, row: usize, weights: &[f32]) -> Result<f32, DecisionHeadError> {
        if weights.len() != self.width {
            return Err(DecisionHeadError::HiddenShape(
                "readout row width does not match hidden width",
            ));
        }
        let start = row
            .checked_mul(self.width)
            .ok_or(DecisionHeadError::HiddenShape("hidden row offset overflow"))?;
        let end = start
            .checked_add(self.width)
            .ok_or(DecisionHeadError::HiddenShape("hidden row end overflow"))?;
        match self.data {
            HiddenData::F32(values) => {
                let values = values
                    .get(start..end)
                    .ok_or(DecisionHeadError::HiddenShape("hidden row exceeds storage"))?;
                Ok(values
                    .iter()
                    .zip(weights)
                    .fold(0.0f32, |sum, (value, weight)| sum + value * weight))
            }
            HiddenData::Bf16(values) => {
                let values = values
                    .get(start..end)
                    .ok_or(DecisionHeadError::HiddenShape("hidden row exceeds storage"))?;
                Ok(values
                    .iter()
                    .zip(weights)
                    .fold(0.0f32, |sum, (bits, weight)| {
                        sum + bf16::from_bits(*bits).to_f32() * weight
                    }))
            }
        }
    }
}

/// Answer rows selected by one prepared batch.
pub(crate) struct PreparedDecision {
    head_id: u64,
    hidden: usize,
    vocab_rows: usize,
    boundaries: Vec<u32>,
    row_ids: Vec<u32>,
    counts: Vec<usize>,
    row_offsets: Vec<usize>,
    selected: Vec<f32>,
}

impl PreparedDecision {
    pub(crate) fn output_counts(&self) -> &[usize] {
        &self.counts
    }

    pub(crate) fn selected_rows(&self) -> usize {
        self.selected.len() / self.hidden
    }
}

/// Owned cold OneJev readout over a checkpoint's BF16 lexical matrix.
pub(crate) struct OneJevHead {
    id: u64,
    weight: WeightSource,
    namespace: &'static str,
    vocab_rows: usize,
    hidden: usize,
    block_attention: bool,
}

impl OneJevHead {
    /// Load only OneJev metadata and the lexical readout handle.
    ///
    /// No tensor payload is read here. Selected answer rows are read later by
    /// `prepare`, so a 248k-row vocabulary is never materialized on the host.
    pub(crate) fn load(path: &Path) -> Result<Self, DecisionHeadError> {
        let family = classify_readout(path)?;
        if family != ReadoutFamily::OneJevLexical {
            return Err(DecisionHeadError::UnsupportedReadout(family));
        }
        let config = RawOneJevConfig::load(path)?;
        if !ONEJEV_FAMILIES.contains(&config.model_type.as_str()) {
            return Err(DecisionHeadError::InvalidConfig(format!(
                "OneJev requires a Qwen3.5 model, found {}",
                config.model_type
            )));
        }
        let hidden = config.hidden_size()?;
        let block_attention = block_attention_config::block_attention(path)
            .map_err(DecisionHeadError::InvalidConfig)?;
        let checkpoint = Checkpoint::open(path)
            .map_err(|error| DecisionHeadError::WeightRead(error.to_string()))?;
        let namespace = if config.tied() {
            TIED_NAMESPACES
                .iter()
                .copied()
                .find(|name| checkpoint.contains(name))
                .ok_or(DecisionHeadError::MissingWeight("tied embed_tokens.weight"))?
        } else {
            if !checkpoint.contains(LM_HEAD) {
                return Err(DecisionHeadError::MissingWeight(LM_HEAD));
            }
            LM_HEAD
        };
        let shape = checkpoint
            .shape(namespace)
            .ok_or(DecisionHeadError::MissingWeight(namespace))?
            .to_vec();
        if shape.len() != 2 || shape.first() == Some(&0) || shape.get(1) == Some(&0) {
            return Err(DecisionHeadError::InvalidWeightGeometry {
                name: namespace,
                shape,
            });
        }
        let weights_width = shape[1];
        if weights_width != hidden {
            return Err(DecisionHeadError::ConfigWidthMismatch {
                configured: hidden,
                weights: weights_width,
            });
        }
        let weight = checkpoint
            .source(namespace, &shape)
            .map_err(|error| DecisionHeadError::WeightRead(error.to_string()))?;
        if weight.dtype() != Dtype::BF16 {
            return Err(DecisionHeadError::WrongWeightDtype {
                name: namespace,
                found: format!("{:?}", weight.dtype()),
            });
        }
        let elements = shape[0].checked_mul(shape[1]).ok_or_else(|| {
            DecisionHeadError::InvalidWeightGeometry {
                name: namespace,
                shape: shape.clone(),
            }
        })?;
        if weight.len() != elements {
            return Err(DecisionHeadError::InvalidWeightGeometry {
                name: namespace,
                shape,
            });
        }
        Ok(Self {
            id: NEXT_HEAD_ID.fetch_add(1, Ordering::Relaxed),
            weight,
            namespace,
            vocab_rows: shape[0],
            hidden,
            block_attention,
        })
    }

    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    /// Consume the cold metadata handle and transfer the canonical BF16 row
    /// source to a prepared GPU upload. No payload has been read yet.
    pub(crate) fn into_weight_source(self) -> WeightSource {
        self.weight
    }

    pub(crate) fn namespace(&self) -> &'static str {
        self.namespace
    }

    pub(crate) fn hidden_size(&self) -> usize {
        self.hidden
    }

    pub(crate) fn vocab_rows(&self) -> usize {
        self.vocab_rows
    }

    /// Whether the checkpoint's shared modeling metadata selects OneJev block
    /// attention. The head does not consume the flag; the future backbone adapter
    /// uses it to choose `BlockOptionTokens` versus `OptionTokens`.
    pub(crate) fn block_attention(&self) -> bool {
        self.block_attention
    }

    /// Validate metadata and report the reserved option-logit count.
    pub(crate) fn output_count(&self, request: &DecisionInput) -> Result<usize, DecisionHeadError> {
        self.output_count_for(0, request)
    }

    /// Native OptionHead eligibility for a pooled backbone readout.
    ///
    /// Warmup deliberately stays full-token, so it disqualifies pooling even
    /// when every other request reads only its last token.
    pub(crate) fn last_token_only(
        &self,
        batch: &PackedDecisionBatch<'_>,
        requests: &[DecisionInput],
    ) -> bool {
        requests_are_last_token_only(batch, requests)
    }

    /// Select and copy answer rows once before the backbone starts.
    ///
    /// This is the cold allocation plan for the request batch. `forward` only
    /// consumes the resulting FP32 rows and allocates its returned logits.
    pub(crate) fn prepare(
        &self,
        batch: &PackedDecisionBatch<'_>,
        requests: &[DecisionInput],
    ) -> Result<PreparedDecision, DecisionHeadError> {
        let plan = plan_decision_requests(self.vocab_rows, batch, requests)?;
        let elements = plan
            .total
            .checked_mul(self.hidden)
            .ok_or(DecisionHeadError::OutputCountOverflow)?;
        let mut selected = Vec::new();
        selected
            .try_reserve_exact(elements)
            .map_err(|_| DecisionHeadError::Allocation)?;
        let mut scratch = Vec::new();
        scratch
            .try_reserve_exact(self.hidden)
            .map_err(|_| DecisionHeadError::Allocation)?;
        scratch.extend(std::iter::repeat_n(0u16, self.hidden));
        for &token_id in &plan.row_ids {
            self.read_row(token_id, &mut scratch, &mut selected)?;
        }
        Ok(PreparedDecision {
            head_id: self.id,
            hidden: self.hidden,
            vocab_rows: self.vocab_rows,
            boundaries: plan.boundaries,
            row_ids: plan.row_ids,
            counts: plan.counts,
            row_offsets: plan.row_offsets,
            selected,
        })
    }

    fn output_count_for(
        &self,
        request: usize,
        input: &DecisionInput,
    ) -> Result<usize, DecisionHeadError> {
        output_count_for(self.vocab_rows, request, input)
    }

    fn read_row(
        &self,
        token_id: u32,
        scratch: &mut [u16],
        selected: &mut Vec<f32>,
    ) -> Result<(), DecisionHeadError> {
        let row = usize::try_from(token_id).map_err(|_| DecisionHeadError::TokenIdOutOfRange {
            request: 0,
            token_id,
            rows: self.vocab_rows,
        })?;
        let offset = row
            .checked_mul(self.hidden)
            .ok_or(DecisionHeadError::OutputCountOverflow)?;
        self.weight
            .read_bf16(offset, scratch)
            .map_err(|error| DecisionHeadError::WeightRead(error.to_string()))?;
        for &bits in scratch.iter() {
            selected.push(bf16::from_bits(bits).to_f32());
        }
        Ok(())
    }

    /// Read packed last-token states from complete backbone hidden rows.
    pub(crate) fn forward(
        &self,
        hidden: HiddenRows<'_>,
        batch: &PackedDecisionBatch<'_>,
        requests: &[DecisionInput],
        prepared: &PreparedDecision,
    ) -> Result<Vec<DecisionOutput>, DecisionHeadError> {
        self.readout(hidden, batch, requests, prepared, false)
    }

    /// Read pooled one-row-per-request states.
    ///
    /// As in the native head, pooling is refused for any Warmup or other
    /// full-token request.
    pub(crate) fn forward_pooled(
        &self,
        hidden: HiddenRows<'_>,
        batch: &PackedDecisionBatch<'_>,
        requests: &[DecisionInput],
        prepared: &PreparedDecision,
    ) -> Result<Vec<DecisionOutput>, DecisionHeadError> {
        if !self.last_token_only(batch, requests) {
            return Err(DecisionHeadError::UnsupportedRequest(
                "pooled decisions require last-token-only requests",
            ));
        }
        self.readout(hidden, batch, requests, prepared, true)
    }

    fn readout(
        &self,
        hidden: HiddenRows<'_>,
        batch: &PackedDecisionBatch<'_>,
        requests: &[DecisionInput],
        prepared: &PreparedDecision,
        pooled: bool,
    ) -> Result<Vec<DecisionOutput>, DecisionHeadError> {
        if requests.len() != batch.len() {
            return Err(DecisionHeadError::InvalidBatch(
                "request count does not match packed sequences",
            ));
        }
        if hidden.width() != self.hidden {
            return Err(DecisionHeadError::HiddenShape(
                "hidden width does not match the OneJev readout",
            ));
        }
        if pooled {
            if hidden.rows() != batch.len() {
                return Err(DecisionHeadError::HiddenShape(
                    "pooled hidden row count does not match requests",
                ));
            }
        } else if hidden.rows() != batch.tokens() {
            return Err(DecisionHeadError::HiddenShape(
                "packed hidden row count does not match token boundaries",
            ));
        }
        if prepared.head_id != self.id
            || prepared.hidden != self.hidden
            || prepared.vocab_rows != self.vocab_rows
            || prepared.boundaries.as_slice() != batch.cumulative_seq_lengths
            || prepared.counts.len() != requests.len()
            || prepared.row_offsets.len() != requests.len() + 1
        {
            return Err(DecisionHeadError::PreparedMismatch(
                "prepared metadata belongs to another head or batch",
            ));
        }
        let mut results = Vec::new();
        results
            .try_reserve_exact(requests.len())
            .map_err(|_| DecisionHeadError::Allocation)?;
        for (index, request) in requests.iter().enumerate() {
            let expected = self.output_count_for(index, request)?;
            let count = prepared
                .counts
                .get(index)
                .copied()
                .ok_or(DecisionHeadError::PreparedMismatch("missing output count"))?;
            if count != expected {
                return Err(DecisionHeadError::PreparedMismatch(
                    "output count does not match request metadata",
                ));
            }
            let (_, end) = batch.sequence(index)?;
            let hidden_row = if pooled {
                index
            } else {
                end.checked_sub(1).ok_or(DecisionHeadError::InvalidBatch(
                    "decision sequences must be nonempty",
                ))?
            };
            let row_begin = prepared
                .row_offsets
                .get(index)
                .copied()
                .ok_or(DecisionHeadError::PreparedMismatch("missing row offset"))?;
            let row_end = prepared
                .row_offsets
                .get(index + 1)
                .copied()
                .ok_or(DecisionHeadError::PreparedMismatch("missing row offset"))?;
            if row_end.checked_sub(row_begin) != Some(count) {
                return Err(DecisionHeadError::PreparedMismatch(
                    "prepared row range does not match output count",
                ));
            }
            let prepared_ids = prepared.row_ids.get(row_begin..row_end).ok_or(
                DecisionHeadError::PreparedMismatch("prepared row ids are incomplete"),
            )?;
            let same_ids = match request {
                DecisionInput::Warmup => prepared_ids == [0],
                DecisionInput::OptionTokens { token_ids }
                | DecisionInput::BlockOptionTokens { token_ids, .. } => {
                    prepared_ids == token_ids.as_slice()
                }
                DecisionInput::InlineOptions { .. }
                | DecisionInput::Laya { .. }
                | DecisionInput::Clef { .. } => false,
            };
            if !same_ids {
                return Err(DecisionHeadError::PreparedMismatch(
                    "prepared answer rows do not match request metadata",
                ));
            }
            let mut logits = Vec::new();
            logits
                .try_reserve_exact(count)
                .map_err(|_| DecisionHeadError::Allocation)?;
            for option in 0..count {
                let row = row_begin
                    .checked_add(option)
                    .ok_or(DecisionHeadError::OutputCountOverflow)?;
                let weight_start = row
                    .checked_mul(self.hidden)
                    .ok_or(DecisionHeadError::OutputCountOverflow)?;
                let weight_end = weight_start
                    .checked_add(self.hidden)
                    .ok_or(DecisionHeadError::OutputCountOverflow)?;
                let weights = prepared.selected.get(weight_start..weight_end).ok_or(
                    DecisionHeadError::PreparedMismatch("prepared readout rows are incomplete"),
                )?;
                logits.push(hidden.dot(hidden_row, weights)?);
            }
            results.push(DecisionOutput {
                logits,
                action_probability: 1.0,
                memory: None,
            });
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{serialize_to_file, TensorView};
    use serde_json::{json, Value};
    use std::{
        fs,
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
    };

    const HIDDEN: usize = 3;
    const ROWS: usize = 7;
    const OFFSETS: [u32; 3] = [0, 3, 5];

    struct Fixture(PathBuf);

    impl Fixture {
        fn new(config: Value) -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "infinity-onejev-head-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            fs::write(
                path.join("config.json"),
                serde_json::to_vec(&config).unwrap(),
            )
            .unwrap();
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }

        fn weights(&self, tensors: &[(&str, Dtype, Vec<usize>, Vec<u8>)]) {
            let views = tensors
                .iter()
                .map(|(name, dtype, shape, bytes)| {
                    (
                        name.to_string(),
                        TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
                    )
                })
                .collect::<Vec<_>>();
            serialize_to_file(
                views
                    .iter()
                    .map(|(name, view)| (name.clone(), view.clone())),
                None,
                &self.0.join("model.safetensors"),
            )
            .unwrap();
        }

        fn decision_config(&self, value: Value) {
            fs::write(
                self.0.join(DECISION_CONFIG),
                serde_json::to_vec(&value).unwrap(),
            )
            .unwrap();
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn config() -> Value {
        json!({
            "model_type": "qwen3_5",
            "tie_word_embeddings": false,
            "final_logit_softcapping": 0.5,
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": HIDDEN,
                "tie_word_embeddings": false
            },
            "modeling_overrides": {
                "attention": {
                    "full_attention": "block_bidirectional",
                    "linear_attention": "causal"
                },
                "readout": "slot_lm"
            }
        })
    }

    fn weight_value(row: usize, column: usize) -> f32 {
        (row * 16 + column) as f32 * 0.25
    }

    fn hidden_value(row: usize, column: usize) -> f32 {
        (row * 8 + column) as f32 * 0.5
    }

    fn bf16_bytes(values: impl Iterator<Item = f32>) -> Vec<u8> {
        values
            .flat_map(|value| bf16::from_f32(value).to_bits().to_le_bytes())
            .collect()
    }

    fn lexical_bytes() -> Vec<u8> {
        bf16_bytes(
            (0..ROWS).flat_map(|row| (0..HIDDEN).map(move |column| weight_value(row, column))),
        )
    }

    fn fixture() -> (Fixture, OneJevHead) {
        let fixture = Fixture::new(config());
        fixture.weights(&[(LM_HEAD, Dtype::BF16, vec![ROWS, HIDDEN], lexical_bytes())]);
        let head = OneJevHead::load(fixture.path()).unwrap();
        (fixture, head)
    }

    fn option(token_ids: &[u32]) -> DecisionInput {
        DecisionInput::OptionTokens {
            token_ids: token_ids.to_vec(),
        }
    }

    fn expected(row: usize, hidden_row: usize) -> f32 {
        (0..HIDDEN)
            .map(|column| weight_value(row, column) * hidden_value(hidden_row, column))
            .sum()
    }

    #[test]
    fn resolves_external_readouts_and_rejects_them_for_onejev() {
        let no_config = Fixture::new(config());
        assert_eq!(
            classify_readout(no_config.path()).unwrap(),
            ReadoutFamily::OneJevLexical
        );

        let d3 = Fixture::new(config());
        d3.decision_config(json!({"format_id": D3_FORMAT}));
        assert_eq!(
            classify_readout(d3.path()).unwrap(),
            ReadoutFamily::ExternalD3F32
        );
        assert!(matches!(
            OneJevHead::load(d3.path()),
            Err(DecisionHeadError::UnsupportedReadout(
                ReadoutFamily::ExternalD3F32
            ))
        ));

        let pplx = Fixture::new(config());
        pplx.decision_config(json!({"format_version": 2}));
        assert_eq!(
            classify_readout(pplx.path()).unwrap(),
            ReadoutFamily::ExternalPerplexityBf16
        );
        assert!(matches!(
            OneJevHead::load(pplx.path()),
            Err(DecisionHeadError::UnsupportedReadout(
                ReadoutFamily::ExternalPerplexityBf16
            ))
        ));
    }

    #[test]
    fn selects_answer_rows_at_last_real_token_without_qwen_softcap() {
        let (_fixture, head) = fixture();
        assert_eq!(head.namespace(), LM_HEAD);
        assert_eq!(head.hidden_size(), HIDDEN);
        assert_eq!(head.vocab_rows(), ROWS);
        assert!(head.block_attention());

        let batch = PackedDecisionBatch::new(&OFFSETS).unwrap();
        let requests = [option(&[4, 1]), DecisionInput::Warmup];
        let prepared = head.prepare(&batch, &requests).unwrap();
        assert_eq!(prepared.output_counts(), &[2, 1]);
        assert_eq!(prepared.selected_rows(), 3);

        let rows = (0..5)
            .flat_map(|row| (0..HIDDEN).map(move |column| hidden_value(row, column)))
            .collect::<Vec<_>>();
        let hidden = HiddenRows::from_f32(&rows, HIDDEN).unwrap();
        let results = head.forward(hidden, &batch, &requests, &prepared).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].logits, vec![expected(4, 2), expected(1, 2)]);
        assert_eq!(results[1].logits, vec![expected(0, 4)]);
        assert_eq!(results[0].action_probability, 1.0);
        assert!(results[0].memory.is_none());
        let single = PackedDecisionBatch::new(&[0, 3]).unwrap();
        assert!(head.last_token_only(&single, &[option(&[1])]));
        assert!(!head.last_token_only(&batch, &requests));
    }

    #[test]
    fn bf16_hidden_rows_and_duplicate_options_preserve_request_order() {
        let (_fixture, head) = fixture();
        let batch = PackedDecisionBatch::new(&[0, 3]).unwrap();
        let requests = [option(&[5, 0, 5])];
        let prepared = head.prepare(&batch, &requests).unwrap();
        assert_eq!(prepared.selected_rows(), 3);
        let bits = (0..3)
            .flat_map(|row| {
                (0..HIDDEN).map(move |column| bf16::from_f32(hidden_value(row, column)).to_bits())
            })
            .collect::<Vec<_>>();
        let hidden = HiddenRows::from_bf16(&bits, HIDDEN).unwrap();
        let results = head.forward(hidden, &batch, &requests, &prepared).unwrap();
        assert_eq!(
            results[0].logits,
            vec![expected(5, 2), expected(0, 2), expected(5, 2)]
        );
    }

    #[test]
    fn pooled_readout_matches_full_rows_only_for_last_token_requests() {
        let (_fixture, head) = fixture();
        let batch = PackedDecisionBatch::new(&OFFSETS).unwrap();
        let requests = [option(&[4, 1]), option(&[2])];
        let prepared = head.prepare(&batch, &requests).unwrap();
        let full_rows = (0..5)
            .flat_map(|row| (0..HIDDEN).map(move |column| hidden_value(row, column)))
            .collect::<Vec<_>>();
        let full = head
            .forward(
                HiddenRows::from_f32(&full_rows, HIDDEN).unwrap(),
                &batch,
                &requests,
                &prepared,
            )
            .unwrap();
        let pooled_rows = [2usize, 4]
            .into_iter()
            .flat_map(|row| (0..HIDDEN).map(move |column| hidden_value(row, column)))
            .collect::<Vec<_>>();
        let pooled = head
            .forward_pooled(
                HiddenRows::from_f32(&pooled_rows, HIDDEN).unwrap(),
                &batch,
                &requests,
                &prepared,
            )
            .unwrap();
        for (full, pooled) in full.iter().zip(&pooled) {
            assert_eq!(full.logits, pooled.logits);
        }

        let with_warmup = [option(&[4]), DecisionInput::Warmup];
        let prepared = head.prepare(&batch, &with_warmup).unwrap();
        assert!(matches!(
            head.forward_pooled(
                HiddenRows::from_f32(&pooled_rows, HIDDEN).unwrap(),
                &batch,
                &with_warmup,
                &prepared,
            ),
            Err(DecisionHeadError::UnsupportedRequest(_))
        ));
    }

    #[test]
    fn block_requests_validate_the_prefix_and_score_the_last_token() {
        let (_fixture, head) = fixture();
        let batch = PackedDecisionBatch::new(&OFFSETS).unwrap();
        let block = DecisionInput::BlockOptionTokens {
            token_ids: vec![4, 1],
            prefix_length: 2,
        };
        let plain = option(&[4, 1]);
        let prepared_block = head
            .prepare(&batch, &[block.clone(), DecisionInput::Warmup])
            .unwrap();
        let prepared_plain = head
            .prepare(&batch, &[plain, DecisionInput::Warmup])
            .unwrap();
        let rows = (0..5)
            .flat_map(|row| (0..HIDDEN).map(move |column| hidden_value(row, column)))
            .collect::<Vec<_>>();
        let hidden = HiddenRows::from_f32(&rows, HIDDEN).unwrap();
        let requests = [block, DecisionInput::Warmup];
        let block_results = head
            .forward(hidden, &batch, &requests, &prepared_block)
            .unwrap();
        let plain_results = head
            .forward(hidden, &batch, &requests, &prepared_plain)
            .unwrap();
        assert_eq!(block_results[0].logits, plain_results[0].logits);
        assert_eq!(block_results[1].logits, plain_results[1].logits);

        for prefix_length in [0, 4] {
            let request = DecisionInput::BlockOptionTokens {
                token_ids: vec![1],
                prefix_length,
            };
            assert!(matches!(
                head.prepare(&batch, &[request, DecisionInput::Warmup]),
                Err(DecisionHeadError::PrefixOutOfRange { .. })
            ));
        }
    }

    #[test]
    fn rejects_malformed_metadata_and_unsupported_heads() {
        let (_fixture, head) = fixture();
        let batch = PackedDecisionBatch::new(&OFFSETS).unwrap();
        assert!(matches!(
            head.output_count(&option(&[])),
            Err(DecisionHeadError::EmptyOptions)
        ));
        assert!(matches!(
            head.output_count(&option(&[ROWS as u32])),
            Err(DecisionHeadError::TokenIdOutOfRange { .. })
        ));
        assert!(matches!(
            head.output_count(&DecisionInput::InlineOptions {
                markers: vec![1],
                option_counts: vec![2],
            }),
            Err(DecisionHeadError::UnsupportedRequest(_))
        ));
        assert!(matches!(
            head.output_count(&DecisionInput::Laya {
                question_type: 0,
                markers: vec![0],
            }),
            Err(DecisionHeadError::UnsupportedRequest(_))
        ));
        assert!(matches!(
            head.output_count(&DecisionInput::Clef { fields: vec![] }),
            Err(DecisionHeadError::UnsupportedRequest(_))
        ));
        assert!(matches!(
            head.prepare(&batch, &[option(&[1])]),
            Err(DecisionHeadError::InvalidBatch(_))
        ));
        assert!(PackedDecisionBatch::new(&[]).is_err());
        assert!(PackedDecisionBatch::new(&[1, 3, 5]).is_err());
        assert!(PackedDecisionBatch::new(&[0, 3, 3]).is_err());
    }

    #[test]
    fn tied_embeddings_follow_the_native_namespace_fallback() {
        let mut config = config();
        config["tie_word_embeddings"] = json!(true);
        config["text_config"]["tie_word_embeddings"] = json!(true);
        let fixture = Fixture::new(config);
        let embedded = bf16_bytes(
            (0..ROWS)
                .flat_map(|row| (0..HIDDEN).map(move |column| weight_value(row, column) + 100.0)),
        );
        fixture.weights(&[
            (LM_HEAD, Dtype::BF16, vec![ROWS, HIDDEN], lexical_bytes()),
            (
                "model.language_model.embed_tokens.weight",
                Dtype::BF16,
                vec![ROWS, HIDDEN],
                embedded,
            ),
        ]);
        let head = OneJevHead::load(fixture.path()).unwrap();
        assert_eq!(head.namespace(), "model.language_model.embed_tokens.weight");

        let batch = PackedDecisionBatch::new(&[0, 3]).unwrap();
        let requests = [option(&[1])];
        let prepared = head.prepare(&batch, &requests).unwrap();
        let rows = (0..3)
            .flat_map(|row| (0..HIDDEN).map(move |column| hidden_value(row, column)))
            .collect::<Vec<_>>();
        let results = head
            .forward(
                HiddenRows::from_f32(&rows, HIDDEN).unwrap(),
                &batch,
                &requests,
                &prepared,
            )
            .unwrap();
        let expected = (0..HIDDEN)
            .map(|column| {
                bf16::from_f32(weight_value(1, column) + 100.0).to_f32() * hidden_value(2, column)
            })
            .sum::<f32>();
        assert_eq!(results[0].logits, vec![expected]);
    }

    #[test]
    fn transfers_the_validated_bf16_source_for_cold_gpu_upload() {
        let (_fixture, head) = fixture();
        let source = head.into_weight_source();
        assert_eq!(source.len(), ROWS * HIDDEN);
        assert_eq!(source.shape(), &[ROWS, HIDDEN]);
        assert_eq!(source.dtype(), Dtype::BF16);
    }

    #[test]
    fn rejects_wrong_precision_missing_weights_and_width_mismatch() {
        let missing = Fixture::new(config());
        missing.weights(&[(
            "model.language_model.embed_tokens.weight",
            Dtype::BF16,
            vec![ROWS, HIDDEN],
            lexical_bytes(),
        )]);
        assert!(matches!(
            OneJevHead::load(missing.path()),
            Err(DecisionHeadError::MissingWeight(LM_HEAD))
        ));

        let f32_fixture = Fixture::new(config());
        let f32_bytes = (0..ROWS)
            .flat_map(|row| (0..HIDDEN).map(move |column| weight_value(row, column)))
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        f32_fixture.weights(&[(LM_HEAD, Dtype::F32, vec![ROWS, HIDDEN], f32_bytes)]);
        assert!(matches!(
            OneJevHead::load(f32_fixture.path()),
            Err(DecisionHeadError::WrongWeightDtype { .. })
        ));

        let wrong_width = Fixture::new(config());
        wrong_width.weights(&[(
            LM_HEAD,
            Dtype::BF16,
            vec![ROWS, HIDDEN + 1],
            bf16_bytes((0..ROWS * (HIDDEN + 1)).map(|index| index as f32 * 0.25)),
        )]);
        assert!(matches!(
            OneJevHead::load(wrong_width.path()),
            Err(DecisionHeadError::ConfigWidthMismatch { .. })
        ));
    }

    #[test]
    fn rejects_pointer_readout_metadata_but_accepts_causal_onejev() {
        let mut invalid = config();
        invalid["modeling_overrides"]["readout"] = json!("pointer");
        let fixture = Fixture::new(invalid);
        fixture.weights(&[(LM_HEAD, Dtype::BF16, vec![ROWS, HIDDEN], lexical_bytes())]);
        assert!(matches!(
            OneJevHead::load(fixture.path()),
            Err(DecisionHeadError::InvalidConfig(_))
        ));

        let mut causal = config();
        causal["modeling_overrides"]["attention"]["full_attention"] = json!("causal");
        let fixture = Fixture::new(causal);
        fixture.weights(&[(LM_HEAD, Dtype::BF16, vec![ROWS, HIDDEN], lexical_bytes())]);
        let head = OneJevHead::load(fixture.path()).unwrap();
        assert!(!head.block_attention());
    }

    #[test]
    fn rejects_prepared_rows_from_another_head() {
        let (_first_fixture, first) = fixture();
        let (_second_fixture, second) = fixture();
        let batch = PackedDecisionBatch::new(&[0, 3]).unwrap();
        let requests = [option(&[4, 1])];
        let prepared = first.prepare(&batch, &requests).unwrap();
        let rows = (0..3)
            .flat_map(|row| (0..HIDDEN).map(move |column| hidden_value(row, column)))
            .collect::<Vec<_>>();
        assert!(matches!(
            second.forward(
                HiddenRows::from_f32(&rows, HIDDEN).unwrap(),
                &batch,
                &requests,
                &prepared,
            ),
            Err(DecisionHeadError::PreparedMismatch(_))
        ));
    }

    #[test]
    fn rejects_prepared_rows_from_another_batch() {
        let (_fixture, head) = fixture();
        let original = PackedDecisionBatch::new(&[0, 3]).unwrap();
        let requests = [option(&[4, 1])];
        let prepared = head.prepare(&original, &requests).unwrap();
        let rows = (0..3)
            .flat_map(|row| (0..HIDDEN).map(move |column| hidden_value(row, column)))
            .collect::<Vec<_>>();
        let hidden = HiddenRows::from_f32(&rows, HIDDEN).unwrap();

        let other_batch = PackedDecisionBatch::new(&[0, 2]).unwrap();
        let other_rows = (0..2)
            .flat_map(|row| (0..HIDDEN).map(move |column| hidden_value(row, column)))
            .collect::<Vec<_>>();
        assert!(matches!(
            head.forward(
                HiddenRows::from_f32(&other_rows, HIDDEN).unwrap(),
                &other_batch,
                &[option(&[4, 1])],
                &prepared,
            ),
            Err(DecisionHeadError::PreparedMismatch(_))
        ));
        assert!(matches!(
            head.forward(hidden, &original, &[option(&[1, 4])], &prepared),
            Err(DecisionHeadError::PreparedMismatch(_))
        ));
    }
}
