// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Cold D3 code-readout and Perplexity joint-readout metadata/plans.
//!
//! Native `OptionHead` in `decision_models.cpp` resolves two external 255-row
//! readouts:
//!
//! * D3 (`d3-code-readout-v1`) uses an FP32 `readout.safetensors` matrix. Each
//!   request projects the last real token against the full 255-row matrix and
//!   only then selects the requested leading code logits.
//! * Perplexity uses a BF16 matrix. Ordinary option-token requests project only
//!   the selected rows, while `InlineOptions` projects every joint marker
//!   against the full matrix and slices each marker's declared option count.
//!
//! This leaf module owns cold sidecar validation and request planning. It does
//! not create CUDA state and does not own the backbone.

use crate::{
    decision_head::{DecisionHeadError, HiddenRows, PackedDecisionBatch},
    model::{path_present, read_json, Checkpoint, WeightSource},
};
use half::bf16;
use infinity_v3_backend_core::{DecisionInput, DecisionOutput};
use safetensors::Dtype;
use serde::Deserialize;
use std::{
    fmt,
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
};

const DECISION_CONFIG: &str = "decision_config.json";
const JOINT_CONFIG: &str = "joint_config.json";
const READOUT_FILE: &str = "readout.safetensors";
const READOUT_TENSOR: &str = "weight";
const D3_FORMAT: &str = "d3-code-readout-v1";
const D3_PROMPT: &str = "d3";
const PPLX_JOINT_TEMPLATE: &str = "joint-question-inline-readout-v1";
const PPLX_JOINT_MARKER: &str = "<|fim_suffix|>";
const EXTERNAL_ROWS: usize = 255;
static NEXT_EXTERNAL_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ExternalDecisionError {
    InvalidMetadata(String),
    UnsupportedProtocol(String),
    InvalidJoint(String),
    MissingWeight(&'static str),
    InvalidWeightGeometry {
        shape: Vec<usize>,
    },
    WrongWeightDtype {
        expected: &'static str,
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
    OptionCountOutOfRange {
        request: usize,
        option: usize,
        count: usize,
        rows: usize,
    },
    MarkerOutOfRange {
        request: usize,
        marker: usize,
        length: usize,
    },
    OutputCountOverflow,
    Allocation,
    WeightRead(String),
    PreparedMismatch(&'static str),
    HiddenShape(&'static str),
}

impl fmt::Display for ExternalDecisionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMetadata(error) => {
                write!(f, "invalid external decision metadata: {error}")
            }
            Self::UnsupportedProtocol(error) => {
                write!(f, "unsupported external decision protocol: {error}")
            }
            Self::InvalidJoint(error) => write!(f, "invalid joint readout config: {error}"),
            Self::MissingWeight(name) => write!(f, "missing external readout weight {name}"),
            Self::InvalidWeightGeometry { shape } => {
                write!(f, "invalid external readout geometry: {shape:?}")
            }
            Self::WrongWeightDtype { expected, found } => {
                write!(f, "external readout must be {expected}, found {found}")
            }
            Self::ConfigWidthMismatch {
                configured,
                weights,
            } => write!(
                f,
                "external readout hidden width mismatch: config={configured}, weights={weights}"
            ),
            Self::InvalidBatch(reason) => write!(f, "invalid packed decision batch: {reason}"),
            Self::UnsupportedRequest(reason) => {
                write!(f, "unsupported external decision request: {reason}")
            }
            Self::EmptyOptions => f.write_str("external option metadata must not be empty"),
            Self::TokenIdOutOfRange {
                request,
                token_id,
                rows,
            } => write!(
                f,
                "external request {request} option row {token_id} is outside 0..{rows}"
            ),
            Self::OptionCountOutOfRange {
                request,
                option,
                count,
                rows,
            } => write!(
                f,
                "external request {request} marker {option} count {count} is outside 1..={rows}"
            ),
            Self::MarkerOutOfRange {
                request,
                marker,
                length,
            } => write!(
                f,
                "external request {request} marker {marker} is outside sequence length {length}"
            ),
            Self::OutputCountOverflow => f.write_str("external decision output count overflow"),
            Self::Allocation => f.write_str("external decision allocation failed"),
            Self::WeightRead(error) => write!(f, "external readout weight read failed: {error}"),
            Self::PreparedMismatch(reason) => {
                write!(f, "prepared external readout mismatch: {reason}")
            }
            Self::HiddenShape(reason) => write!(f, "invalid external hidden states: {reason}"),
        }
    }
}

impl std::error::Error for ExternalDecisionError {}

impl From<DecisionHeadError> for ExternalDecisionError {
    fn from(error: DecisionHeadError) -> Self {
        Self::InvalidMetadata(error.to_string())
    }
}

/// The two external 255-row contracts. They intentionally differ in dtype and
/// projection/selection order; neither is a OneJev lexical head.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ExternalFamily {
    D3F32,
    PplxBf16,
}

impl ExternalFamily {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::D3F32 => "D3 FP32 code readout",
            Self::PplxBf16 => "Perplexity BF16 readout",
        }
    }

    pub(crate) const fn weight_dtype(self) -> Dtype {
        match self {
            Self::D3F32 => Dtype::F32,
            Self::PplxBf16 => Dtype::BF16,
        }
    }

    pub(crate) const fn dtype_label(self) -> &'static str {
        match self {
            Self::D3F32 => "FP32",
            Self::PplxBf16 => "BF16",
        }
    }

    pub(crate) const fn supports_inline_options(self) -> bool {
        matches!(self, Self::PplxBf16)
    }
}

/// The explicit decision attention mode from the checkpoint sidecar.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ExternalAttention {
    Causal,
    NoncausalFull,
    PrefixBidirectional,
}

impl ExternalAttention {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Causal => "causal",
            Self::NoncausalFull => "noncausal_full_attention",
            Self::PrefixBidirectional => "prefix_bidir",
        }
    }

    pub(crate) const fn block_attention(self) -> bool {
        matches!(self, Self::PrefixBidirectional)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TemperatureTable {
    max_questions: Vec<usize>,
    temperature: Vec<f32>,
}

impl TemperatureTable {
    fn validate(&self) -> bool {
        self.temperature.len() == self.max_questions.len() + 1
            && self.max_questions.iter().all(|&n| n > 0)
            && self.max_questions.windows(2).all(|w| w[0] < w[1])
            && self.temperature.iter().all(|t| t.is_finite() && *t > 0.)
    }

    pub(crate) fn for_questions(&self, count: usize) -> f32 {
        let index = self.max_questions.partition_point(|&bound| count > bound);
        self.temperature.get(index).copied().unwrap_or(1.)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct JointReadout {
    pub(crate) marker: String,
    pub(crate) readout_offset: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ExternalConfig {
    pub(crate) family: ExternalFamily,
    pub(crate) hidden: usize,
    pub(crate) rows: usize,
    pub(crate) attention: ExternalAttention,
    pub(crate) prompt_layout: Option<String>,
    pub(crate) temperature: f32,
    pub(crate) temperature_by_questions: Option<TemperatureTable>,
    pub(crate) joint: Option<JointReadout>,
    pub(crate) codes: Vec<String>,
    pub(crate) token_ids: Vec<u32>,
}

impl ExternalConfig {
    /// D3 applies `temperature_by_questions` first, then its constant.
    pub(crate) fn temperature_for(&self, question_count: usize) -> f32 {
        self.temperature_by_questions
            .as_ref()
            .map_or(self.temperature, |table| {
                table.for_questions(question_count)
            })
    }
}

#[derive(Clone, Debug, Deserialize, Default)]
struct RawTemperatureTable {
    max_questions: Vec<usize>,
    temperature: Vec<f32>,
}

#[derive(Debug, Deserialize, Default)]
struct RawDecisionConfig {
    format_version: Option<usize>,
    format_id: Option<String>,
    prompt: Option<String>,
    codes: Option<Vec<String>>,
    token_ids: Option<Vec<u32>>,
    temperature: Option<f32>,
    attention_mode: Option<String>,
    pooling: Option<String>,
    readout_dtype: Option<String>,
    max_length: Option<usize>,
    prompt_layout: Option<String>,
    temperature_by_questions: Option<RawTemperatureTable>,
}

#[derive(Debug, Deserialize, Default)]
struct RawJointConfig {
    version: Option<usize>,
    template: Option<String>,
    marker: Option<String>,
    readout_offset: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct RawTextConfig {
    model_type: Option<String>,
    hidden_size: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct RawBackboneConfig {
    model_type: Option<String>,
    hidden_size: Option<usize>,
    text_config: Option<RawTextConfig>,
}

fn metadata(error: impl fmt::Display) -> ExternalDecisionError {
    ExternalDecisionError::InvalidMetadata(error.to_string())
}

fn validate_codes(
    codes: Option<Vec<String>>,
    token_ids: Option<Vec<u32>>,
) -> Result<(Vec<String>, Vec<u32>), ExternalDecisionError> {
    match (codes, token_ids) {
        (None, None) => Ok((Vec::new(), Vec::new())),
        (Some(codes), Some(ids)) => {
            if codes.len() != EXTERNAL_ROWS || ids.len() != EXTERNAL_ROWS {
                return Err(metadata(format!(
                    "answer vocabulary must contain {EXTERNAL_ROWS} codes and token ids"
                )));
            }
            let mut unique = ids.clone();
            unique.sort_unstable();
            unique.dedup();
            if unique.len() != EXTERNAL_ROWS {
                return Err(metadata("answer token ids must be unique"));
            }
            Ok((codes, ids))
        }
        _ => Err(metadata(
            "codes and token_ids must either both be present or both be absent",
        )),
    }
}

fn attention(
    value: Option<&str>,
    family: ExternalFamily,
) -> Result<ExternalAttention, ExternalDecisionError> {
    match (family, value) {
        (ExternalFamily::D3F32, Some("noncausal_full_attention")) => {
            Ok(ExternalAttention::NoncausalFull)
        }
        (ExternalFamily::D3F32, Some("prefix_bidir")) => Ok(ExternalAttention::PrefixBidirectional),
        (ExternalFamily::D3F32, Some(other)) => Err(metadata(format!(
            "D3 attention_mode {other:?} is not noncausal_full_attention or prefix_bidir"
        ))),
        (ExternalFamily::D3F32, None) => Err(metadata("D3 requires an explicit attention_mode")),
        (ExternalFamily::PplxBf16, Some("noncausal_full_attention")) => {
            Ok(ExternalAttention::NoncausalFull)
        }
        (ExternalFamily::PplxBf16, Some("causal")) => Ok(ExternalAttention::Causal),
        (ExternalFamily::PplxBf16, Some(other)) => Err(metadata(format!(
            "Pplx attention_mode {other:?} is not causal or noncausal_full_attention"
        ))),
        (ExternalFamily::PplxBf16, None) => Ok(ExternalAttention::Causal),
    }
}

fn load_joint(
    path: &Path,
    config: &RawDecisionConfig,
) -> Result<Option<JointReadout>, ExternalDecisionError> {
    let joint_path = path.join(JOINT_CONFIG);
    if !path_present(&joint_path).map_err(metadata)? {
        return Ok(None);
    }
    let joint: RawJointConfig = read_json(&joint_path)
        .map_err(|error| ExternalDecisionError::InvalidJoint(error.to_string()))?;
    if joint.version != Some(3)
        || joint.template.as_deref() != Some(PPLX_JOINT_TEMPLATE)
        || joint.marker.as_deref() != Some(PPLX_JOINT_MARKER)
        || joint.readout_offset != Some(0)
    {
        return Err(ExternalDecisionError::InvalidJoint(
            "expected version 3, joint-question-inline-readout-v1, <|fim_suffix|> and offset 0"
                .into(),
        ));
    }
    if config.format_version != Some(2) {
        return Err(ExternalDecisionError::InvalidJoint(
            "joint readouts require decision_config format_version 2".into(),
        ));
    }
    if config.attention_mode.as_deref() != Some("noncausal_full_attention")
        || config.pooling.as_deref() != Some("last")
    {
        return Err(ExternalDecisionError::InvalidJoint(
            "joint readouts require noncausal_full_attention and last pooling".into(),
        ));
    }
    Ok(Some(JointReadout {
        marker: PPLX_JOINT_MARKER.into(),
        readout_offset: 0,
    }))
}

fn load_backbone(path: &Path) -> Result<(String, usize), ExternalDecisionError> {
    let raw: RawBackboneConfig = read_json(&path.join("config.json")).map_err(metadata)?;
    let model_type = raw
        .text_config
        .as_ref()
        .and_then(|text| text.model_type.clone())
        .or(raw.model_type)
        .ok_or_else(|| metadata("missing model_type"))?;
    let hidden = raw
        .text_config
        .as_ref()
        .and_then(|text| text.hidden_size)
        .or(raw.hidden_size)
        .filter(|&hidden| hidden > 0)
        .ok_or_else(|| metadata("missing positive hidden_size"))?;
    Ok((model_type, hidden))
}

/// Read and validate the complete external sidecar contract without touching
/// the readout tensor payload.
pub(crate) fn load_external_config(path: &Path) -> Result<ExternalConfig, ExternalDecisionError> {
    let raw: RawDecisionConfig = read_json(&path.join(DECISION_CONFIG)).map_err(metadata)?;
    let family = if raw.format_id.as_deref() == Some(D3_FORMAT) {
        ExternalFamily::D3F32
    } else {
        ExternalFamily::PplxBf16
    };
    let (model_type, hidden) = load_backbone(path)?;
    if family == ExternalFamily::D3F32 && !matches!(model_type.as_str(), "qwen3_5" | "qwen3_5_text")
    {
        return Err(ExternalDecisionError::UnsupportedProtocol(format!(
            "D3 requires the dense Qwen3.5 backbone, found {model_type}"
        )));
    }
    if !model_type.starts_with("qwen3_5") {
        return Err(ExternalDecisionError::UnsupportedProtocol(format!(
            "external decision heads require Qwen3.5, found {model_type}"
        )));
    }
    let (codes, token_ids) = validate_codes(raw.codes.clone(), raw.token_ids.clone())?;
    let attention = attention(raw.attention_mode.as_deref(), family)?;
    let prompt_layout = raw.prompt_layout.clone();
    if prompt_layout
        .as_deref()
        .is_some_and(|layout| layout != "header-v1")
    {
        return Err(metadata(format!(
            "unsupported D3 prompt layout {:?}",
            prompt_layout.as_deref().unwrap_or_default()
        )));
    }
    let temperature = raw.temperature.unwrap_or(1.);
    if !temperature.is_finite() || temperature <= 0. {
        return Err(metadata("decision temperature must be finite and positive"));
    }
    let temperature_by_questions =
        raw.temperature_by_questions
            .clone()
            .map(|table| TemperatureTable {
                max_questions: table.max_questions,
                temperature: table.temperature,
            });
    if temperature_by_questions
        .as_ref()
        .is_some_and(|table| !table.validate())
    {
        return Err(metadata("invalid question-count temperature table"));
    }
    match family {
        ExternalFamily::D3F32 => {
            if raw.format_version != Some(1)
                || raw.prompt.as_deref() != Some(D3_PROMPT)
                || raw.pooling.as_deref() != Some("last")
                || raw.readout_dtype.as_deref() != Some("float32")
            {
                return Err(metadata(
                    "D3 requires format_version 1, prompt=d3, pooling=last and readout_dtype=float32",
                ));
            }
        }
        ExternalFamily::PplxBf16 => {
            if !matches!(raw.format_version, Some(1 | 2)) {
                return Err(metadata(
                    "Pplx requires decision_config format_version 1 or 2",
                ));
            }
            if raw
                .pooling
                .as_deref()
                .is_some_and(|pooling| pooling != "last")
            {
                return Err(metadata("Pplx requires last-token pooling"));
            }
        }
    }
    let joint = load_joint(path, &raw)?;
    Ok(ExternalConfig {
        family,
        hidden,
        rows: EXTERNAL_ROWS,
        attention,
        prompt_layout,
        temperature,
        temperature_by_questions,
        joint,
        codes,
        token_ids,
    })
}

/// One per-request prepared readout shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ExternalRequestPlan {
    /// Read one real token row. `full` selects the native D3 order: project all
    /// 255 rows, then keep the leading `count` code columns.
    LastToken {
        row: u32,
        count: usize,
        full: bool,
        full_row: Option<usize>,
    },
    /// Perplexity joint markers. Every marker row projects against all 255 rows;
    /// each marker then keeps its declared leading option count.
    Inline {
        markers: Vec<u32>,
        counts: Vec<usize>,
        full_row_start: usize,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct ExternalPlan {
    pub(crate) counts: Vec<usize>,
    pub(crate) row_offsets: Vec<usize>,
    pub(crate) boundaries: Vec<u32>,
    pub(crate) requests: Vec<ExternalRequestPlan>,
    pub(crate) marker_rows: Vec<u32>,
    pub(crate) full_rows: usize,
    pub(crate) total: usize,
}

pub(crate) fn request_output_count(
    family: ExternalFamily,
    rows: usize,
    request: usize,
    input: &DecisionInput,
) -> Result<usize, ExternalDecisionError> {
    match input {
        DecisionInput::Warmup => Ok(1),
        DecisionInput::OptionTokens { token_ids }
        | DecisionInput::BlockOptionTokens { token_ids, .. } => {
            if token_ids.is_empty() {
                return Err(ExternalDecisionError::EmptyOptions);
            }
            for (index, &token_id) in token_ids.iter().enumerate() {
                if usize::try_from(token_id).map_or(true, |id| id >= rows) {
                    return Err(ExternalDecisionError::TokenIdOutOfRange {
                        request,
                        token_id,
                        rows,
                    });
                }
                if (family == ExternalFamily::D3F32 || family == ExternalFamily::PplxBf16)
                    && token_id != u32::try_from(index).unwrap_or(u32::MAX)
                {
                    return Err(ExternalDecisionError::UnsupportedRequest(
                        "external readouts require sequential option rows starting at zero",
                    ));
                }
            }
            Ok(token_ids.len())
        }
        DecisionInput::InlineOptions {
            markers,
            option_counts,
        } => {
            if !family.supports_inline_options() {
                return Err(ExternalDecisionError::UnsupportedRequest(
                    "D3 does not accept joint inline markers",
                ));
            }
            if markers.is_empty() || markers.len() != option_counts.len() {
                return Err(ExternalDecisionError::UnsupportedRequest(
                    "inline readouts require one option count per marker",
                ));
            }
            let mut total = 0usize;
            for (index, (&marker, &count)) in markers.iter().zip(option_counts).enumerate() {
                if index > 0 && marker <= markers[index - 1] {
                    return Err(ExternalDecisionError::UnsupportedRequest(
                        "inline markers must be strictly increasing",
                    ));
                }
                if count == 0 || count > rows {
                    return Err(ExternalDecisionError::OptionCountOutOfRange {
                        request,
                        option: index,
                        count,
                        rows,
                    });
                }
                total = total
                    .checked_add(count)
                    .ok_or(ExternalDecisionError::OutputCountOverflow)?;
            }
            Ok(total)
        }
        DecisionInput::Laya { .. } | DecisionInput::Clef { .. } => {
            Err(ExternalDecisionError::UnsupportedRequest(
                "external 255-row readouts accept only option-token or inline-marker metadata",
            ))
        }
    }
}

/// Build the immutable request plan for one packed external decision batch.
pub(crate) fn plan_external_requests(
    family: ExternalFamily,
    rows: usize,
    batch: &PackedDecisionBatch<'_>,
    requests: &[DecisionInput],
) -> Result<ExternalPlan, ExternalDecisionError> {
    if requests.len() != batch.len() {
        return Err(ExternalDecisionError::InvalidBatch(
            "request count does not match packed sequences",
        ));
    }
    let mut counts = Vec::new();
    counts
        .try_reserve_exact(requests.len())
        .map_err(|_| ExternalDecisionError::Allocation)?;
    let mut row_offsets = Vec::new();
    row_offsets
        .try_reserve_exact(requests.len() + 1)
        .map_err(|_| ExternalDecisionError::Allocation)?;
    row_offsets.push(0);
    let mut total = 0usize;
    for (index, request) in requests.iter().enumerate() {
        let count = request_output_count(family, rows, index, request)?;
        total = total
            .checked_add(count)
            .ok_or(ExternalDecisionError::OutputCountOverflow)?;
        counts.push(count);
        row_offsets.push(total);
    }
    let mut per_request = Vec::new();
    per_request
        .try_reserve_exact(requests.len())
        .map_err(|_| ExternalDecisionError::Allocation)?;
    let mut marker_rows = Vec::new();
    let mut full_rows = 0usize;
    for (index, request) in requests.iter().enumerate() {
        let (_, end) = batch.sequence(index)?;
        let length = end
            .checked_sub(
                usize::try_from(batch.boundaries()[index])
                    .map_err(|_| ExternalDecisionError::InvalidBatch("boundary exceeds usize"))?,
            )
            .ok_or(ExternalDecisionError::InvalidBatch(
                "sequence length overflow",
            ))?;
        let last = end
            .checked_sub(1)
            .ok_or(ExternalDecisionError::InvalidBatch(
                "external decisions require nonempty sequences",
            ))?;
        let last = u32::try_from(last)
            .map_err(|_| ExternalDecisionError::InvalidBatch("hidden row exceeds u32"))?;
        match request {
            DecisionInput::Warmup => {
                let full_row = (family == ExternalFamily::D3F32).then_some(full_rows);
                if full_row.is_some() {
                    full_rows += 1;
                }
                per_request.push(ExternalRequestPlan::LastToken {
                    row: last,
                    count: 1,
                    full: family == ExternalFamily::D3F32,
                    full_row,
                });
            }
            DecisionInput::OptionTokens { token_ids }
            | DecisionInput::BlockOptionTokens { token_ids, .. } => {
                let full_row = (family == ExternalFamily::D3F32).then_some(full_rows);
                if full_row.is_some() {
                    full_rows += 1;
                }
                per_request.push(ExternalRequestPlan::LastToken {
                    row: last,
                    count: token_ids.len(),
                    full: family == ExternalFamily::D3F32,
                    full_row,
                });
            }
            DecisionInput::InlineOptions {
                markers,
                option_counts,
            } => {
                let start = usize::try_from(batch.boundaries()[index])
                    .map_err(|_| ExternalDecisionError::InvalidBatch("boundary exceeds usize"))?;
                let mut packed = Vec::new();
                packed
                    .try_reserve_exact(markers.len())
                    .map_err(|_| ExternalDecisionError::Allocation)?;
                for &marker in markers {
                    if marker >= length {
                        return Err(ExternalDecisionError::MarkerOutOfRange {
                            request: index,
                            marker,
                            length,
                        });
                    }
                    let row = start
                        .checked_add(marker)
                        .ok_or(ExternalDecisionError::OutputCountOverflow)?;
                    let row = u32::try_from(row).map_err(|_| {
                        ExternalDecisionError::InvalidBatch("hidden row exceeds u32")
                    })?;
                    marker_rows.push(row);
                    packed.push(row);
                }
                let full_row_start = full_rows;
                full_rows = full_rows
                    .checked_add(markers.len())
                    .ok_or(ExternalDecisionError::OutputCountOverflow)?;
                per_request.push(ExternalRequestPlan::Inline {
                    markers: packed,
                    counts: option_counts.clone(),
                    full_row_start,
                });
            }
            DecisionInput::Laya { .. } | DecisionInput::Clef { .. } => {
                return Err(ExternalDecisionError::UnsupportedRequest(
                    "external 255-row readouts accept only option-token or inline-marker metadata",
                ));
            }
        }
    }
    Ok(ExternalPlan {
        counts,
        row_offsets,
        boundaries: batch.boundaries().to_vec(),
        requests: per_request,
        marker_rows,
        full_rows,
        total,
    })
}

/// Native `OptionHead::last_token_only` for the two external heads.
pub(crate) fn external_last_token_only(
    family: ExternalFamily,
    batch: &PackedDecisionBatch<'_>,
    requests: &[DecisionInput],
) -> bool {
    if requests.len() != batch.len() {
        return false;
    }
    requests
        .iter()
        .enumerate()
        .all(|(index, request)| match request {
            DecisionInput::OptionTokens { .. } | DecisionInput::BlockOptionTokens { .. } => true,
            DecisionInput::InlineOptions { markers, .. } => {
                if family != ExternalFamily::PplxBf16 || markers.len() != 1 {
                    return false;
                }
                let Ok((start, end)) = batch.sequence(index) else {
                    return false;
                };
                end.checked_sub(1).and_then(|last| last.checked_sub(start)) == Some(markers[0])
            }
            _ => false,
        })
}

/// Request-owned cold CPU readout plan.
pub(crate) struct PreparedExternal {
    head_id: u64,
    hidden: usize,
    rows: usize,
    plan: ExternalPlan,
}

impl PreparedExternal {
    pub(crate) fn output_counts(&self) -> &[usize] {
        &self.plan.counts
    }
}

/// Cold external readout with a validated sidecar and borrowed weight source.
pub(crate) struct ExternalHead {
    id: u64,
    config: ExternalConfig,
    weight: WeightSource,
}

impl ExternalHead {
    pub(crate) fn load(path: &Path) -> Result<Self, ExternalDecisionError> {
        let config = load_external_config(path)?;
        let checkpoint = Checkpoint::open_file(&path.join(READOUT_FILE))
            .map_err(|error| ExternalDecisionError::WeightRead(error.to_string()))?;
        let shape = checkpoint
            .shape(READOUT_TENSOR)
            .ok_or(ExternalDecisionError::MissingWeight(READOUT_TENSOR))?
            .to_vec();
        if shape.len() != 2 || shape[0] != EXTERNAL_ROWS || shape[1] != config.hidden {
            return Err(ExternalDecisionError::InvalidWeightGeometry { shape });
        }
        let weight = checkpoint
            .source(READOUT_TENSOR, &shape)
            .map_err(|error| ExternalDecisionError::WeightRead(error.to_string()))?;
        if weight.dtype() != config.family.weight_dtype() {
            return Err(ExternalDecisionError::WrongWeightDtype {
                expected: config.family.dtype_label(),
                found: format!("{:?}", weight.dtype()),
            });
        }
        let elements = EXTERNAL_ROWS
            .checked_mul(config.hidden)
            .ok_or(ExternalDecisionError::OutputCountOverflow)?;
        if weight.len() != elements {
            return Err(ExternalDecisionError::InvalidWeightGeometry { shape });
        }
        Ok(Self {
            id: NEXT_EXTERNAL_ID.fetch_add(1, Ordering::Relaxed),
            config,
            weight,
        })
    }

    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    pub(crate) fn config(&self) -> &ExternalConfig {
        &self.config
    }

    pub(crate) fn family(&self) -> ExternalFamily {
        self.config.family
    }

    pub(crate) fn hidden_size(&self) -> usize {
        self.config.hidden
    }

    pub(crate) fn rows(&self) -> usize {
        self.config.rows
    }

    pub(crate) fn into_weight_source(self) -> WeightSource {
        self.weight
    }

    pub(crate) fn output_count(
        &self,
        request: usize,
        input: &DecisionInput,
    ) -> Result<usize, ExternalDecisionError> {
        request_output_count(self.config.family, self.config.rows, request, input)
    }

    pub(crate) fn last_token_only(
        &self,
        batch: &PackedDecisionBatch<'_>,
        requests: &[DecisionInput],
    ) -> bool {
        external_last_token_only(self.config.family, batch, requests)
    }

    pub(crate) fn prepare(
        &self,
        batch: &PackedDecisionBatch<'_>,
        requests: &[DecisionInput],
    ) -> Result<PreparedExternal, ExternalDecisionError> {
        let plan = plan_external_requests(self.config.family, self.config.rows, batch, requests)?;
        Ok(PreparedExternal {
            head_id: self.id,
            hidden: self.config.hidden,
            rows: self.config.rows,
            plan,
        })
    }

    fn read_weight_row(
        &self,
        row: usize,
        scratch: &mut Vec<f32>,
    ) -> Result<(), ExternalDecisionError> {
        if scratch.len() != self.config.hidden {
            scratch.clear();
            scratch
                .try_reserve_exact(self.config.hidden)
                .map_err(|_| ExternalDecisionError::Allocation)?;
            scratch.resize(self.config.hidden, 0.);
        }
        let offset = row
            .checked_mul(self.config.hidden)
            .ok_or(ExternalDecisionError::OutputCountOverflow)?;
        match self.config.family {
            ExternalFamily::D3F32 => self
                .weight
                .read_f32(offset, scratch)
                .map_err(|error| ExternalDecisionError::WeightRead(error.to_string())),
            ExternalFamily::PplxBf16 => {
                let mut bits = Vec::new();
                bits.try_reserve_exact(self.config.hidden)
                    .map_err(|_| ExternalDecisionError::Allocation)?;
                bits.resize(self.config.hidden, 0u16);
                self.weight
                    .read_bf16(offset, &mut bits)
                    .map_err(|error| ExternalDecisionError::WeightRead(error.to_string()))?;
                for (value, bits) in scratch.iter_mut().zip(bits) {
                    *value = bf16::from_bits(bits).to_f32();
                }
                Ok(())
            }
        }
    }

    /// Full 255-row projection for one hidden row, in request order.
    fn project_full(
        &self,
        hidden: HiddenRows<'_>,
        hidden_row: usize,
        scratch: &mut Vec<f32>,
    ) -> Result<Vec<f32>, ExternalDecisionError> {
        let mut values = Vec::new();
        values
            .try_reserve_exact(self.config.rows)
            .map_err(|_| ExternalDecisionError::Allocation)?;
        for row in 0..self.config.rows {
            self.read_weight_row(row, scratch)?;
            values.push(hidden.dot(hidden_row, scratch)?);
        }
        Ok(values)
    }

    pub(crate) fn forward(
        &self,
        hidden: HiddenRows<'_>,
        batch: &PackedDecisionBatch<'_>,
        requests: &[DecisionInput],
        prepared: &PreparedExternal,
    ) -> Result<Vec<DecisionOutput>, ExternalDecisionError> {
        self.readout(hidden, batch, requests, prepared, false)
    }

    pub(crate) fn forward_pooled(
        &self,
        hidden: HiddenRows<'_>,
        batch: &PackedDecisionBatch<'_>,
        requests: &[DecisionInput],
        prepared: &PreparedExternal,
    ) -> Result<Vec<DecisionOutput>, ExternalDecisionError> {
        if !external_last_token_only(self.config.family, batch, requests) {
            return Err(ExternalDecisionError::UnsupportedRequest(
                "pooled external decisions require last-token-only requests",
            ));
        }
        self.readout(hidden, batch, requests, prepared, true)
    }

    fn readout(
        &self,
        hidden: HiddenRows<'_>,
        batch: &PackedDecisionBatch<'_>,
        requests: &[DecisionInput],
        prepared: &PreparedExternal,
        pooled: bool,
    ) -> Result<Vec<DecisionOutput>, ExternalDecisionError> {
        if requests.len() != batch.len() {
            return Err(ExternalDecisionError::InvalidBatch(
                "request count does not match packed sequences",
            ));
        }
        if hidden.width() != self.config.hidden {
            return Err(ExternalDecisionError::HiddenShape(
                "hidden width does not match the external readout",
            ));
        }
        if pooled {
            if hidden.rows() != batch.len() {
                return Err(ExternalDecisionError::HiddenShape(
                    "pooled hidden row count does not match requests",
                ));
            }
        } else if hidden.rows() != batch.tokens() {
            return Err(ExternalDecisionError::HiddenShape(
                "packed hidden row count does not match token boundaries",
            ));
        }
        if prepared.head_id != self.id
            || prepared.hidden != self.config.hidden
            || prepared.rows != self.config.rows
            || prepared.plan.boundaries.as_slice() != batch.boundaries()
            || prepared.plan.counts.len() != requests.len()
        {
            return Err(ExternalDecisionError::PreparedMismatch(
                "prepared metadata belongs to another head or batch",
            ));
        }
        let mut scratch = Vec::new();
        let mut results = Vec::new();
        results
            .try_reserve_exact(requests.len())
            .map_err(|_| ExternalDecisionError::Allocation)?;
        for (index, request) in requests.iter().enumerate() {
            let expected =
                request_output_count(self.config.family, self.config.rows, index, request)?;
            if prepared.plan.counts.get(index) != Some(&expected) {
                return Err(ExternalDecisionError::PreparedMismatch(
                    "prepared output count does not match the request",
                ));
            }
            let planned = prepared.plan.requests.get(index).ok_or(
                ExternalDecisionError::PreparedMismatch("prepared request plan is missing"),
            )?;
            let mut logits = Vec::new();
            logits
                .try_reserve_exact(expected)
                .map_err(|_| ExternalDecisionError::Allocation)?;
            match planned {
                ExternalRequestPlan::LastToken {
                    row, count, full, ..
                } => {
                    let source_row = if pooled {
                        index
                    } else {
                        usize::try_from(*row).map_err(|_| {
                            ExternalDecisionError::HiddenShape("hidden row exceeds usize")
                        })?
                    };
                    if *full {
                        let full_values = self.project_full(hidden, source_row, &mut scratch)?;
                        logits.extend_from_slice(&full_values[..*count]);
                    } else {
                        for weight_row in 0..*count {
                            self.read_weight_row(weight_row, &mut scratch)?;
                            logits.push(hidden.dot(source_row, &scratch)?);
                        }
                    }
                }
                ExternalRequestPlan::Inline {
                    markers, counts, ..
                } => {
                    if pooled {
                        let full_values = self.project_full(hidden, index, &mut scratch)?;
                        logits.extend_from_slice(&full_values[..counts[0]]);
                    } else {
                        for (&marker, &count) in markers.iter().zip(counts) {
                            let row = usize::try_from(marker).map_err(|_| {
                                ExternalDecisionError::HiddenShape("marker row exceeds usize")
                            })?;
                            let full_values = self.project_full(hidden, row, &mut scratch)?;
                            logits.extend_from_slice(&full_values[..count]);
                        }
                    }
                }
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
    use std::{fs, path::PathBuf};

    struct Fixture(PathBuf);

    impl Fixture {
        fn new(config: Value) -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "infinity-external-head-{}-{}",
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

        fn decision_config(&self, value: Value) {
            fs::write(
                self.0.join(DECISION_CONFIG),
                serde_json::to_vec(&value).unwrap(),
            )
            .unwrap();
        }

        fn joint_config(&self, value: Value) {
            fs::write(
                self.0.join(JOINT_CONFIG),
                serde_json::to_vec(&value).unwrap(),
            )
            .unwrap();
        }

        fn readout(&self, dtype: Dtype, shape: Vec<usize>, bytes: Vec<u8>) {
            let view = TensorView::new(dtype, shape, &bytes).unwrap();
            serialize_to_file([(READOUT_TENSOR, view)], None, &self.0.join(READOUT_FILE)).unwrap();
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    const HIDDEN: usize = 4;

    fn backbone() -> Value {
        json!({
            "model_type": "qwen3_5",
            "text_config": {"model_type": "qwen3_5_text", "hidden_size": HIDDEN}
        })
    }

    fn codes() -> Value {
        let values = (0..EXTERNAL_ROWS)
            .map(|index| Value::String(format!("C{index}")))
            .collect::<Vec<_>>();
        json!(values)
    }

    fn token_ids() -> Value {
        json!((0..EXTERNAL_ROWS as u32).collect::<Vec<_>>())
    }

    fn d3_config() -> Value {
        json!({
            "format_version": 1,
            "format_id": D3_FORMAT,
            "prompt": "d3",
            "pooling": "last",
            "attention_mode": "noncausal_full_attention",
            "readout_dtype": "float32",
            "temperature": 0.5,
            "temperature_by_questions": {
                "max_questions": [1, 4],
                "temperature": [1.25, 1.0, 0.75]
            },
            "codes": codes(),
            "token_ids": token_ids()
        })
    }

    fn pplx_config() -> Value {
        json!({
            "format_version": 2,
            "codes": codes(),
            "token_ids": token_ids(),
            "temperature": 0.25,
            "attention_mode": "noncausal_full_attention",
            "pooling": "last"
        })
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| bf16::from_f32(*value).to_bits().to_le_bytes())
            .collect()
    }

    fn weight_values(stride: f32) -> Vec<f32> {
        (0..EXTERNAL_ROWS * HIDDEN)
            .map(|index| (index as f32 - 127.) * stride)
            .collect()
    }

    fn hidden_values() -> Vec<f32> {
        (0..5 * HIDDEN)
            .map(|index| (index as f32 - 7.) * 0.25)
            .collect()
    }

    fn hidden_bits(values: &[f32]) -> Vec<u16> {
        values
            .iter()
            .map(|value| bf16::from_f32(*value).to_bits())
            .collect()
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let scale = expected.abs().max(1.);
            assert!(
                (actual - expected).abs() <= 1e-4 * scale,
                "value {index}: actual={actual}, expected={expected}"
            );
        }
    }

    fn expected_dot(hidden: &[f32], weights: &[f32], row: usize) -> f32 {
        hidden
            .iter()
            .zip(&weights[row * HIDDEN..(row + 1) * HIDDEN])
            .map(|(hidden, weight)| hidden * weight)
            .sum()
    }

    #[test]
    fn d3_sidecar_requires_explicit_prefix_bidir_attention_and_fp32_readout() {
        let fixture = Fixture::new(backbone());
        fixture.decision_config(d3_config());
        fixture.readout(
            Dtype::F32,
            vec![EXTERNAL_ROWS, HIDDEN],
            f32_bytes(&weight_values(0.01)),
        );
        let head = ExternalHead::load(fixture.path()).unwrap();
        assert_eq!(head.family(), ExternalFamily::D3F32);
        assert_eq!(head.config().attention, ExternalAttention::NoncausalFull);
        assert!(!head.config().attention.block_attention());
        assert_eq!(head.config().temperature_for(0), 1.25);
        assert_eq!(head.config().temperature_for(1), 1.25);
        assert_eq!(head.config().temperature_for(2), 1.0);
        assert_eq!(head.config().temperature_for(9), 0.75);

        let mut block = d3_config();
        block["attention_mode"] = json!("prefix_bidir");
        fixture.decision_config(block);
        assert!(load_external_config(fixture.path())
            .unwrap()
            .attention
            .block_attention());

        for (field, value) in [
            ("format_version", json!(2)),
            ("prompt", json!("pplx")),
            ("pooling", json!("mean")),
            ("attention_mode", json!("causal")),
            ("readout_dtype", json!("bfloat16")),
        ] {
            let mut bad = d3_config();
            bad[field] = value;
            fixture.decision_config(bad);
            assert!(
                load_external_config(fixture.path()).is_err(),
                "{field} must be rejected"
            );
        }
        fixture.decision_config(d3_config());
        fixture.readout(
            Dtype::BF16,
            vec![EXTERNAL_ROWS, HIDDEN],
            bf16_bytes(&weight_values(0.01)),
        );
        assert!(matches!(
            ExternalHead::load(fixture.path()),
            Err(ExternalDecisionError::WrongWeightDtype { .. })
        ));
    }

    #[test]
    fn pplx_joint_sidecar_validates_calibration_and_case_marker_contract() {
        let fixture = Fixture::new(backbone());
        fixture.decision_config(pplx_config());
        fixture.joint_config(json!({
            "version": 3,
            "template": PPLX_JOINT_TEMPLATE,
            "marker": PPLX_JOINT_MARKER,
            "readout_offset": 0
        }));
        fixture.readout(
            Dtype::BF16,
            vec![EXTERNAL_ROWS, HIDDEN],
            bf16_bytes(&weight_values(0.01)),
        );
        let head = ExternalHead::load(fixture.path()).unwrap();
        assert_eq!(head.family(), ExternalFamily::PplxBf16);
        assert_eq!(
            head.config().joint.as_ref().unwrap().marker,
            PPLX_JOINT_MARKER
        );
        assert_eq!(head.config().temperature, 0.25);

        for (field, value) in [
            ("version", json!(2)),
            ("template", json!("legacy")),
            ("marker", json!("<|marker|>")),
            ("readout_offset", json!(1)),
        ] {
            let mut bad = json!({
                "version": 3,
                "template": PPLX_JOINT_TEMPLATE,
                "marker": PPLX_JOINT_MARKER,
                "readout_offset": 0
            });
            bad[field] = value;
            fixture.joint_config(bad);
            assert!(
                load_external_config(fixture.path()).is_err(),
                "{field} must be rejected"
            );
        }
        fixture.joint_config(json!({
            "version": 3,
            "template": PPLX_JOINT_TEMPLATE,
            "marker": PPLX_JOINT_MARKER,
            "readout_offset": 0
        }));
        let mut bad = pplx_config();
        bad["attention_mode"] = json!("causal");
        fixture.decision_config(bad);
        assert!(load_external_config(fixture.path()).is_err());
        fixture.decision_config(pplx_config());
        fixture.readout(
            Dtype::F32,
            vec![EXTERNAL_ROWS, HIDDEN],
            f32_bytes(&weight_values(0.01)),
        );
        assert!(matches!(
            ExternalHead::load(fixture.path()),
            Err(ExternalDecisionError::WrongWeightDtype { .. })
        ));
    }

    #[test]
    fn d3_plan_uses_full_projection_then_leading_code_selection() {
        let fixture = Fixture::new(backbone());
        fixture.decision_config(d3_config());
        let weights = weight_values(0.013);
        fixture.readout(Dtype::F32, vec![EXTERNAL_ROWS, HIDDEN], f32_bytes(&weights));
        let head = ExternalHead::load(fixture.path()).unwrap();
        let hidden = hidden_values();
        let bits = hidden_bits(&hidden);
        let rows = HiddenRows::from_bf16(&bits, HIDDEN).unwrap();
        let batch = PackedDecisionBatch::new(&[0, 3, 5]).unwrap();
        let requests = [
            DecisionInput::OptionTokens {
                token_ids: vec![0, 1, 2],
            },
            DecisionInput::Warmup,
        ];
        let prepared = head.prepare(&batch, &requests).unwrap();
        assert_eq!(prepared.output_counts(), &[3, 1]);
        assert_eq!(prepared.plan.full_rows, 2);
        let outputs = head.forward(rows, &batch, &requests, &prepared).unwrap();
        let last_a = bits[2 * HIDDEN..3 * HIDDEN]
            .iter()
            .map(|bits| bf16::from_bits(*bits).to_f32())
            .collect::<Vec<_>>();
        let last_b = bits[4 * HIDDEN..5 * HIDDEN]
            .iter()
            .map(|bits| bf16::from_bits(*bits).to_f32())
            .collect::<Vec<_>>();
        for (option, offset) in outputs[0].logits.iter().enumerate() {
            assert_close(&[*offset], &[expected_dot(&last_a, &weights, option)]);
        }
        // Warmup row zero is column zero of the same full projection.
        assert_close(&outputs[1].logits, &[expected_dot(&last_b, &weights, 0)]);
        assert_eq!(outputs[0].action_probability, 1.0);

        for token_ids in [vec![1, 0], vec![0, 2], vec![]] {
            let bad = [
                DecisionInput::OptionTokens { token_ids },
                DecisionInput::Warmup,
            ];
            assert!(
                plan_external_requests(ExternalFamily::D3F32, EXTERNAL_ROWS, &batch, &bad).is_err()
            );
        }
        let inline = [
            DecisionInput::InlineOptions {
                markers: vec![0],
                option_counts: vec![1],
            },
            DecisionInput::Warmup,
        ];
        assert!(
            plan_external_requests(ExternalFamily::D3F32, EXTERNAL_ROWS, &batch, &inline).is_err()
        );
    }

    #[test]
    fn pplx_plan_preserves_order_last_queries_warmup_and_joint_markers() {
        let fixture = Fixture::new(backbone());
        fixture.decision_config(pplx_config());
        let weights = weight_values(0.017);
        let rounded = weights
            .iter()
            .map(|value| bf16::from_f32(*value).to_f32())
            .collect::<Vec<_>>();
        fixture.readout(
            Dtype::BF16,
            vec![EXTERNAL_ROWS, HIDDEN],
            bf16_bytes(&weights),
        );
        let head = ExternalHead::load(fixture.path()).unwrap();
        let hidden = hidden_values();
        let bits = hidden_bits(&hidden);
        let rows = HiddenRows::from_bf16(&bits, HIDDEN).unwrap();
        let batch = PackedDecisionBatch::new(&[0, 3, 5]).unwrap();

        let token_requests = [
            DecisionInput::OptionTokens {
                token_ids: vec![0, 1],
            },
            DecisionInput::Warmup,
        ];
        let prepared = head.prepare(&batch, &token_requests).unwrap();
        assert_eq!(prepared.output_counts(), &[2, 1]);
        let outputs = head
            .forward(rows, &batch, &token_requests, &prepared)
            .unwrap();
        let last_a = bits[2 * HIDDEN..3 * HIDDEN]
            .iter()
            .map(|bits| bf16::from_bits(*bits).to_f32())
            .collect::<Vec<_>>();
        let last_b = bits[4 * HIDDEN..5 * HIDDEN]
            .iter()
            .map(|bits| bf16::from_bits(*bits).to_f32())
            .collect::<Vec<_>>();
        assert_close(
            &outputs[0].logits,
            &[
                expected_dot(&last_a, &rounded, 0),
                expected_dot(&last_a, &rounded, 1),
            ],
        );
        // Warmup always reads readout row zero from the last real token.
        assert_close(&outputs[1].logits, &[expected_dot(&last_b, &rounded, 0)]);

        let joint = [
            DecisionInput::InlineOptions {
                markers: vec![0, 1],
                option_counts: vec![2, 3],
            },
            DecisionInput::Warmup,
        ];
        let planned =
            plan_external_requests(ExternalFamily::PplxBf16, EXTERNAL_ROWS, &batch, &joint)
                .unwrap();
        assert_eq!(planned.counts, vec![5, 1]);
        assert_eq!(planned.row_offsets, vec![0, 5, 6]);
        assert_eq!(planned.marker_rows, vec![0, 1]);
        assert_eq!(planned.full_rows, 2);
        let prepared = head.prepare(&batch, &joint).unwrap();
        let outputs = head.forward(rows, &batch, &joint, &prepared).unwrap();
        let expected = expected_dot(&hidden, &rounded, 0);
        let _ = expected;
        let first = bits[..HIDDEN]
            .iter()
            .map(|bits| bf16::from_bits(*bits).to_f32())
            .collect::<Vec<_>>();
        let second = bits[HIDDEN..2 * HIDDEN]
            .iter()
            .map(|bits| bf16::from_bits(*bits).to_f32())
            .collect::<Vec<_>>();
        let mut expected = Vec::new();
        for option in 0..2 {
            expected.push(expected_dot(&first, &rounded, option));
        }
        for option in 0..3 {
            expected.push(expected_dot(&second, &rounded, option));
        }
        expected.push(expected_dot(&last_b, &rounded, 0));
        assert_close(&outputs[0].logits, &expected[..5]);
        assert_close(&outputs[1].logits, &expected[5..]);

        assert!(external_last_token_only(
            ExternalFamily::PplxBf16,
            &batch,
            &[
                DecisionInput::InlineOptions {
                    markers: vec![2],
                    option_counts: vec![1],
                },
                DecisionInput::OptionTokens { token_ids: vec![0] }
            ]
        ));
        assert!(!external_last_token_only(
            ExternalFamily::PplxBf16,
            &batch,
            &[
                DecisionInput::InlineOptions {
                    markers: vec![1],
                    option_counts: vec![1],
                },
                DecisionInput::OptionTokens { token_ids: vec![0] }
            ]
        ));
        assert!(!external_last_token_only(
            ExternalFamily::PplxBf16,
            &batch,
            &[
                DecisionInput::InlineOptions {
                    markers: vec![2],
                    option_counts: vec![1],
                },
                DecisionInput::Warmup
            ]
        ));
    }

    #[test]
    fn inline_plan_rejects_marker_order_counts_and_bounds_before_readout() {
        let batch = PackedDecisionBatch::new(&[0, 3, 5]).unwrap();
        for (markers, counts) in [
            (vec![1, 0], vec![1, 1]),
            (vec![0, 0], vec![1, 1]),
            (vec![0, 1], vec![0, 1]),
            (vec![0, 1], vec![1, EXTERNAL_ROWS + 1]),
            (vec![0, 5], vec![1, 1]),
        ] {
            let requests = [
                DecisionInput::InlineOptions {
                    markers,
                    option_counts: counts,
                },
                DecisionInput::Warmup,
            ];
            assert!(plan_external_requests(
                ExternalFamily::PplxBf16,
                EXTERNAL_ROWS,
                &batch,
                &requests
            )
            .is_err());
        }
    }
}
