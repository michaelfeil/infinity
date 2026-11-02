// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Cold Qwen2/Qwen3/Llama decoder score-head admission and prepared BF16 readout.
//!
//! Native dense decoders implement `Qwen{2,3}ForSequenceClassification`,
//! `Qwen{2,3}ForTokenClassification`, `LlamaForSequenceClassification` and
//! `LlamaForTokenClassification` by applying the root `score.weight` (and an
//! optional root `score.bias`) to final-model-dtype hidden states. This module
//! keeps that contract without routing through an F32 embedding copy:
//!
//! * cold metadata resolves exactly one matching dense family/task architecture;
//! * `score.weight` is admitted as BF16 `[labels, hidden]` and `score.bias` as
//!   BF16 `[labels]`, with Native `id2label`/`num_labels` and token-bias rules;
//! * the head uploads those tensors exactly once through `WeightUploader`;
//! * a sequence request selects the last non-PAD packed row (first row when the
//!   entire sequence is PAD), while a token request projects every packed row;
//! * a token head may also answer sequence readouts, but a sequence head rejects
//!   token readouts exactly as Native `predict(tokenwise=true)` does;
//! * biasless regression uses prepared plain BF16 cuBLAS; an actual checkpoint
//!   bias uses the shared `BiasedLinear` native-Lt/beta-one-preseed policy.
//!
//! The leaf never creates or binds the parent CUDA context, stream or cuBLAS
//! handle; it only borrows them while owning its weights, kernels and scratch.
//! The parent engine passes one generic final BF16 activation buffer, so the
//! same head can consume a decoder `Work` activation or a later Qwen3.5
//! `hidden0`/`hidden1` slab without a second facade.

#![allow(dead_code)]

use crate::{
    arg,
    biased_linear::{BiasedLinear, PreparedBiasedLinear},
    blocks,
    cublas::Blas,
    decoder_model::DecoderFamily,
    infer, launch, launch_threads,
    lt::{LtProvider, ShapeError, NATIVE_DEFAULT_SEED_ROWS},
    math::{GemmArguments, LinearPlan},
    model::{path_present, read_json, Checkpoint, WeightSource},
    product,
    tensor::Bf16,
    u32_size,
    upload::WeightUploader,
};
use cuda_core::{Context, DeviceBuffer, Function, Stream};
use infinity_v3_backend_core::BackendError;
use serde::Deserialize;
use std::{collections::BTreeMap, mem::size_of, path::Path, sync::Arc};

fn rejected(error: impl std::fmt::Display) -> ShapeError {
    ShapeError::Rejected(infer(error))
}

fn fatal(error: impl std::fmt::Display) -> ShapeError {
    ShapeError::Fatal(infer(error))
}

/// The two Native decoder classifier readouts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DecoderClassifierTask {
    Sequence,
    Tokens,
    /// Native `LlamaBidirectionalForSequenceClassification`: noncausal trunk,
    /// average pooling over each sequence, optional temperature divisor.
    BidirectionalSequence,
}

impl DecoderClassifierTask {
    pub(crate) fn architecture(self, family: DecoderFamily) -> Option<&'static str> {
        match (family, self) {
            (DecoderFamily::Qwen2, Self::Sequence) => Some("Qwen2ForSequenceClassification"),
            (DecoderFamily::Qwen2, Self::Tokens) => Some("Qwen2ForTokenClassification"),
            (DecoderFamily::Qwen3, Self::Sequence) => Some("Qwen3ForSequenceClassification"),
            (DecoderFamily::Qwen3, Self::Tokens) => Some("Qwen3ForTokenClassification"),
            (DecoderFamily::Llama, Self::Sequence) => Some("LlamaForSequenceClassification"),
            (DecoderFamily::Llama, Self::Tokens) => Some("LlamaForTokenClassification"),
            (DecoderFamily::Llama, Self::BidirectionalSequence) => {
                Some("LlamaBidirectionalForSequenceClassification")
            }
            _ => None,
        }
    }

    pub(crate) const fn is_sequence(self) -> bool {
        !matches!(self, Self::Tokens)
    }
}

/// Requested readout. A token-trained head supports both Native readouts; a
/// sequence-trained head supports only the sequence readout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DecoderClassifierReadout {
    Sequence,
    Tokens,
}

pub(crate) fn admit_readout(
    task: DecoderClassifierTask,
    readout: DecoderClassifierReadout,
) -> Result<(), BackendError> {
    if task.is_sequence() && readout == DecoderClassifierReadout::Tokens {
        return Err(infer(
            "Decoder sequence classifier does not implement token classification",
        ));
    }
    Ok(())
}

/// Native label-count rule, split out so admission is testable without files.
fn resolve_label_count(
    id2label_count: Option<usize>,
    num_labels: Option<usize>,
) -> Result<usize, BackendError> {
    let mapped = id2label_count.unwrap_or(0);
    let labels = if mapped > 0 {
        mapped
    } else {
        num_labels.unwrap_or(2)
    };
    if labels == 0 {
        return Err(crate::start("decoder classifier labels must be nonzero"));
    }
    if let Some(configured) = num_labels {
        if configured != labels {
            return Err(crate::start(format!(
                "decoder classifier num_labels={configured} disagrees with {labels} id2label entries"
            )));
        }
    }
    Ok(labels)
}

/// Native token-head bias rule. `token_classification_bias` defaults to true;
/// sequence heads keep the optional bias behavior.
fn validate_score_bias_contract(
    task: DecoderClassifierTask,
    configured: bool,
    present: bool,
) -> Result<(), BackendError> {
    if task == DecoderClassifierTask::Tokens && present != configured {
        return Err(crate::start(
            "decoder token classifier score.bias must match token_classification_bias",
        ));
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
struct RawDecoderClassifierConfig {
    model_type: String,
    architectures: Option<Vec<String>>,
    hidden_size: Option<usize>,
    num_labels: Option<usize>,
    id2label: Option<BTreeMap<String, String>>,
    pad_token_id: Option<i64>,
    token_classification_bias: Option<bool>,
    pooling: Option<String>,
    temperature: Option<f64>,
    use_linear_output_projection: Option<bool>,
    linear_output_size: Option<usize>,
    quantization_config: Option<serde_json::Value>,
    #[serde(flatten)]
    extra: BTreeMap<String, serde_json::Value>,
}

/// Validated cold classification metadata for one dense decoder checkpoint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct DecoderClassifierConfig {
    pub(crate) task: DecoderClassifierTask,
    pub(crate) family: DecoderFamily,
    pub(crate) hidden_size: usize,
    pub(crate) num_labels: usize,
    pub(crate) pad_token_id: Option<u32>,
    pub(crate) token_classification_bias: bool,
    /// Native bidirectional sequence head divisor. Kept as F32 because the
    /// score projection produces model-dtype BF16 before the divide.
    pub(crate) temperature: f32,
}

impl DecoderClassifierConfig {
    pub(crate) fn load(path: &Path) -> Result<Self, BackendError> {
        let value: serde_json::Value = read_json(&path.join("config.json"))?;
        Self::from_value(value)
    }

    pub(crate) fn from_value(value: serde_json::Value) -> Result<Self, BackendError> {
        let raw: RawDecoderClassifierConfig =
            serde_json::from_value(value).map_err(crate::start)?;
        Self::from_raw(raw)
    }

    fn from_raw(raw: RawDecoderClassifierConfig) -> Result<Self, BackendError> {
        let model_type = match raw.model_type.as_str() {
            "bidirectional_pplx_qwen3" => "qwen3",
            "llama_bidirec" => "llama",
            other => other,
        };
        let family = match model_type {
            "qwen2" => DecoderFamily::Qwen2,
            "qwen3" => DecoderFamily::Qwen3,
            "llama" => DecoderFamily::Llama,
            "mistral" | "ministral3" | "mistral3" => {
                return Err(crate::start(
                    "Native Mistral/Mistral3 classification does not use the standard root score architecture",
                ))
            }
            "qwen3_moe" => {
                return Err(crate::start(
                    "decoder classifier supports dense Qwen2/Qwen3/Llama only, not qwen3_moe",
                ))
            }
            other => {
                return Err(crate::start(format!(
                    "unsupported decoder classifier model_type {other}; expected qwen2, qwen3 or llama"
                )))
            }
        };
        let Some(architectures) = raw.architectures.as_deref() else {
            return Err(crate::start(
                "decoder classifier requires exactly one trained architecture",
            ));
        };
        let [architecture] = architectures else {
            return Err(crate::start(
                "decoder classifier requires exactly one trained architecture",
            ));
        };
        let task = [
            DecoderClassifierTask::Sequence,
            DecoderClassifierTask::Tokens,
            DecoderClassifierTask::BidirectionalSequence,
        ]
        .into_iter()
        .find(|task| task.architecture(family) == Some(architecture.as_str()))
        .ok_or_else(|| {
            crate::start(format!(
                "decoder classifier architecture {architecture} does not match model_type {:?}",
                raw.model_type
            ))
        })?;
        if raw.use_linear_output_projection == Some(true)
            || raw.linear_output_size.is_some_and(|size| size != 0)
        {
            return Err(crate::start(
                "decoder classifier does not accept an output projection",
            ));
        }
        let hidden_size = raw
            .hidden_size
            .filter(|hidden| *hidden > 0)
            .ok_or_else(|| crate::start("decoder classifier hidden_size must be nonzero"))?;
        let labels = resolve_label_count(raw.id2label.as_ref().map(BTreeMap::len), raw.num_labels)?;
        u32_size(hidden_size)?;
        u32_size(labels)?;
        product(hidden_size, labels)?;
        let pad_token_id = match raw.pad_token_id {
            Some(value) => Some(u32::try_from(value).map_err(|_| {
                crate::start("decoder classifier pad_token_id must fit the packed u32 token ABI")
            })?),
            None => None,
        };
        if raw
            .quantization_config
            .as_ref()
            .is_some_and(|value| !value.is_null())
        {
            return Err(crate::start(
                "decoder classifier does not accept quantized checkpoints",
            ));
        }
        let token_classification_bias = raw.token_classification_bias.unwrap_or(true);
        let temperature = if task == DecoderClassifierTask::BidirectionalSequence {
            if let Some(pooling) = raw.pooling.as_deref() {
                if pooling != "avg" {
                    return Err(crate::start(
                        "LlamaBidirectionalForSequenceClassification requires average pooling",
                    ));
                }
            }
            let temperature = raw.temperature.unwrap_or(1.0);
            if !temperature.is_finite() || temperature <= 0.0 || !(temperature as f32).is_finite() {
                return Err(crate::start(
                    "decoder classifier temperature must be finite and positive",
                ));
            }
            temperature as f32
        } else {
            1.0
        };
        let extra = raw.extra.keys().map(String::as_str).collect::<Vec<_>>();
        if extra.iter().any(|key| {
            let key = key.to_ascii_lowercase();
            key.contains("expert") || key.contains("quantiz") || key.contains("vision")
        }) {
            return Err(crate::start(
                "decoder classifier config contains unsupported expert/quantized/vision modifiers",
            ));
        }
        Ok(Self {
            task,
            family,
            hidden_size,
            num_labels: labels,
            pad_token_id,
            token_classification_bias,
            temperature,
        })
    }

    pub(crate) fn architecture(self) -> &'static str {
        self.task
            .architecture(self.family)
            .unwrap_or("DecoderClassification")
    }
}

/// Cheap metadata-only detection used by the backend before any CUDA context
/// exists. It admits exactly the six dense classification architectures this
/// leaf can serve and deliberately ignores malformed values for later errors.
pub(crate) fn value_is_classifier_architecture(value: &serde_json::Value) -> bool {
    let Some([architecture]) = value
        .get("architectures")
        .and_then(serde_json::Value::as_array)
        .map(Vec::as_slice)
    else {
        return false;
    };
    let Some(architecture) = architecture.as_str() else {
        return false;
    };
    is_supported_classifier_architecture(architecture)
}

pub(crate) fn is_supported_classifier_architecture(architecture: &str) -> bool {
    matches!(
        architecture,
        "Qwen2ForSequenceClassification"
            | "Qwen2ForTokenClassification"
            | "Qwen3ForSequenceClassification"
            | "Qwen3ForTokenClassification"
            | "LlamaForSequenceClassification"
            | "LlamaForTokenClassification"
            | "LlamaBidirectionalForSequenceClassification"
    )
}

/// Reject recognized decoder classification families this leaf does not
/// implement on the host. Without this check a Mistral/MoE/bidirectional-Llama
/// classification checkpoint would fall through to generic encoder admission.
pub(crate) fn reject_unsupported_classifier_architecture(
    value: &serde_json::Value,
) -> Result<(), BackendError> {
    let Some(architectures) = value
        .get("architectures")
        .and_then(serde_json::Value::as_array)
    else {
        return Ok(());
    };
    for architecture in architectures.iter().filter_map(serde_json::Value::as_str) {
        let classification = architecture.ends_with("ForSequenceClassification")
            || architecture.ends_with("ForTokenClassification");
        if classification && !is_supported_classifier_architecture(architecture) {
            return Err(infer(format!(
                "decoder classifier architecture {architecture} is a recognized Native family whose actual math is not ported; no encoder fallback"
            )));
        }
    }
    Ok(())
}

/// A conflicting root tensor that Native would interpret as an embedding
/// output projection instead of the required root classifier head.
fn output_projection_conflict(name: &str) -> Option<&'static str> {
    match name {
        "linear.weight" | "linear.bias" => Some("decoder output projection linear.*"),
        "linear_output_projection.weight" | "linear_output_projection.bias" => {
            Some("decoder output projection linear_output_projection.*")
        }
        _ => None,
    }
}

/// Single admission decision shared by both cold classifier loaders. An
/// explicit path list or any discovered admitted Dense module is rejected; an
/// empty path list alone is not.
pub(crate) fn classifier_dense_rejected(explicit_dense: bool, discovered: bool) -> bool {
    explicit_dense || discovered
}

pub(crate) fn reject_declared_dense_modules(path: &Path) -> Result<(), BackendError> {
    let file = path.join("modules.json");
    if !path_present(&file)? {
        return Ok(());
    }
    let modules: Vec<serde_json::Value> = read_json(&file)?;
    if modules.iter().any(|module| {
        module
            .get("type")
            .and_then(serde_json::Value::as_str)
            .is_some_and(crate::dense::is_dense_module_type)
    }) {
        return Err(crate::start(
            "decoder classifier does not accept declared Dense / Sentence-Transformers modules",
        ));
    }
    Ok(())
}

fn reject_conflicting_checkpoint_tensors(checkpoint: &Checkpoint) -> Result<(), BackendError> {
    for name in checkpoint.names() {
        if let Some(kind) = output_projection_conflict(name) {
            return Err(crate::start(format!(
                "decoder classifier does not accept {kind} tensor {name}"
            )));
        }
        if name != "score.weight" && name.ends_with("score.weight") {
            return Err(crate::start(format!(
                "decoder classifier score.weight must be at the checkpoint root, found {name}"
            )));
        }
        if name != "score.bias" && name.ends_with("score.bias") {
            return Err(crate::start(format!(
                "decoder classifier score.bias must be at the checkpoint root, found {name}"
            )));
        }
    }
    Ok(())
}

/// Host weight sources loaded once from one opened checkpoint. Weight payload
/// bytes remain lazy; `DecoderClassifierHead::new` uploads them exactly once.
pub(crate) struct DecoderClassifierHeadHost {
    pub(crate) config: DecoderClassifierConfig,
    weight: WeightSource,
    bias: Option<WeightSource>,
}

impl DecoderClassifierHeadHost {
    /// Validate config, projection/Dense exclusions and the exact root score
    /// tensors before any CUDA context or upload exists.
    pub(crate) fn load(
        path: &Path,
        dense_paths: Option<Vec<String>>,
    ) -> Result<Self, BackendError> {
        let config = DecoderClassifierConfig::load(path)?;
        reject_declared_dense_modules(path)?;
        let explicit_dense = dense_paths.as_ref().is_some_and(|paths| !paths.is_empty());
        let discovered = if explicit_dense {
            false
        } else {
            crate::dense::DenseChainHost::load_with_paths(path, config.hidden_size, dense_paths)?
                .is_some()
        };
        if classifier_dense_rejected(explicit_dense, discovered) {
            return Err(crate::start(
                "decoder classifier does not accept Dense / Sentence-Transformers modules",
            ));
        }
        let checkpoint = Checkpoint::open(path)?;
        Self::load_from_checkpoint(&checkpoint, config)
    }

    /// Extract the exact root score sources from an already opened checkpoint.
    /// This is the single head-weight admission point; callers that already own
    /// a decoder `Checkpoint` need not open a second metadata view.
    pub(crate) fn load_from_checkpoint(
        checkpoint: &Checkpoint,
        config: DecoderClassifierConfig,
    ) -> Result<Self, BackendError> {
        reject_conflicting_checkpoint_tensors(checkpoint)?;
        let weight = checkpoint.source("score.weight", &[config.num_labels, config.hidden_size])?;
        let bias = checkpoint.optional_source("score.bias", &[config.num_labels])?;
        validate_score_bias_contract(
            config.task,
            config.token_classification_bias,
            bias.is_some(),
        )?;
        Ok(Self {
            config,
            weight,
            bias,
        })
    }
}

/// Pure packed-row plan computed on the host before any request metadata upload.
///
/// `PackedRows` carries exactly one token-row id per sequence. Duplicates and
/// caller order are preserved by `from_row_ids`; every id is checked against the
/// physical row count. `Selected` is used when the trunk has already applied the
/// last-non-PAD selection before its final pointwise work. `Tokens` projects the
/// full packed token matrix.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RowPlan {
    kind: RowPlanKind,
    readout: DecoderClassifierReadout,
    rows: usize,
    source_rows: usize,
    row_ids: Vec<u32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RowPlanKind {
    Selected,
    PackedRows,
    Tokens,
    /// Full packed hidden rows reduced by the Native model-dtype mean over each
    /// contiguous sequence span before the score projection.
    MeanSequence,
}

/// Cold-prepared pointers and launch geometry for the Native BF16 mean. The
/// host computes the plan once from the request boundaries; the hot enqueue only
/// revalidates the pointers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MeanRows {
    pub(crate) boundaries: u64,
    pub(crate) indices: u64,
    pub(crate) grid: [u32; 3],
    pub(crate) threads: u32,
}

fn validate_sequence(
    input_ids: &[u32],
    boundaries: &[u32],
) -> Result<(usize, usize), BackendError> {
    let tokens = input_ids.len();
    if tokens == 0 {
        return Err(infer("decoder classifier requires at least one token"));
    }
    if boundaries.len() < 2 || boundaries.first() != Some(&0) {
        return Err(infer("decoder classifier boundaries are malformed"));
    }
    let mut previous = 0usize;
    for pair in boundaries.windows(2) {
        let (Some(&start), Some(&end)) = (pair.first(), pair.get(1)) else {
            return Err(infer("decoder classifier boundaries are malformed"));
        };
        let start = start as usize;
        let end = end as usize;
        if start != previous || end <= start || end > tokens {
            return Err(infer(
                "decoder classifier boundaries must be contiguous nonempty ranges",
            ));
        }
        previous = end;
    }
    if previous != tokens {
        return Err(infer(
            "decoder classifier boundaries must end at the packed token count",
        ));
    }
    Ok((tokens, boundaries.len() - 1))
}

fn selected_row(
    input_ids: &[u32],
    start: usize,
    end: usize,
    pad_token_id: Option<u32>,
) -> Result<u32, BackendError> {
    let row = match pad_token_id {
        None => end - 1,
        Some(pad) => {
            let mut chosen = start;
            for index in (start..end).rev() {
                let id = input_ids
                    .get(index)
                    .ok_or_else(|| infer("decoder classifier token range is invalid"))?;
                if *id != pad {
                    chosen = index;
                    break;
                }
            }
            chosen
        }
    };
    u32::try_from(row).map_err(|_| infer("decoder classifier selected row exceeds the u32 row ABI"))
}

impl RowPlan {
    pub(crate) fn kind(&self) -> RowPlanKind {
        self.kind
    }

    pub(crate) fn readout(&self) -> DecoderClassifierReadout {
        self.readout
    }

    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    pub(crate) fn source_rows(&self) -> usize {
        self.source_rows
    }

    pub(crate) fn row_ids(&self) -> &[u32] {
        &self.row_ids
    }

    pub(crate) fn requires_gather(&self) -> bool {
        self.kind == RowPlanKind::PackedRows
    }

    pub(crate) fn requires_mean(&self) -> bool {
        self.kind == RowPlanKind::MeanSequence
    }

    /// Native bidirectional sequence head: one mean row per packed sequence.
    pub(crate) fn mean_sequence(
        input_ids: &[u32],
        boundaries: &[u32],
    ) -> Result<Self, BackendError> {
        let (tokens, sequences) = validate_sequence(input_ids, boundaries)?;
        u32_size(tokens)?;
        u32_size(sequences)?;
        Ok(Self {
            kind: RowPlanKind::MeanSequence,
            readout: DecoderClassifierReadout::Sequence,
            rows: sequences,
            source_rows: tokens,
            row_ids: Vec::new(),
        })
    }

    /// Source rows already selected by the final decoder pointwise seam.
    pub(crate) fn selected_sequence(
        sequences: usize,
        source_rows: usize,
    ) -> Result<Self, BackendError> {
        if sequences == 0 {
            return Err(infer("decoder classifier requires at least one sequence"));
        }
        if source_rows < sequences {
            return Err(infer(
                "decoder classifier selected source has fewer rows than sequences",
            ));
        }
        u32_size(sequences)?;
        u32_size(source_rows)?;
        Ok(Self {
            kind: RowPlanKind::Selected,
            readout: DecoderClassifierReadout::Sequence,
            rows: sequences,
            source_rows,
            row_ids: Vec::new(),
        })
    }

    /// Every final packed token row, in physical order.
    pub(crate) fn tokens(tokens: usize) -> Result<Self, BackendError> {
        if tokens == 0 {
            return Err(infer("decoder classifier requires at least one token"));
        }
        u32_size(tokens)?;
        Ok(Self {
            kind: RowPlanKind::Tokens,
            readout: DecoderClassifierReadout::Tokens,
            rows: tokens,
            source_rows: tokens,
            row_ids: Vec::new(),
        })
    }

    /// Explicit row ids. Duplicates and order are preserved; ids must name
    /// physical rows inside `source_rows`.
    pub(crate) fn from_row_ids(
        source_rows: usize,
        row_ids: Vec<u32>,
    ) -> Result<Self, BackendError> {
        if source_rows == 0 {
            return Err(infer("decoder classifier source row count must be nonzero"));
        }
        if row_ids.is_empty() {
            return Err(infer("decoder classifier row plan must be nonempty"));
        }
        u32_size(source_rows)?;
        u32_size(row_ids.len())?;
        if row_ids.iter().any(|&row| row as usize >= source_rows) {
            return Err(infer(
                "decoder classifier row id exceeds the physical source rows",
            ));
        }
        Ok(Self {
            kind: RowPlanKind::PackedRows,
            readout: DecoderClassifierReadout::Sequence,
            rows: row_ids.len(),
            source_rows,
            row_ids,
        })
    }

    /// Native causal sequence selection: the last non-PAD token of each packed
    /// sequence, or its first token when every id is PAD. With no configured
    /// PAD id, the packed end row is selected.
    pub(crate) fn sequence(
        input_ids: &[u32],
        boundaries: &[u32],
        pad_token_id: Option<u32>,
    ) -> Result<Self, BackendError> {
        let (tokens, sequences) = validate_sequence(input_ids, boundaries)?;
        let mut selected = Vec::new();
        selected
            .try_reserve_exact(sequences)
            .map_err(|error| BackendError::Overloaded(error.to_string()))?;
        for pair in boundaries.windows(2) {
            let (Some(&start), Some(&end)) = (pair.first(), pair.get(1)) else {
                return Err(infer("decoder classifier boundaries are malformed"));
            };
            selected.push(selected_row(
                input_ids,
                start as usize,
                end as usize,
                pad_token_id,
            )?);
        }
        Self::from_row_ids(tokens, selected)
    }

    /// Allocation-free Native sequence selection. Writes exactly one physical
    /// token-row id per sequence into caller-owned host staging and returns the
    /// PackedRows plan describing that source. The caller already owns the
    /// `InputLayout.selections` slice, so the warm path performs no new Vec.
    pub(crate) fn sequence_into(
        input_ids: &[u32],
        boundaries: &[u32],
        pad_token_id: Option<u32>,
        out: &mut [u32],
    ) -> Result<Self, BackendError> {
        let (tokens, sequences) = validate_sequence(input_ids, boundaries)?;
        let staged = out
            .get_mut(..sequences)
            .ok_or_else(|| infer("decoder classifier row-id staging is too short"))?;
        for (pair, slot) in boundaries.windows(2).zip(staged.iter_mut()) {
            let (Some(&start), Some(&end)) = (pair.first(), pair.get(1)) else {
                return Err(infer("decoder classifier boundaries are malformed"));
            };
            *slot = selected_row(input_ids, start as usize, end as usize, pad_token_id)?;
        }
        Ok(Self {
            kind: RowPlanKind::PackedRows,
            readout: DecoderClassifierReadout::Sequence,
            rows: sequences,
            source_rows: tokens,
            row_ids: Vec::new(),
        })
    }

    /// Write explicitly supplied physical row ids into caller-owned packed
    /// input staging. This performs no upload. `Selected` rows were already
    /// gathered by the decoder seam, and `Tokens` has no selection ids, so
    /// neither may be staged here. A `sequence_into` plan is already staged.
    pub(crate) fn write_row_ids(&self, out: &mut [u32]) -> Result<usize, BackendError> {
        let RowPlanKind::PackedRows = self.kind else {
            return Err(infer(
                "decoder classifier selected/token plans have no writable row ids",
            ));
        };
        if self.row_ids.is_empty() {
            return Err(infer(
                "decoder classifier sequence_into rows are already staged",
            ));
        }
        let target = out
            .get_mut(..self.rows)
            .ok_or_else(|| infer("decoder classifier row-id staging is too short"))?;
        target.copy_from_slice(&self.row_ids);
        Ok(self.rows)
    }
}

/// Borrowed final BF16 hidden rows. No ownership or Qwen3.5-specific slab
/// identity is implied; the caller retains the owner through completion.
#[derive(Clone, Copy)]
pub(crate) struct DecoderClassifierRows<'a> {
    buffer: &'a DeviceBuffer<Bf16>,
    physical_rows: usize,
    width: usize,
}

impl<'a> DecoderClassifierRows<'a> {
    pub(crate) fn new(
        buffer: &'a DeviceBuffer<Bf16>,
        physical_rows: usize,
        width: usize,
    ) -> Result<Self, BackendError> {
        if physical_rows == 0 || width == 0 {
            return Err(infer(
                "decoder classifier hidden rows and width must be nonzero",
            ));
        }
        let elements = product(physical_rows, width)?;
        u32_size(elements)?;
        if buffer.len() < elements {
            return Err(infer(
                "decoder classifier hidden storage is shorter than its logical rows",
            ));
        }
        Ok(Self {
            buffer,
            physical_rows,
            width,
        })
    }

    pub(crate) fn buffer(&self) -> &'a DeviceBuffer<Bf16> {
        self.buffer
    }

    pub(crate) fn physical_rows(&self) -> usize {
        self.physical_rows
    }

    pub(crate) fn width(&self) -> usize {
        self.width
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DecoderClassifierScratchLayout {
    hidden: usize,
    labels: usize,
    sequence_rows: usize,
    token_rows: usize,
    gather: usize,
    logits: usize,
    mean: usize,
}

impl DecoderClassifierScratchLayout {
    pub(crate) fn for_key(
        hidden: usize,
        labels: usize,
        key: crate::WorkKey,
    ) -> Result<Self, BackendError> {
        if hidden == 0 || labels == 0 {
            return Err(infer(
                "decoder classifier hidden width and label count must be nonzero",
            ));
        }
        let (token_rows, sequence_rows, _) = key;
        if token_rows == 0 || sequence_rows == 0 {
            return Err(infer(
                "decoder classifier scratch requires nonzero token/sequence capacity",
            ));
        }
        u32_size(token_rows)?;
        u32_size(sequence_rows)?;
        u32_size(hidden)?;
        u32_size(labels)?;
        let gather = product(sequence_rows, hidden)?;
        let logits_rows = token_rows.max(sequence_rows);
        let logits = product(logits_rows, labels)?;
        let mean = product(sequence_rows, hidden)?;
        u32_size(gather)?;
        u32_size(logits)?;
        u32_size(mean)?;
        Ok(Self {
            hidden,
            labels,
            sequence_rows,
            token_rows,
            gather,
            logits,
            mean,
        })
    }

    pub(crate) fn hidden(&self) -> usize {
        self.hidden
    }

    pub(crate) fn labels(&self) -> usize {
        self.labels
    }

    pub(crate) fn sequence_rows(&self) -> usize {
        self.sequence_rows
    }

    pub(crate) fn token_rows(&self) -> usize {
        self.token_rows
    }

    pub(crate) fn gather_elements(&self) -> usize {
        self.gather
    }

    pub(crate) fn logits_elements(&self) -> usize {
        self.logits
    }

    pub(crate) fn mean_elements(&self) -> usize {
        self.mean
    }

    pub(crate) fn bytes(self) -> Result<usize, BackendError> {
        let bf16 = self
            .gather
            .checked_mul(size_of::<Bf16>())
            .and_then(|bytes| bytes.checked_add(self.logits.checked_mul(size_of::<Bf16>())?))
            .and_then(|bytes| bytes.checked_add(self.logits.checked_mul(size_of::<f32>())?))
            .and_then(|bytes| bytes.checked_add(self.mean.checked_mul(size_of::<f32>())?))
            .ok_or_else(|| infer("decoder classifier scratch byte size overflow"))?;
        Ok(bf16)
    }

    fn fits(&self, hidden: usize, labels: usize, plan: &RowPlan) -> Result<(), BackendError> {
        if self.hidden != hidden || self.labels != labels {
            return Err(infer(
                "decoder classifier scratch width does not match the trained head",
            ));
        }
        if (plan.requires_gather() || plan.requires_mean()) && plan.rows > self.sequence_rows {
            return Err(infer(
                "decoder classifier row gather/mean exceeds retained sequence capacity",
            ));
        }
        let logits_rows = self.logits / labels;
        if plan.rows > logits_rows {
            return Err(infer(
                "decoder classifier logits exceed retained row capacity",
            ));
        }
        Ok(())
    }
}

type MeanGatherLogits<'a> = (
    &'a DeviceBuffer<f32>,
    &'a mut DeviceBuffer<Bf16>,
    &'a mut DeviceBuffer<Bf16>,
);

/// Capacity-backed BF16 row gather, BF16 projection output and F32 logit read.
/// The outer Work owns this scratch and accounts `layout.bytes()`.
pub(crate) struct DecoderClassifierScratchBuffers {
    stream: Arc<Stream>,
    gather: Option<DeviceBuffer<Bf16>>,
    logits_bf16: Option<DeviceBuffer<Bf16>>,
    logits_f32: Option<DeviceBuffer<f32>>,
    mean_f32: Option<DeviceBuffer<f32>>,
    layout: DecoderClassifierScratchLayout,
    stream_drained: bool,
}

/// Pure failure-path disposition for classifier scratch. A Work is the single
/// byte owner: a failed Work drain already charged the whole byte class, so the
/// owned scratch must be retained without a second charge or synchronize. A
/// partially built scratch that failed before any Work existed bills exactly
/// its allocated bytes once and retains buffers plus stream/context.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ScratchDropPlan {
    pub(crate) charge_bytes: usize,
    pub(crate) forget_buffers: bool,
    pub(crate) forget_stream: bool,
}

impl ScratchDropPlan {
    pub(crate) const fn released() -> Self {
        Self {
            charge_bytes: 0,
            forget_buffers: false,
            forget_stream: false,
        }
    }

    pub(crate) const fn owned_by_failed_work() -> Self {
        Self {
            charge_bytes: 0,
            forget_buffers: true,
            forget_stream: true,
        }
    }

    pub(crate) const fn unowned_failure(allocated: usize) -> Self {
        Self {
            charge_bytes: allocated,
            forget_buffers: true,
            forget_stream: true,
        }
    }

    pub(crate) fn for_scratch_drop(allocated: usize, drained: bool, sync_ok: bool) -> Self {
        if drained || sync_ok {
            Self::released()
        } else {
            Self::unowned_failure(allocated)
        }
    }
}

fn scratch_buffer_bytes(
    gather: Option<&DeviceBuffer<Bf16>>,
    logits_bf16: Option<&DeviceBuffer<Bf16>>,
    logits_f32: Option<&DeviceBuffer<f32>>,
    mean_f32: Option<&DeviceBuffer<f32>>,
) -> usize {
    gather
        .map_or(0, DeviceBuffer::num_bytes)
        .saturating_add(logits_bf16.map_or(0, DeviceBuffer::num_bytes))
        .saturating_add(logits_f32.map_or(0, DeviceBuffer::num_bytes))
        .saturating_add(mean_f32.map_or(0, DeviceBuffer::num_bytes))
}

fn apply_scratch_drop_plan(
    plan: ScratchDropPlan,
    gather: &mut Option<DeviceBuffer<Bf16>>,
    logits_bf16: &mut Option<DeviceBuffer<Bf16>>,
    logits_f32: &mut Option<DeviceBuffer<f32>>,
    mean_f32: &mut Option<DeviceBuffer<f32>>,
    stream: &Arc<Stream>,
) {
    if plan.charge_bytes != 0 {
        crate::record_uncharged_retained_bytes(plan.charge_bytes);
    }
    if plan.forget_buffers {
        if let Some(buffer) = gather.take() {
            std::mem::forget(buffer);
        }
        if let Some(buffer) = logits_bf16.take() {
            std::mem::forget(buffer);
        }
        if let Some(buffer) = logits_f32.take() {
            std::mem::forget(buffer);
        }
        if let Some(buffer) = mean_f32.take() {
            std::mem::forget(buffer);
        }
        if plan.forget_stream {
            std::mem::forget(Arc::clone(stream));
        }
    } else {
        gather.take();
        logits_bf16.take();
        logits_f32.take();
        mean_f32.take();
    }
}

struct ScratchBuilding<'a> {
    stream: &'a Arc<Stream>,
    gather: Option<DeviceBuffer<Bf16>>,
    logits_bf16: Option<DeviceBuffer<Bf16>>,
    logits_f32: Option<DeviceBuffer<f32>>,
    mean_f32: Option<DeviceBuffer<f32>>,
}

impl Drop for ScratchBuilding<'_> {
    fn drop(&mut self) {
        let allocated = scratch_buffer_bytes(
            self.gather.as_ref(),
            self.logits_bf16.as_ref(),
            self.logits_f32.as_ref(),
            self.mean_f32.as_ref(),
        );
        if allocated == 0 {
            return;
        }
        let plan =
            ScratchDropPlan::for_scratch_drop(allocated, false, self.stream.synchronize().is_ok());
        apply_scratch_drop_plan(
            plan,
            &mut self.gather,
            &mut self.logits_bf16,
            &mut self.logits_f32,
            &mut self.mean_f32,
            self.stream,
        );
    }
}

fn initialized_bf16(
    stream: &Arc<Stream>,
    elements: usize,
) -> Result<DeviceBuffer<Bf16>, BackendError> {
    if elements == 0 {
        return Err(infer("decoder classifier BF16 scratch must be nonzero"));
    }
    // SAFETY: private allocation; the zero-fill below initializes the complete
    // capacity before any vectorized or vendor read observes the padding tail.
    let buffer =
        unsafe { DeviceBuffer::<Bf16>::uninitialized_async(stream, elements) }.map_err(infer)?;
    // SAFETY: the buffer is private and the memset covers its whole allocation
    // on the same ordered stream that later reads it.
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

fn uninitialized_f32(
    stream: &Arc<Stream>,
    elements: usize,
) -> Result<DeviceBuffer<f32>, BackendError> {
    if elements == 0 {
        return Err(infer("decoder classifier F32 scratch must be nonzero"));
    }
    // SAFETY: the BF16-to-F32 cast writes every requested logit before readback.
    unsafe { DeviceBuffer::<f32>::uninitialized_async(stream, elements) }.map_err(infer)
}

impl DecoderClassifierScratchBuffers {
    pub(crate) fn new(
        stream: Arc<Stream>,
        layout: DecoderClassifierScratchLayout,
    ) -> Result<Self, BackendError> {
        let (gather, logits_bf16, logits_f32, mean_f32) = {
            let mut building = ScratchBuilding {
                stream: &stream,
                gather: None,
                logits_bf16: None,
                logits_f32: None,
                mean_f32: None,
            };
            building.gather = Some(initialized_bf16(&stream, layout.gather)?);
            building.logits_bf16 = Some(initialized_bf16(&stream, layout.logits)?);
            building.logits_f32 = Some(uninitialized_f32(&stream, layout.logits)?);
            building.mean_f32 = Some(uninitialized_f32(&stream, layout.mean)?);
            (
                building.gather.take(),
                building.logits_bf16.take(),
                building.logits_f32.take(),
                building.mean_f32.take(),
            )
        };
        Ok(Self {
            stream,
            gather,
            logits_bf16,
            logits_f32,
            mean_f32,
            layout,
            stream_drained: false,
        })
    }

    pub(crate) fn layout(&self) -> DecoderClassifierScratchLayout {
        self.layout
    }

    /// # Safety
    /// The owning stream must have completed all work using these buffers.
    pub(crate) unsafe fn mark_stream_drained(&mut self) {
        self.stream_drained = true;
    }

    fn gather(&self) -> Result<&DeviceBuffer<Bf16>, BackendError> {
        self.gather
            .as_ref()
            .ok_or_else(|| infer("missing decoder classifier row-gather scratch"))
    }

    fn logits_bf16(&self) -> Result<&DeviceBuffer<Bf16>, BackendError> {
        self.logits_bf16
            .as_ref()
            .ok_or_else(|| infer("missing decoder classifier BF16 logits scratch"))
    }

    fn logits_bf16_mut(&mut self) -> Result<&mut DeviceBuffer<Bf16>, BackendError> {
        self.logits_bf16
            .as_mut()
            .ok_or_else(|| infer("missing decoder classifier BF16 logits scratch"))
    }

    fn gather_and_logits_mut(
        &mut self,
    ) -> Result<(&DeviceBuffer<Bf16>, &mut DeviceBuffer<Bf16>), BackendError> {
        let gather = self
            .gather
            .as_ref()
            .ok_or_else(|| infer("missing decoder classifier row-gather scratch"))?;
        let logits = self
            .logits_bf16
            .as_mut()
            .ok_or_else(|| infer("missing decoder classifier BF16 logits scratch"))?;
        Ok((gather, logits))
    }

    pub(crate) fn f32_logits(&self) -> Result<&DeviceBuffer<f32>, BackendError> {
        self.logits_f32
            .as_ref()
            .ok_or_else(|| infer("missing decoder classifier F32 logits scratch"))
    }

    pub(crate) fn mean_f32(&self) -> Result<&DeviceBuffer<f32>, BackendError> {
        self.mean_f32
            .as_ref()
            .ok_or_else(|| infer("missing decoder classifier F32 mean scratch"))
    }

    fn mean_gather_and_logits_mut(&mut self) -> Result<MeanGatherLogits<'_>, BackendError> {
        let mean = self
            .mean_f32
            .as_ref()
            .ok_or_else(|| infer("missing decoder classifier F32 mean scratch"))?;
        let gather = self
            .gather
            .as_mut()
            .ok_or_else(|| infer("missing decoder classifier row-gather scratch"))?;
        let logits = self
            .logits_bf16
            .as_mut()
            .ok_or_else(|| infer("missing decoder classifier BF16 logits scratch"))?;
        Ok((mean, gather, logits))
    }

    /// The owning Work already charged its full byte class and is retaining it
    /// after a failed drain. Leak buffers and stream without a second charge or
    /// synchronize so this scratch's Drop cannot run twice.
    pub(crate) fn retain_owned_by_failed_work(self) {
        let plan = ScratchDropPlan::owned_by_failed_work();
        debug_assert_eq!(plan.charge_bytes, 0);
        debug_assert!(plan.forget_buffers && plan.forget_stream);
        std::mem::forget(self);
    }
}

impl Drop for DecoderClassifierScratchBuffers {
    fn drop(&mut self) {
        if !self.stream_drained {
            #[cfg(test)]
            crate::note_default_drop_drain();
        }
        let allocated = scratch_buffer_bytes(
            self.gather.as_ref(),
            self.logits_bf16.as_ref(),
            self.logits_f32.as_ref(),
            self.mean_f32.as_ref(),
        );
        let plan = if self.stream_drained {
            ScratchDropPlan::released()
        } else {
            ScratchDropPlan::for_scratch_drop(allocated, false, self.stream.synchronize().is_ok())
        };
        apply_scratch_drop_plan(
            plan,
            &mut self.gather,
            &mut self.logits_bf16,
            &mut self.logits_f32,
            &mut self.mean_f32,
            &self.stream,
        );
    }
}

struct DecoderClassifierKernels {
    gather: Function,
    cast: Function,
    mean: Function,
    f32_to_bf16: Function,
    divide: Function,
    bias_seed: Option<Function>,
}

impl DecoderClassifierKernels {
    fn load(context: &Arc<Context>, with_bias: bool) -> Result<Self, BackendError> {
        let path = std::env::var_os(infinity_v3_config::infinity_env!("OXIDE_KERNEL_MODULE"))
            .ok_or_else(|| {
                crate::start(
                    "Set INFINITY_OXIDE_KERNEL_MODULE to the artifact compiled by cargo oxide",
                )
            })?;
        let path = std::path::PathBuf::from(path);
        let path = if path.is_dir() {
            let (major, minor) = context.compute_capability().map_err(crate::start)?;
            path.join(format!("sm_{major}{minor}"))
                .join("kernels.cubin")
        } else {
            path
        };
        let data = std::fs::read(&path)
            .map_err(|error| crate::start(format!("{}: {error}", path.display())))?;
        let module = context
            .load_module_from_image(&data)
            .map_err(crate::start)?;
        Ok(Self {
            gather: module
                .load_function("infinity_embedding_gather")
                .map_err(crate::start)?,
            cast: module
                .load_function("infinity_bf16_to_f32")
                .map_err(crate::start)?,
            mean: module
                .load_function("infinity_gemma2_mean_pool_bf16")
                .map_err(crate::start)?,
            f32_to_bf16: module
                .load_function("infinity_f32_to_bf16")
                .map_err(crate::start)?,
            divide: module
                .load_function("infinity_deberta_divide")
                .map_err(crate::start)?,
            bias_seed: with_bias
                .then(|| {
                    module
                        .load_function("infinity_classifier_bias_bf16")
                        .map_err(crate::start)
                })
                .transpose()?,
        })
    }
}

struct DeviceClassifierWeight {
    weight: DeviceBuffer<Bf16>,
    bias: Option<DeviceBuffer<Bf16>>,
}

#[allow(clippy::large_enum_variant)]
enum LinearPolicy {
    Biased(BiasedLinear),
    Plain(LinearPlan<Bf16>),
}

enum PreparedLinear {
    Biased(PreparedBiasedLinear),
    Plain(GemmArguments),
}

struct DecoderClassifierPrepared {
    kind: RowPlanKind,
    rows: usize,
    source_rows: usize,
    source: u64,
    stream: usize,
    row_ids: u64,
    mean_bounds: u64,
    mean_indices: u64,
    mean_grid: [u32; 3],
    mean_threads: u32,
    gather: u64,
    logits_bf16: u64,
    logits_f32: u64,
    linear: PreparedLinear,
}

/// Cold-uploaded decoder score head. Weights live as one BF16 matrix and an
/// optional BF16 bias; no FP32 duplicate is created.
pub(crate) struct DecoderClassifierHead {
    config: DecoderClassifierConfig,
    context: Arc<Context>,
    stream: Arc<Stream>,
    weights: DeviceClassifierWeight,
    policy: LinearPolicy,
    kernels: DecoderClassifierKernels,
    prepared: Option<DecoderClassifierPrepared>,
}

/// Plan-aware admission matching Native `predict`: a sequence head owns the
/// pre-final-seam Selected/PackedRows sequence readout and rejects Tokens;
/// a token head serves sequence readout only from a full-T PackedRows plan
/// and token readout only from a full-T Tokens plan. A Selected source can
/// never answer a token head sequence request because that would skip the
/// required full-T forward.
fn admit_plan(task: DecoderClassifierTask, plan: &RowPlan) -> Result<(), BackendError> {
    match (task, plan.kind(), plan.readout()) {
        (DecoderClassifierTask::Sequence, RowPlanKind::Selected, DecoderClassifierReadout::Sequence)
        | (DecoderClassifierTask::Sequence, RowPlanKind::PackedRows, DecoderClassifierReadout::Sequence)
        | (
            DecoderClassifierTask::BidirectionalSequence,
            RowPlanKind::MeanSequence,
            DecoderClassifierReadout::Sequence,
        )
        | (DecoderClassifierTask::Tokens, RowPlanKind::PackedRows, DecoderClassifierReadout::Sequence)
        | (DecoderClassifierTask::Tokens, RowPlanKind::Tokens, DecoderClassifierReadout::Tokens) => Ok(()),
        (
            DecoderClassifierTask::Sequence | DecoderClassifierTask::BidirectionalSequence,
            _,
            DecoderClassifierReadout::Tokens,
        ) => Err(infer(
            "Decoder sequence classifier does not implement token classification",
        )),
        (DecoderClassifierTask::Tokens, RowPlanKind::Selected, _) => Err(infer(
            "Decoder token classifier sequence readout requires the full token matrix before gathering",
        )),
        _ => Err(infer(
            "Decoder classifier row plan does not match the trained head task/readout",
        )),
    }
}

impl DecoderClassifierHead {
    pub(crate) fn new(
        host: DecoderClassifierHeadHost,
        context: &Arc<Context>,
        stream: &Arc<Stream>,
        provider: &LtProvider,
    ) -> Result<Self, BackendError> {
        let DecoderClassifierHeadHost {
            config,
            weight,
            bias,
        } = host;
        let policy = if bias.is_some() {
            LinearPolicy::Biased(
                BiasedLinear::new(provider, config.hidden_size, config.num_labels)
                    .map_err(ShapeError::into_backend)?,
            )
        } else {
            LinearPolicy::Plain(
                LinearPlan::<Bf16>::contiguous(
                    NATIVE_DEFAULT_SEED_ROWS,
                    config.hidden_size,
                    config.num_labels,
                )
                .map_err(infer)?,
            )
        };
        let kernels = DecoderClassifierKernels::load(context, bias.is_some())?;
        let mut uploader = WeightUploader::new(stream.clone())?;
        let device_weight = uploader.upload(&[&weight])?;
        let device_bias = bias
            .as_ref()
            .map(|bias| uploader.upload(&[bias]))
            .transpose()?;
        uploader.finish()?;
        Ok(Self {
            config,
            context: context.clone(),
            stream: stream.clone(),
            weights: DeviceClassifierWeight {
                weight: device_weight,
                bias: device_bias,
            },
            policy,
            kernels,
            prepared: None,
        })
    }

    pub(crate) fn config(&self) -> DecoderClassifierConfig {
        self.config
    }

    pub(crate) fn task(&self) -> DecoderClassifierTask {
        self.config.task
    }

    pub(crate) fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    pub(crate) fn pad_token_id(&self) -> Option<u32> {
        self.config.pad_token_id
    }

    pub(crate) fn width(&self) -> usize {
        self.config.num_labels
    }

    pub(crate) fn scratch_layout(
        &self,
        key: crate::WorkKey,
    ) -> Result<DecoderClassifierScratchLayout, BackendError> {
        DecoderClassifierScratchLayout::for_key(
            self.config.hidden_size,
            self.config.num_labels,
            key,
        )
    }

    pub(crate) fn admit(&self, readout: DecoderClassifierReadout) -> Result<(), BackendError> {
        admit_readout(self.config.task, readout)
    }

    pub(crate) fn preflight(
        &self,
        stream: &Stream,
        source: DecoderClassifierRows<'_>,
        plan: &RowPlan,
        row_ids: Option<u64>,
        mean: Option<MeanRows>,
        scratch: &DecoderClassifierScratchBuffers,
    ) -> Result<(), ShapeError> {
        admit_plan(self.config.task, plan).map_err(ShapeError::Rejected)?;
        if source.width() != self.config.hidden_size {
            return Err(rejected(format!(
                "decoder classifier hidden width {} differs from trained width {}",
                source.width(),
                self.config.hidden_size
            )));
        }
        if plan.rows() == 0 || plan.rows() > u32::MAX as usize {
            return Err(rejected(
                "decoder classifier row count must be nonzero and fit u32",
            ));
        }
        match plan.kind() {
            RowPlanKind::Selected | RowPlanKind::Tokens => {
                if source.physical_rows() != plan.source_rows() {
                    return Err(rejected(
                        "decoder classifier source row count differs from its row plan",
                    ));
                }
            }
            RowPlanKind::PackedRows => {
                if source.physical_rows() != plan.source_rows() {
                    return Err(rejected(
                        "decoder classifier packed source row count differs from its row plan",
                    ));
                }
                if row_ids.is_none_or(|address| address == 0) {
                    return Err(rejected(
                        "decoder classifier packed rows require uploaded host-prepared row ids",
                    ));
                }
            }
            RowPlanKind::MeanSequence => {
                if source.physical_rows() != plan.source_rows() {
                    return Err(rejected(
                        "decoder classifier mean source row count differs from its row plan",
                    ));
                }
                if mean.is_none_or(|rows| {
                    rows.boundaries == 0 || rows.indices == 0 || rows.threads == 0
                }) {
                    return Err(rejected(
                        "decoder classifier mean rows require uploaded boundaries and canonical indices",
                    ));
                }
            }
        }
        scratch
            .layout()
            .fits(self.config.hidden_size, self.config.num_labels, plan)
            .map_err(ShapeError::Rejected)?;
        if !Arc::ptr_eq(&self.context, source.buffer().context())
            || !Arc::ptr_eq(&self.context, stream.context())
            || !std::ptr::eq(stream, self.stream.as_ref())
            || !std::ptr::eq(stream, scratch.stream.as_ref())
            || !Arc::ptr_eq(&self.context, scratch.stream.context())
        {
            return Err(fatal(
                "decoder classifier source, scratch, stream and weights must share one CUDA context/stream",
            ));
        }
        Ok(())
    }

    fn prepare_linear(
        &mut self,
        provider: &LtProvider,
        blas: &Blas,
        rows: usize,
        stream: &Stream,
        input: &DeviceBuffer<Bf16>,
        output: &mut DeviceBuffer<Bf16>,
    ) -> Result<PreparedLinear, ShapeError> {
        match &mut self.policy {
            LinearPolicy::Biased(plan) => {
                let bias = self.weights.bias.as_ref().ok_or_else(|| {
                    fatal("decoder classifier biased policy has no uploaded score.bias")
                })?;
                // SAFETY: the caller's bound-context and retained-buffer contract
                // covers all pointers; geometry was checked just above.
                let prepared = unsafe {
                    plan.prepare(
                        provider,
                        blas,
                        rows,
                        stream,
                        input,
                        &self.weights.weight,
                        bias,
                        output,
                    )?
                };
                Ok(PreparedLinear::Biased(prepared))
            }
            LinearPolicy::Plain(plan) => {
                plan.set_rows(rows)
                    .map_err(|error| ShapeError::Rejected(infer(error)))?;
                let arguments =
                    blas.preflight_beta_zero(plan, input, &self.weights.weight, output, stream)?;
                Ok(PreparedLinear::Plain(arguments))
            }
        }
    }

    /// Bind every actual head pointer and select the Native-faithful method
    /// before request submission. A cold bias plan may run the existing Lt
    /// heuristic; this method allocates, synchronizes and queries nothing else.
    ///
    /// # Safety
    /// The bound CUDA context and all owners must remain current through the
    /// ordered launch. `row_ids` must address `plan.rows()` host-prepared u32
    /// row ids in caller-owned staging whenever `plan.requires_gather()`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn prepare(
        &mut self,
        provider: &LtProvider,
        blas: &Blas,
        stream: &Stream,
        source: DecoderClassifierRows<'_>,
        plan: &RowPlan,
        row_ids: Option<u64>,
        mean: Option<MeanRows>,
        scratch: &mut DecoderClassifierScratchBuffers,
    ) -> Result<(), ShapeError> {
        self.prepared = None;
        self.preflight(stream, source, plan, row_ids, mean, scratch)?;
        let rows = plan.rows();
        let source_pointer = source.buffer().cu_deviceptr();
        let stream_identity = stream.cu_stream() as usize;
        let row_id_pointer = row_ids.unwrap_or(0);
        let mean = mean.unwrap_or(MeanRows {
            boundaries: 0,
            indices: 0,
            grid: [0; 3],
            threads: 0,
        });
        let gather_pointer = scratch
            .gather()
            .map(DeviceBuffer::cu_deviceptr)
            .map_err(fatal)?;
        let logits_bf16 = scratch
            .logits_bf16()
            .map(DeviceBuffer::cu_deviceptr)
            .map_err(fatal)?;
        let logits_f32 = scratch
            .f32_logits()
            .map(DeviceBuffer::cu_deviceptr)
            .map_err(fatal)?;
        let linear = match plan.kind() {
            RowPlanKind::PackedRows => {
                let (gather, output) = scratch.gather_and_logits_mut().map_err(fatal)?;
                self.prepare_linear(provider, blas, rows, stream, gather, output)?
            }
            RowPlanKind::MeanSequence => {
                let (_, gather, output) = scratch.mean_gather_and_logits_mut().map_err(fatal)?;
                self.prepare_linear(provider, blas, rows, stream, gather, output)?
            }
            RowPlanKind::Selected | RowPlanKind::Tokens => {
                let output = scratch.logits_bf16_mut().map_err(fatal)?;
                self.prepare_linear(provider, blas, rows, stream, source.buffer(), output)?
            }
        };
        self.prepared = Some(DecoderClassifierPrepared {
            kind: plan.kind(),
            rows,
            source_rows: plan.source_rows(),
            source: source_pointer,
            stream: stream_identity,
            row_ids: row_id_pointer,
            mean_bounds: mean.boundaries,
            mean_indices: mean.indices,
            mean_grid: mean.grid,
            mean_threads: mean.threads,
            gather: gather_pointer,
            logits_bf16,
            logits_f32,
            linear,
        });
        Ok(())
    }

    fn launch_linear(
        &mut self,
        provider: &LtProvider,
        blas: &Blas,
        stream: &Stream,
        input: &DeviceBuffer<Bf16>,
        output: &mut DeviceBuffer<Bf16>,
        prepared: PreparedLinear,
    ) -> Result<(), BackendError> {
        if !std::ptr::eq(blas.stream(), stream) {
            return Err(infer(
                "decoder classifier cuBLAS handle differs from the ordered request stream",
            ));
        }
        match (&mut self.policy, prepared) {
            (LinearPolicy::Biased(plan), PreparedLinear::Biased(prepared)) => {
                let bias =
                    self.weights.bias.as_ref().ok_or_else(|| {
                        infer("decoder classifier biased policy has no score.bias")
                    })?;
                let seed = self
                    .kernels
                    .bias_seed
                    .as_ref()
                    .ok_or_else(|| infer("decoder classifier bias-seed kernel is missing"))?;
                // SAFETY: the caller's retained-buffer contract covers the
                // preselected launch and its revalidated boundary.
                unsafe {
                    plan.launch(
                        &prepared,
                        provider,
                        blas,
                        seed,
                        stream,
                        input,
                        &self.weights.weight,
                        bias,
                        output,
                    )
                }
            }
            (LinearPolicy::Plain(plan), PreparedLinear::Plain(arguments)) => {
                let current = plan
                    .bind(input, &self.weights.weight, output)
                    .map_err(|error| {
                        infer(format!("decoder classifier binding changed: {error}"))
                    })?;
                if current != arguments {
                    return Err(infer(
                        "decoder classifier GEMM arguments changed after preparation",
                    ));
                }
                // SAFETY: current binding equals the prepared arguments and the
                // caller retains every owner through stream completion.
                unsafe { blas.linear(plan, input, &self.weights.weight, output) }
            }
            _ => Err(infer(
                "decoder classifier prepared method does not match the loaded bias policy",
            )),
        }
    }

    /// Launch the prepared row gather, BF16 score projection and BF16-to-F32
    /// cast on the shared ordered stream. No allocation/query/sync occurs.
    ///
    /// # Safety
    /// The caller retains source, row-id staging, scratch, weights, module and
    /// stream through completion and passes the same buffers used at prepare.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn enqueue(
        &mut self,
        provider: &LtProvider,
        blas: &Blas,
        stream: &Stream,
        source: DecoderClassifierRows<'_>,
        row_ids: Option<u64>,
        mean: Option<MeanRows>,
        scratch: &mut DecoderClassifierScratchBuffers,
    ) -> Result<(), BackendError> {
        let prepared = self
            .prepared
            .take()
            .ok_or_else(|| infer("decoder classifier readout was not prepared before launch"))?;
        let DecoderClassifierPrepared {
            kind,
            rows,
            source_rows,
            source: prepared_source,
            stream: prepared_stream,
            row_ids: prepared_row_ids,
            mean_bounds: prepared_mean_bounds,
            mean_indices: prepared_mean_indices,
            mean_grid: prepared_mean_grid,
            mean_threads: prepared_mean_threads,
            gather,
            logits_bf16,
            logits_f32,
            linear,
        } = prepared;
        let current_row_ids = row_ids.unwrap_or(0);
        let mean = mean.unwrap_or(MeanRows {
            boundaries: 0,
            indices: 0,
            grid: [0; 3],
            threads: 0,
        });
        let current_mean_bounds = mean.boundaries;
        let current_mean_indices = mean.indices;
        if source.buffer().cu_deviceptr() != prepared_source
            || source.physical_rows() != source_rows
            || source.width() != self.config.hidden_size
            || stream.cu_stream() as usize != prepared_stream
            || current_row_ids != prepared_row_ids
            || current_mean_bounds != prepared_mean_bounds
            || current_mean_indices != prepared_mean_indices
            || mean.grid != prepared_mean_grid
            || mean.threads != prepared_mean_threads
            || scratch
                .gather()
                .map(DeviceBuffer::cu_deviceptr)
                .map_err(infer)?
                != gather
            || scratch
                .logits_bf16()
                .map(DeviceBuffer::cu_deviceptr)
                .map_err(infer)?
                != logits_bf16
            || scratch
                .f32_logits()
                .map(DeviceBuffer::cu_deviceptr)
                .map_err(infer)?
                != logits_f32
        {
            return Err(infer(
                "decoder classifier source, row ids, scratch or stream differs from prepared metadata",
            ));
        }
        match kind {
            RowPlanKind::PackedRows => {
                if current_row_ids == 0 {
                    return Err(infer(
                        "decoder classifier row gather requires uploaded row ids",
                    ));
                }
                let (gather, output) = scratch.gather_and_logits_mut()?;
                let mut ids = current_row_ids;
                let mut table = source.buffer().cu_deviceptr();
                let mut out = gather.cu_deviceptr();
                let mut count = u32_size(rows)?;
                let mut width = u32_size(self.config.hidden_size)?;
                launch(
                    &self.kernels.gather,
                    stream,
                    blocks(product(rows, self.config.hidden_size)?)?,
                    0,
                    &mut [
                        arg(&mut ids),
                        arg(&mut table),
                        arg(&mut out),
                        arg(&mut count),
                        arg(&mut width),
                    ],
                )?;
                self.launch_linear(provider, blas, stream, gather, output, linear)?;
            }
            RowPlanKind::MeanSequence => {
                if current_mean_bounds == 0 || current_mean_indices == 0 {
                    return Err(infer(
                        "decoder classifier mean rows require uploaded boundaries and indices",
                    ));
                }
                let (mean_buffer, gather, output) = scratch.mean_gather_and_logits_mut()?;
                let mut input = source.buffer().cu_deviceptr();
                let mut bounds = current_mean_bounds;
                let mut indices = current_mean_indices;
                let mut mean_out = mean_buffer.cu_deviceptr();
                let mut count = u32_size(rows)?;
                let mut hidden = u32_size(self.config.hidden_size)?;
                launch_threads(
                    &self.kernels.mean,
                    stream,
                    (mean.grid[0], mean.grid[1], mean.grid[2]),
                    mean.threads,
                    0,
                    &mut [
                        arg(&mut input),
                        arg(&mut bounds),
                        arg(&mut indices),
                        arg(&mut mean_out),
                        arg(&mut count),
                        arg(&mut hidden),
                    ],
                )?;
                let mut mean_input = mean_buffer.cu_deviceptr();
                let mut gather_out = gather.cu_deviceptr();
                let mut elements = u32_size(product(rows, self.config.hidden_size)?)?;
                launch(
                    &self.kernels.f32_to_bf16,
                    stream,
                    blocks(elements as usize)?,
                    0,
                    &mut [
                        arg(&mut mean_input),
                        arg(&mut gather_out),
                        arg(&mut elements),
                    ],
                )?;
                self.launch_linear(provider, blas, stream, gather, output, linear)?;
            }
            RowPlanKind::Selected | RowPlanKind::Tokens => {
                let output = scratch.logits_bf16_mut()?;
                self.launch_linear(provider, blas, stream, source.buffer(), output, linear)?;
            }
        }
        if self.config.task == DecoderClassifierTask::BidirectionalSequence
            && self.config.temperature != 1.0
        {
            let mut logits = scratch.logits_bf16_mut()?.cu_deviceptr();
            let mut elements = u32_size(product(rows, self.config.num_labels)?)?;
            let mut temperature = self.config.temperature;
            launch(
                &self.kernels.divide,
                stream,
                blocks(elements as usize)?,
                0,
                &mut [arg(&mut logits), arg(&mut elements), arg(&mut temperature)],
            )?;
        }
        let input = scratch.logits_bf16()?;
        let output = scratch.f32_logits()?;
        let mut input = input.cu_deviceptr();
        let mut output = output.cu_deviceptr();
        let mut selected = 0u64;
        let mut rows = u32_size(rows)?;
        let mut width = u32_size(self.config.num_labels)?;
        launch(
            &self.kernels.cast,
            stream,
            blocks(product(rows as usize, self.config.num_labels)?)?,
            0,
            &mut [
                arg(&mut input),
                arg(&mut output),
                arg(&mut selected),
                arg(&mut rows),
                arg(&mut width),
            ],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const PAD: u32 = 31;

    fn raw(value: serde_json::Value) -> Result<DecoderClassifierConfig, BackendError> {
        DecoderClassifierConfig::from_value(value)
    }

    fn config(architecture: &str) -> serde_json::Value {
        json!({
            "model_type": "qwen3",
            "architectures": [architecture],
            "hidden_size": 8,
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "vocab_size": 64,
            "max_position_embeddings": 64,
            "id2label": {"0": "no", "1": "yes"},
            "label2id": {"no": 0, "yes": 1},
            "pad_token_id": PAD
        })
    }

    #[test]
    fn row_plan_selects_last_nonpad_internal_pad_and_allpad_first() {
        let ids = [5, 9, PAD, 7, PAD, PAD, 11, PAD, 13, PAD];
        let boundaries = [0, 4, 6, 10];
        let plan = RowPlan::sequence(&ids, &boundaries, Some(PAD)).unwrap();
        assert_eq!(plan.kind(), RowPlanKind::PackedRows);
        assert_eq!(plan.readout(), DecoderClassifierReadout::Sequence);
        assert_eq!(plan.row_ids(), &[3, 4, 8]);
        assert_eq!((plan.rows(), plan.source_rows()), (3, 10));
        assert!(plan.requires_gather());
    }

    #[test]
    fn row_plan_without_pad_and_ragged_boundaries_use_end_rows() {
        let ids = [7, 8, 9, 10, 11, 12, 13, 14];
        let boundaries = [0, 1, 6, 8];
        let plan = RowPlan::sequence(&ids, &boundaries, None).unwrap();
        assert_eq!(plan.row_ids(), &[0, 5, 7]);
        let explicit = RowPlan::from_row_ids(ids.len(), vec![5, 0, 5, 7]).unwrap();
        assert_eq!(explicit.row_ids(), &[5, 0, 5, 7]);
        let mut staged = [u32::MAX; 4];
        assert_eq!(explicit.write_row_ids(&mut staged).unwrap(), 4);
        assert_eq!(staged, [5, 0, 5, 7]);
    }

    #[test]
    fn row_plan_rejects_malformed_empty_and_out_of_bounds_inputs() {
        assert!(RowPlan::tokens(0).is_err());
        assert!(RowPlan::selected_sequence(0, 4).is_err());
        assert!(RowPlan::selected_sequence(3, 2).is_err());
        assert!(RowPlan::from_row_ids(4, Vec::new()).is_err());
        assert!(RowPlan::from_row_ids(4, vec![4]).is_err());
        assert!(RowPlan::from_row_ids(usize::MAX, vec![0]).is_err());
        assert!(RowPlan::sequence(&[], &[0], Some(PAD)).is_err());
        assert!(RowPlan::sequence(&[1], &[0], Some(PAD)).is_err());
        assert!(RowPlan::sequence(&[1], &[1, 1], Some(PAD)).is_err());
        assert!(RowPlan::sequence(&[1, 2], &[0, 0, 2], Some(PAD)).is_err());
        assert!(RowPlan::sequence(&[1, 2], &[0, 2, 1], Some(PAD)).is_err());
        assert!(RowPlan::sequence(&[1, 2], &[0, 1, 3], Some(PAD)).is_err());
        assert!(RowPlan::sequence(&[1, 2], &[0, 1, 2], Some(PAD)).is_ok());
    }

    #[test]
    fn config_requires_one_exact_family_architecture_and_native_labels() {
        let sequence = raw(config("Qwen3ForSequenceClassification")).unwrap();
        assert_eq!(sequence.task, DecoderClassifierTask::Sequence);
        assert_eq!(sequence.family, DecoderFamily::Qwen3);
        assert_eq!((sequence.hidden_size, sequence.num_labels), (8, 2));
        assert_eq!(sequence.pad_token_id, Some(PAD));

        let tokens = raw(config("Qwen3ForTokenClassification")).unwrap();
        assert_eq!(tokens.task, DecoderClassifierTask::Tokens);
        assert!(tokens.token_classification_bias);

        let mut alias = config("Qwen3ForSequenceClassification");
        alias["model_type"] = json!("bidirectional_pplx_qwen3");
        let alias = raw(alias).unwrap();
        assert_eq!(alias.family, DecoderFamily::Qwen3);
        assert_eq!(alias.task, DecoderClassifierTask::Sequence);

        assert!(raw(config("Qwen2ForSequenceClassification")).is_err());
        assert!(raw(config("Qwen3ForCausalLM")).is_err());
        assert!(raw(json!({
            "model_type": "qwen3",
            "architectures": ["Qwen3ForSequenceClassification", "Qwen3ForTokenClassification"],
            "hidden_size": 8
        }))
        .is_err());
        assert!(raw(json!({
            "model_type": "mistral",
            "architectures": ["MistralForSequenceClassification"],
            "hidden_size": 8
        }))
        .is_err());
        let mut bidirectional = config("LlamaBidirectionalForSequenceClassification");
        bidirectional["model_type"] = json!("llama_bidirec");
        bidirectional["pooling"] = json!("avg");
        bidirectional["temperature"] = json!(0.7);
        bidirectional["num_labels"] = json!(1);
        bidirectional["id2label"] = json!({"0": "LABEL_0"});
        let bidirectional = raw(bidirectional).unwrap();
        assert_eq!(
            bidirectional.task,
            DecoderClassifierTask::BidirectionalSequence
        );
        assert_eq!(bidirectional.family, DecoderFamily::Llama);
        assert_eq!(bidirectional.num_labels, 1);
        assert_eq!(bidirectional.temperature, 0.7);

        let mut wrong_pooling = config("LlamaBidirectionalForSequenceClassification");
        wrong_pooling["model_type"] = json!("llama");
        wrong_pooling["pooling"] = json!("cls");
        assert!(raw(wrong_pooling).is_err());
        let mut bad_temperature = config("LlamaBidirectionalForSequenceClassification");
        bad_temperature["model_type"] = json!("llama");
        bad_temperature["temperature"] = json!(0.0);
        assert!(raw(bad_temperature).is_err());
        assert!(raw(json!({
            "model_type": "qwen3",
            "architectures": ["Qwen3ForSequenceClassification"],
            "hidden_size": 8,
            "use_linear_output_projection": true
        }))
        .is_err());
    }

    #[test]
    fn actual_qwen3_reranker_metadata_resolves_h2560_score2() {
        let config = raw(json!({
            "architectures": ["Qwen3ForSequenceClassification"],
            "attention_bias": false,
            "attention_dropout": 0.0,
            "bos_token_id": 151643,
            "eos_token_id": 151645,
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 2560,
            "id2label": {"0": "no", "1": "yes"},
            "initializer_range": 0.02,
            "intermediate_size": 9728,
            "label2id": {"no": 0, "yes": 1},
            "max_position_embeddings": 40960,
            "max_window_layers": 36,
            "model_type": "qwen3",
            "num_attention_heads": 32,
            "num_hidden_layers": 36,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-06,
            "rope_scaling": null,
            "rope_theta": 1000000,
            "sliding_window": null,
            "tie_word_embeddings": true,
            "torch_dtype": "float16",
            "transformers_version": "4.52.4",
            "use_cache": true,
            "use_sliding_window": false,
            "vocab_size": 151669
        }))
        .unwrap();
        assert_eq!(config.task, DecoderClassifierTask::Sequence);
        assert_eq!(config.family, DecoderFamily::Qwen3);
        assert_eq!((config.hidden_size, config.num_labels), (2560, 2));
        assert_eq!(config.pad_token_id, None);
        assert_eq!(config.architecture(), "Qwen3ForSequenceClassification");
    }

    #[test]
    fn labels_and_token_bias_follow_native_agreement_rules() {
        assert_eq!(resolve_label_count(None, None).unwrap(), 2);
        assert_eq!(resolve_label_count(None, Some(1)).unwrap(), 1);
        assert_eq!(resolve_label_count(Some(1), Some(1)).unwrap(), 1);
        assert_eq!(resolve_label_count(Some(2), Some(2)).unwrap(), 2);
        assert_eq!(resolve_label_count(Some(9), None).unwrap(), 9);
        assert!(resolve_label_count(Some(2), Some(3)).is_err());
        assert!(resolve_label_count(None, Some(0)).is_err());
        assert!(resolve_label_count(Some(0), Some(0)).is_err());

        assert!(validate_score_bias_contract(DecoderClassifierTask::Tokens, true, true).is_ok());
        assert!(validate_score_bias_contract(DecoderClassifierTask::Tokens, true, false).is_err());
        assert!(validate_score_bias_contract(DecoderClassifierTask::Tokens, false, true).is_err());
        assert!(validate_score_bias_contract(DecoderClassifierTask::Tokens, false, false).is_ok());
        assert!(validate_score_bias_contract(DecoderClassifierTask::Sequence, true, false).is_ok());
    }

    #[test]
    fn readout_admission_and_conflict_names_match_native_contract() {
        assert!(admit_readout(
            DecoderClassifierTask::Tokens,
            DecoderClassifierReadout::Sequence
        )
        .is_ok());
        assert!(admit_readout(
            DecoderClassifierTask::Tokens,
            DecoderClassifierReadout::Tokens
        )
        .is_ok());
        assert!(admit_readout(
            DecoderClassifierTask::Sequence,
            DecoderClassifierReadout::Sequence
        )
        .is_ok());
        assert!(admit_readout(
            DecoderClassifierTask::Sequence,
            DecoderClassifierReadout::Tokens
        )
        .is_err());
        assert!(output_projection_conflict("linear.weight").is_some());
        assert!(output_projection_conflict("model.linear.weight").is_none());
        assert!(output_projection_conflict("score.weight").is_none());
    }

    #[test]
    fn scratch_layout_accounts_gather_bf16_logits_and_f32_readback() {
        let layout = DecoderClassifierScratchLayout::for_key(8, 2, (32, 4, 2)).unwrap();
        assert_eq!(layout.gather_elements(), 32);
        assert_eq!(layout.logits_elements(), 64);
        assert_eq!(layout.mean_elements(), 32);
        assert_eq!(layout.bytes().unwrap(), 32 * 2 + 64 * 2 + 64 * 4 + 32 * 4);
        assert!(DecoderClassifierScratchLayout::for_key(0, 2, (32, 4, 2)).is_err());
        assert!(DecoderClassifierScratchLayout::for_key(usize::MAX, 2, (32, 4, 2)).is_err());
        assert!(DecoderClassifierScratchLayout::for_key(8, 2, (0, 4, 2)).is_err());
        let plan = RowPlan::from_row_ids(32, vec![0, 3, 3]).unwrap();
        assert!(layout.fits(8, 2, &plan).is_ok());
        let too_many = RowPlan::from_row_ids(32, vec![0, 1, 2, 3, 4]).unwrap();
        assert!(layout.fits(8, 2, &too_many).is_err());
    }

    #[test]
    fn scratch_drop_plan_charges_partial_once_and_never_the_work_owner() {
        let owned = ScratchDropPlan::owned_by_failed_work();
        assert_eq!(owned.charge_bytes, 0);
        assert!(owned.forget_buffers);
        assert!(owned.forget_stream);

        let partial = ScratchDropPlan::unowned_failure(1234);
        assert_eq!(
            partial.charge_bytes, 1234,
            "partial leak bills exactly once"
        );
        assert!(partial.forget_buffers && partial.forget_stream);
        assert_eq!(
            ScratchDropPlan::for_scratch_drop(1234, true, false),
            ScratchDropPlan::released(),
            "a proved drain releases without synchronize or charge"
        );
        assert_eq!(
            ScratchDropPlan::for_scratch_drop(1234, false, true),
            ScratchDropPlan::released()
        );
        assert_eq!(
            ScratchDropPlan::for_scratch_drop(1234, false, false),
            partial
        );
    }

    #[test]
    fn cold_classifier_label_bias_and_dense_guards_are_explicit() {
        assert!(resolve_label_count(Some(2), Some(3)).is_err());
        assert!(resolve_label_count(None, Some(0)).is_err());
        assert!(validate_score_bias_contract(DecoderClassifierTask::Tokens, true, false).is_err());
        assert!(validate_score_bias_contract(DecoderClassifierTask::Tokens, false, true).is_err());
        assert!(classifier_dense_rejected(true, false));
        assert!(classifier_dense_rejected(false, true));
        assert!(!classifier_dense_rejected(false, false));
    }

    #[test]
    fn unsupported_native_classifier_architectures_fail_on_the_host() {
        for architecture in [
            "MistralForSequenceClassification",
            "MistralForTokenClassification",
            "Mistral3ForSequenceClassification",
            "Ministral3ForSequenceClassification",
            "Qwen3MoeForSequenceClassification",
            "Qwen3MoeForTokenClassification",
        ] {
            let value = json!({"model_type": "qwen3", "architectures": [architecture]});
            assert!(
                reject_unsupported_classifier_architecture(&value).is_err(),
                "{architecture} must fail explicitly"
            );
            assert!(!value_is_classifier_architecture(&value));
        }
        for architecture in [
            "Qwen2ForSequenceClassification",
            "Qwen2ForTokenClassification",
            "Qwen3ForSequenceClassification",
            "Qwen3ForTokenClassification",
            "LlamaForSequenceClassification",
            "LlamaForTokenClassification",
            "LlamaBidirectionalForSequenceClassification",
        ] {
            let value = json!({"model_type": "qwen3", "architectures": [architecture]});
            assert!(reject_unsupported_classifier_architecture(&value).is_ok());
            assert!(value_is_classifier_architecture(&value));
        }
        assert!(reject_unsupported_classifier_architecture(
            &json!({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]})
        )
        .is_ok());
    }

    #[test]
    fn write_row_ids_only_accepts_explicit_packed_rows() {
        let explicit = RowPlan::from_row_ids(8, vec![1, 3]).unwrap();
        let mut staged = [u32::MAX; 2];
        assert_eq!(explicit.write_row_ids(&mut staged).unwrap(), 2);
        assert_eq!(staged, [1, 3]);

        let selected = RowPlan::selected_sequence(2, 4).unwrap();
        assert!(selected.write_row_ids(&mut staged).is_err());
        let tokens = RowPlan::tokens(4).unwrap();
        assert!(tokens.write_row_ids(&mut staged).is_err());
        let unstaged =
            RowPlan::sequence_into(&[1, 2], &[0, 1, 2], None, &mut [u32::MAX; 2]).unwrap();
        assert!(unstaged.write_row_ids(&mut staged).is_err());
    }

    #[test]
    fn sequence_into_stages_the_same_last_nonpad_ids_without_allocating() {
        let ids = [5, 9, PAD, 7, PAD, PAD, 11, PAD, 13, PAD];
        let boundaries = [0, 4, 6, 10];
        let expected = RowPlan::sequence(&ids, &boundaries, Some(PAD)).unwrap();
        let mut staged = [u32::MAX; 3];
        let staged_plan =
            RowPlan::sequence_into(&ids, &boundaries, Some(PAD), &mut staged).unwrap();
        assert_eq!(staged, [3, 4, 8]);
        assert_eq!(staged_plan.row_ids(), &[] as &[u32]);
        assert_eq!(staged_plan.kind(), expected.kind());
        assert_eq!(staged_plan.readout(), expected.readout());
        assert_eq!(staged_plan.rows(), expected.rows());
        assert_eq!(staged_plan.source_rows(), expected.source_rows());
        assert!(RowPlan::sequence_into(&ids, &boundaries, Some(PAD), &mut [u32::MAX; 2]).is_err());
    }

    #[test]
    fn plan_aware_admission_matches_native_token_and_sequence_rules() {
        let selected = RowPlan::selected_sequence(3, 3).unwrap();
        let packed = RowPlan::from_row_ids(10, vec![1, 3, 9]).unwrap();
        let tokens = RowPlan::tokens(10).unwrap();
        assert!(
            admit_plan(DecoderClassifierTask::Sequence, &selected).is_ok(),
            "sequence head admits pre-selected rows"
        );
        assert!(admit_plan(DecoderClassifierTask::Sequence, &packed).is_ok());
        assert!(
            admit_plan(DecoderClassifierTask::Sequence, &tokens).is_err(),
            "sequence head rejects token readout"
        );
        assert!(
            admit_plan(DecoderClassifierTask::Tokens, &selected).is_err(),
            "token head sequence readout requires full T first"
        );
        assert!(admit_plan(DecoderClassifierTask::Tokens, &packed).is_ok());
        assert!(admit_plan(DecoderClassifierTask::Tokens, &tokens).is_ok());
        let mean = RowPlan::mean_sequence(&[1, 2, 3, 4, 5], &[0, 2, 5]).unwrap();
        assert_eq!(mean.kind(), RowPlanKind::MeanSequence);
        assert_eq!((mean.rows(), mean.source_rows()), (2, 5));
        assert!(mean.requires_mean() && !mean.requires_gather());
        assert!(admit_plan(DecoderClassifierTask::BidirectionalSequence, &mean).is_ok());
        assert!(admit_plan(DecoderClassifierTask::Sequence, &mean).is_err());
        assert!(admit_plan(DecoderClassifierTask::BidirectionalSequence, &tokens).is_err());
        assert_eq!(tokens.source_rows(), tokens.rows());
        assert!(
            tokens.row_ids().is_empty(),
            "token source mode stages zero selection ids"
        );
    }

    #[test]
    fn geometry_keeps_projection_rows_distinct_from_hidden_width() {
        // D = labels may be narrower or wider than H; the gather is sized by
        // H and the logits by D independently.
        let layout = DecoderClassifierScratchLayout::for_key(12, 3, (32, 4, 2)).unwrap();
        assert_eq!((layout.hidden(), layout.labels()), (12, 3));
        assert_eq!(layout.gather_elements(), 4 * 12);
        assert_eq!(layout.logits_elements(), 32 * 3);
        assert!(DecoderClassifierScratchLayout::for_key(usize::MAX, 3, (32, 4, 2)).is_err());
        assert!(DecoderClassifierScratchLayout::for_key(12, usize::MAX, (32, 4, 2)).is_err());
    }

    #[test]
    fn null_quantization_is_not_a_quantized_checkpoint() {
        let mut value = config("Qwen3ForSequenceClassification");
        value["quantization_config"] = serde_json::Value::Null;
        assert!(raw(value).is_ok());
        let mut value = config("Qwen3ForSequenceClassification");
        value["quantization_config"] = json!({"quant_method": "gptq"});
        assert!(raw(value).is_err());
    }
}
