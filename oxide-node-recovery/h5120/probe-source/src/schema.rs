use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::ProbeError;

pub const MAX_REQUEST_BYTES: u64 = 32 * 1024 * 1024;
pub const MAX_SEQUENCES: usize = 65_535;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BatchFile {
    pub input_ids: Vec<u32>,
    pub position_ids: Vec<u32>,
    pub token_type_ids: Vec<u32>,
    pub cumulative_seq_lengths: Vec<u32>,
    pub pooled_indices: Vec<u32>,
    pub raw_indices: Vec<u32>,
    pub max_length: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Operation {
    Embed,
    Predict,
    PredictTokens,
    ExpectRejectPredictTokens,
}

impl Operation {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Embed => "embed",
            Self::Predict => "predict",
            Self::PredictTokens => "predict_tokens",
            Self::ExpectRejectPredictTokens => "expect_reject_predict_tokens",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequestLine {
    pub id: String,
    pub operation: Operation,
    pub batch: BatchFile,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Task {
    Embedding,
    Classifier,
}

impl Task {
    pub fn parse(value: &str) -> Result<Self, ProbeError> {
        match value {
            "embedding" => Ok(Self::Embedding),
            "classifier" => Ok(Self::Classifier),
            other => Err(ProbeError::Cli(format!(
                "task must be embedding or classifier, got {other}"
            ))),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Embedding => "embedding",
            Self::Classifier => "classifier",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HeadKind {
    CausalLm,
    SequenceClassification,
    TokenClassification,
    Unknown,
}

impl HeadKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CausalLm => "causal_lm",
            Self::SequenceClassification => "sequence_classification",
            Self::TokenClassification => "token_classification",
            Self::Unknown => "unknown",
        }
    }

    pub const fn is_classifier(self) -> bool {
        matches!(
            self,
            Self::SequenceClassification | Self::TokenClassification
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Api {
    Factory,
    Core,
}

impl Api {
    pub fn parse(value: &str) -> Result<Self, ProbeError> {
        match value {
            "factory" => Ok(Self::Factory),
            "core" => Ok(Self::Core),
            other => Err(ProbeError::Cli(format!(
                "api must be factory or core, got {other}"
            ))),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Factory => "factory",
            Self::Core => "core",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum PoolArg {
    Cls,
    Mean,
    LastToken,
}

impl PoolArg {
    pub fn parse(value: &str) -> Result<Self, ProbeError> {
        match value {
            "cls" => Ok(Self::Cls),
            "mean" => Ok(Self::Mean),
            "last-token" => Ok(Self::LastToken),
            other => Err(ProbeError::Cli(format!(
                "pool must be cls, mean, or last-token, got {other}"
            ))),
        }
    }

    pub fn backend_pool(self) -> infinity_v3_backend_core::Pool {
        match self {
            Self::Cls => infinity_v3_backend_core::Pool::Cls,
            Self::Mean => infinity_v3_backend_core::Pool::Mean,
            Self::LastToken => infinity_v3_backend_core::Pool::LastToken,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ModelBounds {
    pub vocab_size: Option<usize>,
    pub max_position_embeddings: Option<usize>,
    pub decoder_family: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedBatch {
    pub file: BatchFile,
    pub batch_count: usize,
    pub sequence_lengths: Vec<usize>,
}

pub fn parse_request_line(bytes: &[u8]) -> Result<RequestLine, ProbeError> {
    serde_json::from_slice(bytes).map_err(ProbeError::Json)
}

pub fn validate_task_model(task: Task, head: HeadKind) -> Result<(), ProbeError> {
    match (task, head) {
        (Task::Embedding, HeadKind::SequenceClassification | HeadKind::TokenClassification) => {
            Err(ProbeError::Config(format!(
                "ordinary embedding admission rejects classifier architecture head {}",
                head.as_str()
            )))
        }
        (Task::Classifier, HeadKind::CausalLm | HeadKind::Unknown) => Err(ProbeError::Config(
            format!(
                "classifier task requires a decoder sequence/token classification architecture, got head {}",
                head.as_str()
            ),
        )),
        _ => Ok(()),
    }
}

pub fn validate_batch(file: BatchFile, bounds: &ModelBounds) -> Result<ValidatedBatch, ProbeError> {
    let token_count = file.input_ids.len();
    if token_count == 0 {
        return Err(ProbeError::Batch("input_ids must not be empty".into()));
    }
    if file.position_ids.len() != token_count {
        return Err(ProbeError::Batch(format!(
            "position_ids length {} does not match input_ids length {token_count}",
            file.position_ids.len()
        )));
    }
    if file.token_type_ids.len() != token_count {
        return Err(ProbeError::Batch(format!(
            "token_type_ids length {} does not match input_ids length {token_count}",
            file.token_type_ids.len()
        )));
    }

    let cumulative = &file.cumulative_seq_lengths;
    if cumulative.len() < 2 {
        return Err(ProbeError::Batch(
            "cumulative_seq_lengths must contain at least start and end".into(),
        ));
    }
    if cumulative.first().copied() != Some(0) {
        return Err(ProbeError::Batch(
            "cumulative_seq_lengths must start at 0".into(),
        ));
    }
    if cumulative.len() - 1 > MAX_SEQUENCES {
        return Err(ProbeError::Batch(format!(
            "batch contains {} sequences, exceeding the {MAX_SEQUENCES} sequence limit",
            cumulative.len() - 1
        )));
    }
    let last = cumulative
        .last()
        .copied()
        .ok_or_else(|| ProbeError::Batch("cumulative_seq_lengths is empty".into()))?;
    if last as usize != token_count {
        return Err(ProbeError::Batch(format!(
            "cumulative_seq_lengths ends at {last}, expected {token_count}"
        )));
    }
    let mut sequence_lengths = Vec::with_capacity(cumulative.len().saturating_sub(1));
    let mut previous: Option<u32> = None;
    for &boundary in cumulative {
        if let Some(prev) = previous {
            if boundary <= prev {
                return Err(ProbeError::Batch(
                    "cumulative_seq_lengths must strictly increase".into(),
                ));
            }
            sequence_lengths.push((boundary - prev) as usize);
        }
        previous = Some(boundary);
    }
    let batch_count = sequence_lengths.len();
    if batch_count == 0 {
        return Err(ProbeError::Batch("batch must contain a sequence".into()));
    }
    let actual_max = sequence_lengths
        .iter()
        .copied()
        .max()
        .ok_or_else(|| ProbeError::Batch("batch has no sequence lengths".into()))?;
    let actual_max = u32::try_from(actual_max)
        .map_err(|_| ProbeError::Batch("max length overflows u32".into()))?;
    if file.max_length != actual_max {
        return Err(ProbeError::Batch(format!(
            "max_length {} does not match actual maximum sequence length {actual_max}",
            file.max_length
        )));
    }
    for &index in &file.pooled_indices {
        if index as usize >= batch_count {
            return Err(ProbeError::Batch(format!(
                "pooled index {index} is out of range for {batch_count} sequences"
            )));
        }
    }
    for &index in &file.raw_indices {
        if index as usize >= batch_count {
            return Err(ProbeError::Batch(format!(
                "raw index {index} is out of range for {batch_count} sequences"
            )));
        }
    }
    if let Some(vocab_size) = bounds.vocab_size {
        if let Some((position, token)) = file
            .input_ids
            .iter()
            .copied()
            .enumerate()
            .find(|(_, token)| *token as usize >= vocab_size)
        {
            return Err(ProbeError::Batch(format!(
                "input_ids[{position}]={token} is outside vocab_size {vocab_size}"
            )));
        }
    }
    if let Some(limit) = bounds.max_position_embeddings {
        if let Some((position, value)) = file
            .position_ids
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| *value as usize >= limit)
        {
            return Err(ProbeError::Batch(format!(
                "position_ids[{position}]={value} is outside max_position_embeddings {limit}"
            )));
        }
        if let Some((index, length)) = sequence_lengths
            .iter()
            .copied()
            .enumerate()
            .find(|(_, length)| *length > limit)
        {
            return Err(ProbeError::Batch(format!(
                "sequence {index} length {length} exceeds max_position_embeddings {limit}"
            )));
        }
    }
    if bounds.decoder_family {
        if let Some((position, value)) = file
            .token_type_ids
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| *value != 0)
        {
            return Err(ProbeError::Batch(format!(
                "decoder token_type_ids[{position}]={value} must be zero"
            )));
        }
    }
    Ok(ValidatedBatch {
        file,
        batch_count,
        sequence_lengths,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionMode {
    LastNonPadOrFirstAllPad,
    EndRow,
    AllTokens,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SelectionPlan {
    pub mode: SelectionMode,
    pub pad_token_id: Option<u32>,
    pub sequences: usize,
    pub source_rows: usize,
    pub selected_row_ids: Vec<u32>,
    pub all_pad_sequences: Vec<usize>,
}

pub fn sequence_selection(
    validated: &ValidatedBatch,
    pad_token_id: Option<u32>,
) -> Result<SelectionPlan, ProbeError> {
    let mut selected_row_ids = Vec::with_capacity(validated.batch_count);
    let mut all_pad_sequences = Vec::new();
    let mut start = 0usize;
    for (sequence, &length) in validated.sequence_lengths.iter().enumerate() {
        let end = start
            .checked_add(length)
            .ok_or_else(|| ProbeError::Layout("sequence boundary overflow".into()))?;
        let rows = validated
            .file
            .input_ids
            .get(start..end)
            .ok_or_else(|| ProbeError::Layout("sequence boundary exceeds input_ids".into()))?;
        let row = match pad_token_id {
            None => end - 1,
            Some(pad) => match rows.iter().rposition(|token| *token != pad) {
                Some(offset) => start + offset,
                None => {
                    all_pad_sequences.push(sequence);
                    start
                }
            },
        };
        let row = u32::try_from(row)
            .map_err(|_| ProbeError::Layout("selected row exceeds u32".into()))?;
        selected_row_ids.push(row);
        start = end;
    }
    if start != validated.file.input_ids.len() {
        return Err(ProbeError::Layout(
            "selection plan did not consume every packed token".into(),
        ));
    }
    Ok(SelectionPlan {
        mode: if pad_token_id.is_some() {
            SelectionMode::LastNonPadOrFirstAllPad
        } else {
            SelectionMode::EndRow
        },
        pad_token_id,
        sequences: validated.batch_count,
        source_rows: validated.file.input_ids.len(),
        selected_row_ids,
        all_pad_sequences,
    })
}

pub fn token_selection(validated: &ValidatedBatch) -> SelectionPlan {
    SelectionPlan {
        mode: SelectionMode::AllTokens,
        pad_token_id: None,
        sequences: validated.batch_count,
        source_rows: validated.file.input_ids.len(),
        selected_row_ids: (0..validated.file.input_ids.len())
            .map(|row| row as u32)
            .collect(),
        all_pad_sequences: Vec::new(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpectedKind {
    Pooled,
    Raw,
}

impl ExpectedKind {
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Pooled => "pooled",
            Self::Raw => "raw",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpectedEntry {
    pub kind: ExpectedKind,
    pub rows: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpectedCanonical {
    pub entries: BTreeMap<usize, ExpectedEntry>,
    pub raw_overrides_pooled: Vec<usize>,
    pub duplicate_selection_indices: Vec<usize>,
}

pub fn expected_canonical(validated: &ValidatedBatch) -> Result<ExpectedCanonical, ProbeError> {
    let mut entries = BTreeMap::new();
    let mut seen_pooled = BTreeSet::new();
    let mut seen_raw = BTreeSet::new();
    let mut duplicates = BTreeSet::new();
    let mut overrides = BTreeSet::new();
    for &index in &validated.file.pooled_indices {
        let key = index as usize;
        if !seen_pooled.insert(key) {
            duplicates.insert(key);
        }
        entries.insert(
            key,
            ExpectedEntry {
                kind: ExpectedKind::Pooled,
                rows: 1,
            },
        );
    }
    for &index in &validated.file.raw_indices {
        let key = index as usize;
        if !seen_raw.insert(key) {
            duplicates.insert(key);
        }
        if matches!(entries.get(&key), Some(entry) if entry.kind == ExpectedKind::Pooled) {
            overrides.insert(key);
        }
        let rows = validated
            .sequence_lengths
            .get(key)
            .copied()
            .ok_or_else(|| ProbeError::Layout(format!("selection index {key} is out of range")))?;
        entries.insert(
            key,
            ExpectedEntry {
                kind: ExpectedKind::Raw,
                rows,
            },
        );
    }
    Ok(ExpectedCanonical {
        entries,
        raw_overrides_pooled: overrides.into_iter().collect(),
        duplicate_selection_indices: duplicates.into_iter().collect(),
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExpectedEntryView {
    pub index: usize,
    pub kind: String,
    pub rows: usize,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CanonicalSummary {
    pub entry_count: usize,
    pub float_count: Option<usize>,
    pub raw_overrides_pooled: Vec<usize>,
    pub duplicate_selection_indices: Vec<usize>,
    pub entries: Vec<ExpectedEntryView>,
}

pub fn canonical_summary(
    expected: &ExpectedCanonical,
    width: Option<usize>,
) -> Result<CanonicalSummary, ProbeError> {
    let mut float_count = width.map(|_| 0usize);
    let mut entries = Vec::with_capacity(expected.entries.len());
    for (&index, entry) in &expected.entries {
        let shape = match entry.kind {
            ExpectedKind::Pooled => match width {
                Some(width) => vec![width],
                None => vec![1],
            },
            ExpectedKind::Raw => match width {
                Some(width) => vec![entry.rows, width],
                None => vec![entry.rows],
            },
        };
        if let Some(total) = float_count.as_mut() {
            let length = match entry.kind {
                ExpectedKind::Pooled => width.unwrap_or(0),
                ExpectedKind::Raw => width
                    .ok_or_else(|| ProbeError::Layout("missing output width".into()))?
                    .checked_mul(entry.rows)
                    .ok_or_else(|| {
                        ProbeError::Layout("canonical raw float count overflow".into())
                    })?,
            };
            *total = total
                .checked_add(length)
                .ok_or_else(|| ProbeError::Layout("canonical float count overflow".into()))?;
        }
        entries.push(ExpectedEntryView {
            index,
            kind: entry.kind.as_str().into(),
            rows: entry.rows,
            shape,
        });
    }
    Ok(CanonicalSummary {
        entry_count: expected.entries.len(),
        float_count,
        raw_overrides_pooled: expected.raw_overrides_pooled.clone(),
        duplicate_selection_indices: expected.duplicate_selection_indices.clone(),
        entries,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SegmentReport {
    pub kind: String,
    pub index: u32,
    pub shape: Vec<usize>,
    pub offset_floats: usize,
    pub length_floats: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LayoutReport {
    pub pooled_count: usize,
    pub raw_count: usize,
    pub raw_token_rows: usize,
    pub physical_float_count: usize,
    pub physical_bytes: usize,
    pub segments: Vec<SegmentReport>,
}

pub fn build_layout(validated: &ValidatedBatch, width: usize) -> Result<LayoutReport, ProbeError> {
    if width == 0 {
        return Err(ProbeError::Layout(
            "hidden width must be greater than zero".into(),
        ));
    }
    let mut segments = Vec::new();
    let mut offset = 0usize;
    for &index in &validated.file.pooled_indices {
        segments.push(SegmentReport {
            kind: "pooled".into(),
            index,
            shape: vec![width],
            offset_floats: offset,
            length_floats: width,
        });
        offset = offset
            .checked_add(width)
            .ok_or_else(|| ProbeError::Layout("pooled float offset overflow".into()))?;
    }
    let pooled_count = segments.len();
    let mut raw_token_rows = 0usize;
    for &index in &validated.file.raw_indices {
        let rows = validated
            .sequence_lengths
            .get(index as usize)
            .copied()
            .ok_or_else(|| {
                ProbeError::Layout(format!("selection index {index} is out of range"))
            })?;
        let length = width
            .checked_mul(rows)
            .ok_or_else(|| ProbeError::Layout("raw segment length overflow".into()))?;
        raw_token_rows = raw_token_rows
            .checked_add(rows)
            .ok_or_else(|| ProbeError::Layout("raw token row overflow".into()))?;
        segments.push(SegmentReport {
            kind: "raw".into(),
            index,
            shape: vec![rows, width],
            offset_floats: offset,
            length_floats: length,
        });
        offset = offset
            .checked_add(length)
            .ok_or_else(|| ProbeError::Layout("raw float offset overflow".into()))?;
    }
    let physical_bytes = offset
        .checked_mul(4)
        .ok_or_else(|| ProbeError::Layout("physical byte count overflow".into()))?;
    Ok(LayoutReport {
        pooled_count,
        raw_count: validated.file.raw_indices.len(),
        raw_token_rows,
        physical_float_count: offset,
        physical_bytes,
        segments,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BatchSummary {
    pub token_count: usize,
    pub batch_count: usize,
    pub sequence_lengths: Vec<usize>,
    pub cumulative_seq_lengths: Vec<u32>,
    pub max_length: u32,
    pub pooled_indices: Vec<u32>,
    pub raw_indices: Vec<u32>,
}

impl BatchSummary {
    pub fn new(validated: &ValidatedBatch) -> Self {
        Self {
            token_count: validated.file.input_ids.len(),
            batch_count: validated.batch_count,
            sequence_lengths: validated.sequence_lengths.clone(),
            cumulative_seq_lengths: validated.file.cumulative_seq_lengths.clone(),
            max_length: validated.file.max_length,
            pooled_indices: validated.file.pooled_indices.clone(),
            raw_indices: validated.file.raw_indices.clone(),
        }
    }
}

pub fn validate_operation(
    task: Task,
    head: HeadKind,
    request: &RequestLine,
) -> Result<(), ProbeError> {
    let supported = matches!(
        (task, head, request.operation),
        (Task::Embedding, _, Operation::Embed)
            | (Task::Embedding, _, Operation::ExpectRejectPredictTokens)
            | (
                Task::Classifier,
                HeadKind::SequenceClassification,
                Operation::Predict
            )
            | (
                Task::Classifier,
                HeadKind::SequenceClassification,
                Operation::ExpectRejectPredictTokens
            )
            | (
                Task::Classifier,
                HeadKind::TokenClassification,
                Operation::Predict
            )
            | (
                Task::Classifier,
                HeadKind::TokenClassification,
                Operation::PredictTokens
            )
    );
    if supported {
        return Ok(());
    }
    let reason = match (task, head, request.operation) {
        (
            Task::Classifier,
            HeadKind::SequenceClassification,
            Operation::PredictTokens,
        ) => "decoder sequence classification head rejects Native token readout; use expect_reject_predict_tokens",
        (
            Task::Classifier,
            HeadKind::TokenClassification,
            Operation::ExpectRejectPredictTokens,
        ) => "token classification head implements token readout and must not expect a rejection",
        _ => "operation is not supported by this task/head combination",
    };
    Err(ProbeError::Operation(format!(
        "case {} operation {} is not supported by task {} head {}: {reason}",
        request.id,
        request.operation.as_str(),
        task.as_str(),
        head.as_str()
    )))
}

impl fmt::Display for Task {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds() -> ModelBounds {
        ModelBounds {
            vocab_size: Some(1024),
            max_position_embeddings: Some(128),
            decoder_family: true,
        }
    }

    fn batch() -> BatchFile {
        BatchFile {
            input_ids: vec![1, 2, 3],
            position_ids: vec![0, 1, 2],
            token_type_ids: vec![0, 0, 0],
            cumulative_seq_lengths: vec![0, 1, 3],
            pooled_indices: vec![0, 1],
            raw_indices: vec![1],
            max_length: 2,
        }
    }

    fn request(operation: Operation) -> RequestLine {
        RequestLine {
            id: "case".into(),
            operation,
            batch: batch(),
        }
    }

    #[test]
    fn exact_seven_field_batch_is_accepted() {
        let validated = validate_batch(batch(), &bounds()).expect("batch");
        assert_eq!(validated.batch_count, 2);
        assert_eq!(validated.sequence_lengths, vec![1, 2]);
    }

    #[test]
    fn malformed_cumulative_is_rejected() {
        for cumulative in [vec![0], vec![1, 3], vec![0, 3, 3], vec![0, 2, 4]] {
            let mut file = batch();
            file.cumulative_seq_lengths = cumulative;
            assert!(validate_batch(file, &bounds()).is_err());
        }
    }

    #[test]
    fn selection_bounds_and_max_length_are_rejected() {
        let mut file = batch();
        file.pooled_indices = vec![2];
        assert!(validate_batch(file, &bounds()).is_err());

        let mut file = batch();
        file.raw_indices = vec![9];
        assert!(validate_batch(file, &bounds()).is_err());

        let mut file = batch();
        file.max_length = 3;
        assert!(validate_batch(file, &bounds()).is_err());
    }

    #[test]
    fn token_and_position_bounds_match_decoder_validation() {
        let mut file = batch();
        file.input_ids = vec![1, 2, 2048];
        assert!(validate_batch(file, &bounds()).is_err());

        let mut file = batch();
        file.position_ids = vec![0, 1, 128];
        assert!(validate_batch(file, &bounds()).is_err());

        let mut file = batch();
        file.token_type_ids = vec![0, 1, 0];
        assert!(validate_batch(file, &bounds()).is_err());
    }

    #[test]
    fn selection_plan_selects_last_nonpad_internal_pad_and_allpad_first() {
        let file = BatchFile {
            input_ids: vec![5, 9, 0, 7, 0, 0, 11, 0, 13, 0],
            position_ids: vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 3],
            token_type_ids: vec![0; 10],
            cumulative_seq_lengths: vec![0, 4, 6, 10],
            pooled_indices: vec![0, 1, 2],
            raw_indices: vec![],
            max_length: 4,
        };
        let validated = validate_batch(file, &bounds()).expect("batch");
        let plan = sequence_selection(&validated, Some(0)).expect("selection");
        assert_eq!(plan.mode, SelectionMode::LastNonPadOrFirstAllPad);
        assert_eq!(plan.selected_row_ids, vec![3, 4, 8]);
        assert_eq!(plan.all_pad_sequences, vec![1]);
    }

    #[test]
    fn selection_plan_uses_first_row_for_an_all_pad_sequence() {
        let file = BatchFile {
            input_ids: vec![0, 0, 0, 7, 0],
            position_ids: vec![0, 1, 2, 0, 1],
            token_type_ids: vec![0; 5],
            cumulative_seq_lengths: vec![0, 3, 5],
            pooled_indices: vec![0, 1],
            raw_indices: vec![],
            max_length: 3,
        };
        let validated = validate_batch(file, &bounds()).expect("batch");
        let plan = sequence_selection(&validated, Some(0)).expect("selection");
        assert_eq!(plan.selected_row_ids, vec![0, 3]);
        assert_eq!(plan.all_pad_sequences, vec![0]);
    }

    #[test]
    fn selection_plan_without_pad_uses_end_rows() {
        let file = BatchFile {
            input_ids: vec![7, 8, 9, 10, 11, 12, 13, 14],
            position_ids: vec![0, 1, 2, 0, 1, 2, 3, 0],
            token_type_ids: vec![0; 8],
            cumulative_seq_lengths: vec![0, 1, 6, 8],
            pooled_indices: vec![0, 1, 2],
            raw_indices: vec![],
            max_length: 5,
        };
        let validated = validate_batch(file, &bounds()).expect("batch");
        let plan = sequence_selection(&validated, None).expect("selection");
        assert_eq!(plan.mode, SelectionMode::EndRow);
        assert_eq!(plan.selected_row_ids, vec![0, 5, 7]);
        assert!(plan.all_pad_sequences.is_empty());
    }

    #[test]
    fn token_selection_projects_every_packed_row() {
        let validated = validate_batch(batch(), &bounds()).expect("batch");
        let plan = token_selection(&validated);
        assert_eq!(plan.mode, SelectionMode::AllTokens);
        assert_eq!(plan.selected_row_ids, vec![0, 1, 2]);
    }

    #[test]
    fn raw_selection_overrides_pooled_index() {
        let validated = validate_batch(batch(), &bounds()).expect("batch");
        let expected = expected_canonical(&validated).expect("canonical");
        assert_eq!(expected.entries.len(), 2);
        assert_eq!(
            expected.entries.get(&1).map(|entry| entry.kind),
            Some(ExpectedKind::Raw)
        );
        assert_eq!(expected.raw_overrides_pooled, vec![1]);
        assert_eq!(expected.duplicate_selection_indices, Vec::<usize>::new());
    }

    #[test]
    fn task_operation_table_is_fail_closed() {
        assert!(validate_operation(
            Task::Embedding,
            HeadKind::CausalLm,
            &request(Operation::Embed)
        )
        .is_ok());
        assert!(validate_operation(
            Task::Embedding,
            HeadKind::CausalLm,
            &request(Operation::Predict)
        )
        .is_err());
        assert!(validate_operation(
            Task::Classifier,
            HeadKind::SequenceClassification,
            &request(Operation::Predict)
        )
        .is_ok());
        assert!(validate_operation(
            Task::Classifier,
            HeadKind::SequenceClassification,
            &request(Operation::PredictTokens)
        )
        .is_err());
        assert!(validate_operation(
            Task::Classifier,
            HeadKind::SequenceClassification,
            &request(Operation::ExpectRejectPredictTokens)
        )
        .is_ok());
        assert!(validate_operation(
            Task::Classifier,
            HeadKind::TokenClassification,
            &request(Operation::PredictTokens)
        )
        .is_ok());
        assert!(validate_operation(
            Task::Classifier,
            HeadKind::TokenClassification,
            &request(Operation::ExpectRejectPredictTokens)
        )
        .is_err());
    }

    #[test]
    fn request_line_rejects_unknown_fields() {
        let bytes = br#"{"id":"a","operation":"embed","batch":{"input_ids":[1],"position_ids":[0],"token_type_ids":[0],"cumulative_seq_lengths":[0,1],"pooled_indices":[0],"raw_indices":[],"max_length":1,"extra":1}}"#;
        assert!(parse_request_line(bytes).is_err());
    }
}
