use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use infinity_v3_backend_core::{Batch, Embedding, Embeddings, Pool};

pub const MAX_BATCH_BYTES: u64 = 32 * 1024 * 1024;
pub const SCHEMA: &str = "infinity-rust-raw-probe/v1";
pub const BACKEND_NAME: &str = "rust-oxide-backend";

#[derive(Debug)]
pub enum ProbeError {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Json(serde_json::Error),
    Batch(String),
    Layout(String),
    Embedding(String),
    Config(String),
    Cli(String),
    Backend(String),
}

impl fmt::Display for ProbeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "I/O error for {}: {source}", path.display()),
            Self::Json(err) => write!(f, "invalid JSON: {err}"),
            Self::Batch(message) => write!(f, "invalid batch: {message}"),
            Self::Layout(message) => write!(f, "invalid layout: {message}"),
            Self::Embedding(message) => write!(f, "invalid embedding: {message}"),
            Self::Config(message) => write!(f, "invalid model config: {message}"),
            Self::Cli(message) => write!(f, "invalid command line: {message}"),
            Self::Backend(message) => write!(f, "backend failure: {message}"),
        }
    }
}

impl std::error::Error for ProbeError {}

impl From<serde_json::Error> for ProbeError {
    fn from(err: serde_json::Error) -> Self {
        Self::Json(err)
    }
}

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedBatch {
    pub file: BatchFile,
    pub batch_count: usize,
    pub sequence_lengths: Vec<usize>,
}

pub fn parse_batch_bytes(bytes: &[u8]) -> Result<BatchFile, ProbeError> {
    if bytes.len() as u64 > MAX_BATCH_BYTES {
        return Err(ProbeError::Batch(format!(
            "batch file exceeds the {MAX_BATCH_BYTES} byte cap"
        )));
    }
    serde_json::from_slice(bytes).map_err(ProbeError::Json)
}

pub fn read_batch_bytes(path: &Path) -> Result<Vec<u8>, ProbeError> {
    let file = File::open(path).map_err(|source| ProbeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut bytes = Vec::new();
    file.take(MAX_BATCH_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|source| ProbeError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    if bytes.len() as u64 > MAX_BATCH_BYTES {
        return Err(ProbeError::Batch(format!(
            "batch file exceeds the {MAX_BATCH_BYTES} byte cap"
        )));
    }
    Ok(bytes)
}

fn checked_u32_len(value: usize, field: &str) -> Result<u32, ProbeError> {
    u32::try_from(value).map_err(|_| ProbeError::Batch(format!("{field} does not fit in u32")))
}

pub fn validate_batch(file: BatchFile) -> Result<ValidatedBatch, ProbeError> {
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
    let actual_max = checked_u32_len(actual_max, "maximum sequence length")?;
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
    Ok(ValidatedBatch {
        file,
        batch_count,
        sequence_lengths,
    })
}

pub fn parse_and_validate(bytes: &[u8]) -> Result<ValidatedBatch, ProbeError> {
    validate_batch(parse_batch_bytes(bytes)?)
}

pub fn build_batch(validated: &ValidatedBatch) -> Batch {
    Batch {
        multimodal: Vec::new(),
        input_ids: validated.file.input_ids.clone(),
        token_type_ids: validated.file.token_type_ids.clone(),
        position_ids: validated.file.position_ids.clone(),
        cumulative_seq_lengths: validated.file.cumulative_seq_lengths.clone(),
        max_length: validated.file.max_length,
        pooled_indices: validated.file.pooled_indices.clone(),
        raw_indices: validated.file.raw_indices.clone(),
        compact_input_ids: None,
        compact_position_ids: None,
        scatter_unfold: None,
        fold_gather: None,
        tokens: Vec::new(),
        offsets: Vec::new(),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Segment {
    pub kind: String,
    pub index: u32,
    pub shape: Vec<usize>,
    pub offset_floats: usize,
    pub length_floats: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Layout {
    pub pooled_count: usize,
    pub raw_count: usize,
    pub raw_token_rows: usize,
    pub physical_float_count: usize,
    pub physical_bytes: usize,
    pub segments: Vec<Segment>,
}

fn sequence_length(validated: &ValidatedBatch, index: u32) -> Result<usize, ProbeError> {
    validated
        .sequence_lengths
        .get(index as usize)
        .copied()
        .ok_or_else(|| ProbeError::Layout(format!("selection index {index} is out of range")))
}

pub fn build_layout(validated: &ValidatedBatch, width: usize) -> Result<Layout, ProbeError> {
    if width == 0 {
        return Err(ProbeError::Layout(
            "hidden width must be greater than zero".into(),
        ));
    }
    let mut segments = Vec::new();
    let mut offset = 0usize;
    for &index in &validated.file.pooled_indices {
        let length = width;
        segments.push(Segment {
            kind: "pooled".into(),
            index,
            shape: vec![width],
            offset_floats: offset,
            length_floats: length,
        });
        offset = offset
            .checked_add(length)
            .ok_or_else(|| ProbeError::Layout("pooled float offset overflow".into()))?;
    }
    let pooled_count = segments.len();
    let mut raw_token_rows = 0usize;
    for &index in &validated.file.raw_indices {
        let rows = sequence_length(validated, index)?;
        let length = width
            .checked_mul(rows)
            .ok_or_else(|| ProbeError::Layout("raw segment length overflow".into()))?;
        raw_token_rows = raw_token_rows
            .checked_add(rows)
            .ok_or_else(|| ProbeError::Layout("raw token row overflow".into()))?;
        segments.push(Segment {
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
    Ok(Layout {
        pooled_count,
        raw_count: validated.file.raw_indices.len(),
        raw_token_rows,
        physical_float_count: offset,
        physical_bytes,
        segments,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpectedKind {
    Pooled,
    Raw,
}

impl ExpectedKind {
    pub fn as_str(self) -> &'static str {
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
        entries.insert(
            key,
            ExpectedEntry {
                kind: ExpectedKind::Raw,
                rows: sequence_length(validated, index)?,
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
pub struct CanonicalReport {
    pub entry_count: usize,
    pub float_count: usize,
    pub raw_overrides_pooled: Vec<usize>,
    pub duplicate_selection_indices: Vec<usize>,
}

pub fn canonical_report(
    expected: &ExpectedCanonical,
    width: usize,
) -> Result<CanonicalReport, ProbeError> {
    let mut float_count = 0usize;
    for entry in expected.entries.values() {
        let length = match entry.kind {
            ExpectedKind::Pooled => width,
            ExpectedKind::Raw => width
                .checked_mul(entry.rows)
                .ok_or_else(|| ProbeError::Layout("canonical raw float count overflow".into()))?,
        };
        float_count = float_count
            .checked_add(length)
            .ok_or_else(|| ProbeError::Layout("canonical float count overflow".into()))?;
    }
    Ok(CanonicalReport {
        entry_count: expected.entries.len(),
        float_count,
        raw_overrides_pooled: expected.raw_overrides_pooled.clone(),
        duplicate_selection_indices: expected.duplicate_selection_indices.clone(),
    })
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OutputEntry {
    pub index: usize,
    pub kind: String,
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

fn ensure_finite(values: &[f32], index: usize) -> Result<(), ProbeError> {
    if let Some(position) = values.iter().position(|value| !value.is_finite()) {
        return Err(ProbeError::Embedding(format!(
            "output index {index} contains a non-finite value at flat position {position}"
        )));
    }
    Ok(())
}

pub fn validate_embeddings(
    expected: &ExpectedCanonical,
    embeddings: &Embeddings,
    width: usize,
) -> Result<Vec<OutputEntry>, ProbeError> {
    if width == 0 {
        return Err(ProbeError::Embedding(
            "hidden width must be greater than zero".into(),
        ));
    }
    if embeddings.len() != expected.entries.len() {
        return Err(ProbeError::Embedding(format!(
            "backend returned {} entries, expected {}",
            embeddings.len(),
            expected.entries.len()
        )));
    }
    let mut outputs = Vec::with_capacity(expected.entries.len());
    for (&index, wanted) in &expected.entries {
        let actual = embeddings.get(&index).ok_or_else(|| {
            ProbeError::Embedding(format!("backend output is missing expected index {index}"))
        })?;
        match (wanted.kind, actual) {
            (ExpectedKind::Pooled, Embedding::Pooled(values)) => {
                if values.len() != width {
                    return Err(ProbeError::Embedding(format!(
                        "pooled index {index} has width {}, expected {width}",
                        values.len()
                    )));
                }
                ensure_finite(values, index)?;
                outputs.push(OutputEntry {
                    index,
                    kind: wanted.kind.as_str().into(),
                    shape: vec![width],
                    values: values.clone(),
                });
            }
            (ExpectedKind::Raw, Embedding::All(rows)) => {
                if rows.len() != wanted.rows {
                    return Err(ProbeError::Embedding(format!(
                        "raw index {index} has {} rows, expected {}",
                        rows.len(),
                        wanted.rows
                    )));
                }
                for (row_index, row) in rows.iter().enumerate() {
                    if row.len() != width {
                        return Err(ProbeError::Embedding(format!(
                            "raw index {index} row {row_index} has width {}, expected {width}",
                            row.len()
                        )));
                    }
                    ensure_finite(row, index)?;
                }
                let flat_len = width.checked_mul(wanted.rows).ok_or_else(|| {
                    ProbeError::Embedding("raw output float count overflow".into())
                })?;
                let mut values = Vec::with_capacity(flat_len);
                for row in rows {
                    values.extend_from_slice(row);
                }
                outputs.push(OutputEntry {
                    index,
                    kind: wanted.kind.as_str().into(),
                    shape: vec![wanted.rows, width],
                    values,
                });
            }
            (ExpectedKind::Pooled, Embedding::All(_)) => {
                return Err(ProbeError::Embedding(format!(
                    "index {index} is expected to be pooled but backend returned All"
                )));
            }
            (ExpectedKind::Raw, Embedding::Pooled(_)) => {
                return Err(ProbeError::Embedding(format!(
                    "index {index} is expected to be raw but backend returned Pooled"
                )));
            }
        }
    }
    Ok(outputs)
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    use fmt::Write as _;
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        let _ = write!(encoded, "{byte:02x}");
    }
    encoded
}

fn width_from_json(bytes: &[u8]) -> Result<usize, ProbeError> {
    let config: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|err| ProbeError::Config(format!("config.json is invalid JSON: {err}")))?;
    let object = config
        .as_object()
        .ok_or_else(|| ProbeError::Config("config.json must be a JSON object".into()))?;
    for key in ["hidden_size", "dim"] {
        if let Some(value) = object.get(key) {
            let number = value.as_u64().ok_or_else(|| {
                ProbeError::Config(format!("{key} must be a non-negative integer"))
            })?;
            let width = usize::try_from(number)
                .map_err(|_| ProbeError::Config(format!("{key} does not fit in usize")))?;
            if width == 0 {
                return Err(ProbeError::Config(format!(
                    "{key} must be greater than zero"
                )));
            }
            return Ok(width);
        }
    }
    Err(ProbeError::Config(
        "config.json has neither hidden_size nor dim".into(),
    ))
}

pub fn read_model_width(model: &Path) -> Result<usize, ProbeError> {
    let config_path = model.join("config.json");
    let bytes = std::fs::read(&config_path).map_err(|source| ProbeError::Io {
        path: config_path.clone(),
        source,
    })?;
    width_from_json(&bytes)
        .map_err(|err| ProbeError::Config(format!("{}: {err}", config_path.display())))
}

pub fn parse_width_json(bytes: &[u8]) -> Result<usize, ProbeError> {
    width_from_json(bytes)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cls => "cls",
            Self::Mean => "mean",
            Self::LastToken => "last-token",
        }
    }

    pub fn backend_pool(self) -> Pool {
        match self {
            Self::Cls => Pool::Cls,
            Self::Mean => Pool::Mean,
            Self::LastToken => Pool::LastToken,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DType {
    Float16,
    #[default]
    BFloat16,
}

impl DType {
    pub fn parse(value: &str) -> Result<Self, ProbeError> {
        match value {
            "float16" => Ok(Self::Float16),
            "bfloat16" => Ok(Self::BFloat16),
            other => Err(ProbeError::Cli(format!(
                "dtype must be float16 or bfloat16, got {other}"
            ))),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Float16 => "float16",
            Self::BFloat16 => "bfloat16",
        }
    }

    pub fn code(self) -> i32 {
        match self {
            Self::Float16 => 1,
            Self::BFloat16 => 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Report {
    pub schema: String,
    pub backend: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pool: Option<String>,
    pub dtype: String,
    pub physical_batch: BatchFile,
    pub batch_sha256: String,
    pub width: usize,
    pub layout: Layout,
    pub canonical: CanonicalReport,
    pub outputs: Vec<OutputEntry>,
    pub validated_only: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CliArgs {
    pub model: Option<PathBuf>,
    pub batch: PathBuf,
    pub pool: Option<PoolArg>,
    pub dtype: DType,
    pub device: usize,
    pub output: Option<PathBuf>,
    pub validate_batch_only: bool,
    pub width: Option<usize>,
}

pub fn usage() -> &'static str {
    "raw_probe --model MODEL --batch EXACT.json --pool cls|mean|last-token \
--dtype float16|bfloat16 --device 0 --output JSON [--width H]\n\
       raw_probe --batch EXACT.json --validate-batch-only --width H [--output JSON]"
}

fn take_value<I>(values: &mut I, option: &str) -> Result<String, ProbeError>
where
    I: Iterator<Item = String>,
{
    values
        .next()
        .ok_or_else(|| ProbeError::Cli(format!("{option} requires a value")))
}

pub fn parse_args<I>(values: I) -> Result<CliArgs, ProbeError>
where
    I: IntoIterator<Item = String>,
{
    let mut values = values.into_iter();
    let mut model = None;
    let mut batch = None;
    let mut pool = None;
    let mut dtype = DType::default();
    let mut device = 0usize;
    let mut output = None;
    let mut validate_batch_only = false;
    let mut width = None;
    while let Some(option) = values.next() {
        match option.as_str() {
            "--model" => model = Some(PathBuf::from(take_value(&mut values, "--model")?)),
            "--batch" => batch = Some(PathBuf::from(take_value(&mut values, "--batch")?)),
            "--pool" => {
                let value = take_value(&mut values, "--pool")?;
                pool = Some(PoolArg::parse(&value)?);
            }
            "--dtype" => {
                let value = take_value(&mut values, "--dtype")?;
                dtype = DType::parse(&value)?;
            }
            "--device" => {
                let value = take_value(&mut values, "--device")?;
                device = value
                    .parse::<usize>()
                    .map_err(|_| ProbeError::Cli(format!("invalid device id {value}")))?;
            }
            "--output" => output = Some(PathBuf::from(take_value(&mut values, "--output")?)),
            "--width" => {
                let value = take_value(&mut values, "--width")?;
                let parsed = value
                    .parse::<usize>()
                    .map_err(|_| ProbeError::Cli(format!("invalid width {value}")))?;
                if parsed == 0 {
                    return Err(ProbeError::Cli("width must be greater than zero".into()));
                }
                width = Some(parsed);
            }
            "--validate-batch-only" => validate_batch_only = true,
            other => return Err(ProbeError::Cli(format!("unknown option {other}"))),
        }
    }
    let batch = batch.ok_or_else(|| ProbeError::Cli("--batch is required".into()))?;
    Ok(CliArgs {
        model,
        batch,
        pool,
        dtype,
        device,
        output,
        validate_batch_only,
        width,
    })
}

pub fn make_report(
    validated: &ValidatedBatch,
    batch_sha256: String,
    width: usize,
    pool: Option<PoolArg>,
    dtype: DType,
    outputs: Vec<OutputEntry>,
    validated_only: bool,
) -> Result<Report, ProbeError> {
    let layout = build_layout(validated, width)?;
    let expected = expected_canonical(validated)?;
    let canonical = canonical_report(&expected, width)?;
    Ok(Report {
        schema: SCHEMA.into(),
        backend: BACKEND_NAME.into(),
        pool: pool.map(|value| value.as_str().to_string()),
        dtype: dtype.as_str().into(),
        physical_batch: validated.file.clone(),
        batch_sha256,
        width,
        layout,
        canonical,
        outputs,
        validated_only,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(values: &[&str]) -> Result<CliArgs, ProbeError> {
        parse_args(values.iter().map(|value| (*value).to_string()))
    }

    fn sample_batch() -> ValidatedBatch {
        validate_batch(BatchFile {
            input_ids: vec![1, 2],
            position_ids: vec![0, 1],
            token_type_ids: vec![0, 0],
            cumulative_seq_lengths: vec![0, 2],
            pooled_indices: vec![0],
            raw_indices: Vec::new(),
            max_length: 2,
        })
        .expect("sample batch should validate")
    }

    #[test]
    fn parser_defaults_to_bfloat16() {
        let args = parse(&["--batch", "batch.json"]).expect("batch-only args");
        assert_eq!(args.dtype, DType::BFloat16);
        assert_eq!(args.dtype.as_str(), "bfloat16");
        assert_eq!(args.dtype.code(), 2);
    }

    #[test]
    fn parser_accepts_float16() {
        let args = parse(&["--batch", "batch.json", "--dtype", "float16"]).expect("float16 args");
        assert_eq!(args.dtype, DType::Float16);
        assert_eq!(args.dtype.as_str(), "float16");
        assert_eq!(args.dtype.code(), 1);
    }

    #[test]
    fn parser_rejects_other_dtypes() {
        for value in ["auto", "float32", "fp16", ""] {
            assert!(parse(&["--batch", "batch.json", "--dtype", value]).is_err());
        }
    }

    #[test]
    fn report_metadata_uses_selected_dtype() {
        let batch = sample_batch();
        let report = make_report(
            &batch,
            "a".repeat(64),
            2,
            Some(PoolArg::Mean),
            DType::Float16,
            Vec::new(),
            false,
        )
        .expect("report should build");
        assert_eq!(report.dtype, "float16");
    }
}
