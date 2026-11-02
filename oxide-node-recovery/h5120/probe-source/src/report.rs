use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::schema::{
    Api, BatchFile, BatchSummary, CanonicalSummary, HeadKind, LayoutReport, ModelBounds, Operation,
    PoolArg, SelectionPlan, Task,
};
use crate::ProbeError;

pub const SCHEMA: &str = "infinity-dense-unified-probe/v2";
pub const FROZEN_ROOT: &str =
    "/tmp/infinity-oxide-qwen3-h5120-public252-v1-worktree/libs/infinity-v3";

pub fn sha256_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        let _ = write!(encoded, "{byte:02x}");
    }
    encoded
}

pub fn sha256_file(path: &Path) -> Result<String, ProbeError> {
    let bytes = fs::read(path).map_err(|source| ProbeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(sha256_hex(&bytes))
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FileRef {
    pub path: String,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelRef {
    pub path: String,
    pub config_path: String,
    pub config_sha256: String,
    pub hidden_size: usize,
    pub head: HeadKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub head_labels: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label_source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_type: Option<String>,
    pub architectures: Vec<String>,
    pub decoder_family: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stored_dtype: Option<String>,
    pub runtime_dtype: String,
    pub dtype_conversion: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vocab_size: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_position_embeddings: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_hidden_layers: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_attention_heads: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_key_value_heads: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub head_dim: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attention_window: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pad_token_id: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub reference: ModelRef,
    pub width: usize,
    pub head: HeadKind,
    pub bounds: ModelBounds,
    pub pad_token_id: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FrozenDependency {
    pub name: String,
    pub path: String,
    pub cargo_toml_sha256: String,
    pub lib_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SourceCapsule {
    pub root: String,
    pub files: Vec<FileRef>,
    pub frozen_dependencies: Vec<FrozenDependency>,
}

fn optional_usize(
    object: &serde_json::Map<String, Value>,
    key: &str,
) -> Result<Option<usize>, ProbeError> {
    match object.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => {
            let number = value.as_u64().ok_or_else(|| {
                ProbeError::Config(format!("{key} must be a non-negative integer"))
            })?;
            usize::try_from(number)
                .map(Some)
                .map_err(|_| ProbeError::Config(format!("{key} does not fit in usize")))
        }
    }
}

fn optional_u32(
    object: &serde_json::Map<String, Value>,
    key: &str,
) -> Result<Option<u32>, ProbeError> {
    match object.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => {
            let number = value.as_u64().ok_or_else(|| {
                ProbeError::Config(format!("{key} must be a non-negative integer"))
            })?;
            u32::try_from(number)
                .map(Some)
                .map_err(|_| ProbeError::Config(format!("{key} does not fit in u32")))
        }
    }
}

fn optional_string(object: &serde_json::Map<String, Value>, key: &str) -> Option<String> {
    object
        .get(key)
        .and_then(Value::as_str)
        .map(|value| value.to_ascii_lowercase())
}

fn architectures(object: &serde_json::Map<String, Value>) -> Result<Vec<String>, ProbeError> {
    match object.get("architectures") {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Array(values)) => values
            .iter()
            .map(|value| {
                value.as_str().map(str::to_owned).ok_or_else(|| {
                    ProbeError::Config("architectures entries must be strings".into())
                })
            })
            .collect(),
        Some(_) => Err(ProbeError::Config(
            "architectures must be an array of strings".into(),
        )),
    }
}

fn decoder_family(model_type: Option<&str>) -> bool {
    matches!(
        model_type,
        Some("qwen2" | "qwen3" | "llama" | "mistral" | "bidirectional_pplx_qwen3")
    )
}

fn classify_head(
    architectures: &[String],
    model_type: Option<&str>,
) -> Result<HeadKind, ProbeError> {
    let mut causal = false;
    let mut sequence = false;
    let mut tokens = false;
    for architecture in architectures {
        if architecture.ends_with("ForSequenceClassification") {
            sequence = true;
        } else if architecture.ends_with("ForTokenClassification") {
            tokens = true;
        } else if matches!(
            architecture.as_str(),
            "Qwen2ForCausalLM"
                | "Qwen3ForCausalLM"
                | "LlamaForCausalLM"
                | "MistralForCausalLM"
                | "Qwen2Model"
                | "Qwen3Model"
                | "LlamaModel"
                | "MistralModel"
        ) {
            causal = true;
        }
    }
    let count = usize::from(causal) + usize::from(sequence) + usize::from(tokens);
    if count > 1 {
        return Err(ProbeError::Config(format!(
            "config architectures mix causal, sequence and token heads: {architectures:?}"
        )));
    }
    if sequence {
        if !decoder_family(model_type) {
            return Err(ProbeError::Config(format!(
                "recognized unsupported decoder classifier architecture for model_type {:?}",
                model_type
            )));
        }
        if !matches!(
            model_type,
            Some("qwen2" | "qwen3" | "llama" | "bidirectional_pplx_qwen3")
        ) {
            return Err(ProbeError::Config(format!(
                "decoder classifier architecture is not admitted for model_type {:?}",
                model_type
            )));
        }
        return Ok(HeadKind::SequenceClassification);
    }
    if tokens {
        if !matches!(
            model_type,
            Some("qwen2" | "qwen3" | "llama" | "bidirectional_pplx_qwen3")
        ) {
            return Err(ProbeError::Config(format!(
                "decoder classifier architecture is not admitted for model_type {:?}",
                model_type
            )));
        }
        return Ok(HeadKind::TokenClassification);
    }
    if causal || decoder_family(model_type) {
        return Ok(HeadKind::CausalLm);
    }
    Ok(HeadKind::Unknown)
}

fn label_count(
    object: &serde_json::Map<String, Value>,
) -> Result<Option<(usize, String)>, ProbeError> {
    let mapped = match object.get("id2label") {
        None | Some(Value::Null) => 0,
        Some(Value::Object(labels)) => labels.len(),
        Some(_) => {
            return Err(ProbeError::Config(
                "id2label must be an object when present".into(),
            ))
        }
    };
    let configured = optional_usize(object, "num_labels")?;
    let (labels, source) = if mapped > 0 {
        (mapped, "id2label")
    } else {
        (configured.unwrap_or(2), "num_labels_or_default_2")
    };
    if labels == 0 {
        return Err(ProbeError::Config(
            "decoder classifier labels must be nonzero".into(),
        ));
    }
    if let Some(configured) = configured {
        if configured != labels {
            return Err(ProbeError::Config(format!(
                "num_labels={configured} disagrees with {labels} id2label entries"
            )));
        }
    }
    Ok(Some((labels, source.into())))
}

fn dtype_reference(stored: Option<&str>) -> Result<(Option<String>, String, String), ProbeError> {
    match stored {
        None => Ok((None, "bfloat16".into(), "unspecified_to_bfloat16".into())),
        Some("float16" | "f16" | "half") => Ok((
            Some("float16".into()),
            "bfloat16".into(),
            "f16_to_bf16".into(),
        )),
        Some("bfloat16" | "bf16") => {
            Ok((Some("bfloat16".into()), "bfloat16".into(), "none".into()))
        }
        Some("float32" | "f32" | "float") => Ok((
            Some("float32".into()),
            "bfloat16".into(),
            "f32_to_bf16".into(),
        )),
        Some(other) => Err(ProbeError::Config(format!(
            "unsupported stored model dtype {other:?}; expected float16, bfloat16, or float32"
        ))),
    }
}

pub fn load_model_info(path: &Path) -> Result<ModelInfo, ProbeError> {
    if !path.is_absolute() {
        return Err(ProbeError::Config(format!(
            "model path must be absolute: {}",
            path.display()
        )));
    }
    if !path.is_dir() {
        return Err(ProbeError::Config(format!(
            "model path must be a directory: {}",
            path.display()
        )));
    }
    // This bounded probe uses one width for pooled and raw embeddings. Dense
    // projection chains require the separate D/H contract and are exercised by
    // the dedicated Dense controls; reject them before creating any engine.
    let modules_path = path.join("modules.json");
    if modules_path.exists() {
        let modules_bytes = fs::read(&modules_path).map_err(|source| ProbeError::Io {
            path: modules_path.clone(),
            source,
        })?;
        let modules: Value = serde_json::from_slice(&modules_bytes)
            .map_err(|err| ProbeError::Config(format!("invalid modules.json: {err}")))?;
        let modules = modules
            .as_array()
            .ok_or_else(|| ProbeError::Config("modules.json must be an array".into()))?;
        if modules.iter().any(|module| {
            module
                .get("type")
                .and_then(Value::as_str)
                .is_some_and(|kind| kind.rsplit('.').next() == Some("Dense"))
        }) {
            return Err(ProbeError::Config(
                "this probe requires equal pooled/raw widths; Dense chains need the dedicated D/H controls".into(),
            ));
        }
    }
    let config_path = path.join("config.json");
    let bytes = fs::read(&config_path).map_err(|source| ProbeError::Io {
        path: config_path.clone(),
        source,
    })?;
    let config: Value = serde_json::from_slice(&bytes).map_err(|err| {
        ProbeError::Config(format!("{} is invalid JSON: {err}", config_path.display()))
    })?;
    let object = config
        .as_object()
        .ok_or_else(|| ProbeError::Config("config.json must be a JSON object".into()))?;

    let hidden_size = match (
        optional_usize(object, "hidden_size")?,
        optional_usize(object, "dim")?,
    ) {
        (Some(hidden), Some(dim)) if hidden != dim => {
            return Err(ProbeError::Config(format!(
                "hidden_size={hidden} disagrees with dim={dim}"
            )))
        }
        (Some(width), _) | (None, Some(width)) => width,
        (None, None) => {
            return Err(ProbeError::Config(
                "config.json has neither hidden_size nor dim".into(),
            ))
        }
    };
    if hidden_size == 0 {
        return Err(ProbeError::Config(
            "hidden_size must be greater than zero".into(),
        ));
    }
    let vocab_size = optional_usize(object, "vocab_size")?;
    if matches!(vocab_size, Some(0)) {
        return Err(ProbeError::Config(
            "vocab_size must be greater than zero".into(),
        ));
    }
    let max_position_embeddings = optional_usize(object, "max_position_embeddings")?;
    if matches!(max_position_embeddings, Some(0)) {
        return Err(ProbeError::Config(
            "max_position_embeddings must be greater than zero".into(),
        ));
    }
    let model_type = optional_string(object, "model_type");
    let architectures = architectures(object)?;
    let head = classify_head(&architectures, model_type.as_deref())?;
    let labels = if head.is_classifier() {
        label_count(object)?
    } else {
        None
    };
    let stored =
        optional_string(object, "torch_dtype").or_else(|| optional_string(object, "dtype"));
    let (stored_dtype, runtime_dtype, dtype_conversion) = dtype_reference(stored.as_deref())?;
    let attention_window = match optional_usize(object, "sliding_window")? {
        Some(0) | None => None,
        Some(window) => Some(window),
    };
    let pad_token_id = optional_u32(object, "pad_token_id")?;
    let decoder = decoder_family(model_type.as_deref())
        || matches!(
            head,
            HeadKind::CausalLm | HeadKind::SequenceClassification | HeadKind::TokenClassification
        );
    let bounds = ModelBounds {
        vocab_size,
        max_position_embeddings,
        decoder_family: decoder,
    };
    let (head_labels, label_source) = match labels {
        Some((labels, source)) => (Some(labels), Some(source)),
        None => (None, None),
    };
    let reference = ModelRef {
        path: path.display().to_string(),
        config_path: config_path.display().to_string(),
        config_sha256: sha256_hex(&bytes),
        hidden_size,
        head,
        head_labels,
        label_source,
        model_type,
        architectures,
        decoder_family: decoder,
        stored_dtype,
        runtime_dtype,
        dtype_conversion,
        vocab_size,
        max_position_embeddings,
        num_hidden_layers: optional_usize(object, "num_hidden_layers")?,
        num_attention_heads: optional_usize(object, "num_attention_heads")?,
        num_key_value_heads: optional_usize(object, "num_key_value_heads")?,
        head_dim: optional_usize(object, "head_dim")?,
        attention_window,
        pad_token_id,
    };
    Ok(ModelInfo {
        width: hidden_size,
        head,
        bounds,
        pad_token_id,
        reference,
    })
}

pub fn source_capsule() -> Result<SourceCapsule, ProbeError> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    for relative in [
        "Cargo.toml",
        "Cargo.lock",
        "src/lib.rs",
        "src/main.rs",
        "src/schema.rs",
        "src/report.rs",
        "src/runtime.rs",
        "tests/cli.rs",
        "fixtures/embedding_requests.jsonl",
        "fixtures/reranker_requests.jsonl",
        "fixtures/reranker_invalid_operation.jsonl",
        "fixtures/invalid_recovery.jsonl",
    ] {
        let path = root.join(relative);
        if path.is_file() {
            files.push(FileRef {
                path: path.display().to_string(),
                sha256: sha256_file(&path)?,
            });
        }
    }
    let mut frozen_dependencies = Vec::new();
    for (name, relative) in [
        ("infinity_v3-backend", "backends"),
        ("infinity-backend-oxide", "backends/oxide"),
        ("infinity_v3-backend-core", "backends/core"),
    ] {
        let path = Path::new(FROZEN_ROOT).join(relative);
        let cargo = path.join("Cargo.toml");
        let lib = path.join("src/lib.rs");
        frozen_dependencies.push(FrozenDependency {
            name: name.into(),
            path: path.display().to_string(),
            cargo_toml_sha256: sha256_file(&cargo)?,
            lib_sha256: if lib.is_file() {
                Some(sha256_file(&lib)?)
            } else {
                None
            },
        });
    }
    Ok(SourceCapsule {
        root: root.display().to_string(),
        files,
        frozen_dependencies,
    })
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EmbeddingEntry {
    pub index: usize,
    pub kind: String,
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
    pub bits: Vec<u32>,
    pub finite: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EmbeddingResult {
    pub api_value_kind: String,
    pub entry_count: usize,
    pub order: String,
    pub raw_overrides_pooled: Vec<usize>,
    pub duplicate_selection_indices: Vec<usize>,
    pub entries: Vec<EmbeddingEntry>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PredictionRow {
    pub index: usize,
    pub values: Vec<f32>,
    pub bits: Vec<u32>,
    pub finite: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PredictionResult {
    pub api_value_kind: String,
    pub row_count: usize,
    pub class_count: usize,
    pub rows: Vec<PredictionRow>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TokenRow {
    pub token_index: usize,
    pub values: Vec<f32>,
    pub bits: Vec<u32>,
    pub finite: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TokenPredictionSequence {
    pub index: usize,
    pub token_count: usize,
    pub rows: Vec<TokenRow>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TokenPredictionResult {
    pub api_value_kind: String,
    pub sequence_count: usize,
    pub class_count: usize,
    pub sequences: Vec<TokenPredictionSequence>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CaseOutput {
    Embedding(EmbeddingResult),
    Predictions(PredictionResult),
    TokenPredictions(TokenPredictionResult),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CaseStatus {
    Validated,
    Ok,
    ExpectedReject,
    Failed,
    Skipped,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StructuredError {
    pub kind: String,
    pub phase: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub case_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line_number: Option<usize>,
}

impl StructuredError {
    pub fn new(kind: &str, phase: &str, message: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            phase: phase.into(),
            message: message.into(),
            case_id: None,
            line_number: None,
        }
    }

    pub fn with_case(mut self, case_id: &str, line_number: usize) -> Self {
        self.case_id = Some(case_id.into());
        self.line_number = Some(line_number);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DurationReport {
    pub source: String,
    pub backend_reported_nanos: u64,
    pub timing_claim: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CaseReport {
    pub id: String,
    pub operation: Operation,
    pub task: Task,
    pub pool: PoolArg,
    pub status: CaseStatus,
    pub line_number: usize,
    pub request_line_sha256: String,
    pub physical_batch: BatchFile,
    pub batch_sha256: String,
    pub batch_summary: BatchSummary,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub canonical: Option<CanonicalSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layout: Option<LayoutReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selection: Option<SelectionPlan>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<CaseOutput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backend_duration: Option<DurationReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_reject_error: Option<StructuredError>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub health_after: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<StructuredError>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LifecycleStep {
    pub name: String,
    pub status: String,
    pub detail: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bit_exact_match: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<StructuredError>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LifecycleReport {
    pub status: String,
    pub task: Task,
    pub head: HeadKind,
    pub lease_capacity: usize,
    pub observed_max_live_leases: usize,
    pub third_submission_while_two_live: bool,
    pub polling_loop: bool,
    pub capacity_note: String,
    pub steps: Vec<LifecycleStep>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<StructuredError>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RequestFileRef {
    pub path: String,
    pub sha256: String,
    pub line_count: usize,
    pub request_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProbeReport {
    pub schema: String,
    pub mode: String,
    pub api: Api,
    pub task: Task,
    pub pool: PoolArg,
    pub dtype: String,
    pub engine_created: bool,
    pub parser_only: bool,
    pub lifecycle: bool,
    pub cuda_visible_devices: String,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<ModelRef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requests: Option<RequestFileRef>,
    pub source_capsule: SourceCapsule,
    pub output_dir: String,
    pub report_path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub primary_error: Option<StructuredError>,
    pub cases: Vec<CaseReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lifecycle_report: Option<LifecycleReport>,
}

pub fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<(), ProbeError> {
    let encoded = serde_json::to_vec_pretty(value).map_err(ProbeError::Json)?;
    let parent = path
        .parent()
        .ok_or_else(|| ProbeError::Config(format!("path has no parent: {}", path.display())))?;
    fs::create_dir_all(parent).map_err(|source| ProbeError::Io {
        path: parent.to_path_buf(),
        source,
    })?;
    let temporary = parent.join(format!(
        ".{}.tmp-{}",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("report"),
        std::process::id()
    ));
    {
        let mut file = fs::File::create(&temporary).map_err(|source| ProbeError::Io {
            path: temporary.clone(),
            source,
        })?;
        file.write_all(&encoded)
            .and_then(|()| file.write_all(b"\n"))
            .map_err(|source| ProbeError::Io {
                path: temporary.clone(),
                source,
            })?;
        file.sync_all().map_err(|source| ProbeError::Io {
            path: temporary.clone(),
            source,
        })?;
    }
    fs::rename(&temporary, path).map_err(|source| ProbeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(())
}

pub fn safe_file_stem(id: &str) -> String {
    if !id.is_empty()
        && id != "."
        && id != ".."
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        return id.into();
    }
    format!("case-{}", &sha256_hex(id.as_bytes())[..16])
}

pub fn prepare_fresh_dir(path: &Path) -> Result<(), ProbeError> {
    if path.exists() {
        if !path.is_dir() {
            return Err(ProbeError::Config(format!(
                "output path exists and is not a directory: {}",
                path.display()
            )));
        }
        let mut entries = fs::read_dir(path).map_err(|source| ProbeError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        if entries.next().is_some() {
            return Err(ProbeError::Config(format!(
                "output directory must be fresh and empty: {}",
                path.display()
            )));
        }
    } else {
        fs::create_dir_all(path).map_err(|source| ProbeError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_model_config(root: &Path, value: &Value) -> PathBuf {
        let model = root.join("model");
        fs::create_dir_all(&model).expect("model dir");
        fs::write(
            model.join("config.json"),
            serde_json::to_vec(value).unwrap(),
        )
        .expect("write config");
        model
    }

    #[test]
    fn reranker_metadata_is_sequence_c2_f16_to_bf16() {
        let root =
            std::env::temp_dir().join(format!("dense-probe-report-meta-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let model = write_model_config(
            &root,
            &serde_json::json!({
                "model_type": "qwen3",
                "architectures": ["Qwen3ForSequenceClassification"],
                "hidden_size": 2560,
                "vocab_size": 151669,
                "max_position_embeddings": 40960,
                "torch_dtype": "float16",
                "id2label": {"0": "no", "1": "yes"}
            }),
        );
        let info = load_model_info(&model).expect("metadata");
        assert_eq!(info.head, HeadKind::SequenceClassification);
        assert_eq!(info.reference.head_labels, Some(2));
        assert_eq!(info.reference.stored_dtype.as_deref(), Some("float16"));
        assert_eq!(info.reference.runtime_dtype, "bfloat16");
        assert_eq!(info.reference.dtype_conversion, "f16_to_bf16");
        assert!(info.bounds.decoder_family);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn mistral_positive_sliding_window_selects_local_attention() {
        let root =
            std::env::temp_dir().join(format!("dense-probe-report-window-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let model = write_model_config(
            &root,
            &serde_json::json!({
                "model_type": "mistral",
                "architectures": ["MistralForCausalLM"],
                "hidden_size": 4096,
                "vocab_size": 32000,
                "max_position_embeddings": 32768,
                "torch_dtype": "bfloat16",
                "sliding_window": 4096,
                "use_sliding_window": false
            }),
        );
        let info = load_model_info(&model).expect("metadata");
        assert_eq!(info.head, HeadKind::CausalLm);
        assert_eq!(info.reference.attention_window, Some(4096));
        assert_eq!(info.reference.dtype_conversion, "none");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn dense_modules_are_rejected_before_single_width_layout() {
        let root =
            std::env::temp_dir().join(format!("dense-probe-report-dense-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let model = write_model_config(
            &root,
            &serde_json::json!({
                "model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"],
                "hidden_size": 2560, "vocab_size": 32000
            }),
        );
        let modules = model.join("modules.json");
        fs::write(
            &modules,
            br#"[{"type":"sentence_transformers.models.Dense","path":"1_Dense"}]"#,
        )
        .unwrap();
        let error = load_model_info(&model).unwrap_err().to_string();
        assert!(error.contains("equal pooled/raw widths"));
        fs::write(
            &modules,
            br#"[{"type":"sentence_transformers.models.Transformer","path":""}]"#,
        )
        .unwrap();
        assert!(load_model_info(&model).is_ok());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn conflicting_label_counts_are_rejected() {
        let root =
            std::env::temp_dir().join(format!("dense-probe-report-labels-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let model = write_model_config(
            &root,
            &serde_json::json!({
                "model_type": "qwen3",
                "architectures": ["Qwen3ForSequenceClassification"],
                "hidden_size": 8,
                "num_labels": 3,
                "id2label": {"0": "no", "1": "yes"}
            }),
        );
        assert!(load_model_info(&model).is_err());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn signed_zero_bits_are_preserved_in_serialization() {
        let values: Vec<f32> = vec![0.0, -0.0, 1.5];
        let entry = EmbeddingEntry {
            index: 0,
            kind: "pooled".into(),
            shape: vec![3],
            bits: values.iter().map(|value| value.to_bits()).collect(),
            values,
            finite: true,
        };
        assert_eq!(entry.bits[0], 0);
        assert_eq!(entry.bits[1], 0x8000_0000);
        let encoded = serde_json::to_string(&entry).expect("json");
        assert!(encoded.contains("\"bits\":[0,2147483648,1069547520]"));
    }

    #[test]
    fn safe_file_stem_falls_back_for_unsafe_ids() {
        assert_eq!(safe_file_stem("case-1"), "case-1");
        let stem = safe_file_stem("../bad id");
        assert!(stem.starts_with("case-"));
        assert!(!stem.contains('/'));
    }
}
