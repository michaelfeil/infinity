use std::collections::BTreeSet;
use std::env;
use std::fmt;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use crate::report::{
    CaseOutput, CaseReport, CaseStatus, DurationReport, LifecycleReport, LifecycleStep, ModelInfo,
    ProbeReport, RequestFileRef, StructuredError, SCHEMA,
};

mod report;
mod runtime;
mod schema;

use report::{
    load_model_info, prepare_fresh_dir, safe_file_stem, sha256_hex, source_capsule,
    write_json_atomic,
};
use runtime::{
    backend_error, construct_core, construct_factory, embedding_output, model_type,
    predictions_output, token_predictions_output, Completed, Engine, LeaseTracker, TrackedSubmit,
};
use schema::{
    build_layout, canonical_summary, expected_canonical, sequence_selection, token_selection,
    validate_batch, validate_operation, validate_task_model, Api, BatchSummary, HeadKind,
    Operation, PoolArg, RequestLine, Task, ValidatedBatch, MAX_REQUEST_BYTES,
};

#[derive(Debug)]
pub enum ProbeError {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Json(serde_json::Error),
    Batch(String),
    Layout(String),
    Config(String),
    Cli(String),
    Operation(String),
    Output(String),
    Backend(StructuredError),
    Env(String),
    Lifecycle(String),
    AtLine {
        line: usize,
        kind: &'static str,
        phase: &'static str,
        message: String,
    },
}

impl fmt::Display for ProbeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "I/O error for {}: {source}", path.display()),
            Self::Json(err) => write!(f, "invalid JSON: {err}"),
            Self::Batch(message) => write!(f, "invalid batch: {message}"),
            Self::Layout(message) => write!(f, "invalid layout: {message}"),
            Self::Config(message) => write!(f, "invalid model config: {message}"),
            Self::Cli(message) => write!(f, "invalid command line: {message}"),
            Self::Operation(message) => write!(f, "invalid operation: {message}"),
            Self::Output(message) => write!(f, "invalid backend output: {message}"),
            Self::Backend(error) => write!(f, "backend failure: {}", error.message),
            Self::Env(message) => write!(f, "environment failure: {message}"),
            Self::Lifecycle(message) => write!(f, "lifecycle failure: {message}"),
            Self::AtLine { line, message, .. } => write!(f, "request line {line}: {message}"),
        }
    }
}

impl std::error::Error for ProbeError {}

impl From<serde_json::Error> for ProbeError {
    fn from(err: serde_json::Error) -> Self {
        Self::Json(err)
    }
}

impl ProbeError {
    fn kind(&self) -> &str {
        match self {
            Self::Io { .. } => "io",
            Self::Json(_) => "json",
            Self::Batch(_) => "batch",
            Self::Layout(_) => "layout",
            Self::Config(_) => "config",
            Self::Cli(_) => "cli",
            Self::Operation(_) => "operation",
            Self::Output(_) => "output",
            Self::Backend(error) => error.kind.as_str(),
            Self::Env(_) => "env",
            Self::Lifecycle(_) => "lifecycle",
            Self::AtLine { kind, .. } => kind,
        }
    }

    fn phase(&self) -> &'static str {
        match self {
            Self::Io { .. } => "io",
            Self::Json(_) => "request_parse",
            Self::Batch(_) | Self::Layout(_) => "request_validation",
            Self::Config(_) => "model_config",
            Self::Cli(_) => "cli",
            Self::Operation(_) => "operation_validation",
            Self::Output(_) => "output_validation",
            Self::Backend(_) => "backend",
            Self::Env(_) => "preflight_env",
            Self::Lifecycle(_) => "lifecycle",
            Self::AtLine { phase, .. } => phase,
        }
    }

    fn structured(&self) -> StructuredError {
        let line = match self {
            Self::AtLine { line, .. } => Some(*line),
            _ => None,
        };
        StructuredError {
            kind: self.kind().into(),
            phase: self.phase().into(),
            message: self.to_string(),
            case_id: None,
            line_number: line,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    CheckOnly,
    Run,
}

#[derive(Debug, Clone)]
struct Cli {
    model: PathBuf,
    task: Task,
    pool: PoolArg,
    api: Api,
    requests: PathBuf,
    output: PathBuf,
    mode: Mode,
    lifecycle: bool,
}

fn take_value<I>(values: &mut I, option: &str) -> Result<String, ProbeError>
where
    I: Iterator<Item = String>,
{
    values
        .next()
        .ok_or_else(|| ProbeError::Cli(format!("{option} requires a value")))
}

fn parse_cli<I>(values: I) -> Result<Cli, ProbeError>
where
    I: IntoIterator<Item = String>,
{
    let mut values = values.into_iter();
    let mut model = None;
    let mut task = None;
    let mut pool = None;
    let mut api = None;
    let mut requests = None;
    let mut output = None;
    let mut check_only = false;
    let mut run = false;
    let mut lifecycle = false;
    while let Some(option) = values.next() {
        match option.as_str() {
            "--model" => model = Some(PathBuf::from(take_value(&mut values, "--model")?)),
            "--task" => task = Some(Task::parse(&take_value(&mut values, "--task")?)?),
            "--pool" => pool = Some(PoolArg::parse(&take_value(&mut values, "--pool")?)?),
            "--api" => api = Some(Api::parse(&take_value(&mut values, "--api")?)?),
            "--requests" => {
                requests = Some(PathBuf::from(take_value(&mut values, "--requests")?));
            }
            "--output" => output = Some(PathBuf::from(take_value(&mut values, "--output")?)),
            "--check-only" => check_only = true,
            "--run" => run = true,
            "--exercise-lifecycle" => lifecycle = true,
            "--help" | "-h" => {
                return Err(ProbeError::Cli("help requested".into()));
            }
            other => return Err(ProbeError::Cli(format!("unknown option {other}"))),
        }
    }
    if check_only && run {
        return Err(ProbeError::Cli(
            "--check-only and --run are mutually exclusive".into(),
        ));
    }
    let mode = if check_only {
        Mode::CheckOnly
    } else if run {
        Mode::Run
    } else {
        return Err(ProbeError::Cli(
            "exactly one of --check-only or --run is required".into(),
        ));
    };
    let task = task.ok_or_else(|| ProbeError::Cli("--task is required".into()))?;
    let model = model.ok_or_else(|| ProbeError::Cli("--model is required".into()))?;
    let api = api.ok_or_else(|| ProbeError::Cli("--api is required".into()))?;
    let requests = requests.ok_or_else(|| ProbeError::Cli("--requests is required".into()))?;
    let output = output.ok_or_else(|| ProbeError::Cli("--output is required".into()))?;
    let pool = match (task, pool) {
        (Task::Embedding, Some(pool)) => pool,
        (Task::Embedding, None) => {
            return Err(ProbeError::Cli(
                "--pool is required for an embedding task".into(),
            ));
        }
        (Task::Classifier, None | Some(PoolArg::Cls)) => PoolArg::Cls,
        (Task::Classifier, Some(_)) => {
            return Err(ProbeError::Cli(
                "classifier tasks use the sequence head and accept no --pool other than cls".into(),
            ));
        }
    };
    if lifecycle && (api != Api::Core || mode != Mode::Run) {
        return Err(ProbeError::Cli(
            "--exercise-lifecycle requires --api core and --run".into(),
        ));
    }
    if !model.is_absolute() {
        return Err(ProbeError::Cli(format!(
            "--model must be an absolute path: {}",
            model.display()
        )));
    }
    Ok(Cli {
        model,
        task,
        pool,
        api,
        requests,
        output,
        mode,
        lifecycle,
    })
}

pub fn usage() -> &'static str {
    "dense_probe --model ABS --task embedding|classifier [--pool cls|mean|last-token] \
--api factory|core --requests REQUESTS.jsonl --output FRESHDIR \
--check-only|--run [--exercise-lifecycle]"
}

#[derive(Debug, Clone)]
struct LoadedCase {
    line_number: usize,
    id: String,
    operation: Operation,
    request_sha256: String,
    batch_sha256: String,
    validated: ValidatedBatch,
}

#[derive(Debug)]
struct LoadedRequests {
    cases: Vec<LoadedCase>,
}

fn read_capped(path: &Path, cap: u64) -> Result<Vec<u8>, ProbeError> {
    let file = File::open(path).map_err(|source| ProbeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut bytes = Vec::new();
    file.take(cap + 1)
        .read_to_end(&mut bytes)
        .map_err(|source| ProbeError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    if bytes.len() as u64 > cap {
        return Err(ProbeError::Config(format!(
            "{} exceeds the {cap} byte cap",
            path.display()
        )));
    }
    Ok(bytes)
}

fn parse_requests(
    bytes: &[u8],
    task: Task,
    model: &ModelInfo,
) -> Result<LoadedRequests, ProbeError> {
    let mut cases = Vec::new();
    let mut ids = BTreeSet::new();
    for (index, raw) in bytes.split(|byte| *byte == b'\n').enumerate() {
        let line_number = index + 1;
        let raw = raw.strip_suffix(b"\r").unwrap_or(raw);
        if raw.iter().all(|byte| byte.is_ascii_whitespace()) {
            continue;
        }
        let parsed: RequestLine =
            schema::parse_request_line(raw).map_err(|err| ProbeError::AtLine {
                line: line_number,
                kind: "request_line",
                phase: "request_parse",
                message: err.to_string(),
            })?;
        if parsed.id.is_empty() || parsed.id == "." || parsed.id == ".." {
            return Err(ProbeError::AtLine {
                line: line_number,
                kind: "request_line",
                phase: "request_parse",
                message: "id must be a non-empty safe case identifier".into(),
            });
        }
        if !ids.insert(parsed.id.clone()) {
            return Err(ProbeError::AtLine {
                line: line_number,
                kind: "request_line",
                phase: "request_parse",
                message: format!("duplicate case id {}", parsed.id),
            });
        }
        validate_operation(task, model.head, &parsed).map_err(|err| ProbeError::AtLine {
            line: line_number,
            kind: "operation",
            phase: "operation_validation",
            message: err.to_string(),
        })?;
        let validated = validate_batch(parsed.batch.clone(), &model.bounds).map_err(|err| {
            ProbeError::AtLine {
                line: line_number,
                kind: "batch",
                phase: "request_validation",
                message: err.to_string(),
            }
        })?;
        let batch_sha256 =
            sha256_hex(&serde_json::to_vec(&parsed.batch).map_err(ProbeError::Json)?);
        cases.push(LoadedCase {
            line_number,
            id: parsed.id,
            operation: parsed.operation,
            request_sha256: sha256_hex(raw),
            batch_sha256,
            validated,
        });
    }
    if cases.is_empty() {
        return Err(ProbeError::Config(
            "request file must contain at least one request".into(),
        ));
    }
    Ok(LoadedRequests { cases })
}

fn validate_lifecycle_contract(
    cli: &Cli,
    model: &ModelInfo,
    loaded: &LoadedRequests,
) -> Result<(), ProbeError> {
    if !cli.lifecycle {
        return Ok(());
    }
    if loaded.cases.len() < 2 {
        return Err(ProbeError::Config(
            "--exercise-lifecycle requires at least two requests".into(),
        ));
    }
    if cli.task == Task::Classifier
        && model.head == HeadKind::SequenceClassification
        && !loaded
            .cases
            .iter()
            .any(|case| case.operation == Operation::ExpectRejectPredictTokens)
    {
        return Err(ProbeError::Config(
            "sequence-head lifecycle requires an expect_reject_predict_tokens request".into(),
        ));
    }
    Ok(())
}

fn base_case_report(
    case: &LoadedCase,
    cli: &Cli,
    model: &ModelInfo,
) -> Result<CaseReport, ProbeError> {
    let mut canonical = None;
    let mut layout = None;
    let mut selection = None;
    if cli.task == Task::Embedding {
        let expected = expected_canonical(&case.validated)?;
        canonical = Some(canonical_summary(&expected, Some(model.width))?);
        layout = Some(build_layout(&case.validated, model.width)?);
    } else {
        selection = Some(match model.head {
            HeadKind::SequenceClassification => {
                sequence_selection(&case.validated, model.pad_token_id)?
            }
            HeadKind::TokenClassification => token_selection(&case.validated),
            _ => {
                return Err(ProbeError::Config(format!(
                    "classifier task cannot annotate head {}",
                    model.head.as_str()
                )))
            }
        });
    }
    Ok(CaseReport {
        id: case.id.clone(),
        operation: case.operation,
        task: cli.task,
        pool: cli.pool,
        status: CaseStatus::Validated,
        line_number: case.line_number,
        request_line_sha256: case.request_sha256.clone(),
        physical_batch: case.validated.file.clone(),
        batch_sha256: case.batch_sha256.clone(),
        batch_summary: BatchSummary::new(&case.validated),
        canonical,
        layout,
        selection,
        result: None,
        backend_duration: None,
        expected_reject_error: None,
        health_after: None,
        error: None,
    })
}

fn build_batch(validated: &ValidatedBatch) -> infinity_v3_backend_core::Batch {
    infinity_v3_backend_core::Batch {
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

fn duration_report(duration: Option<std::time::Duration>) -> Option<DurationReport> {
    duration.map(|duration| DurationReport {
        source: "backend_reported".into(),
        backend_reported_nanos: u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX),
        timing_claim: "none".into(),
    })
}

async fn execute_case(
    engine: &Engine,
    case: &LoadedCase,
    cli: &Cli,
    model: &ModelInfo,
) -> Result<CaseReport, ProbeError> {
    let mut case_report = base_case_report(case, cli, model)?;
    let expected = if cli.task == Task::Embedding {
        Some(expected_canonical(&case.validated)?)
    } else {
        None
    };
    let batch = build_batch(&case.validated);
    match case.operation {
        Operation::Embed => {
            let expected = expected.as_ref().ok_or_else(|| {
                ProbeError::Operation("embedding operation is missing canonical selection".into())
            })?;
            let (embeddings, duration) = engine
                .embed(batch)
                .await
                .map_err(|err| ProbeError::Backend(backend_error("embed", &err)))?;
            let output = embedding_output(expected, &embeddings)?;
            drop(embeddings);
            case_report.result = Some(output);
            case_report.backend_duration = duration_report(duration);
        }
        Operation::Predict => {
            let sequences = case.validated.batch_count;
            let (predictions, duration) = engine
                .predict(batch)
                .await
                .map_err(|err| ProbeError::Backend(backend_error("predict", &err)))?;
            let output = predictions_output(&predictions, sequences)?;
            drop(predictions);
            case_report.result = Some(output);
            case_report.backend_duration = duration_report(duration);
        }
        Operation::PredictTokens => {
            let lengths = case.validated.sequence_lengths.clone();
            let (predictions, duration) = engine
                .predict_tokens(batch)
                .await
                .map_err(|err| ProbeError::Backend(backend_error("predict_tokens", &err)))?;
            let output = token_predictions_output(&predictions, &lengths)?;
            drop(predictions);
            case_report.result = Some(output);
            case_report.backend_duration = duration_report(duration);
        }
        Operation::ExpectRejectPredictTokens => {
            match engine.predict_tokens(batch).await {
                Ok((predictions, _duration)) => {
                    drop(predictions);
                    case_report.status = CaseStatus::Failed;
                    case_report.error = Some(StructuredError::new(
                        "expected_reject_succeeded",
                        "predict_tokens",
                        "expected predict_tokens to reject, but it completed successfully",
                    ));
                    return Ok(case_report);
                }
                Err(err) => {
                    case_report.expected_reject_error = Some(backend_error("predict_tokens", &err));
                }
            }
            match engine.health().await {
                Ok(()) => {
                    case_report.status = CaseStatus::ExpectedReject;
                    case_report.health_after = Some("ok".into());
                }
                Err(err) => {
                    case_report.status = CaseStatus::Failed;
                    case_report.health_after = Some("failed".into());
                    case_report.error = Some(backend_error("health_after_expected_reject", &err));
                }
            }
            return Ok(case_report);
        }
    }
    if let Some(expected_labels) = model.reference.head_labels {
        let output = case_report
            .result
            .as_ref()
            .ok_or_else(|| ProbeError::Output("classifier completed without an output".into()))?;
        runtime::require_class_width(output, expected_labels)?;
    }
    case_report.status = CaseStatus::Ok;
    Ok(case_report)
}

fn persist(report: &mut ProbeReport, cli: &Cli) -> Result<(), ProbeError> {
    write_json_atomic(&cli.output.join("report.json"), report)?;
    let cases_dir = cli.output.join("cases");
    for case in &report.cases {
        write_json_atomic(
            &cases_dir.join(format!("{}.json", safe_file_stem(&case.id))),
            case,
        )?;
    }
    Ok(())
}

fn mark_skipped(
    report: &mut ProbeReport,
    loaded: &[LoadedCase],
    from: usize,
    cli: &Cli,
    model: &ModelInfo,
) {
    for case in loaded.iter().skip(from) {
        if let Ok(mut case_report) = base_case_report(case, cli, model) {
            case_report.status = CaseStatus::Skipped;
            report.cases.push(case_report);
        }
    }
}

struct PreparedCase<'a> {
    case: &'a LoadedCase,
    batch: infinity_v3_backend_core::Batch,
    expected: Option<schema::ExpectedCanonical>,
}

fn prepare_case<'a>(case: &'a LoadedCase, task: Task) -> Result<PreparedCase<'a>, ProbeError> {
    Ok(PreparedCase {
        case,
        batch: build_batch(&case.validated),
        expected: if task == Task::Embedding {
            Some(expected_canonical(&case.validated)?)
        } else {
            None
        },
    })
}

fn reference_output(
    backend: &infinity_backend_oxide::OxideBackend,
    task: Task,
    prepared: &PreparedCase<'_>,
) -> Result<CaseOutput, ProbeError> {
    match task {
        Task::Embedding => {
            let embeddings =
                infinity_v3_backend_core::Backend::embed(backend, prepared.batch.clone())
                    .map_err(|err| ProbeError::Backend(backend_error("reference_embed", &err)))?;
            let expected = prepared.expected.as_ref().ok_or_else(|| {
                ProbeError::Lifecycle("reference embedding has no canonical selection".into())
            })?;
            let output = embedding_output(expected, &embeddings)?;
            drop(embeddings);
            Ok(output)
        }
        Task::Classifier => {
            let predictions =
                infinity_v3_backend_core::Backend::predict(backend, prepared.batch.clone())
                    .map_err(|err| ProbeError::Backend(backend_error("reference_predict", &err)))?;
            let output = predictions_output(&predictions, prepared.case.validated.batch_count)?;
            drop(predictions);
            Ok(output)
        }
    }
}

fn completed_output(
    completed: Completed,
    task: Task,
    prepared: &PreparedCase<'_>,
) -> Result<CaseOutput, ProbeError> {
    match (task, completed) {
        (Task::Embedding, Completed::Embed(embeddings)) => {
            let expected = prepared.expected.as_ref().ok_or_else(|| {
                ProbeError::Lifecycle("held embedding has no canonical selection".into())
            })?;
            let output = embedding_output(expected, &embeddings)?;
            drop(embeddings);
            Ok(output)
        }
        (Task::Classifier, Completed::Predict(predictions)) => {
            let output = predictions_output(&predictions, prepared.case.validated.batch_count)?;
            drop(predictions);
            Ok(output)
        }
        _ => Err(ProbeError::Lifecycle(
            "completed result kind does not match the task".into(),
        )),
    }
}

fn step(
    name: &str,
    status: &str,
    detail: impl Into<String>,
    bit_exact_match: Option<bool>,
) -> LifecycleStep {
    LifecycleStep {
        name: name.into(),
        status: status.into(),
        detail: detail.into(),
        bit_exact_match,
        error: None,
    }
}

fn failed_step(name: &str, detail: impl Into<String>, error: StructuredError) -> LifecycleStep {
    LifecycleStep {
        name: name.into(),
        status: "failed".into(),
        detail: detail.into(),
        bit_exact_match: None,
        error: Some(error),
    }
}

fn expect_live(tracker: &LeaseTracker, expected: usize, step_name: &str) -> Result<(), ProbeError> {
    if tracker.live() == expected {
        Ok(())
    } else {
        Err(ProbeError::Lifecycle(format!(
            "{step_name}: expected {expected} live leases, observed {}",
            tracker.live()
        )))
    }
}

fn run_lifecycle(
    backend: &infinity_backend_oxide::OxideBackend,
    cases: &[LoadedCase],
    task: Task,
    model: &ModelInfo,
    steps: &mut Vec<LifecycleStep>,
    tracker: &Arc<LeaseTracker>,
) -> Result<(), ProbeError> {
    if cases.len() < 2 {
        return Err(ProbeError::Lifecycle(
            "--exercise-lifecycle requires at least two requests".into(),
        ));
    }
    let prepared_a = prepare_case(&cases[0], task)?;
    let prepared_b = prepare_case(&cases[1], task)?;
    let later_index = if cases.len() > 2 { 2 } else { 0 };
    let prepared_later = prepare_case(&cases[later_index], task)?;
    let reference_a = reference_output(backend, task, &prepared_a)?;
    let reference_b = reference_output(backend, task, &prepared_b)?;
    steps.push(step(
        "synchronous_reference",
        "ok",
        format!(
            "captured bit-exact reference results for cases {} and {}",
            prepared_a.case.id, prepared_b.case.id
        ),
        None,
    ));

    let held_a = TrackedSubmit::submit(backend, task, prepared_a.batch.clone(), tracker)?;
    let held_b = TrackedSubmit::submit(backend, task, prepared_b.batch.clone(), tracker)?;
    expect_live(tracker, 2, "hold_two_deferred_leases")?;
    steps.push(step(
        "hold_two_deferred_leases",
        "ok",
        format!(
            "case {} deferred={}, case {} deferred={}, live_leases={} capacity={}",
            prepared_a.case.id,
            held_a.is_deferred(),
            prepared_b.case.id,
            held_b.is_deferred(),
            tracker.live(),
            tracker.capacity()
        ),
        None,
    ));
    steps.push(step(
        "two_lease_capacity_saturated",
        "ok",
        "no third submission was attempted while A and B were live; the tracker rejects a third reservation",
        None,
    ));

    let completed_b = held_b
        .complete()
        .map_err(|err| ProbeError::Backend(backend_error("lifecycle_complete_b", &err)))?;
    expect_live(tracker, 1, "complete_b_out_of_order")?;
    let output_b = completed_output(completed_b, task, &prepared_b)?;
    let match_b = output_b == reference_b;
    steps.push(step(
        "complete_b_out_of_order",
        if match_b { "ok" } else { "failed" },
        format!(
            "completed case {} before case {}; live_leases={}",
            prepared_b.case.id,
            prepared_a.case.id,
            tracker.live()
        ),
        Some(match_b),
    ));
    drop(output_b);
    drop(reference_b);
    if !match_b {
        return Err(ProbeError::Lifecycle(
            "out-of-order completion did not match the synchronous reference bits".into(),
        ));
    }

    let held_c = TrackedSubmit::submit(backend, task, prepared_later.batch.clone(), tracker)?;
    expect_live(tracker, 2, "drop_deferred_c_while_a_held")?;
    let c_deferred = held_c.is_deferred();
    drop(held_c);
    expect_live(tracker, 1, "drop_deferred_c_while_a_held")?;
    steps.push(step(
        "drop_deferred_c_while_a_held",
        if c_deferred { "ok" } else { "failed" },
        format!(
            "dropped deferred case {} without completing it; live_leases={}",
            prepared_later.case.id,
            tracker.live()
        ),
        None,
    ));
    if !c_deferred {
        return Err(ProbeError::Lifecycle(
            "core submit did not return a deferred result".into(),
        ));
    }
    infinity_v3_backend_core::Backend::health(backend)
        .map_err(|err| ProbeError::Backend(backend_error("lifecycle_health_after_drop", &err)))?;
    steps.push(step(
        "health_after_cancellation",
        "ok",
        "backend remained healthy after dropping the deferred lease",
        None,
    ));

    let completed_a = held_a
        .complete()
        .map_err(|err| ProbeError::Backend(backend_error("lifecycle_complete_a", &err)))?;
    expect_live(tracker, 0, "complete_a_after_cancellation")?;
    let output_a = completed_output(completed_a, task, &prepared_a)?;
    let match_a = output_a == reference_a;
    steps.push(step(
        "complete_a_after_cancellation",
        if match_a { "ok" } else { "failed" },
        format!(
            "completed case {} after dropping case {}; live_leases={}",
            prepared_a.case.id,
            prepared_later.case.id,
            tracker.live()
        ),
        Some(match_a),
    ));
    drop(output_a);
    drop(reference_a);
    if !match_a {
        return Err(ProbeError::Lifecycle(
            "held A completion did not match the synchronous reference bits".into(),
        ));
    }

    let reference_later = reference_output(backend, task, &prepared_later)?;
    let held_d = TrackedSubmit::submit(backend, task, prepared_later.batch.clone(), tracker)?;
    let completed_d = held_d
        .complete()
        .map_err(|err| ProbeError::Backend(backend_error("lifecycle_complete_later", &err)))?;
    expect_live(tracker, 0, "later_request_after_cancellation")?;
    let output_d = completed_output(completed_d, task, &prepared_later)?;
    let match_d = output_d == reference_later;
    steps.push(step(
        "later_request_after_cancellation",
        if match_d { "ok" } else { "failed" },
        format!(
            "completed a later request for case {} after cancellation; live_leases={}",
            prepared_later.case.id,
            tracker.live()
        ),
        Some(match_d),
    ));
    drop(output_d);
    if !match_d {
        return Err(ProbeError::Lifecycle(
            "later request did not match the synchronous reference bits".into(),
        ));
    }

    let held_e = TrackedSubmit::submit(backend, task, prepared_later.batch.clone(), tracker)?;
    expect_live(tracker, 1, "hold_e_before_invalid")?;
    let mut invalid = prepared_later.batch.clone();
    invalid.cumulative_seq_lengths = vec![0, 0];
    let invalid_step = match TrackedSubmit::submit(backend, task, invalid, tracker) {
        Ok(holder) => {
            drop(holder);
            failed_step(
                "invalid_admission_while_held",
                "invalid batch was admitted",
                StructuredError::new(
                    "invalid_admission_succeeded",
                    "lifecycle",
                    "invalid cumulative boundaries were admitted while a lease was held",
                ),
            )
        }
        Err(err) => {
            let mut typed = err.structured();
            typed.case_id = Some(prepared_later.case.id.clone());
            step(
                "invalid_admission_while_held",
                "ok",
                format!("typed rejection: {}: {}", typed.kind, typed.message),
                None,
            )
            .tap_error(Some(typed))
        }
    };
    expect_live(tracker, 1, "invalid_admission_while_held")?;
    let invalid_failed = invalid_step.status == "failed";
    steps.push(invalid_step);
    if invalid_failed {
        drop(held_e);
        return Err(ProbeError::Lifecycle(
            "invalid admission was not rejected while a lease was held".into(),
        ));
    }
    infinity_v3_backend_core::Backend::health(backend).map_err(|err| {
        ProbeError::Backend(backend_error("lifecycle_health_after_invalid", &err))
    })?;
    steps.push(step(
        "health_after_invalid_admission",
        "ok",
        "backend remained healthy after the typed admission rejection",
        None,
    ));
    let completed_e = held_e
        .complete()
        .map_err(|err| ProbeError::Backend(backend_error("lifecycle_complete_held", &err)))?;
    expect_live(tracker, 0, "complete_held_after_invalid_admission")?;
    let output_e = completed_output(completed_e, task, &prepared_later)?;
    let match_e = output_e == reference_later;
    steps.push(step(
        "complete_held_after_invalid_admission",
        if match_e { "ok" } else { "failed" },
        format!(
            "completed held case {} after the invalid admission; live_leases={}",
            prepared_later.case.id,
            tracker.live()
        ),
        Some(match_e),
    ));
    drop(output_e);
    if !match_e {
        return Err(ProbeError::Lifecycle(
            "held completion after invalid admission did not match the reference bits".into(),
        ));
    }

    let held_f = TrackedSubmit::submit(backend, task, prepared_later.batch.clone(), tracker)?;
    let completed_f = held_f
        .complete()
        .map_err(|err| ProbeError::Backend(backend_error("lifecycle_complete_recovery", &err)))?;
    expect_live(tracker, 0, "recovery_request")?;
    let output_f = completed_output(completed_f, task, &prepared_later)?;
    let match_f = output_f == reference_later;
    steps.push(step(
        "recovery_request",
        if match_f { "ok" } else { "failed" },
        format!(
            "recovery request for case {} completed; live_leases={}",
            prepared_later.case.id,
            tracker.live()
        ),
        Some(match_f),
    ));
    drop(output_f);
    drop(reference_later);
    if !match_f {
        return Err(ProbeError::Lifecycle(
            "recovery request did not match the synchronous reference bits".into(),
        ));
    }

    if task == Task::Classifier && model.head == HeadKind::SequenceClassification {
        let case = cases
            .iter()
            .find(|case| case.operation == Operation::ExpectRejectPredictTokens)
            .ok_or_else(|| {
                ProbeError::Lifecycle(
                    "sequence-head lifecycle requires an expect_reject_predict_tokens case".into(),
                )
            })?;
        let batch = build_batch(&case.validated);
        let rejection = match infinity_v3_backend_core::Backend::predict_tokens(backend, batch) {
            Ok(predictions) => {
                drop(predictions);
                failed_step(
                    "sequence_head_token_rejection",
                    format!("case {} token prediction unexpectedly completed", case.id),
                    StructuredError::new(
                        "expected_reject_succeeded",
                        "lifecycle",
                        "token prediction on this sequence head did not reject",
                    ),
                )
            }
            Err(err) => {
                let mut typed = backend_error("predict_tokens", &err);
                typed.case_id = Some(case.id.clone());
                step(
                    "sequence_head_token_rejection",
                    "ok",
                    format!("typed rejection: {}: {}", typed.kind, typed.message),
                    None,
                )
                .tap_error(Some(typed))
            }
        };
        let failed = rejection.status == "failed";
        steps.push(rejection);
        if failed {
            return Err(ProbeError::Lifecycle(
                "expected token prediction rejection did not happen".into(),
            ));
        }
        infinity_v3_backend_core::Backend::health(backend)
            .map_err(|err| ProbeError::Backend(backend_error("lifecycle_health_final", &err)))?;
        steps.push(step(
            "health_after_token_rejection",
            "ok",
            "backend remained healthy after the typed sequence-head token rejection",
            None,
        ));
    }

    expect_live(tracker, 0, "lifecycle_end")?;
    Ok(())
}

trait TapError {
    fn tap_error(self, error: Option<StructuredError>) -> Self;
}

impl TapError for LifecycleStep {
    fn tap_error(mut self, error: Option<StructuredError>) -> Self {
        if self.error.is_none() {
            self.error = error;
        }
        self
    }
}

fn require_cuda_zero() -> Result<String, ProbeError> {
    match env::var("CUDA_VISIBLE_DEVICES") {
        Ok(value) if value == "0" => Ok(value),
        Ok(value) => Err(ProbeError::Env(format!(
            "--run requires CUDA_VISIBLE_DEVICES=0, got {value:?}"
        ))),
        Err(_) => Err(ProbeError::Env(
            "--run requires CUDA_VISIBLE_DEVICES=0 (the variable is unset)".into(),
        )),
    }
}

fn base_report(cli: &Cli, model: Option<&ModelInfo>) -> Result<ProbeReport, ProbeError> {
    let output_dir = cli.output.display().to_string();
    Ok(ProbeReport {
        schema: SCHEMA.into(),
        mode: match cli.mode {
            Mode::CheckOnly => "check-only".into(),
            Mode::Run => "run".into(),
        },
        api: cli.api,
        task: cli.task,
        pool: cli.pool,
        dtype: "bfloat16".into(),
        engine_created: false,
        parser_only: cli.mode == Mode::CheckOnly,
        lifecycle: cli.lifecycle,
        cuda_visible_devices: env::var("CUDA_VISIBLE_DEVICES").unwrap_or_default(),
        status: "starting".into(),
        model: model.map(|info| info.reference.clone()),
        requests: None,
        source_capsule: source_capsule()?,
        output_dir,
        report_path: cli.output.join("report.json").display().to_string(),
        primary_error: None,
        cases: Vec::new(),
        lifecycle_report: None,
    })
}

fn write_failure_report(
    report: &mut ProbeReport,
    cli: &Cli,
    error: &ProbeError,
) -> Result<(), ProbeError> {
    report.status = "failed".into();
    report.primary_error = Some(error.structured());
    persist(report, cli)
}

fn setup(cli: &Cli) -> Result<(ProbeReport, ModelInfo, LoadedRequests), ProbeError> {
    prepare_fresh_dir(&cli.output)?;
    let mut report = base_report(cli, None)?;
    let model = match load_model_info(&cli.model) {
        Ok(model) => model,
        Err(error) => {
            write_failure_report(&mut report, cli, &error)?;
            return Err(error);
        }
    };
    report.model = Some(model.reference.clone());
    if let Err(error) = validate_task_model(cli.task, model.head) {
        write_failure_report(&mut report, cli, &error)?;
        return Err(error);
    }
    let requests_bytes = match read_capped(&cli.requests, MAX_REQUEST_BYTES) {
        Ok(bytes) => bytes,
        Err(error) => {
            write_failure_report(&mut report, cli, &error)?;
            return Err(error);
        }
    };
    let request_sha256 = sha256_hex(&requests_bytes);
    let line_count = requests_bytes
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.iter().all(|byte| byte.is_ascii_whitespace()))
        .count();
    report.requests = Some(RequestFileRef {
        path: cli.requests.display().to_string(),
        sha256: request_sha256,
        line_count,
        request_count: 0,
    });
    let loaded = match parse_requests(&requests_bytes, cli.task, &model) {
        Ok(loaded) => loaded,
        Err(error) => {
            write_failure_report(&mut report, cli, &error)?;
            return Err(error);
        }
    };
    if let Some(requests) = report.requests.as_mut() {
        requests.request_count = loaded.cases.len();
    }
    if let Err(error) = validate_lifecycle_contract(cli, &model, &loaded) {
        write_failure_report(&mut report, cli, &error)?;
        return Err(error);
    }
    Ok((report, model, loaded))
}

async fn process_run(
    engine: &Engine,
    loaded: &LoadedRequests,
    cli: &Cli,
    model: &ModelInfo,
    report: &mut ProbeReport,
) -> Result<Option<StructuredError>, ProbeError> {
    let mut failure: Option<StructuredError> = None;
    for (position, case) in loaded.cases.iter().enumerate() {
        let outcome = execute_case(engine, case, cli, model).await;
        match outcome {
            Ok(case_report) => {
                let failed = case_report.status == CaseStatus::Failed;
                let error = case_report.error.clone();
                report.cases.push(case_report);
                if failed {
                    failure = error;
                    persist(report, cli)?;
                    mark_skipped(report, &loaded.cases, position + 1, cli, model);
                    persist(report, cli)?;
                    break;
                }
                persist(report, cli)?;
            }
            Err(error) => {
                let mut case_report = base_case_report(case, cli, model)?;
                case_report.status = CaseStatus::Failed;
                case_report.error = Some(error.structured());
                report.cases.push(case_report);
                failure = Some(error.structured());
                persist(report, cli)?;
                mark_skipped(report, &loaded.cases, position + 1, cli, model);
                persist(report, cli)?;
                break;
            }
        }
    }

    if failure.is_none() && cli.lifecycle {
        match engine.core() {
            Some(backend) => {
                let mut steps = Vec::new();
                let tracker = LeaseTracker::new(2);
                let lifecycle_result = run_lifecycle(
                    backend,
                    &loaded.cases,
                    cli.task,
                    model,
                    &mut steps,
                    &tracker,
                );
                let lifecycle_error = lifecycle_result.err();
                if let Some(error) = lifecycle_error.as_ref() {
                    failure = Some(error.structured());
                }
                report.lifecycle_report = Some(LifecycleReport {
                    status: if lifecycle_error.is_some() {
                        "failed".into()
                    } else {
                        "passed".into()
                    },
                    task: cli.task,
                    head: model.head,
                    lease_capacity: tracker.capacity(),
                    observed_max_live_leases: tracker.max_observed(),
                    third_submission_while_two_live: false,
                    polling_loop: false,
                    capacity_note: "the probe tracker admits at most two live deferred leases; the lifecycle never submits a third while A/B are live and completion is backend-owned blocking complete(), not a polling loop".into(),
                    steps,
                    error: lifecycle_error.as_ref().map(ProbeError::structured),
                });
            }
            None => {
                failure = Some(StructuredError::new(
                    "lifecycle_requires_core",
                    "lifecycle",
                    "lifecycle requires the core API",
                ));
            }
        }
        persist(report, cli)?;
    }
    Ok(failure)
}

pub fn execute(args: Vec<String>) -> Result<(), ProbeError> {
    let cli = parse_cli(args)?;
    let (mut report, model, loaded) = setup(&cli)?;
    if cli.mode == Mode::CheckOnly {
        for case in &loaded.cases {
            report.cases.push(base_case_report(case, &cli, &model)?);
        }
        report.status = "validated".into();
        persist(&mut report, &cli)?;
        println!(
            "dense_probe: check-only parser_only=true engine_created=false cases={} report={}",
            loaded.cases.len(),
            report.report_path
        );
        return Ok(());
    }
    let cuda = match require_cuda_zero() {
        Ok(value) => value,
        Err(error) => {
            write_failure_report(&mut report, &cli, &error)?;
            return Err(error);
        }
    };
    report.cuda_visible_devices = cuda;
    report.status = "constructing".into();
    persist(&mut report, &cli)?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .map_err(|err| ProbeError::Config(format!("could not build runtime: {err}")))?;
    let selected_model_type = model_type(cli.task, cli.pool);
    let engine = match cli.api {
        Api::Factory => runtime.block_on(construct_factory(
            &cli.model,
            infinity_v3_backend::DType::Bfloat16,
            selected_model_type,
        )),
        Api::Core => construct_core(&cli.model, selected_model_type),
    };
    let engine = match engine {
        Ok(engine) => engine,
        Err(err) => {
            let error = ProbeError::Backend(backend_error("engine_constructor", &err));
            write_failure_report(&mut report, &cli, &error)?;
            return Err(error);
        }
    };
    report.cases.clear();
    report.engine_created = true;
    report.status = "running".into();
    persist(&mut report, &cli)?;

    let failure = runtime.block_on(process_run(&engine, &loaded, &cli, &model, &mut report))?;
    report.status = if failure.is_some() {
        "failed".into()
    } else {
        "passed".into()
    };
    report.primary_error = failure.clone();
    persist(&mut report, &cli)?;
    match failure {
        Some(error) => Err(ProbeError::Backend(error)),
        None => {
            println!(
                "dense_probe: run api={} cases={} status=passed report={}",
                cli.api.as_str(),
                loaded.cases.len(),
                report.report_path
            );
            Ok(())
        }
    }
}

pub fn main_entry() -> ExitCode {
    let raw: Vec<String> = env::args().skip(1).collect();
    if raw.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("{}", usage());
        return ExitCode::SUCCESS;
    }
    match execute(raw) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("dense_probe: {err}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifier_rejects_non_cls_pool() {
        let error = parse_cli([
            "--model".into(),
            "/tmp/model".into(),
            "--task".into(),
            "classifier".into(),
            "--pool".into(),
            "mean".into(),
            "--api".into(),
            "core".into(),
            "--requests".into(),
            "/tmp/requests.jsonl".into(),
            "--output".into(),
            "/tmp/out".into(),
            "--check-only".into(),
        ])
        .unwrap_err();
        assert!(error.to_string().contains("accept no --pool"));
    }
}
