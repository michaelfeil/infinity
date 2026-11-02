use crate::fixtures;
use crate::life;
use crate::metadata;
use crate::quality;
use crate::report::{write_report, Report};
use infinity_v3_backend_core::Pool;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Quality,
    LifeService,
    LifeCore,
    Metadata,
    ValidateOnly,
}

#[derive(Debug, Clone)]
pub struct Args {
    pub mode: Mode,
    pub model: Option<PathBuf>,
    pub pool: Option<Pool>,
    pub cases: PathBuf,
    pub report: PathBuf,
    pub limit: Option<usize>,
    pub validate_only: bool,
}

fn usage() -> &'static str {
    "usage: q35moe-unified-runtime <quality|life-service|life-core|metadata|validate-only> \
     --cases MANIFEST [--report FILE | --output-dir DIR] [--model DIR] \
     [--pool cls|mean|last-token] [--limit N] [--validate-only]"
}

fn parse_pool(value: &str) -> Result<Pool, String> {
    match value {
        "cls" => Ok(Pool::Cls),
        "mean" => Ok(Pool::Mean),
        "last-token" => Ok(Pool::LastToken),
        _ => Err(format!(
            "invalid --pool {value:?}; expected cls, mean or last-token"
        )),
    }
}

pub fn parse(args: &[String]) -> Result<Args, String> {
    if args.len() < 2 {
        return Err(usage().into());
    }
    let first = &args[1];
    if first.starts_with('-') {
        return Err(format!("mode must be the first argument\n{}", usage()));
    }
    let mut mode = match first.as_str() {
        "quality" => Mode::Quality,
        "life-service" => Mode::LifeService,
        "life-core" => Mode::LifeCore,
        "metadata" => Mode::Metadata,
        "validate-only" => Mode::ValidateOnly,
        other => return Err(format!("unknown mode {other:?}\n{}", usage())),
    };
    let mut seen = BTreeSet::new();
    let mut model = None;
    let mut pool = None;
    let mut cases = None;
    let mut report = None;
    let mut output_dir = None;
    let mut limit = None;
    let mut validate_only = false;
    let mut index = 2;
    while index < args.len() {
        let flag = &args[index];
        if !flag.starts_with("--") {
            return Err(format!("unexpected positional argument {flag:?}"));
        }
        if !seen.insert(flag.clone()) {
            return Err(format!("duplicate argument {flag}"));
        }
        match flag.as_str() {
            "--validate-only" => {
                validate_only = true;
                index += 1;
            }
            "--model" | "--pool" | "--cases" | "--report" | "--output-dir" | "--limit" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| format!("{flag} requires a value"))?;
                if value.starts_with("--") {
                    return Err(format!("{flag} requires a value"));
                }
                match flag.as_str() {
                    "--model" => model = Some(PathBuf::from(value)),
                    "--pool" => pool = Some(parse_pool(value)?),
                    "--cases" => cases = Some(PathBuf::from(value)),
                    "--report" => report = Some(PathBuf::from(value)),
                    "--output-dir" => output_dir = Some(PathBuf::from(value)),
                    "--limit" => {
                        let parsed = value
                            .parse::<usize>()
                            .map_err(|_| format!("invalid --limit {value:?}"))?;
                        if parsed == 0 {
                            return Err("--limit must be positive".into());
                        }
                        limit = Some(parsed);
                    }
                    _ => unreachable!(),
                }
                index += 2;
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    if validate_only || mode == Mode::ValidateOnly {
        mode = Mode::ValidateOnly;
    }
    if report.is_some() && output_dir.is_some() {
        return Err("use only one of --report or --output-dir".into());
    }
    let report = report
        .or_else(|| output_dir.map(|dir| dir.join("harness-report.json")))
        .unwrap_or_else(|| PathBuf::from("q35moe-report.json"));
    let cases = cases.unwrap_or_else(|| PathBuf::from("fixtures/manifest.json"));
    if !matches!(mode, Mode::ValidateOnly) && model.is_none() {
        return Err("--model is required outside validate-only".into());
    }
    if matches!(mode, Mode::Quality | Mode::LifeService | Mode::LifeCore) && pool.is_none() {
        return Err("--pool is required for quality and lifecycle modes".into());
    }
    Ok(Args {
        mode,
        model,
        pool,
        cases,
        report,
        limit,
        validate_only: mode == Mode::ValidateOnly,
    })
}

pub async fn run(args: &[String]) -> Result<(), String> {
    let args = parse(args)?;
    let mut report = Report::new(mode_name(args.mode));
    report.set_path(args.report.clone());
    report.model = args.model.as_ref().map(|path| path.display().to_string());
    report.pool = args.pool.clone().map(|pool| pool.to_string());
    let result = match args.mode {
        Mode::ValidateOnly => validate_only(&args, &mut report),
        Mode::Metadata => metadata::run(&args, &mut report),
        Mode::Quality => quality::run(&args, &mut report).await,
        Mode::LifeService => life::run_service(&args, &mut report).await,
        Mode::LifeCore => life::run_core(&args, &mut report).await,
    };
    match result {
        Ok(()) => report.finish("ok"),
        Err(error) => {
            report.errors.push(error.clone());
            report.finish("error");
            let _ = write_report(&args.report, &report);
            return Err(error);
        }
    }
    write_report(&args.report, &report)?;
    Ok(())
}

fn mode_name(mode: Mode) -> &'static str {
    match mode {
        Mode::Quality => "quality",
        Mode::LifeService => "life-service",
        Mode::LifeCore => "life-core",
        Mode::Metadata => "metadata",
        Mode::ValidateOnly => "validate-only",
    }
}

fn validate_only(args: &Args, report: &mut Report) -> Result<(), String> {
    let cases = fixtures::load_cases(&args.cases, args.limit)?;
    let valid = cases.iter().filter(|case| case.case.valid).count();
    let invalid = cases.len() - valid;
    report.phase(
        "fixtures",
        "ok",
        serde_json::json!({
            "manifest": args.cases.display().to_string(),
            "total": cases.len(),
            "valid": valid,
            "invalid": invalid,
            "model_loaded": false,
            "engine_count": 0,
            "gpu_path_taken": false,
        }),
    );
    Ok(())
}

pub fn ensure_model_is_directory(model: &Path) -> Result<(), String> {
    let metadata = std::fs::metadata(model)
        .map_err(|e| format!("missing model directory {}: {e}", model.display()))?;
    if !metadata.is_dir() {
        return Err(format!(
            "model path is not a directory: {}",
            model.display()
        ));
    }
    Ok(())
}
