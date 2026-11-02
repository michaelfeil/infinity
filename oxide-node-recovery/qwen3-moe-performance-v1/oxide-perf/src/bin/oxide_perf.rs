use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use infinity_backend_oxide::OxideBackend;
use infinity_v3_backend_core::{Backend, Embedding, Embeddings, ModelType};
use sha2::{Digest, Sha256};

use oxide_moe_perf_lib::{
    build_batch, expected_canonical, make_report, parse_batch_bytes, read_batch_bytes,
    read_model_width, validate_batch, validate_embeddings, DType, ExpectedCanonical, PoolArg,
    ProbeError, ValidatedBatch,
};

struct Case {
    id: String,
    batch_path: PathBuf,
    batch_sha256: String,
    validated: ValidatedBatch,
    batch: infinity_v3_backend_core::Batch,
    expected: ExpectedCanonical,
}

struct Timing {
    first_us: f64,
    warmup_us: Vec<f64>,
    timed_us: Vec<f64>,
    result_sha256: String,
    report: String,
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

fn pack_hash(
    expected: &ExpectedCanonical,
    embeddings: &Embeddings,
    width: usize,
) -> Result<String, ProbeError> {
    let mut hasher = Sha256::new();
    for (&index, entry) in &expected.entries {
        let embedding = embeddings
            .get(&index)
            .ok_or_else(|| ProbeError::Embedding(format!("missing embedding index {index}")))?;
        match (entry.kind, embedding) {
            (oxide_moe_perf_lib::ExpectedKind::Pooled, Embedding::Pooled(values)) => {
                if values.len() != width {
                    return Err(ProbeError::Embedding(format!(
                        "pooled index {index} width {} != {width}",
                        values.len()
                    )));
                }
                for value in values {
                    hasher.update(value.to_bits().to_le_bytes());
                }
            }
            (oxide_moe_perf_lib::ExpectedKind::Raw, Embedding::All(rows)) => {
                if rows.len() != entry.rows {
                    return Err(ProbeError::Embedding(format!(
                        "raw index {index} rows {} != {}",
                        rows.len(),
                        entry.rows
                    )));
                }
                for row in rows {
                    if row.len() != width {
                        return Err(ProbeError::Embedding(format!(
                            "raw index {index} row width {} != {width}",
                            row.len()
                        )));
                    }
                    for value in row {
                        hasher.update(value.to_bits().to_le_bytes());
                    }
                }
            }
            _ => {
                return Err(ProbeError::Embedding(format!(
                    "embedding kind mismatch at index {index}"
                )))
            }
        }
    }
    Ok(hex(&hasher.finalize()))
}

fn measure(
    backend: &OxideBackend,
    batch: &infinity_v3_backend_core::Batch,
    expected: &ExpectedCanonical,
    width: usize,
) -> Result<(f64, String, Embeddings), ProbeError> {
    let started = Instant::now();
    let call = backend
        .submit_embed(batch.clone())
        .map_err(|error| ProbeError::Backend(error.to_string()))?;
    let embeddings = call
        .complete()
        .map_err(|error| ProbeError::Backend(error.to_string()))?;
    let digest = pack_hash(expected, &embeddings, width)?;
    let elapsed_us = started.elapsed().as_secs_f64() * 1_000_000.0;
    Ok((elapsed_us, digest, embeddings))
}

fn load_cases(path: &Path) -> Result<(Vec<Case>, serde_json::Value), ProbeError> {
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(path).map_err(|source| ProbeError::Io {
            path: path.to_path_buf(),
            source,
        })?)?;
    let root = path.parent().unwrap_or_else(|| Path::new("."));
    let entries = manifest
        .get("cases")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| ProbeError::Config("cases manifest requires cases".into()))?;
    let mut cases = Vec::new();
    for entry in entries {
        let id = entry
            .get("id")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| ProbeError::Config("case id required".into()))?
            .to_string();
        let rel = entry
            .get("batch")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| ProbeError::Config(format!("case {id} batch required")))?;
        let batch_path = root.join(rel);
        let bytes = read_batch_bytes(&batch_path)?;
        let batch_sha256 = oxide_moe_perf_lib::sha256_hex(&bytes);
        let validated = validate_batch(parse_batch_bytes(&bytes)?)?;
        let expected = expected_canonical(&validated)?;
        let batch = build_batch(&validated);
        cases.push(Case {
            id,
            batch_path,
            batch_sha256,
            validated,
            batch,
            expected,
        });
    }
    Ok((cases, manifest))
}

fn run_case(
    backend: &OxideBackend,
    case: &Case,
    width: usize,
    pool: PoolArg,
    warmup: usize,
    iterations: usize,
    output_dir: &Path,
) -> Result<Timing, ProbeError> {
    let (first_us, first_hash, first_embeddings) =
        measure(backend, &case.batch, &case.expected, width)?;
    let outputs = validate_embeddings(&case.expected, &first_embeddings, width)?;
    drop(first_embeddings);
    let report = make_report(
        &case.validated,
        case.batch_sha256.clone(),
        width,
        Some(pool),
        DType::BFloat16,
        outputs,
        false,
    )?;
    let report_path = output_dir.join(format!("{}-report.json", case.id));
    fs::write(
        &report_path,
        serde_json::to_vec_pretty(&report).map_err(ProbeError::Json)?,
    )
    .map_err(|source| ProbeError::Io {
        path: report_path.clone(),
        source,
    })?;
    let mut warmup_us = Vec::with_capacity(warmup);
    for _ in 0..warmup {
        let (us, digest, _) = measure(backend, &case.batch, &case.expected, width)?;
        if digest != first_hash {
            return Err(ProbeError::Backend(format!(
                "{} warmup output differs from first call",
                case.id
            )));
        }
        warmup_us.push(us);
    }
    let mut timed_us = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let (us, digest, _) = measure(backend, &case.batch, &case.expected, width)?;
        if digest != first_hash {
            return Err(ProbeError::Backend(format!(
                "{} timed output differs from first call",
                case.id
            )));
        }
        timed_us.push(us);
    }
    Ok(Timing {
        first_us,
        warmup_us,
        timed_us,
        result_sha256: first_hash,
        report: report_path.display().to_string(),
    })
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut model = None;
    let mut pool = None;
    let mut cases_path = None;
    let mut output = None;
    let mut device = 0usize;
    let mut warmup = 20usize;
    let mut iterations = 200usize;
    let mut aba_cycles = 5usize;
    let mut index = 0;
    while index < args.len() {
        let key = args[index].as_str();
        let value = || {
            args.get(index + 1)
                .cloned()
                .ok_or_else(|| format!("{key} requires a value"))
        };
        match key {
            "--model" => model = Some(PathBuf::from(value().unwrap())),
            "--pool" => pool = Some(PoolArg::parse(&value().unwrap()).unwrap()),
            "--cases" => cases_path = Some(PathBuf::from(value().unwrap())),
            "--output" => output = Some(PathBuf::from(value().unwrap())),
            "--device" => device = value().unwrap().parse().unwrap(),
            "--warmup" => warmup = value().unwrap().parse().unwrap(),
            "--iterations" => iterations = value().unwrap().parse().unwrap(),
            "--aba-cycles" => aba_cycles = value().unwrap().parse().unwrap(),
            other => panic!("unknown argument {other}"),
        }
        index += 2;
    }
    let model = model.expect("--model required");
    let pool = pool.unwrap_or(PoolArg::Mean);
    let cases_path = cases_path.expect("--cases required");
    let output = output.expect("--output required");
    if output.exists() {
        panic!("refusing existing output directory {}", output.display());
    }
    let (cases, _manifest) = load_cases(&cases_path).unwrap();
    if cases.len() < 2 {
        panic!("paired ABA requires at least two cases");
    }
    let width = read_model_width(&model).unwrap();
    let backend = OxideBackend::new(
        &model,
        "bfloat16",
        ModelType::Embedding(pool.backend_pool()),
        device,
    )
    .unwrap();
    fs::create_dir_all(&output).unwrap();

    // Variable-shape ABA warm, separate from steady timing.
    let mut aba = Vec::new();
    for cycle in 0..aba_cycles {
        for idx in [0usize, 1usize] {
            let case = &cases[idx];
            let (us, digest, _) = measure(&backend, &case.batch, &case.expected, width).unwrap();
            aba.push(serde_json::json!({
                "cycle": cycle,
                "case": case.id,
                "elapsed_us": us,
                "result_sha256": digest,
            }));
        }
    }

    let mut summary_cases = Vec::new();
    for case in &cases {
        let timing = run_case(&backend, case, width, pool, warmup, iterations, &output).unwrap();
        let sorted = {
            let mut values = timing.timed_us.clone();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            values
        };
        let p50 = sorted[sorted.len() / 2];
        let p95 = sorted[(sorted.len() * 95 / 100).min(sorted.len() - 1)];
        summary_cases.push(serde_json::json!({
            "id": case.id,
            "batch": case.batch_path.display().to_string(),
            "batch_sha256": case.batch_sha256,
            "tokens": case.validated.file.input_ids.len(),
            "first_call_us": timing.first_us,
            "warmup": timing.warmup_us,
            "timed_us": timing.timed_us,
            "timed_count": timing.timed_us.len(),
            "p50_us": p50,
            "p95_us": p95,
            "mean_us": timing.timed_us.iter().sum::<f64>() / timing.timed_us.len() as f64,
            "result_sha256": timing.result_sha256,
            "report": timing.report,
        }));
    }
    let summary = serde_json::json!({
        "schema": "qwen3-moe-oxide-perf/v1",
        "backend": "public-oxide-normal",
        "model": model.display().to_string(),
        "pool": pool.as_str(),
        "device": device,
        "width": width,
        "cases_manifest": cases_path.display().to_string(),
        "cases_manifest_sha256": oxide_moe_perf_lib::sha256_hex(&fs::read(&cases_path).unwrap()),
        "warmup": warmup,
        "iterations": iterations,
        "aba_cycles": aba_cycles,
        "aba": aba,
        "graphs": false,
        "cases": summary_cases,
    });
    let summary_path = output.join("summary.json");
    fs::write(&summary_path, serde_json::to_vec_pretty(&summary).unwrap()).unwrap();
    println!("{}", summary_path.display());
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxide_moe_perf_lib::{ExpectedEntry, ExpectedKind};
    use std::collections::BTreeMap;

    #[test]
    fn pack_hash_is_stable_and_detects_value_change() {
        let mut expected = ExpectedCanonical {
            entries: BTreeMap::new(),
            raw_overrides_pooled: Vec::new(),
            duplicate_selection_indices: Vec::new(),
        };
        expected.entries.insert(
            0,
            ExpectedEntry {
                kind: ExpectedKind::Pooled,
                rows: 1,
            },
        );
        expected.entries.insert(
            1,
            ExpectedEntry {
                kind: ExpectedKind::Raw,
                rows: 2,
            },
        );
        let mut embeddings = Embeddings::default();
        embeddings.insert(0, Embedding::Pooled(vec![1.0, 2.0]));
        embeddings.insert(1, Embedding::All(vec![vec![3.0, 4.0], vec![5.0, 6.0]]));
        let first = pack_hash(&expected, &embeddings, 2).unwrap();
        assert_eq!(first, pack_hash(&expected, &embeddings, 2).unwrap());
        embeddings.insert(1, Embedding::All(vec![vec![3.0, 4.0], vec![5.0, 7.0]]));
        assert_ne!(first, pack_hash(&expected, &embeddings, 2).unwrap());
    }
}
