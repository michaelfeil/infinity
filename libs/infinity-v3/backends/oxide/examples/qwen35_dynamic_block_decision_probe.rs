// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Dynamic-prefix B1 block decision probe. One backend retains one weight set
//! and one Work across actual per-request prefixes at the same token count.
//! Finite checks only: no model-quality, admission, cache or performance claim.
use infinity_backend_oxide::{OxideBackend, Qwen35TextOptions};
use infinity_v3_backend_core::{Backend, Batch, DecisionInput, ModelType};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

type R<T> = Result<T, String>;
const SCHEMA_IN: &str = "infinity-onejev-physical-cases/v1";
const SCHEMA_OUT: &str = "infinity-qwen35-dynamic-block-decision-probe/v1";
const REPEATS: usize = 8;
const USAGE: &str = "qwen35_dynamic_block_decision_probe MODEL FA4_ROOT GDN_RAW GDN_PROVIDER GDN_AUX CASES_JSON OUTPUT_JSON [--validate-only] [--case-index N] [--serving]";

#[derive(Clone, Debug)]
struct PhysicalCase {
    id: Value,
    input_ids: Vec<u32>,
    position_ids: Vec<u32>,
    token_type_ids: Vec<u32>,
    cumulative_seq_lengths: Vec<u32>,
    max_length: u32,
    prefix_length: usize,
    token_ids: Vec<u32>,
}

fn exact_u32(value: &Value, label: &str) -> R<u32> {
    value
        .as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| format!("{label} must be an exact u32"))
}

fn u32_array(object: &serde_json::Map<String, Value>, key: &str) -> R<Vec<u32>> {
    object
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("case needs {key} array"))?
        .iter()
        .map(|value| exact_u32(value, key))
        .collect()
}

fn field_u32(object: &serde_json::Map<String, Value>, key: &str) -> R<u32> {
    exact_u32(
        object.get(key).ok_or_else(|| format!("case needs {key}"))?,
        key,
    )
}

fn parse_case(value: &Value) -> R<PhysicalCase> {
    let object = value.as_object().ok_or("case must be an object")?;
    let prefix = field_u32(object, "prefix_length")?;
    let case = PhysicalCase {
        id: object.get("id").cloned().ok_or("case needs id")?,
        input_ids: u32_array(object, "input_ids")?,
        position_ids: u32_array(object, "position_ids")?,
        token_type_ids: u32_array(object, "token_type_ids")?,
        cumulative_seq_lengths: u32_array(object, "cumulative_seq_lengths")?,
        max_length: field_u32(object, "max_length")?,
        prefix_length: usize::try_from(prefix).map_err(|_| "prefix_length too large")?,
        token_ids: u32_array(object, "token_ids")?,
    };
    validate_case(&case)?;
    Ok(case)
}

fn validate_case(case: &PhysicalCase) -> R<()> {
    let tokens = case.input_ids.len();
    if tokens == 0 || tokens > 262_144 {
        return Err("token count must be 1..=262144".into());
    }
    let total = tokens as u32;
    if case.cumulative_seq_lengths != [0, total] {
        return Err("case must be one packed B1 sequence [0,T]".into());
    }
    if case.position_ids.len() != tokens || case.token_type_ids.len() != tokens {
        return Err("position/token-type count must match input_ids".into());
    }
    if case.position_ids.iter().copied().ne(0..total) {
        return Err("position_ids must be contiguous from zero without padding".into());
    }
    if case.token_type_ids.iter().any(|&kind| kind != 0) {
        return Err("token_type_ids must be zero".into());
    }
    if case.max_length != total {
        return Err("max_length must equal T without repadding".into());
    }
    if case.prefix_length == 0 || case.prefix_length > tokens {
        return Err("prefix_length must be positive and at most T".into());
    }
    if case.token_ids.is_empty() || case.token_ids.len() > 26 {
        return Err("token_ids must contain 1..=26 options".into());
    }
    Ok(())
}

fn to_batch(case: &PhysicalCase) -> Batch {
    Batch {
        multimodal: vec![],
        input_ids: case.input_ids.clone(),
        token_type_ids: case.token_type_ids.clone(),
        position_ids: case.position_ids.clone(),
        cumulative_seq_lengths: case.cumulative_seq_lengths.clone(),
        max_length: case.max_length,
        pooled_indices: vec![],
        raw_indices: vec![],
        compact_input_ids: None,
        compact_position_ids: None,
        scatter_unfold: None,
        fold_gather: None,
        tokens: vec![],
        offsets: vec![],
    }
}

fn to_request(case: &PhysicalCase) -> DecisionInput {
    DecisionInput::BlockOptionTokens {
        token_ids: case.token_ids.clone(),
        prefix_length: case.prefix_length,
    }
}

fn load_cases(path: &Path) -> R<Vec<PhysicalCase>> {
    let root: Value = serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    if root.get("schema").and_then(Value::as_str) != Some(SCHEMA_IN) {
        return Err(format!("expected schema {SCHEMA_IN}"));
    }
    root.get("cases")
        .and_then(Value::as_array)
        .ok_or("cases must be an array")?
        .iter()
        .map(parse_case)
        .collect()
}

fn ensure_fresh(path: &Path) -> R<()> {
    (!path.exists())
        .then_some(())
        .ok_or_else(|| "Fresh output required".into())
}

fn write_report(path: &Path, report: &Value) -> R<()> {
    let bytes = serde_json::to_vec_pretty(report).map_err(|e| e.to_string())?;
    std::fs::write(path, bytes).map_err(|e| e.to_string())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.first().is_some_and(|arg| arg == "--help") {
        println!("{USAGE}");
        return Ok(());
    }
    let get = |i| args.get(i).map(String::as_str).ok_or(USAGE);
    let model = Path::new(get(0)?);
    let options = Qwen35TextOptions {
        causal: false,
        fa4_root: PathBuf::from(get(1)?),
        gdn_raw_kernels: PathBuf::from(get(2)?),
        gdn_provider: PathBuf::from(get(3)?),
        gdn_aux_kernels: PathBuf::from(get(4)?),
    };
    let cases_path = Path::new(get(5)?);
    let output = PathBuf::from(get(6)?);
    ensure_fresh(&output)?;
    let (mut validate_only, mut case_index): (bool, Option<usize>) = (false, None);
    let mut serving = false;
    let mut index = 7;
    while index < args.len() {
        match args[index].as_str() {
            "--validate-only" => validate_only = true,
            "--serving" => serving = true,
            "--case-index" => {
                index += 1;
                case_index = Some(
                    args.get(index)
                        .ok_or("--case-index needs N")?
                        .parse()
                        .map_err(|_| "--case-index must be an integer")?,
                );
            }
            other => return Err(format!("unknown argument {other}").into()),
        }
        index += 1;
    }
    let all = load_cases(cases_path)?;
    if all.is_empty() {
        return Err("cases must not be empty".into());
    }
    let selected = match case_index {
        Some(i) => vec![all
            .into_iter()
            .nth(i)
            .ok_or_else(|| format!("case index {i} out of range"))?],
        None => all,
    };

    if serving {
        for (name, expected) in [
            (
                infinity_v3_config::infinity_env!("FA4_ROOT"),
                &options.fa4_root,
            ),
            (
                infinity_v3_config::infinity_env!("OXIDE_GDN_RAW_KERNELS"),
                &options.gdn_raw_kernels,
            ),
            (
                infinity_v3_config::infinity_env!("OXIDE_GDN_PROVIDER"),
                &options.gdn_provider,
            ),
            (
                infinity_v3_config::infinity_env!("OXIDE_GDN_AUX_KERNELS"),
                &options.gdn_aux_kernels,
            ),
        ] {
            let actual =
                std::env::var_os(name).ok_or_else(|| format!("serving control requires {name}"))?;
            if Path::new(&actual).canonicalize()? != expected.canonicalize()? {
                return Err(format!("serving {name} differs from the pinned argument").into());
            }
        }
        OxideBackend::validate_decision_checkpoint(model, "bfloat16")?;
    }

    if validate_only {
        let cases = selected
            .iter()
            .map(|case| json!({"id":case.id,"status":"cpu-validation-only","tokens":to_batch(case).input_ids.len(),"prefix_length":case.prefix_length,"token_ids":case.token_ids}))
            .collect::<Vec<_>>();
        let report = json!({
            "schema": SCHEMA_OUT,
            "status": "cpu-validation-only",
            "engine_created": false,
            "CUDA": false,
            "cases": cases,
            "dtype": "bfloat16",
            "dynamic_fp8": false,
            "CUDA_graphs": false,
            "dynamic_prefix": true,
            "serving_constructor": serving,
            "scope": "Finite-only CPU validation; one backend would serve every case; no model-quality, admission, cache or performance claim",
        });
        write_report(&output, &report)?;
        println!(
            "{}",
            json!({"status":"cpu-validation-only","cases":selected.len(),"output":output})
        );
        return Ok(());
    }

    eprintln!("constructor:start");
    let backend = if serving {
        OxideBackend::new(model, "bfloat16", ModelType::Decision, 0)?
    } else {
        OxideBackend::new_qwen35_dynamic_block_decision_candidate(model, 0, options)?
    };
    eprintln!("constructor:end");

    let mut results = Vec::new();
    for case in &selected {
        eprintln!(
            "request:prefix case={} prefix={} tokens={}",
            case.id,
            case.prefix_length,
            case.input_ids.len()
        );
        let (batch, request) = (to_batch(case), to_request(case));
        let (mut expected, mut logits, mut probability) = (None, Vec::new(), 0.0f32);
        let mut bits_exact = true;
        for repeat in 0..REPEATS {
            eprintln!("repeat:start case={} {}/{}", case.id, repeat + 1, REPEATS);
            let mut outputs = if serving {
                // The real serving queue supplies selectors independently of
                // the Decision protocol. Exercise the public entry with them.
                let mut physical = batch.clone();
                physical.raw_indices.push(0);
                backend.decide(physical, vec![request.clone()])?
            } else {
                backend
                    .submit_qwen35_block_decision_candidate(batch.clone(), vec![request.clone()])?
                    .complete()?
            };
            if outputs.len() != 1 {
                return Err(format!("case {} returned {} outputs", case.id, outputs.len()).into());
            }
            let output = outputs.remove(0);
            if output.logits.len() != case.token_ids.len() {
                return Err(format!("case {} logits count mismatch", case.id).into());
            }
            if !output.logits.iter().all(|value| value.is_finite()) {
                return Err(format!("case {} produced nonfinite logits", case.id).into());
            }
            if output.action_probability != 1.0 {
                return Err(format!("case {} action_probability is not 1", case.id).into());
            }
            let bits = output
                .logits
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>();
            match &expected {
                Some(reference) => bits_exact &= *reference == bits,
                None => {
                    expected = Some(bits);
                    logits = output.logits;
                    probability = output.action_probability;
                }
            }
            backend.health()?;
            eprintln!("repeat:end case={} {}/{}", case.id, repeat + 1, REPEATS);
        }
        if !bits_exact {
            return Err(format!("case {} F32 bits changed across repeats", case.id).into());
        }
        let argmax = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
            .map(|(i, _)| i)
            .unwrap_or(0);
        results.push(json!({"id":case.id,"status":"passed-finite-only","tokens":case.input_ids.len(),"prefix_length":case.prefix_length,"token_ids":case.token_ids,"logits":logits,"argmax":argmax,"action_probability":probability,"repeat_F32_bits_exact":bits_exact,"repeats":REPEATS}));
    }
    let report = json!({
        "schema": SCHEMA_OUT,
        "status": "passed-finite-only",
        "model": model,
        "cases": results,
        "dtype": "bfloat16",
        "dynamic_fp8": false,
        "CUDA_graphs": false,
        "dynamic_prefix": true,
        "serving_constructor": serving,
        "scope": "Finite-only experimental dynamic-prefix block decision probe; one backend retained across cases; not a model-quality, admission, cache or performance claim",
    });
    write_report(&output, &report)?;
    println!(
        "{}",
        json!({"status":"passed-finite-only","cases":results.len(),"output":output})
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Value {
        json!({"id":"case-1","input_ids":[10,11,12],"position_ids":[0,1,2],"token_type_ids":[0,0,0],"cumulative_seq_lengths":[0,3],"max_length":3,"prefix_length":2,"token_ids":[32,33],"kind":"choice","provenance":{"source":"exporter"}})
    }

    fn temp_path() -> PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "qwen35-dynamic-probe-{}-{unique}.json",
            std::process::id()
        ))
    }

    #[test]
    fn prefixes_must_be_positive_and_inside_the_sequence() {
        let mut value = fixture();
        value["prefix_length"] = json!(0);
        assert!(parse_case(&value).is_err());
        value["prefix_length"] = json!(4);
        assert!(parse_case(&value).is_err());
        value["prefix_length"] = json!(3);
        assert!(parse_case(&value).is_ok());
    }

    #[test]
    fn boundaries_and_lengths_must_be_exact_b1() {
        let mut value = fixture();
        value["cumulative_seq_lengths"] = json!([0, 1, 3]);
        assert!(parse_case(&value).is_err());
        value["cumulative_seq_lengths"] = json!([0, 2]);
        assert!(parse_case(&value).is_err());
        value = fixture();
        value["max_length"] = json!(2);
        assert!(parse_case(&value).is_err());
    }

    #[test]
    fn ids_must_be_exact_nonfractional_u32() {
        let mut value = fixture();
        value["input_ids"] = json!([10, 11, 12.5]);
        assert!(parse_case(&value).is_err());
        value["input_ids"] = json!([10, 11, 12]);
        value["token_ids"] = json!([32, 4_294_967_296u64]);
        assert!(parse_case(&value).is_err());
    }

    #[test]
    fn batch_constructor_carries_the_parsed_case() {
        let case = parse_case(&fixture()).unwrap();
        let batch = to_batch(&case);
        assert_eq!(batch.input_ids, [10, 11, 12]);
        assert_eq!(batch.position_ids, [0, 1, 2]);
        assert_eq!(batch.token_type_ids, [0, 0, 0]);
        assert_eq!(batch.cumulative_seq_lengths, [0, 3]);
        assert_eq!(batch.max_length, 3);
        assert!(batch.pooled_indices.is_empty() && batch.raw_indices.is_empty());
        assert!(matches!(
            to_request(&case),
            DecisionInput::BlockOptionTokens { token_ids, prefix_length: 2 } if token_ids == [32, 33]
        ));
    }

    #[test]
    fn same_tokens_accept_different_actual_prefixes() {
        for prefix in [1usize, 2, 3] {
            let mut value = fixture();
            value["prefix_length"] = json!(prefix);
            assert_eq!(parse_case(&value).unwrap().prefix_length, prefix);
        }
    }

    #[test]
    fn existing_output_is_never_overwritten() {
        let path = temp_path();
        assert!(ensure_fresh(&path).is_ok());
        std::fs::write(&path, b"{}").unwrap();
        assert!(ensure_fresh(&path).is_err());
        let _ = std::fs::remove_file(path);
    }
}
