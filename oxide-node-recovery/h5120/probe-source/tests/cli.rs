use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);

fn case_dir(name: &str) -> PathBuf {
    let unique = NEXT.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "dense-probe-test-{}-{unique}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).expect("test directory");
    path
}

fn write_text(path: &Path, text: &str) {
    fs::write(path, text).expect("write test file");
}

fn model_dir(root: &Path, config: &serde_json::Value) -> PathBuf {
    let path = root.join("model");
    fs::create_dir_all(&path).expect("model dir");
    write_text(
        &path.join("config.json"),
        &serde_json::to_string_pretty(config).expect("config json"),
    );
    path
}

fn causal_lm() -> serde_json::Value {
    serde_json::json!({
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 8,
        "vocab_size": 64,
        "max_position_embeddings": 64,
        "torch_dtype": "bfloat16"
    })
}

fn sequence_head() -> serde_json::Value {
    serde_json::json!({
        "model_type": "qwen3",
        "architectures": ["Qwen3ForSequenceClassification"],
        "hidden_size": 8,
        "vocab_size": 64,
        "max_position_embeddings": 64,
        "torch_dtype": "float16",
        "id2label": {"0": "no", "1": "yes"}
    })
}

fn token_head() -> serde_json::Value {
    serde_json::json!({
        "model_type": "qwen3",
        "architectures": ["Qwen3ForTokenClassification"],
        "hidden_size": 8,
        "vocab_size": 64,
        "max_position_embeddings": 64,
        "torch_dtype": "bfloat16",
        "id2label": {"0": "O", "1": "B"}
    })
}

fn run(args: &[&str], cwd: &Path, cuda: Option<&str>) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_dense_probe"));
    command.args(args).current_dir(cwd);
    match cuda {
        Some(value) => {
            command.env("CUDA_VISIBLE_DEVICES", value);
        }
        None => {
            command.env_remove("CUDA_VISIBLE_DEVICES");
        }
    }
    command.output().expect("run dense_probe")
}

fn s(value: &Path) -> String {
    value.display().to_string()
}

fn report(output: &Path) -> serde_json::Value {
    let bytes = fs::read(output.join("report.json")).expect("report bytes");
    serde_json::from_slice(&bytes).expect("report json")
}

const EMBED_TWO: &str = "{\"id\":\"one\",\"operation\":\"embed\",\"batch\":{\"input_ids\":[1,2],\"position_ids\":[0,1],\"token_type_ids\":[0,0],\"cumulative_seq_lengths\":[0,2],\"pooled_indices\":[0],\"raw_indices\":[],\"max_length\":2}}\n\
{\"id\":\"two\",\"operation\":\"embed\",\"batch\":{\"input_ids\":[3,4,5],\"position_ids\":[0,1,2],\"token_type_ids\":[0,0,0],\"cumulative_seq_lengths\":[0,3],\"pooled_indices\":[],\"raw_indices\":[0],\"max_length\":3}}\n";

#[test]
fn check_only_reports_parser_only_and_writes_per_case_json() {
    let root = case_dir("check-only");
    let model = model_dir(&root, &causal_lm());
    let requests = root.join("requests.jsonl");
    write_text(&requests, EMBED_TWO);
    let output = root.join("out");
    let result = run(
        &[
            "--model",
            &s(&model),
            "--task",
            "embedding",
            "--pool",
            "mean",
            "--api",
            "factory",
            "--requests",
            &s(&requests),
            "--output",
            &s(&output),
            "--check-only",
        ],
        &root,
        Some(""),
    );
    assert!(
        result.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report = report(&output);
    assert_eq!(report["schema"], "infinity-dense-unified-probe/v2");
    assert_eq!(report["mode"], "check-only");
    assert_eq!(report["engine_created"], false);
    assert_eq!(report["parser_only"], true);
    assert_eq!(report["status"], "validated");
    assert_eq!(report["model"]["head"], "causal_lm");
    assert_eq!(report["model"]["dtype_conversion"], "none");
    assert_eq!(report["cases"].as_array().map(Vec::len), Some(2));
    assert_eq!(report["cases"][0]["canonical"]["entry_count"], 1);
    assert_eq!(report["cases"][1]["layout"]["raw_count"], 1);
    assert!(report["cases"][0].get("selection").is_none());
    assert!(output.join("cases").join("one.json").is_file());
    assert!(output.join("cases").join("two.json").is_file());
    let encoded = fs::read_to_string(output.join("cases").join("one.json")).expect("case json");
    assert!(encoded.contains("\"physical_batch\""));
    assert!(encoded.contains("\"batch_sha256\""));
    assert!(!encoded.contains("\"result\":"));
    assert!(!encoded.contains("\"width\":"));
}

#[test]
fn check_only_rejects_malformed_cumulative_before_engine() {
    let root = case_dir("malformed");
    let model = model_dir(&root, &causal_lm());
    let requests = root.join("requests.jsonl");
    write_text(
        &requests,
        "{\"id\":\"bad\",\"operation\":\"embed\",\"batch\":{\"input_ids\":[1,2],\"position_ids\":[0,1],\"token_type_ids\":[0,0],\"cumulative_seq_lengths\":[0,2,2],\"pooled_indices\":[0],\"raw_indices\":[],\"max_length\":2}}\n",
    );
    let output = root.join("out");
    let result = run(
        &[
            "--model",
            &s(&model),
            "--task",
            "embedding",
            "--pool",
            "cls",
            "--api",
            "core",
            "--requests",
            &s(&requests),
            "--output",
            &s(&output),
            "--check-only",
        ],
        &root,
        Some(""),
    );
    assert_eq!(result.status.code(), Some(2));
    let report = report(&output);
    assert_eq!(report["status"], "failed");
    assert_eq!(report["engine_created"], false);
    assert_eq!(report["primary_error"]["line_number"], 1);
    assert!(report["primary_error"]["message"]
        .as_str()
        .unwrap_or_default()
        .contains("strictly increase"));
}

#[test]
fn check_only_rejects_operation_not_supported_by_task() {
    let root = case_dir("operation");
    let model = model_dir(&root, &causal_lm());
    let requests = root.join("requests.jsonl");
    write_text(
        &requests,
        "{\"id\":\"wrong\",\"operation\":\"predict\",\"batch\":{\"input_ids\":[1],\"position_ids\":[0],\"token_type_ids\":[0],\"cumulative_seq_lengths\":[0,1],\"pooled_indices\":[0],\"raw_indices\":[],\"max_length\":1}}\n",
    );
    let output = root.join("out");
    let result = run(
        &[
            "--model",
            &s(&model),
            "--task",
            "embedding",
            "--pool",
            "cls",
            "--api",
            "factory",
            "--requests",
            &s(&requests),
            "--output",
            &s(&output),
            "--check-only",
        ],
        &root,
        Some(""),
    );
    assert_eq!(result.status.code(), Some(2));
    let report = report(&output);
    assert!(report["primary_error"]["message"]
        .as_str()
        .unwrap_or_default()
        .contains("not supported by task"));
}

#[test]
fn run_without_cuda_visibility_fails_before_engine_creation() {
    let root = case_dir("cuda-guard");
    let model = model_dir(&root, &causal_lm());
    let requests = root.join("requests.jsonl");
    write_text(
        &requests,
        "{\"id\":\"one\",\"operation\":\"embed\",\"batch\":{\"input_ids\":[1],\"position_ids\":[0],\"token_type_ids\":[0],\"cumulative_seq_lengths\":[0,1],\"pooled_indices\":[0],\"raw_indices\":[],\"max_length\":1}}\n",
    );
    let output = root.join("out");
    let result = run(
        &[
            "--model",
            &s(&model),
            "--task",
            "embedding",
            "--pool",
            "cls",
            "--api",
            "core",
            "--requests",
            &s(&requests),
            "--output",
            &s(&output),
            "--run",
        ],
        &root,
        None,
    );
    assert_eq!(result.status.code(), Some(2));
    let report = report(&output);
    assert_eq!(report["engine_created"], false);
    assert_eq!(report["primary_error"]["kind"], "env");
    assert!(report["primary_error"]["message"]
        .as_str()
        .unwrap_or_default()
        .contains("CUDA_VISIBLE_DEVICES=0"));
}

#[test]
fn unknown_option_is_rejected_before_any_output() {
    let root = case_dir("unknown");
    let model = model_dir(&root, &causal_lm());
    let requests = root.join("requests.jsonl");
    write_text(&requests, "{}\n");
    let output = root.join("out");
    let result = run(
        &[
            "--model",
            &s(&model),
            "--task",
            "embedding",
            "--pool",
            "cls",
            "--api",
            "factory",
            "--requests",
            &s(&requests),
            "--output",
            &s(&output),
            "--check-only",
            "--bogus",
        ],
        &root,
        Some(""),
    );
    assert_eq!(result.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&result.stderr).contains("unknown option --bogus"));
    assert!(!output.exists());
}

#[test]
fn classifier_task_rejects_causal_lm_metadata() {
    let root = case_dir("classifier-causal");
    let model = model_dir(&root, &causal_lm());
    let requests = root.join("requests.jsonl");
    write_text(
        &requests,
        "{\"id\":\"score\",\"operation\":\"predict\",\"batch\":{\"input_ids\":[1],\"position_ids\":[0],\"token_type_ids\":[0],\"cumulative_seq_lengths\":[0,1],\"pooled_indices\":[0],\"raw_indices\":[],\"max_length\":1}}\n",
    );
    let output = root.join("out");
    let result = run(
        &[
            "--model",
            &s(&model),
            "--task",
            "classifier",
            "--api",
            "core",
            "--requests",
            &s(&requests),
            "--output",
            &s(&output),
            "--check-only",
        ],
        &root,
        Some(""),
    );
    assert_eq!(result.status.code(), Some(2));
    let report = report(&output);
    assert_eq!(report["model"]["head"], "causal_lm");
    assert!(report["primary_error"]["message"]
        .as_str()
        .unwrap_or_default()
        .contains("classifier task requires"));
}

#[test]
fn embedding_task_rejects_sequence_classifier_metadata() {
    let root = case_dir("embedding-seq");
    let model = model_dir(&root, &sequence_head());
    let requests = root.join("requests.jsonl");
    write_text(&requests, EMBED_TWO);
    let output = root.join("out");
    let result = run(
        &[
            "--model",
            &s(&model),
            "--task",
            "embedding",
            "--pool",
            "mean",
            "--api",
            "core",
            "--requests",
            &s(&requests),
            "--output",
            &s(&output),
            "--check-only",
        ],
        &root,
        Some(""),
    );
    assert_eq!(result.status.code(), Some(2));
    let report = report(&output);
    assert_eq!(report["model"]["head"], "sequence_classification");
    assert!(report["primary_error"]["message"]
        .as_str()
        .unwrap_or_default()
        .contains("ordinary embedding admission rejects"));
}

#[test]
fn reranker_check_only_annotates_sequence_selection_and_dtype_conversion() {
    let root = case_dir("reranker");
    let model = model_dir(&root, &sequence_head());
    let requests = root.join("requests.jsonl");
    write_text(
        &requests,
        "{\"id\":\"score\",\"operation\":\"predict\",\"batch\":{\"input_ids\":[1,2,0,3,4,5],\"position_ids\":[0,1,2,0,1,2],\"token_type_ids\":[0,0,0,0,0,0],\"cumulative_seq_lengths\":[0,3,6],\"pooled_indices\":[0,1],\"raw_indices\":[],\"max_length\":3}}\n\
         {\"id\":\"reject\",\"operation\":\"expect_reject_predict_tokens\",\"batch\":{\"input_ids\":[1,2,3],\"position_ids\":[0,1,2],\"token_type_ids\":[0,0,0],\"cumulative_seq_lengths\":[0,3],\"pooled_indices\":[],\"raw_indices\":[0],\"max_length\":3}}\n",
    );
    let output = root.join("out");
    let result = run(
        &[
            "--model",
            &s(&model),
            "--task",
            "classifier",
            "--api",
            "core",
            "--requests",
            &s(&requests),
            "--output",
            &s(&output),
            "--check-only",
        ],
        &root,
        Some(""),
    );
    assert!(
        result.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report = report(&output);
    assert_eq!(report["model"]["head"], "sequence_classification");
    assert_eq!(report["model"]["head_labels"], 2);
    assert_eq!(report["model"]["stored_dtype"], "float16");
    assert_eq!(report["model"]["runtime_dtype"], "bfloat16");
    assert_eq!(report["model"]["dtype_conversion"], "f16_to_bf16");
    assert_eq!(report["cases"][0]["selection"]["mode"], "end_row");
    assert_eq!(
        report["cases"][0]["selection"]["selected_row_ids"],
        serde_json::json!([2, 5])
    );
    assert_eq!(
        report["cases"][1]["operation"],
        "expect_reject_predict_tokens"
    );
}

#[test]
fn reranker_predict_tokens_operation_is_typed_rejected() {
    let root = case_dir("reranker-token-op");
    let model = model_dir(&root, &sequence_head());
    let requests = root.join("requests.jsonl");
    write_text(
        &requests,
        "{\"id\":\"token\",\"operation\":\"predict_tokens\",\"batch\":{\"input_ids\":[1,2,3],\"position_ids\":[0,1,2],\"token_type_ids\":[0,0,0],\"cumulative_seq_lengths\":[0,3],\"pooled_indices\":[],\"raw_indices\":[0],\"max_length\":3}}\n",
    );
    let output = root.join("out");
    let result = run(
        &[
            "--model",
            &s(&model),
            "--task",
            "classifier",
            "--api",
            "core",
            "--requests",
            &s(&requests),
            "--output",
            &s(&output),
            "--check-only",
        ],
        &root,
        Some(""),
    );
    assert_eq!(result.status.code(), Some(2));
    let report = report(&output);
    assert_eq!(report["primary_error"]["kind"], "operation");
    assert!(report["primary_error"]["message"]
        .as_str()
        .unwrap_or_default()
        .contains("sequence classification head rejects Native token readout"));
}

#[test]
fn token_head_allows_predict_tokens_and_rejects_false_expectation() {
    let root = case_dir("token-head");
    let model = model_dir(&root, &token_head());
    let requests = root.join("requests.jsonl");
    write_text(
        &requests,
        "{\"id\":\"tokens\",\"operation\":\"predict_tokens\",\"batch\":{\"input_ids\":[1,2,3],\"position_ids\":[0,1,2],\"token_type_ids\":[0,0,0],\"cumulative_seq_lengths\":[0,3],\"pooled_indices\":[],\"raw_indices\":[0],\"max_length\":3}}\n",
    );
    let output = root.join("out");
    let result = run(
        &[
            "--model",
            &s(&model),
            "--task",
            "classifier",
            "--api",
            "core",
            "--requests",
            &s(&requests),
            "--output",
            &s(&output),
            "--check-only",
        ],
        &root,
        Some(""),
    );
    assert!(
        result.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let ok_report = report(&output);
    assert_eq!(ok_report["model"]["head"], "token_classification");
    assert_eq!(ok_report["cases"][0]["selection"]["mode"], "all_tokens");

    let wrong = root.join("wrong.jsonl");
    write_text(
        &wrong,
        "{\"id\":\"reject\",\"operation\":\"expect_reject_predict_tokens\",\"batch\":{\"input_ids\":[1,2,3],\"position_ids\":[0,1,2],\"token_type_ids\":[0,0,0],\"cumulative_seq_lengths\":[0,3],\"pooled_indices\":[],\"raw_indices\":[0],\"max_length\":3}}\n",
    );
    let output = root.join("out2");
    let result = run(
        &[
            "--model",
            &s(&model),
            "--task",
            "classifier",
            "--api",
            "core",
            "--requests",
            &s(&wrong),
            "--output",
            &s(&output),
            "--check-only",
        ],
        &root,
        Some(""),
    );
    assert_eq!(result.status.code(), Some(2));
    let wrong_report = report(&output);
    assert!(wrong_report["primary_error"]["message"]
        .as_str()
        .unwrap_or_default()
        .contains("must not expect a rejection"));
}

#[test]
fn configured_pad_id_annotates_last_nonpad_and_all_pad_first_row() {
    let root = case_dir("pad-selection");
    let mut config = sequence_head();
    config["pad_token_id"] = serde_json::json!(0);
    let model = model_dir(&root, &config);
    let requests = root.join("requests.jsonl");
    write_text(
        &requests,
        "{\"id\":\"selection\",\"operation\":\"predict\",\"batch\":{\"input_ids\":[5,9,0,7,0,0,11,0,13,0],\"position_ids\":[0,1,2,0,1,2,0,1,2,3],\"token_type_ids\":[0,0,0,0,0,0,0,0,0,0],\"cumulative_seq_lengths\":[0,4,6,10],\"pooled_indices\":[0,1,2],\"raw_indices\":[],\"max_length\":4}}\n",
    );
    let output = root.join("out");
    let result = run(
        &[
            "--model",
            &s(&model),
            "--task",
            "classifier",
            "--api",
            "core",
            "--requests",
            &s(&requests),
            "--output",
            &s(&output),
            "--check-only",
        ],
        &root,
        Some(""),
    );
    assert!(
        result.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report = report(&output);
    assert_eq!(
        report["cases"][0]["selection"]["mode"],
        "last_non_pad_or_first_all_pad"
    );
    assert_eq!(
        report["cases"][0]["selection"]["selected_row_ids"],
        serde_json::json!([3, 4, 8])
    );
    assert_eq!(
        report["cases"][0]["selection"]["all_pad_sequences"],
        serde_json::json!([1])
    );
}

#[test]
fn all_pad_sequence_selects_first_row_in_check_only_annotation() {
    let root = case_dir("all-pad");
    let mut config = sequence_head();
    config["pad_token_id"] = serde_json::json!(0);
    let model = model_dir(&root, &config);
    let requests = root.join("requests.jsonl");
    write_text(
        &requests,
        "{\"id\":\"allpad\",\"operation\":\"predict\",\"batch\":{\"input_ids\":[0,0,0,7,0],\"position_ids\":[0,1,2,0,1],\"token_type_ids\":[0,0,0,0,0],\"cumulative_seq_lengths\":[0,3,5],\"pooled_indices\":[0,1],\"raw_indices\":[],\"max_length\":3}}\n",
    );
    let output = root.join("out");
    let result = run(
        &[
            "--model",
            &s(&model),
            "--task",
            "classifier",
            "--api",
            "core",
            "--requests",
            &s(&requests),
            "--output",
            &s(&output),
            "--check-only",
        ],
        &root,
        Some(""),
    );
    assert!(
        result.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report = report(&output);
    assert_eq!(
        report["cases"][0]["selection"]["selected_row_ids"],
        serde_json::json!([0, 3])
    );
    assert_eq!(
        report["cases"][0]["selection"]["all_pad_sequences"],
        serde_json::json!([0])
    );
}

#[test]
fn classifier_rejects_non_cls_pool() {
    let root = case_dir("classifier-pool");
    let model = model_dir(&root, &sequence_head());
    let requests = root.join("requests.jsonl");
    write_text(&requests, "{}\n");
    let output = root.join("out");
    let result = run(
        &[
            "--model",
            &s(&model),
            "--task",
            "classifier",
            "--pool",
            "mean",
            "--api",
            "core",
            "--requests",
            &s(&requests),
            "--output",
            &s(&output),
            "--check-only",
        ],
        &root,
        Some(""),
    );
    assert_eq!(result.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&result.stderr).contains("accept no --pool"));
    assert!(!output.exists());
}
