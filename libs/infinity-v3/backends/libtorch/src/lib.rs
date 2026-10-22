//! Experimental native packed inference. Python and exported TorchScript models are not required.
//!
//! `Native*` descriptors mirror `cpp/bridge.h`. The constructors below load config
//! and weights; `submit_inner` adapts a core Batch to that ABI. On the CUDA path,
//! `NativeOutput` keeps the submitted buffers alive while a completion worker
//! waits and calls `pack_output`. CPU and decision calls complete synchronously.
//! See `cpp/README.md` for model entry points and tests.
mod radix;
use radix::{NativeRadix, OwnedRadix};
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    ffi::{c_char, c_void, CStr, CString},
    fs,
    path::Path,
    ptr::NonNull,
};
use text_embeddings_backend_core::{
    Backend, BackendError, Batch, DecisionInput, DecisionOutput, Embedding, Embeddings, Inference,
    ModelType, Pool, Predictions, TokenPredictions,
};

// Matches Candle's Gemma4 cache: source host pixels and every vision layout input.
// Token/sequence starts are deliberately excluded: projected features do not use them.
fn image_feature_key(image: &text_embeddings_backend_core::ImagePatches) -> [u8; 32] {
    let mut hash = Sha256::new();
    for value in image
        .grid_thw
        .into_iter()
        .chain([image.patch_dim, image.merge_size])
    {
        hash.update(value.to_le_bytes());
    }
    hash.update(bytemuck::cast_slice(&image.pixels));
    hash.finalize().into()
}

#[repr(C)]
struct NativeConfig {
    hidden: i64,
    heads: i64,
    layers: i64,
    intermediate: i64,
    vocab: i64,
    positions: i64,
    types: i64,
    epsilon: f64,
    activation: i32,
}
#[repr(C)]
struct NativeImage {
    pixels: *const f32,
    rows: i64,
    patch_dim: i64,
    grid: [i64; 3],
    merge_size: i64,
    token_start: i64,
    token_count: i64,
    sequence_start: i64,
    feature_key: [u8; 32],
    has_feature_key: u8,
}
#[repr(C)]
struct NativeAudio {
    values: *const f32,
    mask: *const u8,
    frames: i64,
    feature_size: i64,
    token_start: i64,
    token_count: i64,
}
#[repr(C)]
struct NativeDecisionField {
    kind: i64,
    question_start: i64,
    question_end: i64,
    options: *const i64,
    option_count: usize,
}
#[repr(C)]
struct NativeDecision {
    kind: i32,
    question_type: i64,
    markers: *const i64,
    marker_count: usize,
    token_ids: *const i64,
    token_count: usize,
    fields: *const NativeDecisionField,
    field_count: usize,
}
struct OwnedDecision {
    kind: i32,
    question_type: i64,
    markers: Vec<i64>,
    token_ids: Vec<i64>,
    fields: Vec<NativeDecisionField>,
    // Own the backing storage referenced by each native field descriptor.
    _spans: Vec<Vec<i64>>,
}
impl OwnedDecision {
    fn new(input: &DecisionInput) -> Result<Self, BackendError> {
        let mut result = Self {
            kind: 3,
            question_type: 0,
            markers: vec![],
            token_ids: vec![],
            fields: vec![],
            _spans: vec![],
        };
        match input {
            DecisionInput::Laya {
                question_type,
                markers,
            } => {
                result.kind = 0;
                result.question_type = i64::try_from(*question_type).map_err(inference)?;
                result.markers =
                    collect_results(markers.iter().map(|&n| i64::try_from(n).map_err(inference)))?;
            }
            DecisionInput::OptionTokens { token_ids } => {
                result.kind = 1;
                result.token_ids = collect_fallible(token_ids.iter().map(|&n| i64::from(n)))?;
            }
            DecisionInput::InlineOptions {
                markers,
                option_counts,
            } => {
                result.kind = 5;
                result.markers =
                    collect_results(markers.iter().map(|&n| i64::try_from(n).map_err(inference)))?;
                result.token_ids = collect_results(
                    option_counts
                        .iter()
                        .map(|&n| i64::try_from(n).map_err(inference)),
                )?;
            }
            DecisionInput::BlockOptionTokens {
                token_ids,
                prefix_length,
            } => {
                result.kind = 4;
                result.question_type = i64::try_from(*prefix_length).map_err(inference)?;
                result.token_ids = collect_fallible(token_ids.iter().map(|&n| i64::from(n)))?;
            }
            DecisionInput::Clef { fields } => {
                result.kind = 2;
                result.fields = reserved_vec(fields.len())?;
                result._spans = reserved_vec(fields.len())?;
                for field in fields {
                    let mut spans = reserved_vec(
                        field
                            .options
                            .len()
                            .checked_mul(2)
                            .ok_or_else(|| inference("Decision option count overflow"))?,
                    )?;
                    for &(a, b) in &field.options {
                        spans.push(i64::try_from(a).map_err(inference)?);
                        spans.push(i64::try_from(b).map_err(inference)?);
                    }
                    result.fields.push(NativeDecisionField {
                        kind: i64::try_from(field.kind).map_err(inference)?,
                        question_start: i64::try_from(field.question.0).map_err(inference)?,
                        question_end: i64::try_from(field.question.1).map_err(inference)?,
                        options: spans.as_ptr(),
                        option_count: field.options.len(),
                    });
                    result._spans.push(spans);
                }
            }
            DecisionInput::Warmup => {}
        }
        Ok(result)
    }
    fn native(&self) -> NativeDecision {
        NativeDecision {
            kind: self.kind,
            question_type: self.question_type,
            markers: self.markers.as_ptr(),
            marker_count: self.markers.len(),
            token_ids: self.token_ids.as_ptr(),
            token_count: self.token_ids.len(),
            fields: self.fields.as_ptr(),
            field_count: self.fields.len(),
        }
    }
}
extern "C" {
    fn tei_error() -> *const c_char;
    fn tei_last_error_kind() -> i32;
    fn tei_device_count(device: *const c_char) -> i64;
    fn tei_create(config: *const NativeConfig, device: *const c_char, dtype: i32) -> *mut c_void;
    fn tei_weight(
        handle: *mut c_void,
        name: *const c_char,
        data: *const u8,
        shape: *const i64,
        rank: usize,
        dtype: i32,
    ) -> i32;
    fn tei_option(handle: *mut c_void, key: *const c_char, value: *const c_char) -> i32;
    fn tei_output_width(handle: *mut c_void) -> i64;
    fn tei_pooled_width(handle: *mut c_void) -> i64;
    fn tei_classification_width(handle: *mut c_void) -> i64;
    fn tei_supports_radix(handle: *mut c_void) -> i32;
    fn tei_prefix_cache_supported(handle: *mut c_void, operation: i32) -> i32;
    fn tei_prefix_cache_set_probe(handle: *mut c_void, enabled: i32) -> i32;
    fn tei_graph_count(handle: *mut c_void) -> i64;
    fn tei_ready(handle: *mut c_void) -> i32;
    fn tei_decision_counts(
        handle: *mut c_void,
        requests: *const NativeDecision,
        count: usize,
        counts: *mut i64,
    ) -> i32;
    fn tei_decide_radix(
        handle: *mut c_void,
        ids: *const i64,
        types: *const i64,
        positions: *const i64,
        cumulative: *const i32,
        batch: i64,
        max_sequence: i64,
        requests: *const NativeDecision,
        logits: *mut f32,
        capacity: usize,
        actions: *mut f32,
        media_positions: *const i64,
        images: *const NativeImage,
        image_count: usize,
        audios: *const NativeAudio,
        audio_count: usize,
        radix: *const NativeRadix,
    ) -> i32;
    fn tei_forward_radix(
        handle: *mut c_void,
        ids: *const i64,
        types: *const i64,
        positions: *const i64,
        cumulative: *const i32,
        batch: i64,
        max_sequence: i64,
        pool: i32,
        pooled: *const i64,
        pooled_count: usize,
        raw: *const i64,
        raw_count: usize,
        output: *mut f32,
        capacity: usize,
        media_positions: *const i64,
        images: *const NativeImage,
        image_count: usize,
        audios: *const NativeAudio,
        audio_count: usize,
        radix: *const NativeRadix,
    ) -> i32;
    fn tei_forward_submit(
        handle: *mut c_void,
        ids: *const i64,
        types: *const i64,
        positions: *const i64,
        cumulative: *const i32,
        batch: i64,
        max_sequence: i64,
        pool: i32,
        pooled: *const i64,
        pooled_count: usize,
        raw: *const i64,
        raw_count: usize,
        capacity: usize,
        media_positions: *const i64,
        images: *const NativeImage,
        image_count: usize,
        audios: *const NativeAudio,
        audio_count: usize,
        radix: *const NativeRadix,
    ) -> *mut c_void;
    fn tei_output_wait(handle: *mut c_void) -> *const f32;
    fn tei_output_destroy(handle: *mut c_void);
    fn tei_destroy(handle: *mut c_void);
}

fn native_error() -> String {
    // SAFETY: the bridge owns a thread-local, NUL-terminated string valid until the next call.
    unsafe { CStr::from_ptr(tei_error()).to_string_lossy().into_owned() }
}
fn native_failure() -> BackendError {
    // Read the structured thread-local kind before any subsequent bridge call.
    let kind = unsafe { tei_last_error_kind() };
    let message = native_error();
    match kind {
        2 => BackendError::Overloaded(message),
        3 => {
            eprintln!("Native allocator failure: {message}");
            BackendError::Overloaded("Native allocator memory exhausted".into())
        }
        _ => inference(message),
    }
}
fn start(error: impl ToString) -> BackendError {
    BackendError::Start(error.to_string())
}

fn resolve_fp8_scope(
    requested: Option<&str>,
    enabled: bool,
    family: &str,
) -> Result<String, BackendError> {
    let dense_decoder = matches!(
        family,
        "llama" | "mistral" | "qwen2" | "qwen3" | "qwen3_5" | "qwen3_5_text"
    );
    let scope = requested.unwrap_or(if enabled && dense_decoder {
        "all"
    } else {
        "mlp"
    });
    if !matches!(scope, "mlp" | "all") {
        return Err(start("TORCH_FP8_SCOPE must be mlp or all"));
    }
    if scope == "all" && (!enabled || !dense_decoder) {
        return Err(start(
            "All-linear FP8 requires dynamic FP8 and a supported dense decoder",
        ));
    }
    Ok(scope.to_owned())
}

fn upload_weight(
    handle: NonNull<c_void>,
    name: &str,
    data: &[u8],
    shape: &[usize],
    dtype: i32,
) -> Result<(), BackendError> {
    let element_size = match dtype {
        0 => 4usize,
        1 | 2 => 2usize,
        _ => return Err(start("Unsupported weight dtype")),
    };
    let expected = shape
        .iter()
        .try_fold(element_size, |bytes, &extent| bytes.checked_mul(extent));
    if expected != Some(data.len()) {
        return Err(start(format!("Invalid weight byte length for {name}")));
    }
    let shape = shape
        .iter()
        .map(|&n| i64::try_from(n).map_err(start))
        .collect::<Result<Vec<_>, _>>()?;
    let name = CString::new(name).map_err(start)?;
    // SAFETY: byte length matches the checked shape and dtype; native code copies
    // the data synchronously. The caller owns the live model handle.
    if unsafe {
        tei_weight(
            handle.as_ptr(),
            name.as_ptr(),
            data.as_ptr(),
            shape.as_ptr(),
            shape.len(),
            dtype,
        )
    } != 0
    {
        return Err(start(native_error()));
    }
    Ok(())
}

fn inference(error: impl ToString) -> BackendError {
    BackendError::Inference(error.to_string())
}

fn reserved_vec<T>(count: usize) -> Result<Vec<T>, BackendError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|error| BackendError::Overloaded(error.to_string()))?;
    Ok(values)
}

fn collect_fallible<T>(values: impl ExactSizeIterator<Item = T>) -> Result<Vec<T>, BackendError> {
    let mut result = reserved_vec(values.len())?;
    result.extend(values);
    Ok(result)
}

fn collect_results<T>(
    values: impl ExactSizeIterator<Item = Result<T, BackendError>>,
) -> Result<Vec<T>, BackendError> {
    let mut result = reserved_vec(values.len())?;
    for value in values {
        result.push(value?);
    }
    Ok(result)
}

fn filled_vec<T: Copy>(count: usize, value: T) -> Result<Vec<T>, BackendError> {
    let mut result = reserved_vec(count)?;
    result.resize(count, value);
    Ok(result)
}

fn host_output_bytes(
    elements: usize,
    raw_rows: usize,
    outputs: usize,
    cuda: bool,
) -> Result<usize, BackendError> {
    let payload = elements
        .checked_mul(std::mem::size_of::<f32>())
        .and_then(|bytes| bytes.checked_mul(if cuda { 1 } else { 2 }));
    // Include token-row Vec metadata and conservative map bucket storage,
    // including the second map used by classification output conversion.
    payload
        .and_then(|bytes| {
            raw_rows
                .checked_mul(std::mem::size_of::<Vec<f32>>())
                .and_then(|rows| bytes.checked_add(rows))
        })
        .and_then(|bytes| {
            outputs
                .checked_mul(8 * std::mem::size_of::<(usize, Embedding)>() + 8)
                .and_then(|entries| bytes.checked_add(entries))
        })
        .ok_or_else(|| inference("Host output memory size overflow"))
}

pub fn device_count(device: &str) -> Result<usize, BackendError> {
    let device = CString::new(device).map_err(start)?;
    // SAFETY: the C string is valid for the duration of the synchronous call.
    let count = unsafe { tei_device_count(device.as_ptr()) };
    usize::try_from(count).map_err(|_| start(native_error()))
}

#[derive(Deserialize)]
struct Config {
    model_type: String,
    hidden_size: i64,
    num_attention_heads: i64,
    num_hidden_layers: i64,
    intermediate_size: i64,
    vocab_size: i64,
    max_position_embeddings: i64,
    type_vocab_size: i64,
    layer_norm_eps: f64,
    hidden_act: String,
    #[serde(default)]
    is_decoder: bool,
    #[serde(default)]
    add_cross_attention: bool,
    #[serde(default)]
    position_embedding_type: Option<String>,
    #[serde(default)]
    relative_attention: bool,
    #[serde(default)]
    position_biased_input: Option<bool>,
}

pub struct LibtorchBackend {
    handle: NonNull<c_void>,
    config: Config,
    pool: Pool,
    output_width: usize,
    pooled_width: usize,
    classification_width: usize,
    decision: bool,
    supports_radix: bool,
    cuda: bool,
    host_output_max_bytes: usize,
    host_result_pool: std::sync::Arc<text_embeddings_backend_core::result_memory::ResultPool>,
    prefix_cache: bool,
}
// SAFETY: the handle owns its tensors and is used by TEI's single backend worker.
// It can move between threads; it is deliberately not Sync. InferenceMode is set per native call.
unsafe impl Send for LibtorchBackend {}

impl Drop for LibtorchBackend {
    fn drop(&mut self) {
        // SAFETY: this handle is uniquely owned and destroyed exactly once.
        unsafe { tei_destroy(self.handle.as_ptr()) };
    }
}

impl LibtorchBackend {
    /// Number of captured packed shapes; zero also covers eager fallback.
    pub fn cuda_graph_count(&self) -> usize {
        // SAFETY: this uniquely owned handle stays live throughout the synchronous query.
        unsafe { tei_graph_count(self.handle.as_ptr()) as usize }
    }
    pub fn new(
        path: &Path,
        dtype: &str,
        model_type: ModelType,
        device: &str,
    ) -> Result<Self, BackendError> {
        Self::new_with_dense_paths(path, dtype, model_type, device, None)
    }
    pub fn new_with_dense_paths(
        path: &Path,
        dtype: &str,
        model_type: ModelType,
        device: &str,
        dense_paths: Option<Vec<String>>,
    ) -> Result<Self, BackendError> {
        Self::new_with_dense_paths_and_fp8(path, dtype, model_type, device, dense_paths, false)
    }
    pub fn new_with_dense_paths_and_fp8(
        path: &Path,
        dtype: &str,
        model_type: ModelType,
        device: &str,
        dense_paths: Option<Vec<String>>,
        enable_fp8_dynamic: bool,
    ) -> Result<Self, BackendError> {
        if device != "cpu" && device != "auto" && !device.starts_with("cuda:") {
            return Err(start("LibTorch 2.14.1 varlen attention requires CUDA; only CPU reference and CUDA are supported"));
        }
        if device.starts_with("cuda:") && !matches!(dtype, "float16" | "bfloat16") {
            return Err(start("CUDA varlen attention requires float16 or bfloat16"));
        }
        let pool = match &model_type {
            ModelType::Embedding(pool) => pool.clone(),
            ModelType::Classifier => Pool::Cls,
            ModelType::Decision => Pool::LastToken,
        };
        let decision = model_type == ModelType::Decision;
        let read_config = |name: &str| -> Result<serde_json::Value, BackendError> {
            serde_json::from_slice(&fs::read(path.join(name)).map_err(start)?).map_err(start)
        };
        let laya = decision && path.join("rl_agent_config.json").exists();
        let mut options = read_config(if laya {
            "encoder/config.json"
        } else {
            "config.json"
        })?;
        // Head tensors can also be present when the requested operation is embedding.
        options["_classifier"] = serde_json::json!(model_type == ModelType::Classifier);
        if laya {
            if options["model_type"] != "modernbert" {
                return Err(start("Laya requires a ModernBERT encoder"));
            }
            for (kind, field) in [
                ("full_attention", "global_rope_theta"),
                ("sliding_attention", "local_rope_theta"),
            ] {
                if options.get(field).is_none() {
                    options[field] = options["rope_parameters"][kind]["rope_theta"].clone();
                }
            }
        }
        let decision_kind = if decision {
            let (kind, config) = if laya {
                ("laya", read_config("rl_agent_config.json")?)
            } else if path.join("decision_config.json").exists() {
                let config = read_config("decision_config.json")?;
                let kind = if config["format_id"] == "d3-code-readout-v1" {
                    "d3"
                } else {
                    "pplx"
                };
                (kind, config)
            } else if path.join("joint_head_config.json").exists() {
                ("clef", read_config("joint_head_config.json")?)
            } else {
                ("option_tokens", serde_json::json!({}))
            };
            if matches!(kind, "pplx" | "d3") {
                options["_decision_attention_mode"] = config
                    .get("attention_mode")
                    .cloned()
                    .unwrap_or_else(|| "causal".into());
            }
            options["_decision"] = serde_json::json!({"kind":kind,"config":config});
            Some(kind)
        } else {
            None
        };
        if text_embeddings_backend_core::block_attention_config::block_attention(path)
            .map_err(start)?
        {
            if decision_kind != Some("option_tokens")
                || !matches!(
                    options["model_type"].as_str(),
                    Some("qwen3_5" | "qwen3_5_text")
                )
            {
                return Err(start(
                    "Block attention requires a dense Qwen3.5 decision model",
                ));
            }
            options["_decision_attention_mode"] = "block_bidirectional".into();
        }
        if decision_kind == Some("d3") && dtype != "bfloat16" {
            return Err(start(
                "D3 requires bfloat16 for its backbone and vision tower",
            ));
        }
        let family = options["model_type"]
            .as_str()
            .ok_or_else(|| start("Missing model_type"))?
            .to_owned();
        let normalized_family = match family.as_str() {
            "bidirectional_pplx_qwen3" => "qwen3",
            "llama_bidirec" => "llama",
            "ministral3" => "mistral",
            "new" => "gte",
            "gemma4_unified" => "gemma4",
            other => other,
        }
        .to_owned();
        options["model_type"] = normalized_family.clone().into();
        if family == "bidirectional_pplx_qwen3" {
            options["use_bidirectional_attention"] = true.into();
            if !options["rope_theta"].is_number() {
                options["rope_theta"] = options["rope_parameters"]["rope_theta"].clone();
            }
            options["is_causal"] = false.into();
        }
        if enable_fp8_dynamic {
            if options
                .get("quantization_config")
                .is_some_and(|value| !value.is_null())
            {
                return Err(start("Dynamic FP8 requires an unquantized checkpoint"));
            }
            if !matches!(dtype, "float16" | "bfloat16")
                || device == "cpu"
                || !matches!(
                    normalized_family.as_str(),
                    "bert"
                        | "roberta"
                        | "xlm-roberta"
                        | "camembert"
                        | "modernbert"
                        | "llama"
                        | "mistral"
                        | "qwen2"
                        | "qwen3"
                        | "qwen3_5"
                        | "qwen3_5_text"
                )
            {
                return Err(start(
                    "Dynamic FP8 requires CUDA float16/bfloat16 and a supported dense MLP family",
                ));
            }
        }
        options["_fp8_dynamic"] = enable_fp8_dynamic.into();
        let requested_scope =
            std::env::var(infinity_v3_config::infinity_env!("TORCH_FP8_SCOPE")).ok();
        options["_fp8_scope"] = resolve_fp8_scope(
            requested_scope.as_deref(),
            enable_fp8_dynamic,
            &normalized_family,
        )?
        .into();
        if decision
            && !laya
            && !matches!(
                normalized_family.as_str(),
                "gemma4"
                    | "gemma4_text"
                    | "qwen3_5"
                    | "qwen3_5_text"
                    | "qwen3_5_moe"
                    | "qwen3_5_moe_text"
            )
        {
            return Err(start("Typed decisions require Laya, Gemma4, or Qwen3.5"));
        }
        if matches!(decision_kind, Some("pplx" | "clef"))
            && !normalized_family.starts_with("qwen3_5")
        {
            return Err(start("Pplx and Clef heads require Qwen3.5"));
        }
        options["_prefix_cache_max_bytes"] = infinity_v3_config::prefix_cache_size_bytes()
            .map_err(start)?
            .into();
        for (env, key, default, minimum) in [
            (
                infinity_v3_config::infinity_env!("TORCH_PREFIX_CACHE_MAX_TOKENS"),
                "_prefix_cache_max_tokens",
                8192,
                1,
            ),
            (
                infinity_v3_config::infinity_env!("TORCH_PREFIX_CACHE_ENTRIES"),
                "_prefix_cache_entries",
                64,
                1,
            ),
        ] {
            let value = std::env::var(env).map_or(Ok(default), |value| {
                value
                    .parse::<i64>()
                    .map_err(|_| start(format!("Invalid {env}")))
            })?;
            if value < minimum {
                return Err(start(format!("{env} must be at least {minimum}")));
            }
            options[key] = value.into();
        }
        options["_cuda_graphs"] =
            std::env::var(infinity_v3_config::infinity_env!("TORCH_CUDA_GRAPHS"))
                .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
                .into();
        options["_media_cuda_graphs"] =
            std::env::var(infinity_v3_config::infinity_env!("TORCH_MEDIA_CUDA_GRAPHS"))
                .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
                .into();
        let graph_max_tokens = std::env::var(infinity_v3_config::infinity_env!(
            "TORCH_CUDA_GRAPH_MAX_TOKENS"
        ))
        .map(|value| value.parse::<i64>().map_err(start))
        .unwrap_or(Ok(4096))?;
        if graph_max_tokens <= 0 {
            return Err(start("CUDA graph maximum tokens must be positive"));
        }
        options["_cuda_graph_max_tokens"] = graph_max_tokens.into();
        for (env, key, default, minimum) in [
            (
                infinity_v3_config::infinity_env!("TORCH_RETAINED_WORKSPACE_MAX_BYTES"),
                "_retained_workspace_max_bytes",
                512i64 << 20,
                1,
            ),
            (
                infinity_v3_config::infinity_env!("TORCH_INPUT_UPLOAD_MAX_BYTES"),
                "_input_upload_max_bytes",
                512 << 20,
                1,
            ),
            (
                infinity_v3_config::infinity_env!("TORCH_OUTPUT_STAGING_MAX_BYTES"),
                "_output_staging_max_bytes",
                512 << 20,
                1,
            ),
            (
                infinity_v3_config::infinity_env!("TORCH_CUDA_GRAPH_ENTRIES"),
                "_cuda_graph_entries",
                4,
                0,
            ),
            (
                infinity_v3_config::infinity_env!("TORCH_ATTENTION_BIAS_MAX_BYTES"),
                "_attention_bias_max_bytes",
                512 << 20,
                1,
            ),
            (
                infinity_v3_config::infinity_env!("TORCH_CUDA_GRAPH_REUSE"),
                "_cuda_graph_reuse",
                3,
                1,
            ),
            (
                infinity_v3_config::infinity_env!("TORCH_CUDA_GRAPH_MAX_BYTES"),
                "_cuda_graph_max_bytes",
                512 << 20,
                1,
            ),
            (
                infinity_v3_config::infinity_env!("TORCH_CUDA_GRAPH_MAX_CAPTURES"),
                "_cuda_graph_max_captures",
                8,
                0,
            ),
        ] {
            let value = std::env::var(env)
                .map(|value| value.parse::<i64>().map_err(start))
                .unwrap_or(Ok(default))?;
            if value < minimum {
                return Err(start(format!("{env} must be at least {minimum}")));
            }
            options[key] = value.into();
        }
        options["_cudnn_varlen"] =
            std::env::var(infinity_v3_config::infinity_env!("TORCH_CUDNN_VARLEN"))
                .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
                .into();
        if family == "llama_bidirec" {
            options["use_bidirectional_attention"] = true.into();
        }
        if family == "ultrabert" && options["max_position_embeddings"].is_null() {
            options["max_position_embeddings"] = 32768.into();
        }
        let text = options.get("text_config").unwrap_or(&options);
        let integer = |key: &str, alias: &str, fallback: i64| {
            let nomic_alias = match key {
                "hidden_size" => "n_embd",
                "num_attention_heads" => "n_head",
                "num_hidden_layers" => "n_layer",
                "intermediate_size" => "n_inner",
                "max_position_embeddings" => "n_positions",
                _ => key,
            };
            text.get(key)
                .or_else(|| text.get(alias))
                .or_else(|| {
                    if family == "nomic_bert" {
                        text.get(nomic_alias)
                    } else {
                        None
                    }
                })
                .and_then(serde_json::Value::as_i64)
                .unwrap_or(fallback)
        };
        let config = Config {
            model_type: normalized_family,
            hidden_size: integer("hidden_size", "dim", 0),
            num_attention_heads: integer("num_attention_heads", "n_heads", 0),
            num_hidden_layers: integer("num_hidden_layers", "n_layers", 0),
            intermediate_size: integer("intermediate_size", "hidden_dim", 0),
            vocab_size: integer("vocab_size", "vocab_size", 0),
            max_position_embeddings: integer(
                "max_position_embeddings",
                "max_position_embeddings",
                i64::MAX,
            ),
            // UltraBERT ignores segment IDs, but reranking tokenizers still emit 0/1.
            type_vocab_size: integer(
                "type_vocab_size",
                "type_vocab_size",
                if family == "ultrabert" { 2 } else { 1 },
            ),
            layer_norm_eps: text
                .get("layer_norm_eps")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(1e-5),
            hidden_act: text
                .get("hidden_act")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("gelu")
                .into(),
            is_decoder: text
                .get("is_decoder")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            add_cross_attention: text
                .get("add_cross_attention")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            position_embedding_type: text
                .get("position_embedding_type")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned),
            relative_attention: text
                .get("relative_attention")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            position_biased_input: text
                .get("position_biased_input")
                .and_then(serde_json::Value::as_bool),
        };
        if config.model_type == "bert"
            && ![
                options["auto_map"]["AutoConfig"].as_str(),
                options["_name_or_path"].as_str(),
            ]
            .into_iter()
            .flatten()
            .any(|name| {
                name.contains("jina-bert-implementation")
                    || name.contains("jina-bert-v2-qk-post-norm")
            })
            && (config.is_decoder
                || config.add_cross_attention
                || config
                    .position_embedding_type
                    .as_deref()
                    .is_some_and(|p| p != "absolute"))
        {
            return Err(start(
                "BERT requires an encoder with absolute position embeddings",
            ));
        }
        // These decoders project explicitly sized attention heads; their
        // attention width need not equal the residual stream width.
        // Match normalized families: Ministral3 uses the Mistral decoder.
        // Encoder families still require hidden_size to divide into heads.
        let independent_attention_width = matches!(
            config.model_type.as_str(),
            "llama"
                | "mistral"
                | "qwen2"
                | "qwen3"
                | "qwen3_moe"
                | "qwen3_5"
                | "qwen3_5_text"
                | "qwen3_5_moe"
                | "qwen3_5_moe_text"
                | "mistral"
                | "mistral3"
                | "ultrabert"
        ) && text
            .get("head_dim")
            .and_then(serde_json::Value::as_i64)
            .is_some_and(|head_dim| head_dim > 0);
        if config.hidden_size <= 0
            || config.vocab_size <= 0
            || config.num_attention_heads <= 0
            || config.num_hidden_layers <= 0
            || (!independent_attention_width
                && config.hidden_size % config.num_attention_heads != 0)
        {
            return Err(start("Invalid model dimensions"));
        }
        let mut module_dense_paths = Vec::new();
        if path.join("modules.json").exists() {
            let modules: serde_json::Value =
                serde_json::from_slice(&fs::read(path.join("modules.json")).map_err(start)?)
                    .map_err(start)?;
            for module in modules
                .as_array()
                .ok_or_else(|| start("modules.json must contain an array"))?
            {
                let kind = module["type"]
                    .as_str()
                    .ok_or_else(|| start("Missing module type"))?;
                // Output quantization is omitted: serve floating-point embeddings.
                if family == "bidirectional_pplx_qwen3" && kind == "st_quantize.FlexibleQuantizer" {
                    continue;
                }
                if !kind.starts_with("sentence_transformers.")
                    || !matches!(
                        kind.rsplit('.').next(),
                        Some("Transformer" | "Pooling" | "Normalize" | "Dense")
                    )
                {
                    return Err(start(format!(
                        "LibTorch does not yet support Sentence Transformers module {kind}"
                    )));
                }
                if kind.rsplit('.').next() == Some("Dense") {
                    module_dense_paths.push(
                        module["path"]
                            .as_str()
                            .ok_or_else(|| start("Dense module is missing its path"))?
                            .to_owned(),
                    );
                }
            }
        }
        // Mistral3 cross-encoders also ship their score head as a Sentence Transformers
        // Dense module; the backend applies the checkpoint's own copy.
        let score_head_in_checkpoint = model_type == ModelType::Classifier
            && options["architectures"][0].as_str() == Some("Mistral3ForSequenceClassification");
        let dense_paths = match dense_paths {
            Some(paths) => paths,
            None if score_head_in_checkpoint => Vec::new(),
            None => module_dense_paths,
        };
        if !dense_paths.is_empty() && !matches!(model_type, ModelType::Embedding(_)) {
            return Err(start(
                "Sentence Transformers Dense modules require an embedding model",
            ));
        }
        options["_dense_count"] = serde_json::json!(dense_paths.len());
        options["_pooling"] = serde_json::json!(if pool == Pool::Splade {
            "splade"
        } else {
            "embedding"
        });
        // Keep the head contract above separate from the actual pooling mode
        // used to select narrowly qualified native embedding kernels.
        options["_embedding_pool"] = serde_json::json!(pool.to_string());
        for (i, dense_path) in dense_paths.iter().enumerate() {
            let relative = Path::new(dense_path);
            if relative.as_os_str().is_empty()
                || relative
                    .components()
                    .any(|part| !matches!(part, std::path::Component::Normal(_)))
            {
                return Err(start(
                    "Dense module path must stay inside the model directory",
                ));
            }
            let dense_config: serde_json::Value = serde_json::from_slice(
                &fs::read(path.join(relative).join("config.json")).map_err(start)?,
            )
            .map_err(start)?;
            if !dense_config.is_object() {
                return Err(start("Dense config must be an object"));
            }
            if dense_config["bias"].as_bool().is_none() {
                return Err(start("Dense config bias must be a boolean"));
            }
            options["_dense"][i.to_string()] = dense_config;
        }
        let activation = match config.hidden_act.as_str() {
            "gelu" => 0,
            "gelu_new" | "gelu_pytorch_tanh" => 1,
            "relu" => 2,
            "silu" => 3,
            _ => 0,
        };
        let native = NativeConfig {
            hidden: config.hidden_size,
            heads: config.num_attention_heads,
            layers: config.num_hidden_layers,
            intermediate: config.intermediate_size,
            vocab: config.vocab_size,
            positions: config.max_position_embeddings,
            types: config.type_vocab_size,
            epsilon: config.layer_norm_eps,
            activation,
        };
        let dtype = match dtype {
            "float32" | "auto" => 0,
            "float16" => 1,
            "bfloat16" => 2,
            _ => return Err(start("Unsupported LibTorch dtype")),
        };
        let device = CString::new(device).map_err(start)?;
        // SAFETY: config and device stay valid during creation, and the bridge copies them.
        let host_result_pool = text_embeddings_backend_core::result_memory::global_pool()?;
        let host_output_max_bytes = std::env::var(infinity_v3_config::infinity_env!(
            "TORCH_HOST_OUTPUT_MAX_BYTES"
        ))
        .map(|value| value.parse::<usize>().map_err(start))
        .unwrap_or(Ok(512 << 20))?;
        if host_output_max_bytes == 0 {
            return Err(start("Host output memory budget must be positive"));
        }
        let handle = NonNull::new(unsafe { tei_create(&native, device.as_ptr(), dtype) })
            .ok_or_else(|| start(native_error()))?;
        let mut model = Self {
            handle,
            config,
            pool,
            output_width: 0,
            pooled_width: 0,
            classification_width: 0,
            supports_radix: false,
            prefix_cache: options["_prefix_cache_max_bytes"].as_i64().unwrap_or(0) > 0,
            cuda: device.to_bytes().starts_with(b"cuda")
                || (device.to_bytes() == b"auto" && device_count("cuda").unwrap_or(0) > 0),
            decision,
            host_output_max_bytes,
            host_result_pool,
        };
        fn flatten(value: &serde_json::Value, key: &str, output: &mut Vec<(String, String)>) {
            match value {
                serde_json::Value::Object(map) => {
                    for (name, value) in map {
                        let key = if key.is_empty() {
                            name.clone()
                        } else {
                            format!("{key}.{name}")
                        };
                        flatten(value, &key, output);
                    }
                }
                serde_json::Value::Array(items) => {
                    output.push((key.into(), value.to_string()));
                    for (index, value) in items.iter().enumerate() {
                        flatten(value, &format!("{key}.{index}"), output);
                    }
                }
                serde_json::Value::Null => {}
                serde_json::Value::String(value) => output.push((key.into(), value.clone())),
                value => output.push((key.into(), value.to_string())),
            }
        }
        let mut flattened = Vec::new();
        flatten(&options, "", &mut flattened);
        for (key, value) in flattened {
            let key = CString::new(key).map_err(start)?;
            let value = CString::new(value).map_err(start)?;
            // SAFETY: the bridge copies both NUL-terminated strings before returning.
            if unsafe { tei_option(handle.as_ptr(), key.as_ptr(), value.as_ptr()) } != 0 {
                return Err(start(native_error()));
            }
        }
        let index = path.join("model.safetensors.index.json");
        let files = if path.join("model.safetensors").exists() {
            vec!["model.safetensors".to_owned()]
        } else if path.join("pytorch_model.bin").exists() {
            vec!["pytorch_model.bin".to_owned()]
        } else if index.exists() {
            let index: serde_json::Value =
                serde_json::from_slice(&fs::read(index).map_err(start)?).map_err(start)?;
            let map = index["weight_map"]
                .as_object()
                .ok_or_else(|| start("Invalid safetensors index"))?;
            let mut files = std::collections::BTreeSet::new();
            for file in map.values() {
                let file = file
                    .as_str()
                    .ok_or_else(|| start("Invalid shard filename"))?;
                // Restrict shard references to files inside the model directory.
                if Path::new(file).components().count() != 1 || !file.ends_with(".safetensors") {
                    return Err(start("Invalid shard filename"));
                }
                files.insert(file.to_owned());
            }
            files.into_iter().collect::<Vec<_>>()
        } else {
            vec!["model.safetensors".to_owned()]
        };
        let mut files = files
            .into_iter()
            .map(|file| (file, String::new()))
            .collect::<Vec<_>>();
        if let Some(kind) = decision_kind {
            if matches!(kind, "pplx" | "d3") {
                files.push(("readout.safetensors".into(), "__tei_decision.".into()));
            }
            if kind == "clef" {
                files.push(("joint_head.safetensors".into(), "__tei_decision.".into()));
            }
        }
        for (i, dense_path) in dense_paths.iter().enumerate() {
            let filename = if path.join(dense_path).join("model.safetensors").exists() {
                "model.safetensors"
            } else {
                "pytorch_model.bin"
            };
            files.push((format!("{dense_path}/{filename}"), format!("_dense.{i}.")));
        }
        for (file, prefix) in files {
            if file.ends_with(".bin") {
                // Parse checkpoint data in Rust; no Python or inference engine is
                // involved. The reader gathers strided/shared-storage views.
                let reader = pytorch_reader::PytorchReader::new(path.join(&file)).map_err(start)?;
                for (name, tensor) in reader.tensors() {
                    if name.ends_with("position_ids") || name.ends_with("token_type_ids") {
                        continue;
                    }
                    let dtype = match tensor.dtype() {
                        pytorch_reader::DType::F32 => 0,
                        pytorch_reader::DType::F16 => 1,
                        pytorch_reader::DType::BF16 => 2,
                        other => return Err(start(format!("Unsupported weight dtype {other:?}"))),
                    };
                    let data = tensor.read().map_err(start)?;
                    upload_weight(
                        handle,
                        &format!("{prefix}{name}"),
                        &data,
                        tensor.shape(),
                        dtype,
                    )?;
                }
                continue;
            }
            let data = fs::read(path.join(file)).map_err(start)?;
            let tensors = SafeTensors::deserialize(&data).map_err(start)?;
            for (name, tensor) in tensors.tensors() {
                // Integer buffers are derived from packed positions rather than uploaded as weights.
                if name.ends_with("position_ids") || name.ends_with("token_type_ids") {
                    continue;
                }
                let dtype = match tensor.dtype() {
                    Dtype::F32 => 0,
                    Dtype::F16 => 1,
                    Dtype::BF16 => 2,
                    other => return Err(start(format!("Unsupported weight dtype {other:?}"))),
                };
                upload_weight(
                    handle,
                    &format!("{prefix}{name}"),
                    tensor.data(),
                    tensor.shape(),
                    dtype,
                )?;
            }
        }
        // SAFETY: the live handle is owned by model; this checks every required weight and its shape.
        if unsafe { tei_ready(handle.as_ptr()) } != 0 {
            return Err(start(native_error()));
        }
        // SAFETY: ready() has created and validated the concrete model.
        model.output_width = usize::try_from(unsafe { tei_output_width(handle.as_ptr()) })
            .map_err(|_| start(native_error()))?;
        // SAFETY: ready() has validated the projection chain and its output width.
        model.pooled_width = usize::try_from(unsafe { tei_pooled_width(handle.as_ptr()) })
            .map_err(|_| start(native_error()))?;
        // SAFETY: ready() has validated the concrete model and owns its weights.
        model.classification_width =
            usize::try_from(unsafe { tei_classification_width(handle.as_ptr()) })
                .map_err(|_| start(native_error()))?;
        if model_type == ModelType::Classifier && model.classification_width == 0 {
            return Err(start("Checkpoint has no supported classification head"));
        }
        // SAFETY: the initialized handle owns immutable model capability flags.
        model.supports_radix = unsafe { tei_supports_radix(handle.as_ptr()) } == 1;
        if model.prefix_cache {
            let operation = i32::from(model_type == ModelType::Classifier);
            // SAFETY: the initialized handle owns the concrete model capability.
            let supported = unsafe { tei_prefix_cache_supported(handle.as_ptr(), operation) };
            if supported < 0 {
                return Err(start(native_error()));
            }
            model.prefix_cache = supported == 1;
            tracing::info!(
                enabled = model.prefix_cache,
                bytes = options["_prefix_cache_max_bytes"].as_i64(),
                budget_gb = options["_prefix_cache_max_bytes"]
                    .as_i64()
                    .map(|b| b as f64 / 1e9),
                max_tokens = options["_prefix_cache_max_tokens"].as_i64(),
                entries = options["_prefix_cache_entries"].as_i64(),
                "LibTorch prefix cache configured"
            );
        }
        Ok(model)
    }
}

// A native lease owns pinned inputs/output and GPU temporaries independently
// of the model. Only the completion worker reads it after submission.
struct NativeOutput {
    handle: NonNull<c_void>,
    size: usize,
}

struct PrefixProbe<'a> {
    handle: NonNull<c_void>,
    previous: i32,
    _backend: std::marker::PhantomData<&'a LibtorchBackend>,
}
impl<'a> PrefixProbe<'a> {
    fn new(backend: &'a LibtorchBackend) -> Result<Self, BackendError> {
        let handle = backend.handle;
        // SAFETY: the backend worker serializes access to this live engine.
        let previous = unsafe { tei_prefix_cache_set_probe(handle.as_ptr(), 1) };
        if previous < 0 {
            return Err(inference(native_error()));
        }
        Ok(Self {
            handle,
            previous,
            _backend: std::marker::PhantomData,
        })
    }
}
impl Drop for PrefixProbe<'_> {
    fn drop(&mut self) {
        // SAFETY: this scope cannot outlive the borrowed backend; restore even on errors.
        unsafe { tei_prefix_cache_set_probe(self.handle.as_ptr(), self.previous) };
    }
}
// SAFETY: the uniquely owned lease is transferred once; it never calls the model.
unsafe impl Send for NativeOutput {}
impl NativeOutput {
    fn values(&self) -> Result<&[f32], BackendError> {
        // SAFETY: wait synchronizes the lease's event before exposing its pinned output.
        let values = unsafe { tei_output_wait(self.handle.as_ptr()) };
        if values.is_null() {
            return Err(native_failure());
        }
        // SAFETY: native capacity was checked, and the lease stays live through this borrow.
        Ok(unsafe { std::slice::from_raw_parts(values, self.size) })
    }
}
impl Drop for NativeOutput {
    fn drop(&mut self) {
        // SAFETY: uniquely owned lease; destruction drains work before buffer reuse.
        unsafe { tei_output_destroy(self.handle.as_ptr()) };
    }
}

fn pack_output(
    output: &[f32],
    pooled: Vec<i64>,
    raw: Vec<i64>,
    lengths: Vec<i64>,
    hidden: usize,
    raw_width: usize,
) -> Result<Embeddings, BackendError> {
    let mut result = Embeddings::default();
    result
        .try_reserve(
            pooled
                .len()
                .checked_add(raw.len())
                .ok_or_else(|| inference("Output count overflow"))?,
        )
        .map_err(|error| BackendError::Overloaded(error.to_string()))?;
    let mut offset = 0;
    for index in pooled {
        result.insert(
            index as usize,
            Embedding::Pooled(collect_fallible(
                output[offset..offset + hidden].iter().copied(),
            )?),
        );
        offset += hidden;
    }
    for index in raw {
        let size = lengths[index as usize] as usize * raw_width;
        let values = collect_results(
            output[offset..offset + size]
                .chunks_exact(raw_width)
                .map(|row| collect_fallible(row.iter().copied())),
        )?;
        result.insert(index as usize, Embedding::All(values));
        offset += size;
    }
    Ok(result)
}

impl LibtorchBackend {
    fn check_host_output(&self, bytes: usize) -> Result<(), BackendError> {
        if bytes > self.host_output_max_bytes {
            return Err(BackendError::Overloaded(format!(
                "Host output exceeds {}",
                infinity_v3_config::infinity_env!("TORCH_HOST_OUTPUT_MAX_BYTES")
            )));
        }
        Ok(())
    }
    fn run_inner(
        &self,
        batch: Batch,
        pool: i32,
        hidden: usize,
        requests: Option<&[DecisionInput]>,
    ) -> Result<(Embeddings, Vec<DecisionOutput>), BackendError> {
        self.submit_inner(batch, pool, hidden, requests)?.complete()
    }
    fn submit_inner(
        &self,
        batch: Batch,
        pool: i32,
        hidden: usize,
        requests: Option<&[DecisionInput]>,
    ) -> Result<Inference<(Embeddings, Vec<DecisionOutput>)>, BackendError> {
        let ends = &batch.cumulative_seq_lengths;
        if ends.len() < 2
            || ends[0] != 0
            || ends.last().copied() != Some(batch.input_ids.len() as u32)
            || ends.windows(2).any(|w| w[0] >= w[1])
            || batch.token_type_ids.len() != batch.input_ids.len()
            || batch.position_ids.len() != batch.input_ids.len()
        {
            return Err(inference("Invalid packed batch"));
        }
        let owned_radix = OwnedRadix::from_batch(&batch)?;
        let native_radix = owned_radix.as_ref().map(OwnedRadix::native);
        let b = ends.len() - 1;
        let lengths = collect_fallible(ends.windows(2).map(|w| (w[1] - w[0]) as i64))?;
        let seq = *lengths.iter().max().unwrap() as usize;
        if !batch.multimodal.is_empty() && batch.multimodal.len() != b {
            return Err(inference("Invalid multimodal batch length"));
        }
        if owned_radix.is_some()
            && !batch.multimodal.is_empty()
            && !text_embeddings_backend_core::MultimodalEncoding::allows_radix_fold(
                &batch.multimodal,
                &batch.input_ids,
                &batch.cumulative_seq_lengths,
                batch.scatter_unfold.as_ref().unwrap(),
                batch.fold_gather.as_ref().unwrap(),
            )
        {
            return Err(inference(
                "Radix media folding requires identical images and image prefixes",
            ));
        }
        let mut images = Vec::new();
        let cache_images = self.prefix_cache
            || matches!(
                self.config.model_type.as_str(),
                "gemma4" | "gemma4_text" | "embedding_gemma2"
            );
        let mut image_keys = std::collections::HashMap::new();
        let mut audios = Vec::new();
        let mut media_positions = Vec::new();
        if batch.multimodal.iter().any(Option::is_some) {
            media_positions = reserved_vec(
                batch
                    .position_ids
                    .len()
                    .checked_mul(3)
                    .ok_or_else(|| inference("Media position count overflow"))?,
            )?;
            for _ in 0..3 {
                media_positions.extend(batch.position_ids.iter().map(|&p| i64::from(p)));
            }
            for (row, media) in batch.multimodal.iter().enumerate() {
                let Some(media) = media else { continue };
                let begin = ends[row] as usize;
                let length = lengths[row] as usize;
                for axis in 0..3 {
                    if media.position_ids[axis].len() != length {
                        return Err(inference("Invalid multimodal position length"));
                    }
                    for (index, &value) in media.position_ids[axis].iter().enumerate() {
                        media_positions[axis * batch.input_ids.len() + begin + index] =
                            i64::from(value);
                    }
                }
                for (start, image) in &media.images {
                    let patches = image
                        .grid_thw
                        .iter()
                        .try_fold(1usize, |n, &d| n.checked_mul(d))
                        .ok_or_else(|| inference("Image grid overflow"))?;
                    let merge = image
                        .merge_size
                        .checked_mul(image.merge_size)
                        .filter(|&n| n > 0)
                        .ok_or_else(|| inference("Invalid image merge size"))?;
                    if image.patch_dim == 0
                        || patches == 0
                        || patches % merge != 0
                        || patches.checked_mul(image.patch_dim) != Some(image.pixels.len())
                        || start
                            .checked_add(patches / merge)
                            .is_none_or(|end| end > length)
                    {
                        return Err(inference("Invalid image layout or token span"));
                    }
                    images.try_reserve(1).map_err(inference)?;
                    image_keys.try_reserve(1).map_err(inference)?;
                    images.push(NativeImage {
                        pixels: image.pixels.as_ptr(),
                        rows: patches as i64,
                        patch_dim: image.patch_dim as i64,
                        grid: image.grid_thw.map(|n| n as i64),
                        merge_size: image.merge_size as i64,
                        token_start: (begin + start) as i64,
                        token_count: (patches / merge) as i64,
                        sequence_start: begin as i64,
                        feature_key: if cache_images {
                            *image_keys
                                .entry(std::sync::Arc::as_ptr(image))
                                .or_insert_with(|| image_feature_key(image))
                        } else {
                            [0; 32]
                        },
                        has_feature_key: u8::from(cache_images),
                    });
                }
                for (start, audio) in &media.audios {
                    if audio.feature_size == 0
                        || audio.mask.is_empty()
                        || audio.mask.len().checked_mul(audio.feature_size)
                            != Some(audio.values.len())
                        || start
                            .checked_add(audio.token_count())
                            .is_none_or(|end| end > length)
                    {
                        return Err(inference("Invalid audio layout or token span"));
                    }
                    audios.try_reserve(1).map_err(inference)?;
                    audios.push(NativeAudio {
                        values: audio.values.as_ptr(),
                        mask: audio.mask.as_ptr(),
                        frames: audio.mask.len() as i64,
                        feature_size: audio.feature_size as i64,
                        token_start: (begin + start) as i64,
                        token_count: audio.token_count() as i64,
                    });
                }
            }
        }
        // Preserve TEI's packed layout: allocations scale with actual tokens, never B * max_length.
        let ids = collect_fallible(batch.input_ids.iter().map(|&id| id as i64))?;
        let types = collect_fallible(batch.token_type_ids.iter().map(|&id| id as i64))?;
        let positions = collect_fallible(batch.position_ids.iter().map(|&id| id as i64))?;
        let cumulative = collect_results(
            ends.iter()
                .map(|&end| i32::try_from(end).map_err(inference)),
        )?;
        // Relative-only DeBERTa buckets long distances instead of indexing a
        // learned absolute-position table. Keep that table bound for other models
        // and for DeBERTa configurations that enable position-biased input.
        let relative_positions = self.config.model_type == "deberta-v2"
            && self.config.relative_attention
            && self.config.position_biased_input == Some(false);
        if ids.iter().any(|&id| id >= self.config.vocab_size)
            || (self.config.type_vocab_size > 0
                && types.iter().any(|&id| id >= self.config.type_vocab_size))
            || (!relative_positions
                && positions
                    .iter()
                    .any(|&id| id >= self.config.max_position_embeddings))
        {
            return Err(inference(
                "Token/type/position ID exceeds configured limits",
            ));
        }
        if let Some(requests) = requests {
            if !self.decision || requests.len() != b {
                return Err(inference(
                    "Decision metadata count does not match a typed decision model/batch",
                ));
            }
            let owned = collect_results(requests.iter().map(OwnedDecision::new))?;
            let native = collect_fallible(owned.iter().map(OwnedDecision::native))?;
            let mut counts = filled_vec(b, 0i64)?;
            // SAFETY: descriptors and their backing span/token buffers remain owned above.
            if unsafe {
                tei_decision_counts(
                    self.handle.as_ptr(),
                    native.as_ptr(),
                    native.len(),
                    counts.as_mut_ptr(),
                )
            } != 0
            {
                return Err(native_failure());
            }
            let counts = collect_results(
                counts
                    .into_iter()
                    .map(|count| usize::try_from(count).map_err(inference)),
            )?;
            let total = counts.iter().try_fold(0usize, |n, &count| {
                n.checked_add(count)
                    .ok_or_else(|| inference("Decision output size overflow"))
            })?;
            let decision_bytes = total
                .checked_mul(2 * std::mem::size_of::<f32>())
                .and_then(|bytes| {
                    b.checked_mul(
                        std::mem::size_of::<DecisionOutput>()
                            + std::mem::size_of::<usize>()
                            + std::mem::size_of::<f32>(),
                    )
                    .and_then(|rows| bytes.checked_add(rows))
                })
                .ok_or_else(|| inference("Decision host output memory size overflow"))?;
            self.check_host_output(decision_bytes)?;
            let memory = self.host_result_pool.reserve(decision_bytes)?;
            let mut logits = filled_vec(total, 0f32)?;
            let mut actions = filled_vec(b, 0f32)?;
            // SAFETY: native output counts bound capacity, validated packed/media buffers stay live,
            // and all exceptions are caught by the bridge before returning across the C ABI.
            if unsafe {
                tei_decide_radix(
                    self.handle.as_ptr(),
                    ids.as_ptr(),
                    types.as_ptr(),
                    positions.as_ptr(),
                    cumulative.as_ptr(),
                    b as i64,
                    seq as i64,
                    native.as_ptr(),
                    logits.as_mut_ptr(),
                    logits.len(),
                    actions.as_mut_ptr(),
                    if media_positions.is_empty() {
                        std::ptr::null()
                    } else {
                        media_positions.as_ptr()
                    },
                    images.as_ptr(),
                    images.len(),
                    audios.as_ptr(),
                    audios.len(),
                    native_radix
                        .as_ref()
                        .map_or(std::ptr::null(), |radix| radix),
                )
            } != 0
            {
                return Err(native_failure());
            }
            let mut offset = 0;
            let decisions = collect_results(counts.into_iter().zip(actions).map(
                |(count, action_probability)| {
                    let result = DecisionOutput {
                        memory: Some(memory.clone()),
                        logits: collect_fallible(logits[offset..offset + count].iter().copied())?,
                        action_probability,
                    };
                    offset += count;
                    Ok(result)
                },
            ))?;
            return Ok(Inference::Ready((Embeddings::default(), decisions)));
        }
        let pooled = collect_fallible(batch.pooled_indices.iter().map(|&n| n as i64))?;
        let raw = collect_fallible(batch.raw_indices.iter().map(|&n| n as i64))?;
        if pooled.iter().chain(&raw).any(|&n| n as usize >= b) {
            return Err(inference("Invalid output index"));
        }
        let raw_width = if pool >= 4 { hidden } else { self.output_width };
        let raw_rows = raw.iter().try_fold(0usize, |rows, &i| {
            rows.checked_add(lengths[i as usize] as usize)
                .ok_or_else(|| inference("Raw output row count overflow"))
        })?;
        let size = pooled
            .len()
            .checked_mul(hidden)
            .and_then(|n| {
                raw_rows
                    .checked_mul(raw_width)
                    .and_then(|r| n.checked_add(r))
            })
            .ok_or_else(|| inference("Output size overflow"))?;
        let output_bytes = host_output_bytes(
            size,
            raw_rows,
            pooled
                .len()
                .checked_add(raw.len())
                .ok_or_else(|| inference("Output count overflow"))?,
            self.cuda,
        )?;
        self.check_host_output(output_bytes)?;
        let memory = self.host_result_pool.reserve(output_bytes)?;
        // CUDA results share one bounded completion path at every output size.
        // The model thread can submit the next forward while CPU packing finishes.
        if self.cuda {
            // SAFETY: all descriptors remain live during submission. Native code
            // copies staged inputs and owns every buffer needed after return.
            let handle = NonNull::new(unsafe {
                tei_forward_submit(
                    self.handle.as_ptr(),
                    ids.as_ptr(),
                    types.as_ptr(),
                    positions.as_ptr(),
                    cumulative.as_ptr(),
                    b as i64,
                    seq as i64,
                    pool,
                    pooled.as_ptr(),
                    pooled.len(),
                    raw.as_ptr(),
                    raw.len(),
                    size,
                    if media_positions.is_empty() {
                        std::ptr::null()
                    } else {
                        media_positions.as_ptr()
                    },
                    images.as_ptr(),
                    images.len(),
                    audios.as_ptr(),
                    audios.len(),
                    native_radix
                        .as_ref()
                        .map_or(std::ptr::null(), |radix| radix),
                )
            })
            .ok_or_else(native_failure)?;
            let output = NativeOutput { handle, size };
            return Ok(Inference::Deferred(Box::new(move || {
                let mut result =
                    pack_output(output.values()?, pooled, raw, lengths, hidden, raw_width)?;
                result.memory = Some(memory);
                Ok((result, vec![]))
            })));
        }
        let mut output = filled_vec(size, 0f32)?;
        // SAFETY: validated dimensions and indices bound every native read/write. All buffers stay
        // live until this synchronous call returns; exceptions cannot cross the C ABI.
        if unsafe {
            tei_forward_radix(
                self.handle.as_ptr(),
                ids.as_ptr(),
                types.as_ptr(),
                positions.as_ptr(),
                cumulative.as_ptr(),
                b as i64,
                seq as i64,
                pool,
                pooled.as_ptr(),
                pooled.len(),
                raw.as_ptr(),
                raw.len(),
                output.as_mut_ptr(),
                output.len(),
                if media_positions.is_empty() {
                    std::ptr::null()
                } else {
                    media_positions.as_ptr()
                },
                images.as_ptr(),
                images.len(),
                audios.as_ptr(),
                audios.len(),
                native_radix
                    .as_ref()
                    .map_or(std::ptr::null(), |radix| radix),
            )
        } != 0
        {
            return Err(native_failure());
        }
        let mut results = pack_output(&output, pooled, raw, lengths, hidden, raw_width)?;
        results.memory = Some(memory);
        Ok(Inference::Ready((results, vec![])))
    }
}

impl Backend for LibtorchBackend {
    fn probe(&self, batch: Batch, model_type: &ModelType) -> Result<(), BackendError> {
        let _probe = PrefixProbe::new(self)?;
        match model_type {
            ModelType::Decision => {
                let count = batch.len();
                self.decide(batch, vec![DecisionInput::Warmup; count])
                    .map(|_| ())
            }
            ModelType::Classifier => self.predict(batch).map(|_| ()),
            ModelType::Embedding(_) => self.embed(batch).map(|_| ()),
        }
    }
    fn submit_embed(&self, batch: Batch) -> Result<Inference<Embeddings>, BackendError> {
        let (pool, width) = match self.pool {
            Pool::Cls => (0, self.pooled_width),
            Pool::Mean => (1, self.pooled_width),
            Pool::LastToken => (2, self.pooled_width),
            Pool::Splade => (3, self.pooled_width),
        };
        self.submit_inner(batch, pool, width, None)
            .map(|result| result.map(|(values, _)| values))
    }
    fn submit_predict(&self, mut batch: Batch) -> Result<Inference<Predictions>, BackendError> {
        if self.classification_width == 0 {
            return Err(inference("Checkpoint has no classification head"));
        }
        batch.pooled_indices = collect_fallible(0..batch.len() as u32)?;
        batch.raw_indices.clear();
        let count = batch.len();
        self.submit_inner(batch, 4, self.classification_width, None)?
            .try_map(move |(values, _)| {
                let mut predictions = Predictions::default();
                predictions.memory = values.memory.clone();
                predictions
                    .try_reserve(count)
                    .map_err(|error| BackendError::Overloaded(error.to_string()))?;
                for (index, value) in values {
                    if let Embedding::Pooled(value) = value {
                        predictions.insert(index, value);
                    }
                }
                Ok(predictions)
            })
    }
    fn submit_predict_tokens(
        &self,
        mut batch: Batch,
    ) -> Result<Inference<TokenPredictions>, BackendError> {
        if self.classification_width == 0 {
            return Err(inference("Checkpoint has no classification head"));
        }
        batch.raw_indices = collect_fallible(0..batch.len() as u32)?;
        batch.pooled_indices.clear();
        let count = batch.len();
        self.submit_inner(batch, 5, self.classification_width, None)?
            .try_map(move |(values, _)| {
                let mut predictions = TokenPredictions::default();
                predictions.memory = values.memory.clone();
                predictions
                    .try_reserve(count)
                    .map_err(|error| BackendError::Overloaded(error.to_string()))?;
                for (index, value) in values {
                    if let Embedding::All(value) = value {
                        predictions.insert(index, value);
                    }
                }
                Ok(predictions)
            })
    }
    fn supports_radix_mlp(&self) -> bool {
        // Native models advertise folding only when their complete forward path supports it.
        self.supports_radix
    }
    fn decide(
        &self,
        mut batch: Batch,
        inputs: Vec<DecisionInput>,
    ) -> Result<Vec<DecisionOutput>, BackendError> {
        batch.pooled_indices.clear();
        batch.raw_indices.clear();
        self.run_inner(batch, 2, self.output_width, Some(&inputs))
            .map(|(_, decisions)| decisions)
    }
    fn health(&self) -> Result<(), BackendError> {
        Ok(())
    }
    fn embed(&self, batch: Batch) -> Result<Embeddings, BackendError> {
        self.submit_embed(batch)?.complete()
    }
    fn predict(&self, batch: Batch) -> Result<Predictions, BackendError> {
        self.submit_predict(batch)?.complete()
    }
    fn predict_tokens(&self, batch: Batch) -> Result<TokenPredictions, BackendError> {
        self.submit_predict_tokens(batch)?.complete()
    }
}

#[cfg(test)]
#[path = "../tests/support/pytorch_checkpoint.rs"]
mod pytorch_checkpoint;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_output_admission_and_reservation_failures_are_recoverable() {
        assert!(reserved_vec::<u64>(usize::MAX).is_err());
        assert!(host_output_bytes(usize::MAX, 0, 0, false).is_err());
        let fixture = fixture("", false);
        let mut model = LibtorchBackend::new(
            &fixture.0,
            "float32",
            ModelType::Embedding(Pool::Mean),
            "cpu",
        )
        .unwrap();
        let expected = model.embed(batch()).unwrap();
        model.host_output_max_bytes = 1;
        assert!(model
            .embed(batch())
            .err()
            .unwrap()
            .to_string()
            .contains("HOST_OUTPUT_MAX_BYTES"));
        model.host_output_max_bytes = 512 << 20;
        let actual = model.embed(batch()).unwrap();
        model.host_result_pool =
            text_embeddings_backend_core::result_memory::ResultPool::new(4096).unwrap();
        let held = model.host_result_pool.reserve(4096).unwrap();
        assert!(matches!(
            model.embed(batch()),
            Err(BackendError::Overloaded(_))
        ));
        drop(held);
        let recovered = model.embed(batch()).unwrap();
        for (index, expected) in expected {
            assert_eq!(values(&expected), values(&recovered[&index]));
            assert_eq!(values(&expected), values(&actual[&index]));
        }
    }

    #[test]
    fn d3_constructor_rejects_unsupported_dtype_before_loading_weights() {
        let fixture = fixture("", false);
        fs::write(
            fixture.0.join("decision_config.json"),
            r#"{"format_id":"d3-code-readout-v1"}"#,
        )
        .unwrap();
        for dtype in ["float32", "float16"] {
            let error = LibtorchBackend::new(&fixture.0, dtype, ModelType::Decision, "cpu")
                .err()
                .expect("D3 must reject unsupported precision");
            assert!(error.to_string().contains("D3 requires bfloat16"));
        }
    }

    #[test]
    fn synchronous_native_output_allows_spare_capacity() {
        let fixture = fixture("", false);
        let model = LibtorchBackend::new(
            &fixture.0,
            "float32",
            ModelType::Embedding(Pool::Mean),
            "cpu",
        )
        .unwrap();
        let mut output = [42.0f32; 20];
        // SAFETY: a valid two-token packed input and one 16-dimensional pooled
        // result; the caller intentionally supplies four unused output elements.
        let status = unsafe {
            tei_forward_radix(
                model.handle.as_ptr(),
                [1i64, 2].as_ptr(),
                [0i64, 0].as_ptr(),
                [0i64, 1].as_ptr(),
                [0i32, 2].as_ptr(),
                1,
                2,
                1,
                [0i64].as_ptr(),
                1,
                std::ptr::null(),
                0,
                output.as_mut_ptr(),
                output.len(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
            )
        };
        assert_eq!(status, 0, "{}", native_error());
        assert!(output[..16].iter().all(|value| value.is_finite()));
        assert_eq!(output[16..], [42.0; 4]);
    }

    #[test]
    fn image_feature_keys_and_descriptor_layout() {
        assert_eq!(std::mem::size_of::<NativeImage>(), 120);
        assert_eq!(std::mem::offset_of!(NativeImage, feature_key), 80);
        let image = text_embeddings_backend_core::ImagePatches {
            pixels: (0..18).map(|i| i as f32 * 0.125).collect(),
            grid_thw: [1, 3, 3],
            patch_dim: 2,
            merge_size: 3,
        };
        let expected = [
            0x47, 0xe6, 0x4d, 0xf4, 0x66, 0x26, 0x25, 0x20, 0x1a, 0x29, 0x71, 0xb6, 0xd2, 0x85,
            0x68, 0xf0, 0x4c, 0x19, 0xa7, 0x5a, 0x34, 0x8e, 0x9f, 0x59, 0xb3, 0x1d, 0xe4, 0x73,
            0x6c, 0x43, 0x63, 0x08,
        ];
        assert_eq!(image_feature_key(&image), expected);
        assert_eq!(image_feature_key(&image.clone()), expected);
        let mut changed = image.clone();
        changed.pixels[0] = 0.5;
        assert_ne!(image_feature_key(&changed), expected);
        for field in 0..5 {
            let mut changed = image.clone();
            match field {
                0..=2 => changed.grid_thw[field] += 1,
                3 => changed.patch_dim += 1,
                _ => changed.merge_size += 1,
            }
            assert_ne!(image_feature_key(&changed), expected);
        }
    }

    use safetensors::tensor::{serialize, TensorView};
    use std::sync::atomic::{AtomicUsize, Ordering};
    #[cfg(not(feature = "benchmark-cuda"))]
    use text_embeddings_backend_candle::CandleBackend;

    struct Fixture(std::path::PathBuf);
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    fn fixture(prefix: &str, sharded: bool) -> Fixture {
        fixture_with_intermediate(prefix, sharded, 24)
    }
    fn fixture_with_intermediate(prefix: &str, sharded: bool, intermediate: usize) -> Fixture {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "tei-libtorch-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        let config = serde_json::json!({"model_type":"bert", "hidden_size":16, "num_attention_heads":2,
            "num_hidden_layers":2, "intermediate_size":intermediate, "vocab_size":16, "max_position_embeddings":16,
            "type_vocab_size":2, "id2label":{"0":"a","1":"b","2":"c"}, "layer_norm_eps":1e-5, "hidden_act":"gelu_pytorch_tanh", "pad_token_id":0,
            "hidden_dropout_prob":0.0, "attention_probs_dropout_prob":0.0, "initializer_range":0.02});
        fs::write(
            path.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        let mut weights = Vec::new();
        let mut add = |name: String, shape: Vec<usize>| {
            let salt = weights.len();
            let values = (0..shape.iter().product())
                .map(|i| {
                    let value = (((i * 17 + salt * 13) % 101) as f32 - 50.) / 200.;
                    if name.ends_with("LayerNorm.weight") {
                        1.0 + value
                    } else {
                        value
                    }
                })
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>();
            weights.push((format!("{prefix}{name}"), shape, values));
        };
        add("embeddings.word_embeddings.weight".into(), vec![16, 16]);
        add("embeddings.position_embeddings.weight".into(), vec![16, 16]);
        add(
            "embeddings.token_type_embeddings.weight".into(),
            vec![2, 16],
        );
        add("embeddings.LayerNorm.weight".into(), vec![16]);
        add("embeddings.LayerNorm.bias".into(), vec![16]);
        for i in 0..2 {
            let p = format!("encoder.layer.{i}.");
            for name in [
                "attention.self.query",
                "attention.self.key",
                "attention.self.value",
                "attention.output.dense",
            ] {
                add(format!("{p}{name}.weight"), vec![16, 16]);
                add(format!("{p}{name}.bias"), vec![16]);
            }
            for name in ["attention.output.LayerNorm", "output.LayerNorm"] {
                add(format!("{p}{name}.weight"), vec![16]);
                add(format!("{p}{name}.bias"), vec![16]);
            }
            add(
                format!("{p}intermediate.dense.weight"),
                vec![intermediate, 16],
            );
            add(format!("{p}intermediate.dense.bias"), vec![intermediate]);
            add(format!("{p}output.dense.weight"), vec![16, intermediate]);
            add(format!("{p}output.dense.bias"), vec![16]);
        }
        if prefix.is_empty() {
            add("classifier.weight".into(), vec![3, 16]);
            add("classifier.bias".into(), vec![3]);
            add(
                "cls.predictions.transform.dense.weight".into(),
                vec![16, 16],
            );
            add("cls.predictions.transform.dense.bias".into(), vec![16]);
            add(
                "cls.predictions.transform.LayerNorm.weight".into(),
                vec![16],
            );
            add("cls.predictions.transform.LayerNorm.bias".into(), vec![16]);
            add("cls.predictions.decoder.weight".into(), vec![16, 16]);
            add("cls.predictions.bias".into(), vec![16]);
        }
        let views = weights
            .iter()
            .map(|(name, shape, data)| {
                (
                    name.clone(),
                    TensorView::new(Dtype::F32, shape.clone(), data).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        if sharded {
            let mut map = serde_json::Map::new();
            for (i, slice) in views.chunks(views.len().div_ceil(2)).enumerate() {
                let file = format!("model-{i}.safetensors");
                fs::write(
                    path.join(&file),
                    serialize(slice.iter().map(|(n, t)| (n.clone(), t.clone())), None).unwrap(),
                )
                .unwrap();
                for (name, _) in slice {
                    map.insert(name.clone(), serde_json::json!(file));
                }
            }
            fs::write(
                path.join("model.safetensors.index.json"),
                serde_json::to_vec(&serde_json::json!({"weight_map":map})).unwrap(),
            )
            .unwrap();
        } else {
            fs::write(
                path.join("model.safetensors"),
                serialize(views, None).unwrap(),
            )
            .unwrap();
        }
        Fixture(path)
    }
    fn batch() -> Batch {
        Batch {
            input_ids: vec![1, 5, 7, 2, 4],
            token_type_ids: vec![0, 0, 1, 1, 1],
            position_ids: vec![0, 1, 2, 0, 1],
            cumulative_seq_lengths: vec![0, 3, 5],
            max_length: 3,
            pooled_indices: vec![1],
            raw_indices: vec![0],
            multimodal: vec![],
            compact_input_ids: None,
            compact_position_ids: None,
            scatter_unfold: None,
            fold_gather: None,
            tokens: vec![],
            offsets: vec![],
        }
    }
    fn values(value: &Embedding) -> Vec<f32> {
        match value {
            Embedding::Pooled(v) => v.clone(),
            Embedding::All(v) => v.iter().flatten().copied().collect(),
        }
    }
    fn dense_fixture() -> (Fixture, Vec<String>) {
        let fixture = fixture("", false);
        let paths = vec!["2_Dense".to_owned(), "3_Dense".to_owned()];
        fs::write(fixture.0.join("modules.json"), serde_json::to_vec(&serde_json::json!([
            {"idx":0,"name":"0","path":"","type":"sentence_transformers.models.Transformer"},
            {"idx":1,"name":"1","path":"1_Pooling","type":"sentence_transformers.models.Pooling"},
            {"idx":2,"name":"2","path":"2_Dense","type":"sentence_transformers.models.Dense"},
            {"idx":3,"name":"3","path":"3_Dense","type":"sentence_transformers.models.Dense"}
        ])).unwrap()).unwrap();
        for (index, (input, output)) in [(16, 7), (7, 3)].into_iter().enumerate() {
            let dir = fixture.0.join(&paths[index]);
            fs::create_dir(&dir).unwrap();
            fs::write(dir.join("config.json"), serde_json::to_vec(&serde_json::json!({
                "in_features":input,"out_features":output,"bias":index == 0,
                "activation_function":if index == 0 { "torch.nn.modules.activation.Tanh" } else { "torch.nn.modules.linear.Identity" }
            })).unwrap()).unwrap();
            let weight = (0..input * output)
                .flat_map(|i| (((i % 11) as f32 - 5.) * 0.04).to_le_bytes())
                .collect::<Vec<_>>();
            let bias = (0..output)
                .flat_map(|i| (i as f32 * 0.01).to_le_bytes())
                .collect::<Vec<_>>();
            let mut tensors = vec![(
                "linear.weight",
                TensorView::new(Dtype::F32, vec![output, input], &weight).unwrap(),
            )];
            if index == 0 {
                tensors.push((
                    "linear.bias",
                    TensorView::new(Dtype::F32, vec![output], &bias).unwrap(),
                ));
            }
            fs::write(
                dir.join("model.safetensors"),
                serialize(tensors, None).unwrap(),
            )
            .unwrap();
        }
        (fixture, paths)
    }
    #[cfg(not(feature = "benchmark-cuda"))]
    #[test]
    fn dense_chain_matches_candle_and_preserves_raw_width() {
        let (fixture, paths) = dense_fixture();
        for pool in [Pool::Cls, Pool::Mean, Pool::LastToken] {
            let model_type = ModelType::Embedding(pool);
            let native =
                LibtorchBackend::new(&fixture.0, "float32", model_type.clone(), "cpu").unwrap();
            let candle = CandleBackend::new(
                &fixture.0,
                "float32".into(),
                model_type,
                Some(paths.clone()),
                0,
            )
            .unwrap();
            let expected = candle.embed(batch()).unwrap();
            let actual = native.embed(batch()).unwrap();
            for (row, embedding) in &actual {
                match embedding {
                    Embedding::Pooled(v) => assert_eq!(v.len(), 3),
                    Embedding::All(v) => assert!(v.iter().all(|token| token.len() == 16)),
                }
                let got = values(embedding);
                let want = values(&expected[row]);
                assert_eq!(got.len(), want.len());
                assert!(got.iter().zip(want).all(|(a, b)| (a - b).abs() < 0.0001));
            }
            let mut canonical_raw = batch();
            canonical_raw.pooled_indices.clear();
            canonical_raw.raw_indices = (0..canonical_raw.len() as u32).collect();
            let canonical_raw = candle.embed(canonical_raw).unwrap();
            // Duplicate selectors preserve request order; raw overwrites pooled
            // when both request the same sequence, matching Candle's public map.
            for (pooled, raw) in [
                (vec![0, 0], vec![]),
                (vec![1], vec![]),
                (vec![1, 0], vec![]),
                (vec![0, 1, 0, 1], vec![]),
                (vec![], vec![]),
                (vec![], vec![0, 0]),
                (vec![], vec![1]),
                (vec![], vec![1, 0]),
                (vec![], vec![0, 1, 0, 1]),
                (vec![1, 1], vec![0]),
                (vec![1], vec![0, 0]),
                (vec![1, 0, 1], vec![0, 1, 0]),
            ] {
                let mut selected = batch();
                selected.pooled_indices = pooled;
                selected.raw_indices = raw;
                let expected = candle.embed(selected.clone()).unwrap();
                let actual = native.embed(selected).unwrap();
                assert_eq!(actual.len(), expected.len());
                for (row, got) in &actual {
                    match (got, &expected[row]) {
                        (Embedding::Pooled(v), Embedding::Pooled(_)) => assert_eq!(v.len(), 3),
                        (Embedding::All(v), Embedding::All(w)) => {
                            assert_eq!(v.len(), w.len());
                            assert!(v.iter().all(|token| token.len() == 16));
                            assert_eq!(values(&expected[row]), values(&canonical_raw[row]));
                        }
                        _ => panic!("Duplicate selector changed pooled/raw overwrite kind"),
                    }
                    let want = values(&expected[row]);
                    assert!(values(got)
                        .iter()
                        .zip(want)
                        .all(|(a, b)| (a - b).abs() < 0.0001));
                }
            }
            let mut single = batch();
            single.input_ids.truncate(3);
            single.token_type_ids.truncate(3);
            single.position_ids.truncate(3);
            single.cumulative_seq_lengths.truncate(2);
            single.pooled_indices.clear();
            single.raw_indices = vec![0];
            let canonical = candle.embed(single.clone()).unwrap();
            for raw in [vec![0, 0], vec![0], vec![0, 0, 0, 0], vec![]] {
                single.raw_indices = raw.clone();
                let expected = candle.embed(single.clone()).unwrap();
                let actual = native.embed(single.clone()).unwrap();
                assert_eq!(expected.len(), usize::from(!raw.is_empty()));
                assert_eq!(actual.len(), expected.len());
                if !raw.is_empty() {
                    assert_eq!(values(&expected[&0]), values(&canonical[&0]));
                    let want = values(&expected[&0]);
                    assert!(values(&actual[&0])
                        .iter()
                        .zip(want)
                        .all(|(a, b)| (a - b).abs() < 0.0001));
                }
            }
        }
        fs::write(
            fixture.0.join("2_Dense/config.json"),
            br#"{"in_features":15,"out_features":7,"bias":true}"#,
        )
        .unwrap();
        assert!(LibtorchBackend::new(
            &fixture.0,
            "float32",
            ModelType::Embedding(Pool::Cls),
            "cpu"
        )
        .is_err());
    }

    #[test]
    fn pytorch_checkpoint_fallback_preserves_dense_and_raw_outputs() {
        for (base_bin, dense_bin) in [(true, false), (false, true), (true, true)] {
            let (fixture, paths) = dense_fixture();
            let model_type = ModelType::Embedding(Pool::Mean);
            // Safetensors takes priority over a present incompatible fallback.
            fs::write(fixture.0.join("pytorch_model.bin"), b"invalid pickle").unwrap();
            let baseline = LibtorchBackend::new(&fixture.0, "float32", model_type.clone(), "cpu")
                .unwrap()
                .embed(batch())
                .unwrap();
            if base_bin {
                pytorch_checkpoint::from_safetensors(
                    &fixture.0.join("model.safetensors"),
                    &fixture.0.join("pytorch_model.bin"),
                );
                fs::remove_file(fixture.0.join("model.safetensors")).unwrap();
            }
            if dense_bin {
                for path in &paths {
                    let dir = fixture.0.join(path);
                    pytorch_checkpoint::from_safetensors(
                        &dir.join("model.safetensors"),
                        &dir.join("pytorch_model.bin"),
                    );
                    fs::remove_file(dir.join("model.safetensors")).unwrap();
                }
            }
            let actual = LibtorchBackend::new(&fixture.0, "float32", model_type, "cpu")
                .unwrap()
                .embed(batch())
                .unwrap();
            for (index, expected) in &baseline {
                assert_eq!(values(&actual[index]), values(expected));
            }
            if base_bin {
                fs::write(fixture.0.join("pytorch_model.bin"), b"invalid pickle").unwrap();
                assert!(matches!(
                    LibtorchBackend::new(
                        &fixture.0,
                        "float32",
                        ModelType::Embedding(Pool::Mean),
                        "cpu",
                    ),
                    Err(BackendError::Start(_))
                ));
            }
        }
    }
    #[test]
    #[ignore = "Requires an NVIDIA GPU and CUDA LibTorch 2.14.1"]
    fn cuda_bulk_completion_preserves_mixed_projection_widths() {
        let (fixture, _) = dense_fixture();
        let mut gpu = LibtorchBackend::new(
            &fixture.0,
            "float16",
            ModelType::Embedding(Pool::Mean),
            "cuda:0",
        )
        .unwrap();
        // The same CUDA path also accepts a valid batch with no requested rows.
        let mut empty = batch();
        empty.pooled_indices.clear();
        empty.raw_indices.clear();
        assert!(gpu
            .submit_embed(empty)
            .unwrap()
            .complete()
            .unwrap()
            .is_empty());
        let input = |pattern: u32| {
            let mut input = batch();
            input.input_ids = (0..6400).map(|i| (i + pattern) % 16).collect();
            input.token_type_ids = vec![0; 6400];
            input.position_ids = (0..6400).map(|i| i % 4).collect();
            input.cumulative_seq_lengths = (0..=1600).map(|i| i * 4).collect();
            input.max_length = 4;
            input.pooled_indices = (0..400).collect();
            input.raw_indices = (400..1600).collect();
            input
        };
        // Use the existing synchronous native ABI as a full-output reference.
        gpu.cuda = false;
        let expected = [gpu.embed(input(0)).unwrap(), gpu.embed(input(3)).unwrap()];
        gpu.cuda = true;
        let first = gpu.submit_embed(input(0)).unwrap();
        let second = gpu.submit_embed(input(3)).unwrap();
        assert!(matches!(first, Inference::Deferred(_)));
        assert!(matches!(second, Inference::Deferred(_)));
        let actual =
            std::thread::spawn(move || [first.complete().unwrap(), second.complete().unwrap()])
                .join()
                .unwrap();
        for (expected, actual) in expected.into_iter().zip(actual) {
            assert_eq!(actual.len(), expected.len());
            for (index, value) in actual {
                assert_eq!(values(&value), values(&expected[&index]));
            }
        }
        // Tiny pooled results use the same staged path as larger outputs.
        gpu.cuda = false;
        let expected = gpu.embed(batch()).unwrap();
        gpu.cuda = true;
        let tiny = gpu.submit_embed(batch()).unwrap();
        assert!(matches!(tiny, Inference::Deferred(_)));
        let actual = tiny.complete().unwrap();
        for (index, value) in actual {
            assert_eq!(values(&value), values(&expected[&index]));
        }
    }

    #[test]
    #[ignore = "Requires an NVIDIA GPU and CUDA LibTorch 2.14.1"]
    fn cuda_dense_chain_matches_cpu_with_mixed_widths() {
        let (fixture, _) = dense_fixture();
        for pool in [Pool::Cls, Pool::Mean, Pool::LastToken] {
            let model_type = ModelType::Embedding(pool);
            let cpu =
                LibtorchBackend::new(&fixture.0, "float32", model_type.clone(), "cpu").unwrap();
            let gpu = LibtorchBackend::new(&fixture.0, "float16", model_type, "cuda:0").unwrap();
            let expected = cpu.embed(batch()).unwrap();
            let actual = gpu.embed(batch()).unwrap();
            for (row, embedding) in actual {
                let got = values(&embedding);
                let want = values(&expected[&row]);
                assert_eq!(got.len(), want.len());
                assert!(got.iter().zip(want).all(|(a, b)| (a - b).abs() < 0.02));
                match embedding {
                    Embedding::Pooled(v) => assert_eq!(v.len(), 3),
                    Embedding::All(v) => assert!(v.iter().all(|token| token.len() == 16)),
                }
            }
        }
    }
    #[cfg(not(feature = "benchmark-cuda"))]
    #[test]
    fn native_classifier_and_splade_match_candle() {
        let fixture = fixture("", false);
        let torch =
            LibtorchBackend::new(&fixture.0, "float32", ModelType::Classifier, "cpu").unwrap();
        let candle =
            CandleBackend::new(&fixture.0, "float32".into(), ModelType::Classifier, None, 0)
                .unwrap();
        let mut classifier_batch = batch();
        classifier_batch.pooled_indices = vec![0, 1];
        classifier_batch.raw_indices.clear();
        let expected = candle.predict(classifier_batch.clone()).unwrap();
        let actual = torch.predict(classifier_batch).unwrap();
        for (row, values) in expected {
            for (a, b) in actual[&row].iter().zip(values) {
                assert!((a - b).abs() < 0.0001);
            }
        }
        let mut token_batch = batch();
        token_batch.pooled_indices.clear();
        token_batch.raw_indices = vec![0, 1];
        let expected = candle.predict_tokens(token_batch.clone()).unwrap();
        let actual = torch.predict_tokens(token_batch).unwrap();
        for (row, values) in expected {
            for (a, b) in actual[&row].iter().flatten().zip(values.iter().flatten()) {
                assert!((a - b).abs() < 0.0001);
            }
        }
        let torch = LibtorchBackend::new(
            &fixture.0,
            "float32",
            ModelType::Embedding(Pool::Splade),
            "cpu",
        )
        .unwrap();
        let candle = CandleBackend::new(
            &fixture.0,
            "float32".into(),
            ModelType::Embedding(Pool::Splade),
            None,
            0,
        )
        .unwrap();
        let actual = torch.embed(batch()).unwrap();
        let expected = candle.embed(batch()).unwrap();
        for (row, value) in expected {
            let actual = values(&actual[&row]);
            let expected = values(&value);
            assert_eq!(actual.len(), expected.len());
            for (a, b) in actual.iter().zip(expected) {
                assert!((a - b).abs() < 0.0001);
            }
        }
    }
    #[cfg(not(feature = "benchmark-cuda"))]
    #[test]
    fn bert_relu_and_silu_match_candle_for_embeddings_and_heads() {
        for activation in ["relu", "silu"] {
            let fixture = fixture("", false);
            let mut config: serde_json::Value =
                serde_json::from_slice(&fs::read(fixture.0.join("config.json")).unwrap()).unwrap();
            config["hidden_act"] = activation.into();
            fs::write(
                fixture.0.join("config.json"),
                serde_json::to_vec(&config).unwrap(),
            )
            .unwrap();
            for pool in [Pool::Cls, Pool::Mean, Pool::LastToken, Pool::Splade] {
                let model_type = ModelType::Embedding(pool);
                let torch =
                    LibtorchBackend::new(&fixture.0, "float32", model_type.clone(), "cpu").unwrap();
                let candle =
                    CandleBackend::new(&fixture.0, "float32".into(), model_type, None, 0).unwrap();
                let actual = torch.embed(batch()).unwrap();
                for (row, expected) in candle.embed(batch()).unwrap() {
                    let got = values(&actual[&row]);
                    let want = values(&expected);
                    assert_eq!(got.len(), want.len());
                    assert!(
                        got.iter().zip(want).all(|(a, b)| (a - b).abs() < 0.0001),
                        "{activation} embedding row {row}"
                    );
                }
            }
            let torch =
                LibtorchBackend::new(&fixture.0, "float32", ModelType::Classifier, "cpu").unwrap();
            let candle =
                CandleBackend::new(&fixture.0, "float32".into(), ModelType::Classifier, None, 0)
                    .unwrap();
            let mut input = batch();
            input.pooled_indices = vec![0, 1];
            input.raw_indices.clear();
            let actual = torch.predict(input.clone()).unwrap();
            for (row, expected) in candle.predict(input.clone()).unwrap() {
                assert!(
                    actual[&row]
                        .iter()
                        .zip(expected)
                        .all(|(a, b)| (a - b).abs() < 0.0001),
                    "{activation} classifier row {row}"
                );
            }
            input.pooled_indices.clear();
            input.raw_indices = vec![0, 1];
            let actual = torch.predict_tokens(input.clone()).unwrap();
            for (row, expected) in candle.predict_tokens(input).unwrap() {
                assert!(
                    actual[&row]
                        .iter()
                        .flatten()
                        .zip(expected.iter().flatten())
                        .all(|(a, b)| (a - b).abs() < 0.0001),
                    "{activation} token classifier row {row}"
                );
            }
            config["hidden_act"] = "unsupported".into();
            fs::write(
                fixture.0.join("config.json"),
                serde_json::to_vec(&config).unwrap(),
            )
            .unwrap();
            assert!(LibtorchBackend::new(
                &fixture.0,
                "float32",
                ModelType::Embedding(Pool::Mean),
                "cpu"
            )
            .is_err());
        }
    }
    // Candle selects CUDA when compiled with it; CPU parity runs in the default feature build.
    #[cfg(not(feature = "benchmark-cuda"))]
    #[test]
    fn roberta_aliases_match_candle_with_offset_positions_and_heads() {
        for (family, prefix) in [
            ("roberta", "roberta."),
            ("xlm-roberta", "xlm-roberta."),
            ("camembert", "camembert."),
        ] {
            let fixture = fixture(prefix, false);
            let mut config: serde_json::Value =
                serde_json::from_slice(&fs::read(fixture.0.join("config.json")).unwrap()).unwrap();
            config["model_type"] = family.into();
            config["pad_token_id"] = 1.into();
            fs::write(
                fixture.0.join("config.json"),
                serde_json::to_vec(&config).unwrap(),
            )
            .unwrap();
            let bytes = fs::read(fixture.0.join("model.safetensors")).unwrap();
            let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
            let mut weights = tensors
                .tensors()
                .into_iter()
                .map(|(name, tensor)| (name, tensor.shape().to_vec(), tensor.data().to_vec()))
                .collect::<Vec<_>>();
            for (name, shape) in [
                ("classifier.dense.weight", vec![16, 16]),
                ("classifier.dense.bias", vec![16]),
                ("classifier.out_proj.weight", vec![3, 16]),
                ("classifier.out_proj.bias", vec![3]),
                ("lm_head.dense.weight", vec![16, 16]),
                ("lm_head.dense.bias", vec![16]),
                ("lm_head.layer_norm.weight", vec![16]),
                ("lm_head.layer_norm.bias", vec![16]),
                ("lm_head.decoder.weight", vec![16, 16]),
                ("lm_head.bias", vec![16]),
            ] {
                let data = (0..shape.iter().product())
                    .map(|i| {
                        let value = ((i * 13 % 79) as f32 - 39.) / 180.;
                        if name.ends_with("layer_norm.weight") {
                            1. + value
                        } else {
                            value
                        }
                    })
                    .flat_map(f32::to_le_bytes)
                    .collect::<Vec<_>>();
                weights.push((name.to_owned(), shape, data));
            }
            let views = weights.iter().map(|(name, shape, data)| {
                (
                    name.clone(),
                    TensorView::new(Dtype::F32, shape.clone(), data).unwrap(),
                )
            });
            fs::write(
                fixture.0.join("model.safetensors"),
                serialize(views, None).unwrap(),
            )
            .unwrap();
            let mut input = batch();
            for position in &mut input.position_ids {
                *position += 2;
            }
            for pool in [Pool::Cls, Pool::Mean, Pool::LastToken, Pool::Splade] {
                let kind = ModelType::Embedding(pool);
                let native =
                    LibtorchBackend::new(&fixture.0, "float32", kind.clone(), "cpu").unwrap();
                let candle =
                    CandleBackend::new(&fixture.0, "float32".into(), kind, None, 0).unwrap();
                let actual = native.embed(input.clone()).unwrap();
                let expected = candle.embed(input.clone()).unwrap();
                for (row, embedding) in actual {
                    let got = values(&embedding);
                    let want = values(&expected[&row]);
                    assert_eq!(got.len(), want.len());
                    assert!(
                        got.iter().zip(want).all(|(a, b)| (a - b).abs() < 0.0002),
                        "{family} pooling mismatch"
                    );
                }
            }
            let native =
                LibtorchBackend::new(&fixture.0, "float32", ModelType::Classifier, "cpu").unwrap();
            let candle =
                CandleBackend::new(&fixture.0, "float32".into(), ModelType::Classifier, None, 0)
                    .unwrap();
            input.pooled_indices = vec![0, 1];
            input.raw_indices.clear();
            let actual = native.predict(input.clone()).unwrap();
            let expected = candle.predict(input.clone()).unwrap();
            let embedding = LibtorchBackend::new(
                &fixture.0,
                "float32",
                ModelType::Embedding(Pool::Cls),
                "cpu",
            )
            .unwrap()
            .embed(input.clone())
            .unwrap();
            let parameters = weights
                .iter()
                .map(|(name, _, data)| {
                    (
                        name.as_str(),
                        data.chunks_exact(4)
                            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                            .collect::<Vec<_>>(),
                    )
                })
                .collect::<std::collections::HashMap<_, _>>();
            for (&row, got) in &actual {
                let hidden = values(&embedding[&row]);
                let mid = (0..16)
                    .map(|i| {
                        (parameters["classifier.dense.bias"][i]
                            + hidden
                                .iter()
                                .enumerate()
                                .map(|(j, x)| x * parameters["classifier.dense.weight"][i * 16 + j])
                                .sum::<f32>())
                        .tanh()
                    })
                    .collect::<Vec<_>>();
                let reference = (0..3)
                    .map(|i| {
                        parameters["classifier.out_proj.bias"][i]
                            + mid
                                .iter()
                                .enumerate()
                                .map(|(j, x)| {
                                    x * parameters["classifier.out_proj.weight"][i * 16 + j]
                                })
                                .sum::<f32>()
                    })
                    .collect::<Vec<_>>();
                assert!(got.iter().zip(&reference).all(|(a,b)| (a-b).abs()<0.0002), "Native head disagrees with independent scalar reference: {got:?} vs {reference:?}");
            }
            for (row, got) in actual {
                assert!(
                    got.iter()
                        .zip(&expected[&row])
                        .all(|(a, b)| (a - b).abs() < 0.0002),
                    "{family} classifier mismatch for row {row}: {got:?} vs {:?}",
                    expected[&row]
                );
            }
            input.pooled_indices.clear();
            input.raw_indices = vec![0, 1];
            let actual = native.predict_tokens(input.clone()).unwrap();
            let expected = candle.predict_tokens(input).unwrap();
            for (row, got) in actual {
                assert!(
                    got.iter()
                        .flatten()
                        .zip(expected[&row].iter().flatten())
                        .all(|(a, b)| (a - b).abs() < 0.0002),
                    "{family} token classifier mismatch"
                );
            }
        }
    }
    #[cfg(not(feature = "benchmark-cuda"))]
    #[test]
    fn native_bert_matches_candle_for_ragged_pooled_and_raw_outputs() {
        for (prefix, sharded) in [("", false), ("bert.", true)] {
            let fixture = fixture(prefix, sharded);
            for pool in [Pool::Cls, Pool::Mean, Pool::LastToken] {
                let model_type = ModelType::Embedding(pool);
                let torch =
                    LibtorchBackend::new(&fixture.0, "float32", model_type.clone(), "cpu").unwrap();
                let candle =
                    CandleBackend::new(&fixture.0, "float32".into(), model_type, None, 0).unwrap();
                let actual = torch.embed(batch()).unwrap();
                let expected = candle.embed(batch()).unwrap();
                for index in [0, 1] {
                    let actual = values(&actual[&index]);
                    let expected = values(&expected[&index]);
                    assert_eq!(actual.len(), expected.len());
                    for (a, e) in actual.iter().zip(&expected) {
                        assert!((a - e).abs() < 2e-4, "index {index}: {a} != {e}");
                    }
                }
                // Packed attention must preserve sequence boundaries.
                let mut short = batch();
                short.input_ids = vec![2, 4];
                short.token_type_ids = vec![1, 1];
                short.position_ids = vec![0, 1];
                short.cumulative_seq_lengths = vec![0, 2];
                short.pooled_indices = vec![0];
                short.raw_indices.clear();
                short.max_length = 2;
                let single = torch.embed(short).unwrap();
                for (a, e) in values(&single[&0]).iter().zip(values(&actual[&1])) {
                    assert!((a - e).abs() < 2e-5);
                }
            }
        }
    }
    #[test]
    #[ignore = "Requires an NVIDIA GPU and CUDA LibTorch 2.14.1"]
    fn cuda_varlen_matches_cpu_for_ragged_outputs() {
        let fixture = fixture("", false);
        let mut config: serde_json::Value =
            serde_json::from_slice(&fs::read(fixture.0.join("config.json")).unwrap()).unwrap();
        for activation in ["gelu_pytorch_tanh", "relu", "silu"] {
            config["hidden_act"] = activation.into();
            fs::write(
                fixture.0.join("config.json"),
                serde_json::to_vec(&config).unwrap(),
            )
            .unwrap();
            for pool in [Pool::Cls, Pool::Mean, Pool::LastToken, Pool::Splade] {
                let model_type = ModelType::Embedding(pool);
                let cpu =
                    LibtorchBackend::new(&fixture.0, "float32", model_type.clone(), "cpu").unwrap();
                let gpu =
                    LibtorchBackend::new(&fixture.0, "float16", model_type, "cuda:0").unwrap();
                let expected = cpu.embed(batch()).unwrap();
                let actual = std::thread::spawn(move || gpu.embed(batch()).unwrap())
                    .join()
                    .unwrap();
                for index in [0, 1] {
                    for (a, e) in values(&actual[&index])
                        .iter()
                        .zip(values(&expected[&index]))
                    {
                        assert!((a - e).abs() < 1e-2, "CUDA varlen {activation}: {a} != {e}");
                    }
                }
            }
        }
    }

    #[test]
    fn dynamic_fp8_scope_defaults_and_overrides() {
        for family in [
            "llama",
            "mistral",
            "qwen2",
            "qwen3",
            "qwen3_5",
            "qwen3_5_text",
        ] {
            assert_eq!(resolve_fp8_scope(None, true, family).unwrap(), "all");
            assert_eq!(resolve_fp8_scope(Some("all"), true, family).unwrap(), "all");
            assert_eq!(resolve_fp8_scope(Some("mlp"), true, family).unwrap(), "mlp");
        }
        for family in ["bert", "roberta", "xlm-roberta", "camembert", "modernbert"] {
            assert_eq!(resolve_fp8_scope(None, true, family).unwrap(), "mlp");
            assert!(resolve_fp8_scope(Some("all"), true, family).is_err());
        }
        for family in [
            "qwen3",
            "qwen3_5",
            "qwen3_5_text",
            "modernbert",
            "qwen3_moe",
            "gemma4",
        ] {
            assert_eq!(resolve_fp8_scope(None, false, family).unwrap(), "mlp");
            assert_eq!(
                resolve_fp8_scope(Some("mlp"), false, family).unwrap(),
                "mlp"
            );
            assert!(resolve_fp8_scope(Some("all"), false, family).is_err());
        }
        assert!(resolve_fp8_scope(Some("unknown"), true, "qwen3").is_err());
        assert!(resolve_fp8_scope(Some("all"), true, "qwen3_moe").is_err());
        assert!(resolve_fp8_scope(Some("all"), true, "qwen3_5_moe").is_err());
        assert!(resolve_fp8_scope(Some("all"), true, "qwen3_5_moe_text").is_err());
    }

    #[test]
    fn dynamic_fp8_rejects_cpu_and_requantized_checkpoints() {
        let fixture = fixture_with_intermediate("", false, 32);
        let cpu = LibtorchBackend::new_with_dense_paths_and_fp8(
            &fixture.0,
            "float32",
            ModelType::Embedding(Pool::Mean),
            "cpu",
            None,
            true,
        )
        .err()
        .unwrap();
        assert!(cpu.to_string().contains("requires CUDA"));
        let mut config: serde_json::Value =
            serde_json::from_slice(&fs::read(fixture.0.join("config.json")).unwrap()).unwrap();
        config["quantization_config"] = serde_json::json!({"quant_method":"fp8"});
        fs::write(
            fixture.0.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        let quantized = LibtorchBackend::new_with_dense_paths_and_fp8(
            &fixture.0,
            "float16",
            ModelType::Embedding(Pool::Mean),
            "cuda:0",
            None,
            true,
        )
        .err()
        .unwrap();
        assert!(quantized.to_string().contains("unquantized checkpoint"));
    }

    #[test]
    #[ignore = "Requires an FP8-capable NVIDIA GPU and CUDA LibTorch 2.14.1"]
    fn cuda_dynamic_fp8_bert_preserves_ragged_outputs_and_heads() {
        let fixture = fixture_with_intermediate("", false, 32);
        for dtype in ["float16", "bfloat16"] {
            for pool in [Pool::Mean, Pool::Splade] {
                let dense = LibtorchBackend::new(
                    &fixture.0,
                    dtype,
                    ModelType::Embedding(pool.clone()),
                    "cuda:0",
                )
                .unwrap();
                let quantized = LibtorchBackend::new_with_dense_paths_and_fp8(
                    &fixture.0,
                    dtype,
                    ModelType::Embedding(pool),
                    "cuda:0",
                    None,
                    true,
                )
                .unwrap();
                let expected = dense.embed(batch()).unwrap();
                let actual = quantized.embed(batch()).unwrap();
                let mut changed = false;
                for (row, embedding) in &actual {
                    let got = values(embedding);
                    let want = values(&expected[row]);
                    assert_eq!(got.len(), want.len());
                    assert!(got.iter().all(|v| v.is_finite()));
                    let error: f32 = got.iter().zip(&want).map(|(a, b)| (a - b).powi(2)).sum();
                    let norm: f32 = want.iter().map(|v| v.powi(2)).sum();
                    assert!((error / norm.max(1e-12)).sqrt() < 0.10);
                    changed |= got != want;
                }
                assert!(changed, "FP8 flag must change dense MLP arithmetic");
                let mut different = batch();
                different.input_ids[0] = 7;
                let replay = quantized.embed(different).unwrap();
                assert_ne!(values(&actual[&0]), values(&replay[&0]));
            }
            let classifier = LibtorchBackend::new_with_dense_paths_and_fp8(
                &fixture.0,
                dtype,
                ModelType::Classifier,
                "cuda:0",
                None,
                true,
            )
            .unwrap();
            let mut classifier_batch = batch();
            classifier_batch.pooled_indices = vec![0, 1];
            classifier_batch.raw_indices.clear();
            let logits = classifier.predict(classifier_batch).unwrap();
            assert_eq!(logits.len(), 2);
            assert!(logits
                .values()
                .all(|row| row.len() == 3 && row.iter().all(|v| v.is_finite())));
        }
    }

    #[test]
    fn malformed_batches_and_native_weight_errors_are_recoverable() {
        let fixture = fixture("", false);
        let torch = LibtorchBackend::new(
            &fixture.0,
            "float32",
            ModelType::Embedding(Pool::Mean),
            "cpu",
        )
        .unwrap();
        assert!(
            !torch.supports_radix_mlp(),
            "BERT must not advertise causal folding"
        );
        let mut bad = batch();
        bad.cumulative_seq_lengths = vec![];
        assert!(torch.embed(bad).is_err());
        let mut bad = batch();
        bad.input_ids[0] = 16;
        assert!(torch.embed(bad).is_err());
        let mut bad = batch();
        bad.pooled_indices = vec![2];
        assert!(torch.embed(bad).is_err());
        let mut bad = batch();
        bad.compact_input_ids = Some(bad.input_ids.clone());
        bad.compact_position_ids = Some(bad.position_ids.clone());
        let identity: Vec<u32> = (0..bad.input_ids.len() as u32).collect();
        bad.scatter_unfold = Some(identity.clone());
        bad.fold_gather = Some(identity);
        let error = torch.embed(bad).err().unwrap();
        assert!(error.to_string().contains("does not support radix folding"));
        assert!(torch.embed(batch()).is_ok());
        fs::write(
            fixture.0.join("model.safetensors"),
            serialize(Vec::<(String, TensorView<'_>)>::new(), None).unwrap(),
        )
        .unwrap();
        let err = LibtorchBackend::new(
            &fixture.0,
            "float32",
            ModelType::Embedding(Pool::Mean),
            "cpu",
        )
        .err()
        .unwrap();
        assert!(matches!(err, BackendError::Start(_)));
    }
}
