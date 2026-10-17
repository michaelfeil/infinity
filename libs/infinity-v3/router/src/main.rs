use anyhow::{Context, Result};
use clap::{CommandFactory, FromArgMatches, Parser};
use infinity_v3_backend::DType;
use opentelemetry::global;
use veil::Redact;

#[cfg(not(target_os = "linux"))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn pct_parser(s: &str) -> Result<f32, String> {
    let v = s.parse::<f32>().map_err(|e| e.to_string())?;
    if !(0.0..=1.0).contains(&v) {
        return Err(format!(
            "The value must be between 0.0 and 1.0, but got {}",
            v
        ));
    }
    Ok(v)
}

#[derive(Clone, Debug)]
struct Optional<T>(Option<T>);
fn optional<T>(value: &str) -> Result<Optional<T>, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    if value.eq_ignore_ascii_case("none") {
        Ok(Optional(None))
    } else {
        value
            .parse()
            .map(|v| Optional(Some(v)))
            .map_err(|e: T::Err| e.to_string())
    }
}
fn optional_string(value: &str) -> Result<Optional<String>, String> {
    optional(value)
}
fn optional_usize(value: &str) -> Result<Optional<usize>, String> {
    optional(value)
}
fn optional_pool(value: &str) -> Result<Optional<infinity_v3_backend::Pool>, String> {
    if matches!(value, "none" | "auto") {
        Ok(Optional(None))
    } else {
        clap::ValueEnum::from_str(value, false).map(|v| Optional(Some(v)))
    }
}
fn optional_protocol(
    value: &str,
) -> Result<Optional<infinity_v3_router::DecisionProtocol>, String> {
    if value == "none" {
        Ok(Optional(None))
    } else {
        clap::ValueEnum::from_str(value, false).map(|v| Optional(Some(v)))
    }
}
fn expand<T: Clone>(values: &[T], count: usize, name: &str) -> Result<Vec<T>> {
    anyhow::ensure!(
        values.is_empty() || values.len() == 1 || values.len() == count,
        "--{name} expects one value or {count} values, got {}",
        values.len()
    );
    Ok(if values.len() == 1 {
        vec![values[0].clone(); count]
    } else {
        values.to_vec()
    })
}
fn expand_optional<T: Clone>(
    values: &[Optional<T>],
    count: usize,
    name: &str,
) -> Result<Vec<Option<T>>> {
    if values.is_empty() {
        Ok(vec![None; count])
    } else {
        Ok(expand(values, count, name)?
            .into_iter()
            .map(|v| v.0)
            .collect())
    }
}

/// App Configuration
#[derive(Parser, Redact)]
#[clap(name = "infinity_v3", author, version, about, long_about = None)]
struct Args {
    /// The Hugging Face model ID, can be any model listed on <https://huggingface.co/models> with
    /// the `text-embeddings-inference` tag (meaning it's compatible with Text Embeddings
    /// Inference)
    ///
    /// Alternatively, the specified ID can also be a path to a local directory containing the
    /// necessary model files saved by the `save_pretrained(...)` methods of either Transformers or
    /// Sentence Transformers.
    /// The aliases `pplx-decider-v1.1-27b` and `d3` select their native decision protocols.
    #[clap(
        long,
        env = infinity_v3_config::infinity_env!("MODEL_ID"),
        required = true,
        value_delimiter = ';'
    )]
    #[redact(partial)]
    model_id: Vec<String>,

    /// Public model names. Repeat per model; defaults to the checkpoint ID.
    #[clap(long, env = infinity_v3_config::infinity_env!("SERVED_MODEL_NAME"), value_delimiter = ';', value_parser = optional_string)]
    served_model_name: Vec<Optional<String>>,

    /// The actual revision of the model if you're referring to a model
    /// on the hub. You can use a specific commit id or a branch like `refs/pr/2`.
    #[clap(long, env = infinity_v3_config::infinity_env!("REVISION"), value_parser = optional_string, value_delimiter = ';')]
    revision: Vec<Optional<String>>,

    /// Optionally control the number of tokenizer workers used for payload tokenization, validation
    /// and truncation.
    /// Defaults to CPU cores minus one. Values are clamped to 1–16.
    #[clap(long, env = infinity_v3_config::infinity_env!("TOKENIZATION_WORKERS"), value_parser = optional_usize, value_delimiter = ';')]
    tokenization_workers: Vec<Optional<usize>>,

    /// Maximum waiting tokenizer jobs, independent of worker count. Counts individual inputs,
    /// not HTTP requests. When full, producers wait for space.
    #[clap(
        long,
        env = infinity_v3_config::infinity_env!("TOKENIZATION_QUEUE_CAPACITY"),
        default_value = "1024",
        value_delimiter = ';'
    )]
    tokenization_queue_capacity: Vec<std::num::NonZeroUsize>,

    /// Model dtype. Auto selects bfloat16 from model config when supported, otherwise the backend default.
    #[clap(
        long,
        env = infinity_v3_config::infinity_env!("DTYPE"),
        value_enum,
        default_value = "auto",
        value_delimiter = ';'
    )]
    dtype: Vec<DType>,

    /// Advanced backend override; normally determined by the build/image.
    /// Runtime selection is rarely needed; single-backend builds select it automatically.
    /// When both are compiled, defaults to Candle. This flag cannot enable an uncompiled backend.
    #[clap(
        long,
        env = infinity_v3_config::infinity_env!("BACKEND"),
        value_enum,
        default_value = infinity_v3_backend::BackendKind::default().as_str(),
        value_delimiter = ';'
    )]
    backend: Vec<infinity_v3_backend::BackendKind>,

    /// LibTorch accelerator family. Auto selects CUDA when available, otherwise CPU.
    #[clap(
        long,
        env = infinity_v3_config::infinity_env!("TORCH_DEVICE"),
        value_enum,
        default_value = "auto",
        value_delimiter = ';'
    )]
    torch_device: Vec<infinity_v3_backend::TorchDevice>,

    /// Optionally control the pooling method for embedding models.
    ///
    /// If `pooling` is not set, the pooling configuration will be parsed from the
    /// model `1_Pooling/config.json` configuration.
    ///
    /// If `pooling` is set, it will override the model pooling configuration
    #[clap(long, env = infinity_v3_config::infinity_env!("POOLING"), value_parser = optional_pool, value_delimiter = ';')]
    pooling: Vec<Optional<infinity_v3_backend::Pool>>,

    /// Serve a decision model with its trained prompt on /v1/systemone.
    #[clap(long, env = infinity_v3_config::infinity_env!("DECISION_PROTOCOL"), value_parser = optional_protocol, value_delimiter = ';')]
    decision_protocol: Vec<Optional<infinity_v3_router::DecisionProtocol>>,

    /// Exact HTTPS hostnames allowed for image downloads. Empty permits inline data URLs only.
    #[clap(long, env = infinity_v3_config::infinity_env!("IMAGE_ALLOWED_HOSTS"), value_delimiter = ',')]
    image_allowed_hosts: Vec<String>,

    /// Memory budget for encoded images, CPU processing and prepared image patches, in MiB.
    #[clap(
        long,
        env = infinity_v3_config::infinity_env!("IMAGE_MEMORY_BUDGET_MIB"),
        default_value = "512",
        value_delimiter = ';'
    )]
    image_memory_budget_mib: Vec<usize>,

    /// The maximum amount of concurrent requests for this particular deployment.
    /// Having a low limit will refuse clients requests instead of having them
    /// wait for too long and is usually good to handle backpressure correctly.
    #[clap(
        default_value = "512",
        long,
        env = infinity_v3_config::infinity_env!("MAX_CONCURRENT_REQUESTS"),
        value_delimiter = ';'
    )]
    max_concurrent_requests: Vec<usize>,

    /// **IMPORTANT** This is one critical control to allow maximum usage
    /// of the available hardware.
    ///
    /// This represents the total amount of potential tokens within a batch.
    ///
    /// For `max_batch_tokens=1000`, you could fit `10` queries of `total_tokens=100`
    /// or a single query of `1000` tokens.
    ///
    /// Overall this number should be the largest possible until the model is compute bound.
    /// Since the actual memory overhead depends on the model implementation,
    /// infinity_v3 cannot infer this number automatically.
    #[clap(
        default_value = "16384",
        long,
        env = infinity_v3_config::infinity_env!("MAX_BATCH_TOKENS"),
        value_delimiter = ';'
    )]
    max_batch_tokens: Vec<usize>,

    /// Optionally control the maximum number of individual requests in a batch
    #[clap(long, env = infinity_v3_config::infinity_env!("MAX_BATCH_REQUESTS"), value_parser = optional_usize, value_delimiter = ';')]
    max_batch_requests: Vec<Optional<usize>>,

    /// Experimental calibration-free FP8 MLPs (Ada/Hopper/Blackwell FP16/BF16 dense decoder and BERT/ModernBERT encoders).
    /// Bundled in official SM89/90/100/120 images; disabled by default. May change embedding, reranking, and decision accuracy.
    #[clap(long, env = infinity_v3_config::infinity_env!("ENABLE_FP8_DYNAMIC"), action = clap::ArgAction::Append, num_args = 0..=1, default_missing_value = "true", default_value = "false", value_delimiter = ';')]
    enable_fp8_dynamic: Vec<bool>,

    /// RadixMLP threshold.
    ///
    /// Set the threshold for RadixMLP.
    /// If the compression ratio is lower than the threshold, RadixMLP will be used.
    /// The default is 0.92 for most models, and 0.0 (force disabled) for bidirectional models.
    #[clap(long, env = infinity_v3_config::infinity_env!("RADIX_MLP_THRESHOLD"), default_value = "0.92", value_parser = pct_parser, value_delimiter = ';')]
    radix_mlp_threshold: Vec<f32>,

    /// Control the maximum number of inputs that a client can send in a single request
    #[clap(
        default_value = "32",
        long,
        env = infinity_v3_config::infinity_env!("MAX_CLIENT_BATCH_SIZE"),
        value_delimiter = ';'
    )]
    max_client_batch_size: Vec<usize>,

    /// Maximum number of questions in one /v1/systemone request.
    /// Independent of max-client-batch-size; model option budgets still apply.
    #[clap(
        default_value = "512",
        long,
        env = infinity_v3_config::infinity_env!("MAX_DECISION_QUESTIONS"),
        value_delimiter = ';'
    )]
    max_decision_questions: Vec<std::num::NonZeroUsize>,

    /// Automatically truncate inputs that are longer than the maximum supported size
    ///
    #[clap(long, env = infinity_v3_config::infinity_env!("AUTO_TRUNCATE"), action = clap::ArgAction::Append, num_args = 0..=1, default_missing_value = "true", default_value = "false", value_delimiter = ';')]
    auto_truncate: Vec<bool>,

    /// The name of the prompt that should be used by default for encoding. If not set, no prompt
    /// will be applied.
    ///
    /// Must be a key in the `sentence-transformers` configuration `prompts` dictionary.
    ///
    /// For example if ``default_prompt_name`` is "query" and the ``prompts`` is {"query": "query: ", ...},
    /// then the sentence "What is the capital of France?" will be encoded as
    /// "query: What is the capital of France?" because the prompt text will be prepended before
    /// any text to encode.
    ///
    /// The argument '--default-prompt-name <DEFAULT_PROMPT_NAME>' cannot be used with
    /// '--default-prompt <DEFAULT_PROMPT>`
    #[clap(long, env = infinity_v3_config::infinity_env!("DEFAULT_PROMPT_NAME"), value_parser = optional_string, value_delimiter = ';')]
    default_prompt_name: Vec<Optional<String>>,

    /// The prompt that should be used by default for encoding. If not set, no prompt
    /// will be applied.
    ///
    /// For example if ``default_prompt`` is "query: " then the sentence "What is the capital of
    /// France?" will be encoded as "query: What is the capital of France?" because the prompt
    /// text will be prepended before any text to encode.
    ///
    /// The argument '--default-prompt <DEFAULT_PROMPT>' cannot be used with
    /// '--default-prompt-name <DEFAULT_PROMPT_NAME>`
    #[clap(long, env = infinity_v3_config::infinity_env!("DEFAULT_PROMPT"), value_parser = optional_string)]
    default_prompt: Vec<Optional<String>>,

    /// Optionally, define the path to the Dense module required for some embedding models.
    ///
    /// Some embedding models require an extra `Dense` module which contains a single Linear layer
    /// and an activation function. By default, those `Dense` modules are stored under the `2_Dense`
    /// directory, but there might be cases where different `Dense` modules are provided, to
    /// convert the pooled embeddings into different dimensions, available as `2_Dense_<dims>` e.g.
    /// https://huggingface.co/NovaSearch/stella_en_400M_v5.
    ///
    /// Note that this argument is optional, only required to be set if there is no `modules.json`
    /// file or when you want to override a single Dense module path, only when running with the
    /// `candle` backend.
    #[clap(long, env = infinity_v3_config::infinity_env!("DENSE_PATH"), value_parser = optional_string, value_delimiter = ';')]
    dense_path: Vec<Optional<String>>,

    /// Use one explicit accelerator device. CUDA otherwise uses all visible devices.
    #[clap(long, env = infinity_v3_config::infinity_env!("DEVICE_ID"), value_parser = optional_usize, value_delimiter = ';')]
    device_id: Vec<Optional<usize>>,

    /// Visible CUDA ordinals for data-parallel replicas, e.g. 0,1, or auto.
    #[clap(long, env = infinity_v3_config::infinity_env!("BACKEND_DEVICE_IDS"), value_parser = optional_string, value_delimiter = ';')]
    backend_device_ids: Vec<Optional<String>>,

    /// [DEPRECATED IN FAVOR OF `--hf-token`] Your Hugging Face Hub token
    #[clap(long, env = infinity_v3_config::infinity_env!("HF_API_TOKEN"), hide = true)]
    #[redact(partial)]
    hf_api_token: Option<String>,

    /// Your Hugging Face Hub token
    #[clap(long, env = infinity_v3_config::infinity_env!("HF_TOKEN"), conflicts_with = "hf_api_token")]
    #[redact(partial)]
    hf_token: Option<String>,

    /// The IP address to listen on
    #[clap(default_value = "0.0.0.0", long, env = infinity_v3_config::infinity_env!("HOSTNAME"))]
    hostname: String,

    /// The port to listen on.
    #[clap(default_value = "3000", long, short, env = infinity_v3_config::infinity_env!("PORT"))]
    port: u16,

    /// The location of the huggingface hub cache.
    /// Used to override the location if you want to provide a mounted disk for instance
    #[clap(long, env = infinity_v3_config::infinity_env!("HUGGINGFACE_HUB_CACHE"))]
    huggingface_hub_cache: Option<String>,

    /// Payload size limit in bytes
    ///
    /// Default is 2MB
    #[clap(default_value = "2000000", long, env = infinity_v3_config::infinity_env!("PAYLOAD_LIMIT"))]
    payload_limit: usize,

    /// Set an api key for request authorization.
    ///
    /// By default the server responds to every request. With an api key set, the requests must have the Authorization header set with the api key as Bearer token.
    #[clap(long, env = infinity_v3_config::infinity_env!("API_KEY"))]
    api_key: Option<String>,

    /// Outputs the logs in JSON format (useful for telemetry)
    #[clap(long, env = infinity_v3_config::infinity_env!("JSON_OUTPUT"))]
    json_output: bool,

    // Whether or not to include the log trace through spans
    #[clap(long, env = infinity_v3_config::infinity_env!("DISABLE_SPANS"))]
    disable_spans: bool,

    /// The grpc endpoint for opentelemetry. Telemetry is sent to this endpoint as OTLP over gRPC.
    /// e.g. `http://localhost:4317`
    #[clap(long, env = infinity_v3_config::infinity_env!("OTLP_ENDPOINT"))]
    otlp_endpoint: Option<String>,

    /// The service name for opentelemetry.
    /// e.g. `infinity_v3.server`
    #[clap(
        default_value = "infinity_v3.server",
        long,
        env = infinity_v3_config::infinity_env!("OTLP_SERVICE_NAME")
    )]
    otlp_service_name: String,

    /// The Prometheus port to listen on.
    #[clap(default_value = "9000", long, env = infinity_v3_config::infinity_env!("PROMETHEUS_PORT"))]
    prometheus_port: u16,

    #[clap(long, env = infinity_v3_config::infinity_env!("CORS_ALLOW_ORIGIN"))]
    cors_allow_origin: Option<Vec<String>>,
}

impl Args {
    fn models(&self) -> Result<Vec<infinity_v3_router::ModelSettings>> {
        let count = self.model_id.len();
        anyhow::ensure!(count > 0, "At least one --model-id is required");
        let served_model_name =
            expand_optional(&self.served_model_name, count, "served-model-name")?;
        let tokenization_queue_capacity = expand(
            &self.tokenization_queue_capacity,
            count,
            "tokenization-queue-capacity",
        )?;
        let backend = expand(&self.backend, count, "backend")?;
        let torch_device = expand(&self.torch_device, count, "torch-device")?;
        let image_memory_budget_mib = expand(
            &self.image_memory_budget_mib,
            count,
            "image-memory-budget-mib",
        )?;
        let max_concurrent_requests = expand(
            &self.max_concurrent_requests,
            count,
            "max-concurrent-requests",
        )?;
        let max_batch_tokens = expand(&self.max_batch_tokens, count, "max-batch-tokens")?;
        let radix_mlp_threshold = expand(&self.radix_mlp_threshold, count, "radix-mlp-threshold")?;
        let max_client_batch_size =
            expand(&self.max_client_batch_size, count, "max-client-batch-size")?;
        let max_decision_questions = expand(
            &self.max_decision_questions,
            count,
            "max-decision-questions",
        )?;
        let revision = expand_optional(&self.revision, count, "revision")?;
        let tokenization_workers =
            expand_optional(&self.tokenization_workers, count, "tokenization-workers")?;
        let dtype = expand(&self.dtype, count, "dtype")?;
        let pooling = expand_optional(&self.pooling, count, "pooling")?;
        let decision_protocol =
            expand_optional(&self.decision_protocol, count, "decision-protocol")?;
        let max_batch_requests =
            expand_optional(&self.max_batch_requests, count, "max-batch-requests")?;
        let default_prompt_name =
            expand_optional(&self.default_prompt_name, count, "default-prompt-name")?;
        let default_prompt = expand_optional(&self.default_prompt, count, "default-prompt")?;
        let dense_path = expand_optional(&self.dense_path, count, "dense-path")?;
        let device_id = expand_optional(&self.device_id, count, "device-id")?;
        let backend_device_ids =
            expand_optional(&self.backend_device_ids, count, "backend-device-ids")?;
        let enable_fp8_dynamic = expand(&self.enable_fp8_dynamic, count, "enable-fp8-dynamic")?;
        let auto_truncate = expand(&self.auto_truncate, count, "auto-truncate")?;
        (0..count)
            .map(|i| {
                Ok(infinity_v3_router::ModelSettings {
                    model_id: self.model_id[i].clone(),
                    served_model_name: served_model_name[i].clone(),
                    revision: revision[i].clone(),
                    tokenization_workers: tokenization_workers[i],
                    tokenization_queue_capacity: tokenization_queue_capacity[i],
                    dtype: Some(dtype[i].clone()),
                    pooling: pooling[i].clone(),
                    decision_protocol: decision_protocol[i].clone(),
                    max_concurrent_requests: max_concurrent_requests[i],
                    max_batch_tokens: max_batch_tokens[i],
                    max_batch_requests: max_batch_requests[i],
                    radix_mlp_threshold: radix_mlp_threshold[i],
                    enable_fp8_dynamic: enable_fp8_dynamic[i],
                    max_client_batch_size: max_client_batch_size[i],
                    max_decision_questions: max_decision_questions[i].get(),
                    auto_truncate: auto_truncate[i],
                    default_prompt: default_prompt[i].clone(),
                    default_prompt_name: default_prompt_name[i].clone(),
                    dense_path: dense_path[i].clone(),
                    device_id: device_id[i],
                    backend_device_ids: backend_device_ids[i].clone(),
                    backend_kind: backend[i],
                    torch_device: torch_device[i],
                    multimodal_config: infinity_v3_core::multimodal::MultimodalConfig {
                        allowed_image_hosts: self.image_allowed_hosts.clone(),
                        memory_budget_bytes: image_memory_budget_mib[i]
                            .checked_mul(1024 * 1024)
                            .ok_or_else(|| anyhow::anyhow!("Image memory budget is too large"))?,
                        ..Default::default()
                    },
                })
            })
            .collect()
    }
}

fn command_with_environment() -> clap::Command {
    let mut command = Args::command();
    let fallbacks = command
        .get_arguments()
        .filter_map(|arg| {
            let env = arg.get_env()?.to_str()?;
            let legacy = env.strip_prefix(infinity_v3_config::infinity_env!(""))?;
            if std::env::var_os(env).is_none() && std::env::var_os(legacy).is_some() {
                Some((arg.get_id().to_string(), legacy.to_owned()))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    for (id, env) in fallbacks {
        command = command.mut_arg(id, |arg| {
            arg.env(Box::leak(env.into_boxed_str()) as &'static str)
        });
    }
    command
}

fn args_from_matches(matches: &clap::ArgMatches) -> Result<Args> {
    let mut args = Args::from_arg_matches(matches)?;
    if matches.value_source("default_prompt") == Some(clap::parser::ValueSource::EnvVariable) {
        if let Some(Optional(Some(value))) = args.default_prompt.first() {
            if value.trim_start().starts_with('[') {
                args.default_prompt = serde_json::from_str::<Vec<Option<String>>>(value)
                    .context("Default prompt ENV list must be a JSON array of strings or null")?
                    .into_iter()
                    .map(Optional)
                    .collect();
            }
        }
    }
    Ok(args)
}

#[tokio::main]
async fn main() -> Result<()> {
    // Pattern match configuration
    let args = args_from_matches(&command_with_environment().get_matches())?;

    // Initialize logging and telemetry
    let global_tracer = infinity_v3_router::init_logging(
        args.otlp_endpoint.as_ref(),
        args.otlp_service_name.clone(),
        args.json_output,
        args.disable_spans,
    );

    tracing::info!("{args:?}");

    // Hack to trim pages regularly
    // see: https://www.algolia.com/blog/engineering/when-allocators-are-hoarding-your-precious-memory/
    // and: https://github.com/huggingface/text-embeddings-inference/issues/156
    #[cfg(target_os = "linux")]
    tokio::spawn(async move {
        use tokio::time::Duration;
        loop {
            tokio::time::sleep(Duration::from_millis(100)).await;
            unsafe {
                libc::malloc_trim(0);
            }
        }
    });

    // Since `--hf-api-token` is deprecated in favor of `--hf-token`, we need to still make sure
    // that if the user provides the token with `--hf-api-token` the token is still parsed properly
    if args.hf_api_token.is_some() {
        tracing::warn!("The `--hf-api-token` argument (and the `HF_API_TOKEN` env var) is deprecated and will be removed in a future version. Please use `--hf-token` (or the `HF_TOKEN` env var) instead.");
    }
    let configs = args.models()?;
    infinity_v3_router::run_models(
        configs,
        args.hf_token.or(args.hf_api_token),
        Some(args.hostname),
        args.port,
        args.huggingface_hub_cache,
        args.payload_limit,
        args.api_key,
        args.prometheus_port,
        args.cors_allow_origin,
    )
    .await?;

    if global_tracer {
        // Shutdown tracer
        global::shutdown_tracer_provider();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn model_options_broadcast_align_and_preserve_single_model_flags() {
        let args = Args::try_parse_from([
            "infinity_v3",
            "--model-id",
            "a",
            "--model-id",
            "b",
            "--served-model-name",
            "embed;rerank",
            "--backend",
            "libtorch",
            "--torch-device",
            "cpu",
            "--max-batch-tokens",
            "100;200",
            "--revision",
            "none;main",
            "--pooling",
            "mean;none",
            "--decision-protocol",
            "none;pplx",
            "--auto-truncate",
            "true;false",
        ])
        .unwrap();
        let models = args.models().unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].served_model_name.as_deref(), Some("embed"));
        assert_eq!(models[1].served_model_name.as_deref(), Some("rerank"));
        assert_eq!(models[0].max_batch_tokens, 100);
        assert_eq!(models[1].max_batch_tokens, 200);
        assert_eq!(models[0].revision, None);
        assert_eq!(models[1].revision.as_deref(), Some("main"));
        assert!(models[0].auto_truncate && !models[1].auto_truncate);
        assert_eq!(models[0].backend_kind, models[1].backend_kind);
        let legacy = Args::try_parse_from([
            "infinity_v3",
            "--model-id",
            "a",
            "--auto-truncate",
            "--enable-fp8-dynamic",
        ])
        .unwrap()
        .models()
        .unwrap();
        assert!(legacy[0].auto_truncate && legacy[0].enable_fp8_dynamic);
        for ids in ["a", "a;b"] {
            let matches = Args::command()
                .try_get_matches_from([
                    "infinity_v3",
                    "--model-id",
                    ids,
                    "--default-prompt",
                    "Represent this; retrieve it",
                ])
                .unwrap();
            let models = args_from_matches(&matches).unwrap().models().unwrap();
            assert!(models.iter().all(
                |model| model.default_prompt.as_deref() == Some("Represent this; retrieve it")
            ));
        }
        let matches = Args::command()
            .try_get_matches_from([
                "infinity_v3",
                "--model-id",
                "a;b",
                "--default-prompt",
                "first; prompt",
                "--default-prompt",
                "second; prompt",
            ])
            .unwrap();
        let models = args_from_matches(&matches).unwrap().models().unwrap();
        assert_eq!(models[0].default_prompt.as_deref(), Some("first; prompt"));
        assert_eq!(models[1].default_prompt.as_deref(), Some("second; prompt"));
        let mismatch =
            Args::try_parse_from(["infinity_v3", "--model-id", "a;b;c", "--device-id", "0;1"])
                .unwrap();
        assert!(mismatch.models().is_err());
        // The build default applies to every model; per-model overrides remain available.
        let backend_env = infinity_v3_config::infinity_env!("BACKEND");
        let previous_backend = std::env::var_os(backend_env);
        std::env::remove_var(backend_env);
        let models = Args::try_parse_from(["infinity_v3", "--model-id", "a;b"])
            .unwrap()
            .models()
            .unwrap();
        assert!(models
            .iter()
            .all(|m| m.backend_kind == infinity_v3_backend::BackendKind::default()));
        let models = Args::try_parse_from([
            "infinity_v3",
            "--model-id",
            "a;b",
            "--backend",
            "libtorch;candle",
        ])
        .unwrap()
        .models()
        .unwrap();
        assert_eq!(
            models[0].backend_kind,
            infinity_v3_backend::BackendKind::Libtorch
        );
        assert_eq!(
            models[1].backend_kind,
            infinity_v3_backend::BackendKind::Candle
        );
        std::env::set_var(backend_env, "libtorch");
        let models = Args::try_parse_from(["infinity_v3", "--model-id", "a;b"])
            .unwrap()
            .models()
            .unwrap();
        assert!(models
            .iter()
            .all(|m| m.backend_kind == infinity_v3_backend::BackendKind::Libtorch));
        match previous_backend {
            Some(value) => std::env::set_var(backend_env, value),
            None => std::env::remove_var(backend_env),
        }
        // Verify that help and environment parsing use the same compiled names.
        let help = Args::command().render_long_help().to_string();
        assert!(help.contains(infinity_v3_config::infinity_env!("MODEL_ID")));
        let model_env = infinity_v3_config::infinity_env!("MODEL_ID");
        let previous = std::env::var_os(model_env);
        let previous_legacy = std::env::var_os("MODEL_ID");
        std::env::set_var("MODEL_ID", "legacy-model");
        std::env::remove_var(model_env);
        let matches = command_with_environment()
            .try_get_matches_from(["infinity_v3"])
            .unwrap();
        assert_eq!(
            args_from_matches(&matches).unwrap().model_id,
            vec!["legacy-model"]
        );
        std::env::set_var(model_env, "prefixed-model");
        let matches = command_with_environment()
            .try_get_matches_from(["infinity_v3"])
            .unwrap();
        assert_eq!(
            args_from_matches(&matches).unwrap().model_id,
            vec!["prefixed-model"]
        );
        let matches = command_with_environment()
            .try_get_matches_from(["infinity_v3", "--model-id", "cli-model"])
            .unwrap();
        assert_eq!(
            args_from_matches(&matches).unwrap().model_id,
            vec!["cli-model"]
        );
        for (key, value) in [(model_env, previous), ("MODEL_ID", previous_legacy)] {
            match value {
                Some(value) => std::env::set_var(key, value),
                None => std::env::remove_var(key),
            }
        }
        // This binary has one test, so environment changes do not race other tests.
        let original_prompt = std::env::var_os(infinity_v3_config::infinity_env!("DEFAULT_PROMPT"));
        std::env::set_var(
            infinity_v3_config::infinity_env!("DEFAULT_PROMPT"),
            r#"["first; prompt",null]"#,
        );
        let matches = Args::command()
            .try_get_matches_from(["infinity_v3", "--model-id", "a;b"])
            .unwrap();
        let models = args_from_matches(&matches).unwrap().models().unwrap();
        assert_eq!(models[0].default_prompt.as_deref(), Some("first; prompt"));
        assert_eq!(models[1].default_prompt, None);
        let matches = Args::command()
            .try_get_matches_from([
                "infinity_v3",
                "--model-id",
                "a;b",
                "--default-prompt",
                "[literal; prompt]",
            ])
            .unwrap();
        let models = args_from_matches(&matches).unwrap().models().unwrap();
        assert!(models
            .iter()
            .all(|model| model.default_prompt.as_deref() == Some("[literal; prompt]")));
        match original_prompt {
            Some(value) => {
                std::env::set_var(infinity_v3_config::infinity_env!("DEFAULT_PROMPT"), value)
            }
            None => std::env::remove_var(infinity_v3_config::infinity_env!("DEFAULT_PROMPT")),
        }
    }
}
