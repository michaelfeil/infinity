/// Text Embedding Inference Webserver
mod logging;
mod prometheus;
mod replicas;

#[cfg(feature = "http")]
mod http;

#[cfg(feature = "http")]
use ::http::{HeaderMap, HeaderName, HeaderValue};

mod shutdown;

use anyhow::{anyhow, Context, Result};
use bigdecimal::{BigDecimal, RoundingMode};
use hf_hub::api::tokio::ApiBuilder;
use hf_hub::{Repo, RepoType};
use infinity_v3_backend::{DType, Pool};
use infinity_v3_core::download::{download_artifacts, ST_CONFIG_NAMES};
use infinity_v3_core::infer::Infer;
use infinity_v3_core::queue_actor::{Queue, ServiceTier as QueueTier};
use infinity_v3_core::tokenization::{Tokenization, MAX_TOKENIZATION_WORKERS};
use infinity_v3_core::TextEmbeddingsError;
use serde::Deserialize;
use serde::Serialize;
use std::collections::HashMap;
use std::fs;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::Path;
use std::sync::OnceLock;
use std::time::{Duration, Instant};
use tokenizers::processors::sequence::Sequence;
use tokenizers::processors::template::TemplateProcessing;
use tokenizers::{PostProcessorWrapper, Tokenizer};
use tracing::{Instrument, Span};

pub use logging::init_logging;

mod model_aliases;
pub mod model_registry;
mod vela3;

#[derive(Debug, Clone, PartialEq, Eq, clap::ValueEnum)]
pub enum DecisionProtocol {
    Rune,
    Onejev,
    Clef,
    Pplx,
    D3,
    Vela3,
}

/// Settings for one resident model. Server settings are shared across models.
#[derive(Clone, Debug)]
pub struct ModelSettings {
    pub served_model_name: Option<String>,
    pub model_id: String,
    pub revision: Option<String>,
    pub tokenization_workers: Option<usize>,
    pub tokenization_queue_capacity: std::num::NonZeroUsize,
    pub dtype: Option<DType>,
    pub pooling: Option<infinity_v3_backend::Pool>,
    pub decision_protocol: Option<DecisionProtocol>,
    pub priority_queue_bonus_ms: u64,
    pub flex_queue_offset_ms: u64,
    pub max_concurrent_requests: usize,
    pub max_batch_tokens: usize,
    pub max_batch_expanded_tokens: Option<usize>,
    pub max_sequence_length: Option<std::num::NonZeroUsize>,
    pub max_batch_requests: Option<usize>,
    pub radix_mlp_threshold: f32,
    pub enable_fp8_dynamic: bool,
    pub max_client_batch_size: usize,
    pub max_decision_questions: usize,
    pub auto_truncate: bool,
    pub default_prompt: Option<String>,
    pub default_prompt_name: Option<String>,
    pub dense_path: Option<String>,
    pub device_id: Option<usize>,
    pub backend_device_ids: Option<String>,
    pub backend_kind: infinity_v3_backend::BackendKind,
    pub torch_device: infinity_v3_backend::TorchDevice,
    pub multimodal_config: infinity_v3_core::multimodal::MultimodalConfig,
}

/// One loaded model, including its tokenizer, queue and replica pool.
#[derive(Clone)]
pub struct ModelRuntime {
    pub infer: Infer,
    pub info: Info,
    pub checkpoint_id: String,
    pub(crate) named: bool,
    pub backend_kind: infinity_v3_backend::BackendKind,
    pub(crate) systemone: Option<std::sync::Arc<http::systemone::SystemOne>>,
}

impl ModelRuntime {
    pub async fn shutdown(&self) {
        self.infer.shutdown().await;
    }
}

/// Create entrypoint
#[allow(clippy::too_many_arguments)]
pub async fn run(
    model_id: String,
    revision: Option<String>,
    tokenization_workers: Option<usize>,
    tokenization_queue_capacity: std::num::NonZeroUsize,
    dtype: Option<DType>,
    pooling: Option<infinity_v3_backend::Pool>,
    decision_protocol: Option<DecisionProtocol>,
    max_concurrent_requests: usize,
    max_batch_tokens: usize,
    max_batch_requests: Option<usize>,
    radix_mlp_threshold: f32,
    enable_fp8_dynamic: bool,
    max_client_batch_size: usize,
    max_decision_questions: usize,
    auto_truncate: bool,
    default_prompt: Option<String>,
    default_prompt_name: Option<String>,
    dense_path: Option<String>,
    hf_token: Option<String>,
    hostname: Option<String>,
    port: u16,
    huggingface_hub_cache: Option<String>,
    payload_limit: usize,
    api_key: Option<String>,
    prometheus_port: u16,
    cors_allow_origin: Option<Vec<String>>,
    device_id: Option<usize>,
    backend_device_ids: Option<String>,
    backend_kind: infinity_v3_backend::BackendKind,
    torch_device: infinity_v3_backend::TorchDevice,
    multimodal_config: infinity_v3_core::multimodal::MultimodalConfig,
) -> Result<()> {
    run_models(
        vec![ModelSettings {
            served_model_name: None,
            model_id,
            revision,
            tokenization_workers,
            tokenization_queue_capacity,
            dtype,
            pooling,
            decision_protocol,
            priority_queue_bonus_ms: 1000,
            flex_queue_offset_ms: 10000,
            max_concurrent_requests,
            max_batch_tokens,
            max_batch_expanded_tokens: None,
            max_sequence_length: None,
            max_batch_requests,
            radix_mlp_threshold,
            enable_fp8_dynamic,
            max_client_batch_size,
            max_decision_questions,
            auto_truncate,
            default_prompt,
            default_prompt_name,
            dense_path,
            device_id,
            backend_device_ids,
            backend_kind,
            torch_device,
            multimodal_config,
        }],
        hf_token,
        hostname,
        port,
        huggingface_hub_cache,
        payload_limit,
        api_key,
        prometheus_port,
        cors_allow_origin,
    )
    .await
}

async fn finish_startup_validation(
    validation: &mut Option<tokio::task::JoinHandle<Result<()>>>,
) -> Result<()> {
    if let Some(task) = validation.take() {
        task.await.context("Startup validation task failed")??;
    }
    Ok(())
}

async fn initialize_model(
    config: ModelSettings,
    hf_token: Option<String>,
    huggingface_hub_cache: Option<String>,
    validation: &mut Option<tokio::task::JoinHandle<Result<()>>>,
) -> Result<ModelRuntime> {
    let frontend_started = std::time::Instant::now();
    let ModelSettings {
        priority_queue_bonus_ms,
        flex_queue_offset_ms,
        served_model_name,
        model_id,
        revision,
        tokenization_workers,
        tokenization_queue_capacity,
        dtype,
        pooling,
        decision_protocol,
        max_concurrent_requests,
        max_batch_tokens,
        max_batch_expanded_tokens,
        max_sequence_length,
        max_batch_requests,
        radix_mlp_threshold,
        enable_fp8_dynamic,
        max_client_batch_size,
        max_decision_questions,
        auto_truncate,
        default_prompt,
        default_prompt_name,
        dense_path,
        device_id,
        backend_device_ids,
        backend_kind,
        torch_device,
        multimodal_config,
    } = config;
    let (model_id, mut decision_protocol) =
        model_aliases::resolve(model_id, decision_protocol, pooling.as_ref())?;
    let metric_model_name = served_model_name
        .clone()
        .unwrap_or_else(|| model_id.clone());
    let model_id_path = Path::new(&model_id);
    let (mut model_root, mut api_repo) = if model_id_path.exists() && model_id_path.is_dir() {
        // Using a local model
        (model_id_path.to_path_buf(), None)
    } else {
        let mut builder = ApiBuilder::from_env()
            .with_progress(false)
            .with_token(hf_token.clone());

        if let Some(cache_dir) = huggingface_hub_cache.clone() {
            builder = builder.with_cache_dir(cache_dir.into());
        }

        if let Ok(origin) = std::env::var("HF_HUB_USER_AGENT_ORIGIN") {
            builder = builder.with_user_agent("origin", origin.as_str());
        }

        let api = builder.build().unwrap();
        let api_repo = api.repo(Repo::with_revision(
            model_id.clone(),
            RepoType::Model,
            revision.clone().unwrap_or("main".to_string()),
        ));

        // Download model from the Hub
        let model_root = if matches!(decision_protocol, Some(DecisionProtocol::Vela3)) {
            vela3::download_artifacts(&api_repo, pooling.is_none()).await
        } else {
            download_artifacts(&api_repo, pooling.is_none())
                .await
                .map_err(Into::into)
        }
        .context("Could not download model artifacts")?;
        (model_root, Some(api_repo))
    };

    let vela_policy = if matches!(decision_protocol, Some(DecisionProtocol::Vela3)) {
        let policy = vela3::Policy::load(&model_root)?;
        if let Some(base) = policy.base() {
            let mut builder = ApiBuilder::from_env()
                .with_progress(false)
                .with_token(hf_token);
            if let Some(cache) = huggingface_hub_cache {
                builder = builder.with_cache_dir(cache.into());
            }
            let repo = builder.build()?.repo(Repo::with_revision(
                base.repo_id.clone(),
                RepoType::Model,
                base.revision.clone(),
            ));
            model_root = download_artifacts(&repo, false)
                .await
                .context("Could not download the Vela3 checkpoint's pinned D3 base")?;
            anyhow::ensure!(
                !model_root.join("vela3_config.json").exists(),
                "Nested Vela3 bases are unsupported"
            );
            api_repo = Some(repo);
            decision_protocol = Some(DecisionProtocol::D3);
        }
        tracing::warn!("Vela3 core decisions enabled; task adapters, router signals and span extraction are not enabled");
        Some(policy)
    } else {
        None
    };

    // Load config
    let laya = model_root.join("rl_agent_config.json").exists();
    let clef = matches!(decision_protocol, Some(DecisionProtocol::Clef));
    let rune = matches!(decision_protocol, Some(DecisionProtocol::Rune));
    let onejev = matches!(decision_protocol, Some(DecisionProtocol::Onejev));
    let pplx = matches!(decision_protocol, Some(DecisionProtocol::Pplx));
    let d3 = matches!(decision_protocol, Some(DecisionProtocol::D3));
    anyhow::ensure!(
        !d3 || backend_kind == infinity_v3_backend::BackendKind::Libtorch,
        "D3 requires --backend libtorch"
    );
    let vela_encoder = matches!(decision_protocol, Some(DecisionProtocol::Vela3));
    anyhow::ensure!(
        !vela_encoder || backend_kind == infinity_v3_backend::BackendKind::Libtorch,
        "Vela3 requires --backend libtorch"
    );
    let decoder_decision = rune || onejev || clef || pplx || d3 || vela_encoder;
    let readout = model_root.join("decision_config.json").exists();
    anyhow::ensure!(
        !(pplx || d3) || (readout && model_root.join("readout.safetensors").exists()),
        "Pplx and D3 require decision_config.json and readout.safetensors"
    );
    anyhow::ensure!(
        !readout || pplx || d3,
        "This checkpoint requires --decision-protocol pplx or d3"
    );
    if readout {
        let metadata: serde_json::Value =
            serde_json::from_slice(&fs::read(model_root.join("decision_config.json"))?)?;
        anyhow::ensure!(
            metadata["format_version"] != 2
                || backend_kind == infinity_v3_backend::BackendKind::Libtorch,
            "Perplexity joint readouts require --backend libtorch"
        );
        anyhow::ensure!(
            (metadata["format_id"].as_str() == Some("d3-code-readout-v1")) == d3,
            "D3 checkpoints require --decision-protocol d3; Perplexity checkpoints require pplx"
        );
    }
    let joint_head = model_root.join("joint_head_config.json").exists();
    anyhow::ensure!(
        !clef || (joint_head && model_root.join("joint_head.safetensors").exists()),
        "Clef requires joint_head_config.json and joint_head.safetensors"
    );
    anyhow::ensure!(
        !onejev || !joint_head,
        "A Clef checkpoint requires --decision-protocol clef"
    );
    anyhow::ensure!(
        !(laya && decoder_decision),
        "Decoder decision protocols cannot serve a Laya checkpoint"
    );
    let config_path = model_root.join(if laya {
        "encoder/config.json"
    } else {
        "config.json"
    });
    let config = fs::read_to_string(config_path).context("`config.json` not found")?;
    let mut config: ModelConfig =
        serde_json::from_str(&config).context("Failed to parse `config.json`")?;
    let multi_vector = config.architectures.iter().any(|arch| arch == "HF_ColBERT")
        || matches!(
            config.model_type.as_str(),
            "colqwen3" | "qwen3_vl_nemotron_embed"
        );
    if matches!(
        config.model_type.as_str(),
        "colqwen3" | "qwen3_vl_nemotron_embed"
    ) {
        config.model_type = "qwen3_vl".into();
    }
    config.resolve_text_config()?;
    config.resolve_decoder_classifier_labels()?;
    anyhow::ensure!(
        config.auto_map.as_ref().and_then(|map| map.get("AutoModel")).map(String::as_str)
            != Some("modeling.PPLXQwen3ContextualModel"),
        "Perplexity contextual embeddings are unsupported; use pplx-embed-v1-0.6b or pplx-embed-v1-4b"
    );
    anyhow::ensure!(
        config.max_position_embeddings > 0,
        "`max_position_embeddings` must be positive"
    );

    anyhow::ensure!(
        !rune || config.model_type == "gemma4",
        "Rune protocol requires a Gemma4 model"
    );
    anyhow::ensure!(
        !onejev
            || matches!(
                config.model_type.as_str(),
                "qwen3_5" | "qwen3_5_moe" | "qwen3_5_text" | "qwen3_5_moe_text"
            ),
        "OneJev protocol requires a Qwen3.5 model"
    );

    anyhow::ensure!(
        !(clef || pplx || d3) || matches!(config.model_type.as_str(), "qwen3_5" | "qwen3_5_text"),
        "Clef, Pplx and D3 require a dense Qwen3.5 model"
    );
    anyhow::ensure!(
        config.model_type != "qwen3_vl"
            || (default_prompt.is_none() && default_prompt_name.is_none()),
        "This multimodal model uses its native chat template; default prompt overrides are unsupported"
    );

    // Set model type from config
    let backend_model_type = if laya || decoder_decision {
        anyhow::ensure!(
            pooling.is_none(),
            "Typed decisions do not use embedding pooling"
        );
        infinity_v3_backend::ModelType::Decision
    } else if multi_vector {
        anyhow::ensure!(
            pooling.is_none(),
            "Multi-vector models do not support pooling"
        );
        infinity_v3_backend::ModelType::Embedding(infinity_v3_backend::Pool::Mean)
    } else if config.model_type == "qwen3_vl" {
        anyhow::ensure!(
            pooling.is_none() || pooling == Some(infinity_v3_backend::Pool::LastToken),
            "Qwen3-VL requires last-token pooling"
        );
        infinity_v3_backend::ModelType::Embedding(infinity_v3_backend::Pool::LastToken)
    } else {
        get_backend_model_type(&config, &model_root, pooling)?
    };

    // Info model type
    let model_type = match &backend_model_type {
        infinity_v3_backend::ModelType::Decision => ModelType::Decision,
        infinity_v3_backend::ModelType::Classifier => {
            let id2label = config
                .id2label
                .context("`config.json` does not contain `id2label`")?;
            let n_classes = id2label.len();
            let classifier_model = ClassifierModel {
                id2label,
                label2id: config
                    .label2id
                    .context("`config.json` does not contain `label2id`")?,
            };
            if n_classes > 1 {
                ModelType::Classifier(classifier_model)
            } else {
                ModelType::Reranker(classifier_model)
            }
        }
        infinity_v3_backend::ModelType::Embedding(pool) => ModelType::Embedding(EmbeddingModel {
            pooling: if multi_vector {
                anyhow::ensure!(
                    backend_kind == infinity_v3_backend::BackendKind::Libtorch,
                    "ColBERT requires the LibTorch backend"
                );
                "colbert".into()
            } else {
                pool.to_string()
            },
        }),
    };

    // Load tokenizer
    let tokenizer_path = model_root.join(if laya {
        "tokenizer/tokenizer.json"
    } else {
        "tokenizer.json"
    });
    let mut tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|error| anyhow!("Could not load fast tokenizer: {error}"))?;
    tokenizer.with_padding(None);
    // Old Qwen2  repos updates the post processor manually instead of into the tokenizer.json.
    // Newer ones (https://huggingface.co/jinaai/jina-code-embeddings-0.5b/tree/main) have it in the tokenizer.json. This is to support both cases.
    // https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct/blob/main/tokenization_qwen.py#L246
    if config.model_type == "qwen2"
        && config
            .auto_map
            .as_ref()
            .is_some_and(|m| m.get("AutoModel") == Some(&"modeling_qwen.Qwen2Model".to_string()))
    {
        tracing::warn!("Model is detected as a Qwen2 model with remote code. Adding a post processor manually as the tokenizer.json does not contain a post processor.");
        let template = TemplateProcessing::builder()
            .try_single("$A:0 <|endoftext|>:0")
            .unwrap()
            .try_pair("$A:0 <|endoftext|>:0 $B:1 <|endoftext|>:1")
            .unwrap()
            .special_tokens(vec![("<|endoftext|>", 151643)])
            .build()
            .unwrap();
        match tokenizer.get_post_processor() {
            None => tokenizer.with_post_processor(Some(template)),
            Some(post_processor) => {
                let post_processor = Sequence::new(vec![
                    post_processor.clone(),
                    PostProcessorWrapper::Template(template),
                ]);
                tokenizer.with_post_processor(Some(post_processor))
            }
        };
    }

    // Position IDs offset. Used for Roberta and camembert.
    let position_offset = if &config.model_type == "xlm-roberta"
        || &config.model_type == "camembert"
        || &config.model_type == "roberta"
    {
        config.pad_token_id.unwrap_or(0) + 1
    } else {
        0
    };

    // Try to load ST Config
    let mut st_config: Option<STConfig> = None;
    for name in ST_CONFIG_NAMES {
        // New multimodal ST files describe modalities rather than max_seq_length.
        if matches!(
            config.model_type.as_str(),
            "qwen3_vl" | "embedding_gemma2" | "ultrabert"
        ) {
            break;
        }
        let config_path = model_root.join(name);
        if let Ok(config) = fs::read_to_string(config_path) {
            st_config =
                Some(serde_json::from_str(&config).context(format!("Failed to parse `{}`", name))?);
            break;
        }
    }

    let base_input_length = if laya {
        let cfg: serde_json::Value =
            serde_json::from_slice(&fs::read(model_root.join("rl_agent_config.json"))?)?;
        let length = cfg["max_len"]
            .as_u64()
            .context("Laya max_len must be positive")? as usize;
        anyhow::ensure!(
            length > 0 && length <= config.max_position_embeddings,
            "Invalid Laya max_len"
        );
        length
    } else if config.model_type == "embedding_gemma2" {
        // The checkpoint advertises a larger RoPE range but was trained with
        // an 8192-token shared text/media context.
        config.max_position_embeddings.min(8192)
    } else if decoder_decision {
        // Decision prompts cannot be truncated: reject over-budget requests in
        // the formatter, with the queue capacity as the configured input limit.
        config
            .max_position_embeddings
            .min(max_batch_tokens)
            .min(if vela_policy.is_some() {
                32768
            } else {
                usize::MAX
            })
    } else {
        match st_config {
            Some(config) => config.max_seq_length,
            None => {
                tracing::warn!("Could not find a Sentence Transformers config");
                config.max_position_embeddings - position_offset
            }
        }
    };

    let base_input_length = max_sequence_length.map_or(base_input_length, |limit| {
        base_input_length.min(limit.get())
    });

    // Raise an error when max_input_length is bigger than max_batch tokens to prevent an infinite loop in the queue
    let max_input_length = if base_input_length > max_batch_tokens {
        if !auto_truncate {
            anyhow::bail!(
                "`--max-batch-tokens` cannot be lower than the model `max_input_length` ({} < {}) when `--auto-truncate` is disabled, add the `--auto-truncate` flag to truncate the input sequences to match the `--max-batch-tokens`.",
                base_input_length,
                max_batch_tokens
            );
        }
        tracing::warn!(
            "The input sequences will be truncated to {} tokens even if the model `max_input_length` is greater than the provided `--max-batch-tokens` ({} > {}), as `--auto-truncate` is enabled.",
            max_batch_tokens,
            base_input_length,
            max_batch_tokens
        );
        max_batch_tokens
    } else {
        base_input_length
    };

    tracing::info!("Maximum number of tokens per request: {max_input_length}");

    // Reserve one CPU for the backend and cap both defaults and explicit settings.
    let tokenization_workers = tokenization_workers
        .unwrap_or_else(|| num_cpus::get().saturating_sub(1))
        .clamp(1, MAX_TOKENIZATION_WORKERS);

    // Try to load new ST Config
    let mut new_st_config: Option<NewSTConfig> = None;
    let config_path = model_root.join("config_sentence_transformers.json");
    if let Ok(config) = fs::read_to_string(config_path) {
        new_st_config = Some(
            serde_json::from_str(&config)
                .context("Failed to parse `config_sentence_transformers.json`")?,
        );
    }
    let prompts = new_st_config.and_then(|c| c.prompts);
    let default_prompt = if let Some(default_prompt_name) = default_prompt_name.as_ref() {
        match &prompts {
            None => {
                anyhow::bail!(format!("`default-prompt-name` is set to `{default_prompt_name}` but no prompts were found in the Sentence Transformers configuration"));
            }
            Some(prompts) if !prompts.contains_key(default_prompt_name) => {
                anyhow::bail!(format!("`default-prompt-name` is set to `{default_prompt_name}` but it was not found in the Sentence Transformers prompts. Available prompts: {:?}", prompts.keys()));
            }
            Some(prompts) => prompts.get(default_prompt_name).cloned(),
        }
    } else {
        default_prompt
    };

    #[cfg(feature = "http")]
    let systemone = if laya || decoder_decision {
        Some(std::sync::Arc::new(
            http::systemone::SystemOne::load(
                &model_root,
                tokenizer.clone(),
                max_input_length,
                multimodal_config.clone(),
                decision_protocol,
            )?
            .with_vela_policy(vela_policy),
        ))
    } else {
        None
    };

    // Tokenization logic
    let chat = if matches!(
        &backend_model_type,
        infinity_v3_backend::ModelType::Embedding(_) | infinity_v3_backend::ModelType::Classifier
    ) {
        infinity_v3_core::chat::ChatProcessor::load(&model_root).unwrap_or_else(|error| {
            tracing::warn!("Native conversations disabled: {error}");
            None
        })
    } else {
        None
    };
    let multimodal = if config.model_type == "qwen3_vl" {
        Some(
            infinity_v3_core::multimodal::Qwen3VlProcessor::load(
                &model_root,
                tokenizer.clone(),
                max_input_length,
                multimodal_config.clone(),
            )
            .map_err(|error| anyhow!(error.to_string()))?,
        )
    } else {
        None
    };
    let mistral3 = if config.model_type == "mistral3" {
        anyhow::ensure!(
            backend_kind == infinity_v3_backend::BackendKind::Libtorch,
            "Mistral3 multimodal support requires the LibTorch backend"
        );
        Some(
            infinity_v3_core::multimodal::Mistral3Processor::load(
                &model_root,
                tokenizer.clone(),
                max_input_length,
                multimodal_config.clone(),
                prompts.clone(),
                default_prompt.clone(),
            )
            .map_err(|e| anyhow!(e.to_string()))?,
        )
    } else {
        None
    };
    let embedding_gemma2 = if config.model_type == "embedding_gemma2" {
        Some(
            infinity_v3_core::multimodal::EmbeddingGemma2Processor::load(
                &model_root,
                tokenizer.clone(),
                max_input_length,
                multimodal_config.clone(),
                prompts.clone(),
                default_prompt.clone(),
            )
            .map_err(|error| anyhow!(error.to_string()))?,
        )
    } else {
        None
    };
    let tokenization = Tokenization::with_queue_capacity(
        tokenization_workers,
        tokenization_queue_capacity,
        tokenizer,
        max_input_length,
        position_offset,
        default_prompt,
        prompts,
        chat,
    )
    .with_token_input_vocab_size(
        config
            .text_config
            .as_ref()
            .and_then(|text| text.vocab_size)
            .or(config.vocab_size),
    );

    let tokenization = match multimodal {
        Some(processor) => tokenization.with_multimodal(processor),
        None => tokenization,
    };

    let tokenization = match mistral3 {
        Some(p) => tokenization.with_mistral3(p),
        None => tokenization,
    };
    let tokenization = match embedding_gemma2 {
        Some(processor) => tokenization.with_embedding_gemma2(processor),
        None => tokenization,
    };

    tracing::info!(
        seconds = frontend_started.elapsed().as_secs_f64(),
        "Model frontend initialized"
    );
    // All models must pass validation before any checkpoint weights are loaded.
    finish_startup_validation(validation).await?;
    let torch_device = backend_kind.resolve_device(torch_device)?;
    let model_dtype = if vela_encoder {
        Some("bfloat16")
    } else {
        config.dtype.as_deref().or(config.torch_dtype.as_deref())
    };
    let dtype = match backend_kind {
        infinity_v3_backend::BackendKind::Libtorch => resolve_libtorch_dtype(
            dtype,
            model_dtype,
            &config.model_type,
            torch_device == infinity_v3_backend::TorchDevice::Cpu,
        ),
    };

    anyhow::ensure!(
        !d3 || dtype == DType::Bfloat16,
        "D3 requires --dtype bfloat16 for its backbone (the readout remains FP32)"
    );

    // Create backend
    let devices = replicas::resolve_devices(
        backend_device_ids.as_deref(),
        device_id,
        backend_kind,
        torch_device,
    )?;
    tracing::info!("Starting model backend");
    let api_repo = api_repo.map(std::sync::Arc::new);
    let initializers = devices.iter().enumerate().map(|(replica, &device)| {
        let model_root = model_root.clone();
        let api_repo = api_repo.clone();
        let dtype = dtype.clone();
        let backend_model_type = backend_model_type.clone();
        let dense_path = dense_path.clone();
        let startup_span = tracing::info_span!(
            "backend_startup", model = %model_root.display(), replica, device,
            pid = std::process::id()
        );
        let initialize = async move {
            let started = std::time::Instant::now();
            tracing::info!(device, replica, "Initializing backend replica");
            let backend = infinity_v3_backend::Backend::new_selected(
                model_root,
                api_repo,
                dtype,
                backend_model_type,
                dense_path,
                device,
                enable_fp8_dynamic,
                backend_kind,
                torch_device,
            )
            .await
            .inspect_err(|error| {
                tracing::error!(seconds = started.elapsed().as_secs_f64(), %error,
                    "Backend initialization failed");
            })
            .with_context(|| format!("Could not create backend on device {device}"))?;
            backend
                .health()
                .await
                .with_context(|| format!("Model backend on device {device} is not healthy"))?;
            let replica_batch_limit = match (max_batch_requests, backend.max_batch_size) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            };
            backend
                .warmup(max_input_length, max_batch_tokens, replica_batch_limit)
                .await
                .with_context(|| format!("Warmup failed on device {device}"))?;
            tracing::info!(device, replica, elapsed = ?started.elapsed(), "Backend replica ready");
            Ok(backend)
        };
        initialize.instrument(startup_span)
    });
    let backends = replicas::initialize_all(initializers).await?;
    let backend = infinity_v3_core::backend_pool::BackendPool::new_named(
        backends,
        metric_model_name.clone(),
    )?;
    let max_batch_requests = match (max_batch_requests, backend.max_batch_size) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    };
    tracing::info!(replicas = devices.len(), "Backend pool ready");

    // Queue logic
    let radix_mlp_threshold = if config.model_type == "bert"
        || config.model_type == "xlm-roberta"
        || config.model_type == "camembert"
        || config.model_type == "roberta"
        || config.model_type == "distilbert"
        || config.model_type == "modernbert"
        || config.use_bidirectional_attention.unwrap_or(false)
        || (matches!(
            config.model_type.as_str(),
            "gemma4" | "gemma4_unified" | "embedding_gemma2"
        ) && matches!(
            &backend.model_type,
            infinity_v3_backend::ModelType::Embedding(_)
        ))
        || !backend.radix_mlp_supported
    {
        if radix_mlp_threshold > 0.0 {
            tracing::warn!("`--radix-mlp-threshold` requires a causal model with RadixMLP support. Disabling RadixMLP.");
        }
        0.0
    } else {
        radix_mlp_threshold
    };
    if radix_mlp_threshold > 0.0 {
        tracing::info!(
            "RadixMLP enabled with compression ratio threshold: {}",
            radix_mlp_threshold
        );
    } else {
        tracing::info!("RadixMLP disabled");
    }

    let queue = Queue::with_service_tiers_named(
        max_batch_tokens,
        max_batch_requests,
        radix_mlp_threshold,
        max_concurrent_requests,
        devices.len(),
        metric_model_name,
        std::time::Duration::from_millis(priority_queue_bonus_ms),
        std::time::Duration::from_millis(flex_queue_offset_ms),
        max_batch_expanded_tokens,
    );

    // Create infer task
    let infer = Infer::with_pool(tokenization, queue, max_concurrent_requests, backend);

    // Endpoint info
    let info = Info {
        model_id,
        model_sha: revision,
        model_dtype: dtype.to_string(),
        model_type,
        max_concurrent_requests,
        max_input_length,
        max_batch_tokens,
        tokenization_workers,
        tokenization_queue_capacity: tokenization_queue_capacity.get(),
        max_batch_requests,
        max_client_batch_size,
        max_decision_questions,
        auto_truncate,
        radix_mlp_threshold,
        enable_fp8_dynamic,
        version: env!("CARGO_PKG_VERSION"),
        sha: option_env!("VERGEN_GIT_SHA"),
        docker_label: option_env!("DOCKER_LABEL"),
    };

    let checkpoint_id = info.model_id.clone();
    let named = served_model_name.is_some();
    let mut info = info;
    if let Some(name) = served_model_name {
        info.model_id = name;
    }
    Ok(ModelRuntime {
        infer,
        info,
        checkpoint_id,
        named,
        backend_kind,
        systemone,
    })
}

/// Initialize all models before publishing one registry and starting one listener.
#[allow(clippy::too_many_arguments)]
pub async fn run_models(
    configs: Vec<ModelSettings>,
    hf_token: Option<String>,
    hostname: Option<String>,
    port: u16,
    huggingface_hub_cache: Option<String>,
    payload_limit: usize,
    api_key: Option<String>,
    prometheus_port: u16,
    cors_allow_origin: Option<Vec<String>>,
) -> Result<()> {
    billing_headers()?;
    anyhow::ensure!(!configs.is_empty(), "At least one model is required");
    // CUDA discovery can block on the driver. Overlap it with frontend setup
    // without blocking a Tokio worker or starting unvalidated weight loading.
    let multiple = configs.len() > 1;
    let mut configs = configs;
    for config in &mut configs {
        if multiple && config.served_model_name.is_none() {
            config.served_model_name = Some(
                model_aliases::resolve(
                    config.model_id.clone(),
                    config.decision_protocol.clone(),
                    config.pooling.as_ref(),
                )?
                .0,
            );
        }
    }
    let validation_configs = configs.clone();
    let mut validation = Some(tokio::task::spawn_blocking(move || {
        let started = std::time::Instant::now();
        let result = model_registry::validate_configs(&validation_configs);
        tracing::info!(
            seconds = started.elapsed().as_secs_f64(),
            "Startup configuration validation finished"
        );
        result
    }));
    let mut models = Vec::with_capacity(configs.len());
    for config in configs {
        let name = config
            .served_model_name
            .clone()
            .unwrap_or_else(|| config.model_id.clone());
        let initialized = initialize_model(
            config,
            hf_token.clone(),
            huggingface_hub_cache.clone(),
            &mut validation,
        )
        .await;
        // Drain validation even when frontend setup fails before the barrier.
        let validated = finish_startup_validation(&mut validation).await;
        match initialized.and_then(|model| validated.map(|()| model)) {
            Ok(model) => models.push(std::sync::Arc::new(model)),
            Err(error) => {
                for model in &models {
                    model.shutdown().await;
                }
                return Err(error.context(format!("Could not initialize model {name}")));
            }
        }
    }
    let registry = match model_registry::ModelRegistry::new(models.clone()) {
        Ok(registry) => registry,
        Err(error) => {
            for model in models {
                model.shutdown().await;
            }
            return Err(error);
        }
    };
    let ip = hostname
        .unwrap_or_else(|| "0.0.0.0".into())
        .parse()
        .unwrap_or(IpAddr::V4(Ipv4Addr::UNSPECIFIED));
    let addr = SocketAddr::new(ip, port);
    let max_input_length = registry
        .models()
        .iter()
        .map(|m| m.info.max_input_length)
        .max()
        .unwrap();
    let result = async {
        let prom_builder = prometheus::prometheus_builer(addr, prometheus_port, max_input_length)?;
        http::server::run(
            registry.clone(),
            addr,
            prom_builder,
            payload_limit,
            api_key,
            cors_allow_origin,
        )
        .await
    }
    .await;
    registry.shutdown().await;
    result
}

fn get_backend_model_type(
    config: &ModelConfig,
    model_root: &Path,
    pooling: Option<infinity_v3_backend::Pool>,
) -> Result<infinity_v3_backend::ModelType> {
    if config.architectures.iter().any(|arch| arch == "HF_ColBERT") {
        anyhow::ensure!(
            pooling.is_none(),
            "ColBERT returns token embeddings and does not support pooling"
        );
        return Ok(infinity_v3_backend::ModelType::Embedding(
            infinity_v3_backend::Pool::Mean,
        ));
    }
    for arch in &config.architectures {
        // Edge case affecting `Alibaba-NLP/gte-multilingual-base` and possibly other fine-tunes of
        // the same base model. More context at https://huggingface.co/Alibaba-NLP/gte-multilingual-base/discussions/7
        if arch == "NewForTokenClassification"
            && (config.id2label.is_none() | config.label2id.is_none())
        {
            tracing::warn!("Provided `--model-id` is likely an AlibabaNLP GTE model, but the `config.json` contains the architecture `NewForTokenClassification` but it doesn't contain the `id2label` and `label2id` mapping, so `NewForTokenClassification` architecture will be ignored.");
            continue;
        }

        if Some(infinity_v3_backend::Pool::Splade) == pooling && arch.ends_with("MaskedLM") {
            return Ok(infinity_v3_backend::ModelType::Embedding(
                infinity_v3_backend::Pool::Splade,
            ));
        } else if arch.ends_with("Classification") {
            if pooling.is_some() {
                tracing::warn!(
                    "`--pooling` arg is set but model is a classifier. Ignoring `--pooling` arg."
                );
            }
            return Ok(infinity_v3_backend::ModelType::Classifier);
        }
    }

    if Some(infinity_v3_backend::Pool::Splade) == pooling {
        return Err(anyhow!(
            "Splade pooling is not supported: model is not a ForMaskedLM model"
        ));
    }

    // Set pooling
    let pool = match pooling {
        Some(pool) => pool,
        None => {
            // Load pooling config
            let config_path = model_root.join("1_Pooling/config.json");

            match fs::read_to_string(config_path) {
                Ok(config) => {
                    let config: PoolConfig = serde_json::from_str(&config)
                        .context("Failed to parse `1_Pooling/config.json`")?;
                    Pool::try_from(config)?
                }
                Err(err) => {
                    if config.model_type == "mistral3" && err.kind() == std::io::ErrorKind::NotFound
                    {
                        return Ok(infinity_v3_backend::ModelType::Embedding(
                            infinity_v3_backend::Pool::LastToken,
                        ));
                    }
                    if !config.model_type.to_lowercase().contains("bert") {
                        return Err(err).context("The `--pooling` arg is not set and we could not find a pooling configuration (`1_Pooling/config.json`) for this model.");
                    }
                    tracing::warn!("The `--pooling` arg is not set and we could not find a pooling configuration (`1_Pooling/config.json`) for this model but the model is a BERT variant. Defaulting to `CLS` pooling.");
                    infinity_v3_backend::Pool::Cls
                }
            }
        }
    };

    Ok(infinity_v3_backend::ModelType::Embedding(pool))
}

// Resolve Auto before passing a concrete dtype to the backend or reporting /info.
fn resolve_libtorch_dtype(
    requested: Option<DType>,
    model_dtype: Option<&str>,
    model_type: &str,
    cpu: bool,
) -> DType {
    match requested {
        None | Some(DType::Auto) if cpu => DType::Float32,
        None | Some(DType::Auto) => {
            if model_dtype == Some("bfloat16")
                || matches!(model_type, "gemma3_text" | "embedding_gemma2")
            {
                DType::Bfloat16
            } else {
                DType::Float16
            }
        }
        Some(dtype) => dtype,
    }
}

#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    #[serde(default)]
    pub architectures: Vec<String>,
    pub model_type: String,
    pub dtype: Option<String>,
    pub torch_dtype: Option<String>,
    #[serde(default)]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub n_positions: Option<usize>,
    pub text_config: Option<TextPositionConfig>,
    pub vocab_size: Option<usize>,
    #[serde(default)]
    pub pad_token_id: Option<usize>,
    pub num_labels: Option<usize>,
    pub id2label: Option<HashMap<String, String>>,
    pub label2id: Option<HashMap<String, usize>>,
    pub auto_map: Option<HashMap<String, String>>,
    pub use_bidirectional_attention: Option<bool>,
}

impl ModelConfig {
    fn resolve_decoder_classifier_labels(&mut self) -> anyhow::Result<()> {
        if !self.architectures.iter().any(|arch| {
            matches!(
                arch.as_str(),
                "LlamaForSequenceClassification"
                    | "LlamaBidirectionalForSequenceClassification"
                    | "LlamaForTokenClassification"
                    | "UltrabertForSequenceClassification"
                    | "UltrabertForTokenClassification"
                    | "Qwen2ForSequenceClassification"
                    | "Qwen2ForTokenClassification"
                    | "Qwen3ForSequenceClassification"
                    | "Qwen3ForTokenClassification"
                    | "Qwen3MoeForSequenceClassification"
                    | "Gemma4ForSequenceClassification"
                    | "Mistral3ForSequenceClassification"
            )
        }) {
            return Ok(());
        }
        // HF omits the default two-label mapping from saved configs.
        let labels = self.id2label.get_or_insert_with(|| {
            (0..self.num_labels.unwrap_or(2))
                .map(|i| (i.to_string(), format!("LABEL_{i}")))
                .collect()
        });
        anyhow::ensure!(
            !labels.is_empty() && (0..labels.len()).all(|i| labels.contains_key(&i.to_string())),
            "Decoder classifier id2label must contain consecutive IDs starting at zero"
        );
        if self.label2id.is_none() {
            self.label2id = Some(
                labels
                    .iter()
                    .map(|(id, label)| Ok((label.clone(), id.parse::<usize>()?)))
                    .collect::<Result<_, std::num::ParseIntError>>()?,
            );
        }
        Ok(())
    }

    fn resolve_text_config(&mut self) -> anyhow::Result<()> {
        // Nomic publishes both the original training length and its extended
        // rotary context. Keep the native encoder's n_positions limit.
        if self.model_type == "nomic_bert" || self.max_position_embeddings == 0 {
            self.max_position_embeddings = self.n_positions.unwrap_or(self.max_position_embeddings);
        }
        // UltraBERT omits this field; its published tokenizer supports 32K.
        if self.model_type == "ultrabert" && self.max_position_embeddings == 0 {
            self.max_position_embeddings = 32768;
        }
        if matches!(
            self.model_type.as_str(),
            "gemma4"
                | "gemma4_unified"
                | "embedding_gemma2"
                | "qwen3_5_moe"
                | "qwen3_5_moe_text"
                | "qwen3_5"
                | "qwen3_5_text"
                | "qwen3_vl"
                | "mistral3"
                | "ultravbert"
        ) {
            if self.max_position_embeddings == 0 {
                self.max_position_embeddings = self
                    .text_config
                    .as_ref()
                    .context("Model text_config.max_position_embeddings is missing")?
                    .max_position_embeddings;
            }
            if matches!(
                self.model_type.as_str(),
                "qwen3_5_moe" | "qwen3_5_moe_text" | "qwen3_5" | "qwen3_5_text"
            ) && self.dtype.is_none()
                && self.torch_dtype.is_none()
            {
                if let Some(text) = &self.text_config {
                    self.dtype = text.dtype.clone();
                    self.torch_dtype = text.torch_dtype.clone();
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
pub struct TextPositionConfig {
    pub vocab_size: Option<usize>,
    pub max_position_embeddings: usize,
    pub dtype: Option<String>,
    pub torch_dtype: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct PoolConfig {
    #[serde(default)]
    pooling_mode_cls_token: bool,
    #[serde(default)]
    pooling_mode_mean_tokens: bool,
    #[serde(default)]
    pooling_mode_lasttoken: bool,
    #[serde(default)]
    pooling_mode: Option<String>,
}

impl TryFrom<PoolConfig> for Pool {
    type Error = anyhow::Error;

    fn try_from(config: PoolConfig) -> std::result::Result<Self, Self::Error> {
        if let Some(mode) = &config.pooling_mode {
            return match mode.as_str() {
                "cls" => Ok(Pool::Cls),
                "mean" => Ok(Pool::Mean),
                "lasttoken" => Ok(Pool::LastToken),
                _ => Err(anyhow!("Pooling mode {mode:?} is not supported")),
            };
        }
        if config.pooling_mode_cls_token {
            return Ok(Pool::Cls);
        }
        if config.pooling_mode_mean_tokens {
            return Ok(Pool::Mean);
        }
        if config.pooling_mode_lasttoken {
            return Ok(Pool::LastToken);
        }
        Err(anyhow!("Pooling config {config:?} is not supported"))
    }
}

#[derive(Debug, Deserialize)]
pub struct STConfig {
    pub max_seq_length: usize,
}

#[derive(Debug, Deserialize)]
pub struct NewSTConfig {
    pub prompts: Option<HashMap<String, String>>,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "http", derive(utoipa::ToSchema))]
pub struct EmbeddingModel {
    #[cfg_attr(feature = "http", schema(example = "cls"))]
    pub pooling: String,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "http", derive(utoipa::ToSchema))]
pub struct ClassifierModel {
    #[cfg_attr(feature = "http", schema(example = json!({"0": "LABEL"})))]
    pub id2label: HashMap<String, String>,
    #[cfg_attr(feature = "http", schema(example = json!({"LABEL": 0})))]
    pub label2id: HashMap<String, usize>,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "http", derive(utoipa::ToSchema))]
#[serde(rename_all = "lowercase")]
pub enum ModelType {
    Decision,
    Classifier(ClassifierModel),
    Embedding(EmbeddingModel),
    Reranker(ClassifierModel),
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "http", derive(utoipa::ToSchema))]
pub struct Info {
    /// Model info
    #[cfg_attr(feature = "http", schema(example = "thenlper/gte-base"))]
    pub model_id: String,
    #[cfg_attr(
        feature = "http",
        schema(nullable = true, example = "fca14538aa9956a46526bd1d0d11d69e19b5a101")
    )]
    pub model_sha: Option<String>,
    #[cfg_attr(feature = "http", schema(example = "float16"))]
    pub model_dtype: String,
    pub model_type: ModelType,
    /// Router Parameters
    #[cfg_attr(feature = "http", schema(example = "128"))]
    pub max_concurrent_requests: usize,
    #[cfg_attr(feature = "http", schema(example = "512"))]
    pub max_input_length: usize,
    #[cfg_attr(feature = "http", schema(example = "2048"))]
    pub max_batch_tokens: usize,
    #[cfg_attr(
        feature = "http",
        schema(nullable = true, example = "null", default = "null")
    )]
    pub max_batch_requests: Option<usize>,
    #[cfg_attr(feature = "http", schema(example = "32"))]
    pub max_client_batch_size: usize,
    /// Maximum question fanout for /v1/systemone, independent of the client batch limit.
    pub max_decision_questions: usize,
    pub auto_truncate: bool,
    #[cfg_attr(feature = "http", schema(example = "4"))]
    pub tokenization_workers: usize,
    /// Maximum waiting tokenizer jobs (individual inputs, not HTTP requests).
    #[cfg_attr(feature = "http", schema(example = "1024"))]
    pub tokenization_queue_capacity: usize,
    #[cfg_attr(feature = "http", schema(example = "0.5"))]
    pub radix_mlp_threshold: f32,
    /// Experimental per-token activation and per-row weight FP8 projection quantization.
    pub enable_fp8_dynamic: bool,
    /// Router Info
    #[cfg_attr(feature = "http", schema(example = "0.5.0"))]
    pub version: &'static str,
    #[cfg_attr(feature = "http", schema(nullable = true, example = "null"))]
    pub sha: Option<&'static str>,
    #[cfg_attr(feature = "http", schema(nullable = true, example = "null"))]
    pub docker_label: Option<&'static str>,
}

impl Info {
    pub fn is_multi_vector(&self) -> bool {
        matches!(&self.model_type, ModelType::Embedding(model) if model.pooling == "colbert")
    }
}

#[derive(Serialize)]
#[cfg_attr(feature = "http", derive(utoipa::ToSchema))]
pub enum ErrorType {
    Unhealthy,
    Backend,
    Overloaded,
    Validation,
    Tokenizer,
    Empty,
}

#[derive(Serialize)]
#[cfg_attr(feature = "http", derive(utoipa::ToSchema))]
pub struct ErrorResponse {
    pub error: String,
    pub error_type: ErrorType,
}

impl From<TextEmbeddingsError> for ErrorResponse {
    fn from(err: TextEmbeddingsError) -> Self {
        let error_type = match err {
            TextEmbeddingsError::Tokenizer(_) => ErrorType::Tokenizer,
            TextEmbeddingsError::Validation(_) => ErrorType::Validation,
            TextEmbeddingsError::Empty(_) => ErrorType::Empty,
            TextEmbeddingsError::Overloaded(_) => ErrorType::Overloaded,
            TextEmbeddingsError::Backend(infinity_v3_backend::BackendError::Overloaded(_)) => {
                ErrorType::Overloaded
            }
            TextEmbeddingsError::Backend(_) => ErrorType::Backend,
        };
        Self {
            error: err.to_string(),
            error_type,
        }
    }
}

struct BillingHeaders {
    input_tokens: HeaderName,
    service_tier: HeaderName,
    priority_multiplier: Option<BigDecimal>,
    flex_multiplier: Option<BigDecimal>,
}

static BILLING_HEADERS: OnceLock<Result<Option<BillingHeaders>, String>> = OnceLock::new();

// Called at startup to validate and freeze the process-wide header configuration.
fn billing_headers() -> Result<&'static Option<BillingHeaders>> {
    const ENV: &str = infinity_v3_config::infinity_env!("BILLING_HEADERS");
    BILLING_HEADERS
        .get_or_init(|| match std::env::var(ENV) {
            Ok(prefix) if prefix.is_empty() => Ok(None),
            Ok(prefix) => {
                let priority_multiplier = billing_multiplier(&[
                    infinity_v3_config::infinity_env!("EXPERIMENTAL_MULTIPLIER_PRIORITY"),
                    infinity_v3_config::infinity_env!("EXPERIMENTAL_MULTIPLIER_PRIOIRTY"),
                    infinity_v3_config::infinity_env!("EXPERIMENTAL_MULTIPLIER"),
                ])?;
                let flex_multiplier = billing_multiplier(&[infinity_v3_config::infinity_env!(
                    "EXPERIMENTAL_MULTIPLIER_FLEX"
                )])?;
                let name = |suffix| {
                    HeaderName::from_bytes(format!("{prefix}-{suffix}").as_bytes()).map_err(|_| {
                        format!("{ENV} must be empty or a valid HTTP header-name prefix")
                    })
                };
                Ok(Some(BillingHeaders {
                    input_tokens: name("input-tokens")?,
                    service_tier: name("service-tier")?,
                    priority_multiplier,
                    flex_multiplier,
                }))
            }
            Err(std::env::VarError::NotPresent) => Ok(None),
            Err(std::env::VarError::NotUnicode(_)) => Err(format!("{ENV} must be valid Unicode")),
        })
        .as_ref()
        .map_err(|error| anyhow!(error.clone()))
}

// First configured name wins; invalid values fail startup rather than falling back.
fn billing_multiplier(names: &[&str]) -> std::result::Result<Option<BigDecimal>, String> {
    for name in names {
        match std::env::var(name) {
            Ok(value) => {
                return value
                    .parse::<f64>()
                    .ok()
                    .filter(|value| value.is_finite() && *value > 0.0)
                    .and_then(|_| value.parse::<BigDecimal>().ok())
                    .ok_or_else(|| format!("{name} must be a finite positive float"))
                    .map(Some);
            }
            Err(std::env::VarError::NotPresent) => continue,
            Err(std::env::VarError::NotUnicode(_)) => {
                return Err(format!("{name} must be valid Unicode"));
            }
        }
    }
    Ok(None)
}

impl BillingHeaders {
    fn multiplier(&self, tier: QueueTier) -> Option<&BigDecimal> {
        match tier {
            QueueTier::Default => None,
            QueueTier::Priority => self.priority_multiplier.as_ref(),
            QueueTier::Flex => self.flex_multiplier.as_ref(),
        }
    }
}

fn configured_billing_headers() -> Option<&'static BillingHeaders> {
    BILLING_HEADERS.get()?.as_ref().ok()?.as_ref()
}

fn billing_token_count(tokens: usize, multiplier: Option<&BigDecimal>) -> String {
    multiplier.map_or_else(
        || tokens.to_string(),
        |multiplier| {
            (BigDecimal::from(tokens as u64) * multiplier)
                .with_scale_round(0, RoundingMode::Ceiling)
                .to_string()
        },
    )
}

struct ResponseMetadata {
    service_tier: QueueTier,
    compute_chars: usize,
    compute_tokens: usize,
    start_time: Instant,
    tokenization_time: Duration,
    queue_time: Duration,
    inference_time: Duration,
}

impl ResponseMetadata {
    fn new(
        compute_chars: usize,
        compute_tokens: usize,
        start_time: Instant,
        tokenization_time: Duration,
        queue_time: Duration,
        inference_time: Duration,
        service_tier: QueueTier,
    ) -> Self {
        Self {
            service_tier,
            compute_chars,
            compute_tokens,
            start_time,
            tokenization_time,
            queue_time,
            inference_time,
        }
    }

    fn record_span(&self, span: &Span) {
        // Tracing metadata
        span.record("compute_chars", self.compute_chars);
        span.record("compute_tokens", self.compute_tokens);
        span.record("total_time", format!("{:?}", self.start_time.elapsed()));
        span.record("tokenization_time", format!("{:?}", self.tokenization_time));
        span.record("queue_time", format!("{:?}", self.queue_time));
        span.record("inference_time", format!("{:?}", self.inference_time));
    }

    fn record_metrics(&self, model_name: &str) {
        // Metrics
        let histogram =
            metrics::histogram!("infinity_request_duration", "model" => model_name.to_owned());
        histogram.record(self.start_time.elapsed().as_secs_f64());
        let histogram = metrics::histogram!("infinity_request_tokenization_duration", "model" => model_name.to_owned());
        histogram.record(self.tokenization_time.as_secs_f64());
        let histogram = metrics::histogram!("infinity_request_queue_duration", "model" => model_name.to_owned());
        histogram.record(self.queue_time.as_secs_f64());
        let histogram = metrics::histogram!("infinity_request_inference_duration", "model" => model_name.to_owned());
        histogram.record(self.inference_time.as_secs_f64());
    }
}

#[cfg(feature = "http")]
fn service_tier_header(tier: QueueTier) -> HeaderValue {
    HeaderValue::from_static(tier.as_str())
}

#[cfg(feature = "http")]
fn service_tier_headers(tier: QueueTier) -> HeaderMap {
    configured_billing_headers().map_or_else(HeaderMap::new, |names| {
        HeaderMap::from_iter([(names.service_tier.clone(), service_tier_header(tier))])
    })
}

impl From<ResponseMetadata> for HeaderMap {
    fn from(value: ResponseMetadata) -> Self {
        // Headers
        let mut headers = service_tier_headers(value.service_tier);
        if let Some(names) = configured_billing_headers() {
            let tokens =
                billing_token_count(value.compute_tokens, names.multiplier(value.service_tier));
            // Decimal integer text is valid HTTP header text.
            if let Ok(tokens) = HeaderValue::try_from(tokens) {
                headers.insert(names.input_tokens.clone(), tokens);
            }
        }
        headers.insert("x-compute-type", "gpu+optimized".parse().unwrap());
        headers.insert(
            "x-compute-time",
            value
                .start_time
                .elapsed()
                .as_millis()
                .to_string()
                .parse()
                .unwrap(),
        );
        headers.insert(
            "x-compute-characters",
            value.compute_chars.to_string().parse().unwrap(),
        );
        headers.insert(
            "x-compute-tokens",
            value.compute_tokens.to_string().parse().unwrap(),
        );
        headers.insert(
            "x-total-time",
            value
                .start_time
                .elapsed()
                .as_millis()
                .to_string()
                .parse()
                .unwrap(),
        );
        headers.insert(
            "x-tokenization-time",
            value
                .tokenization_time
                .as_millis()
                .to_string()
                .parse()
                .unwrap(),
        );
        headers.insert(
            "x-queue-time",
            value.queue_time.as_millis().to_string().parse().unwrap(),
        );
        headers.insert(
            "x-inference-time",
            value
                .inference_time
                .as_millis()
                .to_string()
                .parse()
                .unwrap(),
        );
        headers
    }
}

#[cfg(test)]
mod libtorch_auto_dtype_tests {
    use super::*;

    #[test]
    fn nomic_accepts_current_and_legacy_context_fields() {
        for (positions, expected) in [
            (serde_json::json!({"n_positions": 8192}), 8192),
            (serde_json::json!({"max_position_embeddings": 2048}), 2048),
            (
                serde_json::json!({"max_position_embeddings": 2048, "n_positions": 8192}),
                8192,
            ),
        ] {
            let mut value = serde_json::json!({"model_type": "nomic_bert"});
            value
                .as_object_mut()
                .unwrap()
                .extend(positions.as_object().unwrap().clone());
            let mut config: ModelConfig = serde_json::from_value(value).unwrap();
            config.resolve_text_config().unwrap();
            assert_eq!(config.max_position_embeddings, expected);
        }
    }

    #[test]
    fn bf16_checkpoint_auto_dtype_survives_root_and_nested_configs() {
        for field in ["dtype", "torch_dtype"] {
            for nested in [false, true] {
                let mut value = serde_json::json!({
                    "model_type": "qwen3_5_moe",
                    "max_position_embeddings": 2048,
                    "text_config": {"max_position_embeddings": 2048}
                });
                if nested {
                    value["text_config"][field] = "bfloat16".into();
                } else {
                    value[field] = "bfloat16".into();
                }
                let mut config: ModelConfig = serde_json::from_value(value).unwrap();
                config.resolve_text_config().unwrap();
                let declared = config.dtype.as_deref().or(config.torch_dtype.as_deref());
                for requested in [None, Some(DType::Auto)] {
                    assert_eq!(
                        resolve_libtorch_dtype(requested, declared, &config.model_type, false),
                        DType::Bfloat16
                    );
                }
                assert_eq!(
                    resolve_libtorch_dtype(None, declared, &config.model_type, true),
                    DType::Float32
                );
                assert_eq!(
                    resolve_libtorch_dtype(
                        Some(DType::Float16),
                        declared,
                        &config.model_type,
                        false
                    ),
                    DType::Float16
                );
            }
        }
    }

    #[test]
    fn gemma_requirements_and_device_defaults_preserve_explicit_overrides() {
        for model in ["gemma3_text", "embedding_gemma2"] {
            assert_eq!(
                resolve_libtorch_dtype(Some(DType::Auto), Some("float32"), model, false),
                DType::Bfloat16
            );
            assert_eq!(
                resolve_libtorch_dtype(Some(DType::Float32), Some("bfloat16"), model, false),
                DType::Float32
            );
        }
        assert_eq!(
            resolve_libtorch_dtype(None, None, "bert", false),
            DType::Float16
        );
        assert_eq!(
            resolve_libtorch_dtype(Some(DType::Auto), None, "bert", true),
            DType::Float32
        );
        assert_eq!(
            resolve_libtorch_dtype(Some(DType::Bfloat16), None, "bert", true),
            DType::Bfloat16
        );
    }
}

#[cfg(test)]
mod auto_dtype_tests {
    use super::*;

    #[test]
    fn ultrabert_resolves_released_context_limit_and_preserves_overrides() {
        for (limit, expected) in [(None, 32768), (Some(8192), 8192)] {
            let mut value =
                serde_json::json!({"model_type":"ultrabert", "architectures":["UltrabertModel"]});
            if let Some(limit) = limit {
                value["max_position_embeddings"] = limit.into();
            }
            let mut config: ModelConfig = serde_json::from_value(value).unwrap();
            config.resolve_text_config().unwrap();
            assert_eq!(config.max_position_embeddings, expected);
        }
    }

    #[test]
    fn decoder_classifiers_resolve_hf_default_and_explicit_labels() {
        for (arch, model_type) in [
            ("LlamaForSequenceClassification", "llama"),
            ("LlamaForTokenClassification", "llama"),
            ("UltrabertForSequenceClassification", "ultrabert"),
            ("UltrabertForTokenClassification", "ultrabert"),
            ("Qwen2ForSequenceClassification", "qwen2"),
            ("Qwen2ForTokenClassification", "qwen2"),
            ("Qwen3ForSequenceClassification", "qwen3"),
            ("Qwen3ForTokenClassification", "qwen3"),
            ("Qwen3MoeForSequenceClassification", "qwen3_moe"),
            ("Gemma4ForSequenceClassification", "gemma4"),
        ] {
            let mut config: ModelConfig = serde_json::from_value(serde_json::json!({
                "architectures":[arch], "model_type":model_type, "max_position_embeddings":128
            }))
            .unwrap();
            config.resolve_decoder_classifier_labels().unwrap();
            assert!(matches!(
                get_backend_model_type(&config, Path::new("."), None).unwrap(),
                infinity_v3_backend::ModelType::Classifier
            ));
            assert_eq!(config.id2label.as_ref().unwrap()["1"], "LABEL_1");
            assert_eq!(config.label2id.as_ref().unwrap()["LABEL_1"], 1);
            config.num_labels = Some(3);
            config.id2label = None;
            config.label2id = None;
            config.resolve_decoder_classifier_labels().unwrap();
            assert_eq!(config.id2label.as_ref().unwrap()["2"], "LABEL_2");
            assert_eq!(config.label2id.as_ref().unwrap()["LABEL_2"], 2);
            config.num_labels = None;
            config.id2label = Some(HashMap::from([("0".into(), "relevance".into())]));
            config.label2id = None;
            config.resolve_decoder_classifier_labels().unwrap();
            assert_eq!(config.label2id.as_ref().unwrap()["relevance"], 0);
            config.id2label = Some(HashMap::from([("1".into(), "missing_zero".into())]));
            config.label2id = None;
            assert!(config.resolve_decoder_classifier_labels().is_err());
        }
    }

    fn from_config(json: &str, requested: Option<DType>) -> DType {
        let config: ModelConfig = serde_json::from_str(json).unwrap();
        resolve_libtorch_dtype(
            requested,
            config.dtype.as_deref().or(config.torch_dtype.as_deref()),
            &config.model_type,
            false,
        )
    }

    #[test]
    fn auto_selects_bf16_unless_overridden() {
        for model_type in ["qwen3", "qwen3_vl", "qwen3_5"] {
            for field in ["dtype", "torch_dtype"] {
                let json = format!(
                    r#"{{"architectures":[],"model_type":"{model_type}","max_position_embeddings":512,"{field}":"bfloat16"}}"#
                );
                assert_eq!(from_config(&json, Some(DType::Auto)), DType::Bfloat16);
                assert_eq!(from_config(&json, Some(DType::Float16)), DType::Float16);
            }
        }
    }

    #[test]
    fn embeddinggemma_defaults_to_bf16_even_for_fp32_checkpoint() {
        assert_eq!(
            resolve_libtorch_dtype(None, Some("float32"), "gemma3_text", false),
            DType::Bfloat16
        );
        assert_eq!(
            resolve_libtorch_dtype(Some(DType::Auto), Some("float32"), "gemma3_text", false),
            DType::Bfloat16
        );
        // An explicit unsupported precision is rejected by the model loader.
        assert_eq!(
            resolve_libtorch_dtype(Some(DType::Float32), None, "gemma3_text", false),
            DType::Float32
        );
    }

    #[test]
    fn embedding_gemma2_config_and_pooling_formats() {
        let mut config: ModelConfig = serde_json::from_value(serde_json::json!({
            "architectures": ["EmbeddingGemma2Model"], "model_type": "embedding_gemma2",
            "dtype": "bfloat16", "text_config": {"max_position_embeddings":262144, "vocab_size":262144}
        })).unwrap();
        config.resolve_text_config().unwrap();
        assert_eq!(config.max_position_embeddings, 262144);
        assert_eq!(
            resolve_libtorch_dtype(None, None, "embedding_gemma2", false),
            DType::Bfloat16
        );
        for value in [
            serde_json::json!({"embedding_dimension":768,"pooling_mode":"mean","include_prompt":true}),
            serde_json::json!({"pooling_mode_cls_token":false,"pooling_mode_mean_tokens":true}),
        ] {
            assert_eq!(
                Pool::try_from(serde_json::from_value::<PoolConfig>(value).unwrap()).unwrap(),
                Pool::Mean
            );
        }
        assert!(Pool::try_from(
            serde_json::from_value::<PoolConfig>(serde_json::json!({"pooling_mode":"unsupported"}))
                .unwrap()
        )
        .is_err());
    }

    #[test]
    fn qwen35_nested_context_and_dtype() {
        for model_type in ["qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"] {
            let mut config: ModelConfig = serde_json::from_value(serde_json::json!({
                "model_type": model_type,
                "text_config": {"max_position_embeddings":262144,"dtype":"bfloat16"}
            }))
            .unwrap();
            config.resolve_text_config().unwrap();
            assert_eq!(config.max_position_embeddings, 262144);
            assert_eq!(config.dtype.as_deref(), Some("bfloat16"));
        }
    }

    #[test]
    fn mistral3_context_length_is_nested() {
        let mut config:ModelConfig=serde_json::from_value(serde_json::json!({"architectures":["Mistral3ForConditionalGeneration"],"model_type":"mistral3","dtype":"bfloat16","text_config":{"max_position_embeddings":262144,"vocab_size":131072}})).unwrap();
        config.resolve_text_config().unwrap();
        assert_eq!(config.max_position_embeddings, 262144);
        assert_eq!(
            resolve_libtorch_dtype(None, config.dtype.as_deref(), "mistral3", false),
            DType::Bfloat16
        );
    }

    #[test]
    fn gemma4_context_length_is_nested() {
        let config: ModelConfig = serde_json::from_str(
            r#"{"architectures":[],"model_type":"gemma4","text_config":{"max_position_embeddings":131072}}"#,
        )
        .unwrap();
        assert_eq!(config.max_position_embeddings, 0);
        assert_eq!(config.text_config.unwrap().max_position_embeddings, 131072);
    }
}

#[cfg(test)]
mod billing_token_tests {
    use super::*;

    #[test]
    fn decimal_billing_rounds_up_without_binary_float_overcharges() {
        for (tokens, multiplier, expected) in [
            (100, "1.1", "110"),
            (1900, "1.07", "2033"),
            (6, "1.2", "8"),
            (8, "0.9", "8"),
            (100, "0.9", "90"),
        ] {
            let multiplier: BigDecimal = multiplier.parse().unwrap();
            assert_eq!(billing_token_count(tokens, Some(&multiplier)), expected);
        }
        assert_eq!(billing_token_count(100, None), "100");
        let headers = BillingHeaders {
            input_tokens: HeaderName::from_static("x-test-input-tokens"),
            service_tier: HeaderName::from_static("x-test-service-tier"),
            priority_multiplier: Some("2.0".parse().unwrap()),
            flex_multiplier: Some("0.9".parse().unwrap()),
        };
        for (tier, expected) in [
            (QueueTier::Default, "100"),
            (QueueTier::Priority, "200"),
            (QueueTier::Flex, "90"),
        ] {
            assert_eq!(billing_token_count(100, headers.multiplier(tier)), expected);
        }
    }
}
