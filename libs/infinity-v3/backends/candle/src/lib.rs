mod alibi;
#[cfg(any(feature = "cuda", test))]
mod compute_cap;
#[cfg(feature = "fa4")]
mod fa4_native;
mod flash_attn;
mod layers;
mod models;

pub use models::{LayaConfig, LayaModel, LayaOutput};

use anyhow::Context;
use candle::{DType, Device};
use candle_nn::VarBuilder;
use infinity_v3_backend_core::{
    Backend, BackendError, Batch, Embedding, Embeddings, ModelType, Pool, Predictions,
    TokenPredictions,
};
use nohash_hasher::BuildNoHashHasher;
use serde::{de::Deserializer, Deserialize};
use std::collections::HashMap;
use std::path::Path;

#[cfg(feature = "cuda")]
use crate::compute_cap::{
    compatible_compute_cap, get_compile_compute_cap, get_runtime_compute_cap,
};
use crate::models::{
    BertConfig, Dense, DenseConfig, DenseLayer, DistilBertConfig, EmbeddingGemma2Config, GTEConfig,
    Gemma3Config, Gemma4Config, LLamaConfig, MPNetConfig, MistralConfig, Model, ModernBertConfig,
    NomicConfig, Qwen2Config, Qwen3Config,
};
use crate::models::{
    FlashBertModel, FlashDistilBertModel, FlashGTEModel, FlashJinaBertModel,
    FlashJinaCodeBertModel, FlashModernBertModel, FlashNomicBertModel,
};
use crate::models::{FlashMistralModel, FlashQwen2Model, FlashQwen3Model};

/// This enum is needed to be able to differentiate between jina models that also use
/// the `bert` model type and valid Bert models.
#[derive(Debug, Clone, PartialEq)]
pub enum BertConfigWrapper {
    JinaBert(BertConfig),
    JinaCodeBert(BertConfig),
    Bert(BertConfig),
}

/// Custom deserializer is required as we need to capture both whether the `_name_or_path` value
/// is any of the JinaBERT alternatives, or alternatively to also support fine-tunes and re-uploads
/// with Sentence Transformers, we also need to check the value for the `auto_map.AutoConfig`
/// configuration file, and see if that points to the relevant remote code repositories on the Hub
impl<'de> Deserialize<'de> for BertConfigWrapper {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::Error;

        #[allow(unused_mut)]
        let mut value = serde_json::Value::deserialize(deserializer)?;

        let name_or_path = value
            .get("_name_or_path")
            .and_then(|v| v.as_str())
            .map(ToString::to_string)
            .unwrap_or_default();

        let auto_config = value
            .get("auto_map")
            .and_then(|v| v.get("AutoConfig"))
            .and_then(|v| v.as_str())
            .map(ToString::to_string)
            .unwrap_or_default();

        let config = BertConfig::deserialize(value).map_err(Error::custom)?;

        if name_or_path == "jinaai/jina-bert-implementation"
            || auto_config.contains("jinaai/jina-bert-implementation")
        {
            // https://huggingface.co/jinaai/jina-bert-implementation
            Ok(Self::JinaBert(config))
        } else if name_or_path == "jinaai/jina-bert-v2-qk-post-norm"
            || auto_config.contains("jinaai/jina-bert-v2-qk-post-norm")
        {
            // https://huggingface.co/jinaai/jina-bert-v2-qk-post-norm
            Ok(Self::JinaCodeBert(config))
        } else {
            Ok(Self::Bert(config))
        }
    }
}

#[derive(Deserialize)]
#[serde(tag = "model_type", rename_all = "kebab-case")]
enum Config {
    #[serde(rename = "ultrabert")]
    UltraBert(models::UltraBertConfig),
    #[cfg(feature = "experimental-deberta")]
    #[serde(rename = "deberta-v2")]
    Deberta(crate::models::DebertaConfig),
    Bert(BertConfigWrapper),
    Camembert(BertConfig),
    #[serde(rename(deserialize = "distilbert"))]
    DistilBert(DistilBertConfig),
    #[serde(rename(deserialize = "gemma3_text"))]
    // Parsed on CPU builds so unsupported Gemma3 execution gets an explicit error.
    #[cfg_attr(not(feature = "flash-attn"), allow(dead_code))]
    Gemma3(Gemma3Config),
    #[serde(rename = "embedding_gemma2")]
    EmbeddingGemma2(EmbeddingGemma2Config),
    #[serde(rename = "gemma4", alias = "gemma4_unified")]
    Gemma4(Gemma4Config),
    #[serde(alias = "new")]
    Gte(GTEConfig),
    #[serde(rename = "mpnet")]
    #[allow(dead_code)]
    MPNet(MPNetConfig),
    #[allow(dead_code)]
    #[serde(alias = "ministral3")]
    Mistral(MistralConfig),
    #[serde(rename(deserialize = "modernbert"))]
    ModernBert(ModernBertConfig),
    #[serde(rename(deserialize = "nomic_bert"))]
    NomicBert(NomicConfig),
    #[allow(dead_code)]
    Qwen2(Qwen2Config),
    #[allow(dead_code)]
    #[serde(alias = "qwen3_moe", alias = "bidirectional_pplx_qwen3")]
    Qwen3(Qwen3Config),
    #[allow(dead_code)]
    #[serde(rename = "qwen3_vl")]
    Qwen3Vl(serde_json::Value),
    #[serde(
        rename = "qwen3_5_moe",
        alias = "qwen3_5_moe_text",
        alias = "qwen3_5",
        alias = "qwen3_5_text",
        alias = "pplx_contextual_qwen3_5"
    )]
    Qwen35(models::Qwen35Config),
    Roberta(BertConfig),
    XlmRoberta(BertConfig),
    #[allow(dead_code)]
    #[serde(alias = "llama_bidirec")] // Also accept llama_bidirec
    Llama(LLamaConfig),
}

pub struct CandleBackend {
    device: Device,
    model: Box<dyn Model + Send>,
    dense_layers: Vec<Box<dyn DenseLayer + Send>>,
    token_dense: bool,
}

impl CandleBackend {
    pub fn new(
        model_path: &Path,
        dtype: String,
        model_type: ModelType,
        dense_paths: Option<Vec<String>>,
        device_id: usize,
    ) -> Result<Self, BackendError> {
        Self::new_with_fp8(model_path, dtype, model_type, dense_paths, device_id, false)
    }

    pub fn new_with_fp8(
        model_path: &Path,
        dtype: String,
        model_type: ModelType,
        dense_paths: Option<Vec<String>>,
        device_id: usize,
        enable_fp8_dynamic: bool,
    ) -> Result<Self, BackendError> {
        if enable_fp8_dynamic && !cfg!(feature = "experimental-fp8") {
            return Err(BackendError::Start(
                "Dynamic FP8 requires an experimental-fp8 build".into(),
            ));
        }
        if infinity_v3_backend_core::block_attention_config::block_attention(model_path)
            .map_err(BackendError::Start)?
        {
            return Err(BackendError::Start(
                "Block attention requires the LibTorch backend".into(),
            ));
        }
        // Default files
        let default_safetensors = model_path.join("model.safetensors");
        let default_pytorch = model_path.join("pytorch_model.bin");

        // Single Files
        let model_files = if default_safetensors.exists() {
            vec![default_safetensors]
        } else if default_pytorch.exists() {
            vec![default_pytorch]
        }
        // Sharded weights
        else {
            // Get index file
            let index_file = model_path.join("model.safetensors.index.json");

            // Parse file
            let index_file_string: String = std::fs::read_to_string(&index_file)
                .map_err(|err| BackendError::Start(err.to_string()))?;
            let json: serde_json::Value = serde_json::from_str(&index_file_string)
                .map_err(|err| BackendError::Start(err.to_string()))?;

            let weight_map = match json.get("weight_map") {
                None => {
                    return Err(BackendError::Start(format!(
                        "no weight map in {index_file:?}"
                    )));
                }
                Some(serde_json::Value::Object(map)) => map,
                Some(_) => {
                    return Err(BackendError::Start(format!(
                        "weight map in {index_file:?} is not a map"
                    )));
                }
            };
            let mut safetensors_files = std::collections::HashSet::new();
            for value in weight_map.values() {
                if let Some(file) = value.as_str() {
                    safetensors_files.insert(file.to_string());
                }
            }

            // Collect paths
            safetensors_files
                .iter()
                .map(|n| model_path.join(n))
                .collect()
        };

        // Get candle device
        let device = if candle::utils::cuda_is_available() {
            #[cfg(feature = "cuda")]
            match compatible_compute_cap(device_id) {
                Ok(true) => Device::new_cuda(device_id),
                Ok(false) => {
                    return Err(BackendError::Start(format!(
                        "Runtime compute cap {} is not compatible with compile time compute cap {}",
                        get_runtime_compute_cap(device_id).unwrap(),
                        get_compile_compute_cap().unwrap()
                    )));
                }
                Err(err) => {
                    return Err(BackendError::Start(format!(
                        "Could not initialize CUDA device {device_id}: {err:?}"
                    )));
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = device_id;
                Ok(Device::Cpu)
            }
        } else if candle::utils::metal_is_available() {
            Device::new_metal(0)
        } else {
            Ok(Device::Cpu)
        }
        .map_err(|err| BackendError::Start(err.to_string()))?;

        // Get candle dtype
        let dtype = if &dtype == "float32" {
            Ok(DType::F32)
        } else if &dtype == "float16" {
            Ok(DType::F16)
        } else if &dtype == "bfloat16" {
            Ok(DType::BF16)
        } else {
            Err(BackendError::Start(format!(
                "DType {dtype} is not supported"
            )))
        }?;

        #[cfg(feature = "cuda")]
        if dtype == DType::BF16
            && device.is_cuda()
            && get_runtime_compute_cap(device_id).unwrap_or(0) < 80
        {
            return Err(BackendError::Start(
                "bfloat16 CUDA inference requires compute capability 8.0 or newer".into(),
            ));
        }

        if model_type == ModelType::Decision && model_path.join("rl_agent_config.json").exists() {
            #[cfg(feature = "cuda")]
            if enable_fp8_dynamic
                && (!cfg!(feature = "flash-attn")
                    || !matches!(dtype, DType::F16 | DType::BF16)
                    || !device.is_cuda()
                    || !matches!(
                        get_runtime_compute_cap(device_id).unwrap_or(0),
                        89 | 90 | 100 | 120
                    ))
            {
                return Err(BackendError::Start(
                    "Laya dynamic FP8 requires flash-attn and float16/bfloat16 on SM89/90/100/120"
                        .into(),
                ));
            }
            let (model, _) =
                LayaModel::from_model_dir(model_path, dtype, &device, enable_fp8_dynamic)
                    .map_err(|e| BackendError::Start(format!("{e:#}")))?;
            return Ok(Self {
                device,
                model: Box::new(model),
                dense_layers: vec![],
                token_dense: false,
            });
        }

        let token_dense = infinity_v3_backend_core::perplexity::Profile::load(model_path)
            .map_err(BackendError::Start)?
            .is_some_and(|p| p.late());

        // Load config
        let config: String = std::fs::read_to_string(model_path.join("config.json"))
            .context("Unable to read config file")
            .map_err(|err| BackendError::Start(format!("{err:?}")))?;
        if enable_fp8_dynamic {
            let metadata: serde_json::Value = serde_json::from_str(&config)
                .map_err(|err| BackendError::Start(err.to_string()))?;
            if metadata
                .get("quantization_config")
                .is_some_and(|value| !value.is_null())
            {
                return Err(BackendError::Start(
                    "Dynamic FP8 requires an unquantized checkpoint; checkpoint-provided quantization scales are not supported".into(),
                ));
            }
        }
        let mut metadata: serde_json::Value =
            serde_json::from_str(&config).map_err(|e| BackendError::Start(e.to_string()))?;
        if metadata["model_type"] == "bidirectional_pplx_qwen3" {
            metadata["use_bidirectional_attention"] = true.into();
            if !metadata["rope_theta"].is_number() {
                metadata["rope_theta"] = metadata["rope_parameters"]["rope_theta"].clone();
            }
        }
        let config_json = metadata.to_string();
        let config: Config = serde_json::from_str(&config_json)
            .context("Model is not supported")
            .map_err(|err| BackendError::Start(format!("{err:?}")))?;

        if enable_fp8_dynamic {
            if !matches!(dtype, DType::F16 | DType::BF16)
                || !device.is_cuda()
                || !matches!(
                    &config,
                    Config::Qwen2(_)
                        | Config::Qwen3(_)
                        | Config::Llama(_)
                        | Config::Mistral(_)
                        | Config::Bert(BertConfigWrapper::Bert(_))
                        | Config::Roberta(_)
                        | Config::XlmRoberta(_)
                        | Config::Camembert(_)
                        | Config::ModernBert(_)
                )
                || !cfg!(feature = "flash-attn")
                || !std::env::var("USE_FLASH_ATTENTION")
                    .unwrap_or("true".into())
                    .eq_ignore_ascii_case("true")
            {
                return Err(BackendError::Start("Dynamic FP8 currently requires CUDA, float16/bfloat16, the flash-attn feature and a supported dense MLP model (Qwen2/Qwen3/Llama/Mistral/BERT/RoBERTa/ModernBERT)".into()));
            }
            #[cfg(feature = "cuda")]
            if !matches!(
                get_runtime_compute_cap(device_id).unwrap_or(0),
                89 | 90 | 100 | 120
            ) {
                return Err(BackendError::Start(
                    "Dynamic FP8 requires Ada SM89, Hopper SM90, or Blackwell SM100/SM120".into(),
                ));
            }
            tracing::warn!("Experimental dynamic FP8 MLP enabled: per-row weights quantized at load, per-token activations at inference; accuracy may change");
        }

        let vb = if model_files.len() == 1 && model_files[0].extension().unwrap() == "bin" {
            VarBuilder::from_pth(&model_files[0], dtype, &device)
        } else if model_files.len() == 1 && model_files[0].extension().unwrap() == "safetensors" {
            tracing::info!(
                "Loading safetensors model with B10 method from file: {:?}",
                &model_files[0]
            );
            let file_u8 = std::fs::read(&model_files[0])
                .map_err(|err| BackendError::Start(err.to_string()))?;
            tracing::info!("Loaded safetensors, size: {} bytes", file_u8.len());
            VarBuilder::from_buffered_safetensors(file_u8, dtype, &device)
        } else {
            unsafe { VarBuilder::from_mmaped_safetensors(&model_files, dtype, &device) }
        }
        .s()?;

        // Decoder classifiers reuse the existing backbone and batching kernels.
        let sequence_classifier = model_type == ModelType::Classifier
            && matches!(
                &config,
                Config::Qwen2(_) | Config::Qwen3(_) | Config::Llama(_)
            )
            && models::SequenceClassifier::supports(&config_json).s()?;
        let classifier_vb = sequence_classifier.then(|| vb.clone());
        let model_type = if sequence_classifier {
            if let Config::Qwen3(config) = &config {
                if config.use_linear_output_projection
                    || (config.use_bidirectional_attention && vb.contains_tensor("linear.weight"))
                {
                    return Err(BackendError::Start("Sequence classifiers require score.weight, not an embedding output projection".into()));
                }
            }
            ModelType::Embedding(Pool::LastToken)
        } else {
            model_type
        };

        let model: Result<Box<dyn Model + Send>, BackendError> = match config {
            Config::UltraBert(config) => Ok(Box::new(
                models::UltraBertModel::load(vb, &config, model_type).s()?,
            )),
            #[cfg(feature = "experimental-deberta")]
            Config::Deberta(config) => Ok(Box::new(
                models::DebertaModel::load(vb, &config, model_type).s()?,
            )),
            Config::Bert(config) => match config {
                BertConfigWrapper::Bert(config) => Ok(Box::new(
                    FlashBertModel::load(vb, &config, model_type, enable_fp8_dynamic).s()?,
                )),
                BertConfigWrapper::JinaBert(config) => Ok(Box::new(
                    FlashJinaBertModel::load(vb, &config, model_type).s()?,
                )),
                BertConfigWrapper::JinaCodeBert(config) => Ok(Box::new(
                    FlashJinaCodeBertModel::load(vb, &config, model_type).s()?,
                )),
            },
            Config::Camembert(config) | Config::Roberta(config) | Config::XlmRoberta(config) => {
                Ok(Box::new(
                    FlashBertModel::load_roberta(vb, &config, model_type, enable_fp8_dynamic)
                        .s()?,
                ))
            }
            Config::DistilBert(config) => Ok(Box::new(
                FlashDistilBertModel::load(vb, &config, model_type).s()?,
            )),
            Config::Gte(config) => Ok(Box::new(FlashGTEModel::load(vb, &config, model_type).s()?)),
            Config::ModernBert(config) => Ok(Box::new(
                FlashModernBertModel::load(vb, &config, model_type, enable_fp8_dynamic).s()?,
            )),
            Config::NomicBert(config) => Ok(Box::new(
                FlashNomicBertModel::load(vb, &config, model_type).s()?,
            )),
            Config::Mistral(config) => Ok(Box::new(
                FlashMistralModel::load(vb, &config, model_type, enable_fp8_dynamic).s()?,
            )),
            Config::Qwen2(config) => Ok(Box::new(
                FlashQwen2Model::load(vb, &config, model_type, enable_fp8_dynamic).s()?,
            )),
            Config::Qwen3(config) => Ok(Box::new(
                FlashQwen3Model::load(vb, &config, model_type, enable_fp8_dynamic).s()?,
            )),
            Config::Llama(config) => {
                if config.attention_bias.unwrap_or(false)
                    || config.mlp_bias
                    || config.num_attention_heads == 0
                    || !config
                        .hidden_size
                        .is_multiple_of(config.num_attention_heads)
                    || config
                        .head_dim
                        .is_some_and(|dim| dim != config.hidden_size / config.num_attention_heads)
                {
                    return Err(BackendError::Start("Packed Llama requires bias-free projections and head_dim = hidden_size / num_attention_heads".into()));
                }
                let cfg_mistral = MistralConfig {
                    head_dim: config.head_dim,
                    is_causal: None,
                    rope_parameters: None,
                    vocab_size: config.vocab_size,
                    hidden_size: config.hidden_size,
                    intermediate_size: config.intermediate_size,
                    num_hidden_layers: config.num_hidden_layers,
                    num_attention_heads: config.num_attention_heads,
                    num_key_value_heads: config.num_key_value_heads,
                    hidden_act: config.hidden_act,
                    max_position_embeddings: config.max_position_embeddings,
                    initializer_range: config.initializer_range,
                    rms_norm_eps: config.rms_norm_eps,
                    model_type: config.model_type.clone(),
                    rope_theta: config.rope_theta,
                    sliding_window: config.sliding_window,
                    rope_scaling: config.rope_scaling,
                    use_bidirectional_attention: config.use_bidirectional_attention,
                };
                Ok(Box::new(
                    FlashMistralModel::load(vb, &cfg_mistral, model_type, enable_fp8_dynamic)
                        .s()?,
                ))
            }
            Config::MPNet(_) => Err(BackendError::Start(
                "MPNet has no packed implementation".into(),
            )),
            Config::Gemma3(config) => {
                #[cfg(feature = "flash-attn")]
                {
                    Ok(Box::new(
                        models::Gemma3Model::load(vb, &config, model_type).s()?,
                    ))
                }
                #[cfg(not(feature = "flash-attn"))]
                {
                    let _ = config;
                    Err(BackendError::Start(
                        "Gemma3 requires CUDA BF16 with FlashAttention v2".into(),
                    ))
                }
            }
            Config::EmbeddingGemma2(config) => {
                #[cfg(feature = "flash-attn")]
                {
                    Ok(Box::new(
                        models::EmbeddingGemma2Model::load(vb, &config, model_type).s()?,
                    ))
                }
                #[cfg(not(feature = "flash-attn"))]
                {
                    let _ = config;
                    Err(BackendError::Start(
                        "EmbeddingGemma2 requires CUDA BF16 with FlashAttention v2".into(),
                    ))
                }
            }
            Config::Gemma4(config) => {
                #[cfg(feature = "flash-attn")]
                {
                    Ok(Box::new(
                        models::Gemma4Model::load(vb, &config, model_type).s()?,
                    ))
                }
                #[cfg(not(feature = "flash-attn"))]
                {
                    let _ = config;
                    Err(BackendError::Start(
                        "Gemma4 requires CUDA BF16 with FlashAttention v2".into(),
                    ))
                }
            }
            Config::Qwen3Vl(config) => {
                #[cfg(all(feature = "cuda", feature = "flash-attn"))]
                {
                    Ok(Box::new(
                        models::Qwen3VlModel::load(vb, config, model_type).s()?,
                    ))
                }
                #[cfg(not(all(feature = "cuda", feature = "flash-attn")))]
                {
                    let _ = config;
                    Err(BackendError::Start(
                        "Qwen3-VL requires CUDA with FlashAttention v2".into(),
                    ))
                }
            }
            Config::Qwen35(config) => {
                #[cfg(all(feature = "cuda", feature = "flash-attn"))]
                {
                    Ok(Box::new({
                        let decision = model_type == ModelType::Decision;
                        let external_readout =
                            decision && model_path.join("decision_config.json").exists();
                        let model = models::Qwen35Model::load(
                            vb.clone(),
                            &config,
                            model_type,
                            external_readout,
                        )
                        .s()?;
                        if external_readout {
                            model.with_readout(model_path, &config).s()?
                        } else if decision && model_path.join("joint_head_config.json").exists() {
                            model.with_clef(model_path, &config).s()?
                        } else {
                            model.with_vision(vb, &config).s()?
                        }
                    }))
                }
                #[cfg(not(all(feature = "cuda", feature = "flash-attn")))]
                {
                    let _ = config;
                    Err(BackendError::Start(
                        "Qwen3.5-MoE requires CUDA with FlashAttention v2".into(),
                    ))
                }
            }
        };

        let model = model?;
        let model: Box<dyn Model + Send> = match classifier_vb {
            Some(vb) => Box::new(models::SequenceClassifier::load(model, vb, &config_json).s()?),
            None => model,
        };

        let mut dense_layers = Vec::new();
        if let Some(dense_paths) = dense_paths {
            if !dense_paths.is_empty() {
                tracing::info!("Loading Dense module/s from path/s: {dense_paths:?}");

                for dense_path in dense_paths.iter() {
                    let dense_safetensors =
                        model_path.join(format!("{dense_path}/model.safetensors"));
                    let dense_pytorch = model_path.join(format!("{dense_path}/pytorch_model.bin"));

                    if dense_safetensors.exists() || dense_pytorch.exists() {
                        let dense_config_path =
                            model_path.join(format!("{dense_path}/config.json"));

                        let dense_config_str = std::fs::read_to_string(&dense_config_path)
                            .map_err(|err| {
                                BackendError::Start(format!(
                                    "Unable to read `{dense_path}/config.json` file: {err:?}",
                                ))
                            })?;
                        let dense_config: DenseConfig = serde_json::from_str(&dense_config_str)
                            .map_err(|err| {
                                BackendError::Start(format!(
                                    "Unable to parse `{dense_path}/config.json`: {err:?}",
                                ))
                            })?;

                        let dense_vb = if dense_safetensors.exists() {
                            unsafe {
                                VarBuilder::from_mmaped_safetensors(
                                    &[dense_safetensors],
                                    dtype,
                                    &device,
                                )
                            }
                            .s()?
                        } else {
                            VarBuilder::from_pth(&dense_pytorch, dtype, &device).s()?
                        };

                        let dense_layer = Box::new(Dense::load(dense_vb, &dense_config).s()?)
                            as Box<dyn DenseLayer + Send>;
                        dense_layers.push(dense_layer);

                        tracing::info!("Loaded Dense module from path: {dense_path}");
                    } else {
                        tracing::warn!("Dense module files not found for path: {dense_path}",);
                    }
                }
            }
        }

        Ok(Self {
            device,
            model,
            dense_layers,
            token_dense,
        })
    }
}

impl Backend for CandleBackend {
    fn decide(
        &self,
        batch: Batch,
        inputs: Vec<infinity_v3_backend_core::DecisionInput>,
    ) -> Result<Vec<infinity_v3_backend_core::DecisionOutput>, BackendError> {
        self.model.decide(batch, inputs).e()
    }

    fn max_batch_size(&self) -> Option<usize> {
        // Limit max batch size to 4 on CPU
        if matches!(self.device, Device::Cpu) {
            return Some(4);
        }
        None
    }

    fn health(&self) -> Result<(), BackendError> {
        // Simple healthcheck by performing a trivial operation
        // backend is almost unfailable, but e.g. Cuda OOM or cuda "device fallen off the bus"
        // can be detected this way
        use candle::Tensor;

        // 1) enqueue a trivial op on the current device
        let x = Tensor::new(&[1f32], &self.device).e()?;
        let z = Tensor::new(&[2f32], &self.device).e()?;
        let y = (&x * &z).e()?;

        // 2) move storage to CPU to surface async CUDA errors by reading back
        let v = y.to_vec1::<f32>().e()?;
        if v.len() != 1 || (v[0] - 2.0).abs() > 1e-6 {
            // michaelfeil: ideally, we should sleep here for 1.0s/5.0s to allow k8s to detect healthcheck failure
            // without queuing further work on a possibly broken device by blocking the backend.
            // and sending 429s in the meantime.
            tracing::error!(
                "Device {:?} healthcheck failed: expected [2.0], got {v:?}",
                self.device
            );
            return Err(BackendError::Inference(format!(
                "device healthcheck failed: expected [2.0], got {v:?}"
            )));
        }
        tracing::debug!("Device {:?} healthcheck passed", self.device);

        Ok(())
    }

    fn supports_radix_mlp(&self) -> bool {
        self.model.supports_radix_mlp()
    }

    fn embed(&self, mut batch: Batch) -> Result<Embeddings, BackendError> {
        let batch_size = batch.len();
        let pooled_indices = batch.pooled_indices.clone();
        let raw_indices = batch.raw_indices.clone();
        if pooled_indices
            .iter()
            .chain(&raw_indices)
            .any(|&index| index as usize >= batch_size)
        {
            return Err(BackendError::Inference("Invalid output index".into()));
        }
        // Packed model pooling historically assumes identity selectors when no
        // raw rows are requested. Ask for canonical rows, then select before
        // Dense modules so request order and repeated indices work for every model.
        let select_pooled =
            !pooled_indices.is_empty() && pooled_indices.iter().copied().ne(0..batch_size as u32);
        if select_pooled {
            batch.pooled_indices = (0..batch_size as u32).collect();
        }
        // Raw-only packed models can also return canonical token rows regardless
        // of the selectors. Request those rows explicitly and gather real spans
        // on the device before transferring only the requested tokens to the host.
        let select_raw =
            !raw_indices.is_empty() && raw_indices.iter().copied().ne(0..batch_size as u32);
        let raw_offsets = batch.cumulative_seq_lengths.clone();
        if select_raw {
            batch.raw_indices = (0..batch_size as u32).collect();
        }

        // Used for indexing in the raw_embeddings tensor
        let input_lengths: Vec<usize> = (0..batch.len())
            .map(|i| {
                (batch.cumulative_seq_lengths[i + 1] - batch.cumulative_seq_lengths[i]) as usize
            })
            .collect();

        // Run forward
        let (pooled_embeddings, raw_embeddings) = self.model.embed(batch).e()?;
        let pooled_embeddings = if select_pooled {
            pooled_embeddings
                .map(|pooled| {
                    let indices = candle::Tensor::new(pooled_indices.as_slice(), pooled.device())?;
                    crate::layers::index_select(&pooled, &indices, 0)
                })
                .transpose()
                .e()?
        } else {
            pooled_embeddings
        };

        let raw_embeddings = if select_raw {
            raw_embeddings
                .map(|raw| {
                    let spans = raw_indices
                        .iter()
                        .map(|&index| {
                            let index = index as usize;
                            let start = raw_offsets[index] as usize;
                            let length = (raw_offsets[index + 1] - raw_offsets[index]) as usize;
                            raw.narrow(0, start, length)
                        })
                        .collect::<candle::Result<Vec<_>>>()?;
                    candle::Tensor::cat(&spans, 0)
                })
                .transpose()
                .e()?
        } else {
            raw_embeddings
        };

        // Apply Dense layers sequentially if available
        let pooled_embeddings = match pooled_embeddings {
            None => None,
            Some(mut pooled_embeddings) => {
                for dense in &self.dense_layers {
                    pooled_embeddings = dense.forward(&pooled_embeddings).e()?;
                }
                Some(pooled_embeddings)
            }
        };

        let raw_embeddings = if self.token_dense {
            raw_embeddings
                .map(|mut raw| {
                    for dense in &self.dense_layers {
                        raw = dense.forward(&raw)?;
                    }
                    Ok::<_, candle::Error>(raw)
                })
                .transpose()
                .e()?
        } else {
            raw_embeddings
        };

        // Device => Host data transfer
        let pooled_embeddings = match pooled_embeddings {
            None => vec![],
            Some(pooled_embeddings) => pooled_embeddings.to_dtype(DType::F32).e()?.to_vec2().e()?,
        };

        // This transfer is expensive...
        let raw_embeddings = match raw_embeddings {
            None => vec![],
            Some(raw_embeddings) => raw_embeddings.to_dtype(DType::F32).e()?.to_vec2().e()?,
        };

        let mut embeddings =
            HashMap::with_capacity_and_hasher(batch_size, BuildNoHashHasher::default());
        for (i, e) in pooled_indices.into_iter().zip(pooled_embeddings) {
            embeddings.insert(i as usize, Embedding::Pooled(e));
        }

        let mut cumulative_length = 0;
        for i in raw_indices.into_iter() {
            let length = input_lengths[i as usize];
            let e = raw_embeddings[cumulative_length..cumulative_length + length].to_vec();
            embeddings.insert(i as usize, Embedding::All(e));
            cumulative_length += length;
        }

        Ok(embeddings.into())
    }

    fn predict(&self, batch: Batch) -> Result<Predictions, BackendError> {
        let batch_size = batch.len();

        let results = self.model.predict(batch).e()?;

        let results = results.to_dtype(DType::F32).e()?.to_vec2().e()?;

        let mut predictions =
            HashMap::with_capacity_and_hasher(batch_size, BuildNoHashHasher::default());
        for (i, r) in results.into_iter().enumerate() {
            predictions.insert(i, r);
        }

        Ok(predictions.into())
    }

    fn predict_tokens(&self, batch: Batch) -> Result<TokenPredictions, BackendError> {
        let batch_size = batch.len();
        let cumulative_seq_lengths = batch.cumulative_seq_lengths.clone();

        let results = self.model.predict_tokens(batch).e()?;

        let results = results.to_dtype(DType::F32).e()?.to_vec2().e()?;

        let mut predictions =
            HashMap::with_capacity_and_hasher(batch_size, BuildNoHashHasher::default());

        // Split the 2D results back into batches using cumulative_seq_lengths
        for i in 0..batch_size {
            let start = cumulative_seq_lengths[i] as usize;
            let end = cumulative_seq_lengths[i + 1] as usize;

            let token_predictions: Vec<Vec<f32>> = results[start..end].to_vec();
            predictions.insert(i, token_predictions);
        }

        Ok(predictions.into())
    }
}

pub trait WrapErr<O> {
    fn s(self) -> Result<O, BackendError>;
    fn e(self) -> Result<O, BackendError>;
}

impl<O> WrapErr<O> for Result<O, candle::Error> {
    fn s(self) -> Result<O, BackendError> {
        self.map_err(|e| BackendError::Start(e.to_string()))
    }
    fn e(self) -> Result<O, BackendError> {
        self.map_err(|e| BackendError::Inference(e.to_string()))
    }
}

#[cfg(feature = "cuda")]
pub fn visible_cuda_device_count() -> Result<usize, BackendError> {
    candle::cuda_backend::cudarc::driver::CudaContext::device_count()
        .map(|count| count as usize)
        .map_err(|err| BackendError::Start(format!("Cannot enumerate CUDA devices: {err}")))
}
