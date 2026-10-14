use hf_hub::api::tokio::{ApiError, ApiRepo};
use std::path::PathBuf;
use tracing::instrument;

// `sentence_bert_config.json` default Sentence Transformers configuration file name, and other
// former / deprecated file names no longer used, but here for backwards compatibility
pub const ST_CONFIG_NAMES: [&str; 7] = [
    "sentence_bert_config.json",
    "sentence_roberta_config.json",
    "sentence_distilbert_config.json",
    "sentence_camembert_config.json",
    "sentence_albert_config.json",
    "sentence_xlm-roberta_config.json",
    "sentence_xlnet_config.json",
];

async fn download_file(api: &ApiRepo, file_path: &str) -> Result<PathBuf, ApiError> {
    tracing::info!("Downloading `{}`", file_path);
    api.get(file_path).await
}

async fn download_st_config_legacy(api: &ApiRepo) -> Result<PathBuf, ApiError> {
    // Try to download first the default path i.e., `sentence_bert_config.json`
    let err = match download_file(api, ST_CONFIG_NAMES[0]).await {
        Ok(st_config_path) => return Ok(st_config_path),
        Err(err) => err,
    };

    // Then try with the rest of the legacy configuration file names
    for name in &ST_CONFIG_NAMES[1..] {
        if let Ok(st_config_path) = download_file(api, name).await {
            return Ok(st_config_path);
        }
    }

    Err(err)
}

#[instrument(skip_all)]
pub async fn download_artifacts(api: &ApiRepo, pool_config: bool) -> Result<PathBuf, ApiError> {
    let start = std::time::Instant::now();
    tracing::info!("Starting download");

    // Laya checkpoints keep their encoder config and tokenizer in subfolders.
    if let Ok(path) = download_file(api, "rl_agent_config.json").await {
        download_file(api, "encoder/config.json").await?;
        download_file(api, "tokenizer/tokenizer.json").await?;
        return Ok(path.parent().unwrap().to_path_buf());
    }

    // Try to download `1_Pooling`, only if `--pooling` hasn't been provided, otherwise, the
    // `--pooling` argument will be used instead.
    if pool_config {
        let _ = download_file(api, "1_Pooling/config.json")
            .await
            .map_err(|err| {
                tracing::warn!("Download failed: {err}");
                err
            });
    }

    // Download the legacy Sentence Transformers configuration files as defined in
    // `ST_CONFIG_NAMES` (no warn on failure as it's a legacy file)
    // NOTE: used to define the `max_seq_length`, otherwise it will be defined as
    // `max_position_embeddings - position_offset` from the `config.json` file
    let _ = download_st_config_legacy(api).await;

    // Download the actual Sentence Transformers configuration file
    let _ = download_file(api, "config_sentence_transformers.json")
        .await
        .map_err(|err| {
            tracing::warn!("Download failed: {err}");
            err
        });

    let config_path = download_file(api, "config.json").await?;
    // Required for supported vision checkpoints; do not silently serve images without it.
    if let Ok(bytes) = std::fs::read(&config_path) {
        if serde_json::from_slice::<serde_json::Value>(&bytes)
            .is_ok_and(|config| config["model_type"] == "qwen3_vl")
        {
            download_file(api, "preprocessor_config.json").await?;
        } else if serde_json::from_slice::<serde_json::Value>(&bytes).is_ok_and(|config| {
            matches!(
                config["model_type"].as_str(),
                Some("gemma4" | "embedding_gemma2")
            ) && (config["vision_config"].is_object() || config["audio_config"].is_object())
        }) {
            download_file(api, "processor_config.json").await?;
        } else if serde_json::from_slice::<serde_json::Value>(&bytes).is_ok_and(|config| {
            matches!(
                config["model_type"].as_str(),
                Some("qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text")
            ) && config["vision_config"].is_object()
        }) {
            // Text-only Qwen3.5 exports may omit their optional media processor.
            let _ = download_file(api, "processor_config.json").await;
        }
    }
    if download_file(api, "joint_head_config.json").await.is_ok() {
        download_file(api, "joint_head.safetensors").await?;
    }
    if let Ok(path) = download_file(api, "decision_config.json").await {
        download_file(api, "readout.safetensors").await?;
        if std::fs::read(path).is_ok_and(|bytes| {
            serde_json::from_slice::<serde_json::Value>(&bytes)
                .is_ok_and(|config| config["format_id"] == "d3-code-readout-v1")
        }) {
            // D3 saves the Qwen image processor here, rather than processor_config.json.
            download_file(api, "preprocessor_config.json").await?;
        }
    }
    let path = download_file(api, "tokenizer.json").await?;
    // Optional native conversation artifacts, using the same pinned repository revision.
    for name in ["tokenizer_config.json", "chat_template.jinja"] {
        if let Err(err) = download_file(api, name).await {
            tracing::debug!("Optional chat artifact {name} unavailable: {err}");
        }
    }

    tracing::info!("Model artifacts downloaded in {:?}", start.elapsed());

    Ok(path.parent().unwrap().to_path_buf())
}
