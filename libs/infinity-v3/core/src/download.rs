use futures::stream::{FuturesOrdered, StreamExt};
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
    // FuturesOrdered polls every candidate concurrently but yields results in
    // declared priority order. Returning the first success drops the remaining
    // probes, cancelling lower-priority downloads.
    let mut probes: FuturesOrdered<_> = ST_CONFIG_NAMES
        .iter()
        .map(|name| download_file(api, name))
        .collect();
    let Some(first) = probes.next().await else {
        return Err(ApiError::IoError(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "ST_CONFIG_NAMES must not be empty",
        )));
    };
    let first_error = match first {
        Ok(path) => return Ok(path),
        Err(error) => error,
    };
    while let Some(result) = probes.next().await {
        if let Ok(path) = result {
            return Ok(path);
        }
    }
    Err(first_error)
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

    // The required config starts alongside the optional probes. Conditional
    // companion downloads remain sequential after `config.json` is inspected.
    let pooling = async {
        if pool_config {
            Some(download_file(api, "1_Pooling/config.json").await)
        } else {
            None
        }
    };
    let (config_path, pooling, legacy_st_config, new_st_config) = tokio::join!(
        download_file(api, "config.json"),
        pooling,
        download_st_config_legacy(api),
        download_file(api, "config_sentence_transformers.json"),
    );
    if let Some(Err(err)) = pooling {
        tracing::warn!("Download failed: {err}");
    }
    // Used to define `max_seq_length`, otherwise it will be defined as
    // `max_position_embeddings - position_offset` from `config.json`.
    let _ = legacy_st_config;
    if let Err(err) = new_st_config {
        tracing::warn!("Download failed: {err}");
    }

    let config_path = config_path?;
    // Required for supported vision checkpoints; do not silently serve images without it.
    if let Ok(bytes) = std::fs::read(&config_path) {
        if serde_json::from_slice::<serde_json::Value>(&bytes).is_ok_and(|config| {
            matches!(config["model_type"].as_str(), Some("qwen3_vl" | "colqwen3"))
        }) {
            download_file(api, "preprocessor_config.json").await?;
            if serde_json::from_slice::<serde_json::Value>(&bytes)
                .is_ok_and(|config| config["model_type"] == "colqwen3")
            {
                let _ = download_file(api, "1_MultiVectorMask/config.json").await;
            }
        } else if serde_json::from_slice::<serde_json::Value>(&bytes).is_ok_and(|config| {
            matches!(
                config["model_type"].as_str(),
                Some("gemma4" | "embedding_gemma2" | "mistral3" | "qwen3_vl_nemotron_embed")
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
    let (joint_head_config, decision_config) = tokio::join!(
        download_file(api, "joint_head_config.json"),
        download_file(api, "decision_config.json"),
    );
    if joint_head_config.is_ok() {
        download_file(api, "joint_head.safetensors").await?;
    }
    if let Ok(path) = decision_config {
        download_file(api, "readout.safetensors").await?;
        if std::fs::read(&path).is_ok_and(|bytes| {
            serde_json::from_slice::<serde_json::Value>(&bytes)
                .is_ok_and(|config| config["format_version"] == 2)
        }) {
            download_file(api, "joint_config.json").await?;
        }
        if std::fs::read(path).is_ok_and(|bytes| {
            serde_json::from_slice::<serde_json::Value>(&bytes)
                .is_ok_and(|config| config["format_id"] == "d3-code-readout-v1")
        }) {
            // D3 saves the Qwen image processor here, rather than processor_config.json.
            download_file(api, "preprocessor_config.json").await?;
        }
    }
    let path = download_file(api, "tokenizer.json").await?;
    // Optional native conversation artifacts are independent and use the same
    // pinned repository revision.
    let (tokenizer_config, chat_template) = tokio::join!(
        download_file(api, "tokenizer_config.json"),
        download_file(api, "chat_template.jinja"),
    );
    for (name, result) in [
        ("tokenizer_config.json", tokenizer_config),
        ("chat_template.jinja", chat_template),
    ] {
        if let Err(err) = result {
            tracing::debug!("Optional chat artifact {name} unavailable: {err}");
        }
    }

    tracing::info!("Model artifacts downloaded in {:?}", start.elapsed());

    Ok(path.parent().unwrap().to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hf_hub::api::tokio::ApiBuilder;
    use hf_hub::{Repo, RepoType};
    use std::collections::{HashMap, HashSet};
    use std::io::{BufRead, BufReader, Write};
    use std::net::{SocketAddr, TcpListener, TcpStream};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Condvar, Mutex};
    use std::thread;
    use std::time::Duration;

    const REQUEST_PREFIX: &str = "/test/model/resolve/main/";

    struct ProbeGate {
        target: usize,
        blocked: HashSet<String>,
        state: Mutex<(usize, bool)>,
        condvar: Condvar,
    }

    impl ProbeGate {
        fn new(target: usize, blocked: impl IntoIterator<Item = &'static str>) -> Self {
            Self {
                target,
                blocked: blocked.into_iter().map(str::to_owned).collect(),
                state: Mutex::new((0, false)),
                condvar: Condvar::new(),
            }
        }

        fn arrive(&self, filename: &str) {
            if !self.blocked.contains(filename) {
                return;
            }
            let mut state = self.state.lock().unwrap();
            state.0 += 1;
            self.condvar.notify_all();
            while state.0 < self.target && !state.1 {
                state = self.condvar.wait(state).unwrap();
            }
        }

        fn wait_until_arrived(&self, target: usize) {
            let mut state = self.state.lock().unwrap();
            while state.0 < target && !state.1 {
                state = self.condvar.wait(state).unwrap();
            }
        }

        fn release(&self) {
            let mut state = self.state.lock().unwrap();
            state.1 = true;
            self.condvar.notify_all();
        }
    }

    fn release_when_all_arrive(gate: Arc<ProbeGate>, target: usize) -> thread::JoinHandle<()> {
        thread::spawn(move || {
            gate.wait_until_arrived(target);
            gate.release();
        })
    }

    struct MockHub {
        addr: SocketAddr,
        cache: tempfile::TempDir,
        stop: Arc<AtomicBool>,
        server: Option<thread::JoinHandle<()>>,
        gate: Option<Arc<ProbeGate>>,
        requests: Arc<Mutex<Vec<String>>>,
    }

    impl MockHub {
        fn new(files: Vec<(String, Vec<u8>)>, gate: Option<Arc<ProbeGate>>) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            listener.set_nonblocking(true).unwrap();
            let addr = listener.local_addr().unwrap();
            let state = Arc::new(Mutex::new(files.into_iter().collect::<HashMap<_, _>>()));
            let stop = Arc::new(AtomicBool::new(false));
            let requests = Arc::new(Mutex::new(Vec::new()));
            let server_stop = stop.clone();
            let server_gate = gate.clone();
            let server_requests = requests.clone();
            let server = thread::spawn(move || {
                while !server_stop.load(Ordering::Relaxed) {
                    match listener.accept() {
                        Ok((stream, _)) => {
                            let files = state.clone();
                            let gate = server_gate.clone();
                            let requests = server_requests.clone();
                            thread::spawn(move || serve(stream, files, gate, requests));
                        }
                        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                            thread::sleep(Duration::from_millis(1));
                        }
                        Err(_) => break,
                    }
                }
            });
            Self {
                addr,
                cache: tempfile::tempdir().unwrap(),
                stop,
                server: Some(server),
                gate,
                requests,
            }
        }

        fn repo(&self) -> ApiRepo {
            let api = ApiBuilder::new()
                .with_endpoint(format!("http://{}", self.addr))
                .with_cache_dir(self.cache.path().to_path_buf())
                .with_progress(false)
                .with_token(None)
                .build()
                .unwrap();
            api.repo(Repo::with_revision(
                "test/model".into(),
                RepoType::Model,
                "main".into(),
            ))
        }

        fn requests(&self) -> Vec<String> {
            self.requests.lock().unwrap().clone()
        }
    }

    impl Drop for MockHub {
        fn drop(&mut self) {
            if let Some(gate) = &self.gate {
                gate.release();
            }
            self.stop.store(true, Ordering::Relaxed);
            if let Some(server) = self.server.take() {
                let _ = server.join();
            }
        }
    }

    fn serve(
        stream: TcpStream,
        files: Arc<Mutex<HashMap<String, Vec<u8>>>>,
        gate: Option<Arc<ProbeGate>>,
        requests: Arc<Mutex<Vec<String>>>,
    ) {
        let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
        let mut reader = match stream.try_clone() {
            Ok(stream) => BufReader::new(stream),
            Err(_) => return,
        };
        let mut request_line = String::new();
        if reader.read_line(&mut request_line).is_err() {
            return;
        }
        let path = request_line.split_whitespace().nth(1).unwrap_or_default();
        let filename = path.strip_prefix(REQUEST_PREFIX).unwrap_or_default();
        requests.lock().unwrap().push(filename.to_owned());
        let range = loop {
            let mut line = String::new();
            if reader.read_line(&mut line).unwrap_or(0) == 0 {
                break None;
            }
            let lower = line.to_ascii_lowercase();
            if lower == "\r\n" {
                break None;
            }
            if let Some(value) = lower.strip_prefix("range:") {
                break Some(value.trim().to_owned());
            }
        };
        if range.as_deref() == Some("bytes=0-0") {
            if let Some(gate) = &gate {
                gate.arrive(filename);
            }
        }

        let mut stream = stream;
        let Some(content) = files.lock().unwrap().get(filename).cloned() else {
            let _ = stream.write_all(
                b"HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\nconnection: close\r\n\r\n",
            );
            return;
        };
        let etag = filename.replace('/', "_");
        if range.as_deref() == Some("bytes=0-0") {
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-length: {}\r\ncontent-range: bytes 0-0/{}\r\nx-repo-commit: test-commit\r\netag: \"etag-{etag}\"\r\nconnection: close\r\n\r\n",
                content.len(),
                content.len(),
            );
            let _ = stream.write_all(response.as_bytes());
            let _ = stream.write_all(&content);
            return;
        }

        let body = match range.as_deref().and_then(parse_range) {
            Some((start, stop)) if start < content.len() => {
                &content[start..=stop.min(content.len() - 1)]
            }
            _ => &content[..],
        };
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
            body.len()
        );
        let _ = stream.write_all(response.as_bytes());
        let _ = stream.write_all(body);
    }

    fn parse_range(value: &str) -> Option<(usize, usize)> {
        let value = value.strip_prefix("bytes=")?;
        let (start, stop) = value.split_once('-')?;
        Some((start.parse().ok()?, stop.parse().ok()?))
    }

    fn legacy_files() -> Vec<(String, Vec<u8>)> {
        ST_CONFIG_NAMES
            .iter()
            .map(|name| ((*name).to_owned(), b"{}".to_vec()))
            .collect()
    }

    #[tokio::test]
    async fn all_legacy_config_probes_are_concurrent_and_preserve_priority() {
        let gate = Arc::new(ProbeGate::new(ST_CONFIG_NAMES.len(), ST_CONFIG_NAMES));
        let hub = MockHub::new(legacy_files(), Some(gate.clone()));
        let repo = hub.repo();
        let release = release_when_all_arrive(gate.clone(), ST_CONFIG_NAMES.len());

        let result =
            tokio::time::timeout(Duration::from_secs(5), download_st_config_legacy(&repo)).await;
        gate.release();
        let _ = release.join();

        let selected = result
            .expect("legacy probes did not all reach the mock hub concurrently")
            .expect("a present legacy config should be selected");
        assert!(selected.ends_with(ST_CONFIG_NAMES[0]));
    }

    #[tokio::test]
    async fn all_missing_legacy_configs_fail_after_concurrent_probes() {
        let gate = Arc::new(ProbeGate::new(ST_CONFIG_NAMES.len(), ST_CONFIG_NAMES));
        let hub = MockHub::new(Vec::new(), Some(gate.clone()));
        let repo = hub.repo();
        let release = release_when_all_arrive(gate.clone(), ST_CONFIG_NAMES.len());

        let result =
            tokio::time::timeout(Duration::from_secs(5), download_st_config_legacy(&repo)).await;
        gate.release();
        let _ = release.join();

        assert!(result
            .expect("legacy probes did not all reach the mock hub concurrently")
            .is_err());
    }

    #[tokio::test]
    async fn highest_priority_available_returns_while_lower_probe_is_blocked() {
        let lower = &ST_CONFIG_NAMES[1..];
        let gate = Arc::new(ProbeGate::new(lower.len(), lower.iter().copied()));
        let hub = MockHub::new(
            vec![(ST_CONFIG_NAMES[0].to_owned(), b"{}".to_vec())],
            Some(gate.clone()),
        );
        let repo = hub.repo();

        let result =
            tokio::time::timeout(Duration::from_secs(2), download_st_config_legacy(&repo)).await;
        gate.release();

        let selected = result
            .expect("lower-priority probes blocked the successful first probe")
            .expect("the first legacy config should be selected");
        assert!(selected.ends_with(ST_CONFIG_NAMES[0]));
    }

    #[tokio::test]
    async fn required_config_and_tokenizer_are_downloaded_with_optional_misses() {
        let hub = MockHub::new(
            vec![
                ("config.json".to_owned(), b"{}".to_vec()),
                (
                    "tokenizer.json".to_owned(),
                    br#"{"version":"1.0"}"#.to_vec(),
                ),
            ],
            None,
        );
        let repo = hub.repo();

        let root = download_artifacts(&repo, false).await.unwrap();
        assert!(root.join("config.json").is_file());
        assert!(root.join("tokenizer.json").is_file());
    }

    #[tokio::test]
    async fn required_sidecar_failure_propagates_after_parallel_optional_probes() {
        let hub = MockHub::new(
            vec![
                ("config.json".to_owned(), b"{}".to_vec()),
                ("joint_head_config.json".to_owned(), b"{}".to_vec()),
            ],
            None,
        );
        let repo = hub.repo();

        assert!(download_artifacts(&repo, false).await.is_err());
        let requests = hub.requests();
        assert!(requests.iter().any(|file| file == "config.json"));
        assert!(requests.iter().any(|file| file == "joint_head.safetensors"));
    }
}
